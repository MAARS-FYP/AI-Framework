use std::ffi::CString;
use std::io;
use std::os::fd::RawFd;
use std::ptr;

pub struct SharedMemoryRingSpec {
    pub name: String,
    pub num_slots: usize,
    pub slot_capacity: usize,
}

impl SharedMemoryRingSpec {
    pub fn slot_bytes(&self) -> usize {
        self.slot_capacity * 2 * std::mem::size_of::<f32>()
    }

    pub fn total_bytes(&self) -> usize {
        self.num_slots * self.slot_bytes()
    }
}

pub struct SharedMemoryRingBuffer {
    spec: SharedMemoryRingSpec,
    fd: RawFd,
    ptr: *mut u8,
    size: usize,
}

impl SharedMemoryRingBuffer {
    pub fn attach(spec: SharedMemoryRingSpec) -> io::Result<Self> {
        let fd = open_shm_fd(&spec.name)?;
        let size = spec.total_bytes();
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            let err = io::Error::last_os_error();
            unsafe {
                libc::close(fd);
            }
            return Err(err);
        }

        Ok(Self {
            spec,
            fd,
            ptr: ptr as *mut u8,
            size,
        })
    }

    pub fn write_slot(
        &mut self,
        slot_index: usize,
        iq_iq_pairs: &[(f32, f32)],
    ) -> io::Result<usize> {
        if slot_index >= self.spec.num_slots {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("slot_index out of range: {}", slot_index),
            ));
        }
        if iq_iq_pairs.len() > self.spec.slot_capacity {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "IQ sample count {} exceeds slot capacity {}",
                    iq_iq_pairs.len(),
                    self.spec.slot_capacity
                ),
            ));
        }

        let slot_float_offset = slot_index * self.spec.slot_capacity * 2;
        let total_floats = self.size / std::mem::size_of::<f32>();
        let floats = unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut f32, total_floats) };
        let slot = &mut floats[slot_float_offset..slot_float_offset + self.spec.slot_capacity * 2];

        let n_samples = iq_iq_pairs.len();
        for (idx, (i, q)) in iq_iq_pairs.iter().enumerate() {
            let base = idx * 2;
            slot[base] = *i;
            slot[base + 1] = *q;
        }

        for idx in n_samples..self.spec.slot_capacity {
            let base = idx * 2;
            slot[base] = 0.0;
            slot[base + 1] = 0.0;
        }

        Ok(n_samples)
    }

    pub fn num_slots(&self) -> usize {
        self.spec.num_slots
    }
}

impl Drop for SharedMemoryRingBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.size);
            libc::close(self.fd);
        }
    }
}

fn open_shm_fd(name: &str) -> io::Result<RawFd> {
    let candidates = if name.starts_with('/') {
        vec![name.to_string()]
    } else {
        vec![name.to_string(), format!("/{}", name)]
    };

    for candidate in candidates {
        let c_name = CString::new(candidate.clone()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Invalid SHM name: {}", candidate),
            )
        })?;
        let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_RDWR, 0) };
        if fd >= 0 {
            return Ok(fd);
        }
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("Unable to open shared memory segment: {}", name),
    ))
}

pub fn unlink_shm_by_name(name: &str) -> io::Result<()> {
    let candidates = if name.starts_with('/') {
        vec![name.to_string()]
    } else {
        vec![name.to_string(), format!("/{}", name)]
    };

    let mut last_err: Option<io::Error> = None;
    for candidate in candidates {
        let c_name = CString::new(candidate.clone()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Invalid SHM name: {}", candidate),
            )
        })?;

        let rc = unsafe { libc::shm_unlink(c_name.as_ptr()) };
        if rc == 0 {
            return Ok(());
        }
        let err = io::Error::last_os_error();
        if err.kind() != io::ErrorKind::NotFound {
            last_err = Some(err);
        }
    }

    if let Some(err) = last_err {
        Err(err)
    } else {
        Ok(())
    }
}
