mod inference_client;
mod protocol;
mod receive_data;
mod send_data;
mod shm_ring;
mod uart;

use inference_client::InferenceSocketClient;
use protocol::{InferenceRequest, InferenceResponse, InferenceShmRequest};
use receive_data::{CircularBuffer, PowerMeasurement};
use shm_ring::{SharedMemoryRingBuffer, SharedMemoryRingSpec};
use std::f32::consts::PI;
use std::io;
use std::sync::{Arc, Mutex};
use std::thread;
use uart::UartConfig;

#[derive(Clone, Copy)]
enum IpcMode {
    Direct,
    Shm,
}

#[derive(Clone)]
struct AppConfig {
    ipc_mode: IpcMode,
    socket_path: String,
    sample_rate_hz: f64,
    shm_name: String,
    shm_slots: usize,
    shm_slot_capacity: usize,
    dry_run: bool,
    simulate: bool,
    dry_run_cycles: u64,
    simulate_cycles: u64,
    simulate_interval_ms: u64,
    dry_run_samples: usize,
    dry_run_power_lna_dbm: f32,
    dry_run_power_pa_dbm: f32,
    uart_port: String,
    uart_baud: u32,
    udp_bind: String,
    cleanup_shm_on_exit: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        let ipc_mode = match std::env::var("MAARS_IPC_MODE") {
            Ok(v) if v.eq_ignore_ascii_case("shm") => IpcMode::Shm,
            _ => IpcMode::Direct,
        };

        Self {
            ipc_mode,
            socket_path: "/tmp/maars_infer.sock".to_string(),
            sample_rate_hz: 25_000_000.0,
            shm_name: std::env::var("MAARS_SHM_NAME")
                .unwrap_or_else(|_| "maars_iq_ring".to_string()),
            shm_slots: std::env::var("MAARS_SHM_SLOTS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(8),
            shm_slot_capacity: std::env::var("MAARS_SHM_SLOT_CAPACITY")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(8192),
            dry_run: false,
            simulate: false,
            dry_run_cycles: 1,
            simulate_cycles: 0,
            simulate_interval_ms: 200,
            dry_run_samples: 4096,
            dry_run_power_lna_dbm: -35.0,
            dry_run_power_pa_dbm: -22.0,
            uart_port: "/dev/cu.usbmodem1203".to_string(),
            uart_baud: 115200,
            udp_bind: "0.0.0.0:5001".to_string(),
            cleanup_shm_on_exit: false,
        }
    }
}

fn print_help() {
    println!(
        "software-framework\n\
Usage:\n\
  cargo run -- [options]\n\n\
Options:\n\
  --ipc-mode <direct|shm>            IPC mode (default: direct)\n\
  --socket-path <path>               Unix socket path (default: /tmp/maars_infer.sock)\n\
  --sample-rate-hz <float>           Sample rate sent to inference worker\n\
  --shm-name <name>                  SHM segment name (default: maars_iq_ring)\n\
  --shm-slots <int>                  SHM slot count (default: 8)\n\
  --shm-slot-capacity <int>          SHM slot capacity in IQ samples (default: 8192)\n\
  --dry-run                          Run without UART/UDP hardware using synthetic IQ\n\
    --simulate                         Continuous hardware-free simulation in CLI\n\
  --dry-run-cycles <int>             Number of dry-run inference cycles (default: 1)\n\
    --simulate-cycles <int>            Simulation cycles (0 = run continuously, default: 0)\n\
    --simulate-interval-ms <int>       Delay between simulation cycles (default: 200)\n\
  --dry-run-samples <int>            Synthetic IQ sample count per cycle (default: 4096)\n\
  --dry-run-power-lna <float>        Synthetic LNA power dBm (default: -35)\n\
  --dry-run-power-pa <float>         Synthetic PA power dBm (default: -22)\n\
    --uart-port <path>                 UART port path (default: /dev/cu.usbmodem1203)\n\
  --uart-baud <int>                  UART baud (default: 115200)\n\
  --udp-bind <host:port>             UDP bind address for IQ input (default: 127.0.0.1:5000)\n\
    --cleanup-shm-on-exit              Explicitly unlink SHM segment at simulation/dry-run teardown\n\
  --help                             Show this help\n"
    );
}

fn parse_args() -> Result<AppConfig, String> {
    let mut cfg = AppConfig::default();
    let args: Vec<String> = std::env::args().collect();
    let mut idx = 1usize;

    while idx < args.len() {
        let key = &args[idx];
        match key.as_str() {
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--ipc-mode" => {
                idx += 1;
                let val = args.get(idx).ok_or("Missing value for --ipc-mode")?;
                cfg.ipc_mode = match val.as_str() {
                    "direct" => IpcMode::Direct,
                    "shm" => IpcMode::Shm,
                    _ => return Err(format!("Invalid --ipc-mode: {}", val)),
                };
            }
            "--socket-path" => {
                idx += 1;
                cfg.socket_path = args
                    .get(idx)
                    .ok_or("Missing value for --socket-path")?
                    .to_string();
            }
            "--sample-rate-hz" => {
                idx += 1;
                cfg.sample_rate_hz = args
                    .get(idx)
                    .ok_or("Missing value for --sample-rate-hz")?
                    .parse::<f64>()
                    .map_err(|_| "Invalid float for --sample-rate-hz")?;
            }
            "--shm-name" => {
                idx += 1;
                cfg.shm_name = args
                    .get(idx)
                    .ok_or("Missing value for --shm-name")?
                    .to_string();
            }
            "--shm-slots" => {
                idx += 1;
                cfg.shm_slots = args
                    .get(idx)
                    .ok_or("Missing value for --shm-slots")?
                    .parse::<usize>()
                    .map_err(|_| "Invalid int for --shm-slots")?;
            }
            "--shm-slot-capacity" => {
                idx += 1;
                cfg.shm_slot_capacity = args
                    .get(idx)
                    .ok_or("Missing value for --shm-slot-capacity")?
                    .parse::<usize>()
                    .map_err(|_| "Invalid int for --shm-slot-capacity")?;
            }
            "--dry-run" => {
                cfg.dry_run = true;
            }
            "--simulate" => {
                cfg.simulate = true;
            }
            "--dry-run-cycles" => {
                idx += 1;
                cfg.dry_run_cycles = args
                    .get(idx)
                    .ok_or("Missing value for --dry-run-cycles")?
                    .parse::<u64>()
                    .map_err(|_| "Invalid int for --dry-run-cycles")?;
            }
            "--simulate-cycles" => {
                idx += 1;
                cfg.simulate_cycles = args
                    .get(idx)
                    .ok_or("Missing value for --simulate-cycles")?
                    .parse::<u64>()
                    .map_err(|_| "Invalid int for --simulate-cycles")?;
            }
            "--simulate-interval-ms" => {
                idx += 1;
                cfg.simulate_interval_ms = args
                    .get(idx)
                    .ok_or("Missing value for --simulate-interval-ms")?
                    .parse::<u64>()
                    .map_err(|_| "Invalid int for --simulate-interval-ms")?;
            }
            "--dry-run-samples" => {
                idx += 1;
                cfg.dry_run_samples = args
                    .get(idx)
                    .ok_or("Missing value for --dry-run-samples")?
                    .parse::<usize>()
                    .map_err(|_| "Invalid int for --dry-run-samples")?;
            }
            "--dry-run-power-lna" => {
                idx += 1;
                cfg.dry_run_power_lna_dbm = args
                    .get(idx)
                    .ok_or("Missing value for --dry-run-power-lna")?
                    .parse::<f32>()
                    .map_err(|_| "Invalid float for --dry-run-power-lna")?;
            }
            "--dry-run-power-pa" => {
                idx += 1;
                cfg.dry_run_power_pa_dbm = args
                    .get(idx)
                    .ok_or("Missing value for --dry-run-power-pa")?
                    .parse::<f32>()
                    .map_err(|_| "Invalid float for --dry-run-power-pa")?;
            }
            "--uart-port" => {
                idx += 1;
                cfg.uart_port = args
                    .get(idx)
                    .ok_or("Missing value for --uart-port")?
                    .to_string();
            }
            "--uart-baud" => {
                idx += 1;
                cfg.uart_baud = args
                    .get(idx)
                    .ok_or("Missing value for --uart-baud")?
                    .parse::<u32>()
                    .map_err(|_| "Invalid int for --uart-baud")?;
            }
            "--udp-bind" => {
                idx += 1;
                cfg.udp_bind = args
                    .get(idx)
                    .ok_or("Missing value for --udp-bind")?
                    .to_string();
            }
            "--cleanup-shm-on-exit" => {
                cfg.cleanup_shm_on_exit = true;
            }
            other => return Err(format!("Unknown argument: {}", other)),
        }
        idx += 1;
    }

    Ok(cfg)
}

fn main() {
    let cfg = match parse_args() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Argument error: {}", e);
            print_help();
            std::process::exit(2);
        }
    };

    if cfg.dry_run {
        if let Err(e) = run_dry_run(&cfg) {
            eprintln!("Dry-run failed: {}", e);
            std::process::exit(1);
        }
        return;
    }

    if cfg.simulate {
        if let Err(e) = run_simulation(&cfg) {
            eprintln!("Simulation failed: {}", e);
            std::process::exit(1);
        }
        return;
    }

    if let Err(e) = run_hardware_loop(&cfg) {
        eprintln!("Runtime failed: {}", e);
        std::process::exit(1);
    }
}

fn run_hardware_loop(cfg: &AppConfig) -> io::Result<()> {
    let uart_buffer = Arc::new(Mutex::new(CircularBuffer::<PowerMeasurement>::new(1024)));

    let uart_config = UartConfig::new(&cfg.uart_port, cfg.uart_baud);
    let uart = uart::Uart::open(&uart_config)?;

    println!(
        "UART power-sensor mode active on {} @ {} baud (UDP disabled)",
        cfg.uart_port, cfg.uart_baud
    );

    let uart_buf_clone = Arc::clone(&uart_buffer);
    let _uart_read_thread = thread::spawn(move || {
        if let Err(e) = receive_data::receive_uart_adc_measurements(uart, uart_buf_clone) {
            eprintln!("UART receiver error: {}", e);
        }
    });

    loop {
        if let Some(power) = uart_buffer.lock().unwrap().read() {
            println!(
                "Power sensors - value1: 0x{:03X} value2: 0x{:03X}",
                power.power_lna_raw as u16,
                power.power_pa_raw as u16
            );
        }
        thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn run_dry_run(cfg: &AppConfig) -> io::Result<()> {
    let mut inference_client = InferenceSocketClient::new(&cfg.socket_path);
    inference_client.connect()?;

    let mut shm_ring = if let IpcMode::Shm = cfg.ipc_mode {
        Some(SharedMemoryRingBuffer::attach(SharedMemoryRingSpec {
            name: cfg.shm_name.clone(),
            num_slots: cfg.shm_slots,
            slot_capacity: cfg.shm_slot_capacity,
        })?)
    } else {
        None
    };

    let mut slot_index = 0usize;
    for cycle in 0..cfg.dry_run_cycles {
        let iq_iq_pairs = generate_synthetic_iq(cfg.dry_run_samples, cycle as f32 * 0.1);
        let seq_id = cycle + 1;

        let resp = infer_once(
            cfg,
            &mut inference_client,
            &mut shm_ring,
            &mut slot_index,
            seq_id,
            &iq_iq_pairs,
            cfg.dry_run_power_lna_dbm,
            cfg.dry_run_power_pa_dbm,
        )?;

        println!(
            "DRY_RUN seq={} status={} lna={} filter={} center={} mixer_dbm={:.3} ifamp_db={:.3} evm={:.3} pt_ms={:.3}",
            resp.seq_id,
            resp.status_code,
            resp.lna_class,
            resp.filter_class,
            resp.center_class,
            resp.mixer_dbm,
            resp.ifamp_db,
            resp.evm_value,
            resp.processing_time_ms,
        );
    }

    drop(shm_ring);
    maybe_cleanup_shm(cfg);

    Ok(())
}

fn run_simulation(cfg: &AppConfig) -> io::Result<()> {
    let mut inference_client = InferenceSocketClient::new(&cfg.socket_path);
    inference_client.connect()?;

    let mut shm_ring = if let IpcMode::Shm = cfg.ipc_mode {
        Some(SharedMemoryRingBuffer::attach(SharedMemoryRingSpec {
            name: cfg.shm_name.clone(),
            num_slots: cfg.shm_slots,
            slot_capacity: cfg.shm_slot_capacity,
        })?)
    } else {
        None
    };

    println!(
        "SIMULATION START mode={} sample_rate_hz={} samples={} interval_ms={}",
        match cfg.ipc_mode {
            IpcMode::Direct => "direct",
            IpcMode::Shm => "shm",
        },
        cfg.sample_rate_hz,
        cfg.dry_run_samples,
        cfg.simulate_interval_ms,
    );

    let mut slot_index = 0usize;
    let mut cycle: u64 = 0;
    loop {
        cycle += 1;
        if cfg.simulate_cycles > 0 && cycle > cfg.simulate_cycles {
            break;
        }

        let phase = cycle as f32 * 0.07;
        let iq_iq_pairs = generate_synthetic_iq(cfg.dry_run_samples, phase);
        let power_lna = cfg.dry_run_power_lna_dbm + (phase.sin() * 0.8);
        let power_pa = cfg.dry_run_power_pa_dbm + (phase.cos() * 0.8);

        let resp = infer_once(
            cfg,
            &mut inference_client,
            &mut shm_ring,
            &mut slot_index,
            cycle,
            &iq_iq_pairs,
            power_lna,
            power_pa,
        )?;

        let preview_n = iq_iq_pairs.len().min(3);
        let preview = iq_iq_pairs
            .iter()
            .take(preview_n)
            .map(|(i, q)| format!("({:.1},{:.1})", i, q))
            .collect::<Vec<_>>()
            .join(", ");

        println!(
            "SIM cycle={} iq_preview=[{}] power_lna={:.3} power_pa={:.3} -> lna={} filter={} center={} mixer_dbm={:.3} ifamp_db={:.3} evm={:.3} pt_ms={:.3}",
            cycle,
            preview,
            power_lna,
            power_pa,
            resp.lna_class,
            resp.filter_class,
            resp.center_class,
            resp.mixer_dbm,
            resp.ifamp_db,
            resp.evm_value,
            resp.processing_time_ms,
        );

        thread::sleep(std::time::Duration::from_millis(cfg.simulate_interval_ms));
    }

    println!("SIMULATION END total_cycles={}", cycle.saturating_sub(1));

    drop(shm_ring);
    maybe_cleanup_shm(cfg);

    Ok(())
}

fn infer_once(
    cfg: &AppConfig,
    inference_client: &mut InferenceSocketClient,
    shm_ring: &mut Option<SharedMemoryRingBuffer>,
    slot_index: &mut usize,
    seq_id: u64,
    iq_iq_pairs: &[(f32, f32)],
    power_lna_dbm: f32,
    power_pa_dbm: f32,
) -> io::Result<InferenceResponse> {
    if let Some(ring) = shm_ring.as_mut() {
        let n_samples = ring.write_slot(*slot_index, iq_iq_pairs)?;
        let req = InferenceShmRequest {
            seq_id,
            sample_rate_hz: cfg.sample_rate_hz,
            power_lna_dbm,
            power_pa_dbm,
            slot_index: *slot_index as u32,
            n_samples: n_samples as u32,
        };
        *slot_index = if ring.num_slots() > 0 {
            (*slot_index + 1) % ring.num_slots()
        } else {
            0
        };
        inference_client.infer_shm(&req)
    } else {
        let req = InferenceRequest {
            seq_id,
            sample_rate_hz: cfg.sample_rate_hz,
            power_lna_dbm,
            power_pa_dbm,
            iq_iq_pairs,
        };
        inference_client.infer(&req)
    }
}

fn maybe_cleanup_shm(cfg: &AppConfig) {
    if !cfg.cleanup_shm_on_exit {
        return;
    }
    if let IpcMode::Shm = cfg.ipc_mode {
        match shm_ring::unlink_shm_by_name(&cfg.shm_name) {
            Ok(()) => println!("SHM cleanup complete for {}", cfg.shm_name),
            Err(e) => eprintln!("SHM cleanup warning for {}: {}", cfg.shm_name, e),
        }
    }
}

fn generate_synthetic_iq(n_samples: usize, phase_offset: f32) -> Vec<(f32, f32)> {
    let mut out = Vec::with_capacity(n_samples);
    for idx in 0..n_samples {
        let t = (idx as f32 / n_samples as f32) * 2.0 * PI + phase_offset;
        let i = (t.sin() * 12000.0).clamp(-32768.0, 32767.0);
        let q = (t.cos() * 12000.0).clamp(-32768.0, 32767.0);
        out.push((i, q));
    }
    out
}