mod inference_client;
mod protocol;
mod receive_data;
mod send_data;
mod shm_ring;
mod reduced_hardware_client;
mod uart;
mod uart_commands;
mod valon_client;

use inference_client::InferenceSocketClient;
use protocol::{InferenceRequest, InferenceResponse, InferenceShmRequest};
use receive_data::{CircularBuffer, PowerMeasurement};
use shm_ring::{SharedMemoryRingBuffer, SharedMemoryRingSpec};
use reduced_hardware_client::ReducedHardwareInferenceClient;
use std::f32::consts::PI;
use std::io;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use uart::UartConfig;
use uart_commands::{CommandTracker, map_agent_controls};
use valon_client::{ValonCommandTracker, map_valon_controls, send_valon_commands};

const DEFAULT_ENABLE_UART_PATH: bool = true;
const DEFAULT_ENABLE_UDP_PATH: bool = true;
const DEFAULT_UART_USE_SYNTHETIC: bool = false;
const DEFAULT_UDP_USE_SYNTHETIC: bool = false;
const DEFAULT_ENABLE_INFERENCE: bool = true;
const DEFAULT_PRINT_INFERENCE_RESULTS: bool = true;
const DEFAULT_PRINT_UART_INPUT: bool = false;
const DEFAULT_PRINT_UDP_INPUT: bool = false;
const ADC_MAX_U12: f32 = 4095.0;
const ADC_VREF_VOLTS: f32 = 3.3;
const SENSOR_MIN_VOLTS: f32 = 0.2;
const SENSOR_MAX_VOLTS: f32 = 1.7;
const SENSOR_MIN_DBM: f32 = -30.0;
const SENSOR_MAX_DBM: f32 = 15.0;
const SENSOR_DBM_BIAS: f32 = 10.0;

#[derive(Clone, Copy)]
enum IpcMode {
    Direct,
    Shm,
}

#[derive(Clone)]
struct AppConfig {
    ipc_mode: IpcMode,
    socket_path: String,
    reduced_hardware: bool,
    reduced_socket_path: String,
    reduced_capture_path: String,
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
    enable_uart_path: bool,
    enable_udp_path: bool,
    uart_use_synthetic: bool,
    udp_use_synthetic: bool,
    enable_inference: bool,
    print_inference_results: bool,
    print_uart_input: bool,
    print_udp_input: bool,
    cleanup_shm_on_exit: bool,
    enable_valon: bool,
    valon_socket_path: String,
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
            reduced_hardware: false,
            reduced_socket_path: "/tmp/maars_reduced_hw.sock".to_string(),
            reduced_capture_path: "ila_capture.csv".to_string(),
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
            uart_port: "/dev/cu.usbmodem11203".to_string(),
            uart_baud: 115200,
            udp_bind: "0.0.0.0:5001".to_string(),
            enable_uart_path: DEFAULT_ENABLE_UART_PATH,
            enable_udp_path: DEFAULT_ENABLE_UDP_PATH,
            uart_use_synthetic: DEFAULT_UART_USE_SYNTHETIC,
            udp_use_synthetic: DEFAULT_UDP_USE_SYNTHETIC,
            enable_inference: DEFAULT_ENABLE_INFERENCE,
            print_inference_results: DEFAULT_PRINT_INFERENCE_RESULTS,
            print_uart_input: DEFAULT_PRINT_UART_INPUT,
            print_udp_input: DEFAULT_PRINT_UDP_INPUT,
            cleanup_shm_on_exit: false,
            enable_valon: true,
            valon_socket_path: std::env::var("MAARS_VALON_SOCKET_PATH")
                .unwrap_or_else(|_| "/tmp/valon5019.sock".to_string()),
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
    --reduced-hardware                 Run the isolated FFT center/bandwidth path\n\
    --reduced-socket-path <path>       Unix socket path for reduced-hardware worker\n\
    --reduced-capture-path <path>      ADC capture CSV path used in reduced-hardware mode\n\
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
        --enable-uart-path                 Enable UART input path (default: on)\n\
        --disable-uart-path                Disable UART input path\n\
        --uart-use-synthetic               Use synthetic UART input when inference is enabled\n\
        --uart-use-real                    Use real UART hardware input\n\
        --enable-udp-path                  Enable UDP input path (default: on)\n\
        --disable-udp-path                 Disable UDP input path\n\
        --udp-use-synthetic                Use synthetic UDP input when inference is enabled\n\
        --udp-use-real                     Use real UDP hardware input\n\
        --enable-inference                 Enable Python inference path (requires UART + UDP)\n\
        --disable-inference                Disable inference and run one displayed hardware path\n\
    --print-inference-results          Print inference summaries (default: on)\n\
    --no-print-inference-results       Disable inference summaries\n\
    --print-uart-input                 Print UART input data\n\
    --no-print-uart-input              Disable UART input data printing\n\
    --print-udp-input                  Print UDP packet debug output\n\
    --no-print-udp-input               Disable UDP packet debug output\n\
        --enable-valon                     Enable Valon LO socket output (default: on)\n\
        --disable-valon                    Disable Valon LO socket output\n\
        --valon-socket-path <path>         Valon Unix socket path (default: /tmp/valon5019.sock)\n\
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
            "--reduced-hardware" => {
                cfg.reduced_hardware = true;
            }
            "--reduced-socket-path" => {
                idx += 1;
                cfg.reduced_socket_path = args
                    .get(idx)
                    .ok_or("Missing value for --reduced-socket-path")?
                    .to_string();
            }
            "--reduced-capture-path" => {
                idx += 1;
                cfg.reduced_capture_path = args
                    .get(idx)
                    .ok_or("Missing value for --reduced-capture-path")?
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
            "--enable-uart-path" => {
                cfg.enable_uart_path = true;
            }
            "--disable-uart-path" => {
                cfg.enable_uart_path = false;
            }
            "--uart-use-synthetic" => {
                cfg.uart_use_synthetic = true;
            }
            "--uart-use-real" => {
                cfg.uart_use_synthetic = false;
            }
            "--enable-udp-path" => {
                cfg.enable_udp_path = true;
            }
            "--disable-udp-path" => {
                cfg.enable_udp_path = false;
            }
            "--udp-use-synthetic" => {
                cfg.udp_use_synthetic = true;
            }
            "--udp-use-real" => {
                cfg.udp_use_synthetic = false;
            }
            "--enable-inference" => {
                cfg.enable_inference = true;
            }
            "--disable-inference" => {
                cfg.enable_inference = false;
            }
            "--print-inference-results" => {
                cfg.print_inference_results = true;
            }
            "--no-print-inference-results" => {
                cfg.print_inference_results = false;
            }
            "--print-uart-input" => {
                cfg.print_uart_input = true;
            }
            "--no-print-uart-input" => {
                cfg.print_uart_input = false;
            }
            "--print-udp-input" => {
                cfg.print_udp_input = true;
            }
            "--no-print-udp-input" => {
                cfg.print_udp_input = false;
            }
            "--cleanup-shm-on-exit" => {
                cfg.cleanup_shm_on_exit = true;
            }
            "--enable-valon" => {
                cfg.enable_valon = true;
            }
            "--disable-valon" => {
                cfg.enable_valon = false;
            }
            "--valon-socket-path" => {
                idx += 1;
                cfg.valon_socket_path = args
                    .get(idx)
                    .ok_or("Missing value for --valon-socket-path")?
                    .to_string();
            }
            other => return Err(format!("Unknown argument: {}", other)),
        }
        idx += 1;
    }

    Ok(cfg)
}

fn validate_runtime_config(cfg: &AppConfig) -> Result<(), String> {
    if cfg.reduced_hardware {
        return Ok(());
    }

    if cfg.dry_run || cfg.simulate {
        return Ok(());
    }

    if cfg.enable_inference {
        if !cfg.enable_uart_path || !cfg.enable_udp_path {
            return Err(
                "Inference mode requires both --enable-uart-path and --enable-udp-path. Use --uart-use-synthetic and/or --udp-use-synthetic only when a path is not real hardware.".to_string(),
            );
        }

        return Ok(());
    }

    let enabled_paths = u8::from(cfg.enable_uart_path) + u8::from(cfg.enable_udp_path);
    if enabled_paths != 1 {
        return Err(
            "No-inference mode requires exactly one enabled path: either --enable-uart-path or --enable-udp-path.".to_string(),
        );
    }

    if cfg.enable_uart_path {
        if cfg.uart_use_synthetic {
            return Err(
                "No-inference UART mode must use real hardware, so --uart-use-synthetic is not allowed.".to_string(),
            );
        }
        if !cfg.print_uart_input {
            return Err(
                "No-inference UART mode is useless unless --print-uart-input is enabled.".to_string(),
            );
        }
    }

    if cfg.enable_udp_path {
        if cfg.udp_use_synthetic {
            return Err(
                "No-inference UDP mode must use real hardware, so --udp-use-synthetic is not allowed.".to_string(),
            );
        }
        if !cfg.print_udp_input {
            return Err(
                "No-inference UDP mode is useless unless --print-udp-input is enabled.".to_string(),
            );
        }
    }

    Ok(())
}

#[derive(Default)]
struct ReducedHardwareCommandTracker {
    last_filter_mhz: Option<u8>,
}

impl ReducedHardwareCommandTracker {
    fn commands_for(&mut self, bandwidth_class: u8) -> io::Result<Vec<Vec<u8>>> {
        let filter_mhz = match bandwidth_class {
            0 => 1,
            1 => 10,
            2 => 20,
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unsupported bandwidth_class value: {}", other),
                ));
            }
        };

        let mut commands = Vec::new();
        if self.last_filter_mhz != Some(filter_mhz) {
            commands.push(format!("filter {}\r\n", filter_mhz).into_bytes());
            self.last_filter_mhz = Some(filter_mhz);
        }
        Ok(commands)
    }
}

#[derive(Default)]
struct ReducedHardwareValonTracker {
    last_lo_freq_milli: Option<i32>,
}

impl ReducedHardwareValonTracker {
    fn commands_for(&mut self, seq_id: u64, center_class: u8) -> io::Result<Vec<valon_client::ValonCommand>> {
        let center_freq_mhz = reduced_center_frequency_mhz(center_class)?;
        let lo_freq_mhz = center_freq_mhz - 25.0;
        let lo_freq_milli = (lo_freq_mhz * 1000.0).round() as i32;

        let mut commands = Vec::new();
        if self.last_lo_freq_milli != Some(lo_freq_milli) {
            commands.push(valon_client::ValonCommand::SetFreq {
                seq_id,
                value_mhz: lo_freq_mhz,
            });
            self.last_lo_freq_milli = Some(lo_freq_milli);
        }
        Ok(commands)
    }
}

fn reduced_center_frequency_mhz(center_class: u8) -> io::Result<f32> {
    match center_class {
        0 => Ok(2405.0),
        1 => Ok(2420.0),
        2 => Ok(2435.0),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported center_class value: {}", other),
        )),
    }
}

fn generate_reduced_synthetic_adc_samples(
    n_samples: usize,
    center_class: u8,
    bandwidth_class: u8,
    phase_offset: f32,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_samples);
    let center_offset_hz = match center_class {
        0 => 5_000_000.0,
        1 => 12_000_000.0,
        _ => 20_000_000.0,
    };
    let bandwidth_hz = match bandwidth_class {
        0 => 1_000_000.0,
        1 => 10_000_000.0,
        _ => 20_000_000.0,
    };
    let tone_offsets: Vec<f32> = if bandwidth_class == 0 {
        vec![0.0]
    } else {
        vec![-0.35, -0.12, 0.0, 0.12, 0.35]
            .into_iter()
            .map(|fraction| fraction * bandwidth_hz)
            .collect()
    };
    for idx in 0..n_samples {
        let t = idx as f32 / 125_000_000.0;
        let mut sample = 0.0f32;
        for offset in &tone_offsets {
            let phase = phase_offset + (*offset / 1_000_000.0);
            sample += ((2.0 * PI * (center_offset_hz + *offset) * t) + phase).sin();
        }
        sample /= tone_offsets.len().max(1) as f32;
        let envelope = 0.65 + 0.35 * ((idx as f32 / n_samples.max(1) as f32) * PI).sin().abs();
        let noise = (((idx as f32 * 0.017 + phase_offset).sin()) * 0.08)
            + (((idx as f32 * 0.031 + phase_offset).cos()) * 0.04);
        out.push((sample * envelope + noise) * 72.0);
    }
    out
}

fn run_reduced_hardware_loop(cfg: &AppConfig) -> io::Result<()> {
    if cfg.simulate {
        return run_reduced_hardware_simulation(cfg);
    }

    let mut inference_client = ReducedHardwareInferenceClient::new(&cfg.reduced_socket_path);
    inference_client.connect()?;

    let mut uart_command_tx: Option<mpsc::Sender<Vec<u8>>> = None;
    let mut filter_tracker = ReducedHardwareCommandTracker::default();
    let mut valon_command_tx: Option<mpsc::Sender<valon_client::ValonCommand>> = None;
    let mut valon_tracker = ReducedHardwareValonTracker::default();

    if cfg.enable_uart_path {
        let uart_config = UartConfig::new(&cfg.uart_port, cfg.uart_baud);
        let uart = uart::Uart::open(&uart_config)?;
        let uart_writer = uart.try_clone()?;
        let (tx, rx) = mpsc::channel::<Vec<u8>>();
        thread::spawn(move || {
            if let Err(e) = send_data::send_uart_data(uart_writer, rx) {
                eprintln!("Reduced-hardware UART sender error: {}", e);
            }
        });
        uart_command_tx = Some(tx);
    }

    if cfg.enable_valon {
        let (tx, rx) = mpsc::channel::<valon_client::ValonCommand>();
        let valon_socket_path = cfg.valon_socket_path.clone();
        let print_logs = cfg.print_inference_results;
        thread::spawn(move || {
            if let Err(e) = send_valon_commands(&valon_socket_path, rx, print_logs) {
                eprintln!("Reduced-hardware Valon sender error: {}", e);
            }
        });
        valon_command_tx = Some(tx);
    }

    println!(
        "REDUCED HW mode active: uart_out={} valon_out={} capture_path={} socket_path={}",
        if uart_command_tx.is_some() { "on" } else { "off" },
        if valon_command_tx.is_some() { "on" } else { "off" },
        cfg.reduced_capture_path,
        cfg.reduced_socket_path,
    );

    let mut cycle: u64 = 0;
    loop {
        cycle += 1;
        let resp = inference_client.infer_capture_path(cycle, &cfg.reduced_capture_path, cfg.sample_rate_hz)?;

        if cfg.print_inference_results {
            println!(
                "REDUCED seq={} center={} ({:.1} MHz) bw={} ({:.1} MHz) center_conf={:.3} bw_conf={:.3}",
                resp.seq_id,
                resp.center_class,
                resp.center_frequency_mhz,
                resp.bandwidth_class,
                resp.bandwidth_mhz,
                resp.center_confidence,
                resp.bandwidth_confidence,
            );
        }

        if let Some(tx) = uart_command_tx.as_ref() {
            let commands = filter_tracker.commands_for(resp.bandwidth_class)?;
            for command in commands {
                if let Err(e) = tx.send(command.clone()) {
                    return Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        format!("Failed to queue reduced-hardware UART command for seq {}: {}", resp.seq_id, e),
                    ));
                }
                if cfg.print_inference_results {
                    let text = String::from_utf8_lossy(&command);
                    println!("REDUCED seq={} uart_cmd={}", resp.seq_id, text.trim_end_matches(['\r', '\n']));
                }
            }
        }

        if let Some(tx) = valon_command_tx.as_ref() {
            let commands = valon_tracker.commands_for(resp.seq_id, resp.center_class)?;
            for command in commands {
                if let Err(e) = tx.send(command) {
                    return Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        format!("Failed to queue reduced-hardware Valon command for seq {}: {}", resp.seq_id, e),
                    ));
                }
            }
        }

        thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn run_reduced_hardware_simulation(cfg: &AppConfig) -> io::Result<()> {
    let mut inference_client = ReducedHardwareInferenceClient::new(&cfg.reduced_socket_path);
    inference_client.connect()?;

    let mut filter_tracker = ReducedHardwareCommandTracker::default();
    let mut valon_tracker = ReducedHardwareValonTracker::default();

    println!(
        "REDUCED SIM mode active: samples={} interval_ms={}",
        cfg.dry_run_samples,
        cfg.simulate_interval_ms,
    );

    let mut cycle: u64 = 0;
    loop {
        cycle += 1;
        if cfg.simulate_cycles > 0 && cycle > cfg.simulate_cycles {
            break;
        }

        let center_class = ((cycle - 1) % 3) as u8;
        let bandwidth_class = (((cycle - 1) / 3) % 3) as u8;
        let adc_samples = generate_reduced_synthetic_adc_samples(
            cfg.dry_run_samples,
            center_class,
            bandwidth_class,
            cycle as f32 * 0.17,
        );

        let resp = inference_client.infer_samples(cycle, &adc_samples, cfg.sample_rate_hz)?;
        println!(
            "REDUCED SIM seq={} center={} ({:.1} MHz) bw={} ({:.1} MHz) center_conf={:.3} bw_conf={:.3}",
            resp.seq_id,
            resp.center_class,
            resp.center_frequency_mhz,
            resp.bandwidth_class,
            resp.bandwidth_mhz,
            resp.center_confidence,
            resp.bandwidth_confidence,
        );

        let filter_cmds = filter_tracker.commands_for(resp.bandwidth_class)?;
        for cmd in filter_cmds {
            let text = String::from_utf8_lossy(&cmd);
            println!("REDUCED SIM seq={} uart_cmd={}", resp.seq_id, text.trim_end_matches(['\r', '\n']));
        }

        let valon_cmds = valon_tracker.commands_for(resp.seq_id, resp.center_class)?;
        for cmd in valon_cmds {
            match cmd {
                valon_client::ValonCommand::SetFreq { value_mhz, .. } => {
                    println!("REDUCED SIM seq={} valon_cmd=set_freq {:.3}", resp.seq_id, value_mhz);
                }
                valon_client::ValonCommand::SetRfLevel { value_dbm, .. } => {
                    println!("REDUCED SIM seq={} valon_cmd=set_rflevel {:.3}", resp.seq_id, value_dbm);
                }
            }
        }

        thread::sleep(std::time::Duration::from_millis(cfg.simulate_interval_ms));
    }

    Ok(())
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

    if let Err(e) = validate_runtime_config(&cfg) {
        eprintln!("Configuration error: {}", e);
        print_help();
        std::process::exit(2);
    }

    if cfg.reduced_hardware {
        if let Err(e) = run_reduced_hardware_loop(&cfg) {
            eprintln!("Reduced-hardware runtime failed: {}", e);
            std::process::exit(1);
        }
        return;
    }

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
    let mut inference_client = if cfg.enable_inference {
        let mut client = InferenceSocketClient::new(&cfg.socket_path);
        client.connect()?;
        Some(client)
    } else {
        None
    };

    let mut shm_ring = if cfg.enable_inference {
        if let IpcMode::Shm = cfg.ipc_mode {
            Some(SharedMemoryRingBuffer::attach(SharedMemoryRingSpec {
                name: cfg.shm_name.clone(),
                num_slots: cfg.shm_slots,
                slot_capacity: cfg.shm_slot_capacity,
            })?)
        } else {
            None
        }
    } else {
        None
    };

    let uart_buffer = Arc::new(Mutex::new(CircularBuffer::<PowerMeasurement>::new(1024)));
    let udp_buffer = Arc::new(Mutex::new(CircularBuffer::<receive_data::IQSample>::new(1024)));
    let mut uart_command_tx: Option<mpsc::Sender<Vec<u8>>> = None;
    let mut command_tracker = CommandTracker::default();
    let mut valon_command_tx: Option<mpsc::Sender<valon_client::ValonCommand>> = None;
    let mut valon_tracker = ValonCommandTracker::default();

    if cfg.enable_inference && cfg.enable_valon {
        let (tx, rx) = mpsc::channel::<valon_client::ValonCommand>();
        let valon_socket_path = cfg.valon_socket_path.clone();
        let print_logs = cfg.print_inference_results;
        thread::spawn(move || {
            if let Err(e) = send_valon_commands(&valon_socket_path, rx, print_logs) {
                eprintln!("Valon sender error: {}", e);
            }
        });
        valon_command_tx = Some(tx);
    }

    let mut uart_reader_started = false;
    if cfg.enable_uart_path && (!cfg.enable_inference || !cfg.uart_use_synthetic) {
        let uart_config = UartConfig::new(&cfg.uart_port, cfg.uart_baud);
        let uart = uart::Uart::open(&uart_config)?;

        if cfg.enable_inference {
            let uart_writer = uart.try_clone()?;
            let (tx, rx) = mpsc::channel::<Vec<u8>>();
            thread::spawn(move || {
                if let Err(e) = send_data::send_uart_data(uart_writer, rx) {
                    eprintln!("UART sender error: {}", e);
                }
            });
            uart_command_tx = Some(tx);
        }

        let uart_buf_clone = Arc::clone(&uart_buffer);
        let print_uart_input = cfg.print_uart_input;
        thread::spawn(move || {
            if let Err(e) = receive_data::receive_uart_adc_measurements(uart, uart_buf_clone, print_uart_input) {
                eprintln!("UART receiver error: {}", e);
            }
        });
        uart_reader_started = true;
    }

    let mut udp_reader_started = false;
    if cfg.enable_udp_path && (!cfg.enable_inference || !cfg.udp_use_synthetic) {
        let udp_buf_clone = Arc::clone(&udp_buffer);
        let bind_addr = cfg.udp_bind.clone();
        let print_udp_input = cfg.print_udp_input;
        thread::spawn(move || {
            if let Err(e) = receive_data::receive_udp_data_with_bind(&bind_addr, udp_buf_clone, print_udp_input) {
                eprintln!("UDP receiver error: {}", e);
            }
        });
        udp_reader_started = true;
    }

    println!(
        "HW mode active: uart={} ({}) udp={} ({}) inference={} uart_cmd_tx={} valon={} uart_print={} udp_print={} ipc={}",
        if cfg.enable_uart_path { "on" } else { "off" },
        if cfg.enable_inference && cfg.uart_use_synthetic {
            "synthetic"
        } else if uart_reader_started {
            "real"
        } else {
            "off"
        },
        if cfg.enable_udp_path { "on" } else { "off" },
        if cfg.enable_inference && cfg.udp_use_synthetic {
            "synthetic"
        } else if udp_reader_started {
            "real"
        } else {
            "off"
        },
        if cfg.enable_inference { "on" } else { "off" },
        if uart_command_tx.is_some() { "on" } else { "off" },
        if valon_command_tx.is_some() { "on" } else { "off" },
        if cfg.print_uart_input { "on" } else { "off" },
        if cfg.print_udp_input { "on" } else { "off" },
        match cfg.ipc_mode {
            IpcMode::Direct => "direct",
            IpcMode::Shm => "shm",
        },
    );

    let mut slot_index = 0usize;
    let mut cycle: u64 = 0;

    loop {
        cycle += 1;

        if !cfg.enable_inference {
            thread::sleep(std::time::Duration::from_millis(100));
            continue;
        }

        let Some(power_raw) = acquire_power_sample(cfg, &uart_buffer, cycle) else {
            thread::sleep(std::time::Duration::from_millis(20));
            continue;
        };
        let Some(iq_iq_pairs) = acquire_iq_samples(cfg, &udp_buffer, cycle) else {
            thread::sleep(std::time::Duration::from_millis(20));
            continue;
        };

        let power_lna_dbm = calibrate_power_raw_to_dbm(power_raw.power_lna_raw);
        let power_pa_dbm = calibrate_power_raw_to_dbm(power_raw.power_pa_raw);
        let resp = infer_once(
            cfg,
            inference_client.as_mut().unwrap(),
            &mut shm_ring,
            &mut slot_index,
            cycle,
            &iq_iq_pairs,
            power_lna_dbm,
            power_pa_dbm,
        )?;

        if cfg.print_inference_results {
            println!(
                "HW seq={} status={} lna={} filter={} center={} mixer_dbm={:.3} ifamp_db={:.3} evm={:.3} pt_ms={:.3} power_lna_dbm={:.3} power_pa_dbm={:.3}",
                resp.seq_id,
                resp.status_code,
                resp.lna_class,
                resp.filter_class,
                resp.center_class,
                resp.mixer_dbm,
                resp.ifamp_db,
                resp.evm_value,
                resp.processing_time_ms,
                power_lna_dbm,
                power_pa_dbm,
            );
        }

        if let Some(tx) = uart_command_tx.as_ref() {
            let control_values = map_agent_controls(&resp)?;
            let commands = command_tracker.commands_for(&control_values);
            for command in commands {
                if let Err(e) = tx.send(command.clone()) {
                    return Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        format!("Failed to queue UART command for seq {}: {}", resp.seq_id, e),
                    ));
                }
                if cfg.print_inference_results {
                    let text = String::from_utf8_lossy(&command);
                    println!(
                        "HW seq={} uart_cmd={}",
                        resp.seq_id,
                        text.trim_end_matches(['\r', '\n'])
                    );
                }
            }
        }

        if let Some(tx) = valon_command_tx.as_ref() {
            let valon_values = map_valon_controls(&resp)?;
            let commands = valon_tracker.commands_for(resp.seq_id, &valon_values);
            for command in commands {
                if let Err(e) = tx.send(command) {
                    return Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        format!("Failed to queue Valon command for seq {}: {}", resp.seq_id, e),
                    ));
                }
            }
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
    let mut command_tracker = CommandTracker::default();
    let mut valon_tracker = ValonCommandTracker::default();
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

        let control_values = map_agent_controls(&resp)?;
        let commands = command_tracker.commands_for(&control_values);
        for command in commands {
            let text = String::from_utf8_lossy(&command);
            println!(
                "SIM cycle={} uart_cmd={}",
                cycle,
                text.trim_end_matches(['\r', '\n'])
            );
        }

        if cfg.enable_valon {
            let valon_values = map_valon_controls(&resp)?;
            let valon_cmds = valon_tracker.commands_for(resp.seq_id, &valon_values);
            for cmd in valon_cmds {
                match cmd {
                    valon_client::ValonCommand::SetFreq { value_mhz, .. } => {
                        println!("SIM cycle={} valon_cmd=set_freq {:.3}", cycle, value_mhz);
                    }
                    valon_client::ValonCommand::SetRfLevel { value_dbm, .. } => {
                        println!("SIM cycle={} valon_cmd=set_rflevel {:.3}", cycle, value_dbm);
                    }
                }
            }
        }

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

fn generate_synthetic_power_raw(cycle: u64) -> PowerMeasurement {
    let phase = cycle as f32 * 0.11;
    let lna = ((phase.sin() * 0.5 + 0.5) * ADC_MAX_U12).round().clamp(0.0, ADC_MAX_U12);
    let pa = ((phase.cos() * 0.5 + 0.5) * ADC_MAX_U12).round().clamp(0.0, ADC_MAX_U12);
    PowerMeasurement {
        power_lna_raw: lna,
        power_pa_raw: pa,
    }
}

fn acquire_power_sample(
    cfg: &AppConfig,
    uart_buffer: &Arc<Mutex<CircularBuffer<PowerMeasurement>>>,
    cycle: u64,

) -> Option<PowerMeasurement> {
    if !cfg.enable_uart_path {
        return None;
    }

    if cfg.enable_inference && cfg.uart_use_synthetic {
        return Some(generate_synthetic_power_raw(cycle));
    }

    uart_buffer.lock().unwrap().read()
}

fn acquire_iq_samples(
    cfg: &AppConfig,
    udp_buffer: &Arc<Mutex<CircularBuffer<receive_data::IQSample>>>,
    cycle: u64,
) -> Option<Vec<(f32, f32)>> {
    if !cfg.enable_udp_path {
        return None;
    }

    if cfg.enable_inference && cfg.udp_use_synthetic {
        return Some(generate_synthetic_iq(cfg.dry_run_samples, cycle as f32 * 0.1));
    }

    if let Some(sample) = udp_buffer.lock().unwrap().read() {
        if let Ok(parsed) = sample.parse_qi_interleaved_i16_to_iq_f32() {
            return Some(parsed);
        }
    }

    None
}

fn calibrate_power_raw_to_dbm(raw_u12: f32) -> f32 {
    let raw = raw_u12.clamp(0.0, ADC_MAX_U12);
    let voltage = (raw / ADC_MAX_U12) * ADC_VREF_VOLTS;
    let clamped_voltage = voltage.clamp(SENSOR_MIN_VOLTS, SENSOR_MAX_VOLTS);
    let span = SENSOR_MAX_VOLTS - SENSOR_MIN_VOLTS;
    let ratio = if span > 0.0 {
        (clamped_voltage - SENSOR_MIN_VOLTS) / span
    } else {
        0.0
    };
    let dbm = SENSOR_MIN_DBM + ratio * (SENSOR_MAX_DBM - SENSOR_MIN_DBM);
    dbm + SENSOR_DBM_BIAS
}