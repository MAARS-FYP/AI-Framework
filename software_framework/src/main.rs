mod inference_client;
mod protocol;
mod receive_data;
mod rfchain_client;
mod send_data;
mod shm_ring;
mod uart;
mod uart_commands;
mod valon_client;

use inference_client::InferenceSocketClient;
use protocol::{InferenceRequest, InferenceResponse, InferenceShmRequest};
use receive_data::{CircularBuffer, PowerMeasurement};
use rfchain_client::RFChainSocketClient;
use shm_ring::{SharedMemoryRingBuffer, SharedMemoryRingSpec};
use std::f32::consts::PI;
use std::fs;
use std::io;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::process::{Child, Command, Stdio};
use std::thread;
use uart::UartConfig;
use uart_commands::{CommandTracker, map_agent_controls};
use valon_client::{ValonCommandTracker, map_valon_controls, send_valon_commands};

const DEFAULT_ENABLE_UART_PATH: bool = true;
const DEFAULT_ENABLE_ILA_PATH: bool = true;
const DEFAULT_UART_USE_SYNTHETIC: bool = false;
const DEFAULT_ILA_USE_SYNTHETIC: bool = false;
const DEFAULT_ENABLE_INFERENCE: bool = true;
const DEFAULT_PRINT_INFERENCE_RESULTS: bool = true;
const DEFAULT_PRINT_UART_INPUT: bool = false;
const DEFAULT_PRINT_ILA_INPUT: bool = false;
const DEFAULT_ENABLE_CONSTELLATION_PLOT: bool = false;
const DEFAULT_ILA_POLL_INTERVAL_MS: u64 = 20;
const DEFAULT_ILA_REQUEST_TIMEOUT_MS: u64 = 5000;
const DEFAULT_ILA_BATCH_SAMPLES: usize = 2048;
const DEFAULT_PLOT_REFRESH_HZ: f64 = 8.0;
const MIN_ILA_BATCH_SAMPLES_INFERENCE: usize = 1025;
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

#[derive(Clone, Copy, Debug)]
enum OperatingMode {
    Hardware,
    Simulate,
    DigitalTwin,
}

#[derive(Clone)]
struct AppConfig {
    mode: OperatingMode,
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
    ila_csv_path: String,
    ila_request_flag_path: String,
    ila_poll_interval_ms: u64,
    ila_request_timeout_ms: u64,
    ila_batch_samples: usize,
    enable_uart_path: bool,
    enable_ila_path: bool,
    uart_use_synthetic: bool,
    ila_use_synthetic: bool,
    enable_inference: bool,
    print_inference_results: bool,
    print_uart_input: bool,
    print_ila_input: bool,
    cleanup_shm_on_exit: bool,
    enable_valon: bool,
    valon_socket_path: String,
    inference_txt_path: String,
    rf_chain_socket_path: String,
    rf_chain_interval_ms: u64,
    rf_chain_cycles: u64,
    plot_constellation: bool,
    plot_python_bin: String,
    plot_script_path: String,
    plot_slot_index_path: String,
    plot_refresh_hz: f64,
}

impl Default for AppConfig {
    fn default() -> Self {
        let ipc_mode = match std::env::var("MAARS_IPC_MODE") {
            Ok(v) if v.eq_ignore_ascii_case("shm") => IpcMode::Shm,
            _ => IpcMode::Direct,
        };

        Self {
            mode: OperatingMode::Hardware,
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
            uart_port: "/dev/cu.usbmodem11203".to_string(),
            uart_baud: 115200,
            ila_csv_path: "./ila_probe0.csv".to_string(),
            ila_request_flag_path: "./ila_capture_request.txt".to_string(),
            ila_poll_interval_ms: DEFAULT_ILA_POLL_INTERVAL_MS,
            ila_request_timeout_ms: DEFAULT_ILA_REQUEST_TIMEOUT_MS,
            ila_batch_samples: DEFAULT_ILA_BATCH_SAMPLES,
            enable_uart_path: DEFAULT_ENABLE_UART_PATH,
            enable_ila_path: DEFAULT_ENABLE_ILA_PATH,
            uart_use_synthetic: DEFAULT_UART_USE_SYNTHETIC,
            ila_use_synthetic: DEFAULT_ILA_USE_SYNTHETIC,
            enable_inference: DEFAULT_ENABLE_INFERENCE,
            print_inference_results: DEFAULT_PRINT_INFERENCE_RESULTS,
            print_uart_input: DEFAULT_PRINT_UART_INPUT,
            print_ila_input: DEFAULT_PRINT_ILA_INPUT,
            cleanup_shm_on_exit: false,
            enable_valon: true,
            valon_socket_path: std::env::var("MAARS_VALON_SOCKET_PATH")
                .unwrap_or_else(|_| "/tmp/valon5019.sock".to_string()),
            inference_txt_path: std::env::var("MAARS_INFERENCE_TXT_PATH")
                .unwrap_or_else(|_| "./inference_results.txt".to_string()),
            rf_chain_socket_path: std::env::var("MAARS_RFCHAIN_SOCKET_PATH")
                .unwrap_or_else(|_| "/tmp/maars_rfchain.sock".to_string()),
            rf_chain_interval_ms: 100,
            rf_chain_cycles: 0,
            plot_constellation: DEFAULT_ENABLE_CONSTELLATION_PLOT,
            plot_python_bin: std::env::var("MAARS_PLOT_PYTHON_BIN")
                .unwrap_or_else(|_| "python3".to_string()),
            plot_script_path: std::env::var("MAARS_PLOT_SCRIPT_PATH")
                .unwrap_or_else(|_| "../pluto_live_plot.py".to_string()),
            plot_slot_index_path: std::env::var("MAARS_PLOT_SLOT_INDEX_PATH")
                .unwrap_or_else(|_| "/tmp/maars_iq_ring_slot.txt".to_string()),
            plot_refresh_hz: std::env::var("MAARS_PLOT_REFRESH_HZ")
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(DEFAULT_PLOT_REFRESH_HZ),
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
    --dry-run                          Run without UART/ILA hardware using synthetic IQ\n\
    --simulate                         Continuous hardware-free simulation in CLI\n\
  --dry-run-cycles <int>             Number of dry-run inference cycles (default: 1)\n\
    --simulate-cycles <int>            Simulation cycles (0 = run continuously, default: 0)\n\
    --simulate-interval-ms <int>       Delay between simulation cycles (default: 200)\n\
  --dry-run-samples <int>            Synthetic IQ sample count per cycle (default: 4096)\n\
  --dry-run-power-lna <float>        Synthetic LNA power dBm (default: -35)\n\
  --dry-run-power-pa <float>         Synthetic PA power dBm (default: -22)\n\
    --uart-port <path>                 UART port path (default: /dev/cu.usbmodem1203)\n\
  --uart-baud <int>                  UART baud (default: 115200)\n\
    --ila-csv-path <path>              ILA probe0 CSV path (default: ./ila_probe0.csv)\n\
    --ila-request-flag-path <path>     ILA capture request flag file (default: ./ila_capture_request.txt)\n\
    --ila-poll-interval-ms <int>       Poll interval for ILA handshake/CSV reads (default: 20)\n\
    --ila-request-timeout-ms <int>     Timeout waiting for ILA request ack (default: 5000)\n\
    --ila-batch-samples <int>          Probe0 rows consumed per inference (default: 2048)\n\
        --enable-uart-path                 Enable UART input path (default: on)\n\
        --disable-uart-path                Disable UART input path\n\
        --uart-use-synthetic               Use synthetic UART input when inference is enabled\n\
        --uart-use-real                    Use real UART hardware input\n\
                --enable-ila-path                  Enable ILA CSV input path (default: on)\n\
                --disable-ila-path                 Disable ILA CSV input path\n\
                --ila-use-synthetic                Use synthetic IQ input when inference is enabled\n\
                --ila-use-real                     Use ILA CSV input when inference is enabled\n\
                --enable-inference                 Enable Python inference path (requires UART + ILA)\n\
        --disable-inference                Disable inference and run one displayed hardware path\n\
    --print-inference-results          Print inference summaries (default: on)\n\
    --no-print-inference-results       Disable inference summaries\n\
    --print-uart-input                 Print UART input data\n\
    --no-print-uart-input              Disable UART input data printing\n\
    --print-ila-input                  Print ILA CSV decode debug output\n\
    --no-print-ila-input               Disable ILA CSV decode debug output\n\
        --enable-valon                     Enable Valon LO socket output (default: on)\n\
        --disable-valon                    Disable Valon LO socket output\n\
        --valon-socket-path <path>         Valon Unix socket path (default: /tmp/valon5019.sock)\n\
        --inference-txt-path <path>        Snapshot text file path (default: ./inference_results.txt)\n\
                --enable-constellation-plot        Launch the Python constellation plotter from Rust\n\
                --disable-constellation-plot       Disable the Python constellation plotter\n\
                --plot-python-bin <path>           Python executable used for the plotter (default: python3)\n\
                --plot-script-path <path>          Plot helper script path (default: ../pluto_live_plot.py)\n\
                --plot-slot-index-path <path>      Sidecar file containing the latest SHM slot index\n\
                --plot-refresh-hz <float>          Plot refresh rate (default: 8.0)\n\
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
            "--mode" => {
                idx += 1;
                let val = args.get(idx).ok_or("Missing value for --mode")?;
                cfg.mode = match val.as_str() {
                    "hardware" => OperatingMode::Hardware,
                    "simulate" => OperatingMode::Simulate,
                    "digital_twin" | "digital-twin" => OperatingMode::DigitalTwin,
                    _ => return Err(format!("Invalid --mode: {}", val)),
                };
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
            "--rf-chain-socket-path" => {
                idx += 1;
                cfg.rf_chain_socket_path = args
                    .get(idx)
                    .ok_or("Missing value for --rf-chain-socket-path")?
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
            "--ila-csv-path" => {
                idx += 1;
                cfg.ila_csv_path = args
                    .get(idx)
                    .ok_or("Missing value for --ila-csv-path")?
                    .to_string();
            }
            "--ila-request-flag-path" => {
                idx += 1;
                cfg.ila_request_flag_path = args
                    .get(idx)
                    .ok_or("Missing value for --ila-request-flag-path")?
                    .to_string();
            }
            "--ila-poll-interval-ms" => {
                idx += 1;
                cfg.ila_poll_interval_ms = args
                    .get(idx)
                    .ok_or("Missing value for --ila-poll-interval-ms")?
                    .parse::<u64>()
                    .map_err(|_| "Invalid int for --ila-poll-interval-ms")?;
            }
            "--ila-request-timeout-ms" => {
                idx += 1;
                cfg.ila_request_timeout_ms = args
                    .get(idx)
                    .ok_or("Missing value for --ila-request-timeout-ms")?
                    .parse::<u64>()
                    .map_err(|_| "Invalid int for --ila-request-timeout-ms")?;
            }
            "--ila-batch-samples" => {
                idx += 1;
                cfg.ila_batch_samples = args
                    .get(idx)
                    .ok_or("Missing value for --ila-batch-samples")?
                    .parse::<usize>()
                    .map_err(|_| "Invalid int for --ila-batch-samples")?;
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
            "--enable-ila-path" => {
                cfg.enable_ila_path = true;
            }
            "--disable-ila-path" => {
                cfg.enable_ila_path = false;
            }
            "--ila-use-synthetic" => {
                cfg.ila_use_synthetic = true;
            }
            "--ila-use-real" => {
                cfg.ila_use_synthetic = false;
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
            "--print-ila-input" => {
                cfg.print_ila_input = true;
            }
            "--no-print-ila-input" => {
                cfg.print_ila_input = false;
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
            "--inference-txt-path" => {
                idx += 1;
                cfg.inference_txt_path = args
                    .get(idx)
                    .ok_or("Missing value for --inference-txt-path")?
                    .to_string();
            }
            "--enable-constellation-plot" => {
                cfg.plot_constellation = true;
            }
            "--disable-constellation-plot" => {
                cfg.plot_constellation = false;
            }
            "--plot-python-bin" => {
                idx += 1;
                cfg.plot_python_bin = args
                    .get(idx)
                    .ok_or("Missing value for --plot-python-bin")?
                    .to_string();
            }
            "--plot-script-path" => {
                idx += 1;
                cfg.plot_script_path = args
                    .get(idx)
                    .ok_or("Missing value for --plot-script-path")?
                    .to_string();
            }
            "--plot-slot-index-path" => {
                idx += 1;
                cfg.plot_slot_index_path = args
                    .get(idx)
                    .ok_or("Missing value for --plot-slot-index-path")?
                    .to_string();
            }
            "--plot-refresh-hz" => {
                idx += 1;
                cfg.plot_refresh_hz = args
                    .get(idx)
                    .ok_or("Missing value for --plot-refresh-hz")?
                    .parse::<f64>()
                    .map_err(|_| "Invalid float for --plot-refresh-hz")?;
            }
            other => return Err(format!("Unknown argument: {}", other)),
        }
        idx += 1;
    }

    Ok(cfg)
}

fn validate_runtime_config(cfg: &AppConfig) -> Result<(), String> {
    if cfg.dry_run || cfg.simulate {
        return Ok(());
    }

    if cfg.ila_batch_samples == 0 {
        return Err("--ila-batch-samples must be greater than zero.".to_string());
    }

    if cfg.plot_constellation && !matches!(cfg.ipc_mode, IpcMode::Shm) {
        return Err(
            "--enable-constellation-plot requires --ipc-mode shm so Rust can share IQ samples with the plotter.".to_string(),
        );
    }

    if cfg.plot_constellation && !cfg.dry_run && !cfg.simulate && !cfg.enable_inference {
        return Err(
            "--enable-constellation-plot in hardware mode requires --enable-inference so IQ samples are produced.".to_string(),
        );
    }

    if cfg.enable_inference {
        if cfg.ila_batch_samples < MIN_ILA_BATCH_SAMPLES_INFERENCE {
            return Err(
                format!(
                    "Inference mode requires --ila-batch-samples >= {} (current: {}). Recommended: 2048.",
                    MIN_ILA_BATCH_SAMPLES_INFERENCE,
                    cfg.ila_batch_samples,
                ),
            );
        }

        if !cfg.enable_uart_path || !cfg.enable_ila_path {
            return Err(
                "Inference mode requires both --enable-uart-path and --enable-ila-path. Use --uart-use-synthetic and/or --ila-use-synthetic only when a path is not real hardware.".to_string(),
            );
        }

        return Ok(());
    }

    let enabled_paths = u8::from(cfg.enable_uart_path) + u8::from(cfg.enable_ila_path);
    if enabled_paths != 1 {
        return Err(
            "No-inference mode requires exactly one enabled path: either --enable-uart-path or --enable-ila-path.".to_string(),
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

    if cfg.enable_ila_path {
        if cfg.ila_use_synthetic {
            return Err(
                "No-inference ILA mode must use real hardware, so --ila-use-synthetic is not allowed.".to_string(),
            );
        }
        if !cfg.print_ila_input {
            return Err(
                "No-inference ILA mode is useless unless --print-ila-input is enabled.".to_string(),
            );
        }
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

    match cfg.mode {
        OperatingMode::Hardware => {
            if let Err(e) = run_hardware_loop(&cfg) {
                eprintln!("Runtime failed: {}", e);
                std::process::exit(1);
            }
        }
        OperatingMode::Simulate => {
            if let Err(e) = run_simulation(&cfg) {
                eprintln!("Simulation failed: {}", e);
                std::process::exit(1);
            }
        }
        OperatingMode::DigitalTwin => {
            if let Err(e) = run_digital_twin(&cfg) {
                eprintln!("Digital twin failed: {}", e);
                std::process::exit(1);
            }
        }
    }
}

fn run_hardware_loop(cfg: &AppConfig) -> io::Result<()> {
    let mut plotter = spawn_constellation_plotter(cfg)?;

    let result = (|| -> io::Result<()> {
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
        let ila_buffer = Arc::new(Mutex::new(CircularBuffer::<receive_data::IQSample>::new(1024)));
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
                if let Err(e) = receive_data::receive_uart_adc_measurements(
                    uart,
                    uart_buf_clone,
                    print_uart_input,
                ) {
                    eprintln!("UART receiver error: {}", e);
                }
            });
            uart_reader_started = true;
        }

        let mut ila_reader_started = false;
        if cfg.enable_ila_path && (!cfg.enable_inference || !cfg.ila_use_synthetic) {
            let ila_buf_clone = Arc::clone(&ila_buffer);
            let ila_csv_path = cfg.ila_csv_path.clone();
            let ila_request_flag_path = cfg.ila_request_flag_path.clone();
            let print_ila_input = cfg.print_ila_input;
            let ila_poll_interval_ms = cfg.ila_poll_interval_ms;
            let ila_request_timeout_ms = cfg.ila_request_timeout_ms;
            let ila_batch_samples = cfg.ila_batch_samples;
            thread::spawn(move || {
                if let Err(e) = receive_data::receive_ila_csv_probe0_data(
                    &ila_csv_path,
                    &ila_request_flag_path,
                    ila_buf_clone,
                    print_ila_input,
                    ila_poll_interval_ms,
                    ila_request_timeout_ms,
                    ila_batch_samples,
                ) {
                    eprintln!("ILA CSV receiver error: {}", e);
                }
            });
            ila_reader_started = true;
        }

        println!(
            "HW mode active: uart={} ({}) ila={} ({}) inference={} uart_cmd_tx={} valon={} uart_print={} ila_print={} plot={} ipc={}",
            if cfg.enable_uart_path { "on" } else { "off" },
            if cfg.enable_inference && cfg.uart_use_synthetic {
                "synthetic"
            } else if uart_reader_started {
                "real"
            } else {
                "off"
            },
            if cfg.enable_ila_path { "on" } else { "off" },
            if cfg.enable_inference && cfg.ila_use_synthetic {
                "synthetic"
            } else if ila_reader_started {
                "real"
            } else {
                "off"
            },
            if cfg.enable_inference { "on" } else { "off" },
            if uart_command_tx.is_some() { "on" } else { "off" },
            if valon_command_tx.is_some() { "on" } else { "off" },
            if cfg.print_uart_input { "on" } else { "off" },
            if cfg.print_ila_input { "on" } else { "off" },
            if cfg.plot_constellation { "on" } else { "off" },
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
            let Some(iq_iq_pairs) = acquire_iq_samples(cfg, &ila_buffer, cycle) else {
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

            let control_values = map_agent_controls(&resp)?;
            let valon_values = map_valon_controls(&resp)?;
            let mut sent_uart_commands: Vec<String> = Vec::new();

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

            let should_poll_status = cycle % 10 == 0;

            if let Some(tx) = uart_command_tx.as_ref() {
                let commands = command_tracker.commands_for(&control_values);
                for command in commands {
                    let text = String::from_utf8_lossy(&command)
                        .trim_end_matches(['\r', '\n'])
                        .to_string();
                    sent_uart_commands.push(text.clone());

                    if let Err(e) = tx.send(command.clone()) {
                        return Err(io::Error::new(
                            io::ErrorKind::BrokenPipe,
                            format!("Failed to queue UART command for seq {}: {}", resp.seq_id, e),
                        ));
                    }
                    if cfg.print_inference_results {
                        println!("HW seq={} uart_cmd={}", resp.seq_id, text);
                        print_uart_agent_command(resp.seq_id, &command);
                    }
                }

                if should_poll_status {
                    let status_command = b"status\r\n".to_vec();
                    if let Err(e) = tx.send(status_command) {
                        return Err(io::Error::new(
                            io::ErrorKind::BrokenPipe,
                            format!(
                                "Failed to queue UART status command for seq {}: {}",
                                resp.seq_id, e
                            ),
                        ));
                    }
                    sent_uart_commands.push("status".to_string());

                    if cfg.print_inference_results {
                        println!("HW seq={} uart_cmd=status", resp.seq_id);
                    }
                }
            }

            if let Some(tx) = valon_command_tx.as_ref() {
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

            let uart_status_snapshot: Option<&str> = None;

            if let Err(e) = write_inference_snapshot_txt(
                cfg,
                &control_values,
                &valon_values,
                &sent_uart_commands,
                uart_status_snapshot.as_deref(),
            ) {
                eprintln!(
                    "Inference snapshot write warning ({}): {}",
                    cfg.inference_txt_path, e
                );
            }

            thread::sleep(std::time::Duration::from_millis(10));
        }
    })();

    stop_child_process(&mut plotter);
    result
}

fn run_dry_run(cfg: &AppConfig) -> io::Result<()> {
    let mut plotter = spawn_constellation_plotter(cfg)?;
    let result = (|| -> io::Result<()> {
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
    })();

    stop_child_process(&mut plotter);
    result
}

fn run_simulation(cfg: &AppConfig) -> io::Result<()> {
    let mut plotter = spawn_constellation_plotter(cfg)?;
    let result = (|| -> io::Result<()> {
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
    })();

    stop_child_process(&mut plotter);
    result
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
        write_plot_slot_state(cfg, seq_id, *slot_index, n_samples)?;
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

fn print_uart_agent_command(seq_id: u64, command: &[u8]) {
    let text = String::from_utf8_lossy(command);
    let trimmed = text.trim_end_matches(['\r', '\n']).trim();

    if let Some(value) = trimmed.strip_prefix("lna ") {
        println!("HW seq={} uart_lna=voltage_{}V", seq_id, value.trim());
        return;
    }

    if let Some(value) = trimmed.strip_prefix("filter ") {
        let filter_code = value.trim();
        let label = match filter_code {
            "1" => "1MHz",
            "10" => "10MHz",
            "20" => "20MHz",
            _ => "unknown",
        };
        println!(
            "HW seq={} uart_filter=code_{} ({})",
            seq_id, filter_code, label
        );
        return;
    }

    if let Some(value) = trimmed.strip_prefix("ifamp ") {
        println!("HW seq={} uart_ifamp_db={}", seq_id, value.trim());
    }
}

fn write_inference_snapshot_txt(
    cfg: &AppConfig,
    control_values: &uart_commands::AgentControlValues,
    valon_values: &valon_client::ValonControlValues,
    sent_uart_commands: &[String],
    uart_status_snapshot: Option<&str>,
) -> io::Result<()> {
    let commands_block = if sent_uart_commands.is_empty() {
        "(none)".to_string()
    } else {
        sent_uart_commands.join("\n")
    };

    let status_block = if let Some(snapshot) = uart_status_snapshot {
        snapshot
            .split(" | ")
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        "(none)".to_string()
    };

    let content = format!(
        "lna_voltage_v={}\nif_amp_gain_db={:.3}\nvalon_frequency_mhz={:.3}\nvalon_power_dbm={:.3}\nselected_filter={}\nselected_filter_uart_cmd=filter {}\nuart_commands_sent:\n{}\nuart_status_values:\n{}\n",
        control_values.lna_voltage,
        control_values.if_amp_value,
        valon_values.lo_freq_mhz,
        valon_values.rf_level_dbm,
        control_values.filter_mhz,
        control_values.filter_mhz,
        commands_block,
        status_block,
    );
    fs::write(&cfg.inference_txt_path, content)
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

fn write_text_atomic(path: &str, content: &str) -> io::Result<()> {
    let tmp_path = format!("{}.tmp", path);
    fs::write(&tmp_path, content)?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

fn write_plot_slot_state(
    cfg: &AppConfig,
    seq_id: u64,
    slot_index: usize,
    n_samples: usize,
) -> io::Result<()> {
    if !cfg.plot_constellation {
        return Ok(());
    }

    let content = format!(
        "seq_id={}\nslot_index={}\nn_samples={}\n",
        seq_id, slot_index, n_samples
    );
    write_text_atomic(&cfg.plot_slot_index_path, &content)
}

fn spawn_constellation_plotter(cfg: &AppConfig) -> io::Result<Option<Child>> {
    if !cfg.plot_constellation {
        return Ok(None);
    }

    if !matches!(cfg.ipc_mode, IpcMode::Shm) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Constellation plotter requires SHM IPC mode",
        ));
    }

    let mut cmd = Command::new(&cfg.plot_python_bin);
    cmd.arg(&cfg.plot_script_path)
        .arg("--shm-name")
        .arg(&cfg.shm_name)
        .arg("--shm-slots")
        .arg(cfg.shm_slots.to_string())
        .arg("--shm-slot-capacity")
        .arg(cfg.shm_slot_capacity.to_string())
        .arg("--sample-rate-hz")
        .arg(cfg.sample_rate_hz.to_string())
        .arg("--slot-index-path")
        .arg(&cfg.plot_slot_index_path)
        .arg("--refresh-hz")
        .arg(cfg.plot_refresh_hz.to_string())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .stdin(Stdio::null());

    println!(
        "[launcher] starting constellation plotter: {} {}",
        cfg.plot_python_bin, cfg.plot_script_path
    );
    let child = cmd.spawn()?;
    Ok(Some(child))
}

fn stop_child_process(child: &mut Option<Child>) {
    if let Some(mut process) = child.take() {
        let _ = process.kill();
        let _ = process.wait();
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
    ila_buffer: &Arc<Mutex<CircularBuffer<receive_data::IQSample>>>,
    cycle: u64,
) -> Option<Vec<(f32, f32)>> {
    if !cfg.enable_ila_path {
        return None;
    }

    if cfg.enable_inference && cfg.ila_use_synthetic {
        return Some(generate_synthetic_iq(cfg.dry_run_samples, cycle as f32 * 0.1));
    }

    if let Some(sample) = ila_buffer.lock().unwrap().read() {
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
fn run_digital_twin(cfg: &AppConfig) -> io::Result<()> {
    println!("[digital_twin] Starting digital twin mode");
    println!("[digital_twin] RF Chain Socket: {}", cfg.rf_chain_socket_path);

    // Connect to RF chain worker
    let mut rfchain_client = RFChainSocketClient::connect(&cfg.rf_chain_socket_path)?;
    println!("[digital_twin] Connected to RF chain worker");

    // Connect to inference worker if enabled
    let mut inference_client = if cfg.enable_inference {
        match InferenceSocketClient::new(&cfg.socket_path).connect() {
            Ok(_) => {
                println!("[digital_twin] Connected to inference worker");
                Some(InferenceSocketClient::new(&cfg.socket_path))
            }
            Err(e) => {
                eprintln!("[digital_twin] Warning: Could not connect to inference worker: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Digital twin simulation parameters (default values, could be made configurable)
    let mut seq_id = 0u64;
    let _start_time = std::time::Instant::now();
    let max_cycles = if cfg.rf_chain_cycles > 0 {
        cfg.rf_chain_cycles
    } else {
        0  // Run forever if 0
    };

    loop {
        let cycle = seq_id;
        if max_cycles > 0 && cycle >= max_cycles {
            println!("[digital_twin] Completed {} cycles", max_cycles);
            break;
        }

        // Vary parameters over time for demonstration
        let power_pre_lna = -40.0 + (cycle as f32 % 20.0);
        let bandwidth_hz = match (cycle / 20) % 3 {
            0 => 1e6,
            1 => 10e6,
            _ => 20e6,
        };
        let lna_voltage = if (cycle / 60) % 2 == 0 { 3.0 } else { 5.0 };
        let lo_power = -10.0 + ((cycle as f32 % 30.0) - 15.0);
        let pa_gain = 5.0 + ((cycle as f32 % 21.0) - 10.5);

        seq_id += 1;

        // Call RF chain worker
        match rfchain_client.process_signal(
            power_pre_lna,
            bandwidth_hz,
            2420e6,  // Center frequency (fixed for now)
            lna_voltage,
            lo_power,
            pa_gain,
        ) {
            Ok(rf_result) => {
                if cfg.print_inference_results {
                    println!(
                        "[digital_twin] Seq:{} EVM:{:.2}% Power(pre):{:.2}dBm Power(post):{:.2}dBm Time:{:.2}ms",
                        seq_id, rf_result.evm_percent, rf_result.power_pre_lna_dbm,
                        rf_result.power_post_pa_dbm, rf_result.processing_time_ms
                    );
                }

                // Optionally send to inference worker for agent recommendations
                if let Some(ref mut inf_client) = inference_client {
                    // Create I/Q complex samples from RF result
                    let iq_pairs: Vec<(f32, f32)> = rf_result.i_samples.iter()
                        .zip(rf_result.q_samples.iter())
                        .map(|(i, q)| (*i, *q))
                        .collect();

                    let req = InferenceRequest {
                        seq_id,
                        sample_rate_hz: cfg.sample_rate_hz,
                        power_lna_dbm: rf_result.power_pre_lna_dbm,
                        power_pa_dbm: rf_result.power_post_pa_dbm,
                        iq_iq_pairs: &iq_pairs,
                    };

                    match inf_client.infer(&req) {
                        Ok(result) => {
                            if cfg.print_inference_results {
                                println!(
                                    "  [agent] LNA:{} Filter:{} Mixer:{:.2}dBm IFAmp:{:.2}dB EVM:{:.2}%",
                                    result.lna_class, result.filter_class,
                                    result.mixer_dbm, result.ifamp_db, result.evm_value
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!("[digital_twin] Inference error: {}", e);
                        }
                    }
                }

                // Write results to file for telemetry
                let _ = fs::write(
                    &cfg.inference_txt_path,
                    format!(
                        "seq_id: {}\nstatus: {}\nevm: {:.2}\npower_pre_lna: {:.2}\npower_post_pa: {:.2}\n",
                        seq_id, rf_result.status, rf_result.evm_percent,
                        rf_result.power_pre_lna_dbm, rf_result.power_post_pa_dbm
                    ),
                );
            }
            Err(e) => {
                eprintln!("[digital_twin] Error processing RF chain: {}", e);
            }
        }

        // Sleep between iterations
        thread::sleep(std::time::Duration::from_millis(cfg.rf_chain_interval_ms));
    }

    println!("[digital_twin] Digital twin completed after {} cycles", seq_id);
    Ok(())
}
