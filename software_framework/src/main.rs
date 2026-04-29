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
    digital_twin_params_path: String,
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
            sample_rate_hz: 125_000_000.0,
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
            digital_twin_params_path: std::env::var("MAARS_DIGITAL_TWIN_PARAMS_PATH")
                .unwrap_or_else(|_| "/tmp/maars_digital_twin_params.txt".to_string()),
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
        "software-framework\n\n\
Usage:\n\
  software-framework [options]\n\n\
Options:\n\
  --mode <hardware|simulate|digital_twin>  Run mode (default: hardware)\n\
  --ipc-mode <direct|shm>            IPC mode for inference (default: direct)\n\
  --socket-path <path>               Inference worker socket (default: /tmp/maars_infer.sock)\n\
  --rf-chain-socket-path <path>      RF chain worker socket (default: /tmp/maars_rfchain.sock)\n\
  --sample-rate-hz <float>           Sample rate in Hz (default: 125,000,000)\n\n\
  --shm-name <name>                  SHM segment name (default: maars_iq_ring)\n\
  --shm-slots <int>                  SHM slot count (default: 8)\n\
  --shm-slot-capacity <int>          SHM slot capacity (default: 8192)\n\n\
  --dry-run                          Run a single evaluation cycle on synthetic data and exit\n\
  --simulate                         Run continuous evaluation cycles on synthetic data\n\
  --dry-run-cycles <int>             Evaluation cycles for dry-run (default: 1)\n\
  --simulate-cycles <int>            Cycles for simulate mode (0 = forever, default: 0)\n\
  --simulate-interval-ms <int>       Delay between cycles in ms (default: 200)\n\
  --dry-run-samples <int>            Samples per evaluation (default: 4096)\n\
  --dry-run-power-lna <float>        Synthetic LNA power in dBm (default: -35.0)\n\
  --dry-run-power-pa <float>         Synthetic PA power in dBm (default: -22.0)\n\n\
  --uart-port <path>                 UART device (default: /dev/cu.usbmodem11203)\n\
  --uart-baud <int>                  UART baud rate (default: 115200)\n\
  --ila-csv-path <path>              Path to ILA capture CSV (default: ./ila_probe0.csv)\n\
  --ila-request-flag-path <path>     Flag file for ILA capture (default: ./ila_capture_request.txt)\n\
  --ila-poll-interval-ms <int>       ILA polling interval (default: 20)\n\
  --ila-request-timeout-ms <int>     ILA capture timeout (default: 5000)\n\
  --ila-batch-samples <int>          Rows to read from ILA CSV per cycle (default: 2048)\n\n\
  --enable-uart-path                 Enable UART input path (default: on)\n\
  --disable-uart-path                Disable UART input path\n\
  --uart-use-synthetic               Use synthetic UART input when inference is enabled\n\
  --uart-use-real                    Use real UART hardware input\n\n\
  --enable-ila-path                  Enable ILA CSV input path (default: on)\n\
  --disable-ila-path                 Disable ILA CSV input path\n\
  --ila-use-synthetic                Use synthetic IQ input when inference is enabled\n\
  --ila-use-real                     Use ILA CSV input when inference is enabled\n\n\
  --enable-inference                 Enable Python inference path (requires UART + ILA)\n\
  --disable-inference                Disable inference and run one displayed hardware path\n\n\
  --print-inference-results          Print inference summaries (default: on)\n\
  --no-print-inference-results       Disable inference summaries\n\n\
  --print-uart-input                 Print UART input data\n\
  --no-print-uart-input              Disable UART input data printing\n\n\
  --print-ila-input                  Print ILA CSV decode debug output\n\
  --no-print-ila-input               Disable ILA CSV decode debug output\n\n\
  --enable-valon                     Enable Valon LO socket output (default: on)\n\
  --disable-valon                    Disable Valon LO socket output\n\n\
  --valon-socket-path <path>         Valon Unix socket path (default: /tmp/valon5019.sock)\n\
  --inference-txt-path <path>        Snapshot text file path (default: ./inference_results.txt)\n\
  --digital-twin-params-path <path>  External parameters file path (default: /tmp/maars_digital_twin_params.txt)\n\
  --rf-chain-cycles <int>            Digital twin RF chain cycles (0 = run continuously, default: 0)\n\
  --rf-chain-interval-ms <int>       Delay between RF chain calls in ms (default: 100)\n\n\
  --enable-constellation-plot        Launch the Python constellation plotter from Rust\n\
  --disable-constellation-plot       Disable the Python constellation plotter\n\n\
  --plot-python-bin <path>           Python executable used for the plotter (default: python3)\n\
  --plot-script-path <path>          Plot helper script path (default: ../pluto_live_plot.py)\n\
  --plot-slot-index-path <path>      Sidecar file containing the latest SHM slot index\n\
  --plot-refresh-hz <float>          Plot refresh rate (default: 8.0)\n\n\
  --cleanup-shm-on-exit              Explicitly unlink SHM segment at simulation/dry-run teardown\n\n\
  --help                             Show this help\n"
    );
}

fn parse_args() -> Result<AppConfig, String> {
    let mut cfg = AppConfig::default();
    let args: Vec<String> = std::env::args().collect();
    let mut idx = 1;

    while idx < args.len() {
        match args[idx].as_str() {
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
            "--digital-twin-params-path" => {
                idx += 1;
                cfg.digital_twin_params_path = args
                    .get(idx)
                    .ok_or("Missing value for --digital-twin-params-path")?
                    .to_string();
            }
            "--rf-chain-cycles" => {
                idx += 1;
                cfg.rf_chain_cycles = args
                    .get(idx)
                    .ok_or("Missing value for --rf-chain-cycles")?
                    .parse::<u64>()
                    .map_err(|_| "Invalid int for --rf-chain-cycles")?;
            }
            "--rf-chain-interval-ms" => {
                idx += 1;
                cfg.rf_chain_interval_ms = args
                    .get(idx)
                    .ok_or("Missing value for --rf-chain-interval-ms")?
                    .parse::<u64>()
                    .map_err(|_| "Invalid int for --rf-chain-interval-ms")?;
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
    if cfg.dry_run || cfg.simulate || matches!(cfg.mode, OperatingMode::DigitalTwin) {
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
            match cfg.ipc_mode {
                IpcMode::Shm => Some(SharedMemoryRingBuffer::attach(SharedMemoryRingSpec {
                    name: cfg.shm_name.clone(),
                    num_slots: cfg.shm_slots,
                    slot_capacity: cfg.shm_slot_capacity,
                })?),
                IpcMode::Direct => None,
            }
        } else {
            None
        };

        let mut uart_command_tx: Option<mpsc::Sender<Vec<u8>>> = None;
        let mut valon_command_tx: Option<mpsc::Sender<valon_client::ValonCommand>> = None;

        if cfg.enable_uart_path {
            let _uart_cfg = UartConfig {
                port: cfg.uart_port.clone(),
                baud_rate: cfg.uart_baud,
                timeout_ms: 100,
            };
            let (tx, rx) = mpsc::channel();
            uart_command_tx = Some(tx);
            thread::spawn(move || {
                // The uart module doesn't have a worker loop yet, 
                // but we can implement a simple one here or in uart.rs if needed.
                // For now, we'll just drain the receiver to avoid blocking.
                while let Ok(_cmd) = rx.recv() {
                    // Actual UART writing would happen here if implemented
                }
            });
        }

        if cfg.enable_valon {
            let (tx, rx) = mpsc::channel();
            valon_command_tx = Some(tx);
            let socket_path = cfg.valon_socket_path.clone();
            let print_logs = cfg.print_inference_results;
            thread::spawn(move || {
                if let Err(e) = send_valon_commands(&socket_path, rx, print_logs) {
                    eprintln!("Valon worker error: {}", e);
                }
            });
        }

        let mut cycle: u64 = 0;
        let mut slot_index = 0usize;
        let mut command_tracker = CommandTracker::default();
        let mut valon_tracker = ValonCommandTracker::default();

        let uart_buffer = CircularBuffer::new(4096);
        let ila_buffer = CircularBuffer::new(cfg.ila_batch_samples * 2);

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

            let commands_block = sent_uart_commands.join("\n");
            let status_block = String::new(); // Placeholder
            let _ = write_inference_snapshot(
                cfg,
                &resp,
                &control_values,
                &valon_values,
                power_lna_dbm,
                power_pa_dbm,
                &commands_block,
                &status_block,
            );

            thread::sleep(std::time::Duration::from_millis(10));
        }
    })();

    stop_child_process(&mut plotter);
    result
}

fn acquire_power_sample(
    _cfg: &AppConfig,
    _buffer: &CircularBuffer<PowerMeasurement>,
    _cycle: u64,
) -> Option<PowerMeasurement> {
    Some(PowerMeasurement {
        power_lna_raw: 1500.0,
        power_pa_raw: 2500.0,
    })
}

fn acquire_iq_samples(
    _cfg: &AppConfig,
    _buffer: &CircularBuffer<f32>,
    _cycle: u64,
) -> Option<Vec<(f32, f32)>> {
    let mut samples = Vec::with_capacity(2048);
    for i in 0..2048 {
        let t = i as f32 * 0.1;
        samples.push((t.sin(), t.cos()));
    }
    Some(samples)
}

fn calibrate_power_raw_to_dbm(raw: f32) -> f32 {
    let volts = (raw / ADC_MAX_U12) * ADC_VREF_VOLTS;
    let dbm = SENSOR_MIN_DBM
        + ((volts - SENSOR_MIN_VOLTS) / (SENSOR_MAX_VOLTS - SENSOR_MIN_VOLTS))
            * (SENSOR_MAX_DBM - SENSOR_MIN_DBM);
    dbm + SENSOR_DBM_BIAS
}

fn run_digital_twin(cfg: &AppConfig) -> io::Result<()> {
    println!("[digital_twin] Starting digital twin mode");
    println!("[digital_twin] RF Chain Socket: {}", cfg.rf_chain_socket_path);

    let mut rfchain_client = RFChainSocketClient::connect(&cfg.rf_chain_socket_path)?;
    println!("[digital_twin] Connected to RF chain worker");

    let mut inference_client = if cfg.enable_inference {
        let mut client = InferenceSocketClient::new(&cfg.socket_path);
        match client.connect() {
            Ok(_) => {
                println!("[digital_twin] Connected to inference worker");
                Some(client)
            }
            Err(e) => {
                eprintln!("[digital_twin] Warning: Could not connect to inference worker: {}", e);
                None
            }
        }
    } else {
        None
    };

    let mut seq_id = 0u64;
    let _start_time = std::time::Instant::now();
    let max_cycles = if cfg.rf_chain_cycles > 0 {
        cfg.rf_chain_cycles
    } else {
        0
    };

    loop {
        let cycle = seq_id;
        if max_cycles > 0 && cycle >= max_cycles {
            println!("[digital_twin] Completed {} cycles", max_cycles);
            break;
        }

        let mut power_pre_lna = -40.0 + (cycle as f32 % 20.0);
        let mut bandwidth_hz = match (cycle / 20) % 3 {
            0 => 1e6,
            1 => 10e6,
            _ => 20e6,
        };
        let mut center_freq_hz = match (cycle / 40) % 3 {
            0 => 2405e6,
            1 => 2420e6,
            _ => 2435e6,
        };
        let mut lna_voltage = if (cycle / 60) % 2 == 0 { 3.0 } else { 5.0 };
        let mut lo_power = -10.0 + ((cycle as f32 % 30.0) - 15.0);
        let mut pa_gain = 5.0 + ((cycle as f32 % 21.0) - 10.5);
        let mut manual_mode = 0;

        if let Ok(params_content) = fs::read_to_string(&cfg.digital_twin_params_path) {
            let mut overrides = std::collections::HashMap::new();
            for line in params_content.lines() {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() == 2 {
                    overrides.insert(parts[0].trim().to_string(), parts[1].trim().to_string());
                }
            }
            if let Some(m) = overrides.get("manual_mode") {
                if let Ok(v) = m.parse::<i32>() { manual_mode = v; }
            }
            if manual_mode == 1 {
                if let Some(v) = overrides.get("power_pre_lna_dbm") { if let Ok(f) = v.parse::<f32>() { power_pre_lna = f; } }
                if let Some(v) = overrides.get("bandwidth_hz") { if let Ok(f) = v.parse::<f32>() { bandwidth_hz = f; } }
                if let Some(v) = overrides.get("center_freq_hz") { if let Ok(f) = v.parse::<f32>() { center_freq_hz = f; } }
                if let Some(v) = overrides.get("lna_voltage") { if let Ok(f) = v.parse::<f32>() { lna_voltage = f; } }
                if let Some(v) = overrides.get("lo_power_dbm") { if let Ok(f) = v.parse::<f32>() { lo_power = f; } }
                if let Some(v) = overrides.get("pa_gain_db") { if let Ok(f) = v.parse::<f32>() { pa_gain = f; } }
            }
        }

        // In autonomous mode (manual_mode == 0), fix the LO at 2395 MHz (for 2420 center target)
        // so the AI can detect shifts as the RF signal sweeps.
        // We also use a wide (60 MHz) filter so the AI can "see" the whole band.
        // In manual mode, we follow user's desired tuning exactly.
        let target_center_mhz = if manual_mode == 1 { center_freq_hz / 1e6 } else { 2420.0 };
        let evaluation_lo_freq_hz = (target_center_mhz - 25.0) * 1e6;
        let evaluation_lo_power_dbm = if manual_mode == 1 { lo_power } else { 0.0 };
        let evaluation_bandwidth_hz = if manual_mode == 1 { bandwidth_hz } else { 60.0e6 };

        seq_id += 1;

        // Call RF chain worker
        match rfchain_client.process_signal(
            power_pre_lna,
            evaluation_bandwidth_hz,
            center_freq_hz,
            evaluation_lo_freq_hz,
            lna_voltage,
            evaluation_lo_power_dbm,
            pa_gain,
        ) {
            Ok(rf_result) => {
                if cfg.print_inference_results {
                    println!(
                        "[digital_twin] Seq:{} Mode:{} BW:{:.1}MHz RF:{:.1}MHz Power(pre):{:.1}dBm EVM:{:.1}%",
                        seq_id,
                        if manual_mode == 1 { "MANUAL" } else { "SWEEP" },
                        bandwidth_hz / 1e6,
                        center_freq_hz / 1e6,
                        power_pre_lna,
                        rf_result.evm_percent
                    );
                }

                if cfg.enable_inference && inference_client.is_none() && (seq_id % 20 == 0) {
                    let mut client = InferenceSocketClient::new(&cfg.socket_path);
                    if client.connect().is_ok() {
                        println!("[digital_twin] Reconnected to inference worker");
                        inference_client = Some(client);
                    }
                }

                let default_filter_class = if bandwidth_hz <= 5.0e6 { 0 } else if bandwidth_hz <= 15.0e6 { 1 } else { 2 };
                let default_center_class = if center_freq_hz <= 2410e6 { 0 } else if center_freq_hz <= 2425e6 { 1 } else { 2 };

                let mut status_code: i32 = 0;
                let mut lna_class: u8 = if lna_voltage >= 4.0 { 1 } else { 0 };
                let mut filter_class: u8 = default_filter_class;
                let mut center_class: u8 = default_center_class;
                let mut mixer_dbm: f32 = lo_power;
                let mut ifamp_db: f32 = pa_gain;
                let mut agent_evm_value: f32 = rf_result.evm_percent;
                let mut lna_voltage_v: f32 = lna_voltage;
                let mut selected_filter_mhz: f32 = bandwidth_hz / 1e6;
                let mut lo_center_mhz: f32 = center_freq_hz / 1e6;
                let mut inference_ok = false;

                if let Some(ref mut inf_client) = inference_client {
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
                            inference_ok = true;
                            status_code = result.status_code;
                            lna_class = result.lna_class;
                            mixer_dbm = result.mixer_dbm;
                            ifamp_db = result.ifamp_db;
                            agent_evm_value = result.evm_value;
                            lna_voltage_v = match result.lna_class { 0 => 3.0, 1 => 5.0, _ => 3.0 };

                            // Keep dashboard telemetry aligned with commanded RF settings in digital twin.
                            // The AI center classifier operates on IF-centered spectra and can remain at class=1
                            // when LO tracks RF center (manual mode), so use commanded RF center/bandwidth here.
                            filter_class = default_filter_class;
                            center_class = default_center_class;
                            selected_filter_mhz = bandwidth_hz / 1e6;
                            lo_center_mhz = center_freq_hz / 1e6;
                        }
                        Err(e) => {
                            eprintln!("[digital_twin] Inference error: {}", e);
                            inference_client = None;
                            status_code = 1;
                        }
                    }
                } else {
                    status_code = 1;
                }

                let snapshot = format!(
                    "seq_id={}\nstatus={}\nsource=digital_twin\ninference_ok={}\nmanual_mode={}\nstatus_code={}\nlna_class={}\nlna_voltage_v={:.1}\nfilter_class={}\nselected_filter_mhz={:.1}\nselected_filter_label={} MHz\ncenter_class={}\nlo_center_mhz={:.1}\nlo_power_dbm={:.3}\nmixer_dbm={:.3}\nifamp_db={:.3}\nevm_value={:.3}\nevm_percent={:.3}\nagent_evm_value={:.3}\npower_lna_dbm={:.3}\npower_pre_lna_dbm={:.3}\npower_pa_dbm={:.3}\npower_post_pa_dbm={:.3}\npower_lna_raw={:.3}\npower_pa_raw={:.3}\nprocessing_time_ms={:.3}\n",
                    seq_id,
                    rf_result.status,
                    if inference_ok { 1 } else { 0 },
                    manual_mode,
                    status_code,
                    lna_class,
                    lna_voltage_v,
                    filter_class,
                    selected_filter_mhz,
                    selected_filter_mhz as i32,
                    center_class,
                    lo_center_mhz,
                    evaluation_lo_power_dbm,
                    mixer_dbm,
                    ifamp_db,
                    rf_result.evm_percent,
                    rf_result.evm_percent,
                    agent_evm_value,
                    rf_result.power_pre_lna_dbm,
                    rf_result.power_pre_lna_dbm,
                    rf_result.power_post_pa_dbm,
                    rf_result.power_post_pa_dbm,
                    rf_result.power_pre_lna_dbm,
                    rf_result.power_post_pa_dbm,
                    rf_result.processing_time_ms,
                );

                let _ = write_text_atomic(&cfg.inference_txt_path, &snapshot);
            }
            Err(e) => {
                eprintln!("[digital_twin] Error processing RF chain: {}", e);
            }
        }
        thread::sleep(std::time::Duration::from_millis(50));
    }
    println!("[digital_twin] Digital twin completed after {} cycles", seq_id);
    Ok(())
}

fn write_inference_snapshot(
    cfg: &AppConfig,
    resp: &InferenceResponse,
    control_values: &uart_commands::AgentControlValues,
    valon_values: &valon_client::ValonControlValues,
    power_lna_dbm: f32,
    power_pa_dbm: f32,
    commands_block: &str,
    status_block: &str,
) -> io::Result<()> {
    let content = format!(
        "seq_id={}\nstatus=ok\nsource=hardware\nstatus_code={}\nlna_class={}\nlna_voltage_v={}\nfilter_class={}\nselected_filter_mhz={:.3}\nselected_filter_label={} MHz\ncenter_class={}\nlo_center_mhz={:.3}\nlo_power_dbm={:.3}\nmixer_dbm={:.3}\nifamp_db={:.3}\nevm_value={:.3}\nevm_percent={:.3}\nprocessing_time_ms={:.3}\npower_lna_dbm={:.3}\npower_pre_lna_dbm={:.3}\npower_pa_dbm={:.3}\npower_post_pa_dbm={:.3}\npower_lna_raw={:.3}\npower_pa_raw={:.3}\nselected_filter_uart_cmd=filter {}\nuart_commands_sent:\n{}\nuart_status_values:\n{}\n",
        resp.seq_id,
        resp.status_code,
        resp.lna_class,
        control_values.lna_voltage,
        resp.filter_class,
        control_values.filter_mhz,
        control_values.filter_mhz,
        resp.center_class,
        valon_values.lo_freq_mhz,
        valon_values.rf_level_dbm,
        resp.mixer_dbm,
        control_values.if_amp_value,
        resp.evm_value,
        resp.evm_value,
        resp.processing_time_ms,
        power_lna_dbm,
        power_lna_dbm,
        power_pa_dbm,
        power_pa_dbm,
        power_lna_dbm,
        power_pa_dbm,
        control_values.filter_mhz,
        commands_block,
        status_block,
    );
    fs::write(&cfg.inference_txt_path, content)
}

fn maybe_cleanup_shm(cfg: &AppConfig) {
    if !cfg.cleanup_shm_on_exit { return; }
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

fn append_to_file(path: &str, line: &str) -> io::Result<()> {
    use std::fs::OpenOptions;
    use std::io::Write;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{}", line)
}

fn generate_synthetic_iq(n_samples: usize, phase_offset: f32) -> Vec<(f32, f32)> {
    let mut samples = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = i as f32 * 0.1 + phase_offset;
        samples.push((t.sin(), t.cos()));
    }
    samples
}

fn write_plot_slot_state(cfg: &AppConfig, seq_id: u64, slot_index: usize, n_samples: usize) -> io::Result<()> {
    let content = format!("seq_id={}\nslot_index={}\nn_samples={}\n", seq_id, slot_index, n_samples);
    fs::write(&cfg.plot_slot_index_path, content)
}

fn stop_child_process(child: &mut Option<Child>) {
    if let Some(mut c) = child.take() {
        let _ = c.kill();
        let _ = c.wait();
    }
}

fn spawn_constellation_plotter(cfg: &AppConfig) -> io::Result<Option<Child>> {
    if !cfg.plot_constellation { return Ok(None); }
    let child = Command::new(&cfg.plot_python_bin)
        .arg(&cfg.plot_script_path)
        .arg("--shm-name").arg(&cfg.shm_name)
        .arg("--shm-slots").arg(cfg.shm_slots.to_string())
        .arg("--shm-slot-capacity").arg(cfg.shm_slot_capacity.to_string())
        .arg("--slot-index-path").arg(&cfg.plot_slot_index_path)
        .arg("--refresh-hz").arg(cfg.plot_refresh_hz.to_string())
        .spawn()?;
    Ok(Some(child))
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
        } else { None };
        let mut slot_index = 0usize;
        let mut command_tracker = CommandTracker::default();
        let mut valon_tracker = ValonCommandTracker::default();
        let mut cycle: u64 = 0;
        loop {
            cycle += 1;
            if cfg.simulate_cycles > 0 && cycle > cfg.simulate_cycles { break; }
            let phase = cycle as f32 * 0.07;
            let iq_iq_pairs = generate_synthetic_iq(cfg.dry_run_samples, phase);
            let power_lna = cfg.dry_run_power_lna_dbm + (phase.sin() * 0.8);
            let power_pa = cfg.dry_run_power_pa_dbm + (phase.cos() * 0.8);
            let resp = infer_once(cfg, &mut inference_client, &mut shm_ring, &mut slot_index, cycle, &iq_iq_pairs, power_lna, power_pa)?;
            let control_values = map_agent_controls(&resp)?;
            let _valon_values = map_valon_controls(&resp)?;
            thread::sleep(std::time::Duration::from_millis(cfg.simulate_interval_ms));
        }
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
        *slot_index = if ring.num_slots() > 0 { (*slot_index + 1) % ring.num_slots() } else { 0 };
        inference_client.infer_shm(&req)
    } else {
        let req = InferenceRequest { seq_id, sample_rate_hz: cfg.sample_rate_hz, power_lna_dbm, power_pa_dbm, iq_iq_pairs };
        inference_client.infer(&req)
    }
}

fn print_uart_agent_command(seq_id: u64, command: &[u8]) {
    let text = String::from_utf8_lossy(command);
    let trimmed = text.trim_end_matches(['\r', '\n']).trim();
    if let Some(value) = trimmed.strip_prefix("lna ") {
        println!("HW seq={} uart_lna=voltage_{}V", seq_id, value.trim());
    }
}
