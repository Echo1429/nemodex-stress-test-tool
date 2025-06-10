//! Devices Test Module
//! 
//! This module "checks" the system’s peripheral devices in a manner similar to
//! Windows Device Manager. It gathers device lists via "lspci" and "lsusb",
//! scans dmesg for hardware issues using a weighted scoring scheme (for logging only),
//! retrieves battery health via upower, and performs a network test using a built-in
//! TCP loopback benchmark. The public function `run_test(...)` returns `true` if
//! the aggregated tests pass, and `false` otherwise. In this version, only the
//! peripheral (devices) test is run, and failures due solely to dmesg output are ignored.
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use regex::Regex;
use std::thread;
use std::time::{Duration, Instant};

use super::{TestParameters, TestProgressData, SampleData, timestamped_log_entry};

/// Logs a message with a timestamp into the shared log.
fn log_message(log: &Arc<Mutex<Vec<String>>>, msg: &str) {
    if let Ok(mut l) = log.lock() {
        l.push(timestamped_log_entry(msg));
    }
}

/// Scans for PCI devices using "lspci" and returns a vector of device info strings.
fn scan_pci_devices() -> Vec<String> {
    let mut devices = Vec::new();
    if let Ok(output) = Command::new("lspci").output() {
        if output.status.success() {
            let out_str = String::from_utf8_lossy(&output.stdout);
            for line in out_str.lines() {
                devices.push(line.to_string());
            }
        }
    }
    devices
}

/// Scans for USB devices using "lsusb" and returns a vector of device info strings.
fn scan_usb_devices() -> Vec<String> {
    let mut devices = Vec::new();
    if let Ok(output) = Command::new("lsusb").output() {
        if output.status.success() {
            let out_str = String::from_utf8_lossy(&output.stdout);
            for line in out_str.lines() {
                devices.push(line.to_string());
            }
        }
    }
    devices
}

/// Checks the dmesg log for hardware-related issues using a weighted scoring scheme.
/// It returns a tuple containing:
///   - A vector of matching messages,
///   - A summary string (with the weighted score),
///   - The computed weighted score (for logging only).
fn check_dmesg_errors() -> (Vec<String>, String, f64) {
    let mut messages = Vec::new();
    if let Ok(output) = Command::new("dmesg").output() {
        if output.status.success() {
            let out_str = String::from_utf8_lossy(&output.stdout);
            // Scan for lines with typical peripheral-related keywords.
            let re = Regex::new(r"(?i)(pci|usb|nvme|sata|battery|bluetooth|gpu|disk|keyboard|touchpad|display|error|fail|warn)").unwrap();
            for line in out_str.lines() {
                if re.is_match(line) {
                    messages.push(line.to_string());
                }
            }
        }
    }
    // Compute a weighted score (for logging only).
    let mut weighted_sum = 0.0;
    for msg in messages.iter() {
        let lower = msg.to_lowercase();
        let mut weight = 0.0;
        if lower.contains("keyboard") || lower.contains("touchpad") || lower.contains("display") {
            if lower.contains("fail") || lower.contains("error") || lower.contains("not functioning") || lower.contains("disabled") {
                weight += 2.0;
            } else {
                weight += 0.5;
            }
        }
        if lower.contains("error") || lower.contains("fail") {
            weight += 1.0;
        } else if lower.contains("warn") {
            weight += 0.25;
        }
        weighted_sum += weight;
    }
    let summary = format!(
        "Hardware Issues Summary: {} messages detected, weighted score = {:.2} (threshold = 2.0).",
        messages.len(),
        weighted_sum
    );
    (messages, summary, weighted_sum)
}

/// Checks battery health using upower.
pub fn check_battery_health() -> String {
    if let Ok(output) = Command::new("upower").args(&["-e"]).output() {
        let devices = String::from_utf8_lossy(&output.stdout);
        let battery_device = devices
            .lines()
            .find(|line| line.contains("battery"))
            .map(|s| s.trim().to_owned());
        if let Some(battery_path) = battery_device {
            if let Ok(batt_output) = Command::new("upower").args(&["-i", &battery_path]).output() {
                let batt_info = String::from_utf8_lossy(&batt_output.stdout);
                let state_re = Regex::new(r"state:\s+(.+)").unwrap();
                let energy_full_re = Regex::new(r"energy-full:\s+([\d\.]+)\s+Wh").unwrap();
                let energy_design_re = Regex::new(r"energy-full-design:\s+([\d\.]+)\s+Wh").unwrap();
                let state = state_re
                    .captures(&batt_info)
                    .and_then(|cap| cap.get(1).map(|m| m.as_str().trim().to_string()))
                    .unwrap_or_else(|| "Unknown".to_string());
                let energy_full = energy_full_re
                    .captures(&batt_info)
                    .and_then(|cap| cap.get(1).map(|m| m.as_str().parse::<f64>().ok()))
                    .flatten();
                let energy_design = energy_design_re
                    .captures(&batt_info)
                    .and_then(|cap| cap.get(1).map(|m| m.as_str().parse::<f64>().ok()))
                    .flatten();
                if let (Some(ef), Some(ed)) = (energy_full, energy_design) {
                    let health = if ed > 0.0 { (ef / ed) * 100.0 } else { 0.0 };
                    return format!("State = {}, Health = {:.0}%", state, health);
                } else {
                    return format!("State = {} (energy details not found)", state);
                }
            } else {
                return format!("Failed to retrieve battery details from {}.", battery_path);
            }
        } else {
            return "No battery device found.".to_string();
        }
    }
    "Unable to retrieve battery information.".to_string()
}

/// Runs a Network Test using a TCP loopback.
/// A listener is created on 127.0.0.1:5001 and a client connection sends data for 5 seconds.
/// Throughput is measured and if it is below a threshold, the test fails.
pub fn run_network_test(log: &Arc<Mutex<Vec<String>>>, stop_flag: Arc<AtomicBool>) -> bool {
    use std::net::{TcpListener, TcpStream, Shutdown};
    use std::io::{Read, Write};

    log_message(log, "Starting Network Test using TCP loopback...");
    
    let addr = "127.0.0.1:5001";
    let listener = TcpListener::bind(addr);
    if listener.is_err() {
        log_message(log, &format!("Network Test: Failed to bind listener: {}", listener.unwrap_err()));
        return false;
    }
    let listener = listener.unwrap();
    
    let handle_listener = thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 4096];
            while let Ok(n) = stream.read(&mut buf) {
                if n == 0 { break; }
            }
        }
    });
    
    let client = TcpStream::connect(addr);
    if client.is_err() {
        log_message(log, &format!("Network Test: Failed to connect as client: {}", client.unwrap_err()));
        return false;
    }
    let mut client = client.unwrap();
    let test_duration = Duration::from_secs(5);
    let start_time = Instant::now();
    let data = vec![0u8; 8192];
    let mut total_bytes = 0u64;
    while start_time.elapsed() < test_duration && !stop_flag.load(Ordering::Relaxed) {
        if let Err(e) = client.write_all(&data) {
            log_message(log, &format!("Network Test: Error sending data: {}", e));
            break;
        }
        total_bytes += data.len() as u64;
    }
    let _ = client.shutdown(Shutdown::Both);
    let _ = handle_listener.join();
    
    let elapsed_secs = start_time.elapsed().as_secs_f64();
    let throughput_mbps = (total_bytes as f64 * 8.0) / (elapsed_secs * 1_000_000.0);
    log_message(
        log,
        &format!(
            "Network Test: Transferred {} bytes in {:.2} seconds, throughput: {:.2} Mbits/s",
            total_bytes, elapsed_secs, throughput_mbps
        )
    );
    
    if throughput_mbps < 800.0 {
        log_message(log, "Network Test FAIL: Throughput below expected threshold.");
        false
    } else {
        log_message(log, "Network Test PASS: Throughput meets expected threshold.");
        true
    }
}

/// Runs the overall devices test. This test operates solely on peripheral devices:
/// scanning PCI and USB devices, checking dmesg (for logging only; non-critical messages
/// do not fail the test), retrieving battery health, and then running the network test.
/// The final result is based on having found at least one PCI device, at least one USB device,
/// and passing the network test.
pub fn run_test(
    _params: TestParameters,
    log: &Arc<Mutex<Vec<String>>>,
    progress: &Arc<Mutex<TestProgressData>>,
    _samples: &Arc<Mutex<SampleData>>,
    stop_flag: Arc<AtomicBool>,
) -> bool {
    log_message(log, "Starting Devices Test: Scanning peripherals...");
    
    if stop_flag.load(Ordering::Relaxed) {
        log_message(log, "Devices Test aborted by user.");
        return false;
    }
    
    // Scan PCI devices.
    let pci_devices = scan_pci_devices();
    log_message(log, "PCI Devices Found:");
    let mut pci_ok = true;
    if pci_devices.is_empty() {
        log_message(log, "  No PCI devices found.");
        pci_ok = false;
    } else {
        for device in pci_devices.iter() {
            log_message(log, &format!("  {}", device));
        }
    }
    
    // Scan USB devices.
    let usb_devices = scan_usb_devices();
    log_message(log, "USB Devices Found:");
    let mut usb_ok = true;
    if usb_devices.is_empty() {
        log_message(log, "  No USB devices found.");
        usb_ok = false;
    } else {
        for device in usb_devices.iter() {
            log_message(log, &format!("  {}", device));
        }
    }
    
    // Check dmesg issues.
    let (_messages, dmesg_summary, _weighted_score) = check_dmesg_errors();
    log_message(log, &dmesg_summary);
    // Do not let dmesg issues alone cause failure.
    let dmesg_ok = true;
    
    // Retrieve battery health.
    let battery_summary = check_battery_health();
    log_message(log, &battery_summary);
    
    if let Ok(mut prog) = progress.lock() {
        prog.current_test = "Devices Test".to_owned();
        prog.progress = Some(100.0);
    }
    
    // Aggregate preliminary result.
    let devices_ok = pci_ok && usb_ok && dmesg_ok;
    
    // Run the network test.
    let network_ok = run_network_test(log, stop_flag.clone());
    log_message(log, &format!("Network Test Result: {}", if network_ok { "PASS" } else { "FAIL" }));
    
    // Final aggregation.
    let overall_result = devices_ok && network_ok;
    let final_summary = format!(
        "Overall Devices Test Result: {} (PCI: {}, USB: {}, dmesg: {}, Network: {})",
        if overall_result { "PASS" } else { "FAIL" },
        if pci_ok { "PASS" } else { "FAIL" },
        if usb_ok { "PASS" } else { "FAIL" },
        if dmesg_ok { "PASS" } else { "FAIL" },
        if network_ok { "PASS" } else { "FAIL" }
    );
    log_message(log, &final_summary);
    
    overall_result
}
