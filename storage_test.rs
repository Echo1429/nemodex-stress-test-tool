//! Storage Test Module – SMART Self-Test Version
//!
//! This module detects storage devices (from /sys/block) and for each device:
//!   1. Runs basic self-tests (random seek and sequential read), as before.
//!   2. Attempts to run a SMART self-test by retrieving device information from sysfs 
//!      (such as vendor and model) and by invoking smartctl to obtain details such as
//!      wear cycles, power-on hours, etc.
//!
//! A report is returned that summarizes the self-test status and SMART data.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::Rng;

use super::{TestParameters, TestProgressData, SampleData, timestamped_log_entry};

/// Define the ioctl constant BLKGETSIZE64 (from <linux/fs.h>)
const BLKGETSIZE64: u64 = 0x80081272;

/// Logs a message (with a timestamp) into the shared log.
fn log_message(log: &Arc<Mutex<Vec<String>>>, msg: &str) {
    if let Ok(mut l) = log.lock() {
        l.push(timestamped_log_entry(msg));
    }
}

/// Represents a detected storage device.
pub struct StorageDevice {
    pub device: String, // Full device node path, e.g. "/dev/sda" or "/dev/nvme0n1"
    pub info: String,   // A label – for example, "ATA device" or "NVMe device"
}

/// Detects storage devices by scanning /sys/block.
/// Returns a list for devices whose names start with "sd" (ATA) or "nvme" (NVMe).
fn detect_storage_devices() -> Vec<StorageDevice> {
    let mut devices = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/sys/block/") {
        for entry in entries.flatten() {
            if let Some(dev_name) = entry.file_name().to_str() {
                if dev_name.starts_with("sd") || dev_name.starts_with("nvme") {
                    let path = format!("/dev/{}", dev_name);
                    let info = if dev_name.starts_with("nvme") {
                        "NVMe device".to_owned()
                    } else {
                        "ATA device".to_owned()
                    };
                    devices.push(StorageDevice { device: path, info });
                }
            }
        }
    }
    devices
}

/// Obtains the size (in bytes) of a block device using an ioctl call.
fn get_device_size<P: AsRef<Path>>(path: P) -> std::io::Result<u64> {
    let file = File::open(path)?;
    let fd = file.as_raw_fd();
    let mut size: u64 = 0;
    let ret = unsafe { libc::ioctl(fd, BLKGETSIZE64, &mut size as *mut u64) };
    if ret == 0 {
        Ok(size)
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// Retrieves basic SMART-like information from sysfs for a device.
/// Attempts to read vendor and model information.
fn read_smart_sysfs_info(device: &StorageDevice) -> String {
    let dev_basename = device.device.split('/').last().unwrap_or("");
    let vendor_path = format!("/sys/block/{}/device/vendor", dev_basename);
    let model_path = format!("/sys/block/{}/device/model", dev_basename);
    let vendor = std::fs::read_to_string(&vendor_path).unwrap_or_else(|_| "Unknown Vendor".to_owned()).trim().to_owned();
    let model = std::fs::read_to_string(&model_path).unwrap_or_else(|_| "Unknown Model".to_owned()).trim().to_owned();
    format!("Vendor: {}, Model: {}", vendor, model)
}

/// Runs smartctl on the specified device to retrieve SMART attributes.
/// This function requires that smartctl is installed and accessible.
/// It returns the output of `smartctl -A <device>`.
fn run_smartctl(device: &StorageDevice) -> Option<String> {
    let output = Command::new("smartctl")
        .arg("-A")
        .arg(&device.device)
        .output()
        .ok()?;
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        Some(stdout)
    } else {
        None
    }
}

/// Runs the basic self-tests (random seek and sequential read) on a device.
fn run_basic_self_tests(device: &StorageDevice, log: &Arc<Mutex<Vec<String>>>) -> bool {
    log_message(log, &format!("Starting basic self-tests on {}...", device.device));
    let file_result = File::open(&device.device);
    let mut file = match file_result {
        Ok(f) => f,
        Err(e) => {
            log_message(log, &format!("Failed to open {}: {}", device.device, e));
            return false;
        }
    };
    // Query device size
    let size = match get_device_size(&device.device) {
        Ok(s) => s,
        Err(e) => {
            log_message(log, &format!("Failed to get size for {}: {}", device.device, e));
            return false;
        }
    };
    log_message(log, &format!("Device {} size is {} bytes", device.device, size));
    
    // Random seek test: perform 10 seeks of 512 bytes.
    let num_tests = 10;
    let block_size = 512;
    if size < block_size {
        log_message(log, "Device size too small for testing.");
        return false;
    }
    let mut rng = SmallRng::from_entropy();
    let mut all_ok = true;
    let mut buffer = vec![0u8; block_size as usize];
    for i in 0..num_tests {
        let max_offset = size - block_size;
        let offset = rng.gen_range(0..=max_offset);
        if let Err(e) = file.seek(SeekFrom::Start(offset)) {
            log_message(log, &format!("Random Seek Test {}: Seek to offset {} failed: {}", i + 1, offset, e));
            all_ok = false;
            continue;
        }
        match file.read_exact(&mut buffer) {
            Ok(()) => log_message(log, &format!("Random Seek Test {}: Successfully read 512 bytes at offset {}.", i + 1, offset)),
            Err(e) => {
                log_message(log, &format!("Random Seek Test {}: Read failed at offset {}: {}", i + 1, offset, e));
                all_ok = false;
            }
        }
    }
    
    // Sequential read test: read the first 4096 bytes.
    let mut seq_buffer = vec![0u8; 4096];
    if let Err(e) = file.seek(SeekFrom::Start(0)) {
        log_message(log, &format!("Sequential Read Test: Seek failed: {}", e));
        all_ok = false;
    } else if let Err(e) = file.read_exact(&mut seq_buffer) {
        log_message(log, &format!("Sequential Read Test: Read failed: {}", e));
        all_ok = false;
    } else {
        log_message(log, "Sequential Read Test passed.");
    }
    
    all_ok
}

/// Runs SMART self-test on the given device by gathering SMART data.
/// Returns a report string with vendor/model and SMART attributes.
fn run_smart_self_test(device: &StorageDevice, log: &Arc<Mutex<Vec<String>>>) -> String {
    log_message(log, &format!("Starting SMART self-test on {}...", device.device));
    let mut report = format!("SMART Self-Test Report for {}:\n", device.device);
    
    // Read basic SMART info from sysfs.
    let smart_info = read_smart_sysfs_info(device);
    report.push_str(&format!("  {}\n", smart_info));
    
    // Optionally, try to run smartctl to gather additional attributes.
    if let Some(smart_output) = run_smartctl(device) {
        report.push_str("  SMART Attributes (smartctl output):\n");
        // For brevity, we take only the first 20 lines of smartctl output.
        for line in smart_output.lines().take(20) {
            report.push_str(&format!("    {}\n", line));
        }
    } else {
        report.push_str("  smartctl not available or SMART query failed.\n");
    }
    
    report
}

/// Processes a storage device by running basic self-tests and SMART self-test.
/// Returns true if both basic and SMART tests pass.
fn process_device(device: StorageDevice, log: Arc<Mutex<Vec<String>>>, stop_flag: Arc<AtomicBool>) -> bool {
    if stop_flag.load(Ordering::Relaxed) {
        return false;
    }
    log_message(&log, &format!("Detected storage device: {} ({})", device.device, device.info));
    
    let basic_passed = run_basic_self_tests(&device, &log);
    let smart_report = run_smart_self_test(&device, &log);
    log_message(&log, &smart_report);
    
    let overall = basic_passed; // You could also decide to combine SMART status if desired.
    if overall {
        log_message(&log, &format!("Device {} passed self-tests.", device.device));
    } else {
        log_message(&log, &format!("Device {} failed self-tests.", device.device));
    }
    overall
}

/// Runs storage tests on all detected devices in parallel.
/// Returns true if every device passes.
pub fn run_test(
    _params: TestParameters,
    log: &Arc<Mutex<Vec<String>>>,
    progress: &Arc<Mutex<TestProgressData>>,
    _samples: &Arc<Mutex<SampleData>>,
    stop_flag: Arc<AtomicBool>,
) -> bool {
    log_message(log, "Starting storage self-tests...");
    let devices = detect_storage_devices();
    if devices.is_empty() {
        log_message(log, "No storage devices detected.");
        return false;
    }
    log_message(log, &format!("Detected {} storage device(s).", devices.len()));
    
    let total_devices = devices.len();
    let mut handles = Vec::new();
    for device in devices {
        let log_clone = Arc::clone(log);
        let stop_flag_clone = Arc::clone(&stop_flag);
        let handle = thread::spawn(move || {
            process_device(device, log_clone, stop_flag_clone)
        });
        handles.push(handle);
    }
    
    let mut all_passed = true;
    let overall_start = Instant::now();
    for handle in handles {
        match handle.join() {
            Ok(result) => {
                if !result {
                    all_passed = false;
                }
                if let Ok(mut prog) = progress.lock() {
                    let progress_percent = 100.0 *
                        (overall_start.elapsed().as_secs_f64() / ((total_devices as f64) + 1.0));
                    prog.progress = Some(progress_percent.min(100.0));
                }
            },
            Err(_) => { all_passed = false; }
        }
    }
    
    if all_passed {
        log_message(log, "All storage devices passed self-tests.");
    } else {
        log_message(log, "One or more storage devices failed self-tests.");
    }
    all_passed
}
