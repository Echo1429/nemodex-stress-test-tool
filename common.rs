// src/common.rs

use std::fs;
use std::time::Duration;
use std::process::Command;
use regex::Regex;

/// Public test parameters module.
pub mod test_parameters {
    #[derive(Clone)]
    pub struct TestParameters {
        pub duration_minutes: u64,
        pub cpu_temp_threshold: f64,
        pub gpu_temp_threshold: f64,
        pub disk_write_limit: u64,
        pub vrm_voltage_fluctuation_threshold: f64,
    }
    impl Default for TestParameters {
        fn default() -> Self {
            Self {
                duration_minutes: 60,
                cpu_temp_threshold: 90.0,
                gpu_temp_threshold: 85.0,
                disk_write_limit: 10_000_000_000,
                vrm_voltage_fluctuation_threshold: 0.1,
            }
        }
    }
}
pub use test_parameters::TestParameters;

/// Data structure to track progress.
#[derive(Debug)]
pub struct TestProgressData {
    pub current_test: String,
    pub progress: Option<f64>,
    pub test_elapsed: Option<f64>,
    pub cpu_temp: Option<f64>,
}
impl Default for TestProgressData {
    fn default() -> Self {
        Self {
            current_test: "None".to_string(),
            progress: None,
            test_elapsed: None,
            cpu_temp: None,
        }
    }
}

/// Alias for sample data.
pub type SampleData = Vec<(f64, f64)>;

/// Returns a timestamped log entry.
pub fn timestamped_log_entry(entry: &str) -> String {
    use chrono::Local;
    let now = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    format!("[{}] {}", now, entry)
}

/// --- CPU Temperature Functions ---

fn get_cpu_temp_thermal_zones() -> Option<f64> {
    let paths = fs::read_dir("/sys/class/thermal/").ok()?;
    let mut temps = Vec::new();
    for entry in paths.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let temp_path = path.join("temp");
            if temp_path.exists() {
                if let Ok(contents) = fs::read_to_string(&temp_path) {
                    if let Ok(val) = contents.trim().parse::<f64>() {
                        temps.push(val / 1000.0);
                    }
                }
            } else {
                let zone_files = fs::read_dir(&path).ok()?;
                for file in zone_files.flatten() {
                    let fname = file.file_name().into_string().unwrap_or_default();
                    if fname.starts_with("temp") && fname.ends_with("input") {
                        if let Ok(contents) = fs::read_to_string(file.path()) {
                            if let Ok(val) = contents.trim().parse::<f64>() {
                                temps.push(val / 1000.0);
                            }
                        }
                    }
                }
            }
        }
    }
    if temps.is_empty() {
        None
    } else {
        Some(temps.iter().sum::<f64>() / temps.len() as f64)
    }
}

fn get_cpu_temp_from_sysfs() -> Option<f64> {
    for zone in &[
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/thermal/thermal_zone1/temp",
        "/sys/class/thermal/thermal_zone2/temp",
    ] {
        if let Ok(contents) = fs::read_to_string(zone) {
            if let Ok(val) = contents.trim().parse::<f64>() {
                return Some(val / 1000.0);
            }
        }
    }
    None
}

fn get_cpu_temp_hwmon() -> Option<f64> {
    if let Ok(entries) = fs::read_dir("/sys/class/hwmon/") {
        for entry in entries.flatten() {
            let hwmon_path = entry.path();
            if let Ok(sensor_name) = fs::read_to_string(hwmon_path.join("name")) {
                if sensor_name.trim().contains("k10temp") {
                    if let Ok(hwmon_files) = fs::read_dir(&hwmon_path) {
                        for file in hwmon_files.flatten() {
                            let file_name = file.file_name().into_string().unwrap_or_default();
                            if file_name.starts_with("temp") && file_name.ends_with("_input") {
                                if let Ok(contents) = fs::read_to_string(file.path()) {
                                    if let Ok(val) = contents.trim().parse::<f64>() {
                                        return Some(val / 1000.0);
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if let Ok(hwmon_files) = fs::read_dir(&hwmon_path) {
                    for file in hwmon_files.flatten() {
                        let file_name = file.file_name().into_string().unwrap_or_default();
                        if file_name.starts_with("temp") && file_name.ends_with("_input") {
                            if let Ok(contents) = fs::read_to_string(file.path()) {
                                if let Ok(val) = contents.trim().parse::<f64>() {
                                    return Some(val / 1000.0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

pub fn get_overall_cpu_temp() -> Option<f64> {
    if let Some(temp) = get_cpu_temp_thermal_zones() {
        Some(temp)
    } else if let Some(temp) = get_cpu_temp_from_sysfs() {
        Some(temp)
    } else {
        get_cpu_temp_hwmon()
    }
}

/// --- End CPU Temperature Functions ---

pub fn get_cpu_usage() -> Option<f32> {
    use systemstat::{Platform, System};
    let sys = System::new();
    match sys.cpu_load() {
        Ok(delayed) => {
            std::thread::sleep(Duration::from_secs(1));
            match delayed.done() {
                Ok(loads) => {
                    let sum_idle: f64 = loads.iter().map(|l| l.idle as f64).sum();
                    let avg_idle = sum_idle / loads.len() as f64;
                    Some(((1.0 - avg_idle) * 100.0) as f32)
                },
                Err(e) => {
                    eprintln!("Error computing CPU loads: {}", e);
                    None
                },
            }
        },
        Err(e) => {
            eprintln!("Error retrieving CPU loads: {}", e);
            None
        },
    }
}

pub fn get_memory_usage() -> Option<(f64, f64)> {
    use systemstat::{Platform, System};
    let sys = System::new();
    match sys.memory() {
        Ok(mem) => {
            let total_bytes = mem.total.0;
            let free_bytes = mem.free.0;
            let used_bytes = total_bytes.saturating_sub(free_bytes);
            let used_gb = used_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
            let total_gb = total_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
            Some((used_gb, total_gb))
        },
        Err(e) => {
            eprintln!("Error retrieving memory usage: {}", e);
            None
        },
    }
}

/// --- GPU Info Using sensors ---
/// Uses the `sensors` command to extract GPU temperature.
fn get_gpu_temp_from_sensors() -> Option<f64> {
    if let Ok(output) = Command::new("sensors").output() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if let Ok(re) = Regex::new(r"GPU\s+Temp:\s+\+([\d\.]+)°C") {
            for line in stdout.lines() {
                if let Some(cap) = re.captures(line) {
                    if let Some(ts) = cap.get(1) {
                        if let Ok(temp) = ts.as_str().parse::<f64>() {
                            return Some(temp);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Returns GPU info: (temperature, utilization). Utilization is unavailable here.
pub fn get_gpu_info() -> (Option<f64>, Option<f32>) {
    let temp = get_gpu_temp_from_sensors();
    (temp, None)
}
