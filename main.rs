use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

mod common;
pub use common::test_parameters::TestParameters;
pub use common::{
    TestProgressData, SampleData, timestamped_log_entry, 
    get_overall_cpu_temp, get_cpu_usage, get_memory_usage, get_gpu_info,
};

mod cpu_test;
mod memory_test;
mod storage_test;
mod devices_test;
// Note: Do not include mod gpu_test here.

use eframe::egui::{self, CentralPanel, Context, SidePanel, TopBottomPanel};
use eframe::App;
use rfd::FileDialog;
use std::process::Command;

/// Spawns the external GPU test executable ("gpu_test") using a relative path.
/// When built with Cargo (release mode) the GPU test binary is in the same directory.
fn run_gpu_test_separately() -> bool {
    println!("GPU Test: Spawning separate process for GPU test.");
    // Production: Both main and gpu_test are in the same directory.
    let gpu_test_path = "./gpu_test";
    match Command::new(gpu_test_path).status() {
        Ok(status) if status.success() => true,
        Ok(status) => {
            eprintln!("GPU test process returned non-zero status: {:?}", status);
            false
        }
        Err(e) => {
            eprintln!("Failed to spawn gpu_test process: {:?}", e);
            false
        }
    }
}

/// Generates a diagnostic summary using the test results.
fn generate_diagnostic_summary(selected_mode: TestMode, results: &TestResults) -> String {
    format!(
        "\n=== Diagnostic Summary ===\n\
         CPU Stress Test: {}\n\
         Memory Test: {}\n\
         SMART Storage Test: {}\n\
         Peripheral Test: {}\n\
         GPU Stress Test: {}\n",
        results.cpu_stress.clone().unwrap_or_else(|| "N/A".to_string()),
        results.memory.clone().unwrap_or_else(|| "N/A".to_string()),
        results.smart_storage.clone().unwrap_or_else(|| "N/A".to_string()),
        results.peripheral.clone().unwrap_or_else(|| "N/A".to_string()),
        results.gpu_stress.clone().unwrap_or_else(|| "N/A".to_string())
    )
}

/// PDF generation function remains unchanged.
fn generate_pdf_report_interactive(company: &str, technician: &str, log_content: &str) {
    // ... (your existing PDF generation code)
}

/// Test mode and test result types.
#[derive(PartialEq, Clone, Debug)]
enum TestMode {
    DefaultBurnIn,
    CpuStress,
    Memory,
    SmartStorage,
    Peripheral,
    GpuStress,
}
impl Default for TestMode {
    fn default() -> Self {
        TestMode::DefaultBurnIn
    }
}

#[derive(Debug, Default)]
struct TestResults {
    cpu_stress: Option<String>,
    memory: Option<String>,
    smart_storage: Option<String>,
    peripheral: Option<String>,
    gpu_stress: Option<String>,
}

/// The main application struct.
struct NemodexApp {
    log: Arc<Mutex<Vec<String>>>,
    selected_mode: TestMode,
    progress: Arc<Mutex<TestProgressData>>,
    stop_flag: Arc<AtomicBool>,
    test_active: Arc<AtomicBool>,
    final_elapsed: Arc<Mutex<Option<u64>>>,
    overall_start_time: Option<Instant>,
    last_sensor_update: Instant,
    cached_cpu_temp: Option<f64>,
    cached_cpu_usage: Option<f32>,
    cached_ram_usage: Option<f64>,
    report_form_open: bool,
    company_name: String,
    technician_name: String,
}

impl Default for NemodexApp {
    fn default() -> Self {
        Self {
            log: Arc::new(Mutex::new(Vec::new())),
            selected_mode: TestMode::default(),
            progress: Arc::new(Mutex::new(TestProgressData::default())),
            stop_flag: Arc::new(AtomicBool::new(false)),
            test_active: Arc::new(AtomicBool::new(false)),
            final_elapsed: Arc::new(Mutex::new(None)),
            overall_start_time: None,
            last_sensor_update: Instant::now(),
            cached_cpu_temp: None,
            cached_cpu_usage: None,
            cached_ram_usage: None,
            report_form_open: false,
            company_name: String::new(),
            technician_name: String::new(),
        }
    }
}

impl NemodexApp {
    fn start_test(&mut self) {
        self.stop_flag.store(false, Ordering::Relaxed);
        self.overall_start_time = Some(Instant::now());
        self.test_active.store(true, Ordering::Relaxed);
        {
            let mut log = self.log.lock().unwrap();
            log.clear();
            log.push(timestamped_log_entry("Starting selected test suite..."));
        }
        let test_start = self.overall_start_time.unwrap();
        let final_elapsed = Arc::clone(&self.final_elapsed);
        let results = Arc::new(Mutex::new(TestResults::default()));
        let mut params = TestParameters::default();
        if self.selected_mode == TestMode::DefaultBurnIn {
            params.duration_minutes = 5;
        }
        let log_arc = Arc::clone(&self.log);
        let progress_arc = Arc::clone(&self.progress);
        let samples_arc = Arc::new(Mutex::new(Vec::<(f64, f64)>::new()));
        let stop_flag = self.stop_flag.clone();
        let test_active = self.test_active.clone();
        let selected_mode = self.selected_mode.clone();

        thread::spawn(move || {
            // For DefaultBurnIn, run all tests sequentially.
            if selected_mode == TestMode::DefaultBurnIn {
                log_arc.lock().unwrap().push(timestamped_log_entry("Starting Default Burn-In Stress Test..."));
                let cpu_success = cpu_test::run_test(params.clone(), &log_arc, &progress_arc, &samples_arc, stop_flag.clone());
                {
                    let mut res = results.lock().unwrap();
                    res.cpu_stress = Some(if cpu_success { "PASS" } else { "FAIL" }.to_string());
                }
                if !cpu_success {
                    log_arc.lock().unwrap().push(timestamped_log_entry("CPU Test Failed."));
                }
                let mem_success = memory_test::run_test(params.clone(), &log_arc, &progress_arc, &samples_arc, stop_flag.clone());
                {
                    let mut res = results.lock().unwrap();
                    res.memory = Some(if mem_success { "PASS" } else { "FAIL" }.to_string());
                }
                if !mem_success {
                    log_arc.lock().unwrap().push(timestamped_log_entry("Memory Test Failed."));
                }
                let storage_success = storage_test::run_test(params.clone(), &log_arc, &progress_arc, &samples_arc, stop_flag.clone());
                {
                    let mut res = results.lock().unwrap();
                    res.smart_storage = Some(if storage_success { "PASS" } else { "FAIL" }.to_string());
                }
                if !storage_success {
                    log_arc.lock().unwrap().push(timestamped_log_entry("SMART Storage Test Failed."));
                }
                let peripheral_success = devices_test::run_test(params.clone(), &log_arc, &progress_arc, &samples_arc, stop_flag.clone());
                {
                    let mut res = results.lock().unwrap();
                    res.peripheral = Some(if peripheral_success { "PASS" } else { "FAIL" }.to_string());
                }
                if !peripheral_success {
                    log_arc.lock().unwrap().push(timestamped_log_entry("Peripheral Test Failed."));
                }
                let gpu_success = run_gpu_test_separately();
                {
                    let mut res = results.lock().unwrap();
                    res.gpu_stress = Some(if gpu_success { "PASS" } else { "FAIL" }.to_string());
                }
                if !gpu_success {
                    log_arc.lock().unwrap().push(timestamped_log_entry("GPU Test Failed or Stopped."));
                }
            } else {
                // For each individual test mode, run only the corresponding test.
                match selected_mode {
                    TestMode::CpuStress => {
                        log_arc.lock().unwrap().push(timestamped_log_entry("Starting CPU Stress Test..."));
                        let cpu_success = cpu_test::run_test(params.clone(), &log_arc, &progress_arc, &samples_arc, stop_flag.clone());
                        {
                            let mut res = results.lock().unwrap();
                            res.cpu_stress = Some(if cpu_success { "PASS" } else { "FAIL" }.to_string());
                        }
                        if !cpu_success {
                            log_arc.lock().unwrap().push(timestamped_log_entry("CPU Stress Test Failed."));
                        }
                    }
                    TestMode::Memory => {
                        log_arc.lock().unwrap().push(timestamped_log_entry("Starting Memory Test..."));
                        let mem_success = memory_test::run_test(params.clone(), &log_arc, &progress_arc, &samples_arc, stop_flag.clone());
                        {
                            let mut res = results.lock().unwrap();
                            res.memory = Some(if mem_success { "PASS" } else { "FAIL" }.to_string());
                        }
                        if !mem_success {
                            log_arc.lock().unwrap().push(timestamped_log_entry("Memory Test Failed."));
                        }
                    }
                    TestMode::SmartStorage => {
                        log_arc.lock().unwrap().push(timestamped_log_entry("Starting SMART Storage Test..."));
                        let storage_success = storage_test::run_test(params.clone(), &log_arc, &progress_arc, &samples_arc, stop_flag.clone());
                        {
                            let mut res = results.lock().unwrap();
                            res.smart_storage = Some(if storage_success { "PASS" } else { "FAIL" }.to_string());
                        }
                        if !storage_success {
                            log_arc.lock().unwrap().push(timestamped_log_entry("SMART Storage Test Failed."));
                        }
                    }
                    TestMode::Peripheral => {
                        log_arc.lock().unwrap().push(timestamped_log_entry("Starting Peripheral Test..."));
                        let peripheral_success = devices_test::run_test(params.clone(), &log_arc, &progress_arc, &samples_arc, stop_flag.clone());
                        {
                            let mut res = results.lock().unwrap();
                            res.peripheral = Some(if peripheral_success { "PASS" } else { "FAIL" }.to_string());
                        }
                        if !peripheral_success {
                            log_arc.lock().unwrap().push(timestamped_log_entry("Peripheral Test Failed."));
                        }
                    }
                    TestMode::GpuStress => {
                        log_arc.lock().unwrap().push(timestamped_log_entry("Starting GPU Stress Test..."));
                        let gpu_success = run_gpu_test_separately();
                        {
                            let mut res = results.lock().unwrap();
                            res.gpu_stress = Some(if gpu_success { "PASS" } else { "FAIL" }.to_string());
                        }
                        if !gpu_success {
                            log_arc.lock().unwrap().push(timestamped_log_entry("GPU Test Failed or Stopped."));
                        }
                    }
                    _ => {}
                }
            }
            let res = results.lock().unwrap();
            let summary = generate_diagnostic_summary(selected_mode, &res);
            log_arc.lock().unwrap().push(timestamped_log_entry(&summary));
            let elapsed = Instant::now().duration_since(test_start).as_secs();
            *final_elapsed.lock().unwrap() = Some(elapsed);
            test_active.store(false, Ordering::Relaxed);
        });
    }
    
    fn stop_test(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        self.test_active.store(false, Ordering::Relaxed);
        if let Some(start) = self.overall_start_time {
            let elapsed = Instant::now().duration_since(start).as_secs();
            *self.final_elapsed.lock().unwrap() = Some(elapsed);
        }
        if let Ok(mut log) = self.log.lock() {
            log.push(timestamped_log_entry("Testing stopped by user."));
        }
    }
    
    fn open_report_form(&mut self) {
        self.report_form_open = true;
    }
}

impl App for NemodexApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        let mut visuals = egui::Visuals::dark();
        visuals.override_text_color = Some(egui::Color32::from_rgb(0xcf, 0xa3, 0x4d));
        visuals.widgets.noninteractive.bg_fill = egui::Color32::BLACK;
        ctx.set_visuals(visuals);
        let mut style = (*ctx.style()).clone();
        style.visuals.widgets.active.bg_fill = egui::Color32::from_rgb(0xcf, 0xa3, 0x4d);
        style.visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(0xcc, 0x94, 0x4c);
        ctx.set_style(style);

        if Instant::now().duration_since(self.last_sensor_update) > Duration::from_secs(1) {
            if let Some(temp) = common::get_overall_cpu_temp() {
                self.cached_cpu_temp = Some(temp);
            }
            if let Some(usage) = common::get_cpu_usage() {
                self.cached_cpu_usage = Some(usage);
            }
            if let Some((used, total)) = common::get_memory_usage() {
                if total > 0.0 {
                    self.cached_ram_usage = Some((used / total) * 100.0);
                }
            }
            self.last_sensor_update = Instant::now();
        }
        
        let elapsed_display = if self.test_active.load(Ordering::Relaxed) {
            if let Some(start) = self.overall_start_time {
                format!("Elapsed: {} s", Instant::now().duration_since(start).as_secs())
            } else {
                "Elapsed: N/A".to_string()
            }
        } else {
            if let Some(final_time) = *self.final_elapsed.lock().unwrap() {
                format!("Final Time: {} s", final_time)
            } else {
                "Elapsed: N/A".to_string()
            }
        };

        let cpu_temp_display = if let Some(temp) = self.cached_cpu_temp {
            format!("CPU Temp: {:.1}°C", temp)
        } else {
            "CPU Temp: N/A".to_string()
        };
        let cpu_usage_display = if let Some(usage) = self.cached_cpu_usage {
            format!("CPU Usage: {:.1}%", usage)
        } else {
            "CPU Usage: N/A".to_string()
        };
        let ram_usage_display = if let Some(usage) = self.cached_ram_usage {
            format!("RAM Usage: {:.1}%", usage)
        } else {
            "RAM Usage: N/A".to_string()
        };

        let (gpu_temp_opt, gpu_usage_opt) = common::get_gpu_info();
        let gpu_temp_display = if let Some(temp) = gpu_temp_opt {
            format!("GPU Temp: {:.1}°C", temp)
        } else {
            "GPU Temp: N/A".to_string()
        };
        let gpu_usage_display = if let Some(usage) = gpu_usage_opt {
            format!("GPU Usage: {:.1}%", usage)
        } else {
            "GPU Usage: N/A".to_string()
        };

        TopBottomPanel::top("header").show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.add_space(10.0);
                ui.heading("NemoDex Stress Test Tool");
                ui.add_space(10.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(elapsed_display);
                    ui.label(cpu_temp_display);
                    ui.label(cpu_usage_display);
                    ui.label(ram_usage_display);
                    ui.label(gpu_temp_display);
                    ui.label(gpu_usage_display);
                });
                ui.add_space(10.0);
            });
        });
       
        SidePanel::left("side_panel").show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.label("Select Test Mode:");
                ui.radio_value(&mut self.selected_mode, TestMode::DefaultBurnIn, "Default Burn-In Stress Test");
                ui.radio_value(&mut self.selected_mode, TestMode::CpuStress, "CPU Stress Test");
                ui.radio_value(&mut self.selected_mode, TestMode::Memory, "Memory Test");
                ui.radio_value(&mut self.selected_mode, TestMode::SmartStorage, "SMART Storage Test");
                ui.radio_value(&mut self.selected_mode, TestMode::Peripheral, "Peripheral Test");
                ui.radio_value(&mut self.selected_mode, TestMode::GpuStress, "GPU Stress Test");
                ui.separator();
                if ui.button("START TEST").clicked() {
                    self.start_test();
                }
                if ui.button("STOP").clicked() {
                    self.stop_test();
                }
                ui.separator();
                if ui.button("Generate Report").clicked() {
                    self.open_report_form();
                }
            });
        });
       
        CentralPanel::default().show(ctx, |ui| {
            ui.heading("Log Output");
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                if let Ok(log) = self.log.lock() {
                    for entry in log.iter() {
                        ui.label(entry);
                    }
                } else {
                    ui.label("Log is updating...");
                }
            });
        });
       
        if self.report_form_open {
            egui::Window::new("Generate Diagnostic Report")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("Enter Company Name:");
                    ui.text_edit_singleline(&mut self.company_name);
                    ui.label("Enter Technician Name:");
                    ui.text_edit_singleline(&mut self.technician_name);
                    if ui.button("Save Report").clicked() {
                        let log_content = {
                            let log = self.log.lock().unwrap();
                            log.join("\n")
                        };
                        generate_pdf_report_interactive(&self.company_name, &self.technician_name, &log_content);
                        self.report_form_open = false;
                    }
                    if ui.button("Cancel").clicked() {
                        self.report_form_open = false;
                    }
                });
        }
       
        ctx.request_repaint();
    }
}

fn main() {
    let custom_lib_path = "/path/to/custom_module/usr/lib";
    let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let new_ld_path = if existing.is_empty() {
        custom_lib_path.to_owned()
    } else {
        format!("{}:{}", custom_lib_path, existing)
    };
    std::env::set_var("LD_LIBRARY_PATH", new_ld_path);
    
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(800.0,600.0)),
        ..Default::default()
    };
    let _ = eframe::run_native(
        "NemoDex Stress Test Tool",
        options,
        Box::new(|_cc| Box::<NemodexApp>::default()),
    );
}
