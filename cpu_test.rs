//! Professional-grade CPU stress test module.
//!
//! This module spawns worker threads to conduct two types of heavy CPU work:
//!   1. Matrix multiplication of large 512×512 matrices combined with the Lucas–Lehmer
//!      test (using p = 31) to stress the CPU.
//!   2. Compression and decompression of a 1 MB block of data using gzip (via flate2).
//!
//! The test monitors elapsed time and CPU temperature (via helper functions in main.rs)
//! and aborts gracefully if the temperature exceeds a threshold (e.g. 95°C).
//!
//! # Usage
//! Call `run_test` from the main GUI.  
//!
//! Note: The Lucas–Lehmer test is expected to succeed for p = 31 on properly functioning hardware.

use std::hint::black_box;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use super::{TestParameters, TestProgressData, SampleData, timestamped_log_entry};

/// Performs the Lucas–Lehmer test for a Mersenne number 2^p - 1.
/// Returns true if the test passes, false otherwise.
fn lucas_lehmer(p: usize) -> bool {
    if p < 2 {
        return false;
    }
    let mut s: u64 = 4;
    let mersenne = (1_u64 << p) - 1; // 2^p - 1
    for _ in 0..(p - 2) {
        s = (s * s - 2) % mersenne;
    }
    s == 0
}

/// Performs matrix multiplication on two square matrices of the given size.
/// The function multiplies matrix `a` by matrix `b` repeatedly.
fn matrix_multiplication(size: usize) {
    let a = vec![vec![1.0_f64; size]; size];
    let b = vec![vec![2.0_f64; size]; size];
    let mut result = vec![vec![0.0_f64; size]; size];
    for _ in 0..10 {
        for i in 0..size {
            for j in 0..size {
                let mut sum = 0.0;
                for k in 0..size {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
    }
}

/// Compress a slice of bytes using gzip compression (via flate2).
fn compress_data(data: &[u8]) -> Vec<u8> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap()
}

/// Decompress a slice of bytes that was compressed with gzip.
fn decompress_data(data: &[u8]) -> Vec<u8> {
    use flate2::read::GzDecoder;
    use std::io::Read;
    let mut decoder = GzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed).unwrap();
    decompressed
}

/// Runs a heavy CPU stress test that includes:
///   - Matrix multiplication and Lucas–Lehmer tests.
///   - Multi-threaded compression/decompression tasks.
/// The test runs for the duration specified in `params.duration_minutes`,
/// monitoring CPU temperature and aborting if it exceeds 95°C.
/// 
/// # Parameters
/// - `params`: Global test parameters.
/// - `log`: Shared log (Arc‑wrapped Mutex) for printing timestamped messages.
/// - `progress`: Shared progress data.
/// - `samples`: Shared vector collecting (elapsed time, temperature) samples.
/// - `stop_flag`: Atomic flag to signal when to abort the test.
/// 
/// # Returns
/// Returns `true` if the test completes normally or is aborted gracefully.
pub fn run_test(
    params: TestParameters,
    log: &Arc<Mutex<Vec<String>>>,
    progress: &Arc<Mutex<TestProgressData>>,
    samples: &Arc<Mutex<SampleData>>,
    stop_flag: Arc<AtomicBool>,
) -> bool {
    {
        let mut l = log.lock().unwrap();
        l.push(timestamped_log_entry("Entered run_test() in cpu_test."));
        l.push(timestamped_log_entry("Starting CPU Stress Test with matrix/Lucas and compression tasks."));
    }

    let total_duration = Duration::from_secs(params.duration_minutes * 60);
    let start_time = Instant::now();

    // Update progress.
    {
        let mut prog = progress.lock().unwrap();
        prog.current_test = "CPU Stress Test".to_owned();
        prog.progress = Some(0.0);
        prog.test_elapsed = Some(0.0);
    }

    // Spawn two types of worker threads.
    let total_cores = num_cpus::get().max(1);
    let matrix_cores = total_cores / 2;
    let compress_cores = total_cores - matrix_cores;

    {
        let mut l = log.lock().unwrap();
        l.push(timestamped_log_entry(&format!(
            "Launching {} matrix worker threads and {} compression worker threads.",
            matrix_cores, compress_cores
        )));
    }

    let matrix_size = 512;
    let lucas_p = 31;
    let mut handles = Vec::new();

    // Spawn matrix/Lucas worker threads.
    for _ in 0..matrix_cores {
        let stop_flag_clone = stop_flag.clone();
        let handle = thread::spawn(move || {
            while !stop_flag_clone.load(Ordering::Relaxed) {
                // Perform heavy matrix multiplication.
                matrix_multiplication(matrix_size);
                // Run the Lucas-Lehmer test.
                let is_prime = lucas_lehmer(lucas_p);
                // Use assert (or log and continue) as needed.
                assert!(is_prime, "Lucas-Lehmer test failed for p = {}", lucas_p);
            }
        });
        handles.push(handle);
    }

    // Prepare sample data for compression: 1 MB of the letter 'A'.
    let sample_data = vec![b'A'; 1024 * 1024];
    // Spawn compression worker threads.
    for _ in 0..compress_cores {
        let stop_flag_clone = stop_flag.clone();
        let sample = sample_data.clone();
        let handle = thread::spawn(move || {
            while !stop_flag_clone.load(Ordering::Relaxed) {
                let compressed = compress_data(&sample);
                let decompressed = decompress_data(&compressed);
                assert_eq!(sample, decompressed, "Compression/Decompression mismatch!");
            }
        });
        handles.push(handle);
    }

    // Monitoring loop: update progress every 2 seconds.
    while start_time.elapsed() < total_duration && !stop_flag.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_secs(2));
        let elapsed = start_time.elapsed().as_secs_f64();
        let percentage = (elapsed / total_duration.as_secs_f64()) * 100.0;
        if let Some(temp) = super::get_overall_cpu_temp() {
            {
                let mut prog = progress.lock().unwrap();
                prog.progress = Some(percentage.min(100.0));
                prog.test_elapsed = Some(elapsed);
                prog.cpu_temp = Some(temp);
            }
            {
                let mut samp = samples.lock().unwrap();
                samp.push((elapsed, temp));
            }
            if temp > 95.0 {
                let mut l = log.lock().unwrap();
                l.push(timestamped_log_entry("Failsafe Triggered: CPU Overheating (>95°C)! Aborting test."));
                stop_flag.store(true, Ordering::Relaxed);
                break;
            }
        } else {
            let mut l = log.lock().unwrap();
            l.push(timestamped_log_entry("No CPU temperature sensor detected."));
        }
    }

    // Signal all worker threads to stop.
    stop_flag.store(true, Ordering::Relaxed);
    for handle in handles {
        let _ = handle.join();
    }

    {
        let mut prog = progress.lock().unwrap();
        prog.progress = Some(100.0);
        prog.test_elapsed = Some(start_time.elapsed().as_secs_f64());
    }
    {
        let mut l = log.lock().unwrap();
        l.push(timestamped_log_entry("CPU Stress Test completed."));
    }
    true
}
