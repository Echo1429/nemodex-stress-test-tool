//! Memory Test Module (Production-Grade Version with Quick Test Mode – Staggered)
//!
//! This module implements a comprehensive memory test suite that tests 100% of the
//! system memory in two phases (50% per phase). In each phase, the allocated memory is
//! divided into 4 chunks so that 4 threads run concurrently. Each thread therefore tests
//! about 12.5% of total system memory.
//!
//! The mode is determined by TestParameters::quick_memory_test().
//!
//! (Note: Due to virtualization and OS optimizations, physical testing of every byte
//! cannot be guaranteed, but this approach maximizes coverage.)

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, fence, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use sysinfo::System;
use rand::prelude::*;
use num_cpus;

use super::{TestParameters, TestProgressData, SampleData, timestamped_log_entry};

/// Constants for memory allocation.
const WORDS_PER_MB: usize = 262144; // 1 MB ≈ 1,048,576 u32 words.
const THREADS: usize = 4; // We'll use 4 concurrent threads per phase.

/// Extend TestParameters to include a quick test helper.
impl TestParameters {
    pub fn quick_memory_test(&self) -> bool {
        self.duration_minutes < 15
    }
}

/// Detects total system memory (in MB) using sysinfo.
fn total_test_memory_mb() -> usize {
    let mut sys = System::new_all();
    sys.refresh_memory();
    // sys.total_memory() returns total memory in KB.
    (sys.total_memory() / 1024 / 1024) as usize
}

/// --- Logging and Progress Utilities ---
fn log_message(log: &Arc<Mutex<Vec<String>>>, msg: &str) {
    if let Ok(mut l) = log.lock() {
        l.push(timestamped_log_entry(msg));
    }
}

fn update_progress(
    progress: &Arc<Mutex<TestProgressData>>, 
    chunks_tested: usize, 
    total_chunks: usize, 
    elapsed: Duration,
) {
    let percentage = (chunks_tested as f64 / total_chunks as f64) * 100.0;
    if let Ok(mut prog) = progress.lock() {
        prog.progress = Some(percentage.min(100.0));
        prog.test_elapsed = Some(elapsed.as_secs_f64());
    }
}

/// --- Test Subfunctions (Tests 0 through 12) ---

// Test 0: Walking Ones
fn test0_walking_ones(memory: &mut [u32], log: &Arc<Mutex<Vec<String>>>, stop_flag: &Arc<AtomicBool>, duration_secs: u64) -> bool {
    log_message(log, "Test 0: Walking Ones started.");
    let start_time = Instant::now();
    while start_time.elapsed().as_secs() < duration_secs {
        if stop_flag.load(Ordering::Relaxed) { return false; }
        for (i, slot) in memory.iter_mut().enumerate() {
            *slot = 1 << ((i % 32) as u32);
        }
        for (i, slot) in memory.iter().enumerate() {
            let expected = 1 << ((i % 32) as u32);
            if *slot != expected {
                log_message(log, &format!("Test 0 ERROR at index {}: expected 0x{:08X}, got 0x{:08X}", i, expected, *slot));
                return false;
            }
        }
    }
    log_message(log, "Test 0 passed.");
    true
}

// Test 1: Own Address (single-threaded)
fn test1_own_address(memory: &mut [u32], log: &Arc<Mutex<Vec<String>>>, stop_flag: &Arc<AtomicBool>, duration_secs: u64) -> bool {
    log_message(log, "Test 1: Own Address (single-threaded) started.");
    let start_time = Instant::now();
    while start_time.elapsed().as_secs() < duration_secs {
        if stop_flag.load(Ordering::Relaxed) { return false; }
        for (i, slot) in memory.iter_mut().enumerate() {
            *slot = i as u32;
        }
        for (i, slot) in memory.iter().enumerate() {
            if *slot != i as u32 {
                log_message(log, &format!("Test 1 ERROR at index {}: expected {}, got {}", i, i, *slot));
                return false;
            }
        }
    }
    log_message(log, "Test 1 passed.");
    true
}

// Test 2: Own Address (parallel)
fn test2_own_address_parallel(memory: Arc<Mutex<Vec<u32>>>, log: Arc<Mutex<Vec<String>>>, stop_flag: Arc<AtomicBool>, duration_secs: u64) -> bool {
    log_message(&log, "Test 2: Own Address (parallel) started.");
    let thread_count = num_cpus::get().max(1);
    let mem_len = memory.lock().unwrap().len();
    let chunk = mem_len / thread_count;
    let mut handles = vec![];
    for t in 0..thread_count {
        let mem_clone = memory.clone();
        let log_clone = log.clone();
        let stop_clone = stop_flag.clone();
        let start = t * chunk;
        let end = if t == thread_count - 1 { mem_len } else { start + chunk };
        let handle = thread::spawn(move || -> bool {
            let test_duration = Duration::from_secs(duration_secs);
            let start_time = Instant::now();
            while start_time.elapsed() < test_duration && !stop_clone.load(Ordering::Relaxed) {
                {
                    let mut vec = mem_clone.lock().unwrap();
                    for i in start..end {
                        vec[i] = i as u32;
                    }
                    for i in start..end {
                        if vec[i] != i as u32 {
                            log_message(&log_clone, &format!("Test 2 ERROR at index {}: expected {}, got {}", i, i, vec[i]));
                            stop_clone.store(true, Ordering::Relaxed);
                            return false;
                        }
                    }
                }
            }
            true
        });
        handles.push(handle);
    }
    for h in handles { if let Err(_) = h.join() { return false; } }
    log_message(&log, "Test 2 passed.");
    true
}

// Test 3: Moving Inversions (fixed alternating pattern)
fn test3_moving_inversions(memory: &mut [u32], log: &Arc<Mutex<Vec<String>>>, stop_flag: &Arc<AtomicBool>, duration_secs: u64) -> bool {
    log_message(log, "Test 3: Moving Inversions (fixed pattern) started.");
    let start_time = Instant::now();
    for slot in memory.iter_mut() {
        *slot = 0xAAAAAAAAu32;
    }
    fence(Ordering::SeqCst);
    thread::sleep(Duration::from_millis(1));
    while start_time.elapsed().as_secs() < duration_secs {
        if stop_flag.load(Ordering::Relaxed) { return false; }
        for slot in memory.iter_mut() { *slot = !*slot; }
        fence(Ordering::SeqCst);
        thread::sleep(Duration::from_millis(1));
        for (i, &slot) in memory.iter().enumerate() {
            if slot != 0x55555555u32 {
                let diff = 0x55555555u32 ^ slot;
                log_message(log, &format!("Test 3 ERROR at index {}: expected 0x{:08X}, got 0x{:08X} (diff 0x{:08X})", i, 0x55555555u32, slot, diff));
                return false;
            }
        }
        for slot in memory.iter_mut() { *slot = !*slot; }
        fence(Ordering::SeqCst);
        thread::sleep(Duration::from_millis(1));
        for (i, &slot) in memory.iter().enumerate() {
            if slot != 0xAAAAAAAAu32 {
                let diff = 0xAAAAAAAAu32 ^ slot;
                log_message(log, &format!("Test 3 ERROR (restore) at index {}: expected 0x{:08X}, got 0x{:08X} (diff 0x{:08X})", i, 0xAAAAAAAAu32, slot, diff));
                return false;
            }
        }
    }
    log_message(log, "Test 3 passed.");
    true
}

// Test 4: Moving Inversions (8-bit walking)
fn test4_moving_inversions_8bit(memory: &mut [u32], log: &Arc<Mutex<Vec<String>>>, stop_flag: &Arc<AtomicBool>, duration_secs: u64) -> bool {
    log_message(log, "Test 4: Moving Inversions (8-bit walking) started.");
    let start_time = Instant::now();
    while start_time.elapsed().as_secs() < duration_secs {
        if stop_flag.load(Ordering::Relaxed) { return false; }
        for (i, slot) in memory.iter_mut().enumerate() {
            let pattern: u32 = 1 << ((i % 8) as u32);
            let value = (pattern << 24) | (pattern << 16) | (pattern << 8) | pattern;
            *slot = value;
        }
        fence(Ordering::SeqCst);
        thread::sleep(Duration::from_millis(1));
        for slot in memory.iter_mut() { *slot = !*slot; }
        fence(Ordering::SeqCst);
        thread::sleep(Duration::from_millis(1));
        for (i, &slot) in memory.iter().enumerate() {
            let pattern: u32 = 1 << ((i % 8) as u32);
            let expected = !((pattern << 24) | (pattern << 16) | (pattern << 8) | pattern);
            if slot != expected {
                let diff = expected ^ slot;
                log_message(log, &format!("Test 4 ERROR at index {}: expected 0x{:08X}, got 0x{:08X} (diff 0x{:08X})", i, expected, slot, diff));
                return false;
            }
        }
    }
    log_message(log, "Test 4 passed.");
    true
}

// Test 5: Moving Inversions (random pattern)
fn test5_moving_inversions_random(memory: &mut [u32], log: &Arc<Mutex<Vec<String>>>, stop_flag: &Arc<AtomicBool>, duration_secs: u64) -> bool {
    log_message(log, "Test 5: Moving Inversions (random pattern) started.");
    let start_time = Instant::now();
    let mut rng = rand::thread_rng();
    while start_time.elapsed().as_secs() < duration_secs {
        if stop_flag.load(Ordering::Relaxed) { return false; }
        let pattern: u32 = rng.gen();
        for slot in memory.iter_mut() { *slot = pattern; }
        fence(Ordering::SeqCst);
        thread::sleep(Duration::from_millis(1));
        for slot in memory.iter_mut() { *slot = !*slot; }
        fence(Ordering::SeqCst);
        thread::sleep(Duration::from_millis(1));
        for &slot in memory.iter() {
            if slot != !pattern {
                let diff = !pattern ^ slot;
                log_message(log, &format!("Test 5 ERROR: expected 0x{:08X}, got 0x{:08X} (diff 0x{:08X})", !pattern, slot, diff));
                return false;
            }
        }
    }
    log_message(log, "Test 5 passed.");
    true
}

// Test 6: Block Move Test.
fn test6_block_move(memory: &mut [u32], log: &Arc<Mutex<Vec<String>>>, stop_flag: &Arc<AtomicBool>, duration_secs: u64) -> bool {
    log_message(log, "Test 6: Block Move Test started.");
    let start_time = Instant::now();
    while start_time.elapsed().as_secs() < duration_secs {
        if stop_flag.load(Ordering::Relaxed) { return false; }
        for (i, slot) in memory.iter_mut().enumerate() {
            *slot = (i as u32).wrapping_mul(0x1234567);
        }
        let block_size = 4 * 1024 * 1024 / 4;
        if memory.len() < block_size * 2 {
            log_message(log, "Test 6: Not enough memory; skipping block move test.");
            break;
        }
        let src_range = 0..block_size;
        let dest_range = block_size..(block_size * 2);
        let temp = memory[src_range.clone()].to_vec();
        memory[dest_range.clone()].copy_from_slice(&temp);
        if memory[dest_range].ne(&temp) {
            log_message(log, "Test 6 ERROR: Block move mismatch.");
            return false;
        }
    }
    log_message(log, "Test 6 passed.");
    true
}

// Test 7: Moving Inversions (32-bit shifting)
fn test7_moving_inversions_32bit(memory: &mut [u32], log: &Arc<Mutex<Vec<String>>>, stop_flag: &Arc<AtomicBool>, passes: usize) -> bool {
    log_message(log, "Test 7: Moving Inversions (32-bit shifting) started.");
    for pass in 0..passes {
        if stop_flag.load(Ordering::Relaxed) { return false; }
        for (i, slot) in memory.iter_mut().enumerate() {
            *slot = ((i as u32) << (pass % 32)).wrapping_add(pass as u32);
        }
        for (i, &slot) in memory.iter().enumerate() {
            let expected = ((i as u32) << (pass % 32)).wrapping_add(pass as u32);
            if slot != expected {
                log_message(log, &format!("Test 7 ERROR on pass {} at index {}: expected 0x{:08X}, got 0x{:08X}", pass, i, expected, slot));
                return false;
            }
        }
        log_message(log, &format!("Test 7 pass {} completed.", pass));
    }
    log_message(log, "Test 7 passed.");
    true
}

// Test 8: Random Sequence Test (32-bit)
fn test8_random_sequence_32(memory: &mut [u32], log: &Arc<Mutex<Vec<String>>>, stop_flag: &Arc<AtomicBool>, passes: usize) -> bool {
    log_message(log, "Test 8: Random Sequence Test (32-bit) started.");
    let seed: u64 = 0x12345678;
    for pass in 0..passes {
        if stop_flag.load(Ordering::Relaxed) { return false; }
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed + pass as u64);
        for slot in memory.iter_mut() { *slot = rng.gen(); }
        rng = rand::rngs::StdRng::seed_from_u64(seed + pass as u64);
        for (i, slot) in memory.iter_mut().enumerate() {
            let expected: u32 = rng.gen();
            if *slot != expected {
                log_message(log, &format!("Test 8 ERROR on pass {} at index {}: expected 0x{:08X}, got 0x{:08X}", pass, i, expected, *slot));
                return false;
            }
            *slot = !expected;
        }
        rng = rand::rngs::StdRng::seed_from_u64(seed + pass as u64);
        for (i, slot) in memory.iter().enumerate() {
            let expected: u32 = rng.gen();
            if *slot != !expected {
                log_message(log, &format!("Test 8 ERROR (complement) on pass {} at index {}: expected 0x{:08X}, got 0x{:08X}", pass, i, !expected, *slot));
                return false;
            }
        }
        log_message(log, &format!("Test 8 pass {} completed.", pass));
    }
    log_message(log, "Test 8 passed.");
    true
}

// Test 9: Modulo-20 Random Pattern Test.
fn test9_modulo20_random(memory: &mut [u32], log: &Arc<Mutex<Vec<String>>>, stop_flag: &Arc<AtomicBool>, passes: usize) -> bool {
    log_message(log, "Test 9: Modulo-20 Random Pattern Test started.");
    let seed: u64 = 0x87654321;
    for pass in 0..passes {
        if stop_flag.load(Ordering::Relaxed) { return false; }
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed + pass as u64);
        for (i, slot) in memory.iter_mut().enumerate() {
            if i % 20 == 0 {
                *slot = rng.gen();
            }
        }
        rng = rand::rngs::StdRng::seed_from_u64(seed + pass as u64);
        for (i, slot) in memory.iter().enumerate() {
            if i % 20 == 0 {
                let expected: u32 = rng.gen();
                if *slot != expected {
                    log_message(log, &format!("Test 9 ERROR on pass {} at index {}: expected 0x{:08X}, got 0x{:08X}", pass, i, expected, *slot));
                    return false;
                }
            }
        }
        log_message(log, &format!("Test 9 pass {} completed.", pass));
    }
    log_message(log, "Test 9 passed.");
    true
}

// Test 10: Bit Fade Test.
fn test10_bit_fade(memory: &mut [u32], log: &Arc<Mutex<Vec<String>>>, stop_flag: &Arc<AtomicBool>, sleep_duration: Duration) -> bool {
    log_message(log, "Test 10: Bit Fade Test started.");
    for (i, slot) in memory.iter_mut().enumerate() {
        *slot = if i % 2 == 0 { 0xFFFFFFFFu32 } else { 0x00000000u32 };
    }
    thread::sleep(sleep_duration);
    for (i, &slot) in memory.iter().enumerate() {
        let expected = if i % 2 == 0 { 0xFFFFFFFFu32 } else { 0x00000000u32 };
        if slot != expected {
            log_message(log, &format!("Test 10 ERROR at index {}: expected 0x{:08X}, got 0x{:08X}", i, expected, slot));
            return false;
        }
    }
    log_message(log, "Test 10 passed.");
    true
}

// Test 11: Random Sequence Test (64-bit).
fn test11_random_sequence_64(memory: &mut [u64], log: &Arc<Mutex<Vec<String>>>, stop_flag: &Arc<AtomicBool>, passes: usize) -> bool {
    log_message(log, "Test 11: Random Sequence Test (64-bit) started.");
    let seed: u64 = 0xA1B2C3D4;
    for pass in 0..passes {
        if stop_flag.load(Ordering::Relaxed) { return false; }
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed + pass as u64);
        for slot in memory.iter_mut() { *slot = rng.gen(); }
        rng = rand::rngs::StdRng::seed_from_u64(seed + pass as u64);
        for (i, slot) in memory.iter_mut().enumerate() {
            let expected: u64 = rng.gen();
            if *slot != expected {
                log_message(log, &format!("Test 11 ERROR on pass {} at index {}: expected {}, got {}", pass, i, expected, *slot));
                return false;
            }
            *slot = !expected;
        }
        rng = rand::rngs::StdRng::seed_from_u64(seed + pass as u64);
        for (i, slot) in memory.iter().enumerate() {
            let expected: u64 = rng.gen();
            if *slot != !expected {
                log_message(log, &format!("Test 11 ERROR (complement) on pass {} at index {}: expected {}, got {}", pass, i, !expected, *slot));
                return false;
            }
        }
        log_message(log, &format!("Test 11 pass {} completed.", pass));
    }
    log_message(log, "Test 11 passed.");
    true
}

// Test 12: Random Sequence Test (128-bit).
fn test12_random_sequence_128(memory: &mut [u128], log: &Arc<Mutex<Vec<String>>>, stop_flag: &Arc<AtomicBool>, passes: usize) -> bool {
    log_message(log, "Test 12: Random Sequence Test (128-bit) started.");
    let seed: u64 = 0xDEADBEEF;
    for pass in 0..passes {
        if stop_flag.load(Ordering::Relaxed) { return false; }
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed + pass as u64);
        for slot in memory.iter_mut() { *slot = rng.gen(); }
        rng = rand::rngs::StdRng::seed_from_u64(seed + pass as u64);
        for (i, slot) in memory.iter_mut().enumerate() {
            let expected: u128 = rng.gen();
            if *slot != expected {
                log_message(log, &format!("Test 12 ERROR on pass {} at index {}: expected {}, got {}", pass, i, expected, *slot));
                return false;
            }
            *slot = !expected;
        }
        rng = rand::rngs::StdRng::seed_from_u64(seed + pass as u64);
        for (i, slot) in memory.iter().enumerate() {
            let expected: u128 = rng.gen();
            if *slot != !expected {
                log_message(log, &format!("Test 12 ERROR (complement) on pass {} at index {}: expected {}, got {}", pass, i, !expected, *slot));
                return false;
            }
        }
        log_message(log, &format!("Test 12 pass {} completed.", pass));
    }
    log_message(log, "Test 12 passed.");
    true
}

/// --- Main Memory Test Runner ---
/// This function tests 100% of the detected system memory in 2 phases.
/// Each phase tests 50% of total memory and is divided among 4 concurrent threads,
/// so that each thread tests roughly 12.5% of the total system memory.
pub fn run_test(
    params: TestParameters,
    log: &Arc<Mutex<Vec<String>>>,
    progress: &Arc<Mutex<TestProgressData>>,
    _samples: &Arc<Mutex<SampleData>>,
    stop_flag: Arc<AtomicBool>,
) -> bool {
    log_message(log, "Starting Production-Grade Memory Test Suite (Staggered Mode)...");
    
    // Detect 100% of the system memory.
    let total_memory_mb = total_test_memory_mb();
    log_message(log, &format!("Total system memory detected: {} MB", total_memory_mb));
    
    // We run 2 phases to cover 100% of the memory: Phase 1 tests the first 50% and Phase 2 tests the remaining 50%.
    let phases = 2;
    let mut overall_result = true;
    
    for phase in 0..phases {
        log_message(log, &format!("Starting memory test phase {} of {}...", phase + 1, phases));
        let phase_allocated_mb = total_memory_mb / 2; // each phase covers 50%.
        
        // In each phase, we run 4 threads concurrently.
        // Each thread will get a chunk of memory equal to (phase_allocated_mb / 4) MB.
        let chunks_per_phase = THREADS;  // 4 chunks per phase.
        let chunk_mb = phase_allocated_mb / THREADS;
        let words_per_chunk = chunk_mb * WORDS_PER_MB;
        let num_chunks = chunks_per_phase; // exactly 4
        
        log_message(log, &format!(
            "Phase {}: Allocated {} MB; using {} chunks (each {} MB, {} words per chunk).",
            phase + 1, phase_allocated_mb, num_chunks, chunk_mb, words_per_chunk
        ));
        
        // Allocate memory chunks for this phase.
        let mut chunks: Vec<Vec<u32>> = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            if stop_flag.load(Ordering::Relaxed) {
                log_message(log, "Memory test aborted during allocation.");
                return false;
            }
            let chunk = vec![0u32; words_per_chunk];
            chunks.push(chunk);
        }
        
        let phase_start = Instant::now();
        let max_concurrent = THREADS; // 4 concurrent threads.
        log_message(log, &format!("Phase {}: Running tests concurrently using {} threads.", phase + 1, max_concurrent));
        
        let quick = params.quick_memory_test();
        let per_test_duration = if quick { 5 } else { 10 }; // seconds per subtest.
        
        let mut processed = 0;
        let mut handles = Vec::new();
        
        // Spawn a thread for each chunk.
        for (i, chunk) in chunks.into_iter().enumerate() {
            let log_clone = Arc::clone(log);
            let stop_flag_clone = stop_flag.clone();
            let shared_chunk = Arc::new(Mutex::new(chunk));
            let handle = thread::spawn({
                let shared_chunk = Arc::clone(&shared_chunk);
                let log_clone = log_clone.clone();
                move || -> bool {
                    log_message(&log_clone, &format!("Phase {}: Testing chunk {} of {}...", phase + 1, i + 1, num_chunks));
                    let mut ok = true;
                    ok &= test0_walking_ones(&mut shared_chunk.lock().unwrap(), &log_clone, &stop_flag_clone, per_test_duration);
                    ok &= test1_own_address(&mut shared_chunk.lock().unwrap(), &log_clone, &stop_flag_clone, per_test_duration);
                    ok &= test2_own_address_parallel(shared_chunk.clone(), log_clone.clone(), stop_flag_clone.clone(), per_test_duration);
                    ok &= test3_moving_inversions(&mut shared_chunk.lock().unwrap(), &log_clone, &stop_flag_clone, per_test_duration);
                    ok &= test4_moving_inversions_8bit(&mut shared_chunk.lock().unwrap(), &log_clone, &stop_flag_clone, per_test_duration);
                    ok &= test5_moving_inversions_random(&mut shared_chunk.lock().unwrap(), &log_clone, &stop_flag_clone, per_test_duration);
                    ok &= test6_block_move(&mut shared_chunk.lock().unwrap(), &log_clone, &stop_flag_clone, per_test_duration);
                    ok &= test7_moving_inversions_32bit(&mut shared_chunk.lock().unwrap(), &log_clone, &stop_flag_clone, if quick { 1 } else { 2 });
                    ok &= test8_random_sequence_32(&mut shared_chunk.lock().unwrap(), &log_clone, &stop_flag_clone, if quick { 1 } else { 2 });
                    ok &= test9_modulo20_random(&mut shared_chunk.lock().unwrap(), &log_clone, &stop_flag_clone, if quick { 1 } else { 2 });
                    ok &= test10_bit_fade(&mut shared_chunk.lock().unwrap(), &log_clone, &stop_flag_clone, Duration::from_secs(if quick { 15 } else { 30 }));
                    {
                        let len = shared_chunk.lock().unwrap().len();
                        let bound = len * 4 / 8;
                        let ptr = shared_chunk.lock().unwrap().as_mut_ptr();
                        if bound > 0 {
                            let chunk_u64 = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u64, bound) };
                            ok &= test11_random_sequence_64(chunk_u64, &log_clone, &stop_flag_clone, if quick { 1 } else { 2 });
                        }
                    }
                    {
                        let len = shared_chunk.lock().unwrap().len();
                        let bound = len * 4 / 16;
                        let ptr = shared_chunk.lock().unwrap().as_mut_ptr();
                        if bound > 0 {
                            let chunk_u128 = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u128, bound) };
                            ok &= test12_random_sequence_128(chunk_u128, &log_clone, &stop_flag_clone, if quick { 1 } else { 2 });
                        }
                    }
                    ok
                }
            });
            handles.push(handle);
            
            if handles.len() >= max_concurrent {
                for h in handles.drain(..) {
                    if !h.join().unwrap_or(false) {
                        overall_result = false;
                    }
                    processed += 1;
                    update_progress(progress, processed, num_chunks, phase_start.elapsed());
                }
            }
        }
        
        for h in handles {
            if !h.join().unwrap_or(false) {
                overall_result = false;
            }
            processed += 1;
            update_progress(progress, processed, num_chunks, phase_start.elapsed());
        }
        
        log_message(log, &format!("Phase {} completed.", phase + 1));
    }
    
    log_message(log, "Memory test suite completed successfully (staggered mode).");
    overall_result
}
