//! GPU Stress Test Module
//!
//! This module launches a full-screen window that displays a Fibonacci (golden) spiral,
//! computed on the CPU and rendered as a continuous line strip. Meanwhile, a compute shader
//! continuously multiplies two 64×64 matrices (each element of the product should equal 64.0)
//! to stress the GPU. Every 60 frames, the GPU temperature is logged and the matrix multiplication
//! result is read back; if any artifact is detected or if the GPU temperature exceeds 95°C,
//! the test is terminated immediately. Pressing Escape exits the application.

#[path = "../common.rs"]
mod common;

use std::error::Error;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use futures::executor::block_on;
use futures::channel::oneshot;
use futures::FutureExt; // Import FutureExt for now_or_never()
use wgpu::util::DeviceExt;
use winit::{
    event::{ElementState, Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Fullscreen, WindowBuilder},
};
use regex::Regex;

// ---------- CPU Spiral Vertex Computation ----------

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
}

/// Computes the Fibonacci (golden) spiral vertices using:
///     r = a * exp(b * theta)
/// with a = 0.3, b = 0.3063, and maps them into normalized device coordinates.
fn compute_spiral_vertices(time: f32) -> Vec<Vertex> {
    let a: f32 = 0.3;
    let b: f32 = 0.3063;
    let theta_max = 2.0 * time + 1.0;
    let dtheta = 0.05;

    let mut vertices = Vec::new();
    let r_max = a * (b * theta_max).exp();
    let world_range = r_max * 1.2;

    let mut theta = 0.0;
    while theta <= theta_max {
        let r = a * (b * theta).exp();
        let x = r * theta.cos();
        let y = r * theta.sin();
        let ndc_x = x / world_range;
        let ndc_y = y / world_range;
        vertices.push(Vertex { position: [ndc_x, ndc_y] });
        theta += dtheta;
    }
    vertices
}

// ---------- GPU Temperature Check using sensors ----------
fn get_gpu_temp() -> Option<f64> {
    let output = Command::new("sensors").output().ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let re = Regex::new(r"GPU\s+Temp:\s+\+([\d\.]+)°C").ok()?;
    for line in stdout.lines() {
        if let Some(cap) = re.captures(line) {
            if let Some(ts) = cap.get(1) {
                if let Ok(temp) = ts.as_str().parse::<f64>() {
                    return Some(temp);
                }
            }
        }
    }
    None
}

// ---------- Main GPU Work (Render + Compute) ----------
pub async fn async_run_test(_stop_flag: Arc<AtomicBool>) -> Result<(), Box<dyn Error>> {
    println!("GPU Test: Starting async_run_test (Fibonacci Spiral + MatMul Stress).");

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("GPU Stress Test – Spiral & MatMul")
        .with_fullscreen(Some(Fullscreen::Borderless(None)))
        .build(&event_loop)?;

    let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
    let surface  = unsafe { instance.create_surface(&window) };
    let adapter  = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }).await.ok_or("No appropriate adapter found")?;
    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("GPU Device"),
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
        },
        None,
    ).await?;

    let size = window.inner_size();
    let surface_format = surface.get_supported_formats(&adapter)[0];
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
    };
    surface.configure(&device, &config);

    // ---------- Spiral Rendering Setup ----------
    let vertex_buffer_size = 10000 * std::mem::size_of::<Vertex>() as u64;
    let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Vertex Buffer"),
        size: vertex_buffer_size,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let vs_src = r#"
        @vertex
        fn vs_main(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
            return vec4<f32>(pos, 0.0, 1.0);
        }
    "#;
    let fs_src = r#"
        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 1.0, 1.0, 1.0);
        }
    "#;
    let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Spiral Vertex Shader"),
        source: wgpu::ShaderSource::Wgsl(vs_src.into()),
    });
    let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Spiral Fragment Shader"),
        source: wgpu::ShaderSource::Wgsl(fs_src.into()),
    });
    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Spiral Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vs_module,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                }],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &fs_module,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineStrip,
            strip_index_format: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    // ---------- Matrix Multiplication Compute Setup ----------
    let matrix_dim = 64;
    let matrix_elements = matrix_dim * matrix_dim;
    let matrix_buffer_size = (matrix_elements * std::mem::size_of::<f32>()) as u64;
    let a_data = vec![1.0f32; matrix_elements];
    let b_data = vec![1.0f32; matrix_elements];
    let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matrix Buffer A"),
        contents: bytemuck::cast_slice(&a_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matrix Buffer B"),
        contents: bytemuck::cast_slice(&b_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buffer_c = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix Buffer C"),
        size: matrix_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let compute_shader_src = r#"
        struct Matrix {
            elements: array<f32, 4096>,
        };
        @group(0) @binding(0)
        var<storage, read> A: Matrix;
        @group(0) @binding(1)
        var<storage, read> B: Matrix;
        @group(0) @binding(2)
        var<storage, read_write> C: Matrix;

        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let row = id.y;
            let col = id.x;
            var sum: f32 = 0.0;
            for (var k: u32 = 0u; k < 64u; k = k + 1u) {
                let indexA = row * 64u + k;
                let indexB = k * 64u + col;
                sum = sum + A.elements[indexA] * B.elements[indexB];
            }
            C.elements[row * 64u + col] = sum;
        }
    "#;
    let compute_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(compute_shader_src.into()),
    });
    let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&compute_bind_group_layout],
        push_constant_ranges: &[],
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader_module,
        entry_point: "main",
    });
    let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Compute Bind Group"),
        layout: &compute_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_c.as_entire_binding(),
            },
        ],
    });

    // ---------- Main Loop ----------
    let start_time = Instant::now();
    let mut frame_count: u32 = 0;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(key) = input.virtual_keycode {
                        if key == VirtualKeyCode::Escape && input.state == ElementState::Pressed {
                            *control_flow = ControlFlow::Exit;
                        }
                    }
                },
                WindowEvent::Resized(new_size) => {
                    let new_config = wgpu::SurfaceConfiguration {
                        width: new_size.width,
                        height: new_size.height,
                        ..config
                    };
                    surface.configure(&device, &new_config);
                },
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                },
                _ => {}
            },
            Event::MainEventsCleared => {
                let time_sec = start_time.elapsed().as_secs_f32();
                // ---------- Update Spiral Vertex Buffer ----------
                let vertices = compute_spiral_vertices(time_sec);
                let vertex_data = bytemuck::cast_slice(&vertices);
                queue.write_buffer(&vertex_buffer, 0, vertex_data);

                // ---------- Render Pass: Draw Spiral ----------
                let frame = match surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(e) => {
                        eprintln!("Failed to acquire next texture: {:?}", e);
                        *control_flow = ControlFlow::Exit;
                        return;
                    }
                };
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: None,
                    });
                    render_pass.set_pipeline(&render_pipeline);
                    render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    render_pass.draw(0..(vertices.len() as u32), 0..1);
                }
                // ---------- Compute Pass: Perform Matrix Multiplication ----------
                {
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Compute Pass"),
                    });
                    compute_pass.set_pipeline(&compute_pipeline);
                    compute_pass.set_bind_group(0, &compute_bind_group, &[]);
                    compute_pass.dispatch_workgroups(8, 8, 1);
                }
                queue.submit(Some(encoder.finish()));
                frame.present();
                device.poll(wgpu::Maintain::Poll);

                frame_count += 1;
                // ---------- Error Checks Every 60 Frames ----------
                if frame_count % 60 == 0 {
                    // Log GPU Temperature.
                    if let Some(temp) = get_gpu_temp() {
                        println!("GPU Temp: {:.1} °C", temp);
                        if temp > 95.0 {
                            eprintln!("Error: GPU Temperature exceeded 95°C ({:.1}°C).", temp);
                            *control_flow = ControlFlow::Exit;
                        }
                    }
                    // Artifact Check: For matrices of ones, product should be 64.
                    let buffer_size = matrix_buffer_size;
                    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Readback Buffer"),
                        size: buffer_size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let mut check_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Artifact Check Encoder"),
                    });
                    check_encoder.copy_buffer_to_buffer(&buffer_c, 0, &readback_buffer, 0, buffer_size);
                    queue.submit(Some(check_encoder.finish()));
                    
                    let readback_slice = readback_buffer.slice(..);
                    let (sender, receiver) = oneshot::channel();
                    readback_slice.map_async(wgpu::MapMode::Read, move |result| {
                        let _ = sender.send(result);
                    });
                    device.poll(wgpu::Maintain::Poll);
                    if let Some(mapping_result) = receiver.now_or_never() {
                        match mapping_result {
                            Ok(Ok(())) => {
                                let data = readback_buffer.slice(..).get_mapped_range();
                                let results: &[f32] = bytemuck::cast_slice(&data);
                                for &val in results.iter() {
                                    if (val - 64.0).abs() > 0.01 {
                                        eprintln!("Error: Matrix multiplication artifact detected (value: {}).", val);
                                        *control_flow = ControlFlow::Exit;
                                        break;
                                    }
                                }
                                drop(data);
                                readback_buffer.unmap();
                            },
                            _ => {
                                eprintln!("Error: Failed to map readback buffer for artifact check.");
                                *control_flow = ControlFlow::Exit;
                            }
                        }
                    }
                }
            },
            _ => {}
        }
    });
}

fn main() {
    let stop_flag = Arc::new(AtomicBool::new(false));
    match pollster::block_on(async_run_test(stop_flag)) {
        Ok(_) => println!("GPU Stress Test completed successfully."),
        Err(e) => {
            eprintln!("GPU Stress Test failed with error: {:?}", e);
            std::process::exit(1);
        }
    }
}
