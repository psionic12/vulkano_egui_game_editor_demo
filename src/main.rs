use std::collections::HashMap;
use std::io::{BufReader, Cursor};
use std::mem::size_of;
use std::ops::Index;
use std::slice::from_raw_parts;
use std::sync::Arc;
use std::time::Instant;
use bytemuck::cast_slice;
use egui::{ScrollArea, TextEdit, TextStyle};
use egui_winit_vulkano::Gui;
use image::{GenericImageView, ImageFormat};
use tobj::LoadOptions;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, ImmutableBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SecondaryCommandBuffer, SubpassContents};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::Subpass;
use vulkano::swapchain::{AcquireError, SwapchainCreationError};
use vulkano::{SafeDeref, swapchain, sync};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::image::{ImageAccess, ImageDimensions, ImmutableImage, MipmapsCount};
use vulkano::image::view::ImageView;
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::shader::ShaderModule;
use vulkano::sync::{FenceSignalFuture, FlushError, GpuFuture};
use vulkano::format::Format;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use winit::event::{DeviceEvent, DeviceId, ElementState, Event, MouseButton, VirtualKeyCode, WindowEvent};
use winit::event::WindowEvent::KeyboardInput;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;
use crate::camera::Camera;
use vulkan_device::VulkanDevice;

mod camera;
mod vulkan_device;

fn main() {
    let event_loop = EventLoop::new();
    let render_device = VulkanDevice::new(&event_loop);

    #[repr(C)]
    #[derive(Default, Debug, Clone)]
    struct Position {
        position: [f32; 3],
    }
    vulkano::impl_vertex!(Position, position);

    #[repr(C)]
    #[derive(Default, Debug, Clone)]
    pub struct Normal {
        normal: [f32; 3],
    }
    vulkano::impl_vertex!(Normal, normal);

    #[repr(C)]
    #[derive(Default, Debug, Clone)]
    pub struct Texture {
        texture: [f32; 2],
    }
    vulkano::impl_vertex!(Texture, texture);

    let obj = include_bytes!("cube.obj").to_vec();
    let cursor = Cursor::new(obj);

    let mut reader = BufReader::new(cursor);
    let (models, materials)
        = tobj::load_obj_buf(&mut reader,
                             &LoadOptions {
                                 single_index: true,
                                 triangulate: true,
                                 ignore_points: false,
                                 ignore_lines: false,
                             },
                             |_| { Err(tobj::LoadError::OpenFileFailed) },
    ).unwrap();

    let (vertex_buffer, vertex_future) = ImmutableBuffer::from_iter(
        models.first().unwrap().mesh.positions.iter().cloned(),
        BufferUsage::vertex_buffer_transfer_destination(),
        render_device.get_transfer_queue(),
    ).unwrap();

    let (normal_buffer, normal_future) = ImmutableBuffer::from_iter(
        models.first().unwrap().mesh.normals.iter().cloned(),
        BufferUsage::vertex_buffer_transfer_destination(),
        render_device.get_transfer_queue(),
    ).unwrap();


    let (texcoords_buffer, texcoords_future) = ImmutableBuffer::from_iter(
        models.first().unwrap().mesh.texcoords.iter().cloned(),
        BufferUsage::vertex_buffer_transfer_destination(),
        render_device.get_transfer_queue(),
    ).unwrap();

    let (index_buffer, index_future) = ImmutableBuffer::from_iter(
        models.first().unwrap().mesh.indices.iter().cloned(),
        BufferUsage::index_buffer_transfer_destination(),
        render_device.get_transfer_queue(),
    ).unwrap();

    let (texture, tex_future) = {
        let png_bytes = include_bytes!("cube_texture.png").to_vec();
        let cursor = Cursor::new(png_bytes);

        let mut reader = image::io::Reader::with_format(cursor, ImageFormat::Png)
            .decode().unwrap();
        let image_data = reader.flipv().into_rgba8();
        let dimensions = ImageDimensions::Dim2d {
            width: image_data.width(),
            height: image_data.height(),
            array_layers: 1,
        };

        let (image, future)
            = ImmutableImage::from_iter(
            image_data.iter().cloned(),
            dimensions,
            MipmapsCount::One,
            Format::R8G8B8A8_SRGB,
            render_device.get_transfer_queue(),
        ).unwrap();
        (ImageView::new(image).unwrap(), future)
    };

    let sampler = Sampler::new(
        render_device.get_device(),
        Filter::Linear,
        Filter::Linear,
        MipmapMode::Nearest,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        0.0,
        1.0,
        0.0,
        0.0,
    ).unwrap();

    let init_future = vertex_future
        .join(normal_future)
        .join(texcoords_future)
        .join(index_future)
        .join(tex_future)
        .then_signal_fence_and_flush().unwrap();

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
				#version 450

                layout(location = 0) in vec3 position;
                layout(location = 1) in vec3 normal;
                layout(location = 2) in vec2 texture;

                layout(location = 0) out vec3 v_normal;
                layout(location = 1) out vec2 v_texture;

                layout(push_constant) uniform Data {
                    mat4 world;
                    mat4 view;
                    mat4 proj;
                } uniforms;

                void main() {
                    mat4 worldview = uniforms.view * uniforms.world;
                    v_normal = transpose(inverse(mat3(worldview))) * normal;
                    v_texture = texture;
                    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
                }

			"
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
				#version 450

                layout(location = 0) in vec3 v_normal;
                layout(location = 1) in vec2 tex_coords;

                layout(location = 0) out vec4 f_color;

                layout(set = 0, binding = 0) uniform sampler2D tex;

                const vec3 LIGHT = vec3(0.0, 0.0, 1.0);

                void main() {
                    float brightness = dot(normalize(v_normal), normalize(LIGHT));
                    vec3 dark_color = vec3(0.6, 0.0, 0.0);
                    vec3 regular_color = vec3(1.0, 0.0, 0.0);

                    f_color = texture(tex, tex_coords) * vec4(mix(dark_color, regular_color, brightness), 1.0);
                }

			"
        }
    }

    let vs = vs::load(render_device.get_device()).unwrap();
    let fs = fs::load(render_device.get_device()).unwrap();

    const CODE: &str = r#"
# Some markup
```
let mut gui = Gui::new(renderer.surface(), renderer.queue());
```
Vulkan(o) is hard, that I know...
"#;

    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new()
            .vertex::<Position>()
            .vertex::<Normal>()
            .vertex::<Texture>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(render_device.get_render_pass(), 0).unwrap())
        .build(render_device.get_device())
        .unwrap();

    let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
    let mut set_builder = PersistentDescriptorSet::start(layout.clone());

    set_builder
        .add_sampled_image(texture.clone(), sampler.clone())
        .unwrap();
    // .add_buffer(uniform_buffer.clone()).unwrap();

    let set = set_builder.build().unwrap();

    let mut previous_frame_end = Some(init_future.boxed());
    let mut start = Instant::now();

    let mut camera = Camera::default();
    camera.transform.position = ultraviolet::Vec3::new(0.0, 2.0, 5.0);

    let mut w_pressed = ElementState::Released;
    let mut s_pressed = ElementState::Released;
    let mut a_pressed = ElementState::Released;
    let mut d_pressed = ElementState::Released;
    let mut lb_pressed = ElementState::Released;
    let mut rb_pressed = ElementState::Released;

    let camera_speed = 0.05;
    let front = ultraviolet::Vec3::new(0.0, 0.0, -1.0);
    let right = ultraviolet::Vec3::new(1.0, 0.0, 0.0);
    let up = ultraviolet::Vec3::new(0.0, 1.0, 0.0);

    let mut code = CODE.to_owned();
    let mut gui = Gui::new_with_subpass(render_device.get_surface(), render_device.get_render_queue(), Subpass::from(render_device.get_render_pass(), 1).unwrap());

    event_loop.run(move |event, _, control_flow| {
        gui.update(&event);
        match event {
            Event::DeviceEvent { device_id, event } => {
                match event {
                    DeviceEvent::MouseMotion { delta } => {
                        if lb_pressed == ElementState::Pressed {
                            let rotation = &mut camera.transform.rotation;
                            rotation.x = (rotation.x + delta.1.to_radians() as f32).clamp(-89.9f32.to_radians(), 89.0f32.to_radians());
                            rotation.y -= (delta.0 as f32).to_radians();
                            rotation.z = 0.0;
                        }
                    }
                    _ => {}
                }
            }
            Event::WindowEvent { window_id, event } => {
                match event {
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    WindowEvent::KeyboardInput { device_id, input, is_synthetic } => {
                        if let Some(virtual_keycode) = input.virtual_keycode {
                            match virtual_keycode {
                                VirtualKeyCode::W => {
                                    w_pressed = input.state;
                                }
                                VirtualKeyCode::S => {
                                    s_pressed = input.state;
                                }
                                VirtualKeyCode::A => {
                                    a_pressed = input.state;
                                }
                                VirtualKeyCode::D => {
                                    d_pressed = input.state;
                                }
                                _ => {}
                            }
                        }
                    }
                    WindowEvent::MouseInput { device_id, state, button, .. } => {
                        match button {
                            MouseButton::Left => {
                                lb_pressed = state;
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
            Event::RedrawEventsCleared => {
                // let now = Instant::now();
                // let duration = now - start;
                // total += duration.as_micros();
                // loop_counts += 1;
                // start = now;

                previous_frame_end.as_mut().unwrap().cleanup_finished();

                gui.immediate_ui(|gui| {
                    let ctx = gui.context();
                    egui::Window::new("Settings").show(&ctx, |ui| {
                        &ctx.settings_ui(ui);
                    });

                });

                let elapsed = start.elapsed();
                let rotation =
                    elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
                // let rotation = Matrix3::from_angle_y(Rad(rotation as f32));
                let rotation = ultraviolet::Mat4::from_rotation_y(rotation as f32);

                if w_pressed == ElementState::Pressed {
                    camera.transform.position += camera_speed * front;
                }
                if s_pressed == ElementState::Pressed {
                    camera.transform.position -= camera_speed * front;
                }
                if a_pressed == ElementState::Pressed {
                    camera.transform.position -= camera_speed * right;
                }
                if d_pressed == ElementState::Pressed {
                    camera.transform.position += camera_speed * right;
                }

                let (view, proj) = camera.view_and_projection();

                let uniform_data = vs::ty::Data {
                    world: rotation.into(),
                    view: (view).into(),
                    proj: proj.into(),
                };

                let (image_num, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(render_device.get_swap_chain(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()];

                let mut builder = AutoCommandBufferBuilder::primary(
                    render_device.get_device(),
                    render_device.get_render_queue().family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                    .unwrap();

                builder
                    .begin_render_pass(
                        render_device.get_frame_buffer()[image_num].clone(),
                        SubpassContents::SecondaryCommandBuffers,
                        clear_values,
                    )
                    .unwrap();
                let mut secondary_builder = AutoCommandBufferBuilder::secondary_graphics(
                    render_device.get_device(),
                    render_device.get_render_queue().family(),
                    CommandBufferUsage::MultipleSubmit,
                    Subpass::from(render_device.get_render_pass(), 0).unwrap(),
                )
                    .unwrap();
                secondary_builder
                    .push_constants(pipeline.layout().clone(), 0, uniform_data)
                    .bind_pipeline_graphics(pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        0,
                        set.clone(),
                    )
                    .set_viewport(0, [render_device.get_viewport()])
                    .bind_vertex_buffers(0, (vertex_buffer.clone(), normal_buffer.clone(), texcoords_buffer.clone()))
                    .bind_index_buffer(index_buffer.clone())
                    .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0)
                    .unwrap();
                builder.execute_commands(secondary_builder.build().unwrap()).unwrap()
                    .next_subpass(SubpassContents::SecondaryCommandBuffers).unwrap()
                    .execute_commands(gui.draw_on_subpass_image(render_device.get_images()[0].dimensions().width_height())).unwrap()
                    .end_render_pass()
                    .unwrap();

                // Finish building the command buffer by calling `build`.
                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(render_device.get_render_queue(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(render_device.get_render_queue(), render_device.get_swap_chain(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        future.wait(None);
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        previous_frame_end = Some(sync::now(render_device.get_device()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(render_device.get_device()).boxed());
                    }
                }
            }
            _ => (),
        }
    });
}
