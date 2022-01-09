use std::sync::Arc;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageAccess, ImageUsage, SwapchainImage};
use vulkano::instance::Instance;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::{Framebuffer, RenderPass};
use vulkano::swapchain::{Surface, Swapchain};
use vulkano::Version;
use vulkano_win::VkSurfaceBuild;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

pub struct VulkanDevice {
    device: Arc<Device>,
    render_queue: Arc<Queue>,
    transfer_queue: Arc<Queue>,
    swap_chain: Arc<Swapchain<Window>>,
    images: Vec<Arc<SwapchainImage<Window>>>,
    viewport: Viewport,
    render_pass: Arc<RenderPass>,
    frame_buffers: Vec<Arc<Framebuffer>>,
    surface: Arc<Surface<Window>>,
}

impl VulkanDevice {
    pub fn get_device(&self) -> Arc<Device> {
        self.device.clone()
    }

    pub fn get_swap_chain(&self) -> Arc<Swapchain<Window>> {
        self.swap_chain.clone()
    }

    pub fn get_images(&self) -> &[Arc<SwapchainImage<Window>>] {
        self.images.as_slice()
    }

    pub fn get_render_pass(&self) -> Arc<RenderPass> {
        self.render_pass.clone()
    }

    pub fn get_render_queue(&self) -> Arc<Queue> {
        self.render_queue.clone()
    }

    pub fn get_transfer_queue(&self) -> Arc<Queue> {
        self.transfer_queue.clone()
    }

    pub fn get_viewport(&self) -> Viewport {
        self.viewport.clone()
    }

    pub fn get_frame_buffer(&self) -> &[Arc<Framebuffer>] {
        self.frame_buffers.as_slice()
    }

    pub fn get_surface(&self) -> Arc<Surface<Window>> {
        self.surface.clone()
    }
}

impl VulkanDevice {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        let required_extensions = vulkano_win::required_extensions();
        let instance = Instance::new(None, Version::V1_1, &required_extensions, None).unwrap();
        let surface = WindowBuilder::new()
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap();
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };
        let (physical_device, graphic_queue_family) = PhysicalDevice::enumerate(&instance)
            .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
            .filter_map(|p| {
                p.queue_families()
                    .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
                    .map(|q| (p, q))
            })
            .min_by_key(|(p, _)| {
                // We assign a better score to device types that are likely to be faster/better.
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                }
            })
            .unwrap();

        let mut queue_families_and_priorities = Vec::new();
        queue_families_and_priorities.push((graphic_queue_family, 1.0));

        let transfer_queue_family = physical_device
            .queue_families()
            .find(|&q| q.explicitly_supports_transfers());

        if let Some(transfer_queue_family) = transfer_queue_family {
            queue_families_and_priorities.push((transfer_queue_family, 1.0));
        } else if graphic_queue_family.queues_count() > 1 {
            queue_families_and_priorities.push((graphic_queue_family, 0.9));
        }

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );
        let (device, mut queues) = Device::new(
            physical_device,
            &Features::none(),
            // Some devices require certain extensions to be enabled if they are present
            // (e.g. `khr_portability_subset`). We add them to the device extensions that we're going to
            // enable.
            &physical_device
                .required_extensions()
                .union(&device_extensions),
            queue_families_and_priorities.iter().cloned(),
        )
        .unwrap();
        let render_queue = queues.next().unwrap();
        let transfer_queue = if let Some(transfer_queue) = queues.next() {
            transfer_queue
        } else {
            render_queue.clone()
        };
        let (swap_chain, images) = {
            let caps = surface.capabilities(physical_device).unwrap();
            let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
            let format = caps.supported_formats[0].0;
            let dimensions: [u32; 2] = surface.window().inner_size().into();
            Swapchain::start(device.clone(), surface.clone())
                .num_images(caps.min_image_count)
                .format(format)
                .dimensions(dimensions)
                .usage(ImageUsage::color_attachment())
                .sharing_mode(&render_queue)
                .composite_alpha(composite_alpha)
                .build()
                .unwrap()
        };

        let render_pass = vulkano::ordered_passes_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swap_chain.format(),
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16_UNORM,
                    samples: 1,
                }
            },
            passes: [
                    { color: [color], depth_stencil: {depth}, input: [] },
                    { color: [color], depth_stencil: {}, input: [] }]
        )
        .unwrap();

        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [0.0, 0.0],
            depth_range: 0.0..1.0,
        };

        let dimensions = images[0].dimensions().width_height();
        viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

        let depth_buffer = ImageView::new(
            AttachmentImage::transient(device.clone(), dimensions, Format::D16_UNORM).unwrap(),
        )
        .unwrap();

        let frame_buffers = images
            .iter()
            .map(|image| {
                let view = ImageView::new(image.clone()).unwrap();
                Framebuffer::start(render_pass.clone())
                    .add(view)
                    .unwrap()
                    .add(depth_buffer.clone())
                    .unwrap()
                    .build()
                    .unwrap()
            })
            .collect::<Vec<_>>();

        return Self {
            device,
            render_queue,
            swap_chain,
            images,
            viewport,
            render_pass,
            frame_buffers,
            surface,
            transfer_queue,
        };
    }
}
