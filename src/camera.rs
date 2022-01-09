pub struct Transform {
    pub position: ultraviolet::Vec3,
    pub rotation: ultraviolet::Vec3,
    pub scale: ultraviolet::Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: ultraviolet::Vec3::default(),
            rotation: ultraviolet::Vec3::default(),
            scale: ultraviolet::Vec3::new(1.0, 1.0, 1.0),
        }
    }
}

pub struct Camera {
    pub transform: Transform,
    pub fov: f32,
    pub aspect: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            transform: Transform::default(),
            fov: 45.0,
            aspect: 0.0,
        }
    }
}

impl Camera {
    pub fn view_and_projection(&self) -> (ultraviolet::Mat4, ultraviolet::Mat4) {
        let transform = &self.transform;
        let mut view = ultraviolet::Mat4::identity();
        view.translate(&(-transform.position));
        view = ultraviolet::Mat4::from_euler_angles(
            self.transform.rotation.x,
            self.transform.rotation.y,
            self.transform.rotation.z,
        ) * view;

        let projection =
            ultraviolet::projection::perspective_vk(self.fov, 800.0 / 600.0, 0.1, 100.0);
        (view, projection)
    }
}
