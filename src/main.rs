#![allow(warnings)]

use color_eyre::eyre;
use image::{imageops, io::Reader as ImageReader, DynamicImage, ImageBuffer, Pixel, Rgb};
use ndarray::{array, concatenate, s, Array1, ArrayView, Axis, ShapeBuilder};
// use ndarray_image::{NdColor, NdImage};
use ort::{
    tensor::{DynOrtTensor, FromArray, InputTensor, OrtOwnedTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder,
};
use rand::Rng;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn resize_aspect_ratio<I>(
    img: &I,
    max_dim: usize,
    mag_ratio: f32,
) -> (
    ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>,
    f32,
)
where
    I: image::GenericImageView,
    I::Pixel: 'static,
    <I::Pixel as Pixel>::Subpixel: 'static,
{
    let (height, width) = img.dimensions();

    // magnify image size
    let target_size = mag_ratio * height.max(width) as f32;
    let target_size = target_size as usize;

    // set original image size
    let target_size = target_size.min(max_dim);

    let ratio = target_size as f32 / height.max(width) as f32;

    let target_h = (height as f32 * ratio) as u32;
    let target_w = (width as f32 * ratio) as u32;
    let new_img = imageops::resize(img, target_w, target_h, imageops::FilterType::Lanczos3);

    // make canvas whose size is a multiple of 32
    let mut target_h32 = target_h;
    let mut target_w32 = target_w;
    if target_h % 32 != 0 {
        target_h32 = target_h + (32 - target_h % 32);
    }
    if target_w % 32 != 0 {
        target_w32 = target_w + (32 - target_w % 32);
    }

    let mut result = ImageBuffer::new(target_w32, target_h32);
    imageops::overlay(&mut result, &new_img, 0, 0);

    let size_heatmap = (target_w32 / 2, target_h32 / 2);

    (result, ratio)
}

pub struct NdImage<T>(pub T);

pub type NdColor<'a, A = u8> = ArrayView<'a, A, ndarray::Ix3>;

impl<'a, C, P: 'static, A: 'static> Into<NdColor<'a, A>> for NdImage<&'a ImageBuffer<P, C>>
where
    A: image::Primitive,
    P: Pixel<Subpixel = A>,
    C: std::ops::Deref<Target = [P::Subpixel]> + AsRef<[A]>,
{
    fn into(self) -> NdColor<'a, A> {
        let NdImage(image) = self;
        let (width, height) = image.dimensions();
        let (width, height) = (width as usize, height as usize);
        let channels = P::CHANNEL_COUNT as usize;
        let slice: &'a [A] = unsafe { std::mem::transmute(image.as_flat_samples().as_slice()) };
        ArrayView::from_shape(
            (height, width, channels).strides((width * channels, channels, 1)),
            slice,
        )
        .unwrap()
    }
}

/// Note: for logs, run with `RUST_LOG="ort=debug" cargo run`
fn main() -> eyre::Result<()> {
    let start = Instant::now();
    tracing_subscriber::fmt::init();

    let mut rng = rand::thread_rng();

    let environment = Arc::new(
        Environment::builder()
            .with_name("CRAFT")
            // .with_execution_providers([ExecutionProvider::cuda()])
            .with_execution_providers([ExecutionProvider::cpu()])
            .build()?,
    );

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .with_model_from_file(manifest_dir.join("craft_detector.onnx"))?;

    let input_image = manifest_dir.join("examples/english.png");
    assert!(input_image.is_file());

    // create an rgb image using image crate
    let img = ImageReader::open(input_image)?.decode()?;
    let rgb_img: ImageBuffer<Rgb<f32>, Vec<f32>> = img.into_rgb32f();
    dbg!(&rgb_img.dimensions());

    let (rgb_img, target_ratio) = resize_aspect_ratio(&rgb_img, 2560, 1.0);
    dbg!(&rgb_img.dimensions());

    // TODO: more preprocessing on the input image (e.g. channels)

    // convert image to tensor input
    let rgb_img: NdImage<&ImageBuffer<Rgb<f32>, Vec<f32>>> = NdImage(&rgb_img);
    let rgb_img_tensor: NdColor<f32> = rgb_img.into();

    let rgb_img_tensor = rgb_img_tensor.reversed_axes();
    let rgb_img_tensor = ndarray::stack![Axis(0), rgb_img_tensor];
    dbg!(&rgb_img_tensor.shape());

    // run inference on the model
    let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> =
        session.run([InputTensor::from_array(
            rgb_img_tensor.into_owned().into_dyn(),
        )])?;

    // get the output
    let output: OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();
    dbg!(&output.view().shape());

    // TODO: decode the model output

    println!("took {:?}", start.elapsed());
    Ok(())
}
