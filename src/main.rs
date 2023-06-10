#![allow(warnings)]

use color_eyre::eyre;
use ndarray::{array, concatenate, s, Array1, Axis};
use ort::{
    tensor::{DynOrtTensor, FromArray, InputTensor, OrtOwnedTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder,
};
use rand::Rng;
use std::path::PathBuf;
use std::sync::Arc;

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();

    let mut rng = rand::thread_rng();

    let environment = Arc::new(
        Environment::builder()
            .with_name("CRAFT")
            .with_execution_providers([ExecutionProvider::cuda()])
            .build()?,
    );

    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .with_model_from_file("craft_detector.onnx")?;

    let input_image = PathBuf::from("./examples/english.png");
    assert!(input_image.is_file());

    // TODO: create an rgb image using image crate

    // let array = tokens.clone().insert_axis(Axis(0)).into_shape((1, 1, *n_tokens)).unwrap();
    // let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> = session.run([InputTensor::from_array(array.into_dyn())])?;
    // let generated_tokens: OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();
    // let generated_tokens = generated_tokens.view();

    // dbg!(&session);

    println!("Hello, world!");
    Ok(())
}
