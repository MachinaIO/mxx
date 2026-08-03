//! GPU backend binding for the Diamond iO graph compiler.

use super::graph::DiamondIoPoly;
use mxx_primitives::{matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix, poly::dcrt::gpu::GpuDCRTPoly};

impl DiamondIoPoly for GpuDCRTPoly {
    type Matrix = GpuDCRTPolyMatrix;
}
