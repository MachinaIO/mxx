//! Owner-bound host boundary commands used by prepared graph lowering.
//!
//! This module deliberately contains no graph lookup.  The factory supplies
//! already-bound matrix owners and the concrete output contract at warmup;
//! replay only submits the resulting fixed native plan.

use mxx_primitives::matrix::{
    PolyMatrix,
    gpu_dcrt_poly::{
        GpuDCRTPolyMatrix, GpuPreparedConstCoeffReadback, GpuPreparedRnsReconstruction,
        GpuPreparedRnsUpload, PreparedPlanLayout,
    },
};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub(crate) struct PreparedHostCommandSpec {
    pub source: Option<Arc<GpuDCRTPolyMatrix>>,
    pub target: Option<Arc<GpuDCRTPolyMatrix>>,
    pub coefficient_index: usize,
    pub coefficient_count: usize,
    pub words_per_poly: usize,
    pub bytes_per_poly: usize,
    pub format: i32,
    pub transform_to_eval: bool,
    pub plan: PreparedPlanLayout,
}

pub(crate) enum PreparedHostCommand {
    Readback { command: Arc<GpuPreparedConstCoeffReadback>, values: Arc<Mutex<Box<[u64]>>> },
    Reconstruction { command: Arc<GpuPreparedRnsReconstruction> },
    Upload { command: Arc<GpuPreparedRnsUpload> },
}

pub(crate) fn bind_readback(spec: &PreparedHostCommandSpec) -> Result<PreparedHostCommand, String> {
    let source = Arc::clone(spec.source.as_ref().ok_or("readback source owner is missing")?);
    let command = GpuPreparedConstCoeffReadback::bind(
        Arc::clone(&source),
        spec.words_per_poly,
        spec.coefficient_index,
        spec.coefficient_count,
        spec.plan.clone(),
    )?;
    let values = Arc::new(Mutex::new(
        vec![0u64; source.size().0 * source.size().1 * spec.words_per_poly].into_boxed_slice(),
    ));
    Ok(PreparedHostCommand::Readback { command, values })
}

pub(crate) fn bind_reconstruction(
    spec: &PreparedHostCommandSpec,
) -> Result<PreparedHostCommand, String> {
    let source = Arc::clone(spec.source.as_ref().ok_or("reconstruction source owner is missing")?);
    let command = GpuPreparedRnsReconstruction::bind(
        source,
        spec.coefficient_index,
        spec.coefficient_count,
        spec.plan.clone(),
    )?;
    Ok(PreparedHostCommand::Reconstruction { command })
}

pub(crate) fn bind_upload(spec: &PreparedHostCommandSpec) -> Result<PreparedHostCommand, String> {
    let target = Arc::clone(spec.target.as_ref().ok_or("upload target owner is missing")?);
    let command = GpuPreparedRnsUpload::bind(
        target,
        spec.bytes_per_poly,
        spec.format,
        spec.transform_to_eval,
        spec.plan.clone(),
    )?;
    Ok(PreparedHostCommand::Upload { command })
}
