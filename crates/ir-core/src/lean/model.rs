use crate::encoding::IR_VERSION;
use std::path::PathBuf;
use thiserror::Error;

/// One generated Lean module, ordered after every module that it imports.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderedLeanModule {
    pub module_name: String,
    pub relative_path: PathBuf,
    pub source: String,
}

/// The result of rendering one validated linked program as ordered Lean modules.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderedLeanProgram {
    pub modules: Vec<RenderedLeanModule>,
    pub root_module: String,
    pub ir_version: u32,
    pub linked_program_sha256: [u8; 32],
}

impl RenderedLeanProgram {
    pub fn ir_version(&self) -> u32 {
        self.ir_version
    }

    pub fn linked_program_sha256(&self) -> &[u8; 32] {
        &self.linked_program_sha256
    }
}

#[derive(Debug, Error)]
pub enum LeanEmissionError {
    #[error("stage {stage:?}, scope {scope:?}, node {node}: unsupported node kind {kind}")]
    UnsupportedNode { stage: String, scope: String, node: usize, kind: String },
    #[error(
        "stage {stage:?}, scope {scope:?}, node {node}, port {port}: unsupported wire type {wire_type}"
    )]
    UnsupportedWireType {
        stage: String,
        scope: String,
        node: usize,
        port: usize,
        wire_type: String,
    },
    #[error(
        "stage {stage:?}, scope {scope:?}, node {node}, port {port}: invalid concrete shape: {message}"
    )]
    InvalidShape { stage: String, scope: String, node: usize, port: usize, message: String },
    #[error("stage {stage:?}, scope {scope:?}, node {node}: invalid concrete parameter: {message}")]
    InvalidParameter { stage: String, scope: String, node: usize, message: String },
    #[error("cannot encode generated Lean data: {message}")]
    Encoding { message: String },
}

pub(crate) const LEAN_IR_VERSION: u32 = IR_VERSION;
