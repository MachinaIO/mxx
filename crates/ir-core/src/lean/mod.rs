//! Deterministic Rust-to-Lean data emission for validated IR programs.

mod model;
mod render;

pub use model::{LeanEmissionError, RenderedLeanModule, RenderedLeanProgram};
pub use render::{
    render_child_input_hop, render_child_input_path, render_lean_program,
    render_parallel_output_hop, render_structural_value_route,
};
