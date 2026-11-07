//! A diagnostic record of the graph values each lowered Program step produced.
//!
//! [`start`] begins recording on this thread; every Program lowering whose evaluator implements
//! [`super::lowering::ProgramLoweringContext::trace_output`] then appends one [`TraceEntry`] per
//! scalar step output, and [`take`] ends the recording. Proof generators resolve the recorded
//! construction handles in the frozen graphs (`mxx_ir_core::graph::Graph::producer_map`) to know
//! which graph wires carry each Program wire.

use mxx_ir_core::graph::ValueHandle;
use std::cell::RefCell;

/// The graph values of one lowered scalar step output.
#[derive(Clone, Debug)]
pub struct TraceEntry {
    /// The lowering this output belongs to: its Program and its occurrence.
    pub lowering: usize,
    /// The Program wire of the output.
    pub output: u32,
    /// The loop indices of the output's site.
    pub indices: Vec<String>,
    /// The Program step that produced the output (its `Debug` form).
    pub step: String,
    /// The named graph values of the output (row components, key blocks, payload).
    pub values: Vec<(String, ValueHandle)>,
}

thread_local! {
    static RECORD: RefCell<Option<(usize, Vec<TraceEntry>)>> = const { RefCell::new(None) };
}

/// Start recording on this thread, discarding any previous record.
pub fn start() {
    RECORD.with(|record| *record.borrow_mut() = Some((0, Vec::new())));
}

/// Stop recording and return the entries.
pub fn take() -> Vec<TraceEntry> {
    RECORD.with(|record| record.borrow_mut().take().map(|(_, entries)| entries).unwrap_or_default())
}

/// Whether this thread is recording.
pub fn recording() -> bool {
    RECORD.with(|record| record.borrow().is_some())
}

/// A stable key of a lowered Program: the first bytes of its canonical identity, so the same
/// Program lowered in different stages gets the same key in every run.
pub fn lowering_key(program: &super::ExponentLutProgram) -> usize {
    let id = program.id().0;
    usize::from_le_bytes(id[..std::mem::size_of::<usize>()].try_into().expect("id length"))
}

/// A fresh lowering occurrence number, while recording.
pub fn next_lowering() -> usize {
    RECORD.with(|record| match record.borrow_mut().as_mut() {
        Some((next, _)) => {
            *next += 1;
            *next
        }
        None => 0,
    })
}

/// Append an entry, while recording.
pub fn push(entry: TraceEntry) {
    RECORD.with(|record| {
        if let Some((_, entries)) = record.borrow_mut().as_mut() {
            entries.push(entry);
        }
    });
}
