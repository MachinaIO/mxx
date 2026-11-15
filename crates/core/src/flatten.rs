//! Flattening a validated program to one RV64IM proof image.
//!
//! The image runs on the mathcal-A machine (see [`crate::rv64::machine`]) as an algorithm in the
//! sense of `MxxCore.Algorithm.run`:
//!
//! - **Inputs.** Input `i` is fetched with `QUERY` of the request `0x00 ++ u64 i`; a response of
//!   any length other than the input's size faults.
//! - **Outputs.** The run ends with `HALT` of each output as a little-endian `u64` length followed
//!   by its bytes.
//! - **Randomness and queries.** A `coins` leaf uses `COINS`; a `query` leaf uses `QUERY`, and a
//!   response of the wrong length faults.
//!
//! # Layout
//!
//! The image is the glue code, a copy routine, the code of every leaf library once (each 16-byte
//! aligned with `EBREAK` padding, with its own read-only data after its code; libraries without
//! read-only data first), and read-only data
//! (input requests, leaf descriptors whose words are all constant, and the bytes of `constant`
//! leaves, which the glue copies). Past the image, every value
//! of every definition use has its own 8-byte-aligned storage, the body of a loop being one use
//! for all iterations; then come the descriptors that depend on a loop index, the scratch area
//! shared by all leaf calls, and the stack.
//!
//! # Glue
//!
//! The glue is built from fixed templates: a leaf call (`a0` = descriptor, `jal ra`), a copy, an
//! environment call, and a loop. A loop keeps its index in memory, copies its initial carried
//! values into its carried slots, and after each iteration copies the next carried values through
//! a staging area into the slots and each per-iteration output into member `i` of its family. An
//! `until` loop leaves when its flag is 1, faults on any other non-zero flag, and faults after
//! its last iteration. Iterations of an independent loop run in index order.
//!
//! # Step bound
//!
//! [`ProofImage::step_bound`] bounds the steps of every run: each straight-line stretch of glue
//! is charged its instruction count, each copy, leaf call, and environment call its own bound, and
//! each loop its count times one iteration.

use crate::{
    expr::{Env, IntExpr, Interval},
    leaf::{Builtin, LeafRef, Rv64Library},
    program::{Node, Value},
    rv64::{
        asm::{Asm, AsmError, Label},
        isa::{AluOp, BranchOp, Reg, ShiftOp},
        machine::BASE,
    },
    types::{ConcreteFamily, ConcreteType},
    validate::{Instance, ValidatedProgram, scratch_bytes},
};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

#[derive(Debug, thiserror::Error)]
pub enum FlattenError {
    #[error("leaf `{0}` has no RV64IM code")]
    NoCode(String),
    #[error(
        "leaf `{name}` has an instruction outside the machine's set at word {index}: {word:#010x}"
    )]
    BadLeafCode { name: String, index: usize, word: u32 },
    #[error("a value of type {0:?} has no size, so it cannot be stored in a proof image")]
    Unsized(ConcreteType),
    #[error("expression {0:?} cannot be evaluated by glue code: {1}")]
    Expression(IntExpr, String),
    #[error(transparent)]
    Asm(#[from] AsmError),
    #[error("the layout did not converge")]
    Layout,
}

#[derive(Clone, Debug)]
pub struct FlattenOptions {
    /// Bytes reserved for the stack at the top of memory.
    pub stack_bytes: u64,
}

impl Default for FlattenOptions {
    fn default() -> Self {
        Self { stack_bytes: 1 << 16 }
    }
}

/// A library of leaf code inside an image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacedLibrary {
    pub name: String,
    /// Index of the library's first instruction word in the image.
    pub first_word: u64,
    pub words: u64,
}

/// An address `base + Σ factor · index` over the indices of enclosing loops, by slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Addr {
    pub base: u64,
    pub terms: Vec<(u32, i128)>,
}

impl Addr {
    /// Whether every value of the address is a multiple of 8.
    pub fn aligned(&self) -> bool {
        self.base.is_multiple_of(8) && self.terms.iter().all(|(_, f)| f % 8 == 0)
    }
}

/// A leaf argument in terms of the indices of enclosing loops: affine (`constant + Σ factor ·
/// index`, a constant when it has no terms), a table over one index, or any other expression,
/// which the glue computes but composition proofs do not follow yet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Argument {
    Affine {
        constant: i128,
        terms: Vec<(u32, i128)>,
    },
    /// `values[i]` for loop index slot `slot`, read from the table at `at` in read-only data.
    Table {
        slot: u32,
        at: u64,
        values: Vec<i128>,
    },
    Other(IntExpr),
}

/// One step of an image's glue that a composition proof follows, with the index of its key
/// instruction: the `ecall` of a query or of `HALT`, the `auipc` of a call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GlueEvent {
    /// The `QUERY` for input `input`: its request (`0x00 ++ u64 input`) at `request`, its
    /// response, of `size` bytes, into `buffer`; then a check of the response's length.
    Query { ecall: u64, input: u64, request: u64, buffer: u64, size: u64 },
    /// A call of the copy routine: `len` bytes from `src` to `dst`.
    Copy { call: u64, dst: Addr, src: Addr, len: u64 },
    /// A call of a leaf, at instruction `entry`, with its descriptor at `descriptor`: its
    /// arguments, inputs and outputs (address and bytes), scratch (address and bytes), and the
    /// steps the image's bound grants it.
    Leaf {
        call: u64,
        entry: u64,
        name: String,
        descriptor: u64,
        /// Each argument, in terms of the indices of enclosing loops.
        arguments: Vec<Argument>,
        inputs: Vec<(u64, u64)>,
        outputs: Vec<(u64, u64)>,
        scratch: (u64, u64),
        steps: u128,
    },
    /// A call of the copy routine for member `v` of a family of `count` members of `stride`
    /// bytes at `base`, `v` the `words`-word integer at `index`, which the glue checks to be below
    /// `count`; the member lands at `dst`.
    CopyAt { call: u64, dst: u64, base: u64, stride: u64, index: u64, words: u64, count: u64 },
    /// A call of the copy routine for candidate `v` of `len` bytes, `v` the `words`-word
    /// integer at `index`, which the glue checks to be below the number of candidates; the
    /// candidates' addresses are the table at `table` in read-only data; the candidate lands at
    /// `dst`.
    CopySelect {
        call: u64,
        dst: u64,
        table: u64,
        candidates: Vec<u64>,
        index: u64,
        words: u64,
        len: u64,
    },
    /// `COINS`: `len` uniform bytes drawn into `at`.
    Coins { ecall: u64, at: u64, len: u64 },
    /// A counted loop over `count` iterations of index slot `slot`, kept at `index`: its head at
    /// instruction `top`, which leaves for `end` once the index reaches the count; the events of
    /// one iteration (the body, then the copies of its carried values and per-iteration outputs),
    /// whose addresses may read the index; then the increment and the jump back to `top`.
    /// Its carried values live at `carried` (address and bytes) between iterations.
    Loop {
        slot: u32,
        index: u64,
        count: u64,
        top: u64,
        end: u64,
        carried: Vec<(u64, u64)>,
        /// For an `until` loop: the stop flag's address, and where the glue checks it after the
        /// carried values are copied (instruction `check`) and continues when it is 0 (`next`).
        until: Option<(u64, u64, u64)>,
        body: Vec<GlueEvent>,
    },
    /// `HALT` with the `len` bytes at `payload`.
    Halt { ecall: u64, payload: u64, len: u64 },
}

#[derive(Clone, Debug)]
pub struct ProofImage {
    pub bytes: Vec<u8>,
    /// The machine memory size to load the image with.
    pub mem_size: u64,
    pub step_bound: u128,
    /// The instruction words of the glue and copy routine, which come first.
    pub glue_words: u64,
    pub libraries: Vec<PlacedLibrary>,
    /// The glue's events in order, as composition proofs follow them, and the index of the copy
    /// routine; or the first glue construct they do not follow yet.
    pub trace: Result<(Vec<GlueEvent>, u64), &'static str>,
}

/// The encoding of `EBREAK`: padding that decodes, so code stays one run of instructions, and
/// faults if executed.
const EBREAK: u32 = 0x0010_0073;

/// The most runs of a window at constant positions the glue unrolls into copies.
const WINDOW_UNROLL: u64 = 16;

pub fn flatten(
    program: &ValidatedProgram,
    options: &FlattenOptions,
) -> Result<ProofImage, FlattenError> {
    // Addresses enter the glue as constants, whose encoding length depends on their size, so the
    // layout is recomputed until the code and read-only data fit below the bases the attempt
    // assumed (the image pads up to them). The bases only grow, so this ends.
    let mut guess = (BASE, BASE);
    for _ in 0..16 {
        let attempt = Flattener::new(program, guess.0, guess.1, options).build()?;
        let needed = (attempt.rodata_base, attempt.data_base);
        if needed.0 <= guess.0 && needed.1 <= guess.1 {
            return Ok(attempt.image);
        }
        guess = (guess.0.max(needed.0), guess.1.max(needed.1));
    }
    Err(FlattenError::Layout)
}

struct Attempt {
    image: ProofImage,
    /// Where read-only data and storage belong given this attempt's code and data sizes.
    rodata_base: u64,
    data_base: u64,
}

fn round8(n: u64) -> u64 {
    n.next_multiple_of(8)
}

/// The steps of one call of the copy routine for `len` bytes, plus its call.
fn copy_bound(len: u64, aligned: bool) -> u128 {
    let body = if aligned && len.is_multiple_of(8) {
        7 * u128::from(len / 8)
    } else {
        7 * u128::from(len)
    };
    8 + body
}

/// Storage of one definition use.
#[derive(Clone, Debug, Default)]
struct InstanceLayout {
    params: Vec<u64>,
    nodes: Vec<Vec<u64>>,
    loops: Vec<Option<LoopLayout>>,
    descriptors: Vec<Option<u64>>,
}

#[derive(Clone, Debug)]
struct LoopLayout {
    index_addr: u64,
    staging: u64,
    body: Box<InstanceLayout>,
}

struct Flattener<'a> {
    program: &'a ValidatedProgram,
    options: &'a FlattenOptions,
    asm: Asm,
    rodata: Vec<u8>,
    rodata_base: u64,
    next_data: u64,
    scratch_bytes: u64,
    /// Libraries in the order of their first call, so the image does not depend on addresses.
    library_labels: Vec<(*const Rv64Library, Label, Arc<Rv64Library>)>,
    copy_label: Label,
    fault_label: Label,
    /// Loop index slots in scope and the addresses holding their values.
    index_addrs: Vec<(u32, u64)>,
    /// The glue's steps so far, and whether the glue is still straight-line.
    trace: Vec<GlueEvent>,
    /// The first glue construct composition proofs do not follow, if any.
    gap: Option<&'static str>,
    /// Each leaf call's instruction, library, and entry offset, resolved once libraries are
    /// placed.
    leaf_entries: Vec<(u64, *const Rv64Library, u64)>,
    /// The read-only table the last index expression read, if any.
    last_table: Option<u64>,
}

impl<'a> Flattener<'a> {
    fn new(
        program: &'a ValidatedProgram,
        rodata_base: u64,
        data_base: u64,
        options: &'a FlattenOptions,
    ) -> Self {
        let mut asm = Asm::new();
        let copy_label = asm.label();
        let fault_label = asm.label();
        Self {
            program,
            options,
            asm,
            rodata: Vec::new(),
            rodata_base,
            next_data: data_base,
            scratch_bytes: 0,
            library_labels: Vec::new(),
            copy_label,
            fault_label,
            index_addrs: Vec::new(),
            trace: Vec::new(),
            gap: None,
            leaf_entries: Vec::new(),
            last_table: None,
        }
    }

    fn alloc(&mut self, bytes: u64) -> u64 {
        let at = self.next_data;
        self.next_data += round8(bytes.max(1));
        at
    }

    fn sized(ty: &ConcreteType) -> Result<u64, FlattenError> {
        ty.size().ok_or_else(|| FlattenError::Unsized(ty.clone()))
    }

    /// Read-only bytes; their address.
    fn rodata(&mut self, bytes: &[u8]) -> u64 {
        while !self.rodata.len().is_multiple_of(8) {
            self.rodata.push(0);
        }
        let at = self.rodata_base + self.rodata.len() as u64;
        self.rodata.extend_from_slice(bytes);
        at
    }

    fn layout(
        &mut self,
        instance: &Instance,
        params: Vec<u64>,
    ) -> Result<InstanceLayout, FlattenError> {
        let definition = self.program.program.definition(instance.definition);
        let mut layout = InstanceLayout { params, ..InstanceLayout::default() };
        for (index, node) in definition.nodes.iter().enumerate() {
            let types = &instance.node_types[index];
            match node {
                Node::Leaf { leaf, .. } => {
                    let outputs = types
                        .iter()
                        .map(|t| Ok(self.alloc(Self::sized(t)?)))
                        .collect::<Result<_, FlattenError>>()?;
                    layout.nodes.push(outputs);
                    layout.loops.push(None);
                    if let LeafRef::Dsl { .. } = leaf {
                        self.scratch_bytes =
                            self.scratch_bytes.max(scratch_bytes(leaf, &instance.env));
                    }
                }
                Node::Loop(lp) => {
                    let lp_instance = instance.loops[index].as_ref().expect("validated loop");
                    let outputs: Vec<u64> = types
                        .iter()
                        .map(|t| Ok(self.alloc(Self::sized(t)?)))
                        .collect::<Result<_, FlattenError>>()?;
                    let staging_bytes: u64 = types[..lp.carried.len()]
                        .iter()
                        .map(|t| Self::sized(t).map(round8))
                        .sum::<Result<_, _>>()?;
                    let staging = self.alloc(staging_bytes);
                    let index_addr = self.alloc(8);
                    // The body reads its carried values from the loop's carried outputs, and its
                    // other inputs from where the caller keeps them.
                    let mut body_params: Vec<u64> = outputs[..lp.carried.len()].to_vec();
                    body_params.extend(lp.inputs.iter().map(|v| value_addr(&layout, *v)));
                    let body = self.layout(&lp_instance.body, body_params)?;
                    layout.nodes.push(outputs);
                    layout.loops.push(Some(LoopLayout {
                        index_addr,
                        staging,
                        body: Box::new(body),
                    }));
                }
            }
        }
        Ok(layout)
    }

    fn build(mut self) -> Result<Attempt, FlattenError> {
        let root = &self.program.root;
        let params: Vec<u64> = root
            .param_types
            .iter()
            .map(|t| Ok(self.alloc(Self::sized(t)?)))
            .collect::<Result<_, FlattenError>>()?;
        let mut layout = self.layout(root, params.clone())?;
        self.descriptors(root, &mut layout)?;
        let scratch = self.alloc(self.scratch_bytes);
        let output_bytes: u64 = self
            .program
            .output_types
            .iter()
            .map(|t| Self::sized(t).map(|s| 8 + round8(s)))
            .sum::<Result<_, _>>()?;
        let staging = self.alloc(output_bytes);
        let mut bound: u128 = 0;

        // Inputs.
        for (i, (addr, ty)) in params.iter().zip(&root.param_types).enumerate() {
            let size = Self::sized(ty)?;
            let mut request = vec![0u8];
            request.extend_from_slice(&(i as u64).to_le_bytes());
            let request_addr = self.rodata(&request);
            let start = self.asm.here();
            self.ecall(1, &[request_addr as i64, 9, *addr as i64, size as i64]);
            self.trace.push(GlueEvent::Query {
                ecall: self.asm.here() as u64 - 1,
                input: i as u64,
                request: request_addr,
                buffer: *addr,
                size,
            });
            self.asm.li(Reg::T0, size as i64);
            self.asm.branch(BranchOp::Bne, Reg::A0, Reg::T0, self.fault_label);
            bound += (self.asm.here() - start) as u128 + 9 + u128::from(size);
        }

        bound += self.instance(root, &layout, scratch)?;

        // Outputs.
        let start = self.asm.here();
        let mut offset = staging;
        let mut copies = 0u128;
        for ((_, value), ty) in self.program.program.outputs.iter().zip(&self.program.output_types)
        {
            let size = Self::sized(ty)?;
            // The payload is packed, so a length may be unaligned: copy it from read-only data.
            let length = self.rodata(&size.to_le_bytes());
            copies += self.copy(offset, length, 8);
            copies += self.copy(offset + 8, value_addr(&layout, *value), size);
            offset += 8 + size;
        }
        let total = offset - staging;
        self.ecall(0, &[staging as i64, total as i64]);
        self.trace.push(GlueEvent::Halt {
            ecall: self.asm.here() as u64 - 1,
            payload: staging,
            len: total,
        });
        bound += (self.asm.here() - start) as u128 + copies + u128::from(total);

        // The copy routine: a0 = destination, a1 = source, a2 = length.
        let copy = self.copy_label;
        let copy_word = self.asm.here() as u64;
        self.asm.place(copy);
        let (bytes, words, done) = (self.asm.label(), self.asm.label(), self.asm.label());
        self.asm.alu(AluOp::Or, Reg::T0, Reg::A0, Reg::A1);
        self.asm.alu(AluOp::Or, Reg::T0, Reg::T0, Reg::A2);
        self.asm.emit(crate::rv64::isa::Instr::Imm {
            op: crate::rv64::isa::ImmOp::Andi,
            rd: Reg::T0,
            rs1: Reg::T0,
            imm: 7,
        });
        self.asm.branch(BranchOp::Bne, Reg::T0, Reg::ZERO, bytes);
        self.asm.place(words);
        self.asm.branch(BranchOp::Beq, Reg::A2, Reg::ZERO, done);
        self.asm.ld(Reg::T1, Reg::A1, 0);
        self.asm.sd(Reg::T1, Reg::A0, 0);
        self.asm.addi(Reg::A0, Reg::A0, 8);
        self.asm.addi(Reg::A1, Reg::A1, 8);
        self.asm.addi(Reg::A2, Reg::A2, -8);
        self.asm.jump(words);
        self.asm.place(bytes);
        self.asm.branch(BranchOp::Beq, Reg::A2, Reg::ZERO, done);
        self.asm.lbu(Reg::T1, Reg::A1, 0);
        self.asm.sb(Reg::T1, Reg::A0, 0);
        self.asm.addi(Reg::A0, Reg::A0, 1);
        self.asm.addi(Reg::A1, Reg::A1, 1);
        self.asm.addi(Reg::A2, Reg::A2, -1);
        self.asm.jump(bytes);
        self.asm.place(done);
        self.asm.emit(crate::rv64::isa::Instr::Jalr { rd: Reg::ZERO, rs1: Reg::RA, imm: 0 });
        // Faults: an environment call the machine does not know.
        let fault = self.fault_label;
        self.asm.place(fault);
        self.asm.li(Reg::A7, 255);
        self.asm.ecall();
        let glue_words = self.asm.here() as u64;

        // Leaf code, once per library. Libraries without read-only data come first, so all code
        // is one run of instructions from the image's start, as a program logic lists it.
        let mut libraries = Vec::new();
        let mut placed = std::mem::take(&mut self.library_labels);
        placed.sort_by_key(|(_, _, library)| !library.rodata.is_empty());
        let mut first_words = HashMap::new();
        for (key, label, library) in placed {
            // A library is linked at 0 and may align its read-only data to 16 bytes, so it starts
            // 16-byte aligned (the image starts at `BASE`, a multiple of 16). The padding is
            // `EBREAK`, which decodes but faults, so a stray jump into it faults.
            while !self.asm.here().is_multiple_of(4) {
                self.asm.word(EBREAK);
            }
            self.asm.place(label);
            first_words.insert(key, self.asm.here() as u64);
            libraries.push(PlacedLibrary {
                name: library.name.clone(),
                first_word: self.asm.here() as u64,
                words: library.words.len() as u64,
            });
            for word in &library.words {
                self.asm.word(*word);
            }
            for chunk in library.rodata.chunks(4) {
                let mut bytes = [0u8; 4];
                bytes[..chunk.len()].copy_from_slice(chunk);
                self.asm.word(u32::from_le_bytes(bytes));
            }
        }
        let code: Vec<u32> = self.asm.finish()?;
        let mut bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
        // The bases this code and read-only data need: the read-only data after the code, and
        // the data after the read-only data where this attempt placed it.
        let rodata_base = (BASE + bytes.len() as u64).next_multiple_of(8);
        let data_base =
            (self.rodata_base.max(rodata_base) + self.rodata.len() as u64).next_multiple_of(4096);
        // Pad with zero words, which do not decode, so a stray jump into the padding faults.
        bytes.resize((self.rodata_base.max(rodata_base) - BASE) as usize, 0);
        bytes.extend_from_slice(&self.rodata);
        let mem_size = self.next_data - BASE + self.options.stack_bytes;
        let trace = if let Some(gap) = self.gap {
            Err(gap)
        } else {
            let mut events = std::mem::take(&mut self.trace);
            let entries: HashMap<u64, u64> = self
                .leaf_entries
                .drain(..)
                .map(|(call, library, offset)| (call, first_words[&library] + offset))
                .collect();
            resolve_entries(&mut events, &entries);
            Ok((events, copy_word))
        };
        Ok(Attempt {
            image: ProofImage { bytes, mem_size, step_bound: bound, glue_words, libraries, trace },
            rodata_base,
            data_base,
        })
    }

    /// Allocates the descriptor of every DSL leaf call: in read-only data when every word is a
    /// constant, past the image when an argument depends on a loop index.
    fn descriptors(
        &mut self,
        instance: &Instance,
        layout: &mut InstanceLayout,
    ) -> Result<(), FlattenError> {
        let definition = self.program.program.definition(instance.definition);
        for (index, node) in definition.nodes.iter().enumerate() {
            let descriptor = match node {
                Node::Leaf { leaf: LeafRef::Dsl { arguments, .. }, inputs } => {
                    let words = arguments.len() + inputs.len() + layout.nodes[index].len() + 1;
                    if arguments.iter().any(|a| dynamic(a, &instance.env)) {
                        Some(self.alloc(8 * words as u64))
                    } else {
                        // Filled in when the call is emitted, once the scratch address is known.
                        None
                    }
                }
                Node::Loop(_) => {
                    let lp_instance = instance.loops[index].as_ref().expect("validated loop");
                    let body = &mut layout.loops[index].as_mut().expect("loop layout").body;
                    self.descriptors(&lp_instance.body, body)?;
                    None
                }
                Node::Leaf { .. } => None,
            };
            layout.descriptors.push(descriptor);
        }
        Ok(())
    }

    /// Emits `a7 = call`, the arguments in `a0..`, and `ecall`.
    fn ecall(&mut self, call: i64, args: &[i64]) {
        for (reg, value) in [Reg::A0, Reg::A1, Reg::A2, Reg::A3].iter().zip(args) {
            self.asm.li(*reg, *value);
        }
        self.asm.li(Reg::A7, call);
        self.asm.ecall();
    }

    /// Copies `len` bytes between constant addresses; returns the steps of the copy routine.
    fn copy(&mut self, dst: u64, src: u64, len: u64) -> u128 {
        self.asm.li(Reg::A0, dst as i64);
        self.asm.li(Reg::A1, src as i64);
        self.asm.li(Reg::A2, len as i64);
        self.trace.push(GlueEvent::Copy {
            call: self.asm.here() as u64,
            dst: Addr { base: dst, terms: vec![] },
            src: Addr { base: src, terms: vec![] },
            len,
        });
        self.asm.call(self.copy_label, 0);
        copy_bound(len, dst.is_multiple_of(8) && src.is_multiple_of(8))
    }

    /// Copies `len` bytes from the address in `src` (a saved register) to a constant address.
    /// Calls the copy routine from `base + offset` to `base + offset` for a source and a
    /// destination whose offsets may read loop indices; returns a bound on its steps past its
    /// instructions. A constant offset is folded; an index-dependent one is evaluated into `s1`
    /// (source) or `s2` (destination). The trace records each address as affine in the indices;
    /// any other offset leaves the glue outside what composition proofs follow.
    fn copy_at(
        &mut self,
        dst: (u64, Option<&IntExpr>),
        src: (u64, Option<&IntExpr>),
        len: u64,
        env: &Env,
        ranges: &[(u32, Interval)],
    ) -> Result<u128, FlattenError> {
        let address = |this: &mut Self,
                       (base, offset): (u64, Option<&IntExpr>),
                       holder: Reg|
         -> Result<(Option<Reg>, Addr), FlattenError> {
            let Some(offset) = offset.filter(|o| dynamic(o, env)) else {
                let constant = offset.map_or(Ok(0), |o| {
                    o.eval(env).map_err(|e| FlattenError::Expression(o.clone(), e.to_string()))
                })?;
                return Ok((None, Addr { base: (base as i128 + constant) as u64, terms: vec![] }));
            };
            let addr = match affine(offset, env) {
                Some((constant, terms)) => Addr { base: (base as i128 + constant) as u64, terms },
                None => {
                    this.gap("a copy at a non-affine index offset");
                    Addr { base, terms: vec![] }
                }
            };
            this.expr(offset, env, ranges, 0)?;
            this.asm.li(holder, base as i64);
            this.asm.alu(AluOp::Add, holder, holder, Reg::T3);
            Ok((Some(holder), addr))
        };
        let (src_reg, src_addr) = address(self, src, Reg(9))?;
        let (dst_reg, dst_addr) = address(self, dst, Reg(18))?;
        match dst_reg {
            Some(r) => self.asm.mv(Reg::A0, r),
            None => self.asm.li(Reg::A0, dst_addr.base as i64),
        }
        match src_reg {
            Some(r) => self.asm.mv(Reg::A1, r),
            None => self.asm.li(Reg::A1, src_addr.base as i64),
        }
        self.asm.li(Reg::A2, len as i64);
        let aligned = dst_addr.aligned() && src_addr.aligned() && len.is_multiple_of(8);
        self.trace.push(GlueEvent::Copy {
            call: self.asm.here() as u64,
            dst: dst_addr,
            src: src_addr,
            len,
        });
        self.asm.call(self.copy_label, 0);
        Ok(copy_bound(len, aligned && src_reg.is_none() && dst_reg.is_none()))
    }

    /// Records that the glue has a construct composition proofs do not follow.
    fn gap(&mut self, why: &'static str) {
        self.gap.get_or_insert(why);
    }

    /// Evaluates `expr` into `TEMPS[rd]`, using the registers after it as scratch. The code jumps
    /// only forwards, so its steps are at most its instruction count.
    fn expr(
        &mut self,
        expr: &IntExpr,
        env: &Env,
        ranges: &[(u32, Interval)],
        rd: usize,
    ) -> Result<(), FlattenError> {
        const TEMPS: [Reg; 7] = [Reg::T3, Reg::T4, Reg::T5, Reg::T6, Reg::A4, Reg::A5, Reg::A6];
        let fail = |message: &str| FlattenError::Expression(expr.clone(), message.to_owned());
        let bound = expr.bound(env, ranges).map_err(|e| fail(&e.to_string()))?;
        if bound.min < i128::from(i64::MIN) || bound.max > i128::from(i64::MAX) {
            return Err(fail("its value may not fit in 64 bits"));
        }
        let reg = *TEMPS.get(rd).ok_or_else(|| fail("too deeply nested"))?;
        if !dynamic(expr, env) {
            let value = expr.eval(env).map_err(|e| fail(&e.to_string()))?;
            self.asm.li(reg, value as i64);
            return Ok(());
        }
        let index_addr = |this: &Self, slot: u32| {
            this.index_addrs
                .iter()
                .find(|(s, _)| *s == slot)
                .map(|(_, a)| *a)
                .ok_or_else(|| fail("the loop index is not in scope"))
        };
        // Two normal forms, each proved once in Lean (`MxxCore.Index`): an affine combination of
        // the loop indices, or a table in read-only data indexed by one loop index.
        if let (Some((constant, terms)), Some(scratch)) = (affine(expr, env), TEMPS.get(rd + 2)) {
            let (index, coefficient) = (TEMPS[rd + 1], *scratch);
            self.asm.li(reg, constant as i64);
            for (slot, factor) in terms {
                let addr = index_addr(self, slot)?;
                self.asm.li(index, addr as i64);
                self.asm.ld(index, index, 0);
                self.asm.li(coefficient, factor as i64);
                self.asm.alu(AluOp::Mul, index, index, coefficient);
                self.asm.alu(AluOp::Add, reg, reg, index);
            }
            return Ok(());
        }
        if let Some((slot, table)) = index_table(expr, env, ranges) &&
            let Some(base) = TEMPS.get(rd + 1)
        {
            let addr = index_addr(self, slot)?;
            let bytes: Vec<u8> = table.iter().flat_map(|v| (*v as i64).to_le_bytes()).collect();
            let at = self.rodata(&bytes);
            self.last_table = Some(at);
            self.asm.li(reg, addr as i64);
            self.asm.ld(reg, reg, 0);
            self.asm.shift(ShiftOp::Slli, reg, reg, 3);
            self.asm.li(*base, at as i64);
            self.asm.alu(AluOp::Add, reg, reg, *base);
            self.asm.ld(reg, reg, 0);
            return Ok(());
        }
        let binary = |this: &mut Self, a: &IntExpr, b: &IntExpr| -> Result<Reg, FlattenError> {
            this.expr(a, env, ranges, rd)?;
            this.expr(b, env, ranges, rd + 1)?;
            Ok(TEMPS[rd + 1])
        };
        match expr {
            IntExpr::LoopIndex(slot) => {
                let addr = self
                    .index_addrs
                    .iter()
                    .find(|(s, _)| s == slot)
                    .map(|(_, a)| *a)
                    .ok_or_else(|| fail("the loop index is not in scope"))?;
                self.asm.li(reg, addr as i64);
                self.asm.ld(reg, reg, 0);
            }
            IntExpr::Add(a, b) => {
                let other = binary(self, a, b)?;
                self.asm.alu(AluOp::Add, reg, reg, other);
            }
            IntExpr::Sub(a, b) => {
                let other = binary(self, a, b)?;
                self.asm.alu(AluOp::Sub, reg, reg, other);
            }
            IntExpr::Mul(a, b) => {
                let other = binary(self, a, b)?;
                self.asm.alu(AluOp::Mul, reg, reg, other);
            }
            IntExpr::FloorDiv(a, b) | IntExpr::Rem(a, b) => {
                let divisor = b.bound(env, ranges).map_err(|e| fail(&e.to_string()))?;
                if divisor.min <= 0 {
                    return Err(fail("glue divides only by positive divisors"));
                }
                let dividend = a.bound(env, ranges).map_err(|e| fail(&e.to_string()))?;
                if dividend.min < 0 {
                    return Err(fail("glue divides only non-negative dividends"));
                }
                // With a non-negative dividend and a positive divisor, flooring and truncating
                // agree, so DIV and REM are exact.
                let other = binary(self, a, b)?;
                let op =
                    if matches!(expr, IntExpr::FloorDiv(..)) { AluOp::Div } else { AluOp::Rem };
                self.asm.alu(op, reg, reg, other);
            }
            IntExpr::Select { selector, branches } => {
                // Validation bounded the selector to the branches, so one of them matches.
                self.expr(selector, env, ranges, rd + 1)?;
                let selected = TEMPS[rd + 1];
                let end = self.asm.label();
                for (k, branch) in branches.iter().enumerate() {
                    let next = self.asm.label();
                    self.asm.li(Reg::T0, k as i64);
                    self.asm.branch(BranchOp::Bne, selected, Reg::T0, next);
                    self.expr(branch, env, ranges, rd)?;
                    self.asm.jump(end);
                    self.asm.place(next);
                }
                self.asm.jump(self.fault_label);
                self.asm.place(end);
            }
            IntExpr::Log2Ceil(_) => {
                return Err(fail("Log2Ceil over a loop index is not supported in glue"));
            }
            IntExpr::Const(_) | IntExpr::Var(_) => unreachable!("constant expressions are folded"),
        }
        Ok(())
    }

    /// The glue of a window as a loop of copies: `runs` runs of `length` bytes, from `offset`
    /// past `src` every `stride` bytes, to consecutive bytes at `out`; returns a bound on its
    /// steps past its instructions.
    #[allow(clippy::too_many_arguments)]
    fn window_loop(
        &mut self,
        offset: &IntExpr,
        runs: u64,
        length: u64,
        stride: &IntExpr,
        src: u64,
        out: u64,
        env: &Env,
        ranges: &[(u32, Interval)],
    ) -> Result<u128, FlattenError> {
        // s1 = source, s2 = destination, s3 = runs left, s4 = stride.
        self.expr(offset, env, ranges, 0)?;
        self.asm.li(Reg(9), src as i64);
        self.asm.alu(AluOp::Add, Reg(9), Reg(9), Reg::T3);
        self.asm.li(Reg(18), out as i64);
        self.asm.li(Reg(19), runs as i64);
        self.expr(stride, env, ranges, 0)?;
        self.asm.mv(Reg(20), Reg::T3);
        let (top, end) = (self.asm.label(), self.asm.label());
        self.asm.place(top);
        let loop_start = self.asm.here();
        self.asm.branch(BranchOp::Beq, Reg(19), Reg::ZERO, end);
        self.asm.mv(Reg::A0, Reg(18));
        self.asm.mv(Reg::A1, Reg(9));
        self.asm.li(Reg::A2, length as i64);
        self.asm.call(self.copy_label, 0);
        self.asm.alu(AluOp::Add, Reg(9), Reg(9), Reg(20));
        self.asm.li(Reg::T0, length as i64);
        self.asm.alu(AluOp::Add, Reg(18), Reg(18), Reg::T0);
        self.asm.addi(Reg(19), Reg(19), -1);
        self.asm.jump(top);
        let per_run = (self.asm.here() - loop_start) as u128 + copy_bound(length, false);
        let loop_instrs = (self.asm.here() - loop_start) as u128;
        self.asm.place(end);
        // Every run, then the final check.
        Ok(per_run * u128::from(runs) + u128::from(Asm::steps_of_branch()) - loop_instrs)
    }

    /// Emits the glue of one definition use; returns a bound on its steps.
    fn instance(
        &mut self,
        instance: &Instance,
        layout: &InstanceLayout,
        scratch: u64,
    ) -> Result<u128, FlattenError> {
        let definition = self.program.program.definition(instance.definition);
        let mut bound = 0u128;
        let env = instance.env.clone();
        let ranges = instance.ranges.clone();
        for (index, node) in definition.nodes.iter().enumerate() {
            let start = self.asm.here();
            let mut extra = 0u128;
            let outputs = &layout.nodes[index];
            match node {
                Node::Leaf { leaf: LeafRef::Dsl { leaf, arguments }, inputs } => {
                    let code = leaf
                        .rv64
                        .as_ref()
                        .ok_or_else(|| FlattenError::NoCode(leaf.signature.name.clone()))?;
                    if let Err((i, word)) = crate::rv64::check_code(&code.library.words) {
                        return Err(FlattenError::BadLeafCode {
                            name: code.library.name.clone(),
                            index: i,
                            word,
                        });
                    }
                    let library = Arc::as_ptr(&code.library);
                    let label = match self.library_labels.iter().find(|(p, _, _)| *p == library) {
                        Some((_, label, _)) => *label,
                        None => {
                            let label = self.asm.label();
                            self.library_labels.push((library, label, code.library.clone()));
                            label
                        }
                    };
                    let mut words: Vec<Option<i64>> = Vec::new();
                    for argument in arguments {
                        words.push(if dynamic(argument, &env) {
                            None
                        } else {
                            Some(argument.eval(&env).map_err(|e| {
                                FlattenError::Expression(argument.clone(), e.to_string())
                            })? as i64)
                        });
                    }
                    words.extend(inputs.iter().map(|v| Some(value_addr(layout, *v) as i64)));
                    words.extend(outputs.iter().map(|a| Some(*a as i64)));
                    words.push(Some(scratch as i64));
                    let mut tables: Vec<Option<u64>> = vec![None; arguments.len()];
                    let descriptor = match layout.descriptors[index] {
                        Some(at) => {
                            for (k, word) in words.iter().enumerate() {
                                match word {
                                    Some(value) => self.asm.li(Reg::T3, *value),
                                    None => {
                                        self.last_table = None;
                                        self.expr(&arguments[k], &env, &ranges, 0)?;
                                        tables[k] = self.last_table.take();
                                    }
                                }
                                self.asm.li(Reg::T0, at as i64 + 8 * k as i64);
                                self.asm.sd(Reg::T3, Reg::T0, 0);
                            }
                            at
                        }
                        None => {
                            let bytes: Vec<u8> = words
                                .iter()
                                .flat_map(|w| w.expect("constant").to_le_bytes())
                                .collect();
                            self.rodata(&bytes)
                        }
                    };
                    self.asm.li(Reg::A0, descriptor as i64);
                    let sized = |v: &Value| Self::sized(instance.value_type(*v));
                    self.leaf_entries.push((self.asm.here() as u64, library, code.entry));
                    self.trace.push(GlueEvent::Leaf {
                        call: self.asm.here() as u64,
                        entry: 0,
                        name: leaf.signature.name.clone(),
                        descriptor,
                        arguments: arguments
                            .iter()
                            .zip(&tables)
                            .map(|(a, table)| match (affine(a, &env), table) {
                                (Some((constant, terms)), _) => {
                                    Argument::Affine { constant, terms }
                                }
                                (None, Some(at)) => match index_table(a, &env, &ranges) {
                                    Some((slot, values)) => {
                                        Argument::Table { slot, at: *at, values }
                                    }
                                    None => Argument::Other(a.clone()),
                                },
                                (None, None) => Argument::Other(a.clone()),
                            })
                            .collect(),
                        inputs: inputs
                            .iter()
                            .map(|v| Ok((value_addr(layout, *v), sized(v)?)))
                            .collect::<Result<_, FlattenError>>()?,
                        outputs: outputs
                            .iter()
                            .zip(&instance.node_types[index])
                            .map(|(a, t)| Ok((*a, Self::sized(t)?)))
                            .collect::<Result<_, FlattenError>>()?,
                        scratch: (scratch, self.scratch_bytes),
                        steps: 0,
                    });
                    self.asm.call(label, code.entry);
                    let substitution: BTreeMap<String, IntExpr> = leaf
                        .signature
                        .arguments
                        .iter()
                        .cloned()
                        .zip(arguments.iter().cloned())
                        .collect();
                    let steps =
                        code.steps.substitute_vars(&substitution).bound(&env, &ranges).map_err(
                            |e| FlattenError::Expression(code.steps.clone(), e.to_string()),
                        )?;
                    let steps = u128::try_from(steps.max.max(0)).expect("non-negative");
                    if let Some(GlueEvent::Leaf { steps: budget, .. }) = self.trace.last_mut() {
                        *budget = steps;
                    }
                    extra += steps;
                }
                Node::Leaf { leaf: LeafRef::Builtin(op), inputs } => {
                    let input_types: Vec<&ConcreteType> =
                        inputs.iter().map(|v| instance.value_type(*v)).collect();
                    let out = outputs[0];
                    let out_size = Self::sized(&instance.node_types[index][0])?;
                    match op {
                        Builtin::Window { offset, runs, length, stride } => {
                            let runs = runs.eval(&env).expect("validated") as u64;
                            let length = length.eval(&env).expect("validated") as u64;
                            let src = value_addr(layout, inputs[0]);
                            if !dynamic(stride, &env) && runs <= WINDOW_UNROLL {
                                // A few runs: one copy each, which composition proofs follow
                                // (`ProofImage::trace`).
                                for run in 0..runs {
                                    let from = offset.clone() + stride.clone() * run as i128;
                                    extra += self.copy_at(
                                        (out + run * length, None),
                                        (src, Some(&from)),
                                        length,
                                        &env,
                                        &ranges,
                                    )?;
                                }
                            } else {
                                self.gap("a window over many runs or a run-time stride");
                                extra += self.window_loop(
                                    offset, runs, length, stride, src, out, &env, &ranges,
                                )?;
                            }
                        }
                        Builtin::Concat | Builtin::Pack => {
                            let mut at = out;
                            for (value, ty) in inputs.iter().zip(&input_types) {
                                let size = Self::sized(ty)?;
                                extra += self.copy(at, value_addr(layout, *value), size);
                                at += size;
                            }
                        }
                        Builtin::Get { index: member } => {
                            let family = input_types[0].family().expect("validated family");
                            let base = value_addr(layout, inputs[0]);
                            match family {
                                ConcreteFamily::Uniform { element, .. } => {
                                    let stride = Self::sized(element)?;
                                    let offset = member.clone() * IntExpr::from(stride as i128);
                                    extra += self.copy_at(
                                        (out, None),
                                        (base, Some(&offset)),
                                        out_size,
                                        &env,
                                        &ranges,
                                    )?;
                                }
                                ConcreteFamily::Dependent(_) => {
                                    let k =
                                        member.eval(&env).expect("validated constant index") as u64;
                                    let offset = family.member_offset(k).expect("validated member");
                                    extra += self.copy(out, base + offset, out_size);
                                }
                            }
                        }
                        Builtin::GetDynamic => {
                            let family = input_types[0].family().expect("validated family");
                            let ConcreteType::Int { words } = input_types[1] else {
                                unreachable!("validated Int")
                            };
                            let index_addr = value_addr(layout, inputs[1]);
                            self.asm.li(Reg::T0, index_addr as i64);
                            for w in 1..*words {
                                self.asm.ld(Reg::T1, Reg::T0, 8 * w as i32);
                                self.asm.branch(
                                    BranchOp::Bne,
                                    Reg::T1,
                                    Reg::ZERO,
                                    self.fault_label,
                                );
                            }
                            self.asm.ld(Reg::T3, Reg::T0, 0);
                            self.asm.li(Reg::T1, family.len() as i64);
                            self.asm.branch(BranchOp::Bgeu, Reg::T3, Reg::T1, self.fault_label);
                            self.asm.li(Reg::T0, out_size as i64);
                            self.asm.alu(AluOp::Mul, Reg::T3, Reg::T3, Reg::T0);
                            let base = value_addr(layout, inputs[0]);
                            self.asm.li(Reg(9), base as i64);
                            self.asm.alu(AluOp::Add, Reg(9), Reg(9), Reg::T3);
                            self.asm.li(Reg::A0, out as i64);
                            self.asm.mv(Reg::A1, Reg(9));
                            self.asm.li(Reg::A2, out_size as i64);
                            self.trace.push(GlueEvent::CopyAt {
                                call: self.asm.here() as u64,
                                dst: out,
                                base,
                                stride: out_size,
                                index: index_addr,
                                words: u64::from(*words),
                                count: family.len(),
                            });
                            self.asm.call(self.copy_label, 0);
                            extra += copy_bound(out_size, false);
                        }
                        Builtin::Select => {
                            // The candidates' addresses are a table in read-only data: the glue
                            // checks the selector, loads entry `v`, and copies from it.
                            let ConcreteType::Int { words } = input_types[0] else {
                                unreachable!("validated Int")
                            };
                            let selector = value_addr(layout, inputs[0]);
                            let candidates: Vec<u64> =
                                inputs[1..].iter().map(|v| value_addr(layout, *v)).collect();
                            let bytes: Vec<u8> =
                                candidates.iter().flat_map(|a| a.to_le_bytes()).collect();
                            let table = self.rodata(&bytes);
                            self.asm.li(Reg::T0, selector as i64);
                            for w in 1..*words {
                                self.asm.ld(Reg::T1, Reg::T0, 8 * w as i32);
                                self.asm.branch(
                                    BranchOp::Bne,
                                    Reg::T1,
                                    Reg::ZERO,
                                    self.fault_label,
                                );
                            }
                            self.asm.ld(Reg::T3, Reg::T0, 0);
                            self.asm.li(Reg::T1, candidates.len() as i64);
                            self.asm.branch(BranchOp::Bgeu, Reg::T3, Reg::T1, self.fault_label);
                            self.asm.shift(ShiftOp::Slli, Reg::T3, Reg::T3, 3);
                            self.asm.li(Reg::T4, table as i64);
                            self.asm.alu(AluOp::Add, Reg::T3, Reg::T3, Reg::T4);
                            self.asm.ld(Reg(9), Reg::T3, 0);
                            self.asm.li(Reg::A0, out as i64);
                            self.asm.mv(Reg::A1, Reg(9));
                            self.asm.li(Reg::A2, out_size as i64);
                            self.trace.push(GlueEvent::CopySelect {
                                call: self.asm.here() as u64,
                                dst: out,
                                table,
                                candidates,
                                index: selector,
                                words: u64::from(*words),
                                len: out_size,
                            });
                            self.asm.call(self.copy_label, 0);
                            extra += copy_bound(out_size, false);
                        }
                        Builtin::Constant { bytes, .. } => {
                            let at = self.rodata(bytes);
                            extra += self.copy(out, at, out_size);
                        }
                        Builtin::Coins { .. } => {
                            self.ecall(2, &[out as i64, out_size as i64]);
                            self.trace.push(GlueEvent::Coins {
                                ecall: self.asm.here() as u64 - 1,
                                at: out,
                                len: out_size,
                            });
                            extra += u128::from(out_size);
                        }
                        Builtin::Query { .. } => {
                            self.gap("a query past the inputs");
                            let request_size = Self::sized(input_types[0])?;
                            self.ecall(
                                1,
                                &[
                                    value_addr(layout, inputs[0]) as i64,
                                    request_size as i64,
                                    out as i64,
                                    out_size as i64,
                                ],
                            );
                            self.asm.li(Reg::T0, out_size as i64);
                            self.asm.branch(BranchOp::Bne, Reg::A0, Reg::T0, self.fault_label);
                            extra += u128::from(request_size + out_size);
                        }
                    }
                }
                Node::Loop(lp) => {
                    let lp_instance = instance.loops[index].as_ref().expect("validated loop");
                    let lp_layout = layout.loops[index].as_ref().expect("loop layout");
                    let body_def = self.program.program.definition(lp_instance.body.definition);
                    let carried = lp.carried.len();
                    let carried_sizes: Vec<u64> = instance.node_types[index][..carried]
                        .iter()
                        .map(Self::sized)
                        .collect::<Result<_, _>>()?;
                    for (k, value) in lp.carried.iter().enumerate() {
                        extra +=
                            self.copy(outputs[k], value_addr(layout, *value), carried_sizes[k]);
                    }
                    if lp_instance.count == 1 && !lp.until {
                        // One iteration, its index bound to 0 (`validate`): the body inline,
                        // then its carried values and member 0 of each per-iteration family.
                        let body_start = self.asm.here();
                        extra += self.instance(&lp_instance.body, &lp_layout.body, scratch)?;
                        // The body's bound counts its own instructions.
                        let body_instrs = (self.asm.here() - body_start) as u128;
                        let mut staged = lp_layout.staging;
                        for (k, value) in body_def.outputs[..carried].iter().enumerate() {
                            extra += self.copy(
                                staged,
                                value_addr(&lp_layout.body, *value),
                                carried_sizes[k],
                            );
                            staged += round8(carried_sizes[k]);
                        }
                        let mut staged = lp_layout.staging;
                        for k in 0..carried {
                            extra += self.copy(outputs[k], staged, carried_sizes[k]);
                            staged += round8(carried_sizes[k]);
                        }
                        for (k, value) in body_def.outputs[carried..].iter().enumerate() {
                            let size = Self::sized(lp_instance.body.value_type(*value))?;
                            extra += self.copy(
                                outputs[carried + k],
                                value_addr(&lp_layout.body, *value),
                                size,
                            );
                        }
                        bound += (self.asm.here() - start) as u128 - body_instrs + extra;
                        continue;
                    }
                    self.asm.li(Reg::T0, lp_layout.index_addr as i64);
                    self.asm.sd(Reg::ZERO, Reg::T0, 0);
                    let (top, end) = (self.asm.label(), self.asm.label());
                    let setup = (self.asm.here() - start) as u128;
                    let top_word = self.asm.here() as u64;
                    self.asm.place(top);
                    let outer = std::mem::take(&mut self.trace);
                    let iteration_start = self.asm.here();
                    self.asm.li(Reg::T0, lp_layout.index_addr as i64);
                    self.asm.ld(Reg::T0, Reg::T0, 0);
                    self.asm.li(Reg::T1, lp_instance.count as i64);
                    let exhausted = if lp.until { self.fault_label } else { end };
                    self.asm.branch(BranchOp::Bgeu, Reg::T0, Reg::T1, exhausted);
                    let head = (self.asm.here() - iteration_start) as u128;
                    self.index_addrs.push((lp.index_slot, lp_layout.index_addr));
                    let body_bound = self.instance(&lp_instance.body, &lp_layout.body, scratch)?;
                    let tail_start = self.asm.here();
                    let mut tail_copies = 0u128;
                    // Next carried values: through the staging area into the slots.
                    let mut staged = lp_layout.staging;
                    for (k, value) in body_def.outputs[..carried].iter().enumerate() {
                        tail_copies += self.copy(
                            staged,
                            value_addr(&lp_layout.body, *value),
                            carried_sizes[k],
                        );
                        staged += round8(carried_sizes[k]);
                    }
                    let mut staged = lp_layout.staging;
                    for k in 0..carried {
                        tail_copies += self.copy(outputs[k], staged, carried_sizes[k]);
                        staged += round8(carried_sizes[k]);
                    }
                    let mut until = None;
                    if lp.until {
                        // Stop on 1, continue on 0, fault on anything else.
                        let flag = value_addr(&lp_layout.body, body_def.outputs[carried]);
                        let next = self.asm.label();
                        let check = self.asm.here() as u64;
                        self.asm.li(Reg::T0, flag as i64);
                        self.asm.ld(Reg::T0, Reg::T0, 0);
                        self.asm.branch(BranchOp::Beq, Reg::T0, Reg::ZERO, next);
                        self.asm.li(Reg::T1, 1);
                        self.asm.branch(BranchOp::Bne, Reg::T0, Reg::T1, self.fault_label);
                        self.asm.jump(end);
                        self.asm.place(next);
                        until = Some((flag, check, self.asm.here() as u64));
                    }
                    // Per-iteration outputs into member `i` of their families.
                    let body_ranges = &lp_instance.body.ranges;
                    for (k, value) in
                        body_def.outputs[carried + usize::from(lp.until)..].iter().enumerate()
                    {
                        let size = Self::sized(lp_instance.body.value_type(*value))?;
                        let member = IntExpr::LoopIndex(lp.index_slot) * size as i128;
                        tail_copies += self.copy_at(
                            (outputs[carried + k], Some(&member)),
                            (value_addr(&lp_layout.body, *value), None),
                            size,
                            &lp_instance.body.env,
                            body_ranges,
                        )?;
                    }
                    self.index_addrs.pop();
                    self.asm.li(Reg::T0, lp_layout.index_addr as i64);
                    self.asm.ld(Reg::T1, Reg::T0, 0);
                    self.asm.addi(Reg::T1, Reg::T1, 1);
                    self.asm.sd(Reg::T1, Reg::T0, 0);
                    self.asm.jump(top);
                    let tail = (self.asm.here() - tail_start) as u128 + tail_copies;
                    let body = std::mem::replace(&mut self.trace, outer);
                    self.trace.push(GlueEvent::Loop {
                        slot: lp.index_slot,
                        index: lp_layout.index_addr,
                        count: lp_instance.count,
                        top: top_word,
                        end: self.asm.here() as u64,
                        carried: outputs[..carried].iter().copied().zip(carried_sizes).collect(),
                        until,
                        body,
                    });
                    self.asm.place(end);
                    let iteration = head + body_bound + tail;
                    bound += setup + extra + iteration * u128::from(lp_instance.count) + head;
                    continue;
                }
            }
            bound += (self.asm.here() - start) as u128 + extra;
        }
        Ok(bound)
    }
}

/// `expr` as `constant + Σ factor · index` over the loop indices unbound in `env`, when it is
/// affine in them.
fn affine(expr: &IntExpr, env: &Env) -> Option<(i128, Vec<(u32, i128)>)> {
    if !dynamic(expr, env) {
        return Some((expr.eval(env).ok()?, Vec::new()));
    }
    let combine = |(a, x): (i128, Vec<(u32, i128)>), (b, y): (i128, Vec<(u32, i128)>), sign| {
        let mut terms: BTreeMap<u32, i128> = x.into_iter().collect();
        for (slot, factor) in y {
            *terms.entry(slot).or_default() += sign * factor;
        }
        let terms = terms.into_iter().filter(|(_, f)| *f != 0).collect();
        Some((a.checked_add(sign * b)?, terms))
    };
    match expr {
        IntExpr::LoopIndex(slot) => Some((0, vec![(*slot, 1)])),
        IntExpr::Add(a, b) => combine(affine(a, env)?, affine(b, env)?, 1),
        IntExpr::Sub(a, b) => combine(affine(a, env)?, affine(b, env)?, -1),
        IntExpr::Mul(a, b) => {
            let (x, y) = (affine(a, env)?, affine(b, env)?);
            let (scale, (c, terms)) = match (x.1.is_empty(), y.1.is_empty()) {
                (true, _) => (x.0, y),
                (_, true) => (y.0, x),
                _ => return None,
            };
            let terms = terms
                .into_iter()
                .map(|(s, f)| Some((s, f.checked_mul(scale)?)))
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .filter(|(_, f)| *f != 0)
                .collect();
            Some((c.checked_mul(scale)?, terms))
        }
        _ => None,
    }
}

/// The most entries of an index table in read-only data.
const INDEX_TABLE: i128 = 1 << 12;

/// `expr` as the table of its values over the one loop index it reads at run time, when that
/// index has at most `INDEX_TABLE` values and every value fits in 64 bits.
fn index_table(expr: &IntExpr, env: &Env, ranges: &[(u32, Interval)]) -> Option<(u32, Vec<i128>)> {
    let slots: Vec<u32> = ranges
        .iter()
        .map(|(s, _)| *s)
        .filter(|s| expr.uses_index(*s) && env.index(*s).is_none())
        .collect();
    let [slot] = slots[..] else { return None };
    let range = ranges.iter().find(|(s, _)| *s == slot)?.1;
    if range.min != 0 || range.max >= INDEX_TABLE {
        return None;
    }
    (0..=range.max)
        .map(|i| {
            let value = expr.eval(&env.with_index(slot, i)).ok()?;
            (i64::try_from(value).is_ok()).then_some(value)
        })
        .collect::<Option<Vec<_>>>()
        .map(|table| (slot, table))
}

/// Sets the entry of every leaf call in `events`, loop bodies included, by its call instruction.
fn resolve_entries(events: &mut [GlueEvent], entries: &HashMap<u64, u64>) {
    for event in events {
        match event {
            GlueEvent::Leaf { call, entry, .. } => *entry = entries[call],
            GlueEvent::Loop { body, .. } => resolve_entries(body, entries),
            _ => {}
        }
    }
}

/// Whether `expr` reads a loop index only known at run time: one not bound in `env` (an index of a
/// loop of one iteration is bound to 0).
fn dynamic(expr: &IntExpr, env: &Env) -> bool {
    expr.uses_any_index() && expr.eval(env).is_err()
}

fn value_addr(layout: &InstanceLayout, value: Value) -> u64 {
    match value {
        Value::Param(i) => layout.params[i],
        Value::Node { node, port } => layout.nodes[node][port],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        expr::Bindings,
        host,
        leaf::{HostCall, Leaf, LeafSignature, Rv64Code},
        program::{DefinitionBuilder, Loop, ProgramBuilder},
        rv64::machine::{Environment, Machine, Outcome},
        types::CoreType,
        validate::validate,
    };
    use rand::Rng;

    fn param(name: &str) -> i128 {
        std::env::var(format!("MXX_CORE_TEST_{name}")).ok().and_then(|v| v.parse().ok()).unwrap_or(
            match name {
                "N" => 7,
                "K" => 5,
                _ => 3,
            },
        )
    }

    /// `out[j] = (a[j] + b[j]) mod q` for `n` words, with `a[j], b[j] < q < 2^63`.
    fn add_mod_leaf() -> Arc<Leaf> {
        let mut asm = Asm::new();
        let (top, skip, done) = (asm.label(), asm.label(), asm.label());
        asm.ld(Reg::T0, Reg::A0, 0);
        asm.ld(Reg::T1, Reg::A0, 8);
        asm.ld(Reg::T2, Reg::A0, 16);
        asm.ld(Reg::A1, Reg::A0, 24);
        asm.ld(Reg::A2, Reg::A0, 32);
        asm.place(top);
        asm.branch(BranchOp::Beq, Reg::T0, Reg::ZERO, done);
        asm.ld(Reg::A3, Reg::T2, 0);
        asm.ld(Reg::A4, Reg::A1, 0);
        asm.alu(AluOp::Add, Reg::A3, Reg::A3, Reg::A4);
        asm.branch(BranchOp::Bltu, Reg::A3, Reg::T1, skip);
        asm.alu(AluOp::Sub, Reg::A3, Reg::A3, Reg::T1);
        asm.place(skip);
        asm.sd(Reg::A3, Reg::A2, 0);
        asm.addi(Reg::T2, Reg::T2, 8);
        asm.addi(Reg::A1, Reg::A1, 8);
        asm.addi(Reg::A2, Reg::A2, 8);
        asm.addi(Reg::T0, Reg::T0, -1);
        asm.jump(top);
        asm.place(done);
        asm.emit(crate::rv64::isa::Instr::Jalr { rd: Reg::ZERO, rs1: Reg::RA, imm: 0 });
        let words = asm.finish().unwrap();
        let column = CoreType::buffer(8, "n");
        Arc::new(Leaf {
            signature: LeafSignature {
                name: "add_mod".into(),
                arguments: vec!["n".into(), "q".into()],
                inputs: vec![column.clone(), column.clone()],
                outputs: vec![column],
                scratch_bytes: 0.into(),
                in_place: vec![(0, 0)],
                associative: true,
            },
            rv64: Some(Rv64Code::standalone(
                "add_mod",
                words,
                IntExpr::from(8) + IntExpr::var("n") * 14,
            )),
            host: Some(Arc::new(|call: HostCall<'_>| {
                let q = call.arguments[1] as u64;
                for (j, out) in call.outputs[0].chunks_mut(8).enumerate() {
                    let a =
                        u64::from_le_bytes(call.inputs[0][8 * j..8 * j + 8].try_into().unwrap());
                    let b =
                        u64::from_le_bytes(call.inputs[1][8 * j..8 * j + 8].try_into().unwrap());
                    let s = a + b;
                    out.copy_from_slice(&(if s >= q { s - q } else { s }).to_le_bytes());
                }
            })),
        })
    }

    /// Increments an `Int` of one word and reports whether it reached the argument `target`.
    fn step_leaf() -> Arc<Leaf> {
        let mut asm = Asm::new();
        asm.ld(Reg::T0, Reg::A0, 0); // target
        asm.ld(Reg::T1, Reg::A0, 8); // input address
        asm.ld(Reg::T2, Reg::A0, 16); // next value address
        asm.ld(Reg::A1, Reg::A0, 24); // flag address
        asm.ld(Reg::A2, Reg::T1, 0);
        asm.addi(Reg::A2, Reg::A2, 1);
        asm.sd(Reg::A2, Reg::T2, 0);
        asm.alu(AluOp::Xor, Reg::A3, Reg::A2, Reg::T0);
        asm.emit(crate::rv64::isa::Instr::Imm {
            op: crate::rv64::isa::ImmOp::Sltiu,
            rd: Reg::A3,
            rs1: Reg::A3,
            imm: 1,
        });
        asm.sd(Reg::A3, Reg::A1, 0);
        asm.emit(crate::rv64::isa::Instr::Jalr { rd: Reg::ZERO, rs1: Reg::RA, imm: 0 });
        Arc::new(Leaf {
            signature: LeafSignature {
                name: "step".into(),
                arguments: vec!["target".into()],
                inputs: vec![CoreType::Int { words: 1 }],
                outputs: vec![CoreType::Int { words: 1 }, CoreType::Bool],
                scratch_bytes: 0.into(),
                in_place: vec![],
                associative: false,
            },
            rv64: Some(Rv64Code::standalone("step", asm.finish().unwrap(), 11.into())),
            host: Some(Arc::new(|call: HostCall<'_>| {
                let next = host::read_int(call.inputs[0]) + 1;
                call.outputs[0].copy_from_slice(&host::write_int(next, 1));
                call.outputs[1].copy_from_slice(
                    &u64::from(next == i128::from(call.arguments[0])).to_le_bytes(),
                );
            })),
        })
    }

    struct Inputs(Vec<Vec<u8>>);

    impl Environment for Inputs {
        fn query(&mut self, request: &[u8]) -> Vec<u8> {
            match request.split_first() {
                Some((0, index)) => {
                    self.0[u64::from_le_bytes(index.try_into().unwrap()) as usize].clone()
                }
                _ => request.iter().rev().copied().collect(),
            }
        }
    }

    fn run_image(image: &ProofImage, inputs: Vec<Vec<u8>>) -> (Vec<Vec<u8>>, u64) {
        let mut machine = Machine::load(&image.bytes, image.mem_size).unwrap();
        let fuel = u64::try_from(image.step_bound).unwrap();
        let (outcome, steps) = machine.run(fuel, &mut Inputs(inputs));
        let Outcome::Halted(payload) = outcome else {
            let word = (machine.pc() - BASE) / 4;
            let code = |w: u64| {
                let at = (4 * w) as usize;
                image.bytes.get(at..at + 4).map(|b| {
                    crate::rv64::isa::Instr::decode(u32::from_le_bytes(b.try_into().unwrap()))
                })
            };
            let context: Vec<_> =
                (word.saturating_sub(6)..word + 2).map(|w| (w, code(w))).collect();
            panic!("the image did not halt: {outcome:?} at word {word}: {context:#?}")
        };
        let mut outputs = Vec::new();
        let mut rest = &payload[..];
        while !rest.is_empty() {
            let len = u64::from_le_bytes(rest[..8].try_into().unwrap()) as usize;
            outputs.push(rest[8..8 + len].to_vec());
            rest = &rest[8 + len..];
        }
        (outputs, steps)
    }

    fn words(values: &[u64]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn test_independent_loop_over_columns_matches_host() {
        check_columns(param("N"), param("K"));
    }

    /// A loop of one iteration is inlined with its index bound to 0, so its index-dependent
    /// window is a constant copy and the glue has no loop.
    #[test]
    fn test_one_iteration_loop_is_straight_line() {
        check_columns(param("N"), 1);
    }

    fn check_columns(n: i128, k: i128) {
        let q: u64 = (1 << 61) - 1;
        let add_mod = add_mod_leaf();
        let mut builder =
            ProgramBuilder::new("columns").parameter("n").parameter("k").parameter("q");
        let matrix = CoreType::buffer(8, IntExpr::var("n") * IntExpr::var("k"));
        let mut body = DefinitionBuilder::new("column", vec![matrix.clone(), matrix.clone()]);
        let column = |b: &mut DefinitionBuilder, input: usize| {
            let input = b.param(input);
            b.builtin(
                Builtin::Window {
                    offset: IntExpr::LoopIndex(0) * IntExpr::var("n") * 8,
                    runs: 1.into(),
                    length: IntExpr::var("n") * 8,
                    stride: 0.into(),
                },
                vec![input],
            )
        };
        let (a, b) = (column(&mut body, 0), column(&mut body, 1));
        let sum = body.leaf(&add_mod, vec!["n".into(), "q".into()], vec![a, b])[0];
        let body = builder.add_definition(body.finish(vec![sum]));
        let mut root = DefinitionBuilder::new("root", vec![matrix.clone(), matrix]);
        let (pa, pb) = (root.param(0), root.param(1));
        let out = root.add_loop(
            Loop {
                count: "k".into(),
                index_slot: 0,
                body,
                bindings: vec![],
                carried: vec![],
                inputs: vec![pa, pb],
                until: false,
            },
            1,
        )[0];
        let program = builder.finish(
            vec!["a".into(), "b".into()],
            root.finish(vec![out]),
            vec!["sum".into()],
        );
        let bindings =
            Bindings::from([("n".into(), n), ("k".into(), k), ("q".into(), i128::from(q))]);
        let validated = validate(Arc::new(program), bindings).unwrap();

        let mut rng = rand::rng();
        let total = (n * k) as usize;
        let a: Vec<u64> = (0..total).map(|_| rng.random_range(0..q)).collect();
        let b: Vec<u64> = (0..total).map(|_| rng.random_range(0..q)).collect();
        let inputs = vec![words(&a), words(&b)];
        let host_out = host::run(&validated, inputs.clone(), None).unwrap();
        let image = flatten(&validated, &FlattenOptions::default()).unwrap();
        let (image_out, steps) = run_image(&image, inputs);
        assert_eq!(image_out, host_out);
        assert!(u128::from(steps) <= image.step_bound);
        assert_eq!(crate::estimate::leaf_counts(&validated)["add_mod"], k as u128);
        let (events, _) = image.trace.as_ref().expect("a trace");
        let loops = events.iter().filter(|e| matches!(e, GlueEvent::Loop { .. })).count();
        assert_eq!(loops, usize::from(k > 1), "a loop event exactly when k > 1");
    }

    #[test]
    fn test_until_loop_and_builtins_match_host() {
        let target = param("TARGET").max(1);
        let step = step_leaf();
        let mut builder = ProgramBuilder::new("until").parameter("target");
        let int = CoreType::Int { words: 1 };
        let mut body = DefinitionBuilder::new("step", vec![int.clone()]);
        let counter = body.param(0);
        let outs = body.leaf(&step, vec!["target".into()], vec![counter]);
        let body = builder.add_definition(body.finish(vec![outs[0], outs[1]]));

        let mut root = DefinitionBuilder::new(
            "root",
            vec![
                int.clone(),
                int.clone(),
                CoreType::bytes(16),
                CoreType::bytes(16),
                CoreType::bytes(5),
            ],
        );
        let (start, selector, x, y, request) =
            (root.param(0), root.param(1), root.param(2), root.param(3), root.param(4));
        let finished = root.add_loop(
            Loop {
                count: IntExpr::var("target") + 10,
                index_slot: 1,
                body,
                bindings: vec![],
                carried: vec![start],
                inputs: vec![],
                until: true,
            },
            0,
        )[0];
        let chosen = root.builtin(Builtin::Select, vec![selector, x, y]);
        let packed = root.builtin(Builtin::Pack, vec![x, y]);
        let second = root.builtin(Builtin::Get { index: 1.into() }, vec![packed]);
        let dynamic = root.builtin(Builtin::GetDynamic, vec![packed, selector]);
        let joined = root.builtin(Builtin::Concat, vec![chosen, second]);
        let strided = root.builtin(
            Builtin::Window {
                offset: 2.into(),
                runs: 3.into(),
                length: 2.into(),
                stride: 5.into(),
            },
            vec![joined],
        );
        let response = root.builtin(Builtin::Query { response_bytes: 5.into() }, vec![request]);
        let coins = root.builtin(Builtin::Coins { bytes: 0.into() }, vec![]);
        let program = builder.finish(
            vec!["start".into(), "selector".into(), "x".into(), "y".into(), "request".into()],
            root.finish(vec![finished, chosen, dynamic, joined, strided, response, coins]),
            ["finished", "chosen", "dynamic", "joined", "strided", "response", "coins"]
                .map(String::from)
                .to_vec(),
        );
        let validated =
            validate(Arc::new(program), Bindings::from([("target".into(), target)])).unwrap();

        let mut rng = rand::rng();
        let mut bytes = |n: usize| (0..n).map(|_| rng.random::<u8>()).collect::<Vec<u8>>();
        let inputs = vec![
            host::write_int(0, 1),
            host::write_int(rand::rng().random_range(0..2), 1),
            bytes(16),
            bytes(16),
            bytes(5),
        ];
        let reverse = |r: &[u8]| r.iter().rev().copied().collect::<Vec<u8>>();
        let host_out = host::run(&validated, inputs.clone(), Some(&reverse)).unwrap();
        assert_eq!(host::read_int(&host_out[0]), target);
        let image = flatten(&validated, &FlattenOptions::default()).unwrap();
        let (image_out, steps) = run_image(&image, inputs);
        assert_eq!(image_out, host_out);
        assert!(u128::from(steps) <= image.step_bound);
    }

    #[test]
    fn test_query_inside_independent_loop_is_rejected() {
        let mut builder = ProgramBuilder::new("s4");
        let mut body = DefinitionBuilder::new("body", vec![CoreType::bytes(4)]);
        let request = body.param(0);
        let response = body.builtin(Builtin::Query { response_bytes: 4.into() }, vec![request]);
        let body = builder.add_definition(body.finish(vec![response]));
        let mut root = DefinitionBuilder::new("root", vec![CoreType::bytes(4)]);
        let input = root.param(0);
        let out = root.add_loop(
            Loop {
                count: 2.into(),
                index_slot: 0,
                body,
                bindings: vec![],
                carried: vec![],
                inputs: vec![input],
                until: false,
            },
            1,
        )[0];
        let program = builder.finish(vec!["r".into()], root.finish(vec![out]), vec!["out".into()]);
        let error = validate(Arc::new(program), Bindings::new()).unwrap_err();
        assert!(error.to_string().contains("S4"), "{error}");
    }
}
