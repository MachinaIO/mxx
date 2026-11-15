//! Leaves: the only nodes that compute.
//!
//! A DSL supplies a [`Leaf`]: a [`LeafSignature`] and the code that implements it, as RV64IM code
//! for proof images ([`Rv64Code`]) and as a host kernel ([`HostKernel`]). The core supplies the
//! [`Builtin`] leaves itself. Every leaf obeys the leaf contract:
//!
//! - **L1.** Its running time depends on its arguments and types, not on the bytes of its inputs.
//! - **L2.** It reads only its inputs and scratch, and writes only its outputs and scratch.
//! - **L3.** It allocates nothing.
//! - **L4.** It keeps no state between calls: it is a function of its arguments and inputs. Only
//!   the built-in `Coins` and `Query` leaves have effects.
//! - **L5.** It does not synchronize with the host.
//!
//! # The RV64IM calling convention
//!
//! The core calls a leaf's code with `jal ra, entry` and `a0` pointing at a descriptor of
//! little-endian `u64` words: the arguments in order (two's complement), then the address of each
//! input, then the address of each output, then the address of the scratch area. Each value is
//! laid out in its byte encoding ([`crate::types`]). The leaf returns with `ret`, keeps `sp` and
//! the saved registers `s0`–`s11` as the standard ABI requires, may use the stack below `sp`, and
//! makes no environment call. The code is position independent: it runs at whatever address the
//! core places it, and it reaches nothing outside itself except through the descriptor.

use crate::{expr::IntExpr, types::CoreType};
use std::{fmt, sync::Arc};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafSignature {
    pub name: String,
    /// The names of the compile-expression arguments. Input, output, and scratch sizes, and the
    /// step bound, may refer to them as parameters.
    pub arguments: Vec<String>,
    pub inputs: Vec<CoreType>,
    pub outputs: Vec<CoreType>,
    pub scratch_bytes: IntExpr,
    /// `(output, input)` pairs: the output may share storage with the input when the input has no
    /// later reader.
    pub in_place: Vec<(usize, usize)>,
    /// A hint: the leaf is a fold of two inputs of its output's type into one, and may be
    /// regrouped. The DSL proves associativity with the leaf's specification.
    pub associative: bool,
}

/// Position-independent RV64IM code that a proof image holds once, such as every leaf of one DSL
/// compiled together with the helpers they share. Its words are in the instruction set of the
/// mathcal-A machine model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv64Library {
    pub name: String,
    pub words: Vec<u32>,
    /// Read-only data the code reaches pc-relatively, placed right after the last word.
    pub rodata: Vec<u8>,
}

/// A leaf's entry into an RV64IM library.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv64Code {
    pub library: Arc<Rv64Library>,
    /// The index of the entry instruction in the library.
    pub entry: u64,
    /// An upper bound on the steps of one call, over the signature's argument names.
    pub steps: IntExpr,
}

impl Rv64Code {
    /// Code that is a library of its own, entered at its first word.
    pub fn standalone(name: impl Into<String>, words: Vec<u32>, steps: IntExpr) -> Self {
        Self {
            library: Arc::new(Rv64Library { name: name.into(), words, rodata: Vec::new() }),
            entry: 0,
            steps,
        }
    }
}

/// One call of a host kernel. Every value is in its byte encoding.
pub struct HostCall<'a> {
    pub arguments: &'a [i64],
    pub inputs: &'a [&'a [u8]],
    /// Zero-initialized and of the declared sizes.
    pub outputs: &'a mut [Vec<u8>],
    pub scratch: &'a mut [u8],
}

pub trait HostKernel: Send + Sync {
    fn run(&self, call: HostCall<'_>);
}

impl<F: Fn(HostCall<'_>) + Send + Sync> HostKernel for F {
    fn run(&self, call: HostCall<'_>) {
        self(call)
    }
}

#[derive(Clone)]
pub struct Leaf {
    pub signature: LeafSignature,
    pub rv64: Option<Rv64Code>,
    pub host: Option<Arc<dyn HostKernel>>,
}

impl fmt::Debug for Leaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Leaf")
            .field("signature", &self.signature)
            .field("rv64", &self.rv64.as_ref().map(|c| (&c.library.name, c.entry)))
            .field("host", &self.host.is_some())
            .finish()
    }
}

/// The leaves the core provides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Builtin {
    /// From one buffer, `runs` runs of `length` bytes, run `r` starting at `offset + r * stride`,
    /// concatenated.
    Window { offset: IntExpr, runs: IntExpr, length: IntExpr, stride: IntExpr },
    /// The concatenation of its inputs' encodings, as one buffer: buffers, integers, Bools, and
    /// families (a family's encoding is its members', in order), so a family of entries becomes
    /// the buffer of a matrix.
    Concat,
    /// A family made of its inputs, in order.
    Pack,
    /// Member `index` of a family.
    Get { index: IntExpr },
    /// Member `index` of a uniform family, for the runtime `Int` `index` (its second input).
    GetDynamic,
    /// Its input `1 + selector` for the runtime `Int` selector (its first input). Every candidate
    /// has the same type.
    Select,
    /// `bytes` fresh uniform bytes.
    Coins { bytes: IntExpr },
    /// Sends its input buffer to the external party and returns the response, which must have
    /// exactly `response_bytes` bytes.
    Query { response_bytes: IntExpr },
    /// The value of type `ty` whose bytes are `bytes`. A proof image holds them in its read-only
    /// data and copies them, so constants add no code to it.
    Constant { ty: crate::types::CoreType, bytes: Vec<u8> },
}

#[derive(Clone, Debug)]
pub enum LeafRef {
    Builtin(Builtin),
    Dsl { leaf: Arc<Leaf>, arguments: Vec<IntExpr> },
}

impl LeafRef {
    pub fn name(&self) -> &str {
        match self {
            Self::Builtin(Builtin::Window { .. }) => "window",
            Self::Builtin(Builtin::Concat) => "concat",
            Self::Builtin(Builtin::Pack) => "pack",
            Self::Builtin(Builtin::Get { .. }) => "get",
            Self::Builtin(Builtin::GetDynamic) => "get_dynamic",
            Self::Builtin(Builtin::Select) => "select",
            Self::Builtin(Builtin::Coins { .. }) => "coins",
            Self::Builtin(Builtin::Query { .. }) => "query",
            Self::Builtin(Builtin::Constant { .. }) => "constant",
            Self::Dsl { leaf, .. } => &leaf.signature.name,
        }
    }
}
