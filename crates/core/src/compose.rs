//! Generated run proofs of proof images.
//!
//! When an image's glue has a trace ([`ProofImage::trace`]), [`run_module`] writes a Lean module
//! that proves what the image computes from the specifications of its leaves alone, without
//! knowing what any leaf computes. It follows the trace: each glue segment between two events is
//! one `mrun` theorem about the concrete program (registers set, words stored, every other word
//! kept), and each event is one lemma of `MxxCore.Compose` (`query`, `leaf`, `copy`, `loop`,
//! `halt`). Along the way the generator tracks which bytes each memory range holds, as Lean
//! expressions over the inputs (`ins.getD i []`), the image, and the leaves' functions
//! (`L.f args xs`), and passes each fact through every later event by its frame.
//!
//! A counted loop is one use of `Compose.loop` with an invariant at its head: the index slot holds
//! `i`, the facts from before the loop that it leaves alone still hold, and of each family the
//! loop fills one member per iteration, members `[0, i)` hold their values (`Compose.Members`).
//! The body is generated once, for a symbolic index `i < N`: addresses are affine in the indices
//! of the enclosing loops (`flatten::Addr`), and conditions on them are proved by `omega`. After
//! the loop, each family holds its members' concatenation.
//!
//! The result is one theorem, `halts`: the run halts within the image's step bound with the
//! payload the tracked expressions describe. Its proof is one term, so the step count is inferred
//! from the events, then bounded by `omega`. Its hypotheses are the input lengths (`hin`), and for
//! each leaf call (for every index of its enclosing loops) its domain (`hdom`), its output lengths
//! (`hlen`), and that it takes no more steps than the bound grants it (`hsteps`). A claim's proof
//! discharges them for its own leaves; the run itself needs no hand proof. The generator is not
//! trusted: Lean checks everything it writes.

use crate::{
    flatten::{Addr, Argument, GlueEvent, ProofImage},
    rv64::isa::Instr,
};
use std::{collections::BTreeMap, fmt::Write};

/// How the Lean side names one leaf library.
#[derive(Clone, Debug)]
pub struct LeanLeaf {
    /// The module that proves the leaf's specification.
    pub module: String,
    /// `At : Prog → Nat → Prop`, the leaf's code from its entry at an instruction (a structure
    /// with one field per instruction, proved by `constructor <;> rfl`).
    pub code_at: String,
    /// A function of `(hcode : At P c)` and one closed condition on `c` (proved by
    /// `norm_num [BASE]`) to `MxxCore.Leaf.Spec P c`.
    pub spec: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ComposeError {
    #[error("the glue has no trace: {0}")]
    NoTrace(&'static str),
    #[error("not supported by the generator yet: {0}")]
    Unsupported(String),
    #[error("no Lean specification for leaf `{0}`")]
    NoSpec(String),
    #[error("the glue from instruction {from} to {to} is not straight-line: {why}")]
    Segment { from: u64, to: u64, why: String },
    #[error("no tracked bytes cover [{addr}, {addr} + {len})")]
    Untracked { addr: String, len: u64 },
    #[error("copy of {len} bytes from {src} to {dst} is not word-aligned")]
    Unaligned { dst: String, src: String, len: u64 },
    #[error("the run takes {steps} steps, above the image's bound {bound}")]
    Bound { steps: u128, bound: u128 },
}

/// An affine Lean term `base + Σ factor · var` over the indices of enclosing loops, outermost
/// first.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Sym {
    base: i128,
    terms: Vec<(String, i128)>,
}

impl Sym {
    fn num(value: u64) -> Self {
        Self { base: value.into(), terms: Vec::new() }
    }

    fn concrete(&self) -> Option<u64> {
        self.terms.is_empty().then_some(self.base as u64)
    }

    fn lean(&self) -> String {
        if self.terms.is_empty() {
            return self.base.to_string();
        }
        if let [(v, 1)] = &self.terms[..] &&
            self.base == 0
        {
            return v.clone();
        }
        let terms: String = self.terms.iter().map(|(v, f)| format!(" + {f} * {v}")).collect();
        format!("({}{terms})", self.base)
    }

    fn minus(&self, other: &Sym) -> Option<Sym> {
        let mut terms = self.terms.clone();
        for (v, f) in &other.terms {
            match terms.iter_mut().find(|(w, _)| w == v) {
                Some((_, g)) => *g -= f,
                None => terms.push((v.clone(), -f)),
            }
        }
        terms.retain(|(_, f)| *f != 0);
        (self.base >= other.base && terms.iter().all(|(_, f)| *f > 0))
            .then(|| Sym { base: self.base - other.base, terms })
    }

    /// The least and greatest values over the indices' ranges.
    fn range(&self, ctx: &[Index]) -> (i128, i128) {
        let mut range = (self.base, self.base);
        for (v, f) in &self.terms {
            let top = ctx.iter().find(|i| &i.var == v).map_or(0, |i| i.count as i128 - 1);
            range.0 += (f * top).min(0);
            range.1 += (f * top).max(0);
        }
        range
    }
}

/// A loop index in scope: its slot, Lean variable and bound hypothesis, count, and the address of
/// its slot.
#[derive(Clone, Debug)]
struct Index {
    slot: u32,
    var: String,
    hyp: String,
    count: u64,
    addr: u64,
    /// Whether it is an `until` loop's stopping iteration, a variable of `halts`.
    stop: bool,
}

/// One memory range whose bytes a Lean fact (`name : Holds s addr expr`) states, with the fact
/// `length : expr.length = len`.
#[derive(Clone, Debug)]
struct Fact {
    name: String,
    addr: Sym,
    len: u64,
    expr: String,
    length: String,
    image: bool,
}

/// The members `[0, i)` of a family the current loop fills (`name : Compose.Members s base n O i`).
#[derive(Clone, Debug)]
struct Family {
    name: String,
    base: Sym,
    n: u64,
    /// The index of the loop that fills it, and its count.
    var: String,
    count: u64,
    /// `O`, or a placeholder until the loop's body is generated.
    members: String,
}

const BASE: u64 = crate::rv64::machine::BASE;

/// The longest image slice stated as a literal (descriptors, requests, lengths); longer ones stay
/// slices of the image.
const MAX_LITERAL: u64 = 256;

/// The most facts one read joins.
const MAX_JOIN: usize = 64;

/// The slot of a run-time value in scope like an index (a member number read from memory).
const DATA: u32 = u32::MAX;

/// A run-time member number a segment reads: the integer's address, its fact, bytes, and words,
/// the family's count, the hypothesis instance bounding it, and its length fact.
#[derive(Clone, Debug)]
struct Member {
    at: u64,
    fact: String,
    expr: String,
    words: u64,
    count: u64,
    bound: String,
    length: String,
}

/// The tactic that closes what `mrun` leaves of a glue segment, with the facts `words` about the
/// words it reads (memory reads after stores need them once the stores are simplified away).
fn close(words: &[String], extra: &str) -> String {
    let with = words.iter().map(|w| format!(", {w}")).collect::<String>();
    format!(
        "  all_goals and_intros
  all_goals first
    | exact fun _ => trivial
    | (simp only [P_len, P_mem, BASE]; omega)
    | (intro w hw; simp [hw{with}]; done)
    | (simp [{}] <;> bv_omega)
    | bv_omega
    | (simp [BitVec.ult] at hbr; omega)
    | (simp at hbr; done)
    | (simp [P_len, P_mem, BASE{with}] <;> omega){extra}
    | (simp at *; done)",
        words.join(", ")
    )
}

fn apart(a: (i128, i128), b: (i128, i128)) -> bool {
    a.1 <= b.0 || b.1 <= a.0
}

/// The source of a runtime selection, stated once the segment before its copy is proved: its
/// fact and length fact's names, address and bytes as Lean terms, the candidates' bytes and
/// addresses as Lean lists, the candidates' addresses, their length, and the proof that the
/// selector is below the number of candidates.
#[derive(Clone, Debug)]
struct SelectSource {
    name: String,
    length: String,
    address: String,
    expr: String,
    list: String,
    addrs: String,
    candidates: Vec<u64>,
    len: u64,
    within: String,
}

/// The stop flag of an `until` loop that a segment reads: its address, fact, bytes, and length
/// fact, its value (0 to continue, 1 to stop), and the hypothesis that gives the value.
#[derive(Clone, Debug)]
struct Flag {
    at: u64,
    fact: String,
    expr: String,
    length: String,
    value: u64,
    hyp: String,
}

/// A table in read-only data that a segment reads at index `var`: its address, its words as a
/// Lean list, its length, and the fact that holds it.
#[derive(Clone, Debug)]
struct Table {
    at: u64,
    list: String,
    var: String,
    len: usize,
    fact: String,
}

struct Gen<'a> {
    image: &'a ProofImage,
    leaves: &'a BTreeMap<String, LeanLeaf>,
    instrs: Vec<Instr>,
    /// Lemmas before the proof: image slices and glue segments.
    lemmas: String,
    /// The term being written.
    proof: String,
    facts: Vec<Fact>,
    families: Vec<Family>,
    state: String,
    fresh: usize,
    image_lemmas: BTreeMap<(u64, u64, String), String>,
    ctx: Vec<Index>,
    checked: Option<(String, u64)>,
    pos: u64,
    /// A bound on the steps of the events generated so far, in the current body.
    total: u128,
    specs: String,
    spec_names: BTreeMap<u64, String>,
    modules: Vec<String>,
    hypotheses: Vec<String>,
    budgets: Vec<String>,
    calls: usize,
    /// The leaf calls directly in the current body, whose step budgets it uses.
    local_calls: Vec<usize>,
    phases: usize,
    family_ids: usize,
    /// The tables the next segment reads.
    tables: Vec<Table>,
    /// The run-time member number the next segment reads.
    member: Option<Member>,
    /// The source of the selection whose copy comes next.
    select: Option<SelectSource>,
    /// The variables in scope, as a quantifier and its instance arguments.
    scope: Vec<(String, String)>,
    /// The draws of the run so far outside loops, as binds around its result.
    wrappers: Vec<String>,
    /// The draw of the current loop's iteration: its variable, length fact, and bytes.
    draw: Option<(String, String, u64)>,
    /// The ranges read after the current event.
    live: Vec<(i128, i128)>,
    /// Definitions of long expressions, and their count.
    defs: String,
    defs_count: usize,
    /// The stop flag the next segment reads, and its value there.
    flag: Option<Flag>,
    /// Variables of `halts` itself (stopping iterations), with their types.
    globals: Vec<(String, String)>,
}

impl Gen<'_> {
    fn name(&mut self, prefix: &str) -> String {
        self.fresh += 1;
        format!("{prefix}{}", self.fresh)
    }

    fn sym(&self, addr: &Addr) -> Result<Sym, ComposeError> {
        sym_in(&self.ctx, addr)
    }
}

/// `addr` over the indices of `ctx`.
fn sym_in(ctx: &[Index], addr: &Addr) -> Result<Sym, ComposeError> {
    {
        let mut terms = Vec::new();
        for index in ctx {
            if let Some((_, f)) = addr.terms.iter().find(|(s, _)| *s == index.slot) {
                if *f < 0 {
                    return Err(ComposeError::Unsupported("a decreasing address".into()));
                }
                terms.push((index.var.clone(), *f));
            }
        }
        if terms.len() != addr.terms.len() {
            return Err(ComposeError::Unsupported("an address over an index out of scope".into()));
        }
        Ok(Sym { base: addr.base.into(), terms })
    }
}

impl Gen<'_> {
    /// A proof of a condition on addresses over the indices in scope: `simp` with `lemmas`, then
    /// `omega`, with the bounds of run-time member numbers in the same normal form.
    fn omega_closer(&self, lemmas: &str) -> String {
        let data: Vec<&str> =
            self.ctx.iter().filter(|i| i.slot == DATA).map(|i| i.hyp.as_str()).collect();
        if data.is_empty() {
            format!("(by simp [{lemmas}] <;> omega)")
        } else {
            format!("(by simp [{lemmas}] at {} ⊢ <;> omega)", data.join(" "))
        }
    }

    /// `expr`, or for a long one a call of a new definition of it over the variables in scope,
    /// so that facts built on facts stay short (`ins`, loop indices, draws, and the current
    /// carried values, which stay arguments so a loop's step can be read off a call).
    fn name_expr(&mut self, expr: &str) -> String {
        if expr.len() < 160 {
            return expr.to_owned();
        }
        let mut params = vec![("ins".to_owned(), "List Bytes".to_owned())];
        let mut args = vec!["ins".to_owned()];
        for (v, ty) in &self.globals {
            params.push((v.clone(), ty.clone()));
            args.push(v.clone());
        }
        for (quant, inst) in &self.scope {
            let var = inst.split_whitespace().next().expect("a variable").to_owned();
            let ty = if quant.contains(" < ") {
                "Nat"
            } else if quant.contains("List Bytes") {
                "List Bytes"
            } else {
                "Bytes"
            };
            params.push((var.clone(), ty.to_owned()));
            args.push(var);
        }
        // Carried values in scope: `(__Cn__ i)`.
        let mut body = expr.to_owned();
        let mut rest = expr;
        while let Some(at) = rest.find("(__C") {
            let tail = &rest[at..];
            let Some(close) = tail.find(')') else { break };
            let call = &tail[..=close];
            let id = call.trim_start_matches("(__C").split("__").next().unwrap_or_default();
            let param = format!("c{id}");
            if !params.iter().any(|(p, _)| *p == param) {
                body = body.replace(call, &param);
                params.push((param, "Bytes".into()));
                args.push(call.to_owned());
            }
            rest = &rest[at + close + 1..];
        }
        let name = format!("e{}", self.defs_count);
        self.defs_count += 1;
        let binders: String = params.iter().map(|(p, t)| format!(" ({p} : {t})")).collect();
        writeln!(self.defs, "noncomputable def {name}{binders} : Bytes :=\n  {body}\n").unwrap();
        format!("({name} {})", args.join(" "))
    }

    /// The loop indices in scope (not the run-time values bounded like them).
    fn loops(&self) -> impl Iterator<Item = &Index> {
        self.ctx.iter().filter(|i| i.slot != DATA)
    }

    /// `∀ i0 < N0, ∀ b0 : Bytes, b0.length = n → ` over the variables in scope: loop indices,
    /// drawn bytes, and the draws of loops that drew.
    fn quantifiers(&self) -> String {
        self.scope.iter().map(|(q, _)| q.as_str()).collect()
    }

    /// ` i0 hi0 b0 hb0` over the variables in scope.
    fn instances(&self) -> String {
        self.scope.iter().map(|(_, i)| i.as_str()).collect()
    }

    /// The quantifiers over the loop indices in scope alone: step budgets do not depend on draws
    /// (a stopping iteration is a variable of `halts`, not quantified).
    fn loop_quantifiers(&self) -> String {
        self.loops().filter(|i| !i.stop).map(|i| format!("∀ {} < {}, ", i.var, i.count)).collect()
    }

    /// The instance arguments of `loop_quantifiers`.
    fn loop_instances(&self) -> String {
        self.loops().filter(|i| !i.stop).map(|i| format!(" {} {}", i.var, i.hyp)).collect()
    }

    /// A Lean term for a leaf argument.
    /// A leaf argument as a Lean term, whether it reads an index, and the table it reads.
    fn argument(
        &mut self,
        argument: &Argument,
    ) -> Result<(String, bool, Option<Table>), ComposeError> {
        match argument {
            Argument::Affine { constant, terms } if *constant >= 0 => {
                let sym =
                    sym_in(&self.ctx, &Addr { base: *constant as u64, terms: terms.clone() })?;
                Ok((sym.lean(), sym.concrete().is_none(), None))
            }
            Argument::Table { slot, at, values } if values.iter().all(|v| *v >= 0) => {
                let var = self
                    .ctx
                    .iter()
                    .find(|i| i.slot == *slot)
                    .ok_or_else(|| {
                        ComposeError::Unsupported("a table over an index out of scope".into())
                    })?
                    .var
                    .clone();
                let list = format!(
                    "[{}]",
                    values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
                );
                let fact = self.read(&Sym::num(*at), 8 * values.len() as u64, None)?;
                let table = Table {
                    at: *at,
                    list: list.clone(),
                    var: var.clone(),
                    len: values.len(),
                    fact: fact.name,
                };
                Ok((format!("({list}.getD {var} 0)"), true, Some(table)))
            }
            _ => Err(ComposeError::Unsupported(format!("the argument {argument:?}"))),
        }
    }

    /// The image bytes `[off, off + len)` as a Lean term: words when aligned whole words.
    fn image_value(&self, off: u64, len: u64, request: Option<u64>) -> String {
        let bytes = &self.image.bytes[off as usize..(off + len) as usize];
        if let Some(input) = request {
            return format!("0 :: Bytes.u64 {input}");
        }
        if off.is_multiple_of(8) && len.is_multiple_of(8) {
            let words: Vec<String> = bytes
                .chunks(8)
                .map(|w| u64::from_le_bytes(w.try_into().expect("8 bytes")).to_string())
                .collect();
            format!("Bytes.u64s [{}]", words.join(", "))
        } else {
            let list: Vec<String> = bytes.iter().map(|b| format!("0x{b:02x}")).collect();
            format!("[{}]", list.join(", "))
        }
    }

    /// A fact stating exactly `[addr, addr + len)`, from the tracked facts.
    fn read(&mut self, addr: &Sym, len: u64, request: Option<u64>) -> Result<Fact, ComposeError> {
        if request.is_none() &&
            let Some(f) = self.facts.iter().find(|f| f.addr == *addr && f.len == len && !f.image)
        {
            return Ok(f.clone());
        }
        let untracked = || ComposeError::Untracked { addr: addr.lean(), len };
        let Some(at0) = addr.concrete() else {
            // A symbolic range lies inside one fact at a closed address.
            let (lo, hi) = addr.range(&self.ctx);
            let f = self
                .facts
                .iter()
                .find(|f| {
                    f.addr.concrete().is_some_and(|a| {
                        !f.image &&
                            i128::from(a) <= lo &&
                            hi + i128::from(len) <= i128::from(a + f.len)
                    })
                })
                .cloned();
            let f = match f {
                Some(f) => f,
                // The range moves over adjacent facts: join those covering all its positions.
                None => self.read(&Sym::num(lo as u64), (hi - lo) as u64 + len, None)?,
            };
            return self.slice(&f, addr, len, request);
        };
        let mut pieces = Vec::new();
        let mut at = at0;
        while at < at0 + len || (len == 0 && pieces.is_empty()) {
            let f = self
                .facts
                .iter()
                .filter(|f| f.addr.concrete().is_some_and(|a| a <= at && at < a + f.len))
                .max_by_key(|f| !f.image)
                .cloned()
                .ok_or_else(untracked)?;
            let end = (f.addr.concrete().expect("closed") + f.len).min(at0 + len);
            pieces.push(self.slice(&f, &Sym::num(at), end - at, request)?);
            at = end;
            if len == 0 {
                break;
            }
        }
        // Joined expressions grow with every piece; a range over many facts is out of reach.
        if pieces.len() > MAX_JOIN {
            return Err(ComposeError::Unsupported(format!(
                "a read of [{}, +{len}) over {} facts",
                addr.lean(),
                pieces.len()
            )));
        }
        let mut acc = pieces.remove(0);
        for piece in pieces {
            let name = self.name("a");
            let length = self.name("la");
            let full = format!("({}) ++ ({})", acc.expr, piece.expr);
            let expr = self.name_expr(&full);
            writeln!(
                self.proof,
                "  have {name} : Holds {} {} ({expr}) :=\n    Compose.append {} {} {} (by decide)",
                self.state,
                acc.addr.lean(),
                acc.name,
                acc.length,
                piece.name
            )
            .unwrap();
            writeln!(
                self.proof,
                "  have {length} : ({expr}).length = {} := by\n    show ({full}).length = _\n    rw [List.length_append, {}, {}]",
                acc.len + piece.len,
                acc.length,
                piece.length
            )
            .unwrap();
            acc =
                Fact { name, addr: acc.addr, len: acc.len + piece.len, expr, length, image: false };
        }
        Ok(acc)
    }

    /// The fact for `[at, at + len)` inside the tracked fact `f`.
    fn slice(
        &mut self,
        f: &Fact,
        at: &Sym,
        len: u64,
        request: Option<u64>,
    ) -> Result<Fact, ComposeError> {
        let off =
            at.minus(&f.addr).ok_or_else(|| ComposeError::Untracked { addr: at.lean(), len })?;
        if !f.image && off == Sym::num(0) && len == f.len {
            return Ok(f.clone());
        }
        let name = self.name("p");
        let length = self.name("lp");
        let expr = if f.image && len > MAX_LITERAL {
            // A large slice of the image stays a slice: no literal, nothing to decide.
            let off = off.concrete().expect("image slices are closed");
            let value = format!("(P.bin.drop {off}).take {len}");
            writeln!(
                self.proof,
                "  have {name} : Holds {} {} ({value}) :=\n    Compose.slice {} {off} {len} {} _ (by decide) (by rw [P_len] <;> decide) rfl\n  have {length} : ({value}).length = {len} := by\n    rw [List.length_take, List.length_drop, P_len] <;> decide",
                self.state,
                at.lean(),
                f.name,
                at.lean()
            )
            .unwrap();
            value
        } else if f.image {
            let off = off.concrete().expect("image slices are closed");
            let value = self.image_value(off, len, request);
            let key = (off, len, value.clone());
            let lemma = match self.image_lemmas.get(&key) {
                Some(lemma) => lemma.clone(),
                None => {
                    let lemma = format!("img{}", self.image_lemmas.len());
                    writeln!(
                        self.lemmas,
                        "set_option maxRecDepth 1000000 in\ntheorem {lemma} : (P.bin.drop {off}).take {len} = {value} := by decide\n"
                    )
                    .unwrap();
                    self.image_lemmas.insert(key, lemma.clone());
                    lemma
                }
            };
            writeln!(
                self.proof,
                "  have {name} : Holds {} {} ({value}) :=\n    Compose.slice {} {off} {len} {} _ (by decide) (by rw [P_len] <;> decide) {lemma}",
                self.state,
                at.lean(),
                f.name,
                at.lean()
            )
            .unwrap();
            writeln!(self.proof, "  have {length} : ({value}).length = {len} := by decide")
                .unwrap();
            value
        } else {
            let full = format!("(({}).drop {}).take {len}", f.expr, off.lean());
            let expr = self.name_expr(&full);
            writeln!(
                self.proof,
                "  have {name} : Holds {} {} ({expr}) :=\n    Compose.slice {} {} {len} {} _ (by omega) (by rw [{}] <;> omega) rfl",
                self.state,
                at.lean(),
                f.name,
                off.lean(),
                at.lean(),
                f.length
            )
            .unwrap();
            writeln!(
                self.proof,
                "  have {length} : ({expr}).length = {len} := by\n    show ({full}).length = {len}\n    rw [List.length_take, List.length_drop, {}] <;> omega",
                f.length
            )
            .unwrap();
            expr
        };
        Ok(Fact { name, addr: at.clone(), len, expr, length, image: false })
    }

    /// Passes every tracked fact and family apart from `written` through the frame `frame`, into
    /// the new state; drops the others.
    fn transfer(&mut self, frame: &str, written: &[(Sym, u64)]) {
        self.transfer_closed(frame, written, true);
    }

    /// `transfer`, with `closed_frame` false when the frame's statement mentions an index even if
    /// the written addresses do not (the stored values of a segment).
    fn transfer_closed(&mut self, frame: &str, written: &[(Sym, u64)], closed_frame: bool) {
        let ranges: Vec<(i128, i128)> = written
            .iter()
            .map(|(a, n)| {
                let (lo, hi) = a.range(&self.ctx);
                (lo, hi + i128::from(*n))
            })
            .collect();
        let concrete = closed_frame && written.iter().all(|(a, _)| a.concrete().is_some());
        let facts = std::mem::take(&mut self.facts);
        for f in facts {
            let (lo, hi) = f.addr.range(&self.ctx);
            if !ranges.iter().all(|r| apart((lo, hi + i128::from(f.len)), *r)) {
                continue;
            }
            // A fact nothing reads any more is dropped (the image is always kept).
            let span = (lo, hi + i128::from(f.len));
            if !f.image && self.live.iter().all(|r| apart(span, *r)) {
                continue;
            }
            let name = self.name("f");
            writeln!(
                self.proof,
                "  have {name} : Holds {} {} ({}) :=\n    {frame} {} {} _ {} (by omega) {} {}",
                self.state,
                f.addr.lean(),
                f.expr,
                f.addr.lean(),
                f.len,
                f.length,
                if concrete && f.addr.concrete().is_some() {
                    "(by decide)".to_owned()
                } else {
                    self.omega_closer("Leaf.Apart, BASE")
                },
                f.name
            )
            .unwrap();
            self.facts.push(Fact { name, ..f });
        }
        let families = std::mem::take(&mut self.families);
        for family in families {
            // Members `[0, i)` lie below the member `i` the iteration writes.
            let (lo, hi) = family.base.range(&self.ctx);
            let whole = (lo, hi + i128::from(family.n) * i128::from(family.count));
            let mut member = family.base.clone();
            member.terms.push((family.var.clone(), family.n.into()));
            let kept = written
                .iter()
                .zip(&ranges)
                .all(|((a, n), r)| (*a == member && *n == family.n) || apart(whole, *r));
            if !kept {
                continue;
            }
            let name = self.name("m");
            writeln!(
                self.proof,
                "  have {name} := Compose.members_frame {frame} (fun j hj => ⟨by omega, by simp [Leaf.Apart, BASE] <;> omega⟩) {}",
                family.name
            )
            .unwrap();
            self.families.push(Family { name, ..family });
        }
    }

    /// The steps of the glue from instruction `from` to `to`, its branches not taken to faults.
    fn segment(&self, from: u64, to: u64) -> Result<u64, ComposeError> {
        let (mut at, mut steps) = (from, 0);
        while at < to {
            let instr = self.instrs.get(at as usize).ok_or_else(|| ComposeError::Segment {
                from,
                to,
                why: "past the code".into(),
            })?;
            at += match instr {
                // An inverted branch over a jump to the fault handler: taken on success.
                Instr::Branch { offset: 8, .. } => 2,
                Instr::Branch { .. } | Instr::Jal { .. } | Instr::Jalr { .. } | Instr::Ecall => {
                    return Err(ComposeError::Segment {
                        from,
                        to,
                        why: format!("{instr:?} at {at}"),
                    });
                }
                _ => 1,
            };
            steps += 1;
        }
        if at != to {
            return Err(ComposeError::Segment { from, to, why: "overshoots".into() });
        }
        Ok(steps)
    }

    /// The facts of the words a segment reads: each index in scope, and `extra`.
    fn words(&self, extra: &[(u64, String)]) -> Result<Vec<(u64, String, String)>, ComposeError> {
        self.loops()
            .map(|i| (i.addr, i.var.clone()))
            .chain(extra.iter().cloned())
            .map(|(addr, value)| {
                let fact = self
                    .facts
                    .iter()
                    .find(|f| f.addr == Sym::num(addr) && f.expr == format!("Bytes.u64 {value}"))
                    .ok_or_else(|| ComposeError::Untracked { addr: addr.to_string(), len: 8 })?;
                Ok((addr, value, fact.name.clone()))
            })
            .collect()
    }

    /// A glue segment from `from` to `to` in `steps` steps that sets `regs` and stores `stores`,
    /// reading the index words and `extra`; continues the proof in the state after it, returning
    /// the register facts' names and the name of the stored words' fact.
    fn phase(
        &mut self,
        from: u64,
        to: u64,
        steps: u64,
        regs: &[(u32, Sym)],
        stores: &[(u64, String)],
        extra: &[(u64, String)],
    ) -> Result<(Vec<String>, String), ComposeError> {
        self.phases += 1;
        let lemma = format!("phase{}", self.phases);
        let words = self.words(extra)?;
        let first = self.state == "P.init";
        let start = if first { "P.init" } else { "t" };
        let mut binders = if first { String::new() } else { " (t : MachineState)".to_owned() };
        let mut simp: Vec<String> = Vec::new();
        let mut body = String::new();
        for (k, (addr, value, _)) in words.iter().enumerate() {
            let bound = self
                .ctx
                .iter()
                .find(|i| &i.var == value)
                .map(|i| format!(" ({} : Nat) ({} : {} < {})", value, i.hyp, value, i.count))
                .unwrap_or_default();
            write!(binders, "{bound} (hx{k} : Holds {start} {addr} (Bytes.u64 {value}))").unwrap();
            writeln!(
                body,
                "  have hm{k} := Compose.u64_getMem hx{k} (by decide) (by decide) (by omega)"
            )
            .unwrap();
            simp.push(format!("hm{k}"));
        }
        // Tables in read-only data the segment reads at an index: word `var` of each.
        let tables = std::mem::take(&mut self.tables);
        let mut close_with: Vec<String> = (0..words.len()).map(|k| format!("hm{k}")).collect();
        for (k, t) in tables.iter().enumerate() {
            write!(binders, " (hT{k} : Holds {start} {} (Bytes.u64s {}))", t.at, t.list).unwrap();
            writeln!(
                body,
                "  have hv{k} := Compose.table_getMem hT{k} (by decide) (by decide) (by decide) (show {} < {} by omega)\n  have ha{k} : (W {} <<< 3 + W {} : Word) = W ({} + 8 * {}) := by bv_omega",
                t.var, t.len, t.var, t.at, t.at, t.var
            )
            .unwrap();
            simp.push(format!("ha{k}"));
            simp.push(format!("hv{k}"));
            close_with.push(format!("ha{k}"));
            close_with.push(format!("hv{k}"));
            // The table lies in read-only data, apart from every word the segment stores.
            for (j, (a, _)) in stores.iter().enumerate() {
                writeln!(
                    body,
                    "  have hne{k}_{j} : (W ({} + 8 * {}) : Word) ≠ W {a} := by bv_omega",
                    t.at, t.var
                )
                .unwrap();
                simp.push(format!("hne{k}_{j}"));
                close_with.push(format!("hne{k}_{j}"));
            }
        }
        let member = self.member.take();
        let flag = self.flag.take();
        if let Some(f) = &flag {
            write!(
                binders,
                " (E : Bytes) (hE : Holds {start} {} E) (hEl : E.length = 8) (hv : (Leaf.words E).getD 0 0 = {})",
                f.at, f.value
            )
            .unwrap();
            writeln!(
                body,
                "  have hf : {start}.getMem (W {}) = W {} := by\n    have := Compose.int_words (n := 1) hE hEl (by decide) (by decide) 0 (by decide)\n    rw [hv] at this; simpa using this",
                f.at, f.value
            )
            .unwrap();
            simp.push("hf".into());
            close_with.push("hf".into());
        }
        if let Some(m) = &member {
            // The lemma is stated over the integer's bytes `E`; the call passes its expression.
            write!(
                binders,
                " (E : Bytes) (hE : Holds {start} {} E) (hEl : E.length = {}) (hix : (Leaf.words E).getD 0 0 < {} ∧ ∀ j, 0 < j → j < {} → (Leaf.words E).getD j 0 = 0)",
                m.at,
                8 * m.words,
                m.count,
                m.words,
            )
            .unwrap();
            writeln!(
                body,
                "  have hd0 : {start}.getMem (W {}) = W ((Leaf.words E).getD 0 0) := by\n    simpa using Compose.int_words (n := {}) hE hEl (by decide) (by decide) 0 (by decide)\n  have hv64 := Compose.words_getD_lt E 0\n  have hvc := hix.1",
                m.at, m.words
            )
            .unwrap();
            simp.push("hd0".into());
            close_with.push("hd0".into());
            for j in 1..m.words {
                writeln!(
                    body,
                    "  have hd{j} : {start}.getMem (W {}) = W 0 := by\n    rw [show ({} : Nat) = {} + 8 * {j} from rfl, Compose.int_words (n := {}) hE hEl (by decide) (by decide) {j} (by decide), hix.2 {j} (by decide) (by decide)]",
                    m.at + 8 * j,
                    m.at + 8 * j,
                    m.at,
                    m.words
                )
                .unwrap();
                simp.push(format!("hd{j}"));
                close_with.push(format!("hd{j}"));
            }
        }
        if let Some(m) = &member {
            // Inside the lemma the member number is over `E`.
            body = body.replace(&format!("({})", m.expr), "E");
        }
        if let Some((_, v)) = &self.checked {
            write!(binders, " (h10 : {start}.getReg .x10 = W {v})").unwrap();
            simp.push("h10".into());
        }
        if first {
            simp.push("Prog.initRegs".into());
        }
        let mut post: Vec<String> =
            regs.iter().map(|(r, v)| format!("s.getReg .x{r} = W {}", v.lean())).collect();
        if let Some(m) = &member {
            for p in &mut post {
                *p = p.replace(&format!("({})", m.expr), "E");
            }
        }
        if post.is_empty() {
            post.push("True".into());
        }
        let store_list = format!(
            "[{}]",
            stores.iter().map(|(a, v)| format!("({a}, {v})")).collect::<Vec<_>>().join(", ")
        );
        let memory = if stores.is_empty() {
            format!("(∀ w, s.getMem w = {start}.getMem w)")
        } else {
            simp.extend(["Compose.Avoid".into(), "Compose.Stored".into()]);
            format!(
                "(∀ w, Compose.Avoid [{}] w → s.getMem w = {start}.getMem w) ∧\n      Compose.Stored {store_list} s",
                stores.iter().map(|(a, _)| a.to_string()).collect::<Vec<_>>().join(", ")
            )
        };
        writeln!(
            self.lemmas,
            "theorem {lemma}{binders} :\n    P.Spec {from} {start} {steps} {to} fun s => {memory} ∧\n      {} := by\n{body}  mrun [{}]\n{}\n",
            post.join(" ∧ "),
            simp.join(", "),
            close(
                &close_with,
                if member.is_some() {
                    "\n    | (simp [BitVec.ult, List.getD_eq_getElem?_getD] at hbr hvc hv64; omega)"
                } else {
                    ""
                }
            )
        )
        .unwrap();
        let mut args = if first { String::new() } else { format!(" {}", self.state) };
        for (addr, value, fact) in &words {
            let _ = addr;
            if let Some(i) = self.ctx.iter().find(|i| &i.var == value) {
                write!(args, " {} {}", i.var, i.hyp).unwrap();
            }
            write!(args, " {fact}").unwrap();
        }
        for t in &tables {
            write!(args, " {}", t.fact).unwrap();
        }
        if let Some(m) = &member {
            write!(args, " ({}) {} {} {}", m.expr, m.fact, m.length, m.bound).unwrap();
        }
        if let Some(f) = &flag {
            write!(args, " ({}) {} {} {}", f.expr, f.fact, f.length, f.hyp).unwrap();
        }
        if let Some((h, _)) = &self.checked {
            write!(args, " {h}").unwrap();
        }
        let state = self.name("s");
        let (r, stored, frame) = (self.name("r"), self.name("st"), self.name("F"));
        if stores.is_empty() {
            writeln!(self.proof, "  Compose.phase ({lemma}{args}) fun {state} {r} {frame} =>")
                .unwrap();
        } else {
            writeln!(
                self.proof,
                "  Compose.phaseStore {store_list} (by simp) ({lemma}{args}) fun {state} {r} {stored} {frame} =>"
            )
            .unwrap();
        }
        let names: Vec<String> = regs.iter().map(|(reg, _)| format!("x{reg}_{state}")).collect();
        for (i, name) in names.iter().enumerate() {
            // The `i`-th component of a right-nested conjunction.
            let mut path = ".2".repeat(i);
            if i + 1 < names.len() {
                path.push_str(".1");
            }
            writeln!(self.proof, "  have {name} := {r}{path}").unwrap();
        }
        self.state = state;
        self.checked = None;
        self.total += u128::from(steps);
        let written: Vec<(Sym, u64)> = stores.iter().map(|(a, _)| (Sym::num(*a), 8)).collect();
        let closed_values = stores.iter().all(|(_, v)| v.parse::<u64>().is_ok());
        self.transfer_closed(&frame, &written, closed_values);
        Ok((names, stored))
    }

    /// The glue from the current position to instruction `to`.
    fn glue(&mut self, to: u64, regs: &[(u32, Sym)]) -> Result<Vec<String>, ComposeError> {
        let steps = self.segment(self.pos, to)?;
        let names = self.phase(self.pos, to, steps, regs, &[], &[])?.0;
        self.pos = to;
        Ok(names)
    }

    /// The events `events`, after which the ranges `after` are still read.
    fn events(
        &mut self,
        events: &[GlueEvent],
        after: &[(i128, i128)],
    ) -> Result<Option<String>, ComposeError> {
        // What each suffix of the events reads, for the facts worth passing on.
        let mut live = vec![after.to_vec()];
        let mut ctx = self.ctx.clone();
        for event in events.iter().rev() {
            let mut reads = live.last().expect("one").clone();
            event_reads(event, &mut ctx, &mut reads)?;
            live.push(merge(reads));
        }
        live.reverse();
        for (at, event) in events.iter().enumerate() {
            // Its own reads come after the segment before it.
            self.live = live[at].clone();
            match event {
                GlueEvent::Query { ecall, input, request, buffer, size } => {
                    if !self.ctx.is_empty() {
                        return Err(ComposeError::Unsupported("a query inside a loop".into()));
                    }
                    let regs = [
                        (10, Sym::num(*request)),
                        (11, Sym::num(9)),
                        (12, Sym::num(*buffer)),
                        (13, Sym::num(*size)),
                        (17, Sym::num(1)),
                    ];
                    let r = self.glue(*ecall, &regs)?;
                    let req = self.read(&Sym::num(*request), 9, Some(*input))?;
                    self.hypotheses
                        .push(format!("(hin{input} : (ins.getD {input} []).length = {size})"));
                    let (state, held, check, frame) =
                        (self.name("s"), self.name("g"), self.name("q"), self.name("F"));
                    writeln!(
                        self.proof,
                        "  Compose.query hP ins {input} {ecall} {request} {buffer} {size} {} (by rfl) {} {} {} {} {} {} hin{input} (by decide)\n    fun {state} {held} {check} {frame} =>",
                        self.state, r[0], r[1], r[2], r[3], r[4], req.name
                    )
                    .unwrap();
                    self.total += u128::from(*size) + 10;
                    self.state = state;
                    self.transfer(&frame, &[(Sym::num(*buffer), *size)]);
                    self.facts.push(Fact {
                        name: held,
                        addr: Sym::num(*buffer),
                        len: *size,
                        expr: format!("ins.getD {input} []"),
                        length: format!("hin{input}"),
                        image: false,
                    });
                    self.checked = Some((check, *size));
                    self.pos = ecall + 1;
                }
                GlueEvent::Leaf {
                    call,
                    entry,
                    name,
                    descriptor,
                    arguments,
                    inputs,
                    outputs,
                    scratch,
                    steps: budget,
                } => self.leaf(
                    *call,
                    *entry,
                    name,
                    *descriptor,
                    arguments,
                    inputs,
                    outputs,
                    *scratch,
                    *budget,
                )?,
                GlueEvent::Copy { call, dst, src, len } => {
                    let (dst, src) = (self.sym(dst)?, self.sym(src)?);
                    self.copy_event(*call, &dst, &src, *len)?;
                }
                GlueEvent::CopyAt { call, dst, base, stride, index, words, count } => {
                    self.copy_member(*call, *dst, *base, *stride, *index, *words, *count)?;
                }
                GlueEvent::Coins { ecall, at, len } => self.coins(*ecall, *at, *len)?,
                GlueEvent::CopySelect { call, dst, table, candidates, index, words, len } => {
                    self.copy_select(*call, *dst, *table, candidates, *index, *words, *len)?;
                }
                GlueEvent::Loop {
                    slot,
                    index,
                    count,
                    top,
                    end,
                    carried,
                    until: Some(until),
                    body,
                } => {
                    if draws_any(body) {
                        self.loop_until_coins(
                            *slot, *index, *count, *top, *end, carried, *until, body,
                        )?;
                    } else {
                        self.loop_until(*slot, *index, *count, *top, *end, carried, *until, body)?;
                    }
                    return self.events(&events[at + 1..], after);
                }
                GlueEvent::Loop { slot, index, count, top, end, carried, until: None, body } => {
                    self.loop_(*slot, *index, *count, *top, *end, carried, body)?;
                    // The rest runs after the loop, inside its exit.
                    return self.events(&events[at + 1..], after);
                }
                GlueEvent::Halt { ecall, payload: at, len } => {
                    if !self.ctx.is_empty() {
                        return Err(ComposeError::Unsupported("HALT inside a loop".into()));
                    }
                    let r = self.glue(
                        *ecall,
                        &[(10, Sym::num(*at)), (11, Sym::num(*len)), (17, Sym::num(0))],
                    )?;
                    let out = self.read(&Sym::num(*at), *len, None)?;
                    writeln!(
                        self.proof,
                        "  Compose.halt hP {ecall} {at} {len} _ {} (by rfl) {} {} {} {} {} (by decide)",
                        self.state, r[0], r[1], r[2], out.name, out.length
                    )
                    .unwrap();
                    self.total += u128::from(*len) + 1;
                    return Ok(Some(out.expr));
                }
            }
        }
        Ok(None)
    }

    /// A call of the copy routine of `len` bytes from `src` to `dst`.
    fn copy_event(
        &mut self,
        call: u64,
        dst: &Sym,
        src: &Sym,
        len: u64,
    ) -> Result<(), ComposeError> {
        self.copy_from(call, dst, src, len, None)
    }

    /// A call of the copy routine, its source fact and copy conditions given or found.
    fn copy_from(
        &mut self,
        call: u64,
        dst: &Sym,
        src: &Sym,
        len: u64,
        given: Option<(Fact, String)>,
    ) -> Result<(), ComposeError> {
        let aligned = |s: &Sym| s.base % 8 == 0 && s.terms.iter().all(|(_, f)| f % 8 == 0);
        if given.is_none() && !(aligned(dst) && aligned(src) && len.is_multiple_of(8)) {
            return Err(ComposeError::Unaligned { dst: dst.lean(), src: src.lean(), len });
        }
        let r = self.glue(call, &[(10, dst.clone()), (11, src.clone()), (12, Sym::num(len))])?;
        if let Some(sel) = self.select.take() {
            // The candidates' facts in the state at the call.
            let cands: Vec<Fact> = sel
                .candidates
                .iter()
                .map(|c| self.read(&Sym::num(*c), sel.len, None))
                .collect::<Result<_, _>>()?;
            let alternatives = |what: fn(&Fact) -> &String| {
                cands.iter().map(|f| format!("exact {}", what(f))).collect::<Vec<_>>().join("; ")
            };
            writeln!(
                self.proof,
                "  have {} : Holds {} {} ({}) :=\n    Compose.holds_select {} {} (by intro k hk; simp only [List.length_cons, List.length_nil] at hk; interval_cases k <;> [{}]) {}\n  have {} : ({}).length = {} :=\n    Compose.select_length (by intro c hc; simp only [List.mem_cons, List.not_mem_nil, or_false] at hc; rcases hc with {} <;> [{}]) {}",
                sel.name,
                self.state,
                sel.address,
                sel.expr,
                sel.addrs,
                sel.list,
                alternatives(|f| &f.name),
                sel.within,
                sel.length,
                sel.expr,
                sel.len,
                vec!["rfl"; cands.len()].join(" | "),
                alternatives(|f| &f.length),
                sel.within
            )
            .unwrap();
        }
        let (from, num) = match given {
            Some((from, num)) => (from, Some(num)),
            None => (self.read(src, len, None)?, None),
        };
        let n = len / 8;
        let (state, held, frame) = (self.name("s"), self.name("g"), self.name("F"));
        let concrete = dst.concrete().is_some() && src.concrete().is_some();
        writeln!(
                self.proof,
                "  Compose.copy hP copyAt {call} {} {} {} {} {n} ({}) (by rfl) (by rfl) (by decide) (by decide) {} {} {} {} (by rw [{}] <;> decide) {}\n    fun {state} {held} {frame} =>",
                call_offset(&self.instrs, call),
                self.state,
                dst.lean(),
                src.lean(),
                from.expr,
                r[0],
                r[1],
                r[2],
                from.name,
                from.length,
                match num {
                    Some(num) => num,
                    None if concrete => "(by decide)".to_owned(),
                    None => self.omega_closer("Compose.CopyNum, Leaf.Apart, BASE"),
                }
            )
            .unwrap();
        self.total += 9 + 7 * u128::from(n);
        self.state = state;
        self.transfer(&frame, &[(dst.clone(), len)]);
        self.facts.push(Fact { name: held, addr: dst.clone(), len, ..from });
        self.pos = call + 2;
        Ok(())
    }

    /// A copy of member `v` of a family, `v` read from memory at run time: the hypothesis
    /// `hix` of `halts` bounds it below the count (its high words zero), and the segment before
    /// the call proves the source address from it.
    #[allow(clippy::too_many_arguments)]
    fn copy_member(
        &mut self,
        call: u64,
        dst: u64,
        base: u64,
        stride: u64,
        index: u64,
        words: u64,
        count: u64,
    ) -> Result<(), ComposeError> {
        if !(dst.is_multiple_of(8) && base.is_multiple_of(8) && stride.is_multiple_of(8)) {
            return Err(ComposeError::Unaligned {
                dst: dst.to_string(),
                src: base.to_string(),
                len: stride,
            });
        }
        let e = self.read(&Sym::num(index), 8 * words, None)?;
        let k = self.calls;
        self.calls += 1;
        let value = format!("((Leaf.words ({})).getD 0 0)", e.expr);
        self.hypotheses.push(format!(
            "(hix{k} : {}{value} < {count} ∧ ∀ j, 0 < j → j < {words} → (Leaf.words ({})).getD j 0 = 0)",
            self.quantifiers(),
            e.expr
        ));
        let bound = format!("(hix{k}{})", self.instances());
        let hv = self.name("hv");
        writeln!(self.proof, "  have {hv} := {bound}.1").unwrap();
        // The member number is in scope like an index while the copy is proved.
        self.ctx.push(Index {
            slot: DATA,
            var: value.clone(),
            hyp: hv,
            count,
            addr: 0,
            stop: false,
        });
        let src = Sym { base: base.into(), terms: vec![(value.clone(), stride.into())] };
        self.member = Some(Member {
            at: index,
            fact: e.name.clone(),
            expr: e.expr.clone(),
            words,
            count,
            bound,
            length: e.length.clone(),
        });
        let result = self.copy_event(call, &Sym::num(dst), &src, stride);
        self.ctx.pop();
        self.member = None;
        result?;
        // Facts at addresses that name the member number are no longer in scope.
        self.facts.retain(|f| f.addr.terms.iter().all(|(v, _)| *v != value));
        Ok(())
    }

    /// `COINS`: `len` uniform bytes land at `at`, as a new variable: outside loops the run's
    /// result binds it (`coinsPC len >>= fun bs => …`); inside a loop it is the iteration's draw.
    fn coins(&mut self, ecall: u64, at: u64, len: u64) -> Result<(), ComposeError> {
        let in_loop = self.loops().next().is_some();
        if in_loop && self.draw.as_ref().is_none_or(|d| !d.0.is_empty()) {
            return Err(ComposeError::Unsupported("a draw inside a loop not drawing once".into()));
        }
        let r = self.glue(ecall, &[(10, Sym::num(at)), (11, Sym::num(len)), (17, Sym::num(2))])?;
        let (bs, hbs) = if in_loop {
            let depth = self.loops().count() - 1;
            (format!("b{depth}"), format!("hb{depth}"))
        } else {
            let k = self.name("");
            (format!("bs{k}"), format!("hbs{k}"))
        };
        let (state, held, frame) = (self.name("s"), self.name("g"), self.name("F"));
        writeln!(
            self.proof,
            "  Compose.coins hP {ecall} {at} {len} {} (by rfl) {} {} {} (by decide)\n    fun {bs} {state} {hbs} {held} {frame} =>",
            self.state, r[0], r[1], r[2]
        )
        .unwrap();
        self.total += u128::from(len) + 1;
        self.state = state;
        self.transfer(&frame, &[(Sym::num(at), len)]);
        self.facts.push(Fact {
            name: held,
            addr: Sym::num(at),
            len,
            expr: bs.clone(),
            length: hbs.clone(),
            image: false,
        });
        self.scope
            .push((format!("∀ {bs} : Bytes, {bs}.length = {len} → "), format!(" {bs} {hbs}")));
        if in_loop {
            self.draw = Some((bs, hbs, len));
        } else {
            self.wrappers.push(format!("coinsPC {len} >>= fun {bs} =>"));
        }
        self.pos = ecall + 1;
        Ok(())
    }

    /// A copy of candidate `v` of several, `v` read from memory at run time: the hypothesis
    /// `hsel` of `halts` bounds it below the number of candidates (its high words zero); the
    /// segment before the call loads the candidate's address from the table in read-only data,
    /// and the copy's source is candidate `v` (`Compose.holds_select`).
    #[allow(clippy::too_many_arguments)]
    fn copy_select(
        &mut self,
        call: u64,
        dst: u64,
        table: u64,
        candidates: &[u64],
        index: u64,
        words: u64,
        len: u64,
    ) -> Result<(), ComposeError> {
        if !(dst.is_multiple_of(8) &&
            len.is_multiple_of(8) &&
            candidates.iter().all(|c| c.is_multiple_of(8)))
        {
            return Err(ComposeError::Unaligned {
                dst: dst.to_string(),
                src: format!("{candidates:?}"),
                len,
            });
        }
        let n = candidates.len();
        let e = self.read(&Sym::num(index), 8 * words, None)?;
        let cands: Vec<Fact> = candidates
            .iter()
            .map(|c| self.read(&Sym::num(*c), len, None))
            .collect::<Result<_, _>>()?;
        let entries = self.read(&Sym::num(table), 8 * n as u64, None)?;
        let k = self.calls;
        self.calls += 1;
        let value = format!("((Leaf.words ({})).getD 0 0)", e.expr);
        self.hypotheses.push(format!(
            "(hsel{k} : {}{value} < {n} ∧ ∀ j, 0 < j → j < {words} → (Leaf.words ({})).getD j 0 = 0)",
            self.quantifiers(),
            e.expr
        ));
        let bound = format!("(hsel{k}{})", self.instances());
        let hv = self.name("hv");
        writeln!(self.proof, "  have {hv} := {bound}.1").unwrap();
        let addrs =
            format!("[{}]", candidates.iter().map(u64::to_string).collect::<Vec<_>>().join(", "));
        let list =
            format!("[{}]", cands.iter().map(|f| f.expr.clone()).collect::<Vec<_>>().join(", "));
        let within = "(by simp only [List.length_cons, List.length_nil]; omega)".to_owned();
        let address = format!("({addrs}.getD {value} 0)");
        self.ctx.push(Index {
            slot: DATA,
            var: value.clone(),
            hyp: hv.clone(),
            count: n as u64,
            addr: 0,
            stop: false,
        });
        self.member = Some(Member {
            at: index,
            fact: e.name.clone(),
            expr: e.expr.clone(),
            words,
            count: n as u64,
            bound,
            length: e.length.clone(),
        });
        self.tables = vec![Table {
            at: table,
            list: addrs.clone(),
            var: value.clone(),
            len: n,
            fact: entries.name.clone(),
        }];
        // The source: candidate `v`, by the facts of all candidates.
        let (name, length) = (self.name("p"), self.name("lp"));
        let expr = format!("{list}.getD {value} []");
        let source = Fact {
            name: name.clone(),
            addr: Sym { base: 0, terms: vec![(address.clone(), 1)] },
            len,
            expr: format!("({expr})"),
            length: length.clone(),
            image: false,
        };
        let num = format!("(Compose.copy_num_select (by decide) {within})");
        let src = source.addr.clone();
        // The facts about the candidates must be stated in the state at the call, so they are
        // written once the segment before it is proved: `copy_from` writes the segment first.
        self.select = Some(SelectSource {
            name,
            length,
            address: address.clone(),
            expr,
            list,
            addrs,
            candidates: candidates.to_vec(),
            len,
            within,
        });
        let result = self.copy_from(call, &Sym::num(dst), &src, len, Some((source, num)));
        self.ctx.pop();
        self.member = None;
        self.tables.clear();
        result?;
        self.facts.retain(|f| f.addr.terms.iter().all(|(v, _)| !v.contains(&value)));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn leaf(
        &mut self,
        call: u64,
        entry: u64,
        name: &str,
        descriptor: u64,
        arguments: &[Argument],
        inputs: &[(u64, u64)],
        outputs: &[(u64, u64)],
        scratch: (u64, u64),
        budget: u128,
    ) -> Result<(), ComposeError> {
        let lean = self.leaves.get(name).ok_or_else(|| ComposeError::NoSpec(name.to_owned()))?;
        let spec = match self.spec_names.get(&entry) {
            Some(spec) => spec.clone(),
            None => {
                let at = format!("leafAt{entry}");
                writeln!(
                    self.specs,
                    "theorem {at} : {} P {entry} := by constructor <;> rfl\n",
                    lean.code_at
                )
                .unwrap();
                let spec = format!("L{entry}");
                writeln!(
                    self.specs,
                    "/-- The specification of `{name}` at instruction {entry}. -/\ndef {spec} := {} {at} (by norm_num [BASE])\n",
                    lean.spec
                )
                .unwrap();
                if !self.modules.contains(&lean.module) {
                    self.modules.push(lean.module.clone());
                }
                self.spec_names.insert(entry, spec.clone());
                spec
            }
        };
        let mut args = Vec::new();
        let mut tables: Vec<Table> = Vec::new();
        let mut dynamic = false;
        for a in arguments {
            let (term, reads, table) = self.argument(a)?;
            args.push(term);
            dynamic |= reads;
            tables.extend(table);
        }
        let words: Vec<String> = args
            .iter()
            .cloned()
            .chain(inputs.iter().map(|x| x.0.to_string()))
            .chain(outputs.iter().map(|x| x.0.to_string()))
            .chain([scratch.0.to_string()])
            .collect();
        let regs = [(10, Sym::num(descriptor))];
        let tables_read = tables.clone();
        let (r, desc) = if dynamic {
            // The glue stores every word of the descriptor before the call.
            let stores: Vec<(u64, String)> = words
                .iter()
                .enumerate()
                .map(|(k, w)| (descriptor + 8 * k as u64, w.clone()))
                .collect();
            let steps = self.segment(self.pos, call)?;
            self.tables = tables.clone();
            let (r, stored) = self.phase(self.pos, call, steps, &regs, &stores, &[])?;
            self.pos = call;
            let (name, length) = (self.name("d"), self.name("ld"));
            let list = format!("[{}]", words.join(", "));
            let each: Vec<String> = stores
                .iter()
                .map(|(a, v)| format!("exact Compose.stored_getMem {stored} ({a}, {v}) (by simp)"))
                .collect();
            writeln!(
                self.proof,
                "  have {name} : Holds {} {descriptor} (Bytes.u64s {list}) :=\n    Leaf.holds_u64s _ {descriptor} {list} (by decide) (by simp)\n      (by intro k hk; simp only [List.length_cons, List.length_nil] at hk; interval_cases k <;> [{}])\n  have {length} : (Bytes.u64s {list}).length = {} := by simp [Mem.length_u64s]",
                self.state,
                each.join("; "),
                8 * words.len()
            )
            .unwrap();
            let fact = Fact {
                name,
                addr: Sym::num(descriptor),
                len: 8 * words.len() as u64,
                expr: format!("Bytes.u64s {list}"),
                length,
                image: false,
            };
            (r, fact)
        } else {
            let r = self.glue(call, &regs)?;
            (r, self.read(&Sym::num(descriptor), 8 * words.len() as u64, None)?)
        };
        let ins: Vec<Fact> = inputs
            .iter()
            .map(|(a, n)| self.read(&Sym::num(*a), *n, None))
            .collect::<Result<_, _>>()?;
        let k = self.calls;
        self.calls += 1;
        self.local_calls.push(k);
        let args = format!("[{}]", args.join(", "));
        let xs = format!("[{}]", ins.iter().map(|f| f.expr.clone()).collect::<Vec<_>>().join(", "));
        let lens =
            format!("[{}]", outputs.iter().map(|o| o.1.to_string()).collect::<Vec<_>>().join(", "));
        let quantifiers = self.quantifiers();
        let instances = self.instances();
        self.hypotheses.push(format!("(hdom{k} : {quantifiers}{spec}.dom {args} {xs})"));
        self.hypotheses.push(format!(
            "(hlen{k} : {quantifiers}({spec}.f {args} {xs}).map List.length = {lens})"
        ));
        self.budgets.push(format!(
            "(hsteps{k} : {}{spec}.steps {args} ≤ {budget})",
            self.loop_quantifiers()
        ));
        let ins_list = format!(
            "[{}]",
            ins.iter()
                .map(|f| format!("({}, {})", f.addr.lean(), f.expr))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let ins_num = format!(
            "[{}]",
            ins.iter()
                .map(|f| format!("({}, {})", f.addr.lean(), f.len))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let outs = format!(
            "[{}]",
            outputs.iter().map(|(a, n)| format!("({a}, {n})")).collect::<Vec<_>>().join(", ")
        );
        let lengths: Vec<String> = ins.iter().map(|f| f.length.clone()).collect();
        let held = match ins.len() {
            0 => "(by intro x hx; simp at hx)".to_owned(),
            n => format!(
                "(by intro x hx; simp only [List.mem_cons, List.not_mem_nil, or_false] at hx; rcases hx with {} <;> assumption)",
                vec!["rfl"; n].join(" | ")
            ),
        };
        let concrete = !dynamic && ins.iter().all(|f| f.addr.concrete().is_some());
        // Each table entry is a word, for the descriptor's conditions.
        let mut bounds = Vec::new();
        for t in &tables_read {
            let name = self.name("hb");
            writeln!(
                self.proof,
                "  have {name} := Compose.getD_lt (vs := {}) (B := 2 ^ 64) (by decide) (by decide) {}",
                t.list, t.var
            )
            .unwrap();
            bounds.push(name);
        }
        let leaf_num = if concrete {
            "(by decide)".to_owned()
        } else if bounds.is_empty() {
            "(by simp [Compose.LeafNum, Leaf.Apart, BASE] <;> omega)".to_owned()
        } else {
            format!(
                "(by simp [Compose.LeafNum, Leaf.Apart, BASE] at {} ⊢ <;> omega)",
                bounds.join(" ")
            )
        };
        let (state, held_out, frame) = (self.name("s"), self.name("g"), self.name("F"));
        writeln!(
            self.proof,
            "  Compose.leaf hP {spec} {call} {} {descriptor} {args} {ins_list} {outs} ({}, {}) {} (by rfl) (by rfl) (by decide) (by decide) {}\n    {ins_num} {} (by simp only [List.map_cons, List.map_nil{}]) {held} {} (hdom{k}{instances}) (hlen{k}{instances})\n    fun {state} {held_out} {frame} =>",
            call_offset(&self.instrs, call),
            scratch.0,
            scratch.1,
            self.state,
            r[0],
            desc.name,
            lengths.iter().map(|l| format!(", {l}")).collect::<String>(),
            leaf_num,
        )
        .unwrap();
        self.total += 3 + budget;
        self.state = state;
        let mut written: Vec<(Sym, u64)> =
            outputs.iter().map(|(a, n)| (Sym::num(*a), *n)).collect();
        written.push((Sym::num(scratch.0), scratch.1));
        self.transfer(&frame, &written);
        for (i, (addr, len)) in outputs.iter().enumerate() {
            let (name, length) = (self.name("o"), self.name("lo"));
            let full = format!("({spec}.f {args} {xs}).getD {i} []");
            let expr = self.name_expr(&full);
            writeln!(
                self.proof,
                "  have {name} : Holds {} {addr} ({expr}) := {held_out} {i} (by decide)\n  have {length} : ({expr}).length = {len} := by\n    show ({full}).length = {len}\n    rw [Compose.length_getD (hlen{k}{instances})] <;> rfl",
                self.state
            )
            .unwrap();
            self.facts.push(Fact {
                name,
                addr: Sym::num(*addr),
                len: *len,
                expr,
                length,
                image: false,
            });
        }
        self.pos = call + 2;
        Ok(())
    }

    /// The steps from instruction `from` to `to`, taking each inverted branch over a jump as
    /// `decisions` says (in order) and following jumps.
    fn walk(&self, from: u64, to: u64, decisions: &[bool]) -> Result<u64, ComposeError> {
        let fail = |why: String| ComposeError::Segment { from, to, why };
        let (mut at, mut steps, mut next) = (from, 0u64, decisions.iter());
        while at != to {
            if steps > 4096 {
                return Err(fail("no end".into()));
            }
            let instr = self.instrs.get(at as usize).ok_or_else(|| fail("past the code".into()))?;
            at = match instr {
                Instr::Branch { offset: 8, .. } => {
                    let take = *next.next().ok_or_else(|| fail("a branch too many".into()))?;
                    if take { at + 2 } else { at + 1 }
                }
                Instr::Jal { rd, offset } if rd.0 == 0 => {
                    (at as i64 + i64::from(*offset) / 4) as u64
                }
                Instr::Branch { .. } | Instr::Jal { .. } | Instr::Jalr { .. } | Instr::Ecall => {
                    return Err(fail(format!("{instr:?} at {at}")));
                }
                _ => at + 1,
            };
            steps += 1;
        }
        Ok(steps)
    }

    /// A counted loop that draws once per iteration and stops when its flag is 1
    /// (`Compose.loopUntilCoins`), carrying one value from iteration to iteration: the body is
    /// generated once, for a symbolic iteration `i`, carried value `x`, and draw `b`, up to the
    /// flag, and then split on the flag's word, which `halts` takes to be 0 or 1 (the hypothesis
    /// `hbool`): 0 goes back to the head with the next carried value, 1 leaves the loop. After the
    /// last iteration the head faults. The run continues after the loop with the last carried
    /// value `xN`, which the payload binds through `Compose.untilCoins` (a fault when it gives
    /// `none`).
    #[allow(clippy::too_many_arguments)]
    fn loop_until_coins(
        &mut self,
        slot: u32,
        index: u64,
        count: u64,
        top: u64,
        end: u64,
        carried: &[(u64, u64)],
        (flag, _check, _next): (u64, u64, u64),
        body: &[GlueEvent],
    ) -> Result<(), ComposeError> {
        let draws: Vec<u64> = body
            .iter()
            .filter_map(|e| match e {
                GlueEvent::Coins { len, .. } => Some(*len),
                _ => None,
            })
            .collect();
        let nested =
            body.iter().any(|e| matches!(e, GlueEvent::Loop { body, .. } if draws_any(body)));
        if self.loops().next().is_some() ||
            !self.families.is_empty() ||
            draws.len() != 1 ||
            nested ||
            carried.len() != 1
        {
            return Err(ComposeError::Unsupported(
                "a drawing until loop other than one draw and one carried value at the top level"
                    .into(),
            ));
        }
        let n = draws[0];
        let (carried_at, size) = carried[0];
        // Facts the body reads are live through the setup.
        let mut live = self.live.clone();
        let mut body_ctx = self.ctx.clone();
        body_ctx.push(Index {
            slot,
            var: String::new(),
            hyp: String::new(),
            count,
            addr: index,
            stop: false,
        });
        for e in body {
            event_reads(e, &mut body_ctx, &mut live)?;
        }
        live.push((i128::from(flag), i128::from(flag) + 8));
        live.push((i128::from(index), i128::from(index) + 8));
        let live = merge(live);
        self.live = live.clone();
        let steps = self.segment(self.pos, top)?;
        let (_, stored) = self.phase(self.pos, top, steps, &[], &[(index, "0".into())], &[])?;
        let zero = self.name("fx");
        writeln!(
            self.proof,
            "  have {zero} : Holds {} {index} (Bytes.u64 0) := Compose.u64_holds {stored}.1 (by decide) (by decide)",
            self.state
        )
        .unwrap();
        let depth = self.ctx.len();
        let (var, hyp, hit) = (format!("i{depth}"), format!("hi{depth}"), format!("hit{depth}"));
        let mut ctx = self.ctx.clone();
        ctx.push(Index {
            slot,
            var: var.clone(),
            hyp: hyp.clone(),
            count,
            addr: index,
            stop: false,
        });
        let mut ranges = vec![(i128::from(index), i128::from(index) + 8)];
        let mut families: Vec<(Sym, u64)> = Vec::new();
        collect_writes(body, &mut ctx, &mut ranges, Some(&mut families))?;
        if !families.is_empty() {
            return Err(ComposeError::Unsupported("an until loop filling a family".into()));
        }
        let outer: Vec<Fact> = self
            .facts
            .iter()
            .filter(|f| {
                let (lo, hi) = f.addr.range(&self.ctx);
                ranges.iter().all(|r| apart((lo, hi + i128::from(f.len)), *r))
            })
            .cloned()
            .collect();
        let initial_carried = self
            .facts
            .iter()
            .find(|f| f.addr == Sym::num(carried_at) && f.len == size && !f.image)
            .cloned()
            .ok_or_else(|| ComposeError::Untracked { addr: carried_at.to_string(), len: size })?;
        let branch = (top..end)
            .find(|at| matches!(self.instrs[*at as usize], Instr::Branch { .. }))
            .ok_or_else(|| ComposeError::Segment { from: top, to: end, why: "no head".into() })?;
        let fresh = self.fresh;
        self.fresh += 1;
        let (x, lx, b, hb) =
            (format!("x{fresh}"), format!("lx{fresh}"), format!("b{depth}"), format!("hb{depth}"));
        let (kc, kx) = (format!("Kc{fresh}"), format!("Kx{fresh}"));
        let (hbool, hflag) = (format!("hbool{fresh}"), format!("hflag{fresh}"));
        let outer_quantifiers = self.quantifiers();
        let outer_instances = self.instances();
        let outer_state = self.state.clone();
        // The invariant at the head: the index, the facts the loop leaves alone, and the carried
        // value `x`; the exit's: the same without the index.
        let parts = 1 + outer.len() + 2;
        let conj = |n: usize, parts: usize| {
            let mut path = ".2".repeat(n);
            if n + 1 < parts {
                path.push_str(".1");
            }
            path
        };
        let enter = |g: &mut Self, inv: &str, state: &str, indexed: bool| -> String {
            let mut header = String::new();
            let (skip, total) = if indexed { (1, parts) } else { (0, parts - 1) };
            g.facts = Vec::new();
            if indexed {
                let fx = g.name("fx");
                writeln!(header, "  have {fx} := {inv}{}", conj(0, total)).unwrap();
                g.facts.push(Fact {
                    name: fx,
                    addr: Sym::num(index),
                    len: 8,
                    expr: format!("Bytes.u64 {var}"),
                    length: format!("(Mem.length_u64 {var})"),
                    image: false,
                });
            }
            for (k, f) in outer.iter().enumerate() {
                let name = g.name("f");
                writeln!(header, "  have {name} := {inv}{}", conj(skip + k, total)).unwrap();
                g.facts.push(Fact { name, ..f.clone() });
            }
            let name = g.name("c");
            writeln!(header, "  have {lx} := {inv}{}", conj(skip + outer.len(), total)).unwrap();
            writeln!(header, "  have {name} := {inv}{}", conj(skip + outer.len() + 1, total))
                .unwrap();
            g.facts.push(Fact {
                name,
                addr: Sym::num(carried_at),
                len: size,
                expr: x.clone(),
                length: lx.clone(),
                image: false,
            });
            g.state = state.to_owned();
            g.checked = None;
            g.live = live.clone();
            header
        };
        let hypotheses_from = self.hypotheses.len();
        let budgets_from = self.budgets.len();

        // The body, for iteration `var`, carried value `x`, and draw `b`.
        let saved = (
            std::mem::take(&mut self.proof),
            std::mem::take(&mut self.facts),
            std::mem::take(&mut self.total),
            std::mem::take(&mut self.local_calls),
        );
        let (state, inv, r) = (self.name("s"), self.name("hI"), self.name("r"));
        let (kcont, kstop) = (self.name("k"), self.name("k"));
        // The index stays out of scope: the step and the flag must not depend on it.
        let scope_from = self.scope.len();
        self.scope.push((format!("∀ {x} : Bytes, {x}.length = {size} → "), format!(" {x} {lx}")));
        let header = enter(self, &inv, &state, true);
        self.ctx.push(Index {
            slot,
            var: var.clone(),
            hyp: hyp.clone(),
            count,
            addr: index,
            stop: false,
        });
        self.pos = top;
        self.draw = Some((String::new(), String::new(), 0));
        self.glue(branch + 2, &[])?;
        if self.events(body, &live)?.is_some() {
            return Err(ComposeError::Unsupported("HALT inside a loop".into()));
        }
        let Some((drawn, drawn_length, _)) = self.draw.clone().filter(|d| !d.0.is_empty()) else {
            return Err(ComposeError::Unsupported("a draw not reached".into()));
        };
        debug_assert_eq!((drawn.as_str(), drawn_length.as_str()), (b.as_str(), hb.as_str()));
        let go = self.read(&Sym::num(flag), 8, None)?;
        let flag_expr = go.expr.clone();
        let step_fact = self
            .facts
            .iter()
            .find(|f| f.addr == Sym::num(carried_at) && f.len == size)
            .cloned()
            .ok_or_else(|| ComposeError::Untracked { addr: carried_at.to_string(), len: size })?;
        let step_expr = step_fact.expr.clone();
        let mentions = |text: &str, name: &str| {
            text.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.'))
                .any(|token| token == name)
        };
        if mentions(&step_expr, &var) || mentions(&flag_expr, &var) {
            return Err(ComposeError::Unsupported("a drawing until loop reading its index".into()));
        }
        let prefix = std::mem::take(&mut self.proof);
        let (prefix_facts, prefix_state, prefix_pos, prefix_total) =
            (self.facts.clone(), self.state.clone(), self.pos, self.total);
        let prefix_checked = self.checked.clone();
        let stop_fn = format!("(fun {x} {b} => decide ((Leaf.words ({flag_expr})).getD 0 0 = 1))");
        // Flag 0: back to the head with the next index and carried value.
        self.flag = Some(Flag {
            at: flag,
            fact: go.name.clone(),
            expr: go.expr.clone(),
            length: go.length.clone(),
            value: 0,
            hyp: hflag.clone(),
        });
        let steps = self.walk(self.pos, top, &[false])?;
        let next = format!("{var} + 1");
        let (_, stored) =
            self.phase(self.pos, top, steps, &[], &[(index, format!("({next})"))], &[])?;
        let mut next_parts =
            vec![format!("Compose.u64_holds (v := {next}) {stored}.1 (by decide) (by decide)")];
        let kept = |g: &Self, o: &Fact| -> Result<String, ComposeError> {
            g.facts
                .iter()
                .find(|f| f.addr == o.addr && f.expr == o.expr)
                .map(|f| f.name.clone())
                .ok_or_else(|| ComposeError::Untracked { addr: o.addr.lean(), len: o.len })
        };
        for o in &outer {
            next_parts.push(kept(self, o)?);
        }
        let carried_now = |g: &Self| -> Result<Fact, ComposeError> {
            g.facts
                .iter()
                .find(|f| f.addr == Sym::num(carried_at) && f.len == size && f.expr == step_expr)
                .cloned()
                .ok_or_else(|| ComposeError::Untracked { addr: carried_at.to_string(), len: size })
        };
        let c_fact = carried_now(self)?;
        next_parts.push(c_fact.length.clone());
        next_parts.push(c_fact.name.clone());
        writeln!(
            self.proof,
            "  {kcont} {b} {} {hb} (by simp only [{hflag}] <;> decide) ⟨{}⟩",
            self.state,
            next_parts.join(", ")
        )
        .unwrap();
        let continue_text = std::mem::take(&mut self.proof);
        let continue_steps = self.total - prefix_total;
        // Flag 1: out of the loop.
        self.facts = prefix_facts;
        self.state = prefix_state;
        self.pos = prefix_pos;
        self.total = prefix_total;
        self.checked = prefix_checked;
        self.flag = Some(Flag {
            at: flag,
            fact: go.name.clone(),
            expr: go.expr.clone(),
            length: go.length.clone(),
            value: 1,
            hyp: hflag.clone(),
        });
        let steps = self.walk(self.pos, end, &[true, true])?;
        self.phase(self.pos, end, steps, &[], &[], &[])?;
        let mut exit_parts = Vec::new();
        for o in &outer {
            exit_parts.push(kept(self, o)?);
        }
        let c_fact = carried_now(self)?;
        exit_parts.push(c_fact.length.clone());
        exit_parts.push(c_fact.name.clone());
        writeln!(
            self.proof,
            "  {kstop} {b} {} {hb} (by simp only [{hflag}] <;> decide) ⟨{}⟩",
            self.state,
            exit_parts.join(", ")
        )
        .unwrap();
        let stop_text = std::mem::take(&mut self.proof);
        let stop_steps = self.total - prefix_total;
        let rest = continue_steps.max(stop_steps);
        let indent =
            |text: &str| -> String { text.lines().map(|l| format!("    {l}\n")).collect() };
        let mut body_text = prefix;
        writeln!(
            body_text,
            "  Or.elim ({hbool}{outer_instances} {x} {b} {lx} {hb})\n    (fun {hflag} => Prog.Runs.mono (n' := {rest} + {r}) (\n{}    ) (by omega))\n    (fun {hflag} => Prog.Runs.mono (n' := {rest} + {r}) (\n{}    ) (by omega))",
            indent(&continue_text),
            indent(&stop_text)
        )
        .unwrap();
        let c = prefix_total + rest;
        let budgets: String = self
            .local_calls
            .iter()
            .map(|k| format!("  have hs{k} := hsteps{k}{}\n", self.loop_instances()))
            .collect();
        self.ctx.pop();
        self.scope.truncate(scope_from);
        self.draw = None;
        self.flag = None;
        let (proof, facts, total, local_calls) = saved;
        self.proof = proof;
        self.facts = facts;
        self.local_calls = local_calls;

        // The head after the last iteration: the branch to the fault handler, and its call.
        let saved_proof = std::mem::take(&mut self.proof);
        let outer_facts = std::mem::take(&mut self.facts);
        let (head_state, head_inv) = (self.name("s"), self.name("hI"));
        self.scope.push((format!("∀ {x} : Bytes, {x}.length = {size} → "), format!(" {x} {lx}")));
        let head_header = enter(self, &head_inv, &head_state, true);
        // The index holds the count here.
        self.facts[0].expr = format!("Bytes.u64 {count}");
        self.facts[0].length = format!("(Mem.length_u64 {count})");
        self.pos = top;
        let Instr::Branch { offset, .. } = self.instrs[branch as usize] else {
            unreachable!("the head's branch")
        };
        // A far branch is an inverted branch over a jump.
        let (handler, through) = match (offset, self.instrs.get(branch as usize + 1)) {
            (8, Some(Instr::Jal { offset: jump, .. })) => {
                ((branch as i64 + 1 + i64::from(*jump) / 4) as u64, 2)
            }
            _ => ((branch as i64 + i64::from(offset) / 4) as u64, 1),
        };
        let fault_call = (handler..)
            .find(|at| matches!(self.instrs.get(*at as usize), Some(Instr::Ecall) | None))
            .filter(|at| (*at as usize) < self.instrs.len())
            .ok_or_else(|| ComposeError::Segment {
                from: top,
                to: handler,
                why: "no fault call".into(),
            })?;
        let steps = (branch - top) + through + (fault_call - handler);
        let fault_regs = self
            .phase(
                top,
                fault_call,
                steps,
                &[(17, Sym::num(255))],
                &[],
                &[(index, count.to_string())],
            )?
            .0;
        writeln!(self.proof, "  Compose.fault (j := {fault_call}) (by rfl) {}", fault_regs[0])
            .unwrap();
        let h = steps + 1;
        let head_text = std::mem::take(&mut self.proof);
        self.scope.pop();
        self.proof = saved_proof;
        self.facts = outer_facts;
        self.total = total + u128::from(count) * c + u128::from(h);

        // The invariants, and their parts at entry.
        let mut invariant = vec![format!("Holds s {index} (Bytes.u64 {var})")];
        let mut exit = Vec::new();
        let mut initial = vec![zero];
        for f in &outer {
            invariant.push(format!("Holds s {} ({})", f.addr.lean(), f.expr));
            exit.push(format!("Holds s {} ({})", f.addr.lean(), f.expr));
            let kept = self.facts.iter().find(|g| g.addr == f.addr && g.expr == f.expr);
            initial.push(kept.expect("kept").name.clone());
        }
        for parts in [&mut invariant, &mut exit] {
            parts.push(format!("{x}.length = {size}"));
            parts.push(format!("Holds s {carried_at} {x}"));
        }
        initial.push(initial_carried.length.clone());
        initial.push(initial_carried.name.clone());
        let header: String = header.lines().map(|l| format!("  {l}\n")).collect();
        let head_header: String = head_header.lines().map(|l| format!("  {l}\n")).collect();
        let budgets: String = budgets.lines().map(|l| format!("  {l}\n")).collect();
        let (exit_state, exit_inv) = (self.name("s"), self.name("hI"));
        let x_last = format!("xN{fresh}");
        let lx_last = format!("lxN{fresh}");
        let step_fn = format!("(fun {x} {b} => {step_expr})");
        writeln!(
            self.proof,
            "  Compose.loopUntilCoins {top} {end} {count} {c} {h} {n} {step_fn} {stop_fn}\n    (fun {var} {x} s => {})\n    (fun {x} s => {})\n    (fun {var} {hit} {x} {state} {inv} {r} {kc} {kx} {kcont} {kstop} =>\n    have {hyp} : {var} < {count} := by omega\n{header}{budgets}    Prog.Runs.mono (\n{}    ) (by omega))\n    (fun {x} {head_state} {head_inv} =>\n{head_header}    Prog.Runs.mono (\n{}    ) (by omega))\n    ({}) {outer_state} ⟨{}⟩ fun {x_last} {exit_state} {exit_inv} =>",
            invariant.join(" ∧ "),
            exit.join(" ∧ "),
            indent(&body_text),
            indent(&head_text),
            initial_carried.expr,
            initial.join(", ")
        )
        .unwrap();
        self.hypotheses.push(format!(
            "({hbool} : {outer_quantifiers}∀ {x} {b} : Bytes, {x}.length = {size} → {b}.length = {n} → (Leaf.words ({flag_expr})).getD 0 0 = 0 ∨ (Leaf.words ({flag_expr})).getD 0 0 = 1)"
        ));
        let _ = (hypotheses_from, budgets_from);
        self.wrappers.push(format!(
            "Compose.untilCoins {n} {step_fn} {stop_fn} {count} ({}) >>= fun o{fresh} => o{fresh}.elim (pure (.fault, ())) fun {x_last} =>",
            initial_carried.expr
        ));
        self.scope.push((
            format!("∀ {x_last} : Bytes, {x_last}.length = {size} → "),
            format!(" {x_last} {lx_last}"),
        ));

        // The exit: the facts the loop left alone and the last carried value.
        self.state = exit_state;
        self.facts = Vec::new();
        for (k, f) in outer.iter().enumerate() {
            let name = self.name("f");
            writeln!(self.proof, "  have {name} := {exit_inv}{}", conj(k, parts - 1)).unwrap();
            self.facts.push(Fact { name, ..f.clone() });
        }
        let name = self.name("c");
        writeln!(
            self.proof,
            "  have {lx_last} := {exit_inv}{}\n  have {name} := {exit_inv}{}",
            conj(outer.len(), parts - 1),
            conj(outer.len() + 1, parts - 1)
        )
        .unwrap();
        self.facts.push(Fact {
            name,
            addr: Sym::num(carried_at),
            len: size,
            expr: x_last.clone(),
            length: lx_last,
            image: false,
        });
        self.checked = None;
        self.pos = end;
        Ok(())
    }

    /// A counted loop that stops when its flag is 1 (`Compose.loopUntil`): `halts` takes the
    /// stopping iteration `t` with the hypotheses that the flag is 0 before it and 1 at it; the
    /// body is generated twice, for an iteration `i < t` that continues and for iteration `t`,
    /// after which the rest of the program runs.
    #[allow(clippy::too_many_arguments)]
    fn loop_until(
        &mut self,
        slot: u32,
        index: u64,
        count: u64,
        top: u64,
        end: u64,
        carried: &[(u64, u64)],
        (flag, _check, _next): (u64, u64, u64),
        body: &[GlueEvent],
    ) -> Result<(), ComposeError> {
        if self.loops().next().is_some() || draws_any(body) || !self.families.is_empty() {
            return Err(ComposeError::Unsupported("an until loop inside a loop or drawing".into()));
        }
        // Facts the body reads are live through the setup.
        let mut live = self.live.clone();
        let mut body_ctx = self.ctx.clone();
        body_ctx.push(Index {
            slot,
            var: String::new(),
            hyp: String::new(),
            count,
            addr: index,
            stop: false,
        });
        for e in body {
            event_reads(e, &mut body_ctx, &mut live)?;
        }
        live.push((i128::from(flag), i128::from(flag) + 8));
        live.push((i128::from(index), i128::from(index) + 8));
        let live = merge(live);
        self.live = live.clone();
        let steps = self.segment(self.pos, top)?;
        let (_, stored) = self.phase(self.pos, top, steps, &[], &[(index, "0".into())], &[])?;
        let zero = self.name("fx");
        writeln!(
            self.proof,
            "  have {zero} : Holds {} {index} (Bytes.u64 0) := Compose.u64_holds {stored}.1 (by decide) (by decide)",
            self.state
        )
        .unwrap();
        let depth = self.ctx.len();
        let (var, hyp, hit) = (format!("i{depth}"), format!("hi{depth}"), format!("hit{depth}"));
        let mut ctx = self.ctx.clone();
        ctx.push(Index {
            slot,
            var: var.clone(),
            hyp: hyp.clone(),
            count,
            addr: index,
            stop: false,
        });
        let mut ranges = vec![(i128::from(index), i128::from(index) + 8)];
        let mut families: Vec<(Sym, u64)> = Vec::new();
        collect_writes(body, &mut ctx, &mut ranges, Some(&mut families))?;
        if !families.is_empty() {
            return Err(ComposeError::Unsupported("an until loop filling a family".into()));
        }
        let outer: Vec<Fact> = self
            .facts
            .iter()
            .filter(|f| {
                let (lo, hi) = f.addr.range(&self.ctx);
                ranges.iter().all(|r| apart((lo, hi + i128::from(f.len)), *r))
            })
            .cloned()
            .collect();
        let initial_carried: Vec<Fact> = carried
            .iter()
            .map(|(a, n)| {
                self.facts
                    .iter()
                    .find(|f| f.addr == Sym::num(*a) && f.len == *n && !f.image)
                    .cloned()
                    .ok_or_else(|| ComposeError::Untracked { addr: a.to_string(), len: *n })
            })
            .collect::<Result<_, _>>()?;
        let first = self.family_ids;
        self.family_ids += carried.len();
        let carried_id = |k: usize| format!("__C{}__", first + k);
        let (t, ht) = (format!("t{}", self.fresh), format!("ht{}", self.fresh));
        let (hgo, hstop) = (format!("hgo{}", self.fresh), format!("hstop{}", self.fresh));
        self.hypotheses.push(format!("({t} : Nat) ({ht} : {t} < {count})"));
        self.globals.push((t.clone(), "Nat".into()));
        let parts = 1 + outer.len() + 2 * carried.len();
        let conj = |n: usize| {
            let mut path = ".2".repeat(n);
            if n + 1 < parts {
                path.push_str(".1");
            }
            path
        };
        let branch = (top..end)
            .find(|at| matches!(self.instrs[*at as usize], Instr::Branch { .. }))
            .ok_or_else(|| ComposeError::Segment { from: top, to: end, why: "no head".into() })?;
        let hypotheses_from = self.hypotheses.len();
        let budgets_from = self.budgets.len();
        let outer_state = self.state.clone();

        // Enters the body for index `v` (bound by `h`) from the invariant `inv` in `state`.
        let enter = |g: &mut Self, v: &str, h: &str, inv: &str, state: &str| -> String {
            let mut header = String::new();
            let fx = g.name("fx");
            writeln!(header, "  have {fx} := {inv}{}", conj(0)).unwrap();
            g.facts = vec![Fact {
                name: fx,
                addr: Sym::num(index),
                len: 8,
                expr: format!("Bytes.u64 {v}"),
                length: format!("(Mem.length_u64 {v})"),
                image: false,
            }];
            for (n, f) in outer.iter().enumerate() {
                let name = g.name("f");
                writeln!(header, "  have {name} := {inv}{}", conj(1 + n)).unwrap();
                g.facts.push(Fact { name, ..f.clone() });
            }
            for (n, (a, size)) in carried.iter().enumerate() {
                let (length, name) = (g.name("lc"), g.name("c"));
                writeln!(header, "  have {length} := {inv}{}", conj(1 + outer.len() + 2 * n))
                    .unwrap();
                writeln!(header, "  have {name} := {inv}{}", conj(2 + outer.len() + 2 * n))
                    .unwrap();
                g.facts.push(Fact {
                    name,
                    addr: Sym::num(*a),
                    len: *size,
                    expr: format!("({} {v})", carried_id(n)),
                    length,
                    image: false,
                });
            }
            g.ctx.push(Index {
                slot,
                var: v.to_owned(),
                hyp: h.to_owned(),
                count,
                addr: index,
                stop: v.starts_with('t'),
            });
            g.state = state.to_owned();
            g.pos = top;
            g.checked = None;
            g.live = live.clone();
            header
        };

        // Pass 1: an iteration `var < t`, which continues.
        let saved = (
            std::mem::take(&mut self.proof),
            std::mem::take(&mut self.facts),
            std::mem::take(&mut self.total),
            std::mem::take(&mut self.local_calls),
        );
        let (state, inv, r, k) = (self.name("s"), self.name("hI"), self.name("r"), self.name("k"));
        let scope_from = self.scope.len();
        self.scope.push((format!("∀ {var} < {count}, "), format!(" {var} {hyp}")));
        let header = enter(self, &var, &hyp, &inv, &state);
        self.glue(branch + 2, &[])?;
        if self.events(body, &live)?.is_some() {
            return Err(ComposeError::Unsupported("HALT inside a loop".into()));
        }
        let go = self.read(&Sym::num(flag), 8, None)?;
        let flag_go = go.expr.clone();
        self.flag = Some(Flag {
            at: flag,
            fact: go.name.clone(),
            expr: go.expr.clone(),
            length: go.length.clone(),
            value: 0,
            hyp: format!("({hgo} {var} {hit})"),
        });
        let steps = self.walk(self.pos, top, &[false])?;
        let next = format!("{var} + 1");
        let (_, stored) =
            self.phase(self.pos, top, steps, &[], &[(index, format!("({next})"))], &[])?;
        let mut next_parts =
            vec![format!("Compose.u64_holds (v := {next}) {stored}.1 (by decide) (by decide)")];
        for o in &outer {
            let f = self
                .facts
                .iter()
                .find(|f| f.addr == o.addr && f.expr == o.expr)
                .ok_or_else(|| ComposeError::Untracked { addr: o.addr.lean(), len: o.len })?;
            next_parts.push(f.name.clone());
        }
        let mut steps_of = Vec::new();
        for (n, (a, size)) in carried.iter().enumerate() {
            let f = self
                .facts
                .iter()
                .find(|f| f.addr == Sym::num(*a) && f.len == *size)
                .cloned()
                .ok_or_else(|| ComposeError::Untracked { addr: a.to_string(), len: *size })?;
            steps_of.push(f.expr.replace(&format!("({} {var})", carried_id(n)), "c"));
            next_parts.push(f.length.clone());
            next_parts.push(f.name.clone());
        }
        writeln!(self.proof, "  {k} {} ⟨{}⟩", self.state, next_parts.join(", ")).unwrap();
        let c = self.total;
        let budgets: String = self
            .local_calls
            .iter()
            .map(|k| format!("  have hs{k} := hsteps{k}{}\n", self.loop_instances()))
            .collect();
        self.ctx.pop();
        self.scope.truncate(scope_from);
        let functions: Vec<String> = initial_carried
            .iter()
            .zip(&steps_of)
            .map(|(init, step)| {
                format!(
                    "(fun n => Nat.rec (motive := fun _ => Bytes) ({}) (fun {var} c => {step}) n)",
                    init.expr
                )
            })
            .collect();
        let substitute = |text: &str| {
            let mut text = text.to_owned();
            for (n, f) in functions.iter().enumerate() {
                text = text.replace(&carried_id(n), f);
            }
            text
        };
        let body_text = substitute(&std::mem::take(&mut self.proof));
        let (proof, facts, total, local_calls) = saved;
        self.proof = proof;
        self.facts = facts;
        self.local_calls = local_calls;
        self.total = total + u128::from(count) * c;

        // The invariant at the head, and its parts at entry.
        let mut invariant = vec![format!("Holds s {index} (Bytes.u64 {var})")];
        let mut initial = vec![zero];
        for f in &outer {
            invariant.push(format!("Holds s {} ({})", f.addr.lean(), f.expr));
            let kept = self.facts.iter().find(|g| g.addr == f.addr && g.expr == f.expr);
            initial.push(kept.expect("kept").name.clone());
        }
        for (((a, size), func), init) in carried.iter().zip(&functions).zip(&initial_carried) {
            invariant.push(format!("({func} {var}).length = {size}"));
            invariant.push(format!("Holds s {a} ({func} {var})"));
            initial.push(init.length.clone());
            initial.push(init.name.clone());
        }
        let indented: String = body_text.lines().map(|l| format!("    {l}\n")).collect();
        let header: String = header.lines().map(|l| format!("  {l}\n")).collect();
        let budgets: String = budgets.lines().map(|l| format!("  {l}\n")).collect();
        let (last_state, last_inv) = (self.name("s"), self.name("hI"));
        writeln!(
            self.proof,
            "  Compose.loopUntil {top} {c} {t} (fun {var} s => {})\n    (fun {var} {hit} {state} {inv} {r} {k} =>\n    have {hyp} : {var} < {count} := by omega\n{header}{budgets}    Prog.Runs.mono (\n{indented}    ) (by omega))\n    {outer_state} ⟨{}⟩ fun {last_state} {last_inv} =>",
            invariant.join(" ∧ "),
            initial.join(", ")
        )
        .unwrap();

        // Pass 2: iteration `t`, which stops; the rest of the program follows in this term.
        let header = enter(self, &t, &ht, &last_inv, &last_state);
        self.proof.push_str(&header);
        self.glue(branch + 2, &[])?;
        if self.events(body, &live)?.is_some() {
            return Err(ComposeError::Unsupported("HALT inside a loop".into()));
        }
        let stop = self.read(&Sym::num(flag), 8, None)?;
        let flag_stop = stop.expr.clone();
        self.flag = Some(Flag {
            at: flag,
            fact: stop.name.clone(),
            expr: stop.expr.clone(),
            length: stop.length.clone(),
            value: 1,
            hyp: hstop.clone(),
        });
        let steps = self.walk(self.pos, end, &[true, true])?;
        self.phase(self.pos, end, steps, &[], &[], &[])?;
        self.ctx.pop();
        self.pos = end;
        // What follows names the carried values by their functions.
        self.proof = substitute(&self.proof);
        for f in &mut self.facts {
            f.expr = substitute(&f.expr);
        }
        for h in &mut self.hypotheses[hypotheses_from..] {
            *h = substitute(h);
        }
        for h in &mut self.budgets[budgets_from..] {
            *h = substitute(h);
        }
        self.hypotheses.push(format!(
            "({hgo} : ∀ {var} < {t}, (Leaf.words ({})).getD 0 0 = 0)",
            substitute(&flag_go)
        ));
        self.hypotheses
            .push(format!("({hstop} : (Leaf.words ({})).getD 0 0 = 1)", substitute(&flag_stop)));
        Ok(())
    }

    /// A counted loop: the setup that zeroes its index, `Compose.loop` with the body generated for
    /// a symbolic index, and the exit, in which the rest of the program continues.
    ///
    /// The invariant at the head, for index `i`, is a conjunction of parts: the index slot holds
    /// `i`; each fact the loop leaves alone still holds; each carried value `C i` has its bytes
    /// and holds its slot, where `C` is `Nat.rec` over the body's step; the members `[0, i)` of
    /// each family the loop fills hold (`Compose.Members`); and the members of enclosing loops'
    /// families still hold.
    #[allow(clippy::too_many_arguments)]
    fn loop_(
        &mut self,
        slot: u32,
        index: u64,
        count: u64,
        top: u64,
        end: u64,
        carried: &[(u64, u64)],
        body: &[GlueEvent],
    ) -> Result<(), ComposeError> {
        // Facts the body reads are live through the setup.
        let mut live = self.live.clone();
        let mut body_ctx = self.ctx.clone();
        body_ctx.push(Index {
            slot,
            var: String::new(),
            hyp: String::new(),
            count,
            addr: index,
            stop: false,
        });
        for e in body {
            event_reads(e, &mut body_ctx, &mut live)?;
        }
        self.live = merge(live);
        // The setup: the index slot holds 0.
        let steps = self.segment(self.pos, top)?;
        let (_, stored) = self.phase(self.pos, top, steps, &[], &[(index, "0".into())], &[])?;
        let zero = self.name("fx");
        writeln!(
            self.proof,
            "  have {zero} : Holds {} {index} (Bytes.u64 0) := Compose.u64_holds {stored}.1 (by decide) (by decide)",
            self.state
        )
        .unwrap();
        // What the loop writes, over all its iterations.
        let depth = self.ctx.len();
        let var = format!("i{depth}");
        let hyp = format!("hi{depth}");
        let mut ctx = self.ctx.clone();
        ctx.push(Index {
            slot,
            var: var.clone(),
            hyp: hyp.clone(),
            count,
            addr: index,
            stop: false,
        });
        let mut ranges = vec![(i128::from(index), i128::from(index) + 8)];
        let mut families: Vec<(Sym, u64)> = Vec::new();
        collect_writes(body, &mut ctx, &mut ranges, Some(&mut families))?;
        // The facts the loop leaves alone, and the initial carried values.
        let outer: Vec<Fact> = self
            .facts
            .iter()
            .filter(|f| {
                let (lo, hi) = f.addr.range(&self.ctx);
                ranges.iter().all(|r| apart((lo, hi + i128::from(f.len)), *r))
            })
            .cloned()
            .collect();
        let initial_carried: Vec<Fact> = carried
            .iter()
            .map(|(a, n)| {
                self.facts
                    .iter()
                    .find(|f| f.addr == Sym::num(*a) && f.len == *n && !f.image)
                    .cloned()
                    .ok_or_else(|| ComposeError::Untracked { addr: a.to_string(), len: *n })
            })
            .collect::<Result<_, _>>()?;
        let outer_families = std::mem::take(&mut self.families);
        for family in &outer_families {
            let (lo, hi) = family.base.range(&self.ctx);
            let whole = (lo, hi + i128::from(family.n) * i128::from(family.count));
            if !ranges.iter().all(|r| apart(whole, *r)) {
                return Err(ComposeError::Unsupported("a loop writing its parent's family".into()));
            }
        }
        // A body may draw once per iteration (`Compose.loopCoins`), at its top level.
        let draws: Vec<u64> = body
            .iter()
            .filter_map(|e| match e {
                GlueEvent::Coins { len, .. } => Some(*len),
                _ => None,
            })
            .collect();
        if body.iter().any(|e| matches!(e, GlueEvent::Loop { body, .. } if draws_any(body))) ||
            draws.len() > 1
        {
            return Err(ComposeError::Unsupported("draws other than one per iteration".into()));
        }
        let drawing = draws.first().copied();
        if drawing.is_some() &&
            (!carried.is_empty() || !outer_families.is_empty() || self.loops().next().is_some())
        {
            return Err(ComposeError::Unsupported(
                "a drawing loop with carried values or nested".into(),
            ));
        }
        let first = self.family_ids;
        self.family_ids += families.len() + carried.len();
        let carried_id = |k: usize| format!("__C{}__", first + families.len() + k);

        // The parts' count and positions.
        let parts = 1 + outer.len() + 2 * carried.len() + families.len() + outer_families.len();
        let conj = |n: usize| {
            let mut path = ".2".repeat(n);
            if n + 1 < parts {
                path.push_str(".1");
            }
            path
        };
        let at_outer = 1;
        let at_carried = at_outer + outer.len();
        let at_families = at_carried + 2 * carried.len();
        let at_parents = at_families + families.len();

        // The body, for index `var < count`, from the head back to the head.
        let saved = (
            std::mem::take(&mut self.proof),
            std::mem::take(&mut self.facts),
            self.state.clone(),
            std::mem::take(&mut self.total),
            std::mem::take(&mut self.local_calls),
        );
        self.ctx.push(Index {
            slot,
            var: var.clone(),
            hyp: hyp.clone(),
            count,
            addr: index,
            stop: false,
        });
        let scope_from = self.scope.len();
        self.scope.push((format!("∀ {var} < {count}, "), format!(" {var} {hyp}")));
        if drawing.is_some() {
            self.draw = Some((String::new(), String::new(), 0));
        }
        let hypotheses_from = (self.hypotheses.len(), self.budgets.len());
        let (state, inv, r, k) = (self.name("s"), self.name("hI"), self.name("r"), self.name("k"));
        let mut header = String::new();
        let fx = self.name("fx");
        writeln!(header, "  have {fx} := {inv}{}", conj(0)).unwrap();
        self.facts.push(Fact {
            name: fx,
            addr: Sym::num(index),
            len: 8,
            expr: format!("Bytes.u64 {var}"),
            length: format!("(Mem.length_u64 {var})"),
            image: false,
        });
        for (n, f) in outer.iter().enumerate() {
            let name = self.name("f");
            writeln!(header, "  have {name} := {inv}{}", conj(at_outer + n)).unwrap();
            self.facts.push(Fact { name, ..f.clone() });
        }
        for (n, (a, size)) in carried.iter().enumerate() {
            let (length, name) = (self.name("lc"), self.name("c"));
            writeln!(header, "  have {length} := {inv}{}", conj(at_carried + 2 * n)).unwrap();
            writeln!(header, "  have {name} := {inv}{}", conj(at_carried + 2 * n + 1)).unwrap();
            self.facts.push(Fact {
                name,
                addr: Sym::num(*a),
                len: *size,
                expr: format!("({} {var})", carried_id(n)),
                length,
                image: false,
            });
        }
        for (n, (base, size)) in families.iter().enumerate() {
            let name = self.name("m");
            writeln!(header, "  have {name} := {inv}{}", conj(at_families + n)).unwrap();
            self.families.push(Family {
                name,
                base: base.clone(),
                n: *size,
                var: var.clone(),
                count,
                members: format!("__O{}__", first + n),
            });
        }
        for (n, family) in outer_families.iter().enumerate() {
            let name = self.name("m");
            writeln!(header, "  have {name} := {inv}{}", conj(at_parents + n)).unwrap();
            self.families.push(Family { name, ..family.clone() });
        }
        self.state = state.clone();
        self.pos = top;
        self.checked = None;
        // After an iteration: the next one, the index slots, and what follows the loop.
        let mut next = self.live.clone();
        let mut body_ctx = self.ctx.clone();
        for e in body {
            event_reads(e, &mut body_ctx, &mut next)?;
        }
        for i in self.loops() {
            next.push((i128::from(i.addr), i128::from(i.addr) + 8));
        }
        let next = merge(next);
        self.live = next.clone();
        // The head: the index is below the count, so the branch skips the exit.
        let branch = (top..end)
            .find(|at| matches!(self.instrs[*at as usize], Instr::Branch { .. }))
            .ok_or_else(|| ComposeError::Segment { from: top, to: end, why: "no head".into() })?;
        self.glue(branch + 2, &[])?;
        if self.events(body, &next)?.is_some() {
            return Err(ComposeError::Unsupported("HALT inside a loop".into()));
        }
        // The tail: the index slot holds `var + 1`, then the jump back to the head.
        let steps = self.segment(self.pos, end - 1)? + 1;
        let next = format!("{var} + 1");
        let (_, stored) =
            self.phase(self.pos, top, steps, &[], &[(index, format!("({next})"))], &[])?;
        let find = |g: &Self, a: &Sym, n: u64| {
            g.facts
                .iter()
                .find(|f| f.addr == *a && f.len == n)
                .cloned()
                .ok_or_else(|| ComposeError::Untracked { addr: a.lean(), len: n })
        };
        let mut next_parts =
            vec![format!("Compose.u64_holds (v := {next}) {stored}.1 (by decide) (by decide)")];
        for o in &outer {
            let f = self
                .facts
                .iter()
                .find(|f| f.addr == o.addr && f.expr == o.expr)
                .ok_or_else(|| ComposeError::Untracked { addr: o.addr.lean(), len: o.len })?;
            next_parts.push(f.name.clone());
        }
        // Each carried value's step, from the fact at its slot.
        let mut steps_of = Vec::new();
        for (n, (a, size)) in carried.iter().enumerate() {
            let f = find(self, &Sym::num(*a), *size)?;
            steps_of.push(f.expr.replace(&format!("({} {var})", carried_id(n)), "c"));
            next_parts.push(f.length.clone());
            next_parts.push(f.name.clone());
        }
        let mut members = Vec::new();
        let mut parents = Vec::new();
        for family in self.families.clone() {
            if family.var != var {
                parents.push(family.name.clone());
                continue;
            }
            let mut at = family.base.clone();
            at.terms.push((var.clone(), family.n.into()));
            let f = find(self, &at, family.n)?;
            match &self.draw {
                Some((b, _, _)) if drawing.is_some() => {
                    let step = format!("fun {var} {b} => {}", f.expr);
                    members.push(format!("fun j => ({step}) j (pre.getD j [])"));
                    next_parts.push(format!(
                        "Compose.members_snoc ({step}) hpre {} {} {}",
                        family.name, f.length, f.name
                    ));
                }
                _ => {
                    members.push(format!("fun {var} => {}", f.expr));
                    next_parts.push(format!(
                        "Compose.members_succ {} {} {}",
                        family.name, f.length, f.name
                    ));
                }
            }
        }
        next_parts.extend(parents);
        let drawn = match (&self.draw, drawing) {
            (Some((b, hb, _)), Some(_)) if !b.is_empty() => format!(" {b} {} {hb}", self.state),
            (_, Some(_)) => return Err(ComposeError::Unsupported("a draw not reached".into())),
            _ => format!(" {}", self.state),
        };
        writeln!(self.proof, "  {k}{drawn} ⟨{}⟩", next_parts.join(", ")).unwrap();
        let c = self.total;
        let budgets: String = self
            .local_calls
            .iter()
            .map(|k| format!("  have hs{k} := hsteps{k}{}\n", self.loop_instances()))
            .collect();
        self.ctx.pop();
        self.scope.truncate(scope_from);
        self.draw = None;

        // `C k` as a term: `Nat.rec` from the initial value over the step.
        let functions: Vec<String> = initial_carried
            .iter()
            .zip(&steps_of)
            .map(|(init, step)| {
                format!(
                    "(fun n => Nat.rec (motive := fun _ => Bytes) ({}) (fun {var} c => {step}) n)",
                    init.expr
                )
            })
            .collect();
        let substitute = |text: &str| {
            let mut text = text.to_owned();
            for (n, m) in members.iter().enumerate() {
                text = text.replace(&format!("__O{}__", first + n), m);
            }
            for (n, f) in functions.iter().enumerate() {
                text = text.replace(&carried_id(n), f);
            }
            text
        };
        let body_text = substitute(&std::mem::take(&mut self.proof));
        // The leaves' hypotheses within the body name the carried values and members too.
        for h in &mut self.hypotheses[hypotheses_from.0..] {
            *h = substitute(h);
        }
        for h in &mut self.budgets[hypotheses_from.1..] {
            *h = substitute(h);
        }
        let (proof, facts, outer_state, total, local_calls) = saved;
        self.proof = proof;
        self.facts = facts;
        self.families.clear();
        self.state = outer_state;
        self.total = total + u128::from(count) * c;
        self.local_calls = local_calls;

        // The invariant at the head, and its parts at entry.
        let mut invariant = vec![format!("Holds s {index} (Bytes.u64 {var})")];
        let mut initial = vec![zero];
        for f in &outer {
            invariant.push(format!("Holds s {} ({})", f.addr.lean(), f.expr));
            let kept = self.facts.iter().find(|g| g.addr == f.addr && g.expr == f.expr);
            initial.push(kept.expect("kept").name.clone());
        }
        for (((a, size), func), init) in carried.iter().zip(&functions).zip(&initial_carried) {
            invariant.push(format!("({func} {var}).length = {size}"));
            invariant.push(format!("Holds s {a} ({func} {var})"));
            initial.push(init.length.clone());
            initial.push(init.name.clone());
        }
        for ((base, n), m) in families.iter().zip(&members) {
            invariant.push(format!("Compose.Members s {} {n} ({m}) {var}", base.lean()));
            initial.push(if drawing.is_some() {
                "Compose.members_zero _ _ _ _".to_owned()
            } else {
                format!("Compose.members_zero {} {} {n} ({m})", self.state, base.lean())
            });
        }
        for f in &outer_families {
            invariant.push(format!(
                "Compose.Members s {} {} ({}) {}",
                f.base.lean(),
                f.n,
                f.members,
                f.var
            ));
            initial.push(f.name.clone());
        }
        let indented: String = body_text.lines().map(|l| format!("    {l}\n")).collect();
        let header: String = header.lines().map(|l| format!("  {l}\n")).collect();
        let budgets: String = budgets.lines().map(|l| format!("  {l}\n")).collect();
        let (exit_state, exit_inv) = (self.name("s"), self.name("hI"));
        // A drawing loop's invariant and exit also take the draws.
        let (bss, hbss) = (format!("bss{}", self.fresh), format!("hbss{}", self.fresh));
        let (lemma, pre, body_args, exit_args) = match drawing {
            Some(n) => (
                format!("Compose.loopCoins {top} {count} {c} {n}"),
                " pre",
                format!("{var} {hyp} pre {state} hpre {inv} {r} Kc {k}"),
                format!("{bss} {exit_state} {hbss} {exit_inv}"),
            ),
            None => (
                format!("Compose.loop {top} {count} {c}"),
                "",
                format!("{var} {hyp} {state} {inv} {r} {k}"),
                format!("{exit_state} {exit_inv}"),
            ),
        };
        writeln!(
            self.proof,
            "  {lemma} (fun {var}{pre} s => {})\n    (fun {body_args} =>\n{header}{budgets}    Prog.Runs.mono (\n{indented}    ) (by omega))\n    {} ⟨{}⟩ fun {exit_args} =>",
            invariant.join(" ∧ "),
            self.state,
            initial.join(", ")
        )
        .unwrap();
        if let Some(n) = drawing {
            self.scope.push((
                format!("∀ {bss} : List Bytes, {bss}.length = {count} → "),
                format!(" {bss} {hbss}"),
            ));
            self.wrappers.push(format!("Compose.coinsSeq {n} {count} >>= fun {bss} =>"));
            let _ = n;
        }

        // The exit: the facts the loop left alone, the index at the count, the carried values at
        // the count, and each family whole.
        self.state = exit_state;
        self.facts = Vec::new();
        let fx = self.name("fx");
        writeln!(self.proof, "  have {fx} := {exit_inv}{}", conj(0)).unwrap();
        self.facts.push(Fact {
            name: fx,
            addr: Sym::num(index),
            len: 8,
            expr: format!("Bytes.u64 {count}"),
            length: format!("(Mem.length_u64 {count})"),
            image: false,
        });
        for (n, f) in outer.iter().enumerate() {
            let name = self.name("f");
            writeln!(self.proof, "  have {name} := {exit_inv}{}", conj(at_outer + n)).unwrap();
            self.facts.push(Fact { name, ..f.clone() });
        }
        for (n, ((a, size), func)) in carried.iter().zip(&functions).enumerate() {
            let (length, name) = (self.name("lc"), self.name("c"));
            writeln!(
                self.proof,
                "  have {length} := {exit_inv}{}\n  have {name} := {exit_inv}{}",
                conj(at_carried + 2 * n),
                conj(at_carried + 2 * n + 1)
            )
            .unwrap();
            self.facts.push(Fact {
                name,
                addr: Sym::num(*a),
                len: *size,
                expr: format!("({func} {count})"),
                length,
                image: false,
            });
        }
        for (n, ((base, size), m)) in families.iter().zip(&members).enumerate() {
            let (members_fact, name, length) = (self.name("m"), self.name("w"), self.name("lw"));
            let expr = format!(
                "(List.range {count}).flatMap ({})",
                m.replace("pre.getD", &format!("{bss}.getD"))
            );
            let len = size * count;
            writeln!(
                self.proof,
                "  have {members_fact} := {exit_inv}{}\n  have {name} : Holds {} {} ({expr}) := Compose.members_holds {members_fact}\n  have {length} : ({expr}).length = {len} := by\n    rw [Compose.length_flatMap_range fun j hj => ({members_fact} j hj).1]",
                conj(at_families + n),
                self.state,
                base.lean()
            )
            .unwrap();
            self.facts.push(Fact { name, addr: base.clone(), len, expr, length, image: false });
        }
        for (n, family) in outer_families.iter().enumerate() {
            let name = self.name("m");
            writeln!(self.proof, "  have {name} := {exit_inv}{}", conj(at_parents + n)).unwrap();
            self.families.push(Family { name, ..family.clone() });
        }
        let steps = (branch - top) + 2;
        let words = [(index, count.to_string())];
        self.phase(top, end, steps, &[], &[], &words)?;
        self.pos = end;
        Ok(())
    }
}

/// Sorted, merged ranges.
fn merge(mut ranges: Vec<(i128, i128)>) -> Vec<(i128, i128)> {
    ranges.sort_unstable();
    let mut out: Vec<(i128, i128)> = Vec::new();
    for r in ranges {
        match out.last_mut() {
            Some(last) if r.0 <= last.1 => last.1 = last.1.max(r.1),
            _ => out.push(r),
        }
    }
    out
}

/// The ranges `event` reads (over all values of the indices of `ctx`, and of its loops), apart
/// from the image, which is always kept.
fn event_reads(
    event: &GlueEvent,
    ctx: &mut Vec<Index>,
    out: &mut Vec<(i128, i128)>,
) -> Result<(), ComposeError> {
    let mut push = |ctx: &[Index], a: &Sym, n: u64| {
        let (lo, hi) = a.range(ctx);
        out.push((lo, hi + i128::from(n)));
    };
    match event {
        GlueEvent::Copy { src, len, .. } => push(ctx, &sym_in(ctx, src)?, *len),
        GlueEvent::CopyAt { base, stride, index, words, count, .. } => {
            push(ctx, &Sym::num(*base), stride * count);
            push(ctx, &Sym::num(*index), 8 * words);
        }
        GlueEvent::CopySelect { candidates, index, words, len, .. } => {
            for c in candidates {
                push(ctx, &Sym::num(*c), *len);
            }
            push(ctx, &Sym::num(*index), 8 * words);
        }
        GlueEvent::Leaf { inputs, .. } => {
            for (a, n) in inputs {
                push(ctx, &Sym::num(*a), *n);
            }
        }
        GlueEvent::Halt { payload, len, .. } => push(ctx, &Sym::num(*payload), *len),
        GlueEvent::Loop { slot, index, count, body, .. } => {
            let depth = ctx.len();
            ctx.push(Index {
                slot: *slot,
                var: format!("i{depth}"),
                hyp: format!("hi{depth}"),
                count: *count,
                addr: *index,
                stop: false,
            });
            let inner: Result<(), ComposeError> =
                body.iter().try_for_each(|e| event_reads(e, ctx, out));
            ctx.pop();
            inner?;
        }
        GlueEvent::Query { .. } | GlueEvent::Coins { .. } => {}
    }
    Ok(())
}

/// Whether `events` draw coins, in loops included.
fn draws_any(events: &[GlueEvent]) -> bool {
    events.iter().any(|e| match e {
        GlueEvent::Coins { .. } => true,
        GlueEvent::Loop { body, .. } => draws_any(body),
        _ => false,
    })
}

/// The ranges `body` writes over all iterations of the innermost loop of `ctx` and of the loops
/// it contains, and, when `families` is given, the families the innermost loop fills one member
/// per iteration.
fn collect_writes(
    body: &[GlueEvent],
    ctx: &mut Vec<Index>,
    ranges: &mut Vec<(i128, i128)>,
    mut families: Option<&mut Vec<(Sym, u64)>>,
) -> Result<(), ComposeError> {
    let var = ctx.last().expect("the loop").var.clone();
    let range = |ranges: &mut Vec<(i128, i128)>, ctx: &[Index], a: &Sym, n: u64| {
        let (lo, hi) = a.range(ctx);
        ranges.push((lo, hi + i128::from(n)));
    };
    for event in body {
        match event {
            GlueEvent::Copy { dst, len, .. } => {
                let dst = sym_in(ctx, dst)?;
                if let Some(families) = families.as_deref_mut() &&
                    let Some((_, f)) = dst.terms.iter().find(|(v, _)| *v == var) &&
                    *f == i128::from(*len)
                {
                    let mut base = dst.clone();
                    base.terms.retain(|(v, _)| *v != var);
                    if !families.contains(&(base.clone(), *len)) {
                        families.push((base, *len));
                    }
                }
                range(ranges, ctx, &dst, *len);
            }
            GlueEvent::Leaf { descriptor, arguments, inputs, outputs, scratch, .. } => {
                for (a, n) in outputs {
                    range(ranges, ctx, &Sym::num(*a), *n);
                }
                range(ranges, ctx, &Sym::num(scratch.0), scratch.1);
                if arguments
                    .iter()
                    .any(|a| !matches!(a, Argument::Affine { terms, .. } if terms.is_empty()))
                {
                    let words = arguments.len() + inputs.len() + outputs.len() + 1;
                    range(ranges, ctx, &Sym::num(*descriptor), 8 * words as u64);
                }
            }
            GlueEvent::Loop { slot, index, count, body, .. } => {
                range(ranges, ctx, &Sym::num(*index), 8);
                let depth = ctx.len();
                ctx.push(Index {
                    slot: *slot,
                    var: format!("i{depth}"),
                    hyp: format!("hi{depth}"),
                    count: *count,
                    addr: *index,
                    stop: false,
                });
                let inner = collect_writes(body, ctx, ranges, None);
                ctx.pop();
                inner?;
            }
            GlueEvent::CopyAt { dst, stride, .. } => range(ranges, ctx, &Sym::num(*dst), *stride),
            GlueEvent::CopySelect { dst, len, .. } => range(ranges, ctx, &Sym::num(*dst), *len),
            GlueEvent::Coins { at, len, .. } => range(ranges, ctx, &Sym::num(*at), *len),
            GlueEvent::Query { .. } | GlueEvent::Halt { .. } => {}
        }
    }
    Ok(())
}

/// The `jalr` offset of the call at instruction `call`.
fn call_offset(instrs: &[Instr], call: u64) -> String {
    match instrs[call as usize + 1] {
        Instr::Jalr { imm, .. } => format!("0x{:x}#12", (imm as u64) & 0xfff),
        other => panic!("call at {call} is not auipc/jalr: {other:?}"),
    }
}

/// The run module of the algorithm `algorithm` of the claim in namespace `namespace` (whose
/// `Progs` module defines its program), as `(module name, source)`.
pub fn run_module(
    namespace: &str,
    algorithm: &str,
    image: &ProofImage,
    leaves: &BTreeMap<String, LeanLeaf>,
) -> Result<(String, String), ComposeError> {
    let (events, copy_word) = image.trace.as_ref().map_err(|gap| ComposeError::NoTrace(gap))?;
    let mut g = Gen {
        image,
        leaves,
        instrs: crate::lean::code_prefix(&image.bytes),
        lemmas: String::new(),
        proof: String::new(),
        facts: vec![Fact {
            name: "f0".into(),
            addr: Sym::num(BASE),
            len: image.bytes.len() as u64,
            expr: "P.bin".into(),
            length: "P_len".into(),
            image: true,
        }],
        families: Vec::new(),
        state: "P.init".into(),
        fresh: 0,
        image_lemmas: BTreeMap::new(),
        ctx: Vec::new(),
        checked: None,
        pos: 0,
        total: 0,
        specs: String::new(),
        spec_names: BTreeMap::new(),
        modules: Vec::new(),
        hypotheses: Vec::new(),
        budgets: Vec::new(),
        calls: 0,
        local_calls: Vec::new(),
        phases: 0,
        family_ids: 0,
        tables: Vec::new(),
        member: None,
        select: None,
        scope: Vec::new(),
        wrappers: Vec::new(),
        draw: None,
        live: Vec::new(),
        defs: String::new(),
        defs_count: 0,
        flag: None,
        globals: Vec::new(),
    };
    let payload =
        g.events(events, &[])?.ok_or_else(|| ComposeError::Unsupported("no HALT".into()))?;
    if g.total > image.step_bound {
        return Err(ComposeError::Bound { steps: g.total, bound: image.step_bound });
    }
    let binlen = image.bytes.len();
    let mem = image.mem_size;

    let module = format!("{namespace}.Run.{algorithm}");
    let mut out = String::new();
    writeln!(out, "-- Generated by mxx-core. Do not edit.").unwrap();
    writeln!(out, "import {namespace}.Progs").unwrap();
    writeln!(out, "import MxxCore.Compose").unwrap();
    for m in &g.modules {
        writeln!(out, "import {m}").unwrap();
    }
    writeln!(
        out,
        "\n/-!\n# The run of `{algorithm}`, composed from its leaves' specifications\n\nGenerated by `mxx_core::compose` from the trace of the image's glue.\n-/\n"
    )
    .unwrap();
    writeln!(out, "namespace {module}\n").unwrap();
    writeln!(out, "open MathcalA MxxCore MxxCore.Leaf RiscvZkvm.Rv64\n").unwrap();
    // Each segment's closing tactic covers every shape `mrun` leaves; most use one alternative.
    writeln!(
        out,
        "set_option linter.unusedTactic false\nset_option linter.unreachableTactic false\nset_option linter.unnecessarySeqFocus false\nset_option linter.unusedVariables false\nset_option linter.unusedSimpArgs false\n"
    )
    .unwrap();
    writeln!(out, "abbrev P : Prog := {namespace}.Progs.{algorithm}\n").unwrap();
    writeln!(
        out,
        "set_option maxRecDepth 1000000 in\ntheorem P_len : P.bin.length = {binlen} := by decide\n"
    )
    .unwrap();
    writeln!(
        out,
        "theorem P_mem : P.mem = {mem} := rfl\n\ntheorem hP : Compose.Sizes P {mem} {binlen} := ⟨P_mem, P_len, by decide, by decide⟩\n"
    )
    .unwrap();
    writeln!(
        out,
        "theorem copyAt : MxxCore.Copy.At P {copy_word} := by\n  intro i hi\n  simp only [MxxCore.Copy.code, List.length_cons, List.length_nil] at hi\n  interval_cases i <;> rfl\n"
    )
    .unwrap();
    out.push_str(&g.specs);
    out.push_str(&g.defs);
    out.push_str(&g.lemmas);
    let binders: String =
        g.hypotheses.iter().chain(&g.budgets).map(|h| format!("\n    {h}")).collect();
    let budgets: String =
        g.local_calls.iter().map(|k| format!("  have hs{k} := hsteps{k}\n")).collect();
    // The run as one term, so its step count is inferred from the events, then bounded.
    let term: String = g.proof.lines().map(|l| format!("    {l}\n")).collect();
    writeln!(
        out,
        "set_option maxRecDepth 100000 in\nset_option maxHeartbeats 0 in\n/-- The run halts with the payload the leaves' functions describe, within the image's step\nbound. -/\ntheorem halts (ins : List Bytes){binders} :\n    P.Runs (MxxCore.inputGame ins) 0 P.init () {}\n      ({}pure (.halted ({payload}), ())) := by\n  have f0 : Holds P.init BASE P.bin := Compose.init hP\n{budgets}  have h :=\n{term}  exact Prog.Runs.mono h (by omega)\n",
        image.step_bound,
        g.wrappers.iter().map(|w| format!("{w} ")).collect::<String>()
    )
    .unwrap();
    writeln!(out, "end {module}").unwrap();
    Ok((module, out))
}
