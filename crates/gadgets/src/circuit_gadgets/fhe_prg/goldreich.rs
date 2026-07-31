//! Goldreich/TSA local PRG gadget over Ring-GSW ciphertext bits.
//!
//! For a secret bit vector `x in {0,1}^n` and a public graph with `m` predicate edges, each
//! output bit is
//!
//! ```text
//! y_i = x[a_i] XOR x[b_i] XOR x[c_i] XOR (x[d_i] AND x[e_i]).
//! ```
//!
//! The main artifact in this module is [`GoldreichFhePrg`], which fixes the public graph at setup
//! time and then evaluates the predicate family over encrypted bits represented as
//! `Vec<RingGswCiphertext<P>>`.
//!
//! The public graph is generated deterministically from `graph_seed` alone, independently of the
//! secret PRG seed. Each edge contains five distinct indices from `[0, n)`, preserves the role
//! split as three XOR inputs plus two AND inputs, and rejects duplicates by canonicalizing only
//! within the XOR triple and within the AND pair. An optional strict mode also rejects reuse of
//! the same underlying 5-set regardless of role assignment.
//!
//! At the Boolean-ring level, XOR is addition mod 2 and AND is multiplication mod 2, so the TSA
//! predicate has one nonlinear term per output. In this repository, however,
//! [`RingGswCiphertext::xor`] internally uses ciphertext multiplication, so the concrete
//! `PolyCircuit` is deeper than the abstract Boolean circuit. The implementation therefore keeps
//! XOR composition balanced instead of chaining it left-to-right.
use crate::{
    circuit::{BatchedWire, PolyCircuit, gate::GateId},
    circuit_gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner},
        fhe::ring_gsw::{RingGswCiphertext, RingGswContext},
    },
    matrix::PolyMatrix,
    poly::Poly,
};
use digest::Digest;
use keccak_asm::Keccak256;
use num_bigint::BigUint;
use num_traits::Zero;
use rayon::prelude::*;
use std::{collections::HashSet, sync::Arc};
use tracing::debug;

pub trait BooleanCiphertext<P: Poly>: Clone {
    type Context;

    fn context(&self) -> &Arc<Self::Context>;

    fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self;

    fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self;

    fn and(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self;

    fn xor(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self;

    fn sub_circuit_input(context: Arc<Self::Context>, circuit: &mut PolyCircuit<P>) -> Self;

    fn sub_circuit_wires(&self) -> Vec<BatchedWire>;

    fn from_sub_circuit_outputs(template: &Self, outputs: &[BatchedWire]) -> Self;
}

impl<P: Poly + 'static, A> BooleanCiphertext<P> for RingGswCiphertext<P, A>
where
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
{
    type Context = RingGswContext<P, A>;

    fn context(&self) -> &Arc<Self::Context> {
        &self.ctx
    }

    fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        RingGswCiphertext::add(self, other, circuit)
    }

    fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        RingGswCiphertext::sub(self, other, circuit)
    }

    fn and(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        RingGswCiphertext::and(self, other, circuit)
    }

    fn xor(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        RingGswCiphertext::xor(self, other, circuit)
    }

    fn sub_circuit_input(context: Arc<Self::Context>, circuit: &mut PolyCircuit<P>) -> Self {
        RingGswCiphertext::input(context, None, circuit)
    }

    fn sub_circuit_wires(&self) -> Vec<BatchedWire> {
        self.sub_circuit_wires()
    }

    fn from_sub_circuit_outputs(template: &Self, outputs: &[BatchedWire]) -> Self {
        RingGswCiphertext::from_sub_circuit_outputs(template, outputs)
    }
}
/// Public graph-generation options for the Goldreich/TSA PRG.
///
/// The default mode rejects only role-aware duplicates:
/// two edges collide if they use the same XOR triple and the same AND pair,
/// regardless of order inside each role group. When `reject_same_vertex_set` is enabled, the
/// generator also rejects edges that reuse the same 5-set with a different role split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoldreichGraphGeneration {
    pub reject_same_vertex_set: bool,
}

impl Default for GoldreichGraphGeneration {
    fn default() -> Self {
        Self { reject_same_vertex_set: false }
    }
}

impl GoldreichGraphGeneration {
    fn max_unique_edges(self, input_size: usize) -> u128 {
        if self.reject_same_vertex_set {
            binomial(input_size, 5)
        } else {
            binomial(input_size, 3) * binomial(input_size - 3, 2)
        }
    }
}

/// Returns whether a Goldreich/TSA PRG output length satisfies `m < n^1.4`.
///
/// The comparison is evaluated exactly as `m^5 < n^7` to avoid floating-point rounding at the
/// boundary.
pub fn goldreich_output_bound_holds(input_size: usize, output_size: usize) -> bool {
    if input_size < 5 || output_size == 0 {
        return false;
    }
    let output = BigUint::from(output_size);
    let input = BigUint::from(input_size);
    output.pow(5) < input.pow(7)
}

/// Asserts the Goldreich/TSA PRG output-length bound for a named construction site.
pub fn assert_goldreich_output_bound(input_size: usize, output_size: usize, context: &str) {
    assert!(
        goldreich_output_bound_holds(input_size, output_size),
        "{context} violates Goldreich PRG safety bound m < n^1.4: input_size={input_size}, output_size={output_size}"
    );
}

/// Returns the smallest Goldreich/TSA input size `n` satisfying `output_size < n^1.4`.
pub fn minimum_goldreich_input_size(output_size: usize) -> usize {
    assert!(output_size > 0, "Goldreich PRG output_size must be positive");
    let mut high = 5usize;
    while !goldreich_output_bound_holds(high, output_size) {
        high = high.checked_mul(2).expect("Goldreich minimum input size search overflow");
    }
    let mut low = 5usize;
    while low < high {
        let mid = low + (high - low) / 2;
        if goldreich_output_bound_holds(mid, output_size) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    low
}

/// One public Goldreich/TSA predicate edge.
///
/// The role split is preserved explicitly:
/// `xor_inputs` feeds the linear XOR part and `and_inputs` feeds the nonlinear AND part.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GoldreichEdge {
    pub xor_inputs: [usize; 3],
    pub and_inputs: [usize; 2],
}

/// Canonical role-aware key used for duplicate rejection.
///
/// The XOR triple is sorted internally, and the AND pair is sorted internally,
/// but the two role groups remain separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GoldreichEdgeKey {
    pub xor_inputs: [usize; 3],
    pub and_inputs: [usize; 2],
}

impl GoldreichEdge {
    /// Creates one public TSA edge with explicit XOR and AND role groups.
    ///
    /// All five indices must be distinct because the public graph must not reuse an input bit
    /// inside a single predicate application.
    pub fn new(xor_inputs: [usize; 3], and_inputs: [usize; 2]) -> Self {
        let edge = Self { xor_inputs, and_inputs };
        assert!(
            all_distinct(&edge.all_inputs()),
            "Goldreich edge inputs must be pairwise distinct across XOR and AND roles"
        );
        edge
    }

    pub fn all_inputs(&self) -> [usize; 5] {
        [
            self.xor_inputs[0],
            self.xor_inputs[1],
            self.xor_inputs[2],
            self.and_inputs[0],
            self.and_inputs[1],
        ]
    }

    pub fn role_aware_key(&self) -> GoldreichEdgeKey {
        let mut xor_inputs = self.xor_inputs;
        xor_inputs.sort_unstable();
        let mut and_inputs = self.and_inputs;
        and_inputs.sort_unstable();
        GoldreichEdgeKey { xor_inputs, and_inputs }
    }

    pub fn same_vertex_set_key(&self) -> [usize; 5] {
        let mut all_inputs = self.all_inputs();
        all_inputs.sort_unstable();
        all_inputs
    }
}

/// Public Goldreich graph fixed at setup time.
///
/// The graph is a public parameter of the PRG, not a circuit input. It may either be generated
/// deterministically from a public `graph_seed` or validated from an explicit edge list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoldreichGraph {
    pub input_size: usize,
    pub edges: Vec<GoldreichEdge>,
    pub graph_seed: Option<[u8; 32]>,
    pub generation: GoldreichGraphGeneration,
}

/// Stateful generator for contiguous chunks of one full-domain Goldreich graph.
///
/// It preserves the exact duplicate-rejection state of full graph generation while allowing callers
/// to stream chunks without holding every edge or replaying earlier prefixes for each chunk.
pub struct GoldreichFullDomainRangeGenerator {
    input_size: usize,
    conceptual_output_size: usize,
    full_range_seed: [u8; 32],
    generation: GoldreichGraphGeneration,
    stream: GraphSeedStream,
    accepted_count: usize,
    seen_role_keys: HashSet<GoldreichEdgeKey>,
    seen_vertex_sets: Option<HashSet<[usize; 5]>>,
}

impl GoldreichFullDomainRangeGenerator {
    pub fn new(
        input_size: usize,
        conceptual_output_size: usize,
        graph_seed: [u8; 32],
        generation: GoldreichGraphGeneration,
    ) -> Self {
        validate_graph_dimensions(input_size, conceptual_output_size);
        assert_goldreich_output_bound(
            input_size,
            conceptual_output_size,
            "Goldreich full-domain range generator",
        );
        let capacity = generation.max_unique_edges(input_size);
        assert!(
            (conceptual_output_size as u128) <= capacity,
            "requested conceptual Goldreich graph output size {} exceeds unique-edge capacity {} for input_size={input_size}",
            conceptual_output_size,
            capacity
        );
        let full_range_seed =
            derive_range_graph_seed(graph_seed, conceptual_output_size, 0, conceptual_output_size);
        Self {
            input_size,
            conceptual_output_size,
            full_range_seed,
            generation,
            stream: GraphSeedStream::new(full_range_seed),
            accepted_count: 0,
            seen_role_keys: HashSet::new(),
            seen_vertex_sets: if generation.reject_same_vertex_set {
                Some(HashSet::new())
            } else {
                None
            },
        }
    }

    pub fn next_range(&mut self, range_start: usize, range_len: usize) -> GoldreichGraph {
        assert!(range_len > 0, "Goldreich graph output range length must be positive");
        assert!(
            range_start >= self.accepted_count,
            "Goldreich full-domain range generator only supports forward ranges"
        );
        let range_end =
            range_start.checked_add(range_len).expect("Goldreich graph output range end overflow");
        assert!(
            range_end <= self.conceptual_output_size,
            "Goldreich graph output range [{range_start}, {range_end}) exceeds conceptual output size {}",
            self.conceptual_output_size
        );

        while self.accepted_count < range_start {
            sample_next_unique_edge(
                self.input_size,
                &mut self.stream,
                &mut self.seen_role_keys,
                &mut self.seen_vertex_sets,
            );
            self.accepted_count += 1;
        }

        let mut edges = Vec::with_capacity(range_len);
        while self.accepted_count < range_end {
            edges.push(sample_next_unique_edge(
                self.input_size,
                &mut self.stream,
                &mut self.seen_role_keys,
                &mut self.seen_vertex_sets,
            ));
            self.accepted_count += 1;
        }

        GoldreichGraph {
            input_size: self.input_size,
            edges,
            graph_seed: Some(self.full_range_seed),
            generation: self.generation,
        }
    }
}

impl GoldreichGraph {
    /// Deterministically generates a public Goldreich graph from `graph_seed`.
    ///
    /// The sampler uses a counter-mode Keccak stream plus rejection sampling for unbiased vertex
    /// selection in `[0, input_size)`. Each accepted edge keeps the first three sampled indices as
    /// `xor_inputs`, the last two as `and_inputs`, and rejects duplicates by the role-aware key
    /// `(sort(xor_inputs), sort(and_inputs))`. In strict mode it also rejects reuse of the same
    /// sorted 5-set.
    pub fn generate(
        input_size: usize,
        output_size: usize,
        graph_seed: [u8; 32],
        generation: GoldreichGraphGeneration,
    ) -> Self {
        validate_graph_dimensions(input_size, output_size);
        assert_goldreich_output_bound(input_size, output_size, "Goldreich graph generation");
        let capacity = generation.max_unique_edges(input_size);
        assert!(
            (output_size as u128) <= capacity,
            "requested Goldreich graph output size {} exceeds unique-edge capacity {} for input_size={input_size}",
            output_size,
            capacity
        );

        let mut stream = GraphSeedStream::new(graph_seed);
        let mut edges = Vec::with_capacity(output_size);
        let mut seen_role_keys = HashSet::with_capacity(output_size);
        let mut seen_vertex_sets = if generation.reject_same_vertex_set {
            Some(HashSet::with_capacity(output_size))
        } else {
            None
        };

        while edges.len() < output_size {
            edges.push(sample_next_unique_edge(
                input_size,
                &mut stream,
                &mut seen_role_keys,
                &mut seen_vertex_sets,
            ));
        }

        Self { input_size, edges, graph_seed: Some(graph_seed), generation }
    }

    /// Generates only one output interval of a conceptual full Goldreich graph.
    ///
    /// The selected range is domain-separated by `(conceptual_output_size, range_start,
    /// range_len)`, then generated with exactly `range_len` public edges.  This keeps the cost and
    /// capacity requirements proportional to the selected interval while preserving the important
    /// indexing invariant: two ranges with the same width but different starts use different public
    /// graph edges.
    ///
    /// TODO: This range-local generation does not verify collisions against every edge in the
    /// conceptual full graph.  Doing so would require generating the full prefix or maintaining a
    /// shared global duplicate-rejection state, which is currently too expensive for the large
    /// noise-refresh mask/error offsets.  The range seed still domain-separates different output
    /// intervals, but it does not provide the same global no-collision guarantee as full graph
    /// generation.
    pub fn generate_range(
        input_size: usize,
        conceptual_output_size: usize,
        range_start: usize,
        range_len: usize,
        graph_seed: [u8; 32],
        generation: GoldreichGraphGeneration,
    ) -> Self {
        validate_graph_dimensions(input_size, conceptual_output_size);
        assert_goldreich_output_bound(
            input_size,
            conceptual_output_size,
            "Goldreich range graph generation",
        );
        assert!(range_len > 0, "Goldreich graph output range length must be positive");
        let range_end =
            range_start.checked_add(range_len).expect("Goldreich graph output range end overflow");
        assert!(
            range_end <= conceptual_output_size,
            "Goldreich graph output range [{range_start}, {range_end}) exceeds conceptual output size {conceptual_output_size}"
        );
        let range_seed =
            derive_range_graph_seed(graph_seed, conceptual_output_size, range_start, range_len);
        Self::generate(input_size, range_len, range_seed, generation)
    }

    /// Generates one interval using the same public graph domain as the full conceptual range.
    ///
    /// This is more expensive than [`generate_range`] because it replays full-range generation up
    /// to `range_start + range_len`, including duplicate-rejection state. It is needed when a
    /// caller persists artifacts against the full-range graph and later wants to evaluate only one
    /// contiguous output chunk without changing the graph semantics.
    pub fn generate_full_domain_range(
        input_size: usize,
        conceptual_output_size: usize,
        range_start: usize,
        range_len: usize,
        graph_seed: [u8; 32],
        generation: GoldreichGraphGeneration,
    ) -> Self {
        validate_graph_dimensions(input_size, conceptual_output_size);
        assert_goldreich_output_bound(
            input_size,
            conceptual_output_size,
            "Goldreich full-domain range graph generation",
        );
        assert!(range_len > 0, "Goldreich graph output range length must be positive");
        let range_end =
            range_start.checked_add(range_len).expect("Goldreich graph output range end overflow");
        assert!(
            range_end <= conceptual_output_size,
            "Goldreich graph output range [{range_start}, {range_end}) exceeds conceptual output size {conceptual_output_size}"
        );
        let capacity = generation.max_unique_edges(input_size);
        assert!(
            (conceptual_output_size as u128) <= capacity,
            "requested conceptual Goldreich graph output size {} exceeds unique-edge capacity {} for input_size={input_size}",
            conceptual_output_size,
            capacity
        );

        GoldreichFullDomainRangeGenerator::new(
            input_size,
            conceptual_output_size,
            graph_seed,
            generation,
        )
        .next_range(range_start, range_len)
    }

    /// Validates an explicit public Goldreich graph against the same invariants as [`generate`].
    ///
    /// This is useful for tests or for callers that want to pin a hand-written public graph while
    /// still enforcing distinct indices, in-range vertices, and the configured duplicate-rejection
    /// policy.
    pub fn from_edges(
        input_size: usize,
        edges: Vec<GoldreichEdge>,
        generation: GoldreichGraphGeneration,
    ) -> Self {
        validate_graph_dimensions(input_size, edges.len());
        assert_goldreich_output_bound(input_size, edges.len(), "explicit Goldreich graph");
        let capacity = generation.max_unique_edges(input_size);
        assert!(
            (edges.len() as u128) <= capacity,
            "explicit Goldreich graph output size {} exceeds unique-edge capacity {} for input_size={input_size}",
            edges.len(),
            capacity
        );

        edges.par_iter().for_each(|edge| {
            let all_inputs = edge.all_inputs();
            assert!(
                all_distinct(&all_inputs),
                "Goldreich edge inputs must be pairwise distinct across XOR and AND roles"
            );
            for index in all_inputs {
                assert!(
                    index < input_size,
                    "Goldreich edge index {} must lie in [0, {})",
                    index,
                    input_size
                );
            }
        });

        let seen_role_keys =
            edges.par_iter().map(GoldreichEdge::role_aware_key).collect::<HashSet<_>>();
        assert_eq!(
            seen_role_keys.len(),
            edges.len(),
            "Goldreich graph must not contain duplicate role-aware edge keys"
        );
        if generation.reject_same_vertex_set {
            let seen_vertex_sets =
                edges.par_iter().map(GoldreichEdge::same_vertex_set_key).collect::<HashSet<_>>();
            assert_eq!(
                seen_vertex_sets.len(),
                edges.len(),
                "Goldreich graph strict mode must not reuse the same 5-set with a different role split"
            );
        }

        Self { input_size, edges, graph_seed: None, generation }
    }

    pub fn output_size(&self) -> usize {
        self.edges.len()
    }
}

/// Fixed-data setup object for the Goldreich/TSA PRG evaluated over Ring-GSW ciphertext bits.
///
/// This struct owns the Ring-GSW context together with the fixed public graph and fixed PRG
/// dimensions. Those values are setup-time constants rather than runtime circuit inputs; the only
/// runtime inputs to [`GoldreichFhePrg::evaluate`] are encrypted secret bits.
#[derive(Debug, Clone)]
pub struct GoldreichFhePrg<P: Poly, C: BooleanCiphertext<P>> {
    ring_gsw: Arc<C::Context>,
    input_size: usize,
    output_size: usize,
    public_graph: GoldreichGraph,
}

impl<P: Poly, C: BooleanCiphertext<P>> GoldreichFhePrg<P, C> {
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Returns the fixed public graph used by this PRG instance.
    pub fn graph(&self) -> &GoldreichGraph {
        &self.public_graph
    }
}

impl<P: Poly + 'static, C> GoldreichFhePrg<P, C>
where
    C: BooleanCiphertext<P>,
{
    /// Generates the fixed public graph from a public `graph_seed` and stores it with the
    /// Ring-GSW context.
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        ring_gsw: Arc<C::Context>,
        input_size: usize,
        output_size: usize,
        graph_seed: [u8; 32],
    ) -> Self {
        Self::setup_with_options(
            circuit,
            ring_gsw,
            input_size,
            output_size,
            graph_seed,
            GoldreichGraphGeneration::default(),
        )
    }

    /// Like [`GoldreichFhePrg::setup`], but allows callers to enable the optional stricter
    /// duplicate-rejection mode used by [`GoldreichGraphGeneration`].
    pub fn setup_with_options(
        circuit: &mut PolyCircuit<P>,
        ring_gsw: Arc<C::Context>,
        input_size: usize,
        output_size: usize,
        graph_seed: [u8; 32],
        generation: GoldreichGraphGeneration,
    ) -> Self {
        Self::from_public_graph(
            circuit,
            ring_gsw,
            GoldreichGraph::generate(input_size, output_size, graph_seed, generation),
        )
    }

    pub fn setup_range(
        circuit: &mut PolyCircuit<P>,
        ring_gsw: Arc<C::Context>,
        input_size: usize,
        conceptual_output_size: usize,
        range_start: usize,
        range_len: usize,
        graph_seed: [u8; 32],
    ) -> Self {
        Self::setup_range_with_options(
            circuit,
            ring_gsw,
            input_size,
            conceptual_output_size,
            range_start,
            range_len,
            graph_seed,
            GoldreichGraphGeneration::default(),
        )
    }

    pub fn setup_range_with_options(
        circuit: &mut PolyCircuit<P>,
        ring_gsw: Arc<C::Context>,
        input_size: usize,
        conceptual_output_size: usize,
        range_start: usize,
        range_len: usize,
        graph_seed: [u8; 32],
        generation: GoldreichGraphGeneration,
    ) -> Self {
        Self::from_public_graph(
            circuit,
            ring_gsw,
            GoldreichGraph::generate_range(
                input_size,
                conceptual_output_size,
                range_start,
                range_len,
                graph_seed,
                generation,
            ),
        )
    }

    pub fn setup_full_domain_range(
        circuit: &mut PolyCircuit<P>,
        ring_gsw: Arc<C::Context>,
        input_size: usize,
        conceptual_output_size: usize,
        range_start: usize,
        range_len: usize,
        graph_seed: [u8; 32],
    ) -> Self {
        Self::setup_full_domain_range_with_options(
            circuit,
            ring_gsw,
            input_size,
            conceptual_output_size,
            range_start,
            range_len,
            graph_seed,
            GoldreichGraphGeneration::default(),
        )
    }

    pub fn setup_full_domain_range_with_options(
        circuit: &mut PolyCircuit<P>,
        ring_gsw: Arc<C::Context>,
        input_size: usize,
        conceptual_output_size: usize,
        range_start: usize,
        range_len: usize,
        graph_seed: [u8; 32],
        generation: GoldreichGraphGeneration,
    ) -> Self {
        Self::from_public_graph(
            circuit,
            ring_gsw,
            GoldreichGraph::generate_full_domain_range(
                input_size,
                conceptual_output_size,
                range_start,
                range_len,
                graph_seed,
                generation,
            ),
        )
    }

    /// Builds the PRG from an already validated public graph instead of generating one from a
    /// `graph_seed`.
    pub fn from_public_graph(
        _circuit: &mut PolyCircuit<P>,
        ring_gsw: Arc<C::Context>,
        public_graph: GoldreichGraph,
    ) -> Self {
        let input_size = public_graph.input_size;
        let output_size = public_graph.output_size();
        assert_goldreich_output_bound(input_size, output_size, "explicit Goldreich graph");
        Self { ring_gsw, input_size, output_size, public_graph }
    }

    fn validate_input_bits(&self, input_bits: &[C]) {
        assert_eq!(
            input_bits.len(),
            self.input_size,
            "Goldreich PRG expects {} encrypted input bits but received {}",
            self.input_size,
            input_bits.len()
        );
        for (idx, bit) in input_bits.iter().enumerate() {
            assert!(
                Arc::ptr_eq(bit.context(), &self.ring_gsw),
                "Goldreich PRG input bit {} must share the GoldreichFhePrg RingGswContext",
                idx
            );
        }
    }

    fn evaluate_uniform_with_graph(
        &self,
        graph: &GoldreichGraph,
        input_bits: &[C],
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<C> {
        debug_assert_eq!(graph.input_size, self.input_size);
        debug_assert_eq!(graph.output_size(), self.output_size);
        graph
            .edges
            .iter()
            .map(|edge| {
                let and_term =
                    input_bits[edge.and_inputs[0]].and(&input_bits[edge.and_inputs[1]], circuit);
                reduce_ciphertext_terms_pairwise(
                    vec![
                        input_bits[edge.xor_inputs[0]].clone(),
                        input_bits[edge.xor_inputs[1]].clone(),
                        input_bits[edge.xor_inputs[2]].clone(),
                        and_term,
                    ],
                    circuit,
                    |lhs: &C, rhs: &C, circuit| lhs.xor(rhs, circuit),
                )
            })
            .collect::<Vec<_>>()
    }

    /// Homomorphically evaluates all TSA predicate edges on encrypted input bits and returns
    /// uniform Goldreich output bits.
    ///
    /// For one edge the logical structure is:
    ///
    /// ```text
    /// t_and = x[d] AND x[e]
    /// y     = XOR_tree([x[a], x[b], x[c], t_and])
    /// ```
    ///
    /// The XOR reduction is assembled as a balanced pairwise tree to minimize depth growth in the
    /// repository's concrete Ring-GSW implementation.
    pub fn evaluate_uniform(&self, input_bits: &[C], circuit: &mut PolyCircuit<P>) -> Vec<C> {
        self.validate_input_bits(input_bits);
        let outputs = self.evaluate_uniform_with_graph(&self.public_graph, input_bits, circuit);
        debug!(
            "Goldreich PRG uniform evaluation produced {} output bits: input_size={}, output_size={}",
            outputs.len(),
            self.input_size,
            self.output_size,
        );
        outputs
    }
}

/// Adds a nonempty list of scalar wires with a balanced ordinary-addition tree.
pub(crate) fn sum_gate_ids<P: Poly>(circuit: &mut PolyCircuit<P>, values: &[GateId]) -> GateId {
    assert!(!values.is_empty(), "at least one gate is required");
    let mut layer = values.to_vec();
    while layer.len() > 1 {
        let mut next = Vec::with_capacity(layer.len().div_ceil(2));
        let mut chunks = layer.chunks_exact(2);
        for pair in &mut chunks {
            next.push(circuit.add_gate(pair[0], pair[1]).as_single_wire());
        }
        if let Some(&carry) = chunks.remainder().first() {
            next.push(carry);
        }
        layer = next;
    }
    layer[0]
}

/// Homomorphically evaluates a selected interval of a conceptual Goldreich PRG output.
///
/// This helper centralizes the common `setup_range` followed by `evaluate_uniform` pattern used by
/// callers that only need a contiguous subset of a larger conceptual Goldreich output stream.  The
/// returned ciphertexts stay encrypted; callers that need a decoded scalar should evaluate this
/// circuit first, then feed the resulting ciphertexts into
/// [`decrypt_bit_decomposed_scalar_outputs`] in a separate runtime-safe decrypt circuit.
pub fn evaluate_goldreich_uniform_range<P, A>(
    circuit: &mut PolyCircuit<P>,
    ring_gsw: Arc<RingGswContext<P, A>>,
    encrypted_seeds: &[RingGswCiphertext<P, A>],
    conceptual_output_bits: usize,
    range_start: usize,
    range_len: usize,
    graph_seed: [u8; 32],
) -> Vec<RingGswCiphertext<P, A>>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
{
    let goldreich = GoldreichFhePrg::setup_range(
        circuit,
        ring_gsw,
        encrypted_seeds.len(),
        conceptual_output_bits,
        range_start,
        range_len,
        graph_seed,
    );
    goldreich.evaluate_uniform(encrypted_seeds, circuit)
}

/// Homomorphically evaluates a selected interval using the full-range graph domain.
///
/// Unlike [`evaluate_goldreich_uniform_range`], this helper produces the same public edges as
/// evaluating the full conceptual range and slicing `[range_start, range_start + range_len)`. It
/// avoids retaining the full output vector, but it still replays graph generation up to the end of
/// the requested range so persisted public-key artifacts and online encodings stay aligned.
pub fn evaluate_goldreich_uniform_full_domain_range<P, A>(
    circuit: &mut PolyCircuit<P>,
    ring_gsw: Arc<RingGswContext<P, A>>,
    encrypted_seeds: &[RingGswCiphertext<P, A>],
    conceptual_output_bits: usize,
    range_start: usize,
    range_len: usize,
    graph_seed: [u8; 32],
) -> Vec<RingGswCiphertext<P, A>>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
{
    let goldreich = GoldreichFhePrg::setup_full_domain_range(
        circuit,
        ring_gsw,
        encrypted_seeds.len(),
        conceptual_output_bits,
        range_start,
        range_len,
        graph_seed,
    );
    goldreich.evaluate_uniform(encrypted_seeds, circuit)
}

/// Decrypts encrypted Boolean bits and returns their binary linear recomposition.
///
/// Bit `j` is decrypted with `plaintext_moduli[j]`.  The Ring-GSW plaintext modulus selection is
/// responsible for making that decrypt contribute the intended binary weight, so the circuit only
/// needs to add the decrypted terms.  This is the scalar counterpart of the bit-decomposed
/// polynomial mask decode used by noise refresh.
pub fn decrypt_bit_decomposed_scalar_outputs<P, A, M>(
    circuit: &mut PolyCircuit<P>,
    encrypted_bits: &[RingGswCiphertext<P, A>],
    decryption_key: GateId,
    plaintext_moduli: &[BigUint],
) -> GateId
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
{
    assert!(!encrypted_bits.is_empty(), "at least one encrypted bit is required");
    assert_eq!(
        encrypted_bits.len(),
        plaintext_moduli.len(),
        "encrypted scalar bit count must match plaintext modulus count"
    );
    assert!(
        plaintext_moduli.iter().all(|modulus| !modulus.is_zero()),
        "all bit plaintext moduli must be positive"
    );
    let bit_terms = encrypted_bits
        .iter()
        .zip(plaintext_moduli.iter())
        .map(|(encrypted_bit, plaintext_modulus)| {
            encrypted_bit
                .decrypt::<M>(decryption_key, plaintext_modulus.clone(), circuit)
                .add_in_circuit(circuit)
        })
        .collect::<Vec<_>>();
    sum_gate_ids(circuit, &bit_terms)
}

/// Fixed-`n` CBD-style error evaluator built from setup-time Goldreich uniform samplers.
///
/// The wrapped [`GoldreichFhePrg`] remains responsible for evaluating one public Goldreich graph
/// into uniform output bits. This wrapper fixes the CBD sample count `n` at setup time, derives
/// `2n` distinct Goldreich graphs from a public seed, registers one reusable CBD coefficient
/// sub-circuit for that `n`, and then evaluates one centered-binomial-style error ciphertext per
/// output position.
pub struct GoldreichFheCbdPrg<P: Poly, C: BooleanCiphertext<P>> {
    pub uniform_prg: GoldreichFhePrg<P, C>,
    pub cbd_n: usize,
    uniform_graphs: Vec<GoldreichGraph>,
    cbd_prf_sub_circuit_id: usize,
    cbd_output_templates: Vec<C>,
}

impl<P: Poly, C: BooleanCiphertext<P>> GoldreichFheCbdPrg<P, C> {
    pub fn uniform_graphs(&self) -> &[GoldreichGraph] {
        &self.uniform_graphs
    }
}

impl<P: Poly + 'static, C> GoldreichFheCbdPrg<P, C>
where
    C: BooleanCiphertext<P>,
{
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        ring_gsw: Arc<C::Context>,
        input_size: usize,
        output_size: usize,
        graph_seed: [u8; 32],
        cbd_n: usize,
    ) -> Self {
        Self::setup_with_options(
            circuit,
            ring_gsw,
            input_size,
            output_size,
            graph_seed,
            cbd_n,
            GoldreichGraphGeneration::default(),
        )
    }

    pub fn setup_with_options(
        circuit: &mut PolyCircuit<P>,
        ring_gsw: Arc<C::Context>,
        input_size: usize,
        output_size: usize,
        graph_seed: [u8; 32],
        cbd_n: usize,
        generation: GoldreichGraphGeneration,
    ) -> Self {
        assert!(cbd_n > 0, "Goldreich CBD evaluator requires cbd_n > 0");
        assert_goldreich_output_bound(
            input_size,
            goldreich_cbd_uniform_output_bits(output_size, cbd_n),
            "Goldreich CBD evaluator",
        );
        let uniform_prg = GoldreichFhePrg::setup_with_options(
            circuit,
            ring_gsw,
            input_size,
            output_size,
            graph_seed,
            generation,
        );
        let uniform_graphs = derive_distinct_goldreich_graphs(
            input_size,
            output_size,
            graph_seed,
            generation,
            2 * cbd_n,
        );
        let (cbd_prf_sub_circuit, cbd_output_templates) =
            goldreich_cbd_prf_sub_circuit(&uniform_prg, &uniform_graphs, cbd_n, circuit);
        let cbd_prf_sub_circuit_id = circuit.register_sub_circuit(cbd_prf_sub_circuit);
        Self { uniform_prg, cbd_n, uniform_graphs, cbd_prf_sub_circuit_id, cbd_output_templates }
    }

    pub fn setup_range(
        circuit: &mut PolyCircuit<P>,
        ring_gsw: Arc<C::Context>,
        input_size: usize,
        conceptual_output_size: usize,
        range_start: usize,
        range_len: usize,
        graph_seed: [u8; 32],
        cbd_n: usize,
    ) -> Self {
        Self::setup_range_with_options(
            circuit,
            ring_gsw,
            input_size,
            conceptual_output_size,
            range_start,
            range_len,
            graph_seed,
            cbd_n,
            GoldreichGraphGeneration::default(),
        )
    }

    pub fn setup_range_with_options(
        circuit: &mut PolyCircuit<P>,
        ring_gsw: Arc<C::Context>,
        input_size: usize,
        conceptual_output_size: usize,
        range_start: usize,
        range_len: usize,
        graph_seed: [u8; 32],
        cbd_n: usize,
        generation: GoldreichGraphGeneration,
    ) -> Self {
        assert!(cbd_n > 0, "Goldreich CBD evaluator requires cbd_n > 0");
        assert_goldreich_output_bound(
            input_size,
            goldreich_cbd_uniform_output_bits(conceptual_output_size, cbd_n),
            "Goldreich CBD range evaluator",
        );
        let uniform_prg = GoldreichFhePrg::setup_range_with_options(
            circuit,
            ring_gsw,
            input_size,
            conceptual_output_size,
            range_start,
            range_len,
            graph_seed,
            generation,
        );
        let uniform_graphs = derive_distinct_goldreich_graph_ranges(
            input_size,
            conceptual_output_size,
            range_start,
            range_len,
            graph_seed,
            generation,
            2 * cbd_n,
        );
        let (cbd_prf_sub_circuit, cbd_output_templates) =
            goldreich_cbd_prf_sub_circuit(&uniform_prg, &uniform_graphs, cbd_n, circuit);
        let cbd_prf_sub_circuit_id = circuit.register_sub_circuit(cbd_prf_sub_circuit);
        Self { uniform_prg, cbd_n, uniform_graphs, cbd_prf_sub_circuit_id, cbd_output_templates }
    }

    pub fn evaluate_cbd_prf(&self, input_bits: &[C], circuit: &mut PolyCircuit<P>) -> Vec<C> {
        self.uniform_prg.validate_input_bits(input_bits);
        let mut cbd_inputs = Vec::with_capacity(input_bits.len());
        for input_bit in input_bits {
            cbd_inputs.extend(input_bit.sub_circuit_wires());
        }
        let outputs = circuit.call_sub_circuit(self.cbd_prf_sub_circuit_id, &cbd_inputs);
        let mut next_output_start = 0usize;
        self.cbd_output_templates
            .iter()
            .map(|template| {
                let output_gate_count =
                    template.sub_circuit_wires().into_iter().map(BatchedWire::len).sum::<usize>();
                let next_output_end = next_output_start + output_gate_count;
                let output = C::from_sub_circuit_outputs(
                    template,
                    &outputs[next_output_start..next_output_end],
                );
                next_output_start = next_output_end;
                output
            })
            .collect::<Vec<_>>()
    }
}

fn validate_graph_dimensions(input_size: usize, output_size: usize) {
    assert!(input_size >= 5, "Goldreich graph input_size must be at least 5");
    assert!(output_size > 0, "Goldreich graph output_size must be positive");
}

fn goldreich_cbd_uniform_output_bits(output_size: usize, cbd_n: usize) -> usize {
    output_size
        .checked_mul(2)
        .and_then(|count| count.checked_mul(cbd_n))
        .expect("Goldreich CBD uniform output size overflow")
}

fn sample_next_unique_edge(
    input_size: usize,
    stream: &mut GraphSeedStream,
    seen_role_keys: &mut HashSet<GoldreichEdgeKey>,
    seen_vertex_sets: &mut Option<HashSet<[usize; 5]>>,
) -> GoldreichEdge {
    loop {
        let mut sampled = Vec::with_capacity(5);
        while sampled.len() < 5 {
            let candidate = stream.sample_below(input_size);
            if sampled.contains(&candidate) {
                continue;
            }
            sampled.push(candidate);
        }

        let edge =
            GoldreichEdge::new([sampled[0], sampled[1], sampled[2]], [sampled[3], sampled[4]]);
        let role_aware_key = edge.role_aware_key();
        if seen_role_keys.contains(&role_aware_key) {
            continue;
        }
        if let Some(seen_vertex_sets) = seen_vertex_sets.as_mut() {
            let same_vertex_set_key = edge.same_vertex_set_key();
            if seen_vertex_sets.contains(&same_vertex_set_key) {
                continue;
            }
            seen_vertex_sets.insert(same_vertex_set_key);
        }
        seen_role_keys.insert(role_aware_key);
        return edge;
    }
}

fn derive_range_graph_seed(
    graph_seed: [u8; 32],
    conceptual_output_size: usize,
    range_start: usize,
    range_len: usize,
) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(b"GoldreichGraphRange/v1");
    hasher.update(graph_seed);
    hasher.update((conceptual_output_size as u128).to_le_bytes());
    hasher.update((range_start as u128).to_le_bytes());
    hasher.update((range_len as u128).to_le_bytes());
    let digest = hasher.finalize();
    let mut derived = [0u8; 32];
    derived.copy_from_slice(digest.as_ref());
    derived
}

fn binomial(n: usize, k: usize) -> u128 {
    assert!(k <= n, "binomial requires k <= n");
    let k = k.min(n - k);
    let mut numerator = 1u128;
    let mut denominator = 1u128;
    for i in 0..k {
        numerator *= (n - i) as u128;
        denominator *= (i + 1) as u128;
    }
    numerator / denominator
}

fn all_distinct(values: &[usize]) -> bool {
    for left in 0..values.len() {
        for right in left + 1..values.len() {
            if values[left] == values[right] {
                return false;
            }
        }
    }
    true
}

fn reduce_ciphertext_terms_pairwise<P, C, F>(
    mut current_layer: Vec<C>,
    circuit: &mut PolyCircuit<P>,
    mut combine: F,
) -> C
where
    P: Poly + 'static,
    C: BooleanCiphertext<P>,
    F: FnMut(&C, &C, &mut PolyCircuit<P>) -> C,
{
    assert!(!current_layer.is_empty(), "pairwise reduction requires at least one ciphertext term");
    while current_layer.len() > 1 {
        let mut next_layer = Vec::with_capacity(current_layer.len().div_ceil(2));
        let mut iter = current_layer.into_iter();
        while let Some(left) = iter.next() {
            if let Some(right) = iter.next() {
                next_layer.push(combine(&left, &right, circuit));
            } else {
                next_layer.push(left);
            }
        }
        current_layer = next_layer;
    }
    current_layer.pop().expect("pairwise reduction must leave one term")
}

fn goldreich_cbd_prf_sub_circuit<P, C>(
    uniform_prg: &GoldreichFhePrg<P, C>,
    uniform_graphs: &[GoldreichGraph],
    cbd_n: usize,
    source_circuit: &PolyCircuit<P>,
) -> (PolyCircuit<P>, Vec<C>)
where
    P: Poly + 'static,
    C: BooleanCiphertext<P>,
{
    assert!(cbd_n > 0, "Goldreich CBD coefficient sub-circuit requires cbd_n > 0");
    assert_eq!(
        uniform_graphs.len(),
        2 * cbd_n,
        "Goldreich CBD sub-circuit requires exactly 2 * cbd_n distinct uniform graphs"
    );
    let mut circuit = source_circuit.fresh_sub_circuit();
    let inputs = (0..uniform_prg.input_size)
        .map(|_| C::sub_circuit_input(Arc::clone(&uniform_prg.ring_gsw), &mut circuit))
        .collect::<Vec<_>>();
    let uniform_samples = uniform_graphs
        .iter()
        .map(|graph| uniform_prg.evaluate_uniform_with_graph(graph, &inputs, &mut circuit))
        .collect::<Vec<_>>();
    let outputs = (0..uniform_prg.output_size)
        .map(|output_idx| {
            let positive = reduce_ciphertext_terms_pairwise(
                uniform_samples[..cbd_n]
                    .iter()
                    .map(|sample| sample[output_idx].clone())
                    .collect::<Vec<_>>(),
                &mut circuit,
                |lhs, rhs, circuit| lhs.add(rhs, circuit),
            );
            let negative = reduce_ciphertext_terms_pairwise(
                uniform_samples[cbd_n..]
                    .iter()
                    .map(|sample| sample[output_idx].clone())
                    .collect::<Vec<_>>(),
                &mut circuit,
                |lhs, rhs, circuit| lhs.add(rhs, circuit),
            );
            positive.sub(&negative, &mut circuit)
        })
        .collect::<Vec<_>>();
    let flat_outputs =
        outputs.iter().flat_map(|output| output.sub_circuit_wires()).collect::<Vec<_>>();
    circuit.output(flat_outputs);
    (circuit, outputs)
}

fn derive_graph_seed(base_seed: [u8; 32], counter: u64) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(b"GoldreichCBDGraph/v1");
    hasher.update(base_seed);
    hasher.update(counter.to_le_bytes());
    let digest = hasher.finalize();
    let mut derived = [0u8; 32];
    derived.copy_from_slice(digest.as_ref());
    derived
}

fn same_graph_structure(lhs: &GoldreichGraph, rhs: &GoldreichGraph) -> bool {
    lhs.input_size == rhs.input_size && lhs.edges == rhs.edges && lhs.generation == rhs.generation
}

fn derive_distinct_goldreich_graphs(
    input_size: usize,
    output_size: usize,
    graph_seed: [u8; 32],
    generation: GoldreichGraphGeneration,
    sample_count: usize,
) -> Vec<GoldreichGraph> {
    let mut graphs = Vec::with_capacity(sample_count);
    let mut counter = 0u64;
    while graphs.len() < sample_count {
        let candidate_seed = derive_graph_seed(graph_seed, counter);
        let candidate =
            GoldreichGraph::generate(input_size, output_size, candidate_seed, generation);
        counter = counter.wrapping_add(1);
        if graphs.iter().any(|existing| same_graph_structure(existing, &candidate)) {
            continue;
        }
        graphs.push(candidate);
    }
    graphs
}

fn derive_distinct_goldreich_graph_ranges(
    input_size: usize,
    conceptual_output_size: usize,
    range_start: usize,
    range_len: usize,
    graph_seed: [u8; 32],
    generation: GoldreichGraphGeneration,
    sample_count: usize,
) -> Vec<GoldreichGraph> {
    let mut graphs = Vec::with_capacity(sample_count);
    let mut counter = 0u64;
    while graphs.len() < sample_count {
        let candidate_seed = derive_graph_seed(graph_seed, counter);
        let candidate = GoldreichGraph::generate_range(
            input_size,
            conceptual_output_size,
            range_start,
            range_len,
            candidate_seed,
            generation,
        );
        counter = counter.wrapping_add(1);
        if graphs.iter().any(|existing| same_graph_structure(existing, &candidate)) {
            continue;
        }
        graphs.push(candidate);
    }
    graphs
}

#[cfg(test)]
fn evaluate_plaintext_goldreich(graph: &GoldreichGraph, input_bits: &[u64]) -> Vec<u64> {
    assert_eq!(
        input_bits.len(),
        graph.input_size,
        "Goldreich plaintext evaluation expects {} input bits but received {}",
        graph.input_size,
        input_bits.len()
    );
    assert!(
        input_bits.iter().all(|bit| *bit <= 1),
        "Goldreich plaintext evaluation expects only Boolean input bits"
    );

    graph
        .edges
        .par_iter()
        .map(|edge| {
            input_bits[edge.xor_inputs[0]] ^
                input_bits[edge.xor_inputs[1]] ^
                input_bits[edge.xor_inputs[2]] ^
                (input_bits[edge.and_inputs[0]] & input_bits[edge.and_inputs[1]])
        })
        .collect()
}

#[derive(Debug, Clone)]
struct GraphSeedStream {
    seed: [u8; 32],
    block_counter: u64,
    block: [u8; 32],
    next_offset: usize,
}

impl GraphSeedStream {
    fn new(seed: [u8; 32]) -> Self {
        Self { seed, block_counter: 0, block: [0u8; 32], next_offset: 32 }
    }

    fn sample_below(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "Goldreich graph sampler upper bound must be positive");
        let upper = upper as u128;
        let bound = ((u128::from(u64::MAX) + 1) / upper) * upper;
        loop {
            let candidate = u128::from(self.next_u64());
            if candidate < bound {
                return (candidate % upper) as usize;
            }
        }
    }

    fn next_u64(&mut self) -> u64 {
        if self.next_offset + 8 > self.block.len() {
            self.refill_block();
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.block[self.next_offset..self.next_offset + 8]);
        self.next_offset += 8;
        u64::from_le_bytes(bytes)
    }

    fn refill_block(&mut self) {
        let mut hasher = Keccak256::new();
        hasher.update(b"GoldreichGraph/v1");
        hasher.update(self.seed);
        hasher.update(self.block_counter.to_le_bytes());
        let digest = hasher.finalize();
        self.block.copy_from_slice(digest.as_ref());
        self.block_counter = self.block_counter.wrapping_add(1);
        self.next_offset = 0;
    }
}

#[cfg(test)]
mod graph_tests {
    use super::*;
    use std::collections::HashSet;

    const SEED: [u8; 32] = [0x5a; 32];

    #[test]
    fn output_bound_uses_strict_seven_over_five() {
        assert!(goldreich_output_bound_holds(5, 9));
        assert!(!goldreich_output_bound_holds(5, 10));
        assert_eq!(minimum_goldreich_input_size(9), 5);
        assert_eq!(minimum_goldreich_input_size(10), 6);
    }

    #[test]
    fn deterministic_range_generation_preserves_coordinates() {
        let options = GoldreichGraphGeneration::default();
        let first = GoldreichGraph::generate_range(10, 24, 3, 4, SEED, options);
        let repeated = GoldreichGraph::generate_range(10, 24, 3, 4, SEED, options);
        let later = GoldreichGraph::generate_range(10, 24, 15, 4, SEED, options);
        assert_eq!(first, repeated);
        assert_eq!(first.edges.len(), 4);
        assert_eq!(later.edges.len(), 4);
        assert_ne!(first.edges, later.edges);
    }

    #[test]
    fn full_domain_ranges_match_the_full_graph() {
        let options = GoldreichGraphGeneration::default();
        let output_size = 32;
        let full_seed = derive_range_graph_seed(SEED, output_size, 0, output_size);
        let full = GoldreichGraph::generate(12, output_size, full_seed, options);
        let range =
            GoldreichGraph::generate_full_domain_range(12, output_size, 11, 7, SEED, options);
        assert_eq!(range.edges, full.edges[11..18]);
    }

    #[test]
    fn generated_edges_are_distinct_and_role_unique() {
        let graph = GoldreichGraph::generate(10, 10, SEED, GoldreichGraphGeneration::default());
        let mut roles = HashSet::new();
        for edge in &graph.edges {
            assert!(all_distinct(&edge.all_inputs()));
            assert!(roles.insert(edge.role_aware_key()));
        }
    }

    #[test]
    fn strict_generation_rejects_vertex_set_reuse() {
        let graph = GoldreichGraph::generate(
            9,
            8,
            SEED,
            GoldreichGraphGeneration { reject_same_vertex_set: true },
        );
        let mut sets = HashSet::new();
        assert!(graph.edges.iter().all(|edge| sets.insert(edge.same_vertex_set_key())));
    }

    #[test]
    fn plaintext_example_matches_the_tsa_predicate() {
        let graph = GoldreichGraph::from_edges(
            5,
            vec![GoldreichEdge::new([0, 1, 2], [3, 4]), GoldreichEdge::new([0, 2, 4], [1, 3])],
            GoldreichGraphGeneration::default(),
        );
        assert_eq!(evaluate_plaintext_goldreich(&graph, &[1, 0, 1, 1, 1]), vec![1, 1]);
    }

    #[test]
    #[should_panic(expected = "violates Goldreich PRG safety bound")]
    fn unsafe_conceptual_output_is_rejected() {
        let _ =
            GoldreichGraph::generate_range(5, 10, 0, 1, SEED, GoldreichGraphGeneration::default());
    }
}
