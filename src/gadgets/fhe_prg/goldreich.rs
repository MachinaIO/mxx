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
    circuit::PolyCircuit,
    gadgets::fhe::ring_gsw::{RingGswCiphertext, RingGswContext},
    poly::Poly,
};
use digest::Digest;
use keccak_asm::Keccak256;
use rayon::prelude::*;
use std::{collections::HashSet, sync::Arc};
use tracing::debug;
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
            edges.push(edge);
        }

        Self { input_size, edges, graph_seed: Some(graph_seed), generation }
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
pub struct GoldreichFhePrg<P: Poly> {
    pub ring_gsw: Arc<RingGswContext<P>>,
    pub input_size: usize,
    pub output_size: usize,
    pub public_graph: GoldreichGraph,
}

impl<P: Poly> GoldreichFhePrg<P> {
    /// Returns the fixed public graph used by this PRG instance.
    pub fn graph(&self) -> &GoldreichGraph {
        &self.public_graph
    }
}

impl<P: Poly + 'static> GoldreichFhePrg<P> {
    /// Generates the fixed public graph from a public `graph_seed` and stores it with the
    /// Ring-GSW context.
    pub fn setup(
        ring_gsw: Arc<RingGswContext<P>>,
        input_size: usize,
        output_size: usize,
        graph_seed: [u8; 32],
    ) -> Self {
        Self::setup_with_options(
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
        ring_gsw: Arc<RingGswContext<P>>,
        input_size: usize,
        output_size: usize,
        graph_seed: [u8; 32],
        generation: GoldreichGraphGeneration,
    ) -> Self {
        Self::from_public_graph(
            ring_gsw,
            GoldreichGraph::generate(input_size, output_size, graph_seed, generation),
        )
    }

    /// Builds the PRG from an already validated public graph instead of generating one from a
    /// `graph_seed`.
    pub fn from_public_graph(
        ring_gsw: Arc<RingGswContext<P>>,
        public_graph: GoldreichGraph,
    ) -> Self {
        let input_size = public_graph.input_size;
        let output_size = public_graph.output_size();
        Self { ring_gsw, input_size, output_size, public_graph }
    }

    /// Homomorphically evaluates all TSA predicate edges on encrypted input bits.
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
    pub fn evaluate(
        &self,
        input_bits: &[RingGswCiphertext<P>],
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<RingGswCiphertext<P>> {
        assert_eq!(
            input_bits.len(),
            self.input_size,
            "Goldreich PRG expects {} encrypted input bits but received {}",
            self.input_size,
            input_bits.len()
        );
        for (idx, bit) in input_bits.iter().enumerate() {
            assert!(
                Arc::ptr_eq(&bit.ctx, &self.ring_gsw),
                "Goldreich PRG input bit {} must share the GoldreichFhePrg RingGswContext",
                idx
            );
        }

        let outputs = self
            .public_graph
            .edges
            .iter()
            .map(|edge| {
                let and_term =
                    input_bits[edge.and_inputs[0]].and(&input_bits[edge.and_inputs[1]], circuit);
                reduce_ring_gsw_terms_pairwise(
                    vec![
                        input_bits[edge.xor_inputs[0]].clone(),
                        input_bits[edge.xor_inputs[1]].clone(),
                        input_bits[edge.xor_inputs[2]].clone(),
                        and_term,
                    ],
                    circuit,
                    |lhs, rhs, circuit| lhs.xor(rhs, circuit),
                )
            })
            .collect::<Vec<_>>();
        debug!(
            "Goldreich PRG evaluated {} edges with Ring-GSW ciphertexts: input_size={}, output_size={}",
            outputs.len(),
            self.input_size,
            self.output_size,
        );
        outputs
    }
}

fn validate_graph_dimensions(input_size: usize, output_size: usize) {
    assert!(input_size >= 5, "Goldreich graph input_size must be at least 5");
    assert!(output_size > 0, "Goldreich graph output_size must be positive");
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

fn reduce_ring_gsw_terms_pairwise<P, F>(
    mut current_layer: Vec<RingGswCiphertext<P>>,
    circuit: &mut PolyCircuit<P>,
    mut combine: F,
) -> RingGswCiphertext<P>
where
    P: Poly + 'static,
    F: FnMut(
        &RingGswCiphertext<P>,
        &RingGswCiphertext<P>,
        &mut PolyCircuit<P>,
    ) -> RingGswCiphertext<P>,
{
    assert!(
        !current_layer.is_empty(),
        "pairwise reduction requires at least one RingGswCiphertext term"
    );
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
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        circuit::{PolyCircuit, evaluable::PolyVec},
        gadgets::{
            arith::DEFAULT_MAX_UNREDUCED_MULS,
            fhe::ring_gsw::{
                NativeRingGswCiphertext, RingGswContext, ciphertext_from_outputs,
                ciphertext_inputs_from_native, decrypt_ciphertext, encrypt_plaintext_bit,
                sample_public_key, sample_secret_key,
            },
        },
        lookup::{poly::PolyPltEvaluator, poly_vec::PolyVecPltEvaluator},
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        slot_transfer::PolyVecSlotTransferEvaluator,
    };
    use num_bigint::BigUint;
    use num_traits::ToPrimitive;
    use rand::Rng;
    use std::{collections::HashSet, sync::Arc};
    use tempfile::tempdir;

    const BASE_BITS: u32 = 6;
    const CRT_BITS: usize = 12;
    const ACTIVE_LEVELS: usize = 1;
    const P_MODULI_BITS: usize = 6;
    const SCALE: u64 = 1 << 8;
    const NUM_SLOTS: usize = 2;

    fn create_test_context_with(
        circuit: &mut PolyCircuit<DCRTPoly>,
        ring_dim: u32,
        num_slots: usize,
        active_levels: usize,
        crt_bits: usize,
        p_moduli_bits: usize,
        max_unused_muls: usize,
    ) -> (DCRTPolyParams, Arc<RingGswContext<DCRTPoly>>) {
        let params = DCRTPolyParams::new(ring_dim, active_levels, crt_bits, BASE_BITS);
        let ctx = Arc::new(RingGswContext::setup(
            circuit,
            &params,
            num_slots,
            p_moduli_bits,
            max_unused_muls,
            SCALE,
            Some(active_levels),
            None,
        ));
        (params, ctx)
    }

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<RingGswContext<DCRTPoly>>) {
        create_test_context_with(
            circuit,
            NUM_SLOTS as u32,
            NUM_SLOTS,
            ACTIVE_LEVELS,
            CRT_BITS,
            P_MODULI_BITS,
            DEFAULT_MAX_UNREDUCED_MULS,
        )
    }

    fn sample_hash_key() -> [u8; 32] {
        let mut rng = rand::rng();
        let mut key = [0u8; 32];
        rng.fill(&mut key);
        key
    }

    fn sample_graph_seed() -> [u8; 32] {
        sample_hash_key()
    }

    fn sample_binary_vector(len: usize) -> Vec<u64> {
        let mut rng = rand::rng();
        (0..len).map(|_| rng.random_range(0..2u64)).collect()
    }

    fn eval_outputs(
        params: &DCRTPolyParams,
        circuit: &PolyCircuit<DCRTPoly>,
        inputs: Vec<PolyVec<DCRTPoly>>,
    ) -> Vec<PolyVec<DCRTPoly>> {
        let one = PolyVec::new(vec![DCRTPoly::const_one(params); NUM_SLOTS]);
        let plt_evaluator = PolyVecPltEvaluator { plt_evaluator: PolyPltEvaluator::new() };
        let slot_transfer_evaluator = PolyVecSlotTransferEvaluator::new();
        circuit.eval(
            params,
            one,
            inputs,
            Some(&plt_evaluator),
            Some(&slot_transfer_evaluator),
            None,
        )
    }

    fn rounded_coeffs(
        decrypted: &DCRTPoly,
        plaintext_modulus: u64,
        q_modulus: &BigUint,
    ) -> Vec<u64> {
        let half_q = q_modulus / BigUint::from(2u64);
        decrypted
            .coeffs_biguints()
            .into_par_iter()
            .map(|slot| {
                let rounded = (BigUint::from(plaintext_modulus) * slot + &half_q) / q_modulus;
                rounded.to_u64().expect("rounded plaintext slot must fit in u64")
            })
            .collect()
    }

    fn expected_coeffs(expected: u64) -> Vec<u64> {
        let mut coeffs = vec![0u64; NUM_SLOTS];
        coeffs[0] = expected;
        coeffs
    }

    fn ciphertexts_from_outputs(
        params: &DCRTPolyParams,
        outputs: &[PolyVec<DCRTPoly>],
        width: usize,
    ) -> Vec<NativeRingGswCiphertext> {
        let ciphertext_size = 2 * width;
        assert!(
            outputs.len().is_multiple_of(ciphertext_size),
            "Goldreich Ring-GSW outputs must be an exact multiple of one ciphertext"
        );
        outputs
            .par_chunks(ciphertext_size)
            .map(|chunk| ciphertext_from_outputs(params, chunk, width))
            .collect()
    }

    #[test]
    fn test_goldreich_graph_generation_is_deterministic() {
        let graph_seed = sample_graph_seed();
        let options = GoldreichGraphGeneration::default();
        let lhs = GoldreichGraph::generate(8, 6, graph_seed, options);
        let rhs = GoldreichGraph::generate(8, 6, graph_seed, options);
        assert_eq!(lhs, rhs, "same graph_seed must reproduce the same public graph exactly");
    }

    #[test]
    fn test_goldreich_graph_edges_use_five_distinct_indices() {
        let graph = GoldreichGraph::generate(
            9,
            7,
            sample_graph_seed(),
            GoldreichGraphGeneration::default(),
        );
        for edge in &graph.edges {
            assert!(
                all_distinct(&edge.all_inputs()),
                "every Goldreich edge must use five distinct input indices"
            );
        }
    }

    #[test]
    fn test_goldreich_graph_rejects_role_aware_duplicates() {
        let graph = GoldreichGraph::generate(
            10,
            10,
            sample_graph_seed(),
            GoldreichGraphGeneration::default(),
        );
        let mut seen = HashSet::new();
        for edge in &graph.edges {
            assert!(
                seen.insert(edge.role_aware_key()),
                "generated Goldreich graph must not repeat a role-aware canonical key"
            );
        }
    }

    #[test]
    fn test_goldreich_graph_strict_mode_rejects_same_vertex_set_reuse() {
        let graph = GoldreichGraph::generate(
            9,
            8,
            sample_graph_seed(),
            GoldreichGraphGeneration { reject_same_vertex_set: true },
        );
        let mut seen = HashSet::new();
        for edge in &graph.edges {
            assert!(
                seen.insert(edge.same_vertex_set_key()),
                "strict Goldreich graph mode must not reuse the same 5-set"
            );
        }
    }

    #[test]
    fn test_goldreich_plaintext_small_example_matches_tsa_formula() {
        let graph = GoldreichGraph::from_edges(
            5,
            vec![GoldreichEdge::new([0, 1, 2], [3, 4]), GoldreichEdge::new([0, 2, 4], [1, 3])],
            GoldreichGraphGeneration::default(),
        );
        let input_bits = vec![1u64, 0, 1, 1, 1];
        let output_bits = evaluate_plaintext_goldreich(&graph, &input_bits);
        assert_eq!(
            output_bits,
            vec![1u64, 1u64],
            "hard-coded Goldreich example should match the direct TSA computation"
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_goldreich_ring_gsw_output_decrypts_to_plaintext_reference() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ring_gsw) = create_test_context(&mut circuit);
        let graph_seed = sample_graph_seed();
        let goldreich = GoldreichFhePrg::setup(ring_gsw.clone(), 5, 1, graph_seed);
        let encrypted_inputs = (0..goldreich.input_size)
            .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
            .collect::<Vec<_>>();
        let encrypted_outputs = goldreich.evaluate(&encrypted_inputs, &mut circuit);
        let reconstructed_outputs = encrypted_outputs
            .iter()
            .flat_map(|ciphertext| ciphertext.reconstruct(&mut circuit))
            .collect::<Vec<_>>();
        circuit.output(reconstructed_outputs);

        let plaintext_modulus = 2u64;
        let secret_key = sample_secret_key(&params);
        let public_key_hash_key = sample_hash_key();
        let randomizer_hash_key = sample_hash_key();
        let public_key = sample_public_key(
            &params,
            ring_gsw.width(),
            &secret_key,
            public_key_hash_key,
            b"goldreich_ring_gsw_public_key",
            None,
        );
        let plaintext_inputs = sample_binary_vector(goldreich.input_size);
        let expected_bits = evaluate_plaintext_goldreich(goldreich.graph(), &plaintext_inputs);
        let native_inputs = plaintext_inputs
            .par_iter()
            .enumerate()
            .map(|(idx, bit)| {
                let tag = format!("goldreich_input_bit_{idx}");
                encrypt_plaintext_bit(
                    &params,
                    ring_gsw.nested_rns.as_ref(),
                    &public_key,
                    *bit,
                    randomizer_hash_key,
                    tag.as_bytes(),
                )
            })
            .collect::<Vec<_>>();

        let circuit_inputs = native_inputs
            .iter()
            .flat_map(|ciphertext| {
                ciphertext_inputs_from_native(
                    &params,
                    ring_gsw.nested_rns.as_ref(),
                    ciphertext,
                    0,
                    Some(ring_gsw.active_levels),
                )
            })
            .collect::<Vec<_>>();
        let outputs = eval_outputs(&params, &circuit, circuit_inputs);
        let reconstructed_ciphertexts =
            ciphertexts_from_outputs(&params, &outputs, ring_gsw.width());
        let q_modulus = BigUint::from(ring_gsw.nested_rns.q_moduli()[0]);

        assert_eq!(
            reconstructed_ciphertexts.len(),
            goldreich.output_size,
            "homomorphic Goldreich evaluation must reconstruct one ciphertext per PRG output bit"
        );
        reconstructed_ciphertexts.par_iter().zip(expected_bits.par_iter()).enumerate().for_each(
            |(idx, (ciphertext, expected_bit))| {
                let decrypted = decrypt_ciphertext(
                    &params,
                    ring_gsw.nested_rns.as_ref(),
                    ciphertext,
                    &secret_key,
                    plaintext_modulus,
                );
                assert_eq!(
                    rounded_coeffs(&decrypted, plaintext_modulus, &q_modulus),
                    expected_coeffs(*expected_bit),
                    "Goldreich Ring-GSW output bit {idx} must decrypt to the plaintext TSA result"
                );
            },
        );
    }

    #[sequential_test::sequential]
    #[test]
    #[ignore = "expensive circuit-structure reporting test; run with --ignored --nocapture"]
    fn test_goldreich_ring_gsw_large_circuit_non_free_depth_metrics() {
        let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).try_init();
        let ring_dim = 1u32 << 16;
        let num_slots = 1usize << 16;
        let active_levels = 1usize;
        let crt_bits = 24usize;
        let p_moduli_bits = 7usize;
        let max_unused_muls = 4usize;

        let disk_dir = tempdir().expect("create temp dir for disk-backed sub-circuits");
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        circuit.enable_subcircuits_in_disk(disk_dir.path());
        let (_params, ring_gsw) = create_test_context_with(
            &mut circuit,
            ring_dim,
            num_slots,
            active_levels,
            crt_bits,
            p_moduli_bits,
            max_unused_muls,
        );
        let graph_seed = sample_graph_seed();
        let goldreich = GoldreichFhePrg::setup(ring_gsw.clone(), 5, 1, graph_seed);
        let encrypted_inputs = (0..goldreich.input_size)
            .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
            .collect::<Vec<_>>();
        let encrypted_outputs = goldreich.evaluate(&encrypted_inputs, &mut circuit);
        let reconstructed_outputs = encrypted_outputs
            .iter()
            .flat_map(|ciphertext| ciphertext.reconstruct(&mut circuit))
            .collect::<Vec<_>>();
        circuit.output(reconstructed_outputs);

        println!(
            "goldreich ring_gsw metrics: crt_bits={crt_bits}, active_levels={active_levels}, ring_dim={ring_dim}, num_slots={num_slots}"
        );
        let depth_contributions = circuit.non_free_depth_contributions();
        println!("goldreich ring_gsw non-free depth contributions {:?}", depth_contributions);
        println!("goldreich ring_gsw gate counts {:?}", circuit.count_gates_by_type_vec());
    }
}
