use digest::Digest;
use keccak_asm::Keccak256;
use mxx_gadgets::{
    Poly,
    circuit::PolyCircuit,
    circuit_gadgets::{
        fhe::ring_gsw_nested_rns::{NestedRnsRingGswCiphertext, NestedRnsRingGswContext},
        fhe_prg::goldreich::{
            GoldreichFhePrg, GoldreichGraph, evaluate_goldreich_uniform_full_domain_range,
        },
    },
};
use std::sync::Arc;

pub fn goldreich_round_seed(
    base_seed: [u8; 32],
    domain: &[u8],
    round: usize,
    branch: Option<usize>,
) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(b"mxx/DiamondIo/Goldreich/v1");
    hasher.update(base_seed);
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    hasher.update(round.to_le_bytes());
    match branch {
        Some(branch) => {
            hasher.update([1]);
            hasher.update(branch.to_le_bytes());
        }
        None => hasher.update([0]),
    }
    hasher.finalize().into()
}

/// Builds one contiguous output range while preserving the public graph of the
/// complete conceptual Goldreich stream.  This is the circuit used by both the
/// public-key preprocessing path and the online encoding path, so their wire
/// order cannot drift.
pub(crate) fn build_goldreich_full_domain_range_circuit<P: Poly + 'static>(
    context: Arc<NestedRnsRingGswContext<P>>,
    seed_bits: usize,
    conceptual_output_bits: usize,
    range_start: usize,
    range_len: usize,
    graph_seed: [u8; 32],
) -> PolyCircuit<P> {
    assert!(seed_bits >= 5, "Goldreich requires at least five encrypted seed bits");
    assert!(range_len > 0, "Goldreich output range must be nonempty");
    assert!(
        range_start.checked_add(range_len).is_some_and(|end| end <= conceptual_output_bits),
        "Goldreich output range must fit in the conceptual stream"
    );
    let mut circuit = context.fresh_circuit();
    let encrypted_seed = (0..seed_bits)
        .map(|_| NestedRnsRingGswCiphertext::input(context.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let outputs = evaluate_goldreich_uniform_full_domain_range(
        &mut circuit,
        context,
        &encrypted_seed,
        conceptual_output_bits,
        range_start,
        range_len,
        graph_seed,
    );
    circuit.output(outputs.iter().flat_map(|ciphertext| ciphertext.sub_circuit_wires()));
    circuit
}

/// Builds the private-seed suffix function used by Diamond iO.
///
/// The inputs are encrypted *private seed bits*. Public iO inputs do not feed
/// this circuit directly: they select the preceding PRF/rebase/refresh path.
pub fn build_goldreich_suffix_circuit<P: Poly + 'static>(
    context: Arc<NestedRnsRingGswContext<P>>,
    public_graph: GoldreichGraph,
) -> PolyCircuit<P> {
    let mut circuit = context.fresh_circuit();
    let encrypted_seed = (0..public_graph.input_size)
        .map(|_| NestedRnsRingGswCiphertext::input(context.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let suffix = GoldreichFhePrg::from_public_graph(&mut circuit, context, public_graph)
        .evaluate_uniform(&encrypted_seed, &mut circuit);
    circuit.output(suffix.iter().flat_map(|ciphertext| ciphertext.sub_circuit_wires()));
    circuit
}

#[cfg(test)]
mod tests {
    use super::goldreich_round_seed;
    use mxx_gadgets::circuit_gadgets::fhe_prg::goldreich::{
        GoldreichGraph, GoldreichGraphGeneration,
    };

    #[test]
    fn suffix_graph_is_about_the_private_seed_not_the_public_input() {
        let graph = GoldreichGraph::generate(5, 1, [4; 32], GoldreichGraphGeneration::default());
        assert_eq!(graph.input_size, 5);
        assert_eq!(graph.output_size(), 1);
    }

    #[test]
    fn round_graph_seeds_are_domain_and_branch_separated() {
        let base = [7; 32];
        let round = goldreich_round_seed(base, b"seed-refresh", 1, None);
        assert_ne!(round, goldreich_round_seed(base, b"seed-refresh", 2, None));
        assert_ne!(round, goldreich_round_seed(base, b"noise-refresh", 1, None));
        assert_ne!(
            goldreich_round_seed(base, b"noise-refresh", 1, Some(0)),
            goldreich_round_seed(base, b"noise-refresh", 1, Some(1))
        );
    }
}
