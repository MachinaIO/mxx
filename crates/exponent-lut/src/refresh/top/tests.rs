use super::*;
use crate::{
    ExpLayout,
    decoder::decoding::fixture::{keys, profile},
    program::{AttributeKnowledge, LutOutputRepresentation, LutTable},
};
use std::collections::BTreeSet;

/// One private input mapped by an exponent LUT (`X^e -> X^(e+1)` on a width
/// of two), and the LUT output; with a coefficient LUT instead when
/// `coefficient`.
pub(crate) fn lut_source(coefficient: bool) -> (ExponentLutProgram, WireId) {
    let mut builder = ExponentLutProgramBuilder::new();
    let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
    let layout = ExpLayout { stride_log2: 0, width_log2: 1 };
    let representation = if coefficient {
        LutOutputRepresentation::Coefficient
    } else {
        LutOutputRepresentation::Exponent(layout)
    };
    let table = builder
        .register_lut(LutTable::unary(2, layout, 2, vec![1, 0], representation).unwrap())
        .unwrap();
    let output = builder.lut(Operand::Scalar(input), table).unwrap();
    (builder.finish(vec![Operand::Scalar(output)]).unwrap(), output)
}

fn layouts(count: usize) -> Vec<PbcPublicLayout> {
    keys(count).into_iter().map(|key| key.public_layout).collect()
}

/// Sixteen 30-bit NTT primes of the ring of dimension eight: the
/// reference's three level-0 primes (level 1 keeps two) and thirteen
/// extension primes for the children's chains.
pub(crate) fn primes() -> Vec<u64> {
    mxx_backends::poly::dcrt::params::DCRTPolyParams::new(8, 16, 30, 15, None, None)
        .moduli()
        .to_vec()
}

/// A reference over the first three [`primes`] (level 1 keeps two).
pub(crate) fn reference() -> TopReference {
    let primes = primes();
    TopReference {
        top: primes[..3].to_vec(),
        next: primes[..2].to_vec(),
        ring_dimension: 8,
        base_bits: 15,
    }
}

/// The refresh of every slot of [`reference`].
pub(crate) fn declaration(
    source: &ExponentLutProgram,
    target: WireId,
    depths: Vec<usize>,
) -> Result<TopRefreshDeclaration, RefreshError> {
    declaration_of(source, target, depths, vec![0, 1, 2])
}

fn declaration_of(
    source: &ExponentLutProgram,
    target: WireId,
    depths: Vec<usize>,
    slots: Vec<usize>,
) -> Result<TopRefreshDeclaration, RefreshError> {
    let n = profile().ring_dimension().unwrap();
    let columns = reference().columns();
    let mut smudge = |_: usize, _: usize, _: usize| layouts(n);
    TopRefreshDeclaration::new(
        source,
        target,
        profile(),
        1,
        reference(),
        depths,
        slots,
        primes()[3..].to_vec(),
        1,
        TopRefreshLayouts {
            mask: layouts(n),
            fresh: [layouts(columns * n), layouts(columns * n)],
            smudge: &mut smudge,
        },
        b"top".to_vec(),
    )
}

/// One chain per prime of `Q_0`, two parts per chain decoding only their
/// output prime over the reference's `w_0` columns, one label for every part,
/// distinct chain namespaces and an `A'` component index per component, pool orders that
/// put the output prime first, and the output prime's position in the
/// canonical tower order of the decoding level.
#[test]
fn top_refresh_lays_out_one_chain_per_prime() {
    let (source, output) = lut_source(false);
    let top = declaration(&source, output, vec![3, 2, 2]).unwrap();
    let columns = reference().columns();
    assert_eq!(columns, 4, "two 30-bit primes in base 2^15");
    assert_eq!((top.chains(), top.parts().len(), top.level()), (3, 6, 1));
    for chain in 0..3 {
        let pool = top.pool_order(chain).unwrap();
        assert_eq!(pool[0], reference().top[chain]);
        assert_eq!(pool.len(), 16);
        // The decoding level keeps two primes: chain 0 keeps `[q_0, q_1]`,
        // chains 1 and 2 keep `q_0` and their own prime after it.
        let position = chain.min(1);
        for component in 0..2 {
            let part = top.part(chain, component).unwrap();
            let details = part.top().unwrap();
            assert_eq!((details.component(), details.output_prime()), (component, position));
            assert_eq!(part.component(), component);
            assert_eq!(part.decoded_primes(), vec![position]);
            assert_eq!(part.decoders().len(), columns);
            assert!(
                part.decoder(position, columns - 1).is_some() &&
                    part.decoder(1 - position, 0).is_none()
            );
            assert_eq!(part.label([3; 32], b"nu").unwrap(), top.label([3; 32], b"nu").unwrap());
        }
        // The components share the matrix tag; `A'` separates them by the component index.
        assert_eq!(
            top.part(chain, 0).unwrap().matrix_tag(),
            top.part(chain, 1).unwrap().matrix_tag()
        );
        assert_ne!(
            top.part(chain, 0).unwrap().component(),
            top.part(chain, 1).unwrap().component()
        );
    }
    assert_eq!(top.part(0, 0).unwrap().matrix_tag(), top.part(2, 0).unwrap().matrix_tag());
    let namespaces = (0..3).map(|chain| top.chain_namespace(chain)).collect::<BTreeSet<_>>();
    assert_eq!(namespaces.len(), 3);
    assert!(!namespaces.contains(&top.matrix_namespace()));
}

/// A refresh of some slots has one chain per refreshed slot, each decoding its slot's prime at
/// that prime's canonical position; empty, unordered, repeated, and out-of-range slots are
/// rejected.
#[test]
fn top_refresh_of_some_slots_lays_out_their_chains() {
    let (source, output) = lut_source(false);
    let top = declaration_of(&source, output, vec![3, 2, 2], vec![2]).unwrap();
    assert_eq!((top.chains(), top.parts().len(), top.slots()), (1, 2, [2].as_slice()));
    assert_eq!(top.output_prime(0), Some(reference().top[2]));
    assert_eq!(top.output_primes(), vec![reference().top[2]]);
    assert_eq!(top.output_prime(1), None);
    assert_eq!(top.pool_order(0).unwrap()[0], reference().top[2]);
    for component in 0..2 {
        // The decoding level keeps `q_0` and the slot's prime after it.
        assert_eq!(top.part(0, component).unwrap().top().unwrap().output_prime(), 1);
    }
    let all = declaration(&source, output, vec![3, 2, 2]).unwrap();
    assert_ne!(top.spec(), all.spec());
    for slots in [vec![], vec![1, 0], vec![1, 1], vec![3]] {
        assert!(
            declaration_of(&source, output, vec![3, 2, 2], slots.clone()).is_err(),
            "{slots:?}"
        );
    }
}

/// Coefficient targets, inputs, and depth profiles that do not match the
/// reference, grow, leave fewer than two decoding primes, or drop every
/// plaintext prime are rejected.
#[test]
fn top_refresh_rejects_unsupported_targets_and_depths() {
    let (source, output) = lut_source(false);
    assert!(declaration(&source, output, vec![3, 2, 2]).is_ok());
    let (coefficient, coefficient_output) = lut_source(true);
    assert!(declaration(&coefficient, coefficient_output, vec![3, 2, 2]).is_err());
    let mut builder = ExponentLutProgramBuilder::new();
    let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
    let identity = builder.finish(vec![Operand::Scalar(input)]).unwrap();
    assert!(declaration(&identity, input, vec![3, 2, 2]).is_err(), "not a LUT output");
    for depths in [vec![2, 2, 2], vec![3, 3, 2], vec![3, 2, 3], vec![3, 1, 1], vec![3, 2, 0]] {
        assert!(declaration(&source, output, depths.clone()).is_err(), "{depths:?}");
    }
}

/// The mask samplers of two chains read the same PRF bit: each returns
/// `+-h` for `h = 2^-1 mod q` of its own output prime `q` (the offset `h`
/// is added in the Program), so `h (1 +- 1) mod q` is the same bit.
#[test]
fn chains_mask_samplers_agree_modulo_their_primes() {
    let (source, output) = lut_source(false);
    let top = declaration(&source, output, vec![3, 2, 2]).unwrap();
    let mask = |chain: usize| top.part(chain, 0).unwrap().top().unwrap().mask().clone();
    let (first, second) = (mask(0), mask(1));
    let primes = [top.output_prime(0).unwrap(), top.output_prime(1).unwrap()];
    let q = profile().q().unwrap();
    let mut ones = 0;
    for z in 0..q {
        let bits = [(&first, primes[0]), (&second, primes[1])].map(|(mask, prime)| {
            let half = (prime as i64 + 1) / 2;
            let value = mask.sampler().sample_at(mask.program(), z).unwrap();
            assert_eq!(value.abs(), half);
            (half + value).rem_euclid(prime as i64)
        });
        assert_eq!(bits[0], bits[1], "z = {z}");
        assert!(bits[0] <= 1);
        ones += bits[0];
    }
    assert!(ones > 0 && ones < q as i64, "both bits occur");
}
