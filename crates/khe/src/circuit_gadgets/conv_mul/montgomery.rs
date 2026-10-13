//! Bit-level negacyclic Ring-GSW products of Montgomery entries with one-bit limbs.
//!
//! A Ring-GSW product row is `sum_h sum_k l_(h,k) G^(-1)(r_h)_k` for left entries `l` and right
//! entries `r_h`. At a q-level `q_i`, the digit `G^(-1)(r_h)_(i,d)` is the polynomial `b_(h,d)`
//! of the `d`-th bits of the regular residues of `r_h`, and the left entry is held in Montgomery
//! form, so the row is `sum_(h,d) l_(h,i,d) b_(h,d) mod (X^N + 1, q_i)`, again in Montgomery form.
//! Because `b_(h,d)` has binary coefficients, every partial product is the AND of one bit of `l`
//! with one coefficient of `b`, and the whole sum needs no modular multiplication.
//!
//! The sum over the `pairs` of `(l, b)` is computed exactly over the integers with an even-odd
//! split. With `Y = X^2`, where `Y^(N/2) = -1`,
//!
//! ```text
//! l b = (l_e b_e + Y l_o b_o)(Y) + X (l_e b_o + l_o b_e)(Y),
//! ```
//!
//! so the sum of dimension `N` calls one sub-circuit of dimension `N / 2` four times, on the
//! even and odd coefficients of every pair. A product of dimension `N` therefore materializes
//! `log2 N + 1` sub-circuit bodies, while its expanded gate count is that of the flat schoolbook
//! product: `N^2 L pairs` ANDs, and full adders that compress the partial products. Every value
//! stays an integer `sum_k 2^k sum(columns[k]) + offset` with at most two bits per column after
//! each compression and a public offset, so the sign of the wrapped term `-Y^(N/2)` costs only
//! the complements of its bits.
//!
//! Each coefficient is then reduced modulo `q_i` by a sub-circuit of the q-level: it adds the
//! offset's residue, replaces each bit of weight `2^k >= 2^L` by the bits of `2^k mod q_i` while
//! this shortens the value, and subtracts `2^j q_i` conditionally for decreasing `j`.

use crate::{
    circuit::PolyCircuit,
    circuit_gadgets::arith::{
        CarryArithPoly, CarryArithPolyContext, Columns, ModularArithmeticContext, MontgomeryPoly,
        MontgomeryPolyContext,
    },
    poly::Poly,
};
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::One;

/// The integer `sum_k 2^k sum(columns[k]) + offset`, whose wires carry bits.
#[derive(Clone)]
struct ColumnSum {
    columns: Columns,
    offset: BigInt,
}

impl ColumnSum {
    fn add(mut self, other: &Self) -> Self {
        if self.columns.len() < other.columns.len() {
            self.columns.resize(other.columns.len(), Vec::new());
        }
        for (column, other_column) in self.columns.iter_mut().zip(&other.columns) {
            column.extend_from_slice(other_column);
        }
        self.offset += &other.offset;
        self
    }

    /// `-self`: a bit `w` of weight `2^k` becomes `(1 - w) 2^k - 2^k`.
    fn negate<P: Poly>(&self, circuit: &mut PolyCircuit<P>) -> Self {
        let mut offset = -self.offset.clone();
        let columns = self
            .columns
            .iter()
            .enumerate()
            .map(|(weight, column)| {
                offset -= BigInt::from(column.len()) << weight;
                column.iter().map(|&bit| circuit.not_gate(bit).as_single_wire()).collect()
            })
            .collect();
        Self { columns, offset }
    }
}

/// The public layout of one output coefficient of a digit-product sub-circuit: the number of
/// output wires of every column, and the offset.
#[derive(Clone)]
struct ColumnLayout {
    heights: Vec<usize>,
    offset: BigInt,
}

/// The sub-circuit computing `sum_p l_p b_p mod X^dimension + 1` exactly over the integers.
///
/// For every pair `p` the inputs are the `dimension` coefficients of `l_p`, `bits` bits each, then
/// the `dimension` bits of `b_p`. The outputs are the columns of every coefficient, as described
/// by `outputs`.
struct DigitProductCircuit<P: Poly> {
    circuit: PolyCircuit<P>,
    outputs: Vec<ColumnLayout>,
}

fn digit_product_circuit<P: Poly + 'static>(
    source: &PolyCircuit<P>,
    carry_ctx: &CarryArithPolyContext<P>,
    dimension: usize,
    pairs: usize,
    bits: usize,
) -> DigitProductCircuit<P> {
    let mut circuit = source.fresh_sub_circuit();
    let inputs = (0..pairs)
        .map(|_| {
            let left = circuit.input(dimension * bits).to_vec();
            let right = circuit.input(dimension).to_vec();
            (left, right)
        })
        .collect::<Vec<_>>();
    let sums = if dimension == 1 {
        let mut columns: Columns = vec![Vec::with_capacity(pairs); bits];
        for (left, right) in &inputs {
            for (weight, &bit) in left.iter().enumerate() {
                columns[weight].push(circuit.and_gate(bit, right[0]).as_single_wire());
            }
        }
        vec![ColumnSum { columns, offset: BigInt::ZERO }]
    } else {
        let half = dimension / 2;
        let child = digit_product_circuit(source, carry_ctx, half, pairs, bits);
        let child_outputs = child.outputs;
        let child_id = circuit.register_sub_circuit(child.circuit);
        let mut call_half = |left_parity: usize, right_parity: usize| {
            let call_inputs = inputs
                .iter()
                .flat_map(|(left, right)| {
                    let left_half = (0..half).flat_map(move |j| {
                        let start = (2 * j + left_parity) * bits;
                        left[start..start + bits].iter().copied()
                    });
                    let right_half = (0..half).map(move |j| right[2 * j + right_parity]);
                    left_half.chain(right_half)
                })
                .collect::<Vec<_>>();
            let outputs = circuit.call_sub_circuit(child_id, call_inputs);
            let mut outputs = outputs.into_iter().map(|output| output.as_single_wire());
            child_outputs
                .iter()
                .map(|layout| ColumnSum {
                    columns: layout
                        .heights
                        .iter()
                        .map(|&height| outputs.by_ref().take(height).collect())
                        .collect(),
                    offset: layout.offset.clone(),
                })
                .collect::<Vec<_>>()
        };
        let even_even = call_half(0, 0);
        let odd_odd = call_half(1, 1);
        let even_odd = call_half(0, 1);
        let odd_even = call_half(1, 0);
        let mut sums = Vec::with_capacity(dimension);
        for i in 0..half {
            // `(Y P)[i] = P[i - 1]`, and `Y^(N/2) = -1` wraps `P[N/2 - 1]` to `-P[N/2 - 1]`.
            let wrapped = if i == 0 {
                odd_odd[half - 1].negate(&mut circuit)
            } else {
                odd_odd[i - 1].clone()
            };
            sums.push(even_even[i].clone().add(&wrapped));
            sums.push(even_odd[i].clone().add(&odd_even[i]));
        }
        sums
    };
    let mut outputs = Vec::with_capacity(dimension);
    let mut output_wires = Vec::new();
    for sum in sums {
        let columns = carry_ctx.compress_columns(&mut circuit, sum.columns);
        outputs.push(ColumnLayout {
            heights: columns.iter().map(Vec::len).collect(),
            offset: sum.offset,
        });
        output_wires.extend(columns.into_iter().flatten());
    }
    circuit.output(output_wires);
    DigitProductCircuit { circuit, outputs }
}

/// The sub-circuit reducing `row0 + row1 + offset` modulo the q-level `q_idx`, for two rows of
/// `width` bits and an offset below `q_i` of `num_limbs` bits. The output is the residue in
/// `num_limbs` bits.
fn reduction_circuit<P: Poly + 'static>(
    source: &PolyCircuit<P>,
    ctx: &MontgomeryPolyContext<P>,
    q_idx: usize,
    width: usize,
) -> PolyCircuit<P> {
    let modulus = &ctx.q_moduli_big[q_idx];
    let bits = ctx.num_limbs;
    let carry_ctx = ctx.carry_arith_ctx.clone();
    let mut circuit = source.fresh_sub_circuit();
    let row0 = circuit.input(width).to_vec();
    let row1 = circuit.input(width).to_vec();
    let offset = circuit.input(bits).to_vec();
    let columns = (0..width.max(bits))
        .map(|weight| {
            let mut column = Vec::with_capacity(3);
            if weight < width {
                column.push(row0[weight]);
                column.push(row1[weight]);
            }
            if weight < bits {
                column.push(offset[weight]);
            }
            column
        })
        .collect();
    let mut bound = ((BigUint::one() << width) - BigUint::one()) * 2u32 + modulus - BigUint::one();
    let mut value = CarryArithPoly::from_columns(carry_ctx.clone(), &mut circuit, columns).limbs;
    value.truncate(bound.bits() as usize);
    // Replace every bit of weight `2^k >= 2^bits` by the bits of `2^k mod q_i` while this
    // shortens the value.
    while value.len() > bits {
        let folded_bound = (bits..value.len())
            .fold((BigUint::one() << bits) - BigUint::one(), |acc, weight| {
                acc + (BigUint::one() << weight) % modulus
            });
        if folded_bound.bits() >= bound.bits() {
            break;
        }
        let mut columns: Columns = value[..bits].iter().map(|&bit| vec![bit]).collect();
        for (weight, &bit) in value.iter().enumerate().skip(bits) {
            let residue = (BigUint::one() << weight) % modulus;
            for (residue_weight, column) in columns.iter_mut().enumerate() {
                if residue.bit(residue_weight as u64) {
                    column.push(bit);
                }
            }
        }
        value = CarryArithPoly::from_columns(carry_ctx.clone(), &mut circuit, columns).limbs;
        bound = folded_bound;
        value.truncate(bound.bits() as usize);
    }
    // Now `value <= bound < 2^(j + 1) q_i` for the largest `j` with `2^j q_i <= bound`; each
    // conditional subtraction of `2^j q_i` halves this bound.
    let width = value.len().max(bits);
    if value.len() < width {
        value.resize(width, circuit.const_zero_gate().as_single_wire());
    }
    let mut value = CarryArithPoly::new(carry_ctx.clone(), value);
    if &bound >= modulus {
        let largest_shift = (&bound / modulus).bits() - 1;
        for shift in (0..=largest_shift).rev() {
            let subtrahend = CarryArithPoly::constant(
                carry_ctx.clone(),
                &mut circuit,
                width,
                &(modulus << shift),
            );
            let (is_less, difference) = value.less_than(&subtrahend, &mut circuit);
            value = value.cmux(&difference, is_less, &mut circuit);
        }
    }
    circuit.output(value.limbs[..bits].to_vec());
    circuit
}

/// For every left row, `sum_h row[h-th block] G^(-1)(rhs[h])`, where a row holds one block of
/// `gadget_len` entries per right entry. The result is in Montgomery form at every active
/// q-level of the right entries.
pub(crate) fn negacyclic_decomposed_products<P: Poly + 'static>(
    rhs: &[&MontgomeryPoly<P>],
    left_rows: &[&[MontgomeryPoly<P>]],
    circuit: &mut PolyCircuit<P>,
) -> Vec<MontgomeryPoly<P>> {
    let first = rhs.first().expect("a Ring-GSW product requires a right entry");
    let ctx = first.ctx.clone();
    assert_eq!(
        ctx.carry_arith_ctx.limb_bit_size, 1,
        "the Montgomery Ring-GSW product requires one-bit limbs"
    );
    let window = first.window;
    let dimension = first.num_coefficients();
    assert!(dimension.is_power_of_two(), "the ring dimension must be a power of two");
    let bits = ctx.num_limbs;
    let gadget_len = ctx.gadget_len(window);
    for entry in rhs.iter().copied().chain(left_rows.iter().flat_map(|row| row.iter())) {
        assert_eq!(entry.ctx, ctx, "Ring-GSW product entries must share one context");
        assert_eq!(entry.window, window, "Ring-GSW product entries must share one CRT window");
        assert_eq!(
            entry.num_coefficients(),
            dimension,
            "Ring-GSW product entries must share one ring dimension"
        );
    }
    for row in left_rows {
        assert_eq!(row.len(), rhs.len() * gadget_len, "a left row needs one gadget block per rhs");
    }

    let regular = rhs.iter().map(|entry| entry.to_regular(circuit)).collect::<Vec<_>>();
    let product =
        digit_product_circuit(circuit, &ctx.carry_arith_ctx, dimension, rhs.len() * bits, bits);
    let layouts = product.outputs;
    let product_id = circuit.register_sub_circuit(product.circuit);
    let width = layouts.iter().map(|layout| layout.heights.len()).max().unwrap_or(0);
    let reduction_ids = (window.offset..window.end())
        .map(|q_idx| {
            let reduction = reduction_circuit(circuit, ctx.as_ref(), q_idx, width);
            circuit.register_sub_circuit(reduction)
        })
        .collect::<Vec<_>>();

    let zero = circuit.const_zero_gate().as_single_wire();
    let one = circuit.const_one_gate().as_single_wire();
    left_rows
        .iter()
        .map(|row| {
            let value = (0..window.depth)
                .map(|level| {
                    let q_idx = window.offset + level;
                    let modulus = BigInt::from(ctx.q_moduli_big[q_idx].clone());
                    let inputs = (0..rhs.len())
                        .flat_map(|h| (0..bits).map(move |digit| (h, digit)))
                        .flat_map(|(h, digit)| {
                            let left = &row[h * gadget_len + level * bits + digit].value[level];
                            let left_bits =
                                left.iter().flat_map(|coefficient| coefficient.limbs.clone());
                            let right_bits =
                                regular[h][level].iter().map(move |residue| residue.limbs[digit]);
                            left_bits.chain(right_bits).collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let mut outputs = circuit
                        .call_sub_circuit(product_id, inputs)
                        .into_iter()
                        .map(|output| output.as_single_wire());
                    layouts
                        .iter()
                        .map(|layout| {
                            let mut row0 = vec![zero; width];
                            let mut row1 = vec![zero; width];
                            for (weight, &height) in layout.heights.iter().enumerate() {
                                for slot in 0..height {
                                    let bit = outputs.next().expect("a product output per wire");
                                    if slot == 0 {
                                        row0[weight] = bit;
                                    } else {
                                        row1[weight] = bit;
                                    }
                                }
                            }
                            let offset = layout
                                .offset
                                .mod_floor(&modulus)
                                .to_biguint()
                                .expect("a residue is non-negative");
                            let offset_bits = (0..bits)
                                .map(|weight| if offset.bit(weight as u64) { one } else { zero });
                            let reduced = circuit.call_sub_circuit(
                                reduction_ids[level],
                                row0.into_iter().chain(row1).chain(offset_bits).collect::<Vec<_>>(),
                            );
                            CarryArithPoly::new(
                                ctx.carry_arith_ctx.clone(),
                                reduced.into_iter().map(|bit| bit.as_single_wire()).collect(),
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            MontgomeryPoly::normalized(ctx.clone(), value, window)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuit::PolyGateKind,
        circuit_gadgets::{
            arith::{CrtWindow, DecomposeArithmeticGadget, encode_montgomery_poly},
            fhe::ring_gsw::{RingGswCiphertext, RingGswContext},
        },
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        test_utils::execute_circuit,
        utils::gen_biguint_for_modulus,
    };
    use mxx_backends::poly::{
        Poly as _, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use std::sync::Arc;

    fn env_parameter(name: &str, default: usize) -> usize {
        std::env::var(format!("MXX_KHE_TEST_MONTGOMERY_RING_GSW_{name}"))
            .map_or(default, |value| value.parse().expect("a positive integer parameter"))
    }

    /// Random polynomials of every entry of a `2 x width` Ring-GSW ciphertext, row by row.
    fn random_ciphertext(parameters: &DCRTPolyParams, width: usize) -> Vec<Vec<BigUint>> {
        let mut rng = rand::rng();
        let modulus = parameters.modulus();
        (0..2 * width)
            .map(|_| {
                (0..parameters.ring_dimension())
                    .map(|_| gen_biguint_for_modulus(&mut rng, modulus.as_ref()))
                    .collect()
            })
            .collect()
    }

    fn native_ciphertext(
        parameters: &DCRTPolyParams,
        entries: &[Vec<BigUint>],
        width: usize,
    ) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec(
            parameters,
            entries
                .chunks(width)
                .map(|row| {
                    row.iter().map(|coeffs| DCRTPoly::from_biguints(parameters, coeffs)).collect()
                })
                .collect(),
        )
    }

    #[test]
    fn ring_gsw_product_of_montgomery_entries_matches_the_native_product_at_runtime() {
        let parameters = DCRTPolyParams::new(
            env_parameter("RING_DIMENSION", 4) as u32,
            env_parameter("CRT_DEPTH", 1),
            env_parameter("CRT_BITS", 8),
            4,
            None,
            None,
        );
        let dimension = parameters.ring_dimension() as usize;
        let window = CrtWindow::full(parameters.to_crt().0.len());
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let arithmetic =
            Arc::new(MontgomeryPolyContext::setup(&mut circuit, &parameters, 1, false));
        let context = Arc::new(RingGswContext::from_arith_context(
            &mut circuit,
            &parameters,
            dimension,
            arithmetic.clone(),
            window,
        ));
        let width = context.width();
        let left = RingGswCiphertext::<DCRTPoly, MontgomeryPoly<DCRTPoly>>::input(
            context.clone(),
            None,
            &mut circuit,
        );
        let right = RingGswCiphertext::input(context, None, &mut circuit);
        let product = left.mul(&right, &mut circuit);
        let outputs = product.reconstruct(&mut circuit);
        circuit.output(outputs);
        let counts = circuit.count_gates_by_type_vec();
        assert!(
            [PolyGateKind::PubLut, PolyGateKind::SlotTransfer, PolyGateKind::SlotReduce]
                .iter()
                .all(|kind| !counts.contains_key(kind)),
            "the Montgomery Ring-GSW product must not use lookups or slot operations: {counts:?}"
        );

        let left_entries = random_ciphertext(&parameters, width);
        let right_entries = random_ciphertext(&parameters, width);
        let inputs = left_entries
            .iter()
            .chain(&right_entries)
            .flat_map(|coeffs| encode_montgomery_poly::<DCRTPoly>(1, &parameters, coeffs))
            .map(|poly| DCRTPolyMatrix::from_poly_vec_row(&parameters, vec![poly]))
            .collect::<Vec<_>>();
        let actual =
            execute_circuit("montgomery-ring-gsw-product-runtime", &parameters, &circuit, &inputs);

        let right_native = native_ciphertext(&parameters, &right_entries, width);
        let expected = native_ciphertext(&parameters, &left_entries, width) *
            MontgomeryPoly::<DCRTPoly>::gadget_decomposed::<DCRTPolyMatrix>(
                &parameters,
                &arithmetic,
                &right_native,
                window,
            );
        assert_eq!(actual.len(), 2 * width);
        for (index, actual) in actual.iter().enumerate() {
            assert_eq!(
                actual.entry(0, 0),
                expected.entry(index / width, index % width),
                "Ring-GSW product entry {index}"
            );
        }
    }
}
