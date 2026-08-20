use super::*;
use crate::circuit_gadgets::arith::ModularArithmeticContext;

/// Pack the p-residue wires of one lane-packed nested-RNS value into a `BatchedWire`.
pub(super) fn nested_rns_level_from_wires<I, W>(wires: I) -> BatchedWire
where
    I: IntoIterator<Item = W>,
    W: Into<BatchedWire>,
{
    BatchedWire::from_batches(wires)
}

/// Build a placeholder LUT used when a real lookup is not needed yet but an id must exist.
///
/// `NestedRnsPolyContext::setup` allocates all helper LUTs and sub-circuits in one pass. This tiny
/// LUT preserves the previous initialization flow without changing any call sites.
fn dummy_lut() -> PublicLutProgram {
    PublicLutProgram::new(1, LutExpr::constant(0))
        .expect("the constant-zero lookup program is valid")
}

// Conservative output bound for the integer represented by one q-level after full_reduce.
//
// The current implementation uses canonical nonnegative residues throughout:
// - each y_i produced by lut_x_to_y lies in [0, p_i),
// - each coefficient [p_hat_i]_q and [p]_q is represented in [0, q),
// - real_i = round(y_i * scale / p_i) is nonnegative, and because y_i <= p_i - 1 we have real_i <=
//   scale, so v = round(sum_i real_i / scale) satisfies 0 <= v <= k where k = p_moduli.len().
//
// For one q-level with modulus q, full_reduce evaluates the integer
//   x' = sum_i y_i * [p_hat_i]_q - v * [p]_q.
// Using the bounds above,
//   |x'|
//   <= sum_i |y_i| * |[p_hat_i]_q| + |v| * |[p]_q|
//   <  sum_i p_i * q + k * q
//   =  (sum_i p_i + k) * q.
//
// This is intentionally looser than the centered-residue bound from the paper: it matches the
// repository's [0, q) / [0, p_i) representation contract rather than the paper's symmetric one.
fn full_reduce_output_max_plaintext_bound(p_moduli: &[u64], q_modulus: u64) -> BigUint {
    let sum_p_moduli = p_moduli.iter().fold(BigUint::ZERO, |acc, &p_i| acc + BigUint::from(p_i));
    let modulus_count =
        u64::try_from(p_moduli.len()).expect("p_moduli length must fit in u64 for bound tracking");
    (sum_p_moduli + BigUint::from(modulus_count)) * BigUint::from(q_modulus)
}

/// Upper bound for the lookup domain used by the "mod p" and trace-management LUTs.
///
/// The arithmetic helpers track a conservative `p_max_trace` for each active q-level. This size
/// must cover both the product-style traces and the additive trace-offset path used by subtraction.
fn lut_mod_p_map_size(p_i: u64, max_p_modulus: u64, modulus_count: usize) -> u128 {
    (p_i as u128 * max_p_modulus as u128).max(p_i as u128 * (2 * modulus_count) as u128)
}

/// Every mod-p lookup receives wires governed by the context-wide trace invariant, not merely
/// one canonical p_i residue. Therefore every physical table uses the same authoritative input
/// coefficient domain. The expression remains parameter-generic and is exactly the capacity
/// tracked by `lut_mod_p_max_map_size`.
fn lut_mod_p_table_len(max_p_modulus: u64, modulus_count: usize) -> u128 {
    lut_mod_p_map_size(max_p_modulus, max_p_modulus, modulus_count)
}

fn lookup_input_coefficient_fits(bound: &BigUint, table_len: &BigUint) -> bool {
    bound < table_len
}

/// Precompute the gadget vector residues used by decomposition and reconstruction helpers.
///
/// The returned rows are consumed by the encoding helpers in `encoding.rs` and by the full-reduce
/// support sub-circuits registered during context setup.
fn precompute_nested_rns_gadget_values(
    q_moduli: &[u64],
    p_full: &BigUint,
    p_over_pis: &[BigUint],
) -> Vec<Vec<BigUint>> {
    q_moduli
        .iter()
        .map(|&q_i| {
            let q_i_big = BigUint::from(q_i);
            let p_mod_qi = p_full % &q_i_big;
            let mut level_values =
                p_over_pis.iter().map(|p_hat_i| p_hat_i % &q_i_big).collect::<Vec<_>>();
            level_values.push(if p_mod_qi == BigUint::ZERO {
                BigUint::ZERO
            } else {
                &q_i_big - &p_mod_qi
            });
            level_values
        })
        .collect()
}

impl NestedRnsPolyContext {
    pub fn q_moduli(&self) -> &[u64] {
        &self.q_moduli
    }

    pub(crate) fn full_reduce_output_metadata(
        &self,
        window: CrtWindow,
    ) -> (Vec<BigUint>, Vec<BigUint>) {
        let window = CrtWindow::new(window.offset, window.depth, self.q_moduli_depth);
        let max_plaintexts = self.full_reduce_max_plaintexts[window.offset..window.end()]
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let p_max_traces = vec![self.reduced_p_max_trace(); window.depth];
        (max_plaintexts, p_max_traces)
    }

    pub(crate) fn reduced_p_max_trace(&self) -> BigUint {
        BigUint::from(self.p_max - 1)
    }

    pub(super) fn unreduced_trace_threshold(&self) -> BigUint {
        BigUint::from(self.p_max)
    }

    pub(crate) fn trace_capacity_bound(&self) -> BigUint {
        self.lut_mod_p_max_map_size.clone()
    }

    /// Return the per-input maximum coefficients for a nested-RNS lookup call.
    ///
    /// Nested-RNS stores only one trace bound per active q-level.  That bound has the explicit
    /// invariant that every canonical, nonnegative constant coefficient presented to a residue
    /// lookup is at most that trace.  A call covers all active q-levels at once, so its one
    /// authoritative bound is their maximum.  Validate it against every actual per-residue LUT:
    /// comparing only with the largest table would incorrectly admit a value that does not fit a
    /// smaller `p_i` table.
    pub(crate) fn lookup_input_ranges_for_trace(
        &self,
        trace: &BigUint,
    ) -> Vec<SubCircuitInputMaxPlaintextNormRange> {
        for &p_i in &self.p_moduli {
            let table_len = self.lut_mod_p_max_map_size.clone();
            assert!(
                lookup_input_coefficient_fits(trace, &table_len),
                "nested-RNS lookup input bound {trace} does not fit the p={p_i} table of length {table_len}"
            );
        }
        SubCircuitInputMaxPlaintextNormRange::compress(&vec![trace.clone(); self.p_moduli.len()])
    }

    /// Return exact per-residue bounds for products of canonical residues.
    ///
    /// Each input is in `[0, p_i - 1]`; multiplication therefore presents at most
    /// `(p_i - 1)^2` to the subsequent reduction lookup.  This is intentionally calculated per
    /// residue rather than widened to the largest modulus.
    pub(crate) fn canonical_residue_product_input_ranges(
        &self,
    ) -> Vec<SubCircuitInputMaxPlaintextNormRange> {
        let norms = self
            .p_moduli
            .iter()
            .map(|&p_i| {
                let residue_max = BigUint::from(p_i - 1);
                &residue_max * &residue_max
            })
            .collect::<Vec<_>>();
        self.checked_lookup_input_ranges(norms)
    }

    /// Return exact per-residue bounds for a canonical residue times a canonical scalar.
    pub(crate) fn canonical_residue_scaled_input_ranges(
        &self,
    ) -> Vec<SubCircuitInputMaxPlaintextNormRange> {
        self.canonical_residue_product_input_ranges()
    }

    /// Full reduction has at most `k * (p_i - 1)` in the accumulated reduced terms and adds
    /// `k * p_i` before subtracting a nonnegative term, giving `2 * k * p_i - 1`.
    pub(crate) fn full_reduce_raw_input_ranges(&self) -> Vec<SubCircuitInputMaxPlaintextNormRange> {
        let k = BigUint::from(self.p_moduli.len());
        self.checked_lookup_input_ranges(
            self.p_moduli
                .iter()
                .map(|&p_i| BigUint::from(2u8) * &k * BigUint::from(p_i) - BigUint::from(1u8))
                .collect(),
        )
    }

    /// Construct and validate a complete input contract without scanning lookup table entries.
    pub(crate) fn checked_lookup_input_ranges(
        &self,
        norms: Vec<BigUint>,
    ) -> Vec<SubCircuitInputMaxPlaintextNormRange> {
        assert_eq!(norms.len(), self.p_moduli.len());
        for (&p_i, norm) in self.p_moduli.iter().zip(&norms) {
            let table_len = self.lut_mod_p_max_map_size.clone();
            assert!(
                lookup_input_coefficient_fits(norm, &table_len),
                "nested-RNS lookup input bound {norm} does not fit the p={p_i} table of length {table_len}"
            );
        }
        SubCircuitInputMaxPlaintextNormRange::compress(&norms)
    }

    fn register_local_support_subcircuits<P: Poly + 'static>(
        circuit: &mut PolyCircuit<P>,
        p_moduli: &[u64],
        p_max: u64,
        trace_capacity_bound: &BigUint,
        lut_mod_p_ids: &[usize],
        lut_x_to_y_ids: &[usize],
        lut_x_to_real_ids: &[usize],
        lut_real_to_v_id: usize,
    ) -> NestedRnsRegisteredSubcircuitIds {
        NestedRnsRegisteredSubcircuitIds {
            add_without_reduce_id: circuit
                .register_sub_circuit(Self::add_without_reduce_subcircuit::<P>(p_moduli)),
            sub_with_trace_offsets_id: circuit.register_sub_circuit(
                Self::sub_with_trace_offsets_subcircuit::<P>(p_moduli, p_max, trace_capacity_bound),
            ),
            lazy_reduce_id: circuit
                .register_sub_circuit(Self::lazy_reduce_subcircuit::<P>(p_moduli, lut_mod_p_ids)),
            decomposition_terms_id: circuit.register_sub_circuit(
                Self::decomposition_terms_subcircuit::<P>(
                    lut_x_to_y_ids,
                    lut_x_to_real_ids,
                    lut_real_to_v_id,
                ),
            ),
            gadget_decompose_id: circuit.register_sub_circuit(
                Self::gadget_decompose_subcircuit::<P>(
                    p_moduli,
                    lut_mod_p_ids,
                    lut_x_to_y_ids,
                    lut_x_to_real_ids,
                    lut_real_to_v_id,
                ),
            ),
        }
    }

    pub fn setup<P: Poly + 'static>(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        p_moduli_bits: usize,
        max_unreduced_muls: usize,
        scale: u64,
        dummy_scalar: bool,
        q_level: Option<usize>,
    ) -> Self {
        let (q_moduli, _q_moduli_bits, max_q_moduli_depth) = params.to_crt();
        let requested_q_level = q_level.unwrap_or(max_q_moduli_depth);
        assert!(
            requested_q_level <= max_q_moduli_depth,
            "q_level exceeds q_moduli_depth: q_level={}, q_moduli_depth={}",
            requested_q_level,
            max_q_moduli_depth
        );
        let q_moduli_depth = max_q_moduli_depth;
        let q_moduli_min = *q_moduli.iter().min().expect("there should be at least one q modulus");
        let q_moduli_max = *q_moduli.iter().max().expect("there should be at least one q modulus");
        let p_moduli = sample_crt_primes(p_moduli_bits, q_moduli_max, max_unreduced_muls);
        debug!(
            "NestedRnsPolyContext setup: p_moduli = {:?}, q_moduli = {:?}, scale = {}, max_unreduced_muls = {}",
            p_moduli, q_moduli, scale, max_unreduced_muls
        );
        let p_moduli_depth = p_moduli.len();
        let max_p_modulus = *p_moduli.iter().max().expect("p_moduli must not be empty");
        let p_moduli_depth_u64 =
            u64::try_from(p_moduli_depth).expect("p_moduli length must fit in u64");
        assert!(
            p_moduli_depth_u64 < max_p_modulus,
            "NestedRnsPolyContext requires p_moduli.len() < p_max, got s={} and p_max={}",
            p_moduli_depth,
            max_p_modulus
        );
        let lut_mod_p_max_map_size =
            BigUint::from(lut_mod_p_table_len(max_p_modulus, p_moduli_depth));
        let active_q_moduli = q_moduli;
        let p_full = p_moduli.iter().fold(BigUint::from(1u64), |acc, &pi| acc * BigUint::from(pi));
        let full_reduce_max_plaintexts = active_q_moduli
            .iter()
            .map(|&q_i| full_reduce_output_max_plaintext_bound(&p_moduli, q_i))
            .collect::<Vec<_>>();
        let p_over_pis =
            p_moduli.iter().map(|&p_i| &p_full / BigUint::from(p_i)).collect::<Vec<_>>();
        let gadget_values =
            precompute_nested_rns_gadget_values(&active_q_moduli, &p_full, &p_over_pis);
        if dummy_scalar {
            let dummy_lut = dummy_lut();
            let dummy_lut_id = circuit.register_public_lookup(dummy_lut);
            let lut_mod_p_ids = vec![dummy_lut_id; p_moduli_depth];
            let lut_x_to_y_ids = vec![dummy_lut_id; p_moduli_depth];
            let lut_x_to_real_ids = vec![dummy_lut_id; p_moduli_depth];
            let lut_real_to_v_id = dummy_lut_id;
            let registered_ids = Self::register_local_support_subcircuits::<P>(
                circuit,
                &p_moduli,
                max_p_modulus,
                &lut_mod_p_max_map_size,
                &lut_mod_p_ids,
                &lut_x_to_y_ids,
                &lut_x_to_real_ids,
                lut_real_to_v_id,
            );
            return Self {
                p_moduli_bits,
                max_unreduced_muls,
                scale,
                p_moduli,
                q_moduli: active_q_moduli,
                q_moduli_depth,
                p_max: max_p_modulus,
                lut_mod_p_max_map_size,
                p_full,
                p_over_pis,
                gadget_values,
                full_reduce_max_plaintexts,
                add_without_reduce_id: registered_ids.add_without_reduce_id,
                sub_with_trace_offsets_id: registered_ids.sub_with_trace_offsets_id,
                lazy_reduce_id: registered_ids.lazy_reduce_id,
                decomposition_terms_id: registered_ids.decomposition_terms_id,
                gadget_decompose_id: registered_ids.gadget_decompose_id,
            };
        }

        let mut lut_mod_p = Vec::with_capacity(p_moduli_depth);
        let mut lut_x_to_y = Vec::with_capacity(p_moduli_depth);
        let mut lut_x_to_real = Vec::with_capacity(p_moduli_depth);

        for (p_i_idx, &p_i) in p_moduli.iter().enumerate() {
            let lut_mod_p_map_size = lut_mod_p_table_len(max_p_modulus, p_moduli.len());
            debug_assert!(
                lut_mod_p_map_size < q_moduli_min as u128,
                "LUT size exceeds q modulus size; increase q_moduli_bits or decrease p_moduli_bits"
            );
            let lut_mod_p_len = lut_mod_p_map_size as usize;
            let input = LutExpr::input();
            let lut_mod_p_lut =
                PublicLutProgram::new(lut_mod_p_len as u64, input.clone().modulo(p_i))
                    .expect("nested-RNS modulus lookup program is valid");
            lut_mod_p.push(lut_mod_p_lut);

            let p_moduli_big = BigUint::from(p_i);
            let p_over_pi_mod_pi = (&p_over_pis[p_i_idx] % &p_moduli_big)
                .to_u64()
                .expect("CRT residue must fit in u64");
            let p_over_pi_inv = BigUint::from(
                mod_inverse(p_over_pi_mod_pi, p_i).expect("CRT moduli must be coprime"),
            );

            let lut_x_to_y_len = lut_mod_p_map_size as usize;
            let y = input.clone().mul(LutExpr::constant(p_over_pi_inv)).modulo(p_i);
            let lut_x_to_y_lut = PublicLutProgram::new(lut_x_to_y_len as u64, y.clone())
                .expect("nested-RNS CRT conversion lookup program is valid");
            lut_x_to_y.push(lut_x_to_y_lut);
            let lut_x_to_real_len = lut_mod_p_map_size as usize;
            let lut_x_to_real_lut = PublicLutProgram::new(
                lut_x_to_real_len as u64,
                y.mul(LutExpr::constant(scale)).round_div(p_i),
            )
            .expect("nested-RNS scaled lookup program is valid");
            lut_x_to_real.push(lut_x_to_real_lut);
        }

        let max_real = scale * p_moduli_depth as u64;
        let lut_real_to_v_len = max_real as usize + 1;
        let lut_real_to_v_lut =
            PublicLutProgram::new(lut_real_to_v_len as u64, LutExpr::input().round_div(scale))
                .expect("nested-RNS rounding lookup program is valid");

        let lut_mod_p_ids = lut_mod_p
            .iter()
            .map(|lut| circuit.register_public_lookup(lut.clone()))
            .collect::<Vec<_>>();
        let lut_x_to_y_ids = lut_x_to_y
            .iter()
            .map(|lut| circuit.register_public_lookup(lut.clone()))
            .collect::<Vec<_>>();
        let lut_x_to_real_ids = lut_x_to_real
            .iter()
            .map(|lut| circuit.register_public_lookup(lut.clone()))
            .collect::<Vec<_>>();
        let lut_real_to_v_id = circuit.register_public_lookup(lut_real_to_v_lut);

        let registered_ids = Self::register_local_support_subcircuits::<P>(
            circuit,
            &p_moduli,
            max_p_modulus,
            &lut_mod_p_max_map_size,
            &lut_mod_p_ids,
            &lut_x_to_y_ids,
            &lut_x_to_real_ids,
            lut_real_to_v_id,
        );
        Self {
            p_moduli_bits,
            max_unreduced_muls,
            scale,
            p_moduli,
            q_moduli: active_q_moduli,
            q_moduli_depth,
            p_max: max_p_modulus,
            lut_mod_p_max_map_size,
            p_full,
            p_over_pis,
            gadget_values,
            full_reduce_max_plaintexts,
            add_without_reduce_id: registered_ids.add_without_reduce_id,
            sub_with_trace_offsets_id: registered_ids.sub_with_trace_offsets_id,
            lazy_reduce_id: registered_ids.lazy_reduce_id,
            decomposition_terms_id: registered_ids.decomposition_terms_id,
            gadget_decompose_id: registered_ids.gadget_decompose_id,
        }
    }

    pub(crate) fn reduce_q_level_row<P: Poly>(
        &self,
        row: &[GateId],
        input_norms: &[BigUint],
        circuit: &mut PolyCircuit<P>,
    ) -> (Vec<GateId>, Vec<BigUint>) {
        assert_eq!(
            row.len(),
            self.p_moduli.len(),
            "q-level row depth {} must match p_moduli depth {}",
            row.len(),
            self.p_moduli.len()
        );
        assert_eq!(input_norms.len(), self.p_moduli.len());
        let outputs = circuit
            .call_sub_circuit_with_max_plaintext_norms(
                self.lazy_reduce_id,
                row.iter().copied(),
                self.checked_lookup_input_ranges(input_norms.to_vec()),
            )
            .into_iter()
            .map(BatchedWire::as_single_wire)
            .collect();
        let output_norms = self.p_moduli.iter().map(|&p_i| BigUint::from(p_i - 1)).collect();
        (outputs, output_norms)
    }

    pub(crate) fn mul_q_level_rows<P: Poly>(
        &self,
        left: &[GateId],
        right: &[GateId],
        left_norms: &[BigUint],
        right_norms: &[BigUint],
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<GateId> {
        assert_eq!(left.len(), self.p_moduli.len(), "left q-level row depth mismatch");
        assert_eq!(right.len(), self.p_moduli.len(), "right q-level row depth mismatch");
        assert_eq!(left_norms.len(), self.p_moduli.len());
        assert_eq!(right_norms.len(), self.p_moduli.len());
        let products = left
            .iter()
            .zip(right)
            .map(|(&lhs, &rhs)| circuit.mul_gate(lhs, rhs).as_single_wire())
            .collect::<Vec<_>>();
        let product_norms =
            left_norms.iter().zip(right_norms).map(|(lhs, rhs)| lhs * rhs).collect::<Vec<_>>();
        circuit
            .call_sub_circuit_with_max_plaintext_norms(
                self.lazy_reduce_id,
                products,
                self.checked_lookup_input_ranges(product_norms),
            )
            .into_iter()
            .map(BatchedWire::as_single_wire)
            .collect()
    }

    fn add_without_reduce_subcircuit<P: Poly>(p_moduli: &[u64]) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth * 2);
        let left = inputs.slice(0..p_moduli_depth).to_vec();
        let right = inputs.slice(p_moduli_depth..inputs.len()).to_vec();
        let outputs = (0..p_moduli_depth)
            .map(|p_idx| circuit.add_gate(left[p_idx], right[p_idx]))
            .collect::<Vec<_>>();
        circuit.output(outputs);
        circuit
    }

    fn lazy_reduce_subcircuit<P: Poly>(
        p_moduli: &[u64],
        lut_mod_p_ids: &[usize],
    ) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth);
        let outputs = (0..p_moduli_depth)
            .map(|p_idx| circuit.public_lookup_gate(inputs.at(p_idx), lut_mod_p_ids[p_idx]))
            .collect::<Vec<_>>();
        circuit.output(outputs);
        circuit
    }

    fn decomposition_terms_subcircuit<P: Poly>(
        lut_x_to_y_ids: &[usize],
        lut_x_to_real_ids: &[usize],
        lut_real_to_v_id: usize,
    ) -> PolyCircuit<P> {
        assert_eq!(lut_x_to_y_ids.len(), lut_x_to_real_ids.len(), "decomposition LUT mismatch");
        let mut circuit = PolyCircuit::<P>::new();
        let p_moduli_depth = lut_x_to_y_ids.len();
        let inputs = circuit.input(p_moduli_depth);
        let outputs = Self::decomposition_term_gates(
            &mut circuit,
            &inputs.gate_ids().collect::<Vec<_>>(),
            lut_x_to_y_ids,
            lut_x_to_real_ids,
            lut_real_to_v_id,
        );
        circuit.output(outputs);
        circuit
    }

    fn decomposition_term_gates<P: Poly>(
        circuit: &mut PolyCircuit<P>,
        inputs: &[GateId],
        lut_x_to_y_ids: &[usize],
        lut_x_to_real_ids: &[usize],
        lut_real_to_v_id: usize,
    ) -> Vec<BatchedWire> {
        assert_eq!(inputs.len(), lut_x_to_y_ids.len());
        assert_eq!(inputs.len(), lut_x_to_real_ids.len());
        let (first_input, remaining_inputs) =
            inputs.split_first().expect("nested-RNS decomposition requires at least one p modulus");
        let mut outputs = Vec::with_capacity(inputs.len() + 1);
        let first_y = circuit.public_lookup_gate(*first_input, lut_x_to_y_ids[0]);
        outputs.push(first_y);
        let mut real_sum = circuit.public_lookup_gate(*first_input, lut_x_to_real_ids[0]);
        for (offset, input) in remaining_inputs.iter().enumerate() {
            let p_idx = offset + 1;
            let y_i = circuit.public_lookup_gate(*input, lut_x_to_y_ids[p_idx]);
            outputs.push(y_i);
            let real_i = circuit.public_lookup_gate(*input, lut_x_to_real_ids[p_idx]);
            real_sum = circuit.add_gate(real_sum, real_i);
        }
        outputs.push(circuit.public_lookup_gate(real_sum, lut_real_to_v_id));
        outputs
    }

    fn gadget_decompose_subcircuit<P: Poly>(
        p_moduli: &[u64],
        lut_mod_p_ids: &[usize],
        lut_x_to_y_ids: &[usize],
        lut_x_to_real_ids: &[usize],
        lut_real_to_v_id: usize,
    ) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth);
        let lazy_reduce_id = circuit
            .register_sub_circuit(Self::lazy_reduce_subcircuit::<P>(p_moduli, lut_mod_p_ids));
        let decomposition_terms = Self::decomposition_term_gates(
            &mut circuit,
            &inputs.gate_ids().collect::<Vec<_>>(),
            lut_x_to_y_ids,
            lut_x_to_real_ids,
            lut_real_to_v_id,
        );
        let ys = decomposition_terms[..p_moduli_depth].to_vec();
        let w = decomposition_terms[p_moduli_depth];
        let mut outputs = Vec::with_capacity((p_moduli_depth + 1) * p_moduli_depth);
        for (p_idx, y_i) in ys.into_iter().enumerate() {
            let repeated = vec![y_i; p_moduli_depth];
            outputs.extend(circuit.call_sub_circuit_with_max_plaintext_norms(
                lazy_reduce_id,
                &repeated,
                SubCircuitInputMaxPlaintextNormRange::compress(&vec![
                    BigUint::from(
                        p_moduli[p_idx] - 1
                    );
                    p_moduli_depth
                ]),
            ));
        }
        outputs.extend(circuit.call_sub_circuit_with_max_plaintext_norms(
            lazy_reduce_id,
            &vec![w; p_moduli_depth],
            SubCircuitInputMaxPlaintextNormRange::compress(&vec![
                BigUint::from(p_moduli_depth);
                p_moduli_depth
            ]),
        ));
        circuit.output(outputs);
        circuit
    }

    fn sub_with_trace_offsets_subcircuit<P: Poly>(
        p_moduli: &[u64],
        p_max: u64,
        trace_capacity_bound: &BigUint,
    ) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(2 * p_moduli_depth);
        let (left, right) = inputs.split_at(p_moduli_depth);
        let one = circuit.const_one_gate();
        let max_trace = trace_capacity_bound - BigUint::from(1u64);
        let max_offset_multiplier = (&max_trace + BigUint::from(p_max - 1)) / BigUint::from(p_max);
        let offset_param_ids = (0..p_moduli_depth)
            .map(|p_idx| {
                circuit.register_sub_circuit_param(SubCircuitParamSpec::LargeScalarMul {
                    max_scalar: &max_offset_multiplier * BigUint::from(p_moduli[p_idx]),
                })
            })
            .collect::<Vec<_>>();
        let outputs = (0..p_moduli_depth)
            .map(|p_idx| {
                let offset_gate = circuit.large_scalar_mul_param(one, offset_param_ids[p_idx]);
                let shifted_left = circuit.add_gate(left.at(p_idx), offset_gate);
                circuit.sub_gate(shifted_left, right.at(p_idx))
            })
            .collect::<Vec<_>>();
        circuit.output(outputs);
        circuit
    }
}

impl<P: Poly + 'static> ModularArithmeticContext<P> for NestedRnsPolyContext {
    fn q_moduli_depth(&self) -> usize {
        self.q_moduli_depth
    }

    fn decomposition_len(&self) -> usize {
        self.p_moduli.len() + 1
    }

    fn q_level_row_width(&self) -> usize {
        self.p_moduli.len()
    }

    fn randomizer_decomposition_bound(&self) -> u64 {
        self.p_moduli
            .iter()
            .copied()
            .max()
            .expect("NestedRnsPolyContext requires at least one p modulus")
    }

    fn decomposition_term_bound(&self, term_idx: usize) -> BigUint {
        if term_idx < self.p_moduli.len() {
            BigUint::from(self.p_moduli[term_idx] - 1)
        } else {
            BigUint::from(
                u64::try_from(self.p_moduli.len()).expect("p_moduli length must fit in u64"),
            )
        }
    }

    fn plaintext_capacity_bound(&self) -> BigUint {
        self.p_full.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mod_p_table_domain_accepts_its_largest_input_coefficient() {
        let table_len = BigUint::from(lut_mod_p_table_len(29, 3));
        let largest_valid = &table_len - BigUint::from(1u8);
        assert!(lookup_input_coefficient_fits(&largest_valid, &table_len));
    }

    #[test]
    fn mod_p_table_domain_rejects_the_first_larger_input_coefficient() {
        let table_len = BigUint::from(lut_mod_p_table_len(29, 3));
        assert!(!lookup_input_coefficient_fits(&table_len, &table_len));
    }
}
