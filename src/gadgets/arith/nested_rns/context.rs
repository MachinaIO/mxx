use super::*;

/// Pack one q-level worth of p-residue wires into the repository's `BatchedWire` shape.
///
/// Both `NestedRnsPolyContext` and `NestedRnsPoly<P>` construct one batch per active q-level.
/// Keeping the conversion here lets `poly.rs` import the exact same helper without reimplementing
/// the batching rule.
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
fn dummy_lut<P: Poly + 'static>(params: &P::Params) -> PublicLut<P> {
    PublicLut::new(
        params,
        1,
        |params, x| {
            if x != 0 {
                return None;
            }
            let y_elem = P::from_usize_to_constant(params, 0)
                .coeffs()
                .into_iter()
                .next()
                .expect("constant-term coefficient must exist");
            Some((0, y_elem))
        },
        None,
    )
}

/// Convert one integer constant into the `(row_idx, coeff)` format expected by `PublicLut::new`.
///
/// Several setup-time LUT constructors start from a `BigUint` semantic value, then need the maximum
/// coefficient row representation that the lookup API consumes.
fn max_output_row_from_biguint<P: Poly>(
    params: &P::Params,
    idx: usize,
    value: BigUint,
) -> (u64, P::Elem) {
    let poly = P::from_biguint_to_constant(params, value);
    let coeff =
        poly.coeffs().into_iter().max().expect("max_output_row requires at least one coefficient");
    (u64::try_from(idx).expect("row index must fit in u64"), coeff)
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

/// Flatten the per-level full-reduce scalar tables into sub-circuit parameter bindings.
///
/// `build_full_reduce_bindings` computes the raw `y` and `v` scalars in matrix form; this helper
/// converts one q-level worth of those scalars into the exact binding order expected by the
/// pre-registered full-reduce sub-circuit.
fn full_reduce_param_bindings(
    scalars_y: &[Vec<u32>],
    scalars_v: &[u32],
) -> Vec<SubCircuitParamValue> {
    let (mut y_bindings, v_bindings) = rayon::join(
        || {
            scalars_y
                .par_iter()
                .flat_map_iter(|row| {
                    row.iter().map(|scalar| SubCircuitParamValue::SmallScalarMul(vec![*scalar]))
                })
                .collect::<Vec<_>>()
        },
        || {
            scalars_v
                .par_iter()
                .map(|scalar| SubCircuitParamValue::SmallScalarMul(vec![*scalar]))
                .collect::<Vec<_>>()
        },
    );
    let total_len = y_bindings.len() + v_bindings.len();
    y_bindings.reserve(v_bindings.len());
    y_bindings.extend(v_bindings);
    debug_assert_eq!(y_bindings.len(), total_len);
    y_bindings
}

/// Precompute all parameter bindings for the full-reduce helper sub-circuit.
///
/// Every active q-level shares the same circuit shape but uses different CRT residues. Precomputing
/// the bindings once inside the context lets `NestedRnsPoly::full_reduce` call the helper by id
/// without recomputing these scalar tables each time.
fn build_full_reduce_bindings(
    p_moduli: &[u64],
    q_moduli: &[u64],
    p_full: &BigUint,
    p_over_pis: &[BigUint],
) -> Vec<Vec<SubCircuitParamValue>> {
    let p_moduli_depth = p_moduli.len();
    let q_moduli_depth = q_moduli.len();
    let mut scalars_y = vec![vec![vec![0; p_moduli_depth]; p_moduli_depth]; q_moduli_depth];
    let mut scalars_v = vec![vec![0; p_moduli_depth]; q_moduli_depth];

    for (p_i_idx, &p_i) in p_moduli.iter().enumerate() {
        for (q_idx, &q_k) in q_moduli.iter().enumerate() {
            let q_k_big = BigUint::from(q_k);
            for (p_j_idx, p_over_pj) in p_over_pis.iter().enumerate() {
                let p_over_pj_mod_qk =
                    (p_over_pj % &q_k_big).to_u64().expect("CRT residue must fit in u64");
                scalars_y[q_idx][p_i_idx][p_j_idx] = (p_over_pj_mod_qk % p_i) as u32;
            }
            let p_mod_qk = (p_full % &q_k_big).to_u64().expect("CRT residue must fit in u64");
            scalars_v[q_idx][p_i_idx] = (p_mod_qk % p_i) as u32;
        }
    }

    (0..q_moduli_depth)
        .into_par_iter()
        .map(|q_idx| full_reduce_param_bindings(&scalars_y[q_idx], &scalars_v[q_idx]))
        .collect()
}

impl NestedRnsPolyContext {
    pub(crate) fn zero_level_batch<P: Poly>(&self, circuit: &mut PolyCircuit<P>) -> BatchedWire {
        nested_rns_level_from_wires((0..self.p_moduli.len()).map(|_| circuit.const_zero_gate()))
    }

    pub(crate) fn q_moduli(&self) -> &[u64] {
        &self.q_moduli
    }

    pub(crate) fn full_reduce_output_metadata(
        &self,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> (Vec<BigUint>, Vec<BigUint>) {
        let level_offset = level_offset.unwrap_or(0);
        let input_count = enable_levels.unwrap_or(self.q_moduli_depth);
        assert!(
            level_offset + input_count <= self.q_moduli_depth,
            "active range exceeds q_moduli_depth: level_offset={level_offset}, enable_levels={input_count}, q_moduli_depth={}",
            self.q_moduli_depth
        );
        let max_plaintexts = self.full_reduce_max_plaintexts
            [level_offset..level_offset + input_count]
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let p_max_traces = vec![self.reduced_p_max_trace(); input_count];
        (max_plaintexts, p_max_traces)
    }

    pub(crate) fn reduced_p_max_trace(&self) -> BigUint {
        BigUint::from(self.p_max - 1)
    }

    pub(crate) fn p_full_ref(&self) -> &BigUint {
        &self.p_full
    }

    pub(crate) fn lut_mod_p_max_map_size_ref(&self) -> &BigUint {
        &self.lut_mod_p_max_map_size
    }

    pub(super) fn unreduced_trace_threshold(&self) -> BigUint {
        BigUint::from(self.p_max)
    }

    fn register_local_support_subcircuits<P: Poly + 'static>(
        circuit: &mut PolyCircuit<P>,
        p_moduli: &[u64],
        lut_mod_p_ids: &[usize],
        lut_x_to_y_ids: &[usize],
        lut_x_to_real_ids: &[usize],
        lut_real_to_v_id: usize,
    ) -> NestedRnsRegisteredSubcircuitIds {
        let p_moduli_depth = p_moduli.len();
        NestedRnsRegisteredSubcircuitIds {
            add_without_reduce_id: circuit
                .register_sub_circuit(Self::add_without_reduce_subcircuit::<P>(p_moduli)),
            sub_with_trace_offsets_id: circuit
                .register_sub_circuit(Self::sub_with_trace_offsets_subcircuit::<P>(p_moduli_depth)),
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
            full_reduce_id: circuit.register_sub_circuit(Self::full_reduce_subcircuit::<P>(
                p_moduli,
                lut_mod_p_ids,
                lut_x_to_y_ids,
                lut_x_to_real_ids,
                lut_real_to_v_id,
            )),
            mul_lazy_reduce_id: circuit.register_sub_circuit(
                Self::mul_lazy_reduce_subcircuit::<P>(p_moduli, lut_mod_p_ids),
            ),
            mul_right_sparse_id: circuit.register_sub_circuit(
                Self::mul_right_sparse_subcircuit::<P>(p_moduli, lut_mod_p_ids),
            ),
        }
    }

    fn register_shared_support_subcircuits<P: Poly + 'static>(
        &self,
        source_circuit: &PolyCircuit<P>,
        circuit: &mut PolyCircuit<P>,
    ) -> NestedRnsRegisteredSubcircuitIds {
        circuit.inherit_registries_from_parent(source_circuit);
        NestedRnsRegisteredSubcircuitIds {
            add_without_reduce_id: circuit.register_shared_sub_circuit(
                source_circuit.registered_sub_circuit_ref(self.add_without_reduce_id),
            ),
            sub_with_trace_offsets_id: circuit.register_shared_sub_circuit(
                source_circuit.registered_sub_circuit_ref(self.sub_with_trace_offsets_id),
            ),
            lazy_reduce_id: circuit.register_shared_sub_circuit(
                source_circuit.registered_sub_circuit_ref(self.lazy_reduce_id),
            ),
            decomposition_terms_id: circuit.register_shared_sub_circuit(
                source_circuit.registered_sub_circuit_ref(self.decomposition_terms_id),
            ),
            gadget_decompose_id: circuit.register_shared_sub_circuit(
                source_circuit.registered_sub_circuit_ref(self.gadget_decompose_id),
            ),
            full_reduce_id: circuit.register_shared_sub_circuit(
                source_circuit.registered_sub_circuit_ref(self.full_reduce_id),
            ),
            mul_lazy_reduce_id: circuit.register_shared_sub_circuit(
                source_circuit.registered_sub_circuit_ref(self.mul_lazy_reduce_id),
            ),
            mul_right_sparse_id: circuit.register_shared_sub_circuit(
                source_circuit.registered_sub_circuit_ref(self.mul_right_sparse_id),
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
        let q_moduli_depth = q_level.unwrap_or(max_q_moduli_depth);
        assert!(
            q_moduli_depth <= max_q_moduli_depth,
            "q_level exceeds q_moduli_depth: q_level={}, q_moduli_depth={}",
            q_moduli_depth,
            max_q_moduli_depth
        );
        let q_moduli_min = q_moduli.iter().min().expect("there should be at least one q modulus");
        let q_moduli_max = q_moduli.iter().max().expect("there should be at least one q modulus");
        let p_moduli = sample_crt_primes(p_moduli_bits, *q_moduli_max, max_unreduced_muls);
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
            BigUint::from(lut_mod_p_map_size(max_p_modulus, max_p_modulus, p_moduli_depth));
        let active_q_moduli = q_moduli.iter().take(q_moduli_depth).copied().collect::<Vec<_>>();
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
            let dummy_lut = dummy_lut::<P>(params);
            let dummy_lut_id = circuit.register_public_lookup(dummy_lut);
            let lut_mod_p_ids = vec![dummy_lut_id; p_moduli_depth];
            let lut_x_to_y_ids = vec![dummy_lut_id; p_moduli_depth];
            let lut_x_to_real_ids = vec![dummy_lut_id; p_moduli_depth];
            let lut_real_to_v_id = dummy_lut_id;
            let registered_ids = Self::register_local_support_subcircuits::<P>(
                circuit,
                &p_moduli,
                &lut_mod_p_ids,
                &lut_x_to_y_ids,
                &lut_x_to_real_ids,
                lut_real_to_v_id,
            );
            let full_reduce_bindings =
                build_full_reduce_bindings(&p_moduli, &active_q_moduli, &p_full, &p_over_pis);
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
                lut_mod_p_ids,
                lut_x_to_y_ids,
                lut_x_to_real_ids,
                lut_real_to_v_id,
                add_without_reduce_id: registered_ids.add_without_reduce_id,
                sub_with_trace_offsets_id: registered_ids.sub_with_trace_offsets_id,
                lazy_reduce_id: registered_ids.lazy_reduce_id,
                decomposition_terms_id: registered_ids.decomposition_terms_id,
                gadget_decompose_id: registered_ids.gadget_decompose_id,
                full_reduce_id: registered_ids.full_reduce_id,
                full_reduce_bindings,
                mul_lazy_reduce_id: registered_ids.mul_lazy_reduce_id,
                mul_right_sparse_id: registered_ids.mul_right_sparse_id,
            };
        }

        let mut lut_mod_p = Vec::with_capacity(p_moduli_depth);
        let mut lut_x_to_y = Vec::with_capacity(p_moduli_depth);
        let mut lut_x_to_real = Vec::with_capacity(p_moduli_depth);

        for (p_i_idx, &p_i) in p_moduli.iter().enumerate() {
            let lut_mod_p_map_size = lut_mod_p_map_size(p_i, max_p_modulus, p_moduli.len());
            debug_assert!(
                lut_mod_p_map_size < *q_moduli_min as u128,
                "LUT size exceeds q modulus size; increase q_moduli_bits or decrease p_moduli_bits"
            );
            let lut_mod_p_len = lut_mod_p_map_size as usize;
            let max_mod_p_row = max_output_row_from_biguint::<P>(
                params,
                (p_i - 1) as usize,
                BigUint::from(p_i - 1),
            );
            let lut_mod_p_lut = PublicLut::<P>::new(
                params,
                lut_mod_p_len as u64,
                move |params, t| {
                    if t >= lut_mod_p_len as u64 {
                        return None;
                    }
                    let output = BigUint::from(t % p_i);
                    let y_elem = P::from_biguint_to_constant(params, output)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist");
                    Some((t, y_elem))
                },
                Some(max_mod_p_row),
            );
            lut_mod_p.push(lut_mod_p_lut);

            let p_moduli_big = BigUint::from(p_i);
            let p_over_pi_mod_pi = (&p_over_pis[p_i_idx] % &p_moduli_big)
                .to_u64()
                .expect("CRT residue must fit in u64");
            let p_over_pi_inv = Arc::new(BigUint::from(
                mod_inverse(p_over_pi_mod_pi, p_i).expect("CRT moduli must be coprime"),
            ));
            let max_idx_mod_pi =
                (((p_i - 1) as u128 * p_over_pi_mod_pi as u128) % p_i as u128) as usize;
            let max_x_to_y_row =
                max_output_row_from_biguint::<P>(params, max_idx_mod_pi, BigUint::from(p_i - 1));
            let max_x_to_real_value = round_div((p_i - 1) * scale, p_i);
            let max_x_to_real_row = max_output_row_from_biguint::<P>(
                params,
                max_idx_mod_pi,
                BigUint::from(max_x_to_real_value),
            );

            let p_moduli_big = Arc::new(p_moduli_big);
            let lut_x_to_y_len = lut_mod_p_map_size as usize;
            let lut_x_to_y_lut = PublicLut::<P>::new(
                params,
                lut_x_to_y_len as u64,
                {
                    let p_over_pi_inv = Arc::clone(&p_over_pi_inv);
                    let p_moduli_big = Arc::clone(&p_moduli_big);
                    move |params, t| {
                        if t >= lut_x_to_y_len as u64 {
                            return None;
                        }
                        let input = BigUint::from(t);
                        let output = (&input * p_over_pi_inv.as_ref()) % p_moduli_big.as_ref();
                        let y_elem = P::from_biguint_to_constant(params, output)
                            .coeffs()
                            .into_iter()
                            .next()
                            .expect("constant-term coefficient must exist");
                        Some((t, y_elem))
                    }
                },
                Some(max_x_to_y_row),
            );
            lut_x_to_y.push(lut_x_to_y_lut);

            let lut_x_to_real_len = lut_mod_p_map_size as usize;
            let lut_x_to_real_lut = PublicLut::<P>::new(
                params,
                lut_x_to_real_len as u64,
                {
                    let p_over_pi_inv = Arc::clone(&p_over_pi_inv);
                    let p_moduli_big = Arc::clone(&p_moduli_big);
                    move |params, t| {
                        if t >= lut_x_to_real_len as u64 {
                            return None;
                        }
                        let input = BigUint::from(t);
                        let y = ((&input * p_over_pi_inv.as_ref()) % p_moduli_big.as_ref())
                            .to_u64()
                            .expect("y must fit in u64");
                        let output = BigUint::from(round_div(y * scale, p_i));
                        let y_elem = P::from_biguint_to_constant(params, output)
                            .coeffs()
                            .into_iter()
                            .next()
                            .expect("constant-term coefficient must exist");
                        Some((t, y_elem))
                    }
                },
                Some(max_x_to_real_row),
            );
            lut_x_to_real.push(lut_x_to_real_lut);
        }

        let max_real = scale * p_moduli_depth as u64;
        let lut_real_to_v_len = max_real as usize + 1;
        let max_real_to_v_row = max_output_row_from_biguint::<P>(
            params,
            max_real as usize,
            BigUint::from(round_div(max_real, scale)),
        );
        let lut_real_to_v_lut = PublicLut::<P>::new(
            params,
            lut_real_to_v_len as u64,
            move |params, t| {
                if t >= lut_real_to_v_len as u64 {
                    return None;
                }
                let output = BigUint::from(round_div(t, scale));
                let y_elem = P::from_biguint_to_constant(params, output)
                    .coeffs()
                    .into_iter()
                    .next()
                    .expect("constant-term coefficient must exist");
                Some((t, y_elem))
            },
            Some(max_real_to_v_row),
        );

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
            &lut_mod_p_ids,
            &lut_x_to_y_ids,
            &lut_x_to_real_ids,
            lut_real_to_v_id,
        );
        let full_reduce_bindings =
            build_full_reduce_bindings(&p_moduli, &active_q_moduli, &p_full, &p_over_pis);
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
            lut_mod_p_ids,
            lut_x_to_y_ids,
            lut_x_to_real_ids,
            lut_real_to_v_id,
            add_without_reduce_id: registered_ids.add_without_reduce_id,
            sub_with_trace_offsets_id: registered_ids.sub_with_trace_offsets_id,
            lazy_reduce_id: registered_ids.lazy_reduce_id,
            decomposition_terms_id: registered_ids.decomposition_terms_id,
            gadget_decompose_id: registered_ids.gadget_decompose_id,
            full_reduce_id: registered_ids.full_reduce_id,
            full_reduce_bindings,
            mul_lazy_reduce_id: registered_ids.mul_lazy_reduce_id,
            mul_right_sparse_id: registered_ids.mul_right_sparse_id,
        }
    }

    pub(crate) fn register_subcircuits_in<P: Poly + 'static>(
        &self,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let p_moduli = self.p_moduli.clone();
        let q_moduli_depth = self.q_moduli_depth;
        let registered_ids = Self::register_local_support_subcircuits::<P>(
            circuit,
            &p_moduli,
            &self.lut_mod_p_ids,
            &self.lut_x_to_y_ids,
            &self.lut_x_to_real_ids,
            self.lut_real_to_v_id,
        );
        let full_reduce_bindings = build_full_reduce_bindings(
            &p_moduli,
            &self.q_moduli[..q_moduli_depth],
            &self.p_full,
            &self.p_over_pis,
        );
        Self {
            p_moduli_bits: self.p_moduli_bits,
            max_unreduced_muls: self.max_unreduced_muls,
            scale: self.scale,
            p_moduli,
            q_moduli: self.q_moduli.clone(),
            q_moduli_depth,
            p_max: self.p_max,
            lut_mod_p_max_map_size: self.lut_mod_p_max_map_size.clone(),
            p_full: self.p_full.clone(),
            p_over_pis: self.p_over_pis.clone(),
            gadget_values: self.gadget_values.clone(),
            full_reduce_max_plaintexts: self.full_reduce_max_plaintexts.clone(),
            lut_mod_p_ids: self.lut_mod_p_ids.clone(),
            lut_x_to_y_ids: self.lut_x_to_y_ids.clone(),
            lut_x_to_real_ids: self.lut_x_to_real_ids.clone(),
            lut_real_to_v_id: self.lut_real_to_v_id,
            add_without_reduce_id: registered_ids.add_without_reduce_id,
            sub_with_trace_offsets_id: registered_ids.sub_with_trace_offsets_id,
            lazy_reduce_id: registered_ids.lazy_reduce_id,
            decomposition_terms_id: registered_ids.decomposition_terms_id,
            gadget_decompose_id: registered_ids.gadget_decompose_id,
            full_reduce_id: registered_ids.full_reduce_id,
            full_reduce_bindings,
            mul_lazy_reduce_id: registered_ids.mul_lazy_reduce_id,
            mul_right_sparse_id: registered_ids.mul_right_sparse_id,
        }
    }

    pub(crate) fn register_shared_subcircuits_in<P: Poly + 'static>(
        &self,
        source_circuit: &PolyCircuit<P>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let p_moduli = self.p_moduli.clone();
        let q_moduli_depth = self.q_moduli_depth;
        let registered_ids = self.register_shared_support_subcircuits(source_circuit, circuit);
        let full_reduce_bindings = build_full_reduce_bindings(
            &p_moduli,
            &self.q_moduli[..q_moduli_depth],
            &self.p_full,
            &self.p_over_pis,
        );

        Self {
            p_moduli_bits: self.p_moduli_bits,
            max_unreduced_muls: self.max_unreduced_muls,
            scale: self.scale,
            p_moduli,
            q_moduli: self.q_moduli.clone(),
            q_moduli_depth,
            p_max: self.p_max,
            lut_mod_p_max_map_size: self.lut_mod_p_max_map_size.clone(),
            p_full: self.p_full.clone(),
            p_over_pis: self.p_over_pis.clone(),
            gadget_values: self.gadget_values.clone(),
            full_reduce_max_plaintexts: self.full_reduce_max_plaintexts.clone(),
            lut_mod_p_ids: self.lut_mod_p_ids.clone(),
            lut_x_to_y_ids: self.lut_x_to_y_ids.clone(),
            lut_x_to_real_ids: self.lut_x_to_real_ids.clone(),
            lut_real_to_v_id: self.lut_real_to_v_id,
            add_without_reduce_id: registered_ids.add_without_reduce_id,
            sub_with_trace_offsets_id: registered_ids.sub_with_trace_offsets_id,
            lazy_reduce_id: registered_ids.lazy_reduce_id,
            decomposition_terms_id: registered_ids.decomposition_terms_id,
            gadget_decompose_id: registered_ids.gadget_decompose_id,
            full_reduce_id: registered_ids.full_reduce_id,
            full_reduce_bindings,
            mul_lazy_reduce_id: registered_ids.mul_lazy_reduce_id,
            mul_right_sparse_id: registered_ids.mul_right_sparse_id,
        }
    }

    pub(crate) fn reduce_q_level_row<P: Poly>(
        &self,
        row: &[GateId],
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<GateId> {
        assert_eq!(
            row.len(),
            self.p_moduli.len(),
            "q-level row depth {} must match p_moduli depth {}",
            row.len(),
            self.p_moduli.len()
        );
        circuit
            .call_sub_circuit(self.lazy_reduce_id, row.iter().copied())
            .into_iter()
            .map(BatchedWire::as_single_wire)
            .collect()
    }

    pub(crate) fn mul_q_level_rows<P: Poly>(
        &self,
        left: &[GateId],
        right: &[GateId],
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<GateId> {
        assert_eq!(left.len(), self.p_moduli.len(), "left q-level row depth mismatch");
        assert_eq!(right.len(), self.p_moduli.len(), "right q-level row depth mismatch");
        circuit
            .call_sub_circuit(
                self.mul_lazy_reduce_id,
                left.iter().copied().chain(right.iter().copied()),
            )
            .into_iter()
            .map(BatchedWire::as_single_wire)
            .collect()
    }

    fn mul_lazy_reduce_subcircuit<P: Poly>(
        p_moduli: &[u64],
        lut_mod_p_ids: &[usize],
    ) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let mul_circuit = Self::mul_without_reduce_subcircuit::<P>(p_moduli);
        let reduce_circuit = Self::lazy_reduce_subcircuit::<P>(p_moduli, lut_mod_p_ids);
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth * 2);
        let mul_circuit_id = circuit.register_sub_circuit(mul_circuit);
        let prod = circuit.call_sub_circuit(mul_circuit_id, inputs.gate_ids());
        let reduce_circuit_id = circuit.register_sub_circuit(reduce_circuit);
        let reduced = circuit.call_sub_circuit(reduce_circuit_id, &prod);
        circuit.output(reduced);
        circuit
    }

    fn mul_right_sparse_subcircuit<P: Poly>(
        p_moduli: &[u64],
        lut_mod_p_ids: &[usize],
    ) -> PolyCircuit<P> {
        Self::mul_lazy_reduce_subcircuit::<P>(p_moduli, lut_mod_p_ids)
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

    fn mul_without_reduce_subcircuit<P: Poly>(p_moduli: &[u64]) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth * 2);
        let left = inputs.slice(0..p_moduli_depth).to_vec();
        let right = inputs.slice(p_moduli_depth..inputs.len()).to_vec();
        let outputs = (0..p_moduli_depth)
            .map(|p_idx| circuit.mul_gate(left[p_idx], right[p_idx]))
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
        let mut outputs = Vec::with_capacity(p_moduli_depth + 1);
        let mut real_sum = circuit.const_zero_gate();
        for p_idx in 0..p_moduli_depth {
            let y_i = circuit.public_lookup_gate(inputs.at(p_idx), lut_x_to_y_ids[p_idx]);
            outputs.push(y_i);
            let real_i = circuit.public_lookup_gate(inputs.at(p_idx), lut_x_to_real_ids[p_idx]);
            real_sum = circuit.add_gate(real_sum, real_i);
        }
        outputs.push(circuit.public_lookup_gate(real_sum, lut_real_to_v_id));
        circuit.output(outputs);
        circuit
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
        let decomposition_terms_id =
            circuit.register_sub_circuit(Self::decomposition_terms_subcircuit::<P>(
                lut_x_to_y_ids,
                lut_x_to_real_ids,
                lut_real_to_v_id,
            ));
        let lazy_reduce_id = circuit
            .register_sub_circuit(Self::lazy_reduce_subcircuit::<P>(p_moduli, lut_mod_p_ids));
        let decomposition_terms =
            circuit.call_sub_circuit(decomposition_terms_id, inputs.gate_ids());
        let ys = decomposition_terms[..p_moduli_depth].to_vec();
        let w = decomposition_terms[p_moduli_depth];
        let mut outputs = Vec::with_capacity((p_moduli_depth + 1) * p_moduli_depth);
        for y_i in ys {
            let repeated = vec![y_i; p_moduli_depth];
            outputs.extend(circuit.call_sub_circuit(lazy_reduce_id, &repeated));
        }
        outputs.extend(circuit.call_sub_circuit(lazy_reduce_id, &vec![w; p_moduli_depth]));
        circuit.output(outputs);
        circuit
    }

    fn full_reduce_subcircuit<P: Poly>(
        p_moduli: &[u64],
        lut_mod_p_ids: &[usize],
        lut_x_to_y_ids: &[usize],
        lut_x_to_real_ids: &[usize],
        lut_real_to_v_id: usize,
    ) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let p_moduli_depth = p_moduli.len();
        let inputs = circuit.input(p_moduli_depth);
        let x = inputs;
        let scalar_y_param_ids = (0..p_moduli_depth)
            .map(|_| {
                (0..p_moduli_depth)
                    .map(|_| {
                        circuit.register_sub_circuit_param(SubCircuitParamKind::SmallScalarMul)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let scalar_v_param_ids = (0..p_moduli_depth)
            .map(|_| circuit.register_sub_circuit_param(SubCircuitParamKind::SmallScalarMul))
            .collect::<Vec<_>>();

        let ys = (0..p_moduli_depth)
            .map(|p_idx| circuit.public_lookup_gate(x.at(p_idx), lut_x_to_y_ids[p_idx]))
            .collect::<Vec<_>>();
        let reals = (0..p_moduli_depth)
            .map(|p_idx| circuit.public_lookup_gate(x.at(p_idx), lut_x_to_real_ids[p_idx]))
            .collect::<Vec<_>>();
        let mut real_sum = circuit.const_zero_gate();
        for &real in reals.iter() {
            real_sum = circuit.add_gate(real_sum, real);
        }
        let v = circuit.public_lookup_gate(real_sum, lut_real_to_v_id);
        let mut p_i_sums = Vec::with_capacity(p_moduli_depth);
        for p_idx in 0..p_moduli_depth {
            let mut p_i_sum = circuit.const_zero_gate();
            for p_j_idx in 0..p_moduli_depth {
                let y_j = ys[p_j_idx];
                let term = circuit.small_scalar_mul_param(y_j, scalar_y_param_ids[p_idx][p_j_idx]);
                let term_mod_p = circuit.public_lookup_gate(term, lut_mod_p_ids[p_idx]);
                p_i_sum = circuit.add_gate(p_i_sum, term_mod_p);
            }
            let term = circuit.small_scalar_mul_param(v, scalar_v_param_ids[p_idx]);
            let p_i_const = circuit.const_digits(&[p_moduli.len() as u32 * p_moduli[p_idx] as u32]);
            let sum = circuit.add_gate(p_i_sum, p_i_const);
            let p_i_sum = circuit.sub_gate(sum, term);
            let p_i_sum_mod_p = circuit.public_lookup_gate(p_i_sum, lut_mod_p_ids[p_idx]);
            p_i_sums.push(p_i_sum_mod_p);
        }
        circuit.output(p_i_sums);
        circuit
    }

    fn sub_with_trace_offsets_subcircuit<P: Poly>(p_moduli_depth: usize) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::<P>::new();
        let inputs = circuit.input(2 * p_moduli_depth);
        let (left, right) = inputs.split_at(p_moduli_depth);
        let one = circuit.const_one_gate();
        let offset_param_ids = (0..p_moduli_depth)
            .map(|_| circuit.register_sub_circuit_param(SubCircuitParamKind::LargeScalarMul))
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
