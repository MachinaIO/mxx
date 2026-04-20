use super::*;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
/// Scheduling node used by the grouped execution-layer planner.
///
/// `non_free_depth()` and `eval_error` both want to schedule one whole direct or summed
/// sub-circuit call at once rather than treating each `SubCircuitOutput` placeholder gate as a
/// separate node. This enum is the minimal structure needed for that shared planner:
///
/// - `Regular(gate_id)` means a plain gate evaluated directly in the current circuit.
/// - `SubCircuitCall(call_id)` means "all outputs produced by one direct sub-circuit call".
/// - `SummedSubCircuitCall(summed_call_id)` means "all outputs produced by one summed call".
///
/// Grouping multi-output calls this way is important for `non_free_depth()` because the cache is
/// built per child-call profile and returns all child output depths together.
enum GroupedCallExecutionNodeId {
    Regular(GateId),
    SubCircuitCall(usize),
    SummedSubCircuitCall(usize),
}

#[derive(Debug, Clone)]
/// One prepared child-depth request for the current `non_free_depth()` layer.
///
/// The cache is keyed by the *child* circuit identity plus the exact input-depth vector observed at
/// one call site. We prepare these requests in parallel for a whole layer, deduplicate equal keys,
/// and only then recurse into children. That keeps repeated direct calls, repeated summed-call
/// profiles, and nested callers from rebuilding the same child depths over and over.
struct PreparedNonFreeDepthRequest<P: Poly> {
    key: NonFreeDepthCacheKey,
    sub_circuit: Arc<PolyCircuit<P>>,
}

const COMPAT_NON_FREE_DEPTH_KIND_ORDER: [PolyGateKind; 10] = [
    PolyGateKind::Input,
    PolyGateKind::Add,
    PolyGateKind::Sub,
    PolyGateKind::Mul,
    PolyGateKind::SmallScalarMul,
    PolyGateKind::LargeScalarMul,
    PolyGateKind::SlotTransfer,
    PolyGateKind::PubLut,
    PolyGateKind::SubCircuitOutput,
    PolyGateKind::SummedSubCircuitOutput,
];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct CompatNonFreeDepthContributionVector {
    counts: [u32; 10],
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct CompatNonFreeDepthProfile {
    total_depth: u32,
    contributions: CompatNonFreeDepthContributionVector,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct CompatNonFreeDepthCacheKey {
    circuit_key: usize,
    input_profiles: Box<[CompatNonFreeDepthProfile]>,
}

impl CompatNonFreeDepthContributionVector {
    fn incremented(mut self, kind: PolyGateKind) -> Self {
        self.counts[Self::kind_index(kind)] += 1;
        self
    }

    fn to_hash_map(self) -> HashMap<PolyGateKind, usize> {
        COMPAT_NON_FREE_DEPTH_KIND_ORDER
            .into_iter()
            .zip(self.counts)
            .filter_map(|(kind, count)| (count > 0).then_some((kind, count as usize)))
            .collect()
    }

    fn kind_index(kind: PolyGateKind) -> usize {
        match kind {
            PolyGateKind::Input => 0,
            PolyGateKind::Add => 1,
            PolyGateKind::Sub => 2,
            PolyGateKind::Mul => 3,
            PolyGateKind::SmallScalarMul => 4,
            PolyGateKind::LargeScalarMul => 5,
            PolyGateKind::SlotTransfer => 6,
            PolyGateKind::PubLut => 7,
            PolyGateKind::SubCircuitOutput => 8,
            PolyGateKind::SummedSubCircuitOutput => 9,
        }
    }
}

impl CompatNonFreeDepthProfile {
    fn incremented(self, kind: PolyGateKind) -> Self {
        Self {
            total_depth: self.total_depth + 1,
            contributions: self.contributions.incremented(kind),
        }
    }
}

impl<P: Poly> PolyCircuit<P> {
    pub fn count_gates_by_type_vec(&self) -> HashMap<PolyGateKind, usize> {
        self.expanded_gate_counts(true)
    }

    pub fn total_registered_public_lut_entries(&self) -> usize {
        self.lookup_registry.lookups.iter().map(|lookup| lookup.value().len()).sum()
    }

    fn expanded_gate_counts(&self, include_inputs: bool) -> HashMap<PolyGateKind, usize> {
        let mut counts: HashMap<PolyGateKind, usize> = HashMap::new();
        for gate in self.gates.values() {
            let kind = gate.gate_type.kind();
            if matches!(kind, PolyGateKind::SubCircuitOutput | PolyGateKind::SummedSubCircuitOutput)
            {
                continue;
            }
            if !include_inputs && matches!(kind, PolyGateKind::Input) {
                continue;
            }
            *counts.entry(kind).or_insert(0) += 1;
        }

        let mut call_counts: HashMap<usize, usize> = HashMap::new();
        for call in self.sub_circuit_calls.values() {
            *call_counts.entry(call.sub_circuit_id).or_insert(0) += 1;
        }
        for (sub_id, times) in call_counts {
            let sub_counts = self.with_sub_circuit(sub_id, |sub| sub.expanded_gate_counts(false));
            for (kind, count) in sub_counts {
                *counts.entry(kind).or_insert(0) += count * times;
            }
        }
        for summed_call in self.summed_sub_circuit_calls.values() {
            let times = summed_call.call_input_set_ids.len();
            let sub_counts = self.with_sub_circuit(summed_call.sub_circuit_id, |sub| {
                sub.expanded_gate_counts(false)
            });
            for (kind, count) in sub_counts {
                *counts.entry(kind).or_insert(0) += count * times;
            }
            if times > 0 {
                *counts.entry(PolyGateKind::Add).or_insert(0) +=
                    summed_call.num_outputs * (times - 1);
            }
        }
        counts
    }

    /// Computes the circuit depth excluding Add gates, including sub-circuits.
    ///
    /// Definition:
    /// - Inputs and the reserved constant-one gate contribute 0 to depth.
    /// - Add, Sub, SmallScalarMul gates do not increase depth: level(add) = max(level(inputs)).
    /// - Any other non-input gate increases depth by 1: level(g) = max(level(inputs)) + 1.
    /// - Sub-circuits contribute their internal non-free depth based on the call inputs.
    /// - If there are no outputs, returns 0.
    pub fn non_free_depth(&self) -> usize {
        if self.output_ids.is_empty() {
            return 0;
        }
        let input_levels = vec![0u32; self.num_input()];
        let depth_cache = DashMap::<NonFreeDepthCacheKey, Arc<[u32]>>::new();
        let output_levels =
            self.non_free_depths_with_input_levels_cached(&input_levels, &depth_cache);
        output_levels.par_iter().copied().max().unwrap_or(0) as usize
    }

    /// Compatibility helper kept for older tests that still inspect one max-depth gate-kind
    /// contribution profile.
    pub fn non_free_depth_contributions(&self) -> HashMap<PolyGateKind, usize> {
        if self.output_ids.is_empty() {
            return HashMap::new();
        }
        let input_profiles = vec![CompatNonFreeDepthProfile::default(); self.num_input()];
        let depth_cache =
            DashMap::<CompatNonFreeDepthCacheKey, Arc<[CompatNonFreeDepthProfile]>>::new();
        let output_profiles = self.compat_non_free_depth_profiles_with_input_profiles_cached(
            &input_profiles,
            &depth_cache,
        );
        output_profiles.iter().copied().max().unwrap_or_default().contributions.to_hash_map()
    }

    pub(crate) fn non_free_depths_with_input_levels_cached(
        &self,
        input_levels: &[u32],
        depth_cache: &DashMap<NonFreeDepthCacheKey, Arc<[u32]>>,
    ) -> Arc<[u32]> {
        // Execute one circuit under a concrete input-depth profile.
        //
        // The implementation mirrors the "layer, prepare, dedup, recurse, commit" structure used
        // by the extracted error simulator, but the propagated value here is just one `u32`
        // non-free depth per live wire.
        //
        // High-level algorithm:
        //
        // 1. Build grouped execution layers where regular gates, direct calls, and summed calls
        //    appear as separate scheduling items.
        // 2. Preload the current circuit inputs and the reserved constant-one gate into the live
        //    depth table.
        // 3. For each layer:
        //    - evaluate regular gates in parallel,
        //    - prepare direct and summed child-call input-depth profiles in parallel,
        //    - deduplicate identical child requests by `(child circuit identity, exact input
        //      depths)`,
        //    - recurse once per unique child request,
        //    - commit all resulting child output depths back into the parent live table.
        // 4. Release any wire whose remaining consumer count reaches zero, unless that wire is a
        //    final output of the current circuit.
        //
        // Why the cache key keeps the *exact* input-depth vector:
        //
        // Non-free depth is not purely a function of input-depth differences. A child may contain
        // a constant-only non-free branch that clamps an output to depth 1 when every input
        // arrives at depth 0, but becomes irrelevant when an input arrives at depth 5. Because of
        // that absolute-depth floor, normalizing the input vector by subtracting its minimum would
        // alias distinct correct results.
        if self.output_ids.is_empty() {
            return Arc::from(Vec::<u32>::new());
        }
        debug_assert_eq!(self.num_input(), input_levels.len());
        let cache_key = NonFreeDepthCacheKey {
            circuit_key: self as *const Self as usize,
            input_levels: input_levels.to_vec().into_boxed_slice(),
        };
        if let Some(cached) = depth_cache.get(&cache_key) {
            return Arc::clone(cached.value());
        }
        // The grouped planner gives us a dependency-safe schedule where each direct or summed
        // sub-circuit call appears once, even if the caller later consumes multiple outputs from
        // that call.
        let execution_layers = self.grouped_execution_layers();
        let mut remaining_use_count = self.non_free_depth_remaining_use_count(&execution_layers);
        let mut is_output_gate = vec![false; self.num_gates() + 1];
        for &output_id in &self.output_ids {
            is_output_gate[output_id.0] = true;
        }
        let mut live_gate_levels = vec![None; self.num_gates() + 1];
        live_gate_levels[GateId(0).0] = Some(0);
        for (input_idx, gate_id) in self.sorted_input_gate_ids().into_iter().enumerate() {
            live_gate_levels[gate_id.0] = Some(input_levels[input_idx]);
        }

        for layer in execution_layers {
            // Regular gates depend only on values from earlier layers, so their level computation
            // is embarrassingly parallel. The results are committed afterward in deterministic
            // gate-id order.
            let regular_results = layer
                .regular_gate_ids
                .par_iter()
                .map(|gate_id| {
                    (*gate_id, self.non_free_depth_regular_gate_level(*gate_id, &live_gate_levels))
                })
                .collect::<Vec<_>>();
            for (gate_id, gate_level) in regular_results {
                Self::store_non_free_depth_level(
                    &mut live_gate_levels,
                    &remaining_use_count,
                    &is_output_gate,
                    gate_id,
                    gate_level,
                );
            }
            for gate_id in &layer.regular_gate_ids {
                let gate = self.gate(*gate_id);
                Self::release_non_free_depth_inputs(
                    &mut live_gate_levels,
                    &mut remaining_use_count,
                    &is_output_gate,
                    gate.input_gates.iter().copied(),
                );
            }

            // Build all child input-depth profiles for this layer before recursing. Doing this at
            // layer granularity lets us deduplicate identical child requests, so repeated call
            // sites with the same exact profile share one cached `Arc<[u32]>`.
            let direct_prepared = layer
                .sub_circuit_call_ids
                .par_iter()
                .map(|call_id| {
                    let call =
                        self.sub_circuit_calls.get(call_id).expect("sub-circuit call missing");
                    let sub_circuit = self.registered_sub_circuit_ref(call.sub_circuit_id);
                    let child_input_levels =
                        self.with_sub_circuit_call_inputs(call, |shared_prefix, suffix| {
                            let mut levels = Vec::with_capacity(
                                batched_wire_slice_len(shared_prefix) +
                                    batched_wire_slice_len(suffix),
                            );
                            levels.extend(iter_batched_wire_gates(shared_prefix).map(|input_id| {
                                Self::live_non_free_depth_level(&live_gate_levels, input_id)
                            }));
                            levels.extend(iter_batched_wire_gates(suffix).map(|input_id| {
                                Self::live_non_free_depth_level(&live_gate_levels, input_id)
                            }));
                            levels
                        });
                    (
                        *call_id,
                        PreparedNonFreeDepthRequest {
                            key: NonFreeDepthCacheKey {
                                circuit_key: Arc::as_ptr(&sub_circuit) as usize,
                                input_levels: child_input_levels.into_boxed_slice(),
                            },
                            sub_circuit,
                        },
                    )
                })
                .collect::<Vec<_>>();
            let summed_prepared = layer
                .summed_sub_circuit_call_ids
                .par_iter()
                .map(|summed_call_id| {
                    let call = self
                        .summed_sub_circuit_calls
                        .get(summed_call_id)
                        .expect("summed sub-circuit call missing");
                    let sub_circuit = self.registered_sub_circuit_ref(call.sub_circuit_id);
                    let prepared = call
                        .call_input_set_ids
                        .par_iter()
                        .map(|input_set_id| {
                            let child_input_levels = self
                                .input_set(*input_set_id)
                                .as_ref()
                                .iter()
                                .copied()
                                .flat_map(BatchedWire::gate_ids)
                                .map(|input_id| {
                                    Self::live_non_free_depth_level(&live_gate_levels, input_id)
                                })
                                .collect::<Vec<_>>();
                            PreparedNonFreeDepthRequest {
                                key: NonFreeDepthCacheKey {
                                    circuit_key: Arc::as_ptr(&sub_circuit) as usize,
                                    input_levels: child_input_levels.into_boxed_slice(),
                                },
                                sub_circuit: Arc::clone(&sub_circuit),
                            }
                        })
                        .collect::<Vec<_>>();
                    (*summed_call_id, prepared)
                })
                .collect::<Vec<_>>();

            // `unique_requests` owns the exact set of child profiles that must be evaluated for
            // this layer. Every direct or summed call stores only the index of its shared unique
            // request.
            let mut unique_request_index_by_key = HashMap::<NonFreeDepthCacheKey, usize>::new();
            let mut unique_requests = Vec::<PreparedNonFreeDepthRequest<P>>::new();
            let direct_unique_indices = direct_prepared
                .iter()
                .map(|(_, request)| {
                    if let Some(&unique_idx) = unique_request_index_by_key.get(&request.key) {
                        unique_idx
                    } else {
                        let unique_idx = unique_requests.len();
                        unique_request_index_by_key.insert(request.key.clone(), unique_idx);
                        unique_requests.push(request.clone());
                        unique_idx
                    }
                })
                .collect::<Vec<_>>();
            let summed_unique_indices = summed_prepared
                .iter()
                .map(|(_, requests)| {
                    requests
                        .iter()
                        .map(|request| {
                            if let Some(&unique_idx) = unique_request_index_by_key.get(&request.key)
                            {
                                unique_idx
                            } else {
                                let unique_idx = unique_requests.len();
                                unique_request_index_by_key.insert(request.key.clone(), unique_idx);
                                unique_requests.push(request.clone());
                                unique_idx
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            // Recurse once per unique child profile. The recursive call hits the same concurrent
            // cache, so equal profiles shared across nested callers are also reused.
            let unique_results = unique_requests
                .par_iter()
                .map(|request| {
                    request.sub_circuit.non_free_depths_with_input_levels_cached(
                        request.key.input_levels.as_ref(),
                        depth_cache,
                    )
                })
                .collect::<Vec<_>>();

            for ((call_id, _), unique_idx) in
                direct_prepared.iter().zip(direct_unique_indices.iter())
            {
                let call = self.sub_circuit_calls.get(call_id).expect("sub-circuit call missing");
                let output_levels = &unique_results[*unique_idx];
                assert_eq!(
                    output_levels.len(),
                    call.output_gate_ids.len(),
                    "sub-circuit output arity mismatch for call {}",
                    call_id
                );
                for (output_gate_id, output_level) in
                    call.output_gate_ids.iter().copied().zip(output_levels.iter().copied())
                {
                    Self::store_non_free_depth_level(
                        &mut live_gate_levels,
                        &remaining_use_count,
                        &is_output_gate,
                        output_gate_id,
                        output_level,
                    );
                }
                Self::release_non_free_depth_inputs(
                    &mut live_gate_levels,
                    &mut remaining_use_count,
                    &is_output_gate,
                    self.non_free_depth_inputs_for_direct_call(*call_id),
                );
            }

            // For summed calls, addition is depth-free, so the parent output depth is the
            // elementwise maximum of the inner-call output depths. There is intentionally no
            // extra `+ 1` here.
            for ((summed_call_id, _), unique_indices) in
                summed_prepared.iter().zip(summed_unique_indices.iter())
            {
                let call = self
                    .summed_sub_circuit_calls
                    .get(summed_call_id)
                    .expect("summed sub-circuit call missing");
                let accumulated = Self::parallel_max_accumulate_non_free_depth_outputs(
                    *summed_call_id,
                    call,
                    unique_indices,
                    &unique_results,
                );
                for (output_gate_id, output_level) in
                    call.output_gate_ids.iter().copied().zip(accumulated.into_iter())
                {
                    Self::store_non_free_depth_level(
                        &mut live_gate_levels,
                        &remaining_use_count,
                        &is_output_gate,
                        output_gate_id,
                        output_level,
                    );
                }
                Self::release_non_free_depth_inputs(
                    &mut live_gate_levels,
                    &mut remaining_use_count,
                    &is_output_gate,
                    self.non_free_depth_inputs_for_summed_call(*summed_call_id),
                );
            }
        }

        let output_levels = Arc::<[u32]>::from(
            self.output_ids
                .par_iter()
                .map(|output_id| Self::live_non_free_depth_level(&live_gate_levels, *output_id))
                .collect::<Vec<_>>(),
        );
        depth_cache.insert(cache_key, output_levels.clone());
        output_levels
    }

    fn compat_non_free_depth_profiles_with_input_profiles_cached(
        &self,
        input_profiles: &[CompatNonFreeDepthProfile],
        depth_cache: &DashMap<CompatNonFreeDepthCacheKey, Arc<[CompatNonFreeDepthProfile]>>,
    ) -> Arc<[CompatNonFreeDepthProfile]> {
        if self.output_ids.is_empty() {
            return Arc::from(Vec::<CompatNonFreeDepthProfile>::new());
        }
        debug_assert_eq!(self.num_input(), input_profiles.len());
        let cache_key = CompatNonFreeDepthCacheKey {
            circuit_key: self as *const Self as usize,
            input_profiles: input_profiles.to_vec().into_boxed_slice(),
        };
        if let Some(cached) = depth_cache.get(&cache_key) {
            return Arc::clone(cached.value());
        }

        let mut gate_memo = HashMap::<GateId, CompatNonFreeDepthProfile>::new();
        gate_memo.insert(GateId(0), CompatNonFreeDepthProfile::default());
        for (input_idx, gate_id) in self.sorted_input_gate_ids().into_iter().enumerate() {
            gate_memo.insert(gate_id, input_profiles[input_idx]);
        }
        let mut direct_call_memo = HashMap::<usize, Arc<[CompatNonFreeDepthProfile]>>::new();
        let mut summed_call_memo = HashMap::<usize, Arc<[CompatNonFreeDepthProfile]>>::new();
        let output_profiles = Arc::<[CompatNonFreeDepthProfile]>::from(
            self.output_ids
                .iter()
                .copied()
                .map(|output_id| {
                    self.compat_non_free_depth_profile_for_gate(
                        output_id,
                        depth_cache,
                        &mut gate_memo,
                        &mut direct_call_memo,
                        &mut summed_call_memo,
                    )
                })
                .collect::<Vec<_>>(),
        );
        depth_cache.insert(cache_key, output_profiles.clone());
        output_profiles
    }

    fn compat_non_free_depth_profile_for_gate(
        &self,
        gate_id: GateId,
        depth_cache: &DashMap<CompatNonFreeDepthCacheKey, Arc<[CompatNonFreeDepthProfile]>>,
        gate_memo: &mut HashMap<GateId, CompatNonFreeDepthProfile>,
        direct_call_memo: &mut HashMap<usize, Arc<[CompatNonFreeDepthProfile]>>,
        summed_call_memo: &mut HashMap<usize, Arc<[CompatNonFreeDepthProfile]>>,
    ) -> CompatNonFreeDepthProfile {
        if let Some(profile) = gate_memo.get(&gate_id).copied() {
            return profile;
        }

        let profile = match &self.gate(gate_id).gate_type {
            PolyGateType::Add | PolyGateType::Sub | PolyGateType::SmallScalarMul { .. } => self
                .gate(gate_id)
                .input_gates
                .iter()
                .copied()
                .map(|input_id| {
                    self.compat_non_free_depth_profile_for_gate(
                        input_id,
                        depth_cache,
                        gate_memo,
                        direct_call_memo,
                        summed_call_memo,
                    )
                })
                .max()
                .unwrap_or_default(),
            PolyGateType::LargeScalarMul { .. } |
            PolyGateType::Mul |
            PolyGateType::PubLut { .. } |
            PolyGateType::SlotTransfer { .. } => self
                .gate(gate_id)
                .input_gates
                .iter()
                .copied()
                .map(|input_id| {
                    self.compat_non_free_depth_profile_for_gate(
                        input_id,
                        depth_cache,
                        gate_memo,
                        direct_call_memo,
                        summed_call_memo,
                    )
                })
                .max()
                .unwrap_or_default()
                .incremented(self.gate(gate_id).gate_type.kind()),
            PolyGateType::SubCircuitOutput { call_id, output_idx, .. } => self
                .compat_non_free_depth_direct_call_outputs(
                    *call_id,
                    depth_cache,
                    gate_memo,
                    direct_call_memo,
                    summed_call_memo,
                )
                .get(*output_idx)
                .copied()
                .unwrap_or_default(),
            PolyGateType::SummedSubCircuitOutput { summed_call_id, output_idx, .. } => self
                .compat_non_free_depth_summed_call_outputs(
                    *summed_call_id,
                    depth_cache,
                    gate_memo,
                    direct_call_memo,
                    summed_call_memo,
                )
                .get(*output_idx)
                .copied()
                .unwrap_or_default(),
            PolyGateType::Input => CompatNonFreeDepthProfile::default(),
        };

        gate_memo.insert(gate_id, profile);
        profile
    }

    fn compat_non_free_depth_direct_call_outputs(
        &self,
        call_id: usize,
        depth_cache: &DashMap<CompatNonFreeDepthCacheKey, Arc<[CompatNonFreeDepthProfile]>>,
        gate_memo: &mut HashMap<GateId, CompatNonFreeDepthProfile>,
        direct_call_memo: &mut HashMap<usize, Arc<[CompatNonFreeDepthProfile]>>,
        summed_call_memo: &mut HashMap<usize, Arc<[CompatNonFreeDepthProfile]>>,
    ) -> Arc<[CompatNonFreeDepthProfile]> {
        if let Some(cached) = direct_call_memo.get(&call_id) {
            return Arc::clone(cached);
        }

        let call = self.sub_circuit_calls.get(&call_id).expect("sub-circuit call missing");
        let sub_circuit = self.registered_sub_circuit_ref(call.sub_circuit_id);
        let child_input_profiles =
            self.with_sub_circuit_call_inputs(call, |shared_prefix, suffix| {
                iter_batched_wire_gates(shared_prefix)
                    .chain(iter_batched_wire_gates(suffix))
                    .map(|input_id| {
                        self.compat_non_free_depth_profile_for_gate(
                            input_id,
                            depth_cache,
                            gate_memo,
                            direct_call_memo,
                            summed_call_memo,
                        )
                    })
                    .collect::<Vec<_>>()
            });
        let outputs = sub_circuit.compat_non_free_depth_profiles_with_input_profiles_cached(
            &child_input_profiles,
            depth_cache,
        );
        direct_call_memo.insert(call_id, outputs.clone());
        outputs
    }

    fn compat_non_free_depth_summed_call_outputs(
        &self,
        summed_call_id: usize,
        depth_cache: &DashMap<CompatNonFreeDepthCacheKey, Arc<[CompatNonFreeDepthProfile]>>,
        gate_memo: &mut HashMap<GateId, CompatNonFreeDepthProfile>,
        direct_call_memo: &mut HashMap<usize, Arc<[CompatNonFreeDepthProfile]>>,
        summed_call_memo: &mut HashMap<usize, Arc<[CompatNonFreeDepthProfile]>>,
    ) -> Arc<[CompatNonFreeDepthProfile]> {
        if let Some(cached) = summed_call_memo.get(&summed_call_id) {
            return Arc::clone(cached);
        }

        let call = self
            .summed_sub_circuit_calls
            .get(&summed_call_id)
            .expect("summed sub-circuit call missing");
        let sub_circuit = self.registered_sub_circuit_ref(call.sub_circuit_id);
        let mut accumulated = vec![CompatNonFreeDepthProfile::default(); call.num_outputs];
        for input_set_id in &call.call_input_set_ids {
            let child_input_profiles = self
                .input_set(*input_set_id)
                .as_ref()
                .iter()
                .copied()
                .flat_map(BatchedWire::gate_ids)
                .map(|input_id| {
                    self.compat_non_free_depth_profile_for_gate(
                        input_id,
                        depth_cache,
                        gate_memo,
                        direct_call_memo,
                        summed_call_memo,
                    )
                })
                .collect::<Vec<_>>();
            let output_profiles = sub_circuit
                .compat_non_free_depth_profiles_with_input_profiles_cached(
                    &child_input_profiles,
                    depth_cache,
                );
            assert_eq!(
                output_profiles.len(),
                call.output_gate_ids.len(),
                "summed sub-circuit output arity mismatch for call {}",
                summed_call_id
            );
            for (acc_profile, output_profile) in accumulated.iter_mut().zip(output_profiles.iter())
            {
                *acc_profile = (*acc_profile).max(*output_profile);
            }
        }
        let outputs = Arc::<[CompatNonFreeDepthProfile]>::from(accumulated);
        summed_call_memo.insert(summed_call_id, outputs.clone());
        outputs
    }

    /// Compute one regular gate's non-free depth from its already-live inputs.
    ///
    /// This is the core local rule of the whole algorithm:
    ///
    /// - `Add`, `Sub`, and `SmallScalarMul` are depth-free, so they just forward the maximum input
    ///   depth.
    /// - `Mul`, `LargeScalarMul`, `PubLut`, and `SlotTransfer` are non-free, so they forward the
    ///   maximum input depth plus one.
    fn non_free_depth_regular_gate_level(
        &self,
        gate_id: GateId,
        live_gate_levels: &[Option<u32>],
    ) -> u32 {
        let gate = self.gate(gate_id);
        match &gate.gate_type {
            PolyGateType::Add | PolyGateType::Sub | PolyGateType::SmallScalarMul { .. } => gate
                .input_gates
                .iter()
                .map(|input_id| Self::live_non_free_depth_level(live_gate_levels, *input_id))
                .max()
                .unwrap_or(0),
            PolyGateType::LargeScalarMul { .. } |
            PolyGateType::Mul |
            PolyGateType::PubLut { .. } |
            PolyGateType::SlotTransfer { .. } => {
                gate.input_gates
                    .iter()
                    .map(|input_id| Self::live_non_free_depth_level(live_gate_levels, *input_id))
                    .max()
                    .unwrap_or(0) +
                    1
            }
            PolyGateType::Input => {
                panic!("input gate {gate_id} should not appear in grouped execution layers")
            }
            PolyGateType::SubCircuitOutput { .. } | PolyGateType::SummedSubCircuitOutput { .. } => {
                panic!("sub-circuit output gate {gate_id} should not appear as a regular gate")
            }
        }
    }

    fn non_free_depth_remaining_use_count(
        &self,
        execution_layers: &[GroupedCallExecutionLayer],
    ) -> Vec<u32> {
        // Count consumers from the grouped execution plan rather than from raw placeholder gates.
        // That matches the way values are committed and released during grouped execution: one
        // direct or summed call consumes all of its inputs in one scheduling step.
        let mut counts = vec![0u32; self.num_gates() + 1];
        for layer in execution_layers {
            for gate_id in &layer.regular_gate_ids {
                let gate = self.gate(*gate_id);
                for input_id in gate.input_gates.iter().copied() {
                    counts[input_id.0] += 1;
                }
            }
            for call_id in &layer.sub_circuit_call_ids {
                self.with_sub_circuit_call_inputs_by_id(*call_id, |shared_prefix, suffix| {
                    for input_id in iter_batched_wire_gates(shared_prefix) {
                        counts[input_id.0] += 1;
                    }
                    for input_id in iter_batched_wire_gates(suffix) {
                        counts[input_id.0] += 1;
                    }
                });
            }
            for summed_call_id in &layer.summed_sub_circuit_call_ids {
                self.for_each_summed_sub_circuit_call_input(*summed_call_id, |input_id| {
                    counts[input_id.0] += 1;
                });
            }
        }
        counts
    }

    fn live_non_free_depth_level(live_gate_levels: &[Option<u32>], gate_id: GateId) -> u32 {
        live_gate_levels[gate_id.0]
            .unwrap_or_else(|| panic!("non-free depth level missing for gate {gate_id}"))
    }

    fn store_non_free_depth_level(
        live_gate_levels: &mut [Option<u32>],
        remaining_use_count: &[u32],
        is_output_gate: &[bool],
        gate_id: GateId,
        gate_level: u32,
    ) {
        // Dead temporaries are never stored. This keeps the live table sparse in practice even
        // though we use a dense `Vec<Option<u32>>` for fast indexed access.
        if remaining_use_count[gate_id.0] > 0 || is_output_gate[gate_id.0] {
            live_gate_levels[gate_id.0] = Some(gate_level);
        }
    }

    fn release_non_free_depth_inputs<I>(
        live_gate_levels: &mut [Option<u32>],
        remaining_use_count: &mut [u32],
        is_output_gate: &[bool],
        input_ids: I,
    ) where
        I: IntoIterator<Item = GateId>,
    {
        // Release happens after the whole scheduling item finishes, never mid-item. That is why
        // direct and summed calls first commit all outputs and only then release all consumed
        // inputs.
        for input_id in input_ids {
            let remaining_uses = &mut remaining_use_count[input_id.0];
            if *remaining_uses == 0 {
                continue;
            }
            *remaining_uses -= 1;
            if *remaining_uses == 0 && !is_output_gate[input_id.0] {
                live_gate_levels[input_id.0] = None;
            }
        }
    }

    fn non_free_depth_inputs_for_direct_call(&self, call_id: usize) -> Vec<GateId> {
        self.with_sub_circuit_call_by_id(call_id, |_sub_id, _bindings, shared_prefix, suffix, _| {
            iter_batched_wire_gates(shared_prefix).chain(iter_batched_wire_gates(suffix)).collect()
        })
    }

    fn non_free_depth_inputs_for_summed_call(&self, summed_call_id: usize) -> Vec<GateId> {
        let mut inputs = Vec::new();
        self.for_each_summed_sub_circuit_call_input(summed_call_id, |input_id| {
            inputs.push(input_id)
        });
        inputs
    }

    /// Reduce the outputs of a summed sub-circuit call with elementwise `max` in parallel.
    ///
    /// The outer summation is depth-free, so the combined output depth is the maximum depth seen
    /// among the inner calls for each output position. The reduction is independent per inner call,
    /// which makes it safe to parallelize with a local `Vec<u32>` accumulator per worker.
    fn parallel_max_accumulate_non_free_depth_outputs(
        summed_call_id: usize,
        call: &SummedSubCircuitCall,
        unique_indices: &[usize],
        unique_results: &[Arc<[u32]>],
    ) -> Vec<u32> {
        unique_indices
            .par_iter()
            .fold(
                || vec![0u32; call.num_outputs],
                |mut acc, unique_idx| {
                    let output_levels = &unique_results[*unique_idx];
                    assert_eq!(
                        output_levels.len(),
                        call.output_gate_ids.len(),
                        "summed sub-circuit output arity mismatch for call {}",
                        summed_call_id
                    );
                    for (acc_level, level) in acc.iter_mut().zip(output_levels.iter()) {
                        *acc_level = (*acc_level).max(*level);
                    }
                    acc
                },
            )
            .reduce(
                || vec![0u32; call.num_outputs],
                |mut left, right| {
                    for (left_level, right_level) in left.iter_mut().zip(right.iter()) {
                        *left_level = (*left_level).max(*right_level);
                    }
                    left
                },
            )
    }

    pub(crate) fn gate_dependency_input_count(&self, gate: &PolyGate) -> usize {
        match &gate.gate_type {
            PolyGateType::SubCircuitOutput { call_id, .. } => self
                .sub_circuit_calls
                .get(call_id)
                .map(|call| self.sub_circuit_call_input_len(call))
                .expect("sub-circuit call missing"),
            PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } => self
                .summed_sub_circuit_calls
                .get(summed_call_id)
                .expect("summed sub-circuit call missing")
                .call_input_set_ids
                .iter()
                .map(|input_set_id| batched_wire_slice_len(self.input_set(*input_set_id).as_ref()))
                .sum(),
            _ => gate.input_gates.len(),
        }
    }

    pub(crate) fn for_each_gate_dependency_input(
        &self,
        gate: &PolyGate,
        mut f: impl FnMut(GateId),
    ) {
        match &gate.gate_type {
            PolyGateType::SubCircuitOutput { call_id, .. } => {
                let call = self.sub_circuit_calls.get(call_id).expect("sub-circuit call missing");
                self.with_sub_circuit_call_inputs(call, |shared_prefix, suffix| {
                    for input_id in iter_batched_wire_gates(shared_prefix) {
                        f(input_id);
                    }
                    for input_id in iter_batched_wire_gates(suffix) {
                        f(input_id);
                    }
                });
            }
            PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } => {
                let call = self
                    .summed_sub_circuit_calls
                    .get(summed_call_id)
                    .expect("summed sub-circuit call missing");
                for input_set_id in &call.call_input_set_ids {
                    for input_id in iter_batched_wire_gates(self.input_set(*input_set_id).as_ref())
                    {
                        f(input_id);
                    }
                }
            }
            _ => {
                for &input_id in &gate.input_gates {
                    f(input_id);
                }
            }
        }
    }

    fn topological_order(&self) -> Vec<GateId> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        let mut stack = Vec::new();
        for &output_gate in &self.output_ids {
            if visited.insert(output_gate) {
                stack.push((output_gate, 0));
            }
        }

        while let Some((node, child_idx)) = stack.pop() {
            let gate = self.gates.get(&node).expect("gate not found");
            let dependency_inputs = {
                let mut deps = Vec::with_capacity(self.gate_dependency_input_count(gate));
                self.for_each_gate_dependency_input(gate, |input_id| deps.push(input_id));
                deps
            };

            if child_idx < dependency_inputs.len() {
                stack.push((node, child_idx + 1));
                let child = dependency_inputs[child_idx];
                if visited.insert(child) {
                    stack.push((child, 0));
                }
            } else {
                order.push(node);
            }
        }

        order
    }

    pub(crate) fn compute_levels(&self) -> Vec<Vec<GateId>> {
        let mut gate_levels: HashMap<GateId, usize> = HashMap::new();
        let mut levels: Vec<Vec<GateId>> = vec![];
        let orders = self.topological_order();
        for gate_id in orders {
            let gate = self.gates.get(&gate_id).expect("gate not found");
            let dependency_count = self.gate_dependency_input_count(gate);
            if dependency_count == 0 {
                gate_levels.insert(gate_id, 0);
                if levels.is_empty() {
                    levels.push(vec![]);
                }
                levels[0].push(gate_id);
                continue;
            }
            let mut max_input_level: Option<usize> = None;
            self.for_each_gate_dependency_input(gate, |input_id| {
                let level = gate_levels[&input_id];
                max_input_level = Some(max_input_level.map_or(level, |curr| curr.max(level)));
            });
            let max_input_level =
                max_input_level.expect("gate has dependencies but max() returned None");
            let level = max_input_level + 1;
            gate_levels.insert(gate_id, level);
            if levels.len() <= level {
                levels.resize(level + 1, vec![]);
            }
            levels[level].push(gate_id);
        }
        levels
    }

    fn grouped_execution_node_for_gate(
        &self,
        gate_id: GateId,
    ) -> Option<GroupedCallExecutionNodeId> {
        let gate = self.gate(gate_id);
        match &gate.gate_type {
            PolyGateType::Input => None,
            PolyGateType::SubCircuitOutput { call_id, .. } => {
                Some(GroupedCallExecutionNodeId::SubCircuitCall(*call_id))
            }
            PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } => {
                Some(GroupedCallExecutionNodeId::SummedSubCircuitCall(*summed_call_id))
            }
            _ => Some(GroupedCallExecutionNodeId::Regular(gate_id)),
        }
    }

    fn populate_grouped_execution_input_set_max_level(
        &self,
        input_set_id: usize,
        node_levels: &mut HashMap<GroupedCallExecutionNodeId, usize>,
        input_set_levels: &mut HashMap<usize, Option<usize>>,
        reachable_inputs: &mut BTreeSet<GateId>,
        visiting: &mut HashSet<GroupedCallExecutionNodeId>,
        levels: &mut Vec<GroupedCallExecutionLayer>,
    ) -> Option<usize> {
        // Input sets are interned and heavily reused by direct-call shared prefixes and summed
        // calls. Memoizing their producer level avoids rescanning the same batched wires for every
        // consumer call node.
        if let Some(&max_level) = input_set_levels.get(&input_set_id) {
            return max_level;
        }
        let mut max_dependency_level: Option<usize> = None;
        let mut has_input_dependency = false;
        for input_id in iter_batched_wire_gates(self.input_set(input_set_id).as_ref()) {
            if let Some(dep_node) = self.grouped_execution_node_for_gate(input_id) {
                let dep_level = self.populate_grouped_execution_node_level(
                    dep_node,
                    node_levels,
                    input_set_levels,
                    reachable_inputs,
                    visiting,
                    levels,
                );
                max_dependency_level = Some(match max_dependency_level {
                    Some(curr) => curr.max(dep_level),
                    None => dep_level,
                });
            } else {
                has_input_dependency = true;
                self.record_reachable_input(input_id, reachable_inputs);
            }
        }
        let max_dependency_level = if has_input_dependency {
            Some(max_dependency_level.map_or(0, |dep_level| dep_level))
        } else {
            max_dependency_level
        };
        input_set_levels.insert(input_set_id, max_dependency_level);
        max_dependency_level
    }

    fn populate_grouped_execution_node_level(
        &self,
        node: GroupedCallExecutionNodeId,
        node_levels: &mut HashMap<GroupedCallExecutionNodeId, usize>,
        input_set_levels: &mut HashMap<usize, Option<usize>>,
        reachable_inputs: &mut BTreeSet<GateId>,
        visiting: &mut HashSet<GroupedCallExecutionNodeId>,
        levels: &mut Vec<GroupedCallExecutionLayer>,
    ) -> usize {
        // This is a memoized DFS over the grouped scheduling graph. The result is analogous to a
        // topological level: each node is placed at `max(dependency levels) + 1`, while a node
        // that depends only on already-preloaded circuit inputs starts at level 0.
        if let Some(&level) = node_levels.get(&node) {
            return level;
        }
        assert!(visiting.insert(node), "cycle detected while computing grouped execution layers");
        let mut max_dependency_level: Option<usize> = None;
        let mut has_input_dependency = false;
        let mut update_dep_level = |dep_level: usize| {
            max_dependency_level = Some(match max_dependency_level {
                Some(curr) => curr.max(dep_level),
                None => dep_level,
            });
        };
        match node {
            GroupedCallExecutionNodeId::Regular(gate_id) => {
                for &input_id in &self.gate(gate_id).input_gates {
                    if let Some(dep_node) = self.grouped_execution_node_for_gate(input_id) {
                        let dep_level = self.populate_grouped_execution_node_level(
                            dep_node,
                            node_levels,
                            input_set_levels,
                            reachable_inputs,
                            visiting,
                            levels,
                        );
                        update_dep_level(dep_level);
                    } else {
                        has_input_dependency = true;
                        self.record_reachable_input(input_id, reachable_inputs);
                    }
                }
            }
            GroupedCallExecutionNodeId::SubCircuitCall(call_id) => {
                let call = self.sub_circuit_calls.get(&call_id).expect("sub-circuit call missing");
                if let Some(input_set_id) = call.shared_input_prefix_set_id {
                    if let Some(shared_prefix_level) = self
                        .populate_grouped_execution_input_set_max_level(
                            input_set_id,
                            node_levels,
                            input_set_levels,
                            reachable_inputs,
                            visiting,
                            levels,
                        )
                    {
                        update_dep_level(shared_prefix_level);
                    }
                }
                for input_id in iter_batched_wire_gates(&call.input_suffix) {
                    if let Some(dep_node) = self.grouped_execution_node_for_gate(input_id) {
                        let dep_level = self.populate_grouped_execution_node_level(
                            dep_node,
                            node_levels,
                            input_set_levels,
                            reachable_inputs,
                            visiting,
                            levels,
                        );
                        update_dep_level(dep_level);
                    } else {
                        has_input_dependency = true;
                        self.record_reachable_input(input_id, reachable_inputs);
                    }
                }
            }
            GroupedCallExecutionNodeId::SummedSubCircuitCall(summed_call_id) => {
                let call = self
                    .summed_sub_circuit_calls
                    .get(&summed_call_id)
                    .expect("summed sub-circuit call missing");
                for &input_set_id in &call.call_input_set_ids {
                    if let Some(input_set_level) = self
                        .populate_grouped_execution_input_set_max_level(
                            input_set_id,
                            node_levels,
                            input_set_levels,
                            reachable_inputs,
                            visiting,
                            levels,
                        )
                    {
                        update_dep_level(input_set_level);
                    }
                }
            }
        }
        visiting.remove(&node);
        let max_dependency_level = if has_input_dependency {
            Some(max_dependency_level.map_or(0, |dep_level| dep_level))
        } else {
            max_dependency_level
        };
        let level = max_dependency_level.map_or(0, |dep_level| dep_level + 1);
        if levels.len() <= level {
            levels.resize(level + 1, GroupedCallExecutionLayer::default());
        }
        match node {
            GroupedCallExecutionNodeId::Regular(gate_id) => {
                levels[level].regular_gate_ids.push(gate_id)
            }
            GroupedCallExecutionNodeId::SubCircuitCall(call_id) => {
                levels[level].sub_circuit_call_ids.push(call_id)
            }
            GroupedCallExecutionNodeId::SummedSubCircuitCall(summed_call_id) => {
                levels[level].summed_sub_circuit_call_ids.push(summed_call_id)
            }
        }
        node_levels.insert(node, level);
        level
    }

    fn record_reachable_input(&self, gate_id: GateId, reachable_inputs: &mut BTreeSet<GateId>) {
        if gate_id.0 != 0 && matches!(self.gate(gate_id).gate_type, PolyGateType::Input) {
            reachable_inputs.insert(gate_id);
        }
    }

    fn build_grouped_execution_plan(&self) -> GroupedExecutionPlan {
        let mut node_levels: HashMap<GroupedCallExecutionNodeId, usize> = HashMap::new();
        let mut input_set_levels: HashMap<usize, Option<usize>> = HashMap::new();
        let mut levels = Vec::<GroupedCallExecutionLayer>::new();
        let mut reachable_inputs = BTreeSet::<GateId>::new();
        let mut visiting = HashSet::new();
        for &output_gate in &self.output_ids {
            let Some(node) = self.grouped_execution_node_for_gate(output_gate) else {
                self.record_reachable_input(output_gate, &mut reachable_inputs);
                continue;
            };
            self.populate_grouped_execution_node_level(
                node,
                &mut node_levels,
                &mut input_set_levels,
                &mut reachable_inputs,
                &mut visiting,
                &mut levels,
            );
        }
        GroupedExecutionPlan {
            layers: levels,
            reachable_input_gate_ids: reachable_inputs.into_iter().collect(),
        }
    }

    /// Build the grouped execution schedule shared by `non_free_depth()` and `eval_error`.
    ///
    /// The generic circuit-level planner lives here because the grouping itself is purely
    /// structural: it depends only on the wiring graph and sub-circuit call layout, not on the
    /// propagated value type. `non_free_depth()` uses the resulting layers to propagate `u32`
    /// depths, while `eval_error` reuses the same grouping to propagate symbolic or concrete error
    /// values.
    pub(crate) fn grouped_execution_layers(&self) -> Vec<GroupedCallExecutionLayer> {
        self.build_grouped_execution_plan().layers
    }

    pub(crate) fn grouped_execution_plan(&self) -> GroupedExecutionPlan {
        self.build_grouped_execution_plan()
    }

    /// Returns the circuit depth defined as the maximum level index among
    /// all gates required to compute the outputs.
    ///
    /// - Inputs and constant-one gate reside at level 0.
    /// - Each non-input gate is assigned level = max(input levels) + 1.
    /// - If there are no outputs, depth is 0.
    pub fn depth(&self) -> usize {
        let levels = self.compute_levels();
        if levels.is_empty() { 0 } else { levels.len() - 1 }
    }
}
