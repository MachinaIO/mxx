use super::*;
use std::path::Path;

impl<P: Poly> PolyCircuit<P> {
    pub(crate) fn inherit_shared_registries(
        &mut self,
        lookup_registry: Arc<LookupRegistry<P>>,
        binding_registry: Arc<BindingRegistry>,
        input_set_registry: Arc<InputSetRegistry>,
        sub_circuit_registry: Arc<SubCircuitRegistry<P>>,
    ) {
        if !Arc::ptr_eq(&self.lookup_registry, &lookup_registry) {
            if !self.lookup_registry.is_empty() {
                panic!("sub-circuit may not register lookup tables");
            }
            self.lookup_registry = Arc::clone(&lookup_registry);
        }
        if !Arc::ptr_eq(&self.binding_registry, &binding_registry) {
            if !self.binding_registry.is_empty() {
                for call in self.sub_circuit_calls.values_mut() {
                    let bindings = self.binding_registry.get(call.binding_set_id);
                    call.binding_set_id = binding_registry.register_arc(bindings);
                }
                for call in self.summed_sub_circuit_calls.values_mut() {
                    call.call_binding_set_ids.iter_mut().for_each(|binding_set_id| {
                        let bindings = self.binding_registry.get(*binding_set_id);
                        *binding_set_id = binding_registry.register_arc(bindings);
                    });
                }
            }
            self.binding_registry = Arc::clone(&binding_registry);
        }
        if !Arc::ptr_eq(&self.input_set_registry, &input_set_registry) {
            if !self.input_set_registry.is_empty() {
                for call in self.sub_circuit_calls.values_mut() {
                    if let Some(input_set_id) = call.shared_input_prefix_set_id.as_mut() {
                        let input_ids = self.input_set_registry.get(*input_set_id);
                        *input_set_id = input_set_registry.register_arc(input_ids);
                    }
                }
                for call in self.summed_sub_circuit_calls.values_mut() {
                    call.call_input_set_ids.iter_mut().for_each(|input_set_id| {
                        let input_ids = self.input_set_registry.get(*input_set_id);
                        *input_set_id = input_set_registry.register_arc(input_ids);
                    });
                }
            }
            self.input_set_registry = Arc::clone(&input_set_registry);
        }
        if !Arc::ptr_eq(&self.sub_circuit_registry, &sub_circuit_registry) {
            let previous_registry = Arc::clone(&self.sub_circuit_registry);
            let previous_registry_key = Arc::as_ptr(&previous_registry) as usize;
            let mut remapped_sub_circuit_ids = HashMap::new();
            for call in self.sub_circuit_calls.values_mut() {
                call.sub_circuit_id = Self::import_sub_circuit_to_registry(
                    &previous_registry,
                    previous_registry_key,
                    &lookup_registry,
                    &binding_registry,
                    &input_set_registry,
                    &sub_circuit_registry,
                    call.sub_circuit_id,
                    &mut remapped_sub_circuit_ids,
                );
            }
            for call in self.summed_sub_circuit_calls.values_mut() {
                call.sub_circuit_id = Self::import_sub_circuit_to_registry(
                    &previous_registry,
                    previous_registry_key,
                    &lookup_registry,
                    &binding_registry,
                    &input_set_registry,
                    &sub_circuit_registry,
                    call.sub_circuit_id,
                    &mut remapped_sub_circuit_ids,
                );
            }
            self.sub_circuit_registry = Arc::clone(&sub_circuit_registry);
        }
        self.allow_register_lookup = false;
    }

    pub(crate) fn registry_handles(&self) -> PolyCircuitRegistryHandles<P> {
        PolyCircuitRegistryHandles {
            lookup_registry: Arc::clone(&self.lookup_registry),
            binding_registry: Arc::clone(&self.binding_registry),
            input_set_registry: Arc::clone(&self.input_set_registry),
            sub_circuit_registry: Arc::clone(&self.sub_circuit_registry),
        }
    }

    pub(crate) fn inherit_registry_handles(&mut self, handles: &PolyCircuitRegistryHandles<P>) {
        self.inherit_shared_registries(
            Arc::clone(&handles.lookup_registry),
            Arc::clone(&handles.binding_registry),
            Arc::clone(&handles.input_set_registry),
            Arc::clone(&handles.sub_circuit_registry),
        );
    }

    fn import_sub_circuit_to_registry(
        previous_registry: &Arc<SubCircuitRegistry<P>>,
        previous_registry_key: usize,
        lookup_registry: &Arc<LookupRegistry<P>>,
        binding_registry: &Arc<BindingRegistry>,
        input_set_registry: &Arc<InputSetRegistry>,
        target_registry: &Arc<SubCircuitRegistry<P>>,
        circuit_id: usize,
        remapped_sub_circuit_ids: &mut HashMap<(usize, usize), usize>,
    ) -> usize {
        if Arc::ptr_eq(previous_registry, target_registry) {
            return circuit_id;
        }
        let remap_key = (previous_registry_key, circuit_id);
        if let Some(&remapped_id) = remapped_sub_circuit_ids.get(&remap_key) {
            return remapped_id;
        }

        let mut sub_circuit = previous_registry.get(circuit_id);
        Arc::make_mut(&mut sub_circuit).inherit_shared_registries(
            Arc::clone(lookup_registry),
            Arc::clone(binding_registry),
            Arc::clone(input_set_registry),
            Arc::clone(target_registry),
        );
        let remapped_id = target_registry.register_arc(sub_circuit);
        remapped_sub_circuit_ids.insert(remap_key, remapped_id);
        remapped_id
    }

    pub(crate) fn inherit_registries_from_parent(&mut self, parent: &Self) {
        self.inherit_shared_registries(
            Arc::clone(&parent.lookup_registry),
            Arc::clone(&parent.binding_registry),
            Arc::clone(&parent.input_set_registry),
            Arc::clone(&parent.sub_circuit_registry),
        );
    }

    pub fn fresh_sub_circuit(&self) -> Self {
        let mut circuit = Self::new();
        circuit.inherit_registries_from_parent(self);
        circuit
    }

    pub(crate) fn with_sub_circuit<R>(&self, circuit_id: usize, f: impl FnOnce(&Self) -> R) -> R {
        let stored = self.sub_circuit_registry.get(circuit_id);
        f(stored.as_ref())
    }

    pub(crate) fn sub_circuit_num_output(&self, circuit_id: usize) -> usize {
        let stored = self.sub_circuit_registry.get(circuit_id);
        stored.as_ref().num_output()
    }

    pub(crate) fn lookup_table(&self, lut_id: usize) -> Arc<PublicLut<P>> {
        self.lookup_registry.lookups.get(&lut_id).expect("lookup table missing").clone()
    }

    pub(crate) fn binding_set(&self, binding_set_id: usize) -> Arc<[SubCircuitParamValue]> {
        self.binding_registry.get(binding_set_id)
    }

    pub(crate) fn intern_binding_set(&self, bindings: &[SubCircuitParamValue]) -> usize {
        self.binding_registry.register(bindings)
    }

    pub(crate) fn input_set(&self, input_set_id: usize) -> Arc<[BatchedWire]> {
        self.input_set_registry.get(input_set_id)
    }

    pub(crate) fn intern_input_set<I, W>(&self, input_ids: I) -> usize
    where
        I: IntoIterator<Item = W>,
        W: Into<BatchedWire>,
    {
        let batches = input_ids.into_iter().map(Into::into).collect::<Vec<_>>();
        self.input_set_registry.register(&batches)
    }

    pub(crate) fn sub_circuit_call_input_len(&self, call: &SubCircuitCall) -> usize {
        call.shared_input_prefix_set_id
            .map(|input_set_id| batched_wire_slice_len(self.input_set(input_set_id).as_ref()))
            .unwrap_or(0) +
            batched_wire_slice_len(&call.input_suffix)
    }

    pub(crate) fn with_sub_circuit_call_inputs<T>(
        &self,
        call: &SubCircuitCall,
        f: impl FnOnce(&[BatchedWire], &[BatchedWire]) -> T,
    ) -> T {
        if let Some(input_set_id) = call.shared_input_prefix_set_id {
            let shared_prefix = self.input_set(input_set_id);
            f(shared_prefix.as_ref(), &call.input_suffix)
        } else {
            f(&[], &call.input_suffix)
        }
    }

    pub(crate) fn with_sub_circuit_call_inputs_by_id<T>(
        &self,
        call_id: usize,
        f: impl FnOnce(&[BatchedWire], &[BatchedWire]) -> T,
    ) -> T {
        let call = self.sub_circuit_calls.get(&call_id).expect("sub-circuit call missing");
        self.with_sub_circuit_call_inputs(call, f)
    }

    pub(crate) fn with_sub_circuit_call_by_id<T>(
        &self,
        call_id: usize,
        f: impl FnOnce(
            usize,
            Arc<[SubCircuitParamValue]>,
            &[BatchedWire],
            &[BatchedWire],
            &[GateId],
        ) -> T,
    ) -> T {
        let call = self.sub_circuit_calls.get(&call_id).expect("sub-circuit call missing");
        let param_bindings = self.binding_set(call.binding_set_id);
        self.with_sub_circuit_call_inputs(call, |shared_prefix, suffix| {
            f(call.sub_circuit_id, param_bindings, shared_prefix, suffix, &call.output_gate_ids)
        })
    }

    pub(crate) fn sub_circuit_call_shared_prefix_set_id(&self, call_id: usize) -> Option<usize> {
        self.sub_circuit_calls
            .get(&call_id)
            .expect("sub-circuit call missing")
            .shared_input_prefix_set_id
    }

    pub(crate) fn for_each_summed_sub_circuit_call_input(
        &self,
        summed_call_id: usize,
        mut f: impl FnMut(GateId),
    ) {
        let call = self
            .summed_sub_circuit_calls
            .get(&summed_call_id)
            .expect("summed sub-circuit call missing");
        for input_set_id in &call.call_input_set_ids {
            for input_id in iter_batched_wire_gates(self.input_set(*input_set_id).as_ref()) {
                f(input_id);
            }
        }
    }

    fn collect_sub_circuit_call_inputs(&self, call: &SubCircuitCall) -> Vec<BatchedWire> {
        self.with_sub_circuit_call_inputs(call, |shared_prefix, suffix| {
            let mut inputs = Vec::with_capacity(shared_prefix.len() + suffix.len());
            inputs.extend_from_slice(shared_prefix);
            inputs.extend_from_slice(suffix);
            inputs
        })
    }

    pub(crate) fn sub_circuit_call_info(&self, call_id: usize) -> SubCircuitCallInfo {
        let call = self.sub_circuit_calls.get(&call_id).expect("sub-circuit call missing");
        SubCircuitCallInfo {
            sub_circuit_id: call.sub_circuit_id,
            inputs: self.collect_sub_circuit_call_inputs(call),
            param_bindings: self.binding_set(call.binding_set_id),
            input_max_plaintext_norm_ranges: call.input_max_plaintext_norm_ranges.clone(),
            output_gate_ids: call.output_gate_ids.clone(),
        }
    }

    pub(crate) fn summed_sub_circuit_call_info(
        &self,
        summed_call_id: usize,
    ) -> SummedSubCircuitCallInfo {
        let call = self
            .summed_sub_circuit_calls
            .get(&summed_call_id)
            .expect("summed sub-circuit call missing");
        let (call_inputs, param_bindings) = rayon::join(
            || {
                call.call_input_set_ids
                    .iter()
                    .map(|input_set_id| self.input_set(*input_set_id).as_ref().to_vec())
                    .collect::<Vec<_>>()
            },
            || {
                call.call_binding_set_ids
                    .iter()
                    .map(|binding_set_id| self.binding_set(*binding_set_id))
                    .collect::<Vec<_>>()
            },
        );
        SummedSubCircuitCallInfo {
            sub_circuit_id: call.sub_circuit_id,
            call_inputs,
            param_bindings,
            input_max_plaintext_norm_ranges: call.input_max_plaintext_norm_ranges.clone(),
            output_gate_ids: call.output_gate_ids.clone(),
        }
    }

    pub(crate) fn with_summed_sub_circuit_call_by_id<T>(
        &self,
        summed_call_id: usize,
        f: impl FnOnce(usize, &[usize], &[usize], &[GateId]) -> T,
    ) -> T {
        let call = self
            .summed_sub_circuit_calls
            .get(&summed_call_id)
            .expect("summed sub-circuit call missing");
        f(
            call.sub_circuit_id,
            &call.call_input_set_ids,
            &call.call_binding_set_ids,
            &call.output_gate_ids,
        )
    }

    pub fn register_sub_circuit_param(&mut self, spec: SubCircuitParamSpec) -> usize {
        let param_id = self.sub_circuit_params.len();
        self.sub_circuit_params.push(spec);
        param_id
    }

    pub(crate) fn expect_sub_circuit_param_kind(
        &self,
        param_id: usize,
        expected: SubCircuitParamKind,
    ) {
        let actual = self
            .sub_circuit_params
            .get(param_id)
            .map(SubCircuitParamSpec::kind)
            .unwrap_or_else(|| panic!("sub-circuit parameter {param_id} is out of range"));
        assert_eq!(
            actual, expected,
            "sub-circuit parameter kind mismatch for param {param_id}: expected {:?}, got {:?}",
            expected, actual
        );
    }

    pub(crate) fn simulator_param_bindings(&self) -> Arc<[SubCircuitParamValue]> {
        Arc::from(
            self.sub_circuit_params
                .iter()
                .map(SubCircuitParamSpec::simulator_binding)
                .collect::<Vec<_>>(),
        )
    }

    pub(crate) fn sub_circuit_input_max_plaintext_norm_ranges(
        &self,
    ) -> Option<&[SubCircuitInputMaxPlaintextNormRange]> {
        self.sub_circuit_input_max_plaintext_norm_ranges.as_deref()
    }

    pub(crate) fn sub_circuit_call_input_max_plaintext_norm_ranges(
        &self,
        call_id: usize,
    ) -> Option<&[SubCircuitInputMaxPlaintextNormRange]> {
        self.sub_circuit_calls
            .get(&call_id)
            .expect("sub-circuit call missing")
            .input_max_plaintext_norm_ranges
            .as_deref()
    }

    pub(crate) fn summed_sub_circuit_call_input_max_plaintext_norm_ranges(
        &self,
        summed_call_id: usize,
    ) -> Option<&[SubCircuitInputMaxPlaintextNormRange]> {
        self.summed_sub_circuit_calls
            .get(&summed_call_id)
            .expect("summed sub-circuit call missing")
            .input_max_plaintext_norm_ranges
            .as_deref()
    }

    fn validate_sub_circuit_input_max_plaintext_norm_ranges(
        &self,
        ranges: &[SubCircuitInputMaxPlaintextNormRange],
        context: &str,
    ) {
        let mut expected_start = 0usize;
        for range in ranges {
            assert!(
                !range.is_empty(),
                "{context}: sub-circuit plaintext range [{}, {}) must not be empty",
                range.start,
                range.end
            );
            assert_eq!(
                range.start, expected_start,
                "{context}: sub-circuit plaintext ranges must be contiguous and sorted"
            );
            assert!(
                range.end <= self.num_input(),
                "{context}: sub-circuit plaintext range [{}, {}) exceeds num_input={}",
                range.start,
                range.end,
                self.num_input()
            );
            expected_start = range.end;
        }
        assert_eq!(
            expected_start,
            self.num_input(),
            "{context}: sub-circuit plaintext ranges must cover every input wire"
        );
    }

    fn normalized_sub_circuit_input_max_plaintext_norm_ranges<I>(
        &self,
        circuit_id: usize,
        ranges: Option<I>,
        context: &str,
    ) -> Option<Arc<[SubCircuitInputMaxPlaintextNormRange]>>
    where
        I: IntoIterator<Item = SubCircuitInputMaxPlaintextNormRange>,
    {
        let ranges = ranges?;
        let ranges = Arc::from(ranges.into_iter().collect::<Vec<_>>());
        let stored = self.sub_circuit_registry.get(circuit_id);
        stored.validate_sub_circuit_input_max_plaintext_norm_ranges(&ranges, context);
        Some(ranges)
    }

    fn validate_sub_circuit_param_bindings(
        &self,
        circuit_id: usize,
        param_bindings: &[SubCircuitParamValue],
        context: &str,
    ) {
        let stored = self.sub_circuit_registry.get(circuit_id);
        let expected_param_specs = &stored.sub_circuit_params;
        assert_eq!(
            param_bindings.len(),
            expected_param_specs.len(),
            "{context}: sub-circuit parameter count mismatch"
        );
        for (param_idx, (binding, expected_spec)) in
            param_bindings.iter().zip(expected_param_specs.iter()).enumerate()
        {
            expected_spec.validate_binding(binding, param_idx);
        }
    }

    pub(crate) fn direct_sub_circuit_ids(&self) -> BTreeSet<usize> {
        self.sub_circuit_calls
            .values()
            .map(|call| call.sub_circuit_id)
            .chain(self.summed_sub_circuit_calls.values().map(|call| call.sub_circuit_id))
            .collect()
    }

    pub(crate) fn registered_sub_circuit_ref(&self, circuit_id: usize) -> Arc<Self> {
        self.sub_circuit_registry.get(circuit_id)
    }

    pub fn register_public_lookup(&mut self, public_lookup: PublicLut<P>) -> usize {
        if !self.allow_register_lookup {
            panic!("lookup table registration is only allowed on top-level circuits");
        }
        self.lookup_registry.register(public_lookup)
    }

    pub fn lut_vector_len_with_subcircuits(&self) -> usize {
        self.lut_vector_len_with_subcircuits_and_bindings(&[])
    }

    fn lut_vector_len_with_subcircuits_and_bindings(
        &self,
        param_bindings: &[SubCircuitParamValue],
    ) -> usize {
        let mut total = 0usize;
        for gate in self.gates.values() {
            if let PolyGateType::PubLut { lut_id } = &gate.gate_type {
                let lut_id = lut_id.resolve_public_lookup(param_bindings);
                let lookup =
                    self.lookup_registry.lookups.get(&lut_id).expect("lookup table missing");
                total += lookup.len();
            }
        }
        for call in self.sub_circuit_calls.values() {
            let param_bindings = self.binding_set(call.binding_set_id);
            total += self.with_sub_circuit(call.sub_circuit_id, |sub| {
                sub.lut_vector_len_with_subcircuits_and_bindings(param_bindings.as_ref())
            });
        }
        for call in self.summed_sub_circuit_calls.values() {
            for binding_set_id in &call.call_binding_set_ids {
                let param_bindings = self.binding_set(*binding_set_id);
                total += self.with_sub_circuit(call.sub_circuit_id, |sub| {
                    sub.lut_vector_len_with_subcircuits_and_bindings(param_bindings.as_ref())
                });
            }
        }
        total
    }

    /// Compatibility no-op.
    ///
    /// Older callers can still opt into "disk" storage, but sub-circuits are now stored only as
    /// shared in-memory `Arc<PolyCircuit<_>>` values. Keeping this method avoids a broader API
    /// churn outside the current refactor scope.
    pub fn enable_subcircuits_in_disk(&mut self, _dir_path: impl AsRef<Path>) {}

    pub fn register_sub_circuit<T>(&mut self, sub_circuit: T) -> usize
    where
        T: Into<Arc<Self>>,
    {
        let mut sub_circuit = sub_circuit.into();
        Arc::make_mut(&mut sub_circuit).inherit_registries_from_parent(self);
        self.sub_circuit_registry.register_arc(sub_circuit)
    }

    pub fn register_sub_circuit_with_max_plaintext_norms<T, I>(
        &mut self,
        sub_circuit: T,
        max_plaintext_norm_ranges: I,
    ) -> usize
    where
        T: Into<Arc<Self>>,
        I: IntoIterator<Item = SubCircuitInputMaxPlaintextNormRange>,
    {
        let mut sub_circuit = sub_circuit.into();
        let ranges = Arc::from(max_plaintext_norm_ranges.into_iter().collect::<Vec<_>>());
        {
            let sub_circuit_mut = Arc::make_mut(&mut sub_circuit);
            sub_circuit_mut.inherit_registries_from_parent(self);
            sub_circuit_mut.validate_sub_circuit_input_max_plaintext_norm_ranges(
                &ranges,
                "sub-circuit registration",
            );
            sub_circuit_mut.sub_circuit_input_max_plaintext_norm_ranges = Some(ranges);
        }
        self.sub_circuit_registry.register_arc(sub_circuit)
    }

    pub fn call_sub_circuit<I, W>(&mut self, circuit_id: usize, inputs: I) -> Vec<BatchedWire>
    where
        I: IntoIterator<Item = W>,
        W: Into<BatchedWire>,
    {
        self.call_sub_circuit_with_bindings(circuit_id, inputs, &[])
    }

    pub fn call_sub_circuit_with_max_plaintext_norms<I, W, R>(
        &mut self,
        circuit_id: usize,
        inputs: I,
        input_max_plaintext_norm_ranges: R,
    ) -> Vec<BatchedWire>
    where
        I: IntoIterator<Item = W>,
        W: Into<BatchedWire>,
        R: IntoIterator<Item = SubCircuitInputMaxPlaintextNormRange>,
    {
        self.call_sub_circuit_with_bindings_and_max_plaintext_norms(
            circuit_id,
            inputs,
            &[],
            input_max_plaintext_norm_ranges,
        )
    }

    pub fn call_sub_circuit_with_shared_input_prefix_and_bindings<I, W>(
        &mut self,
        circuit_id: usize,
        shared_input_prefix_set_id: usize,
        input_suffix: I,
        param_bindings: &[SubCircuitParamValue],
    ) -> Vec<BatchedWire>
    where
        I: IntoIterator<Item = W>,
        W: Into<BatchedWire>,
    {
        self.call_sub_circuit_with_prefix_and_bindings_and_max_plaintext_norms(
            circuit_id,
            Some(shared_input_prefix_set_id),
            input_suffix,
            param_bindings,
            Option::<Vec<SubCircuitInputMaxPlaintextNormRange>>::None,
        )
    }

    pub fn call_sub_circuit_with_shared_input_prefix_and_bindings_and_max_plaintext_norms<I, W, R>(
        &mut self,
        circuit_id: usize,
        shared_input_prefix_set_id: usize,
        input_suffix: I,
        param_bindings: &[SubCircuitParamValue],
        input_max_plaintext_norm_ranges: R,
    ) -> Vec<BatchedWire>
    where
        I: IntoIterator<Item = W>,
        W: Into<BatchedWire>,
        R: IntoIterator<Item = SubCircuitInputMaxPlaintextNormRange>,
    {
        self.call_sub_circuit_with_prefix_and_bindings_and_max_plaintext_norms(
            circuit_id,
            Some(shared_input_prefix_set_id),
            input_suffix,
            param_bindings,
            Some(input_max_plaintext_norm_ranges),
        )
    }

    pub fn call_sub_circuit_with_bindings_and_max_plaintext_norms<I, W, R>(
        &mut self,
        circuit_id: usize,
        inputs: I,
        param_bindings: &[SubCircuitParamValue],
        input_max_plaintext_norm_ranges: R,
    ) -> Vec<BatchedWire>
    where
        I: IntoIterator<Item = W>,
        W: Into<BatchedWire>,
        R: IntoIterator<Item = SubCircuitInputMaxPlaintextNormRange>,
    {
        self.call_sub_circuit_with_prefix_and_bindings_and_max_plaintext_norms(
            circuit_id,
            None,
            inputs,
            param_bindings,
            Some(input_max_plaintext_norm_ranges),
        )
    }

    pub fn call_sub_circuit_with_bindings<I, W>(
        &mut self,
        circuit_id: usize,
        inputs: I,
        param_bindings: &[SubCircuitParamValue],
    ) -> Vec<BatchedWire>
    where
        I: IntoIterator<Item = W>,
        W: Into<BatchedWire>,
    {
        self.call_sub_circuit_with_prefix_and_bindings_and_max_plaintext_norms(
            circuit_id,
            None,
            inputs,
            param_bindings,
            Option::<Vec<SubCircuitInputMaxPlaintextNormRange>>::None,
        )
    }

    fn call_sub_circuit_with_prefix_and_bindings_and_max_plaintext_norms<I, W, R>(
        &mut self,
        circuit_id: usize,
        shared_input_prefix_set_id: Option<usize>,
        input_suffix: I,
        param_bindings: &[SubCircuitParamValue],
        input_max_plaintext_norm_ranges: Option<R>,
    ) -> Vec<BatchedWire>
    where
        I: IntoIterator<Item = W>,
        W: Into<BatchedWire>,
        R: IntoIterator<Item = SubCircuitInputMaxPlaintextNormRange>,
    {
        let input_suffix = input_suffix.into_iter().map(Into::into).collect::<Vec<_>>();
        let total_num_inputs = shared_input_prefix_set_id
            .map(|input_set_id| batched_wire_slice_len(self.input_set(input_set_id).as_ref()))
            .unwrap_or(0) +
            batched_wire_slice_len(&input_suffix);
        let stored = self.sub_circuit_registry.get(circuit_id);
        let num_inputs = stored.num_input();
        assert_eq!(total_num_inputs, num_inputs);
        self.validate_sub_circuit_param_bindings(
            circuit_id,
            param_bindings,
            "direct sub-circuit call",
        );
        let input_max_plaintext_norm_ranges = self
            .normalized_sub_circuit_input_max_plaintext_norm_ranges(
                circuit_id,
                input_max_plaintext_norm_ranges,
                "direct sub-circuit call",
            );
        let num_outputs = self.sub_circuit_num_output(circuit_id);
        let call_id = self.sub_circuit_calls.len();
        let binding_set_id = self.binding_registry.register(param_bindings);
        let scoped_call_id = self.next_scoped_call_id;
        self.next_scoped_call_id += 1;
        let mut output_gate_ids = Vec::with_capacity(num_outputs);
        for output_idx in 0..num_outputs {
            output_gate_ids.push(self.new_sub_circuit_output_gate(
                call_id,
                output_idx,
                total_num_inputs,
            ));
        }
        let call = SubCircuitCall {
            sub_circuit_id: circuit_id,
            shared_input_prefix_set_id,
            input_suffix,
            binding_set_id,
            input_max_plaintext_norm_ranges,
            scoped_call_id,
            output_gate_ids: output_gate_ids.clone(),
            num_outputs,
        };
        self.sub_circuit_calls.insert(call_id, call);
        output_gate_ids.into_iter().map(BatchedWire::single).collect()
    }

    pub fn call_sub_circuit_sum_many_with_binding_set_ids(
        &mut self,
        circuit_id: usize,
        call_input_set_ids: Vec<usize>,
        call_binding_set_ids: Vec<usize>,
    ) -> Vec<BatchedWire> {
        self.call_sub_circuit_sum_many_with_binding_set_ids_and_max_plaintext_norms(
            circuit_id,
            call_input_set_ids,
            call_binding_set_ids,
            Option::<Vec<SubCircuitInputMaxPlaintextNormRange>>::None,
        )
    }

    pub fn call_sub_circuit_sum_many_with_binding_set_ids_and_max_plaintext_norms<R>(
        &mut self,
        circuit_id: usize,
        call_input_set_ids: Vec<usize>,
        call_binding_set_ids: Vec<usize>,
        input_max_plaintext_norm_ranges: Option<R>,
    ) -> Vec<BatchedWire>
    where
        R: IntoIterator<Item = SubCircuitInputMaxPlaintextNormRange>,
    {
        assert!(
            !call_input_set_ids.is_empty(),
            "summed sub-circuit call requires at least one inner call"
        );
        assert_eq!(
            call_input_set_ids.len(),
            call_binding_set_ids.len(),
            "summed sub-circuit call requires one binding set per inner call"
        );
        {
            let stored = self.sub_circuit_registry.get(circuit_id);
            let num_inputs = stored.num_input();
            for (call_idx, (input_set_id, binding_set_id)) in
                call_input_set_ids.iter().zip(call_binding_set_ids.iter()).enumerate()
            {
                let inputs = self.input_set(*input_set_id);
                assert_eq!(
                    batched_wire_slice_len(inputs.as_ref()),
                    num_inputs,
                    "summed sub-circuit input count mismatch at inner call {call_idx}"
                );
                let bindings = self.binding_set(*binding_set_id);
                self.validate_sub_circuit_param_bindings(
                    circuit_id,
                    bindings.as_ref(),
                    "summed sub-circuit call",
                );
            }
        }
        let input_max_plaintext_norm_ranges = self
            .normalized_sub_circuit_input_max_plaintext_norm_ranges(
                circuit_id,
                input_max_plaintext_norm_ranges,
                "summed sub-circuit call",
            );
        let num_outputs = self.sub_circuit_num_output(circuit_id);
        let summed_call_id = self.summed_sub_circuit_calls.len();
        let scoped_call_ids = (0..call_input_set_ids.len())
            .map(|_| {
                let scoped_call_id = self.next_scoped_call_id;
                self.next_scoped_call_id += 1;
                scoped_call_id
            })
            .collect::<Vec<_>>();
        let flattened_num_inputs = call_input_set_ids
            .iter()
            .map(|input_set_id| batched_wire_slice_len(self.input_set(*input_set_id).as_ref()))
            .sum();
        let output_gate_ids = (0..num_outputs)
            .map(|output_idx| {
                self.new_summed_sub_circuit_output_gate(
                    summed_call_id,
                    output_idx,
                    flattened_num_inputs,
                )
            })
            .collect::<Vec<_>>();
        self.summed_sub_circuit_calls.insert(
            summed_call_id,
            SummedSubCircuitCall {
                sub_circuit_id: circuit_id,
                call_input_set_ids,
                call_binding_set_ids,
                input_max_plaintext_norm_ranges,
                scoped_call_ids,
                output_gate_ids: output_gate_ids.clone(),
                num_outputs,
            },
        );
        output_gate_ids.into_iter().map(BatchedWire::single).collect()
    }
}
