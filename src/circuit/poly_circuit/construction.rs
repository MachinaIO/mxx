use super::*;

impl<P: Poly> PolyCircuit<P> {
    pub fn new() -> Self {
        let mut gates = BTreeMap::new();
        // Ensure the reserved constant-one gate exists at GateId(0).
        gates.insert(GateId(0), PolyGate::new(GateId(0), PolyGateType::Input, vec![]));
        let mut gate_counts = HashMap::new();
        gate_counts.insert(PolyGateKind::Input, 1);
        let lookup_registry = Arc::new(LookupRegistry::new());
        let binding_registry = Arc::new(BindingRegistry::new());
        let input_set_registry = Arc::new(InputSetRegistry::new());
        Self {
            gates,
            print_value: BTreeMap::new(),
            sub_circuits: BTreeMap::new(),
            sub_circuit_calls: BTreeMap::new(),
            summed_sub_circuit_calls: BTreeMap::new(),
            sub_circuit_params: vec![],
            output_ids: vec![],
            num_input: 0,
            gate_counts,
            lookup_registry,
            binding_registry,
            input_set_registry,
            next_scoped_call_id: 0,
            allow_register_lookup: true,
            sub_circuit_disk_storage: None,
        }
    }

    /// Get number of inputs
    pub fn num_input(&self) -> usize {
        self.num_input
    }

    /// Get number of outputs
    pub fn num_output(&self) -> usize {
        self.output_ids.len()
    }

    /// Get number of sub-circuit parameters
    pub fn num_sub_circuit_params(&self) -> usize {
        self.sub_circuit_params.len()
    }

    /// Get number of gates
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    pub fn gates_in_id_order(&self) -> impl Iterator<Item = (&GateId, &PolyGate)> {
        self.gates.iter()
    }

    pub(crate) fn gate(&self, gate_id: GateId) -> &PolyGate {
        self.gates.get(&gate_id).unwrap_or_else(|| panic!("gate {gate_id} not found"))
    }

    pub(crate) fn sorted_input_gate_ids(&self) -> Vec<GateId> {
        let mut input_gate_ids = self
            .gates
            .iter()
            .filter_map(|(id, gate)| match &gate.gate_type {
                PolyGateType::Input if id.0 != 0 => Some(*id),
                _ => None,
            })
            .collect::<Vec<_>>();
        input_gate_ids.sort_by_key(|gid| gid.0);
        input_gate_ids
    }

    pub(crate) fn output_gate_ids(&self) -> &[GateId] {
        &self.output_ids
    }

    pub fn recompute_gate_counts(&mut self) {
        self.gate_counts.clear();
        for gate in self.gates.values() {
            let kind = gate.gate_type.kind();
            *self.gate_counts.entry(kind).or_insert(0) += 1;
        }
    }

    pub fn print<I: Into<BatchedWire>>(&mut self, gate_id: I, prefix: String) {
        self.print_value.insert(gate_id.into().as_single_wire(), prefix);
    }

    pub fn input(&mut self, num_input: usize) -> BatchedWire {
        let start = GateId(self.gates.len());
        for _ in 0..num_input {
            let next_id = self.gates.len();
            let gid = GateId(next_id);
            self.increment_gate_kind(PolyGateKind::Input);
            self.gates.insert(gid, PolyGate::new(gid, PolyGateType::Input, vec![]));
        }
        self.num_input += num_input;
        BatchedWire::from_start_len(start, num_input)
    }

    pub fn output<I, W>(&mut self, outputs: I)
    where
        I: IntoIterator<Item = W>,
        W: Into<BatchedWire>,
    {
        #[cfg(debug_assertions)]
        assert_eq!(self.output_ids.len(), 0);
        for output in outputs.into_iter() {
            self.output_ids.extend(output.into().gate_ids());
        }
    }

    pub fn const_zero_gate(&mut self) -> BatchedWire {
        self.not_gate(GateId(0))
    }

    /// index 0 have value 1
    pub fn const_one_gate(&mut self) -> BatchedWire {
        BatchedWire::single(GateId(0))
    }

    pub fn const_minus_one_gate(&mut self) -> BatchedWire {
        let zero = self.const_zero_gate();
        self.sub_gate(zero, GateId(0))
    }

    pub fn and_gate<L: Into<BatchedWire>, R: Into<BatchedWire>>(
        &mut self,
        left: L,
        right: R,
    ) -> BatchedWire {
        self.mul_gate(left, right)
    }

    /// Computes the NOT gate using arithmetic inversion: `1 - x`.
    /// This operation assumes that `x` is restricted to binary values (0 or 1),
    /// meaning it should only be used with polynomials sampled from a bit distribution.
    /// The computation is achieved by subtracting `x` from 1 (i.e., `0 - x + 1`).
    pub fn not_gate<I: Into<BatchedWire>>(&mut self, input: I) -> BatchedWire {
        self.sub_gate(GateId(0), input)
    }

    pub fn or_gate<L: Into<BatchedWire>, R: Into<BatchedWire>>(
        &mut self,
        left: L,
        right: R,
    ) -> BatchedWire {
        let left = left.into();
        let right = right.into();
        let add = self.add_gate(left, right);
        let mul = self.mul_gate(left, right);
        self.sub_gate(add, mul)
    }

    /// Computes the NAND gate as `NOT(AND(left, right))`.
    /// This operation follows the same restriction as the NOT gate:
    /// `left` and `right` must be bit distribution (0 or 1)
    pub fn nand_gate<L: Into<BatchedWire>, R: Into<BatchedWire>>(
        &mut self,
        left: L,
        right: R,
    ) -> BatchedWire {
        let left = left.into();
        let right = right.into();
        let and_result = self.and_gate(left, right);
        self.not_gate(and_result)
    }

    /// Computes the NOR gate as `NOT(OR(left, right))`.
    /// This operation follows the same restriction as the NOT gate:
    /// `left` and `right` must be bit distribution (0 or 1)
    pub fn nor_gate<L: Into<BatchedWire>, R: Into<BatchedWire>>(
        &mut self,
        left: L,
        right: R,
    ) -> BatchedWire {
        let left = left.into();
        let right = right.into();
        let or_result = self.or_gate(left, right);
        self.not_gate(or_result)
    }

    pub fn xor_gate<L: Into<BatchedWire>, R: Into<BatchedWire>>(
        &mut self,
        left: L,
        right: R,
    ) -> BatchedWire {
        let left = left.into();
        let right = right.into();
        let two = self.add_gate(GateId(0), GateId(0));
        let mul = self.mul_gate(left, right);
        let two_mul = self.mul_gate(two, mul);
        let add = self.add_gate(left, right);
        self.sub_gate(add, two_mul)
    }

    /// Computes the XNOR gate as `NOT(XOR(left, right))`.
    /// This operation follows the same restriction as the NOT gate:
    /// `left` and `right` must be bit distribution (0 or 1)
    pub fn xnor_gate<L: Into<BatchedWire>, R: Into<BatchedWire>>(
        &mut self,
        left: L,
        right: R,
    ) -> BatchedWire {
        let left = left.into();
        let right = right.into();
        let xor_result = self.xor_gate(left, right);
        self.not_gate(xor_result)
    }

    pub fn const_digits(&mut self, digits: &[u32]) -> BatchedWire {
        let one = self.const_one_gate();
        self.small_scalar_mul(one, digits)
    }

    pub fn const_poly(&mut self, poly: &P) -> BatchedWire {
        let one = self.const_one_gate();
        self.large_scalar_mul(one, &poly.coeffs_biguints())
    }

    pub fn add_gate<L: Into<BatchedWire>, R: Into<BatchedWire>>(
        &mut self,
        left_input: L,
        right_input: R,
    ) -> BatchedWire {
        let left_input = left_input.into();
        let right_input = right_input.into();
        debug_assert!(left_input.is_single_wire());
        debug_assert!(right_input.is_single_wire());
        BatchedWire::single(self.new_gate_generic(
            vec![left_input.as_single_wire(), right_input.as_single_wire()],
            PolyGateType::Add,
        ))
    }

    pub fn sub_gate<L: Into<BatchedWire>, R: Into<BatchedWire>>(
        &mut self,
        left_input: L,
        right_input: R,
    ) -> BatchedWire {
        let left_input = left_input.into();
        let right_input = right_input.into();
        debug_assert!(left_input.is_single_wire());
        debug_assert!(right_input.is_single_wire());
        BatchedWire::single(self.new_gate_generic(
            vec![left_input.as_single_wire(), right_input.as_single_wire()],
            PolyGateType::Sub,
        ))
    }

    pub fn mul_gate<L: Into<BatchedWire>, R: Into<BatchedWire>>(
        &mut self,
        left_input: L,
        right_input: R,
    ) -> BatchedWire {
        let left_input = left_input.into();
        let right_input = right_input.into();
        debug_assert!(left_input.is_single_wire());
        debug_assert!(right_input.is_single_wire());
        BatchedWire::single(self.new_gate_generic(
            vec![left_input.as_single_wire(), right_input.as_single_wire()],
            PolyGateType::Mul,
        ))
    }

    pub fn small_scalar_mul<I: Into<BatchedWire>>(
        &mut self,
        input: I,
        scalar: &[u32],
    ) -> BatchedWire {
        let input = input.into();
        debug_assert!(input.is_single_wire());
        BatchedWire::single(self.new_gate_generic(
            vec![input.as_single_wire()],
            PolyGateType::SmallScalarMul { scalar: GateParamSource::Const(scalar.to_vec()) },
        ))
    }

    pub fn small_scalar_mul_param<I: Into<BatchedWire>>(
        &mut self,
        input: I,
        param_id: usize,
    ) -> BatchedWire {
        let input = input.into();
        debug_assert!(input.is_single_wire());
        self.expect_sub_circuit_param_kind(param_id, SubCircuitParamKind::SmallScalarMul);
        BatchedWire::single(self.new_gate_generic(
            vec![input.as_single_wire()],
            PolyGateType::SmallScalarMul { scalar: GateParamSource::Param(param_id) },
        ))
    }

    pub fn large_scalar_mul<I: Into<BatchedWire>>(
        &mut self,
        input: I,
        scalar: &[BigUint],
    ) -> BatchedWire {
        let input = input.into();
        debug_assert!(input.is_single_wire());
        BatchedWire::single(self.new_gate_generic(
            vec![input.as_single_wire()],
            PolyGateType::LargeScalarMul { scalar: GateParamSource::Const(scalar.to_vec()) },
        ))
    }

    pub fn large_scalar_mul_param<I: Into<BatchedWire>>(
        &mut self,
        input: I,
        param_id: usize,
    ) -> BatchedWire {
        let input = input.into();
        debug_assert!(input.is_single_wire());
        self.expect_sub_circuit_param_kind(param_id, SubCircuitParamKind::LargeScalarMul);
        BatchedWire::single(self.new_gate_generic(
            vec![input.as_single_wire()],
            PolyGateType::LargeScalarMul { scalar: GateParamSource::Param(param_id) },
        ))
    }

    pub fn poly_scalar_mul<I: Into<BatchedWire>>(&mut self, input: I, scalar: &P) -> BatchedWire {
        let input = input.into();
        debug_assert!(input.is_single_wire());
        BatchedWire::single(self.new_gate_generic(
            vec![input.as_single_wire()],
            PolyGateType::LargeScalarMul {
                scalar: GateParamSource::Const(scalar.coeffs_biguints()),
            },
        ))
    }

    /// Lowers a ring-dimension-normalized rotation into multiplication by the
    /// monomial `x^shift`.
    ///
    /// `shift` must already be reduced modulo the ring dimension.
    pub fn rotate_gate<I: Into<BatchedWire>>(&mut self, input: I, shift: u64) -> BatchedWire {
        let shift = usize::try_from(shift)
            .expect("PolyCircuit::rotate_gate shift does not fit in usize on this platform");
        let mut scalar = vec![0; shift + 1];
        scalar[shift] = 1;
        self.small_scalar_mul(input, &scalar)
    }

    pub fn public_lookup_gate<I: Into<BatchedWire>>(
        &mut self,
        input: I,
        lut_id: usize,
    ) -> BatchedWire {
        let input = input.into();
        debug_assert!(input.is_single_wire());
        BatchedWire::single(self.new_gate_generic(
            vec![input.as_single_wire()],
            PolyGateType::PubLut { lut_id: GateParamSource::Const(lut_id) },
        ))
    }

    pub fn public_lookup_gate_param<I: Into<BatchedWire>>(
        &mut self,
        input: I,
        param_id: usize,
    ) -> BatchedWire {
        let input = input.into();
        debug_assert!(input.is_single_wire());
        self.expect_sub_circuit_param_kind(param_id, SubCircuitParamKind::PubLut);
        BatchedWire::single(self.new_gate_generic(
            vec![input.as_single_wire()],
            PolyGateType::PubLut { lut_id: GateParamSource::Param(param_id) },
        ))
    }

    pub fn slot_transfer_gate<I: Into<BatchedWire>>(
        &mut self,
        input: I,
        src_slots: &[(u32, Option<u32>)],
    ) -> BatchedWire {
        let input = input.into();
        debug_assert!(input.is_single_wire());
        BatchedWire::single(self.new_gate_generic(
            vec![input.as_single_wire()],
            PolyGateType::SlotTransfer {
                src_slots: GateParamSource::Const(SlotTransferSpec::explicit(src_slots.to_vec())),
            },
        ))
    }

    pub fn slot_transfer_gate_param<I: Into<BatchedWire>>(
        &mut self,
        input: I,
        param_id: usize,
    ) -> BatchedWire {
        let input = input.into();
        debug_assert!(input.is_single_wire());
        self.expect_sub_circuit_param_kind(param_id, SubCircuitParamKind::SlotTransfer);
        BatchedWire::single(self.new_gate_generic(
            vec![input.as_single_wire()],
            PolyGateType::SlotTransfer { src_slots: GateParamSource::Param(param_id) },
        ))
    }

    fn new_gate_generic(&mut self, inputs: Vec<GateId>, gate_type: PolyGateType) -> GateId {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.output_ids.len(), 0);
            assert_eq!(inputs.len(), gate_type.num_input());
            for gate_id in &inputs {
                assert!(self.gates.contains_key(gate_id));
            }
        }
        let gate_id = self.gates.len();
        let gate_kind = gate_type.kind();
        self.increment_gate_kind(gate_kind);
        self.gates.insert(GateId(gate_id), PolyGate::new(GateId(gate_id), gate_type, inputs));
        GateId(gate_id)
    }

    pub(crate) fn new_sub_circuit_output_gate(
        &mut self,
        call_id: usize,
        output_idx: usize,
        num_inputs: usize,
    ) -> GateId {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.output_ids.len(), 0);
        }
        let gate_id = GateId(self.gates.len());
        let gate_type = PolyGateType::SubCircuitOutput { call_id, output_idx, num_inputs };
        self.increment_gate_kind(gate_type.kind());
        self.gates.insert(gate_id, PolyGate::new(gate_id, gate_type, Vec::new()));
        gate_id
    }

    pub(crate) fn new_summed_sub_circuit_output_gate(
        &mut self,
        summed_call_id: usize,
        output_idx: usize,
        num_inputs: usize,
    ) -> GateId {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.output_ids.len(), 0);
        }
        let gate_id = GateId(self.gates.len());
        let gate_type =
            PolyGateType::SummedSubCircuitOutput { summed_call_id, output_idx, num_inputs };
        self.increment_gate_kind(gate_type.kind());
        self.gates.insert(gate_id, PolyGate::new(gate_id, gate_type, Vec::new()));
        gate_id
    }

    fn increment_gate_kind(&mut self, kind: PolyGateKind) {
        *self.gate_counts.entry(kind).or_insert(0) += 1;
    }
}
