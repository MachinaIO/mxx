//! Compilation of Boolean NAND circuits to the fixed-width Exponent-LUT IR.
//!
//! The compiler is the host-side part of the construction in §5.2 of the
//! accompanying paper.  A three-register program is packed as
//! `R0 + 2 * R1 + 4 * R2`; each instruction is consequently one binary LUT
//! with an 8-valued left input and a 2-valued right input.  The right input is
//! always an original circuit bit.  In particular, an evaluated intermediate
//! state is never made into an RHS package.

use crate::{
    layout::ExpLayout,
    program::{
        AttributeKnowledge, ExponentLutProgram, ExponentLutProgramBuilder, ProgramInputId,
        ProgramValidationError, ProgramWireType, RhsId,
    },
};
use thiserror::Error;

const STATE_WIDTH: usize = 8;
const RHS_WIDTH: usize = 2;
const LUT_DOMAIN: usize = STATE_WIDTH * RHS_WIDTH;

/// A Boolean circuit whose gates are in topological order.
///
/// Node indices `[0, input_count)` are original input bits.  Gate `i` has
/// node index `input_count + i`, and its two entries are the indices of its
/// NAND operands.  The output may be any node, including an input node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NandCircuit {
    input_count: usize,
    gates: Vec<[usize; 2]>,
    output: usize,
}

/// Errors from circuit validation or bounded program expansion.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum NandCircuitError {
    #[error("a NAND circuit must declare at least one input")]
    NoInputs,
    #[error("NAND gate {gate} references node {node}, which is not available yet")]
    NonTopologicalGate { gate: usize, node: usize },
    #[error("circuit output node {output} is undefined")]
    UndefinedOutput { output: usize },
    #[error("the circuit node count overflows usize")]
    NodeCountOverflow,
    #[error("reachable NAND expansion exceeds the checked budget of {budget} LUT rounds")]
    ExpansionBudgetExceeded { budget: usize },
    #[error("reachable NAND expansion overflows the LUT-round count")]
    ExpansionOverflow,
    #[error(transparent)]
    Program(#[from] ProgramValidationError),
    #[error("the exponent layout must contain the 16-entry NAND LUT domain")]
    LayoutTooNarrow,
}

/// A compiled NAND circuit and the bindings required by its caller.
///
/// `rhs_ids` is in LUT-round order.  For round `r`, bind `rhs_ids[r]` to
/// `mono_m(8 * circuit_input_indices[r])`, and bind `state_input` to
/// `mono_m(2)`.  The latter is the packed initial register state `(0,1,0)`.
#[derive(Clone, Debug)]
pub struct CompiledNandCircuit {
    program: ExponentLutProgram,
    state_input: ProgramInputId,
    rhs_ids: Vec<RhsId>,
    circuit_input_indices: Vec<usize>,
}

impl CompiledNandCircuit {
    /// The immutable Exponent-LUT program.
    pub const fn program(&self) -> &ExponentLutProgram {
        &self.program
    }

    /// The sole program input, carrying packed state and bound to `mono_m(2)`.
    pub const fn state_input(&self) -> ProgramInputId {
        self.state_input
    }

    /// RHS declarations in the order of the binary LUT rounds.
    pub fn rhs_ids(&self) -> &[RhsId] {
        &self.rhs_ids
    }

    /// Original circuit input read by each corresponding RHS round.
    pub fn circuit_input_indices(&self) -> &[usize] {
        &self.circuit_input_indices
    }

    /// Consumes the wrapper and returns the program and its binding metadata.
    pub fn into_parts(self) -> (ExponentLutProgram, ProgramInputId, Vec<RhsId>, Vec<usize>) {
        (self.program, self.state_input, self.rhs_ids, self.circuit_input_indices)
    }
}

impl NandCircuit {
    /// Validates and creates a topologically ordered NAND circuit.
    pub fn new(
        input_count: usize,
        gates: Vec<[usize; 2]>,
        output: usize,
    ) -> Result<Self, NandCircuitError> {
        if input_count == 0 {
            return Err(NandCircuitError::NoInputs);
        }
        let node_count =
            input_count.checked_add(gates.len()).ok_or(NandCircuitError::NodeCountOverflow)?;
        for (gate, operands) in gates.iter().enumerate() {
            let available =
                input_count.checked_add(gate).ok_or(NandCircuitError::NodeCountOverflow)?;
            for node in operands {
                if *node >= available {
                    return Err(NandCircuitError::NonTopologicalGate { gate, node: *node });
                }
            }
        }
        if output >= node_count {
            return Err(NandCircuitError::UndefinedOutput { output });
        }
        Ok(Self { input_count, gates, output })
    }

    pub const fn input_count(&self) -> usize {
        self.input_count
    }

    pub fn gates(&self) -> &[[usize; 2]] {
        &self.gates
    }

    pub const fn output(&self) -> usize {
        self.output
    }

    /// Compile the reachable output using at most `max_lut_rounds` rounds.
    ///
    /// The layout's native domain may be wider than 16, but it must be wide
    /// enough for the fixed 8-by-2 table.  The returned program has one state
    /// input and one explicit RHS declaration per emitted LUT round.
    pub fn compile(
        &self,
        layout: ExpLayout,
        ring_dimension: usize,
        max_lut_rounds: usize,
    ) -> Result<CompiledNandCircuit, NandCircuitError> {
        layout.validate(ring_dimension).map_err(|_| NandCircuitError::LayoutTooNarrow)?;
        if layout.width() < LUT_DOMAIN {
            return Err(NandCircuitError::LayoutTooNarrow);
        }

        let node_count = self
            .input_count
            .checked_add(self.gates.len())
            .ok_or(NandCircuitError::NodeCountOverflow)?;
        let mut memo = vec![None; node_count];
        let round_count = self.reachable_round_count(self.output, &mut memo)?;
        if round_count > max_lut_rounds {
            return Err(NandCircuitError::ExpansionBudgetExceeded { budget: max_lut_rounds });
        }

        let mut instructions = Vec::with_capacity(round_count);
        self.emit(self.output, 0, 1, 2, &mut instructions);
        debug_assert_eq!(instructions.len(), round_count);

        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(layout, ring_dimension)?;
        let state_input = builder.input(
            STATE_WIDTH,
            ProgramWireType {
                stride_log2: layout.stride_log2,
                knowledge: AttributeKnowledge::Private,
            },
        )?;
        let mut state = builder.input_wire(state_input)?;
        let rhs_family = builder.rhs_family(RHS_WIDTH)?;
        let mut rhs_ids = Vec::with_capacity(instructions.len());
        let mut circuit_input_indices = Vec::with_capacity(instructions.len());

        for (round, instruction) in instructions.iter().enumerate() {
            let rhs = builder.rhs_input(rhs_family, STATE_WIDTH, RHS_WIDTH)?;
            let output_width = if round + 1 == instructions.len() { 2 } else { STATE_WIDTH };
            let values = (0..RHS_WIDTH)
                .flat_map(|bit| {
                    (0..STATE_WIDTH).map(move |packed| {
                        let transformed = instruction.apply(packed, bit);
                        if output_width == 2 { transformed & 1 } else { transformed }
                    })
                })
                .collect();
            let table =
                crate::program::LutTable::binary(STATE_WIDTH, RHS_WIDTH, output_width, values)?;
            state = builder.lut(state, table, Some(rhs))?;
            rhs_ids.push(rhs);
            circuit_input_indices.push(instruction.input);
        }

        builder.output(state)?;
        let program = builder.build()?;
        Ok(CompiledNandCircuit { program, state_input, rhs_ids, circuit_input_indices })
    }

    fn reachable_round_count(
        &self,
        node: usize,
        memo: &mut [Option<usize>],
    ) -> Result<usize, NandCircuitError> {
        if let Some(count) = memo[node] {
            return Ok(count);
        }
        let count = if node < self.input_count {
            1
        } else {
            let [lhs, rhs] = self.gates[node - self.input_count];
            let left = self.reachable_round_count(lhs, memo)?;
            let right = self.reachable_round_count(rhs, memo)?;
            left.checked_add(right)
                .and_then(|sum| sum.checked_add(left))
                .and_then(|sum| sum.checked_add(right))
                .ok_or(NandCircuitError::ExpansionOverflow)?
        };
        memo[node] = Some(count);
        Ok(count)
    }

    fn emit(&self, node: usize, i: usize, j: usize, k: usize, output: &mut Vec<Instruction>) {
        if node < self.input_count {
            output.push(Instruction::conditional_shear(node, i, j));
            return;
        }
        let [lhs, rhs] = self.gates[node - self.input_count];
        let mut left = Vec::new();
        self.emit(lhs, k, j, i, &mut left);
        // P_g starts with the unconditional shear R_i <- R_i + R_j.  It is
        // fused into the first input-dependent matrix of P_lhs, as in §5.2.
        if let Some(first) = left.first_mut() {
            first.compose_right(Instruction::unconditional_shear(i, j));
        }
        output.extend(left);
        self.emit(rhs, i, k, j, output);
        self.emit(lhs, k, j, i, output);
        self.emit(rhs, i, k, j, output);
    }
}

#[derive(Clone, Copy, Debug)]
struct Instruction {
    input: usize,
    matrices: [[[u8; 3]; 3]; 2],
}

impl Instruction {
    fn conditional_shear(input: usize, i: usize, j: usize) -> Self {
        Self { input, matrices: [identity(), shear(i, j)] }
    }

    fn unconditional_shear(i: usize, j: usize) -> Self {
        let matrix = shear(i, j);
        Self { input: 0, matrices: [matrix, matrix] }
    }

    fn compose_right(&mut self, preceding: Self) {
        self.matrices[0] = multiply(self.matrices[0], preceding.matrices[0]);
        self.matrices[1] = multiply(self.matrices[1], preceding.matrices[1]);
    }

    fn apply(self, packed: usize, bit: usize) -> usize {
        let mut input = [0u8; 3];
        for (index, value) in input.iter_mut().enumerate() {
            *value = ((packed >> index) & 1) as u8;
        }
        let mut output = [0u8; 3];
        for row in 0..3 {
            output[row] =
                (0..3).map(|column| self.matrices[bit][row][column] * input[column]).sum::<u8>() &
                    1;
        }
        output[0] as usize | ((output[1] as usize) << 1) | ((output[2] as usize) << 2)
    }
}

fn identity() -> [[u8; 3]; 3] {
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
}

fn shear(i: usize, j: usize) -> [[u8; 3]; 3] {
    let mut matrix = identity();
    matrix[i][j] = 1;
    matrix
}

fn multiply(lhs: [[u8; 3]; 3], rhs: [[u8; 3]; 3]) -> [[u8; 3]; 3] {
    let mut result = [[0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            result[row][column] =
                (0..3).map(|index| lhs[row][index] * rhs[index][column]).sum::<u8>() & 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{CompiledNandCircuit, NandCircuit, NandCircuitError};
    use crate::{ExpLayout, program::Gate};

    fn compile(circuit: NandCircuit, budget: usize) -> CompiledNandCircuit {
        circuit.compile(ExpLayout { stride_log2: 0, width_log2: 4 }, 32, budget).unwrap()
    }

    #[test]
    fn validates_all_nodes_before_reaching_output() {
        assert!(matches!(
            NandCircuit::new(2, vec![[0, 3]], 0),
            Err(NandCircuitError::NonTopologicalGate { .. })
        ));
        assert!(matches!(
            NandCircuit::new(2, vec![[0, 1]], 4),
            Err(NandCircuitError::UndefinedOutput { .. })
        ));
    }

    #[test]
    fn compiles_input_only_and_extracts_the_low_bit() {
        let compiled = compile(NandCircuit::new(1, vec![], 0).unwrap(), 1);
        assert_eq!(compiled.rhs_ids().len(), 1);
        assert_eq!(compiled.circuit_input_indices(), &[0]);
        assert_eq!(compiled.program().outputs().len(), 1);
        let gates = compiled.program().gates();
        let Gate::Lut { lut_id, .. } = gates.last().unwrap() else {
            panic!("last operation must be a LUT")
        };
        assert_eq!(compiled.program().lut(*lut_id).unwrap().output_width(), 2);
        assert_eq!(
            compiled.program().wire_type(compiled.program().outputs()[0]).unwrap().stride_log2,
            0
        );
    }

    #[test]
    fn balanced_depth_has_exactly_four_to_the_depth_rounds() {
        let circuit = NandCircuit::new(4, vec![[0, 1], [2, 3], [4, 5]], 6).unwrap();
        let compiled = compile(circuit, 16);
        assert_eq!(compiled.rhs_ids().len(), 16);
        assert_eq!(compiled.circuit_input_indices().len(), 16);
    }

    #[test]
    fn unreachable_gates_do_not_consume_budget() {
        let circuit = NandCircuit::new(2, vec![[0, 1], [2, 2], [3, 3]], 2).unwrap();
        let compiled = compile(circuit, 4);
        assert_eq!(compiled.rhs_ids().len(), 4);
    }

    #[test]
    fn rejects_checked_expansion_budget() {
        let circuit = NandCircuit::new(2, vec![[0, 1]], 2).unwrap();
        assert!(matches!(
            circuit.compile(ExpLayout { stride_log2: 0, width_log2: 4 }, 32, 3),
            Err(NandCircuitError::ExpansionBudgetExceeded { budget: 3 })
        ));
    }

    #[test]
    fn trusted_boolean_oracle_matches_compiled_round_schedule() {
        let circuit = NandCircuit::new(2, vec![[0, 1]], 2).unwrap();
        let compiled = compile(circuit.clone(), 4);
        for a in 0..=1 {
            for b in 0..=1 {
                let inputs = [a, b];
                let mut instructions = Vec::new();
                circuit.emit(circuit.output, 0, 1, 2, &mut instructions);
                let state =
                    instructions.iter().enumerate().fold(2usize, |state, (round, instruction)| {
                        let value = instruction
                            .apply(state, inputs[compiled.circuit_input_indices()[round]]);
                        if round + 1 == instructions.len() { value & 1 } else { value }
                    });
                assert_eq!(state, 1 ^ (a & b));
            }
        }
        assert!(
            compiled
                .program()
                .gates()
                .iter()
                .all(|gate| matches!(gate, Gate::MulPrivate { .. } | Gate::Lut { .. }))
        );
    }

    #[test]
    fn production_binding_tables_match_rhs_monomials_and_state_pack() {
        let circuit = NandCircuit::new(2, vec![[0, 1]], 2).unwrap();
        let compiled = compile(circuit.clone(), 4);
        let mut instructions = Vec::new();
        circuit.emit(circuit.output, 0, 1, 2, &mut instructions);
        for (round, instruction) in instructions.iter().enumerate() {
            let Gate::MulPrivate { rhs, .. } = compiled.program().gates()[2 * round] else {
                panic!("binary LUT must bind an explicit RHS package")
            };
            assert_eq!(rhs, compiled.rhs_ids()[round]);
            let Gate::Lut { lut_id, .. } = compiled.program().gates()[2 * round + 1] else {
                panic!("binary LUT must lower through a LUT gate")
            };
            let table = compiled.program().lut(lut_id).unwrap();
            for bit in 0..2 {
                for packed in 0..8 {
                    let expected = instruction.apply(packed, bit);
                    let expected =
                        if round + 1 == instructions.len() { expected & 1 } else { expected };
                    assert_eq!(table.values()[packed + 8 * bit], expected);
                }
            }
        }
        // The caller's state binding mono_m(2) is exactly the packed register
        // state (R0,R1,R2)=(0,1,0), and each RHS mono_m(8*x) selects bit 3.
        assert_eq!(2usize & 1, 0);
        assert_eq!((2usize >> 1) & 1, 1);
        assert_eq!((2usize >> 2) & 1, 0);
        assert!(compiled.circuit_input_indices().iter().all(|index| *index < 2));
    }

    #[test]
    fn depth_two_schedule_matches_exhaustive_boolean_oracle() {
        // NAND(NAND(x0,x1), NAND(x2,x3)).
        let circuit = NandCircuit::new(4, vec![[0, 1], [2, 3], [4, 5]], 6).unwrap();
        let compiled = compile(circuit.clone(), 16);
        for assignment in 0..16 {
            let inputs = [
                assignment & 1,
                (assignment >> 1) & 1,
                (assignment >> 2) & 1,
                (assignment >> 3) & 1,
            ];
            let mut instructions = Vec::new();
            circuit.emit(circuit.output, 0, 1, 2, &mut instructions);
            let state =
                instructions.iter().enumerate().fold(2usize, |state, (round, instruction)| {
                    let value =
                        instruction.apply(state, inputs[compiled.circuit_input_indices()[round]]);
                    if round + 1 == instructions.len() { value & 1 } else { value }
                });
            let expected = 1 ^ ((1 ^ (inputs[0] & inputs[1])) & (1 ^ (inputs[2] & inputs[3])));
            assert_eq!(state, expected, "assignment {assignment:04b}");
        }
    }
}
