//! The uniform streaming function of `crates/io/DIO2-SPEC.md`, §3.
//!
//! Every step shares one [`StepProgram`], including wires, LUT registrations and data shapes.
//! Step-dependent behavior is a bounded integer row, never a different Program body. Evaluator
//! bits enter through the public `xbits` family; its monomial exponents are computed outside
//! the body from graph-valued bit indices and values.

use mxx_dsl::{DslError, Family, Int};
use mxx_exponent_lut::{
    ExpLayout,
    program::{
        DataId, ExponentLutProgramBuilder, LutOutputRepresentation, LutTable, Operand,
        ProgramValidationError, RhsId, WireId,
    },
};
use mxx_ir_core::IntExpr;
use std::sync::Arc;

/// The bounded integer data of a uniform body, with exactly one row per step.
#[derive(Clone, Debug)]
pub struct StepData {
    /// The common number of entries in each row.
    pub width: usize,
    /// Exclusive upper bound on every entry.
    pub bound: usize,
    rows: Vec<Vec<usize>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_backends::{
        MemoryArtifactStore, RuntimeValue, authority::CpuExecution, backend::poly::cpu_backend,
    };
    use mxx_dsl::DslContext;
    use mxx_exponent_lut::api::{StageGraph, execute_graph};
    use mxx_ir_core::ParamEnv;
    use rand::Rng;
    use std::collections::BTreeMap;

    /// Different data rows use one body and agree with its ideal graph and a clear table lookup.
    #[test]
    fn uniform_table_function_matches_clear_steps() {
        let steps = std::env::var("MXX_IO_DIO2_UNIFORM_STEPS")
            .map_or(3, |value| value.parse::<usize>().unwrap());
        assert!(steps > 0);
        let mut rng = rand::rng();
        let rows = (0..steps)
            .map(|_| (0..4).map(|_| rng.random_range(0..2)).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let function = rows.iter().fold(
            StreamingFunction::additive(1, 1, 1, ExpLayout { stride_log2: 1, width_log2: 2 }),
            |function, row| function.with_step_row(row.clone()).unwrap(),
        );
        let x = (0..steps).map(|_| vec![rng.random()]).collect::<Vec<_>>();
        let y = (0..steps).map(|_| vec![rng.random()]).collect::<Vec<_>>();
        let initial = rng.random_range(0..2);
        let mut expected = initial;
        let mut state = Int::constant(initial);
        for step in 0..steps {
            assert_eq!(function.step_row(step), rows[step]);
            expected = rows[step][expected + chunk_value(&x[step]) + chunk_value(&y[step])];
            let row = mxx_dsl::parallel(4usize, |index| {
                mxx_dsl::select(
                    index,
                    rows[step].iter().map(|value| Int::constant(*value)).collect(),
                )
            })
            .unwrap();
            state = function
                .body
                .ideal(
                    &row,
                    state,
                    Int::constant(chunk_value(&x[step])),
                    &[Int::constant(usize::from(y[step][0]))],
                )
                .unwrap();
        }
        assert_eq!(function.evaluate(initial, &x, &y), expected);
        let graph =
            DslContext::new("uniform-table").output("state", state).unwrap().build().unwrap();
        let mut execution = CpuExecution::new(cpu_backend(Vec::<
            mxx_backends::poly::dcrt::params::DCRTPolyParams,
        >::new()));
        let result = execute_graph(
            StageGraph::new(graph, BTreeMap::new(), BTreeMap::new()),
            &ParamEnv::default(),
            &mut execution,
            &mut MemoryArtifactStore::default(),
            |_| Ok(()),
        )
        .unwrap();
        let RuntimeValue::Int(actual) = &result.outputs["state"] else { panic!("integer output") };
        assert_eq!(actual, &num_bigint::BigInt::from(expected));
    }

    /// Data declarations reject ragged rows, out-of-range entries and empty shapes.
    #[test]
    fn step_data_validates_the_uniform_shape() {
        assert!(StepData::new(0, 2, Vec::new()).is_err());
        assert!(StepData::new(2, 0, Vec::new()).is_err());
        assert!(StepData::new(2, 2, vec![vec![0]]).is_err());
        assert!(StepData::new(2, 2, vec![vec![0, 2]]).is_err());
        assert!(StepData::new(2, 2, vec![vec![0, 1], vec![1, 0]]).is_ok());
    }
}

impl StepData {
    /// Validate common nonzero dimensions and bounds, including when there are no steps yet.
    pub fn new(
        width: usize,
        bound: usize,
        rows: Vec<Vec<usize>>,
    ) -> Result<Self, super::Dio2Error> {
        if width == 0 ||
            bound == 0 ||
            rows.iter().any(|row| row.len() != width || row.iter().any(|value| *value >= bound))
        {
            return Err(super::Dio2Error::Input("step data must have the declared width and bound"));
        }
        Ok(Self { width, bound, rows })
    }
}

/// The fixed root wires and integer data family a uniform step body reads.
pub struct StepWires {
    /// The private state monomial `mono(z_i)`.
    pub state: WireId,
    /// The public family of evaluator-bit monomials, one member per bit.
    pub x: WireId,
    /// The private RHS family of obfuscator-bit monomials.
    pub y: RhsId,
    /// The selected step's integer row, with the shape declared by [`StepData`].
    pub data: DataId,
}

type BuildStep = dyn Fn(&mut ExponentLutProgramBuilder, &StepWires) -> Result<WireId, ProgramValidationError>
    + Send
    + Sync;
type ClearStep = dyn Fn(&[usize], usize, &[bool], &[bool]) -> usize + Send + Sync;
type IdealStep = dyn Fn(&Family<Int>, Int, Int, &[Int]) -> Result<Int, DslError> + Send + Sync;

/// One common Program body, its clear evaluation and its graph-valued ideal evaluation.
#[derive(Clone)]
pub struct StepProgram {
    build: Arc<BuildStep>,
    clear: Arc<ClearStep>,
    ideal: Arc<IdealStep>,
}

impl StepProgram {
    /// Declare the common body and its clear/ideal semantics. Both semantics read the selected
    /// integer row first; the ideal evaluator takes the little-endian chunk value as an `Int`.
    pub fn new(
        build: impl Fn(
            &mut ExponentLutProgramBuilder,
            &StepWires,
        ) -> Result<WireId, ProgramValidationError>
        + Send
        + Sync
        + 'static,
        clear: impl Fn(&[usize], usize, &[bool], &[bool]) -> usize + Send + Sync + 'static,
        ideal: impl Fn(&Family<Int>, Int, Int, &[Int]) -> Result<Int, DslError> + Send + Sync + 'static,
    ) -> Self {
        Self { build: Arc::new(build), clear: Arc::new(clear), ideal: Arc::new(ideal) }
    }

    /// The data-backed table body `row[z + val(x) + val(y)]`, with little-endian bit chunks.
    /// Every row has `layout.width()` entries, each below `2^state_bits`.
    pub fn table(state_bits: usize, eval_bits: usize, obf_bits: usize, layout: ExpLayout) -> Self {
        Self::new(
            move |builder, wires| {
                let mut state = wires.state;
                for bit in 0..eval_bits {
                    state = builder.mul_public(
                        Operand::Member { family: wires.x, index: IntExpr::constant(bit) },
                        Operand::Scalar(state),
                    )?;
                }
                for bit in 0..obf_bits {
                    state = builder.mul_private(
                        Operand::Scalar(state),
                        Operand::Member { family: wires.y, index: IntExpr::constant(bit) },
                    )?;
                }
                let lut = builder.register_lut(LutTable::from_data(
                    layout.width(),
                    layout,
                    1 << state_bits,
                    wires.data,
                    0,
                    LutOutputRepresentation::Exponent(layout),
                )?)?;
                builder.lut(Operand::Scalar(state), lut)
            },
            |row, state, x, y| row[state + chunk_value(x) + chunk_value(y)],
            |row, state, x, y| {
                let index = y.iter().enumerate().fold(state.add(x), |index, (bit, set)| {
                    index.add(set.clone().mul(1usize << bit))
                });
                Ok(row.at(index))
            },
        )
    }

    /// Append the common body to a builder with the selected data row and root wires.
    pub(crate) fn build(
        &self,
        builder: &mut ExponentLutProgramBuilder,
        wires: &StepWires,
    ) -> Result<WireId, ProgramValidationError> {
        (self.build)(builder, wires)
    }

    /// Evaluate the ideal state transition on a graph-valued row, state and evaluator chunk.
    pub fn ideal(&self, row: &Family<Int>, state: Int, x: Int, y: &[Int]) -> Result<Int, DslError> {
        (self.ideal)(row, state, x, y)
    }

    /// Evaluate the clear state transition on one selected data row.
    pub fn clear(&self, row: &[usize], state: usize, x: &[bool], y: &[bool]) -> usize {
        (self.clear)(row, state, x, y)
    }
}

/// Interpret a bit chunk in the same little-endian order as the public monomial encoder.
pub(crate) fn chunk_value(bits: &[bool]) -> usize {
    bits.iter().enumerate().map(|(bit, set)| usize::from(*set) << bit).sum()
}

/// The exponent of a chunk bit as a graph function of the bit index and the bit value.
pub type BitExponent = Arc<dyn Fn(Int, Int) -> Result<Int, DslError> + Send + Sync>;

/// The exponent `set 2^bit stride` of bit `bit < bits` with value `set`, from graph integers:
/// one loop over the bit positions selects the power of two, so the graph does not grow with
/// `bits`.
fn additive_exponent(bits: usize, stride: usize) -> BitExponent {
    Arc::new(move |bit, set| {
        let (_, power) = mxx_dsl::iterate(
            bits,
            (Int::constant(1), Int::constant(0)),
            |index, (power, selected)| {
                let selected = selected.add(index.equal(bit.clone()).to_int().mul(power.clone()));
                Ok((power.mul(2usize), selected))
            },
        )?;
        Ok(set.mul(power).mul(stride))
    })
}

/// A streaming function with one body and bounded per-step integer data.
#[derive(Clone)]
pub struct StreamingFunction {
    /// Number of state bits.
    pub state_bits: usize,
    /// Number of evaluator bits per chunk.
    pub eval_chunk_bits: usize,
    /// Number of obfuscator bits per chunk.
    pub obf_chunk_bits: usize,
    /// The layout of `mono(z) = X^(z * stride)`.
    pub state_layout: ExpLayout,
    /// Graph-valued exponent of evaluator bit `bit` with value `set`.
    pub x_monomial: BitExponent,
    /// Graph-valued exponent of obfuscator bit `bit` with value `set`: the obfuscator encodes
    /// `mono(y_monomial(b, y_b))` as the RHS plaintext of the bit.
    pub y_monomial: BitExponent,
    /// The common body of every step.
    pub body: StepProgram,
    /// The common data shape and one row per step.
    pub data: StepData,
}

impl StreamingFunction {
    /// Start a table function whose bit `b` contributes `set * 2^b * stride` to the exponent.
    /// Append bounded rows with [`Self::with_step_row`], or replace [`Self::body`] and
    /// [`Self::data`] together to declare another uniform function.
    pub fn additive(
        state_bits: usize,
        eval_chunk_bits: usize,
        obf_chunk_bits: usize,
        state_layout: ExpLayout,
    ) -> Self {
        let stride = state_layout.stride();
        Self {
            state_bits,
            eval_chunk_bits,
            obf_chunk_bits,
            state_layout,
            x_monomial: additive_exponent(eval_chunk_bits, stride),
            y_monomial: additive_exponent(obf_chunk_bits, stride),
            body: StepProgram::table(state_bits, eval_chunk_bits, obf_chunk_bits, state_layout),
            data: StepData::new(state_layout.width(), 1 << state_bits, Vec::new())
                .expect("a nonempty exponent layout and state range"),
        }
    }

    /// Append a row without changing the common body or any Program declaration.
    pub fn with_step_row(mut self, row: Vec<usize>) -> Result<Self, super::Dio2Error> {
        if row.len() != self.data.width || row.iter().any(|value| *value >= self.data.bound) {
            return Err(super::Dio2Error::Input("step data must have the declared width and bound"));
        }
        self.data.rows.push(row);
        Ok(self)
    }

    /// Number of rows, hence steps, in this function.
    pub fn stream_length(&self) -> usize {
        self.data.rows.len()
    }

    /// The selected public integer data row.
    pub fn step_row(&self, step: usize) -> &[usize] {
        &self.data.rows[step]
    }

    /// Evaluate every row of the common body from `z_0` on equal-length chunk streams.
    pub fn evaluate(&self, z_0: usize, x: &[Vec<bool>], y: &[Vec<bool>]) -> usize {
        assert_eq!(x.len(), self.stream_length(), "one evaluator chunk per step");
        assert_eq!(y.len(), self.stream_length(), "one obfuscator chunk per step");
        self.data
            .rows
            .iter()
            .zip(x)
            .zip(y)
            .fold(z_0, |state, ((row, x), y)| self.body.clear(row, state, x, y))
    }
}
