use super::*;
use mxx_ir_core::{graph::ValueHandle, types::WireType};

/// A reference to a logical RHS input and, for family inputs, its member.
///
/// The identifier is schema metadata.  Only the optional family index is
/// represented as a graph port, so runtime graph values cannot fabricate a
/// different RHS identity.
#[derive(Clone, Debug, PartialEq)]
pub struct RhsRef<I> {
    pub id: RhsId,
    pub index: Option<I>,
}

impl<I: mxx_dsl::GraphValue> mxx_dsl::GraphValue for RhsRef<I> {
    type Schema = RhsRef<I::Schema>;

    fn flatten(&self) -> Vec<ValueHandle> {
        self.index.as_ref().map_or_else(Vec::new, mxx_dsl::GraphValue::flatten)
    }

    fn schema(&self) -> Self::Schema {
        RhsRef { id: self.id, index: self.index.as_ref().map(mxx_dsl::GraphValue::schema) }
    }

    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
    ) -> Result<Self, mxx_dsl::DslError> {
        match &schema.index {
            None if values.is_empty() => Ok(Self { id: schema.id, index: None }),
            Some(index_schema) if values.len() == 1 => {
                Ok(Self { id: schema.id, index: Some(I::from_values(index_schema, values)?) })
            }
            _ => Err(mxx_dsl::DslError::Schema),
        }
    }
}

impl<S: mxx_dsl::GraphValueSchema> mxx_dsl::GraphValueSchema for RhsRef<S> {
    type Value = RhsRef<S::Value>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RhsRef {
            id: self.id,
            index: self.index.as_ref().map(|schema| schema.placeholders_from(next)),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        self.index.as_ref().map_or_else(Vec::new, mxx_dsl::GraphValueSchema::wire_types)
    }
}

/// A family of composite elements assembled directly from the family of
/// each field. `fields` lists those families' values in the order in which
/// [`mxx_dsl::GraphValue::flatten`] lists the fields of one element, and
/// `element` is any element of the family's schema, used only for that
/// schema (for example one built from each field family's first member).
///
/// Building an input family this way, instead of in a loop that reads each
/// field family's member, keeps the field families at the scope where they
/// were made: a stage binds its imported families before lowering, and a
/// loop body that reads a member then captures the family instead of
/// importing it.
pub fn family_of_fields<T: mxx_dsl::GraphValue>(
    element: &T,
    count: IntExpr,
    fields: &[ValueHandle],
) -> Result<mxx_dsl::Family<T>, mxx_dsl::DslError> {
    let schema = mxx_dsl::FamilyType { element: element.schema(), count, member_slot: None };
    <mxx_dsl::Family<T> as mxx_dsl::GraphValue>::from_values(&schema, fields)
}

/// Graph-valued execution adapter shared by all graph lowerers.
/// Graph-valued family execution used by the built-in scheme adapters.
pub struct GraphFamilyExecution;

/// Optional transport used by graph lowerers to retain artifacts created by a
/// parallel body.  The hook is deliberately on the graph adapter boundary:
/// non-graph evaluators (noise and workload collection) keep the original
/// family walker untouched.
pub(crate) trait GraphParallelLowering:
    ExponentLUTProgramLowering + ProgramLoweringContext
{
    /// Graph values one parallel body produces besides its result, returned
    /// by the same loop as a family of their own.
    type SideOutput: mxx_dsl::GraphValue;
    /// A body of `count` instances starts; `index` is its member index.
    fn begin_graph_parallel(&mut self, count: &IntExpr, index: &IntExpr);
    fn finish_graph_parallel(&mut self) -> Result<Self::SideOutput, Self::Error>;
    /// The loop was built: `side` holds each instance's side output.
    fn commit_graph_parallel(
        &mut self,
        side: mxx_dsl::Family<Self::SideOutput>,
    ) -> Result<(), Self::Error>;
}

impl<E> FamilyExecution<E> for GraphFamilyExecution
where
    E: GraphParallelLowering,
    E::Wire: mxx_dsl::GraphValue,
    E::Rhs: mxx_dsl::GraphValue,
{
    type WireFamily = mxx_dsl::Family<E::Wire>;
    type RhsFamily = mxx_dsl::Family<E::Rhs>;
    type Table = mxx_dsl::Family<mxx_dsl::Int>;

    fn wire_at(
        &mut self,
        _evaluator: &mut E,
        family: &Self::WireFamily,
        index: &IntExpr,
    ) -> Result<E::Wire, ProgramLowerError<E::Error>> {
        Ok(family.at(mxx_dsl::Int::evaluate(index.clone())))
    }

    /// `family.at(table.at(index))`: a dynamic member read whose index is a
    /// table entry. In a loop over the table's entries, an artifact family
    /// read this way lowers to a gather by the table.
    fn wire_at_table(
        &mut self,
        _evaluator: &mut E,
        family: &Self::WireFamily,
        table: &Self::Table,
        index: &IntExpr,
    ) -> Result<E::Wire, ProgramLowerError<E::Error>> {
        Ok(family.at(table.at(mxx_dsl::Int::evaluate(index.clone()))))
    }

    fn rhs_at(
        &mut self,
        family: &Self::RhsFamily,
        index: &IntExpr,
    ) -> Result<E::Rhs, ProgramLowerError<E::Error>> {
        Ok(family.at(mxx_dsl::Int::evaluate(index.clone())))
    }

    fn parallel(
        &mut self,
        evaluator: &mut E,
        range: &FamilyRange,
        body: impl FnOnce(&mut Self, &mut E, IntExpr) -> Result<E::Wire, ProgramLowerError<E::Error>>,
    ) -> Result<Self::WireFamily, ProgramLowerError<E::Error>> {
        graph_parallel(self, evaluator, range, body)
    }

    fn sum(
        &mut self,
        evaluator: &mut E,
        family: &Self::WireFamily,
        columns: usize,
    ) -> Result<E::Wire, ProgramLowerError<E::Error>> {
        // A generic wire has no column placement; the scheme families place
        // the columns of a row sum.
        if columns != 1 {
            return Err(ProgramLowerError::UnsupportedStructure {
                reason: "a row Sum needs a scheme family execution".into(),
            });
        }
        graph_sum(evaluator, family.count().clone(), |index| Ok(family.at(index)))
    }
}

/// Lower `body` once as a graph parallel loop over `range`, for any family
/// execution whose parallel results are graph families. The body receives
/// the absolute member index.
pub(crate) fn graph_parallel<X, E>(
    execution: &mut X,
    evaluator: &mut E,
    range: &FamilyRange,
    body: impl FnOnce(&mut X, &mut E, IntExpr) -> Result<E::Wire, ProgramLowerError<E::Error>>,
) -> Result<mxx_dsl::Family<E::Wire>, ProgramLowerError<E::Error>>
where
    E: GraphParallelLowering,
    E::Wire: mxx_dsl::GraphValue,
{
    let mut failure = None;
    let start = range.start().clone();
    let family = mxx_dsl::parallel(range.count().clone(), |local| {
        if failure.is_some() {
            return Err(mxx_dsl::DslError::Schema);
        }
        let index = local.expression().map_err(|_| mxx_dsl::DslError::CompileTimeIndex)?;
        let index = IntExpr::Add(Box::new(start.clone()), Box::new(index)).canonicalize();
        evaluator.begin_graph_parallel(range.count(), &index);
        match body(execution, evaluator, index) {
            Ok(value) => {
                let side =
                    evaluator.finish_graph_parallel().map_err(|_| mxx_dsl::DslError::Schema)?;
                Ok((value, side))
            }
            Err(error) => {
                failure = Some(error);
                Err(mxx_dsl::DslError::Schema)
            }
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    let family = family.map_err(ProgramLowerError::Dsl)?;
    let wires = family.field(|value| value.0).map_err(ProgramLowerError::Dsl)?;
    let side = family.field(|value| value.1).map_err(ProgramLowerError::Dsl)?;
    evaluator.commit_graph_parallel(side).map_err(ProgramLowerError::Backend)?;
    Ok(wires)
}

/// Sum the `count` members that `member` reads by index, as one graph
/// reduction. A family execution passes how it reads one member.
pub(crate) fn graph_sum<E>(
    evaluator: &mut E,
    count: IntExpr,
    mut member: impl FnMut(mxx_dsl::Int) -> Result<E::Wire, ProgramLowerError<E::Error>>,
) -> Result<E::Wire, ProgramLowerError<E::Error>>
where
    E: ExponentLUTProgramLowering,
    E::Wire: mxx_dsl::GraphValue,
{
    let count_value = count.evaluate(&mxx_ir_core::ParamEnv::default()).ok();
    if count_value.as_ref().is_some_and(|count| *count <= 0.into()) {
        return Err(ProgramLowerError::UnsupportedStructure {
            reason: "cannot reduce an empty family".into(),
        });
    }
    let first = member(mxx_dsl::Int::constant(0))?;
    if count_value.as_ref().is_some_and(|count| *count == 1.into()) {
        return Ok(first);
    }
    // A sum is labelled with the reduction's output wire, which a family
    // member is not. Add the first two members before the loop so that
    // every loop state carries that label and the state schema is the
    // same in each iteration.
    let initial = evaluator
        .add(&first, &member(mxx_dsl::Int::constant(1))?)
        .map_err(ProgramLowerError::Backend)?;
    let mut failure = None;
    let result = mxx_dsl::iterate(count - 2usize, initial, |index, state| {
        if failure.is_some() {
            return Err(mxx_dsl::DslError::Schema);
        }
        let rhs = match member(index.add(2usize)) {
            Ok(rhs) => rhs,
            Err(error) => {
                failure = Some(error);
                return Err(mxx_dsl::DslError::Schema);
            }
        };
        match evaluator.add(&state, &rhs) {
            Ok(value) => Ok(value),
            Err(error) => {
                failure = Some(ProgramLowerError::Backend(error));
                Err(mxx_dsl::DslError::Schema)
            }
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    result.map_err(ProgramLowerError::Dsl)
}

/// The row of the column sums of a family whose column `j < columns` is the
/// `per_column` members from `j per_column` that `member` reads by index:
/// each column is one graph reduction ([`graph_sum`]), `place` puts a column
/// sum (`1 x 1` entries) in its column of a `1 x columns` row, and one
/// sequential loop over the columns adds the placed sums. The graph does not
/// grow with `columns`.
pub(crate) fn graph_column_sums<E>(
    evaluator: &mut E,
    (columns, per_column): (usize, usize),
    mut member: impl FnMut(mxx_dsl::Int) -> Result<E::Wire, ProgramLowerError<E::Error>>,
    place: impl Fn(E::Wire, &IntExpr) -> E::Wire,
) -> Result<E::Wire, ProgramLowerError<E::Error>>
where
    E: ExponentLUTProgramLowering,
    E::Wire: mxx_dsl::GraphValue,
{
    if columns < 2 {
        return Err(ProgramLowerError::UnsupportedStructure {
            reason: "a column sum row needs at least two columns".into(),
        });
    }
    let first = graph_sum(evaluator, IntExpr::constant(per_column), |index| member(index))?;
    let initial = place(first, &IntExpr::constant(0));
    let mut failure = None;
    let result = mxx_dsl::iterate(columns - 1, initial, |column, row| {
        if failure.is_some() {
            return Err(mxx_dsl::DslError::Schema);
        }
        let column = column.add(1usize);
        let start = column.clone().mul(per_column);
        let sum = graph_sum(evaluator, IntExpr::constant(per_column), |index| {
            member(start.clone().add(index))
        })
        .and_then(|sum| {
            let column = column.expression().map_err(ProgramLowerError::Dsl)?;
            let placed = place(sum, &column);
            evaluator.add(&row, &placed).map_err(ProgramLowerError::Backend)
        });
        sum.map_err(|error| {
            failure = Some(error);
            mxx_dsl::DslError::Schema
        })
    });
    if let Some(error) = failure {
        return Err(error);
    }
    result.map_err(ProgramLowerError::Dsl)
}

impl GraphFamilyExecution {
    /// Lower a parallel body while retaining graph-valued side outputs.
    ///
    /// A body may return a tuple whose second field is an existing
    /// tuple/map [`mxx_dsl::GraphValue`] containing per-operation actions.
    /// Both fields are emitted by the same structural loop, so nested scopes
    /// and repeated inner indices remain qualified by every lexical axis.
    /// The returned side family is intended for publication/import by the
    /// caller; this helper does not invent artifact names or a second walker.
    pub fn parallel_with_side_outputs<E, S>(
        &mut self,
        evaluator: &mut E,
        range: &FamilyRange,
        body: impl FnOnce(
            &mut Self,
            &mut E,
            IntExpr,
        ) -> Result<(E::Wire, S), ProgramLowerError<E::Error>>,
    ) -> Result<(mxx_dsl::Family<E::Wire>, mxx_dsl::Family<S>), ProgramLowerError<E::Error>>
    where
        E: ExponentLUTProgramLowering + ProgramLoweringContext,
        E::Wire: mxx_dsl::GraphValue,
        E::Rhs: mxx_dsl::GraphValue,
        S: mxx_dsl::GraphValue,
    {
        let mut failure = None;
        let start = range.start().clone();
        let combined = mxx_dsl::parallel(range.count().clone(), |local| {
            if failure.is_some() {
                return Err(mxx_dsl::DslError::Schema);
            }
            let index = local.expression().map_err(|_| mxx_dsl::DslError::CompileTimeIndex)?;
            match body(
                self,
                evaluator,
                IntExpr::Add(Box::new(start.clone()), Box::new(index)).canonicalize(),
            ) {
                Ok(value) => Ok(value),
                Err(error) => {
                    failure = Some(error);
                    Err(mxx_dsl::DslError::Schema)
                }
            }
        });
        if let Some(error) = failure {
            return Err(error);
        }
        let combined = combined.map_err(ProgramLowerError::Dsl)?;
        let wires = combined.field(|value| value.0).map_err(ProgramLowerError::Dsl)?;
        let side_outputs = combined.field(|value| value.1).map_err(ProgramLowerError::Dsl)?;
        Ok((wires, side_outputs))
    }
}

#[cfg(test)]
mod tests {
    use super::{ExpLayout, GraphFamilyExecution, RhsRef};
    use crate::program::{
        ExponentLUTProgramLowering, FamilyExecution, FamilyRange, LoweringSite,
        ProgramLoweringContext,
    };
    use mxx_dsl::Int;
    use mxx_ir_core::IntExpr;
    use num_bigint::BigInt;

    #[test]
    fn strided_monomial_wraps_at_the_full_negacyclic_modulus() {
        let layout = ExpLayout { stride_log2: 1, width_log2: 3 };
        let ring = crate::test_support::explicit_ring(&[257], (16) as u32);
        let modulus = layout.effective_modulus(16);
        let zero = layout.mono(&ring, &BigInt::from(0)).unwrap();
        let half = layout.mono(&ring, &BigInt::from(modulus / 2)).unwrap();
        let wrapped = layout.mono(&ring, &BigInt::from(modulus)).unwrap();

        assert_eq!(zero.matrix_type(), wrapped.matrix_type());
        assert_eq!(half.matrix_type(), (-zero).matrix_type());
    }

    #[test]
    fn rhs_reference_keeps_id_in_schema_and_only_indexes_are_ports() {
        use crate::program::RhsId;
        use mxx_dsl::{GraphValue, GraphValueSchema, Int};

        let scalar = RhsRef { id: RhsId(7), index: None::<Int> };
        assert!(scalar.flatten().is_empty());
        assert!(scalar.schema().wire_types().is_empty());
        assert_eq!(scalar.schema().id, RhsId(7));

        let indexed = RhsRef { id: RhsId(9), index: Some(Int::constant(3)) };
        let schema = indexed.schema();
        assert_eq!(schema.id, RhsId(9));
        assert_eq!(schema.wire_types().len(), 1);
        let restored = RhsRef::<Int>::from_values(&schema, &indexed.flatten()).unwrap();
        assert_eq!(restored.id, RhsId(9));
        assert_eq!(restored.flatten().len(), 1);
    }

    #[test]
    fn rhs_reference_rejects_wrong_port_count() {
        use crate::program::RhsId;
        use mxx_dsl::{GraphValue, Int};

        let scalar = RhsRef { id: RhsId(1), index: None::<Int> };
        assert!(RhsRef::<Int>::from_values(&scalar.schema(), &scalar.flatten()).is_ok());
        let indexed = RhsRef { id: RhsId(1), index: Some(Int::constant(0)) };
        assert!(RhsRef::<Int>::from_values(&indexed.schema(), &[]).is_err());
    }

    #[derive(Debug)]
    struct GraphError;
    impl std::fmt::Display for GraphError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("graph test error")
        }
    }
    impl std::error::Error for GraphError {}

    #[derive(Default)]
    struct GraphEvaluator {
        site: Option<LoweringSite>,
    }

    impl ExponentLUTProgramLowering for GraphEvaluator {
        type Wire = Int;
        type Rhs = Int;
        type Error = GraphError;

        fn add(&mut self, lhs: &Self::Wire, rhs: &Self::Wire) -> Result<Self::Wire, Self::Error> {
            Ok(lhs.clone().add(rhs.clone()))
        }
        fn mul_public(
            &mut self,
            lhs: &Self::Wire,
            rhs: &Self::Wire,
        ) -> Result<Self::Wire, Self::Error> {
            Ok(lhs.clone().add(rhs.clone()))
        }
        fn mul_private(
            &mut self,
            input: &Self::Wire,
            rhs: &Self::Rhs,
        ) -> Result<Self::Wire, Self::Error> {
            Ok(input.clone().add(rhs.clone()))
        }
        fn lut(
            &mut self,
            input: &Self::Wire,
            table: &crate::program::LutTable,
        ) -> Result<Self::Wire, Self::Error> {
            assert!(table.input_width() > 0);
            Ok(input.clone())
        }
        fn mul_small_constant(
            &mut self,
            input: &Self::Wire,
            _: &[i64],
            _: &IntExpr,
        ) -> Result<Self::Wire, Self::Error> {
            Ok(input.clone())
        }
    }

    impl ProgramLoweringContext for GraphEvaluator {
        fn replace_site(&mut self, site: Option<LoweringSite>) -> Option<LoweringSite> {
            std::mem::replace(&mut self.site, site)
        }
    }

    impl super::GraphParallelLowering for GraphEvaluator {
        type SideOutput = Vec<mxx_dsl::Mat>;
        fn begin_graph_parallel(&mut self, _: &IntExpr, _: &IntExpr) {}
        fn finish_graph_parallel(&mut self) -> Result<Self::SideOutput, Self::Error> {
            Ok(Vec::new())
        }
        fn commit_graph_parallel(
            &mut self,
            _: mxx_dsl::Family<Self::SideOutput>,
        ) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    #[test]
    fn graph_parallel_lifts_tuple_side_outputs_with_scope() {
        use mxx_dsl::{DslContext, Int};

        let mut families = GraphFamilyExecution;
        let mut evaluator = GraphEvaluator::default();
        let range =
            FamilyRange::bounded(IntExpr::constant(2), IntExpr::constant(2), IntExpr::constant(3))
                .unwrap();
        let (wires, actions) = families
            .parallel_with_side_outputs(&mut evaluator, &range, |families, evaluator, index| {
                assert!(matches!(families, GraphFamilyExecution));
                assert!(evaluator.site.is_none());
                let value = Int::evaluate(index);
                let side = (value.clone(), value.clone().add(Int::constant(1)));
                Ok((value, side))
            })
            .unwrap();
        assert_eq!(wires.count(), &IntExpr::constant(2));
        assert_eq!(actions.count(), &IntExpr::constant(2));
        let graph = DslContext::new("qualified-side-output")
            .transferred_output("wires", wires)
            .unwrap()
            .transferred_output("actions", actions)
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&mxx_ir_core::ParamEnv::default()).unwrap();
    }

    /// A lookup in a loop over the table's entries reads the artifact family
    /// at each entry: the loop gathers the family by the table.
    #[test]
    fn graph_lookup_gathers_an_artifact_family_by_the_table() {
        use crate::program::{
            AttributeKnowledge, Environment, ExponentLutProgramBuilder, InputBinding, Operand,
        };
        use mxx_dsl::{DslContext, Family, FamilyType, IntType};
        use mxx_ir_core::{
            artifact::{ArtifactAvailability, ProductionId, SpecHash},
            node::{LoopInputMode, NodeKind},
        };

        let mut builder = ExponentLutProgramBuilder::new();
        let input =
            builder.input_family(IntExpr::constant(3), AttributeKnowledge::PrivatePayload).unwrap();
        let table = builder.table(IntExpr::constant(5), IntExpr::constant(3)).unwrap();
        let reads = builder
            .parallel(FamilyRange::full(IntExpr::constant(5)).unwrap(), |builder, index| {
                builder
                    .mul_small_constant(
                        Operand::Lookup { family: input, table, index },
                        vec![1],
                        IntExpr::constant(0),
                    )
                    .map(Operand::Scalar)
            })
            .unwrap();
        let output = builder.sum(reads).unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();

        let context = DslContext::new("lookup-gather");
        let production = ProductionId { spec_hash: SpecHash([3; 32]), execution_nonce: [4; 32] };
        let members: Family<Int> = context
            .artifact_input(
                production,
                "members",
                FamilyType { element: IntType, count: IntExpr::constant(3), member_slot: None },
                ArtifactAvailability::Transferred,
            )
            .unwrap();
        let mut environment = Environment::new();
        environment.bind_wire(input, InputBinding::Family(members)).unwrap();
        environment.bind_table(table, context.int_family_input("table", 5)).unwrap();
        let outputs = crate::program::lower_program(
            &program,
            &mut GraphEvaluator::default(),
            &mut GraphFamilyExecution,
            &mut environment,
        )
        .unwrap();
        let built = context.output("sum", outputs[0].clone()).unwrap().build().unwrap();
        let modes = built
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .find_map(|node| match node.kind() {
                NodeKind::ParallelLoop(spec) => Some(spec.input_modes.clone()),
                _ => None,
            })
            .unwrap();
        let index_argument =
            modes.iter().position(|mode| *mode == LoopInputMode::Zip).expect("table entries");
        assert!(modes.contains(&LoopInputMode::Gather { index_argument }), "{modes:?}");
    }

    /// An evaluator of plain matrices: the column sums of a row `Sum`.
    struct MatEvaluator;

    impl ExponentLUTProgramLowering for MatEvaluator {
        type Wire = mxx_dsl::Mat;
        type Rhs = mxx_dsl::Mat;
        type Error = GraphError;

        fn add(&mut self, lhs: &Self::Wire, rhs: &Self::Wire) -> Result<Self::Wire, Self::Error> {
            Ok(lhs.clone() + rhs.clone())
        }
        fn mul_public(&mut self, _: &Self::Wire, _: &Self::Wire) -> Result<Self::Wire, GraphError> {
            Err(GraphError)
        }
        fn mul_private(&mut self, _: &Self::Wire, _: &Self::Rhs) -> Result<Self::Wire, GraphError> {
            Err(GraphError)
        }
        fn lut(
            &mut self,
            _: &Self::Wire,
            _: &crate::program::LutTable,
        ) -> Result<Self::Wire, GraphError> {
            Err(GraphError)
        }
        fn mul_small_constant(
            &mut self,
            _: &Self::Wire,
            _: &[i64],
            _: &IntExpr,
        ) -> Result<Self::Wire, GraphError> {
            Err(GraphError)
        }
    }

    /// Three columns of two members `m_i = i + 1`: the row of column sums is
    /// `[3, 7, 11]`, each sum placed in its column by the unit row tensor.
    #[test]
    fn graph_column_sums_place_each_block_sum_in_its_column() {
        use mxx_dsl::DslContext;
        use mxx_ir_core::node::ConstantMatrix;
        let parameters =
            mxx_backends::poly::dcrt::params::DCRTPolyParams::new(8, 1, 30, 15, None, None);
        let ring = crate::test_support::ring_of(&parameters);
        let members = mxx_dsl::parallel(IntExpr::constant(6), |index| {
            let value = IntExpr::Add(Box::new(index.expression()?), Box::new(IntExpr::constant(1)));
            Ok(ring.polynomial([value.canonicalize()]))
        })
        .unwrap();
        let row = super::graph_column_sums(
            &mut MatEvaluator,
            (3, 2),
            |index| Ok(members.at(index)),
            |sum, column| {
                ring.constant((1, 3), ConstantMatrix::UnitRow { index: column.clone() }).tensor(sum)
            },
        )
        .unwrap();
        let graph = DslContext::new("column-sums").output("row", row).unwrap().build().unwrap();
        let result = crate::test_support::execute_cpu(
            &crate::test_support::validate(&graph),
            [parameters],
            Default::default(),
        );
        let row = crate::test_support::full(&result.outputs["row"]);
        let sums = (0..3)
            .map(|column| {
                mxx_backends::poly::Poly::coeffs_biguints(&row.entry(0, column))[0].clone()
            })
            .collect::<Vec<_>>();
        assert_eq!(sums, [3u8, 7, 11].map(num_bigint::BigUint::from));
    }

    #[test]
    fn graph_parallel_preserves_nonzero_nested_indices() {
        use mxx_dsl::DslContext;
        let mut families = GraphFamilyExecution;
        let mut evaluator = GraphEvaluator::default();
        let outer =
            FamilyRange::bounded(IntExpr::constant(2), IntExpr::constant(2), IntExpr::constant(3))
                .unwrap();
        let nested = families
            .parallel(&mut evaluator, &outer, |families, evaluator, index| {
                let inner = FamilyRange::bounded(
                    IntExpr::constant(4),
                    IntExpr::constant(2),
                    IntExpr::constant(3),
                )
                .unwrap();
                let family = families.parallel(evaluator, &inner, |_, _, inner_index| {
                    Ok(Int::evaluate(index.clone()).mul(Int::constant(10)).add(inner_index))
                })?;
                Ok(family.at(Int::constant(0)))
            })
            .unwrap();
        let graph = DslContext::new("nonzero-nested-indices")
            .transferred_output("nested", nested)
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&mxx_ir_core::ParamEnv::default()).unwrap();
    }
}
