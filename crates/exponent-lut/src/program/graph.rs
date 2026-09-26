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
    let schema = mxx_dsl::FamilyType { element: element.schema(), count };
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
    /// Backend-local side state produced for one parallel body.  The ordinary
    /// `parallel` adapter only uses the hook for lifecycle bookkeeping; graph
    /// values are handled by `parallel_with_side_outputs` below.
    type SideOutput;
    fn begin_graph_parallel(&mut self);
    fn finish_graph_parallel(&mut self) -> Result<Self::SideOutput, Self::Error>;
    fn commit_graph_parallel(&mut self) -> Result<(), Self::Error>;
}

impl<E> FamilyExecution<E> for GraphFamilyExecution
where
    E: GraphParallelLowering,
    E::Wire: mxx_dsl::GraphValue,
    E::Rhs: mxx_dsl::GraphValue,
{
    type WireFamily = mxx_dsl::Family<E::Wire>;
    type RhsFamily = mxx_dsl::Family<E::Rhs>;

    fn wire_at(
        &mut self,
        family: &Self::WireFamily,
        index: &IntExpr,
    ) -> Result<E::Wire, ProgramLowerError<E::Error>> {
        Ok(family.at(mxx_dsl::Int::evaluate(index.clone())))
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
        let mut failure = None;
        let start = range.start().clone();
        let family = mxx_dsl::parallel(range.count().clone(), |local| {
            if failure.is_some() {
                return Err(mxx_dsl::DslError::Schema);
            }
            let index = local.expression().map_err(|_| mxx_dsl::DslError::CompileTimeIndex)?;
            evaluator.begin_graph_parallel();
            match body(
                self,
                evaluator,
                IntExpr::Add(Box::new(start.clone()), Box::new(index)).canonicalize(),
            ) {
                Ok(value) => {
                    evaluator.finish_graph_parallel().map_err(|_| mxx_dsl::DslError::Schema)?;
                    Ok(value)
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
        let wires = family.map_err(ProgramLowerError::Dsl)?;
        evaluator.commit_graph_parallel().map_err(ProgramLowerError::Backend)?;
        Ok(wires)
    }

    fn sum(
        &mut self,
        evaluator: &mut E,
        family: &Self::WireFamily,
    ) -> Result<E::Wire, ProgramLowerError<E::Error>> {
        let count = family.count().clone();
        if let Ok(count_value) = count.evaluate(&mxx_ir_core::ParamEnv::default()) &&
            count_value <= 0.into()
        {
            return Err(ProgramLowerError::UnsupportedStructure {
                reason: "cannot reduce an empty family".into(),
            });
        }
        let first = family.at(mxx_dsl::Int::constant(0));
        if count.evaluate(&mxx_ir_core::ParamEnv::default()).is_ok_and(|count| count == 1.into()) {
            return Ok(first);
        }
        // A sum is labelled with the reduction's output wire, which a family
        // member is not. Add the first two members before the loop so that
        // every loop state carries that label and the state schema is the
        // same in each iteration.
        let initial = evaluator
            .add(&first, &family.at(mxx_dsl::Int::constant(1)))
            .map_err(ProgramLowerError::Backend)?;
        let mut failure = None;
        let result = mxx_dsl::iterate(count - 2usize, initial, |index, state| {
            if failure.is_some() {
                return Err(mxx_dsl::DslError::Schema);
            }
            let rhs = family.at(index.add(2usize));
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
            assert!(!table.values().is_empty());
            Ok(input.clone())
        }
    }

    impl ProgramLoweringContext for GraphEvaluator {
        fn replace_site(&mut self, site: Option<LoweringSite>) -> Option<LoweringSite> {
            std::mem::replace(&mut self.site, site)
        }
    }

    impl super::GraphParallelLowering for GraphEvaluator {
        type SideOutput = ();
        fn begin_graph_parallel(&mut self) {}
        fn finish_graph_parallel(&mut self) -> Result<Self::SideOutput, Self::Error> {
            Ok(())
        }
        fn commit_graph_parallel(&mut self) -> Result<(), Self::Error> {
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
        graph
            .validate(
                &mxx_ir_core::ParamEnv::default(),
                mxx_backends::openfhe_guard::gen_modulus_and_warmup,
            )
            .unwrap();
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
        graph
            .validate(
                &mxx_ir_core::ParamEnv::default(),
                mxx_backends::openfhe_guard::gen_modulus_and_warmup,
            )
            .unwrap();
    }
}
