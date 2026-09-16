//! Ordered admission events compiled independently of the candidate's capacity
//! slots. Replaying an event allocates its slot in the current timeline, so a
//! smaller scratch width may safely change both class membership and reuse.

use super::*;

type ContextKey = (String, usize);
type DynamicDemand =
    Box<dyn Fn(usize, usize, &mut GpuContextDemand) -> Result<(), PolyBackendError>>;

pub(super) fn import_event(
    params: GpuDCRTPolyParams,
    output: ConcreteWireType,
    begin: usize,
    end: usize,
) -> DynamicDemand {
    Box::new(move |width, _, context| {
        let (matrices, layouts) = import_inventory(&params, &output, width)?;
        for (rows, columns) in matrices {
            context.matrix(begin, end, rows, columns);
        }
        for layout in layouts {
            context.workspace(begin, end, layout, false);
        }
        Ok(())
    })
}

pub(super) fn primitive_event(
    operation: PreparedOperation,
    params: GpuDCRTPolyParams,
    ty: ConcreteMatrixType,
    possible_levels: Vec<usize>,
    formats: Vec<bool>,
    copies: usize,
    scalar_input: bool,
    position: usize,
) -> Result<DynamicDemand, PolyBackendError> {
    let level = params.crt_depth() - 1;
    let rows = operation.scratch_rows()?;
    let fixed = operation.fixed_workspaces(&params, level)?;
    Ok(Box::new(move |width, wave, context| {
        let columns = ty.columns.min(width);
        if possible_levels.iter().any(|possible| *possible != level) {
            let mut group = GpuContextDemand::default();
            for &possible_level in &possible_levels {
                for &evaluation in &formats {
                    let mut alternative = GpuContextDemand::default();
                    for &rows in &rows {
                        alternative.matrix_claim(
                            position,
                            position,
                            GpuTracedClaim::matrix(rows, columns, possible_level, evaluation),
                        );
                    }
                    // The source builder creates no context for an empty
                    // alternative; do not merge an empty branch into the group.
                    if !rows.is_empty() {
                        group.include_alternative(alternative);
                    }
                }
            }
            context.append_alternative_group(group);
        } else {
            for &rows in &rows {
                context.matrix(position, position, rows, columns);
            }
        }
        for &layout in &fixed {
            context.workspace(position, position, layout, false);
        }
        let workspaces = operation.width_workspaces(&params, level, columns)?;
        for _ in 0..copies {
            for &layout in &workspaces {
                context.workspace(position, position, layout, false);
            }
        }
        for layout in operation.batch_workspaces(
            &params,
            level,
            (ty.rows.max(1), 1),
            wave.max(1),
            scalar_input,
        )? {
            context.workspace(position, position, layout, true);
        }
        Ok(())
    }))
}

pub(super) fn preimage_event(
    plan: PreimageClaimPlan,
    params: GpuDCRTPolyParams,
    public_rows: usize,
    ty: ConcreteMatrixType,
    bound: BigUint,
    position: usize,
) -> DynamicDemand {
    Box::new(move |width, wave, context| {
        let width = ty.columns.min(width);
        let mut claims = plan
            .tile_claims(&params, public_rows, width)
            .map_err(PolyBackendError::GpuSubmission)?;
        claims.extend(
            plan.attempt_claims(&params, public_rows, width, ty.rows, &bound)
                .map_err(PolyBackendError::GpuSubmission)?
                .into_iter()
                .flatten(),
        );
        for claim in claims {
            if claim.kind() == GpuPreparedSlotKind::Matrix {
                context.matrix_claim(position, position, claim);
            } else if let Some(layout) = claim.layout() {
                context.workspace(position, position, layout, false);
            }
        }
        for claim in PreimageClaimPlan::residual_batch_metadata(&params, public_rows, 1, wave)
            .map_err(PolyBackendError::GpuSubmission)?
        {
            context.workspace(position, position, claim.layout().expect("batch metadata"), true);
        }
        Ok(())
    })
}

pub(super) fn row_scratch_event(columns: usize, copies: usize, position: usize) -> DynamicDemand {
    Box::new(move |width, _, context| {
        for _ in 0..copies {
            context.matrix(position, position, 1, columns.min(width));
        }
        Ok(())
    })
}

enum DemandEvent {
    Matrix(MatrixDemand),
    Workspace(WorkspaceDemand),
    Alternative(GpuContextDemand),
    Dynamic(DynamicDemand),
}

enum RecipeEvent {
    Resource { context: usize, event: DemandEvent },
    Child(ChildNodePlan),
}

#[derive(Default)]
pub(super) struct ScopeDemandRecipe {
    contexts: BTreeMap<ContextKey, (usize, ConcreteMatrixType)>,
    events: Vec<RecipeEvent>,
    aliases: Vec<Arc<ValueAliases>>,
    pub(super) outputs: Vec<GpuInventoryValue>,
    pub(super) layouts: BTreeMap<WireRef, GpuInventoryValue>,
}

impl ScopeDemandRecipe {
    fn context(&mut self, ty: &ConcreteMatrixType) -> usize {
        let next = self.contexts.len();
        self.contexts
            .entry((ty.modulus.to_string(), ty.ring_dimension))
            .or_insert_with(|| (next, ty.clone()))
            .0
    }

    pub(super) fn matrix(&mut self, ty: &ConcreteMatrixType, claim: &MatrixDemand) {
        let context = self.context(ty);
        self.events
            .push(RecipeEvent::Resource { context, event: DemandEvent::Matrix(claim.clone()) });
    }

    pub(super) fn workspace(&mut self, ty: &ConcreteMatrixType, claim: &WorkspaceDemand) {
        let context = self.context(ty);
        self.events
            .push(RecipeEvent::Resource { context, event: DemandEvent::Workspace(claim.clone()) });
    }

    pub(super) fn preparation(&mut self, preparation: (MatrixInputOwner, usize, bool)) {
        let Some(RecipeEvent::Resource { event: DemandEvent::Matrix(claim), .. }) =
            self.events.last_mut()
        else {
            unreachable!("preparation immediately follows its matrix claim")
        };
        claim.preparation = Some(preparation);
    }

    pub(super) fn alternative(&mut self, ty: &ConcreteMatrixType, group: &GpuContextDemand) {
        let context = self.context(ty);
        self.events.push(RecipeEvent::Resource {
            context,
            event: DemandEvent::Alternative(group.clone()),
        });
    }

    pub(super) fn dynamic(&mut self, ty: &ConcreteMatrixType, event: DynamicDemand) {
        let context = self.context(ty);
        self.events.push(RecipeEvent::Resource { context, event: DemandEvent::Dynamic(event) });
    }

    pub(super) fn child(&mut self, plan: ChildNodePlan) {
        self.events.push(RecipeEvent::Child(plan));
    }

    pub(super) fn child_event_count(&self) -> usize {
        self.events.iter().filter(|event| matches!(event, RecipeEvent::Child(_))).count()
    }

    pub(super) fn finish(
        &mut self,
        contexts: &BTreeMap<ContextKey, (ConcreteMatrixType, GpuContextDemand)>,
        outputs: Vec<GpuInventoryValue>,
        layouts: BTreeMap<WireRef, GpuInventoryValue>,
    ) {
        for (ty, context) in contexts.values() {
            self.context(ty);
            if self.aliases.is_empty() {
                self.aliases.clone_from(&context.aliases);
            }
        }
        self.outputs = outputs;
        self.layouts = layouts;
    }

    pub(super) fn event_counts(&self) -> (usize, usize) {
        self.events.iter().fold((0, 0), |(fixed, dynamic), event| {
            if matches!(
                event,
                RecipeEvent::Child(_) |
                    RecipeEvent::Resource { event: DemandEvent::Dynamic(_), .. }
            ) {
                (fixed, dynamic + 1)
            } else {
                (fixed + 1, dynamic)
            }
        })
    }

    pub(super) fn replay(
        &self,
        backend: &mut GpuDcrtBackend,
        validated: &ValidatedGraph,
        width: usize,
        wave: usize,
    ) -> Result<BTreeMap<ContextKey, (ConcreteMatrixType, GpuContextDemand)>, PolyBackendError>
    {
        let mut contexts = self
            .contexts
            .iter()
            .map(|(key, (_, ty))| (key.clone(), (ty.clone(), GpuContextDemand::default())))
            .collect::<BTreeMap<_, _>>();
        self.append(backend, validated, width, wave, &mut contexts)?;
        for context in contexts.values_mut().map(|value| &mut value.1) {
            context.aliases.clone_from(&self.aliases);
        }
        Ok(contexts)
    }

    fn append(
        &self,
        backend: &mut GpuDcrtBackend,
        validated: &ValidatedGraph,
        width: usize,
        wave: usize,
        contexts: &mut BTreeMap<ContextKey, (ConcreteMatrixType, GpuContextDemand)>,
    ) -> Result<(), PolyBackendError> {
        let mut keys = vec![None; self.contexts.len()];
        for (key, value) in &self.contexts {
            keys[value.0] = Some(key);
        }
        for event in &self.events {
            if let RecipeEvent::Child(plan) = event {
                record_dynamic_child();
                let child = backend.evaluate_child_node(validated, plan, width, wave, false)?;
                if child.layout_updates != plan.expected_layout_updates {
                    return Err(PolyBackendError::GpuSubmission(
                        "root staged child output layouts changed between admission candidates"
                            .into(),
                    ));
                }
                child.append(contexts);
                continue;
            }
            let RecipeEvent::Resource { context: event_context, event } = event else {
                unreachable!("child event handled above")
            };
            let key = keys[*event_context].expect("compiled context");
            let context = &mut contexts.get_mut(key).expect("compiled context").1;
            match event {
                DemandEvent::Matrix(source) => {
                    if let Some(claim) = source.claim {
                        context.matrix_claim(source.begin, source.end, claim);
                    } else {
                        context.matrix(source.begin, source.end, source.rows, source.columns);
                    }
                    let target = context.matrix_demands.last_mut().unwrap();
                    target.owners.clone_from(&source.owners);
                    target.origins = source.origins.clone();
                    target.preparation = source.preparation;
                    target.ordinary = source.ordinary;
                }
                DemandEvent::Workspace(source) => {
                    context.workspace(source.begin, source.end, source.layout, source.shared);
                    let target = context.workspace_demands.last_mut().unwrap();
                    target.owners.clone_from(&source.owners);
                    target.origins = source.origins.clone();
                }
                DemandEvent::Alternative(group) => context.append_alternative_group(group.clone()),
                DemandEvent::Dynamic(event) => event(width, wave, context)?,
            }
        }
        Ok(())
    }
}
