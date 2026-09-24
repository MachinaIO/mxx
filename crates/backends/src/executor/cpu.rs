use super::*;
use crate::{
    backend::{
        PreimageRequest,
        poly::{CpuDcrtBackend, PolyBackendError},
    },
    matrix::{CpuSmallMatrix, PolyMatrix as PrimitivePolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::PolyParams,
    sampler::trapdoor::DCRTTrapdoor,
};

impl<S: SessionStore> Executor<'_, S> {
    fn execute_preimage_fresh_batch(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        node: &ExecutableNode<'_>,
        envs: &[ParamEnv],
        paths: &[Vec<InstantiationFrame>],
        placements: &[usize],
        values: &mut [BTreeMap<WireRef, RuntimeValue>],
    ) -> Result<bool, ExecutionError> {
        let NodeKind::PreimageSample { max_coefficient_bound, .. } = node.kind else {
            return Ok(false);
        };
        if envs.len() <= 1 ||
            self.session.is_some() ||
            !matches!(self.sampling_mode, SamplingMode::Fresh)
        {
            return Ok(false);
        }
        let mut prepared = Vec::with_capacity(envs.len());
        for index in 0..envs.len() {
            self.set_placement(placements[index])?;
            let public = self.matrix(&mut values[index], node.args[0])?;
            let (secret, trapdoor_public, _, sigma, gadget_base, digit_count, gadget_small) =
                self.trapdoor(&mut values[index], node.args[1])?;
            if gadget_small.is_some() {
                return Ok(false);
            }
            if !Arc::ptr_eq(&public, &trapdoor_public) &&
                public.as_ref() != trapdoor_public.as_ref()
            {
                return Err(ExecutionError::PreimagePublicMismatch(node.id));
            }
            let wire = WireRef { node: node.id, port: Port(0) };
            let (mut schema, semantic) =
                self.bounded_matrix_schema(scope_id, &paths[index], &envs[index], wire)?;
            if semantic != SmallMatrixSemanticKind::Preimage {
                return Err(ExecutionError::ValueKind(wire));
            }
            schema.max_coefficient_bound = max_coefficient_bound
                .evaluate_with_rings(&envs[index], crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| self.expression_error(node.id, error))?;
            self.backend
                .validate_preimage_bound(
                    &schema.matrix,
                    sigma,
                    &gadget_base,
                    digit_count,
                    &schema.max_coefficient_bound,
                )
                .map_err(Self::backend_error)?;
            let target_type =
                self.matrix_type(scope_id, &paths[index], &envs[index], node.args[2])?;
            let target = self.preimage_source(
                &mut values[index],
                &paths[index],
                node.args[2],
                &target_type,
            )?;
            let randomness_seed =
                preimage_request_seed(self.production.execution_nonce, &paths[index], wire);
            let secret = secret.expect("sampled trapdoor carries secret material");
            prepared.push(PreimageRequest {
                instance_slot: paths[index]
                    .last()
                    .and_then(|frame| frame.loop_index)
                    .map(|slot| slot as usize)
                    .unwrap_or(index),
                fixed_metadata: None,
                matrix_type: schema.matrix,
                sigma,
                gadget_base,
                digit_count,
                max_coefficient_bound: schema.max_coefficient_bound,
                trapdoor: secret,
                public,
                target,
                randomness_seed,
            });
        }
        let mut batches = Vec::new();
        let mut grouped_indices = Vec::new();
        for placement in 0..self.backend.placement_count() {
            let indices = placements
                .iter()
                .enumerate()
                .filter_map(|(index, candidate)| (*candidate == placement).then_some(index))
                .collect::<Vec<_>>();
            if indices.is_empty() {
                continue;
            }
            let requests = indices.iter().map(|index| prepared[*index].clone()).collect();
            grouped_indices.push(indices);
            batches.push((placement, requests));
        }
        let batches = self
            .backend
            .sample_preimage_batches_by_placement(batches)
            .map_err(Self::backend_error)?;
        if batches.len() != grouped_indices.len() {
            return Err(ExecutionError::InvalidBatch(node.id));
        }
        for ((_, outputs), indices) in batches.into_iter().zip(grouped_indices) {
            if outputs.len() != indices.len() {
                return Err(ExecutionError::InvalidBatch(node.id));
            }
            for (index, output) in indices.into_iter().zip(outputs) {
                self.put(&mut values[index], node.id, 0, RuntimeValue::preimage(output));
            }
        }
        self.record_preimages(envs.len());
        Ok(true)
    }

    fn shared_matrix_value(
        &self,
        scope_id: &FrozenGraphScopeId,
        env: &ParamEnv,
        wire: WireRef,
        value: Arc<DCRTPolyMatrix>,
    ) -> Result<RuntimeValue, ExecutionError> {
        let ty = self.resolved_wire_type(scope_id, wire, env)?;
        PolyMatrix::from_shared_cpu_full(ty, value)
            .map(RuntimeValue::Matrix)
            .map_err(|message| ExecutionError::Backend(message.to_owned()))
    }

    fn execute_fused_cpu_node(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        node: &ExecutableNode<'_>,
        envs: &[ParamEnv],
        paths: &[Vec<InstantiationFrame>],
        placements: &[usize],
        values: &mut [BTreeMap<WireRef, RuntimeValue>],
        aliases: &RootBlockAliases,
        row_blocks: &mut [BTreeMap<NodeId, Vec<Arc<DCRTPolyMatrix>>>],
        row_sum_sources: &mut [BTreeMap<
            NodeId,
            (Arc<DCRTPolyMatrix>, Option<Arc<DCRTPolyMatrix>>),
        >],
    ) -> Result<bool, ExecutionError> {
        let is_fused = aliases.tensor_row_sum_groups.contains_key(&node.id) ||
            aliases.row_sums.contains_key(&node.id) ||
            aliases.decompositions.contains_key(&node.id) ||
            aliases.compact_products.contains_key(&node.id) ||
            aliases.adds.contains_key(&node.id);
        if !is_fused {
            return Ok(false);
        }
        for placement in 0..self.backend.placement_count() {
            let indices = placements
                .iter()
                .enumerate()
                .filter_map(|(index, candidate)| (*candidate == placement).then_some(index))
                .collect::<Vec<_>>();
            if indices.is_empty() {
                continue;
            }
            self.set_placement(placement)?;
            let mut requests = Vec::with_capacity(indices.len());
            for &index in &indices {
                let request = if let Some(group) = aliases.tensor_row_sum_groups.get(&node.id) {
                    let (source, right) = row_sum_sources[index]
                        .remove(&node.id)
                        .expect("shared row-sum source survives until leader");
                    let right = right.ok_or(ExecutionError::InvalidBatch(node.id))?;
                    Some(DynamicFusedBatchRequest::TensorRowSums {
                        source,
                        right,
                        rows: group
                            .iter()
                            .map(|output| aliases.row_sums[output].rows.clone())
                            .collect(),
                    })
                } else if let Some(plan) = aliases.row_sums.get(&node.id) {
                    let (source, right) = row_sum_sources[index]
                        .remove(&node.id)
                        .expect("row-sum source survives until output");
                    Some(DynamicFusedBatchRequest::RowSum {
                        source,
                        right,
                        rows: plan.rows.clone(),
                    })
                } else if let Some(concat) = aliases.decompositions.get(&node.id) {
                    let NodeKind::GadgetDecompose { base, small, digit_count } = node.kind else {
                        return Err(ExecutionError::InvalidBatch(node.id));
                    };
                    let input_type =
                        self.matrix_type(scope_id, &paths[index], &envs[index], node.args[0])?;
                    let base = base
                        .evaluate_with_rings(
                            &envs[index],
                            crate::openfhe_guard::gen_modulus_and_warmup,
                        )
                        .map_err(|error| self.expression_error(node.id, error))?;
                    let digits = self.eval_usize(node.id, digit_count, &envs[index])?;
                    self.backend
                        .validate_gadget_layout(&input_type, &base, digits, *small)
                        .map_err(Self::backend_error)?;
                    let blocks = row_blocks[index]
                        .remove(concat)
                        .expect("decomposition blocks survive until use");
                    Some(DynamicFusedBatchRequest::Decompose { blocks, small: *small, digits })
                } else if let Some((concat, rhs, _)) = aliases.compact_products.get(&node.id) {
                    let rhs = self.small_matrix(&mut values[index], *rhs)?;
                    let blocks = row_blocks[index]
                        .remove(concat)
                        .expect("compact product blocks survive until use");
                    Some(DynamicFusedBatchRequest::SmallProduct { blocks, rhs })
                } else if let Some((concat, right)) = aliases.adds.get(&node.id) {
                    let right = self.matrix(&mut values[index], *right)?;
                    let blocks =
                        row_blocks[index].remove(concat).expect("add blocks survive until use");
                    Some(DynamicFusedBatchRequest::Add { blocks, right })
                } else {
                    None
                };
                requests.push(request.expect("fused node has a matching request"));
            }
            let results = self.backend.fused_batch(requests).map_err(Self::backend_error)?;
            if results.len() != indices.len() {
                return Err(ExecutionError::InvalidBatch(node.id));
            }
            for (index, result) in indices.into_iter().zip(results) {
                if let Some(group) = aliases.tensor_row_sum_groups.get(&node.id) {
                    for output in group {
                        row_sum_sources[index].remove(output);
                    }
                    let FusedBatchOutput::Matrices(outputs) = result else {
                        return Err(ExecutionError::InvalidBatch(node.id));
                    };
                    if outputs.len() != group.len() {
                        return Err(ExecutionError::InvalidBatch(node.id));
                    }
                    for (output, matrix) in group.iter().zip(outputs) {
                        self.put(&mut values[index], *output, 0, RuntimeValue::matrix(matrix));
                    }
                } else if aliases.decompositions.contains_key(&node.id) {
                    let FusedBatchOutput::Small(output) = result else {
                        return Err(ExecutionError::InvalidBatch(node.id));
                    };
                    let semantic = self
                        .bounded_matrix_schema(
                            scope_id,
                            &paths[index],
                            &envs[index],
                            WireRef { node: node.id, port: Port(0) },
                        )?
                        .1;
                    self.put(
                        &mut values[index],
                        node.id,
                        0,
                        compact_runtime_value(output, semantic),
                    );
                } else if let Some((_, _, output_aliases)) = aliases.compact_products.get(&node.id)
                {
                    let FusedBatchOutput::Matrices(outputs) = result else {
                        return Err(ExecutionError::InvalidBatch(node.id));
                    };
                    if outputs.len() != output_aliases.len() {
                        return Err(ExecutionError::InvalidBatch(node.id));
                    }
                    for (output, targets) in outputs.into_iter().zip(output_aliases) {
                        let shared = Arc::new(output);
                        for target in targets {
                            let value = self.shared_matrix_value(
                                scope_id,
                                &envs[index],
                                *target,
                                Arc::clone(&shared),
                            )?;
                            values[index].insert(*target, value);
                        }
                    }
                } else {
                    let FusedBatchOutput::Matrices(mut outputs) = result else {
                        return Err(ExecutionError::InvalidBatch(node.id));
                    };
                    if outputs.len() != 1 {
                        return Err(ExecutionError::InvalidBatch(node.id));
                    }
                    self.put(
                        &mut values[index],
                        node.id,
                        0,
                        RuntimeValue::matrix(outputs.remove(0)),
                    );
                }
                self.has_pending_releases = true;
            }
        }
        Ok(true)
    }

    fn indexed_family_output(
        &self,
        scope_id: &FrozenGraphScopeId,
        env: &ParamEnv,
        node: NodeId,
        port: u32,
        members: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, ExecutionError> {
        let wire = WireRef { node, port: Port(port) };
        let ConcreteWireType::IndexedFamily { element, count } =
            self.resolved_wire_type(scope_id, wire, env)?
        else {
            return Err(ExecutionError::MissingMetadata(WireId {
                instantiation_path: Vec::new(),
                wire,
            }));
        };
        if members.len() != count {
            return Err(ExecutionError::ValueKind(wire));
        }
        RuntimeValue::indexed_family(element.as_ref().clone(), members)
            .map_err(|message| ExecutionError::Backend(message.to_owned()))
    }

    pub(super) fn execute_instance(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        env: &ParamEnv,
        path: Vec<InstantiationFrame>,
        inputs: BTreeMap<String, RuntimeValue>,
        placement: usize,
    ) -> Result<InstanceResult, ExecutionError> {
        self.execute_instances_batch(
            scope_id,
            std::slice::from_ref(env),
            &[path],
            &[inputs],
            &[placement],
        )
        .map(|mut instances| instances.pop().expect("single execution returns one instance"))
    }

    fn execute_instances_batch(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        envs: &[ParamEnv],
        paths: &[Vec<InstantiationFrame>],
        inputs: &[BTreeMap<String, RuntimeValue>],
        placements: &[usize],
    ) -> Result<Vec<InstanceResult>, ExecutionError> {
        assert_eq!(envs.len(), paths.len());
        assert_eq!(envs.len(), inputs.len());
        assert_eq!(envs.len(), placements.len());
        let scope = self.validated.source.scope(scope_id).ok_or_else(|| {
            ExecutionError::MissingSubgraph { node: NodeId(0), name: format!("{scope_id:?}") }
        })?;
        let validated_scope = self.validated.scope(scope_id).ok_or_else(|| {
            ExecutionError::MissingSubgraph { node: NodeId(0), name: format!("{scope_id:?}") }
        })?;
        let aliases = root_block_aliases(self.validated, scope_id, self.trace.is_some());
        if envs.len() == 1 &&
            self.trace.is_none() &&
            self.config.release_fence_interval.is_none() &&
            !tracing::enabled!(tracing::Level::INFO)
        {
            if let Some((node, names)) = &aliases.input_row_sum {
                let matrices = names
                    .iter()
                    .map(|name| match inputs[0].get(name) {
                        Some(RuntimeValue::Matrix(matrix)) => matrix.cpu_full_arc(),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>();
                if let Some(matrices) = matrices {
                    self.set_placement(placements[0])?;
                    let row_sum = &aliases.row_sums[node];
                    let output = if let [left, right] = matrices.as_slice() {
                        self.backend.tensor_sum_rows(left, right, &row_sum.rows)
                    } else {
                        self.backend.sum_rows(&matrices[0], &row_sum.rows)
                    }
                    .map_err(Self::backend_error)?;
                    self.executed_node_count = self
                        .executed_node_count
                        .saturating_add(validated_scope.execution_order.len());
                    self.has_pending_releases = true;
                    return Ok(vec![InstanceResult { outputs: vec![RuntimeValue::matrix(output)] }]);
                }
            }
        }
        let mut values = vec![BTreeMap::<WireRef, RuntimeValue>::new(); envs.len()];
        let mut row_blocks = vec![BTreeMap::<NodeId, Vec<Arc<DCRTPolyMatrix>>>::new(); envs.len()];
        let mut row_sum_sources =
            vec![
                BTreeMap::<NodeId, (Arc<DCRTPolyMatrix>, Option<Arc<DCRTPolyMatrix>>)>::new();
                envs.len()
            ];
        let mut position = 0;
        while position < validated_scope.execution_order.len() {
            if let Some(end) = aliases.absent_run_end(
                position,
                self.config.release_fence_interval,
                tracing::enabled!(tracing::Level::INFO),
            ) {
                self.executed_node_count = self
                    .executed_node_count
                    .saturating_add((end - position).saturating_mul(envs.len()));
                position = end;
                continue;
            }
            let handle = &validated_scope.execution_order[position];
            let arguments = scope.arguments(handle).expect("validated node belongs to scope");
            let node = ExecutableNode {
                id: NodeId(position as u64),
                kind: handle.kind(),
                args: &arguments,
            };
            if let Some(outputs) = aliases.row_sum_captures.get(&node.id) {
                for index in 0..envs.len() {
                    self.set_placement(placements[index])?;
                    let mut tensor: Option<(Arc<DCRTPolyMatrix>, Option<Arc<DCRTPolyMatrix>>)> =
                        None;
                    for output in outputs {
                        let plan = &aliases.row_sums[output];
                        let source = if let Some([left, right]) = plan.tensor_operands {
                            if let Some(pair) = &tensor {
                                pair.clone()
                            } else {
                                let pair = (
                                    self.matrix(&mut values[index], left)?,
                                    Some(self.matrix(&mut values[index], right)?),
                                );
                                tensor = Some(pair.clone());
                                pair
                            }
                        } else {
                            (self.matrix(&mut values[index], plan.source)?, None)
                        };
                        row_sum_sources[index].insert(*output, source);
                    }
                }
            }
            if aliases.row_block_concats.contains(&node.id) {
                for index in 0..envs.len() {
                    self.set_placement(placements[index])?;
                    let blocks = node
                        .args
                        .iter()
                        .map(|wire| self.matrix(&mut values[index], *wire))
                        .collect::<Result<Vec<_>, _>>()?;
                    row_blocks[index].insert(node.id, blocks);
                }
            }
            if let Some(links) = aliases.concats.get(&node.id) {
                for index in 0..envs.len() {
                    self.set_placement(placements[index])?;
                    let matrices = node
                        .args
                        .iter()
                        .map(|wire| self.matrix(&mut values[index], *wire))
                        .collect::<Result<Vec<_>, _>>()?;
                    for (output, source) in links {
                        let source_index = node
                            .args
                            .iter()
                            .position(|wire| wire == source)
                            .expect("validated concat alias source");
                        let value = self.shared_matrix_value(
                            scope_id,
                            &envs[index],
                            *output,
                            matrices[source_index].clone(),
                        )?;
                        values[index].insert(*output, value);
                    }
                }
            }
            let aliased = aliases.slices.contains(&node.id) ||
                aliases.concats.contains_key(&node.id) ||
                aliases.row_block_concats.contains(&node.id) ||
                aliases.row_sum_interiors.contains(&node.id) ||
                aliases
                    .tensor_row_sum_leaders
                    .get(&node.id)
                    .is_some_and(|leader| *leader != node.id);
            let fused = if aliased {
                true
            } else {
                self.execute_fused_cpu_node(
                    scope_id,
                    &node,
                    envs,
                    paths,
                    placements,
                    &mut values,
                    &aliases,
                    &mut row_blocks,
                    &mut row_sum_sources,
                )?
            };
            let sampled = if fused {
                false
            } else {
                self.execute_preimage_fresh_batch(
                    scope_id,
                    &node,
                    envs,
                    paths,
                    placements,
                    &mut values,
                )?
            };
            let batched = !fused &&
                !sampled &&
                envs.len() > 1 &&
                matches!(
                    node.kind,
                    NodeKind::MatrixBinary(_) |
                        NodeKind::MatrixMulAccumulate { .. } |
                        NodeKind::MatrixNegate |
                        NodeKind::MatrixScale { .. }
                );
            if batched {
                for placement in 0..self.backend.placement_count() {
                    let indices = placements
                        .iter()
                        .enumerate()
                        .filter_map(|(index, candidate)| (*candidate == placement).then_some(index))
                        .collect::<Vec<_>>();
                    if !indices.is_empty() {
                        self.execute_parallel_matrix_node(
                            placement,
                            envs,
                            &node,
                            &mut values,
                            &indices,
                        )?;
                    }
                }
            }
            for index in 0..envs.len() {
                self.set_placement(placements[index])?;
                if !batched && !fused && !sampled {
                    self.execute_node(
                        scope_id,
                        &envs[index],
                        &paths[index],
                        &node,
                        &inputs[index],
                        &mut values[index],
                    )?;
                }
                if let Some(trace) = &mut self.trace {
                    for (wire, value) in
                        values[index].iter().filter(|(wire, _)| wire.node == node.id)
                    {
                        trace.insert(
                            WireId { instantiation_path: paths[index].clone(), wire: *wire },
                            value.clone(),
                        );
                    }
                }
                release_last_uses(
                    &validated_scope.liveness,
                    position,
                    node.args,
                    &mut values[index],
                );
            }
            self.executed_node_count = self.executed_node_count.saturating_add(envs.len());
            if self.has_pending_releases &&
                self.config.release_fence_interval.is_some_and(|interval| {
                    self.executed_node_count.saturating_sub(self.last_release_fence_node_count) >=
                        interval.get()
                })
            {
                self.fence_pending_releases()?;
            }
            position += 1;
        }
        values
            .into_iter()
            .enumerate()
            .map(|(index, mut values)| {
                self.set_placement(placements[index])?;
                let outputs = scope
                    .outputs()
                    .iter()
                    .map(|wire| self.materialize(&mut values, *wire))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(InstanceResult { outputs })
            })
            .collect()
    }

    fn execute_parallel_matrix_node(
        &mut self,
        placement: usize,
        envs: &[ParamEnv],
        node: &ExecutableNode<'_>,
        values: &mut [BTreeMap<WireRef, RuntimeValue>],
        indices: &[usize],
    ) -> Result<(), ExecutionError> {
        self.set_placement(placement)?;
        let outputs = match node.kind {
            NodeKind::MatrixBinary(operation) => {
                let mut inputs = Vec::with_capacity(indices.len());
                for index in indices {
                    let instance = &mut values[*index];
                    let left = self.matrix(instance, node.args[0])?;
                    let right = self.matrix(instance, node.args[1])?;
                    inputs.push((left, right));
                }
                match operation {
                    MatrixBinaryOp::Add => self.backend.add_batch(inputs),
                    MatrixBinaryOp::Subtract => self.backend.sub_batch(inputs),
                    MatrixBinaryOp::Multiply => self.backend.multiply_batch(inputs),
                }
                .map_err(Self::backend_error)?
            }
            NodeKind::MatrixMulAccumulate { coefficients, has_bias } => {
                let mut requests = Vec::with_capacity(indices.len());
                for index in indices {
                    let env = &envs[*index];
                    let instance = &mut values[*index];
                    let mut products = Vec::with_capacity(coefficients.len());
                    for (product, coefficient) in coefficients.iter().enumerate() {
                        products.push((
                            coefficient
                                .evaluate_with_rings(
                                    env,
                                    crate::openfhe_guard::gen_modulus_and_warmup,
                                )
                                .map_err(|error| self.expression_error(node.id, error))?,
                            self.matrix(instance, node.args[2 * product])?,
                            self.matrix(instance, node.args[2 * product + 1])?,
                        ));
                    }
                    let bias = if *has_bias {
                        Some(self.matrix(instance, node.args[2 * coefficients.len()])?)
                    } else {
                        None
                    };
                    requests.push(MatrixMulAccumulateRequest { products, bias });
                }
                self.backend.matrix_mul_accumulate_batch(requests).map_err(Self::backend_error)?
            }
            NodeKind::MatrixNegate => {
                let mut inputs = Vec::with_capacity(indices.len());
                for index in indices {
                    let instance = &mut values[*index];
                    inputs.push(self.matrix(instance, node.args[0])?);
                }
                self.backend.negate_batch(inputs).map_err(Self::backend_error)?
            }
            NodeKind::MatrixScale { scalar } => {
                let mut inputs = Vec::with_capacity(indices.len());
                for index in indices {
                    let env = &envs[*index];
                    let instance = &mut values[*index];
                    let value = self.matrix(instance, node.args[0])?;
                    let scalar = scalar
                        .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                        .map_err(|error| self.expression_error(node.id, error))?;
                    inputs.push((value, scalar));
                }
                self.backend.scale_integer_batch(inputs).map_err(Self::backend_error)?
            }
            _ => unreachable!("matrix batch kind checked by caller"),
        };
        if outputs.len() != indices.len() {
            return Err(ExecutionError::InvalidBatch(node.id));
        }
        for (index, output) in indices.iter().zip(outputs) {
            self.put(&mut values[*index], node.id, 0, RuntimeValue::matrix(output));
        }
        Ok(())
    }

    /// The key and the tag bytes of a hash sampler node: the static prefix,
    /// then each length-framed typed component.
    fn hash_key_and_tag(
        &mut self,
        values: &mut BTreeMap<WireRef, RuntimeValue>,
        node: &ExecutableNode<'_>,
        env: &ParamEnv,
        tag_prefix: &[u8],
        tag_components: &[mxx_ir_core::node::HashTagComponent],
    ) -> Result<([u8; 32], Vec<u8>), ExecutionError> {
        let key = self.bytes(values, node.args[0])?;
        let key: [u8; 32] = key.try_into().map_err(|_| ExecutionError::ValueKind(node.args[0]))?;
        let mut tag = tag_prefix.to_vec();
        for component in tag_components {
            use mxx_ir_core::node::HashTagComponent;
            match component {
                HashTagComponent::Bytes(bytes) => {
                    tag.push(0);
                    tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
                    tag.extend_from_slice(bytes);
                }
                HashTagComponent::Integer(expression) => {
                    let value = expression
                        .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                        .map_err(|error| self.expression_error(node.id, error))?;
                    tag.push(1);
                    append_tag_integer(&mut tag, &value);
                }
                HashTagComponent::Decimal(expression) => {
                    let value = expression
                        .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                        .map_err(|error| self.expression_error(node.id, error))?;
                    let decimal = value.to_string();
                    tag.push(2);
                    tag.extend_from_slice(&(decimal.len() as u64).to_be_bytes());
                    tag.extend_from_slice(decimal.as_bytes());
                }
                HashTagComponent::U64Le(expression) => {
                    let value = expression
                        .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                        .map_err(|error| self.expression_error(node.id, error))?
                        .to_u64()
                        .ok_or_else(|| ExecutionError::Expression {
                            node: node.id,
                            message: "little-endian hash tag component must fit in u64".to_owned(),
                        })?;
                    tag.push(3);
                    tag.extend_from_slice(&value.to_le_bytes());
                }
                HashTagComponent::Operand(index) => {
                    tag.push(1);
                    append_tag_integer(&mut tag, &self.int(values, node.args[*index])?);
                }
            }
        }
        Ok((key, tag))
    }

    fn execute_node(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        env: &ParamEnv,
        path: &[InstantiationFrame],
        node: &ExecutableNode<'_>,
        inputs: &BTreeMap<String, RuntimeValue>,
        values: &mut BTreeMap<WireRef, RuntimeValue>,
    ) -> Result<(), ExecutionError> {
        match &node.kind {
            NodeKind::Input { name, wire_type: _, artifact } => {
                if let Some(artifact) = artifact {
                    let wire = WireRef { node: node.id, port: Port(0) };
                    let wire_id = WireId { instantiation_path: path.to_vec(), wire };
                    let concrete = self.resolved_wire_type(scope_id, wire, env)?;
                    let descriptor = self
                        .validated
                        .scope(scope_id)
                        .and_then(|scope| scope.artifact_inputs.get(&wire))
                        .cloned()
                        .ok_or_else(|| ExecutionError::MissingMetadata(wire_id.clone()))?;
                    if let ConcreteWireType::IndexedFamily { element, count: declared_count } =
                        &concrete
                    {
                        let artifact_type =
                            ArtifactType::from_wire_type(element).ok_or_else(|| {
                                ExecutionError::Manifest(
                                    "indexed artifact has unsupported element type".to_owned(),
                                )
                            })?;
                        if descriptor.artifact_type != artifact_type ||
                            descriptor.family_count != Some(*declared_count)
                        {
                            return Err(ExecutionError::Manifest(
                                "validated artifact descriptor does not match its wire metadata"
                                    .to_owned(),
                            ));
                        }
                        values.insert(
                            wire,
                            RuntimeValue::LazyArtifactFamily {
                                production: artifact.production_id.clone(),
                                name: artifact.artifact_name.clone(),
                                descriptor,
                            },
                        );
                        return Ok(());
                    }
                    let artifact_type =
                        ArtifactType::from_wire_type(&concrete).ok_or_else(|| {
                            ExecutionError::Manifest(
                                "artifact has unsupported wire type".to_owned(),
                            )
                        })?;
                    if descriptor.artifact_type != artifact_type ||
                        descriptor.family_count.is_some()
                    {
                        return Err(ExecutionError::Manifest(
                            "validated artifact descriptor does not match its wire metadata"
                                .to_owned(),
                        ));
                    }
                    values.insert(
                        wire,
                        RuntimeValue::LazyArtifact {
                            production: artifact.production_id.clone(),
                            name: artifact.artifact_name.clone(),
                            index: None,
                            descriptor,
                        },
                    );
                } else {
                    let wire = WireRef { node: node.id, port: Port(0) };
                    let concrete = self.resolved_wire_type(scope_id, wire, env)?;
                    let value =
                        clone_typed_runtime_input(inputs, name, &concrete).map_err(|error| {
                            match error {
                                crate::host_control::RuntimeValueAccessError::MissingInput {
                                    ..
                                } => ExecutionError::MissingInput(name.clone()),
                                crate::host_control::RuntimeValueAccessError::TypeMismatch {
                                    ..
                                } |
                                crate::host_control::RuntimeValueAccessError::NotTrapdoor => {
                                    ExecutionError::ValueKind(wire)
                                }
                            }
                        })?;
                    values.insert(wire, value);
                }
            }
            NodeKind::ConstantInt(_) => {
                let output = dispatch_host_primitive(node.id, &node.kind, env, &[])
                    .map_err(|error| self.host_primitive_error(node.id, error))?;
                self.put(values, node.id, 0, Self::runtime_host_primitive(output));
            }
            NodeKind::EvaluateInt(_) => {
                let output = dispatch_host_primitive(node.id, &node.kind, env, &[])
                    .map_err(|error| self.host_primitive_error(node.id, error))?;
                self.put(values, node.id, 0, Self::runtime_host_primitive(output));
            }
            NodeKind::ConstantReal(_) => {
                let output = dispatch_host_primitive(node.id, &node.kind, env, &[])
                    .map_err(|error| self.host_primitive_error(node.id, error))?;
                self.put(values, node.id, 0, Self::runtime_host_primitive(output));
            }
            NodeKind::ConstantBool(_) => {
                let output = dispatch_host_primitive(node.id, &node.kind, env, &[])
                    .map_err(|error| self.host_primitive_error(node.id, error))?;
                self.put(values, node.id, 0, Self::runtime_host_primitive(output));
            }
            NodeKind::ConstantMatrix { value, .. } => {
                let ty = self.matrix_type(
                    scope_id,
                    path,
                    env,
                    WireRef { node: node.id, port: Port(0) },
                )?;
                let matrix =
                    self.backend.constant_matrix(&ty, value, env).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(matrix));
            }
            NodeKind::GadgetTrapdoor { .. } => {
                let trapdoor_wire = WireRef { node: node.id, port: Port(0) };
                let ty = self.trapdoor_type(scope_id, path, env, trapdoor_wire)?;
                let public = self
                    .backend
                    .constant_matrix(
                        &ty,
                        &mxx_ir_core::node::ConstantMatrix::Gadget {
                            base: match &node.kind {
                                NodeKind::GadgetTrapdoor { base, .. } => base.clone(),
                                _ => unreachable!(),
                            },
                            small: false,
                        },
                        env,
                    )
                    .map_err(Self::backend_error)?;
                let wire_type = self.resolved_wire_type(scope_id, trapdoor_wire, env)?;
                let trapdoor = TrapdoorValue::public_gadget(
                    wire_type,
                    PolyMatrix::from_backend_cpu_full(public),
                )
                .map_err(|message| ExecutionError::Backend(message.to_owned()))?;
                self.put(values, node.id, 0, RuntimeValue::Trapdoor(trapdoor));
            }
            NodeKind::TrapdoorPublic => {
                let value = self.materialize(values, node.args[0])?;
                let public = project_trapdoor_public(&value)
                    .map_err(|_| ExecutionError::ValueKind(node.args[0]))?;
                self.put(values, node.id, 0, RuntimeValue::Matrix(public));
            }
            NodeKind::IntBinary(_) => {
                let inputs = node
                    .args
                    .iter()
                    .map(|wire| self.host_primitive_value(values, *wire))
                    .collect::<Result<Vec<_>, _>>()?;
                let output = dispatch_host_primitive(node.id, &node.kind, env, &inputs)
                    .map_err(|error| self.host_primitive_error(node.id, error))?;
                self.put(values, node.id, 0, Self::runtime_host_primitive(output));
            }
            NodeKind::IntCompare(_) => {
                let inputs = node
                    .args
                    .iter()
                    .map(|wire| self.host_primitive_value(values, *wire))
                    .collect::<Result<Vec<_>, _>>()?;
                let output = dispatch_host_primitive(node.id, &node.kind, env, &inputs)
                    .map_err(|error| self.host_primitive_error(node.id, error))?;
                self.put(values, node.id, 0, Self::runtime_host_primitive(output));
            }
            NodeKind::BitExtract { bit: _ } => {
                let inputs = node
                    .args
                    .iter()
                    .map(|wire| self.host_primitive_value(values, *wire))
                    .collect::<Result<Vec<_>, _>>()?;
                let output = dispatch_host_primitive(node.id, &node.kind, env, &inputs)
                    .map_err(|error| self.host_primitive_error(node.id, error))?;
                self.put(values, node.id, 0, Self::runtime_host_primitive(output));
            }
            NodeKind::PolynomialFromValues { matrix_type, evaluation } => {
                let concrete = mxx_ir_core::concretize_wire_type(
                    &mxx_ir_core::types::WireType::Matrix(matrix_type.clone()),
                    env,
                    scope_id,
                    node.id,
                    crate::openfhe_guard::gen_modulus_and_warmup,
                )
                .map_err(|error| self.expression_error(node.id, error))?;
                let ConcreteWireType::Matrix(ty) = concrete else { unreachable!() };
                let wire = node.args[0];
                let count = self.family_count(values, wire)?;
                // Materialization borrows the artifact store and backend mutably.
                let members = (0..count)
                    .map(|index| match self.family_member(values, wire, index, node.id)? {
                        RuntimeValue::Int(value) => Ok(value),

                        _ => Err(ExecutionError::ValueKind(wire)),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let output = crate::host_control::dispatch_polynomial_from_values(
                    self.backend,
                    &ty,
                    &members,
                    *evaluation,
                )
                .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::PolynomialValues { evaluation } => {
                let input = self.matrix(values, node.args[0])?;
                let output = crate::host_control::dispatch_polynomial_values(
                    self.backend,
                    &input,
                    *evaluation,
                )
                .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::integer_values(output));
            }
            NodeKind::IntToReal => {
                let inputs = node
                    .args
                    .iter()
                    .map(|wire| self.host_primitive_value(values, *wire))
                    .collect::<Result<Vec<_>, _>>()?;
                let output = dispatch_host_primitive(node.id, &node.kind, env, &inputs)
                    .map_err(|error| self.host_primitive_error(node.id, error))?;
                self.put(values, node.id, 0, Self::runtime_host_primitive(output));
            }
            NodeKind::BoolToInt => {
                let inputs = node
                    .args
                    .iter()
                    .map(|wire| self.host_primitive_value(values, *wire))
                    .collect::<Result<Vec<_>, _>>()?;
                let output = dispatch_host_primitive(node.id, &node.kind, env, &inputs)
                    .map_err(|error| self.host_primitive_error(node.id, error))?;
                self.put(values, node.id, 0, Self::runtime_host_primitive(output));
            }
            NodeKind::RealBinary(_) => {
                let inputs = node
                    .args
                    .iter()
                    .map(|wire| self.host_primitive_value(values, *wire))
                    .collect::<Result<Vec<_>, _>>()?;
                let output = dispatch_host_primitive(node.id, &node.kind, env, &inputs)
                    .map_err(|error| self.host_primitive_error(node.id, error))?;
                self.put(values, node.id, 0, Self::runtime_host_primitive(output));
            }
            NodeKind::RealSqrt => {
                let inputs = node
                    .args
                    .iter()
                    .map(|wire| self.host_primitive_value(values, *wire))
                    .collect::<Result<Vec<_>, _>>()?;
                let output = dispatch_host_primitive(node.id, &node.kind, env, &inputs)
                    .map_err(|error| self.host_primitive_error(node.id, error))?;
                self.put(values, node.id, 0, Self::runtime_host_primitive(output));
            }
            NodeKind::MatrixBinary(operation) => {
                let left = self.matrix(values, node.args[0])?;
                let right = self.matrix(values, node.args[1])?;
                let output = match operation {
                    MatrixBinaryOp::Add => self.backend.add(&left, &right),
                    MatrixBinaryOp::Subtract => self.backend.sub(&left, &right),
                    MatrixBinaryOp::Multiply => self.backend.multiply(&left, &right),
                }
                .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::MatrixMulSmallRhs => {
                let lhs = self.matrix(values, node.args[0])?;
                let rhs = self.small_matrix(values, node.args[1])?;
                let output = self
                    .backend
                    .multiply_small_rhs(lhs.as_ref(), rhs.as_ref())
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::MatrixMulAccumulate { coefficients, has_bias } => {
                let mut products = Vec::with_capacity(coefficients.len());
                for (product, coefficient) in coefficients.iter().enumerate() {
                    products.push((
                        coefficient
                            .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                            .map_err(|error| self.expression_error(node.id, error))?,
                        self.matrix(values, node.args[2 * product])?,
                        self.matrix(values, node.args[2 * product + 1])?,
                    ));
                }
                let bias = if *has_bias {
                    Some(self.matrix(values, node.args[2 * coefficients.len()])?)
                } else {
                    None
                };
                let output = self
                    .backend
                    .matrix_mul_accumulate(MatrixMulAccumulateRequest { products, bias })
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::MatrixNegate => {
                let input = self.matrix(values, node.args[0])?;
                let output = self.backend.negate(&input).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::MatrixScale { scalar } => {
                let input = self.matrix(values, node.args[0])?;
                let scalar = scalar
                    .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| self.expression_error(node.id, error))?;
                let output =
                    self.backend.scale_integer(&input, &scalar).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::Transpose => {
                let input = self.matrix(values, node.args[0])?;
                let output = self.backend.transpose(&input).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::Slice { rows, columns } => {
                let input = self.matrix(values, node.args[0])?;
                let rows = rows
                    .as_ref()
                    .map(|range| {
                        Ok::<_, ExecutionError>(RuntimeIndexRange {
                            start: self.eval_usize(node.id, &range.start, env)?,
                            end: self.eval_usize(node.id, &range.end, env)?,
                        })
                    })
                    .transpose()?;
                let columns = columns
                    .as_ref()
                    .map(|range| {
                        Ok::<_, ExecutionError>(RuntimeIndexRange {
                            start: self.eval_usize(node.id, &range.start, env)?,
                            end: self.eval_usize(node.id, &range.end, env)?,
                        })
                    })
                    .transpose()?;
                let output = self
                    .backend
                    .slice(&input, rows.as_ref(), columns.as_ref())
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::Tensor => {
                let left = self.matrix(values, node.args[0])?;
                let right = self.matrix(values, node.args[1])?;
                let output = self.backend.tensor(&left, &right).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::Concat { axis } => {
                let inputs = node
                    .args
                    .iter()
                    .map(|wire| self.matrix(values, *wire))
                    .collect::<Result<Vec<_>, _>>()?;
                let inputs = inputs.iter().map(Arc::as_ref).collect::<Vec<_>>();
                let output = self.backend.concat(&inputs, *axis).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::UniformResidueSample { .. } => {
                let wire = WireRef { node: node.id, port: Port(0) };
                let ty = self.matrix_type(scope_id, path, env, wire)?;
                let range = RuntimeSampleRange {
                    minimum: BigInt::from(0),
                    maximum: ty.ring.modulus() - BigInt::from(1),
                };
                let value = self.sample_matrix(path, wire, &ty, |backend| {
                    backend.sample_uniform(&ty, &range)
                })?;
                self.put(values, node.id, 0, RuntimeValue::matrix(value));
            }
            NodeKind::UniformIntervalSample { range, .. } => {
                let wire = WireRef { node: node.id, port: Port(0) };
                let ty = self.matrix_type(scope_id, path, env, wire)?;
                let range = RuntimeSampleRange {
                    minimum: range
                        .minimum
                        .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                        .map_err(|error| self.expression_error(node.id, error))?,
                    maximum: range
                        .maximum
                        .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                        .map_err(|error| self.expression_error(node.id, error))?,
                };
                let value = self.sample_matrix(path, wire, &ty, |backend| {
                    backend.sample_uniform(&ty, &range)
                })?;
                self.put(values, node.id, 0, RuntimeValue::matrix(value));
            }
            NodeKind::GaussianSample { sigma, max_coefficient_bound, .. } => {
                let wire = WireRef { node: node.id, port: Port(0) };
                let ty = self.matrix_type(scope_id, path, env, wire)?;
                let sigma = sigma
                    .evaluate_f64_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| self.expression_error(node.id, error))?;
                let max_coefficient_bound = max_coefficient_bound
                    .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| self.expression_error(node.id, error))?;
                let value = self.sample_matrix(path, wire, &ty, |backend| {
                    backend.sample_gaussian(&ty, sigma, &max_coefficient_bound)
                })?;
                self.put(values, node.id, 0, RuntimeValue::matrix(value));
            }
            NodeKind::HashIntFamily { count, modulus, tag_prefix, tag_components } => {
                let (key, tag) =
                    self.hash_key_and_tag(values, node, env, tag_prefix, tag_components)?;
                let count = self.eval_usize(node.id, count, env)?;
                let modulus = modulus
                    .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| self.expression_error(node.id, error))?
                    .to_biguint()
                    .ok_or_else(|| ExecutionError::Expression {
                        node: node.id,
                        message: "hash integer family modulus is negative".to_owned(),
                    })?;
                let sampled = crate::sampler::hash::sample_hash_integers::<keccak_asm::Keccak256>(
                    key, &tag, count, &modulus,
                );
                self.put(
                    values,
                    node.id,
                    0,
                    RuntimeValue::integer_values(sampled.into_iter().map(BigInt::from).collect()),
                );
            }
            NodeKind::HashSample {
                variant, tag_prefix, tag_components, base, digit_count, ..
            } => {
                let (key, tag) =
                    self.hash_key_and_tag(values, node, env, tag_prefix, tag_components)?;
                let wire = WireRef { node: node.id, port: Port(0) };
                let ty = self.matrix_type(scope_id, path, env, wire)?;
                let gadget_base = base
                    .as_ref()
                    .map(|base| {
                        base.evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                            .map_err(|error| self.expression_error(node.id, error))
                    })
                    .transpose()?;
                let digit_count = digit_count
                    .as_ref()
                    .map(|count| self.eval_usize(node.id, count, env))
                    .transpose()?;
                match (variant, gadget_base.as_ref(), digit_count) {
                    (HashVariant::Plain, None, None) => {
                        let value = self.sample_matrix(path, wire, &ty, |backend| {
                            backend.sample_hash(&ty, key, &tag)
                        })?;
                        self.put(values, node.id, 0, RuntimeValue::matrix(value));
                    }
                    (
                        HashVariant::Decomposed | HashVariant::SmallDecomposed,
                        Some(base),
                        Some(count),
                    ) => {
                        if count == 0 || ty.rows % count != 0 {
                            return Err(ExecutionError::Expression {
                                node: node.id,
                                message: "decomposed hash rows must be divisible by digit count"
                                    .to_owned(),
                            });
                        }
                        let (schema, semantic_kind) =
                            self.bounded_matrix_schema(scope_id, path, env, wire)?;
                        if semantic_kind != SmallMatrixSemanticKind::Generic {
                            return Err(ExecutionError::Manifest(
                                "decomposed hash output is not a generic small matrix".to_owned(),
                            ));
                        }
                        let value = self.sample_small_matrix(
                            path,
                            wire,
                            &schema,
                            semantic_kind,
                            |backend| match variant {
                                HashVariant::Decomposed => {
                                    backend.sample_hash_decomposed(&ty, key, &tag, base, count)
                                }
                                HashVariant::SmallDecomposed => backend
                                    .sample_hash_small_decomposed(&ty, key, &tag, base, count),
                                HashVariant::Plain => unreachable!("plain hash handled above"),
                            },
                        )?;
                        self.put(values, node.id, 0, RuntimeValue::small_matrix(value));
                    }
                    _ => {
                        return Err(ExecutionError::Expression {
                            node: node.id,
                            message: "hash variant and gadget layout do not match".to_owned(),
                        });
                    }
                }
            }
            NodeKind::TrapdoorSample { sigma, gadget_base, digit_count, .. } => {
                let matrix_wire = WireRef { node: node.id, port: Port(0) };
                let trapdoor_wire = WireRef { node: node.id, port: Port(1) };
                let ty = self.matrix_type(scope_id, path, env, matrix_wire)?;
                let sigma = sigma
                    .evaluate_f64_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| self.expression_error(node.id, error))?;
                let gadget_base = gadget_base
                    .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| self.expression_error(node.id, error))?
                    .abs();
                let digit_count = self.eval_usize(node.id, digit_count, env)?;
                let (public, secret) = self.sample_trapdoor(
                    path,
                    matrix_wire,
                    trapdoor_wire,
                    &ty,
                    sigma,
                    &gadget_base,
                    digit_count,
                )?;
                let public = PolyMatrix::from_backend_cpu_full(public);
                self.put(values, node.id, 0, RuntimeValue::Matrix(public.clone()));
                let wire_type = self.resolved_wire_type(scope_id, trapdoor_wire, env)?;
                let trapdoor = TrapdoorValue::new(wire_type, public, Some(Arc::new(secret)))
                    .map_err(|message| ExecutionError::Backend(message.to_owned()))?;
                self.put(values, node.id, 1, RuntimeValue::Trapdoor(trapdoor));
            }
            NodeKind::PreimageSample { max_coefficient_bound, .. } => {
                let public = self.matrix(values, node.args[0])?;
                let (secret, trapdoor_public, _, sigma, gadget_base, digit_count, gadget_small) =
                    self.trapdoor(values, node.args[1])?;
                if !Arc::ptr_eq(&public, &trapdoor_public) &&
                    public.as_ref() != trapdoor_public.as_ref()
                {
                    return Err(ExecutionError::PreimagePublicMismatch(node.id));
                }
                let target_type = self.matrix_type(scope_id, path, env, node.args[2])?;
                let wire = WireRef { node: node.id, port: Port(0) };
                let (mut schema, semantic_kind) =
                    self.bounded_matrix_schema(scope_id, path, env, wire)?;
                schema.max_coefficient_bound = max_coefficient_bound
                    .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| self.expression_error(node.id, error))?;
                if semantic_kind != SmallMatrixSemanticKind::Preimage {
                    return Err(ExecutionError::Manifest(
                        "preimage sampler output is not relation typed".to_owned(),
                    ));
                }
                let (value, sampled) = if let Some(small) = gadget_small {
                    let target = self.matrix(values, node.args[2])?;
                    if !small &&
                        self.backend
                            .gadget_error_bound(&target_type, Some(digit_count))
                            .map_err(Self::backend_error)? !=
                            BigInt::from(0u8)
                    {
                        return Err(ExecutionError::Manifest(
                            "exact public gadget preimage requires dropped_moduli = 0".into(),
                        ));
                    }
                    self.backend
                        .validate_gadget_layout(&target_type, &gadget_base, digit_count, small)
                        .map_err(Self::backend_error)?;
                    (
                        self.backend
                            .gadget_decompose(&target, small, Some(digit_count))
                            .map_err(Self::backend_error)?,
                        false,
                    )
                } else {
                    let secret =
                        secret.as_ref().expect("sampled trapdoor must carry secret material");
                    let target_source =
                        self.preimage_source(values, path, node.args[2], &target_type)?;
                    let randomness_seed =
                        preimage_request_seed(self.production.execution_nonce, path, wire);
                    self.sample_small_matrix_with_status(
                        path,
                        wire,
                        &schema,
                        semantic_kind,
                        |backend| {
                            backend.sample_preimage(
                                &schema.matrix,
                                sigma,
                                &gadget_base,
                                digit_count,
                                &schema.max_coefficient_bound,
                                secret,
                                &public,
                                target_source.as_ref(),
                                randomness_seed,
                            )
                        },
                    )?
                };
                if value.bound_domain() != schema.bound_domain ||
                    value.max_coefficient_bound().clone() !=
                        schema.max_coefficient_bound.to_biguint().ok_or_else(|| {
                            ExecutionError::Manifest("preimage bound must be nonnegative".into())
                        })? ||
                    value.size() != (schema.matrix.rows, schema.matrix.columns) ||
                    value.value().params().moduli() != schema.matrix.ring.crt_moduli() ||
                    value.value().params().ring_dimension() !=
                        schema.matrix.ring.ring_dimension()
                {
                    return Err(ExecutionError::Manifest(
                        "preimage owner does not match its validated bounded wire type".into(),
                    ));
                }
                if sampled {
                    self.record_preimages(1);
                }
                self.put(
                    values,
                    node.id,
                    0,
                    compact_runtime_value(value, SmallMatrixSemanticKind::Preimage),
                );
            }
            NodeKind::GadgetDecompose { base, small, digit_count } => {
                let input = self.matrix(values, node.args[0])?;
                let input_type = self.matrix_type(scope_id, path, env, node.args[0])?;
                let output_type = self.matrix_type(
                    scope_id,
                    path,
                    env,
                    WireRef { node: node.id, port: Port(0) },
                )?;
                let base = base
                    .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| self.expression_error(node.id, error))?;
                let digit_count = self.eval_usize(node.id, digit_count, env)?;
                self.backend
                    .validate_gadget_layout(&input_type, &base, digit_count, *small)
                    .map_err(Self::backend_error)?;
                let expected_rows = input_type.rows.checked_mul(digit_count).ok_or_else(|| {
                    ExecutionError::Expression {
                        node: node.id,
                        message: "gadget decomposition output row count overflow".to_owned(),
                    }
                })?;
                if output_type.rows != expected_rows {
                    return Err(ExecutionError::Expression {
                        node: node.id,
                        message: "gadget decomposition output type disagrees with digit count"
                            .to_owned(),
                    });
                }
                let output = self
                    .backend
                    .gadget_decompose(&input, *small, Some(digit_count))
                    .map_err(Self::backend_error)?;
                let semantic_kind = self
                    .bounded_matrix_schema(
                        scope_id,
                        path,
                        env,
                        WireRef { node: node.id, port: Port(0) },
                    )?
                    .1;
                self.put(values, node.id, 0, compact_runtime_value(output, semantic_kind));
            }
            NodeKind::ModulusSwitch { .. } |
            NodeKind::ModulusReduce { .. } |
            NodeKind::CenteredRebase { .. } => {
                let ty = self.matrix_type(
                    scope_id,
                    path,
                    env,
                    WireRef { node: node.id, port: Port(0) },
                )?;
                if matches!(node.kind, NodeKind::CenteredRebase { .. }) {
                    let RuntimeValue::Matrix(input) = self.materialize(values, node.args[0])?
                    else {
                        return Err(ExecutionError::ValueKind(node.args[0]));
                    };
                    if let Some(full) = input.cpu_full_arc() {
                        let output = self
                            .backend
                            .centered_rebase(&full, &ty)
                            .map_err(Self::backend_error)?;
                        self.put(values, node.id, 0, RuntimeValue::matrix(output));
                    } else if let Some(compact) = input.cpu_compact_arc() {
                        let output_kind = self
                            .bounded_matrix_schema(
                                scope_id,
                                path,
                                env,
                                WireRef { node: node.id, port: Port(0) },
                            )?
                            .1;
                        let output = self
                            .backend
                            .centered_rebase_small(&compact, &ty)
                            .map_err(Self::backend_error)?;
                        self.put(values, node.id, 0, compact_runtime_value(output, output_kind));
                    } else {
                        return Err(ExecutionError::ValueKind(node.args[0]));
                    }
                } else {
                    let input = self.matrix(values, node.args[0])?;
                    let output = if matches!(node.kind, NodeKind::ModulusSwitch { .. }) {
                        self.backend.modulus_switch(&input, &ty)
                    } else {
                        self.backend.reduce_modulus(&input, &ty)
                    }
                    .map_err(Self::backend_error)?;
                    self.put(values, node.id, 0, RuntimeValue::matrix(output));
                }
            }
            NodeKind::CenteredRoundDivide { divisor } => {
                let input = self.matrix(values, node.args[0])?;
                let divisor = divisor
                    .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| self.expression_error(node.id, error))?;
                let output = self
                    .backend
                    .centered_round_divide(&input, &divisor)
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::RnsModUp { digit_size, normalize, .. } => {
                let input = self.matrix(values, node.args[0])?;
                let source_moduli =
                    self.matrix_type(scope_id, path, env, node.args[0])?.ring.crt_moduli().to_vec();
                let ty = self.matrix_type(
                    scope_id,
                    path,
                    env,
                    WireRef { node: node.id, port: Port(0) },
                )?;
                let output = self
                    .backend
                    .rns_mod_up(&input, &ty, &source_moduli, *digit_size, *normalize)
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::RnsModDown { plaintext_modulus, .. } => {
                let input = self.matrix(values, node.args[0])?;
                let source_moduli =
                    self.matrix_type(scope_id, path, env, node.args[0])?.ring.crt_moduli().to_vec();
                let ty = self.matrix_type(
                    scope_id,
                    path,
                    env,
                    WireRef { node: node.id, port: Port(0) },
                )?;
                let plaintext_modulus = plaintext_modulus
                    .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| self.expression_error(node.id, error))?
                    .to_u64()
                    .ok_or_else(|| {
                        self.expression_error(node.id, "plaintext modulus does not fit u64")
                    })?;
                let output = self
                    .backend
                    .rns_mod_down(&input, &ty, &source_moduli, plaintext_modulus)
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::BlockModSwitch { plaintext_modulus, .. } => {
                let input = self.matrix(values, node.args[0])?;
                let source_moduli =
                    self.matrix_type(scope_id, path, env, node.args[0])?.ring.crt_moduli().to_vec();
                let ty = self.matrix_type(
                    scope_id,
                    path,
                    env,
                    WireRef { node: node.id, port: Port(0) },
                )?;
                let plaintext_modulus = plaintext_modulus
                    .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| self.expression_error(node.id, error))?;
                let output = self
                    .backend
                    .block_mod_switch(&input, &ty, &source_moduli, &plaintext_modulus)
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::RingAutomorphism { index } => {
                let input = self.matrix(values, node.args[0])?;
                let index = self.eval_usize(node.id, index, env)?;
                let output =
                    self.backend.ring_automorphism(&input, index).map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::IntMatrixVectorProduct { transpose } => {
                use rayon::prelude::*;
                let members = |executor: &mut Self, wire| {
                    let count = executor.family_count(values, wire)?;
                    (0..count)
                        .map(|index| {
                            match executor.family_member(values, wire, index, node.id)? {
                                RuntimeValue::Int(value) => Ok(value),
                                _ => Err(ExecutionError::ValueKind(wire)),
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()
                };
                let matrix = members(self, node.args[0])?;
                let vector = members(self, node.args[1])?;
                let inner = vector.len();
                let outer = matrix.len() / inner;
                let output = (0..outer)
                    .into_par_iter()
                    .map(|position| {
                        (0..inner)
                            .map(|term| {
                                let entry = if *transpose {
                                    term * outer + position
                                } else {
                                    position * inner + term
                                };
                                &matrix[entry] * &vector[term]
                            })
                            .sum::<BigInt>()
                    })
                    .collect();
                self.put(values, node.id, 0, RuntimeValue::integer_values(output));
            }
            NodeKind::MultiplyMonomial => {
                let input = self.matrix(values, node.args[0])?;
                let exponent = self.int(values, node.args[1])?;
                let period = BigInt::from(2 * input.params().ring_dimension() as u64);
                let exponent = mxx_ir_core::expr::euclidean_div_rem(&exponent, &period)
                    .map_err(|error| self.expression_error(node.id, error))?
                    .1
                    .to_usize()
                    .expect("residue below 2n fits usize");
                let output = self
                    .backend
                    .multiply_monomial(&input, exponent)
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::ExtractCoefficient { position, .. } => {
                let input = self.matrix(values, node.args[0])?;
                let position = self.eval_usize(node.id, position, env)?;
                let output = crate::host_control::dispatch_extract_coefficient(
                    self.backend,
                    &input,
                    position,
                )
                .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::Int(output));
            }
            NodeKind::LiftIntegerToConstantPolynomial { .. } => {
                let coefficient = self.int(values, node.args[0])?;
                let ty = self.matrix_type(
                    scope_id,
                    path,
                    env,
                    WireRef { node: node.id, port: Port(0) },
                )?;
                let identity = self
                    .backend
                    .constant_matrix(&ty, &mxx_ir_core::node::ConstantMatrix::Identity, env)
                    .map_err(Self::backend_error)?;
                let output = self
                    .backend
                    .scale_integer(&identity, &coefficient)
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool } => {
                let input = self.matrix(values, node.args[0])?;
                let plaintext = plaintext_modulus
                    .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| self.expression_error(node.id, error))?;
                let length = self.eval_usize(node.id, length, env)?;
                let decoded = crate::host_control::dispatch_threshold_decode(
                    self.backend,
                    &input,
                    &plaintext,
                    length,
                )
                .map_err(Self::backend_error)?;
                for (port, value) in decoded.into_iter().enumerate() {
                    let value = if *output_bool {
                        RuntimeValue::Bool(!value.is_zero())
                    } else {
                        RuntimeValue::Int(value)
                    };
                    self.put(values, node.id, port as u32, value);
                }
            }
            NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients, .. } => {
                let levels = node
                    .args
                    .iter()
                    .map(|wire| self.matrix(values, *wire).map(|value| value.as_ref().clone()))
                    .collect::<Result<Vec<_>, _>>()?;
                let plaintext_moduli = plaintext_moduli
                    .iter()
                    .map(|value| {
                        value
                            .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                            .map_err(|error| self.expression_error(node.id, error))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let reconstruction_coefficients = reconstruction_coefficients
                    .iter()
                    .map(|value| {
                        value
                            .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                            .map_err(|error| self.expression_error(node.id, error))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let destination = self.matrix_type(
                    scope_id,
                    path,
                    env,
                    WireRef { node: node.id, port: Port(0) },
                )?;
                let output = self
                    .backend
                    .crt_recompose(
                        &levels,
                        &plaintext_moduli,
                        &reconstruction_coefficients,
                        &destination,
                    )
                    .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::PackPolynomialCoefficients { coefficient_bits, .. } => {
                let coefficient_bits = self.eval_usize(node.id, coefficient_bits, env)?;
                let count = self.family_count(values, node.args[0])?;
                let mut bits = Vec::with_capacity(count);
                for index in 0..count {
                    let member = self.family_member(values, node.args[0], index, node.id)?;
                    let RuntimeValue::Bool(bit) = member else {
                        return Err(ExecutionError::ValueKind(node.args[0]));
                    };
                    bits.push(bit);
                }
                let ty = self.matrix_type(
                    scope_id,
                    path,
                    env,
                    WireRef { node: node.id, port: Port(0) },
                )?;
                let output = crate::host_control::dispatch_pack_polynomial_coefficients(
                    self.backend,
                    &ty,
                    &bits,
                    coefficient_bits,
                )
                .map_err(Self::backend_error)?;
                self.put(values, node.id, 0, RuntimeValue::matrix(output));
            }
            NodeKind::SubgraphCall(call) => {
                let child_id =
                    self.validated.source.child_scope_id(scope_id, node.id).ok_or_else(|| {
                        ExecutionError::MissingSubgraph {
                            node: node.id,
                            name: call.definition.clone(),
                        }
                    })?;
                let child = self.validated.source.scope(&child_id).ok_or_else(|| {
                    ExecutionError::MissingSubgraph { node: node.id, name: call.definition.clone() }
                })?;
                let child_inputs = self.child_inputs(child, node, values)?;
                let mut child_path = path.to_vec();
                child_path.push(InstantiationFrame { call: node.id, loop_index: None });
                let placement = self.backend.active_placement();
                let mut outputs = None;
                let mut callback_error = None;
                crate::host_control::dispatch_host_control(node.id, node.kind, env, |invocation| {
                    match self.execute_instance(
                        &child_id,
                        &invocation.environment,
                        child_path.clone(),
                        child_inputs.clone(),
                        placement,
                    ) {
                        Ok(instance) => outputs = Some(instance.outputs),
                        Err(error) => callback_error = Some(error),
                    }
                    Ok::<(), ExecutionError>(())
                })
                .map_err(|error| ExecutionError::Expression {
                    node: node.id,
                    message: error.to_string(),
                })?;
                if let Some(error) = callback_error {
                    return Err(error);
                }
                let outputs = outputs.expect("subgraph dispatch invokes exactly once");
                for (port, value) in outputs.into_iter().enumerate() {
                    self.put(values, node.id, port as u32, value);
                }
            }
            NodeKind::ParallelLoop(loop_node) => {
                let child_id =
                    self.validated.source.child_scope_id(scope_id, node.id).ok_or_else(|| {
                        ExecutionError::MissingSubgraph {
                            node: node.id,
                            name: format!("parallel body at {:?}", node.id),
                        }
                    })?;
                let child = self.validated.source.scope(&child_id).ok_or_else(|| {
                    ExecutionError::MissingSubgraph {
                        node: node.id,
                        name: format!("parallel body at {:?}", node.id),
                    }
                })?;
                let count = self.eval_usize(node.id, &loop_node.count, env)?;
                let staged = (0..child.outputs().len())
                    .map(|port| {
                        self.staged_family_descriptor(scope_id, path, node.id, port as u32, count)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut families =
                    staged
                        .iter()
                        .map(|descriptor| {
                            if descriptor.is_some() {
                                Vec::new()
                            } else {
                                Vec::with_capacity(count)
                            }
                        })
                        .collect::<Vec<_>>();
                let placement_count = self.backend.placement_count();
                if placement_count == 0 {
                    return Err(ExecutionError::BackendPlacement { placement: 0, count: 0 });
                }
                let parent_placement = self.backend.active_placement();
                let loop_result = (|| {
                    let mut broadcast_inputs = (0..placement_count)
                        .map(|_| (0..node.args.len()).map(|_| None).collect::<Vec<_>>())
                        .collect::<Vec<_>>();
                    for (argument, (wire, mode)) in
                        node.args.iter().zip(&loop_node.input_modes).enumerate()
                    {
                        if matches!(mode, LoopInputMode::Broadcast) {
                            let broadcast = self.materialize(values, *wire)?;
                            let placed = self.values_for_placements(broadcast)?;
                            for (placement, value) in placed.into_iter().enumerate() {
                                broadcast_inputs[placement][argument] = Some(value);
                            }
                        }
                    }
                    self.set_placement(parent_placement)?;
                    let wave_size = self.config.max_parallel_instances.get();
                    for wave_start in (0..count).step_by(wave_size) {
                        let wave_end = count.min(wave_start.saturating_add(wave_size));
                        let wave_len = wave_end - wave_start;
                        let mut child_envs = Vec::with_capacity(wave_len);
                        let mut child_paths = Vec::with_capacity(wave_len);
                        let mut child_inputs = Vec::with_capacity(wave_len);
                        let mut child_placements = Vec::with_capacity(wave_len);
                        for index in wave_start..wave_end {
                            let placement = index % placement_count;
                            child_envs.push(
                                crate::host_control::bind_host_control_environment(
                                    node.id,
                                    env,
                                    &loop_node.bindings,
                                    Some((loop_node.index_slot, index)),
                                )
                                .map_err(|error| {
                                    ExecutionError::Expression {
                                        node: node.id,
                                        message: error.to_string(),
                                    }
                                })?,
                            );
                            let mut child_path = path.to_vec();
                            child_path.push(InstantiationFrame {
                                call: node.id,
                                loop_index: Some(index as u64),
                            });
                            child_paths.push(child_path);
                            child_placements.push(placement);
                            child_inputs.push(self.loop_child_inputs(
                                child,
                                node,
                                &loop_node.input_modes,
                                index,
                                placement,
                                &broadcast_inputs[placement],
                                values,
                            )?);
                        }
                        let instances = self.execute_instances_batch(
                            &child_id,
                            &child_envs,
                            &child_paths,
                            &child_inputs,
                            &child_placements,
                        )?;
                        // Match the owned-call lifetime: release child inputs
                        // before serializing this wave's returned outputs.
                        drop(child_inputs);
                        for (offset, instance) in instances.into_iter().enumerate() {
                            self.set_placement(child_placements[offset])?;
                            for (port, value) in instance.outputs.into_iter().enumerate() {
                                if let Some((name, descriptor)) = &staged[port] {
                                    let payload =
                                        self.encode_artifact(&value, &descriptor.artifact_type)?;
                                    self.artifact_store
                                        .store(
                                            ArtifactKey {
                                                production: self.scratch_production.as_ref().expect("staged descriptor initializes scratch identity").clone(),
                                                name: name.clone(),
                                                index: Some(wave_start + offset),
                                            },
                                            &descriptor.artifact_type,
                                            descriptor.availability,
                                            descriptor.layout.as_deref(),
                                            payload,
                                        )
                                        .map_err(Self::artifact_error)?;
                                } else {
                                    families[port].push(value);
                                }
                            }
                        }
                    }
                    for (port, family) in families.into_iter().enumerate() {
                        let value = match &staged[port] {
                            Some((name, descriptor)) => RuntimeValue::StagedArtifactFamily {
                                production: self
                                    .scratch_production
                                    .as_ref()
                                    .expect("staged descriptor initializes scratch identity")
                                    .clone(),
                                name: name.clone(),
                                descriptor: descriptor.clone(),
                            },
                            None => self.indexed_family_output(
                                scope_id,
                                env,
                                node.id,
                                port as u32,
                                family,
                            )?,
                        };
                        self.put(values, node.id, port as u32, value);
                    }
                    Ok::<(), ExecutionError>(())
                })();
                let restore_result = self.set_placement(parent_placement);
                loop_result?;
                restore_result?;
            }
            NodeKind::SequentialLoop(loop_node) => {
                let child_id =
                    self.validated.source.child_scope_id(scope_id, node.id).ok_or_else(|| {
                        ExecutionError::MissingSubgraph {
                            node: node.id,
                            name: format!("sequential body at {:?}", node.id),
                        }
                    })?;
                let child = self.validated.source.scope(&child_id).ok_or_else(|| {
                    ExecutionError::MissingSubgraph {
                        node: node.id,
                        name: format!("sequential body at {:?}", node.id),
                    }
                })?;
                let count = self.eval_usize(node.id, &loop_node.count, env)?;
                let parent_placement = self.backend.active_placement();
                let mut carried = node.args[..loop_node.carried_count]
                    .iter()
                    .map(|wire| self.value(values, *wire))
                    .collect::<Result<Vec<_>, _>>()?;
                let invariants = node.args[loop_node.carried_count..]
                    .iter()
                    .map(|wire| self.value(values, *wire))
                    .collect::<Result<Vec<_>, _>>()?;
                let input_names = child
                    .inputs()
                    .iter()
                    .map(|wire| {
                        let input = child.node(wire.node).expect("validated sequential input node");
                        let NodeKind::Input { name, .. } = input.kind() else {
                            unreachable!("validated sequential input must reference an input node")
                        };
                        name.clone()
                    })
                    .collect::<Vec<_>>();
                for index in 0..count {
                    self.set_placement(parent_placement)?;
                    let child_env = crate::host_control::bind_host_control_environment(
                        node.id,
                        env,
                        &loop_node.bindings,
                        Some((loop_node.index_slot, index)),
                    )
                    .map_err(|error| ExecutionError::Expression {
                        node: node.id,
                        message: error.to_string(),
                    })?;
                    let child_inputs = input_names
                        .iter()
                        .cloned()
                        .zip(carried.iter().chain(&invariants).cloned())
                        .collect::<BTreeMap<_, _>>();
                    let mut child_path = path.to_vec();
                    child_path
                        .push(InstantiationFrame { call: node.id, loop_index: Some(index as u64) });
                    carried = self
                        .execute_instance(
                            &child_id,
                            &child_env,
                            child_path,
                            child_inputs,
                            parent_placement,
                        )?
                        .outputs;
                }
                self.set_placement(parent_placement)?;
                for (port, value) in carried.into_iter().enumerate() {
                    self.put(values, node.id, port as u32, value);
                }
            }
            NodeKind::FamilyPack { count } => {
                let count = self.eval_usize(node.id, count, env)?;
                if count == 0 || node.args.len() != count {
                    return Err(ExecutionError::Expression {
                        node: node.id,
                        message: "family pack argument count mismatch".to_owned(),
                    });
                }
                // Imported members are loaded here: a packed family holds
                // values of its element type, not artifact references.
                let members = node
                    .args
                    .iter()
                    .map(|wire| self.materialize(values, *wire))
                    .collect::<Result<Vec<_>, _>>()?;
                let family = self.indexed_family_output(scope_id, env, node.id, 0, members)?;
                self.put(values, node.id, 0, family);
            }
            NodeKind::FamilyGetStatic { index } => {
                let index = self.eval_usize(node.id, index, env)?;
                let selected = self.family_member(values, node.args[0], index, node.id)?;
                self.put(values, node.id, 0, selected);
            }
            NodeKind::FamilyGetDynamic => {
                let index = self.int(values, node.args[1])?;
                let Some(index) = index.to_usize() else {
                    let count = self.family_count(values, node.args[0])?;
                    return Err(ExecutionError::SelectIndexOutOfRange {
                        node: node.id,
                        index,
                        count,
                    });
                };
                let selected = self.family_member(values, node.args[0], index, node.id)?;
                self.put(values, node.id, 0, selected);
            }
            NodeKind::Select { count } => {
                let count = self.eval_usize(node.id, count, env)?;
                let index = self.int(values, node.args[0])?;
                let Some(index_usize) = index.to_usize().filter(|index| *index < count) else {
                    return Err(ExecutionError::SelectIndexOutOfRange {
                        node: node.id,
                        index,
                        count,
                    });
                };
                let selected_wire = node.args[index_usize + 1];
                let selected = self.materialize(values, selected_wire)?;
                self.put(values, node.id, 0, selected);
            }
        }
        Ok(())
    }

    pub(super) fn family_count(
        &self,
        values: &BTreeMap<WireRef, RuntimeValue>,
        wire: WireRef,
    ) -> Result<usize, ExecutionError> {
        match values.get(&wire).ok_or(ExecutionError::MissingWire(wire))? {
            RuntimeValue::LazyArtifactFamily { descriptor, .. } |
            RuntimeValue::StagedArtifactFamily { descriptor, .. } => {
                descriptor.family_count.ok_or(ExecutionError::ValueKind(wire))
            }
            RuntimeValue::IndexedFamily { values, .. } => Ok(values.len()),
            _ => Err(ExecutionError::ValueKind(wire)),
        }
    }

    fn family_member(
        &mut self,
        values: &BTreeMap<WireRef, RuntimeValue>,
        wire: WireRef,
        index: usize,
        node: NodeId,
    ) -> Result<RuntimeValue, ExecutionError> {
        let count = self.family_count(values, wire)?;
        if index >= count {
            return Err(ExecutionError::SelectIndexOutOfRange {
                node,
                index: BigInt::from(index),
                count,
            });
        }
        let member = self.family_member_value(values, wire, index)?;
        self.materialize_value(member)
    }

    pub(super) fn family_member_value(
        &self,
        values: &BTreeMap<WireRef, RuntimeValue>,
        wire: WireRef,
        index: usize,
    ) -> Result<RuntimeValue, ExecutionError> {
        match values.get(&wire).ok_or(ExecutionError::MissingWire(wire))? {
            RuntimeValue::LazyArtifactFamily { production, name, descriptor }
                if descriptor.family_count.is_some_and(|count| index < count) =>
            {
                Ok(RuntimeValue::LazyArtifact {
                    production: production.clone(),
                    name: name.clone(),
                    index: Some(index),
                    descriptor: descriptor.clone(),
                })
            }
            RuntimeValue::StagedArtifactFamily { production, name, descriptor }
                if descriptor.family_count.is_some_and(|count| index < count) =>
            {
                Ok(RuntimeValue::StagedArtifact {
                    production: production.clone(),
                    name: name.clone(),
                    index,
                    descriptor: descriptor.clone(),
                })
            }
            RuntimeValue::IndexedFamily { values, .. } => {
                values.get(index).cloned().ok_or(ExecutionError::ValueKind(wire))
            }
            _ => Err(ExecutionError::ValueKind(wire)),
        }
    }

    fn preimage_source(
        &mut self,
        values: &mut BTreeMap<WireRef, RuntimeValue>,
        path: &[InstantiationFrame],
        wire: WireRef,
        matrix_type: &ConcreteMatrixType,
    ) -> Result<Arc<dyn crate::matrix::PolyMatrixColumnSource<DCRTPolyMatrix>>, ExecutionError>
    {
        let value = values.remove(&wire).ok_or(ExecutionError::MissingWire(wire))?;
        let RuntimeValue::Matrix(matrix) = self.materialize_value(value)? else {
            return Err(ExecutionError::ValueKind(wire));
        };
        if matrix.matrix_type() != matrix_type {
            return Err(ExecutionError::ValueKind(wire));
        }
        let native = matrix.cpu_full_arc().ok_or(ExecutionError::ValueKind(wire))?;
        let source = self.backend.preimage_target(native).map_err(Self::backend_error)?.0;
        let retained = RuntimeValue::Matrix(matrix);
        if let Some(trace) = &mut self.trace {
            trace.insert(WireId { instantiation_path: path.to_vec(), wire }, retained.clone());
        }
        values.insert(wire, retained);
        Ok(source)
    }

    fn sample_matrix<F>(
        &mut self,
        path: &[InstantiationFrame],
        wire: WireRef,
        ty: &ConcreteMatrixType,
        fresh: F,
    ) -> Result<DCRTPolyMatrix, ExecutionError>
    where
        F: FnOnce(&mut CpuDcrtBackend) -> Result<DCRTPolyMatrix, PolyBackendError>,
    {
        self.sample_matrix_with_status(path, wire, ty, fresh).map(|(value, _)| value)
    }

    fn sample_matrix_with_status<F>(
        &mut self,
        path: &[InstantiationFrame],
        wire: WireRef,
        ty: &ConcreteMatrixType,
        fresh: F,
    ) -> Result<(DCRTPolyMatrix, bool), ExecutionError>
    where
        F: FnOnce(&mut CpuDcrtBackend) -> Result<DCRTPolyMatrix, PolyBackendError>,
    {
        let site = DrawSite { instantiation_path: path.to_vec(), node: wire.node, port: wire.port };
        if let Some(production) = self.session.clone() {
            if let Some(recorded) = self
                .artifact_store
                .transcript_entry(&production, &site)
                .map_err(Self::artifact_error)?
            {
                return match recorded {
                    RecordedValue::Matrix { matrix_type, bytes } if matrix_type == *ty => self
                        .backend
                        .matrix_from_bytes(ty, &bytes)
                        .map(|value| (value, false))
                        .map_err(Self::backend_error),
                    RecordedValue::Matrix { .. } |
                    RecordedValue::SmallMatrix { .. } |
                    RecordedValue::Trapdoor { .. } => {
                        Err(TranscriptError::KindMismatch(site).into())
                    }
                };
            }
            let value = fresh(self.backend).map_err(Self::backend_error)?;
            self.artifact_store
                .record_transcript_batch(
                    &production,
                    &[(
                        site,
                        RecordedValue::Matrix {
                            matrix_type: ty.clone(),
                            bytes: self.backend.matrix_to_bytes(&value),
                        },
                    )],
                )
                .map_err(Self::artifact_error)?;
            return Ok((value, true));
        }
        match &mut self.sampling_mode {
            SamplingMode::Fresh => {
                fresh(self.backend).map(|value| (value, true)).map_err(Self::backend_error)
            }
            SamplingMode::Record(recorder) => {
                let value = fresh(self.backend).map_err(Self::backend_error)?;
                recorder.record(
                    site,
                    RecordedValue::Matrix {
                        matrix_type: ty.clone(),
                        bytes: self.backend.matrix_to_bytes(&value),
                    },
                )?;
                Ok((value, true))
            }
            SamplingMode::Replay(replayer) => match replayer.get(&site)? {
                RecordedValue::Matrix { bytes, .. } => self
                    .backend
                    .matrix_from_bytes(ty, bytes)
                    .map(|value| (value, false))
                    .map_err(Self::backend_error),
                RecordedValue::SmallMatrix { .. } | RecordedValue::Trapdoor { .. } => {
                    Err(TranscriptError::KindMismatch(site).into())
                }
            },
        }
    }

    fn sample_small_matrix<F>(
        &mut self,
        path: &[InstantiationFrame],
        wire: WireRef,
        schema: &ConcreteBoundedMatrixSchema,
        semantic_kind: SmallMatrixSemanticKind,
        fresh: F,
    ) -> Result<CpuSmallMatrix<DCRTPolyMatrix>, ExecutionError>
    where
        F: FnOnce(&mut CpuDcrtBackend) -> Result<CpuSmallMatrix<DCRTPolyMatrix>, PolyBackendError>,
    {
        self.sample_small_matrix_with_status(path, wire, schema, semantic_kind, fresh)
            .map(|(value, _)| value)
    }

    fn sample_small_matrix_with_status<F>(
        &mut self,
        path: &[InstantiationFrame],
        wire: WireRef,
        schema: &ConcreteBoundedMatrixSchema,
        semantic_kind: SmallMatrixSemanticKind,
        fresh: F,
    ) -> Result<(CpuSmallMatrix<DCRTPolyMatrix>, bool), ExecutionError>
    where
        F: FnOnce(&mut CpuDcrtBackend) -> Result<CpuSmallMatrix<DCRTPolyMatrix>, PolyBackendError>,
    {
        let site = DrawSite { instantiation_path: path.to_vec(), node: wire.node, port: wire.port };
        if let Some(production) = self.session.clone() {
            if let Some(recorded) = self
                .artifact_store
                .transcript_entry(&production, &site)
                .map_err(Self::artifact_error)?
            {
                return match recorded {
                    RecordedValue::SmallMatrix {
                        schema: recorded_schema,
                        semantic_kind: recorded_kind,
                        bytes,
                    } if recorded_schema == *schema && recorded_kind == semantic_kind => self
                        .backend
                        .small_matrix_from_bytes(schema, &bytes, semantic_kind)
                        .map(|value| (value, false))
                        .map_err(Self::backend_error),
                    RecordedValue::Matrix { .. } |
                    RecordedValue::SmallMatrix { .. } |
                    RecordedValue::Trapdoor { .. } => {
                        Err(TranscriptError::KindMismatch(site).into())
                    }
                };
            }
            let value = fresh(self.backend).map_err(Self::backend_error)?;
            let bytes = self
                .backend
                .small_matrix_to_bytes(&value, schema, semantic_kind)
                .map_err(Self::backend_error)?;
            self.artifact_store
                .record_transcript_batch(
                    &production,
                    &[(
                        (site),
                        RecordedValue::SmallMatrix { schema: schema.clone(), semantic_kind, bytes },
                    )],
                )
                .map_err(Self::artifact_error)?;
            return Ok((value, true));
        }
        match &mut self.sampling_mode {
            SamplingMode::Fresh => {
                fresh(self.backend).map(|value| (value, true)).map_err(Self::backend_error)
            }
            SamplingMode::Record(recorder) => {
                let value = fresh(self.backend).map_err(Self::backend_error)?;
                let bytes = self
                    .backend
                    .small_matrix_to_bytes(&value, schema, semantic_kind)
                    .map_err(Self::backend_error)?;
                recorder.record(
                    site,
                    RecordedValue::SmallMatrix { schema: schema.clone(), semantic_kind, bytes },
                )?;
                Ok((value, true))
            }
            SamplingMode::Replay(replayer) => match replayer.get(&site)? {
                RecordedValue::SmallMatrix {
                    schema: recorded_schema,
                    semantic_kind: recorded_kind,
                    bytes,
                } if recorded_schema == schema && *recorded_kind == semantic_kind => self
                    .backend
                    .small_matrix_from_bytes(schema, bytes, semantic_kind)
                    .map(|value| (value, false))
                    .map_err(Self::backend_error),
                RecordedValue::Matrix { .. } |
                RecordedValue::SmallMatrix { .. } |
                RecordedValue::Trapdoor { .. } => Err(TranscriptError::KindMismatch(site).into()),
            },
        }
    }

    fn record_preimages(&mut self, count: usize) {
        if let Some(progress) = &mut self.preimage_progress {
            progress.record(count);
        }
    }

    pub(super) fn finish_preimage_progress(&self) -> Result<(), ExecutionError> {
        self.preimage_progress.as_ref().map_or(Ok(()), PreimageProgress::finish)
    }

    fn sample_trapdoor(
        &mut self,
        path: &[InstantiationFrame],
        matrix_wire: WireRef,
        trapdoor_wire: WireRef,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
    ) -> Result<(DCRTPolyMatrix, DCRTTrapdoor), ExecutionError> {
        let matrix_site = DrawSite {
            instantiation_path: path.to_vec(),
            node: matrix_wire.node,
            port: matrix_wire.port,
        };
        let trapdoor_site = DrawSite {
            instantiation_path: path.to_vec(),
            node: trapdoor_wire.node,
            port: trapdoor_wire.port,
        };
        if let Some(production) = self.session.clone() {
            let recorded_public = self
                .artifact_store
                .transcript_entry(&production, &matrix_site)
                .map_err(Self::artifact_error)?;
            let recorded_trapdoor = self
                .artifact_store
                .transcript_entry(&production, &trapdoor_site)
                .map_err(Self::artifact_error)?;
            return match (recorded_public, recorded_trapdoor) {
                (
                    Some(RecordedValue::Matrix { matrix_type, bytes }),
                    Some(RecordedValue::Trapdoor {
                        matrix_type: secret_type,
                        public_bytes,
                        trapdoor_bytes,
                    }),
                ) if matrix_type == *ty && secret_type == *ty && bytes == public_bytes => {
                    let public =
                        self.backend.matrix_from_bytes(ty, &bytes).map_err(Self::backend_error)?;
                    let secret = self
                        .backend
                        .trapdoor_from_bytes(ty, &trapdoor_bytes)
                        .map_err(Self::backend_error)?;
                    Ok((public, secret))
                }
                (None, None) => {
                    let (public, secret) = self
                        .backend
                        .sample_trapdoor(ty, sigma, gadget_base, digit_count)
                        .map_err(Self::backend_error)?;
                    let public_bytes = self.backend.matrix_to_bytes(&public);
                    self.artifact_store
                        .record_transcript_batch(
                            &production,
                            &[
                                (
                                    matrix_site,
                                    RecordedValue::Matrix {
                                        matrix_type: ty.clone(),
                                        bytes: public_bytes.clone(),
                                    },
                                ),
                                (
                                    trapdoor_site,
                                    RecordedValue::Trapdoor {
                                        matrix_type: ty.clone(),
                                        public_bytes,
                                        trapdoor_bytes: self.backend.trapdoor_to_bytes(&secret),
                                    },
                                ),
                            ],
                        )
                        .map_err(Self::artifact_error)?;
                    Ok((public, secret))
                }
                _ => Err(ExecutionError::Manifest(
                    "session contains an incomplete or inconsistent trapdoor draw".to_owned(),
                )),
            };
        }
        match &mut self.sampling_mode {
            SamplingMode::Fresh => self
                .backend
                .sample_trapdoor(ty, sigma, gadget_base, digit_count)
                .map_err(Self::backend_error),
            SamplingMode::Record(recorder) => {
                let (public, secret) = self
                    .backend
                    .sample_trapdoor(ty, sigma, gadget_base, digit_count)
                    .map_err(Self::backend_error)?;
                let public_bytes = self.backend.matrix_to_bytes(&public);
                recorder.record(
                    matrix_site,
                    RecordedValue::Matrix { matrix_type: ty.clone(), bytes: public_bytes.clone() },
                )?;
                recorder.record(
                    trapdoor_site,
                    RecordedValue::Trapdoor {
                        matrix_type: ty.clone(),
                        public_bytes,
                        trapdoor_bytes: self.backend.trapdoor_to_bytes(&secret),
                    },
                )?;
                Ok((public, secret))
            }
            SamplingMode::Replay(replayer) => {
                let public = match replayer.get(&matrix_site)? {
                    RecordedValue::Matrix { bytes, .. } => {
                        self.backend.matrix_from_bytes(ty, bytes).map_err(Self::backend_error)?
                    }
                    RecordedValue::SmallMatrix { .. } | RecordedValue::Trapdoor { .. } => {
                        return Err(TranscriptError::KindMismatch(matrix_site).into());
                    }
                };
                let secret = match replayer.get(&trapdoor_site)? {
                    RecordedValue::Trapdoor { trapdoor_bytes, .. } => self
                        .backend
                        .trapdoor_from_bytes(ty, trapdoor_bytes)
                        .map_err(Self::backend_error)?,
                    RecordedValue::Matrix { .. } | RecordedValue::SmallMatrix { .. } => {
                        return Err(TranscriptError::KindMismatch(trapdoor_site).into());
                    }
                };
                Ok((public, secret))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::MemoryArtifactStore,
        backend::poly::cpu_backend,
        matrix::{PolyMatrix as PrimitivePolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_dsl::{DslContext, Family, Int, Mat, Ring, parallel};
    use mxx_ir_core::{
        IntExpr,
        artifact::ArtifactAvailability,
        node::{ConcatAxis, IndexRange},
        types::CoefficientBoundDomain,
    };
    use num_bigint::BigInt;

    fn ring(parameters: &DCRTPolyParams) -> Ring {
        Ring::from_crt_moduli(
            parameters.to_crt().0.into_iter().map(IntExpr::from).collect(),
            parameters.ring_dimension(),
        )
    }

    fn matrix_output<'a>(result: &'a ExecutionResult, name: &str) -> &'a DCRTPolyMatrix {
        let RuntimeValue::Matrix(value) = &result.outputs[name] else { panic!("matrix output") };
        value.as_cpu_full().expect("CPU matrix output")
    }

    #[test]
    fn gadget_trapdoor_preimage_round_trips_with_declared_bound() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
        let ring = ring(&parameters);
        let base = BigInt::from(1u8) << parameters.base_bits();
        let digits = parameters.modulus_digits();
        let trapdoor = ring.gadget_trapdoor(1, base.clone(), digits);
        let preimage = trapdoor.sample_preimage(ring.zero((1, 2)), (digits, 2));
        let graph = DslContext::new("cpu-gadget-preimage")
            .transferred_output("transferred", preimage.clone())
            .unwrap()
            .cached_output("cached", preimage)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let mut result = execute(
            &graph,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        let expected = backend
            .gadget_decompose(&DCRTPolyMatrix::zero(&parameters, 1, 2), false, Some(digits))
            .unwrap();
        for name in ["transferred", "cached"] {
            let RuntimeValue::Matrix(value) =
                result.materialize_output(name, &backend, &mut store).unwrap()
            else {
                panic!("preimage output")
            };
            let compact = value.as_cpu_compact().expect("CPU preimage");
            assert_eq!(compact, &expected);
            assert_eq!(compact.bound_domain(), CoefficientBoundDomain::Global);
            assert_eq!(
                compact.max_coefficient_bound(),
                &((&base + 1u8) / 2u8).to_biguint().unwrap()
            );
        }
    }

    #[test]
    fn parallel_preimages_use_one_cpu_batch() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
        let ring = ring(&parameters);
        let digits = parameters.modulus_digits();
        let base = BigInt::from(1u8) << parameters.base_bits();
        let samples = parallel(2, |_| {
            let trapdoor = ring.sample_trapdoor(1, 5, base.clone(), digits, 1_000_000);
            Ok(trapdoor.sample_preimage(ring.zero((1, 1)), (digits + 2, 1)))
        })
        .unwrap();
        let graph = DslContext::new("cpu-parallel-preimages")
            .output("samples", samples)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let mut result = execute(
            &graph,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig {
                max_parallel_instances: NonZeroUsize::new(2).unwrap(),
                ..ExecutionConfig::default()
            },
        )
        .unwrap();
        assert_eq!(backend.preimage_batch_calls(), 1);
        let RuntimeValue::IndexedFamily { values, .. } =
            result.materialize_output("samples", &backend, &mut store).unwrap()
        else {
            panic!("preimage samples must be an indexed family")
        };
        assert_eq!(values.len(), 2);
        assert!(values.iter().all(|value| matches!(
            value,
            RuntimeValue::Matrix(matrix) if matrix.as_cpu_compact().is_some()
        )));
    }

    #[test]
    fn sparse_row_sum_alias_matches_trace() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
        let ring = ring(&parameters);
        let source = Mat::concat(
            ConcatAxis::Rows,
            vec![ring.input("first", (1, 1)), ring.input("rest", (3, 1))],
        );
        let row = |index: usize| {
            source
                .clone()
                .slice(Some(IndexRange { start: index.into(), end: (index + 1).into() }), None)
        };
        let sum = Mat::concat(ConcatAxis::Rows, vec![row(0), row(1) + row(2), row(3)]);
        let graph = DslContext::new("cpu-sparse-row-sum")
            .output("sum", sum)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let matrices = [1u8, 3, 5, 7].map(|value| {
            DCRTPolyMatrix::from_poly_vec_row(
                &parameters,
                vec![DCRTPoly::from_biguints(&parameters, &[value.into()])],
            )
        });
        let inputs = BTreeMap::from([
            ("first".to_owned(), RuntimeValue::matrix(matrices[0].clone())),
            (
                "rest".to_owned(),
                RuntimeValue::matrix(matrices[1].concat_rows(&[&matrices[2], &matrices[3]])),
            ),
        ]);
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let optimized = execute(
            &graph,
            &mut backend,
            inputs.clone(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        let (reference, _) = execute_with_trace(
            &graph,
            &mut backend,
            inputs,
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        assert_eq!(matrix_output(&optimized, "sum"), matrix_output(&reference, "sum"));
        let middle = &matrices[1] + &matrices[2];
        let expected = matrices[0].concat_rows(&[&middle, &matrices[3]]);
        assert_eq!(matrix_output(&optimized, "sum"), &expected);
    }
    #[test]
    fn root_block_aliases_preserve_multiple_consumers_and_trace_results() {
        use mxx_dsl::Mat;
        use mxx_ir_core::node::{ConcatAxis, IndexRange};

        let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
        let ring = ring(&parameters);
        let first = ring.input("first", (1, 1));
        let second = ring.input("second", (2, 1));
        let joined = Mat::concat(ConcatAxis::Rows, vec![first, second]);
        let top = joined.clone().slice(Some(IndexRange { start: 0.into(), end: 1.into() }), None);
        let bottom = joined.slice(Some(IndexRange { start: 1.into(), end: 3.into() }), None);
        let graph = DslContext::new("root-block-aliases")
            .output("top", top.clone())
            .unwrap()
            .output("twice", top.clone() + top)
            .unwrap()
            .output("bottom", bottom)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let plan = root_block_aliases(&graph, &FrozenGraphScopeId::Root, false);
        assert_eq!(plan.concats.len(), 1);
        assert_eq!(plan.slices.len(), 2);
        let dynamic_scope = FrozenGraphScopeId::ParallelBody {
            parent: Box::new(FrozenGraphScopeId::Root),
            owner: NodeId(0),
        };
        assert!(root_block_aliases(&graph, &dynamic_scope, false).concats.is_empty());
        assert!(root_block_aliases(&graph, &FrozenGraphScopeId::Root, true).concats.is_empty());

        let first_matrix = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            vec![DCRTPoly::from_biguints(&parameters, &[7u8.into()])],
        );
        let second_matrix = first_matrix.concat_rows(&[&first_matrix]);
        let inputs = BTreeMap::from([
            ("first".to_owned(), RuntimeValue::matrix(first_matrix.clone())),
            ("second".to_owned(), RuntimeValue::matrix(second_matrix.clone())),
        ]);
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let optimized = execute(
            &graph,
            &mut backend,
            inputs.clone(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        let (reference, _) = execute_with_trace(
            &graph,
            &mut backend,
            inputs,
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        for name in ["top", "twice", "bottom"] {
            assert_eq!(matrix_output(&optimized, name), matrix_output(&reference, name));
        }
        assert_eq!(matrix_output(&optimized, "top"), &first_matrix);
        assert_eq!(matrix_output(&optimized, "bottom"), &second_matrix);
    }

    #[test]

    fn shared_tensor_row_sum_group_executes_tensor_once_and_matches_trace() {
        use mxx_dsl::Mat;
        use mxx_ir_core::node::{ConcatAxis, IndexRange};
        let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
        let scalar = |value: u8| {
            DCRTPolyMatrix::from_poly_vec_row(
                &parameters,
                vec![DCRTPoly::from_biguints(&parameters, &[value.into()])],
            )
        };
        let left = scalar(2).concat_rows(&[&scalar(3)]);
        let right = scalar(5).concat_rows(&[&scalar(7)]);
        let ring = ring(&parameters);
        let tensor = ring.input("left", (2, 1)).tensor(ring.input("right", (2, 1)));
        let row = |index: usize| {
            tensor
                .clone()
                .slice(Some(IndexRange { start: index.into(), end: (index + 1).into() }), None)
        };
        let leading = Mat::concat(ConcatAxis::Rows, vec![row(0)]);
        let carry = Mat::concat(ConcatAxis::Rows, vec![row(1) + row(2), row(3)]);
        let graph = DslContext::new("shared-tensor-row-sum")
            .output("leading", leading)
            .unwrap()
            .output("carry", carry)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let plan = root_block_aliases(&graph, &FrozenGraphScopeId::Root, false);
        assert_eq!(plan.tensor_row_sum_groups.len(), 1);
        assert_eq!(plan.tensor_row_sum_groups.values().next().unwrap().len(), 2);
        let expected = left.tensor(&right);
        let expected_leading = expected.sum_rows(&[vec![0]]);
        let expected_carry = expected.sum_rows(&[vec![1, 2], vec![3]]);
        let inputs = BTreeMap::from([
            ("left".to_owned(), RuntimeValue::matrix(left)),
            ("right".to_owned(), RuntimeValue::matrix(right)),
        ]);
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let optimized = execute(
            &graph,
            &mut backend,
            inputs.clone(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        let (reference, _) = execute_with_trace(
            &graph,
            &mut backend,
            inputs,
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        assert_eq!(matrix_output(&optimized, "leading"), &expected_leading);
        assert_eq!(matrix_output(&optimized, "carry"), &expected_carry);
        assert_eq!(matrix_output(&optimized, "leading"), matrix_output(&reference, "leading"));
        assert_eq!(matrix_output(&optimized, "carry"), matrix_output(&reference, "carry"));
    }

    #[test]

    fn composed_sparse_rows_preserve_outputs_and_observable_parent_boundaries() {
        use mxx_dsl::Mat;
        use mxx_ir_core::node::{ConcatAxis, IndexRange};
        for mode in 0..3 {
            let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
            let ring = ring(&parameters);
            let source = ring.input("source", (4, 1));
            let row = |matrix: &Mat, index: usize| {
                matrix
                    .clone()
                    .slice(Some(IndexRange { start: index.into(), end: (index + 1).into() }), None)
            };
            let product = Mat::concat(
                ConcatAxis::Rows,
                vec![row(&source, 0), row(&source, 1) + row(&source, 2), row(&source, 3)],
            );
            let leading = row(&product, 0);
            let carry = Mat::concat(ConcatAxis::Rows, vec![row(&product, 1), row(&product, 2)]);
            let mut context = DslContext::new("composed-sparse-rows")
                .output("leading", leading)
                .unwrap()
                .output("carry", carry)
                .unwrap();
            if mode == 1 {
                context = context.output("parent", product.clone()).unwrap();
            }
            if mode == 2 {
                context = context.output("transpose", product.transpose()).unwrap();
            }
            let graph = context
                .build()
                .unwrap()
                .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
                .unwrap();
            let plans = root_row_sum_plans(&graph);
            assert_eq!(plans.len(), 2);
            if mode == 0 {
                assert!(plans.values().any(|plan| plan.rows == vec![vec![0]]));
                assert!(plans.values().any(|plan| plan.rows == vec![vec![1, 2], vec![3]]));
                assert_eq!(
                    plans.values().next().unwrap().source,
                    plans.values().next_back().unwrap().source
                );
            } else {
                assert!(plans.values().any(|plan| plan.rows == vec![vec![0], vec![1, 2], vec![3]]));
            }
            let matrices = [1u8, 3, 5, 7].map(|value| {
                DCRTPolyMatrix::from_poly_vec_row(
                    &parameters,
                    vec![DCRTPoly::from_biguints(&parameters, &[value.into()])],
                )
            });
            let source = matrices[0].concat_rows(&[&matrices[1], &matrices[2], &matrices[3]]);
            let inputs = BTreeMap::from([("source".to_owned(), RuntimeValue::matrix(source))]);
            let mut backend = cpu_backend([parameters]);
            let mut store = MemoryArtifactStore::default();
            let optimized = execute(
                &graph,
                &mut backend,
                inputs.clone(),
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig::default(),
            )
            .unwrap();
            let (reference, _) = execute_with_trace(
                &graph,
                &mut backend,
                inputs,
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig::default(),
            )
            .unwrap();
            for name in optimized.outputs.keys() {
                assert_eq!(matrix_output(&optimized, name), matrix_output(&reference, name));
            }
            let middle = &matrices[1] + &matrices[2];
            assert_eq!(matrix_output(&optimized, "leading"), &matrices[0]);
            assert_eq!(matrix_output(&optimized, "carry"), &middle.concat_rows(&[&matrices[3]]));
        }
    }

    #[test]
    fn producer_artifact_family_loads_only_selected_member() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
        let ring = ring(&parameters);
        let identity = ring.identity(1);
        let members =
            parallel(3, |_| Ok(identity.clone() + ring.identity(1))).expect("artifact family");
        let producer = DslContext::new("cpu-artifact-producer")
            .cached_output("members", members)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let mut store = MemoryArtifactStore::default();
        let produced = execute_in_session(
            &producer,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::new(),
            &mut store,
            [0x71; 32],
            ExecutionConfig::default(),
        )
        .unwrap();
        let production = produced.production_id.expect("producer identity");
        let manifest = store.load_finalized_manifest(&production).unwrap();
        let family = ring.family_artifact_input(
            production.clone(),
            "members",
            3,
            (1, 1),
            ArtifactAvailability::Cached,
        );
        let consumer = DslContext::new("cpu-artifact-consumer")
            .output("selected", family.at(1) + ring.zero((1, 1)))
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production.clone(), manifest)]),
                crate::openfhe_guard::gen_modulus_and_warmup,
            )
            .unwrap();
        let keys = (0..3)
            .map(|index| ArtifactKey {
                production: production.clone(),
                name: "members".to_owned(),
                index: Some(index),
            })
            .collect::<Vec<_>>();
        let loads_before = keys.iter().map(|key| store.load_count(key)).collect::<Vec<_>>();
        let consumed = execute(
            &consumer,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        let expected = DCRTPolyMatrix::identity(&parameters, 1, None)
            .add_out_of_place(&DCRTPolyMatrix::identity(&parameters, 1, None));
        assert_eq!(matrix_output(&consumed, "selected"), &expected);
        assert_eq!(store.load_count(&keys[0]), loads_before[0]);
        assert_eq!(store.load_count(&keys[1]), loads_before[1] + 1);
        assert_eq!(store.load_count(&keys[2]), loads_before[2]);
    }

    #[test]
    fn loop_ring_modulus_uses_each_instance_index() {
        let first_prime = IntExpr::Add(
            Box::new(17.into()),
            Box::new(IntExpr::Mul(Box::new(96.into()), Box::new(IntExpr::LoopIndex(0)))),
        );
        let ring = Ring::from_crt_moduli(vec![first_prime, 97.into()], 8);
        let moduli = parallel(2, |_| Ok(Int::evaluate(ring.modulus()))).unwrap();
        let graph = DslContext::new("cpu-loop-ring-property")
            .output("moduli", moduli)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let mut backend = cpu_backend([]);
        let mut store = MemoryArtifactStore::default();
        let mut result = execute(
            &graph,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        let RuntimeValue::IndexedFamily { values, .. } =
            result.materialize_output("moduli", &backend, &mut store).unwrap()
        else {
            panic!("moduli must be an integer family")
        };
        assert!(matches!(&values[0], RuntimeValue::Int(value) if value == &BigInt::from(1649)));
        assert!(matches!(&values[1], RuntimeValue::Int(value) if value == &BigInt::from(10961)));
    }

    #[test]
    fn parallel_matrix_ops_execute_in_bounded_waves() {
        let parameters = DCRTPolyParams::default();
        let ring = ring(&parameters);
        let families = (0..4)
            .map(|_| Family::pack(vec![ring.identity(1), ring.zero((1, 1))]).unwrap())
            .collect::<Vec<_>>();
        let sums = parallel(2, |index| {
            Ok(families
                .iter()
                .map(|family| family.at(&index))
                .reduce(|left, right| left + right)
                .expect("non-empty matrix batch"))
        })
        .unwrap();
        let graph = DslContext::new("cpu-parallel-matrix-batch")
            .output("first", sums.at(0))
            .unwrap()
            .output("second", sums.at(1))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let result = execute(
            &graph,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
            ExecutionConfig {
                max_parallel_instances: NonZeroUsize::new(2).unwrap(),
                ..ExecutionConfig::default()
            },
        )
        .unwrap();
        let four = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            vec![DCRTPoly::from_usize_to_constant(&parameters, 4)],
        );
        assert_eq!(matrix_output(&result, "first"), &four);
        assert_eq!(matrix_output(&result, "second"), &DCRTPolyMatrix::zero(&parameters, 1, 1));
    }
}
