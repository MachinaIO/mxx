//! Process-local measured probe reuse across runtimes, graphs and loop counts.
//!
//! Only identical probes are reused. Static structural loops are sampled with
//! one/two iterations by the existing estimator; their production counts never
//! scale an unrelated kernel or a changed parallel DAG. No whole-graph identity
//! is included. GPU UUID, driver, context, build and physical mapping namespace
//! the cache. It stores only bounded scalar timing metadata, never GPU owners.
use super::*;
use std::{
    hash::{Hash, Hasher},
    sync::{Mutex, OnceLock},
};

const MAX_PROBES: usize = 65_536;
static PROBES: OnceLock<Mutex<BTreeMap<(u64, u64), f64>>> = OnceLock::new();

/// A measurement batch sees only previously completed batches, so warmups and
/// repeated measurement iterations still run on cache misses. Publish averages
/// only after the entire batch succeeds; failed trials cannot poison the cache.
pub(super) struct ProbeMeasurements {
    namespace: u64,
    cached: BTreeMap<u64, f64>,
    measured: BTreeMap<u64, (f64, usize)>,
    pub hits: usize,
    pub misses: usize,
    pub record: bool,
}
impl ProbeMeasurements {
    pub fn new(devices: &[i32], streamed_executables: bool) -> Result<Self, String> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        streamed_executables.hash(&mut hasher);
        for &device in devices {
            device.hash(&mut hasher);
            format!("{:?}", crate::poly::dcrt::gpu::gpu_device_identity(device)?).hash(&mut hasher);
        }
        Ok(Self::in_namespace(hasher.finish()))
    }
    pub(super) fn in_namespace(namespace: u64) -> Self {
        let cache = PROBES.get_or_init(Default::default).lock().unwrap();
        let cached = cache
            .range((namespace, 0)..=(namespace, u64::MAX))
            .map(|((_, key), seconds)| (*key, *seconds))
            .collect();
        Self { namespace, cached, measured: BTreeMap::new(), hits: 0, misses: 0, record: true }
    }
    pub fn time(
        &mut self,
        key: u64,
        launch: impl FnOnce() -> Result<f64, GpuRuntimeError>,
    ) -> Result<f64, GpuRuntimeError> {
        if let Some(&seconds) = self.cached.get(&key) {
            self.hits += 1;
            return Ok(seconds);
        }
        self.misses += 1;
        let seconds = launch()?;
        if self.record && seconds.is_finite() && seconds >= 0.0 {
            let entry = self.measured.entry(key).or_default();
            entry.0 += seconds;
            entry.1 += 1;
        }
        Ok(seconds)
    }
    pub fn publish(self) {
        let mut cache = PROBES.get_or_init(Default::default).lock().unwrap();
        for (key, (seconds, count)) in self.measured {
            if cache.len() >= MAX_PROBES {
                cache.pop_first();
            }
            cache.insert((self.namespace, key), seconds / count as f64);
        }
    }
}

/// Canonical identities of the actual one/two-iteration probes. Group membership
/// is represented by the limits of its operations, not by graph-local group IDs.
pub(super) fn probe_keys(
    frame: &PhysicalFrame,
    start: usize,
    end: usize,
    loops: &[SampledLoop<'_>],
) -> Vec<u64> {
    let groups = loops
        .iter()
        .enumerate()
        .flat_map(|(group, sample)| {
            sample.limits.iter().map(move |limit| ((*limit as *const _) as usize, group))
        })
        .collect::<BTreeMap<_, _>>();
    let mut limits = BTreeMap::new();
    fn visit(
        frame: &PhysicalFrame,
        ops: &[CompiledGpuOp],
        groups: &BTreeMap<usize, usize>,
        limits: &mut BTreeMap<PhysicalValueId, usize>,
    ) {
        for op in ops {
            if frame.program.implementations.resolve(op.implementation).unwrap().primitive ==
                GpuNativePrimitive::LoopWhile
            {
                let KernelArg::Value(id) = op.arguments[2] else { unreachable!() };
                let (_, bound) = frame.owners[&id].storages().next().unwrap();
                if let Some(&group) = groups.get(&(Arc::as_ptr(&bound.owner).cast::<()>() as usize))
                {
                    limits.insert(id, group);
                }
            }
            if let Some(body) = &op.body {
                visit(frame, body, groups, limits);
            }
        }
    }
    visit(frame, &frame.program.operations[start..end], &groups, &mut limits);
    std::iter::once(None)
        .chain((0..loops.len()).map(Some))
        .map(|probe| {
            let counts = limits
                .iter()
                .map(|(&id, &group)| (id, if probe == Some(group) { 2 } else { 1 }))
                .collect();
            work_key(frame, start, end, false, &counts)
        })
        .collect()
}

/// A key for the work of the operations `start..end` of `frame`, equal for
/// two regions that launch the same work: each operation's primitive, the
/// type, encoding and view shape of every value it reads or writes, its U64,
/// I64 and F64 arguments, the integer opcode, the kind of every other argument, its launch
/// shape and device, its dependencies relative to `start`, and its body.
/// Storage slots, byte offsets and binding numbers are left out: they say
/// where the data is, not how much work it is. With `drop_constants`, scalar
/// U64/U64List/I64/F64 values are omitted for I/O group reuse; argument kinds
/// and positions remain. Static loop counts and only the referenced native kernels
/// are included; unrelated registrations do not invalidate a region.
pub(super) fn region_key(
    frame: &PhysicalFrame,
    start: usize,
    end: usize,
    drop_constants: bool,
) -> u64 {
    work_key(frame, start, end, drop_constants, &BTreeMap::new())
}

fn work_key(
    frame: &PhysicalFrame,
    start: usize,
    end: usize,
    drop_constants: bool,
    probe_limits: &BTreeMap<PhysicalValueId, u64>,
) -> u64 {
    use std::hash::{Hash, Hasher};
    fn value(frame: &PhysicalFrame, id: PhysicalValueId, hasher: &mut impl Hasher) {
        match frame.program.values.get(id.0 as usize) {
            Some(physical) => {
                format!("{:?}{:?}", physical.ty, physical.encodings).hash(hasher);
                for part in physical.parts.iter() {
                    (part.leaf, part.device, &part.view.origin, &part.view.extent).hash(hasher);
                    (&part.view.byte_strides, part.view.element_bytes).hash(hasher);
                }
            }
            None => u64::MAX.hash(hasher),
        }
    }
    fn operation(
        frame: &PhysicalFrame,
        op: &CompiledGpuOp,
        base: usize,
        drop_constants: bool,
        probe_limits: &BTreeMap<PhysicalValueId, u64>,
        hasher: &mut impl Hasher,
    ) {
        format!(
            "{:?}",
            frame.program.implementations.resolve(op.implementation).map(|i| &i.primitive)
        )
        .hash(hasher);
        let primitive =
            &frame.program.implementations.resolve(op.implementation).unwrap().primitive;
        let probe_limit = if *primitive == GpuNativePrimitive::LoopWhile {
            let KernelArg::Value(id) = op.arguments[2] else { unreachable!() };
            let probe = probe_limits.get(&id).copied();
            if let Some(count) = probe {
                count.hash(hasher);
            } else {
                let (_, bound) = frame.owners[&id].storages().next().unwrap();
                let pointer = Arc::as_ptr(&bound.owner).cast::<()>();
                for control in &frame.control_resets {
                    if let crate::gpu_physical_control::ControlReset::Loop {
                        limit, count, ..
                    } = control
                    {
                        if Arc::as_ptr(limit).cast::<()>() == pointer {
                            count.hash(hasher);
                        }
                    }
                }
            }
            probe
        } else {
            None
        };
        if *primitive == GpuNativePrimitive::SubgraphKernel {
            let KernelArg::U32(index) = op.arguments[0] else { unreachable!() };
            let kernel = &frame.program.subgraph_kernels[index as usize];
            (&kernel.name, &kernel.parameters, kernel.scratch_bytes).hash(hasher);
            format!("{:?}{:?}{:?}", kernel.build_identity, kernel.inputs, kernel.outputs)
                .hash(hasher);
            // Process-local caches also distinguish different native entry points.
            (kernel.entry as usize).hash(hasher);
        }
        op.arguments.len().hash(hasher);
        for (position, argument) in op.arguments.iter().enumerate() {
            position.hash(hasher);
            std::mem::discriminant(argument).hash(hasher);
            match argument {
                KernelArg::Value(id) | KernelArg::OptionalValue(Some(id)) => {
                    value(frame, *id, hasher)
                }
                KernelArg::U32(word)
                    if position == 0 &&
                        frame
                            .program
                            .implementations
                            .resolve(op.implementation)
                            .unwrap()
                            .primitive ==
                            GpuNativePrimitive::IntegerOperation =>
                {
                    word.hash(hasher)
                }
                KernelArg::U64(word) => {
                    if !drop_constants {
                        if position == 6 && probe_limit.is_some() {
                            probe_limit.hash(hasher);
                        } else {
                            word.hash(hasher);
                        }
                    }
                }
                KernelArg::U64List(words) => {
                    if !drop_constants {
                        words.hash(hasher);
                    }
                }
                KernelArg::I64(word) => {
                    if !drop_constants {
                        word.hash(hasher);
                    }
                }
                KernelArg::F64(word) => {
                    if !drop_constants {
                        word.to_bits().hash(hasher);
                    }
                }
                KernelArg::OptionalValue(None) |
                KernelArg::U32(_) |
                KernelArg::OptionalBinding(_) => {}
            }
        }
        for &output in op.outputs.iter() {
            value(frame, output, hasher);
        }
        (op.device, op.grid, op.block, op.shared_bytes).hash(hasher);
        op.predecessors.len().hash(hasher);
        for &predecessor in op.predecessors.iter() {
            (predecessor as usize).checked_sub(base).unwrap_or(usize::MAX).hash(hasher);
        }
        match op.body.as_deref() {
            Some(body) => {
                body.len().hash(hasher);
                for inner in body {
                    operation(frame, inner, 0, drop_constants, probe_limits, hasher);
                }
            }
            None => usize::MAX.hash(hasher),
        }
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    format!("{:?}", crate::gpu_execution_plan::GpuControlExecution::current()).hash(&mut hasher);
    let native = crate::gpu_subgraph_kernel::GpuKernelBuildIdentity::current();
    (&native.backend, &native.architecture, &native.native_revision).hash(&mut hasher);
    for op in &frame.program.operations[start..end] {
        operation(frame, op, start, drop_constants, probe_limits, &mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_cache_batch_averages_and_isolation() {
        let namespace = rand::random();
        let mut first = ProbeMeasurements::in_namespace(namespace);
        first.record = false;
        assert_eq!(first.time(7, || Ok(100.0)).unwrap(), 100.0);
        first.record = true;
        assert_eq!(first.time(7, || Ok(2.0)).unwrap(), 2.0);
        assert_eq!(first.time(7, || Ok(4.0)).unwrap(), 4.0);
        assert_eq!(first.misses, 3, "warmups do not suppress measured repetitions");
        first.publish();
        let mut second = ProbeMeasurements::in_namespace(namespace);
        assert_eq!(second.time(7, || panic!("must reuse completed measurements")).unwrap(), 3.0);
        assert_eq!(second.hits, 1);
        let mut other = ProbeMeasurements::in_namespace(namespace ^ 1);
        assert_eq!(other.time(7, || Ok(9.0)).unwrap(), 9.0);
        assert_eq!(other.hits, 0);
        assert!(second.time(8, || Err(GpuRuntimeError::Execution("probe failed".into()))).is_err());
        second.publish();
        let mut third = ProbeMeasurements::in_namespace(namespace);
        assert_eq!(third.time(8, || Ok(5.0)).unwrap(), 5.0);
    }
}
