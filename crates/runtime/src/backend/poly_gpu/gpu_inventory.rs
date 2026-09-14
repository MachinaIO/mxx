//! Derive a complete prepared inventory from a validated graph.
//!
//! Root-scope liveness assigns retained outputs and transient operation demand
//! to reusable shape/class slots. A slot can serve a later node as soon as its
//! Rust owners have died; native reader events still order physical reuse. Artifact
//! inputs and exported outputs contribute the codec's staging and readback
//! resources. Unsupported kinds fail here, before any node executes. Scratch
//! is sized by the same native queries the admitted runners use; it is never
//! a bytes-per-column estimate.

use super::{
    gpu_compiled::{
        AdmittedScopeOperations, InventoryOperation, PreimageClaimPlan, PreimagePlanKey,
        PreparedMatrixOperation, TrapdoorPlanKey,
    },
    *,
};
use mxx_ir_core::{
    ValidatedGraph,
    graph::FrozenGraphScopeId,
    node::{ConcatAxis, LoopInputMode, NodeKind},
    types::{ConcreteWireType, Port, WireRef},
};
use mxx_primitives::{
    matrix::{
        PolyMatrix, PolyMatrixColumnData,
        gpu_dcrt_poly::{
            GpuCompactTransferKind, GpuCpuStagingLayout, GpuDCRTPolyMatrix, GpuGraphAdmissionGuard,
            GpuPreparedSlotKind, GpuPreparedSlotSnapshot, GpuPreparedStorage,
            GpuPreparedWorkspaceLayout, GpuTracedClaim, trace_native_claims,
        },
    },
    sampler::{PolyTrapdoorSampler, trapdoor::gpu::GpuDCRTPolyTrapdoorSampler},
};
use std::collections::BTreeMap;

/// Possible source boundaries and a descending list of containing owner widths.
/// Capacities are not execution intervals. For alternative family layouts, rank
/// k contains the kth largest fragment of every possible selected member.
#[derive(Clone, Default)]
pub struct GpuInventoryValue {
    pub cuts: Vec<usize>,
    pub capacities: Vec<usize>,
    pub alternatives: bool,
    // May contain descriptors with no native owner until materialization.
    // Alternatives use OR: eager members retain their separately counted owners.
    pub lazy: bool,
    pub packed_family: bool,
    pub borrowed: bool,
    // Preserve actual owner identity through aliases, never infer it from shape.
    pub owner: Option<u64>,
    /// CPU planning identity in a namespace separate from actual native owners.
    /// Equal shapes or equal bytes never establish this identity.
    pub symbolic_owner: Option<[u8; 32]>,
    pub broadcast: Option<WireRef>,
    pub fragments: Option<Arc<[super::gpu_prepare::MatrixInputFragment]>>,
}

impl GpuInventoryValue {
    pub fn new(mut cuts: Vec<usize>) -> Self {
        cuts.sort_unstable();
        cuts.dedup();
        let mut capacities = cuts.windows(2).map(|range| range[1] - range[0]).collect::<Vec<_>>();
        capacities.sort_unstable_by(|a, b| b.cmp(a));
        Self {
            cuts,
            capacities,
            alternatives: false,
            lazy: false,
            packed_family: false,
            borrowed: false,
            owner: None,
            symbolic_owner: None,
            broadcast: None,
            fragments: None,
        }
    }

    /// Fold a containing member/selection envelope without equating distinct
    /// identities. The caller supplies complete, valid layout metadata.
    pub fn include_alternative(&mut self, other: Self) {
        if other.cuts.is_empty() {
            return;
        }
        if self.cuts.is_empty() {
            *self = other;
            return;
        }
        if self.owner != other.owner {
            self.owner = None;
        }
        if self.symbolic_owner != other.symbolic_owner {
            self.symbolic_owner = None;
        }
        if self.broadcast != other.broadcast {
            self.broadcast = None;
        }
        if self.fragments != other.fragments {
            self.fragments = None;
        }
        self.borrowed &= other.borrowed;
        self.lazy |= other.lazy;
        self.packed_family |= other.packed_family;
        self.alternatives |= other.alternatives || self.cuts != other.cuts;
        self.cuts.extend(other.cuts);
        self.cuts.sort_unstable();
        self.cuts.dedup();
        self.capacities.resize(self.capacities.len().max(other.capacities.len()), 0);
        for (capacity, alternative) in self.capacities.iter_mut().zip(other.capacities) {
            *capacity = (*capacity).max(alternative);
        }
    }

    fn include_input(
        &mut self,
        value: &crate::backend::RuntimeValue<GpuDcrtBackend>,
        columns: usize,
    ) {
        use crate::backend::RuntimeValue;
        let cuts = match value {
            RuntimeValue::Matrix(matrix) | RuntimeValue::Trapdoor { public: matrix, .. } => matrix
                .shards()
                .iter()
                .flat_map(|shard| {
                    [shard.global_column_start, shard.global_column_start + shard.value.size().1]
                })
                .collect(),
            RuntimeValue::SmallMatrix(matrix) => matrix
                .shards()
                .iter()
                .flat_map(|shard| {
                    [shard.global_column_start, shard.global_column_start + shard.value.size().1]
                })
                .collect(),
            RuntimeValue::IndexedFamily(members) => {
                // Accumulate one bounded summary; do not retain a layout per
                // member or a traversal queue proportional to family cardinality.
                for member in members {
                    self.include_input(member, columns);
                }
                self.packed_family = true;
                return;
            }
            // Host/lazy members materialize as complete fresh matrices.
            _ => vec![0, columns],
        };
        let mut layout = Self::new(cuts);
        // The descriptor carries whether materialization already has a native
        // owner. This is the same fact at root and child input boundaries;
        // borrowing a family of resident owners does not allocate its members.
        layout.borrowed = matches!(
            value,
            RuntimeValue::Matrix(_) | RuntimeValue::SmallMatrix(_) | RuntimeValue::Trapdoor { .. }
        );
        if let RuntimeValue::Matrix(matrix) | RuntimeValue::Trapdoor { public: matrix, .. } = value
        {
            layout.owner = Some(matrix.id);
            layout.fragments = Some(matrix.input_layout.clone());
        }
        layout.lazy = matches!(
            value,
            RuntimeValue::LazyArtifact { .. } |
                RuntimeValue::StagedArtifact { .. } |
                RuntimeValue::HostMatrix { .. } |
                RuntimeValue::LazyArtifactFamily { .. } |
                RuntimeValue::StagedArtifactFamily { .. }
        );
        self.include_alternative(layout);
    }
}

/// Proven owner identity. A cold broadcast names the single import that the
/// executor performs after admission; it is never a fabricated native owner ID.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum MatrixInputOwner {
    Actual(u64),
    Symbolic([u8; 32]),
    Broadcast(WireRef),
}

/// Host dispatch progress of one sibling within its scope. Issued keeps the
/// current node's resource bound until group liveness/release is applied; it
/// does not assert GPU completion. After group release use Before(next_node).
/// Positions are indices in the validated scope's execution order. This is a
/// conservative capacity bound, not a replacement for native pending-reader
/// eligibility: an issued operation may still own temporary storage.
#[derive(Clone, Copy, Debug)]
pub enum GpuScopeProgress {
    Before(usize),
    Issued(usize),
}

#[derive(Clone, PartialEq, Eq)]
enum ResourceOrigins {
    // Imported once before body entry; its enclosing placement retains the bound.
    Placement,
    One(usize),
    Shared(Vec<usize>),
}

impl ResourceOrigins {
    fn indices(&self) -> &[usize] {
        match self {
            Self::Placement => &[],
            Self::One(index) => std::slice::from_ref(index),
            Self::Shared(indices) => indices,
        }
    }
    fn shift(&mut self, offset: usize) {
        match self {
            Self::Placement => {}
            Self::One(index) => *index += offset,
            Self::Shared(indices) => {
                for index in indices {
                    *index += offset;
                }
            }
        }
    }
    fn include(&mut self, other: Self) {
        if *self == other {
            return;
        }
        if matches!(self, Self::Placement) || matches!(other, Self::Placement) {
            *self = Self::Placement;
            return;
        }
        let mut indices = self.indices().iter().chain(other.indices()).copied().collect::<Vec<_>>();
        indices.sort_unstable();
        indices.dedup();
        *self = Self::Shared(indices);
    }
    fn retained(
        &self,
        begin: usize,
        end: usize,
        progress: &impl Fn(usize) -> GpuScopeProgress,
    ) -> bool {
        if matches!(self, Self::Placement) {
            return true;
        }
        self.indices().iter().any(|&index| match progress(index) {
            GpuScopeProgress::Before(position) => begin < position && end >= position,
            GpuScopeProgress::Issued(position) => begin <= position && end >= position,
        })
    }
}

// Scope-local value provenance identifies containing resource bounds, never
// actual native addresses. Child references preserve the scope tree without
// expanding aliases or nested family members into flat owner lists.
#[derive(Clone, PartialEq, Eq)]
enum ValueOrigin {
    Direct {
        instance: Option<usize>,
        wire: WireRef,
        members: Vec<usize>,
    },
    Child {
        instance: usize,
        wire: WireRef,
        output: WireRef,
        parallel: bool,
        demand: Arc<GpuContextDemand>,
        claim: usize,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MemberProjection {
    Same,
    Pack(usize),
    Static(usize),
    Dynamic,
}
type ValueAliases = BTreeMap<WireRef, Vec<(WireRef, MemberProjection)>>;
type ValueQuery = (usize, usize, WireRef, Vec<Option<usize>>);

fn merge_value_origins(
    target: &mut Vec<ValueOrigin>,
    extra: impl IntoIterator<Item = ValueOrigin>,
) {
    for origin in extra {
        if target.contains(&origin) {
            continue;
        }
        if let ValueOrigin::Child { instance, wire, output, parallel, demand, claim } = &origin {
            let compatible = target.iter_mut().find_map(|previous| {
                let ValueOrigin::Child {
                    instance: previous_instance,
                    wire: previous_wire,
                    output: previous_output,
                    parallel: previous_parallel,
                    demand: previous_demand,
                    claim: previous_claim,
                } = previous
                else {
                    return None
                };
                (*instance == *previous_instance &&
                    *wire == *previous_wire &&
                    *output == *previous_output &&
                    *parallel == *previous_parallel &&
                    *claim == *previous_claim &&
                    previous_demand.same_timeline(demand))
                .then_some(previous_demand)
            });
            if let Some(previous) = compatible {
                // The claim numbering and numeric lifetimes are identical.
                // Merge only the value-choice metadata, recursively, rather
                // than keeping a child tree for every concrete loop index.
                Arc::make_mut(previous).include_alternative((**demand).clone());
                continue;
            }
        }
        target.push(origin);
    }
}

fn merge_value_aliases<'a>(
    maps: impl IntoIterator<Item = &'a Arc<ValueAliases>>,
) -> Arc<ValueAliases> {
    let mut result = ValueAliases::new();
    for map in maps {
        for (&wire, edges) in map.iter() {
            let output = result.entry(wire).or_default();
            for &(source, projection) in edges {
                // An alternative index is a containing member-choice bound;
                // do not retain one copy of the alias graph per loop index.
                if let Some((_, previous)) = output.iter_mut().find(|(s, p)| {
                    *s == source &&
                        matches!(
                            (p, projection),
                            (
                                MemberProjection::Static(_) | MemberProjection::Dynamic,
                                MemberProjection::Static(_) | MemberProjection::Dynamic
                            )
                        )
                }) {
                    if *previous != projection {
                        *previous = MemberProjection::Dynamic;
                    }
                } else if !output.contains(&(source, projection)) {
                    output.push((source, projection));
                }
            }
        }
    }
    Arc::new(result)
}

/// One matrix claim at an existing IR position, not another instruction stream.
/// Only input preparation shared by a native sibling batch has an identity.
#[derive(Clone, PartialEq, Eq)]
struct MatrixDemand {
    owners: Vec<ValueOrigin>,
    origins: ResourceOrigins,
    capacity_slot: usize,
    begin: usize,
    end: usize,
    rows: usize,
    columns: usize,
    preparation: Option<(MatrixInputOwner, usize, bool)>, // owner, fragment ordinal, format
}

/// A workspace or opaque resource at its actual IR lifetime. Shared entries
/// describe one native sibling-batch resource, not equal-shaped value owners.
#[derive(Clone, PartialEq, Eq)]
struct WorkspaceDemand {
    owners: Vec<ValueOrigin>,
    origins: ResourceOrigins,
    capacity_slot: usize,
    begin: usize,
    end: usize,
    layout: GpuPreparedWorkspaceLayout,
    shared: bool,
}

/// Native demand of one parameter context on one device.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct GpuContextDemand {
    aliases: Vec<Arc<ValueAliases>>,
    // Zero represents one unmerged scope; combined scopes record their count.
    instance_count: usize,
    workspace_demands: Vec<WorkspaceDemand>,
    matrix_demands: Vec<MatrixDemand>,
    matrices: Vec<(usize, usize)>,
    matrix_until: Vec<usize>,
    layouts: Vec<GpuPreparedWorkspaceLayout>,
    layout_until: Vec<usize>,
    layout_shared: Vec<bool>,
}

impl GpuContextDemand {
    fn same_timeline(&self, other: &Self) -> bool {
        self.instance_count == other.instance_count &&
            self.matrix_demands.len() == other.matrix_demands.len() &&
            self.workspace_demands.len() == other.workspace_demands.len() &&
            self.matrix_demands.iter().zip(&other.matrix_demands).all(|(a, b)| {
                (&a.origins, a.capacity_slot, a.begin, a.end, a.rows, a.columns, a.preparation) ==
                    (
                        &b.origins,
                        b.capacity_slot,
                        b.begin,
                        b.end,
                        b.rows,
                        b.columns,
                        b.preparation,
                    )
            }) &&
            self.workspace_demands.iter().zip(&other.workspace_demands).all(|(a, b)| {
                (&a.origins, a.capacity_slot, a.begin, a.end, a.layout, a.shared) ==
                    (&b.origins, b.capacity_slot, b.begin, b.end, b.layout, b.shared)
            })
    }

    /// Siblings execute the same IR positions together. Merge their lifetimes
    /// before fitting slots, instead of adding independently maximized peaks.
    /// Only proven preparation identities at the same position are shared;
    /// ordinary outputs and unresolved aliases always remain separate owners.
    pub fn simultaneous(instances: impl IntoIterator<Item = Self>) -> Self {
        let mut merged = Self::default();
        let mut matrices = Vec::<MatrixDemand>::new();
        let mut shared_matrices = HashMap::new();
        let mut workspaces = Vec::<WorkspaceDemand>::new();
        let mut shared_workspaces = HashMap::new();
        for instance in instances {
            let offset = merged.instance_count;
            merged.instance_count += instance.instance_count.max(1);
            if instance.aliases.is_empty() {
                merged.aliases.extend(
                    (0..instance.instance_count.max(1)).map(|_| Arc::new(ValueAliases::new())),
                );
            } else {
                merged.aliases.extend(instance.aliases);
            }
            for mut matrix in instance.matrix_demands {
                matrix.origins.shift(offset);
                for owner in &mut matrix.owners {
                    match owner {
                        ValueOrigin::Direct { instance: Some(instance), .. } |
                        ValueOrigin::Child { instance, .. } => *instance += offset,
                        ValueOrigin::Direct { instance: None, .. } => {}
                    }
                }
                if let Some(identity) = matrix.preparation {
                    let index =
                        *shared_matrices.entry((matrix.begin, identity)).or_insert(matrices.len());
                    if let Some(previous) = matrices.get_mut(index) {
                        previous.rows = previous.rows.max(matrix.rows);
                        previous.columns = previous.columns.max(matrix.columns);
                        previous.end = previous.end.max(matrix.end);
                        previous.origins.include(matrix.origins);
                        merge_value_origins(&mut previous.owners, matrix.owners);
                        continue;
                    }
                }
                matrices.push(matrix);
            }
            let mut ordinals = HashMap::new();
            for mut workspace in instance.workspace_demands {
                workspace.origins.shift(offset);
                for owner in &mut workspace.owners {
                    match owner {
                        ValueOrigin::Direct { instance: Some(instance), .. } |
                        ValueOrigin::Child { instance, .. } => *instance += offset,
                        ValueOrigin::Direct { instance: None, .. } => {}
                    }
                }
                if workspace.shared {
                    // Preserve simultaneous duplicate claims inside one body,
                    // while sharing the corresponding batch claim across bodies.
                    let ordinal = ordinals.entry((workspace.begin, workspace.layout)).or_insert(0);
                    let index = *shared_workspaces
                        .entry((workspace.begin, workspace.layout, *ordinal))
                        .or_insert(workspaces.len());
                    *ordinal += 1;
                    if let Some(previous) = workspaces.get_mut(index) {
                        previous.end = previous.end.max(workspace.end);
                        previous.origins.include(workspace.origins);
                        merge_value_origins(&mut previous.owners, workspace.owners);
                        continue;
                    }
                }
                workspaces.push(workspace);
            }
        }
        matrices.par_sort_unstable_by_key(|matrix| matrix.begin);
        for matrix in matrices {
            merged.matrix(matrix.begin, matrix.end, matrix.rows, matrix.columns);
            // Another enclosing sibling merge must still recognize this exact
            // prepared input. Equal-shaped ordinary outputs remain anonymous.
            let retained = merged.matrix_demands.last_mut().unwrap();
            retained.preparation = matrix.preparation;
            retained.origins = matrix.origins;
            retained.owners = matrix.owners;
        }
        workspaces.par_sort_unstable_by_key(|workspace| workspace.begin);
        for workspace in workspaces {
            merged.workspace(workspace.begin, workspace.end, workspace.layout, workspace.shared);
            let retained = merged.workspace_demands.last_mut().unwrap();
            retained.origins = workspace.origins;
            retained.owners = workspace.owners;
        }
        merged
    }

    /// Bound alternative executions without charging their peaks simultaneously.
    /// Keep at most the largest observed slot count in each native size class.
    fn include_alternative(&mut self, other: Self) {
        use super::gpu_prepare::{capacity_class, matrix_capacity_class};
        // One resolved execution is not an unresolved alternative. Preserve
        // its positions and proven preparation identities until another
        // possible execution actually needs a containing bound.
        if self.matrices.is_empty() && self.layouts.is_empty() {
            *self = other;
            return;
        }
        if *self == other {
            // Repeating an unchanged class is not a new unresolved alternative.
            return;
        }
        if self.same_timeline(&other) {
            self.aliases = (0..self.aliases.len().max(other.aliases.len()))
                .map(|index| {
                    merge_value_aliases(
                        self.aliases.get(index).into_iter().chain(other.aliases.get(index)),
                    )
                })
                .collect();
            for (a, b) in self.matrix_demands.iter_mut().zip(other.matrix_demands) {
                merge_value_origins(&mut a.owners, b.owners);
            }
            for (a, b) in self.workspace_demands.iter_mut().zip(other.workspace_demands) {
                merge_value_origins(&mut a.owners, b.owners);
            }
            return;
        }
        self.aliases = vec![merge_value_aliases(self.aliases.iter().chain(&other.aliases))];
        let mut matrix_owners = vec![Vec::new(); self.matrices.len()];
        for demand in &self.matrix_demands {
            merge_value_origins(
                &mut matrix_owners[demand.capacity_slot],
                demand.owners.iter().cloned(),
            );
        }
        let mut other_matrix_owners = vec![Vec::new(); other.matrices.len()];
        for demand in &other.matrix_demands {
            merge_value_origins(
                &mut other_matrix_owners[demand.capacity_slot],
                demand.owners.iter().cloned(),
            );
        }
        let mut layout_owners = vec![Vec::new(); self.layouts.len()];
        for demand in &self.workspace_demands {
            merge_value_origins(
                &mut layout_owners[demand.capacity_slot],
                demand.owners.iter().cloned(),
            );
        }
        let mut other_layout_owners = vec![Vec::new(); other.layouts.len()];
        for demand in &other.workspace_demands {
            merge_value_origins(
                &mut other_layout_owners[demand.capacity_slot],
                demand.owners.iter().cloned(),
            );
        }
        let mut matrix_ranks = HashMap::new();
        for (((rows, columns), end), owners) in
            other.matrices.into_iter().zip(other.matrix_until).zip(other_matrix_owners)
        {
            let class = matrix_capacity_class(rows, columns);
            let rank = matrix_ranks.entry(class).or_insert(0);
            let slot = self
                .matrices
                .iter()
                .enumerate()
                .filter(|(_, (r, c))| matrix_capacity_class(*r, *c) == class)
                .nth(*rank)
                .map(|(index, _)| index);
            *rank += 1;
            if let Some(index) = slot {
                self.matrices[index].0 = self.matrices[index].0.max(rows);
                self.matrices[index].1 = self.matrices[index].1.max(columns);
                self.matrix_until[index] = self.matrix_until[index].max(end);
                merge_value_origins(&mut matrix_owners[index], owners);
            } else {
                self.matrices.push((rows, columns));
                self.matrix_until.push(end);
                matrix_owners.push(owners);
            }
        }
        let mut layout_ranks = HashMap::new();
        for (((layout, end), shared), owners) in other
            .layouts
            .into_iter()
            .zip(other.layout_until)
            .zip(other.layout_shared)
            .zip(other_layout_owners)
        {
            let class = (layout.kind, layout.alignment, capacity_class(layout.bytes), shared);
            let rank = layout_ranks.entry(class).or_insert(0);
            let slot = self
                .layouts
                .iter()
                .enumerate()
                .filter(|(index, previous)| {
                    (
                        previous.kind,
                        previous.alignment,
                        capacity_class(previous.bytes),
                        self.layout_shared[*index],
                    ) == class
                })
                .nth(*rank)
                .map(|(index, _)| index);
            *rank += 1;
            if let Some(index) = slot {
                self.layouts[index].bytes = self.layouts[index].bytes.max(layout.bytes);
                self.layout_until[index] = self.layout_until[index].max(end);
                merge_value_origins(&mut layout_owners[index], owners);
            } else {
                self.layouts.push(layout);
                self.layout_until.push(end);
                self.layout_shared.push(shared);
                layout_owners.push(owners);
            }
        }
        // Alternatives are one enclosing scope's conservative envelope.
        self.instance_count = 0;
        self.matrix_demands = self
            .matrices
            .iter()
            .zip(&self.matrix_until)
            .enumerate()
            .map(|(capacity_slot, (&(rows, columns), &end))| MatrixDemand {
                owners: matrix_owners[capacity_slot]
                    .iter()
                    .cloned()
                    .map(|mut origin| {
                        match &mut origin {
                            ValueOrigin::Direct { instance: Some(instance), .. } |
                            ValueOrigin::Child { instance, .. } => *instance = 0,
                            ValueOrigin::Direct { instance: None, .. } => {}
                        }
                        origin
                    })
                    .collect(),
                origins: ResourceOrigins::One(0),
                capacity_slot,
                begin: 0,
                end,
                rows,
                columns,
                preparation: None,
            })
            .collect();
        self.workspace_demands = self
            .layouts
            .iter()
            .zip(&self.layout_until)
            .zip(&self.layout_shared)
            .enumerate()
            .map(|(capacity_slot, ((&layout, &end), &shared))| WorkspaceDemand {
                owners: layout_owners[capacity_slot]
                    .iter()
                    .cloned()
                    .map(|mut origin| {
                        match &mut origin {
                            ValueOrigin::Direct { instance: Some(instance), .. } |
                            ValueOrigin::Child { instance, .. } => *instance = 0,
                            ValueOrigin::Direct { instance: None, .. } => {}
                        }
                        origin
                    })
                    .collect(),
                origins: ResourceOrigins::One(0),
                capacity_slot,
                begin: 0,
                end,
                layout,
                shared,
            })
            .collect();
    }
    fn matrix(&mut self, begin: usize, end: usize, rows: usize, columns: usize) {
        use super::gpu_prepare::matrix_capacity_class;
        let class = matrix_capacity_class(rows, columns);
        let mut shape = (rows, columns);
        let mut available = None;
        for (index, &(r, c)) in self.matrices.iter().enumerate() {
            if matrix_capacity_class(r, c) == class {
                shape.0 = shape.0.max(r);
                shape.1 = shape.1.max(c);
                if self.matrix_until[index] < begin {
                    available = Some(index);
                }
            }
        }
        // Every slot of a class has the same containing shape, making the
        // native smallest-fit order irrelevant within that class. Dimensions
        // use observed maxima, not power-of-two padded allocations.
        for slot in &mut self.matrices {
            if matrix_capacity_class(slot.0, slot.1) == class {
                *slot = shape;
            }
        }
        let capacity_slot = if let Some(index) = available {
            self.matrix_until[index] = end;
            index
        } else {
            self.matrices.push(shape);
            self.matrix_until.push(end);
            self.matrices.len() - 1
        };
        self.matrix_demands.push(MatrixDemand {
            owners: Vec::new(),
            origins: ResourceOrigins::One(0),
            capacity_slot,
            begin,
            end,
            rows,
            columns,
            preparation: None,
        });
    }

    /// Indices in `claims()` retained at each sibling's supplied host frontier.
    /// Sibling indices follow flattened merge order, including empty siblings;
    /// a shared resource stays retained if any originating sibling retains it.
    /// Before excludes future producers; Issued includes the current node's
    /// bound until group release. These are capacity-assignment indices, never
    /// native IDs, and require this exact demand, assignment and scope timeline.
    /// When issued_at is supplied, follow only that allocation epoch. A pooled
    /// index reused by a later node must not resurrect an earlier assignment.
    pub fn retained_claim_indices(
        &self,
        issued_at: Option<usize>,
        progress: impl Fn(usize) -> GpuScopeProgress,
    ) -> std::collections::BTreeSet<usize> {
        self.matrix_demands
            .iter()
            .filter(|demand| {
                issued_at.is_none_or(|node| demand.begin == node) &&
                    demand.origins.retained(demand.begin, demand.end, &progress)
            })
            .map(|demand| demand.capacity_slot)
            .chain(
                self.workspace_demands
                    .iter()
                    .filter(|demand| {
                        issued_at.is_none_or(|node| demand.begin == node) &&
                            demand.origins.retained(demand.begin, demand.end, &progress)
                    })
                    .map(|demand| self.matrices.len() + demand.capacity_slot),
            )
            .collect()
    }

    /// Claims in this exact scope demand that contain the selected value's
    /// owned resource bound at the supplied host frontier. An empty member path
    /// selects the whole value/family; a prefix selects that nested member.
    /// Borrowed outer-scope owners are not allocated by this demand and must be
    /// resolved through the caller's established source bindings. These are
    /// assignment indices, not proof of actual native output addresses or readiness.
    pub fn value_claim_indices(
        &self,
        instance: usize,
        wire: WireRef,
        members: &[usize],
        progress: GpuScopeProgress,
    ) -> std::collections::BTreeSet<usize> {
        self.value_claims(
            instance,
            wire,
            &members.iter().copied().map(Some).collect::<Vec<_>>(),
            progress,
            &mut HashMap::new(),
        )
    }

    fn value_claims(
        &self,
        instance: usize,
        wire: WireRef,
        members: &[Option<usize>],
        progress: GpuScopeProgress,
        memo: &mut HashMap<ValueQuery, std::collections::BTreeSet<usize>>,
    ) -> std::collections::BTreeSet<usize> {
        let mut queries = std::collections::BTreeSet::new();
        let mut pending = vec![(wire, members.to_vec())];
        while let Some((wire, path)) = pending.pop() {
            if !queries.insert((wire, path.clone())) {
                continue;
            }
            for &(source, projection) in self
                .aliases
                .get(instance)
                .and_then(|aliases| aliases.get(&wire))
                .into_iter()
                .flatten()
            {
                let path = match projection {
                    MemberProjection::Same => path.clone(),
                    MemberProjection::Pack(index) => {
                        if path.first().is_some_and(|selected| {
                            selected.is_some_and(|selected| selected != index)
                        }) {
                            continue;
                        }
                        if path.is_empty() { Vec::new() } else { path[1..].to_vec() }
                    }
                    MemberProjection::Static(index) => {
                        let mut result = vec![Some(index)];
                        result.extend(&path);
                        result
                    }
                    MemberProjection::Dynamic => {
                        let mut result = vec![None];
                        result.extend(&path);
                        result
                    }
                };
                pending.push((source, path));
            }
        }
        let mut matches = |owners: &[ValueOrigin]| {
            owners.iter().any(|owner| match owner {
                ValueOrigin::Direct { instance: owner, wire, members } => {
                    owner.is_none_or(|owner| owner == instance) &&
                        queries.iter().any(|(value, path)| {
                            value == wire &&
                                path.len() <= members.len() &&
                                path.iter().zip(members).all(|(requested, actual)| {
                                    requested.is_none_or(|requested| requested == *actual)
                                })
                        })
                }
                ValueOrigin::Child { instance: owner, wire, output, parallel, demand, claim } => {
                    if *owner != instance {
                        return false;
                    }
                    queries.iter().filter(|(value, _)| value == wire).any(|(_, path)| {
                        let (start, end, path) = if *parallel {
                            match path.first().copied().flatten() {
                                Some(index) => (index, index + 1, &path[1..]),
                                None => (
                                    0,
                                    demand.instance_count.max(1),
                                    if path.is_empty() { &path[..] } else { &path[1..] },
                                ),
                            }
                        } else {
                            (0, 1, &path[..])
                        };
                        (start..end).any(|child| {
                            let key = (Arc::as_ptr(demand) as usize, child, *output, path.to_vec());
                            if !memo.contains_key(&key) {
                                let claims = demand.value_claims(
                                    child,
                                    *output,
                                    path,
                                    GpuScopeProgress::Before(usize::MAX),
                                    memo,
                                );
                                memo.insert(key.clone(), claims);
                            }
                            memo[&key].contains(claim)
                        })
                    })
                }
            })
        };
        let mut result = std::collections::BTreeSet::new();
        for demand in &self.matrix_demands {
            if demand.origins.retained(demand.begin, demand.end, &|_| progress) &&
                matches(&demand.owners)
            {
                result.insert(demand.capacity_slot);
            }
        }
        for demand in &self.workspace_demands {
            if demand.origins.retained(demand.begin, demand.end, &|_| progress) &&
                matches(&demand.owners)
            {
                result.insert(self.matrices.len() + demand.capacity_slot);
            }
        }
        result
    }

    /// Complete typed capacity of this demand using the same claims as live
    /// reservation. This neither allocates backing nor acquires native leases.
    pub fn claims(&self, params: &GpuDCRTPolyParams) -> Vec<GpuTracedClaim> {
        self.matrices
            .iter()
            .map(|&(rows, columns)| {
                GpuTracedClaim::matrix(rows.max(1), columns.max(1), params.crt_depth() - 1, true)
            })
            .chain(self.layouts.iter().copied().map(GpuTracedClaim::workspace))
            .collect()
    }

    fn missing(
        &self,
        params: &GpuDCRTPolyParams,
        inventory: &[(usize, Arc<GpuPreparedStorage>)],
        device: usize,
        used: &mut HashSet<u64>,
    ) -> Result<(Vec<GpuTracedClaim>, Vec<GpuPreparedWorkspaceLayout>), PolyBackendError> {
        let broker = super::gpu_compiled::PreparedClaimBroker::new(
            params,
            inventory
                .iter()
                .filter(|(owner, _)| *owner == device)
                .map(|(_, storage)| storage.clone())
                .collect(),
        );
        let claims = self.claims(params);
        let mut matrices = Vec::new();
        let mut layouts = Vec::new();
        for claim in broker.missing(&claims, used).map_err(PolyBackendError::GpuSubmission)? {
            if claim.kind() == GpuPreparedSlotKind::Matrix {
                matrices.push(claim);
            } else {
                layouts.push(claim.layout().unwrap());
            }
        }
        // A new workspace-only storage still needs the native constructor's
        // minimum matrix owner. Include it in both planning and real allocation.
        if matrices.is_empty() && !layouts.is_empty() {
            matrices.push(GpuTracedClaim::matrix(1, 1, params.crt_depth() - 1, true));
        }
        Ok((matrices, layouts))
    }

    fn workspace(
        &mut self,
        begin: usize,
        end: usize,
        mut layout: GpuPreparedWorkspaceLayout,
        shared: bool,
    ) {
        let requested_layout = layout;
        use super::gpu_prepare::capacity_class;
        let matches = |previous: &GpuPreparedWorkspaceLayout| {
            previous.kind == layout.kind &&
                previous.alignment == layout.alignment &&
                capacity_class(previous.bytes) == capacity_class(layout.bytes)
        };
        let indices = self
            .layouts
            .iter()
            .enumerate()
            .filter_map(|(index, previous)| {
                (matches(previous) && self.layout_shared[index] == shared).then_some(index)
            })
            .collect::<Vec<_>>();
        let available = indices.iter().copied().find(|&index| self.layout_until[index] < begin);
        for &index in &indices {
            layout.bytes = layout.bytes.max(self.layouts[index].bytes);
        }
        for index in indices {
            self.layouts[index].bytes = layout.bytes;
        }
        // Zero-byte events and streams still need one slot per simultaneous
        // owner; kind and alignment remain distinct native resource domains.
        let capacity_slot = if let Some(index) = available {
            self.layout_until[index] = end;
            index
        } else {
            self.layouts.push(layout);
            self.layout_until.push(end);
            self.layout_shared.push(shared);
            self.layouts.len() - 1
        };
        self.workspace_demands.push(WorkspaceDemand {
            owners: Vec::new(),
            origins: ResourceOrigins::One(0),
            capacity_slot,
            begin,
            end,
            layout: requested_layout,
            shared,
        });
    }
}

fn stream() -> GpuPreparedWorkspaceLayout {
    GpuPreparedWorkspaceLayout {
        kind: GpuPreparedSlotKind::SubmissionStream,
        bytes: 0,
        alignment: 1,
    }
}

fn event() -> GpuPreparedWorkspaceLayout {
    GpuPreparedWorkspaceLayout {
        kind: GpuPreparedSlotKind::CompletionEvent,
        bytes: 0,
        alignment: 1,
    }
}

fn compact_payload(
    parameters: &GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
    bound: &num_bigint::BigUint,
) -> Result<GpuPreparedWorkspaceLayout, PolyBackendError> {
    Ok(GpuPreparedWorkspaceLayout {
        kind: GpuPreparedSlotKind::CompactPayload,
        bytes: GpuSmallMatrix::allocation_bytes(parameters, rows, columns, bound)
            .map_err(|e| PolyBackendError::GpuSubmission(e.to_string()))?,
        alignment: 256,
    })
}

fn bound_of(wire_type: &ConcreteWireType) -> Option<(&ConcreteMatrixType, num_bigint::BigUint)> {
    match wire_type {
        ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } |
        ConcreteWireType::Preimage { matrix, max_coefficient_bound } => {
            Some((matrix, max_coefficient_bound.to_biguint()?))
        }
        _ => None,
    }
}

// Families own their leaves; structural selections only alias those owners.
fn family_leaves(
    ty: &ConcreteWireType,
) -> Box<dyn Iterator<Item = (Vec<usize>, &ConcreteWireType)> + '_> {
    match ty {
        ConcreteWireType::IndexedFamily { count, element } => {
            Box::new((0..*count).flat_map(move |index| {
                family_leaves(element).map(move |(mut path, leaf)| {
                    path.insert(0, index);
                    (path, leaf)
                })
            }))
        }
        _ => Box::new(std::iter::once((Vec::new(), ty))),
    }
}

fn import_inventory(
    parameters: &GpuDCRTPolyParams,
    output: &ConcreteWireType,
    scratch_columns: usize,
) -> Result<(Vec<(usize, usize)>, Vec<GpuPreparedWorkspaceLayout>), PolyBackendError> {
    let Some(ty) = output.matrix_type() else { return Ok((Vec::new(), Vec::new())) };
    let operations = match output {
        ConcreteWireType::Matrix(_) => vec![
            PreparedMatrixOperation::ImportMatrix {
                ty: ty.clone(),
                evaluation: true,
                max_coefficient_bits: u16::try_from(parameters.modulus().bits())
                    .map_err(|_| PolyBackendError::InvalidInteger)?,
            },
            // A descriptor may resolve to compact artifact bytes or raw RNS
            // staging. Bound both through the same production resource methods;
            // payload-dependent fields do not affect these native queries.
            PreparedMatrixOperation::ImportStaging {
                ty: ty.clone(),
                evaluation: true,
                bytes_per_poly: 0,
                payload_len: 0,
            },
        ],
        ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. } => {
            let (_, bound) = bound_of(output).ok_or(PolyBackendError::InvalidInteger)?;
            vec![PreparedMatrixOperation::ImportCompact { ty: ty.clone(), bound }]
        }
        _ => return Ok((Vec::new(), Vec::new())),
    };
    let level = parameters.crt_depth() - 1;
    let columns = ty.columns.min(scratch_columns);
    let mut demand = GpuContextDemand::default();
    for operation in operations {
        let mut alternative = GpuContextDemand::default();
        for rows in operation.scratch_rows()? {
            alternative.matrix(0, usize::MAX, rows, columns);
        }
        for layout in operation
            .fixed_workspaces(parameters, level)?
            .into_iter()
            .chain(operation.width_workspaces(parameters, level, columns)?)
        {
            alternative.workspace(0, usize::MAX, layout, false);
        }
        demand.include_alternative(alternative);
    }
    Ok((demand.matrices, demand.layouts))
}

/// Add traced claims as inventory demand: matrix owners become backing
/// matrices of the traced shape, other kinds become typed workspace layouts.
fn push_traced(
    demand: &mut BTreeMap<(String, usize), (ConcreteMatrixType, GpuContextDemand)>,
    ty: &ConcreteMatrixType,
    parameters: &GpuDCRTPolyParams,
    claims: &[GpuTracedClaim],
    add_matrix: &impl Fn(
        &mut BTreeMap<(String, usize), (ConcreteMatrixType, GpuContextDemand)>,
        &ConcreteMatrixType,
        usize,
        usize,
    ),
    add_layouts: &impl Fn(
        &mut BTreeMap<(String, usize), (ConcreteMatrixType, GpuContextDemand)>,
        &ConcreteMatrixType,
        Vec<GpuPreparedWorkspaceLayout>,
    ),
) {
    let _ = parameters;
    for claim in claims {
        if claim.kind() == GpuPreparedSlotKind::Matrix {
            add_matrix(demand, ty, claim.rows(), claim.columns());
        } else if let Some(layout) = claim.layout() {
            add_layouts(demand, ty, vec![layout]);
        }
    }
}

/// Largest wave that fits together with its widest fitting scratch-column cap.
///
/// Waves are searched from `wave_bound` down to one; inside each wave the cap
/// list is the deduplicated halving chain `maximum, ceil(maximum / 2), ..., 1`.
/// The first fitting pair is the answer, so a wave that only fits after the
/// scratch envelope shrinks is preferred over a narrower wave at a wider
/// envelope. `probe` is metadata-only candidate evaluation: it must not
/// allocate, reserve, materialize or launch anything, so a rejected candidate
/// leaves no trace. The probe returns the candidate's payload when it fits.
fn select_wave_and_width<T, E>(
    wave_bound: usize,
    maximum: usize,
    mut probe: impl FnMut(usize, usize) -> Result<Option<T>, E>,
) -> Result<Option<(usize, usize, T)>, E> {
    for wave in (1..=wave_bound.max(1)).rev() {
        let mut width = maximum.max(1);
        loop {
            if let Some(value) = probe(wave, width)? {
                return Ok(Some((wave, width, value)));
            }
            if width == 1 {
                break;
            }
            width = width.div_ceil(2);
        }
    }
    Ok(None)
}

/// CPU resources and derived source layouts for one ordered scope candidate.
/// The per-instance maps retain IR wire identity and complete resource bounds,
/// including alternatives whose exact source format/placement is unresolved.
/// Optional fragments are concrete only when established by shared lowering;
/// a containing bound never invents an exact format, source or native owner.
pub struct GpuScopeResources {
    pub contexts: BTreeMap<(String, usize), (ConcreteMatrixType, GpuContextDemand)>,
    pub value_layouts: Vec<BTreeMap<WireRef, GpuInventoryValue>>,
}

impl GpuDcrtBackend {
    /// Trace the exact claims of sampling one trapdoor class and preparing its
    /// preimage covariance cache, on the still-open domains of device 0.
    fn trace_trapdoor_sampling(
        &mut self,
        matrix: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &num_bigint::BigInt,
        digit_count: usize,
        warm_up: bool,
    ) -> Result<super::gpu_compiled::TrapdoorClaimPlan, PolyBackendError> {
        let key = TrapdoorPlanKey {
            modulus: matrix.modulus.to_string(),
            ring_dimension: matrix.ring_dimension,
            rows: matrix.rows,
            columns: matrix.columns,
            sigma_bits: sigma.to_bits(),
            gadget_base: gadget_base.to_string(),
            digit_count,
        };
        if let Some(claims) = self.trapdoor_plans.get(&key) {
            return Ok(claims.clone());
        }
        if !warm_up {
            return Err(PolyBackendError::GpuSubmission(format!(
                "explicit graph warmup required: missing trace_trapdoor_sampling resource plan for {key:?}"
            )));
        }
        let params = self.devices[0].1.parameters(matrix)?.clone();
        let device = &mut self.devices[0].1;
        let (sampled, mut claims) =
            trace_native_claims(|| device.sample_trapdoor(matrix, sigma, gadget_base, digit_count))
                .map_err(PolyBackendError::GpuSubmission)?;
        let (_, trapdoor) = sampled?;
        let sampler = <GpuDCRTPolyTrapdoorSampler as PolyTrapdoorSampler>::new(&params, sigma);
        let (_, cache) =
            trace_native_claims(|| sampler.prepare_preimage_cache(&params, &trapdoor, matrix.rows))
                .map_err(PolyBackendError::GpuSubmission)?;
        claims.extend(cache);
        let (export, import) = if self.devices.len() > 1 {
            let (snapshots, export) = trace_native_claims(|| trapdoor.to_rns_snapshots())
                .map_err(PolyBackendError::GpuSubmission)?;
            let (_, import) = trace_native_claims(|| {
                let replica =
                    mxx_primitives::sampler::trapdoor::gpu::GpuDCRTTrapdoor::from_rns_snapshots(
                        &params, &snapshots,
                    );
                sampler.prepare_preimage_cache(&params, &replica, matrix.rows);
                replica
            })
            .map_err(PolyBackendError::GpuSubmission)?;
            (export, import)
        } else {
            (Vec::new(), Vec::new())
        };
        let plan = super::gpu_compiled::TrapdoorClaimPlan { sample: claims, export, import };
        self.trapdoor_plans.insert(key, plan.clone());
        Ok(plan)
    }

    /// Trace the exact claims of scalar polynomial value readback for `ty` in
    /// the requested domain, for both possible input formats (the consumed
    /// matrix's format is only known at execution).
    fn trace_polynomial_values(
        &mut self,
        ty: &ConcreteMatrixType,
        evaluation: bool,
        warm_up: bool,
    ) -> Result<Vec<GpuTracedClaim>, PolyBackendError> {
        let params = self.devices[0].1.parameters(ty)?.clone();
        let mut all = Vec::new();
        for input_ntt in [false, true] {
            let key = (
                params.modulus().to_string(),
                params.ring_dimension() as usize,
                evaluation,
                input_ntt,
            );
            if let Some(claims) = self.polynomial_value_plans.get(&key) {
                all.extend(claims.iter().cloned());
                continue;
            }
            if !warm_up {
                return Err(PolyBackendError::GpuSubmission(format!(
                    "explicit graph warmup required: missing trace_polynomial_values resource plan for {key:?}"
                )));
            }
            let mut probe = <GpuDCRTPolyMatrix as PolyMatrix>::zero(&params, 1, 1);
            if input_ntt {
                probe.ntt_all_in_place();
            }
            let device = &mut self.devices[0].1;
            let (values, claims) =
                trace_native_claims(|| device.polynomial_values(&probe, evaluation))
                    .map_err(PolyBackendError::GpuSubmission)?;
            values?;
            all.extend(claims.iter().cloned());
            self.polynomial_value_plans.insert(key, claims);
        }
        Ok(all)
    }

    /// Trace the exact claims of one preimage class: the destination's
    /// hard-cutoff plan, the staged target tile and one single-column candidate.
    /// Native layout queries specialize the traced claim order for other widths.
    fn trace_preimage_plan(
        &mut self,
        trapdoor_matrix: &ConcreteMatrixType,
        public: &ConcreteMatrixType,
        ty: &ConcreteMatrixType,
        bound: &num_bigint::BigUint,
        sigma: f64,
        gadget_base: &num_bigint::BigInt,
        digit_count: usize,
        warm_up: bool,
    ) -> Result<PreimageClaimPlan, PolyBackendError> {
        let key = PreimagePlanKey {
            modulus: ty.modulus.to_string(),
            ring_dimension: ty.ring_dimension,
            rows: ty.rows,
            columns: ty.columns,
            public_rows: public.rows,
            bound: bound.to_string(),
            sigma_bits: sigma.to_bits(),
            gadget_base: gadget_base.to_string(),
            digit_count,
        };
        if let Some(plan) = self.preimage_plans.get(&key) {
            return Ok(plan.clone());
        }
        if !warm_up {
            return Err(PolyBackendError::GpuSubmission(format!(
                "explicit graph warmup required: missing trace_preimage_plan resource plan for {key:?}"
            )));
        }
        let params = self.devices[0].1.parameters(ty)?.clone();
        let (public_matrix, trapdoor) =
            self.devices[0].1.sample_trapdoor(trapdoor_matrix, sigma, gadget_base, digit_count)?;
        let sampler = <GpuDCRTPolyTrapdoorSampler as PolyTrapdoorSampler>::new(&params, sigma);
        sampler.prepare_preimage_cache(&params, &trapdoor, public.rows);
        let (destination, destination_claims) = trace_native_claims(|| {
            GpuDCRTPolyTrapdoorSampler::preimage_destination(&params, ty.rows, 1, bound.clone())
        })
        .map_err(PolyBackendError::GpuSubmission)?;
        let mut destination = destination?;
        if destination_claims.first().map(|c| c.kind()) != Some(GpuPreparedSlotKind::CompactPayload)
        {
            return Err(PolyBackendError::GpuSubmission(
                "preimage destination trace does not start with its payload".into(),
            ));
        }
        let level = params.crt_depth() - 1;
        let bytes_per_poly = GpuDCRTPolyMatrix::cpu_staging_layout(
            &params,
            &GpuDCRTPolyMatrix::zero(&params, 1, 1).into_cpu_staging_bytes(),
        )
        .map_err(PolyBackendError::GpuCalibration)?
        .bytes_per_poly;
        let staging = GpuCpuStagingLayout {
            rows: public.rows,
            columns: 1,
            level,
            is_ntt: true,
            bytes_per_poly,
        }
        .zero_bytes(&params)
        .map_err(PolyBackendError::GpuCalibration)?;
        let target =
            PolyMatrixColumnData::<GpuFleetMatrix>::staged(&params, Arc::new(staging), 0, 1);
        let seed: [u8; 32] = rand::random();
        let (materialized, tile) = trace_native_claims(|| {
            super::gpu_compiled::materialize_preimage_tile(&params, &target, public.rows, 0, 1)
        })
        .map_err(PolyBackendError::GpuSubmission)?;
        let materialized = materialized.map_err(PolyBackendError::GpuSubmission)?;
        struct AttemptTrace([Vec<GpuTracedClaim>; 16]);
        impl mxx_primitives::sampler::trapdoor::gpu::GpuPreimageBatchResources for AttemptTrace {
            fn run<T>(
                &mut self,
                phase: mxx_primitives::sampler::trapdoor::gpu::GpuPreimageBatchPhase,
                jobs: usize,
                operation: impl FnOnce() -> Result<T, String>,
            ) -> Result<T, String> {
                debug_assert_eq!(jobs, 1, "discovery uses one representative per class");
                let (result, claims) = trace_native_claims(operation)?;
                self.0[phase as usize] = claims;
                result
            }
        }
        let mut attempt = AttemptTrace(Default::default());
        sampler
            .preimage_attempt_batch(
                &params,
                vec![mxx_primitives::sampler::trapdoor::gpu::GpuPreimageAttempt {
                    trapdoor: &trapdoor,
                    public: &public_matrix,
                    target: &materialized,
                    destination: &mut destination,
                    column_start: 0,
                    global_column_start: 0,
                    attempt: 0,
                    seed,
                }],
                &mut attempt,
            )
            .map_err(PolyBackendError::GpuSubmission)?;
        let plan = PreimageClaimPlan {
            destination: destination_claims[1..].to_vec(),
            tile,
            attempt: attempt.0,
            attempts: mxx_primitives::env::gpu_preimage_max_tile_attempts()
                .map_err(PolyBackendError::GpuSubmission)?,
            bytes_per_poly,
        };
        self.preimage_plans.insert(key, plan.clone());
        Ok(plan)
    }

    /// Provision the complete prepared inventory for `validated` and accept it
    /// through `prepare_memory`. Explicitly installed ledgers retain their
    /// caller-managed lifecycle; automatic inventories reuse free backing and
    /// add only missing demand. The caller asserts exclusive device observation.
    /// Explicit resource warmup uses `warm_up = true`: it discovers and caches
    /// the resource classes reachable from the graph and provisions no
    /// graph-sized backing, so the returned guard is absent. Reuse this backend
    /// for every warmed graph. Production always uses false: it verifies the
    /// discovered classes, then provisions the required production storage;
    /// cache misses fail before discovery trials.
    pub fn prepare_graph_admission(
        &mut self,
        validated: &ValidatedGraph,
        capture_trace: bool,
        inputs: &BTreeMap<String, crate::backend::RuntimeValue<Self>>,
        wave_bound: usize,
        warm_up: bool,
    ) -> Result<Option<Box<dyn std::any::Any>>, PolyBackendError> {
        if self.prepared_ledger.is_some() && !self.graph_prepared {
            return Ok(None);
        }
        let guard = GpuGraphAdmissionGuard::new(self.device_parameters())
            .map_err(PolyBackendError::GpuSubmission)?;
        // Caller-owned fragments remain distinct in the compiled runner, even
        // when they occupy the same device. Preserve their actual boundaries.
        let input_columns = validated
            .root_scope()
            .execution_order
            .par_iter()
            .enumerate()
            .filter_map(|(position, handle)| {
                let NodeKind::Input { name, artifact, .. } = handle.kind() else { return None };
                let wire =
                    WireRef { node: mxx_ir_core::types::NodeId(position as u64), port: Port(0) };
                let mut layout = GpuInventoryValue::default();
                let mut ty = &validated.root_scope().wire_types[&wire];
                while let ConcreteWireType::IndexedFamily { element, .. } = ty {
                    ty = element;
                }
                if let Some(ty) = ty.matrix_type() {
                    // A root artifact descriptor takes precedence over caller
                    // inputs in the executor. Such a value has no observed owner
                    // or fragment layout until its artifact is materialized.
                    if artifact.is_none() &&
                        let Some(value) = inputs.get(name)
                    {
                        layout.include_input(value, ty.columns);
                    }
                    if layout.cuts.is_empty() {
                        layout = GpuInventoryValue::new(vec![0, ty.columns]);
                    }
                }
                layout.lazy |= artifact.is_some();
                Some((wire, layout))
            })
            .collect::<BTreeMap<_, _>>();
        let operations = Self::lower_inventory_scope(
            &self.devices[0].1,
            validated,
            &FrozenGraphScopeId::Root,
            &validated.bindings,
        )?;
        if warm_up {
            // Warmup owns class discovery only. The scratch column envelope is
            // a production storage decision, not a class property, so discover
            // at one column; graph-sized backing stays out of warmup entirely.
            self.graph_admission_demand(
                validated,
                &FrozenGraphScopeId::Root,
                &validated.bindings,
                Some(&operations),
                capture_trace,
                Some(&input_columns),
                1,
                1,
                true,
            )?;
            return Ok(None);
        }
        let parameters = self.device_parameters();
        let available = parameters
            .par_iter()
            .map(|params| {
                let memory = mxx_primitives::poly::dcrt::gpu::gpu_device_memory_usage(
                    params.device_ids()[0],
                )
                .map_err(PolyBackendError::GpuCalibration)?;
                Ok(params.vram_budget_bytes().saturating_sub(memory.resident))
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        let (wave, _, resources) = self.graph_resource_demand(
            validated,
            capture_trace,
            &input_columns,
            wave_bound,
            &available,
        )?;
        let demand = resources.contexts;
        let value_layouts = resources.value_layouts.into_iter().next().unwrap();
        self.graph_wave = wave;
        self.prepare_graph_storage(demand)?;
        self.graph_prepared = true;
        let operations = Arc::new(AdmittedScopeOperations {
            scope: FrozenGraphScopeId::Root,
            instances: vec![operations],
            value_layouts: vec![
                value_layouts
                    .into_iter()
                    .filter_map(|(wire, layout)| Some((wire, layout.fragments?)))
                    .collect(),
            ],
            input_layouts: input_columns
                .values()
                .filter_map(|layout| Some((layout.owner?, layout.fragments.clone()?)))
                .collect(),
        });
        self.admitted_scope_operations.retain(|scope| scope.strong_count() != 0);
        self.admitted_scope_operations.push(Arc::downgrade(&operations));
        Ok(Some(Box::new((guard, operations))))
    }

    /// CPU-only setup selection shared by production and hypothetical estimation.
    /// Inputs describe the scenario's root owners; available bytes are one frozen
    /// budget per configured device. Required primitive templates must already be
    /// warmed. This neither provisions backing nor runs discovery or measurements.
    pub fn graph_resource_demand(
        &mut self,
        validated: &ValidatedGraph,
        capture_trace: bool,
        input_columns: &BTreeMap<WireRef, GpuInventoryValue>,
        wave_bound: usize,
        available: &[usize],
    ) -> Result<(usize, usize, GpuScopeResources), PolyBackendError> {
        let operations = Self::lower_inventory_scope(
            &self.devices[0].1,
            validated,
            &FrozenGraphScopeId::Root,
            &validated.bindings,
        )?;
        let maximum = validated
            .root_scope()
            .wire_types
            .values()
            .filter_map(ConcreteWireType::matrix_type)
            .map(|ty| ty.columns)
            .max()
            .unwrap_or(1)
            .max(1);
        let inventory = self
            .prepared_ledger
            .as_ref()
            .map(|ledger| {
                ledger
                    .prepared_inventory()
                    .map(|(device, storage)| (device, storage.clone()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        // Production backing of a parallel loop covers one complete configured
        // wave: `wave_bound` bodies may run at once, so one body's peak slots
        // must exist `wave_bound` times. The bound is the configured concurrency
        // limit, never the loop count or the number of graph uses, so repeating
        // a loop or reusing an unchanged class never multiplies the backing.
        //
        // The wave and the scratch-column cap are searched together: every wave
        // from the configured bound down to one is tried in order, and inside
        // each wave the complete cap list `maximum, ceil(maximum / 2), ..., 1`.
        // The first fitting pair wins, so a wave that fits only after the scratch
        // envelope shrinks is still preferred over a narrower wave at a wider
        // envelope. Shrinking the cap only after rejecting every wave would
        // wrongly provision W = 1 for a graph that can run more bodies at a
        // smaller legal width.
        let (wave, cap, demand, value_layouts) = match select_wave_and_width(
            wave_bound.max(1),
            maximum,
            |wave, width| {
                let demand = self.graph_admission_demand(
                    validated,
                    &FrozenGraphScopeId::Root,
                    &validated.bindings,
                    Some(&operations),
                    capture_trace,
                    Some(&input_columns),
                    width,
                    wave,
                    false,
                )?;
                let requested = self.prepared_demand_bytes(&demand.0, &inventory)?;
                let fits = requested
                    .iter()
                    .zip(available)
                    .all(|(requested, available)| requested <= available);
                Ok::<_, PolyBackendError>(fits.then_some(demand))
            },
        )? {
            Some((wave, scratch_columns, demand)) => {
                tracing::debug!(
                    wave,
                    scratch_columns,
                    "prepared graph wave search accepted its widest fitting pair"
                );
                (wave, scratch_columns, demand.0, demand.2)
            }
            None => {
                // No pair fits: report the smallest candidate's exact demand, the
                // one-column single-body prefix, so the error names a real bound.
                let demand = self
                    .graph_admission_demand(
                        validated,
                        &FrozenGraphScopeId::Root,
                        &validated.bindings,
                        Some(&operations),
                        capture_trace,
                        Some(&input_columns),
                        1,
                        1,
                        false,
                    )?
                    .0;
                let requested = self.prepared_demand_bytes(&demand, &inventory)?;
                return Err(PolyBackendError::GpuSubmission(format!(
                    "retained graph owners and one-column scratch exceed the setup memory budget: requested bytes per device {requested:?}, available bytes per device {available:?}"
                )));
            }
        };
        Ok((wave, cap, GpuScopeResources { contexts: demand, value_layouts: vec![value_layouts] }))
    }

    /// Setup-time native byte demand of one prepared demand snapshot on every
    /// device, from the same layout queries the allocation uses. It measures the
    /// demand, allocates nothing and reserves nothing.
    fn prepared_demand_bytes(
        &self,
        demand: &BTreeMap<(String, usize), (ConcreteMatrixType, GpuContextDemand)>,
        inventory: &[(usize, Arc<GpuPreparedStorage>)],
    ) -> Result<Vec<usize>, PolyBackendError> {
        self.devices
            .par_iter()
            .enumerate()
            .map(|(device, (_, backend))| -> Result<usize, PolyBackendError> {
                let mut bytes = 0usize;
                let mut used = HashSet::new();
                for (ty, context) in demand.values() {
                    let params = backend.parameters(ty)?;
                    let (matrices, layouts) =
                        context.missing(params, inventory, device, &mut used)?;
                    let claims = matrices
                        .into_iter()
                        .chain(layouts.into_iter().map(GpuTracedClaim::workspace))
                        .collect::<Vec<_>>();
                    let slots =
                        mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSlotSnapshot::plan(
                            params, &claims,
                        )
                        .map_err(PolyBackendError::GpuCalibration)?;
                    for slot in slots {
                        let identity = slot.identity();
                        if !matches!(
                            identity.kind(),
                            GpuPreparedSlotKind::PinnedHost |
                                GpuPreparedSlotKind::CompletionEvent |
                                GpuPreparedSlotKind::SubmissionStream
                        ) {
                            bytes = bytes
                                .checked_add(identity.requested_backing_bytes())
                                .ok_or(PolyBackendError::InvalidInteger)?;
                        }
                    }
                }
                Ok(bytes)
            })
            .collect()
    }

    fn prepare_graph_storage(
        &mut self,
        demand: BTreeMap<(String, usize), (ConcreteMatrixType, GpuContextDemand)>,
    ) -> Result<(), PolyBackendError> {
        // Build one storage per context per device before any domain closes.
        let mut prepared = self
            .prepared_ledger
            .as_ref()
            .map(|ledger| {
                ledger
                    .prepared_inventory()
                    .map(|(device, storage)| (device, storage.clone()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let original_count = prepared.len();
        let mut used = HashSet::new();
        for (device, (_, backend)) in self.devices.iter().enumerate() {
            for (ty, context) in demand.values() {
                let params = backend.parameters(ty)?;
                let (missing_matrices, missing_layouts) =
                    context.missing(params, &prepared, device, &mut used)?;
                if missing_matrices.is_empty() && missing_layouts.is_empty() {
                    continue;
                }
                // Production backing is allocated, not zeroed: these slots are
                // full-overwrite destinations claimed by dispatch, so a
                // placeholder zero kernel would be disposable work. The shapes
                // are appended uninitialized inside the primitives storage
                // constructor and never exist as a readable matrix here. A
                // layouts-only context still needs one matrix slot, and that
                // fallback is allocated the same way.
                let storage = GpuPreparedStorage::new(
                    Some(params),
                    Vec::new(),
                    Some(&missing_matrices),
                    Some(&missing_layouts),
                )
                .map_err(PolyBackendError::GpuSubmission)?;
                prepared.push((device, Arc::new(storage)));
            }
        }
        if prepared.is_empty() {
            return Ok(());
        }
        if original_count != 0 && prepared.len() == original_count {
            // Re-enter the graph seal without adding backing or imposing a new
            // initial-budget check on already accepted capacity. Output leases
            // and their asynchronous reader/release edges remain untouched.
            for device in 0..self.devices.len() {
                let stores = prepared
                    .iter()
                    .filter(|(owner, _)| *owner == device)
                    .map(|(_, storage)| storage.as_ref())
                    .collect::<Vec<_>>();
                GpuPreparedStorage::finish_setup(&stores)
                    .map_err(PolyBackendError::GpuSubmission)?;
            }
            return Ok(());
        }
        let previous = self.prepared_ledger.take();
        self.prepared_required = false;
        match self.prepare_memory(prepared, true) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.prepared_required = previous.is_some();
                self.prepared_ledger = previous;
                Err(error)
            }
        }
    }
    pub(super) fn admit_loop_wave(
        &mut self,
        request: &crate::executor::WaveAdmissionRequest<'_, Self>,
    ) -> Result<crate::executor::WaveAdmission, PolyBackendError> {
        use crate::{backend::RuntimeValue, gpu_memory::GpuPreparedAllocationRequirement};
        let limit = request.caller_cap().min(request.remaining()).min(u16::MAX as usize);
        let bindings = (request.next_index()..request.next_index() + limit)
            .into_par_iter()
            .map(|index| {
                request
                    .body_bindings(index)
                    .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let backend = &self.devices[0].1;
        // Each body's metadata is independent and bounded by the caller cap.
        // No payload loading, native allocation, or sampling occurs here.
        let metadata = bindings
            .par_iter()
            .enumerate()
            .map(|(offset, env)| {
                let operations = Self::lower_inventory_scope(
                    backend,
                    request.validated(),
                    request.child_id(),
                    env,
                )?;
                let mut columns = BTreeMap::new();
                for (index, input) in request
                    .selected_inputs(request.next_index() + offset)
                    .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?
                    .into_iter()
                    .enumerate()
                {
                    let metadata = &operations[input.wire.node.0 as usize].0;
                    if let Some(ty) = metadata.outputs()[input.wire.port.0 as usize].matrix_type() {
                        let mut layout = GpuInventoryValue::default();
                        layout.include_input(input.value, ty.columns);
                        if matches!(
                            input.value,
                            RuntimeValue::LazyArtifact { .. } | RuntimeValue::StagedArtifact { .. }
                        ) && matches!(request.input_modes()[index], LoopInputMode::Broadcast) &&
                            matches!(
                                metadata.outputs()[input.wire.port.0 as usize],
                                ConcreteWireType::Matrix(_)
                            )
                        {
                            // The executor imports this parent argument once, before
                            // binding any sibling. Account for that import below.
                            layout.broadcast = Some(request.arguments()[index]);
                            layout.borrowed = true;
                            layout.lazy = false;
                        }
                        columns.insert(input.wire, layout);
                    }
                }
                let maximum = operations
                    .iter()
                    .flat_map(|(operation, _)| {
                        operation.arguments().iter().chain(operation.outputs())
                    })
                    .filter_map(|ty| ty.matrix_type().map(|ty| ty.columns))
                    .max()
                    .unwrap_or(1)
                    .max(1);
                Ok((columns, maximum, operations))
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        let instances = bindings
            .iter()
            .zip(&metadata)
            .map(|(bindings, (columns, _, operations))| (bindings, columns, operations.as_slice()))
            .collect::<Vec<_>>();
        'retry: loop {
            let inventory = self
                .prepared_ledger
                .as_ref()
                .expect("prepared ledger")
                .prepared_inventory()
                .collect::<Vec<_>>();
            let pending = inventory
                .par_iter()
                .map(|(_, storage)| {
                    storage.poll_releases(&[]).map(|indices| {
                        indices
                            .into_iter()
                            .map(|index| storage.slot_identity(index).unwrap().slot_id())
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(PolyBackendError::GpuSubmission)?
                .into_iter()
                .flatten()
                .collect::<HashSet<_>>();
            // Inspect native ownership once per search, after reconciling
            // releases. Every W/C candidate fits this same CPU snapshot; the
            // native transaction rechecks it and retries if ownership changed.
            let snapshots = inventory
                .par_iter()
                .map(|(device, storage)| storage.snapshot().map(|slots| (*device, storage, slots)))
                .collect::<Result<Vec<_>, _>>()
                .map_err(PolyBackendError::GpuSubmission)?;
            let storage_by_id = inventory
                .iter()
                .map(|(device, storage)| (storage.identity(), (*device, storage)))
                .collect::<HashMap<_, _>>();
            for wave in (1..=limit).rev() {
                let mut cap = metadata[..wave]
                    .iter()
                    .map(|(_, maximum, _)| *maximum)
                    .max()
                    .unwrap_or(1)
                    .min(self.prepared_ledger.as_ref().expect("prepared ledger").column_cap());
                loop {
                    let GpuScopeResources { contexts, value_layouts } = self.wave_resource_demand(
                        request.validated(),
                        request.child_id(),
                        &instances[..wave],
                        cap,
                        request.capture_trace(),
                    )?;
                    let claims = contexts
                        .into_iter()
                        .map(|(key, (ty, contexts))| {
                            let params = self.devices[0].1.parameters(&ty)?;
                            let claims = contexts.claims(params);
                            Ok((key, (ty, claims)))
                        })
                        .collect::<Result<BTreeMap<_, (_, Vec<_>)>, PolyBackendError>>()?;
                    let select = |excluded: &HashSet<u64>| -> Result<_, PolyBackendError> {
                        let mut selected = BTreeMap::<
                            u64,
                            (
                                usize,
                                Arc<GpuPreparedStorage>,
                                Vec<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedRequest>,
                            ),
                        >::new();
                        for (_, (ty, claims)) in &claims {
                            for (device, (_, backend)) in self.devices.iter().enumerate() {
                                let params = backend.parameters(ty)?;
                                let slots = snapshots
                                    .iter()
                                    .filter(|(owner, storage, _)| {
                                        *owner == device && storage.matches_parameters(params)
                                    })
                                    .flat_map(|(_, _, slots)| slots)
                                    .map(|slot| {
                                        let identity = slot.identity();
                                        let deferred_transfer = pending
                                            .contains(&identity.slot_id()) &&
                                            matches!(
                                                identity.kind(),
                                                GpuPreparedSlotKind::PinnedHost |
                                                    GpuPreparedSlotKind::CompletionEvent
                                            );
                                        let eligible = !excluded.contains(&identity.slot_id()) &&
                                            (slot.is_available() || deferred_transfer);
                                        (*slot, eligible)
                                    })
                                    .collect::<Vec<_>>();
                                let Some(assignment) =
                                    GpuPreparedSlotSnapshot::assign(params, &slots, claims)
                                        .map_err(PolyBackendError::GpuSubmission)?
                                        .into_iter()
                                        .collect::<Option<Vec<_>>>()
                                else {
                                    return Ok(None);
                                };
                                for claim in assignment {
                                    let (device, storage) = storage_by_id[&claim.slot_key().0];
                                    selected
                                        .entry(storage.identity())
                                        .or_insert_with(|| (device, storage.clone(), Vec::new()))
                                        .2
                                        .push(claim);
                                }
                            }
                        }
                        Ok(Some(selected))
                    };
                    if let Some(selected) = select(&pending)? {
                        let requirements = selected
                            .values()
                            .map(|(device, storage, requests)| GpuPreparedAllocationRequirement {
                                device: *device,
                                storage,
                                requests,
                            })
                            .collect::<Vec<_>>();
                        let region = match self
                            .prepared_ledger
                            .as_mut()
                            .expect("prepared ledger")
                            .reserve_region(&requirements, cap)
                        {
                            Ok(region) => region,
                            Err(crate::gpu_memory::GpuAdmissionError::StaleFit { .. }) => {
                                // The native transaction rolled back every partial
                                // region. Reinspect from the largest candidate; no
                                // input or production command has been submitted.
                                continue 'retry;
                            }
                            Err(error) => {
                                return Err(PolyBackendError::GpuSubmission(error.to_string()))
                            }
                        };
                        let input_layouts = metadata[..wave]
                            .iter()
                            .flat_map(|(columns, _, _)| columns.values())
                            .filter_map(|layout| Some((layout.owner?, layout.fragments.clone()?)))
                            .collect();
                        let operations = Arc::new(AdmittedScopeOperations {
                            scope: request.child_id().clone(),
                            input_layouts,
                            value_layouts: value_layouts
                                .into_iter()
                                .map(|layouts| {
                                    layouts
                                        .into_iter()
                                        .filter_map(|(wire, layout)| {
                                            Some((wire, layout.fragments?))
                                        })
                                        .collect()
                                })
                                .collect(),
                            instances: metadata
                                .into_iter()
                                .take(wave)
                                .map(|(_, _, operations)| operations)
                                .collect(),
                        });
                        // Weak access is only an index into owned active waves.
                        // Completed/rejected candidates never become a cache.
                        self.admitted_scope_operations.retain(|scope| scope.strong_count() != 0);
                        self.admitted_scope_operations.push(Arc::downgrade(&operations));
                        let mut admission = request.admission_of_size(wave);
                        admission.reservation = Some(Box::new((region, operations)));
                        if let Some(sink) = &mut self.admitted_measurement_sink {
                            sink(crate::gpu_measurement::GpuAdmittedMeasurement::Admission {
                                scope: request.child_id().clone(),
                                instances: wave,
                                column_cap: cap,
                            });
                        }
                        tracing::debug!(wave, cap, "live wave region admitted");
                        return Ok(admission);
                    }
                    if wave == 1 && cap == 1 && !pending.is_empty() {
                        if let Some(selected) = select(&HashSet::new())? {
                            // No immediately eligible candidate fits. Only
                            // releases in a feasible minimum candidate can help
                            // progress; do not wait for unrelated retained owners.
                            selected
                                .values()
                                .collect::<Vec<_>>()
                                .par_iter()
                                .map(|(_, storage, claims)| {
                                    let slots = claims
                                        .iter()
                                        .filter(|claim| pending.contains(&claim.slot_key().1))
                                        .map(|claim| claim.slot_key().2)
                                        .collect::<Vec<_>>();
                                    storage.poll_releases(&slots).map(|_| ())
                                })
                                .collect::<Result<Vec<_>, _>>()
                                .map_err(PolyBackendError::GpuSubmission)?;
                            continue 'retry;
                        }
                    }
                    if cap == 1 {
                        break;
                    }
                    cap = cap.div_ceil(2);
                }
            }
            return Err(PolyBackendError::GpuSubmission(
                "one live body with one-column scratch does not fit the available typed inventory"
                    .into(),
            ));
        }
    }

    fn lower_inventory_scope(
        backend: &DeviceBackend,
        validated: &ValidatedGraph,
        scope_id: &FrozenGraphScopeId,
        bindings: &ParamEnv,
    ) -> Result<Vec<InventoryOperation>, PolyBackendError> {
        validated
            .scope(scope_id)
            .expect("validated scope")
            .execution_order
            .par_iter()
            .enumerate()
            .map(|(index, _)| {
                let node = crate::gpu_invocation::GpuNodeOperation::new(
                    validated,
                    scope_id,
                    mxx_ir_core::types::NodeId(index as u64),
                    bindings,
                )
                .map_err(PolyBackendError::GpuSubmission)?;
                let operation = node
                    .outputs()
                    .first()
                    .and_then(ConcreteWireType::matrix_type)
                    .map(|ty| {
                        PreparedMatrixOperation::from_ir(
                            node.kind(),
                            node.arguments(),
                            node.outputs(),
                            bindings,
                            backend.parameters(ty)?,
                        )
                    })
                    .transpose()?
                    .flatten();
                Ok((node, operation))
            })
            .collect()
    }

    /// Derive a complete sibling candidate with the same resource lowering as
    /// live admission, including shared cold-broadcast imports. Inputs contain
    /// explicit CPU layout/owner metadata. The slice is the candidate's ordered
    /// instances; its length supplies the wave size. Discovery is disabled.
    /// No backing, reservation, input materialization or GPU work is created.
    /// Fit the returned claims against the caller's complete eligible inventory.
    pub fn scope_resource_demand(
        &mut self,
        validated: &ValidatedGraph,
        scope: &FrozenGraphScopeId,
        instances: &[(ParamEnv, BTreeMap<WireRef, GpuInventoryValue>)],
        scratch_columns: usize,
        capture_trace: bool,
    ) -> Result<GpuScopeResources, PolyBackendError> {
        let operations = instances
            .par_iter()
            .map(|(bindings, _)| {
                Self::lower_inventory_scope(&self.devices[0].1, validated, scope, bindings)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let instances = instances
            .iter()
            .zip(&operations)
            .map(|((bindings, inputs), operations)| (bindings, inputs, operations.as_slice()))
            .collect::<Vec<_>>();
        self.wave_resource_demand(validated, scope, &instances, scratch_columns, capture_trace)
    }

    fn wave_resource_demand(
        &mut self,
        validated: &ValidatedGraph,
        scope: &FrozenGraphScopeId,
        instances: &[(&ParamEnv, &BTreeMap<WireRef, GpuInventoryValue>, &[InventoryOperation])],
        cap: usize,
        capture_trace: bool,
    ) -> Result<GpuScopeResources, PolyBackendError> {
        let wave = instances.len();
        let mut contexts =
            BTreeMap::<(String, usize), (ConcreteMatrixType, Vec<GpuContextDemand>)>::new();
        let mut value_layouts = Vec::with_capacity(wave);
        for (index, (bindings, columns, operations)) in instances.iter().enumerate() {
            let (demand, _, layouts) = self.graph_admission_demand(
                validated,
                scope,
                bindings,
                Some(operations),
                capture_trace,
                Some(columns),
                cap,
                wave,
                false,
            )?;
            value_layouts.push(layouts);
            for (key, (ty, demand)) in demand {
                contexts
                    .entry(key)
                    .or_insert_with(|| (ty, vec![GpuContextDemand::default(); wave]))
                    .1[index] = demand;
            }
        }
        // Cold broadcasts are retained placement owners shared by
        // every sibling. Reserve their import resources once per parent
        // wire, independently of the chosen body count.
        let mut broadcasts = HashSet::new();
        if let Some((_, columns, operations)) = instances.first() {
            for (wire, layout) in *columns {
                let Some(parent) = layout.broadcast else { continue };
                if !broadcasts.insert(parent) {
                    continue;
                }
                let output = &operations[wire.node.0 as usize].0.outputs()[wire.port.0 as usize];
                let ConcreteWireType::Matrix(ty) = output else { unreachable!() };
                let params = self.devices[0].1.parameters(ty)?;
                let (matrices, layouts) = import_inventory(params, output, cap)?;
                let import = &mut contexts
                    .entry((ty.modulus.to_string(), ty.ring_dimension))
                    .or_insert_with(|| (ty.clone(), vec![GpuContextDemand::default(); wave]))
                    .1[0];
                let placement_owners = columns
                    .iter()
                    .filter(|(_, layout)| layout.broadcast == Some(parent))
                    .map(|(wire, _)| ValueOrigin::Direct {
                        instance: None,
                        wire: *wire,
                        members: Vec::new(),
                    })
                    .collect::<Vec<_>>();
                let matrix_start = import.matrix_demands.len();
                let workspace_start = import.workspace_demands.len();
                import.matrix(0, usize::MAX, ty.rows, ty.columns);
                for (rows, columns) in matrices {
                    import.matrix(0, usize::MAX, rows, columns);
                }
                for layout in layouts {
                    import.workspace(0, usize::MAX, layout, false);
                }
                for matrix in &mut import.matrix_demands[matrix_start..] {
                    matrix.origins = ResourceOrigins::Placement;
                    matrix.owners = placement_owners.clone();
                }
                for workspace in &mut import.workspace_demands[workspace_start..] {
                    workspace.origins = ResourceOrigins::Placement;
                    workspace.owners = placement_owners.clone();
                }
            }
        }
        let contexts = contexts
            .into_iter()
            .map(|(key, (ty, demands))| (key, (ty, GpuContextDemand::simultaneous(demands))))
            .collect();
        Ok(GpuScopeResources { contexts, value_layouts })
    }

    /// Size candidates using native layout queries, before backing allocation.
    /// Diagnostic residency chooses a range cap; it is not admission evidence.
    /// The completed inventory is still accepted against coherent native receipts.
    fn graph_admission_demand(
        &mut self,
        validated: &ValidatedGraph,
        scope_id: &FrozenGraphScopeId,
        bindings: &ParamEnv,
        operations: Option<&[InventoryOperation]>,
        capture_trace: bool,
        input_columns: Option<&BTreeMap<WireRef, GpuInventoryValue>>,
        scratch_columns: usize,
        wave: usize,
        warm_up: bool,
    ) -> Result<
        (
            BTreeMap<(String, usize), (ConcreteMatrixType, GpuContextDemand)>,
            Vec<GpuInventoryValue>,
            BTreeMap<WireRef, GpuInventoryValue>,
        ),
        PolyBackendError,
    > {
        let scope = validated.source.scope(scope_id).ok_or_else(|| {
            PolyBackendError::GpuSubmission("graph has no requested scope".into())
        })?;
        let checked = validated.scope(scope_id).unwrap();
        let resolved;
        let operations = if let Some(operations) = operations {
            operations
        } else {
            resolved =
                Self::lower_inventory_scope(&self.devices[0].1, validated, scope_id, bindings)?;
            &resolved
        };

        let optimizer = if *scope_id == FrozenGraphScopeId::Root {
            crate::executor::gpu_plan::inventory_plan(validated, capture_trace)
        } else {
            crate::executor::gpu_plan::InventoryPlan::default()
        };
        let mut column_layouts = BTreeMap::<WireRef, GpuInventoryValue>::new();
        if let Some(inputs) = input_columns {
            column_layouts.extend(inputs.iter().map(|(wire, cuts)| (*wire, cuts.clone())));
        }
        let liveness = crate::executor::gpu_plan::owner_liveness(
            validated,
            scope_id,
            bindings,
            &optimizer,
            capture_trace,
        )
        .map_err(|error| match error {
            crate::executor::gpu_plan::LivenessError::InvalidInteger => {
                PolyBackendError::InvalidInteger
            }
        })?;

        let unsupported = |kind: &NodeKind| {
            PolyBackendError::GpuSubmission(format!(
                "prepared GPU admission has no compiled runner for {kind:?}"
            ))
        };
        // Demand per parameter context, keyed by the context's matrix type
        // identity (modulus and ring dimension); replicated on every device.
        let mut demand = BTreeMap::<(String, usize), (ConcreteMatrixType, GpuContextDemand)>::new();
        let position = std::cell::Cell::new(0usize);
        let until = std::cell::Cell::new(0usize);
        let shared_layout = std::cell::Cell::new(false);
        let owners = std::cell::RefCell::new(Vec::<(WireRef, Vec<usize>)>::new());
        let add_matrix = |demand: &mut BTreeMap<_, (ConcreteMatrixType, GpuContextDemand)>,
                          ty: &ConcreteMatrixType,
                          rows: usize,
                          columns: usize| {
            let context = &mut demand
                .entry((ty.modulus.to_string(), ty.ring_dimension))
                .or_insert_with(|| (ty.clone(), GpuContextDemand::default()))
                .1;
            context.matrix(position.get(), until.get(), rows, columns);
            context.matrix_demands.last_mut().unwrap().owners = owners
                .borrow()
                .iter()
                .map(|(wire, path)| ValueOrigin::Direct {
                    instance: Some(0),
                    wire: *wire,
                    members: path.clone(),
                })
                .collect();
        };
        let add_layouts = |demand: &mut BTreeMap<_, (ConcreteMatrixType, GpuContextDemand)>,
                           ty: &ConcreteMatrixType,
                           layouts: Vec<GpuPreparedWorkspaceLayout>| {
            let context = &mut demand
                .entry((ty.modulus.to_string(), ty.ring_dimension))
                .or_insert_with(|| (ty.clone(), GpuContextDemand::default()))
                .1;
            for layout in layouts {
                context.workspace(position.get(), until.get(), layout, shared_layout.get());
                context.workspace_demands.last_mut().unwrap().owners = owners
                    .borrow()
                    .iter()
                    .map(|(wire, path)| ValueOrigin::Direct {
                        instance: Some(0),
                        wire: *wire,
                        members: path.clone(),
                    })
                    .collect();
            }
        };
        let parameters =
            |backend: &Self, ty: &ConcreteMatrixType| backend.devices[0].1.parameters(ty).cloned();
        for (node_position, handle) in checked.execution_order.iter().enumerate() {
            position.set(node_position);
            until.set(node_position);
            owners.borrow_mut().clear();
            let id = mxx_ir_core::types::NodeId(node_position as u64);
            let retained_until = |port: usize| liveness.output_end(node_position, port);
            let arguments = scope
                .arguments(handle)
                .ok_or_else(|| PolyBackendError::GpuSubmission("missing node arguments".into()))?;
            let (operation, prepared) = &operations[node_position];
            let argument_types = operation.arguments();
            let output_types = operation.outputs();
            let kind = handle.kind();
            // Stored fragments survive column-wise operations. Each interval
            // needs its own retained destination, even on the same device.
            for (port, output) in output_types.iter().enumerate() {
                let wire = WireRef { node: id, port: Port(port as u32) };
                if let Some(source) = liveness.alias_source(wire) {
                    if let Some(boundaries) = column_layouts.get(source).cloned() {
                        column_layouts.insert(wire, boundaries);
                        continue;
                    }
                }
                let mut leaf = output;
                while let ConcreteWireType::IndexedFamily { element, .. } = leaf {
                    leaf = element;
                }
                let Some(ty) = leaf.matrix_type() else { continue };
                if matches!(kind, NodeKind::Input { .. }) && column_layouts.contains_key(&wire) {
                    continue;
                }
                // The production operation owns column projection. Structural
                // values only identify possible inputs and keep alias alternatives.
                let source_indices = match &prepared {
                    Some(
                        PreparedMatrixOperation::Transpose | PreparedMatrixOperation::Tensor { .. },
                    ) => Vec::new(),
                    Some(operation) if operation.fresh_type().is_some() => Vec::new(),
                    Some(PreparedMatrixOperation::ConcatColumns { .. }) => {
                        (0..arguments.len()).collect()
                    }
                    Some(PreparedMatrixOperation::Multiply { scales_left }) => {
                        vec![usize::from(!scales_left)]
                    }
                    Some(PreparedMatrixOperation::MultiplyCompact { .. }) => vec![1],
                    Some(PreparedMatrixOperation::Slice { .. }) => vec![0],
                    None if matches!(
                        kind,
                        NodeKind::Input { .. } |
                            NodeKind::ConstantMatrix { .. } |
                            NodeKind::UniformResidueSample { .. } |
                            NodeKind::UniformIntervalSample { .. } |
                            NodeKind::GaussianSample { .. } |
                            NodeKind::HashSample { .. } |
                            NodeKind::PolynomialFromValues { .. } |
                            NodeKind::TrapdoorSample { .. } |
                            NodeKind::PreimageSample { .. }
                    ) =>
                    {
                        Vec::new()
                    }
                    _ => argument_types
                        .iter()
                        .enumerate()
                        .filter_map(|(index, argument)| {
                            let mut input = argument;
                            while let ConcreteWireType::IndexedFamily { element, .. } = input {
                                input = element;
                            }
                            input
                                .matrix_type()
                                .is_some_and(|input| input.columns == ty.columns)
                                .then_some(index)
                        })
                        .collect(),
                };
                let sources = source_indices
                    .iter()
                    .filter_map(|index| column_layouts.get(&arguments[*index]))
                    .collect::<Vec<_>>();
                let cuts = if let Some(operation) = &prepared &&
                    !matches!(operation, PreparedMatrixOperation::Accumulate { .. })
                {
                    let inputs = source_indices
                        .iter()
                        .map(|index| {
                            column_layouts
                                .get(&arguments[*index])
                                .map(|layout| layout.cuts.clone())
                                .unwrap_or_default()
                        })
                        .collect::<Vec<_>>();
                    operation.column_boundaries(&inputs, ty.columns)
                } else {
                    // Structural alternatives and fused accumulations summarize
                    // all possible member boundaries; no concrete input order
                    // has been chosen at this graph-level boundary yet.
                    std::iter::once(0)
                        .chain(std::iter::once(ty.columns))
                        .chain(sources.iter().flat_map(|layout| layout.cuts.iter().copied()))
                        .collect()
                };
                let mut layout = GpuInventoryValue::new(cuts);
                if matches!(kind, NodeKind::FamilyPack { .. } | NodeKind::Select { .. }) &&
                    !sources.is_empty()
                {
                    // A selection takes one partition, rather than refining
                    // every member together. Retain rank-wise peak capacities.
                    layout = GpuInventoryValue::default();
                    for source in sources {
                        layout.include_alternative(source.clone());
                    }
                } else if sources.iter().any(|source| source.alternatives) {
                    layout.alternatives = true;
                    let limit = layout.cuts.len().saturating_sub(1);
                    let mut largest = std::collections::BinaryHeap::with_capacity(limit);
                    // Bound temporary storage as well as the final inventory:
                    // a many-input refinement never collects every capacity list.
                    for &width in sources.iter().flat_map(|source| &source.capacities) {
                        if largest.len() < limit {
                            largest.push(std::cmp::Reverse(width));
                        } else if let Some(mut smallest) = largest.peek_mut() {
                            if width > smallest.0 {
                                *smallest = std::cmp::Reverse(width);
                            }
                        }
                    }
                    layout.capacities = largest.into_iter().map(|width| width.0).collect();
                    layout.capacities.sort_unstable_by(|a, b| b.cmp(a));
                    // Each common-refinement interval starts at an operand
                    // interval's left endpoint, so distinct operand intervals
                    // contain them. A slice can only shorten those intervals.
                    // The kth largest nonoverlapping output is also <= C/k.
                    // Together these bounds avoid retaining every family layout.
                    for (index, width) in layout.capacities.iter_mut().enumerate() {
                        *width = (*width).min(ty.columns / (index + 1));
                    }
                }
                layout.lazy = match kind {
                    NodeKind::Input { artifact, .. } => artifact.is_some(),
                    NodeKind::FamilyPack { .. } => arguments
                        .iter()
                        .any(|wire| column_layouts.get(wire).is_some_and(|source| source.lazy)),
                    NodeKind::FamilyGetStatic { .. } => {
                        column_layouts.get(&arguments[0]).is_some_and(|source| source.lazy)
                    }
                    _ => false,
                };
                layout.packed_family = matches!(kind, NodeKind::FamilyPack { .. }) ||
                    (matches!(
                        kind,
                        NodeKind::FamilyGetStatic { .. } |
                            NodeKind::FamilyGetDynamic |
                            NodeKind::Select { .. }
                    ) && matches!(output, ConcreteWireType::IndexedFamily { .. }) &&
                        arguments.iter().any(|wire| {
                            column_layouts.get(wire).is_some_and(|source| source.packed_family)
                        }));
                if matches!(output, ConcreteWireType::IndexedFamily { .. }) &&
                    matches!(kind, NodeKind::FamilyGetDynamic | NodeKind::Select { .. })
                {
                    layout.lazy |= arguments
                        .iter()
                        .any(|wire| column_layouts.get(wire).is_some_and(|source| source.lazy));
                }
                // Exact inherited placement uses the same ordered source-range
                // selection as native preflight. Each fragment retains its own
                // device/context and format, including mixed-format inputs.
                if !layout.lazy &&
                    !layout.alternatives &&
                    matches!(output, ConcreteWireType::Matrix(_)) &&
                    let Some(prepared) = prepared &&
                    !source_indices.is_empty() &&
                    !matches!(prepared, PreparedMatrixOperation::Accumulate { .. })
                {
                    let inputs = source_indices
                        .iter()
                        .map(|&index| {
                            let ty = argument_types[index].matrix_type()?;
                            let source = column_layouts.get(&arguments[index])?;
                            let fragments = source.fragments.as_deref()?;
                            (!source.lazy && !source.alternatives && !fragments.is_empty())
                                .then_some((ty.columns, fragments))
                        })
                        .collect::<Option<Vec<_>>>();
                    if let Some(inputs) = inputs {
                        let ranges = prepared.inherited_output_ranges(&inputs, ty.columns)?;
                        let fragments = ranges
                            .into_iter()
                            .map(|range| {
                                let source = inputs[range.primary].1[range.fragment];
                                let (_, backend) = self
                                    .devices
                                    .iter()
                                    .find(|(device, _)| *device == source.device)
                                    .ok_or(PolyBackendError::UnsupportedPlacement)?;
                                let input_ty = argument_types[source_indices[range.primary]]
                                    .matrix_type()
                                    .unwrap();
                                let input_params = backend.parameters(input_ty)?;
                                let (params, level, evaluation) = prepared.output_layout(
                                    backend,
                                    input_params,
                                    source.level,
                                    prepared.input_evaluation(source.evaluation),
                                )?;
                                Ok(super::gpu_prepare::MatrixInputFragment {
                                    device: source.device,
                                    context: params.context_identity(),
                                    start: range.columns.start,
                                    end: range.columns.end,
                                    level,
                                    evaluation,
                                })
                            })
                            .collect::<Result<Vec<_>, PolyBackendError>>()?;
                        layout.fragments = Some(fragments.into());
                    }
                }
                if layout.fragments.is_none() &&
                    self.devices.len() == 1 &&
                    !layout.lazy &&
                    !layout.alternatives &&
                    matches!(output, ConcreteWireType::Matrix(_)) &&
                    let Some(prepared) = prepared
                {
                    let backend = &self.devices[0].1;
                    let params = backend.parameters(ty)?;
                    let inputs = arguments
                        .iter()
                        .zip(argument_types)
                        .filter(|(_, ty)| matches!(ty, ConcreteWireType::Matrix(_)))
                        .map(|(wire, _)| {
                            column_layouts.get(wire).and_then(|layout| layout.fragments.as_deref())
                        })
                        .collect::<Option<Vec<_>>>();
                    let original_level = inputs.as_ref().and_then(|inputs| {
                        let mut fragments = inputs.iter().flat_map(|fragments| fragments.iter());
                        let level = fragments.next()?.level;
                        fragments.all(|fragment| fragment.level == level).then_some(level)
                    });
                    let level = if prepared.fresh_type().is_some() {
                        Some(params.crt_depth() - 1)
                    } else {
                        original_level
                    };
                    if let Some(level) = level {
                        let format = |input| {
                            prepared.output_layout(
                                backend,
                                params,
                                level,
                                prepared.input_evaluation(input),
                            )
                        };
                        let coefficient = format(false)?;
                        let evaluation = format(true)?;
                        let original = inputs.as_ref().and_then(|inputs| {
                            let mut fragments =
                                inputs.iter().flat_map(|fragments| fragments.iter());
                            let format = fragments.next()?.evaluation;
                            fragments
                                .all(|fragment| fragment.evaluation == format)
                                .then_some(format)
                        });
                        let known = if coefficient == evaluation {
                            Some(evaluation)
                        } else {
                            original.map(|input| if input { evaluation } else { coefficient })
                        };
                        if let Some((params, level, evaluation)) = known {
                            // C_cap divides jobs, not retained output owners.
                            // Only known levels/formats establish concrete fragments.
                            layout.fragments = Some(
                                layout
                                    .cuts
                                    .windows(2)
                                    .map(|range| super::gpu_prepare::MatrixInputFragment {
                                        device: self.devices[0].0,
                                        context: params.context_identity(),
                                        start: range[0],
                                        end: range[1],
                                        level,
                                        evaluation,
                                    })
                                    .collect(),
                            );
                        }
                    }
                }
                if matches!(kind, NodeKind::TrapdoorSample { .. }) {
                    // Sampling creates the complete public owner on device 0;
                    // Preimage admission selects any necessary consumer replicas.
                    let params = self.devices[0].1.parameters(ty)?;
                    layout.fragments = Some(Arc::from([super::gpu_prepare::MatrixInputFragment {
                        device: self.devices[0].0,
                        context: params.context_identity(),
                        start: 0,
                        end: ty.columns,
                        level: params.crt_depth() - 1,
                        evaluation: true,
                    }]));
                }
                column_layouts.insert(wire, layout);
            }
            // Inputs already on the device keep their native owners.
            // Merely borrowing them neither imports nor allocates another copy.
            if matches!(kind, NodeKind::Input { .. }) &&
                column_layouts
                    .get(&WireRef { node: id, port: Port(0) })
                    .is_some_and(|layout| layout.borrowed)
            {
                continue;
            }
            // Static family access only clones a value/descriptor. Dynamic
            // access and selection likewise clone an existing owner when all
            // candidates are eager. owner_liveness propagates the result's
            // lifetime to those owners; no import, destination, or normalization
            // belongs to this node. Lazy dynamic candidates still need the
            // materialization allowance below.
            if matches!(kind, NodeKind::FamilyGetStatic { .. }) ||
                (matches!(kind, NodeKind::FamilyGetDynamic | NodeKind::Select { .. }) &&
                    arguments.iter().all(|wire| {
                        !column_layouts.get(wire).is_some_and(|layout| layout.lazy)
                    }))
            {
                continue;
            }
            // Only materializing consumers create owners for lazy inputs. Packs
            // and static selection copy descriptors; loop Zip placement imports
            // one member into the borrowed child input map instead.
            let structural = matches!(
                kind,
                NodeKind::Input { .. } |
                    NodeKind::FamilyPack { .. } |
                    NodeKind::FamilyGetStatic { .. } |
                    NodeKind::FamilyGetDynamic |
                    NodeKind::Select { .. } |
                    NodeKind::SubgraphCall(_) |
                    NodeKind::SequentialLoop(_)
            );
            let materialized = arguments
                .iter()
                .enumerate()
                .filter_map(|(index, wire)| {
                    let layout = column_layouts.get(wire)?;
                    if !layout.lazy || structural {
                        return None;
                    }
                    let end = if let NodeKind::ParallelLoop(body) = kind {
                        if !matches!(
                            body.input_modes[index],
                            mxx_ir_core::node::LoopInputMode::Broadcast
                        ) || (matches!(
                            argument_types[index],
                            ConcreteWireType::IndexedFamily { .. }
                        ) && !layout.packed_family)
                        {
                            return None;
                        }
                        node_position
                    } else {
                        liveness.until(*wire).unwrap_or(node_position)
                    };
                    Some((*wire, argument_types[index].clone(), end))
                })
                .collect::<Vec<_>>();
            let mut imports = Vec::new();
            for (wire, ty, end) in materialized {
                until.set(if capture_trace { usize::MAX } else { end });
                for (path, leaf) in family_leaves(&ty) {
                    *owners.borrow_mut() = vec![(wire, path)];
                    if let ConcreteWireType::Matrix(ty) = leaf {
                        add_matrix(&mut demand, ty, ty.rows, ty.columns);
                    } else if let Some((ty, bound)) = bound_of(leaf) {
                        let params = parameters(self, ty)?;
                        add_layouts(
                            &mut demand,
                            ty,
                            vec![compact_payload(&params, ty.rows, ty.columns, &bound)?],
                        );
                    }
                    imports.push(leaf.clone());
                }
                // Placement imports clone the descriptor; ordinary materialize
                // replaces this wire's cached value in the executor map.
                if !matches!(kind, NodeKind::ParallelLoop(_)) {
                    column_layouts.get_mut(&wire).unwrap().lazy = false;
                }
            }
            if matches!(
                kind,
                NodeKind::Input { .. } |
                    NodeKind::FamilyGetStatic { .. } |
                    NodeKind::FamilyGetDynamic |
                    NodeKind::Select { .. }
            ) {
                for (port, ty) in output_types.iter().enumerate() {
                    let wire = WireRef { node: id, port: Port(port as u32) };
                    if !column_layouts.get(&wire).is_some_and(|layout| layout.lazy) &&
                        !matches!(ty, ConcreteWireType::IndexedFamily { .. })
                    {
                        imports.extend(family_leaves(ty).map(|(_, leaf)| leaf.clone()));
                    }
                }
            }
            until.set(node_position);
            owners.borrow_mut().clear();
            for output in imports {
                if let Some(ty) = output.matrix_type() {
                    let params = parameters(self, ty)?;
                    let (matrices, layouts) = import_inventory(&params, &output, scratch_columns)?;
                    for (rows, columns) in matrices {
                        add_matrix(&mut demand, ty, rows, columns);
                    }
                    add_layouts(&mut demand, ty, layouts);
                }
            }
            // Fold alternative child peaks; only simultaneously admitted
            // siblings multiply the bound, never the total iteration count.
            if let Some(child_id) = validated.source.child_scope_id(scope_id, id) {
                let (child_bindings, loop_slot, count) = match kind {
                    NodeKind::SubgraphCall(call) => (&call.bindings, None, 1),
                    NodeKind::ParallelLoop(body) => (
                        &body.bindings,
                        Some(body.index_slot),
                        body.count
                            .evaluate(bindings)
                            .ok()
                            .and_then(|value| value.to_usize())
                            .ok_or(PolyBackendError::InvalidInteger)?,
                    ),
                    NodeKind::SequentialLoop(body) => (
                        &body.bindings,
                        Some(body.index_slot),
                        body.count
                            .evaluate(bindings)
                            .ok()
                            .and_then(|value| value.to_usize())
                            .ok_or(PolyBackendError::InvalidInteger)?,
                    ),
                    _ => unreachable!("validated child scope"),
                };
                if count == 0 {
                    // Zero-iteration carried aliases were bound by owner lowering.
                    continue;
                }
                let mut child_inputs = arguments
                    .iter()
                    .map(|wire| column_layouts.get(wire).cloned().unwrap_or_default())
                    .zip(validated.source.scope(&child_id).unwrap().inputs().iter().copied())
                    .map(|(mut cuts, wire)| {
                        if matches!(kind, NodeKind::ParallelLoop(_)) &&
                            !matches!(
                                validated.scope(&child_id).unwrap().wire_types[&wire],
                                ConcreteWireType::IndexedFamily { .. }
                            )
                        {
                            cuts.lazy = false;
                        }
                        (wire, cuts)
                    })
                    .collect::<BTreeMap<_, _>>();
                let parallel_wave =
                    if matches!(kind, NodeKind::ParallelLoop(_)) { wave.min(count) } else { 1 };
                let mut child =
                    BTreeMap::<(String, usize), (ConcreteMatrixType, GpuContextDemand)>::new();
                let mut child_outputs = Vec::<GpuInventoryValue>::new();
                // Walk finite alternatives on CPU and immediately fold their
                // resource bounds. Never store descriptors for future bodies.
                // Explicit discovery caches primitive classes, so repeated
                // bindings do not add GPU probes. Production only looks up
                // those templates and errors if warmup missed a class.
                for index in 0..count {
                    let child_env = bindings
                        .child(child_bindings, loop_slot.map(|slot| (slot, index)))
                        .map_err(|_| PolyBackendError::InvalidInteger)?;
                    let (alternative, outputs, _) = self.graph_admission_demand(
                        validated,
                        &child_id,
                        &child_env,
                        None,
                        capture_trace,
                        Some(&child_inputs),
                        scratch_columns,
                        if matches!(kind, NodeKind::ParallelLoop(_)) {
                            parallel_wave
                        } else {
                            wave
                        },
                        warm_up,
                    )?;
                    if let NodeKind::SequentialLoop(body) = kind {
                        // The next iteration receives these actual carried
                        // layouts, not the original loop inputs. In particular,
                        // a staged descriptor can become an eager fragmented
                        // owner (or the reverse) after the first iteration.
                        for (wire, layout) in validated
                            .source
                            .scope(&child_id)
                            .unwrap()
                            .inputs()
                            .iter()
                            .take(body.carried_count)
                            .zip(&outputs)
                        {
                            child_inputs.insert(*wire, layout.clone());
                        }
                    }
                    for (key, (ty, alternative)) in alternative {
                        child
                            .entry(key)
                            .or_insert_with(|| (ty, GpuContextDemand::default()))
                            .1
                            .include_alternative(alternative);
                    }
                    if child_outputs.is_empty() || matches!(kind, NodeKind::SequentialLoop(_)) {
                        // A sequential loop returns only its final iteration's
                        // layout. Earlier iterations contribute to peak demand
                        // above, but are not alternative returned partitions.
                        child_outputs = outputs;
                    } else {
                        for (layout, alternative) in child_outputs.iter_mut().zip(outputs) {
                            layout.include_alternative(alternative);
                        }
                    }
                }
                // Inventory reads the executor's shared per-port staging
                // decision instead of a second, node-level rule, using the same
                // configured wave the backing is provisioned for: a parallel
                // loop whose whole count fits one wave keeps its members as
                // device owners, exactly as production does when it admits that
                // wave. A body the fleet cannot run simultaneously is never
                // planned as unstaged.
                let staged_ports = if matches!(kind, NodeKind::ParallelLoop(_)) {
                    let child_scope =
                        validated.source.scope(&child_id).expect("validated child scope");
                    let retained = crate::executor::retained_loop_output_ports(
                        validated,
                        scope_id,
                        id,
                        child_scope.outputs(),
                    );
                    crate::executor::staged_loop_outputs(scope_id, count, parallel_wave, &retained)
                } else {
                    Vec::new()
                };
                // Per output port: a staged port returns a fresh complete matrix
                // through host storage, so its previous device fragments do not
                // survive; an unstaged port keeps the child's derived layout and
                // lives until its own owner end. A retained child output that
                // maps to any unstaged port stays charged to that port's end.
                let mut output_end = node_position;
                for (port, boundaries) in child_outputs.into_iter().enumerate() {
                    if staged_ports.get(port).copied().unwrap_or(false) {
                        let Some(output) = output_types.get(port) else { continue };
                        let mut leaf = output;
                        while let ConcreteWireType::IndexedFamily { element, .. } = leaf {
                            leaf = element;
                        }
                        if let Some(ty) = leaf.matrix_type() {
                            let mut layout = GpuInventoryValue::new(vec![0, ty.columns]);
                            layout.lazy = true;
                            column_layouts
                                .insert(WireRef { node: id, port: Port(port as u32) }, layout);
                        }
                    } else {
                        column_layouts
                            .insert(WireRef { node: id, port: Port(port as u32) }, boundaries);
                        output_end = output_end.max(retained_until(port));
                    }
                }
                // Provision the same simultaneous resource lifetimes as live
                // admission. A single resolved child preserves shared owner
                // identities; unresolved alternatives remain containing bounds.
                for (_, (ty, child)) in &child {
                    let child = Arc::new(GpuContextDemand::simultaneous(
                        std::iter::repeat_with(|| child.clone()).take(parallel_wave),
                    ));
                    let child_scope = validated.source.scope(&child_id).unwrap();
                    let escaping = |claim| {
                        child_scope
                            .outputs()
                            .iter()
                            .enumerate()
                            .filter(|(port, _)| !staged_ports.get(*port).copied().unwrap_or(false))
                            .map(|(port, output)| ValueOrigin::Child {
                                instance: 0,
                                wire: WireRef { node: id, port: Port(port as u32) },
                                output: *output,
                                parallel: matches!(kind, NodeKind::ParallelLoop(_)),
                                demand: child.clone(),
                                claim,
                            })
                            .collect::<Vec<_>>()
                    };
                    owners.borrow_mut().clear();
                    for (index, ((rows, columns), end)) in
                        child.matrices.iter().zip(&child.matrix_until).enumerate()
                    {
                        until.set(if *end == usize::MAX { output_end } else { node_position });
                        add_matrix(&mut demand, ty, *rows, *columns);
                        if *end == usize::MAX {
                            demand
                                .get_mut(&(ty.modulus.to_string(), ty.ring_dimension))
                                .unwrap()
                                .1
                                .matrix_demands
                                .last_mut()
                                .unwrap()
                                .owners = escaping(index);
                        }
                    }
                    for (index, ((layout, end), shared)) in child
                        .layouts
                        .iter()
                        .zip(&child.layout_until)
                        .zip(&child.layout_shared)
                        .enumerate()
                    {
                        until.set(if *end == usize::MAX { output_end } else { node_position });
                        shared_layout.set(*shared);
                        add_layouts(&mut demand, ty, vec![*layout]);
                        shared_layout.set(false);
                        if *end == usize::MAX {
                            demand
                                .get_mut(&(ty.modulus.to_string(), ty.ring_dimension))
                                .unwrap()
                                .1
                                .workspace_demands
                                .last_mut()
                                .unwrap()
                                .owners = escaping(child.matrices.len() + index);
                        }
                    }
                }
                if matches!(kind, NodeKind::SequentialLoop(_)) {
                    // The preceding iteration's carried outputs remain live
                    // while the next body computes their replacements.
                    until.set(node_position);
                    owners.borrow_mut().clear();
                    for (_, ty) in output_types.iter().flat_map(family_leaves) {
                        if let ConcreteWireType::Matrix(ty) = ty {
                            add_matrix(&mut demand, ty, ty.rows, ty.columns);
                        } else if let Some((ty, bound)) = bound_of(ty) {
                            let params = parameters(self, ty)?;
                            add_layouts(
                                &mut demand,
                                ty,
                                vec![compact_payload(&params, ty.rows, ty.columns, &bound)?],
                            );
                        }
                    }
                }
                continue;
            }
            // Scalars create no native owners; family packs preserve aliases.
            match kind {
                NodeKind::ConstantInt(_) |
                NodeKind::EvaluateInt(_) |
                NodeKind::ConstantReal(_) |
                NodeKind::ConstantBool(_) |
                NodeKind::IntBinary(_) |
                NodeKind::IntCompare(_) |
                NodeKind::BitExtract { .. } |
                NodeKind::IntToReal |
                NodeKind::BoolToInt |
                NodeKind::RealBinary(_) |
                NodeKind::RealSqrt |
                NodeKind::FamilyPack { .. } => continue,
                NodeKind::Input { .. } |
                NodeKind::FamilyGetStatic { .. } |
                NodeKind::FamilyGetDynamic |
                NodeKind::Select { .. } => {}
                // Type-determined GPU operations are recognized by the same
                // lowering used for execution, not a second supported-kind list.
                _ if prepared.is_some() => {}
                NodeKind::PolynomialFromValues { .. } |
                NodeKind::PolynomialValues { .. } |
                NodeKind::TrapdoorSample { .. } |
                NodeKind::TrapdoorPublic |
                NodeKind::PreimageSample { .. } => {}
                other => return Err(unsupported(other)),
            }
            // Family inputs are collections of existing owners or lazy member
            // descriptors. Merely binding the family imports no members; the
            // selecting node accounts for each materialized member below.
            if matches!(kind, NodeKind::Input { .. }) &&
                output_types
                    .iter()
                    .all(|ty| matches!(ty, ConcreteWireType::IndexedFamily { .. }))
            {
                continue;
            }
            // Retained outputs.
            for (port, path, output) in output_types
                .iter()
                .enumerate()
                .flat_map(|(port, ty)| {
                    family_leaves(ty).map(move |(path, leaf)| (port, path, leaf))
                })
                .filter(|_| !optimizer.omitted.contains(&id))
                .filter(|(port, _, _)| {
                    !column_layouts
                        .get(&WireRef { node: id, port: Port(*port as u32) })
                        .is_some_and(|layout| layout.lazy)
                })
            {
                until.set(retained_until(port));
                *owners.borrow_mut() = vec![(WireRef { node: id, port: Port(port as u32) }, path)];
                match output {
                    ConcreteWireType::Matrix(ty) => {
                        for &columns in &column_layouts
                            [&WireRef { node: id, port: Port(port as u32) }]
                            .capacities
                        {
                            add_matrix(&mut demand, ty, ty.rows, columns);
                        }
                    }
                    ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. } => {
                        let (ty, bound) =
                            bound_of(output).ok_or(PolyBackendError::InvalidInteger)?;
                        let params = parameters(self, ty)?;
                        for &columns in &column_layouts
                            [&WireRef { node: id, port: Port(port as u32) }]
                            .capacities
                        {
                            let layout = compact_payload(&params, ty.rows, columns, &bound)?;
                            add_layouts(&mut demand, ty, vec![layout]);
                        }
                    }
                    ConcreteWireType::Bytes { .. } |
                    ConcreteWireType::Trapdoor { .. } |
                    ConcreteWireType::Int |
                    ConcreteWireType::Real |
                    ConcreteWireType::Bool => {}
                    // Scalar value families (polynomial readback) hold no owners.
                    ConcreteWireType::IndexedFamily { element, .. }
                        if matches!(**element, ConcreteWireType::Int) => {}
                    _ => return Err(unsupported(kind)),
                }
            }
            if let Some(outputs) = optimizer.outputs.get(&id) {
                for (ty, aliases) in outputs {
                    let end = aliases
                        .iter()
                        .map(|wire| {
                            if checked.liveness.retained.contains(wire) {
                                usize::MAX
                            } else {
                                liveness.until(*wire).unwrap_or(node_position)
                            }
                        })
                        .max()
                        .unwrap_or(node_position);
                    until.set(end);
                    *owners.borrow_mut() = aliases.iter().map(|wire| (*wire, Vec::new())).collect();
                    for &columns in &column_layouts[&WireRef { node: id, port: Port(0) }].capacities
                    {
                        add_matrix(&mut demand, ty, ty.rows, columns);
                    }
                }
            }
            until.set(node_position);
            owners.borrow_mut().clear();
            // Replication may need a full owner; normalization preserves every
            // input fragment and keeps all normalized fragments live together.
            let mut prepared_inputs = HashSet::new();
            for (wire, argument) in arguments.iter().zip(argument_types) {
                if column_layouts.get(wire).is_some_and(|layout| layout.lazy) {
                    continue;
                }
                if let ConcreteWireType::Matrix(ty) = argument {
                    let mut source = *wire;
                    while let Some(alias) = liveness.alias_source(source) {
                        source = *alias;
                    }
                    let identity = column_layouts
                        .get(wire)
                        .and_then(|layout| {
                            layout
                                .owner
                                .map(MatrixInputOwner::Actual)
                                .or_else(|| layout.symbolic_owner.map(MatrixInputOwner::Symbolic))
                                .or_else(|| layout.broadcast.map(MatrixInputOwner::Broadcast))
                        })
                        .ok_or(source);
                    // Native preparation shares by actual matrix owner. Equal
                    // shapes or fragment boundaries alone do not establish aliasing.
                    if !prepared_inputs.insert(identity) {
                        continue;
                    }
                    let layout = column_layouts.get(wire);
                    let shapes = if let Some(operation) = &prepared &&
                        let Some(fragments) = layout
                            .and_then(|layout| layout.fragments.as_ref())
                            .filter(|fragments| !fragments.is_empty())
                    {
                        let columns = if matches!(operation, PreparedMatrixOperation::Slice { .. })
                        {
                            operation.source_columns(
                                ty.columns,
                                0,
                                output_types[0].matrix_type().unwrap().columns,
                            )
                        } else {
                            0..ty.columns
                        };
                        let mut bound = GpuContextDemand::default();
                        for (device, (_, backend)) in self.devices.iter().enumerate() {
                            let params = backend.parameters(ty)?;
                            let mut target = GpuContextDemand::default();
                            let mut selected = HashSet::new();
                            // Bound both a full-range replica and owner-local
                            // ranges. Actual inherited ranges may need several
                            // fragment normalizations rather than one replica.
                            let ranges = std::iter::once(columns.clone()).chain(
                                fragments.iter().filter_map(|fragment| {
                                    let start = fragment.start.max(columns.start);
                                    let end = fragment.end.min(columns.end);
                                    (start < end).then_some(start..end)
                                }),
                            );
                            for columns in ranges {
                                let (_, preparation) =
                                    super::gpu_prepare::PreparedMatrixSource::plan(
                                        fragments.iter().copied(),
                                        (ty.rows, ty.columns),
                                        columns,
                                        device,
                                        params,
                                        operation.input_evaluation(fragments[0].evaluation),
                                    );
                                for input in preparation {
                                    if selected.insert(input.source) {
                                        target.matrix(0, 0, input.shape.0, input.shape.1);
                                    }
                                }
                            }
                            // This inventory is replicated per target. Preserve
                            // the containing typed peak across target layouts,
                            // not their sum; native fitting remains per device.
                            bound.include_alternative(target);
                        }
                        bound.matrices
                    } else {
                        // No actual/derived source layout is available yet.
                        // Retain the declared full-owner and fragment envelope.
                        std::iter::once((ty.rows, ty.columns))
                            .chain(
                                layout
                                    .filter(|layout| layout.capacities.len() > 1)
                                    .into_iter()
                                    .flat_map(|layout| {
                                        layout.capacities.iter().map(|&columns| (ty.rows, columns))
                                    }),
                            )
                            .collect()
                    };
                    // The concrete preflight shares all preparation for an
                    // proven owner across a supported native sibling batch.
                    // Broadcast identity comes from the parent argument, never
                    // from equal shapes. Scalar-dispatched operations
                    // do not promise one shared preparation lifetime.
                    let identity = prepared.as_ref().and_then(|operation| {
                        operation.matrix_batch_kind(false)?;
                        let layout = layout?;
                        let owner = layout
                            .owner
                            .map(MatrixInputOwner::Actual)
                            .or_else(|| layout.symbolic_owner.map(MatrixInputOwner::Symbolic))
                            .or_else(|| layout.broadcast.map(MatrixInputOwner::Broadcast))?;
                        let evaluation = layout
                            .fragments
                            .as_ref()
                            .and_then(|fragments| fragments.first())
                            .is_some_and(|fragment| fragment.evaluation);
                        Some((owner, operation.input_evaluation(evaluation)))
                    });
                    for (ordinal, (rows, columns)) in shapes.into_iter().enumerate() {
                        add_matrix(&mut demand, ty, rows, columns);
                        if let Some((owner, evaluation)) = identity {
                            demand
                                .get_mut(&(ty.modulus.to_string(), ty.ring_dimension))
                                .unwrap()
                                .1
                                .matrix_demands
                                .last_mut()
                                .unwrap()
                                .preparation = Some((owner, ordinal, evaluation));
                        }
                    }
                }
            }
            // Every type-determined primitive consumes the production operation's
            // resource description. Graph structure only adds fusion-specific
            // or explicitly traced resources below.
            if let Some(ty) = output_types.first().and_then(ConcreteWireType::matrix_type) {
                let params = parameters(self, ty)?;
                if let Some(operation) = &prepared {
                    let level = params.crt_depth() - 1;
                    let columns = ty.columns.min(scratch_columns);
                    for rows in operation.scratch_rows()? {
                        add_matrix(&mut demand, ty, rows, columns);
                    }
                    add_layouts(&mut demand, ty, operation.fixed_workspaces(&params, level)?);
                    let workspaces = operation.width_workspaces(&params, level, columns)?;
                    let copies =
                        if matches!(operation, PreparedMatrixOperation::MultiplyCompact { .. }) {
                            optimizer.outputs.get(&id).map_or(1, Vec::len)
                        } else {
                            1
                        };
                    for _ in 0..copies {
                        add_layouts(&mut demand, ty, workspaces.clone());
                    }
                    // The smallest retained owner gives the conservative native
                    // auxiliary-workspace bound for all admitted siblings.
                    let shared = operation.batch_workspaces(
                        &params,
                        level,
                        (ty.rows.max(1), 1),
                        wave.max(1),
                        argument_types.iter().any(|ty| {
                            matches!(ty,
                            ConcreteWireType::Matrix(ty) if ty.rows == 1 && ty.columns == 1)
                        }),
                    )?;
                    shared_layout.set(true);
                    add_layouts(&mut demand, ty, shared);
                    shared_layout.set(false);
                }
            }
            // Kind-specific native scratch at the candidate range width.
            match kind {
                NodeKind::PolynomialFromValues { .. } => {
                    if let ConcreteWireType::Matrix(ty) = &output_types[0] {
                        if ty.rows == 1 && ty.columns == 1 {
                            let params = parameters(self, ty)?;
                            // Runtime polynomial values use the same
                            // polynomial loader's resource description.
                            let fallback = PreparedMatrixOperation::Polynomial {
                                ty: ty.clone(),
                                coefficients: Vec::new(),
                                evaluation: false,
                            };
                            let operation = prepared.as_ref().unwrap_or(&fallback);
                            add_layouts(
                                &mut demand,
                                ty,
                                operation.fixed_workspaces(&params, params.crt_depth() - 1)?,
                            );
                        }
                    }
                }
                NodeKind::PolynomialValues { evaluation } => {
                    let Some(ConcreteWireType::Matrix(ty)) = argument_types.first() else {
                        return Err(unsupported(kind));
                    };
                    let claims = self.trace_polynomial_values(ty, *evaluation, warm_up)?;
                    let params = parameters(self, ty)?;
                    push_traced(&mut demand, ty, &params, &claims, &add_matrix, &add_layouts);
                }
                NodeKind::TrapdoorSample { .. } => {
                    // Port 0 is the public matrix, port 1 the trapdoor.
                    let Some(ConcreteWireType::Trapdoor {
                        matrix,
                        sigma,
                        gadget_base,
                        digit_count,
                        ..
                    }) = output_types.get(1)
                    else {
                        return Err(unsupported(kind));
                    };
                    let sigma = sigma
                        .evaluate_f64(bindings)
                        .map_err(|e| PolyBackendError::GpuSubmission(e.to_string()))?;
                    until.set(retained_until(0).max(retained_until(1)));
                    *owners.borrow_mut() = vec![
                        (WireRef { node: id, port: Port(0) }, Vec::new()),
                        (WireRef { node: id, port: Port(1) }, Vec::new()),
                    ];
                    let plan = self.trace_trapdoor_sampling(
                        matrix,
                        sigma,
                        gadget_base,
                        *digit_count,
                        warm_up,
                    )?;
                    let params = parameters(self, matrix)?;
                    for claims in [&plan.sample, &plan.export, &plan.import] {
                        push_traced(
                            &mut demand,
                            matrix,
                            &params,
                            claims,
                            &add_matrix,
                            &add_layouts,
                        );
                    }
                }
                NodeKind::PreimageSample { .. } => {
                    let (
                        Some(ConcreteWireType::Matrix(public)),
                        Some(ConcreteWireType::Trapdoor {
                            matrix: trapdoor_matrix,
                            sigma,
                            gadget_base,
                            digit_count,
                            ..
                        }),
                    ) = (argument_types.first(), argument_types.get(1))
                    else {
                        return Err(unsupported(kind));
                    };
                    let (ty, bound) =
                        bound_of(&output_types[0]).ok_or(PolyBackendError::InvalidInteger)?;
                    let sigma = sigma
                        .evaluate_f64(bindings)
                        .map_err(|e| PolyBackendError::GpuSubmission(e.to_string()))?;
                    let plan = self.trace_preimage_plan(
                        trapdoor_matrix,
                        public,
                        ty,
                        &bound,
                        sigma,
                        gadget_base,
                        *digit_count,
                        warm_up,
                    )?;
                    let params = parameters(self, ty)?;
                    // The hard-cutoff plan is owned by the compact output,
                    // including its acceptance word. It outlives this node.
                    until.set(retained_until(0));
                    *owners.borrow_mut() = vec![(WireRef { node: id, port: Port(0) }, Vec::new())];
                    push_traced(
                        &mut demand,
                        ty,
                        &params,
                        &plan.destination,
                        &add_matrix,
                        &add_layouts,
                    );
                    until.set(node_position);
                    owners.borrow_mut().clear();
                    // The real trapdoor is already a retained input. Synthetic
                    // trapdoor construction belongs only to explicit discovery.
                    let mut claims = Vec::new();
                    let width = ty.columns.min(scratch_columns);
                    claims.extend(
                        plan.tile_claims(&params, public.rows, width)
                            .map_err(PolyBackendError::GpuSubmission)?,
                    );
                    claims.extend(
                        plan.attempt_claims(&params, public.rows, width, ty.rows, &bound)
                            .map_err(PolyBackendError::GpuSubmission)?
                            .into_iter()
                            .flatten(),
                    );
                    push_traced(&mut demand, ty, &params, &claims, &add_matrix, &add_layouts);
                    shared_layout.set(true);
                    add_layouts(
                        &mut demand,
                        ty,
                        // Descriptor bytes depend on the sibling count, but a
                        // wide result may embed them in its matrix allocation.
                        // Split/tail results have less embedded space. Query the
                        // smallest legal result to bound every admitted width.
                        PreimageClaimPlan::residual_batch_metadata(&params, public.rows, 1, wave)
                            .map_err(PolyBackendError::GpuSubmission)?
                            .into_iter()
                            .map(|claim| claim.layout().expect("batch metadata"))
                            .collect(),
                    );
                    shared_layout.set(false);
                }
                NodeKind::Concat { axis: ConcatAxis::Rows } => {
                    // Fused row sums reduce large groups through one-row
                    // intermediates; provision one per concatenated block.
                    if let ConcreteWireType::Matrix(ty) = &output_types[0] {
                        for _ in argument_types {
                            add_matrix(&mut demand, ty, 1, ty.columns.min(scratch_columns));
                        }
                    }
                }
                NodeKind::Tensor => {
                    if let ConcreteWireType::Matrix(ty) = &output_types[0] {
                        add_matrix(&mut demand, ty, 1, ty.columns.min(scratch_columns));
                    }
                }
                _ => {}
            }
        }
        owners.borrow_mut().clear();
        position.set(checked.execution_order.len());
        until.set(position.get());
        // Exports: the codec clones the owner, opens its private stream and
        // uses the store workspace; compact exports record one completion.
        for output in scope.outputs() {
            let producer = crate::gpu_invocation::GpuNodeOperation::new(
                validated,
                scope_id,
                output.node,
                bindings,
            )
            .map_err(PolyBackendError::GpuSubmission)?;
            let declared_output = &producer.outputs()[output.port.0 as usize];
            let mut output_type = declared_output;
            while let ConcreteWireType::IndexedFamily { element, .. } = output_type {
                output_type = element;
            }
            if *scope_id == FrozenGraphScopeId::Root &&
                column_layouts.get(output).is_some_and(|layout| layout.lazy)
            {
                // materialize_output recursively loads the entire requested
                // family. Every imported member remains owned by the result;
                // only its transfer scratch can be reused between members.
                until.set(usize::MAX);
                for (path, leaf) in family_leaves(declared_output) {
                    *owners.borrow_mut() = vec![(*output, path)];
                    if let ConcreteWireType::Matrix(ty) = leaf {
                        add_matrix(&mut demand, ty, ty.rows, ty.columns);
                    } else if let Some((ty, bound)) = bound_of(leaf) {
                        let params = parameters(self, ty)?;
                        add_layouts(
                            &mut demand,
                            ty,
                            vec![compact_payload(&params, ty.rows, ty.columns, &bound)?],
                        );
                    }
                }
                until.set(position.get());
                owners.borrow_mut().clear();
                if let Some(ty) = output_type.matrix_type() {
                    let params = parameters(self, ty)?;
                    let (matrices, layouts) =
                        import_inventory(&params, output_type, scratch_columns)?;
                    for (rows, columns) in matrices {
                        add_matrix(&mut demand, ty, rows, columns);
                    }
                    add_layouts(&mut demand, ty, layouts);
                }
            }
            match output_type {
                ConcreteWireType::Matrix(ty) => {
                    let params = parameters(self, ty)?;
                    let store = params
                        .compact_transfer_workspace(
                            params.crt_depth() - 1,
                            ty.rows,
                            ty.columns,
                            GpuCompactTransferKind::Store,
                        )
                        .map_err(PolyBackendError::GpuSubmission)?;
                    add_matrix(&mut demand, ty, ty.rows, ty.columns);
                    add_layouts(&mut demand, ty, vec![stream(), store]);
                    // Family staging pipelines at most two full shard snapshots.
                    let transfer = params
                        .rns_transfer_workspace(params.crt_depth() - 1, ty.rows, ty.columns)
                        .map_err(PolyBackendError::GpuSubmission)?;
                    for _ in 0..2 {
                        add_layouts(
                            &mut demand,
                            ty,
                            vec![
                                GpuPreparedWorkspaceLayout {
                                    kind: GpuPreparedSlotKind::PinnedHost,
                                    bytes: transfer.bytes,
                                    alignment: 1,
                                },
                                transfer,
                                event(),
                            ],
                        );
                    }
                }
                other if bound_of(other).is_some() => {
                    let (ty, _) = bound_of(other).unwrap();
                    add_layouts(&mut demand, ty, vec![event()]);
                }
                _ => {}
            }
        }
        // Keep a compact reverse alias graph. Value queries resolve only the
        // requested member paths; nested repeated packs never expand eagerly.
        let mut aliases = ValueAliases::new();
        for (index, handle) in checked.execution_order.iter().enumerate() {
            let node = mxx_ir_core::types::NodeId(index as u64);
            let arguments = scope.arguments(handle).unwrap();
            for port in 0..handle.output_types().len() {
                let output = WireRef { node, port: Port(port as u32) };
                if let Some(source) = liveness.alias_source(output) {
                    aliases.entry(output).or_default().push((*source, MemberProjection::Same));
                    continue;
                }
                match handle.kind() {
                    NodeKind::FamilyPack { .. } => {
                        for (index, source) in arguments.iter().enumerate() {
                            aliases
                                .entry(output)
                                .or_default()
                                .push((*source, MemberProjection::Pack(index)));
                        }
                    }
                    NodeKind::FamilyGetStatic { index } => {
                        let index = index
                            .evaluate(bindings)
                            .map_err(|_| PolyBackendError::InvalidInteger)?
                            .to_usize()
                            .ok_or(PolyBackendError::InvalidInteger)?;
                        aliases
                            .entry(output)
                            .or_default()
                            .push((arguments[0], MemberProjection::Static(index)));
                    }
                    NodeKind::FamilyGetDynamic => {
                        aliases
                            .entry(output)
                            .or_default()
                            .push((arguments[0], MemberProjection::Dynamic));
                    }
                    NodeKind::Select { .. } => {
                        for source in arguments.iter().skip(1) {
                            aliases
                                .entry(output)
                                .or_default()
                                .push((*source, MemberProjection::Same));
                        }
                    }
                    _ => {}
                }
            }
        }
        let aliases = Arc::new(aliases);
        for (_, context) in demand.values_mut() {
            context.aliases = vec![aliases.clone()];
        }
        let outputs = scope
            .outputs()
            .iter()
            .map(|wire| column_layouts.get(wire).cloned().unwrap_or_default())
            .collect();
        Ok((demand, outputs, column_layouts))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_value_claims_preserve_reuse_and_sibling_identity() {
        use super::{
            GpuScopeProgress::{Before, Issued},
            ValueOrigin,
        };
        let wire = |index| super::WireRef {
            node: mxx_ir_core::types::NodeId(index),
            port: super::Port(0),
        };
        let mut demand = super::GpuContextDemand::default();
        demand.matrix(0, 2, 2, 3);
        demand.matrix_demands[0].owners =
            vec![ValueOrigin::Direct { instance: Some(0), wire: wire(0), members: vec![] }];
        demand.matrix(3, 4, 2, 3);
        demand.matrix_demands[1].owners =
            vec![ValueOrigin::Direct { instance: Some(0), wire: wire(3), members: vec![] }];
        assert_eq!(demand.matrices.len(), 1);
        assert_eq!(demand.value_claim_indices(0, wire(0), &[], Before(1)).len(), 1);
        assert!(demand.value_claim_indices(0, wire(3), &[], Before(1)).is_empty());
        assert!(demand.value_claim_indices(0, wire(0), &[], Issued(3)).is_empty());
        assert_eq!(demand.value_claim_indices(0, wire(3), &[], Issued(3)).len(), 1);
        let mut alternative = demand.clone();
        alternative.matrix_demands[0].owners =
            vec![ValueOrigin::Direct { instance: Some(0), wire: wire(1), members: vec![] }];
        demand.include_alternative(alternative);
        assert_eq!(
            demand.matrix_demands[1].begin, 3,
            "alias alternatives do not erase numeric lifetimes"
        );
        assert_eq!(
            demand.value_claim_indices(0, wire(0), &[], Before(1)),
            demand.value_claim_indices(0, wire(1), &[], Before(1))
        );
        let merged = super::GpuContextDemand::simultaneous([demand.clone(), demand]);
        let first = merged.value_claim_indices(0, wire(0), &[], Before(1));
        let second = merged.value_claim_indices(1, wire(0), &[], Before(1));
        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);
        assert!(first.is_disjoint(&second));

        // Sixty-four repeated packs describe 2^64 references to one owner.
        // Resolve their reverse edges without expanding the family members.
        let mut packed = merged.clone();
        let mut aliases = super::ValueAliases::new();
        let mut source = wire(0);
        for index in 10..74 {
            let output = wire(index);
            aliases.insert(
                output,
                vec![
                    (source, super::MemberProjection::Pack(0)),
                    (source, super::MemberProjection::Pack(1)),
                ],
            );
            source = output;
        }
        packed.aliases[0] = std::sync::Arc::new(aliases);
        assert_eq!(packed.value_claim_indices(0, source, &[], Before(1)), first);
        assert_eq!(packed.value_claim_indices(0, source, &[1; 64], Before(1)), first);
    }

    #[test]
    fn test_child_value_alternatives_merge_without_expanding_index_histories() {
        use super::{GpuContextDemand, MemberProjection, ValueAliases, ValueOrigin};
        use std::sync::Arc;
        let wire = |index| super::WireRef {
            node: mxx_ir_core::types::NodeId(index),
            port: super::Port(0),
        };
        let mut child = GpuContextDemand::default();
        for member in 0..2 {
            child.matrix(0, usize::MAX, 1, 1);
            child.matrix_demands.last_mut().unwrap().owners = vec![ValueOrigin::Direct {
                instance: Some(0),
                wire: wire(0),
                members: vec![member],
            }];
        }
        let mut origins = Vec::new();
        for index in 0..512 {
            let mut alternative = child.clone();
            alternative.aliases = vec![Arc::new(ValueAliases::from([(
                wire(1),
                vec![(wire(0), MemberProjection::Static(index % 2))],
            )]))];
            super::merge_value_origins(
                &mut origins,
                [ValueOrigin::Child {
                    instance: 0,
                    wire: wire(2),
                    output: wire(1),
                    parallel: false,
                    demand: Arc::new(alternative),
                    claim: 0,
                }],
            );
        }
        assert_eq!(origins.len(), 1);
        let ValueOrigin::Child { demand, .. } = &origins[0] else { unreachable!() };
        assert_eq!(demand.aliases[0][&wire(1)].len(), 1);
        assert!(matches!(demand.aliases[0][&wire(1)][0].1, MemberProjection::Dynamic));
        assert!(demand.same_timeline(&child));
        assert_eq!(
            demand.value_claim_indices(
                0,
                wire(1),
                &[],
                super::GpuScopeProgress::Before(usize::MAX),
            ),
            [0, 1].into_iter().collect()
        );
        child.matrix(0, usize::MAX, 2, 2);
        super::merge_value_origins(
            &mut origins,
            [ValueOrigin::Child {
                instance: 0,
                wire: wire(2),
                output: wire(1),
                parallel: false,
                demand: Arc::new(child),
                claim: 0,
            }],
        );
        assert_eq!(origins.len(), 2, "different numeric claims cannot share claim numbering");
    }

    #[test]
    fn test_retained_claim_indices_follow_original_and_reused_capacity_lifetimes() {
        use std::collections::BTreeSet;
        let mut demand = super::GpuContextDemand::default();
        demand.matrix(0, 2, 2, 3);
        demand.matrix(1, 5, 2, 3);
        demand.matrix(3, 4, 2, 3);
        demand.workspace(0, 0, super::event(), false);
        demand.workspace(2, 4, super::event(), false);
        demand.workspace(0, usize::MAX, super::stream(), true);
        for (position, expected) in [
            (0, vec![]),
            (1, vec![0, 3]),
            (2, vec![0, 1, 3]),
            (3, vec![1, 2, 3]),
            (4, vec![0, 1, 2, 3]),
            (5, vec![1, 3]),
            (6, vec![3]),
        ] {
            assert_eq!(
                demand.retained_claim_indices(None, |_| super::GpuScopeProgress::Before(position)),
                expected.into_iter().collect::<BTreeSet<_>>()
            );
        }
        assert!(
            !demand
                .retained_claim_indices(Some(0), |_| super::GpuScopeProgress::Before(4))
                .contains(&0),
            "the old allocation expired even though capacity slot zero is reused"
        );
        assert!(
            demand
                .retained_claim_indices(Some(3), |_| super::GpuScopeProgress::Before(4))
                .contains(&0)
        );
        let merged = super::GpuContextDemand::simultaneous([demand.clone(), demand.clone()]);
        assert_eq!(
            merged.retained_claim_indices(None, |_| super::GpuScopeProgress::Before(3)).len(),
            5
        );
        assert_eq!(
            merged.retained_claim_indices(None, |_| super::GpuScopeProgress::Before(4)).len(),
            7
        );
        use super::GpuScopeProgress::{Before, Issued};
        assert_eq!(
            merged
                .retained_claim_indices(None, |index| if index == 0 {
                    Issued(3)
                } else {
                    Before(3)
                })
                .len(),
            6
        );
        for issued in [0, 1] {
            assert_eq!(
                merged
                    .retained_claim_indices(None, |index| if index == issued {
                        Issued(0)
                    } else {
                        Before(0)
                    })
                    .len(),
                3
            );
        }
        let flat = super::GpuContextDemand::simultaneous([
            demand.clone(),
            demand.clone(),
            demand.clone(),
            demand.clone(),
        ]);
        let grouped = super::GpuContextDemand::simultaneous([merged.clone(), merged]);
        let progress = [Before(0), Before(1), Before(3), Issued(3)];
        assert_eq!(flat.retained_claim_indices(None, |index| progress[index]).len(), 7);
        assert_eq!(grouped.retained_claim_indices(None, |index| progress[index]).len(), 7);
        let empty_then_value = super::GpuContextDemand::simultaneous([
            super::GpuContextDemand::default(),
            demand.clone(),
        ]);
        assert!(
            empty_then_value
                .retained_claim_indices(None, |index| if index == 0 {
                    Issued(3)
                } else {
                    Before(0)
                })
                .is_empty()
        );

        // Placement belongs to the enclosing admitted region, not to an
        // invented body. It is held even before body zero and survives another
        // sibling merge without changing the callback's instance numbering.
        let mut placed = super::GpuContextDemand::default();
        placed.matrix(0, usize::MAX, 2, 3);
        placed.matrix_demands[0].origins = super::ResourceOrigins::Placement;
        let with_placement =
            super::GpuContextDemand::simultaneous([placed, super::GpuContextDemand::default()]);
        let nested = super::GpuContextDemand::simultaneous([with_placement, demand.clone()]);
        assert_eq!(
            nested
                .retained_claim_indices(None, |index| {
                    assert_eq!(index, 2);
                    Before(0)
                })
                .len(),
            1
        );
        assert_eq!(
            nested
                .retained_claim_indices(None, |index| {
                    assert_eq!(index, 2);
                    Issued(0)
                })
                .len(),
            4
        );

        let mut alternative = demand.clone();
        alternative.matrix(0, 8, 2, 3);
        demand.include_alternative(alternative);
        assert_eq!(
            demand.retained_claim_indices(None, |_| super::GpuScopeProgress::Before(1)),
            (0..demand.matrices.len() + demand.layouts.len()).collect()
        );
    }

    #[test]
    fn test_inventory_empty_envelope_preserves_shared_identity_on_both_sides() {
        let mut source = super::GpuInventoryValue::new(vec![0, 4]);
        source.borrowed = true;
        source.symbolic_owner = Some([7; 32]);
        let mut left = super::GpuInventoryValue::default();
        left.include_alternative(source.clone());
        let mut right = source.clone();
        right.include_alternative(super::GpuInventoryValue::default());
        for value in [left, right] {
            assert_eq!(value.cuts, source.cuts);
            assert_eq!(value.capacities, source.capacities);
            assert_eq!(value.symbolic_owner, source.symbolic_owner);
            assert!(value.borrowed);
            assert!(!value.alternatives);
        }
    }

    use super::select_wave_and_width;
    use crate::{
        Backend, MemoryArtifactStore, RuntimeValue,
        executor::{ExecutionConfig, execute_with_config},
        transcript::SamplingMode,
    };
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::{ParamEnv, graph::FrozenGraphScopeId, types::ConcreteMatrixType};
    use mxx_primitives::{
        matrix::{
            PolyMatrix,
            gpu_dcrt_poly::{GpuDCRTPolyMatrix, GpuPreparedSlotKind},
        },
        poly::{
            Poly as _, PolyParams,
            dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
        },
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    };
    use std::collections::BTreeMap;

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_inventory_deduplicates_actual_input_owners_not_equal_shapes() {
        use mxx_ir_core::{
            node::NodeKind,
            types::{Port, WireRef},
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        // The transposed export uses a different matrix size class, so its
        // serialization copy cannot hide the input preparation peak.
        let graph = DslContext::new("shared-owner-inventory")
            .output("sum", (ring.input("a", (2, 3)) + ring.input("b", (2, 3))).transpose())
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let sample =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 3, DistType::FinRingDist);
        let make = || {
            let mut matrix = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &sample);
            matrix.intt_all_in_place();
            RuntimeValue::matrix(matrix.into())
        };
        let first = make();
        let second = make();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        let wires = graph
            .root_scope()
            .execution_order
            .iter()
            .enumerate()
            .filter(|(_, node)| matches!(node.kind(), NodeKind::Input { .. }))
            .map(|(index, _)| WireRef {
                node: mxx_ir_core::types::NodeId(index as u64),
                port: Port(0),
            })
            .collect::<Vec<_>>();
        assert_eq!(wires.len(), 2);
        let mut demand =
            |left: &RuntimeValue<crate::backend::poly_gpu::GpuDcrtBackend>,
             right: &RuntimeValue<crate::backend::poly_gpu::GpuDcrtBackend>| {
                let inputs = wires
                    .iter()
                    .copied()
                    .zip([left, right])
                    .map(|(wire, value)| {
                        let mut layout = super::GpuInventoryValue::default();
                        layout.include_input(value, 3);
                        assert!(layout.borrowed, "resident input ownership is intrinsic metadata");
                        (wire, layout)
                    })
                    .collect();
                let (contexts, outputs, _) = backend
                    .graph_admission_demand(
                        &graph,
                        &FrozenGraphScopeId::Root,
                        &graph.bindings,
                        None,
                        false,
                        Some(&inputs),
                        1,
                        1,
                        false,
                    )
                    .unwrap();
                let fragments = outputs[0].fragments.as_ref().expect("known generated output");
                assert_eq!(fragments.len(), 1, "scratch jobs must not split retained owners");
                assert_eq!((fragments[0].start, fragments[0].end), (0, 2));
                assert!(fragments[0].evaluation);
                assert_eq!(contexts.len(), 1);
                contexts.into_values().next().unwrap().1
            };
        let shared = demand(&first, &first);
        let distinct = demand(&first, &second);
        assert_eq!(
            distinct.matrices.len(),
            shared.matrices.len() + 1,
            "one shared coefficient input needs one normalization owner, distinct equal-shaped inputs need two"
        );
        let other = demand(&second, &second);
        let shared_wave = super::GpuContextDemand::simultaneous([shared.clone(), shared.clone()]);
        let distinct_wave = super::GpuContextDemand::simultaneous([shared, other]);
        assert_eq!(
            distinct_wave.matrices.len(),
            shared_wave.matrices.len() + 1,
            "siblings share one actual normalization owner, not equal-shaped independent inputs"
        );
    }

    #[test]
    fn test_simultaneous_demand_merges_lifetimes_without_aliasing_equal_shapes() {
        use super::GpuContextDemand;
        let phase = |begin, end| {
            let mut demand = GpuContextDemand::default();
            demand.matrix(begin, end, 2, 3);
            demand.matrix(begin, end, 2, 3);
            demand
        };
        assert_eq!(
            GpuContextDemand::simultaneous([phase(0, 0), phase(1, 1)]).matrices.len(),
            2,
            "different IR positions reuse the same two slots"
        );
        assert_eq!(
            GpuContextDemand::simultaneous([phase(0, 1), phase(1, 1)]).matrices.len(),
            4,
            "last readers and new outputs at the same position coexist"
        );
        let mut resolved = GpuContextDemand::default();
        resolved.include_alternative(phase(1, 1));
        assert_eq!(
            GpuContextDemand::simultaneous([resolved, phase(0, 0)]).matrices.len(),
            2,
            "one resolved child must preserve its actual operation positions"
        );
        let mut escaping = phase(0, usize::MAX);
        escaping.include_alternative(phase(0, 0));
        assert_eq!(
            GpuContextDemand::simultaneous([escaping, phase(1, 1)]).matrices.len(),
            4,
            "an unresolved alternative preserves escaping owners"
        );
    }

    #[test]
    fn test_simultaneous_demand_preserves_shared_preparation_through_nested_merges() {
        use super::{GpuContextDemand, MatrixInputOwner};
        // Each sibling borrows the same input but owns a separate result.
        // Re-grouping the siblings must not allocate additional copies of the
        // shared normalization, or merge the equally shaped result owners.
        let body = |owner, ordinal, evaluation| {
            let mut demand = GpuContextDemand::default();
            demand.matrix(0, 1, 2, 3);
            demand.matrix_demands.last_mut().unwrap().preparation =
                Some((MatrixInputOwner::Actual(owner), ordinal, evaluation));
            demand.matrix(0, usize::MAX, 2, 3);
            demand
        };
        for count in 2..=8 {
            let flat = GpuContextDemand::simultaneous((0..count).map(|_| body(7, 0, true)));
            assert_eq!(flat.matrices.len(), count + 1);
            for split in 1..count {
                let nested = GpuContextDemand::simultaneous([
                    GpuContextDemand::simultaneous((0..split).map(|_| body(7, 0, true))),
                    GpuContextDemand::simultaneous((split..count).map(|_| body(7, 0, true))),
                ]);
                assert_eq!(nested.matrices.len(), flat.matrices.len());
                assert_eq!(
                    nested
                        .matrix_demands
                        .iter()
                        .filter(|claim| claim.preparation.is_some())
                        .count(),
                    1
                );
            }
        }
        for distinct in [body(8, 0, true), body(7, 1, true), body(7, 0, false)] {
            let merged = GpuContextDemand::simultaneous([
                GpuContextDemand::simultaneous([body(7, 0, true)]),
                distinct,
            ]);
            assert_eq!(
                merged.matrices.len(),
                4,
                "owner, fragment and format all belong to preparation identity"
            );
        }
    }

    #[test]
    fn test_simultaneous_workspace_demand_matches_live_intervals() {
        use super::{GpuContextDemand, event};
        let intervals = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
        // An independent occupancy oracle: count all inclusive live intervals
        // at each IR position, rather than reproducing slot reuse or matching.
        for a in intervals {
            for b in intervals {
                for c in intervals {
                    for d in intervals {
                        let groups = [[a, b], [c, d]];
                        let instances = groups.map(|intervals| {
                            let mut demand = GpuContextDemand::default();
                            for (begin, end) in intervals {
                                demand.workspace(begin, end, event(), false);
                            }
                            demand
                        });
                        let merged = GpuContextDemand::simultaneous(instances);
                        let peak = (0..=2)
                            .map(|position| {
                                groups
                                    .iter()
                                    .flatten()
                                    .filter(|&&(begin, end)| begin <= position && position <= end)
                                    .count()
                            })
                            .max()
                            .unwrap();
                        assert_eq!(merged.layouts.len(), peak, "{groups:?}");
                    }
                }
            }
        }
        let mut shared = GpuContextDemand::default();
        shared.workspace(0, 0, event(), true);
        shared.workspace(0, 0, event(), true);
        shared.workspace(1, 1, event(), true);
        assert_eq!(
            GpuContextDemand::simultaneous([shared.clone(), shared]).layouts.len(),
            2,
            "distinct shared batch claims must not collapse within a body"
        );
        let mut resolved = GpuContextDemand::default();
        resolved.workspace(1, 1, event(), false);
        let identical = resolved.clone();
        resolved.include_alternative(identical);
        let mut early = GpuContextDemand::default();
        early.workspace(0, 0, event(), false);
        assert_eq!(
            GpuContextDemand::simultaneous([resolved, early]).layouts.len(),
            1,
            "identical alternatives preserve exact lifetimes"
        );
        let mut escaping = GpuContextDemand::default();
        escaping.workspace(0, usize::MAX, event(), false);
        let mut transient = GpuContextDemand::default();
        transient.workspace(1, 1, event(), false);
        assert_eq!(
            GpuContextDemand::simultaneous([escaping, transient]).layouts.len(),
            2,
            "escaping resources cannot alias a later transient resource"
        );
    }

    #[test]
    fn test_alternative_demand_preserves_peak_counts_and_escaping_owners() {
        use super::{GpuContextDemand, event};
        let mut combined = GpuContextDemand::default();
        for iteration in 0..20 {
            let mut alternative = GpuContextDemand::default();
            alternative.matrix(0, usize::MAX, 2, 3);
            alternative.matrix(0, 1, 2, 4);
            alternative.workspace(0, 1, event(), true);
            if iteration % 2 == 1 {
                alternative.matrix(0, 1, 1, 1);
                alternative.workspace(0, usize::MAX, event(), true);
                alternative.workspace(0, 1, event(), false);
            }
            combined.include_alternative(alternative);
        }
        assert_eq!(combined.matrices.len(), 3);
        assert_eq!(combined.matrix_until.iter().filter(|&&end| end == usize::MAX).count(), 1);
        assert!(combined.matrices.contains(&(1, 1)));
        assert_eq!(combined.layouts.len(), 3);
        assert_eq!(combined.layout_shared.iter().filter(|&&shared| shared).count(), 2);
        assert_eq!(combined.layout_until.iter().filter(|&&end| end == usize::MAX).count(), 1);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_nested_warmup_covers_index_dependent_classes_without_reprobing_uses() {
        use mxx_ir_core::{IntExpr, RealExpr};
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let digits = params.modulus_digits();
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        for count in [2usize, 5] {
            // Bind a compile parameter through the supported IR loop binding.
            // Direct LoopIndex occurrences in structural wire types are illegal.
            let (body, public_type) = mxx_ir_core::with_new_construction_scope(|scope| {
                let public = ring
                    .sample_trapdoor(
                        1,
                        RealExpr::FromInt(IntExpr::Var("sigma".into())),
                        16,
                        digits,
                        100_000_000,
                    )
                    .public_matrix();
                let value = public.value_handle().clone();
                let ty = value.wire_type().clone();
                (
                    mxx_ir_core::SubgraphHandle::new("sample", scope, vec![], vec![value]).unwrap(),
                    ty,
                )
            });
            let outputs = mxx_ir_core::NodeHandle::parallel_loop(
                body,
                vec![],
                vec![mxx_ir_core::WireType::IndexedFamily {
                    element: Box::new(public_type),
                    count: count.into(),
                }],
                mxx_ir_core::node::ParallelLoop {
                    count: count.into(),
                    minimum_count: 0,
                    index_slot: 0,
                    bindings: vec![(
                        "sigma".into(),
                        IntExpr::Rem(Box::new(IntExpr::LoopIndex(0)), Box::new(2.into())) + 5,
                    )],
                    input_modes: vec![],
                },
            )
            .output(0)
            .unwrap();
            let graph = mxx_ir_core::Graph::freeze(
                "index-dependent-warmup-classes",
                vec![mxx_ir_core::graph::CompileParameter {
                    name: "sigma".into(),
                    kind: mxx_ir_core::graph::CompileParameterKind::Integer,
                }],
                BTreeMap::from([(
                    "public".into(),
                    mxx_ir_core::GraphOutput { value: outputs, confidentiality: None },
                )]),
                vec![],
                vec![],
                BTreeMap::new(),
            )
            .unwrap()
            .0;
            let graph = mxx_ir_core::validate(
                &graph,
                &ParamEnv {
                    integers: BTreeMap::from([("sigma".into(), 5.into())]),
                    ..ParamEnv::default()
                },
            )
            .unwrap();
            drop(
                backend.prepare_graph_admission(&graph, false, &BTreeMap::new(), 1, true).unwrap(),
            );
            assert_warmup_provisioned_no_graph_storage(&backend);
            let mut sigmas =
                backend.trapdoor_plans.keys().map(|key| key.sigma_bits).collect::<Vec<_>>();
            sigmas.sort_unstable();
            let mut expected = vec![5.0f64.to_bits(), 6.0f64.to_bits()];
            expected.sort_unstable();
            assert_eq!(sigmas, expected, "class count must not follow loop count");
            // Remove the nonzero-index class. Production lookup must fail at
            // that later alternative without silently performing its probe.
            let key = backend
                .trapdoor_plans
                .keys()
                .find(|key| key.sigma_bits == 6.0f64.to_bits())
                .unwrap()
                .clone();
            let saved = backend.trapdoor_plans.remove(&key).unwrap();
            let error = backend
                .graph_admission_demand(
                    &graph,
                    &FrozenGraphScopeId::Root,
                    &graph.bindings,
                    None,
                    false,
                    None,
                    1,
                    1,
                    false,
                )
                .err()
                .expect("later class must be required before execution");
            assert!(error.to_string().contains("explicit graph warmup required"), "{error}");
            assert_eq!(backend.trapdoor_plans.len(), 1);
            backend.trapdoor_plans.insert(key, saved);
        }
    }

    /// Setup wave selection is pure metadata evaluation, so its priority can be
    /// tested without a device. A graph that fits a wide wave only at a narrower
    /// scratch cap must still be provisioned for that wave: the search tries
    /// every cap of a wave before it drops to a smaller wave.
    #[test]
    fn test_gpu_setup_wave_search_prefers_a_wider_wave_that_fits_after_column_shrink() {
        // Monotone in both the simultaneous body count and the scratch width.
        let fits = |wave: usize, width: usize| wave * width <= 8;
        let probe = |wave: usize, width: usize| {
            Ok::<_, std::convert::Infallible>(fits(wave, width).then_some(wave * width))
        };
        assert_eq!(select_wave_and_width(4, 8, probe).unwrap(), Some((4, 2, 8)));
        // The historic priority could not see this candidate: it tried every
        // wave at the largest cap first and shrank the cap only at W = 1, which
        // accepts W = 1 here even though W = 4 fits at a legal cap of two.
        assert!(fits(1, 8));
        assert!(!fits(2, 8) && !fits(3, 8) && !fits(4, 8));
        assert!(fits(4, 2));
    }

    /// The largest wave wins over the widest cap, and the cap list is exactly
    /// the deduplicated halving chain of the graph's maximum width.
    #[test]
    fn test_gpu_setup_wave_search_visits_every_cap_of_each_wave_before_shrinking() {
        let mut visits = Vec::new();
        let exhausted = select_wave_and_width(3, 6, |wave: usize, width: usize| {
            visits.push((wave, width));
            Ok::<_, std::convert::Infallible>(None::<()>)
        })
        .unwrap();
        assert_eq!(exhausted, None);
        assert_eq!(
            visits,
            vec![
                (3, 6),
                (3, 3),
                (3, 2),
                (3, 1),
                (2, 6),
                (2, 3),
                (2, 2),
                (2, 1),
                (1, 6),
                (1, 3),
                (1, 2),
                (1, 1),
            ]
        );
        // A wave that only fits at the narrowest cap is still accepted, and the
        // widest cap of that wave is not revisited after acceptance.
        let mut accepted_visits = Vec::new();
        let accepted = select_wave_and_width(3, 6, |wave: usize, width: usize| {
            accepted_visits.push((wave, width));
            Ok::<_, std::convert::Infallible>((wave == 3 && width == 1).then_some((wave, width)))
        })
        .unwrap();
        assert_eq!(accepted, Some((3, 1, (3, 1))));
        assert_eq!(accepted_visits, vec![(3, 6), (3, 3), (3, 2), (3, 1)]);
        // The caller's bound is the search's upper wave, and a single-body bound
        // still scans the whole cap list.
        let single = select_wave_and_width(1, 2, |wave: usize, width: usize| {
            Ok::<_, std::convert::Infallible>((width <= 1).then_some((wave, width)))
        })
        .unwrap();
        assert_eq!(single, Some((1, 1, (1, 1))));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_inventory_mixed_family_layouts_survive_selection_and_child_calls() {
        use super::super::{GpuColumnShard, GpuFleetMatrix};
        use rayon::prelude::*;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let fragments = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(9)
            .max(9);
        let columns = fragments * 4;
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, columns, DistType::FinRingDist);
        let evaluation = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
        let host_bytes = std::sync::Arc::new(evaluation.clone().into_cpu_staging_bytes());
        let expected = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &(&original + &original))
            .to_compact_bytes();
        let members = [4usize, 1]
            .into_iter()
            .map(|width| {
                let shards = (0..columns / width)
                    .into_par_iter()
                    .map(|index| {
                        let start = index * width;
                        let mut value = GpuDCRTPolyMatrix::from_cpu_matrix(
                            &params,
                            &original.slice_columns(start, start + width),
                        );
                        value.intt_all_in_place();
                        GpuColumnShard { device_id: device, global_column_start: start, value }
                    })
                    .collect();
                RuntimeValue::matrix(GpuFleetMatrix::new(2, columns, shards))
            })
            .chain(std::iter::once(RuntimeValue::HostMatrix {
                matrix_type: mxx_ir_core::types::ConcreteMatrixType {
                    rows: 2,
                    columns,
                    ring_dimension: n as usize,
                    modulus: params.modulus().as_ref().clone().into(),
                },
                bytes: host_bytes,
            }))
            .collect::<Vec<_>>();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        for mode in ["direct", "child", "zero", "repeat"] {
            let context = DslContext::new("mixed-family-layout-inventory");
            let choice = context.int_family_input("choice", 1).at(0);
            let family = ring.input_family("members", members.len(), (2, columns));
            let selected = family.at(choice);
            let left = ring.input("a", (2, columns));
            let output = match mode {
                "child" => mxx_dsl::iterate(1, selected, |_, value| Ok(left + value)).unwrap(),
                "zero" => {
                    let coarse = ring.input("coarse", (2, columns));
                    let fine = ring.input("fine", (2, columns));
                    left + mxx_dsl::iterate(0, coarse, |_, value| Ok(value + fine)).unwrap()
                }
                "repeat" => {
                    let coarse = ring.input("coarse", (2, columns));
                    let fine = ring.input("fine", (2, columns));
                    mxx_dsl::iterate(3, coarse, |_, value| Ok(value + fine)).unwrap()
                }
                _ => left + selected,
            };
            let expected = if mode == "repeat" {
                let twice = &original + &original;
                GpuDCRTPolyMatrix::from_cpu_matrix(&params, &(&twice + &twice)).to_compact_bytes()
            } else {
                expected.clone()
            };
            let graph = context
                .output("sum", output)
                .unwrap()
                .build()
                .unwrap()
                .validate(&ParamEnv::default())
                .unwrap();
            for (choice, width) in [4usize, 1, columns]
                .into_iter()
                .enumerate()
                .take(if matches!(mode, "zero" | "repeat") { 1 } else { 3 })
            {
                let mut store = MemoryArtifactStore::default();
                let mut inputs =
                    BTreeMap::from([("a".into(), RuntimeValue::matrix(evaluation.clone().into()))]);
                if matches!(mode, "zero" | "repeat") {
                    inputs.insert("coarse".into(), members[0].clone());
                    inputs.insert("fine".into(), members[1].clone());
                } else {
                    inputs.insert("members".into(), RuntimeValue::IndexedFamily(members.clone()));
                    inputs.insert(
                        "choice".into(),
                        RuntimeValue::IndexedFamily(vec![RuntimeValue::Int(choice.into())]),
                    );
                }
                let mut result = execute_with_config(
                    &graph,
                    &mut backend,
                    inputs,
                    &mut store,
                    SamplingMode::Fresh,
                    ExecutionConfig::default(),
                )
                .unwrap();
                let RuntimeValue::Matrix(output) =
                    result.materialize_output("sum", &mut backend, &mut store).unwrap()
                else {
                    panic!("matrix output")
                };
                assert_eq!(
                    output.shards().len(),
                    if mode == "repeat" { columns } else { columns / width },
                    "capacity summaries must not change execution partitions"
                );
                assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected);
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_inventory_fused_blocks_retain_family_aliases() {
        use mxx_dsl::{Family, Mat};
        use mxx_ir_core::node::{ConcatAxis, IndexRange};
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let digits = params.modulus_digits();
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let input =
            Mat::concat(ConcatAxis::Rows, vec![ring.input("a", (1, 3)), ring.input("b", (1, 3))]);
        let left = Mat::concat(
            ConcatAxis::Rows,
            vec![ring.input("x", (1, 2 * digits)), ring.input("y", (1, 2 * digits))],
        );
        let product = input.decompose(16, digits).mul_small_rhs(left);
        let blocks = Family::pack(
            (0..2usize)
                .map(|row| {
                    product
                        .clone()
                        .slice(Some(IndexRange { start: row.into(), end: (row + 1).into() }), None)
                })
                .collect(),
        )
        .unwrap();
        let graph = DslContext::new("fused-block-family-lifetimes")
            .output("blocks", blocks)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let plan = crate::executor::gpu_plan::inventory_plan(&graph, false);
        assert_eq!(plan.outputs.len(), 1, "fixture must select the fused product");
        assert_eq!(plan.outputs.values().next().unwrap().len(), 2);
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        let (demand, _, _) = backend
            .graph_admission_demand(
                &graph,
                &FrozenGraphScopeId::Root,
                &ParamEnv::default(),
                None,
                false,
                None,
                3,
                1,
                true,
            )
            .unwrap();
        let retained_blocks = demand
            .values()
            .flat_map(|(_, demand)| demand.matrices.iter().zip(&demand.matrix_until))
            .filter(|(shape, until)| **shape == (1, 3) && **until == usize::MAX)
            .count();
        assert_eq!(
            retained_blocks, 2,
            "packing aliases into a returned family must retain both fused owners"
        );
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_inventory_lazy_family_demand_depends_on_selected_members() {
        use mxx_ir_core::artifact::{
            ArtifactConfidentiality, ArtifactType, Manifest, ManifestArtifact, ProductionId,
            SpecHash,
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let ty = mxx_ir_core::types::ConcreteMatrixType {
            rows: 2,
            columns: 3,
            ring_dimension: n as usize,
            modulus: params.modulus().as_ref().clone().into(),
        };
        let production =
            ProductionId { spec_hash: SpecHash(rand::random()), execution_nonce: rand::random() };
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        let mut inventories = Vec::new();
        for count in [2usize, 65_536] {
            let manifest = Manifest {
                ir_version: mxx_ir_core::encoding::IR_VERSION,
                production_id: production.clone(),
                artifacts: BTreeMap::from([(
                    "members".into(),
                    ManifestArtifact {
                        artifact_type: ArtifactType::Matrix(ty.clone()),
                        family_count: Some(count),
                        confidentiality: ArtifactConfidentiality::Private,
                        content_hash: None,
                        layout: None,
                    },
                )]),
            };
            let members = ring.family_artifact_input(
                production.clone(),
                "members",
                count,
                (2, 3),
                ArtifactConfidentiality::Private,
            );
            let graph = DslContext::new("lazy-family-inventory")
                .output("selected", -members.at(0))
                .unwrap()
                .build()
                .unwrap()
                .validate_with_manifests(
                    &ParamEnv::default(),
                    &BTreeMap::from([(production.clone(), manifest)]),
                )
                .unwrap();
            let (demand, _, _) = backend
                .graph_admission_demand(
                    &graph,
                    &FrozenGraphScopeId::Root,
                    &ParamEnv::default(),
                    None,
                    false,
                    None,
                    3,
                    1,
                    true,
                )
                .unwrap();
            inventories.push(
                demand
                    .into_values()
                    .map(|(_, demand)| (demand.matrices, demand.layouts))
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(inventories[0], inventories[1]);
        assert!(!inventories[0][0].0.is_empty(), "selected member still needs native owners");
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_inventory_lazy_scalar_packs_match_lazy_families() {
        use mxx_ir_core::artifact::{
            ArtifactConfidentiality, ArtifactType, Manifest, ManifestArtifact, ProductionId,
            SpecHash,
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let count = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(128)
            .max(3);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let ty = ConcreteMatrixType {
            rows: 2,
            columns: 3,
            ring_dimension: n as usize,
            modulus: params.modulus().as_ref().clone().into(),
        };
        let production =
            ProductionId { spec_hash: SpecHash(rand::random()), execution_nonce: rand::random() };
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        for (compact, select_candidates) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let mut inventories = Vec::new();
            for count in [2, count] {
                for packed in [false, true] {
                    let artifact_type = if compact {
                        ArtifactType::Preimage {
                            matrix: ty.clone(),
                            max_coefficient_bound: 7.into(),
                        }
                    } else {
                        ArtifactType::Matrix(ty.clone())
                    };
                    let names = if packed {
                        (0..count).map(|index| format!("member-{index}")).collect::<Vec<_>>()
                    } else {
                        vec!["members".into()]
                    };
                    let manifest = Manifest {
                        ir_version: mxx_ir_core::encoding::IR_VERSION,
                        production_id: production.clone(),
                        artifacts: names
                            .iter()
                            .map(|name| {
                                (
                                    name.clone(),
                                    ManifestArtifact {
                                        artifact_type: artifact_type.clone(),
                                        family_count: (!packed).then_some(count),
                                        confidentiality: ArtifactConfidentiality::Private,
                                        content_hash: None,
                                        layout: None,
                                    },
                                )
                            })
                            .collect(),
                    };
                    let result = if compact {
                        let members = if packed {
                            mxx_dsl::Family::pack(
                                names
                                    .iter()
                                    .map(|name| {
                                        ring.preimage_artifact_input(
                                            production.clone(),
                                            name.clone(),
                                            (2, 3),
                                            7,
                                            ArtifactConfidentiality::Private,
                                        )
                                    })
                                    .collect(),
                            )
                            .unwrap()
                        } else {
                            ring.preimage_family_artifact_input(
                                production.clone(),
                                "members",
                                count,
                                (2, 3),
                                7,
                                ArtifactConfidentiality::Private,
                            )
                        };
                        let selected = if select_candidates {
                            mxx_dsl::select(0, (0..count).map(|index| members.at(index)).collect())
                                .unwrap()
                        } else {
                            members.at(0)
                        };
                        selected.mul_small_rhs(ring.input("left", (1, 2)))
                    } else {
                        let members = if packed {
                            mxx_dsl::Family::pack(
                                names
                                    .iter()
                                    .map(|name| {
                                        ring.artifact_input(
                                            production.clone(),
                                            name.clone(),
                                            (2, 3),
                                            ArtifactConfidentiality::Private,
                                        )
                                    })
                                    .collect(),
                            )
                            .unwrap()
                        } else {
                            ring.family_artifact_input(
                                production.clone(),
                                "members",
                                count,
                                (2, 3),
                                ArtifactConfidentiality::Private,
                            )
                        };
                        let selected = if select_candidates {
                            mxx_dsl::select(0, (0..count).map(|index| members.at(index)).collect())
                                .unwrap()
                        } else {
                            members.at(0)
                        };
                        -selected
                    };
                    let graph = DslContext::new("scalar-artifact-pack-inventory")
                        .output("result", result)
                        .unwrap()
                        .build()
                        .unwrap()
                        .validate_with_manifests(
                            &ParamEnv::default(),
                            &BTreeMap::from([(production.clone(), manifest)]),
                        )
                        .unwrap();
                    let (demand, _, _) = backend
                        .graph_admission_demand(
                            &graph,
                            &FrozenGraphScopeId::Root,
                            &ParamEnv::default(),
                            None,
                            false,
                            None,
                            3,
                            1,
                            true,
                        )
                        .unwrap();
                    inventories.push(
                        demand
                            .into_values()
                            .map(|(_, demand)| (demand.matrices, demand.layouts))
                            .collect::<Vec<_>>(),
                    );
                }
            }
            if !compact {
                // A mixed packed family is eagerly placed when broadcast,
                // including lazy members not selected by the child body.
                let names = (0..count).map(|index| format!("mixed-{index}")).collect::<Vec<_>>();
                let manifest = Manifest {
                    ir_version: mxx_ir_core::encoding::IR_VERSION,
                    production_id: production.clone(),
                    artifacts: names
                        .iter()
                        .map(|name| {
                            (
                                name.clone(),
                                ManifestArtifact {
                                    artifact_type: ArtifactType::Matrix(ty.clone()),
                                    family_count: None,
                                    confidentiality: ArtifactConfidentiality::Private,
                                    content_hash: None,
                                    layout: None,
                                },
                            )
                        })
                        .collect(),
                };
                let members = mxx_dsl::Family::pack(
                    std::iter::once(ring.input("eager", (2, 3)))
                        .chain(names.iter().map(|name| {
                            ring.artifact_input(
                                production.clone(),
                                name.clone(),
                                (2, 3),
                                ArtifactConfidentiality::Private,
                            )
                        }))
                        .collect(),
                )
                .unwrap();
                // Direct and offset indices lower to Zip/ZipOffset. A runtime
                // remainder keeps the whole family as a broadcast input.
                let result = mxx_dsl::parallel(2, |index| Ok(-members.at(index % 2))).unwrap();
                let graph = DslContext::new("mixed-family-broadcast")
                    .output("result", result)
                    .unwrap()
                    .build()
                    .unwrap()
                    .validate_with_manifests(
                        &ParamEnv::default(),
                        &BTreeMap::from([(production.clone(), manifest)]),
                    )
                    .unwrap();
                assert!(graph.root_scope().execution_order.iter().any(|node| {
                    matches!(node.kind(), mxx_ir_core::node::NodeKind::ParallelLoop(body)
                        if body.input_modes.contains(&mxx_ir_core::node::LoopInputMode::Broadcast))
                }));
                let (demand, _, _) = backend
                    .graph_admission_demand(
                        &graph,
                        &FrozenGraphScopeId::Root,
                        &ParamEnv::default(),
                        None,
                        false,
                        None,
                        3,
                        1,
                        true,
                    )
                    .unwrap();
                let owners = demand
                    .values()
                    .flat_map(|(_, demand)| &demand.matrices)
                    .filter(|&&(rows, columns)| rows >= 2 && columns >= 3)
                    .count();
                assert!(
                    owners >= count,
                    "broadcast placement must reserve every lazy member of a mixed packed family: {owners} < {count}"
                );
            }
            for inventory in &inventories[1..] {
                assert_eq!(
                    &inventories[0], inventory,
                    "lazy scalar packs must reserve only selected members; compact={compact}"
                );
            }
            assert!(
                inventories[0].iter().any(|(_, layouts)| layouts
                    .iter()
                    .any(|layout| layout.kind == GpuPreparedSlotKind::PinnedHost)),
                "materialization still requires native import staging"
            );
        }
    }

    /// A lazy matrix artifact captured as a Broadcast input by every body of a
    /// loop is loaded once for the whole loop: each body's capture input is
    /// bound to the loop-level owner the parent materialized, so a loop of N
    /// bodies performs one import, not 1 + N. Trusted arithmetic predicts the
    /// negation of the stored matrix for every member.
    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_loop_broadcast_of_a_lazy_input_loads_once() {
        use crate::artifact::{ArtifactKey, ArtifactPayload, ArtifactStore};
        use mxx_ir_core::artifact::{
            ArtifactConfidentiality, ArtifactType, Manifest, ManifestArtifact, ProductionId,
            SpecHash,
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let count = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(5);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let ty = ConcreteMatrixType {
            rows: 2,
            columns: 3,
            ring_dimension: n as usize,
            modulus: params.modulus().as_ref().clone().into(),
        };
        let production =
            ProductionId { spec_hash: SpecHash(rand::random()), execution_nonce: rand::random() };
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 3, DistType::FinRingDist);
        let mut store = MemoryArtifactStore::default();
        store
            .store(
                ArtifactKey { production: production.clone(), name: "input".into(), index: None },
                &ArtifactType::Matrix(ty.clone()),
                ArtifactConfidentiality::Private,
                None,
                ArtifactPayload::Matrix(
                    GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original).to_compact_bytes(),
                ),
            )
            .unwrap();
        let manifest = Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([(
                "input".into(),
                ManifestArtifact {
                    artifact_type: ArtifactType::Matrix(ty),
                    family_count: None,
                    confidentiality: ArtifactConfidentiality::Private,
                    content_hash: None,
                    layout: None,
                },
            )]),
        };
        store.store_manifest(manifest.clone()).unwrap();
        let input = ring.artifact_input(
            production.clone(),
            "input",
            (2, 3),
            ArtifactConfidentiality::Private,
        );
        let negated = mxx_dsl::parallel(count, |_| Ok(-input.clone())).unwrap();
        let graph = DslContext::new("gpu-loop-broadcast-import-once")
            .output("negated", negated)
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production.clone(), manifest)]),
            )
            .unwrap();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        let cap = std::env::var("MXX_PRIMITIVE_TEST_WAVE_BOUND")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(4)
            .max(2)
            .min(count - 1);
        drop(backend.prepare_graph_admission(&graph, false, &BTreeMap::new(), cap, true).unwrap());
        let (sender, receiver) = std::sync::mpsc::channel();
        backend.set_admitted_measurement_sink(Some(Box::new(move |event| {
            sender.send(event).unwrap();
        })));
        let mut result = execute_with_config(
            &graph,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig {
                max_parallel_instances: std::num::NonZeroUsize::new(cap).expect("nonzero"),
                ..ExecutionConfig::default()
            },
        )
        .unwrap();
        let RuntimeValue::IndexedFamily(members) =
            result.materialize_output("negated", &mut backend, &mut store).unwrap()
        else {
            panic!("broadcast result is an indexed family")
        };
        assert_eq!(members.len(), count);
        let expected = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &(-original)).to_compact_bytes();
        for member in members {
            let RuntimeValue::Matrix(matrix) = member else { panic!("matrix member") };
            assert_eq!(backend.matrix_to_bytes(&matrix).unwrap(), expected);
        }
        let first_wave = receiver.try_iter().find_map(|event| match event {
            crate::gpu_measurement::GpuAdmittedMeasurement::Admission { instances, .. } => {
                Some(instances)
            }
            _ => None,
        });
        assert_eq!(first_wave, Some(cap), "cold broadcast import must not shrink the first wave");
        let key = ArtifactKey { production, name: "input".into(), index: None };
        assert_eq!(
            store.load_count(&key),
            1,
            "one loop-level import must serve every body of the loop"
        );
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_inventory_lazy_input_cached_across_intervening_operations() {
        use crate::artifact::{ArtifactKey, ArtifactPayload, ArtifactStore};
        use mxx_ir_core::artifact::{
            ArtifactConfidentiality, ArtifactType, Manifest, ManifestArtifact, ProductionId,
            SpecHash,
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let ty = ConcreteMatrixType {
            rows: 2,
            columns: 3,
            ring_dimension: n as usize,
            modulus: params.modulus().as_ref().clone().into(),
        };
        let production =
            ProductionId { spec_hash: SpecHash(rand::random()), execution_nonce: rand::random() };
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 3, DistType::FinRingDist);
        let bytes = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original).to_compact_bytes();
        let mut store = MemoryArtifactStore::default();
        store
            .store(
                ArtifactKey { production: production.clone(), name: "input".into(), index: None },
                &ArtifactType::Matrix(ty.clone()),
                ArtifactConfidentiality::Private,
                None,
                ArtifactPayload::Matrix(bytes.clone()),
            )
            .unwrap();
        let manifest = Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([(
                "input".into(),
                ManifestArtifact {
                    artifact_type: ArtifactType::Matrix(ty),
                    family_count: None,
                    confidentiality: ArtifactConfidentiality::Private,
                    content_hash: None,
                    layout: None,
                },
            )]),
        };
        store.store_manifest(manifest.clone()).unwrap();
        let input = ring.artifact_input(
            production.clone(),
            "input",
            (2, 3),
            ArtifactConfidentiality::Private,
        );
        let mut intermediate = -mxx_dsl::select(0, vec![input.clone(), input.clone()]).unwrap();
        let count = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(8)
            .max(2);
        for _ in 0..count {
            intermediate = -(-intermediate);
        }
        // The second access must read the cached imported owner after temporary
        // slots have been reused repeatedly. Trusted arithmetic predicts input.
        let output = (intermediate + input.clone()) + input;
        let graph = DslContext::new("cached-lazy-input-lifetime")
            .output("result", output)
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production.clone(), manifest.clone())]),
            )
            .unwrap();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        drop(backend.prepare_graph_admission(&graph, false, &BTreeMap::new(), 1, true).unwrap());
        let mut result = execute_with_config(
            &graph,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig { prepared_gpu_admission: true, ..ExecutionConfig::default() },
        )
        .unwrap();
        let RuntimeValue::Matrix(output) =
            result.materialize_output("result", &mut backend, &mut store).unwrap()
        else {
            panic!("matrix result");
        };
        assert_eq!(backend.matrix_to_bytes(&output).unwrap(), bytes);
        drop(result);
        drop(backend);

        // Returned lazy descriptors must also be importable when the caller
        // requests the whole output family after graph execution.
        let input = ring.artifact_input(
            production.clone(),
            "input",
            (2, 3),
            ArtifactConfidentiality::Private,
        );
        let family = mxx_dsl::Family::pack(vec![input; count]).unwrap();
        let graph = DslContext::new("retained-lazy-output-family")
            .output("result", family)
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production, manifest)]),
            )
            .unwrap();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        drop(backend.prepare_graph_admission(&graph, false, &BTreeMap::new(), 1, true).unwrap());
        let mut result = execute_with_config(
            &graph,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        let RuntimeValue::IndexedFamily(members) =
            result.materialize_output("result", &mut backend, &mut store).unwrap()
        else {
            panic!("family result")
        };
        assert_eq!(members.len(), count);
        for member in members {
            let RuntimeValue::Matrix(matrix) = member else { panic!("matrix member") };
            assert_eq!(backend.matrix_to_bytes(matrix).unwrap(), bytes);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_inventory_zero_iteration_outputs_keep_carried_owners() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 3, DistType::FinRingDist);
        let expected = [&original, &(-original.clone())]
            .map(|value| GpuDCRTPolyMatrix::from_cpu_matrix(&params, value).to_compact_bytes());
        let input = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
        let count = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(9)
            .max(5);
        let mut context = DslContext::new("zero-iteration-alias-inventory");
        let mut value = ring.input("a", (2, 3));
        for index in 0..count {
            value = -value;
            let alias = mxx_dsl::iterate(0, value.clone(), |_, carried| Ok(-carried)).unwrap();
            context = context.output(format!("value-{index:04}"), alias).unwrap();
        }
        let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        let mut store = MemoryArtifactStore::default();
        let mut result = execute_with_config(
            &graph,
            &mut backend,
            BTreeMap::from([("a".into(), RuntimeValue::matrix(input.into()))]),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        for index in 0..count {
            let RuntimeValue::Matrix(output) = result
                .materialize_output(&format!("value-{index:04}"), &mut backend, &mut store)
                .unwrap()
            else {
                panic!("matrix output")
            };
            assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected[(index + 1) % 2]);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_graph_inventory_preserves_fragmented_coefficient_inputs() {
        use super::super::{GpuColumnShard, GpuFleetMatrix};
        use rayon::prelude::*;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(17)
            .max(9);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let graph = DslContext::new("fragmented-input-inventory")
            .output("result", ring.input("a", (2, columns)) + ring.input("b", (2, columns)))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let input =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, columns, DistType::FinRingDist);
        let expected =
            GpuDCRTPolyMatrix::from_cpu_matrix(&params, &(&input + &input)).to_compact_bytes();
        let evaluation = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &input);
        let shards = (0..columns)
            .into_par_iter()
            .map(|start| {
                let mut value = GpuDCRTPolyMatrix::from_cpu_matrix(
                    &params,
                    &input.slice_columns(start, start + 1),
                );
                value.intt_all_in_place();
                GpuColumnShard { device_id: device, global_column_start: start, value }
            })
            .collect();
        let input = GpuFleetMatrix::new(2, columns, shards);
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        let mut store = MemoryArtifactStore::default();
        let mut result = execute_with_config(
            &graph,
            &mut backend,
            BTreeMap::from([
                ("a".into(), RuntimeValue::matrix(evaluation.into())),
                ("b".into(), RuntimeValue::matrix(input.clone())),
            ]),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        let RuntimeValue::Matrix(output) =
            result.materialize_output("result", &mut backend, &mut store).unwrap()
        else {
            panic!("matrix result")
        };
        assert_eq!(output.shards().len(), columns);
        assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected);
        assert!(input.shards().iter().all(|shard| !shard.value.is_ntt()));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_loop_inventory_scales_with_live_wave_not_iteration_count() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let input = DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 3, DistType::FinRingDist);
        let expected =
            GpuDCRTPolyMatrix::from_cpu_matrix(&params, &(-input.clone())).to_compact_bytes();
        let input = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &input);
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        let mut capacities = Vec::new();
        let mut diagnostic_plan_counts = Vec::new();
        // Both counts exceed the same live-wave cap, so staging and
        // simultaneous ownership are identical; only total iterations differ.
        for count in [3usize, 65] {
            let matrix = ring.input("matrix", (2, 3));
            let outputs = mxx_dsl::parallel(count, |_| Ok(-matrix.clone())).unwrap();
            let graph = DslContext::new("bounded-loop-inventory")
                .output("result", outputs.at(count - 1))
                .unwrap()
                .build()
                .unwrap()
                .validate(&ParamEnv::default())
                .unwrap();
            let inputs =
                BTreeMap::from([("matrix".into(), RuntimeValue::matrix(input.clone().into()))]);
            let mut store = MemoryArtifactStore::default();
            let mut result = execute_with_config(
                &graph,
                &mut backend,
                inputs,
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig {
                    max_parallel_instances: std::num::NonZeroUsize::new(2).unwrap(),
                    ..ExecutionConfig::default()
                },
            )
            .unwrap();
            assert_eq!(backend.graph_wave, 2, "compare the same provisioned live wave");
            let RuntimeValue::Matrix(output) =
                result.materialize_output("result", &mut backend, &mut store).unwrap()
            else {
                panic!("matrix result")
            };
            assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected);
            diagnostic_plan_counts.push(backend.admitted_plan_log().len());
            capacities.push(
                backend
                    .prepared_ledger
                    .as_ref()
                    .unwrap()
                    .prepared_inventory()
                    .map(|(_, storage)| storage.occupancy().unwrap().requested_capacity_bytes())
                    .sum::<usize>(),
            );
        }
        assert_eq!(
            capacities[0], capacities[1],
            "loop count must not multiply GPU scratch backing"
        );
        assert!(diagnostic_plan_counts[0] > 0);
        assert_eq!(
            diagnostic_plan_counts[0], diagnostic_plan_counts[1],
            "repeated loop members must not accumulate identical diagnostic plans"
        );
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_live_wave_shrinks_under_slot_pressure_and_recovers() {
        unsafe extern "C" {
            fn gpu_test_context_streams_gate(
                context: *mut std::ffi::c_void,
            ) -> *mut std::ffi::c_void;
            fn gpu_test_release_stream_gate(gate: *mut std::ffi::c_void);
        }
        struct Gate(*mut std::ffi::c_void);
        impl Drop for Gate {
            fn drop(&mut self) {
                unsafe { gpu_test_release_stream_gate(self.0) };
            }
        }
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(3);
        let wave = std::env::var("MXX_PRIMITIVE_TEST_WAVE_BOUND")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(3)
            .max(2);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let matrix = ring.input("matrix", (2, columns));
        let members = mxx_dsl::parallel(wave, |_| Ok(-matrix.clone())).unwrap();
        let graph = DslContext::new("live-wave-pressure")
            .output("members", members)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let sample =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, columns, DistType::FinRingDist);
        // The reader is independent of the prepared inventory and begins with
        // trusted random input; adding prepared zeros must preserve it.
        let mut reader = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &sample);
        reader.wait_until_ready();
        let inputs = BTreeMap::from([(
            "matrix".to_owned(),
            RuntimeValue::matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &sample).into()),
        )]);
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        let guard = backend.prepare_graph_admission(&graph, false, &inputs, wave, false).unwrap();
        let scope_id = FrozenGraphScopeId::Root;
        let scope = graph.source.scope(&scope_id).unwrap();
        let (position, handle) = graph
            .root_scope()
            .execution_order
            .iter()
            .enumerate()
            .find(|(_, handle)| {
                matches!(handle.kind(), mxx_ir_core::node::NodeKind::ParallelLoop(_))
            })
            .unwrap();
        let loop_id = mxx_ir_core::types::NodeId(position as u64);
        let child_id = graph.source.child_scope_id(&scope_id, loop_id).unwrap();
        let child = graph.source.scope(&child_id).unwrap();
        let arguments = scope.arguments(handle).unwrap();
        let values = BTreeMap::from([(arguments[0], inputs["matrix"].clone())]);
        let request = crate::executor::WaveAdmissionRequest::new(
            &graph,
            &scope_id,
            loop_id,
            &child_id,
            child.inputs(),
            child.outputs(),
            &arguments,
            &graph.bindings,
            &values,
            None,
            false,
            wave,
            0,
            wave,
        );
        let admitted = backend.admit_wave(&request).unwrap();
        assert_eq!(admitted.wave_size, wave);
        let operations = backend.admitted_scope_operations.last().unwrap().clone();
        assert_eq!(operations.upgrade().unwrap().instances.len(), wave);
        let RuntimeValue::Matrix(input) = &inputs["matrix"] else { unreachable!() };
        assert!(
            std::sync::Arc::ptr_eq(
                &operations.upgrade().unwrap().input_layouts[&input.id],
                &input.input_layout
            ),
            "admission shares the layout published by this actual owner"
        );
        drop(admitted);
        assert!(operations.upgrade().is_none(), "operation bindings end with their admitted wave");
        let inventory =
            backend.prepared_ledger.as_ref().unwrap().prepared_inventory().collect::<Vec<_>>();
        let mut blockers = Vec::new();
        let mut shrunk = None;
        'pressure: for (_, storage) in inventory {
            for index in 0..storage.slot_count() {
                let slot = storage.slot_identity(index).unwrap();
                if slot.kind() != GpuPreparedSlotKind::Matrix ||
                    slot.rows() != 2 ||
                    slot.columns() != columns
                {
                    continue;
                }
                let claim = slot.matrix_request(slot.rows(), slot.columns(), true);
                if !storage.fits(&[claim]).unwrap() {
                    continue;
                }
                blockers.push((storage.clone(), index, storage.reserve(&[claim]).unwrap()));
                if let Ok(admission) = backend.admit_wave(&request) {
                    if admission.wave_size < wave {
                        shrunk = Some(admission.wave_size);
                        break 'pressure;
                    }
                }
            }
        }
        assert!(shrunk.is_some(), "live typed pressure must reduce an otherwise wider wave");
        // Turn the held leases into actual sources, then drop their Rust
        // owners while a real GPU consumer is deterministically blocked. This
        // exercises runtime admission's pending-reader exclusion, not only
        // the native poll API or a manually retained lease.
        let pending_slots = blockers
            .iter()
            .map(|(storage, index, _)| (storage.clone(), *index))
            .collect::<Vec<_>>();
        let sources = blockers
            .into_iter()
            .map(|(_, _, reservation)| {
                let dispatch = reservation.enter(Vec::new()).unwrap();
                let source = GpuDCRTPolyMatrix::zero(&params, 2, columns);
                drop(dispatch.finish().unwrap());
                source.wait_until_ready();
                source
            })
            .collect::<Vec<_>>();
        // This extra reader belongs to the fixture, not the protocol being
        // admitted. Enqueue it outside the graph seal, then restore the seal
        // before testing the production admission path.
        drop(guard);
        // Declared after every GPU owner: unwinding opens the streams first.
        let gate = Gate(unsafe {
            gpu_test_context_streams_gate(params.context_identity() as *mut std::ffi::c_void)
        });
        assert!(!gate.0.is_null(), "install the test-only compute-stream gate");
        for source in sources {
            reader.add_in_place(&source);
            drop(source);
        }
        for (storage, index) in &pending_slots {
            assert!(storage.poll_releases(&[]).unwrap().contains(index));
        }
        let guard = mxx_primitives::matrix::gpu_dcrt_poly::GpuGraphAdmissionGuard::new(
            backend.device_parameters(),
        )
        .unwrap();
        let pending_admission = backend.admit_wave(&request).unwrap();
        assert_eq!(
            Some(pending_admission.wave_size),
            shrunk,
            "pending readers constrain W exactly as the same live owners did"
        );
        drop(pending_admission);
        drop(gate);
        for (storage, index) in &pending_slots {
            storage.poll_releases(&[*index]).unwrap();
        }
        let recovered = backend.admit_wave(&request).unwrap();
        assert_eq!(recovered.wave_size, wave, "restored capacity is reconsidered next admission");
        drop(recovered);
        drop(guard);
        assert_eq!(reader.to_cpu_matrix(), sample, "delayed readers preserve the input");
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_loop_admits_the_configured_wave_with_a_tail() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        // Small configured concurrency bound: the fixture must be able to see a
        // real multi-body wave without provisioning a large backing.
        let wave = std::env::var("MXX_PRIMITIVE_TEST_WAVE_BOUND")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(2)
            .max(2);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let sample =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 3, DistType::FinRingDist);
        let left = DCRTPolyUniformSampler::new().sample_uniform(&cpu, 3, 2, DistType::FinRingDist);
        let expected = GpuDCRTPolyMatrix::from_cpu_matrix(
            &params,
            &(-left.multiply_out_of_place(&sample)).ring_automorphism_out_of_place(3),
        )
        .to_compact_bytes();
        let mut capacities = Vec::new();
        let mut observed = Vec::new();
        // `wave` bodies fit one wave exactly; `wave + 1` adds a one-body tail.
        for count in [wave, wave + 1] {
            let input = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &sample);
            let matrix = ring.input("matrix", (2, 3));
            let factor = ring.input("factor", (3, 2));
            let bias = ring.input("bias", (3, 3));
            let members = mxx_dsl::parallel(count, |_| {
                let added = matrix.clone() + matrix.clone();
                let restored = added - matrix.clone();
                let product = factor.clone() * restored.clone();
                // Five terms cross the native four-term descriptor chunk.
                // Coefficients sum to one; subtracting the random bias makes
                // the trusted product oracle independent of the fused runner.
                let fused = mxx_dsl::Mat::multi_row_gemm_accumulate(
                    [3, -2, 5, -4, -1]
                        .into_iter()
                        .map(|coefficient| (coefficient, factor.clone(), restored.clone()))
                        .collect(),
                    Some(bias.clone()),
                );
                let product = (fused - bias.clone()) + (product.clone() - product);
                let left_scaled = ring.identity(1) * product;
                let right_scaled = left_scaled * ring.identity(1);
                Ok((-right_scaled).ring_automorphism(3))
            })
            .unwrap();
            let graph = DslContext::new("configured-wave-loop")
                .output("members", members)
                .unwrap()
                .build()
                .unwrap()
                .validate(&ParamEnv::default())
                .unwrap();
            let inputs = BTreeMap::from([
                ("matrix".into(), RuntimeValue::matrix(input.into())),
                (
                    "bias".into(),
                    RuntimeValue::matrix(
                        GpuDCRTPolyMatrix::from_cpu_matrix(
                            &params,
                            &DCRTPolyUniformSampler::new().sample_uniform(
                                &cpu,
                                3,
                                3,
                                DistType::FinRingDist,
                            ),
                        )
                        .into(),
                    ),
                ),
                (
                    "factor".into(),
                    RuntimeValue::matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &left).into()),
                ),
            ]);
            let mut store = MemoryArtifactStore::default();
            let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
            let (sender, receiver) = std::sync::mpsc::channel();
            backend.set_admitted_measurement_sink(Some(Box::new(move |event| {
                sender.send(event).unwrap();
            })));
            let mut result = execute_with_config(
                &graph,
                &mut backend,
                inputs,
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig {
                    max_parallel_instances: std::num::NonZeroUsize::new(wave).expect("nonzero"),
                    prepared_gpu_admission: true,
                    ..ExecutionConfig::default()
                },
            )
            .unwrap();
            backend.set_admitted_measurement_sink(None);
            // Distinct body counts of the loop body's own scope, in the order
            // the executor ran its waves. Root-scope nodes always run one
            // instance and are not part of a bounded body wave.
            let mut wave_sizes = Vec::new();
            let mut submitted_wave_sizes = Vec::new();
            let mut measured_plans = Vec::<crate::gpu_memory::GpuAdmittedPlanSummary>::new();
            let mut measured_wave = 0;
            for event in receiver {
                if let crate::gpu_measurement::GpuAdmittedMeasurement::Invocation {
                    ref plans,
                    ..
                } = event
                {
                    measured_plans = plans.clone();
                    measured_wave = 0;
                }
                if let crate::gpu_measurement::GpuAdmittedMeasurement::Wave { ref jobs, .. } = event
                {
                    submitted_wave_sizes.push(jobs.len());
                    let mut interval_offset = 0;
                    let expected = measured_plans
                        .iter()
                        .flat_map(|plan| {
                            let jobs = plan
                                .schedule
                                .wave_jobs(measured_wave)
                                .into_iter()
                                .map(|mut job| {
                                    job.source_interval += interval_offset;
                                    job
                                })
                                .collect::<Vec<_>>();
                            interval_offset += plan.schedule.intervals().len();
                            jobs
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        *jobs, expected,
                        "batch timing must retain every sibling's admitted ranges"
                    );
                    measured_wave += 1;
                }
                let crate::gpu_measurement::GpuAdmittedMeasurement::Node(node) = event else {
                    continue;
                };
                if node.scope == FrozenGraphScopeId::Root {
                    continue;
                }
                if wave_sizes.last().copied() != Some(node.instances) {
                    wave_sizes.push(node.instances);
                }
            }
            assert!(
                submitted_wave_sizes.contains(&wave),
                "arithmetic siblings must reach one prepared device submission, not scalar calls: {submitted_wave_sizes:?}"
            );
            let RuntimeValue::IndexedFamily(members) =
                result.materialize_output("members", &mut backend, &mut store).unwrap()
            else {
                panic!("loop result is an indexed family")
            };
            assert_eq!(members.len(), count);
            for member in members {
                let RuntimeValue::Matrix(matrix) = member else { panic!("matrix member") };
                assert_eq!(backend.matrix_to_bytes(&matrix).unwrap(), expected);
            }
            capacities.push(
                backend
                    .prepared_ledger
                    .as_ref()
                    .unwrap()
                    .prepared_inventory()
                    .map(|(_, storage)| storage.occupancy().unwrap().requested_capacity_bytes())
                    .sum::<usize>(),
            );
            observed.push(wave_sizes);
        }
        assert_eq!(
            observed[0],
            vec![wave],
            "a loop of exactly one configured wave must run every body together"
        );
        assert_eq!(
            observed[1],
            vec![wave, 1],
            "a longer loop must keep the configured wave and admit the tail separately"
        );
        assert!(
            observed[0].iter().max() > Some(&1),
            "the fixture must observe a real multi-body wave"
        );
        assert_eq!(
            capacities[0], capacities[1],
            "prepared backing must follow the configured wave, not the loop count"
        );
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_admitted_polynomial_values_preserve_both_domains() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let expected =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 1, 1, DistType::FinRingDist);
        let expected_bytes =
            GpuDCRTPolyMatrix::from_cpu_matrix(&params, &expected).to_compact_bytes();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        for evaluation in [false, true] {
            let context = DslContext::new("admitted-polynomial-values");
            let values = context.int_family_input("values", n as usize);
            let output = if evaluation {
                ring.from_evaluations(&values)
            } else {
                ring.from_coefficients(&values)
            };
            let graph = context
                .output("result", output)
                .unwrap()
                .build()
                .unwrap()
                .validate(&ParamEnv::default())
                .unwrap();
            let polynomial = expected.entry(0, 0);
            let values =
                if evaluation { polynomial.evals_biguints() } else { polynomial.coeffs_biguints() };
            let inputs = BTreeMap::from([(
                "values".into(),
                RuntimeValue::IndexedFamily(
                    values.into_iter().map(|value| RuntimeValue::Int(value.into())).collect(),
                ),
            )]);
            let mut store = MemoryArtifactStore::default();
            let mut result = execute_with_config(
                &graph,
                &mut backend,
                inputs,
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig::default(),
            )
            .unwrap();
            let RuntimeValue::Matrix(output) =
                result.materialize_output("result", &mut backend, &mut store).unwrap()
            else {
                panic!("matrix result")
            };
            assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected_bytes);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_graph_hash_decomposition_uses_compiled_resource_layout() {
        use mxx_primitives::{
            matrix::{PolyMatrixSmallRhs, SmallPolyMatrix},
            sampler::{PolyHashSampler, gpu::GpuDCRTPolyHashSampler},
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(3);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        for small in [false, true] {
            let params = GpuDCRTPolyParams::new_with_gpu(
                n,
                cpu.to_crt().0,
                4,
                vec![device],
                Some(1),
                None,
                None,
            );
            let layout = params.compact_decomposition_layout(small, None).unwrap();
            let digits = layout.rows_per_input_row;
            let key: [u8; 32] = rand::random();
            let tag = b"compiled-hash-resource-layout";
            let expected = GpuDCRTPolyHashSampler::<keccak_asm::Keccak256>::new()
                .sample_hash_gadget_source(&params, key, tag, 2, columns, DistType::FinRingDist)
                .gadget_decompose(small, Some(digits))
                .unwrap()
                .to_canonical_coefficients()
                .unwrap();
            let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
            let key_input = ring.bytes_input("key", 32);
            let output = if small {
                ring.hash_small_decomposed(
                    key_input,
                    tag.as_slice(),
                    (2 * digits, columns),
                    16,
                    digits,
                )
            } else {
                ring.hash_decomposed(key_input, tag.as_slice(), (2 * digits, columns), 16, digits)
            };
            let graph = DslContext::new("compiled-hash-resources")
                .output("output", output)
                .unwrap()
                .build()
                .unwrap()
                .validate(&ParamEnv::default())
                .unwrap();
            let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
            let mut store = MemoryArtifactStore::default();
            let mut result = execute_with_config(
                &graph,
                &mut backend,
                BTreeMap::from([("key".into(), RuntimeValue::Bytes(key.to_vec()))]),
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig { prepared_gpu_admission: true, ..ExecutionConfig::default() },
            )
            .unwrap();
            let RuntimeValue::SmallMatrix(output) =
                result.materialize_output("output", &mut backend, &mut store).unwrap()
            else {
                panic!("compact hash result")
            };
            assert_eq!(output.shards().len(), 1, "one fresh output owner on this device");
            assert_eq!(output.shards()[0].value.to_canonical_coefficients().unwrap(), expected);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_graph_range_scratch_retains_complete_outputs() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(7);
        assert!(columns >= 3);
        let width = columns.div_ceil(3);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let digits = params.modulus_digits();
        let value = ring.input("a", (2, columns));
        let result = value.decompose(16, digits).mul_small_rhs(ring.gadget(2, 16, digits));
        let graph = DslContext::new("prepared-range-scratch")
            .output("result", result)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let input =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, columns, DistType::FinRingDist);
        let matrix = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &input);
        let expected = matrix.to_compact_bytes();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        // Exercise the same demand/provisioning passes with a deliberately
        // smaller scratch envelope. This avoids allocating a VRAM-sized test.
        let demand = backend
            .graph_admission_demand(
                &graph,
                &FrozenGraphScopeId::Root,
                &graph.bindings,
                None,
                false,
                None,
                width,
                1,
                true,
            )
            .unwrap()
            .0;
        backend.prepare_graph_storage(demand).unwrap();
        let mut store = MemoryArtifactStore::default();
        let mut result = execute_with_config(
            &graph,
            &mut backend,
            BTreeMap::from([("a".into(), RuntimeValue::matrix(matrix.into()))]),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig { prepared_gpu_admission: true, ..ExecutionConfig::default() },
        )
        .unwrap();
        let product = backend
            .admitted_plan_log()
            .iter()
            .map(|(_, invocation)| invocation)
            .find(|invocation| invocation.operation == "MultiplyCompact")
            .unwrap();
        assert_eq!(product.plan.columns, columns);
        assert_eq!(product.plan.widths, vec![width]);
        assert_eq!(product.plan.wave_count, columns.div_ceil(width));
        let RuntimeValue::Matrix(output) =
            result.materialize_output("result", &mut backend, &mut store).unwrap()
        else {
            panic!("matrix output");
        };
        assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_measured_ranges_reuse_representatives_and_measure_the_tail() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(7);
        assert!(columns >= 3);
        let width = columns.div_ceil(3);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let digits = params.modulus_digits();
        let value = ring.input("a", (2, columns));
        let result = value.decompose(16, digits).mul_small_rhs(ring.gadget(2, 16, digits));
        let graph = DslContext::new("prepared-range-scratch")
            .output("result", result)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let input =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, columns, DistType::FinRingDist);
        let matrix = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &input);
        let expected = matrix.to_compact_bytes();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        // Exercise the same demand/provisioning passes with a deliberately
        // smaller scratch envelope. This avoids allocating a VRAM-sized test.
        let demand = backend
            .graph_admission_demand(
                &graph,
                &FrozenGraphScopeId::Root,
                &graph.bindings,
                None,
                false,
                None,
                width,
                1,
                true,
            )
            .unwrap()
            .0;
        backend.prepare_graph_storage(demand).unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        backend.set_admitted_measurement_sink(Some(Box::new(move |event| {
            sender.send(event).unwrap();
        })));
        let mut store = MemoryArtifactStore::default();
        let mut result = execute_with_config(
            &graph,
            &mut backend,
            BTreeMap::from([("a".into(), RuntimeValue::matrix(matrix.into()))]),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig { prepared_gpu_admission: true, ..ExecutionConfig::default() },
        )
        .unwrap();
        backend.set_admitted_measurement_sink(None);
        let mut measured = Vec::new();
        let mut invocation = Vec::new();
        for event in receiver {
            match event {
                crate::gpu_measurement::GpuAdmittedMeasurement::Invocation { .. } => {
                    if !invocation.is_empty() {
                        measured.push(std::mem::take(&mut invocation));
                    }
                }
                crate::gpu_measurement::GpuAdmittedMeasurement::Wave { jobs, measured, .. } => {
                    invocation.push((jobs, measured));
                }
                _ => {}
            }
        }
        if !invocation.is_empty() {
            measured.push(invocation);
        }
        let product = backend
            .admitted_plan_log()
            .iter()
            .map(|(_, invocation)| invocation)
            .find(|invocation| invocation.operation == "MultiplyCompact")
            .unwrap();
        assert_eq!(product.plan.columns, columns);
        assert_eq!(product.plan.widths, vec![width]);
        assert_eq!(product.plan.wave_count, columns.div_ceil(width));
        let repeated = measured
            .iter()
            .filter(|waves| waves.len() == columns.div_ceil(width))
            .collect::<Vec<_>>();
        assert!(!repeated.is_empty());
        for waves in repeated {
            let mut classes = std::collections::BTreeSet::new();
            for (jobs, measured) in waves {
                let shape = jobs
                    .iter()
                    .map(|job| (job.device, job.source_interval, job.end - job.start))
                    .collect::<Vec<_>>();
                assert_eq!(
                    *measured,
                    classes.insert(shape),
                    "only the first wave of each range/width class is sampled"
                );
            }
            if columns > width * 2 {
                assert!(waves.iter().any(|(_, measured)| !measured));
            }
            assert!(
                waves.last().unwrap().1 || columns % width == 0,
                "a new tail width needs its own measurement"
            );
        }
        let RuntimeValue::Matrix(output) =
            result.materialize_output("result", &mut backend, &mut store).unwrap()
        else {
            panic!("matrix output");
        };
        assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_eager_selection_reuses_existing_owner_inventory() {
        use mxx_ir_core::artifact::{
            ArtifactConfidentiality, ArtifactType, Manifest, ManifestArtifact, ProductionId,
            SpecHash,
        };
        let production =
            ProductionId { spec_hash: SpecHash(rand::random()), execution_nonce: rand::random() };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let input = DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 5, DistType::FinRingDist);
        let mut capacities = Vec::new();
        for selection in 0..4 {
            let params = GpuDCRTPolyParams::new_with_gpu(
                n,
                cpu.to_crt().0,
                4,
                vec![device],
                Some(1),
                None,
                None,
            );
            let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
            let input_wire = ring.input("a", (2, 5));
            let selected = match selection {
                0 => input_wire.clone(),
                1 => mxx_dsl::Family::pack(vec![input_wire.clone(), input_wire.clone()])
                    .unwrap()
                    .at(0),
                2 => mxx_dsl::select(0, vec![input_wire.clone(), input_wire.clone()]).unwrap(),
                _ => mxx_dsl::Family::pack(vec![
                    input_wire.clone(),
                    ring.artifact_input(
                        production.clone(),
                        "unused",
                        (2, 5),
                        ArtifactConfidentiality::Private,
                    ),
                ])
                .unwrap()
                .at(0),
            };
            let manifest = Manifest {
                ir_version: mxx_ir_core::encoding::IR_VERSION,
                production_id: production.clone(),
                artifacts: BTreeMap::from([(
                    "unused".into(),
                    ManifestArtifact {
                        artifact_type: ArtifactType::Matrix(ConcreteMatrixType {
                            rows: 2,
                            columns: 5,
                            ring_dimension: n as usize,
                            modulus: params.modulus().as_ref().clone().into(),
                        }),
                        family_count: None,
                        confidentiality: ArtifactConfidentiality::Private,
                        content_hash: None,
                        layout: None,
                    },
                )]),
            };
            let graph = DslContext::new("eager-selection-owner")
                .output("source", input_wire)
                .unwrap()
                .output("consumed", -selected.clone())
                .unwrap()
                .output("selected", selected)
                .unwrap()
                .build()
                .unwrap()
                .validate_with_manifests(
                    &ParamEnv::default(),
                    &BTreeMap::from([(production.clone(), manifest)]),
                )
                .unwrap();
            let matrix = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &input);
            let expected = matrix.to_compact_bytes();
            let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
            let mut store = MemoryArtifactStore::default();
            let mut result = execute_with_config(
                &graph,
                &mut backend,
                BTreeMap::from([("a".into(), RuntimeValue::matrix(matrix.into()))]),
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig { prepared_gpu_admission: true, ..ExecutionConfig::default() },
            )
            .unwrap();
            for name in ["source", "selected"] {
                let RuntimeValue::Matrix(output) =
                    result.materialize_output(name, &mut backend, &mut store).unwrap()
                else {
                    panic!("matrix result")
                };
                assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected);
            }
            capacities.push(
                backend
                    .prepared_ledger
                    .as_ref()
                    .unwrap()
                    .prepared_inventory()
                    .map(|(_, storage)| storage.occupancy().unwrap().requested_capacity_bytes())
                    .sum::<usize>(),
            );
        }
        assert_eq!(capacities, vec![capacities[0]; 4]);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_graph_inventory_reuses_dead_values_in_long_chains() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let input = DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 5, DistType::FinRingDist);
        let mut capacities = Vec::new();
        for length in [8, 128] {
            let params = GpuDCRTPolyParams::new_with_gpu(
                n,
                cpu.to_crt().0,
                4,
                vec![device],
                Some(1),
                None,
                None,
            );
            let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
            let mut value = ring.input("a", (2, 5));
            for _ in 0..length {
                value = -value;
            }
            let graph = DslContext::new("prepared-live-chain")
                .output("value", value)
                .unwrap()
                .build()
                .unwrap()
                .validate(&ParamEnv::default())
                .unwrap();
            let matrix = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &input);
            let expected = matrix.to_compact_bytes();
            let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
            let mut store = MemoryArtifactStore::default();
            let mut result = execute_with_config(
                &graph,
                &mut backend,
                BTreeMap::from([("a".into(), RuntimeValue::matrix(matrix.into()))]),
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig { prepared_gpu_admission: true, ..ExecutionConfig::default() },
            )
            .unwrap();
            let RuntimeValue::Matrix(output) =
                result.materialize_output("value", &mut backend, &mut store).unwrap()
            else {
                panic!("matrix result");
            };
            assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected);
            capacities.push(
                backend
                    .prepared_ledger
                    .as_ref()
                    .unwrap()
                    .prepared_inventory()
                    .map(|(_, storage)| storage.occupancy().unwrap().requested_capacity_bytes())
                    .sum::<usize>(),
            );
        }
        assert_eq!(
            capacities[0], capacities[1],
            "sequential depth must not increase prepared backing"
        );
    }

    /// Explicit warmup discovers resource classes and must not provision the
    /// graph-sized production inventory or close the allocation-domain seal on
    /// its own. Production preparation stays the only storage boundary.
    fn assert_warmup_provisioned_no_graph_storage(
        backend: &crate::backend::poly_gpu::GpuDcrtBackend,
    ) {
        assert!(backend.prepared_ledger.is_none(), "warmup must not provision graph-sized backing");
        assert!(!backend.prepared_required, "warmup must not require prepared admission");
        assert!(!backend.graph_prepared, "warmup must not mark the graph prepared");
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_graph_execution_derives_complete_prepared_inventory() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let digits = params.modulus_digits();
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let a = ring.input("a", (2, 3));
        let b = ring.input("b", (2, 3));
        let sum = a.clone() + b;
        let reconstructed = a.decompose(16, digits).mul_small_rhs(ring.gadget(2, 16, digits));
        let coefficients = ring.input("c", (1, 1)).coefficients();
        let graph = DslContext::new("prepared-graph-admission")
            .output("sum", sum)
            .unwrap()
            .output("reconstructed", reconstructed)
            .unwrap()
            .output("coefficients", coefficients)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let sampler = DCRTPolyUniformSampler::new();
        let left = sampler.sample_uniform(&cpu, 2, 3, DistType::FinRingDist);
        let right = sampler.sample_uniform(&cpu, 2, 3, DistType::FinRingDist);
        let expected_sum =
            GpuDCRTPolyMatrix::from_cpu_matrix(&params, &(&left + &right)).to_compact_bytes();
        let expected_left = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &left).to_compact_bytes();
        let scalar = sampler.sample_uniform(&cpu, 1, 1, DistType::FinRingDist);
        let expected_coefficients = scalar
            .entry(0, 0)
            .coeffs_biguints()
            .into_iter()
            .map(num_bigint::BigInt::from)
            .collect::<Vec<_>>();
        // One resident input and one host-staged artifact input.
        let inputs = BTreeMap::from([
            (
                "a".to_string(),
                RuntimeValue::matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &left).into()),
            ),
            (
                "b".to_string(),
                RuntimeValue::HostMatrix {
                    matrix_type: mxx_ir_core::types::ConcreteMatrixType {
                        modulus: num_bigint::BigInt::from(params.modulus().as_ref().clone()),
                        ring_dimension: n as usize,
                        rows: 2,
                        columns: 3,
                    },
                    bytes: std::sync::Arc::new(
                        GpuDCRTPolyMatrix::from_cpu_matrix(&params, &right)
                            .into_cpu_staging_bytes(),
                    ),
                },
            ),
            (
                "c".to_string(),
                RuntimeValue::matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &scalar).into()),
            ),
        ]);
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        let error = backend
            .prepare_graph_admission(&graph, false, &inputs, 1, false)
            .err()
            .expect("polynomial readback requires explicit resource warmup");
        assert!(error.to_string().contains("explicit graph warmup required"), "{error}");
        assert!(backend.polynomial_value_plans.is_empty());
        drop(backend.prepare_graph_admission(&graph, false, &inputs, 1, true).unwrap());
        assert!(!backend.polynomial_value_plans.is_empty());
        assert_warmup_provisioned_no_graph_storage(&backend);
        let mut store = MemoryArtifactStore::default();
        let config = ExecutionConfig { prepared_gpu_admission: true, ..ExecutionConfig::default() };
        let mut result = execute_with_config(
            &graph,
            &mut backend,
            inputs,
            &mut store,
            SamplingMode::Fresh,
            config,
        )
        .unwrap();
        assert!(backend.prepared_ledger.is_some());
        for (name, expected) in [("sum", &expected_sum), ("reconstructed", &expected_left)] {
            let RuntimeValue::Matrix(output) =
                result.materialize_output(name, &mut backend, &mut store).unwrap()
            else {
                panic!("matrix output")
            };
            assert_eq!(&backend.matrix_to_bytes(&output).unwrap(), expected, "{name}");
        }
        // Scalar polynomial readback runs inside its traced claim boundary.
        let RuntimeValue::IndexedFamily(values) =
            result.materialize_output("coefficients", &mut backend, &mut store).unwrap()
        else {
            panic!("coefficient family output")
        };
        let values = values
            .into_iter()
            .map(|value| match value {
                RuntimeValue::Int(value) => value.clone(),
                _ => panic!("integer coefficient"),
            })
            .collect::<Vec<_>>();
        assert_eq!(values, expected_coefficients);

        // A kind without a compiled runner is rejected before any node runs and
        // before any domain is sealed. Trapdoor sampling is admitted since
        // Step 3, so use a runtime polynomial import, which has no runner.
        let values = mxx_dsl::Family::pack(
            (0..n).map(|value| mxx_dsl::Int::constant(value as u64)).collect(),
        )
        .unwrap();
        let trapdoor_graph = DslContext::new("prepared-graph-admission-unsupported")
            .output("polynomial", ring.from_coefficients(&values))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let mut fresh = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        let config = ExecutionConfig { prepared_gpu_admission: true, ..ExecutionConfig::default() };
        assert!(
            execute_with_config(
                &trapdoor_graph,
                &mut fresh,
                BTreeMap::new(),
                &mut MemoryArtifactStore::default(),
                SamplingMode::Fresh,
                config,
            )
            .is_err()
        );
        assert!(fresh.prepared_ledger.is_none());
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_graph_execution_admits_trapdoor_and_preimage_sampling() {
        run_preimage_graph(None, 1, false);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_preimage_range_scratch_handles_tail() {
        run_preimage_graph(Some(2), 1, false);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_preimage_batch_preserves_every_sibling_relation() {
        run_preimage_graph(None, 3, false);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_preimage_fleet_preserves_every_target_column() {
        run_preimage_graph(None, 3, true);
    }

    /// Explicit matched-boundary comparison. The profiled and unprofiled runs
    /// both produce/check real outputs; class warmup never executes this graph.
    #[test]
    #[ignore = "explicit four-workload performance comparison"]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_bounded_wave_performance_comparison() {
        use crate::{
            FileArtifactStore,
            artifact::{ArtifactKey, ArtifactPayload, ArtifactStore},
            gpu_measurement::GpuAdmittedMeasurement,
        };
        use mxx_ir_core::artifact::{
            ArtifactConfidentiality, ArtifactType, Manifest, ManifestArtifact, ProductionId,
            SpecHash,
        };
        use std::{collections::BTreeSet, num::NonZeroUsize, time::Instant};
        // Explicit benchmark capture boundaries exclude class discovery and
        // input setup. Nsight can collect each profiled round trip separately.
        unsafe extern "C" {
            fn cudaProfilerStart() -> i32;
            fn cudaProfilerStop() -> i32;
        }
        let setting = |name: &str, default: usize| {
            std::env::var(name).map(|value| value.parse::<usize>().unwrap()).unwrap_or(default)
        };
        let n = setting("MXX_PRIMITIVE_TEST_RING_DIMENSION", 1024) as u32;
        let wave = setting("MXX_PRIMITIVE_TEST_WAVE_BOUND", 4).max(2);
        let count = wave + 1;
        let wide_columns = setting("MXX_PRIMITIVE_TEST_MATRIX_SIZE", 256).max(4);
        let devices = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids();
        for &device in &devices {
            crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
            println!(
                "PERF_HARDWARE {:?}",
                mxx_primitives::poly::dcrt::gpu::gpu_device_identity(device).unwrap()
            );
        }
        let device = devices[0];
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let run_root = std::path::PathBuf::from("test_data/test_gpu_bounded_wave_performance")
            .join(format!("{:016x}", rand::random::<u64>()));
        for case in ["small_batch", "column_heavy", "preimage", "lazy_checkpoint"] {
            let rows = if case == "preimage" { 1 } else { 2 };
            let columns = if case == "column_heavy" { wide_columns } else { 3 };
            let original = DCRTPolyUniformSampler::new().sample_uniform(
                &cpu,
                rows,
                columns,
                DistType::FinRingDist,
            );
            let resident = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
            let input_bytes = resident.to_compact_bytes();
            let expected = GpuDCRTPolyMatrix::from_cpu_matrix(
                &params,
                &if case == "preimage" { original.clone() } else { -original },
            )
            .to_compact_bytes();
            let input_value = RuntimeValue::matrix(resident.into());
            let production = ProductionId {
                spec_hash: SpecHash(rand::random()),
                execution_nonce: rand::random(),
            };
            let key =
                ArtifactKey { production: production.clone(), name: "input".into(), index: None };
            let ty = ConcreteMatrixType {
                rows,
                columns,
                ring_dimension: n as usize,
                modulus: params.modulus().as_ref().clone().into(),
            };
            let descriptor = ManifestArtifact {
                artifact_type: ArtifactType::Matrix(ty.clone()),
                family_count: None,
                confidentiality: ArtifactConfidentiality::Private,
                content_hash: None,
                layout: None,
            };
            let manifest = Manifest {
                ir_version: mxx_ir_core::encoding::IR_VERSION,
                production_id: production.clone(),
                artifacts: BTreeMap::from([("input".into(), descriptor)]),
            };
            let input = if case == "lazy_checkpoint" {
                ring.artifact_input(
                    production.clone(),
                    "input",
                    (rows, columns),
                    ArtifactConfidentiality::Private,
                )
            } else {
                ring.input("input", (rows, columns))
            };
            let outputs = if case == "preimage" {
                let digits = params.modulus_digits();
                let trapdoor = ring.sample_trapdoor(1, 5, 16, digits, 100_000_000);
                mxx_dsl::parallel(count, |_| {
                    Ok(trapdoor
                        .sample_preimage(input.clone(), (digits + 2, columns))
                        .mul_small_rhs(trapdoor.public_matrix()))
                })
                .unwrap()
            } else if case == "column_heavy" {
                let identity = ring.identity(rows);
                mxx_dsl::parallel(count, |_| Ok(-(identity.clone() * input.clone()))).unwrap()
            } else {
                mxx_dsl::parallel(count, |_| Ok(-((input.clone() + input.clone()) - input.clone())))
                    .unwrap()
            };
            let graph = DslContext::new(format!("bounded-wave-perf-{case}"))
                .output("outputs", outputs)
                .unwrap()
                .build()
                .unwrap()
                .validate_with_manifests(
                    &ParamEnv::default(),
                    &BTreeMap::from([(production.clone(), manifest.clone())]),
                )
                .unwrap();
            let inputs = if case == "lazy_checkpoint" {
                BTreeMap::new()
            } else {
                BTreeMap::from([("input".into(), input_value)])
            };
            for cap in [1, wave] {
                let mut backend =
                    crate::backend::poly_gpu::gpu_backend_on([params.clone()], devices.clone());
                drop(backend.prepare_graph_admission(&graph, false, &inputs, cap, true).unwrap());
                let classes = (backend.trapdoor_plans.len(), backend.preimage_plans.len());
                let mut admissions = BTreeSet::new();
                let mut widths = BTreeSet::new();
                let mut device_widths = BTreeSet::new();
                let mut batches = 0usize;
                let mut jobs = 0usize;
                let mut peak_prepared = 0usize;
                for profile in [true, false] {
                    let path = run_root.join(format!("{case}-{cap}-{profile}"));
                    let mut store = FileArtifactStore::new(&path).unwrap();
                    if case == "lazy_checkpoint" {
                        store
                            .store(
                                key.clone(),
                                &ArtifactType::Matrix(ty.clone()),
                                ArtifactConfidentiality::Private,
                                None,
                                ArtifactPayload::Matrix(input_bytes.clone()),
                            )
                            .unwrap();
                        store.store_manifest(manifest.clone()).unwrap();
                    }
                    let (sender, receiver) = std::sync::mpsc::channel();
                    if profile {
                        backend.set_admitted_measurement_sink(Some(Box::new(move |event| {
                            sender.send(event).unwrap();
                        })));
                    }
                    if profile {
                        assert_eq!(unsafe { cudaProfilerStart() }, 0);
                    }
                    let start = Instant::now();
                    let mut result = execute_with_config(
                        &graph,
                        &mut backend,
                        inputs.clone(),
                        &mut store,
                        SamplingMode::Fresh,
                        ExecutionConfig {
                            max_parallel_instances: NonZeroUsize::new(cap).unwrap(),
                            prepared_gpu_admission: true,
                            ..ExecutionConfig::default()
                        },
                    )
                    .unwrap();
                    let RuntimeValue::IndexedFamily(outputs) =
                        result.materialize_output("outputs", &mut backend, &mut store).unwrap()
                    else {
                        panic!("matrix output family")
                    };
                    assert_eq!(outputs.len(), count);
                    let canonical = outputs
                        .iter()
                        .map(|value| {
                            let RuntimeValue::Matrix(matrix) = value else {
                                panic!("matrix output")
                            };
                            backend.matrix_to_bytes(matrix).unwrap()
                        })
                        .collect::<Vec<_>>();
                    let seconds = start.elapsed().as_secs_f64();
                    if profile {
                        assert_eq!(unsafe { cudaProfilerStop() }, 0);
                    }
                    backend.set_admitted_measurement_sink(None);
                    for bytes in &canonical {
                        assert_eq!(bytes, &expected);
                    }
                    assert_eq!(
                        (backend.trapdoor_plans.len(), backend.preimage_plans.len()),
                        classes,
                        "production must not discover new classes"
                    );
                    if case == "lazy_checkpoint" {
                        assert_eq!(store.load_count(&key), 1);
                    }
                    for event in receiver.try_iter() {
                        match event {
                            GpuAdmittedMeasurement::Admission { instances, column_cap, .. } => {
                                admissions.insert((instances, column_cap));
                            }
                            GpuAdmittedMeasurement::Invocation { plans, .. } => {
                                for plan in plans {
                                    device_widths.insert(plan.widths.clone());
                                    for width in plan.widths {
                                        widths.insert(width);
                                    }
                                }
                            }
                            GpuAdmittedMeasurement::Wave { jobs: wave_jobs, timing, .. } => {
                                batches += 1;
                                jobs += wave_jobs.len();
                                peak_prepared = peak_prepared.max(
                                    timing
                                        .prepared_memory
                                        .iter()
                                        .map(|memory| memory.peak_bytes)
                                        .sum(),
                                );
                            }
                            _ => {}
                        }
                    }
                    if !profile {
                        println!(
                            "PERF_RESULT {}",
                            serde_json::json!({
                                "workload": case, "wave_limit": cap, "instances": count,
                                "ring_dimension": n, "rows": rows, "columns": columns,
                                "admissions": admissions, "column_widths": widths,
                                "device_ids": devices, "device_column_widths": device_widths,
                                "round_trip_seconds": seconds, "outputs_per_second": count as f64 / seconds,
                                "profiled_column_batches": batches, "profiled_column_jobs": jobs,
                                "profiled_peak_prepared_device_bytes": peak_prepared,
                                "canonical_output_bytes": canonical.iter().map(Vec::len).sum::<usize>(),
                                "lazy_input_payload_bytes": if case == "lazy_checkpoint" { input_bytes.len() } else { 0 },
                                "lazy_input_loads": store.load_count(&key),
                                "host_staging_peak_bytes": null, "cuda_kernel_launches": null,
                                "total_dma_bytes": null,
                            })
                        );
                    }
                    drop(result);
                    drop(store);
                    std::fs::remove_dir_all(path).unwrap();
                }
            }
        }
        std::fs::remove_dir_all(run_root).unwrap();
    }

    fn run_preimage_graph(scratch_columns: Option<usize>, instances: usize, fleet: bool) {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let mut devices = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids();
        if fleet {
            assert!(devices.len() > 1, "remote multi-GPU test");
        } else {
            devices.truncate(1);
        }
        for &device in &devices {
            crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        }
        let device = devices[0];
        let columns = if fleet { devices.len() + 1 } else { 3 };
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let digits = params.modulus_digits();
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let trapdoor = ring.sample_trapdoor(1, 5, 16, digits, 100_000_000);
        let target = ring.input("t", (1, columns));
        let context = DslContext::new("prepared-graph-preimage");
        let context = if instances == 1 {
            let preimage = trapdoor.sample_preimage(target, (digits + 2, columns));
            context.output("check", preimage.mul_small_rhs(trapdoor.public_matrix())).unwrap()
        } else {
            let checks = mxx_dsl::parallel(instances, |_| {
                let preimage = trapdoor.sample_preimage(target.clone(), (digits + 2, columns));
                Ok(preimage.mul_small_rhs(trapdoor.public_matrix()))
            })
            .unwrap();
            context.output("check", checks).unwrap()
        };
        let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let sampler = DCRTPolyUniformSampler::new();
        let target_value = sampler.sample_uniform(&cpu, 1, columns, DistType::FinRingDist);
        let expected =
            GpuDCRTPolyMatrix::from_cpu_matrix(&params, &target_value).to_compact_bytes();
        let inputs = BTreeMap::from([(
            "t".to_string(),
            RuntimeValue::HostMatrix {
                matrix_type: mxx_ir_core::types::ConcreteMatrixType {
                    modulus: num_bigint::BigInt::from(params.modulus().as_ref().clone()),
                    ring_dimension: n as usize,
                    rows: 1,
                    columns,
                },
                bytes: std::sync::Arc::new(
                    GpuDCRTPolyMatrix::from_cpu_matrix(&params, &target_value)
                        .into_cpu_staging_bytes(),
                ),
            },
        )]);
        let mut backend =
            crate::backend::poly_gpu::gpu_backend_on([params.clone()], devices.clone());
        let error = execute_with_config(
            &graph,
            &mut backend,
            inputs.clone(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .err()
        .expect("production cannot discover missing resource plans");
        assert!(error.to_string().contains("explicit graph warmup required"), "{error}");
        assert!(backend.trapdoor_plans.is_empty());
        assert!(backend.preimage_plans.is_empty());
        if let Some(width) = scratch_columns {
            // Warm discovery plans without installing wider automatic storage:
            // this fixture supplies its own width-limited inventory below.
            backend
                .graph_admission_demand(
                    &graph,
                    &FrozenGraphScopeId::Root,
                    &graph.bindings,
                    None,
                    false,
                    None,
                    width,
                    1,
                    true,
                )
                .unwrap();
        } else {
            drop(backend.prepare_graph_admission(&graph, false, &inputs, 1, true).unwrap());
            assert_warmup_provisioned_no_graph_storage(&backend);
        }
        let warmed = (backend.trapdoor_plans.len(), backend.preimage_plans.len());
        assert!(warmed.0 > 0 && warmed.1 > 0);
        let saved = std::mem::take(&mut backend.preimage_plans);
        let error = backend
            .prepare_graph_admission(&graph, false, &inputs, 1, false)
            .err()
            .expect("lost warmup plan must not trigger a replacement trial");
        assert!(error.to_string().contains("explicit graph warmup required"), "{error}");
        assert!(backend.preimage_plans.is_empty());
        backend.preimage_plans = saved;
        if let Some(width) = scratch_columns {
            let demand = backend
                .graph_admission_demand(
                    &graph,
                    &FrozenGraphScopeId::Root,
                    &graph.bindings,
                    None,
                    false,
                    None,
                    width,
                    1,
                    true,
                )
                .unwrap()
                .0;
            backend.prepare_graph_storage(demand).unwrap();
        }
        let mut store = MemoryArtifactStore::default();
        let config = ExecutionConfig {
            prepared_gpu_admission: true,
            max_parallel_instances: std::num::NonZeroUsize::new(instances).unwrap(),
            ..ExecutionConfig::default()
        };
        let (sender, receiver) = std::sync::mpsc::channel();
        if instances > 1 {
            backend.set_admitted_measurement_sink(Some(Box::new(move |event| {
                sender.send(event).unwrap();
            })));
        }
        let mut result = execute_with_config(
            &graph,
            &mut backend,
            inputs,
            &mut store,
            SamplingMode::Fresh,
            config,
        )
        .unwrap();
        assert_eq!((backend.trapdoor_plans.len(), backend.preimage_plans.len()), warmed);
        assert!(backend.prepared_ledger.is_some());
        if instances > 1 {
            backend.set_admitted_measurement_sink(None);
            assert_eq!(backend.graph_wave, instances);
            let mut invocation_batch = 0;
            let mut submitted_jobs = 0;
            for event in receiver.try_iter() {
                match event {
                    crate::gpu_measurement::GpuAdmittedMeasurement::Invocation {
                        plans, ..
                    } => {
                        invocation_batch = invocation_batch.max(plans.len());
                    }
                    crate::gpu_measurement::GpuAdmittedMeasurement::Wave { jobs, .. } => {
                        submitted_jobs = submitted_jobs.max(jobs.len());
                    }
                    _ => {}
                }
            }
            assert_eq!(invocation_batch, instances, "observe actual batch multiplicity");
            if fleet {
                assert!(submitted_jobs >= instances, "observe grouped fleet submission");
            } else {
                assert_eq!(submitted_jobs, instances, "observe grouped column submission");
            }
        }
        if fleet {
            let preimage = backend
                .admitted_plan_log()
                .iter()
                .map(|(_, invocation)| invocation)
                .find(|invocation| invocation.operation == "Preimage")
                .unwrap();
            assert_eq!(preimage.plan.widths.len(), devices.len());
            assert!(preimage.plan.widths.iter().filter(|&&width| width > 0).count() > 1);
        }
        if let Some(width) = scratch_columns {
            let preimage = backend
                .admitted_plan_log()
                .iter()
                .map(|(_, invocation)| invocation)
                .find(|invocation| invocation.operation == "Preimage")
                .unwrap();
            assert_eq!(preimage.plan.widths, vec![width]);
            assert_eq!(preimage.plan.wave_count, 3usize.div_ceil(width));
        }

        let output = result.materialize_output("check", &mut backend, &mut store).unwrap();
        let outputs = match output {
            RuntimeValue::Matrix(_) => std::slice::from_ref(output),
            RuntimeValue::IndexedFamily(outputs) => outputs.as_slice(),
            _ => panic!("Preimage relation outputs"),
        };
        assert_eq!(outputs.len(), instances);
        for output in outputs {
            let RuntimeValue::Matrix(output) = output else { panic!("relation matrix") };
            assert_eq!(backend.matrix_to_bytes(&output).unwrap(), expected);
        }
    }
}
