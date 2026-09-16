//! Fixed trapdoor owners and sampler/arithmetic commands.

use super::GpuDCRTTrapdoor;
use crate::{
    matrix::{
        PolyMatrix,
        gpu_dcrt_poly::{
            GpuDCRTPolyMatrix, GpuMatrixRangeConstant, GpuMatrixSampleDist, GpuPreparedArithmetic,
            GpuPreparedArithmeticCommand, GpuPreparedArithmeticKind, GpuPreparedRange,
            GpuPreparedSampling, GpuPreparedSchedule, GpuPreparedSchedulePlan, GpuPreparedSlotKind,
            GpuPreparedTransform, GpuPreparedTranspose, GpuPreparedView,
            GpuPreparedWorkspaceLayout, GpuTracedClaim, PreparedOwnerLayout,
            PreparedOwnerLayoutCursor, PreparedPlanLayout,
        },
    },
    poly::{
        PolyParams,
        dcrt::gpu::{GPU_MATRIX_DIST_GAUSS, GpuDCRTPolyParams, GpuRngSeed},
    },
};
use std::{
    cell::Cell,
    sync::{Arc, Mutex},
};

/// Warmup-owned structural description of the prepared trapdoor sampler.
/// The matrix claims preserve the fixed owner order while `sampler` is the
/// exact native descriptor consumed by the first Gaussian sampler bind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuPreparedTrapdoorLayout {
    pub sampler: PreparedPlanLayout,
    pub matrix_owners: Box<[PreparedOwnerLayout]>,
    pub stages: Box<[PreparedPlanLayout]>,
    pub matrix_claims: [GpuTracedClaim; 13],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuPreparedTrapdoorEntryKind {
    MatrixOwner,
    Sampler,
    Transpose,
    Arithmetic,
    Transform,
}

/// The fixed order of all native substages in a prepared trapdoor bind.
///
/// This is the single order contract shared by planning, binding, admission
/// metadata, and no-device tests.  In particular, the three public arithmetic
/// commands and their three destination copies have distinct stages.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TrapdoorStage {
    SamplerR,
    SamplerE,
    SamplerAbar,
    TransposeR,
    TransposeE,
    GramA,
    GramB,
    GramD,
    GramATransform,
    GramBTransform,
    GramDTransform,
    PublicProduct,
    PublicSum,
    PublicTail,
    PublicCopyAbar,
    PublicCopyIdentity,
    PublicCopyTail,
}

impl TrapdoorStage {
    pub const ALL: [Self; 17] = [
        Self::SamplerR,
        Self::SamplerE,
        Self::SamplerAbar,
        Self::TransposeR,
        Self::TransposeE,
        Self::GramA,
        Self::GramB,
        Self::GramD,
        Self::GramATransform,
        Self::GramBTransform,
        Self::GramDTransform,
        Self::PublicProduct,
        Self::PublicSum,
        Self::PublicTail,
        Self::PublicCopyAbar,
        Self::PublicCopyIdentity,
        Self::PublicCopyTail,
    ];

    pub const fn index(self) -> usize {
        match self {
            Self::SamplerR => 0,
            Self::SamplerE => 1,
            Self::SamplerAbar => 2,
            Self::TransposeR => 3,
            Self::TransposeE => 4,
            Self::GramA => 5,
            Self::GramB => 6,
            Self::GramD => 7,
            Self::GramATransform => 8,
            Self::GramBTransform => 9,
            Self::GramDTransform => 10,
            Self::PublicProduct => 11,
            Self::PublicSum => 12,
            Self::PublicTail => 13,
            Self::PublicCopyAbar => 14,
            Self::PublicCopyIdentity => 15,
            Self::PublicCopyTail => 16,
        }
    }

    pub const fn next(self) -> Option<Self> {
        match self {
            Self::SamplerR => Some(Self::SamplerE),
            Self::SamplerE => Some(Self::SamplerAbar),
            Self::SamplerAbar => Some(Self::TransposeR),
            Self::TransposeR => Some(Self::TransposeE),
            Self::TransposeE => Some(Self::GramA),
            Self::GramA => Some(Self::GramB),
            Self::GramB => Some(Self::GramD),
            Self::GramD => Some(Self::GramATransform),
            Self::GramATransform => Some(Self::GramBTransform),
            Self::GramBTransform => Some(Self::GramDTransform),
            Self::GramDTransform => Some(Self::PublicProduct),
            Self::PublicProduct => Some(Self::PublicSum),
            Self::PublicSum => Some(Self::PublicTail),
            Self::PublicTail => Some(Self::PublicCopyAbar),
            Self::PublicCopyAbar => Some(Self::PublicCopyIdentity),
            Self::PublicCopyIdentity => Some(Self::PublicCopyTail),
            Self::PublicCopyTail => None,
        }
    }

    pub const fn kind(self) -> GpuPreparedTrapdoorEntryKind {
        match self {
            Self::SamplerR | Self::SamplerE | Self::SamplerAbar => {
                GpuPreparedTrapdoorEntryKind::Sampler
            }
            Self::TransposeR | Self::TransposeE => GpuPreparedTrapdoorEntryKind::Transpose,
            Self::GramATransform | Self::GramBTransform | Self::GramDTransform => {
                GpuPreparedTrapdoorEntryKind::Transform
            }
            Self::GramA |
            Self::GramB |
            Self::GramD |
            Self::PublicProduct |
            Self::PublicSum |
            Self::PublicTail |
            Self::PublicCopyAbar |
            Self::PublicCopyIdentity |
            Self::PublicCopyTail => GpuPreparedTrapdoorEntryKind::Arithmetic,
        }
    }
}

pub const TRAPDOOR_ENTRY_KINDS: [GpuPreparedTrapdoorEntryKind; 5] = [
    GpuPreparedTrapdoorEntryKind::MatrixOwner,
    GpuPreparedTrapdoorEntryKind::Sampler,
    GpuPreparedTrapdoorEntryKind::Transpose,
    GpuPreparedTrapdoorEntryKind::Arithmetic,
    GpuPreparedTrapdoorEntryKind::Transform,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuPreparedTrapdoorBindEntry {
    pub kind: GpuPreparedTrapdoorEntryKind,
    pub owner: Option<usize>,
    pub stage: Option<TrapdoorStage>,
}

const fn trapdoor_stage_kinds() -> [GpuPreparedTrapdoorEntryKind; 17] {
    [
        TrapdoorStage::SamplerR.kind(),
        TrapdoorStage::SamplerE.kind(),
        TrapdoorStage::SamplerAbar.kind(),
        TrapdoorStage::TransposeR.kind(),
        TrapdoorStage::TransposeE.kind(),
        TrapdoorStage::GramA.kind(),
        TrapdoorStage::GramB.kind(),
        TrapdoorStage::GramD.kind(),
        TrapdoorStage::GramATransform.kind(),
        TrapdoorStage::GramBTransform.kind(),
        TrapdoorStage::GramDTransform.kind(),
        TrapdoorStage::PublicProduct.kind(),
        TrapdoorStage::PublicSum.kind(),
        TrapdoorStage::PublicTail.kind(),
        TrapdoorStage::PublicCopyAbar.kind(),
        TrapdoorStage::PublicCopyIdentity.kind(),
        TrapdoorStage::PublicCopyTail.kind(),
    ]
}

pub const TRAPDOOR_STAGE_KINDS: [GpuPreparedTrapdoorEntryKind; 17] = trapdoor_stage_kinds();

impl GpuPreparedTrapdoorLayout {
    pub fn entry_kinds() -> &'static [GpuPreparedTrapdoorEntryKind; 5] {
        &TRAPDOOR_ENTRY_KINDS
    }

    pub fn bind_entries() -> Vec<GpuPreparedTrapdoorBindEntry> {
        let mut entries = (0..13)
            .map(|owner| GpuPreparedTrapdoorBindEntry {
                kind: GpuPreparedTrapdoorEntryKind::MatrixOwner,
                owner: Some(owner),
                stage: None,
            })
            .collect::<Vec<_>>();
        entries.extend(TrapdoorStage::ALL.into_iter().map(|stage| GpuPreparedTrapdoorBindEntry {
            kind: stage.kind(),
            owner: None,
            stage: Some(stage),
        }));
        entries
    }
}

impl GpuPreparedTrapdoorLayout {
    fn stage(&self, stage: TrapdoorStage) -> &PreparedPlanLayout {
        &self.stages[stage.index()]
    }

    pub fn streams(&self) -> Vec<crate::matrix::gpu_dcrt_poly::PreparedStreamFootprint> {
        TrapdoorStage::ALL
            .into_iter()
            .flat_map(|stage| self.stage(stage).streams().iter().copied())
            .collect()
    }

    pub fn claims(&self) -> Vec<GpuTracedClaim> {
        let mut claims = self.matrix_claims.to_vec();
        for stage in TrapdoorStage::ALL {
            claims.extend(self.stage(stage).allocations().iter().filter_map(|allocation| {
            if allocation.kind == 100 {
                return None;
            }
            if allocation.kind == 0 {
                Some(GpuTracedClaim::matrix(
                    allocation.rows,
                    allocation.columns,
                    usize::try_from(allocation.level).unwrap_or(0),
                    allocation.format == crate::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL,
                ))
            } else {
                allocation
                    .allocation_kind()
                    .and_then(|kind| match kind {
                        crate::matrix::gpu_dcrt_poly::PreparedAllocationKind::Matrix |
                        crate::matrix::gpu_dcrt_poly::PreparedAllocationKind::HostOnly => None,
                        kind => Some(GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                            kind: match kind {
                                crate::matrix::gpu_dcrt_poly::PreparedAllocationKind::BatchWorkspace => GpuPreparedSlotKind::BatchWorkspace,
                                crate::matrix::gpu_dcrt_poly::PreparedAllocationKind::TransformWorkspace => GpuPreparedSlotKind::TransformWorkspace,
                                crate::matrix::gpu_dcrt_poly::PreparedAllocationKind::PinnedHost => GpuPreparedSlotKind::PinnedHost,
                                crate::matrix::gpu_dcrt_poly::PreparedAllocationKind::CompactPayload => GpuPreparedSlotKind::CompactPayload,
                                crate::matrix::gpu_dcrt_poly::PreparedAllocationKind::CompactWorkspace => GpuPreparedSlotKind::CompactWorkspace,
                                crate::matrix::gpu_dcrt_poly::PreparedAllocationKind::SamplerWorkspace => GpuPreparedSlotKind::SamplerWorkspace,
                                crate::matrix::gpu_dcrt_poly::PreparedAllocationKind::TransferWorkspace => GpuPreparedSlotKind::TransferWorkspace,
                                crate::matrix::gpu_dcrt_poly::PreparedAllocationKind::CompletionEvent => GpuPreparedSlotKind::CompletionEvent,
                                crate::matrix::gpu_dcrt_poly::PreparedAllocationKind::SubmissionStream => GpuPreparedSlotKind::SubmissionStream,
                                crate::matrix::gpu_dcrt_poly::PreparedAllocationKind::Matrix |
                                crate::matrix::gpu_dcrt_poly::PreparedAllocationKind::HostOnly => unreachable!(),
                            },
                            bytes: allocation.bytes,
                            alignment: allocation.alignment.max(1),
                        })),
                    })
            }
            }));
        }
        claims
    }

    /// Native allocation descriptors in the exact same order as `claims`.
    /// The leading logical matrix owners have no native allocation key; all
    /// stage allocations retain their full physical identity for admission.
    pub fn claim_layouts(
        &self,
    ) -> Vec<Option<crate::matrix::gpu_dcrt_poly::PreparedAllocationLayout>> {
        let mut layouts = vec![None; self.matrix_claims.len()];
        for stage in TrapdoorStage::ALL {
            for allocation in self.stage(stage).allocations() {
                if allocation.kind == 100 {
                    continue;
                }
                if allocation.kind == 0 {
                    layouts.push(Some(*allocation));
                } else if let Some(kind) = allocation.allocation_kind() {
                    if !matches!(
                        kind,
                        crate::matrix::gpu_dcrt_poly::PreparedAllocationKind::Matrix |
                            crate::matrix::gpu_dcrt_poly::PreparedAllocationKind::HostOnly
                    ) {
                        layouts.push(Some(*allocation));
                    }
                }
            }
        }
        layouts
    }
}

pub struct GpuPreparedTrapdoorSampler {
    samplers: [Arc<GpuPreparedSampling>; 3],
    transposes: [Arc<GpuPreparedTranspose>; 2],
    grams: [GpuPreparedArithmeticCommand; 3],
    gram_transforms: [(GpuPreparedTransform, Arc<GpuDCRTPolyMatrix>); 3],
    public_commands: Box<[GpuPreparedArithmeticCommand]>,
    trapdoor: Arc<GpuDCRTTrapdoor>,
    public: Arc<GpuDCRTPolyMatrix>,
}

impl GpuPreparedTrapdoorSampler {
    fn plan_stages(
        params: &GpuDCRTPolyParams,
        owners: &[PreparedOwnerLayout],
        rows: usize,
    ) -> Result<Box<[PreparedPlanLayout]>, String> {
        if owners.len() != 13 {
            return Err("prepared trapdoor owner count mismatch".into());
        }
        let k = params.modulus_digits();
        let level = params.crt_depth() - 1;
        let device =
            params.device_ids().first().copied().ok_or("prepared trapdoor has no device")?;
        let mut stages = Vec::with_capacity(TrapdoorStage::ALL.len());
        macro_rules! push_stage {
            ($stage:expr, $layout:expr) => {{
                let stage = $stage;
                if stages.len() != stage.index() {
                    return Err(format!(
                        "prepared trapdoor planner stage order mismatch at {}",
                        stage.index()
                    ));
                }
                stages.push($layout);
            }};
        }
        push_stage!(
            TrapdoorStage::SamplerR,
            PreparedPlanLayout::sampling_with_owner(
                params,
                rows,
                rows * k,
                rows * k,
                0,
                level,
                crate::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL,
                GPU_MATRIX_DIST_GAUSS,
                &owners[0],
            )?
        );
        push_stage!(
            TrapdoorStage::SamplerE,
            PreparedPlanLayout::sampling_with_owner(
                params,
                rows,
                rows * k,
                rows * k,
                0,
                level,
                crate::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL,
                GPU_MATRIX_DIST_GAUSS,
                &owners[1],
            )?
        );
        push_stage!(
            TrapdoorStage::SamplerAbar,
            PreparedPlanLayout::sampling_with_owner(
                params,
                rows,
                rows,
                rows,
                0,
                level,
                crate::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL,
                crate::poly::dcrt::gpu::GPU_MATRIX_DIST_UNIFORM,
                &owners[2],
            )?
        );
        push_stage!(
            TrapdoorStage::TransposeR,
            PreparedPlanLayout::transpose_with_owner(
                params,
                rows,
                rows * k,
                rows * k,
                rows,
                level,
                crate::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL,
                &owners[3],
            )?
        );
        push_stage!(
            TrapdoorStage::TransposeE,
            PreparedPlanLayout::transpose_with_owner(
                params,
                rows,
                rows * k,
                rows * k,
                rows,
                level,
                crate::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL,
                &owners[4],
            )?
        );
        let arith = |kind,
                     left_rows,
                     left_columns,
                     right_rows,
                     right_columns,
                     output_rows,
                     output_columns,
                     owner: &PreparedOwnerLayout| {
            PreparedPlanLayout::arithmetic_with_owner(
                params,
                params.ring_dimension() as usize,
                level + 1,
                left_rows,
                left_columns,
                right_rows,
                right_columns,
                output_rows,
                output_columns,
                0,
                0,
                0,
                kind,
                device,
                true,
                false,
                false,
                owner,
            )
        };
        push_stage!(
            TrapdoorStage::GramA,
            arith(4, rows, rows * k, rows * k, rows, rows, rows, &owners[5])?
        );
        push_stage!(
            TrapdoorStage::GramB,
            arith(4, rows, rows * k, rows * k, rows, rows, rows, &owners[6])?
        );
        push_stage!(
            TrapdoorStage::GramD,
            arith(4, rows, rows * k, rows * k, rows, rows, rows, &owners[7])?
        );
        for (stage, owner) in [
            (TrapdoorStage::GramATransform, &owners[5]),
            (TrapdoorStage::GramBTransform, &owners[6]),
            (TrapdoorStage::GramDTransform, &owners[7]),
        ] {
            push_stage!(
                stage,
                PreparedPlanLayout::ntt_with_owner(params, rows, rows, level, None, false, owner,)?
            );
        }
        push_stage!(
            TrapdoorStage::PublicProduct,
            arith(4, rows, rows, rows, rows * k, rows, rows * k, &owners[8])?
        );
        push_stage!(
            TrapdoorStage::PublicSum,
            arith(1, rows, rows * k, rows, rows * k, rows, rows * k, &owners[9])?
        );
        push_stage!(
            TrapdoorStage::PublicTail,
            arith(5, rows, rows * k, rows, rows * k, rows, rows * k, &owners[10])?
        );
        let public_owner = PreparedOwnerLayout::plan(
            params,
            rows,
            rows * (2 + k),
            level,
            crate::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL,
            0,
        )?;
        for ((source, start), stage) in [(rows, 0), (rows, rows), (rows, 2 * rows)]
            .into_iter()
            .zip(TrapdoorStage::ALL.into_iter().skip(TrapdoorStage::PublicCopyAbar.index()))
        {
            push_stage!(
                stage,
                PreparedPlanLayout::arithmetic_with_owner(
                    params,
                    params.ring_dimension() as usize,
                    level + 1,
                    source,
                    rows * k,
                    source,
                    rows * k,
                    rows,
                    rows * (2 + k),
                    start,
                    0,
                    0,
                    0,
                    device,
                    true,
                    false,
                    false,
                    &public_owner,
                )?
            );
        }
        if stages.len() != TrapdoorStage::ALL.len() {
            return Err("prepared trapdoor planner left stages unplanned".into());
        }
        Ok(stages.into_boxed_slice())
    }

    fn planned_matrix_owners(
        params: &GpuDCRTPolyParams,
        rows: usize,
    ) -> Result<Box<[PreparedOwnerLayout]>, String> {
        let mut cursor = PreparedOwnerLayoutCursor::default();
        Self::allocation_claims(params, rows)
            .into_iter()
            .map(|claim| {
                cursor.plan(
                    params,
                    claim.rows(),
                    claim.columns(),
                    claim.level().ok_or("trapdoor matrix claim has no level")?,
                    crate::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Vec::into_boxed_slice)
    }

    /// Plan the first fixed sampler without allocating a matrix, stream, event,
    /// or native prepared command.  The returned descriptor is consumed by
    /// [`Self::bind_with_layout`]; binding never replans this stage.
    pub fn plan_layout(
        params: &GpuDCRTPolyParams,
        public: &GpuDCRTPolyMatrix,
        sigma: f64,
    ) -> Result<PreparedPlanLayout, String> {
        let d = public.row_size();
        let k = params.modulus_digits();
        if d == 0 || public.col_size() != d * (2 + k) || !sigma.is_finite() || sigma <= 0.0 {
            return Err("prepared trapdoor layout contract mismatch".into());
        }
        Self::plan_layout_for_shape(params, d)
    }

    /// Metadata-only planner used by resource resolution before the output
    /// owner is materialized.
    pub fn plan_layout_for_shape(
        params: &GpuDCRTPolyParams,
        d: usize,
    ) -> Result<PreparedPlanLayout, String> {
        let k = params.modulus_digits();
        if d == 0 {
            return Err("prepared trapdoor layout shape is empty".into());
        }
        let owner = PreparedOwnerLayout::plan(
            params,
            d,
            d * k,
            params.crt_depth() - 1,
            crate::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL,
            0,
        )?;
        PreparedPlanLayout::sampling_with_owner(
            params,
            d,
            d * k,
            d * k,
            0,
            params.crt_depth() - 1,
            crate::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL,
            GPU_MATRIX_DIST_GAUSS,
            &owner,
        )
    }

    pub fn plan_layout_bundle(
        params: &GpuDCRTPolyParams,
        public: &GpuDCRTPolyMatrix,
        sigma: f64,
    ) -> Result<GpuPreparedTrapdoorLayout, String> {
        let sampler = Self::plan_layout(params, public, sigma)?;
        let matrix_owners = Self::planned_matrix_owners(params, public.row_size())?;
        let mut stages = Self::plan_stages(params, &matrix_owners, public.row_size())?;
        stages[TrapdoorStage::SamplerR.index()] = sampler.clone();
        Ok(GpuPreparedTrapdoorLayout {
            sampler,
            matrix_owners,
            stages,
            matrix_claims: Self::allocation_claims(params, public.row_size()),
        })
    }

    /// Shape-only trapdoor bundle planner used before live output owners are
    /// allocated by the runtime reservation transaction.
    pub fn plan_layout_bundle_for_shape(
        params: &GpuDCRTPolyParams,
        rows: usize,
        _sigma: f64,
    ) -> Result<GpuPreparedTrapdoorLayout, String> {
        let sampler = Self::plan_layout_for_shape(params, rows)?;
        let matrix_owners = Self::planned_matrix_owners(params, rows)?;
        let mut stages = Self::plan_stages(params, &matrix_owners, rows)?;
        stages[TrapdoorStage::SamplerR.index()] = sampler.clone();
        Ok(GpuPreparedTrapdoorLayout {
            sampler,
            matrix_owners,
            stages,
            matrix_claims: Self::allocation_claims(params, rows),
        })
    }

    pub fn bind_with_layout(
        params: &GpuDCRTPolyParams,
        public: Arc<GpuDCRTPolyMatrix>,
        sigma: f64,
        layout: GpuPreparedTrapdoorLayout,
    ) -> Result<Self, String> {
        if layout.matrix_claims != Self::allocation_claims(params, public.row_size()) {
            return Err("prepared trapdoor saved layout owner order mismatch".into());
        }
        Self::bind_saved(params, public, sigma, layout)
    }

    pub fn allocation_claims(params: &GpuDCRTPolyParams, d: usize) -> [GpuTracedClaim; 13] {
        let k = params.modulus_digits();
        [
            (d, d * k),
            (d, d * k),
            (d, d),
            (d * k, d),
            (d * k, d),
            (d, d),
            (d, d),
            (d, d),
            (d, d * k),
            (d, d * k),
            (d, d * k),
            (d, d * k),
            (d, d),
        ]
        .map(|(rows, columns)| GpuTracedClaim::matrix(rows, columns, params.crt_depth() - 1, true))
    }
    fn bind_saved(
        params: &GpuDCRTPolyParams,
        public: Arc<GpuDCRTPolyMatrix>,
        sigma: f64,
        layout: GpuPreparedTrapdoorLayout,
    ) -> Result<Self, String> {
        let d = public.row_size();
        let k = params.modulus_digits();
        if params.dropped_moduli() != 0 ||
            d == 0 ||
            public.col_size() != d * (2 + k) ||
            public.params() != params ||
            !public.is_ntt() ||
            !sigma.is_finite() ||
            sigma <= 0.0
        {
            return Err("prepared trapdoor contract mismatch".into());
        }
        if layout.matrix_owners.len() != 13 || layout.stages.len() != TrapdoorStage::ALL.len() {
            return Err("prepared trapdoor saved stage/owner count mismatch".into());
        }
        let [
            r,
            e,
            abar,
            rt,
            et,
            gram_a,
            gram_b,
            gram_d,
            product,
            sum,
            tail,
            mut gadget,
            mut identity,
        ] = Self::allocation_claims(params, d)
            .into_iter()
            .zip(layout.matrix_owners.iter())
            .map(|(claim, owner)| {
                GpuDCRTPolyMatrix::new_empty_with_owner_layout(
                    params,
                    claim.rows(),
                    claim.columns(),
                    claim.level().expect("matrix layout level"),
                    claim.is_evaluation().expect("matrix layout format"),
                    None,
                    owner,
                )
                .map(Arc::new)
            })
            .collect::<Result<Vec<_>, _>>()
            .and_then(|owners| {
                owners.try_into().map_err(|_| "trapdoor owner count mismatch".into())
            })?;
        Arc::get_mut(&mut gadget).expect("fresh gadget owner").fill_constant_columns(
            0..d,
            0..d * k,
            0,
            GpuMatrixRangeConstant::Gadget { small: false, digit_count: Some(k) },
        )?;
        Arc::get_mut(&mut identity).expect("fresh identity owner").fill_constant_columns(
            0..d,
            0..d,
            0,
            GpuMatrixRangeConstant::Identity,
        )?;
        let header = |owner: &Arc<GpuDCRTPolyMatrix>, evaluation| {
            GpuDCRTPolyMatrix::prepared_shape(
                Arc::clone(owner),
                owner.row_size(),
                owner.col_size(),
                owner.level(),
                evaluation,
            )
        };
        let stage_cursor = Cell::new(Some(TrapdoorStage::SamplerR));
        let consume_stage = |stage: TrapdoorStage| {
            let expected = stage_cursor.get();
            if expected != Some(stage) {
                return Err(match expected {
                    Some(expected) => format!(
                        "prepared trapdoor stage order mismatch: expected {}, got {}",
                        expected.index(),
                        stage.index(),
                    ),
                    None => format!(
                        "prepared trapdoor stage {} is after the end of the bind tape",
                        stage.index()
                    ),
                });
            }
            stage_cursor.set(stage.next());
            layout
                .stages
                .get(stage.index())
                .ok_or_else(|| format!("prepared trapdoor stage {} is missing", stage.index()))
        };
        let arithmetic = |stage: TrapdoorStage,
                          kind,
                          lhs: &Arc<GpuDCRTPolyMatrix>,
                          rhs: Option<&Arc<GpuDCRTPolyMatrix>>,
                          output: &Arc<GpuDCRTPolyMatrix>| {
            GpuPreparedArithmetic::bind_with_view_and_layout(
                kind,
                Arc::clone(lhs),
                rhs.cloned(),
                Arc::clone(output),
                None,
                0,
                consume_stage(stage)?,
            )
        };
        let samplers = [
            GpuPreparedSampling::bind_with_layout(
                Arc::clone(&r),
                GpuMatrixSampleDist::Gauss,
                sigma,
                u64::MAX,
                d * k,
                0,
                None,
                consume_stage(TrapdoorStage::SamplerR)?,
            )?,
            GpuPreparedSampling::bind_with_layout(
                Arc::clone(&e),
                GpuMatrixSampleDist::Gauss,
                sigma,
                u64::MAX,
                d * k,
                0,
                None,
                consume_stage(TrapdoorStage::SamplerE)?,
            )?,
            GpuPreparedSampling::bind_with_layout(
                Arc::clone(&abar),
                GpuMatrixSampleDist::Uniform,
                0.0,
                0,
                d,
                0,
                None,
                consume_stage(TrapdoorStage::SamplerAbar)?,
            )?,
        ];
        let transposes = [
            GpuPreparedTranspose::bind_with_layout(
                Arc::clone(&r),
                Arc::clone(&rt),
                None,
                consume_stage(TrapdoorStage::TransposeR)?.clone(),
            )?,
            GpuPreparedTranspose::bind_with_layout(
                Arc::clone(&e),
                Arc::clone(&et),
                None,
                consume_stage(TrapdoorStage::TransposeE)?.clone(),
            )?,
        ];
        let gram_owners = [gram_a, gram_b, gram_d];
        let grams = [
            arithmetic(
                TrapdoorStage::GramA,
                GpuPreparedArithmeticKind::Multiply,
                &r,
                Some(&rt),
                &gram_owners[0],
            )?,
            arithmetic(
                TrapdoorStage::GramB,
                GpuPreparedArithmeticKind::Multiply,
                &r,
                Some(&et),
                &gram_owners[1],
            )?,
            arithmetic(
                TrapdoorStage::GramD,
                GpuPreparedArithmeticKind::Multiply,
                &e,
                Some(&et),
                &gram_owners[2],
            )?,
        ];
        let coefficients = [
            header(&gram_owners[0], false)?,
            header(&gram_owners[1], false)?,
            header(&gram_owners[2], false)?,
        ];
        let gram_transforms = [
            (
                GpuPreparedTransform::new_with_layout(
                    &coefficients[0],
                    false,
                    consume_stage(TrapdoorStage::GramATransform)?,
                )?,
                Arc::clone(&coefficients[0]),
            ),
            (
                GpuPreparedTransform::new_with_layout(
                    &coefficients[1],
                    false,
                    consume_stage(TrapdoorStage::GramBTransform)?,
                )?,
                Arc::clone(&coefficients[1]),
            ),
            (
                GpuPreparedTransform::new_with_layout(
                    &coefficients[2],
                    false,
                    consume_stage(TrapdoorStage::GramDTransform)?,
                )?,
                Arc::clone(&coefficients[2]),
            ),
        ];
        let own_header = |owner: &Arc<GpuDCRTPolyMatrix>, evaluation| {
            Arc::try_unwrap(header(owner, evaluation)?)
                .map_err(|_| "fresh prepared trapdoor header is unexpectedly shared".to_owned())
        };
        let trapdoor = Arc::new(GpuDCRTTrapdoor {
            r: own_header(&r, true)?,
            e: own_header(&e, true)?,
            a_mat_coeff: own_header(&gram_owners[0], false)?,
            b_mat_coeff: own_header(&gram_owners[1], false)?,
            d_mat_coeff: own_header(&gram_owners[2], false)?,
            p1_covariance_cache: Arc::new(Mutex::new(None)),
        });
        let mut public_commands = vec![
            arithmetic(
                TrapdoorStage::PublicProduct,
                GpuPreparedArithmeticKind::Multiply,
                &abar,
                Some(&r),
                &product,
            )?,
            arithmetic(
                TrapdoorStage::PublicSum,
                GpuPreparedArithmeticKind::Add,
                &product,
                Some(&e),
                &sum,
            )?,
            arithmetic(
                TrapdoorStage::PublicTail,
                GpuPreparedArithmeticKind::Subtract,
                &gadget,
                Some(&sum),
                &tail,
            )?,
        ];
        for ((source, start), stage) in [(abar, 0), (identity, d), (tail, 2 * d)].into_iter().zip([
            TrapdoorStage::PublicCopyAbar,
            TrapdoorStage::PublicCopyIdentity,
            TrapdoorStage::PublicCopyTail,
        ]) {
            public_commands.push(GpuPreparedArithmetic::bind_with_view_and_layout(
                GpuPreparedArithmeticKind::Copy,
                Arc::clone(&source),
                None,
                Arc::clone(&public),
                Some(GpuPreparedView {
                    left: GpuPreparedRange { rows: 0..d, columns: 0..source.col_size() },
                    right: GpuPreparedRange { rows: 0..d, columns: 0..source.col_size() },
                    output: GpuPreparedRange {
                        rows: 0..d,
                        columns: start..start + source.col_size(),
                    },
                }),
                0,
                consume_stage(stage)?,
            )?);
        }
        if let Some(stage) = stage_cursor.get() {
            return Err(format!("prepared trapdoor bind left stage {} unconsumed", stage.index()));
        }
        Ok(Self {
            samplers,
            transposes,
            grams,
            gram_transforms,
            public_commands: public_commands.into_boxed_slice(),
            trapdoor,
            public,
        })
    }

    pub fn submit(&mut self, seeds: [GpuRngSeed; 3]) -> Result<(), String> {
        for (sampler, seed) in self.samplers.iter().zip(seeds) {
            sampler.submit(seed)?;
        }
        for command in &self.transposes {
            command.submit()?;
        }
        for command in &self.grams {
            command.submit()?;
        }
        for (command, owner) in &self.gram_transforms {
            command.submit_shared(owner)?;
        }
        for command in &self.public_commands {
            command.submit()?;
        }
        Ok(())
    }

    pub fn trapdoor(&self) -> &Arc<GpuDCRTTrapdoor> {
        &self.trapdoor
    }

    /// Upload a recorded public matrix and complete trapdoor payload into the
    /// fixed owners of this command. This is the prepared replay boundary;
    /// no sampler invocation or destination allocation is performed.
    pub fn load_replay_bytes(
        &self,
        public_bytes: &[u8],
        trapdoor_bytes: &[u8],
    ) -> Result<(), String> {
        self.public.load_compact_bytes(public_bytes)?;
        self.trapdoor.load_compact_bytes(trapdoor_bytes)
    }

    pub fn validate_replay_trapdoor_bytes(&self, bytes: &[u8]) -> Result<(), String> {
        // Validation is performed against the exact existing destination
        // owners without constructing another GPU trapdoor.
        let mut offset = 0usize;
        let next = |offset: &mut usize| -> Result<&[u8], String> {
            let end = offset.checked_add(8).ok_or("trapdoor payload length overflow")?;
            if end > bytes.len() {
                return Err("trapdoor payload is truncated".into());
            }
            let mut len_bytes = [0u8; 8];
            len_bytes.copy_from_slice(&bytes[*offset..end]);
            let len = usize::try_from(u64::from_le_bytes(len_bytes))
                .map_err(|_| "trapdoor payload length overflow")?;
            let start = end;
            let end = start.checked_add(len).ok_or("trapdoor payload length overflow")?;
            if end > bytes.len() {
                return Err("trapdoor payload is truncated".into());
            }
            *offset = end;
            Ok(&bytes[start..end])
        };
        let payloads = [
            next(&mut offset)?,
            next(&mut offset)?,
            next(&mut offset)?,
            next(&mut offset)?,
            next(&mut offset)?,
        ];
        if offset != bytes.len() {
            return Err("trapdoor payload has trailing bytes".into());
        }
        for (payload, owner) in payloads.into_iter().zip([
            &self.trapdoor().r,
            &self.trapdoor().e,
            &self.trapdoor().a_mat_coeff,
            &self.trapdoor().b_mat_coeff,
            &self.trapdoor().d_mat_coeff,
        ]) {
            GpuDCRTPolyMatrix::validate_compact_bytes(
                payload,
                owner.row_size(),
                owner.col_size(),
                owner.level(),
                owner.params().ring_dimension() as usize,
                owner.is_ntt(),
            )?;
        }
        Ok(())
    }
    pub fn public(&self) -> &Arc<GpuDCRTPolyMatrix> {
        &self.public
    }

    pub fn schedule(&self) -> Result<GpuPreparedSchedule, String> {
        let mut plans = Vec::new();
        plans.extend(self.samplers.iter().map(|plan| GpuPreparedSchedulePlan::Sampling(plan)));
        plans.extend(self.transposes.iter().map(|plan| GpuPreparedSchedulePlan::Transpose(plan)));
        plans.extend(self.grams.iter().map(GpuPreparedSchedulePlan::Arithmetic));
        for (transform, owner) in &self.gram_transforms {
            plans.push(GpuPreparedSchedulePlan::Transform(transform, owner));
        }
        plans.extend(self.public_commands.iter().map(GpuPreparedSchedulePlan::Arithmetic));
        GpuPreparedSchedule::new(&plans, &[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{poly::dcrt::params::DCRTPolyParams, sampler::trapdoor::gpu::gpu_params_from_cpu};

    #[test]
    #[serial_test::serial]
    fn test_gpu_prepared_trapdoor_preserves_gadget_identity_on_reuse() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let d = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(1);
        let params = gpu_params_from_cpu(&DCRTPolyParams::new(n, 2, 17, 2, None, None));
        let public = Arc::new(GpuDCRTPolyMatrix::new_empty_with_state(
            &params,
            d,
            d * (2 + params.modulus_digits()),
            params.crt_depth() - 1,
            true,
            None,
        ));
        let layout = GpuPreparedTrapdoorSampler::plan_layout_bundle(&params, &public, 3.2).unwrap();
        let mut plan =
            GpuPreparedTrapdoorSampler::bind_with_layout(&params, Arc::clone(&public), 3.2, layout)
                .unwrap();
        let identity = GpuDCRTPolyMatrix::identity(&params, d * params.modulus_digits(), None);
        let gadget = GpuDCRTPolyMatrix::gadget_matrix(&params, d, None);
        for _ in 0..2 {
            let seeds = std::array::from_fn(|_| GpuRngSeed::from_bytes(rand::random()));
            #[cfg(feature = "gpu-instrumentation")]
            {
                crate::poly::dcrt::gpu::gpu_test_reset_work_counters();
                crate::poly::dcrt::gpu::gpu_test_set_work_gate(true);
            }
            let submitted = plan.submit(seeds);
            #[cfg(feature = "gpu-instrumentation")]
            {
                crate::poly::dcrt::gpu::gpu_test_set_work_gate(false);
                let (events, streams, validations, allocations, launches, measurements) =
                    crate::poly::dcrt::gpu::gpu_test_work_counters();
                assert_eq!(
                    (events, streams, validations, allocations, measurements),
                    (0, 0, 0, 0, 0)
                );
                assert!(launches > 0);
            }
            submitted.unwrap();
            let trapdoor = plan.trapdoor();
            let secret = trapdoor.r.concat_rows(&[&trapdoor.e, &identity]);
            assert_eq!(&*public * &secret, gadget);
        }
    }
}
