//! Prepared preimage sequence. Allocation and binding happen only in `bind`.

use super::{GpuDCRTTrapdoor, preimage_c, preimage_seed, preimage_smoothing_parameter};
use crate::{
    matrix::{
        PolyMatrix, SmallMatrixError, SmallPolyMatrix,
        gpu_dcrt_poly::{
            GpuDCRTPolyMatrix, GpuMatrixSampleDist, GpuPreparedArithmetic,
            GpuPreparedArithmeticCommand, GpuPreparedArithmeticKind, GpuPreparedInputCopy,
            GpuPreparedPreimageCutoff, GpuPreparedPreimagePhases, GpuPreparedPreimagePhasesLayout,
            GpuPreparedRange, GpuPreparedSampling, GpuPreparedSchedule, GpuPreparedSchedulePlan,
            GpuPreparedSlotKind, GpuPreparedTransform, GpuPreparedTranspose, GpuPreparedView,
            GpuPreparedWorkspaceLayout, GpuSmallMatrix, GpuTracedClaim, PreparedAllocationKind,
            PreparedAllocationLayout, PreparedOwnerLayout, PreparedPlanLayout,
        },
    },
    poly::{PolyParams, dcrt::gpu::GpuDCRTPolyParams},
    sampler::bounds::default_preimage_cutoff,
};
use std::{cell::Cell, sync::Arc};

/// Warmup-owned structural description of a prepared preimage invocation.
/// `sampler` is the native descriptor consumed by the random P2 sampler; the
/// remaining fields preserve the ordered phase/cutoff workspace contract for
/// admission and diagnostics without retaining any CUDA allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuPreparedPreimageLayout {
    pub sampler: PreparedPlanLayout,
    pub matrix_owners: Box<[PreparedOwnerLayout]>,
    /// Native substage descriptors in the exact order consumed by bind.
    /// Index 10 is the P2 sampler; all other entries are matrix/NTT stages.
    pub stages: Box<[PreparedPlanLayout]>,
    pub matrix_claims: Box<[GpuTracedClaim]>,
    pub phases: GpuPreparedPreimagePhasesLayout,
    pub cutoff_workspaces: Box<[GpuPreparedWorkspaceLayout]>,
    pub attempts: usize,
    pub rows: usize,
    pub columns: usize,
    pub column_start: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuPreparedPreimageEntryKind {
    MatrixOwner,
    InputCopy,
    Transpose,
    Arithmetic,
    Transform,
    Sampler,
    Phase,
    Cutoff,
}

pub const PREIMAGE_ENTRY_KINDS: [GpuPreparedPreimageEntryKind; 8] = [
    GpuPreparedPreimageEntryKind::MatrixOwner,
    GpuPreparedPreimageEntryKind::InputCopy,
    GpuPreparedPreimageEntryKind::Transpose,
    GpuPreparedPreimageEntryKind::Arithmetic,
    GpuPreparedPreimageEntryKind::Transform,
    GpuPreparedPreimageEntryKind::Sampler,
    GpuPreparedPreimageEntryKind::Phase,
    GpuPreparedPreimageEntryKind::Cutoff,
];

/// One entry in the composite bind tape.  The indices refer to the saved
/// owner/stage tables, so the table is usable by both the binder and a
/// no-device contract test without rediscovering native geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuPreparedPreimageBindEntry {
    pub kind: GpuPreparedPreimageEntryKind,
    pub owner: Option<usize>,
    pub stage: Option<usize>,
}

/// The fixed order of the native substages in a prepared preimage bind.
/// Keeping the order typed prevents the planner, binder, and admission tests
/// from silently acquiring different numeric stage tables.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreimageStage {
    InputR,
    InputE,
    TransposeR,
    TransposeE,
    GramA,
    GramB,
    GramD,
    GramATransform,
    GramBTransform,
    GramDTransform,
    P2Sampler,
    ProductR,
    ProductE,
    ProductTransform,
    AssembleP1,
    AssembleP2,
    ResidualMultiply,
    ResidualSubtract,
    ResidualTransform,
    CorrectionR,
    CorrectionE,
    PublishP1,
    PublishP2,
    CandidateTransform,
}

impl PreimageStage {
    pub const ALL: [Self; 24] = [
        Self::InputR,
        Self::InputE,
        Self::TransposeR,
        Self::TransposeE,
        Self::GramA,
        Self::GramB,
        Self::GramD,
        Self::GramATransform,
        Self::GramBTransform,
        Self::GramDTransform,
        Self::P2Sampler,
        Self::ProductR,
        Self::ProductE,
        Self::ProductTransform,
        Self::AssembleP1,
        Self::AssembleP2,
        Self::ResidualMultiply,
        Self::ResidualSubtract,
        Self::ResidualTransform,
        Self::CorrectionR,
        Self::CorrectionE,
        Self::PublishP1,
        Self::PublishP2,
        Self::CandidateTransform,
    ];

    pub const fn index(self) -> usize {
        match self {
            Self::InputR => 0,
            Self::InputE => 1,
            Self::TransposeR => 2,
            Self::TransposeE => 3,
            Self::GramA => 4,
            Self::GramB => 5,
            Self::GramD => 6,
            Self::GramATransform => 7,
            Self::GramBTransform => 8,
            Self::GramDTransform => 9,
            Self::P2Sampler => 10,
            Self::ProductR => 11,
            Self::ProductE => 12,
            Self::ProductTransform => 13,
            Self::AssembleP1 => 14,
            Self::AssembleP2 => 15,
            Self::ResidualMultiply => 16,
            Self::ResidualSubtract => 17,
            Self::ResidualTransform => 18,
            Self::CorrectionR => 19,
            Self::CorrectionE => 20,
            Self::PublishP1 => 21,
            Self::PublishP2 => 22,
            Self::CandidateTransform => 23,
        }
    }

    pub const fn kind(self) -> GpuPreparedPreimageEntryKind {
        match self {
            Self::InputR | Self::InputE => GpuPreparedPreimageEntryKind::InputCopy,
            Self::TransposeR | Self::TransposeE => GpuPreparedPreimageEntryKind::Transpose,
            Self::GramA |
            Self::GramB |
            Self::GramD |
            Self::ProductR |
            Self::ProductE |
            Self::AssembleP1 |
            Self::AssembleP2 |
            Self::ResidualMultiply |
            Self::ResidualSubtract |
            Self::CorrectionR |
            Self::CorrectionE |
            Self::PublishP1 |
            Self::PublishP2 => GpuPreparedPreimageEntryKind::Arithmetic,
            Self::GramATransform |
            Self::GramBTransform |
            Self::GramDTransform |
            Self::ProductTransform |
            Self::ResidualTransform |
            Self::CandidateTransform => GpuPreparedPreimageEntryKind::Transform,
            Self::P2Sampler => GpuPreparedPreimageEntryKind::Sampler,
        }
    }
}

const fn preimage_stage_kinds() -> [GpuPreparedPreimageEntryKind; 24] {
    [
        PreimageStage::InputR.kind(),
        PreimageStage::InputE.kind(),
        PreimageStage::TransposeR.kind(),
        PreimageStage::TransposeE.kind(),
        PreimageStage::GramA.kind(),
        PreimageStage::GramB.kind(),
        PreimageStage::GramD.kind(),
        PreimageStage::GramATransform.kind(),
        PreimageStage::GramBTransform.kind(),
        PreimageStage::GramDTransform.kind(),
        PreimageStage::P2Sampler.kind(),
        PreimageStage::ProductR.kind(),
        PreimageStage::ProductE.kind(),
        PreimageStage::ProductTransform.kind(),
        PreimageStage::AssembleP1.kind(),
        PreimageStage::AssembleP2.kind(),
        PreimageStage::ResidualMultiply.kind(),
        PreimageStage::ResidualSubtract.kind(),
        PreimageStage::ResidualTransform.kind(),
        PreimageStage::CorrectionR.kind(),
        PreimageStage::CorrectionE.kind(),
        PreimageStage::PublishP1.kind(),
        PreimageStage::PublishP2.kind(),
        PreimageStage::CandidateTransform.kind(),
    ]
}

pub const PREIMAGE_STAGE_KINDS: [GpuPreparedPreimageEntryKind; 24] = preimage_stage_kinds();

impl GpuPreparedPreimageLayout {
    pub fn entry_kinds() -> &'static [GpuPreparedPreimageEntryKind; 8] {
        &PREIMAGE_ENTRY_KINDS
    }

    pub fn bind_entries() -> Vec<GpuPreparedPreimageBindEntry> {
        let mut entries = (0..16)
            .map(|owner| GpuPreparedPreimageBindEntry {
                kind: GpuPreparedPreimageEntryKind::MatrixOwner,
                owner: Some(owner),
                stage: None,
            })
            .collect::<Vec<_>>();
        entries.extend(PreimageStage::ALL.into_iter().map(|stage| GpuPreparedPreimageBindEntry {
            kind: stage.kind(),
            owner: None,
            stage: Some(stage.index()),
        }));
        entries.push(GpuPreparedPreimageBindEntry {
            kind: GpuPreparedPreimageEntryKind::Phase,
            owner: None,
            stage: None,
        });
        entries.push(GpuPreparedPreimageBindEntry {
            kind: GpuPreparedPreimageEntryKind::Cutoff,
            owner: None,
            stage: None,
        });
        entries
    }
}

impl GpuPreparedPreimageLayout {
    pub fn streams(&self) -> Vec<crate::matrix::gpu_dcrt_poly::PreparedStreamFootprint> {
        let mut streams = self
            .stages
            .iter()
            .take(PreimageStage::ResidualTransform.index() + 1)
            .flat_map(|stage| stage.streams().iter().copied())
            .collect::<Vec<_>>();
        streams.extend(self.phases.p1_ntt.streams().iter().copied());
        streams.extend(self.phases.gadget_ntt.streams().iter().copied());
        streams.extend(
            self.stages
                .iter()
                .skip(PreimageStage::CorrectionR.index())
                .flat_map(|stage| stage.streams().iter().copied()),
        );
        streams
    }

    pub fn claims(&self) -> Vec<GpuTracedClaim> {
        let mut claims = self.matrix_claims.to_vec();
        let append = |claims: &mut Vec<GpuTracedClaim>, allocation: &PreparedAllocationLayout| {
            if allocation.kind != 100 {
                if allocation.kind == 0 {
                    claims.push(GpuTracedClaim::matrix(
                        allocation.rows,
                        allocation.columns,
                        usize::try_from(allocation.level).unwrap_or(0),
                        allocation.format == crate::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL,
                    ));
                } else if let Some(kind) = allocation.allocation_kind() {
                    let kind = match kind {
                        PreparedAllocationKind::BatchWorkspace => {
                            GpuPreparedSlotKind::BatchWorkspace
                        }
                        PreparedAllocationKind::TransformWorkspace => {
                            GpuPreparedSlotKind::TransformWorkspace
                        }
                        PreparedAllocationKind::PinnedHost => GpuPreparedSlotKind::PinnedHost,
                        PreparedAllocationKind::CompactPayload => {
                            GpuPreparedSlotKind::CompactPayload
                        }
                        PreparedAllocationKind::CompactWorkspace => {
                            GpuPreparedSlotKind::CompactWorkspace
                        }
                        PreparedAllocationKind::SamplerWorkspace => {
                            GpuPreparedSlotKind::SamplerWorkspace
                        }
                        PreparedAllocationKind::TransferWorkspace => {
                            GpuPreparedSlotKind::TransferWorkspace
                        }
                        PreparedAllocationKind::CompletionEvent => {
                            GpuPreparedSlotKind::CompletionEvent
                        }
                        PreparedAllocationKind::SubmissionStream => {
                            GpuPreparedSlotKind::SubmissionStream
                        }
                        PreparedAllocationKind::Matrix | PreparedAllocationKind::HostOnly => return,
                    };
                    claims.push(GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                        kind,
                        bytes: allocation.bytes,
                        alignment: allocation.alignment.max(1),
                    }));
                }
            }
        };
        for stage in self.stages.iter().take(PreimageStage::ResidualTransform.index() + 1) {
            for allocation in stage.allocations() {
                append(&mut claims, allocation);
            }
        }
        claims.extend(
            self.phases
                .workspaces
                .iter()
                .copied()
                .filter(|layout| {
                    layout.bytes != 0 || layout.kind == GpuPreparedSlotKind::CompletionEvent
                })
                .map(GpuTracedClaim::workspace),
        );
        for ntt in [&self.phases.p1_ntt, &self.phases.gadget_ntt] {
            for allocation in ntt.allocations() {
                if allocation.kind == 100 {
                    continue;
                }
                if let Some(kind) = allocation.allocation_kind() {
                    if let PreparedAllocationKind::Matrix | PreparedAllocationKind::HostOnly = kind
                    {
                        continue;
                    }
                    let kind = match kind {
                        PreparedAllocationKind::BatchWorkspace => {
                            GpuPreparedSlotKind::BatchWorkspace
                        }
                        PreparedAllocationKind::TransformWorkspace => {
                            GpuPreparedSlotKind::TransformWorkspace
                        }
                        PreparedAllocationKind::PinnedHost => GpuPreparedSlotKind::PinnedHost,
                        PreparedAllocationKind::CompactPayload => {
                            GpuPreparedSlotKind::CompactPayload
                        }
                        PreparedAllocationKind::CompactWorkspace => {
                            GpuPreparedSlotKind::CompactWorkspace
                        }
                        PreparedAllocationKind::SamplerWorkspace => {
                            GpuPreparedSlotKind::SamplerWorkspace
                        }
                        PreparedAllocationKind::TransferWorkspace => {
                            GpuPreparedSlotKind::TransferWorkspace
                        }
                        PreparedAllocationKind::CompletionEvent => {
                            GpuPreparedSlotKind::CompletionEvent
                        }
                        PreparedAllocationKind::SubmissionStream => {
                            GpuPreparedSlotKind::SubmissionStream
                        }
                        PreparedAllocationKind::Matrix | PreparedAllocationKind::HostOnly => {
                            unreachable!()
                        }
                    };
                    claims.push(GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                        kind,
                        bytes: allocation.bytes,
                        alignment: allocation.alignment.max(1),
                    }));
                }
            }
        }
        for stage in self.stages.iter().skip(PreimageStage::CorrectionR.index()) {
            for allocation in stage.allocations() {
                append(&mut claims, allocation);
            }
        }
        claims.extend(self.cutoff_workspaces.iter().copied().map(GpuTracedClaim::workspace));
        claims
    }

    /// Native allocation descriptors in the exact same order as `claims`.
    /// `None` marks a logical owner or a phase/cutoff workspace whose claim
    /// has no native allocation key; this parallel table is consumed by the
    /// runtime admission layer to bind stream footprints to their exact
    /// device/partition/limb/role descriptor.
    pub fn claim_layouts(&self) -> Vec<Option<PreparedAllocationLayout>> {
        let mut layouts = vec![None; self.matrix_claims.len()];
        let append = |layouts: &mut Vec<Option<PreparedAllocationLayout>>,
                      allocation: &PreparedAllocationLayout| {
            if allocation.kind == 100 {
                return;
            }
            if allocation.kind == 0 {
                layouts.push(Some(*allocation));
            } else if let Some(kind) = allocation.allocation_kind() {
                if !matches!(
                    kind,
                    PreparedAllocationKind::Matrix | PreparedAllocationKind::HostOnly
                ) {
                    layouts.push(Some(*allocation));
                }
            }
        };
        for stage in self.stages.iter().take(PreimageStage::ResidualTransform.index() + 1) {
            for allocation in stage.allocations() {
                append(&mut layouts, allocation);
            }
        }
        layouts.extend(
            self.phases
                .workspaces
                .iter()
                .filter(|layout| {
                    layout.bytes != 0 || layout.kind == GpuPreparedSlotKind::CompletionEvent
                })
                .map(|_| None),
        );
        for ntt in [&self.phases.p1_ntt, &self.phases.gadget_ntt] {
            for allocation in ntt.allocations() {
                append(&mut layouts, allocation);
            }
        }
        for stage in self.stages.iter().skip(PreimageStage::CorrectionR.index()) {
            for allocation in stage.allocations() {
                append(&mut layouts, allocation);
            }
        }
        layouts.extend(self.cutoff_workspaces.iter().map(|_| None));
        layouts
    }

    /// Exact logical owner for every entry returned by [`Self::claims`].
    /// Matrix/stage allocations use the exact owner embedded in their native
    /// stage descriptor; phase workspaces run on the P1 descriptor owner and
    /// cutoff workspaces run on the candidate-transform descriptor owner.
    pub fn claim_owner_layouts(&self) -> Result<Vec<PreparedOwnerLayout>, String> {
        let mut owners = self.matrix_owners.to_vec();
        let should_append = |allocation: &PreparedAllocationLayout| {
            allocation.kind != 100 &&
                (allocation.kind == 0 ||
                    allocation.allocation_kind().is_some_and(|kind| {
                        !matches!(
                            kind,
                            PreparedAllocationKind::Matrix | PreparedAllocationKind::HostOnly
                        )
                    }))
        };
        for stage in self.stages.iter().take(PreimageStage::ResidualTransform.index() + 1) {
            for allocation in stage.allocations() {
                if should_append(allocation) {
                    owners.push(
                        stage.owner_layout().ok_or(
                            "prepared preimage stage owner layout is missing or conflicting",
                        )?,
                    );
                }
            }
        }
        let phase_owner = self
            .phases
            .p1_ntt
            .owner_layout()
            .ok_or("prepared preimage phase owner layout is missing or conflicting")?;
        let phase_claim_count = self
            .phases
            .workspaces
            .iter()
            .filter(|layout| {
                layout.bytes != 0 || layout.kind == GpuPreparedSlotKind::CompletionEvent
            })
            .count();
        owners.extend(std::iter::repeat_n(phase_owner, phase_claim_count));
        for ntt in [&self.phases.p1_ntt, &self.phases.gadget_ntt] {
            for allocation in ntt.allocations() {
                if should_append(allocation) {
                    owners.push(
                        ntt.owner_layout().ok_or(
                            "prepared preimage NTT owner layout is missing or conflicting",
                        )?,
                    );
                }
            }
        }
        for stage in self.stages.iter().skip(PreimageStage::CorrectionR.index()) {
            for allocation in stage.allocations() {
                if should_append(allocation) {
                    owners.push(
                        stage.owner_layout().ok_or(
                            "prepared preimage stage owner layout is missing or conflicting",
                        )?,
                    );
                }
            }
        }
        owners.extend(std::iter::repeat_n(
            self.stages[PreimageStage::CandidateTransform.index()]
                .owner_layout()
                .ok_or("prepared preimage cutoff owner layout is missing or conflicting")?,
            self.cutoff_workspaces.len(),
        ));
        if owners.len() != self.claims().len() {
            return Err("prepared preimage claim owner table length mismatch".into());
        }
        Ok(owners)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GpuPreparedPreimageError {
    #[error("{0}")]
    Sampling(#[from] SmallMatrixError),
    #[error("prepared preimage GPU failure: {0}")]
    Gpu(String),
}

pub struct GpuPreparedPreimageSampler {
    inputs: [GpuPreparedInputCopy; 2],
    transposes: [Arc<GpuPreparedTranspose>; 2],
    grams: [GpuPreparedArithmeticCommand; 3],
    gram_transforms: [(GpuPreparedTransform, Arc<GpuDCRTPolyMatrix>); 3],
    p2: Arc<GpuPreparedSampling>,
    product: [GpuPreparedArithmeticCommand; 2],
    product_transform: (GpuPreparedTransform, Arc<GpuDCRTPolyMatrix>),
    phases: GpuPreparedPreimagePhases,
    assemble: [GpuPreparedArithmeticCommand; 2],
    residual: [GpuPreparedArithmeticCommand; 2],
    residual_transform: (GpuPreparedTransform, Arc<GpuDCRTPolyMatrix>),
    correction: [GpuPreparedArithmeticCommand; 2],
    publish: [GpuPreparedArithmeticCommand; 2],
    candidate_transform: (GpuPreparedTransform, Arc<GpuDCRTPolyMatrix>),
    attempts: usize,
    column_start: usize,
    columns: usize,
    submitted: bool,
    // All masked sampler plans above drop before their acceptance owner.
    cutoff: GpuPreparedPreimageCutoff,
}

impl GpuPreparedPreimageSampler {
    fn plan_stages(
        params: &GpuDCRTPolyParams,
        owners: &[PreparedOwnerLayout],
        rows: usize,
        columns: usize,
    ) -> Result<Box<[PreparedPlanLayout]>, String> {
        if owners.len() != 16 {
            return Err("prepared preimage owner count mismatch".into());
        }
        let k = params.modulus_digits();
        let level = params.crt_depth() - 1;
        let device =
            params.device_ids().first().copied().ok_or("prepared preimage has no device")?;
        let matrix_format = crate::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL;
        let arith = |kind: i32,
                     left_rows: usize,
                     left_columns: usize,
                     right_rows: usize,
                     right_columns: usize,
                     output_rows: usize,
                     output_columns: usize,
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
        let input = |owner: &PreparedOwnerLayout, rows, columns| {
            PreparedPlanLayout::input_copy_with_owner(
                params,
                rows,
                columns,
                level,
                matrix_format,
                owner,
            )
        };
        let transpose = |owner: &PreparedOwnerLayout,
                         source_rows,
                         source_columns,
                         output_rows,
                         output_columns| {
            PreparedPlanLayout::transpose_with_owner(
                params,
                source_rows,
                source_columns,
                output_rows,
                output_columns,
                level,
                matrix_format,
                owner,
            )
        };
        let ntt = |owner: &PreparedOwnerLayout, rows, columns| {
            PreparedPlanLayout::ntt_with_owner(params, rows, columns, level, None, false, owner)
        };
        let mut stages = Vec::with_capacity(24);
        macro_rules! push_stage {
            ($stage:expr, $layout:expr) => {{
                let stage = $stage;
                if stages.len() != stage.index() {
                    return Err(format!(
                        "prepared preimage planner stage order mismatch at {}",
                        stage.index()
                    ));
                }
                stages.push($layout);
            }};
        }
        push_stage!(PreimageStage::InputR, input(&owners[0], rows, rows * k)?);
        push_stage!(PreimageStage::InputE, input(&owners[1], rows, rows * k)?);
        push_stage!(
            PreimageStage::TransposeR,
            transpose(&owners[2], rows, rows * k, rows * k, rows)?
        );
        push_stage!(
            PreimageStage::TransposeE,
            transpose(&owners[3], rows, rows * k, rows * k, rows)?
        );
        push_stage!(
            PreimageStage::GramA,
            arith(4, rows, rows * k, rows * k, rows, rows, rows, &owners[4])?
        );
        push_stage!(
            PreimageStage::GramB,
            arith(4, rows, rows * k, rows * k, rows, rows, rows, &owners[5])?
        );
        push_stage!(
            PreimageStage::GramD,
            arith(4, rows, rows * k, rows * k, rows, rows, rows, &owners[6])?
        );
        push_stage!(PreimageStage::GramATransform, ntt(&owners[4], rows * rows, 1)?);
        push_stage!(PreimageStage::GramBTransform, ntt(&owners[5], rows * rows, 1)?);
        push_stage!(PreimageStage::GramDTransform, ntt(&owners[6], rows * rows, 1)?);
        push_stage!(
            PreimageStage::P2Sampler,
            PreparedPlanLayout::sampling_with_owner(
                params,
                rows * k,
                columns,
                columns,
                0,
                level,
                crate::poly::dcrt::gpu::GPU_POLY_FORMAT_COEFF,
                crate::poly::dcrt::gpu::GPU_MATRIX_DIST_GAUSS,
                &owners[7],
            )?
        );
        push_stage!(
            PreimageStage::ProductR,
            arith(4, rows, rows * k, rows * k, columns, rows, columns, &owners[8])?
        );
        push_stage!(
            PreimageStage::ProductE,
            arith(4, rows, rows * k, rows * k, columns, rows, columns, &owners[8])?
        );
        push_stage!(PreimageStage::ProductTransform, ntt(&owners[8], 2 * rows * columns, 1)?);
        push_stage!(
            PreimageStage::AssembleP1,
            arith(0, 2 * rows, columns, 2 * rows, columns, 2 * rows, columns, &owners[10])?
        );
        push_stage!(
            PreimageStage::AssembleP2,
            arith(0, rows * k, columns, rows * k, columns, rows * k, columns, &owners[10])?
        );
        push_stage!(
            PreimageStage::ResidualMultiply,
            arith(4, rows * (2 + k), columns, 2 * rows, columns, rows, columns, &owners[11],)?
        );
        push_stage!(
            PreimageStage::ResidualSubtract,
            arith(5, rows, columns, rows, columns, rows, columns, &owners[12])?
        );
        push_stage!(PreimageStage::ResidualTransform, ntt(&owners[12], rows * columns, 1)?);
        push_stage!(
            PreimageStage::CorrectionR,
            arith(4, rows, rows * k, rows * k, columns, rows, columns, &owners[14])?
        );
        push_stage!(
            PreimageStage::CorrectionE,
            arith(4, rows, rows * k, rows * k, columns, rows, columns, &owners[14])?
        );
        push_stage!(
            PreimageStage::PublishP1,
            arith(1, 2 * rows, columns, 2 * rows, columns, 2 * rows, columns, &owners[15],)?
        );
        push_stage!(
            PreimageStage::PublishP2,
            arith(1, rows * k, columns, rows * k, columns, rows * k, columns, &owners[15],)?
        );
        push_stage!(PreimageStage::CandidateTransform, ntt(&owners[15], rows * (2 + k), columns)?);
        Ok(stages.into_boxed_slice())
    }

    fn planned_matrix_owners(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
    ) -> Result<Box<[PreparedOwnerLayout]>, String> {
        let claims = Self::matrix_layout(params, rows, columns);
        claims
            .into_iter()
            .map(|claim| {
                PreparedOwnerLayout::plan(
                    params,
                    claim.rows(),
                    claim.columns(),
                    claim.level().ok_or("preimage matrix claim has no level")?,
                    claim.is_evaluation().map_or(
                        crate::poly::dcrt::gpu::GPU_POLY_FORMAT_COEFF,
                        |evaluation| {
                            if evaluation {
                                crate::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL
                            } else {
                                crate::poly::dcrt::gpu::GPU_POLY_FORMAT_COEFF
                            }
                        },
                    ),
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Vec::into_boxed_slice)
    }

    fn matrix_layout(params: &GpuDCRTPolyParams, d: usize, columns: usize) -> [GpuTracedClaim; 16] {
        let k = params.modulus_digits();
        [
            (d, d * k),
            (d, d * k),
            (d * k, d),
            (d * k, d),
            (d, d),
            (d, d),
            (d, d),
            (d * k, columns),
            (2 * d, columns),
            (2 * d, columns),
            (d * (2 + k), columns),
            (d, columns),
            (d, columns),
            (d * k, columns),
            (2 * d, columns),
            (d * (2 + k), columns),
        ]
        .map(|(rows, columns)| GpuTracedClaim::matrix(rows, columns, params.crt_depth() - 1, true))
    }

    /// Plan the fixed Gaussian sampler stage.  This is metadata-only and is
    /// intended to run during warmup, before any native prepared command is
    /// bound. [`Self::bind_with_layout`] consumes the returned descriptor.
    pub fn plan_layout(
        params: &GpuDCRTPolyParams,
        public: &GpuDCRTPolyMatrix,
        target: &GpuDCRTPolyMatrix,
        output: &GpuSmallMatrix,
        sigma: f64,
    ) -> Result<PreparedPlanLayout, String> {
        let d = public.row_size();
        let columns = target.col_size();
        let k = params.modulus_digits();
        if d == 0 ||
            columns == 0 ||
            target.row_size() != d ||
            public.col_size() != d * (2 + k) ||
            output.size() != (d * (2 + k), columns) ||
            !sigma.is_finite() ||
            sigma <= 0.0
        {
            return Err("prepared preimage layout contract mismatch".into());
        }
        // P2 is the only submit-time random payload.  Its descriptor carries
        // the exact stream/event geometry and is consumed by the native
        // sampler bind below; all other fixed stages remain owner-bound.
        Self::plan_layout_for_shape(params, d, columns)
    }

    /// Metadata-only planner used by the resource resolver, before any live
    /// matrix owner or compact output exists.
    pub fn plan_layout_for_shape(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
    ) -> Result<PreparedPlanLayout, String> {
        let k = params.modulus_digits();
        if rows == 0 || columns == 0 {
            return Err("prepared preimage layout shape is empty".into());
        }
        let owner = PreparedOwnerLayout::plan(
            params,
            rows * k,
            columns,
            params.crt_depth() - 1,
            crate::poly::dcrt::gpu::GPU_POLY_FORMAT_COEFF,
        )?;
        PreparedPlanLayout::sampling_with_owner(
            params,
            rows * k,
            columns,
            columns,
            0,
            params.crt_depth() - 1,
            crate::poly::dcrt::gpu::GPU_POLY_FORMAT_COEFF,
            crate::poly::dcrt::gpu::GPU_MATRIX_DIST_GAUSS,
            &owner,
        )
    }

    /// Save the complete ordered preimage layout used by warmup admission.
    /// This includes all fixed matrix owners followed by phase and cutoff
    /// workspaces; no native owner or CUDA resource is created here.
    pub fn plan_layout_bundle(
        params: &GpuDCRTPolyParams,
        public: &GpuDCRTPolyMatrix,
        target: &GpuDCRTPolyMatrix,
        output: &GpuSmallMatrix,
        sigma: f64,
        column_start: usize,
    ) -> Result<GpuPreparedPreimageLayout, String> {
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err("prepared preimage layout contract mismatch".into());
        }
        let matrix_owners =
            Self::planned_matrix_owners(params, public.row_size(), target.col_size())?;
        let sampler = PreparedPlanLayout::sampling_with_owner(
            params,
            public.row_size() * params.modulus_digits(),
            target.col_size(),
            target.col_size(),
            0,
            params.crt_depth() - 1,
            crate::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL,
            crate::poly::dcrt::gpu::GPU_MATRIX_DIST_GAUSS,
            &matrix_owners[7],
        )?;
        let mut stages =
            Self::plan_stages(params, &matrix_owners, public.row_size(), target.col_size())?;
        stages[PreimageStage::P2Sampler.index()] = sampler.clone();
        let phases = GpuPreparedPreimagePhases::plan_layout(
            params,
            public.row_size(),
            target.col_size(),
            Self::gadget_digits(params),
            &matrix_owners[9],
            &matrix_owners[13],
        )?;
        Ok(GpuPreparedPreimageLayout {
            stages,
            sampler: sampler.clone(),
            matrix_owners,
            matrix_claims: Self::matrix_layout(params, public.row_size(), target.col_size())
                .to_vec()
                .into_boxed_slice(),
            phases,
            cutoff_workspaces: GpuPreparedPreimageCutoff::allocation_layout(output, 1)?
                .into_boxed_slice(),
            attempts: crate::env::gpu_preimage_max_tile_attempts()?,
            rows: public.row_size(),
            columns: target.col_size(),
            column_start,
        })
    }

    /// Shape-only form of [`Self::plan_layout_bundle`] used by the runtime
    /// resolver before concrete matrix owners are materialized.  The native
    /// cutoff planner is fed the exact compact geometry, so this descriptor is
    /// byte-for-byte identical to the owner-backed warmup bundle.
    pub fn plan_layout_bundle_for_shape(
        params: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        magnitude_bytes: usize,
        _sigma: f64,
        column_start: usize,
    ) -> Result<GpuPreparedPreimageLayout, String> {
        let k = params.modulus_digits();
        if rows == 0 || columns == 0 || magnitude_bytes == 0 {
            return Err("prepared preimage layout shape is empty".into());
        }
        let matrix_owners = Self::planned_matrix_owners(params, rows, columns)?;
        let sampler = PreparedPlanLayout::sampling_with_owner(
            params,
            rows * k,
            columns,
            columns,
            0,
            params.crt_depth() - 1,
            crate::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL,
            crate::poly::dcrt::gpu::GPU_MATRIX_DIST_GAUSS,
            &matrix_owners[7],
        )?;
        let mut stages = Self::plan_stages(params, &matrix_owners, rows, columns)?;
        stages[PreimageStage::P2Sampler.index()] = sampler.clone();
        let phases = GpuPreparedPreimagePhases::plan_layout(
            params,
            rows,
            columns,
            Self::gadget_digits(params),
            &matrix_owners[9],
            &matrix_owners[13],
        )?;
        Ok(GpuPreparedPreimageLayout {
            stages,
            sampler: sampler.clone(),
            matrix_owners,
            matrix_claims: Self::matrix_layout(params, rows, columns).to_vec().into_boxed_slice(),
            phases,
            cutoff_workspaces: GpuPreparedPreimageCutoff::allocation_layout_for_shape(
                params,
                rows * (2 + k),
                columns,
                magnitude_bytes,
                1,
            )?
            .into_boxed_slice(),
            attempts: crate::env::gpu_preimage_max_tile_attempts()?,
            rows,
            columns,
            column_start,
        })
    }

    pub fn bind_with_layout(
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        public: Arc<GpuDCRTPolyMatrix>,
        target: Arc<GpuDCRTPolyMatrix>,
        output: Arc<GpuSmallMatrix>,
        sigma: f64,
        layout: GpuPreparedPreimageLayout,
    ) -> Result<Self, String> {
        if layout.rows != public.row_size() || layout.columns != target.col_size() {
            return Err("prepared preimage saved layout shape mismatch".into());
        }
        if layout.attempts != crate::env::gpu_preimage_max_tile_attempts()? {
            return Err("prepared preimage saved layout retry count mismatch".into());
        }
        Self::bind_with_saved_layout(
            params,
            trapdoor,
            public,
            target,
            output,
            sigma,
            layout.column_start,
            layout,
        )
    }

    fn gadget_digits(params: &GpuDCRTPolyParams) -> usize {
        let bits = params
            .moduli()
            .iter()
            .map(|modulus| (u64::BITS - modulus.leading_zeros()) as usize)
            .max()
            .unwrap_or(0);
        bits.saturating_add(params.base_bits() as usize).saturating_sub(1) /
            params.base_bits().max(1) as usize
    }

    fn bind_with_saved_layout(
        params: &GpuDCRTPolyParams,
        trapdoor: &GpuDCRTTrapdoor,
        public: Arc<GpuDCRTPolyMatrix>,
        target: Arc<GpuDCRTPolyMatrix>,
        output: Arc<GpuSmallMatrix>,
        sigma: f64,
        column_start: usize,
        layout: GpuPreparedPreimageLayout,
    ) -> Result<Self, String> {
        let d = public.row_size();
        let columns = target.col_size();
        let k = params.modulus_digits();
        let base = 1u32.checked_shl(params.base_bits()).ok_or("preimage base overflow")?;
        let minimum = default_preimage_cutoff(params.ring_dimension(), d, k, base, sigma)
            .ok_or("invalid prepared preimage parameters")?;
        if params.dropped_moduli() != 0 ||
            d == 0 ||
            columns == 0 ||
            target.row_size() != d ||
            public.col_size() != d * (2 + k) ||
            trapdoor.r.size() != (d, d * k) ||
            trapdoor.e.size() != (d, d * k) ||
            output.size() != (d * (2 + k), columns) ||
            output.max_coefficient_bound() < &minimum ||
            public.params() != params ||
            target.params() != params ||
            output.params() != params ||
            trapdoor.r.params() != params ||
            trapdoor.e.params() != params ||
            !public.is_ntt() ||
            !target.is_ntt()
        {
            return Err("prepared preimage input/output contract mismatch".into());
        }
        let attempts = crate::env::gpu_preimage_max_tile_attempts()?;
        let c = preimage_c(base, sigma);
        let smoothing =
            preimage_smoothing_parameter(base, sigma, d, params.ring_dimension() as usize, k);
        if layout.matrix_owners.len() != 16 {
            return Err("prepared preimage saved owner count mismatch".into());
        }
        let bind_entries = GpuPreparedPreimageLayout::bind_entries();
        if bind_entries.len() != 42 ||
            bind_entries[..16].iter().enumerate().any(|(owner, entry)| {
                entry.kind != GpuPreparedPreimageEntryKind::MatrixOwner ||
                    entry.owner != Some(owner)
            }) ||
            bind_entries[16..40].iter().enumerate().any(|(stage, entry)| {
                entry.stage != Some(stage) || entry.kind != PREIMAGE_STAGE_KINDS[stage]
            })
        {
            return Err("prepared preimage bind tape contract mismatch".into());
        }
        let [
            r,
            e,
            rt,
            et,
            gram_a,
            gram_b,
            gram_d,
            p2_owner,
            product_owner,
            p1_owner,
            perturbation,
            public_product,
            residual_owner,
            z,
            correction_owner,
            candidate,
        ] = Self::matrix_layout(params, d, columns)
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
                owners.try_into().map_err(|_| "preimage owner count mismatch".into())
            })?;
        let coefficient = |owner: &Arc<GpuDCRTPolyMatrix>| {
            GpuDCRTPolyMatrix::prepared_shape(
                Arc::clone(owner),
                owner.row_size(),
                owner.col_size(),
                owner.level(),
                false,
            )
        };
        if layout.stages.len() != 24 {
            return Err("prepared preimage saved stage count mismatch".into());
        }
        let stage_index = Cell::new(PreimageStage::AssembleP1.index());
        let consume_stage = |stage: PreimageStage| {
            let expected = stage_index.get();
            if expected != stage.index() {
                return Err(format!(
                    "prepared preimage stage order mismatch: expected {expected}, got {}",
                    stage.index(),
                ));
            }
            stage_index.set(expected + 1);
            layout
                .stages
                .get(stage.index())
                .ok_or_else(|| format!("prepared preimage stage {} is missing", stage.index()))
        };
        let arithmetic = |stage: PreimageStage,
                          kind,
                          lhs: &Arc<GpuDCRTPolyMatrix>,
                          rhs: Option<&Arc<GpuDCRTPolyMatrix>>,
                          out: &Arc<GpuDCRTPolyMatrix>| {
            GpuPreparedArithmetic::bind_with_view_and_layout(
                kind,
                Arc::clone(lhs),
                rhs.cloned(),
                Arc::clone(out),
                None,
                0,
                consume_stage(stage)?,
            )
        };
        let ranged = |stage: PreimageStage,
                      kind,
                      lhs: &Arc<GpuDCRTPolyMatrix>,
                      rhs: Option<&Arc<GpuDCRTPolyMatrix>>,
                      out: &Arc<GpuDCRTPolyMatrix>,
                      left_rows,
                      right_rows,
                      output_rows| {
            GpuPreparedArithmetic::bind_with_view_and_layout(
                kind,
                Arc::clone(lhs),
                rhs.cloned(),
                Arc::clone(out),
                Some(GpuPreparedView {
                    left: GpuPreparedRange { rows: left_rows, columns: 0..lhs.col_size() },
                    right: GpuPreparedRange {
                        rows: right_rows,
                        columns: 0..rhs.map_or(lhs.col_size(), |value| value.col_size()),
                    },
                    output: GpuPreparedRange { rows: output_rows, columns: 0..out.col_size() },
                }),
                0,
                consume_stage(stage)?,
            )
        };
        let inputs = [
            GpuPreparedInputCopy::bind_with_layout(
                Arc::clone(&r),
                Arc::clone(&r),
                None,
                layout.stages[PreimageStage::InputR.index()].clone(),
            )?,
            GpuPreparedInputCopy::bind_with_layout(
                Arc::clone(&e),
                Arc::clone(&e),
                None,
                layout.stages[PreimageStage::InputE.index()].clone(),
            )?,
        ];
        let transposes = [
            GpuPreparedTranspose::bind_with_layout(
                Arc::clone(&r),
                Arc::clone(&rt),
                None,
                layout.stages[PreimageStage::TransposeR.index()].clone(),
            )?,
            GpuPreparedTranspose::bind_with_layout(
                Arc::clone(&e),
                Arc::clone(&et),
                None,
                layout.stages[PreimageStage::TransposeE.index()].clone(),
            )?,
        ];
        let gram_owners = [gram_a, gram_b, gram_d];
        let grams = [
            GpuPreparedArithmetic::bind_with_view_and_layout(
                GpuPreparedArithmeticKind::Multiply,
                Arc::clone(&r),
                Some(Arc::clone(&rt)),
                Arc::clone(&gram_owners[0]),
                None,
                0,
                &layout.stages[PreimageStage::GramA.index()],
            )?,
            GpuPreparedArithmetic::bind_with_view_and_layout(
                GpuPreparedArithmeticKind::Multiply,
                Arc::clone(&r),
                Some(Arc::clone(&et)),
                Arc::clone(&gram_owners[1]),
                None,
                0,
                &layout.stages[PreimageStage::GramB.index()],
            )?,
            GpuPreparedArithmetic::bind_with_view_and_layout(
                GpuPreparedArithmeticKind::Multiply,
                Arc::clone(&e),
                Some(Arc::clone(&et)),
                Arc::clone(&gram_owners[2]),
                None,
                0,
                &layout.stages[PreimageStage::GramD.index()],
            )?,
        ];
        let gram_coeff = [
            coefficient(&gram_owners[0])?,
            coefficient(&gram_owners[1])?,
            coefficient(&gram_owners[2])?,
        ];
        let gram_transforms = [
            (
                GpuPreparedTransform::new_with_layout(
                    &gram_coeff[0],
                    false,
                    &layout.stages[PreimageStage::GramATransform.index()],
                )?,
                Arc::clone(&gram_coeff[0]),
            ),
            (
                GpuPreparedTransform::new_with_layout(
                    &gram_coeff[1],
                    false,
                    &layout.stages[PreimageStage::GramBTransform.index()],
                )?,
                Arc::clone(&gram_coeff[1]),
            ),
            (
                GpuPreparedTransform::new_with_layout(
                    &gram_coeff[2],
                    false,
                    &layout.stages[PreimageStage::GramDTransform.index()],
                )?,
                Arc::clone(&gram_coeff[2]),
            ),
        ];
        let p2 = GpuPreparedSampling::bind_with_layout(
            Arc::clone(&p2_owner),
            GpuMatrixSampleDist::Gauss,
            (smoothing * smoothing - c * c).sqrt(),
            u64::MAX,
            columns,
            0,
            None,
            &layout.stages[PreimageStage::P2Sampler.index()],
        )?;
        let product = [
            GpuPreparedArithmetic::bind_with_view_and_layout(
                GpuPreparedArithmeticKind::Multiply,
                Arc::clone(&r),
                Some(Arc::clone(&p2_owner)),
                Arc::clone(&product_owner),
                Some(GpuPreparedView {
                    left: GpuPreparedRange { rows: 0..d, columns: 0..r.col_size() },
                    right: GpuPreparedRange { rows: 0..d * k, columns: 0..p2_owner.col_size() },
                    output: GpuPreparedRange { rows: 0..d, columns: 0..product_owner.col_size() },
                }),
                0,
                &layout.stages[PreimageStage::ProductR.index()],
            )?,
            GpuPreparedArithmetic::bind_with_view_and_layout(
                GpuPreparedArithmeticKind::Multiply,
                Arc::clone(&e),
                Some(Arc::clone(&p2_owner)),
                Arc::clone(&product_owner),
                Some(GpuPreparedView {
                    left: GpuPreparedRange { rows: 0..d, columns: 0..e.col_size() },
                    right: GpuPreparedRange { rows: 0..d * k, columns: 0..p2_owner.col_size() },
                    output: GpuPreparedRange {
                        rows: d..2 * d,
                        columns: 0..product_owner.col_size(),
                    },
                }),
                0,
                &layout.stages[PreimageStage::ProductE.index()],
            )?,
        ];
        let product_coeff = coefficient(&product_owner)?;
        let product_transform = (
            GpuPreparedTransform::new_with_layout(
                &product_coeff,
                false,
                &layout.stages[PreimageStage::ProductTransform.index()],
            )?,
            Arc::clone(&product_coeff),
        );
        let assemble = [
            ranged(
                PreimageStage::AssembleP1,
                GpuPreparedArithmeticKind::Copy,
                &p1_owner,
                None,
                &perturbation,
                0..2 * d,
                0..2 * d,
                0..2 * d,
            )?,
            ranged(
                PreimageStage::AssembleP2,
                GpuPreparedArithmeticKind::Copy,
                &p2_owner,
                None,
                &perturbation,
                0..d * k,
                0..d * k,
                2 * d..d * (2 + k),
            )?,
        ];
        let residual = [
            arithmetic(
                PreimageStage::ResidualMultiply,
                GpuPreparedArithmeticKind::Multiply,
                &public,
                Some(&perturbation),
                &public_product,
            )?,
            arithmetic(
                PreimageStage::ResidualSubtract,
                GpuPreparedArithmeticKind::Subtract,
                &target,
                Some(&public_product),
                &residual_owner,
            )?,
        ];
        let residual_coeff = coefficient(&residual_owner)?;
        let residual_transform = (
            GpuPreparedTransform::new_with_layout(
                &residual_coeff,
                false,
                consume_stage(PreimageStage::ResidualTransform)?,
            )?,
            Arc::clone(&residual_coeff),
        );
        let mut phases = GpuPreparedPreimagePhases::bind_with_layout(
            [
                Arc::clone(&gram_coeff[0]),
                Arc::clone(&gram_coeff[1]),
                Arc::clone(&gram_coeff[2]),
                product_coeff,
                Arc::clone(&p1_owner),
                residual_coeff,
                Arc::clone(&z),
            ],
            params.base_bits(),
            c,
            smoothing,
            sigma,
            &layout.phases,
        )?;
        let correction = [
            ranged(
                PreimageStage::CorrectionR,
                GpuPreparedArithmeticKind::Multiply,
                &r,
                Some(&z),
                &correction_owner,
                0..d,
                0..d * k,
                0..d,
            )?,
            ranged(
                PreimageStage::CorrectionE,
                GpuPreparedArithmeticKind::Multiply,
                &e,
                Some(&z),
                &correction_owner,
                0..d,
                0..d * k,
                d..2 * d,
            )?,
        ];
        let publish = [
            ranged(
                PreimageStage::PublishP1,
                GpuPreparedArithmeticKind::Add,
                &p1_owner,
                Some(&correction_owner),
                &candidate,
                0..2 * d,
                0..2 * d,
                0..2 * d,
            )?,
            ranged(
                PreimageStage::PublishP2,
                GpuPreparedArithmeticKind::Add,
                &p2_owner,
                Some(&z),
                &candidate,
                0..d * k,
                0..d * k,
                2 * d..d * (2 + k),
            )?,
        ];
        let candidate_coeff = coefficient(&candidate)?;
        let candidate_transform = (
            GpuPreparedTransform::new_with_layout(
                &candidate_coeff,
                false,
                consume_stage(PreimageStage::CandidateTransform)?,
            )?,
            Arc::clone(&candidate_coeff),
        );
        if stage_index.get() != PreimageStage::ALL.len() {
            return Err(format!(
                "prepared preimage bind left {} unconsumed stages",
                PreimageStage::ALL.len().saturating_sub(stage_index.get()),
            ));
        }
        let cutoff_layout = layout.cutoff_workspaces.as_ref();
        let cutoff = GpuPreparedPreimageCutoff::bind_with_layout(
            vec![(output, candidate_coeff, 0, 0)],
            cutoff_layout,
        )?;
        // This struct drops the phase/sampler owners before cutoff; submission
        // is exclusive through &mut self and all readers precede each update.
        unsafe {
            phases.bind_acceptance(&cutoff, 0)?;
            p2.bind_preimage_acceptance(&cutoff, 0)?;
        }
        Ok(Self {
            inputs,
            transposes,
            grams,
            gram_transforms,
            p2,
            product,
            product_transform,
            phases,
            assemble,
            residual,
            residual_transform,
            correction,
            publish,
            candidate_transform,
            attempts,
            column_start,
            columns,
            submitted: false,
            cutoff,
        })
    }

    pub fn submit(&mut self, trapdoor: &GpuDCRTTrapdoor, seed: [u8; 32]) -> Result<(), String> {
        self.cutoff.begin()?;
        self.submitted = true;
        self.inputs[0].submit_borrowed(&trapdoor.r)?;
        self.inputs[1].submit_borrowed(&trapdoor.e)?;
        for command in &self.transposes {
            command.submit()?;
        }
        for command in &self.grams {
            command.submit()?;
        }
        for (transform, owner) in &self.gram_transforms {
            transform.submit_shared(owner)?;
        }
        self.phases.refresh_covariance()?;
        for attempt in 0..self.attempts {
            let candidate = preimage_seed(seed, b"candidate", self.column_start, attempt);
            let perturb = preimage_seed(candidate.to_bytes(), b"perturb", 0, 0);
            self.p2.submit(preimage_seed(perturb.to_bytes(), b"p2", 0, 0))?;
            for command in &self.product {
                command.submit()?;
            }
            self.product_transform.0.submit_shared(&self.product_transform.1)?;
            self.phases.sample_p1(preimage_seed(perturb.to_bytes(), b"p1", 0, 0))?;
            for command in &self.assemble {
                command.submit()?;
            }
            for command in &self.residual {
                command.submit()?;
            }
            self.residual_transform.0.submit_shared(&self.residual_transform.1)?;
            self.phases.sample_gadget(preimage_seed(candidate.to_bytes(), b"z", 0, 0))?;
            for command in &self.correction {
                command.submit()?;
            }
            for command in &self.publish {
                command.submit()?;
            }
            self.candidate_transform.0.submit_shared(&self.candidate_transform.1)?;
            self.cutoff.submit()?;
        }
        Ok(())
    }

    pub fn wait(&mut self) -> Result<(), GpuPreparedPreimageError> {
        if !self.submitted {
            return Ok(());
        }
        if self.cutoff.wait().map_err(GpuPreparedPreimageError::Gpu)?[0] == 0 {
            return Err(SmallMatrixError::AttemptExhausted {
                column_start: self.column_start,
                column_count: self.columns,
                attempts: self.attempts,
            }
            .into());
        }
        Ok(())
    }

    pub fn is_ready(&mut self) -> Result<bool, GpuPreparedPreimageError> {
        if !self.submitted {
            return Ok(true);
        }
        let Some(status) = self.cutoff.poll().map_err(GpuPreparedPreimageError::Gpu)? else {
            return Ok(false);
        };
        if status[0] == 0 {
            return Err(SmallMatrixError::AttemptExhausted {
                column_start: self.column_start,
                column_count: self.columns,
                attempts: self.attempts,
            }
            .into());
        }
        Ok(true)
    }

    pub fn schedule(&self) -> Result<GpuPreparedSchedule, String> {
        let mut plans = Vec::new();
        plans.extend(self.inputs.iter().map(GpuPreparedSchedulePlan::InputCopy));
        plans.extend(self.transposes.iter().map(|plan| GpuPreparedSchedulePlan::Transpose(plan)));
        for commands in [
            &self.grams[..],
            &self.product,
            &self.assemble,
            &self.residual,
            &self.correction,
            &self.publish,
        ] {
            plans.extend(commands.iter().map(GpuPreparedSchedulePlan::Arithmetic));
        }
        for (transform, owner) in self.gram_transforms.iter().chain([
            &self.product_transform,
            &self.residual_transform,
            &self.candidate_transform,
        ]) {
            plans.push(GpuPreparedSchedulePlan::Transform(transform, owner));
        }
        plans.push(GpuPreparedSchedulePlan::Sampling(&self.p2));
        plans.push(GpuPreparedSchedulePlan::PreimagePhases(&self.phases));
        plans.push(GpuPreparedSchedulePlan::PreimageCutoff(&self.cutoff));
        GpuPreparedSchedule::new(&plans, &[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        matrix::ResidentPolyMatrixColumnSource,
        poly::dcrt::params::DCRTPolyParams,
        sampler::{
            DistType, PolyTrapdoorSampler, PolyUniformSampler,
            gpu::GpuDCRTPolyUniformSampler,
            trapdoor::gpu::{GpuDCRTPolyTrapdoorSampler, gpu_params_from_cpu},
        },
    };

    #[test]
    #[serial_test::serial]
    fn test_gpu_prepared_preimage_matches_existing_sampler_and_changed_trapdoor() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let d = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(1);
        let params = gpu_params_from_cpu(&DCRTPolyParams::new(n, 2, 17, 2, None, None));
        let sigma = 3.2;
        let sampler = GpuDCRTPolyTrapdoorSampler::new(&params, sigma);
        let bound =
            default_preimage_cutoff(n, d, params.modulus_digits(), 1 << params.base_bits(), sigma)
                .unwrap();
        let (trapdoor, public) = sampler.trapdoor(&params, d);
        let public = Arc::new(public);
        let target = Arc::new(GpuDCRTPolyUniformSampler::new().sample_uniform(
            &params,
            d,
            1,
            DistType::FinRingDist,
        ));
        let output = Arc::new(
            GpuSmallMatrix::new_empty(&params, d * (2 + params.modulus_digits()), 1, bound.clone())
                .unwrap(),
        );
        output.prepare_preimage_hard_cutoff();
        let layout = GpuPreparedPreimageSampler::plan_layout_bundle(
            &params, &public, &target, &output, sigma, 0,
        )
        .unwrap();
        let mut plan = GpuPreparedPreimageSampler::bind_with_layout(
            &params,
            &trapdoor,
            Arc::clone(&public),
            Arc::clone(&target),
            Arc::clone(&output),
            sigma,
            layout,
        )
        .unwrap();
        let seed = rand::random();
        let target_source = ResidentPolyMatrixColumnSource::new(target.as_ref().clone());
        let expected = sampler
            .preimage(&params, &trapdoor, public.as_ref(), &target_source, bound.clone(), seed)
            .unwrap();
        #[cfg(feature = "gpu-instrumentation")]
        {
            crate::poly::dcrt::gpu::gpu_test_reset_work_counters();
            crate::poly::dcrt::gpu::gpu_test_set_work_gate(true);
        }
        let submitted = plan.submit(&trapdoor, seed);
        #[cfg(feature = "gpu-instrumentation")]
        {
            crate::poly::dcrt::gpu::gpu_test_set_work_gate(false);
            let (events, streams, validations, allocations, launches, measurements) =
                crate::poly::dcrt::gpu::gpu_test_work_counters();
            assert_eq!((events, streams, validations, allocations, measurements), (0, 0, 0, 0, 0));
            assert!(launches > 0);
        }
        submitted.unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !plan.is_ready().unwrap() {
            assert!(std::time::Instant::now() < deadline, "prepared preimage completion timed out");
            std::thread::yield_now();
        }
        plan.wait().unwrap();
        assert_eq!(
            output.to_canonical_coefficients().unwrap(),
            expected.to_canonical_coefficients().unwrap()
        );

        let (next_trapdoor, next_public) = sampler.trapdoor(&params, d);
        let next_public = Arc::new(next_public);
        let owner_layout = public.prepared_owner_layout().unwrap();
        let copy_layout = PreparedPlanLayout::input_copy_with_owner(
            &params,
            next_public.row_size(),
            next_public.col_size(),
            next_public.level(),
            crate::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL,
            &owner_layout,
        )
        .unwrap();
        let copy = GpuPreparedInputCopy::bind_with_layout(
            Arc::clone(&public),
            Arc::clone(&next_public),
            None,
            copy_layout,
        )
        .unwrap();
        copy.submit_borrowed(&next_public).unwrap();
        let next_seed = rand::random();
        let expected = sampler
            .preimage(&params, &next_trapdoor, &next_public, &target_source, bound, next_seed)
            .unwrap();
        plan.submit(&next_trapdoor, next_seed).unwrap();
        plan.wait().unwrap();
        assert_eq!(
            output.to_canonical_coefficients().unwrap(),
            expected.to_canonical_coefficients().unwrap()
        );
    }
}
