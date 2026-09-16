//! Typed prepared values and fixed input normalization in reserved native storage.

use super::*;
use mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedRequest;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Serialize)]
pub enum PreparedMatrixSource {
    Shard(usize),
    Replica { device: usize, context: usize, evaluation: bool },
    Fragment { device: usize, context: usize, index: usize, evaluation: bool },
}

/// CPU-only column-owner layout, observed on an input or derived for an output.
/// It carries no native pointer or lease. Derived layouts describe coverage and
/// format only; they do not establish an actual matrix identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixInputFragment {
    pub device: i32,
    pub context: usize,
    pub start: usize,
    pub end: usize,
    pub level: usize,
    pub evaluation: bool,
}

impl PreparedMatrixSource {
    /// Shared source choice for metadata admission and concrete preparation.
    /// None means a replica is required; Fragment means a format conversion.
    pub(super) fn existing(
        fragments: impl IntoIterator<Item = MatrixInputFragment>,
        columns: std::ops::Range<usize>,
        device: usize,
        parameters: &GpuDCRTPolyParams,
        evaluation: bool,
    ) -> Option<Self> {
        fragments.into_iter().enumerate().find_map(|(index, fragment)| {
            (fragment.device == parameters.device_ids()[0] &&
                fragment.context == parameters.context_identity() &&
                fragment.start <= columns.start &&
                columns.end <= fragment.end)
                .then_some(if fragment.evaluation == evaluation {
                    Self::Shard(index)
                } else {
                    Self::Fragment {
                        device,
                        context: parameters.context_identity(),
                        index,
                        evaluation,
                    }
                })
        })
    }

    /// Lower source selection and every required copy/normalization from
    /// layout metadata alone. Order is significant: mixed-format fragments
    /// are normalized before the replica that consumes them.
    pub fn plan(
        fragments: impl Iterator<Item = MatrixInputFragment> + Clone,
        shape: (usize, usize),
        columns: std::ops::Range<usize>,
        device: usize,
        parameters: &GpuDCRTPolyParams,
        evaluation: bool,
    ) -> (Self, Vec<MatrixInputLayout>) {
        let source =
            Self::existing(fragments.clone(), columns, device, parameters, evaluation).unwrap_or(
                Self::Replica { device, context: parameters.context_identity(), evaluation },
            );
        if matches!(source, Self::Shard(_)) {
            return (source, Vec::new());
        }
        let first = fragments.clone().next().expect("nonempty matrix layout");
        let mixed = fragments.clone().any(|fragment| fragment.evaluation != first.evaluation);
        let mut preparation = Vec::new();
        if let Self::Replica { device, context, evaluation } = source &&
            mixed
        {
            for (index, fragment) in fragments.clone().enumerate() {
                if fragment.evaluation != evaluation {
                    preparation.push(MatrixInputLayout {
                        source: Self::Fragment { device, context, index, evaluation },
                        shape: (shape.0, fragment.end - fragment.start),
                        level: fragment.level,
                        evaluation: fragment.evaluation,
                    });
                }
            }
        }
        preparation.push(match source {
            Self::Fragment { index, .. } => {
                let fragment = fragments.clone().nth(index).expect("selected fragment");
                MatrixInputLayout {
                    source,
                    shape: (shape.0, fragment.end - fragment.start),
                    level: fragment.level,
                    evaluation: fragment.evaluation,
                }
            }
            Self::Replica { .. } => MatrixInputLayout {
                source,
                shape,
                level: first.level,
                evaluation: if mixed { evaluation } else { first.evaluation },
            },
            Self::Shard(_) => unreachable!("borrowed source needs no preparation"),
        });
        (source, preparation)
    }
}

/// A fully determined input allocation and its initial format, before its
/// required normalization. Both inventory and real preparation use this layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixInputLayout {
    pub source: PreparedMatrixSource,
    pub shape: (usize, usize),
    pub level: usize,
    pub evaluation: bool,
}

impl MatrixInputLayout {
    /// Assign the complete ordered preparation transaction without reserving or
    /// executing anything. Callers supply only preparations absent for this owner,
    /// with their actual source levels. Failure leaves the supplied inventory intact.
    pub fn assign(
        layouts: impl IntoIterator<Item = Self>,
        parameters: &GpuDCRTPolyParams,
        slots: &[(mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSlotSnapshot, bool)],
    ) -> Result<Option<Vec<(Self, GpuPreparedRequest)>>, String> {
        crate::backend::poly_gpu::record_prepared_forbidden(2);
        use mxx_primitives::matrix::gpu_dcrt_poly::{GpuPreparedSlotSnapshot, GpuTracedClaim};
        let mut layouts = layouts.into_iter().peekable();
        if layouts.peek().is_none() {
            return Ok(Some(Vec::new()));
        }
        let mut slots = slots.to_vec();
        let mut selected = Vec::new();
        // Normalize fragments before selecting the replica that consumes them.
        // Sequential choice preserves native size-class preference and prevents
        // two preparations from claiming the same backing slot.
        for layout in layouts {
            let claim = GpuTracedClaim::matrix(
                layout.shape.0,
                layout.shape.1,
                layout.level,
                layout.evaluation,
            );
            let Some(request) = GpuPreparedSlotSnapshot::assign(parameters, &slots, &[claim])?
                .into_iter()
                .next()
                .flatten()
            else {
                return Ok(None);
            };
            for (slot, eligible) in &mut slots {
                let id = slot.identity();
                *eligible &= (id.storage_id(), id.slot_id(), id.slot_index()) != request.slot_key();
            }
            selected.push((layout, request));
        }
        Ok(Some(selected))
    }
}

/// Immutable capacity view for one registered native parameter context.
/// It holds no backing storage or reservation. Eligibility is the observed or
/// hypothetical scenario; native commit independently rechecks every request.
pub struct MatrixSlotContext {
    pub device: usize,
    pub context: usize,
    pub slots: Vec<(mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSlotSnapshot, bool)>,
}

pub type MatrixSlotInventory = Vec<MatrixSlotContext>;

/// Input geometry and parameter handles, with no GPU payload or backing owner.
/// Native admission obtains this from its real inputs; hypothetical admission
/// supplies the same metadata from an explicit placement scenario.
#[derive(Clone)]
pub struct MatrixDescriptor {
    pub id: u64,
    pub rows: usize,
    pub columns: usize,
    pub shards: Vec<GpuColumnShard<MatrixFragmentDescriptor>>,
    pub input_layout: Arc<[MatrixInputFragment]>,
}

#[derive(Clone)]
pub struct MatrixFragmentDescriptor {
    pub parameters: GpuDCRTPolyParams,
    pub level: usize,
    pub columns: usize,
    pub evaluation: bool,
}

impl MatrixFragmentDescriptor {
    pub fn params(&self) -> &GpuDCRTPolyParams {
        &self.parameters
    }
    pub fn level(&self) -> usize {
        self.level
    }
    pub fn is_ntt(&self) -> bool {
        self.evaluation
    }
    pub fn columns_count(&self) -> usize {
        self.columns
    }

    pub fn registered_parameters<'a>(
        &self,
        backend: &'a DeviceBackend,
    ) -> Result<&'a GpuDCRTPolyParams, PolyBackendError> {
        let key = crate::backend::poly::RingKey {
            modulus: BigInt::from(self.parameters.modulus().as_ref().clone()),
            ring_dimension: self.parameters.ring_dimension() as usize,
        };
        backend.parameters[backend.active_placement]
            .get(&key)
            .ok_or(PolyBackendError::MissingParameters(key))
    }
}

impl MatrixDescriptor {
    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }
}

impl From<&GpuFleetMatrix> for MatrixDescriptor {
    fn from(matrix: &GpuFleetMatrix) -> Self {
        Self {
            id: matrix.id,
            rows: matrix.rows,
            columns: matrix.columns,
            input_layout: matrix.input_layout.clone(),
            shards: matrix
                .shards
                .iter()
                .map(|shard| GpuColumnShard {
                    device_id: shard.device_id,
                    global_column_start: shard.global_column_start,
                    value: MatrixFragmentDescriptor {
                        parameters: shard.value.params().clone(),
                        level: shard.value.level(),
                        columns: shard.value.col_size(),
                        evaluation: shard.value.is_ntt(),
                    },
                })
                .collect(),
        }
    }
}

impl From<&GpuFleetSmallMatrix> for MatrixDescriptor {
    fn from(matrix: &GpuFleetSmallMatrix) -> Self {
        let shards = matrix
            .shards
            .iter()
            .map(|shard| GpuColumnShard {
                device_id: shard.device_id,
                global_column_start: shard.global_column_start,
                value: MatrixFragmentDescriptor {
                    parameters: shard.value.params().clone(),
                    level: shard.value.params().crt_depth() - 1,
                    columns: shard.value.columns_count(),
                    evaluation: false,
                },
            })
            .collect::<Vec<_>>();
        let input_layout = shards
            .iter()
            .map(|shard| MatrixInputFragment {
                device: shard.device_id,
                context: shard.value.parameters.context_identity(),
                start: shard.global_column_start,
                end: shard.global_column_start + shard.value.columns,
                level: shard.value.level,
                evaluation: false,
            })
            .collect();
        Self { id: matrix.id, rows: matrix.rows, columns: matrix.columns, shards, input_layout }
    }
}

/// One selected copy/normalization, before native values or storage are bound.
#[derive(Clone)]
pub struct MatrixInputRequest {
    pub owner: u64,
    pub layout: MatrixInputLayout,
    pub parameters: GpuDCRTPolyParams,
    pub device: usize,
    pub request: GpuPreparedRequest,
}
