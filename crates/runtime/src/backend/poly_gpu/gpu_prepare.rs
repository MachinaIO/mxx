//! Typed prepared values and fixed input normalization in reserved native storage.

use super::*;
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

/// Input geometry and parameter handles, with no GPU payload or backing owner.
/// Native admission obtains this from its real inputs.
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
