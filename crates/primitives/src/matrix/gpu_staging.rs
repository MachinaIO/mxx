//! Allocation-free validation of the native RNS staging representation.

use super::{GpuDCRTPolyMatrix, rns_bytes_len_for_level};
use crate::poly::dcrt::gpu::GpuDCRTPolyParams;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuCpuStagingLayout {
    pub rows: usize,
    pub columns: usize,
    pub level: usize,
    pub is_ntt: bool,
    pub bytes_per_poly: usize,
}

impl GpuCpuStagingLayout {
    /// Construct an isolated zero input in the production staging format without
    /// allocating a GPU matrix or reading a production target. Callers choose
    /// the finite representative column range before constructing this payload.
    pub fn zero_bytes(&self, parameters: &GpuDCRTPolyParams) -> Result<Vec<u8>, String> {
        if self.level >= parameters.crt_depth() ||
            self.bytes_per_poly != rns_bytes_len_for_level(parameters, self.level)
        {
            return Err("GPU RNS staging layout does not match parameters".into());
        }
        let length = self
            .rows
            .checked_mul(self.columns)
            .and_then(|count| count.checked_mul(self.bytes_per_poly))
            .ok_or_else(|| "GPU RNS staging payload size overflow".to_owned())?;
        bincode::encode_to_vec(
            (
                1u8,
                self.rows,
                self.columns,
                self.level,
                self.is_ntt,
                self.bytes_per_poly,
                vec![0u8; length],
            ),
            bincode::config::standard(),
        )
        .map_err(|error| format!("GPU RNS staging encoding failed: {error}"))
    }
}

pub(super) fn decode<'a>(
    parameters: &GpuDCRTPolyParams,
    bytes: &'a [u8],
) -> Result<(GpuCpuStagingLayout, &'a [u8]), String> {
    let ((version, rows, columns, level, is_ntt, bytes_per_poly, payload), consumed): (
        (u8, usize, usize, usize, bool, usize, &[u8]),
        usize,
    ) = bincode::borrow_decode_from_slice(bytes, bincode::config::standard())
        .map_err(|error| format!("invalid GPU RNS staging bytes: {error}"))?;
    if version != 1 || consumed != bytes.len() || level >= parameters.crt_depth() {
        return Err("invalid GPU RNS staging version, level, or trailing data".into());
    }
    if bytes_per_poly != rns_bytes_len_for_level(parameters, level) {
        return Err("GPU RNS staging polynomial width does not match parameters".into());
    }
    let expected_bytes = rows
        .checked_mul(columns)
        .and_then(|size| size.checked_mul(bytes_per_poly))
        .ok_or_else(|| "GPU RNS staging payload size overflow".to_owned())?;
    if payload.len() != expected_bytes {
        return Err("GPU RNS staging payload size does not match its shape".into());
    }
    Ok((GpuCpuStagingLayout { rows, columns, level, is_ntt, bytes_per_poly }, payload))
}

impl GpuDCRTPolyMatrix {
    /// Read the same validated header used by the production staging loader.
    /// The borrowed payload is not copied, loaded onto a GPU, or sampled.
    pub fn cpu_staging_layout(
        parameters: &GpuDCRTPolyParams,
        bytes: &[u8],
    ) -> Result<GpuCpuStagingLayout, String> {
        decode(parameters, bytes).map(|(layout, _)| layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        matrix::PolyMatrix,
        poly::{PolyParams, dcrt::params::DCRTPolyParams},
    };

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_zero_staging_uses_validated_layout_and_production_loader() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let cpu = DCRTPolyParams::new(n, 2, 54, 8, None, None);
        let parameters = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        for is_ntt in [false, true] {
            let layout = GpuCpuStagingLayout {
                rows: 2,
                columns: 1,
                level: parameters.crt_depth() - 1,
                is_ntt,
                bytes_per_poly: rns_bytes_len_for_level(&parameters, parameters.crt_depth() - 1),
            };
            let bytes = layout.zero_bytes(&parameters).unwrap();
            assert_eq!(GpuDCRTPolyMatrix::cpu_staging_layout(&parameters, &bytes).unwrap(), layout);
            let loaded = GpuDCRTPolyMatrix::from_cpu_staging_bytes(&parameters, &bytes);
            assert_eq!(
                loaded.to_cpu_matrix(),
                crate::matrix::dcrt_poly::DCRTPolyMatrix::zero(&cpu, 2, 1)
            );
            assert!(
                GpuCpuStagingLayout { rows: usize::MAX, ..layout }.zero_bytes(&parameters).is_err()
            );
            assert!(
                GpuCpuStagingLayout { level: parameters.crt_depth(), ..layout }
                    .zero_bytes(&parameters)
                    .is_err()
            );
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_staging_layout_validates_without_loading_payload() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let cpu = DCRTPolyParams::new(n, 2, 54, 8, None, None);
        let parameters = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 8, None);
        let matrix = GpuDCRTPolyMatrix::identity(&parameters, 2, None);
        let bytes = matrix.to_cpu_staging_bytes();
        let (layout, payload) = decode(&parameters, &bytes).unwrap();
        assert_eq!((layout.rows, layout.columns), matrix.size());
        assert_eq!(layout.level, parameters.crt_depth() - 1);
        assert_eq!(payload.len(), 4 * layout.bytes_per_poly);
        assert_eq!(GpuDCRTPolyMatrix::cpu_staging_layout(&parameters, &bytes).unwrap(), layout);
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(decode(&parameters, &trailing).is_err());
        for (rows, level, width, length) in [
            (3usize, layout.level, layout.bytes_per_poly, payload.len()),
            (2, parameters.crt_depth(), layout.bytes_per_poly, payload.len()),
            (2, layout.level, layout.bytes_per_poly + 1, payload.len()),
            (2, layout.level, layout.bytes_per_poly, payload.len() - 1),
        ] {
            let changed = bincode::encode_to_vec(
                (1u8, rows, 2usize, level, layout.is_ntt, width, &payload[..length]),
                bincode::config::standard(),
            )
            .unwrap();
            assert!(decode(&parameters, &changed).is_err());
        }
    }
}
