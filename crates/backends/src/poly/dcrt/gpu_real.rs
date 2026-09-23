//! Pointer-stable real scalars and direct CUDA Graph arithmetic.

use std::{ffi::c_void, ptr};

use super::gpu::{
    GpuDCRTPolyParams, GpuDeviceBuffer, GpuNativeGraphError, GpuNativeLaunchStream,
    GpuRawControlStatusView, GpuRawIntegerView, GpuSignedValuesEncoding, last_error_string,
};

unsafe extern "C" {
    fn gpu_real_emit(
        ctx: *mut super::gpu::GpuContextOpaque,
        stream: *mut c_void,
        operation: u32,
        output: *mut f64,
        left: *const c_void,
        right: *const f64,
        left_integer_encoding: i32,
        constant_bits: u64,
        status: *mut u32,
        output_binding: u32,
        left_binding: u32,
        right_binding: u32,
        status_binding: u32,
    ) -> i32;
}

/// One fixed-address device f64. The allocation remains owned through graph
/// execution; uploads happen only between sequential executions.
pub struct GpuDeviceReal {
    buffer: GpuDeviceBuffer,
    params: GpuDCRTPolyParams,
    physical_device: i32,
}

impl GpuDeviceReal {
    pub fn new(
        params: &GpuDCRTPolyParams,
        physical_device: i32,
    ) -> Result<Self, GpuNativeGraphError> {
        let stream = params.native_launch_stream(physical_device)?;
        let buffer = GpuDeviceBuffer::allocate(&stream, 8)?;
        buffer.upload(0, &0f64.to_le_bytes())?;
        Ok(Self { buffer, params: params.clone(), physical_device })
    }

    pub fn upload_f64(&self, value: f64) -> Result<(), GpuNativeGraphError> {
        if !value.is_finite() {
            return Err(GpuNativeGraphError::Native("resident real must be finite".into()));
        }
        self.buffer.upload(0, &value.to_le_bytes())
    }

    /// Read only after the producer Graph completion event has been joined.
    pub fn read_f64(&self) -> Result<f64, GpuNativeGraphError> {
        let mut bytes = [0u8; 8];
        self.params.download_device_bytes(
            self.physical_device,
            self.device_address(),
            &mut bytes,
        )?;
        Ok(f64::from_le_bytes(bytes))
    }

    pub fn physical_device(&self) -> i32 {
        self.physical_device
    }
    pub fn device_address(&self) -> u64 {
        self.buffer.as_ptr() as u64
    }
    pub fn byte_len(&self) -> usize {
        8
    }

    pub fn prepare_graph_launch(
        &self,
        stream: &GpuNativeLaunchStream,
    ) -> Result<(), GpuNativeGraphError> {
        if stream.physical_device() != self.physical_device {
            return Err(GpuNativeGraphError::Native("real belongs to another GPU".into()));
        }
        self.buffer.wait_compiled_inputs(self.physical_device, stream, false)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuRawRealView {
    pub address: u64,
    pub binding: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuRawRealInput {
    Real(GpuRawRealView),
    Integer(GpuRawIntegerView),
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuRealOperation {
    CopyConstant = 0,
    IntToReal = 1,
    Add = 2,
    Subtract = 3,
    Multiply = 4,
    Divide = 5,
    Sqrt = 6,
    Copy = 7,
}

impl GpuDCRTPolyParams {
    /// Emit one allocation-free real operation. `status` is a separate
    /// four-byte owner reset before each Graph replay and checked afterward.
    pub fn emit_raw_real(
        &self,
        stream: &GpuNativeLaunchStream,
        operation: GpuRealOperation,
        output: GpuRawRealView,
        left: Option<GpuRawRealInput>,
        right: Option<GpuRawRealView>,
        constant: f64,
        status: GpuRawControlStatusView,
    ) -> Result<(), GpuNativeGraphError> {
        let (left_address, left_binding, integer_encoding) = match left {
            Some(GpuRawRealInput::Real(view)) if view.address != 0 => {
                (view.address, view.binding, 0)
            }
            Some(GpuRawRealInput::Integer(view)) if view.address != 0 && view.count == 1 => {
                let encoding = match view.encoding {
                    GpuSignedValuesEncoding::SignedI64 => 0,
                    GpuSignedValuesEncoding::CanonicalU64 => 1,
                    GpuSignedValuesEncoding::SignedWords(words)
                        if words > 0 && words <= (i32::MAX as usize - 2) =>
                    {
                        2 + words as i32
                    }
                    _ => {
                        return Err(GpuNativeGraphError::Native("invalid real integer width".into()))
                    }
                };
                (view.address, view.binding, encoding)
            }
            None => (0, 0, 0),
            _ => return Err(GpuNativeGraphError::Native("invalid real input".into())),
        };
        let shape_valid = match operation {
            GpuRealOperation::CopyConstant => {
                left.is_none() && right.is_none() && constant.is_finite()
            }
            GpuRealOperation::IntToReal => {
                matches!(left, Some(GpuRawRealInput::Integer(_))) && right.is_none()
            }
            GpuRealOperation::Sqrt => {
                matches!(left, Some(GpuRawRealInput::Real(_))) && right.is_none()
            }
            GpuRealOperation::Copy => {
                matches!(left, Some(GpuRawRealInput::Real(_))) && right.is_none()
            }
            GpuRealOperation::Add |
            GpuRealOperation::Subtract |
            GpuRealOperation::Multiply |
            GpuRealOperation::Divide => {
                matches!(left, Some(GpuRawRealInput::Real(_))) && right.is_some()
            }
        };
        if !shape_valid ||
            output.address == 0 ||
            status.address == 0 ||
            right.is_some_and(|view| view.address == 0)
        {
            return Err(GpuNativeGraphError::Native("invalid raw real view or operation".into()));
        }
        let result = unsafe {
            gpu_real_emit(
                self.ctx_raw(),
                stream.raw_ptr(),
                operation as u32,
                output.address as *mut f64,
                left_address as *const c_void,
                right.map_or(ptr::null(), |view| view.address as *const f64),
                integer_encoding,
                constant.to_bits(),
                status.address as *mut u32,
                output.binding,
                left_binding,
                right.map_or(0, |view| view.binding),
                status.binding,
            )
        };
        if result != 0 {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::gpu::{
            GpuExportStatus, GpuGraphBindingValue, GpuSignedValues, detected_gpu_device_ids,
        },
        *,
    };
    use num_bigint::BigInt;
    use serial_test::serial;

    #[test]
    #[serial]
    fn real_graph_replays_arithmetic_and_rejects_invalid_domain() {
        let Some(&device) = detected_gpu_device_ids().first() else {
            return;
        };
        let params = GpuDCRTPolyParams::new(8, vec![97], 3, None);
        let stream = params.native_launch_stream(device).unwrap();
        let left = GpuDeviceReal::new(&params, device).unwrap();
        let right = GpuDeviceReal::new(&params, device).unwrap();
        let output = GpuDeviceReal::new(&params, device).unwrap();
        let status = GpuExportStatus::new(&params, device).unwrap();
        left.upload_f64(9.0).unwrap();
        right.upload_f64(3.0).unwrap();
        left.prepare_graph_launch(&stream).unwrap();
        right.prepare_graph_launch(&stream).unwrap();
        output.prepare_graph_launch(&stream).unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        let mut builder = stream.begin_graph().unwrap();
        builder.begin_operation(0, &[]).unwrap();
        for (binding, address, bytes) in [
            (0, output.device_address(), 8),
            (1, left.device_address(), 8),
            (2, right.device_address(), 8),
            (3, status.device_address(), 4),
        ] {
            builder.bind_resident_address(address, bytes, binding).unwrap();
        }
        params
            .emit_raw_real(
                builder.launch_stream(),
                GpuRealOperation::Divide,
                GpuRawRealView { address: output.device_address(), binding: 0 },
                Some(GpuRawRealInput::Real(GpuRawRealView {
                    address: left.device_address(),
                    binding: 1,
                })),
                Some(GpuRawRealView { address: right.device_address(), binding: 2 }),
                0.0,
                GpuRawControlStatusView { address: status.device_address(), binding: 3 },
            )
            .unwrap();
        builder.finish_operation().unwrap();
        let mut graph = builder.finish().unwrap();
        graph
            .bind(&[
                GpuGraphBindingValue::DeviceAddress(output.device_address()),
                GpuGraphBindingValue::DeviceAddress(left.device_address()),
                GpuGraphBindingValue::DeviceAddress(right.device_address()),
                GpuGraphBindingValue::DeviceAddress(status.device_address()),
            ])
            .unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 0);
        assert_eq!(output.read_f64().unwrap(), 3.0);

        right.upload_f64(-0.0).unwrap();
        status.reset().unwrap();
        right.prepare_graph_launch(&stream).unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 2);
        assert_eq!(output.read_f64().unwrap(), 0.0);

        let integer = GpuSignedValues::from_bigints_with_words(
            &params,
            device,
            &[BigInt::from(9_007_199_254_740_993u64)],
            2,
        )
        .unwrap();
        let integer_address = integer.binding().unwrap().device_address;
        integer.wait_until_ready().unwrap();
        status.reset().unwrap();
        let mut builder = stream.begin_graph().unwrap();
        builder.begin_operation(0, &[]).unwrap();
        for (binding, address, bytes) in [
            (0, output.device_address(), 8),
            (1, integer_address, 24),
            (2, status.device_address(), 4),
        ] {
            builder.bind_resident_address(address, bytes, binding).unwrap();
        }
        params
            .emit_raw_real(
                builder.launch_stream(),
                GpuRealOperation::IntToReal,
                GpuRawRealView { address: output.device_address(), binding: 0 },
                Some(GpuRawRealInput::Integer(GpuRawIntegerView {
                    address: integer_address,
                    count: 1,
                    encoding: GpuSignedValuesEncoding::SignedWords(2),
                    binding: 1,
                })),
                None,
                0.0,
                GpuRawControlStatusView { address: status.device_address(), binding: 2 },
            )
            .unwrap();
        builder.finish_operation().unwrap();
        let mut integer_graph = builder.finish().unwrap();
        integer_graph
            .bind(&[
                GpuGraphBindingValue::DeviceAddress(output.device_address()),
                GpuGraphBindingValue::DeviceAddress(integer_address),
                GpuGraphBindingValue::DeviceAddress(status.device_address()),
            ])
            .unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        integer_graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 0);
        assert_eq!(output.read_f64().unwrap(), 9_007_199_254_740_992.0);

        left.upload_f64(-1.0).unwrap();
        status.reset().unwrap();
        let mut builder = stream.begin_graph().unwrap();
        builder.begin_operation(0, &[]).unwrap();
        for (binding, address, bytes) in [
            (0, output.device_address(), 8),
            (1, left.device_address(), 8),
            (2, status.device_address(), 4),
        ] {
            builder.bind_resident_address(address, bytes, binding).unwrap();
        }
        params
            .emit_raw_real(
                builder.launch_stream(),
                GpuRealOperation::Sqrt,
                GpuRawRealView { address: output.device_address(), binding: 0 },
                Some(GpuRawRealInput::Real(GpuRawRealView {
                    address: left.device_address(),
                    binding: 1,
                })),
                None,
                0.0,
                GpuRawControlStatusView { address: status.device_address(), binding: 2 },
            )
            .unwrap();
        builder.finish_operation().unwrap();
        let mut sqrt_graph = builder.finish().unwrap();
        sqrt_graph
            .bind(&[
                GpuGraphBindingValue::DeviceAddress(output.device_address()),
                GpuGraphBindingValue::DeviceAddress(left.device_address()),
                GpuGraphBindingValue::DeviceAddress(status.device_address()),
            ])
            .unwrap();
        left.prepare_graph_launch(&stream).unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        sqrt_graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 2);
        left.upload_f64(-0.0).unwrap();
        status.reset().unwrap();
        left.prepare_graph_launch(&stream).unwrap();
        status.prepare_graph_launch(&stream).unwrap();
        sqrt_graph.launch(&stream).unwrap().wait().unwrap();
        assert_eq!(status.read().unwrap(), 0);
        assert_eq!(output.read_f64().unwrap().to_bits(), (-0.0f64).to_bits());
    }
}
