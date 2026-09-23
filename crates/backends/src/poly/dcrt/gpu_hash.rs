use super::*;

#[repr(C)]
struct GpuRawHashTagSegmentAbi {
    kind: u32,
    operand_index: u32,
    static_offset: u64,
    static_length: u64,
}

#[repr(C)]
struct GpuRawHashPlanOpaque {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn gpu_raw_hash_plan_create(
        ctx: *mut GpuContextOpaque,
        physical_device: i32,
        stream: *mut c_void,
        moduli: *const u64,
        limb_count: usize,
        segments: *const GpuRawHashTagSegmentAbi,
        segment_count: usize,
        static_bytes: *const u8,
        static_byte_count: usize,
        operand_encodings: *const i32,
        operand_count: usize,
        out_plan: *mut *mut GpuRawHashPlanOpaque,
    ) -> c_int;
    fn gpu_raw_hash_plan_prepare_graph_launch(
        plan: *mut GpuRawHashPlanOpaque,
        stream: *mut c_void,
        operand_addresses: *const u64,
        operand_encodings: *const i32,
        operand_count: usize,
    ) -> c_int;
    fn gpu_raw_hash_plan_allocation_range(
        plan: *const GpuRawHashPlanOpaque,
        address: *mut u64,
        bytes: *mut usize,
    ) -> c_int;
    fn gpu_raw_hash_sample_emit(
        plan: *mut GpuRawHashPlanOpaque,
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        key: *const u8,
        destination: *const GpuRawMatrixViewAbi,
        status: *mut u32,
        key_binding: u32,
        destination_binding_base: u32,
        status_binding: u32,
    ) -> c_int;
    fn gpu_raw_hash_integers_emit(
        plan: *mut GpuRawHashPlanOpaque,
        ctx: *mut GpuContextOpaque,
        stream: *mut c_void,
        key: *const u8,
        destination: *mut u64,
        count: u64,
        words: usize,
        bits: usize,
        status: *mut u32,
        key_binding: u32,
        destination_binding: u32,
        status_binding: u32,
    ) -> c_int;
    fn gpu_raw_hash_plan_destroy(plan: *mut GpuRawHashPlanOpaque);
}

/// Ordered tag pieces after CPU-equivalent framing of constant components.
/// A dynamic component names one of the plan's resident scalar operands.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuHashTagPart {
    Static(Vec<u8>),
    Integer(usize),
    Decimal(usize),
    U64Le(usize),
}

impl GpuHashTagPart {
    /// The namespace prefix is copied verbatim before framed components.
    pub fn prefix(bytes: &[u8]) -> Self {
        Self::Static(bytes.to_vec())
    }

    pub fn bytes_component(bytes: &[u8]) -> Result<Self, GpuNativeGraphError> {
        let length = u64::try_from(bytes.len()).map_err(|_| {
            GpuNativeGraphError::Native("hash tag byte component is too long".into())
        })?;
        let mut framed = Vec::with_capacity(bytes.len().saturating_add(9));
        framed.push(0);
        framed.extend_from_slice(&length.to_be_bytes());
        framed.extend_from_slice(bytes);
        Ok(Self::Static(framed))
    }

    pub fn integer_constant(value: &BigInt) -> Result<Self, GpuNativeGraphError> {
        let (sign, magnitude) = value.to_bytes_be();
        let length = u64::try_from(magnitude.len())
            .map_err(|_| GpuNativeGraphError::Native("hash tag integer is too long".into()))?;
        let mut framed = Vec::with_capacity(magnitude.len().saturating_add(10));
        framed.push(1);
        framed.push(u8::from(sign == num_bigint::Sign::Minus));
        framed.extend_from_slice(&length.to_be_bytes());
        framed.extend_from_slice(&magnitude);
        Ok(Self::Static(framed))
    }

    pub fn decimal_constant(value: &BigInt) -> Result<Self, GpuNativeGraphError> {
        let decimal = value.to_string();
        let length = u64::try_from(decimal.len())
            .map_err(|_| GpuNativeGraphError::Native("hash tag decimal is too long".into()))?;
        let mut framed = Vec::with_capacity(decimal.len().saturating_add(9));
        framed.push(2);
        framed.extend_from_slice(&length.to_be_bytes());
        framed.extend_from_slice(decimal.as_bytes());
        Ok(Self::Static(framed))
    }

    pub fn u64_le_constant(value: u64) -> Self {
        let mut framed = Vec::with_capacity(9);
        framed.push(3);
        framed.extend_from_slice(&value.to_le_bytes());
        Self::Static(framed)
    }
}

/// Fixed CRT product, tag template and replayable scalar pointer table for
/// exact Keccak-256 HashSample. One plan may execute again after its previous
/// GPU work and I/O observers have both completed.
pub struct GpuHashSamplePlan {
    raw: *mut GpuRawHashPlanOpaque,
    context: Arc<GpuContext>,
    physical_device: i32,
    moduli: Box<[u64]>,
    operand_encodings: Box<[GpuSignedValuesEncoding]>,
    prepare_lock: Mutex<()>,
}

unsafe impl Send for GpuHashSamplePlan {}
unsafe impl Sync for GpuHashSamplePlan {}

impl GpuHashSamplePlan {
    pub fn new(
        params: &GpuDCRTPolyParams,
        stream: &GpuNativeLaunchStream,
        moduli: &[u64],
        parts: &[GpuHashTagPart],
        operand_encodings: &[GpuSignedValuesEncoding],
    ) -> Result<Self, GpuNativeGraphError> {
        // Empty `moduli` plans an integer-family sample.
        if moduli.len() > 64 ||
            !params.moduli.starts_with(moduli) ||
            !Arc::ptr_eq(&params.ctx, &stream._context)
        {
            return Err(GpuNativeGraphError::Native("invalid raw hash plan ring or context".into()));
        }
        let mut segments = Vec::with_capacity(parts.len());
        let mut static_bytes = Vec::new();
        for part in parts {
            let (kind, operand_index, offset, length) = match part {
                GpuHashTagPart::Static(bytes) => {
                    let offset = u64::try_from(static_bytes.len()).map_err(|_| {
                        GpuNativeGraphError::Native("raw hash static tag is too long".into())
                    })?;
                    let length = u64::try_from(bytes.len()).map_err(|_| {
                        GpuNativeGraphError::Native("raw hash static tag is too long".into())
                    })?;
                    static_bytes.extend_from_slice(bytes);
                    (0, 0, offset, length)
                }
                GpuHashTagPart::Integer(index) => (1, *index, 0, 0),
                GpuHashTagPart::Decimal(index) => (2, *index, 0, 0),
                GpuHashTagPart::U64Le(index) => (3, *index, 0, 0),
            };
            if kind != 0 &&
                (operand_index >= operand_encodings.len() || operand_index > u32::MAX as usize)
            {
                return Err(GpuNativeGraphError::Native(
                    "raw hash tag operand index is out of range".into(),
                ));
            }
            segments.push(GpuRawHashTagSegmentAbi {
                kind,
                operand_index: operand_index as u32,
                static_offset: offset,
                static_length: length,
            });
        }
        let mut encoding_codes = Vec::with_capacity(operand_encodings.len());
        for encoding in operand_encodings {
            if matches!(encoding, GpuSignedValuesEncoding::SignedWords(words)
                if *words == 0 || *words > i32::MAX as usize - 2)
            {
                return Err(GpuNativeGraphError::Native(
                    "invalid raw hash signed-word width".into(),
                ));
            }
            encoding_codes.push(encoding.native_code());
        }
        let mut raw = ptr::null_mut();
        if unsafe {
            gpu_raw_hash_plan_create(
                params.ctx.raw_ptr(),
                stream.physical_device,
                stream.raw_ptr(),
                moduli.as_ptr(),
                moduli.len(),
                segments.as_ptr(),
                segments.len(),
                static_bytes.as_ptr(),
                static_bytes.len(),
                encoding_codes.as_ptr(),
                encoding_codes.len(),
                &mut raw,
            )
        } != 0 ||
            raw.is_null()
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(Self {
            raw,
            context: Arc::clone(&params.ctx),
            physical_device: stream.physical_device,
            moduli: moduli.into(),
            operand_encodings: operand_encodings.into(),
            prepare_lock: Mutex::new(()),
        })
    }

    /// Refresh the current frame's owner-derived scalar addresses on the
    /// launch stream. The previous submission must be fully joined first.
    pub fn prepare_graph_launch(
        &self,
        stream: &GpuNativeLaunchStream,
        operands: &[GpuRawIntegerView],
    ) -> Result<(), GpuNativeGraphError> {
        // Any context's stream on the plan's device orders the refresh.
        if stream.physical_device != self.physical_device ||
            operands.len() != self.operand_encodings.len() ||
            operands.iter().zip(self.operand_encodings.iter()).any(|(view, encoding)| {
                view.address == 0 || view.count != 1 || view.encoding != *encoding
            })
        {
            return Err(GpuNativeGraphError::Native(
                "raw hash replay scalar owner/encoding mismatch".into(),
            ));
        }
        let addresses = operands.iter().map(|view| view.address).collect::<Vec<_>>();
        let encodings = self
            .operand_encodings
            .iter()
            .map(|encoding| encoding.native_code())
            .collect::<Vec<_>>();
        let _guard = self.prepare_lock.lock().map_err(|_| {
            GpuNativeGraphError::Native("raw hash replay plan lock is poisoned".into())
        })?;
        if unsafe {
            gpu_raw_hash_plan_prepare_graph_launch(
                self.raw,
                stream.raw_ptr(),
                addresses.as_ptr(),
                encodings.as_ptr(),
                addresses.len(),
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }

    pub fn allocation_range(&self) -> Result<(u64, usize), GpuNativeGraphError> {
        let mut address = 0;
        let mut bytes = 0;
        if unsafe { gpu_raw_hash_plan_allocation_range(self.raw, &mut address, &mut bytes) } != 0 ||
            address == 0 ||
            bytes == 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok((address, bytes))
    }

    /// Emit tag construction, owner-derived limb descriptors and full-Q
    /// Keccak rejection directly into the active explicit CUDA Graph.
    pub fn emit_raw_hash_sample(
        &self,
        stream: &GpuNativeLaunchStream,
        key_address: u64,
        destination: &GpuRawMatrixView,
        status: GpuRawControlStatusView,
        key_binding: u32,
        destination_binding_base: u32,
    ) -> Result<(), GpuNativeGraphError> {
        if !Arc::ptr_eq(&self.context, &stream._context) ||
            stream.physical_device != self.physical_device ||
            destination.physical_device != self.physical_device ||
            destination.limbs.len() != self.moduli.len() ||
            destination.limbs.iter().zip(self.moduli.iter()).enumerate().any(
                |(index, (limb, modulus))| {
                    limb.crt_limb_index as usize != index || limb.modulus != *modulus
                },
            ) ||
            key_address == 0 ||
            status.address == 0
        {
            return Err(GpuNativeGraphError::Native(
                "raw hash sample physical owner/basis mismatch".into(),
            ));
        }
        if unsafe {
            gpu_raw_hash_sample_emit(
                self.raw,
                self.context.raw_ptr(),
                stream.raw_ptr(),
                key_address as *const u8,
                &destination.abi(),
                status.address as *mut u32,
                key_binding,
                destination_binding_base,
                status.binding,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }
}

impl GpuHashSamplePlan {
    /// Emit tag construction and the `bits`-bit integers of a signed-word
    /// family (one sign word, then `words` magnitude words per element)
    /// into the active explicit CUDA Graph.
    pub fn emit_raw_hash_integers(
        &self,
        stream: &GpuNativeLaunchStream,
        key_address: u64,
        destination: &GpuRawIntegerView,
        bits: usize,
        status: GpuRawControlStatusView,
        key_binding: u32,
    ) -> Result<(), GpuNativeGraphError> {
        let GpuSignedValuesEncoding::SignedWords(words) = destination.encoding else {
            return Err(GpuNativeGraphError::Native(
                "raw hash integers need a signed-word family".into(),
            ));
        };
        // An integer family has no ring, so the graph's stream may belong to
        // another context of the same device.
        if stream.physical_device != self.physical_device ||
            !self.moduli.is_empty() ||
            key_address == 0 ||
            status.address == 0 ||
            destination.address == 0
        {
            return Err(GpuNativeGraphError::Native(
                "raw hash integer family owner/plan mismatch".into(),
            ));
        }
        if unsafe {
            gpu_raw_hash_integers_emit(
                self.raw,
                self.context.raw_ptr(),
                stream.raw_ptr(),
                key_address as *const u8,
                destination.address as *mut u64,
                destination.count as u64,
                words,
                bits,
                status.address as *mut u32,
                key_binding,
                destination.binding,
                status.binding,
            )
        } != 0
        {
            return Err(GpuNativeGraphError::Native(last_error_string()));
        }
        Ok(())
    }
}

impl Drop for GpuHashSamplePlan {
    fn drop(&mut self) {
        unsafe { gpu_raw_hash_plan_destroy(self.raw) };
    }
}
