//! Prepared device scalar payloads and their GPU polynomial consumers.
use super::*;
use crate::poly::dcrt::gpu::{
    GpuPreparedScalarBufferOpaque, GpuPreparedScalarMatrixSelectOpaque, GpuPreparedScalarOpOpaque,
    GpuPreparedScalarPackOpaque, GpuPreparedScalarRef, GpuPreparedThresholdOpaque,
    GpuReleaseCompletion, gpu_matrix_defer_scalar_buffer_pinned_free,
    gpu_matrix_destroy_scalar_buffer, gpu_matrix_destroy_scalar_matrix_select,
    gpu_matrix_destroy_scalar_op, gpu_matrix_destroy_scalar_pack, gpu_matrix_destroy_threshold,
    gpu_matrix_finalize_scalar_buffer, gpu_matrix_finalize_scalar_op,
    gpu_matrix_prepare_scalar_buffer, gpu_matrix_prepare_scalar_matrix_select,
    gpu_matrix_prepare_scalar_op, gpu_matrix_prepare_scalar_pack, gpu_matrix_prepare_threshold,
    gpu_matrix_read_scalar_buffer, gpu_matrix_scalar_matrix_select_workspace_bytes,
    gpu_matrix_scalar_op_workspace_bytes, gpu_matrix_scalar_pack_workspace_bytes,
    gpu_matrix_submit_scalar_matrix_select, gpu_matrix_submit_scalar_op,
    gpu_matrix_submit_scalar_pack, gpu_matrix_submit_threshold,
    gpu_matrix_threshold_workspace_bytes, gpu_matrix_upload_scalar_buffer,
    gpu_matrix_validate_scalar_buffer, gpu_matrix_wait_scalar_buffer,
};
use std::sync::atomic::{AtomicBool, AtomicUsize};

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
pub enum GpuPreparedScalarOpcode {
    Add = 0,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    Equal,
    Less,
    LessEqual,
    BitExtract,
    IntToReal,
    BoolToInt,
    RealAdd,
    RealSubtract,
    RealMultiply,
    RealDivide,
    RealSqrt,
    Copy,
    Select,
}

pub struct GpuPreparedScalarOp {
    pub(super) raw: NonNull<GpuPreparedScalarOpOpaque>,
    layout: super::super::PreparedPlanLayout,
    left: Arc<GpuPreparedScalarBuffer>,
    right: Option<Arc<GpuPreparedScalarBuffer>>,
    candidates: Box<[Arc<GpuPreparedScalarBuffer>]>,
    pub(super) output: Arc<GpuPreparedScalarBuffer>,
    finalized: AtomicBool,
}
unsafe impl Send for GpuPreparedScalarOp {}
unsafe impl Sync for GpuPreparedScalarOp {}
impl GpuPreparedScalarOp {
    pub fn workspace_bytes(left: usize, right: usize, output: usize, candidates: usize) -> usize {
        unsafe { gpu_matrix_scalar_op_workspace_bytes(left, right, output, candidates) }
    }
    pub fn bind(
        opcode: GpuPreparedScalarOpcode,
        left: (Arc<GpuPreparedScalarBuffer>, usize),
        right: Option<(Arc<GpuPreparedScalarBuffer>, usize)>,
        output: (Arc<GpuPreparedScalarBuffer>, usize),
        bit: usize,
        candidates: &[(Arc<GpuPreparedScalarBuffer>, usize)],
    ) -> Result<Arc<Self>, String> {
        let layout = super::super::PreparedPlanLayout::scalar_op(
            left.0.anchor().params(),
            left.0.words(),
            right.as_ref().map_or(0, |v| v.0.words()),
            output.0.words(),
            candidates.len(),
        )?;
        Self::bind_with_layout(opcode, left, right, output, bit, candidates, layout)
    }

    /// Bind using the descriptor resolved during warmup.  Production replay
    /// passes this saved layout; the convenience `bind` above is retained for
    /// direct primitive users and tests that construct a plan ad hoc.
    pub fn bind_with_layout(
        opcode: GpuPreparedScalarOpcode,
        left: (Arc<GpuPreparedScalarBuffer>, usize),
        right: Option<(Arc<GpuPreparedScalarBuffer>, usize)>,
        output: (Arc<GpuPreparedScalarBuffer>, usize),
        bit: usize,
        candidates: &[(Arc<GpuPreparedScalarBuffer>, usize)],
        layout: super::super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        let mut raw = std::ptr::null_mut();
        let operand = |value: &(Arc<GpuPreparedScalarBuffer>, usize)| GpuPreparedScalarRef {
            owner: value.0.raw.as_ptr(),
            index: value.1,
        };
        let candidate_refs = candidates.iter().map(operand).collect::<Vec<_>>();
        if unsafe {
            gpu_matrix_prepare_scalar_op(
                opcode as i32,
                operand(&left),
                right
                    .as_ref()
                    .map(operand)
                    .unwrap_or(GpuPreparedScalarRef { owner: std::ptr::null(), index: 0 }),
                operand(&output),
                bit,
                candidate_refs.as_ptr(),
                candidate_refs.len(),
                layout.native_ptr(),
                &mut raw,
            )
        } != 0
        {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or("missing scalar operation")?,
            layout,
            left: left.0,
            right: right.map(|value| value.0),
            candidates: candidates.iter().map(|(owner, _)| Arc::clone(owner)).collect(),
            output: output.0,
            finalized: AtomicBool::new(false),
        }))
    }
    pub fn submit(&self) -> Result<(), String> {
        if unsafe { gpu_matrix_submit_scalar_op(self.raw.as_ptr()) } != 0 {
            return Err(last_error_string());
        }
        Ok(())
    }
    pub fn finalize(&self) -> Result<GpuReleaseCompletion, String> {
        if self.finalized.swap(true, Ordering::AcqRel) {
            return Err("scalar operation already finalized".into());
        }
        let mut events = std::ptr::null_mut();
        let status = unsafe { gpu_matrix_finalize_scalar_op(self.raw.as_ptr(), &mut events) };
        if status != 0 {
            return Err(last_error_string());
        }
        let events =
            NonNull::new(events).ok_or("scalar operation finalization returned no completion")?;
        Ok(GpuReleaseCompletion::from_event_set(self.output.anchor().params().clone(), events))
    }
    pub fn output(&self) -> &Arc<GpuPreparedScalarBuffer> {
        &self.output
    }
    pub fn inputs(&self) -> impl Iterator<Item = &Arc<GpuPreparedScalarBuffer>> {
        std::iter::once(&self.left).chain(self.right.iter()).chain(self.candidates.iter())
    }
    pub fn allocation_layout(&self) -> &[super::super::PreparedAllocationLayout] {
        self.layout.allocations()
    }
}
impl Drop for GpuPreparedScalarOp {
    fn drop(&mut self) {
        if !self.finalized.swap(true, Ordering::AcqRel) {
            let mut events = std::ptr::null_mut();
            if unsafe { gpu_matrix_finalize_scalar_op(self.raw.as_ptr(), &mut events) } == 0 {
                if let Some(events) = NonNull::new(events) {
                    drop(GpuReleaseCompletion::from_event_set(
                        self.output.anchor().params().clone(),
                        events,
                    ));
                }
            }
        }
        unsafe { gpu_matrix_destroy_scalar_op(self.raw.as_ptr()) }
    }
}

pub struct GpuPreparedScalarMatrixSelect {
    pub(super) raw: NonNull<GpuPreparedScalarMatrixSelectOpaque>,
    layout: super::super::PreparedPlanLayout,
    pub(super) output: Arc<GpuDCRTPolyMatrix>,
    sources: Box<[Arc<GpuDCRTPolyMatrix>]>,
    selector: Arc<GpuPreparedScalarBuffer>,
}
unsafe impl Send for GpuPreparedScalarMatrixSelect {}
unsafe impl Sync for GpuPreparedScalarMatrixSelect {}
impl GpuPreparedScalarMatrixSelect {
    pub fn workspace_bytes(count: usize) -> usize {
        unsafe { gpu_matrix_scalar_matrix_select_workspace_bytes(count) }
    }
    pub fn bind(
        output: Arc<GpuDCRTPolyMatrix>,
        selector: (Arc<GpuPreparedScalarBuffer>, usize),
        sources: &[(Arc<GpuDCRTPolyMatrix>, GpuPreparedView)],
    ) -> Result<Arc<Self>, String> {
        if sources.is_empty() {
            return Err("scalar matrix selection requires candidates".into());
        }
        let rows = sources[0].1.output.rows.end - sources[0].1.output.rows.start;
        let columns = sources[0].1.output.columns.end - sources[0].1.output.columns.start;
        let layout = super::super::PreparedPlanLayout::scalar_matrix_select(
            output.params(),
            rows,
            columns,
            output.level(),
            sources.len(),
        )?;
        Self::bind_with_layout(output, selector, sources, layout)
    }

    pub fn bind_with_layout(
        output: Arc<GpuDCRTPolyMatrix>,
        selector: (Arc<GpuPreparedScalarBuffer>, usize),
        sources: &[(Arc<GpuDCRTPolyMatrix>, GpuPreparedView)],
        layout: super::super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        let matrices =
            sources.iter().map(|(source, _)| source.raw.cast_const()).collect::<Vec<_>>();
        let range = |value: &GpuPreparedRange| GpuMatrixRange {
            row_start: value.rows.start,
            row_end: value.rows.end,
            column_start: value.columns.start,
            column_end: value.columns.end,
        };
        let views = sources
            .iter()
            .map(|(_, view)| GpuMatrixBatchView {
                left: range(&view.left),
                right: range(&view.right),
                output: range(&view.output),
            })
            .collect::<Vec<_>>();
        if views.is_empty() {
            return Err("scalar matrix selection requires candidates".into());
        }
        let mut raw = std::ptr::null_mut();
        if unsafe {
            gpu_matrix_prepare_scalar_matrix_select(
                output.raw,
                GpuPreparedScalarRef { owner: selector.0.raw.as_ptr(), index: selector.1 },
                matrices.as_ptr(),
                views.as_ptr(),
                sources.len(),
                layout.native_ptr(),
                &mut raw,
            )
        } != 0
        {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or("missing scalar matrix selection")?,
            layout,
            output,
            sources: sources.iter().map(|(source, _)| Arc::clone(source)).collect(),
            selector: selector.0,
        }))
    }
    pub fn submit(&self) -> Result<(), String> {
        if unsafe { gpu_matrix_submit_scalar_matrix_select(self.raw.as_ptr()) } != 0 {
            return Err(last_error_string());
        }
        Ok(())
    }
    pub fn check_selector(&self) -> Result<(), String> {
        self.selector.with_words(|_, _| ())
    }
    pub fn sources(&self) -> &[Arc<GpuDCRTPolyMatrix>] {
        &self.sources
    }
    pub fn allocation_layout(&self) -> &[super::super::PreparedAllocationLayout] {
        self.layout.allocations()
    }
}
impl Drop for GpuPreparedScalarMatrixSelect {
    fn drop(&mut self) {
        unsafe { gpu_matrix_destroy_scalar_matrix_select(self.raw.as_ptr()) }
    }
}

/// Fixed-capacity scalar backing independent of command variants writing it.
/// Integer width is established during warmup and cannot grow during replay.
pub struct GpuPreparedScalarBuffer {
    pub(super) raw: NonNull<GpuPreparedScalarBufferOpaque>,
    layout: super::super::PreparedPlanLayout,
    anchor: Arc<GpuDCRTPolyMatrix>,
    host: Mutex<Option<PinnedHostBuffer<u64>>>,
    count: usize,
    words: AtomicUsize,
    upload_pending: AtomicBool,
}
unsafe impl Send for GpuPreparedScalarBuffer {}
unsafe impl Sync for GpuPreparedScalarBuffer {}
impl GpuPreparedScalarBuffer {
    /// Claim order: pinned host, batch workspace, two completion events.
    pub fn bind(
        anchor: Arc<GpuDCRTPolyMatrix>,
        count: usize,
        words: usize,
    ) -> Result<Arc<Self>, String> {
        let layout =
            super::super::PreparedPlanLayout::scalar_buffer(anchor.params(), count, words)?;
        Self::bind_with_layout(anchor, count, words, layout)
    }

    pub fn bind_with_layout(
        anchor: Arc<GpuDCRTPolyMatrix>,
        count: usize,
        words: usize,
        layout: super::super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        let capacity = words
            .checked_add(1)
            .and_then(|words| count.checked_mul(words))
            .ok_or("scalar capacity overflow")?;
        if unsafe {
            gpu_matrix_validate_scalar_buffer(anchor.raw, count, words, layout.native_ptr())
        } != 0
        {
            return Err(last_error_string());
        }
        let mut host = PinnedHostBuffer::zeroed(anchor.params(), capacity);
        let mut raw = std::ptr::null_mut();
        if unsafe {
            gpu_matrix_prepare_scalar_buffer(
                anchor.raw,
                count,
                words,
                host.as_mut_slice().as_mut_ptr(),
                layout.native_ptr(),
                &mut raw,
            )
        } != 0
        {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or("missing scalar backing")?,
            layout,
            anchor,
            host: Mutex::new(Some(host)),
            count,
            words: AtomicUsize::new(words),
            upload_pending: AtomicBool::new(false),
        }))
    }
    pub fn words(&self) -> usize {
        self.words.load(Ordering::Acquire)
    }
    pub fn count(&self) -> usize {
        self.count
    }
    pub fn anchor(&self) -> &Arc<GpuDCRTPolyMatrix> {
        &self.anchor
    }

    pub fn allocation_layout(&self) -> &super::super::PreparedPlanLayout {
        &self.layout
    }

    /// Root staging only; the caller exclusively owns the invocation instance.
    pub fn upload(
        &self,
        fill: impl FnOnce(&mut [u64], usize) -> Result<(), String>,
    ) -> Result<(), String> {
        if self.upload_pending.swap(true, Ordering::AcqRel) {
            return Err("prepared scalar upload still in flight".into());
        }
        let mut host = self.host.lock().map_err(|_| "scalar staging poisoned")?;
        let words = host.as_mut().ok_or("scalar staging quarantined")?.as_mut_slice();
        let capacity = self.words();
        words[self.count * capacity..].fill(0);
        if let Err(error) = fill(&mut words[..self.count * capacity], capacity) {
            self.upload_pending.store(false, Ordering::Release);
            return Err(error);
        }
        if unsafe { gpu_matrix_upload_scalar_buffer(self.raw.as_ptr()) } != 0 {
            // A failed record cannot prove whether the preceding H2D copy still
            // references this host allocation. Quarantine it with the execution.
            std::mem::forget(host.take());
            return Err(last_error_string());
        }
        Ok(())
    }
    /// Explicit completion/output boundary; never used by a device consumer.
    pub fn wait(&self) -> Result<(), String> {
        if unsafe { gpu_matrix_wait_scalar_buffer(self.raw.as_ptr()) } != 0 {
            return Err(last_error_string());
        }
        self.upload_pending.store(false, Ordering::Release);
        Ok(())
    }
    pub fn with_words<T>(&self, read: impl FnOnce(&[u64], usize) -> T) -> Result<T, String> {
        let mut host = self.host.lock().map_err(|_| "scalar readback poisoned")?;
        if host.is_none() {
            return Err("scalar readback quarantined".into());
        }
        if unsafe { gpu_matrix_read_scalar_buffer(self.raw.as_ptr()) } != 0 {
            std::mem::forget(host.take());
            return Err(last_error_string());
        }
        let capacity = self.words();
        let host = host.as_ref().expect("checked pinned owner").as_slice();
        self.upload_pending.store(false, Ordering::Release);
        let status =
            host[self.count * capacity..].iter().fold(0, |combined, status| combined | status);
        if status & 1 != 0 {
            return Err("prepared scalar arithmetic failed (division by zero)".into());
        }
        if status & 2 != 0 {
            return Err("prepared scalar selection index is outside its fixed candidates".into());
        }
        Ok(read(&host[..self.count * capacity], capacity))
    }
}
impl Drop for GpuPreparedScalarBuffer {
    fn drop(&mut self) {
        let mut events = std::ptr::null_mut();
        let status = unsafe { gpu_matrix_finalize_scalar_buffer(self.raw.as_ptr(), &mut events) };
        if status == 0 {
            let _ = NonNull::new(events).map(|events| {
                GpuReleaseCompletion::from_event_set(self.anchor.params().clone(), events)
            });
        }
        let mut pending = Vec::<*mut std::ffi::c_void>::new();
        if self.upload_pending.load(Ordering::Acquire) {
            let host = match self.host.get_mut() {
                Ok(host) => host,
                Err(poisoned) => poisoned.into_inner(),
            };
            if let Some(host) = host.take() {
                pending.push(host.into_raw().cast());
            }
        }
        if !pending.is_empty() {
            let _ = unsafe {
                gpu_matrix_defer_scalar_buffer_pinned_free(
                    self.raw.as_ptr(),
                    pending.as_ptr(),
                    pending.len(),
                )
            };
        }
        unsafe { gpu_matrix_destroy_scalar_buffer(self.raw.as_ptr()) }
    }
}

pub struct GpuPreparedThreshold {
    pub(super) raw: NonNull<GpuPreparedThresholdOpaque>,
    layout: super::super::PreparedPlanLayout,
    pub(super) source: Arc<GpuDCRTPolyMatrix>,
    output: Arc<GpuPreparedScalarBuffer>,
}
unsafe impl Send for GpuPreparedThreshold {}
unsafe impl Sync for GpuPreparedThreshold {}

impl GpuPreparedThreshold {
    /// Demand order: pinned payload, batch workspace, two completion events.
    pub fn layout(
        source: &GpuDCRTPolyMatrix,
        plaintext: &BigUint,
        count: usize,
        output_bool: bool,
    ) -> Result<(usize, usize), String> {
        let words = plaintext.iter_u64_digits().len();
        let mut workspace = 0;
        if unsafe { gpu_matrix_threshold_workspace_bytes(source.raw, count, words, &mut workspace) } !=
            0
        {
            return Err(last_error_string());
        }
        Ok((count * if output_bool { 2 } else { words + 2 } * 8, workspace))
    }

    pub fn bind(
        source: Arc<GpuDCRTPolyMatrix>,
        plaintext: &BigUint,
        count: usize,
        output_bool: bool,
        output: Option<Arc<GpuPreparedScalarBuffer>>,
    ) -> Result<Arc<Self>, String> {
        let words = plaintext.iter_u64_digits().len();
        let output = match output {
            Some(output) => output,
            None => GpuPreparedScalarBuffer::bind(
                Arc::clone(&source),
                count,
                if output_bool { 1 } else { words + 1 },
            )?,
        };
        let layout =
            super::super::PreparedPlanLayout::threshold(source.params(), count, words.max(1))?;
        Self::bind_with_layout(source, plaintext, count, output_bool, output, layout)
    }

    pub fn bind_with_layout(
        source: Arc<GpuDCRTPolyMatrix>,
        plaintext: &BigUint,
        count: usize,
        output_bool: bool,
        output: Arc<GpuPreparedScalarBuffer>,
        layout: super::super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        let plaintext = plaintext.to_u64_digits();
        let mut raw = std::ptr::null_mut();
        if unsafe {
            gpu_matrix_prepare_threshold(
                source.raw,
                count,
                plaintext.as_ptr(),
                plaintext.len(),
                output_bool,
                output.raw.as_ptr(),
                layout.native_ptr(),
                &mut raw,
            )
        } != 0
        {
            return Err(last_error_string());
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or("missing threshold plan")?,
            layout,
            source,
            output,
        }))
    }

    pub fn submit(&self) -> Result<(), String> {
        if unsafe { gpu_matrix_submit_threshold(self.raw.as_ptr()) } == 0 {
            Ok(())
        } else {
            Err(last_error_string())
        }
    }

    /// Explicit output boundary only. Internal consumers read the device payload.
    pub fn with_words<T>(&self, read: impl FnOnce(&[u64], usize) -> T) -> Result<T, String> {
        self.output.with_words(read)
    }
    pub fn output(&self) -> &Arc<GpuPreparedScalarBuffer> {
        &self.output
    }
    pub fn allocation_layout(&self) -> &[super::super::PreparedAllocationLayout] {
        self.layout.allocations()
    }
}
impl Drop for GpuPreparedThreshold {
    fn drop(&mut self) {
        unsafe { gpu_matrix_destroy_threshold(self.raw.as_ptr()) }
    }
}

pub struct GpuPreparedScalarPack {
    pub(super) raw: NonNull<GpuPreparedScalarPackOpaque>,
    layout: super::super::PreparedPlanLayout,
    pub(super) output: Arc<GpuDCRTPolyMatrix>,
    sources: Box<[Arc<GpuPreparedScalarBuffer>]>,
}
unsafe impl Send for GpuPreparedScalarPack {}
unsafe impl Sync for GpuPreparedScalarPack {}
impl GpuPreparedScalarPack {
    pub fn workspace_bytes(count: usize) -> usize {
        unsafe { gpu_matrix_scalar_pack_workspace_bytes(count) }
    }

    /// A zero coefficient_bits packs integers; otherwise pack nonzero predicates.
    pub fn bind(
        output: Arc<GpuDCRTPolyMatrix>,
        values: &[(Arc<GpuPreparedScalarBuffer>, usize)],
        coefficient_bits: usize,
    ) -> Result<Arc<Self>, String> {
        let owner = output.prepared_owner_layout()?;
        let layout = super::super::PreparedPlanLayout::scalar_pack_with_owner(
            output.params(),
            values.len(),
            coefficient_bits,
            output.level(),
            crate::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL,
            &owner,
        )?;
        Self::bind_with_layout(output, values, coefficient_bits, layout)
    }

    pub fn bind_with_layout(
        output: Arc<GpuDCRTPolyMatrix>,
        values: &[(Arc<GpuPreparedScalarBuffer>, usize)],
        coefficient_bits: usize,
        layout: super::super::PreparedPlanLayout,
    ) -> Result<Arc<Self>, String> {
        let values_native = values
            .iter()
            .map(|(owner, index)| GpuPreparedScalarRef { owner: owner.raw.as_ptr(), index: *index })
            .collect::<Vec<_>>();
        let mut raw = std::ptr::null_mut();
        if unsafe {
            gpu_matrix_prepare_scalar_pack(
                output.raw,
                values_native.as_ptr(),
                values.len(),
                coefficient_bits,
                layout.native_ptr(),
                &mut raw,
            )
        } != 0
        {
            return Err(last_error_string());
        }
        let mut sources = Vec::new();
        for (owner, _) in values {
            if !sources.iter().any(|candidate| Arc::ptr_eq(candidate, owner)) {
                sources.push(Arc::clone(owner));
            }
        }
        Ok(Arc::new(Self {
            raw: NonNull::new(raw).ok_or("missing scalar pack plan")?,
            layout,
            output,
            sources: sources.into_boxed_slice(),
        }))
    }
    pub fn submit(&self) -> Result<(), String> {
        if unsafe { gpu_matrix_submit_scalar_pack(self.raw.as_ptr()) } == 0 {
            Ok(())
        } else {
            Err(last_error_string())
        }
    }
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }
    pub fn allocation_layout(&self) -> &[super::super::PreparedAllocationLayout] {
        self.layout.allocations()
    }
    /// Called only when materializing/retiring the enclosing graph output.
    pub fn check_sources(&self) -> Result<(), String> {
        for source in &self.sources {
            source.with_words(|_, _| ())?;
        }
        Ok(())
    }
}
impl Drop for GpuPreparedScalarPack {
    fn drop(&mut self) {
        unsafe { gpu_matrix_destroy_scalar_pack(self.raw.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::{
        Poly,
        dcrt::{
            gpu::{GpuDCRTPolyParams, GpuRngSeed},
            params::DCRTPolyParams,
        },
    };
    use rand::Rng;

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_scalar_signed_arithmetic_matches_bigint() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse().unwrap())
            .unwrap_or(32);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, None);
        let anchor = Arc::new(GpuDCRTPolyMatrix::new_zero(&params, 1, 1));
        let left = GpuPreparedScalarBuffer::bind(Arc::clone(&anchor), 1, 4).unwrap();
        let right = GpuPreparedScalarBuffer::bind(Arc::clone(&anchor), 1, 3).unwrap();
        let a: BigInt = -(BigInt::from(rand::rng().random::<u64>()) << 97usize) - 17;
        let b: BigInt = (BigInt::from(rand::rng().random::<u64>()) << 41usize) + 3;
        let upload = |owner: &GpuPreparedScalarBuffer, value: &BigInt| {
            let bytes = value.to_signed_bytes_le();
            owner
                .upload(|words, _| {
                    words.fill(if value.sign() == Sign::Minus { u64::MAX } else { 0 });
                    for (index, byte) in bytes.iter().enumerate() {
                        let shift = (index % 8) * 8;
                        words[index / 8] =
                            (words[index / 8] & !(255u64 << shift)) | (u64::from(*byte) << shift);
                    }
                    Ok(())
                })
                .unwrap();
        };
        for (a, b) in [
            (a.clone(), b.clone()),
            (-&a, b.clone()),
            (a.clone(), -&b),
            (-&a, -&b),
            (BigInt::from(0), b.clone()),
        ] {
            upload(&left, &a);
            upload(&right, &b);
            let cases = [
                (GpuPreparedScalarOpcode::Add, &a + &b),
                (GpuPreparedScalarOpcode::Subtract, &a - &b),
                (GpuPreparedScalarOpcode::Multiply, &a * &b),
                (GpuPreparedScalarOpcode::Divide, &a / &b),
                (GpuPreparedScalarOpcode::Remainder, &a % &b),
                (GpuPreparedScalarOpcode::Equal, BigInt::from(u8::from(a == b))),
                (GpuPreparedScalarOpcode::Less, BigInt::from(u8::from(a < b))),
                (GpuPreparedScalarOpcode::LessEqual, BigInt::from(u8::from(a <= b))),
                (GpuPreparedScalarOpcode::BitExtract, (&a >> 63usize) & BigInt::from(1)),
            ];
            for (opcode, expected) in cases {
                let output = GpuPreparedScalarBuffer::bind(Arc::clone(&anchor), 1, 8).unwrap();
                let operation = GpuPreparedScalarOp::bind(
                    opcode,
                    (Arc::clone(&left), 0),
                    Some((Arc::clone(&right), 0)),
                    (Arc::clone(&output), 0),
                    63,
                    &[],
                )
                .unwrap();
                operation.submit().unwrap();
                output
                    .with_words(|words, _| {
                        let bytes =
                            words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
                        assert_eq!(BigInt::from_signed_bytes_le(&bytes), expected, "{opcode:?}");
                    })
                    .unwrap();
            }
            left.wait().unwrap();
            right.wait().unwrap();
        }
        upload(&left, &a);
        let output = GpuPreparedScalarBuffer::bind(Arc::clone(&anchor), 1, 1).unwrap();
        let operation = GpuPreparedScalarOp::bind(
            GpuPreparedScalarOpcode::IntToReal,
            (left, 0),
            None,
            (Arc::clone(&output), 0),
            0,
            &[],
        )
        .unwrap();
        operation.submit().unwrap();
        output
            .with_words(|words, _| assert_eq!(f64::from_bits(words[0]), a.to_f64().unwrap()))
            .unwrap();
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_scalar_real_and_division_error() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse().unwrap())
            .unwrap_or(32);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, None);
        let anchor = Arc::new(GpuDCRTPolyMatrix::new_zero(&params, 1, 1));
        let left = GpuPreparedScalarBuffer::bind(Arc::clone(&anchor), 1, 1).unwrap();
        let right = GpuPreparedScalarBuffer::bind(Arc::clone(&anchor), 1, 1).unwrap();
        let output = GpuPreparedScalarBuffer::bind(Arc::clone(&anchor), 1, 1).unwrap();
        let a = rand::rng().random_range(0.01f64..100.0);
        let b = -rand::rng().random_range(0.01f64..100.0);
        left.upload(|words, _| {
            words[0] = a.to_bits();
            Ok(())
        })
        .unwrap();
        right
            .upload(|words, _| {
                words[0] = b.to_bits();
                Ok(())
            })
            .unwrap();
        for (opcode, expected) in [
            (GpuPreparedScalarOpcode::RealAdd, a + b),
            (GpuPreparedScalarOpcode::RealSubtract, a - b),
            (GpuPreparedScalarOpcode::RealMultiply, a * b),
            (GpuPreparedScalarOpcode::RealDivide, a / b),
            (GpuPreparedScalarOpcode::RealSqrt, a.sqrt()),
        ] {
            let plan = GpuPreparedScalarOp::bind(
                opcode,
                (Arc::clone(&left), 0),
                Some((Arc::clone(&right), 0)),
                (Arc::clone(&output), 0),
                0,
                &[],
            )
            .unwrap();
            plan.submit().unwrap();
            output.with_words(|words, _| assert_eq!(f64::from_bits(words[0]), expected)).unwrap();
        }
        right.wait().unwrap();
        right
            .upload(|words, _| {
                words.fill(0);
                Ok(())
            })
            .unwrap();
        let plan = GpuPreparedScalarOp::bind(
            GpuPreparedScalarOpcode::Divide,
            (left, 0),
            Some((right, 0)),
            (Arc::clone(&output), 0),
            0,
            &[],
        )
        .unwrap();
        plan.submit().unwrap();
        assert!(output.with_words(|_, _| ()).unwrap_err().contains("division by zero"));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_wide_threshold_device_pack_roundtrip() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse().unwrap())
            .unwrap_or(32);
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, None);
        let source = Arc::new(
            GpuDCRTPolyMatrix::sample_distribution(
                &params,
                1,
                1,
                GpuMatrixSampleDist::Uniform,
                0.0,
                u64::MAX,
                GpuRngSeed::from_bytes(rand::rng().random()),
            )
            .into_coeff_domain(),
        );
        let modulus = params.moduli().iter().fold(BigUint::from(1u8), |q, p| q * p);
        assert!(modulus.bits() > 64);
        let expected = source.to_cpu_matrix();
        let shared = GpuPreparedScalarBuffer::bind(
            Arc::clone(&source),
            n as usize,
            params.modulus().bits().div_ceil(64) as usize + 2,
        )
        .unwrap();
        for factor in [1u32, 2] {
            let plaintext = &modulus * factor;
            let threshold = GpuPreparedThreshold::bind(
                Arc::clone(&source),
                &plaintext,
                n as usize,
                false,
                Some(Arc::clone(&shared)),
            )
            .unwrap();
            let output = Arc::new(GpuDCRTPolyMatrix::new_empty_with_state(
                &params,
                1,
                1,
                source.level(),
                true,
                None,
            ));
            let values = (0..n as usize)
                .map(|index| (Arc::clone(threshold.output()), index))
                .collect::<Vec<_>>();
            let pack = GpuPreparedScalarPack::bind(Arc::clone(&output), &values, 0).unwrap();
            for _ in 0..3 {
                threshold.submit().unwrap();
                pack.submit().unwrap();
                let expected = if factor == 1 { expected.clone() } else { &expected + &expected };
                assert_eq!(output.to_cpu_matrix(), expected);
                threshold
                    .with_words(|words, width| {
                        let actual = words
                            .chunks_exact(width)
                            .map(|digits| {
                                BigUint::from_slice(
                                    &digits
                                        .iter()
                                        .flat_map(|word| [*word as u32, (*word >> 32) as u32])
                                        .collect::<Vec<_>>(),
                                )
                            })
                            .collect::<Vec<_>>();
                        let coefficients = source.to_cpu_matrix().entry(0, 0).coeffs_biguints();
                        assert_eq!(
                            actual,
                            coefficients
                                .into_iter()
                                .map(|value| value * factor)
                                .collect::<Vec<_>>()
                        );
                    })
                    .unwrap();
            }
        }
    }
}
