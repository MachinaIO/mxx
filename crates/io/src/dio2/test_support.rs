//! The toy parameters and function of the unit tests.

use super::{Dio2Parameters, StreamingFunction};
use bigdecimal::BigDecimal;
use mxx_backends::openfhe_guard::gen_modulus_and_warmup;
use mxx_exponent_lut::{ExpLayout, refresh::PrfSettings};

/// Ring dimension `8`; five primes of 50 bits with gadget base `2^12` and six levels of `Q_0`
/// (decoding level 4); the PRF on `Q = 16`, `P = 8` with one LWR coordinate and weight one; one
/// smudge digit; noise-free preprocessing (`MXX_IO_DIO2_TEST_{CRT_BITS,PRIMES,BASE_BITS,LEVELS}`
/// override the ring). The
/// function has one state bit, one evaluator bit and one obfuscator bit, on the state layout
/// `(1, 2)` (`mono(e) = X^(2 e)`, `e < 4`), and two table steps.
pub(crate) fn toy() -> (Dio2Parameters, StreamingFunction) {
    let read = |name: &str, default: usize| {
        std::env::var(format!("MXX_IO_DIO2_TEST_{name}"))
            .map_or(default, |value| value.parse().unwrap())
    };
    let crt_bits = read("CRT_BITS", 50);
    let primes =
        gen_modulus_and_warmup(8, read("PRIMES", 5), crt_bits, None).expect("generated CRT basis");
    let parameters = Dio2Parameters {
        ring_dimension: 8,
        crt_bits,
        base_bits: read("BASE_BITS", 12) as u32,
        primes,
        levels: read("LEVELS", 6),
        prf: PrfSettings {
            lwr_dimension: 1,
            support_weight: 1,
            input_modulus: 16,
            output_modulus: 8,
            bucket_width: 0,
        },
        smudge_digits: 1,
        trapdoor_sigma: 4,
        encoding_error_sigma: BigDecimal::from(0),
    };
    let function =
        StreamingFunction::additive(1, 1, 1, ExpLayout { stride_log2: 1, width_log2: 2 });
    let steps = [vec![1, 0, 1, 0], vec![0, 1, 1, 0]];
    let function =
        steps.into_iter().fold(function.clone(), |acc, table| acc.with_step_row(table).unwrap());
    (parameters, function)
}

/// An execution that validates graphs without executing them, for graph-only tests: a producing
/// graph returns the production its execution would have, an evaluated graph the integer `0`
/// for each output. Downloads read from the wrapped CPU execution.
pub(crate) struct GraphOnly(pub(crate) mxx_backends::authority::CpuExecution);

impl mxx_exponent_lut::api::StageExecution<mxx_backends::MemoryArtifactStore> for GraphOnly {
    fn produce(
        &mut self,
        validated: mxx_ir_core::ValidatedGraph,
        _: std::collections::BTreeMap<String, mxx_backends::RuntimeValue>,
        _: &mut mxx_backends::MemoryArtifactStore,
        execution_nonce: [u8; 32],
    ) -> Result<mxx_ir_core::artifact::ProductionId, mxx_exponent_lut::api::ApiError> {
        let specification =
            mxx_ir_core::encoding::spec_hash(&validated.source, &validated.bindings)
                .expect("stage specification hash");
        Ok(mxx_ir_core::artifact::production_id(specification, execution_nonce))
    }

    fn evaluate(
        &mut self,
        validated: mxx_ir_core::ValidatedGraph,
        _: std::collections::BTreeMap<String, mxx_backends::RuntimeValue>,
        _: &mut mxx_backends::MemoryArtifactStore,
        _: [u8; 32],
    ) -> Result<mxx_exponent_lut::api::StageOutputs, mxx_exponent_lut::api::ApiError> {
        Ok(mxx_exponent_lut::api::StageOutputs {
            outputs: validated
                .source
                .outputs()
                .keys()
                .map(|name| (name.clone(), mxx_backends::RuntimeValue::Int(0.into())))
                .collect(),
        })
    }

    fn download_matrix(
        &self,
        value: &mxx_backends::RuntimeValue,
    ) -> Result<mxx_backends::matrix::dcrt_poly::DCRTPolyMatrix, mxx_exponent_lut::api::ApiError>
    {
        mxx_exponent_lut::api::StageExecution::<mxx_backends::MemoryArtifactStore>::download_matrix(
            &self.0, value,
        )
    }

    fn download_integers(
        &self,
        value: &mxx_backends::RuntimeValue,
    ) -> Result<Vec<num_bigint::BigInt>, mxx_exponent_lut::api::ApiError> {
        mxx_exponent_lut::api::StageExecution::<mxx_backends::MemoryArtifactStore>::download_integers(
            &self.0, value,
        )
    }
}
