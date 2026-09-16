//! Compile coverage for the deliberately small crate-root facade.

use mxx_exponent_lut::{
    CompiledNandCircuit, EncodedOutput, EncodedOutputs, EncodingSecrets, ExpLayout,
    ExpectedEncoding, ExponentLut, ExponentLutProgram, FixedRhs, NandCircuit, SparseLwrPrfProgram,
    SparseLwrPrfSelectorKey,
};

#[allow(dead_code)]
fn sparse_lwr_public_signatures<B: mxx_runtime::Backend>(
    engine: &ExponentLut<B>,
    program: &SparseLwrPrfProgram,
    selector: &SparseLwrPrfSelectorKey,
    outputs: &EncodedOutputs,
) {
    let _: std::result::Result<FixedRhs, _> = program.fixed_rhs(&[], selector);
    let _: EncodingSecrets = program.sample_secrets(engine);
    let _: std::result::Result<&EncodedOutput, _> = program.output(outputs);
}

#[allow(dead_code)]
fn exponent_assert_public_signature<B: mxx_runtime::Backend>(
    engine: &ExponentLut<B>,
    output: &EncodedOutput,
    secrets: &EncodingSecrets,
) -> std::result::Result<(), mxx_exponent_lut::ExecutionError> {
    engine.assert_encoding(output, ExpectedEncoding::Exponent(0.into()), secrets)
}

#[test]
fn public_nand_and_lwr_facade_types_compile_without_low_level_bindings() {
    let circuit = NandCircuit::new(2, vec![[0, 1]], 2).unwrap();
    let compiled: CompiledNandCircuit =
        circuit.compile(ExpLayout { stride_log2: 0, width_log2: 4 }, 16, 4).unwrap();
    let _: &ExponentLutProgram = compiled.program();
    let _: std::result::Result<FixedRhs, _> = compiled.fixed_rhs(&[false, true]);
    let _: for<'a> fn(
        &'a CompiledNandCircuit,
        &'a EncodedOutputs,
    ) -> std::result::Result<&'a EncodedOutput, _> = CompiledNandCircuit::output;
}
