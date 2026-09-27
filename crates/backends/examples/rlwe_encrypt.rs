//! Encrypts and decrypts one bit with Ring-LWE on the GPU.
//!
//! Run with `cargo run -r -p mxx-backends --example rlwe_encrypt --features gpu`.

use bigdecimal::BigDecimal;
use mxx_backends::{
    GpuRuntime, RuntimeValue, backend::poly_gpu::gpu_backend, poly::dcrt::gpu::GpuDCRTPolyParams,
    sampler::bounds::hard_cutoff_from_sigma_bound,
};
use mxx_dsl::{BuiltGraph, DslContext, DslError, HashTag, Int, IntType, Ring};
use mxx_ir_core::{IntExpr, ParamEnv, Rational, RealExpr, generate_crt_basis};
use num_bigint::BigInt;
use std::{collections::BTreeMap, sync::Arc};

/// Describes the protocol. Named parameters stay symbolic until they are bound.
fn rlwe_program(ring_dimension: u32) -> Result<BuiltGraph, DslError> {
    // R_Q = Z_Q[X]/(X^N + 1), where Q is a product of `crt_depth` primes of `crt_bits` bits.
    let ring = Ring::new(
        IntExpr::Var("crt_bits".into()),
        IntExpr::Var("crt_depth".into()),
        ring_dimension,
    );
    // Gaussian parameters: the width sigma and the bound on every sampled coefficient.
    let sigma = RealExpr::Var("sigma".into());
    let cutoff = IntExpr::Var("cutoff".into());
    let context = DslContext::new("rlwe-encrypt")
        .int_parameter("crt_bits")
        .int_parameter("crt_depth")
        .int_parameter("gadget_base_bits")
        .int_parameter("cutoff")
        .real_parameter("sigma");

    // Inputs supplied at run time: a 32-byte public seed and the message bit (0 or 1).
    let seed = ring.bytes_input("seed", 32);
    let message: Int = context.input("message", IntType)?;

    // The public element a is derived from the seed, so another party can recompute it.
    let a = ring.hash_matrix(seed, HashTag::from(b"rlwe-example/a".as_slice()), (1, 1));
    let s = ring.gaussian((1, 1), sigma.clone(), cutoff.clone());
    let e = ring.gaussian((1, 1), sigma, cutoff);

    // Encrypt the message as the constant term: b = a*s + e + floor(Q/2) * message.
    let delta = ring.polynomial([ring.modulus().floor_div(2)]);
    let plaintext = message.lift_to_constant_polynomial(ring.matrix_type((1, 1)));
    let b = &a * &s + e + &delta * &plaintext;

    // Decrypt: round the constant term of b - a*s to the nearest multiple of Q/2.
    let decrypted = (b.clone() - &a * &s).threshold_decode_bools(2, 1).remove(0);

    context.output("ciphertext", b)?.output("decrypted", decrypted)?.build()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ring_dimension = 4096;
    let (crt_bits, crt_depth, gadget_base_bits) = (60, 3, 20);
    let sigma = 4;
    let cutoff = hard_cutoff_from_sigma_bound(&BigDecimal::from(sigma)); // 6.5 sigma

    // Bind the parameters, then check the whole program under them.
    let bindings = ParamEnv {
        integers: BTreeMap::from([
            ("crt_bits".to_owned(), BigInt::from(crt_bits)),
            ("crt_depth".to_owned(), BigInt::from(crt_depth)),
            ("gadget_base_bits".to_owned(), BigInt::from(gadget_base_bits)),
            ("cutoff".to_owned(), BigInt::from(cutoff)),
        ]),
        reals: BTreeMap::from([("sigma".to_owned(), Rational::from_integer(BigInt::from(sigma)))]),
        ..ParamEnv::default()
    };
    let program = rlwe_program(ring_dimension)?.validate(&bindings)?;

    // Register the same ring with a GPU backend.
    let moduli = generate_crt_basis(ring_dimension, crt_depth, crt_bits)?;
    let gpu_params = GpuDCRTPolyParams::new(ring_dimension, moduli, gadget_base_bits, None);
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_params]))?;

    let inputs = |seed: u8, bit: bool| {
        BTreeMap::from([
            ("seed".to_owned(), RuntimeValue::Bytes(Arc::from([seed; 32]))),
            ("message".to_owned(), RuntimeValue::Int(BigInt::from(bit))),
        ])
    };

    // Planning picks the parallelism and memory schedule that fit this GPU. It needs inputs of
    // the right types, so it gets dummy ones.
    let mut plan = runtime.plan(program, &inputs(0, false))?;

    // The plan then runs on real inputs without being planned again.
    for (seed, bit) in [(1, false), (2, true)] {
        let result = runtime.execute(&mut plan, inputs(seed, bit))?;
        assert_eq!(runtime.download_bool(&result["decrypted"])?, bit);
        println!("bit {bit} decrypted correctly");
    }
    Ok(())
}
