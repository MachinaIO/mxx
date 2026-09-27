//! Encrypts and decrypts a polynomial of bits with Ring-LWE on the GPU.
//!
//! Run with `cargo run -r -p mxx-backends --example rlwe_encrypt --features gpu`.

use bigdecimal::BigDecimal;
use mxx_backends::{
    GpuRuntime, RuntimeValue, backend::poly_gpu::gpu_backend, poly::dcrt::gpu::GpuDCRTPolyParams,
    sampler::bounds::hard_cutoff_from_sigma_bound,
};
use mxx_dsl::{BuiltGraph, DslContext, DslError, Family, HashTag, Ring};
use mxx_ir_core::{IntExpr, ParamEnv, Rational, RealExpr, generate_crt_basis};
use num_bigint::BigInt;
use std::{collections::BTreeMap, sync::Arc};

/// Describes the protocol. Named parameters stay symbolic until they are bound.
fn rlwe_program(ring_dimension: u32) -> Result<BuiltGraph, DslError> {
    let sigma = RealExpr::Var("sigma".into());
    let cutoff = IntExpr::Var("cutoff".into());
    // R_Q = Z_Q[X]/(X^N + 1), where Q is a product of `crt_depth` primes of `crt_bits` bits.
    let ring = Ring::new(
        IntExpr::Var("crt_bits".into()),
        IntExpr::Var("crt_depth".into()),
        ring_dimension,
    );
    let context = DslContext::new("rlwe-encrypt")
        .int_parameter("crt_bits")
        .int_parameter("crt_depth")
        .int_parameter("gadget_base_bits")
        .int_parameter("cutoff")
        .real_parameter("sigma");

    // Inputs supplied at run time: a 32-byte public seed and one message bit per coefficient.
    let seed = ring.bytes_input("seed", 32);
    let bits = context.int_family_input("bits", ring_dimension);

    // The public element a is derived from the seed, so another party can recompute it.
    let a = ring.hash_matrix(seed, HashTag::from(b"rlwe-example/a".as_slice()), (1, 1));
    let s = ring.gaussian((1, 1), sigma.clone(), cutoff.clone());
    let e = ring.gaussian((1, 1), sigma, cutoff);

    // Encrypt m(X) as b = a*s + e + floor(Q/2) * m(X).
    let delta = ring.polynomial([ring.modulus().floor_div(2)]);
    let b = &a * &s + e + &delta * &ring.from_coefficients(&bits);

    // Decrypt: round each coefficient of b - a*s to the nearest multiple of Q/2.
    let decrypted = (b.clone() - &a * &s).threshold_decode_ints(2, ring_dimension as usize);

    context.output("ciphertext", b)?.output("decrypted", Family::pack(decrypted)?)?.build()
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

    let message = |seed: usize| {
        (0..ring_dimension as usize).map(|i| BigInt::from((i * 7 + seed) % 2)).collect::<Vec<_>>()
    };
    let inputs = |seed: usize| {
        BTreeMap::from([
            ("seed".to_owned(), RuntimeValue::Bytes(Arc::from([seed as u8; 32]))),
            ("bits".to_owned(), RuntimeValue::integer_values(message(seed))),
        ])
    };

    // Planning picks the parallelism and memory schedule that fit this GPU.
    let mut plan = runtime.plan(program, &inputs(1))?;

    // The plan then runs on new inputs without being planned again.
    for seed in 1..=3 {
        let result = runtime.execute(&mut plan, inputs(seed))?;
        let decrypted = result.output("decrypted").ok_or("missing output")?;
        let decrypted = runtime.download_integer_family_output(&decrypted)?;
        assert_eq!(decrypted, message(seed), "decryption mismatch for seed {seed}");
        println!("seed {seed}: {ring_dimension} bits decrypted correctly");
    }
    Ok(())
}
