//! Bounded, numeric-only exploration of real DCRT layouts for the compositional proof.
//! No security estimate, execution, or application-theorem acceptance is asserted.
use mxx_gadgets::circuit::BooleanCircuitShape;
use mxx_ir_core::{
    RealExpr,
    artifact::{ProductionId, SpecHash, export_validated_manifest},
    validate, validate_with_manifests,
};
use mxx_primitives::poly::dcrt::params::DCRTPolyParams;
use mxx_runtime::lean::{export_dcrt_layouts, render_backend_context};
use mxx_we::diamond::{
    DiamondWeCompiler, DiamondWeConfig, default_error_max_coefficient_bound,
    default_preimage_max_coefficient_bound,
};
use num_bigint::{BigInt, BigUint};
use std::{collections::BTreeMap, error::Error};

fn main() -> Result<(), Box<dyn Error>> {
    // Two regular digits PER CRT TOWER: ceil(48 / 24) = 2. Total regular digits
    // grow with CRT depth. The four candidates retain the same positive sampler sigmas.
    let candidates = [(2, 48, 24), (4, 48, 24), (8, 48, 24), (16, 48, 24)];
    println!("numeric_only=true security_estimated=false runtime_executed=false");
    // Backend parameter construction warms process-global OpenFHE state, so this small
    // diagnostic visits the four candidates sequentially.
    for (crt_depth, crt_bits, base_bits) in candidates {
        let parameters = DCRTPolyParams::new(8, crt_depth, crt_bits, base_bits);
        let layouts = export_dcrt_layouts([&parameters])?;
        let layout = &layouts[0];
        let error_sigma = RealExpr::from_f64_exact(4.578)?;
        let trapdoor_sigma = RealExpr::from_f64_exact(4.578)?;
        let base = BigInt::from(1u64 << base_bits);
        let error_bound = default_error_max_coefficient_bound(&error_sigma)?;
        let preimage_bound = default_preimage_max_coefficient_bound(
            &trapdoor_sigma,
            8,
            layout.regular_digit_count,
            &base,
        )?;
        let compiler = DiamondWeCompiler::new(
            DiamondWeConfig {
                modulus: layout.modulus.clone(),
                ring_dimension: 8,
                input_count: 1,
                digit_base: 2,
                batch_bits: 1,
                gadget_base: base.clone(),
                digit_count: layout.regular_digit_count,
                trapdoor_sigma,
                error_sigma,
                error_max_coefficient_bound: error_bound.clone(),
                preimage_max_coefficient_bound: preimage_bound.clone(),
                bgg_tag: b"compositional-candidate-probe".to_vec(),
            },
            BooleanCircuitShape {
                instance_width: 1,
                witness_width: 1,
                depth: 2,
                max_layer_width: 3,
            },
        )?;
        let protocol = compiler.protocol_decl()?;
        let bindings = compiler.circuit_bindings()?;
        let declaration = protocol.protocol();
        let encryption = declaration
            .stages()
            .iter()
            .find(|stage| stage.id.0 == "encrypt")
            .ok_or("missing encryption graph")?;
        let producer = validate(&encryption.graph, &bindings)?;
        let production = ProductionId { spec_hash: SpecHash([0; 32]), execution_nonce: [0; 32] };
        let manifests = BTreeMap::from([(
            production.clone(),
            export_validated_manifest(production, &producer)?,
        )]);
        for stage in declaration.stages() {
            validate_with_manifests(&stage.graph, &bindings, &manifests)?;
        }
        for requirement in &declaration.bundle.requirements {
            validate_with_manifests(&requirement.graph, &bindings, &manifests)?;
        }
        validate_with_manifests(&declaration.bundle.ideal.graph, &bindings, &manifests)?;
        render_backend_context(&layouts, "Backend", "DiamondBackend")?;

        let q = layout.modulus.to_biguint().ok_or("nonpositive modulus")?;
        let n = BigUint::from(8u32);
        let ell = BigUint::from(layout.regular_digit_count);
        let inner = BigUint::from(compiler.config.input_config().state_columns()?);
        let e = error_bound.to_biguint().ok_or("negative error bound")?;
        let k = preimage_bound.to_biguint().ok_or("negative preimage bound")?;
        let d = base.to_biguint().ok_or("negative base")? / 2u32;
        // The proved recurrence at L=1: P0=1, N0=E, N1=2*n*E+inner*n*K*E.
        let n1 = 2u32 * &n * &e + &inner * &n * &k * &e;
        let b0 = &inner * &n * &k * &n1;
        let a = &ell * &n * &d;
        let factor = 2u32 * &a + 4u32;
        let bh = factor.pow(2) * &b0;
        let final_bound = 2u32 * &b0 + &a * (&b0 + &bh);
        let quarter = &q / 4u32;
        let half = &q / 2u32;
        let radius = [
            quarter.clone(),
            &q - 3u32 * &quarter,
            &half - &quarter + 1u32,
            3u32 * &quarter - &half + 1u32,
        ]
        .into_iter()
        .min()
        .ok_or("empty decoder interval")?;
        println!(
            "n=8 crt_depth={crt_depth} crt_bits={crt_bits} base_bits={base_bits} \
             digits_per_tower={} regular_digits={} inner={inner} E={e} K={k} D={d} L=1 H=2 \
             q={q} q_bits={} N1={n1} B0={b0} factor={factor} BH={bh} \
             final_bound={final_bound} radius={radius} numeric_pass={} \
             layout_valid=true graphs_valid=true crt_moduli={:?}",
            layout.crt_bits.div_ceil(layout.base_bits as usize),
            layout.regular_digit_count,
            q.bits(),
            final_bound < radius,
            layout.crt_moduli,
        );
    }
    Ok(())
}
