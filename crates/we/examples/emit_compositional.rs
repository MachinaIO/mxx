//! Export the actual Diamond workflow and its kernel-checked correctness certificate.
use mxx_gadgets::circuit::BooleanCircuitShape;
use mxx_ir_core::RealExpr;
use mxx_primitives::poly::dcrt::params::DCRTPolyParams;
use mxx_runtime::lean::export_dcrt_layouts;
use mxx_we::diamond::{
    DiamondWeCompiler, DiamondWeConfig, default_error_max_coefficient_bound,
    default_preimage_max_coefficient_bound,
};
use num_bigint::BigInt;
use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let directory = PathBuf::from(arguments.next().ok_or("output directory required")?);
    let configuration = arguments.next().unwrap_or_else(|| "fixture".to_owned());
    let (candidate, scaled, expanded, batched, wider) = match configuration.as_str() {
        "fixture" => (false, false, false, false, false),
        "candidate" | "minimal" => (true, false, false, false, false),
        "scaled" => (true, true, false, false, false),
        "expanded" => (true, true, true, false, false),
        "batched" => (true, true, false, true, false),
        "wider" | "radix" => (true, true, false, true, true),
        _ => {
            return Err("configuration must be fixture, candidate, scaled, expanded, batched, wider, radix, or minimal".into())
        }
    };
    let check = match arguments.next().as_deref() {
        None => false,
        Some("check") => true,
        Some(_) => return Err("optional final argument must be check".into()),
    };
    if arguments.next().is_some() {
        return Err("unexpected extra argument".into());
    }
    let parameters = if scaled {
        DCRTPolyParams::new(8, 8, 48, 24)
    } else if candidate {
        DCRTPolyParams::new(8, 4, 48, 24)
    } else {
        DCRTPolyParams::new(8, 1, 10, 4)
    };
    let layouts = export_dcrt_layouts([&parameters])?;
    let layout = &layouts[0];
    let gadget_base = BigInt::from(1u32) << layout.base_bits;
    let (trapdoor_sigma, error_sigma, error_bound, preimage_bound) = if candidate {
        let sigma = RealExpr::from_f64_exact(4.578)?;
        let error_bound = default_error_max_coefficient_bound(&sigma)?;
        let preimage_bound = default_preimage_max_coefficient_bound(
            &sigma,
            layout.ring_dimension as usize,
            layout.regular_digit_count,
            &gadget_base,
        )?;
        (sigma.clone(), sigma, error_bound, preimage_bound)
    } else {
        (RealExpr::from_integer(4), RealExpr::from_integer(1), 6.into(), 26.into())
    };
    let compiler = DiamondWeCompiler::new(
        DiamondWeConfig {
            modulus: layout.modulus.clone(),
            ring_dimension: layout.ring_dimension as usize,
            input_count: if expanded { 2 } else { 1 },
            digit_base: if configuration == "radix" {
                8
            } else if batched {
                4
            } else {
                2
            },
            batch_bits: if batched { 2 } else { 1 },
            gadget_base,
            digit_count: layout.regular_digit_count,
            trapdoor_sigma,
            error_sigma,
            error_max_coefficient_bound: error_bound,
            preimage_max_coefficient_bound: preimage_bound,
            bgg_tag: b"compositional-fixture".to_vec(),
        },
        BooleanCircuitShape {
            instance_width: if configuration == "minimal" {
                0
            } else if wider {
                2
            } else {
                1
            },
            witness_width: if expanded || batched { 2 } else { 1 },
            depth: if scaled { 4 } else { 2 },
            max_layer_width: if configuration == "minimal" {
                1
            } else if scaled {
                5
            } else {
                3
            },
        },
    )?;
    let artifact =
        mxx_we::lean::diamond::export_diamond_certificate(&parameters, &compiler, &directory)?;
    println!(
        "numeric_pass={} bound={} radius={} directory={}",
        artifact.numeric_pass(),
        artifact.numeric_bound,
        artifact.radius,
        artifact.directory.display()
    );
    if check {
        mxx_we::lean::check::check_generated_modules(
            &directory,
            std::time::Duration::from_secs(600),
        )?;
    }
    Ok(())
}
