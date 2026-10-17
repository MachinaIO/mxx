//! Compare small GPU smoke-test parameter candidates using the noise model only.
//! No ciphertext is evaluated here; round trips remain GPU-only.
use bigdecimal::BigDecimal;
use mxx_exponent_lut::{ExpLayout, refresh::PrfSettings};
use mxx_io::dio2::{Dio2Parameters, StreamingFunction, noise::simulate_noise};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let function =
        StreamingFunction::additive(1, 1, 1, ExpLayout { stride_log2: 0, width_log2: 2 })
            .with_step_row(vec![1, 0, 1, 0])?
            .with_step_row(vec![0, 1, 1, 0])?;
    for (n, q, p, primes, base_bits) in [
        (4, 4, 4, 4, 15),
        (4, 4, 4, 5, 20),
        (4, 4, 4, 7, 30),
        (2, 4, 4, 3, 8),
        (2, 4, 4, 4, 12),
        (4, 4, 4, 2, 3),
        (4, 4, 4, 3, 8),
        (4, 4, 4, 4, 12),
        (4, 4, 2, 2, 4),
        (4, 4, 2, 3, 8),
        (4, 4, 2, 3, 10),
        (4, 4, 2, 4, 12),
        (4, 8, 4, 3, 8),
        (4, 8, 4, 4, 12),
    ] {
        let parameters = Dio2Parameters {
            ring_dimension: n,
            crt_bits: 60,
            base_bits,
            primes: mxx_ir_core::generate_crt_basis(n, primes, 60)?,
            levels: 6,
            prf: PrfSettings {
                lwr_dimension: 1,
                support_weight: 1,
                input_modulus: q,
                output_modulus: p,
                bucket_width: 0,
            },
            smudge_digits: 1,
            trapdoor_sigma: 4,
            encoding_error_sigma: BigDecimal::from(0),
        };
        match simulate_noise(&parameters, &function, None) {
            Ok(noises) => println!(
                "n={n} Q={q} P={p} primes={primes} base_bits={base_bits} w0={} fits={} max_error_bits={} min_spacing_bits={}",
                parameters.columns()?,
                noises.iter().all(|noise| noise.fits()),
                noises.iter().flat_map(|noise| &noise.errors).map(|e| e.bits()).max().unwrap(),
                noises.iter().flat_map(|noise| &noise.spacings).map(|s| s.bits()).min().unwrap()
            ),
            Err(error) => {
                println!("n={n} Q={q} P={p} primes={primes} base_bits={base_bits} error={error}")
            }
        }
    }
    Ok(())
}
