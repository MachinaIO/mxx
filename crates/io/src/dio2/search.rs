//! The production parameter search of dio2 (an ignored test; run it with
//! `cargo test -p mxx-io --release --lib dio2::search -- --ignored --nocapture`).
//!
//! For every ring dimension and CRT width, the search looks for the smallest CRT depth (the
//! number of primes of `Q_0`) at which every decoder of the evaluation decodes exactly and hides
//! its noise (`noise::noise_accepted` under the average-case model), with the smudge range
//! `P^d` the smallest that hides, and then checks the lattice security of `Q_0` with
//! `lattice-estimator-cli`: the root rows (binary or sign mask, CBD(1) errors) and the Static
//! preprocessing ciphertexts (ternary secret, Gaussian errors of `ERROR_SIGMA`). The PRF is the
//! recommended 100-bit Sparse-LWR row of `crates/exponent-lut/tests/lwr_prf_parameters.csv`
//! (`nu = 1024`, `h = 31`, `Q = 256`, `P = 32`).
//! Only the noise search uses representative label targets; all sampler sizes and joint event
//! counts remain those of the full plan, and production/claim graph construction is untouched.
//!
//! Settings (`MXX_IO_DIO2_SEARCH_*`): `LOG_N` (list, default `16,17`), `CRT_BITS` (list, default
//! `60,55,50`), `BASE_BITS` (list of gadget bases, default `CRT_BITS / 2`), `MIN_PRIMES`,
//! `MAX_PRIMES` (default `1`, `64`), `SECURITY_BITS` (default `100`), `STATISTICAL_BITS` (default
//! `100`), `FAILURE_EXPONENT` (default `100`), `NOISE_MODEL` (`average` or `worst`, default
//! `average`), `ERROR_SIGMA` (default `4`), `TRAPDOOR_SIGMA` (default `4`).

#[cfg(test)]
mod tests {
    use crate::dio2::{
        Dio2Error, Dio2Parameters, StepProgram, StreamingFunction,
        noise::{average_model, simulate_noise_targets},
        step::{ChainKeys, ChainProgram, NoiseTargets, StepPlan},
    };
    use bigdecimal::BigDecimal;
    use mxx_backends::{
        lattice_estimator::{binary_secret_cbd_security_bits, lattice_security_bits},
        openfhe_guard::gen_modulus_and_warmup,
    };
    use mxx_exponent_lut::{
        ExpLayout,
        program::{AttributeKnowledge, Operand},
        refresh::PrfSettings,
        rns_bgg::RnsBggDerivedWorkload,
    };
    use num_bigint::BigUint;
    use std::str::FromStr;

    fn setting(name: &str, default: &str) -> String {
        std::env::var(format!("MXX_IO_DIO2_SEARCH_{name}")).unwrap_or_else(|_| default.to_owned())
    }

    fn list(name: &str, default: &str) -> Vec<usize> {
        setting(name, default).split(',').map(|value| value.trim().parse().unwrap()).collect()
    }

    fn number(name: &str, default: usize) -> usize {
        setting(name, &default.to_string()).parse().unwrap()
    }

    /// The smallest positive `d` with `P^d > F 2^bits`, checked with exact integers rather
    /// than adding an unnecessary digit when the error-bit count is a multiple of `log2 P`.
    fn smudge_digits(error: &BigUint, bits: usize, modulus: usize) -> usize {
        let target = error << bits;
        let base = BigUint::from(modulus);
        let mut digits = ((target.bits() as f64 / (modulus as f64).log2()).ceil() as usize).max(1);
        while base.pow(digits as u32) <= target {
            digits += 1;
        }
        while digits > 1 && base.pow((digits - 1) as u32) > target {
            digits -= 1;
        }
        digits
    }

    #[test]
    fn smudge_digits_use_the_exact_strict_hiding_threshold() {
        let power = BigUint::from(32u8).pow(4);
        assert_eq!(smudge_digits(&(&power - 1u8), 0, 32), 4);
        assert_eq!(smudge_digits(&power, 0, 32), 5);
        assert_eq!(smudge_digits(&BigUint::from(1u8), 100, 32), 21);
    }

    /// The function of the claim export: one state bit, one evaluator bit, one obfuscator bit,
    /// and two table steps.
    fn function() -> StreamingFunction {
        let function =
            StreamingFunction::additive(1, 1, 1, ExpLayout { stride_log2: 1, width_log2: 2 });
        [vec![1, 0, 1, 0], vec![0, 1, 1, 0]].into_iter().fold(function.clone(), |acc, table| {
            acc.with_step(StepProgram::table(&function, table))
        })
    }

    /// The highest gadget level of a refreshing step's outputs (the level after a private
    /// output's): the decoding level the chain needs.
    fn decoding_level(parameters: &Dio2Parameters, function: &StreamingFunction) -> usize {
        let plan = StepPlan::for_noise(parameters, function, NoiseTargets::Representative).unwrap();
        let shape = |role: &str, members| parameters.prf.declared_shape(1, role, members).unwrap();
        let keys = ChainKeys {
            sigma: Some(shape("sigma", plan.ring_dimension)),
            cbd: Some(shape("cbd", plan.cbd_members())),
            smudge: shape("smudge", plan.noise_smudge_members(false)),
        };
        let start = std::time::Instant::now();
        let program =
            ChainProgram::refreshing(parameters, function, &plan, 0, &[true], &keys).unwrap();
        eprintln!("  program built in {:?}", start.elapsed());
        let workload = RnsBggDerivedWorkload::from_program(&program.program).unwrap();
        eprintln!("  workload derived in {:?}", start.elapsed());
        program
            .program
            .outputs()
            .iter()
            .map(|output| {
                let wire = match output {
                    Operand::Scalar(wire) | Operand::Member { family: wire, .. } => *wire,
                    Operand::Selected { .. } => unreachable!("a step outputs fixed members"),
                };
                let level = workload.level(wire).unwrap();
                match program.program.wire_knowledge(wire).unwrap() {
                    AttributeKnowledge::PublicPayload => level,
                    AttributeKnowledge::PrivatePayload => level + 1,
                }
            })
            .max()
            .unwrap()
    }

    /// The largest accepted decoder error `F` of a simulation, and whether every decoder fits
    /// and hides by `bits`.
    fn simulate(
        parameters: &Dio2Parameters,
        function: &StreamingFunction,
        bits: usize,
        failure_exponent: u32,
    ) -> Result<(BigUint, bool), Dio2Error> {
        let average = average_model(failure_exponent);
        let worst_case = setting("NOISE_MODEL", "average") == "worst";
        let noise = simulate_noise_targets(
            parameters,
            function,
            (!worst_case).then_some(&average),
            NoiseTargets::Representative,
        )?;
        let worst = noise
            .iter()
            .flat_map(|decoder| decoder.accepted_errors().iter().cloned())
            .max()
            .unwrap();
        let accepted = noise.iter().all(|decoder| decoder.fits() && decoder.hides(bits));
        Ok((worst, accepted))
    }

    #[test]
    #[ignore = "production parameter search; needs lattice-estimator-cli and minutes per candidate"]
    fn dio2_production_search() {
        let security_bits = number("SECURITY_BITS", 100) as u64;
        let statistical_bits = number("STATISTICAL_BITS", 100);
        let failure_exponent = number("FAILURE_EXPONENT", 100) as u32;
        let error_sigma = BigDecimal::from_str(&setting("ERROR_SIGMA", "4")).unwrap();
        let prf = PrfSettings {
            lwr_dimension: number("PRF_NU", 1024),
            support_weight: number("PRF_H", 31),
            input_modulus: number("PRF_Q", 256),
            output_modulus: number("PRF_P", 32),
            bucket_width: 0,
        };
        let function = function();
        for log_n in list("LOG_N", "16,17") {
            let n = 1u32 << log_n;
            for crt_bits in list("CRT_BITS", "60,55,50") {
                let bases = list("BASE_BITS", &(crt_bits.div_ceil(2)).to_string());
                for base_bits in bases {
                    search(
                        n,
                        crt_bits,
                        base_bits as u32,
                        &prf,
                        &function,
                        &error_sigma,
                        security_bits,
                        statistical_bits,
                        failure_exponent,
                    );
                }
            }
        }
    }

    /// The smallest CRT depth for one ring dimension, CRT width and gadget base.
    #[allow(clippy::too_many_arguments)]
    fn search(
        n: u32,
        crt_bits: usize,
        base_bits: u32,
        prf: &PrfSettings,
        function: &StreamingFunction,
        error_sigma: &BigDecimal,
        security_bits: u64,
        statistical_bits: usize,
        failure_exponent: u32,
    ) {
        let output_bits = (prf.output_modulus as f64).log2();
        'depth: for depth in number("MIN_PRIMES", 1)..=number("MAX_PRIMES", 64) {
            let primes = gen_modulus_and_warmup(n, depth, crt_bits, None).expect("CRT basis");
            let mut parameters = Dio2Parameters {
                ring_dimension: n,
                crt_bits,
                base_bits,
                primes,
                levels: 64,
                prf: prf.clone(),
                smudge_digits: 1,
                trapdoor_sigma: number("TRAPDOOR_SIGMA", 4) as u32,
                encoding_error_sigma: error_sigma.clone(),
            };
            if parameters.validate().is_err() {
                continue;
            }
            eprintln!(
                "depth {depth}: building the step Program ({:?})",
                std::time::SystemTime::now()
            );
            parameters.levels = decoding_level(&parameters, function) + 2;
            eprintln!(
                "depth {depth}: levels {} ({:?})",
                parameters.levels,
                std::time::SystemTime::now()
            );
            // The smudge digits that hide the decoder error `F`: `P^d > F 2^bits`.
            let (error, _) =
                match simulate(&parameters, function, statistical_bits, failure_exponent) {
                    Ok(result) => result,
                    Err(Dio2Error::RnsBgg(mxx_exponent_lut::rns_bgg::RnsBggError::Budget(
                        reason,
                    ))) => {
                        eprintln!("depth {depth}: rejected by {reason}");
                        continue;
                    }
                    Err(error) => panic!("noise simulation: {error}"),
                };
            let error_bits = error.bits() as f64;
            let mut digits = std::env::var("MXX_IO_DIO2_SEARCH_DIGITS").map_or_else(
                |_| smudge_digits(&error, statistical_bits, prf.output_modulus),
                |value| value.parse().unwrap(),
            );
            eprintln!("decoder error 2^{error_bits}: trying {digits} smudge digits");
            let mut accepted = false;
            for _ in 0..3 {
                parameters.smudge_digits = digits;
                let (error, ok) =
                    match simulate(&parameters, function, statistical_bits, failure_exponent) {
                        Ok(result) => result,
                        Err(Dio2Error::RnsBgg(mxx_exponent_lut::rns_bgg::RnsBggError::Budget(
                            reason,
                        ))) => {
                            eprintln!("depth {depth}, digits {digits}: rejected by {reason}");
                            continue 'depth;
                        }
                        Err(error) => panic!("noise simulation: {error}"),
                    };
                eprintln!(
                    "n=2^{} crt={crt_bits} base={base_bits} depth={depth} levels={} digits={digits} \
                     error=2^{} log2(Q0)={} accepted={ok}",
                    n.trailing_zeros(),
                    parameters.levels,
                    error.bits(),
                    depth * crt_bits,
                );
                if ok {
                    accepted = true;
                    break;
                }
                digits += 1;
            }
            if !accepted {
                continue;
            }
            let modulus = parameters.primes.iter().map(|q| BigUint::from(*q)).product::<BigUint>();
            let rows = binary_secret_cbd_security_bits(n, &modulus, 1, 2 * n as u64)
                .expect("root-row estimate");
            let encodings =
                lattice_security_bits(n, &modulus, error_sigma).expect("encoding estimate");
            eprintln!(
                "RESULT n=2^{} crt={crt_bits} base={base_bits} depth={depth} levels={} \
                 smudge_digits={} smudge_bits={} security(rows)={rows} security(encodings)={encodings} \
                 target={security_bits}",
                n.trailing_zeros(),
                parameters.levels,
                parameters.smudge_digits,
                (parameters.smudge_digits as f64 * output_bits) as usize,
            );
            // A larger depth only lowers the security, so the first accepted depth decides.
            return;
        }
        eprintln!(
            "RESULT n=2^{} crt={crt_bits} base={base_bits}: no depth up to the maximum is accepted",
            n.trailing_zeros()
        );
    }
}
