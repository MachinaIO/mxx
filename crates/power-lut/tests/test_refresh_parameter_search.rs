//! Symbolic Section-7 Power-LUT refresh parameter search.
//!
//! This is deliberately ignored: it invokes Sage's `lattice-estimator-cli`
//! and constructs large symbolic candidates, but does not execute a GPU/CPU
//! backend, sample runtime artifacts, perform an encryption/decryption round
//! trip, or run a benchmark.  The production integration gate performs generic
//! request validation for the frozen graph, while refresh acceptance is decided
//! by the Power-LUT application-specific exact noise bound.
//!
//! Set `MXX_POWER_LUT_REFRESH_PHASE1_CHECKPOINT` to an explicit JSON path to
//! persist the public Phase-1 rows and reuse them on a later run. Existing
//! checkpoints are accepted only when their declared grid and security model
//! exactly match this test; without the variable every run starts fresh.

mod refresh_parameter_search_support;

use mxx_primitives::poly::PolyParams;
use refresh_parameter_search_support::{
    PHASE1_CHECKPOINT_ENV, SearchConfig, check_refresh_bundle, lattice_security_bits,
    load_or_search_phase1, persist_accepted_phase2_profile, prepare_candidate, run_estimator,
    search_qualified_profiles, search_report, search_with_hooks,
};
use std::{env, path::PathBuf};
use tracing_subscriber::EnvFilter;

fn install_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("mxx_power_lut=debug,mxx_runtime=debug,info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).with_test_writer().try_init();
}

/// Exhaust the ordered Phase-1 tuple grid, preferring a primary-128 row over
/// any fallback-100 row; then search the declared CRT/base/depth/ring grid in
/// order, stopping at the first DCRT candidate satisfying security and the
/// application-specific exact noise threshold after generic request validation.
#[test]
#[ignore = "CPU/Sage symbolic parameter search; not GPU runtime"]
fn test_refresh_parameter_search() {
    install_tracing();
    let estimator_root = env::var_os("MXX_LATTICE_ESTIMATOR_ROOT")
        .map(PathBuf::from)
        .expect("MXX_LATTICE_ESTIMATOR_ROOT must point to the pinned lattice-estimator checkout");
    let sage_binary =
        env::var_os("MXX_SAGE_BINARY").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("sage"));
    let estimator_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("sparse_lwr_estimator.sage.py");
    let mut config = SearchConfig::from_env().expect("valid configured Power-LUT refresh profile");
    config.pbc_max_attempts = 128;
    let checkpoint_path = env::var_os(PHASE1_CHECKPOINT_ENV).map(PathBuf::from);
    let sparse_profile =
        load_or_search_phase1(&config, checkpoint_path.as_deref(), |universe, weight, q_l, p| {
            let tuple = config
                .sparse_lwr_phase1_grid
                .iter()
                .find(|tuple| {
                    tuple.nu == universe && tuple.h == weight && tuple.q_l == q_l && tuple.p == p
                })
                .ok_or_else(|| {
                    "estimator callback received an undeclared Phase-1 tuple".to_owned()
                })?;
            let candidate = tuple.candidate()?;
            let estimator = run_estimator(
                &candidate,
                &estimator_script,
                &estimator_root,
                &sage_binary,
                Some(&config.phase1_estimator_commit),
            )?;
            if (estimator.minimum_classical_bits - tuple.estimator_minimum_classical_bits).abs() >
                1e-12
            {
                return Err(format!(
                    "estimator minimum for (Q_L={}, p={}, nu={}, h={}) was {}, expected {}",
                    tuple.q_l,
                    tuple.p,
                    tuple.nu,
                    tuple.h,
                    estimator.minimum_classical_bits,
                    tuple.estimator_minimum_classical_bits
                ));
            }
            if !estimator.minimum_classical_bits.is_finite() ||
                estimator.minimum_classical_bits < 0.0
            {
                return Err("estimator minimum must be finite and nonnegative".to_owned());
            }
            Ok(estimator.minimum_classical_bits.floor() as u64)
        })
        .expect("configured sparse-LWR Phase-1 search must select a profile");
    let (sparse_profile, result) = search_qualified_profiles(&sparse_profile, |profile| {
        config.sparse_lwr_universe = profile.universe;
        config.sparse_lwr_weight = profile.weight;
        config.sparse_lwr_modulus = profile.q_l;
        config.sparse_lwr_output_modulus = profile.p;
        config.lut_width = profile.tuple.lut_width;
        search_with_hooks(
            &config,
            &profile,
            |candidate| prepare_candidate(&config, candidate),
            |candidate| {
                let ring_dimension = 1usize
                    .checked_shl(candidate.log_ring_dimension as u32)
                    .ok_or_else(|| "ring dimension shift overflow".to_owned())?;
                let dcrt = mxx_primitives::poly::dcrt::params::DCRTPolyParams::new(
                    ring_dimension as u32,
                    candidate.crt_depth,
                    candidate.crt_bits,
                    candidate.base_bits,
                );
                let bgg = lattice_security_bits(
                    ring_dimension,
                    dcrt.modulus().as_ref(),
                    config.error_sigma,
                )?;
                Ok(bgg)
            },
            |prepared| check_refresh_bundle(&config, prepared.candidate, prepared),
        )
    })
    .expect("configured Power-LUT refresh search must produce a valid candidate");
    if let Some(path) = checkpoint_path.as_deref() {
        persist_accepted_phase2_profile(path, &config, &sparse_profile, result.candidate)
            .expect("persist accepted Phase-2 profile");
    }
    config.crt_bits = result.candidate.crt_bits;
    config.base_bits = result.candidate.base_bits;
    let report =
        serde_json::to_string_pretty(&search_report(&config, &sparse_profile, result)).unwrap();
    if let Some(path) = env::var_os("MXX_POWER_LUT_REFRESH_SEARCH_OUTPUT") {
        std::fs::write(&path, &report).expect("write Power-LUT refresh search report");
    } else {
        println!("{report}");
    }
}
