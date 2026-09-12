//! Parameters, security estimates and durable reports for GPU FHE integration tests.
use bigdecimal::BigDecimal;
use mxx_fhe::{BgvHybridParams, BgvParams, FheCommonParams, RingGswParams};
use mxx_ir_core::node::SampleRange;
use mxx_primitives::{
    poly::{PolyParams, dcrt::params::DCRTPolyParams},
    sampler::bounds::hard_cutoff_from_sigma_bound,
};
use num_bigint::BigUint;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

pub fn integer(name: &str, default: usize) -> usize {
    env::var(name)
        .map(|v| v.parse().unwrap_or_else(|_| panic!("invalid {name}: {v}")))
        .unwrap_or(default)
}

pub fn manifest() -> Option<Value> {
    env::var("FHE_TEST_MANIFEST").ok().map(|path| {
        serde_json::from_slice(&fs::read(path).expect("read FHE manifest"))
            .expect("valid FHE manifest JSON")
    })
}

fn primes(name: &str, defaults: Vec<u64>) -> Vec<u64> {
    env::var(name)
        .map(|value| {
            value
                .split(',')
                .map(|p| p.trim().parse().expect("comma-separated CRT primes"))
                .collect()
        })
        .unwrap_or(defaults)
}

fn common(
    n: usize,
    q: Vec<u64>,
    base_bits: usize,
    default_sigma: &str,
    binary: bool,
) -> FheCommonParams {
    let sigma = env::var("FHE_TEST_SIGMA").unwrap_or_else(|_| default_sigma.into());
    let sigma_bound: BigDecimal = sigma.parse().expect("positive decimal sigma");
    let error_sigma: f64 = sigma.parse().expect("finite positive sigma");
    assert!(error_sigma.is_finite() && error_sigma > 0.0, "finite positive sigma required");
    let error_cutoff = hard_cutoff_from_sigma_bound(&sigma_bound);
    let binary = match env::var("FHE_TEST_SECRET_DISTRIBUTION").as_deref() {
        Ok("binary") => true,
        Ok("ternary") => false,
        Err(_) => binary,
        _ => panic!("FHE_TEST_SECRET_DISTRIBUTION must be binary or ternary"),
    };
    let bits = q.iter().map(|p| (64 - p.leading_zeros()) as usize).max().expect("nonempty Q");
    let params = FheCommonParams {
        ring: DCRTPolyParams::try_new(
            n.try_into().expect("ring dimension fits u32"),
            q.len(),
            bits,
            base_bits.try_into().expect("base bits fit u32"),
            Some(q),
            None,
        )
        .expect("valid exact CRT basis"),
        secret_range: SampleRange {
            minimum: (if binary { 0 } else { -1 }).into(),
            maximum: 1.into(),
        },
        error_sigma,
        error_cutoff,
    };
    params.validate().expect("valid FHE parameters");
    params
}

pub fn bgv_params() -> BgvParams {
    let profile = env::var("FHE_TEST_PROFILE").unwrap_or_else(|_| "bgv-54".into());
    let (mut q, mut p) = match profile.as_str() {
        "bgv-54" => {
            (vec![18014398507892737, 18014398508138497, 18014398508400641], vec![72057594037616641])
        }
        "bgv-36" => (
            vec![68717740033, 68718346241, 68718428161, 68719230977],
            vec![137438773249, 137438822401],
        ),
        _ => panic!("FHE_TEST_PROFILE must be bgv-54 or bgv-36"),
    };
    let mut n = 8192;
    let mut t = 1032193;
    let mut digit_size = p.len();
    let mut sigma = "3.2".to_owned();
    if let Some(m) = manifest() {
        n = m["n"].as_u64().expect("manifest n") as usize;
        t = m["t"].as_u64().expect("manifest t") as usize;
        q = m["q"]
            .as_array()
            .expect("manifest q")
            .iter()
            .map(|v| v.as_u64().expect("u64 prime"))
            .collect();
        p = m["p"]
            .as_array()
            .expect("manifest p")
            .iter()
            .map(|v| v.as_u64().expect("u64 prime"))
            .collect();
        digit_size = m["digit_size"].as_u64().expect("manifest digit_size") as usize;
        sigma = m["sigma"].to_string();
    }
    let common = common(
        integer("FHE_TEST_RING_DIMENSION", n),
        primes("FHE_TEST_Q_PRIMES", q),
        integer("FHE_TEST_BASE_BITS", 8),
        &sigma,
        false,
    );
    BgvParams::new(
        common,
        integer("FHE_TEST_PLAINTEXT_MODULUS", t) as u64,
        Some(BgvHybridParams {
            digit_size: integer("FHE_TEST_HYBRID_DIGIT_SIZE", digit_size),
            auxiliary_primes: primes("FHE_TEST_P_PRIMES", p),
        }),
    )
    .expect("valid BGV parameters")
}

pub fn bgv_digit_size() -> usize {
    let default = manifest()
        .map(|m| m["digit_size"].as_u64().expect("manifest digit_size") as usize)
        .unwrap_or_else(|| {
            if env::var("FHE_TEST_PROFILE").as_deref() == Ok("bgv-36") { 2 } else { 1 }
        });
    integer("FHE_TEST_HYBRID_DIGIT_SIZE", default)
}

pub fn ring_gsw_params() -> RingGswParams {
    let n = integer("FHE_TEST_RING_DIMENSION", 2048);
    let bits = integer("FHE_TEST_CRT_BITS", 60);
    let depth = integer("FHE_TEST_CRT_DEPTH", 1);
    let base = integer("FHE_TEST_BASE_BITS", 8);
    let q = DCRTPolyParams::new(n as u32, depth, bits, base as u32, None, None).to_crt().0;
    let common = common(n, primes("FHE_TEST_Q_PRIMES", q), base, "339.0", true);
    let scale = env::var("FHE_TEST_SCALE")
        .map(|s| s.parse().expect("integer scale"))
        .unwrap_or_else(|_| common.ring.modulus().as_ref() / BigUint::from(4 * n));
    RingGswParams::new(common, scale, BigUint::from(1u8)).expect("valid Ring-GSW parameters")
}

pub fn repetitions() -> usize {
    let count = integer("FHE_BENCH_REPEATS", 100);
    assert!(count > 0, "positive measurement count required");
    count
}

pub fn modswitch_steps() -> usize {
    let steps = integer("FHE_TEST_MODSWITCH_STEPS", 1);
    assert!(steps > 0, "the round trip must include modulus switching");
    steps
}

pub fn security_report(common: &FheCommonParams, auxiliary: Option<&[u64]>) -> Value {
    let secret = if common.secret_range.minimum == 0.into() { "Binary" } else { "Ternary" };
    let s_dist = json!({"name":secret}).to_string();
    let e_dist = json!({"name":"DiscreteGaussian", "stddev":common.error_sigma}).to_string();
    let exact = env::var("FHE_TEST_ESTIMATOR_MODE").unwrap_or_else(|_| "rough".into());
    assert!(exact == "exact" || exact == "rough", "estimator mode must be exact or rough");
    // Reporting is mandatory; a model-specific lower bound is an explicit opt-in.
    // In particular Core-SVP bits must not be compared to the HE Standard label.
    let required = integer("FHE_TEST_SECURITY_BITS", 0);
    // These are different attack/cost models, not different numerical precision.
    // Pinned estimator source: estimator/lwe.py and estimator/conf.py.
    let cost_model =
        if exact == "rough" { "ADPS16 / Core-SVP" } else { "MATZOV (estimator default)" };
    let attack_set = if exact == "rough" {
        "usvp, dual_hybrid; conditional arora-gb (LWE.estimate.rough)"
    } else {
        "arora-gb, bkw, usvp, bdd, bdd_hybrid, bdd_mitm_hybrid, dual, dual_hybrid (LWE.estimate)"
    };
    let q = common.ring.modulus();
    let mut moduli = vec![("Q", q.as_ref().clone())];
    if let Some(p) = auxiliary {
        moduli.push(("QP", q.as_ref() * p.iter().copied().map(BigUint::from).product::<BigUint>()));
    }
    // PhantomFHE's HE Standard-derived ternary/classical-128 modulus table.
    // This table classification is separate from the selected estimator model.
    let table_limit = match common.ring.ring_dimension() {
        1024 => Some(27u64),
        2048 => Some(54),
        4096 => Some(109),
        8192 => Some(218),
        16384 => Some(438),
        32768 => Some(881),
        65536 => Some(1777),
        131072 => Some(3576),
        _ => None,
    };
    let (security_basis, security_modulus) = moduli.last().unwrap();
    let he_standard = json!({"classification":"HE Standard-derived ternary classical 128-bit table",
        "source":"https://github.com/encryptorion-lab/phantom-fhe/blob/1f4a198443b3af77118e51f53d5b8f332154b875/include/host/hestdparms.h",
        "basis":security_basis,"actual_modulus_bits":security_modulus.bits(),"table_modulus_bits":table_limit,
        "table_parameters_match":table_limit.map(|limit| secret == "Ternary" && common.error_sigma == 3.2 && security_modulus.bits() <= limit)});
    let version = Command::new("lattice-estimator-cli")
        .arg("--version")
        .output()
        .expect("lattice-estimator-cli is required");
    assert!(version.status.success(), "estimator version failed");
    let executable = env::split_paths(&env::var_os("PATH").expect("PATH"))
        .map(|dir| dir.join("lattice-estimator-cli"))
        .find(|path| path.is_file())
        .expect("estimator executable on PATH")
        .canonicalize()
        .expect("resolve estimator path");
    let revision = |directory: &std::path::Path| {
        let output = Command::new("git")
            .arg("-C")
            .arg(directory)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("read estimator revision");
        if output.status.success() {
            String::from_utf8_lossy(&output.stdout).trim().to_owned()
        } else {
            "unavailable".into()
        }
    };
    let cli_directory = executable.parent().expect("estimator directory");
    let root = Command::new("git")
        .arg("-C")
        .arg(cli_directory)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .expect("estimator repository");
    let estimator_revision = if root.status.success() {
        revision(&PathBuf::from(String::from_utf8_lossy(&root.stdout).trim()).join("estimator"))
    } else {
        "unavailable".into()
    };
    let estimates = moduli.into_iter().map(|(basis, modulus)| {
        println!("Estimating {basis} security: N={}, modulus_bits={}, mode={exact}", common.ring.ring_dimension(), modulus.bits());
        let request = json!({"n":common.ring.ring_dimension(),"q":modulus.to_string(),"secret":s_dist,"error":e_dist,"mode":exact,"m":"infinity","cli_revision":revision(cli_directory),"estimator_revision":estimator_revision});
        let digest = Sha256::digest(serde_json::to_vec(&request).unwrap());
        let cache_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data/fhe-round-trip/security");
        fs::create_dir_all(&cache_directory).expect("create security cache directory");
        let cache_path = cache_directory.join(format!("{digest:x}.json"));
        let cached = fs::read(&cache_path).ok().map(|bytes| serde_json::from_slice::<Value>(&bytes).expect("valid cached security report"))
            .filter(|v| v["request"] == request && estimator_revision != "unavailable" && revision(cli_directory) != "unavailable");
        let mut command = Command::new("lattice-estimator-cli");
        command.args([common.ring.ring_dimension().to_string(), modulus.to_string(), "--s-dist".into(), s_dist.clone(), "--e-dist".into(), e_dist.clone()]);
        if exact == "exact" { command.arg("--exact"); }
        let (stdout,stderr) = if let Some(cached) = &cached {
            println!("Reusing security estimate for identical parameters and estimator revisions");
            (cached["stdout"].as_str().expect("cached stdout").to_owned(),cached["stderr"].as_str().expect("cached stderr").to_owned())
        } else {
            let output = command.output().expect("run lattice-estimator-cli (Sage must be on PATH)");
            let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            assert!(output.status.success(), "estimator failed: {stdout}\n{stderr}");
            (stdout,stderr)
        };
        let bits: u64 = stdout.lines().rev().find(|s| !s.trim().is_empty()).expect("estimator output").trim().parse().expect("integer security bits");
        if cached.is_none() {
            fs::write(&cache_path,serde_json::to_vec_pretty(&json!({"request":request,"stdout":stdout,"stderr":stderr})).unwrap()).expect("save security estimate");
        }
        println!("{basis} security_bits={bits}, model={cost_model}, required_in_this_model={required}");
        assert!(bits >= required as u64, "{basis} estimate {bits} under {cost_model} is below requested {required}; this is not the HE Standard table classification");
        json!({"basis":basis,"modulus":modulus.to_string(),"security_bits":bits,"cached":cached.is_some(),"stdout":stdout,"stderr":stderr})
    }).collect::<Vec<_>>();
    json!({"estimator_version":String::from_utf8_lossy(&version.stdout), "cli_revision":revision(cli_directory),"estimator_revision":estimator_revision,"mode":exact,"cost_model":cost_model,"attack_set":attack_set,"he_standard":he_standard,"samples":"infinity", "ring_dimension":common.ring.ring_dimension(), "secret_distribution":secret,"error_sigma":common.error_sigma,"error_cutoff":common.error_cutoff.to_string(),"required_bits":required,"estimates":estimates})
}

pub fn write_report(scheme: &str, report: Value) {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/fhe-round-trip")
        .join(scheme);
    fs::create_dir_all(&directory).expect("create report directory");
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = directory.join(format!("{now}-{}.json", std::process::id()));
    let git = Command::new("git").args(["rev-parse", "HEAD"]).output().expect("git revision");
    let status =
        Command::new("git").args(["status", "--porcelain"]).output().expect("worktree status");
    let binary_digest = Sha256::digest(
        fs::read(env::current_exe().expect("test executable")).expect("read test executable"),
    );
    let gpu = Command::new("nvidia-smi")
        .args(["--query-gpu=name,uuid,driver_version", "--format=csv,noheader"])
        .output()
        .expect("GPU metadata");
    let full = json!({"timestamp_unix_ns":now.to_string(),"commit":String::from_utf8_lossy(&git.stdout).trim(),"worktree_dirty":!status.stdout.is_empty(),"test_binary_sha256":format!("{binary_digest:x}"),"gpu":String::from_utf8_lossy(&gpu.stdout).trim(),"result":report});
    fs::write(&path, serde_json::to_vec_pretty(&full).unwrap()).expect("write report");
    println!("FHE_REPORT={}", path.display());
}
