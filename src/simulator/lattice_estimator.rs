use num_bigint::BigUint;
use serde_json::{Value, json};
use std::{path::Path, process::Command, result::Result};
/// Enum describing supported noise distributions.
/// This mirrors the JSON spec expected by the Python CLI.
#[derive(Debug, Clone)]
pub enum Distribution {
    DiscreteGaussian { stddev: f64, mean: Option<f64>, n: Option<u64> },
    DiscreteGaussianAlpha { alpha: f64, mean: Option<f64>, n: Option<u64> }, /* 'q' is taken
                                                                              * from top-level
                                                                              * <q> */
    CenteredBinomial { eta: u64, n: Option<u64> },
    Uniform { a: i64, b: i64, n: Option<u64> },
    UniformMod { n: Option<u64> }, // 'q' is taken from top-level <q>
    SparseTernary { p: u64, m: u64, n: Option<u64> },
    SparseBinary { hw: u64, n: Option<u64> },
    Binary,
    Ternary,
}

impl Distribution {
    /// Convert this distribution to a JSON value expected by the CLI.
    fn to_json_value(&self) -> Value {
        match self {
            Distribution::DiscreteGaussian { stddev, mean, n } => {
                let mut o = serde_json::Map::new();
                o.insert("name".into(), json!("DiscreteGaussian"));
                o.insert("stddev".into(), json!(stddev));
                if let Some(m) = mean {
                    o.insert("mean".into(), json!(m));
                }
                if let Some(nn) = n {
                    o.insert("n".into(), json!(nn));
                }
                Value::Object(o)
            }
            Distribution::DiscreteGaussianAlpha { alpha, mean, n } => {
                let mut o = serde_json::Map::new();
                o.insert("name".into(), json!("DiscreteGaussianAlpha"));
                o.insert("alpha".into(), json!(alpha));
                if let Some(m) = mean {
                    o.insert("mean".into(), json!(m));
                }
                if let Some(nn) = n {
                    o.insert("n".into(), json!(nn));
                }
                Value::Object(o)
            }
            Distribution::CenteredBinomial { eta, n } => {
                let mut o = serde_json::Map::new();
                o.insert("name".into(), json!("CenteredBinomial"));
                o.insert("eta".into(), json!(eta));
                if let Some(nn) = n {
                    o.insert("n".into(), json!(nn));
                }
                Value::Object(o)
            }
            Distribution::Uniform { a, b, n } => {
                let mut o = serde_json::Map::new();
                o.insert("name".into(), json!("Uniform"));
                o.insert("a".into(), json!(a));
                o.insert("b".into(), json!(b));
                if let Some(nn) = n {
                    o.insert("n".into(), json!(nn));
                }
                Value::Object(o)
            }
            Distribution::UniformMod { n } => {
                let mut o = serde_json::Map::new();
                o.insert("name".into(), json!("UniformMod"));
                if let Some(nn) = n {
                    o.insert("n".into(), json!(nn));
                }
                Value::Object(o)
            }
            Distribution::SparseTernary { p, m, n } => {
                let mut o = serde_json::Map::new();
                o.insert("name".into(), json!("SparseTernary"));
                o.insert("p".into(), json!(p));
                o.insert("m".into(), json!(m));
                if let Some(nn) = n {
                    o.insert("n".into(), json!(nn));
                }
                Value::Object(o)
            }
            Distribution::SparseBinary { hw, n } => {
                let mut o = serde_json::Map::new();
                o.insert("name".into(), json!("SparseBinary"));
                o.insert("hw".into(), json!(hw));
                if let Some(nn) = n {
                    o.insert("n".into(), json!(nn));
                }
                Value::Object(o)
            }
            Distribution::Binary => json!({ "name": "Binary" }),
            Distribution::Ternary => json!({ "name": "Ternary" }),
        }
    }

    /// Convert this distribution to a compact JSON string.
    fn to_json_string(&self) -> String {
        serde_json::to_string(&self.to_json_value()).expect("serialize distribution")
    }
}

/// Errors that can occur when invoking the CLI.
#[derive(thiserror::Error, Debug)]
pub enum EstimatorCliError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),

    #[error("lattice-estimator-cli exited with code {0:?}. stdout: {1} stderr: {2}")]
    NonZeroExit(Option<i32>, String, String), // (exit_code, stdout, stderr)

    #[error("parse int error: {0}")]
    ParseInt(#[from] std::num::ParseIntError),
}

/// Invoke `lattice-estimator-cli` and return the security parameter as u64.
/// - `cli_path`: path to the `lattice-estimator-cli` wrapper (or a filename on PATH).
/// - `ring_dim`, `q`: large integers represented as `BigUint`.
/// - `s_dist`, `e_dist`: noise distributions converted to JSON for the CLI.
/// - `m`: optional number of samples; omitted means CLI default.
/// - `exact`: if true, use exact estimation; otherwise rough.
pub fn run_lattice_estimator_cli_with_path(
    cli_path: impl AsRef<Path>,
    ring_dim: &BigUint,
    q: &BigUint,
    s_dist: &Distribution,
    e_dist: &Distribution,
    m: Option<&BigUint>,
    exact: bool,
) -> Result<u64, EstimatorCliError> {
    // Prepare string arguments so they can be passed as discrete argv entries.
    let ring_dim_s = ring_dim.to_str_radix(10);
    let q_s = q.to_str_radix(10);
    let s_json = s_dist.to_json_string();
    let e_json = e_dist.to_json_string();

    // Build the command line:
    // lattice-estimator-cli <ring_dim> <q> --s-dist <json> --e-dist <json> [--m <m>] [--exact]
    let mut cmd = Command::new(cli_path.as_ref());
    cmd.arg(&ring_dim_s).arg(&q_s).arg("--s-dist").arg(&s_json).arg("--e-dist").arg(&e_json);
    if let Some(m) = m {
        cmd.arg("--m").arg(m.to_string());
    }
    if exact {
        cmd.arg("--exact");
    }

    // Execute and capture output.
    let output = cmd.output()?;
    let stdout = String::from_utf8(output.stdout)?;
    let stderr = String::from_utf8(output.stderr)?;

    if !output.status.success() {
        return Err(EstimatorCliError::NonZeroExit(output.status.code(), stdout, stderr));
    }

    // The CLI may print logs; parse only the last (non-empty) line as integer.
    let last_line = stdout
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    let secpar: u64 = last_line.parse()?;
    Ok(secpar)
}

/// Convenience wrapper that relies on PATH to find `lattice-estimator-cli`.
pub fn run_lattice_estimator_cli(
    ring_dim: &BigUint,
    q: &BigUint,
    s_dist: &Distribution,
    e_dist: &Distribution,
    m: Option<&BigUint>,
    exact: bool,
) -> Result<u64, EstimatorCliError> {
    run_lattice_estimator_cli_with_path(
        "lattice-estimator-cli",
        ring_dim,
        q,
        s_dist,
        e_dist,
        m,
        exact,
    )
}
