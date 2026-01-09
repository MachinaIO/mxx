use num_bigint::BigUint;
use serde_json::{Value, json};
use std::{path::Path, process::Command, result::Result};
/// Enum describing supported noise distributions.
/// This mirrors the JSON spec expected by the Python CLI.
#[derive(Debug, Clone)]
pub enum Distribution {
    DiscreteGaussian { stddev: String, mean: Option<f64>, n: Option<u64> },
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
    let last_line = stdout.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
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

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::{__PAIR, __TestState};
    use num_bigint::BigUint;
    use std::{fs, os::unix::fs::PermissionsExt};
    use tempfile::TempDir;

    // Helper to create a mock CLI executable for unit tests.
    fn create_mock_cli(dir: &TempDir, output: &str, exit_code: i32) -> std::path::PathBuf {
        let mock_path = dir.path().join("mock-lattice-estimator-cli");
        let script_content = format!(
            r#"#!/bin/sh
echo "{}"
exit {}
"#,
            output, exit_code
        );
        fs::write(&mock_path, script_content).expect("write mock script");

        // Make it executable.
        let mut perms = fs::metadata(&mock_path).expect("get metadata").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&mock_path, perms).expect("set permissions");

        mock_path
    }

    #[test]
    #[sequential_test::sequential]
    fn test_run_lattice_estimator_cli_success() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let mock_path = create_mock_cli(&temp_dir, "128", 0);

        let ring_dim = BigUint::from(2048u32);
        let q = BigUint::from(3329u32);
        let s_dist = Distribution::CenteredBinomial { eta: 2, n: None };
        let e_dist = Distribution::CenteredBinomial { eta: 2, n: None };

        let result = run_lattice_estimator_cli_with_path(
            mock_path, &ring_dim, &q, &s_dist, &e_dist, None, false,
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 128);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_run_lattice_estimator_cli_with_logs() {
        let temp_dir = TempDir::new().expect("create temp dir");
        // Simulate CLI output with logs before the actual result.
        let mock_path = create_mock_cli(
            &temp_dir,
            "INFO: Starting estimation...\nDEBUG: Parameters loaded\n256",
            0,
        );

        let ring_dim = BigUint::from(4096u32);
        let q = BigUint::from(8192u32);
        let s_dist =
            Distribution::DiscreteGaussian { stddev: 3.2.to_string(), mean: None, n: None };
        let e_dist =
            Distribution::DiscreteGaussian { stddev: 3.2.to_string(), mean: None, n: None };

        let result = run_lattice_estimator_cli_with_path(
            mock_path, &ring_dim, &q, &s_dist, &e_dist, None, false,
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 256);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_run_lattice_estimator_cli_with_exact_and_m() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let mock_path = create_mock_cli(&temp_dir, "512", 0);

        let ring_dim = BigUint::from(8192u32);
        let q = BigUint::from(16384u32);
        let s_dist = Distribution::Uniform { a: -1, b: 1, n: None };
        let e_dist = Distribution::Uniform { a: -1, b: 1, n: None };
        let m = BigUint::from(1000u32);

        let result = run_lattice_estimator_cli_with_path(
            mock_path,
            &ring_dim,
            &q,
            &s_dist,
            &e_dist,
            Some(&m),
            true,
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 512);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_run_lattice_estimator_cli_non_zero_exit() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let mock_path = create_mock_cli(&temp_dir, "Error: Invalid parameters", 1);

        let ring_dim = BigUint::from(1024u32);
        let q = BigUint::from(2048u32);
        let s_dist = Distribution::Binary;
        let e_dist = Distribution::Binary;

        let result = run_lattice_estimator_cli_with_path(
            mock_path, &ring_dim, &q, &s_dist, &e_dist, None, false,
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            EstimatorCliError::NonZeroExit(code, stdout, _stderr) => {
                assert_eq!(code, Some(1));
                assert!(stdout.contains("Error: Invalid parameters"));
            }
            _ => panic!("Expected NonZeroExit error"),
        }
    }

    #[test]
    #[sequential_test::sequential]
    fn test_run_lattice_estimator_cli_parse_error() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let mock_path = create_mock_cli(&temp_dir, "not_a_number", 0);

        let ring_dim = BigUint::from(1024u32);
        let q = BigUint::from(2048u32);
        let s_dist = Distribution::Ternary;
        let e_dist = Distribution::Ternary;

        let result = run_lattice_estimator_cli_with_path(
            mock_path, &ring_dim, &q, &s_dist, &e_dist, None, false,
        );

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EstimatorCliError::ParseInt(_)));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_run_lattice_estimator_cli_empty_output() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let mock_path = create_mock_cli(&temp_dir, "", 0);

        let ring_dim = BigUint::from(2048u32);
        let q = BigUint::from(3329u32);
        let s_dist = Distribution::SparseTernary { p: 64, m: 128, n: None };
        let e_dist = Distribution::SparseTernary { p: 64, m: 128, n: None };

        let result = run_lattice_estimator_cli_with_path(
            mock_path, &ring_dim, &q, &s_dist, &e_dist, None, false,
        );

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EstimatorCliError::ParseInt(_)));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_run_lattice_estimator_cli_whitespace_output() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let mock_path = create_mock_cli(&temp_dir, "  \n  192  \n  ", 0);

        let ring_dim = BigUint::from(3072u32);
        let q = BigUint::from(8192u32);
        let s_dist = Distribution::SparseBinary { hw: 64, n: None };
        let e_dist = Distribution::SparseBinary { hw: 64, n: None };

        let result = run_lattice_estimator_cli_with_path(
            mock_path, &ring_dim, &q, &s_dist, &e_dist, None, false,
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 192);
    }

    #[test]
    #[sequential_test::sequential]
    fn test_distribution_to_json_string() {
        // Test various distribution types serialize correctly.
        let dist = Distribution::DiscreteGaussian {
            stddev: 3.19.to_string(),
            mean: Some(0.0),
            n: Some(256),
        };
        let json = dist.to_json_string();
        assert!(json.contains(r#""name":"DiscreteGaussian""#));
        assert!(json.contains(r#""stddev":"3.19""#));
        assert!(json.contains(r#""mean":0.0"#));
        assert!(json.contains(r#""n":256"#));

        let dist = Distribution::CenteredBinomial { eta: 3, n: None };
        let json = dist.to_json_string();
        assert!(json.contains(r#""name":"CenteredBinomial""#));
        assert!(json.contains(r#""eta":3"#));
        assert!(!json.contains(r#""n""#));

        let dist = Distribution::UniformMod { n: Some(512) };
        let json = dist.to_json_string();
        assert!(json.contains(r#""name":"UniformMod""#));
        assert!(json.contains(r#""n":512"#));

        let dist = Distribution::Binary;
        let json = dist.to_json_string();
        assert_eq!(json, r#"{"name":"Binary"}"#);
    }

    // Integration test with actual CLI (requires lattice-estimator-cli in PATH).
    #[test]
    #[sequential_test::sequential]
    #[ignore] // Use `cargo test -- --ignored` to run this test.
    fn test_integration_with_actual_cli() {
        let ring_dim = BigUint::from(1024u32);
        let q = BigUint::from(12289u32);
        let s_dist = Distribution::Binary;
        let e_dist =
            Distribution::DiscreteGaussian { stddev: 3.2.to_string(), mean: None, n: None };
        let m = BigUint::from(100000u32);

        // Test with exact mode.
        let result = run_lattice_estimator_cli(&ring_dim, &q, &s_dist, &e_dist, Some(&m), true);

        match result {
            Ok(secpar) => {
                println!("Security parameter (exact): {}", secpar);
                assert!(secpar > 0);
            }
            Err(e) => {
                eprintln!("CLI not found or error: {}", e);
                panic!("Integration test failed");
            }
        }

        // Test without exact mode.
        let result_rough = run_lattice_estimator_cli(&ring_dim, &q, &s_dist, &e_dist, None, false);

        match result_rough {
            Ok(secpar) => {
                println!("Security parameter (rough): {}", secpar);
                assert!(secpar > 0);
            }
            Err(e) => {
                eprintln!("CLI not found or error: {}", e);
                panic!("Integration test failed");
            }
        }
    }

    // Test with custom path to CLI.
    #[test]
    #[sequential_test::sequential]
    #[ignore] // Use `cargo test -- --ignored` to run this test.
    fn test_integration_with_custom_cli_path() {
        let cli_path =
            "/Users/piapark/Documents/GitHub/lattice-estimator-cli/scripts/lattice-estimator-cli";
        let ring_dim = BigUint::from(2048u32);
        let q = BigUint::from(65537u32);
        let s_dist = Distribution::CenteredBinomial { eta: 2, n: None };
        let e_dist = Distribution::CenteredBinomial { eta: 2, n: None };

        let result = run_lattice_estimator_cli_with_path(
            cli_path, &ring_dim, &q, &s_dist, &e_dist, None, false,
        );

        match result {
            Ok(secpar) => {
                println!("Security parameter: {}", secpar);
                assert!(secpar > 0);
            }
            Err(e) => {
                eprintln!("CLI not found at custom path or error: {}", e);
                panic!("Integration test failed");
            }
        }
    }
}
