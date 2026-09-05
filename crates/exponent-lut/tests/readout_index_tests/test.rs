use super::helper_sigma_for_flat_index;
use mxx_ir_core::{IntExpr, ParamEnv};

#[test]
fn test_readout_helper_sigma_visits_every_flat_label_coordinate() {
    for sigmas in [vec![3], vec![3, 5, 7], vec![3, 5, 7, 9, 11, 13, 15]] {
        for label in 0..4 {
            for (helper, sigma) in sigmas.iter().enumerate() {
                let flat = label * sigmas.len() + helper;
                let selected = helper_sigma_for_flat_index(
                    IntExpr::constant(flat),
                    sigmas.iter().copied().map(IntExpr::constant).collect(),
                );
                assert_eq!(selected.evaluate(&ParamEnv::default()).unwrap(), (*sigma).into());
            }
        }
    }
}
