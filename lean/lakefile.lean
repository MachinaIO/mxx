import Lake
open Lake DSL

package mxx where
  leanOptions := #[
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩
  ]

require "leanprover-community" / "mathlib" @ git "v4.32.0"
require VCVio from git
  "https://github.com/MachinaIO/VCVio.git" @
  "afb5b6c0d49041944db180f47b58afc2c6dca92b"

@[default_target] lean_lib Mxx

lean_lib MxxCorrectness where
  srcDir := "../crates/correctness/lean"

lean_lib MxxWe where
  srcDir := "../crates/we/lean"

lean_exe mxx_diamond_checker where
  root := `MxxWe.DiamondChecker
  srcDir := "../crates/we/lean"
