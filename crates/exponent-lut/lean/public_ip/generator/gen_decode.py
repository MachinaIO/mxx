"""Write `PublicIpDecode.lean` of a public inner-product parameter set from its exported claim.

The decode stage threshold-decodes every coefficient of every column of both residual components
and sums the decoded integers. The proof shows that both residuals are the output encoding's error
(`residuals`), that every decoded coefficient of a slice of a small residual is zero
(`decode_zero`), and then applies `decode_zero` once per `thresholdDecode` of the stage, in the
stage's order, before summing.

Usage: python3 generator/gen_decode.py <parameter id>
"""

import re
import sys
from pathlib import Path

from patterns import pattern

HEADER = '''import PublicIpOnlineEval

/-!
# The decode stage

The decode stage subtracts the mask part `s₀ A₀ⱼ + A₁ⱼ` from each row component of the output and
adds back the claimed plaintext `μ` times `sⱼ G`. For the plaintext of the output encoding, both
residuals are its error, so every coefficient of every column threshold-decodes to zero when the
error is below a quarter of `Q`, and the decoded residual count is zero.
-/

namespace PublicIpProof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {tape : SampleTape} {path : List Nat}

set_option maxHeartbeats 4000000 in
/-- Both residuals of a public encoding of `μ` against `μ` are its error. -/
theorem residuals {sm μ : ErrorPoly N} {A00 A01 A10 A11 c0 c1 G : ExactMatrix Q N 1 4} {B : Nat}
    (hrel : Rel sm ![![A00, A01], ![A10, A11]] (fun j ↦ reducePoly Q N (μ * unitPair sm j) • G)
      ![c0, c1] B)
    {mask message : ExactMatrix Q N 1 1} (hmask : mask = fun _ _ ↦ reducePoly Q N sm)
    (hmessage : message = fun _ _ ↦ reducePoly Q N μ) :
    ∃ e : Fin 2 → ErrorMatrix N 1 4, (∀ j, CoeffBound (e j) B) ∧
      matrixAdd (matrixSub c0 (matrixAdd (matrixMulScalarLeft mask A00)
          (matrixMulScalarLeft (1 : ExactMatrix Q N 1 1) A10)))
        (matrixMulScalarLeft mask (matrixMulScalarLeft message G)) = reduceMatrix Q N 1 4 (e 0) ∧
      matrixAdd (matrixSub c1 (matrixAdd (matrixMulScalarLeft mask A01)
          (matrixMulScalarLeft (1 : ExactMatrix Q N 1 1) A11)))
        (matrixMulScalarLeft (1 : ExactMatrix Q N 1 1) (matrixMulScalarLeft message G)) =
          reduceMatrix Q N 1 4 (e 1) := by
  obtain ⟨e, he, hc⟩ := hrel
  refine ⟨e, he, ?_, ?_⟩
  · have h0 : c0 = reducePoly Q N sm • A00 + A10 - reducePoly Q N (μ * sm) • G +
        reduceMatrix Q N 1 4 (e 0) := hc 0
    rw [h0, hmask, hmessage]
    funext r c
    simp only [matrixAdd, matrixSub, matrixMulScalarLeft, Matrix.add_apply, Matrix.sub_apply,
      Matrix.smul_apply, smul_eq_mul, Matrix.one_apply_eq, one_mul, map_mul]
    ring
  · have h1 : c1 = reducePoly Q N sm • A01 + A11 - reducePoly Q N (μ * 1) • G +
        reduceMatrix Q N 1 4 (e 1) := hc 1
    rw [h1, hmask, hmessage]
    funext r c
    simp only [matrixAdd, matrixSub, matrixMulScalarLeft, Matrix.add_apply, Matrix.sub_apply,
      Matrix.smul_apply, smul_eq_mul, Matrix.one_apply_eq, one_mul, mul_one]
    ring

/-- Every decoded coefficient of a column of a residual with error below a quarter of `Q` is zero. -/
theorem decode_zero {X : ExactMatrix Q N 1 4} {E : ErrorMatrix N 1 4} {B : Nat}
    (hX : X = reduceMatrix Q N 1 4 E) (hE : CoeffBound E B) (hB : 4 * B + 4 < Q)
    {r0 r1 c0 c1 : Int} {s : ExactMatrix Q N 1 1} (hs : sliceMatrix X r0 r1 c0 c1 s)
    {length position : Int} {out : Int} (hdecode : thresholdDecode 2 length position s out) :
    out = 0 := by
  obtain ⟨hr0, -, hr1, hc0, -, hc1, hrows, hcols, hval⟩ := hs
  obtain ⟨_, _, _, _, index, -, -, hout⟩ := hdecode
  simp only [Int.ofNat_eq_natCast] at hr1 hc1 hrows hcols
  obtain ⟨i, j, hij⟩ : ∃ i j, s 0 0 = reducePoly Q N (E i j) :=
    ⟨_, _, (hval 0 0 (by simp; omega) (by simp; omega)).trans (by rw [hX]; rfl)⟩
  have heabs : 4 * |(E i j).coeff index| + 4 < Q := by
    have := hE i j index
    rw [Int.abs_eq_natAbs]
    omega
  have hv := threshold_decode_value hQ hQodd (Or.inl rfl) heabs
  rw [mul_zero, neg_zero, zero_add] at hv
  rw [hout, hij, reducePoly_coeff_eq hQ hN]
  exact hv
'''


def main() -> None:
    parameter = sys.argv[1]
    root = Path(__file__).resolve().parent.parent / parameter
    stage = (root / "generated" / "Stage_decode.lean").read_text(encoding="utf-8")
    names = pattern(stage, "generatedRoot")
    # Which residual every slice reads, and which slice every decode reads.
    residual_of = {}
    for matrix, output in re.findall(r"MxxRuntime\.sliceMatrix (w_\d+_0) 0 1 \d+ \d+ w_(\d+)_0", stage):
        residual_of[output] = {"w_17_0": "hr0", "w_162_0": "hr1"}[matrix]
    decodes = re.findall(r"MxxRuntime\.thresholdDecode \(2\) \(32\) \d+ w_(\d+)_0 w_(\d+)_(\d+)", stage)
    lines = [HEADER]
    lines.append('''set_option maxRecDepth 100000 in
set_option maxHeartbeats 64000000 in
/-- The decode stage outputs zero for the output encoding of the claimed plaintext `c X^e`. -/
theorem decode_value {mask : ExactMatrix Q N 1 1} {A00 A01 A10 A11 c0 c1 G : ExactMatrix Q N 1 4}
    {ec ee : Int} {out : Int}
    (h : Stage_decode.generatedRoot Backend.backend tape path GeneratedClaim.stage_3_params
      (c0, mask, A00, A10, ec, ee, c1, A01, A11, ()) out)
    {sm : ErrorPoly N} (hmask : mask = fun _ _ ↦ reducePoly Q N sm)
    (hG : gadgetMatrixRuns Backend.backend 1048576 4 G) {B : Nat}
    (hrel : Rel sm ![![A00, A01], ![A10, A11]]
      (fun j ↦ reducePoly Q N (message (n := N) ec ee * unitPair sm j) • G) ![c0, c1] B)
    (hB : 4 * B + 4 < Q) : out = 0 := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_decode.generatedRoot.body at hb
  dsimp only at hb
  obtain ''' + "⟨" + ", ".join(names) + '''⟩ := hb
  have hgg := gadgetMatrixRuns_unique h14 hG
  rw [← hgg] at hrel
  obtain ⟨e, he, hr0, hr1⟩ := residuals hrel hmask (multiplyMonomial_liftInteger ec ee)''')
    zeros = []
    for slice_, wire, index in decodes:
        name = f"h{wire}" if index == "0" else f"h{wire}_{index}"
        component = residual_of[slice_]
        bound = "(he 0)" if component == "hr0" else "(he 1)"
        lines.append(f"  have z{wire}_{index} := decode_zero {component} {bound} hB h{slice_} {name}")
        zeros.append(f"z{wire}_{index}")
    lines.append("  rw [hout]")
    for start in range(0, len(zeros), 16):
        lines.append("  rw [" + ", ".join(zeros[start:start + 16]) + "]")
    lines.append("  norm_num")
    lines.append("")
    lines.append("end PublicIpProof")
    (root / "PublicIpDecode.lean").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
