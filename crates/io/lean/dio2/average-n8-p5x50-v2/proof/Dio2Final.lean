import GenFinal

/-!
# The output decoder of the last step

The last step unmasks its state row re-targeted to the scalar form `Q_q` of the first prime `q`:
`ζ = b P - (smudge + x) - H`, where `x` is the re-targeted state component, then switches `ζ`
to `q` (`switch_gen`). This is refresh_top's `unmask_gen` without the fresh row and with the sum
in the graph's order.
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof
  RefreshTopProof.GenFinal

variable {n : Nat}

/-- **The output unmasking.** The hint `b = s B + e_b`, the preimage `P` with
`B_k P = K_x k + K_s k`, the re-targeted state `x` (target `Q_q Y`) and the smudge (target `T`,
`T - H = M` small): `b P - (smudge + x) - H = Q_q Y + ν` with `ν = e_b P_d - e_x - e_s + M`. -/
theorem unmask_out_gen (hn : 0 < n) {R h m : Nat} {s : ExactPoly R n} {Qq : Int}
    {b : ExactMatrix R n 1 h} {Bk : Fin 2 → ExactMatrix R n 1 h} {eb : ErrorMatrix n 1 h} {Beb : Nat}
    (hb : b = maskPart s Bk + reduceMatrix R n 1 h eb) (heb : CoeffBound eb Beb)
    {P : ExactMatrix R n h m} {Pd : ErrorMatrix n h m} {BP : Nat}
    (hP : P = reduceMatrix R n h m Pd) (hPd : CoeffBound Pd BP)
    {Kx Ks : Fin 2 → ExactMatrix R n 1 m} (hpre : ∀ k, Bk k * P = Kx k + Ks k)
    {x Y : ExactMatrix R n 1 m} {ex : ErrorMatrix n 1 m} {Bx : Nat}
    (hx : x = maskPart s Kx - ((Qq : Int) : ExactPoly R n) • Y + reduceMatrix R n 1 m ex)
    (hex : CoeffBound ex Bx)
    {smudge T H : ExactMatrix R n 1 m} {es : ErrorMatrix n 1 m} {Bs : Nat}
    (hs : smudge = maskPart s Ks - T + reduceMatrix R n 1 m es) (hes : CoeffBound es Bs)
    {M : ErrorMatrix n 1 m} {BM : Nat} (hTH : T - H = reduceMatrix R n 1 m M) (hM : CoeffBound M BM) :
    ∃ ν : ErrorMatrix n 1 m, CoeffBound ν (h * n * Beb * BP + Bx + Bs + BM) ∧
      matrixSub (matrixSub (matrixMul b P) (matrixAdd smudge x)) H =
        ((Qq : Int) : ExactPoly R n) • Y + reduceMatrix R n 1 m ν := by
  refine ⟨eb * Pd - ex - es + M,
    coeffBound_add (coeffBound_sub (coeffBound_sub (coeffBound_mul hn heb hPd) hex) hes) hM, ?_⟩
  have hT : T = H + reduceMatrix R n 1 m M := by rw [← hTH]; abel
  simp only [matrixSub, matrixAdd, matrixMul]
  rw [reduceMatrix_add, reduceMatrix_sub, reduceMatrix_sub, reduceMatrix_mul,
    ← hP, hb, hx, hs, hT, Matrix.add_mul, maskPart_mul]
  simp only [hpre, maskPart, smul_add]
  module

/-- **The output decoder**: `ζ = b P - (smudge + x) - H` switched from `R = Q_q q` to `q` holds
the residues of `Y` when the noise stays below `Q_q / 2`. -/
theorem out_gen (hn : 0 < n) {R q Qq h m : Nat} (hR : R = Qq * q) (hQq : 0 < Qq) (hq : 1 < q)
    {s : ExactPoly R n}
    {b : ExactMatrix R n 1 h} {Bk : Fin 2 → ExactMatrix R n 1 h} {eb : ErrorMatrix n 1 h} {Beb : Nat}
    (hb : b = maskPart s Bk + reduceMatrix R n 1 h eb) (heb : CoeffBound eb Beb)
    {P : ExactMatrix R n h m} {Pd : ErrorMatrix n h m} {BP : Nat}
    (hP : P = reduceMatrix R n h m Pd) (hPd : CoeffBound Pd BP)
    {Kx Ks : Fin 2 → ExactMatrix R n 1 m} (hpre : ∀ k, Bk k * P = Kx k + Ks k)
    {x Y : ExactMatrix R n 1 m} {ex : ErrorMatrix n 1 m} {Bx : Nat}
    (hx : x = maskPart s Kx - ((Qq : Int) : ExactPoly R n) • Y + reduceMatrix R n 1 m ex)
    (hex : CoeffBound ex Bx)
    {smudge T H : ExactMatrix R n 1 m} {es : ErrorMatrix n 1 m} {Bs : Nat}
    (hs : smudge = maskPart s Ks - T + reduceMatrix R n 1 m es) (hes : CoeffBound es Bs)
    {M : ErrorMatrix n 1 m} {BM : Nat} (hTH : T - H = reduceMatrix R n 1 m M) (hM : CoeffBound M BM)
    (hnoise : 2 * (h * n * Beb * BP + Bx + Bs + BM) < Qq)
    {ζ : ExactMatrix R n 1 m}
    (hζ : ζ = matrixSub (matrixSub (matrixMul b P) (matrixAdd smudge x)) H)
    {r : ExactMatrix q n 1 m} (hsw : modulusSwitchRuns ζ r) (c : Fin m) (i : Fin n) :
    (r 0 c).coeff i = ((((Y 0 c).coeff i).val : Nat) : ZMod q) := by
  obtain ⟨ν, hν, hval⟩ := unmask_out_gen hn hb heb hP hPd hpre hx hex hs hes hTH hM
  refine switch_gen hn hR hQq hq hsw (hζ.trans hval) (fun c i ↦ ?_) c i
  have := hν 0 c i
  rw [Int.abs_eq_natAbs]
  omega

end Dio2Proof
