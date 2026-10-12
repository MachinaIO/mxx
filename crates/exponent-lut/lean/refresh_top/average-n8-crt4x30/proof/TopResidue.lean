import TopRefresh

/-!
# Residues of the output

The decoder's modulus switch returns `ŷ mod q` of every coefficient `Q_q ŷ + ν` of its column with
`2 |ν| < Q_q`; the residue of a reduction modulo a multiple of `q` is the residue of the integer
polynomial.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh

theorem val_cast_dvd {Qd q : Nat} [NeZero Qd] (hdvd : q ∣ Qd) (a : Int) :
    ((((a : ZMod Qd).val : Nat) : Int) : ZMod q) = (a : ZMod q) := by
  rw [ZMod.val_intCast]
  push_cast
  rw [ZMod.intCast_eq_intCast_iff']
  exact (Int.emod_emod_of_dvd a (Int.natCast_dvd_natCast.mpr hdvd))

/-- The residue modulo `q` of a coefficient of a reduction modulo `Qd`, `q ∣ Qd`. -/
theorem residue_transfer {Qd Q q : Nat} (hQd : 1 < Qd) (hQ : 1 < Q) (hdQd : q ∣ Qd) (hdQ : q ∣ Q)
    {Y V : ErrorPoly N} (h : ∀ i, Y.coeff i ≡ V.coeff i [ZMOD q]) (i : Fin N) :
    (((((reducePoly Qd N Y).coeff i).val : Nat) : Int) : ZMod q) =
      (((((reducePoly Q N V).coeff i).val : Nat) : Int) : ZMod q) := by
  haveI : NeZero Qd := ⟨by omega⟩
  haveI : NeZero Q := ⟨by omega⟩
  rw [reducePoly_coeff_eq hQd hN, reducePoly_coeff_eq hQ hN, val_cast_dvd hdQd, val_cast_dvd hdQ]
  exact (ZMod.intCast_eq_intCast_iff _ _ _).mpr (h i)

/-- The decoder's modulus switch of a row whose columns are `Q_q y + ν`. -/
theorem switch_residue {Qd qc Qq : Nat} (hQ : Qd = Qq * qc) (hQq : 0 < Qq) (hqc : 1 < qc)
    {input : ExactMatrix Qd N 1 6} {out : ExactMatrix qc N 1 6} (h : modulusSwitchRuns input out)
    {y : ExactMatrix Qd N 1 6} {ν : Fin 6 → ErrorPoly N}
    (hcol : ∀ k, input 0 k = ((Qq : Int) : ExactPoly Qd N) * y 0 k + reducePoly Qd N (ν k))
    (hν : ∀ k i, 2 * |(ν k).coeff i| < Qq) (k : Fin 6) (i : Fin N) :
    (out 0 k).coeff i = ((((y 0 k).coeff i).val : Nat) : ZMod qc) := by
  have hQd : 1 < Qd := by rw [hQ]; nlinarith
  haveI : NeZero Qd := ⟨by omega⟩
  haveI : Fact (1 < Qd) := ⟨hQd⟩
  have hr := modulusSwitch_residue hN hQq hQ input out h 0 k i
    (((y 0 k).coeff i).val : Int) ((ν k).coeff i) ?_ (hν k i)
  · rw [hr]
    push_cast
    rfl
  · apply (ZMod.intCast_eq_intCast_iff' _ _ _).mp
    rw [hcol k]
    simp only [Int.ofNat_eq_natCast]
    have hc : ((((Qq : Int) : ExactPoly Qd N) * y 0 k + reducePoly Qd N (ν k)).coeff i) =
        (Qq : ZMod Qd) * (y 0 k).coeff i + ((ν k).coeff i : ZMod Qd) := by
      rw [Negacyclic.coeff_add, reducePoly_coeff_eq hQd hN]
      congr 1
      rw [intCast_eq_algebraMap, Negacyclic.coeff_smul]
      push_cast
      rfl
    rw [ZMod.natCast_val, ZMod.intCast_cast, ZMod.cast_id', id, hc]
    push_cast
    rw [ZMod.natCast_zmod_val]

end RefreshTopProof
