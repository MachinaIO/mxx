import ExponentReadout
import ExponentSetup

namespace MxxExponentLut

open Mxx.Primitives
open scoped BigOperators

/-- Odd substitution respects the defining negacyclic relation. -/
noncomputable def readoutAutomorphismHom {q n : Nat} (index : Nat) (hodd : Odd index) :
    ExactPoly q n →+* ExactPoly q n :=
  AdjoinRoot.lift (algebraMap (ZMod q) (ExactPoly q n))
    (AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ index) (by
      have hpower : (AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ index) ^ n = -1 := by
        rw [← pow_mul, Nat.mul_comm index n, pow_mul, Negacyclic.root_pow_n,
          hodd.neg_one_pow]
      simpa [negacyclicModulus, Polynomial.eval₂_add, Polynomial.eval₂_pow] using
        congrArg (fun x : ExactPoly q n => x + 1) hpower)

/-- The quotient-ring homomorphism is the actual canonical-coefficient runtime
operation, rather than an independent interpretation of automorphism. -/
theorem readoutAutomorphismHom_runtime {q n rows columns : Nat}
    (hq : 1 < q) (hn : 0 < n) (index : Nat) (hodd : Odd index)
    (input : ExactMatrix q n rows columns) :
    MxxRuntime.ringAutomorphism index input =
      fun row column => readoutAutomorphismHom index hodd (input row column) := by
  letI : Fact (1 < q) := ⟨hq⟩
  letI : NeZero q := ⟨by omega⟩
  ext row column
  conv_rhs => rw [Negacyclic.expansion hn (input row column)]
  simp only [MxxRuntime.ringAutomorphism, map_sum, map_mul, map_pow]
  apply Finset.sum_congr rfl
  intro i _
  have hcast : (Int.ofNat ((input row column).coeff i).val : ExactPoly q n) =
      algebraMap (ZMod q) (ExactPoly q n) ((input row column).coeff i) := by
    have h := congrArg (algebraMap (ZMod q) (ExactPoly q n))
      (ZMod.natCast_zmod_val ((input row column).coeff i))
    simpa only [map_natCast, Int.ofNat_eq_natCast, Int.cast_natCast] using h
  rw [hcast]
  simp [readoutAutomorphismHom, ← pow_mul, Nat.mul_comm]

theorem ringAutomorphism_mul {q n rows inner columns : Nat}
    (hq : 1 < q) (hn : 0 < n) (index : Nat) (hodd : Odd index)
    (left : ExactMatrix q n rows inner) (right : ExactMatrix q n inner columns) :
    MxxRuntime.ringAutomorphism index (left * right) =
      MxxRuntime.ringAutomorphism index left * MxxRuntime.ringAutomorphism index right := by
  simp only [readoutAutomorphismHom_runtime hq hn index hodd]
  ext row column
  simp only [Matrix.mul_apply, map_sum, map_mul]

theorem ringAutomorphism_sub {q n rows columns : Nat}
    (hq : 1 < q) (hn : 0 < n) (index : Nat) (hodd : Odd index)
    (left right : ExactMatrix q n rows columns) :
    MxxRuntime.ringAutomorphism index (left - right) =
      MxxRuntime.ringAutomorphism index left - MxxRuntime.ringAutomorphism index right := by
  simp only [readoutAutomorphismHom_runtime hq hn index hodd]
  ext row column
  exact map_sub _ _ _

theorem ringAutomorphism_polynomial_action {q n rows columns : Nat}
    (hq : 1 < q) (hn : 0 < n) (index : Nat) (hodd : Odd index)
    (scalar : ExactPoly q n) (matrix : ExactMatrix q n rows columns) :
    MxxRuntime.ringAutomorphism index (scalar • matrix) =
      readoutAutomorphismHom index hodd scalar • MxxRuntime.ringAutomorphism index matrix := by
  simp only [readoutAutomorphismHom_runtime hq hn index hodd]
  ext row column
  exact map_mul _ _ _

/-- Integer gadget entries are fixed by the actual runtime automorphism. -/
theorem ringAutomorphism_integer_constants {q n rows columns : Nat}
    (hq : 1 < q) (hn : 0 < n) (index : Nat) (hodd : Odd index)
    (entries : Matrix (Fin rows) (Fin columns) Int) :
    MxxRuntime.ringAutomorphism index (fun r c => (entries r c : ExactPoly q n)) =
      (fun r c => (entries r c : ExactPoly q n)) := by
  rw [readoutAutomorphismHom_runtime hq hn index hodd]
  ext r c
  exact map_intCast _ _

/-- The sampled switch error appears with a minus sign because the native
encoding subtracts its payload times the setup key. Premises are the two
cache reconstruction equations, rather than the resulting setup equation. -/
theorem readout_setup_of_cache_equations {q n rows key columns : Nat}
    (hq : 1 < q) (hn : 0 < n) (index : Nat) (hodd : Odd index)
    (secret setup : ExactMatrix q n rows key)
    (publicKey maskKey gadget : ExactMatrix q n key columns)
    (action helperAction : ErrorMatrix n columns columns)
    (switchError : ErrorMatrix n rows columns)
    (payload : ExactPoly q n) (message : ErrorPoly n)
    (hgadget : MxxRuntime.ringAutomorphism index gadget = gadget)
    (hswitch : MxxRuntime.ringAutomorphism index setup * gadget *
        reduceMatrix q n columns columns action =
      setup * gadget + reduceMatrix q n rows columns switchError)
    (halign : gadget * reduceMatrix q n columns columns helperAction =
      MxxRuntime.ringAutomorphism index publicKey * reduceMatrix q n columns columns action)
    (hmessage : readoutAutomorphismHom index hodd payload = reducePoly q n message) :
    MxxRuntime.ringAutomorphism index (secret * publicKey - payload • (setup * gadget)) *
        reduceMatrix q n columns columns action +
      (secret * maskKey - MxxRuntime.ringAutomorphism index secret * gadget) *
        reduceMatrix q n columns columns helperAction =
      (secret * (maskKey * reduceMatrix q n columns columns helperAction) -
        readoutAutomorphismHom index hodd payload • (setup * gadget)) +
      reduceMatrix q n rows columns ((-message) • switchError) := by
  rw [ringAutomorphism_sub hq hn index hodd,
    ringAutomorphism_mul hq hn index hodd,
    ringAutomorphism_polynomial_action hq hn index hodd,
    ringAutomorphism_mul hq hn index hodd, hgadget]
  rw [Matrix.sub_mul, Matrix.sub_mul, Matrix.smul_mul, hswitch, smul_add]
  rw [Matrix.mul_assoc secret maskKey, Matrix.mul_assoc _ gadget, halign,
    ← Matrix.mul_assoc]
  rw [reduce_polynomial_action, map_neg, neg_smul, ← hmessage]
  simp only [Matrix.mul_assoc]
  abel

@[simp] theorem polynomialL1_neg {n : Nat} (message : ErrorPoly n) :
    polynomialL1 (-message) = polynomialL1 message := by
  simp [polynomialL1, Negacyclic.coeff_neg]

end MxxExponentLut
