import W_Claim
import ExponentLut
import ExponentDecode
import ExponentPublic
import ExponentRuntime
import ExponentGadget
import ExponentIterate

/-!
# The parameter set

The pool primes `p₀ … p₆` (`p₀ … p₃` the primes of `Q₀`, `p₄ p₅ p₆` the extension), and the
moduli of the generated claim. The children of every part run on the prefix rings
`Q7 ⊃ Q6 ⊃ Q5 ⊃ Q4 = Q₀`; chain `c` decodes at `D_c` (three primes of `Q₀`, keeping `q_c`) with
plaintext modulus `M_c` (two primes) and outputs residues modulo `q_c`. Every ring lists its primes
in the canonical order, and the backend gives every modulus its regular gadget layout.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

abbrev N : Nat := 8
abbrev q0 : Nat := 1073741441
abbrev q1 : Nat := 1073741329
abbrev q2 : Nat := 1073740609
abbrev q3 : Nat := 1073740529
abbrev p4 : Nat := 1073740177
abbrev p5 : Nat := 1073740049
abbrev p6 : Nat := 1073739937
abbrev Q7 : Nat := 1645491229252209031554101362188915932898173126614280606821001729
abbrev Q6 : Nat := 1532485821333670884548742795052537878078547363014104417
abbrev Q5 : Nat := 1427241000054819492486624008799114726956177233
abbrev Q0 : Nat := 1329223801648636169536407324729467329
/-- `Q₁ = q₀ q₁ q₂`, the decoding modulus of chains `0`, `1`, `2` and the gadget of `G^st₀`. -/
abbrev Q1 : Nat := 1237937626222020133447349201
/-- The plaintext moduli `q₀ q₁` (chains `0`, `1`), `q₀ q₂` (chain `2`), `q₀ q₃` (chain `3`). -/
abbrev M01 : Nat := 1152920561861715089
/-- `D₃ = q₀ q₁ q₃`, the decoding modulus of chain `3`. -/
abbrev D3 : Nat := 1237937533988375184510142081
abbrev M02 : Nat := 1152919788767877569
abbrev M03 : Nat := 1152919702868562289
theorem hN : 0 < N := by decide
theorem hN1 : 1 < N := by decide
theorem hpow : N = 2 ^ 3 := rfl

theorem q7_eq : Q7 = p6 * Q6 := by norm_num
theorem q6_eq : Q6 = p5 * Q5 := by norm_num
theorem q5_eq : Q5 = p4 * Q0 := by norm_num
theorem q0_eq : Q0 = q3 * Q1 := by norm_num

theorem layoutM03 : Backend.backend.regularLayout M03 N = some Backend.layout0 := by
  simp [Backend.backend]
theorem layoutM02 : Backend.backend.regularLayout M02 N = some Backend.layout1 := by
  simp [Backend.backend]
theorem layoutM01 : Backend.backend.regularLayout M01 N = some Backend.layout2 := by
  simp [Backend.backend]
theorem layoutD3 : Backend.backend.regularLayout D3 N = some Backend.layout3 := by
  simp [Backend.backend]
theorem layoutQ1 : Backend.backend.regularLayout Q1 N = some Backend.layout4 := by
  simp [Backend.backend]
theorem layoutQ0 : Backend.backend.regularLayout Q0 N = some Backend.layout5 := by
  simp [Backend.backend]
theorem layoutQ5 : Backend.backend.regularLayout Q5 N = some Backend.layout6 := by
  simp [Backend.backend]
theorem layoutQ6 : Backend.backend.regularLayout Q6 N = some Backend.layout7 := by
  simp [Backend.backend]
theorem layoutQ7 : Backend.backend.regularLayout Q7 N = some Backend.layout8 := by
  simp [Backend.backend]

theorem matrixPolynomial_single {q : Nat} (k : Int) :
    (matrixPolynomial [k] : ExactMatrix q N 1 1) 0 0 = (k : ExactPoly q N) := by
  simp [matrixPolynomial]

/-- The centered rebase of a small integer polynomial is its reduction. -/
theorem rebase_small {q p : Nat} (hq : 2 < q) {s : ErrorPoly N} (hs : polyNorm s ≤ 1)
    {out : ExactMatrix p N 1 1} (h : centeredRebaseRuns (fun _ _ ↦ reducePoly q N s) out) :
    out = fun _ _ ↦ reducePoly p N s := by
  rw [centeredRebaseRuns_reduce h]
  funext r c
  rw [centeredIntLift_reducePoly (by omega) hN _ fun i ↦ by
    have := coeff_natAbs_le_polyNorm s i
    omega]

/-- The centered rebase of one is one. -/
theorem rebase_one {q p : Nat} (hq : 2 < q) {out : ExactMatrix p N 1 1}
    (h : centeredRebaseRuns (1 : ExactMatrix q N 1 1) out) : out = 1 := by
  have h1 : (1 : ExactMatrix q N 1 1) = fun _ _ ↦ reducePoly q N 1 := by
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  rw [h1] at h
  rw [rebase_small hq (by simpa using polyNorm_Xe (n := N) hN1 0) h]
  funext r c
  obtain rfl : r = 0 := Subsingleton.elim _ _
  obtain rfl : c = 0 := Subsingleton.elim _ _
  simp

end RefreshTopProof
