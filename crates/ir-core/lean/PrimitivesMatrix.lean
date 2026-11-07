import PrimitivesReduction
import Mathlib.Data.Matrix.Basic

namespace Mxx.Primitives

abbrev ExactMatrix (q n rows columns : Nat) := ModMatrix q n rows columns

abbrev ErrorMatrix (n rows columns : Nat) := IntMatrix n rows columns

/-- A matrix over a modulus equal to `q`: a merged stage passes instance `i` the member
`i / d` of a family over different rings (`LoopInputMode::ZipQuotient`), whose modulus equals the
instance's for every `i` but not syntactically (`h` follows `a`, whose index proof it reads). -/
def castModulus {q q' n rows columns : Nat} (a : ExactMatrix q n rows columns) (h : q = q') :
    ExactMatrix q' n rows columns :=
  h ▸ a

/-- Equal matrices cast to equal matrices (whatever the proofs of the moduli's equality). -/
theorem castModulus_congr {q q' n rows columns : Nat} {a b : ExactMatrix q n rows columns}
    (h : a = b) (hq hq' : q = q') : castModulus a hq = castModulus b hq' := by
  subst h
  rfl

/-- A statement about every modulus holds for a cast matrix when it holds for the matrix. -/
theorem castModulus_transport {q q' n rows columns : Nat} (hq : q = q')
    (P : (q : Nat) → ExactMatrix q n rows columns → Prop) {a : ExactMatrix q n rows columns}
    (h : P q a) : P q' (castModulus a hq) := by
  subst hq
  exact h

theorem exactMatrix_ext {q n rows columns : Nat} {a b : ExactMatrix q n rows columns}
    (h : ∀ row column, a row column = b row column) : a = b := by
  funext row column
  exact h row column

theorem errorMatrix_ext {n rows columns : Nat} {a b : ErrorMatrix n rows columns}
    (h : ∀ row column, a row column = b row column) : a = b := by
  funext row column
  exact h row column

end Mxx.Primitives
