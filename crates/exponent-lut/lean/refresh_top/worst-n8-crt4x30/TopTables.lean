import Stage_preprocess
import Stage_encrypt
import Stage_online_eval
import Mathlib.Algebra.Ring.Int.Defs

/-! The scalar constants of the instances of the merged stages (written by
`generator/gen_tables.py`). -/

namespace RefreshTopProof

theorem selectInt_nonneg {i : Int} {l : List Int} (h : ∀ x ∈ l, 0 ≤ x) :
    0 ≤ MxxIR.selectInt i l := by
  unfold MxxIR.selectInt
  cases hx : l[i.toNat]? with
  | none => simp
  | some x => exact h x (List.mem_of_getElem? hx)

/-- The prime of `Q₀` that the decoding ring `R` drops (`Q₀ = qd R`). (the table, over the instance index) -/
def qdZ (i : Int) : Int := MxxIR.selectInt i [1073740529, 1073740529, 1073740529, 1073740529, 1073740529, 1073740529, 1073740609, 1073740609]

@[simp] theorem qdZ_preprocess_sel_3 (i : Int) : Stage_preprocess.sel_3 i = qdZ i := rfl
@[simp] theorem qdZ_encrypt_sel_1 (i : Int) : Stage_encrypt.sel_1 i = qdZ i := rfl

/-- The prime of `Q₀` that the decoding ring `R` drops (`Q₀ = qd R`). -/
def qd (i0 : Nat) : Nat := (qdZ (i0 : Int)).toNat

@[simp] theorem qd_cast (i0 : Nat) : ((qd i0 : Nat) : Int) = qdZ (i0 : Int) :=
  Int.toNat_of_nonneg (selectInt_nonneg (by decide))

@[simp] theorem qd_cast_ring {S : Type*} [Ring S] (i0 : Nat) :
    ((qd i0 : Nat) : S) = ((qdZ (i0 : Int) : Int) : S) := by
  rw [← qd_cast, Int.cast_natCast]

/-- The third prime of the decoding ring (`R = q₀ q₁ rp`). (the table, over the instance index) -/
def rpZ (i : Int) : Int := MxxIR.selectInt i [1073740609, 1073740609, 1073740609, 1073740609, 1073740609, 1073740609, 1073740529, 1073740529]

@[simp] theorem rpZ_preprocess_sel_0 (i : Int) : Stage_preprocess.sel_0 i = rpZ i := rfl
@[simp] theorem rpZ_encrypt_sel_0 (i : Int) : Stage_encrypt.sel_0 i = rpZ i := rfl
@[simp] theorem rpZ_online_eval_sel_0 (i : Int) : Stage_online_eval.sel_0 i = rpZ i := rfl

/-- The third prime of the decoding ring (`R = q₀ q₁ rp`). -/
def rp (i0 : Nat) : Nat := (rpZ (i0 : Int)).toNat

@[simp] theorem rp_cast (i0 : Nat) : ((rp i0 : Nat) : Int) = rpZ (i0 : Int) :=
  Int.toNat_of_nonneg (selectInt_nonneg (by decide))

@[simp] theorem rp_cast_ring {S : Type*} [Ring S] (i0 : Nat) :
    ((rp i0 : Nat) : S) = ((rpZ (i0 : Int) : Int) : S) := by
  rw [← rp_cast, Int.cast_natCast]

/-- The prime `p` of `R = p Mx`. (the table, over the instance index) -/
def ppZ (i : Int) : Int := MxxIR.selectInt i [1073740609, 1073740609, 1073740609, 1073740609, 1073741329, 1073741329, 1073741329, 1073741329]

@[simp] theorem ppZ_preprocess_sel_8 (i : Int) : Stage_preprocess.sel_8 i = ppZ i := rfl
@[simp] theorem ppZ_encrypt_sel_3 (i : Int) : Stage_encrypt.sel_3 i = ppZ i := rfl

/-- The prime `p` of `R = p Mx`. -/
def pp (i0 : Nat) : Nat := (ppZ (i0 : Int)).toNat

@[simp] theorem pp_cast (i0 : Nat) : ((pp i0 : Nat) : Int) = ppZ (i0 : Int) :=
  Int.toNat_of_nonneg (selectInt_nonneg (by decide))

@[simp] theorem pp_cast_ring {S : Type*} [Ring S] (i0 : Nat) :
    ((pp i0 : Nat) : S) = ((ppZ (i0 : Int) : Int) : S) := by
  rw [← pp_cast, Int.cast_natCast]

/-- The second prime of the plaintext modulus (`Mx = q₀ mxp`). (the table, over the instance index) -/
def mxpZ (i : Int) : Int := MxxIR.selectInt i [1073741329, 1073741329, 1073741329, 1073741329, 1073740609, 1073740609, 1073740529, 1073740529]

@[simp] theorem mxpZ_preprocess_sel_7 (i : Int) : Stage_preprocess.sel_7 i = mxpZ i := rfl
@[simp] theorem mxpZ_encrypt_sel_2 (i : Int) : Stage_encrypt.sel_2 i = mxpZ i := rfl
@[simp] theorem mxpZ_online_eval_sel_6 (i : Int) : Stage_online_eval.sel_6 i = mxpZ i := rfl

/-- The second prime of the plaintext modulus (`Mx = q₀ mxp`). -/
def mxp (i0 : Nat) : Nat := (mxpZ (i0 : Int)).toNat

@[simp] theorem mxp_cast (i0 : Nat) : ((mxp i0 : Nat) : Int) = mxpZ (i0 : Int) :=
  Int.toNat_of_nonneg (selectInt_nonneg (by decide))

@[simp] theorem mxp_cast_ring {S : Type*} [Ring S] (i0 : Nat) :
    ((mxp i0 : Nat) : S) = ((mxpZ (i0 : Int) : Int) : S) := by
  rw [← mxp_cast, Int.cast_natCast]

/-- The CRT prime `q_c` of the chain. (the table, over the instance index) -/
def qcNZ (i : Int) : Int := MxxIR.selectInt i [1073741441, 1073741441, 1073741329, 1073741329, 1073740609, 1073740609, 1073740529, 1073740529]

@[simp] theorem qcNZ_preprocess_sel_10 (i : Int) : Stage_preprocess.sel_10 i = qcNZ i := rfl
@[simp] theorem qcNZ_online_eval_sel_5 (i : Int) : Stage_online_eval.sel_5 i = qcNZ i := rfl

/-- The CRT prime `q_c` of the chain. -/
def qcN (i0 : Nat) : Nat := (qcNZ (i0 : Int)).toNat

@[simp] theorem qcN_cast (i0 : Nat) : ((qcN i0 : Nat) : Int) = qcNZ (i0 : Int) :=
  Int.toNat_of_nonneg (selectInt_nonneg (by decide))

@[simp] theorem qcN_cast_ring {S : Type*} [Ring S] (i0 : Nat) :
    ((qcN i0 : Nat) : S) = ((qcNZ (i0 : Int) : Int) : S) := by
  rw [← qcN_cast, Int.cast_natCast]

/-- The output component `j`. (the table, over the instance index) -/
def compZ (i : Int) : Int := MxxIR.selectInt i [0, 1, 0, 1, 0, 1, 0, 1]

@[simp] theorem compZ_preprocess_sel_13 (i : Int) : Stage_preprocess.sel_13 i = compZ i := rfl
@[simp] theorem compZ_online_eval_sel_8 (i : Int) : Stage_online_eval.sel_8 i = compZ i := rfl

/-- The output component `j`. -/
def comp (i0 : Nat) : Nat := (compZ (i0 : Int)).toNat

@[simp] theorem comp_cast (i0 : Nat) : ((comp i0 : Nat) : Int) = compZ (i0 : Int) :=
  Int.toNat_of_nonneg (selectInt_nonneg (by decide))

@[simp] theorem comp_cast_ring {S : Type*} [Ring S] (i0 : Nat) :
    ((comp i0 : Nat) : S) = ((compZ (i0 : Int) : Int) : S) := by
  rw [← comp_cast, Int.cast_natCast]

/-- The quotient `Qq = R / qc`. (the table, over the instance index) -/
def QqZ (i : Int) : Int := MxxIR.selectInt i [1152919668508929361, 1152919668508929361, 1152919788767877569, 1152919788767877569, 1152920561861715089, 1152920561861715089, 1152920561861715089, 1152920561861715089]

@[simp] theorem QqZ_preprocess_sel_12 (i : Int) : Stage_preprocess.sel_12 i = QqZ i := rfl
@[simp] theorem QqZ_online_eval_sel_7 (i : Int) : Stage_online_eval.sel_7 i = QqZ i := rfl

/-- The quotient `Qq = R / qc`. -/
def Qq (i0 : Nat) : Nat := (QqZ (i0 : Int)).toNat

@[simp] theorem Qq_cast (i0 : Nat) : ((Qq i0 : Nat) : Int) = QqZ (i0 : Int) :=
  Int.toNat_of_nonneg (selectInt_nonneg (by decide))

@[simp] theorem Qq_cast_ring {S : Type*} [Ring S] (i0 : Nat) :
    ((Qq i0 : Nat) : S) = ((QqZ (i0 : Int) : Int) : S) := by
  rw [← Qq_cast, Int.cast_natCast]

/-- The action constant `c'` (`p c' = Qq k`). (the table, over the instance index) -/
def cpZ (i : Int) : Int := MxxIR.selectInt i [339699808396096934, 339699808396096934, 813220753465618156, 813220753465618156, 885475655076619601, 885475655076619601, 585309015821511054, 585309015821511054]

@[simp] theorem cpZ_preprocess_sel_16 (i : Int) : Stage_preprocess.sel_16 i = cpZ i := rfl
@[simp] theorem cpZ_online_eval_sel_11 (i : Int) : Stage_online_eval.sel_11 i = cpZ i := rfl

/-- The action constant `c'` (`p c' = Qq k`). -/
def cp (i0 : Nat) : Nat := (cpZ (i0 : Int)).toNat

@[simp] theorem cp_cast (i0 : Nat) : ((cp i0 : Nat) : Int) = cpZ (i0 : Int) :=
  Int.toNat_of_nonneg (selectInt_nonneg (by decide))

@[simp] theorem cp_cast_ring {S : Type*} [Ring S] (i0 : Nat) :
    ((cp i0 : Nat) : S) = ((cpZ (i0 : Int) : Int) : S) := by
  rw [← cp_cast, Int.cast_natCast]

/-- The divisor of the action constant (`k = c' / kdiv`). (the table, over the instance index) -/
def kdivZ (i : Int) : Int := MxxIR.selectInt i [1073741329, 1073741329, 1073741441, 1073741441, 1073741441, 1073741441, 1073741441, 1073741441]


/-- The divisor of the action constant (`k = c' / kdiv`). -/
def kdiv (i0 : Nat) : Nat := (kdivZ (i0 : Int)).toNat

@[simp] theorem kdiv_cast (i0 : Nat) : ((kdiv i0 : Nat) : Int) = kdivZ (i0 : Int) :=
  Int.toNat_of_nonneg (selectInt_nonneg (by decide))

@[simp] theorem kdiv_cast_ring {S : Type*} [Ring S] (i0 : Nat) :
    ((kdiv i0 : Nat) : S) = ((kdivZ (i0 : Int) : Int) : S) := by
  rw [← kdiv_cast, Int.cast_natCast]

/-- The first online scale factor. (the table, over the instance index) -/
def k3Z (i : Int) : Int := MxxIR.selectInt i [1073740529, 1073740529, 1073740529, 1073740529, 1073740529, 1073740529, 0, 0]

@[simp] theorem k3Z_preprocess_sel_14 (i : Int) : Stage_preprocess.sel_14 i = k3Z i := rfl
@[simp] theorem k3Z_online_eval_sel_9 (i : Int) : Stage_online_eval.sel_9 i = k3Z i := rfl

/-- The first online scale factor. -/
def k3 (i0 : Nat) : Nat := (k3Z (i0 : Int)).toNat

@[simp] theorem k3_cast (i0 : Nat) : ((k3 i0 : Nat) : Int) = k3Z (i0 : Int) :=
  Int.toNat_of_nonneg (selectInt_nonneg (by decide))

@[simp] theorem k3_cast_ring {S : Type*} [Ring S] (i0 : Nat) :
    ((k3 i0 : Nat) : S) = ((k3Z (i0 : Int) : Int) : S) := by
  rw [← k3_cast, Int.cast_natCast]

/-- The second online scale factor. (the table, over the instance index) -/
def k1Z (i : Int) : Int := MxxIR.selectInt i [1073741329, 1073741329, 112, 112, 832, 832, 912, 912]

@[simp] theorem k1Z_preprocess_sel_15 (i : Int) : Stage_preprocess.sel_15 i = k1Z i := rfl
@[simp] theorem k1Z_online_eval_sel_10 (i : Int) : Stage_online_eval.sel_10 i = k1Z i := rfl

/-- The second online scale factor. -/
def k1 (i0 : Nat) : Nat := (k1Z (i0 : Int)).toNat

@[simp] theorem k1_cast (i0 : Nat) : ((k1 i0 : Nat) : Int) = k1Z (i0 : Int) :=
  Int.toNat_of_nonneg (selectInt_nonneg (by decide))

@[simp] theorem k1_cast_ring {S : Type*} [Ring S] (i0 : Nat) :
    ((k1 i0 : Nat) : S) = ((k1Z (i0 : Int) : Int) : S) := by
  rw [← k1_cast, Int.cast_natCast]

/-- The mask sampling value `h` (`2 h = qc + 1`). (the table, over the instance index) -/
def hmZ (i : Int) : Int := MxxIR.selectInt i [536870721, 536870721, 536870665, 536870665, 536870305, 536870305, 536870265, 536870265]

@[simp] theorem hmZ_preprocess_sel_6 (i : Int) : Stage_preprocess.sel_6 i = hmZ i := rfl

/-- The mask sampling value `h` (`2 h = qc + 1`). -/
def hm (i0 : Nat) : Nat := (hmZ (i0 : Int)).toNat

@[simp] theorem hm_cast (i0 : Nat) : ((hm i0 : Nat) : Int) = hmZ (i0 : Int) :=
  Int.toNat_of_nonneg (selectInt_nonneg (by decide))

@[simp] theorem hm_cast_ring {S : Type*} [Ring S] (i0 : Nat) :
    ((hm i0 : Nat) : S) = ((hmZ (i0 : Int) : Int) : S) := by
  rw [← hm_cast, Int.cast_natCast]

/-- The half of the decoding ring. (the table, over the instance index) -/
def halfZ (i : Int) : Int := MxxIR.selectInt i [618968813111010066723674601, 618968813111010066723674601, 618968813111010066723674601, 618968813111010066723674601, 618968813111010066723674601, 618968813111010066723674601, 618968766994187592255071041, 618968766994187592255071041]

@[simp] theorem halfZ_preprocess_sel_9 (i : Int) : Stage_preprocess.sel_9 i = halfZ i := rfl
@[simp] theorem halfZ_online_eval_sel_4 (i : Int) : Stage_online_eval.sel_4 i = halfZ i := rfl

/-- The half of the decoding ring. -/
def half (i0 : Nat) : Nat := (halfZ (i0 : Int)).toNat

@[simp] theorem half_cast (i0 : Nat) : ((half i0 : Nat) : Int) = halfZ (i0 : Int) :=
  Int.toNat_of_nonneg (selectInt_nonneg (by decide))

@[simp] theorem half_cast_ring {S : Type*} [Ring S] (i0 : Nat) :
    ((half i0 : Nat) : S) = ((halfZ (i0 : Int) : Int) : S) := by
  rw [← half_cast, Int.cast_natCast]

/-- The inverse `(3 R + 1) / 4` of `4` modulo the decoding ring. (the table, over the instance index) -/
def inv4Z (i : Int) : Int := MxxIR.selectInt i [928453219666515100085511901, 928453219666515100085511901, 928453219666515100085511901, 928453219666515100085511901, 928453219666515100085511901, 928453219666515100085511901, 928453150491281388382606561, 928453150491281388382606561]

@[simp] theorem inv4Z_preprocess_sel_1 (i : Int) : Stage_preprocess.sel_1 i = inv4Z i := rfl

/-- The inverse `(3 R + 1) / 4` of `4` modulo the decoding ring. -/
def inv4 (i0 : Nat) : Nat := (inv4Z (i0 : Int)).toNat

@[simp] theorem inv4_cast (i0 : Nat) : ((inv4 i0 : Nat) : Int) = inv4Z (i0 : Int) :=
  Int.toNat_of_nonneg (selectInt_nonneg (by decide))

@[simp] theorem inv4_cast_ring {S : Type*} [Ring S] (i0 : Nat) :
    ((inv4 i0 : Nat) : S) = ((inv4Z (i0 : Int) : Int) : S) := by
  rw [← inv4_cast, Int.cast_natCast]

/-- The decoding ring of instance `i0`: `q₀ q₁ q₂` (chains `0`–`2`) or `q₀ q₁ q₃` (chain `3`). -/
abbrev R (i0 : Nat) : Nat := Stage_online_eval.mod_0 (i0 : Int)

/-- The CRT prime `q_c` of the chain of instance `i0`, the modulus of its residues. -/
abbrev qc (i0 : Nat) : Nat := Stage_online_eval.mod_1 (i0 : Int)

/-- The plaintext modulus of instance `i0`: `q₀ q₁` (chains `0`, `1`), `q₀ q₂`, `q₀ q₃`. -/
abbrev Mx (i0 : Nat) : Nat := Stage_online_eval.mod_2 (i0 : Int)

/-- `exact h` up to the casts of the instance's constants (natural numbers to their tables). -/
macro "cast_exact " h:term : tactic => `(tactic| first
  | exact $h
  | (have h' := $h; simp only [qd_cast, rp_cast, pp_cast, mxp_cast, qcN_cast, comp_cast, Qq_cast, cp_cast, kdiv_cast, k3_cast, k1_cast, hm_cast, half_cast, inv4_cast] at h' ⊢; exact h'))

end RefreshTopProof
