import PeAnchor

/-!
# Decryption

The helper encodings and the output encoding in the form of the phase identity, the key of each slot
from its anchor preimage, and the integer phase error with its coefficient bound.
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem joinRow_add (x y : Fin 2 → ExactMatrix Q N 1 30) :
    joinRow (fun j ↦ x j + y j) = joinRow x + joinRow y := by
  funext r p
  simp only [joinRow, Matrix.add_apply, joinedEntry_split]
  split <;> rfl

theorem joinRow_sub (x y : Fin 2 → ExactMatrix Q N 1 30) :
    joinRow (fun j ↦ x j - y j) = joinRow x - joinRow y := by
  funext r p
  simp only [joinRow, Matrix.sub_apply, joinedEntry_split]
  split <;> rfl

theorem joinRow_smul (a : ExactPoly Q N) (x : Fin 2 → ExactMatrix Q N 1 30) :
    joinRow (fun j ↦ a • x j) = a • joinRow x := by
  funext r p
  simp only [joinRow, Matrix.smul_apply, joinedEntry_split]
  split <;> rfl

theorem joinedEntry_gen {α : Type} (r : Fin 2 → Matrix (Fin 1) (Fin 30) α) (p : Fin 60) :
    joinedEntry r p = if h : p.val < 30 then r 0 0 ⟨p.val, h⟩ else r 1 0 ⟨p.val - 30, by omega⟩ := by
  unfold joinedEntry Fin.addCases
  split <;> simp_all <;> rfl

/-- The joined integer row `[e₀ | e₁]`. -/
def jZ (e : Fin 2 → ErrorMatrix N 1 30) : ErrorMatrix N 1 60 := fun _ p ↦ joinedEntry e p

theorem joinRow_reduce (e : Fin 2 → ErrorMatrix N 1 30) :
    joinRow (fun j ↦ reduceMatrix Q N 1 30 (e j)) = reduceMatrix Q N 1 60 (jZ e) := by
  funext r p
  simp only [joinRow, jZ, reduceMatrix, joinedEntry_gen]
  split <;> rfl

theorem jZ_bound {e : Fin 2 → ErrorMatrix N 1 30} {B : Nat} (he : ∀ j, CoeffBound (e j) B) :
    CoeffBound (jZ e) B := by
  intro r p k
  simp only [jZ, joinedEntry_gen]
  split
  · exact he 0 0 _ k
  · exact he 1 0 _ k

theorem joinRow_pair (x : Fin 2 → ExactMatrix Q N 1 30) : joinRow ![x 0, x 1] = joinRow x := by
  rw [pair_eta]

/-- The joined target of a helper of member `k`: `σ_k(s) [g | 0] + [0 | g]`. -/
theorem joinRow_helperTarget (k : Nat) (hk : Odd k) (sm : ErrorPoly N) :
    joinRow (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
      unitPair (substitution k hk sm) j) • g0) =
      substitution k hk (reducePoly Q N sm) • gZero + zeroG := by
  have key : reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * substitution k hk sm) =
      substitution k hk (reducePoly Q N sm) := by
    rw [Nat.cast_one, Int.cast_one, one_mul]
    exact reducePoly_substitution k hk sm
  have key1 : reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * 1) = 1 := by simp
  funext r p
  simp only [joinRow, gZero, zeroG, Matrix.add_apply, Matrix.smul_apply, joinedEntry_split]
  split
  · simp only [unitPair, Matrix.cons_val_zero, key, Matrix.zero_apply, add_zero, smul_eq_mul]
  · simp only [unitPair, Matrix.cons_val_one, Matrix.cons_val_zero, key1, Matrix.zero_apply,
      mul_zero, zero_add, smul_eq_mul, one_mul]

/-- A helper encoding in the form of the phase identity. -/
theorem helper_form {sm : ErrorPoly N} {model : HashModel} {key : ByteArray} {k : Nat} (hk : Odd k)
    {hf : Fin 2 → ExactMatrix Q N 1 30}
    (h : Rel sm (helperKey model key k)
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * unitPair (substitution k hk sm) j) • g0)
      hf 31) :
    ∃ e : Fin 2 → ErrorMatrix N 1 30, (∀ j, CoeffBound (e j) 31) ∧
      joinRow hf - joinRow (helperKey model key k 1) =
        reducePoly Q N sm • joinRow (helperKey model key k 0) -
          substitution k hk (reducePoly Q N sm) • gZero - zeroG + reduceMatrix Q N 1 60 (jZ e) := by
  obtain ⟨e, he, hc⟩ := h
  refine ⟨e, he, ?_⟩
  have hf' : hf = fun j ↦ ((reducePoly Q N sm • helperKey model key k 0 j + helperKey model key k 1 j) -
      reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * unitPair (substitution k hk sm) j) • g0) +
      reduceMatrix Q N 1 30 (e j) := funext hc
  rw [hf', joinRow_add, joinRow_sub, joinRow_add, joinRow_smul, joinRow_helperTarget, joinRow_reduce]
  abel

/-- The output encoding of a zero target in the form of the phase identity. -/
theorem out_form {sm : ErrorPoly N} {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 30} {c : Fin 2 → ExactMatrix Q N 1 30}
    {B : Nat} (h : Rel sm A (fun _ ↦ 0) c B) :
    ∃ e : Fin 2 → ErrorMatrix N 1 30, (∀ j, CoeffBound (e j) B) ∧
      joinRow c - joinRow (A 1) = reducePoly Q N sm • joinRow (A 0) + reduceMatrix Q N 1 60 (jZ e) := by
  obtain ⟨e, he, hc⟩ := h
  refine ⟨e, he, ?_⟩
  have hc' : c = fun j ↦ ((reducePoly Q N sm • A 0 j + A 1 j) - 0) + reduceMatrix Q N 1 30 (e j) := funext hc
  rw [hc', joinRow_add, joinRow_sub, joinRow_add, joinRow_smul, joinRow_reduce]
  have h0 : joinRow (fun _ : Fin 2 ↦ (0 : ExactMatrix Q N 1 30)) = 0 := by
    funext r p
    simp only [joinRow, joinedEntry_split]
    split <;> rfl
  rw [h0]
  abel

/-- `σ_i σ_i⁻¹ = 1`. -/
theorem slot_inv (i : Fin 256) : slotAut i * slotInv i % (2 * N) = 1 := by
  rw [slotAut_eq i i.isLt, slotInv_eq i i.isLt, hIndex_mul]
  have : ((i : Nat) + (256 - i) % 256) % 256 = 0 := by omega
  rw [this]
  rfl

end PeProof
