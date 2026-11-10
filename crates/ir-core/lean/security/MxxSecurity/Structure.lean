import MxxSecurity.Uniform

/-!
# Structure of sampled families

A family of `k + l` independent draws is a family of `k` draws followed by one of `l`.
-/

namespace MxxSecurity

open MxxRuntime

theorem sampleFamily_cast (law : SamplerLaw) {m m' : Nat} (h : m = m') :
    sampleFamily law m = (fun v ↦ v ∘ Fin.cast h) <$> sampleFamily law m' := by
  subst h
  have : (fun v : Fin m → Int ↦ v ∘ Fin.cast rfl) = id := by
    funext v i
    rfl
  rw [this, id_map]

theorem cons_append_cast {k l : Nat} (head : Int) (x : Fin k → Int) (y : Fin l → Int) :
    (Fin.cons head (Fin.append x y) : Fin (k + l + 1) → Int) ∘ Fin.cast (Nat.succ_add k l) =
      Fin.append (Fin.cons head x : Fin (k + 1) → Int) y := by
  funext i
  refine Fin.addCases (fun j ↦ ?_) (fun j ↦ ?_) i
  · refine Fin.cases ?_ (fun j' ↦ ?_) j
    · simp [Fin.append_left]
      rfl
    · simp only [Function.comp_apply, Fin.append_left, Fin.cons_succ]
      have : Fin.cast (Nat.succ_add k l) (Fin.castAdd l j'.succ) = (Fin.castAdd l j').succ := by
        ext
        simp
      rw [this, Fin.cons_succ, Fin.append_left]
  · simp only [Function.comp_apply, Fin.append_right]
    have : Fin.cast (Nat.succ_add k l) (Fin.natAdd (k + 1) j) = (Fin.natAdd k j).succ := by
      ext
      simp
      omega
    rw [this, Fin.cons_succ, Fin.append_right]

/-- A family of `k + l` draws is a family of `k` draws followed by one of `l`. -/
theorem sampleFamily_add (law : SamplerLaw) (l : Nat) : ∀ k : Nat,
    sampleFamily law (k + l) = (do
      let x ← sampleFamily law k
      let y ← sampleFamily law l
      pure (Fin.append x y)) := by
  intro k
  induction k with
  | zero =>
    rw [sampleFamily_cast law (Nat.zero_add l)]
    simp only [sampleFamily_zero, pure_bind, map_eq_bind_pure_comp, Function.comp_def]
    congr 1
    funext y
    congr 1
    funext i
    simp [Fin.append, Fin.addCases]
  | succ k ih =>
    rw [sampleFamily_cast law (Nat.succ_add k l), sampleFamily_succ, ih]
    simp only [sampleFamily_succ, map_eq_bind_pure_comp, bind_assoc, pure_bind, Function.comp_def]
    congr 1
    funext head
    congr 1
    funext x
    congr 1
    funext y
    congr 1
    exact cons_append_cast head x y

theorem clamp_clamp (law : SamplerLaw) (x : Int) : law.clamp (law.clamp x) = law.clamp x := by
  cases law with
  | gaussian sigma cutoff =>
    simp only [SamplerLaw.clamp]
    split_ifs <;> simp_all
  | interval minimum maximum =>
    simp only [SamplerLaw.clamp]
    split_ifs <;> simp_all <;> omega
  | residue modulus =>
    simp only [SamplerLaw.clamp]
    split_ifs <;> simp_all <;> omega

/-- Every value a family returns is already clamped into the law's support. -/
theorem sampleFamily_clamp (law : SamplerLaw) :
    ∀ k, sampleFamily law k = (fun v ↦ law.clamp ∘ v) <$> sampleFamily law k := by
  intro k
  induction k with
  | zero =>
    simp only [sampleFamily_zero, map_pure]
    congr 1
    funext i
    exact i.elim0
  | succ k ih =>
    conv_lhs => rw [sampleFamily_succ, ih]
    simp only [sampleFamily_succ, sample_eq, map_eq_bind_pure_comp, bind_assoc, pure_bind,
      Function.comp_def]
    congr 1
    funext x
    congr 1
    funext tail
    congr 1
    funext i
    refine Fin.cases ?_ (fun j ↦ ?_) i
    · simp [clamp_clamp]
    · simp [clamp_clamp]

/-- A sampled matrix is the matrix of clamped values. -/
theorem sampleMatrix_clamp (law : SamplerLaw) (q n rows columns : Nat) :
    sampleMatrix law q n rows columns =
      (do let values ← sampleFamily law (rows * columns * n)
          pure (matrixOf (q := q) fun i ↦ law.clamp (values i))) := by
  rw [sampleMatrix_eq, sampleFamily_clamp law]
  simp only [map_eq_bind_pure_comp, bind_assoc, pure_bind, Function.comp_def, clamp_clamp]

section Columns
open Mxx.Primitives

/-- Two `1 × 1` matrices side by side. -/
def joinCols {q n : Nat} (x y : ExactMatrix q n 1 1) : ExactMatrix q n 1 2 :=
  fun _ j ↦ if j.val = 0 then x 0 0 else y 0 0

theorem idx_val {n : Nat} (r : Fin 1) (j : Fin 2) (k : Fin n) :
    (finProdFinEquiv (finProdFinEquiv (r, j), k) : Fin (1 * 2 * n)).val = k.val + n * j.val := by
  simp [finProdFinEquiv]

theorem idx_val11 {n : Nat} (r : Fin 1) (j : Fin 1) (k : Fin n) :
    (finProdFinEquiv (finProdFinEquiv (r, j), k) : Fin (1 * 1 * n)).val = k.val := by
  simp [finProdFinEquiv]

/-- A `1 × 2` sample is two independent `1 × 1` samples side by side. -/
theorem sampleMatrix_split_columns (law : SamplerLaw) (q n : Nat) :
    sampleMatrix law q n 1 2 = (do
      let x ← sampleMatrix law q n 1 1
      let y ← sampleMatrix law q n 1 1
      pure (joinCols x y)) := by
  have h : 1 * 2 * n = 1 * 1 * n + 1 * 1 * n := by ring
  rw [sampleMatrix_eq, sampleMatrix_eq, sampleFamily_cast law h, sampleFamily_add]
  simp only [map_eq_bind_pure_comp, bind_assoc, pure_bind, Function.comp_def]
  refine bind_congr fun x ↦ bind_congr fun y ↦ ?_
  congr 1
  funext r j
  unfold matrixOf joinCols
  by_cases hj : j.val = 0
  · rw [if_pos hj]
    refine congrArg polynomialOfCoefficients (funext fun k ↦ ?_)
    have hidx : Fin.cast h (finProdFinEquiv (finProdFinEquiv (r, j), k)) =
        Fin.castAdd _ (finProdFinEquiv (finProdFinEquiv ((0 : Fin 1), (0 : Fin 1)), k)) :=
      Fin.ext (by simp only [Fin.val_cast, Fin.val_castAdd, idx_val, idx_val11, hj]; simp)
    show Fin.append x y (Fin.cast h _) = _
    rw [hidx, Fin.append_left]
  · rw [if_neg hj]
    have hj1 : j.val = 1 := by omega
    refine congrArg polynomialOfCoefficients (funext fun k ↦ ?_)
    have hidx : Fin.cast h (finProdFinEquiv (finProdFinEquiv (r, j), k)) =
        Fin.natAdd _ (finProdFinEquiv (finProdFinEquiv ((0 : Fin 1), (0 : Fin 1)), k)) :=
      Fin.ext (by simp only [Fin.val_cast, Fin.val_natAdd, idx_val, idx_val11, hj1]; simp; omega)
    show Fin.append x y (Fin.cast h _) = _
    rw [hidx, Fin.append_right]
end Columns

end MxxSecurity
