import GenFinalFresh

/-!
# Member values of the fresh noise and the mask, for any parameter set

* `walsh_termsC`, `walsh_termsE`: a key table summed by the preprocess (`keyTableG`, terms
  `keyTermC` — `t_j σ(X^(j s))` — or `keyTermE` — `σ(X^(j s)) X^(t_j o)`, `s = -stride`) is the
  Fourier coefficient `lutFourierCoefficient W (X^stride) b T σ` of the table (`n = stride W`).
* `message_Xe`: a label payload `c X^e` (`c = ±1`) is `X^(s₀ a)` under the requirement's relation.
* `prf_mu`: the PRF output of two buckets is `X^(s₀ (a₀ + a₁))`.
-/

namespace RefreshTopProof.GenFinal

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof Gen
open scoped BigOperators

variable {n : Nat}

/-- `X^k` only depends on `k mod 2n`. -/
theorem root_pow_mod2n {R : Type} [CommRing R] (k : Nat) :
    (AdjoinRoot.root (negacyclicModulus n R)) ^ k = (AdjoinRoot.root (negacyclicModulus n R)) ^ (k % (2 * n)) := by
  apply pow_mod_period
  rw [mul_comm, pow_mul, Negacyclic.root_pow_n]; norm_num

/-- The exponent of a monomial `X^(-j s)` reduced into `[0, 2n)` is `s (2W - j) mod 2n`. -/
theorem neg_exp_mod {s W j : Nat} (hn : n = s * W) (hj : j < W) :
    (((j : Int) * -(s : Int)) % (2 * (n : Int))).toNat % (2 * n) = (s * (2 * W - j)) % (2 * n) := by
  have hsW : s * (2 * W - j) = 2 * n - s * j := by
    rw [Nat.mul_sub, hn]; ring_nf
  rw [hsW]
  rcases Nat.eq_zero_or_pos (s * j) with h0 | hpos
  · have : (j : Int) * -(s : Int) = 0 := by
      have : ((s * j : Nat) : Int) = 0 := by rw [h0]; rfl
      push_cast at this; linarith
    rw [this, h0]; simp
  have hlt : s * j < 2 * n := by
    have : s * j ≤ s * W := Nat.mul_le_mul_left _ hj.le
    rw [hn]; omega
  have e1 : ((j : Int) * -(s : Int)) % (2 * (n : Int)) = ((2 * n - s * j : Nat) : Int) := by
    have hc : ((2 * n - s * j : Nat) : Int) = 2 * (n : Int) - (s : Int) * j := by
      rw [Nat.cast_sub hlt.le]; push_cast; ring
    have : ((j : Int) * -(s : Int)) = ((2 * n - s * j : Nat) : Int) + (-1) * (2 * (n : Int)) := by
      rw [hc]; ring
    rw [this, Int.add_mul_emod_self_right, Int.emod_eq_of_lt (by positivity) (by rw [hc]; push_cast at hpos; nlinarith)]
  rw [e1, Int.toNat_natCast]

/-- The generic per-term shape of a coefficient table term at the root: `σ(X^(j s))`. -/
theorem term_root {q : Nat} (hq : 1 < q) (hn0 : 0 < n) {σ : Nat} (hodd : Odd σ) {s W : Nat} (hn : n = s * W)
    (j : Fin W) :
    ringAutomorphism σ (multiplyMonomial (matrixPolynomial [1] : ExactMatrix q n 1 1)
      ((j : Nat) * -(s : Int))) 0 0 = (X q ^ s) ^ (σ * (2 * W - j)) := by
  rw [ringAutomorphism_eq hq hn0 σ hodd]
  simp only [multiplyMonomial, matrixPolynomial_single, Int.cast_one, one_mul]
  rw [substitution_root_pow, root_pow_mod2n, Nat.mul_mod, neg_exp_mod hn j.isLt, ← Nat.mul_mod,
    ← root_pow_mod2n, ← pow_mul, X]
  ring_nf

/-- **The coefficient table** (`keyTermC` terms): `b Σ_j t_j σ(X^(-j stride))` is the Fourier
coefficient of the table `t`. -/
theorem walsh_termsC {q : Nat} (hq : 1 < q) (hn0 : 0 < n) {σ : Nat} (hodd : Odd σ) {s W : Nat}
    (hn : n = s * W) (b : ExactPoly q n) (tv : Fin W → Int) :
    b * ∑ j : Fin W, (matrixMulScalarLeft (liftInteger (tv j) : ExactMatrix q n 1 1)
      (ringAutomorphism σ (multiplyMonomial (matrixPolynomial [1] : ExactMatrix q n 1 1)
        ((j : Nat) * -(s : Int))))) 0 0 =
      lutFourierCoefficient W (X q ^ s) b (fun k ↦ ((tv k : Int) : ExactPoly q n)) σ := by
  unfold lutFourierCoefficient
  congr 1
  refine Finset.sum_congr rfl fun j _ ↦ ?_
  simp only [matrixMulScalarLeft, liftInteger]
  rw [term_root hq hn0 hodd hn j]

/-- **The exponent table** (`keyTermE` terms, readout): `b Σ_j σ(X^(-j stride)) X^(t_j o)` is the
Fourier coefficient of the table `k ↦ X^(o t_k)` (`0 ≤ o t_k < 2n`). -/
theorem walsh_termsE {q : Nat} (hq : 1 < q) (hn0 : 0 < n) {σ : Nat} (hodd : Odd σ) {s W : Nat}
    (hn : n = s * W) (b : ExactPoly q n) (o : Nat) (tv : Fin W → Nat) :
    b * ∑ j : Fin W, (multiplyMonomial (ringAutomorphism σ (multiplyMonomial (matrixPolynomial [1] : ExactMatrix q n 1 1)
        ((j : Nat) * -(s : Int)))) ((tv j : Int) * (o : Int))) 0 0 =
      lutFourierCoefficient W (X q ^ s) b (fun k ↦ reducePoly q n (Xe (n := n) (o * tv k))) σ := by
  unfold lutFourierCoefficient
  congr 1
  refine Finset.sum_congr rfl fun j _ ↦ ?_
  simp only [multiplyMonomial]
  rw [show ringAutomorphism σ (multiplyMonomial (matrixPolynomial [1] : ExactMatrix q n 1 1)
      ((j : Nat) * -(s : Int))) 0 0 = (X q ^ s) ^ (σ * (2 * W - j)) from term_root hq hn0 hodd hn j]
  rw [reducePoly_Xe, mul_comm]
  congr 1
  rw [root_pow_mod2n (o * tv j)]
  congr 1
  have : (((tv j : Int) * (o : Int)) % (2 * (n : Int))).toNat = (o * tv j) % (2 * n) := by
    rw [show ((tv j : Int) * (o : Int)) = ((o * tv j : Nat) : Int) by push_cast; ring,
      show (2 * (n : Int)) = ((2 * n : Nat) : Int) by push_cast; ring]
    rw [← Int.natCast_mod, Int.toNat_natCast]
  rw [this]

/-! ## Labels and the PRF output -/

/-- **A label payload** `c X^e` with `c = ±1` is the monomial `X^k` when `e = k` (`c = 1`) or
`e + n = k` (`c = -1`), `0 ≤ e < 2n`. -/
theorem message_Xe {c e : Int} {k : Nat} (he : 0 ≤ e) (he2 : e < 2 * n)
    (h : (c = 1 ∧ e = k) ∨ (c = -1 ∧ e + n = k)) : message (n := n) c e = Xe (n := n) k := by
  have hmod : (e % ((2 * n : Nat) : Int)).toNat = e.toNat := by
    rw [Int.emod_eq_of_lt he (by push_cast; omega)]
  unfold message
  rw [hmod]
  rcases h with ⟨rfl, rfl⟩ | ⟨rfl, hk⟩
  · simp [Xe]
  · have hk' : k = e.toNat + n := by omega
    rw [hk', Xe, pow_add, Negacyclic.root_pow_n]
    simp

/-- **The PRF output** of two buckets on the start `1`: `X^(s₀ a₁) (X^(s₀ a₀) 1) = X^(s₀ (a₀ + a₁))`. -/
theorem prf_mu {s0 a0 a1 : Nat} :
    Xe (n := n) (s0 * a1) * (Xe (n := n) (s0 * a0) * 1) = Xe (n := n) (s0 * (a0 + a1)) := by
  simp only [Xe, mul_one, ← pow_add]
  ring_nf

/-- A table vanishing at the even entries is the bin table of its odd entries. -/
theorem table_binTable {q P : Nat} (hq : 1 < q) (tv : Nat → Int) (h0 : ∀ k : Fin P, k.val % 2 = 0 → tv k = 0) :
    (fun k : Fin P ↦ ((tv k : Int) : ExactPoly q n)) =
      fun k ↦ reducePoly q n (binTable (n := n) P (fun b ↦ tv (2 * b + 1)) k) := by
  funext k
  unfold binTable
  split_ifs with hk
  · rw [reducePoly_intCast]
    congr 2
    omega
  · rw [h0 k (by omega), map_zero]; simp

/-- **A fresh member's value**: the PRF output `X^(s₀ z)`, read out by the readout LUT
(`hwr`, coefficients `Cr` of the exponent table `X^(s₁ readoutVal)` at `Qr`) and sampled by the
sampling LUT (`hws`, coefficients `Cs` of the bin table `g` at `R ∣ Qr`), is `samplerVal P W' g z`
modulo `R`. -/
theorem fresh_value_gen (hn0 : 0 < n) {Qr R : Nat} (hR : 1 < R) (hQr : 1 < Qr) (hRQ : R ∣ Qr)
    {s0 s1 logr logs P invr invs : Nat} (hP : P = 2 ^ logs) (hP2 : 2 ≤ P)
    (hn0' : n = s0 * 2 ^ logr) (hn1 : n = s1 * P) (g : Nat → Int)
    (hinvr : (invr : ExactPoly Qr n) * ((2 ^ logr : Nat) : ExactPoly Qr n) = 1)
    (hinvs : (invs : ExactPoly R n) * ((2 ^ logs : Nat) : ExactPoly R n) = 1)
    (ir : Fin (2 ^ logr) → Nat) (hor : ∀ i, Odd (ir i)) (pr : Fin (2 ^ logr) ≃ Fin (2 ^ logr))
    (hir : ∀ i, ir i % (2 * 2 ^ logr) = 1 + 2 * 1 * (pr i).val)
    (is : Fin (2 ^ logs) → Nat) (hos : ∀ i, Odd (is i)) (ps : Fin (2 ^ logs) ≃ Fin (2 ^ logs))
    (his : ∀ i, is i % (2 * 2 ^ logs) = 1 + 2 * 1 * (ps i).val)
    {Cr : Fin (2 ^ logr) → ExactPoly Qr n}
    (hCr : ∀ i, Cr i = lutFourierCoefficient (2 ^ logr) (X Qr ^ s0) (invr : ExactPoly Qr n)
      (fun k : Fin (2 ^ logr) ↦ reducePoly Qr n (Xe (n := n) (s1 * readoutVal (P / 2) (2 ^ logr) k))) (ir i))
    {Cs : Fin (2 ^ logs) → ExactPoly R n}
    (hCs : ∀ i, Cs i = lutFourierCoefficient (2 ^ logs) (X R ^ s1) (invs : ExactPoly R n)
      (fun k ↦ reducePoly R n (binTable (n := n) (2 ^ logs) g k)) (is i))
    {μ νr νs : ErrorPoly n} {z : Nat} (hμ : μ = Xe (n := n) (s0 * z))
    (hwr : ∑ i, reducePoly Qr n (substitution (ir i) (hor i) μ) * Cr i = reducePoly Qr n νr)
    (hws : ∑ i, reducePoly R n (substitution (is i) (hos i) νr) * Cs i = reducePoly R n νs) :
    reducePoly R n νs = reducePoly R n (((samplerVal P (2 ^ logr) g (z : Int) : Int)) : ErrorPoly n) := by
  have hr := readout_value_of_walsh hn0' hn1 hn0 hinvr ir hor pr hir hCr (z := z) (by rw [hμ]) hwr
  have hrR : reducePoly R n νr = reducePoly R n (Xe (n := n) (s1 * readoutExpG P (2 ^ logr) z)) := by
    rw [← rho_reduce hn0 hR hQr hRQ, hr, rho_reduce hn0 hR hQr hRQ]
  exact sampler_value_gen hP hP2 (by positivity) (hP ▸ hn1) hn0 g hinvs is hos ps his hCs hrR hws

/-- **A mask member's value** (and, with `z = 0`, the offset's: `readoutExpG P W' 0 = 1`): the
readout then the sampling LUT of table `T` gives `sampledG P T (readoutExpG P W' z)` modulo `R`. -/
theorem mask_value_gen (hn0 : 0 < n) {Qr R : Nat} (hR : 1 < R) (hQr : 1 < Qr) (hRQ : R ∣ Qr)
    {s0 s1 logr logs P invr invs : Nat} (hP : P = 2 ^ logs)
    (hn0' : n = s0 * 2 ^ logr) (hn1 : n = s1 * P) (T : Fin (2 ^ logs) → ErrorPoly n)
    (hinvr : (invr : ExactPoly Qr n) * ((2 ^ logr : Nat) : ExactPoly Qr n) = 1)
    (hinvs : (invs : ExactPoly R n) * ((2 ^ logs : Nat) : ExactPoly R n) = 1)
    (ir : Fin (2 ^ logr) → Nat) (hor : ∀ i, Odd (ir i)) (pr : Fin (2 ^ logr) ≃ Fin (2 ^ logr))
    (hir : ∀ i, ir i % (2 * 2 ^ logr) = 1 + 2 * 1 * (pr i).val)
    (is : Fin (2 ^ logs) → Nat) (hos : ∀ i, Odd (is i)) (ps : Fin (2 ^ logs) ≃ Fin (2 ^ logs))
    (his : ∀ i, is i % (2 * 2 ^ logs) = 1 + 2 * 1 * (ps i).val)
    {Cr : Fin (2 ^ logr) → ExactPoly Qr n}
    (hCr : ∀ i, Cr i = lutFourierCoefficient (2 ^ logr) (X Qr ^ s0) (invr : ExactPoly Qr n)
      (fun k : Fin (2 ^ logr) ↦ reducePoly Qr n (Xe (n := n) (s1 * readoutVal (P / 2) (2 ^ logr) k))) (ir i))
    {Cs : Fin (2 ^ logs) → ExactPoly R n}
    (hCs : ∀ i, Cs i = lutFourierCoefficient (2 ^ logs) (X R ^ s1) (invs : ExactPoly R n)
      (fun k ↦ reducePoly R n (T k)) (is i))
    {μ νr νs : ErrorPoly n} {z : Nat} (hμ : μ = Xe (n := n) (s0 * z))
    (hwr : ∑ i, reducePoly Qr n (substitution (ir i) (hor i) μ) * Cr i = reducePoly Qr n νr)
    (hws : ∑ i, reducePoly R n (substitution (is i) (hos i) νr) * Cs i = reducePoly R n νs) :
    haveI : NeZero (2 ^ logs) := ⟨by positivity⟩
    reducePoly R n νs = reducePoly R n (sampledG (2 ^ logs) T (readoutExpG P (2 ^ logr) z)) := by
  have hr := readout_value_of_walsh hn0' hn1 hn0 hinvr ir hor pr hir hCr (z := z) (by rw [hμ]) hwr
  have hrR : reducePoly R n νr = reducePoly R n (Xe (n := n) (s1 * readoutExpG P (2 ^ logr) z)) := by
    rw [← rho_reduce hn0 hR hQr hRQ, hr, rho_reduce hn0 hR hQr hRQ]
  exact sampled_of_walsh (hP ▸ hn1) hn0 T hinvs is hos ps his hCs hrR hws

/-! ## F's summed key tables (`keyTableG` of `keyTermC` / `keyTermE` terms, unfolded) -/

theorem range_dite_sum {M : Type} [AddCommMonoid M] {W : Nat} (f : Fin W → M) :
    ∑ k ∈ Finset.range W, (if h : k < W then f ⟨k, h⟩ else 0) = ∑ j : Fin W, f j := by
  rw [Finset.sum_range]
  exact Finset.sum_congr rfl fun j _ ↦ by rw [dif_pos j.isLt]

/-- **A coefficient key table** (`keyTableG [a] [b] (Σ_k keyTermC σ t 1 (-s) k)`, unfolded) is
`a` times the Fourier coefficient of the table `t` with inverse `b`. -/
theorem keyTableC_walsh {q : Nat} (hq : 1 < q) (hn0 : 0 < n) {σ : Int} (hodd : Odd σ.toNat) {s W : Nat}
    (hn : n = s * W) (a b : Int) (tv : Fin W → Int) :
    (matrixMulScalarLeft (matrixPolynomial [a] : ExactMatrix q n 1 1) (matrixMulScalarLeft (matrixPolynomial [b])
      (∑ k ∈ Finset.range W, (if h : k < W then matrixMulScalarLeft (liftInteger (tv ⟨k, h⟩) : ExactMatrix q n 1 1)
        (ringAutomorphism σ.toNat (multiplyMonomial (matrixPolynomial [1] : ExactMatrix q n 1 1)
          (Int.ofNat k * -(s : Int)))) else 0)))) 0 0 =
      (a : ExactPoly q n) * lutFourierCoefficient W (X q ^ s) (b : ExactPoly q n)
        (fun k ↦ ((tv k : Int) : ExactPoly q n)) σ.toNat := by
  rw [range_dite_sum (fun j : Fin W ↦ matrixMulScalarLeft (liftInteger (tv j) : ExactMatrix q n 1 1)
    (ringAutomorphism σ.toNat (multiplyMonomial (matrixPolynomial [1] : ExactMatrix q n 1 1)
      (Int.ofNat (j : Nat) * -(s : Int)))))]
  rw [← walsh_termsC hq hn0 hodd hn]
  simp only [matrixMulScalarLeft, matrixPolynomial_single, Matrix.sum_apply, Int.ofNat_eq_natCast]

/-- **An exponent key table** (`keyTableG [a] [b] (Σ_k keyTermE σ o t 1 (-s) k)`, unfolded; readout
LUT) is `a` times the Fourier coefficient of the table `k ↦ X^(o t_k)`. -/
theorem keyTableE_walsh {q : Nat} (hq : 1 < q) (hn0 : 0 < n) {σ : Int} (hodd : Odd σ.toNat) {s W : Nat}
    (hn : n = s * W) (a b : Int) (o : Nat) (tv : Fin W → Nat) :
    (matrixMulScalarLeft (matrixPolynomial [a] : ExactMatrix q n 1 1) (matrixMulScalarLeft (matrixPolynomial [b])
      (∑ k ∈ Finset.range W, (if h : k < W then multiplyMonomial (ringAutomorphism σ.toNat
        (multiplyMonomial (matrixPolynomial [1] : ExactMatrix q n 1 1) (Int.ofNat k * -(s : Int))))
          (((tv ⟨k, h⟩ : Nat) : Int) * (o : Int)) else 0)))) 0 0 =
      (a : ExactPoly q n) * lutFourierCoefficient W (X q ^ s) (b : ExactPoly q n)
        (fun k ↦ reducePoly q n (Xe (n := n) (o * tv k))) σ.toNat := by
  rw [range_dite_sum (fun j : Fin W ↦ multiplyMonomial (ringAutomorphism σ.toNat
    (multiplyMonomial (matrixPolynomial [1] : ExactMatrix q n 1 1) (Int.ofNat (j : Nat) * -(s : Int))))
      (((tv j : Nat) : Int) * (o : Int)))]
  rw [← walsh_termsE hq hn0 hodd hn]
  simp only [matrixMulScalarLeft, matrixPolynomial_single, Matrix.sum_apply, Int.ofNat_eq_natCast]

/-- **The PRF output form** (`μ_PRF ≡ X^(s₀ z)`, `z = prfZB`): a member's PRF output is the product
over the `B` buckets of the label payloads `α` at the keys; with label payloads `X^(s₀ a)`
(`message_Xe`), keys the table entries at the selected cells (`hkey`) and one-hot selector rows
(`hs`), it is `X^(s₀ (prfZB W a s t o).toNat)`. -/
theorem prf_z_gen {B W : Nat} (s0 : Nat) (a : Nat → Int) (ha : ∀ k, 0 ≤ a k) (α : Nat → ErrorPoly n)
    (hα : ∀ k, α k = Xe (n := n) (s0 * (a k).toNat)) (s t : Fin B → Nat → Int) (o : Nat)
    (sel : Fin B → Fin W) (hs : ∀ b (k : Fin W), s b (o * W + k) = if k = sel b then 1 else 0)
    (key : Fin B → Nat) (hkey : ∀ b, key b = (t b (o * W + sel b)).toNat) :
    ∏ b, α (key b) = Xe (n := n) (s0 * (prfZB W a s t o).toNat) := by
  rw [prfZB_onehot a s t o sel hs]
  have hsum : (∑ b : Fin B, a (t b (o * W + sel b)).toNat).toNat =
      ∑ b : Fin B, (a (t b (o * W + sel b)).toNat).toNat := by
    have h1 : (0 : Int) ≤ ∑ b : Fin B, a (t b (o * W + sel b)).toNat := Finset.sum_nonneg fun b _ ↦ ha _
    rw [← Int.natCast_inj, Int.toNat_of_nonneg h1, Nat.cast_sum]
    exact Finset.sum_congr rfl fun b _ ↦ (Int.toNat_of_nonneg (ha _)).symm
  rw [hsum, Finset.mul_sum]
  simp only [hα, hkey, Xe, Finset.prod_pow_eq_pow_sum]

/-- An inverse certificate: `(b P) mod q = 1` gives `b P = 1` in the ring. -/
theorem cast_inverse {q b P : Nat} (hq : 1 < q) (h : b * P % q = 1) :
    (b : ExactPoly q n) * (P : ExactPoly q n) = 1 := by
  haveI : Fact (1 < q) := ⟨hq⟩
  rw [← Nat.cast_mul, ← map_natCast (algebraMap (ZMod q) (ExactPoly q n)), ← ZMod.natCast_mod, h,
    Nat.cast_one, map_one]

/-- The label check of one entry list triple, linear in the length: `c = 1 ∧ e = s₀ a` or
`c = -1 ∧ e + N = s₀ a` entrywise. -/
def labelOk (s0 N : Int) : List Int → List Int → List Int → Bool
  | c :: cs, e :: es, a :: as =>
      ((c == 1 && e == s0 * a) || (c == -1 && e + N == s0 * a)) && labelOk s0 N cs es as
  | _, _, _ => true

theorem labelOk_get {s0 N : Int} : ∀ {cs es as : List Int} (_ : labelOk s0 N cs es as = true)
    (k : Nat) (_ : k < cs.length) (_ : k < es.length) (_ : k < as.length),
    (cs[k]?.getD 0 = 1 ∧ es[k]?.getD 0 = s0 * as[k]?.getD 0) ∨
      (cs[k]?.getD 0 = -1 ∧ es[k]?.getD 0 + N = s0 * as[k]?.getD 0)
  | c :: cs, e :: es, a :: as, h, k, hc, he, ha => by
    simp only [labelOk, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at h
    cases k with
    | zero => simpa using h.1
    | succ k =>
      simpa using labelOk_get h.2 k (by simpa using hc) (by simpa using he) (by simpa using ha)
  | [], _, _, _, _, hc, _, _ => by simp at hc
  | _ :: _, [], _, _, _, _, he, _ => by simp at he
  | _ :: _, _ :: _, [], _, _, _, _, ha => by simp at ha

/-- **Label certificate** from the linear check `labelOk` (one `decide` on a `Bool`, no
`Fin K` enumeration): the statement of the generated `label_rel`. -/
theorem label_rel_of {s0 N : Int} {cs es as : List Int} (K : Nat) (h : labelOk s0 N cs es as = true)
    (hc : K ≤ cs.length) (he : K ≤ es.length) (ha : K ≤ as.length) :
    ∀ k : Fin K,
      (MxxIR.selectInt (Int.ofNat (k : Nat)) cs = 1 ∧
        MxxIR.selectInt (Int.ofNat (k : Nat)) es = s0 * MxxIR.selectInt (Int.ofNat (k : Nat)) as) ∨
      (MxxIR.selectInt (Int.ofNat (k : Nat)) cs = -1 ∧
        MxxIR.selectInt (Int.ofNat (k : Nat)) es + N = s0 * MxxIR.selectInt (Int.ofNat (k : Nat)) as) := by
  intro k
  have hk : (Int.ofNat (k : Nat)).toNat = k := rfl
  simp only [MxxIR.selectInt, hk]
  exact labelOk_get h k (by omega) (by omega) (by omega)

end RefreshTopProof.GenFinal
