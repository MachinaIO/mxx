import PeInteger
import PeTail

/-!
# The average-case correctness certificate

On every tape, a run decrypts coefficient `i` correctly unless the linear form `lin ω i` in the key
extension reaches `mT`: the remaining error, `e_m` and the bias `e_b k̃`, is bounded on every tape.
The coefficients of `lin` read only the keys of `preKeys`, so `tapeMeasure_linear_tail` bounds the
probability that some coefficient reaches `mT` by `N · 2 exp(-mT² / (2 σ² E))`.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut MeasureTheory

/-- The error bound of the online output. -/
abbrev BF : Nat := onlineBound {{CF}} ({{MB}} * N * {{CE}} * {{CE}}) {{CF}} {{CE}}

/-- The bound of one bias term `e_b k̃`. -/
def biasB : Nat := {{MB}} * N * {{CE}} * keyCutoff

/-- The budget of the linear form. -/
def mT : Nat := (Q - 8) / 4 - {{CE2}} - 2 * biasB

/-- The output error, clamped to its bound so that every tape has a bounded coefficient. -/
noncomputable def eFcP (model : HashModel) (key : ByteArray) (P : Prims) : Fin 2 → ErrorMatrix N 1 {{D}} := by
  classical
  exact if ∀ j, CoeffBound (eFP model key P j) BF then eFP model key P else 0

/-- The helper error, clamped to its cutoff. -/
noncomputable def ehcP (P : Prims) : Fin 2 → ErrorMatrix N 1 {{D}} := by
  classical
  exact if ∀ j, CoeffBound (P.eh j) {{CE}} then P.eh else 0

/-- The rows that multiply the three extension blocks `v₀`, `v₁`, `v_F`. -/
noncomputable def crowP (model : HashModel) (key : ByteArray) (P : Prims) : Fin 3 → Fin {{M}} → ErrorPoly N :=
  ![fun r ↦ jZ (ehcP P) 0 r, fun r ↦ substitution K hK (jZ (ehcP P) 0 r),
    fun r ↦ jZ (eFcP model key P) 0 r]

/-- The sites of the extension blocks. -/
def keySite : Fin 3 → List Nat := ![[6, 10, 0, 0], [6, 10, 1, 0], [6, 7]]

/-- The tape coordinate of coefficient `k` of row `r` of extension block `p`. -/
def keyOf (x : Fin 3 × Fin {{M}} × Fin N) : SampleKey := ⟨keySite x.1, x.2.1, 0, x.2.2, keyLaw⟩

/-- The coefficient of a tape coordinate in coefficient `i` of the linear form. -/
noncomputable def coefZ (model : HashModel) (key : ByteArray) (ω : SampleTape) (i : Fin N)
    (x : Fin 3 × Fin {{M}} × Fin N) : Int :=
  maskCoefficient (crowP model key (prims ω) x.1 x.2.1) i x.2.2

/-- Coefficient `i` of the linear form in the key extension. -/
noncomputable def lin (model : HashModel) (key : ByteArray) (ω : SampleTape) (i : Fin N) : Int :=
  ∑ x, coefZ model key ω i x * ω (keyOf x)

/-- The requirement accepts exactly the attribute `x₀ = 0`, for which the predicate `x ↦ x₀`
is `0`. -/
theorem requirement_zero {tape : SampleTape} {x : Fin 1 → Int} {out : Bool}
    (h : Requirement_0.generatedRoot tape [9] GeneratedClaim.requirement_0_params x out)
    (ht : out = true) : x 0 = 0 := by
  obtain ⟨w, h1, -, h8, -, h9, -, hsel, hout⟩ := h
  obtain ⟨_, hw8⟩ := familyGet_eq (k := 0) h8
  subst ht
  obtain ⟨pos, hpos, hw16⟩ := hsel
  fin_cases pos
  · simp only [List.get] at hw16
    simp only [Nat.cast_zero] at hpos
    rw [hw8] at hpos
    simpa using hpos.symm
  · simp only [List.get] at hw16
    rw [hw16] at hout
    simp at hout

theorem ideal_eq {tape : SampleTape} {m out : Fin N → Int}
    (h : Ideal.generatedRoot tape [10] GeneratedClaim.ideal_params m out) : out = m := by
  obtain ⟨_, h⟩ := h
  exact h

theorem joinRow_add (x y : Fin 2 → ExactMatrix Q N 1 {{D}}) :
    joinRow (fun j ↦ x j + y j) = joinRow x + joinRow y := by
  funext r p
  simp only [joinRow, Matrix.add_apply, joinedEntry_split]
  split <;> rfl

theorem joinRow_sub (x y : Fin 2 → ExactMatrix Q N 1 {{D}}) :
    joinRow (fun j ↦ x j - y j) = joinRow x - joinRow y := by
  funext r p
  simp only [joinRow, Matrix.sub_apply, joinedEntry_split]
  split <;> rfl

theorem joinRow_smul (a : ExactPoly Q N) (x : Fin 2 → ExactMatrix Q N 1 {{D}}) :
    joinRow (fun j ↦ a • x j) = a • joinRow x := by
  funext r p
  simp only [joinRow, Matrix.smul_apply, joinedEntry_split]
  split <;> rfl

/-- The joined row of an explicit encoding. -/
theorem joinRow_encRow (sm : ErrorPoly N) (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 {{D}})
    (t : Fin 2 → ExactMatrix Q N 1 {{D}}) (e : Fin 2 → ErrorMatrix N 1 {{D}}) :
    joinRow (encRow sm A t e) = reducePoly Q N sm • joinRow (A 0) + joinRow (A 1) - joinRow t +
      joinRow (fun j ↦ reduceMatrix Q N 1 {{D}} (e j)) := by
  have h : encRow sm A t e = fun j ↦ ((reducePoly Q N sm • A 0 j + A 1 j) - t j) +
      reduceMatrix Q N 1 {{D}} (e j) := rfl
  rw [h, joinRow_add, joinRow_sub, joinRow_add, joinRow_smul]

/-- The joined target of the helper. -/
theorem joinRow_helperTarget (sm : ErrorPoly N) :
    joinRow (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
      unitPair (substitution K hK sm) j) • g0) =
    σp (reducePoly Q N sm) • gZero g0 + zeroG g0 := by
  funext r p
  simp only [joinRow, gZero, zeroG, Matrix.add_apply, Matrix.smul_apply, joinedEntry_split]
  have h1 : (((1 : Nat) : Int) : ErrorPoly N) = 1 := by simp
  split
  · simp only [h1, one_mul, unitPair, Matrix.cons_val_zero, Matrix.zero_apply, add_zero, σp,
      reducePoly_substitution]
  · simp only [h1, one_mul, unitPair, Matrix.cons_val_one, Matrix.cons_val_zero, Matrix.zero_apply,
      smul_zero, zero_add, map_one, one_smul]

/-- The error of the online output, from its encoding relation. -/
theorem eF_props {model : HashModel} {key : ByteArray} {ω : SampleTape}
    (hrel : MxxExponentLut.Rel (secOf ω) (AFM model key) (fun _ ↦ 0) (cFOf model key ω) BF) :
    (∀ j, CoeffBound (eFOf model key ω j) BF) ∧ ∀ j, cFOf model key ω j =
      reducePoly Q N (secOf ω) • AFM model key 0 j + AFM model key 1 j +
        reduceMatrix Q N 1 {{D}} (eFOf model key ω j) := by
  obtain ⟨e, he, hc⟩ := hrel
  have hBF : 2 * BF < Q := by decide
  have heq : ∀ j, eFOf model key ω j = e j := by
    intro j
    rw [eFOf_eq, hc j]
    simp only [sub_zero, add_sub_cancel_left]
    exact liftMatrix_reduce hQ hN (he j) hBF
  refine ⟨fun j ↦ by rw [heq j]; exact he j, fun j ↦ ?_⟩
  rw [hc j, heq j]
  simp

/-- On a tape whose errors lie within their bounds, `lin` is the coefficient of
`e_h v₀ + σ_K(e_h) v₁ + e_F v_F`. -/
theorem lin_eq {model : HashModel} {key : ByteArray} {ω : SampleTape}
    (heh : ehcP (prims ω) = ehOf ω) (heF : eFcP model key (prims ω) = eFOf model key ω) (i : Fin N) :
    lin model key ω i = ((jZ (ehOf ω) * vAt ω [6] 0) 0 0).coeff i +
      ((substituteMatrix K hK (jZ (ehOf ω)) * vAt ω [6] 1) 0 0).coeff i +
      ((jZ (eFOf model key ω) * vFAt ω [6]) 0 0).coeff i := by
  unfold vAt vFAt
  rw [lin_coeff, lin_coeff, lin_coeff]
  unfold lin coefZ
  rw [Fintype.sum_prod_type, Fin.sum_univ_three]
  simp only [Fintype.sum_prod_type, crowP, heh, heF, keyOf, keySite]
  rfl

set_option maxHeartbeats 64000000 in
/-- Every run decrypts coefficient `i` of the message up to the error `e`, which is bounded by
`e_m`, the bias, the message bit, and the linear form. -/
theorem run_decode {model : HashModel} {external : GeneratedClaim.ExternalInputs} {ω : SampleTape}
    {exec : GeneratedClaim.Execution} (hruns : GeneratedClaim.Runs model external ω exec) :
    exec.«ideal» = external.input_2 ∧ (∀ k, external.input_2 k = 0 ∨ external.input_2 k = 1) ∧
    ∀ i : Fin N, ∃ e : Int, |e| ≤ {{CE1}} + 2 * biasB + |lin model external.input_0 ω i| ∧
      exec.«stage_8» i = ((2 * ((((-(((Q / 2 : Nat) : Int) * external.input_2 i) + e : Int) : ZMod Q).val :
        Nat) : Int) + (Q : Int) / 2) / (Q : Int)) % 2 := by
  obtain ⟨⟨B0, td, ⟨⟩⟩, s1, s2, ⟨C00, C01, C10, C11, b, cm, ⟨⟩⟩,
    ⟨za0, za1, hf0, hf1, c0a, c0b, z0, z1, ⟨⟩⟩,
    ⟨M00, M01, M10, M11, T0, T1, a00, a01, a10, a11, a00', a01', a10', a11', ⟨⟩⟩,
    ⟨k0, k1, vfam, vF, ⟨⟩⟩, ⟨o0, o1, o2, o3, cF0, cF1, ⟨⟩⟩, out, rq, idl⟩ := exec
  obtain ⟨hvalid, h0, h1, h2, h3, h4, h5, h6, h7, h8, hreq, hideal, hrq⟩ := hruns
  dsimp only at h0 h1 h2 h3 h4 h5 h6 h7 h8 hreq hideal hrq ⊢
  obtain ⟨-, -, hm, ⟨h31, h32⟩, ⟨h41, h42⟩⟩ := hvalid
  have hx0 := requirement_zero hreq hrq
  have hidl := ideal_eq hideal
  obtain ⟨hB, htdB, hkind⟩ := setup_explicit h0
  have hs1 : s1 = B0 := by
    obtain ⟨_, h⟩ := h1
    exact h
  obtain ⟨hs2, hsn⟩ := payload_explicit h2
  have hc1 : external.input_3 = 1 := by omega
  have he0 : external.input_4 = 0 := by omega
  dsimp only at hB htdB hkind
  subst hs1 hs2 hB
  obtain ⟨hCt, hCb, hb, hcm, heb, hzb, hem⟩ := prelude_link h3 hx0
  dsimp only at hCt hCb hb hcm
  rw [hc1, he0] at h4
  obtain ⟨hc0, hhf, hz, hza⟩ := encrypt_link h4 hCt hCb
  dsimp only at hc0 hhf hz hza
  obtain ⟨hb0, hbh, hbz, hbza⟩ := encrypt_err_bounds h4
  obtain ⟨hMM, hT0, hT1, hAF, hAF'⟩ := preprocess_link h5
  dsimp only at hMM hT0 hT1 hAF hAF'
  have hAF0 : ![a00', a01'] = AFM model external.input_0 0 := congrFun hAF' 0
  have hAF1 : ![a10', a11'] = AFM model external.input_0 1 := congrFun hAF' 1
  -- Key generation.
  obtain ⟨hvF, hv, hbvF, hbv, hk0, hk1, ⟨k0', hk0', hbk0⟩, ⟨k1', hk1', hbk1⟩⟩ :=
    keygen_explicit h6 hkind g0_runs
  dsimp only at hvF hv hk0 hk1 hk0' hk1'
  rw [htdB] at hk0 hk1
  subst hv hvF
  -- Online evaluation and its error.
  obtain ⟨hcF, hD, hDt, hDK, hV1, hVK⟩ := online_link h7 hc0 hCt hCb hz hhf hza hMM hT0 hT1
  dsimp only at hcF
  have hrel := online_error (model := model) (key := external.input_0) (ω := ω) hb0 hbh hbz hbza heb hzb
    hD hDt hDK hV1 hVK
  obtain ⟨hbF, hcFe⟩ := eF_props hrel
  -- Decryption.
  have hdec := decode_explicit h8 g0_runs
  set s := reducePoly Q N (secOf ω) with hs
  have hW : b - anchorSecond model external.input_0 = s • BOf ω + reduceMatrix Q N 1 {{MB}} (ebAt ω [3]) := by
    rw [hb, bOf_eq]
    abel
  have hDe : joinRow ![hf0 0, hf1 0] - joinRow (helperKey model external.input_0 1) =
      s • joinRow (helperKey model external.input_0 0) - σp s • gZero g0 - zeroG g0 +
        joinRow (fun j ↦ reduceMatrix Q N 1 {{D}} (ehOf ω j)) := by
    rw [hhf, hrowOf_eq, joinRow_encRow, joinRow_helperTarget]
    abel
  have hCF : joinRow ![cF0, cF1] - joinRow ![a10', a11'] =
      s • joinRow ![a00', a01'] + joinRow (fun j ↦ reduceMatrix Q N 1 {{D}} (eFOf model external.input_0 ω j)) := by
    rw [hcF, hAF0, hAF1]
    have : cFOf model external.input_0 ω = fun j ↦ (s • AFM model external.input_0 0 j +
        AFM model external.input_0 1 j) + reduceMatrix Q N 1 {{D}} (eFOf model external.input_0 ω j) :=
      funext hcFe
    rw [this, joinRow_add, joinRow_add, joinRow_smul]
    abel
  have hcm' : cm - messageRow model external.input_0 1 = s • messageRow model external.input_0 0 +
      reduceMatrix Q N 1 1 (emAt ω [3]) +
      ((halfQ : ExactPoly Q N) • (fun _ _ ↦ polynomialOfCoefficients external.input_2) : ExactMatrix Q N 1 1) := by
    rw [hcm]
    abel
  have hphase := phase_eq g0_runs hW hk0 hk1 hDe hCF hcm'
  refine ⟨hidl, fun k ↦ ?_, fun i ↦ ?_⟩
  · have := hm k
    omega
  -- The clamped errors are the errors of the run.
  have heh : ehcP (prims ω) = ehOf ω := by
    have hb : ∀ j, CoeffBound ((prims ω).eh j) {{CE}} := fun j ↦ hbh j
    simp only [ehcP, if_pos hb]
    rfl
  have heF : eFcP model external.input_0 (prims ω) = eFOf model external.input_0 ω := by
    have hb : ∀ j, CoeffBound (eFP model external.input_0 (prims ω) j) BF := hbF
    simp only [eFcP, if_pos hb]
    rfl
  have hlin := lin_eq heh heF i
  -- The phase of coefficient `i`.
  simp only [σK] at hphase
  have hd : thresholdDecode 2 8192 (((i : Fin N) : Nat) : Int)
      (reduceMatrix Q N 1 1 (emAt ω [3]) +
        ((halfQ : ExactPoly Q N) • (fun _ _ ↦ polynomialOfCoefficients external.input_2) : ExactMatrix Q N 1 1) -
      (reduceMatrix Q N 1 {{MB}} (ebAt ω [3]) * k0 +
        substituteMatrix K hK (reduceMatrix Q N 1 {{MB}} (ebAt ω [3]) * k1) +
        joinRow (fun j ↦ reduceMatrix Q N 1 {{D}} (ehOf ω j)) * reduceMatrix Q N {{M}} 1 (vAt ω [6] 0) +
        substituteMatrix K hK (joinRow (fun j ↦ reduceMatrix Q N 1 {{D}} (ehOf ω j))) *
          reduceMatrix Q N {{M}} 1 (vAt ω [6] 1) +
        joinRow (fun j ↦ reduceMatrix Q N 1 {{D}} (eFOf model external.input_0 ω j)) *
          reduceMatrix Q N {{M}} 1 (vFAt ω [6]))) (out i) := by
    exact (congrArg (fun p ↦ thresholdDecode 2 8192 (((i : Fin N) : Nat) : Int) p (out i)) hphase).mp (hdec i)
  rw [hk0', hk1'] at hd
  obtain ⟨-, -, -, -, idx, hidx, -, hout⟩ := hd
  have hidx' : i = idx := Fin.ext (by exact_mod_cast hidx.symm)
  subst hidx'
  have hentry := phase_entry (em := emAt ω [3]) (m := external.input_2) (eb := ebAt ω [3]) (k0 := k0')
    (k1 := k1') (eh := ehOf ω) (eF := eFOf model external.input_0 ω) (v0 := vAt ω [6] 0)
    (v1 := vAt ω [6] 1) (vF := vFAt ω [6])
  simp only [σK] at hentry
  rw [hentry, reducePoly_coeff hQ hN] at hout
  set m := external.input_2 with hmdef
  set emi := ((emAt ω [3]) 0 0).coeff i
  set b0 := ((ebAt ω [3] * k0') 0 0).coeff i
  set b1 := (substitution K hK ((ebAt ω [3] * k1') 0 0)).coeff i
  set L := lin model external.input_0 ω i
  have hX : (emAt ω [3] 0 0 + (halfQ : ErrorPoly N) * intPoly m -
      ((ebAt ω [3] * k0') 0 0 + substitution K hK ((ebAt ω [3] * k1') 0 0) +
        (jZ (ehOf ω) * vAt ω [6] 0) 0 0 + (substituteMatrix K hK (jZ (ehOf ω)) * vAt ω [6] 1) 0 0 +
        (jZ (eFOf model external.input_0 ω) * vFAt ω [6]) 0 0)).coeff i =
      emi + halfQ * m i - (b0 + b1 + L) := by
    rw [hlin]
    simp only [Negacyclic.coeff_add, Negacyclic.coeff_sub, halfQ_coeff]
    ring
  rw [hX] at hout
  have hmi := hm i
  refine ⟨emi - b0 - b1 - L - m i, ?_, ?_⟩
  · have h1 : emi.natAbs ≤ {{CE}} := hem 0 0 i
    have h2 : b0.natAbs ≤ biasB := coeffBound_mul hN heb hbk0 0 0 i
    have h3 : b1.natAbs ≤ biasB :=
      (coeff_natAbs_le_polyNorm _ i).trans ((polyNorm_substitution_le hN hK hKcoprime _).trans
        (coeffBound_polyNorm (coeffBound_mul hN heb hbk1) 0 0))
    have a1 : |emi| ≤ {{CE}} := by rw [Int.abs_eq_natAbs]; exact_mod_cast h1
    have a2 : |b0| ≤ (biasB : Int) := by rw [Int.abs_eq_natAbs]; exact_mod_cast h2
    have a3 : |b1| ≤ (biasB : Int) := by rw [Int.abs_eq_natAbs]; exact_mod_cast h3
    obtain ⟨c1, c2⟩ := abs_le.mp a1
    obtain ⟨c3, c4⟩ := abs_le.mp a2
    obtain ⟨c5, c6⟩ := abs_le.mp a3
    have c7 := neg_abs_le L
    have c8 := le_abs_self L
    obtain ⟨c9, c10⟩ := hmi
    clear_value emi b0 b1 L m
    exact abs_le.mpr ⟨by linarith only [c1, c4, c6, c8, c10], by linarith only [c2, c3, c5, c7, c9]⟩
  · have hq2 : ((Q / 2 : Nat) : Int) = halfQ := by norm_num
    have hcast : ((emi + halfQ * m i - (b0 + b1 + L) : Int) : ZMod Q) =
        ((-(((Q / 2 : Nat) : Int) * m i) + (emi - b0 - b1 - L - m i) : Int) : ZMod Q) := by
      rw [ZMod.intCast_eq_intCast_iff_dvd_sub]
      refine ⟨-(m i), ?_⟩
      rw [hq2]
      norm_num
      ring
    rw [hcast] at hout
    exact hout

/-- One coefficient of `w` per term: `|maskCoefficient w i k| ≤ B` for `|w_j| ≤ B`. -/
theorem maskCoefficient_le {w : ErrorPoly N} {B : Nat} (hw : ∀ j, (w.coeff j).natAbs ≤ B) (i k : Fin N) :
    (maskCoefficient w i k).natAbs ≤ B := by
  let j0 : Fin N := ⟨((i : Nat) + N - k) % N, Nat.mod_lt _ (by decide)⟩
  have hsum : maskCoefficient w i k = w.coeff j0 * negSign N k j0 i := by
    unfold maskCoefficient
    refine Finset.sum_eq_single j0 (fun j _ hj ↦ ?_) (by simp)
    have hne : ((k : Nat) + j) % N ≠ i := by
      intro h
      apply hj
      apply Fin.ext
      simp only [j0]
      have := k.isLt
      have := j.isLt
      have := i.isLt
      simp only [show N = 8192 from rfl] at *
      omega
    simp [negSign, hne]
  rw [hsum, Int.natAbs_mul]
  have hs : (negSign N k j0 i).natAbs ≤ 1 := by
    unfold negSign
    split_ifs <;> simp
  calc (w.coeff j0).natAbs * (negSign N k j0 i).natAbs ≤ B * 1 := Nat.mul_le_mul (hw j0) hs
    _ = B := mul_one B

/-- Every entry of a joined row of a clamped error is within its bound. -/
theorem jZ_bound {e : Fin 2 → ErrorMatrix N 1 {{D}}} {B : Nat} (he : ∀ j, CoeffBound (e j) B) (r : Fin {{M}})
    (k : Fin N) : ((jZ e 0 r).coeff k).natAbs ≤ B := by
  simp only [jZ, joinedEntry_split']
  split
  · exact he 0 0 _ k
  · exact he 1 0 _ k

theorem ehcP_bound (P : Prims) (j : Fin 2) : CoeffBound (ehcP P j) {{CE}} := by
  unfold ehcP
  split
  · next h => exact h j
  · intro r c k
    simp

theorem eFcP_bound (model : HashModel) (key : ByteArray) (P : Prims) (j : Fin 2) :
    CoeffBound (eFcP model key P j) BF := by
  unfold eFcP
  split
  · next h => exact h j
  · intro r c k
    simp

/-- The bound of the rows of extension block `p`. -/
def rowBound : Fin 3 → Nat := ![{{CE}}, {{CE}}, BF]

theorem crowP_bound (model : HashModel) (key : ByteArray) (P : Prims) (p : Fin 3) (r : Fin {{M}})
    (k : Fin N) : ((crowP model key P p r).coeff k).natAbs ≤ rowBound p := by
  fin_cases p
  · exact jZ_bound (ehcP_bound P) r k
  · exact (coeff_natAbs_le_polyNorm _ k).trans ((polyNorm_substitution_le hN hK hKcoprime _).trans
      (polyNorm_le_of_coeff fun k ↦ jZ_bound (ehcP_bound P) r k))
  · exact jZ_bound (eFcP_bound model key P) r k

/-- The width of the key extension. -/
abbrev keySigma : Rat := ({{KEYSIGMA}})

/-- The energy bound of a coefficient of the linear form. -/
def energyB : Nat := {{M}} * N * ({{CE}} * {{CE}} + {{CE}} * {{CE}} + BF * BF)

theorem numeric_margin : 240 * ({{SIGMABOUND}} * {{SIGMABOUND}}) * energyB ≤ mT * mT := by
  unfold energyB mT biasB BF onlineBound keyCutoff
  decide

theorem keySigma_le : (keySigma : ℝ) ≤ {{SIGMABOUND}} := by
  have : keySigma ≤ {{SIGMABOUND}} := by unfold keySigma; norm_num
  exact_mod_cast this

theorem keySigma_pos : (0 : ℝ) < keySigma := by
  have : (0 : Rat) < keySigma := by unfold keySigma; norm_num
  exact_mod_cast this

/-- `N · 2 exp(-mT² / (2 σ² E)) ≤ 2^-100`. -/
theorem tail_numeric :
    (N : ENNReal) * ENNReal.ofReal (2 * Real.exp (-(mT : ℝ) ^ 2 / (2 * ((keySigma : ℝ) ^ 2 * energyB)))) ≤
      (2 : ENNReal)⁻¹ ^ 100 := by
  have hE : (0 : ℝ) < energyB := by
    have : 0 < energyB := by unfold energyB; positivity
    exact_mod_cast this
  have hσ2 : (keySigma : ℝ) ^ 2 ≤ {{SIGMABOUND}} ^ 2 := pow_le_pow_left₀ keySigma_pos.le keySigma_le 2
  have hm : (240 : ℝ) * ({{SIGMABOUND}} * {{SIGMABOUND}}) * energyB ≤ (mT : ℝ) * mT := by
    exact_mod_cast numeric_margin
  have hx : 120 ≤ (mT : ℝ) ^ 2 / (2 * ((keySigma : ℝ) ^ 2 * energyB)) := by
    have hpos : 0 < 2 * ((keySigma : ℝ) ^ 2 * energyB) := by
      have := keySigma_pos
      positivity
    rw [le_div_iff₀ hpos]
    nlinarith
  have hexp : Real.exp (-(mT : ℝ) ^ 2 / (2 * ((keySigma : ℝ) ^ 2 * energyB))) ≤ Real.exp (-120) := by
    apply Real.exp_le_exp.mpr
    rw [neg_div]
    linarith
  have h120 : Real.exp (-120) ≤ ((2 : ℝ) ^ 120)⁻¹ := by
    rw [Real.exp_neg]
    apply inv_anti₀ (by positivity)
    have h2 : (2 : ℝ) ≤ Real.exp 1 := by
      have := Real.add_one_le_exp (1 : ℝ)
      linarith
    calc (2 : ℝ) ^ 120 ≤ Real.exp 1 ^ 120 := pow_le_pow_left₀ (by norm_num) h2 120
      _ = Real.exp 120 := by rw [← Real.exp_nat_mul]; norm_num
  have hreal : (N : ℝ) * (2 * Real.exp (-(mT : ℝ) ^ 2 / (2 * ((keySigma : ℝ) ^ 2 * energyB)))) ≤
      (2⁻¹ : ℝ) ^ 100 := by
    calc (N : ℝ) * (2 * Real.exp (-(mT : ℝ) ^ 2 / (2 * ((keySigma : ℝ) ^ 2 * energyB))))
        ≤ (N : ℝ) * (2 * ((2 : ℝ) ^ 120)⁻¹) := by gcongr; exact hexp.trans h120
      _ ≤ (2⁻¹ : ℝ) ^ 100 := by norm_num
  have hlhs : (N : ENNReal) * ENNReal.ofReal (2 * Real.exp (-(mT : ℝ) ^ 2 /
      (2 * ((keySigma : ℝ) ^ 2 * energyB)))) = ENNReal.ofReal ((N : ℝ) * (2 * Real.exp (-(mT : ℝ) ^ 2 /
      (2 * ((keySigma : ℝ) ^ 2 * energyB))))) := by
    rw [ENNReal.ofReal_mul (p := (N : ℝ)) (by positivity), ENNReal.ofReal_natCast]
  have hrhs : (2 : ENNReal)⁻¹ ^ 100 = ENNReal.ofReal ((2⁻¹ : ℝ) ^ 100) := by
    rw [ENNReal.ofReal_pow (by norm_num), ENNReal.ofReal_inv_of_pos (by norm_num)]
    norm_num
  rw [hlhs, hrhs]
  exact ENNReal.ofReal_le_ofReal hreal

theorem keyOf_injective : Function.Injective keyOf := by
  intro ⟨p, r, k⟩ ⟨p', r', k'⟩ h
  simp only [keyOf, SampleKey.mk.injEq] at h
  obtain ⟨hs, hr, -, hk, -⟩ := h
  have hp : p = p' := by
    fin_cases p <;> fin_cases p' <;> simp_all [keySite]
  subst hp
  simp only [Prod.mk.injEq, true_and]
  exact ⟨Fin.ext hr, Fin.ext hk⟩

theorem keyOf_not_mem (x : Fin 3 × Fin {{M}} × Fin N) : keyOf x ∉ preKeys := by
  obtain ⟨p, r, k⟩ := x
  apply not_mem_preKeys
  fin_cases p <;> rfl

theorem margin_lt : 4 * ({{CE1}} + 2 * biasB + mT) + 4 < Q := by
  unfold mT biasB keyCutoff
  decide

set_option maxHeartbeats 64000000 in
/-- The average-case correctness claim of `{{ID}}`: for every hash model and
external inputs, the tapes with a run whose decrypted message differs from the ideal one have
measure at most `2^-100`. -/
theorem certificate : GeneratedClaim.CorrectnessClaim := by
  intro model external
  let c : SampleTape → Fin N → (Fin 3 × Fin {{M}} × Fin N) → ℝ := fun ω i x ↦ (coefZ model external.input_0 ω i x : ℝ)
  have hsub : {tape | ∃ execution, GeneratedClaim.Runs model external tape execution ∧
      ¬ (execution.«stage_8» = execution.«ideal»)} ⊆
      {ω | ∃ i, (mT : ℝ) ≤ |∑ x, c ω i x * (ω (keyOf x) : ℝ)|} := by
    rintro ω ⟨exec, hruns, hwrong⟩
    by_contra hnot
    simp only [Set.mem_setOf_eq, not_exists, not_le] at hnot
    apply hwrong
    obtain ⟨hidl, hbits, hdec⟩ := run_decode hruns
    funext i
    obtain ⟨e, he, hout⟩ := hdec i
    rw [hout, hidl]
    have hlin : |(lin model external.input_0 ω i : ℝ)| < mT := by
      have := hnot i
      simpa [c, lin, Int.cast_sum, Int.cast_mul] using this
    have hlinZ : |lin model external.input_0 ω i| < (mT : Int) := by exact_mod_cast hlin
    apply threshold_decode_value hQ hQodd (hbits i)
    have hmt : (4 * ({{CE1}} + 2 * biasB + mT) + 4 : Int) < Q := by exact_mod_cast margin_lt
    have he' := he
    linarith only [he', hlinZ, hmt, abs_nonneg e]
  have hdep : ∀ i, DependsOn (fun ω ↦ c ω i) (preKeys : Set SampleKey) := by
    intro i ω ω' h
    funext x
    simp only [c, coefZ]
    rw [prims_congr fun k hk ↦ h k (Finset.mem_coe.mpr hk)]
  have hE : (0 : ℝ) < energyB := by
    have : 0 < energyB := by unfold energyB; positivity
    exact_mod_cast this
  have henergy : ∀ ω i, ∑ x, c ω i x ^ 2 ≤ (energyB : ℝ) := by
    intro ω i
    have hterm : ∀ x : Fin 3 × Fin {{M}} × Fin N, c ω i x ^ 2 ≤ ((rowBound x.1 : Nat) : ℝ) ^ 2 := by
      intro x
      have hb := maskCoefficient_le (crowP_bound model external.input_0 (prims ω) x.1 x.2.1) i x.2.2
      have habs : |c ω i x| ≤ ((rowBound x.1 : Nat) : ℝ) := by
        simp only [c, coefZ]
        rw [← Int.cast_abs, Int.abs_eq_natAbs]
        exact_mod_cast hb
      calc c ω i x ^ 2 = |c ω i x| ^ 2 := (sq_abs _).symm
        _ ≤ ((rowBound x.1 : Nat) : ℝ) ^ 2 := pow_le_pow_left₀ (abs_nonneg _) habs 2
    calc ∑ x, c ω i x ^ 2 ≤ ∑ x : Fin 3 × Fin {{M}} × Fin N, ((rowBound x.1 : Nat) : ℝ) ^ 2 :=
          Finset.sum_le_sum fun x _ ↦ hterm x
      _ = (energyB : ℝ) := by
        rw [Fintype.sum_prod_type, Fin.sum_univ_three]
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_prod, Fintype.card_fin, nsmul_eq_mul,
          rowBound, energyB]
        push_cast
        ring
  calc tapeMeasure {tape | ∃ execution, GeneratedClaim.Runs model external tape execution ∧
        ¬ (execution.«stage_8» = execution.«ideal»)}
      ≤ tapeMeasure {ω | ∃ i, (mT : ℝ) ≤ |∑ x, c ω i x * (ω (keyOf x) : ℝ)|} := measure_mono hsub
    _ ≤ (N : ENNReal) * ENNReal.ofReal (2 * Real.exp (-(mT : ℝ) ^ 2 /
          (2 * ((keySigma : ℝ) ^ 2 * energyB)))) :=
        MxxPe.tapeMeasure_linear_tail keyOf keyOf_injective (sigma := keySigma)
          (by unfold keySigma; norm_num) (cutoff := {{KEYCUTOFF}}) (fun _ ↦ rfl) c preKeys keyOf_not_mem
          hdep hE henergy (by positivity)
    _ ≤ (2 : ENNReal)⁻¹ ^ 100 := tail_numeric

/-- info: 'PeProof.certificate' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms certificate

end PeProof
