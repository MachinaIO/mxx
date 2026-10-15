import GenFinalLink

/-!
# One prime of one refresh part

Part `p` of step `i`'s refresh decodes, for every prime `q` of `Q_0`, the CRT term of its row of
`S_(i+1)` (DIO2-SPEC §6.7, §15): `ζ = b P - ((comb + fresh) + smudge) - H` over `Q_0`, switched to
`q`, lifted centered to `Q_0` and times the CRT basis element `K`. The combination is
`[g | c₁] D - cₓ Y` with `D` the digits of `Q_q A'` and `Y` the digits of `c' lift(R_q)`, where
`R_q = s₁₀ s₉ (G mod q)` (refresh_top's action chain at one modulus, `p = 1`).

`prime_term` is refresh_top's `GenFinalInst.final_inst` for a chain without modulus drops: the
term is a CRT term of the intended row `β A'₀ + A'₁ - T G' + E` of the part, whatever the prime
(the target's residue modulo `q` is that of `T G'` by `tx_rho_gen`).
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof
  RefreshTopProof.GenFinal GenDecode

variable {n : Nat}

/-- A modulus reduction to the same modulus is the identity. -/
theorem modulusReduce_same (hn : 0 < n) {R rows cols : Nat} {x y : ExactMatrix R n rows cols}
    (h : modulusReduceRuns x y) : y = x := by
  have hR : 1 < R := h.1
  funext r c
  rw [modulusReduce_rho hn h r c, rho_self hn hR]

set_option maxHeartbeats 16000000 in
/-- **One prime of one refresh part.** The switched, lifted and scaled `ζ` is the CRT term for
`q` (basis element `K`) of `β A'₀ + A'₁ - T G' + E`. -/
theorem prime_term (hn : 0 < n) {R q Qq h m : Nat} (hRq : R = Qq * q) (hQq : 0 < Qq)
    (hq : 1 < q) (hR : 1 < R)
    {backend : BackendContext} {base digits : Int}
    (L : RegularLayout R) (hL : backend.regularLayout R n = some L) (hLx : L.droppedModuli = 0)
    -- the action chain: `Y = digits (c' lift (s₁₀ s₉ (G' mod q)))`
    {G : ExactMatrix R n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {Gx : ExactMatrix R n 1 m} (hGx : gadgetMatrixRuns backend base digits Gx)
    {Gx' : ExactMatrix R n 1 m} (hGx' : centeredRebaseRuns Gx Gx')
    {w16 : ExactMatrix q n 1 m} (h16 : modulusReduceRuns Gx' w16) {s9 s10 c' k : Int}
    {w21 : ExactMatrix R n 1 m}
    (h21 : centeredRebaseRuns (matrixMulScalarRight (matrixMulScalarRight w16
      (matrixPolynomial [s9])) (matrixPolynomial [s10])) w21)
    {w24 Y : ExactMatrix R n m m}
    (h24 : gadgetDecomposeRuns backend base digits (matrixMulScalarRight w21 (matrixPolynomial [c'])) w24)
    {boundY : Nat} (h25 : compactCenteredRebaseRuns boundY w24 Y)
    {YI : ErrorMatrix n m m} (hYI : Y = reduceMatrix R n m m YI) {BY : Nat} (hYb : CoeffBound YI BY)
    (hc : ((1 : Nat) : Int) * c' = (Qq : Int) * k)
    -- the combination: `D = digits (Q_q A')`
    {A' : ExactMatrix R n 2 m} {w10 D : ExactMatrix R n (m + m) m}
    (h10 : gadgetDecomposeRuns backend base digits
      (matrixMulScalarLeft (matrixPolynomial [(Qq : Int)]) A') w10)
    {boundD : Nat} (h11 : compactCenteredRebaseRuns boundD w10 D)
    {DI : ErrorMatrix n (m + m) m} (hDI : D = reduceMatrix R n (m + m) m DI) {BD : Nat}
    (hDb : CoeffBound DI BD)
    {g c₁ : ExactMatrix R n 1 m} {gc : ExactMatrix R n 1 (m + m)} (h6 : concatColumns g c₁ gc)
    {cx : ExactMatrix R n 1 m}
    -- the rows of the combination
    {s β : ExactPoly R n} {x : ErrorPoly n}
    {Kg K₁ : Fin 2 → ExactMatrix R n 1 m} {Kx : Fin 2 → ExactMatrix R n 1 m}
    {eg e₁ : ErrorMatrix n 1 m} {ex : ErrorMatrix n 1 m} {Beg Be1 Bex : Nat}
    (heg : CoeffBound eg Beg) (he1 : CoeffBound e₁ Be1) (hex : CoeffBound ex Bex)
    (hg : g = maskPart s Kg - β • G + reduceMatrix R n 1 m eg)
    (hc₁ : c₁ = maskPart s K₁ - G + reduceMatrix R n 1 m e₁)
    (hcx : cx = maskPart s Kx - reducePoly R n (((1 : Nat) : Int) * x) • Gx' +
      reduceMatrix R n 1 m ex)
    -- the target's residue: `k x s₉ s₁₀ ≡ T (mod q)`
    {T : ExactPoly R n}
    (hscalar : reducePoly q n (((k : Int) : ErrorPoly n) * x) * (s9 : ExactPoly q n) *
      (s10 : ExactPoly q n) = GenFinal.rho q T)
    -- hint, preimage, fresh, smudge
    {b : ExactMatrix R n 1 h} {Bk : Fin 2 → ExactMatrix R n 1 h} {eb : ErrorMatrix n 1 h} {Beb : Nat}
    (hb : b = maskPart s Bk + reduceMatrix R n 1 h eb) (heb : CoeffBound eb Beb)
    {P : ExactMatrix R n h m} {Pd : ErrorMatrix n h m} {BP : Nat}
    (hP : P = reduceMatrix R n h m Pd) (hPd : CoeffBound Pd BP)
    {w3 : ExactMatrix R n 1 m} (h3 : modulusReduceRuns (matrixMul b P) w3)
    {Kf Ks : Fin 2 → ExactMatrix R n 1 m}
    (hpre : ∀ k', Bk k' * P =
      (Kg k' * topRows D + K₁ k' * bottomRows D - Kx k' * Y) + Kf k' + Ks k')
    {fresh : ExactMatrix R n 1 m} {E ef : ErrorMatrix n 1 m} {Bf : Nat}
    (hf : fresh = maskPart s Kf - ((Qq : Int) : ExactPoly R n) • reduceMatrix R n 1 m E +
      reduceMatrix R n 1 m ef) (hef : CoeffBound ef Bf)
    {smudge Tsm H : ExactMatrix R n 1 m} {es : ErrorMatrix n 1 m} {Bs : Nat}
    (hs : smudge = maskPart s Ks - Tsm + reduceMatrix R n 1 m es) (hes : CoeffBound es Bs)
    {M : ErrorMatrix n 1 m} {BM : Nat} (hTH : Tsm - H = reduceMatrix R n 1 m M)
    (hM : CoeffBound M BM)
    (hnoise : 2 * (h * n * Beb * BP + (m * n * Beg * BD + m * n * Be1 * BD + m * n * Bex * BY) +
      Bf + Bs + BM) < Qq)
    -- the switch, the lift and the CRT term
    {r : ExactMatrix q n 1 m}
    (hsw : modulusSwitchRuns (matrixSub (matrixSub w3 (matrixAdd (matrixAdd
      (matrixSub (matrixMul gc D) (matrixMul cx Y)) fresh) smudge)) H) r)
    {l : ExactMatrix R n 1 m} (hl : centeredRebaseRuns r l) (K : Int) :
    TermOf q K (matrixMulScalarRight l (matrixPolynomial [K]))
      (β • rowOf A' 0 + rowOf A' 1 - T • Gx' + reduceMatrix R n 1 m E) := by
  have hqR : q ∣ R := ⟨Qq, by rw [hRq, Nat.mul_comm]⟩
  have hact := action_chain_gen hn (R := R) (Mx := R) (p := 1) (by rw [one_mul]) hR L hL hLx hc hGx
    hGx' (w := matrixMulScalarRight (matrixMulScalarRight w16 (matrixPolynomial [s9]))
      (matrixPolynomial [s10])) h21 rfl h24 h25 x
  obtain ⟨h0, h1⟩ := decompose_rows_gen hn L hL hLx hR h10 h11 hG
  rw [scaled_rows_gen rfl 0] at h0
  rw [scaled_rows_gen rfl 1] at h1
  obtain ⟨ec, hec, hcomb⟩ := comb_bound_gen hn h6 hDI hDb h0 h1 hYI hYb hact heg he1 hex hg hc₁ hcx
  rw [modulusReduce_same hn h3] at hsw
  refine final_gen hn hRq hQq hq hb heb hP hPd hpre hcomb hec hf hef hs hes hTH hM hnoise rfl hsw
    hl rfl ?_
  -- The residues of `β A'₀ + A'₁ - Tₓ + E` and `β A'₀ + A'₁ - T G' + E` agree modulo `q`.
  apply residueEq_of_rho hn hq
  intro col
  have ht := tx_rho_gen hn hq hR hqR hR hqR (k := k) (x := x) (s9 := s9) (s10 := s10) (G' := Gx')
    h16 rfl rfl hscalar col
  simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul]
  rw [rho_add hn hq hR hqR, rho_add hn hq hR hqR, rho_sub hn hq hR hqR, rho_sub hn hq hR hqR, ht,
    rho_mul hn hq hR hqR]

/-- **The part's row.** The graph's sum `(((t₀ + t₁) + t₂) + t₃) + t₄` of the CRT terms of `v` for
the five primes of `Q = ∏ q_i` (pairwise coprime, CRT basis `K`) is `v`. -/
theorem part_row (hn : 0 < n) {Q m : Nat} (hQ : 1 < Q) (q : Fin 5 → Nat) (hq : ∀ i, 1 < q i)
    (K : Fin 5 → Int) (hK : ∀ i j, K i ≡ (if i = j then 1 else 0) [ZMOD (q j : Int)])
    (hcop : ∀ i j, i ≠ j → Nat.Coprime (q i) (q j)) (hprod : Q = ∏ i, q i)
    {v : ExactMatrix Q n 1 m} {t : Fin 5 → ExactMatrix Q n 1 m}
    (ht : ∀ i, TermOf (q i) (K i) (t i) v) :
    matrixAdd (matrixAdd (matrixAdd (matrixAdd (t 0) (t 1)) (t 2)) (t 3)) (t 4) = v := by
  rw [← crt_terms_sum hn hQ q hq K hK hcop hprod ht, Fin.sum_univ_five]
  rfl

end Dio2Proof
