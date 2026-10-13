import Claim
import RingGswBggLayers

/-!
# One layer of the evaluation and of the ideal circuit

Every slot of a layer of `Stage_evaluate` reads the same gate record as the slot of
`Ideal` at the same layer, so if every slot of the preceding evaluation layer encodes the bit of the
preceding ideal layer, so does every slot of the next one, with the error bound `nextBound`.
-/

namespace RingGswBggProof

open Mxx.Primitives MxxRuntime RingGswBgg

set_option maxHeartbeats 4000000

/-- The ring, encoding width, and gadget of this claim. -/
abbrev Q : Nat := {{Q}}
abbrev N : Nat := {{N}}
abbrev M : Nat := {{M}}
abbrev BASE : Int := {{BASE}}
abbrev DIGITS : Int := {{DIGITS}}

theorem layer_step (hq : 1 < Q) (hn : 0 < N) {backend : BackendContext}
    (hexact : ∀ layout, backend.regularLayout Q N = some layout → layout.droppedModuli = 0)
    {G : ExactMatrix Q N 2 M} (hG : gadgetMatrixRuns backend BASE DIGITS G)
    {s : ExactMatrix Q N 1 2} {sG : ExactMatrix Q N 1 M} (hsG : sG = s * G)
    {E E₀ : Nat} (hE₀ : E₀ ≤ E) {oneV : ExactMatrix Q N 1 M} {oneP : ExactMatrix Q N 2 M}
    {onePt : ExactMatrix Q N 1 1} (hone : Enc s sG oneV oneP onePt E₀) (hone1 : onePt = 1)
    {hashModel tape₁ tape₂ path₁ path₂ i} {params₁ : Stage_evaluate.Params} {params₂ : Ideal.Params}
    (hW : params₁.«max_layer_width» = params₂.«max_layer_width»)
    {act kinds left right cv cp cx nxt bcur bnxt}
    (hinv : ∀ p, Enc s sG (cv p) (cp p) (cx p) E ∧ cx p = bitMatrix (bcur p))
    (heval : Stage_evaluate.sequential_generatedRoot_15 backend hashModel tape₁ path₁ params₁ i
      (cv, (cp, (cx, (act, (oneV, (kinds, (left, (right, (oneP, (onePt, ())))))))))) nxt)
    (hideal : Ideal.sequential_generatedRoot_6 tape₂ path₂ params₂ i
      (bcur, (act, (kinds, (left, (right, ()))))) bnxt) :
    ∀ p, Enc s sG (nxt.1 p) (nxt.2.1 p) (nxt.2.2.1 p) (nextBound M N (BASE.toNat / 2) E) ∧
      nxt.2.2.1 p = bitMatrix (bnxt p) := by
  obtain ⟨⟨a₁, nv, np, nx⟩, hb⟩ := heval
  dsimp only [Stage_evaluate.sequential_generatedRoot_15.body,
    Stage_evaluate.sequential_generatedRoot_15.constraints_0] at hb
  obtain ⟨-, -, hact₁, -, hslots, rfl⟩ := hb
  obtain ⟨⟨a₂, nb⟩, hi⟩ := hideal
  dsimp only [Ideal.sequential_generatedRoot_6.body,
    Ideal.sequential_generatedRoot_6.constraints_0] at hi
  obtain ⟨-, -, hact₂, -, hislots, rfl⟩ := hi
  have hact : a₂ = a₁ := familyGetDynamic_eq hact₂ hact₁
  subst hact
  intro p
  have hs := hslots p
  dsimp only [Stage_evaluate.parallel_sequential_generatedRoot_15_12] at hs
  obtain ⟨w13, w16, w17, w21, w22, w23, w25, w27, w34, w35, w38, w44, w45, w49, w54, w55, hc⟩ := hs
  dsimp only [Stage_evaluate.parallel_sequential_generatedRoot_15_12.constraints_0,
    Stage_evaluate.parallel_sequential_generatedRoot_15_12.constraints_1] at hc
  obtain ⟨-, -, hk₁, -, -, hl₁, -, -, hv17, -, -, hr₁, -, -, hp22, hdec, -, -, hv25, -, -, hx27,
    -, -, -, hsel34, -, -, -, hsel35, -, -, hp38, -, -, -, hsel44, -, -, -, hsel45, -, -, hx49,
    -, -, -, hsel54, -, -, -, hsel55, hout⟩ := hc
  have hj := hislots p
  dsimp only [Ideal.parallel_sequential_generatedRoot_6_7] at hj
  obtain ⟨v15, v26, v27, v33, v34, v44, v45, hd⟩ := hj
  dsimp only [Ideal.parallel_sequential_generatedRoot_6_7.constraints_0] at hd
  obtain ⟨-, -, hk₂, -, -, hl₂, -, -, hb27, -, -, hr₂, -, -, hb34, -, -, -, hsel44', -, -, -,
    hsel45', hout'⟩ := hd
  rw [hW] at hk₁ hl₁ hr₁
  have hk : v15 = w13 := familyGetDynamic_eq hk₂ hk₁
  have hl : v26 = w16 := familyGetDynamic_eq hl₂ hl₁
  have hr : v33 = w21 := familyGetDynamic_eq hr₂ hr₁
  subst hk hl hr
  -- The left operand.
  obtain ⟨pL, hpL, rfl⟩ := hv17
  obtain ⟨pL₁, hpL₁, rfl⟩ := hp38
  obtain ⟨pL₂, hpL₂, rfl⟩ := hx27
  obtain ⟨pL₃, hpL₃, rfl⟩ := hb27
  have e₁ : pL = pL₁ := Fin.ext (by omega)
  have e₂ : pL = pL₂ := Fin.ext (by omega)
  have e₃ : pL = pL₃ := Fin.ext (by omega)
  subst e₁ e₂ e₃
  -- The right operand.
  obtain ⟨pR, hpR, rfl⟩ := hp22
  obtain ⟨pR₁, hpR₁, rfl⟩ := hv25
  obtain ⟨pR₂, hpR₂, rfl⟩ := hx49
  obtain ⟨pR₃, hpR₃, rfl⟩ := hb34
  have f₁ : pR = pR₁ := Fin.ext (by omega)
  have f₂ : pR = pR₂ := Fin.ext (by omega)
  have f₃ : pR = pR₃ := Fin.ext (by omega)
  subst f₁ f₂ f₃
  obtain ⟨hGD, d, hD, hdb⟩ := gadget_exact hq hn hG hdec hexact
  obtain ⟨hL, hxL⟩ := hinv pL
  obtain ⟨hR, hxR⟩ := hinv pR
  obtain ⟨c₁, c₂, c₃, c₄, c₅⟩ := ideal_candidates (bcur pL) (bcur pR)
  rw [c₁, c₂, c₃, c₄, c₅] at hsel44'
  rw [c₁] at hsel45'
  have htwo : (matrixPolynomial [2] : ExactMatrix Q N 1 1) 0 0 = 2 := by
    rw [MxxRuntime.matrixPolynomial_single]; norm_num
  obtain ⟨henc, hbit⟩ := slot_layer hsG hn hE₀ hone hone1 hL hxL hR hxR hGD hD hdb htwo
    hsel34 hsel44 hsel54 hsel44' hsel35 hsel45 hsel55 hsel45'
  simp only [Prod.mk.injEq] at hout
  obtain ⟨rfl, rfl, rfl, -⟩ := hout
  subst hout'
  exact ⟨henc, hbit⟩

end RingGswBggProof
