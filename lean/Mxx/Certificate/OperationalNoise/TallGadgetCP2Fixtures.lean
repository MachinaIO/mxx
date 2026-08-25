import Mxx.Certificate.OperationalNoise.TallSemantics

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallGadgetCP2Fixtures

open Mxx.Certificate.OperationalNoise
open EventReplay
open SchemaV1
open TallSecurity0ABI
open TallSemantics

/-! This is the compressed hand-written slice of the real root-wide frame (frameStart = 0).
    It retains event 6520's P214/E10369 owner and the factor rows used by the reached
    relation/product/fold claims; the 107,567-event enclosing history is intentionally not
    reproduced here. -/

def p214 (expression : Nat) : Owner := ⟨.program ⟨214⟩, ⟨expression⟩⟩

def eventOwner6520 : Owner := p214 10369
def appliedRelationEvent : Nat := 6520
def rootResultEvent : Nat := 6513
def decompositionResultEvent : Nat := 6483
def inputResultEvent : Nat := 6457
def productTransferEvent : Nat := 6516
def monomialTransferEvent : Nat := 6517
def productResultEvent : Nat := 6518
def survivorFoldEvent : Nat := 6512
def invocationEndEvent : Nat := 6523
def gadgetEvent : Owner := p214 6544
def decompositionEvent : Owner := p214 7882
def inputEvent : Owner := p214 6790

def sourceKey : MonomialKey Owner :=
  { centralFactors := [p214 5519, p214 10365]
    orderedFactors := [p214 6544, p214 7882] }

def lhsKey : MonomialKey Owner :=
  { centralFactors := []
    orderedFactors := [p214 6544, p214 7882] }

def relationRhs : Polynomial Owner :=
  [{ coefficient := 1
     key := { centralFactors := []
              orderedFactors := [p214 6790] } }]

def result6513 : Polynomial Owner :=
  [{ coefficient := 1
     key := { centralFactors := [p214 5519]
              orderedFactors := [p214 6770] } },
   { coefficient := -1
     key := { centralFactors := [p214 5519, p214 10365]
              orderedFactors := [p214 6544] } }]

def transfer6517 : Polynomial Owner :=
  [{ coefficient := 1
     key := { centralFactors := []
              orderedFactors := [p214 7882] } }]

def decompositionResult6483 : Polynomial Owner := transfer6517
def inputResult6457 : Polynomial Owner :=
  [{ coefficient := 1
     key := { centralFactors := []
              orderedFactors := [inputEvent] } }]

def product6518 : Polynomial Owner := productPoly result6513 transfer6517 false false

def result6523 : Polynomial Owner :=
  [{ coefficient := 1
     key := { centralFactors := [p214 5519]
              orderedFactors := [p214 6770, p214 7882] } },
   { coefficient := -1
     key := { centralFactors := [p214 5519, p214 10365]
              orderedFactors := [p214 6790] } }]

def relationRaw : Polynomial Owner :=
  [{ coefficient := 1
     key := { centralFactors := [p214 5519]
              orderedFactors := [p214 6770, p214 7882] } },
   { coefficient := -1
     key := { centralFactors := [p214 5519, p214 10365]
              orderedFactors := [p214 6544, p214 7882] } },
   { coefficient := 1
     key := sourceKey },
   { coefficient := -1
     key := { centralFactors := [p214 5519, p214 10365]
              orderedFactors := [p214 6790] } }]

def fixtureEnv : Env Owner := fun _ => 1

theorem product6518_shape : product6518 =
    [{ coefficient := 1
       key := { centralFactors := [p214 5519]
                orderedFactors := [p214 6770, p214 7882] } },
     { coefficient := -1
       key := { centralFactors := [p214 5519, p214 10365]
                orderedFactors := [p214 6544, p214 7882] } }] := by
  rfl

theorem product6518_sound (env : Env Owner) :
    evalPolynomial env product6518 =
      evalPolynomial env result6513 * evalPolynomial env transfer6517 := by
  apply productResultSound env result6513 transfer6517 product6518 false false
  intro term h
  rfl

theorem relation6520_source_context :
    KeyEquivalent sourceKey
      ((relationContext sourceKey sourceKey.centralFactors 0 2).plug lhsKey) := by
  constructor
  · rfl
  · rfl

theorem relation6520_base_congruence :
    evalMonomial fixtureEnv lhsKey % 257 =
      evalPolynomial fixtureEnv relationRhs % 257 := by
  simp [evalMonomial, evalPolynomial, fixtureEnv, lhsKey, relationRhs]

theorem relation6520_agreement :
    CoefficientAgreement result6523
      (relationPoly product6518 sourceKey
        (relationContext sourceKey sourceKey.centralFactors 0 2) (-1) relationRhs) := by
  have shape : relationPoly product6518 sourceKey
      (relationContext sourceKey sourceKey.centralFactors 0 2) (-1) relationRhs =
      relationRaw := by
    rfl
  rw [shape]
  intro term h
  rcases List.mem_append.mp h with h | h
  · simp [result6523] at h
    rcases h with rfl | rfl <;> rfl
  · simp [relationRaw] at h
    rcases h with rfl | rfl | rfl | rfl <;> rfl

theorem relation6520_sound (env : Env Owner)
    (baseRelation : evalMonomial env lhsKey % 257 =
      evalPolynomial env relationRhs % 257) :
    evalPolynomial env result6523 % 257 =
      evalPolynomial env product6518 % 257 := by
  exact relationResultSound 257 env product6518 sourceKey lhsKey
    sourceKey.centralFactors 0 2 (-1) relationRhs result6523
    relation6520_source_context baseRelation relation6520_agreement

theorem preFold6523_sound :
    result6523 = result6523 ∧
      preFoldBound 26 95420416 [0] [0] := by
  exact preFoldSound result6523 result6523 rfl (by
    change 26 ≤ 95420416
    change 26 ≤ 26 + 95420390
    exact Nat.le_add_right _ _)
    (List.Forall₂.cons (Nat.zero_le _) List.Forall₂.nil)

theorem gadget6520_relation_product_prefold_end (env : Env Owner)
    (baseRelation : evalMonomial env lhsKey % 257 =
      evalPolynomial env relationRhs % 257)
    (remainderBound : boundInterprets 257 (.finite 95420416) 0) :
    preFoldBound 26 95420416 [0] [0] ∧
      ValueClaim.Interprets 257 env
        (evalPolynomial env result6513 * evalPolynomial env transfer6517)
        (.exact result6523 (.finite 95420416)) := by
  have productSound := product6518_sound env
  have relationSound := relation6520_sound env baseRelation
  have remainderCongruence :
      (evalPolynomial env result6513 * evalPolynomial env transfer6517 -
        evalPolynomial env result6523) % 257 = 0 % 257 := by
    rw [← productSound]
    exact Int.emod_eq_emod_iff_emod_sub_eq_zero.mp relationSound.symm
  have resultClaim : ValueClaim.Interprets 257 env
      (evalPolynomial env result6513 * evalPolynomial env transfer6517)
      (.exact result6523 (.finite 95420416)) := by
    refine ⟨0, remainderCongruence, ?_⟩
    exact remainderBound
  have prefold := preFoldSound result6523 result6523 rfl
    (summaryActual := 26) (summaryBound := 95420416)
    (survivorContributions := [0]) (survivorBounds := [0]) (by
      change 26 ≤ 95420416
      change 26 ≤ 26 + 95420390
      exact Nat.le_add_right _ _)
    (List.Forall₂.cons (Nat.zero_le _) List.Forall₂.nil)
  have endClaim := invocationEndSound 257 env
    (evalPolynomial env result6513 * evalPolynomial env transfer6517)
    result6523 result6523 (.finite 95420416) (.finite 95420416)
    resultClaim rfl rfl
  exact ⟨prefold.2, endClaim⟩

theorem gadget6520_real_slice :
    preFoldBound 26 95420416 [0] [0] ∧
      ValueClaim.Interprets 257 fixtureEnv
        (evalPolynomial fixtureEnv result6513 * evalPolynomial fixtureEnv transfer6517)
        (.exact result6523 (.finite 95420416)) := by
  apply gadget6520_relation_product_prefold_end fixtureEnv relation6520_base_congruence
  simp [boundInterprets, centeredNorm, centeredCoefficient]

end Mxx.Certificate.OperationalNoise.TallGadgetCP2Fixtures

#print axioms Mxx.Certificate.OperationalNoise.TallSemantics.coefficientAgreement_eval
#print axioms Mxx.Certificate.OperationalNoise.TallSemantics.addResultSound
#print axioms Mxx.Certificate.OperationalNoise.TallSemantics.subResultSound
#print axioms Mxx.Certificate.OperationalNoise.TallSemantics.productResultSound
#print axioms Mxx.Certificate.OperationalNoise.TallSemantics.relationResultSound
#print axioms Mxx.Certificate.OperationalNoise.TallSemantics.preFoldSound
#print axioms Mxx.Certificate.OperationalNoise.TallSemantics.invocationEndSound
#print axioms Mxx.Certificate.OperationalNoise.TallGadgetCP2Fixtures.gadget6520_relation_product_prefold_end
