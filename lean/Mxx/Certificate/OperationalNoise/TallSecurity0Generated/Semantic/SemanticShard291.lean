import Mxx.Certificate.OperationalNoise.TallSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard291

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

def shardIndex : Nat := 291
def shardStartEvent : Nat := 74496
def shardEndEvent : Nat := 74752
def rawSemanticCount : Nat := 177
def rawBoundTransferCount : Nat := 96
def rawResultCount : Nat := 57
def rawRelationCount : Nat := 0
def rawSurvivorFoldCount : Nat := 22
def rawPreFoldCount : Nat := 1
def rawInvocationEndCount : Nat := 1
def canonicalWork : Nat := 37

namespace Operation0
def selectedEvent : Nat := 74725
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18556⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18555⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨18555⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 74723
def selectedLeftResultEvent : Nat := 74720
def selectedRightResultEvent : Nat := 74718
def selectedResultEvent : Nat := 74725
def leftScalar : Bool := false
def rightScalar : Bool := false

theorem resultAgreement : CanonicalAgreement output (productPoly left right leftScalar rightScalar) := by
  decide +kernel

theorem resultSound (env : Env Owner) :
    evalPolynomial env output = evalPolynomial env left * evalPolynomial env right := by
  exact productCanonicalResultSound env left right output leftScalar rightScalar resultAgreement

theorem selectedLeftResultAt : (history.lookup selectedLeftResultEvent).map AnnotatedEvent.event = some (.resultExact leftOwner leftRaw leftSummary) := by
  rfl

theorem selectedRightResultAt : (history.lookup selectedRightResultEvent).map AnnotatedEvent.event = some (.resultExact rightOwner rightRaw rightSummary) := by
  rfl

theorem selectedResultAt : (history.lookup selectedResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner outputRaw outputSummary) := by
  rfl

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 74721 .coefficient) (.predecessor 1 74722 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation0

namespace Bound0
def selectedEvent : Nat := 74727
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18556⟩⟩
def rootResultEvent : Nat := 74725
def prefoldEvent : Nat := 74726
def endEvent : Nat := 74727
def survivorEvents : List Nat := [74170, 74194, 74218, 74242, 74266, 74290, 74314, 74338, 74362, 74386, 74410, 74434, 74458, 74482, 74506, 74530, 74554, 74578, 74596, 74598, 74605, 74612, 74619, 74626, 74633, 74640, 74647, 74654, 74661, 74668, 74675, 74682, 74689, 74696, 74703, 74710]
def rootRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩] } }]
def prefoldRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩] } }]
def endRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩] } }]
def rootTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩] } }]
def prefoldTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩] } }]
def endTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩] } }]
def rootSummary : Bound := .exactZero
def prefoldSummary : Bound := .exactZero
def endSummary : Bound := .exactZero
def rootBound : Nat := 0
def prefoldBound : Nat := 0
def survivorContributionsChunk0 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk0 : List Nat := [74169, 74193, 74217, 74241, 74265, 74289, 74313, 74337, 74361, 74385, 74409, 74433, 74457, 74481, 74505, 74529]
theorem survivorBoundsSoundChunk0 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk0 survivorBoundsChunk0 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk1 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk1 : List Nat := [74553, 74577, 74595, 74597, 74604, 74611, 74618, 74625, 74632, 74639, 74646, 74653, 74660, 74667, 74674, 74681]
theorem survivorBoundsSoundChunk1 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk1 survivorBoundsChunk1 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          constructor
          · omega
          ·
            constructor
            · omega
            ·
              constructor
              · omega
              ·
                constructor
                · omega
                ·
                  constructor
                  · omega
                  ·
                    constructor
                    · omega
                    ·
                      constructor
                      · omega
                      ·
                        constructor
                        · omega
                        ·
                          constructor
                          · omega
                          ·
                            constructor
                            · omega
                            ·
                              constructor
                              · omega
                              ·
                                constructor
                                · omega
                                ·
                                  exact List.Forall₂.nil

def survivorContributionsChunk2 : List Nat := [1, 1, 1, 1]
def survivorBoundsChunk2 : List Nat := [74688, 74695, 74702, 74709]
theorem survivorBoundsSoundChunk2 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsChunk2 survivorBoundsChunk2 :=
by
  constructor
  · omega
  ·
    constructor
    · omega
    ·
      constructor
      · omega
      ·
        constructor
        · omega
        ·
          exact List.Forall₂.nil

def survivorContributionsTree0_0 : List Nat := survivorContributionsChunk0 ++ survivorContributionsChunk1
def survivorBoundsTree0_0 : List Nat := survivorBoundsChunk0 ++ survivorBoundsChunk1
theorem survivorBoundsSoundTree0_0 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree0_0 survivorBoundsTree0_0 := by
  exact forall₂_append survivorBoundsSoundChunk0 survivorBoundsSoundChunk1
def survivorContributionsTree1_0 : List Nat := survivorContributionsTree0_0 ++ survivorContributionsChunk2
def survivorBoundsTree1_0 : List Nat := survivorBoundsTree0_0 ++ survivorBoundsChunk2
theorem survivorBoundsSoundTree1_0 : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributionsTree1_0 survivorBoundsTree1_0 := by
  exact forall₂_append survivorBoundsSoundTree0_0 survivorBoundsSoundChunk2
def survivorContributions : List Nat := survivorContributionsTree1_0
def survivorBounds : List Nat := survivorBoundsTree1_0
theorem survivorBoundsSound : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributions survivorBounds := by
  exact survivorBoundsSoundTree1_0

theorem prefoldResult : prefoldTerms = rootTerms := by rfl

theorem prefoldBoundSound : rootBound ≤ prefoldBound := by decide +kernel


theorem prefoldSound :
  preFoldBound rootBound prefoldBound survivorContributions survivorBounds := by
  exact (preFoldSound rootTerms prefoldTerms prefoldResult prefoldBoundSound survivorBoundsSound).2

theorem endResult : endTerms = prefoldTerms := by rfl

theorem endSummaryResult : endSummary = prefoldSummary := by rfl

theorem endSound :
  endTerms = prefoldTerms ∧ endSummary = prefoldSummary := by
  exact ⟨endResult, endSummaryResult⟩

theorem invocationEndClaimSound (env : Env Owner) (actual : Int)
    (claim : ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 env actual (.exact rootTerms rootSummary)) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 env actual (.exact endTerms endSummary) := by
  exact invocationEndSound 100418593683253592432016548326729029359133068138294319235841 env actual rootTerms endTerms rootSummary endSummary
    claim endResult endSummaryResult

theorem selectedRootResultAt : (history.lookup rootResultEvent).map AnnotatedEvent.event = some (.resultExact selectedOwner rootRaw rootSummary) := by
  rfl

theorem selectedPreFoldAt : (history.lookup prefoldEvent).map AnnotatedEvent.event = some (.preFoldPolynomial rootResultEvent prefoldRaw prefoldSummary (none)) := by
  rfl

theorem selectedInvocationEndAt : (history.lookup endEvent).map AnnotatedEvent.event = some (.invocationEndExact selectedOwner prefoldEvent endRaw endSummary) := by
  rfl

end Bound0

def theoremCount : Nat := 22

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard291
