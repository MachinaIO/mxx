import Mxx.Certificate.OperationalNoise.TallSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard234

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

def shardIndex : Nat := 234
def shardStartEvent : Nat := 59904
def shardEndEvent : Nat := 60160
def rawSemanticCount : Nat := 175
def rawBoundTransferCount : Nat := 95
def rawResultCount : Nat := 57
def rawRelationCount : Nat := 0
def rawSurvivorFoldCount : Nat := 21
def rawPreFoldCount : Nat := 1
def rawInvocationEndCount : Nat := 1
def canonicalWork : Nat := 39

namespace Operation0
def selectedEvent : Nat := 60100
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18564⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18563⟩⟩] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨6⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨18563⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 60098
def selectedLeftResultEvent : Nat := 60095
def selectedRightResultEvent : Nat := 60093
def selectedResultEvent : Nat := 60100
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 60096 .coefficient) (.predecessor 1 60097 .coefficient) ⟨false, false, none, none, none⟩)) := by
  rfl

end Operation0

namespace Operation1
def selectedEvent : Nat := 60134
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨13359⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10350⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13358⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨10350⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨13358⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 60132
def selectedLeftResultEvent : Nat := 60129
def selectedRightResultEvent : Nat := 60126
def selectedResultEvent : Nat := 60134
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 60130 .coefficient) (.predecessor 1 60131 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation1

namespace Operation2
def selectedEvent : Nat := 60157
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨13163⟩⟩
def leftRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10245⟩⟩], orderedFactors := [] } }]
def rightRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13162⟩⟩], orderedFactors := [] } }]
def outputRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10245⟩⟩, ⟨.program ⟨214⟩, ⟨13162⟩⟩], orderedFactors := [] } }]
def left : Polynomial Owner := leftRaw.map Term.toExact
def right : Polynomial Owner := rightRaw.map Term.toExact
def output : Polynomial Owner := outputRaw.map Term.toExact
def leftOwner : Owner := ⟨.program ⟨214⟩, ⟨10245⟩⟩
def rightOwner : Owner := ⟨.program ⟨214⟩, ⟨13162⟩⟩
def leftSummary : Bound := .exactZero
def rightSummary : Bound := .exactZero
def outputSummary : Bound := .exactZero
def selectedRawWork : Nat := 1
def selectedSumRuleEvent : Nat := 60155
def selectedLeftResultEvent : Nat := 60152
def selectedRightResultEvent : Nat := 60149
def selectedResultEvent : Nat := 60157
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

theorem selectedRuleAt : (history.lookup selectedSumRuleEvent).map AnnotatedEvent.event = some (.boundTransfer selectedOwner (.product (.predecessor 0 60153 .coefficient) (.predecessor 1 60154 .coefficient) ⟨true, true, none, some 1, some 1⟩)) := by
  rfl

end Operation2

namespace Bound0
def selectedEvent : Nat := 60102
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18564⟩⟩
def rootResultEvent : Nat := 60100
def prefoldEvent : Nat := 60101
def endEvent : Nat := 60102
def survivorEvents : List Nat := [59545, 59569, 59593, 59617, 59641, 59665, 59689, 59713, 59737, 59761, 59785, 59809, 59833, 59857, 59881, 59905, 59929, 59953, 59971, 59973, 59980, 59987, 59994, 60001, 60008, 60015, 60022, 60029, 60036, 60043, 60050, 60057, 60064, 60071, 60078, 60085]
def rootRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩] } }]
def prefoldRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩] } }]
def endRaw : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩] } }]
def rootTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩] } }]
def prefoldTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩] } }]
def endTerms : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18563⟩⟩] } }]
def rootSummary : Bound := .exactZero
def prefoldSummary : Bound := .exactZero
def endSummary : Bound := .exactZero
def rootBound : Nat := 0
def prefoldBound : Nat := 0
def survivorContributionsChunk0 : List Nat := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
def survivorBoundsChunk0 : List Nat := [59544, 59568, 59592, 59616, 59640, 59664, 59688, 59712, 59736, 59760, 59784, 59808, 59832, 59856, 59880, 59904]
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
def survivorBoundsChunk1 : List Nat := [59928, 59952, 59970, 59972, 59979, 59986, 59993, 60000, 60007, 60014, 60021, 60028, 60035, 60042, 60049, 60056]
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
def survivorBoundsChunk2 : List Nat := [60063, 60070, 60077, 60084]
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

def theoremCount : Nat := 34

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticShard234
