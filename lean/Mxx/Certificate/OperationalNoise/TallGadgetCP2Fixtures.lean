import Mxx.Certificate.OperationalNoise.TallSemantics

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallGadgetCP2Fixtures

open Mxx.Certificate.OperationalNoise
open EventReplay
open SchemaV1
open TallSecurity0ABI
open TallSemantics

def p214 (expression : Nat) : Owner :=
  ⟨.program ⟨214⟩, ⟨expression⟩⟩

/- Events025.lean records the 6520 root-wide rows as a local relation slice only:
   6512 is survivorFold, 6513 is owner E10368's result, 6516--6518 are transfers,
   6520 is the E10369 appliedRelation, and 6523 is its resultExact.  They all have
   frameStart 0; this slice deliberately has no fabricated InvocationEnd. -/
def rootRelationEvent : Nat := 6520
def rootRelationOwner : Owner := p214 10369
def rootRelationSource : Monomial :=
  { centralFactors := [p214 5519, p214 10365]
    orderedFactors := [p214 6544, p214 7882] }
def rootRelationGadget : Owner := p214 6544
def rootRelationDecomposition : Owner := p214 7882
def rootRelationInput : Owner := p214 6790
def rootRelationInputResult : Nat := 6457
def rootRelationResult : Nat := 6523

def rootRelationRecords : List (Nat × AnnotatedEvent) :=
  [(6457,
    ⟨.resultExact (p214 6790)
      [{ coefficient := 1
         monomial := { centralFactors := []
                       orderedFactors := [p214 6790] } }] .exactZero 0 .exactZero none, 0⟩),
   (6483,
    ⟨.resultExact (p214 7882)
      [{ coefficient := 1
         monomial := { centralFactors := []
                       orderedFactors := [p214 7882] } }] .exactZero 0 .exactZero none, 0⟩),
   (6512, ⟨.survivorFold 1 6511, 0⟩),
   (6513,
    ⟨.resultExact (p214 10368)
      [{ coefficient := 1
         monomial := { centralFactors := [p214 5519]
                       orderedFactors := [p214 6770] } },
       { coefficient := -1
         monomial := { centralFactors := [p214 5519, p214 10365]
                       orderedFactors := [p214 6544] } }] (.finite 26) 6511
      (.finite 26) (some 6511), 0⟩),
   (6514,
    ⟨.predecessor rootRelationOwner 0 ⟨10368⟩ 6513, 0⟩),
   (6515,
    ⟨.predecessor rootRelationOwner 1 ⟨7883⟩ 6487, 0⟩),
   (6516,
    ⟨.boundTransfer rootRelationOwner
      (.product (.predecessor 0 6514 .coefficient)
        (.predecessor 1 6515 .coefficient)
        { leftIsConstantPolynomial := false
          rightIsConstantPolynomial := false
          rightKnownZeroRows := none
          leftSupportUpper := none
          rightSupportUpper := none }), 0⟩),
   (6517,
    ⟨.boundTransfer rootRelationOwner
      (.monomialProduct
        { centralFactors := []
          orderedFactors := [rootRelationDecomposition] }
        [{ bound := .result 6483 .coefficient
           isConstantPolynomial := false
           supportUpper := none }]), 0⟩),
   (6518,
    ⟨.boundTransfer rootRelationOwner
      (.product (.result 6513 .summary) (.transfer 6517)
        { leftIsConstantPolynomial := false
          rightIsConstantPolynomial := false
          rightKnownZeroRows := none
          leftSupportUpper := none
          rightSupportUpper := none }), 0⟩),
   (rootRelationEvent,
    ⟨.appliedRelation rootRelationOwner rootRelationSource (-1) 0 2
      (.gadget rootRelationGadget rootRelationDecomposition ⟨6790⟩
        rootRelationInputResult), 0⟩),
   (rootRelationResult,
    ⟨.resultExact rootRelationOwner
      [{ coefficient := 1
         monomial := { centralFactors := [p214 5519]
                       orderedFactors := [p214 6770, p214 7882] } },
       { coefficient := -1
         monomial := { centralFactors := [p214 5519, p214 10365]
                       orderedFactors := [p214 6790] } }] (.finite 95420416) 6522
      (.finite 95420416) (some 6522), 0⟩)]

def rootRelationLookup (event : Nat) : Option AnnotatedEvent :=
  (rootRelationRecords.find? (fun record => record.1 == event)).map Prod.snd

theorem rootRelationRecords_source_faithful :
    rootRelationLookup 6512 = some ⟨.survivorFold 1 6511, 0⟩ ∧
      rootRelationLookup rootRelationEvent =
        some ⟨.appliedRelation rootRelationOwner rootRelationSource (-1) 0 2
          (.gadget rootRelationGadget rootRelationDecomposition ⟨6790⟩
            rootRelationInputResult), 0⟩ ∧
      rootRelationLookup rootRelationResult = some
        ⟨.resultExact rootRelationOwner
          [{ coefficient := 1
             monomial := { centralFactors := [p214 5519]
                           orderedFactors := [p214 6770, p214 7882] } },
           { coefficient := -1
             monomial := { centralFactors := [p214 5519, p214 10365]
                           orderedFactors := [p214 6790] } }] (.finite 95420416) 6522
          (.finite 95420416) (some 6522), 0⟩ := by
  repeat' apply And.intro
  all_goals
    simp only [rootRelationLookup, rootRelationRecords, rootRelationEvent,
      rootRelationResult, rootRelationInputResult, rootRelationDecomposition,
      List.find?_cons]
    rfl

/- The smallest complete Gadget-containing frame used by the E2E gate is the actual
   frameStart 6616 sequence in Events025.lean/Events026.lean: invocationStart 6616,
   appliedRelation 6709, resultExact 6712, resultExact 6731, preFoldPolynomial
   6732, and invocationEndExact 6733.  The enclosing root history is not copied. -/
def frameStart : Nat := 6616
def frameOwner : Owner := p214 25782
def relationEvent : Nat := 6709
def relationOwner : Owner := p214 25781
def relationSource : Monomial :=
  { centralFactors := [p214 10365, p214 13382]
    orderedFactors := [p214 6544, p214 25778] }
def relationGadget : Owner := p214 6544
def relationDecomposition : Owner := p214 25778
def relationInput : Owner := p214 23424
def relationInputResult : Nat := 6658
def relationBaseResult : Nat := 6704
def relationTransferResult : Nat := 6661
def relationResult : Nat := 6712
def rootIntermediateResult : Nat := 6727
def rootResult : Nat := 6731
def preFoldEvent : Nat := 6732
def invocationEndEvent : Nat := 6733

def sourceKey : MonomialKey Owner :=
  { centralFactors := [p214 10365, p214 13382]
    orderedFactors := [p214 6544, p214 25778] }

def lhsKey : MonomialKey Owner :=
  { centralFactors := []
    orderedFactors := [p214 6544, p214 25778] }

def relationRhs : Polynomial Owner :=
  [{ coefficient := 1
     key := { centralFactors := []
              orderedFactors := [p214 23424] } }]

def baseResult : Polynomial Owner :=
  [{ coefficient := 1
     key := { centralFactors := []
              orderedFactors := [p214 6770, p214 7882] } },
   { coefficient := -1
     key := { centralFactors := [p214 10365, p214 13382]
              orderedFactors := [p214 6544] } }]

def transferResult : Polynomial Owner :=
  [{ coefficient := 1
     key := { centralFactors := []
              orderedFactors := [p214 25778] } }]

def productResult : Polynomial Owner := productPoly baseResult transferResult false false

def relationOutput : Polynomial Owner :=
  [{ coefficient := 1
     key := { centralFactors := []
              orderedFactors := [p214 6770, p214 7882, p214 25778] } },
   { coefficient := -1
     key := { centralFactors := [p214 10365, p214 13382]
              orderedFactors := [p214 23424] } }]

def intermediateResult : Polynomial Owner :=
  [{ coefficient := 1
     key := { centralFactors := []
              orderedFactors := [p214 6707] } },
   { coefficient := -1
     key := { centralFactors := [p214 17027]
              orderedFactors := [p214 6544] } }]

def finalResult : Polynomial Owner :=
  [{ coefficient := 1
     key := { centralFactors := []
              orderedFactors := [p214 6707] } },
   { coefficient := -1
     key := { centralFactors := []
              orderedFactors := [p214 6770, p214 7882, p214 25778] } },
   { coefficient := 1
     key := { centralFactors := [p214 10365, p214 13382]
              orderedFactors := [p214 23424] } },
   { coefficient := -1
     key := { centralFactors := [p214 17027]
              orderedFactors := [p214 6544] } }]

def baseResultTerms : List Term :=
  [{ coefficient := 1
     monomial := { centralFactors := []
                   orderedFactors := [p214 6770, p214 7882] } },
   { coefficient := -1
     monomial := { centralFactors := [p214 10365, p214 13382]
                   orderedFactors := [p214 6544] } }]

def transferResultTerms : List Term :=
  [{ coefficient := 1
     monomial := { centralFactors := []
                   orderedFactors := [p214 25778] } }]

def relationResultTerms : List Term :=
  [{ coefficient := 1
     monomial := { centralFactors := []
                   orderedFactors := [p214 6770, p214 7882, p214 25778] } },
   { coefficient := -1
     monomial := { centralFactors := [p214 10365, p214 13382]
                   orderedFactors := [p214 23424] } }]

def intermediateResultTerms : List Term :=
  [{ coefficient := 1
     monomial := { centralFactors := []
                   orderedFactors := [p214 6707] } },
   { coefficient := -1
     monomial := { centralFactors := [p214 17027]
                   orderedFactors := [p214 6544] } }]

def finalResultTerms : List Term :=
  [{ coefficient := 1
     monomial := { centralFactors := []
                   orderedFactors := [p214 6707] } },
   { coefficient := -1
     monomial := { centralFactors := []
                   orderedFactors := [p214 6770, p214 7882, p214 25778] } },
   { coefficient := 1
     monomial := { centralFactors := [p214 10365, p214 13382]
                   orderedFactors := [p214 23424] } },
   { coefficient := -1
     monomial := { centralFactors := [p214 17027]
                   orderedFactors := [p214 6544] } }]

def sourceMonomial : Monomial :=
  { centralFactors := sourceKey.centralFactors
    orderedFactors := sourceKey.orderedFactors }

def frameRecords : List (Nat × AnnotatedEvent) :=
  [(frameStart, ⟨.invocationStart frameOwner, frameStart⟩),
   (relationInputResult,
    ⟨.resultExact relationInput
      [{ coefficient := 1
         monomial := { centralFactors := []
                       orderedFactors := [relationInput] } }] .exactZero 0 .exactZero none,
      frameStart⟩),
   (relationTransferResult,
    ⟨.resultExact (p214 25778) transferResultTerms .exactZero 0 .exactZero none,
      frameStart⟩),
   (relationBaseResult,
    ⟨.resultExact (p214 13465) baseResultTerms .exactZero 0 .exactZero none, frameStart⟩),
   (relationEvent,
    ⟨.appliedRelation relationOwner sourceMonomial (-1) 0 2
      (.gadget relationGadget relationDecomposition ⟨23424⟩ relationInputResult),
      frameStart⟩),
   (relationResult,
    ⟨.resultExact relationOwner relationResultTerms .exactZero 0 .exactZero none, frameStart⟩),
   (rootIntermediateResult,
    ⟨.resultExact (p214 17030) intermediateResultTerms .exactZero 0 .exactZero none,
      frameStart⟩),
   (rootResult,
    ⟨.resultExact frameOwner finalResultTerms .exactZero 0 .exactZero none, frameStart⟩),
   (preFoldEvent,
    ⟨.preFoldPolynomial rootResult finalResultTerms .exactZero none, frameStart⟩),
   (invocationEndEvent,
    ⟨.invocationEndExact frameOwner preFoldEvent finalResultTerms .exactZero,
      frameStart⟩)]

def frameLookup (event : Nat) : Option AnnotatedEvent :=
  (frameRecords.find? (fun record => record.1 == event)).map Prod.snd

def frameRelationTerms : List Term :=
  match frameLookup relationResult with
  | some ⟨.resultExact _ terms _ _ _ _, _⟩ => terms
  | _ => []

def frameBaseTerms : List Term :=
  match frameLookup relationBaseResult with
  | some ⟨.resultExact _ terms _ _ _ _, _⟩ => terms
  | _ => []

def frameTransferTerms : List Term :=
  match frameLookup relationTransferResult with
  | some ⟨.resultExact _ terms _ _ _ _, _⟩ => terms
  | _ => []

def frameRelationOwner : Owner :=
  match frameLookup relationEvent with
  | some ⟨.appliedRelation owner _ _ _ _ _, _⟩ => owner
  | _ => p214 0

def frameRelationSource : Monomial :=
  match frameLookup relationEvent with
  | some ⟨.appliedRelation _ source _ _ _ _, _⟩ => source
  | _ => { centralFactors := [], orderedFactors := [] }

def frameRelationOuter : Int :=
  match frameLookup relationEvent with
  | some ⟨.appliedRelation _ _ outer _ _ _, _⟩ => outer
  | _ => 0

def frameRelationStart : Nat :=
  match frameLookup relationEvent with
  | some ⟨.appliedRelation _ _ _ start _ _, _⟩ => start
  | _ => 0

def frameRelationEnd : Nat :=
  match frameLookup relationEvent with
  | some ⟨.appliedRelation _ _ _ _ finish _, _⟩ => finish
  | _ => 0

def frameRelationGadget : Owner :=
  match frameLookup relationEvent with
  | some ⟨.appliedRelation _ _ _ _ _ (.gadget gadget _ _ _), _⟩ => gadget
  | _ => p214 0

def frameRelationDecomposition : Owner :=
  match frameLookup relationEvent with
  | some ⟨.appliedRelation _ _ _ _ _ (.gadget _ decomposition _ _), _⟩ => decomposition
  | _ => p214 0

def frameRelationInput : Nat :=
  match frameLookup relationEvent with
  | some ⟨.appliedRelation _ _ _ _ _ (.gadget _ _ input _), _⟩ => input.row
  | _ => 0

def frameRelationInputResult : Nat :=
  match frameLookup relationEvent with
  | some ⟨.appliedRelation _ _ _ _ _ (.gadget _ _ _ inputResult), _⟩ => inputResult
  | _ => 0

def frameRelationSourceKey : MonomialKey Owner := frameRelationSource.toKey

def frameRootTerms : List Term :=
  match frameLookup rootResult with
  | some ⟨.resultExact _ terms _ _ _ _, _⟩ => terms
  | _ => []

def framePreFoldTerms : List Term :=
  match frameLookup preFoldEvent with
  | some ⟨.preFoldPolynomial _ terms _ _, _⟩ => terms
  | _ => []

def framePreFoldSummary : Bound :=
  match frameLookup preFoldEvent with
  | some ⟨.preFoldPolynomial _ _ summary _, _⟩ => summary
  | _ => .missing

def frameEndTerms : List Term :=
  match frameLookup invocationEndEvent with
  | some ⟨.invocationEndExact _ _ terms _, _⟩ => terms
  | _ => []

def frameEndSummary : Bound :=
  match frameLookup invocationEndEvent with
  | some ⟨.invocationEndExact _ _ _ summary, _⟩ => summary
  | _ => .missing

theorem frameRecords_source_faithful :
    frameLookup relationBaseResult =
        some ⟨.resultExact (p214 13465) baseResultTerms .exactZero 0 .exactZero none,
          frameStart⟩ ∧
      frameLookup relationTransferResult =
        some ⟨.resultExact (p214 25778) transferResultTerms .exactZero 0 .exactZero none,
          frameStart⟩ ∧
    frameLookup relationEvent =
        some ⟨.appliedRelation relationOwner sourceMonomial (-1) 0 2
          (.gadget relationGadget relationDecomposition ⟨23424⟩ relationInputResult),
          frameStart⟩ ∧
      frameLookup relationResult =
        some ⟨.resultExact relationOwner relationResultTerms .exactZero 0 .exactZero none,
          frameStart⟩ ∧
      frameLookup rootResult =
        some ⟨.resultExact frameOwner finalResultTerms .exactZero 0 .exactZero none,
          frameStart⟩ ∧
      frameLookup preFoldEvent =
        some ⟨.preFoldPolynomial rootResult finalResultTerms .exactZero none, frameStart⟩ ∧
      frameLookup invocationEndEvent =
        some ⟨.invocationEndExact frameOwner preFoldEvent finalResultTerms .exactZero,
          frameStart⟩ := by
  repeat' apply And.intro
  all_goals
    simp only [frameLookup, frameRecords, frameStart, relationEvent, relationResult,
      preFoldEvent, invocationEndEvent, relationInputResult, relationTransferResult,
      relationBaseResult, rootIntermediateResult, rootResult, List.find?_cons]
    rfl

theorem term_rows_shape :
    termPolynomial baseResultTerms = baseResult ∧
      termPolynomial transferResultTerms = transferResult ∧
      termPolynomial relationResultTerms = relationOutput ∧
      termPolynomial intermediateResultTerms = intermediateResult ∧
      termPolynomial finalResultTerms = finalResult := by
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

def fixtureEnv : Env Owner := fun _ => 1

theorem product_result_sound (env : Env Owner) :
    evalPolynomial env productResult =
      evalPolynomial env baseResult * evalPolynomial env transferResult := by
  apply productResultSound env baseResult transferResult productResult false false
  intro term h
  rfl

theorem relation_source_context :
    KeyEquivalent sourceKey
      ((relationContext sourceKey sourceKey.centralFactors 0 2).plug lhsKey) := by
  constructor <;> rfl

theorem relation_base_congruence :
    evalMonomial fixtureEnv lhsKey % 257 =
      evalPolynomial fixtureEnv relationRhs % 257 := by
  simp [evalMonomial, evalPolynomial, fixtureEnv, lhsKey, relationRhs]

theorem relation_agreement :
    CoefficientAgreement relationOutput
      (relationPoly productResult sourceKey
        (relationContext sourceKey sourceKey.centralFactors 0 2) (-1) relationRhs) := by
  have shape :
      relationPoly productResult sourceKey
        (relationContext sourceKey sourceKey.centralFactors 0 2) (-1) relationRhs =
        [{ coefficient := 1
           key := { centralFactors := []
                    orderedFactors := [p214 6770, p214 7882, p214 25778] } },
         { coefficient := -1
           key := { centralFactors := [p214 10365, p214 13382]
                    orderedFactors := [p214 6544, p214 25778] } },
         { coefficient := 1, key := sourceKey },
         { coefficient := -1
           key := { centralFactors := [p214 10365, p214 13382]
                    orderedFactors := [p214 23424] } }] := by
    rfl
  rw [shape]
  intro term h
  rcases List.mem_append.mp h with h | h
  · simp [relationOutput] at h
    rcases h with rfl | rfl <;> rfl
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl <;> rfl

theorem relation_result_sound (env : Env Owner)
    (baseRelation : evalMonomial env lhsKey % 257 =
      evalPolynomial env relationRhs % 257) :
    evalPolynomial env relationOutput % 257 = evalPolynomial env productResult % 257 := by
  exact relationResultSound 257 env productResult sourceKey lhsKey
    sourceKey.centralFactors 0 2 (-1) relationRhs relationOutput
    relation_source_context baseRelation relation_agreement

theorem final_agreement :
    CoefficientAgreement finalResult
      (subtract intermediateResult relationOutput) := by
  rw [show subtract intermediateResult relationOutput =
    [{ coefficient := 1, key := { centralFactors := [], orderedFactors := [p214 6707] } },
     { coefficient := -1,
       key := { centralFactors := [p214 17027], orderedFactors := [p214 6544] } },
     { coefficient := -1,
       key := { centralFactors := [], orderedFactors := [p214 6770, p214 7882, p214 25778] } },
     { coefficient := 1,
       key := { centralFactors := [p214 10365, p214 13382], orderedFactors := [p214 23424] } }] by
      rfl]
  intro term h
  rcases List.mem_append.mp h with h | h
  · simp [finalResult] at h
    rcases h with rfl | rfl | rfl | rfl <;> rfl
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl <;> rfl

theorem final_result_sound (env : Env Owner) :
    evalPolynomial env finalResult =
      evalPolynomial env intermediateResult - evalPolynomial env relationOutput := by
  exact subResultSound env intermediateResult relationOutput finalResult final_agreement

theorem frame_relation_product_prefold_end (env : Env Owner)
    (baseRelation : evalMonomial env lhsKey % 257 =
      evalPolynomial env relationRhs % 257) :
    ValueClaim.Interprets 257 env
      (evalPolynomial env intermediateResult -
        (evalPolynomial env baseResult * evalPolynomial env transferResult))
      (.exact finalResult .exactZero) := by
  rcases frameRecords_source_faithful with
    ⟨baseLookup, transferLookup, relationLookup, relationResultLookup, rootLookup,
      preFoldLookup, endLookup⟩
  have baseTermsExact : termPolynomial frameBaseTerms = baseResult := by
    dsimp [frameBaseTerms]
    rw [baseLookup]
    rfl
  have transferTermsExact : termPolynomial frameTransferTerms = transferResult := by
    dsimp [frameTransferTerms]
    rw [transferLookup]
    rfl
  have relationTermsExact : termPolynomial frameRelationTerms = relationOutput := by
    dsimp [frameRelationTerms]
    rw [relationResultLookup]
    rfl
  have rootTermsExact : termPolynomial frameRootTerms = finalResult := by
    dsimp [frameRootTerms]
    rw [rootLookup]
    exact (term_rows_shape).2.2.2.2
  have preFoldTermsExact : termPolynomial framePreFoldTerms = finalResult := by
    dsimp [framePreFoldTerms]
    rw [preFoldLookup]
    exact (term_rows_shape).2.2.2.2
  have preFoldSummaryExact : framePreFoldSummary = .exactZero := by
    dsimp [framePreFoldSummary]
    rw [preFoldLookup]
  have endTermsExact : termPolynomial frameEndTerms = finalResult := by
    dsimp [frameEndTerms]
    rw [endLookup]
    exact (term_rows_shape).2.2.2.2
  have endSummaryExact : frameEndSummary = .exactZero := by
    dsimp [frameEndSummary]
    rw [endLookup]
  have extractedProductAgreement :
      CoefficientAgreement productResult
        (productPoly (termPolynomial frameBaseTerms)
          (termPolynomial frameTransferTerms) false false) := by
    rw [baseTermsExact, transferTermsExact]
    intro term h
    rfl
  have productSoundExtracted :
      evalPolynomial env productResult =
        evalPolynomial env (termPolynomial frameBaseTerms) *
          evalPolynomial env (termPolynomial frameTransferTerms) := by
    exact productResultSound env (termPolynomial frameBaseTerms)
      (termPolynomial frameTransferTerms) productResult false false
      extractedProductAgreement
  have productSound :
      evalPolynomial env productResult =
        evalPolynomial env baseResult * evalPolynomial env transferResult := by
    rw [baseTermsExact, transferTermsExact] at productSoundExtracted
    exact productSoundExtracted
  have relationOwnerExact : frameRelationOwner = relationOwner := by
    dsimp [frameRelationOwner]
    rw [relationLookup]
  have relationSourceExact : frameRelationSourceKey = sourceKey := by
    dsimp [frameRelationSourceKey, frameRelationSource]
    rw [relationLookup]
    rfl
  have relationOuterExact : frameRelationOuter = -1 := by
    dsimp [frameRelationOuter]
    rw [relationLookup]
  have relationStartExact : frameRelationStart = 0 := by
    dsimp [frameRelationStart]
    rw [relationLookup]
  have relationEndExact : frameRelationEnd = 2 := by
    dsimp [frameRelationEnd]
    rw [relationLookup]
  have relationGadgetExact : frameRelationGadget = relationGadget := by
    dsimp [frameRelationGadget]
    rw [relationLookup]
  have relationDecompositionExact :
      frameRelationDecomposition = relationDecomposition := by
    dsimp [frameRelationDecomposition]
    rw [relationLookup]
  have relationInputExact : frameRelationInput = 23424 := by
    dsimp [frameRelationInput]
    rw [relationLookup]
  have relationInputResultExact : frameRelationInputResult = relationInputResult := by
    dsimp [frameRelationInputResult]
    rw [relationLookup]
  have extractedSourceContext :
      KeyEquivalent frameRelationSourceKey
        ((relationContext frameRelationSourceKey frameRelationSourceKey.centralFactors
          frameRelationStart frameRelationEnd).plug lhsKey) := by
    rw [relationSourceExact, relationStartExact, relationEndExact]
    exact relation_source_context
  have extractedRelationAgreement :
      CoefficientAgreement (termPolynomial frameRelationTerms)
        (relationPoly productResult frameRelationSourceKey
          (relationContext frameRelationSourceKey frameRelationSourceKey.centralFactors
            frameRelationStart frameRelationEnd)
          frameRelationOuter relationRhs) := by
    rw [relationTermsExact, relationSourceExact, relationStartExact, relationEndExact,
      relationOuterExact]
    exact relation_agreement
  have preFoldToRoot : termPolynomial framePreFoldTerms = termPolynomial frameRootTerms :=
    preFoldTermsExact.trans rootTermsExact.symm
  rw [← productSound]
  have relationSoundRaw := relationResultSound 257 env productResult frameRelationSourceKey lhsKey
    frameRelationSourceKey.centralFactors frameRelationStart frameRelationEnd frameRelationOuter
    relationRhs (termPolynomial frameRelationTerms) extractedSourceContext baseRelation
    extractedRelationAgreement
  have relationSound :
      evalPolynomial env (termPolynomial frameRelationTerms) % 257 =
        evalPolynomial env productResult % 257 := by
    rw [relationTermsExact]
    exact relationSoundRaw
  have remainderCongruence :
      (evalPolynomial env intermediateResult - evalPolynomial env productResult -
        evalPolynomial env finalResult) % 257 = 0 % 257 := by
    rw [final_result_sound]
    rw [Int.sub_emod]
    rw [Int.sub_emod (evalPolynomial env intermediateResult) (evalPolynomial env productResult) 257,
      Int.sub_emod (evalPolynomial env intermediateResult) (evalPolynomial env relationOutput) 257]
    rw [← relationTermsExact]
    rw [relationSound]
    rw [Int.sub_self]
  have prefold := preFoldSound (termPolynomial frameRootTerms)
    (termPolynomial framePreFoldTerms) preFoldToRoot (Nat.zero_le 0)
    (survivors := List.Forall₂.nil)
  have prefoldClaim : ValueClaim.Interprets 257 env
      (evalPolynomial env intermediateResult - evalPolynomial env productResult)
      (.exact (termPolynomial framePreFoldTerms) framePreFoldSummary) := by
    refine ⟨0, ?_, ?_⟩
    · rw [prefold.1, rootTermsExact]
      exact remainderCongruence
    · rw [preFoldSummaryExact]
      simp [boundInterprets, centeredNorm, centeredCoefficient]
  have endClaim := invocationEndSound 257 env
    (evalPolynomial env intermediateResult - evalPolynomial env productResult)
    (termPolynomial framePreFoldTerms) (termPolynomial frameEndTerms)
    framePreFoldSummary frameEndSummary prefoldClaim
    (endTermsExact.symm.trans preFoldTermsExact) (endSummaryExact.symm.trans preFoldSummaryExact)
  rw [endTermsExact, endSummaryExact] at endClaim
  exact endClaim

theorem gadget_frame6616_e2e :
    ValueClaim.Interprets 257 fixtureEnv
      (evalPolynomial fixtureEnv intermediateResult - evalPolynomial fixtureEnv productResult)
      (.exact finalResult .exactZero) := by
  apply frame_relation_product_prefold_end fixtureEnv relation_base_congruence

end Mxx.Certificate.OperationalNoise.TallGadgetCP2Fixtures

#print axioms Mxx.Certificate.OperationalNoise.TallSemantics.coefficientAgreement_eval
#print axioms Mxx.Certificate.OperationalNoise.TallSemantics.addResultSound
#print axioms Mxx.Certificate.OperationalNoise.TallSemantics.subResultSound
#print axioms Mxx.Certificate.OperationalNoise.TallSemantics.productResultSound
#print axioms Mxx.Certificate.OperationalNoise.TallSemantics.relationResultSound
#print axioms Mxx.Certificate.OperationalNoise.TallSemantics.preFoldSound
#print axioms Mxx.Certificate.OperationalNoise.TallSemantics.invocationEndSound
#print axioms Mxx.Certificate.OperationalNoise.TallGadgetCP2Fixtures.gadget_frame6616_e2e
