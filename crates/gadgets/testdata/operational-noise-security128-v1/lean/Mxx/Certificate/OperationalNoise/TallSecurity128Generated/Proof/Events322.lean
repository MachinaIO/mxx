import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events322

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event82432 : Event := .preFoldPolynomial 82431 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52585⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], [⟨.program ⟨257⟩, ⟨52045⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact82433RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52585⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], [⟨.program ⟨257⟩, ⟨52045⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event82433 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨52589⟩⟩) 82432 exact82433RawTerms .large 82430 .exactZero (none)

def event82434 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨50709⟩⟩) ⟨⟨62⟩, ⟨40⟩, ⟨135⟩⟩ ⟨82268, 82434⟩

def event82435 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨51512⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51509⟩⟩]⟩) (1) 0 2 (.universal 82434 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51509⟩⟩]⟩) (none) 82433)

def event82436 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51512⟩⟩, .relation 82435 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩)

def event82437 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51512⟩⟩, .relation 82435 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52585⟩⟩]⟩, (-1)⟩)

def event82438 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51512⟩⟩, .relation 82435 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], [⟨.program ⟨257⟩, ⟨52045⟩⟩]⟩, (1)⟩)

def event82439 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51512⟩⟩, .relation 82435 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact82440RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52585⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], [⟨.program ⟨257⟩, ⟨52045⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82440RawTermsValid :
    exact82440RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82440 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51512⟩⟩) exact82440RawTerms .large 82264 (.finite 202072841853861888) (some (82266))

def event82441 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52587⟩⟩) 0 ⟨51512⟩ 82440

def event82442 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52587⟩⟩) 1 ⟨52586⟩ 82254

def event82443 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52587⟩⟩) (.sum [.predecessor 0 82441 .coefficient, .predecessor 1 82442 .coefficient])

def event82444 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52587⟩⟩, .operator (⟨82440, 2⟩, ⟨82254, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], [⟨.program ⟨257⟩, ⟨52045⟩⟩]⟩, (-1)⟩)

def event82445 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52587⟩⟩, .operator (⟨82440, 1⟩, ⟨82254, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52585⟩⟩]⟩, (1)⟩)

def event82446 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52587⟩⟩) (.sum [.result 82440 .summary, .result 82254 .summary])

def exact82447RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82447RawTermsValid :
    exact82447RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82447 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52587⟩⟩) exact82447RawTerms .large 82443 (.finite 2997889464187086962688) (some (82446))

def event82448 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53140⟩⟩) 0 ⟨52587⟩ 82447

def event82449 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53140⟩⟩) 1 ⟨53138⟩ 82170

def event82450 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53140⟩⟩) (.product (.predecessor 0 82448 .coefficient) (.predecessor 1 82449 .coefficient) (⟨false, false, none, none, none⟩))

def event82451 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53140⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩) [⟨.result 82170 .coefficient, false, none⟩])

def event82452 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53140⟩⟩) (.product (.result 82447 .summary) (.transfer 82451) (⟨false, false, none, none, none⟩))

def event82453 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53140⟩⟩, .operator (⟨82447, 0⟩, ⟨82170, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩, (1)⟩)

def event82454 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53140⟩⟩, .operator (⟨82447, 1⟩, ⟨82170, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩, (-1)⟩)

def event82455 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨53140⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨53138⟩⟩) ⟨52215⟩ 82167)

def event82456 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53140⟩⟩, .relation 82455 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨52215⟩⟩]⟩, (-1)⟩)

def exact82457RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨52215⟩⟩]⟩, (-1)⟩]

theorem exact82457RawTermsValid :
    exact82457RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82457 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53140⟩⟩) exact82457RawTerms .large 82450 (.finite 32189593014266254325632330629120) (some (82452))

def event82458 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51876⟩⟩) 0 ⟨50937⟩ 3402

def event82459 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51876⟩⟩) (.authority (.relationPreimageSource ⟨65⟩))

def exact82460RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51876⟩⟩]⟩, (1)⟩]

theorem exact82460RawTermsValid :
    exact82460RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82460 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51876⟩⟩) exact82460RawTerms (.finite 5647228698) 82459 .exactZero (none)

def event82461 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51878⟩⟩) 0 ⟨51876⟩ 82460

def event82462 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51878⟩⟩) 1 ⟨2370⟩ 4

def event82463 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51878⟩⟩) (.scale (.predecessor 0 82461 .coefficient) (.value (.predecessor 1 82462 .coefficient)))

def exact82464RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51876⟩⟩]⟩, (1)⟩]

theorem exact82464RawTermsValid :
    exact82464RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82464 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51878⟩⟩) exact82464RawTerms (.finite 5647228698) 82463 .exactZero (none)

def event82465 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51879⟩⟩) 0 ⟨10368⟩ 75995

def event82466 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51879⟩⟩) 1 ⟨51878⟩ 82464

def event82467 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51879⟩⟩) (.product (.predecessor 0 82465 .coefficient) (.predecessor 1 82466 .coefficient) (⟨false, false, none, none, none⟩))

def event82468 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51879⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨51876⟩⟩]⟩) [⟨.result 82460 .coefficient, false, none⟩])

def event82469 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51879⟩⟩) (.product (.result 75995 .summary) (.transfer 82468) (⟨false, false, none, none, none⟩))

def event82470 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51879⟩⟩, .operator (⟨75995, 0⟩, ⟨82464, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51876⟩⟩]⟩, (1)⟩)

def event82471 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨51877⟩⟩)

def event82472 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event82473 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event82474 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event82475 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event82476 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event82477 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event82478 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event82479 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event82480 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 82479

def event82481 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 82477

def event82482 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 82480 .coefficient) (.value (.predecessor 1 82481 .coefficient)))

def event82483 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event82484 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 82483

def event82485 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 82475

def event82486 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 82484 .coefficient, .predecessor 1 82485 .coefficient])

def event82487 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event82488 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 82487

def event82489 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 82473

def event82490 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 82489 .coefficient))

def event82491 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event82492 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24602⟩⟩) 0 ⟨10325⟩ 82491

def event82493 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24602⟩⟩) (.authority (.programFamilyFact))

def exact82494RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24602⟩⟩], []⟩, (1)⟩]

theorem exact82494RawTermsValid :
    exact82494RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82494 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24602⟩⟩) exact82494RawTerms (.finite 10) 82493 .exactZero (none)

def event82495 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50707⟩⟩) 0 ⟨10325⟩ 82491

def event82496 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50707⟩⟩) (.authority (.programFamilyFact))

def exact82497RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50707⟩⟩], []⟩, (1)⟩]

theorem exact82497RawTermsValid :
    exact82497RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82497 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50707⟩⟩) exact82497RawTerms (.finite 10) 82496 .exactZero (none)

def event82498 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50708⟩⟩) 0 ⟨50707⟩ 82497

def event82499 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50708⟩⟩) 1 ⟨24602⟩ 82494

def event82500 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50708⟩⟩) (.product (.predecessor 0 82498 .coefficient) (.predecessor 1 82499 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event82501 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50708⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], []⟩) [⟨.result 82497 .coefficient, true, some 1⟩, ⟨.result 82494 .coefficient, true, some 1⟩])

def event82502 : Event := .survivorFold (1) 82501

def exact82503RawTerms : List Term := []

theorem exact82503RawTermsValid :
    exact82503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82503 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50708⟩⟩) exact82503RawTerms (.finite 100) 82500 (.finite 100) (some (82501))

def event82504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50709⟩⟩) 0 ⟨50708⟩ 82503

def event82505 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50709⟩⟩) (.identity (.predecessor 0 82504 .coefficient))

def event82506 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50709⟩⟩) (.finite 100)

def event82507 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50936⟩⟩) 0 ⟨50709⟩ 82506

def event82508 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50936⟩⟩) (.authority (.programFamilyFact))

def exact82509RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50936⟩⟩], []⟩, (1)⟩]

theorem exact82509RawTermsValid :
    exact82509RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82509 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50936⟩⟩) exact82509RawTerms (.finite 10) 82508 .exactZero (none)

def event82510 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50937⟩⟩) 0 ⟨50936⟩ 82509

def event82511 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50937⟩⟩) (.identity (.predecessor 0 82510 .coefficient))

def event82512 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50937⟩⟩) (.finite 10)

def event82513 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51876⟩⟩) 0 ⟨50937⟩ 82512

def event82514 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51876⟩⟩) (.authority (.relationPreimageSource ⟨65⟩))

def exact82515RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51876⟩⟩]⟩, (1)⟩]

theorem exact82515RawTermsValid :
    exact82515RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82515 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51876⟩⟩) exact82515RawTerms (.finite 5647228698) 82514 .exactZero (none)

def event82516 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact82517RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact82517RawTermsValid :
    exact82517RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82517 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact82517RawTerms .large 82516 .exactZero (none)

def event82518 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51877⟩⟩) 0 ⟨35⟩ 82517

def event82519 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51877⟩⟩) 1 ⟨51876⟩ 82515

def event82520 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51877⟩⟩) (.product (.predecessor 0 82518 .coefficient) (.predecessor 1 82519 .coefficient) (⟨false, false, none, none, none⟩))

def event82521 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51877⟩⟩, .operator (⟨82517, 0⟩, ⟨82515, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51876⟩⟩]⟩, (1)⟩)

def exact82522RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51876⟩⟩]⟩, (1)⟩]

theorem exact82522RawTermsValid :
    exact82522RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82522 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51877⟩⟩) exact82522RawTerms .large 82520 .exactZero (none)

def event82523 : Event := .preFoldPolynomial 82522 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51876⟩⟩]⟩, (1)⟩] .exactZero none

def exact82524RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51876⟩⟩]⟩, (1)⟩]

def event82524 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨51877⟩⟩) 82523 exact82524RawTerms .large 82520 .exactZero (none)

def event82525 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨53143⟩⟩)

def event82526 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event82527 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event82528 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event82529 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event82530 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event82531 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event82532 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event82533 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event82534 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 82533

def event82535 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 82531

def event82536 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 82534 .coefficient) (.value (.predecessor 1 82535 .coefficient)))

def event82537 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event82538 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 82537

def event82539 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 82529

def event82540 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 82538 .coefficient, .predecessor 1 82539 .coefficient])

def event82541 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event82542 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 82541

def event82543 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 82527

def event82544 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 82543 .coefficient))

def event82545 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event82546 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24602⟩⟩) 0 ⟨10325⟩ 82545

def event82547 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24602⟩⟩) (.authority (.programFamilyFact))

def exact82548RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24602⟩⟩], []⟩, (1)⟩]

theorem exact82548RawTermsValid :
    exact82548RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82548 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24602⟩⟩) exact82548RawTerms (.finite 10) 82547 .exactZero (none)

def event82549 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50707⟩⟩) 0 ⟨10325⟩ 82545

def event82550 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50707⟩⟩) (.authority (.programFamilyFact))

def exact82551RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50707⟩⟩], []⟩, (1)⟩]

theorem exact82551RawTermsValid :
    exact82551RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82551 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50707⟩⟩) exact82551RawTerms (.finite 10) 82550 .exactZero (none)

def event82552 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50708⟩⟩) 0 ⟨50707⟩ 82551

def event82553 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50708⟩⟩) 1 ⟨24602⟩ 82548

def event82554 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50708⟩⟩) (.product (.predecessor 0 82552 .coefficient) (.predecessor 1 82553 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event82555 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50708⟩⟩, .operator (⟨82551, 0⟩, ⟨82548, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], []⟩, (1)⟩)

def exact82556RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24602⟩⟩, ⟨.program ⟨257⟩, ⟨50707⟩⟩], []⟩, (1)⟩]

theorem exact82556RawTermsValid :
    exact82556RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82556 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50708⟩⟩) exact82556RawTerms (.finite 100) 82554 .exactZero (none)

def event82557 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50709⟩⟩) 0 ⟨50708⟩ 82556

def event82558 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50709⟩⟩) (.identity (.predecessor 0 82557 .coefficient))

def event82559 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50709⟩⟩) (.finite 100)

def event82560 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50936⟩⟩) 0 ⟨50709⟩ 82559

def event82561 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50936⟩⟩) (.authority (.programFamilyFact))

def exact82562RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50936⟩⟩], []⟩, (1)⟩]

theorem exact82562RawTermsValid :
    exact82562RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82562 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50936⟩⟩) exact82562RawTerms (.finite 10) 82561 .exactZero (none)

def event82563 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50937⟩⟩) 0 ⟨50936⟩ 82562

def event82564 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50937⟩⟩) (.identity (.predecessor 0 82563 .coefficient))

def event82565 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50937⟩⟩) (.finite 10)

def event82566 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52213⟩⟩) 0 ⟨50937⟩ 82565

def event82567 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52213⟩⟩) (.authority (.programFamilyFact))

def event82568 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52213⟩⟩) (.finite 3720)

def event82569 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event82570 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52215⟩⟩) 0 ⟨7177⟩ 82569

def event82571 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52215⟩⟩) 1 ⟨52213⟩ 82568

def event82572 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52215⟩⟩) (.authority (.operator))

def exact82573RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52215⟩⟩]⟩, (1)⟩]

theorem exact82573RawTermsValid :
    exact82573RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82573 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52215⟩⟩) exact82573RawTerms .large 82572 .exactZero (none)

def event82574 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53138⟩⟩) 0 ⟨52215⟩ 82573

def event82575 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53138⟩⟩) (.authority (.operator))

def exact82576RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩, (1)⟩]

theorem exact82576RawTermsValid :
    exact82576RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82576 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53138⟩⟩) exact82576RawTerms (.finite 8192) 82575 .exactZero (none)

def event82577 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event82578 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event82579 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52390⟩⟩) 0 ⟨50937⟩ 82565

def event82580 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52390⟩⟩) 1 ⟨136⟩ 82578

def event82581 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52390⟩⟩) (.sum [.predecessor 0 82579 .coefficient, .predecessor 1 82580 .coefficient])

def event82582 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52390⟩⟩) (.finite 10)

def event82583 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52391⟩⟩) 0 ⟨52390⟩ 82582

def event82584 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52391⟩⟩) (.identity (.predecessor 0 82583 .coefficient))

def exact82585RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50936⟩⟩], []⟩, (1)⟩]

theorem exact82585RawTermsValid :
    exact82585RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82585 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52391⟩⟩) exact82585RawTerms (.finite 10) 82584 .exactZero (none)

def event82586 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact82587RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact82587RawTermsValid :
    exact82587RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82587 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact82587RawTerms .large 82586 .exactZero (none)

def event82588 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52392⟩⟩) 0 ⟨6908⟩ 82587

def event82589 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52392⟩⟩) 1 ⟨52391⟩ 82585

def event82590 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52392⟩⟩) (.product (.predecessor 0 82588 .coefficient) (.predecessor 1 82589 .coefficient) (⟨false, false, none, none, none⟩))

def event82591 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52392⟩⟩, .operator (⟨82587, 0⟩, ⟨82585, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact82592RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact82592RawTermsValid :
    exact82592RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82592 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52392⟩⟩) exact82592RawTerms .large 82590 .exactZero (none)

def event82593 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7183⟩⟩) 0 ⟨7177⟩ 82569

def event82594 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7183⟩⟩) (.authority (.operator))

def exact82595RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩]

theorem exact82595RawTermsValid :
    exact82595RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82595 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7183⟩⟩) exact82595RawTerms .large 82594 .exactZero (none)

def event82596 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52393⟩⟩) 0 ⟨7183⟩ 82595

def event82597 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52393⟩⟩) 1 ⟨52392⟩ 82592

def event82598 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52393⟩⟩) (.sum [.predecessor 0 82596 .coefficient, .predecessor 1 82597 .coefficient])

def exact82599RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82599RawTermsValid :
    exact82599RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82599 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52393⟩⟩) exact82599RawTerms .large 82598 .exactZero (none)

def event82600 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53139⟩⟩) 0 ⟨52393⟩ 82599

def event82601 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53139⟩⟩) 1 ⟨53138⟩ 82576

def event82602 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53139⟩⟩) (.product (.predecessor 0 82600 .coefficient) (.predecessor 1 82601 .coefficient) (⟨false, false, none, none, none⟩))

def event82603 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53139⟩⟩, .operator (⟨82599, 0⟩, ⟨82576, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩, (1)⟩)

def event82604 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53139⟩⟩, .operator (⟨82599, 1⟩, ⟨82576, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩, (-1)⟩)

def event82605 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨53139⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨53138⟩⟩) ⟨52215⟩ 82573)

def event82606 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53139⟩⟩, .relation 82605 0, ⟨[⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨52215⟩⟩]⟩, (-1)⟩)

def exact82607RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨52215⟩⟩]⟩, (-1)⟩]

theorem exact82607RawTermsValid :
    exact82607RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82607 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53139⟩⟩) exact82607RawTerms .large 82602 .exactZero (none)

def event82608 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51275⟩⟩) 0 ⟨50937⟩ 82565

def event82609 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51275⟩⟩) (.authority (.programFamilyFact))

def exact82610RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], []⟩, (1)⟩]

theorem exact82610RawTermsValid :
    exact82610RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82610 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51275⟩⟩) exact82610RawTerms (.finite 58) 82609 .exactZero (none)

def event82611 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51277⟩⟩) 0 ⟨6908⟩ 82587

def event82612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51277⟩⟩) 1 ⟨51275⟩ 82610

def event82613 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51277⟩⟩) (.product (.predecessor 0 82611 .coefficient) (.predecessor 1 82612 .coefficient) (⟨false, true, none, none, some 1⟩))

def event82614 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51277⟩⟩, .operator (⟨82587, 0⟩, ⟨82610, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact82615RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact82615RawTermsValid :
    exact82615RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82615 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51277⟩⟩) exact82615RawTerms .large 82613 .exactZero (none)

def event82616 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7206⟩⟩) 0 ⟨7177⟩ 82569

def event82617 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7206⟩⟩) (.authority (.operator))

def exact82618RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩]

theorem exact82618RawTermsValid :
    exact82618RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82618 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7206⟩⟩) exact82618RawTerms .large 82617 .exactZero (none)

def event82619 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51278⟩⟩) 0 ⟨7206⟩ 82618

def event82620 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51278⟩⟩) 1 ⟨51277⟩ 82615

def event82621 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51278⟩⟩) (.sum [.predecessor 0 82619 .coefficient, .predecessor 1 82620 .coefficient])

def exact82622RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82622RawTermsValid :
    exact82622RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82622 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51278⟩⟩) exact82622RawTerms .large 82621 .exactZero (none)

def event82623 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53143⟩⟩) 0 ⟨51278⟩ 82622

def event82624 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53143⟩⟩) 1 ⟨53139⟩ 82607

def event82625 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53143⟩⟩) (.sum [.predecessor 0 82623 .coefficient, .predecessor 1 82624 .coefficient])

def exact82626RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨52215⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82626RawTermsValid :
    exact82626RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82626 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53143⟩⟩) exact82626RawTerms .large 82625 .exactZero (none)

def event82627 : Event := .preFoldPolynomial 82626 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨52215⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact82628RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨52215⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event82628 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨53143⟩⟩) 82627 exact82628RawTerms .large 82625 .exactZero (none)

def event82629 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨50937⟩⟩) ⟨⟨85⟩, ⟨65⟩, ⟨135⟩⟩ ⟨82471, 82629⟩

def event82630 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨51879⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51876⟩⟩]⟩) (1) 0 2 (.universal 82629 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51876⟩⟩]⟩) (none) 82628)

def event82631 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51879⟩⟩, .relation 82630 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩)

def event82632 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51879⟩⟩, .relation 82630 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩, (-1)⟩)

def event82633 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51879⟩⟩, .relation 82630 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨52215⟩⟩]⟩, (1)⟩)

def event82634 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51879⟩⟩, .relation 82630 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact82635RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨52215⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82635RawTermsValid :
    exact82635RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82635 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51879⟩⟩) exact82635RawTerms .large 82467 (.finite 202072841853861888) (some (82469))

def event82636 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53141⟩⟩) 0 ⟨51879⟩ 82635

def event82637 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53141⟩⟩) 1 ⟨53140⟩ 82457

def event82638 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53141⟩⟩) (.sum [.predecessor 0 82636 .coefficient, .predecessor 1 82637 .coefficient])

def event82639 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53141⟩⟩, .operator (⟨82635, 0⟩, ⟨82457, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53138⟩⟩]⟩, (1)⟩)

def event82640 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53141⟩⟩, .operator (⟨82635, 2⟩, ⟨82457, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨50936⟩⟩], [⟨.program ⟨257⟩, ⟨52215⟩⟩]⟩, (-1)⟩)

def event82641 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53141⟩⟩) (.sum [.result 82635 .summary, .result 82457 .summary])

def exact82642RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82642RawTermsValid :
    exact82642RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82642 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53141⟩⟩) exact82642RawTerms .large 82638 (.finite 32189593014266456398474184491008) (some (82641))

def event82643 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33153⟩⟩) 0 ⟨31877⟩ 3425

def event82644 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33153⟩⟩) (.authority (.programFamilyFact))

def event82645 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨33153⟩⟩) (.finite 3720)

def event82646 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33155⟩⟩) 0 ⟨7177⟩ 15500

def event82647 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33155⟩⟩) 1 ⟨33153⟩ 82645

def event82648 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33155⟩⟩) (.authority (.operator))

def exact82649RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨33155⟩⟩]⟩, (1)⟩]

theorem exact82649RawTermsValid :
    exact82649RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82649 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33155⟩⟩) exact82649RawTerms .large 82648 .exactZero (none)

def event82650 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34078⟩⟩) 0 ⟨33155⟩ 82649

def event82651 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34078⟩⟩) (.authority (.operator))

def exact82652RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩, (1)⟩]

theorem exact82652RawTermsValid :
    exact82652RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82652 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34078⟩⟩) exact82652RawTerms (.finite 8192) 82651 .exactZero (none)

def event82653 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32984⟩⟩) 0 ⟨31649⟩ 3419

def event82654 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32984⟩⟩) (.authority (.programFamilyFact))

def event82655 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨32984⟩⟩) (.finite 3720)

def event82656 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32985⟩⟩) 0 ⟨7177⟩ 15500

def event82657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32985⟩⟩) 1 ⟨32984⟩ 82655

def event82658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32985⟩⟩) (.authority (.operator))

def exact82659RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32985⟩⟩]⟩, (1)⟩]

theorem exact82659RawTermsValid :
    exact82659RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82659 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32985⟩⟩) exact82659RawTerms .large 82658 .exactZero (none)

def event82660 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33525⟩⟩) 0 ⟨32985⟩ 82659

def event82661 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33525⟩⟩) (.authority (.operator))

def exact82662RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨33525⟩⟩]⟩, (1)⟩]

theorem exact82662RawTermsValid :
    exact82662RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82662 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33525⟩⟩) exact82662RawTerms (.finite 8192) 82661 .exactZero (none)

def event82663 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24363⟩⟩) 0 ⟨24362⟩ 3408

def event82664 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24363⟩⟩) 1 ⟨10328⟩ 75903

def event82665 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24363⟩⟩) (.tensor (.predecessor 0 82663 .coefficient) (.predecessor 1 82664 .coefficient) true false)

def event82666 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24363⟩⟩, .operator (⟨3408, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact82667RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact82667RawTermsValid :
    exact82667RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82667 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24363⟩⟩) exact82667RawTerms .large 82665 .exactZero (none)

def event82668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10365⟩⟩) 0 ⟨10327⟩ 75773

def event82669 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10365⟩⟩) 1 ⟨7307⟩ 24094

def event82670 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10365⟩⟩) (.product (.predecessor 0 82668 .coefficient) (.predecessor 1 82669 .coefficient) (⟨false, false, none, none, none⟩))

def event82671 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10365⟩⟩, .operator (⟨75773, 0⟩, ⟨24094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩)

def exact82672RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩]

theorem exact82672RawTermsValid :
    exact82672RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82672 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10365⟩⟩) exact82672RawTerms .large 82670 .exactZero (none)

def event82673 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24364⟩⟩) 0 ⟨10365⟩ 82672

def event82674 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24364⟩⟩) 1 ⟨24363⟩ 82667

def event82675 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24364⟩⟩) (.sum [.predecessor 0 82673 .coefficient, .predecessor 1 82674 .coefficient])

def exact82676RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82676RawTermsValid :
    exact82676RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82676 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24364⟩⟩) exact82676RawTerms .large 82675 .exactZero (none)

def event82677 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24365⟩⟩) 0 ⟨24364⟩ 82676

def event82678 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24365⟩⟩) 1 ⟨133⟩ 24086

def event82679 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24365⟩⟩) (.sum [.predecessor 0 82677 .coefficient, .predecessor 1 82678 .coefficient])

def event82680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24365⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨133⟩⟩]⟩) [⟨.result 24086 .coefficient, false, none⟩])

def event82681 : Event := .survivorFold (1) 82680

def exact82682RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24362⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact82682RawTermsValid :
    exact82682RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82682 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24365⟩⟩) exact82682RawTerms .large 82679 (.finite 26) (some (82680))

def event82683 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31650⟩⟩) 0 ⟨24365⟩ 82682

def event82684 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31650⟩⟩) 1 ⟨31647⟩ 3411

def event82685 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31650⟩⟩) (.product (.predecessor 0 82683 .coefficient) (.predecessor 1 82684 .coefficient) (⟨false, true, none, none, some 1⟩))

def event82686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31650⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨31647⟩⟩], []⟩) [⟨.result 3411 .coefficient, true, some 1⟩])

def event82687 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31650⟩⟩) (.product (.result 82682 .summary) (.transfer 82686) (⟨false, false, none, none, none⟩))

def eventLeaf5152 : Array AnnotatedEvent := #[
  { event := event82432
    frameStart := 82316 },
  { event := event82433
    frameStart := 82316 },
  { event := event82434
    frameStart := 0 },
  { event := event82435
    frameStart := 0 },
  { event := event82436
    frameStart := 0 },
  { event := event82437
    frameStart := 0 },
  { event := event82438
    frameStart := 0 },
  { event := event82439
    frameStart := 0 },
  { event := event82440
    frameStart := 0 },
  { event := event82441
    frameStart := 0 },
  { event := event82442
    frameStart := 0 },
  { event := event82443
    frameStart := 0 },
  { event := event82444
    frameStart := 0 },
  { event := event82445
    frameStart := 0 },
  { event := event82446
    frameStart := 0 },
  { event := event82447
    frameStart := 0 }
]

def eventLeaf5153 : Array AnnotatedEvent := #[
  { event := event82448
    frameStart := 0 },
  { event := event82449
    frameStart := 0 },
  { event := event82450
    frameStart := 0 },
  { event := event82451
    frameStart := 0 },
  { event := event82452
    frameStart := 0 },
  { event := event82453
    frameStart := 0 },
  { event := event82454
    frameStart := 0 },
  { event := event82455
    frameStart := 0 },
  { event := event82456
    frameStart := 0 },
  { event := event82457
    frameStart := 0 },
  { event := event82458
    frameStart := 0 },
  { event := event82459
    frameStart := 0 },
  { event := event82460
    frameStart := 0 },
  { event := event82461
    frameStart := 0 },
  { event := event82462
    frameStart := 0 },
  { event := event82463
    frameStart := 0 }
]

def eventLeaf5154 : Array AnnotatedEvent := #[
  { event := event82464
    frameStart := 0 },
  { event := event82465
    frameStart := 0 },
  { event := event82466
    frameStart := 0 },
  { event := event82467
    frameStart := 0 },
  { event := event82468
    frameStart := 0 },
  { event := event82469
    frameStart := 0 },
  { event := event82470
    frameStart := 0 },
  { event := event82471
    frameStart := 82471 },
  { event := event82472
    frameStart := 82471 },
  { event := event82473
    frameStart := 82471 },
  { event := event82474
    frameStart := 82471 },
  { event := event82475
    frameStart := 82471 },
  { event := event82476
    frameStart := 82471 },
  { event := event82477
    frameStart := 82471 },
  { event := event82478
    frameStart := 82471 },
  { event := event82479
    frameStart := 82471 }
]

def eventLeaf5155 : Array AnnotatedEvent := #[
  { event := event82480
    frameStart := 82471 },
  { event := event82481
    frameStart := 82471 },
  { event := event82482
    frameStart := 82471 },
  { event := event82483
    frameStart := 82471 },
  { event := event82484
    frameStart := 82471 },
  { event := event82485
    frameStart := 82471 },
  { event := event82486
    frameStart := 82471 },
  { event := event82487
    frameStart := 82471 },
  { event := event82488
    frameStart := 82471 },
  { event := event82489
    frameStart := 82471 },
  { event := event82490
    frameStart := 82471 },
  { event := event82491
    frameStart := 82471 },
  { event := event82492
    frameStart := 82471 },
  { event := event82493
    frameStart := 82471 },
  { event := event82494
    frameStart := 82471 },
  { event := event82495
    frameStart := 82471 }
]

def eventLeaf5156 : Array AnnotatedEvent := #[
  { event := event82496
    frameStart := 82471 },
  { event := event82497
    frameStart := 82471 },
  { event := event82498
    frameStart := 82471 },
  { event := event82499
    frameStart := 82471 },
  { event := event82500
    frameStart := 82471 },
  { event := event82501
    frameStart := 82471 },
  { event := event82502
    frameStart := 82471 },
  { event := event82503
    frameStart := 82471 },
  { event := event82504
    frameStart := 82471 },
  { event := event82505
    frameStart := 82471 },
  { event := event82506
    frameStart := 82471 },
  { event := event82507
    frameStart := 82471 },
  { event := event82508
    frameStart := 82471 },
  { event := event82509
    frameStart := 82471 },
  { event := event82510
    frameStart := 82471 },
  { event := event82511
    frameStart := 82471 }
]

def eventLeaf5157 : Array AnnotatedEvent := #[
  { event := event82512
    frameStart := 82471 },
  { event := event82513
    frameStart := 82471 },
  { event := event82514
    frameStart := 82471 },
  { event := event82515
    frameStart := 82471 },
  { event := event82516
    frameStart := 82471 },
  { event := event82517
    frameStart := 82471 },
  { event := event82518
    frameStart := 82471 },
  { event := event82519
    frameStart := 82471 },
  { event := event82520
    frameStart := 82471 },
  { event := event82521
    frameStart := 82471 },
  { event := event82522
    frameStart := 82471 },
  { event := event82523
    frameStart := 82471 },
  { event := event82524
    frameStart := 82471 },
  { event := event82525
    frameStart := 82525 },
  { event := event82526
    frameStart := 82525 },
  { event := event82527
    frameStart := 82525 }
]

def eventLeaf5158 : Array AnnotatedEvent := #[
  { event := event82528
    frameStart := 82525 },
  { event := event82529
    frameStart := 82525 },
  { event := event82530
    frameStart := 82525 },
  { event := event82531
    frameStart := 82525 },
  { event := event82532
    frameStart := 82525 },
  { event := event82533
    frameStart := 82525 },
  { event := event82534
    frameStart := 82525 },
  { event := event82535
    frameStart := 82525 },
  { event := event82536
    frameStart := 82525 },
  { event := event82537
    frameStart := 82525 },
  { event := event82538
    frameStart := 82525 },
  { event := event82539
    frameStart := 82525 },
  { event := event82540
    frameStart := 82525 },
  { event := event82541
    frameStart := 82525 },
  { event := event82542
    frameStart := 82525 },
  { event := event82543
    frameStart := 82525 }
]

def eventLeaf5159 : Array AnnotatedEvent := #[
  { event := event82544
    frameStart := 82525 },
  { event := event82545
    frameStart := 82525 },
  { event := event82546
    frameStart := 82525 },
  { event := event82547
    frameStart := 82525 },
  { event := event82548
    frameStart := 82525 },
  { event := event82549
    frameStart := 82525 },
  { event := event82550
    frameStart := 82525 },
  { event := event82551
    frameStart := 82525 },
  { event := event82552
    frameStart := 82525 },
  { event := event82553
    frameStart := 82525 },
  { event := event82554
    frameStart := 82525 },
  { event := event82555
    frameStart := 82525 },
  { event := event82556
    frameStart := 82525 },
  { event := event82557
    frameStart := 82525 },
  { event := event82558
    frameStart := 82525 },
  { event := event82559
    frameStart := 82525 }
]

def eventLeaf5160 : Array AnnotatedEvent := #[
  { event := event82560
    frameStart := 82525 },
  { event := event82561
    frameStart := 82525 },
  { event := event82562
    frameStart := 82525 },
  { event := event82563
    frameStart := 82525 },
  { event := event82564
    frameStart := 82525 },
  { event := event82565
    frameStart := 82525 },
  { event := event82566
    frameStart := 82525 },
  { event := event82567
    frameStart := 82525 },
  { event := event82568
    frameStart := 82525 },
  { event := event82569
    frameStart := 82525 },
  { event := event82570
    frameStart := 82525 },
  { event := event82571
    frameStart := 82525 },
  { event := event82572
    frameStart := 82525 },
  { event := event82573
    frameStart := 82525 },
  { event := event82574
    frameStart := 82525 },
  { event := event82575
    frameStart := 82525 }
]

def eventLeaf5161 : Array AnnotatedEvent := #[
  { event := event82576
    frameStart := 82525 },
  { event := event82577
    frameStart := 82525 },
  { event := event82578
    frameStart := 82525 },
  { event := event82579
    frameStart := 82525 },
  { event := event82580
    frameStart := 82525 },
  { event := event82581
    frameStart := 82525 },
  { event := event82582
    frameStart := 82525 },
  { event := event82583
    frameStart := 82525 },
  { event := event82584
    frameStart := 82525 },
  { event := event82585
    frameStart := 82525 },
  { event := event82586
    frameStart := 82525 },
  { event := event82587
    frameStart := 82525 },
  { event := event82588
    frameStart := 82525 },
  { event := event82589
    frameStart := 82525 },
  { event := event82590
    frameStart := 82525 },
  { event := event82591
    frameStart := 82525 }
]

def eventLeaf5162 : Array AnnotatedEvent := #[
  { event := event82592
    frameStart := 82525 },
  { event := event82593
    frameStart := 82525 },
  { event := event82594
    frameStart := 82525 },
  { event := event82595
    frameStart := 82525 },
  { event := event82596
    frameStart := 82525 },
  { event := event82597
    frameStart := 82525 },
  { event := event82598
    frameStart := 82525 },
  { event := event82599
    frameStart := 82525 },
  { event := event82600
    frameStart := 82525 },
  { event := event82601
    frameStart := 82525 },
  { event := event82602
    frameStart := 82525 },
  { event := event82603
    frameStart := 82525 },
  { event := event82604
    frameStart := 82525 },
  { event := event82605
    frameStart := 82525 },
  { event := event82606
    frameStart := 82525 },
  { event := event82607
    frameStart := 82525 }
]

def eventLeaf5163 : Array AnnotatedEvent := #[
  { event := event82608
    frameStart := 82525 },
  { event := event82609
    frameStart := 82525 },
  { event := event82610
    frameStart := 82525 },
  { event := event82611
    frameStart := 82525 },
  { event := event82612
    frameStart := 82525 },
  { event := event82613
    frameStart := 82525 },
  { event := event82614
    frameStart := 82525 },
  { event := event82615
    frameStart := 82525 },
  { event := event82616
    frameStart := 82525 },
  { event := event82617
    frameStart := 82525 },
  { event := event82618
    frameStart := 82525 },
  { event := event82619
    frameStart := 82525 },
  { event := event82620
    frameStart := 82525 },
  { event := event82621
    frameStart := 82525 },
  { event := event82622
    frameStart := 82525 },
  { event := event82623
    frameStart := 82525 }
]

def eventLeaf5164 : Array AnnotatedEvent := #[
  { event := event82624
    frameStart := 82525 },
  { event := event82625
    frameStart := 82525 },
  { event := event82626
    frameStart := 82525 },
  { event := event82627
    frameStart := 82525 },
  { event := event82628
    frameStart := 82525 },
  { event := event82629
    frameStart := 0 },
  { event := event82630
    frameStart := 0 },
  { event := event82631
    frameStart := 0 },
  { event := event82632
    frameStart := 0 },
  { event := event82633
    frameStart := 0 },
  { event := event82634
    frameStart := 0 },
  { event := event82635
    frameStart := 0 },
  { event := event82636
    frameStart := 0 },
  { event := event82637
    frameStart := 0 },
  { event := event82638
    frameStart := 0 },
  { event := event82639
    frameStart := 0 }
]

def eventLeaf5165 : Array AnnotatedEvent := #[
  { event := event82640
    frameStart := 0 },
  { event := event82641
    frameStart := 0 },
  { event := event82642
    frameStart := 0 },
  { event := event82643
    frameStart := 0 },
  { event := event82644
    frameStart := 0 },
  { event := event82645
    frameStart := 0 },
  { event := event82646
    frameStart := 0 },
  { event := event82647
    frameStart := 0 },
  { event := event82648
    frameStart := 0 },
  { event := event82649
    frameStart := 0 },
  { event := event82650
    frameStart := 0 },
  { event := event82651
    frameStart := 0 },
  { event := event82652
    frameStart := 0 },
  { event := event82653
    frameStart := 0 },
  { event := event82654
    frameStart := 0 },
  { event := event82655
    frameStart := 0 }
]

def eventLeaf5166 : Array AnnotatedEvent := #[
  { event := event82656
    frameStart := 0 },
  { event := event82657
    frameStart := 0 },
  { event := event82658
    frameStart := 0 },
  { event := event82659
    frameStart := 0 },
  { event := event82660
    frameStart := 0 },
  { event := event82661
    frameStart := 0 },
  { event := event82662
    frameStart := 0 },
  { event := event82663
    frameStart := 0 },
  { event := event82664
    frameStart := 0 },
  { event := event82665
    frameStart := 0 },
  { event := event82666
    frameStart := 0 },
  { event := event82667
    frameStart := 0 },
  { event := event82668
    frameStart := 0 },
  { event := event82669
    frameStart := 0 },
  { event := event82670
    frameStart := 0 },
  { event := event82671
    frameStart := 0 }
]

def eventLeaf5167 : Array AnnotatedEvent := #[
  { event := event82672
    frameStart := 0 },
  { event := event82673
    frameStart := 0 },
  { event := event82674
    frameStart := 0 },
  { event := event82675
    frameStart := 0 },
  { event := event82676
    frameStart := 0 },
  { event := event82677
    frameStart := 0 },
  { event := event82678
    frameStart := 0 },
  { event := event82679
    frameStart := 0 },
  { event := event82680
    frameStart := 0 },
  { event := event82681
    frameStart := 0 },
  { event := event82682
    frameStart := 0 },
  { event := event82683
    frameStart := 0 },
  { event := event82684
    frameStart := 0 },
  { event := event82685
    frameStart := 0 },
  { event := event82686
    frameStart := 0 },
  { event := event82687
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events322
