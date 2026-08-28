import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events338

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact86528RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], []⟩, (1)⟩]

theorem exact86528RawTermsValid :
    exact86528RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86528 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45299⟩⟩) exact86528RawTerms (.finite 3364) 86526 .exactZero (none)

def event86529 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45300⟩⟩) 0 ⟨45299⟩ 86528

def event86530 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45300⟩⟩) (.identity (.predecessor 0 86529 .coefficient))

def event86531 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45300⟩⟩) (.finite 3364)

def event86532 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45516⟩⟩) 0 ⟨45300⟩ 86531

def event86533 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45516⟩⟩) (.authority (.programFamilyFact))

def exact86534RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], []⟩, (1)⟩]

theorem exact86534RawTermsValid :
    exact86534RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86534 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45516⟩⟩) exact86534RawTerms (.finite 58) 86533 .exactZero (none)

def event86535 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45517⟩⟩) 0 ⟨45516⟩ 86534

def event86536 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45517⟩⟩) (.identity (.predecessor 0 86535 .coefficient))

def event86537 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45517⟩⟩) (.finite 58)

def event86538 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46673⟩⟩) 0 ⟨45517⟩ 86537

def event86539 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46673⟩⟩) (.authority (.programFamilyFact))

def event86540 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46673⟩⟩) (.finite 3720)

def event86541 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event86542 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46674⟩⟩) 0 ⟨7177⟩ 86541

def event86543 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46674⟩⟩) 1 ⟨46673⟩ 86540

def event86544 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46674⟩⟩) (.authority (.operator))

def exact86545RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46674⟩⟩]⟩, (1)⟩]

theorem exact86545RawTermsValid :
    exact86545RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86545 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46674⟩⟩) exact86545RawTerms .large 86544 .exactZero (none)

def event86546 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47493⟩⟩) 0 ⟨46674⟩ 86545

def event86547 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47493⟩⟩) (.authority (.operator))

def exact86548RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47493⟩⟩]⟩, (1)⟩]

theorem exact86548RawTermsValid :
    exact86548RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86548 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47493⟩⟩) exact86548RawTerms (.finite 8192) 86547 .exactZero (none)

def event86549 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event86550 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event86551 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46850⟩⟩) 0 ⟨45517⟩ 86537

def event86552 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46850⟩⟩) 1 ⟨136⟩ 86550

def event86553 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46850⟩⟩) (.sum [.predecessor 0 86551 .coefficient, .predecessor 1 86552 .coefficient])

def event86554 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46850⟩⟩) (.finite 58)

def event86555 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46851⟩⟩) 0 ⟨46850⟩ 86554

def event86556 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46851⟩⟩) (.identity (.predecessor 0 86555 .coefficient))

def exact86557RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], []⟩, (1)⟩]

theorem exact86557RawTermsValid :
    exact86557RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86557 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46851⟩⟩) exact86557RawTerms (.finite 58) 86556 .exactZero (none)

def event86558 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact86559RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact86559RawTermsValid :
    exact86559RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86559 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact86559RawTerms .large 86558 .exactZero (none)

def event86560 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46852⟩⟩) 0 ⟨6908⟩ 86559

def event86561 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46852⟩⟩) 1 ⟨46851⟩ 86557

def event86562 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46852⟩⟩) (.product (.predecessor 0 86560 .coefficient) (.predecessor 1 86561 .coefficient) (⟨false, false, none, none, none⟩))

def event86563 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46852⟩⟩, .operator (⟨86559, 0⟩, ⟨86557, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact86564RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact86564RawTermsValid :
    exact86564RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86564 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46852⟩⟩) exact86564RawTerms .large 86562 .exactZero (none)

def event86565 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7195⟩⟩) 0 ⟨7177⟩ 86541

def event86566 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7195⟩⟩) (.authority (.operator))

def exact86567RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩]

theorem exact86567RawTermsValid :
    exact86567RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86567 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7195⟩⟩) exact86567RawTerms .large 86566 .exactZero (none)

def event86568 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46853⟩⟩) 0 ⟨7195⟩ 86567

def event86569 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46853⟩⟩) 1 ⟨46852⟩ 86564

def event86570 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46853⟩⟩) (.sum [.predecessor 0 86568 .coefficient, .predecessor 1 86569 .coefficient])

def exact86571RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact86571RawTermsValid :
    exact86571RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86571 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46853⟩⟩) exact86571RawTerms .large 86570 .exactZero (none)

def event86572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47494⟩⟩) 0 ⟨46853⟩ 86571

def event86573 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47494⟩⟩) 1 ⟨47493⟩ 86548

def event86574 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47494⟩⟩) (.product (.predecessor 0 86572 .coefficient) (.predecessor 1 86573 .coefficient) (⟨false, false, none, none, none⟩))

def event86575 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47494⟩⟩, .operator (⟨86571, 0⟩, ⟨86548, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47493⟩⟩]⟩, (1)⟩)

def event86576 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47494⟩⟩, .operator (⟨86571, 1⟩, ⟨86548, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47493⟩⟩]⟩, (-1)⟩)

def event86577 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47494⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47493⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47493⟩⟩) ⟨46674⟩ 86545)

def event86578 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47494⟩⟩, .relation 86577 0, ⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨46674⟩⟩]⟩, (-1)⟩)

def exact86579RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47493⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨46674⟩⟩]⟩, (-1)⟩]

theorem exact86579RawTermsValid :
    exact86579RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86579 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47494⟩⟩) exact86579RawTerms .large 86574 .exactZero (none)

def event86580 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45757⟩⟩) 0 ⟨45517⟩ 86537

def event86581 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45757⟩⟩) (.authority (.programFamilyFact))

def exact86582RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45757⟩⟩], []⟩, (1)⟩]

theorem exact86582RawTermsValid :
    exact86582RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86582 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45757⟩⟩) exact86582RawTerms (.finite 58) 86581 .exactZero (none)

def event86583 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45759⟩⟩) 0 ⟨6908⟩ 86559

def event86584 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45759⟩⟩) 1 ⟨45757⟩ 86582

def event86585 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45759⟩⟩) (.product (.predecessor 0 86583 .coefficient) (.predecessor 1 86584 .coefficient) (⟨false, true, none, none, some 1⟩))

def event86586 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45759⟩⟩, .operator (⟨86559, 0⟩, ⟨86582, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45757⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact86587RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45757⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact86587RawTermsValid :
    exact86587RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86587 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45759⟩⟩) exact86587RawTerms .large 86585 .exactZero (none)

def event86588 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7229⟩⟩) 0 ⟨7177⟩ 86541

def event86589 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7229⟩⟩) (.authority (.operator))

def exact86590RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩]

theorem exact86590RawTermsValid :
    exact86590RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86590 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7229⟩⟩) exact86590RawTerms .large 86589 .exactZero (none)

def event86591 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45760⟩⟩) 0 ⟨7229⟩ 86590

def event86592 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45760⟩⟩) 1 ⟨45759⟩ 86587

def event86593 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45760⟩⟩) (.sum [.predecessor 0 86591 .coefficient, .predecessor 1 86592 .coefficient])

def exact86594RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45757⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact86594RawTermsValid :
    exact86594RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86594 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45760⟩⟩) exact86594RawTerms .large 86593 .exactZero (none)

def event86595 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47498⟩⟩) 0 ⟨45760⟩ 86594

def event86596 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47498⟩⟩) 1 ⟨47494⟩ 86579

def event86597 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47498⟩⟩) (.sum [.predecessor 0 86595 .coefficient, .predecessor 1 86596 .coefficient])

def exact86598RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47493⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨46674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45757⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact86598RawTermsValid :
    exact86598RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86598 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47498⟩⟩) exact86598RawTerms .large 86597 .exactZero (none)

def event86599 : Event := .preFoldPolynomial 86598 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47493⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨46674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45757⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact86600RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47493⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨46674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45757⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event86600 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨47498⟩⟩) 86599 exact86600RawTerms .large 86597 .exactZero (none)

def event86601 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨45517⟩⟩) ⟨⟨108⟩, ⟨91⟩, ⟨135⟩⟩ ⟨86443, 86601⟩

def event86602 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨46335⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46332⟩⟩]⟩) (1) 0 2 (.universal 86601 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46332⟩⟩]⟩) (none) 86600)

def event86603 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46335⟩⟩, .relation 86602 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩)

def event86604 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46335⟩⟩, .relation 86602 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47493⟩⟩]⟩, (-1)⟩)

def event86605 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46335⟩⟩, .relation 86602 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨46674⟩⟩]⟩, (1)⟩)

def event86606 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46335⟩⟩, .relation 86602 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact86607RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47493⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨46674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact86607RawTermsValid :
    exact86607RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86607 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46335⟩⟩) exact86607RawTerms .large 86439 (.finite 202072841853861888) (some (86441))

def event86608 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47496⟩⟩) 0 ⟨46335⟩ 86607

def event86609 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47496⟩⟩) 1 ⟨47495⟩ 86429

def event86610 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47496⟩⟩) (.sum [.predecessor 0 86608 .coefficient, .predecessor 1 86609 .coefficient])

def event86611 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47496⟩⟩, .operator (⟨86607, 0⟩, ⟨86429, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47493⟩⟩]⟩, (1)⟩)

def event86612 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47496⟩⟩, .operator (⟨86607, 2⟩, ⟨86429, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨46674⟩⟩]⟩, (-1)⟩)

def event86613 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47496⟩⟩) (.sum [.result 86607 .summary, .result 86429 .summary])

def exact86614RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact86614RawTermsValid :
    exact86614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86614 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47496⟩⟩) exact86614RawTerms .large 86610 (.finite 32194307824962953452255538577408) (some (86613))

def event86615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47497⟩⟩) 0 ⟨47496⟩ 86614

def event86616 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47497⟩⟩) 1 ⟨7152⟩ 15562

def event86617 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47497⟩⟩) (.product (.predecessor 0 86615 .coefficient) (.predecessor 1 86616 .coefficient) (⟨false, false, none, none, none⟩))

def event86618 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47497⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩) [⟨.result 15558 .coefficient, false, none⟩])

def event86619 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47497⟩⟩) (.product (.result 86614 .summary) (.transfer 86618) (⟨false, false, none, none, none⟩))

def event86620 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47497⟩⟩, .operator (⟨86614, 0⟩, ⟨15562, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩)

def event86621 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47497⟩⟩, .operator (⟨86614, 1⟩, ⟨15562, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (-1)⟩)

def event86622 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47497⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7151⟩⟩) ⟨7041⟩ 15555)

def event86623 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47497⟩⟩, .relation 86622 0, ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact86624RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩]

theorem exact86624RawTermsValid :
    exact86624RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86624 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47497⟩⟩) exact86624RawTerms .large 86617 (.finite 345683748063931943722519589062084311121920) (some (86619))

def event86625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43994⟩⟩) 0 ⟨7177⟩ 15500

def event86626 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43994⟩⟩) 1 ⟨43993⟩ 76861

def event86627 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43994⟩⟩) (.authority (.operator))

def exact86628RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43994⟩⟩]⟩, (1)⟩]

theorem exact86628RawTermsValid :
    exact86628RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86628 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43994⟩⟩) exact86628RawTerms .large 86627 .exactZero (none)

def event86629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44813⟩⟩) 0 ⟨43994⟩ 86628

def event86630 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44813⟩⟩) (.authority (.operator))

def exact86631RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44813⟩⟩]⟩, (1)⟩]

theorem exact86631RawTermsValid :
    exact86631RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86631 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44813⟩⟩) exact86631RawTerms (.finite 8192) 86630 .exactZero (none)

def event86632 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44815⟩⟩) 0 ⟨44367⟩ 77145

def event86633 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44815⟩⟩) 1 ⟨44813⟩ 86631

def event86634 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44815⟩⟩) (.product (.predecessor 0 86632 .coefficient) (.predecessor 1 86633 .coefficient) (⟨false, false, none, none, none⟩))

def event86635 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44815⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨44813⟩⟩]⟩) [⟨.result 86631 .coefficient, false, none⟩])

def event86636 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44815⟩⟩) (.product (.result 77145 .summary) (.transfer 86635) (⟨false, false, none, none, none⟩))

def event86637 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44815⟩⟩, .operator (⟨77145, 0⟩, ⟨86631, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44813⟩⟩]⟩, (1)⟩)

def event86638 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44815⟩⟩, .operator (⟨77145, 1⟩, ⟨86631, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44813⟩⟩]⟩, (-1)⟩)

def event86639 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44815⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44813⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44813⟩⟩) ⟨43994⟩ 86628)

def event86640 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44815⟩⟩, .relation 86639 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨43994⟩⟩]⟩, (-1)⟩)

def exact86641RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44813⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨43994⟩⟩]⟩, (-1)⟩]

theorem exact86641RawTermsValid :
    exact86641RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86641 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44815⟩⟩) exact86641RawTerms .large 86634 (.finite 32193718473625689247691015454720) (some (86636))

def event86642 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43652⟩⟩) 0 ⟨42837⟩ 3149

def event86643 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43652⟩⟩) (.authority (.relationPreimageSource ⟨89⟩))

def exact86644RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43652⟩⟩]⟩, (1)⟩]

theorem exact86644RawTermsValid :
    exact86644RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86644 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43652⟩⟩) exact86644RawTerms (.finite 5647228698) 86643 .exactZero (none)

def event86645 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43654⟩⟩) 0 ⟨43652⟩ 86644

def event86646 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43654⟩⟩) 1 ⟨2370⟩ 4

def event86647 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43654⟩⟩) (.scale (.predecessor 0 86645 .coefficient) (.value (.predecessor 1 86646 .coefficient)))

def exact86648RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43652⟩⟩]⟩, (1)⟩]

theorem exact86648RawTermsValid :
    exact86648RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86648 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43654⟩⟩) exact86648RawTerms (.finite 5647228698) 86647 .exactZero (none)

def event86649 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43655⟩⟩) 0 ⟨10368⟩ 75995

def event86650 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43655⟩⟩) 1 ⟨43654⟩ 86648

def event86651 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43655⟩⟩) (.product (.predecessor 0 86649 .coefficient) (.predecessor 1 86650 .coefficient) (⟨false, false, none, none, none⟩))

def event86652 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43655⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨43652⟩⟩]⟩) [⟨.result 86644 .coefficient, false, none⟩])

def event86653 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43655⟩⟩) (.product (.result 75995 .summary) (.transfer 86652) (⟨false, false, none, none, none⟩))

def event86654 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43655⟩⟩, .operator (⟨75995, 0⟩, ⟨86648, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43652⟩⟩]⟩, (1)⟩)

def event86655 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨43653⟩⟩)

def event86656 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event86657 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event86658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event86659 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event86660 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event86661 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event86662 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event86663 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event86664 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 86663

def event86665 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 86661

def event86666 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 86664 .coefficient) (.value (.predecessor 1 86665 .coefficient)))

def event86667 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event86668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 86667

def event86669 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 86659

def event86670 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 86668 .coefficient, .predecessor 1 86669 .coefficient])

def event86671 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event86672 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 86671

def event86673 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 86657

def event86674 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 86673 .coefficient))

def event86675 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event86676 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42618⟩⟩) 0 ⟨10325⟩ 86675

def event86677 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42618⟩⟩) (.authority (.programFamilyFact))

def exact86678RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42618⟩⟩], []⟩, (1)⟩]

theorem exact86678RawTermsValid :
    exact86678RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86678 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42618⟩⟩) exact86678RawTerms (.finite 52) 86677 .exactZero (none)

def event86679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14571⟩⟩) 0 ⟨10325⟩ 86675

def event86680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14571⟩⟩) (.authority (.programFamilyFact))

def exact86681RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩], []⟩, (1)⟩]

theorem exact86681RawTermsValid :
    exact86681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86681 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14571⟩⟩) exact86681RawTerms (.finite 52) 86680 .exactZero (none)

def event86682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42619⟩⟩) 0 ⟨14571⟩ 86681

def event86683 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42619⟩⟩) 1 ⟨42618⟩ 86678

def event86684 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42619⟩⟩) (.product (.predecessor 0 86682 .coefficient) (.predecessor 1 86683 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event86685 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42619⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], []⟩) [⟨.result 86681 .coefficient, true, some 1⟩, ⟨.result 86678 .coefficient, true, some 1⟩])

def event86686 : Event := .survivorFold (1) 86685

def exact86687RawTerms : List Term := []

theorem exact86687RawTermsValid :
    exact86687RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86687 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42619⟩⟩) exact86687RawTerms (.finite 2704) 86684 (.finite 2704) (some (86685))

def event86688 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42620⟩⟩) 0 ⟨42619⟩ 86687

def event86689 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42620⟩⟩) (.identity (.predecessor 0 86688 .coefficient))

def event86690 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42620⟩⟩) (.finite 2704)

def event86691 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42836⟩⟩) 0 ⟨42620⟩ 86690

def event86692 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42836⟩⟩) (.authority (.programFamilyFact))

def exact86693RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], []⟩, (1)⟩]

theorem exact86693RawTermsValid :
    exact86693RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86693 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42836⟩⟩) exact86693RawTerms (.finite 52) 86692 .exactZero (none)

def event86694 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42837⟩⟩) 0 ⟨42836⟩ 86693

def event86695 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42837⟩⟩) (.identity (.predecessor 0 86694 .coefficient))

def event86696 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42837⟩⟩) (.finite 52)

def event86697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43652⟩⟩) 0 ⟨42837⟩ 86696

def event86698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43652⟩⟩) (.authority (.relationPreimageSource ⟨89⟩))

def exact86699RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43652⟩⟩]⟩, (1)⟩]

theorem exact86699RawTermsValid :
    exact86699RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86699 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43652⟩⟩) exact86699RawTerms (.finite 5647228698) 86698 .exactZero (none)

def event86700 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact86701RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact86701RawTermsValid :
    exact86701RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86701 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact86701RawTerms .large 86700 .exactZero (none)

def event86702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43653⟩⟩) 0 ⟨35⟩ 86701

def event86703 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43653⟩⟩) 1 ⟨43652⟩ 86699

def event86704 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43653⟩⟩) (.product (.predecessor 0 86702 .coefficient) (.predecessor 1 86703 .coefficient) (⟨false, false, none, none, none⟩))

def event86705 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43653⟩⟩, .operator (⟨86701, 0⟩, ⟨86699, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43652⟩⟩]⟩, (1)⟩)

def exact86706RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43652⟩⟩]⟩, (1)⟩]

theorem exact86706RawTermsValid :
    exact86706RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86706 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43653⟩⟩) exact86706RawTerms .large 86704 .exactZero (none)

def event86707 : Event := .preFoldPolynomial 86706 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43652⟩⟩]⟩, (1)⟩] .exactZero none

def exact86708RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43652⟩⟩]⟩, (1)⟩]

def event86708 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨43653⟩⟩) 86707 exact86708RawTerms .large 86704 .exactZero (none)

def event86709 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨44818⟩⟩)

def event86710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event86711 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event86712 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event86713 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event86714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event86715 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event86716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event86717 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event86718 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 86717

def event86719 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 86715

def event86720 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 86718 .coefficient) (.value (.predecessor 1 86719 .coefficient)))

def event86721 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event86722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 86721

def event86723 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 86713

def event86724 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 86722 .coefficient, .predecessor 1 86723 .coefficient])

def event86725 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event86726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 86725

def event86727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 86711

def event86728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 86727 .coefficient))

def event86729 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event86730 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42618⟩⟩) 0 ⟨10325⟩ 86729

def event86731 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42618⟩⟩) (.authority (.programFamilyFact))

def exact86732RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42618⟩⟩], []⟩, (1)⟩]

theorem exact86732RawTermsValid :
    exact86732RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86732 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42618⟩⟩) exact86732RawTerms (.finite 52) 86731 .exactZero (none)

def event86733 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14571⟩⟩) 0 ⟨10325⟩ 86729

def event86734 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14571⟩⟩) (.authority (.programFamilyFact))

def exact86735RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩], []⟩, (1)⟩]

theorem exact86735RawTermsValid :
    exact86735RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86735 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14571⟩⟩) exact86735RawTerms (.finite 52) 86734 .exactZero (none)

def event86736 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42619⟩⟩) 0 ⟨14571⟩ 86735

def event86737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42619⟩⟩) 1 ⟨42618⟩ 86732

def event86738 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42619⟩⟩) (.product (.predecessor 0 86736 .coefficient) (.predecessor 1 86737 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event86739 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42619⟩⟩, .operator (⟨86735, 0⟩, ⟨86732, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], []⟩, (1)⟩)

def exact86740RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], []⟩, (1)⟩]

theorem exact86740RawTermsValid :
    exact86740RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86740 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42619⟩⟩) exact86740RawTerms (.finite 2704) 86738 .exactZero (none)

def event86741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42620⟩⟩) 0 ⟨42619⟩ 86740

def event86742 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42620⟩⟩) (.identity (.predecessor 0 86741 .coefficient))

def event86743 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42620⟩⟩) (.finite 2704)

def event86744 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42836⟩⟩) 0 ⟨42620⟩ 86743

def event86745 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42836⟩⟩) (.authority (.programFamilyFact))

def exact86746RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], []⟩, (1)⟩]

theorem exact86746RawTermsValid :
    exact86746RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86746 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42836⟩⟩) exact86746RawTerms (.finite 52) 86745 .exactZero (none)

def event86747 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42837⟩⟩) 0 ⟨42836⟩ 86746

def event86748 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42837⟩⟩) (.identity (.predecessor 0 86747 .coefficient))

def event86749 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42837⟩⟩) (.finite 52)

def event86750 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43993⟩⟩) 0 ⟨42837⟩ 86749

def event86751 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43993⟩⟩) (.authority (.programFamilyFact))

def event86752 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨43993⟩⟩) (.finite 3720)

def event86753 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event86754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43994⟩⟩) 0 ⟨7177⟩ 86753

def event86755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43994⟩⟩) 1 ⟨43993⟩ 86752

def event86756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43994⟩⟩) (.authority (.operator))

def exact86757RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43994⟩⟩]⟩, (1)⟩]

theorem exact86757RawTermsValid :
    exact86757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86757 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43994⟩⟩) exact86757RawTerms .large 86756 .exactZero (none)

def event86758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44813⟩⟩) 0 ⟨43994⟩ 86757

def event86759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44813⟩⟩) (.authority (.operator))

def exact86760RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44813⟩⟩]⟩, (1)⟩]

theorem exact86760RawTermsValid :
    exact86760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86760 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44813⟩⟩) exact86760RawTerms (.finite 8192) 86759 .exactZero (none)

def event86761 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event86762 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event86763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44170⟩⟩) 0 ⟨42837⟩ 86749

def event86764 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44170⟩⟩) 1 ⟨136⟩ 86762

def event86765 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44170⟩⟩) (.sum [.predecessor 0 86763 .coefficient, .predecessor 1 86764 .coefficient])

def event86766 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44170⟩⟩) (.finite 52)

def event86767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44171⟩⟩) 0 ⟨44170⟩ 86766

def event86768 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44171⟩⟩) (.identity (.predecessor 0 86767 .coefficient))

def exact86769RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], []⟩, (1)⟩]

theorem exact86769RawTermsValid :
    exact86769RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86769 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44171⟩⟩) exact86769RawTerms (.finite 52) 86768 .exactZero (none)

def event86770 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact86771RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact86771RawTermsValid :
    exact86771RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86771 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact86771RawTerms .large 86770 .exactZero (none)

def event86772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44172⟩⟩) 0 ⟨6908⟩ 86771

def event86773 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44172⟩⟩) 1 ⟨44171⟩ 86769

def event86774 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44172⟩⟩) (.product (.predecessor 0 86772 .coefficient) (.predecessor 1 86773 .coefficient) (⟨false, false, none, none, none⟩))

def event86775 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44172⟩⟩, .operator (⟨86771, 0⟩, ⟨86769, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact86776RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact86776RawTermsValid :
    exact86776RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86776 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44172⟩⟩) exact86776RawTerms .large 86774 .exactZero (none)

def event86777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7194⟩⟩) 0 ⟨7177⟩ 86753

def event86778 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7194⟩⟩) (.authority (.operator))

def exact86779RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩]

theorem exact86779RawTermsValid :
    exact86779RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86779 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7194⟩⟩) exact86779RawTerms .large 86778 .exactZero (none)

def event86780 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44173⟩⟩) 0 ⟨7194⟩ 86779

def event86781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44173⟩⟩) 1 ⟨44172⟩ 86776

def event86782 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44173⟩⟩) (.sum [.predecessor 0 86780 .coefficient, .predecessor 1 86781 .coefficient])

def exact86783RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact86783RawTermsValid :
    exact86783RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86783 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44173⟩⟩) exact86783RawTerms .large 86782 .exactZero (none)

def eventLeaf5408 : Array AnnotatedEvent := #[
  { event := event86528
    frameStart := 86497 },
  { event := event86529
    frameStart := 86497 },
  { event := event86530
    frameStart := 86497 },
  { event := event86531
    frameStart := 86497 },
  { event := event86532
    frameStart := 86497 },
  { event := event86533
    frameStart := 86497 },
  { event := event86534
    frameStart := 86497 },
  { event := event86535
    frameStart := 86497 },
  { event := event86536
    frameStart := 86497 },
  { event := event86537
    frameStart := 86497 },
  { event := event86538
    frameStart := 86497 },
  { event := event86539
    frameStart := 86497 },
  { event := event86540
    frameStart := 86497 },
  { event := event86541
    frameStart := 86497 },
  { event := event86542
    frameStart := 86497 },
  { event := event86543
    frameStart := 86497 }
]

def eventLeaf5409 : Array AnnotatedEvent := #[
  { event := event86544
    frameStart := 86497 },
  { event := event86545
    frameStart := 86497 },
  { event := event86546
    frameStart := 86497 },
  { event := event86547
    frameStart := 86497 },
  { event := event86548
    frameStart := 86497 },
  { event := event86549
    frameStart := 86497 },
  { event := event86550
    frameStart := 86497 },
  { event := event86551
    frameStart := 86497 },
  { event := event86552
    frameStart := 86497 },
  { event := event86553
    frameStart := 86497 },
  { event := event86554
    frameStart := 86497 },
  { event := event86555
    frameStart := 86497 },
  { event := event86556
    frameStart := 86497 },
  { event := event86557
    frameStart := 86497 },
  { event := event86558
    frameStart := 86497 },
  { event := event86559
    frameStart := 86497 }
]

def eventLeaf5410 : Array AnnotatedEvent := #[
  { event := event86560
    frameStart := 86497 },
  { event := event86561
    frameStart := 86497 },
  { event := event86562
    frameStart := 86497 },
  { event := event86563
    frameStart := 86497 },
  { event := event86564
    frameStart := 86497 },
  { event := event86565
    frameStart := 86497 },
  { event := event86566
    frameStart := 86497 },
  { event := event86567
    frameStart := 86497 },
  { event := event86568
    frameStart := 86497 },
  { event := event86569
    frameStart := 86497 },
  { event := event86570
    frameStart := 86497 },
  { event := event86571
    frameStart := 86497 },
  { event := event86572
    frameStart := 86497 },
  { event := event86573
    frameStart := 86497 },
  { event := event86574
    frameStart := 86497 },
  { event := event86575
    frameStart := 86497 }
]

def eventLeaf5411 : Array AnnotatedEvent := #[
  { event := event86576
    frameStart := 86497 },
  { event := event86577
    frameStart := 86497 },
  { event := event86578
    frameStart := 86497 },
  { event := event86579
    frameStart := 86497 },
  { event := event86580
    frameStart := 86497 },
  { event := event86581
    frameStart := 86497 },
  { event := event86582
    frameStart := 86497 },
  { event := event86583
    frameStart := 86497 },
  { event := event86584
    frameStart := 86497 },
  { event := event86585
    frameStart := 86497 },
  { event := event86586
    frameStart := 86497 },
  { event := event86587
    frameStart := 86497 },
  { event := event86588
    frameStart := 86497 },
  { event := event86589
    frameStart := 86497 },
  { event := event86590
    frameStart := 86497 },
  { event := event86591
    frameStart := 86497 }
]

def eventLeaf5412 : Array AnnotatedEvent := #[
  { event := event86592
    frameStart := 86497 },
  { event := event86593
    frameStart := 86497 },
  { event := event86594
    frameStart := 86497 },
  { event := event86595
    frameStart := 86497 },
  { event := event86596
    frameStart := 86497 },
  { event := event86597
    frameStart := 86497 },
  { event := event86598
    frameStart := 86497 },
  { event := event86599
    frameStart := 86497 },
  { event := event86600
    frameStart := 86497 },
  { event := event86601
    frameStart := 0 },
  { event := event86602
    frameStart := 0 },
  { event := event86603
    frameStart := 0 },
  { event := event86604
    frameStart := 0 },
  { event := event86605
    frameStart := 0 },
  { event := event86606
    frameStart := 0 },
  { event := event86607
    frameStart := 0 }
]

def eventLeaf5413 : Array AnnotatedEvent := #[
  { event := event86608
    frameStart := 0 },
  { event := event86609
    frameStart := 0 },
  { event := event86610
    frameStart := 0 },
  { event := event86611
    frameStart := 0 },
  { event := event86612
    frameStart := 0 },
  { event := event86613
    frameStart := 0 },
  { event := event86614
    frameStart := 0 },
  { event := event86615
    frameStart := 0 },
  { event := event86616
    frameStart := 0 },
  { event := event86617
    frameStart := 0 },
  { event := event86618
    frameStart := 0 },
  { event := event86619
    frameStart := 0 },
  { event := event86620
    frameStart := 0 },
  { event := event86621
    frameStart := 0 },
  { event := event86622
    frameStart := 0 },
  { event := event86623
    frameStart := 0 }
]

def eventLeaf5414 : Array AnnotatedEvent := #[
  { event := event86624
    frameStart := 0 },
  { event := event86625
    frameStart := 0 },
  { event := event86626
    frameStart := 0 },
  { event := event86627
    frameStart := 0 },
  { event := event86628
    frameStart := 0 },
  { event := event86629
    frameStart := 0 },
  { event := event86630
    frameStart := 0 },
  { event := event86631
    frameStart := 0 },
  { event := event86632
    frameStart := 0 },
  { event := event86633
    frameStart := 0 },
  { event := event86634
    frameStart := 0 },
  { event := event86635
    frameStart := 0 },
  { event := event86636
    frameStart := 0 },
  { event := event86637
    frameStart := 0 },
  { event := event86638
    frameStart := 0 },
  { event := event86639
    frameStart := 0 }
]

def eventLeaf5415 : Array AnnotatedEvent := #[
  { event := event86640
    frameStart := 0 },
  { event := event86641
    frameStart := 0 },
  { event := event86642
    frameStart := 0 },
  { event := event86643
    frameStart := 0 },
  { event := event86644
    frameStart := 0 },
  { event := event86645
    frameStart := 0 },
  { event := event86646
    frameStart := 0 },
  { event := event86647
    frameStart := 0 },
  { event := event86648
    frameStart := 0 },
  { event := event86649
    frameStart := 0 },
  { event := event86650
    frameStart := 0 },
  { event := event86651
    frameStart := 0 },
  { event := event86652
    frameStart := 0 },
  { event := event86653
    frameStart := 0 },
  { event := event86654
    frameStart := 0 },
  { event := event86655
    frameStart := 86655 }
]

def eventLeaf5416 : Array AnnotatedEvent := #[
  { event := event86656
    frameStart := 86655 },
  { event := event86657
    frameStart := 86655 },
  { event := event86658
    frameStart := 86655 },
  { event := event86659
    frameStart := 86655 },
  { event := event86660
    frameStart := 86655 },
  { event := event86661
    frameStart := 86655 },
  { event := event86662
    frameStart := 86655 },
  { event := event86663
    frameStart := 86655 },
  { event := event86664
    frameStart := 86655 },
  { event := event86665
    frameStart := 86655 },
  { event := event86666
    frameStart := 86655 },
  { event := event86667
    frameStart := 86655 },
  { event := event86668
    frameStart := 86655 },
  { event := event86669
    frameStart := 86655 },
  { event := event86670
    frameStart := 86655 },
  { event := event86671
    frameStart := 86655 }
]

def eventLeaf5417 : Array AnnotatedEvent := #[
  { event := event86672
    frameStart := 86655 },
  { event := event86673
    frameStart := 86655 },
  { event := event86674
    frameStart := 86655 },
  { event := event86675
    frameStart := 86655 },
  { event := event86676
    frameStart := 86655 },
  { event := event86677
    frameStart := 86655 },
  { event := event86678
    frameStart := 86655 },
  { event := event86679
    frameStart := 86655 },
  { event := event86680
    frameStart := 86655 },
  { event := event86681
    frameStart := 86655 },
  { event := event86682
    frameStart := 86655 },
  { event := event86683
    frameStart := 86655 },
  { event := event86684
    frameStart := 86655 },
  { event := event86685
    frameStart := 86655 },
  { event := event86686
    frameStart := 86655 },
  { event := event86687
    frameStart := 86655 }
]

def eventLeaf5418 : Array AnnotatedEvent := #[
  { event := event86688
    frameStart := 86655 },
  { event := event86689
    frameStart := 86655 },
  { event := event86690
    frameStart := 86655 },
  { event := event86691
    frameStart := 86655 },
  { event := event86692
    frameStart := 86655 },
  { event := event86693
    frameStart := 86655 },
  { event := event86694
    frameStart := 86655 },
  { event := event86695
    frameStart := 86655 },
  { event := event86696
    frameStart := 86655 },
  { event := event86697
    frameStart := 86655 },
  { event := event86698
    frameStart := 86655 },
  { event := event86699
    frameStart := 86655 },
  { event := event86700
    frameStart := 86655 },
  { event := event86701
    frameStart := 86655 },
  { event := event86702
    frameStart := 86655 },
  { event := event86703
    frameStart := 86655 }
]

def eventLeaf5419 : Array AnnotatedEvent := #[
  { event := event86704
    frameStart := 86655 },
  { event := event86705
    frameStart := 86655 },
  { event := event86706
    frameStart := 86655 },
  { event := event86707
    frameStart := 86655 },
  { event := event86708
    frameStart := 86655 },
  { event := event86709
    frameStart := 86709 },
  { event := event86710
    frameStart := 86709 },
  { event := event86711
    frameStart := 86709 },
  { event := event86712
    frameStart := 86709 },
  { event := event86713
    frameStart := 86709 },
  { event := event86714
    frameStart := 86709 },
  { event := event86715
    frameStart := 86709 },
  { event := event86716
    frameStart := 86709 },
  { event := event86717
    frameStart := 86709 },
  { event := event86718
    frameStart := 86709 },
  { event := event86719
    frameStart := 86709 }
]

def eventLeaf5420 : Array AnnotatedEvent := #[
  { event := event86720
    frameStart := 86709 },
  { event := event86721
    frameStart := 86709 },
  { event := event86722
    frameStart := 86709 },
  { event := event86723
    frameStart := 86709 },
  { event := event86724
    frameStart := 86709 },
  { event := event86725
    frameStart := 86709 },
  { event := event86726
    frameStart := 86709 },
  { event := event86727
    frameStart := 86709 },
  { event := event86728
    frameStart := 86709 },
  { event := event86729
    frameStart := 86709 },
  { event := event86730
    frameStart := 86709 },
  { event := event86731
    frameStart := 86709 },
  { event := event86732
    frameStart := 86709 },
  { event := event86733
    frameStart := 86709 },
  { event := event86734
    frameStart := 86709 },
  { event := event86735
    frameStart := 86709 }
]

def eventLeaf5421 : Array AnnotatedEvent := #[
  { event := event86736
    frameStart := 86709 },
  { event := event86737
    frameStart := 86709 },
  { event := event86738
    frameStart := 86709 },
  { event := event86739
    frameStart := 86709 },
  { event := event86740
    frameStart := 86709 },
  { event := event86741
    frameStart := 86709 },
  { event := event86742
    frameStart := 86709 },
  { event := event86743
    frameStart := 86709 },
  { event := event86744
    frameStart := 86709 },
  { event := event86745
    frameStart := 86709 },
  { event := event86746
    frameStart := 86709 },
  { event := event86747
    frameStart := 86709 },
  { event := event86748
    frameStart := 86709 },
  { event := event86749
    frameStart := 86709 },
  { event := event86750
    frameStart := 86709 },
  { event := event86751
    frameStart := 86709 }
]

def eventLeaf5422 : Array AnnotatedEvent := #[
  { event := event86752
    frameStart := 86709 },
  { event := event86753
    frameStart := 86709 },
  { event := event86754
    frameStart := 86709 },
  { event := event86755
    frameStart := 86709 },
  { event := event86756
    frameStart := 86709 },
  { event := event86757
    frameStart := 86709 },
  { event := event86758
    frameStart := 86709 },
  { event := event86759
    frameStart := 86709 },
  { event := event86760
    frameStart := 86709 },
  { event := event86761
    frameStart := 86709 },
  { event := event86762
    frameStart := 86709 },
  { event := event86763
    frameStart := 86709 },
  { event := event86764
    frameStart := 86709 },
  { event := event86765
    frameStart := 86709 },
  { event := event86766
    frameStart := 86709 },
  { event := event86767
    frameStart := 86709 }
]

def eventLeaf5423 : Array AnnotatedEvent := #[
  { event := event86768
    frameStart := 86709 },
  { event := event86769
    frameStart := 86709 },
  { event := event86770
    frameStart := 86709 },
  { event := event86771
    frameStart := 86709 },
  { event := event86772
    frameStart := 86709 },
  { event := event86773
    frameStart := 86709 },
  { event := event86774
    frameStart := 86709 },
  { event := event86775
    frameStart := 86709 },
  { event := event86776
    frameStart := 86709 },
  { event := event86777
    frameStart := 86709 },
  { event := event86778
    frameStart := 86709 },
  { event := event86779
    frameStart := 86709 },
  { event := event86780
    frameStart := 86709 },
  { event := event86781
    frameStart := 86709 },
  { event := event86782
    frameStart := 86709 },
  { event := event86783
    frameStart := 86709 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events338
