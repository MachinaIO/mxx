import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events338

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event86528 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 86518

def event86529 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 86527 .coefficient, .predecessor 1 86528 .coefficient])

def event86530 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event86531 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 86530

def event86532 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 86516

def event86533 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 86532 .coefficient))

def event86534 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event86535 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11217⟩⟩) 0 ⟨5536⟩ 86534

def event86536 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11217⟩⟩) (.authority (.programFamilyFact))

def exact86537RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11217⟩⟩], []⟩, (1)⟩]

theorem exact86537RawTermsValid :
    exact86537RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86537 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11217⟩⟩) exact86537RawTerms (.finite 10) 86536 .exactZero (none)

def event86538 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13556⟩⟩) 0 ⟨5536⟩ 86534

def event86539 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13556⟩⟩) (.authority (.programFamilyFact))

def exact86540RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13556⟩⟩], []⟩, (1)⟩]

theorem exact86540RawTermsValid :
    exact86540RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86540 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13556⟩⟩) exact86540RawTerms (.finite 10) 86539 .exactZero (none)

def event86541 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13557⟩⟩) 0 ⟨13556⟩ 86540

def event86542 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13557⟩⟩) 1 ⟨11217⟩ 86537

def event86543 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13557⟩⟩) (.product (.predecessor 0 86541 .coefficient) (.predecessor 1 86542 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event86544 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13557⟩⟩, .operator (⟨86540, 0⟩, ⟨86537, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11217⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], []⟩, (1)⟩)

def exact86545RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11217⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], []⟩, (1)⟩]

theorem exact86545RawTermsValid :
    exact86545RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86545 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13557⟩⟩) exact86545RawTerms (.finite 100) 86543 .exactZero (none)

def event86546 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13558⟩⟩) 0 ⟨13557⟩ 86545

def event86547 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13558⟩⟩) (.identity (.predecessor 0 86546 .coefficient))

def event86548 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13558⟩⟩) (.finite 100)

def event86549 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15583⟩⟩) 0 ⟨13558⟩ 86548

def event86550 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15583⟩⟩) (.authority (.programFamilyFact))

def exact86551RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15583⟩⟩], []⟩, (1)⟩]

theorem exact86551RawTermsValid :
    exact86551RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86551 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15583⟩⟩) exact86551RawTerms (.finite 10) 86550 .exactZero (none)

def event86552 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15584⟩⟩) 0 ⟨15583⟩ 86551

def event86553 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15584⟩⟩) (.identity (.predecessor 0 86552 .coefficient))

def event86554 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15584⟩⟩) (.finite 10)

def event86555 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23971⟩⟩) 0 ⟨15584⟩ 86554

def event86556 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23971⟩⟩) (.authority (.programFamilyFact))

def event86557 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23971⟩⟩) (.finite 3720)

def event86558 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event86559 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23973⟩⟩) 0 ⟨6689⟩ 86558

def event86560 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23973⟩⟩) 1 ⟨23971⟩ 86557

def event86561 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23973⟩⟩) (.authority (.operator))

def exact86562RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23973⟩⟩]⟩, (1)⟩]

theorem exact86562RawTermsValid :
    exact86562RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86562 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23973⟩⟩) exact86562RawTerms .large 86561 .exactZero (none)

def event86563 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27215⟩⟩) 0 ⟨23973⟩ 86562

def event86564 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27215⟩⟩) (.authority (.operator))

def exact86565RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27215⟩⟩]⟩, (1)⟩]

theorem exact86565RawTermsValid :
    exact86565RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86565 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27215⟩⟩) exact86565RawTerms (.finite 8192) 86564 .exactZero (none)

def event86566 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event86567 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event86568 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15658⟩⟩) 0 ⟨15584⟩ 86554

def event86569 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15658⟩⟩) 1 ⟨110⟩ 86567

def event86570 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15658⟩⟩) (.sum [.predecessor 0 86568 .coefficient, .predecessor 1 86569 .coefficient])

def event86571 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15658⟩⟩) (.finite 10)

def event86572 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15659⟩⟩) 0 ⟨15658⟩ 86571

def event86573 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15659⟩⟩) (.identity (.predecessor 0 86572 .coefficient))

def exact86574RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15583⟩⟩], []⟩, (1)⟩]

theorem exact86574RawTermsValid :
    exact86574RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86574 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15659⟩⟩) exact86574RawTerms (.finite 10) 86573 .exactZero (none)

def event86575 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact86576RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact86576RawTermsValid :
    exact86576RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86576 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact86576RawTerms .large 86575 .exactZero (none)

def event86577 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15660⟩⟩) 0 ⟨6544⟩ 86576

def event86578 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15660⟩⟩) 1 ⟨15659⟩ 86574

def event86579 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15660⟩⟩) (.product (.predecessor 0 86577 .coefficient) (.predecessor 1 86578 .coefficient) (⟨false, false, none, none, none⟩))

def event86580 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15660⟩⟩, .operator (⟨86576, 0⟩, ⟨86574, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact86581RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact86581RawTermsValid :
    exact86581RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86581 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15660⟩⟩) exact86581RawTerms .large 86579 .exactZero (none)

def event86582 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6694⟩⟩) 0 ⟨6689⟩ 86558

def event86583 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6694⟩⟩) (.authority (.operator))

def exact86584RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩]

theorem exact86584RawTermsValid :
    exact86584RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86584 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6694⟩⟩) exact86584RawTerms .large 86583 .exactZero (none)

def event86585 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15661⟩⟩) 0 ⟨6694⟩ 86584

def event86586 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15661⟩⟩) 1 ⟨15660⟩ 86581

def event86587 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15661⟩⟩) (.sum [.predecessor 0 86585 .coefficient, .predecessor 1 86586 .coefficient])

def exact86588RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86588RawTermsValid :
    exact86588RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86588 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15661⟩⟩) exact86588RawTerms .large 86587 .exactZero (none)

def event86589 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27216⟩⟩) 0 ⟨15661⟩ 86588

def event86590 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27216⟩⟩) 1 ⟨27215⟩ 86565

def event86591 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27216⟩⟩) (.product (.predecessor 0 86589 .coefficient) (.predecessor 1 86590 .coefficient) (⟨false, false, none, none, none⟩))

def event86592 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27216⟩⟩, .operator (⟨86588, 0⟩, ⟨86565, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27215⟩⟩]⟩, (1)⟩)

def event86593 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27216⟩⟩, .operator (⟨86588, 1⟩, ⟨86565, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27215⟩⟩]⟩, (-1)⟩)

def event86594 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27216⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27215⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27215⟩⟩) ⟨23973⟩ 86562)

def event86595 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27216⟩⟩, .relation 86594 0, ⟨[⟨.program ⟨214⟩, ⟨15583⟩⟩], [⟨.program ⟨214⟩, ⟨23973⟩⟩]⟩, (-1)⟩)

def exact86596RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27215⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15583⟩⟩], [⟨.program ⟨214⟩, ⟨23973⟩⟩]⟩, (-1)⟩]

theorem exact86596RawTermsValid :
    exact86596RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86596 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27216⟩⟩) exact86596RawTerms .large 86591 .exactZero (none)

def event86597 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15629⟩⟩) 0 ⟨15584⟩ 86554

def event86598 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15629⟩⟩) (.authority (.programFamilyFact))

def exact86599RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15629⟩⟩], []⟩, (1)⟩]

theorem exact86599RawTermsValid :
    exact86599RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86599 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15629⟩⟩) exact86599RawTerms (.finite 58) 86598 .exactZero (none)

def event86600 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15630⟩⟩) 0 ⟨6544⟩ 86576

def event86601 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15630⟩⟩) 1 ⟨15629⟩ 86599

def event86602 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15630⟩⟩) (.product (.predecessor 0 86600 .coefficient) (.predecessor 1 86601 .coefficient) (⟨false, true, none, none, some 1⟩))

def event86603 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15630⟩⟩, .operator (⟨86576, 0⟩, ⟨86599, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact86604RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact86604RawTermsValid :
    exact86604RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86604 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15630⟩⟩) exact86604RawTerms .large 86602 .exactZero (none)

def event86605 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6717⟩⟩) 0 ⟨6689⟩ 86558

def event86606 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6717⟩⟩) (.authority (.operator))

def exact86607RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩]

theorem exact86607RawTermsValid :
    exact86607RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86607 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6717⟩⟩) exact86607RawTerms .large 86606 .exactZero (none)

def event86608 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15631⟩⟩) 0 ⟨6717⟩ 86607

def event86609 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15631⟩⟩) 1 ⟨15630⟩ 86604

def event86610 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15631⟩⟩) (.sum [.predecessor 0 86608 .coefficient, .predecessor 1 86609 .coefficient])

def exact86611RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86611RawTermsValid :
    exact86611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86611 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15631⟩⟩) exact86611RawTerms .large 86610 .exactZero (none)

def event86612 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27220⟩⟩) 0 ⟨15631⟩ 86611

def event86613 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27220⟩⟩) 1 ⟨27216⟩ 86596

def event86614 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27220⟩⟩) (.sum [.predecessor 0 86612 .coefficient, .predecessor 1 86613 .coefficient])

def exact86615RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27215⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15583⟩⟩], [⟨.program ⟨214⟩, ⟨23973⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86615RawTermsValid :
    exact86615RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86615 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27220⟩⟩) exact86615RawTerms .large 86614 .exactZero (none)

def event86616 : Event := .preFoldPolynomial 86615 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27215⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15583⟩⟩], [⟨.program ⟨214⟩, ⟨23973⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact86617RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27215⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15583⟩⟩], [⟨.program ⟨214⟩, ⟨23973⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event86617 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27220⟩⟩) 86616 exact86617RawTerms .large 86614 .exactZero (none)

def event86618 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15584⟩⟩) ⟨⟨130⟩, ⟨37⟩, ⟨109⟩⟩ ⟨86460, 86618⟩

def event86619 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20971⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20968⟩⟩]⟩) (1) 0 2 (.universal 86618 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20968⟩⟩]⟩) (none) 86617)

def event86620 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20971⟩⟩, .relation 86619 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩)

def event86621 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20971⟩⟩, .relation 86619 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27215⟩⟩]⟩, (-1)⟩)

def event86622 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20971⟩⟩, .relation 86619 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15583⟩⟩], [⟨.program ⟨214⟩, ⟨23973⟩⟩]⟩, (1)⟩)

def event86623 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20971⟩⟩, .relation 86619 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact86624RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27215⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15583⟩⟩], [⟨.program ⟨214⟩, ⟨23973⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86624RawTermsValid :
    exact86624RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86624 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20971⟩⟩) exact86624RawTerms .large 86456 (.finite 1811303510016) (some (86458))

def event86625 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27218⟩⟩) 0 ⟨20971⟩ 86624

def event86626 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27218⟩⟩) 1 ⟨27217⟩ 86446

def event86627 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27218⟩⟩) (.sum [.predecessor 0 86625 .coefficient, .predecessor 1 86626 .coefficient])

def event86628 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27218⟩⟩, .operator (⟨86624, 0⟩, ⟨86446, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27215⟩⟩]⟩, (1)⟩)

def event86629 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27218⟩⟩, .operator (⟨86624, 2⟩, ⟨86446, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15583⟩⟩], [⟨.program ⟨214⟩, ⟨23973⟩⟩]⟩, (-1)⟩)

def event86630 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27218⟩⟩) (.sum [.result 86624 .summary, .result 86446 .summary])

def exact86631RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86631RawTermsValid :
    exact86631RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86631 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27218⟩⟩) exact86631RawTerms .large 86627 (.finite 1291978824159503986688) (some (86630))

def event86632 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23908⟩⟩) 0 ⟨15423⟩ 4167

def event86633 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23908⟩⟩) (.authority (.programFamilyFact))

def event86634 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23908⟩⟩) (.finite 3720)

def event86635 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23910⟩⟩) 0 ⟨6689⟩ 5477

def event86636 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23910⟩⟩) 1 ⟨23908⟩ 86634

def event86637 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23910⟩⟩) (.authority (.operator))

def exact86638RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23910⟩⟩]⟩, (1)⟩]

theorem exact86638RawTermsValid :
    exact86638RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86638 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23910⟩⟩) exact86638RawTerms .large 86637 .exactZero (none)

def event86639 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26998⟩⟩) 0 ⟨23910⟩ 86638

def event86640 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26998⟩⟩) (.authority (.operator))

def exact86641RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26998⟩⟩]⟩, (1)⟩]

theorem exact86641RawTermsValid :
    exact86641RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86641 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26998⟩⟩) exact86641RawTerms (.finite 8192) 86640 .exactZero (none)

def event86642 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23163⟩⟩) 0 ⟨12165⟩ 4161

def event86643 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23163⟩⟩) (.authority (.programFamilyFact))

def event86644 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23163⟩⟩) (.finite 3720)

def event86645 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23164⟩⟩) 0 ⟨6689⟩ 5477

def event86646 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23164⟩⟩) 1 ⟨23163⟩ 86644

def event86647 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23164⟩⟩) (.authority (.operator))

def exact86648RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23164⟩⟩]⟩, (1)⟩]

theorem exact86648RawTermsValid :
    exact86648RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86648 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23164⟩⟩) exact86648RawTerms .large 86647 .exactZero (none)

def event86649 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25296⟩⟩) 0 ⟨23164⟩ 86648

def event86650 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25296⟩⟩) (.authority (.operator))

def exact86651RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩, (1)⟩]

theorem exact86651RawTermsValid :
    exact86651RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86651 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25296⟩⟩) exact86651RawTerms (.finite 8192) 86650 .exactZero (none)

def event86652 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11134⟩⟩) 0 ⟨11133⟩ 4150

def event86653 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11134⟩⟩) 1 ⟨6567⟩ 79920

def event86654 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11134⟩⟩) (.tensor (.predecessor 0 86652 .coefficient) (.predecessor 1 86653 .coefficient) true false)

def event86655 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11134⟩⟩, .operator (⟨4150, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact86656RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact86656RawTermsValid :
    exact86656RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86656 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11134⟩⟩) exact86656RawTerms .large 86654 .exactZero (none)

def event86657 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7231⟩⟩) 0 ⟨5539⟩ 79790

def event86658 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7231⟩⟩) 1 ⟨6775⟩ 13486

def event86659 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7231⟩⟩) (.product (.predecessor 0 86657 .coefficient) (.predecessor 1 86658 .coefficient) (⟨false, false, none, none, none⟩))

def event86660 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7231⟩⟩, .operator (⟨79790, 0⟩, ⟨13486, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩)

def exact86661RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩]

theorem exact86661RawTermsValid :
    exact86661RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86661 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7231⟩⟩) exact86661RawTerms .large 86659 .exactZero (none)

def event86662 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11135⟩⟩) 0 ⟨7231⟩ 86661

def event86663 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11135⟩⟩) 1 ⟨11134⟩ 86656

def event86664 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11135⟩⟩) (.sum [.predecessor 0 86662 .coefficient, .predecessor 1 86663 .coefficient])

def exact86665RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86665RawTermsValid :
    exact86665RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86665 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11135⟩⟩) exact86665RawTerms .large 86664 .exactZero (none)

def event86666 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11136⟩⟩) 0 ⟨11135⟩ 86665

def event86667 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11136⟩⟩) 1 ⟨89⟩ 13478

def event86668 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11136⟩⟩) (.sum [.predecessor 0 86666 .coefficient, .predecessor 1 86667 .coefficient])

def event86669 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11136⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨89⟩⟩]⟩) [⟨.result 13478 .coefficient, false, none⟩])

def event86670 : Event := .survivorFold (1) 86669

def exact86671RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86671RawTermsValid :
    exact86671RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86671 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11136⟩⟩) exact86671RawTerms .large 86668 (.finite 26) (some (86669))

def event86672 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12166⟩⟩) 0 ⟨11136⟩ 86671

def event86673 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12166⟩⟩) 1 ⟨12163⟩ 4153

def event86674 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12166⟩⟩) (.product (.predecessor 0 86672 .coefficient) (.predecessor 1 86673 .coefficient) (⟨false, true, none, none, some 1⟩))

def event86675 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12166⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩) [⟨.result 4153 .coefficient, true, some 1⟩])

def event86676 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12166⟩⟩) (.product (.result 86671 .summary) (.transfer 86675) (⟨false, false, none, none, none⟩))

def event86677 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12166⟩⟩, .operator (⟨86671, 1⟩, ⟨4153, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event86678 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12166⟩⟩, .operator (⟨86671, 0⟩, ⟨4153, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩)

def exact86679RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩]

theorem exact86679RawTermsValid :
    exact86679RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86679 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12166⟩⟩) exact86679RawTerms .large 86674 (.finite 4992) (some (86676))

def event86680 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12167⟩⟩) 0 ⟨12163⟩ 4153

def event86681 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12167⟩⟩) 1 ⟨6567⟩ 79920

def event86682 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12167⟩⟩) (.tensor (.predecessor 0 86680 .coefficient) (.predecessor 1 86681 .coefficient) true false)

def event86683 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12167⟩⟩, .operator (⟨4153, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact86684RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact86684RawTermsValid :
    exact86684RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86684 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12167⟩⟩) exact86684RawTerms .large 86682 .exactZero (none)

def event86685 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7248⟩⟩) 0 ⟨5539⟩ 79790

def event86686 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7248⟩⟩) 1 ⟨6792⟩ 13527

def event86687 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7248⟩⟩) (.product (.predecessor 0 86685 .coefficient) (.predecessor 1 86686 .coefficient) (⟨false, false, none, none, none⟩))

def event86688 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7248⟩⟩, .operator (⟨79790, 0⟩, ⟨13527, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩]⟩, (1)⟩)

def exact86689RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩]⟩, (1)⟩]

theorem exact86689RawTermsValid :
    exact86689RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86689 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7248⟩⟩) exact86689RawTerms .large 86687 .exactZero (none)

def event86690 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12168⟩⟩) 0 ⟨7248⟩ 86689

def event86691 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12168⟩⟩) 1 ⟨12167⟩ 86684

def event86692 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12168⟩⟩) (.sum [.predecessor 0 86690 .coefficient, .predecessor 1 86691 .coefficient])

def exact86693RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86693RawTermsValid :
    exact86693RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86693 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12168⟩⟩) exact86693RawTerms .large 86692 .exactZero (none)

def event86694 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12169⟩⟩) 0 ⟨12168⟩ 86693

def event86695 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12169⟩⟩) 1 ⟨106⟩ 13519

def event86696 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12169⟩⟩) (.sum [.predecessor 0 86694 .coefficient, .predecessor 1 86695 .coefficient])

def event86697 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12169⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨106⟩⟩]⟩) [⟨.result 13519 .coefficient, false, none⟩])

def event86698 : Event := .survivorFold (1) 86697

def exact86699RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86699RawTermsValid :
    exact86699RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86699 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12169⟩⟩) exact86699RawTerms .large 86696 (.finite 26) (some (86697))

def event86700 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12170⟩⟩) 0 ⟨12169⟩ 86699

def event86701 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12170⟩⟩) 1 ⟨7841⟩ 13516

def event86702 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12170⟩⟩) (.product (.predecessor 0 86700 .coefficient) (.predecessor 1 86701 .coefficient) (⟨false, false, none, none, none⟩))

def event86703 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12170⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) [⟨.result 13512 .coefficient, false, none⟩])

def event86704 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12170⟩⟩) (.product (.result 86699 .summary) (.transfer 86703) (⟨false, false, none, none, none⟩))

def event86705 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12170⟩⟩, .operator (⟨86699, 1⟩, ⟨13516, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (-1)⟩)

def event86706 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨12170⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7840⟩⟩) ⟨6775⟩ 13486)

def event86707 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12170⟩⟩, .relation 86706 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (-1)⟩)

def event86708 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12170⟩⟩, .operator (⟨86699, 0⟩, ⟨13516, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩)

def exact86709RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (-1)⟩]

theorem exact86709RawTermsValid :
    exact86709RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86709 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12170⟩⟩) exact86709RawTerms .large 86702 (.finite 95420416) (some (86704))

def event86710 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12171⟩⟩) 0 ⟨12170⟩ 86709

def event86711 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12171⟩⟩) 1 ⟨12166⟩ 86679

def event86712 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12171⟩⟩) (.sum [.predecessor 0 86710 .coefficient, .predecessor 1 86711 .coefficient])

def event86713 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12171⟩⟩, .operator (⟨86709, 1⟩, ⟨86679, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩)

def event86714 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12171⟩⟩) (.sum [.result 86709 .summary, .result 86679 .summary])

def exact86715RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86715RawTermsValid :
    exact86715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86715 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12171⟩⟩) exact86715RawTerms .large 86712 (.finite 95425408) (some (86714))

def event86716 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25297⟩⟩) 0 ⟨12171⟩ 86715

def event86717 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25297⟩⟩) 1 ⟨25296⟩ 86651

def event86718 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25297⟩⟩) (.product (.predecessor 0 86716 .coefficient) (.predecessor 1 86717 .coefficient) (⟨false, false, none, none, none⟩))

def event86719 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25297⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩) [⟨.result 86651 .coefficient, false, none⟩])

def event86720 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25297⟩⟩) (.product (.result 86715 .summary) (.transfer 86719) (⟨false, false, none, none, none⟩))

def event86721 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25297⟩⟩, .operator (⟨86715, 1⟩, ⟨86651, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩, (-1)⟩)

def event86722 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25297⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25296⟩⟩) ⟨23164⟩ 86648)

def event86723 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25297⟩⟩, .relation 86722 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨23164⟩⟩]⟩, (-1)⟩)

def event86724 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25297⟩⟩, .operator (⟨86715, 0⟩, ⟨86651, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩, (1)⟩)

def exact86725RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨23164⟩⟩]⟩, (-1)⟩]

theorem exact86725RawTermsValid :
    exact86725RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86725 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25297⟩⟩) exact86725RawTerms .large 86718 (.finite 350212774166528) (some (86720))

def event86726 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19240⟩⟩) 0 ⟨12165⟩ 4161

def event86727 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19240⟩⟩) (.authority (.relationPreimageSource ⟨10⟩))

def exact86728RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩, (1)⟩]

theorem exact86728RawTermsValid :
    exact86728RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86728 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19240⟩⟩) exact86728RawTerms (.finite 136065468) 86727 .exactZero (none)

def event86729 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19242⟩⟩) 0 ⟨19240⟩ 86728

def event86730 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19242⟩⟩) 1 ⟨2348⟩ 4

def event86731 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19242⟩⟩) (.scale (.predecessor 0 86729 .coefficient) (.value (.predecessor 1 86730 .coefficient)))

def exact86732RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩, (1)⟩]

theorem exact86732RawTermsValid :
    exact86732RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86732 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19242⟩⟩) exact86732RawTerms (.finite 136065468) 86731 .exactZero (none)

def event86733 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19243⟩⟩) 0 ⟨5541⟩ 80012

def event86734 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19243⟩⟩) 1 ⟨19242⟩ 86732

def event86735 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19243⟩⟩) (.product (.predecessor 0 86733 .coefficient) (.predecessor 1 86734 .coefficient) (⟨false, false, none, none, none⟩))

def event86736 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19243⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩) [⟨.result 86728 .coefficient, false, none⟩])

def event86737 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19243⟩⟩) (.product (.result 80012 .summary) (.transfer 86736) (⟨false, false, none, none, none⟩))

def event86738 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19243⟩⟩, .operator (⟨80012, 0⟩, ⟨86732, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩, (1)⟩)

def event86739 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19241⟩⟩)

def event86740 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event86741 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event86742 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event86743 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event86744 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event86745 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event86746 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event86747 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event86748 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 86747

def event86749 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 86745

def event86750 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 86748 .coefficient) (.value (.predecessor 1 86749 .coefficient)))

def event86751 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event86752 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 86751

def event86753 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 86743

def event86754 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 86752 .coefficient, .predecessor 1 86753 .coefficient])

def event86755 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event86756 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 86755

def event86757 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 86741

def event86758 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 86757 .coefficient))

def event86759 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event86760 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11133⟩⟩) 0 ⟨5536⟩ 86759

def event86761 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11133⟩⟩) (.authority (.programFamilyFact))

def exact86762RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩], []⟩, (1)⟩]

theorem exact86762RawTermsValid :
    exact86762RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86762 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11133⟩⟩) exact86762RawTerms (.finite 6) 86761 .exactZero (none)

def event86763 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12163⟩⟩) 0 ⟨5536⟩ 86759

def event86764 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12163⟩⟩) (.authority (.programFamilyFact))

def exact86765RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩, (1)⟩]

theorem exact86765RawTermsValid :
    exact86765RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86765 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12163⟩⟩) exact86765RawTerms (.finite 6) 86764 .exactZero (none)

def event86766 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12164⟩⟩) 0 ⟨12163⟩ 86765

def event86767 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12164⟩⟩) 1 ⟨11133⟩ 86762

def event86768 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12164⟩⟩) (.product (.predecessor 0 86766 .coefficient) (.predecessor 1 86767 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event86769 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12164⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩) [⟨.result 86765 .coefficient, true, some 1⟩, ⟨.result 86762 .coefficient, true, some 1⟩])

def event86770 : Event := .survivorFold (1) 86769

def exact86771RawTerms : List Term := []

theorem exact86771RawTermsValid :
    exact86771RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86771 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12164⟩⟩) exact86771RawTerms (.finite 36) 86768 (.finite 36) (some (86769))

def event86772 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12165⟩⟩) 0 ⟨12164⟩ 86771

def event86773 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12165⟩⟩) (.identity (.predecessor 0 86772 .coefficient))

def event86774 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12165⟩⟩) (.finite 36)

def event86775 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19240⟩⟩) 0 ⟨12165⟩ 86774

def event86776 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19240⟩⟩) (.authority (.relationPreimageSource ⟨10⟩))

def exact86777RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩, (1)⟩]

theorem exact86777RawTermsValid :
    exact86777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86777 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19240⟩⟩) exact86777RawTerms (.finite 136065468) 86776 .exactZero (none)

def event86778 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact86779RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact86779RawTermsValid :
    exact86779RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86779 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact86779RawTerms .large 86778 .exactZero (none)

def event86780 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19241⟩⟩) 0 ⟨6⟩ 86779

def event86781 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19241⟩⟩) 1 ⟨19240⟩ 86777

def event86782 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19241⟩⟩) (.product (.predecessor 0 86780 .coefficient) (.predecessor 1 86781 .coefficient) (⟨false, false, none, none, none⟩))

def event86783 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19241⟩⟩, .operator (⟨86779, 0⟩, ⟨86777, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩, (1)⟩)

def eventLeaf5408 : Array AnnotatedEvent := #[
  { event := event86528
    frameStart := 86514 },
  { event := event86529
    frameStart := 86514 },
  { event := event86530
    frameStart := 86514 },
  { event := event86531
    frameStart := 86514 },
  { event := event86532
    frameStart := 86514 },
  { event := event86533
    frameStart := 86514 },
  { event := event86534
    frameStart := 86514 },
  { event := event86535
    frameStart := 86514 },
  { event := event86536
    frameStart := 86514 },
  { event := event86537
    frameStart := 86514 },
  { event := event86538
    frameStart := 86514 },
  { event := event86539
    frameStart := 86514 },
  { event := event86540
    frameStart := 86514 },
  { event := event86541
    frameStart := 86514 },
  { event := event86542
    frameStart := 86514 },
  { event := event86543
    frameStart := 86514 }
]

def eventLeaf5409 : Array AnnotatedEvent := #[
  { event := event86544
    frameStart := 86514 },
  { event := event86545
    frameStart := 86514 },
  { event := event86546
    frameStart := 86514 },
  { event := event86547
    frameStart := 86514 },
  { event := event86548
    frameStart := 86514 },
  { event := event86549
    frameStart := 86514 },
  { event := event86550
    frameStart := 86514 },
  { event := event86551
    frameStart := 86514 },
  { event := event86552
    frameStart := 86514 },
  { event := event86553
    frameStart := 86514 },
  { event := event86554
    frameStart := 86514 },
  { event := event86555
    frameStart := 86514 },
  { event := event86556
    frameStart := 86514 },
  { event := event86557
    frameStart := 86514 },
  { event := event86558
    frameStart := 86514 },
  { event := event86559
    frameStart := 86514 }
]

def eventLeaf5410 : Array AnnotatedEvent := #[
  { event := event86560
    frameStart := 86514 },
  { event := event86561
    frameStart := 86514 },
  { event := event86562
    frameStart := 86514 },
  { event := event86563
    frameStart := 86514 },
  { event := event86564
    frameStart := 86514 },
  { event := event86565
    frameStart := 86514 },
  { event := event86566
    frameStart := 86514 },
  { event := event86567
    frameStart := 86514 },
  { event := event86568
    frameStart := 86514 },
  { event := event86569
    frameStart := 86514 },
  { event := event86570
    frameStart := 86514 },
  { event := event86571
    frameStart := 86514 },
  { event := event86572
    frameStart := 86514 },
  { event := event86573
    frameStart := 86514 },
  { event := event86574
    frameStart := 86514 },
  { event := event86575
    frameStart := 86514 }
]

def eventLeaf5411 : Array AnnotatedEvent := #[
  { event := event86576
    frameStart := 86514 },
  { event := event86577
    frameStart := 86514 },
  { event := event86578
    frameStart := 86514 },
  { event := event86579
    frameStart := 86514 },
  { event := event86580
    frameStart := 86514 },
  { event := event86581
    frameStart := 86514 },
  { event := event86582
    frameStart := 86514 },
  { event := event86583
    frameStart := 86514 },
  { event := event86584
    frameStart := 86514 },
  { event := event86585
    frameStart := 86514 },
  { event := event86586
    frameStart := 86514 },
  { event := event86587
    frameStart := 86514 },
  { event := event86588
    frameStart := 86514 },
  { event := event86589
    frameStart := 86514 },
  { event := event86590
    frameStart := 86514 },
  { event := event86591
    frameStart := 86514 }
]

def eventLeaf5412 : Array AnnotatedEvent := #[
  { event := event86592
    frameStart := 86514 },
  { event := event86593
    frameStart := 86514 },
  { event := event86594
    frameStart := 86514 },
  { event := event86595
    frameStart := 86514 },
  { event := event86596
    frameStart := 86514 },
  { event := event86597
    frameStart := 86514 },
  { event := event86598
    frameStart := 86514 },
  { event := event86599
    frameStart := 86514 },
  { event := event86600
    frameStart := 86514 },
  { event := event86601
    frameStart := 86514 },
  { event := event86602
    frameStart := 86514 },
  { event := event86603
    frameStart := 86514 },
  { event := event86604
    frameStart := 86514 },
  { event := event86605
    frameStart := 86514 },
  { event := event86606
    frameStart := 86514 },
  { event := event86607
    frameStart := 86514 }
]

def eventLeaf5413 : Array AnnotatedEvent := #[
  { event := event86608
    frameStart := 86514 },
  { event := event86609
    frameStart := 86514 },
  { event := event86610
    frameStart := 86514 },
  { event := event86611
    frameStart := 86514 },
  { event := event86612
    frameStart := 86514 },
  { event := event86613
    frameStart := 86514 },
  { event := event86614
    frameStart := 86514 },
  { event := event86615
    frameStart := 86514 },
  { event := event86616
    frameStart := 86514 },
  { event := event86617
    frameStart := 86514 },
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
    frameStart := 0 }
]

def eventLeaf5416 : Array AnnotatedEvent := #[
  { event := event86656
    frameStart := 0 },
  { event := event86657
    frameStart := 0 },
  { event := event86658
    frameStart := 0 },
  { event := event86659
    frameStart := 0 },
  { event := event86660
    frameStart := 0 },
  { event := event86661
    frameStart := 0 },
  { event := event86662
    frameStart := 0 },
  { event := event86663
    frameStart := 0 },
  { event := event86664
    frameStart := 0 },
  { event := event86665
    frameStart := 0 },
  { event := event86666
    frameStart := 0 },
  { event := event86667
    frameStart := 0 },
  { event := event86668
    frameStart := 0 },
  { event := event86669
    frameStart := 0 },
  { event := event86670
    frameStart := 0 },
  { event := event86671
    frameStart := 0 }
]

def eventLeaf5417 : Array AnnotatedEvent := #[
  { event := event86672
    frameStart := 0 },
  { event := event86673
    frameStart := 0 },
  { event := event86674
    frameStart := 0 },
  { event := event86675
    frameStart := 0 },
  { event := event86676
    frameStart := 0 },
  { event := event86677
    frameStart := 0 },
  { event := event86678
    frameStart := 0 },
  { event := event86679
    frameStart := 0 },
  { event := event86680
    frameStart := 0 },
  { event := event86681
    frameStart := 0 },
  { event := event86682
    frameStart := 0 },
  { event := event86683
    frameStart := 0 },
  { event := event86684
    frameStart := 0 },
  { event := event86685
    frameStart := 0 },
  { event := event86686
    frameStart := 0 },
  { event := event86687
    frameStart := 0 }
]

def eventLeaf5418 : Array AnnotatedEvent := #[
  { event := event86688
    frameStart := 0 },
  { event := event86689
    frameStart := 0 },
  { event := event86690
    frameStart := 0 },
  { event := event86691
    frameStart := 0 },
  { event := event86692
    frameStart := 0 },
  { event := event86693
    frameStart := 0 },
  { event := event86694
    frameStart := 0 },
  { event := event86695
    frameStart := 0 },
  { event := event86696
    frameStart := 0 },
  { event := event86697
    frameStart := 0 },
  { event := event86698
    frameStart := 0 },
  { event := event86699
    frameStart := 0 },
  { event := event86700
    frameStart := 0 },
  { event := event86701
    frameStart := 0 },
  { event := event86702
    frameStart := 0 },
  { event := event86703
    frameStart := 0 }
]

def eventLeaf5419 : Array AnnotatedEvent := #[
  { event := event86704
    frameStart := 0 },
  { event := event86705
    frameStart := 0 },
  { event := event86706
    frameStart := 0 },
  { event := event86707
    frameStart := 0 },
  { event := event86708
    frameStart := 0 },
  { event := event86709
    frameStart := 0 },
  { event := event86710
    frameStart := 0 },
  { event := event86711
    frameStart := 0 },
  { event := event86712
    frameStart := 0 },
  { event := event86713
    frameStart := 0 },
  { event := event86714
    frameStart := 0 },
  { event := event86715
    frameStart := 0 },
  { event := event86716
    frameStart := 0 },
  { event := event86717
    frameStart := 0 },
  { event := event86718
    frameStart := 0 },
  { event := event86719
    frameStart := 0 }
]

def eventLeaf5420 : Array AnnotatedEvent := #[
  { event := event86720
    frameStart := 0 },
  { event := event86721
    frameStart := 0 },
  { event := event86722
    frameStart := 0 },
  { event := event86723
    frameStart := 0 },
  { event := event86724
    frameStart := 0 },
  { event := event86725
    frameStart := 0 },
  { event := event86726
    frameStart := 0 },
  { event := event86727
    frameStart := 0 },
  { event := event86728
    frameStart := 0 },
  { event := event86729
    frameStart := 0 },
  { event := event86730
    frameStart := 0 },
  { event := event86731
    frameStart := 0 },
  { event := event86732
    frameStart := 0 },
  { event := event86733
    frameStart := 0 },
  { event := event86734
    frameStart := 0 },
  { event := event86735
    frameStart := 0 }
]

def eventLeaf5421 : Array AnnotatedEvent := #[
  { event := event86736
    frameStart := 0 },
  { event := event86737
    frameStart := 0 },
  { event := event86738
    frameStart := 0 },
  { event := event86739
    frameStart := 86739 },
  { event := event86740
    frameStart := 86739 },
  { event := event86741
    frameStart := 86739 },
  { event := event86742
    frameStart := 86739 },
  { event := event86743
    frameStart := 86739 },
  { event := event86744
    frameStart := 86739 },
  { event := event86745
    frameStart := 86739 },
  { event := event86746
    frameStart := 86739 },
  { event := event86747
    frameStart := 86739 },
  { event := event86748
    frameStart := 86739 },
  { event := event86749
    frameStart := 86739 },
  { event := event86750
    frameStart := 86739 },
  { event := event86751
    frameStart := 86739 }
]

def eventLeaf5422 : Array AnnotatedEvent := #[
  { event := event86752
    frameStart := 86739 },
  { event := event86753
    frameStart := 86739 },
  { event := event86754
    frameStart := 86739 },
  { event := event86755
    frameStart := 86739 },
  { event := event86756
    frameStart := 86739 },
  { event := event86757
    frameStart := 86739 },
  { event := event86758
    frameStart := 86739 },
  { event := event86759
    frameStart := 86739 },
  { event := event86760
    frameStart := 86739 },
  { event := event86761
    frameStart := 86739 },
  { event := event86762
    frameStart := 86739 },
  { event := event86763
    frameStart := 86739 },
  { event := event86764
    frameStart := 86739 },
  { event := event86765
    frameStart := 86739 },
  { event := event86766
    frameStart := 86739 },
  { event := event86767
    frameStart := 86739 }
]

def eventLeaf5423 : Array AnnotatedEvent := #[
  { event := event86768
    frameStart := 86739 },
  { event := event86769
    frameStart := 86739 },
  { event := event86770
    frameStart := 86739 },
  { event := event86771
    frameStart := 86739 },
  { event := event86772
    frameStart := 86739 },
  { event := event86773
    frameStart := 86739 },
  { event := event86774
    frameStart := 86739 },
  { event := event86775
    frameStart := 86739 },
  { event := event86776
    frameStart := 86739 },
  { event := event86777
    frameStart := 86739 },
  { event := event86778
    frameStart := 86739 },
  { event := event86779
    frameStart := 86739 },
  { event := event86780
    frameStart := 86739 },
  { event := event86781
    frameStart := 86739 },
  { event := event86782
    frameStart := 86739 },
  { event := event86783
    frameStart := 86739 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events338
