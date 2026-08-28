import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events342

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event87552 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29953⟩⟩) (.product (.predecessor 0 87550 .coefficient) (.predecessor 1 87551 .coefficient) (⟨false, false, none, none, none⟩))

def event87553 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29953⟩⟩, .operator (⟨87549, 0⟩, ⟨87547, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29952⟩⟩]⟩, (1)⟩)

def exact87554RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29952⟩⟩]⟩, (1)⟩]

theorem exact87554RawTermsValid :
    exact87554RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87554 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29953⟩⟩) exact87554RawTerms .large 87552 .exactZero (none)

def event87555 : Event := .preFoldPolynomial 87554 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29952⟩⟩]⟩, (1)⟩] .exactZero none

def exact87556RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29952⟩⟩]⟩, (1)⟩]

def event87556 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨29953⟩⟩) 87555 exact87556RawTerms .large 87552 .exactZero (none)

def event87557 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨31118⟩⟩)

def event87558 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event87559 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event87560 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event87561 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event87562 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event87563 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event87564 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event87565 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event87566 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 87565

def event87567 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 87563

def event87568 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 87566 .coefficient) (.value (.predecessor 1 87567 .coefficient)))

def event87569 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event87570 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 87569

def event87571 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 87561

def event87572 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 87570 .coefficient, .predecessor 1 87571 .coefficient])

def event87573 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event87574 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 87573

def event87575 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 87559

def event87576 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 87575 .coefficient))

def event87577 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event87578 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28918⟩⟩) 0 ⟨10325⟩ 87577

def event87579 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28918⟩⟩) (.authority (.programFamilyFact))

def exact87580RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨28918⟩⟩], []⟩, (1)⟩]

theorem exact87580RawTermsValid :
    exact87580RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87580 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28918⟩⟩) exact87580RawTerms (.finite 36) 87579 .exactZero (none)

def event87581 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13371⟩⟩) 0 ⟨10325⟩ 87577

def event87582 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13371⟩⟩) (.authority (.programFamilyFact))

def exact87583RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩], []⟩, (1)⟩]

theorem exact87583RawTermsValid :
    exact87583RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87583 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13371⟩⟩) exact87583RawTerms (.finite 36) 87582 .exactZero (none)

def event87584 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28919⟩⟩) 0 ⟨13371⟩ 87583

def event87585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28919⟩⟩) 1 ⟨28918⟩ 87580

def event87586 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28919⟩⟩) (.product (.predecessor 0 87584 .coefficient) (.predecessor 1 87585 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event87587 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28919⟩⟩, .operator (⟨87583, 0⟩, ⟨87580, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], []⟩, (1)⟩)

def exact87588RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], []⟩, (1)⟩]

theorem exact87588RawTermsValid :
    exact87588RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87588 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28919⟩⟩) exact87588RawTerms (.finite 1296) 87586 .exactZero (none)

def event87589 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28920⟩⟩) 0 ⟨28919⟩ 87588

def event87590 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28920⟩⟩) (.identity (.predecessor 0 87589 .coefficient))

def event87591 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨28920⟩⟩) (.finite 1296)

def event87592 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29136⟩⟩) 0 ⟨28920⟩ 87591

def event87593 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29136⟩⟩) (.authority (.programFamilyFact))

def exact87594RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], []⟩, (1)⟩]

theorem exact87594RawTermsValid :
    exact87594RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87594 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29136⟩⟩) exact87594RawTerms (.finite 36) 87593 .exactZero (none)

def event87595 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29137⟩⟩) 0 ⟨29136⟩ 87594

def event87596 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29137⟩⟩) (.identity (.predecessor 0 87595 .coefficient))

def event87597 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨29137⟩⟩) (.finite 36)

def event87598 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30293⟩⟩) 0 ⟨29137⟩ 87597

def event87599 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30293⟩⟩) (.authority (.programFamilyFact))

def event87600 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨30293⟩⟩) (.finite 3720)

def event87601 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event87602 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30294⟩⟩) 0 ⟨7177⟩ 87601

def event87603 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30294⟩⟩) 1 ⟨30293⟩ 87600

def event87604 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30294⟩⟩) (.authority (.operator))

def exact87605RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨30294⟩⟩]⟩, (1)⟩]

theorem exact87605RawTermsValid :
    exact87605RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87605 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30294⟩⟩) exact87605RawTerms .large 87604 .exactZero (none)

def event87606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31113⟩⟩) 0 ⟨30294⟩ 87605

def event87607 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31113⟩⟩) (.authority (.operator))

def exact87608RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩, (1)⟩]

theorem exact87608RawTermsValid :
    exact87608RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87608 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31113⟩⟩) exact87608RawTerms (.finite 8192) 87607 .exactZero (none)

def event87609 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event87610 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event87611 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30470⟩⟩) 0 ⟨29137⟩ 87597

def event87612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30470⟩⟩) 1 ⟨136⟩ 87610

def event87613 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30470⟩⟩) (.sum [.predecessor 0 87611 .coefficient, .predecessor 1 87612 .coefficient])

def event87614 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨30470⟩⟩) (.finite 36)

def event87615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30471⟩⟩) 0 ⟨30470⟩ 87614

def event87616 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30471⟩⟩) (.identity (.predecessor 0 87615 .coefficient))

def exact87617RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], []⟩, (1)⟩]

theorem exact87617RawTermsValid :
    exact87617RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87617 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30471⟩⟩) exact87617RawTerms (.finite 36) 87616 .exactZero (none)

def event87618 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact87619RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact87619RawTermsValid :
    exact87619RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87619 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact87619RawTerms .large 87618 .exactZero (none)

def event87620 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30472⟩⟩) 0 ⟨6908⟩ 87619

def event87621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30472⟩⟩) 1 ⟨30471⟩ 87617

def event87622 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30472⟩⟩) (.product (.predecessor 0 87620 .coefficient) (.predecessor 1 87621 .coefficient) (⟨false, false, none, none, none⟩))

def event87623 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30472⟩⟩, .operator (⟨87619, 0⟩, ⟨87617, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact87624RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact87624RawTermsValid :
    exact87624RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87624 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30472⟩⟩) exact87624RawTerms .large 87622 .exactZero (none)

def event87625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7190⟩⟩) 0 ⟨7177⟩ 87601

def event87626 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7190⟩⟩) (.authority (.operator))

def exact87627RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩]

theorem exact87627RawTermsValid :
    exact87627RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87627 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7190⟩⟩) exact87627RawTerms .large 87626 .exactZero (none)

def event87628 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30473⟩⟩) 0 ⟨7190⟩ 87627

def event87629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30473⟩⟩) 1 ⟨30472⟩ 87624

def event87630 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30473⟩⟩) (.sum [.predecessor 0 87628 .coefficient, .predecessor 1 87629 .coefficient])

def exact87631RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact87631RawTermsValid :
    exact87631RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87631 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30473⟩⟩) exact87631RawTerms .large 87630 .exactZero (none)

def event87632 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31114⟩⟩) 0 ⟨30473⟩ 87631

def event87633 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31114⟩⟩) 1 ⟨31113⟩ 87608

def event87634 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31114⟩⟩) (.product (.predecessor 0 87632 .coefficient) (.predecessor 1 87633 .coefficient) (⟨false, false, none, none, none⟩))

def event87635 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31114⟩⟩, .operator (⟨87631, 0⟩, ⟨87608, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩, (1)⟩)

def event87636 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31114⟩⟩, .operator (⟨87631, 1⟩, ⟨87608, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩, (-1)⟩)

def event87637 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨31114⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨31113⟩⟩) ⟨30294⟩ 87605)

def event87638 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31114⟩⟩, .relation 87637 0, ⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30294⟩⟩]⟩, (-1)⟩)

def exact87639RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30294⟩⟩]⟩, (-1)⟩]

theorem exact87639RawTermsValid :
    exact87639RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87639 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31114⟩⟩) exact87639RawTerms .large 87634 .exactZero (none)

def event87640 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29380⟩⟩) 0 ⟨29137⟩ 87597

def event87641 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29380⟩⟩) (.authority (.programFamilyFact))

def exact87642RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29380⟩⟩], []⟩, (1)⟩]

theorem exact87642RawTermsValid :
    exact87642RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87642 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29380⟩⟩) exact87642RawTerms (.finite 36) 87641 .exactZero (none)

def event87643 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29382⟩⟩) 0 ⟨6908⟩ 87619

def event87644 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29382⟩⟩) 1 ⟨29380⟩ 87642

def event87645 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29382⟩⟩) (.product (.predecessor 0 87643 .coefficient) (.predecessor 1 87644 .coefficient) (⟨false, true, none, none, some 1⟩))

def event87646 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29382⟩⟩, .operator (⟨87619, 0⟩, ⟨87642, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact87647RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact87647RawTermsValid :
    exact87647RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87647 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29382⟩⟩) exact87647RawTerms .large 87645 .exactZero (none)

def event87648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7219⟩⟩) 0 ⟨7177⟩ 87601

def event87649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7219⟩⟩) (.authority (.operator))

def exact87650RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩]

theorem exact87650RawTermsValid :
    exact87650RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87650 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7219⟩⟩) exact87650RawTerms .large 87649 .exactZero (none)

def event87651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29383⟩⟩) 0 ⟨7219⟩ 87650

def event87652 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29383⟩⟩) 1 ⟨29382⟩ 87647

def event87653 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29383⟩⟩) (.sum [.predecessor 0 87651 .coefficient, .predecessor 1 87652 .coefficient])

def exact87654RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact87654RawTermsValid :
    exact87654RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87654 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29383⟩⟩) exact87654RawTerms .large 87653 .exactZero (none)

def event87655 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31118⟩⟩) 0 ⟨29383⟩ 87654

def event87656 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31118⟩⟩) 1 ⟨31114⟩ 87639

def event87657 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31118⟩⟩) (.sum [.predecessor 0 87655 .coefficient, .predecessor 1 87656 .coefficient])

def exact87658RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30294⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact87658RawTermsValid :
    exact87658RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87658 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31118⟩⟩) exact87658RawTerms .large 87657 .exactZero (none)

def event87659 : Event := .preFoldPolynomial 87658 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30294⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact87660RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30294⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event87660 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨31118⟩⟩) 87659 exact87660RawTerms .large 87657 .exactZero (none)

def event87661 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨29137⟩⟩) ⟨⟨98⟩, ⟨80⟩, ⟨135⟩⟩ ⟨87503, 87661⟩

def event87662 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨29955⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29952⟩⟩]⟩) (1) 0 2 (.universal 87661 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29952⟩⟩]⟩) (none) 87660)

def event87663 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29955⟩⟩, .relation 87662 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩)

def event87664 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29955⟩⟩, .relation 87662 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩, (-1)⟩)

def event87665 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29955⟩⟩, .relation 87662 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30294⟩⟩]⟩, (1)⟩)

def event87666 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29955⟩⟩, .relation 87662 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact87667RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30294⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact87667RawTermsValid :
    exact87667RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87667 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29955⟩⟩) exact87667RawTerms .large 87499 (.finite 202072841853861888) (some (87501))

def event87668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31116⟩⟩) 0 ⟨29955⟩ 87667

def event87669 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31116⟩⟩) 1 ⟨31115⟩ 87489

def event87670 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31116⟩⟩) (.sum [.predecessor 0 87668 .coefficient, .predecessor 1 87669 .coefficient])

def event87671 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31116⟩⟩, .operator (⟨87667, 0⟩, ⟨87489, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩, (1)⟩)

def event87672 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31116⟩⟩, .operator (⟨87667, 2⟩, ⟨87489, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨30294⟩⟩]⟩, (-1)⟩)

def event87673 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31116⟩⟩) (.sum [.result 87667 .summary, .result 87489 .summary])

def exact87674RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact87674RawTermsValid :
    exact87674RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87674 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31116⟩⟩) exact87674RawTerms .large 87670 (.finite 32192146870060392302605751287808) (some (87673))

def event87675 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31117⟩⟩) 0 ⟨31116⟩ 87674

def event87676 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31117⟩⟩) 1 ⟨7168⟩ 15662

def event87677 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31117⟩⟩) (.product (.predecessor 0 87675 .coefficient) (.predecessor 1 87676 .coefficient) (⟨false, false, none, none, none⟩))

def event87678 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31117⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩) [⟨.result 15658 .coefficient, false, none⟩])

def event87679 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31117⟩⟩) (.product (.result 87674 .summary) (.transfer 87678) (⟨false, false, none, none, none⟩))

def event87680 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31117⟩⟩, .operator (⟨87674, 0⟩, ⟨15662, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩)

def event87681 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31117⟩⟩, .operator (⟨87674, 1⟩, ⟨15662, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (-1)⟩)

def event87682 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨31117⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7167⟩⟩) ⟨7049⟩ 15655)

def event87683 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31117⟩⟩, .relation 87682 0, ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact87684RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩]

theorem exact87684RawTermsValid :
    exact87684RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87684 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31117⟩⟩) exact87684RawTerms .large 87677 (.finite 345660544987345366211554593406613108817920) (some (87679))

def event87685 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27614⟩⟩) 0 ⟨7177⟩ 15500

def event87686 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27614⟩⟩) 1 ⟨27613⟩ 79271

def event87687 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27614⟩⟩) (.authority (.operator))

def exact87688RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27614⟩⟩]⟩, (1)⟩]

theorem exact87688RawTermsValid :
    exact87688RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87688 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27614⟩⟩) exact87688RawTerms .large 87687 .exactZero (none)

def event87689 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28433⟩⟩) 0 ⟨27614⟩ 87688

def event87690 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28433⟩⟩) (.authority (.operator))

def exact87691RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨28433⟩⟩]⟩, (1)⟩]

theorem exact87691RawTermsValid :
    exact87691RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87691 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28433⟩⟩) exact87691RawTerms (.finite 8192) 87690 .exactZero (none)

def event87692 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28435⟩⟩) 0 ⟨27987⟩ 79555

def event87693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28435⟩⟩) 1 ⟨28433⟩ 87691

def event87694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28435⟩⟩) (.product (.predecessor 0 87692 .coefficient) (.predecessor 1 87693 .coefficient) (⟨false, false, none, none, none⟩))

def event87695 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28435⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨28433⟩⟩]⟩) [⟨.result 87691 .coefficient, false, none⟩])

def event87696 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28435⟩⟩) (.product (.result 79555 .summary) (.transfer 87695) (⟨false, false, none, none, none⟩))

def event87697 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28435⟩⟩, .operator (⟨79555, 0⟩, ⟨87691, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28433⟩⟩]⟩, (1)⟩)

def event87698 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28435⟩⟩, .operator (⟨79555, 1⟩, ⟨87691, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28433⟩⟩]⟩, (-1)⟩)

def event87699 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨28435⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28433⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28433⟩⟩) ⟨27614⟩ 87688)

def event87700 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28435⟩⟩, .relation 87699 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27614⟩⟩]⟩, (-1)⟩)

def exact87701RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28433⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27614⟩⟩]⟩, (-1)⟩]

theorem exact87701RawTermsValid :
    exact87701RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87701 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28435⟩⟩) exact87701RawTerms .large 87694 (.finite 32191557518723128098041228165120) (some (87696))

def event87702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27272⟩⟩) 0 ⟨26457⟩ 3264

def event87703 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27272⟩⟩) (.authority (.relationPreimageSource ⟨78⟩))

def exact87704RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27272⟩⟩]⟩, (1)⟩]

theorem exact87704RawTermsValid :
    exact87704RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87704 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27272⟩⟩) exact87704RawTerms (.finite 5647228698) 87703 .exactZero (none)

def event87705 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27274⟩⟩) 0 ⟨27272⟩ 87704

def event87706 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27274⟩⟩) 1 ⟨2370⟩ 4

def event87707 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27274⟩⟩) (.scale (.predecessor 0 87705 .coefficient) (.value (.predecessor 1 87706 .coefficient)))

def exact87708RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27272⟩⟩]⟩, (1)⟩]

theorem exact87708RawTermsValid :
    exact87708RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87708 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27274⟩⟩) exact87708RawTerms (.finite 5647228698) 87707 .exactZero (none)

def event87709 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27275⟩⟩) 0 ⟨10368⟩ 75995

def event87710 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27275⟩⟩) 1 ⟨27274⟩ 87708

def event87711 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27275⟩⟩) (.product (.predecessor 0 87709 .coefficient) (.predecessor 1 87710 .coefficient) (⟨false, false, none, none, none⟩))

def event87712 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27275⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨27272⟩⟩]⟩) [⟨.result 87704 .coefficient, false, none⟩])

def event87713 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27275⟩⟩) (.product (.result 75995 .summary) (.transfer 87712) (⟨false, false, none, none, none⟩))

def event87714 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27275⟩⟩, .operator (⟨75995, 0⟩, ⟨87708, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27272⟩⟩]⟩, (1)⟩)

def event87715 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨27273⟩⟩)

def event87716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event87717 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event87718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event87719 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event87720 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event87721 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event87722 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event87723 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event87724 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 87723

def event87725 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 87721

def event87726 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 87724 .coefficient) (.value (.predecessor 1 87725 .coefficient)))

def event87727 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event87728 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 87727

def event87729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 87719

def event87730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 87728 .coefficient, .predecessor 1 87729 .coefficient])

def event87731 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event87732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 87731

def event87733 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 87717

def event87734 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 87733 .coefficient))

def event87735 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event87736 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26238⟩⟩) 0 ⟨10325⟩ 87735

def event87737 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26238⟩⟩) (.authority (.programFamilyFact))

def exact87738RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26238⟩⟩], []⟩, (1)⟩]

theorem exact87738RawTermsValid :
    exact87738RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87738 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26238⟩⟩) exact87738RawTerms (.finite 30) 87737 .exactZero (none)

def event87739 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13071⟩⟩) 0 ⟨10325⟩ 87735

def event87740 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13071⟩⟩) (.authority (.programFamilyFact))

def exact87741RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩], []⟩, (1)⟩]

theorem exact87741RawTermsValid :
    exact87741RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87741 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13071⟩⟩) exact87741RawTerms (.finite 30) 87740 .exactZero (none)

def event87742 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26239⟩⟩) 0 ⟨13071⟩ 87741

def event87743 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26239⟩⟩) 1 ⟨26238⟩ 87738

def event87744 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26239⟩⟩) (.product (.predecessor 0 87742 .coefficient) (.predecessor 1 87743 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event87745 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26239⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], []⟩) [⟨.result 87741 .coefficient, true, some 1⟩, ⟨.result 87738 .coefficient, true, some 1⟩])

def event87746 : Event := .survivorFold (1) 87745

def exact87747RawTerms : List Term := []

theorem exact87747RawTermsValid :
    exact87747RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87747 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26239⟩⟩) exact87747RawTerms (.finite 900) 87744 (.finite 900) (some (87745))

def event87748 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26240⟩⟩) 0 ⟨26239⟩ 87747

def event87749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26240⟩⟩) (.identity (.predecessor 0 87748 .coefficient))

def event87750 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26240⟩⟩) (.finite 900)

def event87751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26456⟩⟩) 0 ⟨26240⟩ 87750

def event87752 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26456⟩⟩) (.authority (.programFamilyFact))

def exact87753RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], []⟩, (1)⟩]

theorem exact87753RawTermsValid :
    exact87753RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87753 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26456⟩⟩) exact87753RawTerms (.finite 30) 87752 .exactZero (none)

def event87754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26457⟩⟩) 0 ⟨26456⟩ 87753

def event87755 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26457⟩⟩) (.identity (.predecessor 0 87754 .coefficient))

def event87756 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26457⟩⟩) (.finite 30)

def event87757 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27272⟩⟩) 0 ⟨26457⟩ 87756

def event87758 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27272⟩⟩) (.authority (.relationPreimageSource ⟨78⟩))

def exact87759RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27272⟩⟩]⟩, (1)⟩]

theorem exact87759RawTermsValid :
    exact87759RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87759 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27272⟩⟩) exact87759RawTerms (.finite 5647228698) 87758 .exactZero (none)

def event87760 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact87761RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact87761RawTermsValid :
    exact87761RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87761 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact87761RawTerms .large 87760 .exactZero (none)

def event87762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27273⟩⟩) 0 ⟨35⟩ 87761

def event87763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27273⟩⟩) 1 ⟨27272⟩ 87759

def event87764 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27273⟩⟩) (.product (.predecessor 0 87762 .coefficient) (.predecessor 1 87763 .coefficient) (⟨false, false, none, none, none⟩))

def event87765 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27273⟩⟩, .operator (⟨87761, 0⟩, ⟨87759, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27272⟩⟩]⟩, (1)⟩)

def exact87766RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27272⟩⟩]⟩, (1)⟩]

theorem exact87766RawTermsValid :
    exact87766RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87766 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27273⟩⟩) exact87766RawTerms .large 87764 .exactZero (none)

def event87767 : Event := .preFoldPolynomial 87766 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27272⟩⟩]⟩, (1)⟩] .exactZero none

def exact87768RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27272⟩⟩]⟩, (1)⟩]

def event87768 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨27273⟩⟩) 87767 exact87768RawTerms .large 87764 .exactZero (none)

def event87769 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨28438⟩⟩)

def event87770 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event87771 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event87772 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event87773 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event87774 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event87775 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event87776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event87777 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event87778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 87777

def event87779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 87775

def event87780 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 87778 .coefficient) (.value (.predecessor 1 87779 .coefficient)))

def event87781 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event87782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 87781

def event87783 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 87773

def event87784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 87782 .coefficient, .predecessor 1 87783 .coefficient])

def event87785 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event87786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 87785

def event87787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 87771

def event87788 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 87787 .coefficient))

def event87789 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event87790 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26238⟩⟩) 0 ⟨10325⟩ 87789

def event87791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26238⟩⟩) (.authority (.programFamilyFact))

def exact87792RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26238⟩⟩], []⟩, (1)⟩]

theorem exact87792RawTermsValid :
    exact87792RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87792 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26238⟩⟩) exact87792RawTerms (.finite 30) 87791 .exactZero (none)

def event87793 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13071⟩⟩) 0 ⟨10325⟩ 87789

def event87794 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13071⟩⟩) (.authority (.programFamilyFact))

def exact87795RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩], []⟩, (1)⟩]

theorem exact87795RawTermsValid :
    exact87795RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87795 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13071⟩⟩) exact87795RawTerms (.finite 30) 87794 .exactZero (none)

def event87796 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26239⟩⟩) 0 ⟨13071⟩ 87795

def event87797 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26239⟩⟩) 1 ⟨26238⟩ 87792

def event87798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26239⟩⟩) (.product (.predecessor 0 87796 .coefficient) (.predecessor 1 87797 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event87799 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26239⟩⟩, .operator (⟨87795, 0⟩, ⟨87792, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], []⟩, (1)⟩)

def exact87800RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], []⟩, (1)⟩]

theorem exact87800RawTermsValid :
    exact87800RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87800 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26239⟩⟩) exact87800RawTerms (.finite 900) 87798 .exactZero (none)

def event87801 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26240⟩⟩) 0 ⟨26239⟩ 87800

def event87802 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26240⟩⟩) (.identity (.predecessor 0 87801 .coefficient))

def event87803 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26240⟩⟩) (.finite 900)

def event87804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26456⟩⟩) 0 ⟨26240⟩ 87803

def event87805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26456⟩⟩) (.authority (.programFamilyFact))

def exact87806RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], []⟩, (1)⟩]

theorem exact87806RawTermsValid :
    exact87806RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87806 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26456⟩⟩) exact87806RawTerms (.finite 30) 87805 .exactZero (none)

def event87807 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26457⟩⟩) 0 ⟨26456⟩ 87806

def eventLeaf5472 : Array AnnotatedEvent := #[
  { event := event87552
    frameStart := 87503 },
  { event := event87553
    frameStart := 87503 },
  { event := event87554
    frameStart := 87503 },
  { event := event87555
    frameStart := 87503 },
  { event := event87556
    frameStart := 87503 },
  { event := event87557
    frameStart := 87557 },
  { event := event87558
    frameStart := 87557 },
  { event := event87559
    frameStart := 87557 },
  { event := event87560
    frameStart := 87557 },
  { event := event87561
    frameStart := 87557 },
  { event := event87562
    frameStart := 87557 },
  { event := event87563
    frameStart := 87557 },
  { event := event87564
    frameStart := 87557 },
  { event := event87565
    frameStart := 87557 },
  { event := event87566
    frameStart := 87557 },
  { event := event87567
    frameStart := 87557 }
]

def eventLeaf5473 : Array AnnotatedEvent := #[
  { event := event87568
    frameStart := 87557 },
  { event := event87569
    frameStart := 87557 },
  { event := event87570
    frameStart := 87557 },
  { event := event87571
    frameStart := 87557 },
  { event := event87572
    frameStart := 87557 },
  { event := event87573
    frameStart := 87557 },
  { event := event87574
    frameStart := 87557 },
  { event := event87575
    frameStart := 87557 },
  { event := event87576
    frameStart := 87557 },
  { event := event87577
    frameStart := 87557 },
  { event := event87578
    frameStart := 87557 },
  { event := event87579
    frameStart := 87557 },
  { event := event87580
    frameStart := 87557 },
  { event := event87581
    frameStart := 87557 },
  { event := event87582
    frameStart := 87557 },
  { event := event87583
    frameStart := 87557 }
]

def eventLeaf5474 : Array AnnotatedEvent := #[
  { event := event87584
    frameStart := 87557 },
  { event := event87585
    frameStart := 87557 },
  { event := event87586
    frameStart := 87557 },
  { event := event87587
    frameStart := 87557 },
  { event := event87588
    frameStart := 87557 },
  { event := event87589
    frameStart := 87557 },
  { event := event87590
    frameStart := 87557 },
  { event := event87591
    frameStart := 87557 },
  { event := event87592
    frameStart := 87557 },
  { event := event87593
    frameStart := 87557 },
  { event := event87594
    frameStart := 87557 },
  { event := event87595
    frameStart := 87557 },
  { event := event87596
    frameStart := 87557 },
  { event := event87597
    frameStart := 87557 },
  { event := event87598
    frameStart := 87557 },
  { event := event87599
    frameStart := 87557 }
]

def eventLeaf5475 : Array AnnotatedEvent := #[
  { event := event87600
    frameStart := 87557 },
  { event := event87601
    frameStart := 87557 },
  { event := event87602
    frameStart := 87557 },
  { event := event87603
    frameStart := 87557 },
  { event := event87604
    frameStart := 87557 },
  { event := event87605
    frameStart := 87557 },
  { event := event87606
    frameStart := 87557 },
  { event := event87607
    frameStart := 87557 },
  { event := event87608
    frameStart := 87557 },
  { event := event87609
    frameStart := 87557 },
  { event := event87610
    frameStart := 87557 },
  { event := event87611
    frameStart := 87557 },
  { event := event87612
    frameStart := 87557 },
  { event := event87613
    frameStart := 87557 },
  { event := event87614
    frameStart := 87557 },
  { event := event87615
    frameStart := 87557 }
]

def eventLeaf5476 : Array AnnotatedEvent := #[
  { event := event87616
    frameStart := 87557 },
  { event := event87617
    frameStart := 87557 },
  { event := event87618
    frameStart := 87557 },
  { event := event87619
    frameStart := 87557 },
  { event := event87620
    frameStart := 87557 },
  { event := event87621
    frameStart := 87557 },
  { event := event87622
    frameStart := 87557 },
  { event := event87623
    frameStart := 87557 },
  { event := event87624
    frameStart := 87557 },
  { event := event87625
    frameStart := 87557 },
  { event := event87626
    frameStart := 87557 },
  { event := event87627
    frameStart := 87557 },
  { event := event87628
    frameStart := 87557 },
  { event := event87629
    frameStart := 87557 },
  { event := event87630
    frameStart := 87557 },
  { event := event87631
    frameStart := 87557 }
]

def eventLeaf5477 : Array AnnotatedEvent := #[
  { event := event87632
    frameStart := 87557 },
  { event := event87633
    frameStart := 87557 },
  { event := event87634
    frameStart := 87557 },
  { event := event87635
    frameStart := 87557 },
  { event := event87636
    frameStart := 87557 },
  { event := event87637
    frameStart := 87557 },
  { event := event87638
    frameStart := 87557 },
  { event := event87639
    frameStart := 87557 },
  { event := event87640
    frameStart := 87557 },
  { event := event87641
    frameStart := 87557 },
  { event := event87642
    frameStart := 87557 },
  { event := event87643
    frameStart := 87557 },
  { event := event87644
    frameStart := 87557 },
  { event := event87645
    frameStart := 87557 },
  { event := event87646
    frameStart := 87557 },
  { event := event87647
    frameStart := 87557 }
]

def eventLeaf5478 : Array AnnotatedEvent := #[
  { event := event87648
    frameStart := 87557 },
  { event := event87649
    frameStart := 87557 },
  { event := event87650
    frameStart := 87557 },
  { event := event87651
    frameStart := 87557 },
  { event := event87652
    frameStart := 87557 },
  { event := event87653
    frameStart := 87557 },
  { event := event87654
    frameStart := 87557 },
  { event := event87655
    frameStart := 87557 },
  { event := event87656
    frameStart := 87557 },
  { event := event87657
    frameStart := 87557 },
  { event := event87658
    frameStart := 87557 },
  { event := event87659
    frameStart := 87557 },
  { event := event87660
    frameStart := 87557 },
  { event := event87661
    frameStart := 0 },
  { event := event87662
    frameStart := 0 },
  { event := event87663
    frameStart := 0 }
]

def eventLeaf5479 : Array AnnotatedEvent := #[
  { event := event87664
    frameStart := 0 },
  { event := event87665
    frameStart := 0 },
  { event := event87666
    frameStart := 0 },
  { event := event87667
    frameStart := 0 },
  { event := event87668
    frameStart := 0 },
  { event := event87669
    frameStart := 0 },
  { event := event87670
    frameStart := 0 },
  { event := event87671
    frameStart := 0 },
  { event := event87672
    frameStart := 0 },
  { event := event87673
    frameStart := 0 },
  { event := event87674
    frameStart := 0 },
  { event := event87675
    frameStart := 0 },
  { event := event87676
    frameStart := 0 },
  { event := event87677
    frameStart := 0 },
  { event := event87678
    frameStart := 0 },
  { event := event87679
    frameStart := 0 }
]

def eventLeaf5480 : Array AnnotatedEvent := #[
  { event := event87680
    frameStart := 0 },
  { event := event87681
    frameStart := 0 },
  { event := event87682
    frameStart := 0 },
  { event := event87683
    frameStart := 0 },
  { event := event87684
    frameStart := 0 },
  { event := event87685
    frameStart := 0 },
  { event := event87686
    frameStart := 0 },
  { event := event87687
    frameStart := 0 },
  { event := event87688
    frameStart := 0 },
  { event := event87689
    frameStart := 0 },
  { event := event87690
    frameStart := 0 },
  { event := event87691
    frameStart := 0 },
  { event := event87692
    frameStart := 0 },
  { event := event87693
    frameStart := 0 },
  { event := event87694
    frameStart := 0 },
  { event := event87695
    frameStart := 0 }
]

def eventLeaf5481 : Array AnnotatedEvent := #[
  { event := event87696
    frameStart := 0 },
  { event := event87697
    frameStart := 0 },
  { event := event87698
    frameStart := 0 },
  { event := event87699
    frameStart := 0 },
  { event := event87700
    frameStart := 0 },
  { event := event87701
    frameStart := 0 },
  { event := event87702
    frameStart := 0 },
  { event := event87703
    frameStart := 0 },
  { event := event87704
    frameStart := 0 },
  { event := event87705
    frameStart := 0 },
  { event := event87706
    frameStart := 0 },
  { event := event87707
    frameStart := 0 },
  { event := event87708
    frameStart := 0 },
  { event := event87709
    frameStart := 0 },
  { event := event87710
    frameStart := 0 },
  { event := event87711
    frameStart := 0 }
]

def eventLeaf5482 : Array AnnotatedEvent := #[
  { event := event87712
    frameStart := 0 },
  { event := event87713
    frameStart := 0 },
  { event := event87714
    frameStart := 0 },
  { event := event87715
    frameStart := 87715 },
  { event := event87716
    frameStart := 87715 },
  { event := event87717
    frameStart := 87715 },
  { event := event87718
    frameStart := 87715 },
  { event := event87719
    frameStart := 87715 },
  { event := event87720
    frameStart := 87715 },
  { event := event87721
    frameStart := 87715 },
  { event := event87722
    frameStart := 87715 },
  { event := event87723
    frameStart := 87715 },
  { event := event87724
    frameStart := 87715 },
  { event := event87725
    frameStart := 87715 },
  { event := event87726
    frameStart := 87715 },
  { event := event87727
    frameStart := 87715 }
]

def eventLeaf5483 : Array AnnotatedEvent := #[
  { event := event87728
    frameStart := 87715 },
  { event := event87729
    frameStart := 87715 },
  { event := event87730
    frameStart := 87715 },
  { event := event87731
    frameStart := 87715 },
  { event := event87732
    frameStart := 87715 },
  { event := event87733
    frameStart := 87715 },
  { event := event87734
    frameStart := 87715 },
  { event := event87735
    frameStart := 87715 },
  { event := event87736
    frameStart := 87715 },
  { event := event87737
    frameStart := 87715 },
  { event := event87738
    frameStart := 87715 },
  { event := event87739
    frameStart := 87715 },
  { event := event87740
    frameStart := 87715 },
  { event := event87741
    frameStart := 87715 },
  { event := event87742
    frameStart := 87715 },
  { event := event87743
    frameStart := 87715 }
]

def eventLeaf5484 : Array AnnotatedEvent := #[
  { event := event87744
    frameStart := 87715 },
  { event := event87745
    frameStart := 87715 },
  { event := event87746
    frameStart := 87715 },
  { event := event87747
    frameStart := 87715 },
  { event := event87748
    frameStart := 87715 },
  { event := event87749
    frameStart := 87715 },
  { event := event87750
    frameStart := 87715 },
  { event := event87751
    frameStart := 87715 },
  { event := event87752
    frameStart := 87715 },
  { event := event87753
    frameStart := 87715 },
  { event := event87754
    frameStart := 87715 },
  { event := event87755
    frameStart := 87715 },
  { event := event87756
    frameStart := 87715 },
  { event := event87757
    frameStart := 87715 },
  { event := event87758
    frameStart := 87715 },
  { event := event87759
    frameStart := 87715 }
]

def eventLeaf5485 : Array AnnotatedEvent := #[
  { event := event87760
    frameStart := 87715 },
  { event := event87761
    frameStart := 87715 },
  { event := event87762
    frameStart := 87715 },
  { event := event87763
    frameStart := 87715 },
  { event := event87764
    frameStart := 87715 },
  { event := event87765
    frameStart := 87715 },
  { event := event87766
    frameStart := 87715 },
  { event := event87767
    frameStart := 87715 },
  { event := event87768
    frameStart := 87715 },
  { event := event87769
    frameStart := 87769 },
  { event := event87770
    frameStart := 87769 },
  { event := event87771
    frameStart := 87769 },
  { event := event87772
    frameStart := 87769 },
  { event := event87773
    frameStart := 87769 },
  { event := event87774
    frameStart := 87769 },
  { event := event87775
    frameStart := 87769 }
]

def eventLeaf5486 : Array AnnotatedEvent := #[
  { event := event87776
    frameStart := 87769 },
  { event := event87777
    frameStart := 87769 },
  { event := event87778
    frameStart := 87769 },
  { event := event87779
    frameStart := 87769 },
  { event := event87780
    frameStart := 87769 },
  { event := event87781
    frameStart := 87769 },
  { event := event87782
    frameStart := 87769 },
  { event := event87783
    frameStart := 87769 },
  { event := event87784
    frameStart := 87769 },
  { event := event87785
    frameStart := 87769 },
  { event := event87786
    frameStart := 87769 },
  { event := event87787
    frameStart := 87769 },
  { event := event87788
    frameStart := 87769 },
  { event := event87789
    frameStart := 87769 },
  { event := event87790
    frameStart := 87769 },
  { event := event87791
    frameStart := 87769 }
]

def eventLeaf5487 : Array AnnotatedEvent := #[
  { event := event87792
    frameStart := 87769 },
  { event := event87793
    frameStart := 87769 },
  { event := event87794
    frameStart := 87769 },
  { event := event87795
    frameStart := 87769 },
  { event := event87796
    frameStart := 87769 },
  { event := event87797
    frameStart := 87769 },
  { event := event87798
    frameStart := 87769 },
  { event := event87799
    frameStart := 87769 },
  { event := event87800
    frameStart := 87769 },
  { event := event87801
    frameStart := 87769 },
  { event := event87802
    frameStart := 87769 },
  { event := event87803
    frameStart := 87769 },
  { event := event87804
    frameStart := 87769 },
  { event := event87805
    frameStart := 87769 },
  { event := event87806
    frameStart := 87769 },
  { event := event87807
    frameStart := 87769 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events342
