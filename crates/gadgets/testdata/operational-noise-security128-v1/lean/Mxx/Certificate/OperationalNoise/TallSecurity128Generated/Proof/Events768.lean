import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events768

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event196608 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26143⟩⟩) (.product (.predecessor 0 196606 .coefficient) (.predecessor 1 196607 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event196609 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26143⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13011⟩⟩, ⟨.program ⟨257⟩, ⟨26142⟩⟩], []⟩) [⟨.result 196605 .coefficient, true, some 1⟩, ⟨.result 196602 .coefficient, true, some 1⟩])

def event196610 : Event := .survivorFold (1) 196609

def exact196611RawTerms : List Term := []

theorem exact196611RawTermsValid :
    exact196611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196611 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26143⟩⟩) exact196611RawTerms (.finite 900) 196608 (.finite 900) (some (196609))

def event196612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26144⟩⟩) 0 ⟨26143⟩ 196611

def event196613 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26144⟩⟩) (.identity (.predecessor 0 196612 .coefficient))

def event196614 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26144⟩⟩) (.finite 900)

def event196615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26424⟩⟩) 0 ⟨26144⟩ 196614

def event196616 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26424⟩⟩) (.authority (.programFamilyFact))

def exact196617RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], []⟩, (1)⟩]

theorem exact196617RawTermsValid :
    exact196617RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196617 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26424⟩⟩) exact196617RawTerms (.finite 30) 196616 .exactZero (none)

def event196618 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26425⟩⟩) 0 ⟨26424⟩ 196617

def event196619 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26425⟩⟩) (.identity (.predecessor 0 196618 .coefficient))

def event196620 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26425⟩⟩) (.finite 30)

def event196621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27196⟩⟩) 0 ⟨26425⟩ 196620

def event196622 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27196⟩⟩) (.authority (.relationPreimageSource ⟨79⟩))

def exact196623RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27196⟩⟩]⟩, (1)⟩]

theorem exact196623RawTermsValid :
    exact196623RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196623 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27196⟩⟩) exact196623RawTerms (.finite 5647228698) 196622 .exactZero (none)

def event196624 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact196625RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact196625RawTermsValid :
    exact196625RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196625 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact196625RawTerms .large 196624 .exactZero (none)

def event196626 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27197⟩⟩) 0 ⟨35⟩ 196625

def event196627 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27197⟩⟩) 1 ⟨27196⟩ 196623

def event196628 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27197⟩⟩) (.product (.predecessor 0 196626 .coefficient) (.predecessor 1 196627 .coefficient) (⟨false, false, none, none, none⟩))

def event196629 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27197⟩⟩, .operator (⟨196625, 0⟩, ⟨196623, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27196⟩⟩]⟩, (1)⟩)

def exact196630RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27196⟩⟩]⟩, (1)⟩]

theorem exact196630RawTermsValid :
    exact196630RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196630 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27197⟩⟩) exact196630RawTerms .large 196628 .exactZero (none)

def event196631 : Event := .preFoldPolynomial 196630 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27196⟩⟩]⟩, (1)⟩] .exactZero none

def exact196632RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27196⟩⟩]⟩, (1)⟩]

def event196632 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨27197⟩⟩) 196631 exact196632RawTerms .large 196628 .exactZero (none)

def event196633 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨28343⟩⟩)

def event196634 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event196635 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event196636 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5893⟩⟩) (.authority (.operator))

def event196637 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5893⟩⟩) (.finite 7)

def event196638 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event196639 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event196640 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event196641 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event196642 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 196641

def event196643 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 196639

def event196644 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 196642 .coefficient) (.value (.predecessor 1 196643 .coefficient)))

def event196645 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event196646 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5895⟩⟩) 0 ⟨392⟩ 196645

def event196647 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5895⟩⟩) 1 ⟨5893⟩ 196637

def event196648 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5895⟩⟩) (.sum [.predecessor 0 196646 .coefficient, .predecessor 1 196647 .coefficient])

def event196649 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5895⟩⟩) (.finite 655347)

def event196650 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5905⟩⟩) 0 ⟨5895⟩ 196649

def event196651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5905⟩⟩) 1 ⟨5426⟩ 196635

def event196652 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5905⟩⟩) (.identity (.predecessor 1 196651 .coefficient))

def event196653 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5905⟩⟩) (.finite 655360)

def event196654 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26142⟩⟩) 0 ⟨5905⟩ 196653

def event196655 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26142⟩⟩) (.authority (.programFamilyFact))

def exact196656RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26142⟩⟩], []⟩, (1)⟩]

theorem exact196656RawTermsValid :
    exact196656RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196656 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26142⟩⟩) exact196656RawTerms (.finite 30) 196655 .exactZero (none)

def event196657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13011⟩⟩) 0 ⟨5905⟩ 196653

def event196658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13011⟩⟩) (.authority (.programFamilyFact))

def exact196659RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13011⟩⟩], []⟩, (1)⟩]

theorem exact196659RawTermsValid :
    exact196659RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196659 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13011⟩⟩) exact196659RawTerms (.finite 30) 196658 .exactZero (none)

def event196660 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26143⟩⟩) 0 ⟨13011⟩ 196659

def event196661 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26143⟩⟩) 1 ⟨26142⟩ 196656

def event196662 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26143⟩⟩) (.product (.predecessor 0 196660 .coefficient) (.predecessor 1 196661 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event196663 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26143⟩⟩, .operator (⟨196659, 0⟩, ⟨196656, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13011⟩⟩, ⟨.program ⟨257⟩, ⟨26142⟩⟩], []⟩, (1)⟩)

def exact196664RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13011⟩⟩, ⟨.program ⟨257⟩, ⟨26142⟩⟩], []⟩, (1)⟩]

theorem exact196664RawTermsValid :
    exact196664RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196664 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26143⟩⟩) exact196664RawTerms (.finite 900) 196662 .exactZero (none)

def event196665 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26144⟩⟩) 0 ⟨26143⟩ 196664

def event196666 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26144⟩⟩) (.identity (.predecessor 0 196665 .coefficient))

def event196667 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26144⟩⟩) (.finite 900)

def event196668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26424⟩⟩) 0 ⟨26144⟩ 196667

def event196669 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26424⟩⟩) (.authority (.programFamilyFact))

def exact196670RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], []⟩, (1)⟩]

theorem exact196670RawTermsValid :
    exact196670RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196670 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26424⟩⟩) exact196670RawTerms (.finite 30) 196669 .exactZero (none)

def event196671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26425⟩⟩) 0 ⟨26424⟩ 196670

def event196672 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26425⟩⟩) (.identity (.predecessor 0 196671 .coefficient))

def event196673 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26425⟩⟩) (.finite 30)

def event196674 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27577⟩⟩) 0 ⟨26425⟩ 196673

def event196675 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27577⟩⟩) (.authority (.programFamilyFact))

def event196676 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27577⟩⟩) (.finite 3720)

def event196677 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event196678 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27579⟩⟩) 0 ⟨7177⟩ 196677

def event196679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27579⟩⟩) 1 ⟨27577⟩ 196676

def event196680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27579⟩⟩) (.authority (.operator))

def exact196681RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27579⟩⟩]⟩, (1)⟩]

theorem exact196681RawTermsValid :
    exact196681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196681 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27579⟩⟩) exact196681RawTerms .large 196680 .exactZero (none)

def event196682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28339⟩⟩) 0 ⟨27579⟩ 196681

def event196683 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28339⟩⟩) (.authority (.operator))

def exact196684RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨28339⟩⟩]⟩, (1)⟩]

theorem exact196684RawTermsValid :
    exact196684RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196684 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28339⟩⟩) exact196684RawTerms (.finite 8192) 196683 .exactZero (none)

def event196685 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event196686 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event196687 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27774⟩⟩) 0 ⟨26425⟩ 196673

def event196688 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27774⟩⟩) 1 ⟨136⟩ 196686

def event196689 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27774⟩⟩) (.sum [.predecessor 0 196687 .coefficient, .predecessor 1 196688 .coefficient])

def event196690 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27774⟩⟩) (.finite 30)

def event196691 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27775⟩⟩) 0 ⟨27774⟩ 196690

def event196692 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27775⟩⟩) (.identity (.predecessor 0 196691 .coefficient))

def exact196693RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], []⟩, (1)⟩]

theorem exact196693RawTermsValid :
    exact196693RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196693 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27775⟩⟩) exact196693RawTerms (.finite 30) 196692 .exactZero (none)

def event196694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact196695RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact196695RawTermsValid :
    exact196695RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196695 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact196695RawTerms .large 196694 .exactZero (none)

def event196696 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27776⟩⟩) 0 ⟨6908⟩ 196695

def event196697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27776⟩⟩) 1 ⟨27775⟩ 196693

def event196698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27776⟩⟩) (.product (.predecessor 0 196696 .coefficient) (.predecessor 1 196697 .coefficient) (⟨false, false, none, none, none⟩))

def event196699 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27776⟩⟩, .operator (⟨196695, 0⟩, ⟨196693, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact196700RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact196700RawTermsValid :
    exact196700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196700 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27776⟩⟩) exact196700RawTerms .large 196698 .exactZero (none)

def event196701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7189⟩⟩) 0 ⟨7177⟩ 196677

def event196702 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7189⟩⟩) (.authority (.operator))

def exact196703RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩]

theorem exact196703RawTermsValid :
    exact196703RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196703 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7189⟩⟩) exact196703RawTerms .large 196702 .exactZero (none)

def event196704 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27777⟩⟩) 0 ⟨7189⟩ 196703

def event196705 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27777⟩⟩) 1 ⟨27776⟩ 196700

def event196706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27777⟩⟩) (.sum [.predecessor 0 196704 .coefficient, .predecessor 1 196705 .coefficient])

def exact196707RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact196707RawTermsValid :
    exact196707RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196707 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27777⟩⟩) exact196707RawTerms .large 196706 .exactZero (none)

def event196708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28340⟩⟩) 0 ⟨27777⟩ 196707

def event196709 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28340⟩⟩) 1 ⟨28339⟩ 196684

def event196710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28340⟩⟩) (.product (.predecessor 0 196708 .coefficient) (.predecessor 1 196709 .coefficient) (⟨false, false, none, none, none⟩))

def event196711 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28340⟩⟩, .operator (⟨196707, 0⟩, ⟨196684, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28339⟩⟩]⟩, (1)⟩)

def event196712 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28340⟩⟩, .operator (⟨196707, 1⟩, ⟨196684, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28339⟩⟩]⟩, (-1)⟩)

def event196713 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨28340⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28339⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28339⟩⟩) ⟨27579⟩ 196681)

def event196714 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28340⟩⟩, .relation 196713 0, ⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨27579⟩⟩]⟩, (-1)⟩)

def exact196715RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28339⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨27579⟩⟩]⟩, (-1)⟩]

theorem exact196715RawTermsValid :
    exact196715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196715 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28340⟩⟩) exact196715RawTerms .large 196710 .exactZero (none)

def event196716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26645⟩⟩) 0 ⟨26425⟩ 196673

def event196717 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26645⟩⟩) (.authority (.programFamilyFact))

def exact196718RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26645⟩⟩], []⟩, (1)⟩]

theorem exact196718RawTermsValid :
    exact196718RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196718 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26645⟩⟩) exact196718RawTerms (.finite 62) 196717 .exactZero (none)

def event196719 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26646⟩⟩) 0 ⟨6908⟩ 196695

def event196720 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26646⟩⟩) 1 ⟨26645⟩ 196718

def event196721 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26646⟩⟩) (.product (.predecessor 0 196719 .coefficient) (.predecessor 1 196720 .coefficient) (⟨false, true, none, none, some 1⟩))

def event196722 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26646⟩⟩, .operator (⟨196695, 0⟩, ⟨196718, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact196723RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact196723RawTermsValid :
    exact196723RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196723 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26646⟩⟩) exact196723RawTerms .large 196721 .exactZero (none)

def event196724 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7218⟩⟩) 0 ⟨7177⟩ 196677

def event196725 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7218⟩⟩) (.authority (.operator))

def exact196726RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact196726RawTermsValid :
    exact196726RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196726 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7218⟩⟩) exact196726RawTerms .large 196725 .exactZero (none)

def event196727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26647⟩⟩) 0 ⟨7218⟩ 196726

def event196728 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26647⟩⟩) 1 ⟨26646⟩ 196723

def event196729 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26647⟩⟩) (.sum [.predecessor 0 196727 .coefficient, .predecessor 1 196728 .coefficient])

def exact196730RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact196730RawTermsValid :
    exact196730RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196730 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26647⟩⟩) exact196730RawTerms .large 196729 .exactZero (none)

def event196731 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28343⟩⟩) 0 ⟨26647⟩ 196730

def event196732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28343⟩⟩) 1 ⟨28340⟩ 196715

def event196733 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28343⟩⟩) (.sum [.predecessor 0 196731 .coefficient, .predecessor 1 196732 .coefficient])

def exact196734RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28339⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨27579⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact196734RawTermsValid :
    exact196734RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196734 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28343⟩⟩) exact196734RawTerms .large 196733 .exactZero (none)

def event196735 : Event := .preFoldPolynomial 196734 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28339⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨27579⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact196736RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28339⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨27579⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event196736 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨28343⟩⟩) 196735 exact196736RawTerms .large 196733 .exactZero (none)

def event196737 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨26425⟩⟩) ⟨⟨97⟩, ⟨79⟩, ⟨135⟩⟩ ⟨196579, 196737⟩

def event196738 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨27199⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27196⟩⟩]⟩) (1) 0 2 (.universal 196737 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27196⟩⟩]⟩) (none) 196736)

def event196739 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27199⟩⟩, .relation 196738 1, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩)

def event196740 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27199⟩⟩, .relation 196738 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28339⟩⟩]⟩, (-1)⟩)

def event196741 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27199⟩⟩, .relation 196738 2, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨27579⟩⟩]⟩, (1)⟩)

def event196742 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27199⟩⟩, .relation 196738 3, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact196743RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28339⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨27579⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact196743RawTermsValid :
    exact196743RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196743 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27199⟩⟩) exact196743RawTerms .large 196575 (.finite 202072841853861888) (some (196577))

def event196744 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28342⟩⟩) 0 ⟨27199⟩ 196743

def event196745 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28342⟩⟩) 1 ⟨28341⟩ 196565

def event196746 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28342⟩⟩) (.sum [.predecessor 0 196744 .coefficient, .predecessor 1 196745 .coefficient])

def event196747 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28342⟩⟩, .operator (⟨196743, 0⟩, ⟨196565, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28339⟩⟩]⟩, (1)⟩)

def event196748 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28342⟩⟩, .operator (⟨196743, 2⟩, ⟨196565, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26424⟩⟩], [⟨.program ⟨257⟩, ⟨27579⟩⟩]⟩, (-1)⟩)

def event196749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28342⟩⟩) (.sum [.result 196743 .summary, .result 196565 .summary])

def exact196750RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact196750RawTermsValid :
    exact196750RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196750 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28342⟩⟩) exact196750RawTerms .large 196746 (.finite 32191557518723330170883082027008) (some (196749))

def event196751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68698⟩⟩) 0 ⟨65805⟩ 9271

def event196752 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68698⟩⟩) (.authority (.programFamilyFact))

def event196753 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68698⟩⟩) (.finite 3720)

def event196754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68700⟩⟩) 0 ⟨7177⟩ 15500

def event196755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68700⟩⟩) 1 ⟨68698⟩ 196753

def event196756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68700⟩⟩) (.authority (.operator))

def exact196757RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68700⟩⟩]⟩, (1)⟩]

theorem exact196757RawTermsValid :
    exact196757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196757 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68700⟩⟩) exact196757RawTerms .large 196756 .exactZero (none)

def event196758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70335⟩⟩) 0 ⟨68700⟩ 196757

def event196759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70335⟩⟩) (.authority (.operator))

def exact196760RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨70335⟩⟩]⟩, (1)⟩]

theorem exact196760RawTermsValid :
    exact196760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196760 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70335⟩⟩) exact196760RawTerms (.finite 8192) 196759 .exactZero (none)

def event196761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68541⟩⟩) 0 ⟨65501⟩ 9265

def event196762 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68541⟩⟩) (.authority (.programFamilyFact))

def event196763 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68541⟩⟩) (.finite 3720)

def event196764 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68542⟩⟩) 0 ⟨7177⟩ 15500

def event196765 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68542⟩⟩) 1 ⟨68541⟩ 196763

def event196766 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68542⟩⟩) (.authority (.operator))

def exact196767RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68542⟩⟩]⟩, (1)⟩]

theorem exact196767RawTermsValid :
    exact196767RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196767 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68542⟩⟩) exact196767RawTerms .large 196766 .exactZero (none)

def event196768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69262⟩⟩) 0 ⟨68542⟩ 196767

def event196769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69262⟩⟩) (.authority (.operator))

def exact196770RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨69262⟩⟩]⟩, (1)⟩]

theorem exact196770RawTermsValid :
    exact196770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196770 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69262⟩⟩) exact196770RawTerms (.finite 8192) 196769 .exactZero (none)

def event196771 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25755⟩⟩) 0 ⟨25754⟩ 9254

def event196772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25755⟩⟩) 1 ⟨6998⟩ 192903

def event196773 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25755⟩⟩) (.tensor (.predecessor 0 196771 .coefficient) (.predecessor 1 196772 .coefficient) true false)

def event196774 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨25755⟩⟩, .operator (⟨9254, 0⟩, ⟨192903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25754⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact196775RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25754⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact196775RawTermsValid :
    exact196775RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196775 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25755⟩⟩) exact196775RawTerms .large 196773 .exactZero (none)

def event196776 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8810⟩⟩) 0 ⟨5907⟩ 192773

def event196777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8810⟩⟩) 1 ⟨7276⟩ 21088

def event196778 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8810⟩⟩) (.product (.predecessor 0 196776 .coefficient) (.predecessor 1 196777 .coefficient) (⟨false, false, none, none, none⟩))

def event196779 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8810⟩⟩, .operator (⟨192773, 0⟩, ⟨21088, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩)

def exact196780RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩]

theorem exact196780RawTermsValid :
    exact196780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196780 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8810⟩⟩) exact196780RawTerms .large 196778 .exactZero (none)

def event196781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25756⟩⟩) 0 ⟨8810⟩ 196780

def event196782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25756⟩⟩) 1 ⟨25755⟩ 196775

def event196783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25756⟩⟩) (.sum [.predecessor 0 196781 .coefficient, .predecessor 1 196782 .coefficient])

def exact196784RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25754⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact196784RawTermsValid :
    exact196784RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196784 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25756⟩⟩) exact196784RawTerms .large 196783 .exactZero (none)

def event196785 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25757⟩⟩) 0 ⟨25756⟩ 196784

def event196786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25757⟩⟩) 1 ⟨102⟩ 21080

def event196787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25757⟩⟩) (.sum [.predecessor 0 196785 .coefficient, .predecessor 1 196786 .coefficient])

def event196788 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25757⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨102⟩⟩]⟩) [⟨.result 21080 .coefficient, false, none⟩])

def event196789 : Event := .survivorFold (1) 196788

def exact196790RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25754⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact196790RawTermsValid :
    exact196790RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196790 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25757⟩⟩) exact196790RawTerms .large 196787 (.finite 26) (some (196788))

def event196791 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65502⟩⟩) 0 ⟨25757⟩ 196790

def event196792 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65502⟩⟩) 1 ⟨65499⟩ 9257

def event196793 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65502⟩⟩) (.product (.predecessor 0 196791 .coefficient) (.predecessor 1 196792 .coefficient) (⟨false, true, none, none, some 1⟩))

def event196794 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65502⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨65499⟩⟩], []⟩) [⟨.result 9257 .coefficient, true, some 1⟩])

def event196795 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65502⟩⟩) (.product (.result 196790 .summary) (.transfer 196794) (⟨false, false, none, none, none⟩))

def event196796 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65502⟩⟩, .operator (⟨196790, 1⟩, ⟨9257, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25754⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event196797 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65502⟩⟩, .operator (⟨196790, 0⟩, ⟨9257, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩)

def exact196798RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25754⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩]

theorem exact196798RawTermsValid :
    exact196798RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196798 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65502⟩⟩) exact196798RawTerms .large 196793 (.finite 23855104) (some (196795))

def event196799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65503⟩⟩) 0 ⟨65499⟩ 9257

def event196800 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65503⟩⟩) 1 ⟨6998⟩ 192903

def event196801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65503⟩⟩) (.tensor (.predecessor 0 196799 .coefficient) (.predecessor 1 196800 .coefficient) true false)

def event196802 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65503⟩⟩, .operator (⟨9257, 0⟩, ⟨192903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact196803RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact196803RawTermsValid :
    exact196803RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196803 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65503⟩⟩) exact196803RawTerms .large 196801 .exactZero (none)

def event196804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8828⟩⟩) 0 ⟨5907⟩ 192773

def event196805 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8828⟩⟩) 1 ⟨7294⟩ 21129

def event196806 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8828⟩⟩) (.product (.predecessor 0 196804 .coefficient) (.predecessor 1 196805 .coefficient) (⟨false, false, none, none, none⟩))

def event196807 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8828⟩⟩, .operator (⟨192773, 0⟩, ⟨21129, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩]⟩, (1)⟩)

def exact196808RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩]⟩, (1)⟩]

theorem exact196808RawTermsValid :
    exact196808RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196808 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8828⟩⟩) exact196808RawTerms .large 196806 .exactZero (none)

def event196809 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65504⟩⟩) 0 ⟨8828⟩ 196808

def event196810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65504⟩⟩) 1 ⟨65503⟩ 196803

def event196811 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65504⟩⟩) (.sum [.predecessor 0 196809 .coefficient, .predecessor 1 196810 .coefficient])

def exact196812RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact196812RawTermsValid :
    exact196812RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196812 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65504⟩⟩) exact196812RawTerms .large 196811 .exactZero (none)

def event196813 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65505⟩⟩) 0 ⟨65504⟩ 196812

def event196814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65505⟩⟩) 1 ⟨120⟩ 21121

def event196815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65505⟩⟩) (.sum [.predecessor 0 196813 .coefficient, .predecessor 1 196814 .coefficient])

def event196816 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65505⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨120⟩⟩]⟩) [⟨.result 21121 .coefficient, false, none⟩])

def event196817 : Event := .survivorFold (1) 196816

def exact196818RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact196818RawTermsValid :
    exact196818RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196818 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65505⟩⟩) exact196818RawTerms .large 196815 (.finite 26) (some (196816))

def event196819 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65506⟩⟩) 0 ⟨65505⟩ 196818

def event196820 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65506⟩⟩) 1 ⟨9542⟩ 21118

def event196821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65506⟩⟩) (.product (.predecessor 0 196819 .coefficient) (.predecessor 1 196820 .coefficient) (⟨false, false, none, none, none⟩))

def event196822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65506⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) [⟨.result 21114 .coefficient, false, none⟩])

def event196823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65506⟩⟩) (.product (.result 196818 .summary) (.transfer 196822) (⟨false, false, none, none, none⟩))

def event196824 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65506⟩⟩, .operator (⟨196818, 1⟩, ⟨21118, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (-1)⟩)

def event196825 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨65506⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9541⟩⟩) ⟨7276⟩ 21088)

def event196826 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65506⟩⟩, .relation 196825 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (-1)⟩)

def event196827 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65506⟩⟩, .operator (⟨196818, 0⟩, ⟨21118, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩)

def exact196828RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (-1)⟩]

theorem exact196828RawTermsValid :
    exact196828RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196828 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65506⟩⟩) exact196828RawTerms .large 196821 (.finite 279172874240) (some (196823))

def event196829 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65507⟩⟩) 0 ⟨65506⟩ 196828

def event196830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65507⟩⟩) 1 ⟨65502⟩ 196798

def event196831 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65507⟩⟩) (.sum [.predecessor 0 196829 .coefficient, .predecessor 1 196830 .coefficient])

def event196832 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65507⟩⟩, .operator (⟨196828, 1⟩, ⟨196798, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩)

def event196833 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65507⟩⟩) (.sum [.result 196828 .summary, .result 196798 .summary])

def exact196834RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25754⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact196834RawTermsValid :
    exact196834RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196834 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65507⟩⟩) exact196834RawTerms .large 196831 (.finite 279196729344) (some (196833))

def event196835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69263⟩⟩) 0 ⟨65507⟩ 196834

def event196836 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69263⟩⟩) 1 ⟨69262⟩ 196770

def event196837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69263⟩⟩) (.product (.predecessor 0 196835 .coefficient) (.predecessor 1 196836 .coefficient) (⟨false, false, none, none, none⟩))

def event196838 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69263⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨69262⟩⟩]⟩) [⟨.result 196770 .coefficient, false, none⟩])

def event196839 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69263⟩⟩) (.product (.result 196834 .summary) (.transfer 196838) (⟨false, false, none, none, none⟩))

def event196840 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69263⟩⟩, .operator (⟨196834, 1⟩, ⟨196770, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25754⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69262⟩⟩]⟩, (-1)⟩)

def event196841 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨69263⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25754⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69262⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69262⟩⟩) ⟨68542⟩ 196767)

def event196842 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69263⟩⟩, .relation 196841 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25754⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨68542⟩⟩]⟩, (-1)⟩)

def event196843 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69263⟩⟩, .operator (⟨196834, 0⟩, ⟨196770, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69262⟩⟩]⟩, (1)⟩)

def exact196844RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69262⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25754⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨68542⟩⟩]⟩, (-1)⟩]

theorem exact196844RawTermsValid :
    exact196844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196844 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69263⟩⟩) exact196844RawTerms .large 196837 (.finite 2997852054206608834560) (some (196839))

def event196845 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67790⟩⟩) 0 ⟨65501⟩ 9265

def event196846 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67790⟩⟩) (.authority (.relationPreimageSource ⟨46⟩))

def exact196847RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨67790⟩⟩]⟩, (1)⟩]

theorem exact196847RawTermsValid :
    exact196847RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196847 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67790⟩⟩) exact196847RawTerms (.finite 5647228698) 196846 .exactZero (none)

def event196848 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67792⟩⟩) 0 ⟨67790⟩ 196847

def event196849 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67792⟩⟩) 1 ⟨2370⟩ 4

def event196850 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67792⟩⟩) (.scale (.predecessor 0 196848 .coefficient) (.value (.predecessor 1 196849 .coefficient)))

def exact196851RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨67790⟩⟩]⟩, (1)⟩]

theorem exact196851RawTermsValid :
    exact196851RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event196851 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67792⟩⟩) exact196851RawTerms (.finite 5647228698) 196850 .exactZero (none)

def event196852 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67793⟩⟩) 0 ⟨5909⟩ 192995

def event196853 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67793⟩⟩) 1 ⟨67792⟩ 196851

def event196854 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67793⟩⟩) (.product (.predecessor 0 196852 .coefficient) (.predecessor 1 196853 .coefficient) (⟨false, false, none, none, none⟩))

def event196855 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67793⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨67790⟩⟩]⟩) [⟨.result 196847 .coefficient, false, none⟩])

def event196856 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67793⟩⟩) (.product (.result 192995 .summary) (.transfer 196855) (⟨false, false, none, none, none⟩))

def event196857 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67793⟩⟩, .operator (⟨192995, 0⟩, ⟨196851, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67790⟩⟩]⟩, (1)⟩)

def event196858 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨67791⟩⟩)

def event196859 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event196860 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event196861 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5893⟩⟩) (.authority (.operator))

def event196862 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5893⟩⟩) (.finite 7)

def event196863 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def eventLeaf12288 : Array AnnotatedEvent := #[
  { event := event196608
    frameStart := 196579 },
  { event := event196609
    frameStart := 196579 },
  { event := event196610
    frameStart := 196579 },
  { event := event196611
    frameStart := 196579 },
  { event := event196612
    frameStart := 196579 },
  { event := event196613
    frameStart := 196579 },
  { event := event196614
    frameStart := 196579 },
  { event := event196615
    frameStart := 196579 },
  { event := event196616
    frameStart := 196579 },
  { event := event196617
    frameStart := 196579 },
  { event := event196618
    frameStart := 196579 },
  { event := event196619
    frameStart := 196579 },
  { event := event196620
    frameStart := 196579 },
  { event := event196621
    frameStart := 196579 },
  { event := event196622
    frameStart := 196579 },
  { event := event196623
    frameStart := 196579 }
]

def eventLeaf12289 : Array AnnotatedEvent := #[
  { event := event196624
    frameStart := 196579 },
  { event := event196625
    frameStart := 196579 },
  { event := event196626
    frameStart := 196579 },
  { event := event196627
    frameStart := 196579 },
  { event := event196628
    frameStart := 196579 },
  { event := event196629
    frameStart := 196579 },
  { event := event196630
    frameStart := 196579 },
  { event := event196631
    frameStart := 196579 },
  { event := event196632
    frameStart := 196579 },
  { event := event196633
    frameStart := 196633 },
  { event := event196634
    frameStart := 196633 },
  { event := event196635
    frameStart := 196633 },
  { event := event196636
    frameStart := 196633 },
  { event := event196637
    frameStart := 196633 },
  { event := event196638
    frameStart := 196633 },
  { event := event196639
    frameStart := 196633 }
]

def eventLeaf12290 : Array AnnotatedEvent := #[
  { event := event196640
    frameStart := 196633 },
  { event := event196641
    frameStart := 196633 },
  { event := event196642
    frameStart := 196633 },
  { event := event196643
    frameStart := 196633 },
  { event := event196644
    frameStart := 196633 },
  { event := event196645
    frameStart := 196633 },
  { event := event196646
    frameStart := 196633 },
  { event := event196647
    frameStart := 196633 },
  { event := event196648
    frameStart := 196633 },
  { event := event196649
    frameStart := 196633 },
  { event := event196650
    frameStart := 196633 },
  { event := event196651
    frameStart := 196633 },
  { event := event196652
    frameStart := 196633 },
  { event := event196653
    frameStart := 196633 },
  { event := event196654
    frameStart := 196633 },
  { event := event196655
    frameStart := 196633 }
]

def eventLeaf12291 : Array AnnotatedEvent := #[
  { event := event196656
    frameStart := 196633 },
  { event := event196657
    frameStart := 196633 },
  { event := event196658
    frameStart := 196633 },
  { event := event196659
    frameStart := 196633 },
  { event := event196660
    frameStart := 196633 },
  { event := event196661
    frameStart := 196633 },
  { event := event196662
    frameStart := 196633 },
  { event := event196663
    frameStart := 196633 },
  { event := event196664
    frameStart := 196633 },
  { event := event196665
    frameStart := 196633 },
  { event := event196666
    frameStart := 196633 },
  { event := event196667
    frameStart := 196633 },
  { event := event196668
    frameStart := 196633 },
  { event := event196669
    frameStart := 196633 },
  { event := event196670
    frameStart := 196633 },
  { event := event196671
    frameStart := 196633 }
]

def eventLeaf12292 : Array AnnotatedEvent := #[
  { event := event196672
    frameStart := 196633 },
  { event := event196673
    frameStart := 196633 },
  { event := event196674
    frameStart := 196633 },
  { event := event196675
    frameStart := 196633 },
  { event := event196676
    frameStart := 196633 },
  { event := event196677
    frameStart := 196633 },
  { event := event196678
    frameStart := 196633 },
  { event := event196679
    frameStart := 196633 },
  { event := event196680
    frameStart := 196633 },
  { event := event196681
    frameStart := 196633 },
  { event := event196682
    frameStart := 196633 },
  { event := event196683
    frameStart := 196633 },
  { event := event196684
    frameStart := 196633 },
  { event := event196685
    frameStart := 196633 },
  { event := event196686
    frameStart := 196633 },
  { event := event196687
    frameStart := 196633 }
]

def eventLeaf12293 : Array AnnotatedEvent := #[
  { event := event196688
    frameStart := 196633 },
  { event := event196689
    frameStart := 196633 },
  { event := event196690
    frameStart := 196633 },
  { event := event196691
    frameStart := 196633 },
  { event := event196692
    frameStart := 196633 },
  { event := event196693
    frameStart := 196633 },
  { event := event196694
    frameStart := 196633 },
  { event := event196695
    frameStart := 196633 },
  { event := event196696
    frameStart := 196633 },
  { event := event196697
    frameStart := 196633 },
  { event := event196698
    frameStart := 196633 },
  { event := event196699
    frameStart := 196633 },
  { event := event196700
    frameStart := 196633 },
  { event := event196701
    frameStart := 196633 },
  { event := event196702
    frameStart := 196633 },
  { event := event196703
    frameStart := 196633 }
]

def eventLeaf12294 : Array AnnotatedEvent := #[
  { event := event196704
    frameStart := 196633 },
  { event := event196705
    frameStart := 196633 },
  { event := event196706
    frameStart := 196633 },
  { event := event196707
    frameStart := 196633 },
  { event := event196708
    frameStart := 196633 },
  { event := event196709
    frameStart := 196633 },
  { event := event196710
    frameStart := 196633 },
  { event := event196711
    frameStart := 196633 },
  { event := event196712
    frameStart := 196633 },
  { event := event196713
    frameStart := 196633 },
  { event := event196714
    frameStart := 196633 },
  { event := event196715
    frameStart := 196633 },
  { event := event196716
    frameStart := 196633 },
  { event := event196717
    frameStart := 196633 },
  { event := event196718
    frameStart := 196633 },
  { event := event196719
    frameStart := 196633 }
]

def eventLeaf12295 : Array AnnotatedEvent := #[
  { event := event196720
    frameStart := 196633 },
  { event := event196721
    frameStart := 196633 },
  { event := event196722
    frameStart := 196633 },
  { event := event196723
    frameStart := 196633 },
  { event := event196724
    frameStart := 196633 },
  { event := event196725
    frameStart := 196633 },
  { event := event196726
    frameStart := 196633 },
  { event := event196727
    frameStart := 196633 },
  { event := event196728
    frameStart := 196633 },
  { event := event196729
    frameStart := 196633 },
  { event := event196730
    frameStart := 196633 },
  { event := event196731
    frameStart := 196633 },
  { event := event196732
    frameStart := 196633 },
  { event := event196733
    frameStart := 196633 },
  { event := event196734
    frameStart := 196633 },
  { event := event196735
    frameStart := 196633 }
]

def eventLeaf12296 : Array AnnotatedEvent := #[
  { event := event196736
    frameStart := 196633 },
  { event := event196737
    frameStart := 0 },
  { event := event196738
    frameStart := 0 },
  { event := event196739
    frameStart := 0 },
  { event := event196740
    frameStart := 0 },
  { event := event196741
    frameStart := 0 },
  { event := event196742
    frameStart := 0 },
  { event := event196743
    frameStart := 0 },
  { event := event196744
    frameStart := 0 },
  { event := event196745
    frameStart := 0 },
  { event := event196746
    frameStart := 0 },
  { event := event196747
    frameStart := 0 },
  { event := event196748
    frameStart := 0 },
  { event := event196749
    frameStart := 0 },
  { event := event196750
    frameStart := 0 },
  { event := event196751
    frameStart := 0 }
]

def eventLeaf12297 : Array AnnotatedEvent := #[
  { event := event196752
    frameStart := 0 },
  { event := event196753
    frameStart := 0 },
  { event := event196754
    frameStart := 0 },
  { event := event196755
    frameStart := 0 },
  { event := event196756
    frameStart := 0 },
  { event := event196757
    frameStart := 0 },
  { event := event196758
    frameStart := 0 },
  { event := event196759
    frameStart := 0 },
  { event := event196760
    frameStart := 0 },
  { event := event196761
    frameStart := 0 },
  { event := event196762
    frameStart := 0 },
  { event := event196763
    frameStart := 0 },
  { event := event196764
    frameStart := 0 },
  { event := event196765
    frameStart := 0 },
  { event := event196766
    frameStart := 0 },
  { event := event196767
    frameStart := 0 }
]

def eventLeaf12298 : Array AnnotatedEvent := #[
  { event := event196768
    frameStart := 0 },
  { event := event196769
    frameStart := 0 },
  { event := event196770
    frameStart := 0 },
  { event := event196771
    frameStart := 0 },
  { event := event196772
    frameStart := 0 },
  { event := event196773
    frameStart := 0 },
  { event := event196774
    frameStart := 0 },
  { event := event196775
    frameStart := 0 },
  { event := event196776
    frameStart := 0 },
  { event := event196777
    frameStart := 0 },
  { event := event196778
    frameStart := 0 },
  { event := event196779
    frameStart := 0 },
  { event := event196780
    frameStart := 0 },
  { event := event196781
    frameStart := 0 },
  { event := event196782
    frameStart := 0 },
  { event := event196783
    frameStart := 0 }
]

def eventLeaf12299 : Array AnnotatedEvent := #[
  { event := event196784
    frameStart := 0 },
  { event := event196785
    frameStart := 0 },
  { event := event196786
    frameStart := 0 },
  { event := event196787
    frameStart := 0 },
  { event := event196788
    frameStart := 0 },
  { event := event196789
    frameStart := 0 },
  { event := event196790
    frameStart := 0 },
  { event := event196791
    frameStart := 0 },
  { event := event196792
    frameStart := 0 },
  { event := event196793
    frameStart := 0 },
  { event := event196794
    frameStart := 0 },
  { event := event196795
    frameStart := 0 },
  { event := event196796
    frameStart := 0 },
  { event := event196797
    frameStart := 0 },
  { event := event196798
    frameStart := 0 },
  { event := event196799
    frameStart := 0 }
]

def eventLeaf12300 : Array AnnotatedEvent := #[
  { event := event196800
    frameStart := 0 },
  { event := event196801
    frameStart := 0 },
  { event := event196802
    frameStart := 0 },
  { event := event196803
    frameStart := 0 },
  { event := event196804
    frameStart := 0 },
  { event := event196805
    frameStart := 0 },
  { event := event196806
    frameStart := 0 },
  { event := event196807
    frameStart := 0 },
  { event := event196808
    frameStart := 0 },
  { event := event196809
    frameStart := 0 },
  { event := event196810
    frameStart := 0 },
  { event := event196811
    frameStart := 0 },
  { event := event196812
    frameStart := 0 },
  { event := event196813
    frameStart := 0 },
  { event := event196814
    frameStart := 0 },
  { event := event196815
    frameStart := 0 }
]

def eventLeaf12301 : Array AnnotatedEvent := #[
  { event := event196816
    frameStart := 0 },
  { event := event196817
    frameStart := 0 },
  { event := event196818
    frameStart := 0 },
  { event := event196819
    frameStart := 0 },
  { event := event196820
    frameStart := 0 },
  { event := event196821
    frameStart := 0 },
  { event := event196822
    frameStart := 0 },
  { event := event196823
    frameStart := 0 },
  { event := event196824
    frameStart := 0 },
  { event := event196825
    frameStart := 0 },
  { event := event196826
    frameStart := 0 },
  { event := event196827
    frameStart := 0 },
  { event := event196828
    frameStart := 0 },
  { event := event196829
    frameStart := 0 },
  { event := event196830
    frameStart := 0 },
  { event := event196831
    frameStart := 0 }
]

def eventLeaf12302 : Array AnnotatedEvent := #[
  { event := event196832
    frameStart := 0 },
  { event := event196833
    frameStart := 0 },
  { event := event196834
    frameStart := 0 },
  { event := event196835
    frameStart := 0 },
  { event := event196836
    frameStart := 0 },
  { event := event196837
    frameStart := 0 },
  { event := event196838
    frameStart := 0 },
  { event := event196839
    frameStart := 0 },
  { event := event196840
    frameStart := 0 },
  { event := event196841
    frameStart := 0 },
  { event := event196842
    frameStart := 0 },
  { event := event196843
    frameStart := 0 },
  { event := event196844
    frameStart := 0 },
  { event := event196845
    frameStart := 0 },
  { event := event196846
    frameStart := 0 },
  { event := event196847
    frameStart := 0 }
]

def eventLeaf12303 : Array AnnotatedEvent := #[
  { event := event196848
    frameStart := 0 },
  { event := event196849
    frameStart := 0 },
  { event := event196850
    frameStart := 0 },
  { event := event196851
    frameStart := 0 },
  { event := event196852
    frameStart := 0 },
  { event := event196853
    frameStart := 0 },
  { event := event196854
    frameStart := 0 },
  { event := event196855
    frameStart := 0 },
  { event := event196856
    frameStart := 0 },
  { event := event196857
    frameStart := 0 },
  { event := event196858
    frameStart := 196858 },
  { event := event196859
    frameStart := 196858 },
  { event := event196860
    frameStart := 196858 },
  { event := event196861
    frameStart := 196858 },
  { event := event196862
    frameStart := 196858 },
  { event := event196863
    frameStart := 196858 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events768
