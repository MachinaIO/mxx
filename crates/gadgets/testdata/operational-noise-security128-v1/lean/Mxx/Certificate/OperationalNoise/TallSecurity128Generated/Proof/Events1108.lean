import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1108

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event283648 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event283649 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 283648

def event283650 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 283634

def event283651 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 283650 .coefficient))

def event283652 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event283653 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28630⟩⟩) 0 ⟨5487⟩ 283652

def event283654 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28630⟩⟩) (.authority (.programFamilyFact))

def exact283655RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨28630⟩⟩], []⟩, (1)⟩]

theorem exact283655RawTermsValid :
    exact283655RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283655 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28630⟩⟩) exact283655RawTerms (.finite 36) 283654 .exactZero (none)

def event283656 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13191⟩⟩) 0 ⟨5487⟩ 283652

def event283657 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13191⟩⟩) (.authority (.programFamilyFact))

def exact283658RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13191⟩⟩], []⟩, (1)⟩]

theorem exact283658RawTermsValid :
    exact283658RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283658 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13191⟩⟩) exact283658RawTerms (.finite 36) 283657 .exactZero (none)

def event283659 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28631⟩⟩) 0 ⟨13191⟩ 283658

def event283660 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28631⟩⟩) 1 ⟨28630⟩ 283655

def event283661 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28631⟩⟩) (.product (.predecessor 0 283659 .coefficient) (.predecessor 1 283660 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event283662 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28631⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13191⟩⟩, ⟨.program ⟨257⟩, ⟨28630⟩⟩], []⟩) [⟨.result 283658 .coefficient, true, some 1⟩, ⟨.result 283655 .coefficient, true, some 1⟩])

def event283663 : Event := .survivorFold (1) 283662

def exact283664RawTerms : List Term := []

theorem exact283664RawTermsValid :
    exact283664RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283664 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28631⟩⟩) exact283664RawTerms (.finite 1296) 283661 (.finite 1296) (some (283662))

def event283665 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28632⟩⟩) 0 ⟨28631⟩ 283664

def event283666 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28632⟩⟩) (.identity (.predecessor 0 283665 .coefficient))

def event283667 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨28632⟩⟩) (.finite 1296)

def event283668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29469⟩⟩) 0 ⟨28632⟩ 283667

def event283669 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29469⟩⟩) (.authority (.relationPreimageSource ⟨48⟩))

def exact283670RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29469⟩⟩]⟩, (1)⟩]

theorem exact283670RawTermsValid :
    exact283670RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283670 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29469⟩⟩) exact283670RawTerms (.finite 5647228698) 283669 .exactZero (none)

def event283671 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact283672RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact283672RawTermsValid :
    exact283672RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283672 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact283672RawTerms .large 283671 .exactZero (none)

def event283673 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29470⟩⟩) 0 ⟨35⟩ 283672

def event283674 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29470⟩⟩) 1 ⟨29469⟩ 283670

def event283675 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29470⟩⟩) (.product (.predecessor 0 283673 .coefficient) (.predecessor 1 283674 .coefficient) (⟨false, false, none, none, none⟩))

def event283676 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29470⟩⟩, .operator (⟨283672, 0⟩, ⟨283670, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29469⟩⟩]⟩, (1)⟩)

def exact283677RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29469⟩⟩]⟩, (1)⟩]

theorem exact283677RawTermsValid :
    exact283677RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283677 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29470⟩⟩) exact283677RawTerms .large 283675 .exactZero (none)

def event283678 : Event := .preFoldPolynomial 283677 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29469⟩⟩]⟩, (1)⟩] .exactZero none

def exact283679RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29469⟩⟩]⟩, (1)⟩]

def event283679 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨29470⟩⟩) 283678 exact283679RawTerms .large 283675 .exactZero (none)

def event283680 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨30537⟩⟩)

def event283681 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event283682 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event283683 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event283684 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event283685 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event283686 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event283687 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event283688 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event283689 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 283688

def event283690 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 283686

def event283691 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 283689 .coefficient) (.value (.predecessor 1 283690 .coefficient)))

def event283692 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event283693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 283692

def event283694 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 283684

def event283695 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 283693 .coefficient, .predecessor 1 283694 .coefficient])

def event283696 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event283697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 283696

def event283698 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 283682

def event283699 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 283698 .coefficient))

def event283700 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event283701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28630⟩⟩) 0 ⟨5487⟩ 283700

def event283702 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28630⟩⟩) (.authority (.programFamilyFact))

def exact283703RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨28630⟩⟩], []⟩, (1)⟩]

theorem exact283703RawTermsValid :
    exact283703RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283703 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28630⟩⟩) exact283703RawTerms (.finite 36) 283702 .exactZero (none)

def event283704 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13191⟩⟩) 0 ⟨5487⟩ 283700

def event283705 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13191⟩⟩) (.authority (.programFamilyFact))

def exact283706RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13191⟩⟩], []⟩, (1)⟩]

theorem exact283706RawTermsValid :
    exact283706RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283706 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13191⟩⟩) exact283706RawTerms (.finite 36) 283705 .exactZero (none)

def event283707 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28631⟩⟩) 0 ⟨13191⟩ 283706

def event283708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28631⟩⟩) 1 ⟨28630⟩ 283703

def event283709 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28631⟩⟩) (.product (.predecessor 0 283707 .coefficient) (.predecessor 1 283708 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event283710 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28631⟩⟩, .operator (⟨283706, 0⟩, ⟨283703, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13191⟩⟩, ⟨.program ⟨257⟩, ⟨28630⟩⟩], []⟩, (1)⟩)

def exact283711RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13191⟩⟩, ⟨.program ⟨257⟩, ⟨28630⟩⟩], []⟩, (1)⟩]

theorem exact283711RawTermsValid :
    exact283711RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283711 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28631⟩⟩) exact283711RawTerms (.finite 1296) 283709 .exactZero (none)

def event283712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28632⟩⟩) 0 ⟨28631⟩ 283711

def event283713 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28632⟩⟩) (.identity (.predecessor 0 283712 .coefficient))

def event283714 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨28632⟩⟩) (.finite 1296)

def event283715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30052⟩⟩) 0 ⟨28632⟩ 283714

def event283716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30052⟩⟩) (.authority (.programFamilyFact))

def event283717 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨30052⟩⟩) (.finite 3720)

def event283718 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event283719 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30053⟩⟩) 0 ⟨7177⟩ 283718

def event283720 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30053⟩⟩) 1 ⟨30052⟩ 283717

def event283721 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30053⟩⟩) (.authority (.operator))

def exact283722RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨30053⟩⟩]⟩, (1)⟩]

theorem exact283722RawTermsValid :
    exact283722RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283722 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30053⟩⟩) exact283722RawTerms .large 283721 .exactZero (none)

def event283723 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30533⟩⟩) 0 ⟨30053⟩ 283722

def event283724 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30533⟩⟩) (.authority (.operator))

def exact283725RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨30533⟩⟩]⟩, (1)⟩]

theorem exact283725RawTermsValid :
    exact283725RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283725 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30533⟩⟩) exact283725RawTerms (.finite 8192) 283724 .exactZero (none)

def event283726 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event283727 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event283728 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30342⟩⟩) 0 ⟨28632⟩ 283714

def event283729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30342⟩⟩) 1 ⟨136⟩ 283727

def event283730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30342⟩⟩) (.sum [.predecessor 0 283728 .coefficient, .predecessor 1 283729 .coefficient])

def event283731 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨30342⟩⟩) (.finite 1296)

def event283732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30343⟩⟩) 0 ⟨30342⟩ 283731

def event283733 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30343⟩⟩) (.identity (.predecessor 0 283732 .coefficient))

def exact283734RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13191⟩⟩, ⟨.program ⟨257⟩, ⟨28630⟩⟩], []⟩, (1)⟩]

theorem exact283734RawTermsValid :
    exact283734RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283734 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30343⟩⟩) exact283734RawTerms (.finite 1296) 283733 .exactZero (none)

def event283735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact283736RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact283736RawTermsValid :
    exact283736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283736 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact283736RawTerms .large 283735 .exactZero (none)

def event283737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30344⟩⟩) 0 ⟨6908⟩ 283736

def event283738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30344⟩⟩) 1 ⟨30343⟩ 283734

def event283739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30344⟩⟩) (.product (.predecessor 0 283737 .coefficient) (.predecessor 1 283738 .coefficient) (⟨false, false, none, none, none⟩))

def event283740 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30344⟩⟩, .operator (⟨283736, 0⟩, ⟨283734, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13191⟩⟩, ⟨.program ⟨257⟩, ⟨28630⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact283741RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13191⟩⟩, ⟨.program ⟨257⟩, ⟨28630⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact283741RawTermsValid :
    exact283741RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283741 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30344⟩⟩) exact283741RawTerms .large 283739 .exactZero (none)

def event283742 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 283718

def event283743 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact283744RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact283744RawTermsValid :
    exact283744RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283744 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact283744RawTerms .large 283743 .exactZero (none)

def event283745 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7279⟩⟩) 0 ⟨7178⟩ 283744

def event283746 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7279⟩⟩) (.identity (.predecessor 0 283745 .coefficient))

def exact283747RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩, (1)⟩]

theorem exact283747RawTermsValid :
    exact283747RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283747 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7279⟩⟩) exact283747RawTerms .large 283746 .exactZero (none)

def event283748 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9547⟩⟩) 0 ⟨7279⟩ 283747

def event283749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9547⟩⟩) (.authority (.operator))

def exact283750RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩, (1)⟩]

theorem exact283750RawTermsValid :
    exact283750RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283750 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9547⟩⟩) exact283750RawTerms (.finite 8192) 283749 .exactZero (none)

def event283751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9548⟩⟩) 0 ⟨9547⟩ 283750

def event283752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9548⟩⟩) 1 ⟨2370⟩ 283684

def event283753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9548⟩⟩) (.scale (.predecessor 0 283751 .coefficient) (.value (.predecessor 1 283752 .coefficient)))

def exact283754RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩, (1)⟩]

theorem exact283754RawTermsValid :
    exact283754RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283754 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9548⟩⟩) exact283754RawTerms (.finite 8192) 283753 .exactZero (none)

def event283755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7296⟩⟩) 0 ⟨7178⟩ 283744

def event283756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7296⟩⟩) (.identity (.predecessor 0 283755 .coefficient))

def exact283757RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩]⟩, (1)⟩]

theorem exact283757RawTermsValid :
    exact283757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283757 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7296⟩⟩) exact283757RawTerms .large 283756 .exactZero (none)

def event283758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9549⟩⟩) 0 ⟨7296⟩ 283757

def event283759 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9549⟩⟩) 1 ⟨9548⟩ 283754

def event283760 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9549⟩⟩) (.product (.predecessor 0 283758 .coefficient) (.predecessor 1 283759 .coefficient) (⟨false, false, none, none, none⟩))

def event283761 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9549⟩⟩, .operator (⟨283757, 0⟩, ⟨283754, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩, (1)⟩)

def exact283762RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩, (1)⟩]

theorem exact283762RawTermsValid :
    exact283762RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283762 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9549⟩⟩) exact283762RawTerms .large 283760 .exactZero (none)

def event283763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30345⟩⟩) 0 ⟨9549⟩ 283762

def event283764 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30345⟩⟩) 1 ⟨30344⟩ 283741

def event283765 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30345⟩⟩) (.sum [.predecessor 0 283763 .coefficient, .predecessor 1 283764 .coefficient])

def exact283766RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13191⟩⟩, ⟨.program ⟨257⟩, ⟨28630⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283766RawTermsValid :
    exact283766RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283766 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30345⟩⟩) exact283766RawTerms .large 283765 .exactZero (none)

def event283767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30536⟩⟩) 0 ⟨30345⟩ 283766

def event283768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30536⟩⟩) 1 ⟨30533⟩ 283725

def event283769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30536⟩⟩) (.product (.predecessor 0 283767 .coefficient) (.predecessor 1 283768 .coefficient) (⟨false, false, none, none, none⟩))

def event283770 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30536⟩⟩, .operator (⟨283766, 0⟩, ⟨283725, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30533⟩⟩]⟩, (1)⟩)

def event283771 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30536⟩⟩, .operator (⟨283766, 1⟩, ⟨283725, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13191⟩⟩, ⟨.program ⟨257⟩, ⟨28630⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30533⟩⟩]⟩, (-1)⟩)

def event283772 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨30536⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨13191⟩⟩, ⟨.program ⟨257⟩, ⟨28630⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30533⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30533⟩⟩) ⟨30053⟩ 283722)

def event283773 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30536⟩⟩, .relation 283772 0, ⟨[⟨.program ⟨257⟩, ⟨13191⟩⟩, ⟨.program ⟨257⟩, ⟨28630⟩⟩], [⟨.program ⟨257⟩, ⟨30053⟩⟩]⟩, (-1)⟩)

def exact283774RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30533⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13191⟩⟩, ⟨.program ⟨257⟩, ⟨28630⟩⟩], [⟨.program ⟨257⟩, ⟨30053⟩⟩]⟩, (-1)⟩]

theorem exact283774RawTermsValid :
    exact283774RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283774 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30536⟩⟩) exact283774RawTerms .large 283769 .exactZero (none)

def event283775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29040⟩⟩) 0 ⟨28632⟩ 283714

def event283776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29040⟩⟩) (.authority (.programFamilyFact))

def exact283777RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29040⟩⟩], []⟩, (1)⟩]

theorem exact283777RawTermsValid :
    exact283777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283777 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29040⟩⟩) exact283777RawTerms (.finite 36) 283776 .exactZero (none)

def event283778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29042⟩⟩) 0 ⟨6908⟩ 283736

def event283779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29042⟩⟩) 1 ⟨29040⟩ 283777

def event283780 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29042⟩⟩) (.product (.predecessor 0 283778 .coefficient) (.predecessor 1 283779 .coefficient) (⟨false, true, none, none, some 1⟩))

def event283781 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29042⟩⟩, .operator (⟨283736, 0⟩, ⟨283777, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact283782RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact283782RawTermsValid :
    exact283782RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283782 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29042⟩⟩) exact283782RawTerms .large 283780 .exactZero (none)

def event283783 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7190⟩⟩) 0 ⟨7177⟩ 283718

def event283784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7190⟩⟩) (.authority (.operator))

def exact283785RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩]

theorem exact283785RawTermsValid :
    exact283785RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283785 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7190⟩⟩) exact283785RawTerms .large 283784 .exactZero (none)

def event283786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29043⟩⟩) 0 ⟨7190⟩ 283785

def event283787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29043⟩⟩) 1 ⟨29042⟩ 283782

def event283788 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29043⟩⟩) (.sum [.predecessor 0 283786 .coefficient, .predecessor 1 283787 .coefficient])

def exact283789RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283789RawTermsValid :
    exact283789RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283789 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29043⟩⟩) exact283789RawTerms .large 283788 .exactZero (none)

def event283790 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30537⟩⟩) 0 ⟨29043⟩ 283789

def event283791 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30537⟩⟩) 1 ⟨30536⟩ 283774

def event283792 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30537⟩⟩) (.sum [.predecessor 0 283790 .coefficient, .predecessor 1 283791 .coefficient])

def exact283793RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30533⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13191⟩⟩, ⟨.program ⟨257⟩, ⟨28630⟩⟩], [⟨.program ⟨257⟩, ⟨30053⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283793RawTermsValid :
    exact283793RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283793 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30537⟩⟩) exact283793RawTerms .large 283792 .exactZero (none)

def event283794 : Event := .preFoldPolynomial 283793 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30533⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13191⟩⟩, ⟨.program ⟨257⟩, ⟨28630⟩⟩], [⟨.program ⟨257⟩, ⟨30053⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact283795RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30533⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13191⟩⟩, ⟨.program ⟨257⟩, ⟨28630⟩⟩], [⟨.program ⟨257⟩, ⟨30053⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event283795 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨30537⟩⟩) 283794 exact283795RawTerms .large 283792 .exactZero (none)

def event283796 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨28632⟩⟩) ⟨⟨69⟩, ⟨48⟩, ⟨135⟩⟩ ⟨283632, 283796⟩

def event283797 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨29472⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29469⟩⟩]⟩) (1) 0 2 (.universal 283796 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29469⟩⟩]⟩) (none) 283795)

def event283798 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29472⟩⟩, .relation 283797 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩)

def event283799 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29472⟩⟩, .relation 283797 1, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30533⟩⟩]⟩, (-1)⟩)

def event283800 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29472⟩⟩, .relation 283797 2, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13191⟩⟩, ⟨.program ⟨257⟩, ⟨28630⟩⟩], [⟨.program ⟨257⟩, ⟨30053⟩⟩]⟩, (1)⟩)

def event283801 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29472⟩⟩, .relation 283797 3, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact283802RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30533⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13191⟩⟩, ⟨.program ⟨257⟩, ⟨28630⟩⟩], [⟨.program ⟨257⟩, ⟨30053⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283802RawTermsValid :
    exact283802RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283802 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29472⟩⟩) exact283802RawTerms .large 283628 (.finite 202072841853861888) (some (283630))

def event283803 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30535⟩⟩) 0 ⟨29472⟩ 283802

def event283804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30535⟩⟩) 1 ⟨30534⟩ 283618

def event283805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30535⟩⟩) (.sum [.predecessor 0 283803 .coefficient, .predecessor 1 283804 .coefficient])

def event283806 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30535⟩⟩, .operator (⟨283802, 2⟩, ⟨283618, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13191⟩⟩, ⟨.program ⟨257⟩, ⟨28630⟩⟩], [⟨.program ⟨257⟩, ⟨30053⟩⟩]⟩, (-1)⟩)

def event283807 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30535⟩⟩, .operator (⟨283802, 1⟩, ⟨283618, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30533⟩⟩]⟩, (1)⟩)

def event283808 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30535⟩⟩) (.sum [.result 283802 .summary, .result 283618 .summary])

def exact283809RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283809RawTermsValid :
    exact283809RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283809 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30535⟩⟩) exact283809RawTerms .large 283805 (.finite 2998127310542407467008) (some (283808))

def event283810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30821⟩⟩) 0 ⟨30535⟩ 283809

def event283811 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30821⟩⟩) 1 ⟨30819⟩ 283534

def event283812 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30821⟩⟩) (.product (.predecessor 0 283810 .coefficient) (.predecessor 1 283811 .coefficient) (⟨false, false, none, none, none⟩))

def event283813 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30821⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨30819⟩⟩]⟩) [⟨.result 283534 .coefficient, false, none⟩])

def event283814 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30821⟩⟩) (.product (.result 283809 .summary) (.transfer 283813) (⟨false, false, none, none, none⟩))

def event283815 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30821⟩⟩, .operator (⟨283809, 0⟩, ⟨283534, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30819⟩⟩]⟩, (1)⟩)

def event283816 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30821⟩⟩, .operator (⟨283809, 1⟩, ⟨283534, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30819⟩⟩]⟩, (-1)⟩)

def event283817 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨30821⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30819⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30819⟩⟩) ⟨30187⟩ 283531)

def event283818 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨30821⟩⟩, .relation 283817 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨30187⟩⟩]⟩, (-1)⟩)

def exact283819RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30819⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨29040⟩⟩], [⟨.program ⟨257⟩, ⟨30187⟩⟩]⟩, (-1)⟩]

theorem exact283819RawTermsValid :
    exact283819RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283819 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30821⟩⟩) exact283819RawTerms .large 283812 (.finite 32192146870060190229763897425920) (some (283814))

def event283820 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29716⟩⟩) 0 ⟨29041⟩ 13707

def event283821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29716⟩⟩) (.authority (.relationPreimageSource ⟨81⟩))

def exact283822RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29716⟩⟩]⟩, (1)⟩]

theorem exact283822RawTermsValid :
    exact283822RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283822 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29716⟩⟩) exact283822RawTerms (.finite 5647228698) 283821 .exactZero (none)

def event283823 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29718⟩⟩) 0 ⟨29716⟩ 283822

def event283824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29718⟩⟩) 1 ⟨2370⟩ 4

def event283825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29718⟩⟩) (.scale (.predecessor 0 283823 .coefficient) (.value (.predecessor 1 283824 .coefficient)))

def exact283826RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29716⟩⟩]⟩, (1)⟩]

theorem exact283826RawTermsValid :
    exact283826RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283826 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29718⟩⟩) exact283826RawTerms (.finite 5647228698) 283825 .exactZero (none)

def event283827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29719⟩⟩) 0 ⟨5491⟩ 280745

def event283828 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29719⟩⟩) 1 ⟨29718⟩ 283826

def event283829 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29719⟩⟩) (.product (.predecessor 0 283827 .coefficient) (.predecessor 1 283828 .coefficient) (⟨false, false, none, none, none⟩))

def event283830 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29719⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨29716⟩⟩]⟩) [⟨.result 283822 .coefficient, false, none⟩])

def event283831 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29719⟩⟩) (.product (.result 280745 .summary) (.transfer 283830) (⟨false, false, none, none, none⟩))

def event283832 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29719⟩⟩, .operator (⟨280745, 0⟩, ⟨283826, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29716⟩⟩]⟩, (1)⟩)

def event283833 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨29717⟩⟩)

def event283834 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event283835 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event283836 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event283837 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event283838 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event283839 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event283840 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event283841 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event283842 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 283841

def event283843 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 283839

def event283844 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 283842 .coefficient) (.value (.predecessor 1 283843 .coefficient)))

def event283845 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event283846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 283845

def event283847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 283837

def event283848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 283846 .coefficient, .predecessor 1 283847 .coefficient])

def event283849 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event283850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 283849

def event283851 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 283835

def event283852 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 283851 .coefficient))

def event283853 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event283854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28630⟩⟩) 0 ⟨5487⟩ 283853

def event283855 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28630⟩⟩) (.authority (.programFamilyFact))

def exact283856RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨28630⟩⟩], []⟩, (1)⟩]

theorem exact283856RawTermsValid :
    exact283856RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283856 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28630⟩⟩) exact283856RawTerms (.finite 36) 283855 .exactZero (none)

def event283857 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13191⟩⟩) 0 ⟨5487⟩ 283853

def event283858 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13191⟩⟩) (.authority (.programFamilyFact))

def exact283859RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13191⟩⟩], []⟩, (1)⟩]

theorem exact283859RawTermsValid :
    exact283859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283859 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13191⟩⟩) exact283859RawTerms (.finite 36) 283858 .exactZero (none)

def event283860 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28631⟩⟩) 0 ⟨13191⟩ 283859

def event283861 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28631⟩⟩) 1 ⟨28630⟩ 283856

def event283862 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28631⟩⟩) (.product (.predecessor 0 283860 .coefficient) (.predecessor 1 283861 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event283863 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28631⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13191⟩⟩, ⟨.program ⟨257⟩, ⟨28630⟩⟩], []⟩) [⟨.result 283859 .coefficient, true, some 1⟩, ⟨.result 283856 .coefficient, true, some 1⟩])

def event283864 : Event := .survivorFold (1) 283863

def exact283865RawTerms : List Term := []

theorem exact283865RawTermsValid :
    exact283865RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283865 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28631⟩⟩) exact283865RawTerms (.finite 1296) 283862 (.finite 1296) (some (283863))

def event283866 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28632⟩⟩) 0 ⟨28631⟩ 283865

def event283867 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28632⟩⟩) (.identity (.predecessor 0 283866 .coefficient))

def event283868 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨28632⟩⟩) (.finite 1296)

def event283869 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29040⟩⟩) 0 ⟨28632⟩ 283868

def event283870 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29040⟩⟩) (.authority (.programFamilyFact))

def exact283871RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29040⟩⟩], []⟩, (1)⟩]

theorem exact283871RawTermsValid :
    exact283871RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283871 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29040⟩⟩) exact283871RawTerms (.finite 36) 283870 .exactZero (none)

def event283872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29041⟩⟩) 0 ⟨29040⟩ 283871

def event283873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29041⟩⟩) (.identity (.predecessor 0 283872 .coefficient))

def event283874 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨29041⟩⟩) (.finite 36)

def event283875 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29716⟩⟩) 0 ⟨29041⟩ 283874

def event283876 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29716⟩⟩) (.authority (.relationPreimageSource ⟨81⟩))

def exact283877RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨29716⟩⟩]⟩, (1)⟩]

theorem exact283877RawTermsValid :
    exact283877RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283877 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29716⟩⟩) exact283877RawTerms (.finite 5647228698) 283876 .exactZero (none)

def event283878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact283879RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact283879RawTermsValid :
    exact283879RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283879 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact283879RawTerms .large 283878 .exactZero (none)

def event283880 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29717⟩⟩) 0 ⟨35⟩ 283879

def event283881 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29717⟩⟩) 1 ⟨29716⟩ 283877

def event283882 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29717⟩⟩) (.product (.predecessor 0 283880 .coefficient) (.predecessor 1 283881 .coefficient) (⟨false, false, none, none, none⟩))

def event283883 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29717⟩⟩, .operator (⟨283879, 0⟩, ⟨283877, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29716⟩⟩]⟩, (1)⟩)

def exact283884RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29716⟩⟩]⟩, (1)⟩]

theorem exact283884RawTermsValid :
    exact283884RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283884 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29717⟩⟩) exact283884RawTerms .large 283882 .exactZero (none)

def event283885 : Event := .preFoldPolynomial 283884 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29716⟩⟩]⟩, (1)⟩] .exactZero none

def exact283886RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29716⟩⟩]⟩, (1)⟩]

def event283886 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨29717⟩⟩) 283885 exact283886RawTerms .large 283882 .exactZero (none)

def event283887 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨30823⟩⟩)

def event283888 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event283889 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event283890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event283891 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event283892 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event283893 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event283894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event283895 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event283896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 283895

def event283897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 283893

def event283898 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 283896 .coefficient) (.value (.predecessor 1 283897 .coefficient)))

def event283899 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event283900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 283899

def event283901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 283891

def event283902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 283900 .coefficient, .predecessor 1 283901 .coefficient])

def event283903 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def eventLeaf17728 : Array AnnotatedEvent := #[
  { event := event283648
    frameStart := 283632 },
  { event := event283649
    frameStart := 283632 },
  { event := event283650
    frameStart := 283632 },
  { event := event283651
    frameStart := 283632 },
  { event := event283652
    frameStart := 283632 },
  { event := event283653
    frameStart := 283632 },
  { event := event283654
    frameStart := 283632 },
  { event := event283655
    frameStart := 283632 },
  { event := event283656
    frameStart := 283632 },
  { event := event283657
    frameStart := 283632 },
  { event := event283658
    frameStart := 283632 },
  { event := event283659
    frameStart := 283632 },
  { event := event283660
    frameStart := 283632 },
  { event := event283661
    frameStart := 283632 },
  { event := event283662
    frameStart := 283632 },
  { event := event283663
    frameStart := 283632 }
]

def eventLeaf17729 : Array AnnotatedEvent := #[
  { event := event283664
    frameStart := 283632 },
  { event := event283665
    frameStart := 283632 },
  { event := event283666
    frameStart := 283632 },
  { event := event283667
    frameStart := 283632 },
  { event := event283668
    frameStart := 283632 },
  { event := event283669
    frameStart := 283632 },
  { event := event283670
    frameStart := 283632 },
  { event := event283671
    frameStart := 283632 },
  { event := event283672
    frameStart := 283632 },
  { event := event283673
    frameStart := 283632 },
  { event := event283674
    frameStart := 283632 },
  { event := event283675
    frameStart := 283632 },
  { event := event283676
    frameStart := 283632 },
  { event := event283677
    frameStart := 283632 },
  { event := event283678
    frameStart := 283632 },
  { event := event283679
    frameStart := 283632 }
]

def eventLeaf17730 : Array AnnotatedEvent := #[
  { event := event283680
    frameStart := 283680 },
  { event := event283681
    frameStart := 283680 },
  { event := event283682
    frameStart := 283680 },
  { event := event283683
    frameStart := 283680 },
  { event := event283684
    frameStart := 283680 },
  { event := event283685
    frameStart := 283680 },
  { event := event283686
    frameStart := 283680 },
  { event := event283687
    frameStart := 283680 },
  { event := event283688
    frameStart := 283680 },
  { event := event283689
    frameStart := 283680 },
  { event := event283690
    frameStart := 283680 },
  { event := event283691
    frameStart := 283680 },
  { event := event283692
    frameStart := 283680 },
  { event := event283693
    frameStart := 283680 },
  { event := event283694
    frameStart := 283680 },
  { event := event283695
    frameStart := 283680 }
]

def eventLeaf17731 : Array AnnotatedEvent := #[
  { event := event283696
    frameStart := 283680 },
  { event := event283697
    frameStart := 283680 },
  { event := event283698
    frameStart := 283680 },
  { event := event283699
    frameStart := 283680 },
  { event := event283700
    frameStart := 283680 },
  { event := event283701
    frameStart := 283680 },
  { event := event283702
    frameStart := 283680 },
  { event := event283703
    frameStart := 283680 },
  { event := event283704
    frameStart := 283680 },
  { event := event283705
    frameStart := 283680 },
  { event := event283706
    frameStart := 283680 },
  { event := event283707
    frameStart := 283680 },
  { event := event283708
    frameStart := 283680 },
  { event := event283709
    frameStart := 283680 },
  { event := event283710
    frameStart := 283680 },
  { event := event283711
    frameStart := 283680 }
]

def eventLeaf17732 : Array AnnotatedEvent := #[
  { event := event283712
    frameStart := 283680 },
  { event := event283713
    frameStart := 283680 },
  { event := event283714
    frameStart := 283680 },
  { event := event283715
    frameStart := 283680 },
  { event := event283716
    frameStart := 283680 },
  { event := event283717
    frameStart := 283680 },
  { event := event283718
    frameStart := 283680 },
  { event := event283719
    frameStart := 283680 },
  { event := event283720
    frameStart := 283680 },
  { event := event283721
    frameStart := 283680 },
  { event := event283722
    frameStart := 283680 },
  { event := event283723
    frameStart := 283680 },
  { event := event283724
    frameStart := 283680 },
  { event := event283725
    frameStart := 283680 },
  { event := event283726
    frameStart := 283680 },
  { event := event283727
    frameStart := 283680 }
]

def eventLeaf17733 : Array AnnotatedEvent := #[
  { event := event283728
    frameStart := 283680 },
  { event := event283729
    frameStart := 283680 },
  { event := event283730
    frameStart := 283680 },
  { event := event283731
    frameStart := 283680 },
  { event := event283732
    frameStart := 283680 },
  { event := event283733
    frameStart := 283680 },
  { event := event283734
    frameStart := 283680 },
  { event := event283735
    frameStart := 283680 },
  { event := event283736
    frameStart := 283680 },
  { event := event283737
    frameStart := 283680 },
  { event := event283738
    frameStart := 283680 },
  { event := event283739
    frameStart := 283680 },
  { event := event283740
    frameStart := 283680 },
  { event := event283741
    frameStart := 283680 },
  { event := event283742
    frameStart := 283680 },
  { event := event283743
    frameStart := 283680 }
]

def eventLeaf17734 : Array AnnotatedEvent := #[
  { event := event283744
    frameStart := 283680 },
  { event := event283745
    frameStart := 283680 },
  { event := event283746
    frameStart := 283680 },
  { event := event283747
    frameStart := 283680 },
  { event := event283748
    frameStart := 283680 },
  { event := event283749
    frameStart := 283680 },
  { event := event283750
    frameStart := 283680 },
  { event := event283751
    frameStart := 283680 },
  { event := event283752
    frameStart := 283680 },
  { event := event283753
    frameStart := 283680 },
  { event := event283754
    frameStart := 283680 },
  { event := event283755
    frameStart := 283680 },
  { event := event283756
    frameStart := 283680 },
  { event := event283757
    frameStart := 283680 },
  { event := event283758
    frameStart := 283680 },
  { event := event283759
    frameStart := 283680 }
]

def eventLeaf17735 : Array AnnotatedEvent := #[
  { event := event283760
    frameStart := 283680 },
  { event := event283761
    frameStart := 283680 },
  { event := event283762
    frameStart := 283680 },
  { event := event283763
    frameStart := 283680 },
  { event := event283764
    frameStart := 283680 },
  { event := event283765
    frameStart := 283680 },
  { event := event283766
    frameStart := 283680 },
  { event := event283767
    frameStart := 283680 },
  { event := event283768
    frameStart := 283680 },
  { event := event283769
    frameStart := 283680 },
  { event := event283770
    frameStart := 283680 },
  { event := event283771
    frameStart := 283680 },
  { event := event283772
    frameStart := 283680 },
  { event := event283773
    frameStart := 283680 },
  { event := event283774
    frameStart := 283680 },
  { event := event283775
    frameStart := 283680 }
]

def eventLeaf17736 : Array AnnotatedEvent := #[
  { event := event283776
    frameStart := 283680 },
  { event := event283777
    frameStart := 283680 },
  { event := event283778
    frameStart := 283680 },
  { event := event283779
    frameStart := 283680 },
  { event := event283780
    frameStart := 283680 },
  { event := event283781
    frameStart := 283680 },
  { event := event283782
    frameStart := 283680 },
  { event := event283783
    frameStart := 283680 },
  { event := event283784
    frameStart := 283680 },
  { event := event283785
    frameStart := 283680 },
  { event := event283786
    frameStart := 283680 },
  { event := event283787
    frameStart := 283680 },
  { event := event283788
    frameStart := 283680 },
  { event := event283789
    frameStart := 283680 },
  { event := event283790
    frameStart := 283680 },
  { event := event283791
    frameStart := 283680 }
]

def eventLeaf17737 : Array AnnotatedEvent := #[
  { event := event283792
    frameStart := 283680 },
  { event := event283793
    frameStart := 283680 },
  { event := event283794
    frameStart := 283680 },
  { event := event283795
    frameStart := 283680 },
  { event := event283796
    frameStart := 0 },
  { event := event283797
    frameStart := 0 },
  { event := event283798
    frameStart := 0 },
  { event := event283799
    frameStart := 0 },
  { event := event283800
    frameStart := 0 },
  { event := event283801
    frameStart := 0 },
  { event := event283802
    frameStart := 0 },
  { event := event283803
    frameStart := 0 },
  { event := event283804
    frameStart := 0 },
  { event := event283805
    frameStart := 0 },
  { event := event283806
    frameStart := 0 },
  { event := event283807
    frameStart := 0 }
]

def eventLeaf17738 : Array AnnotatedEvent := #[
  { event := event283808
    frameStart := 0 },
  { event := event283809
    frameStart := 0 },
  { event := event283810
    frameStart := 0 },
  { event := event283811
    frameStart := 0 },
  { event := event283812
    frameStart := 0 },
  { event := event283813
    frameStart := 0 },
  { event := event283814
    frameStart := 0 },
  { event := event283815
    frameStart := 0 },
  { event := event283816
    frameStart := 0 },
  { event := event283817
    frameStart := 0 },
  { event := event283818
    frameStart := 0 },
  { event := event283819
    frameStart := 0 },
  { event := event283820
    frameStart := 0 },
  { event := event283821
    frameStart := 0 },
  { event := event283822
    frameStart := 0 },
  { event := event283823
    frameStart := 0 }
]

def eventLeaf17739 : Array AnnotatedEvent := #[
  { event := event283824
    frameStart := 0 },
  { event := event283825
    frameStart := 0 },
  { event := event283826
    frameStart := 0 },
  { event := event283827
    frameStart := 0 },
  { event := event283828
    frameStart := 0 },
  { event := event283829
    frameStart := 0 },
  { event := event283830
    frameStart := 0 },
  { event := event283831
    frameStart := 0 },
  { event := event283832
    frameStart := 0 },
  { event := event283833
    frameStart := 283833 },
  { event := event283834
    frameStart := 283833 },
  { event := event283835
    frameStart := 283833 },
  { event := event283836
    frameStart := 283833 },
  { event := event283837
    frameStart := 283833 },
  { event := event283838
    frameStart := 283833 },
  { event := event283839
    frameStart := 283833 }
]

def eventLeaf17740 : Array AnnotatedEvent := #[
  { event := event283840
    frameStart := 283833 },
  { event := event283841
    frameStart := 283833 },
  { event := event283842
    frameStart := 283833 },
  { event := event283843
    frameStart := 283833 },
  { event := event283844
    frameStart := 283833 },
  { event := event283845
    frameStart := 283833 },
  { event := event283846
    frameStart := 283833 },
  { event := event283847
    frameStart := 283833 },
  { event := event283848
    frameStart := 283833 },
  { event := event283849
    frameStart := 283833 },
  { event := event283850
    frameStart := 283833 },
  { event := event283851
    frameStart := 283833 },
  { event := event283852
    frameStart := 283833 },
  { event := event283853
    frameStart := 283833 },
  { event := event283854
    frameStart := 283833 },
  { event := event283855
    frameStart := 283833 }
]

def eventLeaf17741 : Array AnnotatedEvent := #[
  { event := event283856
    frameStart := 283833 },
  { event := event283857
    frameStart := 283833 },
  { event := event283858
    frameStart := 283833 },
  { event := event283859
    frameStart := 283833 },
  { event := event283860
    frameStart := 283833 },
  { event := event283861
    frameStart := 283833 },
  { event := event283862
    frameStart := 283833 },
  { event := event283863
    frameStart := 283833 },
  { event := event283864
    frameStart := 283833 },
  { event := event283865
    frameStart := 283833 },
  { event := event283866
    frameStart := 283833 },
  { event := event283867
    frameStart := 283833 },
  { event := event283868
    frameStart := 283833 },
  { event := event283869
    frameStart := 283833 },
  { event := event283870
    frameStart := 283833 },
  { event := event283871
    frameStart := 283833 }
]

def eventLeaf17742 : Array AnnotatedEvent := #[
  { event := event283872
    frameStart := 283833 },
  { event := event283873
    frameStart := 283833 },
  { event := event283874
    frameStart := 283833 },
  { event := event283875
    frameStart := 283833 },
  { event := event283876
    frameStart := 283833 },
  { event := event283877
    frameStart := 283833 },
  { event := event283878
    frameStart := 283833 },
  { event := event283879
    frameStart := 283833 },
  { event := event283880
    frameStart := 283833 },
  { event := event283881
    frameStart := 283833 },
  { event := event283882
    frameStart := 283833 },
  { event := event283883
    frameStart := 283833 },
  { event := event283884
    frameStart := 283833 },
  { event := event283885
    frameStart := 283833 },
  { event := event283886
    frameStart := 283833 },
  { event := event283887
    frameStart := 283887 }
]

def eventLeaf17743 : Array AnnotatedEvent := #[
  { event := event283888
    frameStart := 283887 },
  { event := event283889
    frameStart := 283887 },
  { event := event283890
    frameStart := 283887 },
  { event := event283891
    frameStart := 283887 },
  { event := event283892
    frameStart := 283887 },
  { event := event283893
    frameStart := 283887 },
  { event := event283894
    frameStart := 283887 },
  { event := event283895
    frameStart := 283887 },
  { event := event283896
    frameStart := 283887 },
  { event := event283897
    frameStart := 283887 },
  { event := event283898
    frameStart := 283887 },
  { event := event283899
    frameStart := 283887 },
  { event := event283900
    frameStart := 283887 },
  { event := event283901
    frameStart := 283887 },
  { event := event283902
    frameStart := 283887 },
  { event := event283903
    frameStart := 283887 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1108
