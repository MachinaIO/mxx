import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events530

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event135680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 135678 .coefficient) (.value (.predecessor 1 135679 .coefficient)))

def event135681 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event135682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 135681

def event135683 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 135673

def event135684 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 135682 .coefficient, .predecessor 1 135683 .coefficient])

def event135685 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event135686 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 135685

def event135687 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 135671

def event135688 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 135687 .coefficient))

def event135689 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event135690 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42306⟩⟩) 0 ⟨5469⟩ 135689

def event135691 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42306⟩⟩) (.authority (.programFamilyFact))

def exact135692RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42306⟩⟩], []⟩, (1)⟩]

theorem exact135692RawTermsValid :
    exact135692RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135692 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42306⟩⟩) exact135692RawTerms (.finite 52) 135691 .exactZero (none)

def event135693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14376⟩⟩) 0 ⟨5469⟩ 135689

def event135694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14376⟩⟩) (.authority (.programFamilyFact))

def exact135695RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14376⟩⟩], []⟩, (1)⟩]

theorem exact135695RawTermsValid :
    exact135695RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135695 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14376⟩⟩) exact135695RawTerms (.finite 52) 135694 .exactZero (none)

def event135696 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42307⟩⟩) 0 ⟨14376⟩ 135695

def event135697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42307⟩⟩) 1 ⟨42306⟩ 135692

def event135698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42307⟩⟩) (.product (.predecessor 0 135696 .coefficient) (.predecessor 1 135697 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event135699 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42307⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14376⟩⟩, ⟨.program ⟨257⟩, ⟨42306⟩⟩], []⟩) [⟨.result 135695 .coefficient, true, some 1⟩, ⟨.result 135692 .coefficient, true, some 1⟩])

def event135700 : Event := .survivorFold (1) 135699

def exact135701RawTerms : List Term := []

theorem exact135701RawTermsValid :
    exact135701RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135701 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42307⟩⟩) exact135701RawTerms (.finite 2704) 135698 (.finite 2704) (some (135699))

def event135702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42308⟩⟩) 0 ⟨42307⟩ 135701

def event135703 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42308⟩⟩) (.identity (.predecessor 0 135702 .coefficient))

def event135704 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42308⟩⟩) (.finite 2704)

def event135705 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42732⟩⟩) 0 ⟨42308⟩ 135704

def event135706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42732⟩⟩) (.authority (.programFamilyFact))

def exact135707RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], []⟩, (1)⟩]

theorem exact135707RawTermsValid :
    exact135707RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135707 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42732⟩⟩) exact135707RawTerms (.finite 52) 135706 .exactZero (none)

def event135708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42733⟩⟩) 0 ⟨42732⟩ 135707

def event135709 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42733⟩⟩) (.identity (.predecessor 0 135708 .coefficient))

def event135710 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42733⟩⟩) (.finite 52)

def event135711 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43396⟩⟩) 0 ⟨42733⟩ 135710

def event135712 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43396⟩⟩) (.authority (.relationPreimageSource ⟨90⟩))

def exact135713RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43396⟩⟩]⟩, (1)⟩]

theorem exact135713RawTermsValid :
    exact135713RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135713 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43396⟩⟩) exact135713RawTerms (.finite 5647228698) 135712 .exactZero (none)

def event135714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact135715RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact135715RawTermsValid :
    exact135715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135715 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact135715RawTerms .large 135714 .exactZero (none)

def event135716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43397⟩⟩) 0 ⟨35⟩ 135715

def event135717 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43397⟩⟩) 1 ⟨43396⟩ 135713

def event135718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43397⟩⟩) (.product (.predecessor 0 135716 .coefficient) (.predecessor 1 135717 .coefficient) (⟨false, false, none, none, none⟩))

def event135719 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43397⟩⟩, .operator (⟨135715, 0⟩, ⟨135713, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43396⟩⟩]⟩, (1)⟩)

def exact135720RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43396⟩⟩]⟩, (1)⟩]

theorem exact135720RawTermsValid :
    exact135720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135720 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43397⟩⟩) exact135720RawTerms .large 135718 .exactZero (none)

def event135721 : Event := .preFoldPolynomial 135720 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43396⟩⟩]⟩, (1)⟩] .exactZero none

def exact135722RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43396⟩⟩]⟩, (1)⟩]

def event135722 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨43397⟩⟩) 135721 exact135722RawTerms .large 135718 .exactZero (none)

def event135723 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨44498⟩⟩)

def event135724 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event135725 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event135726 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event135727 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event135728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event135729 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event135730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event135731 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event135732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 135731

def event135733 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 135729

def event135734 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 135732 .coefficient) (.value (.predecessor 1 135733 .coefficient)))

def event135735 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event135736 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 135735

def event135737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 135727

def event135738 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 135736 .coefficient, .predecessor 1 135737 .coefficient])

def event135739 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event135740 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 135739

def event135741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 135725

def event135742 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 135741 .coefficient))

def event135743 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event135744 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42306⟩⟩) 0 ⟨5469⟩ 135743

def event135745 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42306⟩⟩) (.authority (.programFamilyFact))

def exact135746RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42306⟩⟩], []⟩, (1)⟩]

theorem exact135746RawTermsValid :
    exact135746RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135746 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42306⟩⟩) exact135746RawTerms (.finite 52) 135745 .exactZero (none)

def event135747 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14376⟩⟩) 0 ⟨5469⟩ 135743

def event135748 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14376⟩⟩) (.authority (.programFamilyFact))

def exact135749RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14376⟩⟩], []⟩, (1)⟩]

theorem exact135749RawTermsValid :
    exact135749RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135749 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14376⟩⟩) exact135749RawTerms (.finite 52) 135748 .exactZero (none)

def event135750 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42307⟩⟩) 0 ⟨14376⟩ 135749

def event135751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42307⟩⟩) 1 ⟨42306⟩ 135746

def event135752 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42307⟩⟩) (.product (.predecessor 0 135750 .coefficient) (.predecessor 1 135751 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event135753 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42307⟩⟩, .operator (⟨135749, 0⟩, ⟨135746, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14376⟩⟩, ⟨.program ⟨257⟩, ⟨42306⟩⟩], []⟩, (1)⟩)

def exact135754RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14376⟩⟩, ⟨.program ⟨257⟩, ⟨42306⟩⟩], []⟩, (1)⟩]

theorem exact135754RawTermsValid :
    exact135754RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135754 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42307⟩⟩) exact135754RawTerms (.finite 2704) 135752 .exactZero (none)

def event135755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42308⟩⟩) 0 ⟨42307⟩ 135754

def event135756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42308⟩⟩) (.identity (.predecessor 0 135755 .coefficient))

def event135757 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42308⟩⟩) (.finite 2704)

def event135758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42732⟩⟩) 0 ⟨42308⟩ 135757

def event135759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42732⟩⟩) (.authority (.programFamilyFact))

def exact135760RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], []⟩, (1)⟩]

theorem exact135760RawTermsValid :
    exact135760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135760 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42732⟩⟩) exact135760RawTerms (.finite 52) 135759 .exactZero (none)

def event135761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42733⟩⟩) 0 ⟨42732⟩ 135760

def event135762 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42733⟩⟩) (.identity (.predecessor 0 135761 .coefficient))

def event135763 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42733⟩⟩) (.finite 52)

def event135764 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43876⟩⟩) 0 ⟨42733⟩ 135763

def event135765 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43876⟩⟩) (.authority (.programFamilyFact))

def event135766 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨43876⟩⟩) (.finite 3720)

def event135767 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event135768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43878⟩⟩) 0 ⟨7177⟩ 135767

def event135769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43878⟩⟩) 1 ⟨43876⟩ 135766

def event135770 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43878⟩⟩) (.authority (.operator))

def exact135771RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43878⟩⟩]⟩, (1)⟩]

theorem exact135771RawTermsValid :
    exact135771RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135771 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43878⟩⟩) exact135771RawTerms .large 135770 .exactZero (none)

def event135772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44494⟩⟩) 0 ⟨43878⟩ 135771

def event135773 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44494⟩⟩) (.authority (.operator))

def exact135774RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44494⟩⟩]⟩, (1)⟩]

theorem exact135774RawTermsValid :
    exact135774RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135774 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44494⟩⟩) exact135774RawTerms (.finite 8192) 135773 .exactZero (none)

def event135775 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event135776 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event135777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44118⟩⟩) 0 ⟨42733⟩ 135763

def event135778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44118⟩⟩) 1 ⟨136⟩ 135776

def event135779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44118⟩⟩) (.sum [.predecessor 0 135777 .coefficient, .predecessor 1 135778 .coefficient])

def event135780 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44118⟩⟩) (.finite 52)

def event135781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44119⟩⟩) 0 ⟨44118⟩ 135780

def event135782 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44119⟩⟩) (.identity (.predecessor 0 135781 .coefficient))

def exact135783RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], []⟩, (1)⟩]

theorem exact135783RawTermsValid :
    exact135783RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135783 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44119⟩⟩) exact135783RawTerms (.finite 52) 135782 .exactZero (none)

def event135784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact135785RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact135785RawTermsValid :
    exact135785RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135785 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact135785RawTerms .large 135784 .exactZero (none)

def event135786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44120⟩⟩) 0 ⟨6908⟩ 135785

def event135787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44120⟩⟩) 1 ⟨44119⟩ 135783

def event135788 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44120⟩⟩) (.product (.predecessor 0 135786 .coefficient) (.predecessor 1 135787 .coefficient) (⟨false, false, none, none, none⟩))

def event135789 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44120⟩⟩, .operator (⟨135785, 0⟩, ⟨135783, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact135790RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact135790RawTermsValid :
    exact135790RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135790 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44120⟩⟩) exact135790RawTerms .large 135788 .exactZero (none)

def event135791 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7194⟩⟩) 0 ⟨7177⟩ 135767

def event135792 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7194⟩⟩) (.authority (.operator))

def exact135793RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩]

theorem exact135793RawTermsValid :
    exact135793RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135793 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7194⟩⟩) exact135793RawTerms .large 135792 .exactZero (none)

def event135794 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44121⟩⟩) 0 ⟨7194⟩ 135793

def event135795 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44121⟩⟩) 1 ⟨44120⟩ 135790

def event135796 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44121⟩⟩) (.sum [.predecessor 0 135794 .coefficient, .predecessor 1 135795 .coefficient])

def exact135797RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact135797RawTermsValid :
    exact135797RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135797 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44121⟩⟩) exact135797RawTerms .large 135796 .exactZero (none)

def event135798 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44495⟩⟩) 0 ⟨44121⟩ 135797

def event135799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44495⟩⟩) 1 ⟨44494⟩ 135774

def event135800 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44495⟩⟩) (.product (.predecessor 0 135798 .coefficient) (.predecessor 1 135799 .coefficient) (⟨false, false, none, none, none⟩))

def event135801 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44495⟩⟩, .operator (⟨135797, 0⟩, ⟨135774, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44494⟩⟩]⟩, (1)⟩)

def event135802 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44495⟩⟩, .operator (⟨135797, 1⟩, ⟨135774, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44494⟩⟩]⟩, (-1)⟩)

def event135803 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44495⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44494⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44494⟩⟩) ⟨43878⟩ 135771)

def event135804 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44495⟩⟩, .relation 135803 0, ⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨43878⟩⟩]⟩, (-1)⟩)

def exact135805RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44494⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨43878⟩⟩]⟩, (-1)⟩]

theorem exact135805RawTermsValid :
    exact135805RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135805 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44495⟩⟩) exact135805RawTerms .large 135800 .exactZero (none)

def event135806 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42908⟩⟩) 0 ⟨42733⟩ 135763

def event135807 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42908⟩⟩) (.authority (.programFamilyFact))

def exact135808RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], []⟩, (1)⟩]

theorem exact135808RawTermsValid :
    exact135808RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135808 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42908⟩⟩) exact135808RawTerms (.finite 63) 135807 .exactZero (none)

def event135809 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42909⟩⟩) 0 ⟨6908⟩ 135785

def event135810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42909⟩⟩) 1 ⟨42908⟩ 135808

def event135811 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42909⟩⟩) (.product (.predecessor 0 135809 .coefficient) (.predecessor 1 135810 .coefficient) (⟨false, true, none, none, some 1⟩))

def event135812 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42909⟩⟩, .operator (⟨135785, 0⟩, ⟨135808, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact135813RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact135813RawTermsValid :
    exact135813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135813 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42909⟩⟩) exact135813RawTerms .large 135811 .exactZero (none)

def event135814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7228⟩⟩) 0 ⟨7177⟩ 135767

def event135815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7228⟩⟩) (.authority (.operator))

def exact135816RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact135816RawTermsValid :
    exact135816RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135816 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7228⟩⟩) exact135816RawTerms .large 135815 .exactZero (none)

def event135817 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42910⟩⟩) 0 ⟨7228⟩ 135816

def event135818 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42910⟩⟩) 1 ⟨42909⟩ 135813

def event135819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42910⟩⟩) (.sum [.predecessor 0 135817 .coefficient, .predecessor 1 135818 .coefficient])

def exact135820RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact135820RawTermsValid :
    exact135820RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135820 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42910⟩⟩) exact135820RawTerms .large 135819 .exactZero (none)

def event135821 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44498⟩⟩) 0 ⟨42910⟩ 135820

def event135822 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44498⟩⟩) 1 ⟨44495⟩ 135805

def event135823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44498⟩⟩) (.sum [.predecessor 0 135821 .coefficient, .predecessor 1 135822 .coefficient])

def exact135824RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44494⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨43878⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact135824RawTermsValid :
    exact135824RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135824 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44498⟩⟩) exact135824RawTerms .large 135823 .exactZero (none)

def event135825 : Event := .preFoldPolynomial 135824 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44494⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨43878⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact135826RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44494⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨43878⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event135826 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨44498⟩⟩) 135825 exact135826RawTerms .large 135823 .exactZero (none)

def event135827 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨42733⟩⟩) ⟨⟨107⟩, ⟨90⟩, ⟨135⟩⟩ ⟨135669, 135827⟩

def event135828 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨43399⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43396⟩⟩]⟩) (1) 0 2 (.universal 135827 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43396⟩⟩]⟩) (none) 135826)

def event135829 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43399⟩⟩, .relation 135828 1, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩)

def event135830 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43399⟩⟩, .relation 135828 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44494⟩⟩]⟩, (-1)⟩)

def event135831 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43399⟩⟩, .relation 135828 2, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨43878⟩⟩]⟩, (1)⟩)

def event135832 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43399⟩⟩, .relation 135828 3, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact135833RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44494⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨43878⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact135833RawTermsValid :
    exact135833RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135833 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43399⟩⟩) exact135833RawTerms .large 135665 (.finite 202072841853861888) (some (135667))

def event135834 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44497⟩⟩) 0 ⟨43399⟩ 135833

def event135835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44497⟩⟩) 1 ⟨44496⟩ 135655

def event135836 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44497⟩⟩) (.sum [.predecessor 0 135834 .coefficient, .predecessor 1 135835 .coefficient])

def event135837 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44497⟩⟩, .operator (⟨135833, 0⟩, ⟨135655, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44494⟩⟩]⟩, (1)⟩)

def event135838 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44497⟩⟩, .operator (⟨135833, 2⟩, ⟨135655, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42732⟩⟩], [⟨.program ⟨257⟩, ⟨43878⟩⟩]⟩, (-1)⟩)

def event135839 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44497⟩⟩) (.sum [.result 135833 .summary, .result 135655 .summary])

def exact135840RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨42908⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact135840RawTermsValid :
    exact135840RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135840 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44497⟩⟩) exact135840RawTerms .large 135836 (.finite 32193718473625891320532869316608) (some (135839))

def event135841 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41196⟩⟩) 0 ⟨40053⟩ 6164

def event135842 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41196⟩⟩) (.authority (.programFamilyFact))

def event135843 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41196⟩⟩) (.finite 3720)

def event135844 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41198⟩⟩) 0 ⟨7177⟩ 15500

def event135845 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41198⟩⟩) 1 ⟨41196⟩ 135843

def event135846 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41198⟩⟩) (.authority (.operator))

def exact135847RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41198⟩⟩]⟩, (1)⟩]

theorem exact135847RawTermsValid :
    exact135847RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135847 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41198⟩⟩) exact135847RawTerms .large 135846 .exactZero (none)

def event135848 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41814⟩⟩) 0 ⟨41198⟩ 135847

def event135849 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41814⟩⟩) (.authority (.operator))

def exact135850RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41814⟩⟩]⟩, (1)⟩]

theorem exact135850RawTermsValid :
    exact135850RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135850 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41814⟩⟩) exact135850RawTerms (.finite 8192) 135849 .exactZero (none)

def event135851 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41066⟩⟩) 0 ⟨39628⟩ 6158

def event135852 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41066⟩⟩) (.authority (.programFamilyFact))

def event135853 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41066⟩⟩) (.finite 3720)

def event135854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41067⟩⟩) 0 ⟨7177⟩ 15500

def event135855 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41067⟩⟩) 1 ⟨41066⟩ 135853

def event135856 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41067⟩⟩) (.authority (.operator))

def exact135857RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41067⟩⟩]⟩, (1)⟩]

theorem exact135857RawTermsValid :
    exact135857RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135857 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41067⟩⟩) exact135857RawTerms .large 135856 .exactZero (none)

def event135858 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41542⟩⟩) 0 ⟨41067⟩ 135857

def event135859 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41542⟩⟩) (.authority (.operator))

def exact135860RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41542⟩⟩]⟩, (1)⟩]

theorem exact135860RawTermsValid :
    exact135860RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135860 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41542⟩⟩) exact135860RawTerms (.finite 8192) 135859 .exactZero (none)

def event135861 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39629⟩⟩) 0 ⟨39626⟩ 6147

def event135862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39629⟩⟩) 1 ⟨6919⟩ 134403

def event135863 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39629⟩⟩) (.tensor (.predecessor 0 135861 .coefficient) (.predecessor 1 135862 .coefficient) true false)

def event135864 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39629⟩⟩, .operator (⟨6147, 0⟩, ⟨134403, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact135865RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact135865RawTermsValid :
    exact135865RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135865 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39629⟩⟩) exact135865RawTerms .large 135863 .exactZero (none)

def event135866 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7790⟩⟩) 0 ⟨5471⟩ 134273

def event135867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7790⟩⟩) 1 ⟨7282⟩ 18583

def event135868 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7790⟩⟩) (.product (.predecessor 0 135866 .coefficient) (.predecessor 1 135867 .coefficient) (⟨false, false, none, none, none⟩))

def event135869 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7790⟩⟩, .operator (⟨134273, 0⟩, ⟨18583, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩)

def exact135870RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩]

theorem exact135870RawTermsValid :
    exact135870RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135870 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7790⟩⟩) exact135870RawTerms .large 135868 .exactZero (none)

def event135871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39630⟩⟩) 0 ⟨7790⟩ 135870

def event135872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39630⟩⟩) 1 ⟨39629⟩ 135865

def event135873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39630⟩⟩) (.sum [.predecessor 0 135871 .coefficient, .predecessor 1 135872 .coefficient])

def exact135874RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact135874RawTermsValid :
    exact135874RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135874 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39630⟩⟩) exact135874RawTerms .large 135873 .exactZero (none)

def event135875 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39631⟩⟩) 0 ⟨39630⟩ 135874

def event135876 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39631⟩⟩) 1 ⟨108⟩ 18575

def event135877 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39631⟩⟩) (.sum [.predecessor 0 135875 .coefficient, .predecessor 1 135876 .coefficient])

def event135878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39631⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨108⟩⟩]⟩) [⟨.result 18575 .coefficient, false, none⟩])

def event135879 : Event := .survivorFold (1) 135878

def exact135880RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact135880RawTermsValid :
    exact135880RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135880 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39631⟩⟩) exact135880RawTerms .large 135877 (.finite 26) (some (135878))

def event135881 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39632⟩⟩) 0 ⟨39631⟩ 135880

def event135882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39632⟩⟩) 1 ⟨14076⟩ 6150

def event135883 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39632⟩⟩) (.product (.predecessor 0 135881 .coefficient) (.predecessor 1 135882 .coefficient) (⟨false, true, none, none, some 1⟩))

def event135884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39632⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14076⟩⟩], []⟩) [⟨.result 6150 .coefficient, true, some 1⟩])

def event135885 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39632⟩⟩) (.product (.result 135880 .summary) (.transfer 135884) (⟨false, false, none, none, none⟩))

def event135886 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39632⟩⟩, .operator (⟨135880, 1⟩, ⟨6150, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event135887 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39632⟩⟩, .operator (⟨135880, 0⟩, ⟨6150, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩)

def exact135888RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact135888RawTermsValid :
    exact135888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135888 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39632⟩⟩) exact135888RawTerms .large 135883 (.finite 39190528) (some (135885))

def event135889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14077⟩⟩) 0 ⟨14076⟩ 6150

def event135890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14077⟩⟩) 1 ⟨6919⟩ 134403

def event135891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14077⟩⟩) (.tensor (.predecessor 0 135889 .coefficient) (.predecessor 1 135890 .coefficient) true false)

def event135892 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14077⟩⟩, .operator (⟨6150, 0⟩, ⟨134403, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact135893RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact135893RawTermsValid :
    exact135893RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135893 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14077⟩⟩) exact135893RawTerms .large 135891 .exactZero (none)

def event135894 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7807⟩⟩) 0 ⟨5471⟩ 134273

def event135895 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7807⟩⟩) 1 ⟨7299⟩ 18624

def event135896 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7807⟩⟩) (.product (.predecessor 0 135894 .coefficient) (.predecessor 1 135895 .coefficient) (⟨false, false, none, none, none⟩))

def event135897 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7807⟩⟩, .operator (⟨134273, 0⟩, ⟨18624, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩)

def exact135898RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩]

theorem exact135898RawTermsValid :
    exact135898RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135898 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7807⟩⟩) exact135898RawTerms .large 135896 .exactZero (none)

def event135899 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14078⟩⟩) 0 ⟨7807⟩ 135898

def event135900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14078⟩⟩) 1 ⟨14077⟩ 135893

def event135901 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14078⟩⟩) (.sum [.predecessor 0 135899 .coefficient, .predecessor 1 135900 .coefficient])

def exact135902RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact135902RawTermsValid :
    exact135902RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135902 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14078⟩⟩) exact135902RawTerms .large 135901 .exactZero (none)

def event135903 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14079⟩⟩) 0 ⟨14078⟩ 135902

def event135904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14079⟩⟩) 1 ⟨125⟩ 18616

def event135905 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14079⟩⟩) (.sum [.predecessor 0 135903 .coefficient, .predecessor 1 135904 .coefficient])

def event135906 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14079⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨125⟩⟩]⟩) [⟨.result 18616 .coefficient, false, none⟩])

def event135907 : Event := .survivorFold (1) 135906

def exact135908RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact135908RawTermsValid :
    exact135908RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135908 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14079⟩⟩) exact135908RawTerms .large 135905 (.finite 26) (some (135906))

def event135909 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14080⟩⟩) 0 ⟨14079⟩ 135908

def event135910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14080⟩⟩) 1 ⟨9557⟩ 18613

def event135911 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14080⟩⟩) (.product (.predecessor 0 135909 .coefficient) (.predecessor 1 135910 .coefficient) (⟨false, false, none, none, none⟩))

def event135912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14080⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) [⟨.result 18609 .coefficient, false, none⟩])

def event135913 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14080⟩⟩) (.product (.result 135908 .summary) (.transfer 135912) (⟨false, false, none, none, none⟩))

def event135914 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14080⟩⟩, .operator (⟨135908, 1⟩, ⟨18613, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (-1)⟩)

def event135915 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨14080⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9556⟩⟩) ⟨7282⟩ 18583)

def event135916 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14080⟩⟩, .relation 135915 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (-1)⟩)

def event135917 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14080⟩⟩, .operator (⟨135908, 0⟩, ⟨18613, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩)

def exact135918RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (-1)⟩]

theorem exact135918RawTermsValid :
    exact135918RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135918 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14080⟩⟩) exact135918RawTerms .large 135911 (.finite 279172874240) (some (135913))

def event135919 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39633⟩⟩) 0 ⟨14080⟩ 135918

def event135920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39633⟩⟩) 1 ⟨39632⟩ 135888

def event135921 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39633⟩⟩) (.sum [.predecessor 0 135919 .coefficient, .predecessor 1 135920 .coefficient])

def event135922 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39633⟩⟩, .operator (⟨135918, 1⟩, ⟨135888, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩)

def event135923 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39633⟩⟩) (.sum [.result 135918 .summary, .result 135888 .summary])

def exact135924RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact135924RawTermsValid :
    exact135924RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135924 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39633⟩⟩) exact135924RawTerms .large 135921 (.finite 279212064768) (some (135923))

def event135925 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41543⟩⟩) 0 ⟨39633⟩ 135924

def event135926 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41543⟩⟩) 1 ⟨41542⟩ 135860

def event135927 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41543⟩⟩) (.product (.predecessor 0 135925 .coefficient) (.predecessor 1 135926 .coefficient) (⟨false, false, none, none, none⟩))

def event135928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41543⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨41542⟩⟩]⟩) [⟨.result 135860 .coefficient, false, none⟩])

def event135929 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41543⟩⟩) (.product (.result 135924 .summary) (.transfer 135928) (⟨false, false, none, none, none⟩))

def event135930 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41543⟩⟩, .operator (⟨135924, 1⟩, ⟨135860, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41542⟩⟩]⟩, (-1)⟩)

def event135931 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41543⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41542⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41542⟩⟩) ⟨41067⟩ 135857)

def event135932 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41543⟩⟩, .relation 135931 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], [⟨.program ⟨257⟩, ⟨41067⟩⟩]⟩, (-1)⟩)

def event135933 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41543⟩⟩, .operator (⟨135924, 0⟩, ⟨135860, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41542⟩⟩]⟩, (1)⟩)

def exact135934RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41542⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14076⟩⟩, ⟨.program ⟨257⟩, ⟨39626⟩⟩], [⟨.program ⟨257⟩, ⟨41067⟩⟩]⟩, (-1)⟩]

theorem exact135934RawTermsValid :
    exact135934RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event135934 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41543⟩⟩) exact135934RawTerms .large 135927 (.finite 2998016717067984568320) (some (135929))

def event135935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40479⟩⟩) 0 ⟨39628⟩ 6158

def eventLeaf8480 : Array AnnotatedEvent := #[
  { event := event135680
    frameStart := 135669 },
  { event := event135681
    frameStart := 135669 },
  { event := event135682
    frameStart := 135669 },
  { event := event135683
    frameStart := 135669 },
  { event := event135684
    frameStart := 135669 },
  { event := event135685
    frameStart := 135669 },
  { event := event135686
    frameStart := 135669 },
  { event := event135687
    frameStart := 135669 },
  { event := event135688
    frameStart := 135669 },
  { event := event135689
    frameStart := 135669 },
  { event := event135690
    frameStart := 135669 },
  { event := event135691
    frameStart := 135669 },
  { event := event135692
    frameStart := 135669 },
  { event := event135693
    frameStart := 135669 },
  { event := event135694
    frameStart := 135669 },
  { event := event135695
    frameStart := 135669 }
]

def eventLeaf8481 : Array AnnotatedEvent := #[
  { event := event135696
    frameStart := 135669 },
  { event := event135697
    frameStart := 135669 },
  { event := event135698
    frameStart := 135669 },
  { event := event135699
    frameStart := 135669 },
  { event := event135700
    frameStart := 135669 },
  { event := event135701
    frameStart := 135669 },
  { event := event135702
    frameStart := 135669 },
  { event := event135703
    frameStart := 135669 },
  { event := event135704
    frameStart := 135669 },
  { event := event135705
    frameStart := 135669 },
  { event := event135706
    frameStart := 135669 },
  { event := event135707
    frameStart := 135669 },
  { event := event135708
    frameStart := 135669 },
  { event := event135709
    frameStart := 135669 },
  { event := event135710
    frameStart := 135669 },
  { event := event135711
    frameStart := 135669 }
]

def eventLeaf8482 : Array AnnotatedEvent := #[
  { event := event135712
    frameStart := 135669 },
  { event := event135713
    frameStart := 135669 },
  { event := event135714
    frameStart := 135669 },
  { event := event135715
    frameStart := 135669 },
  { event := event135716
    frameStart := 135669 },
  { event := event135717
    frameStart := 135669 },
  { event := event135718
    frameStart := 135669 },
  { event := event135719
    frameStart := 135669 },
  { event := event135720
    frameStart := 135669 },
  { event := event135721
    frameStart := 135669 },
  { event := event135722
    frameStart := 135669 },
  { event := event135723
    frameStart := 135723 },
  { event := event135724
    frameStart := 135723 },
  { event := event135725
    frameStart := 135723 },
  { event := event135726
    frameStart := 135723 },
  { event := event135727
    frameStart := 135723 }
]

def eventLeaf8483 : Array AnnotatedEvent := #[
  { event := event135728
    frameStart := 135723 },
  { event := event135729
    frameStart := 135723 },
  { event := event135730
    frameStart := 135723 },
  { event := event135731
    frameStart := 135723 },
  { event := event135732
    frameStart := 135723 },
  { event := event135733
    frameStart := 135723 },
  { event := event135734
    frameStart := 135723 },
  { event := event135735
    frameStart := 135723 },
  { event := event135736
    frameStart := 135723 },
  { event := event135737
    frameStart := 135723 },
  { event := event135738
    frameStart := 135723 },
  { event := event135739
    frameStart := 135723 },
  { event := event135740
    frameStart := 135723 },
  { event := event135741
    frameStart := 135723 },
  { event := event135742
    frameStart := 135723 },
  { event := event135743
    frameStart := 135723 }
]

def eventLeaf8484 : Array AnnotatedEvent := #[
  { event := event135744
    frameStart := 135723 },
  { event := event135745
    frameStart := 135723 },
  { event := event135746
    frameStart := 135723 },
  { event := event135747
    frameStart := 135723 },
  { event := event135748
    frameStart := 135723 },
  { event := event135749
    frameStart := 135723 },
  { event := event135750
    frameStart := 135723 },
  { event := event135751
    frameStart := 135723 },
  { event := event135752
    frameStart := 135723 },
  { event := event135753
    frameStart := 135723 },
  { event := event135754
    frameStart := 135723 },
  { event := event135755
    frameStart := 135723 },
  { event := event135756
    frameStart := 135723 },
  { event := event135757
    frameStart := 135723 },
  { event := event135758
    frameStart := 135723 },
  { event := event135759
    frameStart := 135723 }
]

def eventLeaf8485 : Array AnnotatedEvent := #[
  { event := event135760
    frameStart := 135723 },
  { event := event135761
    frameStart := 135723 },
  { event := event135762
    frameStart := 135723 },
  { event := event135763
    frameStart := 135723 },
  { event := event135764
    frameStart := 135723 },
  { event := event135765
    frameStart := 135723 },
  { event := event135766
    frameStart := 135723 },
  { event := event135767
    frameStart := 135723 },
  { event := event135768
    frameStart := 135723 },
  { event := event135769
    frameStart := 135723 },
  { event := event135770
    frameStart := 135723 },
  { event := event135771
    frameStart := 135723 },
  { event := event135772
    frameStart := 135723 },
  { event := event135773
    frameStart := 135723 },
  { event := event135774
    frameStart := 135723 },
  { event := event135775
    frameStart := 135723 }
]

def eventLeaf8486 : Array AnnotatedEvent := #[
  { event := event135776
    frameStart := 135723 },
  { event := event135777
    frameStart := 135723 },
  { event := event135778
    frameStart := 135723 },
  { event := event135779
    frameStart := 135723 },
  { event := event135780
    frameStart := 135723 },
  { event := event135781
    frameStart := 135723 },
  { event := event135782
    frameStart := 135723 },
  { event := event135783
    frameStart := 135723 },
  { event := event135784
    frameStart := 135723 },
  { event := event135785
    frameStart := 135723 },
  { event := event135786
    frameStart := 135723 },
  { event := event135787
    frameStart := 135723 },
  { event := event135788
    frameStart := 135723 },
  { event := event135789
    frameStart := 135723 },
  { event := event135790
    frameStart := 135723 },
  { event := event135791
    frameStart := 135723 }
]

def eventLeaf8487 : Array AnnotatedEvent := #[
  { event := event135792
    frameStart := 135723 },
  { event := event135793
    frameStart := 135723 },
  { event := event135794
    frameStart := 135723 },
  { event := event135795
    frameStart := 135723 },
  { event := event135796
    frameStart := 135723 },
  { event := event135797
    frameStart := 135723 },
  { event := event135798
    frameStart := 135723 },
  { event := event135799
    frameStart := 135723 },
  { event := event135800
    frameStart := 135723 },
  { event := event135801
    frameStart := 135723 },
  { event := event135802
    frameStart := 135723 },
  { event := event135803
    frameStart := 135723 },
  { event := event135804
    frameStart := 135723 },
  { event := event135805
    frameStart := 135723 },
  { event := event135806
    frameStart := 135723 },
  { event := event135807
    frameStart := 135723 }
]

def eventLeaf8488 : Array AnnotatedEvent := #[
  { event := event135808
    frameStart := 135723 },
  { event := event135809
    frameStart := 135723 },
  { event := event135810
    frameStart := 135723 },
  { event := event135811
    frameStart := 135723 },
  { event := event135812
    frameStart := 135723 },
  { event := event135813
    frameStart := 135723 },
  { event := event135814
    frameStart := 135723 },
  { event := event135815
    frameStart := 135723 },
  { event := event135816
    frameStart := 135723 },
  { event := event135817
    frameStart := 135723 },
  { event := event135818
    frameStart := 135723 },
  { event := event135819
    frameStart := 135723 },
  { event := event135820
    frameStart := 135723 },
  { event := event135821
    frameStart := 135723 },
  { event := event135822
    frameStart := 135723 },
  { event := event135823
    frameStart := 135723 }
]

def eventLeaf8489 : Array AnnotatedEvent := #[
  { event := event135824
    frameStart := 135723 },
  { event := event135825
    frameStart := 135723 },
  { event := event135826
    frameStart := 135723 },
  { event := event135827
    frameStart := 0 },
  { event := event135828
    frameStart := 0 },
  { event := event135829
    frameStart := 0 },
  { event := event135830
    frameStart := 0 },
  { event := event135831
    frameStart := 0 },
  { event := event135832
    frameStart := 0 },
  { event := event135833
    frameStart := 0 },
  { event := event135834
    frameStart := 0 },
  { event := event135835
    frameStart := 0 },
  { event := event135836
    frameStart := 0 },
  { event := event135837
    frameStart := 0 },
  { event := event135838
    frameStart := 0 },
  { event := event135839
    frameStart := 0 }
]

def eventLeaf8490 : Array AnnotatedEvent := #[
  { event := event135840
    frameStart := 0 },
  { event := event135841
    frameStart := 0 },
  { event := event135842
    frameStart := 0 },
  { event := event135843
    frameStart := 0 },
  { event := event135844
    frameStart := 0 },
  { event := event135845
    frameStart := 0 },
  { event := event135846
    frameStart := 0 },
  { event := event135847
    frameStart := 0 },
  { event := event135848
    frameStart := 0 },
  { event := event135849
    frameStart := 0 },
  { event := event135850
    frameStart := 0 },
  { event := event135851
    frameStart := 0 },
  { event := event135852
    frameStart := 0 },
  { event := event135853
    frameStart := 0 },
  { event := event135854
    frameStart := 0 },
  { event := event135855
    frameStart := 0 }
]

def eventLeaf8491 : Array AnnotatedEvent := #[
  { event := event135856
    frameStart := 0 },
  { event := event135857
    frameStart := 0 },
  { event := event135858
    frameStart := 0 },
  { event := event135859
    frameStart := 0 },
  { event := event135860
    frameStart := 0 },
  { event := event135861
    frameStart := 0 },
  { event := event135862
    frameStart := 0 },
  { event := event135863
    frameStart := 0 },
  { event := event135864
    frameStart := 0 },
  { event := event135865
    frameStart := 0 },
  { event := event135866
    frameStart := 0 },
  { event := event135867
    frameStart := 0 },
  { event := event135868
    frameStart := 0 },
  { event := event135869
    frameStart := 0 },
  { event := event135870
    frameStart := 0 },
  { event := event135871
    frameStart := 0 }
]

def eventLeaf8492 : Array AnnotatedEvent := #[
  { event := event135872
    frameStart := 0 },
  { event := event135873
    frameStart := 0 },
  { event := event135874
    frameStart := 0 },
  { event := event135875
    frameStart := 0 },
  { event := event135876
    frameStart := 0 },
  { event := event135877
    frameStart := 0 },
  { event := event135878
    frameStart := 0 },
  { event := event135879
    frameStart := 0 },
  { event := event135880
    frameStart := 0 },
  { event := event135881
    frameStart := 0 },
  { event := event135882
    frameStart := 0 },
  { event := event135883
    frameStart := 0 },
  { event := event135884
    frameStart := 0 },
  { event := event135885
    frameStart := 0 },
  { event := event135886
    frameStart := 0 },
  { event := event135887
    frameStart := 0 }
]

def eventLeaf8493 : Array AnnotatedEvent := #[
  { event := event135888
    frameStart := 0 },
  { event := event135889
    frameStart := 0 },
  { event := event135890
    frameStart := 0 },
  { event := event135891
    frameStart := 0 },
  { event := event135892
    frameStart := 0 },
  { event := event135893
    frameStart := 0 },
  { event := event135894
    frameStart := 0 },
  { event := event135895
    frameStart := 0 },
  { event := event135896
    frameStart := 0 },
  { event := event135897
    frameStart := 0 },
  { event := event135898
    frameStart := 0 },
  { event := event135899
    frameStart := 0 },
  { event := event135900
    frameStart := 0 },
  { event := event135901
    frameStart := 0 },
  { event := event135902
    frameStart := 0 },
  { event := event135903
    frameStart := 0 }
]

def eventLeaf8494 : Array AnnotatedEvent := #[
  { event := event135904
    frameStart := 0 },
  { event := event135905
    frameStart := 0 },
  { event := event135906
    frameStart := 0 },
  { event := event135907
    frameStart := 0 },
  { event := event135908
    frameStart := 0 },
  { event := event135909
    frameStart := 0 },
  { event := event135910
    frameStart := 0 },
  { event := event135911
    frameStart := 0 },
  { event := event135912
    frameStart := 0 },
  { event := event135913
    frameStart := 0 },
  { event := event135914
    frameStart := 0 },
  { event := event135915
    frameStart := 0 },
  { event := event135916
    frameStart := 0 },
  { event := event135917
    frameStart := 0 },
  { event := event135918
    frameStart := 0 },
  { event := event135919
    frameStart := 0 }
]

def eventLeaf8495 : Array AnnotatedEvent := #[
  { event := event135920
    frameStart := 0 },
  { event := event135921
    frameStart := 0 },
  { event := event135922
    frameStart := 0 },
  { event := event135923
    frameStart := 0 },
  { event := event135924
    frameStart := 0 },
  { event := event135925
    frameStart := 0 },
  { event := event135926
    frameStart := 0 },
  { event := event135927
    frameStart := 0 },
  { event := event135928
    frameStart := 0 },
  { event := event135929
    frameStart := 0 },
  { event := event135930
    frameStart := 0 },
  { event := event135931
    frameStart := 0 },
  { event := event135932
    frameStart := 0 },
  { event := event135933
    frameStart := 0 },
  { event := event135934
    frameStart := 0 },
  { event := event135935
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events530
