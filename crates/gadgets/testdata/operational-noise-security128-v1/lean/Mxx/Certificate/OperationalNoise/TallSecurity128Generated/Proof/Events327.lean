import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events327

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event83712 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19212⟩⟩) (.product (.result 75995 .summary) (.transfer 83711) (⟨false, false, none, none, none⟩))

def event83713 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19212⟩⟩, .operator (⟨75995, 0⟩, ⟨83707, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19209⟩⟩]⟩, (1)⟩)

def event83714 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨19210⟩⟩)

def event83715 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event83716 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event83717 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event83718 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event83719 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event83720 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event83721 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event83722 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event83723 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 83722

def event83724 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 83720

def event83725 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 83723 .coefficient) (.value (.predecessor 1 83724 .coefficient)))

def event83726 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event83727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 83726

def event83728 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 83718

def event83729 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 83727 .coefficient, .predecessor 1 83728 .coefficient])

def event83730 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event83731 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 83730

def event83732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 83716

def event83733 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 83732 .coefficient))

def event83734 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event83735 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18418⟩⟩) 0 ⟨10325⟩ 83734

def event83736 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18418⟩⟩) (.authority (.programFamilyFact))

def exact83737RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18418⟩⟩], []⟩, (1)⟩]

theorem exact83737RawTermsValid :
    exact83737RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83737 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18418⟩⟩) exact83737RawTerms (.finite 3) 83736 .exactZero (none)

def event83738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12771⟩⟩) 0 ⟨10325⟩ 83734

def event83739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12771⟩⟩) (.authority (.programFamilyFact))

def exact83740RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩], []⟩, (1)⟩]

theorem exact83740RawTermsValid :
    exact83740RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83740 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12771⟩⟩) exact83740RawTerms (.finite 3) 83739 .exactZero (none)

def event83741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18419⟩⟩) 0 ⟨12771⟩ 83740

def event83742 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18419⟩⟩) 1 ⟨18418⟩ 83737

def event83743 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18419⟩⟩) (.product (.predecessor 0 83741 .coefficient) (.predecessor 1 83742 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event83744 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18419⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], []⟩) [⟨.result 83740 .coefficient, true, some 1⟩, ⟨.result 83737 .coefficient, true, some 1⟩])

def event83745 : Event := .survivorFold (1) 83744

def exact83746RawTerms : List Term := []

theorem exact83746RawTermsValid :
    exact83746RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83746 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18419⟩⟩) exact83746RawTerms (.finite 9) 83743 (.finite 9) (some (83744))

def event83747 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18420⟩⟩) 0 ⟨18419⟩ 83746

def event83748 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18420⟩⟩) (.identity (.predecessor 0 83747 .coefficient))

def event83749 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18420⟩⟩) (.finite 9)

def event83750 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19209⟩⟩) 0 ⟨18420⟩ 83749

def event83751 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19209⟩⟩) (.authority (.relationPreimageSource ⟨37⟩))

def exact83752RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19209⟩⟩]⟩, (1)⟩]

theorem exact83752RawTermsValid :
    exact83752RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83752 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19209⟩⟩) exact83752RawTerms (.finite 5647228698) 83751 .exactZero (none)

def event83753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact83754RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact83754RawTermsValid :
    exact83754RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83754 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact83754RawTerms .large 83753 .exactZero (none)

def event83755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19210⟩⟩) 0 ⟨35⟩ 83754

def event83756 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19210⟩⟩) 1 ⟨19209⟩ 83752

def event83757 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19210⟩⟩) (.product (.predecessor 0 83755 .coefficient) (.predecessor 1 83756 .coefficient) (⟨false, false, none, none, none⟩))

def event83758 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19210⟩⟩, .operator (⟨83754, 0⟩, ⟨83752, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19209⟩⟩]⟩, (1)⟩)

def exact83759RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19209⟩⟩]⟩, (1)⟩]

theorem exact83759RawTermsValid :
    exact83759RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83759 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19210⟩⟩) exact83759RawTerms .large 83757 .exactZero (none)

def event83760 : Event := .preFoldPolynomial 83759 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19209⟩⟩]⟩, (1)⟩] .exactZero none

def exact83761RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19209⟩⟩]⟩, (1)⟩]

def event83761 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨19210⟩⟩) 83760 exact83761RawTerms .large 83757 .exactZero (none)

def event83762 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨20289⟩⟩)

def event83763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event83764 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event83765 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event83766 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event83767 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event83768 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event83769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event83770 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event83771 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 83770

def event83772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 83768

def event83773 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 83771 .coefficient) (.value (.predecessor 1 83772 .coefficient)))

def event83774 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event83775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 83774

def event83776 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 83766

def event83777 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 83775 .coefficient, .predecessor 1 83776 .coefficient])

def event83778 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event83779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 83778

def event83780 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 83764

def event83781 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 83780 .coefficient))

def event83782 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event83783 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18418⟩⟩) 0 ⟨10325⟩ 83782

def event83784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18418⟩⟩) (.authority (.programFamilyFact))

def exact83785RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18418⟩⟩], []⟩, (1)⟩]

theorem exact83785RawTermsValid :
    exact83785RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83785 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18418⟩⟩) exact83785RawTerms (.finite 3) 83784 .exactZero (none)

def event83786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12771⟩⟩) 0 ⟨10325⟩ 83782

def event83787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12771⟩⟩) (.authority (.programFamilyFact))

def exact83788RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩], []⟩, (1)⟩]

theorem exact83788RawTermsValid :
    exact83788RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83788 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12771⟩⟩) exact83788RawTerms (.finite 3) 83787 .exactZero (none)

def event83789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18419⟩⟩) 0 ⟨12771⟩ 83788

def event83790 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18419⟩⟩) 1 ⟨18418⟩ 83785

def event83791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18419⟩⟩) (.product (.predecessor 0 83789 .coefficient) (.predecessor 1 83790 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event83792 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18419⟩⟩, .operator (⟨83788, 0⟩, ⟨83785, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], []⟩, (1)⟩)

def exact83793RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], []⟩, (1)⟩]

theorem exact83793RawTermsValid :
    exact83793RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83793 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18419⟩⟩) exact83793RawTerms (.finite 9) 83791 .exactZero (none)

def event83794 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18420⟩⟩) 0 ⟨18419⟩ 83793

def event83795 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18420⟩⟩) (.identity (.predecessor 0 83794 .coefficient))

def event83796 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18420⟩⟩) (.finite 9)

def event83797 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19744⟩⟩) 0 ⟨18420⟩ 83796

def event83798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19744⟩⟩) (.authority (.programFamilyFact))

def event83799 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨19744⟩⟩) (.finite 3720)

def event83800 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event83801 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19745⟩⟩) 0 ⟨7177⟩ 83800

def event83802 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19745⟩⟩) 1 ⟨19744⟩ 83799

def event83803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19745⟩⟩) (.authority (.operator))

def exact83804RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19745⟩⟩]⟩, (1)⟩]

theorem exact83804RawTermsValid :
    exact83804RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83804 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19745⟩⟩) exact83804RawTerms .large 83803 .exactZero (none)

def event83805 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20285⟩⟩) 0 ⟨19745⟩ 83804

def event83806 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20285⟩⟩) (.authority (.operator))

def exact83807RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨20285⟩⟩]⟩, (1)⟩]

theorem exact83807RawTermsValid :
    exact83807RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83807 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20285⟩⟩) exact83807RawTerms (.finite 8192) 83806 .exactZero (none)

def event83808 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event83809 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event83810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20010⟩⟩) 0 ⟨18420⟩ 83796

def event83811 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20010⟩⟩) 1 ⟨136⟩ 83809

def event83812 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20010⟩⟩) (.sum [.predecessor 0 83810 .coefficient, .predecessor 1 83811 .coefficient])

def event83813 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨20010⟩⟩) (.finite 9)

def event83814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20011⟩⟩) 0 ⟨20010⟩ 83813

def event83815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20011⟩⟩) (.identity (.predecessor 0 83814 .coefficient))

def exact83816RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], []⟩, (1)⟩]

theorem exact83816RawTermsValid :
    exact83816RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83816 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20011⟩⟩) exact83816RawTerms (.finite 9) 83815 .exactZero (none)

def event83817 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact83818RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact83818RawTermsValid :
    exact83818RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83818 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact83818RawTerms .large 83817 .exactZero (none)

def event83819 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20012⟩⟩) 0 ⟨6908⟩ 83818

def event83820 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20012⟩⟩) 1 ⟨20011⟩ 83816

def event83821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20012⟩⟩) (.product (.predecessor 0 83819 .coefficient) (.predecessor 1 83820 .coefficient) (⟨false, false, none, none, none⟩))

def event83822 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20012⟩⟩, .operator (⟨83818, 0⟩, ⟨83816, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact83823RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact83823RawTermsValid :
    exact83823RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83823 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20012⟩⟩) exact83823RawTerms .large 83821 .exactZero (none)

def event83824 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event83825 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event83826 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 83800

def event83827 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact83828RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact83828RawTermsValid :
    exact83828RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83828 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact83828RawTerms .large 83827 .exactZero (none)

def event83829 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7305⟩⟩) 0 ⟨7178⟩ 83828

def event83830 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7305⟩⟩) (.identity (.predecessor 0 83829 .coefficient))

def exact83831RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7305⟩⟩]⟩, (1)⟩]

theorem exact83831RawTermsValid :
    exact83831RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83831 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7305⟩⟩) exact83831RawTerms .large 83830 .exactZero (none)

def event83832 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9571⟩⟩) 0 ⟨7305⟩ 83831

def event83833 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9571⟩⟩) (.authority (.operator))

def exact83834RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩, (1)⟩]

theorem exact83834RawTermsValid :
    exact83834RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83834 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9571⟩⟩) exact83834RawTerms (.finite 8192) 83833 .exactZero (none)

def event83835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9572⟩⟩) 0 ⟨9571⟩ 83834

def event83836 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9572⟩⟩) 1 ⟨2370⟩ 83825

def event83837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9572⟩⟩) (.scale (.predecessor 0 83835 .coefficient) (.value (.predecessor 1 83836 .coefficient)))

def exact83838RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩, (1)⟩]

theorem exact83838RawTermsValid :
    exact83838RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83838 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9572⟩⟩) exact83838RawTerms (.finite 8192) 83837 .exactZero (none)

def event83839 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7277⟩⟩) 0 ⟨7178⟩ 83828

def event83840 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7277⟩⟩) (.identity (.predecessor 0 83839 .coefficient))

def exact83841RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩]⟩, (1)⟩]

theorem exact83841RawTermsValid :
    exact83841RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83841 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7277⟩⟩) exact83841RawTerms .large 83840 .exactZero (none)

def event83842 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9573⟩⟩) 0 ⟨7277⟩ 83841

def event83843 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9573⟩⟩) 1 ⟨9572⟩ 83838

def event83844 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9573⟩⟩) (.product (.predecessor 0 83842 .coefficient) (.predecessor 1 83843 .coefficient) (⟨false, false, none, none, none⟩))

def event83845 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9573⟩⟩, .operator (⟨83841, 0⟩, ⟨83838, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩, (1)⟩)

def exact83846RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩, (1)⟩]

theorem exact83846RawTermsValid :
    exact83846RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83846 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9573⟩⟩) exact83846RawTerms .large 83844 .exactZero (none)

def event83847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20013⟩⟩) 0 ⟨9573⟩ 83846

def event83848 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20013⟩⟩) 1 ⟨20012⟩ 83823

def event83849 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20013⟩⟩) (.sum [.predecessor 0 83847 .coefficient, .predecessor 1 83848 .coefficient])

def exact83850RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83850RawTermsValid :
    exact83850RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83850 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20013⟩⟩) exact83850RawTerms .large 83849 .exactZero (none)

def event83851 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20288⟩⟩) 0 ⟨20013⟩ 83850

def event83852 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20288⟩⟩) 1 ⟨20285⟩ 83807

def event83853 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20288⟩⟩) (.product (.predecessor 0 83851 .coefficient) (.predecessor 1 83852 .coefficient) (⟨false, false, none, none, none⟩))

def event83854 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20288⟩⟩, .operator (⟨83850, 0⟩, ⟨83807, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20285⟩⟩]⟩, (1)⟩)

def event83855 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20288⟩⟩, .operator (⟨83850, 1⟩, ⟨83807, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20285⟩⟩]⟩, (-1)⟩)

def event83856 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨20288⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20285⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20285⟩⟩) ⟨19745⟩ 83804)

def event83857 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20288⟩⟩, .relation 83856 0, ⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], [⟨.program ⟨257⟩, ⟨19745⟩⟩]⟩, (-1)⟩)

def exact83858RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], [⟨.program ⟨257⟩, ⟨19745⟩⟩]⟩, (-1)⟩]

theorem exact83858RawTermsValid :
    exact83858RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83858 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20288⟩⟩) exact83858RawTerms .large 83853 .exactZero (none)

def event83859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18636⟩⟩) 0 ⟨18420⟩ 83796

def event83860 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18636⟩⟩) (.authority (.programFamilyFact))

def exact83861RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], []⟩, (1)⟩]

theorem exact83861RawTermsValid :
    exact83861RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83861 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18636⟩⟩) exact83861RawTerms (.finite 3) 83860 .exactZero (none)

def event83862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18638⟩⟩) 0 ⟨6908⟩ 83818

def event83863 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18638⟩⟩) 1 ⟨18636⟩ 83861

def event83864 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18638⟩⟩) (.product (.predecessor 0 83862 .coefficient) (.predecessor 1 83863 .coefficient) (⟨false, true, none, none, some 1⟩))

def event83865 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18638⟩⟩, .operator (⟨83818, 0⟩, ⟨83861, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact83866RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact83866RawTermsValid :
    exact83866RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83866 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18638⟩⟩) exact83866RawTerms .large 83864 .exactZero (none)

def event83867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7180⟩⟩) 0 ⟨7177⟩ 83800

def event83868 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7180⟩⟩) (.authority (.operator))

def exact83869RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩, (1)⟩]

theorem exact83869RawTermsValid :
    exact83869RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83869 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7180⟩⟩) exact83869RawTerms .large 83868 .exactZero (none)

def event83870 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18639⟩⟩) 0 ⟨7180⟩ 83869

def event83871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18639⟩⟩) 1 ⟨18638⟩ 83866

def event83872 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18639⟩⟩) (.sum [.predecessor 0 83870 .coefficient, .predecessor 1 83871 .coefficient])

def exact83873RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83873RawTermsValid :
    exact83873RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83873 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18639⟩⟩) exact83873RawTerms .large 83872 .exactZero (none)

def event83874 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20289⟩⟩) 0 ⟨18639⟩ 83873

def event83875 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20289⟩⟩) 1 ⟨20288⟩ 83858

def event83876 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20289⟩⟩) (.sum [.predecessor 0 83874 .coefficient, .predecessor 1 83875 .coefficient])

def exact83877RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20285⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], [⟨.program ⟨257⟩, ⟨19745⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83877RawTermsValid :
    exact83877RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83877 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20289⟩⟩) exact83877RawTerms .large 83876 .exactZero (none)

def event83878 : Event := .preFoldPolynomial 83877 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20285⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], [⟨.program ⟨257⟩, ⟨19745⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact83879RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20285⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], [⟨.program ⟨257⟩, ⟨19745⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event83879 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨20289⟩⟩) 83878 exact83879RawTerms .large 83876 .exactZero (none)

def event83880 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨18420⟩⟩) ⟨⟨59⟩, ⟨37⟩, ⟨135⟩⟩ ⟨83714, 83880⟩

def event83881 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨19212⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19209⟩⟩]⟩) (1) 0 2 (.universal 83880 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19209⟩⟩]⟩) (none) 83879)

def event83882 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19212⟩⟩, .relation 83881 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩, (1)⟩)

def event83883 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19212⟩⟩, .relation 83881 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20285⟩⟩]⟩, (-1)⟩)

def event83884 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19212⟩⟩, .relation 83881 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], [⟨.program ⟨257⟩, ⟨19745⟩⟩]⟩, (1)⟩)

def event83885 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19212⟩⟩, .relation 83881 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact83886RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20285⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], [⟨.program ⟨257⟩, ⟨19745⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83886RawTermsValid :
    exact83886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83886 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19212⟩⟩) exact83886RawTerms .large 83710 (.finite 202072841853861888) (some (83712))

def event83887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20287⟩⟩) 0 ⟨19212⟩ 83886

def event83888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20287⟩⟩) 1 ⟨20286⟩ 83700

def event83889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20287⟩⟩) (.sum [.predecessor 0 83887 .coefficient, .predecessor 1 83888 .coefficient])

def event83890 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20287⟩⟩, .operator (⟨83886, 2⟩, ⟨83700, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], [⟨.program ⟨257⟩, ⟨19745⟩⟩]⟩, (-1)⟩)

def event83891 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20287⟩⟩, .operator (⟨83886, 1⟩, ⟨83700, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20285⟩⟩]⟩, (1)⟩)

def event83892 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20287⟩⟩) (.sum [.result 83886 .summary, .result 83700 .summary])

def exact83893RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83893RawTermsValid :
    exact83893RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83893 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20287⟩⟩) exact83893RawTerms .large 83889 (.finite 2997825428629885288448) (some (83892))

def event83894 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20840⟩⟩) 0 ⟨20287⟩ 83893

def event83895 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20840⟩⟩) 1 ⟨20838⟩ 83616

def event83896 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20840⟩⟩) (.product (.predecessor 0 83894 .coefficient) (.predecessor 1 83895 .coefficient) (⟨false, false, none, none, none⟩))

def event83897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20840⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩) [⟨.result 83616 .coefficient, false, none⟩])

def event83898 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20840⟩⟩) (.product (.result 83893 .summary) (.transfer 83897) (⟨false, false, none, none, none⟩))

def event83899 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20840⟩⟩, .operator (⟨83893, 0⟩, ⟨83616, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩, (1)⟩)

def event83900 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20840⟩⟩, .operator (⟨83893, 1⟩, ⟨83616, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩, (-1)⟩)

def event83901 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨20840⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20838⟩⟩) ⟨19915⟩ 83613)

def event83902 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20840⟩⟩, .relation 83901 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨19915⟩⟩]⟩, (-1)⟩)

def exact83903RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨19915⟩⟩]⟩, (-1)⟩]

theorem exact83903RawTermsValid :
    exact83903RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83903 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20840⟩⟩) exact83903RawTerms .large 83896 (.finite 32188905437706348505289216491520) (some (83898))

def event83904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19576⟩⟩) 0 ⟨18637⟩ 3471

def event83905 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19576⟩⟩) (.authority (.relationPreimageSource ⟨59⟩))

def exact83906RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩, (1)⟩]

theorem exact83906RawTermsValid :
    exact83906RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83906 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19576⟩⟩) exact83906RawTerms (.finite 5647228698) 83905 .exactZero (none)

def event83907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19578⟩⟩) 0 ⟨19576⟩ 83906

def event83908 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19578⟩⟩) 1 ⟨2370⟩ 4

def event83909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19578⟩⟩) (.scale (.predecessor 0 83907 .coefficient) (.value (.predecessor 1 83908 .coefficient)))

def exact83910RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩, (1)⟩]

theorem exact83910RawTermsValid :
    exact83910RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83910 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19578⟩⟩) exact83910RawTerms (.finite 5647228698) 83909 .exactZero (none)

def event83911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19579⟩⟩) 0 ⟨10368⟩ 75995

def event83912 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19579⟩⟩) 1 ⟨19578⟩ 83910

def event83913 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19579⟩⟩) (.product (.predecessor 0 83911 .coefficient) (.predecessor 1 83912 .coefficient) (⟨false, false, none, none, none⟩))

def event83914 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19579⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩) [⟨.result 83906 .coefficient, false, none⟩])

def event83915 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19579⟩⟩) (.product (.result 75995 .summary) (.transfer 83914) (⟨false, false, none, none, none⟩))

def event83916 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19579⟩⟩, .operator (⟨75995, 0⟩, ⟨83910, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩, (1)⟩)

def event83917 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨19577⟩⟩)

def event83918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event83919 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event83920 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event83921 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event83922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event83923 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event83924 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event83925 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event83926 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 83925

def event83927 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 83923

def event83928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 83926 .coefficient) (.value (.predecessor 1 83927 .coefficient)))

def event83929 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event83930 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 83929

def event83931 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 83921

def event83932 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 83930 .coefficient, .predecessor 1 83931 .coefficient])

def event83933 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event83934 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 83933

def event83935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 83919

def event83936 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 83935 .coefficient))

def event83937 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event83938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18418⟩⟩) 0 ⟨10325⟩ 83937

def event83939 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18418⟩⟩) (.authority (.programFamilyFact))

def exact83940RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18418⟩⟩], []⟩, (1)⟩]

theorem exact83940RawTermsValid :
    exact83940RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83940 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18418⟩⟩) exact83940RawTerms (.finite 3) 83939 .exactZero (none)

def event83941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12771⟩⟩) 0 ⟨10325⟩ 83937

def event83942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12771⟩⟩) (.authority (.programFamilyFact))

def exact83943RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩], []⟩, (1)⟩]

theorem exact83943RawTermsValid :
    exact83943RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83943 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12771⟩⟩) exact83943RawTerms (.finite 3) 83942 .exactZero (none)

def event83944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18419⟩⟩) 0 ⟨12771⟩ 83943

def event83945 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18419⟩⟩) 1 ⟨18418⟩ 83940

def event83946 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18419⟩⟩) (.product (.predecessor 0 83944 .coefficient) (.predecessor 1 83945 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event83947 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18419⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], []⟩) [⟨.result 83943 .coefficient, true, some 1⟩, ⟨.result 83940 .coefficient, true, some 1⟩])

def event83948 : Event := .survivorFold (1) 83947

def exact83949RawTerms : List Term := []

theorem exact83949RawTermsValid :
    exact83949RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83949 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18419⟩⟩) exact83949RawTerms (.finite 9) 83946 (.finite 9) (some (83947))

def event83950 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18420⟩⟩) 0 ⟨18419⟩ 83949

def event83951 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18420⟩⟩) (.identity (.predecessor 0 83950 .coefficient))

def event83952 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18420⟩⟩) (.finite 9)

def event83953 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18636⟩⟩) 0 ⟨18420⟩ 83952

def event83954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18636⟩⟩) (.authority (.programFamilyFact))

def exact83955RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], []⟩, (1)⟩]

theorem exact83955RawTermsValid :
    exact83955RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83955 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18636⟩⟩) exact83955RawTerms (.finite 3) 83954 .exactZero (none)

def event83956 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18637⟩⟩) 0 ⟨18636⟩ 83955

def event83957 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18637⟩⟩) (.identity (.predecessor 0 83956 .coefficient))

def event83958 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18637⟩⟩) (.finite 3)

def event83959 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19576⟩⟩) 0 ⟨18637⟩ 83958

def event83960 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19576⟩⟩) (.authority (.relationPreimageSource ⟨59⟩))

def exact83961RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩, (1)⟩]

theorem exact83961RawTermsValid :
    exact83961RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83961 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19576⟩⟩) exact83961RawTerms (.finite 5647228698) 83960 .exactZero (none)

def event83962 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact83963RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact83963RawTermsValid :
    exact83963RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83963 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact83963RawTerms .large 83962 .exactZero (none)

def event83964 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19577⟩⟩) 0 ⟨35⟩ 83963

def event83965 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19577⟩⟩) 1 ⟨19576⟩ 83961

def event83966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19577⟩⟩) (.product (.predecessor 0 83964 .coefficient) (.predecessor 1 83965 .coefficient) (⟨false, false, none, none, none⟩))

def event83967 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19577⟩⟩, .operator (⟨83963, 0⟩, ⟨83961, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩, (1)⟩)

def eventLeaf5232 : Array AnnotatedEvent := #[
  { event := event83712
    frameStart := 0 },
  { event := event83713
    frameStart := 0 },
  { event := event83714
    frameStart := 83714 },
  { event := event83715
    frameStart := 83714 },
  { event := event83716
    frameStart := 83714 },
  { event := event83717
    frameStart := 83714 },
  { event := event83718
    frameStart := 83714 },
  { event := event83719
    frameStart := 83714 },
  { event := event83720
    frameStart := 83714 },
  { event := event83721
    frameStart := 83714 },
  { event := event83722
    frameStart := 83714 },
  { event := event83723
    frameStart := 83714 },
  { event := event83724
    frameStart := 83714 },
  { event := event83725
    frameStart := 83714 },
  { event := event83726
    frameStart := 83714 },
  { event := event83727
    frameStart := 83714 }
]

def eventLeaf5233 : Array AnnotatedEvent := #[
  { event := event83728
    frameStart := 83714 },
  { event := event83729
    frameStart := 83714 },
  { event := event83730
    frameStart := 83714 },
  { event := event83731
    frameStart := 83714 },
  { event := event83732
    frameStart := 83714 },
  { event := event83733
    frameStart := 83714 },
  { event := event83734
    frameStart := 83714 },
  { event := event83735
    frameStart := 83714 },
  { event := event83736
    frameStart := 83714 },
  { event := event83737
    frameStart := 83714 },
  { event := event83738
    frameStart := 83714 },
  { event := event83739
    frameStart := 83714 },
  { event := event83740
    frameStart := 83714 },
  { event := event83741
    frameStart := 83714 },
  { event := event83742
    frameStart := 83714 },
  { event := event83743
    frameStart := 83714 }
]

def eventLeaf5234 : Array AnnotatedEvent := #[
  { event := event83744
    frameStart := 83714 },
  { event := event83745
    frameStart := 83714 },
  { event := event83746
    frameStart := 83714 },
  { event := event83747
    frameStart := 83714 },
  { event := event83748
    frameStart := 83714 },
  { event := event83749
    frameStart := 83714 },
  { event := event83750
    frameStart := 83714 },
  { event := event83751
    frameStart := 83714 },
  { event := event83752
    frameStart := 83714 },
  { event := event83753
    frameStart := 83714 },
  { event := event83754
    frameStart := 83714 },
  { event := event83755
    frameStart := 83714 },
  { event := event83756
    frameStart := 83714 },
  { event := event83757
    frameStart := 83714 },
  { event := event83758
    frameStart := 83714 },
  { event := event83759
    frameStart := 83714 }
]

def eventLeaf5235 : Array AnnotatedEvent := #[
  { event := event83760
    frameStart := 83714 },
  { event := event83761
    frameStart := 83714 },
  { event := event83762
    frameStart := 83762 },
  { event := event83763
    frameStart := 83762 },
  { event := event83764
    frameStart := 83762 },
  { event := event83765
    frameStart := 83762 },
  { event := event83766
    frameStart := 83762 },
  { event := event83767
    frameStart := 83762 },
  { event := event83768
    frameStart := 83762 },
  { event := event83769
    frameStart := 83762 },
  { event := event83770
    frameStart := 83762 },
  { event := event83771
    frameStart := 83762 },
  { event := event83772
    frameStart := 83762 },
  { event := event83773
    frameStart := 83762 },
  { event := event83774
    frameStart := 83762 },
  { event := event83775
    frameStart := 83762 }
]

def eventLeaf5236 : Array AnnotatedEvent := #[
  { event := event83776
    frameStart := 83762 },
  { event := event83777
    frameStart := 83762 },
  { event := event83778
    frameStart := 83762 },
  { event := event83779
    frameStart := 83762 },
  { event := event83780
    frameStart := 83762 },
  { event := event83781
    frameStart := 83762 },
  { event := event83782
    frameStart := 83762 },
  { event := event83783
    frameStart := 83762 },
  { event := event83784
    frameStart := 83762 },
  { event := event83785
    frameStart := 83762 },
  { event := event83786
    frameStart := 83762 },
  { event := event83787
    frameStart := 83762 },
  { event := event83788
    frameStart := 83762 },
  { event := event83789
    frameStart := 83762 },
  { event := event83790
    frameStart := 83762 },
  { event := event83791
    frameStart := 83762 }
]

def eventLeaf5237 : Array AnnotatedEvent := #[
  { event := event83792
    frameStart := 83762 },
  { event := event83793
    frameStart := 83762 },
  { event := event83794
    frameStart := 83762 },
  { event := event83795
    frameStart := 83762 },
  { event := event83796
    frameStart := 83762 },
  { event := event83797
    frameStart := 83762 },
  { event := event83798
    frameStart := 83762 },
  { event := event83799
    frameStart := 83762 },
  { event := event83800
    frameStart := 83762 },
  { event := event83801
    frameStart := 83762 },
  { event := event83802
    frameStart := 83762 },
  { event := event83803
    frameStart := 83762 },
  { event := event83804
    frameStart := 83762 },
  { event := event83805
    frameStart := 83762 },
  { event := event83806
    frameStart := 83762 },
  { event := event83807
    frameStart := 83762 }
]

def eventLeaf5238 : Array AnnotatedEvent := #[
  { event := event83808
    frameStart := 83762 },
  { event := event83809
    frameStart := 83762 },
  { event := event83810
    frameStart := 83762 },
  { event := event83811
    frameStart := 83762 },
  { event := event83812
    frameStart := 83762 },
  { event := event83813
    frameStart := 83762 },
  { event := event83814
    frameStart := 83762 },
  { event := event83815
    frameStart := 83762 },
  { event := event83816
    frameStart := 83762 },
  { event := event83817
    frameStart := 83762 },
  { event := event83818
    frameStart := 83762 },
  { event := event83819
    frameStart := 83762 },
  { event := event83820
    frameStart := 83762 },
  { event := event83821
    frameStart := 83762 },
  { event := event83822
    frameStart := 83762 },
  { event := event83823
    frameStart := 83762 }
]

def eventLeaf5239 : Array AnnotatedEvent := #[
  { event := event83824
    frameStart := 83762 },
  { event := event83825
    frameStart := 83762 },
  { event := event83826
    frameStart := 83762 },
  { event := event83827
    frameStart := 83762 },
  { event := event83828
    frameStart := 83762 },
  { event := event83829
    frameStart := 83762 },
  { event := event83830
    frameStart := 83762 },
  { event := event83831
    frameStart := 83762 },
  { event := event83832
    frameStart := 83762 },
  { event := event83833
    frameStart := 83762 },
  { event := event83834
    frameStart := 83762 },
  { event := event83835
    frameStart := 83762 },
  { event := event83836
    frameStart := 83762 },
  { event := event83837
    frameStart := 83762 },
  { event := event83838
    frameStart := 83762 },
  { event := event83839
    frameStart := 83762 }
]

def eventLeaf5240 : Array AnnotatedEvent := #[
  { event := event83840
    frameStart := 83762 },
  { event := event83841
    frameStart := 83762 },
  { event := event83842
    frameStart := 83762 },
  { event := event83843
    frameStart := 83762 },
  { event := event83844
    frameStart := 83762 },
  { event := event83845
    frameStart := 83762 },
  { event := event83846
    frameStart := 83762 },
  { event := event83847
    frameStart := 83762 },
  { event := event83848
    frameStart := 83762 },
  { event := event83849
    frameStart := 83762 },
  { event := event83850
    frameStart := 83762 },
  { event := event83851
    frameStart := 83762 },
  { event := event83852
    frameStart := 83762 },
  { event := event83853
    frameStart := 83762 },
  { event := event83854
    frameStart := 83762 },
  { event := event83855
    frameStart := 83762 }
]

def eventLeaf5241 : Array AnnotatedEvent := #[
  { event := event83856
    frameStart := 83762 },
  { event := event83857
    frameStart := 83762 },
  { event := event83858
    frameStart := 83762 },
  { event := event83859
    frameStart := 83762 },
  { event := event83860
    frameStart := 83762 },
  { event := event83861
    frameStart := 83762 },
  { event := event83862
    frameStart := 83762 },
  { event := event83863
    frameStart := 83762 },
  { event := event83864
    frameStart := 83762 },
  { event := event83865
    frameStart := 83762 },
  { event := event83866
    frameStart := 83762 },
  { event := event83867
    frameStart := 83762 },
  { event := event83868
    frameStart := 83762 },
  { event := event83869
    frameStart := 83762 },
  { event := event83870
    frameStart := 83762 },
  { event := event83871
    frameStart := 83762 }
]

def eventLeaf5242 : Array AnnotatedEvent := #[
  { event := event83872
    frameStart := 83762 },
  { event := event83873
    frameStart := 83762 },
  { event := event83874
    frameStart := 83762 },
  { event := event83875
    frameStart := 83762 },
  { event := event83876
    frameStart := 83762 },
  { event := event83877
    frameStart := 83762 },
  { event := event83878
    frameStart := 83762 },
  { event := event83879
    frameStart := 83762 },
  { event := event83880
    frameStart := 0 },
  { event := event83881
    frameStart := 0 },
  { event := event83882
    frameStart := 0 },
  { event := event83883
    frameStart := 0 },
  { event := event83884
    frameStart := 0 },
  { event := event83885
    frameStart := 0 },
  { event := event83886
    frameStart := 0 },
  { event := event83887
    frameStart := 0 }
]

def eventLeaf5243 : Array AnnotatedEvent := #[
  { event := event83888
    frameStart := 0 },
  { event := event83889
    frameStart := 0 },
  { event := event83890
    frameStart := 0 },
  { event := event83891
    frameStart := 0 },
  { event := event83892
    frameStart := 0 },
  { event := event83893
    frameStart := 0 },
  { event := event83894
    frameStart := 0 },
  { event := event83895
    frameStart := 0 },
  { event := event83896
    frameStart := 0 },
  { event := event83897
    frameStart := 0 },
  { event := event83898
    frameStart := 0 },
  { event := event83899
    frameStart := 0 },
  { event := event83900
    frameStart := 0 },
  { event := event83901
    frameStart := 0 },
  { event := event83902
    frameStart := 0 },
  { event := event83903
    frameStart := 0 }
]

def eventLeaf5244 : Array AnnotatedEvent := #[
  { event := event83904
    frameStart := 0 },
  { event := event83905
    frameStart := 0 },
  { event := event83906
    frameStart := 0 },
  { event := event83907
    frameStart := 0 },
  { event := event83908
    frameStart := 0 },
  { event := event83909
    frameStart := 0 },
  { event := event83910
    frameStart := 0 },
  { event := event83911
    frameStart := 0 },
  { event := event83912
    frameStart := 0 },
  { event := event83913
    frameStart := 0 },
  { event := event83914
    frameStart := 0 },
  { event := event83915
    frameStart := 0 },
  { event := event83916
    frameStart := 0 },
  { event := event83917
    frameStart := 83917 },
  { event := event83918
    frameStart := 83917 },
  { event := event83919
    frameStart := 83917 }
]

def eventLeaf5245 : Array AnnotatedEvent := #[
  { event := event83920
    frameStart := 83917 },
  { event := event83921
    frameStart := 83917 },
  { event := event83922
    frameStart := 83917 },
  { event := event83923
    frameStart := 83917 },
  { event := event83924
    frameStart := 83917 },
  { event := event83925
    frameStart := 83917 },
  { event := event83926
    frameStart := 83917 },
  { event := event83927
    frameStart := 83917 },
  { event := event83928
    frameStart := 83917 },
  { event := event83929
    frameStart := 83917 },
  { event := event83930
    frameStart := 83917 },
  { event := event83931
    frameStart := 83917 },
  { event := event83932
    frameStart := 83917 },
  { event := event83933
    frameStart := 83917 },
  { event := event83934
    frameStart := 83917 },
  { event := event83935
    frameStart := 83917 }
]

def eventLeaf5246 : Array AnnotatedEvent := #[
  { event := event83936
    frameStart := 83917 },
  { event := event83937
    frameStart := 83917 },
  { event := event83938
    frameStart := 83917 },
  { event := event83939
    frameStart := 83917 },
  { event := event83940
    frameStart := 83917 },
  { event := event83941
    frameStart := 83917 },
  { event := event83942
    frameStart := 83917 },
  { event := event83943
    frameStart := 83917 },
  { event := event83944
    frameStart := 83917 },
  { event := event83945
    frameStart := 83917 },
  { event := event83946
    frameStart := 83917 },
  { event := event83947
    frameStart := 83917 },
  { event := event83948
    frameStart := 83917 },
  { event := event83949
    frameStart := 83917 },
  { event := event83950
    frameStart := 83917 },
  { event := event83951
    frameStart := 83917 }
]

def eventLeaf5247 : Array AnnotatedEvent := #[
  { event := event83952
    frameStart := 83917 },
  { event := event83953
    frameStart := 83917 },
  { event := event83954
    frameStart := 83917 },
  { event := event83955
    frameStart := 83917 },
  { event := event83956
    frameStart := 83917 },
  { event := event83957
    frameStart := 83917 },
  { event := event83958
    frameStart := 83917 },
  { event := event83959
    frameStart := 83917 },
  { event := event83960
    frameStart := 83917 },
  { event := event83961
    frameStart := 83917 },
  { event := event83962
    frameStart := 83917 },
  { event := event83963
    frameStart := 83917 },
  { event := event83964
    frameStart := 83917 },
  { event := event83965
    frameStart := 83917 },
  { event := event83966
    frameStart := 83917 },
  { event := event83967
    frameStart := 83917 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events327
