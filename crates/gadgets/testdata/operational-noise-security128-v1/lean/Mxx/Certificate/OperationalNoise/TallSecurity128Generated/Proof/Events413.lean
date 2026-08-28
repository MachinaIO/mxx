import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events413

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event105728 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45922⟩⟩) 0 ⟨5770⟩ 105245

def event105729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45922⟩⟩) 1 ⟨45921⟩ 105727

def event105730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45922⟩⟩) (.product (.predecessor 0 105728 .coefficient) (.predecessor 1 105729 .coefficient) (⟨false, false, none, none, none⟩))

def event105731 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45922⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨45919⟩⟩]⟩) [⟨.result 105723 .coefficient, false, none⟩])

def event105732 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45922⟩⟩) (.product (.result 105245 .summary) (.transfer 105731) (⟨false, false, none, none, none⟩))

def event105733 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45922⟩⟩, .operator (⟨105245, 0⟩, ⟨105727, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45919⟩⟩]⟩, (1)⟩)

def event105734 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨45920⟩⟩)

def event105735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event105736 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event105737 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.authority (.operator))

def event105738 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.finite 13)

def event105739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event105740 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event105741 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event105742 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event105743 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 105742

def event105744 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 105740

def event105745 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 105743 .coefficient) (.value (.predecessor 1 105744 .coefficient)))

def event105746 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event105747 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 0 ⟨392⟩ 105746

def event105748 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 1 ⟨5754⟩ 105738

def event105749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.sum [.predecessor 0 105747 .coefficient, .predecessor 1 105748 .coefficient])

def event105750 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.finite 655353)

def event105751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 0 ⟨5756⟩ 105750

def event105752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 1 ⟨5426⟩ 105736

def event105753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.identity (.predecessor 1 105752 .coefficient))

def event105754 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.finite 655360)

def event105755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45178⟩⟩) 0 ⟨5766⟩ 105754

def event105756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45178⟩⟩) (.authority (.programFamilyFact))

def exact105757RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45178⟩⟩], []⟩, (1)⟩]

theorem exact105757RawTermsValid :
    exact105757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105757 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45178⟩⟩) exact105757RawTerms (.finite 58) 105756 .exactZero (none)

def event105758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14796⟩⟩) 0 ⟨5766⟩ 105754

def event105759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14796⟩⟩) (.authority (.programFamilyFact))

def exact105760RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩], []⟩, (1)⟩]

theorem exact105760RawTermsValid :
    exact105760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105760 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14796⟩⟩) exact105760RawTerms (.finite 58) 105759 .exactZero (none)

def event105761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45179⟩⟩) 0 ⟨14796⟩ 105760

def event105762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45179⟩⟩) 1 ⟨45178⟩ 105757

def event105763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45179⟩⟩) (.product (.predecessor 0 105761 .coefficient) (.predecessor 1 105762 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event105764 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45179⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], []⟩) [⟨.result 105760 .coefficient, true, some 1⟩, ⟨.result 105757 .coefficient, true, some 1⟩])

def event105765 : Event := .survivorFold (1) 105764

def exact105766RawTerms : List Term := []

theorem exact105766RawTermsValid :
    exact105766RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105766 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45179⟩⟩) exact105766RawTerms (.finite 3364) 105763 (.finite 3364) (some (105764))

def event105767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45180⟩⟩) 0 ⟨45179⟩ 105766

def event105768 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45180⟩⟩) (.identity (.predecessor 0 105767 .coefficient))

def event105769 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45180⟩⟩) (.finite 3364)

def event105770 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45919⟩⟩) 0 ⟨45180⟩ 105769

def event105771 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45919⟩⟩) (.authority (.relationPreimageSource ⟨53⟩))

def exact105772RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨45919⟩⟩]⟩, (1)⟩]

theorem exact105772RawTermsValid :
    exact105772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105772 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45919⟩⟩) exact105772RawTerms (.finite 5647228698) 105771 .exactZero (none)

def event105773 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact105774RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact105774RawTermsValid :
    exact105774RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105774 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact105774RawTerms .large 105773 .exactZero (none)

def event105775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45920⟩⟩) 0 ⟨35⟩ 105774

def event105776 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45920⟩⟩) 1 ⟨45919⟩ 105772

def event105777 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45920⟩⟩) (.product (.predecessor 0 105775 .coefficient) (.predecessor 1 105776 .coefficient) (⟨false, false, none, none, none⟩))

def event105778 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45920⟩⟩, .operator (⟨105774, 0⟩, ⟨105772, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45919⟩⟩]⟩, (1)⟩)

def exact105779RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45919⟩⟩]⟩, (1)⟩]

theorem exact105779RawTermsValid :
    exact105779RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105779 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45920⟩⟩) exact105779RawTerms .large 105777 .exactZero (none)

def event105780 : Event := .preFoldPolynomial 105779 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45919⟩⟩]⟩, (1)⟩] .exactZero none

def exact105781RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45919⟩⟩]⟩, (1)⟩]

def event105781 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨45920⟩⟩) 105780 exact105781RawTerms .large 105777 .exactZero (none)

def event105782 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨46994⟩⟩)

def event105783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event105784 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event105785 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.authority (.operator))

def event105786 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.finite 13)

def event105787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event105788 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event105789 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event105790 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event105791 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 105790

def event105792 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 105788

def event105793 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 105791 .coefficient) (.value (.predecessor 1 105792 .coefficient)))

def event105794 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event105795 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 0 ⟨392⟩ 105794

def event105796 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 1 ⟨5754⟩ 105786

def event105797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.sum [.predecessor 0 105795 .coefficient, .predecessor 1 105796 .coefficient])

def event105798 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.finite 655353)

def event105799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 0 ⟨5756⟩ 105798

def event105800 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 1 ⟨5426⟩ 105784

def event105801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.identity (.predecessor 1 105800 .coefficient))

def event105802 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.finite 655360)

def event105803 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45178⟩⟩) 0 ⟨5766⟩ 105802

def event105804 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45178⟩⟩) (.authority (.programFamilyFact))

def exact105805RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45178⟩⟩], []⟩, (1)⟩]

theorem exact105805RawTermsValid :
    exact105805RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105805 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45178⟩⟩) exact105805RawTerms (.finite 58) 105804 .exactZero (none)

def event105806 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14796⟩⟩) 0 ⟨5766⟩ 105802

def event105807 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14796⟩⟩) (.authority (.programFamilyFact))

def exact105808RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩], []⟩, (1)⟩]

theorem exact105808RawTermsValid :
    exact105808RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105808 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14796⟩⟩) exact105808RawTerms (.finite 58) 105807 .exactZero (none)

def event105809 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45179⟩⟩) 0 ⟨14796⟩ 105808

def event105810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45179⟩⟩) 1 ⟨45178⟩ 105805

def event105811 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45179⟩⟩) (.product (.predecessor 0 105809 .coefficient) (.predecessor 1 105810 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event105812 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45179⟩⟩, .operator (⟨105808, 0⟩, ⟨105805, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], []⟩, (1)⟩)

def exact105813RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], []⟩, (1)⟩]

theorem exact105813RawTermsValid :
    exact105813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105813 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45179⟩⟩) exact105813RawTerms (.finite 3364) 105811 .exactZero (none)

def event105814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45180⟩⟩) 0 ⟨45179⟩ 105813

def event105815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45180⟩⟩) (.identity (.predecessor 0 105814 .coefficient))

def event105816 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45180⟩⟩) (.finite 3364)

def event105817 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46474⟩⟩) 0 ⟨45180⟩ 105816

def event105818 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46474⟩⟩) (.authority (.programFamilyFact))

def event105819 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46474⟩⟩) (.finite 3720)

def event105820 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event105821 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46475⟩⟩) 0 ⟨7177⟩ 105820

def event105822 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46475⟩⟩) 1 ⟨46474⟩ 105819

def event105823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46475⟩⟩) (.authority (.operator))

def exact105824RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46475⟩⟩]⟩, (1)⟩]

theorem exact105824RawTermsValid :
    exact105824RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105824 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46475⟩⟩) exact105824RawTerms .large 105823 .exactZero (none)

def event105825 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46990⟩⟩) 0 ⟨46475⟩ 105824

def event105826 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46990⟩⟩) (.authority (.operator))

def exact105827RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46990⟩⟩]⟩, (1)⟩]

theorem exact105827RawTermsValid :
    exact105827RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105827 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46990⟩⟩) exact105827RawTerms (.finite 8192) 105826 .exactZero (none)

def event105828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event105829 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event105830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46750⟩⟩) 0 ⟨45180⟩ 105816

def event105831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46750⟩⟩) 1 ⟨136⟩ 105829

def event105832 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46750⟩⟩) (.sum [.predecessor 0 105830 .coefficient, .predecessor 1 105831 .coefficient])

def event105833 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46750⟩⟩) (.finite 3364)

def event105834 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46751⟩⟩) 0 ⟨46750⟩ 105833

def event105835 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46751⟩⟩) (.identity (.predecessor 0 105834 .coefficient))

def exact105836RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], []⟩, (1)⟩]

theorem exact105836RawTermsValid :
    exact105836RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105836 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46751⟩⟩) exact105836RawTerms (.finite 3364) 105835 .exactZero (none)

def event105837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact105838RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact105838RawTermsValid :
    exact105838RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105838 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact105838RawTerms .large 105837 .exactZero (none)

def event105839 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46752⟩⟩) 0 ⟨6908⟩ 105838

def event105840 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46752⟩⟩) 1 ⟨46751⟩ 105836

def event105841 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46752⟩⟩) (.product (.predecessor 0 105839 .coefficient) (.predecessor 1 105840 .coefficient) (⟨false, false, none, none, none⟩))

def event105842 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46752⟩⟩, .operator (⟨105838, 0⟩, ⟨105836, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact105843RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact105843RawTermsValid :
    exact105843RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105843 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46752⟩⟩) exact105843RawTerms .large 105841 .exactZero (none)

def event105844 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event105845 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event105846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 105820

def event105847 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact105848RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact105848RawTermsValid :
    exact105848RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105848 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact105848RawTerms .large 105847 .exactZero (none)

def event105849 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7284⟩⟩) 0 ⟨7178⟩ 105848

def event105850 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7284⟩⟩) (.identity (.predecessor 0 105849 .coefficient))

def exact105851RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩]

theorem exact105851RawTermsValid :
    exact105851RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105851 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7284⟩⟩) exact105851RawTerms .large 105850 .exactZero (none)

def event105852 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9562⟩⟩) 0 ⟨7284⟩ 105851

def event105853 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9562⟩⟩) (.authority (.operator))

def exact105854RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩]

theorem exact105854RawTermsValid :
    exact105854RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105854 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9562⟩⟩) exact105854RawTerms (.finite 8192) 105853 .exactZero (none)

def event105855 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9563⟩⟩) 0 ⟨9562⟩ 105854

def event105856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9563⟩⟩) 1 ⟨2370⟩ 105845

def event105857 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9563⟩⟩) (.scale (.predecessor 0 105855 .coefficient) (.value (.predecessor 1 105856 .coefficient)))

def exact105858RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩]

theorem exact105858RawTermsValid :
    exact105858RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105858 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9563⟩⟩) exact105858RawTerms (.finite 8192) 105857 .exactZero (none)

def event105859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7301⟩⟩) 0 ⟨7178⟩ 105848

def event105860 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7301⟩⟩) (.identity (.predecessor 0 105859 .coefficient))

def exact105861RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩]

theorem exact105861RawTermsValid :
    exact105861RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105861 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7301⟩⟩) exact105861RawTerms .large 105860 .exactZero (none)

def event105862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9564⟩⟩) 0 ⟨7301⟩ 105861

def event105863 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9564⟩⟩) 1 ⟨9563⟩ 105858

def event105864 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9564⟩⟩) (.product (.predecessor 0 105862 .coefficient) (.predecessor 1 105863 .coefficient) (⟨false, false, none, none, none⟩))

def event105865 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9564⟩⟩, .operator (⟨105861, 0⟩, ⟨105858, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩)

def exact105866RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩]

theorem exact105866RawTermsValid :
    exact105866RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105866 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9564⟩⟩) exact105866RawTerms .large 105864 .exactZero (none)

def event105867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46753⟩⟩) 0 ⟨9564⟩ 105866

def event105868 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46753⟩⟩) 1 ⟨46752⟩ 105843

def event105869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46753⟩⟩) (.sum [.predecessor 0 105867 .coefficient, .predecessor 1 105868 .coefficient])

def exact105870RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact105870RawTermsValid :
    exact105870RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105870 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46753⟩⟩) exact105870RawTerms .large 105869 .exactZero (none)

def event105871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46993⟩⟩) 0 ⟨46753⟩ 105870

def event105872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46993⟩⟩) 1 ⟨46990⟩ 105827

def event105873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46993⟩⟩) (.product (.predecessor 0 105871 .coefficient) (.predecessor 1 105872 .coefficient) (⟨false, false, none, none, none⟩))

def event105874 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46993⟩⟩, .operator (⟨105870, 0⟩, ⟨105827, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩]⟩, (1)⟩)

def event105875 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46993⟩⟩, .operator (⟨105870, 1⟩, ⟨105827, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩]⟩, (-1)⟩)

def event105876 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨46993⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨46990⟩⟩) ⟨46475⟩ 105824)

def event105877 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46993⟩⟩, .relation 105876 0, ⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨46475⟩⟩]⟩, (-1)⟩)

def exact105878RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨46475⟩⟩]⟩, (-1)⟩]

theorem exact105878RawTermsValid :
    exact105878RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105878 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46993⟩⟩) exact105878RawTerms .large 105873 .exactZero (none)

def event105879 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45476⟩⟩) 0 ⟨45180⟩ 105816

def event105880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45476⟩⟩) (.authority (.programFamilyFact))

def exact105881RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45476⟩⟩], []⟩, (1)⟩]

theorem exact105881RawTermsValid :
    exact105881RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105881 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45476⟩⟩) exact105881RawTerms (.finite 58) 105880 .exactZero (none)

def event105882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45478⟩⟩) 0 ⟨6908⟩ 105838

def event105883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45478⟩⟩) 1 ⟨45476⟩ 105881

def event105884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45478⟩⟩) (.product (.predecessor 0 105882 .coefficient) (.predecessor 1 105883 .coefficient) (⟨false, true, none, none, some 1⟩))

def event105885 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45478⟩⟩, .operator (⟨105838, 0⟩, ⟨105881, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact105886RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact105886RawTermsValid :
    exact105886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105886 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45478⟩⟩) exact105886RawTerms .large 105884 .exactZero (none)

def event105887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7195⟩⟩) 0 ⟨7177⟩ 105820

def event105888 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7195⟩⟩) (.authority (.operator))

def exact105889RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩]

theorem exact105889RawTermsValid :
    exact105889RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105889 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7195⟩⟩) exact105889RawTerms .large 105888 .exactZero (none)

def event105890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45479⟩⟩) 0 ⟨7195⟩ 105889

def event105891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45479⟩⟩) 1 ⟨45478⟩ 105886

def event105892 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45479⟩⟩) (.sum [.predecessor 0 105890 .coefficient, .predecessor 1 105891 .coefficient])

def exact105893RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact105893RawTermsValid :
    exact105893RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105893 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45479⟩⟩) exact105893RawTerms .large 105892 .exactZero (none)

def event105894 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46994⟩⟩) 0 ⟨45479⟩ 105893

def event105895 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46994⟩⟩) 1 ⟨46993⟩ 105878

def event105896 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46994⟩⟩) (.sum [.predecessor 0 105894 .coefficient, .predecessor 1 105895 .coefficient])

def exact105897RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨46475⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact105897RawTermsValid :
    exact105897RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105897 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46994⟩⟩) exact105897RawTerms .large 105896 .exactZero (none)

def event105898 : Event := .preFoldPolynomial 105897 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨46475⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact105899RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨46475⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event105899 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨46994⟩⟩) 105898 exact105899RawTerms .large 105896 .exactZero (none)

def event105900 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨45180⟩⟩) ⟨⟨74⟩, ⟨53⟩, ⟨135⟩⟩ ⟨105734, 105900⟩

def event105901 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨45922⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45919⟩⟩]⟩) (1) 0 2 (.universal 105900 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45919⟩⟩]⟩) (none) 105899)

def event105902 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45922⟩⟩, .relation 105901 0, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩)

def event105903 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45922⟩⟩, .relation 105901 1, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩]⟩, (-1)⟩)

def event105904 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45922⟩⟩, .relation 105901 2, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨46475⟩⟩]⟩, (1)⟩)

def event105905 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45922⟩⟩, .relation 105901 3, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨45476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact105906RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨46475⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨45476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact105906RawTermsValid :
    exact105906RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105906 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45922⟩⟩) exact105906RawTerms .large 105730 (.finite 202072841853861888) (some (105732))

def event105907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46992⟩⟩) 0 ⟨45922⟩ 105906

def event105908 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46992⟩⟩) 1 ⟨46991⟩ 105720

def event105909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46992⟩⟩) (.sum [.predecessor 0 105907 .coefficient, .predecessor 1 105908 .coefficient])

def event105910 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46992⟩⟩, .operator (⟨105906, 2⟩, ⟨105720, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], [⟨.program ⟨257⟩, ⟨46475⟩⟩]⟩, (-1)⟩)

def event105911 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46992⟩⟩, .operator (⟨105906, 1⟩, ⟨105720, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46990⟩⟩]⟩, (1)⟩)

def event105912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46992⟩⟩) (.sum [.result 105906 .summary, .result 105720 .summary])

def exact105913RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨45476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact105913RawTermsValid :
    exact105913RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105913 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46992⟩⟩) exact105913RawTerms .large 105909 (.finite 2998328565150755586048) (some (105912))

def event105914 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47376⟩⟩) 0 ⟨46992⟩ 105913

def event105915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47376⟩⟩) 1 ⟨47374⟩ 105636

def event105916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47376⟩⟩) (.product (.predecessor 0 105914 .coefficient) (.predecessor 1 105915 .coefficient) (⟨false, false, none, none, none⟩))

def event105917 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47376⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨47374⟩⟩]⟩) [⟨.result 105636 .coefficient, false, none⟩])

def event105918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47376⟩⟩) (.product (.result 105913 .summary) (.transfer 105917) (⟨false, false, none, none, none⟩))

def event105919 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47376⟩⟩, .operator (⟨105913, 0⟩, ⟨105636, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47374⟩⟩]⟩, (1)⟩)

def event105920 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47376⟩⟩, .operator (⟨105913, 1⟩, ⟨105636, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨45476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47374⟩⟩]⟩, (-1)⟩)

def event105921 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47376⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨45476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47374⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47374⟩⟩) ⟨46630⟩ 105633)

def event105922 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47376⟩⟩, .relation 105921 0, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨45476⟩⟩], [⟨.program ⟨257⟩, ⟨46630⟩⟩]⟩, (-1)⟩)

def exact105923RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47374⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨45476⟩⟩], [⟨.program ⟨257⟩, ⟨46630⟩⟩]⟩, (-1)⟩]

theorem exact105923RawTermsValid :
    exact105923RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105923 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47376⟩⟩) exact105923RawTerms .large 105916 (.finite 32194307824962751379413684715520) (some (105918))

def event105924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46236⟩⟩) 0 ⟨45477⟩ 4622

def event105925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46236⟩⟩) (.authority (.relationPreimageSource ⟨92⟩))

def exact105926RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46236⟩⟩]⟩, (1)⟩]

theorem exact105926RawTermsValid :
    exact105926RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105926 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46236⟩⟩) exact105926RawTerms (.finite 5647228698) 105925 .exactZero (none)

def event105927 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46238⟩⟩) 0 ⟨46236⟩ 105926

def event105928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46238⟩⟩) 1 ⟨2370⟩ 4

def event105929 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46238⟩⟩) (.scale (.predecessor 0 105927 .coefficient) (.value (.predecessor 1 105928 .coefficient)))

def exact105930RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46236⟩⟩]⟩, (1)⟩]

theorem exact105930RawTermsValid :
    exact105930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105930 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46238⟩⟩) exact105930RawTerms (.finite 5647228698) 105929 .exactZero (none)

def event105931 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46239⟩⟩) 0 ⟨5770⟩ 105245

def event105932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46239⟩⟩) 1 ⟨46238⟩ 105930

def event105933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46239⟩⟩) (.product (.predecessor 0 105931 .coefficient) (.predecessor 1 105932 .coefficient) (⟨false, false, none, none, none⟩))

def event105934 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46239⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨46236⟩⟩]⟩) [⟨.result 105926 .coefficient, false, none⟩])

def event105935 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46239⟩⟩) (.product (.result 105245 .summary) (.transfer 105934) (⟨false, false, none, none, none⟩))

def event105936 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46239⟩⟩, .operator (⟨105245, 0⟩, ⟨105930, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46236⟩⟩]⟩, (1)⟩)

def event105937 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨46237⟩⟩)

def event105938 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event105939 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event105940 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.authority (.operator))

def event105941 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.finite 13)

def event105942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event105943 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event105944 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event105945 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event105946 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 105945

def event105947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 105943

def event105948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 105946 .coefficient) (.value (.predecessor 1 105947 .coefficient)))

def event105949 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event105950 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 0 ⟨392⟩ 105949

def event105951 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 1 ⟨5754⟩ 105941

def event105952 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.sum [.predecessor 0 105950 .coefficient, .predecessor 1 105951 .coefficient])

def event105953 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.finite 655353)

def event105954 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 0 ⟨5756⟩ 105953

def event105955 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 1 ⟨5426⟩ 105939

def event105956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.identity (.predecessor 1 105955 .coefficient))

def event105957 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.finite 655360)

def event105958 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45178⟩⟩) 0 ⟨5766⟩ 105957

def event105959 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45178⟩⟩) (.authority (.programFamilyFact))

def exact105960RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45178⟩⟩], []⟩, (1)⟩]

theorem exact105960RawTermsValid :
    exact105960RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105960 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45178⟩⟩) exact105960RawTerms (.finite 58) 105959 .exactZero (none)

def event105961 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14796⟩⟩) 0 ⟨5766⟩ 105957

def event105962 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14796⟩⟩) (.authority (.programFamilyFact))

def exact105963RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩], []⟩, (1)⟩]

theorem exact105963RawTermsValid :
    exact105963RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105963 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14796⟩⟩) exact105963RawTerms (.finite 58) 105962 .exactZero (none)

def event105964 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45179⟩⟩) 0 ⟨14796⟩ 105963

def event105965 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45179⟩⟩) 1 ⟨45178⟩ 105960

def event105966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45179⟩⟩) (.product (.predecessor 0 105964 .coefficient) (.predecessor 1 105965 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event105967 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45179⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14796⟩⟩, ⟨.program ⟨257⟩, ⟨45178⟩⟩], []⟩) [⟨.result 105963 .coefficient, true, some 1⟩, ⟨.result 105960 .coefficient, true, some 1⟩])

def event105968 : Event := .survivorFold (1) 105967

def exact105969RawTerms : List Term := []

theorem exact105969RawTermsValid :
    exact105969RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105969 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45179⟩⟩) exact105969RawTerms (.finite 3364) 105966 (.finite 3364) (some (105967))

def event105970 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45180⟩⟩) 0 ⟨45179⟩ 105969

def event105971 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45180⟩⟩) (.identity (.predecessor 0 105970 .coefficient))

def event105972 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45180⟩⟩) (.finite 3364)

def event105973 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45476⟩⟩) 0 ⟨45180⟩ 105972

def event105974 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45476⟩⟩) (.authority (.programFamilyFact))

def exact105975RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45476⟩⟩], []⟩, (1)⟩]

theorem exact105975RawTermsValid :
    exact105975RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105975 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45476⟩⟩) exact105975RawTerms (.finite 58) 105974 .exactZero (none)

def event105976 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45477⟩⟩) 0 ⟨45476⟩ 105975

def event105977 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45477⟩⟩) (.identity (.predecessor 0 105976 .coefficient))

def event105978 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45477⟩⟩) (.finite 58)

def event105979 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46236⟩⟩) 0 ⟨45477⟩ 105978

def event105980 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46236⟩⟩) (.authority (.relationPreimageSource ⟨92⟩))

def exact105981RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46236⟩⟩]⟩, (1)⟩]

theorem exact105981RawTermsValid :
    exact105981RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105981 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46236⟩⟩) exact105981RawTerms (.finite 5647228698) 105980 .exactZero (none)

def event105982 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact105983RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact105983RawTermsValid :
    exact105983RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event105983 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact105983RawTerms .large 105982 .exactZero (none)

def eventLeaf6608 : Array AnnotatedEvent := #[
  { event := event105728
    frameStart := 0 },
  { event := event105729
    frameStart := 0 },
  { event := event105730
    frameStart := 0 },
  { event := event105731
    frameStart := 0 },
  { event := event105732
    frameStart := 0 },
  { event := event105733
    frameStart := 0 },
  { event := event105734
    frameStart := 105734 },
  { event := event105735
    frameStart := 105734 },
  { event := event105736
    frameStart := 105734 },
  { event := event105737
    frameStart := 105734 },
  { event := event105738
    frameStart := 105734 },
  { event := event105739
    frameStart := 105734 },
  { event := event105740
    frameStart := 105734 },
  { event := event105741
    frameStart := 105734 },
  { event := event105742
    frameStart := 105734 },
  { event := event105743
    frameStart := 105734 }
]

def eventLeaf6609 : Array AnnotatedEvent := #[
  { event := event105744
    frameStart := 105734 },
  { event := event105745
    frameStart := 105734 },
  { event := event105746
    frameStart := 105734 },
  { event := event105747
    frameStart := 105734 },
  { event := event105748
    frameStart := 105734 },
  { event := event105749
    frameStart := 105734 },
  { event := event105750
    frameStart := 105734 },
  { event := event105751
    frameStart := 105734 },
  { event := event105752
    frameStart := 105734 },
  { event := event105753
    frameStart := 105734 },
  { event := event105754
    frameStart := 105734 },
  { event := event105755
    frameStart := 105734 },
  { event := event105756
    frameStart := 105734 },
  { event := event105757
    frameStart := 105734 },
  { event := event105758
    frameStart := 105734 },
  { event := event105759
    frameStart := 105734 }
]

def eventLeaf6610 : Array AnnotatedEvent := #[
  { event := event105760
    frameStart := 105734 },
  { event := event105761
    frameStart := 105734 },
  { event := event105762
    frameStart := 105734 },
  { event := event105763
    frameStart := 105734 },
  { event := event105764
    frameStart := 105734 },
  { event := event105765
    frameStart := 105734 },
  { event := event105766
    frameStart := 105734 },
  { event := event105767
    frameStart := 105734 },
  { event := event105768
    frameStart := 105734 },
  { event := event105769
    frameStart := 105734 },
  { event := event105770
    frameStart := 105734 },
  { event := event105771
    frameStart := 105734 },
  { event := event105772
    frameStart := 105734 },
  { event := event105773
    frameStart := 105734 },
  { event := event105774
    frameStart := 105734 },
  { event := event105775
    frameStart := 105734 }
]

def eventLeaf6611 : Array AnnotatedEvent := #[
  { event := event105776
    frameStart := 105734 },
  { event := event105777
    frameStart := 105734 },
  { event := event105778
    frameStart := 105734 },
  { event := event105779
    frameStart := 105734 },
  { event := event105780
    frameStart := 105734 },
  { event := event105781
    frameStart := 105734 },
  { event := event105782
    frameStart := 105782 },
  { event := event105783
    frameStart := 105782 },
  { event := event105784
    frameStart := 105782 },
  { event := event105785
    frameStart := 105782 },
  { event := event105786
    frameStart := 105782 },
  { event := event105787
    frameStart := 105782 },
  { event := event105788
    frameStart := 105782 },
  { event := event105789
    frameStart := 105782 },
  { event := event105790
    frameStart := 105782 },
  { event := event105791
    frameStart := 105782 }
]

def eventLeaf6612 : Array AnnotatedEvent := #[
  { event := event105792
    frameStart := 105782 },
  { event := event105793
    frameStart := 105782 },
  { event := event105794
    frameStart := 105782 },
  { event := event105795
    frameStart := 105782 },
  { event := event105796
    frameStart := 105782 },
  { event := event105797
    frameStart := 105782 },
  { event := event105798
    frameStart := 105782 },
  { event := event105799
    frameStart := 105782 },
  { event := event105800
    frameStart := 105782 },
  { event := event105801
    frameStart := 105782 },
  { event := event105802
    frameStart := 105782 },
  { event := event105803
    frameStart := 105782 },
  { event := event105804
    frameStart := 105782 },
  { event := event105805
    frameStart := 105782 },
  { event := event105806
    frameStart := 105782 },
  { event := event105807
    frameStart := 105782 }
]

def eventLeaf6613 : Array AnnotatedEvent := #[
  { event := event105808
    frameStart := 105782 },
  { event := event105809
    frameStart := 105782 },
  { event := event105810
    frameStart := 105782 },
  { event := event105811
    frameStart := 105782 },
  { event := event105812
    frameStart := 105782 },
  { event := event105813
    frameStart := 105782 },
  { event := event105814
    frameStart := 105782 },
  { event := event105815
    frameStart := 105782 },
  { event := event105816
    frameStart := 105782 },
  { event := event105817
    frameStart := 105782 },
  { event := event105818
    frameStart := 105782 },
  { event := event105819
    frameStart := 105782 },
  { event := event105820
    frameStart := 105782 },
  { event := event105821
    frameStart := 105782 },
  { event := event105822
    frameStart := 105782 },
  { event := event105823
    frameStart := 105782 }
]

def eventLeaf6614 : Array AnnotatedEvent := #[
  { event := event105824
    frameStart := 105782 },
  { event := event105825
    frameStart := 105782 },
  { event := event105826
    frameStart := 105782 },
  { event := event105827
    frameStart := 105782 },
  { event := event105828
    frameStart := 105782 },
  { event := event105829
    frameStart := 105782 },
  { event := event105830
    frameStart := 105782 },
  { event := event105831
    frameStart := 105782 },
  { event := event105832
    frameStart := 105782 },
  { event := event105833
    frameStart := 105782 },
  { event := event105834
    frameStart := 105782 },
  { event := event105835
    frameStart := 105782 },
  { event := event105836
    frameStart := 105782 },
  { event := event105837
    frameStart := 105782 },
  { event := event105838
    frameStart := 105782 },
  { event := event105839
    frameStart := 105782 }
]

def eventLeaf6615 : Array AnnotatedEvent := #[
  { event := event105840
    frameStart := 105782 },
  { event := event105841
    frameStart := 105782 },
  { event := event105842
    frameStart := 105782 },
  { event := event105843
    frameStart := 105782 },
  { event := event105844
    frameStart := 105782 },
  { event := event105845
    frameStart := 105782 },
  { event := event105846
    frameStart := 105782 },
  { event := event105847
    frameStart := 105782 },
  { event := event105848
    frameStart := 105782 },
  { event := event105849
    frameStart := 105782 },
  { event := event105850
    frameStart := 105782 },
  { event := event105851
    frameStart := 105782 },
  { event := event105852
    frameStart := 105782 },
  { event := event105853
    frameStart := 105782 },
  { event := event105854
    frameStart := 105782 },
  { event := event105855
    frameStart := 105782 }
]

def eventLeaf6616 : Array AnnotatedEvent := #[
  { event := event105856
    frameStart := 105782 },
  { event := event105857
    frameStart := 105782 },
  { event := event105858
    frameStart := 105782 },
  { event := event105859
    frameStart := 105782 },
  { event := event105860
    frameStart := 105782 },
  { event := event105861
    frameStart := 105782 },
  { event := event105862
    frameStart := 105782 },
  { event := event105863
    frameStart := 105782 },
  { event := event105864
    frameStart := 105782 },
  { event := event105865
    frameStart := 105782 },
  { event := event105866
    frameStart := 105782 },
  { event := event105867
    frameStart := 105782 },
  { event := event105868
    frameStart := 105782 },
  { event := event105869
    frameStart := 105782 },
  { event := event105870
    frameStart := 105782 },
  { event := event105871
    frameStart := 105782 }
]

def eventLeaf6617 : Array AnnotatedEvent := #[
  { event := event105872
    frameStart := 105782 },
  { event := event105873
    frameStart := 105782 },
  { event := event105874
    frameStart := 105782 },
  { event := event105875
    frameStart := 105782 },
  { event := event105876
    frameStart := 105782 },
  { event := event105877
    frameStart := 105782 },
  { event := event105878
    frameStart := 105782 },
  { event := event105879
    frameStart := 105782 },
  { event := event105880
    frameStart := 105782 },
  { event := event105881
    frameStart := 105782 },
  { event := event105882
    frameStart := 105782 },
  { event := event105883
    frameStart := 105782 },
  { event := event105884
    frameStart := 105782 },
  { event := event105885
    frameStart := 105782 },
  { event := event105886
    frameStart := 105782 },
  { event := event105887
    frameStart := 105782 }
]

def eventLeaf6618 : Array AnnotatedEvent := #[
  { event := event105888
    frameStart := 105782 },
  { event := event105889
    frameStart := 105782 },
  { event := event105890
    frameStart := 105782 },
  { event := event105891
    frameStart := 105782 },
  { event := event105892
    frameStart := 105782 },
  { event := event105893
    frameStart := 105782 },
  { event := event105894
    frameStart := 105782 },
  { event := event105895
    frameStart := 105782 },
  { event := event105896
    frameStart := 105782 },
  { event := event105897
    frameStart := 105782 },
  { event := event105898
    frameStart := 105782 },
  { event := event105899
    frameStart := 105782 },
  { event := event105900
    frameStart := 0 },
  { event := event105901
    frameStart := 0 },
  { event := event105902
    frameStart := 0 },
  { event := event105903
    frameStart := 0 }
]

def eventLeaf6619 : Array AnnotatedEvent := #[
  { event := event105904
    frameStart := 0 },
  { event := event105905
    frameStart := 0 },
  { event := event105906
    frameStart := 0 },
  { event := event105907
    frameStart := 0 },
  { event := event105908
    frameStart := 0 },
  { event := event105909
    frameStart := 0 },
  { event := event105910
    frameStart := 0 },
  { event := event105911
    frameStart := 0 },
  { event := event105912
    frameStart := 0 },
  { event := event105913
    frameStart := 0 },
  { event := event105914
    frameStart := 0 },
  { event := event105915
    frameStart := 0 },
  { event := event105916
    frameStart := 0 },
  { event := event105917
    frameStart := 0 },
  { event := event105918
    frameStart := 0 },
  { event := event105919
    frameStart := 0 }
]

def eventLeaf6620 : Array AnnotatedEvent := #[
  { event := event105920
    frameStart := 0 },
  { event := event105921
    frameStart := 0 },
  { event := event105922
    frameStart := 0 },
  { event := event105923
    frameStart := 0 },
  { event := event105924
    frameStart := 0 },
  { event := event105925
    frameStart := 0 },
  { event := event105926
    frameStart := 0 },
  { event := event105927
    frameStart := 0 },
  { event := event105928
    frameStart := 0 },
  { event := event105929
    frameStart := 0 },
  { event := event105930
    frameStart := 0 },
  { event := event105931
    frameStart := 0 },
  { event := event105932
    frameStart := 0 },
  { event := event105933
    frameStart := 0 },
  { event := event105934
    frameStart := 0 },
  { event := event105935
    frameStart := 0 }
]

def eventLeaf6621 : Array AnnotatedEvent := #[
  { event := event105936
    frameStart := 0 },
  { event := event105937
    frameStart := 105937 },
  { event := event105938
    frameStart := 105937 },
  { event := event105939
    frameStart := 105937 },
  { event := event105940
    frameStart := 105937 },
  { event := event105941
    frameStart := 105937 },
  { event := event105942
    frameStart := 105937 },
  { event := event105943
    frameStart := 105937 },
  { event := event105944
    frameStart := 105937 },
  { event := event105945
    frameStart := 105937 },
  { event := event105946
    frameStart := 105937 },
  { event := event105947
    frameStart := 105937 },
  { event := event105948
    frameStart := 105937 },
  { event := event105949
    frameStart := 105937 },
  { event := event105950
    frameStart := 105937 },
  { event := event105951
    frameStart := 105937 }
]

def eventLeaf6622 : Array AnnotatedEvent := #[
  { event := event105952
    frameStart := 105937 },
  { event := event105953
    frameStart := 105937 },
  { event := event105954
    frameStart := 105937 },
  { event := event105955
    frameStart := 105937 },
  { event := event105956
    frameStart := 105937 },
  { event := event105957
    frameStart := 105937 },
  { event := event105958
    frameStart := 105937 },
  { event := event105959
    frameStart := 105937 },
  { event := event105960
    frameStart := 105937 },
  { event := event105961
    frameStart := 105937 },
  { event := event105962
    frameStart := 105937 },
  { event := event105963
    frameStart := 105937 },
  { event := event105964
    frameStart := 105937 },
  { event := event105965
    frameStart := 105937 },
  { event := event105966
    frameStart := 105937 },
  { event := event105967
    frameStart := 105937 }
]

def eventLeaf6623 : Array AnnotatedEvent := #[
  { event := event105968
    frameStart := 105937 },
  { event := event105969
    frameStart := 105937 },
  { event := event105970
    frameStart := 105937 },
  { event := event105971
    frameStart := 105937 },
  { event := event105972
    frameStart := 105937 },
  { event := event105973
    frameStart := 105937 },
  { event := event105974
    frameStart := 105937 },
  { event := event105975
    frameStart := 105937 },
  { event := event105976
    frameStart := 105937 },
  { event := event105977
    frameStart := 105937 },
  { event := event105978
    frameStart := 105937 },
  { event := event105979
    frameStart := 105937 },
  { event := event105980
    frameStart := 105937 },
  { event := event105981
    frameStart := 105937 },
  { event := event105982
    frameStart := 105937 },
  { event := event105983
    frameStart := 105937 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events413
