import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events382

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event97792 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28483⟩⟩) 1 ⟨28482⟩ 97767

def event97793 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28483⟩⟩) (.product (.predecessor 0 97791 .coefficient) (.predecessor 1 97792 .coefficient) (⟨false, false, none, none, none⟩))

def event97794 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28483⟩⟩, .operator (⟨97790, 0⟩, ⟨97767, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28482⟩⟩]⟩, (1)⟩)

def event97795 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28483⟩⟩, .operator (⟨97790, 1⟩, ⟨97767, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28482⟩⟩]⟩, (-1)⟩)

def event97796 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28483⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28482⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28482⟩⟩) ⟨24342⟩ 97764)

def event97797 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28483⟩⟩, .relation 97796 0, ⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24342⟩⟩]⟩, (-1)⟩)

def exact97798RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28482⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24342⟩⟩]⟩, (-1)⟩]

theorem exact97798RawTermsValid :
    exact97798RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97798 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28483⟩⟩) exact97798RawTerms .large 97793 .exactZero (none)

def event97799 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16301⟩⟩) 0 ⟨16253⟩ 97756

def event97800 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16301⟩⟩) (.authority (.programFamilyFact))

def exact97801RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16301⟩⟩], []⟩, (1)⟩]

theorem exact97801RawTermsValid :
    exact97801RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97801 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16301⟩⟩) exact97801RawTerms (.finite 62) 97800 .exactZero (none)

def event97802 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16302⟩⟩) 0 ⟨6544⟩ 97778

def event97803 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16302⟩⟩) 1 ⟨16301⟩ 97801

def event97804 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16302⟩⟩) (.product (.predecessor 0 97802 .coefficient) (.predecessor 1 97803 .coefficient) (⟨false, true, none, none, some 1⟩))

def event97805 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16302⟩⟩, .operator (⟨97778, 0⟩, ⟨97801, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact97806RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact97806RawTermsValid :
    exact97806RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97806 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16302⟩⟩) exact97806RawTerms .large 97804 .exactZero (none)

def event97807 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6729⟩⟩) 0 ⟨6689⟩ 97760

def event97808 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6729⟩⟩) (.authority (.operator))

def exact97809RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩]

theorem exact97809RawTermsValid :
    exact97809RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97809 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6729⟩⟩) exact97809RawTerms .large 97808 .exactZero (none)

def event97810 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16303⟩⟩) 0 ⟨6729⟩ 97809

def event97811 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16303⟩⟩) 1 ⟨16302⟩ 97806

def event97812 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16303⟩⟩) (.sum [.predecessor 0 97810 .coefficient, .predecessor 1 97811 .coefficient])

def exact97813RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97813RawTermsValid :
    exact97813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97813 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16303⟩⟩) exact97813RawTerms .large 97812 .exactZero (none)

def event97814 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28487⟩⟩) 0 ⟨16303⟩ 97813

def event97815 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28487⟩⟩) 1 ⟨28483⟩ 97798

def event97816 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28487⟩⟩) (.sum [.predecessor 0 97814 .coefficient, .predecessor 1 97815 .coefficient])

def exact97817RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28482⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24342⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97817RawTermsValid :
    exact97817RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97817 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28487⟩⟩) exact97817RawTerms .large 97816 .exactZero (none)

def event97818 : Event := .preFoldPolynomial 97817 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28482⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24342⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact97819RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28482⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24342⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event97819 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28487⟩⟩) 97818 exact97819RawTerms .large 97816 .exactZero (none)

def event97820 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16253⟩⟩) ⟨⟨142⟩, ⟨50⟩, ⟨109⟩⟩ ⟨97686, 97820⟩

def event97821 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21824⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21821⟩⟩]⟩) (1) 0 2 (.universal 97820 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21821⟩⟩]⟩) (none) 97819)

def event97822 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21824⟩⟩, .relation 97821 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩)

def event97823 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21824⟩⟩, .relation 97821 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28482⟩⟩]⟩, (-1)⟩)

def event97824 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21824⟩⟩, .relation 97821 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24342⟩⟩]⟩, (1)⟩)

def event97825 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21824⟩⟩, .relation 97821 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact97826RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28482⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24342⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97826RawTermsValid :
    exact97826RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97826 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21824⟩⟩) exact97826RawTerms .large 97682 (.finite 1811303510016) (some (97684))

def event97827 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28485⟩⟩) 0 ⟨21824⟩ 97826

def event97828 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28485⟩⟩) 1 ⟨28484⟩ 97672

def event97829 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28485⟩⟩) (.sum [.predecessor 0 97827 .coefficient, .predecessor 1 97828 .coefficient])

def event97830 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28485⟩⟩, .operator (⟨97826, 0⟩, ⟨97672, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28482⟩⟩]⟩, (1)⟩)

def event97831 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28485⟩⟩, .operator (⟨97826, 2⟩, ⟨97672, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16252⟩⟩], [⟨.program ⟨214⟩, ⟨24342⟩⟩]⟩, (-1)⟩)

def event97832 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28485⟩⟩) (.sum [.result 97826 .summary, .result 97672 .summary])

def exact97833RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97833RawTermsValid :
    exact97833RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97833 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28485⟩⟩) exact97833RawTerms .large 97829 (.finite 1292202948609709846528) (some (97832))

def event97834 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24277⟩⟩) 0 ⟨16169⟩ 4767

def event97835 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24277⟩⟩) (.authority (.programFamilyFact))

def event97836 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24277⟩⟩) (.finite 3720)

def event97837 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24279⟩⟩) 0 ⟨6689⟩ 5477

def event97838 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24279⟩⟩) 1 ⟨24277⟩ 97836

def event97839 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24279⟩⟩) (.authority (.operator))

def exact97840RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24279⟩⟩]⟩, (1)⟩]

theorem exact97840RawTermsValid :
    exact97840RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97840 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24279⟩⟩) exact97840RawTerms .large 97839 .exactZero (none)

def event97841 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28265⟩⟩) 0 ⟨24279⟩ 97840

def event97842 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28265⟩⟩) (.authority (.operator))

def exact97843RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩, (1)⟩]

theorem exact97843RawTermsValid :
    exact97843RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97843 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28265⟩⟩) exact97843RawTerms (.finite 8192) 97842 .exactZero (none)

def event97844 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23661⟩⟩) 0 ⟨14616⟩ 4761

def event97845 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23661⟩⟩) (.authority (.programFamilyFact))

def event97846 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23661⟩⟩) (.finite 3720)

def event97847 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23662⟩⟩) 0 ⟨6689⟩ 5477

def event97848 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23662⟩⟩) 1 ⟨23661⟩ 97846

def event97849 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23662⟩⟩) (.authority (.operator))

def exact97850RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23662⟩⟩]⟩, (1)⟩]

theorem exact97850RawTermsValid :
    exact97850RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97850 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23662⟩⟩) exact97850RawTerms .large 97849 .exactZero (none)

def event97851 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26207⟩⟩) 0 ⟨23662⟩ 97850

def event97852 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26207⟩⟩) (.authority (.operator))

def exact97853RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26207⟩⟩]⟩, (1)⟩]

theorem exact97853RawTermsValid :
    exact97853RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97853 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26207⟩⟩) exact97853RawTerms (.finite 8192) 97852 .exactZero (none)

def event97854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11626⟩⟩) 0 ⟨11625⟩ 4750

def event97855 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11626⟩⟩) 1 ⟨6564⟩ 32

def event97856 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11626⟩⟩) (.tensor (.predecessor 0 97854 .coefficient) (.predecessor 1 97855 .coefficient) true false)

def event97857 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11626⟩⟩, .operator (⟨4750, 0⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11625⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact97858RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11625⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact97858RawTermsValid :
    exact97858RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97858 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11626⟩⟩) exact97858RawTerms .large 97856 .exactZero (none)

def event97859 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7118⟩⟩) 0 ⟨5506⟩ 27

def event97860 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7118⟩⟩) 1 ⟨6781⟩ 10480

def event97861 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7118⟩⟩) (.product (.predecessor 0 97859 .coefficient) (.predecessor 1 97860 .coefficient) (⟨false, false, none, none, none⟩))

def event97862 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7118⟩⟩, .operator (⟨27, 0⟩, ⟨10480, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩)

def exact97863RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩]

theorem exact97863RawTermsValid :
    exact97863RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97863 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7118⟩⟩) exact97863RawTerms .large 97861 .exactZero (none)

def event97864 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11627⟩⟩) 0 ⟨7118⟩ 97863

def event97865 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11627⟩⟩) 1 ⟨11626⟩ 97858

def event97866 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11627⟩⟩) (.sum [.predecessor 0 97864 .coefficient, .predecessor 1 97865 .coefficient])

def exact97867RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11625⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97867RawTermsValid :
    exact97867RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97867 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11627⟩⟩) exact97867RawTerms .large 97866 .exactZero (none)

def event97868 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11628⟩⟩) 0 ⟨11627⟩ 97867

def event97869 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11628⟩⟩) 1 ⟨95⟩ 10472

def event97870 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11628⟩⟩) (.sum [.predecessor 0 97868 .coefficient, .predecessor 1 97869 .coefficient])

def event97871 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11628⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨95⟩⟩]⟩) [⟨.result 10472 .coefficient, false, none⟩])

def event97872 : Event := .survivorFold (1) 97871

def exact97873RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11625⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97873RawTermsValid :
    exact97873RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97873 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11628⟩⟩) exact97873RawTerms .large 97870 (.finite 26) (some (97871))

def event97874 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14617⟩⟩) 0 ⟨11628⟩ 97873

def event97875 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14617⟩⟩) 1 ⟨14614⟩ 4753

def event97876 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14617⟩⟩) (.product (.predecessor 0 97874 .coefficient) (.predecessor 1 97875 .coefficient) (⟨false, true, none, none, some 1⟩))

def event97877 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14617⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨14614⟩⟩], []⟩) [⟨.result 4753 .coefficient, true, some 1⟩])

def event97878 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14617⟩⟩) (.product (.result 97873 .summary) (.transfer 97877) (⟨false, false, none, none, none⟩))

def event97879 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14617⟩⟩, .operator (⟨97873, 1⟩, ⟨4753, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event97880 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14617⟩⟩, .operator (⟨97873, 0⟩, ⟨4753, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩)

def exact97881RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩]

theorem exact97881RawTermsValid :
    exact97881RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97881 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14617⟩⟩) exact97881RawTerms .large 97876 (.finite 23296) (some (97878))

def event97882 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14618⟩⟩) 0 ⟨14614⟩ 4753

def event97883 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14618⟩⟩) 1 ⟨6564⟩ 32

def event97884 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14618⟩⟩) (.tensor (.predecessor 0 97882 .coefficient) (.predecessor 1 97883 .coefficient) true false)

def event97885 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14618⟩⟩, .operator (⟨4753, 0⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact97886RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact97886RawTermsValid :
    exact97886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97886 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14618⟩⟩) exact97886RawTerms .large 97884 .exactZero (none)

def event97887 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7099⟩⟩) 0 ⟨5506⟩ 27

def event97888 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7099⟩⟩) 1 ⟨6762⟩ 10521

def event97889 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7099⟩⟩) (.product (.predecessor 0 97887 .coefficient) (.predecessor 1 97888 .coefficient) (⟨false, false, none, none, none⟩))

def event97890 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7099⟩⟩, .operator (⟨27, 0⟩, ⟨10521, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩]⟩, (1)⟩)

def exact97891RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩]⟩, (1)⟩]

theorem exact97891RawTermsValid :
    exact97891RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97891 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7099⟩⟩) exact97891RawTerms .large 97889 .exactZero (none)

def event97892 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14619⟩⟩) 0 ⟨7099⟩ 97891

def event97893 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14619⟩⟩) 1 ⟨14618⟩ 97886

def event97894 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14619⟩⟩) (.sum [.predecessor 0 97892 .coefficient, .predecessor 1 97893 .coefficient])

def exact97895RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97895RawTermsValid :
    exact97895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97895 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14619⟩⟩) exact97895RawTerms .large 97894 .exactZero (none)

def event97896 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14620⟩⟩) 0 ⟨14619⟩ 97895

def event97897 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14620⟩⟩) 1 ⟨76⟩ 10513

def event97898 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14620⟩⟩) (.sum [.predecessor 0 97896 .coefficient, .predecessor 1 97897 .coefficient])

def event97899 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14620⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨76⟩⟩]⟩) [⟨.result 10513 .coefficient, false, none⟩])

def event97900 : Event := .survivorFold (1) 97899

def exact97901RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97901RawTermsValid :
    exact97901RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97901 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14620⟩⟩) exact97901RawTerms .large 97898 (.finite 26) (some (97899))

def event97902 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14621⟩⟩) 0 ⟨14620⟩ 97901

def event97903 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14621⟩⟩) 1 ⟨7859⟩ 10510

def event97904 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14621⟩⟩) (.product (.predecessor 0 97902 .coefficient) (.predecessor 1 97903 .coefficient) (⟨false, false, none, none, none⟩))

def event97905 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14621⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩) [⟨.result 10506 .coefficient, false, none⟩])

def event97906 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14621⟩⟩) (.product (.result 97901 .summary) (.transfer 97905) (⟨false, false, none, none, none⟩))

def event97907 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14621⟩⟩, .operator (⟨97901, 1⟩, ⟨10510, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (-1)⟩)

def event97908 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨14621⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7858⟩⟩) ⟨6781⟩ 10480)

def event97909 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14621⟩⟩, .relation 97908 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (-1)⟩)

def event97910 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14621⟩⟩, .operator (⟨97901, 0⟩, ⟨10510, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩)

def exact97911RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (-1)⟩]

theorem exact97911RawTermsValid :
    exact97911RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97911 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14621⟩⟩) exact97911RawTerms .large 97904 (.finite 95420416) (some (97906))

def event97912 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14622⟩⟩) 0 ⟨14621⟩ 97911

def event97913 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14622⟩⟩) 1 ⟨14617⟩ 97881

def event97914 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14622⟩⟩) (.sum [.predecessor 0 97912 .coefficient, .predecessor 1 97913 .coefficient])

def event97915 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14622⟩⟩, .operator (⟨97911, 1⟩, ⟨97881, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩)

def event97916 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14622⟩⟩) (.sum [.result 97911 .summary, .result 97881 .summary])

def exact97917RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact97917RawTermsValid :
    exact97917RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97917 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14622⟩⟩) exact97917RawTerms .large 97914 (.finite 95443712) (some (97916))

def event97918 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26208⟩⟩) 0 ⟨14622⟩ 97917

def event97919 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26208⟩⟩) 1 ⟨26207⟩ 97853

def event97920 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26208⟩⟩) (.product (.predecessor 0 97918 .coefficient) (.predecessor 1 97919 .coefficient) (⟨false, false, none, none, none⟩))

def event97921 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26208⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26207⟩⟩]⟩) [⟨.result 97853 .coefficient, false, none⟩])

def event97922 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26208⟩⟩) (.product (.result 97917 .summary) (.transfer 97921) (⟨false, false, none, none, none⟩))

def event97923 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26208⟩⟩, .operator (⟨97917, 1⟩, ⟨97853, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26207⟩⟩]⟩, (-1)⟩)

def event97924 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26208⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26207⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26207⟩⟩) ⟨23662⟩ 97850)

def event97925 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26208⟩⟩, .relation 97924 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨23662⟩⟩]⟩, (-1)⟩)

def event97926 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26208⟩⟩, .operator (⟨97917, 0⟩, ⟨97853, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26207⟩⟩]⟩, (1)⟩)

def exact97927RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26207⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨23662⟩⟩]⟩, (-1)⟩]

theorem exact97927RawTermsValid :
    exact97927RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97927 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26208⟩⟩) exact97927RawTerms .large 97920 (.finite 350279950139392) (some (97922))

def event97928 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19661⟩⟩) 0 ⟨14616⟩ 4761

def event97929 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19661⟩⟩) (.authority (.relationPreimageSource ⟨17⟩))

def exact97930RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19661⟩⟩]⟩, (1)⟩]

theorem exact97930RawTermsValid :
    exact97930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97930 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19661⟩⟩) exact97930RawTerms (.finite 136065468) 97929 .exactZero (none)

def event97931 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19663⟩⟩) 0 ⟨19661⟩ 97930

def event97932 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19663⟩⟩) 1 ⟨2348⟩ 4

def event97933 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19663⟩⟩) (.scale (.predecessor 0 97931 .coefficient) (.value (.predecessor 1 97932 .coefficient)))

def exact97934RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19661⟩⟩]⟩, (1)⟩]

theorem exact97934RawTermsValid :
    exact97934RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97934 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19663⟩⟩) exact97934RawTerms (.finite 136065468) 97933 .exactZero (none)

def event97935 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19664⟩⟩) 0 ⟨5509⟩ 94462

def event97936 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19664⟩⟩) 1 ⟨19663⟩ 97934

def event97937 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19664⟩⟩) (.product (.predecessor 0 97935 .coefficient) (.predecessor 1 97936 .coefficient) (⟨false, false, none, none, none⟩))

def event97938 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19664⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19661⟩⟩]⟩) [⟨.result 97930 .coefficient, false, none⟩])

def event97939 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19664⟩⟩) (.product (.result 94462 .summary) (.transfer 97938) (⟨false, false, none, none, none⟩))

def event97940 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19664⟩⟩, .operator (⟨94462, 0⟩, ⟨97934, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19661⟩⟩]⟩, (1)⟩)

def event97941 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19662⟩⟩)

def event97942 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event97943 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event97944 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event97945 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event97946 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 97945

def event97947 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 97943

def event97948 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 97946 .coefficient) (.value (.predecessor 1 97947 .coefficient)))

def event97949 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event97950 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11625⟩⟩) 0 ⟨5503⟩ 97949

def event97951 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11625⟩⟩) (.authority (.programFamilyFact))

def exact97952RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩], []⟩, (1)⟩]

theorem exact97952RawTermsValid :
    exact97952RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97952 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11625⟩⟩) exact97952RawTerms (.finite 28) 97951 .exactZero (none)

def event97953 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14614⟩⟩) 0 ⟨5503⟩ 97949

def event97954 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14614⟩⟩) (.authority (.programFamilyFact))

def exact97955RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14614⟩⟩], []⟩, (1)⟩]

theorem exact97955RawTermsValid :
    exact97955RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97955 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14614⟩⟩) exact97955RawTerms (.finite 28) 97954 .exactZero (none)

def event97956 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14615⟩⟩) 0 ⟨14614⟩ 97955

def event97957 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14615⟩⟩) 1 ⟨11625⟩ 97952

def event97958 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14615⟩⟩) (.product (.predecessor 0 97956 .coefficient) (.predecessor 1 97957 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event97959 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14615⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], []⟩) [⟨.result 97955 .coefficient, true, some 1⟩, ⟨.result 97952 .coefficient, true, some 1⟩])

def event97960 : Event := .survivorFold (1) 97959

def exact97961RawTerms : List Term := []

theorem exact97961RawTermsValid :
    exact97961RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97961 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14615⟩⟩) exact97961RawTerms (.finite 784) 97958 (.finite 784) (some (97959))

def event97962 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14616⟩⟩) 0 ⟨14615⟩ 97961

def event97963 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14616⟩⟩) (.identity (.predecessor 0 97962 .coefficient))

def event97964 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14616⟩⟩) (.finite 784)

def event97965 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19661⟩⟩) 0 ⟨14616⟩ 97964

def event97966 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19661⟩⟩) (.authority (.relationPreimageSource ⟨17⟩))

def exact97967RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19661⟩⟩]⟩, (1)⟩]

theorem exact97967RawTermsValid :
    exact97967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97967 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19661⟩⟩) exact97967RawTerms (.finite 136065468) 97966 .exactZero (none)

def event97968 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact97969RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact97969RawTermsValid :
    exact97969RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97969 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact97969RawTerms .large 97968 .exactZero (none)

def event97970 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19662⟩⟩) 0 ⟨6⟩ 97969

def event97971 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19662⟩⟩) 1 ⟨19661⟩ 97967

def event97972 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19662⟩⟩) (.product (.predecessor 0 97970 .coefficient) (.predecessor 1 97971 .coefficient) (⟨false, false, none, none, none⟩))

def event97973 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19662⟩⟩, .operator (⟨97969, 0⟩, ⟨97967, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19661⟩⟩]⟩, (1)⟩)

def exact97974RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19661⟩⟩]⟩, (1)⟩]

theorem exact97974RawTermsValid :
    exact97974RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97974 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19662⟩⟩) exact97974RawTerms .large 97972 .exactZero (none)

def event97975 : Event := .preFoldPolynomial 97974 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19661⟩⟩]⟩, (1)⟩] .exactZero none

def exact97976RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19661⟩⟩]⟩, (1)⟩]

def event97976 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19662⟩⟩) 97975 exact97976RawTerms .large 97972 .exactZero (none)

def event97977 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26211⟩⟩)

def event97978 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event97979 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event97980 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event97981 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event97982 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 97981

def event97983 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 97979

def event97984 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 97982 .coefficient) (.value (.predecessor 1 97983 .coefficient)))

def event97985 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event97986 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11625⟩⟩) 0 ⟨5503⟩ 97985

def event97987 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11625⟩⟩) (.authority (.programFamilyFact))

def exact97988RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩], []⟩, (1)⟩]

theorem exact97988RawTermsValid :
    exact97988RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97988 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11625⟩⟩) exact97988RawTerms (.finite 28) 97987 .exactZero (none)

def event97989 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14614⟩⟩) 0 ⟨5503⟩ 97985

def event97990 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14614⟩⟩) (.authority (.programFamilyFact))

def exact97991RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14614⟩⟩], []⟩, (1)⟩]

theorem exact97991RawTermsValid :
    exact97991RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97991 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14614⟩⟩) exact97991RawTerms (.finite 28) 97990 .exactZero (none)

def event97992 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14615⟩⟩) 0 ⟨14614⟩ 97991

def event97993 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14615⟩⟩) 1 ⟨11625⟩ 97988

def event97994 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14615⟩⟩) (.product (.predecessor 0 97992 .coefficient) (.predecessor 1 97993 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event97995 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14615⟩⟩, .operator (⟨97991, 0⟩, ⟨97988, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], []⟩, (1)⟩)

def exact97996RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], []⟩, (1)⟩]

theorem exact97996RawTermsValid :
    exact97996RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event97996 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14615⟩⟩) exact97996RawTerms (.finite 784) 97994 .exactZero (none)

def event97997 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14616⟩⟩) 0 ⟨14615⟩ 97996

def event97998 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14616⟩⟩) (.identity (.predecessor 0 97997 .coefficient))

def event97999 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14616⟩⟩) (.finite 784)

def event98000 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23661⟩⟩) 0 ⟨14616⟩ 97999

def event98001 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23661⟩⟩) (.authority (.programFamilyFact))

def event98002 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23661⟩⟩) (.finite 3720)

def event98003 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event98004 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23662⟩⟩) 0 ⟨6689⟩ 98003

def event98005 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23662⟩⟩) 1 ⟨23661⟩ 98002

def event98006 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23662⟩⟩) (.authority (.operator))

def exact98007RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23662⟩⟩]⟩, (1)⟩]

theorem exact98007RawTermsValid :
    exact98007RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98007 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23662⟩⟩) exact98007RawTerms .large 98006 .exactZero (none)

def event98008 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26207⟩⟩) 0 ⟨23662⟩ 98007

def event98009 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26207⟩⟩) (.authority (.operator))

def exact98010RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26207⟩⟩]⟩, (1)⟩]

theorem exact98010RawTermsValid :
    exact98010RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98010 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26207⟩⟩) exact98010RawTerms (.finite 8192) 98009 .exactZero (none)

def event98011 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event98012 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event98013 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14740⟩⟩) 0 ⟨14616⟩ 97999

def event98014 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14740⟩⟩) 1 ⟨110⟩ 98012

def event98015 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14740⟩⟩) (.sum [.predecessor 0 98013 .coefficient, .predecessor 1 98014 .coefficient])

def event98016 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14740⟩⟩) (.finite 784)

def event98017 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14741⟩⟩) 0 ⟨14740⟩ 98016

def event98018 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14741⟩⟩) (.identity (.predecessor 0 98017 .coefficient))

def exact98019RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], []⟩, (1)⟩]

theorem exact98019RawTermsValid :
    exact98019RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98019 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14741⟩⟩) exact98019RawTerms (.finite 784) 98018 .exactZero (none)

def event98020 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact98021RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact98021RawTermsValid :
    exact98021RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98021 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact98021RawTerms .large 98020 .exactZero (none)

def event98022 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14742⟩⟩) 0 ⟨6544⟩ 98021

def event98023 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14742⟩⟩) 1 ⟨14741⟩ 98019

def event98024 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14742⟩⟩) (.product (.predecessor 0 98022 .coefficient) (.predecessor 1 98023 .coefficient) (⟨false, false, none, none, none⟩))

def event98025 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14742⟩⟩, .operator (⟨98021, 0⟩, ⟨98019, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact98026RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11625⟩⟩, ⟨.program ⟨214⟩, ⟨14614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact98026RawTermsValid :
    exact98026RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98026 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14742⟩⟩) exact98026RawTerms .large 98024 .exactZero (none)

def event98027 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event98028 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event98029 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 98003

def event98030 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact98031RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact98031RawTermsValid :
    exact98031RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98031 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact98031RawTerms .large 98030 .exactZero (none)

def event98032 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6781⟩⟩) 0 ⟨6757⟩ 98031

def event98033 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6781⟩⟩) (.identity (.predecessor 0 98032 .coefficient))

def exact98034RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩]

theorem exact98034RawTermsValid :
    exact98034RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98034 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6781⟩⟩) exact98034RawTerms .large 98033 .exactZero (none)

def event98035 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7858⟩⟩) 0 ⟨6781⟩ 98034

def event98036 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7858⟩⟩) (.authority (.operator))

def exact98037RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩]

theorem exact98037RawTermsValid :
    exact98037RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98037 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7858⟩⟩) exact98037RawTerms (.finite 8192) 98036 .exactZero (none)

def event98038 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7859⟩⟩) 0 ⟨7858⟩ 98037

def event98039 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7859⟩⟩) 1 ⟨2348⟩ 98028

def event98040 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7859⟩⟩) (.scale (.predecessor 0 98038 .coefficient) (.value (.predecessor 1 98039 .coefficient)))

def exact98041RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩]

theorem exact98041RawTermsValid :
    exact98041RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98041 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7859⟩⟩) exact98041RawTerms (.finite 8192) 98040 .exactZero (none)

def event98042 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6762⟩⟩) 0 ⟨6757⟩ 98031

def event98043 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6762⟩⟩) (.identity (.predecessor 0 98042 .coefficient))

def exact98044RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩]⟩, (1)⟩]

theorem exact98044RawTermsValid :
    exact98044RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event98044 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6762⟩⟩) exact98044RawTerms .large 98043 .exactZero (none)

def event98045 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7860⟩⟩) 0 ⟨6762⟩ 98044

def event98046 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7860⟩⟩) 1 ⟨7859⟩ 98041

def event98047 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7860⟩⟩) (.product (.predecessor 0 98045 .coefficient) (.predecessor 1 98046 .coefficient) (⟨false, false, none, none, none⟩))

def eventLeaf6112 : Array AnnotatedEvent := #[
  { event := event97792
    frameStart := 97728 },
  { event := event97793
    frameStart := 97728 },
  { event := event97794
    frameStart := 97728 },
  { event := event97795
    frameStart := 97728 },
  { event := event97796
    frameStart := 97728 },
  { event := event97797
    frameStart := 97728 },
  { event := event97798
    frameStart := 97728 },
  { event := event97799
    frameStart := 97728 },
  { event := event97800
    frameStart := 97728 },
  { event := event97801
    frameStart := 97728 },
  { event := event97802
    frameStart := 97728 },
  { event := event97803
    frameStart := 97728 },
  { event := event97804
    frameStart := 97728 },
  { event := event97805
    frameStart := 97728 },
  { event := event97806
    frameStart := 97728 },
  { event := event97807
    frameStart := 97728 }
]

def eventLeaf6113 : Array AnnotatedEvent := #[
  { event := event97808
    frameStart := 97728 },
  { event := event97809
    frameStart := 97728 },
  { event := event97810
    frameStart := 97728 },
  { event := event97811
    frameStart := 97728 },
  { event := event97812
    frameStart := 97728 },
  { event := event97813
    frameStart := 97728 },
  { event := event97814
    frameStart := 97728 },
  { event := event97815
    frameStart := 97728 },
  { event := event97816
    frameStart := 97728 },
  { event := event97817
    frameStart := 97728 },
  { event := event97818
    frameStart := 97728 },
  { event := event97819
    frameStart := 97728 },
  { event := event97820
    frameStart := 0 },
  { event := event97821
    frameStart := 0 },
  { event := event97822
    frameStart := 0 },
  { event := event97823
    frameStart := 0 }
]

def eventLeaf6114 : Array AnnotatedEvent := #[
  { event := event97824
    frameStart := 0 },
  { event := event97825
    frameStart := 0 },
  { event := event97826
    frameStart := 0 },
  { event := event97827
    frameStart := 0 },
  { event := event97828
    frameStart := 0 },
  { event := event97829
    frameStart := 0 },
  { event := event97830
    frameStart := 0 },
  { event := event97831
    frameStart := 0 },
  { event := event97832
    frameStart := 0 },
  { event := event97833
    frameStart := 0 },
  { event := event97834
    frameStart := 0 },
  { event := event97835
    frameStart := 0 },
  { event := event97836
    frameStart := 0 },
  { event := event97837
    frameStart := 0 },
  { event := event97838
    frameStart := 0 },
  { event := event97839
    frameStart := 0 }
]

def eventLeaf6115 : Array AnnotatedEvent := #[
  { event := event97840
    frameStart := 0 },
  { event := event97841
    frameStart := 0 },
  { event := event97842
    frameStart := 0 },
  { event := event97843
    frameStart := 0 },
  { event := event97844
    frameStart := 0 },
  { event := event97845
    frameStart := 0 },
  { event := event97846
    frameStart := 0 },
  { event := event97847
    frameStart := 0 },
  { event := event97848
    frameStart := 0 },
  { event := event97849
    frameStart := 0 },
  { event := event97850
    frameStart := 0 },
  { event := event97851
    frameStart := 0 },
  { event := event97852
    frameStart := 0 },
  { event := event97853
    frameStart := 0 },
  { event := event97854
    frameStart := 0 },
  { event := event97855
    frameStart := 0 }
]

def eventLeaf6116 : Array AnnotatedEvent := #[
  { event := event97856
    frameStart := 0 },
  { event := event97857
    frameStart := 0 },
  { event := event97858
    frameStart := 0 },
  { event := event97859
    frameStart := 0 },
  { event := event97860
    frameStart := 0 },
  { event := event97861
    frameStart := 0 },
  { event := event97862
    frameStart := 0 },
  { event := event97863
    frameStart := 0 },
  { event := event97864
    frameStart := 0 },
  { event := event97865
    frameStart := 0 },
  { event := event97866
    frameStart := 0 },
  { event := event97867
    frameStart := 0 },
  { event := event97868
    frameStart := 0 },
  { event := event97869
    frameStart := 0 },
  { event := event97870
    frameStart := 0 },
  { event := event97871
    frameStart := 0 }
]

def eventLeaf6117 : Array AnnotatedEvent := #[
  { event := event97872
    frameStart := 0 },
  { event := event97873
    frameStart := 0 },
  { event := event97874
    frameStart := 0 },
  { event := event97875
    frameStart := 0 },
  { event := event97876
    frameStart := 0 },
  { event := event97877
    frameStart := 0 },
  { event := event97878
    frameStart := 0 },
  { event := event97879
    frameStart := 0 },
  { event := event97880
    frameStart := 0 },
  { event := event97881
    frameStart := 0 },
  { event := event97882
    frameStart := 0 },
  { event := event97883
    frameStart := 0 },
  { event := event97884
    frameStart := 0 },
  { event := event97885
    frameStart := 0 },
  { event := event97886
    frameStart := 0 },
  { event := event97887
    frameStart := 0 }
]

def eventLeaf6118 : Array AnnotatedEvent := #[
  { event := event97888
    frameStart := 0 },
  { event := event97889
    frameStart := 0 },
  { event := event97890
    frameStart := 0 },
  { event := event97891
    frameStart := 0 },
  { event := event97892
    frameStart := 0 },
  { event := event97893
    frameStart := 0 },
  { event := event97894
    frameStart := 0 },
  { event := event97895
    frameStart := 0 },
  { event := event97896
    frameStart := 0 },
  { event := event97897
    frameStart := 0 },
  { event := event97898
    frameStart := 0 },
  { event := event97899
    frameStart := 0 },
  { event := event97900
    frameStart := 0 },
  { event := event97901
    frameStart := 0 },
  { event := event97902
    frameStart := 0 },
  { event := event97903
    frameStart := 0 }
]

def eventLeaf6119 : Array AnnotatedEvent := #[
  { event := event97904
    frameStart := 0 },
  { event := event97905
    frameStart := 0 },
  { event := event97906
    frameStart := 0 },
  { event := event97907
    frameStart := 0 },
  { event := event97908
    frameStart := 0 },
  { event := event97909
    frameStart := 0 },
  { event := event97910
    frameStart := 0 },
  { event := event97911
    frameStart := 0 },
  { event := event97912
    frameStart := 0 },
  { event := event97913
    frameStart := 0 },
  { event := event97914
    frameStart := 0 },
  { event := event97915
    frameStart := 0 },
  { event := event97916
    frameStart := 0 },
  { event := event97917
    frameStart := 0 },
  { event := event97918
    frameStart := 0 },
  { event := event97919
    frameStart := 0 }
]

def eventLeaf6120 : Array AnnotatedEvent := #[
  { event := event97920
    frameStart := 0 },
  { event := event97921
    frameStart := 0 },
  { event := event97922
    frameStart := 0 },
  { event := event97923
    frameStart := 0 },
  { event := event97924
    frameStart := 0 },
  { event := event97925
    frameStart := 0 },
  { event := event97926
    frameStart := 0 },
  { event := event97927
    frameStart := 0 },
  { event := event97928
    frameStart := 0 },
  { event := event97929
    frameStart := 0 },
  { event := event97930
    frameStart := 0 },
  { event := event97931
    frameStart := 0 },
  { event := event97932
    frameStart := 0 },
  { event := event97933
    frameStart := 0 },
  { event := event97934
    frameStart := 0 },
  { event := event97935
    frameStart := 0 }
]

def eventLeaf6121 : Array AnnotatedEvent := #[
  { event := event97936
    frameStart := 0 },
  { event := event97937
    frameStart := 0 },
  { event := event97938
    frameStart := 0 },
  { event := event97939
    frameStart := 0 },
  { event := event97940
    frameStart := 0 },
  { event := event97941
    frameStart := 97941 },
  { event := event97942
    frameStart := 97941 },
  { event := event97943
    frameStart := 97941 },
  { event := event97944
    frameStart := 97941 },
  { event := event97945
    frameStart := 97941 },
  { event := event97946
    frameStart := 97941 },
  { event := event97947
    frameStart := 97941 },
  { event := event97948
    frameStart := 97941 },
  { event := event97949
    frameStart := 97941 },
  { event := event97950
    frameStart := 97941 },
  { event := event97951
    frameStart := 97941 }
]

def eventLeaf6122 : Array AnnotatedEvent := #[
  { event := event97952
    frameStart := 97941 },
  { event := event97953
    frameStart := 97941 },
  { event := event97954
    frameStart := 97941 },
  { event := event97955
    frameStart := 97941 },
  { event := event97956
    frameStart := 97941 },
  { event := event97957
    frameStart := 97941 },
  { event := event97958
    frameStart := 97941 },
  { event := event97959
    frameStart := 97941 },
  { event := event97960
    frameStart := 97941 },
  { event := event97961
    frameStart := 97941 },
  { event := event97962
    frameStart := 97941 },
  { event := event97963
    frameStart := 97941 },
  { event := event97964
    frameStart := 97941 },
  { event := event97965
    frameStart := 97941 },
  { event := event97966
    frameStart := 97941 },
  { event := event97967
    frameStart := 97941 }
]

def eventLeaf6123 : Array AnnotatedEvent := #[
  { event := event97968
    frameStart := 97941 },
  { event := event97969
    frameStart := 97941 },
  { event := event97970
    frameStart := 97941 },
  { event := event97971
    frameStart := 97941 },
  { event := event97972
    frameStart := 97941 },
  { event := event97973
    frameStart := 97941 },
  { event := event97974
    frameStart := 97941 },
  { event := event97975
    frameStart := 97941 },
  { event := event97976
    frameStart := 97941 },
  { event := event97977
    frameStart := 97977 },
  { event := event97978
    frameStart := 97977 },
  { event := event97979
    frameStart := 97977 },
  { event := event97980
    frameStart := 97977 },
  { event := event97981
    frameStart := 97977 },
  { event := event97982
    frameStart := 97977 },
  { event := event97983
    frameStart := 97977 }
]

def eventLeaf6124 : Array AnnotatedEvent := #[
  { event := event97984
    frameStart := 97977 },
  { event := event97985
    frameStart := 97977 },
  { event := event97986
    frameStart := 97977 },
  { event := event97987
    frameStart := 97977 },
  { event := event97988
    frameStart := 97977 },
  { event := event97989
    frameStart := 97977 },
  { event := event97990
    frameStart := 97977 },
  { event := event97991
    frameStart := 97977 },
  { event := event97992
    frameStart := 97977 },
  { event := event97993
    frameStart := 97977 },
  { event := event97994
    frameStart := 97977 },
  { event := event97995
    frameStart := 97977 },
  { event := event97996
    frameStart := 97977 },
  { event := event97997
    frameStart := 97977 },
  { event := event97998
    frameStart := 97977 },
  { event := event97999
    frameStart := 97977 }
]

def eventLeaf6125 : Array AnnotatedEvent := #[
  { event := event98000
    frameStart := 97977 },
  { event := event98001
    frameStart := 97977 },
  { event := event98002
    frameStart := 97977 },
  { event := event98003
    frameStart := 97977 },
  { event := event98004
    frameStart := 97977 },
  { event := event98005
    frameStart := 97977 },
  { event := event98006
    frameStart := 97977 },
  { event := event98007
    frameStart := 97977 },
  { event := event98008
    frameStart := 97977 },
  { event := event98009
    frameStart := 97977 },
  { event := event98010
    frameStart := 97977 },
  { event := event98011
    frameStart := 97977 },
  { event := event98012
    frameStart := 97977 },
  { event := event98013
    frameStart := 97977 },
  { event := event98014
    frameStart := 97977 },
  { event := event98015
    frameStart := 97977 }
]

def eventLeaf6126 : Array AnnotatedEvent := #[
  { event := event98016
    frameStart := 97977 },
  { event := event98017
    frameStart := 97977 },
  { event := event98018
    frameStart := 97977 },
  { event := event98019
    frameStart := 97977 },
  { event := event98020
    frameStart := 97977 },
  { event := event98021
    frameStart := 97977 },
  { event := event98022
    frameStart := 97977 },
  { event := event98023
    frameStart := 97977 },
  { event := event98024
    frameStart := 97977 },
  { event := event98025
    frameStart := 97977 },
  { event := event98026
    frameStart := 97977 },
  { event := event98027
    frameStart := 97977 },
  { event := event98028
    frameStart := 97977 },
  { event := event98029
    frameStart := 97977 },
  { event := event98030
    frameStart := 97977 },
  { event := event98031
    frameStart := 97977 }
]

def eventLeaf6127 : Array AnnotatedEvent := #[
  { event := event98032
    frameStart := 97977 },
  { event := event98033
    frameStart := 97977 },
  { event := event98034
    frameStart := 97977 },
  { event := event98035
    frameStart := 97977 },
  { event := event98036
    frameStart := 97977 },
  { event := event98037
    frameStart := 97977 },
  { event := event98038
    frameStart := 97977 },
  { event := event98039
    frameStart := 97977 },
  { event := event98040
    frameStart := 97977 },
  { event := event98041
    frameStart := 97977 },
  { event := event98042
    frameStart := 97977 },
  { event := event98043
    frameStart := 97977 },
  { event := event98044
    frameStart := 97977 },
  { event := event98045
    frameStart := 97977 },
  { event := event98046
    frameStart := 97977 },
  { event := event98047
    frameStart := 97977 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events382
