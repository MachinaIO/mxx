import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events300

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event76800 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16503⟩⟩) 0 ⟨6544⟩ 76799

def event76801 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16503⟩⟩) 1 ⟨16502⟩ 76797

def event76802 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16503⟩⟩) (.product (.predecessor 0 76800 .coefficient) (.predecessor 1 76801 .coefficient) (⟨false, false, none, none, none⟩))

def event76803 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16503⟩⟩, .operator (⟨76799, 0⟩, ⟨76797, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact76804RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact76804RawTermsValid :
    exact76804RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76804 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16503⟩⟩) exact76804RawTerms .large 76802 .exactZero (none)

def event76805 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6702⟩⟩) 0 ⟨6689⟩ 76781

def event76806 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6702⟩⟩) (.authority (.operator))

def exact76807RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩]

theorem exact76807RawTermsValid :
    exact76807RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76807 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6702⟩⟩) exact76807RawTerms .large 76806 .exactZero (none)

def event76808 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16504⟩⟩) 0 ⟨6702⟩ 76807

def event76809 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16504⟩⟩) 1 ⟨16503⟩ 76804

def event76810 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16504⟩⟩) (.sum [.predecessor 0 76808 .coefficient, .predecessor 1 76809 .coefficient])

def exact76811RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact76811RawTermsValid :
    exact76811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76811 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16504⟩⟩) exact76811RawTerms .large 76810 .exactZero (none)

def event76812 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28932⟩⟩) 0 ⟨16504⟩ 76811

def event76813 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28932⟩⟩) 1 ⟨28931⟩ 76788

def event76814 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28932⟩⟩) (.product (.predecessor 0 76812 .coefficient) (.predecessor 1 76813 .coefficient) (⟨false, false, none, none, none⟩))

def event76815 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28932⟩⟩, .operator (⟨76811, 0⟩, ⟨76788, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28931⟩⟩]⟩, (1)⟩)

def event76816 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28932⟩⟩, .operator (⟨76811, 1⟩, ⟨76788, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28931⟩⟩]⟩, (-1)⟩)

def event76817 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28932⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28931⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28931⟩⟩) ⟨24473⟩ 76785)

def event76818 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28932⟩⟩, .relation 76817 0, ⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24473⟩⟩]⟩, (-1)⟩)

def exact76819RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28931⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24473⟩⟩]⟩, (-1)⟩]

theorem exact76819RawTermsValid :
    exact76819RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76819 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28932⟩⟩) exact76819RawTerms .large 76814 .exactZero (none)

def event76820 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17546⟩⟩) 0 ⟨16462⟩ 76777

def event76821 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17546⟩⟩) (.authority (.programFamilyFact))

def exact76822RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17546⟩⟩], []⟩, (1)⟩]

theorem exact76822RawTermsValid :
    exact76822RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76822 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17546⟩⟩) exact76822RawTerms (.finite 40) 76821 .exactZero (none)

def event76823 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17548⟩⟩) 0 ⟨6544⟩ 76799

def event76824 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17548⟩⟩) 1 ⟨17546⟩ 76822

def event76825 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17548⟩⟩) (.product (.predecessor 0 76823 .coefficient) (.predecessor 1 76824 .coefficient) (⟨false, true, none, none, some 1⟩))

def event76826 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17548⟩⟩, .operator (⟨76799, 0⟩, ⟨76822, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17546⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact76827RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17546⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact76827RawTermsValid :
    exact76827RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76827 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17548⟩⟩) exact76827RawTerms .large 76825 .exactZero (none)

def event76828 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6732⟩⟩) 0 ⟨6689⟩ 76781

def event76829 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6732⟩⟩) (.authority (.operator))

def exact76830RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩]⟩, (1)⟩]

theorem exact76830RawTermsValid :
    exact76830RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76830 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6732⟩⟩) exact76830RawTerms .large 76829 .exactZero (none)

def event76831 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17549⟩⟩) 0 ⟨6732⟩ 76830

def event76832 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17549⟩⟩) 1 ⟨17548⟩ 76827

def event76833 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17549⟩⟩) (.sum [.predecessor 0 76831 .coefficient, .predecessor 1 76832 .coefficient])

def exact76834RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17546⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact76834RawTermsValid :
    exact76834RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76834 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17549⟩⟩) exact76834RawTerms .large 76833 .exactZero (none)

def event76835 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28937⟩⟩) 0 ⟨17549⟩ 76834

def event76836 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28937⟩⟩) 1 ⟨28932⟩ 76819

def event76837 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28937⟩⟩) (.sum [.predecessor 0 76835 .coefficient, .predecessor 1 76836 .coefficient])

def exact76838RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28931⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24473⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17546⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact76838RawTermsValid :
    exact76838RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76838 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28937⟩⟩) exact76838RawTerms .large 76837 .exactZero (none)

def event76839 : Event := .preFoldPolynomial 76838 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28931⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24473⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17546⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact76840RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28931⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6732⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24473⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17546⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event76840 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28937⟩⟩) 76839 exact76840RawTerms .large 76837 .exactZero (none)

def event76841 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16462⟩⟩) ⟨⟨145⟩, ⟨53⟩, ⟨109⟩⟩ ⟨76683, 76841⟩

def event76842 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨22047⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22044⟩⟩]⟩) (1) 0 2 (.universal 76841 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22044⟩⟩]⟩) (none) 76840)

def event76843 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22047⟩⟩, .relation 76842 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6732⟩⟩]⟩, (1)⟩)

def event76844 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22047⟩⟩, .relation 76842 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28931⟩⟩]⟩, (-1)⟩)

def event76845 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22047⟩⟩, .relation 76842 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24473⟩⟩]⟩, (1)⟩)

def event76846 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22047⟩⟩, .relation 76842 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact76847RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28931⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6732⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24473⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact76847RawTermsValid :
    exact76847RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76847 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22047⟩⟩) exact76847RawTerms .large 76679 (.finite 1811303510016) (some (76681))

def event76848 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28934⟩⟩) 0 ⟨22047⟩ 76847

def event76849 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28934⟩⟩) 1 ⟨28933⟩ 76669

def event76850 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28934⟩⟩) (.sum [.predecessor 0 76848 .coefficient, .predecessor 1 76849 .coefficient])

def event76851 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28934⟩⟩, .operator (⟨76847, 0⟩, ⟨76669, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28931⟩⟩]⟩, (1)⟩)

def event76852 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28934⟩⟩, .operator (⟨76847, 2⟩, ⟨76669, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24473⟩⟩]⟩, (-1)⟩)

def event76853 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28934⟩⟩) (.sum [.result 76847 .summary, .result 76669 .summary])

def exact76854RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6732⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact76854RawTermsValid :
    exact76854RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76854 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28934⟩⟩) exact76854RawTerms .large 76850 (.finite 1292315010834812776448) (some (76853))

def event76855 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28935⟩⟩) 0 ⟨28934⟩ 76854

def event76856 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28935⟩⟩) 1 ⟨6670⟩ 5619

def event76857 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28935⟩⟩) (.product (.predecessor 0 76855 .coefficient) (.predecessor 1 76856 .coefficient) (⟨false, false, none, none, none⟩))

def event76858 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28935⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩) [⟨.result 5615 .coefficient, false, none⟩])

def event76859 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28935⟩⟩) (.product (.result 76854 .summary) (.transfer 76858) (⟨false, false, none, none, none⟩))

def event76860 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28935⟩⟩, .operator (⟨76854, 0⟩, ⟨5619, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩, (1)⟩)

def event76861 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28935⟩⟩, .operator (⟨76854, 1⟩, ⟨5619, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩, (-1)⟩)

def event76862 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28935⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6669⟩⟩) ⟨6606⟩ 5612)

def event76863 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28935⟩⟩, .relation 76862 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact76864RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact76864RawTermsValid :
    exact76864RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76864 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28935⟩⟩) exact76864RawTerms .large 76857 (.finite 4742816766803936246568583168) (some (76859))

def event76865 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24410⟩⟩) 0 ⟨6689⟩ 5477

def event76866 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24410⟩⟩) 1 ⟨24409⟩ 68181

def event76867 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24410⟩⟩) (.authority (.operator))

def exact76868RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24410⟩⟩]⟩, (1)⟩]

theorem exact76868RawTermsValid :
    exact76868RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76868 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24410⟩⟩) exact76868RawTerms .large 76867 .exactZero (none)

def event76869 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28714⟩⟩) 0 ⟨24410⟩ 76868

def event76870 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28714⟩⟩) (.authority (.operator))

def exact76871RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28714⟩⟩]⟩, (1)⟩]

theorem exact76871RawTermsValid :
    exact76871RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76871 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28714⟩⟩) exact76871RawTerms (.finite 8192) 76870 .exactZero (none)

def event76872 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28716⟩⟩) 0 ⟨25216⟩ 68465

def event76873 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28716⟩⟩) 1 ⟨28714⟩ 76871

def event76874 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28716⟩⟩) (.product (.predecessor 0 76872 .coefficient) (.predecessor 1 76873 .coefficient) (⟨false, false, none, none, none⟩))

def event76875 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28716⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28714⟩⟩]⟩) [⟨.result 76871 .coefficient, false, none⟩])

def event76876 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28716⟩⟩) (.product (.result 68465 .summary) (.transfer 76875) (⟨false, false, none, none, none⟩))

def event76877 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28716⟩⟩, .operator (⟨68465, 0⟩, ⟨76871, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28714⟩⟩]⟩, (1)⟩)

def event76878 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28716⟩⟩, .operator (⟨68465, 1⟩, ⟨76871, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28714⟩⟩]⟩, (-1)⟩)

def event76879 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28716⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28714⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28714⟩⟩) ⟨24410⟩ 76868)

def event76880 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28716⟩⟩, .relation 76879 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨24410⟩⟩]⟩, (-1)⟩)

def exact76881RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28714⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨24410⟩⟩]⟩, (-1)⟩]

theorem exact76881RawTermsValid :
    exact76881RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76881 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28716⟩⟩) exact76881RawTerms .large 76874 (.finite 1292270184133468094464) (some (76876))

def event76882 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21900⟩⟩) 0 ⟨16378⟩ 3241

def event76883 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21900⟩⟩) (.authority (.relationPreimageSource ⟨51⟩))

def exact76884RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21900⟩⟩]⟩, (1)⟩]

theorem exact76884RawTermsValid :
    exact76884RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76884 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21900⟩⟩) exact76884RawTerms (.finite 136065468) 76883 .exactZero (none)

def event76885 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21902⟩⟩) 0 ⟨21900⟩ 76884

def event76886 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21902⟩⟩) 1 ⟨2348⟩ 4

def event76887 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21902⟩⟩) (.scale (.predecessor 0 76885 .coefficient) (.value (.predecessor 1 76886 .coefficient)))

def exact76888RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21900⟩⟩]⟩, (1)⟩]

theorem exact76888RawTermsValid :
    exact76888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76888 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21902⟩⟩) exact76888RawTerms (.finite 136065468) 76887 .exactZero (none)

def event76889 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21903⟩⟩) 0 ⟨5535⟩ 65387

def event76890 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21903⟩⟩) 1 ⟨21902⟩ 76888

def event76891 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21903⟩⟩) (.product (.predecessor 0 76889 .coefficient) (.predecessor 1 76890 .coefficient) (⟨false, false, none, none, none⟩))

def event76892 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21903⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21900⟩⟩]⟩) [⟨.result 76884 .coefficient, false, none⟩])

def event76893 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21903⟩⟩) (.product (.result 65387 .summary) (.transfer 76892) (⟨false, false, none, none, none⟩))

def event76894 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21903⟩⟩, .operator (⟨65387, 0⟩, ⟨76888, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21900⟩⟩]⟩, (1)⟩)

def event76895 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21901⟩⟩)

def event76896 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event76897 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event76898 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event76899 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event76900 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event76901 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event76902 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event76903 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event76904 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 76903

def event76905 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 76901

def event76906 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 76904 .coefficient) (.value (.predecessor 1 76905 .coefficient)))

def event76907 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event76908 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 76907

def event76909 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 76899

def event76910 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 76908 .coefficient, .predecessor 1 76909 .coefficient])

def event76911 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event76912 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 76911

def event76913 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 76897

def event76914 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 76913 .coefficient))

def event76915 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event76916 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11949⟩⟩) 0 ⟨5530⟩ 76915

def event76917 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11949⟩⟩) (.authority (.programFamilyFact))

def exact76918RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11949⟩⟩], []⟩, (1)⟩]

theorem exact76918RawTermsValid :
    exact76918RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76918 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11949⟩⟩) exact76918RawTerms (.finite 36) 76917 .exactZero (none)

def event76919 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9710⟩⟩) 0 ⟨5530⟩ 76915

def event76920 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9710⟩⟩) (.authority (.programFamilyFact))

def exact76921RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩], []⟩, (1)⟩]

theorem exact76921RawTermsValid :
    exact76921RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76921 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9710⟩⟩) exact76921RawTerms (.finite 36) 76920 .exactZero (none)

def event76922 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11950⟩⟩) 0 ⟨9710⟩ 76921

def event76923 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11950⟩⟩) 1 ⟨11949⟩ 76918

def event76924 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11950⟩⟩) (.product (.predecessor 0 76922 .coefficient) (.predecessor 1 76923 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event76925 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11950⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], []⟩) [⟨.result 76921 .coefficient, true, some 1⟩, ⟨.result 76918 .coefficient, true, some 1⟩])

def event76926 : Event := .survivorFold (1) 76925

def exact76927RawTerms : List Term := []

theorem exact76927RawTermsValid :
    exact76927RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76927 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11950⟩⟩) exact76927RawTerms (.finite 1296) 76924 (.finite 1296) (some (76925))

def event76928 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11951⟩⟩) 0 ⟨11950⟩ 76927

def event76929 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11951⟩⟩) (.identity (.predecessor 0 76928 .coefficient))

def event76930 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11951⟩⟩) (.finite 1296)

def event76931 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16377⟩⟩) 0 ⟨11951⟩ 76930

def event76932 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16377⟩⟩) (.authority (.programFamilyFact))

def exact76933RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], []⟩, (1)⟩]

theorem exact76933RawTermsValid :
    exact76933RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76933 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16377⟩⟩) exact76933RawTerms (.finite 36) 76932 .exactZero (none)

def event76934 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16378⟩⟩) 0 ⟨16377⟩ 76933

def event76935 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16378⟩⟩) (.identity (.predecessor 0 76934 .coefficient))

def event76936 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16378⟩⟩) (.finite 36)

def event76937 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21900⟩⟩) 0 ⟨16378⟩ 76936

def event76938 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21900⟩⟩) (.authority (.relationPreimageSource ⟨51⟩))

def exact76939RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21900⟩⟩]⟩, (1)⟩]

theorem exact76939RawTermsValid :
    exact76939RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76939 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21900⟩⟩) exact76939RawTerms (.finite 136065468) 76938 .exactZero (none)

def event76940 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact76941RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact76941RawTermsValid :
    exact76941RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76941 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact76941RawTerms .large 76940 .exactZero (none)

def event76942 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21901⟩⟩) 0 ⟨6⟩ 76941

def event76943 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21901⟩⟩) 1 ⟨21900⟩ 76939

def event76944 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21901⟩⟩) (.product (.predecessor 0 76942 .coefficient) (.predecessor 1 76943 .coefficient) (⟨false, false, none, none, none⟩))

def event76945 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21901⟩⟩, .operator (⟨76941, 0⟩, ⟨76939, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21900⟩⟩]⟩, (1)⟩)

def exact76946RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21900⟩⟩]⟩, (1)⟩]

theorem exact76946RawTermsValid :
    exact76946RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76946 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21901⟩⟩) exact76946RawTerms .large 76944 .exactZero (none)

def event76947 : Event := .preFoldPolynomial 76946 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21900⟩⟩]⟩, (1)⟩] .exactZero none

def exact76948RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21900⟩⟩]⟩, (1)⟩]

def event76948 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21901⟩⟩) 76947 exact76948RawTerms .large 76944 .exactZero (none)

def event76949 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28720⟩⟩)

def event76950 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event76951 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event76952 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event76953 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event76954 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event76955 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event76956 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event76957 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event76958 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 76957

def event76959 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 76955

def event76960 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 76958 .coefficient) (.value (.predecessor 1 76959 .coefficient)))

def event76961 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event76962 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 76961

def event76963 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 76953

def event76964 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 76962 .coefficient, .predecessor 1 76963 .coefficient])

def event76965 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event76966 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 76965

def event76967 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 76951

def event76968 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 76967 .coefficient))

def event76969 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event76970 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11949⟩⟩) 0 ⟨5530⟩ 76969

def event76971 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11949⟩⟩) (.authority (.programFamilyFact))

def exact76972RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11949⟩⟩], []⟩, (1)⟩]

theorem exact76972RawTermsValid :
    exact76972RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76972 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11949⟩⟩) exact76972RawTerms (.finite 36) 76971 .exactZero (none)

def event76973 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9710⟩⟩) 0 ⟨5530⟩ 76969

def event76974 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9710⟩⟩) (.authority (.programFamilyFact))

def exact76975RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩], []⟩, (1)⟩]

theorem exact76975RawTermsValid :
    exact76975RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76975 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9710⟩⟩) exact76975RawTerms (.finite 36) 76974 .exactZero (none)

def event76976 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11950⟩⟩) 0 ⟨9710⟩ 76975

def event76977 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11950⟩⟩) 1 ⟨11949⟩ 76972

def event76978 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11950⟩⟩) (.product (.predecessor 0 76976 .coefficient) (.predecessor 1 76977 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event76979 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11950⟩⟩, .operator (⟨76975, 0⟩, ⟨76972, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], []⟩, (1)⟩)

def exact76980RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], []⟩, (1)⟩]

theorem exact76980RawTermsValid :
    exact76980RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76980 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11950⟩⟩) exact76980RawTerms (.finite 1296) 76978 .exactZero (none)

def event76981 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11951⟩⟩) 0 ⟨11950⟩ 76980

def event76982 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11951⟩⟩) (.identity (.predecessor 0 76981 .coefficient))

def event76983 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11951⟩⟩) (.finite 1296)

def event76984 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16377⟩⟩) 0 ⟨11951⟩ 76983

def event76985 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16377⟩⟩) (.authority (.programFamilyFact))

def exact76986RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], []⟩, (1)⟩]

theorem exact76986RawTermsValid :
    exact76986RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76986 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16377⟩⟩) exact76986RawTerms (.finite 36) 76985 .exactZero (none)

def event76987 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16378⟩⟩) 0 ⟨16377⟩ 76986

def event76988 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16378⟩⟩) (.identity (.predecessor 0 76987 .coefficient))

def event76989 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16378⟩⟩) (.finite 36)

def event76990 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24409⟩⟩) 0 ⟨16378⟩ 76989

def event76991 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24409⟩⟩) (.authority (.programFamilyFact))

def event76992 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24409⟩⟩) (.finite 3720)

def event76993 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event76994 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24410⟩⟩) 0 ⟨6689⟩ 76993

def event76995 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24410⟩⟩) 1 ⟨24409⟩ 76992

def event76996 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24410⟩⟩) (.authority (.operator))

def exact76997RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24410⟩⟩]⟩, (1)⟩]

theorem exact76997RawTermsValid :
    exact76997RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76997 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24410⟩⟩) exact76997RawTerms .large 76996 .exactZero (none)

def event76998 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28714⟩⟩) 0 ⟨24410⟩ 76997

def event76999 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28714⟩⟩) (.authority (.operator))

def exact77000RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28714⟩⟩]⟩, (1)⟩]

theorem exact77000RawTermsValid :
    exact77000RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77000 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28714⟩⟩) exact77000RawTerms (.finite 8192) 76999 .exactZero (none)

def event77001 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event77002 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event77003 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16417⟩⟩) 0 ⟨16378⟩ 76989

def event77004 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16417⟩⟩) 1 ⟨110⟩ 77002

def event77005 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16417⟩⟩) (.sum [.predecessor 0 77003 .coefficient, .predecessor 1 77004 .coefficient])

def event77006 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16417⟩⟩) (.finite 36)

def event77007 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16418⟩⟩) 0 ⟨16417⟩ 77006

def event77008 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16418⟩⟩) (.identity (.predecessor 0 77007 .coefficient))

def exact77009RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], []⟩, (1)⟩]

theorem exact77009RawTermsValid :
    exact77009RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77009 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16418⟩⟩) exact77009RawTerms (.finite 36) 77008 .exactZero (none)

def event77010 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact77011RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact77011RawTermsValid :
    exact77011RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77011 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact77011RawTerms .large 77010 .exactZero (none)

def event77012 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16419⟩⟩) 0 ⟨6544⟩ 77011

def event77013 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16419⟩⟩) 1 ⟨16418⟩ 77009

def event77014 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16419⟩⟩) (.product (.predecessor 0 77012 .coefficient) (.predecessor 1 77013 .coefficient) (⟨false, false, none, none, none⟩))

def event77015 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16419⟩⟩, .operator (⟨77011, 0⟩, ⟨77009, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact77016RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact77016RawTermsValid :
    exact77016RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77016 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16419⟩⟩) exact77016RawTerms .large 77014 .exactZero (none)

def event77017 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6701⟩⟩) 0 ⟨6689⟩ 76993

def event77018 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6701⟩⟩) (.authority (.operator))

def exact77019RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩]

theorem exact77019RawTermsValid :
    exact77019RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77019 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6701⟩⟩) exact77019RawTerms .large 77018 .exactZero (none)

def event77020 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16420⟩⟩) 0 ⟨6701⟩ 77019

def event77021 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16420⟩⟩) 1 ⟨16419⟩ 77016

def event77022 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16420⟩⟩) (.sum [.predecessor 0 77020 .coefficient, .predecessor 1 77021 .coefficient])

def exact77023RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77023RawTermsValid :
    exact77023RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77023 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16420⟩⟩) exact77023RawTerms .large 77022 .exactZero (none)

def event77024 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28715⟩⟩) 0 ⟨16420⟩ 77023

def event77025 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28715⟩⟩) 1 ⟨28714⟩ 77000

def event77026 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28715⟩⟩) (.product (.predecessor 0 77024 .coefficient) (.predecessor 1 77025 .coefficient) (⟨false, false, none, none, none⟩))

def event77027 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28715⟩⟩, .operator (⟨77023, 0⟩, ⟨77000, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28714⟩⟩]⟩, (1)⟩)

def event77028 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28715⟩⟩, .operator (⟨77023, 1⟩, ⟨77000, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28714⟩⟩]⟩, (-1)⟩)

def event77029 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28715⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28714⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28714⟩⟩) ⟨24410⟩ 76997)

def event77030 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28715⟩⟩, .relation 77029 0, ⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨24410⟩⟩]⟩, (-1)⟩)

def exact77031RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28714⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨24410⟩⟩]⟩, (-1)⟩]

theorem exact77031RawTermsValid :
    exact77031RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77031 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28715⟩⟩) exact77031RawTerms .large 77026 .exactZero (none)

def event77032 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18818⟩⟩) 0 ⟨16378⟩ 76989

def event77033 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18818⟩⟩) (.authority (.programFamilyFact))

def exact77034RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18818⟩⟩], []⟩, (1)⟩]

theorem exact77034RawTermsValid :
    exact77034RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77034 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18818⟩⟩) exact77034RawTerms (.finite 36) 77033 .exactZero (none)

def event77035 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18827⟩⟩) 0 ⟨6544⟩ 77011

def event77036 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18827⟩⟩) 1 ⟨18818⟩ 77034

def event77037 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18827⟩⟩) (.product (.predecessor 0 77035 .coefficient) (.predecessor 1 77036 .coefficient) (⟨false, true, none, none, some 1⟩))

def event77038 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18827⟩⟩, .operator (⟨77011, 0⟩, ⟨77034, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18818⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact77039RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18818⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact77039RawTermsValid :
    exact77039RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77039 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18827⟩⟩) exact77039RawTerms .large 77037 .exactZero (none)

def event77040 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6730⟩⟩) 0 ⟨6689⟩ 76993

def event77041 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6730⟩⟩) (.authority (.operator))

def exact77042RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩]⟩, (1)⟩]

theorem exact77042RawTermsValid :
    exact77042RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77042 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6730⟩⟩) exact77042RawTerms .large 77041 .exactZero (none)

def event77043 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18831⟩⟩) 0 ⟨6730⟩ 77042

def event77044 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18831⟩⟩) 1 ⟨18827⟩ 77039

def event77045 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18831⟩⟩) (.sum [.predecessor 0 77043 .coefficient, .predecessor 1 77044 .coefficient])

def exact77046RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18818⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77046RawTermsValid :
    exact77046RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77046 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18831⟩⟩) exact77046RawTerms .large 77045 .exactZero (none)

def event77047 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28720⟩⟩) 0 ⟨18831⟩ 77046

def event77048 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28720⟩⟩) 1 ⟨28715⟩ 77031

def event77049 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28720⟩⟩) (.sum [.predecessor 0 77047 .coefficient, .predecessor 1 77048 .coefficient])

def exact77050RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28714⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨24410⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18818⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact77050RawTermsValid :
    exact77050RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77050 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28720⟩⟩) exact77050RawTerms .large 77049 .exactZero (none)

def event77051 : Event := .preFoldPolynomial 77050 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28714⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨24410⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18818⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact77052RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28714⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6730⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨24410⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18818⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event77052 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28720⟩⟩) 77051 exact77052RawTerms .large 77049 .exactZero (none)

def event77053 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16378⟩⟩) ⟨⟨143⟩, ⟨51⟩, ⟨109⟩⟩ ⟨76895, 77053⟩

def event77054 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21903⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21900⟩⟩]⟩) (1) 0 2 (.universal 77053 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21900⟩⟩]⟩) (none) 77052)

def event77055 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21903⟩⟩, .relation 77054 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6730⟩⟩]⟩, (1)⟩)

def eventLeaf4800 : Array AnnotatedEvent := #[
  { event := event76800
    frameStart := 76737 },
  { event := event76801
    frameStart := 76737 },
  { event := event76802
    frameStart := 76737 },
  { event := event76803
    frameStart := 76737 },
  { event := event76804
    frameStart := 76737 },
  { event := event76805
    frameStart := 76737 },
  { event := event76806
    frameStart := 76737 },
  { event := event76807
    frameStart := 76737 },
  { event := event76808
    frameStart := 76737 },
  { event := event76809
    frameStart := 76737 },
  { event := event76810
    frameStart := 76737 },
  { event := event76811
    frameStart := 76737 },
  { event := event76812
    frameStart := 76737 },
  { event := event76813
    frameStart := 76737 },
  { event := event76814
    frameStart := 76737 },
  { event := event76815
    frameStart := 76737 }
]

def eventLeaf4801 : Array AnnotatedEvent := #[
  { event := event76816
    frameStart := 76737 },
  { event := event76817
    frameStart := 76737 },
  { event := event76818
    frameStart := 76737 },
  { event := event76819
    frameStart := 76737 },
  { event := event76820
    frameStart := 76737 },
  { event := event76821
    frameStart := 76737 },
  { event := event76822
    frameStart := 76737 },
  { event := event76823
    frameStart := 76737 },
  { event := event76824
    frameStart := 76737 },
  { event := event76825
    frameStart := 76737 },
  { event := event76826
    frameStart := 76737 },
  { event := event76827
    frameStart := 76737 },
  { event := event76828
    frameStart := 76737 },
  { event := event76829
    frameStart := 76737 },
  { event := event76830
    frameStart := 76737 },
  { event := event76831
    frameStart := 76737 }
]

def eventLeaf4802 : Array AnnotatedEvent := #[
  { event := event76832
    frameStart := 76737 },
  { event := event76833
    frameStart := 76737 },
  { event := event76834
    frameStart := 76737 },
  { event := event76835
    frameStart := 76737 },
  { event := event76836
    frameStart := 76737 },
  { event := event76837
    frameStart := 76737 },
  { event := event76838
    frameStart := 76737 },
  { event := event76839
    frameStart := 76737 },
  { event := event76840
    frameStart := 76737 },
  { event := event76841
    frameStart := 0 },
  { event := event76842
    frameStart := 0 },
  { event := event76843
    frameStart := 0 },
  { event := event76844
    frameStart := 0 },
  { event := event76845
    frameStart := 0 },
  { event := event76846
    frameStart := 0 },
  { event := event76847
    frameStart := 0 }
]

def eventLeaf4803 : Array AnnotatedEvent := #[
  { event := event76848
    frameStart := 0 },
  { event := event76849
    frameStart := 0 },
  { event := event76850
    frameStart := 0 },
  { event := event76851
    frameStart := 0 },
  { event := event76852
    frameStart := 0 },
  { event := event76853
    frameStart := 0 },
  { event := event76854
    frameStart := 0 },
  { event := event76855
    frameStart := 0 },
  { event := event76856
    frameStart := 0 },
  { event := event76857
    frameStart := 0 },
  { event := event76858
    frameStart := 0 },
  { event := event76859
    frameStart := 0 },
  { event := event76860
    frameStart := 0 },
  { event := event76861
    frameStart := 0 },
  { event := event76862
    frameStart := 0 },
  { event := event76863
    frameStart := 0 }
]

def eventLeaf4804 : Array AnnotatedEvent := #[
  { event := event76864
    frameStart := 0 },
  { event := event76865
    frameStart := 0 },
  { event := event76866
    frameStart := 0 },
  { event := event76867
    frameStart := 0 },
  { event := event76868
    frameStart := 0 },
  { event := event76869
    frameStart := 0 },
  { event := event76870
    frameStart := 0 },
  { event := event76871
    frameStart := 0 },
  { event := event76872
    frameStart := 0 },
  { event := event76873
    frameStart := 0 },
  { event := event76874
    frameStart := 0 },
  { event := event76875
    frameStart := 0 },
  { event := event76876
    frameStart := 0 },
  { event := event76877
    frameStart := 0 },
  { event := event76878
    frameStart := 0 },
  { event := event76879
    frameStart := 0 }
]

def eventLeaf4805 : Array AnnotatedEvent := #[
  { event := event76880
    frameStart := 0 },
  { event := event76881
    frameStart := 0 },
  { event := event76882
    frameStart := 0 },
  { event := event76883
    frameStart := 0 },
  { event := event76884
    frameStart := 0 },
  { event := event76885
    frameStart := 0 },
  { event := event76886
    frameStart := 0 },
  { event := event76887
    frameStart := 0 },
  { event := event76888
    frameStart := 0 },
  { event := event76889
    frameStart := 0 },
  { event := event76890
    frameStart := 0 },
  { event := event76891
    frameStart := 0 },
  { event := event76892
    frameStart := 0 },
  { event := event76893
    frameStart := 0 },
  { event := event76894
    frameStart := 0 },
  { event := event76895
    frameStart := 76895 }
]

def eventLeaf4806 : Array AnnotatedEvent := #[
  { event := event76896
    frameStart := 76895 },
  { event := event76897
    frameStart := 76895 },
  { event := event76898
    frameStart := 76895 },
  { event := event76899
    frameStart := 76895 },
  { event := event76900
    frameStart := 76895 },
  { event := event76901
    frameStart := 76895 },
  { event := event76902
    frameStart := 76895 },
  { event := event76903
    frameStart := 76895 },
  { event := event76904
    frameStart := 76895 },
  { event := event76905
    frameStart := 76895 },
  { event := event76906
    frameStart := 76895 },
  { event := event76907
    frameStart := 76895 },
  { event := event76908
    frameStart := 76895 },
  { event := event76909
    frameStart := 76895 },
  { event := event76910
    frameStart := 76895 },
  { event := event76911
    frameStart := 76895 }
]

def eventLeaf4807 : Array AnnotatedEvent := #[
  { event := event76912
    frameStart := 76895 },
  { event := event76913
    frameStart := 76895 },
  { event := event76914
    frameStart := 76895 },
  { event := event76915
    frameStart := 76895 },
  { event := event76916
    frameStart := 76895 },
  { event := event76917
    frameStart := 76895 },
  { event := event76918
    frameStart := 76895 },
  { event := event76919
    frameStart := 76895 },
  { event := event76920
    frameStart := 76895 },
  { event := event76921
    frameStart := 76895 },
  { event := event76922
    frameStart := 76895 },
  { event := event76923
    frameStart := 76895 },
  { event := event76924
    frameStart := 76895 },
  { event := event76925
    frameStart := 76895 },
  { event := event76926
    frameStart := 76895 },
  { event := event76927
    frameStart := 76895 }
]

def eventLeaf4808 : Array AnnotatedEvent := #[
  { event := event76928
    frameStart := 76895 },
  { event := event76929
    frameStart := 76895 },
  { event := event76930
    frameStart := 76895 },
  { event := event76931
    frameStart := 76895 },
  { event := event76932
    frameStart := 76895 },
  { event := event76933
    frameStart := 76895 },
  { event := event76934
    frameStart := 76895 },
  { event := event76935
    frameStart := 76895 },
  { event := event76936
    frameStart := 76895 },
  { event := event76937
    frameStart := 76895 },
  { event := event76938
    frameStart := 76895 },
  { event := event76939
    frameStart := 76895 },
  { event := event76940
    frameStart := 76895 },
  { event := event76941
    frameStart := 76895 },
  { event := event76942
    frameStart := 76895 },
  { event := event76943
    frameStart := 76895 }
]

def eventLeaf4809 : Array AnnotatedEvent := #[
  { event := event76944
    frameStart := 76895 },
  { event := event76945
    frameStart := 76895 },
  { event := event76946
    frameStart := 76895 },
  { event := event76947
    frameStart := 76895 },
  { event := event76948
    frameStart := 76895 },
  { event := event76949
    frameStart := 76949 },
  { event := event76950
    frameStart := 76949 },
  { event := event76951
    frameStart := 76949 },
  { event := event76952
    frameStart := 76949 },
  { event := event76953
    frameStart := 76949 },
  { event := event76954
    frameStart := 76949 },
  { event := event76955
    frameStart := 76949 },
  { event := event76956
    frameStart := 76949 },
  { event := event76957
    frameStart := 76949 },
  { event := event76958
    frameStart := 76949 },
  { event := event76959
    frameStart := 76949 }
]

def eventLeaf4810 : Array AnnotatedEvent := #[
  { event := event76960
    frameStart := 76949 },
  { event := event76961
    frameStart := 76949 },
  { event := event76962
    frameStart := 76949 },
  { event := event76963
    frameStart := 76949 },
  { event := event76964
    frameStart := 76949 },
  { event := event76965
    frameStart := 76949 },
  { event := event76966
    frameStart := 76949 },
  { event := event76967
    frameStart := 76949 },
  { event := event76968
    frameStart := 76949 },
  { event := event76969
    frameStart := 76949 },
  { event := event76970
    frameStart := 76949 },
  { event := event76971
    frameStart := 76949 },
  { event := event76972
    frameStart := 76949 },
  { event := event76973
    frameStart := 76949 },
  { event := event76974
    frameStart := 76949 },
  { event := event76975
    frameStart := 76949 }
]

def eventLeaf4811 : Array AnnotatedEvent := #[
  { event := event76976
    frameStart := 76949 },
  { event := event76977
    frameStart := 76949 },
  { event := event76978
    frameStart := 76949 },
  { event := event76979
    frameStart := 76949 },
  { event := event76980
    frameStart := 76949 },
  { event := event76981
    frameStart := 76949 },
  { event := event76982
    frameStart := 76949 },
  { event := event76983
    frameStart := 76949 },
  { event := event76984
    frameStart := 76949 },
  { event := event76985
    frameStart := 76949 },
  { event := event76986
    frameStart := 76949 },
  { event := event76987
    frameStart := 76949 },
  { event := event76988
    frameStart := 76949 },
  { event := event76989
    frameStart := 76949 },
  { event := event76990
    frameStart := 76949 },
  { event := event76991
    frameStart := 76949 }
]

def eventLeaf4812 : Array AnnotatedEvent := #[
  { event := event76992
    frameStart := 76949 },
  { event := event76993
    frameStart := 76949 },
  { event := event76994
    frameStart := 76949 },
  { event := event76995
    frameStart := 76949 },
  { event := event76996
    frameStart := 76949 },
  { event := event76997
    frameStart := 76949 },
  { event := event76998
    frameStart := 76949 },
  { event := event76999
    frameStart := 76949 },
  { event := event77000
    frameStart := 76949 },
  { event := event77001
    frameStart := 76949 },
  { event := event77002
    frameStart := 76949 },
  { event := event77003
    frameStart := 76949 },
  { event := event77004
    frameStart := 76949 },
  { event := event77005
    frameStart := 76949 },
  { event := event77006
    frameStart := 76949 },
  { event := event77007
    frameStart := 76949 }
]

def eventLeaf4813 : Array AnnotatedEvent := #[
  { event := event77008
    frameStart := 76949 },
  { event := event77009
    frameStart := 76949 },
  { event := event77010
    frameStart := 76949 },
  { event := event77011
    frameStart := 76949 },
  { event := event77012
    frameStart := 76949 },
  { event := event77013
    frameStart := 76949 },
  { event := event77014
    frameStart := 76949 },
  { event := event77015
    frameStart := 76949 },
  { event := event77016
    frameStart := 76949 },
  { event := event77017
    frameStart := 76949 },
  { event := event77018
    frameStart := 76949 },
  { event := event77019
    frameStart := 76949 },
  { event := event77020
    frameStart := 76949 },
  { event := event77021
    frameStart := 76949 },
  { event := event77022
    frameStart := 76949 },
  { event := event77023
    frameStart := 76949 }
]

def eventLeaf4814 : Array AnnotatedEvent := #[
  { event := event77024
    frameStart := 76949 },
  { event := event77025
    frameStart := 76949 },
  { event := event77026
    frameStart := 76949 },
  { event := event77027
    frameStart := 76949 },
  { event := event77028
    frameStart := 76949 },
  { event := event77029
    frameStart := 76949 },
  { event := event77030
    frameStart := 76949 },
  { event := event77031
    frameStart := 76949 },
  { event := event77032
    frameStart := 76949 },
  { event := event77033
    frameStart := 76949 },
  { event := event77034
    frameStart := 76949 },
  { event := event77035
    frameStart := 76949 },
  { event := event77036
    frameStart := 76949 },
  { event := event77037
    frameStart := 76949 },
  { event := event77038
    frameStart := 76949 },
  { event := event77039
    frameStart := 76949 }
]

def eventLeaf4815 : Array AnnotatedEvent := #[
  { event := event77040
    frameStart := 76949 },
  { event := event77041
    frameStart := 76949 },
  { event := event77042
    frameStart := 76949 },
  { event := event77043
    frameStart := 76949 },
  { event := event77044
    frameStart := 76949 },
  { event := event77045
    frameStart := 76949 },
  { event := event77046
    frameStart := 76949 },
  { event := event77047
    frameStart := 76949 },
  { event := event77048
    frameStart := 76949 },
  { event := event77049
    frameStart := 76949 },
  { event := event77050
    frameStart := 76949 },
  { event := event77051
    frameStart := 76949 },
  { event := event77052
    frameStart := 76949 },
  { event := event77053
    frameStart := 0 },
  { event := event77054
    frameStart := 0 },
  { event := event77055
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events300
