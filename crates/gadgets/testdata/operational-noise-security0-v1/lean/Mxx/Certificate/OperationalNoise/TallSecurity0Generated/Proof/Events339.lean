import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events339

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact86784RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩, (1)⟩]

theorem exact86784RawTermsValid :
    exact86784RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86784 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19241⟩⟩) exact86784RawTerms .large 86782 .exactZero (none)

def event86785 : Event := .preFoldPolynomial 86784 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩, (1)⟩] .exactZero none

def exact86786RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩, (1)⟩]

def event86786 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19241⟩⟩) 86785 exact86786RawTerms .large 86782 .exactZero (none)

def event86787 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25300⟩⟩)

def event86788 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event86789 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event86790 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event86791 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event86792 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event86793 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event86794 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event86795 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event86796 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 86795

def event86797 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 86793

def event86798 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 86796 .coefficient) (.value (.predecessor 1 86797 .coefficient)))

def event86799 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event86800 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 86799

def event86801 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 86791

def event86802 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 86800 .coefficient, .predecessor 1 86801 .coefficient])

def event86803 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event86804 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 86803

def event86805 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 86789

def event86806 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 86805 .coefficient))

def event86807 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event86808 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11133⟩⟩) 0 ⟨5536⟩ 86807

def event86809 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11133⟩⟩) (.authority (.programFamilyFact))

def exact86810RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩], []⟩, (1)⟩]

theorem exact86810RawTermsValid :
    exact86810RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86810 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11133⟩⟩) exact86810RawTerms (.finite 6) 86809 .exactZero (none)

def event86811 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12163⟩⟩) 0 ⟨5536⟩ 86807

def event86812 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12163⟩⟩) (.authority (.programFamilyFact))

def exact86813RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩, (1)⟩]

theorem exact86813RawTermsValid :
    exact86813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86813 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12163⟩⟩) exact86813RawTerms (.finite 6) 86812 .exactZero (none)

def event86814 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12164⟩⟩) 0 ⟨12163⟩ 86813

def event86815 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12164⟩⟩) 1 ⟨11133⟩ 86810

def event86816 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12164⟩⟩) (.product (.predecessor 0 86814 .coefficient) (.predecessor 1 86815 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event86817 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12164⟩⟩, .operator (⟨86813, 0⟩, ⟨86810, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩, (1)⟩)

def exact86818RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩, (1)⟩]

theorem exact86818RawTermsValid :
    exact86818RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86818 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12164⟩⟩) exact86818RawTerms (.finite 36) 86816 .exactZero (none)

def event86819 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12165⟩⟩) 0 ⟨12164⟩ 86818

def event86820 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12165⟩⟩) (.identity (.predecessor 0 86819 .coefficient))

def event86821 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12165⟩⟩) (.finite 36)

def event86822 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23163⟩⟩) 0 ⟨12165⟩ 86821

def event86823 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23163⟩⟩) (.authority (.programFamilyFact))

def event86824 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23163⟩⟩) (.finite 3720)

def event86825 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event86826 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23164⟩⟩) 0 ⟨6689⟩ 86825

def event86827 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23164⟩⟩) 1 ⟨23163⟩ 86824

def event86828 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23164⟩⟩) (.authority (.operator))

def exact86829RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23164⟩⟩]⟩, (1)⟩]

theorem exact86829RawTermsValid :
    exact86829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86829 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23164⟩⟩) exact86829RawTerms .large 86828 .exactZero (none)

def event86830 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25296⟩⟩) 0 ⟨23164⟩ 86829

def event86831 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25296⟩⟩) (.authority (.operator))

def exact86832RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩, (1)⟩]

theorem exact86832RawTermsValid :
    exact86832RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86832 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25296⟩⟩) exact86832RawTerms (.finite 8192) 86831 .exactZero (none)

def event86833 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event86834 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event86835 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12270⟩⟩) 0 ⟨12165⟩ 86821

def event86836 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12270⟩⟩) 1 ⟨110⟩ 86834

def event86837 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12270⟩⟩) (.sum [.predecessor 0 86835 .coefficient, .predecessor 1 86836 .coefficient])

def event86838 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12270⟩⟩) (.finite 36)

def event86839 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12271⟩⟩) 0 ⟨12270⟩ 86838

def event86840 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12271⟩⟩) (.identity (.predecessor 0 86839 .coefficient))

def exact86841RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩, (1)⟩]

theorem exact86841RawTermsValid :
    exact86841RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86841 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12271⟩⟩) exact86841RawTerms (.finite 36) 86840 .exactZero (none)

def event86842 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact86843RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact86843RawTermsValid :
    exact86843RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86843 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact86843RawTerms .large 86842 .exactZero (none)

def event86844 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12272⟩⟩) 0 ⟨6544⟩ 86843

def event86845 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12272⟩⟩) 1 ⟨12271⟩ 86841

def event86846 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12272⟩⟩) (.product (.predecessor 0 86844 .coefficient) (.predecessor 1 86845 .coefficient) (⟨false, false, none, none, none⟩))

def event86847 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12272⟩⟩, .operator (⟨86843, 0⟩, ⟨86841, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact86848RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact86848RawTermsValid :
    exact86848RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86848 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12272⟩⟩) exact86848RawTerms .large 86846 .exactZero (none)

def event86849 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 86825

def event86850 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact86851RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact86851RawTermsValid :
    exact86851RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86851 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact86851RawTerms .large 86850 .exactZero (none)

def event86852 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6775⟩⟩) 0 ⟨6757⟩ 86851

def event86853 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6775⟩⟩) (.identity (.predecessor 0 86852 .coefficient))

def exact86854RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩]

theorem exact86854RawTermsValid :
    exact86854RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86854 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6775⟩⟩) exact86854RawTerms .large 86853 .exactZero (none)

def event86855 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7840⟩⟩) 0 ⟨6775⟩ 86854

def event86856 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7840⟩⟩) (.authority (.operator))

def exact86857RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩]

theorem exact86857RawTermsValid :
    exact86857RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86857 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7840⟩⟩) exact86857RawTerms (.finite 8192) 86856 .exactZero (none)

def event86858 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7841⟩⟩) 0 ⟨7840⟩ 86857

def event86859 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7841⟩⟩) 1 ⟨2348⟩ 86791

def event86860 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7841⟩⟩) (.scale (.predecessor 0 86858 .coefficient) (.value (.predecessor 1 86859 .coefficient)))

def exact86861RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩]

theorem exact86861RawTermsValid :
    exact86861RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86861 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7841⟩⟩) exact86861RawTerms (.finite 8192) 86860 .exactZero (none)

def event86862 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6792⟩⟩) 0 ⟨6757⟩ 86851

def event86863 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6792⟩⟩) (.identity (.predecessor 0 86862 .coefficient))

def exact86864RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩]⟩, (1)⟩]

theorem exact86864RawTermsValid :
    exact86864RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86864 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6792⟩⟩) exact86864RawTerms .large 86863 .exactZero (none)

def event86865 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7842⟩⟩) 0 ⟨6792⟩ 86864

def event86866 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7842⟩⟩) 1 ⟨7841⟩ 86861

def event86867 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7842⟩⟩) (.product (.predecessor 0 86865 .coefficient) (.predecessor 1 86866 .coefficient) (⟨false, false, none, none, none⟩))

def event86868 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7842⟩⟩, .operator (⟨86864, 0⟩, ⟨86861, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩)

def exact86869RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩]

theorem exact86869RawTermsValid :
    exact86869RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86869 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7842⟩⟩) exact86869RawTerms .large 86867 .exactZero (none)

def event86870 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12273⟩⟩) 0 ⟨7842⟩ 86869

def event86871 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12273⟩⟩) 1 ⟨12272⟩ 86848

def event86872 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12273⟩⟩) (.sum [.predecessor 0 86870 .coefficient, .predecessor 1 86871 .coefficient])

def exact86873RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86873RawTermsValid :
    exact86873RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86873 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12273⟩⟩) exact86873RawTerms .large 86872 .exactZero (none)

def event86874 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25299⟩⟩) 0 ⟨12273⟩ 86873

def event86875 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25299⟩⟩) 1 ⟨25296⟩ 86832

def event86876 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25299⟩⟩) (.product (.predecessor 0 86874 .coefficient) (.predecessor 1 86875 .coefficient) (⟨false, false, none, none, none⟩))

def event86877 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25299⟩⟩, .operator (⟨86873, 0⟩, ⟨86832, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩, (1)⟩)

def event86878 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25299⟩⟩, .operator (⟨86873, 1⟩, ⟨86832, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩, (-1)⟩)

def event86879 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25299⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25296⟩⟩) ⟨23164⟩ 86829)

def event86880 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25299⟩⟩, .relation 86879 0, ⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨23164⟩⟩]⟩, (-1)⟩)

def exact86881RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨23164⟩⟩]⟩, (-1)⟩]

theorem exact86881RawTermsValid :
    exact86881RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86881 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25299⟩⟩) exact86881RawTerms .large 86876 .exactZero (none)

def event86882 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15422⟩⟩) 0 ⟨12165⟩ 86821

def event86883 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15422⟩⟩) (.authority (.programFamilyFact))

def exact86884RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], []⟩, (1)⟩]

theorem exact86884RawTermsValid :
    exact86884RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86884 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15422⟩⟩) exact86884RawTerms (.finite 6) 86883 .exactZero (none)

def event86885 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15424⟩⟩) 0 ⟨6544⟩ 86843

def event86886 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15424⟩⟩) 1 ⟨15422⟩ 86884

def event86887 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15424⟩⟩) (.product (.predecessor 0 86885 .coefficient) (.predecessor 1 86886 .coefficient) (⟨false, true, none, none, some 1⟩))

def event86888 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15424⟩⟩, .operator (⟨86843, 0⟩, ⟨86884, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact86889RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact86889RawTermsValid :
    exact86889RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86889 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15424⟩⟩) exact86889RawTerms .large 86887 .exactZero (none)

def event86890 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6693⟩⟩) 0 ⟨6689⟩ 86825

def event86891 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6693⟩⟩) (.authority (.operator))

def exact86892RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩]

theorem exact86892RawTermsValid :
    exact86892RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86892 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6693⟩⟩) exact86892RawTerms .large 86891 .exactZero (none)

def event86893 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15425⟩⟩) 0 ⟨6693⟩ 86892

def event86894 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15425⟩⟩) 1 ⟨15424⟩ 86889

def event86895 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15425⟩⟩) (.sum [.predecessor 0 86893 .coefficient, .predecessor 1 86894 .coefficient])

def exact86896RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86896RawTermsValid :
    exact86896RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86896 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15425⟩⟩) exact86896RawTerms .large 86895 .exactZero (none)

def event86897 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25300⟩⟩) 0 ⟨15425⟩ 86896

def event86898 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25300⟩⟩) 1 ⟨25299⟩ 86881

def event86899 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25300⟩⟩) (.sum [.predecessor 0 86897 .coefficient, .predecessor 1 86898 .coefficient])

def exact86900RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨23164⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86900RawTermsValid :
    exact86900RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86900 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25300⟩⟩) exact86900RawTerms .large 86899 .exactZero (none)

def event86901 : Event := .preFoldPolynomial 86900 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨23164⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact86902RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨23164⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event86902 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25300⟩⟩) 86901 exact86902RawTerms .large 86899 .exactZero (none)

def event86903 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨12165⟩⟩) ⟨⟨106⟩, ⟨10⟩, ⟨109⟩⟩ ⟨86739, 86903⟩

def event86904 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19243⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩) (1) 0 2 (.universal 86903 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19240⟩⟩]⟩) (none) 86902)

def event86905 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19243⟩⟩, .relation 86904 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩)

def event86906 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19243⟩⟩, .relation 86904 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩, (-1)⟩)

def event86907 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19243⟩⟩, .relation 86904 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨23164⟩⟩]⟩, (1)⟩)

def event86908 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19243⟩⟩, .relation 86904 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact86909RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨23164⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86909RawTermsValid :
    exact86909RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86909 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19243⟩⟩) exact86909RawTerms .large 86735 (.finite 1811303510016) (some (86737))

def event86910 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25298⟩⟩) 0 ⟨19243⟩ 86909

def event86911 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25298⟩⟩) 1 ⟨25297⟩ 86725

def event86912 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25298⟩⟩) (.sum [.predecessor 0 86910 .coefficient, .predecessor 1 86911 .coefficient])

def event86913 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25298⟩⟩, .operator (⟨86909, 2⟩, ⟨86725, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨23164⟩⟩]⟩, (-1)⟩)

def event86914 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25298⟩⟩, .operator (⟨86909, 1⟩, ⟨86725, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩, (1)⟩)

def event86915 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25298⟩⟩) (.sum [.result 86909 .summary, .result 86725 .summary])

def exact86916RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86916RawTermsValid :
    exact86916RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86916 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25298⟩⟩) exact86916RawTerms .large 86912 (.finite 352024077676544) (some (86915))

def event86917 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27000⟩⟩) 0 ⟨25298⟩ 86916

def event86918 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27000⟩⟩) 1 ⟨26998⟩ 86641

def event86919 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27000⟩⟩) (.product (.predecessor 0 86917 .coefficient) (.predecessor 1 86918 .coefficient) (⟨false, false, none, none, none⟩))

def event86920 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27000⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26998⟩⟩]⟩) [⟨.result 86641 .coefficient, false, none⟩])

def event86921 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27000⟩⟩) (.product (.result 86916 .summary) (.transfer 86920) (⟨false, false, none, none, none⟩))

def event86922 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27000⟩⟩, .operator (⟨86916, 0⟩, ⟨86641, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26998⟩⟩]⟩, (1)⟩)

def event86923 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27000⟩⟩, .operator (⟨86916, 1⟩, ⟨86641, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26998⟩⟩]⟩, (-1)⟩)

def event86924 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27000⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26998⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26998⟩⟩) ⟨23910⟩ 86638)

def event86925 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27000⟩⟩, .relation 86924 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨23910⟩⟩]⟩, (-1)⟩)

def exact86926RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26998⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨23910⟩⟩]⟩, (-1)⟩]

theorem exact86926RawTermsValid :
    exact86926RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86926 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27000⟩⟩) exact86926RawTerms .large 86919 (.finite 1291933997458159304704) (some (86921))

def event86927 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20824⟩⟩) 0 ⟨15423⟩ 4167

def event86928 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20824⟩⟩) (.authority (.relationPreimageSource ⟨35⟩))

def exact86929RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20824⟩⟩]⟩, (1)⟩]

theorem exact86929RawTermsValid :
    exact86929RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86929 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20824⟩⟩) exact86929RawTerms (.finite 136065468) 86928 .exactZero (none)

def event86930 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20826⟩⟩) 0 ⟨20824⟩ 86929

def event86931 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20826⟩⟩) 1 ⟨2348⟩ 4

def event86932 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20826⟩⟩) (.scale (.predecessor 0 86930 .coefficient) (.value (.predecessor 1 86931 .coefficient)))

def exact86933RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20824⟩⟩]⟩, (1)⟩]

theorem exact86933RawTermsValid :
    exact86933RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86933 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20826⟩⟩) exact86933RawTerms (.finite 136065468) 86932 .exactZero (none)

def event86934 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20827⟩⟩) 0 ⟨5541⟩ 80012

def event86935 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20827⟩⟩) 1 ⟨20826⟩ 86933

def event86936 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20827⟩⟩) (.product (.predecessor 0 86934 .coefficient) (.predecessor 1 86935 .coefficient) (⟨false, false, none, none, none⟩))

def event86937 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20827⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20824⟩⟩]⟩) [⟨.result 86929 .coefficient, false, none⟩])

def event86938 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20827⟩⟩) (.product (.result 80012 .summary) (.transfer 86937) (⟨false, false, none, none, none⟩))

def event86939 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20827⟩⟩, .operator (⟨80012, 0⟩, ⟨86933, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20824⟩⟩]⟩, (1)⟩)

def event86940 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20825⟩⟩)

def event86941 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event86942 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event86943 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event86944 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event86945 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event86946 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event86947 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event86948 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event86949 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 86948

def event86950 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 86946

def event86951 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 86949 .coefficient) (.value (.predecessor 1 86950 .coefficient)))

def event86952 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event86953 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 86952

def event86954 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 86944

def event86955 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 86953 .coefficient, .predecessor 1 86954 .coefficient])

def event86956 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event86957 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 86956

def event86958 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 86942

def event86959 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 86958 .coefficient))

def event86960 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event86961 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11133⟩⟩) 0 ⟨5536⟩ 86960

def event86962 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11133⟩⟩) (.authority (.programFamilyFact))

def exact86963RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩], []⟩, (1)⟩]

theorem exact86963RawTermsValid :
    exact86963RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86963 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11133⟩⟩) exact86963RawTerms (.finite 6) 86962 .exactZero (none)

def event86964 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12163⟩⟩) 0 ⟨5536⟩ 86960

def event86965 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12163⟩⟩) (.authority (.programFamilyFact))

def exact86966RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩, (1)⟩]

theorem exact86966RawTermsValid :
    exact86966RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86966 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12163⟩⟩) exact86966RawTerms (.finite 6) 86965 .exactZero (none)

def event86967 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12164⟩⟩) 0 ⟨12163⟩ 86966

def event86968 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12164⟩⟩) 1 ⟨11133⟩ 86963

def event86969 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12164⟩⟩) (.product (.predecessor 0 86967 .coefficient) (.predecessor 1 86968 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event86970 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12164⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩) [⟨.result 86966 .coefficient, true, some 1⟩, ⟨.result 86963 .coefficient, true, some 1⟩])

def event86971 : Event := .survivorFold (1) 86970

def exact86972RawTerms : List Term := []

theorem exact86972RawTermsValid :
    exact86972RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86972 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12164⟩⟩) exact86972RawTerms (.finite 36) 86969 (.finite 36) (some (86970))

def event86973 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12165⟩⟩) 0 ⟨12164⟩ 86972

def event86974 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12165⟩⟩) (.identity (.predecessor 0 86973 .coefficient))

def event86975 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12165⟩⟩) (.finite 36)

def event86976 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15422⟩⟩) 0 ⟨12165⟩ 86975

def event86977 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15422⟩⟩) (.authority (.programFamilyFact))

def exact86978RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], []⟩, (1)⟩]

theorem exact86978RawTermsValid :
    exact86978RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86978 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15422⟩⟩) exact86978RawTerms (.finite 6) 86977 .exactZero (none)

def event86979 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15423⟩⟩) 0 ⟨15422⟩ 86978

def event86980 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15423⟩⟩) (.identity (.predecessor 0 86979 .coefficient))

def event86981 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15423⟩⟩) (.finite 6)

def event86982 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20824⟩⟩) 0 ⟨15423⟩ 86981

def event86983 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20824⟩⟩) (.authority (.relationPreimageSource ⟨35⟩))

def exact86984RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20824⟩⟩]⟩, (1)⟩]

theorem exact86984RawTermsValid :
    exact86984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86984 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20824⟩⟩) exact86984RawTerms (.finite 136065468) 86983 .exactZero (none)

def event86985 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact86986RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact86986RawTermsValid :
    exact86986RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86986 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact86986RawTerms .large 86985 .exactZero (none)

def event86987 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20825⟩⟩) 0 ⟨6⟩ 86986

def event86988 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20825⟩⟩) 1 ⟨20824⟩ 86984

def event86989 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20825⟩⟩) (.product (.predecessor 0 86987 .coefficient) (.predecessor 1 86988 .coefficient) (⟨false, false, none, none, none⟩))

def event86990 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20825⟩⟩, .operator (⟨86986, 0⟩, ⟨86984, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20824⟩⟩]⟩, (1)⟩)

def exact86991RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20824⟩⟩]⟩, (1)⟩]

theorem exact86991RawTermsValid :
    exact86991RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86991 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20825⟩⟩) exact86991RawTerms .large 86989 .exactZero (none)

def event86992 : Event := .preFoldPolynomial 86991 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20824⟩⟩]⟩, (1)⟩] .exactZero none

def exact86993RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20824⟩⟩]⟩, (1)⟩]

def event86993 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20825⟩⟩) 86992 exact86993RawTerms .large 86989 .exactZero (none)

def event86994 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27003⟩⟩)

def event86995 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event86996 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event86997 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event86998 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event86999 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event87000 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event87001 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event87002 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event87003 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 87002

def event87004 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 87000

def event87005 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 87003 .coefficient) (.value (.predecessor 1 87004 .coefficient)))

def event87006 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event87007 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 87006

def event87008 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 86998

def event87009 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 87007 .coefficient, .predecessor 1 87008 .coefficient])

def event87010 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event87011 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 87010

def event87012 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 86996

def event87013 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 87012 .coefficient))

def event87014 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event87015 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11133⟩⟩) 0 ⟨5536⟩ 87014

def event87016 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11133⟩⟩) (.authority (.programFamilyFact))

def exact87017RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩], []⟩, (1)⟩]

theorem exact87017RawTermsValid :
    exact87017RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87017 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11133⟩⟩) exact87017RawTerms (.finite 6) 87016 .exactZero (none)

def event87018 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12163⟩⟩) 0 ⟨5536⟩ 87014

def event87019 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12163⟩⟩) (.authority (.programFamilyFact))

def exact87020RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩, (1)⟩]

theorem exact87020RawTermsValid :
    exact87020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87020 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12163⟩⟩) exact87020RawTerms (.finite 6) 87019 .exactZero (none)

def event87021 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12164⟩⟩) 0 ⟨12163⟩ 87020

def event87022 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12164⟩⟩) 1 ⟨11133⟩ 87017

def event87023 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12164⟩⟩) (.product (.predecessor 0 87021 .coefficient) (.predecessor 1 87022 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event87024 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12164⟩⟩, .operator (⟨87020, 0⟩, ⟨87017, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩, (1)⟩)

def exact87025RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩, (1)⟩]

theorem exact87025RawTermsValid :
    exact87025RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87025 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12164⟩⟩) exact87025RawTerms (.finite 36) 87023 .exactZero (none)

def event87026 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12165⟩⟩) 0 ⟨12164⟩ 87025

def event87027 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12165⟩⟩) (.identity (.predecessor 0 87026 .coefficient))

def event87028 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12165⟩⟩) (.finite 36)

def event87029 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15422⟩⟩) 0 ⟨12165⟩ 87028

def event87030 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15422⟩⟩) (.authority (.programFamilyFact))

def exact87031RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], []⟩, (1)⟩]

theorem exact87031RawTermsValid :
    exact87031RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87031 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15422⟩⟩) exact87031RawTerms (.finite 6) 87030 .exactZero (none)

def event87032 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15423⟩⟩) 0 ⟨15422⟩ 87031

def event87033 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15423⟩⟩) (.identity (.predecessor 0 87032 .coefficient))

def event87034 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15423⟩⟩) (.finite 6)

def event87035 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23908⟩⟩) 0 ⟨15423⟩ 87034

def event87036 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23908⟩⟩) (.authority (.programFamilyFact))

def event87037 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23908⟩⟩) (.finite 3720)

def event87038 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event87039 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23910⟩⟩) 0 ⟨6689⟩ 87038

def eventLeaf5424 : Array AnnotatedEvent := #[
  { event := event86784
    frameStart := 86739 },
  { event := event86785
    frameStart := 86739 },
  { event := event86786
    frameStart := 86739 },
  { event := event86787
    frameStart := 86787 },
  { event := event86788
    frameStart := 86787 },
  { event := event86789
    frameStart := 86787 },
  { event := event86790
    frameStart := 86787 },
  { event := event86791
    frameStart := 86787 },
  { event := event86792
    frameStart := 86787 },
  { event := event86793
    frameStart := 86787 },
  { event := event86794
    frameStart := 86787 },
  { event := event86795
    frameStart := 86787 },
  { event := event86796
    frameStart := 86787 },
  { event := event86797
    frameStart := 86787 },
  { event := event86798
    frameStart := 86787 },
  { event := event86799
    frameStart := 86787 }
]

def eventLeaf5425 : Array AnnotatedEvent := #[
  { event := event86800
    frameStart := 86787 },
  { event := event86801
    frameStart := 86787 },
  { event := event86802
    frameStart := 86787 },
  { event := event86803
    frameStart := 86787 },
  { event := event86804
    frameStart := 86787 },
  { event := event86805
    frameStart := 86787 },
  { event := event86806
    frameStart := 86787 },
  { event := event86807
    frameStart := 86787 },
  { event := event86808
    frameStart := 86787 },
  { event := event86809
    frameStart := 86787 },
  { event := event86810
    frameStart := 86787 },
  { event := event86811
    frameStart := 86787 },
  { event := event86812
    frameStart := 86787 },
  { event := event86813
    frameStart := 86787 },
  { event := event86814
    frameStart := 86787 },
  { event := event86815
    frameStart := 86787 }
]

def eventLeaf5426 : Array AnnotatedEvent := #[
  { event := event86816
    frameStart := 86787 },
  { event := event86817
    frameStart := 86787 },
  { event := event86818
    frameStart := 86787 },
  { event := event86819
    frameStart := 86787 },
  { event := event86820
    frameStart := 86787 },
  { event := event86821
    frameStart := 86787 },
  { event := event86822
    frameStart := 86787 },
  { event := event86823
    frameStart := 86787 },
  { event := event86824
    frameStart := 86787 },
  { event := event86825
    frameStart := 86787 },
  { event := event86826
    frameStart := 86787 },
  { event := event86827
    frameStart := 86787 },
  { event := event86828
    frameStart := 86787 },
  { event := event86829
    frameStart := 86787 },
  { event := event86830
    frameStart := 86787 },
  { event := event86831
    frameStart := 86787 }
]

def eventLeaf5427 : Array AnnotatedEvent := #[
  { event := event86832
    frameStart := 86787 },
  { event := event86833
    frameStart := 86787 },
  { event := event86834
    frameStart := 86787 },
  { event := event86835
    frameStart := 86787 },
  { event := event86836
    frameStart := 86787 },
  { event := event86837
    frameStart := 86787 },
  { event := event86838
    frameStart := 86787 },
  { event := event86839
    frameStart := 86787 },
  { event := event86840
    frameStart := 86787 },
  { event := event86841
    frameStart := 86787 },
  { event := event86842
    frameStart := 86787 },
  { event := event86843
    frameStart := 86787 },
  { event := event86844
    frameStart := 86787 },
  { event := event86845
    frameStart := 86787 },
  { event := event86846
    frameStart := 86787 },
  { event := event86847
    frameStart := 86787 }
]

def eventLeaf5428 : Array AnnotatedEvent := #[
  { event := event86848
    frameStart := 86787 },
  { event := event86849
    frameStart := 86787 },
  { event := event86850
    frameStart := 86787 },
  { event := event86851
    frameStart := 86787 },
  { event := event86852
    frameStart := 86787 },
  { event := event86853
    frameStart := 86787 },
  { event := event86854
    frameStart := 86787 },
  { event := event86855
    frameStart := 86787 },
  { event := event86856
    frameStart := 86787 },
  { event := event86857
    frameStart := 86787 },
  { event := event86858
    frameStart := 86787 },
  { event := event86859
    frameStart := 86787 },
  { event := event86860
    frameStart := 86787 },
  { event := event86861
    frameStart := 86787 },
  { event := event86862
    frameStart := 86787 },
  { event := event86863
    frameStart := 86787 }
]

def eventLeaf5429 : Array AnnotatedEvent := #[
  { event := event86864
    frameStart := 86787 },
  { event := event86865
    frameStart := 86787 },
  { event := event86866
    frameStart := 86787 },
  { event := event86867
    frameStart := 86787 },
  { event := event86868
    frameStart := 86787 },
  { event := event86869
    frameStart := 86787 },
  { event := event86870
    frameStart := 86787 },
  { event := event86871
    frameStart := 86787 },
  { event := event86872
    frameStart := 86787 },
  { event := event86873
    frameStart := 86787 },
  { event := event86874
    frameStart := 86787 },
  { event := event86875
    frameStart := 86787 },
  { event := event86876
    frameStart := 86787 },
  { event := event86877
    frameStart := 86787 },
  { event := event86878
    frameStart := 86787 },
  { event := event86879
    frameStart := 86787 }
]

def eventLeaf5430 : Array AnnotatedEvent := #[
  { event := event86880
    frameStart := 86787 },
  { event := event86881
    frameStart := 86787 },
  { event := event86882
    frameStart := 86787 },
  { event := event86883
    frameStart := 86787 },
  { event := event86884
    frameStart := 86787 },
  { event := event86885
    frameStart := 86787 },
  { event := event86886
    frameStart := 86787 },
  { event := event86887
    frameStart := 86787 },
  { event := event86888
    frameStart := 86787 },
  { event := event86889
    frameStart := 86787 },
  { event := event86890
    frameStart := 86787 },
  { event := event86891
    frameStart := 86787 },
  { event := event86892
    frameStart := 86787 },
  { event := event86893
    frameStart := 86787 },
  { event := event86894
    frameStart := 86787 },
  { event := event86895
    frameStart := 86787 }
]

def eventLeaf5431 : Array AnnotatedEvent := #[
  { event := event86896
    frameStart := 86787 },
  { event := event86897
    frameStart := 86787 },
  { event := event86898
    frameStart := 86787 },
  { event := event86899
    frameStart := 86787 },
  { event := event86900
    frameStart := 86787 },
  { event := event86901
    frameStart := 86787 },
  { event := event86902
    frameStart := 86787 },
  { event := event86903
    frameStart := 0 },
  { event := event86904
    frameStart := 0 },
  { event := event86905
    frameStart := 0 },
  { event := event86906
    frameStart := 0 },
  { event := event86907
    frameStart := 0 },
  { event := event86908
    frameStart := 0 },
  { event := event86909
    frameStart := 0 },
  { event := event86910
    frameStart := 0 },
  { event := event86911
    frameStart := 0 }
]

def eventLeaf5432 : Array AnnotatedEvent := #[
  { event := event86912
    frameStart := 0 },
  { event := event86913
    frameStart := 0 },
  { event := event86914
    frameStart := 0 },
  { event := event86915
    frameStart := 0 },
  { event := event86916
    frameStart := 0 },
  { event := event86917
    frameStart := 0 },
  { event := event86918
    frameStart := 0 },
  { event := event86919
    frameStart := 0 },
  { event := event86920
    frameStart := 0 },
  { event := event86921
    frameStart := 0 },
  { event := event86922
    frameStart := 0 },
  { event := event86923
    frameStart := 0 },
  { event := event86924
    frameStart := 0 },
  { event := event86925
    frameStart := 0 },
  { event := event86926
    frameStart := 0 },
  { event := event86927
    frameStart := 0 }
]

def eventLeaf5433 : Array AnnotatedEvent := #[
  { event := event86928
    frameStart := 0 },
  { event := event86929
    frameStart := 0 },
  { event := event86930
    frameStart := 0 },
  { event := event86931
    frameStart := 0 },
  { event := event86932
    frameStart := 0 },
  { event := event86933
    frameStart := 0 },
  { event := event86934
    frameStart := 0 },
  { event := event86935
    frameStart := 0 },
  { event := event86936
    frameStart := 0 },
  { event := event86937
    frameStart := 0 },
  { event := event86938
    frameStart := 0 },
  { event := event86939
    frameStart := 0 },
  { event := event86940
    frameStart := 86940 },
  { event := event86941
    frameStart := 86940 },
  { event := event86942
    frameStart := 86940 },
  { event := event86943
    frameStart := 86940 }
]

def eventLeaf5434 : Array AnnotatedEvent := #[
  { event := event86944
    frameStart := 86940 },
  { event := event86945
    frameStart := 86940 },
  { event := event86946
    frameStart := 86940 },
  { event := event86947
    frameStart := 86940 },
  { event := event86948
    frameStart := 86940 },
  { event := event86949
    frameStart := 86940 },
  { event := event86950
    frameStart := 86940 },
  { event := event86951
    frameStart := 86940 },
  { event := event86952
    frameStart := 86940 },
  { event := event86953
    frameStart := 86940 },
  { event := event86954
    frameStart := 86940 },
  { event := event86955
    frameStart := 86940 },
  { event := event86956
    frameStart := 86940 },
  { event := event86957
    frameStart := 86940 },
  { event := event86958
    frameStart := 86940 },
  { event := event86959
    frameStart := 86940 }
]

def eventLeaf5435 : Array AnnotatedEvent := #[
  { event := event86960
    frameStart := 86940 },
  { event := event86961
    frameStart := 86940 },
  { event := event86962
    frameStart := 86940 },
  { event := event86963
    frameStart := 86940 },
  { event := event86964
    frameStart := 86940 },
  { event := event86965
    frameStart := 86940 },
  { event := event86966
    frameStart := 86940 },
  { event := event86967
    frameStart := 86940 },
  { event := event86968
    frameStart := 86940 },
  { event := event86969
    frameStart := 86940 },
  { event := event86970
    frameStart := 86940 },
  { event := event86971
    frameStart := 86940 },
  { event := event86972
    frameStart := 86940 },
  { event := event86973
    frameStart := 86940 },
  { event := event86974
    frameStart := 86940 },
  { event := event86975
    frameStart := 86940 }
]

def eventLeaf5436 : Array AnnotatedEvent := #[
  { event := event86976
    frameStart := 86940 },
  { event := event86977
    frameStart := 86940 },
  { event := event86978
    frameStart := 86940 },
  { event := event86979
    frameStart := 86940 },
  { event := event86980
    frameStart := 86940 },
  { event := event86981
    frameStart := 86940 },
  { event := event86982
    frameStart := 86940 },
  { event := event86983
    frameStart := 86940 },
  { event := event86984
    frameStart := 86940 },
  { event := event86985
    frameStart := 86940 },
  { event := event86986
    frameStart := 86940 },
  { event := event86987
    frameStart := 86940 },
  { event := event86988
    frameStart := 86940 },
  { event := event86989
    frameStart := 86940 },
  { event := event86990
    frameStart := 86940 },
  { event := event86991
    frameStart := 86940 }
]

def eventLeaf5437 : Array AnnotatedEvent := #[
  { event := event86992
    frameStart := 86940 },
  { event := event86993
    frameStart := 86940 },
  { event := event86994
    frameStart := 86994 },
  { event := event86995
    frameStart := 86994 },
  { event := event86996
    frameStart := 86994 },
  { event := event86997
    frameStart := 86994 },
  { event := event86998
    frameStart := 86994 },
  { event := event86999
    frameStart := 86994 },
  { event := event87000
    frameStart := 86994 },
  { event := event87001
    frameStart := 86994 },
  { event := event87002
    frameStart := 86994 },
  { event := event87003
    frameStart := 86994 },
  { event := event87004
    frameStart := 86994 },
  { event := event87005
    frameStart := 86994 },
  { event := event87006
    frameStart := 86994 },
  { event := event87007
    frameStart := 86994 }
]

def eventLeaf5438 : Array AnnotatedEvent := #[
  { event := event87008
    frameStart := 86994 },
  { event := event87009
    frameStart := 86994 },
  { event := event87010
    frameStart := 86994 },
  { event := event87011
    frameStart := 86994 },
  { event := event87012
    frameStart := 86994 },
  { event := event87013
    frameStart := 86994 },
  { event := event87014
    frameStart := 86994 },
  { event := event87015
    frameStart := 86994 },
  { event := event87016
    frameStart := 86994 },
  { event := event87017
    frameStart := 86994 },
  { event := event87018
    frameStart := 86994 },
  { event := event87019
    frameStart := 86994 },
  { event := event87020
    frameStart := 86994 },
  { event := event87021
    frameStart := 86994 },
  { event := event87022
    frameStart := 86994 },
  { event := event87023
    frameStart := 86994 }
]

def eventLeaf5439 : Array AnnotatedEvent := #[
  { event := event87024
    frameStart := 86994 },
  { event := event87025
    frameStart := 86994 },
  { event := event87026
    frameStart := 86994 },
  { event := event87027
    frameStart := 86994 },
  { event := event87028
    frameStart := 86994 },
  { event := event87029
    frameStart := 86994 },
  { event := event87030
    frameStart := 86994 },
  { event := event87031
    frameStart := 86994 },
  { event := event87032
    frameStart := 86994 },
  { event := event87033
    frameStart := 86994 },
  { event := event87034
    frameStart := 86994 },
  { event := event87035
    frameStart := 86994 },
  { event := event87036
    frameStart := 86994 },
  { event := event87037
    frameStart := 86994 },
  { event := event87038
    frameStart := 86994 },
  { event := event87039
    frameStart := 86994 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events339
