import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events269

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event68864 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11853⟩⟩) 0 ⟨11755⟩ 68850

def event68865 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11853⟩⟩) 1 ⟨110⟩ 68863

def event68866 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11853⟩⟩) (.sum [.predecessor 0 68864 .coefficient, .predecessor 1 68865 .coefficient])

def event68867 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11853⟩⟩) (.finite 900)

def event68868 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11854⟩⟩) 0 ⟨11853⟩ 68867

def event68869 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11854⟩⟩) (.identity (.predecessor 0 68868 .coefficient))

def exact68870RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], []⟩, (1)⟩]

theorem exact68870RawTermsValid :
    exact68870RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68870 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11854⟩⟩) exact68870RawTerms (.finite 900) 68869 .exactZero (none)

def event68871 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact68872RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact68872RawTermsValid :
    exact68872RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68872 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact68872RawTerms .large 68871 .exactZero (none)

def event68873 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11855⟩⟩) 0 ⟨6544⟩ 68872

def event68874 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11855⟩⟩) 1 ⟨11854⟩ 68870

def event68875 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11855⟩⟩) (.product (.predecessor 0 68873 .coefficient) (.predecessor 1 68874 .coefficient) (⟨false, false, none, none, none⟩))

def event68876 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11855⟩⟩, .operator (⟨68872, 0⟩, ⟨68870, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact68877RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact68877RawTermsValid :
    exact68877RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68877 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11855⟩⟩) exact68877RawTerms .large 68875 .exactZero (none)

def event68878 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event68879 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event68880 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 68854

def event68881 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact68882RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact68882RawTermsValid :
    exact68882RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68882 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact68882RawTerms .large 68881 .exactZero (none)

def event68883 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6783⟩⟩) 0 ⟨6757⟩ 68882

def event68884 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6783⟩⟩) (.identity (.predecessor 0 68883 .coefficient))

def exact68885RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩]

theorem exact68885RawTermsValid :
    exact68885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68885 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6783⟩⟩) exact68885RawTerms .large 68884 .exactZero (none)

def event68886 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7861⟩⟩) 0 ⟨6783⟩ 68885

def event68887 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7861⟩⟩) (.authority (.operator))

def exact68888RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩]

theorem exact68888RawTermsValid :
    exact68888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68888 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7861⟩⟩) exact68888RawTerms (.finite 8192) 68887 .exactZero (none)

def event68889 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7862⟩⟩) 0 ⟨7861⟩ 68888

def event68890 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7862⟩⟩) 1 ⟨2348⟩ 68879

def event68891 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7862⟩⟩) (.scale (.predecessor 0 68889 .coefficient) (.value (.predecessor 1 68890 .coefficient)))

def exact68892RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩]

theorem exact68892RawTermsValid :
    exact68892RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68892 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7862⟩⟩) exact68892RawTerms (.finite 8192) 68891 .exactZero (none)

def event68893 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6763⟩⟩) 0 ⟨6757⟩ 68882

def event68894 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6763⟩⟩) (.identity (.predecessor 0 68893 .coefficient))

def exact68895RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩]⟩, (1)⟩]

theorem exact68895RawTermsValid :
    exact68895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68895 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6763⟩⟩) exact68895RawTerms .large 68894 .exactZero (none)

def event68896 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7863⟩⟩) 0 ⟨6763⟩ 68895

def event68897 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7863⟩⟩) 1 ⟨7862⟩ 68892

def event68898 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7863⟩⟩) (.product (.predecessor 0 68896 .coefficient) (.predecessor 1 68897 .coefficient) (⟨false, false, none, none, none⟩))

def event68899 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7863⟩⟩, .operator (⟨68895, 0⟩, ⟨68892, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩)

def exact68900RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩]

theorem exact68900RawTermsValid :
    exact68900RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68900 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7863⟩⟩) exact68900RawTerms .large 68898 .exactZero (none)

def event68901 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11856⟩⟩) 0 ⟨7863⟩ 68900

def event68902 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11856⟩⟩) 1 ⟨11855⟩ 68877

def event68903 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11856⟩⟩) (.sum [.predecessor 0 68901 .coefficient, .predecessor 1 68902 .coefficient])

def exact68904RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68904RawTermsValid :
    exact68904RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68904 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11856⟩⟩) exact68904RawTerms .large 68903 .exactZero (none)

def event68905 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25140⟩⟩) 0 ⟨11856⟩ 68904

def event68906 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25140⟩⟩) 1 ⟨25137⟩ 68861

def event68907 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25140⟩⟩) (.product (.predecessor 0 68905 .coefficient) (.predecessor 1 68906 .coefficient) (⟨false, false, none, none, none⟩))

def event68908 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25140⟩⟩, .operator (⟨68904, 0⟩, ⟨68861, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25137⟩⟩]⟩, (1)⟩)

def event68909 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25140⟩⟩, .operator (⟨68904, 1⟩, ⟨68861, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25137⟩⟩]⟩, (-1)⟩)

def event68910 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25140⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25137⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25137⟩⟩) ⟨23078⟩ 68858)

def event68911 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25140⟩⟩, .relation 68910 0, ⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨23078⟩⟩]⟩, (-1)⟩)

def exact68912RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25137⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨23078⟩⟩]⟩, (-1)⟩]

theorem exact68912RawTermsValid :
    exact68912RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68912 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25140⟩⟩) exact68912RawTerms .large 68907 .exactZero (none)

def event68913 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16258⟩⟩) 0 ⟨11755⟩ 68850

def event68914 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16258⟩⟩) (.authority (.programFamilyFact))

def exact68915RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], []⟩, (1)⟩]

theorem exact68915RawTermsValid :
    exact68915RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68915 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16258⟩⟩) exact68915RawTerms (.finite 30) 68914 .exactZero (none)

def event68916 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16260⟩⟩) 0 ⟨6544⟩ 68872

def event68917 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16260⟩⟩) 1 ⟨16258⟩ 68915

def event68918 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16260⟩⟩) (.product (.predecessor 0 68916 .coefficient) (.predecessor 1 68917 .coefficient) (⟨false, true, none, none, some 1⟩))

def event68919 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16260⟩⟩, .operator (⟨68872, 0⟩, ⟨68915, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact68920RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact68920RawTermsValid :
    exact68920RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68920 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16260⟩⟩) exact68920RawTerms .large 68918 .exactZero (none)

def event68921 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6700⟩⟩) 0 ⟨6689⟩ 68854

def event68922 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6700⟩⟩) (.authority (.operator))

def exact68923RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩]

theorem exact68923RawTermsValid :
    exact68923RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68923 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6700⟩⟩) exact68923RawTerms .large 68922 .exactZero (none)

def event68924 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16261⟩⟩) 0 ⟨6700⟩ 68923

def event68925 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16261⟩⟩) 1 ⟨16260⟩ 68920

def event68926 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16261⟩⟩) (.sum [.predecessor 0 68924 .coefficient, .predecessor 1 68925 .coefficient])

def exact68927RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68927RawTermsValid :
    exact68927RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68927 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16261⟩⟩) exact68927RawTerms .large 68926 .exactZero (none)

def event68928 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25141⟩⟩) 0 ⟨16261⟩ 68927

def event68929 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25141⟩⟩) 1 ⟨25140⟩ 68912

def event68930 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25141⟩⟩) (.sum [.predecessor 0 68928 .coefficient, .predecessor 1 68929 .coefficient])

def exact68931RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25137⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨23078⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68931RawTermsValid :
    exact68931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68931 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25141⟩⟩) exact68931RawTerms .large 68930 .exactZero (none)

def event68932 : Event := .preFoldPolynomial 68931 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25137⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨23078⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact68933RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25137⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨23078⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event68933 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25141⟩⟩) 68932 exact68933RawTerms .large 68930 .exactZero (none)

def event68934 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨11755⟩⟩) ⟨⟨113⟩, ⟨18⟩, ⟨109⟩⟩ ⟨68768, 68934⟩

def event68935 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19743⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19740⟩⟩]⟩) (1) 0 2 (.universal 68934 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19740⟩⟩]⟩) (none) 68933)

def event68936 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19743⟩⟩, .relation 68935 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩)

def event68937 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19743⟩⟩, .relation 68935 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25137⟩⟩]⟩, (-1)⟩)

def event68938 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19743⟩⟩, .relation 68935 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨23078⟩⟩]⟩, (1)⟩)

def event68939 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19743⟩⟩, .relation 68935 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact68940RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25137⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨23078⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68940RawTermsValid :
    exact68940RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68940 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19743⟩⟩) exact68940RawTerms .large 68764 (.finite 1811303510016) (some (68766))

def event68941 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25139⟩⟩) 0 ⟨19743⟩ 68940

def event68942 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25139⟩⟩) 1 ⟨25138⟩ 68754

def event68943 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25139⟩⟩) (.sum [.predecessor 0 68941 .coefficient, .predecessor 1 68942 .coefficient])

def event68944 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25139⟩⟩, .operator (⟨68940, 2⟩, ⟨68754, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], [⟨.program ⟨214⟩, ⟨23078⟩⟩]⟩, (-1)⟩)

def event68945 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25139⟩⟩, .operator (⟨68940, 1⟩, ⟨68754, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25137⟩⟩]⟩, (1)⟩)

def event68946 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25139⟩⟩) (.sum [.result 68940 .summary, .result 68754 .summary])

def exact68947RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68947RawTermsValid :
    exact68947RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68947 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25139⟩⟩) exact68947RawTerms .large 68943 (.finite 352097360556032) (some (68946))

def event68948 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28506⟩⟩) 0 ⟨25139⟩ 68947

def event68949 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28506⟩⟩) 1 ⟨28504⟩ 68670

def event68950 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28506⟩⟩) (.product (.predecessor 0 68948 .coefficient) (.predecessor 1 68949 .coefficient) (⟨false, false, none, none, none⟩))

def event68951 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28506⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28504⟩⟩]⟩) [⟨.result 68670 .coefficient, false, none⟩])

def event68952 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28506⟩⟩) (.product (.result 68947 .summary) (.transfer 68951) (⟨false, false, none, none, none⟩))

def event68953 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28506⟩⟩, .operator (⟨68947, 0⟩, ⟨68670, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩]⟩, (1)⟩)

def event68954 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28506⟩⟩, .operator (⟨68947, 1⟩, ⟨68670, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩]⟩, (-1)⟩)

def event68955 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28506⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28504⟩⟩) ⟨24348⟩ 68667)

def event68956 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28506⟩⟩, .relation 68955 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨24348⟩⟩]⟩, (-1)⟩)

def exact68957RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨24348⟩⟩]⟩, (-1)⟩]

theorem exact68957RawTermsValid :
    exact68957RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68957 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28506⟩⟩) exact68957RawTerms .large 68950 (.finite 1292202946798406336512) (some (68952))

def event68958 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21828⟩⟩) 0 ⟨16259⟩ 3264

def event68959 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21828⟩⟩) (.authority (.relationPreimageSource ⟨50⟩))

def exact68960RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21828⟩⟩]⟩, (1)⟩]

theorem exact68960RawTermsValid :
    exact68960RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68960 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21828⟩⟩) exact68960RawTerms (.finite 136065468) 68959 .exactZero (none)

def event68961 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21830⟩⟩) 0 ⟨21828⟩ 68960

def event68962 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21830⟩⟩) 1 ⟨2348⟩ 4

def event68963 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21830⟩⟩) (.scale (.predecessor 0 68961 .coefficient) (.value (.predecessor 1 68962 .coefficient)))

def exact68964RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21828⟩⟩]⟩, (1)⟩]

theorem exact68964RawTermsValid :
    exact68964RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68964 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21830⟩⟩) exact68964RawTerms (.finite 136065468) 68963 .exactZero (none)

def event68965 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21831⟩⟩) 0 ⟨5535⟩ 65387

def event68966 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21831⟩⟩) 1 ⟨21830⟩ 68964

def event68967 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21831⟩⟩) (.product (.predecessor 0 68965 .coefficient) (.predecessor 1 68966 .coefficient) (⟨false, false, none, none, none⟩))

def event68968 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21831⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21828⟩⟩]⟩) [⟨.result 68960 .coefficient, false, none⟩])

def event68969 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21831⟩⟩) (.product (.result 65387 .summary) (.transfer 68968) (⟨false, false, none, none, none⟩))

def event68970 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21831⟩⟩, .operator (⟨65387, 0⟩, ⟨68964, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21828⟩⟩]⟩, (1)⟩)

def event68971 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21829⟩⟩)

def event68972 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event68973 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event68974 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event68975 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event68976 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event68977 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event68978 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event68979 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event68980 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 68979

def event68981 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 68977

def event68982 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 68980 .coefficient) (.value (.predecessor 1 68981 .coefficient)))

def event68983 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event68984 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 68983

def event68985 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 68975

def event68986 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 68984 .coefficient, .predecessor 1 68985 .coefficient])

def event68987 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event68988 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 68987

def event68989 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 68973

def event68990 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 68989 .coefficient))

def event68991 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event68992 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11753⟩⟩) 0 ⟨5530⟩ 68991

def event68993 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11753⟩⟩) (.authority (.programFamilyFact))

def exact68994RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11753⟩⟩], []⟩, (1)⟩]

theorem exact68994RawTermsValid :
    exact68994RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68994 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11753⟩⟩) exact68994RawTerms (.finite 30) 68993 .exactZero (none)

def event68995 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9605⟩⟩) 0 ⟨5530⟩ 68991

def event68996 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9605⟩⟩) (.authority (.programFamilyFact))

def exact68997RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩], []⟩, (1)⟩]

theorem exact68997RawTermsValid :
    exact68997RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68997 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9605⟩⟩) exact68997RawTerms (.finite 30) 68996 .exactZero (none)

def event68998 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11754⟩⟩) 0 ⟨9605⟩ 68997

def event68999 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11754⟩⟩) 1 ⟨11753⟩ 68994

def event69000 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11754⟩⟩) (.product (.predecessor 0 68998 .coefficient) (.predecessor 1 68999 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event69001 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11754⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], []⟩) [⟨.result 68997 .coefficient, true, some 1⟩, ⟨.result 68994 .coefficient, true, some 1⟩])

def event69002 : Event := .survivorFold (1) 69001

def exact69003RawTerms : List Term := []

theorem exact69003RawTermsValid :
    exact69003RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69003 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11754⟩⟩) exact69003RawTerms (.finite 900) 69000 (.finite 900) (some (69001))

def event69004 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11755⟩⟩) 0 ⟨11754⟩ 69003

def event69005 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11755⟩⟩) (.identity (.predecessor 0 69004 .coefficient))

def event69006 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11755⟩⟩) (.finite 900)

def event69007 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16258⟩⟩) 0 ⟨11755⟩ 69006

def event69008 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16258⟩⟩) (.authority (.programFamilyFact))

def exact69009RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], []⟩, (1)⟩]

theorem exact69009RawTermsValid :
    exact69009RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69009 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16258⟩⟩) exact69009RawTerms (.finite 30) 69008 .exactZero (none)

def event69010 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16259⟩⟩) 0 ⟨16258⟩ 69009

def event69011 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16259⟩⟩) (.identity (.predecessor 0 69010 .coefficient))

def event69012 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16259⟩⟩) (.finite 30)

def event69013 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21828⟩⟩) 0 ⟨16259⟩ 69012

def event69014 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21828⟩⟩) (.authority (.relationPreimageSource ⟨50⟩))

def exact69015RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21828⟩⟩]⟩, (1)⟩]

theorem exact69015RawTermsValid :
    exact69015RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69015 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21828⟩⟩) exact69015RawTerms (.finite 136065468) 69014 .exactZero (none)

def event69016 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact69017RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact69017RawTermsValid :
    exact69017RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69017 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact69017RawTerms .large 69016 .exactZero (none)

def event69018 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21829⟩⟩) 0 ⟨6⟩ 69017

def event69019 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21829⟩⟩) 1 ⟨21828⟩ 69015

def event69020 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21829⟩⟩) (.product (.predecessor 0 69018 .coefficient) (.predecessor 1 69019 .coefficient) (⟨false, false, none, none, none⟩))

def event69021 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21829⟩⟩, .operator (⟨69017, 0⟩, ⟨69015, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21828⟩⟩]⟩, (1)⟩)

def exact69022RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21828⟩⟩]⟩, (1)⟩]

theorem exact69022RawTermsValid :
    exact69022RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69022 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21829⟩⟩) exact69022RawTerms .large 69020 .exactZero (none)

def event69023 : Event := .preFoldPolynomial 69022 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21828⟩⟩]⟩, (1)⟩] .exactZero none

def exact69024RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21828⟩⟩]⟩, (1)⟩]

def event69024 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21829⟩⟩) 69023 exact69024RawTerms .large 69020 .exactZero (none)

def event69025 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28509⟩⟩)

def event69026 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event69027 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event69028 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event69029 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event69030 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event69031 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event69032 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event69033 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event69034 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 69033

def event69035 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 69031

def event69036 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 69034 .coefficient) (.value (.predecessor 1 69035 .coefficient)))

def event69037 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event69038 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 69037

def event69039 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 69029

def event69040 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 69038 .coefficient, .predecessor 1 69039 .coefficient])

def event69041 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event69042 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 69041

def event69043 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 69027

def event69044 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 69043 .coefficient))

def event69045 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event69046 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11753⟩⟩) 0 ⟨5530⟩ 69045

def event69047 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11753⟩⟩) (.authority (.programFamilyFact))

def exact69048RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11753⟩⟩], []⟩, (1)⟩]

theorem exact69048RawTermsValid :
    exact69048RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69048 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11753⟩⟩) exact69048RawTerms (.finite 30) 69047 .exactZero (none)

def event69049 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9605⟩⟩) 0 ⟨5530⟩ 69045

def event69050 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9605⟩⟩) (.authority (.programFamilyFact))

def exact69051RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩], []⟩, (1)⟩]

theorem exact69051RawTermsValid :
    exact69051RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69051 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9605⟩⟩) exact69051RawTerms (.finite 30) 69050 .exactZero (none)

def event69052 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11754⟩⟩) 0 ⟨9605⟩ 69051

def event69053 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11754⟩⟩) 1 ⟨11753⟩ 69048

def event69054 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11754⟩⟩) (.product (.predecessor 0 69052 .coefficient) (.predecessor 1 69053 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event69055 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11754⟩⟩, .operator (⟨69051, 0⟩, ⟨69048, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], []⟩, (1)⟩)

def exact69056RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩, ⟨.program ⟨214⟩, ⟨11753⟩⟩], []⟩, (1)⟩]

theorem exact69056RawTermsValid :
    exact69056RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69056 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11754⟩⟩) exact69056RawTerms (.finite 900) 69054 .exactZero (none)

def event69057 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11755⟩⟩) 0 ⟨11754⟩ 69056

def event69058 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11755⟩⟩) (.identity (.predecessor 0 69057 .coefficient))

def event69059 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11755⟩⟩) (.finite 900)

def event69060 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16258⟩⟩) 0 ⟨11755⟩ 69059

def event69061 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16258⟩⟩) (.authority (.programFamilyFact))

def exact69062RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], []⟩, (1)⟩]

theorem exact69062RawTermsValid :
    exact69062RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69062 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16258⟩⟩) exact69062RawTerms (.finite 30) 69061 .exactZero (none)

def event69063 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16259⟩⟩) 0 ⟨16258⟩ 69062

def event69064 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16259⟩⟩) (.identity (.predecessor 0 69063 .coefficient))

def event69065 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16259⟩⟩) (.finite 30)

def event69066 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24346⟩⟩) 0 ⟨16259⟩ 69065

def event69067 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24346⟩⟩) (.authority (.programFamilyFact))

def event69068 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24346⟩⟩) (.finite 3720)

def event69069 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event69070 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24348⟩⟩) 0 ⟨6689⟩ 69069

def event69071 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24348⟩⟩) 1 ⟨24346⟩ 69068

def event69072 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24348⟩⟩) (.authority (.operator))

def exact69073RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24348⟩⟩]⟩, (1)⟩]

theorem exact69073RawTermsValid :
    exact69073RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69073 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24348⟩⟩) exact69073RawTerms .large 69072 .exactZero (none)

def event69074 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28504⟩⟩) 0 ⟨24348⟩ 69073

def event69075 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28504⟩⟩) (.authority (.operator))

def exact69076RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28504⟩⟩]⟩, (1)⟩]

theorem exact69076RawTermsValid :
    exact69076RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69076 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28504⟩⟩) exact69076RawTerms (.finite 8192) 69075 .exactZero (none)

def event69077 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event69078 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event69079 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16333⟩⟩) 0 ⟨16259⟩ 69065

def event69080 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16333⟩⟩) 1 ⟨110⟩ 69078

def event69081 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16333⟩⟩) (.sum [.predecessor 0 69079 .coefficient, .predecessor 1 69080 .coefficient])

def event69082 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16333⟩⟩) (.finite 30)

def event69083 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16334⟩⟩) 0 ⟨16333⟩ 69082

def event69084 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16334⟩⟩) (.identity (.predecessor 0 69083 .coefficient))

def exact69085RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], []⟩, (1)⟩]

theorem exact69085RawTermsValid :
    exact69085RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69085 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16334⟩⟩) exact69085RawTerms (.finite 30) 69084 .exactZero (none)

def event69086 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact69087RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact69087RawTermsValid :
    exact69087RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69087 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact69087RawTerms .large 69086 .exactZero (none)

def event69088 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16335⟩⟩) 0 ⟨6544⟩ 69087

def event69089 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16335⟩⟩) 1 ⟨16334⟩ 69085

def event69090 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16335⟩⟩) (.product (.predecessor 0 69088 .coefficient) (.predecessor 1 69089 .coefficient) (⟨false, false, none, none, none⟩))

def event69091 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16335⟩⟩, .operator (⟨69087, 0⟩, ⟨69085, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact69092RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact69092RawTermsValid :
    exact69092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69092 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16335⟩⟩) exact69092RawTerms .large 69090 .exactZero (none)

def event69093 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6700⟩⟩) 0 ⟨6689⟩ 69069

def event69094 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6700⟩⟩) (.authority (.operator))

def exact69095RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩]

theorem exact69095RawTermsValid :
    exact69095RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69095 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6700⟩⟩) exact69095RawTerms .large 69094 .exactZero (none)

def event69096 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16336⟩⟩) 0 ⟨6700⟩ 69095

def event69097 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16336⟩⟩) 1 ⟨16335⟩ 69092

def event69098 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16336⟩⟩) (.sum [.predecessor 0 69096 .coefficient, .predecessor 1 69097 .coefficient])

def exact69099RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69099RawTermsValid :
    exact69099RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69099 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16336⟩⟩) exact69099RawTerms .large 69098 .exactZero (none)

def event69100 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28505⟩⟩) 0 ⟨16336⟩ 69099

def event69101 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28505⟩⟩) 1 ⟨28504⟩ 69076

def event69102 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28505⟩⟩) (.product (.predecessor 0 69100 .coefficient) (.predecessor 1 69101 .coefficient) (⟨false, false, none, none, none⟩))

def event69103 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28505⟩⟩, .operator (⟨69099, 0⟩, ⟨69076, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩]⟩, (1)⟩)

def event69104 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28505⟩⟩, .operator (⟨69099, 1⟩, ⟨69076, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩]⟩, (-1)⟩)

def event69105 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28505⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28504⟩⟩) ⟨24348⟩ 69073)

def event69106 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28505⟩⟩, .relation 69105 0, ⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨24348⟩⟩]⟩, (-1)⟩)

def exact69107RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨24348⟩⟩]⟩, (-1)⟩]

theorem exact69107RawTermsValid :
    exact69107RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69107 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28505⟩⟩) exact69107RawTerms .large 69102 .exactZero (none)

def event69108 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16305⟩⟩) 0 ⟨16259⟩ 69065

def event69109 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16305⟩⟩) (.authority (.programFamilyFact))

def exact69110RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16305⟩⟩], []⟩, (1)⟩]

theorem exact69110RawTermsValid :
    exact69110RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69110 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16305⟩⟩) exact69110RawTerms (.finite 62) 69109 .exactZero (none)

def event69111 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16306⟩⟩) 0 ⟨6544⟩ 69087

def event69112 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16306⟩⟩) 1 ⟨16305⟩ 69110

def event69113 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16306⟩⟩) (.product (.predecessor 0 69111 .coefficient) (.predecessor 1 69112 .coefficient) (⟨false, true, none, none, some 1⟩))

def event69114 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16306⟩⟩, .operator (⟨69087, 0⟩, ⟨69110, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16305⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact69115RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16305⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact69115RawTermsValid :
    exact69115RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69115 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16306⟩⟩) exact69115RawTerms .large 69113 .exactZero (none)

def event69116 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6729⟩⟩) 0 ⟨6689⟩ 69069

def event69117 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6729⟩⟩) (.authority (.operator))

def exact69118RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩]

theorem exact69118RawTermsValid :
    exact69118RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69118 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6729⟩⟩) exact69118RawTerms .large 69117 .exactZero (none)

def event69119 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16307⟩⟩) 0 ⟨6729⟩ 69118

def eventLeaf4304 : Array AnnotatedEvent := #[
  { event := event68864
    frameStart := 68816 },
  { event := event68865
    frameStart := 68816 },
  { event := event68866
    frameStart := 68816 },
  { event := event68867
    frameStart := 68816 },
  { event := event68868
    frameStart := 68816 },
  { event := event68869
    frameStart := 68816 },
  { event := event68870
    frameStart := 68816 },
  { event := event68871
    frameStart := 68816 },
  { event := event68872
    frameStart := 68816 },
  { event := event68873
    frameStart := 68816 },
  { event := event68874
    frameStart := 68816 },
  { event := event68875
    frameStart := 68816 },
  { event := event68876
    frameStart := 68816 },
  { event := event68877
    frameStart := 68816 },
  { event := event68878
    frameStart := 68816 },
  { event := event68879
    frameStart := 68816 }
]

def eventLeaf4305 : Array AnnotatedEvent := #[
  { event := event68880
    frameStart := 68816 },
  { event := event68881
    frameStart := 68816 },
  { event := event68882
    frameStart := 68816 },
  { event := event68883
    frameStart := 68816 },
  { event := event68884
    frameStart := 68816 },
  { event := event68885
    frameStart := 68816 },
  { event := event68886
    frameStart := 68816 },
  { event := event68887
    frameStart := 68816 },
  { event := event68888
    frameStart := 68816 },
  { event := event68889
    frameStart := 68816 },
  { event := event68890
    frameStart := 68816 },
  { event := event68891
    frameStart := 68816 },
  { event := event68892
    frameStart := 68816 },
  { event := event68893
    frameStart := 68816 },
  { event := event68894
    frameStart := 68816 },
  { event := event68895
    frameStart := 68816 }
]

def eventLeaf4306 : Array AnnotatedEvent := #[
  { event := event68896
    frameStart := 68816 },
  { event := event68897
    frameStart := 68816 },
  { event := event68898
    frameStart := 68816 },
  { event := event68899
    frameStart := 68816 },
  { event := event68900
    frameStart := 68816 },
  { event := event68901
    frameStart := 68816 },
  { event := event68902
    frameStart := 68816 },
  { event := event68903
    frameStart := 68816 },
  { event := event68904
    frameStart := 68816 },
  { event := event68905
    frameStart := 68816 },
  { event := event68906
    frameStart := 68816 },
  { event := event68907
    frameStart := 68816 },
  { event := event68908
    frameStart := 68816 },
  { event := event68909
    frameStart := 68816 },
  { event := event68910
    frameStart := 68816 },
  { event := event68911
    frameStart := 68816 }
]

def eventLeaf4307 : Array AnnotatedEvent := #[
  { event := event68912
    frameStart := 68816 },
  { event := event68913
    frameStart := 68816 },
  { event := event68914
    frameStart := 68816 },
  { event := event68915
    frameStart := 68816 },
  { event := event68916
    frameStart := 68816 },
  { event := event68917
    frameStart := 68816 },
  { event := event68918
    frameStart := 68816 },
  { event := event68919
    frameStart := 68816 },
  { event := event68920
    frameStart := 68816 },
  { event := event68921
    frameStart := 68816 },
  { event := event68922
    frameStart := 68816 },
  { event := event68923
    frameStart := 68816 },
  { event := event68924
    frameStart := 68816 },
  { event := event68925
    frameStart := 68816 },
  { event := event68926
    frameStart := 68816 },
  { event := event68927
    frameStart := 68816 }
]

def eventLeaf4308 : Array AnnotatedEvent := #[
  { event := event68928
    frameStart := 68816 },
  { event := event68929
    frameStart := 68816 },
  { event := event68930
    frameStart := 68816 },
  { event := event68931
    frameStart := 68816 },
  { event := event68932
    frameStart := 68816 },
  { event := event68933
    frameStart := 68816 },
  { event := event68934
    frameStart := 0 },
  { event := event68935
    frameStart := 0 },
  { event := event68936
    frameStart := 0 },
  { event := event68937
    frameStart := 0 },
  { event := event68938
    frameStart := 0 },
  { event := event68939
    frameStart := 0 },
  { event := event68940
    frameStart := 0 },
  { event := event68941
    frameStart := 0 },
  { event := event68942
    frameStart := 0 },
  { event := event68943
    frameStart := 0 }
]

def eventLeaf4309 : Array AnnotatedEvent := #[
  { event := event68944
    frameStart := 0 },
  { event := event68945
    frameStart := 0 },
  { event := event68946
    frameStart := 0 },
  { event := event68947
    frameStart := 0 },
  { event := event68948
    frameStart := 0 },
  { event := event68949
    frameStart := 0 },
  { event := event68950
    frameStart := 0 },
  { event := event68951
    frameStart := 0 },
  { event := event68952
    frameStart := 0 },
  { event := event68953
    frameStart := 0 },
  { event := event68954
    frameStart := 0 },
  { event := event68955
    frameStart := 0 },
  { event := event68956
    frameStart := 0 },
  { event := event68957
    frameStart := 0 },
  { event := event68958
    frameStart := 0 },
  { event := event68959
    frameStart := 0 }
]

def eventLeaf4310 : Array AnnotatedEvent := #[
  { event := event68960
    frameStart := 0 },
  { event := event68961
    frameStart := 0 },
  { event := event68962
    frameStart := 0 },
  { event := event68963
    frameStart := 0 },
  { event := event68964
    frameStart := 0 },
  { event := event68965
    frameStart := 0 },
  { event := event68966
    frameStart := 0 },
  { event := event68967
    frameStart := 0 },
  { event := event68968
    frameStart := 0 },
  { event := event68969
    frameStart := 0 },
  { event := event68970
    frameStart := 0 },
  { event := event68971
    frameStart := 68971 },
  { event := event68972
    frameStart := 68971 },
  { event := event68973
    frameStart := 68971 },
  { event := event68974
    frameStart := 68971 },
  { event := event68975
    frameStart := 68971 }
]

def eventLeaf4311 : Array AnnotatedEvent := #[
  { event := event68976
    frameStart := 68971 },
  { event := event68977
    frameStart := 68971 },
  { event := event68978
    frameStart := 68971 },
  { event := event68979
    frameStart := 68971 },
  { event := event68980
    frameStart := 68971 },
  { event := event68981
    frameStart := 68971 },
  { event := event68982
    frameStart := 68971 },
  { event := event68983
    frameStart := 68971 },
  { event := event68984
    frameStart := 68971 },
  { event := event68985
    frameStart := 68971 },
  { event := event68986
    frameStart := 68971 },
  { event := event68987
    frameStart := 68971 },
  { event := event68988
    frameStart := 68971 },
  { event := event68989
    frameStart := 68971 },
  { event := event68990
    frameStart := 68971 },
  { event := event68991
    frameStart := 68971 }
]

def eventLeaf4312 : Array AnnotatedEvent := #[
  { event := event68992
    frameStart := 68971 },
  { event := event68993
    frameStart := 68971 },
  { event := event68994
    frameStart := 68971 },
  { event := event68995
    frameStart := 68971 },
  { event := event68996
    frameStart := 68971 },
  { event := event68997
    frameStart := 68971 },
  { event := event68998
    frameStart := 68971 },
  { event := event68999
    frameStart := 68971 },
  { event := event69000
    frameStart := 68971 },
  { event := event69001
    frameStart := 68971 },
  { event := event69002
    frameStart := 68971 },
  { event := event69003
    frameStart := 68971 },
  { event := event69004
    frameStart := 68971 },
  { event := event69005
    frameStart := 68971 },
  { event := event69006
    frameStart := 68971 },
  { event := event69007
    frameStart := 68971 }
]

def eventLeaf4313 : Array AnnotatedEvent := #[
  { event := event69008
    frameStart := 68971 },
  { event := event69009
    frameStart := 68971 },
  { event := event69010
    frameStart := 68971 },
  { event := event69011
    frameStart := 68971 },
  { event := event69012
    frameStart := 68971 },
  { event := event69013
    frameStart := 68971 },
  { event := event69014
    frameStart := 68971 },
  { event := event69015
    frameStart := 68971 },
  { event := event69016
    frameStart := 68971 },
  { event := event69017
    frameStart := 68971 },
  { event := event69018
    frameStart := 68971 },
  { event := event69019
    frameStart := 68971 },
  { event := event69020
    frameStart := 68971 },
  { event := event69021
    frameStart := 68971 },
  { event := event69022
    frameStart := 68971 },
  { event := event69023
    frameStart := 68971 }
]

def eventLeaf4314 : Array AnnotatedEvent := #[
  { event := event69024
    frameStart := 68971 },
  { event := event69025
    frameStart := 69025 },
  { event := event69026
    frameStart := 69025 },
  { event := event69027
    frameStart := 69025 },
  { event := event69028
    frameStart := 69025 },
  { event := event69029
    frameStart := 69025 },
  { event := event69030
    frameStart := 69025 },
  { event := event69031
    frameStart := 69025 },
  { event := event69032
    frameStart := 69025 },
  { event := event69033
    frameStart := 69025 },
  { event := event69034
    frameStart := 69025 },
  { event := event69035
    frameStart := 69025 },
  { event := event69036
    frameStart := 69025 },
  { event := event69037
    frameStart := 69025 },
  { event := event69038
    frameStart := 69025 },
  { event := event69039
    frameStart := 69025 }
]

def eventLeaf4315 : Array AnnotatedEvent := #[
  { event := event69040
    frameStart := 69025 },
  { event := event69041
    frameStart := 69025 },
  { event := event69042
    frameStart := 69025 },
  { event := event69043
    frameStart := 69025 },
  { event := event69044
    frameStart := 69025 },
  { event := event69045
    frameStart := 69025 },
  { event := event69046
    frameStart := 69025 },
  { event := event69047
    frameStart := 69025 },
  { event := event69048
    frameStart := 69025 },
  { event := event69049
    frameStart := 69025 },
  { event := event69050
    frameStart := 69025 },
  { event := event69051
    frameStart := 69025 },
  { event := event69052
    frameStart := 69025 },
  { event := event69053
    frameStart := 69025 },
  { event := event69054
    frameStart := 69025 },
  { event := event69055
    frameStart := 69025 }
]

def eventLeaf4316 : Array AnnotatedEvent := #[
  { event := event69056
    frameStart := 69025 },
  { event := event69057
    frameStart := 69025 },
  { event := event69058
    frameStart := 69025 },
  { event := event69059
    frameStart := 69025 },
  { event := event69060
    frameStart := 69025 },
  { event := event69061
    frameStart := 69025 },
  { event := event69062
    frameStart := 69025 },
  { event := event69063
    frameStart := 69025 },
  { event := event69064
    frameStart := 69025 },
  { event := event69065
    frameStart := 69025 },
  { event := event69066
    frameStart := 69025 },
  { event := event69067
    frameStart := 69025 },
  { event := event69068
    frameStart := 69025 },
  { event := event69069
    frameStart := 69025 },
  { event := event69070
    frameStart := 69025 },
  { event := event69071
    frameStart := 69025 }
]

def eventLeaf4317 : Array AnnotatedEvent := #[
  { event := event69072
    frameStart := 69025 },
  { event := event69073
    frameStart := 69025 },
  { event := event69074
    frameStart := 69025 },
  { event := event69075
    frameStart := 69025 },
  { event := event69076
    frameStart := 69025 },
  { event := event69077
    frameStart := 69025 },
  { event := event69078
    frameStart := 69025 },
  { event := event69079
    frameStart := 69025 },
  { event := event69080
    frameStart := 69025 },
  { event := event69081
    frameStart := 69025 },
  { event := event69082
    frameStart := 69025 },
  { event := event69083
    frameStart := 69025 },
  { event := event69084
    frameStart := 69025 },
  { event := event69085
    frameStart := 69025 },
  { event := event69086
    frameStart := 69025 },
  { event := event69087
    frameStart := 69025 }
]

def eventLeaf4318 : Array AnnotatedEvent := #[
  { event := event69088
    frameStart := 69025 },
  { event := event69089
    frameStart := 69025 },
  { event := event69090
    frameStart := 69025 },
  { event := event69091
    frameStart := 69025 },
  { event := event69092
    frameStart := 69025 },
  { event := event69093
    frameStart := 69025 },
  { event := event69094
    frameStart := 69025 },
  { event := event69095
    frameStart := 69025 },
  { event := event69096
    frameStart := 69025 },
  { event := event69097
    frameStart := 69025 },
  { event := event69098
    frameStart := 69025 },
  { event := event69099
    frameStart := 69025 },
  { event := event69100
    frameStart := 69025 },
  { event := event69101
    frameStart := 69025 },
  { event := event69102
    frameStart := 69025 },
  { event := event69103
    frameStart := 69025 }
]

def eventLeaf4319 : Array AnnotatedEvent := #[
  { event := event69104
    frameStart := 69025 },
  { event := event69105
    frameStart := 69025 },
  { event := event69106
    frameStart := 69025 },
  { event := event69107
    frameStart := 69025 },
  { event := event69108
    frameStart := 69025 },
  { event := event69109
    frameStart := 69025 },
  { event := event69110
    frameStart := 69025 },
  { event := event69111
    frameStart := 69025 },
  { event := event69112
    frameStart := 69025 },
  { event := event69113
    frameStart := 69025 },
  { event := event69114
    frameStart := 69025 },
  { event := event69115
    frameStart := 69025 },
  { event := event69116
    frameStart := 69025 },
  { event := event69117
    frameStart := 69025 },
  { event := event69118
    frameStart := 69025 },
  { event := event69119
    frameStart := 69025 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events269
