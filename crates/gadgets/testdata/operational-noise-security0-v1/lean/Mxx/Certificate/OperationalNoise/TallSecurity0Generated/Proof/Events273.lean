import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events273

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event69888 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16058⟩⟩) 0 ⟨6698⟩ 69887

def event69889 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16058⟩⟩) 1 ⟨16057⟩ 69884

def event69890 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16058⟩⟩) (.sum [.predecessor 0 69888 .coefficient, .predecessor 1 69889 .coefficient])

def exact69891RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69891RawTermsValid :
    exact69891RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69891 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16058⟩⟩) exact69891RawTerms .large 69890 .exactZero (none)

def event69892 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26142⟩⟩) 0 ⟨16058⟩ 69891

def event69893 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26142⟩⟩) 1 ⟨26141⟩ 69876

def event69894 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26142⟩⟩) (.sum [.predecessor 0 69892 .coefficient, .predecessor 1 69893 .coefficient])

def exact69895RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26138⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨23624⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69895RawTermsValid :
    exact69895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69895 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26142⟩⟩) exact69895RawTerms .large 69894 .exactZero (none)

def event69896 : Event := .preFoldPolynomial 69895 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26138⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨23624⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact69897RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26138⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨23624⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event69897 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26142⟩⟩) 69896 exact69897RawTerms .large 69894 .exactZero (none)

def event69898 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14417⟩⟩) ⟨⟨111⟩, ⟨16⟩, ⟨109⟩⟩ ⟨69732, 69898⟩

def event69899 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19599⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19596⟩⟩]⟩) (1) 0 2 (.universal 69898 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19596⟩⟩]⟩) (none) 69897)

def event69900 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19599⟩⟩, .relation 69899 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩)

def event69901 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19599⟩⟩, .relation 69899 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26138⟩⟩]⟩, (-1)⟩)

def event69902 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19599⟩⟩, .relation 69899 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨23624⟩⟩]⟩, (1)⟩)

def event69903 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19599⟩⟩, .relation 69899 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact69904RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26138⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨23624⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69904RawTermsValid :
    exact69904RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69904 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19599⟩⟩) exact69904RawTerms .large 69728 (.finite 1811303510016) (some (69730))

def event69905 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26140⟩⟩) 0 ⟨19599⟩ 69904

def event69906 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26140⟩⟩) 1 ⟨26139⟩ 69718

def event69907 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26140⟩⟩) (.sum [.predecessor 0 69905 .coefficient, .predecessor 1 69906 .coefficient])

def event69908 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26140⟩⟩, .operator (⟨69904, 2⟩, ⟨69718, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], [⟨.program ⟨214⟩, ⟨23624⟩⟩]⟩, (-1)⟩)

def event69909 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26140⟩⟩, .operator (⟨69904, 1⟩, ⟨69718, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26138⟩⟩]⟩, (1)⟩)

def event69910 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26140⟩⟩) (.sum [.result 69904 .summary, .result 69718 .summary])

def exact69911RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69911RawTermsValid :
    exact69911RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69911 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26140⟩⟩) exact69911RawTerms .large 69907 (.finite 352072932929536) (some (69910))

def event69912 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28072⟩⟩) 0 ⟨26140⟩ 69911

def event69913 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28072⟩⟩) 1 ⟨28070⟩ 69634

def event69914 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28072⟩⟩) (.product (.predecessor 0 69912 .coefficient) (.predecessor 1 69913 .coefficient) (⟨false, false, none, none, none⟩))

def event69915 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28072⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28070⟩⟩]⟩) [⟨.result 69634 .coefficient, false, none⟩])

def event69916 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28072⟩⟩) (.product (.result 69911 .summary) (.transfer 69915) (⟨false, false, none, none, none⟩))

def event69917 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28072⟩⟩, .operator (⟨69911, 0⟩, ⟨69634, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28070⟩⟩]⟩, (1)⟩)

def event69918 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28072⟩⟩, .operator (⟨69911, 1⟩, ⟨69634, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28070⟩⟩]⟩, (-1)⟩)

def event69919 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28072⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28070⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28070⟩⟩) ⟨24222⟩ 69631)

def event69920 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28072⟩⟩, .relation 69919 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24222⟩⟩]⟩, (-1)⟩)

def exact69921RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28070⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24222⟩⟩]⟩, (-1)⟩]

theorem exact69921RawTermsValid :
    exact69921RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69921 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28072⟩⟩) exact69921RawTerms .large 69914 (.finite 1292113297018323992576) (some (69916))

def event69922 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21540⟩⟩) 0 ⟨16056⟩ 3310

def event69923 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21540⟩⟩) (.authority (.relationPreimageSource ⟨46⟩))

def exact69924RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21540⟩⟩]⟩, (1)⟩]

theorem exact69924RawTermsValid :
    exact69924RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69924 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21540⟩⟩) exact69924RawTerms (.finite 136065468) 69923 .exactZero (none)

def event69925 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21542⟩⟩) 0 ⟨21540⟩ 69924

def event69926 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21542⟩⟩) 1 ⟨2348⟩ 4

def event69927 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21542⟩⟩) (.scale (.predecessor 0 69925 .coefficient) (.value (.predecessor 1 69926 .coefficient)))

def exact69928RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21540⟩⟩]⟩, (1)⟩]

theorem exact69928RawTermsValid :
    exact69928RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69928 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21542⟩⟩) exact69928RawTerms (.finite 136065468) 69927 .exactZero (none)

def event69929 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21543⟩⟩) 0 ⟨5535⟩ 65387

def event69930 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21543⟩⟩) 1 ⟨21542⟩ 69928

def event69931 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21543⟩⟩) (.product (.predecessor 0 69929 .coefficient) (.predecessor 1 69930 .coefficient) (⟨false, false, none, none, none⟩))

def event69932 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21543⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21540⟩⟩]⟩) [⟨.result 69924 .coefficient, false, none⟩])

def event69933 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21543⟩⟩) (.product (.result 65387 .summary) (.transfer 69932) (⟨false, false, none, none, none⟩))

def event69934 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21543⟩⟩, .operator (⟨65387, 0⟩, ⟨69928, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21540⟩⟩]⟩, (1)⟩)

def event69935 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21541⟩⟩)

def event69936 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event69937 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event69938 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event69939 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event69940 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event69941 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event69942 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event69943 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event69944 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 69943

def event69945 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 69941

def event69946 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 69944 .coefficient) (.value (.predecessor 1 69945 .coefficient)))

def event69947 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event69948 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 69947

def event69949 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 69939

def event69950 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 69948 .coefficient, .predecessor 1 69949 .coefficient])

def event69951 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event69952 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 69951

def event69953 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 69937

def event69954 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 69953 .coefficient))

def event69955 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event69956 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11549⟩⟩) 0 ⟨5530⟩ 69955

def event69957 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11549⟩⟩) (.authority (.programFamilyFact))

def exact69958RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩], []⟩, (1)⟩]

theorem exact69958RawTermsValid :
    exact69958RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69958 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11549⟩⟩) exact69958RawTerms (.finite 22) 69957 .exactZero (none)

def event69959 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14415⟩⟩) 0 ⟨5530⟩ 69955

def event69960 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14415⟩⟩) (.authority (.programFamilyFact))

def exact69961RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14415⟩⟩], []⟩, (1)⟩]

theorem exact69961RawTermsValid :
    exact69961RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69961 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14415⟩⟩) exact69961RawTerms (.finite 22) 69960 .exactZero (none)

def event69962 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14416⟩⟩) 0 ⟨14415⟩ 69961

def event69963 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14416⟩⟩) 1 ⟨11549⟩ 69958

def event69964 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14416⟩⟩) (.product (.predecessor 0 69962 .coefficient) (.predecessor 1 69963 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event69965 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14416⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], []⟩) [⟨.result 69961 .coefficient, true, some 1⟩, ⟨.result 69958 .coefficient, true, some 1⟩])

def event69966 : Event := .survivorFold (1) 69965

def exact69967RawTerms : List Term := []

theorem exact69967RawTermsValid :
    exact69967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69967 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14416⟩⟩) exact69967RawTerms (.finite 484) 69964 (.finite 484) (some (69965))

def event69968 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14417⟩⟩) 0 ⟨14416⟩ 69967

def event69969 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14417⟩⟩) (.identity (.predecessor 0 69968 .coefficient))

def event69970 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14417⟩⟩) (.finite 484)

def event69971 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16055⟩⟩) 0 ⟨14417⟩ 69970

def event69972 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16055⟩⟩) (.authority (.programFamilyFact))

def exact69973RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], []⟩, (1)⟩]

theorem exact69973RawTermsValid :
    exact69973RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69973 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16055⟩⟩) exact69973RawTerms (.finite 22) 69972 .exactZero (none)

def event69974 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16056⟩⟩) 0 ⟨16055⟩ 69973

def event69975 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16056⟩⟩) (.identity (.predecessor 0 69974 .coefficient))

def event69976 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16056⟩⟩) (.finite 22)

def event69977 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21540⟩⟩) 0 ⟨16056⟩ 69976

def event69978 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21540⟩⟩) (.authority (.relationPreimageSource ⟨46⟩))

def exact69979RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21540⟩⟩]⟩, (1)⟩]

theorem exact69979RawTermsValid :
    exact69979RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69979 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21540⟩⟩) exact69979RawTerms (.finite 136065468) 69978 .exactZero (none)

def event69980 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact69981RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact69981RawTermsValid :
    exact69981RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69981 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact69981RawTerms .large 69980 .exactZero (none)

def event69982 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21541⟩⟩) 0 ⟨6⟩ 69981

def event69983 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21541⟩⟩) 1 ⟨21540⟩ 69979

def event69984 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21541⟩⟩) (.product (.predecessor 0 69982 .coefficient) (.predecessor 1 69983 .coefficient) (⟨false, false, none, none, none⟩))

def event69985 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21541⟩⟩, .operator (⟨69981, 0⟩, ⟨69979, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21540⟩⟩]⟩, (1)⟩)

def exact69986RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21540⟩⟩]⟩, (1)⟩]

theorem exact69986RawTermsValid :
    exact69986RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69986 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21541⟩⟩) exact69986RawTerms .large 69984 .exactZero (none)

def event69987 : Event := .preFoldPolynomial 69986 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21540⟩⟩]⟩, (1)⟩] .exactZero none

def exact69988RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21540⟩⟩]⟩, (1)⟩]

def event69988 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21541⟩⟩) 69987 exact69988RawTerms .large 69984 .exactZero (none)

def event69989 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28075⟩⟩)

def event69990 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event69991 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event69992 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event69993 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event69994 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event69995 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event69996 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event69997 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event69998 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 69997

def event69999 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 69995

def event70000 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 69998 .coefficient) (.value (.predecessor 1 69999 .coefficient)))

def event70001 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event70002 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 70001

def event70003 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 69993

def event70004 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 70002 .coefficient, .predecessor 1 70003 .coefficient])

def event70005 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event70006 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 70005

def event70007 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 69991

def event70008 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 70007 .coefficient))

def event70009 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event70010 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11549⟩⟩) 0 ⟨5530⟩ 70009

def event70011 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11549⟩⟩) (.authority (.programFamilyFact))

def exact70012RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩], []⟩, (1)⟩]

theorem exact70012RawTermsValid :
    exact70012RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70012 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11549⟩⟩) exact70012RawTerms (.finite 22) 70011 .exactZero (none)

def event70013 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14415⟩⟩) 0 ⟨5530⟩ 70009

def event70014 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14415⟩⟩) (.authority (.programFamilyFact))

def exact70015RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14415⟩⟩], []⟩, (1)⟩]

theorem exact70015RawTermsValid :
    exact70015RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70015 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14415⟩⟩) exact70015RawTerms (.finite 22) 70014 .exactZero (none)

def event70016 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14416⟩⟩) 0 ⟨14415⟩ 70015

def event70017 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14416⟩⟩) 1 ⟨11549⟩ 70012

def event70018 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14416⟩⟩) (.product (.predecessor 0 70016 .coefficient) (.predecessor 1 70017 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70019 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14416⟩⟩, .operator (⟨70015, 0⟩, ⟨70012, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], []⟩, (1)⟩)

def exact70020RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11549⟩⟩, ⟨.program ⟨214⟩, ⟨14415⟩⟩], []⟩, (1)⟩]

theorem exact70020RawTermsValid :
    exact70020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70020 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14416⟩⟩) exact70020RawTerms (.finite 484) 70018 .exactZero (none)

def event70021 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14417⟩⟩) 0 ⟨14416⟩ 70020

def event70022 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14417⟩⟩) (.identity (.predecessor 0 70021 .coefficient))

def event70023 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14417⟩⟩) (.finite 484)

def event70024 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16055⟩⟩) 0 ⟨14417⟩ 70023

def event70025 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16055⟩⟩) (.authority (.programFamilyFact))

def exact70026RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], []⟩, (1)⟩]

theorem exact70026RawTermsValid :
    exact70026RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70026 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16055⟩⟩) exact70026RawTerms (.finite 22) 70025 .exactZero (none)

def event70027 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16056⟩⟩) 0 ⟨16055⟩ 70026

def event70028 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16056⟩⟩) (.identity (.predecessor 0 70027 .coefficient))

def event70029 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16056⟩⟩) (.finite 22)

def event70030 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24220⟩⟩) 0 ⟨16056⟩ 70029

def event70031 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24220⟩⟩) (.authority (.programFamilyFact))

def event70032 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24220⟩⟩) (.finite 3720)

def event70033 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event70034 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24222⟩⟩) 0 ⟨6689⟩ 70033

def event70035 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24222⟩⟩) 1 ⟨24220⟩ 70032

def event70036 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24222⟩⟩) (.authority (.operator))

def exact70037RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24222⟩⟩]⟩, (1)⟩]

theorem exact70037RawTermsValid :
    exact70037RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70037 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24222⟩⟩) exact70037RawTerms .large 70036 .exactZero (none)

def event70038 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28070⟩⟩) 0 ⟨24222⟩ 70037

def event70039 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28070⟩⟩) (.authority (.operator))

def exact70040RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28070⟩⟩]⟩, (1)⟩]

theorem exact70040RawTermsValid :
    exact70040RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70040 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28070⟩⟩) exact70040RawTerms (.finite 8192) 70039 .exactZero (none)

def event70041 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event70042 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event70043 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16130⟩⟩) 0 ⟨16056⟩ 70029

def event70044 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16130⟩⟩) 1 ⟨110⟩ 70042

def event70045 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16130⟩⟩) (.sum [.predecessor 0 70043 .coefficient, .predecessor 1 70044 .coefficient])

def event70046 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16130⟩⟩) (.finite 22)

def event70047 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16131⟩⟩) 0 ⟨16130⟩ 70046

def event70048 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16131⟩⟩) (.identity (.predecessor 0 70047 .coefficient))

def exact70049RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], []⟩, (1)⟩]

theorem exact70049RawTermsValid :
    exact70049RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70049 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16131⟩⟩) exact70049RawTerms (.finite 22) 70048 .exactZero (none)

def event70050 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact70051RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact70051RawTermsValid :
    exact70051RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70051 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact70051RawTerms .large 70050 .exactZero (none)

def event70052 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16132⟩⟩) 0 ⟨6544⟩ 70051

def event70053 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16132⟩⟩) 1 ⟨16131⟩ 70049

def event70054 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16132⟩⟩) (.product (.predecessor 0 70052 .coefficient) (.predecessor 1 70053 .coefficient) (⟨false, false, none, none, none⟩))

def event70055 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16132⟩⟩, .operator (⟨70051, 0⟩, ⟨70049, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact70056RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact70056RawTermsValid :
    exact70056RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70056 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16132⟩⟩) exact70056RawTerms .large 70054 .exactZero (none)

def event70057 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6698⟩⟩) 0 ⟨6689⟩ 70033

def event70058 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6698⟩⟩) (.authority (.operator))

def exact70059RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩]

theorem exact70059RawTermsValid :
    exact70059RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70059 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6698⟩⟩) exact70059RawTerms .large 70058 .exactZero (none)

def event70060 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16133⟩⟩) 0 ⟨6698⟩ 70059

def event70061 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16133⟩⟩) 1 ⟨16132⟩ 70056

def event70062 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16133⟩⟩) (.sum [.predecessor 0 70060 .coefficient, .predecessor 1 70061 .coefficient])

def exact70063RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70063RawTermsValid :
    exact70063RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70063 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16133⟩⟩) exact70063RawTerms .large 70062 .exactZero (none)

def event70064 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28071⟩⟩) 0 ⟨16133⟩ 70063

def event70065 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28071⟩⟩) 1 ⟨28070⟩ 70040

def event70066 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28071⟩⟩) (.product (.predecessor 0 70064 .coefficient) (.predecessor 1 70065 .coefficient) (⟨false, false, none, none, none⟩))

def event70067 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28071⟩⟩, .operator (⟨70063, 0⟩, ⟨70040, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28070⟩⟩]⟩, (1)⟩)

def event70068 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28071⟩⟩, .operator (⟨70063, 1⟩, ⟨70040, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28070⟩⟩]⟩, (-1)⟩)

def event70069 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28071⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28070⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28070⟩⟩) ⟨24222⟩ 70037)

def event70070 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28071⟩⟩, .relation 70069 0, ⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24222⟩⟩]⟩, (-1)⟩)

def exact70071RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28070⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24222⟩⟩]⟩, (-1)⟩]

theorem exact70071RawTermsValid :
    exact70071RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70071 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28071⟩⟩) exact70071RawTerms .large 70066 .exactZero (none)

def event70072 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16102⟩⟩) 0 ⟨16056⟩ 70029

def event70073 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16102⟩⟩) (.authority (.programFamilyFact))

def exact70074RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16102⟩⟩], []⟩, (1)⟩]

theorem exact70074RawTermsValid :
    exact70074RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70074 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16102⟩⟩) exact70074RawTerms (.finite 61) 70073 .exactZero (none)

def event70075 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16103⟩⟩) 0 ⟨6544⟩ 70051

def event70076 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16103⟩⟩) 1 ⟨16102⟩ 70074

def event70077 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16103⟩⟩) (.product (.predecessor 0 70075 .coefficient) (.predecessor 1 70076 .coefficient) (⟨false, true, none, none, some 1⟩))

def event70078 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16103⟩⟩, .operator (⟨70051, 0⟩, ⟨70074, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16102⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact70079RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16102⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact70079RawTermsValid :
    exact70079RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70079 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16103⟩⟩) exact70079RawTerms .large 70077 .exactZero (none)

def event70080 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6725⟩⟩) 0 ⟨6689⟩ 70033

def event70081 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6725⟩⟩) (.authority (.operator))

def exact70082RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩]

theorem exact70082RawTermsValid :
    exact70082RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70082 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6725⟩⟩) exact70082RawTerms .large 70081 .exactZero (none)

def event70083 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16104⟩⟩) 0 ⟨6725⟩ 70082

def event70084 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16104⟩⟩) 1 ⟨16103⟩ 70079

def event70085 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16104⟩⟩) (.sum [.predecessor 0 70083 .coefficient, .predecessor 1 70084 .coefficient])

def exact70086RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16102⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70086RawTermsValid :
    exact70086RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70086 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16104⟩⟩) exact70086RawTerms .large 70085 .exactZero (none)

def event70087 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28075⟩⟩) 0 ⟨16104⟩ 70086

def event70088 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28075⟩⟩) 1 ⟨28071⟩ 70071

def event70089 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28075⟩⟩) (.sum [.predecessor 0 70087 .coefficient, .predecessor 1 70088 .coefficient])

def exact70090RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28070⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16102⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70090RawTermsValid :
    exact70090RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70090 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28075⟩⟩) exact70090RawTerms .large 70089 .exactZero (none)

def event70091 : Event := .preFoldPolynomial 70090 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28070⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16102⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact70092RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28070⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16102⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event70092 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28075⟩⟩) 70091 exact70092RawTerms .large 70089 .exactZero (none)

def event70093 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16056⟩⟩) ⟨⟨138⟩, ⟨46⟩, ⟨109⟩⟩ ⟨69935, 70093⟩

def event70094 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21543⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21540⟩⟩]⟩) (1) 0 2 (.universal 70093 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21540⟩⟩]⟩) (none) 70092)

def event70095 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21543⟩⟩, .relation 70094 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩)

def event70096 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21543⟩⟩, .relation 70094 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28070⟩⟩]⟩, (-1)⟩)

def event70097 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21543⟩⟩, .relation 70094 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24222⟩⟩]⟩, (1)⟩)

def event70098 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21543⟩⟩, .relation 70094 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16102⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact70099RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28070⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16102⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70099RawTermsValid :
    exact70099RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70099 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21543⟩⟩) exact70099RawTerms .large 69931 (.finite 1811303510016) (some (69933))

def event70100 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28073⟩⟩) 0 ⟨21543⟩ 70099

def event70101 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28073⟩⟩) 1 ⟨28072⟩ 69921

def event70102 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28073⟩⟩) (.sum [.predecessor 0 70100 .coefficient, .predecessor 1 70101 .coefficient])

def event70103 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28073⟩⟩, .operator (⟨70099, 0⟩, ⟨69921, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28070⟩⟩]⟩, (1)⟩)

def event70104 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28073⟩⟩, .operator (⟨70099, 2⟩, ⟨69921, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16055⟩⟩], [⟨.program ⟨214⟩, ⟨24222⟩⟩]⟩, (-1)⟩)

def event70105 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28073⟩⟩) (.sum [.result 70099 .summary, .result 69921 .summary])

def exact70106RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16102⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70106RawTermsValid :
    exact70106RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70106 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28073⟩⟩) exact70106RawTerms .large 70102 (.finite 1292113298829627502592) (some (70105))

def event70107 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24157⟩⟩) 0 ⟨15937⟩ 3333

def event70108 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24157⟩⟩) (.authority (.programFamilyFact))

def event70109 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24157⟩⟩) (.finite 3720)

def event70110 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24159⟩⟩) 0 ⟨6689⟩ 5477

def event70111 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24159⟩⟩) 1 ⟨24157⟩ 70109

def event70112 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24159⟩⟩) (.authority (.operator))

def exact70113RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24159⟩⟩]⟩, (1)⟩]

theorem exact70113RawTermsValid :
    exact70113RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70113 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24159⟩⟩) exact70113RawTerms .large 70112 .exactZero (none)

def event70114 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27853⟩⟩) 0 ⟨24159⟩ 70113

def event70115 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27853⟩⟩) (.authority (.operator))

def exact70116RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27853⟩⟩]⟩, (1)⟩]

theorem exact70116RawTermsValid :
    exact70116RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70116 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27853⟩⟩) exact70116RawTerms (.finite 8192) 70115 .exactZero (none)

def event70117 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23581⟩⟩) 0 ⟨14200⟩ 3327

def event70118 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23581⟩⟩) (.authority (.programFamilyFact))

def event70119 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23581⟩⟩) (.finite 3720)

def event70120 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23582⟩⟩) 0 ⟨6689⟩ 5477

def event70121 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23582⟩⟩) 1 ⟨23581⟩ 70119

def event70122 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23582⟩⟩) (.authority (.operator))

def exact70123RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23582⟩⟩]⟩, (1)⟩]

theorem exact70123RawTermsValid :
    exact70123RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70123 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23582⟩⟩) exact70123RawTerms .large 70122 .exactZero (none)

def event70124 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26061⟩⟩) 0 ⟨23582⟩ 70123

def event70125 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26061⟩⟩) (.authority (.operator))

def exact70126RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26061⟩⟩]⟩, (1)⟩]

theorem exact70126RawTermsValid :
    exact70126RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70126 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26061⟩⟩) exact70126RawTerms (.finite 8192) 70125 .exactZero (none)

def event70127 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11466⟩⟩) 0 ⟨11465⟩ 3316

def event70128 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11466⟩⟩) 1 ⟨6566⟩ 65295

def event70129 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11466⟩⟩) (.tensor (.predecessor 0 70127 .coefficient) (.predecessor 1 70128 .coefficient) true false)

def event70130 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11466⟩⟩, .operator (⟨3316, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11465⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact70131RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11465⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact70131RawTermsValid :
    exact70131RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70131 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11466⟩⟩) exact70131RawTerms .large 70129 .exactZero (none)

def event70132 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7197⟩⟩) 0 ⟨5533⟩ 65165

def event70133 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7197⟩⟩) 1 ⟨6779⟩ 11482

def event70134 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7197⟩⟩) (.product (.predecessor 0 70132 .coefficient) (.predecessor 1 70133 .coefficient) (⟨false, false, none, none, none⟩))

def event70135 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7197⟩⟩, .operator (⟨65165, 0⟩, ⟨11482, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩)

def exact70136RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩]

theorem exact70136RawTermsValid :
    exact70136RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70136 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7197⟩⟩) exact70136RawTerms .large 70134 .exactZero (none)

def event70137 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11467⟩⟩) 0 ⟨7197⟩ 70136

def event70138 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11467⟩⟩) 1 ⟨11466⟩ 70131

def event70139 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11467⟩⟩) (.sum [.predecessor 0 70137 .coefficient, .predecessor 1 70138 .coefficient])

def exact70140RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11465⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70140RawTermsValid :
    exact70140RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70140 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11467⟩⟩) exact70140RawTerms .large 70139 .exactZero (none)

def event70141 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11468⟩⟩) 0 ⟨11467⟩ 70140

def event70142 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11468⟩⟩) 1 ⟨93⟩ 11474

def event70143 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11468⟩⟩) (.sum [.predecessor 0 70141 .coefficient, .predecessor 1 70142 .coefficient])

def eventLeaf4368 : Array AnnotatedEvent := #[
  { event := event69888
    frameStart := 69780 },
  { event := event69889
    frameStart := 69780 },
  { event := event69890
    frameStart := 69780 },
  { event := event69891
    frameStart := 69780 },
  { event := event69892
    frameStart := 69780 },
  { event := event69893
    frameStart := 69780 },
  { event := event69894
    frameStart := 69780 },
  { event := event69895
    frameStart := 69780 },
  { event := event69896
    frameStart := 69780 },
  { event := event69897
    frameStart := 69780 },
  { event := event69898
    frameStart := 0 },
  { event := event69899
    frameStart := 0 },
  { event := event69900
    frameStart := 0 },
  { event := event69901
    frameStart := 0 },
  { event := event69902
    frameStart := 0 },
  { event := event69903
    frameStart := 0 }
]

def eventLeaf4369 : Array AnnotatedEvent := #[
  { event := event69904
    frameStart := 0 },
  { event := event69905
    frameStart := 0 },
  { event := event69906
    frameStart := 0 },
  { event := event69907
    frameStart := 0 },
  { event := event69908
    frameStart := 0 },
  { event := event69909
    frameStart := 0 },
  { event := event69910
    frameStart := 0 },
  { event := event69911
    frameStart := 0 },
  { event := event69912
    frameStart := 0 },
  { event := event69913
    frameStart := 0 },
  { event := event69914
    frameStart := 0 },
  { event := event69915
    frameStart := 0 },
  { event := event69916
    frameStart := 0 },
  { event := event69917
    frameStart := 0 },
  { event := event69918
    frameStart := 0 },
  { event := event69919
    frameStart := 0 }
]

def eventLeaf4370 : Array AnnotatedEvent := #[
  { event := event69920
    frameStart := 0 },
  { event := event69921
    frameStart := 0 },
  { event := event69922
    frameStart := 0 },
  { event := event69923
    frameStart := 0 },
  { event := event69924
    frameStart := 0 },
  { event := event69925
    frameStart := 0 },
  { event := event69926
    frameStart := 0 },
  { event := event69927
    frameStart := 0 },
  { event := event69928
    frameStart := 0 },
  { event := event69929
    frameStart := 0 },
  { event := event69930
    frameStart := 0 },
  { event := event69931
    frameStart := 0 },
  { event := event69932
    frameStart := 0 },
  { event := event69933
    frameStart := 0 },
  { event := event69934
    frameStart := 0 },
  { event := event69935
    frameStart := 69935 }
]

def eventLeaf4371 : Array AnnotatedEvent := #[
  { event := event69936
    frameStart := 69935 },
  { event := event69937
    frameStart := 69935 },
  { event := event69938
    frameStart := 69935 },
  { event := event69939
    frameStart := 69935 },
  { event := event69940
    frameStart := 69935 },
  { event := event69941
    frameStart := 69935 },
  { event := event69942
    frameStart := 69935 },
  { event := event69943
    frameStart := 69935 },
  { event := event69944
    frameStart := 69935 },
  { event := event69945
    frameStart := 69935 },
  { event := event69946
    frameStart := 69935 },
  { event := event69947
    frameStart := 69935 },
  { event := event69948
    frameStart := 69935 },
  { event := event69949
    frameStart := 69935 },
  { event := event69950
    frameStart := 69935 },
  { event := event69951
    frameStart := 69935 }
]

def eventLeaf4372 : Array AnnotatedEvent := #[
  { event := event69952
    frameStart := 69935 },
  { event := event69953
    frameStart := 69935 },
  { event := event69954
    frameStart := 69935 },
  { event := event69955
    frameStart := 69935 },
  { event := event69956
    frameStart := 69935 },
  { event := event69957
    frameStart := 69935 },
  { event := event69958
    frameStart := 69935 },
  { event := event69959
    frameStart := 69935 },
  { event := event69960
    frameStart := 69935 },
  { event := event69961
    frameStart := 69935 },
  { event := event69962
    frameStart := 69935 },
  { event := event69963
    frameStart := 69935 },
  { event := event69964
    frameStart := 69935 },
  { event := event69965
    frameStart := 69935 },
  { event := event69966
    frameStart := 69935 },
  { event := event69967
    frameStart := 69935 }
]

def eventLeaf4373 : Array AnnotatedEvent := #[
  { event := event69968
    frameStart := 69935 },
  { event := event69969
    frameStart := 69935 },
  { event := event69970
    frameStart := 69935 },
  { event := event69971
    frameStart := 69935 },
  { event := event69972
    frameStart := 69935 },
  { event := event69973
    frameStart := 69935 },
  { event := event69974
    frameStart := 69935 },
  { event := event69975
    frameStart := 69935 },
  { event := event69976
    frameStart := 69935 },
  { event := event69977
    frameStart := 69935 },
  { event := event69978
    frameStart := 69935 },
  { event := event69979
    frameStart := 69935 },
  { event := event69980
    frameStart := 69935 },
  { event := event69981
    frameStart := 69935 },
  { event := event69982
    frameStart := 69935 },
  { event := event69983
    frameStart := 69935 }
]

def eventLeaf4374 : Array AnnotatedEvent := #[
  { event := event69984
    frameStart := 69935 },
  { event := event69985
    frameStart := 69935 },
  { event := event69986
    frameStart := 69935 },
  { event := event69987
    frameStart := 69935 },
  { event := event69988
    frameStart := 69935 },
  { event := event69989
    frameStart := 69989 },
  { event := event69990
    frameStart := 69989 },
  { event := event69991
    frameStart := 69989 },
  { event := event69992
    frameStart := 69989 },
  { event := event69993
    frameStart := 69989 },
  { event := event69994
    frameStart := 69989 },
  { event := event69995
    frameStart := 69989 },
  { event := event69996
    frameStart := 69989 },
  { event := event69997
    frameStart := 69989 },
  { event := event69998
    frameStart := 69989 },
  { event := event69999
    frameStart := 69989 }
]

def eventLeaf4375 : Array AnnotatedEvent := #[
  { event := event70000
    frameStart := 69989 },
  { event := event70001
    frameStart := 69989 },
  { event := event70002
    frameStart := 69989 },
  { event := event70003
    frameStart := 69989 },
  { event := event70004
    frameStart := 69989 },
  { event := event70005
    frameStart := 69989 },
  { event := event70006
    frameStart := 69989 },
  { event := event70007
    frameStart := 69989 },
  { event := event70008
    frameStart := 69989 },
  { event := event70009
    frameStart := 69989 },
  { event := event70010
    frameStart := 69989 },
  { event := event70011
    frameStart := 69989 },
  { event := event70012
    frameStart := 69989 },
  { event := event70013
    frameStart := 69989 },
  { event := event70014
    frameStart := 69989 },
  { event := event70015
    frameStart := 69989 }
]

def eventLeaf4376 : Array AnnotatedEvent := #[
  { event := event70016
    frameStart := 69989 },
  { event := event70017
    frameStart := 69989 },
  { event := event70018
    frameStart := 69989 },
  { event := event70019
    frameStart := 69989 },
  { event := event70020
    frameStart := 69989 },
  { event := event70021
    frameStart := 69989 },
  { event := event70022
    frameStart := 69989 },
  { event := event70023
    frameStart := 69989 },
  { event := event70024
    frameStart := 69989 },
  { event := event70025
    frameStart := 69989 },
  { event := event70026
    frameStart := 69989 },
  { event := event70027
    frameStart := 69989 },
  { event := event70028
    frameStart := 69989 },
  { event := event70029
    frameStart := 69989 },
  { event := event70030
    frameStart := 69989 },
  { event := event70031
    frameStart := 69989 }
]

def eventLeaf4377 : Array AnnotatedEvent := #[
  { event := event70032
    frameStart := 69989 },
  { event := event70033
    frameStart := 69989 },
  { event := event70034
    frameStart := 69989 },
  { event := event70035
    frameStart := 69989 },
  { event := event70036
    frameStart := 69989 },
  { event := event70037
    frameStart := 69989 },
  { event := event70038
    frameStart := 69989 },
  { event := event70039
    frameStart := 69989 },
  { event := event70040
    frameStart := 69989 },
  { event := event70041
    frameStart := 69989 },
  { event := event70042
    frameStart := 69989 },
  { event := event70043
    frameStart := 69989 },
  { event := event70044
    frameStart := 69989 },
  { event := event70045
    frameStart := 69989 },
  { event := event70046
    frameStart := 69989 },
  { event := event70047
    frameStart := 69989 }
]

def eventLeaf4378 : Array AnnotatedEvent := #[
  { event := event70048
    frameStart := 69989 },
  { event := event70049
    frameStart := 69989 },
  { event := event70050
    frameStart := 69989 },
  { event := event70051
    frameStart := 69989 },
  { event := event70052
    frameStart := 69989 },
  { event := event70053
    frameStart := 69989 },
  { event := event70054
    frameStart := 69989 },
  { event := event70055
    frameStart := 69989 },
  { event := event70056
    frameStart := 69989 },
  { event := event70057
    frameStart := 69989 },
  { event := event70058
    frameStart := 69989 },
  { event := event70059
    frameStart := 69989 },
  { event := event70060
    frameStart := 69989 },
  { event := event70061
    frameStart := 69989 },
  { event := event70062
    frameStart := 69989 },
  { event := event70063
    frameStart := 69989 }
]

def eventLeaf4379 : Array AnnotatedEvent := #[
  { event := event70064
    frameStart := 69989 },
  { event := event70065
    frameStart := 69989 },
  { event := event70066
    frameStart := 69989 },
  { event := event70067
    frameStart := 69989 },
  { event := event70068
    frameStart := 69989 },
  { event := event70069
    frameStart := 69989 },
  { event := event70070
    frameStart := 69989 },
  { event := event70071
    frameStart := 69989 },
  { event := event70072
    frameStart := 69989 },
  { event := event70073
    frameStart := 69989 },
  { event := event70074
    frameStart := 69989 },
  { event := event70075
    frameStart := 69989 },
  { event := event70076
    frameStart := 69989 },
  { event := event70077
    frameStart := 69989 },
  { event := event70078
    frameStart := 69989 },
  { event := event70079
    frameStart := 69989 }
]

def eventLeaf4380 : Array AnnotatedEvent := #[
  { event := event70080
    frameStart := 69989 },
  { event := event70081
    frameStart := 69989 },
  { event := event70082
    frameStart := 69989 },
  { event := event70083
    frameStart := 69989 },
  { event := event70084
    frameStart := 69989 },
  { event := event70085
    frameStart := 69989 },
  { event := event70086
    frameStart := 69989 },
  { event := event70087
    frameStart := 69989 },
  { event := event70088
    frameStart := 69989 },
  { event := event70089
    frameStart := 69989 },
  { event := event70090
    frameStart := 69989 },
  { event := event70091
    frameStart := 69989 },
  { event := event70092
    frameStart := 69989 },
  { event := event70093
    frameStart := 0 },
  { event := event70094
    frameStart := 0 },
  { event := event70095
    frameStart := 0 }
]

def eventLeaf4381 : Array AnnotatedEvent := #[
  { event := event70096
    frameStart := 0 },
  { event := event70097
    frameStart := 0 },
  { event := event70098
    frameStart := 0 },
  { event := event70099
    frameStart := 0 },
  { event := event70100
    frameStart := 0 },
  { event := event70101
    frameStart := 0 },
  { event := event70102
    frameStart := 0 },
  { event := event70103
    frameStart := 0 },
  { event := event70104
    frameStart := 0 },
  { event := event70105
    frameStart := 0 },
  { event := event70106
    frameStart := 0 },
  { event := event70107
    frameStart := 0 },
  { event := event70108
    frameStart := 0 },
  { event := event70109
    frameStart := 0 },
  { event := event70110
    frameStart := 0 },
  { event := event70111
    frameStart := 0 }
]

def eventLeaf4382 : Array AnnotatedEvent := #[
  { event := event70112
    frameStart := 0 },
  { event := event70113
    frameStart := 0 },
  { event := event70114
    frameStart := 0 },
  { event := event70115
    frameStart := 0 },
  { event := event70116
    frameStart := 0 },
  { event := event70117
    frameStart := 0 },
  { event := event70118
    frameStart := 0 },
  { event := event70119
    frameStart := 0 },
  { event := event70120
    frameStart := 0 },
  { event := event70121
    frameStart := 0 },
  { event := event70122
    frameStart := 0 },
  { event := event70123
    frameStart := 0 },
  { event := event70124
    frameStart := 0 },
  { event := event70125
    frameStart := 0 },
  { event := event70126
    frameStart := 0 },
  { event := event70127
    frameStart := 0 }
]

def eventLeaf4383 : Array AnnotatedEvent := #[
  { event := event70128
    frameStart := 0 },
  { event := event70129
    frameStart := 0 },
  { event := event70130
    frameStart := 0 },
  { event := event70131
    frameStart := 0 },
  { event := event70132
    frameStart := 0 },
  { event := event70133
    frameStart := 0 },
  { event := event70134
    frameStart := 0 },
  { event := event70135
    frameStart := 0 },
  { event := event70136
    frameStart := 0 },
  { event := event70137
    frameStart := 0 },
  { event := event70138
    frameStart := 0 },
  { event := event70139
    frameStart := 0 },
  { event := event70140
    frameStart := 0 },
  { event := event70141
    frameStart := 0 },
  { event := event70142
    frameStart := 0 },
  { event := event70143
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events273
