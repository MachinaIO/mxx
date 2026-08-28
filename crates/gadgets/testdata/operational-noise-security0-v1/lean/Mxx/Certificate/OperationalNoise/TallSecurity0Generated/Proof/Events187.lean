import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events187

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event47872 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.sum [.predecessor 0 47870 .coefficient, .predecessor 1 47871 .coefficient])

def event47873 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.finite 221)

def event47874 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 0 ⟨5512⟩ 47873

def event47875 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 1 ⟨961⟩ 47859

def event47876 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.identity (.predecessor 1 47875 .coefficient))

def event47877 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.finite 224)

def event47878 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11777⟩⟩) 0 ⟨5548⟩ 47877

def event47879 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11777⟩⟩) (.authority (.programFamilyFact))

def exact47880RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11777⟩⟩], []⟩, (1)⟩]

theorem exact47880RawTermsValid :
    exact47880RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47880 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11777⟩⟩) exact47880RawTerms (.finite 30) 47879 .exactZero (none)

def event47881 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9620⟩⟩) 0 ⟨5548⟩ 47877

def event47882 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9620⟩⟩) (.authority (.programFamilyFact))

def exact47883RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9620⟩⟩], []⟩, (1)⟩]

theorem exact47883RawTermsValid :
    exact47883RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47883 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9620⟩⟩) exact47883RawTerms (.finite 30) 47882 .exactZero (none)

def event47884 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11778⟩⟩) 0 ⟨9620⟩ 47883

def event47885 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11778⟩⟩) 1 ⟨11777⟩ 47880

def event47886 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11778⟩⟩) (.product (.predecessor 0 47884 .coefficient) (.predecessor 1 47885 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event47887 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11778⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9620⟩⟩, ⟨.program ⟨214⟩, ⟨11777⟩⟩], []⟩) [⟨.result 47883 .coefficient, true, some 1⟩, ⟨.result 47880 .coefficient, true, some 1⟩])

def event47888 : Event := .survivorFold (1) 47887

def exact47889RawTerms : List Term := []

theorem exact47889RawTermsValid :
    exact47889RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47889 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11778⟩⟩) exact47889RawTerms (.finite 900) 47886 (.finite 900) (some (47887))

def event47890 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11779⟩⟩) 0 ⟨11778⟩ 47889

def event47891 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11779⟩⟩) (.identity (.predecessor 0 47890 .coefficient))

def event47892 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11779⟩⟩) (.finite 900)

def event47893 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16270⟩⟩) 0 ⟨11779⟩ 47892

def event47894 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16270⟩⟩) (.authority (.programFamilyFact))

def exact47895RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], []⟩, (1)⟩]

theorem exact47895RawTermsValid :
    exact47895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47895 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16270⟩⟩) exact47895RawTerms (.finite 30) 47894 .exactZero (none)

def event47896 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16271⟩⟩) 0 ⟨16270⟩ 47895

def event47897 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16271⟩⟩) (.identity (.predecessor 0 47896 .coefficient))

def event47898 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16271⟩⟩) (.finite 30)

def event47899 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21768⟩⟩) 0 ⟨16271⟩ 47898

def event47900 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21768⟩⟩) (.authority (.relationPreimageSource ⟨49⟩))

def exact47901RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21768⟩⟩]⟩, (1)⟩]

theorem exact47901RawTermsValid :
    exact47901RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47901 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21768⟩⟩) exact47901RawTerms (.finite 136065468) 47900 .exactZero (none)

def event47902 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact47903RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact47903RawTermsValid :
    exact47903RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47903 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact47903RawTerms .large 47902 .exactZero (none)

def event47904 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21769⟩⟩) 0 ⟨6⟩ 47903

def event47905 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21769⟩⟩) 1 ⟨21768⟩ 47901

def event47906 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21769⟩⟩) (.product (.predecessor 0 47904 .coefficient) (.predecessor 1 47905 .coefficient) (⟨false, false, none, none, none⟩))

def event47907 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21769⟩⟩, .operator (⟨47903, 0⟩, ⟨47901, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21768⟩⟩]⟩, (1)⟩)

def exact47908RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21768⟩⟩]⟩, (1)⟩]

theorem exact47908RawTermsValid :
    exact47908RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47908 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21769⟩⟩) exact47908RawTerms .large 47906 .exactZero (none)

def event47909 : Event := .preFoldPolynomial 47908 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21768⟩⟩]⟩, (1)⟩] .exactZero none

def exact47910RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21768⟩⟩]⟩, (1)⟩]

def event47910 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21769⟩⟩) 47909 exact47910RawTerms .large 47906 .exactZero (none)

def event47911 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28542⟩⟩)

def event47912 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event47913 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event47914 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.authority (.operator))

def event47915 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.finite 4)

def event47916 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event47917 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event47918 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event47919 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event47920 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 47919

def event47921 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 47917

def event47922 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 47920 .coefficient) (.value (.predecessor 1 47921 .coefficient)))

def event47923 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event47924 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 0 ⟨5503⟩ 47923

def event47925 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 1 ⟨4733⟩ 47915

def event47926 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.sum [.predecessor 0 47924 .coefficient, .predecessor 1 47925 .coefficient])

def event47927 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.finite 221)

def event47928 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 0 ⟨5512⟩ 47927

def event47929 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 1 ⟨961⟩ 47913

def event47930 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.identity (.predecessor 1 47929 .coefficient))

def event47931 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.finite 224)

def event47932 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11777⟩⟩) 0 ⟨5548⟩ 47931

def event47933 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11777⟩⟩) (.authority (.programFamilyFact))

def exact47934RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11777⟩⟩], []⟩, (1)⟩]

theorem exact47934RawTermsValid :
    exact47934RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47934 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11777⟩⟩) exact47934RawTerms (.finite 30) 47933 .exactZero (none)

def event47935 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9620⟩⟩) 0 ⟨5548⟩ 47931

def event47936 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9620⟩⟩) (.authority (.programFamilyFact))

def exact47937RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9620⟩⟩], []⟩, (1)⟩]

theorem exact47937RawTermsValid :
    exact47937RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47937 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9620⟩⟩) exact47937RawTerms (.finite 30) 47936 .exactZero (none)

def event47938 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11778⟩⟩) 0 ⟨9620⟩ 47937

def event47939 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11778⟩⟩) 1 ⟨11777⟩ 47934

def event47940 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11778⟩⟩) (.product (.predecessor 0 47938 .coefficient) (.predecessor 1 47939 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event47941 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11778⟩⟩, .operator (⟨47937, 0⟩, ⟨47934, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9620⟩⟩, ⟨.program ⟨214⟩, ⟨11777⟩⟩], []⟩, (1)⟩)

def exact47942RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9620⟩⟩, ⟨.program ⟨214⟩, ⟨11777⟩⟩], []⟩, (1)⟩]

theorem exact47942RawTermsValid :
    exact47942RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47942 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11778⟩⟩) exact47942RawTerms (.finite 900) 47940 .exactZero (none)

def event47943 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11779⟩⟩) 0 ⟨11778⟩ 47942

def event47944 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11779⟩⟩) (.identity (.predecessor 0 47943 .coefficient))

def event47945 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11779⟩⟩) (.finite 900)

def event47946 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16270⟩⟩) 0 ⟨11779⟩ 47945

def event47947 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16270⟩⟩) (.authority (.programFamilyFact))

def exact47948RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], []⟩, (1)⟩]

theorem exact47948RawTermsValid :
    exact47948RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47948 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16270⟩⟩) exact47948RawTerms (.finite 30) 47947 .exactZero (none)

def event47949 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16271⟩⟩) 0 ⟨16270⟩ 47948

def event47950 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16271⟩⟩) (.identity (.predecessor 0 47949 .coefficient))

def event47951 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16271⟩⟩) (.finite 30)

def event47952 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24355⟩⟩) 0 ⟨16271⟩ 47951

def event47953 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24355⟩⟩) (.authority (.programFamilyFact))

def event47954 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24355⟩⟩) (.finite 3720)

def event47955 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event47956 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24356⟩⟩) 0 ⟨6689⟩ 47955

def event47957 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24356⟩⟩) 1 ⟨24355⟩ 47954

def event47958 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24356⟩⟩) (.authority (.operator))

def exact47959RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24356⟩⟩]⟩, (1)⟩]

theorem exact47959RawTermsValid :
    exact47959RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47959 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24356⟩⟩) exact47959RawTerms .large 47958 .exactZero (none)

def event47960 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28536⟩⟩) 0 ⟨24356⟩ 47959

def event47961 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28536⟩⟩) (.authority (.operator))

def exact47962RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩, (1)⟩]

theorem exact47962RawTermsValid :
    exact47962RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47962 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28536⟩⟩) exact47962RawTerms (.finite 8192) 47961 .exactZero (none)

def event47963 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event47964 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event47965 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16345⟩⟩) 0 ⟨16271⟩ 47951

def event47966 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16345⟩⟩) 1 ⟨110⟩ 47964

def event47967 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16345⟩⟩) (.sum [.predecessor 0 47965 .coefficient, .predecessor 1 47966 .coefficient])

def event47968 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16345⟩⟩) (.finite 30)

def event47969 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16346⟩⟩) 0 ⟨16345⟩ 47968

def event47970 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16346⟩⟩) (.identity (.predecessor 0 47969 .coefficient))

def exact47971RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], []⟩, (1)⟩]

theorem exact47971RawTermsValid :
    exact47971RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47971 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16346⟩⟩) exact47971RawTerms (.finite 30) 47970 .exactZero (none)

def event47972 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact47973RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact47973RawTermsValid :
    exact47973RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47973 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact47973RawTerms .large 47972 .exactZero (none)

def event47974 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16347⟩⟩) 0 ⟨6544⟩ 47973

def event47975 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16347⟩⟩) 1 ⟨16346⟩ 47971

def event47976 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16347⟩⟩) (.product (.predecessor 0 47974 .coefficient) (.predecessor 1 47975 .coefficient) (⟨false, false, none, none, none⟩))

def event47977 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16347⟩⟩, .operator (⟨47973, 0⟩, ⟨47971, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact47978RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact47978RawTermsValid :
    exact47978RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47978 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16347⟩⟩) exact47978RawTerms .large 47976 .exactZero (none)

def event47979 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6700⟩⟩) 0 ⟨6689⟩ 47955

def event47980 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6700⟩⟩) (.authority (.operator))

def exact47981RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩]

theorem exact47981RawTermsValid :
    exact47981RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47981 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6700⟩⟩) exact47981RawTerms .large 47980 .exactZero (none)

def event47982 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16348⟩⟩) 0 ⟨6700⟩ 47981

def event47983 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16348⟩⟩) 1 ⟨16347⟩ 47978

def event47984 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16348⟩⟩) (.sum [.predecessor 0 47982 .coefficient, .predecessor 1 47983 .coefficient])

def exact47985RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact47985RawTermsValid :
    exact47985RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47985 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16348⟩⟩) exact47985RawTerms .large 47984 .exactZero (none)

def event47986 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28537⟩⟩) 0 ⟨16348⟩ 47985

def event47987 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28537⟩⟩) 1 ⟨28536⟩ 47962

def event47988 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28537⟩⟩) (.product (.predecessor 0 47986 .coefficient) (.predecessor 1 47987 .coefficient) (⟨false, false, none, none, none⟩))

def event47989 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28537⟩⟩, .operator (⟨47985, 0⟩, ⟨47962, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩, (1)⟩)

def event47990 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28537⟩⟩, .operator (⟨47985, 1⟩, ⟨47962, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩, (-1)⟩)

def event47991 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28537⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28536⟩⟩) ⟨24356⟩ 47959)

def event47992 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28537⟩⟩, .relation 47991 0, ⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨24356⟩⟩]⟩, (-1)⟩)

def exact47993RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨24356⟩⟩]⟩, (-1)⟩]

theorem exact47993RawTermsValid :
    exact47993RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47993 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28537⟩⟩) exact47993RawTerms .large 47988 .exactZero (none)

def event47994 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17614⟩⟩) 0 ⟨16271⟩ 47951

def event47995 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17614⟩⟩) (.authority (.programFamilyFact))

def exact47996RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17614⟩⟩], []⟩, (1)⟩]

theorem exact47996RawTermsValid :
    exact47996RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47996 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17614⟩⟩) exact47996RawTerms (.finite 30) 47995 .exactZero (none)

def event47997 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17616⟩⟩) 0 ⟨6544⟩ 47973

def event47998 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17616⟩⟩) 1 ⟨17614⟩ 47996

def event47999 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17616⟩⟩) (.product (.predecessor 0 47997 .coefficient) (.predecessor 1 47998 .coefficient) (⟨false, true, none, none, some 1⟩))

def event48000 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17616⟩⟩, .operator (⟨47973, 0⟩, ⟨47996, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact48001RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact48001RawTermsValid :
    exact48001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48001 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17616⟩⟩) exact48001RawTerms .large 47999 .exactZero (none)

def event48002 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6728⟩⟩) 0 ⟨6689⟩ 47955

def event48003 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6728⟩⟩) (.authority (.operator))

def exact48004RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩]⟩, (1)⟩]

theorem exact48004RawTermsValid :
    exact48004RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48004 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6728⟩⟩) exact48004RawTerms .large 48003 .exactZero (none)

def event48005 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17617⟩⟩) 0 ⟨6728⟩ 48004

def event48006 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17617⟩⟩) 1 ⟨17616⟩ 48001

def event48007 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17617⟩⟩) (.sum [.predecessor 0 48005 .coefficient, .predecessor 1 48006 .coefficient])

def exact48008RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48008RawTermsValid :
    exact48008RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48008 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17617⟩⟩) exact48008RawTerms .large 48007 .exactZero (none)

def event48009 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28542⟩⟩) 0 ⟨17617⟩ 48008

def event48010 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28542⟩⟩) 1 ⟨28537⟩ 47993

def event48011 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28542⟩⟩) (.sum [.predecessor 0 48009 .coefficient, .predecessor 1 48010 .coefficient])

def exact48012RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨24356⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48012RawTermsValid :
    exact48012RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48012 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28542⟩⟩) exact48012RawTerms .large 48011 .exactZero (none)

def event48013 : Event := .preFoldPolynomial 48012 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨24356⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact48014RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6728⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨24356⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event48014 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28542⟩⟩) 48013 exact48014RawTerms .large 48011 .exactZero (none)

def event48015 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16271⟩⟩) ⟨⟨141⟩, ⟨49⟩, ⟨109⟩⟩ ⟨47857, 48015⟩

def event48016 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21771⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21768⟩⟩]⟩) (1) 0 2 (.universal 48015 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21768⟩⟩]⟩) (none) 48014)

def event48017 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21771⟩⟩, .relation 48016 1, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩]⟩, (1)⟩)

def event48018 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21771⟩⟩, .relation 48016 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩, (-1)⟩)

def event48019 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21771⟩⟩, .relation 48016 2, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨24356⟩⟩]⟩, (1)⟩)

def event48020 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21771⟩⟩, .relation 48016 3, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact48021RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨24356⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48021RawTermsValid :
    exact48021RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48021 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21771⟩⟩) exact48021RawTerms .large 47853 (.finite 1811303510016) (some (47855))

def event48022 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28539⟩⟩) 0 ⟨21771⟩ 48021

def event48023 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28539⟩⟩) 1 ⟨28538⟩ 47843

def event48024 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28539⟩⟩) (.sum [.predecessor 0 48022 .coefficient, .predecessor 1 48023 .coefficient])

def event48025 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28539⟩⟩, .operator (⟨48021, 0⟩, ⟨47843, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩, (1)⟩)

def event48026 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28539⟩⟩, .operator (⟨48021, 2⟩, ⟨47843, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨24356⟩⟩]⟩, (-1)⟩)

def event48027 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28539⟩⟩) (.sum [.result 48021 .summary, .result 47843 .summary])

def exact48028RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48028RawTermsValid :
    exact48028RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48028 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28539⟩⟩) exact48028RawTerms .large 48024 (.finite 1292202948609709846528) (some (48027))

def event48029 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28540⟩⟩) 0 ⟨28539⟩ 48028

def event48030 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28540⟩⟩) 1 ⟨6678⟩ 5659

def event48031 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28540⟩⟩) (.product (.predecessor 0 48029 .coefficient) (.predecessor 1 48030 .coefficient) (⟨false, false, none, none, none⟩))

def event48032 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28540⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩) [⟨.result 5655 .coefficient, false, none⟩])

def event48033 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28540⟩⟩) (.product (.result 48028 .summary) (.transfer 48032) (⟨false, false, none, none, none⟩))

def event48034 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28540⟩⟩, .operator (⟨48028, 0⟩, ⟨5659, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (1)⟩)

def event48035 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28540⟩⟩, .operator (⟨48028, 1⟩, ⟨5659, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (-1)⟩)

def event48036 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28540⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6677⟩⟩) ⟨6610⟩ 5652)

def event48037 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28540⟩⟩, .relation 48036 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact48038RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact48038RawTermsValid :
    exact48038RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48038 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28540⟩⟩) exact48038RawTerms .large 48031 (.finite 4742405496644812892115304448) (some (48033))

def event48039 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24293⟩⟩) 0 ⟨6689⟩ 5477

def event48040 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24293⟩⟩) 1 ⟨24292⟩ 39895

def event48041 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24293⟩⟩) (.authority (.operator))

def exact48042RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24293⟩⟩]⟩, (1)⟩]

theorem exact48042RawTermsValid :
    exact48042RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48042 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24293⟩⟩) exact48042RawTerms .large 48041 .exactZero (none)

def event48043 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28319⟩⟩) 0 ⟨24293⟩ 48042

def event48044 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28319⟩⟩) (.authority (.operator))

def exact48045RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28319⟩⟩]⟩, (1)⟩]

theorem exact48045RawTermsValid :
    exact48045RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48045 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28319⟩⟩) exact48045RawTerms (.finite 8192) 48044 .exactZero (none)

def event48046 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28321⟩⟩) 0 ⟨26232⟩ 40179

def event48047 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28321⟩⟩) 1 ⟨28319⟩ 48045

def event48048 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28321⟩⟩) (.product (.predecessor 0 48046 .coefficient) (.predecessor 1 48047 .coefficient) (⟨false, false, none, none, none⟩))

def event48049 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28321⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28319⟩⟩]⟩) [⟨.result 48045 .coefficient, false, none⟩])

def event48050 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28321⟩⟩) (.product (.result 40179 .summary) (.transfer 48049) (⟨false, false, none, none, none⟩))

def event48051 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28321⟩⟩, .operator (⟨40179, 0⟩, ⟨48045, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28319⟩⟩]⟩, (1)⟩)

def event48052 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28321⟩⟩, .operator (⟨40179, 1⟩, ⟨48045, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28319⟩⟩]⟩, (-1)⟩)

def event48053 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28321⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28319⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28319⟩⟩) ⟨24293⟩ 48042)

def event48054 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28321⟩⟩, .relation 48053 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨24293⟩⟩]⟩, (-1)⟩)

def exact48055RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28319⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨24293⟩⟩]⟩, (-1)⟩]

theorem exact48055RawTermsValid :
    exact48055RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48055 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28321⟩⟩) exact48055RawTerms .large 48048 (.finite 1292180534353385750528) (some (48050))

def event48056 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21624⟩⟩) 0 ⟨16187⟩ 1791

def event48057 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21624⟩⟩) (.authority (.relationPreimageSource ⟨47⟩))

def exact48058RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21624⟩⟩]⟩, (1)⟩]

theorem exact48058RawTermsValid :
    exact48058RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48058 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21624⟩⟩) exact48058RawTerms (.finite 136065468) 48057 .exactZero (none)

def event48059 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21626⟩⟩) 0 ⟨21624⟩ 48058

def event48060 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21626⟩⟩) 1 ⟨2348⟩ 4

def event48061 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21626⟩⟩) (.scale (.predecessor 0 48059 .coefficient) (.value (.predecessor 1 48060 .coefficient)))

def exact48062RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21624⟩⟩]⟩, (1)⟩]

theorem exact48062RawTermsValid :
    exact48062RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48062 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21626⟩⟩) exact48062RawTerms (.finite 136065468) 48061 .exactZero (none)

def event48063 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21627⟩⟩) 0 ⟨5553⟩ 36137

def event48064 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21627⟩⟩) 1 ⟨21626⟩ 48062

def event48065 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21627⟩⟩) (.product (.predecessor 0 48063 .coefficient) (.predecessor 1 48064 .coefficient) (⟨false, false, none, none, none⟩))

def event48066 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21627⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21624⟩⟩]⟩) [⟨.result 48058 .coefficient, false, none⟩])

def event48067 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21627⟩⟩) (.product (.result 36137 .summary) (.transfer 48066) (⟨false, false, none, none, none⟩))

def event48068 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21627⟩⟩, .operator (⟨36137, 0⟩, ⟨48062, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21624⟩⟩]⟩, (1)⟩)

def event48069 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21625⟩⟩)

def event48070 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event48071 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event48072 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.authority (.operator))

def event48073 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.finite 4)

def event48074 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event48075 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event48076 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event48077 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event48078 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 48077

def event48079 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 48075

def event48080 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 48078 .coefficient) (.value (.predecessor 1 48079 .coefficient)))

def event48081 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event48082 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 0 ⟨5503⟩ 48081

def event48083 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 1 ⟨4733⟩ 48073

def event48084 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.sum [.predecessor 0 48082 .coefficient, .predecessor 1 48083 .coefficient])

def event48085 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.finite 221)

def event48086 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 0 ⟨5512⟩ 48085

def event48087 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 1 ⟨961⟩ 48071

def event48088 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.identity (.predecessor 1 48087 .coefficient))

def event48089 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.finite 224)

def event48090 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11645⟩⟩) 0 ⟨5548⟩ 48089

def event48091 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11645⟩⟩) (.authority (.programFamilyFact))

def exact48092RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11645⟩⟩], []⟩, (1)⟩]

theorem exact48092RawTermsValid :
    exact48092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48092 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11645⟩⟩) exact48092RawTerms (.finite 28) 48091 .exactZero (none)

def event48093 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14659⟩⟩) 0 ⟨5548⟩ 48089

def event48094 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14659⟩⟩) (.authority (.programFamilyFact))

def exact48095RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14659⟩⟩], []⟩, (1)⟩]

theorem exact48095RawTermsValid :
    exact48095RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48095 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14659⟩⟩) exact48095RawTerms (.finite 28) 48094 .exactZero (none)

def event48096 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14660⟩⟩) 0 ⟨14659⟩ 48095

def event48097 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14660⟩⟩) 1 ⟨11645⟩ 48092

def event48098 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14660⟩⟩) (.product (.predecessor 0 48096 .coefficient) (.predecessor 1 48097 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event48099 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14660⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], []⟩) [⟨.result 48095 .coefficient, true, some 1⟩, ⟨.result 48092 .coefficient, true, some 1⟩])

def event48100 : Event := .survivorFold (1) 48099

def exact48101RawTerms : List Term := []

theorem exact48101RawTermsValid :
    exact48101RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48101 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14660⟩⟩) exact48101RawTerms (.finite 784) 48098 (.finite 784) (some (48099))

def event48102 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14661⟩⟩) 0 ⟨14660⟩ 48101

def event48103 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14661⟩⟩) (.identity (.predecessor 0 48102 .coefficient))

def event48104 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14661⟩⟩) (.finite 784)

def event48105 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16186⟩⟩) 0 ⟨14661⟩ 48104

def event48106 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16186⟩⟩) (.authority (.programFamilyFact))

def exact48107RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16186⟩⟩], []⟩, (1)⟩]

theorem exact48107RawTermsValid :
    exact48107RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48107 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16186⟩⟩) exact48107RawTerms (.finite 28) 48106 .exactZero (none)

def event48108 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16187⟩⟩) 0 ⟨16186⟩ 48107

def event48109 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16187⟩⟩) (.identity (.predecessor 0 48108 .coefficient))

def event48110 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16187⟩⟩) (.finite 28)

def event48111 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21624⟩⟩) 0 ⟨16187⟩ 48110

def event48112 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21624⟩⟩) (.authority (.relationPreimageSource ⟨47⟩))

def exact48113RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21624⟩⟩]⟩, (1)⟩]

theorem exact48113RawTermsValid :
    exact48113RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48113 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21624⟩⟩) exact48113RawTerms (.finite 136065468) 48112 .exactZero (none)

def event48114 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact48115RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact48115RawTermsValid :
    exact48115RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48115 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact48115RawTerms .large 48114 .exactZero (none)

def event48116 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21625⟩⟩) 0 ⟨6⟩ 48115

def event48117 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21625⟩⟩) 1 ⟨21624⟩ 48113

def event48118 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21625⟩⟩) (.product (.predecessor 0 48116 .coefficient) (.predecessor 1 48117 .coefficient) (⟨false, false, none, none, none⟩))

def event48119 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21625⟩⟩, .operator (⟨48115, 0⟩, ⟨48113, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21624⟩⟩]⟩, (1)⟩)

def exact48120RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21624⟩⟩]⟩, (1)⟩]

theorem exact48120RawTermsValid :
    exact48120RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48120 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21625⟩⟩) exact48120RawTerms .large 48118 .exactZero (none)

def event48121 : Event := .preFoldPolynomial 48120 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21624⟩⟩]⟩, (1)⟩] .exactZero none

def exact48122RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21624⟩⟩]⟩, (1)⟩]

def event48122 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21625⟩⟩) 48121 exact48122RawTerms .large 48118 .exactZero (none)

def event48123 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28325⟩⟩)

def event48124 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event48125 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event48126 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.authority (.operator))

def event48127 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.finite 4)

def eventLeaf2992 : Array AnnotatedEvent := #[
  { event := event47872
    frameStart := 47857 },
  { event := event47873
    frameStart := 47857 },
  { event := event47874
    frameStart := 47857 },
  { event := event47875
    frameStart := 47857 },
  { event := event47876
    frameStart := 47857 },
  { event := event47877
    frameStart := 47857 },
  { event := event47878
    frameStart := 47857 },
  { event := event47879
    frameStart := 47857 },
  { event := event47880
    frameStart := 47857 },
  { event := event47881
    frameStart := 47857 },
  { event := event47882
    frameStart := 47857 },
  { event := event47883
    frameStart := 47857 },
  { event := event47884
    frameStart := 47857 },
  { event := event47885
    frameStart := 47857 },
  { event := event47886
    frameStart := 47857 },
  { event := event47887
    frameStart := 47857 }
]

def eventLeaf2993 : Array AnnotatedEvent := #[
  { event := event47888
    frameStart := 47857 },
  { event := event47889
    frameStart := 47857 },
  { event := event47890
    frameStart := 47857 },
  { event := event47891
    frameStart := 47857 },
  { event := event47892
    frameStart := 47857 },
  { event := event47893
    frameStart := 47857 },
  { event := event47894
    frameStart := 47857 },
  { event := event47895
    frameStart := 47857 },
  { event := event47896
    frameStart := 47857 },
  { event := event47897
    frameStart := 47857 },
  { event := event47898
    frameStart := 47857 },
  { event := event47899
    frameStart := 47857 },
  { event := event47900
    frameStart := 47857 },
  { event := event47901
    frameStart := 47857 },
  { event := event47902
    frameStart := 47857 },
  { event := event47903
    frameStart := 47857 }
]

def eventLeaf2994 : Array AnnotatedEvent := #[
  { event := event47904
    frameStart := 47857 },
  { event := event47905
    frameStart := 47857 },
  { event := event47906
    frameStart := 47857 },
  { event := event47907
    frameStart := 47857 },
  { event := event47908
    frameStart := 47857 },
  { event := event47909
    frameStart := 47857 },
  { event := event47910
    frameStart := 47857 },
  { event := event47911
    frameStart := 47911 },
  { event := event47912
    frameStart := 47911 },
  { event := event47913
    frameStart := 47911 },
  { event := event47914
    frameStart := 47911 },
  { event := event47915
    frameStart := 47911 },
  { event := event47916
    frameStart := 47911 },
  { event := event47917
    frameStart := 47911 },
  { event := event47918
    frameStart := 47911 },
  { event := event47919
    frameStart := 47911 }
]

def eventLeaf2995 : Array AnnotatedEvent := #[
  { event := event47920
    frameStart := 47911 },
  { event := event47921
    frameStart := 47911 },
  { event := event47922
    frameStart := 47911 },
  { event := event47923
    frameStart := 47911 },
  { event := event47924
    frameStart := 47911 },
  { event := event47925
    frameStart := 47911 },
  { event := event47926
    frameStart := 47911 },
  { event := event47927
    frameStart := 47911 },
  { event := event47928
    frameStart := 47911 },
  { event := event47929
    frameStart := 47911 },
  { event := event47930
    frameStart := 47911 },
  { event := event47931
    frameStart := 47911 },
  { event := event47932
    frameStart := 47911 },
  { event := event47933
    frameStart := 47911 },
  { event := event47934
    frameStart := 47911 },
  { event := event47935
    frameStart := 47911 }
]

def eventLeaf2996 : Array AnnotatedEvent := #[
  { event := event47936
    frameStart := 47911 },
  { event := event47937
    frameStart := 47911 },
  { event := event47938
    frameStart := 47911 },
  { event := event47939
    frameStart := 47911 },
  { event := event47940
    frameStart := 47911 },
  { event := event47941
    frameStart := 47911 },
  { event := event47942
    frameStart := 47911 },
  { event := event47943
    frameStart := 47911 },
  { event := event47944
    frameStart := 47911 },
  { event := event47945
    frameStart := 47911 },
  { event := event47946
    frameStart := 47911 },
  { event := event47947
    frameStart := 47911 },
  { event := event47948
    frameStart := 47911 },
  { event := event47949
    frameStart := 47911 },
  { event := event47950
    frameStart := 47911 },
  { event := event47951
    frameStart := 47911 }
]

def eventLeaf2997 : Array AnnotatedEvent := #[
  { event := event47952
    frameStart := 47911 },
  { event := event47953
    frameStart := 47911 },
  { event := event47954
    frameStart := 47911 },
  { event := event47955
    frameStart := 47911 },
  { event := event47956
    frameStart := 47911 },
  { event := event47957
    frameStart := 47911 },
  { event := event47958
    frameStart := 47911 },
  { event := event47959
    frameStart := 47911 },
  { event := event47960
    frameStart := 47911 },
  { event := event47961
    frameStart := 47911 },
  { event := event47962
    frameStart := 47911 },
  { event := event47963
    frameStart := 47911 },
  { event := event47964
    frameStart := 47911 },
  { event := event47965
    frameStart := 47911 },
  { event := event47966
    frameStart := 47911 },
  { event := event47967
    frameStart := 47911 }
]

def eventLeaf2998 : Array AnnotatedEvent := #[
  { event := event47968
    frameStart := 47911 },
  { event := event47969
    frameStart := 47911 },
  { event := event47970
    frameStart := 47911 },
  { event := event47971
    frameStart := 47911 },
  { event := event47972
    frameStart := 47911 },
  { event := event47973
    frameStart := 47911 },
  { event := event47974
    frameStart := 47911 },
  { event := event47975
    frameStart := 47911 },
  { event := event47976
    frameStart := 47911 },
  { event := event47977
    frameStart := 47911 },
  { event := event47978
    frameStart := 47911 },
  { event := event47979
    frameStart := 47911 },
  { event := event47980
    frameStart := 47911 },
  { event := event47981
    frameStart := 47911 },
  { event := event47982
    frameStart := 47911 },
  { event := event47983
    frameStart := 47911 }
]

def eventLeaf2999 : Array AnnotatedEvent := #[
  { event := event47984
    frameStart := 47911 },
  { event := event47985
    frameStart := 47911 },
  { event := event47986
    frameStart := 47911 },
  { event := event47987
    frameStart := 47911 },
  { event := event47988
    frameStart := 47911 },
  { event := event47989
    frameStart := 47911 },
  { event := event47990
    frameStart := 47911 },
  { event := event47991
    frameStart := 47911 },
  { event := event47992
    frameStart := 47911 },
  { event := event47993
    frameStart := 47911 },
  { event := event47994
    frameStart := 47911 },
  { event := event47995
    frameStart := 47911 },
  { event := event47996
    frameStart := 47911 },
  { event := event47997
    frameStart := 47911 },
  { event := event47998
    frameStart := 47911 },
  { event := event47999
    frameStart := 47911 }
]

def eventLeaf3000 : Array AnnotatedEvent := #[
  { event := event48000
    frameStart := 47911 },
  { event := event48001
    frameStart := 47911 },
  { event := event48002
    frameStart := 47911 },
  { event := event48003
    frameStart := 47911 },
  { event := event48004
    frameStart := 47911 },
  { event := event48005
    frameStart := 47911 },
  { event := event48006
    frameStart := 47911 },
  { event := event48007
    frameStart := 47911 },
  { event := event48008
    frameStart := 47911 },
  { event := event48009
    frameStart := 47911 },
  { event := event48010
    frameStart := 47911 },
  { event := event48011
    frameStart := 47911 },
  { event := event48012
    frameStart := 47911 },
  { event := event48013
    frameStart := 47911 },
  { event := event48014
    frameStart := 47911 },
  { event := event48015
    frameStart := 0 }
]

def eventLeaf3001 : Array AnnotatedEvent := #[
  { event := event48016
    frameStart := 0 },
  { event := event48017
    frameStart := 0 },
  { event := event48018
    frameStart := 0 },
  { event := event48019
    frameStart := 0 },
  { event := event48020
    frameStart := 0 },
  { event := event48021
    frameStart := 0 },
  { event := event48022
    frameStart := 0 },
  { event := event48023
    frameStart := 0 },
  { event := event48024
    frameStart := 0 },
  { event := event48025
    frameStart := 0 },
  { event := event48026
    frameStart := 0 },
  { event := event48027
    frameStart := 0 },
  { event := event48028
    frameStart := 0 },
  { event := event48029
    frameStart := 0 },
  { event := event48030
    frameStart := 0 },
  { event := event48031
    frameStart := 0 }
]

def eventLeaf3002 : Array AnnotatedEvent := #[
  { event := event48032
    frameStart := 0 },
  { event := event48033
    frameStart := 0 },
  { event := event48034
    frameStart := 0 },
  { event := event48035
    frameStart := 0 },
  { event := event48036
    frameStart := 0 },
  { event := event48037
    frameStart := 0 },
  { event := event48038
    frameStart := 0 },
  { event := event48039
    frameStart := 0 },
  { event := event48040
    frameStart := 0 },
  { event := event48041
    frameStart := 0 },
  { event := event48042
    frameStart := 0 },
  { event := event48043
    frameStart := 0 },
  { event := event48044
    frameStart := 0 },
  { event := event48045
    frameStart := 0 },
  { event := event48046
    frameStart := 0 },
  { event := event48047
    frameStart := 0 }
]

def eventLeaf3003 : Array AnnotatedEvent := #[
  { event := event48048
    frameStart := 0 },
  { event := event48049
    frameStart := 0 },
  { event := event48050
    frameStart := 0 },
  { event := event48051
    frameStart := 0 },
  { event := event48052
    frameStart := 0 },
  { event := event48053
    frameStart := 0 },
  { event := event48054
    frameStart := 0 },
  { event := event48055
    frameStart := 0 },
  { event := event48056
    frameStart := 0 },
  { event := event48057
    frameStart := 0 },
  { event := event48058
    frameStart := 0 },
  { event := event48059
    frameStart := 0 },
  { event := event48060
    frameStart := 0 },
  { event := event48061
    frameStart := 0 },
  { event := event48062
    frameStart := 0 },
  { event := event48063
    frameStart := 0 }
]

def eventLeaf3004 : Array AnnotatedEvent := #[
  { event := event48064
    frameStart := 0 },
  { event := event48065
    frameStart := 0 },
  { event := event48066
    frameStart := 0 },
  { event := event48067
    frameStart := 0 },
  { event := event48068
    frameStart := 0 },
  { event := event48069
    frameStart := 48069 },
  { event := event48070
    frameStart := 48069 },
  { event := event48071
    frameStart := 48069 },
  { event := event48072
    frameStart := 48069 },
  { event := event48073
    frameStart := 48069 },
  { event := event48074
    frameStart := 48069 },
  { event := event48075
    frameStart := 48069 },
  { event := event48076
    frameStart := 48069 },
  { event := event48077
    frameStart := 48069 },
  { event := event48078
    frameStart := 48069 },
  { event := event48079
    frameStart := 48069 }
]

def eventLeaf3005 : Array AnnotatedEvent := #[
  { event := event48080
    frameStart := 48069 },
  { event := event48081
    frameStart := 48069 },
  { event := event48082
    frameStart := 48069 },
  { event := event48083
    frameStart := 48069 },
  { event := event48084
    frameStart := 48069 },
  { event := event48085
    frameStart := 48069 },
  { event := event48086
    frameStart := 48069 },
  { event := event48087
    frameStart := 48069 },
  { event := event48088
    frameStart := 48069 },
  { event := event48089
    frameStart := 48069 },
  { event := event48090
    frameStart := 48069 },
  { event := event48091
    frameStart := 48069 },
  { event := event48092
    frameStart := 48069 },
  { event := event48093
    frameStart := 48069 },
  { event := event48094
    frameStart := 48069 },
  { event := event48095
    frameStart := 48069 }
]

def eventLeaf3006 : Array AnnotatedEvent := #[
  { event := event48096
    frameStart := 48069 },
  { event := event48097
    frameStart := 48069 },
  { event := event48098
    frameStart := 48069 },
  { event := event48099
    frameStart := 48069 },
  { event := event48100
    frameStart := 48069 },
  { event := event48101
    frameStart := 48069 },
  { event := event48102
    frameStart := 48069 },
  { event := event48103
    frameStart := 48069 },
  { event := event48104
    frameStart := 48069 },
  { event := event48105
    frameStart := 48069 },
  { event := event48106
    frameStart := 48069 },
  { event := event48107
    frameStart := 48069 },
  { event := event48108
    frameStart := 48069 },
  { event := event48109
    frameStart := 48069 },
  { event := event48110
    frameStart := 48069 },
  { event := event48111
    frameStart := 48069 }
]

def eventLeaf3007 : Array AnnotatedEvent := #[
  { event := event48112
    frameStart := 48069 },
  { event := event48113
    frameStart := 48069 },
  { event := event48114
    frameStart := 48069 },
  { event := event48115
    frameStart := 48069 },
  { event := event48116
    frameStart := 48069 },
  { event := event48117
    frameStart := 48069 },
  { event := event48118
    frameStart := 48069 },
  { event := event48119
    frameStart := 48069 },
  { event := event48120
    frameStart := 48069 },
  { event := event48121
    frameStart := 48069 },
  { event := event48122
    frameStart := 48069 },
  { event := event48123
    frameStart := 48123 },
  { event := event48124
    frameStart := 48123 },
  { event := event48125
    frameStart := 48123 },
  { event := event48126
    frameStart := 48123 },
  { event := event48127
    frameStart := 48123 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events187
