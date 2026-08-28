import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events277

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event70912 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 70911

def event70913 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 70903

def event70914 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 70912 .coefficient, .predecessor 1 70913 .coefficient])

def event70915 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event70916 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 70915

def event70917 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 70901

def event70918 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 70917 .coefficient))

def event70919 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event70920 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11381⟩⟩) 0 ⟨5530⟩ 70919

def event70921 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11381⟩⟩) (.authority (.programFamilyFact))

def exact70922RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩], []⟩, (1)⟩]

theorem exact70922RawTermsValid :
    exact70922RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70922 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11381⟩⟩) exact70922RawTerms (.finite 16) 70921 .exactZero (none)

def event70923 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13981⟩⟩) 0 ⟨5530⟩ 70919

def event70924 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13981⟩⟩) (.authority (.programFamilyFact))

def exact70925RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩, (1)⟩]

theorem exact70925RawTermsValid :
    exact70925RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70925 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13981⟩⟩) exact70925RawTerms (.finite 16) 70924 .exactZero (none)

def event70926 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13982⟩⟩) 0 ⟨13981⟩ 70925

def event70927 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13982⟩⟩) 1 ⟨11381⟩ 70922

def event70928 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13982⟩⟩) (.product (.predecessor 0 70926 .coefficient) (.predecessor 1 70927 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70929 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13982⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩) [⟨.result 70925 .coefficient, true, some 1⟩, ⟨.result 70922 .coefficient, true, some 1⟩])

def event70930 : Event := .survivorFold (1) 70929

def exact70931RawTerms : List Term := []

theorem exact70931RawTermsValid :
    exact70931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70931 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13982⟩⟩) exact70931RawTerms (.finite 256) 70928 (.finite 256) (some (70929))

def event70932 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13983⟩⟩) 0 ⟨13982⟩ 70931

def event70933 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13983⟩⟩) (.identity (.predecessor 0 70932 .coefficient))

def event70934 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13983⟩⟩) (.finite 256)

def event70935 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15817⟩⟩) 0 ⟨13983⟩ 70934

def event70936 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15817⟩⟩) (.authority (.programFamilyFact))

def exact70937RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], []⟩, (1)⟩]

theorem exact70937RawTermsValid :
    exact70937RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70937 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15817⟩⟩) exact70937RawTerms (.finite 16) 70936 .exactZero (none)

def event70938 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15818⟩⟩) 0 ⟨15817⟩ 70937

def event70939 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15818⟩⟩) (.identity (.predecessor 0 70938 .coefficient))

def event70940 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15818⟩⟩) (.finite 16)

def event70941 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21252⟩⟩) 0 ⟨15818⟩ 70940

def event70942 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21252⟩⟩) (.authority (.relationPreimageSource ⟨41⟩))

def exact70943RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21252⟩⟩]⟩, (1)⟩]

theorem exact70943RawTermsValid :
    exact70943RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70943 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21252⟩⟩) exact70943RawTerms (.finite 136065468) 70942 .exactZero (none)

def event70944 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact70945RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact70945RawTermsValid :
    exact70945RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70945 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact70945RawTerms .large 70944 .exactZero (none)

def event70946 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21253⟩⟩) 0 ⟨6⟩ 70945

def event70947 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21253⟩⟩) 1 ⟨21252⟩ 70943

def event70948 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21253⟩⟩) (.product (.predecessor 0 70946 .coefficient) (.predecessor 1 70947 .coefficient) (⟨false, false, none, none, none⟩))

def event70949 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21253⟩⟩, .operator (⟨70945, 0⟩, ⟨70943, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21252⟩⟩]⟩, (1)⟩)

def exact70950RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21252⟩⟩]⟩, (1)⟩]

theorem exact70950RawTermsValid :
    exact70950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70950 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21253⟩⟩) exact70950RawTerms .large 70948 .exactZero (none)

def event70951 : Event := .preFoldPolynomial 70950 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21252⟩⟩]⟩, (1)⟩] .exactZero none

def exact70952RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21252⟩⟩]⟩, (1)⟩]

def event70952 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21253⟩⟩) 70951 exact70952RawTerms .large 70948 .exactZero (none)

def event70953 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27641⟩⟩)

def event70954 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event70955 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event70956 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event70957 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event70958 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event70959 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event70960 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event70961 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event70962 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 70961

def event70963 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 70959

def event70964 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 70962 .coefficient) (.value (.predecessor 1 70963 .coefficient)))

def event70965 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event70966 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 70965

def event70967 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 70957

def event70968 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 70966 .coefficient, .predecessor 1 70967 .coefficient])

def event70969 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event70970 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 70969

def event70971 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 70955

def event70972 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 70971 .coefficient))

def event70973 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event70974 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11381⟩⟩) 0 ⟨5530⟩ 70973

def event70975 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11381⟩⟩) (.authority (.programFamilyFact))

def exact70976RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩], []⟩, (1)⟩]

theorem exact70976RawTermsValid :
    exact70976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70976 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11381⟩⟩) exact70976RawTerms (.finite 16) 70975 .exactZero (none)

def event70977 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13981⟩⟩) 0 ⟨5530⟩ 70973

def event70978 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13981⟩⟩) (.authority (.programFamilyFact))

def exact70979RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩, (1)⟩]

theorem exact70979RawTermsValid :
    exact70979RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70979 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13981⟩⟩) exact70979RawTerms (.finite 16) 70978 .exactZero (none)

def event70980 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13982⟩⟩) 0 ⟨13981⟩ 70979

def event70981 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13982⟩⟩) 1 ⟨11381⟩ 70976

def event70982 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13982⟩⟩) (.product (.predecessor 0 70980 .coefficient) (.predecessor 1 70981 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70983 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13982⟩⟩, .operator (⟨70979, 0⟩, ⟨70976, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩, (1)⟩)

def exact70984RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩, (1)⟩]

theorem exact70984RawTermsValid :
    exact70984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70984 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13982⟩⟩) exact70984RawTerms (.finite 256) 70982 .exactZero (none)

def event70985 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13983⟩⟩) 0 ⟨13982⟩ 70984

def event70986 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13983⟩⟩) (.identity (.predecessor 0 70985 .coefficient))

def event70987 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13983⟩⟩) (.finite 256)

def event70988 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15817⟩⟩) 0 ⟨13983⟩ 70987

def event70989 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15817⟩⟩) (.authority (.programFamilyFact))

def exact70990RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], []⟩, (1)⟩]

theorem exact70990RawTermsValid :
    exact70990RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70990 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15817⟩⟩) exact70990RawTerms (.finite 16) 70989 .exactZero (none)

def event70991 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15818⟩⟩) 0 ⟨15817⟩ 70990

def event70992 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15818⟩⟩) (.identity (.predecessor 0 70991 .coefficient))

def event70993 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15818⟩⟩) (.finite 16)

def event70994 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24094⟩⟩) 0 ⟨15818⟩ 70993

def event70995 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24094⟩⟩) (.authority (.programFamilyFact))

def event70996 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24094⟩⟩) (.finite 3720)

def event70997 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event70998 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24096⟩⟩) 0 ⟨6689⟩ 70997

def event70999 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24096⟩⟩) 1 ⟨24094⟩ 70996

def event71000 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24096⟩⟩) (.authority (.operator))

def exact71001RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24096⟩⟩]⟩, (1)⟩]

theorem exact71001RawTermsValid :
    exact71001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71001 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24096⟩⟩) exact71001RawTerms .large 71000 .exactZero (none)

def event71002 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27636⟩⟩) 0 ⟨24096⟩ 71001

def event71003 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27636⟩⟩) (.authority (.operator))

def exact71004RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27636⟩⟩]⟩, (1)⟩]

theorem exact71004RawTermsValid :
    exact71004RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71004 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27636⟩⟩) exact71004RawTerms (.finite 8192) 71003 .exactZero (none)

def event71005 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event71006 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event71007 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15892⟩⟩) 0 ⟨15818⟩ 70993

def event71008 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15892⟩⟩) 1 ⟨110⟩ 71006

def event71009 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15892⟩⟩) (.sum [.predecessor 0 71007 .coefficient, .predecessor 1 71008 .coefficient])

def event71010 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15892⟩⟩) (.finite 16)

def event71011 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15893⟩⟩) 0 ⟨15892⟩ 71010

def event71012 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15893⟩⟩) (.identity (.predecessor 0 71011 .coefficient))

def exact71013RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], []⟩, (1)⟩]

theorem exact71013RawTermsValid :
    exact71013RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71013 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15893⟩⟩) exact71013RawTerms (.finite 16) 71012 .exactZero (none)

def event71014 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact71015RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact71015RawTermsValid :
    exact71015RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71015 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact71015RawTerms .large 71014 .exactZero (none)

def event71016 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15894⟩⟩) 0 ⟨6544⟩ 71015

def event71017 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15894⟩⟩) 1 ⟨15893⟩ 71013

def event71018 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15894⟩⟩) (.product (.predecessor 0 71016 .coefficient) (.predecessor 1 71017 .coefficient) (⟨false, false, none, none, none⟩))

def event71019 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15894⟩⟩, .operator (⟨71015, 0⟩, ⟨71013, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact71020RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact71020RawTermsValid :
    exact71020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71020 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15894⟩⟩) exact71020RawTerms .large 71018 .exactZero (none)

def event71021 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6696⟩⟩) 0 ⟨6689⟩ 70997

def event71022 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6696⟩⟩) (.authority (.operator))

def exact71023RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩]

theorem exact71023RawTermsValid :
    exact71023RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71023 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6696⟩⟩) exact71023RawTerms .large 71022 .exactZero (none)

def event71024 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15895⟩⟩) 0 ⟨6696⟩ 71023

def event71025 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15895⟩⟩) 1 ⟨15894⟩ 71020

def event71026 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15895⟩⟩) (.sum [.predecessor 0 71024 .coefficient, .predecessor 1 71025 .coefficient])

def exact71027RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71027RawTermsValid :
    exact71027RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71027 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15895⟩⟩) exact71027RawTerms .large 71026 .exactZero (none)

def event71028 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27637⟩⟩) 0 ⟨15895⟩ 71027

def event71029 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27637⟩⟩) 1 ⟨27636⟩ 71004

def event71030 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27637⟩⟩) (.product (.predecessor 0 71028 .coefficient) (.predecessor 1 71029 .coefficient) (⟨false, false, none, none, none⟩))

def event71031 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27637⟩⟩, .operator (⟨71027, 0⟩, ⟨71004, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27636⟩⟩]⟩, (1)⟩)

def event71032 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27637⟩⟩, .operator (⟨71027, 1⟩, ⟨71004, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27636⟩⟩]⟩, (-1)⟩)

def event71033 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27637⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27636⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27636⟩⟩) ⟨24096⟩ 71001)

def event71034 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27637⟩⟩, .relation 71033 0, ⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24096⟩⟩]⟩, (-1)⟩)

def exact71035RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27636⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24096⟩⟩]⟩, (-1)⟩]

theorem exact71035RawTermsValid :
    exact71035RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71035 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27637⟩⟩) exact71035RawTerms .large 71030 .exactZero (none)

def event71036 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15864⟩⟩) 0 ⟨15818⟩ 70993

def event71037 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15864⟩⟩) (.authority (.programFamilyFact))

def exact71038RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], []⟩, (1)⟩]

theorem exact71038RawTermsValid :
    exact71038RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71038 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15864⟩⟩) exact71038RawTerms (.finite 60) 71037 .exactZero (none)

def event71039 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15865⟩⟩) 0 ⟨6544⟩ 71015

def event71040 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15865⟩⟩) 1 ⟨15864⟩ 71038

def event71041 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15865⟩⟩) (.product (.predecessor 0 71039 .coefficient) (.predecessor 1 71040 .coefficient) (⟨false, true, none, none, some 1⟩))

def event71042 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15865⟩⟩, .operator (⟨71015, 0⟩, ⟨71038, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact71043RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact71043RawTermsValid :
    exact71043RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71043 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15865⟩⟩) exact71043RawTerms .large 71041 .exactZero (none)

def event71044 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6721⟩⟩) 0 ⟨6689⟩ 70997

def event71045 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6721⟩⟩) (.authority (.operator))

def exact71046RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩]

theorem exact71046RawTermsValid :
    exact71046RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71046 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6721⟩⟩) exact71046RawTerms .large 71045 .exactZero (none)

def event71047 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15866⟩⟩) 0 ⟨6721⟩ 71046

def event71048 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15866⟩⟩) 1 ⟨15865⟩ 71043

def event71049 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15866⟩⟩) (.sum [.predecessor 0 71047 .coefficient, .predecessor 1 71048 .coefficient])

def exact71050RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71050RawTermsValid :
    exact71050RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71050 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15866⟩⟩) exact71050RawTerms .large 71049 .exactZero (none)

def event71051 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27641⟩⟩) 0 ⟨15866⟩ 71050

def event71052 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27641⟩⟩) 1 ⟨27637⟩ 71035

def event71053 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27641⟩⟩) (.sum [.predecessor 0 71051 .coefficient, .predecessor 1 71052 .coefficient])

def exact71054RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27636⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24096⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71054RawTermsValid :
    exact71054RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71054 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27641⟩⟩) exact71054RawTerms .large 71053 .exactZero (none)

def event71055 : Event := .preFoldPolynomial 71054 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27636⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24096⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact71056RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27636⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24096⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event71056 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27641⟩⟩) 71055 exact71056RawTerms .large 71053 .exactZero (none)

def event71057 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15818⟩⟩) ⟨⟨134⟩, ⟨41⟩, ⟨109⟩⟩ ⟨70899, 71057⟩

def event71058 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21255⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21252⟩⟩]⟩) (1) 0 2 (.universal 71057 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21252⟩⟩]⟩) (none) 71056)

def event71059 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21255⟩⟩, .relation 71058 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩)

def event71060 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21255⟩⟩, .relation 71058 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27636⟩⟩]⟩, (-1)⟩)

def event71061 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21255⟩⟩, .relation 71058 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24096⟩⟩]⟩, (1)⟩)

def event71062 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21255⟩⟩, .relation 71058 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15864⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact71063RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27636⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24096⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15864⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71063RawTermsValid :
    exact71063RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71063 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21255⟩⟩) exact71063RawTerms .large 70895 (.finite 1811303510016) (some (70897))

def event71064 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27639⟩⟩) 0 ⟨21255⟩ 71063

def event71065 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27639⟩⟩) 1 ⟨27638⟩ 70885

def event71066 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27639⟩⟩) (.sum [.predecessor 0 71064 .coefficient, .predecessor 1 71065 .coefficient])

def event71067 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27639⟩⟩, .operator (⟨71063, 0⟩, ⟨70885, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27636⟩⟩]⟩, (1)⟩)

def event71068 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27639⟩⟩, .operator (⟨71063, 2⟩, ⟨70885, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24096⟩⟩]⟩, (-1)⟩)

def event71069 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27639⟩⟩) (.sum [.result 71063 .summary, .result 70885 .summary])

def exact71070RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15864⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71070RawTermsValid :
    exact71070RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71070 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27639⟩⟩) exact71070RawTerms .large 71066 (.finite 1292046061494565744640) (some (71069))

def event71071 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24031⟩⟩) 0 ⟨15699⟩ 3379

def event71072 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24031⟩⟩) (.authority (.programFamilyFact))

def event71073 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24031⟩⟩) (.finite 3720)

def event71074 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24033⟩⟩) 0 ⟨6689⟩ 5477

def event71075 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24033⟩⟩) 1 ⟨24031⟩ 71073

def event71076 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24033⟩⟩) (.authority (.operator))

def exact71077RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24033⟩⟩]⟩, (1)⟩]

theorem exact71077RawTermsValid :
    exact71077RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71077 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24033⟩⟩) exact71077RawTerms .large 71076 .exactZero (none)

def event71078 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27419⟩⟩) 0 ⟨24033⟩ 71077

def event71079 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27419⟩⟩) (.authority (.operator))

def exact71080RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩, (1)⟩]

theorem exact71080RawTermsValid :
    exact71080RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71080 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27419⟩⟩) exact71080RawTerms (.finite 8192) 71079 .exactZero (none)

def event71081 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23497⟩⟩) 0 ⟨13766⟩ 3373

def event71082 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23497⟩⟩) (.authority (.programFamilyFact))

def event71083 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23497⟩⟩) (.finite 3720)

def event71084 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23498⟩⟩) 0 ⟨6689⟩ 5477

def event71085 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23498⟩⟩) 1 ⟨23497⟩ 71083

def event71086 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23498⟩⟩) (.authority (.operator))

def exact71087RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23498⟩⟩]⟩, (1)⟩]

theorem exact71087RawTermsValid :
    exact71087RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71087 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23498⟩⟩) exact71087RawTerms .large 71086 .exactZero (none)

def event71088 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25907⟩⟩) 0 ⟨23498⟩ 71087

def event71089 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25907⟩⟩) (.authority (.operator))

def exact71090RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩, (1)⟩]

theorem exact71090RawTermsValid :
    exact71090RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71090 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25907⟩⟩) exact71090RawTerms (.finite 8192) 71089 .exactZero (none)

def event71091 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11298⟩⟩) 0 ⟨11297⟩ 3362

def event71092 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11298⟩⟩) 1 ⟨6566⟩ 65295

def event71093 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11298⟩⟩) (.tensor (.predecessor 0 71091 .coefficient) (.predecessor 1 71092 .coefficient) true false)

def event71094 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11298⟩⟩, .operator (⟨3362, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact71095RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact71095RawTermsValid :
    exact71095RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71095 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11298⟩⟩) exact71095RawTerms .large 71093 .exactZero (none)

def event71096 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7195⟩⟩) 0 ⟨5533⟩ 65165

def event71097 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7195⟩⟩) 1 ⟨6777⟩ 12484

def event71098 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7195⟩⟩) (.product (.predecessor 0 71096 .coefficient) (.predecessor 1 71097 .coefficient) (⟨false, false, none, none, none⟩))

def event71099 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7195⟩⟩, .operator (⟨65165, 0⟩, ⟨12484, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩)

def exact71100RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩]

theorem exact71100RawTermsValid :
    exact71100RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71100 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7195⟩⟩) exact71100RawTerms .large 71098 .exactZero (none)

def event71101 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11299⟩⟩) 0 ⟨7195⟩ 71100

def event71102 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11299⟩⟩) 1 ⟨11298⟩ 71095

def event71103 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11299⟩⟩) (.sum [.predecessor 0 71101 .coefficient, .predecessor 1 71102 .coefficient])

def exact71104RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71104RawTermsValid :
    exact71104RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71104 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11299⟩⟩) exact71104RawTerms .large 71103 .exactZero (none)

def event71105 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11300⟩⟩) 0 ⟨11299⟩ 71104

def event71106 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11300⟩⟩) 1 ⟨91⟩ 12476

def event71107 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11300⟩⟩) (.sum [.predecessor 0 71105 .coefficient, .predecessor 1 71106 .coefficient])

def event71108 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11300⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨91⟩⟩]⟩) [⟨.result 12476 .coefficient, false, none⟩])

def event71109 : Event := .survivorFold (1) 71108

def exact71110RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71110RawTermsValid :
    exact71110RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71110 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11300⟩⟩) exact71110RawTerms .large 71107 (.finite 26) (some (71108))

def event71111 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13767⟩⟩) 0 ⟨11300⟩ 71110

def event71112 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13767⟩⟩) 1 ⟨13764⟩ 3365

def event71113 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13767⟩⟩) (.product (.predecessor 0 71111 .coefficient) (.predecessor 1 71112 .coefficient) (⟨false, true, none, none, some 1⟩))

def event71114 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13767⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩) [⟨.result 3365 .coefficient, true, some 1⟩])

def event71115 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13767⟩⟩) (.product (.result 71110 .summary) (.transfer 71114) (⟨false, false, none, none, none⟩))

def event71116 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13767⟩⟩, .operator (⟨71110, 1⟩, ⟨3365, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event71117 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13767⟩⟩, .operator (⟨71110, 0⟩, ⟨3365, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩)

def exact71118RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩]

theorem exact71118RawTermsValid :
    exact71118RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71118 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13767⟩⟩) exact71118RawTerms .large 71113 (.finite 9984) (some (71115))

def event71119 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13768⟩⟩) 0 ⟨13764⟩ 3365

def event71120 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13768⟩⟩) 1 ⟨6566⟩ 65295

def event71121 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13768⟩⟩) (.tensor (.predecessor 0 71119 .coefficient) (.predecessor 1 71120 .coefficient) true false)

def event71122 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13768⟩⟩, .operator (⟨3365, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact71123RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact71123RawTermsValid :
    exact71123RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71123 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13768⟩⟩) exact71123RawTerms .large 71121 .exactZero (none)

def event71124 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7212⟩⟩) 0 ⟨5533⟩ 65165

def event71125 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7212⟩⟩) 1 ⟨6794⟩ 12525

def event71126 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7212⟩⟩) (.product (.predecessor 0 71124 .coefficient) (.predecessor 1 71125 .coefficient) (⟨false, false, none, none, none⟩))

def event71127 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7212⟩⟩, .operator (⟨65165, 0⟩, ⟨12525, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩]⟩, (1)⟩)

def exact71128RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩]⟩, (1)⟩]

theorem exact71128RawTermsValid :
    exact71128RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71128 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7212⟩⟩) exact71128RawTerms .large 71126 .exactZero (none)

def event71129 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13769⟩⟩) 0 ⟨7212⟩ 71128

def event71130 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13769⟩⟩) 1 ⟨13768⟩ 71123

def event71131 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13769⟩⟩) (.sum [.predecessor 0 71129 .coefficient, .predecessor 1 71130 .coefficient])

def exact71132RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71132RawTermsValid :
    exact71132RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71132 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13769⟩⟩) exact71132RawTerms .large 71131 .exactZero (none)

def event71133 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13770⟩⟩) 0 ⟨13769⟩ 71132

def event71134 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13770⟩⟩) 1 ⟨108⟩ 12517

def event71135 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13770⟩⟩) (.sum [.predecessor 0 71133 .coefficient, .predecessor 1 71134 .coefficient])

def event71136 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13770⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨108⟩⟩]⟩) [⟨.result 12517 .coefficient, false, none⟩])

def event71137 : Event := .survivorFold (1) 71136

def exact71138RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71138RawTermsValid :
    exact71138RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71138 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13770⟩⟩) exact71138RawTerms .large 71135 (.finite 26) (some (71136))

def event71139 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13771⟩⟩) 0 ⟨13770⟩ 71138

def event71140 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13771⟩⟩) 1 ⟨7847⟩ 12514

def event71141 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13771⟩⟩) (.product (.predecessor 0 71139 .coefficient) (.predecessor 1 71140 .coefficient) (⟨false, false, none, none, none⟩))

def event71142 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13771⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩) [⟨.result 12510 .coefficient, false, none⟩])

def event71143 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13771⟩⟩) (.product (.result 71138 .summary) (.transfer 71142) (⟨false, false, none, none, none⟩))

def event71144 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13771⟩⟩, .operator (⟨71138, 1⟩, ⟨12514, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (-1)⟩)

def event71145 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨13771⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7846⟩⟩) ⟨6777⟩ 12484)

def event71146 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13771⟩⟩, .relation 71145 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (-1)⟩)

def event71147 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13771⟩⟩, .operator (⟨71138, 0⟩, ⟨12514, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (1)⟩)

def exact71148RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (-1)⟩]

theorem exact71148RawTermsValid :
    exact71148RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71148 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13771⟩⟩) exact71148RawTerms .large 71141 (.finite 95420416) (some (71143))

def event71149 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13772⟩⟩) 0 ⟨13771⟩ 71148

def event71150 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13772⟩⟩) 1 ⟨13767⟩ 71118

def event71151 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13772⟩⟩) (.sum [.predecessor 0 71149 .coefficient, .predecessor 1 71150 .coefficient])

def event71152 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13772⟩⟩, .operator (⟨71148, 1⟩, ⟨71118, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩)

def event71153 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13772⟩⟩) (.sum [.result 71148 .summary, .result 71118 .summary])

def exact71154RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71154RawTermsValid :
    exact71154RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71154 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13772⟩⟩) exact71154RawTerms .large 71151 (.finite 95430400) (some (71153))

def event71155 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25908⟩⟩) 0 ⟨13772⟩ 71154

def event71156 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25908⟩⟩) 1 ⟨25907⟩ 71090

def event71157 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25908⟩⟩) (.product (.predecessor 0 71155 .coefficient) (.predecessor 1 71156 .coefficient) (⟨false, false, none, none, none⟩))

def event71158 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25908⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩) [⟨.result 71090 .coefficient, false, none⟩])

def event71159 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25908⟩⟩) (.product (.result 71154 .summary) (.transfer 71158) (⟨false, false, none, none, none⟩))

def event71160 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25908⟩⟩, .operator (⟨71154, 1⟩, ⟨71090, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩, (-1)⟩)

def event71161 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25908⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25907⟩⟩) ⟨23498⟩ 71087)

def event71162 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25908⟩⟩, .relation 71161 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨23498⟩⟩]⟩, (-1)⟩)

def event71163 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25908⟩⟩, .operator (⟨71154, 0⟩, ⟨71090, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩, (1)⟩)

def exact71164RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨23498⟩⟩]⟩, (-1)⟩]

theorem exact71164RawTermsValid :
    exact71164RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71164 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25908⟩⟩) exact71164RawTerms .large 71157 (.finite 350231094886400) (some (71159))

def event71165 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19380⟩⟩) 0 ⟨13766⟩ 3373

def event71166 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19380⟩⟩) (.authority (.relationPreimageSource ⟨13⟩))

def exact71167RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩, (1)⟩]

theorem exact71167RawTermsValid :
    exact71167RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71167 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19380⟩⟩) exact71167RawTerms (.finite 136065468) 71166 .exactZero (none)

def eventLeaf4432 : Array AnnotatedEvent := #[
  { event := event70912
    frameStart := 70899 },
  { event := event70913
    frameStart := 70899 },
  { event := event70914
    frameStart := 70899 },
  { event := event70915
    frameStart := 70899 },
  { event := event70916
    frameStart := 70899 },
  { event := event70917
    frameStart := 70899 },
  { event := event70918
    frameStart := 70899 },
  { event := event70919
    frameStart := 70899 },
  { event := event70920
    frameStart := 70899 },
  { event := event70921
    frameStart := 70899 },
  { event := event70922
    frameStart := 70899 },
  { event := event70923
    frameStart := 70899 },
  { event := event70924
    frameStart := 70899 },
  { event := event70925
    frameStart := 70899 },
  { event := event70926
    frameStart := 70899 },
  { event := event70927
    frameStart := 70899 }
]

def eventLeaf4433 : Array AnnotatedEvent := #[
  { event := event70928
    frameStart := 70899 },
  { event := event70929
    frameStart := 70899 },
  { event := event70930
    frameStart := 70899 },
  { event := event70931
    frameStart := 70899 },
  { event := event70932
    frameStart := 70899 },
  { event := event70933
    frameStart := 70899 },
  { event := event70934
    frameStart := 70899 },
  { event := event70935
    frameStart := 70899 },
  { event := event70936
    frameStart := 70899 },
  { event := event70937
    frameStart := 70899 },
  { event := event70938
    frameStart := 70899 },
  { event := event70939
    frameStart := 70899 },
  { event := event70940
    frameStart := 70899 },
  { event := event70941
    frameStart := 70899 },
  { event := event70942
    frameStart := 70899 },
  { event := event70943
    frameStart := 70899 }
]

def eventLeaf4434 : Array AnnotatedEvent := #[
  { event := event70944
    frameStart := 70899 },
  { event := event70945
    frameStart := 70899 },
  { event := event70946
    frameStart := 70899 },
  { event := event70947
    frameStart := 70899 },
  { event := event70948
    frameStart := 70899 },
  { event := event70949
    frameStart := 70899 },
  { event := event70950
    frameStart := 70899 },
  { event := event70951
    frameStart := 70899 },
  { event := event70952
    frameStart := 70899 },
  { event := event70953
    frameStart := 70953 },
  { event := event70954
    frameStart := 70953 },
  { event := event70955
    frameStart := 70953 },
  { event := event70956
    frameStart := 70953 },
  { event := event70957
    frameStart := 70953 },
  { event := event70958
    frameStart := 70953 },
  { event := event70959
    frameStart := 70953 }
]

def eventLeaf4435 : Array AnnotatedEvent := #[
  { event := event70960
    frameStart := 70953 },
  { event := event70961
    frameStart := 70953 },
  { event := event70962
    frameStart := 70953 },
  { event := event70963
    frameStart := 70953 },
  { event := event70964
    frameStart := 70953 },
  { event := event70965
    frameStart := 70953 },
  { event := event70966
    frameStart := 70953 },
  { event := event70967
    frameStart := 70953 },
  { event := event70968
    frameStart := 70953 },
  { event := event70969
    frameStart := 70953 },
  { event := event70970
    frameStart := 70953 },
  { event := event70971
    frameStart := 70953 },
  { event := event70972
    frameStart := 70953 },
  { event := event70973
    frameStart := 70953 },
  { event := event70974
    frameStart := 70953 },
  { event := event70975
    frameStart := 70953 }
]

def eventLeaf4436 : Array AnnotatedEvent := #[
  { event := event70976
    frameStart := 70953 },
  { event := event70977
    frameStart := 70953 },
  { event := event70978
    frameStart := 70953 },
  { event := event70979
    frameStart := 70953 },
  { event := event70980
    frameStart := 70953 },
  { event := event70981
    frameStart := 70953 },
  { event := event70982
    frameStart := 70953 },
  { event := event70983
    frameStart := 70953 },
  { event := event70984
    frameStart := 70953 },
  { event := event70985
    frameStart := 70953 },
  { event := event70986
    frameStart := 70953 },
  { event := event70987
    frameStart := 70953 },
  { event := event70988
    frameStart := 70953 },
  { event := event70989
    frameStart := 70953 },
  { event := event70990
    frameStart := 70953 },
  { event := event70991
    frameStart := 70953 }
]

def eventLeaf4437 : Array AnnotatedEvent := #[
  { event := event70992
    frameStart := 70953 },
  { event := event70993
    frameStart := 70953 },
  { event := event70994
    frameStart := 70953 },
  { event := event70995
    frameStart := 70953 },
  { event := event70996
    frameStart := 70953 },
  { event := event70997
    frameStart := 70953 },
  { event := event70998
    frameStart := 70953 },
  { event := event70999
    frameStart := 70953 },
  { event := event71000
    frameStart := 70953 },
  { event := event71001
    frameStart := 70953 },
  { event := event71002
    frameStart := 70953 },
  { event := event71003
    frameStart := 70953 },
  { event := event71004
    frameStart := 70953 },
  { event := event71005
    frameStart := 70953 },
  { event := event71006
    frameStart := 70953 },
  { event := event71007
    frameStart := 70953 }
]

def eventLeaf4438 : Array AnnotatedEvent := #[
  { event := event71008
    frameStart := 70953 },
  { event := event71009
    frameStart := 70953 },
  { event := event71010
    frameStart := 70953 },
  { event := event71011
    frameStart := 70953 },
  { event := event71012
    frameStart := 70953 },
  { event := event71013
    frameStart := 70953 },
  { event := event71014
    frameStart := 70953 },
  { event := event71015
    frameStart := 70953 },
  { event := event71016
    frameStart := 70953 },
  { event := event71017
    frameStart := 70953 },
  { event := event71018
    frameStart := 70953 },
  { event := event71019
    frameStart := 70953 },
  { event := event71020
    frameStart := 70953 },
  { event := event71021
    frameStart := 70953 },
  { event := event71022
    frameStart := 70953 },
  { event := event71023
    frameStart := 70953 }
]

def eventLeaf4439 : Array AnnotatedEvent := #[
  { event := event71024
    frameStart := 70953 },
  { event := event71025
    frameStart := 70953 },
  { event := event71026
    frameStart := 70953 },
  { event := event71027
    frameStart := 70953 },
  { event := event71028
    frameStart := 70953 },
  { event := event71029
    frameStart := 70953 },
  { event := event71030
    frameStart := 70953 },
  { event := event71031
    frameStart := 70953 },
  { event := event71032
    frameStart := 70953 },
  { event := event71033
    frameStart := 70953 },
  { event := event71034
    frameStart := 70953 },
  { event := event71035
    frameStart := 70953 },
  { event := event71036
    frameStart := 70953 },
  { event := event71037
    frameStart := 70953 },
  { event := event71038
    frameStart := 70953 },
  { event := event71039
    frameStart := 70953 }
]

def eventLeaf4440 : Array AnnotatedEvent := #[
  { event := event71040
    frameStart := 70953 },
  { event := event71041
    frameStart := 70953 },
  { event := event71042
    frameStart := 70953 },
  { event := event71043
    frameStart := 70953 },
  { event := event71044
    frameStart := 70953 },
  { event := event71045
    frameStart := 70953 },
  { event := event71046
    frameStart := 70953 },
  { event := event71047
    frameStart := 70953 },
  { event := event71048
    frameStart := 70953 },
  { event := event71049
    frameStart := 70953 },
  { event := event71050
    frameStart := 70953 },
  { event := event71051
    frameStart := 70953 },
  { event := event71052
    frameStart := 70953 },
  { event := event71053
    frameStart := 70953 },
  { event := event71054
    frameStart := 70953 },
  { event := event71055
    frameStart := 70953 }
]

def eventLeaf4441 : Array AnnotatedEvent := #[
  { event := event71056
    frameStart := 70953 },
  { event := event71057
    frameStart := 0 },
  { event := event71058
    frameStart := 0 },
  { event := event71059
    frameStart := 0 },
  { event := event71060
    frameStart := 0 },
  { event := event71061
    frameStart := 0 },
  { event := event71062
    frameStart := 0 },
  { event := event71063
    frameStart := 0 },
  { event := event71064
    frameStart := 0 },
  { event := event71065
    frameStart := 0 },
  { event := event71066
    frameStart := 0 },
  { event := event71067
    frameStart := 0 },
  { event := event71068
    frameStart := 0 },
  { event := event71069
    frameStart := 0 },
  { event := event71070
    frameStart := 0 },
  { event := event71071
    frameStart := 0 }
]

def eventLeaf4442 : Array AnnotatedEvent := #[
  { event := event71072
    frameStart := 0 },
  { event := event71073
    frameStart := 0 },
  { event := event71074
    frameStart := 0 },
  { event := event71075
    frameStart := 0 },
  { event := event71076
    frameStart := 0 },
  { event := event71077
    frameStart := 0 },
  { event := event71078
    frameStart := 0 },
  { event := event71079
    frameStart := 0 },
  { event := event71080
    frameStart := 0 },
  { event := event71081
    frameStart := 0 },
  { event := event71082
    frameStart := 0 },
  { event := event71083
    frameStart := 0 },
  { event := event71084
    frameStart := 0 },
  { event := event71085
    frameStart := 0 },
  { event := event71086
    frameStart := 0 },
  { event := event71087
    frameStart := 0 }
]

def eventLeaf4443 : Array AnnotatedEvent := #[
  { event := event71088
    frameStart := 0 },
  { event := event71089
    frameStart := 0 },
  { event := event71090
    frameStart := 0 },
  { event := event71091
    frameStart := 0 },
  { event := event71092
    frameStart := 0 },
  { event := event71093
    frameStart := 0 },
  { event := event71094
    frameStart := 0 },
  { event := event71095
    frameStart := 0 },
  { event := event71096
    frameStart := 0 },
  { event := event71097
    frameStart := 0 },
  { event := event71098
    frameStart := 0 },
  { event := event71099
    frameStart := 0 },
  { event := event71100
    frameStart := 0 },
  { event := event71101
    frameStart := 0 },
  { event := event71102
    frameStart := 0 },
  { event := event71103
    frameStart := 0 }
]

def eventLeaf4444 : Array AnnotatedEvent := #[
  { event := event71104
    frameStart := 0 },
  { event := event71105
    frameStart := 0 },
  { event := event71106
    frameStart := 0 },
  { event := event71107
    frameStart := 0 },
  { event := event71108
    frameStart := 0 },
  { event := event71109
    frameStart := 0 },
  { event := event71110
    frameStart := 0 },
  { event := event71111
    frameStart := 0 },
  { event := event71112
    frameStart := 0 },
  { event := event71113
    frameStart := 0 },
  { event := event71114
    frameStart := 0 },
  { event := event71115
    frameStart := 0 },
  { event := event71116
    frameStart := 0 },
  { event := event71117
    frameStart := 0 },
  { event := event71118
    frameStart := 0 },
  { event := event71119
    frameStart := 0 }
]

def eventLeaf4445 : Array AnnotatedEvent := #[
  { event := event71120
    frameStart := 0 },
  { event := event71121
    frameStart := 0 },
  { event := event71122
    frameStart := 0 },
  { event := event71123
    frameStart := 0 },
  { event := event71124
    frameStart := 0 },
  { event := event71125
    frameStart := 0 },
  { event := event71126
    frameStart := 0 },
  { event := event71127
    frameStart := 0 },
  { event := event71128
    frameStart := 0 },
  { event := event71129
    frameStart := 0 },
  { event := event71130
    frameStart := 0 },
  { event := event71131
    frameStart := 0 },
  { event := event71132
    frameStart := 0 },
  { event := event71133
    frameStart := 0 },
  { event := event71134
    frameStart := 0 },
  { event := event71135
    frameStart := 0 }
]

def eventLeaf4446 : Array AnnotatedEvent := #[
  { event := event71136
    frameStart := 0 },
  { event := event71137
    frameStart := 0 },
  { event := event71138
    frameStart := 0 },
  { event := event71139
    frameStart := 0 },
  { event := event71140
    frameStart := 0 },
  { event := event71141
    frameStart := 0 },
  { event := event71142
    frameStart := 0 },
  { event := event71143
    frameStart := 0 },
  { event := event71144
    frameStart := 0 },
  { event := event71145
    frameStart := 0 },
  { event := event71146
    frameStart := 0 },
  { event := event71147
    frameStart := 0 },
  { event := event71148
    frameStart := 0 },
  { event := event71149
    frameStart := 0 },
  { event := event71150
    frameStart := 0 },
  { event := event71151
    frameStart := 0 }
]

def eventLeaf4447 : Array AnnotatedEvent := #[
  { event := event71152
    frameStart := 0 },
  { event := event71153
    frameStart := 0 },
  { event := event71154
    frameStart := 0 },
  { event := event71155
    frameStart := 0 },
  { event := event71156
    frameStart := 0 },
  { event := event71157
    frameStart := 0 },
  { event := event71158
    frameStart := 0 },
  { event := event71159
    frameStart := 0 },
  { event := event71160
    frameStart := 0 },
  { event := event71161
    frameStart := 0 },
  { event := event71162
    frameStart := 0 },
  { event := event71163
    frameStart := 0 },
  { event := event71164
    frameStart := 0 },
  { event := event71165
    frameStart := 0 },
  { event := event71166
    frameStart := 0 },
  { event := event71167
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events277
