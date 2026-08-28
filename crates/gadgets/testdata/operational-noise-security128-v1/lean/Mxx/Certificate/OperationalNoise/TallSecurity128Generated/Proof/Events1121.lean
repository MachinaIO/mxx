import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1121

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event286976 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52454⟩⟩, .relation 286975 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨51973⟩⟩]⟩, (-1)⟩)

def event286977 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52454⟩⟩, .operator (⟨286968, 0⟩, ⟨286904, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩, (1)⟩)

def exact286978RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨51973⟩⟩]⟩, (-1)⟩]

theorem exact286978RawTermsValid :
    exact286978RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286978 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52454⟩⟩) exact286978RawTerms .large 286971 (.finite 2997687391345233100800) (some (286973))

def event286979 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51389⟩⟩) 0 ⟨50385⟩ 13862

def event286980 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51389⟩⟩) (.authority (.relationPreimageSource ⟨40⟩))

def exact286981RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51389⟩⟩]⟩, (1)⟩]

theorem exact286981RawTermsValid :
    exact286981RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286981 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51389⟩⟩) exact286981RawTerms (.finite 5647228698) 286980 .exactZero (none)

def event286982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51391⟩⟩) 0 ⟨51389⟩ 286981

def event286983 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51391⟩⟩) 1 ⟨2370⟩ 4

def event286984 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51391⟩⟩) (.scale (.predecessor 0 286982 .coefficient) (.value (.predecessor 1 286983 .coefficient)))

def exact286985RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51389⟩⟩]⟩, (1)⟩]

theorem exact286985RawTermsValid :
    exact286985RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event286985 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51391⟩⟩) exact286985RawTerms (.finite 5647228698) 286984 .exactZero (none)

def event286986 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51392⟩⟩) 0 ⟨5491⟩ 280745

def event286987 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51392⟩⟩) 1 ⟨51391⟩ 286985

def event286988 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51392⟩⟩) (.product (.predecessor 0 286986 .coefficient) (.predecessor 1 286987 .coefficient) (⟨false, false, none, none, none⟩))

def event286989 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51392⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨51389⟩⟩]⟩) [⟨.result 286981 .coefficient, false, none⟩])

def event286990 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51392⟩⟩) (.product (.result 280745 .summary) (.transfer 286989) (⟨false, false, none, none, none⟩))

def event286991 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51392⟩⟩, .operator (⟨280745, 0⟩, ⟨286985, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51389⟩⟩]⟩, (1)⟩)

def event286992 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨51390⟩⟩)

def event286993 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event286994 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event286995 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event286996 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event286997 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event286998 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event286999 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event287000 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event287001 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 287000

def event287002 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 286998

def event287003 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 287001 .coefficient) (.value (.predecessor 1 287002 .coefficient)))

def event287004 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event287005 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 287004

def event287006 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 286996

def event287007 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 287005 .coefficient, .predecessor 1 287006 .coefficient])

def event287008 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event287009 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 287008

def event287010 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 286994

def event287011 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 287010 .coefficient))

def event287012 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event287013 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24458⟩⟩) 0 ⟨5487⟩ 287012

def event287014 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24458⟩⟩) (.authority (.programFamilyFact))

def exact287015RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24458⟩⟩], []⟩, (1)⟩]

theorem exact287015RawTermsValid :
    exact287015RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287015 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24458⟩⟩) exact287015RawTerms (.finite 10) 287014 .exactZero (none)

def event287016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50383⟩⟩) 0 ⟨5487⟩ 287012

def event287017 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50383⟩⟩) (.authority (.programFamilyFact))

def exact287018RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50383⟩⟩], []⟩, (1)⟩]

theorem exact287018RawTermsValid :
    exact287018RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287018 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50383⟩⟩) exact287018RawTerms (.finite 10) 287017 .exactZero (none)

def event287019 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50384⟩⟩) 0 ⟨50383⟩ 287018

def event287020 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50384⟩⟩) 1 ⟨24458⟩ 287015

def event287021 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50384⟩⟩) (.product (.predecessor 0 287019 .coefficient) (.predecessor 1 287020 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event287022 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50384⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], []⟩) [⟨.result 287018 .coefficient, true, some 1⟩, ⟨.result 287015 .coefficient, true, some 1⟩])

def event287023 : Event := .survivorFold (1) 287022

def exact287024RawTerms : List Term := []

theorem exact287024RawTermsValid :
    exact287024RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287024 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50384⟩⟩) exact287024RawTerms (.finite 100) 287021 (.finite 100) (some (287022))

def event287025 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50385⟩⟩) 0 ⟨50384⟩ 287024

def event287026 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50385⟩⟩) (.identity (.predecessor 0 287025 .coefficient))

def event287027 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50385⟩⟩) (.finite 100)

def event287028 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51389⟩⟩) 0 ⟨50385⟩ 287027

def event287029 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51389⟩⟩) (.authority (.relationPreimageSource ⟨40⟩))

def exact287030RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51389⟩⟩]⟩, (1)⟩]

theorem exact287030RawTermsValid :
    exact287030RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287030 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51389⟩⟩) exact287030RawTerms (.finite 5647228698) 287029 .exactZero (none)

def event287031 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact287032RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact287032RawTermsValid :
    exact287032RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287032 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact287032RawTerms .large 287031 .exactZero (none)

def event287033 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51390⟩⟩) 0 ⟨35⟩ 287032

def event287034 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51390⟩⟩) 1 ⟨51389⟩ 287030

def event287035 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51390⟩⟩) (.product (.predecessor 0 287033 .coefficient) (.predecessor 1 287034 .coefficient) (⟨false, false, none, none, none⟩))

def event287036 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51390⟩⟩, .operator (⟨287032, 0⟩, ⟨287030, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51389⟩⟩]⟩, (1)⟩)

def exact287037RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51389⟩⟩]⟩, (1)⟩]

theorem exact287037RawTermsValid :
    exact287037RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287037 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51390⟩⟩) exact287037RawTerms .large 287035 .exactZero (none)

def event287038 : Event := .preFoldPolynomial 287037 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51389⟩⟩]⟩, (1)⟩] .exactZero none

def exact287039RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51389⟩⟩]⟩, (1)⟩]

def event287039 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨51390⟩⟩) 287038 exact287039RawTerms .large 287035 .exactZero (none)

def event287040 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨52457⟩⟩)

def event287041 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event287042 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event287043 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event287044 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event287045 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event287046 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event287047 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event287048 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event287049 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 287048

def event287050 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 287046

def event287051 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 287049 .coefficient) (.value (.predecessor 1 287050 .coefficient)))

def event287052 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event287053 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 287052

def event287054 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 287044

def event287055 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 287053 .coefficient, .predecessor 1 287054 .coefficient])

def event287056 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event287057 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 287056

def event287058 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 287042

def event287059 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 287058 .coefficient))

def event287060 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event287061 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24458⟩⟩) 0 ⟨5487⟩ 287060

def event287062 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24458⟩⟩) (.authority (.programFamilyFact))

def exact287063RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24458⟩⟩], []⟩, (1)⟩]

theorem exact287063RawTermsValid :
    exact287063RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287063 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24458⟩⟩) exact287063RawTerms (.finite 10) 287062 .exactZero (none)

def event287064 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50383⟩⟩) 0 ⟨5487⟩ 287060

def event287065 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50383⟩⟩) (.authority (.programFamilyFact))

def exact287066RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50383⟩⟩], []⟩, (1)⟩]

theorem exact287066RawTermsValid :
    exact287066RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287066 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50383⟩⟩) exact287066RawTerms (.finite 10) 287065 .exactZero (none)

def event287067 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50384⟩⟩) 0 ⟨50383⟩ 287066

def event287068 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50384⟩⟩) 1 ⟨24458⟩ 287063

def event287069 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50384⟩⟩) (.product (.predecessor 0 287067 .coefficient) (.predecessor 1 287068 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event287070 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50384⟩⟩, .operator (⟨287066, 0⟩, ⟨287063, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], []⟩, (1)⟩)

def exact287071RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], []⟩, (1)⟩]

theorem exact287071RawTermsValid :
    exact287071RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287071 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50384⟩⟩) exact287071RawTerms (.finite 100) 287069 .exactZero (none)

def event287072 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50385⟩⟩) 0 ⟨50384⟩ 287071

def event287073 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50385⟩⟩) (.identity (.predecessor 0 287072 .coefficient))

def event287074 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50385⟩⟩) (.finite 100)

def event287075 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51972⟩⟩) 0 ⟨50385⟩ 287074

def event287076 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51972⟩⟩) (.authority (.programFamilyFact))

def event287077 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨51972⟩⟩) (.finite 3720)

def event287078 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event287079 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51973⟩⟩) 0 ⟨7177⟩ 287078

def event287080 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51973⟩⟩) 1 ⟨51972⟩ 287077

def event287081 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51973⟩⟩) (.authority (.operator))

def exact287082RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51973⟩⟩]⟩, (1)⟩]

theorem exact287082RawTermsValid :
    exact287082RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287082 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51973⟩⟩) exact287082RawTerms .large 287081 .exactZero (none)

def event287083 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52453⟩⟩) 0 ⟨51973⟩ 287082

def event287084 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52453⟩⟩) (.authority (.operator))

def exact287085RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩, (1)⟩]

theorem exact287085RawTermsValid :
    exact287085RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287085 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52453⟩⟩) exact287085RawTerms (.finite 8192) 287084 .exactZero (none)

def event287086 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event287087 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event287088 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52262⟩⟩) 0 ⟨50385⟩ 287074

def event287089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52262⟩⟩) 1 ⟨136⟩ 287087

def event287090 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52262⟩⟩) (.sum [.predecessor 0 287088 .coefficient, .predecessor 1 287089 .coefficient])

def event287091 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52262⟩⟩) (.finite 100)

def event287092 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52263⟩⟩) 0 ⟨52262⟩ 287091

def event287093 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52263⟩⟩) (.identity (.predecessor 0 287092 .coefficient))

def exact287094RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], []⟩, (1)⟩]

theorem exact287094RawTermsValid :
    exact287094RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287094 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52263⟩⟩) exact287094RawTerms (.finite 100) 287093 .exactZero (none)

def event287095 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact287096RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact287096RawTermsValid :
    exact287096RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287096 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact287096RawTerms .large 287095 .exactZero (none)

def event287097 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52264⟩⟩) 0 ⟨6908⟩ 287096

def event287098 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52264⟩⟩) 1 ⟨52263⟩ 287094

def event287099 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52264⟩⟩) (.product (.predecessor 0 287097 .coefficient) (.predecessor 1 287098 .coefficient) (⟨false, false, none, none, none⟩))

def event287100 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52264⟩⟩, .operator (⟨287096, 0⟩, ⟨287094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact287101RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact287101RawTermsValid :
    exact287101RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287101 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52264⟩⟩) exact287101RawTerms .large 287099 .exactZero (none)

def event287102 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 287078

def event287103 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact287104RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact287104RawTermsValid :
    exact287104RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287104 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact287104RawTerms .large 287103 .exactZero (none)

def event287105 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7308⟩⟩) 0 ⟨7178⟩ 287104

def event287106 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7308⟩⟩) (.identity (.predecessor 0 287105 .coefficient))

def exact287107RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩]

theorem exact287107RawTermsValid :
    exact287107RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287107 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7308⟩⟩) exact287107RawTerms .large 287106 .exactZero (none)

def event287108 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9580⟩⟩) 0 ⟨7308⟩ 287107

def event287109 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9580⟩⟩) (.authority (.operator))

def exact287110RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩]

theorem exact287110RawTermsValid :
    exact287110RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287110 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9580⟩⟩) exact287110RawTerms (.finite 8192) 287109 .exactZero (none)

def event287111 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9581⟩⟩) 0 ⟨9580⟩ 287110

def event287112 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9581⟩⟩) 1 ⟨2370⟩ 287044

def event287113 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9581⟩⟩) (.scale (.predecessor 0 287111 .coefficient) (.value (.predecessor 1 287112 .coefficient)))

def exact287114RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩]

theorem exact287114RawTermsValid :
    exact287114RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287114 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9581⟩⟩) exact287114RawTerms (.finite 8192) 287113 .exactZero (none)

def event287115 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7288⟩⟩) 0 ⟨7178⟩ 287104

def event287116 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7288⟩⟩) (.identity (.predecessor 0 287115 .coefficient))

def exact287117RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩]

theorem exact287117RawTermsValid :
    exact287117RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287117 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7288⟩⟩) exact287117RawTerms .large 287116 .exactZero (none)

def event287118 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9582⟩⟩) 0 ⟨7288⟩ 287117

def event287119 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9582⟩⟩) 1 ⟨9581⟩ 287114

def event287120 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9582⟩⟩) (.product (.predecessor 0 287118 .coefficient) (.predecessor 1 287119 .coefficient) (⟨false, false, none, none, none⟩))

def event287121 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9582⟩⟩, .operator (⟨287117, 0⟩, ⟨287114, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩)

def exact287122RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩]

theorem exact287122RawTermsValid :
    exact287122RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287122 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9582⟩⟩) exact287122RawTerms .large 287120 .exactZero (none)

def event287123 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52265⟩⟩) 0 ⟨9582⟩ 287122

def event287124 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52265⟩⟩) 1 ⟨52264⟩ 287101

def event287125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52265⟩⟩) (.sum [.predecessor 0 287123 .coefficient, .predecessor 1 287124 .coefficient])

def exact287126RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact287126RawTermsValid :
    exact287126RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287126 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52265⟩⟩) exact287126RawTerms .large 287125 .exactZero (none)

def event287127 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52456⟩⟩) 0 ⟨52265⟩ 287126

def event287128 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52456⟩⟩) 1 ⟨52453⟩ 287085

def event287129 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52456⟩⟩) (.product (.predecessor 0 287127 .coefficient) (.predecessor 1 287128 .coefficient) (⟨false, false, none, none, none⟩))

def event287130 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52456⟩⟩, .operator (⟨287126, 0⟩, ⟨287085, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩, (1)⟩)

def event287131 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52456⟩⟩, .operator (⟨287126, 1⟩, ⟨287085, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩, (-1)⟩)

def event287132 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨52456⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52453⟩⟩) ⟨51973⟩ 287082)

def event287133 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52456⟩⟩, .relation 287132 0, ⟨[⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨51973⟩⟩]⟩, (-1)⟩)

def exact287134RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨51973⟩⟩]⟩, (-1)⟩]

theorem exact287134RawTermsValid :
    exact287134RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287134 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52456⟩⟩) exact287134RawTerms .large 287129 .exactZero (none)

def event287135 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50840⟩⟩) 0 ⟨50385⟩ 287074

def event287136 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50840⟩⟩) (.authority (.programFamilyFact))

def exact287137RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50840⟩⟩], []⟩, (1)⟩]

theorem exact287137RawTermsValid :
    exact287137RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287137 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50840⟩⟩) exact287137RawTerms (.finite 10) 287136 .exactZero (none)

def event287138 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50842⟩⟩) 0 ⟨6908⟩ 287096

def event287139 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50842⟩⟩) 1 ⟨50840⟩ 287137

def event287140 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50842⟩⟩) (.product (.predecessor 0 287138 .coefficient) (.predecessor 1 287139 .coefficient) (⟨false, true, none, none, some 1⟩))

def event287141 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50842⟩⟩, .operator (⟨287096, 0⟩, ⟨287137, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact287142RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact287142RawTermsValid :
    exact287142RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287142 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50842⟩⟩) exact287142RawTerms .large 287140 .exactZero (none)

def event287143 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7183⟩⟩) 0 ⟨7177⟩ 287078

def event287144 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7183⟩⟩) (.authority (.operator))

def exact287145RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩]

theorem exact287145RawTermsValid :
    exact287145RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287145 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7183⟩⟩) exact287145RawTerms .large 287144 .exactZero (none)

def event287146 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50843⟩⟩) 0 ⟨7183⟩ 287145

def event287147 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50843⟩⟩) 1 ⟨50842⟩ 287142

def event287148 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50843⟩⟩) (.sum [.predecessor 0 287146 .coefficient, .predecessor 1 287147 .coefficient])

def exact287149RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact287149RawTermsValid :
    exact287149RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287149 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50843⟩⟩) exact287149RawTerms .large 287148 .exactZero (none)

def event287150 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52457⟩⟩) 0 ⟨50843⟩ 287149

def event287151 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52457⟩⟩) 1 ⟨52456⟩ 287134

def event287152 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52457⟩⟩) (.sum [.predecessor 0 287150 .coefficient, .predecessor 1 287151 .coefficient])

def exact287153RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨51973⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact287153RawTermsValid :
    exact287153RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287153 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52457⟩⟩) exact287153RawTerms .large 287152 .exactZero (none)

def event287154 : Event := .preFoldPolynomial 287153 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨51973⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact287155RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨51973⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event287155 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨52457⟩⟩) 287154 exact287155RawTerms .large 287152 .exactZero (none)

def event287156 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨50385⟩⟩) ⟨⟨62⟩, ⟨40⟩, ⟨135⟩⟩ ⟨286992, 287156⟩

def event287157 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨51392⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51389⟩⟩]⟩) (1) 0 2 (.universal 287156 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51389⟩⟩]⟩) (none) 287155)

def event287158 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51392⟩⟩, .relation 287157 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩)

def event287159 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51392⟩⟩, .relation 287157 1, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩, (-1)⟩)

def event287160 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51392⟩⟩, .relation 287157 2, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨51973⟩⟩]⟩, (1)⟩)

def event287161 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51392⟩⟩, .relation 287157 3, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact287162RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨51973⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact287162RawTermsValid :
    exact287162RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287162 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51392⟩⟩) exact287162RawTerms .large 286988 (.finite 202072841853861888) (some (286990))

def event287163 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52455⟩⟩) 0 ⟨51392⟩ 287162

def event287164 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52455⟩⟩) 1 ⟨52454⟩ 286978

def event287165 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52455⟩⟩) (.sum [.predecessor 0 287163 .coefficient, .predecessor 1 287164 .coefficient])

def event287166 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52455⟩⟩, .operator (⟨287162, 2⟩, ⟨286978, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], [⟨.program ⟨257⟩, ⟨51973⟩⟩]⟩, (-1)⟩)

def event287167 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52455⟩⟩, .operator (⟨287162, 1⟩, ⟨286978, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩, (1)⟩)

def event287168 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52455⟩⟩) (.sum [.result 287162 .summary, .result 286978 .summary])

def exact287169RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact287169RawTermsValid :
    exact287169RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287169 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52455⟩⟩) exact287169RawTerms .large 287165 (.finite 2997889464187086962688) (some (287168))

def event287170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52768⟩⟩) 0 ⟨52455⟩ 287169

def event287171 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52768⟩⟩) 1 ⟨52766⟩ 286894

def event287172 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52768⟩⟩) (.product (.predecessor 0 287170 .coefficient) (.predecessor 1 287171 .coefficient) (⟨false, false, none, none, none⟩))

def event287173 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52768⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨52766⟩⟩]⟩) [⟨.result 286894 .coefficient, false, none⟩])

def event287174 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52768⟩⟩) (.product (.result 287169 .summary) (.transfer 287173) (⟨false, false, none, none, none⟩))

def event287175 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52768⟩⟩, .operator (⟨287169, 0⟩, ⟨286894, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52766⟩⟩]⟩, (1)⟩)

def event287176 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52768⟩⟩, .operator (⟨287169, 1⟩, ⟨286894, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52766⟩⟩]⟩, (-1)⟩)

def event287177 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨52768⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52766⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52766⟩⟩) ⟨52107⟩ 286891)

def event287178 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52768⟩⟩, .relation 287177 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨52107⟩⟩]⟩, (-1)⟩)

def exact287179RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52766⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨50840⟩⟩], [⟨.program ⟨257⟩, ⟨52107⟩⟩]⟩, (-1)⟩]

theorem exact287179RawTermsValid :
    exact287179RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287179 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52768⟩⟩) exact287179RawTerms .large 287172 (.finite 32189593014266254325632330629120) (some (287174))

def event287180 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51636⟩⟩) 0 ⟨50841⟩ 13868

def event287181 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51636⟩⟩) (.authority (.relationPreimageSource ⟨65⟩))

def exact287182RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51636⟩⟩]⟩, (1)⟩]

theorem exact287182RawTermsValid :
    exact287182RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287182 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51636⟩⟩) exact287182RawTerms (.finite 5647228698) 287181 .exactZero (none)

def event287183 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51638⟩⟩) 0 ⟨51636⟩ 287182

def event287184 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51638⟩⟩) 1 ⟨2370⟩ 4

def event287185 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51638⟩⟩) (.scale (.predecessor 0 287183 .coefficient) (.value (.predecessor 1 287184 .coefficient)))

def exact287186RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51636⟩⟩]⟩, (1)⟩]

theorem exact287186RawTermsValid :
    exact287186RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287186 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51638⟩⟩) exact287186RawTerms (.finite 5647228698) 287185 .exactZero (none)

def event287187 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51639⟩⟩) 0 ⟨5491⟩ 280745

def event287188 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51639⟩⟩) 1 ⟨51638⟩ 287186

def event287189 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51639⟩⟩) (.product (.predecessor 0 287187 .coefficient) (.predecessor 1 287188 .coefficient) (⟨false, false, none, none, none⟩))

def event287190 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51639⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨51636⟩⟩]⟩) [⟨.result 287182 .coefficient, false, none⟩])

def event287191 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51639⟩⟩) (.product (.result 280745 .summary) (.transfer 287190) (⟨false, false, none, none, none⟩))

def event287192 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51639⟩⟩, .operator (⟨280745, 0⟩, ⟨287186, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51636⟩⟩]⟩, (1)⟩)

def event287193 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨51637⟩⟩)

def event287194 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event287195 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event287196 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event287197 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event287198 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event287199 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event287200 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event287201 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event287202 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 287201

def event287203 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 287199

def event287204 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 287202 .coefficient) (.value (.predecessor 1 287203 .coefficient)))

def event287205 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event287206 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 287205

def event287207 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 287197

def event287208 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 287206 .coefficient, .predecessor 1 287207 .coefficient])

def event287209 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event287210 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 287209

def event287211 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 287195

def event287212 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 287211 .coefficient))

def event287213 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event287214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24458⟩⟩) 0 ⟨5487⟩ 287213

def event287215 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24458⟩⟩) (.authority (.programFamilyFact))

def exact287216RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24458⟩⟩], []⟩, (1)⟩]

theorem exact287216RawTermsValid :
    exact287216RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287216 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24458⟩⟩) exact287216RawTerms (.finite 10) 287215 .exactZero (none)

def event287217 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50383⟩⟩) 0 ⟨5487⟩ 287213

def event287218 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50383⟩⟩) (.authority (.programFamilyFact))

def exact287219RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50383⟩⟩], []⟩, (1)⟩]

theorem exact287219RawTermsValid :
    exact287219RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287219 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50383⟩⟩) exact287219RawTerms (.finite 10) 287218 .exactZero (none)

def event287220 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50384⟩⟩) 0 ⟨50383⟩ 287219

def event287221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50384⟩⟩) 1 ⟨24458⟩ 287216

def event287222 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50384⟩⟩) (.product (.predecessor 0 287220 .coefficient) (.predecessor 1 287221 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event287223 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50384⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24458⟩⟩, ⟨.program ⟨257⟩, ⟨50383⟩⟩], []⟩) [⟨.result 287219 .coefficient, true, some 1⟩, ⟨.result 287216 .coefficient, true, some 1⟩])

def event287224 : Event := .survivorFold (1) 287223

def exact287225RawTerms : List Term := []

theorem exact287225RawTermsValid :
    exact287225RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287225 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50384⟩⟩) exact287225RawTerms (.finite 100) 287222 (.finite 100) (some (287223))

def event287226 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50385⟩⟩) 0 ⟨50384⟩ 287225

def event287227 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50385⟩⟩) (.identity (.predecessor 0 287226 .coefficient))

def event287228 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50385⟩⟩) (.finite 100)

def event287229 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50840⟩⟩) 0 ⟨50385⟩ 287228

def event287230 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50840⟩⟩) (.authority (.programFamilyFact))

def exact287231RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50840⟩⟩], []⟩, (1)⟩]

theorem exact287231RawTermsValid :
    exact287231RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287231 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50840⟩⟩) exact287231RawTerms (.finite 10) 287230 .exactZero (none)

def eventLeaf17936 : Array AnnotatedEvent := #[
  { event := event286976
    frameStart := 0 },
  { event := event286977
    frameStart := 0 },
  { event := event286978
    frameStart := 0 },
  { event := event286979
    frameStart := 0 },
  { event := event286980
    frameStart := 0 },
  { event := event286981
    frameStart := 0 },
  { event := event286982
    frameStart := 0 },
  { event := event286983
    frameStart := 0 },
  { event := event286984
    frameStart := 0 },
  { event := event286985
    frameStart := 0 },
  { event := event286986
    frameStart := 0 },
  { event := event286987
    frameStart := 0 },
  { event := event286988
    frameStart := 0 },
  { event := event286989
    frameStart := 0 },
  { event := event286990
    frameStart := 0 },
  { event := event286991
    frameStart := 0 }
]

def eventLeaf17937 : Array AnnotatedEvent := #[
  { event := event286992
    frameStart := 286992 },
  { event := event286993
    frameStart := 286992 },
  { event := event286994
    frameStart := 286992 },
  { event := event286995
    frameStart := 286992 },
  { event := event286996
    frameStart := 286992 },
  { event := event286997
    frameStart := 286992 },
  { event := event286998
    frameStart := 286992 },
  { event := event286999
    frameStart := 286992 },
  { event := event287000
    frameStart := 286992 },
  { event := event287001
    frameStart := 286992 },
  { event := event287002
    frameStart := 286992 },
  { event := event287003
    frameStart := 286992 },
  { event := event287004
    frameStart := 286992 },
  { event := event287005
    frameStart := 286992 },
  { event := event287006
    frameStart := 286992 },
  { event := event287007
    frameStart := 286992 }
]

def eventLeaf17938 : Array AnnotatedEvent := #[
  { event := event287008
    frameStart := 286992 },
  { event := event287009
    frameStart := 286992 },
  { event := event287010
    frameStart := 286992 },
  { event := event287011
    frameStart := 286992 },
  { event := event287012
    frameStart := 286992 },
  { event := event287013
    frameStart := 286992 },
  { event := event287014
    frameStart := 286992 },
  { event := event287015
    frameStart := 286992 },
  { event := event287016
    frameStart := 286992 },
  { event := event287017
    frameStart := 286992 },
  { event := event287018
    frameStart := 286992 },
  { event := event287019
    frameStart := 286992 },
  { event := event287020
    frameStart := 286992 },
  { event := event287021
    frameStart := 286992 },
  { event := event287022
    frameStart := 286992 },
  { event := event287023
    frameStart := 286992 }
]

def eventLeaf17939 : Array AnnotatedEvent := #[
  { event := event287024
    frameStart := 286992 },
  { event := event287025
    frameStart := 286992 },
  { event := event287026
    frameStart := 286992 },
  { event := event287027
    frameStart := 286992 },
  { event := event287028
    frameStart := 286992 },
  { event := event287029
    frameStart := 286992 },
  { event := event287030
    frameStart := 286992 },
  { event := event287031
    frameStart := 286992 },
  { event := event287032
    frameStart := 286992 },
  { event := event287033
    frameStart := 286992 },
  { event := event287034
    frameStart := 286992 },
  { event := event287035
    frameStart := 286992 },
  { event := event287036
    frameStart := 286992 },
  { event := event287037
    frameStart := 286992 },
  { event := event287038
    frameStart := 286992 },
  { event := event287039
    frameStart := 286992 }
]

def eventLeaf17940 : Array AnnotatedEvent := #[
  { event := event287040
    frameStart := 287040 },
  { event := event287041
    frameStart := 287040 },
  { event := event287042
    frameStart := 287040 },
  { event := event287043
    frameStart := 287040 },
  { event := event287044
    frameStart := 287040 },
  { event := event287045
    frameStart := 287040 },
  { event := event287046
    frameStart := 287040 },
  { event := event287047
    frameStart := 287040 },
  { event := event287048
    frameStart := 287040 },
  { event := event287049
    frameStart := 287040 },
  { event := event287050
    frameStart := 287040 },
  { event := event287051
    frameStart := 287040 },
  { event := event287052
    frameStart := 287040 },
  { event := event287053
    frameStart := 287040 },
  { event := event287054
    frameStart := 287040 },
  { event := event287055
    frameStart := 287040 }
]

def eventLeaf17941 : Array AnnotatedEvent := #[
  { event := event287056
    frameStart := 287040 },
  { event := event287057
    frameStart := 287040 },
  { event := event287058
    frameStart := 287040 },
  { event := event287059
    frameStart := 287040 },
  { event := event287060
    frameStart := 287040 },
  { event := event287061
    frameStart := 287040 },
  { event := event287062
    frameStart := 287040 },
  { event := event287063
    frameStart := 287040 },
  { event := event287064
    frameStart := 287040 },
  { event := event287065
    frameStart := 287040 },
  { event := event287066
    frameStart := 287040 },
  { event := event287067
    frameStart := 287040 },
  { event := event287068
    frameStart := 287040 },
  { event := event287069
    frameStart := 287040 },
  { event := event287070
    frameStart := 287040 },
  { event := event287071
    frameStart := 287040 }
]

def eventLeaf17942 : Array AnnotatedEvent := #[
  { event := event287072
    frameStart := 287040 },
  { event := event287073
    frameStart := 287040 },
  { event := event287074
    frameStart := 287040 },
  { event := event287075
    frameStart := 287040 },
  { event := event287076
    frameStart := 287040 },
  { event := event287077
    frameStart := 287040 },
  { event := event287078
    frameStart := 287040 },
  { event := event287079
    frameStart := 287040 },
  { event := event287080
    frameStart := 287040 },
  { event := event287081
    frameStart := 287040 },
  { event := event287082
    frameStart := 287040 },
  { event := event287083
    frameStart := 287040 },
  { event := event287084
    frameStart := 287040 },
  { event := event287085
    frameStart := 287040 },
  { event := event287086
    frameStart := 287040 },
  { event := event287087
    frameStart := 287040 }
]

def eventLeaf17943 : Array AnnotatedEvent := #[
  { event := event287088
    frameStart := 287040 },
  { event := event287089
    frameStart := 287040 },
  { event := event287090
    frameStart := 287040 },
  { event := event287091
    frameStart := 287040 },
  { event := event287092
    frameStart := 287040 },
  { event := event287093
    frameStart := 287040 },
  { event := event287094
    frameStart := 287040 },
  { event := event287095
    frameStart := 287040 },
  { event := event287096
    frameStart := 287040 },
  { event := event287097
    frameStart := 287040 },
  { event := event287098
    frameStart := 287040 },
  { event := event287099
    frameStart := 287040 },
  { event := event287100
    frameStart := 287040 },
  { event := event287101
    frameStart := 287040 },
  { event := event287102
    frameStart := 287040 },
  { event := event287103
    frameStart := 287040 }
]

def eventLeaf17944 : Array AnnotatedEvent := #[
  { event := event287104
    frameStart := 287040 },
  { event := event287105
    frameStart := 287040 },
  { event := event287106
    frameStart := 287040 },
  { event := event287107
    frameStart := 287040 },
  { event := event287108
    frameStart := 287040 },
  { event := event287109
    frameStart := 287040 },
  { event := event287110
    frameStart := 287040 },
  { event := event287111
    frameStart := 287040 },
  { event := event287112
    frameStart := 287040 },
  { event := event287113
    frameStart := 287040 },
  { event := event287114
    frameStart := 287040 },
  { event := event287115
    frameStart := 287040 },
  { event := event287116
    frameStart := 287040 },
  { event := event287117
    frameStart := 287040 },
  { event := event287118
    frameStart := 287040 },
  { event := event287119
    frameStart := 287040 }
]

def eventLeaf17945 : Array AnnotatedEvent := #[
  { event := event287120
    frameStart := 287040 },
  { event := event287121
    frameStart := 287040 },
  { event := event287122
    frameStart := 287040 },
  { event := event287123
    frameStart := 287040 },
  { event := event287124
    frameStart := 287040 },
  { event := event287125
    frameStart := 287040 },
  { event := event287126
    frameStart := 287040 },
  { event := event287127
    frameStart := 287040 },
  { event := event287128
    frameStart := 287040 },
  { event := event287129
    frameStart := 287040 },
  { event := event287130
    frameStart := 287040 },
  { event := event287131
    frameStart := 287040 },
  { event := event287132
    frameStart := 287040 },
  { event := event287133
    frameStart := 287040 },
  { event := event287134
    frameStart := 287040 },
  { event := event287135
    frameStart := 287040 }
]

def eventLeaf17946 : Array AnnotatedEvent := #[
  { event := event287136
    frameStart := 287040 },
  { event := event287137
    frameStart := 287040 },
  { event := event287138
    frameStart := 287040 },
  { event := event287139
    frameStart := 287040 },
  { event := event287140
    frameStart := 287040 },
  { event := event287141
    frameStart := 287040 },
  { event := event287142
    frameStart := 287040 },
  { event := event287143
    frameStart := 287040 },
  { event := event287144
    frameStart := 287040 },
  { event := event287145
    frameStart := 287040 },
  { event := event287146
    frameStart := 287040 },
  { event := event287147
    frameStart := 287040 },
  { event := event287148
    frameStart := 287040 },
  { event := event287149
    frameStart := 287040 },
  { event := event287150
    frameStart := 287040 },
  { event := event287151
    frameStart := 287040 }
]

def eventLeaf17947 : Array AnnotatedEvent := #[
  { event := event287152
    frameStart := 287040 },
  { event := event287153
    frameStart := 287040 },
  { event := event287154
    frameStart := 287040 },
  { event := event287155
    frameStart := 287040 },
  { event := event287156
    frameStart := 0 },
  { event := event287157
    frameStart := 0 },
  { event := event287158
    frameStart := 0 },
  { event := event287159
    frameStart := 0 },
  { event := event287160
    frameStart := 0 },
  { event := event287161
    frameStart := 0 },
  { event := event287162
    frameStart := 0 },
  { event := event287163
    frameStart := 0 },
  { event := event287164
    frameStart := 0 },
  { event := event287165
    frameStart := 0 },
  { event := event287166
    frameStart := 0 },
  { event := event287167
    frameStart := 0 }
]

def eventLeaf17948 : Array AnnotatedEvent := #[
  { event := event287168
    frameStart := 0 },
  { event := event287169
    frameStart := 0 },
  { event := event287170
    frameStart := 0 },
  { event := event287171
    frameStart := 0 },
  { event := event287172
    frameStart := 0 },
  { event := event287173
    frameStart := 0 },
  { event := event287174
    frameStart := 0 },
  { event := event287175
    frameStart := 0 },
  { event := event287176
    frameStart := 0 },
  { event := event287177
    frameStart := 0 },
  { event := event287178
    frameStart := 0 },
  { event := event287179
    frameStart := 0 },
  { event := event287180
    frameStart := 0 },
  { event := event287181
    frameStart := 0 },
  { event := event287182
    frameStart := 0 },
  { event := event287183
    frameStart := 0 }
]

def eventLeaf17949 : Array AnnotatedEvent := #[
  { event := event287184
    frameStart := 0 },
  { event := event287185
    frameStart := 0 },
  { event := event287186
    frameStart := 0 },
  { event := event287187
    frameStart := 0 },
  { event := event287188
    frameStart := 0 },
  { event := event287189
    frameStart := 0 },
  { event := event287190
    frameStart := 0 },
  { event := event287191
    frameStart := 0 },
  { event := event287192
    frameStart := 0 },
  { event := event287193
    frameStart := 287193 },
  { event := event287194
    frameStart := 287193 },
  { event := event287195
    frameStart := 287193 },
  { event := event287196
    frameStart := 287193 },
  { event := event287197
    frameStart := 287193 },
  { event := event287198
    frameStart := 287193 },
  { event := event287199
    frameStart := 287193 }
]

def eventLeaf17950 : Array AnnotatedEvent := #[
  { event := event287200
    frameStart := 287193 },
  { event := event287201
    frameStart := 287193 },
  { event := event287202
    frameStart := 287193 },
  { event := event287203
    frameStart := 287193 },
  { event := event287204
    frameStart := 287193 },
  { event := event287205
    frameStart := 287193 },
  { event := event287206
    frameStart := 287193 },
  { event := event287207
    frameStart := 287193 },
  { event := event287208
    frameStart := 287193 },
  { event := event287209
    frameStart := 287193 },
  { event := event287210
    frameStart := 287193 },
  { event := event287211
    frameStart := 287193 },
  { event := event287212
    frameStart := 287193 },
  { event := event287213
    frameStart := 287193 },
  { event := event287214
    frameStart := 287193 },
  { event := event287215
    frameStart := 287193 }
]

def eventLeaf17951 : Array AnnotatedEvent := #[
  { event := event287216
    frameStart := 287193 },
  { event := event287217
    frameStart := 287193 },
  { event := event287218
    frameStart := 287193 },
  { event := event287219
    frameStart := 287193 },
  { event := event287220
    frameStart := 287193 },
  { event := event287221
    frameStart := 287193 },
  { event := event287222
    frameStart := 287193 },
  { event := event287223
    frameStart := 287193 },
  { event := event287224
    frameStart := 287193 },
  { event := event287225
    frameStart := 287193 },
  { event := event287226
    frameStart := 287193 },
  { event := event287227
    frameStart := 287193 },
  { event := event287228
    frameStart := 287193 },
  { event := event287229
    frameStart := 287193 },
  { event := event287230
    frameStart := 287193 },
  { event := event287231
    frameStart := 287193 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1121
