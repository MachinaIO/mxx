import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events328

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact83968RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩, (1)⟩]

theorem exact83968RawTermsValid :
    exact83968RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83968 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19577⟩⟩) exact83968RawTerms .large 83966 .exactZero (none)

def event83969 : Event := .preFoldPolynomial 83968 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩, (1)⟩] .exactZero none

def exact83970RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩, (1)⟩]

def event83970 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨19577⟩⟩) 83969 exact83970RawTerms .large 83966 .exactZero (none)

def event83971 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨20843⟩⟩)

def event83972 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event83973 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event83974 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event83975 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event83976 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event83977 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event83978 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event83979 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event83980 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 83979

def event83981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 83977

def event83982 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 83980 .coefficient) (.value (.predecessor 1 83981 .coefficient)))

def event83983 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event83984 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 83983

def event83985 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 83975

def event83986 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 83984 .coefficient, .predecessor 1 83985 .coefficient])

def event83987 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event83988 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 83987

def event83989 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 83973

def event83990 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 83989 .coefficient))

def event83991 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event83992 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18418⟩⟩) 0 ⟨10325⟩ 83991

def event83993 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18418⟩⟩) (.authority (.programFamilyFact))

def exact83994RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18418⟩⟩], []⟩, (1)⟩]

theorem exact83994RawTermsValid :
    exact83994RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83994 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18418⟩⟩) exact83994RawTerms (.finite 3) 83993 .exactZero (none)

def event83995 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12771⟩⟩) 0 ⟨10325⟩ 83991

def event83996 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12771⟩⟩) (.authority (.programFamilyFact))

def exact83997RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩], []⟩, (1)⟩]

theorem exact83997RawTermsValid :
    exact83997RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83997 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12771⟩⟩) exact83997RawTerms (.finite 3) 83996 .exactZero (none)

def event83998 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18419⟩⟩) 0 ⟨12771⟩ 83997

def event83999 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18419⟩⟩) 1 ⟨18418⟩ 83994

def event84000 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18419⟩⟩) (.product (.predecessor 0 83998 .coefficient) (.predecessor 1 83999 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event84001 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18419⟩⟩, .operator (⟨83997, 0⟩, ⟨83994, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], []⟩, (1)⟩)

def exact84002RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12771⟩⟩, ⟨.program ⟨257⟩, ⟨18418⟩⟩], []⟩, (1)⟩]

theorem exact84002RawTermsValid :
    exact84002RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84002 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18419⟩⟩) exact84002RawTerms (.finite 9) 84000 .exactZero (none)

def event84003 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18420⟩⟩) 0 ⟨18419⟩ 84002

def event84004 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18420⟩⟩) (.identity (.predecessor 0 84003 .coefficient))

def event84005 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18420⟩⟩) (.finite 9)

def event84006 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18636⟩⟩) 0 ⟨18420⟩ 84005

def event84007 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18636⟩⟩) (.authority (.programFamilyFact))

def exact84008RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], []⟩, (1)⟩]

theorem exact84008RawTermsValid :
    exact84008RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84008 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18636⟩⟩) exact84008RawTerms (.finite 3) 84007 .exactZero (none)

def event84009 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18637⟩⟩) 0 ⟨18636⟩ 84008

def event84010 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18637⟩⟩) (.identity (.predecessor 0 84009 .coefficient))

def event84011 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18637⟩⟩) (.finite 3)

def event84012 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19913⟩⟩) 0 ⟨18637⟩ 84011

def event84013 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19913⟩⟩) (.authority (.programFamilyFact))

def event84014 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨19913⟩⟩) (.finite 3720)

def event84015 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event84016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19915⟩⟩) 0 ⟨7177⟩ 84015

def event84017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19915⟩⟩) 1 ⟨19913⟩ 84014

def event84018 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19915⟩⟩) (.authority (.operator))

def exact84019RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨19915⟩⟩]⟩, (1)⟩]

theorem exact84019RawTermsValid :
    exact84019RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84019 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19915⟩⟩) exact84019RawTerms .large 84018 .exactZero (none)

def event84020 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20838⟩⟩) 0 ⟨19915⟩ 84019

def event84021 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20838⟩⟩) (.authority (.operator))

def exact84022RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩, (1)⟩]

theorem exact84022RawTermsValid :
    exact84022RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84022 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20838⟩⟩) exact84022RawTerms (.finite 8192) 84021 .exactZero (none)

def event84023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event84024 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event84025 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20090⟩⟩) 0 ⟨18637⟩ 84011

def event84026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20090⟩⟩) 1 ⟨136⟩ 84024

def event84027 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20090⟩⟩) (.sum [.predecessor 0 84025 .coefficient, .predecessor 1 84026 .coefficient])

def event84028 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨20090⟩⟩) (.finite 3)

def event84029 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20091⟩⟩) 0 ⟨20090⟩ 84028

def event84030 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20091⟩⟩) (.identity (.predecessor 0 84029 .coefficient))

def exact84031RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], []⟩, (1)⟩]

theorem exact84031RawTermsValid :
    exact84031RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84031 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20091⟩⟩) exact84031RawTerms (.finite 3) 84030 .exactZero (none)

def event84032 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact84033RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact84033RawTermsValid :
    exact84033RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84033 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact84033RawTerms .large 84032 .exactZero (none)

def event84034 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20092⟩⟩) 0 ⟨6908⟩ 84033

def event84035 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20092⟩⟩) 1 ⟨20091⟩ 84031

def event84036 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20092⟩⟩) (.product (.predecessor 0 84034 .coefficient) (.predecessor 1 84035 .coefficient) (⟨false, false, none, none, none⟩))

def event84037 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20092⟩⟩, .operator (⟨84033, 0⟩, ⟨84031, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact84038RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact84038RawTermsValid :
    exact84038RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84038 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20092⟩⟩) exact84038RawTerms .large 84036 .exactZero (none)

def event84039 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7180⟩⟩) 0 ⟨7177⟩ 84015

def event84040 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7180⟩⟩) (.authority (.operator))

def exact84041RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩, (1)⟩]

theorem exact84041RawTermsValid :
    exact84041RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84041 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7180⟩⟩) exact84041RawTerms .large 84040 .exactZero (none)

def event84042 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20093⟩⟩) 0 ⟨7180⟩ 84041

def event84043 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20093⟩⟩) 1 ⟨20092⟩ 84038

def event84044 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20093⟩⟩) (.sum [.predecessor 0 84042 .coefficient, .predecessor 1 84043 .coefficient])

def exact84045RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84045RawTermsValid :
    exact84045RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84045 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20093⟩⟩) exact84045RawTerms .large 84044 .exactZero (none)

def event84046 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20839⟩⟩) 0 ⟨20093⟩ 84045

def event84047 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20839⟩⟩) 1 ⟨20838⟩ 84022

def event84048 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20839⟩⟩) (.product (.predecessor 0 84046 .coefficient) (.predecessor 1 84047 .coefficient) (⟨false, false, none, none, none⟩))

def event84049 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20839⟩⟩, .operator (⟨84045, 0⟩, ⟨84022, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩, (1)⟩)

def event84050 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20839⟩⟩, .operator (⟨84045, 1⟩, ⟨84022, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩, (-1)⟩)

def event84051 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨20839⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20838⟩⟩) ⟨19915⟩ 84019)

def event84052 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20839⟩⟩, .relation 84051 0, ⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨19915⟩⟩]⟩, (-1)⟩)

def exact84053RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨19915⟩⟩]⟩, (-1)⟩]

theorem exact84053RawTermsValid :
    exact84053RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84053 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20839⟩⟩) exact84053RawTerms .large 84048 .exactZero (none)

def event84054 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18980⟩⟩) 0 ⟨18637⟩ 84011

def event84055 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18980⟩⟩) (.authority (.programFamilyFact))

def exact84056RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], []⟩, (1)⟩]

theorem exact84056RawTermsValid :
    exact84056RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84056 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18980⟩⟩) exact84056RawTerms (.finite 48) 84055 .exactZero (none)

def event84057 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18982⟩⟩) 0 ⟨6908⟩ 84033

def event84058 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18982⟩⟩) 1 ⟨18980⟩ 84056

def event84059 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18982⟩⟩) (.product (.predecessor 0 84057 .coefficient) (.predecessor 1 84058 .coefficient) (⟨false, true, none, none, some 1⟩))

def event84060 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18982⟩⟩, .operator (⟨84033, 0⟩, ⟨84056, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact84061RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact84061RawTermsValid :
    exact84061RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84061 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18982⟩⟩) exact84061RawTerms .large 84059 .exactZero (none)

def event84062 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7200⟩⟩) 0 ⟨7177⟩ 84015

def event84063 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7200⟩⟩) (.authority (.operator))

def exact84064RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩]

theorem exact84064RawTermsValid :
    exact84064RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84064 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7200⟩⟩) exact84064RawTerms .large 84063 .exactZero (none)

def event84065 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18983⟩⟩) 0 ⟨7200⟩ 84064

def event84066 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18983⟩⟩) 1 ⟨18982⟩ 84061

def event84067 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18983⟩⟩) (.sum [.predecessor 0 84065 .coefficient, .predecessor 1 84066 .coefficient])

def exact84068RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84068RawTermsValid :
    exact84068RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84068 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18983⟩⟩) exact84068RawTerms .large 84067 .exactZero (none)

def event84069 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20843⟩⟩) 0 ⟨18983⟩ 84068

def event84070 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20843⟩⟩) 1 ⟨20839⟩ 84053

def event84071 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20843⟩⟩) (.sum [.predecessor 0 84069 .coefficient, .predecessor 1 84070 .coefficient])

def exact84072RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨19915⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84072RawTermsValid :
    exact84072RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84072 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20843⟩⟩) exact84072RawTerms .large 84071 .exactZero (none)

def event84073 : Event := .preFoldPolynomial 84072 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨19915⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact84074RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨19915⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event84074 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨20843⟩⟩) 84073 exact84074RawTerms .large 84071 .exactZero (none)

def event84075 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨18637⟩⟩) ⟨⟨79⟩, ⟨59⟩, ⟨135⟩⟩ ⟨83917, 84075⟩

def event84076 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨19579⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩) (1) 0 2 (.universal 84075 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19576⟩⟩]⟩) (none) 84074)

def event84077 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19579⟩⟩, .relation 84076 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩)

def event84078 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19579⟩⟩, .relation 84076 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩, (-1)⟩)

def event84079 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19579⟩⟩, .relation 84076 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨19915⟩⟩]⟩, (1)⟩)

def event84080 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19579⟩⟩, .relation 84076 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact84081RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨19915⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84081RawTermsValid :
    exact84081RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84081 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19579⟩⟩) exact84081RawTerms .large 83913 (.finite 202072841853861888) (some (83915))

def event84082 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20841⟩⟩) 0 ⟨19579⟩ 84081

def event84083 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20841⟩⟩) 1 ⟨20840⟩ 83903

def event84084 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20841⟩⟩) (.sum [.predecessor 0 84082 .coefficient, .predecessor 1 84083 .coefficient])

def event84085 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20841⟩⟩, .operator (⟨84081, 0⟩, ⟨83903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20838⟩⟩]⟩, (1)⟩)

def event84086 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20841⟩⟩, .operator (⟨84081, 2⟩, ⟨83903, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18636⟩⟩], [⟨.program ⟨257⟩, ⟨19915⟩⟩]⟩, (-1)⟩)

def event84087 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20841⟩⟩) (.sum [.result 84081 .summary, .result 83903 .summary])

def exact84088RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84088RawTermsValid :
    exact84088RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84088 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20841⟩⟩) exact84088RawTerms .large 84084 (.finite 32188905437706550578131070353408) (some (84087))

def event84089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17053⟩⟩) 0 ⟨15837⟩ 3494

def event84090 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17053⟩⟩) (.authority (.programFamilyFact))

def event84091 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17053⟩⟩) (.finite 3720)

def event84092 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17055⟩⟩) 0 ⟨7177⟩ 15500

def event84093 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17055⟩⟩) 1 ⟨17053⟩ 84091

def event84094 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17055⟩⟩) (.authority (.operator))

def exact84095RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17055⟩⟩]⟩, (1)⟩]

theorem exact84095RawTermsValid :
    exact84095RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84095 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17055⟩⟩) exact84095RawTerms .large 84094 .exactZero (none)

def event84096 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17929⟩⟩) 0 ⟨17055⟩ 84095

def event84097 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17929⟩⟩) (.authority (.operator))

def exact84098RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17929⟩⟩]⟩, (1)⟩]

theorem exact84098RawTermsValid :
    exact84098RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84098 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17929⟩⟩) exact84098RawTerms (.finite 8192) 84097 .exactZero (none)

def event84099 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16884⟩⟩) 0 ⟨15620⟩ 3488

def event84100 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16884⟩⟩) (.authority (.programFamilyFact))

def event84101 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨16884⟩⟩) (.finite 3720)

def event84102 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16885⟩⟩) 0 ⟨7177⟩ 15500

def event84103 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16885⟩⟩) 1 ⟨16884⟩ 84101

def event84104 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16885⟩⟩) (.authority (.operator))

def exact84105RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16885⟩⟩]⟩, (1)⟩]

theorem exact84105RawTermsValid :
    exact84105RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84105 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16885⟩⟩) exact84105RawTerms .large 84104 .exactZero (none)

def event84106 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17425⟩⟩) 0 ⟨16885⟩ 84105

def event84107 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17425⟩⟩) (.authority (.operator))

def exact84108RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17425⟩⟩]⟩, (1)⟩]

theorem exact84108RawTermsValid :
    exact84108RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84108 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17425⟩⟩) exact84108RawTerms (.finite 8192) 84107 .exactZero (none)

def event84109 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15621⟩⟩) 0 ⟨15618⟩ 3477

def event84110 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15621⟩⟩) 1 ⟨10328⟩ 75903

def event84111 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15621⟩⟩) (.tensor (.predecessor 0 84109 .coefficient) (.predecessor 1 84110 .coefficient) true false)

def event84112 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15621⟩⟩, .operator (⟨3477, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact84113RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact84113RawTermsValid :
    exact84113RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84113 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15621⟩⟩) exact84113RawTerms .large 84111 .exactZero (none)

def event84114 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10362⟩⟩) 0 ⟨10327⟩ 75773

def event84115 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10362⟩⟩) 1 ⟨7304⟩ 25597

def event84116 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10362⟩⟩) (.product (.predecessor 0 84114 .coefficient) (.predecessor 1 84115 .coefficient) (⟨false, false, none, none, none⟩))

def event84117 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10362⟩⟩, .operator (⟨75773, 0⟩, ⟨25597, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩)

def exact84118RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩]

theorem exact84118RawTermsValid :
    exact84118RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84118 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10362⟩⟩) exact84118RawTerms .large 84116 .exactZero (none)

def event84119 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15622⟩⟩) 0 ⟨10362⟩ 84118

def event84120 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15622⟩⟩) 1 ⟨15621⟩ 84113

def event84121 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15622⟩⟩) (.sum [.predecessor 0 84119 .coefficient, .predecessor 1 84120 .coefficient])

def exact84122RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84122RawTermsValid :
    exact84122RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84122 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15622⟩⟩) exact84122RawTerms .large 84121 .exactZero (none)

def event84123 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15623⟩⟩) 0 ⟨15622⟩ 84122

def event84124 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15623⟩⟩) 1 ⟨130⟩ 25589

def event84125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15623⟩⟩) (.sum [.predecessor 0 84123 .coefficient, .predecessor 1 84124 .coefficient])

def event84126 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15623⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨130⟩⟩]⟩) [⟨.result 25589 .coefficient, false, none⟩])

def event84127 : Event := .survivorFold (1) 84126

def exact84128RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84128RawTermsValid :
    exact84128RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84128 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15623⟩⟩) exact84128RawTerms .large 84125 (.finite 26) (some (84126))

def event84129 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15624⟩⟩) 0 ⟨15623⟩ 84128

def event84130 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15624⟩⟩) 1 ⟨12471⟩ 3480

def event84131 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15624⟩⟩) (.product (.predecessor 0 84129 .coefficient) (.predecessor 1 84130 .coefficient) (⟨false, true, none, none, some 1⟩))

def event84132 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15624⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩], []⟩) [⟨.result 3480 .coefficient, true, some 1⟩])

def event84133 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15624⟩⟩) (.product (.result 84128 .summary) (.transfer 84132) (⟨false, false, none, none, none⟩))

def event84134 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15624⟩⟩, .operator (⟨84128, 1⟩, ⟨3480, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event84135 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15624⟩⟩, .operator (⟨84128, 0⟩, ⟨3480, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩)

def exact84136RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84136RawTermsValid :
    exact84136RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84136 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15624⟩⟩) exact84136RawTerms .large 84131 (.finite 1703936) (some (84133))

def event84137 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12472⟩⟩) 0 ⟨12471⟩ 3480

def event84138 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12472⟩⟩) 1 ⟨10328⟩ 75903

def event84139 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12472⟩⟩) (.tensor (.predecessor 0 84137 .coefficient) (.predecessor 1 84138 .coefficient) true false)

def event84140 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨12472⟩⟩, .operator (⟨3480, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact84141RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact84141RawTermsValid :
    exact84141RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84141 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12472⟩⟩) exact84141RawTerms .large 84139 .exactZero (none)

def event84142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10361⟩⟩) 0 ⟨10327⟩ 75773

def event84143 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10361⟩⟩) 1 ⟨7303⟩ 25638

def event84144 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10361⟩⟩) (.product (.predecessor 0 84142 .coefficient) (.predecessor 1 84143 .coefficient) (⟨false, false, none, none, none⟩))

def event84145 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10361⟩⟩, .operator (⟨75773, 0⟩, ⟨25638, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩]⟩, (1)⟩)

def exact84146RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩]⟩, (1)⟩]

theorem exact84146RawTermsValid :
    exact84146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84146 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10361⟩⟩) exact84146RawTerms .large 84144 .exactZero (none)

def event84147 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12473⟩⟩) 0 ⟨10361⟩ 84146

def event84148 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12473⟩⟩) 1 ⟨12472⟩ 84141

def event84149 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12473⟩⟩) (.sum [.predecessor 0 84147 .coefficient, .predecessor 1 84148 .coefficient])

def exact84150RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84150RawTermsValid :
    exact84150RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84150 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12473⟩⟩) exact84150RawTerms .large 84149 .exactZero (none)

def event84151 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12474⟩⟩) 0 ⟨12473⟩ 84150

def event84152 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12474⟩⟩) 1 ⟨129⟩ 25630

def event84153 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12474⟩⟩) (.sum [.predecessor 0 84151 .coefficient, .predecessor 1 84152 .coefficient])

def event84154 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12474⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨129⟩⟩]⟩) [⟨.result 25630 .coefficient, false, none⟩])

def event84155 : Event := .survivorFold (1) 84154

def exact84156RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84156RawTermsValid :
    exact84156RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84156 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12474⟩⟩) exact84156RawTerms .large 84153 (.finite 26) (some (84154))

def event84157 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12475⟩⟩) 0 ⟨12474⟩ 84156

def event84158 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12475⟩⟩) 1 ⟨9569⟩ 25627

def event84159 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12475⟩⟩) (.product (.predecessor 0 84157 .coefficient) (.predecessor 1 84158 .coefficient) (⟨false, false, none, none, none⟩))

def event84160 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12475⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) [⟨.result 25623 .coefficient, false, none⟩])

def event84161 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12475⟩⟩) (.product (.result 84156 .summary) (.transfer 84160) (⟨false, false, none, none, none⟩))

def event84162 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨12475⟩⟩, .operator (⟨84156, 1⟩, ⟨25627, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (-1)⟩)

def event84163 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨12475⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9568⟩⟩) ⟨7304⟩ 25597)

def event84164 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨12475⟩⟩, .relation 84163 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (-1)⟩)

def event84165 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨12475⟩⟩, .operator (⟨84156, 0⟩, ⟨25627, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩)

def exact84166RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (-1)⟩]

theorem exact84166RawTermsValid :
    exact84166RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84166 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12475⟩⟩) exact84166RawTerms .large 84159 (.finite 279172874240) (some (84161))

def event84167 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15625⟩⟩) 0 ⟨12475⟩ 84166

def event84168 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15625⟩⟩) 1 ⟨15624⟩ 84136

def event84169 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15625⟩⟩) (.sum [.predecessor 0 84167 .coefficient, .predecessor 1 84168 .coefficient])

def event84170 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15625⟩⟩, .operator (⟨84166, 1⟩, ⟨84136, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩)

def event84171 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15625⟩⟩) (.sum [.result 84166 .summary, .result 84136 .summary])

def exact84172RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84172RawTermsValid :
    exact84172RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84172 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15625⟩⟩) exact84172RawTerms .large 84169 (.finite 279174578176) (some (84171))

def event84173 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17426⟩⟩) 0 ⟨15625⟩ 84172

def event84174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17426⟩⟩) 1 ⟨17425⟩ 84108

def event84175 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17426⟩⟩) (.product (.predecessor 0 84173 .coefficient) (.predecessor 1 84174 .coefficient) (⟨false, false, none, none, none⟩))

def event84176 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17426⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨17425⟩⟩]⟩) [⟨.result 84108 .coefficient, false, none⟩])

def event84177 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17426⟩⟩) (.product (.result 84172 .summary) (.transfer 84176) (⟨false, false, none, none, none⟩))

def event84178 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17426⟩⟩, .operator (⟨84172, 1⟩, ⟨84108, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17425⟩⟩]⟩, (-1)⟩)

def event84179 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17426⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17425⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17425⟩⟩) ⟨16885⟩ 84105)

def event84180 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17426⟩⟩, .relation 84179 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨16885⟩⟩]⟩, (-1)⟩)

def event84181 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17426⟩⟩, .operator (⟨84172, 0⟩, ⟨84108, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17425⟩⟩]⟩, (1)⟩)

def exact84182RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17425⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨16885⟩⟩]⟩, (-1)⟩]

theorem exact84182RawTermsValid :
    exact84182RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84182 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17426⟩⟩) exact84182RawTerms .large 84175 (.finite 2997614207851288330240) (some (84177))

def event84183 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16349⟩⟩) 0 ⟨15620⟩ 3488

def event84184 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16349⟩⟩) (.authority (.relationPreimageSource ⟨36⟩))

def exact84185RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16349⟩⟩]⟩, (1)⟩]

theorem exact84185RawTermsValid :
    exact84185RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84185 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16349⟩⟩) exact84185RawTerms (.finite 5647228698) 84184 .exactZero (none)

def event84186 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16351⟩⟩) 0 ⟨16349⟩ 84185

def event84187 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16351⟩⟩) 1 ⟨2370⟩ 4

def event84188 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16351⟩⟩) (.scale (.predecessor 0 84186 .coefficient) (.value (.predecessor 1 84187 .coefficient)))

def exact84189RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16349⟩⟩]⟩, (1)⟩]

theorem exact84189RawTermsValid :
    exact84189RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84189 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16351⟩⟩) exact84189RawTerms (.finite 5647228698) 84188 .exactZero (none)

def event84190 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16352⟩⟩) 0 ⟨10368⟩ 75995

def event84191 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16352⟩⟩) 1 ⟨16351⟩ 84189

def event84192 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16352⟩⟩) (.product (.predecessor 0 84190 .coefficient) (.predecessor 1 84191 .coefficient) (⟨false, false, none, none, none⟩))

def event84193 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16352⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨16349⟩⟩]⟩) [⟨.result 84185 .coefficient, false, none⟩])

def event84194 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16352⟩⟩) (.product (.result 75995 .summary) (.transfer 84193) (⟨false, false, none, none, none⟩))

def event84195 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16352⟩⟩, .operator (⟨75995, 0⟩, ⟨84189, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16349⟩⟩]⟩, (1)⟩)

def event84196 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨16350⟩⟩)

def event84197 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event84198 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event84199 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event84200 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event84201 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event84202 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event84203 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event84204 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event84205 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 84204

def event84206 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 84202

def event84207 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 84205 .coefficient) (.value (.predecessor 1 84206 .coefficient)))

def event84208 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event84209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 84208

def event84210 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 84200

def event84211 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 84209 .coefficient, .predecessor 1 84210 .coefficient])

def event84212 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event84213 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 84212

def event84214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 84198

def event84215 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 84214 .coefficient))

def event84216 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event84217 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15618⟩⟩) 0 ⟨10325⟩ 84216

def event84218 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15618⟩⟩) (.authority (.programFamilyFact))

def exact84219RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15618⟩⟩], []⟩, (1)⟩]

theorem exact84219RawTermsValid :
    exact84219RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84219 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15618⟩⟩) exact84219RawTerms (.finite 2) 84218 .exactZero (none)

def event84220 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12471⟩⟩) 0 ⟨10325⟩ 84216

def event84221 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12471⟩⟩) (.authority (.programFamilyFact))

def exact84222RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩], []⟩, (1)⟩]

theorem exact84222RawTermsValid :
    exact84222RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84222 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12471⟩⟩) exact84222RawTerms (.finite 2) 84221 .exactZero (none)

def event84223 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15619⟩⟩) 0 ⟨12471⟩ 84222

def eventLeaf5248 : Array AnnotatedEvent := #[
  { event := event83968
    frameStart := 83917 },
  { event := event83969
    frameStart := 83917 },
  { event := event83970
    frameStart := 83917 },
  { event := event83971
    frameStart := 83971 },
  { event := event83972
    frameStart := 83971 },
  { event := event83973
    frameStart := 83971 },
  { event := event83974
    frameStart := 83971 },
  { event := event83975
    frameStart := 83971 },
  { event := event83976
    frameStart := 83971 },
  { event := event83977
    frameStart := 83971 },
  { event := event83978
    frameStart := 83971 },
  { event := event83979
    frameStart := 83971 },
  { event := event83980
    frameStart := 83971 },
  { event := event83981
    frameStart := 83971 },
  { event := event83982
    frameStart := 83971 },
  { event := event83983
    frameStart := 83971 }
]

def eventLeaf5249 : Array AnnotatedEvent := #[
  { event := event83984
    frameStart := 83971 },
  { event := event83985
    frameStart := 83971 },
  { event := event83986
    frameStart := 83971 },
  { event := event83987
    frameStart := 83971 },
  { event := event83988
    frameStart := 83971 },
  { event := event83989
    frameStart := 83971 },
  { event := event83990
    frameStart := 83971 },
  { event := event83991
    frameStart := 83971 },
  { event := event83992
    frameStart := 83971 },
  { event := event83993
    frameStart := 83971 },
  { event := event83994
    frameStart := 83971 },
  { event := event83995
    frameStart := 83971 },
  { event := event83996
    frameStart := 83971 },
  { event := event83997
    frameStart := 83971 },
  { event := event83998
    frameStart := 83971 },
  { event := event83999
    frameStart := 83971 }
]

def eventLeaf5250 : Array AnnotatedEvent := #[
  { event := event84000
    frameStart := 83971 },
  { event := event84001
    frameStart := 83971 },
  { event := event84002
    frameStart := 83971 },
  { event := event84003
    frameStart := 83971 },
  { event := event84004
    frameStart := 83971 },
  { event := event84005
    frameStart := 83971 },
  { event := event84006
    frameStart := 83971 },
  { event := event84007
    frameStart := 83971 },
  { event := event84008
    frameStart := 83971 },
  { event := event84009
    frameStart := 83971 },
  { event := event84010
    frameStart := 83971 },
  { event := event84011
    frameStart := 83971 },
  { event := event84012
    frameStart := 83971 },
  { event := event84013
    frameStart := 83971 },
  { event := event84014
    frameStart := 83971 },
  { event := event84015
    frameStart := 83971 }
]

def eventLeaf5251 : Array AnnotatedEvent := #[
  { event := event84016
    frameStart := 83971 },
  { event := event84017
    frameStart := 83971 },
  { event := event84018
    frameStart := 83971 },
  { event := event84019
    frameStart := 83971 },
  { event := event84020
    frameStart := 83971 },
  { event := event84021
    frameStart := 83971 },
  { event := event84022
    frameStart := 83971 },
  { event := event84023
    frameStart := 83971 },
  { event := event84024
    frameStart := 83971 },
  { event := event84025
    frameStart := 83971 },
  { event := event84026
    frameStart := 83971 },
  { event := event84027
    frameStart := 83971 },
  { event := event84028
    frameStart := 83971 },
  { event := event84029
    frameStart := 83971 },
  { event := event84030
    frameStart := 83971 },
  { event := event84031
    frameStart := 83971 }
]

def eventLeaf5252 : Array AnnotatedEvent := #[
  { event := event84032
    frameStart := 83971 },
  { event := event84033
    frameStart := 83971 },
  { event := event84034
    frameStart := 83971 },
  { event := event84035
    frameStart := 83971 },
  { event := event84036
    frameStart := 83971 },
  { event := event84037
    frameStart := 83971 },
  { event := event84038
    frameStart := 83971 },
  { event := event84039
    frameStart := 83971 },
  { event := event84040
    frameStart := 83971 },
  { event := event84041
    frameStart := 83971 },
  { event := event84042
    frameStart := 83971 },
  { event := event84043
    frameStart := 83971 },
  { event := event84044
    frameStart := 83971 },
  { event := event84045
    frameStart := 83971 },
  { event := event84046
    frameStart := 83971 },
  { event := event84047
    frameStart := 83971 }
]

def eventLeaf5253 : Array AnnotatedEvent := #[
  { event := event84048
    frameStart := 83971 },
  { event := event84049
    frameStart := 83971 },
  { event := event84050
    frameStart := 83971 },
  { event := event84051
    frameStart := 83971 },
  { event := event84052
    frameStart := 83971 },
  { event := event84053
    frameStart := 83971 },
  { event := event84054
    frameStart := 83971 },
  { event := event84055
    frameStart := 83971 },
  { event := event84056
    frameStart := 83971 },
  { event := event84057
    frameStart := 83971 },
  { event := event84058
    frameStart := 83971 },
  { event := event84059
    frameStart := 83971 },
  { event := event84060
    frameStart := 83971 },
  { event := event84061
    frameStart := 83971 },
  { event := event84062
    frameStart := 83971 },
  { event := event84063
    frameStart := 83971 }
]

def eventLeaf5254 : Array AnnotatedEvent := #[
  { event := event84064
    frameStart := 83971 },
  { event := event84065
    frameStart := 83971 },
  { event := event84066
    frameStart := 83971 },
  { event := event84067
    frameStart := 83971 },
  { event := event84068
    frameStart := 83971 },
  { event := event84069
    frameStart := 83971 },
  { event := event84070
    frameStart := 83971 },
  { event := event84071
    frameStart := 83971 },
  { event := event84072
    frameStart := 83971 },
  { event := event84073
    frameStart := 83971 },
  { event := event84074
    frameStart := 83971 },
  { event := event84075
    frameStart := 0 },
  { event := event84076
    frameStart := 0 },
  { event := event84077
    frameStart := 0 },
  { event := event84078
    frameStart := 0 },
  { event := event84079
    frameStart := 0 }
]

def eventLeaf5255 : Array AnnotatedEvent := #[
  { event := event84080
    frameStart := 0 },
  { event := event84081
    frameStart := 0 },
  { event := event84082
    frameStart := 0 },
  { event := event84083
    frameStart := 0 },
  { event := event84084
    frameStart := 0 },
  { event := event84085
    frameStart := 0 },
  { event := event84086
    frameStart := 0 },
  { event := event84087
    frameStart := 0 },
  { event := event84088
    frameStart := 0 },
  { event := event84089
    frameStart := 0 },
  { event := event84090
    frameStart := 0 },
  { event := event84091
    frameStart := 0 },
  { event := event84092
    frameStart := 0 },
  { event := event84093
    frameStart := 0 },
  { event := event84094
    frameStart := 0 },
  { event := event84095
    frameStart := 0 }
]

def eventLeaf5256 : Array AnnotatedEvent := #[
  { event := event84096
    frameStart := 0 },
  { event := event84097
    frameStart := 0 },
  { event := event84098
    frameStart := 0 },
  { event := event84099
    frameStart := 0 },
  { event := event84100
    frameStart := 0 },
  { event := event84101
    frameStart := 0 },
  { event := event84102
    frameStart := 0 },
  { event := event84103
    frameStart := 0 },
  { event := event84104
    frameStart := 0 },
  { event := event84105
    frameStart := 0 },
  { event := event84106
    frameStart := 0 },
  { event := event84107
    frameStart := 0 },
  { event := event84108
    frameStart := 0 },
  { event := event84109
    frameStart := 0 },
  { event := event84110
    frameStart := 0 },
  { event := event84111
    frameStart := 0 }
]

def eventLeaf5257 : Array AnnotatedEvent := #[
  { event := event84112
    frameStart := 0 },
  { event := event84113
    frameStart := 0 },
  { event := event84114
    frameStart := 0 },
  { event := event84115
    frameStart := 0 },
  { event := event84116
    frameStart := 0 },
  { event := event84117
    frameStart := 0 },
  { event := event84118
    frameStart := 0 },
  { event := event84119
    frameStart := 0 },
  { event := event84120
    frameStart := 0 },
  { event := event84121
    frameStart := 0 },
  { event := event84122
    frameStart := 0 },
  { event := event84123
    frameStart := 0 },
  { event := event84124
    frameStart := 0 },
  { event := event84125
    frameStart := 0 },
  { event := event84126
    frameStart := 0 },
  { event := event84127
    frameStart := 0 }
]

def eventLeaf5258 : Array AnnotatedEvent := #[
  { event := event84128
    frameStart := 0 },
  { event := event84129
    frameStart := 0 },
  { event := event84130
    frameStart := 0 },
  { event := event84131
    frameStart := 0 },
  { event := event84132
    frameStart := 0 },
  { event := event84133
    frameStart := 0 },
  { event := event84134
    frameStart := 0 },
  { event := event84135
    frameStart := 0 },
  { event := event84136
    frameStart := 0 },
  { event := event84137
    frameStart := 0 },
  { event := event84138
    frameStart := 0 },
  { event := event84139
    frameStart := 0 },
  { event := event84140
    frameStart := 0 },
  { event := event84141
    frameStart := 0 },
  { event := event84142
    frameStart := 0 },
  { event := event84143
    frameStart := 0 }
]

def eventLeaf5259 : Array AnnotatedEvent := #[
  { event := event84144
    frameStart := 0 },
  { event := event84145
    frameStart := 0 },
  { event := event84146
    frameStart := 0 },
  { event := event84147
    frameStart := 0 },
  { event := event84148
    frameStart := 0 },
  { event := event84149
    frameStart := 0 },
  { event := event84150
    frameStart := 0 },
  { event := event84151
    frameStart := 0 },
  { event := event84152
    frameStart := 0 },
  { event := event84153
    frameStart := 0 },
  { event := event84154
    frameStart := 0 },
  { event := event84155
    frameStart := 0 },
  { event := event84156
    frameStart := 0 },
  { event := event84157
    frameStart := 0 },
  { event := event84158
    frameStart := 0 },
  { event := event84159
    frameStart := 0 }
]

def eventLeaf5260 : Array AnnotatedEvent := #[
  { event := event84160
    frameStart := 0 },
  { event := event84161
    frameStart := 0 },
  { event := event84162
    frameStart := 0 },
  { event := event84163
    frameStart := 0 },
  { event := event84164
    frameStart := 0 },
  { event := event84165
    frameStart := 0 },
  { event := event84166
    frameStart := 0 },
  { event := event84167
    frameStart := 0 },
  { event := event84168
    frameStart := 0 },
  { event := event84169
    frameStart := 0 },
  { event := event84170
    frameStart := 0 },
  { event := event84171
    frameStart := 0 },
  { event := event84172
    frameStart := 0 },
  { event := event84173
    frameStart := 0 },
  { event := event84174
    frameStart := 0 },
  { event := event84175
    frameStart := 0 }
]

def eventLeaf5261 : Array AnnotatedEvent := #[
  { event := event84176
    frameStart := 0 },
  { event := event84177
    frameStart := 0 },
  { event := event84178
    frameStart := 0 },
  { event := event84179
    frameStart := 0 },
  { event := event84180
    frameStart := 0 },
  { event := event84181
    frameStart := 0 },
  { event := event84182
    frameStart := 0 },
  { event := event84183
    frameStart := 0 },
  { event := event84184
    frameStart := 0 },
  { event := event84185
    frameStart := 0 },
  { event := event84186
    frameStart := 0 },
  { event := event84187
    frameStart := 0 },
  { event := event84188
    frameStart := 0 },
  { event := event84189
    frameStart := 0 },
  { event := event84190
    frameStart := 0 },
  { event := event84191
    frameStart := 0 }
]

def eventLeaf5262 : Array AnnotatedEvent := #[
  { event := event84192
    frameStart := 0 },
  { event := event84193
    frameStart := 0 },
  { event := event84194
    frameStart := 0 },
  { event := event84195
    frameStart := 0 },
  { event := event84196
    frameStart := 84196 },
  { event := event84197
    frameStart := 84196 },
  { event := event84198
    frameStart := 84196 },
  { event := event84199
    frameStart := 84196 },
  { event := event84200
    frameStart := 84196 },
  { event := event84201
    frameStart := 84196 },
  { event := event84202
    frameStart := 84196 },
  { event := event84203
    frameStart := 84196 },
  { event := event84204
    frameStart := 84196 },
  { event := event84205
    frameStart := 84196 },
  { event := event84206
    frameStart := 84196 },
  { event := event84207
    frameStart := 84196 }
]

def eventLeaf5263 : Array AnnotatedEvent := #[
  { event := event84208
    frameStart := 84196 },
  { event := event84209
    frameStart := 84196 },
  { event := event84210
    frameStart := 84196 },
  { event := event84211
    frameStart := 84196 },
  { event := event84212
    frameStart := 84196 },
  { event := event84213
    frameStart := 84196 },
  { event := event84214
    frameStart := 84196 },
  { event := event84215
    frameStart := 84196 },
  { event := event84216
    frameStart := 84196 },
  { event := event84217
    frameStart := 84196 },
  { event := event84218
    frameStart := 84196 },
  { event := event84219
    frameStart := 84196 },
  { event := event84220
    frameStart := 84196 },
  { event := event84221
    frameStart := 84196 },
  { event := event84222
    frameStart := 84196 },
  { event := event84223
    frameStart := 84196 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events328
