import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events305

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event78080 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37478⟩⟩) (.product (.predecessor 0 78078 .coefficient) (.predecessor 1 78079 .coefficient) (⟨false, true, none, none, some 1⟩))

def event78081 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37478⟩⟩, .operator (⟨78034, 0⟩, ⟨78077, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact78082RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact78082RawTermsValid :
    exact78082RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78082 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37478⟩⟩) exact78082RawTerms .large 78080 .exactZero (none)

def event78083 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7192⟩⟩) 0 ⟨7177⟩ 78016

def event78084 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7192⟩⟩) (.authority (.operator))

def exact78085RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩]

theorem exact78085RawTermsValid :
    exact78085RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78085 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7192⟩⟩) exact78085RawTerms .large 78084 .exactZero (none)

def event78086 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37479⟩⟩) 0 ⟨7192⟩ 78085

def event78087 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37479⟩⟩) 1 ⟨37478⟩ 78082

def event78088 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37479⟩⟩) (.sum [.predecessor 0 78086 .coefficient, .predecessor 1 78087 .coefficient])

def exact78089RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78089RawTermsValid :
    exact78089RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78089 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37479⟩⟩) exact78089RawTerms .large 78088 .exactZero (none)

def event78090 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39009⟩⟩) 0 ⟨37479⟩ 78089

def event78091 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39009⟩⟩) 1 ⟨39008⟩ 78074

def event78092 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39009⟩⟩) (.sum [.predecessor 0 78090 .coefficient, .predecessor 1 78091 .coefficient])

def exact78093RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨38465⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78093RawTermsValid :
    exact78093RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78093 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39009⟩⟩) exact78093RawTerms .large 78092 .exactZero (none)

def event78094 : Event := .preFoldPolynomial 78093 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨38465⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact78095RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨38465⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event78095 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨39009⟩⟩) 78094 exact78095RawTerms .large 78092 .exactZero (none)

def event78096 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨37260⟩⟩) ⟨⟨71⟩, ⟨50⟩, ⟨135⟩⟩ ⟨77930, 78096⟩

def event78097 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨37932⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩) (1) 0 2 (.universal 78096 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩) (none) 78095)

def event78098 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37932⟩⟩, .relation 78097 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩)

def event78099 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37932⟩⟩, .relation 78097 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩, (-1)⟩)

def event78100 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37932⟩⟩, .relation 78097 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨38465⟩⟩]⟩, (1)⟩)

def event78101 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37932⟩⟩, .relation 78097 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact78102RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨38465⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78102RawTermsValid :
    exact78102RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78102 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37932⟩⟩) exact78102RawTerms .large 77926 (.finite 202072841853861888) (some (77928))

def event78103 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39007⟩⟩) 0 ⟨37932⟩ 78102

def event78104 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39007⟩⟩) 1 ⟨39006⟩ 77916

def event78105 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39007⟩⟩) (.sum [.predecessor 0 78103 .coefficient, .predecessor 1 78104 .coefficient])

def event78106 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39007⟩⟩, .operator (⟨78102, 2⟩, ⟨77916, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨38465⟩⟩]⟩, (-1)⟩)

def event78107 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39007⟩⟩, .operator (⟨78102, 1⟩, ⟨77916, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩, (1)⟩)

def event78108 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39007⟩⟩) (.sum [.result 78102 .summary, .result 77916 .summary])

def exact78109RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78109RawTermsValid :
    exact78109RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78109 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39007⟩⟩) exact78109RawTerms .large 78105 (.finite 2998182198162866044928) (some (78108))

def event78110 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39461⟩⟩) 0 ⟨39007⟩ 78109

def event78111 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39461⟩⟩) 1 ⟨39459⟩ 77832

def event78112 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39461⟩⟩) (.product (.predecessor 0 78110 .coefficient) (.predecessor 1 78111 .coefficient) (⟨false, false, none, none, none⟩))

def event78113 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39461⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩) [⟨.result 77832 .coefficient, false, none⟩])

def event78114 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39461⟩⟩) (.product (.result 78109 .summary) (.transfer 78113) (⟨false, false, none, none, none⟩))

def event78115 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39461⟩⟩, .operator (⟨78109, 0⟩, ⟨77832, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩, (1)⟩)

def event78116 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39461⟩⟩, .operator (⟨78109, 1⟩, ⟨77832, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩, (-1)⟩)

def event78117 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39461⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39459⟩⟩) ⟨38635⟩ 77829)

def event78118 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39461⟩⟩, .relation 78117 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨38635⟩⟩]⟩, (-1)⟩)

def exact78119RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨38635⟩⟩]⟩, (-1)⟩]

theorem exact78119RawTermsValid :
    exact78119RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78119 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39461⟩⟩) exact78119RawTerms .large 78112 (.finite 32192736221397252361486566686720) (some (78114))

def event78120 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38296⟩⟩) 0 ⟨37477⟩ 3195

def event78121 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38296⟩⟩) (.authority (.relationPreimageSource ⟨85⟩))

def exact78122RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38296⟩⟩]⟩, (1)⟩]

theorem exact78122RawTermsValid :
    exact78122RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78122 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38296⟩⟩) exact78122RawTerms (.finite 5647228698) 78121 .exactZero (none)

def event78123 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38298⟩⟩) 0 ⟨38296⟩ 78122

def event78124 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38298⟩⟩) 1 ⟨2370⟩ 4

def event78125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38298⟩⟩) (.scale (.predecessor 0 78123 .coefficient) (.value (.predecessor 1 78124 .coefficient)))

def exact78126RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38296⟩⟩]⟩, (1)⟩]

theorem exact78126RawTermsValid :
    exact78126RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78126 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38298⟩⟩) exact78126RawTerms (.finite 5647228698) 78125 .exactZero (none)

def event78127 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38299⟩⟩) 0 ⟨10368⟩ 75995

def event78128 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38299⟩⟩) 1 ⟨38298⟩ 78126

def event78129 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38299⟩⟩) (.product (.predecessor 0 78127 .coefficient) (.predecessor 1 78128 .coefficient) (⟨false, false, none, none, none⟩))

def event78130 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38299⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨38296⟩⟩]⟩) [⟨.result 78122 .coefficient, false, none⟩])

def event78131 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38299⟩⟩) (.product (.result 75995 .summary) (.transfer 78130) (⟨false, false, none, none, none⟩))

def event78132 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38299⟩⟩, .operator (⟨75995, 0⟩, ⟨78126, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38296⟩⟩]⟩, (1)⟩)

def event78133 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨38297⟩⟩)

def event78134 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event78135 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event78136 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event78137 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event78138 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event78139 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event78140 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event78141 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event78142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 78141

def event78143 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 78139

def event78144 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 78142 .coefficient) (.value (.predecessor 1 78143 .coefficient)))

def event78145 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event78146 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 78145

def event78147 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 78137

def event78148 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 78146 .coefficient, .predecessor 1 78147 .coefficient])

def event78149 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event78150 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 78149

def event78151 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 78135

def event78152 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 78151 .coefficient))

def event78153 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event78154 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37258⟩⟩) 0 ⟨10325⟩ 78153

def event78155 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37258⟩⟩) (.authority (.programFamilyFact))

def exact78156RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37258⟩⟩], []⟩, (1)⟩]

theorem exact78156RawTermsValid :
    exact78156RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78156 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37258⟩⟩) exact78156RawTerms (.finite 42) 78155 .exactZero (none)

def event78157 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13971⟩⟩) 0 ⟨10325⟩ 78153

def event78158 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13971⟩⟩) (.authority (.programFamilyFact))

def exact78159RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩], []⟩, (1)⟩]

theorem exact78159RawTermsValid :
    exact78159RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78159 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13971⟩⟩) exact78159RawTerms (.finite 42) 78158 .exactZero (none)

def event78160 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37259⟩⟩) 0 ⟨13971⟩ 78159

def event78161 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37259⟩⟩) 1 ⟨37258⟩ 78156

def event78162 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37259⟩⟩) (.product (.predecessor 0 78160 .coefficient) (.predecessor 1 78161 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event78163 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37259⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], []⟩) [⟨.result 78159 .coefficient, true, some 1⟩, ⟨.result 78156 .coefficient, true, some 1⟩])

def event78164 : Event := .survivorFold (1) 78163

def exact78165RawTerms : List Term := []

theorem exact78165RawTermsValid :
    exact78165RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78165 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37259⟩⟩) exact78165RawTerms (.finite 1764) 78162 (.finite 1764) (some (78163))

def event78166 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37260⟩⟩) 0 ⟨37259⟩ 78165

def event78167 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37260⟩⟩) (.identity (.predecessor 0 78166 .coefficient))

def event78168 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37260⟩⟩) (.finite 1764)

def event78169 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37476⟩⟩) 0 ⟨37260⟩ 78168

def event78170 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37476⟩⟩) (.authority (.programFamilyFact))

def exact78171RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], []⟩, (1)⟩]

theorem exact78171RawTermsValid :
    exact78171RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78171 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37476⟩⟩) exact78171RawTerms (.finite 42) 78170 .exactZero (none)

def event78172 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37477⟩⟩) 0 ⟨37476⟩ 78171

def event78173 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37477⟩⟩) (.identity (.predecessor 0 78172 .coefficient))

def event78174 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37477⟩⟩) (.finite 42)

def event78175 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38296⟩⟩) 0 ⟨37477⟩ 78174

def event78176 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38296⟩⟩) (.authority (.relationPreimageSource ⟨85⟩))

def exact78177RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38296⟩⟩]⟩, (1)⟩]

theorem exact78177RawTermsValid :
    exact78177RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78177 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38296⟩⟩) exact78177RawTerms (.finite 5647228698) 78176 .exactZero (none)

def event78178 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact78179RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact78179RawTermsValid :
    exact78179RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78179 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact78179RawTerms .large 78178 .exactZero (none)

def event78180 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38297⟩⟩) 0 ⟨35⟩ 78179

def event78181 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38297⟩⟩) 1 ⟨38296⟩ 78177

def event78182 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38297⟩⟩) (.product (.predecessor 0 78180 .coefficient) (.predecessor 1 78181 .coefficient) (⟨false, false, none, none, none⟩))

def event78183 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38297⟩⟩, .operator (⟨78179, 0⟩, ⟨78177, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38296⟩⟩]⟩, (1)⟩)

def exact78184RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38296⟩⟩]⟩, (1)⟩]

theorem exact78184RawTermsValid :
    exact78184RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78184 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38297⟩⟩) exact78184RawTerms .large 78182 .exactZero (none)

def event78185 : Event := .preFoldPolynomial 78184 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38296⟩⟩]⟩, (1)⟩] .exactZero none

def exact78186RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38296⟩⟩]⟩, (1)⟩]

def event78186 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨38297⟩⟩) 78185 exact78186RawTerms .large 78182 .exactZero (none)

def event78187 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨39463⟩⟩)

def event78188 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event78189 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event78190 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event78191 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event78192 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event78193 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event78194 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event78195 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event78196 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 78195

def event78197 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 78193

def event78198 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 78196 .coefficient) (.value (.predecessor 1 78197 .coefficient)))

def event78199 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event78200 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 78199

def event78201 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 78191

def event78202 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 78200 .coefficient, .predecessor 1 78201 .coefficient])

def event78203 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event78204 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 78203

def event78205 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 78189

def event78206 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 78205 .coefficient))

def event78207 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event78208 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37258⟩⟩) 0 ⟨10325⟩ 78207

def event78209 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37258⟩⟩) (.authority (.programFamilyFact))

def exact78210RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37258⟩⟩], []⟩, (1)⟩]

theorem exact78210RawTermsValid :
    exact78210RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78210 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37258⟩⟩) exact78210RawTerms (.finite 42) 78209 .exactZero (none)

def event78211 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13971⟩⟩) 0 ⟨10325⟩ 78207

def event78212 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13971⟩⟩) (.authority (.programFamilyFact))

def exact78213RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩], []⟩, (1)⟩]

theorem exact78213RawTermsValid :
    exact78213RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78213 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13971⟩⟩) exact78213RawTerms (.finite 42) 78212 .exactZero (none)

def event78214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37259⟩⟩) 0 ⟨13971⟩ 78213

def event78215 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37259⟩⟩) 1 ⟨37258⟩ 78210

def event78216 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37259⟩⟩) (.product (.predecessor 0 78214 .coefficient) (.predecessor 1 78215 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event78217 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37259⟩⟩, .operator (⟨78213, 0⟩, ⟨78210, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], []⟩, (1)⟩)

def exact78218RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], []⟩, (1)⟩]

theorem exact78218RawTermsValid :
    exact78218RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78218 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37259⟩⟩) exact78218RawTerms (.finite 1764) 78216 .exactZero (none)

def event78219 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37260⟩⟩) 0 ⟨37259⟩ 78218

def event78220 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37260⟩⟩) (.identity (.predecessor 0 78219 .coefficient))

def event78221 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37260⟩⟩) (.finite 1764)

def event78222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37476⟩⟩) 0 ⟨37260⟩ 78221

def event78223 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37476⟩⟩) (.authority (.programFamilyFact))

def exact78224RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], []⟩, (1)⟩]

theorem exact78224RawTermsValid :
    exact78224RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78224 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37476⟩⟩) exact78224RawTerms (.finite 42) 78223 .exactZero (none)

def event78225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37477⟩⟩) 0 ⟨37476⟩ 78224

def event78226 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37477⟩⟩) (.identity (.predecessor 0 78225 .coefficient))

def event78227 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37477⟩⟩) (.finite 42)

def event78228 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38633⟩⟩) 0 ⟨37477⟩ 78227

def event78229 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38633⟩⟩) (.authority (.programFamilyFact))

def event78230 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38633⟩⟩) (.finite 3720)

def event78231 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event78232 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38635⟩⟩) 0 ⟨7177⟩ 78231

def event78233 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38635⟩⟩) 1 ⟨38633⟩ 78230

def event78234 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38635⟩⟩) (.authority (.operator))

def exact78235RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38635⟩⟩]⟩, (1)⟩]

theorem exact78235RawTermsValid :
    exact78235RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78235 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38635⟩⟩) exact78235RawTerms .large 78234 .exactZero (none)

def event78236 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39459⟩⟩) 0 ⟨38635⟩ 78235

def event78237 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39459⟩⟩) (.authority (.operator))

def exact78238RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩, (1)⟩]

theorem exact78238RawTermsValid :
    exact78238RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78238 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39459⟩⟩) exact78238RawTerms (.finite 8192) 78237 .exactZero (none)

def event78239 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event78240 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event78241 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38810⟩⟩) 0 ⟨37477⟩ 78227

def event78242 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38810⟩⟩) 1 ⟨136⟩ 78240

def event78243 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38810⟩⟩) (.sum [.predecessor 0 78241 .coefficient, .predecessor 1 78242 .coefficient])

def event78244 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38810⟩⟩) (.finite 42)

def event78245 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38811⟩⟩) 0 ⟨38810⟩ 78244

def event78246 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38811⟩⟩) (.identity (.predecessor 0 78245 .coefficient))

def exact78247RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], []⟩, (1)⟩]

theorem exact78247RawTermsValid :
    exact78247RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78247 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38811⟩⟩) exact78247RawTerms (.finite 42) 78246 .exactZero (none)

def event78248 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact78249RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact78249RawTermsValid :
    exact78249RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78249 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact78249RawTerms .large 78248 .exactZero (none)

def event78250 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38812⟩⟩) 0 ⟨6908⟩ 78249

def event78251 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38812⟩⟩) 1 ⟨38811⟩ 78247

def event78252 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38812⟩⟩) (.product (.predecessor 0 78250 .coefficient) (.predecessor 1 78251 .coefficient) (⟨false, false, none, none, none⟩))

def event78253 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38812⟩⟩, .operator (⟨78249, 0⟩, ⟨78247, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact78254RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact78254RawTermsValid :
    exact78254RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78254 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38812⟩⟩) exact78254RawTerms .large 78252 .exactZero (none)

def event78255 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7192⟩⟩) 0 ⟨7177⟩ 78231

def event78256 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7192⟩⟩) (.authority (.operator))

def exact78257RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩]

theorem exact78257RawTermsValid :
    exact78257RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78257 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7192⟩⟩) exact78257RawTerms .large 78256 .exactZero (none)

def event78258 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38813⟩⟩) 0 ⟨7192⟩ 78257

def event78259 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38813⟩⟩) 1 ⟨38812⟩ 78254

def event78260 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38813⟩⟩) (.sum [.predecessor 0 78258 .coefficient, .predecessor 1 78259 .coefficient])

def exact78261RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78261RawTermsValid :
    exact78261RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78261 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38813⟩⟩) exact78261RawTerms .large 78260 .exactZero (none)

def event78262 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39460⟩⟩) 0 ⟨38813⟩ 78261

def event78263 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39460⟩⟩) 1 ⟨39459⟩ 78238

def event78264 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39460⟩⟩) (.product (.predecessor 0 78262 .coefficient) (.predecessor 1 78263 .coefficient) (⟨false, false, none, none, none⟩))

def event78265 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39460⟩⟩, .operator (⟨78261, 0⟩, ⟨78238, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩, (1)⟩)

def event78266 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39460⟩⟩, .operator (⟨78261, 1⟩, ⟨78238, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩, (-1)⟩)

def event78267 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39460⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39459⟩⟩) ⟨38635⟩ 78235)

def event78268 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39460⟩⟩, .relation 78267 0, ⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨38635⟩⟩]⟩, (-1)⟩)

def exact78269RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨38635⟩⟩]⟩, (-1)⟩]

theorem exact78269RawTermsValid :
    exact78269RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78269 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39460⟩⟩) exact78269RawTerms .large 78264 .exactZero (none)

def event78270 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37721⟩⟩) 0 ⟨37477⟩ 78227

def event78271 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37721⟩⟩) (.authority (.programFamilyFact))

def exact78272RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], []⟩, (1)⟩]

theorem exact78272RawTermsValid :
    exact78272RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78272 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37721⟩⟩) exact78272RawTerms (.finite 63) 78271 .exactZero (none)

def event78273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37722⟩⟩) 0 ⟨6908⟩ 78249

def event78274 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37722⟩⟩) 1 ⟨37721⟩ 78272

def event78275 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37722⟩⟩) (.product (.predecessor 0 78273 .coefficient) (.predecessor 1 78274 .coefficient) (⟨false, true, none, none, some 1⟩))

def event78276 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37722⟩⟩, .operator (⟨78249, 0⟩, ⟨78272, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact78277RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact78277RawTermsValid :
    exact78277RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78277 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37722⟩⟩) exact78277RawTerms .large 78275 .exactZero (none)

def event78278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7224⟩⟩) 0 ⟨7177⟩ 78231

def event78279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7224⟩⟩) (.authority (.operator))

def exact78280RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩]

theorem exact78280RawTermsValid :
    exact78280RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78280 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7224⟩⟩) exact78280RawTerms .large 78279 .exactZero (none)

def event78281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37723⟩⟩) 0 ⟨7224⟩ 78280

def event78282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37723⟩⟩) 1 ⟨37722⟩ 78277

def event78283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37723⟩⟩) (.sum [.predecessor 0 78281 .coefficient, .predecessor 1 78282 .coefficient])

def exact78284RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78284RawTermsValid :
    exact78284RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78284 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37723⟩⟩) exact78284RawTerms .large 78283 .exactZero (none)

def event78285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39463⟩⟩) 0 ⟨37723⟩ 78284

def event78286 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39463⟩⟩) 1 ⟨39460⟩ 78269

def event78287 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39463⟩⟩) (.sum [.predecessor 0 78285 .coefficient, .predecessor 1 78286 .coefficient])

def exact78288RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨38635⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78288RawTermsValid :
    exact78288RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78288 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39463⟩⟩) exact78288RawTerms .large 78287 .exactZero (none)

def event78289 : Event := .preFoldPolynomial 78288 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨38635⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact78290RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨38635⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event78290 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨39463⟩⟩) 78289 exact78290RawTerms .large 78287 .exactZero (none)

def event78291 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨37477⟩⟩) ⟨⟨103⟩, ⟨85⟩, ⟨135⟩⟩ ⟨78133, 78291⟩

def event78292 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨38299⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38296⟩⟩]⟩) (1) 0 2 (.universal 78291 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38296⟩⟩]⟩) (none) 78290)

def event78293 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38299⟩⟩, .relation 78292 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩)

def event78294 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38299⟩⟩, .relation 78292 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩, (-1)⟩)

def event78295 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38299⟩⟩, .relation 78292 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨38635⟩⟩]⟩, (1)⟩)

def event78296 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38299⟩⟩, .relation 78292 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact78297RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨38635⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78297RawTermsValid :
    exact78297RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78297 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38299⟩⟩) exact78297RawTerms .large 78129 (.finite 202072841853861888) (some (78131))

def event78298 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39462⟩⟩) 0 ⟨38299⟩ 78297

def event78299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39462⟩⟩) 1 ⟨39461⟩ 78119

def event78300 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39462⟩⟩) (.sum [.predecessor 0 78298 .coefficient, .predecessor 1 78299 .coefficient])

def event78301 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39462⟩⟩, .operator (⟨78297, 0⟩, ⟨78119, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩, (1)⟩)

def event78302 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39462⟩⟩, .operator (⟨78297, 2⟩, ⟨78119, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨38635⟩⟩]⟩, (-1)⟩)

def event78303 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39462⟩⟩) (.sum [.result 78297 .summary, .result 78119 .summary])

def exact78304RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78304RawTermsValid :
    exact78304RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78304 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39462⟩⟩) exact78304RawTerms .large 78300 (.finite 32192736221397454434328420548608) (some (78303))

def event78305 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35953⟩⟩) 0 ⟨34797⟩ 3218

def event78306 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35953⟩⟩) (.authority (.programFamilyFact))

def event78307 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨35953⟩⟩) (.finite 3720)

def event78308 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35955⟩⟩) 0 ⟨7177⟩ 15500

def event78309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35955⟩⟩) 1 ⟨35953⟩ 78307

def event78310 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35955⟩⟩) (.authority (.operator))

def exact78311RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35955⟩⟩]⟩, (1)⟩]

theorem exact78311RawTermsValid :
    exact78311RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78311 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35955⟩⟩) exact78311RawTerms .large 78310 .exactZero (none)

def event78312 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36779⟩⟩) 0 ⟨35955⟩ 78311

def event78313 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36779⟩⟩) (.authority (.operator))

def exact78314RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨36779⟩⟩]⟩, (1)⟩]

theorem exact78314RawTermsValid :
    exact78314RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78314 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36779⟩⟩) exact78314RawTerms (.finite 8192) 78313 .exactZero (none)

def event78315 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35784⟩⟩) 0 ⟨34580⟩ 3212

def event78316 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35784⟩⟩) (.authority (.programFamilyFact))

def event78317 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨35784⟩⟩) (.finite 3720)

def event78318 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35785⟩⟩) 0 ⟨7177⟩ 15500

def event78319 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35785⟩⟩) 1 ⟨35784⟩ 78317

def event78320 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35785⟩⟩) (.authority (.operator))

def exact78321RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35785⟩⟩]⟩, (1)⟩]

theorem exact78321RawTermsValid :
    exact78321RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78321 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35785⟩⟩) exact78321RawTerms .large 78320 .exactZero (none)

def event78322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36325⟩⟩) 0 ⟨35785⟩ 78321

def event78323 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36325⟩⟩) (.authority (.operator))

def exact78324RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨36325⟩⟩]⟩, (1)⟩]

theorem exact78324RawTermsValid :
    exact78324RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78324 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36325⟩⟩) exact78324RawTerms (.finite 8192) 78323 .exactZero (none)

def event78325 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34581⟩⟩) 0 ⟨34578⟩ 3201

def event78326 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34581⟩⟩) 1 ⟨10328⟩ 75903

def event78327 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34581⟩⟩) (.tensor (.predecessor 0 78325 .coefficient) (.predecessor 1 78326 .coefficient) true false)

def event78328 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34581⟩⟩, .operator (⟨3201, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact78329RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact78329RawTermsValid :
    exact78329RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78329 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34581⟩⟩) exact78329RawTerms .large 78327 .exactZero (none)

def event78330 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10338⟩⟩) 0 ⟨10327⟩ 75773

def event78331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10338⟩⟩) 1 ⟨7280⟩ 19585

def event78332 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10338⟩⟩) (.product (.predecessor 0 78330 .coefficient) (.predecessor 1 78331 .coefficient) (⟨false, false, none, none, none⟩))

def event78333 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10338⟩⟩, .operator (⟨75773, 0⟩, ⟨19585, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (1)⟩)

def exact78334RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (1)⟩]

theorem exact78334RawTermsValid :
    exact78334RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78334 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10338⟩⟩) exact78334RawTerms .large 78332 .exactZero (none)

def event78335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34582⟩⟩) 0 ⟨10338⟩ 78334

def eventLeaf4880 : Array AnnotatedEvent := #[
  { event := event78080
    frameStart := 77978 },
  { event := event78081
    frameStart := 77978 },
  { event := event78082
    frameStart := 77978 },
  { event := event78083
    frameStart := 77978 },
  { event := event78084
    frameStart := 77978 },
  { event := event78085
    frameStart := 77978 },
  { event := event78086
    frameStart := 77978 },
  { event := event78087
    frameStart := 77978 },
  { event := event78088
    frameStart := 77978 },
  { event := event78089
    frameStart := 77978 },
  { event := event78090
    frameStart := 77978 },
  { event := event78091
    frameStart := 77978 },
  { event := event78092
    frameStart := 77978 },
  { event := event78093
    frameStart := 77978 },
  { event := event78094
    frameStart := 77978 },
  { event := event78095
    frameStart := 77978 }
]

def eventLeaf4881 : Array AnnotatedEvent := #[
  { event := event78096
    frameStart := 0 },
  { event := event78097
    frameStart := 0 },
  { event := event78098
    frameStart := 0 },
  { event := event78099
    frameStart := 0 },
  { event := event78100
    frameStart := 0 },
  { event := event78101
    frameStart := 0 },
  { event := event78102
    frameStart := 0 },
  { event := event78103
    frameStart := 0 },
  { event := event78104
    frameStart := 0 },
  { event := event78105
    frameStart := 0 },
  { event := event78106
    frameStart := 0 },
  { event := event78107
    frameStart := 0 },
  { event := event78108
    frameStart := 0 },
  { event := event78109
    frameStart := 0 },
  { event := event78110
    frameStart := 0 },
  { event := event78111
    frameStart := 0 }
]

def eventLeaf4882 : Array AnnotatedEvent := #[
  { event := event78112
    frameStart := 0 },
  { event := event78113
    frameStart := 0 },
  { event := event78114
    frameStart := 0 },
  { event := event78115
    frameStart := 0 },
  { event := event78116
    frameStart := 0 },
  { event := event78117
    frameStart := 0 },
  { event := event78118
    frameStart := 0 },
  { event := event78119
    frameStart := 0 },
  { event := event78120
    frameStart := 0 },
  { event := event78121
    frameStart := 0 },
  { event := event78122
    frameStart := 0 },
  { event := event78123
    frameStart := 0 },
  { event := event78124
    frameStart := 0 },
  { event := event78125
    frameStart := 0 },
  { event := event78126
    frameStart := 0 },
  { event := event78127
    frameStart := 0 }
]

def eventLeaf4883 : Array AnnotatedEvent := #[
  { event := event78128
    frameStart := 0 },
  { event := event78129
    frameStart := 0 },
  { event := event78130
    frameStart := 0 },
  { event := event78131
    frameStart := 0 },
  { event := event78132
    frameStart := 0 },
  { event := event78133
    frameStart := 78133 },
  { event := event78134
    frameStart := 78133 },
  { event := event78135
    frameStart := 78133 },
  { event := event78136
    frameStart := 78133 },
  { event := event78137
    frameStart := 78133 },
  { event := event78138
    frameStart := 78133 },
  { event := event78139
    frameStart := 78133 },
  { event := event78140
    frameStart := 78133 },
  { event := event78141
    frameStart := 78133 },
  { event := event78142
    frameStart := 78133 },
  { event := event78143
    frameStart := 78133 }
]

def eventLeaf4884 : Array AnnotatedEvent := #[
  { event := event78144
    frameStart := 78133 },
  { event := event78145
    frameStart := 78133 },
  { event := event78146
    frameStart := 78133 },
  { event := event78147
    frameStart := 78133 },
  { event := event78148
    frameStart := 78133 },
  { event := event78149
    frameStart := 78133 },
  { event := event78150
    frameStart := 78133 },
  { event := event78151
    frameStart := 78133 },
  { event := event78152
    frameStart := 78133 },
  { event := event78153
    frameStart := 78133 },
  { event := event78154
    frameStart := 78133 },
  { event := event78155
    frameStart := 78133 },
  { event := event78156
    frameStart := 78133 },
  { event := event78157
    frameStart := 78133 },
  { event := event78158
    frameStart := 78133 },
  { event := event78159
    frameStart := 78133 }
]

def eventLeaf4885 : Array AnnotatedEvent := #[
  { event := event78160
    frameStart := 78133 },
  { event := event78161
    frameStart := 78133 },
  { event := event78162
    frameStart := 78133 },
  { event := event78163
    frameStart := 78133 },
  { event := event78164
    frameStart := 78133 },
  { event := event78165
    frameStart := 78133 },
  { event := event78166
    frameStart := 78133 },
  { event := event78167
    frameStart := 78133 },
  { event := event78168
    frameStart := 78133 },
  { event := event78169
    frameStart := 78133 },
  { event := event78170
    frameStart := 78133 },
  { event := event78171
    frameStart := 78133 },
  { event := event78172
    frameStart := 78133 },
  { event := event78173
    frameStart := 78133 },
  { event := event78174
    frameStart := 78133 },
  { event := event78175
    frameStart := 78133 }
]

def eventLeaf4886 : Array AnnotatedEvent := #[
  { event := event78176
    frameStart := 78133 },
  { event := event78177
    frameStart := 78133 },
  { event := event78178
    frameStart := 78133 },
  { event := event78179
    frameStart := 78133 },
  { event := event78180
    frameStart := 78133 },
  { event := event78181
    frameStart := 78133 },
  { event := event78182
    frameStart := 78133 },
  { event := event78183
    frameStart := 78133 },
  { event := event78184
    frameStart := 78133 },
  { event := event78185
    frameStart := 78133 },
  { event := event78186
    frameStart := 78133 },
  { event := event78187
    frameStart := 78187 },
  { event := event78188
    frameStart := 78187 },
  { event := event78189
    frameStart := 78187 },
  { event := event78190
    frameStart := 78187 },
  { event := event78191
    frameStart := 78187 }
]

def eventLeaf4887 : Array AnnotatedEvent := #[
  { event := event78192
    frameStart := 78187 },
  { event := event78193
    frameStart := 78187 },
  { event := event78194
    frameStart := 78187 },
  { event := event78195
    frameStart := 78187 },
  { event := event78196
    frameStart := 78187 },
  { event := event78197
    frameStart := 78187 },
  { event := event78198
    frameStart := 78187 },
  { event := event78199
    frameStart := 78187 },
  { event := event78200
    frameStart := 78187 },
  { event := event78201
    frameStart := 78187 },
  { event := event78202
    frameStart := 78187 },
  { event := event78203
    frameStart := 78187 },
  { event := event78204
    frameStart := 78187 },
  { event := event78205
    frameStart := 78187 },
  { event := event78206
    frameStart := 78187 },
  { event := event78207
    frameStart := 78187 }
]

def eventLeaf4888 : Array AnnotatedEvent := #[
  { event := event78208
    frameStart := 78187 },
  { event := event78209
    frameStart := 78187 },
  { event := event78210
    frameStart := 78187 },
  { event := event78211
    frameStart := 78187 },
  { event := event78212
    frameStart := 78187 },
  { event := event78213
    frameStart := 78187 },
  { event := event78214
    frameStart := 78187 },
  { event := event78215
    frameStart := 78187 },
  { event := event78216
    frameStart := 78187 },
  { event := event78217
    frameStart := 78187 },
  { event := event78218
    frameStart := 78187 },
  { event := event78219
    frameStart := 78187 },
  { event := event78220
    frameStart := 78187 },
  { event := event78221
    frameStart := 78187 },
  { event := event78222
    frameStart := 78187 },
  { event := event78223
    frameStart := 78187 }
]

def eventLeaf4889 : Array AnnotatedEvent := #[
  { event := event78224
    frameStart := 78187 },
  { event := event78225
    frameStart := 78187 },
  { event := event78226
    frameStart := 78187 },
  { event := event78227
    frameStart := 78187 },
  { event := event78228
    frameStart := 78187 },
  { event := event78229
    frameStart := 78187 },
  { event := event78230
    frameStart := 78187 },
  { event := event78231
    frameStart := 78187 },
  { event := event78232
    frameStart := 78187 },
  { event := event78233
    frameStart := 78187 },
  { event := event78234
    frameStart := 78187 },
  { event := event78235
    frameStart := 78187 },
  { event := event78236
    frameStart := 78187 },
  { event := event78237
    frameStart := 78187 },
  { event := event78238
    frameStart := 78187 },
  { event := event78239
    frameStart := 78187 }
]

def eventLeaf4890 : Array AnnotatedEvent := #[
  { event := event78240
    frameStart := 78187 },
  { event := event78241
    frameStart := 78187 },
  { event := event78242
    frameStart := 78187 },
  { event := event78243
    frameStart := 78187 },
  { event := event78244
    frameStart := 78187 },
  { event := event78245
    frameStart := 78187 },
  { event := event78246
    frameStart := 78187 },
  { event := event78247
    frameStart := 78187 },
  { event := event78248
    frameStart := 78187 },
  { event := event78249
    frameStart := 78187 },
  { event := event78250
    frameStart := 78187 },
  { event := event78251
    frameStart := 78187 },
  { event := event78252
    frameStart := 78187 },
  { event := event78253
    frameStart := 78187 },
  { event := event78254
    frameStart := 78187 },
  { event := event78255
    frameStart := 78187 }
]

def eventLeaf4891 : Array AnnotatedEvent := #[
  { event := event78256
    frameStart := 78187 },
  { event := event78257
    frameStart := 78187 },
  { event := event78258
    frameStart := 78187 },
  { event := event78259
    frameStart := 78187 },
  { event := event78260
    frameStart := 78187 },
  { event := event78261
    frameStart := 78187 },
  { event := event78262
    frameStart := 78187 },
  { event := event78263
    frameStart := 78187 },
  { event := event78264
    frameStart := 78187 },
  { event := event78265
    frameStart := 78187 },
  { event := event78266
    frameStart := 78187 },
  { event := event78267
    frameStart := 78187 },
  { event := event78268
    frameStart := 78187 },
  { event := event78269
    frameStart := 78187 },
  { event := event78270
    frameStart := 78187 },
  { event := event78271
    frameStart := 78187 }
]

def eventLeaf4892 : Array AnnotatedEvent := #[
  { event := event78272
    frameStart := 78187 },
  { event := event78273
    frameStart := 78187 },
  { event := event78274
    frameStart := 78187 },
  { event := event78275
    frameStart := 78187 },
  { event := event78276
    frameStart := 78187 },
  { event := event78277
    frameStart := 78187 },
  { event := event78278
    frameStart := 78187 },
  { event := event78279
    frameStart := 78187 },
  { event := event78280
    frameStart := 78187 },
  { event := event78281
    frameStart := 78187 },
  { event := event78282
    frameStart := 78187 },
  { event := event78283
    frameStart := 78187 },
  { event := event78284
    frameStart := 78187 },
  { event := event78285
    frameStart := 78187 },
  { event := event78286
    frameStart := 78187 },
  { event := event78287
    frameStart := 78187 }
]

def eventLeaf4893 : Array AnnotatedEvent := #[
  { event := event78288
    frameStart := 78187 },
  { event := event78289
    frameStart := 78187 },
  { event := event78290
    frameStart := 78187 },
  { event := event78291
    frameStart := 0 },
  { event := event78292
    frameStart := 0 },
  { event := event78293
    frameStart := 0 },
  { event := event78294
    frameStart := 0 },
  { event := event78295
    frameStart := 0 },
  { event := event78296
    frameStart := 0 },
  { event := event78297
    frameStart := 0 },
  { event := event78298
    frameStart := 0 },
  { event := event78299
    frameStart := 0 },
  { event := event78300
    frameStart := 0 },
  { event := event78301
    frameStart := 0 },
  { event := event78302
    frameStart := 0 },
  { event := event78303
    frameStart := 0 }
]

def eventLeaf4894 : Array AnnotatedEvent := #[
  { event := event78304
    frameStart := 0 },
  { event := event78305
    frameStart := 0 },
  { event := event78306
    frameStart := 0 },
  { event := event78307
    frameStart := 0 },
  { event := event78308
    frameStart := 0 },
  { event := event78309
    frameStart := 0 },
  { event := event78310
    frameStart := 0 },
  { event := event78311
    frameStart := 0 },
  { event := event78312
    frameStart := 0 },
  { event := event78313
    frameStart := 0 },
  { event := event78314
    frameStart := 0 },
  { event := event78315
    frameStart := 0 },
  { event := event78316
    frameStart := 0 },
  { event := event78317
    frameStart := 0 },
  { event := event78318
    frameStart := 0 },
  { event := event78319
    frameStart := 0 }
]

def eventLeaf4895 : Array AnnotatedEvent := #[
  { event := event78320
    frameStart := 0 },
  { event := event78321
    frameStart := 0 },
  { event := event78322
    frameStart := 0 },
  { event := event78323
    frameStart := 0 },
  { event := event78324
    frameStart := 0 },
  { event := event78325
    frameStart := 0 },
  { event := event78326
    frameStart := 0 },
  { event := event78327
    frameStart := 0 },
  { event := event78328
    frameStart := 0 },
  { event := event78329
    frameStart := 0 },
  { event := event78330
    frameStart := 0 },
  { event := event78331
    frameStart := 0 },
  { event := event78332
    frameStart := 0 },
  { event := event78333
    frameStart := 0 },
  { event := event78334
    frameStart := 0 },
  { event := event78335
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events305
