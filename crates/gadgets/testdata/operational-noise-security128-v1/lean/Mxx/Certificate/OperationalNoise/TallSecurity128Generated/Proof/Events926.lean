import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events926

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact237056RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237056RawTermsValid :
    exact237056RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237056 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49639⟩⟩) exact237056RawTerms .large 237052 (.finite 2998346861024241778688) (some (237055))

def event237057 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49981⟩⟩) 0 ⟨49639⟩ 237056

def event237058 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49981⟩⟩) 1 ⟨49979⟩ 236763

def event237059 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49981⟩⟩) (.product (.predecessor 0 237057 .coefficient) (.predecessor 1 237058 .coefficient) (⟨false, false, none, none, none⟩))

def event237060 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49981⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩) [⟨.result 236763 .coefficient, false, none⟩])

def event237061 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49981⟩⟩) (.product (.result 237056 .summary) (.transfer 237060) (⟨false, false, none, none, none⟩))

def event237062 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49981⟩⟩, .operator (⟨237056, 0⟩, ⟨236763, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩, (1)⟩)

def event237063 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49981⟩⟩, .operator (⟨237056, 1⟩, ⟨236763, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩, (-1)⟩)

def event237064 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49981⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49979⟩⟩) ⟨49283⟩ 236760)

def event237065 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49981⟩⟩, .relation 237064 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨49283⟩⟩]⟩, (-1)⟩)

def exact237066RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨49283⟩⟩]⟩, (-1)⟩]

theorem exact237066RawTermsValid :
    exact237066RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237066 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49981⟩⟩) exact237066RawTerms .large 237059 (.finite 32194504275408438756654574469120) (some (237061))

def event237067 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48856⟩⟩) 0 ⟨48133⟩ 11331

def event237068 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48856⟩⟩) (.authority (.relationPreimageSource ⟨94⟩))

def exact237069RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48856⟩⟩]⟩, (1)⟩]

theorem exact237069RawTermsValid :
    exact237069RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237069 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48856⟩⟩) exact237069RawTerms (.finite 5647228698) 237068 .exactZero (none)

def event237070 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48858⟩⟩) 0 ⟨48856⟩ 237069

def event237071 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48858⟩⟩) 1 ⟨2370⟩ 4

def event237072 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48858⟩⟩) (.scale (.predecessor 0 237070 .coefficient) (.value (.predecessor 1 237071 .coefficient)))

def exact237073RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48856⟩⟩]⟩, (1)⟩]

theorem exact237073RawTermsValid :
    exact237073RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237073 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48858⟩⟩) exact237073RawTerms (.finite 5647228698) 237072 .exactZero (none)

def event237074 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48859⟩⟩) 0 ⟨5563⟩ 236870

def event237075 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48859⟩⟩) 1 ⟨48858⟩ 237073

def event237076 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48859⟩⟩) (.product (.predecessor 0 237074 .coefficient) (.predecessor 1 237075 .coefficient) (⟨false, false, none, none, none⟩))

def event237077 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48859⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨48856⟩⟩]⟩) [⟨.result 237069 .coefficient, false, none⟩])

def event237078 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48859⟩⟩) (.product (.result 236870 .summary) (.transfer 237077) (⟨false, false, none, none, none⟩))

def event237079 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48859⟩⟩, .operator (⟨236870, 0⟩, ⟨237073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48856⟩⟩]⟩, (1)⟩)

def event237080 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨48857⟩⟩)

def event237081 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event237082 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event237083 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.authority (.operator))

def event237084 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.finite 4)

def event237085 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event237086 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event237087 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event237088 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event237089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 237088

def event237090 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 237086

def event237091 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 237089 .coefficient) (.value (.predecessor 1 237090 .coefficient)))

def event237092 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event237093 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 0 ⟨392⟩ 237092

def event237094 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 1 ⟨4740⟩ 237084

def event237095 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.sum [.predecessor 0 237093 .coefficient, .predecessor 1 237094 .coefficient])

def event237096 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.finite 655344)

def event237097 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 0 ⟨4742⟩ 237096

def event237098 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 1 ⟨5426⟩ 237082

def event237099 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.identity (.predecessor 1 237098 .coefficient))

def event237100 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.finite 655360)

def event237101 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47786⟩⟩) 0 ⟨5559⟩ 237100

def event237102 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47786⟩⟩) (.authority (.programFamilyFact))

def exact237103RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47786⟩⟩], []⟩, (1)⟩]

theorem exact237103RawTermsValid :
    exact237103RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237103 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47786⟩⟩) exact237103RawTerms (.finite 60) 237102 .exactZero (none)

def event237104 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15051⟩⟩) 0 ⟨5559⟩ 237100

def event237105 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15051⟩⟩) (.authority (.programFamilyFact))

def exact237106RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15051⟩⟩], []⟩, (1)⟩]

theorem exact237106RawTermsValid :
    exact237106RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237106 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15051⟩⟩) exact237106RawTerms (.finite 60) 237105 .exactZero (none)

def event237107 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47787⟩⟩) 0 ⟨15051⟩ 237106

def event237108 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47787⟩⟩) 1 ⟨47786⟩ 237103

def event237109 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47787⟩⟩) (.product (.predecessor 0 237107 .coefficient) (.predecessor 1 237108 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event237110 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47787⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨15051⟩⟩, ⟨.program ⟨257⟩, ⟨47786⟩⟩], []⟩) [⟨.result 237106 .coefficient, true, some 1⟩, ⟨.result 237103 .coefficient, true, some 1⟩])

def event237111 : Event := .survivorFold (1) 237110

def exact237112RawTerms : List Term := []

theorem exact237112RawTermsValid :
    exact237112RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237112 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47787⟩⟩) exact237112RawTerms (.finite 3600) 237109 (.finite 3600) (some (237110))

def event237113 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47788⟩⟩) 0 ⟨47787⟩ 237112

def event237114 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47788⟩⟩) (.identity (.predecessor 0 237113 .coefficient))

def event237115 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47788⟩⟩) (.finite 3600)

def event237116 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48132⟩⟩) 0 ⟨47788⟩ 237115

def event237117 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48132⟩⟩) (.authority (.programFamilyFact))

def exact237118RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48132⟩⟩], []⟩, (1)⟩]

theorem exact237118RawTermsValid :
    exact237118RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237118 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48132⟩⟩) exact237118RawTerms (.finite 60) 237117 .exactZero (none)

def event237119 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48133⟩⟩) 0 ⟨48132⟩ 237118

def event237120 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48133⟩⟩) (.identity (.predecessor 0 237119 .coefficient))

def event237121 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48133⟩⟩) (.finite 60)

def event237122 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48856⟩⟩) 0 ⟨48133⟩ 237121

def event237123 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48856⟩⟩) (.authority (.relationPreimageSource ⟨94⟩))

def exact237124RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48856⟩⟩]⟩, (1)⟩]

theorem exact237124RawTermsValid :
    exact237124RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237124 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48856⟩⟩) exact237124RawTerms (.finite 5647228698) 237123 .exactZero (none)

def event237125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact237126RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact237126RawTermsValid :
    exact237126RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237126 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact237126RawTerms .large 237125 .exactZero (none)

def event237127 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48857⟩⟩) 0 ⟨35⟩ 237126

def event237128 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48857⟩⟩) 1 ⟨48856⟩ 237124

def event237129 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48857⟩⟩) (.product (.predecessor 0 237127 .coefficient) (.predecessor 1 237128 .coefficient) (⟨false, false, none, none, none⟩))

def event237130 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48857⟩⟩, .operator (⟨237126, 0⟩, ⟨237124, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48856⟩⟩]⟩, (1)⟩)

def exact237131RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48856⟩⟩]⟩, (1)⟩]

theorem exact237131RawTermsValid :
    exact237131RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237131 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48857⟩⟩) exact237131RawTerms .large 237129 .exactZero (none)

def event237132 : Event := .preFoldPolynomial 237131 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48856⟩⟩]⟩, (1)⟩] .exactZero none

def exact237133RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48856⟩⟩]⟩, (1)⟩]

def event237133 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨48857⟩⟩) 237132 exact237133RawTerms .large 237129 .exactZero (none)

def event237134 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨49983⟩⟩)

def event237135 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event237136 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event237137 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.authority (.operator))

def event237138 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.finite 4)

def event237139 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event237140 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event237141 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event237142 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event237143 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 237142

def event237144 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 237140

def event237145 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 237143 .coefficient) (.value (.predecessor 1 237144 .coefficient)))

def event237146 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event237147 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 0 ⟨392⟩ 237146

def event237148 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 1 ⟨4740⟩ 237138

def event237149 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.sum [.predecessor 0 237147 .coefficient, .predecessor 1 237148 .coefficient])

def event237150 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.finite 655344)

def event237151 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 0 ⟨4742⟩ 237150

def event237152 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 1 ⟨5426⟩ 237136

def event237153 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.identity (.predecessor 1 237152 .coefficient))

def event237154 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.finite 655360)

def event237155 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47786⟩⟩) 0 ⟨5559⟩ 237154

def event237156 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47786⟩⟩) (.authority (.programFamilyFact))

def exact237157RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47786⟩⟩], []⟩, (1)⟩]

theorem exact237157RawTermsValid :
    exact237157RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237157 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47786⟩⟩) exact237157RawTerms (.finite 60) 237156 .exactZero (none)

def event237158 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15051⟩⟩) 0 ⟨5559⟩ 237154

def event237159 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15051⟩⟩) (.authority (.programFamilyFact))

def exact237160RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15051⟩⟩], []⟩, (1)⟩]

theorem exact237160RawTermsValid :
    exact237160RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237160 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15051⟩⟩) exact237160RawTerms (.finite 60) 237159 .exactZero (none)

def event237161 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47787⟩⟩) 0 ⟨15051⟩ 237160

def event237162 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47787⟩⟩) 1 ⟨47786⟩ 237157

def event237163 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47787⟩⟩) (.product (.predecessor 0 237161 .coefficient) (.predecessor 1 237162 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event237164 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47787⟩⟩, .operator (⟨237160, 0⟩, ⟨237157, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15051⟩⟩, ⟨.program ⟨257⟩, ⟨47786⟩⟩], []⟩, (1)⟩)

def exact237165RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15051⟩⟩, ⟨.program ⟨257⟩, ⟨47786⟩⟩], []⟩, (1)⟩]

theorem exact237165RawTermsValid :
    exact237165RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237165 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47787⟩⟩) exact237165RawTerms (.finite 3600) 237163 .exactZero (none)

def event237166 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47788⟩⟩) 0 ⟨47787⟩ 237165

def event237167 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47788⟩⟩) (.identity (.predecessor 0 237166 .coefficient))

def event237168 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47788⟩⟩) (.finite 3600)

def event237169 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48132⟩⟩) 0 ⟨47788⟩ 237168

def event237170 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48132⟩⟩) (.authority (.programFamilyFact))

def exact237171RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48132⟩⟩], []⟩, (1)⟩]

theorem exact237171RawTermsValid :
    exact237171RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237171 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48132⟩⟩) exact237171RawTerms (.finite 60) 237170 .exactZero (none)

def event237172 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48133⟩⟩) 0 ⟨48132⟩ 237171

def event237173 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48133⟩⟩) (.identity (.predecessor 0 237172 .coefficient))

def event237174 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48133⟩⟩) (.finite 60)

def event237175 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49281⟩⟩) 0 ⟨48133⟩ 237174

def event237176 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49281⟩⟩) (.authority (.programFamilyFact))

def event237177 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49281⟩⟩) (.finite 3720)

def event237178 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event237179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49283⟩⟩) 0 ⟨7177⟩ 237178

def event237180 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49283⟩⟩) 1 ⟨49281⟩ 237177

def event237181 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49283⟩⟩) (.authority (.operator))

def exact237182RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49283⟩⟩]⟩, (1)⟩]

theorem exact237182RawTermsValid :
    exact237182RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237182 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49283⟩⟩) exact237182RawTerms .large 237181 .exactZero (none)

def event237183 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49979⟩⟩) 0 ⟨49283⟩ 237182

def event237184 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49979⟩⟩) (.authority (.operator))

def exact237185RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩, (1)⟩]

theorem exact237185RawTermsValid :
    exact237185RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237185 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49979⟩⟩) exact237185RawTerms (.finite 8192) 237184 .exactZero (none)

def event237186 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event237187 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event237188 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49498⟩⟩) 0 ⟨48133⟩ 237174

def event237189 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49498⟩⟩) 1 ⟨136⟩ 237187

def event237190 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49498⟩⟩) (.sum [.predecessor 0 237188 .coefficient, .predecessor 1 237189 .coefficient])

def event237191 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49498⟩⟩) (.finite 60)

def event237192 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49499⟩⟩) 0 ⟨49498⟩ 237191

def event237193 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49499⟩⟩) (.identity (.predecessor 0 237192 .coefficient))

def exact237194RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48132⟩⟩], []⟩, (1)⟩]

theorem exact237194RawTermsValid :
    exact237194RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237194 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49499⟩⟩) exact237194RawTerms (.finite 60) 237193 .exactZero (none)

def event237195 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact237196RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact237196RawTermsValid :
    exact237196RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237196 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact237196RawTerms .large 237195 .exactZero (none)

def event237197 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49500⟩⟩) 0 ⟨6908⟩ 237196

def event237198 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49500⟩⟩) 1 ⟨49499⟩ 237194

def event237199 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49500⟩⟩) (.product (.predecessor 0 237197 .coefficient) (.predecessor 1 237198 .coefficient) (⟨false, false, none, none, none⟩))

def event237200 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49500⟩⟩, .operator (⟨237196, 0⟩, ⟨237194, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact237201RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact237201RawTermsValid :
    exact237201RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237201 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49500⟩⟩) exact237201RawTerms .large 237199 .exactZero (none)

def event237202 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7196⟩⟩) 0 ⟨7177⟩ 237178

def event237203 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7196⟩⟩) (.authority (.operator))

def exact237204RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩]

theorem exact237204RawTermsValid :
    exact237204RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237204 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7196⟩⟩) exact237204RawTerms .large 237203 .exactZero (none)

def event237205 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49501⟩⟩) 0 ⟨7196⟩ 237204

def event237206 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49501⟩⟩) 1 ⟨49500⟩ 237201

def event237207 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49501⟩⟩) (.sum [.predecessor 0 237205 .coefficient, .predecessor 1 237206 .coefficient])

def exact237208RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237208RawTermsValid :
    exact237208RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237208 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49501⟩⟩) exact237208RawTerms .large 237207 .exactZero (none)

def event237209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49980⟩⟩) 0 ⟨49501⟩ 237208

def event237210 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49980⟩⟩) 1 ⟨49979⟩ 237185

def event237211 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49980⟩⟩) (.product (.predecessor 0 237209 .coefficient) (.predecessor 1 237210 .coefficient) (⟨false, false, none, none, none⟩))

def event237212 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49980⟩⟩, .operator (⟨237208, 0⟩, ⟨237185, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩, (1)⟩)

def event237213 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49980⟩⟩, .operator (⟨237208, 1⟩, ⟨237185, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩, (-1)⟩)

def event237214 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49980⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49979⟩⟩) ⟨49283⟩ 237182)

def event237215 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49980⟩⟩, .relation 237214 0, ⟨[⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨49283⟩⟩]⟩, (-1)⟩)

def exact237216RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨49283⟩⟩]⟩, (-1)⟩]

theorem exact237216RawTermsValid :
    exact237216RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237216 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49980⟩⟩) exact237216RawTerms .large 237211 .exactZero (none)

def event237217 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48337⟩⟩) 0 ⟨48133⟩ 237174

def event237218 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48337⟩⟩) (.authority (.programFamilyFact))

def exact237219RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48337⟩⟩], []⟩, (1)⟩]

theorem exact237219RawTermsValid :
    exact237219RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237219 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48337⟩⟩) exact237219RawTerms (.finite 63) 237218 .exactZero (none)

def event237220 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48338⟩⟩) 0 ⟨6908⟩ 237196

def event237221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48338⟩⟩) 1 ⟨48337⟩ 237219

def event237222 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48338⟩⟩) (.product (.predecessor 0 237220 .coefficient) (.predecessor 1 237221 .coefficient) (⟨false, true, none, none, some 1⟩))

def event237223 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48338⟩⟩, .operator (⟨237196, 0⟩, ⟨237219, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48337⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact237224RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48337⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact237224RawTermsValid :
    exact237224RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237224 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48338⟩⟩) exact237224RawTerms .large 237222 .exactZero (none)

def event237225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7232⟩⟩) 0 ⟨7177⟩ 237178

def event237226 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7232⟩⟩) (.authority (.operator))

def exact237227RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact237227RawTermsValid :
    exact237227RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237227 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7232⟩⟩) exact237227RawTerms .large 237226 .exactZero (none)

def event237228 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48339⟩⟩) 0 ⟨7232⟩ 237227

def event237229 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48339⟩⟩) 1 ⟨48338⟩ 237224

def event237230 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48339⟩⟩) (.sum [.predecessor 0 237228 .coefficient, .predecessor 1 237229 .coefficient])

def exact237231RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48337⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237231RawTermsValid :
    exact237231RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237231 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48339⟩⟩) exact237231RawTerms .large 237230 .exactZero (none)

def event237232 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49983⟩⟩) 0 ⟨48339⟩ 237231

def event237233 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49983⟩⟩) 1 ⟨49980⟩ 237216

def event237234 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49983⟩⟩) (.sum [.predecessor 0 237232 .coefficient, .predecessor 1 237233 .coefficient])

def exact237235RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨49283⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48337⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237235RawTermsValid :
    exact237235RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237235 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49983⟩⟩) exact237235RawTerms .large 237234 .exactZero (none)

def event237236 : Event := .preFoldPolynomial 237235 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨49283⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48337⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact237237RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨49283⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48337⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event237237 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨49983⟩⟩) 237236 exact237237RawTerms .large 237234 .exactZero (none)

def event237238 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨48133⟩⟩) ⟨⟨111⟩, ⟨94⟩, ⟨135⟩⟩ ⟨237080, 237238⟩

def event237239 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨48859⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48856⟩⟩]⟩) (1) 0 2 (.universal 237238 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48856⟩⟩]⟩) (none) 237237)

def event237240 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48859⟩⟩, .relation 237239 1, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩)

def event237241 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48859⟩⟩, .relation 237239 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩, (-1)⟩)

def event237242 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48859⟩⟩, .relation 237239 2, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨49283⟩⟩]⟩, (1)⟩)

def event237243 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48859⟩⟩, .relation 237239 3, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48337⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact237244RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨49283⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48337⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237244RawTermsValid :
    exact237244RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237244 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48859⟩⟩) exact237244RawTerms .large 237076 (.finite 202072841853861888) (some (237078))

def event237245 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49982⟩⟩) 0 ⟨48859⟩ 237244

def event237246 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49982⟩⟩) 1 ⟨49981⟩ 237066

def event237247 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49982⟩⟩) (.sum [.predecessor 0 237245 .coefficient, .predecessor 1 237246 .coefficient])

def event237248 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49982⟩⟩, .operator (⟨237244, 0⟩, ⟨237066, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩, (1)⟩)

def event237249 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49982⟩⟩, .operator (⟨237244, 2⟩, ⟨237066, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨49283⟩⟩]⟩, (-1)⟩)

def event237250 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49982⟩⟩) (.sum [.result 237244 .summary, .result 237066 .summary])

def exact237251RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48337⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237251RawTermsValid :
    exact237251RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237251 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49982⟩⟩) exact237251RawTerms .large 237247 (.finite 32194504275408640829496428331008) (some (237250))

def event237252 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46601⟩⟩) 0 ⟨45453⟩ 11354

def event237253 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46601⟩⟩) (.authority (.programFamilyFact))

def event237254 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46601⟩⟩) (.finite 3720)

def event237255 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46603⟩⟩) 0 ⟨7177⟩ 15500

def event237256 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46603⟩⟩) 1 ⟨46601⟩ 237254

def event237257 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46603⟩⟩) (.authority (.operator))

def exact237258RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46603⟩⟩]⟩, (1)⟩]

theorem exact237258RawTermsValid :
    exact237258RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237258 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46603⟩⟩) exact237258RawTerms .large 237257 .exactZero (none)

def event237259 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47299⟩⟩) 0 ⟨46603⟩ 237258

def event237260 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47299⟩⟩) (.authority (.operator))

def exact237261RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩, (1)⟩]

theorem exact237261RawTermsValid :
    exact237261RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237261 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47299⟩⟩) exact237261RawTerms (.finite 8192) 237260 .exactZero (none)

def event237262 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46456⟩⟩) 0 ⟨45108⟩ 11348

def event237263 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46456⟩⟩) (.authority (.programFamilyFact))

def event237264 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46456⟩⟩) (.finite 3720)

def event237265 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46457⟩⟩) 0 ⟨7177⟩ 15500

def event237266 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46457⟩⟩) 1 ⟨46456⟩ 237264

def event237267 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46457⟩⟩) (.authority (.operator))

def exact237268RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46457⟩⟩]⟩, (1)⟩]

theorem exact237268RawTermsValid :
    exact237268RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237268 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46457⟩⟩) exact237268RawTerms .large 237267 .exactZero (none)

def event237269 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46957⟩⟩) 0 ⟨46457⟩ 237268

def event237270 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46957⟩⟩) (.authority (.operator))

def exact237271RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46957⟩⟩]⟩, (1)⟩]

theorem exact237271RawTermsValid :
    exact237271RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237271 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46957⟩⟩) exact237271RawTerms (.finite 8192) 237270 .exactZero (none)

def event237272 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45109⟩⟩) 0 ⟨45106⟩ 11337

def event237273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45109⟩⟩) 1 ⟨6934⟩ 236778

def event237274 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45109⟩⟩) (.tensor (.predecessor 0 237272 .coefficient) (.predecessor 1 237273 .coefficient) true false)

def event237275 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45109⟩⟩, .operator (⟨11337, 0⟩, ⟨236778, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact237276RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact237276RawTermsValid :
    exact237276RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237276 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45109⟩⟩) exact237276RawTerms .large 237274 .exactZero (none)

def event237277 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8362⟩⟩) 0 ⟨5561⟩ 236648

def event237278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8362⟩⟩) 1 ⟨7284⟩ 17581

def event237279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8362⟩⟩) (.product (.predecessor 0 237277 .coefficient) (.predecessor 1 237278 .coefficient) (⟨false, false, none, none, none⟩))

def event237280 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8362⟩⟩, .operator (⟨236648, 0⟩, ⟨17581, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩)

def exact237281RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩]

theorem exact237281RawTermsValid :
    exact237281RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237281 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8362⟩⟩) exact237281RawTerms .large 237279 .exactZero (none)

def event237282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45110⟩⟩) 0 ⟨8362⟩ 237281

def event237283 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45110⟩⟩) 1 ⟨45109⟩ 237276

def event237284 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45110⟩⟩) (.sum [.predecessor 0 237282 .coefficient, .predecessor 1 237283 .coefficient])

def exact237285RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237285RawTermsValid :
    exact237285RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237285 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45110⟩⟩) exact237285RawTerms .large 237284 .exactZero (none)

def event237286 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45111⟩⟩) 0 ⟨45110⟩ 237285

def event237287 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45111⟩⟩) 1 ⟨110⟩ 17573

def event237288 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45111⟩⟩) (.sum [.predecessor 0 237286 .coefficient, .predecessor 1 237287 .coefficient])

def event237289 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45111⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨110⟩⟩]⟩) [⟨.result 17573 .coefficient, false, none⟩])

def event237290 : Event := .survivorFold (1) 237289

def exact237291RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237291RawTermsValid :
    exact237291RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237291 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45111⟩⟩) exact237291RawTerms .large 237288 (.finite 26) (some (237289))

def event237292 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45112⟩⟩) 0 ⟨45111⟩ 237291

def event237293 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45112⟩⟩) 1 ⟨14751⟩ 11340

def event237294 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45112⟩⟩) (.product (.predecessor 0 237292 .coefficient) (.predecessor 1 237293 .coefficient) (⟨false, true, none, none, some 1⟩))

def event237295 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45112⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14751⟩⟩], []⟩) [⟨.result 11340 .coefficient, true, some 1⟩])

def event237296 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45112⟩⟩) (.product (.result 237291 .summary) (.transfer 237295) (⟨false, false, none, none, none⟩))

def event237297 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45112⟩⟩, .operator (⟨237291, 1⟩, ⟨11340, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14751⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event237298 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45112⟩⟩, .operator (⟨237291, 0⟩, ⟨11340, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14751⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩)

def exact237299RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14751⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14751⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact237299RawTermsValid :
    exact237299RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237299 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45112⟩⟩) exact237299RawTerms .large 237294 (.finite 49414144) (some (237296))

def event237300 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14752⟩⟩) 0 ⟨14751⟩ 11340

def event237301 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14752⟩⟩) 1 ⟨6934⟩ 236778

def event237302 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14752⟩⟩) (.tensor (.predecessor 0 237300 .coefficient) (.predecessor 1 237301 .coefficient) true false)

def event237303 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14752⟩⟩, .operator (⟨11340, 0⟩, ⟨236778, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14751⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact237304RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14751⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact237304RawTermsValid :
    exact237304RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237304 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14752⟩⟩) exact237304RawTerms .large 237302 .exactZero (none)

def event237305 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8379⟩⟩) 0 ⟨5561⟩ 236648

def event237306 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8379⟩⟩) 1 ⟨7301⟩ 17622

def event237307 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8379⟩⟩) (.product (.predecessor 0 237305 .coefficient) (.predecessor 1 237306 .coefficient) (⟨false, false, none, none, none⟩))

def event237308 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8379⟩⟩, .operator (⟨236648, 0⟩, ⟨17622, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩)

def exact237309RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩]

theorem exact237309RawTermsValid :
    exact237309RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event237309 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8379⟩⟩) exact237309RawTerms .large 237307 .exactZero (none)

def event237310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14753⟩⟩) 0 ⟨8379⟩ 237309

def event237311 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14753⟩⟩) 1 ⟨14752⟩ 237304

def eventLeaf14816 : Array AnnotatedEvent := #[
  { event := event237056
    frameStart := 0 },
  { event := event237057
    frameStart := 0 },
  { event := event237058
    frameStart := 0 },
  { event := event237059
    frameStart := 0 },
  { event := event237060
    frameStart := 0 },
  { event := event237061
    frameStart := 0 },
  { event := event237062
    frameStart := 0 },
  { event := event237063
    frameStart := 0 },
  { event := event237064
    frameStart := 0 },
  { event := event237065
    frameStart := 0 },
  { event := event237066
    frameStart := 0 },
  { event := event237067
    frameStart := 0 },
  { event := event237068
    frameStart := 0 },
  { event := event237069
    frameStart := 0 },
  { event := event237070
    frameStart := 0 },
  { event := event237071
    frameStart := 0 }
]

def eventLeaf14817 : Array AnnotatedEvent := #[
  { event := event237072
    frameStart := 0 },
  { event := event237073
    frameStart := 0 },
  { event := event237074
    frameStart := 0 },
  { event := event237075
    frameStart := 0 },
  { event := event237076
    frameStart := 0 },
  { event := event237077
    frameStart := 0 },
  { event := event237078
    frameStart := 0 },
  { event := event237079
    frameStart := 0 },
  { event := event237080
    frameStart := 237080 },
  { event := event237081
    frameStart := 237080 },
  { event := event237082
    frameStart := 237080 },
  { event := event237083
    frameStart := 237080 },
  { event := event237084
    frameStart := 237080 },
  { event := event237085
    frameStart := 237080 },
  { event := event237086
    frameStart := 237080 },
  { event := event237087
    frameStart := 237080 }
]

def eventLeaf14818 : Array AnnotatedEvent := #[
  { event := event237088
    frameStart := 237080 },
  { event := event237089
    frameStart := 237080 },
  { event := event237090
    frameStart := 237080 },
  { event := event237091
    frameStart := 237080 },
  { event := event237092
    frameStart := 237080 },
  { event := event237093
    frameStart := 237080 },
  { event := event237094
    frameStart := 237080 },
  { event := event237095
    frameStart := 237080 },
  { event := event237096
    frameStart := 237080 },
  { event := event237097
    frameStart := 237080 },
  { event := event237098
    frameStart := 237080 },
  { event := event237099
    frameStart := 237080 },
  { event := event237100
    frameStart := 237080 },
  { event := event237101
    frameStart := 237080 },
  { event := event237102
    frameStart := 237080 },
  { event := event237103
    frameStart := 237080 }
]

def eventLeaf14819 : Array AnnotatedEvent := #[
  { event := event237104
    frameStart := 237080 },
  { event := event237105
    frameStart := 237080 },
  { event := event237106
    frameStart := 237080 },
  { event := event237107
    frameStart := 237080 },
  { event := event237108
    frameStart := 237080 },
  { event := event237109
    frameStart := 237080 },
  { event := event237110
    frameStart := 237080 },
  { event := event237111
    frameStart := 237080 },
  { event := event237112
    frameStart := 237080 },
  { event := event237113
    frameStart := 237080 },
  { event := event237114
    frameStart := 237080 },
  { event := event237115
    frameStart := 237080 },
  { event := event237116
    frameStart := 237080 },
  { event := event237117
    frameStart := 237080 },
  { event := event237118
    frameStart := 237080 },
  { event := event237119
    frameStart := 237080 }
]

def eventLeaf14820 : Array AnnotatedEvent := #[
  { event := event237120
    frameStart := 237080 },
  { event := event237121
    frameStart := 237080 },
  { event := event237122
    frameStart := 237080 },
  { event := event237123
    frameStart := 237080 },
  { event := event237124
    frameStart := 237080 },
  { event := event237125
    frameStart := 237080 },
  { event := event237126
    frameStart := 237080 },
  { event := event237127
    frameStart := 237080 },
  { event := event237128
    frameStart := 237080 },
  { event := event237129
    frameStart := 237080 },
  { event := event237130
    frameStart := 237080 },
  { event := event237131
    frameStart := 237080 },
  { event := event237132
    frameStart := 237080 },
  { event := event237133
    frameStart := 237080 },
  { event := event237134
    frameStart := 237134 },
  { event := event237135
    frameStart := 237134 }
]

def eventLeaf14821 : Array AnnotatedEvent := #[
  { event := event237136
    frameStart := 237134 },
  { event := event237137
    frameStart := 237134 },
  { event := event237138
    frameStart := 237134 },
  { event := event237139
    frameStart := 237134 },
  { event := event237140
    frameStart := 237134 },
  { event := event237141
    frameStart := 237134 },
  { event := event237142
    frameStart := 237134 },
  { event := event237143
    frameStart := 237134 },
  { event := event237144
    frameStart := 237134 },
  { event := event237145
    frameStart := 237134 },
  { event := event237146
    frameStart := 237134 },
  { event := event237147
    frameStart := 237134 },
  { event := event237148
    frameStart := 237134 },
  { event := event237149
    frameStart := 237134 },
  { event := event237150
    frameStart := 237134 },
  { event := event237151
    frameStart := 237134 }
]

def eventLeaf14822 : Array AnnotatedEvent := #[
  { event := event237152
    frameStart := 237134 },
  { event := event237153
    frameStart := 237134 },
  { event := event237154
    frameStart := 237134 },
  { event := event237155
    frameStart := 237134 },
  { event := event237156
    frameStart := 237134 },
  { event := event237157
    frameStart := 237134 },
  { event := event237158
    frameStart := 237134 },
  { event := event237159
    frameStart := 237134 },
  { event := event237160
    frameStart := 237134 },
  { event := event237161
    frameStart := 237134 },
  { event := event237162
    frameStart := 237134 },
  { event := event237163
    frameStart := 237134 },
  { event := event237164
    frameStart := 237134 },
  { event := event237165
    frameStart := 237134 },
  { event := event237166
    frameStart := 237134 },
  { event := event237167
    frameStart := 237134 }
]

def eventLeaf14823 : Array AnnotatedEvent := #[
  { event := event237168
    frameStart := 237134 },
  { event := event237169
    frameStart := 237134 },
  { event := event237170
    frameStart := 237134 },
  { event := event237171
    frameStart := 237134 },
  { event := event237172
    frameStart := 237134 },
  { event := event237173
    frameStart := 237134 },
  { event := event237174
    frameStart := 237134 },
  { event := event237175
    frameStart := 237134 },
  { event := event237176
    frameStart := 237134 },
  { event := event237177
    frameStart := 237134 },
  { event := event237178
    frameStart := 237134 },
  { event := event237179
    frameStart := 237134 },
  { event := event237180
    frameStart := 237134 },
  { event := event237181
    frameStart := 237134 },
  { event := event237182
    frameStart := 237134 },
  { event := event237183
    frameStart := 237134 }
]

def eventLeaf14824 : Array AnnotatedEvent := #[
  { event := event237184
    frameStart := 237134 },
  { event := event237185
    frameStart := 237134 },
  { event := event237186
    frameStart := 237134 },
  { event := event237187
    frameStart := 237134 },
  { event := event237188
    frameStart := 237134 },
  { event := event237189
    frameStart := 237134 },
  { event := event237190
    frameStart := 237134 },
  { event := event237191
    frameStart := 237134 },
  { event := event237192
    frameStart := 237134 },
  { event := event237193
    frameStart := 237134 },
  { event := event237194
    frameStart := 237134 },
  { event := event237195
    frameStart := 237134 },
  { event := event237196
    frameStart := 237134 },
  { event := event237197
    frameStart := 237134 },
  { event := event237198
    frameStart := 237134 },
  { event := event237199
    frameStart := 237134 }
]

def eventLeaf14825 : Array AnnotatedEvent := #[
  { event := event237200
    frameStart := 237134 },
  { event := event237201
    frameStart := 237134 },
  { event := event237202
    frameStart := 237134 },
  { event := event237203
    frameStart := 237134 },
  { event := event237204
    frameStart := 237134 },
  { event := event237205
    frameStart := 237134 },
  { event := event237206
    frameStart := 237134 },
  { event := event237207
    frameStart := 237134 },
  { event := event237208
    frameStart := 237134 },
  { event := event237209
    frameStart := 237134 },
  { event := event237210
    frameStart := 237134 },
  { event := event237211
    frameStart := 237134 },
  { event := event237212
    frameStart := 237134 },
  { event := event237213
    frameStart := 237134 },
  { event := event237214
    frameStart := 237134 },
  { event := event237215
    frameStart := 237134 }
]

def eventLeaf14826 : Array AnnotatedEvent := #[
  { event := event237216
    frameStart := 237134 },
  { event := event237217
    frameStart := 237134 },
  { event := event237218
    frameStart := 237134 },
  { event := event237219
    frameStart := 237134 },
  { event := event237220
    frameStart := 237134 },
  { event := event237221
    frameStart := 237134 },
  { event := event237222
    frameStart := 237134 },
  { event := event237223
    frameStart := 237134 },
  { event := event237224
    frameStart := 237134 },
  { event := event237225
    frameStart := 237134 },
  { event := event237226
    frameStart := 237134 },
  { event := event237227
    frameStart := 237134 },
  { event := event237228
    frameStart := 237134 },
  { event := event237229
    frameStart := 237134 },
  { event := event237230
    frameStart := 237134 },
  { event := event237231
    frameStart := 237134 }
]

def eventLeaf14827 : Array AnnotatedEvent := #[
  { event := event237232
    frameStart := 237134 },
  { event := event237233
    frameStart := 237134 },
  { event := event237234
    frameStart := 237134 },
  { event := event237235
    frameStart := 237134 },
  { event := event237236
    frameStart := 237134 },
  { event := event237237
    frameStart := 237134 },
  { event := event237238
    frameStart := 0 },
  { event := event237239
    frameStart := 0 },
  { event := event237240
    frameStart := 0 },
  { event := event237241
    frameStart := 0 },
  { event := event237242
    frameStart := 0 },
  { event := event237243
    frameStart := 0 },
  { event := event237244
    frameStart := 0 },
  { event := event237245
    frameStart := 0 },
  { event := event237246
    frameStart := 0 },
  { event := event237247
    frameStart := 0 }
]

def eventLeaf14828 : Array AnnotatedEvent := #[
  { event := event237248
    frameStart := 0 },
  { event := event237249
    frameStart := 0 },
  { event := event237250
    frameStart := 0 },
  { event := event237251
    frameStart := 0 },
  { event := event237252
    frameStart := 0 },
  { event := event237253
    frameStart := 0 },
  { event := event237254
    frameStart := 0 },
  { event := event237255
    frameStart := 0 },
  { event := event237256
    frameStart := 0 },
  { event := event237257
    frameStart := 0 },
  { event := event237258
    frameStart := 0 },
  { event := event237259
    frameStart := 0 },
  { event := event237260
    frameStart := 0 },
  { event := event237261
    frameStart := 0 },
  { event := event237262
    frameStart := 0 },
  { event := event237263
    frameStart := 0 }
]

def eventLeaf14829 : Array AnnotatedEvent := #[
  { event := event237264
    frameStart := 0 },
  { event := event237265
    frameStart := 0 },
  { event := event237266
    frameStart := 0 },
  { event := event237267
    frameStart := 0 },
  { event := event237268
    frameStart := 0 },
  { event := event237269
    frameStart := 0 },
  { event := event237270
    frameStart := 0 },
  { event := event237271
    frameStart := 0 },
  { event := event237272
    frameStart := 0 },
  { event := event237273
    frameStart := 0 },
  { event := event237274
    frameStart := 0 },
  { event := event237275
    frameStart := 0 },
  { event := event237276
    frameStart := 0 },
  { event := event237277
    frameStart := 0 },
  { event := event237278
    frameStart := 0 },
  { event := event237279
    frameStart := 0 }
]

def eventLeaf14830 : Array AnnotatedEvent := #[
  { event := event237280
    frameStart := 0 },
  { event := event237281
    frameStart := 0 },
  { event := event237282
    frameStart := 0 },
  { event := event237283
    frameStart := 0 },
  { event := event237284
    frameStart := 0 },
  { event := event237285
    frameStart := 0 },
  { event := event237286
    frameStart := 0 },
  { event := event237287
    frameStart := 0 },
  { event := event237288
    frameStart := 0 },
  { event := event237289
    frameStart := 0 },
  { event := event237290
    frameStart := 0 },
  { event := event237291
    frameStart := 0 },
  { event := event237292
    frameStart := 0 },
  { event := event237293
    frameStart := 0 },
  { event := event237294
    frameStart := 0 },
  { event := event237295
    frameStart := 0 }
]

def eventLeaf14831 : Array AnnotatedEvent := #[
  { event := event237296
    frameStart := 0 },
  { event := event237297
    frameStart := 0 },
  { event := event237298
    frameStart := 0 },
  { event := event237299
    frameStart := 0 },
  { event := event237300
    frameStart := 0 },
  { event := event237301
    frameStart := 0 },
  { event := event237302
    frameStart := 0 },
  { event := event237303
    frameStart := 0 },
  { event := event237304
    frameStart := 0 },
  { event := event237305
    frameStart := 0 },
  { event := event237306
    frameStart := 0 },
  { event := event237307
    frameStart := 0 },
  { event := event237308
    frameStart := 0 },
  { event := event237309
    frameStart := 0 },
  { event := event237310
    frameStart := 0 },
  { event := event237311
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events926
