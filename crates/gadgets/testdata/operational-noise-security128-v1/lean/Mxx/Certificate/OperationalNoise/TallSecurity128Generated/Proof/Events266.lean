import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events266

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event68096 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31682⟩⟩) 0 ⟨31681⟩ 68095

def event68097 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31682⟩⟩) 1 ⟨31677⟩ 68065

def event68098 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31682⟩⟩) (.sum [.predecessor 0 68096 .coefficient, .predecessor 1 68097 .coefficient])

def event68099 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31682⟩⟩, .operator (⟨68095, 1⟩, ⟨68065, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩)

def event68100 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31682⟩⟩) (.sum [.result 68095 .summary, .result 68065 .summary])

def exact68101RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68101RawTermsValid :
    exact68101RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68101 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31682⟩⟩) exact68101RawTerms .large 68098 (.finite 279177986048) (some (68100))

def event68102 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33537⟩⟩) 0 ⟨31682⟩ 68101

def event68103 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33537⟩⟩) 1 ⟨33536⟩ 68037

def event68104 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33537⟩⟩) (.product (.predecessor 0 68102 .coefficient) (.predecessor 1 68103 .coefficient) (⟨false, false, none, none, none⟩))

def event68105 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33537⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨33536⟩⟩]⟩) [⟨.result 68037 .coefficient, false, none⟩])

def event68106 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33537⟩⟩) (.product (.result 68101 .summary) (.transfer 68105) (⟨false, false, none, none, none⟩))

def event68107 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33537⟩⟩, .operator (⟨68101, 1⟩, ⟨68037, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33536⟩⟩]⟩, (-1)⟩)

def event68108 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨33537⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33536⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33536⟩⟩) ⟨32991⟩ 68034)

def event68109 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33537⟩⟩, .relation 68108 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨32991⟩⟩]⟩, (-1)⟩)

def event68110 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33537⟩⟩, .operator (⟨68101, 0⟩, ⟨68037, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33536⟩⟩]⟩, (1)⟩)

def exact68111RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33536⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨32991⟩⟩]⟩, (-1)⟩]

theorem exact68111RawTermsValid :
    exact68111RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68111 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33537⟩⟩) exact68111RawTerms .large 68104 (.finite 2997650799598260715520) (some (68106))

def event68112 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32459⟩⟩) 0 ⟨31676⟩ 2671

def event68113 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32459⟩⟩) (.authority (.relationPreimageSource ⟨39⟩))

def exact68114RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32459⟩⟩]⟩, (1)⟩]

theorem exact68114RawTermsValid :
    exact68114RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68114 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32459⟩⟩) exact68114RawTerms (.finite 5647228698) 68113 .exactZero (none)

def event68115 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32461⟩⟩) 0 ⟨32459⟩ 68114

def event68116 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32461⟩⟩) 1 ⟨2370⟩ 4

def event68117 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32461⟩⟩) (.scale (.predecessor 0 68115 .coefficient) (.value (.predecessor 1 68116 .coefficient)))

def exact68118RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32459⟩⟩]⟩, (1)⟩]

theorem exact68118RawTermsValid :
    exact68118RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68118 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32461⟩⟩) exact68118RawTerms (.finite 5647228698) 68117 .exactZero (none)

def event68119 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32462⟩⟩) 0 ⟨10792⟩ 61370

def event68120 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32462⟩⟩) 1 ⟨32461⟩ 68118

def event68121 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32462⟩⟩) (.product (.predecessor 0 68119 .coefficient) (.predecessor 1 68120 .coefficient) (⟨false, false, none, none, none⟩))

def event68122 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32462⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨32459⟩⟩]⟩) [⟨.result 68114 .coefficient, false, none⟩])

def event68123 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32462⟩⟩) (.product (.result 61370 .summary) (.transfer 68122) (⟨false, false, none, none, none⟩))

def event68124 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32462⟩⟩, .operator (⟨61370, 0⟩, ⟨68118, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32459⟩⟩]⟩, (1)⟩)

def event68125 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨32460⟩⟩)

def event68126 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event68127 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event68128 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.authority (.operator))

def event68129 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.finite 16)

def event68130 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event68131 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event68132 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event68133 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event68134 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 68133

def event68135 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 68131

def event68136 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 68134 .coefficient) (.value (.predecessor 1 68135 .coefficient)))

def event68137 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event68138 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 0 ⟨392⟩ 68137

def event68139 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 1 ⟨10691⟩ 68129

def event68140 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.sum [.predecessor 0 68138 .coefficient, .predecessor 1 68139 .coefficient])

def event68141 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.finite 655356)

def event68142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 0 ⟨10693⟩ 68141

def event68143 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 1 ⟨5426⟩ 68127

def event68144 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.identity (.predecessor 1 68143 .coefficient))

def event68145 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.finite 655360)

def event68146 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24374⟩⟩) 0 ⟨10749⟩ 68145

def event68147 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24374⟩⟩) (.authority (.programFamilyFact))

def exact68148RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩], []⟩, (1)⟩]

theorem exact68148RawTermsValid :
    exact68148RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68148 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24374⟩⟩) exact68148RawTerms (.finite 6) 68147 .exactZero (none)

def event68149 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31674⟩⟩) 0 ⟨10749⟩ 68145

def event68150 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31674⟩⟩) (.authority (.programFamilyFact))

def exact68151RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31674⟩⟩], []⟩, (1)⟩]

theorem exact68151RawTermsValid :
    exact68151RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68151 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31674⟩⟩) exact68151RawTerms (.finite 6) 68150 .exactZero (none)

def event68152 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31675⟩⟩) 0 ⟨31674⟩ 68151

def event68153 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31675⟩⟩) 1 ⟨24374⟩ 68148

def event68154 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31675⟩⟩) (.product (.predecessor 0 68152 .coefficient) (.predecessor 1 68153 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event68155 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31675⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], []⟩) [⟨.result 68151 .coefficient, true, some 1⟩, ⟨.result 68148 .coefficient, true, some 1⟩])

def event68156 : Event := .survivorFold (1) 68155

def exact68157RawTerms : List Term := []

theorem exact68157RawTermsValid :
    exact68157RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68157 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31675⟩⟩) exact68157RawTerms (.finite 36) 68154 (.finite 36) (some (68155))

def event68158 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31676⟩⟩) 0 ⟨31675⟩ 68157

def event68159 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31676⟩⟩) (.identity (.predecessor 0 68158 .coefficient))

def event68160 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31676⟩⟩) (.finite 36)

def event68161 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32459⟩⟩) 0 ⟨31676⟩ 68160

def event68162 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32459⟩⟩) (.authority (.relationPreimageSource ⟨39⟩))

def exact68163RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32459⟩⟩]⟩, (1)⟩]

theorem exact68163RawTermsValid :
    exact68163RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68163 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32459⟩⟩) exact68163RawTerms (.finite 5647228698) 68162 .exactZero (none)

def event68164 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact68165RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact68165RawTermsValid :
    exact68165RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68165 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact68165RawTerms .large 68164 .exactZero (none)

def event68166 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32460⟩⟩) 0 ⟨35⟩ 68165

def event68167 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32460⟩⟩) 1 ⟨32459⟩ 68163

def event68168 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32460⟩⟩) (.product (.predecessor 0 68166 .coefficient) (.predecessor 1 68167 .coefficient) (⟨false, false, none, none, none⟩))

def event68169 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32460⟩⟩, .operator (⟨68165, 0⟩, ⟨68163, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32459⟩⟩]⟩, (1)⟩)

def exact68170RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32459⟩⟩]⟩, (1)⟩]

theorem exact68170RawTermsValid :
    exact68170RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68170 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32460⟩⟩) exact68170RawTerms .large 68168 .exactZero (none)

def event68171 : Event := .preFoldPolynomial 68170 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32459⟩⟩]⟩, (1)⟩] .exactZero none

def exact68172RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32459⟩⟩]⟩, (1)⟩]

def event68172 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨32460⟩⟩) 68171 exact68172RawTerms .large 68168 .exactZero (none)

def event68173 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨33540⟩⟩)

def event68174 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event68175 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event68176 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.authority (.operator))

def event68177 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.finite 16)

def event68178 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event68179 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event68180 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event68181 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event68182 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 68181

def event68183 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 68179

def event68184 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 68182 .coefficient) (.value (.predecessor 1 68183 .coefficient)))

def event68185 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event68186 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 0 ⟨392⟩ 68185

def event68187 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 1 ⟨10691⟩ 68177

def event68188 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.sum [.predecessor 0 68186 .coefficient, .predecessor 1 68187 .coefficient])

def event68189 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.finite 655356)

def event68190 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 0 ⟨10693⟩ 68189

def event68191 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 1 ⟨5426⟩ 68175

def event68192 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.identity (.predecessor 1 68191 .coefficient))

def event68193 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.finite 655360)

def event68194 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24374⟩⟩) 0 ⟨10749⟩ 68193

def event68195 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24374⟩⟩) (.authority (.programFamilyFact))

def exact68196RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩], []⟩, (1)⟩]

theorem exact68196RawTermsValid :
    exact68196RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68196 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24374⟩⟩) exact68196RawTerms (.finite 6) 68195 .exactZero (none)

def event68197 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31674⟩⟩) 0 ⟨10749⟩ 68193

def event68198 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31674⟩⟩) (.authority (.programFamilyFact))

def exact68199RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31674⟩⟩], []⟩, (1)⟩]

theorem exact68199RawTermsValid :
    exact68199RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68199 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31674⟩⟩) exact68199RawTerms (.finite 6) 68198 .exactZero (none)

def event68200 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31675⟩⟩) 0 ⟨31674⟩ 68199

def event68201 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31675⟩⟩) 1 ⟨24374⟩ 68196

def event68202 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31675⟩⟩) (.product (.predecessor 0 68200 .coefficient) (.predecessor 1 68201 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event68203 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31675⟩⟩, .operator (⟨68199, 0⟩, ⟨68196, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], []⟩, (1)⟩)

def exact68204RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], []⟩, (1)⟩]

theorem exact68204RawTermsValid :
    exact68204RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68204 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31675⟩⟩) exact68204RawTerms (.finite 36) 68202 .exactZero (none)

def event68205 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31676⟩⟩) 0 ⟨31675⟩ 68204

def event68206 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31676⟩⟩) (.identity (.predecessor 0 68205 .coefficient))

def event68207 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31676⟩⟩) (.finite 36)

def event68208 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32990⟩⟩) 0 ⟨31676⟩ 68207

def event68209 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32990⟩⟩) (.authority (.programFamilyFact))

def event68210 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨32990⟩⟩) (.finite 3720)

def event68211 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event68212 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32991⟩⟩) 0 ⟨7177⟩ 68211

def event68213 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32991⟩⟩) 1 ⟨32990⟩ 68210

def event68214 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32991⟩⟩) (.authority (.operator))

def exact68215RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32991⟩⟩]⟩, (1)⟩]

theorem exact68215RawTermsValid :
    exact68215RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68215 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32991⟩⟩) exact68215RawTerms .large 68214 .exactZero (none)

def event68216 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33536⟩⟩) 0 ⟨32991⟩ 68215

def event68217 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33536⟩⟩) (.authority (.operator))

def exact68218RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨33536⟩⟩]⟩, (1)⟩]

theorem exact68218RawTermsValid :
    exact68218RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68218 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33536⟩⟩) exact68218RawTerms (.finite 8192) 68217 .exactZero (none)

def event68219 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event68220 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event68221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33254⟩⟩) 0 ⟨31676⟩ 68207

def event68222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33254⟩⟩) 1 ⟨136⟩ 68220

def event68223 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33254⟩⟩) (.sum [.predecessor 0 68221 .coefficient, .predecessor 1 68222 .coefficient])

def event68224 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨33254⟩⟩) (.finite 36)

def event68225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33255⟩⟩) 0 ⟨33254⟩ 68224

def event68226 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33255⟩⟩) (.identity (.predecessor 0 68225 .coefficient))

def exact68227RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], []⟩, (1)⟩]

theorem exact68227RawTermsValid :
    exact68227RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68227 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33255⟩⟩) exact68227RawTerms (.finite 36) 68226 .exactZero (none)

def event68228 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact68229RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact68229RawTermsValid :
    exact68229RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68229 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact68229RawTerms .large 68228 .exactZero (none)

def event68230 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33256⟩⟩) 0 ⟨6908⟩ 68229

def event68231 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33256⟩⟩) 1 ⟨33255⟩ 68227

def event68232 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33256⟩⟩) (.product (.predecessor 0 68230 .coefficient) (.predecessor 1 68231 .coefficient) (⟨false, false, none, none, none⟩))

def event68233 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33256⟩⟩, .operator (⟨68229, 0⟩, ⟨68227, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact68234RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact68234RawTermsValid :
    exact68234RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68234 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33256⟩⟩) exact68234RawTerms .large 68232 .exactZero (none)

def event68235 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event68236 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event68237 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 68211

def event68238 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact68239RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact68239RawTermsValid :
    exact68239RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68239 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact68239RawTerms .large 68238 .exactZero (none)

def event68240 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7307⟩⟩) 0 ⟨7178⟩ 68239

def event68241 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7307⟩⟩) (.identity (.predecessor 0 68240 .coefficient))

def exact68242RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩]

theorem exact68242RawTermsValid :
    exact68242RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68242 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7307⟩⟩) exact68242RawTerms .large 68241 .exactZero (none)

def event68243 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9577⟩⟩) 0 ⟨7307⟩ 68242

def event68244 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9577⟩⟩) (.authority (.operator))

def exact68245RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩]

theorem exact68245RawTermsValid :
    exact68245RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68245 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9577⟩⟩) exact68245RawTerms (.finite 8192) 68244 .exactZero (none)

def event68246 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9578⟩⟩) 0 ⟨9577⟩ 68245

def event68247 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9578⟩⟩) 1 ⟨2370⟩ 68236

def event68248 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9578⟩⟩) (.scale (.predecessor 0 68246 .coefficient) (.value (.predecessor 1 68247 .coefficient)))

def exact68249RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩]

theorem exact68249RawTermsValid :
    exact68249RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68249 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9578⟩⟩) exact68249RawTerms (.finite 8192) 68248 .exactZero (none)

def event68250 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7287⟩⟩) 0 ⟨7178⟩ 68239

def event68251 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7287⟩⟩) (.identity (.predecessor 0 68250 .coefficient))

def exact68252RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩]⟩, (1)⟩]

theorem exact68252RawTermsValid :
    exact68252RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68252 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7287⟩⟩) exact68252RawTerms .large 68251 .exactZero (none)

def event68253 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9579⟩⟩) 0 ⟨7287⟩ 68252

def event68254 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9579⟩⟩) 1 ⟨9578⟩ 68249

def event68255 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9579⟩⟩) (.product (.predecessor 0 68253 .coefficient) (.predecessor 1 68254 .coefficient) (⟨false, false, none, none, none⟩))

def event68256 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9579⟩⟩, .operator (⟨68252, 0⟩, ⟨68249, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩)

def exact68257RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩]

theorem exact68257RawTermsValid :
    exact68257RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68257 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9579⟩⟩) exact68257RawTerms .large 68255 .exactZero (none)

def event68258 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33257⟩⟩) 0 ⟨9579⟩ 68257

def event68259 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33257⟩⟩) 1 ⟨33256⟩ 68234

def event68260 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33257⟩⟩) (.sum [.predecessor 0 68258 .coefficient, .predecessor 1 68259 .coefficient])

def exact68261RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68261RawTermsValid :
    exact68261RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68261 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33257⟩⟩) exact68261RawTerms .large 68260 .exactZero (none)

def event68262 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33539⟩⟩) 0 ⟨33257⟩ 68261

def event68263 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33539⟩⟩) 1 ⟨33536⟩ 68218

def event68264 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33539⟩⟩) (.product (.predecessor 0 68262 .coefficient) (.predecessor 1 68263 .coefficient) (⟨false, false, none, none, none⟩))

def event68265 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33539⟩⟩, .operator (⟨68261, 0⟩, ⟨68218, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33536⟩⟩]⟩, (1)⟩)

def event68266 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33539⟩⟩, .operator (⟨68261, 1⟩, ⟨68218, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33536⟩⟩]⟩, (-1)⟩)

def event68267 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨33539⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33536⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33536⟩⟩) ⟨32991⟩ 68215)

def event68268 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33539⟩⟩, .relation 68267 0, ⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨32991⟩⟩]⟩, (-1)⟩)

def exact68269RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33536⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨32991⟩⟩]⟩, (-1)⟩]

theorem exact68269RawTermsValid :
    exact68269RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68269 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33539⟩⟩) exact68269RawTerms .large 68264 .exactZero (none)

def event68270 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31884⟩⟩) 0 ⟨31676⟩ 68207

def event68271 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31884⟩⟩) (.authority (.programFamilyFact))

def exact68272RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], []⟩, (1)⟩]

theorem exact68272RawTermsValid :
    exact68272RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68272 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31884⟩⟩) exact68272RawTerms (.finite 6) 68271 .exactZero (none)

def event68273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31886⟩⟩) 0 ⟨6908⟩ 68229

def event68274 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31886⟩⟩) 1 ⟨31884⟩ 68272

def event68275 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31886⟩⟩) (.product (.predecessor 0 68273 .coefficient) (.predecessor 1 68274 .coefficient) (⟨false, true, none, none, some 1⟩))

def event68276 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31886⟩⟩, .operator (⟨68229, 0⟩, ⟨68272, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact68277RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact68277RawTermsValid :
    exact68277RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68277 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31886⟩⟩) exact68277RawTerms .large 68275 .exactZero (none)

def event68278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7182⟩⟩) 0 ⟨7177⟩ 68211

def event68279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7182⟩⟩) (.authority (.operator))

def exact68280RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩]

theorem exact68280RawTermsValid :
    exact68280RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68280 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7182⟩⟩) exact68280RawTerms .large 68279 .exactZero (none)

def event68281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31887⟩⟩) 0 ⟨7182⟩ 68280

def event68282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31887⟩⟩) 1 ⟨31886⟩ 68277

def event68283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31887⟩⟩) (.sum [.predecessor 0 68281 .coefficient, .predecessor 1 68282 .coefficient])

def exact68284RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68284RawTermsValid :
    exact68284RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68284 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31887⟩⟩) exact68284RawTerms .large 68283 .exactZero (none)

def event68285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33540⟩⟩) 0 ⟨31887⟩ 68284

def event68286 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33540⟩⟩) 1 ⟨33539⟩ 68269

def event68287 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33540⟩⟩) (.sum [.predecessor 0 68285 .coefficient, .predecessor 1 68286 .coefficient])

def exact68288RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33536⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨32991⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68288RawTermsValid :
    exact68288RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68288 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33540⟩⟩) exact68288RawTerms .large 68287 .exactZero (none)

def event68289 : Event := .preFoldPolynomial 68288 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33536⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨32991⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact68290RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33536⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨32991⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event68290 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨33540⟩⟩) 68289 exact68290RawTerms .large 68287 .exactZero (none)

def event68291 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨31676⟩⟩) ⟨⟨61⟩, ⟨39⟩, ⟨135⟩⟩ ⟨68125, 68291⟩

def event68292 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨32462⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32459⟩⟩]⟩) (1) 0 2 (.universal 68291 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32459⟩⟩]⟩) (none) 68290)

def event68293 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32462⟩⟩, .relation 68292 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩)

def event68294 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32462⟩⟩, .relation 68292 1, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33536⟩⟩]⟩, (-1)⟩)

def event68295 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32462⟩⟩, .relation 68292 2, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨32991⟩⟩]⟩, (1)⟩)

def event68296 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32462⟩⟩, .relation 68292 3, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact68297RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33536⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨32991⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68297RawTermsValid :
    exact68297RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68297 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32462⟩⟩) exact68297RawTerms .large 68121 (.finite 202072841853861888) (some (68123))

def event68298 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33538⟩⟩) 0 ⟨32462⟩ 68297

def event68299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33538⟩⟩) 1 ⟨33537⟩ 68111

def event68300 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33538⟩⟩) (.sum [.predecessor 0 68298 .coefficient, .predecessor 1 68299 .coefficient])

def event68301 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33538⟩⟩, .operator (⟨68297, 2⟩, ⟨68111, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨32991⟩⟩]⟩, (-1)⟩)

def event68302 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33538⟩⟩, .operator (⟨68297, 1⟩, ⟨68111, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33536⟩⟩]⟩, (1)⟩)

def event68303 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33538⟩⟩) (.sum [.result 68297 .summary, .result 68111 .summary])

def exact68304RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68304RawTermsValid :
    exact68304RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68304 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33538⟩⟩) exact68304RawTerms .large 68300 (.finite 2997852872440114577408) (some (68303))

def event68305 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34111⟩⟩) 0 ⟨33538⟩ 68304

def event68306 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34111⟩⟩) 1 ⟨34109⟩ 68027

def event68307 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34111⟩⟩) (.product (.predecessor 0 68305 .coefficient) (.predecessor 1 68306 .coefficient) (⟨false, false, none, none, none⟩))

def event68308 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34111⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩) [⟨.result 68027 .coefficient, false, none⟩])

def event68309 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34111⟩⟩) (.product (.result 68304 .summary) (.transfer 68308) (⟨false, false, none, none, none⟩))

def event68310 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34111⟩⟩, .operator (⟨68304, 0⟩, ⟨68027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩, (1)⟩)

def event68311 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34111⟩⟩, .operator (⟨68304, 1⟩, ⟨68027, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩, (-1)⟩)

def event68312 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨34111⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨34109⟩⟩) ⟨33164⟩ 68024)

def event68313 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34111⟩⟩, .relation 68312 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨33164⟩⟩]⟩, (-1)⟩)

def exact68314RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨33164⟩⟩]⟩, (-1)⟩]

theorem exact68314RawTermsValid :
    exact68314RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68314 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34111⟩⟩) exact68314RawTerms .large 68307 (.finite 32189200113374879571150551121920) (some (68309))

def event68315 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32836⟩⟩) 0 ⟨31885⟩ 2677

def event68316 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32836⟩⟩) (.authority (.relationPreimageSource ⟨63⟩))

def exact68317RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩, (1)⟩]

theorem exact68317RawTermsValid :
    exact68317RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68317 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32836⟩⟩) exact68317RawTerms (.finite 5647228698) 68316 .exactZero (none)

def event68318 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32838⟩⟩) 0 ⟨32836⟩ 68317

def event68319 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32838⟩⟩) 1 ⟨2370⟩ 4

def event68320 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32838⟩⟩) (.scale (.predecessor 0 68318 .coefficient) (.value (.predecessor 1 68319 .coefficient)))

def exact68321RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩, (1)⟩]

theorem exact68321RawTermsValid :
    exact68321RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68321 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32838⟩⟩) exact68321RawTerms (.finite 5647228698) 68320 .exactZero (none)

def event68322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32839⟩⟩) 0 ⟨10792⟩ 61370

def event68323 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32839⟩⟩) 1 ⟨32838⟩ 68321

def event68324 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32839⟩⟩) (.product (.predecessor 0 68322 .coefficient) (.predecessor 1 68323 .coefficient) (⟨false, false, none, none, none⟩))

def event68325 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32839⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩) [⟨.result 68317 .coefficient, false, none⟩])

def event68326 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32839⟩⟩) (.product (.result 61370 .summary) (.transfer 68325) (⟨false, false, none, none, none⟩))

def event68327 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32839⟩⟩, .operator (⟨61370, 0⟩, ⟨68321, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩, (1)⟩)

def event68328 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨32837⟩⟩)

def event68329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event68330 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event68331 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.authority (.operator))

def event68332 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.finite 16)

def event68333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event68334 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event68335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event68336 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event68337 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 68336

def event68338 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 68334

def event68339 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 68337 .coefficient) (.value (.predecessor 1 68338 .coefficient)))

def event68340 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event68341 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 0 ⟨392⟩ 68340

def event68342 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 1 ⟨10691⟩ 68332

def event68343 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.sum [.predecessor 0 68341 .coefficient, .predecessor 1 68342 .coefficient])

def event68344 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.finite 655356)

def event68345 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 0 ⟨10693⟩ 68344

def event68346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 1 ⟨5426⟩ 68330

def event68347 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.identity (.predecessor 1 68346 .coefficient))

def event68348 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.finite 655360)

def event68349 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24374⟩⟩) 0 ⟨10749⟩ 68348

def event68350 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24374⟩⟩) (.authority (.programFamilyFact))

def exact68351RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩], []⟩, (1)⟩]

theorem exact68351RawTermsValid :
    exact68351RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68351 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24374⟩⟩) exact68351RawTerms (.finite 6) 68350 .exactZero (none)

def eventLeaf4256 : Array AnnotatedEvent := #[
  { event := event68096
    frameStart := 0 },
  { event := event68097
    frameStart := 0 },
  { event := event68098
    frameStart := 0 },
  { event := event68099
    frameStart := 0 },
  { event := event68100
    frameStart := 0 },
  { event := event68101
    frameStart := 0 },
  { event := event68102
    frameStart := 0 },
  { event := event68103
    frameStart := 0 },
  { event := event68104
    frameStart := 0 },
  { event := event68105
    frameStart := 0 },
  { event := event68106
    frameStart := 0 },
  { event := event68107
    frameStart := 0 },
  { event := event68108
    frameStart := 0 },
  { event := event68109
    frameStart := 0 },
  { event := event68110
    frameStart := 0 },
  { event := event68111
    frameStart := 0 }
]

def eventLeaf4257 : Array AnnotatedEvent := #[
  { event := event68112
    frameStart := 0 },
  { event := event68113
    frameStart := 0 },
  { event := event68114
    frameStart := 0 },
  { event := event68115
    frameStart := 0 },
  { event := event68116
    frameStart := 0 },
  { event := event68117
    frameStart := 0 },
  { event := event68118
    frameStart := 0 },
  { event := event68119
    frameStart := 0 },
  { event := event68120
    frameStart := 0 },
  { event := event68121
    frameStart := 0 },
  { event := event68122
    frameStart := 0 },
  { event := event68123
    frameStart := 0 },
  { event := event68124
    frameStart := 0 },
  { event := event68125
    frameStart := 68125 },
  { event := event68126
    frameStart := 68125 },
  { event := event68127
    frameStart := 68125 }
]

def eventLeaf4258 : Array AnnotatedEvent := #[
  { event := event68128
    frameStart := 68125 },
  { event := event68129
    frameStart := 68125 },
  { event := event68130
    frameStart := 68125 },
  { event := event68131
    frameStart := 68125 },
  { event := event68132
    frameStart := 68125 },
  { event := event68133
    frameStart := 68125 },
  { event := event68134
    frameStart := 68125 },
  { event := event68135
    frameStart := 68125 },
  { event := event68136
    frameStart := 68125 },
  { event := event68137
    frameStart := 68125 },
  { event := event68138
    frameStart := 68125 },
  { event := event68139
    frameStart := 68125 },
  { event := event68140
    frameStart := 68125 },
  { event := event68141
    frameStart := 68125 },
  { event := event68142
    frameStart := 68125 },
  { event := event68143
    frameStart := 68125 }
]

def eventLeaf4259 : Array AnnotatedEvent := #[
  { event := event68144
    frameStart := 68125 },
  { event := event68145
    frameStart := 68125 },
  { event := event68146
    frameStart := 68125 },
  { event := event68147
    frameStart := 68125 },
  { event := event68148
    frameStart := 68125 },
  { event := event68149
    frameStart := 68125 },
  { event := event68150
    frameStart := 68125 },
  { event := event68151
    frameStart := 68125 },
  { event := event68152
    frameStart := 68125 },
  { event := event68153
    frameStart := 68125 },
  { event := event68154
    frameStart := 68125 },
  { event := event68155
    frameStart := 68125 },
  { event := event68156
    frameStart := 68125 },
  { event := event68157
    frameStart := 68125 },
  { event := event68158
    frameStart := 68125 },
  { event := event68159
    frameStart := 68125 }
]

def eventLeaf4260 : Array AnnotatedEvent := #[
  { event := event68160
    frameStart := 68125 },
  { event := event68161
    frameStart := 68125 },
  { event := event68162
    frameStart := 68125 },
  { event := event68163
    frameStart := 68125 },
  { event := event68164
    frameStart := 68125 },
  { event := event68165
    frameStart := 68125 },
  { event := event68166
    frameStart := 68125 },
  { event := event68167
    frameStart := 68125 },
  { event := event68168
    frameStart := 68125 },
  { event := event68169
    frameStart := 68125 },
  { event := event68170
    frameStart := 68125 },
  { event := event68171
    frameStart := 68125 },
  { event := event68172
    frameStart := 68125 },
  { event := event68173
    frameStart := 68173 },
  { event := event68174
    frameStart := 68173 },
  { event := event68175
    frameStart := 68173 }
]

def eventLeaf4261 : Array AnnotatedEvent := #[
  { event := event68176
    frameStart := 68173 },
  { event := event68177
    frameStart := 68173 },
  { event := event68178
    frameStart := 68173 },
  { event := event68179
    frameStart := 68173 },
  { event := event68180
    frameStart := 68173 },
  { event := event68181
    frameStart := 68173 },
  { event := event68182
    frameStart := 68173 },
  { event := event68183
    frameStart := 68173 },
  { event := event68184
    frameStart := 68173 },
  { event := event68185
    frameStart := 68173 },
  { event := event68186
    frameStart := 68173 },
  { event := event68187
    frameStart := 68173 },
  { event := event68188
    frameStart := 68173 },
  { event := event68189
    frameStart := 68173 },
  { event := event68190
    frameStart := 68173 },
  { event := event68191
    frameStart := 68173 }
]

def eventLeaf4262 : Array AnnotatedEvent := #[
  { event := event68192
    frameStart := 68173 },
  { event := event68193
    frameStart := 68173 },
  { event := event68194
    frameStart := 68173 },
  { event := event68195
    frameStart := 68173 },
  { event := event68196
    frameStart := 68173 },
  { event := event68197
    frameStart := 68173 },
  { event := event68198
    frameStart := 68173 },
  { event := event68199
    frameStart := 68173 },
  { event := event68200
    frameStart := 68173 },
  { event := event68201
    frameStart := 68173 },
  { event := event68202
    frameStart := 68173 },
  { event := event68203
    frameStart := 68173 },
  { event := event68204
    frameStart := 68173 },
  { event := event68205
    frameStart := 68173 },
  { event := event68206
    frameStart := 68173 },
  { event := event68207
    frameStart := 68173 }
]

def eventLeaf4263 : Array AnnotatedEvent := #[
  { event := event68208
    frameStart := 68173 },
  { event := event68209
    frameStart := 68173 },
  { event := event68210
    frameStart := 68173 },
  { event := event68211
    frameStart := 68173 },
  { event := event68212
    frameStart := 68173 },
  { event := event68213
    frameStart := 68173 },
  { event := event68214
    frameStart := 68173 },
  { event := event68215
    frameStart := 68173 },
  { event := event68216
    frameStart := 68173 },
  { event := event68217
    frameStart := 68173 },
  { event := event68218
    frameStart := 68173 },
  { event := event68219
    frameStart := 68173 },
  { event := event68220
    frameStart := 68173 },
  { event := event68221
    frameStart := 68173 },
  { event := event68222
    frameStart := 68173 },
  { event := event68223
    frameStart := 68173 }
]

def eventLeaf4264 : Array AnnotatedEvent := #[
  { event := event68224
    frameStart := 68173 },
  { event := event68225
    frameStart := 68173 },
  { event := event68226
    frameStart := 68173 },
  { event := event68227
    frameStart := 68173 },
  { event := event68228
    frameStart := 68173 },
  { event := event68229
    frameStart := 68173 },
  { event := event68230
    frameStart := 68173 },
  { event := event68231
    frameStart := 68173 },
  { event := event68232
    frameStart := 68173 },
  { event := event68233
    frameStart := 68173 },
  { event := event68234
    frameStart := 68173 },
  { event := event68235
    frameStart := 68173 },
  { event := event68236
    frameStart := 68173 },
  { event := event68237
    frameStart := 68173 },
  { event := event68238
    frameStart := 68173 },
  { event := event68239
    frameStart := 68173 }
]

def eventLeaf4265 : Array AnnotatedEvent := #[
  { event := event68240
    frameStart := 68173 },
  { event := event68241
    frameStart := 68173 },
  { event := event68242
    frameStart := 68173 },
  { event := event68243
    frameStart := 68173 },
  { event := event68244
    frameStart := 68173 },
  { event := event68245
    frameStart := 68173 },
  { event := event68246
    frameStart := 68173 },
  { event := event68247
    frameStart := 68173 },
  { event := event68248
    frameStart := 68173 },
  { event := event68249
    frameStart := 68173 },
  { event := event68250
    frameStart := 68173 },
  { event := event68251
    frameStart := 68173 },
  { event := event68252
    frameStart := 68173 },
  { event := event68253
    frameStart := 68173 },
  { event := event68254
    frameStart := 68173 },
  { event := event68255
    frameStart := 68173 }
]

def eventLeaf4266 : Array AnnotatedEvent := #[
  { event := event68256
    frameStart := 68173 },
  { event := event68257
    frameStart := 68173 },
  { event := event68258
    frameStart := 68173 },
  { event := event68259
    frameStart := 68173 },
  { event := event68260
    frameStart := 68173 },
  { event := event68261
    frameStart := 68173 },
  { event := event68262
    frameStart := 68173 },
  { event := event68263
    frameStart := 68173 },
  { event := event68264
    frameStart := 68173 },
  { event := event68265
    frameStart := 68173 },
  { event := event68266
    frameStart := 68173 },
  { event := event68267
    frameStart := 68173 },
  { event := event68268
    frameStart := 68173 },
  { event := event68269
    frameStart := 68173 },
  { event := event68270
    frameStart := 68173 },
  { event := event68271
    frameStart := 68173 }
]

def eventLeaf4267 : Array AnnotatedEvent := #[
  { event := event68272
    frameStart := 68173 },
  { event := event68273
    frameStart := 68173 },
  { event := event68274
    frameStart := 68173 },
  { event := event68275
    frameStart := 68173 },
  { event := event68276
    frameStart := 68173 },
  { event := event68277
    frameStart := 68173 },
  { event := event68278
    frameStart := 68173 },
  { event := event68279
    frameStart := 68173 },
  { event := event68280
    frameStart := 68173 },
  { event := event68281
    frameStart := 68173 },
  { event := event68282
    frameStart := 68173 },
  { event := event68283
    frameStart := 68173 },
  { event := event68284
    frameStart := 68173 },
  { event := event68285
    frameStart := 68173 },
  { event := event68286
    frameStart := 68173 },
  { event := event68287
    frameStart := 68173 }
]

def eventLeaf4268 : Array AnnotatedEvent := #[
  { event := event68288
    frameStart := 68173 },
  { event := event68289
    frameStart := 68173 },
  { event := event68290
    frameStart := 68173 },
  { event := event68291
    frameStart := 0 },
  { event := event68292
    frameStart := 0 },
  { event := event68293
    frameStart := 0 },
  { event := event68294
    frameStart := 0 },
  { event := event68295
    frameStart := 0 },
  { event := event68296
    frameStart := 0 },
  { event := event68297
    frameStart := 0 },
  { event := event68298
    frameStart := 0 },
  { event := event68299
    frameStart := 0 },
  { event := event68300
    frameStart := 0 },
  { event := event68301
    frameStart := 0 },
  { event := event68302
    frameStart := 0 },
  { event := event68303
    frameStart := 0 }
]

def eventLeaf4269 : Array AnnotatedEvent := #[
  { event := event68304
    frameStart := 0 },
  { event := event68305
    frameStart := 0 },
  { event := event68306
    frameStart := 0 },
  { event := event68307
    frameStart := 0 },
  { event := event68308
    frameStart := 0 },
  { event := event68309
    frameStart := 0 },
  { event := event68310
    frameStart := 0 },
  { event := event68311
    frameStart := 0 },
  { event := event68312
    frameStart := 0 },
  { event := event68313
    frameStart := 0 },
  { event := event68314
    frameStart := 0 },
  { event := event68315
    frameStart := 0 },
  { event := event68316
    frameStart := 0 },
  { event := event68317
    frameStart := 0 },
  { event := event68318
    frameStart := 0 },
  { event := event68319
    frameStart := 0 }
]

def eventLeaf4270 : Array AnnotatedEvent := #[
  { event := event68320
    frameStart := 0 },
  { event := event68321
    frameStart := 0 },
  { event := event68322
    frameStart := 0 },
  { event := event68323
    frameStart := 0 },
  { event := event68324
    frameStart := 0 },
  { event := event68325
    frameStart := 0 },
  { event := event68326
    frameStart := 0 },
  { event := event68327
    frameStart := 0 },
  { event := event68328
    frameStart := 68328 },
  { event := event68329
    frameStart := 68328 },
  { event := event68330
    frameStart := 68328 },
  { event := event68331
    frameStart := 68328 },
  { event := event68332
    frameStart := 68328 },
  { event := event68333
    frameStart := 68328 },
  { event := event68334
    frameStart := 68328 },
  { event := event68335
    frameStart := 68328 }
]

def eventLeaf4271 : Array AnnotatedEvent := #[
  { event := event68336
    frameStart := 68328 },
  { event := event68337
    frameStart := 68328 },
  { event := event68338
    frameStart := 68328 },
  { event := event68339
    frameStart := 68328 },
  { event := event68340
    frameStart := 68328 },
  { event := event68341
    frameStart := 68328 },
  { event := event68342
    frameStart := 68328 },
  { event := event68343
    frameStart := 68328 },
  { event := event68344
    frameStart := 68328 },
  { event := event68345
    frameStart := 68328 },
  { event := event68346
    frameStart := 68328 },
  { event := event68347
    frameStart := 68328 },
  { event := event68348
    frameStart := 68328 },
  { event := event68349
    frameStart := 68328 },
  { event := event68350
    frameStart := 68328 },
  { event := event68351
    frameStart := 68328 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events266
