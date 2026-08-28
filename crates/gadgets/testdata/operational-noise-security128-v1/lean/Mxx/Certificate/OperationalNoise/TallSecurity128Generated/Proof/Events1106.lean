import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1106

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event283136 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36194⟩⟩, .relation 283135 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨35713⟩⟩]⟩, (-1)⟩)

def event283137 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36194⟩⟩, .operator (⟨283128, 0⟩, ⟨283064, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36193⟩⟩]⟩, (1)⟩)

def exact283138RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨35713⟩⟩]⟩, (-1)⟩]

theorem exact283138RawTermsValid :
    exact283138RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283138 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36194⟩⟩) exact283138RawTerms .large 283131 (.finite 2997961829447525990400) (some (283133))

def event283139 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35129⟩⟩) 0 ⟨34292⟩ 13678

def event283140 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35129⟩⟩) (.authority (.relationPreimageSource ⟨49⟩))

def exact283141RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35129⟩⟩]⟩, (1)⟩]

theorem exact283141RawTermsValid :
    exact283141RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283141 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35129⟩⟩) exact283141RawTerms (.finite 5647228698) 283140 .exactZero (none)

def event283142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35131⟩⟩) 0 ⟨35129⟩ 283141

def event283143 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35131⟩⟩) 1 ⟨2370⟩ 4

def event283144 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35131⟩⟩) (.scale (.predecessor 0 283142 .coefficient) (.value (.predecessor 1 283143 .coefficient)))

def exact283145RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35129⟩⟩]⟩, (1)⟩]

theorem exact283145RawTermsValid :
    exact283145RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283145 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35131⟩⟩) exact283145RawTerms (.finite 5647228698) 283144 .exactZero (none)

def event283146 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35132⟩⟩) 0 ⟨5491⟩ 280745

def event283147 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35132⟩⟩) 1 ⟨35131⟩ 283145

def event283148 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35132⟩⟩) (.product (.predecessor 0 283146 .coefficient) (.predecessor 1 283147 .coefficient) (⟨false, false, none, none, none⟩))

def event283149 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35132⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨35129⟩⟩]⟩) [⟨.result 283141 .coefficient, false, none⟩])

def event283150 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35132⟩⟩) (.product (.result 280745 .summary) (.transfer 283149) (⟨false, false, none, none, none⟩))

def event283151 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35132⟩⟩, .operator (⟨280745, 0⟩, ⟨283145, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35129⟩⟩]⟩, (1)⟩)

def event283152 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨35130⟩⟩)

def event283153 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event283154 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event283155 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event283156 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event283157 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event283158 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event283159 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event283160 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event283161 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 283160

def event283162 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 283158

def event283163 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 283161 .coefficient) (.value (.predecessor 1 283162 .coefficient)))

def event283164 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event283165 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 283164

def event283166 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 283156

def event283167 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 283165 .coefficient, .predecessor 1 283166 .coefficient])

def event283168 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event283169 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 283168

def event283170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 283154

def event283171 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 283170 .coefficient))

def event283172 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event283173 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34290⟩⟩) 0 ⟨5487⟩ 283172

def event283174 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34290⟩⟩) (.authority (.programFamilyFact))

def exact283175RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34290⟩⟩], []⟩, (1)⟩]

theorem exact283175RawTermsValid :
    exact283175RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283175 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34290⟩⟩) exact283175RawTerms (.finite 40) 283174 .exactZero (none)

def event283176 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13491⟩⟩) 0 ⟨5487⟩ 283172

def event283177 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13491⟩⟩) (.authority (.programFamilyFact))

def exact283178RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩], []⟩, (1)⟩]

theorem exact283178RawTermsValid :
    exact283178RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283178 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13491⟩⟩) exact283178RawTerms (.finite 40) 283177 .exactZero (none)

def event283179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34291⟩⟩) 0 ⟨13491⟩ 283178

def event283180 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34291⟩⟩) 1 ⟨34290⟩ 283175

def event283181 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34291⟩⟩) (.product (.predecessor 0 283179 .coefficient) (.predecessor 1 283180 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event283182 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34291⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], []⟩) [⟨.result 283178 .coefficient, true, some 1⟩, ⟨.result 283175 .coefficient, true, some 1⟩])

def event283183 : Event := .survivorFold (1) 283182

def exact283184RawTerms : List Term := []

theorem exact283184RawTermsValid :
    exact283184RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283184 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34291⟩⟩) exact283184RawTerms (.finite 1600) 283181 (.finite 1600) (some (283182))

def event283185 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34292⟩⟩) 0 ⟨34291⟩ 283184

def event283186 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34292⟩⟩) (.identity (.predecessor 0 283185 .coefficient))

def event283187 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34292⟩⟩) (.finite 1600)

def event283188 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35129⟩⟩) 0 ⟨34292⟩ 283187

def event283189 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35129⟩⟩) (.authority (.relationPreimageSource ⟨49⟩))

def exact283190RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35129⟩⟩]⟩, (1)⟩]

theorem exact283190RawTermsValid :
    exact283190RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283190 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35129⟩⟩) exact283190RawTerms (.finite 5647228698) 283189 .exactZero (none)

def event283191 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact283192RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact283192RawTermsValid :
    exact283192RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283192 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact283192RawTerms .large 283191 .exactZero (none)

def event283193 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35130⟩⟩) 0 ⟨35⟩ 283192

def event283194 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35130⟩⟩) 1 ⟨35129⟩ 283190

def event283195 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35130⟩⟩) (.product (.predecessor 0 283193 .coefficient) (.predecessor 1 283194 .coefficient) (⟨false, false, none, none, none⟩))

def event283196 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35130⟩⟩, .operator (⟨283192, 0⟩, ⟨283190, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35129⟩⟩]⟩, (1)⟩)

def exact283197RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35129⟩⟩]⟩, (1)⟩]

theorem exact283197RawTermsValid :
    exact283197RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283197 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35130⟩⟩) exact283197RawTerms .large 283195 .exactZero (none)

def event283198 : Event := .preFoldPolynomial 283197 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35129⟩⟩]⟩, (1)⟩] .exactZero none

def exact283199RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35129⟩⟩]⟩, (1)⟩]

def event283199 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨35130⟩⟩) 283198 exact283199RawTerms .large 283195 .exactZero (none)

def event283200 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨36197⟩⟩)

def event283201 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event283202 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event283203 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event283204 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event283205 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event283206 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event283207 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event283208 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event283209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 283208

def event283210 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 283206

def event283211 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 283209 .coefficient) (.value (.predecessor 1 283210 .coefficient)))

def event283212 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event283213 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 283212

def event283214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 283204

def event283215 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 283213 .coefficient, .predecessor 1 283214 .coefficient])

def event283216 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event283217 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 283216

def event283218 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 283202

def event283219 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 283218 .coefficient))

def event283220 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event283221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34290⟩⟩) 0 ⟨5487⟩ 283220

def event283222 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34290⟩⟩) (.authority (.programFamilyFact))

def exact283223RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34290⟩⟩], []⟩, (1)⟩]

theorem exact283223RawTermsValid :
    exact283223RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283223 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34290⟩⟩) exact283223RawTerms (.finite 40) 283222 .exactZero (none)

def event283224 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13491⟩⟩) 0 ⟨5487⟩ 283220

def event283225 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13491⟩⟩) (.authority (.programFamilyFact))

def exact283226RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩], []⟩, (1)⟩]

theorem exact283226RawTermsValid :
    exact283226RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283226 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13491⟩⟩) exact283226RawTerms (.finite 40) 283225 .exactZero (none)

def event283227 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34291⟩⟩) 0 ⟨13491⟩ 283226

def event283228 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34291⟩⟩) 1 ⟨34290⟩ 283223

def event283229 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34291⟩⟩) (.product (.predecessor 0 283227 .coefficient) (.predecessor 1 283228 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event283230 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34291⟩⟩, .operator (⟨283226, 0⟩, ⟨283223, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], []⟩, (1)⟩)

def exact283231RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], []⟩, (1)⟩]

theorem exact283231RawTermsValid :
    exact283231RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283231 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34291⟩⟩) exact283231RawTerms (.finite 1600) 283229 .exactZero (none)

def event283232 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34292⟩⟩) 0 ⟨34291⟩ 283231

def event283233 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34292⟩⟩) (.identity (.predecessor 0 283232 .coefficient))

def event283234 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34292⟩⟩) (.finite 1600)

def event283235 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35712⟩⟩) 0 ⟨34292⟩ 283234

def event283236 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35712⟩⟩) (.authority (.programFamilyFact))

def event283237 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨35712⟩⟩) (.finite 3720)

def event283238 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event283239 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35713⟩⟩) 0 ⟨7177⟩ 283238

def event283240 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35713⟩⟩) 1 ⟨35712⟩ 283237

def event283241 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35713⟩⟩) (.authority (.operator))

def exact283242RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35713⟩⟩]⟩, (1)⟩]

theorem exact283242RawTermsValid :
    exact283242RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283242 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35713⟩⟩) exact283242RawTerms .large 283241 .exactZero (none)

def event283243 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36193⟩⟩) 0 ⟨35713⟩ 283242

def event283244 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36193⟩⟩) (.authority (.operator))

def exact283245RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨36193⟩⟩]⟩, (1)⟩]

theorem exact283245RawTermsValid :
    exact283245RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283245 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36193⟩⟩) exact283245RawTerms (.finite 8192) 283244 .exactZero (none)

def event283246 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event283247 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event283248 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36002⟩⟩) 0 ⟨34292⟩ 283234

def event283249 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36002⟩⟩) 1 ⟨136⟩ 283247

def event283250 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36002⟩⟩) (.sum [.predecessor 0 283248 .coefficient, .predecessor 1 283249 .coefficient])

def event283251 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨36002⟩⟩) (.finite 1600)

def event283252 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36003⟩⟩) 0 ⟨36002⟩ 283251

def event283253 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36003⟩⟩) (.identity (.predecessor 0 283252 .coefficient))

def exact283254RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], []⟩, (1)⟩]

theorem exact283254RawTermsValid :
    exact283254RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283254 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36003⟩⟩) exact283254RawTerms (.finite 1600) 283253 .exactZero (none)

def event283255 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact283256RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact283256RawTermsValid :
    exact283256RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283256 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact283256RawTerms .large 283255 .exactZero (none)

def event283257 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36004⟩⟩) 0 ⟨6908⟩ 283256

def event283258 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36004⟩⟩) 1 ⟨36003⟩ 283254

def event283259 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36004⟩⟩) (.product (.predecessor 0 283257 .coefficient) (.predecessor 1 283258 .coefficient) (⟨false, false, none, none, none⟩))

def event283260 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36004⟩⟩, .operator (⟨283256, 0⟩, ⟨283254, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact283261RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact283261RawTermsValid :
    exact283261RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283261 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36004⟩⟩) exact283261RawTerms .large 283259 .exactZero (none)

def event283262 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 283238

def event283263 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact283264RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact283264RawTermsValid :
    exact283264RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283264 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact283264RawTerms .large 283263 .exactZero (none)

def event283265 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7280⟩⟩) 0 ⟨7178⟩ 283264

def event283266 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7280⟩⟩) (.identity (.predecessor 0 283265 .coefficient))

def exact283267RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (1)⟩]

theorem exact283267RawTermsValid :
    exact283267RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283267 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7280⟩⟩) exact283267RawTerms .large 283266 .exactZero (none)

def event283268 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9550⟩⟩) 0 ⟨7280⟩ 283267

def event283269 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9550⟩⟩) (.authority (.operator))

def exact283270RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩, (1)⟩]

theorem exact283270RawTermsValid :
    exact283270RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283270 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9550⟩⟩) exact283270RawTerms (.finite 8192) 283269 .exactZero (none)

def event283271 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9551⟩⟩) 0 ⟨9550⟩ 283270

def event283272 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9551⟩⟩) 1 ⟨2370⟩ 283204

def event283273 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9551⟩⟩) (.scale (.predecessor 0 283271 .coefficient) (.value (.predecessor 1 283272 .coefficient)))

def exact283274RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩, (1)⟩]

theorem exact283274RawTermsValid :
    exact283274RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283274 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9551⟩⟩) exact283274RawTerms (.finite 8192) 283273 .exactZero (none)

def event283275 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7297⟩⟩) 0 ⟨7178⟩ 283264

def event283276 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7297⟩⟩) (.identity (.predecessor 0 283275 .coefficient))

def exact283277RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩]⟩, (1)⟩]

theorem exact283277RawTermsValid :
    exact283277RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283277 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7297⟩⟩) exact283277RawTerms .large 283276 .exactZero (none)

def event283278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9552⟩⟩) 0 ⟨7297⟩ 283277

def event283279 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9552⟩⟩) 1 ⟨9551⟩ 283274

def event283280 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9552⟩⟩) (.product (.predecessor 0 283278 .coefficient) (.predecessor 1 283279 .coefficient) (⟨false, false, none, none, none⟩))

def event283281 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9552⟩⟩, .operator (⟨283277, 0⟩, ⟨283274, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩, (1)⟩)

def exact283282RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩, (1)⟩]

theorem exact283282RawTermsValid :
    exact283282RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283282 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9552⟩⟩) exact283282RawTerms .large 283280 .exactZero (none)

def event283283 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36005⟩⟩) 0 ⟨9552⟩ 283282

def event283284 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36005⟩⟩) 1 ⟨36004⟩ 283261

def event283285 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36005⟩⟩) (.sum [.predecessor 0 283283 .coefficient, .predecessor 1 283284 .coefficient])

def exact283286RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283286RawTermsValid :
    exact283286RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283286 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36005⟩⟩) exact283286RawTerms .large 283285 .exactZero (none)

def event283287 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36196⟩⟩) 0 ⟨36005⟩ 283286

def event283288 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36196⟩⟩) 1 ⟨36193⟩ 283245

def event283289 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36196⟩⟩) (.product (.predecessor 0 283287 .coefficient) (.predecessor 1 283288 .coefficient) (⟨false, false, none, none, none⟩))

def event283290 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36196⟩⟩, .operator (⟨283286, 0⟩, ⟨283245, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36193⟩⟩]⟩, (1)⟩)

def event283291 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36196⟩⟩, .operator (⟨283286, 1⟩, ⟨283245, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36193⟩⟩]⟩, (-1)⟩)

def event283292 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36196⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36193⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36193⟩⟩) ⟨35713⟩ 283242)

def event283293 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36196⟩⟩, .relation 283292 0, ⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨35713⟩⟩]⟩, (-1)⟩)

def exact283294RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨35713⟩⟩]⟩, (-1)⟩]

theorem exact283294RawTermsValid :
    exact283294RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283294 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36196⟩⟩) exact283294RawTerms .large 283289 .exactZero (none)

def event283295 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34700⟩⟩) 0 ⟨34292⟩ 283234

def event283296 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34700⟩⟩) (.authority (.programFamilyFact))

def exact283297RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34700⟩⟩], []⟩, (1)⟩]

theorem exact283297RawTermsValid :
    exact283297RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283297 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34700⟩⟩) exact283297RawTerms (.finite 40) 283296 .exactZero (none)

def event283298 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34702⟩⟩) 0 ⟨6908⟩ 283256

def event283299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34702⟩⟩) 1 ⟨34700⟩ 283297

def event283300 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34702⟩⟩) (.product (.predecessor 0 283298 .coefficient) (.predecessor 1 283299 .coefficient) (⟨false, true, none, none, some 1⟩))

def event283301 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34702⟩⟩, .operator (⟨283256, 0⟩, ⟨283297, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨34700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact283302RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact283302RawTermsValid :
    exact283302RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283302 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34702⟩⟩) exact283302RawTerms .large 283300 .exactZero (none)

def event283303 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7191⟩⟩) 0 ⟨7177⟩ 283238

def event283304 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7191⟩⟩) (.authority (.operator))

def exact283305RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩]

theorem exact283305RawTermsValid :
    exact283305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283305 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7191⟩⟩) exact283305RawTerms .large 283304 .exactZero (none)

def event283306 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34703⟩⟩) 0 ⟨7191⟩ 283305

def event283307 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34703⟩⟩) 1 ⟨34702⟩ 283302

def event283308 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34703⟩⟩) (.sum [.predecessor 0 283306 .coefficient, .predecessor 1 283307 .coefficient])

def exact283309RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283309RawTermsValid :
    exact283309RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283309 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34703⟩⟩) exact283309RawTerms .large 283308 .exactZero (none)

def event283310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36197⟩⟩) 0 ⟨34703⟩ 283309

def event283311 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36197⟩⟩) 1 ⟨36196⟩ 283294

def event283312 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36197⟩⟩) (.sum [.predecessor 0 283310 .coefficient, .predecessor 1 283311 .coefficient])

def exact283313RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36193⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨35713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283313RawTermsValid :
    exact283313RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283313 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36197⟩⟩) exact283313RawTerms .large 283312 .exactZero (none)

def event283314 : Event := .preFoldPolynomial 283313 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36193⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨35713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact283315RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36193⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨35713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event283315 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨36197⟩⟩) 283314 exact283315RawTerms .large 283312 .exactZero (none)

def event283316 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨34292⟩⟩) ⟨⟨70⟩, ⟨49⟩, ⟨135⟩⟩ ⟨283152, 283316⟩

def event283317 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨35132⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35129⟩⟩]⟩) (1) 0 2 (.universal 283316 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35129⟩⟩]⟩) (none) 283315)

def event283318 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35132⟩⟩, .relation 283317 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩)

def event283319 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35132⟩⟩, .relation 283317 1, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36193⟩⟩]⟩, (-1)⟩)

def event283320 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35132⟩⟩, .relation 283317 2, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨35713⟩⟩]⟩, (1)⟩)

def event283321 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35132⟩⟩, .relation 283317 3, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact283322RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36193⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨35713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283322RawTermsValid :
    exact283322RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283322 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35132⟩⟩) exact283322RawTerms .large 283148 (.finite 202072841853861888) (some (283150))

def event283323 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36195⟩⟩) 0 ⟨35132⟩ 283322

def event283324 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36195⟩⟩) 1 ⟨36194⟩ 283138

def event283325 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36195⟩⟩) (.sum [.predecessor 0 283323 .coefficient, .predecessor 1 283324 .coefficient])

def event283326 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36195⟩⟩, .operator (⟨283322, 2⟩, ⟨283138, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨35713⟩⟩]⟩, (-1)⟩)

def event283327 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36195⟩⟩, .operator (⟨283322, 1⟩, ⟨283138, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36193⟩⟩]⟩, (1)⟩)

def event283328 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36195⟩⟩) (.sum [.result 283322 .summary, .result 283138 .summary])

def exact283329RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283329RawTermsValid :
    exact283329RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283329 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36195⟩⟩) exact283329RawTerms .large 283325 (.finite 2998163902289379852288) (some (283328))

def event283330 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36481⟩⟩) 0 ⟨36195⟩ 283329

def event283331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36481⟩⟩) 1 ⟨36479⟩ 283054

def event283332 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36481⟩⟩) (.product (.predecessor 0 283330 .coefficient) (.predecessor 1 283331 .coefficient) (⟨false, false, none, none, none⟩))

def event283333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36481⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨36479⟩⟩]⟩) [⟨.result 283054 .coefficient, false, none⟩])

def event283334 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36481⟩⟩) (.product (.result 283329 .summary) (.transfer 283333) (⟨false, false, none, none, none⟩))

def event283335 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36481⟩⟩, .operator (⟨283329, 0⟩, ⟨283054, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36479⟩⟩]⟩, (1)⟩)

def event283336 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36481⟩⟩, .operator (⟨283329, 1⟩, ⟨283054, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36479⟩⟩]⟩, (-1)⟩)

def event283337 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36481⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36479⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36479⟩⟩) ⟨35847⟩ 283051)

def event283338 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36481⟩⟩, .relation 283337 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34700⟩⟩], [⟨.program ⟨257⟩, ⟨35847⟩⟩]⟩, (-1)⟩)

def exact283339RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36479⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34700⟩⟩], [⟨.program ⟨257⟩, ⟨35847⟩⟩]⟩, (-1)⟩]

theorem exact283339RawTermsValid :
    exact283339RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283339 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36481⟩⟩) exact283339RawTerms .large 283332 (.finite 32192539770951564984245676933120) (some (283334))

def event283340 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35376⟩⟩) 0 ⟨34701⟩ 13684

def event283341 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35376⟩⟩) (.authority (.relationPreimageSource ⟨83⟩))

def exact283342RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35376⟩⟩]⟩, (1)⟩]

theorem exact283342RawTermsValid :
    exact283342RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283342 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35376⟩⟩) exact283342RawTerms (.finite 5647228698) 283341 .exactZero (none)

def event283343 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35378⟩⟩) 0 ⟨35376⟩ 283342

def event283344 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35378⟩⟩) 1 ⟨2370⟩ 4

def event283345 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35378⟩⟩) (.scale (.predecessor 0 283343 .coefficient) (.value (.predecessor 1 283344 .coefficient)))

def exact283346RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35376⟩⟩]⟩, (1)⟩]

theorem exact283346RawTermsValid :
    exact283346RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283346 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35378⟩⟩) exact283346RawTerms (.finite 5647228698) 283345 .exactZero (none)

def event283347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35379⟩⟩) 0 ⟨5491⟩ 280745

def event283348 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35379⟩⟩) 1 ⟨35378⟩ 283346

def event283349 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35379⟩⟩) (.product (.predecessor 0 283347 .coefficient) (.predecessor 1 283348 .coefficient) (⟨false, false, none, none, none⟩))

def event283350 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35379⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨35376⟩⟩]⟩) [⟨.result 283342 .coefficient, false, none⟩])

def event283351 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35379⟩⟩) (.product (.result 280745 .summary) (.transfer 283350) (⟨false, false, none, none, none⟩))

def event283352 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35379⟩⟩, .operator (⟨280745, 0⟩, ⟨283346, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35376⟩⟩]⟩, (1)⟩)

def event283353 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨35377⟩⟩)

def event283354 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event283355 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event283356 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event283357 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event283358 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event283359 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event283360 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event283361 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event283362 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 283361

def event283363 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 283359

def event283364 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 283362 .coefficient) (.value (.predecessor 1 283363 .coefficient)))

def event283365 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event283366 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 283365

def event283367 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 283357

def event283368 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 283366 .coefficient, .predecessor 1 283367 .coefficient])

def event283369 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event283370 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 283369

def event283371 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 283355

def event283372 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 283371 .coefficient))

def event283373 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event283374 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34290⟩⟩) 0 ⟨5487⟩ 283373

def event283375 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34290⟩⟩) (.authority (.programFamilyFact))

def exact283376RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34290⟩⟩], []⟩, (1)⟩]

theorem exact283376RawTermsValid :
    exact283376RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283376 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34290⟩⟩) exact283376RawTerms (.finite 40) 283375 .exactZero (none)

def event283377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13491⟩⟩) 0 ⟨5487⟩ 283373

def event283378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13491⟩⟩) (.authority (.programFamilyFact))

def exact283379RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩], []⟩, (1)⟩]

theorem exact283379RawTermsValid :
    exact283379RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283379 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13491⟩⟩) exact283379RawTerms (.finite 40) 283378 .exactZero (none)

def event283380 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34291⟩⟩) 0 ⟨13491⟩ 283379

def event283381 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34291⟩⟩) 1 ⟨34290⟩ 283376

def event283382 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34291⟩⟩) (.product (.predecessor 0 283380 .coefficient) (.predecessor 1 283381 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event283383 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34291⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], []⟩) [⟨.result 283379 .coefficient, true, some 1⟩, ⟨.result 283376 .coefficient, true, some 1⟩])

def event283384 : Event := .survivorFold (1) 283383

def exact283385RawTerms : List Term := []

theorem exact283385RawTermsValid :
    exact283385RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283385 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34291⟩⟩) exact283385RawTerms (.finite 1600) 283382 (.finite 1600) (some (283383))

def event283386 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34292⟩⟩) 0 ⟨34291⟩ 283385

def event283387 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34292⟩⟩) (.identity (.predecessor 0 283386 .coefficient))

def event283388 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34292⟩⟩) (.finite 1600)

def event283389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34700⟩⟩) 0 ⟨34292⟩ 283388

def event283390 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34700⟩⟩) (.authority (.programFamilyFact))

def exact283391RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34700⟩⟩], []⟩, (1)⟩]

theorem exact283391RawTermsValid :
    exact283391RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283391 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34700⟩⟩) exact283391RawTerms (.finite 40) 283390 .exactZero (none)

def eventLeaf17696 : Array AnnotatedEvent := #[
  { event := event283136
    frameStart := 0 },
  { event := event283137
    frameStart := 0 },
  { event := event283138
    frameStart := 0 },
  { event := event283139
    frameStart := 0 },
  { event := event283140
    frameStart := 0 },
  { event := event283141
    frameStart := 0 },
  { event := event283142
    frameStart := 0 },
  { event := event283143
    frameStart := 0 },
  { event := event283144
    frameStart := 0 },
  { event := event283145
    frameStart := 0 },
  { event := event283146
    frameStart := 0 },
  { event := event283147
    frameStart := 0 },
  { event := event283148
    frameStart := 0 },
  { event := event283149
    frameStart := 0 },
  { event := event283150
    frameStart := 0 },
  { event := event283151
    frameStart := 0 }
]

def eventLeaf17697 : Array AnnotatedEvent := #[
  { event := event283152
    frameStart := 283152 },
  { event := event283153
    frameStart := 283152 },
  { event := event283154
    frameStart := 283152 },
  { event := event283155
    frameStart := 283152 },
  { event := event283156
    frameStart := 283152 },
  { event := event283157
    frameStart := 283152 },
  { event := event283158
    frameStart := 283152 },
  { event := event283159
    frameStart := 283152 },
  { event := event283160
    frameStart := 283152 },
  { event := event283161
    frameStart := 283152 },
  { event := event283162
    frameStart := 283152 },
  { event := event283163
    frameStart := 283152 },
  { event := event283164
    frameStart := 283152 },
  { event := event283165
    frameStart := 283152 },
  { event := event283166
    frameStart := 283152 },
  { event := event283167
    frameStart := 283152 }
]

def eventLeaf17698 : Array AnnotatedEvent := #[
  { event := event283168
    frameStart := 283152 },
  { event := event283169
    frameStart := 283152 },
  { event := event283170
    frameStart := 283152 },
  { event := event283171
    frameStart := 283152 },
  { event := event283172
    frameStart := 283152 },
  { event := event283173
    frameStart := 283152 },
  { event := event283174
    frameStart := 283152 },
  { event := event283175
    frameStart := 283152 },
  { event := event283176
    frameStart := 283152 },
  { event := event283177
    frameStart := 283152 },
  { event := event283178
    frameStart := 283152 },
  { event := event283179
    frameStart := 283152 },
  { event := event283180
    frameStart := 283152 },
  { event := event283181
    frameStart := 283152 },
  { event := event283182
    frameStart := 283152 },
  { event := event283183
    frameStart := 283152 }
]

def eventLeaf17699 : Array AnnotatedEvent := #[
  { event := event283184
    frameStart := 283152 },
  { event := event283185
    frameStart := 283152 },
  { event := event283186
    frameStart := 283152 },
  { event := event283187
    frameStart := 283152 },
  { event := event283188
    frameStart := 283152 },
  { event := event283189
    frameStart := 283152 },
  { event := event283190
    frameStart := 283152 },
  { event := event283191
    frameStart := 283152 },
  { event := event283192
    frameStart := 283152 },
  { event := event283193
    frameStart := 283152 },
  { event := event283194
    frameStart := 283152 },
  { event := event283195
    frameStart := 283152 },
  { event := event283196
    frameStart := 283152 },
  { event := event283197
    frameStart := 283152 },
  { event := event283198
    frameStart := 283152 },
  { event := event283199
    frameStart := 283152 }
]

def eventLeaf17700 : Array AnnotatedEvent := #[
  { event := event283200
    frameStart := 283200 },
  { event := event283201
    frameStart := 283200 },
  { event := event283202
    frameStart := 283200 },
  { event := event283203
    frameStart := 283200 },
  { event := event283204
    frameStart := 283200 },
  { event := event283205
    frameStart := 283200 },
  { event := event283206
    frameStart := 283200 },
  { event := event283207
    frameStart := 283200 },
  { event := event283208
    frameStart := 283200 },
  { event := event283209
    frameStart := 283200 },
  { event := event283210
    frameStart := 283200 },
  { event := event283211
    frameStart := 283200 },
  { event := event283212
    frameStart := 283200 },
  { event := event283213
    frameStart := 283200 },
  { event := event283214
    frameStart := 283200 },
  { event := event283215
    frameStart := 283200 }
]

def eventLeaf17701 : Array AnnotatedEvent := #[
  { event := event283216
    frameStart := 283200 },
  { event := event283217
    frameStart := 283200 },
  { event := event283218
    frameStart := 283200 },
  { event := event283219
    frameStart := 283200 },
  { event := event283220
    frameStart := 283200 },
  { event := event283221
    frameStart := 283200 },
  { event := event283222
    frameStart := 283200 },
  { event := event283223
    frameStart := 283200 },
  { event := event283224
    frameStart := 283200 },
  { event := event283225
    frameStart := 283200 },
  { event := event283226
    frameStart := 283200 },
  { event := event283227
    frameStart := 283200 },
  { event := event283228
    frameStart := 283200 },
  { event := event283229
    frameStart := 283200 },
  { event := event283230
    frameStart := 283200 },
  { event := event283231
    frameStart := 283200 }
]

def eventLeaf17702 : Array AnnotatedEvent := #[
  { event := event283232
    frameStart := 283200 },
  { event := event283233
    frameStart := 283200 },
  { event := event283234
    frameStart := 283200 },
  { event := event283235
    frameStart := 283200 },
  { event := event283236
    frameStart := 283200 },
  { event := event283237
    frameStart := 283200 },
  { event := event283238
    frameStart := 283200 },
  { event := event283239
    frameStart := 283200 },
  { event := event283240
    frameStart := 283200 },
  { event := event283241
    frameStart := 283200 },
  { event := event283242
    frameStart := 283200 },
  { event := event283243
    frameStart := 283200 },
  { event := event283244
    frameStart := 283200 },
  { event := event283245
    frameStart := 283200 },
  { event := event283246
    frameStart := 283200 },
  { event := event283247
    frameStart := 283200 }
]

def eventLeaf17703 : Array AnnotatedEvent := #[
  { event := event283248
    frameStart := 283200 },
  { event := event283249
    frameStart := 283200 },
  { event := event283250
    frameStart := 283200 },
  { event := event283251
    frameStart := 283200 },
  { event := event283252
    frameStart := 283200 },
  { event := event283253
    frameStart := 283200 },
  { event := event283254
    frameStart := 283200 },
  { event := event283255
    frameStart := 283200 },
  { event := event283256
    frameStart := 283200 },
  { event := event283257
    frameStart := 283200 },
  { event := event283258
    frameStart := 283200 },
  { event := event283259
    frameStart := 283200 },
  { event := event283260
    frameStart := 283200 },
  { event := event283261
    frameStart := 283200 },
  { event := event283262
    frameStart := 283200 },
  { event := event283263
    frameStart := 283200 }
]

def eventLeaf17704 : Array AnnotatedEvent := #[
  { event := event283264
    frameStart := 283200 },
  { event := event283265
    frameStart := 283200 },
  { event := event283266
    frameStart := 283200 },
  { event := event283267
    frameStart := 283200 },
  { event := event283268
    frameStart := 283200 },
  { event := event283269
    frameStart := 283200 },
  { event := event283270
    frameStart := 283200 },
  { event := event283271
    frameStart := 283200 },
  { event := event283272
    frameStart := 283200 },
  { event := event283273
    frameStart := 283200 },
  { event := event283274
    frameStart := 283200 },
  { event := event283275
    frameStart := 283200 },
  { event := event283276
    frameStart := 283200 },
  { event := event283277
    frameStart := 283200 },
  { event := event283278
    frameStart := 283200 },
  { event := event283279
    frameStart := 283200 }
]

def eventLeaf17705 : Array AnnotatedEvent := #[
  { event := event283280
    frameStart := 283200 },
  { event := event283281
    frameStart := 283200 },
  { event := event283282
    frameStart := 283200 },
  { event := event283283
    frameStart := 283200 },
  { event := event283284
    frameStart := 283200 },
  { event := event283285
    frameStart := 283200 },
  { event := event283286
    frameStart := 283200 },
  { event := event283287
    frameStart := 283200 },
  { event := event283288
    frameStart := 283200 },
  { event := event283289
    frameStart := 283200 },
  { event := event283290
    frameStart := 283200 },
  { event := event283291
    frameStart := 283200 },
  { event := event283292
    frameStart := 283200 },
  { event := event283293
    frameStart := 283200 },
  { event := event283294
    frameStart := 283200 },
  { event := event283295
    frameStart := 283200 }
]

def eventLeaf17706 : Array AnnotatedEvent := #[
  { event := event283296
    frameStart := 283200 },
  { event := event283297
    frameStart := 283200 },
  { event := event283298
    frameStart := 283200 },
  { event := event283299
    frameStart := 283200 },
  { event := event283300
    frameStart := 283200 },
  { event := event283301
    frameStart := 283200 },
  { event := event283302
    frameStart := 283200 },
  { event := event283303
    frameStart := 283200 },
  { event := event283304
    frameStart := 283200 },
  { event := event283305
    frameStart := 283200 },
  { event := event283306
    frameStart := 283200 },
  { event := event283307
    frameStart := 283200 },
  { event := event283308
    frameStart := 283200 },
  { event := event283309
    frameStart := 283200 },
  { event := event283310
    frameStart := 283200 },
  { event := event283311
    frameStart := 283200 }
]

def eventLeaf17707 : Array AnnotatedEvent := #[
  { event := event283312
    frameStart := 283200 },
  { event := event283313
    frameStart := 283200 },
  { event := event283314
    frameStart := 283200 },
  { event := event283315
    frameStart := 283200 },
  { event := event283316
    frameStart := 0 },
  { event := event283317
    frameStart := 0 },
  { event := event283318
    frameStart := 0 },
  { event := event283319
    frameStart := 0 },
  { event := event283320
    frameStart := 0 },
  { event := event283321
    frameStart := 0 },
  { event := event283322
    frameStart := 0 },
  { event := event283323
    frameStart := 0 },
  { event := event283324
    frameStart := 0 },
  { event := event283325
    frameStart := 0 },
  { event := event283326
    frameStart := 0 },
  { event := event283327
    frameStart := 0 }
]

def eventLeaf17708 : Array AnnotatedEvent := #[
  { event := event283328
    frameStart := 0 },
  { event := event283329
    frameStart := 0 },
  { event := event283330
    frameStart := 0 },
  { event := event283331
    frameStart := 0 },
  { event := event283332
    frameStart := 0 },
  { event := event283333
    frameStart := 0 },
  { event := event283334
    frameStart := 0 },
  { event := event283335
    frameStart := 0 },
  { event := event283336
    frameStart := 0 },
  { event := event283337
    frameStart := 0 },
  { event := event283338
    frameStart := 0 },
  { event := event283339
    frameStart := 0 },
  { event := event283340
    frameStart := 0 },
  { event := event283341
    frameStart := 0 },
  { event := event283342
    frameStart := 0 },
  { event := event283343
    frameStart := 0 }
]

def eventLeaf17709 : Array AnnotatedEvent := #[
  { event := event283344
    frameStart := 0 },
  { event := event283345
    frameStart := 0 },
  { event := event283346
    frameStart := 0 },
  { event := event283347
    frameStart := 0 },
  { event := event283348
    frameStart := 0 },
  { event := event283349
    frameStart := 0 },
  { event := event283350
    frameStart := 0 },
  { event := event283351
    frameStart := 0 },
  { event := event283352
    frameStart := 0 },
  { event := event283353
    frameStart := 283353 },
  { event := event283354
    frameStart := 283353 },
  { event := event283355
    frameStart := 283353 },
  { event := event283356
    frameStart := 283353 },
  { event := event283357
    frameStart := 283353 },
  { event := event283358
    frameStart := 283353 },
  { event := event283359
    frameStart := 283353 }
]

def eventLeaf17710 : Array AnnotatedEvent := #[
  { event := event283360
    frameStart := 283353 },
  { event := event283361
    frameStart := 283353 },
  { event := event283362
    frameStart := 283353 },
  { event := event283363
    frameStart := 283353 },
  { event := event283364
    frameStart := 283353 },
  { event := event283365
    frameStart := 283353 },
  { event := event283366
    frameStart := 283353 },
  { event := event283367
    frameStart := 283353 },
  { event := event283368
    frameStart := 283353 },
  { event := event283369
    frameStart := 283353 },
  { event := event283370
    frameStart := 283353 },
  { event := event283371
    frameStart := 283353 },
  { event := event283372
    frameStart := 283353 },
  { event := event283373
    frameStart := 283353 },
  { event := event283374
    frameStart := 283353 },
  { event := event283375
    frameStart := 283353 }
]

def eventLeaf17711 : Array AnnotatedEvent := #[
  { event := event283376
    frameStart := 283353 },
  { event := event283377
    frameStart := 283353 },
  { event := event283378
    frameStart := 283353 },
  { event := event283379
    frameStart := 283353 },
  { event := event283380
    frameStart := 283353 },
  { event := event283381
    frameStart := 283353 },
  { event := event283382
    frameStart := 283353 },
  { event := event283383
    frameStart := 283353 },
  { event := event283384
    frameStart := 283353 },
  { event := event283385
    frameStart := 283353 },
  { event := event283386
    frameStart := 283353 },
  { event := event283387
    frameStart := 283353 },
  { event := event283388
    frameStart := 283353 },
  { event := event283389
    frameStart := 283353 },
  { event := event283390
    frameStart := 283353 },
  { event := event283391
    frameStart := 283353 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1106
