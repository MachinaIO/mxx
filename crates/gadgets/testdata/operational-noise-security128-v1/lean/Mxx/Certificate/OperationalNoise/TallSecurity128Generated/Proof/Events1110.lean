import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1110

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event284160 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨27857⟩⟩)

def event284161 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event284162 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event284163 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event284164 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event284165 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event284166 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event284167 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event284168 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event284169 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 284168

def event284170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 284166

def event284171 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 284169 .coefficient) (.value (.predecessor 1 284170 .coefficient)))

def event284172 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event284173 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 284172

def event284174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 284164

def event284175 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 284173 .coefficient, .predecessor 1 284174 .coefficient])

def event284176 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event284177 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 284176

def event284178 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 284162

def event284179 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 284178 .coefficient))

def event284180 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event284181 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25950⟩⟩) 0 ⟨5487⟩ 284180

def event284182 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25950⟩⟩) (.authority (.programFamilyFact))

def exact284183RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25950⟩⟩], []⟩, (1)⟩]

theorem exact284183RawTermsValid :
    exact284183RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284183 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25950⟩⟩) exact284183RawTerms (.finite 30) 284182 .exactZero (none)

def event284184 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12891⟩⟩) 0 ⟨5487⟩ 284180

def event284185 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12891⟩⟩) (.authority (.programFamilyFact))

def exact284186RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩], []⟩, (1)⟩]

theorem exact284186RawTermsValid :
    exact284186RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284186 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12891⟩⟩) exact284186RawTerms (.finite 30) 284185 .exactZero (none)

def event284187 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25951⟩⟩) 0 ⟨12891⟩ 284186

def event284188 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25951⟩⟩) 1 ⟨25950⟩ 284183

def event284189 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25951⟩⟩) (.product (.predecessor 0 284187 .coefficient) (.predecessor 1 284188 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event284190 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨25951⟩⟩, .operator (⟨284186, 0⟩, ⟨284183, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], []⟩, (1)⟩)

def exact284191RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], []⟩, (1)⟩]

theorem exact284191RawTermsValid :
    exact284191RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284191 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25951⟩⟩) exact284191RawTerms (.finite 900) 284189 .exactZero (none)

def event284192 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25952⟩⟩) 0 ⟨25951⟩ 284191

def event284193 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25952⟩⟩) (.identity (.predecessor 0 284192 .coefficient))

def event284194 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨25952⟩⟩) (.finite 900)

def event284195 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27372⟩⟩) 0 ⟨25952⟩ 284194

def event284196 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27372⟩⟩) (.authority (.programFamilyFact))

def event284197 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27372⟩⟩) (.finite 3720)

def event284198 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event284199 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27373⟩⟩) 0 ⟨7177⟩ 284198

def event284200 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27373⟩⟩) 1 ⟨27372⟩ 284197

def event284201 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27373⟩⟩) (.authority (.operator))

def exact284202RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27373⟩⟩]⟩, (1)⟩]

theorem exact284202RawTermsValid :
    exact284202RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284202 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27373⟩⟩) exact284202RawTerms .large 284201 .exactZero (none)

def event284203 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27853⟩⟩) 0 ⟨27373⟩ 284202

def event284204 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27853⟩⟩) (.authority (.operator))

def exact284205RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27853⟩⟩]⟩, (1)⟩]

theorem exact284205RawTermsValid :
    exact284205RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284205 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27853⟩⟩) exact284205RawTerms (.finite 8192) 284204 .exactZero (none)

def event284206 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event284207 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event284208 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27662⟩⟩) 0 ⟨25952⟩ 284194

def event284209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27662⟩⟩) 1 ⟨136⟩ 284207

def event284210 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27662⟩⟩) (.sum [.predecessor 0 284208 .coefficient, .predecessor 1 284209 .coefficient])

def event284211 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27662⟩⟩) (.finite 900)

def event284212 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27663⟩⟩) 0 ⟨27662⟩ 284211

def event284213 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27663⟩⟩) (.identity (.predecessor 0 284212 .coefficient))

def exact284214RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], []⟩, (1)⟩]

theorem exact284214RawTermsValid :
    exact284214RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284214 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27663⟩⟩) exact284214RawTerms (.finite 900) 284213 .exactZero (none)

def event284215 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact284216RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact284216RawTermsValid :
    exact284216RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284216 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact284216RawTerms .large 284215 .exactZero (none)

def event284217 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27664⟩⟩) 0 ⟨6908⟩ 284216

def event284218 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27664⟩⟩) 1 ⟨27663⟩ 284214

def event284219 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27664⟩⟩) (.product (.predecessor 0 284217 .coefficient) (.predecessor 1 284218 .coefficient) (⟨false, false, none, none, none⟩))

def event284220 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27664⟩⟩, .operator (⟨284216, 0⟩, ⟨284214, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact284221RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact284221RawTermsValid :
    exact284221RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284221 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27664⟩⟩) exact284221RawTerms .large 284219 .exactZero (none)

def event284222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 284198

def event284223 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact284224RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact284224RawTermsValid :
    exact284224RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284224 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact284224RawTerms .large 284223 .exactZero (none)

def event284225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7278⟩⟩) 0 ⟨7178⟩ 284224

def event284226 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7278⟩⟩) (.identity (.predecessor 0 284225 .coefficient))

def exact284227RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩, (1)⟩]

theorem exact284227RawTermsValid :
    exact284227RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284227 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7278⟩⟩) exact284227RawTerms .large 284226 .exactZero (none)

def event284228 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9544⟩⟩) 0 ⟨7278⟩ 284227

def event284229 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9544⟩⟩) (.authority (.operator))

def exact284230RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩, (1)⟩]

theorem exact284230RawTermsValid :
    exact284230RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284230 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9544⟩⟩) exact284230RawTerms (.finite 8192) 284229 .exactZero (none)

def event284231 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9545⟩⟩) 0 ⟨9544⟩ 284230

def event284232 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9545⟩⟩) 1 ⟨2370⟩ 284164

def event284233 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9545⟩⟩) (.scale (.predecessor 0 284231 .coefficient) (.value (.predecessor 1 284232 .coefficient)))

def exact284234RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩, (1)⟩]

theorem exact284234RawTermsValid :
    exact284234RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284234 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9545⟩⟩) exact284234RawTerms (.finite 8192) 284233 .exactZero (none)

def event284235 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7295⟩⟩) 0 ⟨7178⟩ 284224

def event284236 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7295⟩⟩) (.identity (.predecessor 0 284235 .coefficient))

def exact284237RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩]⟩, (1)⟩]

theorem exact284237RawTermsValid :
    exact284237RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284237 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7295⟩⟩) exact284237RawTerms .large 284236 .exactZero (none)

def event284238 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9546⟩⟩) 0 ⟨7295⟩ 284237

def event284239 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9546⟩⟩) 1 ⟨9545⟩ 284234

def event284240 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9546⟩⟩) (.product (.predecessor 0 284238 .coefficient) (.predecessor 1 284239 .coefficient) (⟨false, false, none, none, none⟩))

def event284241 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9546⟩⟩, .operator (⟨284237, 0⟩, ⟨284234, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩, (1)⟩)

def exact284242RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩, (1)⟩]

theorem exact284242RawTermsValid :
    exact284242RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284242 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9546⟩⟩) exact284242RawTerms .large 284240 .exactZero (none)

def event284243 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27665⟩⟩) 0 ⟨9546⟩ 284242

def event284244 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27665⟩⟩) 1 ⟨27664⟩ 284221

def event284245 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27665⟩⟩) (.sum [.predecessor 0 284243 .coefficient, .predecessor 1 284244 .coefficient])

def exact284246RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284246RawTermsValid :
    exact284246RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284246 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27665⟩⟩) exact284246RawTerms .large 284245 .exactZero (none)

def event284247 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27856⟩⟩) 0 ⟨27665⟩ 284246

def event284248 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27856⟩⟩) 1 ⟨27853⟩ 284205

def event284249 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27856⟩⟩) (.product (.predecessor 0 284247 .coefficient) (.predecessor 1 284248 .coefficient) (⟨false, false, none, none, none⟩))

def event284250 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27856⟩⟩, .operator (⟨284246, 0⟩, ⟨284205, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27853⟩⟩]⟩, (1)⟩)

def event284251 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27856⟩⟩, .operator (⟨284246, 1⟩, ⟨284205, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27853⟩⟩]⟩, (-1)⟩)

def event284252 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨27856⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27853⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27853⟩⟩) ⟨27373⟩ 284202)

def event284253 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27856⟩⟩, .relation 284252 0, ⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨27373⟩⟩]⟩, (-1)⟩)

def exact284254RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27853⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨27373⟩⟩]⟩, (-1)⟩]

theorem exact284254RawTermsValid :
    exact284254RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284254 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27856⟩⟩) exact284254RawTerms .large 284249 .exactZero (none)

def event284255 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26360⟩⟩) 0 ⟨25952⟩ 284194

def event284256 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26360⟩⟩) (.authority (.programFamilyFact))

def exact284257RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], []⟩, (1)⟩]

theorem exact284257RawTermsValid :
    exact284257RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284257 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26360⟩⟩) exact284257RawTerms (.finite 30) 284256 .exactZero (none)

def event284258 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26362⟩⟩) 0 ⟨6908⟩ 284216

def event284259 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26362⟩⟩) 1 ⟨26360⟩ 284257

def event284260 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26362⟩⟩) (.product (.predecessor 0 284258 .coefficient) (.predecessor 1 284259 .coefficient) (⟨false, true, none, none, some 1⟩))

def event284261 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26362⟩⟩, .operator (⟨284216, 0⟩, ⟨284257, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact284262RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact284262RawTermsValid :
    exact284262RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284262 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26362⟩⟩) exact284262RawTerms .large 284260 .exactZero (none)

def event284263 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7189⟩⟩) 0 ⟨7177⟩ 284198

def event284264 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7189⟩⟩) (.authority (.operator))

def exact284265RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩]

theorem exact284265RawTermsValid :
    exact284265RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284265 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7189⟩⟩) exact284265RawTerms .large 284264 .exactZero (none)

def event284266 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26363⟩⟩) 0 ⟨7189⟩ 284265

def event284267 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26363⟩⟩) 1 ⟨26362⟩ 284262

def event284268 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26363⟩⟩) (.sum [.predecessor 0 284266 .coefficient, .predecessor 1 284267 .coefficient])

def exact284269RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284269RawTermsValid :
    exact284269RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284269 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26363⟩⟩) exact284269RawTerms .large 284268 .exactZero (none)

def event284270 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27857⟩⟩) 0 ⟨26363⟩ 284269

def event284271 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27857⟩⟩) 1 ⟨27856⟩ 284254

def event284272 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27857⟩⟩) (.sum [.predecessor 0 284270 .coefficient, .predecessor 1 284271 .coefficient])

def exact284273RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27853⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨27373⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284273RawTermsValid :
    exact284273RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284273 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27857⟩⟩) exact284273RawTerms .large 284272 .exactZero (none)

def event284274 : Event := .preFoldPolynomial 284273 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27853⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨27373⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact284275RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27853⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨27373⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event284275 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨27857⟩⟩) 284274 exact284275RawTerms .large 284272 .exactZero (none)

def event284276 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨25952⟩⟩) ⟨⟨68⟩, ⟨47⟩, ⟨135⟩⟩ ⟨284112, 284276⟩

def event284277 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨26792⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26789⟩⟩]⟩) (1) 0 2 (.universal 284276 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26789⟩⟩]⟩) (none) 284275)

def event284278 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26792⟩⟩, .relation 284277 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩)

def event284279 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26792⟩⟩, .relation 284277 1, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27853⟩⟩]⟩, (-1)⟩)

def event284280 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26792⟩⟩, .relation 284277 2, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨27373⟩⟩]⟩, (1)⟩)

def event284281 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26792⟩⟩, .relation 284277 3, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact284282RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27853⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨27373⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284282RawTermsValid :
    exact284282RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284282 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26792⟩⟩) exact284282RawTerms .large 284108 (.finite 202072841853861888) (some (284110))

def event284283 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27855⟩⟩) 0 ⟨26792⟩ 284282

def event284284 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27855⟩⟩) 1 ⟨27854⟩ 284098

def event284285 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27855⟩⟩) (.sum [.predecessor 0 284283 .coefficient, .predecessor 1 284284 .coefficient])

def event284286 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27855⟩⟩, .operator (⟨284282, 2⟩, ⟨284098, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], [⟨.program ⟨257⟩, ⟨27373⟩⟩]⟩, (-1)⟩)

def event284287 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27855⟩⟩, .operator (⟨284282, 1⟩, ⟨284098, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27853⟩⟩]⟩, (1)⟩)

def event284288 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27855⟩⟩) (.sum [.result 284282 .summary, .result 284098 .summary])

def exact284289RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284289RawTermsValid :
    exact284289RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284289 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27855⟩⟩) exact284289RawTerms .large 284285 (.finite 2998072422921948889088) (some (284288))

def event284290 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28141⟩⟩) 0 ⟨27855⟩ 284289

def event284291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28141⟩⟩) 1 ⟨28139⟩ 284014

def event284292 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28141⟩⟩) (.product (.predecessor 0 284290 .coefficient) (.predecessor 1 284291 .coefficient) (⟨false, false, none, none, none⟩))

def event284293 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28141⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩) [⟨.result 284014 .coefficient, false, none⟩])

def event284294 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28141⟩⟩) (.product (.result 284289 .summary) (.transfer 284293) (⟨false, false, none, none, none⟩))

def event284295 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28141⟩⟩, .operator (⟨284289, 0⟩, ⟨284014, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩, (1)⟩)

def event284296 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28141⟩⟩, .operator (⟨284289, 1⟩, ⟨284014, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩, (-1)⟩)

def event284297 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨28141⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28139⟩⟩) ⟨27507⟩ 284011)

def event284298 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28141⟩⟩, .relation 284297 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨27507⟩⟩]⟩, (-1)⟩)

def exact284299RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28139⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨26360⟩⟩], [⟨.program ⟨257⟩, ⟨27507⟩⟩]⟩, (-1)⟩]

theorem exact284299RawTermsValid :
    exact284299RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284299 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28141⟩⟩) exact284299RawTerms .large 284292 (.finite 32191557518723128098041228165120) (some (284294))

def event284300 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27036⟩⟩) 0 ⟨26361⟩ 13730

def event284301 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27036⟩⟩) (.authority (.relationPreimageSource ⟨79⟩))

def exact284302RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27036⟩⟩]⟩, (1)⟩]

theorem exact284302RawTermsValid :
    exact284302RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284302 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27036⟩⟩) exact284302RawTerms (.finite 5647228698) 284301 .exactZero (none)

def event284303 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27038⟩⟩) 0 ⟨27036⟩ 284302

def event284304 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27038⟩⟩) 1 ⟨2370⟩ 4

def event284305 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27038⟩⟩) (.scale (.predecessor 0 284303 .coefficient) (.value (.predecessor 1 284304 .coefficient)))

def exact284306RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27036⟩⟩]⟩, (1)⟩]

theorem exact284306RawTermsValid :
    exact284306RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284306 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27038⟩⟩) exact284306RawTerms (.finite 5647228698) 284305 .exactZero (none)

def event284307 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27039⟩⟩) 0 ⟨5491⟩ 280745

def event284308 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27039⟩⟩) 1 ⟨27038⟩ 284306

def event284309 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27039⟩⟩) (.product (.predecessor 0 284307 .coefficient) (.predecessor 1 284308 .coefficient) (⟨false, false, none, none, none⟩))

def event284310 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27039⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨27036⟩⟩]⟩) [⟨.result 284302 .coefficient, false, none⟩])

def event284311 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27039⟩⟩) (.product (.result 280745 .summary) (.transfer 284310) (⟨false, false, none, none, none⟩))

def event284312 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27039⟩⟩, .operator (⟨280745, 0⟩, ⟨284306, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27036⟩⟩]⟩, (1)⟩)

def event284313 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨27037⟩⟩)

def event284314 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event284315 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event284316 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event284317 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event284318 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event284319 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event284320 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event284321 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event284322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 284321

def event284323 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 284319

def event284324 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 284322 .coefficient) (.value (.predecessor 1 284323 .coefficient)))

def event284325 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event284326 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 284325

def event284327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 284317

def event284328 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 284326 .coefficient, .predecessor 1 284327 .coefficient])

def event284329 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event284330 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 284329

def event284331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 284315

def event284332 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 284331 .coefficient))

def event284333 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event284334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25950⟩⟩) 0 ⟨5487⟩ 284333

def event284335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25950⟩⟩) (.authority (.programFamilyFact))

def exact284336RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25950⟩⟩], []⟩, (1)⟩]

theorem exact284336RawTermsValid :
    exact284336RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284336 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25950⟩⟩) exact284336RawTerms (.finite 30) 284335 .exactZero (none)

def event284337 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12891⟩⟩) 0 ⟨5487⟩ 284333

def event284338 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12891⟩⟩) (.authority (.programFamilyFact))

def exact284339RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩], []⟩, (1)⟩]

theorem exact284339RawTermsValid :
    exact284339RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284339 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12891⟩⟩) exact284339RawTerms (.finite 30) 284338 .exactZero (none)

def event284340 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25951⟩⟩) 0 ⟨12891⟩ 284339

def event284341 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25951⟩⟩) 1 ⟨25950⟩ 284336

def event284342 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25951⟩⟩) (.product (.predecessor 0 284340 .coefficient) (.predecessor 1 284341 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event284343 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25951⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], []⟩) [⟨.result 284339 .coefficient, true, some 1⟩, ⟨.result 284336 .coefficient, true, some 1⟩])

def event284344 : Event := .survivorFold (1) 284343

def exact284345RawTerms : List Term := []

theorem exact284345RawTermsValid :
    exact284345RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284345 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25951⟩⟩) exact284345RawTerms (.finite 900) 284342 (.finite 900) (some (284343))

def event284346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25952⟩⟩) 0 ⟨25951⟩ 284345

def event284347 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25952⟩⟩) (.identity (.predecessor 0 284346 .coefficient))

def event284348 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨25952⟩⟩) (.finite 900)

def event284349 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26360⟩⟩) 0 ⟨25952⟩ 284348

def event284350 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26360⟩⟩) (.authority (.programFamilyFact))

def exact284351RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], []⟩, (1)⟩]

theorem exact284351RawTermsValid :
    exact284351RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284351 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26360⟩⟩) exact284351RawTerms (.finite 30) 284350 .exactZero (none)

def event284352 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26361⟩⟩) 0 ⟨26360⟩ 284351

def event284353 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26361⟩⟩) (.identity (.predecessor 0 284352 .coefficient))

def event284354 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26361⟩⟩) (.finite 30)

def event284355 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27036⟩⟩) 0 ⟨26361⟩ 284354

def event284356 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27036⟩⟩) (.authority (.relationPreimageSource ⟨79⟩))

def exact284357RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27036⟩⟩]⟩, (1)⟩]

theorem exact284357RawTermsValid :
    exact284357RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284357 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27036⟩⟩) exact284357RawTerms (.finite 5647228698) 284356 .exactZero (none)

def event284358 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact284359RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact284359RawTermsValid :
    exact284359RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284359 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact284359RawTerms .large 284358 .exactZero (none)

def event284360 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27037⟩⟩) 0 ⟨35⟩ 284359

def event284361 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27037⟩⟩) 1 ⟨27036⟩ 284357

def event284362 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27037⟩⟩) (.product (.predecessor 0 284360 .coefficient) (.predecessor 1 284361 .coefficient) (⟨false, false, none, none, none⟩))

def event284363 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27037⟩⟩, .operator (⟨284359, 0⟩, ⟨284357, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27036⟩⟩]⟩, (1)⟩)

def exact284364RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27036⟩⟩]⟩, (1)⟩]

theorem exact284364RawTermsValid :
    exact284364RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284364 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27037⟩⟩) exact284364RawTerms .large 284362 .exactZero (none)

def event284365 : Event := .preFoldPolynomial 284364 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27036⟩⟩]⟩, (1)⟩] .exactZero none

def exact284366RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27036⟩⟩]⟩, (1)⟩]

def event284366 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨27037⟩⟩) 284365 exact284366RawTerms .large 284362 .exactZero (none)

def event284367 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨28143⟩⟩)

def event284368 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event284369 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event284370 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event284371 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event284372 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event284373 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event284374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event284375 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event284376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 284375

def event284377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 284373

def event284378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 284376 .coefficient) (.value (.predecessor 1 284377 .coefficient)))

def event284379 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event284380 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 284379

def event284381 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 284371

def event284382 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 284380 .coefficient, .predecessor 1 284381 .coefficient])

def event284383 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event284384 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 284383

def event284385 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 284369

def event284386 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 284385 .coefficient))

def event284387 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event284388 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25950⟩⟩) 0 ⟨5487⟩ 284387

def event284389 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25950⟩⟩) (.authority (.programFamilyFact))

def exact284390RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25950⟩⟩], []⟩, (1)⟩]

theorem exact284390RawTermsValid :
    exact284390RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284390 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25950⟩⟩) exact284390RawTerms (.finite 30) 284389 .exactZero (none)

def event284391 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12891⟩⟩) 0 ⟨5487⟩ 284387

def event284392 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12891⟩⟩) (.authority (.programFamilyFact))

def exact284393RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩], []⟩, (1)⟩]

theorem exact284393RawTermsValid :
    exact284393RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284393 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12891⟩⟩) exact284393RawTerms (.finite 30) 284392 .exactZero (none)

def event284394 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25951⟩⟩) 0 ⟨12891⟩ 284393

def event284395 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25951⟩⟩) 1 ⟨25950⟩ 284390

def event284396 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25951⟩⟩) (.product (.predecessor 0 284394 .coefficient) (.predecessor 1 284395 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event284397 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨25951⟩⟩, .operator (⟨284393, 0⟩, ⟨284390, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], []⟩, (1)⟩)

def exact284398RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12891⟩⟩, ⟨.program ⟨257⟩, ⟨25950⟩⟩], []⟩, (1)⟩]

theorem exact284398RawTermsValid :
    exact284398RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284398 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25951⟩⟩) exact284398RawTerms (.finite 900) 284396 .exactZero (none)

def event284399 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25952⟩⟩) 0 ⟨25951⟩ 284398

def event284400 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25952⟩⟩) (.identity (.predecessor 0 284399 .coefficient))

def event284401 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨25952⟩⟩) (.finite 900)

def event284402 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26360⟩⟩) 0 ⟨25952⟩ 284401

def event284403 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26360⟩⟩) (.authority (.programFamilyFact))

def exact284404RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26360⟩⟩], []⟩, (1)⟩]

theorem exact284404RawTermsValid :
    exact284404RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284404 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26360⟩⟩) exact284404RawTerms (.finite 30) 284403 .exactZero (none)

def event284405 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26361⟩⟩) 0 ⟨26360⟩ 284404

def event284406 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26361⟩⟩) (.identity (.predecessor 0 284405 .coefficient))

def event284407 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26361⟩⟩) (.finite 30)

def event284408 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27505⟩⟩) 0 ⟨26361⟩ 284407

def event284409 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27505⟩⟩) (.authority (.programFamilyFact))

def event284410 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27505⟩⟩) (.finite 3720)

def event284411 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event284412 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27507⟩⟩) 0 ⟨7177⟩ 284411

def event284413 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27507⟩⟩) 1 ⟨27505⟩ 284410

def event284414 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27507⟩⟩) (.authority (.operator))

def exact284415RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27507⟩⟩]⟩, (1)⟩]

theorem exact284415RawTermsValid :
    exact284415RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284415 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27507⟩⟩) exact284415RawTerms .large 284414 .exactZero (none)

def eventLeaf17760 : Array AnnotatedEvent := #[
  { event := event284160
    frameStart := 284160 },
  { event := event284161
    frameStart := 284160 },
  { event := event284162
    frameStart := 284160 },
  { event := event284163
    frameStart := 284160 },
  { event := event284164
    frameStart := 284160 },
  { event := event284165
    frameStart := 284160 },
  { event := event284166
    frameStart := 284160 },
  { event := event284167
    frameStart := 284160 },
  { event := event284168
    frameStart := 284160 },
  { event := event284169
    frameStart := 284160 },
  { event := event284170
    frameStart := 284160 },
  { event := event284171
    frameStart := 284160 },
  { event := event284172
    frameStart := 284160 },
  { event := event284173
    frameStart := 284160 },
  { event := event284174
    frameStart := 284160 },
  { event := event284175
    frameStart := 284160 }
]

def eventLeaf17761 : Array AnnotatedEvent := #[
  { event := event284176
    frameStart := 284160 },
  { event := event284177
    frameStart := 284160 },
  { event := event284178
    frameStart := 284160 },
  { event := event284179
    frameStart := 284160 },
  { event := event284180
    frameStart := 284160 },
  { event := event284181
    frameStart := 284160 },
  { event := event284182
    frameStart := 284160 },
  { event := event284183
    frameStart := 284160 },
  { event := event284184
    frameStart := 284160 },
  { event := event284185
    frameStart := 284160 },
  { event := event284186
    frameStart := 284160 },
  { event := event284187
    frameStart := 284160 },
  { event := event284188
    frameStart := 284160 },
  { event := event284189
    frameStart := 284160 },
  { event := event284190
    frameStart := 284160 },
  { event := event284191
    frameStart := 284160 }
]

def eventLeaf17762 : Array AnnotatedEvent := #[
  { event := event284192
    frameStart := 284160 },
  { event := event284193
    frameStart := 284160 },
  { event := event284194
    frameStart := 284160 },
  { event := event284195
    frameStart := 284160 },
  { event := event284196
    frameStart := 284160 },
  { event := event284197
    frameStart := 284160 },
  { event := event284198
    frameStart := 284160 },
  { event := event284199
    frameStart := 284160 },
  { event := event284200
    frameStart := 284160 },
  { event := event284201
    frameStart := 284160 },
  { event := event284202
    frameStart := 284160 },
  { event := event284203
    frameStart := 284160 },
  { event := event284204
    frameStart := 284160 },
  { event := event284205
    frameStart := 284160 },
  { event := event284206
    frameStart := 284160 },
  { event := event284207
    frameStart := 284160 }
]

def eventLeaf17763 : Array AnnotatedEvent := #[
  { event := event284208
    frameStart := 284160 },
  { event := event284209
    frameStart := 284160 },
  { event := event284210
    frameStart := 284160 },
  { event := event284211
    frameStart := 284160 },
  { event := event284212
    frameStart := 284160 },
  { event := event284213
    frameStart := 284160 },
  { event := event284214
    frameStart := 284160 },
  { event := event284215
    frameStart := 284160 },
  { event := event284216
    frameStart := 284160 },
  { event := event284217
    frameStart := 284160 },
  { event := event284218
    frameStart := 284160 },
  { event := event284219
    frameStart := 284160 },
  { event := event284220
    frameStart := 284160 },
  { event := event284221
    frameStart := 284160 },
  { event := event284222
    frameStart := 284160 },
  { event := event284223
    frameStart := 284160 }
]

def eventLeaf17764 : Array AnnotatedEvent := #[
  { event := event284224
    frameStart := 284160 },
  { event := event284225
    frameStart := 284160 },
  { event := event284226
    frameStart := 284160 },
  { event := event284227
    frameStart := 284160 },
  { event := event284228
    frameStart := 284160 },
  { event := event284229
    frameStart := 284160 },
  { event := event284230
    frameStart := 284160 },
  { event := event284231
    frameStart := 284160 },
  { event := event284232
    frameStart := 284160 },
  { event := event284233
    frameStart := 284160 },
  { event := event284234
    frameStart := 284160 },
  { event := event284235
    frameStart := 284160 },
  { event := event284236
    frameStart := 284160 },
  { event := event284237
    frameStart := 284160 },
  { event := event284238
    frameStart := 284160 },
  { event := event284239
    frameStart := 284160 }
]

def eventLeaf17765 : Array AnnotatedEvent := #[
  { event := event284240
    frameStart := 284160 },
  { event := event284241
    frameStart := 284160 },
  { event := event284242
    frameStart := 284160 },
  { event := event284243
    frameStart := 284160 },
  { event := event284244
    frameStart := 284160 },
  { event := event284245
    frameStart := 284160 },
  { event := event284246
    frameStart := 284160 },
  { event := event284247
    frameStart := 284160 },
  { event := event284248
    frameStart := 284160 },
  { event := event284249
    frameStart := 284160 },
  { event := event284250
    frameStart := 284160 },
  { event := event284251
    frameStart := 284160 },
  { event := event284252
    frameStart := 284160 },
  { event := event284253
    frameStart := 284160 },
  { event := event284254
    frameStart := 284160 },
  { event := event284255
    frameStart := 284160 }
]

def eventLeaf17766 : Array AnnotatedEvent := #[
  { event := event284256
    frameStart := 284160 },
  { event := event284257
    frameStart := 284160 },
  { event := event284258
    frameStart := 284160 },
  { event := event284259
    frameStart := 284160 },
  { event := event284260
    frameStart := 284160 },
  { event := event284261
    frameStart := 284160 },
  { event := event284262
    frameStart := 284160 },
  { event := event284263
    frameStart := 284160 },
  { event := event284264
    frameStart := 284160 },
  { event := event284265
    frameStart := 284160 },
  { event := event284266
    frameStart := 284160 },
  { event := event284267
    frameStart := 284160 },
  { event := event284268
    frameStart := 284160 },
  { event := event284269
    frameStart := 284160 },
  { event := event284270
    frameStart := 284160 },
  { event := event284271
    frameStart := 284160 }
]

def eventLeaf17767 : Array AnnotatedEvent := #[
  { event := event284272
    frameStart := 284160 },
  { event := event284273
    frameStart := 284160 },
  { event := event284274
    frameStart := 284160 },
  { event := event284275
    frameStart := 284160 },
  { event := event284276
    frameStart := 0 },
  { event := event284277
    frameStart := 0 },
  { event := event284278
    frameStart := 0 },
  { event := event284279
    frameStart := 0 },
  { event := event284280
    frameStart := 0 },
  { event := event284281
    frameStart := 0 },
  { event := event284282
    frameStart := 0 },
  { event := event284283
    frameStart := 0 },
  { event := event284284
    frameStart := 0 },
  { event := event284285
    frameStart := 0 },
  { event := event284286
    frameStart := 0 },
  { event := event284287
    frameStart := 0 }
]

def eventLeaf17768 : Array AnnotatedEvent := #[
  { event := event284288
    frameStart := 0 },
  { event := event284289
    frameStart := 0 },
  { event := event284290
    frameStart := 0 },
  { event := event284291
    frameStart := 0 },
  { event := event284292
    frameStart := 0 },
  { event := event284293
    frameStart := 0 },
  { event := event284294
    frameStart := 0 },
  { event := event284295
    frameStart := 0 },
  { event := event284296
    frameStart := 0 },
  { event := event284297
    frameStart := 0 },
  { event := event284298
    frameStart := 0 },
  { event := event284299
    frameStart := 0 },
  { event := event284300
    frameStart := 0 },
  { event := event284301
    frameStart := 0 },
  { event := event284302
    frameStart := 0 },
  { event := event284303
    frameStart := 0 }
]

def eventLeaf17769 : Array AnnotatedEvent := #[
  { event := event284304
    frameStart := 0 },
  { event := event284305
    frameStart := 0 },
  { event := event284306
    frameStart := 0 },
  { event := event284307
    frameStart := 0 },
  { event := event284308
    frameStart := 0 },
  { event := event284309
    frameStart := 0 },
  { event := event284310
    frameStart := 0 },
  { event := event284311
    frameStart := 0 },
  { event := event284312
    frameStart := 0 },
  { event := event284313
    frameStart := 284313 },
  { event := event284314
    frameStart := 284313 },
  { event := event284315
    frameStart := 284313 },
  { event := event284316
    frameStart := 284313 },
  { event := event284317
    frameStart := 284313 },
  { event := event284318
    frameStart := 284313 },
  { event := event284319
    frameStart := 284313 }
]

def eventLeaf17770 : Array AnnotatedEvent := #[
  { event := event284320
    frameStart := 284313 },
  { event := event284321
    frameStart := 284313 },
  { event := event284322
    frameStart := 284313 },
  { event := event284323
    frameStart := 284313 },
  { event := event284324
    frameStart := 284313 },
  { event := event284325
    frameStart := 284313 },
  { event := event284326
    frameStart := 284313 },
  { event := event284327
    frameStart := 284313 },
  { event := event284328
    frameStart := 284313 },
  { event := event284329
    frameStart := 284313 },
  { event := event284330
    frameStart := 284313 },
  { event := event284331
    frameStart := 284313 },
  { event := event284332
    frameStart := 284313 },
  { event := event284333
    frameStart := 284313 },
  { event := event284334
    frameStart := 284313 },
  { event := event284335
    frameStart := 284313 }
]

def eventLeaf17771 : Array AnnotatedEvent := #[
  { event := event284336
    frameStart := 284313 },
  { event := event284337
    frameStart := 284313 },
  { event := event284338
    frameStart := 284313 },
  { event := event284339
    frameStart := 284313 },
  { event := event284340
    frameStart := 284313 },
  { event := event284341
    frameStart := 284313 },
  { event := event284342
    frameStart := 284313 },
  { event := event284343
    frameStart := 284313 },
  { event := event284344
    frameStart := 284313 },
  { event := event284345
    frameStart := 284313 },
  { event := event284346
    frameStart := 284313 },
  { event := event284347
    frameStart := 284313 },
  { event := event284348
    frameStart := 284313 },
  { event := event284349
    frameStart := 284313 },
  { event := event284350
    frameStart := 284313 },
  { event := event284351
    frameStart := 284313 }
]

def eventLeaf17772 : Array AnnotatedEvent := #[
  { event := event284352
    frameStart := 284313 },
  { event := event284353
    frameStart := 284313 },
  { event := event284354
    frameStart := 284313 },
  { event := event284355
    frameStart := 284313 },
  { event := event284356
    frameStart := 284313 },
  { event := event284357
    frameStart := 284313 },
  { event := event284358
    frameStart := 284313 },
  { event := event284359
    frameStart := 284313 },
  { event := event284360
    frameStart := 284313 },
  { event := event284361
    frameStart := 284313 },
  { event := event284362
    frameStart := 284313 },
  { event := event284363
    frameStart := 284313 },
  { event := event284364
    frameStart := 284313 },
  { event := event284365
    frameStart := 284313 },
  { event := event284366
    frameStart := 284313 },
  { event := event284367
    frameStart := 284367 }
]

def eventLeaf17773 : Array AnnotatedEvent := #[
  { event := event284368
    frameStart := 284367 },
  { event := event284369
    frameStart := 284367 },
  { event := event284370
    frameStart := 284367 },
  { event := event284371
    frameStart := 284367 },
  { event := event284372
    frameStart := 284367 },
  { event := event284373
    frameStart := 284367 },
  { event := event284374
    frameStart := 284367 },
  { event := event284375
    frameStart := 284367 },
  { event := event284376
    frameStart := 284367 },
  { event := event284377
    frameStart := 284367 },
  { event := event284378
    frameStart := 284367 },
  { event := event284379
    frameStart := 284367 },
  { event := event284380
    frameStart := 284367 },
  { event := event284381
    frameStart := 284367 },
  { event := event284382
    frameStart := 284367 },
  { event := event284383
    frameStart := 284367 }
]

def eventLeaf17774 : Array AnnotatedEvent := #[
  { event := event284384
    frameStart := 284367 },
  { event := event284385
    frameStart := 284367 },
  { event := event284386
    frameStart := 284367 },
  { event := event284387
    frameStart := 284367 },
  { event := event284388
    frameStart := 284367 },
  { event := event284389
    frameStart := 284367 },
  { event := event284390
    frameStart := 284367 },
  { event := event284391
    frameStart := 284367 },
  { event := event284392
    frameStart := 284367 },
  { event := event284393
    frameStart := 284367 },
  { event := event284394
    frameStart := 284367 },
  { event := event284395
    frameStart := 284367 },
  { event := event284396
    frameStart := 284367 },
  { event := event284397
    frameStart := 284367 },
  { event := event284398
    frameStart := 284367 },
  { event := event284399
    frameStart := 284367 }
]

def eventLeaf17775 : Array AnnotatedEvent := #[
  { event := event284400
    frameStart := 284367 },
  { event := event284401
    frameStart := 284367 },
  { event := event284402
    frameStart := 284367 },
  { event := event284403
    frameStart := 284367 },
  { event := event284404
    frameStart := 284367 },
  { event := event284405
    frameStart := 284367 },
  { event := event284406
    frameStart := 284367 },
  { event := event284407
    frameStart := 284367 },
  { event := event284408
    frameStart := 284367 },
  { event := event284409
    frameStart := 284367 },
  { event := event284410
    frameStart := 284367 },
  { event := event284411
    frameStart := 284367 },
  { event := event284412
    frameStart := 284367 },
  { event := event284413
    frameStart := 284367 },
  { event := event284414
    frameStart := 284367 },
  { event := event284415
    frameStart := 284367 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1110
