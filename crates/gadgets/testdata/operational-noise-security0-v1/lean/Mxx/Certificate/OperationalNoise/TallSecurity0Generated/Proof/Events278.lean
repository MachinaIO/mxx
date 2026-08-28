import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events278

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event71168 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19382⟩⟩) 0 ⟨19380⟩ 71167

def event71169 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19382⟩⟩) 1 ⟨2348⟩ 4

def event71170 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19382⟩⟩) (.scale (.predecessor 0 71168 .coefficient) (.value (.predecessor 1 71169 .coefficient)))

def exact71171RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩, (1)⟩]

theorem exact71171RawTermsValid :
    exact71171RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71171 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19382⟩⟩) exact71171RawTerms (.finite 136065468) 71170 .exactZero (none)

def event71172 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19383⟩⟩) 0 ⟨5535⟩ 65387

def event71173 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19383⟩⟩) 1 ⟨19382⟩ 71171

def event71174 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19383⟩⟩) (.product (.predecessor 0 71172 .coefficient) (.predecessor 1 71173 .coefficient) (⟨false, false, none, none, none⟩))

def event71175 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19383⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩) [⟨.result 71167 .coefficient, false, none⟩])

def event71176 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19383⟩⟩) (.product (.result 65387 .summary) (.transfer 71175) (⟨false, false, none, none, none⟩))

def event71177 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19383⟩⟩, .operator (⟨65387, 0⟩, ⟨71171, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩, (1)⟩)

def event71178 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19381⟩⟩)

def event71179 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event71180 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event71181 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event71182 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event71183 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event71184 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event71185 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event71186 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event71187 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 71186

def event71188 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 71184

def event71189 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 71187 .coefficient) (.value (.predecessor 1 71188 .coefficient)))

def event71190 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event71191 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 71190

def event71192 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 71182

def event71193 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 71191 .coefficient, .predecessor 1 71192 .coefficient])

def event71194 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event71195 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 71194

def event71196 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 71180

def event71197 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 71196 .coefficient))

def event71198 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event71199 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11297⟩⟩) 0 ⟨5530⟩ 71198

def event71200 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11297⟩⟩) (.authority (.programFamilyFact))

def exact71201RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩], []⟩, (1)⟩]

theorem exact71201RawTermsValid :
    exact71201RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71201 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11297⟩⟩) exact71201RawTerms (.finite 12) 71200 .exactZero (none)

def event71202 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13764⟩⟩) 0 ⟨5530⟩ 71198

def event71203 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13764⟩⟩) (.authority (.programFamilyFact))

def exact71204RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩, (1)⟩]

theorem exact71204RawTermsValid :
    exact71204RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71204 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13764⟩⟩) exact71204RawTerms (.finite 12) 71203 .exactZero (none)

def event71205 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13765⟩⟩) 0 ⟨13764⟩ 71204

def event71206 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13765⟩⟩) 1 ⟨11297⟩ 71201

def event71207 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13765⟩⟩) (.product (.predecessor 0 71205 .coefficient) (.predecessor 1 71206 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event71208 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13765⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩) [⟨.result 71204 .coefficient, true, some 1⟩, ⟨.result 71201 .coefficient, true, some 1⟩])

def event71209 : Event := .survivorFold (1) 71208

def exact71210RawTerms : List Term := []

theorem exact71210RawTermsValid :
    exact71210RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71210 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13765⟩⟩) exact71210RawTerms (.finite 144) 71207 (.finite 144) (some (71208))

def event71211 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13766⟩⟩) 0 ⟨13765⟩ 71210

def event71212 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13766⟩⟩) (.identity (.predecessor 0 71211 .coefficient))

def event71213 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13766⟩⟩) (.finite 144)

def event71214 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19380⟩⟩) 0 ⟨13766⟩ 71213

def event71215 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19380⟩⟩) (.authority (.relationPreimageSource ⟨13⟩))

def exact71216RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩, (1)⟩]

theorem exact71216RawTermsValid :
    exact71216RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71216 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19380⟩⟩) exact71216RawTerms (.finite 136065468) 71215 .exactZero (none)

def event71217 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact71218RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact71218RawTermsValid :
    exact71218RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71218 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact71218RawTerms .large 71217 .exactZero (none)

def event71219 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19381⟩⟩) 0 ⟨6⟩ 71218

def event71220 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19381⟩⟩) 1 ⟨19380⟩ 71216

def event71221 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19381⟩⟩) (.product (.predecessor 0 71219 .coefficient) (.predecessor 1 71220 .coefficient) (⟨false, false, none, none, none⟩))

def event71222 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19381⟩⟩, .operator (⟨71218, 0⟩, ⟨71216, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩, (1)⟩)

def exact71223RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩, (1)⟩]

theorem exact71223RawTermsValid :
    exact71223RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71223 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19381⟩⟩) exact71223RawTerms .large 71221 .exactZero (none)

def event71224 : Event := .preFoldPolynomial 71223 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩, (1)⟩] .exactZero none

def exact71225RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩, (1)⟩]

def event71225 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19381⟩⟩) 71224 exact71225RawTerms .large 71221 .exactZero (none)

def event71226 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25911⟩⟩)

def event71227 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event71228 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event71229 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event71230 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event71231 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event71232 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event71233 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event71234 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event71235 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 71234

def event71236 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 71232

def event71237 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 71235 .coefficient) (.value (.predecessor 1 71236 .coefficient)))

def event71238 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event71239 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 71238

def event71240 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 71230

def event71241 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 71239 .coefficient, .predecessor 1 71240 .coefficient])

def event71242 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event71243 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 71242

def event71244 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 71228

def event71245 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 71244 .coefficient))

def event71246 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event71247 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11297⟩⟩) 0 ⟨5530⟩ 71246

def event71248 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11297⟩⟩) (.authority (.programFamilyFact))

def exact71249RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩], []⟩, (1)⟩]

theorem exact71249RawTermsValid :
    exact71249RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71249 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11297⟩⟩) exact71249RawTerms (.finite 12) 71248 .exactZero (none)

def event71250 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13764⟩⟩) 0 ⟨5530⟩ 71246

def event71251 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13764⟩⟩) (.authority (.programFamilyFact))

def exact71252RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩, (1)⟩]

theorem exact71252RawTermsValid :
    exact71252RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71252 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13764⟩⟩) exact71252RawTerms (.finite 12) 71251 .exactZero (none)

def event71253 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13765⟩⟩) 0 ⟨13764⟩ 71252

def event71254 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13765⟩⟩) 1 ⟨11297⟩ 71249

def event71255 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13765⟩⟩) (.product (.predecessor 0 71253 .coefficient) (.predecessor 1 71254 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event71256 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13765⟩⟩, .operator (⟨71252, 0⟩, ⟨71249, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩, (1)⟩)

def exact71257RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩, (1)⟩]

theorem exact71257RawTermsValid :
    exact71257RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71257 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13765⟩⟩) exact71257RawTerms (.finite 144) 71255 .exactZero (none)

def event71258 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13766⟩⟩) 0 ⟨13765⟩ 71257

def event71259 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13766⟩⟩) (.identity (.predecessor 0 71258 .coefficient))

def event71260 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13766⟩⟩) (.finite 144)

def event71261 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23497⟩⟩) 0 ⟨13766⟩ 71260

def event71262 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23497⟩⟩) (.authority (.programFamilyFact))

def event71263 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23497⟩⟩) (.finite 3720)

def event71264 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event71265 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23498⟩⟩) 0 ⟨6689⟩ 71264

def event71266 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23498⟩⟩) 1 ⟨23497⟩ 71263

def event71267 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23498⟩⟩) (.authority (.operator))

def exact71268RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23498⟩⟩]⟩, (1)⟩]

theorem exact71268RawTermsValid :
    exact71268RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71268 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23498⟩⟩) exact71268RawTerms .large 71267 .exactZero (none)

def event71269 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25907⟩⟩) 0 ⟨23498⟩ 71268

def event71270 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25907⟩⟩) (.authority (.operator))

def exact71271RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩, (1)⟩]

theorem exact71271RawTermsValid :
    exact71271RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71271 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25907⟩⟩) exact71271RawTerms (.finite 8192) 71270 .exactZero (none)

def event71272 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event71273 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event71274 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13876⟩⟩) 0 ⟨13766⟩ 71260

def event71275 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13876⟩⟩) 1 ⟨110⟩ 71273

def event71276 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13876⟩⟩) (.sum [.predecessor 0 71274 .coefficient, .predecessor 1 71275 .coefficient])

def event71277 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13876⟩⟩) (.finite 144)

def event71278 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13877⟩⟩) 0 ⟨13876⟩ 71277

def event71279 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13877⟩⟩) (.identity (.predecessor 0 71278 .coefficient))

def exact71280RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩, (1)⟩]

theorem exact71280RawTermsValid :
    exact71280RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71280 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13877⟩⟩) exact71280RawTerms (.finite 144) 71279 .exactZero (none)

def event71281 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact71282RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact71282RawTermsValid :
    exact71282RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71282 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact71282RawTerms .large 71281 .exactZero (none)

def event71283 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13878⟩⟩) 0 ⟨6544⟩ 71282

def event71284 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13878⟩⟩) 1 ⟨13877⟩ 71280

def event71285 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13878⟩⟩) (.product (.predecessor 0 71283 .coefficient) (.predecessor 1 71284 .coefficient) (⟨false, false, none, none, none⟩))

def event71286 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13878⟩⟩, .operator (⟨71282, 0⟩, ⟨71280, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact71287RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact71287RawTermsValid :
    exact71287RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71287 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13878⟩⟩) exact71287RawTerms .large 71285 .exactZero (none)

def event71288 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event71289 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event71290 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 71264

def event71291 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact71292RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact71292RawTermsValid :
    exact71292RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71292 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact71292RawTerms .large 71291 .exactZero (none)

def event71293 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6777⟩⟩) 0 ⟨6757⟩ 71292

def event71294 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6777⟩⟩) (.identity (.predecessor 0 71293 .coefficient))

def exact71295RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩]

theorem exact71295RawTermsValid :
    exact71295RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71295 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6777⟩⟩) exact71295RawTerms .large 71294 .exactZero (none)

def event71296 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7846⟩⟩) 0 ⟨6777⟩ 71295

def event71297 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7846⟩⟩) (.authority (.operator))

def exact71298RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (1)⟩]

theorem exact71298RawTermsValid :
    exact71298RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71298 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7846⟩⟩) exact71298RawTerms (.finite 8192) 71297 .exactZero (none)

def event71299 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7847⟩⟩) 0 ⟨7846⟩ 71298

def event71300 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7847⟩⟩) 1 ⟨2348⟩ 71289

def event71301 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7847⟩⟩) (.scale (.predecessor 0 71299 .coefficient) (.value (.predecessor 1 71300 .coefficient)))

def exact71302RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (1)⟩]

theorem exact71302RawTermsValid :
    exact71302RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71302 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7847⟩⟩) exact71302RawTerms (.finite 8192) 71301 .exactZero (none)

def event71303 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6794⟩⟩) 0 ⟨6757⟩ 71292

def event71304 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6794⟩⟩) (.identity (.predecessor 0 71303 .coefficient))

def exact71305RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩]⟩, (1)⟩]

theorem exact71305RawTermsValid :
    exact71305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71305 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6794⟩⟩) exact71305RawTerms .large 71304 .exactZero (none)

def event71306 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7848⟩⟩) 0 ⟨6794⟩ 71305

def event71307 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7848⟩⟩) 1 ⟨7847⟩ 71302

def event71308 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7848⟩⟩) (.product (.predecessor 0 71306 .coefficient) (.predecessor 1 71307 .coefficient) (⟨false, false, none, none, none⟩))

def event71309 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7848⟩⟩, .operator (⟨71305, 0⟩, ⟨71302, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (1)⟩)

def exact71310RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (1)⟩]

theorem exact71310RawTermsValid :
    exact71310RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71310 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7848⟩⟩) exact71310RawTerms .large 71308 .exactZero (none)

def event71311 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13879⟩⟩) 0 ⟨7848⟩ 71310

def event71312 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13879⟩⟩) 1 ⟨13878⟩ 71287

def event71313 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13879⟩⟩) (.sum [.predecessor 0 71311 .coefficient, .predecessor 1 71312 .coefficient])

def exact71314RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71314RawTermsValid :
    exact71314RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71314 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13879⟩⟩) exact71314RawTerms .large 71313 .exactZero (none)

def event71315 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25910⟩⟩) 0 ⟨13879⟩ 71314

def event71316 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25910⟩⟩) 1 ⟨25907⟩ 71271

def event71317 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25910⟩⟩) (.product (.predecessor 0 71315 .coefficient) (.predecessor 1 71316 .coefficient) (⟨false, false, none, none, none⟩))

def event71318 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25910⟩⟩, .operator (⟨71314, 0⟩, ⟨71271, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩, (1)⟩)

def event71319 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25910⟩⟩, .operator (⟨71314, 1⟩, ⟨71271, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩, (-1)⟩)

def event71320 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25910⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25907⟩⟩) ⟨23498⟩ 71268)

def event71321 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25910⟩⟩, .relation 71320 0, ⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨23498⟩⟩]⟩, (-1)⟩)

def exact71322RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨23498⟩⟩]⟩, (-1)⟩]

theorem exact71322RawTermsValid :
    exact71322RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71322 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25910⟩⟩) exact71322RawTerms .large 71317 .exactZero (none)

def event71323 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15698⟩⟩) 0 ⟨13766⟩ 71260

def event71324 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15698⟩⟩) (.authority (.programFamilyFact))

def exact71325RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], []⟩, (1)⟩]

theorem exact71325RawTermsValid :
    exact71325RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71325 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15698⟩⟩) exact71325RawTerms (.finite 12) 71324 .exactZero (none)

def event71326 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15700⟩⟩) 0 ⟨6544⟩ 71282

def event71327 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15700⟩⟩) 1 ⟨15698⟩ 71325

def event71328 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15700⟩⟩) (.product (.predecessor 0 71326 .coefficient) (.predecessor 1 71327 .coefficient) (⟨false, true, none, none, some 1⟩))

def event71329 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15700⟩⟩, .operator (⟨71282, 0⟩, ⟨71325, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact71330RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact71330RawTermsValid :
    exact71330RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71330 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15700⟩⟩) exact71330RawTerms .large 71328 .exactZero (none)

def event71331 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6695⟩⟩) 0 ⟨6689⟩ 71264

def event71332 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6695⟩⟩) (.authority (.operator))

def exact71333RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩]

theorem exact71333RawTermsValid :
    exact71333RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71333 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6695⟩⟩) exact71333RawTerms .large 71332 .exactZero (none)

def event71334 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15701⟩⟩) 0 ⟨6695⟩ 71333

def event71335 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15701⟩⟩) 1 ⟨15700⟩ 71330

def event71336 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15701⟩⟩) (.sum [.predecessor 0 71334 .coefficient, .predecessor 1 71335 .coefficient])

def exact71337RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71337RawTermsValid :
    exact71337RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71337 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15701⟩⟩) exact71337RawTerms .large 71336 .exactZero (none)

def event71338 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25911⟩⟩) 0 ⟨15701⟩ 71337

def event71339 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25911⟩⟩) 1 ⟨25910⟩ 71322

def event71340 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25911⟩⟩) (.sum [.predecessor 0 71338 .coefficient, .predecessor 1 71339 .coefficient])

def exact71341RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨23498⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71341RawTermsValid :
    exact71341RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71341 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25911⟩⟩) exact71341RawTerms .large 71340 .exactZero (none)

def event71342 : Event := .preFoldPolynomial 71341 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨23498⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact71343RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨23498⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event71343 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25911⟩⟩) 71342 exact71343RawTerms .large 71340 .exactZero (none)

def event71344 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨13766⟩⟩) ⟨⟨108⟩, ⟨13⟩, ⟨109⟩⟩ ⟨71178, 71344⟩

def event71345 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19383⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩) (1) 0 2 (.universal 71344 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19380⟩⟩]⟩) (none) 71343)

def event71346 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19383⟩⟩, .relation 71345 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩)

def event71347 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19383⟩⟩, .relation 71345 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩, (-1)⟩)

def event71348 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19383⟩⟩, .relation 71345 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨23498⟩⟩]⟩, (1)⟩)

def event71349 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19383⟩⟩, .relation 71345 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact71350RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨23498⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71350RawTermsValid :
    exact71350RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71350 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19383⟩⟩) exact71350RawTerms .large 71174 (.finite 1811303510016) (some (71176))

def event71351 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25909⟩⟩) 0 ⟨19383⟩ 71350

def event71352 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25909⟩⟩) 1 ⟨25908⟩ 71164

def event71353 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25909⟩⟩) (.sum [.predecessor 0 71351 .coefficient, .predecessor 1 71352 .coefficient])

def event71354 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25909⟩⟩, .operator (⟨71350, 2⟩, ⟨71164, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], [⟨.program ⟨214⟩, ⟨23498⟩⟩]⟩, (-1)⟩)

def event71355 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25909⟩⟩, .operator (⟨71350, 1⟩, ⟨71164, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩, (1)⟩)

def event71356 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25909⟩⟩) (.sum [.result 71350 .summary, .result 71164 .summary])

def exact71357RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71357RawTermsValid :
    exact71357RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71357 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25909⟩⟩) exact71357RawTerms .large 71353 (.finite 352042398396416) (some (71356))

def event71358 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27421⟩⟩) 0 ⟨25909⟩ 71357

def event71359 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27421⟩⟩) 1 ⟨27419⟩ 71080

def event71360 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27421⟩⟩) (.product (.predecessor 0 71358 .coefficient) (.predecessor 1 71359 .coefficient) (⟨false, false, none, none, none⟩))

def event71361 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27421⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩) [⟨.result 71080 .coefficient, false, none⟩])

def event71362 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27421⟩⟩) (.product (.result 71357 .summary) (.transfer 71361) (⟨false, false, none, none, none⟩))

def event71363 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27421⟩⟩, .operator (⟨71357, 0⟩, ⟨71080, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩, (1)⟩)

def event71364 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27421⟩⟩, .operator (⟨71357, 1⟩, ⟨71080, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩, (-1)⟩)

def event71365 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27421⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27419⟩⟩) ⟨24033⟩ 71077)

def event71366 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27421⟩⟩, .relation 71365 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24033⟩⟩]⟩, (-1)⟩)

def exact71367RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24033⟩⟩]⟩, (-1)⟩]

theorem exact71367RawTermsValid :
    exact71367RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71367 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27421⟩⟩) exact71367RawTerms .large 71360 (.finite 1292001234793221062656) (some (71362))

def event71368 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21108⟩⟩) 0 ⟨15699⟩ 3379

def event71369 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21108⟩⟩) (.authority (.relationPreimageSource ⟨39⟩))

def exact71370RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21108⟩⟩]⟩, (1)⟩]

theorem exact71370RawTermsValid :
    exact71370RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71370 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21108⟩⟩) exact71370RawTerms (.finite 136065468) 71369 .exactZero (none)

def event71371 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21110⟩⟩) 0 ⟨21108⟩ 71370

def event71372 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21110⟩⟩) 1 ⟨2348⟩ 4

def event71373 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21110⟩⟩) (.scale (.predecessor 0 71371 .coefficient) (.value (.predecessor 1 71372 .coefficient)))

def exact71374RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21108⟩⟩]⟩, (1)⟩]

theorem exact71374RawTermsValid :
    exact71374RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71374 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21110⟩⟩) exact71374RawTerms (.finite 136065468) 71373 .exactZero (none)

def event71375 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21111⟩⟩) 0 ⟨5535⟩ 65387

def event71376 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21111⟩⟩) 1 ⟨21110⟩ 71374

def event71377 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21111⟩⟩) (.product (.predecessor 0 71375 .coefficient) (.predecessor 1 71376 .coefficient) (⟨false, false, none, none, none⟩))

def event71378 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21111⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21108⟩⟩]⟩) [⟨.result 71370 .coefficient, false, none⟩])

def event71379 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21111⟩⟩) (.product (.result 65387 .summary) (.transfer 71378) (⟨false, false, none, none, none⟩))

def event71380 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21111⟩⟩, .operator (⟨65387, 0⟩, ⟨71374, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21108⟩⟩]⟩, (1)⟩)

def event71381 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21109⟩⟩)

def event71382 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event71383 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event71384 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event71385 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event71386 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event71387 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event71388 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event71389 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event71390 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 71389

def event71391 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 71387

def event71392 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 71390 .coefficient) (.value (.predecessor 1 71391 .coefficient)))

def event71393 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event71394 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 71393

def event71395 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 71385

def event71396 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 71394 .coefficient, .predecessor 1 71395 .coefficient])

def event71397 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event71398 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 71397

def event71399 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 71383

def event71400 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 71399 .coefficient))

def event71401 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event71402 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11297⟩⟩) 0 ⟨5530⟩ 71401

def event71403 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11297⟩⟩) (.authority (.programFamilyFact))

def exact71404RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩], []⟩, (1)⟩]

theorem exact71404RawTermsValid :
    exact71404RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71404 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11297⟩⟩) exact71404RawTerms (.finite 12) 71403 .exactZero (none)

def event71405 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13764⟩⟩) 0 ⟨5530⟩ 71401

def event71406 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13764⟩⟩) (.authority (.programFamilyFact))

def exact71407RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩, (1)⟩]

theorem exact71407RawTermsValid :
    exact71407RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71407 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13764⟩⟩) exact71407RawTerms (.finite 12) 71406 .exactZero (none)

def event71408 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13765⟩⟩) 0 ⟨13764⟩ 71407

def event71409 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13765⟩⟩) 1 ⟨11297⟩ 71404

def event71410 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13765⟩⟩) (.product (.predecessor 0 71408 .coefficient) (.predecessor 1 71409 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event71411 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13765⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩) [⟨.result 71407 .coefficient, true, some 1⟩, ⟨.result 71404 .coefficient, true, some 1⟩])

def event71412 : Event := .survivorFold (1) 71411

def exact71413RawTerms : List Term := []

theorem exact71413RawTermsValid :
    exact71413RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71413 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13765⟩⟩) exact71413RawTerms (.finite 144) 71410 (.finite 144) (some (71411))

def event71414 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13766⟩⟩) 0 ⟨13765⟩ 71413

def event71415 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13766⟩⟩) (.identity (.predecessor 0 71414 .coefficient))

def event71416 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13766⟩⟩) (.finite 144)

def event71417 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15698⟩⟩) 0 ⟨13766⟩ 71416

def event71418 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15698⟩⟩) (.authority (.programFamilyFact))

def exact71419RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], []⟩, (1)⟩]

theorem exact71419RawTermsValid :
    exact71419RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71419 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15698⟩⟩) exact71419RawTerms (.finite 12) 71418 .exactZero (none)

def event71420 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15699⟩⟩) 0 ⟨15698⟩ 71419

def event71421 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15699⟩⟩) (.identity (.predecessor 0 71420 .coefficient))

def event71422 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15699⟩⟩) (.finite 12)

def event71423 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21108⟩⟩) 0 ⟨15699⟩ 71422

def eventLeaf4448 : Array AnnotatedEvent := #[
  { event := event71168
    frameStart := 0 },
  { event := event71169
    frameStart := 0 },
  { event := event71170
    frameStart := 0 },
  { event := event71171
    frameStart := 0 },
  { event := event71172
    frameStart := 0 },
  { event := event71173
    frameStart := 0 },
  { event := event71174
    frameStart := 0 },
  { event := event71175
    frameStart := 0 },
  { event := event71176
    frameStart := 0 },
  { event := event71177
    frameStart := 0 },
  { event := event71178
    frameStart := 71178 },
  { event := event71179
    frameStart := 71178 },
  { event := event71180
    frameStart := 71178 },
  { event := event71181
    frameStart := 71178 },
  { event := event71182
    frameStart := 71178 },
  { event := event71183
    frameStart := 71178 }
]

def eventLeaf4449 : Array AnnotatedEvent := #[
  { event := event71184
    frameStart := 71178 },
  { event := event71185
    frameStart := 71178 },
  { event := event71186
    frameStart := 71178 },
  { event := event71187
    frameStart := 71178 },
  { event := event71188
    frameStart := 71178 },
  { event := event71189
    frameStart := 71178 },
  { event := event71190
    frameStart := 71178 },
  { event := event71191
    frameStart := 71178 },
  { event := event71192
    frameStart := 71178 },
  { event := event71193
    frameStart := 71178 },
  { event := event71194
    frameStart := 71178 },
  { event := event71195
    frameStart := 71178 },
  { event := event71196
    frameStart := 71178 },
  { event := event71197
    frameStart := 71178 },
  { event := event71198
    frameStart := 71178 },
  { event := event71199
    frameStart := 71178 }
]

def eventLeaf4450 : Array AnnotatedEvent := #[
  { event := event71200
    frameStart := 71178 },
  { event := event71201
    frameStart := 71178 },
  { event := event71202
    frameStart := 71178 },
  { event := event71203
    frameStart := 71178 },
  { event := event71204
    frameStart := 71178 },
  { event := event71205
    frameStart := 71178 },
  { event := event71206
    frameStart := 71178 },
  { event := event71207
    frameStart := 71178 },
  { event := event71208
    frameStart := 71178 },
  { event := event71209
    frameStart := 71178 },
  { event := event71210
    frameStart := 71178 },
  { event := event71211
    frameStart := 71178 },
  { event := event71212
    frameStart := 71178 },
  { event := event71213
    frameStart := 71178 },
  { event := event71214
    frameStart := 71178 },
  { event := event71215
    frameStart := 71178 }
]

def eventLeaf4451 : Array AnnotatedEvent := #[
  { event := event71216
    frameStart := 71178 },
  { event := event71217
    frameStart := 71178 },
  { event := event71218
    frameStart := 71178 },
  { event := event71219
    frameStart := 71178 },
  { event := event71220
    frameStart := 71178 },
  { event := event71221
    frameStart := 71178 },
  { event := event71222
    frameStart := 71178 },
  { event := event71223
    frameStart := 71178 },
  { event := event71224
    frameStart := 71178 },
  { event := event71225
    frameStart := 71178 },
  { event := event71226
    frameStart := 71226 },
  { event := event71227
    frameStart := 71226 },
  { event := event71228
    frameStart := 71226 },
  { event := event71229
    frameStart := 71226 },
  { event := event71230
    frameStart := 71226 },
  { event := event71231
    frameStart := 71226 }
]

def eventLeaf4452 : Array AnnotatedEvent := #[
  { event := event71232
    frameStart := 71226 },
  { event := event71233
    frameStart := 71226 },
  { event := event71234
    frameStart := 71226 },
  { event := event71235
    frameStart := 71226 },
  { event := event71236
    frameStart := 71226 },
  { event := event71237
    frameStart := 71226 },
  { event := event71238
    frameStart := 71226 },
  { event := event71239
    frameStart := 71226 },
  { event := event71240
    frameStart := 71226 },
  { event := event71241
    frameStart := 71226 },
  { event := event71242
    frameStart := 71226 },
  { event := event71243
    frameStart := 71226 },
  { event := event71244
    frameStart := 71226 },
  { event := event71245
    frameStart := 71226 },
  { event := event71246
    frameStart := 71226 },
  { event := event71247
    frameStart := 71226 }
]

def eventLeaf4453 : Array AnnotatedEvent := #[
  { event := event71248
    frameStart := 71226 },
  { event := event71249
    frameStart := 71226 },
  { event := event71250
    frameStart := 71226 },
  { event := event71251
    frameStart := 71226 },
  { event := event71252
    frameStart := 71226 },
  { event := event71253
    frameStart := 71226 },
  { event := event71254
    frameStart := 71226 },
  { event := event71255
    frameStart := 71226 },
  { event := event71256
    frameStart := 71226 },
  { event := event71257
    frameStart := 71226 },
  { event := event71258
    frameStart := 71226 },
  { event := event71259
    frameStart := 71226 },
  { event := event71260
    frameStart := 71226 },
  { event := event71261
    frameStart := 71226 },
  { event := event71262
    frameStart := 71226 },
  { event := event71263
    frameStart := 71226 }
]

def eventLeaf4454 : Array AnnotatedEvent := #[
  { event := event71264
    frameStart := 71226 },
  { event := event71265
    frameStart := 71226 },
  { event := event71266
    frameStart := 71226 },
  { event := event71267
    frameStart := 71226 },
  { event := event71268
    frameStart := 71226 },
  { event := event71269
    frameStart := 71226 },
  { event := event71270
    frameStart := 71226 },
  { event := event71271
    frameStart := 71226 },
  { event := event71272
    frameStart := 71226 },
  { event := event71273
    frameStart := 71226 },
  { event := event71274
    frameStart := 71226 },
  { event := event71275
    frameStart := 71226 },
  { event := event71276
    frameStart := 71226 },
  { event := event71277
    frameStart := 71226 },
  { event := event71278
    frameStart := 71226 },
  { event := event71279
    frameStart := 71226 }
]

def eventLeaf4455 : Array AnnotatedEvent := #[
  { event := event71280
    frameStart := 71226 },
  { event := event71281
    frameStart := 71226 },
  { event := event71282
    frameStart := 71226 },
  { event := event71283
    frameStart := 71226 },
  { event := event71284
    frameStart := 71226 },
  { event := event71285
    frameStart := 71226 },
  { event := event71286
    frameStart := 71226 },
  { event := event71287
    frameStart := 71226 },
  { event := event71288
    frameStart := 71226 },
  { event := event71289
    frameStart := 71226 },
  { event := event71290
    frameStart := 71226 },
  { event := event71291
    frameStart := 71226 },
  { event := event71292
    frameStart := 71226 },
  { event := event71293
    frameStart := 71226 },
  { event := event71294
    frameStart := 71226 },
  { event := event71295
    frameStart := 71226 }
]

def eventLeaf4456 : Array AnnotatedEvent := #[
  { event := event71296
    frameStart := 71226 },
  { event := event71297
    frameStart := 71226 },
  { event := event71298
    frameStart := 71226 },
  { event := event71299
    frameStart := 71226 },
  { event := event71300
    frameStart := 71226 },
  { event := event71301
    frameStart := 71226 },
  { event := event71302
    frameStart := 71226 },
  { event := event71303
    frameStart := 71226 },
  { event := event71304
    frameStart := 71226 },
  { event := event71305
    frameStart := 71226 },
  { event := event71306
    frameStart := 71226 },
  { event := event71307
    frameStart := 71226 },
  { event := event71308
    frameStart := 71226 },
  { event := event71309
    frameStart := 71226 },
  { event := event71310
    frameStart := 71226 },
  { event := event71311
    frameStart := 71226 }
]

def eventLeaf4457 : Array AnnotatedEvent := #[
  { event := event71312
    frameStart := 71226 },
  { event := event71313
    frameStart := 71226 },
  { event := event71314
    frameStart := 71226 },
  { event := event71315
    frameStart := 71226 },
  { event := event71316
    frameStart := 71226 },
  { event := event71317
    frameStart := 71226 },
  { event := event71318
    frameStart := 71226 },
  { event := event71319
    frameStart := 71226 },
  { event := event71320
    frameStart := 71226 },
  { event := event71321
    frameStart := 71226 },
  { event := event71322
    frameStart := 71226 },
  { event := event71323
    frameStart := 71226 },
  { event := event71324
    frameStart := 71226 },
  { event := event71325
    frameStart := 71226 },
  { event := event71326
    frameStart := 71226 },
  { event := event71327
    frameStart := 71226 }
]

def eventLeaf4458 : Array AnnotatedEvent := #[
  { event := event71328
    frameStart := 71226 },
  { event := event71329
    frameStart := 71226 },
  { event := event71330
    frameStart := 71226 },
  { event := event71331
    frameStart := 71226 },
  { event := event71332
    frameStart := 71226 },
  { event := event71333
    frameStart := 71226 },
  { event := event71334
    frameStart := 71226 },
  { event := event71335
    frameStart := 71226 },
  { event := event71336
    frameStart := 71226 },
  { event := event71337
    frameStart := 71226 },
  { event := event71338
    frameStart := 71226 },
  { event := event71339
    frameStart := 71226 },
  { event := event71340
    frameStart := 71226 },
  { event := event71341
    frameStart := 71226 },
  { event := event71342
    frameStart := 71226 },
  { event := event71343
    frameStart := 71226 }
]

def eventLeaf4459 : Array AnnotatedEvent := #[
  { event := event71344
    frameStart := 0 },
  { event := event71345
    frameStart := 0 },
  { event := event71346
    frameStart := 0 },
  { event := event71347
    frameStart := 0 },
  { event := event71348
    frameStart := 0 },
  { event := event71349
    frameStart := 0 },
  { event := event71350
    frameStart := 0 },
  { event := event71351
    frameStart := 0 },
  { event := event71352
    frameStart := 0 },
  { event := event71353
    frameStart := 0 },
  { event := event71354
    frameStart := 0 },
  { event := event71355
    frameStart := 0 },
  { event := event71356
    frameStart := 0 },
  { event := event71357
    frameStart := 0 },
  { event := event71358
    frameStart := 0 },
  { event := event71359
    frameStart := 0 }
]

def eventLeaf4460 : Array AnnotatedEvent := #[
  { event := event71360
    frameStart := 0 },
  { event := event71361
    frameStart := 0 },
  { event := event71362
    frameStart := 0 },
  { event := event71363
    frameStart := 0 },
  { event := event71364
    frameStart := 0 },
  { event := event71365
    frameStart := 0 },
  { event := event71366
    frameStart := 0 },
  { event := event71367
    frameStart := 0 },
  { event := event71368
    frameStart := 0 },
  { event := event71369
    frameStart := 0 },
  { event := event71370
    frameStart := 0 },
  { event := event71371
    frameStart := 0 },
  { event := event71372
    frameStart := 0 },
  { event := event71373
    frameStart := 0 },
  { event := event71374
    frameStart := 0 },
  { event := event71375
    frameStart := 0 }
]

def eventLeaf4461 : Array AnnotatedEvent := #[
  { event := event71376
    frameStart := 0 },
  { event := event71377
    frameStart := 0 },
  { event := event71378
    frameStart := 0 },
  { event := event71379
    frameStart := 0 },
  { event := event71380
    frameStart := 0 },
  { event := event71381
    frameStart := 71381 },
  { event := event71382
    frameStart := 71381 },
  { event := event71383
    frameStart := 71381 },
  { event := event71384
    frameStart := 71381 },
  { event := event71385
    frameStart := 71381 },
  { event := event71386
    frameStart := 71381 },
  { event := event71387
    frameStart := 71381 },
  { event := event71388
    frameStart := 71381 },
  { event := event71389
    frameStart := 71381 },
  { event := event71390
    frameStart := 71381 },
  { event := event71391
    frameStart := 71381 }
]

def eventLeaf4462 : Array AnnotatedEvent := #[
  { event := event71392
    frameStart := 71381 },
  { event := event71393
    frameStart := 71381 },
  { event := event71394
    frameStart := 71381 },
  { event := event71395
    frameStart := 71381 },
  { event := event71396
    frameStart := 71381 },
  { event := event71397
    frameStart := 71381 },
  { event := event71398
    frameStart := 71381 },
  { event := event71399
    frameStart := 71381 },
  { event := event71400
    frameStart := 71381 },
  { event := event71401
    frameStart := 71381 },
  { event := event71402
    frameStart := 71381 },
  { event := event71403
    frameStart := 71381 },
  { event := event71404
    frameStart := 71381 },
  { event := event71405
    frameStart := 71381 },
  { event := event71406
    frameStart := 71381 },
  { event := event71407
    frameStart := 71381 }
]

def eventLeaf4463 : Array AnnotatedEvent := #[
  { event := event71408
    frameStart := 71381 },
  { event := event71409
    frameStart := 71381 },
  { event := event71410
    frameStart := 71381 },
  { event := event71411
    frameStart := 71381 },
  { event := event71412
    frameStart := 71381 },
  { event := event71413
    frameStart := 71381 },
  { event := event71414
    frameStart := 71381 },
  { event := event71415
    frameStart := 71381 },
  { event := event71416
    frameStart := 71381 },
  { event := event71417
    frameStart := 71381 },
  { event := event71418
    frameStart := 71381 },
  { event := event71419
    frameStart := 71381 },
  { event := event71420
    frameStart := 71381 },
  { event := event71421
    frameStart := 71381 },
  { event := event71422
    frameStart := 71381 },
  { event := event71423
    frameStart := 71381 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events278
