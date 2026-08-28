import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events364

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event93184 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event93185 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event93186 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event93187 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event93188 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event93189 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 93188

def event93190 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 93186

def event93191 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 93189 .coefficient) (.value (.predecessor 1 93190 .coefficient)))

def event93192 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event93193 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 93192

def event93194 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 93184

def event93195 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 93193 .coefficient, .predecessor 1 93194 .coefficient])

def event93196 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event93197 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 93196

def event93198 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 93182

def event93199 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 93198 .coefficient))

def event93200 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event93201 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11133⟩⟩) 0 ⟨5536⟩ 93200

def event93202 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11133⟩⟩) (.authority (.programFamilyFact))

def exact93203RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩], []⟩, (1)⟩]

theorem exact93203RawTermsValid :
    exact93203RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93203 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11133⟩⟩) exact93203RawTerms (.finite 6) 93202 .exactZero (none)

def event93204 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12163⟩⟩) 0 ⟨5536⟩ 93200

def event93205 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12163⟩⟩) (.authority (.programFamilyFact))

def exact93206RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩, (1)⟩]

theorem exact93206RawTermsValid :
    exact93206RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93206 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12163⟩⟩) exact93206RawTerms (.finite 6) 93205 .exactZero (none)

def event93207 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12164⟩⟩) 0 ⟨12163⟩ 93206

def event93208 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12164⟩⟩) 1 ⟨11133⟩ 93203

def event93209 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12164⟩⟩) (.product (.predecessor 0 93207 .coefficient) (.predecessor 1 93208 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event93210 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12164⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩) [⟨.result 93206 .coefficient, true, some 1⟩, ⟨.result 93203 .coefficient, true, some 1⟩])

def event93211 : Event := .survivorFold (1) 93210

def exact93212RawTerms : List Term := []

theorem exact93212RawTermsValid :
    exact93212RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93212 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12164⟩⟩) exact93212RawTerms (.finite 36) 93209 (.finite 36) (some (93210))

def event93213 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12165⟩⟩) 0 ⟨12164⟩ 93212

def event93214 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12165⟩⟩) (.identity (.predecessor 0 93213 .coefficient))

def event93215 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12165⟩⟩) (.finite 36)

def event93216 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15422⟩⟩) 0 ⟨12165⟩ 93215

def event93217 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15422⟩⟩) (.authority (.programFamilyFact))

def exact93218RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], []⟩, (1)⟩]

theorem exact93218RawTermsValid :
    exact93218RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93218 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15422⟩⟩) exact93218RawTerms (.finite 6) 93217 .exactZero (none)

def event93219 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15423⟩⟩) 0 ⟨15422⟩ 93218

def event93220 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15423⟩⟩) (.identity (.predecessor 0 93219 .coefficient))

def event93221 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15423⟩⟩) (.finite 6)

def event93222 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20752⟩⟩) 0 ⟨15423⟩ 93221

def event93223 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20752⟩⟩) (.authority (.relationPreimageSource ⟨34⟩))

def exact93224RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20752⟩⟩]⟩, (1)⟩]

theorem exact93224RawTermsValid :
    exact93224RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93224 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20752⟩⟩) exact93224RawTerms (.finite 136065468) 93223 .exactZero (none)

def event93225 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact93226RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact93226RawTermsValid :
    exact93226RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93226 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact93226RawTerms .large 93225 .exactZero (none)

def event93227 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20753⟩⟩) 0 ⟨6⟩ 93226

def event93228 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20753⟩⟩) 1 ⟨20752⟩ 93224

def event93229 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20753⟩⟩) (.product (.predecessor 0 93227 .coefficient) (.predecessor 1 93228 .coefficient) (⟨false, false, none, none, none⟩))

def event93230 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20753⟩⟩, .operator (⟨93226, 0⟩, ⟨93224, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20752⟩⟩]⟩, (1)⟩)

def exact93231RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20752⟩⟩]⟩, (1)⟩]

theorem exact93231RawTermsValid :
    exact93231RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93231 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20753⟩⟩) exact93231RawTerms .large 93229 .exactZero (none)

def event93232 : Event := .preFoldPolynomial 93231 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20752⟩⟩]⟩, (1)⟩] .exactZero none

def exact93233RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20752⟩⟩]⟩, (1)⟩]

def event93233 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20753⟩⟩) 93232 exact93233RawTerms .large 93229 .exactZero (none)

def event93234 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26997⟩⟩)

def event93235 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event93236 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event93237 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event93238 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event93239 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event93240 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event93241 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event93242 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event93243 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 93242

def event93244 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 93240

def event93245 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 93243 .coefficient) (.value (.predecessor 1 93244 .coefficient)))

def event93246 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event93247 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 93246

def event93248 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 93238

def event93249 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 93247 .coefficient, .predecessor 1 93248 .coefficient])

def event93250 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event93251 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 93250

def event93252 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 93236

def event93253 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 93252 .coefficient))

def event93254 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event93255 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11133⟩⟩) 0 ⟨5536⟩ 93254

def event93256 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11133⟩⟩) (.authority (.programFamilyFact))

def exact93257RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩], []⟩, (1)⟩]

theorem exact93257RawTermsValid :
    exact93257RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93257 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11133⟩⟩) exact93257RawTerms (.finite 6) 93256 .exactZero (none)

def event93258 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12163⟩⟩) 0 ⟨5536⟩ 93254

def event93259 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12163⟩⟩) (.authority (.programFamilyFact))

def exact93260RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩, (1)⟩]

theorem exact93260RawTermsValid :
    exact93260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93260 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12163⟩⟩) exact93260RawTerms (.finite 6) 93259 .exactZero (none)

def event93261 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12164⟩⟩) 0 ⟨12163⟩ 93260

def event93262 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12164⟩⟩) 1 ⟨11133⟩ 93257

def event93263 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12164⟩⟩) (.product (.predecessor 0 93261 .coefficient) (.predecessor 1 93262 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event93264 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12164⟩⟩, .operator (⟨93260, 0⟩, ⟨93257, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩, (1)⟩)

def exact93265RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], []⟩, (1)⟩]

theorem exact93265RawTermsValid :
    exact93265RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93265 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12164⟩⟩) exact93265RawTerms (.finite 36) 93263 .exactZero (none)

def event93266 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12165⟩⟩) 0 ⟨12164⟩ 93265

def event93267 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12165⟩⟩) (.identity (.predecessor 0 93266 .coefficient))

def event93268 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12165⟩⟩) (.finite 36)

def event93269 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15422⟩⟩) 0 ⟨12165⟩ 93268

def event93270 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15422⟩⟩) (.authority (.programFamilyFact))

def exact93271RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], []⟩, (1)⟩]

theorem exact93271RawTermsValid :
    exact93271RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93271 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15422⟩⟩) exact93271RawTerms (.finite 6) 93270 .exactZero (none)

def event93272 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15423⟩⟩) 0 ⟨15422⟩ 93271

def event93273 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15423⟩⟩) (.identity (.predecessor 0 93272 .coefficient))

def event93274 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15423⟩⟩) (.finite 6)

def event93275 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23908⟩⟩) 0 ⟨15423⟩ 93274

def event93276 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23908⟩⟩) (.authority (.programFamilyFact))

def event93277 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23908⟩⟩) (.finite 3720)

def event93278 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event93279 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23909⟩⟩) 0 ⟨6689⟩ 93278

def event93280 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23909⟩⟩) 1 ⟨23908⟩ 93277

def event93281 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23909⟩⟩) (.authority (.operator))

def exact93282RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23909⟩⟩]⟩, (1)⟩]

theorem exact93282RawTermsValid :
    exact93282RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93282 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23909⟩⟩) exact93282RawTerms .large 93281 .exactZero (none)

def event93283 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26991⟩⟩) 0 ⟨23909⟩ 93282

def event93284 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26991⟩⟩) (.authority (.operator))

def exact93285RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26991⟩⟩]⟩, (1)⟩]

theorem exact93285RawTermsValid :
    exact93285RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93285 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26991⟩⟩) exact93285RawTerms (.finite 8192) 93284 .exactZero (none)

def event93286 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event93287 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event93288 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15462⟩⟩) 0 ⟨15423⟩ 93274

def event93289 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15462⟩⟩) 1 ⟨110⟩ 93287

def event93290 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15462⟩⟩) (.sum [.predecessor 0 93288 .coefficient, .predecessor 1 93289 .coefficient])

def event93291 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15462⟩⟩) (.finite 6)

def event93292 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15463⟩⟩) 0 ⟨15462⟩ 93291

def event93293 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15463⟩⟩) (.identity (.predecessor 0 93292 .coefficient))

def exact93294RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], []⟩, (1)⟩]

theorem exact93294RawTermsValid :
    exact93294RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93294 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15463⟩⟩) exact93294RawTerms (.finite 6) 93293 .exactZero (none)

def event93295 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact93296RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact93296RawTermsValid :
    exact93296RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93296 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact93296RawTerms .large 93295 .exactZero (none)

def event93297 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15464⟩⟩) 0 ⟨6544⟩ 93296

def event93298 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15464⟩⟩) 1 ⟨15463⟩ 93294

def event93299 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15464⟩⟩) (.product (.predecessor 0 93297 .coefficient) (.predecessor 1 93298 .coefficient) (⟨false, false, none, none, none⟩))

def event93300 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15464⟩⟩, .operator (⟨93296, 0⟩, ⟨93294, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact93301RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact93301RawTermsValid :
    exact93301RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93301 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15464⟩⟩) exact93301RawTerms .large 93299 .exactZero (none)

def event93302 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6693⟩⟩) 0 ⟨6689⟩ 93278

def event93303 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6693⟩⟩) (.authority (.operator))

def exact93304RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩]

theorem exact93304RawTermsValid :
    exact93304RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93304 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6693⟩⟩) exact93304RawTerms .large 93303 .exactZero (none)

def event93305 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15465⟩⟩) 0 ⟨6693⟩ 93304

def event93306 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15465⟩⟩) 1 ⟨15464⟩ 93301

def event93307 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15465⟩⟩) (.sum [.predecessor 0 93305 .coefficient, .predecessor 1 93306 .coefficient])

def exact93308RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact93308RawTermsValid :
    exact93308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93308 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15465⟩⟩) exact93308RawTerms .large 93307 .exactZero (none)

def event93309 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26992⟩⟩) 0 ⟨15465⟩ 93308

def event93310 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26992⟩⟩) 1 ⟨26991⟩ 93285

def event93311 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26992⟩⟩) (.product (.predecessor 0 93309 .coefficient) (.predecessor 1 93310 .coefficient) (⟨false, false, none, none, none⟩))

def event93312 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26992⟩⟩, .operator (⟨93308, 0⟩, ⟨93285, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26991⟩⟩]⟩, (1)⟩)

def event93313 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26992⟩⟩, .operator (⟨93308, 1⟩, ⟨93285, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26991⟩⟩]⟩, (-1)⟩)

def event93314 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26992⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26991⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26991⟩⟩) ⟨23909⟩ 93282)

def event93315 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26992⟩⟩, .relation 93314 0, ⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨23909⟩⟩]⟩, (-1)⟩)

def exact93316RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26991⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨23909⟩⟩]⟩, (-1)⟩]

theorem exact93316RawTermsValid :
    exact93316RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93316 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26992⟩⟩) exact93316RawTerms .large 93311 .exactZero (none)

def event93317 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15516⟩⟩) 0 ⟨15423⟩ 93274

def event93318 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15516⟩⟩) (.authority (.programFamilyFact))

def exact93319RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15516⟩⟩], []⟩, (1)⟩]

theorem exact93319RawTermsValid :
    exact93319RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93319 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15516⟩⟩) exact93319RawTerms (.finite 6) 93318 .exactZero (none)

def event93320 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15519⟩⟩) 0 ⟨6544⟩ 93296

def event93321 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15519⟩⟩) 1 ⟨15516⟩ 93319

def event93322 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15519⟩⟩) (.product (.predecessor 0 93320 .coefficient) (.predecessor 1 93321 .coefficient) (⟨false, true, none, none, some 1⟩))

def event93323 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15519⟩⟩, .operator (⟨93296, 0⟩, ⟨93319, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15516⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact93324RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15516⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact93324RawTermsValid :
    exact93324RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93324 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15519⟩⟩) exact93324RawTerms .large 93322 .exactZero (none)

def event93325 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6714⟩⟩) 0 ⟨6689⟩ 93278

def event93326 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6714⟩⟩) (.authority (.operator))

def exact93327RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩]⟩, (1)⟩]

theorem exact93327RawTermsValid :
    exact93327RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93327 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6714⟩⟩) exact93327RawTerms .large 93326 .exactZero (none)

def event93328 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15520⟩⟩) 0 ⟨6714⟩ 93327

def event93329 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15520⟩⟩) 1 ⟨15519⟩ 93324

def event93330 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15520⟩⟩) (.sum [.predecessor 0 93328 .coefficient, .predecessor 1 93329 .coefficient])

def exact93331RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15516⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact93331RawTermsValid :
    exact93331RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93331 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15520⟩⟩) exact93331RawTerms .large 93330 .exactZero (none)

def event93332 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26997⟩⟩) 0 ⟨15520⟩ 93331

def event93333 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26997⟩⟩) 1 ⟨26992⟩ 93316

def event93334 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26997⟩⟩) (.sum [.predecessor 0 93332 .coefficient, .predecessor 1 93333 .coefficient])

def exact93335RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26991⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨23909⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15516⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact93335RawTermsValid :
    exact93335RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93335 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26997⟩⟩) exact93335RawTerms .large 93334 .exactZero (none)

def event93336 : Event := .preFoldPolynomial 93335 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26991⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨23909⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15516⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact93337RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26991⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨23909⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15516⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event93337 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26997⟩⟩) 93336 exact93337RawTerms .large 93334 .exactZero (none)

def event93338 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15423⟩⟩) ⟨⟨127⟩, ⟨34⟩, ⟨109⟩⟩ ⟨93180, 93338⟩

def event93339 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20755⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20752⟩⟩]⟩) (1) 0 2 (.universal 93338 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20752⟩⟩]⟩) (none) 93337)

def event93340 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20755⟩⟩, .relation 93339 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩]⟩, (1)⟩)

def event93341 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20755⟩⟩, .relation 93339 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26991⟩⟩]⟩, (-1)⟩)

def event93342 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20755⟩⟩, .relation 93339 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨23909⟩⟩]⟩, (1)⟩)

def event93343 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20755⟩⟩, .relation 93339 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15516⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact93344RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26991⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨23909⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15516⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact93344RawTermsValid :
    exact93344RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93344 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20755⟩⟩) exact93344RawTerms .large 93176 (.finite 1811303510016) (some (93178))

def event93345 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26994⟩⟩) 0 ⟨20755⟩ 93344

def event93346 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26994⟩⟩) 1 ⟨26993⟩ 93166

def event93347 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26994⟩⟩) (.sum [.predecessor 0 93345 .coefficient, .predecessor 1 93346 .coefficient])

def event93348 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26994⟩⟩, .operator (⟨93344, 0⟩, ⟨93166, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26991⟩⟩]⟩, (1)⟩)

def event93349 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26994⟩⟩, .operator (⟨93344, 2⟩, ⟨93166, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15422⟩⟩], [⟨.program ⟨214⟩, ⟨23909⟩⟩]⟩, (-1)⟩)

def event93350 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26994⟩⟩) (.sum [.result 93344 .summary, .result 93166 .summary])

def exact93351RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15516⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact93351RawTermsValid :
    exact93351RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93351 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26994⟩⟩) exact93351RawTerms .large 93347 (.finite 1291933999269462814720) (some (93350))

def event93352 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26995⟩⟩) 0 ⟨26994⟩ 93351

def event93353 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26995⟩⟩) 1 ⟨6656⟩ 5799

def event93354 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26995⟩⟩) (.product (.predecessor 0 93352 .coefficient) (.predecessor 1 93353 .coefficient) (⟨false, false, none, none, none⟩))

def event93355 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26995⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩) [⟨.result 5795 .coefficient, false, none⟩])

def event93356 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26995⟩⟩) (.product (.result 93351 .summary) (.transfer 93355) (⟨false, false, none, none, none⟩))

def event93357 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26995⟩⟩, .operator (⟨93351, 0⟩, ⟨5799, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩)

def event93358 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26995⟩⟩, .operator (⟨93351, 1⟩, ⟨5799, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15516⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (-1)⟩)

def event93359 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26995⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15516⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6655⟩⟩) ⟨6599⟩ 5792)

def event93360 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26995⟩⟩, .relation 93359 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15516⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact93361RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15516⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact93361RawTermsValid :
    exact93361RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93361 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26995⟩⟩) exact93361RawTerms .large 93354 (.finite 4741418448262916841427435520) (some (93356))

def event93362 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23846⟩⟩) 0 ⟨6689⟩ 5477

def event93363 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23846⟩⟩) 1 ⟨23845⟩ 87114

def event93364 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23846⟩⟩) (.authority (.operator))

def exact93365RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23846⟩⟩]⟩, (1)⟩]

theorem exact93365RawTermsValid :
    exact93365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93365 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23846⟩⟩) exact93365RawTerms .large 93364 .exactZero (none)

def event93366 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26774⟩⟩) 0 ⟨23846⟩ 93365

def event93367 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26774⟩⟩) (.authority (.operator))

def exact93368RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26774⟩⟩]⟩, (1)⟩]

theorem exact93368RawTermsValid :
    exact93368RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93368 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26774⟩⟩) exact93368RawTerms (.finite 8192) 93367 .exactZero (none)

def event93369 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26776⟩⟩) 0 ⟨25067⟩ 87396

def event93370 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26776⟩⟩) 1 ⟨26774⟩ 93368

def event93371 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26776⟩⟩) (.product (.predecessor 0 93369 .coefficient) (.predecessor 1 93370 .coefficient) (⟨false, false, none, none, none⟩))

def event93372 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26776⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26774⟩⟩]⟩) [⟨.result 93368 .coefficient, false, none⟩])

def event93373 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26776⟩⟩) (.product (.result 87396 .summary) (.transfer 93372) (⟨false, false, none, none, none⟩))

def event93374 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26776⟩⟩, .operator (⟨87396, 0⟩, ⟨93368, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26774⟩⟩]⟩, (1)⟩)

def event93375 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26776⟩⟩, .operator (⟨87396, 1⟩, ⟨93368, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26774⟩⟩]⟩, (-1)⟩)

def event93376 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26776⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26774⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26774⟩⟩) ⟨23846⟩ 93365)

def event93377 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26776⟩⟩, .relation 93376 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨23846⟩⟩]⟩, (-1)⟩)

def exact93378RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26774⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15114⟩⟩], [⟨.program ⟨214⟩, ⟨23846⟩⟩]⟩, (-1)⟩]

theorem exact93378RawTermsValid :
    exact93378RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93378 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26776⟩⟩) exact93378RawTerms .large 93371 (.finite 1291911585013138718720) (some (93373))

def event93379 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20608⟩⟩) 0 ⟨15115⟩ 4190

def event93380 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20608⟩⟩) (.authority (.relationPreimageSource ⟨31⟩))

def exact93381RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20608⟩⟩]⟩, (1)⟩]

theorem exact93381RawTermsValid :
    exact93381RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93381 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20608⟩⟩) exact93381RawTerms (.finite 136065468) 93380 .exactZero (none)

def event93382 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20610⟩⟩) 0 ⟨20608⟩ 93381

def event93383 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20610⟩⟩) 1 ⟨2348⟩ 4

def event93384 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20610⟩⟩) (.scale (.predecessor 0 93382 .coefficient) (.value (.predecessor 1 93383 .coefficient)))

def exact93385RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20608⟩⟩]⟩, (1)⟩]

theorem exact93385RawTermsValid :
    exact93385RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93385 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20610⟩⟩) exact93385RawTerms (.finite 136065468) 93384 .exactZero (none)

def event93386 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20611⟩⟩) 0 ⟨5541⟩ 80012

def event93387 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20611⟩⟩) 1 ⟨20610⟩ 93385

def event93388 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20611⟩⟩) (.product (.predecessor 0 93386 .coefficient) (.predecessor 1 93387 .coefficient) (⟨false, false, none, none, none⟩))

def event93389 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20611⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20608⟩⟩]⟩) [⟨.result 93381 .coefficient, false, none⟩])

def event93390 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20611⟩⟩) (.product (.result 80012 .summary) (.transfer 93389) (⟨false, false, none, none, none⟩))

def event93391 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20611⟩⟩, .operator (⟨80012, 0⟩, ⟨93385, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20608⟩⟩]⟩, (1)⟩)

def event93392 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20609⟩⟩)

def event93393 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event93394 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event93395 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event93396 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event93397 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event93398 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event93399 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event93400 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event93401 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 93400

def event93402 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 93398

def event93403 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 93401 .coefficient) (.value (.predecessor 1 93402 .coefficient)))

def event93404 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event93405 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 93404

def event93406 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 93396

def event93407 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 93405 .coefficient, .predecessor 1 93406 .coefficient])

def event93408 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event93409 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 93408

def event93410 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 93394

def event93411 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 93410 .coefficient))

def event93412 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event93413 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10977⟩⟩) 0 ⟨5536⟩ 93412

def event93414 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10977⟩⟩) (.authority (.programFamilyFact))

def exact93415RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10977⟩⟩], []⟩, (1)⟩]

theorem exact93415RawTermsValid :
    exact93415RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93415 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10977⟩⟩) exact93415RawTerms (.finite 4) 93414 .exactZero (none)

def event93416 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10842⟩⟩) 0 ⟨5536⟩ 93412

def event93417 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10842⟩⟩) (.authority (.programFamilyFact))

def exact93418RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩], []⟩, (1)⟩]

theorem exact93418RawTermsValid :
    exact93418RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93418 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10842⟩⟩) exact93418RawTerms (.finite 4) 93417 .exactZero (none)

def event93419 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10978⟩⟩) 0 ⟨10842⟩ 93418

def event93420 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10978⟩⟩) 1 ⟨10977⟩ 93415

def event93421 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10978⟩⟩) (.product (.predecessor 0 93419 .coefficient) (.predecessor 1 93420 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event93422 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10978⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10842⟩⟩, ⟨.program ⟨214⟩, ⟨10977⟩⟩], []⟩) [⟨.result 93418 .coefficient, true, some 1⟩, ⟨.result 93415 .coefficient, true, some 1⟩])

def event93423 : Event := .survivorFold (1) 93422

def exact93424RawTerms : List Term := []

theorem exact93424RawTermsValid :
    exact93424RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93424 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10978⟩⟩) exact93424RawTerms (.finite 16) 93421 (.finite 16) (some (93422))

def event93425 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10979⟩⟩) 0 ⟨10978⟩ 93424

def event93426 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10979⟩⟩) (.identity (.predecessor 0 93425 .coefficient))

def event93427 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10979⟩⟩) (.finite 16)

def event93428 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15114⟩⟩) 0 ⟨10979⟩ 93427

def event93429 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15114⟩⟩) (.authority (.programFamilyFact))

def exact93430RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15114⟩⟩], []⟩, (1)⟩]

theorem exact93430RawTermsValid :
    exact93430RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93430 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15114⟩⟩) exact93430RawTerms (.finite 4) 93429 .exactZero (none)

def event93431 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15115⟩⟩) 0 ⟨15114⟩ 93430

def event93432 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15115⟩⟩) (.identity (.predecessor 0 93431 .coefficient))

def event93433 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15115⟩⟩) (.finite 4)

def event93434 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20608⟩⟩) 0 ⟨15115⟩ 93433

def event93435 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20608⟩⟩) (.authority (.relationPreimageSource ⟨31⟩))

def exact93436RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20608⟩⟩]⟩, (1)⟩]

theorem exact93436RawTermsValid :
    exact93436RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93436 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20608⟩⟩) exact93436RawTerms (.finite 136065468) 93435 .exactZero (none)

def event93437 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact93438RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact93438RawTermsValid :
    exact93438RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93438 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact93438RawTerms .large 93437 .exactZero (none)

def event93439 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20609⟩⟩) 0 ⟨6⟩ 93438

def eventLeaf5824 : Array AnnotatedEvent := #[
  { event := event93184
    frameStart := 93180 },
  { event := event93185
    frameStart := 93180 },
  { event := event93186
    frameStart := 93180 },
  { event := event93187
    frameStart := 93180 },
  { event := event93188
    frameStart := 93180 },
  { event := event93189
    frameStart := 93180 },
  { event := event93190
    frameStart := 93180 },
  { event := event93191
    frameStart := 93180 },
  { event := event93192
    frameStart := 93180 },
  { event := event93193
    frameStart := 93180 },
  { event := event93194
    frameStart := 93180 },
  { event := event93195
    frameStart := 93180 },
  { event := event93196
    frameStart := 93180 },
  { event := event93197
    frameStart := 93180 },
  { event := event93198
    frameStart := 93180 },
  { event := event93199
    frameStart := 93180 }
]

def eventLeaf5825 : Array AnnotatedEvent := #[
  { event := event93200
    frameStart := 93180 },
  { event := event93201
    frameStart := 93180 },
  { event := event93202
    frameStart := 93180 },
  { event := event93203
    frameStart := 93180 },
  { event := event93204
    frameStart := 93180 },
  { event := event93205
    frameStart := 93180 },
  { event := event93206
    frameStart := 93180 },
  { event := event93207
    frameStart := 93180 },
  { event := event93208
    frameStart := 93180 },
  { event := event93209
    frameStart := 93180 },
  { event := event93210
    frameStart := 93180 },
  { event := event93211
    frameStart := 93180 },
  { event := event93212
    frameStart := 93180 },
  { event := event93213
    frameStart := 93180 },
  { event := event93214
    frameStart := 93180 },
  { event := event93215
    frameStart := 93180 }
]

def eventLeaf5826 : Array AnnotatedEvent := #[
  { event := event93216
    frameStart := 93180 },
  { event := event93217
    frameStart := 93180 },
  { event := event93218
    frameStart := 93180 },
  { event := event93219
    frameStart := 93180 },
  { event := event93220
    frameStart := 93180 },
  { event := event93221
    frameStart := 93180 },
  { event := event93222
    frameStart := 93180 },
  { event := event93223
    frameStart := 93180 },
  { event := event93224
    frameStart := 93180 },
  { event := event93225
    frameStart := 93180 },
  { event := event93226
    frameStart := 93180 },
  { event := event93227
    frameStart := 93180 },
  { event := event93228
    frameStart := 93180 },
  { event := event93229
    frameStart := 93180 },
  { event := event93230
    frameStart := 93180 },
  { event := event93231
    frameStart := 93180 }
]

def eventLeaf5827 : Array AnnotatedEvent := #[
  { event := event93232
    frameStart := 93180 },
  { event := event93233
    frameStart := 93180 },
  { event := event93234
    frameStart := 93234 },
  { event := event93235
    frameStart := 93234 },
  { event := event93236
    frameStart := 93234 },
  { event := event93237
    frameStart := 93234 },
  { event := event93238
    frameStart := 93234 },
  { event := event93239
    frameStart := 93234 },
  { event := event93240
    frameStart := 93234 },
  { event := event93241
    frameStart := 93234 },
  { event := event93242
    frameStart := 93234 },
  { event := event93243
    frameStart := 93234 },
  { event := event93244
    frameStart := 93234 },
  { event := event93245
    frameStart := 93234 },
  { event := event93246
    frameStart := 93234 },
  { event := event93247
    frameStart := 93234 }
]

def eventLeaf5828 : Array AnnotatedEvent := #[
  { event := event93248
    frameStart := 93234 },
  { event := event93249
    frameStart := 93234 },
  { event := event93250
    frameStart := 93234 },
  { event := event93251
    frameStart := 93234 },
  { event := event93252
    frameStart := 93234 },
  { event := event93253
    frameStart := 93234 },
  { event := event93254
    frameStart := 93234 },
  { event := event93255
    frameStart := 93234 },
  { event := event93256
    frameStart := 93234 },
  { event := event93257
    frameStart := 93234 },
  { event := event93258
    frameStart := 93234 },
  { event := event93259
    frameStart := 93234 },
  { event := event93260
    frameStart := 93234 },
  { event := event93261
    frameStart := 93234 },
  { event := event93262
    frameStart := 93234 },
  { event := event93263
    frameStart := 93234 }
]

def eventLeaf5829 : Array AnnotatedEvent := #[
  { event := event93264
    frameStart := 93234 },
  { event := event93265
    frameStart := 93234 },
  { event := event93266
    frameStart := 93234 },
  { event := event93267
    frameStart := 93234 },
  { event := event93268
    frameStart := 93234 },
  { event := event93269
    frameStart := 93234 },
  { event := event93270
    frameStart := 93234 },
  { event := event93271
    frameStart := 93234 },
  { event := event93272
    frameStart := 93234 },
  { event := event93273
    frameStart := 93234 },
  { event := event93274
    frameStart := 93234 },
  { event := event93275
    frameStart := 93234 },
  { event := event93276
    frameStart := 93234 },
  { event := event93277
    frameStart := 93234 },
  { event := event93278
    frameStart := 93234 },
  { event := event93279
    frameStart := 93234 }
]

def eventLeaf5830 : Array AnnotatedEvent := #[
  { event := event93280
    frameStart := 93234 },
  { event := event93281
    frameStart := 93234 },
  { event := event93282
    frameStart := 93234 },
  { event := event93283
    frameStart := 93234 },
  { event := event93284
    frameStart := 93234 },
  { event := event93285
    frameStart := 93234 },
  { event := event93286
    frameStart := 93234 },
  { event := event93287
    frameStart := 93234 },
  { event := event93288
    frameStart := 93234 },
  { event := event93289
    frameStart := 93234 },
  { event := event93290
    frameStart := 93234 },
  { event := event93291
    frameStart := 93234 },
  { event := event93292
    frameStart := 93234 },
  { event := event93293
    frameStart := 93234 },
  { event := event93294
    frameStart := 93234 },
  { event := event93295
    frameStart := 93234 }
]

def eventLeaf5831 : Array AnnotatedEvent := #[
  { event := event93296
    frameStart := 93234 },
  { event := event93297
    frameStart := 93234 },
  { event := event93298
    frameStart := 93234 },
  { event := event93299
    frameStart := 93234 },
  { event := event93300
    frameStart := 93234 },
  { event := event93301
    frameStart := 93234 },
  { event := event93302
    frameStart := 93234 },
  { event := event93303
    frameStart := 93234 },
  { event := event93304
    frameStart := 93234 },
  { event := event93305
    frameStart := 93234 },
  { event := event93306
    frameStart := 93234 },
  { event := event93307
    frameStart := 93234 },
  { event := event93308
    frameStart := 93234 },
  { event := event93309
    frameStart := 93234 },
  { event := event93310
    frameStart := 93234 },
  { event := event93311
    frameStart := 93234 }
]

def eventLeaf5832 : Array AnnotatedEvent := #[
  { event := event93312
    frameStart := 93234 },
  { event := event93313
    frameStart := 93234 },
  { event := event93314
    frameStart := 93234 },
  { event := event93315
    frameStart := 93234 },
  { event := event93316
    frameStart := 93234 },
  { event := event93317
    frameStart := 93234 },
  { event := event93318
    frameStart := 93234 },
  { event := event93319
    frameStart := 93234 },
  { event := event93320
    frameStart := 93234 },
  { event := event93321
    frameStart := 93234 },
  { event := event93322
    frameStart := 93234 },
  { event := event93323
    frameStart := 93234 },
  { event := event93324
    frameStart := 93234 },
  { event := event93325
    frameStart := 93234 },
  { event := event93326
    frameStart := 93234 },
  { event := event93327
    frameStart := 93234 }
]

def eventLeaf5833 : Array AnnotatedEvent := #[
  { event := event93328
    frameStart := 93234 },
  { event := event93329
    frameStart := 93234 },
  { event := event93330
    frameStart := 93234 },
  { event := event93331
    frameStart := 93234 },
  { event := event93332
    frameStart := 93234 },
  { event := event93333
    frameStart := 93234 },
  { event := event93334
    frameStart := 93234 },
  { event := event93335
    frameStart := 93234 },
  { event := event93336
    frameStart := 93234 },
  { event := event93337
    frameStart := 93234 },
  { event := event93338
    frameStart := 0 },
  { event := event93339
    frameStart := 0 },
  { event := event93340
    frameStart := 0 },
  { event := event93341
    frameStart := 0 },
  { event := event93342
    frameStart := 0 },
  { event := event93343
    frameStart := 0 }
]

def eventLeaf5834 : Array AnnotatedEvent := #[
  { event := event93344
    frameStart := 0 },
  { event := event93345
    frameStart := 0 },
  { event := event93346
    frameStart := 0 },
  { event := event93347
    frameStart := 0 },
  { event := event93348
    frameStart := 0 },
  { event := event93349
    frameStart := 0 },
  { event := event93350
    frameStart := 0 },
  { event := event93351
    frameStart := 0 },
  { event := event93352
    frameStart := 0 },
  { event := event93353
    frameStart := 0 },
  { event := event93354
    frameStart := 0 },
  { event := event93355
    frameStart := 0 },
  { event := event93356
    frameStart := 0 },
  { event := event93357
    frameStart := 0 },
  { event := event93358
    frameStart := 0 },
  { event := event93359
    frameStart := 0 }
]

def eventLeaf5835 : Array AnnotatedEvent := #[
  { event := event93360
    frameStart := 0 },
  { event := event93361
    frameStart := 0 },
  { event := event93362
    frameStart := 0 },
  { event := event93363
    frameStart := 0 },
  { event := event93364
    frameStart := 0 },
  { event := event93365
    frameStart := 0 },
  { event := event93366
    frameStart := 0 },
  { event := event93367
    frameStart := 0 },
  { event := event93368
    frameStart := 0 },
  { event := event93369
    frameStart := 0 },
  { event := event93370
    frameStart := 0 },
  { event := event93371
    frameStart := 0 },
  { event := event93372
    frameStart := 0 },
  { event := event93373
    frameStart := 0 },
  { event := event93374
    frameStart := 0 },
  { event := event93375
    frameStart := 0 }
]

def eventLeaf5836 : Array AnnotatedEvent := #[
  { event := event93376
    frameStart := 0 },
  { event := event93377
    frameStart := 0 },
  { event := event93378
    frameStart := 0 },
  { event := event93379
    frameStart := 0 },
  { event := event93380
    frameStart := 0 },
  { event := event93381
    frameStart := 0 },
  { event := event93382
    frameStart := 0 },
  { event := event93383
    frameStart := 0 },
  { event := event93384
    frameStart := 0 },
  { event := event93385
    frameStart := 0 },
  { event := event93386
    frameStart := 0 },
  { event := event93387
    frameStart := 0 },
  { event := event93388
    frameStart := 0 },
  { event := event93389
    frameStart := 0 },
  { event := event93390
    frameStart := 0 },
  { event := event93391
    frameStart := 0 }
]

def eventLeaf5837 : Array AnnotatedEvent := #[
  { event := event93392
    frameStart := 93392 },
  { event := event93393
    frameStart := 93392 },
  { event := event93394
    frameStart := 93392 },
  { event := event93395
    frameStart := 93392 },
  { event := event93396
    frameStart := 93392 },
  { event := event93397
    frameStart := 93392 },
  { event := event93398
    frameStart := 93392 },
  { event := event93399
    frameStart := 93392 },
  { event := event93400
    frameStart := 93392 },
  { event := event93401
    frameStart := 93392 },
  { event := event93402
    frameStart := 93392 },
  { event := event93403
    frameStart := 93392 },
  { event := event93404
    frameStart := 93392 },
  { event := event93405
    frameStart := 93392 },
  { event := event93406
    frameStart := 93392 },
  { event := event93407
    frameStart := 93392 }
]

def eventLeaf5838 : Array AnnotatedEvent := #[
  { event := event93408
    frameStart := 93392 },
  { event := event93409
    frameStart := 93392 },
  { event := event93410
    frameStart := 93392 },
  { event := event93411
    frameStart := 93392 },
  { event := event93412
    frameStart := 93392 },
  { event := event93413
    frameStart := 93392 },
  { event := event93414
    frameStart := 93392 },
  { event := event93415
    frameStart := 93392 },
  { event := event93416
    frameStart := 93392 },
  { event := event93417
    frameStart := 93392 },
  { event := event93418
    frameStart := 93392 },
  { event := event93419
    frameStart := 93392 },
  { event := event93420
    frameStart := 93392 },
  { event := event93421
    frameStart := 93392 },
  { event := event93422
    frameStart := 93392 },
  { event := event93423
    frameStart := 93392 }
]

def eventLeaf5839 : Array AnnotatedEvent := #[
  { event := event93424
    frameStart := 93392 },
  { event := event93425
    frameStart := 93392 },
  { event := event93426
    frameStart := 93392 },
  { event := event93427
    frameStart := 93392 },
  { event := event93428
    frameStart := 93392 },
  { event := event93429
    frameStart := 93392 },
  { event := event93430
    frameStart := 93392 },
  { event := event93431
    frameStart := 93392 },
  { event := event93432
    frameStart := 93392 },
  { event := event93433
    frameStart := 93392 },
  { event := event93434
    frameStart := 93392 },
  { event := event93435
    frameStart := 93392 },
  { event := event93436
    frameStart := 93392 },
  { event := event93437
    frameStart := 93392 },
  { event := event93438
    frameStart := 93392 },
  { event := event93439
    frameStart := 93392 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events364
