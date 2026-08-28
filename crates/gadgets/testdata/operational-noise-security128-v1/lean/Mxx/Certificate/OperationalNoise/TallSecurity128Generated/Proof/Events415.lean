import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events415

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event106240 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14496⟩⟩) 0 ⟨5766⟩ 106236

def event106241 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14496⟩⟩) (.authority (.programFamilyFact))

def exact106242RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩], []⟩, (1)⟩]

theorem exact106242RawTermsValid :
    exact106242RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106242 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14496⟩⟩) exact106242RawTerms (.finite 52) 106241 .exactZero (none)

def event106243 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42499⟩⟩) 0 ⟨14496⟩ 106242

def event106244 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42499⟩⟩) 1 ⟨42498⟩ 106239

def event106245 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42499⟩⟩) (.product (.predecessor 0 106243 .coefficient) (.predecessor 1 106244 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event106246 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42499⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], []⟩) [⟨.result 106242 .coefficient, true, some 1⟩, ⟨.result 106239 .coefficient, true, some 1⟩])

def event106247 : Event := .survivorFold (1) 106246

def exact106248RawTerms : List Term := []

theorem exact106248RawTermsValid :
    exact106248RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106248 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42499⟩⟩) exact106248RawTerms (.finite 2704) 106245 (.finite 2704) (some (106246))

def event106249 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42500⟩⟩) 0 ⟨42499⟩ 106248

def event106250 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42500⟩⟩) (.identity (.predecessor 0 106249 .coefficient))

def event106251 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42500⟩⟩) (.finite 2704)

def event106252 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43239⟩⟩) 0 ⟨42500⟩ 106251

def event106253 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43239⟩⟩) (.authority (.relationPreimageSource ⟨52⟩))

def exact106254RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43239⟩⟩]⟩, (1)⟩]

theorem exact106254RawTermsValid :
    exact106254RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106254 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43239⟩⟩) exact106254RawTerms (.finite 5647228698) 106253 .exactZero (none)

def event106255 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact106256RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact106256RawTermsValid :
    exact106256RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106256 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact106256RawTerms .large 106255 .exactZero (none)

def event106257 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43240⟩⟩) 0 ⟨35⟩ 106256

def event106258 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43240⟩⟩) 1 ⟨43239⟩ 106254

def event106259 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43240⟩⟩) (.product (.predecessor 0 106257 .coefficient) (.predecessor 1 106258 .coefficient) (⟨false, false, none, none, none⟩))

def event106260 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43240⟩⟩, .operator (⟨106256, 0⟩, ⟨106254, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43239⟩⟩]⟩, (1)⟩)

def exact106261RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43239⟩⟩]⟩, (1)⟩]

theorem exact106261RawTermsValid :
    exact106261RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106261 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43240⟩⟩) exact106261RawTerms .large 106259 .exactZero (none)

def event106262 : Event := .preFoldPolynomial 106261 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43239⟩⟩]⟩, (1)⟩] .exactZero none

def exact106263RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43239⟩⟩]⟩, (1)⟩]

def event106263 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨43240⟩⟩) 106262 exact106263RawTerms .large 106259 .exactZero (none)

def event106264 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨44314⟩⟩)

def event106265 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event106266 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event106267 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.authority (.operator))

def event106268 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.finite 13)

def event106269 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event106270 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event106271 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event106272 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event106273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 106272

def event106274 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 106270

def event106275 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 106273 .coefficient) (.value (.predecessor 1 106274 .coefficient)))

def event106276 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event106277 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 0 ⟨392⟩ 106276

def event106278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 1 ⟨5754⟩ 106268

def event106279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.sum [.predecessor 0 106277 .coefficient, .predecessor 1 106278 .coefficient])

def event106280 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.finite 655353)

def event106281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 0 ⟨5756⟩ 106280

def event106282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 1 ⟨5426⟩ 106266

def event106283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.identity (.predecessor 1 106282 .coefficient))

def event106284 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.finite 655360)

def event106285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42498⟩⟩) 0 ⟨5766⟩ 106284

def event106286 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42498⟩⟩) (.authority (.programFamilyFact))

def exact106287RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42498⟩⟩], []⟩, (1)⟩]

theorem exact106287RawTermsValid :
    exact106287RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106287 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42498⟩⟩) exact106287RawTerms (.finite 52) 106286 .exactZero (none)

def event106288 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14496⟩⟩) 0 ⟨5766⟩ 106284

def event106289 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14496⟩⟩) (.authority (.programFamilyFact))

def exact106290RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩], []⟩, (1)⟩]

theorem exact106290RawTermsValid :
    exact106290RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106290 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14496⟩⟩) exact106290RawTerms (.finite 52) 106289 .exactZero (none)

def event106291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42499⟩⟩) 0 ⟨14496⟩ 106290

def event106292 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42499⟩⟩) 1 ⟨42498⟩ 106287

def event106293 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42499⟩⟩) (.product (.predecessor 0 106291 .coefficient) (.predecessor 1 106292 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event106294 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42499⟩⟩, .operator (⟨106290, 0⟩, ⟨106287, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], []⟩, (1)⟩)

def exact106295RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], []⟩, (1)⟩]

theorem exact106295RawTermsValid :
    exact106295RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106295 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42499⟩⟩) exact106295RawTerms (.finite 2704) 106293 .exactZero (none)

def event106296 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42500⟩⟩) 0 ⟨42499⟩ 106295

def event106297 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42500⟩⟩) (.identity (.predecessor 0 106296 .coefficient))

def event106298 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42500⟩⟩) (.finite 2704)

def event106299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43794⟩⟩) 0 ⟨42500⟩ 106298

def event106300 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43794⟩⟩) (.authority (.programFamilyFact))

def event106301 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨43794⟩⟩) (.finite 3720)

def event106302 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event106303 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43795⟩⟩) 0 ⟨7177⟩ 106302

def event106304 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43795⟩⟩) 1 ⟨43794⟩ 106301

def event106305 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43795⟩⟩) (.authority (.operator))

def exact106306RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43795⟩⟩]⟩, (1)⟩]

theorem exact106306RawTermsValid :
    exact106306RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106306 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43795⟩⟩) exact106306RawTerms .large 106305 .exactZero (none)

def event106307 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44310⟩⟩) 0 ⟨43795⟩ 106306

def event106308 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44310⟩⟩) (.authority (.operator))

def exact106309RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44310⟩⟩]⟩, (1)⟩]

theorem exact106309RawTermsValid :
    exact106309RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106309 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44310⟩⟩) exact106309RawTerms (.finite 8192) 106308 .exactZero (none)

def event106310 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event106311 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event106312 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44070⟩⟩) 0 ⟨42500⟩ 106298

def event106313 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44070⟩⟩) 1 ⟨136⟩ 106311

def event106314 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44070⟩⟩) (.sum [.predecessor 0 106312 .coefficient, .predecessor 1 106313 .coefficient])

def event106315 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44070⟩⟩) (.finite 2704)

def event106316 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44071⟩⟩) 0 ⟨44070⟩ 106315

def event106317 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44071⟩⟩) (.identity (.predecessor 0 106316 .coefficient))

def exact106318RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], []⟩, (1)⟩]

theorem exact106318RawTermsValid :
    exact106318RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106318 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44071⟩⟩) exact106318RawTerms (.finite 2704) 106317 .exactZero (none)

def event106319 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact106320RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact106320RawTermsValid :
    exact106320RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106320 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact106320RawTerms .large 106319 .exactZero (none)

def event106321 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44072⟩⟩) 0 ⟨6908⟩ 106320

def event106322 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44072⟩⟩) 1 ⟨44071⟩ 106318

def event106323 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44072⟩⟩) (.product (.predecessor 0 106321 .coefficient) (.predecessor 1 106322 .coefficient) (⟨false, false, none, none, none⟩))

def event106324 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44072⟩⟩, .operator (⟨106320, 0⟩, ⟨106318, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact106325RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact106325RawTermsValid :
    exact106325RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106325 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44072⟩⟩) exact106325RawTerms .large 106323 .exactZero (none)

def event106326 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event106327 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event106328 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 106302

def event106329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact106330RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact106330RawTermsValid :
    exact106330RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106330 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact106330RawTerms .large 106329 .exactZero (none)

def event106331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7283⟩⟩) 0 ⟨7178⟩ 106330

def event106332 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7283⟩⟩) (.identity (.predecessor 0 106331 .coefficient))

def exact106333RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩]

theorem exact106333RawTermsValid :
    exact106333RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106333 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7283⟩⟩) exact106333RawTerms .large 106332 .exactZero (none)

def event106334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9559⟩⟩) 0 ⟨7283⟩ 106333

def event106335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9559⟩⟩) (.authority (.operator))

def exact106336RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩]

theorem exact106336RawTermsValid :
    exact106336RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106336 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9559⟩⟩) exact106336RawTerms (.finite 8192) 106335 .exactZero (none)

def event106337 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9560⟩⟩) 0 ⟨9559⟩ 106336

def event106338 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9560⟩⟩) 1 ⟨2370⟩ 106327

def event106339 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9560⟩⟩) (.scale (.predecessor 0 106337 .coefficient) (.value (.predecessor 1 106338 .coefficient)))

def exact106340RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩]

theorem exact106340RawTermsValid :
    exact106340RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106340 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9560⟩⟩) exact106340RawTerms (.finite 8192) 106339 .exactZero (none)

def event106341 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7300⟩⟩) 0 ⟨7178⟩ 106330

def event106342 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7300⟩⟩) (.identity (.predecessor 0 106341 .coefficient))

def exact106343RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩]⟩, (1)⟩]

theorem exact106343RawTermsValid :
    exact106343RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106343 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7300⟩⟩) exact106343RawTerms .large 106342 .exactZero (none)

def event106344 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9561⟩⟩) 0 ⟨7300⟩ 106343

def event106345 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9561⟩⟩) 1 ⟨9560⟩ 106340

def event106346 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9561⟩⟩) (.product (.predecessor 0 106344 .coefficient) (.predecessor 1 106345 .coefficient) (⟨false, false, none, none, none⟩))

def event106347 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9561⟩⟩, .operator (⟨106343, 0⟩, ⟨106340, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩)

def exact106348RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩]

theorem exact106348RawTermsValid :
    exact106348RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106348 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9561⟩⟩) exact106348RawTerms .large 106346 .exactZero (none)

def event106349 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44073⟩⟩) 0 ⟨9561⟩ 106348

def event106350 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44073⟩⟩) 1 ⟨44072⟩ 106325

def event106351 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44073⟩⟩) (.sum [.predecessor 0 106349 .coefficient, .predecessor 1 106350 .coefficient])

def exact106352RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106352RawTermsValid :
    exact106352RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106352 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44073⟩⟩) exact106352RawTerms .large 106351 .exactZero (none)

def event106353 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44313⟩⟩) 0 ⟨44073⟩ 106352

def event106354 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44313⟩⟩) 1 ⟨44310⟩ 106309

def event106355 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44313⟩⟩) (.product (.predecessor 0 106353 .coefficient) (.predecessor 1 106354 .coefficient) (⟨false, false, none, none, none⟩))

def event106356 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44313⟩⟩, .operator (⟨106352, 0⟩, ⟨106309, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩]⟩, (1)⟩)

def event106357 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44313⟩⟩, .operator (⟨106352, 1⟩, ⟨106309, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩]⟩, (-1)⟩)

def event106358 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44313⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44310⟩⟩) ⟨43795⟩ 106306)

def event106359 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44313⟩⟩, .relation 106358 0, ⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], [⟨.program ⟨257⟩, ⟨43795⟩⟩]⟩, (-1)⟩)

def exact106360RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], [⟨.program ⟨257⟩, ⟨43795⟩⟩]⟩, (-1)⟩]

theorem exact106360RawTermsValid :
    exact106360RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106360 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44313⟩⟩) exact106360RawTerms .large 106355 .exactZero (none)

def event106361 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42796⟩⟩) 0 ⟨42500⟩ 106298

def event106362 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42796⟩⟩) (.authority (.programFamilyFact))

def exact106363RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], []⟩, (1)⟩]

theorem exact106363RawTermsValid :
    exact106363RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106363 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42796⟩⟩) exact106363RawTerms (.finite 52) 106362 .exactZero (none)

def event106364 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42798⟩⟩) 0 ⟨6908⟩ 106320

def event106365 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42798⟩⟩) 1 ⟨42796⟩ 106363

def event106366 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42798⟩⟩) (.product (.predecessor 0 106364 .coefficient) (.predecessor 1 106365 .coefficient) (⟨false, true, none, none, some 1⟩))

def event106367 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42798⟩⟩, .operator (⟨106320, 0⟩, ⟨106363, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact106368RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact106368RawTermsValid :
    exact106368RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106368 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42798⟩⟩) exact106368RawTerms .large 106366 .exactZero (none)

def event106369 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7194⟩⟩) 0 ⟨7177⟩ 106302

def event106370 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7194⟩⟩) (.authority (.operator))

def exact106371RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩]

theorem exact106371RawTermsValid :
    exact106371RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106371 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7194⟩⟩) exact106371RawTerms .large 106370 .exactZero (none)

def event106372 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42799⟩⟩) 0 ⟨7194⟩ 106371

def event106373 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42799⟩⟩) 1 ⟨42798⟩ 106368

def event106374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42799⟩⟩) (.sum [.predecessor 0 106372 .coefficient, .predecessor 1 106373 .coefficient])

def exact106375RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106375RawTermsValid :
    exact106375RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106375 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42799⟩⟩) exact106375RawTerms .large 106374 .exactZero (none)

def event106376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44314⟩⟩) 0 ⟨42799⟩ 106375

def event106377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44314⟩⟩) 1 ⟨44313⟩ 106360

def event106378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44314⟩⟩) (.sum [.predecessor 0 106376 .coefficient, .predecessor 1 106377 .coefficient])

def exact106379RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], [⟨.program ⟨257⟩, ⟨43795⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106379RawTermsValid :
    exact106379RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106379 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44314⟩⟩) exact106379RawTerms .large 106378 .exactZero (none)

def event106380 : Event := .preFoldPolynomial 106379 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], [⟨.program ⟨257⟩, ⟨43795⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact106381RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], [⟨.program ⟨257⟩, ⟨43795⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event106381 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨44314⟩⟩) 106380 exact106381RawTerms .large 106378 .exactZero (none)

def event106382 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨42500⟩⟩) ⟨⟨73⟩, ⟨52⟩, ⟨135⟩⟩ ⟨106216, 106382⟩

def event106383 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨43242⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43239⟩⟩]⟩) (1) 0 2 (.universal 106382 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43239⟩⟩]⟩) (none) 106381)

def event106384 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43242⟩⟩, .relation 106383 0, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩)

def event106385 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43242⟩⟩, .relation 106383 1, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩]⟩, (-1)⟩)

def event106386 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43242⟩⟩, .relation 106383 2, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], [⟨.program ⟨257⟩, ⟨43795⟩⟩]⟩, (1)⟩)

def event106387 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43242⟩⟩, .relation 106383 3, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact106388RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], [⟨.program ⟨257⟩, ⟨43795⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106388RawTermsValid :
    exact106388RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106388 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43242⟩⟩) exact106388RawTerms .large 106212 (.finite 202072841853861888) (some (106214))

def event106389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44312⟩⟩) 0 ⟨43242⟩ 106388

def event106390 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44312⟩⟩) 1 ⟨44311⟩ 106202

def event106391 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44312⟩⟩) (.sum [.predecessor 0 106389 .coefficient, .predecessor 1 106390 .coefficient])

def event106392 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44312⟩⟩, .operator (⟨106388, 2⟩, ⟨106202, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], [⟨.program ⟨257⟩, ⟨43795⟩⟩]⟩, (-1)⟩)

def event106393 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44312⟩⟩, .operator (⟨106388, 1⟩, ⟨106202, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44310⟩⟩]⟩, (1)⟩)

def event106394 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44312⟩⟩) (.sum [.result 106388 .summary, .result 106202 .summary])

def exact106395RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106395RawTermsValid :
    exact106395RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106395 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44312⟩⟩) exact106395RawTerms .large 106391 (.finite 2998273677530297008128) (some (106394))

def event106396 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44696⟩⟩) 0 ⟨44312⟩ 106395

def event106397 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44696⟩⟩) 1 ⟨44694⟩ 106118

def event106398 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44696⟩⟩) (.product (.predecessor 0 106396 .coefficient) (.predecessor 1 106397 .coefficient) (⟨false, false, none, none, none⟩))

def event106399 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44696⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩) [⟨.result 106118 .coefficient, false, none⟩])

def event106400 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44696⟩⟩) (.product (.result 106395 .summary) (.transfer 106399) (⟨false, false, none, none, none⟩))

def event106401 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44696⟩⟩, .operator (⟨106395, 0⟩, ⟨106118, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩, (1)⟩)

def event106402 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44696⟩⟩, .operator (⟨106395, 1⟩, ⟨106118, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩, (-1)⟩)

def event106403 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44696⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44694⟩⟩) ⟨43950⟩ 106115)

def event106404 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44696⟩⟩, .relation 106403 0, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨43950⟩⟩]⟩, (-1)⟩)

def exact106405RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨43950⟩⟩]⟩, (-1)⟩]

theorem exact106405RawTermsValid :
    exact106405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106405 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44696⟩⟩) exact106405RawTerms .large 106398 (.finite 32193718473625689247691015454720) (some (106400))

def event106406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43556⟩⟩) 0 ⟨42797⟩ 4645

def event106407 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43556⟩⟩) (.authority (.relationPreimageSource ⟨90⟩))

def exact106408RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43556⟩⟩]⟩, (1)⟩]

theorem exact106408RawTermsValid :
    exact106408RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106408 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43556⟩⟩) exact106408RawTerms (.finite 5647228698) 106407 .exactZero (none)

def event106409 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43558⟩⟩) 0 ⟨43556⟩ 106408

def event106410 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43558⟩⟩) 1 ⟨2370⟩ 4

def event106411 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43558⟩⟩) (.scale (.predecessor 0 106409 .coefficient) (.value (.predecessor 1 106410 .coefficient)))

def exact106412RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43556⟩⟩]⟩, (1)⟩]

theorem exact106412RawTermsValid :
    exact106412RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106412 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43558⟩⟩) exact106412RawTerms (.finite 5647228698) 106411 .exactZero (none)

def event106413 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43559⟩⟩) 0 ⟨5770⟩ 105245

def event106414 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43559⟩⟩) 1 ⟨43558⟩ 106412

def event106415 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43559⟩⟩) (.product (.predecessor 0 106413 .coefficient) (.predecessor 1 106414 .coefficient) (⟨false, false, none, none, none⟩))

def event106416 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43559⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨43556⟩⟩]⟩) [⟨.result 106408 .coefficient, false, none⟩])

def event106417 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43559⟩⟩) (.product (.result 105245 .summary) (.transfer 106416) (⟨false, false, none, none, none⟩))

def event106418 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43559⟩⟩, .operator (⟨105245, 0⟩, ⟨106412, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43556⟩⟩]⟩, (1)⟩)

def event106419 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨43557⟩⟩)

def event106420 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event106421 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event106422 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.authority (.operator))

def event106423 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.finite 13)

def event106424 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event106425 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event106426 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event106427 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event106428 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 106427

def event106429 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 106425

def event106430 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 106428 .coefficient) (.value (.predecessor 1 106429 .coefficient)))

def event106431 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event106432 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 0 ⟨392⟩ 106431

def event106433 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 1 ⟨5754⟩ 106423

def event106434 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.sum [.predecessor 0 106432 .coefficient, .predecessor 1 106433 .coefficient])

def event106435 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.finite 655353)

def event106436 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 0 ⟨5756⟩ 106435

def event106437 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 1 ⟨5426⟩ 106421

def event106438 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.identity (.predecessor 1 106437 .coefficient))

def event106439 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.finite 655360)

def event106440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42498⟩⟩) 0 ⟨5766⟩ 106439

def event106441 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42498⟩⟩) (.authority (.programFamilyFact))

def exact106442RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42498⟩⟩], []⟩, (1)⟩]

theorem exact106442RawTermsValid :
    exact106442RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106442 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42498⟩⟩) exact106442RawTerms (.finite 52) 106441 .exactZero (none)

def event106443 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14496⟩⟩) 0 ⟨5766⟩ 106439

def event106444 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14496⟩⟩) (.authority (.programFamilyFact))

def exact106445RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩], []⟩, (1)⟩]

theorem exact106445RawTermsValid :
    exact106445RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106445 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14496⟩⟩) exact106445RawTerms (.finite 52) 106444 .exactZero (none)

def event106446 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42499⟩⟩) 0 ⟨14496⟩ 106445

def event106447 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42499⟩⟩) 1 ⟨42498⟩ 106442

def event106448 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42499⟩⟩) (.product (.predecessor 0 106446 .coefficient) (.predecessor 1 106447 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event106449 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42499⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], []⟩) [⟨.result 106445 .coefficient, true, some 1⟩, ⟨.result 106442 .coefficient, true, some 1⟩])

def event106450 : Event := .survivorFold (1) 106449

def exact106451RawTerms : List Term := []

theorem exact106451RawTermsValid :
    exact106451RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106451 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42499⟩⟩) exact106451RawTerms (.finite 2704) 106448 (.finite 2704) (some (106449))

def event106452 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42500⟩⟩) 0 ⟨42499⟩ 106451

def event106453 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42500⟩⟩) (.identity (.predecessor 0 106452 .coefficient))

def event106454 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42500⟩⟩) (.finite 2704)

def event106455 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42796⟩⟩) 0 ⟨42500⟩ 106454

def event106456 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42796⟩⟩) (.authority (.programFamilyFact))

def exact106457RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], []⟩, (1)⟩]

theorem exact106457RawTermsValid :
    exact106457RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106457 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42796⟩⟩) exact106457RawTerms (.finite 52) 106456 .exactZero (none)

def event106458 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42797⟩⟩) 0 ⟨42796⟩ 106457

def event106459 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42797⟩⟩) (.identity (.predecessor 0 106458 .coefficient))

def event106460 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42797⟩⟩) (.finite 52)

def event106461 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43556⟩⟩) 0 ⟨42797⟩ 106460

def event106462 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43556⟩⟩) (.authority (.relationPreimageSource ⟨90⟩))

def exact106463RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43556⟩⟩]⟩, (1)⟩]

theorem exact106463RawTermsValid :
    exact106463RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106463 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43556⟩⟩) exact106463RawTerms (.finite 5647228698) 106462 .exactZero (none)

def event106464 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact106465RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact106465RawTermsValid :
    exact106465RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106465 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact106465RawTerms .large 106464 .exactZero (none)

def event106466 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43557⟩⟩) 0 ⟨35⟩ 106465

def event106467 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43557⟩⟩) 1 ⟨43556⟩ 106463

def event106468 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43557⟩⟩) (.product (.predecessor 0 106466 .coefficient) (.predecessor 1 106467 .coefficient) (⟨false, false, none, none, none⟩))

def event106469 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43557⟩⟩, .operator (⟨106465, 0⟩, ⟨106463, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43556⟩⟩]⟩, (1)⟩)

def exact106470RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43556⟩⟩]⟩, (1)⟩]

theorem exact106470RawTermsValid :
    exact106470RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106470 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43557⟩⟩) exact106470RawTerms .large 106468 .exactZero (none)

def event106471 : Event := .preFoldPolynomial 106470 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43556⟩⟩]⟩, (1)⟩] .exactZero none

def exact106472RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43556⟩⟩]⟩, (1)⟩]

def event106472 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨43557⟩⟩) 106471 exact106472RawTerms .large 106468 .exactZero (none)

def event106473 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨44698⟩⟩)

def event106474 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event106475 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event106476 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.authority (.operator))

def event106477 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.finite 13)

def event106478 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event106479 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event106480 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event106481 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event106482 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 106481

def event106483 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 106479

def event106484 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 106482 .coefficient) (.value (.predecessor 1 106483 .coefficient)))

def event106485 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event106486 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 0 ⟨392⟩ 106485

def event106487 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 1 ⟨5754⟩ 106477

def event106488 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.sum [.predecessor 0 106486 .coefficient, .predecessor 1 106487 .coefficient])

def event106489 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.finite 655353)

def event106490 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 0 ⟨5756⟩ 106489

def event106491 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 1 ⟨5426⟩ 106475

def event106492 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.identity (.predecessor 1 106491 .coefficient))

def event106493 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.finite 655360)

def event106494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42498⟩⟩) 0 ⟨5766⟩ 106493

def event106495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42498⟩⟩) (.authority (.programFamilyFact))

def eventLeaf6640 : Array AnnotatedEvent := #[
  { event := event106240
    frameStart := 106216 },
  { event := event106241
    frameStart := 106216 },
  { event := event106242
    frameStart := 106216 },
  { event := event106243
    frameStart := 106216 },
  { event := event106244
    frameStart := 106216 },
  { event := event106245
    frameStart := 106216 },
  { event := event106246
    frameStart := 106216 },
  { event := event106247
    frameStart := 106216 },
  { event := event106248
    frameStart := 106216 },
  { event := event106249
    frameStart := 106216 },
  { event := event106250
    frameStart := 106216 },
  { event := event106251
    frameStart := 106216 },
  { event := event106252
    frameStart := 106216 },
  { event := event106253
    frameStart := 106216 },
  { event := event106254
    frameStart := 106216 },
  { event := event106255
    frameStart := 106216 }
]

def eventLeaf6641 : Array AnnotatedEvent := #[
  { event := event106256
    frameStart := 106216 },
  { event := event106257
    frameStart := 106216 },
  { event := event106258
    frameStart := 106216 },
  { event := event106259
    frameStart := 106216 },
  { event := event106260
    frameStart := 106216 },
  { event := event106261
    frameStart := 106216 },
  { event := event106262
    frameStart := 106216 },
  { event := event106263
    frameStart := 106216 },
  { event := event106264
    frameStart := 106264 },
  { event := event106265
    frameStart := 106264 },
  { event := event106266
    frameStart := 106264 },
  { event := event106267
    frameStart := 106264 },
  { event := event106268
    frameStart := 106264 },
  { event := event106269
    frameStart := 106264 },
  { event := event106270
    frameStart := 106264 },
  { event := event106271
    frameStart := 106264 }
]

def eventLeaf6642 : Array AnnotatedEvent := #[
  { event := event106272
    frameStart := 106264 },
  { event := event106273
    frameStart := 106264 },
  { event := event106274
    frameStart := 106264 },
  { event := event106275
    frameStart := 106264 },
  { event := event106276
    frameStart := 106264 },
  { event := event106277
    frameStart := 106264 },
  { event := event106278
    frameStart := 106264 },
  { event := event106279
    frameStart := 106264 },
  { event := event106280
    frameStart := 106264 },
  { event := event106281
    frameStart := 106264 },
  { event := event106282
    frameStart := 106264 },
  { event := event106283
    frameStart := 106264 },
  { event := event106284
    frameStart := 106264 },
  { event := event106285
    frameStart := 106264 },
  { event := event106286
    frameStart := 106264 },
  { event := event106287
    frameStart := 106264 }
]

def eventLeaf6643 : Array AnnotatedEvent := #[
  { event := event106288
    frameStart := 106264 },
  { event := event106289
    frameStart := 106264 },
  { event := event106290
    frameStart := 106264 },
  { event := event106291
    frameStart := 106264 },
  { event := event106292
    frameStart := 106264 },
  { event := event106293
    frameStart := 106264 },
  { event := event106294
    frameStart := 106264 },
  { event := event106295
    frameStart := 106264 },
  { event := event106296
    frameStart := 106264 },
  { event := event106297
    frameStart := 106264 },
  { event := event106298
    frameStart := 106264 },
  { event := event106299
    frameStart := 106264 },
  { event := event106300
    frameStart := 106264 },
  { event := event106301
    frameStart := 106264 },
  { event := event106302
    frameStart := 106264 },
  { event := event106303
    frameStart := 106264 }
]

def eventLeaf6644 : Array AnnotatedEvent := #[
  { event := event106304
    frameStart := 106264 },
  { event := event106305
    frameStart := 106264 },
  { event := event106306
    frameStart := 106264 },
  { event := event106307
    frameStart := 106264 },
  { event := event106308
    frameStart := 106264 },
  { event := event106309
    frameStart := 106264 },
  { event := event106310
    frameStart := 106264 },
  { event := event106311
    frameStart := 106264 },
  { event := event106312
    frameStart := 106264 },
  { event := event106313
    frameStart := 106264 },
  { event := event106314
    frameStart := 106264 },
  { event := event106315
    frameStart := 106264 },
  { event := event106316
    frameStart := 106264 },
  { event := event106317
    frameStart := 106264 },
  { event := event106318
    frameStart := 106264 },
  { event := event106319
    frameStart := 106264 }
]

def eventLeaf6645 : Array AnnotatedEvent := #[
  { event := event106320
    frameStart := 106264 },
  { event := event106321
    frameStart := 106264 },
  { event := event106322
    frameStart := 106264 },
  { event := event106323
    frameStart := 106264 },
  { event := event106324
    frameStart := 106264 },
  { event := event106325
    frameStart := 106264 },
  { event := event106326
    frameStart := 106264 },
  { event := event106327
    frameStart := 106264 },
  { event := event106328
    frameStart := 106264 },
  { event := event106329
    frameStart := 106264 },
  { event := event106330
    frameStart := 106264 },
  { event := event106331
    frameStart := 106264 },
  { event := event106332
    frameStart := 106264 },
  { event := event106333
    frameStart := 106264 },
  { event := event106334
    frameStart := 106264 },
  { event := event106335
    frameStart := 106264 }
]

def eventLeaf6646 : Array AnnotatedEvent := #[
  { event := event106336
    frameStart := 106264 },
  { event := event106337
    frameStart := 106264 },
  { event := event106338
    frameStart := 106264 },
  { event := event106339
    frameStart := 106264 },
  { event := event106340
    frameStart := 106264 },
  { event := event106341
    frameStart := 106264 },
  { event := event106342
    frameStart := 106264 },
  { event := event106343
    frameStart := 106264 },
  { event := event106344
    frameStart := 106264 },
  { event := event106345
    frameStart := 106264 },
  { event := event106346
    frameStart := 106264 },
  { event := event106347
    frameStart := 106264 },
  { event := event106348
    frameStart := 106264 },
  { event := event106349
    frameStart := 106264 },
  { event := event106350
    frameStart := 106264 },
  { event := event106351
    frameStart := 106264 }
]

def eventLeaf6647 : Array AnnotatedEvent := #[
  { event := event106352
    frameStart := 106264 },
  { event := event106353
    frameStart := 106264 },
  { event := event106354
    frameStart := 106264 },
  { event := event106355
    frameStart := 106264 },
  { event := event106356
    frameStart := 106264 },
  { event := event106357
    frameStart := 106264 },
  { event := event106358
    frameStart := 106264 },
  { event := event106359
    frameStart := 106264 },
  { event := event106360
    frameStart := 106264 },
  { event := event106361
    frameStart := 106264 },
  { event := event106362
    frameStart := 106264 },
  { event := event106363
    frameStart := 106264 },
  { event := event106364
    frameStart := 106264 },
  { event := event106365
    frameStart := 106264 },
  { event := event106366
    frameStart := 106264 },
  { event := event106367
    frameStart := 106264 }
]

def eventLeaf6648 : Array AnnotatedEvent := #[
  { event := event106368
    frameStart := 106264 },
  { event := event106369
    frameStart := 106264 },
  { event := event106370
    frameStart := 106264 },
  { event := event106371
    frameStart := 106264 },
  { event := event106372
    frameStart := 106264 },
  { event := event106373
    frameStart := 106264 },
  { event := event106374
    frameStart := 106264 },
  { event := event106375
    frameStart := 106264 },
  { event := event106376
    frameStart := 106264 },
  { event := event106377
    frameStart := 106264 },
  { event := event106378
    frameStart := 106264 },
  { event := event106379
    frameStart := 106264 },
  { event := event106380
    frameStart := 106264 },
  { event := event106381
    frameStart := 106264 },
  { event := event106382
    frameStart := 0 },
  { event := event106383
    frameStart := 0 }
]

def eventLeaf6649 : Array AnnotatedEvent := #[
  { event := event106384
    frameStart := 0 },
  { event := event106385
    frameStart := 0 },
  { event := event106386
    frameStart := 0 },
  { event := event106387
    frameStart := 0 },
  { event := event106388
    frameStart := 0 },
  { event := event106389
    frameStart := 0 },
  { event := event106390
    frameStart := 0 },
  { event := event106391
    frameStart := 0 },
  { event := event106392
    frameStart := 0 },
  { event := event106393
    frameStart := 0 },
  { event := event106394
    frameStart := 0 },
  { event := event106395
    frameStart := 0 },
  { event := event106396
    frameStart := 0 },
  { event := event106397
    frameStart := 0 },
  { event := event106398
    frameStart := 0 },
  { event := event106399
    frameStart := 0 }
]

def eventLeaf6650 : Array AnnotatedEvent := #[
  { event := event106400
    frameStart := 0 },
  { event := event106401
    frameStart := 0 },
  { event := event106402
    frameStart := 0 },
  { event := event106403
    frameStart := 0 },
  { event := event106404
    frameStart := 0 },
  { event := event106405
    frameStart := 0 },
  { event := event106406
    frameStart := 0 },
  { event := event106407
    frameStart := 0 },
  { event := event106408
    frameStart := 0 },
  { event := event106409
    frameStart := 0 },
  { event := event106410
    frameStart := 0 },
  { event := event106411
    frameStart := 0 },
  { event := event106412
    frameStart := 0 },
  { event := event106413
    frameStart := 0 },
  { event := event106414
    frameStart := 0 },
  { event := event106415
    frameStart := 0 }
]

def eventLeaf6651 : Array AnnotatedEvent := #[
  { event := event106416
    frameStart := 0 },
  { event := event106417
    frameStart := 0 },
  { event := event106418
    frameStart := 0 },
  { event := event106419
    frameStart := 106419 },
  { event := event106420
    frameStart := 106419 },
  { event := event106421
    frameStart := 106419 },
  { event := event106422
    frameStart := 106419 },
  { event := event106423
    frameStart := 106419 },
  { event := event106424
    frameStart := 106419 },
  { event := event106425
    frameStart := 106419 },
  { event := event106426
    frameStart := 106419 },
  { event := event106427
    frameStart := 106419 },
  { event := event106428
    frameStart := 106419 },
  { event := event106429
    frameStart := 106419 },
  { event := event106430
    frameStart := 106419 },
  { event := event106431
    frameStart := 106419 }
]

def eventLeaf6652 : Array AnnotatedEvent := #[
  { event := event106432
    frameStart := 106419 },
  { event := event106433
    frameStart := 106419 },
  { event := event106434
    frameStart := 106419 },
  { event := event106435
    frameStart := 106419 },
  { event := event106436
    frameStart := 106419 },
  { event := event106437
    frameStart := 106419 },
  { event := event106438
    frameStart := 106419 },
  { event := event106439
    frameStart := 106419 },
  { event := event106440
    frameStart := 106419 },
  { event := event106441
    frameStart := 106419 },
  { event := event106442
    frameStart := 106419 },
  { event := event106443
    frameStart := 106419 },
  { event := event106444
    frameStart := 106419 },
  { event := event106445
    frameStart := 106419 },
  { event := event106446
    frameStart := 106419 },
  { event := event106447
    frameStart := 106419 }
]

def eventLeaf6653 : Array AnnotatedEvent := #[
  { event := event106448
    frameStart := 106419 },
  { event := event106449
    frameStart := 106419 },
  { event := event106450
    frameStart := 106419 },
  { event := event106451
    frameStart := 106419 },
  { event := event106452
    frameStart := 106419 },
  { event := event106453
    frameStart := 106419 },
  { event := event106454
    frameStart := 106419 },
  { event := event106455
    frameStart := 106419 },
  { event := event106456
    frameStart := 106419 },
  { event := event106457
    frameStart := 106419 },
  { event := event106458
    frameStart := 106419 },
  { event := event106459
    frameStart := 106419 },
  { event := event106460
    frameStart := 106419 },
  { event := event106461
    frameStart := 106419 },
  { event := event106462
    frameStart := 106419 },
  { event := event106463
    frameStart := 106419 }
]

def eventLeaf6654 : Array AnnotatedEvent := #[
  { event := event106464
    frameStart := 106419 },
  { event := event106465
    frameStart := 106419 },
  { event := event106466
    frameStart := 106419 },
  { event := event106467
    frameStart := 106419 },
  { event := event106468
    frameStart := 106419 },
  { event := event106469
    frameStart := 106419 },
  { event := event106470
    frameStart := 106419 },
  { event := event106471
    frameStart := 106419 },
  { event := event106472
    frameStart := 106419 },
  { event := event106473
    frameStart := 106473 },
  { event := event106474
    frameStart := 106473 },
  { event := event106475
    frameStart := 106473 },
  { event := event106476
    frameStart := 106473 },
  { event := event106477
    frameStart := 106473 },
  { event := event106478
    frameStart := 106473 },
  { event := event106479
    frameStart := 106473 }
]

def eventLeaf6655 : Array AnnotatedEvent := #[
  { event := event106480
    frameStart := 106473 },
  { event := event106481
    frameStart := 106473 },
  { event := event106482
    frameStart := 106473 },
  { event := event106483
    frameStart := 106473 },
  { event := event106484
    frameStart := 106473 },
  { event := event106485
    frameStart := 106473 },
  { event := event106486
    frameStart := 106473 },
  { event := event106487
    frameStart := 106473 },
  { event := event106488
    frameStart := 106473 },
  { event := event106489
    frameStart := 106473 },
  { event := event106490
    frameStart := 106473 },
  { event := event106491
    frameStart := 106473 },
  { event := event106492
    frameStart := 106473 },
  { event := event106493
    frameStart := 106473 },
  { event := event106494
    frameStart := 106473 },
  { event := event106495
    frameStart := 106473 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events415
