import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1087

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event278272 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event278273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 278272

def event278274 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 278270

def event278275 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 278273 .coefficient) (.value (.predecessor 1 278274 .coefficient)))

def event278276 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event278277 : Event := .predecessor (⟨.program ⟨257⟩, ⟨394⟩⟩) 0 ⟨392⟩ 278276

def event278278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨394⟩⟩) 1 ⟨387⟩ 278268

def event278279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨394⟩⟩) (.sum [.predecessor 0 278277 .coefficient, .predecessor 1 278278 .coefficient])

def event278280 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨394⟩⟩) (.finite 655342)

def event278281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5445⟩⟩) 0 ⟨394⟩ 278280

def event278282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5445⟩⟩) 1 ⟨5426⟩ 278266

def event278283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5445⟩⟩) (.identity (.predecessor 1 278282 .coefficient))

def event278284 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5445⟩⟩) (.finite 655360)

def event278285 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25390⟩⟩) 0 ⟨5445⟩ 278284

def event278286 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25390⟩⟩) (.authority (.programFamilyFact))

def exact278287RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25390⟩⟩], []⟩, (1)⟩]

theorem exact278287RawTermsValid :
    exact278287RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278287 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25390⟩⟩) exact278287RawTerms (.finite 22) 278286 .exactZero (none)

def event278288 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62240⟩⟩) 0 ⟨5445⟩ 278284

def event278289 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62240⟩⟩) (.authority (.programFamilyFact))

def exact278290RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62240⟩⟩], []⟩, (1)⟩]

theorem exact278290RawTermsValid :
    exact278290RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278290 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62240⟩⟩) exact278290RawTerms (.finite 22) 278289 .exactZero (none)

def event278291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62241⟩⟩) 0 ⟨62240⟩ 278290

def event278292 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62241⟩⟩) 1 ⟨25390⟩ 278287

def event278293 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62241⟩⟩) (.product (.predecessor 0 278291 .coefficient) (.predecessor 1 278292 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event278294 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62241⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], []⟩) [⟨.result 278290 .coefficient, true, some 1⟩, ⟨.result 278287 .coefficient, true, some 1⟩])

def event278295 : Event := .survivorFold (1) 278294

def exact278296RawTerms : List Term := []

theorem exact278296RawTermsValid :
    exact278296RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278296 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62241⟩⟩) exact278296RawTerms (.finite 484) 278293 (.finite 484) (some (278294))

def event278297 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62242⟩⟩) 0 ⟨62241⟩ 278296

def event278298 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62242⟩⟩) (.identity (.predecessor 0 278297 .coefficient))

def event278299 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨62242⟩⟩) (.finite 484)

def event278300 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62742⟩⟩) 0 ⟨62242⟩ 278299

def event278301 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62742⟩⟩) (.authority (.programFamilyFact))

def exact278302RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62742⟩⟩], []⟩, (1)⟩]

theorem exact278302RawTermsValid :
    exact278302RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278302 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62742⟩⟩) exact278302RawTerms (.finite 22) 278301 .exactZero (none)

def event278303 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62743⟩⟩) 0 ⟨62742⟩ 278302

def event278304 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62743⟩⟩) (.identity (.predecessor 0 278303 .coefficient))

def event278305 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨62743⟩⟩) (.finite 22)

def event278306 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63506⟩⟩) 0 ⟨62743⟩ 278305

def event278307 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63506⟩⟩) (.authority (.relationPreimageSource ⟨73⟩))

def exact278308RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨63506⟩⟩]⟩, (1)⟩]

theorem exact278308RawTermsValid :
    exact278308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278308 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63506⟩⟩) exact278308RawTerms (.finite 5647228698) 278307 .exactZero (none)

def event278309 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact278310RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact278310RawTermsValid :
    exact278310RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278310 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact278310RawTerms .large 278309 .exactZero (none)

def event278311 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63507⟩⟩) 0 ⟨35⟩ 278310

def event278312 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63507⟩⟩) 1 ⟨63506⟩ 278308

def event278313 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63507⟩⟩) (.product (.predecessor 0 278311 .coefficient) (.predecessor 1 278312 .coefficient) (⟨false, false, none, none, none⟩))

def event278314 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63507⟩⟩, .operator (⟨278310, 0⟩, ⟨278308, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63506⟩⟩]⟩, (1)⟩)

def exact278315RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63506⟩⟩]⟩, (1)⟩]

theorem exact278315RawTermsValid :
    exact278315RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278315 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63507⟩⟩) exact278315RawTerms .large 278313 .exactZero (none)

def event278316 : Event := .preFoldPolynomial 278315 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63506⟩⟩]⟩, (1)⟩] .exactZero none

def exact278317RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63506⟩⟩]⟩, (1)⟩]

def event278317 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨63507⟩⟩) 278316 exact278317RawTerms .large 278313 .exactZero (none)

def event278318 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨64614⟩⟩)

def event278319 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event278320 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event278321 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨387⟩⟩) (.authority (.operator))

def event278322 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨387⟩⟩) (.finite 2)

def event278323 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event278324 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event278325 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event278326 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event278327 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 278326

def event278328 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 278324

def event278329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 278327 .coefficient) (.value (.predecessor 1 278328 .coefficient)))

def event278330 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event278331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨394⟩⟩) 0 ⟨392⟩ 278330

def event278332 : Event := .predecessor (⟨.program ⟨257⟩, ⟨394⟩⟩) 1 ⟨387⟩ 278322

def event278333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨394⟩⟩) (.sum [.predecessor 0 278331 .coefficient, .predecessor 1 278332 .coefficient])

def event278334 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨394⟩⟩) (.finite 655342)

def event278335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5445⟩⟩) 0 ⟨394⟩ 278334

def event278336 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5445⟩⟩) 1 ⟨5426⟩ 278320

def event278337 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5445⟩⟩) (.identity (.predecessor 1 278336 .coefficient))

def event278338 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5445⟩⟩) (.finite 655360)

def event278339 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25390⟩⟩) 0 ⟨5445⟩ 278338

def event278340 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25390⟩⟩) (.authority (.programFamilyFact))

def exact278341RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25390⟩⟩], []⟩, (1)⟩]

theorem exact278341RawTermsValid :
    exact278341RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278341 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25390⟩⟩) exact278341RawTerms (.finite 22) 278340 .exactZero (none)

def event278342 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62240⟩⟩) 0 ⟨5445⟩ 278338

def event278343 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62240⟩⟩) (.authority (.programFamilyFact))

def exact278344RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62240⟩⟩], []⟩, (1)⟩]

theorem exact278344RawTermsValid :
    exact278344RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278344 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62240⟩⟩) exact278344RawTerms (.finite 22) 278343 .exactZero (none)

def event278345 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62241⟩⟩) 0 ⟨62240⟩ 278344

def event278346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62241⟩⟩) 1 ⟨25390⟩ 278341

def event278347 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62241⟩⟩) (.product (.predecessor 0 278345 .coefficient) (.predecessor 1 278346 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event278348 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62241⟩⟩, .operator (⟨278344, 0⟩, ⟨278341, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], []⟩, (1)⟩)

def exact278349RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25390⟩⟩, ⟨.program ⟨257⟩, ⟨62240⟩⟩], []⟩, (1)⟩]

theorem exact278349RawTermsValid :
    exact278349RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278349 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62241⟩⟩) exact278349RawTerms (.finite 484) 278347 .exactZero (none)

def event278350 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62242⟩⟩) 0 ⟨62241⟩ 278349

def event278351 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62242⟩⟩) (.identity (.predecessor 0 278350 .coefficient))

def event278352 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨62242⟩⟩) (.finite 484)

def event278353 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62742⟩⟩) 0 ⟨62242⟩ 278352

def event278354 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62742⟩⟩) (.authority (.programFamilyFact))

def exact278355RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62742⟩⟩], []⟩, (1)⟩]

theorem exact278355RawTermsValid :
    exact278355RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278355 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62742⟩⟩) exact278355RawTerms (.finite 22) 278354 .exactZero (none)

def event278356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62743⟩⟩) 0 ⟨62742⟩ 278355

def event278357 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62743⟩⟩) (.identity (.predecessor 0 278356 .coefficient))

def event278358 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨62743⟩⟩) (.finite 22)

def event278359 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64004⟩⟩) 0 ⟨62743⟩ 278358

def event278360 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64004⟩⟩) (.authority (.programFamilyFact))

def event278361 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨64004⟩⟩) (.finite 3720)

def event278362 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event278363 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64005⟩⟩) 0 ⟨7177⟩ 278362

def event278364 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64005⟩⟩) 1 ⟨64004⟩ 278361

def event278365 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64005⟩⟩) (.authority (.operator))

def exact278366RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨64005⟩⟩]⟩, (1)⟩]

theorem exact278366RawTermsValid :
    exact278366RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278366 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64005⟩⟩) exact278366RawTerms .large 278365 .exactZero (none)

def event278367 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64608⟩⟩) 0 ⟨64005⟩ 278366

def event278368 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64608⟩⟩) (.authority (.operator))

def exact278369RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨64608⟩⟩]⟩, (1)⟩]

theorem exact278369RawTermsValid :
    exact278369RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278369 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64608⟩⟩) exact278369RawTerms (.finite 8192) 278368 .exactZero (none)

def event278370 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event278371 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event278372 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64254⟩⟩) 0 ⟨62743⟩ 278358

def event278373 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64254⟩⟩) 1 ⟨136⟩ 278371

def event278374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64254⟩⟩) (.sum [.predecessor 0 278372 .coefficient, .predecessor 1 278373 .coefficient])

def event278375 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨64254⟩⟩) (.finite 22)

def event278376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64255⟩⟩) 0 ⟨64254⟩ 278375

def event278377 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64255⟩⟩) (.identity (.predecessor 0 278376 .coefficient))

def exact278378RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62742⟩⟩], []⟩, (1)⟩]

theorem exact278378RawTermsValid :
    exact278378RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278378 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64255⟩⟩) exact278378RawTerms (.finite 22) 278377 .exactZero (none)

def event278379 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact278380RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact278380RawTermsValid :
    exact278380RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278380 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact278380RawTerms .large 278379 .exactZero (none)

def event278381 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64256⟩⟩) 0 ⟨6908⟩ 278380

def event278382 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64256⟩⟩) 1 ⟨64255⟩ 278378

def event278383 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64256⟩⟩) (.product (.predecessor 0 278381 .coefficient) (.predecessor 1 278382 .coefficient) (⟨false, false, none, none, none⟩))

def event278384 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64256⟩⟩, .operator (⟨278380, 0⟩, ⟨278378, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨62742⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact278385RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62742⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact278385RawTermsValid :
    exact278385RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278385 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64256⟩⟩) exact278385RawTerms .large 278383 .exactZero (none)

def event278386 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7187⟩⟩) 0 ⟨7177⟩ 278362

def event278387 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7187⟩⟩) (.authority (.operator))

def exact278388RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩, (1)⟩]

theorem exact278388RawTermsValid :
    exact278388RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278388 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7187⟩⟩) exact278388RawTerms .large 278387 .exactZero (none)

def event278389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64257⟩⟩) 0 ⟨7187⟩ 278388

def event278390 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64257⟩⟩) 1 ⟨64256⟩ 278385

def event278391 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64257⟩⟩) (.sum [.predecessor 0 278389 .coefficient, .predecessor 1 278390 .coefficient])

def exact278392RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62742⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact278392RawTermsValid :
    exact278392RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278392 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64257⟩⟩) exact278392RawTerms .large 278391 .exactZero (none)

def event278393 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64609⟩⟩) 0 ⟨64257⟩ 278392

def event278394 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64609⟩⟩) 1 ⟨64608⟩ 278369

def event278395 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64609⟩⟩) (.product (.predecessor 0 278393 .coefficient) (.predecessor 1 278394 .coefficient) (⟨false, false, none, none, none⟩))

def event278396 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64609⟩⟩, .operator (⟨278392, 0⟩, ⟨278369, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64608⟩⟩]⟩, (1)⟩)

def event278397 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64609⟩⟩, .operator (⟨278392, 1⟩, ⟨278369, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨62742⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64608⟩⟩]⟩, (-1)⟩)

def event278398 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨64609⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨62742⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64608⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64608⟩⟩) ⟨64005⟩ 278366)

def event278399 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64609⟩⟩, .relation 278398 0, ⟨[⟨.program ⟨257⟩, ⟨62742⟩⟩], [⟨.program ⟨257⟩, ⟨64005⟩⟩]⟩, (-1)⟩)

def exact278400RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64608⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62742⟩⟩], [⟨.program ⟨257⟩, ⟨64005⟩⟩]⟩, (-1)⟩]

theorem exact278400RawTermsValid :
    exact278400RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278400 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64609⟩⟩) exact278400RawTerms .large 278395 .exactZero (none)

def event278401 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62928⟩⟩) 0 ⟨62743⟩ 278358

def event278402 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62928⟩⟩) (.authority (.programFamilyFact))

def exact278403RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62928⟩⟩], []⟩, (1)⟩]

theorem exact278403RawTermsValid :
    exact278403RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278403 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62928⟩⟩) exact278403RawTerms (.finite 22) 278402 .exactZero (none)

def event278404 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62931⟩⟩) 0 ⟨6908⟩ 278380

def event278405 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62931⟩⟩) 1 ⟨62928⟩ 278403

def event278406 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62931⟩⟩) (.product (.predecessor 0 278404 .coefficient) (.predecessor 1 278405 .coefficient) (⟨false, true, none, none, some 1⟩))

def event278407 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62931⟩⟩, .operator (⟨278380, 0⟩, ⟨278403, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact278408RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact278408RawTermsValid :
    exact278408RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278408 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62931⟩⟩) exact278408RawTerms .large 278406 .exactZero (none)

def event278409 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7213⟩⟩) 0 ⟨7177⟩ 278362

def event278410 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7213⟩⟩) (.authority (.operator))

def exact278411RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7213⟩⟩]⟩, (1)⟩]

theorem exact278411RawTermsValid :
    exact278411RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278411 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7213⟩⟩) exact278411RawTerms .large 278410 .exactZero (none)

def event278412 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62932⟩⟩) 0 ⟨7213⟩ 278411

def event278413 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62932⟩⟩) 1 ⟨62931⟩ 278408

def event278414 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62932⟩⟩) (.sum [.predecessor 0 278412 .coefficient, .predecessor 1 278413 .coefficient])

def exact278415RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7213⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact278415RawTermsValid :
    exact278415RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278415 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62932⟩⟩) exact278415RawTerms .large 278414 .exactZero (none)

def event278416 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64614⟩⟩) 0 ⟨62932⟩ 278415

def event278417 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64614⟩⟩) 1 ⟨64609⟩ 278400

def event278418 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64614⟩⟩) (.sum [.predecessor 0 278416 .coefficient, .predecessor 1 278417 .coefficient])

def exact278419RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64608⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7213⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62742⟩⟩], [⟨.program ⟨257⟩, ⟨64005⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact278419RawTermsValid :
    exact278419RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278419 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64614⟩⟩) exact278419RawTerms .large 278418 .exactZero (none)

def event278420 : Event := .preFoldPolynomial 278419 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64608⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7213⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62742⟩⟩], [⟨.program ⟨257⟩, ⟨64005⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact278421RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64608⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7213⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62742⟩⟩], [⟨.program ⟨257⟩, ⟨64005⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event278421 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨64614⟩⟩) 278420 exact278421RawTerms .large 278418 .exactZero (none)

def event278422 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨62743⟩⟩) ⟨⟨92⟩, ⟨73⟩, ⟨135⟩⟩ ⟨278264, 278422⟩

def event278423 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨63509⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63506⟩⟩]⟩) (1) 0 2 (.universal 278422 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63506⟩⟩]⟩) (none) 278421)

def event278424 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63509⟩⟩, .relation 278423 1, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩]⟩, (1)⟩)

def event278425 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63509⟩⟩, .relation 278423 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64608⟩⟩]⟩, (-1)⟩)

def event278426 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63509⟩⟩, .relation 278423 2, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62742⟩⟩], [⟨.program ⟨257⟩, ⟨64005⟩⟩]⟩, (1)⟩)

def event278427 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63509⟩⟩, .relation 278423 3, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact278428RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64608⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62742⟩⟩], [⟨.program ⟨257⟩, ⟨64005⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact278428RawTermsValid :
    exact278428RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278428 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63509⟩⟩) exact278428RawTerms .large 278260 (.finite 202072841853861888) (some (278262))

def event278429 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64611⟩⟩) 0 ⟨63509⟩ 278428

def event278430 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64611⟩⟩) 1 ⟨64610⟩ 278250

def event278431 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64611⟩⟩) (.sum [.predecessor 0 278429 .coefficient, .predecessor 1 278430 .coefficient])

def event278432 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64611⟩⟩, .operator (⟨278428, 0⟩, ⟨278250, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64608⟩⟩]⟩, (1)⟩)

def event278433 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64611⟩⟩, .operator (⟨278428, 2⟩, ⟨278250, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62742⟩⟩], [⟨.program ⟨257⟩, ⟨64005⟩⟩]⟩, (-1)⟩)

def event278434 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64611⟩⟩) (.sum [.result 278428 .summary, .result 278250 .summary])

def exact278435RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact278435RawTermsValid :
    exact278435RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278435 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64611⟩⟩) exact278435RawTerms .large 278431 (.finite 32190771716940580661919523012608) (some (278434))

def event278436 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64612⟩⟩) 0 ⟨64611⟩ 278435

def event278437 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64612⟩⟩) 1 ⟨7100⟩ 15722

def event278438 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64612⟩⟩) (.product (.predecessor 0 278436 .coefficient) (.predecessor 1 278437 .coefficient) (⟨false, false, none, none, none⟩))

def event278439 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64612⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩) [⟨.result 15718 .coefficient, false, none⟩])

def event278440 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64612⟩⟩) (.product (.result 278435 .summary) (.transfer 278439) (⟨false, false, none, none, none⟩))

def event278441 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64612⟩⟩, .operator (⟨278435, 0⟩, ⟨15722, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩)

def event278442 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64612⟩⟩, .operator (⟨278435, 1⟩, ⟨15722, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (-1)⟩)

def event278443 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨64612⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7099⟩⟩) ⟨7015⟩ 15715)

def event278444 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64612⟩⟩, .relation 278443 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact278445RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact278445RawTermsValid :
    exact278445RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278445 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64612⟩⟩) exact278445RawTerms .large 278438 (.finite 345645779393153907795485959807676889169920) (some (278440))

def event278446 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61025⟩⟩) 0 ⟨7177⟩ 15500

def event278447 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61025⟩⟩) 1 ⟨61024⟩ 270842

def event278448 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61025⟩⟩) (.authority (.operator))

def exact278449RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨61025⟩⟩]⟩, (1)⟩]

theorem exact278449RawTermsValid :
    exact278449RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278449 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61025⟩⟩) exact278449RawTerms .large 278448 .exactZero (none)

def event278450 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61628⟩⟩) 0 ⟨61025⟩ 278449

def event278451 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61628⟩⟩) (.authority (.operator))

def exact278452RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨61628⟩⟩]⟩, (1)⟩]

theorem exact278452RawTermsValid :
    exact278452RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278452 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61628⟩⟩) exact278452RawTerms (.finite 8192) 278451 .exactZero (none)

def event278453 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61630⟩⟩) 0 ⟨61370⟩ 271126

def event278454 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61630⟩⟩) 1 ⟨61628⟩ 278452

def event278455 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61630⟩⟩) (.product (.predecessor 0 278453 .coefficient) (.predecessor 1 278454 .coefficient) (⟨false, false, none, none, none⟩))

def event278456 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61630⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨61628⟩⟩]⟩) [⟨.result 278452 .coefficient, false, none⟩])

def event278457 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61630⟩⟩) (.product (.result 271126 .summary) (.transfer 278456) (⟨false, false, none, none, none⟩))

def event278458 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61630⟩⟩, .operator (⟨271126, 0⟩, ⟨278452, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩]⟩, (1)⟩)

def event278459 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61630⟩⟩, .operator (⟨271126, 1⟩, ⟨278452, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩]⟩, (-1)⟩)

def event278460 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨61630⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61628⟩⟩) ⟨61025⟩ 278449)

def event278461 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61630⟩⟩, .relation 278460 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨61025⟩⟩]⟩, (-1)⟩)

def exact278462RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61628⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨61025⟩⟩]⟩, (-1)⟩]

theorem exact278462RawTermsValid :
    exact278462RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278462 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61630⟩⟩) exact278462RawTerms .large 278455 (.finite 32190378816049003834595889643520) (some (278457))

def event278463 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60526⟩⟩) 0 ⟨59763⟩ 13057

def event278464 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60526⟩⟩) (.authority (.relationPreimageSource ⟨71⟩))

def exact278465RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨60526⟩⟩]⟩, (1)⟩]

theorem exact278465RawTermsValid :
    exact278465RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278465 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60526⟩⟩) exact278465RawTerms (.finite 5647228698) 278464 .exactZero (none)

def event278466 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60528⟩⟩) 0 ⟨60526⟩ 278465

def event278467 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60528⟩⟩) 1 ⟨2370⟩ 4

def event278468 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60528⟩⟩) (.scale (.predecessor 0 278466 .coefficient) (.value (.predecessor 1 278467 .coefficient)))

def exact278469RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨60526⟩⟩]⟩, (1)⟩]

theorem exact278469RawTermsValid :
    exact278469RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278469 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60528⟩⟩) exact278469RawTerms (.finite 5647228698) 278468 .exactZero (none)

def event278470 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60529⟩⟩) 0 ⟨5449⟩ 266120

def event278471 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60529⟩⟩) 1 ⟨60528⟩ 278469

def event278472 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60529⟩⟩) (.product (.predecessor 0 278470 .coefficient) (.predecessor 1 278471 .coefficient) (⟨false, false, none, none, none⟩))

def event278473 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60529⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨60526⟩⟩]⟩) [⟨.result 278465 .coefficient, false, none⟩])

def event278474 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60529⟩⟩) (.product (.result 266120 .summary) (.transfer 278473) (⟨false, false, none, none, none⟩))

def event278475 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60529⟩⟩, .operator (⟨266120, 0⟩, ⟨278469, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60526⟩⟩]⟩, (1)⟩)

def event278476 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨60527⟩⟩)

def event278477 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event278478 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event278479 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨387⟩⟩) (.authority (.operator))

def event278480 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨387⟩⟩) (.finite 2)

def event278481 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event278482 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event278483 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event278484 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event278485 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 278484

def event278486 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 278482

def event278487 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 278485 .coefficient) (.value (.predecessor 1 278486 .coefficient)))

def event278488 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event278489 : Event := .predecessor (⟨.program ⟨257⟩, ⟨394⟩⟩) 0 ⟨392⟩ 278488

def event278490 : Event := .predecessor (⟨.program ⟨257⟩, ⟨394⟩⟩) 1 ⟨387⟩ 278480

def event278491 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨394⟩⟩) (.sum [.predecessor 0 278489 .coefficient, .predecessor 1 278490 .coefficient])

def event278492 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨394⟩⟩) (.finite 655342)

def event278493 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5445⟩⟩) 0 ⟨394⟩ 278492

def event278494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5445⟩⟩) 1 ⟨5426⟩ 278478

def event278495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5445⟩⟩) (.identity (.predecessor 1 278494 .coefficient))

def event278496 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5445⟩⟩) (.finite 655360)

def event278497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25150⟩⟩) 0 ⟨5445⟩ 278496

def event278498 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25150⟩⟩) (.authority (.programFamilyFact))

def exact278499RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25150⟩⟩], []⟩, (1)⟩]

theorem exact278499RawTermsValid :
    exact278499RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278499 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25150⟩⟩) exact278499RawTerms (.finite 18) 278498 .exactZero (none)

def event278500 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59260⟩⟩) 0 ⟨5445⟩ 278496

def event278501 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59260⟩⟩) (.authority (.programFamilyFact))

def exact278502RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59260⟩⟩], []⟩, (1)⟩]

theorem exact278502RawTermsValid :
    exact278502RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278502 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59260⟩⟩) exact278502RawTerms (.finite 18) 278501 .exactZero (none)

def event278503 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59261⟩⟩) 0 ⟨59260⟩ 278502

def event278504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59261⟩⟩) 1 ⟨25150⟩ 278499

def event278505 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59261⟩⟩) (.product (.predecessor 0 278503 .coefficient) (.predecessor 1 278504 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event278506 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59261⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25150⟩⟩, ⟨.program ⟨257⟩, ⟨59260⟩⟩], []⟩) [⟨.result 278502 .coefficient, true, some 1⟩, ⟨.result 278499 .coefficient, true, some 1⟩])

def event278507 : Event := .survivorFold (1) 278506

def exact278508RawTerms : List Term := []

theorem exact278508RawTermsValid :
    exact278508RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278508 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59261⟩⟩) exact278508RawTerms (.finite 324) 278505 (.finite 324) (some (278506))

def event278509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59262⟩⟩) 0 ⟨59261⟩ 278508

def event278510 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59262⟩⟩) (.identity (.predecessor 0 278509 .coefficient))

def event278511 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨59262⟩⟩) (.finite 324)

def event278512 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59762⟩⟩) 0 ⟨59262⟩ 278511

def event278513 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59762⟩⟩) (.authority (.programFamilyFact))

def exact278514RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59762⟩⟩], []⟩, (1)⟩]

theorem exact278514RawTermsValid :
    exact278514RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278514 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59762⟩⟩) exact278514RawTerms (.finite 18) 278513 .exactZero (none)

def event278515 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59763⟩⟩) 0 ⟨59762⟩ 278514

def event278516 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59763⟩⟩) (.identity (.predecessor 0 278515 .coefficient))

def event278517 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨59763⟩⟩) (.finite 18)

def event278518 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60526⟩⟩) 0 ⟨59763⟩ 278517

def event278519 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60526⟩⟩) (.authority (.relationPreimageSource ⟨71⟩))

def exact278520RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨60526⟩⟩]⟩, (1)⟩]

theorem exact278520RawTermsValid :
    exact278520RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278520 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60526⟩⟩) exact278520RawTerms (.finite 5647228698) 278519 .exactZero (none)

def event278521 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact278522RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact278522RawTermsValid :
    exact278522RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278522 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact278522RawTerms .large 278521 .exactZero (none)

def event278523 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60527⟩⟩) 0 ⟨35⟩ 278522

def event278524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60527⟩⟩) 1 ⟨60526⟩ 278520

def event278525 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60527⟩⟩) (.product (.predecessor 0 278523 .coefficient) (.predecessor 1 278524 .coefficient) (⟨false, false, none, none, none⟩))

def event278526 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60527⟩⟩, .operator (⟨278522, 0⟩, ⟨278520, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60526⟩⟩]⟩, (1)⟩)

def exact278527RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60526⟩⟩]⟩, (1)⟩]

theorem exact278527RawTermsValid :
    exact278527RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event278527 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60527⟩⟩) exact278527RawTerms .large 278525 .exactZero (none)

def eventLeaf17392 : Array AnnotatedEvent := #[
  { event := event278272
    frameStart := 278264 },
  { event := event278273
    frameStart := 278264 },
  { event := event278274
    frameStart := 278264 },
  { event := event278275
    frameStart := 278264 },
  { event := event278276
    frameStart := 278264 },
  { event := event278277
    frameStart := 278264 },
  { event := event278278
    frameStart := 278264 },
  { event := event278279
    frameStart := 278264 },
  { event := event278280
    frameStart := 278264 },
  { event := event278281
    frameStart := 278264 },
  { event := event278282
    frameStart := 278264 },
  { event := event278283
    frameStart := 278264 },
  { event := event278284
    frameStart := 278264 },
  { event := event278285
    frameStart := 278264 },
  { event := event278286
    frameStart := 278264 },
  { event := event278287
    frameStart := 278264 }
]

def eventLeaf17393 : Array AnnotatedEvent := #[
  { event := event278288
    frameStart := 278264 },
  { event := event278289
    frameStart := 278264 },
  { event := event278290
    frameStart := 278264 },
  { event := event278291
    frameStart := 278264 },
  { event := event278292
    frameStart := 278264 },
  { event := event278293
    frameStart := 278264 },
  { event := event278294
    frameStart := 278264 },
  { event := event278295
    frameStart := 278264 },
  { event := event278296
    frameStart := 278264 },
  { event := event278297
    frameStart := 278264 },
  { event := event278298
    frameStart := 278264 },
  { event := event278299
    frameStart := 278264 },
  { event := event278300
    frameStart := 278264 },
  { event := event278301
    frameStart := 278264 },
  { event := event278302
    frameStart := 278264 },
  { event := event278303
    frameStart := 278264 }
]

def eventLeaf17394 : Array AnnotatedEvent := #[
  { event := event278304
    frameStart := 278264 },
  { event := event278305
    frameStart := 278264 },
  { event := event278306
    frameStart := 278264 },
  { event := event278307
    frameStart := 278264 },
  { event := event278308
    frameStart := 278264 },
  { event := event278309
    frameStart := 278264 },
  { event := event278310
    frameStart := 278264 },
  { event := event278311
    frameStart := 278264 },
  { event := event278312
    frameStart := 278264 },
  { event := event278313
    frameStart := 278264 },
  { event := event278314
    frameStart := 278264 },
  { event := event278315
    frameStart := 278264 },
  { event := event278316
    frameStart := 278264 },
  { event := event278317
    frameStart := 278264 },
  { event := event278318
    frameStart := 278318 },
  { event := event278319
    frameStart := 278318 }
]

def eventLeaf17395 : Array AnnotatedEvent := #[
  { event := event278320
    frameStart := 278318 },
  { event := event278321
    frameStart := 278318 },
  { event := event278322
    frameStart := 278318 },
  { event := event278323
    frameStart := 278318 },
  { event := event278324
    frameStart := 278318 },
  { event := event278325
    frameStart := 278318 },
  { event := event278326
    frameStart := 278318 },
  { event := event278327
    frameStart := 278318 },
  { event := event278328
    frameStart := 278318 },
  { event := event278329
    frameStart := 278318 },
  { event := event278330
    frameStart := 278318 },
  { event := event278331
    frameStart := 278318 },
  { event := event278332
    frameStart := 278318 },
  { event := event278333
    frameStart := 278318 },
  { event := event278334
    frameStart := 278318 },
  { event := event278335
    frameStart := 278318 }
]

def eventLeaf17396 : Array AnnotatedEvent := #[
  { event := event278336
    frameStart := 278318 },
  { event := event278337
    frameStart := 278318 },
  { event := event278338
    frameStart := 278318 },
  { event := event278339
    frameStart := 278318 },
  { event := event278340
    frameStart := 278318 },
  { event := event278341
    frameStart := 278318 },
  { event := event278342
    frameStart := 278318 },
  { event := event278343
    frameStart := 278318 },
  { event := event278344
    frameStart := 278318 },
  { event := event278345
    frameStart := 278318 },
  { event := event278346
    frameStart := 278318 },
  { event := event278347
    frameStart := 278318 },
  { event := event278348
    frameStart := 278318 },
  { event := event278349
    frameStart := 278318 },
  { event := event278350
    frameStart := 278318 },
  { event := event278351
    frameStart := 278318 }
]

def eventLeaf17397 : Array AnnotatedEvent := #[
  { event := event278352
    frameStart := 278318 },
  { event := event278353
    frameStart := 278318 },
  { event := event278354
    frameStart := 278318 },
  { event := event278355
    frameStart := 278318 },
  { event := event278356
    frameStart := 278318 },
  { event := event278357
    frameStart := 278318 },
  { event := event278358
    frameStart := 278318 },
  { event := event278359
    frameStart := 278318 },
  { event := event278360
    frameStart := 278318 },
  { event := event278361
    frameStart := 278318 },
  { event := event278362
    frameStart := 278318 },
  { event := event278363
    frameStart := 278318 },
  { event := event278364
    frameStart := 278318 },
  { event := event278365
    frameStart := 278318 },
  { event := event278366
    frameStart := 278318 },
  { event := event278367
    frameStart := 278318 }
]

def eventLeaf17398 : Array AnnotatedEvent := #[
  { event := event278368
    frameStart := 278318 },
  { event := event278369
    frameStart := 278318 },
  { event := event278370
    frameStart := 278318 },
  { event := event278371
    frameStart := 278318 },
  { event := event278372
    frameStart := 278318 },
  { event := event278373
    frameStart := 278318 },
  { event := event278374
    frameStart := 278318 },
  { event := event278375
    frameStart := 278318 },
  { event := event278376
    frameStart := 278318 },
  { event := event278377
    frameStart := 278318 },
  { event := event278378
    frameStart := 278318 },
  { event := event278379
    frameStart := 278318 },
  { event := event278380
    frameStart := 278318 },
  { event := event278381
    frameStart := 278318 },
  { event := event278382
    frameStart := 278318 },
  { event := event278383
    frameStart := 278318 }
]

def eventLeaf17399 : Array AnnotatedEvent := #[
  { event := event278384
    frameStart := 278318 },
  { event := event278385
    frameStart := 278318 },
  { event := event278386
    frameStart := 278318 },
  { event := event278387
    frameStart := 278318 },
  { event := event278388
    frameStart := 278318 },
  { event := event278389
    frameStart := 278318 },
  { event := event278390
    frameStart := 278318 },
  { event := event278391
    frameStart := 278318 },
  { event := event278392
    frameStart := 278318 },
  { event := event278393
    frameStart := 278318 },
  { event := event278394
    frameStart := 278318 },
  { event := event278395
    frameStart := 278318 },
  { event := event278396
    frameStart := 278318 },
  { event := event278397
    frameStart := 278318 },
  { event := event278398
    frameStart := 278318 },
  { event := event278399
    frameStart := 278318 }
]

def eventLeaf17400 : Array AnnotatedEvent := #[
  { event := event278400
    frameStart := 278318 },
  { event := event278401
    frameStart := 278318 },
  { event := event278402
    frameStart := 278318 },
  { event := event278403
    frameStart := 278318 },
  { event := event278404
    frameStart := 278318 },
  { event := event278405
    frameStart := 278318 },
  { event := event278406
    frameStart := 278318 },
  { event := event278407
    frameStart := 278318 },
  { event := event278408
    frameStart := 278318 },
  { event := event278409
    frameStart := 278318 },
  { event := event278410
    frameStart := 278318 },
  { event := event278411
    frameStart := 278318 },
  { event := event278412
    frameStart := 278318 },
  { event := event278413
    frameStart := 278318 },
  { event := event278414
    frameStart := 278318 },
  { event := event278415
    frameStart := 278318 }
]

def eventLeaf17401 : Array AnnotatedEvent := #[
  { event := event278416
    frameStart := 278318 },
  { event := event278417
    frameStart := 278318 },
  { event := event278418
    frameStart := 278318 },
  { event := event278419
    frameStart := 278318 },
  { event := event278420
    frameStart := 278318 },
  { event := event278421
    frameStart := 278318 },
  { event := event278422
    frameStart := 0 },
  { event := event278423
    frameStart := 0 },
  { event := event278424
    frameStart := 0 },
  { event := event278425
    frameStart := 0 },
  { event := event278426
    frameStart := 0 },
  { event := event278427
    frameStart := 0 },
  { event := event278428
    frameStart := 0 },
  { event := event278429
    frameStart := 0 },
  { event := event278430
    frameStart := 0 },
  { event := event278431
    frameStart := 0 }
]

def eventLeaf17402 : Array AnnotatedEvent := #[
  { event := event278432
    frameStart := 0 },
  { event := event278433
    frameStart := 0 },
  { event := event278434
    frameStart := 0 },
  { event := event278435
    frameStart := 0 },
  { event := event278436
    frameStart := 0 },
  { event := event278437
    frameStart := 0 },
  { event := event278438
    frameStart := 0 },
  { event := event278439
    frameStart := 0 },
  { event := event278440
    frameStart := 0 },
  { event := event278441
    frameStart := 0 },
  { event := event278442
    frameStart := 0 },
  { event := event278443
    frameStart := 0 },
  { event := event278444
    frameStart := 0 },
  { event := event278445
    frameStart := 0 },
  { event := event278446
    frameStart := 0 },
  { event := event278447
    frameStart := 0 }
]

def eventLeaf17403 : Array AnnotatedEvent := #[
  { event := event278448
    frameStart := 0 },
  { event := event278449
    frameStart := 0 },
  { event := event278450
    frameStart := 0 },
  { event := event278451
    frameStart := 0 },
  { event := event278452
    frameStart := 0 },
  { event := event278453
    frameStart := 0 },
  { event := event278454
    frameStart := 0 },
  { event := event278455
    frameStart := 0 },
  { event := event278456
    frameStart := 0 },
  { event := event278457
    frameStart := 0 },
  { event := event278458
    frameStart := 0 },
  { event := event278459
    frameStart := 0 },
  { event := event278460
    frameStart := 0 },
  { event := event278461
    frameStart := 0 },
  { event := event278462
    frameStart := 0 },
  { event := event278463
    frameStart := 0 }
]

def eventLeaf17404 : Array AnnotatedEvent := #[
  { event := event278464
    frameStart := 0 },
  { event := event278465
    frameStart := 0 },
  { event := event278466
    frameStart := 0 },
  { event := event278467
    frameStart := 0 },
  { event := event278468
    frameStart := 0 },
  { event := event278469
    frameStart := 0 },
  { event := event278470
    frameStart := 0 },
  { event := event278471
    frameStart := 0 },
  { event := event278472
    frameStart := 0 },
  { event := event278473
    frameStart := 0 },
  { event := event278474
    frameStart := 0 },
  { event := event278475
    frameStart := 0 },
  { event := event278476
    frameStart := 278476 },
  { event := event278477
    frameStart := 278476 },
  { event := event278478
    frameStart := 278476 },
  { event := event278479
    frameStart := 278476 }
]

def eventLeaf17405 : Array AnnotatedEvent := #[
  { event := event278480
    frameStart := 278476 },
  { event := event278481
    frameStart := 278476 },
  { event := event278482
    frameStart := 278476 },
  { event := event278483
    frameStart := 278476 },
  { event := event278484
    frameStart := 278476 },
  { event := event278485
    frameStart := 278476 },
  { event := event278486
    frameStart := 278476 },
  { event := event278487
    frameStart := 278476 },
  { event := event278488
    frameStart := 278476 },
  { event := event278489
    frameStart := 278476 },
  { event := event278490
    frameStart := 278476 },
  { event := event278491
    frameStart := 278476 },
  { event := event278492
    frameStart := 278476 },
  { event := event278493
    frameStart := 278476 },
  { event := event278494
    frameStart := 278476 },
  { event := event278495
    frameStart := 278476 }
]

def eventLeaf17406 : Array AnnotatedEvent := #[
  { event := event278496
    frameStart := 278476 },
  { event := event278497
    frameStart := 278476 },
  { event := event278498
    frameStart := 278476 },
  { event := event278499
    frameStart := 278476 },
  { event := event278500
    frameStart := 278476 },
  { event := event278501
    frameStart := 278476 },
  { event := event278502
    frameStart := 278476 },
  { event := event278503
    frameStart := 278476 },
  { event := event278504
    frameStart := 278476 },
  { event := event278505
    frameStart := 278476 },
  { event := event278506
    frameStart := 278476 },
  { event := event278507
    frameStart := 278476 },
  { event := event278508
    frameStart := 278476 },
  { event := event278509
    frameStart := 278476 },
  { event := event278510
    frameStart := 278476 },
  { event := event278511
    frameStart := 278476 }
]

def eventLeaf17407 : Array AnnotatedEvent := #[
  { event := event278512
    frameStart := 278476 },
  { event := event278513
    frameStart := 278476 },
  { event := event278514
    frameStart := 278476 },
  { event := event278515
    frameStart := 278476 },
  { event := event278516
    frameStart := 278476 },
  { event := event278517
    frameStart := 278476 },
  { event := event278518
    frameStart := 278476 },
  { event := event278519
    frameStart := 278476 },
  { event := event278520
    frameStart := 278476 },
  { event := event278521
    frameStart := 278476 },
  { event := event278522
    frameStart := 278476 },
  { event := event278523
    frameStart := 278476 },
  { event := event278524
    frameStart := 278476 },
  { event := event278525
    frameStart := 278476 },
  { event := event278526
    frameStart := 278476 },
  { event := event278527
    frameStart := 278476 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1087
