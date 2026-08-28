import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events337

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event86272 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48197⟩⟩) (.finite 60)

def event86273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49012⟩⟩) 0 ⟨48197⟩ 86272

def event86274 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49012⟩⟩) (.authority (.relationPreimageSource ⟨93⟩))

def exact86275RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49012⟩⟩]⟩, (1)⟩]

theorem exact86275RawTermsValid :
    exact86275RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86275 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49012⟩⟩) exact86275RawTerms (.finite 5647228698) 86274 .exactZero (none)

def event86276 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact86277RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact86277RawTermsValid :
    exact86277RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86277 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact86277RawTerms .large 86276 .exactZero (none)

def event86278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49013⟩⟩) 0 ⟨35⟩ 86277

def event86279 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49013⟩⟩) 1 ⟨49012⟩ 86275

def event86280 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49013⟩⟩) (.product (.predecessor 0 86278 .coefficient) (.predecessor 1 86279 .coefficient) (⟨false, false, none, none, none⟩))

def event86281 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49013⟩⟩, .operator (⟨86277, 0⟩, ⟨86275, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49012⟩⟩]⟩, (1)⟩)

def exact86282RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49012⟩⟩]⟩, (1)⟩]

theorem exact86282RawTermsValid :
    exact86282RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86282 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49013⟩⟩) exact86282RawTerms .large 86280 .exactZero (none)

def event86283 : Event := .preFoldPolynomial 86282 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49012⟩⟩]⟩, (1)⟩] .exactZero none

def exact86284RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49012⟩⟩]⟩, (1)⟩]

def event86284 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨49013⟩⟩) 86283 exact86284RawTerms .large 86280 .exactZero (none)

def event86285 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨50178⟩⟩)

def event86286 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event86287 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event86288 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event86289 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event86290 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event86291 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event86292 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event86293 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event86294 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 86293

def event86295 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 86291

def event86296 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 86294 .coefficient) (.value (.predecessor 1 86295 .coefficient)))

def event86297 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event86298 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 86297

def event86299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 86289

def event86300 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 86298 .coefficient, .predecessor 1 86299 .coefficient])

def event86301 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event86302 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 86301

def event86303 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 86287

def event86304 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 86303 .coefficient))

def event86305 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event86306 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47978⟩⟩) 0 ⟨10325⟩ 86305

def event86307 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47978⟩⟩) (.authority (.programFamilyFact))

def exact86308RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47978⟩⟩], []⟩, (1)⟩]

theorem exact86308RawTermsValid :
    exact86308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86308 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47978⟩⟩) exact86308RawTerms (.finite 60) 86307 .exactZero (none)

def event86309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15171⟩⟩) 0 ⟨10325⟩ 86305

def event86310 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15171⟩⟩) (.authority (.programFamilyFact))

def exact86311RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15171⟩⟩], []⟩, (1)⟩]

theorem exact86311RawTermsValid :
    exact86311RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86311 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15171⟩⟩) exact86311RawTerms (.finite 60) 86310 .exactZero (none)

def event86312 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47979⟩⟩) 0 ⟨15171⟩ 86311

def event86313 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47979⟩⟩) 1 ⟨47978⟩ 86308

def event86314 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47979⟩⟩) (.product (.predecessor 0 86312 .coefficient) (.predecessor 1 86313 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event86315 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47979⟩⟩, .operator (⟨86311, 0⟩, ⟨86308, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], []⟩, (1)⟩)

def exact86316RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], []⟩, (1)⟩]

theorem exact86316RawTermsValid :
    exact86316RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86316 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47979⟩⟩) exact86316RawTerms (.finite 3600) 86314 .exactZero (none)

def event86317 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47980⟩⟩) 0 ⟨47979⟩ 86316

def event86318 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47980⟩⟩) (.identity (.predecessor 0 86317 .coefficient))

def event86319 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47980⟩⟩) (.finite 3600)

def event86320 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48196⟩⟩) 0 ⟨47980⟩ 86319

def event86321 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48196⟩⟩) (.authority (.programFamilyFact))

def exact86322RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], []⟩, (1)⟩]

theorem exact86322RawTermsValid :
    exact86322RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86322 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48196⟩⟩) exact86322RawTerms (.finite 60) 86321 .exactZero (none)

def event86323 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48197⟩⟩) 0 ⟨48196⟩ 86322

def event86324 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48197⟩⟩) (.identity (.predecessor 0 86323 .coefficient))

def event86325 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48197⟩⟩) (.finite 60)

def event86326 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49353⟩⟩) 0 ⟨48197⟩ 86325

def event86327 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49353⟩⟩) (.authority (.programFamilyFact))

def event86328 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49353⟩⟩) (.finite 3720)

def event86329 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event86330 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49354⟩⟩) 0 ⟨7177⟩ 86329

def event86331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49354⟩⟩) 1 ⟨49353⟩ 86328

def event86332 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49354⟩⟩) (.authority (.operator))

def exact86333RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49354⟩⟩]⟩, (1)⟩]

theorem exact86333RawTermsValid :
    exact86333RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86333 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49354⟩⟩) exact86333RawTerms .large 86332 .exactZero (none)

def event86334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50173⟩⟩) 0 ⟨49354⟩ 86333

def event86335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50173⟩⟩) (.authority (.operator))

def exact86336RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨50173⟩⟩]⟩, (1)⟩]

theorem exact86336RawTermsValid :
    exact86336RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86336 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50173⟩⟩) exact86336RawTerms (.finite 8192) 86335 .exactZero (none)

def event86337 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event86338 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event86339 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49530⟩⟩) 0 ⟨48197⟩ 86325

def event86340 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49530⟩⟩) 1 ⟨136⟩ 86338

def event86341 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49530⟩⟩) (.sum [.predecessor 0 86339 .coefficient, .predecessor 1 86340 .coefficient])

def event86342 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49530⟩⟩) (.finite 60)

def event86343 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49531⟩⟩) 0 ⟨49530⟩ 86342

def event86344 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49531⟩⟩) (.identity (.predecessor 0 86343 .coefficient))

def exact86345RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], []⟩, (1)⟩]

theorem exact86345RawTermsValid :
    exact86345RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86345 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49531⟩⟩) exact86345RawTerms (.finite 60) 86344 .exactZero (none)

def event86346 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact86347RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact86347RawTermsValid :
    exact86347RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86347 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact86347RawTerms .large 86346 .exactZero (none)

def event86348 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49532⟩⟩) 0 ⟨6908⟩ 86347

def event86349 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49532⟩⟩) 1 ⟨49531⟩ 86345

def event86350 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49532⟩⟩) (.product (.predecessor 0 86348 .coefficient) (.predecessor 1 86349 .coefficient) (⟨false, false, none, none, none⟩))

def event86351 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49532⟩⟩, .operator (⟨86347, 0⟩, ⟨86345, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact86352RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact86352RawTermsValid :
    exact86352RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86352 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49532⟩⟩) exact86352RawTerms .large 86350 .exactZero (none)

def event86353 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7196⟩⟩) 0 ⟨7177⟩ 86329

def event86354 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7196⟩⟩) (.authority (.operator))

def exact86355RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩]

theorem exact86355RawTermsValid :
    exact86355RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86355 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7196⟩⟩) exact86355RawTerms .large 86354 .exactZero (none)

def event86356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49533⟩⟩) 0 ⟨7196⟩ 86355

def event86357 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49533⟩⟩) 1 ⟨49532⟩ 86352

def event86358 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49533⟩⟩) (.sum [.predecessor 0 86356 .coefficient, .predecessor 1 86357 .coefficient])

def exact86359RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact86359RawTermsValid :
    exact86359RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86359 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49533⟩⟩) exact86359RawTerms .large 86358 .exactZero (none)

def event86360 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50174⟩⟩) 0 ⟨49533⟩ 86359

def event86361 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50174⟩⟩) 1 ⟨50173⟩ 86336

def event86362 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50174⟩⟩) (.product (.predecessor 0 86360 .coefficient) (.predecessor 1 86361 .coefficient) (⟨false, false, none, none, none⟩))

def event86363 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50174⟩⟩, .operator (⟨86359, 0⟩, ⟨86336, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50173⟩⟩]⟩, (1)⟩)

def event86364 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50174⟩⟩, .operator (⟨86359, 1⟩, ⟨86336, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50173⟩⟩]⟩, (-1)⟩)

def event86365 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨50174⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50173⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50173⟩⟩) ⟨49354⟩ 86333)

def event86366 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50174⟩⟩, .relation 86365 0, ⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨49354⟩⟩]⟩, (-1)⟩)

def exact86367RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨49354⟩⟩]⟩, (-1)⟩]

theorem exact86367RawTermsValid :
    exact86367RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86367 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50174⟩⟩) exact86367RawTerms .large 86362 .exactZero (none)

def event86368 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48437⟩⟩) 0 ⟨48197⟩ 86325

def event86369 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48437⟩⟩) (.authority (.programFamilyFact))

def exact86370RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48437⟩⟩], []⟩, (1)⟩]

theorem exact86370RawTermsValid :
    exact86370RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86370 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48437⟩⟩) exact86370RawTerms (.finite 60) 86369 .exactZero (none)

def event86371 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48439⟩⟩) 0 ⟨6908⟩ 86347

def event86372 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48439⟩⟩) 1 ⟨48437⟩ 86370

def event86373 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48439⟩⟩) (.product (.predecessor 0 86371 .coefficient) (.predecessor 1 86372 .coefficient) (⟨false, true, none, none, some 1⟩))

def event86374 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48439⟩⟩, .operator (⟨86347, 0⟩, ⟨86370, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact86375RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact86375RawTermsValid :
    exact86375RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86375 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48439⟩⟩) exact86375RawTerms .large 86373 .exactZero (none)

def event86376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7231⟩⟩) 0 ⟨7177⟩ 86329

def event86377 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7231⟩⟩) (.authority (.operator))

def exact86378RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩]

theorem exact86378RawTermsValid :
    exact86378RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86378 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7231⟩⟩) exact86378RawTerms .large 86377 .exactZero (none)

def event86379 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48440⟩⟩) 0 ⟨7231⟩ 86378

def event86380 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48440⟩⟩) 1 ⟨48439⟩ 86375

def event86381 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48440⟩⟩) (.sum [.predecessor 0 86379 .coefficient, .predecessor 1 86380 .coefficient])

def exact86382RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact86382RawTermsValid :
    exact86382RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86382 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48440⟩⟩) exact86382RawTerms .large 86381 .exactZero (none)

def event86383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50178⟩⟩) 0 ⟨48440⟩ 86382

def event86384 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50178⟩⟩) 1 ⟨50174⟩ 86367

def event86385 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50178⟩⟩) (.sum [.predecessor 0 86383 .coefficient, .predecessor 1 86384 .coefficient])

def exact86386RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50173⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨49354⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact86386RawTermsValid :
    exact86386RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86386 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50178⟩⟩) exact86386RawTerms .large 86385 .exactZero (none)

def event86387 : Event := .preFoldPolynomial 86386 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50173⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨49354⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact86388RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50173⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨49354⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event86388 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨50178⟩⟩) 86387 exact86388RawTerms .large 86385 .exactZero (none)

def event86389 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨48197⟩⟩) ⟨⟨110⟩, ⟨93⟩, ⟨135⟩⟩ ⟨86231, 86389⟩

def event86390 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49015⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49012⟩⟩]⟩) (1) 0 2 (.universal 86389 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49012⟩⟩]⟩) (none) 86388)

def event86391 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49015⟩⟩, .relation 86390 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩)

def event86392 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49015⟩⟩, .relation 86390 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50173⟩⟩]⟩, (-1)⟩)

def event86393 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49015⟩⟩, .relation 86390 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨49354⟩⟩]⟩, (1)⟩)

def event86394 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49015⟩⟩, .relation 86390 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact86395RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50173⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨49354⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact86395RawTermsValid :
    exact86395RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86395 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49015⟩⟩) exact86395RawTerms .large 86227 (.finite 202072841853861888) (some (86229))

def event86396 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50176⟩⟩) 0 ⟨49015⟩ 86395

def event86397 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50176⟩⟩) 1 ⟨50175⟩ 86217

def event86398 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50176⟩⟩) (.sum [.predecessor 0 86396 .coefficient, .predecessor 1 86397 .coefficient])

def event86399 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50176⟩⟩, .operator (⟨86395, 0⟩, ⟨86217, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50173⟩⟩]⟩, (1)⟩)

def event86400 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50176⟩⟩, .operator (⟨86395, 2⟩, ⟨86217, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48196⟩⟩], [⟨.program ⟨257⟩, ⟨49354⟩⟩]⟩, (-1)⟩)

def event86401 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50176⟩⟩) (.sum [.result 86395 .summary, .result 86217 .summary])

def exact86402RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact86402RawTermsValid :
    exact86402RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86402 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50176⟩⟩) exact86402RawTerms .large 86398 (.finite 32194504275408640829496428331008) (some (86401))

def event86403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50177⟩⟩) 0 ⟨50176⟩ 86402

def event86404 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50177⟩⟩) 1 ⟨7148⟩ 15542

def event86405 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50177⟩⟩) (.product (.predecessor 0 86403 .coefficient) (.predecessor 1 86404 .coefficient) (⟨false, false, none, none, none⟩))

def event86406 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50177⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩) [⟨.result 15538 .coefficient, false, none⟩])

def event86407 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50177⟩⟩) (.product (.result 86402 .summary) (.transfer 86406) (⟨false, false, none, none, none⟩))

def event86408 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50177⟩⟩, .operator (⟨86402, 0⟩, ⟨15542, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩)

def event86409 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50177⟩⟩, .operator (⟨86402, 1⟩, ⟨15542, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (-1)⟩)

def event86410 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨50177⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7147⟩⟩) ⟨7039⟩ 15535)

def event86411 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50177⟩⟩, .relation 86410 0, ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact86412RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩]

theorem exact86412RawTermsValid :
    exact86412RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86412 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50177⟩⟩) exact86412RawTerms .large 86405 (.finite 345685857434530723496243679576218056785920) (some (86407))

def event86413 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46674⟩⟩) 0 ⟨7177⟩ 15500

def event86414 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46674⟩⟩) 1 ⟨46673⟩ 76379

def event86415 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46674⟩⟩) (.authority (.operator))

def exact86416RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46674⟩⟩]⟩, (1)⟩]

theorem exact86416RawTermsValid :
    exact86416RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86416 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46674⟩⟩) exact86416RawTerms .large 86415 .exactZero (none)

def event86417 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47493⟩⟩) 0 ⟨46674⟩ 86416

def event86418 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47493⟩⟩) (.authority (.operator))

def exact86419RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47493⟩⟩]⟩, (1)⟩]

theorem exact86419RawTermsValid :
    exact86419RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86419 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47493⟩⟩) exact86419RawTerms (.finite 8192) 86418 .exactZero (none)

def event86420 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47495⟩⟩) 0 ⟨47047⟩ 76663

def event86421 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47495⟩⟩) 1 ⟨47493⟩ 86419

def event86422 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47495⟩⟩) (.product (.predecessor 0 86420 .coefficient) (.predecessor 1 86421 .coefficient) (⟨false, false, none, none, none⟩))

def event86423 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47495⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨47493⟩⟩]⟩) [⟨.result 86419 .coefficient, false, none⟩])

def event86424 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47495⟩⟩) (.product (.result 76663 .summary) (.transfer 86423) (⟨false, false, none, none, none⟩))

def event86425 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47495⟩⟩, .operator (⟨76663, 0⟩, ⟨86419, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47493⟩⟩]⟩, (1)⟩)

def event86426 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47495⟩⟩, .operator (⟨76663, 1⟩, ⟨86419, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47493⟩⟩]⟩, (-1)⟩)

def event86427 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47495⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47493⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47493⟩⟩) ⟨46674⟩ 86416)

def event86428 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47495⟩⟩, .relation 86427 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨46674⟩⟩]⟩, (-1)⟩)

def exact86429RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47493⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨46674⟩⟩]⟩, (-1)⟩]

theorem exact86429RawTermsValid :
    exact86429RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86429 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47495⟩⟩) exact86429RawTerms .large 86422 (.finite 32194307824962751379413684715520) (some (86424))

def event86430 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46332⟩⟩) 0 ⟨45517⟩ 3126

def event86431 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46332⟩⟩) (.authority (.relationPreimageSource ⟨91⟩))

def exact86432RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46332⟩⟩]⟩, (1)⟩]

theorem exact86432RawTermsValid :
    exact86432RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86432 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46332⟩⟩) exact86432RawTerms (.finite 5647228698) 86431 .exactZero (none)

def event86433 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46334⟩⟩) 0 ⟨46332⟩ 86432

def event86434 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46334⟩⟩) 1 ⟨2370⟩ 4

def event86435 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46334⟩⟩) (.scale (.predecessor 0 86433 .coefficient) (.value (.predecessor 1 86434 .coefficient)))

def exact86436RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46332⟩⟩]⟩, (1)⟩]

theorem exact86436RawTermsValid :
    exact86436RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86436 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46334⟩⟩) exact86436RawTerms (.finite 5647228698) 86435 .exactZero (none)

def event86437 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46335⟩⟩) 0 ⟨10368⟩ 75995

def event86438 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46335⟩⟩) 1 ⟨46334⟩ 86436

def event86439 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46335⟩⟩) (.product (.predecessor 0 86437 .coefficient) (.predecessor 1 86438 .coefficient) (⟨false, false, none, none, none⟩))

def event86440 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46335⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨46332⟩⟩]⟩) [⟨.result 86432 .coefficient, false, none⟩])

def event86441 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46335⟩⟩) (.product (.result 75995 .summary) (.transfer 86440) (⟨false, false, none, none, none⟩))

def event86442 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46335⟩⟩, .operator (⟨75995, 0⟩, ⟨86436, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46332⟩⟩]⟩, (1)⟩)

def event86443 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨46333⟩⟩)

def event86444 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event86445 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event86446 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event86447 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event86448 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event86449 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event86450 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event86451 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event86452 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 86451

def event86453 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 86449

def event86454 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 86452 .coefficient) (.value (.predecessor 1 86453 .coefficient)))

def event86455 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event86456 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 86455

def event86457 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 86447

def event86458 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 86456 .coefficient, .predecessor 1 86457 .coefficient])

def event86459 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event86460 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 86459

def event86461 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 86445

def event86462 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 86461 .coefficient))

def event86463 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event86464 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45298⟩⟩) 0 ⟨10325⟩ 86463

def event86465 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45298⟩⟩) (.authority (.programFamilyFact))

def exact86466RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45298⟩⟩], []⟩, (1)⟩]

theorem exact86466RawTermsValid :
    exact86466RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86466 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45298⟩⟩) exact86466RawTerms (.finite 58) 86465 .exactZero (none)

def event86467 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14871⟩⟩) 0 ⟨10325⟩ 86463

def event86468 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14871⟩⟩) (.authority (.programFamilyFact))

def exact86469RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩], []⟩, (1)⟩]

theorem exact86469RawTermsValid :
    exact86469RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86469 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14871⟩⟩) exact86469RawTerms (.finite 58) 86468 .exactZero (none)

def event86470 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45299⟩⟩) 0 ⟨14871⟩ 86469

def event86471 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45299⟩⟩) 1 ⟨45298⟩ 86466

def event86472 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45299⟩⟩) (.product (.predecessor 0 86470 .coefficient) (.predecessor 1 86471 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event86473 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45299⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], []⟩) [⟨.result 86469 .coefficient, true, some 1⟩, ⟨.result 86466 .coefficient, true, some 1⟩])

def event86474 : Event := .survivorFold (1) 86473

def exact86475RawTerms : List Term := []

theorem exact86475RawTermsValid :
    exact86475RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86475 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45299⟩⟩) exact86475RawTerms (.finite 3364) 86472 (.finite 3364) (some (86473))

def event86476 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45300⟩⟩) 0 ⟨45299⟩ 86475

def event86477 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45300⟩⟩) (.identity (.predecessor 0 86476 .coefficient))

def event86478 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45300⟩⟩) (.finite 3364)

def event86479 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45516⟩⟩) 0 ⟨45300⟩ 86478

def event86480 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45516⟩⟩) (.authority (.programFamilyFact))

def exact86481RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], []⟩, (1)⟩]

theorem exact86481RawTermsValid :
    exact86481RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86481 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45516⟩⟩) exact86481RawTerms (.finite 58) 86480 .exactZero (none)

def event86482 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45517⟩⟩) 0 ⟨45516⟩ 86481

def event86483 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45517⟩⟩) (.identity (.predecessor 0 86482 .coefficient))

def event86484 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45517⟩⟩) (.finite 58)

def event86485 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46332⟩⟩) 0 ⟨45517⟩ 86484

def event86486 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46332⟩⟩) (.authority (.relationPreimageSource ⟨91⟩))

def exact86487RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46332⟩⟩]⟩, (1)⟩]

theorem exact86487RawTermsValid :
    exact86487RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86487 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46332⟩⟩) exact86487RawTerms (.finite 5647228698) 86486 .exactZero (none)

def event86488 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact86489RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact86489RawTermsValid :
    exact86489RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86489 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact86489RawTerms .large 86488 .exactZero (none)

def event86490 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46333⟩⟩) 0 ⟨35⟩ 86489

def event86491 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46333⟩⟩) 1 ⟨46332⟩ 86487

def event86492 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46333⟩⟩) (.product (.predecessor 0 86490 .coefficient) (.predecessor 1 86491 .coefficient) (⟨false, false, none, none, none⟩))

def event86493 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46333⟩⟩, .operator (⟨86489, 0⟩, ⟨86487, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46332⟩⟩]⟩, (1)⟩)

def exact86494RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46332⟩⟩]⟩, (1)⟩]

theorem exact86494RawTermsValid :
    exact86494RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86494 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46333⟩⟩) exact86494RawTerms .large 86492 .exactZero (none)

def event86495 : Event := .preFoldPolynomial 86494 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46332⟩⟩]⟩, (1)⟩] .exactZero none

def exact86496RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46332⟩⟩]⟩, (1)⟩]

def event86496 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨46333⟩⟩) 86495 exact86496RawTerms .large 86492 .exactZero (none)

def event86497 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨47498⟩⟩)

def event86498 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event86499 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event86500 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event86501 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event86502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event86503 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event86504 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event86505 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event86506 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 86505

def event86507 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 86503

def event86508 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 86506 .coefficient) (.value (.predecessor 1 86507 .coefficient)))

def event86509 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event86510 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 86509

def event86511 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 86501

def event86512 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 86510 .coefficient, .predecessor 1 86511 .coefficient])

def event86513 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event86514 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 86513

def event86515 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 86499

def event86516 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 86515 .coefficient))

def event86517 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event86518 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45298⟩⟩) 0 ⟨10325⟩ 86517

def event86519 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45298⟩⟩) (.authority (.programFamilyFact))

def exact86520RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45298⟩⟩], []⟩, (1)⟩]

theorem exact86520RawTermsValid :
    exact86520RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86520 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45298⟩⟩) exact86520RawTerms (.finite 58) 86519 .exactZero (none)

def event86521 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14871⟩⟩) 0 ⟨10325⟩ 86517

def event86522 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14871⟩⟩) (.authority (.programFamilyFact))

def exact86523RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩], []⟩, (1)⟩]

theorem exact86523RawTermsValid :
    exact86523RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86523 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14871⟩⟩) exact86523RawTerms (.finite 58) 86522 .exactZero (none)

def event86524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45299⟩⟩) 0 ⟨14871⟩ 86523

def event86525 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45299⟩⟩) 1 ⟨45298⟩ 86520

def event86526 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45299⟩⟩) (.product (.predecessor 0 86524 .coefficient) (.predecessor 1 86525 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event86527 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45299⟩⟩, .operator (⟨86523, 0⟩, ⟨86520, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], []⟩, (1)⟩)

def eventLeaf5392 : Array AnnotatedEvent := #[
  { event := event86272
    frameStart := 86231 },
  { event := event86273
    frameStart := 86231 },
  { event := event86274
    frameStart := 86231 },
  { event := event86275
    frameStart := 86231 },
  { event := event86276
    frameStart := 86231 },
  { event := event86277
    frameStart := 86231 },
  { event := event86278
    frameStart := 86231 },
  { event := event86279
    frameStart := 86231 },
  { event := event86280
    frameStart := 86231 },
  { event := event86281
    frameStart := 86231 },
  { event := event86282
    frameStart := 86231 },
  { event := event86283
    frameStart := 86231 },
  { event := event86284
    frameStart := 86231 },
  { event := event86285
    frameStart := 86285 },
  { event := event86286
    frameStart := 86285 },
  { event := event86287
    frameStart := 86285 }
]

def eventLeaf5393 : Array AnnotatedEvent := #[
  { event := event86288
    frameStart := 86285 },
  { event := event86289
    frameStart := 86285 },
  { event := event86290
    frameStart := 86285 },
  { event := event86291
    frameStart := 86285 },
  { event := event86292
    frameStart := 86285 },
  { event := event86293
    frameStart := 86285 },
  { event := event86294
    frameStart := 86285 },
  { event := event86295
    frameStart := 86285 },
  { event := event86296
    frameStart := 86285 },
  { event := event86297
    frameStart := 86285 },
  { event := event86298
    frameStart := 86285 },
  { event := event86299
    frameStart := 86285 },
  { event := event86300
    frameStart := 86285 },
  { event := event86301
    frameStart := 86285 },
  { event := event86302
    frameStart := 86285 },
  { event := event86303
    frameStart := 86285 }
]

def eventLeaf5394 : Array AnnotatedEvent := #[
  { event := event86304
    frameStart := 86285 },
  { event := event86305
    frameStart := 86285 },
  { event := event86306
    frameStart := 86285 },
  { event := event86307
    frameStart := 86285 },
  { event := event86308
    frameStart := 86285 },
  { event := event86309
    frameStart := 86285 },
  { event := event86310
    frameStart := 86285 },
  { event := event86311
    frameStart := 86285 },
  { event := event86312
    frameStart := 86285 },
  { event := event86313
    frameStart := 86285 },
  { event := event86314
    frameStart := 86285 },
  { event := event86315
    frameStart := 86285 },
  { event := event86316
    frameStart := 86285 },
  { event := event86317
    frameStart := 86285 },
  { event := event86318
    frameStart := 86285 },
  { event := event86319
    frameStart := 86285 }
]

def eventLeaf5395 : Array AnnotatedEvent := #[
  { event := event86320
    frameStart := 86285 },
  { event := event86321
    frameStart := 86285 },
  { event := event86322
    frameStart := 86285 },
  { event := event86323
    frameStart := 86285 },
  { event := event86324
    frameStart := 86285 },
  { event := event86325
    frameStart := 86285 },
  { event := event86326
    frameStart := 86285 },
  { event := event86327
    frameStart := 86285 },
  { event := event86328
    frameStart := 86285 },
  { event := event86329
    frameStart := 86285 },
  { event := event86330
    frameStart := 86285 },
  { event := event86331
    frameStart := 86285 },
  { event := event86332
    frameStart := 86285 },
  { event := event86333
    frameStart := 86285 },
  { event := event86334
    frameStart := 86285 },
  { event := event86335
    frameStart := 86285 }
]

def eventLeaf5396 : Array AnnotatedEvent := #[
  { event := event86336
    frameStart := 86285 },
  { event := event86337
    frameStart := 86285 },
  { event := event86338
    frameStart := 86285 },
  { event := event86339
    frameStart := 86285 },
  { event := event86340
    frameStart := 86285 },
  { event := event86341
    frameStart := 86285 },
  { event := event86342
    frameStart := 86285 },
  { event := event86343
    frameStart := 86285 },
  { event := event86344
    frameStart := 86285 },
  { event := event86345
    frameStart := 86285 },
  { event := event86346
    frameStart := 86285 },
  { event := event86347
    frameStart := 86285 },
  { event := event86348
    frameStart := 86285 },
  { event := event86349
    frameStart := 86285 },
  { event := event86350
    frameStart := 86285 },
  { event := event86351
    frameStart := 86285 }
]

def eventLeaf5397 : Array AnnotatedEvent := #[
  { event := event86352
    frameStart := 86285 },
  { event := event86353
    frameStart := 86285 },
  { event := event86354
    frameStart := 86285 },
  { event := event86355
    frameStart := 86285 },
  { event := event86356
    frameStart := 86285 },
  { event := event86357
    frameStart := 86285 },
  { event := event86358
    frameStart := 86285 },
  { event := event86359
    frameStart := 86285 },
  { event := event86360
    frameStart := 86285 },
  { event := event86361
    frameStart := 86285 },
  { event := event86362
    frameStart := 86285 },
  { event := event86363
    frameStart := 86285 },
  { event := event86364
    frameStart := 86285 },
  { event := event86365
    frameStart := 86285 },
  { event := event86366
    frameStart := 86285 },
  { event := event86367
    frameStart := 86285 }
]

def eventLeaf5398 : Array AnnotatedEvent := #[
  { event := event86368
    frameStart := 86285 },
  { event := event86369
    frameStart := 86285 },
  { event := event86370
    frameStart := 86285 },
  { event := event86371
    frameStart := 86285 },
  { event := event86372
    frameStart := 86285 },
  { event := event86373
    frameStart := 86285 },
  { event := event86374
    frameStart := 86285 },
  { event := event86375
    frameStart := 86285 },
  { event := event86376
    frameStart := 86285 },
  { event := event86377
    frameStart := 86285 },
  { event := event86378
    frameStart := 86285 },
  { event := event86379
    frameStart := 86285 },
  { event := event86380
    frameStart := 86285 },
  { event := event86381
    frameStart := 86285 },
  { event := event86382
    frameStart := 86285 },
  { event := event86383
    frameStart := 86285 }
]

def eventLeaf5399 : Array AnnotatedEvent := #[
  { event := event86384
    frameStart := 86285 },
  { event := event86385
    frameStart := 86285 },
  { event := event86386
    frameStart := 86285 },
  { event := event86387
    frameStart := 86285 },
  { event := event86388
    frameStart := 86285 },
  { event := event86389
    frameStart := 0 },
  { event := event86390
    frameStart := 0 },
  { event := event86391
    frameStart := 0 },
  { event := event86392
    frameStart := 0 },
  { event := event86393
    frameStart := 0 },
  { event := event86394
    frameStart := 0 },
  { event := event86395
    frameStart := 0 },
  { event := event86396
    frameStart := 0 },
  { event := event86397
    frameStart := 0 },
  { event := event86398
    frameStart := 0 },
  { event := event86399
    frameStart := 0 }
]

def eventLeaf5400 : Array AnnotatedEvent := #[
  { event := event86400
    frameStart := 0 },
  { event := event86401
    frameStart := 0 },
  { event := event86402
    frameStart := 0 },
  { event := event86403
    frameStart := 0 },
  { event := event86404
    frameStart := 0 },
  { event := event86405
    frameStart := 0 },
  { event := event86406
    frameStart := 0 },
  { event := event86407
    frameStart := 0 },
  { event := event86408
    frameStart := 0 },
  { event := event86409
    frameStart := 0 },
  { event := event86410
    frameStart := 0 },
  { event := event86411
    frameStart := 0 },
  { event := event86412
    frameStart := 0 },
  { event := event86413
    frameStart := 0 },
  { event := event86414
    frameStart := 0 },
  { event := event86415
    frameStart := 0 }
]

def eventLeaf5401 : Array AnnotatedEvent := #[
  { event := event86416
    frameStart := 0 },
  { event := event86417
    frameStart := 0 },
  { event := event86418
    frameStart := 0 },
  { event := event86419
    frameStart := 0 },
  { event := event86420
    frameStart := 0 },
  { event := event86421
    frameStart := 0 },
  { event := event86422
    frameStart := 0 },
  { event := event86423
    frameStart := 0 },
  { event := event86424
    frameStart := 0 },
  { event := event86425
    frameStart := 0 },
  { event := event86426
    frameStart := 0 },
  { event := event86427
    frameStart := 0 },
  { event := event86428
    frameStart := 0 },
  { event := event86429
    frameStart := 0 },
  { event := event86430
    frameStart := 0 },
  { event := event86431
    frameStart := 0 }
]

def eventLeaf5402 : Array AnnotatedEvent := #[
  { event := event86432
    frameStart := 0 },
  { event := event86433
    frameStart := 0 },
  { event := event86434
    frameStart := 0 },
  { event := event86435
    frameStart := 0 },
  { event := event86436
    frameStart := 0 },
  { event := event86437
    frameStart := 0 },
  { event := event86438
    frameStart := 0 },
  { event := event86439
    frameStart := 0 },
  { event := event86440
    frameStart := 0 },
  { event := event86441
    frameStart := 0 },
  { event := event86442
    frameStart := 0 },
  { event := event86443
    frameStart := 86443 },
  { event := event86444
    frameStart := 86443 },
  { event := event86445
    frameStart := 86443 },
  { event := event86446
    frameStart := 86443 },
  { event := event86447
    frameStart := 86443 }
]

def eventLeaf5403 : Array AnnotatedEvent := #[
  { event := event86448
    frameStart := 86443 },
  { event := event86449
    frameStart := 86443 },
  { event := event86450
    frameStart := 86443 },
  { event := event86451
    frameStart := 86443 },
  { event := event86452
    frameStart := 86443 },
  { event := event86453
    frameStart := 86443 },
  { event := event86454
    frameStart := 86443 },
  { event := event86455
    frameStart := 86443 },
  { event := event86456
    frameStart := 86443 },
  { event := event86457
    frameStart := 86443 },
  { event := event86458
    frameStart := 86443 },
  { event := event86459
    frameStart := 86443 },
  { event := event86460
    frameStart := 86443 },
  { event := event86461
    frameStart := 86443 },
  { event := event86462
    frameStart := 86443 },
  { event := event86463
    frameStart := 86443 }
]

def eventLeaf5404 : Array AnnotatedEvent := #[
  { event := event86464
    frameStart := 86443 },
  { event := event86465
    frameStart := 86443 },
  { event := event86466
    frameStart := 86443 },
  { event := event86467
    frameStart := 86443 },
  { event := event86468
    frameStart := 86443 },
  { event := event86469
    frameStart := 86443 },
  { event := event86470
    frameStart := 86443 },
  { event := event86471
    frameStart := 86443 },
  { event := event86472
    frameStart := 86443 },
  { event := event86473
    frameStart := 86443 },
  { event := event86474
    frameStart := 86443 },
  { event := event86475
    frameStart := 86443 },
  { event := event86476
    frameStart := 86443 },
  { event := event86477
    frameStart := 86443 },
  { event := event86478
    frameStart := 86443 },
  { event := event86479
    frameStart := 86443 }
]

def eventLeaf5405 : Array AnnotatedEvent := #[
  { event := event86480
    frameStart := 86443 },
  { event := event86481
    frameStart := 86443 },
  { event := event86482
    frameStart := 86443 },
  { event := event86483
    frameStart := 86443 },
  { event := event86484
    frameStart := 86443 },
  { event := event86485
    frameStart := 86443 },
  { event := event86486
    frameStart := 86443 },
  { event := event86487
    frameStart := 86443 },
  { event := event86488
    frameStart := 86443 },
  { event := event86489
    frameStart := 86443 },
  { event := event86490
    frameStart := 86443 },
  { event := event86491
    frameStart := 86443 },
  { event := event86492
    frameStart := 86443 },
  { event := event86493
    frameStart := 86443 },
  { event := event86494
    frameStart := 86443 },
  { event := event86495
    frameStart := 86443 }
]

def eventLeaf5406 : Array AnnotatedEvent := #[
  { event := event86496
    frameStart := 86443 },
  { event := event86497
    frameStart := 86497 },
  { event := event86498
    frameStart := 86497 },
  { event := event86499
    frameStart := 86497 },
  { event := event86500
    frameStart := 86497 },
  { event := event86501
    frameStart := 86497 },
  { event := event86502
    frameStart := 86497 },
  { event := event86503
    frameStart := 86497 },
  { event := event86504
    frameStart := 86497 },
  { event := event86505
    frameStart := 86497 },
  { event := event86506
    frameStart := 86497 },
  { event := event86507
    frameStart := 86497 },
  { event := event86508
    frameStart := 86497 },
  { event := event86509
    frameStart := 86497 },
  { event := event86510
    frameStart := 86497 },
  { event := event86511
    frameStart := 86497 }
]

def eventLeaf5407 : Array AnnotatedEvent := #[
  { event := event86512
    frameStart := 86497 },
  { event := event86513
    frameStart := 86497 },
  { event := event86514
    frameStart := 86497 },
  { event := event86515
    frameStart := 86497 },
  { event := event86516
    frameStart := 86497 },
  { event := event86517
    frameStart := 86497 },
  { event := event86518
    frameStart := 86497 },
  { event := event86519
    frameStart := 86497 },
  { event := event86520
    frameStart := 86497 },
  { event := event86521
    frameStart := 86497 },
  { event := event86522
    frameStart := 86497 },
  { event := event86523
    frameStart := 86497 },
  { event := event86524
    frameStart := 86497 },
  { event := event86525
    frameStart := 86497 },
  { event := event86526
    frameStart := 86497 },
  { event := event86527
    frameStart := 86497 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events337
