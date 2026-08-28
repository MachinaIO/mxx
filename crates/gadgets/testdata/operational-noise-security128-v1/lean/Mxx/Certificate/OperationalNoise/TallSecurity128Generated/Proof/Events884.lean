import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events884

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact226304RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68057⟩⟩]⟩, (1)⟩]

theorem exact226304RawTermsValid :
    exact226304RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226304 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68059⟩⟩) exact226304RawTerms (.finite 5647228698) 226303 .exactZero (none)

def event226305 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68060⟩⟩) 0 ⟨5581⟩ 222245

def event226306 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68060⟩⟩) 1 ⟨68059⟩ 226304

def event226307 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68060⟩⟩) (.product (.predecessor 0 226305 .coefficient) (.predecessor 1 226306 .coefficient) (⟨false, false, none, none, none⟩))

def event226308 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68060⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨68057⟩⟩]⟩) [⟨.result 226300 .coefficient, false, none⟩])

def event226309 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68060⟩⟩) (.product (.result 222245 .summary) (.transfer 226308) (⟨false, false, none, none, none⟩))

def event226310 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68060⟩⟩, .operator (⟨222245, 0⟩, ⟨226304, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68057⟩⟩]⟩, (1)⟩)

def event226311 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨68058⟩⟩)

def event226312 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event226313 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event226314 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.authority (.operator))

def event226315 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.finite 5)

def event226316 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event226317 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event226318 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event226319 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event226320 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 226319

def event226321 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 226317

def event226322 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 226320 .coefficient) (.value (.predecessor 1 226321 .coefficient)))

def event226323 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event226324 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 0 ⟨392⟩ 226323

def event226325 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 1 ⟨4990⟩ 226315

def event226326 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.sum [.predecessor 0 226324 .coefficient, .predecessor 1 226325 .coefficient])

def event226327 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.finite 655345)

def event226328 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 0 ⟨4992⟩ 226327

def event226329 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 1 ⟨5426⟩ 226313

def event226330 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.identity (.predecessor 1 226329 .coefficient))

def event226331 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.finite 655360)

def event226332 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25718⟩⟩) 0 ⟨5577⟩ 226331

def event226333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25718⟩⟩) (.authority (.programFamilyFact))

def exact226334RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25718⟩⟩], []⟩, (1)⟩]

theorem exact226334RawTermsValid :
    exact226334RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226334 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25718⟩⟩) exact226334RawTerms (.finite 28) 226333 .exactZero (none)

def event226335 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65418⟩⟩) 0 ⟨5577⟩ 226331

def event226336 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65418⟩⟩) (.authority (.programFamilyFact))

def exact226337RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65418⟩⟩], []⟩, (1)⟩]

theorem exact226337RawTermsValid :
    exact226337RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226337 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65418⟩⟩) exact226337RawTerms (.finite 28) 226336 .exactZero (none)

def event226338 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65419⟩⟩) 0 ⟨65418⟩ 226337

def event226339 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65419⟩⟩) 1 ⟨25718⟩ 226334

def event226340 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65419⟩⟩) (.product (.predecessor 0 226338 .coefficient) (.predecessor 1 226339 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event226341 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65419⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], []⟩) [⟨.result 226337 .coefficient, true, some 1⟩, ⟨.result 226334 .coefficient, true, some 1⟩])

def event226342 : Event := .survivorFold (1) 226341

def exact226343RawTerms : List Term := []

theorem exact226343RawTermsValid :
    exact226343RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226343 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65419⟩⟩) exact226343RawTerms (.finite 784) 226340 (.finite 784) (some (226341))

def event226344 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65420⟩⟩) 0 ⟨65419⟩ 226343

def event226345 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65420⟩⟩) (.identity (.predecessor 0 226344 .coefficient))

def event226346 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65420⟩⟩) (.finite 784)

def event226347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65780⟩⟩) 0 ⟨65420⟩ 226346

def event226348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65780⟩⟩) (.authority (.programFamilyFact))

def exact226349RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65780⟩⟩], []⟩, (1)⟩]

theorem exact226349RawTermsValid :
    exact226349RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226349 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65780⟩⟩) exact226349RawTerms (.finite 28) 226348 .exactZero (none)

def event226350 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65781⟩⟩) 0 ⟨65780⟩ 226349

def event226351 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65781⟩⟩) (.identity (.predecessor 0 226350 .coefficient))

def event226352 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65781⟩⟩) (.finite 28)

def event226353 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68057⟩⟩) 0 ⟨65781⟩ 226352

def event226354 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68057⟩⟩) (.authority (.relationPreimageSource ⟨76⟩))

def exact226355RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68057⟩⟩]⟩, (1)⟩]

theorem exact226355RawTermsValid :
    exact226355RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226355 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68057⟩⟩) exact226355RawTerms (.finite 5647228698) 226354 .exactZero (none)

def event226356 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact226357RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact226357RawTermsValid :
    exact226357RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226357 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact226357RawTerms .large 226356 .exactZero (none)

def event226358 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68058⟩⟩) 0 ⟨35⟩ 226357

def event226359 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68058⟩⟩) 1 ⟨68057⟩ 226355

def event226360 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68058⟩⟩) (.product (.predecessor 0 226358 .coefficient) (.predecessor 1 226359 .coefficient) (⟨false, false, none, none, none⟩))

def event226361 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68058⟩⟩, .operator (⟨226357, 0⟩, ⟨226355, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68057⟩⟩]⟩, (1)⟩)

def exact226362RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68057⟩⟩]⟩, (1)⟩]

theorem exact226362RawTermsValid :
    exact226362RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226362 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68058⟩⟩) exact226362RawTerms .large 226360 .exactZero (none)

def event226363 : Event := .preFoldPolynomial 226362 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68057⟩⟩]⟩, (1)⟩] .exactZero none

def exact226364RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68057⟩⟩]⟩, (1)⟩]

def event226364 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨68058⟩⟩) 226363 exact226364RawTerms .large 226360 .exactZero (none)

def event226365 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨70111⟩⟩)

def event226366 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event226367 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event226368 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.authority (.operator))

def event226369 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.finite 5)

def event226370 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event226371 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event226372 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event226373 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event226374 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 226373

def event226375 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 226371

def event226376 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 226374 .coefficient) (.value (.predecessor 1 226375 .coefficient)))

def event226377 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event226378 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 0 ⟨392⟩ 226377

def event226379 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 1 ⟨4990⟩ 226369

def event226380 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.sum [.predecessor 0 226378 .coefficient, .predecessor 1 226379 .coefficient])

def event226381 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.finite 655345)

def event226382 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 0 ⟨4992⟩ 226381

def event226383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 1 ⟨5426⟩ 226367

def event226384 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.identity (.predecessor 1 226383 .coefficient))

def event226385 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.finite 655360)

def event226386 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25718⟩⟩) 0 ⟨5577⟩ 226385

def event226387 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25718⟩⟩) (.authority (.programFamilyFact))

def exact226388RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25718⟩⟩], []⟩, (1)⟩]

theorem exact226388RawTermsValid :
    exact226388RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226388 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25718⟩⟩) exact226388RawTerms (.finite 28) 226387 .exactZero (none)

def event226389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65418⟩⟩) 0 ⟨5577⟩ 226385

def event226390 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65418⟩⟩) (.authority (.programFamilyFact))

def exact226391RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65418⟩⟩], []⟩, (1)⟩]

theorem exact226391RawTermsValid :
    exact226391RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226391 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65418⟩⟩) exact226391RawTerms (.finite 28) 226390 .exactZero (none)

def event226392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65419⟩⟩) 0 ⟨65418⟩ 226391

def event226393 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65419⟩⟩) 1 ⟨25718⟩ 226388

def event226394 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65419⟩⟩) (.product (.predecessor 0 226392 .coefficient) (.predecessor 1 226393 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event226395 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65419⟩⟩, .operator (⟨226391, 0⟩, ⟨226388, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], []⟩, (1)⟩)

def exact226396RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], []⟩, (1)⟩]

theorem exact226396RawTermsValid :
    exact226396RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226396 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65419⟩⟩) exact226396RawTerms (.finite 784) 226394 .exactZero (none)

def event226397 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65420⟩⟩) 0 ⟨65419⟩ 226396

def event226398 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65420⟩⟩) (.identity (.predecessor 0 226397 .coefficient))

def event226399 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65420⟩⟩) (.finite 784)

def event226400 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65780⟩⟩) 0 ⟨65420⟩ 226399

def event226401 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65780⟩⟩) (.authority (.programFamilyFact))

def exact226402RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65780⟩⟩], []⟩, (1)⟩]

theorem exact226402RawTermsValid :
    exact226402RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226402 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65780⟩⟩) exact226402RawTerms (.finite 28) 226401 .exactZero (none)

def event226403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65781⟩⟩) 0 ⟨65780⟩ 226402

def event226404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65781⟩⟩) (.identity (.predecessor 0 226403 .coefficient))

def event226405 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65781⟩⟩) (.finite 28)

def event226406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68671⟩⟩) 0 ⟨65781⟩ 226405

def event226407 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68671⟩⟩) (.authority (.programFamilyFact))

def event226408 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68671⟩⟩) (.finite 3720)

def event226409 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event226410 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68673⟩⟩) 0 ⟨7177⟩ 226409

def event226411 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68673⟩⟩) 1 ⟨68671⟩ 226408

def event226412 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68673⟩⟩) (.authority (.operator))

def exact226413RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68673⟩⟩]⟩, (1)⟩]

theorem exact226413RawTermsValid :
    exact226413RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226413 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68673⟩⟩) exact226413RawTerms .large 226412 .exactZero (none)

def event226414 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70098⟩⟩) 0 ⟨68673⟩ 226413

def event226415 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70098⟩⟩) (.authority (.operator))

def exact226416RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨70098⟩⟩]⟩, (1)⟩]

theorem exact226416RawTermsValid :
    exact226416RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226416 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70098⟩⟩) exact226416RawTerms (.finite 8192) 226415 .exactZero (none)

def event226417 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event226418 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event226419 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69003⟩⟩) 0 ⟨65781⟩ 226405

def event226420 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69003⟩⟩) 1 ⟨136⟩ 226418

def event226421 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69003⟩⟩) (.sum [.predecessor 0 226419 .coefficient, .predecessor 1 226420 .coefficient])

def event226422 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨69003⟩⟩) (.finite 28)

def event226423 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69004⟩⟩) 0 ⟨69003⟩ 226422

def event226424 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69004⟩⟩) (.identity (.predecessor 0 226423 .coefficient))

def exact226425RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65780⟩⟩], []⟩, (1)⟩]

theorem exact226425RawTermsValid :
    exact226425RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226425 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69004⟩⟩) exact226425RawTerms (.finite 28) 226424 .exactZero (none)

def event226426 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact226427RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact226427RawTermsValid :
    exact226427RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226427 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact226427RawTerms .large 226426 .exactZero (none)

def event226428 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69005⟩⟩) 0 ⟨6908⟩ 226427

def event226429 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69005⟩⟩) 1 ⟨69004⟩ 226425

def event226430 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69005⟩⟩) (.product (.predecessor 0 226428 .coefficient) (.predecessor 1 226429 .coefficient) (⟨false, false, none, none, none⟩))

def event226431 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69005⟩⟩, .operator (⟨226427, 0⟩, ⟨226425, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨65780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact226432RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact226432RawTermsValid :
    exact226432RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226432 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69005⟩⟩) exact226432RawTerms .large 226430 .exactZero (none)

def event226433 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7188⟩⟩) 0 ⟨7177⟩ 226409

def event226434 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7188⟩⟩) (.authority (.operator))

def exact226435RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩]

theorem exact226435RawTermsValid :
    exact226435RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226435 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7188⟩⟩) exact226435RawTerms .large 226434 .exactZero (none)

def event226436 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69006⟩⟩) 0 ⟨7188⟩ 226435

def event226437 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69006⟩⟩) 1 ⟨69005⟩ 226432

def event226438 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69006⟩⟩) (.sum [.predecessor 0 226436 .coefficient, .predecessor 1 226437 .coefficient])

def exact226439RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact226439RawTermsValid :
    exact226439RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226439 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69006⟩⟩) exact226439RawTerms .large 226438 .exactZero (none)

def event226440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70099⟩⟩) 0 ⟨69006⟩ 226439

def event226441 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70099⟩⟩) 1 ⟨70098⟩ 226416

def event226442 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70099⟩⟩) (.product (.predecessor 0 226440 .coefficient) (.predecessor 1 226441 .coefficient) (⟨false, false, none, none, none⟩))

def event226443 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70099⟩⟩, .operator (⟨226439, 0⟩, ⟨226416, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70098⟩⟩]⟩, (1)⟩)

def event226444 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70099⟩⟩, .operator (⟨226439, 1⟩, ⟨226416, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨65780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70098⟩⟩]⟩, (-1)⟩)

def event226445 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70099⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨65780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70098⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70098⟩⟩) ⟨68673⟩ 226413)

def event226446 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70099⟩⟩, .relation 226445 0, ⟨[⟨.program ⟨257⟩, ⟨65780⟩⟩], [⟨.program ⟨257⟩, ⟨68673⟩⟩]⟩, (-1)⟩)

def exact226447RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70098⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65780⟩⟩], [⟨.program ⟨257⟩, ⟨68673⟩⟩]⟩, (-1)⟩]

theorem exact226447RawTermsValid :
    exact226447RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226447 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70099⟩⟩) exact226447RawTerms .large 226442 .exactZero (none)

def event226448 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66531⟩⟩) 0 ⟨65781⟩ 226405

def event226449 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66531⟩⟩) (.authority (.programFamilyFact))

def exact226450RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨66531⟩⟩], []⟩, (1)⟩]

theorem exact226450RawTermsValid :
    exact226450RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226450 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66531⟩⟩) exact226450RawTerms (.finite 62) 226449 .exactZero (none)

def event226451 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66542⟩⟩) 0 ⟨6908⟩ 226427

def event226452 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66542⟩⟩) 1 ⟨66531⟩ 226450

def event226453 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66542⟩⟩) (.product (.predecessor 0 226451 .coefficient) (.predecessor 1 226452 .coefficient) (⟨false, true, none, none, some 1⟩))

def event226454 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨66542⟩⟩, .operator (⟨226427, 0⟩, ⟨226450, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact226455RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact226455RawTermsValid :
    exact226455RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226455 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66542⟩⟩) exact226455RawTerms .large 226453 .exactZero (none)

def event226456 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7216⟩⟩) 0 ⟨7177⟩ 226409

def event226457 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7216⟩⟩) (.authority (.operator))

def exact226458RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩]

theorem exact226458RawTermsValid :
    exact226458RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226458 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7216⟩⟩) exact226458RawTerms .large 226457 .exactZero (none)

def event226459 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66543⟩⟩) 0 ⟨7216⟩ 226458

def event226460 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66543⟩⟩) 1 ⟨66542⟩ 226455

def event226461 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66543⟩⟩) (.sum [.predecessor 0 226459 .coefficient, .predecessor 1 226460 .coefficient])

def exact226462RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact226462RawTermsValid :
    exact226462RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226462 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66543⟩⟩) exact226462RawTerms .large 226461 .exactZero (none)

def event226463 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70111⟩⟩) 0 ⟨66543⟩ 226462

def event226464 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70111⟩⟩) 1 ⟨70099⟩ 226447

def event226465 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70111⟩⟩) (.sum [.predecessor 0 226463 .coefficient, .predecessor 1 226464 .coefficient])

def exact226466RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70098⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65780⟩⟩], [⟨.program ⟨257⟩, ⟨68673⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact226466RawTermsValid :
    exact226466RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226466 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70111⟩⟩) exact226466RawTerms .large 226465 .exactZero (none)

def event226467 : Event := .preFoldPolynomial 226466 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70098⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65780⟩⟩], [⟨.program ⟨257⟩, ⟨68673⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact226468RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70098⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65780⟩⟩], [⟨.program ⟨257⟩, ⟨68673⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event226468 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨70111⟩⟩) 226467 exact226468RawTerms .large 226465 .exactZero (none)

def event226469 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨65781⟩⟩) ⟨⟨95⟩, ⟨76⟩, ⟨135⟩⟩ ⟨226311, 226469⟩

def event226470 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨68060⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68057⟩⟩]⟩) (1) 0 2 (.universal 226469 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68057⟩⟩]⟩) (none) 226468)

def event226471 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68060⟩⟩, .relation 226470 1, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩)

def event226472 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68060⟩⟩, .relation 226470 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70098⟩⟩]⟩, (-1)⟩)

def event226473 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68060⟩⟩, .relation 226470 2, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨65780⟩⟩], [⟨.program ⟨257⟩, ⟨68673⟩⟩]⟩, (1)⟩)

def event226474 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68060⟩⟩, .relation 226470 3, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact226475RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70098⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨65780⟩⟩], [⟨.program ⟨257⟩, ⟨68673⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact226475RawTermsValid :
    exact226475RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226475 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68060⟩⟩) exact226475RawTerms .large 226307 (.finite 202072841853861888) (some (226309))

def event226476 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70101⟩⟩) 0 ⟨68060⟩ 226475

def event226477 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70101⟩⟩) 1 ⟨70100⟩ 226297

def event226478 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70101⟩⟩) (.sum [.predecessor 0 226476 .coefficient, .predecessor 1 226477 .coefficient])

def event226479 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70101⟩⟩, .operator (⟨226475, 0⟩, ⟨226297, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70098⟩⟩]⟩, (1)⟩)

def event226480 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70101⟩⟩, .operator (⟨226475, 2⟩, ⟨226297, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨65780⟩⟩], [⟨.program ⟨257⟩, ⟨68673⟩⟩]⟩, (-1)⟩)

def event226481 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70101⟩⟩) (.sum [.result 226475 .summary, .result 226297 .summary])

def exact226482RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact226482RawTermsValid :
    exact226482RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226482 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70101⟩⟩) exact226482RawTerms .large 226478 (.finite 32191361068277642793642192273408) (some (226481))

def event226483 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64070⟩⟩) 0 ⟨62801⟩ 10790

def event226484 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64070⟩⟩) (.authority (.programFamilyFact))

def event226485 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨64070⟩⟩) (.finite 3720)

def event226486 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64072⟩⟩) 0 ⟨7177⟩ 15500

def event226487 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64072⟩⟩) 1 ⟨64070⟩ 226485

def event226488 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64072⟩⟩) (.authority (.operator))

def exact226489RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨64072⟩⟩]⟩, (1)⟩]

theorem exact226489RawTermsValid :
    exact226489RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226489 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64072⟩⟩) exact226489RawTerms .large 226488 .exactZero (none)

def event226490 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64841⟩⟩) 0 ⟨64072⟩ 226489

def event226491 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64841⟩⟩) (.authority (.operator))

def exact226492RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨64841⟩⟩]⟩, (1)⟩]

theorem exact226492RawTermsValid :
    exact226492RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226492 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64841⟩⟩) exact226492RawTerms (.finite 8192) 226491 .exactZero (none)

def event226493 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63922⟩⟩) 0 ⟨62440⟩ 10784

def event226494 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63922⟩⟩) (.authority (.programFamilyFact))

def event226495 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨63922⟩⟩) (.finite 3720)

def event226496 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63923⟩⟩) 0 ⟨7177⟩ 15500

def event226497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63923⟩⟩) 1 ⟨63922⟩ 226495

def event226498 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63923⟩⟩) (.authority (.operator))

def exact226499RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨63923⟩⟩]⟩, (1)⟩]

theorem exact226499RawTermsValid :
    exact226499RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226499 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63923⟩⟩) exact226499RawTerms .large 226498 .exactZero (none)

def event226500 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64428⟩⟩) 0 ⟨63923⟩ 226499

def event226501 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64428⟩⟩) (.authority (.operator))

def exact226502RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨64428⟩⟩]⟩, (1)⟩]

theorem exact226502RawTermsValid :
    exact226502RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226502 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64428⟩⟩) exact226502RawTerms (.finite 8192) 226501 .exactZero (none)

def event226503 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25479⟩⟩) 0 ⟨25478⟩ 10773

def event226504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25479⟩⟩) 1 ⟨6937⟩ 222153

def event226505 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25479⟩⟩) (.tensor (.predecessor 0 226503 .coefficient) (.predecessor 1 226504 .coefficient) true false)

def event226506 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨25479⟩⟩, .operator (⟨10773, 0⟩, ⟨222153, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact226507RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact226507RawTermsValid :
    exact226507RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226507 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25479⟩⟩) exact226507RawTerms .large 226505 .exactZero (none)

def event226508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8467⟩⟩) 0 ⟨5579⟩ 222023

def event226509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8467⟩⟩) 1 ⟨7275⟩ 21589

def event226510 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8467⟩⟩) (.product (.predecessor 0 226508 .coefficient) (.predecessor 1 226509 .coefficient) (⟨false, false, none, none, none⟩))

def event226511 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8467⟩⟩, .operator (⟨222023, 0⟩, ⟨21589, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩)

def exact226512RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩]

theorem exact226512RawTermsValid :
    exact226512RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226512 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8467⟩⟩) exact226512RawTerms .large 226510 .exactZero (none)

def event226513 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25480⟩⟩) 0 ⟨8467⟩ 226512

def event226514 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25480⟩⟩) 1 ⟨25479⟩ 226507

def event226515 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25480⟩⟩) (.sum [.predecessor 0 226513 .coefficient, .predecessor 1 226514 .coefficient])

def exact226516RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact226516RawTermsValid :
    exact226516RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226516 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25480⟩⟩) exact226516RawTerms .large 226515 .exactZero (none)

def event226517 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25481⟩⟩) 0 ⟨25480⟩ 226516

def event226518 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25481⟩⟩) 1 ⟨101⟩ 21581

def event226519 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25481⟩⟩) (.sum [.predecessor 0 226517 .coefficient, .predecessor 1 226518 .coefficient])

def event226520 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25481⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨101⟩⟩]⟩) [⟨.result 21581 .coefficient, false, none⟩])

def event226521 : Event := .survivorFold (1) 226520

def exact226522RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact226522RawTermsValid :
    exact226522RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226522 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25481⟩⟩) exact226522RawTerms .large 226519 (.finite 26) (some (226520))

def event226523 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62441⟩⟩) 0 ⟨25481⟩ 226522

def event226524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62441⟩⟩) 1 ⟨62438⟩ 10776

def event226525 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62441⟩⟩) (.product (.predecessor 0 226523 .coefficient) (.predecessor 1 226524 .coefficient) (⟨false, true, none, none, some 1⟩))

def event226526 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62441⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨62438⟩⟩], []⟩) [⟨.result 10776 .coefficient, true, some 1⟩])

def event226527 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62441⟩⟩) (.product (.result 226522 .summary) (.transfer 226526) (⟨false, false, none, none, none⟩))

def event226528 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62441⟩⟩, .operator (⟨226522, 1⟩, ⟨10776, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25478⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event226529 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62441⟩⟩, .operator (⟨226522, 0⟩, ⟨10776, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩)

def exact226530RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25478⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩]

theorem exact226530RawTermsValid :
    exact226530RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226530 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62441⟩⟩) exact226530RawTerms .large 226525 (.finite 18743296) (some (226527))

def event226531 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62442⟩⟩) 0 ⟨62438⟩ 10776

def event226532 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62442⟩⟩) 1 ⟨6937⟩ 222153

def event226533 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62442⟩⟩) (.tensor (.predecessor 0 226531 .coefficient) (.predecessor 1 226532 .coefficient) true false)

def event226534 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62442⟩⟩, .operator (⟨10776, 0⟩, ⟨222153, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact226535RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact226535RawTermsValid :
    exact226535RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226535 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62442⟩⟩) exact226535RawTerms .large 226533 .exactZero (none)

def event226536 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8485⟩⟩) 0 ⟨5579⟩ 222023

def event226537 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8485⟩⟩) 1 ⟨7293⟩ 21630

def event226538 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8485⟩⟩) (.product (.predecessor 0 226536 .coefficient) (.predecessor 1 226537 .coefficient) (⟨false, false, none, none, none⟩))

def event226539 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8485⟩⟩, .operator (⟨222023, 0⟩, ⟨21630, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩]⟩, (1)⟩)

def exact226540RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩]⟩, (1)⟩]

theorem exact226540RawTermsValid :
    exact226540RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226540 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8485⟩⟩) exact226540RawTerms .large 226538 .exactZero (none)

def event226541 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62443⟩⟩) 0 ⟨8485⟩ 226540

def event226542 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62443⟩⟩) 1 ⟨62442⟩ 226535

def event226543 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62443⟩⟩) (.sum [.predecessor 0 226541 .coefficient, .predecessor 1 226542 .coefficient])

def exact226544RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact226544RawTermsValid :
    exact226544RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226544 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62443⟩⟩) exact226544RawTerms .large 226543 .exactZero (none)

def event226545 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62444⟩⟩) 0 ⟨62443⟩ 226544

def event226546 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62444⟩⟩) 1 ⟨119⟩ 21622

def event226547 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62444⟩⟩) (.sum [.predecessor 0 226545 .coefficient, .predecessor 1 226546 .coefficient])

def event226548 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62444⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨119⟩⟩]⟩) [⟨.result 21622 .coefficient, false, none⟩])

def event226549 : Event := .survivorFold (1) 226548

def exact226550RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact226550RawTermsValid :
    exact226550RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event226550 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62444⟩⟩) exact226550RawTerms .large 226547 (.finite 26) (some (226548))

def event226551 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62445⟩⟩) 0 ⟨62444⟩ 226550

def event226552 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62445⟩⟩) 1 ⟨9539⟩ 21619

def event226553 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62445⟩⟩) (.product (.predecessor 0 226551 .coefficient) (.predecessor 1 226552 .coefficient) (⟨false, false, none, none, none⟩))

def event226554 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62445⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) [⟨.result 21615 .coefficient, false, none⟩])

def event226555 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62445⟩⟩) (.product (.result 226550 .summary) (.transfer 226554) (⟨false, false, none, none, none⟩))

def event226556 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62445⟩⟩, .operator (⟨226550, 1⟩, ⟨21619, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (-1)⟩)

def event226557 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨62445⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9538⟩⟩) ⟨7275⟩ 21589)

def event226558 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62445⟩⟩, .relation 226557 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (-1)⟩)

def event226559 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62445⟩⟩, .operator (⟨226550, 0⟩, ⟨21619, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (1)⟩)

def eventLeaf14144 : Array AnnotatedEvent := #[
  { event := event226304
    frameStart := 0 },
  { event := event226305
    frameStart := 0 },
  { event := event226306
    frameStart := 0 },
  { event := event226307
    frameStart := 0 },
  { event := event226308
    frameStart := 0 },
  { event := event226309
    frameStart := 0 },
  { event := event226310
    frameStart := 0 },
  { event := event226311
    frameStart := 226311 },
  { event := event226312
    frameStart := 226311 },
  { event := event226313
    frameStart := 226311 },
  { event := event226314
    frameStart := 226311 },
  { event := event226315
    frameStart := 226311 },
  { event := event226316
    frameStart := 226311 },
  { event := event226317
    frameStart := 226311 },
  { event := event226318
    frameStart := 226311 },
  { event := event226319
    frameStart := 226311 }
]

def eventLeaf14145 : Array AnnotatedEvent := #[
  { event := event226320
    frameStart := 226311 },
  { event := event226321
    frameStart := 226311 },
  { event := event226322
    frameStart := 226311 },
  { event := event226323
    frameStart := 226311 },
  { event := event226324
    frameStart := 226311 },
  { event := event226325
    frameStart := 226311 },
  { event := event226326
    frameStart := 226311 },
  { event := event226327
    frameStart := 226311 },
  { event := event226328
    frameStart := 226311 },
  { event := event226329
    frameStart := 226311 },
  { event := event226330
    frameStart := 226311 },
  { event := event226331
    frameStart := 226311 },
  { event := event226332
    frameStart := 226311 },
  { event := event226333
    frameStart := 226311 },
  { event := event226334
    frameStart := 226311 },
  { event := event226335
    frameStart := 226311 }
]

def eventLeaf14146 : Array AnnotatedEvent := #[
  { event := event226336
    frameStart := 226311 },
  { event := event226337
    frameStart := 226311 },
  { event := event226338
    frameStart := 226311 },
  { event := event226339
    frameStart := 226311 },
  { event := event226340
    frameStart := 226311 },
  { event := event226341
    frameStart := 226311 },
  { event := event226342
    frameStart := 226311 },
  { event := event226343
    frameStart := 226311 },
  { event := event226344
    frameStart := 226311 },
  { event := event226345
    frameStart := 226311 },
  { event := event226346
    frameStart := 226311 },
  { event := event226347
    frameStart := 226311 },
  { event := event226348
    frameStart := 226311 },
  { event := event226349
    frameStart := 226311 },
  { event := event226350
    frameStart := 226311 },
  { event := event226351
    frameStart := 226311 }
]

def eventLeaf14147 : Array AnnotatedEvent := #[
  { event := event226352
    frameStart := 226311 },
  { event := event226353
    frameStart := 226311 },
  { event := event226354
    frameStart := 226311 },
  { event := event226355
    frameStart := 226311 },
  { event := event226356
    frameStart := 226311 },
  { event := event226357
    frameStart := 226311 },
  { event := event226358
    frameStart := 226311 },
  { event := event226359
    frameStart := 226311 },
  { event := event226360
    frameStart := 226311 },
  { event := event226361
    frameStart := 226311 },
  { event := event226362
    frameStart := 226311 },
  { event := event226363
    frameStart := 226311 },
  { event := event226364
    frameStart := 226311 },
  { event := event226365
    frameStart := 226365 },
  { event := event226366
    frameStart := 226365 },
  { event := event226367
    frameStart := 226365 }
]

def eventLeaf14148 : Array AnnotatedEvent := #[
  { event := event226368
    frameStart := 226365 },
  { event := event226369
    frameStart := 226365 },
  { event := event226370
    frameStart := 226365 },
  { event := event226371
    frameStart := 226365 },
  { event := event226372
    frameStart := 226365 },
  { event := event226373
    frameStart := 226365 },
  { event := event226374
    frameStart := 226365 },
  { event := event226375
    frameStart := 226365 },
  { event := event226376
    frameStart := 226365 },
  { event := event226377
    frameStart := 226365 },
  { event := event226378
    frameStart := 226365 },
  { event := event226379
    frameStart := 226365 },
  { event := event226380
    frameStart := 226365 },
  { event := event226381
    frameStart := 226365 },
  { event := event226382
    frameStart := 226365 },
  { event := event226383
    frameStart := 226365 }
]

def eventLeaf14149 : Array AnnotatedEvent := #[
  { event := event226384
    frameStart := 226365 },
  { event := event226385
    frameStart := 226365 },
  { event := event226386
    frameStart := 226365 },
  { event := event226387
    frameStart := 226365 },
  { event := event226388
    frameStart := 226365 },
  { event := event226389
    frameStart := 226365 },
  { event := event226390
    frameStart := 226365 },
  { event := event226391
    frameStart := 226365 },
  { event := event226392
    frameStart := 226365 },
  { event := event226393
    frameStart := 226365 },
  { event := event226394
    frameStart := 226365 },
  { event := event226395
    frameStart := 226365 },
  { event := event226396
    frameStart := 226365 },
  { event := event226397
    frameStart := 226365 },
  { event := event226398
    frameStart := 226365 },
  { event := event226399
    frameStart := 226365 }
]

def eventLeaf14150 : Array AnnotatedEvent := #[
  { event := event226400
    frameStart := 226365 },
  { event := event226401
    frameStart := 226365 },
  { event := event226402
    frameStart := 226365 },
  { event := event226403
    frameStart := 226365 },
  { event := event226404
    frameStart := 226365 },
  { event := event226405
    frameStart := 226365 },
  { event := event226406
    frameStart := 226365 },
  { event := event226407
    frameStart := 226365 },
  { event := event226408
    frameStart := 226365 },
  { event := event226409
    frameStart := 226365 },
  { event := event226410
    frameStart := 226365 },
  { event := event226411
    frameStart := 226365 },
  { event := event226412
    frameStart := 226365 },
  { event := event226413
    frameStart := 226365 },
  { event := event226414
    frameStart := 226365 },
  { event := event226415
    frameStart := 226365 }
]

def eventLeaf14151 : Array AnnotatedEvent := #[
  { event := event226416
    frameStart := 226365 },
  { event := event226417
    frameStart := 226365 },
  { event := event226418
    frameStart := 226365 },
  { event := event226419
    frameStart := 226365 },
  { event := event226420
    frameStart := 226365 },
  { event := event226421
    frameStart := 226365 },
  { event := event226422
    frameStart := 226365 },
  { event := event226423
    frameStart := 226365 },
  { event := event226424
    frameStart := 226365 },
  { event := event226425
    frameStart := 226365 },
  { event := event226426
    frameStart := 226365 },
  { event := event226427
    frameStart := 226365 },
  { event := event226428
    frameStart := 226365 },
  { event := event226429
    frameStart := 226365 },
  { event := event226430
    frameStart := 226365 },
  { event := event226431
    frameStart := 226365 }
]

def eventLeaf14152 : Array AnnotatedEvent := #[
  { event := event226432
    frameStart := 226365 },
  { event := event226433
    frameStart := 226365 },
  { event := event226434
    frameStart := 226365 },
  { event := event226435
    frameStart := 226365 },
  { event := event226436
    frameStart := 226365 },
  { event := event226437
    frameStart := 226365 },
  { event := event226438
    frameStart := 226365 },
  { event := event226439
    frameStart := 226365 },
  { event := event226440
    frameStart := 226365 },
  { event := event226441
    frameStart := 226365 },
  { event := event226442
    frameStart := 226365 },
  { event := event226443
    frameStart := 226365 },
  { event := event226444
    frameStart := 226365 },
  { event := event226445
    frameStart := 226365 },
  { event := event226446
    frameStart := 226365 },
  { event := event226447
    frameStart := 226365 }
]

def eventLeaf14153 : Array AnnotatedEvent := #[
  { event := event226448
    frameStart := 226365 },
  { event := event226449
    frameStart := 226365 },
  { event := event226450
    frameStart := 226365 },
  { event := event226451
    frameStart := 226365 },
  { event := event226452
    frameStart := 226365 },
  { event := event226453
    frameStart := 226365 },
  { event := event226454
    frameStart := 226365 },
  { event := event226455
    frameStart := 226365 },
  { event := event226456
    frameStart := 226365 },
  { event := event226457
    frameStart := 226365 },
  { event := event226458
    frameStart := 226365 },
  { event := event226459
    frameStart := 226365 },
  { event := event226460
    frameStart := 226365 },
  { event := event226461
    frameStart := 226365 },
  { event := event226462
    frameStart := 226365 },
  { event := event226463
    frameStart := 226365 }
]

def eventLeaf14154 : Array AnnotatedEvent := #[
  { event := event226464
    frameStart := 226365 },
  { event := event226465
    frameStart := 226365 },
  { event := event226466
    frameStart := 226365 },
  { event := event226467
    frameStart := 226365 },
  { event := event226468
    frameStart := 226365 },
  { event := event226469
    frameStart := 0 },
  { event := event226470
    frameStart := 0 },
  { event := event226471
    frameStart := 0 },
  { event := event226472
    frameStart := 0 },
  { event := event226473
    frameStart := 0 },
  { event := event226474
    frameStart := 0 },
  { event := event226475
    frameStart := 0 },
  { event := event226476
    frameStart := 0 },
  { event := event226477
    frameStart := 0 },
  { event := event226478
    frameStart := 0 },
  { event := event226479
    frameStart := 0 }
]

def eventLeaf14155 : Array AnnotatedEvent := #[
  { event := event226480
    frameStart := 0 },
  { event := event226481
    frameStart := 0 },
  { event := event226482
    frameStart := 0 },
  { event := event226483
    frameStart := 0 },
  { event := event226484
    frameStart := 0 },
  { event := event226485
    frameStart := 0 },
  { event := event226486
    frameStart := 0 },
  { event := event226487
    frameStart := 0 },
  { event := event226488
    frameStart := 0 },
  { event := event226489
    frameStart := 0 },
  { event := event226490
    frameStart := 0 },
  { event := event226491
    frameStart := 0 },
  { event := event226492
    frameStart := 0 },
  { event := event226493
    frameStart := 0 },
  { event := event226494
    frameStart := 0 },
  { event := event226495
    frameStart := 0 }
]

def eventLeaf14156 : Array AnnotatedEvent := #[
  { event := event226496
    frameStart := 0 },
  { event := event226497
    frameStart := 0 },
  { event := event226498
    frameStart := 0 },
  { event := event226499
    frameStart := 0 },
  { event := event226500
    frameStart := 0 },
  { event := event226501
    frameStart := 0 },
  { event := event226502
    frameStart := 0 },
  { event := event226503
    frameStart := 0 },
  { event := event226504
    frameStart := 0 },
  { event := event226505
    frameStart := 0 },
  { event := event226506
    frameStart := 0 },
  { event := event226507
    frameStart := 0 },
  { event := event226508
    frameStart := 0 },
  { event := event226509
    frameStart := 0 },
  { event := event226510
    frameStart := 0 },
  { event := event226511
    frameStart := 0 }
]

def eventLeaf14157 : Array AnnotatedEvent := #[
  { event := event226512
    frameStart := 0 },
  { event := event226513
    frameStart := 0 },
  { event := event226514
    frameStart := 0 },
  { event := event226515
    frameStart := 0 },
  { event := event226516
    frameStart := 0 },
  { event := event226517
    frameStart := 0 },
  { event := event226518
    frameStart := 0 },
  { event := event226519
    frameStart := 0 },
  { event := event226520
    frameStart := 0 },
  { event := event226521
    frameStart := 0 },
  { event := event226522
    frameStart := 0 },
  { event := event226523
    frameStart := 0 },
  { event := event226524
    frameStart := 0 },
  { event := event226525
    frameStart := 0 },
  { event := event226526
    frameStart := 0 },
  { event := event226527
    frameStart := 0 }
]

def eventLeaf14158 : Array AnnotatedEvent := #[
  { event := event226528
    frameStart := 0 },
  { event := event226529
    frameStart := 0 },
  { event := event226530
    frameStart := 0 },
  { event := event226531
    frameStart := 0 },
  { event := event226532
    frameStart := 0 },
  { event := event226533
    frameStart := 0 },
  { event := event226534
    frameStart := 0 },
  { event := event226535
    frameStart := 0 },
  { event := event226536
    frameStart := 0 },
  { event := event226537
    frameStart := 0 },
  { event := event226538
    frameStart := 0 },
  { event := event226539
    frameStart := 0 },
  { event := event226540
    frameStart := 0 },
  { event := event226541
    frameStart := 0 },
  { event := event226542
    frameStart := 0 },
  { event := event226543
    frameStart := 0 }
]

def eventLeaf14159 : Array AnnotatedEvent := #[
  { event := event226544
    frameStart := 0 },
  { event := event226545
    frameStart := 0 },
  { event := event226546
    frameStart := 0 },
  { event := event226547
    frameStart := 0 },
  { event := event226548
    frameStart := 0 },
  { event := event226549
    frameStart := 0 },
  { event := event226550
    frameStart := 0 },
  { event := event226551
    frameStart := 0 },
  { event := event226552
    frameStart := 0 },
  { event := event226553
    frameStart := 0 },
  { event := event226554
    frameStart := 0 },
  { event := event226555
    frameStart := 0 },
  { event := event226556
    frameStart := 0 },
  { event := event226557
    frameStart := 0 },
  { event := event226558
    frameStart := 0 },
  { event := event226559
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events884
