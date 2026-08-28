import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events314

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event80384 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63430⟩⟩, .operator (⟨80380, 0⟩, ⟨80378, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63429⟩⟩]⟩, (1)⟩)

def exact80385RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63429⟩⟩]⟩, (1)⟩]

theorem exact80385RawTermsValid :
    exact80385RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80385 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63430⟩⟩) exact80385RawTerms .large 80383 .exactZero (none)

def event80386 : Event := .preFoldPolynomial 80385 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63429⟩⟩]⟩, (1)⟩] .exactZero none

def exact80387RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63429⟩⟩]⟩, (1)⟩]

def event80387 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨63430⟩⟩) 80386 exact80387RawTerms .large 80383 .exactZero (none)

def event80388 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨64509⟩⟩)

def event80389 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event80390 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event80391 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event80392 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event80393 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event80394 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event80395 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event80396 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event80397 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 80396

def event80398 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 80394

def event80399 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 80397 .coefficient) (.value (.predecessor 1 80398 .coefficient)))

def event80400 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event80401 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 80400

def event80402 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 80392

def event80403 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 80401 .coefficient, .predecessor 1 80402 .coefficient])

def event80404 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event80405 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 80404

def event80406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 80390

def event80407 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 80406 .coefficient))

def event80408 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event80409 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25562⟩⟩) 0 ⟨10325⟩ 80408

def event80410 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25562⟩⟩) (.authority (.programFamilyFact))

def exact80411RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩], []⟩, (1)⟩]

theorem exact80411RawTermsValid :
    exact80411RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80411 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25562⟩⟩) exact80411RawTerms (.finite 22) 80410 .exactZero (none)

def event80412 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62627⟩⟩) 0 ⟨10325⟩ 80408

def event80413 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62627⟩⟩) (.authority (.programFamilyFact))

def exact80414RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62627⟩⟩], []⟩, (1)⟩]

theorem exact80414RawTermsValid :
    exact80414RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80414 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62627⟩⟩) exact80414RawTerms (.finite 22) 80413 .exactZero (none)

def event80415 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62628⟩⟩) 0 ⟨62627⟩ 80414

def event80416 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62628⟩⟩) 1 ⟨25562⟩ 80411

def event80417 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62628⟩⟩) (.product (.predecessor 0 80415 .coefficient) (.predecessor 1 80416 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event80418 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62628⟩⟩, .operator (⟨80414, 0⟩, ⟨80411, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], []⟩, (1)⟩)

def exact80419RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], []⟩, (1)⟩]

theorem exact80419RawTermsValid :
    exact80419RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80419 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62628⟩⟩) exact80419RawTerms (.finite 484) 80417 .exactZero (none)

def event80420 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62629⟩⟩) 0 ⟨62628⟩ 80419

def event80421 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62629⟩⟩) (.identity (.predecessor 0 80420 .coefficient))

def event80422 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨62629⟩⟩) (.finite 484)

def event80423 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63964⟩⟩) 0 ⟨62629⟩ 80422

def event80424 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63964⟩⟩) (.authority (.programFamilyFact))

def event80425 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨63964⟩⟩) (.finite 3720)

def event80426 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event80427 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63965⟩⟩) 0 ⟨7177⟩ 80426

def event80428 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63965⟩⟩) 1 ⟨63964⟩ 80425

def event80429 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63965⟩⟩) (.authority (.operator))

def exact80430RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨63965⟩⟩]⟩, (1)⟩]

theorem exact80430RawTermsValid :
    exact80430RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80430 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63965⟩⟩) exact80430RawTerms .large 80429 .exactZero (none)

def event80431 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64505⟩⟩) 0 ⟨63965⟩ 80430

def event80432 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64505⟩⟩) (.authority (.operator))

def exact80433RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨64505⟩⟩]⟩, (1)⟩]

theorem exact80433RawTermsValid :
    exact80433RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80433 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64505⟩⟩) exact80433RawTerms (.finite 8192) 80432 .exactZero (none)

def event80434 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event80435 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event80436 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64230⟩⟩) 0 ⟨62629⟩ 80422

def event80437 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64230⟩⟩) 1 ⟨136⟩ 80435

def event80438 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64230⟩⟩) (.sum [.predecessor 0 80436 .coefficient, .predecessor 1 80437 .coefficient])

def event80439 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨64230⟩⟩) (.finite 484)

def event80440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64231⟩⟩) 0 ⟨64230⟩ 80439

def event80441 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64231⟩⟩) (.identity (.predecessor 0 80440 .coefficient))

def exact80442RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], []⟩, (1)⟩]

theorem exact80442RawTermsValid :
    exact80442RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80442 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64231⟩⟩) exact80442RawTerms (.finite 484) 80441 .exactZero (none)

def event80443 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact80444RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact80444RawTermsValid :
    exact80444RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80444 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact80444RawTerms .large 80443 .exactZero (none)

def event80445 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64232⟩⟩) 0 ⟨6908⟩ 80444

def event80446 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64232⟩⟩) 1 ⟨64231⟩ 80442

def event80447 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64232⟩⟩) (.product (.predecessor 0 80445 .coefficient) (.predecessor 1 80446 .coefficient) (⟨false, false, none, none, none⟩))

def event80448 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64232⟩⟩, .operator (⟨80444, 0⟩, ⟨80442, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact80449RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact80449RawTermsValid :
    exact80449RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80449 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64232⟩⟩) exact80449RawTerms .large 80447 .exactZero (none)

def event80450 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event80451 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event80452 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 80426

def event80453 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact80454RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact80454RawTermsValid :
    exact80454RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80454 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact80454RawTerms .large 80453 .exactZero (none)

def event80455 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7275⟩⟩) 0 ⟨7178⟩ 80454

def event80456 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7275⟩⟩) (.identity (.predecessor 0 80455 .coefficient))

def exact80457RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩]

theorem exact80457RawTermsValid :
    exact80457RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80457 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7275⟩⟩) exact80457RawTerms .large 80456 .exactZero (none)

def event80458 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9538⟩⟩) 0 ⟨7275⟩ 80457

def event80459 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9538⟩⟩) (.authority (.operator))

def exact80460RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (1)⟩]

theorem exact80460RawTermsValid :
    exact80460RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80460 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9538⟩⟩) exact80460RawTerms (.finite 8192) 80459 .exactZero (none)

def event80461 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9539⟩⟩) 0 ⟨9538⟩ 80460

def event80462 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9539⟩⟩) 1 ⟨2370⟩ 80451

def event80463 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9539⟩⟩) (.scale (.predecessor 0 80461 .coefficient) (.value (.predecessor 1 80462 .coefficient)))

def exact80464RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (1)⟩]

theorem exact80464RawTermsValid :
    exact80464RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80464 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9539⟩⟩) exact80464RawTerms (.finite 8192) 80463 .exactZero (none)

def event80465 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7293⟩⟩) 0 ⟨7178⟩ 80454

def event80466 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7293⟩⟩) (.identity (.predecessor 0 80465 .coefficient))

def exact80467RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩]⟩, (1)⟩]

theorem exact80467RawTermsValid :
    exact80467RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80467 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7293⟩⟩) exact80467RawTerms .large 80466 .exactZero (none)

def event80468 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9540⟩⟩) 0 ⟨7293⟩ 80467

def event80469 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9540⟩⟩) 1 ⟨9539⟩ 80464

def event80470 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9540⟩⟩) (.product (.predecessor 0 80468 .coefficient) (.predecessor 1 80469 .coefficient) (⟨false, false, none, none, none⟩))

def event80471 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9540⟩⟩, .operator (⟨80467, 0⟩, ⟨80464, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (1)⟩)

def exact80472RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (1)⟩]

theorem exact80472RawTermsValid :
    exact80472RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80472 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9540⟩⟩) exact80472RawTerms .large 80470 .exactZero (none)

def event80473 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64233⟩⟩) 0 ⟨9540⟩ 80472

def event80474 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64233⟩⟩) 1 ⟨64232⟩ 80449

def event80475 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64233⟩⟩) (.sum [.predecessor 0 80473 .coefficient, .predecessor 1 80474 .coefficient])

def exact80476RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80476RawTermsValid :
    exact80476RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80476 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64233⟩⟩) exact80476RawTerms .large 80475 .exactZero (none)

def event80477 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64508⟩⟩) 0 ⟨64233⟩ 80476

def event80478 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64508⟩⟩) 1 ⟨64505⟩ 80433

def event80479 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64508⟩⟩) (.product (.predecessor 0 80477 .coefficient) (.predecessor 1 80478 .coefficient) (⟨false, false, none, none, none⟩))

def event80480 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64508⟩⟩, .operator (⟨80476, 0⟩, ⟨80433, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64505⟩⟩]⟩, (1)⟩)

def event80481 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64508⟩⟩, .operator (⟨80476, 1⟩, ⟨80433, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64505⟩⟩]⟩, (-1)⟩)

def event80482 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨64508⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64505⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64505⟩⟩) ⟨63965⟩ 80430)

def event80483 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64508⟩⟩, .relation 80482 0, ⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨63965⟩⟩]⟩, (-1)⟩)

def exact80484RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64505⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨63965⟩⟩]⟩, (-1)⟩]

theorem exact80484RawTermsValid :
    exact80484RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80484 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64508⟩⟩) exact80484RawTerms .large 80479 .exactZero (none)

def event80485 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62856⟩⟩) 0 ⟨62629⟩ 80422

def event80486 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62856⟩⟩) (.authority (.programFamilyFact))

def exact80487RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], []⟩, (1)⟩]

theorem exact80487RawTermsValid :
    exact80487RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80487 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62856⟩⟩) exact80487RawTerms (.finite 22) 80486 .exactZero (none)

def event80488 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62858⟩⟩) 0 ⟨6908⟩ 80444

def event80489 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62858⟩⟩) 1 ⟨62856⟩ 80487

def event80490 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62858⟩⟩) (.product (.predecessor 0 80488 .coefficient) (.predecessor 1 80489 .coefficient) (⟨false, true, none, none, some 1⟩))

def event80491 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62858⟩⟩, .operator (⟨80444, 0⟩, ⟨80487, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact80492RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact80492RawTermsValid :
    exact80492RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80492 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62858⟩⟩) exact80492RawTerms .large 80490 .exactZero (none)

def event80493 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7187⟩⟩) 0 ⟨7177⟩ 80426

def event80494 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7187⟩⟩) (.authority (.operator))

def exact80495RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩, (1)⟩]

theorem exact80495RawTermsValid :
    exact80495RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80495 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7187⟩⟩) exact80495RawTerms .large 80494 .exactZero (none)

def event80496 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62859⟩⟩) 0 ⟨7187⟩ 80495

def event80497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62859⟩⟩) 1 ⟨62858⟩ 80492

def event80498 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62859⟩⟩) (.sum [.predecessor 0 80496 .coefficient, .predecessor 1 80497 .coefficient])

def exact80499RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80499RawTermsValid :
    exact80499RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80499 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62859⟩⟩) exact80499RawTerms .large 80498 .exactZero (none)

def event80500 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64509⟩⟩) 0 ⟨62859⟩ 80499

def event80501 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64509⟩⟩) 1 ⟨64508⟩ 80484

def event80502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64509⟩⟩) (.sum [.predecessor 0 80500 .coefficient, .predecessor 1 80501 .coefficient])

def exact80503RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64505⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨63965⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80503RawTermsValid :
    exact80503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80503 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64509⟩⟩) exact80503RawTerms .large 80502 .exactZero (none)

def event80504 : Event := .preFoldPolynomial 80503 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64505⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨63965⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact80505RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64505⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨63965⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event80505 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨64509⟩⟩) 80504 exact80505RawTerms .large 80502 .exactZero (none)

def event80506 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨62629⟩⟩) ⟨⟨66⟩, ⟨45⟩, ⟨135⟩⟩ ⟨80340, 80506⟩

def event80507 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨63432⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63429⟩⟩]⟩) (1) 0 2 (.universal 80506 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63429⟩⟩]⟩) (none) 80505)

def event80508 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63432⟩⟩, .relation 80507 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩, (1)⟩)

def event80509 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63432⟩⟩, .relation 80507 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64505⟩⟩]⟩, (-1)⟩)

def event80510 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63432⟩⟩, .relation 80507 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨63965⟩⟩]⟩, (1)⟩)

def event80511 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63432⟩⟩, .relation 80507 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact80512RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64505⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨63965⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80512RawTermsValid :
    exact80512RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80512 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63432⟩⟩) exact80512RawTerms .large 80336 (.finite 202072841853861888) (some (80338))

def event80513 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64507⟩⟩) 0 ⟨63432⟩ 80512

def event80514 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64507⟩⟩) 1 ⟨64506⟩ 80326

def event80515 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64507⟩⟩) (.sum [.predecessor 0 80513 .coefficient, .predecessor 1 80514 .coefficient])

def event80516 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64507⟩⟩, .operator (⟨80512, 2⟩, ⟨80326, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], [⟨.program ⟨257⟩, ⟨63965⟩⟩]⟩, (-1)⟩)

def event80517 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64507⟩⟩, .operator (⟨80512, 1⟩, ⟨80326, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64505⟩⟩]⟩, (1)⟩)

def event80518 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64507⟩⟩) (.sum [.result 80512 .summary, .result 80326 .summary])

def exact80519RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80519RawTermsValid :
    exact80519RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80519 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64507⟩⟩) exact80519RawTerms .large 80515 (.finite 2997999239428004118528) (some (80518))

def event80520 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65060⟩⟩) 0 ⟨64507⟩ 80519

def event80521 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65060⟩⟩) 1 ⟨65058⟩ 80242

def event80522 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65060⟩⟩) (.product (.predecessor 0 80520 .coefficient) (.predecessor 1 80521 .coefficient) (⟨false, false, none, none, none⟩))

def event80523 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65060⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨65058⟩⟩]⟩) [⟨.result 80242 .coefficient, false, none⟩])

def event80524 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65060⟩⟩) (.product (.result 80519 .summary) (.transfer 80523) (⟨false, false, none, none, none⟩))

def event80525 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65060⟩⟩, .operator (⟨80519, 0⟩, ⟨80242, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65058⟩⟩]⟩, (1)⟩)

def event80526 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65060⟩⟩, .operator (⟨80519, 1⟩, ⟨80242, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65058⟩⟩]⟩, (-1)⟩)

def event80527 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨65060⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65058⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨65058⟩⟩) ⟨64135⟩ 80239)

def event80528 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65060⟩⟩, .relation 80527 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨64135⟩⟩]⟩, (-1)⟩)

def exact80529RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65058⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨64135⟩⟩]⟩, (-1)⟩]

theorem exact80529RawTermsValid :
    exact80529RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80529 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65060⟩⟩) exact80529RawTerms .large 80522 (.finite 32190771716940378589077669150720) (some (80524))

def event80530 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63796⟩⟩) 0 ⟨62857⟩ 3310

def event80531 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63796⟩⟩) (.authority (.relationPreimageSource ⟨74⟩))

def exact80532RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨63796⟩⟩]⟩, (1)⟩]

theorem exact80532RawTermsValid :
    exact80532RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80532 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63796⟩⟩) exact80532RawTerms (.finite 5647228698) 80531 .exactZero (none)

def event80533 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63798⟩⟩) 0 ⟨63796⟩ 80532

def event80534 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63798⟩⟩) 1 ⟨2370⟩ 4

def event80535 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63798⟩⟩) (.scale (.predecessor 0 80533 .coefficient) (.value (.predecessor 1 80534 .coefficient)))

def exact80536RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨63796⟩⟩]⟩, (1)⟩]

theorem exact80536RawTermsValid :
    exact80536RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80536 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63798⟩⟩) exact80536RawTerms (.finite 5647228698) 80535 .exactZero (none)

def event80537 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63799⟩⟩) 0 ⟨10368⟩ 75995

def event80538 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63799⟩⟩) 1 ⟨63798⟩ 80536

def event80539 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63799⟩⟩) (.product (.predecessor 0 80537 .coefficient) (.predecessor 1 80538 .coefficient) (⟨false, false, none, none, none⟩))

def event80540 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63799⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨63796⟩⟩]⟩) [⟨.result 80532 .coefficient, false, none⟩])

def event80541 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63799⟩⟩) (.product (.result 75995 .summary) (.transfer 80540) (⟨false, false, none, none, none⟩))

def event80542 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63799⟩⟩, .operator (⟨75995, 0⟩, ⟨80536, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63796⟩⟩]⟩, (1)⟩)

def event80543 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨63797⟩⟩)

def event80544 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event80545 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event80546 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event80547 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event80548 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event80549 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event80550 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event80551 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event80552 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 80551

def event80553 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 80549

def event80554 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 80552 .coefficient) (.value (.predecessor 1 80553 .coefficient)))

def event80555 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event80556 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 80555

def event80557 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 80547

def event80558 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 80556 .coefficient, .predecessor 1 80557 .coefficient])

def event80559 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event80560 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 80559

def event80561 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 80545

def event80562 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 80561 .coefficient))

def event80563 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event80564 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25562⟩⟩) 0 ⟨10325⟩ 80563

def event80565 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25562⟩⟩) (.authority (.programFamilyFact))

def exact80566RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩], []⟩, (1)⟩]

theorem exact80566RawTermsValid :
    exact80566RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80566 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25562⟩⟩) exact80566RawTerms (.finite 22) 80565 .exactZero (none)

def event80567 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62627⟩⟩) 0 ⟨10325⟩ 80563

def event80568 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62627⟩⟩) (.authority (.programFamilyFact))

def exact80569RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62627⟩⟩], []⟩, (1)⟩]

theorem exact80569RawTermsValid :
    exact80569RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80569 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62627⟩⟩) exact80569RawTerms (.finite 22) 80568 .exactZero (none)

def event80570 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62628⟩⟩) 0 ⟨62627⟩ 80569

def event80571 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62628⟩⟩) 1 ⟨25562⟩ 80566

def event80572 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62628⟩⟩) (.product (.predecessor 0 80570 .coefficient) (.predecessor 1 80571 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event80573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62628⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], []⟩) [⟨.result 80569 .coefficient, true, some 1⟩, ⟨.result 80566 .coefficient, true, some 1⟩])

def event80574 : Event := .survivorFold (1) 80573

def exact80575RawTerms : List Term := []

theorem exact80575RawTermsValid :
    exact80575RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80575 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62628⟩⟩) exact80575RawTerms (.finite 484) 80572 (.finite 484) (some (80573))

def event80576 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62629⟩⟩) 0 ⟨62628⟩ 80575

def event80577 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62629⟩⟩) (.identity (.predecessor 0 80576 .coefficient))

def event80578 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨62629⟩⟩) (.finite 484)

def event80579 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62856⟩⟩) 0 ⟨62629⟩ 80578

def event80580 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62856⟩⟩) (.authority (.programFamilyFact))

def exact80581RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], []⟩, (1)⟩]

theorem exact80581RawTermsValid :
    exact80581RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80581 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62856⟩⟩) exact80581RawTerms (.finite 22) 80580 .exactZero (none)

def event80582 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62857⟩⟩) 0 ⟨62856⟩ 80581

def event80583 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62857⟩⟩) (.identity (.predecessor 0 80582 .coefficient))

def event80584 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨62857⟩⟩) (.finite 22)

def event80585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63796⟩⟩) 0 ⟨62857⟩ 80584

def event80586 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63796⟩⟩) (.authority (.relationPreimageSource ⟨74⟩))

def exact80587RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨63796⟩⟩]⟩, (1)⟩]

theorem exact80587RawTermsValid :
    exact80587RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80587 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63796⟩⟩) exact80587RawTerms (.finite 5647228698) 80586 .exactZero (none)

def event80588 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact80589RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact80589RawTermsValid :
    exact80589RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80589 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact80589RawTerms .large 80588 .exactZero (none)

def event80590 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63797⟩⟩) 0 ⟨35⟩ 80589

def event80591 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63797⟩⟩) 1 ⟨63796⟩ 80587

def event80592 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63797⟩⟩) (.product (.predecessor 0 80590 .coefficient) (.predecessor 1 80591 .coefficient) (⟨false, false, none, none, none⟩))

def event80593 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63797⟩⟩, .operator (⟨80589, 0⟩, ⟨80587, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63796⟩⟩]⟩, (1)⟩)

def exact80594RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63796⟩⟩]⟩, (1)⟩]

theorem exact80594RawTermsValid :
    exact80594RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80594 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63797⟩⟩) exact80594RawTerms .large 80592 .exactZero (none)

def event80595 : Event := .preFoldPolynomial 80594 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63796⟩⟩]⟩, (1)⟩] .exactZero none

def exact80596RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63796⟩⟩]⟩, (1)⟩]

def event80596 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨63797⟩⟩) 80595 exact80596RawTerms .large 80592 .exactZero (none)

def event80597 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨65063⟩⟩)

def event80598 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event80599 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event80600 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event80601 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event80602 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event80603 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event80604 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event80605 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event80606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 80605

def event80607 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 80603

def event80608 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 80606 .coefficient) (.value (.predecessor 1 80607 .coefficient)))

def event80609 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event80610 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 80609

def event80611 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 80601

def event80612 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 80610 .coefficient, .predecessor 1 80611 .coefficient])

def event80613 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event80614 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 80613

def event80615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 80599

def event80616 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 80615 .coefficient))

def event80617 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event80618 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25562⟩⟩) 0 ⟨10325⟩ 80617

def event80619 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25562⟩⟩) (.authority (.programFamilyFact))

def exact80620RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩], []⟩, (1)⟩]

theorem exact80620RawTermsValid :
    exact80620RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80620 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25562⟩⟩) exact80620RawTerms (.finite 22) 80619 .exactZero (none)

def event80621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62627⟩⟩) 0 ⟨10325⟩ 80617

def event80622 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62627⟩⟩) (.authority (.programFamilyFact))

def exact80623RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62627⟩⟩], []⟩, (1)⟩]

theorem exact80623RawTermsValid :
    exact80623RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80623 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62627⟩⟩) exact80623RawTerms (.finite 22) 80622 .exactZero (none)

def event80624 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62628⟩⟩) 0 ⟨62627⟩ 80623

def event80625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62628⟩⟩) 1 ⟨25562⟩ 80620

def event80626 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62628⟩⟩) (.product (.predecessor 0 80624 .coefficient) (.predecessor 1 80625 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event80627 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62628⟩⟩, .operator (⟨80623, 0⟩, ⟨80620, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], []⟩, (1)⟩)

def exact80628RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25562⟩⟩, ⟨.program ⟨257⟩, ⟨62627⟩⟩], []⟩, (1)⟩]

theorem exact80628RawTermsValid :
    exact80628RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80628 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62628⟩⟩) exact80628RawTerms (.finite 484) 80626 .exactZero (none)

def event80629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62629⟩⟩) 0 ⟨62628⟩ 80628

def event80630 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62629⟩⟩) (.identity (.predecessor 0 80629 .coefficient))

def event80631 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨62629⟩⟩) (.finite 484)

def event80632 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62856⟩⟩) 0 ⟨62629⟩ 80631

def event80633 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62856⟩⟩) (.authority (.programFamilyFact))

def exact80634RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], []⟩, (1)⟩]

theorem exact80634RawTermsValid :
    exact80634RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80634 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62856⟩⟩) exact80634RawTerms (.finite 22) 80633 .exactZero (none)

def event80635 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62857⟩⟩) 0 ⟨62856⟩ 80634

def event80636 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62857⟩⟩) (.identity (.predecessor 0 80635 .coefficient))

def event80637 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨62857⟩⟩) (.finite 22)

def event80638 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64133⟩⟩) 0 ⟨62857⟩ 80637

def event80639 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64133⟩⟩) (.authority (.programFamilyFact))

def eventLeaf5024 : Array AnnotatedEvent := #[
  { event := event80384
    frameStart := 80340 },
  { event := event80385
    frameStart := 80340 },
  { event := event80386
    frameStart := 80340 },
  { event := event80387
    frameStart := 80340 },
  { event := event80388
    frameStart := 80388 },
  { event := event80389
    frameStart := 80388 },
  { event := event80390
    frameStart := 80388 },
  { event := event80391
    frameStart := 80388 },
  { event := event80392
    frameStart := 80388 },
  { event := event80393
    frameStart := 80388 },
  { event := event80394
    frameStart := 80388 },
  { event := event80395
    frameStart := 80388 },
  { event := event80396
    frameStart := 80388 },
  { event := event80397
    frameStart := 80388 },
  { event := event80398
    frameStart := 80388 },
  { event := event80399
    frameStart := 80388 }
]

def eventLeaf5025 : Array AnnotatedEvent := #[
  { event := event80400
    frameStart := 80388 },
  { event := event80401
    frameStart := 80388 },
  { event := event80402
    frameStart := 80388 },
  { event := event80403
    frameStart := 80388 },
  { event := event80404
    frameStart := 80388 },
  { event := event80405
    frameStart := 80388 },
  { event := event80406
    frameStart := 80388 },
  { event := event80407
    frameStart := 80388 },
  { event := event80408
    frameStart := 80388 },
  { event := event80409
    frameStart := 80388 },
  { event := event80410
    frameStart := 80388 },
  { event := event80411
    frameStart := 80388 },
  { event := event80412
    frameStart := 80388 },
  { event := event80413
    frameStart := 80388 },
  { event := event80414
    frameStart := 80388 },
  { event := event80415
    frameStart := 80388 }
]

def eventLeaf5026 : Array AnnotatedEvent := #[
  { event := event80416
    frameStart := 80388 },
  { event := event80417
    frameStart := 80388 },
  { event := event80418
    frameStart := 80388 },
  { event := event80419
    frameStart := 80388 },
  { event := event80420
    frameStart := 80388 },
  { event := event80421
    frameStart := 80388 },
  { event := event80422
    frameStart := 80388 },
  { event := event80423
    frameStart := 80388 },
  { event := event80424
    frameStart := 80388 },
  { event := event80425
    frameStart := 80388 },
  { event := event80426
    frameStart := 80388 },
  { event := event80427
    frameStart := 80388 },
  { event := event80428
    frameStart := 80388 },
  { event := event80429
    frameStart := 80388 },
  { event := event80430
    frameStart := 80388 },
  { event := event80431
    frameStart := 80388 }
]

def eventLeaf5027 : Array AnnotatedEvent := #[
  { event := event80432
    frameStart := 80388 },
  { event := event80433
    frameStart := 80388 },
  { event := event80434
    frameStart := 80388 },
  { event := event80435
    frameStart := 80388 },
  { event := event80436
    frameStart := 80388 },
  { event := event80437
    frameStart := 80388 },
  { event := event80438
    frameStart := 80388 },
  { event := event80439
    frameStart := 80388 },
  { event := event80440
    frameStart := 80388 },
  { event := event80441
    frameStart := 80388 },
  { event := event80442
    frameStart := 80388 },
  { event := event80443
    frameStart := 80388 },
  { event := event80444
    frameStart := 80388 },
  { event := event80445
    frameStart := 80388 },
  { event := event80446
    frameStart := 80388 },
  { event := event80447
    frameStart := 80388 }
]

def eventLeaf5028 : Array AnnotatedEvent := #[
  { event := event80448
    frameStart := 80388 },
  { event := event80449
    frameStart := 80388 },
  { event := event80450
    frameStart := 80388 },
  { event := event80451
    frameStart := 80388 },
  { event := event80452
    frameStart := 80388 },
  { event := event80453
    frameStart := 80388 },
  { event := event80454
    frameStart := 80388 },
  { event := event80455
    frameStart := 80388 },
  { event := event80456
    frameStart := 80388 },
  { event := event80457
    frameStart := 80388 },
  { event := event80458
    frameStart := 80388 },
  { event := event80459
    frameStart := 80388 },
  { event := event80460
    frameStart := 80388 },
  { event := event80461
    frameStart := 80388 },
  { event := event80462
    frameStart := 80388 },
  { event := event80463
    frameStart := 80388 }
]

def eventLeaf5029 : Array AnnotatedEvent := #[
  { event := event80464
    frameStart := 80388 },
  { event := event80465
    frameStart := 80388 },
  { event := event80466
    frameStart := 80388 },
  { event := event80467
    frameStart := 80388 },
  { event := event80468
    frameStart := 80388 },
  { event := event80469
    frameStart := 80388 },
  { event := event80470
    frameStart := 80388 },
  { event := event80471
    frameStart := 80388 },
  { event := event80472
    frameStart := 80388 },
  { event := event80473
    frameStart := 80388 },
  { event := event80474
    frameStart := 80388 },
  { event := event80475
    frameStart := 80388 },
  { event := event80476
    frameStart := 80388 },
  { event := event80477
    frameStart := 80388 },
  { event := event80478
    frameStart := 80388 },
  { event := event80479
    frameStart := 80388 }
]

def eventLeaf5030 : Array AnnotatedEvent := #[
  { event := event80480
    frameStart := 80388 },
  { event := event80481
    frameStart := 80388 },
  { event := event80482
    frameStart := 80388 },
  { event := event80483
    frameStart := 80388 },
  { event := event80484
    frameStart := 80388 },
  { event := event80485
    frameStart := 80388 },
  { event := event80486
    frameStart := 80388 },
  { event := event80487
    frameStart := 80388 },
  { event := event80488
    frameStart := 80388 },
  { event := event80489
    frameStart := 80388 },
  { event := event80490
    frameStart := 80388 },
  { event := event80491
    frameStart := 80388 },
  { event := event80492
    frameStart := 80388 },
  { event := event80493
    frameStart := 80388 },
  { event := event80494
    frameStart := 80388 },
  { event := event80495
    frameStart := 80388 }
]

def eventLeaf5031 : Array AnnotatedEvent := #[
  { event := event80496
    frameStart := 80388 },
  { event := event80497
    frameStart := 80388 },
  { event := event80498
    frameStart := 80388 },
  { event := event80499
    frameStart := 80388 },
  { event := event80500
    frameStart := 80388 },
  { event := event80501
    frameStart := 80388 },
  { event := event80502
    frameStart := 80388 },
  { event := event80503
    frameStart := 80388 },
  { event := event80504
    frameStart := 80388 },
  { event := event80505
    frameStart := 80388 },
  { event := event80506
    frameStart := 0 },
  { event := event80507
    frameStart := 0 },
  { event := event80508
    frameStart := 0 },
  { event := event80509
    frameStart := 0 },
  { event := event80510
    frameStart := 0 },
  { event := event80511
    frameStart := 0 }
]

def eventLeaf5032 : Array AnnotatedEvent := #[
  { event := event80512
    frameStart := 0 },
  { event := event80513
    frameStart := 0 },
  { event := event80514
    frameStart := 0 },
  { event := event80515
    frameStart := 0 },
  { event := event80516
    frameStart := 0 },
  { event := event80517
    frameStart := 0 },
  { event := event80518
    frameStart := 0 },
  { event := event80519
    frameStart := 0 },
  { event := event80520
    frameStart := 0 },
  { event := event80521
    frameStart := 0 },
  { event := event80522
    frameStart := 0 },
  { event := event80523
    frameStart := 0 },
  { event := event80524
    frameStart := 0 },
  { event := event80525
    frameStart := 0 },
  { event := event80526
    frameStart := 0 },
  { event := event80527
    frameStart := 0 }
]

def eventLeaf5033 : Array AnnotatedEvent := #[
  { event := event80528
    frameStart := 0 },
  { event := event80529
    frameStart := 0 },
  { event := event80530
    frameStart := 0 },
  { event := event80531
    frameStart := 0 },
  { event := event80532
    frameStart := 0 },
  { event := event80533
    frameStart := 0 },
  { event := event80534
    frameStart := 0 },
  { event := event80535
    frameStart := 0 },
  { event := event80536
    frameStart := 0 },
  { event := event80537
    frameStart := 0 },
  { event := event80538
    frameStart := 0 },
  { event := event80539
    frameStart := 0 },
  { event := event80540
    frameStart := 0 },
  { event := event80541
    frameStart := 0 },
  { event := event80542
    frameStart := 0 },
  { event := event80543
    frameStart := 80543 }
]

def eventLeaf5034 : Array AnnotatedEvent := #[
  { event := event80544
    frameStart := 80543 },
  { event := event80545
    frameStart := 80543 },
  { event := event80546
    frameStart := 80543 },
  { event := event80547
    frameStart := 80543 },
  { event := event80548
    frameStart := 80543 },
  { event := event80549
    frameStart := 80543 },
  { event := event80550
    frameStart := 80543 },
  { event := event80551
    frameStart := 80543 },
  { event := event80552
    frameStart := 80543 },
  { event := event80553
    frameStart := 80543 },
  { event := event80554
    frameStart := 80543 },
  { event := event80555
    frameStart := 80543 },
  { event := event80556
    frameStart := 80543 },
  { event := event80557
    frameStart := 80543 },
  { event := event80558
    frameStart := 80543 },
  { event := event80559
    frameStart := 80543 }
]

def eventLeaf5035 : Array AnnotatedEvent := #[
  { event := event80560
    frameStart := 80543 },
  { event := event80561
    frameStart := 80543 },
  { event := event80562
    frameStart := 80543 },
  { event := event80563
    frameStart := 80543 },
  { event := event80564
    frameStart := 80543 },
  { event := event80565
    frameStart := 80543 },
  { event := event80566
    frameStart := 80543 },
  { event := event80567
    frameStart := 80543 },
  { event := event80568
    frameStart := 80543 },
  { event := event80569
    frameStart := 80543 },
  { event := event80570
    frameStart := 80543 },
  { event := event80571
    frameStart := 80543 },
  { event := event80572
    frameStart := 80543 },
  { event := event80573
    frameStart := 80543 },
  { event := event80574
    frameStart := 80543 },
  { event := event80575
    frameStart := 80543 }
]

def eventLeaf5036 : Array AnnotatedEvent := #[
  { event := event80576
    frameStart := 80543 },
  { event := event80577
    frameStart := 80543 },
  { event := event80578
    frameStart := 80543 },
  { event := event80579
    frameStart := 80543 },
  { event := event80580
    frameStart := 80543 },
  { event := event80581
    frameStart := 80543 },
  { event := event80582
    frameStart := 80543 },
  { event := event80583
    frameStart := 80543 },
  { event := event80584
    frameStart := 80543 },
  { event := event80585
    frameStart := 80543 },
  { event := event80586
    frameStart := 80543 },
  { event := event80587
    frameStart := 80543 },
  { event := event80588
    frameStart := 80543 },
  { event := event80589
    frameStart := 80543 },
  { event := event80590
    frameStart := 80543 },
  { event := event80591
    frameStart := 80543 }
]

def eventLeaf5037 : Array AnnotatedEvent := #[
  { event := event80592
    frameStart := 80543 },
  { event := event80593
    frameStart := 80543 },
  { event := event80594
    frameStart := 80543 },
  { event := event80595
    frameStart := 80543 },
  { event := event80596
    frameStart := 80543 },
  { event := event80597
    frameStart := 80597 },
  { event := event80598
    frameStart := 80597 },
  { event := event80599
    frameStart := 80597 },
  { event := event80600
    frameStart := 80597 },
  { event := event80601
    frameStart := 80597 },
  { event := event80602
    frameStart := 80597 },
  { event := event80603
    frameStart := 80597 },
  { event := event80604
    frameStart := 80597 },
  { event := event80605
    frameStart := 80597 },
  { event := event80606
    frameStart := 80597 },
  { event := event80607
    frameStart := 80597 }
]

def eventLeaf5038 : Array AnnotatedEvent := #[
  { event := event80608
    frameStart := 80597 },
  { event := event80609
    frameStart := 80597 },
  { event := event80610
    frameStart := 80597 },
  { event := event80611
    frameStart := 80597 },
  { event := event80612
    frameStart := 80597 },
  { event := event80613
    frameStart := 80597 },
  { event := event80614
    frameStart := 80597 },
  { event := event80615
    frameStart := 80597 },
  { event := event80616
    frameStart := 80597 },
  { event := event80617
    frameStart := 80597 },
  { event := event80618
    frameStart := 80597 },
  { event := event80619
    frameStart := 80597 },
  { event := event80620
    frameStart := 80597 },
  { event := event80621
    frameStart := 80597 },
  { event := event80622
    frameStart := 80597 },
  { event := event80623
    frameStart := 80597 }
]

def eventLeaf5039 : Array AnnotatedEvent := #[
  { event := event80624
    frameStart := 80597 },
  { event := event80625
    frameStart := 80597 },
  { event := event80626
    frameStart := 80597 },
  { event := event80627
    frameStart := 80597 },
  { event := event80628
    frameStart := 80597 },
  { event := event80629
    frameStart := 80597 },
  { event := event80630
    frameStart := 80597 },
  { event := event80631
    frameStart := 80597 },
  { event := event80632
    frameStart := 80597 },
  { event := event80633
    frameStart := 80597 },
  { event := event80634
    frameStart := 80597 },
  { event := event80635
    frameStart := 80597 },
  { event := event80636
    frameStart := 80597 },
  { event := event80637
    frameStart := 80597 },
  { event := event80638
    frameStart := 80597 },
  { event := event80639
    frameStart := 80597 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events314
