import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events068

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event17408 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47627⟩⟩) (.product (.predecessor 0 17406 .coefficient) (.predecessor 1 17407 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event17409 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47627⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], []⟩) [⟨.result 17405 .coefficient, true, some 1⟩, ⟨.result 17402 .coefficient, true, some 1⟩])

def event17410 : Event := .survivorFold (1) 17409

def exact17411RawTerms : List Term := []

theorem exact17411RawTermsValid :
    exact17411RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17411 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47627⟩⟩) exact17411RawTerms (.finite 3600) 17408 (.finite 3600) (some (17409))

def event17412 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47628⟩⟩) 0 ⟨47627⟩ 17411

def event17413 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47628⟩⟩) (.identity (.predecessor 0 17412 .coefficient))

def event17414 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47628⟩⟩) (.finite 3600)

def event17415 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48078⟩⟩) 0 ⟨47628⟩ 17414

def event17416 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48078⟩⟩) (.authority (.programFamilyFact))

def exact17417RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], []⟩, (1)⟩]

theorem exact17417RawTermsValid :
    exact17417RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17417 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48078⟩⟩) exact17417RawTerms (.finite 60) 17416 .exactZero (none)

def event17418 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48079⟩⟩) 0 ⟨48078⟩ 17417

def event17419 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48079⟩⟩) (.identity (.predecessor 0 17418 .coefficient))

def event17420 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48079⟩⟩) (.finite 60)

def event17421 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48722⟩⟩) 0 ⟨48079⟩ 17420

def event17422 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48722⟩⟩) (.authority (.relationPreimageSource ⟨94⟩))

def exact17423RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48722⟩⟩]⟩, (1)⟩]

theorem exact17423RawTermsValid :
    exact17423RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17423 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48722⟩⟩) exact17423RawTerms (.finite 5647228698) 17422 .exactZero (none)

def event17424 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact17425RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact17425RawTermsValid :
    exact17425RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17425 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact17425RawTerms .large 17424 .exactZero (none)

def event17426 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48723⟩⟩) 0 ⟨35⟩ 17425

def event17427 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48723⟩⟩) 1 ⟨48722⟩ 17423

def event17428 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48723⟩⟩) (.product (.predecessor 0 17426 .coefficient) (.predecessor 1 17427 .coefficient) (⟨false, false, none, none, none⟩))

def event17429 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48723⟩⟩, .operator (⟨17425, 0⟩, ⟨17423, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48722⟩⟩]⟩, (1)⟩)

def exact17430RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48722⟩⟩]⟩, (1)⟩]

theorem exact17430RawTermsValid :
    exact17430RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17430 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48723⟩⟩) exact17430RawTerms .large 17428 .exactZero (none)

def event17431 : Event := .preFoldPolynomial 17430 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48722⟩⟩]⟩, (1)⟩] .exactZero none

def exact17432RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48722⟩⟩]⟩, (1)⟩]

def event17432 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨48723⟩⟩) 17431 exact17432RawTerms .large 17428 .exactZero (none)

def event17433 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨49815⟩⟩)

def event17434 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event17435 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event17436 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨140⟩⟩) (.authority (.operator))

def event17437 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨140⟩⟩) (.finite 19)

def event17438 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event17439 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event17440 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event17441 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event17442 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 17441

def event17443 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 17439

def event17444 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 17442 .coefficient) (.value (.predecessor 1 17443 .coefficient)))

def event17445 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event17446 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 0 ⟨392⟩ 17445

def event17447 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 1 ⟨140⟩ 17437

def event17448 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨393⟩⟩) (.sum [.predecessor 0 17446 .coefficient, .predecessor 1 17447 .coefficient])

def event17449 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨393⟩⟩) (.finite 655359)

def event17450 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 0 ⟨393⟩ 17449

def event17451 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 1 ⟨5426⟩ 17435

def event17452 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.identity (.predecessor 1 17451 .coefficient))

def event17453 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.finite 655360)

def event17454 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47626⟩⟩) 0 ⟨5439⟩ 17453

def event17455 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47626⟩⟩) (.authority (.programFamilyFact))

def exact17456RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47626⟩⟩], []⟩, (1)⟩]

theorem exact17456RawTermsValid :
    exact17456RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17456 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47626⟩⟩) exact17456RawTerms (.finite 60) 17455 .exactZero (none)

def event17457 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14951⟩⟩) 0 ⟨5439⟩ 17453

def event17458 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14951⟩⟩) (.authority (.programFamilyFact))

def exact17459RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩], []⟩, (1)⟩]

theorem exact17459RawTermsValid :
    exact17459RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17459 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14951⟩⟩) exact17459RawTerms (.finite 60) 17458 .exactZero (none)

def event17460 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47627⟩⟩) 0 ⟨14951⟩ 17459

def event17461 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47627⟩⟩) 1 ⟨47626⟩ 17456

def event17462 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47627⟩⟩) (.product (.predecessor 0 17460 .coefficient) (.predecessor 1 17461 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event17463 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47627⟩⟩, .operator (⟨17459, 0⟩, ⟨17456, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], []⟩, (1)⟩)

def exact17464RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], []⟩, (1)⟩]

theorem exact17464RawTermsValid :
    exact17464RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17464 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47627⟩⟩) exact17464RawTerms (.finite 3600) 17462 .exactZero (none)

def event17465 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47628⟩⟩) 0 ⟨47627⟩ 17464

def event17466 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47628⟩⟩) (.identity (.predecessor 0 17465 .coefficient))

def event17467 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47628⟩⟩) (.finite 3600)

def event17468 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48078⟩⟩) 0 ⟨47628⟩ 17467

def event17469 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48078⟩⟩) (.authority (.programFamilyFact))

def exact17470RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], []⟩, (1)⟩]

theorem exact17470RawTermsValid :
    exact17470RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17470 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48078⟩⟩) exact17470RawTerms (.finite 60) 17469 .exactZero (none)

def event17471 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48079⟩⟩) 0 ⟨48078⟩ 17470

def event17472 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48079⟩⟩) (.identity (.predecessor 0 17471 .coefficient))

def event17473 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48079⟩⟩) (.finite 60)

def event17474 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49221⟩⟩) 0 ⟨48079⟩ 17473

def event17475 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49221⟩⟩) (.authority (.programFamilyFact))

def event17476 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49221⟩⟩) (.finite 3720)

def event17477 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event17478 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49223⟩⟩) 0 ⟨7177⟩ 17477

def event17479 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49223⟩⟩) 1 ⟨49221⟩ 17476

def event17480 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49223⟩⟩) (.authority (.operator))

def exact17481RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49223⟩⟩]⟩, (1)⟩]

theorem exact17481RawTermsValid :
    exact17481RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17481 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49223⟩⟩) exact17481RawTerms .large 17480 .exactZero (none)

def event17482 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49811⟩⟩) 0 ⟨49223⟩ 17481

def event17483 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49811⟩⟩) (.authority (.operator))

def exact17484RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩, (1)⟩]

theorem exact17484RawTermsValid :
    exact17484RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17484 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49811⟩⟩) exact17484RawTerms (.finite 8192) 17483 .exactZero (none)

def event17485 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event17486 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event17487 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49470⟩⟩) 0 ⟨48079⟩ 17473

def event17488 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49470⟩⟩) 1 ⟨136⟩ 17486

def event17489 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49470⟩⟩) (.sum [.predecessor 0 17487 .coefficient, .predecessor 1 17488 .coefficient])

def event17490 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49470⟩⟩) (.finite 60)

def event17491 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49471⟩⟩) 0 ⟨49470⟩ 17490

def event17492 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49471⟩⟩) (.identity (.predecessor 0 17491 .coefficient))

def exact17493RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], []⟩, (1)⟩]

theorem exact17493RawTermsValid :
    exact17493RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17493 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49471⟩⟩) exact17493RawTerms (.finite 60) 17492 .exactZero (none)

def event17494 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact17495RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact17495RawTermsValid :
    exact17495RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17495 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact17495RawTerms .large 17494 .exactZero (none)

def event17496 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49472⟩⟩) 0 ⟨6908⟩ 17495

def event17497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49472⟩⟩) 1 ⟨49471⟩ 17493

def event17498 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49472⟩⟩) (.product (.predecessor 0 17496 .coefficient) (.predecessor 1 17497 .coefficient) (⟨false, false, none, none, none⟩))

def event17499 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49472⟩⟩, .operator (⟨17495, 0⟩, ⟨17493, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact17500RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact17500RawTermsValid :
    exact17500RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17500 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49472⟩⟩) exact17500RawTerms .large 17498 .exactZero (none)

def event17501 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7196⟩⟩) 0 ⟨7177⟩ 17477

def event17502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7196⟩⟩) (.authority (.operator))

def exact17503RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩]

theorem exact17503RawTermsValid :
    exact17503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17503 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7196⟩⟩) exact17503RawTerms .large 17502 .exactZero (none)

def event17504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49473⟩⟩) 0 ⟨7196⟩ 17503

def event17505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49473⟩⟩) 1 ⟨49472⟩ 17500

def event17506 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49473⟩⟩) (.sum [.predecessor 0 17504 .coefficient, .predecessor 1 17505 .coefficient])

def exact17507RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17507RawTermsValid :
    exact17507RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17507 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49473⟩⟩) exact17507RawTerms .large 17506 .exactZero (none)

def event17508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49812⟩⟩) 0 ⟨49473⟩ 17507

def event17509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49812⟩⟩) 1 ⟨49811⟩ 17484

def event17510 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49812⟩⟩) (.product (.predecessor 0 17508 .coefficient) (.predecessor 1 17509 .coefficient) (⟨false, false, none, none, none⟩))

def event17511 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49812⟩⟩, .operator (⟨17507, 1⟩, ⟨17484, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩, (-1)⟩)

def event17512 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49812⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49811⟩⟩) ⟨49223⟩ 17481)

def event17513 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49812⟩⟩, .relation 17512 0, ⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨49223⟩⟩]⟩, (-1)⟩)

def event17514 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49812⟩⟩, .operator (⟨17507, 0⟩, ⟨17484, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩, (1)⟩)

def exact17515RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨49223⟩⟩]⟩, (-1)⟩]

theorem exact17515RawTermsValid :
    exact17515RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17515 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49812⟩⟩) exact17515RawTerms .large 17510 .exactZero (none)

def event17516 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48249⟩⟩) 0 ⟨48079⟩ 17473

def event17517 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48249⟩⟩) (.authority (.programFamilyFact))

def exact17518RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48249⟩⟩], []⟩, (1)⟩]

theorem exact17518RawTermsValid :
    exact17518RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17518 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48249⟩⟩) exact17518RawTerms (.finite 63) 17517 .exactZero (none)

def event17519 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48250⟩⟩) 0 ⟨6908⟩ 17495

def event17520 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48250⟩⟩) 1 ⟨48249⟩ 17518

def event17521 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48250⟩⟩) (.product (.predecessor 0 17519 .coefficient) (.predecessor 1 17520 .coefficient) (⟨false, true, none, none, some 1⟩))

def event17522 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48250⟩⟩, .operator (⟨17495, 0⟩, ⟨17518, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48249⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact17523RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48249⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact17523RawTermsValid :
    exact17523RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17523 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48250⟩⟩) exact17523RawTerms .large 17521 .exactZero (none)

def event17524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7232⟩⟩) 0 ⟨7177⟩ 17477

def event17525 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7232⟩⟩) (.authority (.operator))

def exact17526RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact17526RawTermsValid :
    exact17526RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17526 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7232⟩⟩) exact17526RawTerms .large 17525 .exactZero (none)

def event17527 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48251⟩⟩) 0 ⟨7232⟩ 17526

def event17528 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48251⟩⟩) 1 ⟨48250⟩ 17523

def event17529 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48251⟩⟩) (.sum [.predecessor 0 17527 .coefficient, .predecessor 1 17528 .coefficient])

def exact17530RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48249⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17530RawTermsValid :
    exact17530RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17530 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48251⟩⟩) exact17530RawTerms .large 17529 .exactZero (none)

def event17531 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49815⟩⟩) 0 ⟨48251⟩ 17530

def event17532 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49815⟩⟩) 1 ⟨49812⟩ 17515

def event17533 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49815⟩⟩) (.sum [.predecessor 0 17531 .coefficient, .predecessor 1 17532 .coefficient])

def exact17534RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨49223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48249⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17534RawTermsValid :
    exact17534RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17534 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49815⟩⟩) exact17534RawTerms .large 17533 .exactZero (none)

def event17535 : Event := .preFoldPolynomial 17534 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨49223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48249⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact17536RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨49223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48249⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event17536 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨49815⟩⟩) 17535 exact17536RawTerms .large 17533 .exactZero (none)

def event17537 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨48079⟩⟩) ⟨⟨111⟩, ⟨94⟩, ⟨135⟩⟩ ⟨17379, 17537⟩

def event17538 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨48725⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48722⟩⟩]⟩) (1) 0 2 (.universal 17537 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48722⟩⟩]⟩) (none) 17536)

def event17539 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48725⟩⟩, .relation 17538 2, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨49223⟩⟩]⟩, (1)⟩)

def event17540 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48725⟩⟩, .relation 17538 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩, (-1)⟩)

def event17541 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48725⟩⟩, .relation 17538 3, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48249⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event17542 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48725⟩⟩, .relation 17538 1, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩)

def exact17543RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨49223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48249⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17543RawTermsValid :
    exact17543RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17543 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48725⟩⟩) exact17543RawTerms .large 17375 (.finite 202072841853861888) (some (17377))

def event17544 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49814⟩⟩) 0 ⟨48725⟩ 17543

def event17545 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49814⟩⟩) 1 ⟨49813⟩ 17365

def event17546 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49814⟩⟩) (.sum [.predecessor 0 17544 .coefficient, .predecessor 1 17545 .coefficient])

def event17547 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49814⟩⟩, .operator (⟨17543, 2⟩, ⟨17365, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨49223⟩⟩]⟩, (-1)⟩)

def event17548 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49814⟩⟩, .operator (⟨17543, 0⟩, ⟨17365, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩, (1)⟩)

def event17549 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49814⟩⟩) (.sum [.result 17543 .summary, .result 17365 .summary])

def exact17550RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48249⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17550RawTermsValid :
    exact17550RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17550 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49814⟩⟩) exact17550RawTerms .large 17546 (.finite 32194504275408640829496428331008) (some (17549))

def event17551 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46541⟩⟩) 0 ⟨45399⟩ 91

def event17552 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46541⟩⟩) (.authority (.programFamilyFact))

def event17553 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46541⟩⟩) (.finite 3720)

def event17554 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46543⟩⟩) 0 ⟨7177⟩ 15500

def event17555 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46543⟩⟩) 1 ⟨46541⟩ 17553

def event17556 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46543⟩⟩) (.authority (.operator))

def exact17557RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46543⟩⟩]⟩, (1)⟩]

theorem exact17557RawTermsValid :
    exact17557RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17557 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46543⟩⟩) exact17557RawTerms .large 17556 .exactZero (none)

def event17558 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47131⟩⟩) 0 ⟨46543⟩ 17557

def event17559 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47131⟩⟩) (.authority (.operator))

def exact17560RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47131⟩⟩]⟩, (1)⟩]

theorem exact17560RawTermsValid :
    exact17560RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17560 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47131⟩⟩) exact17560RawTerms (.finite 8192) 17559 .exactZero (none)

def event17561 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46416⟩⟩) 0 ⟨44948⟩ 85

def event17562 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46416⟩⟩) (.authority (.programFamilyFact))

def event17563 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46416⟩⟩) (.finite 3720)

def event17564 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46417⟩⟩) 0 ⟨7177⟩ 15500

def event17565 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46417⟩⟩) 1 ⟨46416⟩ 17563

def event17566 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46417⟩⟩) (.authority (.operator))

def exact17567RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46417⟩⟩]⟩, (1)⟩]

theorem exact17567RawTermsValid :
    exact17567RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17567 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46417⟩⟩) exact17567RawTerms .large 17566 .exactZero (none)

def event17568 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46883⟩⟩) 0 ⟨46417⟩ 17567

def event17569 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46883⟩⟩) (.authority (.operator))

def exact17570RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46883⟩⟩]⟩, (1)⟩]

theorem exact17570RawTermsValid :
    exact17570RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17570 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46883⟩⟩) exact17570RawTerms (.finite 8192) 17569 .exactZero (none)

def event17571 : Event := .predecessor (⟨.program ⟨257⟩, ⟨110⟩⟩) 0 ⟨11⟩ 17049

def event17572 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨110⟩⟩) (.identity (.predecessor 0 17571 .coefficient))

def exact17573RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨110⟩⟩]⟩, (1)⟩]

theorem exact17573RawTermsValid :
    exact17573RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17573 : Event := .resultExact (⟨.program ⟨257⟩, ⟨110⟩⟩) exact17573RawTerms (.finite 26) 17572 .exactZero (none)

def event17574 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44949⟩⟩) 0 ⟨44946⟩ 74

def event17575 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44949⟩⟩) 1 ⟨6914⟩ 17057

def event17576 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44949⟩⟩) (.tensor (.predecessor 0 17574 .coefficient) (.predecessor 1 17575 .coefficient) true false)

def event17577 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44949⟩⟩, .operator (⟨74, 0⟩, ⟨17057, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact17578RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact17578RawTermsValid :
    exact17578RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17578 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44949⟩⟩) exact17578RawTerms .large 17576 .exactZero (none)

def event17579 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7284⟩⟩) 0 ⟨7178⟩ 15893

def event17580 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7284⟩⟩) (.identity (.predecessor 0 17579 .coefficient))

def exact17581RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩]

theorem exact17581RawTermsValid :
    exact17581RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17581 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7284⟩⟩) exact17581RawTerms .large 17580 .exactZero (none)

def event17582 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7602⟩⟩) 0 ⟨5441⟩ 16922

def event17583 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7602⟩⟩) 1 ⟨7284⟩ 17581

def event17584 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7602⟩⟩) (.product (.predecessor 0 17582 .coefficient) (.predecessor 1 17583 .coefficient) (⟨false, false, none, none, none⟩))

def event17585 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7602⟩⟩, .operator (⟨16922, 0⟩, ⟨17581, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩)

def exact17586RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩]

theorem exact17586RawTermsValid :
    exact17586RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17586 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7602⟩⟩) exact17586RawTerms .large 17584 .exactZero (none)

def event17587 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44950⟩⟩) 0 ⟨7602⟩ 17586

def event17588 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44950⟩⟩) 1 ⟨44949⟩ 17578

def event17589 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44950⟩⟩) (.sum [.predecessor 0 17587 .coefficient, .predecessor 1 17588 .coefficient])

def exact17590RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17590RawTermsValid :
    exact17590RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17590 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44950⟩⟩) exact17590RawTerms .large 17589 .exactZero (none)

def event17591 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44951⟩⟩) 0 ⟨44950⟩ 17590

def event17592 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44951⟩⟩) 1 ⟨110⟩ 17573

def event17593 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44951⟩⟩) (.sum [.predecessor 0 17591 .coefficient, .predecessor 1 17592 .coefficient])

def event17594 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44951⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨110⟩⟩]⟩) [⟨.result 17573 .coefficient, false, none⟩])

def event17595 : Event := .survivorFold (1) 17594

def exact17596RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17596RawTermsValid :
    exact17596RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17596 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44951⟩⟩) exact17596RawTerms .large 17593 (.finite 26) (some (17594))

def event17597 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44952⟩⟩) 0 ⟨44951⟩ 17596

def event17598 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44952⟩⟩) 1 ⟨14651⟩ 77

def event17599 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44952⟩⟩) (.product (.predecessor 0 17597 .coefficient) (.predecessor 1 17598 .coefficient) (⟨false, true, none, none, some 1⟩))

def event17600 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44952⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14651⟩⟩], []⟩) [⟨.result 77 .coefficient, true, some 1⟩])

def event17601 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44952⟩⟩) (.product (.result 17596 .summary) (.transfer 17600) (⟨false, false, none, none, none⟩))

def event17602 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44952⟩⟩, .operator (⟨17596, 1⟩, ⟨77, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event17603 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44952⟩⟩, .operator (⟨17596, 0⟩, ⟨77, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩)

def exact17604RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17604RawTermsValid :
    exact17604RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17604 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44952⟩⟩) exact17604RawTerms .large 17599 (.finite 49414144) (some (17601))

def event17605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9562⟩⟩) 0 ⟨7284⟩ 17581

def event17606 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9562⟩⟩) (.authority (.operator))

def exact17607RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩]

theorem exact17607RawTermsValid :
    exact17607RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17607 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9562⟩⟩) exact17607RawTerms (.finite 8192) 17606 .exactZero (none)

def event17608 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9563⟩⟩) 0 ⟨9562⟩ 17607

def event17609 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9563⟩⟩) 1 ⟨2370⟩ 4

def event17610 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9563⟩⟩) (.scale (.predecessor 0 17608 .coefficient) (.value (.predecessor 1 17609 .coefficient)))

def exact17611RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩]

theorem exact17611RawTermsValid :
    exact17611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17611 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9563⟩⟩) exact17611RawTerms (.finite 8192) 17610 .exactZero (none)

def event17612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨127⟩⟩) 0 ⟨11⟩ 17049

def event17613 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨127⟩⟩) (.identity (.predecessor 0 17612 .coefficient))

def exact17614RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨127⟩⟩]⟩, (1)⟩]

theorem exact17614RawTermsValid :
    exact17614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17614 : Event := .resultExact (⟨.program ⟨257⟩, ⟨127⟩⟩) exact17614RawTerms (.finite 26) 17613 .exactZero (none)

def event17615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14652⟩⟩) 0 ⟨14651⟩ 77

def event17616 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14652⟩⟩) 1 ⟨6914⟩ 17057

def event17617 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14652⟩⟩) (.tensor (.predecessor 0 17615 .coefficient) (.predecessor 1 17616 .coefficient) true false)

def event17618 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14652⟩⟩, .operator (⟨77, 0⟩, ⟨17057, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact17619RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact17619RawTermsValid :
    exact17619RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17619 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14652⟩⟩) exact17619RawTerms .large 17617 .exactZero (none)

def event17620 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7301⟩⟩) 0 ⟨7178⟩ 15893

def event17621 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7301⟩⟩) (.identity (.predecessor 0 17620 .coefficient))

def exact17622RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩]

theorem exact17622RawTermsValid :
    exact17622RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17622 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7301⟩⟩) exact17622RawTerms .large 17621 .exactZero (none)

def event17623 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7619⟩⟩) 0 ⟨5441⟩ 16922

def event17624 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7619⟩⟩) 1 ⟨7301⟩ 17622

def event17625 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7619⟩⟩) (.product (.predecessor 0 17623 .coefficient) (.predecessor 1 17624 .coefficient) (⟨false, false, none, none, none⟩))

def event17626 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7619⟩⟩, .operator (⟨16922, 0⟩, ⟨17622, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩)

def exact17627RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩]

theorem exact17627RawTermsValid :
    exact17627RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17627 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7619⟩⟩) exact17627RawTerms .large 17625 .exactZero (none)

def event17628 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14653⟩⟩) 0 ⟨7619⟩ 17627

def event17629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14653⟩⟩) 1 ⟨14652⟩ 17619

def event17630 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14653⟩⟩) (.sum [.predecessor 0 17628 .coefficient, .predecessor 1 17629 .coefficient])

def exact17631RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17631RawTermsValid :
    exact17631RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17631 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14653⟩⟩) exact17631RawTerms .large 17630 .exactZero (none)

def event17632 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14654⟩⟩) 0 ⟨14653⟩ 17631

def event17633 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14654⟩⟩) 1 ⟨127⟩ 17614

def event17634 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14654⟩⟩) (.sum [.predecessor 0 17632 .coefficient, .predecessor 1 17633 .coefficient])

def event17635 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14654⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨127⟩⟩]⟩) [⟨.result 17614 .coefficient, false, none⟩])

def event17636 : Event := .survivorFold (1) 17635

def exact17637RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17637RawTermsValid :
    exact17637RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17637 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14654⟩⟩) exact17637RawTerms .large 17634 (.finite 26) (some (17635))

def event17638 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14655⟩⟩) 0 ⟨14654⟩ 17637

def event17639 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14655⟩⟩) 1 ⟨9563⟩ 17611

def event17640 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14655⟩⟩) (.product (.predecessor 0 17638 .coefficient) (.predecessor 1 17639 .coefficient) (⟨false, false, none, none, none⟩))

def event17641 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14655⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) [⟨.result 17607 .coefficient, false, none⟩])

def event17642 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14655⟩⟩) (.product (.result 17637 .summary) (.transfer 17641) (⟨false, false, none, none, none⟩))

def event17643 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14655⟩⟩, .operator (⟨17637, 1⟩, ⟨17611, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (-1)⟩)

def event17644 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨14655⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9562⟩⟩) ⟨7284⟩ 17581)

def event17645 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14655⟩⟩, .relation 17644 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (-1)⟩)

def event17646 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14655⟩⟩, .operator (⟨17637, 0⟩, ⟨17611, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩)

def exact17647RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (-1)⟩]

theorem exact17647RawTermsValid :
    exact17647RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17647 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14655⟩⟩) exact17647RawTerms .large 17640 (.finite 279172874240) (some (17642))

def event17648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44953⟩⟩) 0 ⟨14655⟩ 17647

def event17649 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44953⟩⟩) 1 ⟨44952⟩ 17604

def event17650 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44953⟩⟩) (.sum [.predecessor 0 17648 .coefficient, .predecessor 1 17649 .coefficient])

def event17651 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44953⟩⟩, .operator (⟨17647, 1⟩, ⟨17604, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩)

def event17652 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44953⟩⟩) (.sum [.result 17647 .summary, .result 17604 .summary])

def exact17653RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact17653RawTermsValid :
    exact17653RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17653 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44953⟩⟩) exact17653RawTerms .large 17650 (.finite 279222288384) (some (17652))

def event17654 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46884⟩⟩) 0 ⟨44953⟩ 17653

def event17655 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46884⟩⟩) 1 ⟨46883⟩ 17570

def event17656 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46884⟩⟩) (.product (.predecessor 0 17654 .coefficient) (.predecessor 1 17655 .coefficient) (⟨false, false, none, none, none⟩))

def event17657 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46884⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨46883⟩⟩]⟩) [⟨.result 17570 .coefficient, false, none⟩])

def event17658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46884⟩⟩) (.product (.result 17653 .summary) (.transfer 17657) (⟨false, false, none, none, none⟩))

def event17659 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46884⟩⟩, .operator (⟨17653, 1⟩, ⟨17570, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩]⟩, (-1)⟩)

def event17660 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨46884⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨46883⟩⟩) ⟨46417⟩ 17567)

def event17661 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46884⟩⟩, .relation 17660 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨46417⟩⟩]⟩, (-1)⟩)

def event17662 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46884⟩⟩, .operator (⟨17653, 0⟩, ⟨17570, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩]⟩, (1)⟩)

def exact17663RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨46417⟩⟩]⟩, (-1)⟩]

theorem exact17663RawTermsValid :
    exact17663RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17663 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46884⟩⟩) exact17663RawTerms .large 17656 (.finite 2998126492308901724160) (some (17658))

def eventLeaf1088 : Array AnnotatedEvent := #[
  { event := event17408
    frameStart := 17379 },
  { event := event17409
    frameStart := 17379 },
  { event := event17410
    frameStart := 17379 },
  { event := event17411
    frameStart := 17379 },
  { event := event17412
    frameStart := 17379 },
  { event := event17413
    frameStart := 17379 },
  { event := event17414
    frameStart := 17379 },
  { event := event17415
    frameStart := 17379 },
  { event := event17416
    frameStart := 17379 },
  { event := event17417
    frameStart := 17379 },
  { event := event17418
    frameStart := 17379 },
  { event := event17419
    frameStart := 17379 },
  { event := event17420
    frameStart := 17379 },
  { event := event17421
    frameStart := 17379 },
  { event := event17422
    frameStart := 17379 },
  { event := event17423
    frameStart := 17379 }
]

def eventLeaf1089 : Array AnnotatedEvent := #[
  { event := event17424
    frameStart := 17379 },
  { event := event17425
    frameStart := 17379 },
  { event := event17426
    frameStart := 17379 },
  { event := event17427
    frameStart := 17379 },
  { event := event17428
    frameStart := 17379 },
  { event := event17429
    frameStart := 17379 },
  { event := event17430
    frameStart := 17379 },
  { event := event17431
    frameStart := 17379 },
  { event := event17432
    frameStart := 17379 },
  { event := event17433
    frameStart := 17433 },
  { event := event17434
    frameStart := 17433 },
  { event := event17435
    frameStart := 17433 },
  { event := event17436
    frameStart := 17433 },
  { event := event17437
    frameStart := 17433 },
  { event := event17438
    frameStart := 17433 },
  { event := event17439
    frameStart := 17433 }
]

def eventLeaf1090 : Array AnnotatedEvent := #[
  { event := event17440
    frameStart := 17433 },
  { event := event17441
    frameStart := 17433 },
  { event := event17442
    frameStart := 17433 },
  { event := event17443
    frameStart := 17433 },
  { event := event17444
    frameStart := 17433 },
  { event := event17445
    frameStart := 17433 },
  { event := event17446
    frameStart := 17433 },
  { event := event17447
    frameStart := 17433 },
  { event := event17448
    frameStart := 17433 },
  { event := event17449
    frameStart := 17433 },
  { event := event17450
    frameStart := 17433 },
  { event := event17451
    frameStart := 17433 },
  { event := event17452
    frameStart := 17433 },
  { event := event17453
    frameStart := 17433 },
  { event := event17454
    frameStart := 17433 },
  { event := event17455
    frameStart := 17433 }
]

def eventLeaf1091 : Array AnnotatedEvent := #[
  { event := event17456
    frameStart := 17433 },
  { event := event17457
    frameStart := 17433 },
  { event := event17458
    frameStart := 17433 },
  { event := event17459
    frameStart := 17433 },
  { event := event17460
    frameStart := 17433 },
  { event := event17461
    frameStart := 17433 },
  { event := event17462
    frameStart := 17433 },
  { event := event17463
    frameStart := 17433 },
  { event := event17464
    frameStart := 17433 },
  { event := event17465
    frameStart := 17433 },
  { event := event17466
    frameStart := 17433 },
  { event := event17467
    frameStart := 17433 },
  { event := event17468
    frameStart := 17433 },
  { event := event17469
    frameStart := 17433 },
  { event := event17470
    frameStart := 17433 },
  { event := event17471
    frameStart := 17433 }
]

def eventLeaf1092 : Array AnnotatedEvent := #[
  { event := event17472
    frameStart := 17433 },
  { event := event17473
    frameStart := 17433 },
  { event := event17474
    frameStart := 17433 },
  { event := event17475
    frameStart := 17433 },
  { event := event17476
    frameStart := 17433 },
  { event := event17477
    frameStart := 17433 },
  { event := event17478
    frameStart := 17433 },
  { event := event17479
    frameStart := 17433 },
  { event := event17480
    frameStart := 17433 },
  { event := event17481
    frameStart := 17433 },
  { event := event17482
    frameStart := 17433 },
  { event := event17483
    frameStart := 17433 },
  { event := event17484
    frameStart := 17433 },
  { event := event17485
    frameStart := 17433 },
  { event := event17486
    frameStart := 17433 },
  { event := event17487
    frameStart := 17433 }
]

def eventLeaf1093 : Array AnnotatedEvent := #[
  { event := event17488
    frameStart := 17433 },
  { event := event17489
    frameStart := 17433 },
  { event := event17490
    frameStart := 17433 },
  { event := event17491
    frameStart := 17433 },
  { event := event17492
    frameStart := 17433 },
  { event := event17493
    frameStart := 17433 },
  { event := event17494
    frameStart := 17433 },
  { event := event17495
    frameStart := 17433 },
  { event := event17496
    frameStart := 17433 },
  { event := event17497
    frameStart := 17433 },
  { event := event17498
    frameStart := 17433 },
  { event := event17499
    frameStart := 17433 },
  { event := event17500
    frameStart := 17433 },
  { event := event17501
    frameStart := 17433 },
  { event := event17502
    frameStart := 17433 },
  { event := event17503
    frameStart := 17433 }
]

def eventLeaf1094 : Array AnnotatedEvent := #[
  { event := event17504
    frameStart := 17433 },
  { event := event17505
    frameStart := 17433 },
  { event := event17506
    frameStart := 17433 },
  { event := event17507
    frameStart := 17433 },
  { event := event17508
    frameStart := 17433 },
  { event := event17509
    frameStart := 17433 },
  { event := event17510
    frameStart := 17433 },
  { event := event17511
    frameStart := 17433 },
  { event := event17512
    frameStart := 17433 },
  { event := event17513
    frameStart := 17433 },
  { event := event17514
    frameStart := 17433 },
  { event := event17515
    frameStart := 17433 },
  { event := event17516
    frameStart := 17433 },
  { event := event17517
    frameStart := 17433 },
  { event := event17518
    frameStart := 17433 },
  { event := event17519
    frameStart := 17433 }
]

def eventLeaf1095 : Array AnnotatedEvent := #[
  { event := event17520
    frameStart := 17433 },
  { event := event17521
    frameStart := 17433 },
  { event := event17522
    frameStart := 17433 },
  { event := event17523
    frameStart := 17433 },
  { event := event17524
    frameStart := 17433 },
  { event := event17525
    frameStart := 17433 },
  { event := event17526
    frameStart := 17433 },
  { event := event17527
    frameStart := 17433 },
  { event := event17528
    frameStart := 17433 },
  { event := event17529
    frameStart := 17433 },
  { event := event17530
    frameStart := 17433 },
  { event := event17531
    frameStart := 17433 },
  { event := event17532
    frameStart := 17433 },
  { event := event17533
    frameStart := 17433 },
  { event := event17534
    frameStart := 17433 },
  { event := event17535
    frameStart := 17433 }
]

def eventLeaf1096 : Array AnnotatedEvent := #[
  { event := event17536
    frameStart := 17433 },
  { event := event17537
    frameStart := 0 },
  { event := event17538
    frameStart := 0 },
  { event := event17539
    frameStart := 0 },
  { event := event17540
    frameStart := 0 },
  { event := event17541
    frameStart := 0 },
  { event := event17542
    frameStart := 0 },
  { event := event17543
    frameStart := 0 },
  { event := event17544
    frameStart := 0 },
  { event := event17545
    frameStart := 0 },
  { event := event17546
    frameStart := 0 },
  { event := event17547
    frameStart := 0 },
  { event := event17548
    frameStart := 0 },
  { event := event17549
    frameStart := 0 },
  { event := event17550
    frameStart := 0 },
  { event := event17551
    frameStart := 0 }
]

def eventLeaf1097 : Array AnnotatedEvent := #[
  { event := event17552
    frameStart := 0 },
  { event := event17553
    frameStart := 0 },
  { event := event17554
    frameStart := 0 },
  { event := event17555
    frameStart := 0 },
  { event := event17556
    frameStart := 0 },
  { event := event17557
    frameStart := 0 },
  { event := event17558
    frameStart := 0 },
  { event := event17559
    frameStart := 0 },
  { event := event17560
    frameStart := 0 },
  { event := event17561
    frameStart := 0 },
  { event := event17562
    frameStart := 0 },
  { event := event17563
    frameStart := 0 },
  { event := event17564
    frameStart := 0 },
  { event := event17565
    frameStart := 0 },
  { event := event17566
    frameStart := 0 },
  { event := event17567
    frameStart := 0 }
]

def eventLeaf1098 : Array AnnotatedEvent := #[
  { event := event17568
    frameStart := 0 },
  { event := event17569
    frameStart := 0 },
  { event := event17570
    frameStart := 0 },
  { event := event17571
    frameStart := 0 },
  { event := event17572
    frameStart := 0 },
  { event := event17573
    frameStart := 0 },
  { event := event17574
    frameStart := 0 },
  { event := event17575
    frameStart := 0 },
  { event := event17576
    frameStart := 0 },
  { event := event17577
    frameStart := 0 },
  { event := event17578
    frameStart := 0 },
  { event := event17579
    frameStart := 0 },
  { event := event17580
    frameStart := 0 },
  { event := event17581
    frameStart := 0 },
  { event := event17582
    frameStart := 0 },
  { event := event17583
    frameStart := 0 }
]

def eventLeaf1099 : Array AnnotatedEvent := #[
  { event := event17584
    frameStart := 0 },
  { event := event17585
    frameStart := 0 },
  { event := event17586
    frameStart := 0 },
  { event := event17587
    frameStart := 0 },
  { event := event17588
    frameStart := 0 },
  { event := event17589
    frameStart := 0 },
  { event := event17590
    frameStart := 0 },
  { event := event17591
    frameStart := 0 },
  { event := event17592
    frameStart := 0 },
  { event := event17593
    frameStart := 0 },
  { event := event17594
    frameStart := 0 },
  { event := event17595
    frameStart := 0 },
  { event := event17596
    frameStart := 0 },
  { event := event17597
    frameStart := 0 },
  { event := event17598
    frameStart := 0 },
  { event := event17599
    frameStart := 0 }
]

def eventLeaf1100 : Array AnnotatedEvent := #[
  { event := event17600
    frameStart := 0 },
  { event := event17601
    frameStart := 0 },
  { event := event17602
    frameStart := 0 },
  { event := event17603
    frameStart := 0 },
  { event := event17604
    frameStart := 0 },
  { event := event17605
    frameStart := 0 },
  { event := event17606
    frameStart := 0 },
  { event := event17607
    frameStart := 0 },
  { event := event17608
    frameStart := 0 },
  { event := event17609
    frameStart := 0 },
  { event := event17610
    frameStart := 0 },
  { event := event17611
    frameStart := 0 },
  { event := event17612
    frameStart := 0 },
  { event := event17613
    frameStart := 0 },
  { event := event17614
    frameStart := 0 },
  { event := event17615
    frameStart := 0 }
]

def eventLeaf1101 : Array AnnotatedEvent := #[
  { event := event17616
    frameStart := 0 },
  { event := event17617
    frameStart := 0 },
  { event := event17618
    frameStart := 0 },
  { event := event17619
    frameStart := 0 },
  { event := event17620
    frameStart := 0 },
  { event := event17621
    frameStart := 0 },
  { event := event17622
    frameStart := 0 },
  { event := event17623
    frameStart := 0 },
  { event := event17624
    frameStart := 0 },
  { event := event17625
    frameStart := 0 },
  { event := event17626
    frameStart := 0 },
  { event := event17627
    frameStart := 0 },
  { event := event17628
    frameStart := 0 },
  { event := event17629
    frameStart := 0 },
  { event := event17630
    frameStart := 0 },
  { event := event17631
    frameStart := 0 }
]

def eventLeaf1102 : Array AnnotatedEvent := #[
  { event := event17632
    frameStart := 0 },
  { event := event17633
    frameStart := 0 },
  { event := event17634
    frameStart := 0 },
  { event := event17635
    frameStart := 0 },
  { event := event17636
    frameStart := 0 },
  { event := event17637
    frameStart := 0 },
  { event := event17638
    frameStart := 0 },
  { event := event17639
    frameStart := 0 },
  { event := event17640
    frameStart := 0 },
  { event := event17641
    frameStart := 0 },
  { event := event17642
    frameStart := 0 },
  { event := event17643
    frameStart := 0 },
  { event := event17644
    frameStart := 0 },
  { event := event17645
    frameStart := 0 },
  { event := event17646
    frameStart := 0 },
  { event := event17647
    frameStart := 0 }
]

def eventLeaf1103 : Array AnnotatedEvent := #[
  { event := event17648
    frameStart := 0 },
  { event := event17649
    frameStart := 0 },
  { event := event17650
    frameStart := 0 },
  { event := event17651
    frameStart := 0 },
  { event := event17652
    frameStart := 0 },
  { event := event17653
    frameStart := 0 },
  { event := event17654
    frameStart := 0 },
  { event := event17655
    frameStart := 0 },
  { event := event17656
    frameStart := 0 },
  { event := event17657
    frameStart := 0 },
  { event := event17658
    frameStart := 0 },
  { event := event17659
    frameStart := 0 },
  { event := event17660
    frameStart := 0 },
  { event := event17661
    frameStart := 0 },
  { event := event17662
    frameStart := 0 },
  { event := event17663
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events068
