import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events068

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event17408 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13186⟩⟩) (.authority (.programFamilyFact))

def exact17409RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13186⟩⟩], []⟩, (1)⟩]

theorem exact17409RawTermsValid :
    exact17409RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17409 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13186⟩⟩) exact17409RawTerms (.finite 58) 17408 .exactZero (none)

def event17410 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10260⟩⟩) 0 ⟨5560⟩ 17406

def event17411 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10260⟩⟩) (.authority (.programFamilyFact))

def exact17412RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10260⟩⟩], []⟩, (1)⟩]

theorem exact17412RawTermsValid :
    exact17412RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17412 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10260⟩⟩) exact17412RawTerms (.finite 58) 17411 .exactZero (none)

def event17413 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13187⟩⟩) 0 ⟨10260⟩ 17412

def event17414 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13187⟩⟩) 1 ⟨13186⟩ 17409

def event17415 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13187⟩⟩) (.product (.predecessor 0 17413 .coefficient) (.predecessor 1 17414 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event17416 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13187⟩⟩, .operator (⟨17412, 0⟩, ⟨17409, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], []⟩, (1)⟩)

def exact17417RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], []⟩, (1)⟩]

theorem exact17417RawTermsValid :
    exact17417RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17417 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13187⟩⟩) exact17417RawTerms (.finite 3364) 17415 .exactZero (none)

def event17418 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13188⟩⟩) 0 ⟨13187⟩ 17417

def event17419 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13188⟩⟩) (.identity (.predecessor 0 17418 .coefficient))

def event17420 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13188⟩⟩) (.finite 3364)

def event17421 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16887⟩⟩) 0 ⟨13188⟩ 17420

def event17422 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16887⟩⟩) (.authority (.programFamilyFact))

def exact17423RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16887⟩⟩], []⟩, (1)⟩]

theorem exact17423RawTermsValid :
    exact17423RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17423 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16887⟩⟩) exact17423RawTerms (.finite 58) 17422 .exactZero (none)

def event17424 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16888⟩⟩) 0 ⟨16887⟩ 17423

def event17425 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16888⟩⟩) (.identity (.predecessor 0 17424 .coefficient))

def event17426 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16888⟩⟩) (.finite 58)

def event17427 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24739⟩⟩) 0 ⟨16888⟩ 17426

def event17428 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24739⟩⟩) (.authority (.programFamilyFact))

def event17429 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24739⟩⟩) (.finite 3720)

def event17430 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event17431 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24740⟩⟩) 0 ⟨6689⟩ 17430

def event17432 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24740⟩⟩) 1 ⟨24739⟩ 17429

def event17433 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24740⟩⟩) (.authority (.operator))

def exact17434RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24740⟩⟩]⟩, (1)⟩]

theorem exact17434RawTermsValid :
    exact17434RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17434 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24740⟩⟩) exact17434RawTerms .large 17433 .exactZero (none)

def event17435 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29864⟩⟩) 0 ⟨24740⟩ 17434

def event17436 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29864⟩⟩) (.authority (.operator))

def exact17437RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29864⟩⟩]⟩, (1)⟩]

theorem exact17437RawTermsValid :
    exact17437RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17437 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29864⟩⟩) exact17437RawTerms (.finite 8192) 17436 .exactZero (none)

def event17438 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event17439 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event17440 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16983⟩⟩) 0 ⟨16888⟩ 17426

def event17441 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16983⟩⟩) 1 ⟨110⟩ 17439

def event17442 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16983⟩⟩) (.sum [.predecessor 0 17440 .coefficient, .predecessor 1 17441 .coefficient])

def event17443 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16983⟩⟩) (.finite 58)

def event17444 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16984⟩⟩) 0 ⟨16983⟩ 17443

def event17445 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16984⟩⟩) (.identity (.predecessor 0 17444 .coefficient))

def exact17446RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16887⟩⟩], []⟩, (1)⟩]

theorem exact17446RawTermsValid :
    exact17446RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17446 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16984⟩⟩) exact17446RawTerms (.finite 58) 17445 .exactZero (none)

def event17447 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact17448RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact17448RawTermsValid :
    exact17448RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17448 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact17448RawTerms .large 17447 .exactZero (none)

def event17449 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16985⟩⟩) 0 ⟨6544⟩ 17448

def event17450 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16985⟩⟩) 1 ⟨16984⟩ 17446

def event17451 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16985⟩⟩) (.product (.predecessor 0 17449 .coefficient) (.predecessor 1 17450 .coefficient) (⟨false, false, none, none, none⟩))

def event17452 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16985⟩⟩, .operator (⟨17448, 0⟩, ⟨17446, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16887⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact17453RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16887⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact17453RawTermsValid :
    exact17453RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17453 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16985⟩⟩) exact17453RawTerms .large 17451 .exactZero (none)

def event17454 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6706⟩⟩) 0 ⟨6689⟩ 17430

def event17455 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6706⟩⟩) (.authority (.operator))

def exact17456RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩]⟩, (1)⟩]

theorem exact17456RawTermsValid :
    exact17456RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17456 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6706⟩⟩) exact17456RawTerms .large 17455 .exactZero (none)

def event17457 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16986⟩⟩) 0 ⟨6706⟩ 17456

def event17458 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16986⟩⟩) 1 ⟨16985⟩ 17453

def event17459 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16986⟩⟩) (.sum [.predecessor 0 17457 .coefficient, .predecessor 1 17458 .coefficient])

def exact17460RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16887⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact17460RawTermsValid :
    exact17460RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17460 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16986⟩⟩) exact17460RawTerms .large 17459 .exactZero (none)

def event17461 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29865⟩⟩) 0 ⟨16986⟩ 17460

def event17462 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29865⟩⟩) 1 ⟨29864⟩ 17437

def event17463 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29865⟩⟩) (.product (.predecessor 0 17461 .coefficient) (.predecessor 1 17462 .coefficient) (⟨false, false, none, none, none⟩))

def event17464 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29865⟩⟩, .operator (⟨17460, 1⟩, ⟨17437, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16887⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29864⟩⟩]⟩, (-1)⟩)

def event17465 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29865⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16887⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29864⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29864⟩⟩) ⟨24740⟩ 17434)

def event17466 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29865⟩⟩, .relation 17465 0, ⟨[⟨.program ⟨214⟩, ⟨16887⟩⟩], [⟨.program ⟨214⟩, ⟨24740⟩⟩]⟩, (-1)⟩)

def event17467 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29865⟩⟩, .operator (⟨17460, 0⟩, ⟨17437, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29864⟩⟩]⟩, (1)⟩)

def exact17468RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29864⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16887⟩⟩], [⟨.program ⟨214⟩, ⟨24740⟩⟩]⟩, (-1)⟩]

theorem exact17468RawTermsValid :
    exact17468RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17468 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29865⟩⟩) exact17468RawTerms .large 17463 .exactZero (none)

def event17469 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16943⟩⟩) 0 ⟨16888⟩ 17426

def event17470 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16943⟩⟩) (.authority (.programFamilyFact))

def exact17471RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16943⟩⟩], []⟩, (1)⟩]

theorem exact17471RawTermsValid :
    exact17471RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17471 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16943⟩⟩) exact17471RawTerms (.finite 58) 17470 .exactZero (none)

def event17472 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16945⟩⟩) 0 ⟨6544⟩ 17448

def event17473 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16945⟩⟩) 1 ⟨16943⟩ 17471

def event17474 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16945⟩⟩) (.product (.predecessor 0 17472 .coefficient) (.predecessor 1 17473 .coefficient) (⟨false, true, none, none, some 1⟩))

def event17475 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16945⟩⟩, .operator (⟨17448, 0⟩, ⟨17471, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact17476RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact17476RawTermsValid :
    exact17476RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17476 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16945⟩⟩) exact17476RawTerms .large 17474 .exactZero (none)

def event17477 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6740⟩⟩) 0 ⟨6689⟩ 17430

def event17478 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6740⟩⟩) (.authority (.operator))

def exact17479RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩]⟩, (1)⟩]

theorem exact17479RawTermsValid :
    exact17479RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17479 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6740⟩⟩) exact17479RawTerms .large 17478 .exactZero (none)

def event17480 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16946⟩⟩) 0 ⟨6740⟩ 17479

def event17481 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16946⟩⟩) 1 ⟨16945⟩ 17476

def event17482 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16946⟩⟩) (.sum [.predecessor 0 17480 .coefficient, .predecessor 1 17481 .coefficient])

def exact17483RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact17483RawTermsValid :
    exact17483RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17483 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16946⟩⟩) exact17483RawTerms .large 17482 .exactZero (none)

def event17484 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29870⟩⟩) 0 ⟨16946⟩ 17483

def event17485 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29870⟩⟩) 1 ⟨29865⟩ 17468

def event17486 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29870⟩⟩) (.sum [.predecessor 0 17484 .coefficient, .predecessor 1 17485 .coefficient])

def exact17487RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29864⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16887⟩⟩], [⟨.program ⟨214⟩, ⟨24740⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact17487RawTermsValid :
    exact17487RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17487 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29870⟩⟩) exact17487RawTerms .large 17486 .exactZero (none)

def event17488 : Event := .preFoldPolynomial 17487 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29864⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16887⟩⟩], [⟨.program ⟨214⟩, ⟨24740⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact17489RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29864⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6740⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16887⟩⟩], [⟨.program ⟨214⟩, ⟨24740⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event17489 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨29870⟩⟩) 17488 exact17489RawTerms .large 17486 .exactZero (none)

def event17490 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16888⟩⟩) ⟨⟨153⟩, ⟨62⟩, ⟨109⟩⟩ ⟨17332, 17490⟩

def event17491 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨22643⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22640⟩⟩]⟩) (1) 0 2 (.universal 17490 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22640⟩⟩]⟩) (none) 17489)

def event17492 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22643⟩⟩, .relation 17491 1, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6740⟩⟩]⟩, (1)⟩)

def event17493 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22643⟩⟩, .relation 17491 2, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16887⟩⟩], [⟨.program ⟨214⟩, ⟨24740⟩⟩]⟩, (1)⟩)

def event17494 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22643⟩⟩, .relation 17491 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29864⟩⟩]⟩, (-1)⟩)

def event17495 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22643⟩⟩, .relation 17491 3, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact17496RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29864⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6740⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16887⟩⟩], [⟨.program ⟨214⟩, ⟨24740⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact17496RawTermsValid :
    exact17496RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17496 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22643⟩⟩) exact17496RawTerms .large 17328 (.finite 1811303510016) (some (17330))

def event17497 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29867⟩⟩) 0 ⟨22643⟩ 17496

def event17498 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29867⟩⟩) 1 ⟨29866⟩ 17318

def event17499 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29867⟩⟩) (.sum [.predecessor 0 17497 .coefficient, .predecessor 1 17498 .coefficient])

def event17500 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29867⟩⟩, .operator (⟨17496, 2⟩, ⟨17318, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16887⟩⟩], [⟨.program ⟨214⟩, ⟨24740⟩⟩]⟩, (-1)⟩)

def event17501 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29867⟩⟩, .operator (⟨17496, 0⟩, ⟨17318, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29864⟩⟩]⟩, (1)⟩)

def event17502 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29867⟩⟩) (.sum [.result 17496 .summary, .result 17318 .summary])

def exact17503RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6740⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact17503RawTermsValid :
    exact17503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17503 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29867⟩⟩) exact17503RawTerms .large 17499 (.finite 1292516722839998050304) (some (17502))

def event17504 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29868⟩⟩) 0 ⟨29867⟩ 17503

def event17505 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29868⟩⟩) 1 ⟨6660⟩ 5539

def event17506 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29868⟩⟩) (.product (.predecessor 0 17504 .coefficient) (.predecessor 1 17505 .coefficient) (⟨false, false, none, none, none⟩))

def event17507 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29868⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩) [⟨.result 5535 .coefficient, false, none⟩])

def event17508 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29868⟩⟩) (.product (.result 17503 .summary) (.transfer 17507) (⟨false, false, none, none, none⟩))

def event17509 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29868⟩⟩, .operator (⟨17503, 0⟩, ⟨5539, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩, (1)⟩)

def event17510 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29868⟩⟩, .operator (⟨17503, 1⟩, ⟨5539, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩, (-1)⟩)

def event17511 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29868⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6659⟩⟩) ⟨6601⟩ 5532)

def event17512 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29868⟩⟩, .relation 17511 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact17513RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6740⟩⟩, ⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact17513RawTermsValid :
    exact17513RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17513 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29868⟩⟩) exact17513RawTerms .large 17506 (.finite 4743557053090358284584484864) (some (17508))

def event17514 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24677⟩⟩) 0 ⟨6689⟩ 5477

def event17515 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24677⟩⟩) 1 ⟨24676⟩ 7446

def event17516 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24677⟩⟩) (.authority (.operator))

def exact17517RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24677⟩⟩]⟩, (1)⟩]

theorem exact17517RawTermsValid :
    exact17517RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17517 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24677⟩⟩) exact17517RawTerms .large 17516 .exactZero (none)

def event17518 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29647⟩⟩) 0 ⟨24677⟩ 17517

def event17519 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29647⟩⟩) (.authority (.operator))

def exact17520RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29647⟩⟩]⟩, (1)⟩]

theorem exact17520RawTermsValid :
    exact17520RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17520 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29647⟩⟩) exact17520RawTerms (.finite 8192) 17519 .exactZero (none)

def event17521 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29649⟩⟩) 0 ⟨25626⟩ 7749

def event17522 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29649⟩⟩) 1 ⟨29647⟩ 17520

def event17523 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29649⟩⟩) (.product (.predecessor 0 17521 .coefficient) (.predecessor 1 17522 .coefficient) (⟨false, false, none, none, none⟩))

def event17524 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29649⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨29647⟩⟩]⟩) [⟨.result 17520 .coefficient, false, none⟩])

def event17525 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29649⟩⟩) (.product (.result 7749 .summary) (.transfer 17524) (⟨false, false, none, none, none⟩))

def event17526 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29649⟩⟩, .operator (⟨7749, 1⟩, ⟨17520, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16768⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29647⟩⟩]⟩, (-1)⟩)

def event17527 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29649⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16768⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29647⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29647⟩⟩) ⟨24677⟩ 17517)

def event17528 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29649⟩⟩, .relation 17527 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16768⟩⟩], [⟨.program ⟨214⟩, ⟨24677⟩⟩]⟩, (-1)⟩)

def event17529 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29649⟩⟩, .operator (⟨7749, 0⟩, ⟨17520, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29647⟩⟩]⟩, (1)⟩)

def exact17530RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29647⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16768⟩⟩], [⟨.program ⟨214⟩, ⟨24677⟩⟩]⟩, (-1)⟩]

theorem exact17530RawTermsValid :
    exact17530RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17530 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29649⟩⟩) exact17530RawTerms .large 17523 (.finite 1292449483693632782336) (some (17525))

def event17531 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22496⟩⟩) 0 ⟨16769⟩ 114

def event17532 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22496⟩⟩) (.authority (.relationPreimageSource ⟨60⟩))

def exact17533RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22496⟩⟩]⟩, (1)⟩]

theorem exact17533RawTermsValid :
    exact17533RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17533 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22496⟩⟩) exact17533RawTerms (.finite 136065468) 17532 .exactZero (none)

def event17534 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22498⟩⟩) 0 ⟨22496⟩ 17533

def event17535 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22498⟩⟩) 1 ⟨2348⟩ 4

def event17536 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22498⟩⟩) (.scale (.predecessor 0 17534 .coefficient) (.value (.predecessor 1 17535 .coefficient)))

def exact17537RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22496⟩⟩]⟩, (1)⟩]

theorem exact17537RawTermsValid :
    exact17537RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17537 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22498⟩⟩) exact17537RawTerms (.finite 136065468) 17536 .exactZero (none)

def event17538 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22499⟩⟩) 0 ⟨5565⟩ 6561

def event17539 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22499⟩⟩) 1 ⟨22498⟩ 17537

def event17540 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22499⟩⟩) (.product (.predecessor 0 17538 .coefficient) (.predecessor 1 17539 .coefficient) (⟨false, false, none, none, none⟩))

def event17541 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22499⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22496⟩⟩]⟩) [⟨.result 17533 .coefficient, false, none⟩])

def event17542 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22499⟩⟩) (.product (.result 6561 .summary) (.transfer 17541) (⟨false, false, none, none, none⟩))

def event17543 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22499⟩⟩, .operator (⟨6561, 0⟩, ⟨17537, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22496⟩⟩]⟩, (1)⟩)

def event17544 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22497⟩⟩)

def event17545 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event17546 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event17547 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.authority (.operator))

def event17548 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.finite 6)

def event17549 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event17550 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event17551 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event17552 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event17553 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 17552

def event17554 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 17550

def event17555 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 17553 .coefficient) (.value (.predecessor 1 17554 .coefficient)))

def event17556 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event17557 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 0 ⟨5503⟩ 17556

def event17558 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 1 ⟨5245⟩ 17548

def event17559 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.sum [.predecessor 0 17557 .coefficient, .predecessor 1 17558 .coefficient])

def event17560 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.finite 223)

def event17561 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 0 ⟨5516⟩ 17560

def event17562 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 1 ⟨961⟩ 17546

def event17563 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.identity (.predecessor 1 17562 .coefficient))

def event17564 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.finite 224)

def event17565 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12990⟩⟩) 0 ⟨5560⟩ 17564

def event17566 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12990⟩⟩) (.authority (.programFamilyFact))

def exact17567RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12990⟩⟩], []⟩, (1)⟩]

theorem exact17567RawTermsValid :
    exact17567RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17567 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12990⟩⟩) exact17567RawTerms (.finite 52) 17566 .exactZero (none)

def event17568 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10155⟩⟩) 0 ⟨5560⟩ 17564

def event17569 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10155⟩⟩) (.authority (.programFamilyFact))

def exact17570RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10155⟩⟩], []⟩, (1)⟩]

theorem exact17570RawTermsValid :
    exact17570RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17570 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10155⟩⟩) exact17570RawTerms (.finite 52) 17569 .exactZero (none)

def event17571 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12991⟩⟩) 0 ⟨10155⟩ 17570

def event17572 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12991⟩⟩) 1 ⟨12990⟩ 17567

def event17573 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12991⟩⟩) (.product (.predecessor 0 17571 .coefficient) (.predecessor 1 17572 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event17574 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12991⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], []⟩) [⟨.result 17570 .coefficient, true, some 1⟩, ⟨.result 17567 .coefficient, true, some 1⟩])

def event17575 : Event := .survivorFold (1) 17574

def exact17576RawTerms : List Term := []

theorem exact17576RawTermsValid :
    exact17576RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17576 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12991⟩⟩) exact17576RawTerms (.finite 2704) 17573 (.finite 2704) (some (17574))

def event17577 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12992⟩⟩) 0 ⟨12991⟩ 17576

def event17578 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12992⟩⟩) (.identity (.predecessor 0 17577 .coefficient))

def event17579 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12992⟩⟩) (.finite 2704)

def event17580 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16768⟩⟩) 0 ⟨12992⟩ 17579

def event17581 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16768⟩⟩) (.authority (.programFamilyFact))

def exact17582RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16768⟩⟩], []⟩, (1)⟩]

theorem exact17582RawTermsValid :
    exact17582RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17582 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16768⟩⟩) exact17582RawTerms (.finite 52) 17581 .exactZero (none)

def event17583 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16769⟩⟩) 0 ⟨16768⟩ 17582

def event17584 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16769⟩⟩) (.identity (.predecessor 0 17583 .coefficient))

def event17585 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16769⟩⟩) (.finite 52)

def event17586 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22496⟩⟩) 0 ⟨16769⟩ 17585

def event17587 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22496⟩⟩) (.authority (.relationPreimageSource ⟨60⟩))

def exact17588RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22496⟩⟩]⟩, (1)⟩]

theorem exact17588RawTermsValid :
    exact17588RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17588 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22496⟩⟩) exact17588RawTerms (.finite 136065468) 17587 .exactZero (none)

def event17589 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact17590RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact17590RawTermsValid :
    exact17590RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17590 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact17590RawTerms .large 17589 .exactZero (none)

def event17591 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22497⟩⟩) 0 ⟨6⟩ 17590

def event17592 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22497⟩⟩) 1 ⟨22496⟩ 17588

def event17593 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22497⟩⟩) (.product (.predecessor 0 17591 .coefficient) (.predecessor 1 17592 .coefficient) (⟨false, false, none, none, none⟩))

def event17594 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22497⟩⟩, .operator (⟨17590, 0⟩, ⟨17588, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22496⟩⟩]⟩, (1)⟩)

def exact17595RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22496⟩⟩]⟩, (1)⟩]

theorem exact17595RawTermsValid :
    exact17595RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17595 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22497⟩⟩) exact17595RawTerms .large 17593 .exactZero (none)

def event17596 : Event := .preFoldPolynomial 17595 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22496⟩⟩]⟩, (1)⟩] .exactZero none

def exact17597RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22496⟩⟩]⟩, (1)⟩]

def event17597 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨22497⟩⟩) 17596 exact17597RawTerms .large 17593 .exactZero (none)

def event17598 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨29653⟩⟩)

def event17599 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event17600 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event17601 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.authority (.operator))

def event17602 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.finite 6)

def event17603 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event17604 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event17605 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event17606 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event17607 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 17606

def event17608 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 17604

def event17609 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 17607 .coefficient) (.value (.predecessor 1 17608 .coefficient)))

def event17610 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event17611 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 0 ⟨5503⟩ 17610

def event17612 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 1 ⟨5245⟩ 17602

def event17613 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.sum [.predecessor 0 17611 .coefficient, .predecessor 1 17612 .coefficient])

def event17614 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.finite 223)

def event17615 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 0 ⟨5516⟩ 17614

def event17616 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 1 ⟨961⟩ 17600

def event17617 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.identity (.predecessor 1 17616 .coefficient))

def event17618 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.finite 224)

def event17619 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12990⟩⟩) 0 ⟨5560⟩ 17618

def event17620 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12990⟩⟩) (.authority (.programFamilyFact))

def exact17621RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12990⟩⟩], []⟩, (1)⟩]

theorem exact17621RawTermsValid :
    exact17621RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17621 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12990⟩⟩) exact17621RawTerms (.finite 52) 17620 .exactZero (none)

def event17622 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10155⟩⟩) 0 ⟨5560⟩ 17618

def event17623 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10155⟩⟩) (.authority (.programFamilyFact))

def exact17624RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10155⟩⟩], []⟩, (1)⟩]

theorem exact17624RawTermsValid :
    exact17624RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17624 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10155⟩⟩) exact17624RawTerms (.finite 52) 17623 .exactZero (none)

def event17625 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12991⟩⟩) 0 ⟨10155⟩ 17624

def event17626 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12991⟩⟩) 1 ⟨12990⟩ 17621

def event17627 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12991⟩⟩) (.product (.predecessor 0 17625 .coefficient) (.predecessor 1 17626 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event17628 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12991⟩⟩, .operator (⟨17624, 0⟩, ⟨17621, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], []⟩, (1)⟩)

def exact17629RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10155⟩⟩, ⟨.program ⟨214⟩, ⟨12990⟩⟩], []⟩, (1)⟩]

theorem exact17629RawTermsValid :
    exact17629RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17629 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12991⟩⟩) exact17629RawTerms (.finite 2704) 17627 .exactZero (none)

def event17630 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12992⟩⟩) 0 ⟨12991⟩ 17629

def event17631 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12992⟩⟩) (.identity (.predecessor 0 17630 .coefficient))

def event17632 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12992⟩⟩) (.finite 2704)

def event17633 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16768⟩⟩) 0 ⟨12992⟩ 17632

def event17634 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16768⟩⟩) (.authority (.programFamilyFact))

def exact17635RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16768⟩⟩], []⟩, (1)⟩]

theorem exact17635RawTermsValid :
    exact17635RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17635 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16768⟩⟩) exact17635RawTerms (.finite 52) 17634 .exactZero (none)

def event17636 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16769⟩⟩) 0 ⟨16768⟩ 17635

def event17637 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16769⟩⟩) (.identity (.predecessor 0 17636 .coefficient))

def event17638 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16769⟩⟩) (.finite 52)

def event17639 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24676⟩⟩) 0 ⟨16769⟩ 17638

def event17640 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24676⟩⟩) (.authority (.programFamilyFact))

def event17641 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24676⟩⟩) (.finite 3720)

def event17642 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event17643 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24677⟩⟩) 0 ⟨6689⟩ 17642

def event17644 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24677⟩⟩) 1 ⟨24676⟩ 17641

def event17645 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24677⟩⟩) (.authority (.operator))

def exact17646RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24677⟩⟩]⟩, (1)⟩]

theorem exact17646RawTermsValid :
    exact17646RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17646 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24677⟩⟩) exact17646RawTerms .large 17645 .exactZero (none)

def event17647 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29647⟩⟩) 0 ⟨24677⟩ 17646

def event17648 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29647⟩⟩) (.authority (.operator))

def exact17649RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29647⟩⟩]⟩, (1)⟩]

theorem exact17649RawTermsValid :
    exact17649RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17649 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29647⟩⟩) exact17649RawTerms (.finite 8192) 17648 .exactZero (none)

def event17650 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event17651 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event17652 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16843⟩⟩) 0 ⟨16769⟩ 17638

def event17653 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16843⟩⟩) 1 ⟨110⟩ 17651

def event17654 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16843⟩⟩) (.sum [.predecessor 0 17652 .coefficient, .predecessor 1 17653 .coefficient])

def event17655 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16843⟩⟩) (.finite 52)

def event17656 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16844⟩⟩) 0 ⟨16843⟩ 17655

def event17657 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16844⟩⟩) (.identity (.predecessor 0 17656 .coefficient))

def exact17658RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16768⟩⟩], []⟩, (1)⟩]

theorem exact17658RawTermsValid :
    exact17658RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17658 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16844⟩⟩) exact17658RawTerms (.finite 52) 17657 .exactZero (none)

def event17659 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact17660RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact17660RawTermsValid :
    exact17660RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event17660 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact17660RawTerms .large 17659 .exactZero (none)

def event17661 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16845⟩⟩) 0 ⟨6544⟩ 17660

def event17662 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16845⟩⟩) 1 ⟨16844⟩ 17658

def event17663 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16845⟩⟩) (.product (.predecessor 0 17661 .coefficient) (.predecessor 1 17662 .coefficient) (⟨false, false, none, none, none⟩))

def eventLeaf1088 : Array AnnotatedEvent := #[
  { event := event17408
    frameStart := 17386 },
  { event := event17409
    frameStart := 17386 },
  { event := event17410
    frameStart := 17386 },
  { event := event17411
    frameStart := 17386 },
  { event := event17412
    frameStart := 17386 },
  { event := event17413
    frameStart := 17386 },
  { event := event17414
    frameStart := 17386 },
  { event := event17415
    frameStart := 17386 },
  { event := event17416
    frameStart := 17386 },
  { event := event17417
    frameStart := 17386 },
  { event := event17418
    frameStart := 17386 },
  { event := event17419
    frameStart := 17386 },
  { event := event17420
    frameStart := 17386 },
  { event := event17421
    frameStart := 17386 },
  { event := event17422
    frameStart := 17386 },
  { event := event17423
    frameStart := 17386 }
]

def eventLeaf1089 : Array AnnotatedEvent := #[
  { event := event17424
    frameStart := 17386 },
  { event := event17425
    frameStart := 17386 },
  { event := event17426
    frameStart := 17386 },
  { event := event17427
    frameStart := 17386 },
  { event := event17428
    frameStart := 17386 },
  { event := event17429
    frameStart := 17386 },
  { event := event17430
    frameStart := 17386 },
  { event := event17431
    frameStart := 17386 },
  { event := event17432
    frameStart := 17386 },
  { event := event17433
    frameStart := 17386 },
  { event := event17434
    frameStart := 17386 },
  { event := event17435
    frameStart := 17386 },
  { event := event17436
    frameStart := 17386 },
  { event := event17437
    frameStart := 17386 },
  { event := event17438
    frameStart := 17386 },
  { event := event17439
    frameStart := 17386 }
]

def eventLeaf1090 : Array AnnotatedEvent := #[
  { event := event17440
    frameStart := 17386 },
  { event := event17441
    frameStart := 17386 },
  { event := event17442
    frameStart := 17386 },
  { event := event17443
    frameStart := 17386 },
  { event := event17444
    frameStart := 17386 },
  { event := event17445
    frameStart := 17386 },
  { event := event17446
    frameStart := 17386 },
  { event := event17447
    frameStart := 17386 },
  { event := event17448
    frameStart := 17386 },
  { event := event17449
    frameStart := 17386 },
  { event := event17450
    frameStart := 17386 },
  { event := event17451
    frameStart := 17386 },
  { event := event17452
    frameStart := 17386 },
  { event := event17453
    frameStart := 17386 },
  { event := event17454
    frameStart := 17386 },
  { event := event17455
    frameStart := 17386 }
]

def eventLeaf1091 : Array AnnotatedEvent := #[
  { event := event17456
    frameStart := 17386 },
  { event := event17457
    frameStart := 17386 },
  { event := event17458
    frameStart := 17386 },
  { event := event17459
    frameStart := 17386 },
  { event := event17460
    frameStart := 17386 },
  { event := event17461
    frameStart := 17386 },
  { event := event17462
    frameStart := 17386 },
  { event := event17463
    frameStart := 17386 },
  { event := event17464
    frameStart := 17386 },
  { event := event17465
    frameStart := 17386 },
  { event := event17466
    frameStart := 17386 },
  { event := event17467
    frameStart := 17386 },
  { event := event17468
    frameStart := 17386 },
  { event := event17469
    frameStart := 17386 },
  { event := event17470
    frameStart := 17386 },
  { event := event17471
    frameStart := 17386 }
]

def eventLeaf1092 : Array AnnotatedEvent := #[
  { event := event17472
    frameStart := 17386 },
  { event := event17473
    frameStart := 17386 },
  { event := event17474
    frameStart := 17386 },
  { event := event17475
    frameStart := 17386 },
  { event := event17476
    frameStart := 17386 },
  { event := event17477
    frameStart := 17386 },
  { event := event17478
    frameStart := 17386 },
  { event := event17479
    frameStart := 17386 },
  { event := event17480
    frameStart := 17386 },
  { event := event17481
    frameStart := 17386 },
  { event := event17482
    frameStart := 17386 },
  { event := event17483
    frameStart := 17386 },
  { event := event17484
    frameStart := 17386 },
  { event := event17485
    frameStart := 17386 },
  { event := event17486
    frameStart := 17386 },
  { event := event17487
    frameStart := 17386 }
]

def eventLeaf1093 : Array AnnotatedEvent := #[
  { event := event17488
    frameStart := 17386 },
  { event := event17489
    frameStart := 17386 },
  { event := event17490
    frameStart := 0 },
  { event := event17491
    frameStart := 0 },
  { event := event17492
    frameStart := 0 },
  { event := event17493
    frameStart := 0 },
  { event := event17494
    frameStart := 0 },
  { event := event17495
    frameStart := 0 },
  { event := event17496
    frameStart := 0 },
  { event := event17497
    frameStart := 0 },
  { event := event17498
    frameStart := 0 },
  { event := event17499
    frameStart := 0 },
  { event := event17500
    frameStart := 0 },
  { event := event17501
    frameStart := 0 },
  { event := event17502
    frameStart := 0 },
  { event := event17503
    frameStart := 0 }
]

def eventLeaf1094 : Array AnnotatedEvent := #[
  { event := event17504
    frameStart := 0 },
  { event := event17505
    frameStart := 0 },
  { event := event17506
    frameStart := 0 },
  { event := event17507
    frameStart := 0 },
  { event := event17508
    frameStart := 0 },
  { event := event17509
    frameStart := 0 },
  { event := event17510
    frameStart := 0 },
  { event := event17511
    frameStart := 0 },
  { event := event17512
    frameStart := 0 },
  { event := event17513
    frameStart := 0 },
  { event := event17514
    frameStart := 0 },
  { event := event17515
    frameStart := 0 },
  { event := event17516
    frameStart := 0 },
  { event := event17517
    frameStart := 0 },
  { event := event17518
    frameStart := 0 },
  { event := event17519
    frameStart := 0 }
]

def eventLeaf1095 : Array AnnotatedEvent := #[
  { event := event17520
    frameStart := 0 },
  { event := event17521
    frameStart := 0 },
  { event := event17522
    frameStart := 0 },
  { event := event17523
    frameStart := 0 },
  { event := event17524
    frameStart := 0 },
  { event := event17525
    frameStart := 0 },
  { event := event17526
    frameStart := 0 },
  { event := event17527
    frameStart := 0 },
  { event := event17528
    frameStart := 0 },
  { event := event17529
    frameStart := 0 },
  { event := event17530
    frameStart := 0 },
  { event := event17531
    frameStart := 0 },
  { event := event17532
    frameStart := 0 },
  { event := event17533
    frameStart := 0 },
  { event := event17534
    frameStart := 0 },
  { event := event17535
    frameStart := 0 }
]

def eventLeaf1096 : Array AnnotatedEvent := #[
  { event := event17536
    frameStart := 0 },
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
    frameStart := 17544 },
  { event := event17545
    frameStart := 17544 },
  { event := event17546
    frameStart := 17544 },
  { event := event17547
    frameStart := 17544 },
  { event := event17548
    frameStart := 17544 },
  { event := event17549
    frameStart := 17544 },
  { event := event17550
    frameStart := 17544 },
  { event := event17551
    frameStart := 17544 }
]

def eventLeaf1097 : Array AnnotatedEvent := #[
  { event := event17552
    frameStart := 17544 },
  { event := event17553
    frameStart := 17544 },
  { event := event17554
    frameStart := 17544 },
  { event := event17555
    frameStart := 17544 },
  { event := event17556
    frameStart := 17544 },
  { event := event17557
    frameStart := 17544 },
  { event := event17558
    frameStart := 17544 },
  { event := event17559
    frameStart := 17544 },
  { event := event17560
    frameStart := 17544 },
  { event := event17561
    frameStart := 17544 },
  { event := event17562
    frameStart := 17544 },
  { event := event17563
    frameStart := 17544 },
  { event := event17564
    frameStart := 17544 },
  { event := event17565
    frameStart := 17544 },
  { event := event17566
    frameStart := 17544 },
  { event := event17567
    frameStart := 17544 }
]

def eventLeaf1098 : Array AnnotatedEvent := #[
  { event := event17568
    frameStart := 17544 },
  { event := event17569
    frameStart := 17544 },
  { event := event17570
    frameStart := 17544 },
  { event := event17571
    frameStart := 17544 },
  { event := event17572
    frameStart := 17544 },
  { event := event17573
    frameStart := 17544 },
  { event := event17574
    frameStart := 17544 },
  { event := event17575
    frameStart := 17544 },
  { event := event17576
    frameStart := 17544 },
  { event := event17577
    frameStart := 17544 },
  { event := event17578
    frameStart := 17544 },
  { event := event17579
    frameStart := 17544 },
  { event := event17580
    frameStart := 17544 },
  { event := event17581
    frameStart := 17544 },
  { event := event17582
    frameStart := 17544 },
  { event := event17583
    frameStart := 17544 }
]

def eventLeaf1099 : Array AnnotatedEvent := #[
  { event := event17584
    frameStart := 17544 },
  { event := event17585
    frameStart := 17544 },
  { event := event17586
    frameStart := 17544 },
  { event := event17587
    frameStart := 17544 },
  { event := event17588
    frameStart := 17544 },
  { event := event17589
    frameStart := 17544 },
  { event := event17590
    frameStart := 17544 },
  { event := event17591
    frameStart := 17544 },
  { event := event17592
    frameStart := 17544 },
  { event := event17593
    frameStart := 17544 },
  { event := event17594
    frameStart := 17544 },
  { event := event17595
    frameStart := 17544 },
  { event := event17596
    frameStart := 17544 },
  { event := event17597
    frameStart := 17544 },
  { event := event17598
    frameStart := 17598 },
  { event := event17599
    frameStart := 17598 }
]

def eventLeaf1100 : Array AnnotatedEvent := #[
  { event := event17600
    frameStart := 17598 },
  { event := event17601
    frameStart := 17598 },
  { event := event17602
    frameStart := 17598 },
  { event := event17603
    frameStart := 17598 },
  { event := event17604
    frameStart := 17598 },
  { event := event17605
    frameStart := 17598 },
  { event := event17606
    frameStart := 17598 },
  { event := event17607
    frameStart := 17598 },
  { event := event17608
    frameStart := 17598 },
  { event := event17609
    frameStart := 17598 },
  { event := event17610
    frameStart := 17598 },
  { event := event17611
    frameStart := 17598 },
  { event := event17612
    frameStart := 17598 },
  { event := event17613
    frameStart := 17598 },
  { event := event17614
    frameStart := 17598 },
  { event := event17615
    frameStart := 17598 }
]

def eventLeaf1101 : Array AnnotatedEvent := #[
  { event := event17616
    frameStart := 17598 },
  { event := event17617
    frameStart := 17598 },
  { event := event17618
    frameStart := 17598 },
  { event := event17619
    frameStart := 17598 },
  { event := event17620
    frameStart := 17598 },
  { event := event17621
    frameStart := 17598 },
  { event := event17622
    frameStart := 17598 },
  { event := event17623
    frameStart := 17598 },
  { event := event17624
    frameStart := 17598 },
  { event := event17625
    frameStart := 17598 },
  { event := event17626
    frameStart := 17598 },
  { event := event17627
    frameStart := 17598 },
  { event := event17628
    frameStart := 17598 },
  { event := event17629
    frameStart := 17598 },
  { event := event17630
    frameStart := 17598 },
  { event := event17631
    frameStart := 17598 }
]

def eventLeaf1102 : Array AnnotatedEvent := #[
  { event := event17632
    frameStart := 17598 },
  { event := event17633
    frameStart := 17598 },
  { event := event17634
    frameStart := 17598 },
  { event := event17635
    frameStart := 17598 },
  { event := event17636
    frameStart := 17598 },
  { event := event17637
    frameStart := 17598 },
  { event := event17638
    frameStart := 17598 },
  { event := event17639
    frameStart := 17598 },
  { event := event17640
    frameStart := 17598 },
  { event := event17641
    frameStart := 17598 },
  { event := event17642
    frameStart := 17598 },
  { event := event17643
    frameStart := 17598 },
  { event := event17644
    frameStart := 17598 },
  { event := event17645
    frameStart := 17598 },
  { event := event17646
    frameStart := 17598 },
  { event := event17647
    frameStart := 17598 }
]

def eventLeaf1103 : Array AnnotatedEvent := #[
  { event := event17648
    frameStart := 17598 },
  { event := event17649
    frameStart := 17598 },
  { event := event17650
    frameStart := 17598 },
  { event := event17651
    frameStart := 17598 },
  { event := event17652
    frameStart := 17598 },
  { event := event17653
    frameStart := 17598 },
  { event := event17654
    frameStart := 17598 },
  { event := event17655
    frameStart := 17598 },
  { event := event17656
    frameStart := 17598 },
  { event := event17657
    frameStart := 17598 },
  { event := event17658
    frameStart := 17598 },
  { event := event17659
    frameStart := 17598 },
  { event := event17660
    frameStart := 17598 },
  { event := event17661
    frameStart := 17598 },
  { event := event17662
    frameStart := 17598 },
  { event := event17663
    frameStart := 17598 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events068
