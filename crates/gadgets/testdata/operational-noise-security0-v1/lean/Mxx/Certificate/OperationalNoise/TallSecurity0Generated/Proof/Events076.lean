import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events076

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event19456 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.finite 6)

def event19457 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event19458 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event19459 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event19460 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event19461 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 19460

def event19462 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 19458

def event19463 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 19461 .coefficient) (.value (.predecessor 1 19462 .coefficient)))

def event19464 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event19465 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 0 ⟨5503⟩ 19464

def event19466 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 1 ⟨5245⟩ 19456

def event19467 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.sum [.predecessor 0 19465 .coefficient, .predecessor 1 19466 .coefficient])

def event19468 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.finite 223)

def event19469 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 0 ⟨5516⟩ 19468

def event19470 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 1 ⟨961⟩ 19454

def event19471 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.identity (.predecessor 1 19470 .coefficient))

def event19472 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.finite 224)

def event19473 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11401⟩⟩) 0 ⟨5560⟩ 19472

def event19474 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11401⟩⟩) (.authority (.programFamilyFact))

def exact19475RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11401⟩⟩], []⟩, (1)⟩]

theorem exact19475RawTermsValid :
    exact19475RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19475 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11401⟩⟩) exact19475RawTerms (.finite 16) 19474 .exactZero (none)

def event19476 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14026⟩⟩) 0 ⟨5560⟩ 19472

def event19477 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14026⟩⟩) (.authority (.programFamilyFact))

def exact19478RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14026⟩⟩], []⟩, (1)⟩]

theorem exact19478RawTermsValid :
    exact19478RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19478 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14026⟩⟩) exact19478RawTerms (.finite 16) 19477 .exactZero (none)

def event19479 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14027⟩⟩) 0 ⟨14026⟩ 19478

def event19480 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14027⟩⟩) 1 ⟨11401⟩ 19475

def event19481 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14027⟩⟩) (.product (.predecessor 0 19479 .coefficient) (.predecessor 1 19480 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event19482 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14027⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11401⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], []⟩) [⟨.result 19478 .coefficient, true, some 1⟩, ⟨.result 19475 .coefficient, true, some 1⟩])

def event19483 : Event := .survivorFold (1) 19482

def exact19484RawTerms : List Term := []

theorem exact19484RawTermsValid :
    exact19484RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19484 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14027⟩⟩) exact19484RawTerms (.finite 256) 19481 (.finite 256) (some (19482))

def event19485 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14028⟩⟩) 0 ⟨14027⟩ 19484

def event19486 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14028⟩⟩) (.identity (.predecessor 0 19485 .coefficient))

def event19487 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14028⟩⟩) (.finite 256)

def event19488 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15837⟩⟩) 0 ⟨14028⟩ 19487

def event19489 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15837⟩⟩) (.authority (.programFamilyFact))

def exact19490RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], []⟩, (1)⟩]

theorem exact19490RawTermsValid :
    exact19490RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19490 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15837⟩⟩) exact19490RawTerms (.finite 16) 19489 .exactZero (none)

def event19491 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15838⟩⟩) 0 ⟨15837⟩ 19490

def event19492 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15838⟩⟩) (.identity (.predecessor 0 19491 .coefficient))

def event19493 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15838⟩⟩) (.finite 16)

def event19494 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21200⟩⟩) 0 ⟨15838⟩ 19493

def event19495 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21200⟩⟩) (.authority (.relationPreimageSource ⟨40⟩))

def exact19496RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21200⟩⟩]⟩, (1)⟩]

theorem exact19496RawTermsValid :
    exact19496RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19496 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21200⟩⟩) exact19496RawTerms (.finite 136065468) 19495 .exactZero (none)

def event19497 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact19498RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact19498RawTermsValid :
    exact19498RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19498 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact19498RawTerms .large 19497 .exactZero (none)

def event19499 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21201⟩⟩) 0 ⟨6⟩ 19498

def event19500 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21201⟩⟩) 1 ⟨21200⟩ 19496

def event19501 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21201⟩⟩) (.product (.predecessor 0 19499 .coefficient) (.predecessor 1 19500 .coefficient) (⟨false, false, none, none, none⟩))

def event19502 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21201⟩⟩, .operator (⟨19498, 0⟩, ⟨19496, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21200⟩⟩]⟩, (1)⟩)

def exact19503RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21200⟩⟩]⟩, (1)⟩]

theorem exact19503RawTermsValid :
    exact19503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19503 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21201⟩⟩) exact19503RawTerms .large 19501 .exactZero (none)

def event19504 : Event := .preFoldPolynomial 19503 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21200⟩⟩]⟩, (1)⟩] .exactZero none

def exact19505RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21200⟩⟩]⟩, (1)⟩]

def event19505 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21201⟩⟩) 19504 exact19505RawTerms .large 19501 .exactZero (none)

def event19506 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27700⟩⟩)

def event19507 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event19508 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event19509 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.authority (.operator))

def event19510 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.finite 6)

def event19511 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event19512 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event19513 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event19514 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event19515 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 19514

def event19516 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 19512

def event19517 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 19515 .coefficient) (.value (.predecessor 1 19516 .coefficient)))

def event19518 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event19519 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 0 ⟨5503⟩ 19518

def event19520 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 1 ⟨5245⟩ 19510

def event19521 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.sum [.predecessor 0 19519 .coefficient, .predecessor 1 19520 .coefficient])

def event19522 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.finite 223)

def event19523 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 0 ⟨5516⟩ 19522

def event19524 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 1 ⟨961⟩ 19508

def event19525 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.identity (.predecessor 1 19524 .coefficient))

def event19526 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.finite 224)

def event19527 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11401⟩⟩) 0 ⟨5560⟩ 19526

def event19528 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11401⟩⟩) (.authority (.programFamilyFact))

def exact19529RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11401⟩⟩], []⟩, (1)⟩]

theorem exact19529RawTermsValid :
    exact19529RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19529 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11401⟩⟩) exact19529RawTerms (.finite 16) 19528 .exactZero (none)

def event19530 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14026⟩⟩) 0 ⟨5560⟩ 19526

def event19531 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14026⟩⟩) (.authority (.programFamilyFact))

def exact19532RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14026⟩⟩], []⟩, (1)⟩]

theorem exact19532RawTermsValid :
    exact19532RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19532 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14026⟩⟩) exact19532RawTerms (.finite 16) 19531 .exactZero (none)

def event19533 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14027⟩⟩) 0 ⟨14026⟩ 19532

def event19534 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14027⟩⟩) 1 ⟨11401⟩ 19529

def event19535 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14027⟩⟩) (.product (.predecessor 0 19533 .coefficient) (.predecessor 1 19534 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event19536 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14027⟩⟩, .operator (⟨19532, 0⟩, ⟨19529, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11401⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], []⟩, (1)⟩)

def exact19537RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11401⟩⟩, ⟨.program ⟨214⟩, ⟨14026⟩⟩], []⟩, (1)⟩]

theorem exact19537RawTermsValid :
    exact19537RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19537 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14027⟩⟩) exact19537RawTerms (.finite 256) 19535 .exactZero (none)

def event19538 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14028⟩⟩) 0 ⟨14027⟩ 19537

def event19539 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14028⟩⟩) (.identity (.predecessor 0 19538 .coefficient))

def event19540 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14028⟩⟩) (.finite 256)

def event19541 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15837⟩⟩) 0 ⟨14028⟩ 19540

def event19542 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15837⟩⟩) (.authority (.programFamilyFact))

def exact19543RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], []⟩, (1)⟩]

theorem exact19543RawTermsValid :
    exact19543RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19543 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15837⟩⟩) exact19543RawTerms (.finite 16) 19542 .exactZero (none)

def event19544 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15838⟩⟩) 0 ⟨15837⟩ 19543

def event19545 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15838⟩⟩) (.identity (.predecessor 0 19544 .coefficient))

def event19546 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15838⟩⟩) (.finite 16)

def event19547 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24109⟩⟩) 0 ⟨15838⟩ 19546

def event19548 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24109⟩⟩) (.authority (.programFamilyFact))

def event19549 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24109⟩⟩) (.finite 3720)

def event19550 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event19551 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24110⟩⟩) 0 ⟨6689⟩ 19550

def event19552 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24110⟩⟩) 1 ⟨24109⟩ 19549

def event19553 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24110⟩⟩) (.authority (.operator))

def exact19554RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24110⟩⟩]⟩, (1)⟩]

theorem exact19554RawTermsValid :
    exact19554RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19554 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24110⟩⟩) exact19554RawTerms .large 19553 .exactZero (none)

def event19555 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27694⟩⟩) 0 ⟨24110⟩ 19554

def event19556 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27694⟩⟩) (.authority (.operator))

def exact19557RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27694⟩⟩]⟩, (1)⟩]

theorem exact19557RawTermsValid :
    exact19557RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19557 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27694⟩⟩) exact19557RawTerms (.finite 8192) 19556 .exactZero (none)

def event19558 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event19559 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event19560 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15912⟩⟩) 0 ⟨15838⟩ 19546

def event19561 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15912⟩⟩) 1 ⟨110⟩ 19559

def event19562 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15912⟩⟩) (.sum [.predecessor 0 19560 .coefficient, .predecessor 1 19561 .coefficient])

def event19563 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15912⟩⟩) (.finite 16)

def event19564 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15913⟩⟩) 0 ⟨15912⟩ 19563

def event19565 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15913⟩⟩) (.identity (.predecessor 0 19564 .coefficient))

def exact19566RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], []⟩, (1)⟩]

theorem exact19566RawTermsValid :
    exact19566RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19566 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15913⟩⟩) exact19566RawTerms (.finite 16) 19565 .exactZero (none)

def event19567 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact19568RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact19568RawTermsValid :
    exact19568RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19568 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact19568RawTerms .large 19567 .exactZero (none)

def event19569 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15914⟩⟩) 0 ⟨6544⟩ 19568

def event19570 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15914⟩⟩) 1 ⟨15913⟩ 19566

def event19571 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15914⟩⟩) (.product (.predecessor 0 19569 .coefficient) (.predecessor 1 19570 .coefficient) (⟨false, false, none, none, none⟩))

def event19572 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15914⟩⟩, .operator (⟨19568, 0⟩, ⟨19566, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact19573RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact19573RawTermsValid :
    exact19573RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19573 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15914⟩⟩) exact19573RawTerms .large 19571 .exactZero (none)

def event19574 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6696⟩⟩) 0 ⟨6689⟩ 19550

def event19575 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6696⟩⟩) (.authority (.operator))

def exact19576RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩]

theorem exact19576RawTermsValid :
    exact19576RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19576 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6696⟩⟩) exact19576RawTerms .large 19575 .exactZero (none)

def event19577 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15915⟩⟩) 0 ⟨6696⟩ 19576

def event19578 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15915⟩⟩) 1 ⟨15914⟩ 19573

def event19579 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15915⟩⟩) (.sum [.predecessor 0 19577 .coefficient, .predecessor 1 19578 .coefficient])

def exact19580RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact19580RawTermsValid :
    exact19580RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19580 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15915⟩⟩) exact19580RawTerms .large 19579 .exactZero (none)

def event19581 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27695⟩⟩) 0 ⟨15915⟩ 19580

def event19582 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27695⟩⟩) 1 ⟨27694⟩ 19557

def event19583 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27695⟩⟩) (.product (.predecessor 0 19581 .coefficient) (.predecessor 1 19582 .coefficient) (⟨false, false, none, none, none⟩))

def event19584 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27695⟩⟩, .operator (⟨19580, 1⟩, ⟨19557, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27694⟩⟩]⟩, (-1)⟩)

def event19585 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27695⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27694⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27694⟩⟩) ⟨24110⟩ 19554)

def event19586 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27695⟩⟩, .relation 19585 0, ⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨24110⟩⟩]⟩, (-1)⟩)

def event19587 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27695⟩⟩, .operator (⟨19580, 0⟩, ⟨19557, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27694⟩⟩]⟩, (1)⟩)

def exact19588RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27694⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨24110⟩⟩]⟩, (-1)⟩]

theorem exact19588RawTermsValid :
    exact19588RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19588 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27695⟩⟩) exact19588RawTerms .large 19583 .exactZero (none)

def event19589 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17237⟩⟩) 0 ⟨15838⟩ 19546

def event19590 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17237⟩⟩) (.authority (.programFamilyFact))

def exact19591RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17237⟩⟩], []⟩, (1)⟩]

theorem exact19591RawTermsValid :
    exact19591RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19591 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17237⟩⟩) exact19591RawTerms (.finite 16) 19590 .exactZero (none)

def event19592 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17239⟩⟩) 0 ⟨6544⟩ 19568

def event19593 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17239⟩⟩) 1 ⟨17237⟩ 19591

def event19594 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17239⟩⟩) (.product (.predecessor 0 19592 .coefficient) (.predecessor 1 19593 .coefficient) (⟨false, true, none, none, some 1⟩))

def event19595 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17239⟩⟩, .operator (⟨19568, 0⟩, ⟨19591, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact19596RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact19596RawTermsValid :
    exact19596RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19596 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17239⟩⟩) exact19596RawTerms .large 19594 .exactZero (none)

def event19597 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6720⟩⟩) 0 ⟨6689⟩ 19550

def event19598 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6720⟩⟩) (.authority (.operator))

def exact19599RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩]

theorem exact19599RawTermsValid :
    exact19599RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19599 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6720⟩⟩) exact19599RawTerms .large 19598 .exactZero (none)

def event19600 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17240⟩⟩) 0 ⟨6720⟩ 19599

def event19601 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17240⟩⟩) 1 ⟨17239⟩ 19596

def event19602 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17240⟩⟩) (.sum [.predecessor 0 19600 .coefficient, .predecessor 1 19601 .coefficient])

def exact19603RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact19603RawTermsValid :
    exact19603RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19603 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17240⟩⟩) exact19603RawTerms .large 19602 .exactZero (none)

def event19604 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27700⟩⟩) 0 ⟨17240⟩ 19603

def event19605 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27700⟩⟩) 1 ⟨27695⟩ 19588

def event19606 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27700⟩⟩) (.sum [.predecessor 0 19604 .coefficient, .predecessor 1 19605 .coefficient])

def exact19607RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27694⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨24110⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact19607RawTermsValid :
    exact19607RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19607 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27700⟩⟩) exact19607RawTerms .large 19606 .exactZero (none)

def event19608 : Event := .preFoldPolynomial 19607 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27694⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨24110⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact19609RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27694⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨24110⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event19609 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27700⟩⟩) 19608 exact19609RawTerms .large 19606 .exactZero (none)

def event19610 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15838⟩⟩) ⟨⟨133⟩, ⟨40⟩, ⟨109⟩⟩ ⟨19452, 19610⟩

def event19611 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21203⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21200⟩⟩]⟩) (1) 0 2 (.universal 19610 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21200⟩⟩]⟩) (none) 19609)

def event19612 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21203⟩⟩, .relation 19611 1, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩)

def event19613 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21203⟩⟩, .relation 19611 2, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨24110⟩⟩]⟩, (1)⟩)

def event19614 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21203⟩⟩, .relation 19611 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27694⟩⟩]⟩, (-1)⟩)

def event19615 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21203⟩⟩, .relation 19611 3, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact19616RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27694⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨24110⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact19616RawTermsValid :
    exact19616RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19616 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21203⟩⟩) exact19616RawTerms .large 19448 (.finite 1811303510016) (some (19450))

def event19617 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27697⟩⟩) 0 ⟨21203⟩ 19616

def event19618 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27697⟩⟩) 1 ⟨27696⟩ 19438

def event19619 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27697⟩⟩) (.sum [.predecessor 0 19617 .coefficient, .predecessor 1 19618 .coefficient])

def event19620 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27697⟩⟩, .operator (⟨19616, 2⟩, ⟨19438, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15837⟩⟩], [⟨.program ⟨214⟩, ⟨24110⟩⟩]⟩, (-1)⟩)

def event19621 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27697⟩⟩, .operator (⟨19616, 0⟩, ⟨19438, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27694⟩⟩]⟩, (1)⟩)

def event19622 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27697⟩⟩) (.sum [.result 19616 .summary, .result 19438 .summary])

def exact19623RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact19623RawTermsValid :
    exact19623RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19623 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27697⟩⟩) exact19623RawTerms .large 19619 (.finite 1292046061494565744640) (some (19622))

def event19624 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27698⟩⟩) 0 ⟨27697⟩ 19623

def event19625 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27698⟩⟩) 1 ⟨6644⟩ 5739

def event19626 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27698⟩⟩) (.product (.predecessor 0 19624 .coefficient) (.predecessor 1 19625 .coefficient) (⟨false, false, none, none, none⟩))

def event19627 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27698⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩) [⟨.result 5735 .coefficient, false, none⟩])

def event19628 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27698⟩⟩) (.product (.result 19623 .summary) (.transfer 19627) (⟨false, false, none, none, none⟩))

def event19629 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27698⟩⟩, .operator (⟨19623, 0⟩, ⟨5739, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩)

def event19630 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27698⟩⟩, .operator (⟨19623, 1⟩, ⟨5739, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (-1)⟩)

def event19631 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27698⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6643⟩⟩) ⟨6593⟩ 5732)

def event19632 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27698⟩⟩, .relation 19631 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact19633RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact19633RawTermsValid :
    exact19633RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19633 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27698⟩⟩) exact19633RawTerms .large 19626 (.finite 4741829718422040195880714240) (some (19628))

def event19634 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24047⟩⟩) 0 ⟨6689⟩ 5477

def event19635 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24047⟩⟩) 1 ⟨24046⟩ 12456

def event19636 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24047⟩⟩) (.authority (.operator))

def exact19637RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24047⟩⟩]⟩, (1)⟩]

theorem exact19637RawTermsValid :
    exact19637RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19637 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24047⟩⟩) exact19637RawTerms .large 19636 .exactZero (none)

def event19638 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27477⟩⟩) 0 ⟨24047⟩ 19637

def event19639 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27477⟩⟩) (.authority (.operator))

def exact19640RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27477⟩⟩]⟩, (1)⟩]

theorem exact19640RawTermsValid :
    exact19640RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19640 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27477⟩⟩) exact19640RawTerms (.finite 8192) 19639 .exactZero (none)

def event19641 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27479⟩⟩) 0 ⟨25934⟩ 12759

def event19642 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27479⟩⟩) 1 ⟨27477⟩ 19640

def event19643 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27479⟩⟩) (.product (.predecessor 0 19641 .coefficient) (.predecessor 1 19642 .coefficient) (⟨false, false, none, none, none⟩))

def event19644 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27479⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27477⟩⟩]⟩) [⟨.result 19640 .coefficient, false, none⟩])

def event19645 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27479⟩⟩) (.product (.result 12759 .summary) (.transfer 19644) (⟨false, false, none, none, none⟩))

def event19646 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27479⟩⟩, .operator (⟨12759, 1⟩, ⟨19640, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15718⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩]⟩, (-1)⟩)

def event19647 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27479⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15718⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27477⟩⟩) ⟨24047⟩ 19637)

def event19648 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27479⟩⟩, .relation 19647 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15718⟩⟩], [⟨.program ⟨214⟩, ⟨24047⟩⟩]⟩, (-1)⟩)

def event19649 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27479⟩⟩, .operator (⟨12759, 0⟩, ⟨19640, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩]⟩, (1)⟩)

def exact19650RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27477⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15718⟩⟩], [⟨.program ⟨214⟩, ⟨24047⟩⟩]⟩, (-1)⟩]

theorem exact19650RawTermsValid :
    exact19650RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19650 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27479⟩⟩) exact19650RawTerms .large 19643 (.finite 1292001234793221062656) (some (19645))

def event19651 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21056⟩⟩) 0 ⟨15719⟩ 344

def event19652 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21056⟩⟩) (.authority (.relationPreimageSource ⟨38⟩))

def exact19653RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21056⟩⟩]⟩, (1)⟩]

theorem exact19653RawTermsValid :
    exact19653RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19653 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21056⟩⟩) exact19653RawTerms (.finite 136065468) 19652 .exactZero (none)

def event19654 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21058⟩⟩) 0 ⟨21056⟩ 19653

def event19655 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21058⟩⟩) 1 ⟨2348⟩ 4

def event19656 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21058⟩⟩) (.scale (.predecessor 0 19654 .coefficient) (.value (.predecessor 1 19655 .coefficient)))

def exact19657RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21056⟩⟩]⟩, (1)⟩]

theorem exact19657RawTermsValid :
    exact19657RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19657 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21058⟩⟩) exact19657RawTerms (.finite 136065468) 19656 .exactZero (none)

def event19658 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21059⟩⟩) 0 ⟨5565⟩ 6561

def event19659 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21059⟩⟩) 1 ⟨21058⟩ 19657

def event19660 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21059⟩⟩) (.product (.predecessor 0 19658 .coefficient) (.predecessor 1 19659 .coefficient) (⟨false, false, none, none, none⟩))

def event19661 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21059⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21056⟩⟩]⟩) [⟨.result 19653 .coefficient, false, none⟩])

def event19662 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21059⟩⟩) (.product (.result 6561 .summary) (.transfer 19661) (⟨false, false, none, none, none⟩))

def event19663 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21059⟩⟩, .operator (⟨6561, 0⟩, ⟨19657, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21056⟩⟩]⟩, (1)⟩)

def event19664 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21057⟩⟩)

def event19665 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event19666 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event19667 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.authority (.operator))

def event19668 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.finite 6)

def event19669 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event19670 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event19671 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event19672 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event19673 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 19672

def event19674 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 19670

def event19675 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 19673 .coefficient) (.value (.predecessor 1 19674 .coefficient)))

def event19676 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event19677 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 0 ⟨5503⟩ 19676

def event19678 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 1 ⟨5245⟩ 19668

def event19679 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.sum [.predecessor 0 19677 .coefficient, .predecessor 1 19678 .coefficient])

def event19680 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.finite 223)

def event19681 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 0 ⟨5516⟩ 19680

def event19682 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 1 ⟨961⟩ 19666

def event19683 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.identity (.predecessor 1 19682 .coefficient))

def event19684 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.finite 224)

def event19685 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11317⟩⟩) 0 ⟨5560⟩ 19684

def event19686 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11317⟩⟩) (.authority (.programFamilyFact))

def exact19687RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11317⟩⟩], []⟩, (1)⟩]

theorem exact19687RawTermsValid :
    exact19687RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19687 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11317⟩⟩) exact19687RawTerms (.finite 12) 19686 .exactZero (none)

def event19688 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13809⟩⟩) 0 ⟨5560⟩ 19684

def event19689 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13809⟩⟩) (.authority (.programFamilyFact))

def exact19690RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13809⟩⟩], []⟩, (1)⟩]

theorem exact19690RawTermsValid :
    exact19690RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19690 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13809⟩⟩) exact19690RawTerms (.finite 12) 19689 .exactZero (none)

def event19691 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13810⟩⟩) 0 ⟨13809⟩ 19690

def event19692 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13810⟩⟩) 1 ⟨11317⟩ 19687

def event19693 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13810⟩⟩) (.product (.predecessor 0 19691 .coefficient) (.predecessor 1 19692 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event19694 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13810⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11317⟩⟩, ⟨.program ⟨214⟩, ⟨13809⟩⟩], []⟩) [⟨.result 19690 .coefficient, true, some 1⟩, ⟨.result 19687 .coefficient, true, some 1⟩])

def event19695 : Event := .survivorFold (1) 19694

def exact19696RawTerms : List Term := []

theorem exact19696RawTermsValid :
    exact19696RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19696 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13810⟩⟩) exact19696RawTerms (.finite 144) 19693 (.finite 144) (some (19694))

def event19697 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13811⟩⟩) 0 ⟨13810⟩ 19696

def event19698 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13811⟩⟩) (.identity (.predecessor 0 19697 .coefficient))

def event19699 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13811⟩⟩) (.finite 144)

def event19700 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15718⟩⟩) 0 ⟨13811⟩ 19699

def event19701 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15718⟩⟩) (.authority (.programFamilyFact))

def exact19702RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15718⟩⟩], []⟩, (1)⟩]

theorem exact19702RawTermsValid :
    exact19702RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19702 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15718⟩⟩) exact19702RawTerms (.finite 12) 19701 .exactZero (none)

def event19703 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15719⟩⟩) 0 ⟨15718⟩ 19702

def event19704 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15719⟩⟩) (.identity (.predecessor 0 19703 .coefficient))

def event19705 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15719⟩⟩) (.finite 12)

def event19706 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21056⟩⟩) 0 ⟨15719⟩ 19705

def event19707 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21056⟩⟩) (.authority (.relationPreimageSource ⟨38⟩))

def exact19708RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21056⟩⟩]⟩, (1)⟩]

theorem exact19708RawTermsValid :
    exact19708RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19708 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21056⟩⟩) exact19708RawTerms (.finite 136065468) 19707 .exactZero (none)

def event19709 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact19710RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact19710RawTermsValid :
    exact19710RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event19710 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact19710RawTerms .large 19709 .exactZero (none)

def event19711 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21057⟩⟩) 0 ⟨6⟩ 19710

def eventLeaf1216 : Array AnnotatedEvent := #[
  { event := event19456
    frameStart := 19452 },
  { event := event19457
    frameStart := 19452 },
  { event := event19458
    frameStart := 19452 },
  { event := event19459
    frameStart := 19452 },
  { event := event19460
    frameStart := 19452 },
  { event := event19461
    frameStart := 19452 },
  { event := event19462
    frameStart := 19452 },
  { event := event19463
    frameStart := 19452 },
  { event := event19464
    frameStart := 19452 },
  { event := event19465
    frameStart := 19452 },
  { event := event19466
    frameStart := 19452 },
  { event := event19467
    frameStart := 19452 },
  { event := event19468
    frameStart := 19452 },
  { event := event19469
    frameStart := 19452 },
  { event := event19470
    frameStart := 19452 },
  { event := event19471
    frameStart := 19452 }
]

def eventLeaf1217 : Array AnnotatedEvent := #[
  { event := event19472
    frameStart := 19452 },
  { event := event19473
    frameStart := 19452 },
  { event := event19474
    frameStart := 19452 },
  { event := event19475
    frameStart := 19452 },
  { event := event19476
    frameStart := 19452 },
  { event := event19477
    frameStart := 19452 },
  { event := event19478
    frameStart := 19452 },
  { event := event19479
    frameStart := 19452 },
  { event := event19480
    frameStart := 19452 },
  { event := event19481
    frameStart := 19452 },
  { event := event19482
    frameStart := 19452 },
  { event := event19483
    frameStart := 19452 },
  { event := event19484
    frameStart := 19452 },
  { event := event19485
    frameStart := 19452 },
  { event := event19486
    frameStart := 19452 },
  { event := event19487
    frameStart := 19452 }
]

def eventLeaf1218 : Array AnnotatedEvent := #[
  { event := event19488
    frameStart := 19452 },
  { event := event19489
    frameStart := 19452 },
  { event := event19490
    frameStart := 19452 },
  { event := event19491
    frameStart := 19452 },
  { event := event19492
    frameStart := 19452 },
  { event := event19493
    frameStart := 19452 },
  { event := event19494
    frameStart := 19452 },
  { event := event19495
    frameStart := 19452 },
  { event := event19496
    frameStart := 19452 },
  { event := event19497
    frameStart := 19452 },
  { event := event19498
    frameStart := 19452 },
  { event := event19499
    frameStart := 19452 },
  { event := event19500
    frameStart := 19452 },
  { event := event19501
    frameStart := 19452 },
  { event := event19502
    frameStart := 19452 },
  { event := event19503
    frameStart := 19452 }
]

def eventLeaf1219 : Array AnnotatedEvent := #[
  { event := event19504
    frameStart := 19452 },
  { event := event19505
    frameStart := 19452 },
  { event := event19506
    frameStart := 19506 },
  { event := event19507
    frameStart := 19506 },
  { event := event19508
    frameStart := 19506 },
  { event := event19509
    frameStart := 19506 },
  { event := event19510
    frameStart := 19506 },
  { event := event19511
    frameStart := 19506 },
  { event := event19512
    frameStart := 19506 },
  { event := event19513
    frameStart := 19506 },
  { event := event19514
    frameStart := 19506 },
  { event := event19515
    frameStart := 19506 },
  { event := event19516
    frameStart := 19506 },
  { event := event19517
    frameStart := 19506 },
  { event := event19518
    frameStart := 19506 },
  { event := event19519
    frameStart := 19506 }
]

def eventLeaf1220 : Array AnnotatedEvent := #[
  { event := event19520
    frameStart := 19506 },
  { event := event19521
    frameStart := 19506 },
  { event := event19522
    frameStart := 19506 },
  { event := event19523
    frameStart := 19506 },
  { event := event19524
    frameStart := 19506 },
  { event := event19525
    frameStart := 19506 },
  { event := event19526
    frameStart := 19506 },
  { event := event19527
    frameStart := 19506 },
  { event := event19528
    frameStart := 19506 },
  { event := event19529
    frameStart := 19506 },
  { event := event19530
    frameStart := 19506 },
  { event := event19531
    frameStart := 19506 },
  { event := event19532
    frameStart := 19506 },
  { event := event19533
    frameStart := 19506 },
  { event := event19534
    frameStart := 19506 },
  { event := event19535
    frameStart := 19506 }
]

def eventLeaf1221 : Array AnnotatedEvent := #[
  { event := event19536
    frameStart := 19506 },
  { event := event19537
    frameStart := 19506 },
  { event := event19538
    frameStart := 19506 },
  { event := event19539
    frameStart := 19506 },
  { event := event19540
    frameStart := 19506 },
  { event := event19541
    frameStart := 19506 },
  { event := event19542
    frameStart := 19506 },
  { event := event19543
    frameStart := 19506 },
  { event := event19544
    frameStart := 19506 },
  { event := event19545
    frameStart := 19506 },
  { event := event19546
    frameStart := 19506 },
  { event := event19547
    frameStart := 19506 },
  { event := event19548
    frameStart := 19506 },
  { event := event19549
    frameStart := 19506 },
  { event := event19550
    frameStart := 19506 },
  { event := event19551
    frameStart := 19506 }
]

def eventLeaf1222 : Array AnnotatedEvent := #[
  { event := event19552
    frameStart := 19506 },
  { event := event19553
    frameStart := 19506 },
  { event := event19554
    frameStart := 19506 },
  { event := event19555
    frameStart := 19506 },
  { event := event19556
    frameStart := 19506 },
  { event := event19557
    frameStart := 19506 },
  { event := event19558
    frameStart := 19506 },
  { event := event19559
    frameStart := 19506 },
  { event := event19560
    frameStart := 19506 },
  { event := event19561
    frameStart := 19506 },
  { event := event19562
    frameStart := 19506 },
  { event := event19563
    frameStart := 19506 },
  { event := event19564
    frameStart := 19506 },
  { event := event19565
    frameStart := 19506 },
  { event := event19566
    frameStart := 19506 },
  { event := event19567
    frameStart := 19506 }
]

def eventLeaf1223 : Array AnnotatedEvent := #[
  { event := event19568
    frameStart := 19506 },
  { event := event19569
    frameStart := 19506 },
  { event := event19570
    frameStart := 19506 },
  { event := event19571
    frameStart := 19506 },
  { event := event19572
    frameStart := 19506 },
  { event := event19573
    frameStart := 19506 },
  { event := event19574
    frameStart := 19506 },
  { event := event19575
    frameStart := 19506 },
  { event := event19576
    frameStart := 19506 },
  { event := event19577
    frameStart := 19506 },
  { event := event19578
    frameStart := 19506 },
  { event := event19579
    frameStart := 19506 },
  { event := event19580
    frameStart := 19506 },
  { event := event19581
    frameStart := 19506 },
  { event := event19582
    frameStart := 19506 },
  { event := event19583
    frameStart := 19506 }
]

def eventLeaf1224 : Array AnnotatedEvent := #[
  { event := event19584
    frameStart := 19506 },
  { event := event19585
    frameStart := 19506 },
  { event := event19586
    frameStart := 19506 },
  { event := event19587
    frameStart := 19506 },
  { event := event19588
    frameStart := 19506 },
  { event := event19589
    frameStart := 19506 },
  { event := event19590
    frameStart := 19506 },
  { event := event19591
    frameStart := 19506 },
  { event := event19592
    frameStart := 19506 },
  { event := event19593
    frameStart := 19506 },
  { event := event19594
    frameStart := 19506 },
  { event := event19595
    frameStart := 19506 },
  { event := event19596
    frameStart := 19506 },
  { event := event19597
    frameStart := 19506 },
  { event := event19598
    frameStart := 19506 },
  { event := event19599
    frameStart := 19506 }
]

def eventLeaf1225 : Array AnnotatedEvent := #[
  { event := event19600
    frameStart := 19506 },
  { event := event19601
    frameStart := 19506 },
  { event := event19602
    frameStart := 19506 },
  { event := event19603
    frameStart := 19506 },
  { event := event19604
    frameStart := 19506 },
  { event := event19605
    frameStart := 19506 },
  { event := event19606
    frameStart := 19506 },
  { event := event19607
    frameStart := 19506 },
  { event := event19608
    frameStart := 19506 },
  { event := event19609
    frameStart := 19506 },
  { event := event19610
    frameStart := 0 },
  { event := event19611
    frameStart := 0 },
  { event := event19612
    frameStart := 0 },
  { event := event19613
    frameStart := 0 },
  { event := event19614
    frameStart := 0 },
  { event := event19615
    frameStart := 0 }
]

def eventLeaf1226 : Array AnnotatedEvent := #[
  { event := event19616
    frameStart := 0 },
  { event := event19617
    frameStart := 0 },
  { event := event19618
    frameStart := 0 },
  { event := event19619
    frameStart := 0 },
  { event := event19620
    frameStart := 0 },
  { event := event19621
    frameStart := 0 },
  { event := event19622
    frameStart := 0 },
  { event := event19623
    frameStart := 0 },
  { event := event19624
    frameStart := 0 },
  { event := event19625
    frameStart := 0 },
  { event := event19626
    frameStart := 0 },
  { event := event19627
    frameStart := 0 },
  { event := event19628
    frameStart := 0 },
  { event := event19629
    frameStart := 0 },
  { event := event19630
    frameStart := 0 },
  { event := event19631
    frameStart := 0 }
]

def eventLeaf1227 : Array AnnotatedEvent := #[
  { event := event19632
    frameStart := 0 },
  { event := event19633
    frameStart := 0 },
  { event := event19634
    frameStart := 0 },
  { event := event19635
    frameStart := 0 },
  { event := event19636
    frameStart := 0 },
  { event := event19637
    frameStart := 0 },
  { event := event19638
    frameStart := 0 },
  { event := event19639
    frameStart := 0 },
  { event := event19640
    frameStart := 0 },
  { event := event19641
    frameStart := 0 },
  { event := event19642
    frameStart := 0 },
  { event := event19643
    frameStart := 0 },
  { event := event19644
    frameStart := 0 },
  { event := event19645
    frameStart := 0 },
  { event := event19646
    frameStart := 0 },
  { event := event19647
    frameStart := 0 }
]

def eventLeaf1228 : Array AnnotatedEvent := #[
  { event := event19648
    frameStart := 0 },
  { event := event19649
    frameStart := 0 },
  { event := event19650
    frameStart := 0 },
  { event := event19651
    frameStart := 0 },
  { event := event19652
    frameStart := 0 },
  { event := event19653
    frameStart := 0 },
  { event := event19654
    frameStart := 0 },
  { event := event19655
    frameStart := 0 },
  { event := event19656
    frameStart := 0 },
  { event := event19657
    frameStart := 0 },
  { event := event19658
    frameStart := 0 },
  { event := event19659
    frameStart := 0 },
  { event := event19660
    frameStart := 0 },
  { event := event19661
    frameStart := 0 },
  { event := event19662
    frameStart := 0 },
  { event := event19663
    frameStart := 0 }
]

def eventLeaf1229 : Array AnnotatedEvent := #[
  { event := event19664
    frameStart := 19664 },
  { event := event19665
    frameStart := 19664 },
  { event := event19666
    frameStart := 19664 },
  { event := event19667
    frameStart := 19664 },
  { event := event19668
    frameStart := 19664 },
  { event := event19669
    frameStart := 19664 },
  { event := event19670
    frameStart := 19664 },
  { event := event19671
    frameStart := 19664 },
  { event := event19672
    frameStart := 19664 },
  { event := event19673
    frameStart := 19664 },
  { event := event19674
    frameStart := 19664 },
  { event := event19675
    frameStart := 19664 },
  { event := event19676
    frameStart := 19664 },
  { event := event19677
    frameStart := 19664 },
  { event := event19678
    frameStart := 19664 },
  { event := event19679
    frameStart := 19664 }
]

def eventLeaf1230 : Array AnnotatedEvent := #[
  { event := event19680
    frameStart := 19664 },
  { event := event19681
    frameStart := 19664 },
  { event := event19682
    frameStart := 19664 },
  { event := event19683
    frameStart := 19664 },
  { event := event19684
    frameStart := 19664 },
  { event := event19685
    frameStart := 19664 },
  { event := event19686
    frameStart := 19664 },
  { event := event19687
    frameStart := 19664 },
  { event := event19688
    frameStart := 19664 },
  { event := event19689
    frameStart := 19664 },
  { event := event19690
    frameStart := 19664 },
  { event := event19691
    frameStart := 19664 },
  { event := event19692
    frameStart := 19664 },
  { event := event19693
    frameStart := 19664 },
  { event := event19694
    frameStart := 19664 },
  { event := event19695
    frameStart := 19664 }
]

def eventLeaf1231 : Array AnnotatedEvent := #[
  { event := event19696
    frameStart := 19664 },
  { event := event19697
    frameStart := 19664 },
  { event := event19698
    frameStart := 19664 },
  { event := event19699
    frameStart := 19664 },
  { event := event19700
    frameStart := 19664 },
  { event := event19701
    frameStart := 19664 },
  { event := event19702
    frameStart := 19664 },
  { event := event19703
    frameStart := 19664 },
  { event := event19704
    frameStart := 19664 },
  { event := event19705
    frameStart := 19664 },
  { event := event19706
    frameStart := 19664 },
  { event := event19707
    frameStart := 19664 },
  { event := event19708
    frameStart := 19664 },
  { event := event19709
    frameStart := 19664 },
  { event := event19710
    frameStart := 19664 },
  { event := event19711
    frameStart := 19664 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events076
