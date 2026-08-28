import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events279

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event71424 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21108⟩⟩) (.authority (.relationPreimageSource ⟨39⟩))

def exact71425RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21108⟩⟩]⟩, (1)⟩]

theorem exact71425RawTermsValid :
    exact71425RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71425 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21108⟩⟩) exact71425RawTerms (.finite 136065468) 71424 .exactZero (none)

def event71426 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact71427RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact71427RawTermsValid :
    exact71427RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71427 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact71427RawTerms .large 71426 .exactZero (none)

def event71428 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21109⟩⟩) 0 ⟨6⟩ 71427

def event71429 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21109⟩⟩) 1 ⟨21108⟩ 71425

def event71430 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21109⟩⟩) (.product (.predecessor 0 71428 .coefficient) (.predecessor 1 71429 .coefficient) (⟨false, false, none, none, none⟩))

def event71431 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21109⟩⟩, .operator (⟨71427, 0⟩, ⟨71425, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21108⟩⟩]⟩, (1)⟩)

def exact71432RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21108⟩⟩]⟩, (1)⟩]

theorem exact71432RawTermsValid :
    exact71432RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71432 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21109⟩⟩) exact71432RawTerms .large 71430 .exactZero (none)

def event71433 : Event := .preFoldPolynomial 71432 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21108⟩⟩]⟩, (1)⟩] .exactZero none

def exact71434RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21108⟩⟩]⟩, (1)⟩]

def event71434 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21109⟩⟩) 71433 exact71434RawTerms .large 71430 .exactZero (none)

def event71435 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27424⟩⟩)

def event71436 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event71437 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event71438 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event71439 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event71440 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event71441 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event71442 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event71443 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event71444 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 71443

def event71445 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 71441

def event71446 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 71444 .coefficient) (.value (.predecessor 1 71445 .coefficient)))

def event71447 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event71448 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 71447

def event71449 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 71439

def event71450 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 71448 .coefficient, .predecessor 1 71449 .coefficient])

def event71451 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event71452 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 71451

def event71453 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 71437

def event71454 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 71453 .coefficient))

def event71455 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event71456 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11297⟩⟩) 0 ⟨5530⟩ 71455

def event71457 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11297⟩⟩) (.authority (.programFamilyFact))

def exact71458RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩], []⟩, (1)⟩]

theorem exact71458RawTermsValid :
    exact71458RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71458 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11297⟩⟩) exact71458RawTerms (.finite 12) 71457 .exactZero (none)

def event71459 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13764⟩⟩) 0 ⟨5530⟩ 71455

def event71460 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13764⟩⟩) (.authority (.programFamilyFact))

def exact71461RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩, (1)⟩]

theorem exact71461RawTermsValid :
    exact71461RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71461 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13764⟩⟩) exact71461RawTerms (.finite 12) 71460 .exactZero (none)

def event71462 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13765⟩⟩) 0 ⟨13764⟩ 71461

def event71463 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13765⟩⟩) 1 ⟨11297⟩ 71458

def event71464 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13765⟩⟩) (.product (.predecessor 0 71462 .coefficient) (.predecessor 1 71463 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event71465 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13765⟩⟩, .operator (⟨71461, 0⟩, ⟨71458, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩, (1)⟩)

def exact71466RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11297⟩⟩, ⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩, (1)⟩]

theorem exact71466RawTermsValid :
    exact71466RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71466 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13765⟩⟩) exact71466RawTerms (.finite 144) 71464 .exactZero (none)

def event71467 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13766⟩⟩) 0 ⟨13765⟩ 71466

def event71468 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13766⟩⟩) (.identity (.predecessor 0 71467 .coefficient))

def event71469 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13766⟩⟩) (.finite 144)

def event71470 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15698⟩⟩) 0 ⟨13766⟩ 71469

def event71471 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15698⟩⟩) (.authority (.programFamilyFact))

def exact71472RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], []⟩, (1)⟩]

theorem exact71472RawTermsValid :
    exact71472RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71472 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15698⟩⟩) exact71472RawTerms (.finite 12) 71471 .exactZero (none)

def event71473 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15699⟩⟩) 0 ⟨15698⟩ 71472

def event71474 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15699⟩⟩) (.identity (.predecessor 0 71473 .coefficient))

def event71475 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15699⟩⟩) (.finite 12)

def event71476 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24031⟩⟩) 0 ⟨15699⟩ 71475

def event71477 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24031⟩⟩) (.authority (.programFamilyFact))

def event71478 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24031⟩⟩) (.finite 3720)

def event71479 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event71480 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24033⟩⟩) 0 ⟨6689⟩ 71479

def event71481 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24033⟩⟩) 1 ⟨24031⟩ 71478

def event71482 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24033⟩⟩) (.authority (.operator))

def exact71483RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24033⟩⟩]⟩, (1)⟩]

theorem exact71483RawTermsValid :
    exact71483RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71483 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24033⟩⟩) exact71483RawTerms .large 71482 .exactZero (none)

def event71484 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27419⟩⟩) 0 ⟨24033⟩ 71483

def event71485 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27419⟩⟩) (.authority (.operator))

def exact71486RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩, (1)⟩]

theorem exact71486RawTermsValid :
    exact71486RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71486 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27419⟩⟩) exact71486RawTerms (.finite 8192) 71485 .exactZero (none)

def event71487 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event71488 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event71489 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15773⟩⟩) 0 ⟨15699⟩ 71475

def event71490 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15773⟩⟩) 1 ⟨110⟩ 71488

def event71491 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15773⟩⟩) (.sum [.predecessor 0 71489 .coefficient, .predecessor 1 71490 .coefficient])

def event71492 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15773⟩⟩) (.finite 12)

def event71493 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15774⟩⟩) 0 ⟨15773⟩ 71492

def event71494 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15774⟩⟩) (.identity (.predecessor 0 71493 .coefficient))

def exact71495RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], []⟩, (1)⟩]

theorem exact71495RawTermsValid :
    exact71495RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71495 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15774⟩⟩) exact71495RawTerms (.finite 12) 71494 .exactZero (none)

def event71496 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact71497RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact71497RawTermsValid :
    exact71497RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71497 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact71497RawTerms .large 71496 .exactZero (none)

def event71498 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15775⟩⟩) 0 ⟨6544⟩ 71497

def event71499 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15775⟩⟩) 1 ⟨15774⟩ 71495

def event71500 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15775⟩⟩) (.product (.predecessor 0 71498 .coefficient) (.predecessor 1 71499 .coefficient) (⟨false, false, none, none, none⟩))

def event71501 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15775⟩⟩, .operator (⟨71497, 0⟩, ⟨71495, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact71502RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact71502RawTermsValid :
    exact71502RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71502 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15775⟩⟩) exact71502RawTerms .large 71500 .exactZero (none)

def event71503 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6695⟩⟩) 0 ⟨6689⟩ 71479

def event71504 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6695⟩⟩) (.authority (.operator))

def exact71505RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩]

theorem exact71505RawTermsValid :
    exact71505RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71505 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6695⟩⟩) exact71505RawTerms .large 71504 .exactZero (none)

def event71506 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15776⟩⟩) 0 ⟨6695⟩ 71505

def event71507 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15776⟩⟩) 1 ⟨15775⟩ 71502

def event71508 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15776⟩⟩) (.sum [.predecessor 0 71506 .coefficient, .predecessor 1 71507 .coefficient])

def exact71509RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71509RawTermsValid :
    exact71509RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71509 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15776⟩⟩) exact71509RawTerms .large 71508 .exactZero (none)

def event71510 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27420⟩⟩) 0 ⟨15776⟩ 71509

def event71511 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27420⟩⟩) 1 ⟨27419⟩ 71486

def event71512 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27420⟩⟩) (.product (.predecessor 0 71510 .coefficient) (.predecessor 1 71511 .coefficient) (⟨false, false, none, none, none⟩))

def event71513 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27420⟩⟩, .operator (⟨71509, 0⟩, ⟨71486, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩, (1)⟩)

def event71514 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27420⟩⟩, .operator (⟨71509, 1⟩, ⟨71486, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩, (-1)⟩)

def event71515 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27420⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27419⟩⟩) ⟨24033⟩ 71483)

def event71516 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27420⟩⟩, .relation 71515 0, ⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24033⟩⟩]⟩, (-1)⟩)

def exact71517RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24033⟩⟩]⟩, (-1)⟩]

theorem exact71517RawTermsValid :
    exact71517RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71517 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27420⟩⟩) exact71517RawTerms .large 71512 .exactZero (none)

def event71518 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15745⟩⟩) 0 ⟨15699⟩ 71475

def event71519 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15745⟩⟩) (.authority (.programFamilyFact))

def exact71520RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], []⟩, (1)⟩]

theorem exact71520RawTermsValid :
    exact71520RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71520 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15745⟩⟩) exact71520RawTerms (.finite 59) 71519 .exactZero (none)

def event71521 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15746⟩⟩) 0 ⟨6544⟩ 71497

def event71522 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15746⟩⟩) 1 ⟨15745⟩ 71520

def event71523 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15746⟩⟩) (.product (.predecessor 0 71521 .coefficient) (.predecessor 1 71522 .coefficient) (⟨false, true, none, none, some 1⟩))

def event71524 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15746⟩⟩, .operator (⟨71497, 0⟩, ⟨71520, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact71525RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact71525RawTermsValid :
    exact71525RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71525 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15746⟩⟩) exact71525RawTerms .large 71523 .exactZero (none)

def event71526 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6719⟩⟩) 0 ⟨6689⟩ 71479

def event71527 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6719⟩⟩) (.authority (.operator))

def exact71528RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩]

theorem exact71528RawTermsValid :
    exact71528RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71528 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6719⟩⟩) exact71528RawTerms .large 71527 .exactZero (none)

def event71529 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15747⟩⟩) 0 ⟨6719⟩ 71528

def event71530 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15747⟩⟩) 1 ⟨15746⟩ 71525

def event71531 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15747⟩⟩) (.sum [.predecessor 0 71529 .coefficient, .predecessor 1 71530 .coefficient])

def exact71532RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71532RawTermsValid :
    exact71532RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71532 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15747⟩⟩) exact71532RawTerms .large 71531 .exactZero (none)

def event71533 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27424⟩⟩) 0 ⟨15747⟩ 71532

def event71534 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27424⟩⟩) 1 ⟨27420⟩ 71517

def event71535 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27424⟩⟩) (.sum [.predecessor 0 71533 .coefficient, .predecessor 1 71534 .coefficient])

def exact71536RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24033⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71536RawTermsValid :
    exact71536RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71536 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27424⟩⟩) exact71536RawTerms .large 71535 .exactZero (none)

def event71537 : Event := .preFoldPolynomial 71536 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24033⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact71538RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24033⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event71538 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27424⟩⟩) 71537 exact71538RawTerms .large 71535 .exactZero (none)

def event71539 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15699⟩⟩) ⟨⟨132⟩, ⟨39⟩, ⟨109⟩⟩ ⟨71381, 71539⟩

def event71540 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21111⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21108⟩⟩]⟩) (1) 0 2 (.universal 71539 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21108⟩⟩]⟩) (none) 71538)

def event71541 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21111⟩⟩, .relation 71540 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩)

def event71542 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21111⟩⟩, .relation 71540 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩, (-1)⟩)

def event71543 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21111⟩⟩, .relation 71540 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24033⟩⟩]⟩, (1)⟩)

def event71544 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21111⟩⟩, .relation 71540 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15745⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact71545RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24033⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15745⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71545RawTermsValid :
    exact71545RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71545 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21111⟩⟩) exact71545RawTerms .large 71377 (.finite 1811303510016) (some (71379))

def event71546 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27422⟩⟩) 0 ⟨21111⟩ 71545

def event71547 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27422⟩⟩) 1 ⟨27421⟩ 71367

def event71548 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27422⟩⟩) (.sum [.predecessor 0 71546 .coefficient, .predecessor 1 71547 .coefficient])

def event71549 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27422⟩⟩, .operator (⟨71545, 0⟩, ⟨71367, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩, (1)⟩)

def event71550 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27422⟩⟩, .operator (⟨71545, 2⟩, ⟨71367, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15698⟩⟩], [⟨.program ⟨214⟩, ⟨24033⟩⟩]⟩, (-1)⟩)

def event71551 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27422⟩⟩) (.sum [.result 71545 .summary, .result 71367 .summary])

def exact71552RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15745⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71552RawTermsValid :
    exact71552RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71552 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27422⟩⟩) exact71552RawTerms .large 71548 (.finite 1292001236604524572672) (some (71551))

def event71553 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23968⟩⟩) 0 ⟨15580⟩ 3402

def event71554 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23968⟩⟩) (.authority (.programFamilyFact))

def event71555 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23968⟩⟩) (.finite 3720)

def event71556 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23970⟩⟩) 0 ⟨6689⟩ 5477

def event71557 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23970⟩⟩) 1 ⟨23968⟩ 71555

def event71558 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23970⟩⟩) (.authority (.operator))

def exact71559RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23970⟩⟩]⟩, (1)⟩]

theorem exact71559RawTermsValid :
    exact71559RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71559 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23970⟩⟩) exact71559RawTerms .large 71558 .exactZero (none)

def event71560 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27202⟩⟩) 0 ⟨23970⟩ 71559

def event71561 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27202⟩⟩) (.authority (.operator))

def exact71562RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27202⟩⟩]⟩, (1)⟩]

theorem exact71562RawTermsValid :
    exact71562RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71562 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27202⟩⟩) exact71562RawTerms (.finite 8192) 71561 .exactZero (none)

def event71563 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23455⟩⟩) 0 ⟨13549⟩ 3396

def event71564 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23455⟩⟩) (.authority (.programFamilyFact))

def event71565 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23455⟩⟩) (.finite 3720)

def event71566 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23456⟩⟩) 0 ⟨6689⟩ 5477

def event71567 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23456⟩⟩) 1 ⟨23455⟩ 71565

def event71568 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23456⟩⟩) (.authority (.operator))

def exact71569RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23456⟩⟩]⟩, (1)⟩]

theorem exact71569RawTermsValid :
    exact71569RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71569 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23456⟩⟩) exact71569RawTerms .large 71568 .exactZero (none)

def event71570 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25830⟩⟩) 0 ⟨23456⟩ 71569

def event71571 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25830⟩⟩) (.authority (.operator))

def exact71572RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩, (1)⟩]

theorem exact71572RawTermsValid :
    exact71572RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71572 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25830⟩⟩) exact71572RawTerms (.finite 8192) 71571 .exactZero (none)

def event71573 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11214⟩⟩) 0 ⟨11213⟩ 3385

def event71574 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11214⟩⟩) 1 ⟨6566⟩ 65295

def event71575 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11214⟩⟩) (.tensor (.predecessor 0 71573 .coefficient) (.predecessor 1 71574 .coefficient) true false)

def event71576 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11214⟩⟩, .operator (⟨3385, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11213⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact71577RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11213⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact71577RawTermsValid :
    exact71577RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71577 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11214⟩⟩) exact71577RawTerms .large 71575 .exactZero (none)

def event71578 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7194⟩⟩) 0 ⟨5533⟩ 65165

def event71579 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7194⟩⟩) 1 ⟨6776⟩ 12985

def event71580 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7194⟩⟩) (.product (.predecessor 0 71578 .coefficient) (.predecessor 1 71579 .coefficient) (⟨false, false, none, none, none⟩))

def event71581 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7194⟩⟩, .operator (⟨65165, 0⟩, ⟨12985, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩)

def exact71582RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩]

theorem exact71582RawTermsValid :
    exact71582RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71582 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7194⟩⟩) exact71582RawTerms .large 71580 .exactZero (none)

def event71583 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11215⟩⟩) 0 ⟨7194⟩ 71582

def event71584 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11215⟩⟩) 1 ⟨11214⟩ 71577

def event71585 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11215⟩⟩) (.sum [.predecessor 0 71583 .coefficient, .predecessor 1 71584 .coefficient])

def exact71586RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11213⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71586RawTermsValid :
    exact71586RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71586 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11215⟩⟩) exact71586RawTerms .large 71585 .exactZero (none)

def event71587 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11216⟩⟩) 0 ⟨11215⟩ 71586

def event71588 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11216⟩⟩) 1 ⟨90⟩ 12977

def event71589 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11216⟩⟩) (.sum [.predecessor 0 71587 .coefficient, .predecessor 1 71588 .coefficient])

def event71590 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11216⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨90⟩⟩]⟩) [⟨.result 12977 .coefficient, false, none⟩])

def event71591 : Event := .survivorFold (1) 71590

def exact71592RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11213⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71592RawTermsValid :
    exact71592RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71592 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11216⟩⟩) exact71592RawTerms .large 71589 (.finite 26) (some (71590))

def event71593 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13550⟩⟩) 0 ⟨11216⟩ 71592

def event71594 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13550⟩⟩) 1 ⟨13547⟩ 3388

def event71595 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13550⟩⟩) (.product (.predecessor 0 71593 .coefficient) (.predecessor 1 71594 .coefficient) (⟨false, true, none, none, some 1⟩))

def event71596 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13550⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨13547⟩⟩], []⟩) [⟨.result 3388 .coefficient, true, some 1⟩])

def event71597 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13550⟩⟩) (.product (.result 71592 .summary) (.transfer 71596) (⟨false, false, none, none, none⟩))

def event71598 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13550⟩⟩, .operator (⟨71592, 1⟩, ⟨3388, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event71599 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13550⟩⟩, .operator (⟨71592, 0⟩, ⟨3388, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩)

def exact71600RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩]

theorem exact71600RawTermsValid :
    exact71600RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71600 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13550⟩⟩) exact71600RawTerms .large 71595 (.finite 8320) (some (71597))

def event71601 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13551⟩⟩) 0 ⟨13547⟩ 3388

def event71602 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13551⟩⟩) 1 ⟨6566⟩ 65295

def event71603 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13551⟩⟩) (.tensor (.predecessor 0 71601 .coefficient) (.predecessor 1 71602 .coefficient) true false)

def event71604 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13551⟩⟩, .operator (⟨3388, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact71605RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact71605RawTermsValid :
    exact71605RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71605 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13551⟩⟩) exact71605RawTerms .large 71603 .exactZero (none)

def event71606 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7211⟩⟩) 0 ⟨5533⟩ 65165

def event71607 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7211⟩⟩) 1 ⟨6793⟩ 13026

def event71608 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7211⟩⟩) (.product (.predecessor 0 71606 .coefficient) (.predecessor 1 71607 .coefficient) (⟨false, false, none, none, none⟩))

def event71609 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7211⟩⟩, .operator (⟨65165, 0⟩, ⟨13026, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩]⟩, (1)⟩)

def exact71610RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩]⟩, (1)⟩]

theorem exact71610RawTermsValid :
    exact71610RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71610 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7211⟩⟩) exact71610RawTerms .large 71608 .exactZero (none)

def event71611 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13552⟩⟩) 0 ⟨7211⟩ 71610

def event71612 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13552⟩⟩) 1 ⟨13551⟩ 71605

def event71613 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13552⟩⟩) (.sum [.predecessor 0 71611 .coefficient, .predecessor 1 71612 .coefficient])

def exact71614RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71614RawTermsValid :
    exact71614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71614 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13552⟩⟩) exact71614RawTerms .large 71613 .exactZero (none)

def event71615 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13553⟩⟩) 0 ⟨13552⟩ 71614

def event71616 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13553⟩⟩) 1 ⟨107⟩ 13018

def event71617 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13553⟩⟩) (.sum [.predecessor 0 71615 .coefficient, .predecessor 1 71616 .coefficient])

def event71618 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13553⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨107⟩⟩]⟩) [⟨.result 13018 .coefficient, false, none⟩])

def event71619 : Event := .survivorFold (1) 71618

def exact71620RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71620RawTermsValid :
    exact71620RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71620 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13553⟩⟩) exact71620RawTerms .large 71617 (.finite 26) (some (71618))

def event71621 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13554⟩⟩) 0 ⟨13553⟩ 71620

def event71622 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13554⟩⟩) 1 ⟨7844⟩ 13015

def event71623 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13554⟩⟩) (.product (.predecessor 0 71621 .coefficient) (.predecessor 1 71622 .coefficient) (⟨false, false, none, none, none⟩))

def event71624 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13554⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩) [⟨.result 13011 .coefficient, false, none⟩])

def event71625 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13554⟩⟩) (.product (.result 71620 .summary) (.transfer 71624) (⟨false, false, none, none, none⟩))

def event71626 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13554⟩⟩, .operator (⟨71620, 1⟩, ⟨13015, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (-1)⟩)

def event71627 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨13554⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7843⟩⟩) ⟨6776⟩ 12985)

def event71628 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13554⟩⟩, .relation 71627 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (-1)⟩)

def event71629 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13554⟩⟩, .operator (⟨71620, 0⟩, ⟨13015, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩)

def exact71630RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (-1)⟩]

theorem exact71630RawTermsValid :
    exact71630RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71630 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13554⟩⟩) exact71630RawTerms .large 71623 (.finite 95420416) (some (71625))

def event71631 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13555⟩⟩) 0 ⟨13554⟩ 71630

def event71632 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13555⟩⟩) 1 ⟨13550⟩ 71600

def event71633 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13555⟩⟩) (.sum [.predecessor 0 71631 .coefficient, .predecessor 1 71632 .coefficient])

def event71634 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13555⟩⟩, .operator (⟨71630, 1⟩, ⟨71600, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩)

def event71635 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13555⟩⟩) (.sum [.result 71630 .summary, .result 71600 .summary])

def exact71636RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact71636RawTermsValid :
    exact71636RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71636 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13555⟩⟩) exact71636RawTerms .large 71633 (.finite 95428736) (some (71635))

def event71637 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25831⟩⟩) 0 ⟨13555⟩ 71636

def event71638 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25831⟩⟩) 1 ⟨25830⟩ 71572

def event71639 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25831⟩⟩) (.product (.predecessor 0 71637 .coefficient) (.predecessor 1 71638 .coefficient) (⟨false, false, none, none, none⟩))

def event71640 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25831⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩) [⟨.result 71572 .coefficient, false, none⟩])

def event71641 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25831⟩⟩) (.product (.result 71636 .summary) (.transfer 71640) (⟨false, false, none, none, none⟩))

def event71642 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25831⟩⟩, .operator (⟨71636, 1⟩, ⟨71572, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩, (-1)⟩)

def event71643 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25831⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25830⟩⟩) ⟨23456⟩ 71569)

def event71644 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25831⟩⟩, .relation 71643 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨23456⟩⟩]⟩, (-1)⟩)

def event71645 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25831⟩⟩, .operator (⟨71636, 0⟩, ⟨71572, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩, (1)⟩)

def exact71646RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨23456⟩⟩]⟩, (-1)⟩]

theorem exact71646RawTermsValid :
    exact71646RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71646 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25831⟩⟩) exact71646RawTerms .large 71639 (.finite 350224987979776) (some (71641))

def event71647 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19308⟩⟩) 0 ⟨13549⟩ 3396

def event71648 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19308⟩⟩) (.authority (.relationPreimageSource ⟨12⟩))

def exact71649RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19308⟩⟩]⟩, (1)⟩]

theorem exact71649RawTermsValid :
    exact71649RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71649 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19308⟩⟩) exact71649RawTerms (.finite 136065468) 71648 .exactZero (none)

def event71650 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19310⟩⟩) 0 ⟨19308⟩ 71649

def event71651 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19310⟩⟩) 1 ⟨2348⟩ 4

def event71652 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19310⟩⟩) (.scale (.predecessor 0 71650 .coefficient) (.value (.predecessor 1 71651 .coefficient)))

def exact71653RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19308⟩⟩]⟩, (1)⟩]

theorem exact71653RawTermsValid :
    exact71653RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event71653 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19310⟩⟩) exact71653RawTerms (.finite 136065468) 71652 .exactZero (none)

def event71654 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19311⟩⟩) 0 ⟨5535⟩ 65387

def event71655 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19311⟩⟩) 1 ⟨19310⟩ 71653

def event71656 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19311⟩⟩) (.product (.predecessor 0 71654 .coefficient) (.predecessor 1 71655 .coefficient) (⟨false, false, none, none, none⟩))

def event71657 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19311⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19308⟩⟩]⟩) [⟨.result 71649 .coefficient, false, none⟩])

def event71658 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19311⟩⟩) (.product (.result 65387 .summary) (.transfer 71657) (⟨false, false, none, none, none⟩))

def event71659 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19311⟩⟩, .operator (⟨65387, 0⟩, ⟨71653, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19308⟩⟩]⟩, (1)⟩)

def event71660 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19309⟩⟩)

def event71661 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event71662 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event71663 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event71664 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event71665 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event71666 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event71667 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event71668 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event71669 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 71668

def event71670 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 71666

def event71671 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 71669 .coefficient) (.value (.predecessor 1 71670 .coefficient)))

def event71672 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event71673 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 71672

def event71674 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 71664

def event71675 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 71673 .coefficient, .predecessor 1 71674 .coefficient])

def event71676 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event71677 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 71676

def event71678 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 71662

def event71679 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 71678 .coefficient))

def eventLeaf4464 : Array AnnotatedEvent := #[
  { event := event71424
    frameStart := 71381 },
  { event := event71425
    frameStart := 71381 },
  { event := event71426
    frameStart := 71381 },
  { event := event71427
    frameStart := 71381 },
  { event := event71428
    frameStart := 71381 },
  { event := event71429
    frameStart := 71381 },
  { event := event71430
    frameStart := 71381 },
  { event := event71431
    frameStart := 71381 },
  { event := event71432
    frameStart := 71381 },
  { event := event71433
    frameStart := 71381 },
  { event := event71434
    frameStart := 71381 },
  { event := event71435
    frameStart := 71435 },
  { event := event71436
    frameStart := 71435 },
  { event := event71437
    frameStart := 71435 },
  { event := event71438
    frameStart := 71435 },
  { event := event71439
    frameStart := 71435 }
]

def eventLeaf4465 : Array AnnotatedEvent := #[
  { event := event71440
    frameStart := 71435 },
  { event := event71441
    frameStart := 71435 },
  { event := event71442
    frameStart := 71435 },
  { event := event71443
    frameStart := 71435 },
  { event := event71444
    frameStart := 71435 },
  { event := event71445
    frameStart := 71435 },
  { event := event71446
    frameStart := 71435 },
  { event := event71447
    frameStart := 71435 },
  { event := event71448
    frameStart := 71435 },
  { event := event71449
    frameStart := 71435 },
  { event := event71450
    frameStart := 71435 },
  { event := event71451
    frameStart := 71435 },
  { event := event71452
    frameStart := 71435 },
  { event := event71453
    frameStart := 71435 },
  { event := event71454
    frameStart := 71435 },
  { event := event71455
    frameStart := 71435 }
]

def eventLeaf4466 : Array AnnotatedEvent := #[
  { event := event71456
    frameStart := 71435 },
  { event := event71457
    frameStart := 71435 },
  { event := event71458
    frameStart := 71435 },
  { event := event71459
    frameStart := 71435 },
  { event := event71460
    frameStart := 71435 },
  { event := event71461
    frameStart := 71435 },
  { event := event71462
    frameStart := 71435 },
  { event := event71463
    frameStart := 71435 },
  { event := event71464
    frameStart := 71435 },
  { event := event71465
    frameStart := 71435 },
  { event := event71466
    frameStart := 71435 },
  { event := event71467
    frameStart := 71435 },
  { event := event71468
    frameStart := 71435 },
  { event := event71469
    frameStart := 71435 },
  { event := event71470
    frameStart := 71435 },
  { event := event71471
    frameStart := 71435 }
]

def eventLeaf4467 : Array AnnotatedEvent := #[
  { event := event71472
    frameStart := 71435 },
  { event := event71473
    frameStart := 71435 },
  { event := event71474
    frameStart := 71435 },
  { event := event71475
    frameStart := 71435 },
  { event := event71476
    frameStart := 71435 },
  { event := event71477
    frameStart := 71435 },
  { event := event71478
    frameStart := 71435 },
  { event := event71479
    frameStart := 71435 },
  { event := event71480
    frameStart := 71435 },
  { event := event71481
    frameStart := 71435 },
  { event := event71482
    frameStart := 71435 },
  { event := event71483
    frameStart := 71435 },
  { event := event71484
    frameStart := 71435 },
  { event := event71485
    frameStart := 71435 },
  { event := event71486
    frameStart := 71435 },
  { event := event71487
    frameStart := 71435 }
]

def eventLeaf4468 : Array AnnotatedEvent := #[
  { event := event71488
    frameStart := 71435 },
  { event := event71489
    frameStart := 71435 },
  { event := event71490
    frameStart := 71435 },
  { event := event71491
    frameStart := 71435 },
  { event := event71492
    frameStart := 71435 },
  { event := event71493
    frameStart := 71435 },
  { event := event71494
    frameStart := 71435 },
  { event := event71495
    frameStart := 71435 },
  { event := event71496
    frameStart := 71435 },
  { event := event71497
    frameStart := 71435 },
  { event := event71498
    frameStart := 71435 },
  { event := event71499
    frameStart := 71435 },
  { event := event71500
    frameStart := 71435 },
  { event := event71501
    frameStart := 71435 },
  { event := event71502
    frameStart := 71435 },
  { event := event71503
    frameStart := 71435 }
]

def eventLeaf4469 : Array AnnotatedEvent := #[
  { event := event71504
    frameStart := 71435 },
  { event := event71505
    frameStart := 71435 },
  { event := event71506
    frameStart := 71435 },
  { event := event71507
    frameStart := 71435 },
  { event := event71508
    frameStart := 71435 },
  { event := event71509
    frameStart := 71435 },
  { event := event71510
    frameStart := 71435 },
  { event := event71511
    frameStart := 71435 },
  { event := event71512
    frameStart := 71435 },
  { event := event71513
    frameStart := 71435 },
  { event := event71514
    frameStart := 71435 },
  { event := event71515
    frameStart := 71435 },
  { event := event71516
    frameStart := 71435 },
  { event := event71517
    frameStart := 71435 },
  { event := event71518
    frameStart := 71435 },
  { event := event71519
    frameStart := 71435 }
]

def eventLeaf4470 : Array AnnotatedEvent := #[
  { event := event71520
    frameStart := 71435 },
  { event := event71521
    frameStart := 71435 },
  { event := event71522
    frameStart := 71435 },
  { event := event71523
    frameStart := 71435 },
  { event := event71524
    frameStart := 71435 },
  { event := event71525
    frameStart := 71435 },
  { event := event71526
    frameStart := 71435 },
  { event := event71527
    frameStart := 71435 },
  { event := event71528
    frameStart := 71435 },
  { event := event71529
    frameStart := 71435 },
  { event := event71530
    frameStart := 71435 },
  { event := event71531
    frameStart := 71435 },
  { event := event71532
    frameStart := 71435 },
  { event := event71533
    frameStart := 71435 },
  { event := event71534
    frameStart := 71435 },
  { event := event71535
    frameStart := 71435 }
]

def eventLeaf4471 : Array AnnotatedEvent := #[
  { event := event71536
    frameStart := 71435 },
  { event := event71537
    frameStart := 71435 },
  { event := event71538
    frameStart := 71435 },
  { event := event71539
    frameStart := 0 },
  { event := event71540
    frameStart := 0 },
  { event := event71541
    frameStart := 0 },
  { event := event71542
    frameStart := 0 },
  { event := event71543
    frameStart := 0 },
  { event := event71544
    frameStart := 0 },
  { event := event71545
    frameStart := 0 },
  { event := event71546
    frameStart := 0 },
  { event := event71547
    frameStart := 0 },
  { event := event71548
    frameStart := 0 },
  { event := event71549
    frameStart := 0 },
  { event := event71550
    frameStart := 0 },
  { event := event71551
    frameStart := 0 }
]

def eventLeaf4472 : Array AnnotatedEvent := #[
  { event := event71552
    frameStart := 0 },
  { event := event71553
    frameStart := 0 },
  { event := event71554
    frameStart := 0 },
  { event := event71555
    frameStart := 0 },
  { event := event71556
    frameStart := 0 },
  { event := event71557
    frameStart := 0 },
  { event := event71558
    frameStart := 0 },
  { event := event71559
    frameStart := 0 },
  { event := event71560
    frameStart := 0 },
  { event := event71561
    frameStart := 0 },
  { event := event71562
    frameStart := 0 },
  { event := event71563
    frameStart := 0 },
  { event := event71564
    frameStart := 0 },
  { event := event71565
    frameStart := 0 },
  { event := event71566
    frameStart := 0 },
  { event := event71567
    frameStart := 0 }
]

def eventLeaf4473 : Array AnnotatedEvent := #[
  { event := event71568
    frameStart := 0 },
  { event := event71569
    frameStart := 0 },
  { event := event71570
    frameStart := 0 },
  { event := event71571
    frameStart := 0 },
  { event := event71572
    frameStart := 0 },
  { event := event71573
    frameStart := 0 },
  { event := event71574
    frameStart := 0 },
  { event := event71575
    frameStart := 0 },
  { event := event71576
    frameStart := 0 },
  { event := event71577
    frameStart := 0 },
  { event := event71578
    frameStart := 0 },
  { event := event71579
    frameStart := 0 },
  { event := event71580
    frameStart := 0 },
  { event := event71581
    frameStart := 0 },
  { event := event71582
    frameStart := 0 },
  { event := event71583
    frameStart := 0 }
]

def eventLeaf4474 : Array AnnotatedEvent := #[
  { event := event71584
    frameStart := 0 },
  { event := event71585
    frameStart := 0 },
  { event := event71586
    frameStart := 0 },
  { event := event71587
    frameStart := 0 },
  { event := event71588
    frameStart := 0 },
  { event := event71589
    frameStart := 0 },
  { event := event71590
    frameStart := 0 },
  { event := event71591
    frameStart := 0 },
  { event := event71592
    frameStart := 0 },
  { event := event71593
    frameStart := 0 },
  { event := event71594
    frameStart := 0 },
  { event := event71595
    frameStart := 0 },
  { event := event71596
    frameStart := 0 },
  { event := event71597
    frameStart := 0 },
  { event := event71598
    frameStart := 0 },
  { event := event71599
    frameStart := 0 }
]

def eventLeaf4475 : Array AnnotatedEvent := #[
  { event := event71600
    frameStart := 0 },
  { event := event71601
    frameStart := 0 },
  { event := event71602
    frameStart := 0 },
  { event := event71603
    frameStart := 0 },
  { event := event71604
    frameStart := 0 },
  { event := event71605
    frameStart := 0 },
  { event := event71606
    frameStart := 0 },
  { event := event71607
    frameStart := 0 },
  { event := event71608
    frameStart := 0 },
  { event := event71609
    frameStart := 0 },
  { event := event71610
    frameStart := 0 },
  { event := event71611
    frameStart := 0 },
  { event := event71612
    frameStart := 0 },
  { event := event71613
    frameStart := 0 },
  { event := event71614
    frameStart := 0 },
  { event := event71615
    frameStart := 0 }
]

def eventLeaf4476 : Array AnnotatedEvent := #[
  { event := event71616
    frameStart := 0 },
  { event := event71617
    frameStart := 0 },
  { event := event71618
    frameStart := 0 },
  { event := event71619
    frameStart := 0 },
  { event := event71620
    frameStart := 0 },
  { event := event71621
    frameStart := 0 },
  { event := event71622
    frameStart := 0 },
  { event := event71623
    frameStart := 0 },
  { event := event71624
    frameStart := 0 },
  { event := event71625
    frameStart := 0 },
  { event := event71626
    frameStart := 0 },
  { event := event71627
    frameStart := 0 },
  { event := event71628
    frameStart := 0 },
  { event := event71629
    frameStart := 0 },
  { event := event71630
    frameStart := 0 },
  { event := event71631
    frameStart := 0 }
]

def eventLeaf4477 : Array AnnotatedEvent := #[
  { event := event71632
    frameStart := 0 },
  { event := event71633
    frameStart := 0 },
  { event := event71634
    frameStart := 0 },
  { event := event71635
    frameStart := 0 },
  { event := event71636
    frameStart := 0 },
  { event := event71637
    frameStart := 0 },
  { event := event71638
    frameStart := 0 },
  { event := event71639
    frameStart := 0 },
  { event := event71640
    frameStart := 0 },
  { event := event71641
    frameStart := 0 },
  { event := event71642
    frameStart := 0 },
  { event := event71643
    frameStart := 0 },
  { event := event71644
    frameStart := 0 },
  { event := event71645
    frameStart := 0 },
  { event := event71646
    frameStart := 0 },
  { event := event71647
    frameStart := 0 }
]

def eventLeaf4478 : Array AnnotatedEvent := #[
  { event := event71648
    frameStart := 0 },
  { event := event71649
    frameStart := 0 },
  { event := event71650
    frameStart := 0 },
  { event := event71651
    frameStart := 0 },
  { event := event71652
    frameStart := 0 },
  { event := event71653
    frameStart := 0 },
  { event := event71654
    frameStart := 0 },
  { event := event71655
    frameStart := 0 },
  { event := event71656
    frameStart := 0 },
  { event := event71657
    frameStart := 0 },
  { event := event71658
    frameStart := 0 },
  { event := event71659
    frameStart := 0 },
  { event := event71660
    frameStart := 71660 },
  { event := event71661
    frameStart := 71660 },
  { event := event71662
    frameStart := 71660 },
  { event := event71663
    frameStart := 71660 }
]

def eventLeaf4479 : Array AnnotatedEvent := #[
  { event := event71664
    frameStart := 71660 },
  { event := event71665
    frameStart := 71660 },
  { event := event71666
    frameStart := 71660 },
  { event := event71667
    frameStart := 71660 },
  { event := event71668
    frameStart := 71660 },
  { event := event71669
    frameStart := 71660 },
  { event := event71670
    frameStart := 71660 },
  { event := event71671
    frameStart := 71660 },
  { event := event71672
    frameStart := 71660 },
  { event := event71673
    frameStart := 71660 },
  { event := event71674
    frameStart := 71660 },
  { event := event71675
    frameStart := 71660 },
  { event := event71676
    frameStart := 71660 },
  { event := event71677
    frameStart := 71660 },
  { event := event71678
    frameStart := 71660 },
  { event := event71679
    frameStart := 71660 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events279
