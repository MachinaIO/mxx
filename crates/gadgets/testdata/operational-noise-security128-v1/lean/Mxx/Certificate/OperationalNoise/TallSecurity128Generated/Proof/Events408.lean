import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events408

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event104448 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16692⟩⟩) (.authority (.relationPreimageSource ⟨56⟩))

def exact104449RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16692⟩⟩]⟩, (1)⟩]

theorem exact104449RawTermsValid :
    exact104449RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104449 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16692⟩⟩) exact104449RawTerms (.finite 5647228698) 104448 .exactZero (none)

def event104450 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16694⟩⟩) 0 ⟨16692⟩ 104449

def event104451 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16694⟩⟩) 1 ⟨2370⟩ 4

def event104452 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16694⟩⟩) (.scale (.predecessor 0 104450 .coefficient) (.value (.predecessor 1 104451 .coefficient)))

def exact104453RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16692⟩⟩]⟩, (1)⟩]

theorem exact104453RawTermsValid :
    exact104453RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104453 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16694⟩⟩) exact104453RawTerms (.finite 5647228698) 104452 .exactZero (none)

def event104454 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16695⟩⟩) 0 ⟨9944⟩ 90620

def event104455 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16695⟩⟩) 1 ⟨16694⟩ 104453

def event104456 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16695⟩⟩) (.product (.predecessor 0 104454 .coefficient) (.predecessor 1 104455 .coefficient) (⟨false, false, none, none, none⟩))

def event104457 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16695⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨16692⟩⟩]⟩) [⟨.result 104449 .coefficient, false, none⟩])

def event104458 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16695⟩⟩) (.product (.result 90620 .summary) (.transfer 104457) (⟨false, false, none, none, none⟩))

def event104459 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16695⟩⟩, .operator (⟨90620, 0⟩, ⟨104453, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16692⟩⟩]⟩, (1)⟩)

def event104460 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨16693⟩⟩)

def event104461 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event104462 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event104463 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event104464 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event104465 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event104466 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event104467 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event104468 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event104469 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 104468

def event104470 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 104466

def event104471 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 104469 .coefficient) (.value (.predecessor 1 104470 .coefficient)))

def event104472 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event104473 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 104472

def event104474 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 104464

def event104475 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 104473 .coefficient, .predecessor 1 104474 .coefficient])

def event104476 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event104477 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 104476

def event104478 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 104462

def event104479 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 104478 .coefficient))

def event104480 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event104481 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15594⟩⟩) 0 ⟨9901⟩ 104480

def event104482 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15594⟩⟩) (.authority (.programFamilyFact))

def exact104483RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15594⟩⟩], []⟩, (1)⟩]

theorem exact104483RawTermsValid :
    exact104483RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104483 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15594⟩⟩) exact104483RawTerms (.finite 2) 104482 .exactZero (none)

def event104484 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12456⟩⟩) 0 ⟨9901⟩ 104480

def event104485 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12456⟩⟩) (.authority (.programFamilyFact))

def exact104486RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12456⟩⟩], []⟩, (1)⟩]

theorem exact104486RawTermsValid :
    exact104486RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104486 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12456⟩⟩) exact104486RawTerms (.finite 2) 104485 .exactZero (none)

def event104487 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15595⟩⟩) 0 ⟨12456⟩ 104486

def event104488 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15595⟩⟩) 1 ⟨15594⟩ 104483

def event104489 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15595⟩⟩) (.product (.predecessor 0 104487 .coefficient) (.predecessor 1 104488 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event104490 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15595⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], []⟩) [⟨.result 104486 .coefficient, true, some 1⟩, ⟨.result 104483 .coefficient, true, some 1⟩])

def event104491 : Event := .survivorFold (1) 104490

def exact104492RawTerms : List Term := []

theorem exact104492RawTermsValid :
    exact104492RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104492 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15595⟩⟩) exact104492RawTerms (.finite 4) 104489 (.finite 4) (some (104490))

def event104493 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15596⟩⟩) 0 ⟨15595⟩ 104492

def event104494 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15596⟩⟩) (.identity (.predecessor 0 104493 .coefficient))

def event104495 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15596⟩⟩) (.finite 4)

def event104496 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15828⟩⟩) 0 ⟨15596⟩ 104495

def event104497 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15828⟩⟩) (.authority (.programFamilyFact))

def exact104498RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], []⟩, (1)⟩]

theorem exact104498RawTermsValid :
    exact104498RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104498 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15828⟩⟩) exact104498RawTerms (.finite 2) 104497 .exactZero (none)

def event104499 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15829⟩⟩) 0 ⟨15828⟩ 104498

def event104500 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15829⟩⟩) (.identity (.predecessor 0 104499 .coefficient))

def event104501 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15829⟩⟩) (.finite 2)

def event104502 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16692⟩⟩) 0 ⟨15829⟩ 104501

def event104503 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16692⟩⟩) (.authority (.relationPreimageSource ⟨56⟩))

def exact104504RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16692⟩⟩]⟩, (1)⟩]

theorem exact104504RawTermsValid :
    exact104504RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104504 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16692⟩⟩) exact104504RawTerms (.finite 5647228698) 104503 .exactZero (none)

def event104505 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact104506RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact104506RawTermsValid :
    exact104506RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104506 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact104506RawTerms .large 104505 .exactZero (none)

def event104507 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16693⟩⟩) 0 ⟨35⟩ 104506

def event104508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16693⟩⟩) 1 ⟨16692⟩ 104504

def event104509 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16693⟩⟩) (.product (.predecessor 0 104507 .coefficient) (.predecessor 1 104508 .coefficient) (⟨false, false, none, none, none⟩))

def event104510 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16693⟩⟩, .operator (⟨104506, 0⟩, ⟨104504, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16692⟩⟩]⟩, (1)⟩)

def exact104511RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16692⟩⟩]⟩, (1)⟩]

theorem exact104511RawTermsValid :
    exact104511RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104511 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16693⟩⟩) exact104511RawTerms .large 104509 .exactZero (none)

def event104512 : Event := .preFoldPolynomial 104511 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16692⟩⟩]⟩, (1)⟩] .exactZero none

def exact104513RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16692⟩⟩]⟩, (1)⟩]

def event104513 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨16693⟩⟩) 104512 exact104513RawTerms .large 104509 .exactZero (none)

def event104514 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨17900⟩⟩)

def event104515 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event104516 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event104517 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event104518 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event104519 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event104520 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event104521 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event104522 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event104523 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 104522

def event104524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 104520

def event104525 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 104523 .coefficient) (.value (.predecessor 1 104524 .coefficient)))

def event104526 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event104527 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 104526

def event104528 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 104518

def event104529 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 104527 .coefficient, .predecessor 1 104528 .coefficient])

def event104530 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event104531 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 104530

def event104532 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 104516

def event104533 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 104532 .coefficient))

def event104534 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event104535 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15594⟩⟩) 0 ⟨9901⟩ 104534

def event104536 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15594⟩⟩) (.authority (.programFamilyFact))

def exact104537RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15594⟩⟩], []⟩, (1)⟩]

theorem exact104537RawTermsValid :
    exact104537RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104537 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15594⟩⟩) exact104537RawTerms (.finite 2) 104536 .exactZero (none)

def event104538 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12456⟩⟩) 0 ⟨9901⟩ 104534

def event104539 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12456⟩⟩) (.authority (.programFamilyFact))

def exact104540RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12456⟩⟩], []⟩, (1)⟩]

theorem exact104540RawTermsValid :
    exact104540RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104540 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12456⟩⟩) exact104540RawTerms (.finite 2) 104539 .exactZero (none)

def event104541 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15595⟩⟩) 0 ⟨12456⟩ 104540

def event104542 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15595⟩⟩) 1 ⟨15594⟩ 104537

def event104543 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15595⟩⟩) (.product (.predecessor 0 104541 .coefficient) (.predecessor 1 104542 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event104544 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15595⟩⟩, .operator (⟨104540, 0⟩, ⟨104537, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], []⟩, (1)⟩)

def exact104545RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], []⟩, (1)⟩]

theorem exact104545RawTermsValid :
    exact104545RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104545 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15595⟩⟩) exact104545RawTerms (.finite 4) 104543 .exactZero (none)

def event104546 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15596⟩⟩) 0 ⟨15595⟩ 104545

def event104547 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15596⟩⟩) (.identity (.predecessor 0 104546 .coefficient))

def event104548 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15596⟩⟩) (.finite 4)

def event104549 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15828⟩⟩) 0 ⟨15596⟩ 104548

def event104550 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15828⟩⟩) (.authority (.programFamilyFact))

def exact104551RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], []⟩, (1)⟩]

theorem exact104551RawTermsValid :
    exact104551RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104551 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15828⟩⟩) exact104551RawTerms (.finite 2) 104550 .exactZero (none)

def event104552 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15829⟩⟩) 0 ⟨15828⟩ 104551

def event104553 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15829⟩⟩) (.identity (.predecessor 0 104552 .coefficient))

def event104554 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15829⟩⟩) (.finite 2)

def event104555 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17044⟩⟩) 0 ⟨15829⟩ 104554

def event104556 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17044⟩⟩) (.authority (.programFamilyFact))

def event104557 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17044⟩⟩) (.finite 3720)

def event104558 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event104559 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17045⟩⟩) 0 ⟨7177⟩ 104558

def event104560 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17045⟩⟩) 1 ⟨17044⟩ 104557

def event104561 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17045⟩⟩) (.authority (.operator))

def exact104562RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17045⟩⟩]⟩, (1)⟩]

theorem exact104562RawTermsValid :
    exact104562RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104562 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17045⟩⟩) exact104562RawTerms .large 104561 .exactZero (none)

def event104563 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17894⟩⟩) 0 ⟨17045⟩ 104562

def event104564 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17894⟩⟩) (.authority (.operator))

def exact104565RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17894⟩⟩]⟩, (1)⟩]

theorem exact104565RawTermsValid :
    exact104565RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104565 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17894⟩⟩) exact104565RawTerms (.finite 8192) 104564 .exactZero (none)

def event104566 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event104567 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event104568 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17226⟩⟩) 0 ⟨15829⟩ 104554

def event104569 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17226⟩⟩) 1 ⟨136⟩ 104567

def event104570 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17226⟩⟩) (.sum [.predecessor 0 104568 .coefficient, .predecessor 1 104569 .coefficient])

def event104571 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17226⟩⟩) (.finite 2)

def event104572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17227⟩⟩) 0 ⟨17226⟩ 104571

def event104573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17227⟩⟩) (.identity (.predecessor 0 104572 .coefficient))

def exact104574RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], []⟩, (1)⟩]

theorem exact104574RawTermsValid :
    exact104574RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104574 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17227⟩⟩) exact104574RawTerms (.finite 2) 104573 .exactZero (none)

def event104575 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact104576RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact104576RawTermsValid :
    exact104576RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104576 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact104576RawTerms .large 104575 .exactZero (none)

def event104577 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17228⟩⟩) 0 ⟨6908⟩ 104576

def event104578 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17228⟩⟩) 1 ⟨17227⟩ 104574

def event104579 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17228⟩⟩) (.product (.predecessor 0 104577 .coefficient) (.predecessor 1 104578 .coefficient) (⟨false, false, none, none, none⟩))

def event104580 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17228⟩⟩, .operator (⟨104576, 0⟩, ⟨104574, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact104581RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact104581RawTermsValid :
    exact104581RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104581 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17228⟩⟩) exact104581RawTerms .large 104579 .exactZero (none)

def event104582 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 104558

def event104583 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact104584RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact104584RawTermsValid :
    exact104584RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104584 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact104584RawTerms .large 104583 .exactZero (none)

def event104585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17229⟩⟩) 0 ⟨7179⟩ 104584

def event104586 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17229⟩⟩) 1 ⟨17228⟩ 104581

def event104587 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17229⟩⟩) (.sum [.predecessor 0 104585 .coefficient, .predecessor 1 104586 .coefficient])

def exact104588RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact104588RawTermsValid :
    exact104588RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104588 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17229⟩⟩) exact104588RawTerms .large 104587 .exactZero (none)

def event104589 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17895⟩⟩) 0 ⟨17229⟩ 104588

def event104590 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17895⟩⟩) 1 ⟨17894⟩ 104565

def event104591 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17895⟩⟩) (.product (.predecessor 0 104589 .coefficient) (.predecessor 1 104590 .coefficient) (⟨false, false, none, none, none⟩))

def event104592 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17895⟩⟩, .operator (⟨104588, 0⟩, ⟨104565, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17894⟩⟩]⟩, (1)⟩)

def event104593 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17895⟩⟩, .operator (⟨104588, 1⟩, ⟨104565, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17894⟩⟩]⟩, (-1)⟩)

def event104594 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17895⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17894⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17894⟩⟩) ⟨17045⟩ 104562)

def event104595 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17895⟩⟩, .relation 104594 0, ⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨17045⟩⟩]⟩, (-1)⟩)

def exact104596RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17894⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨17045⟩⟩]⟩, (-1)⟩]

theorem exact104596RawTermsValid :
    exact104596RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104596 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17895⟩⟩) exact104596RawTerms .large 104591 .exactZero (none)

def event104597 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16110⟩⟩) 0 ⟨15829⟩ 104554

def event104598 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16110⟩⟩) (.authority (.programFamilyFact))

def exact104599RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16110⟩⟩], []⟩, (1)⟩]

theorem exact104599RawTermsValid :
    exact104599RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104599 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16110⟩⟩) exact104599RawTerms (.finite 2) 104598 .exactZero (none)

def event104600 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16113⟩⟩) 0 ⟨6908⟩ 104576

def event104601 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16113⟩⟩) 1 ⟨16110⟩ 104599

def event104602 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16113⟩⟩) (.product (.predecessor 0 104600 .coefficient) (.predecessor 1 104601 .coefficient) (⟨false, true, none, none, some 1⟩))

def event104603 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16113⟩⟩, .operator (⟨104576, 0⟩, ⟨104599, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact104604RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact104604RawTermsValid :
    exact104604RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104604 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16113⟩⟩) exact104604RawTerms .large 104602 .exactZero (none)

def event104605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7197⟩⟩) 0 ⟨7177⟩ 104558

def event104606 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7197⟩⟩) (.authority (.operator))

def exact104607RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩]

theorem exact104607RawTermsValid :
    exact104607RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104607 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7197⟩⟩) exact104607RawTerms .large 104606 .exactZero (none)

def event104608 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16114⟩⟩) 0 ⟨7197⟩ 104607

def event104609 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16114⟩⟩) 1 ⟨16113⟩ 104604

def event104610 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16114⟩⟩) (.sum [.predecessor 0 104608 .coefficient, .predecessor 1 104609 .coefficient])

def exact104611RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact104611RawTermsValid :
    exact104611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104611 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16114⟩⟩) exact104611RawTerms .large 104610 .exactZero (none)

def event104612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17900⟩⟩) 0 ⟨16114⟩ 104611

def event104613 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17900⟩⟩) 1 ⟨17895⟩ 104596

def event104614 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17900⟩⟩) (.sum [.predecessor 0 104612 .coefficient, .predecessor 1 104613 .coefficient])

def exact104615RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17894⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨17045⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact104615RawTermsValid :
    exact104615RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104615 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17900⟩⟩) exact104615RawTerms .large 104614 .exactZero (none)

def event104616 : Event := .preFoldPolynomial 104615 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17894⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨17045⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact104617RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17894⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨17045⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event104617 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17900⟩⟩) 104616 exact104617RawTerms .large 104614 .exactZero (none)

def event104618 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15829⟩⟩) ⟨⟨76⟩, ⟨56⟩, ⟨135⟩⟩ ⟨104460, 104618⟩

def event104619 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16695⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16692⟩⟩]⟩) (1) 0 2 (.universal 104618 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16692⟩⟩]⟩) (none) 104617)

def event104620 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16695⟩⟩, .relation 104619 1, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩)

def event104621 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16695⟩⟩, .relation 104619 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17894⟩⟩]⟩, (-1)⟩)

def event104622 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16695⟩⟩, .relation 104619 2, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨17045⟩⟩]⟩, (1)⟩)

def event104623 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16695⟩⟩, .relation 104619 3, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact104624RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17894⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨17045⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact104624RawTermsValid :
    exact104624RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104624 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16695⟩⟩) exact104624RawTerms .large 104456 (.finite 202072841853861888) (some (104458))

def event104625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17897⟩⟩) 0 ⟨16695⟩ 104624

def event104626 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17897⟩⟩) 1 ⟨17896⟩ 104446

def event104627 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17897⟩⟩) (.sum [.predecessor 0 104625 .coefficient, .predecessor 1 104626 .coefficient])

def event104628 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17897⟩⟩, .operator (⟨104624, 0⟩, ⟨104446, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17894⟩⟩]⟩, (1)⟩)

def event104629 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17897⟩⟩, .operator (⟨104624, 2⟩, ⟨104446, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨17045⟩⟩]⟩, (-1)⟩)

def event104630 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17897⟩⟩) (.sum [.result 104624 .summary, .result 104446 .summary])

def exact104631RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact104631RawTermsValid :
    exact104631RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104631 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17897⟩⟩) exact104631RawTerms .large 104627 (.finite 32188807212483706889510625476608) (some (104630))

def event104632 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17898⟩⟩) 0 ⟨17897⟩ 104631

def event104633 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17898⟩⟩) 1 ⟨7172⟩ 15882

def event104634 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17898⟩⟩) (.product (.predecessor 0 104632 .coefficient) (.predecessor 1 104633 .coefficient) (⟨false, false, none, none, none⟩))

def event104635 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17898⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) [⟨.result 15878 .coefficient, false, none⟩])

def event104636 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17898⟩⟩) (.product (.result 104631 .summary) (.transfer 104635) (⟨false, false, none, none, none⟩))

def event104637 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17898⟩⟩, .operator (⟨104631, 0⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩)

def event104638 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17898⟩⟩, .operator (⟨104631, 1⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩)

def event104639 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17898⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7171⟩⟩) ⟨7051⟩ 15875)

def event104640 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17898⟩⟩, .relation 104639 0, ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact104641RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩]

theorem exact104641RawTermsValid :
    exact104641RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104641 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17898⟩⟩) exact104641RawTerms .large 104634 (.finite 345624685687166110058245054666339432529920) (some (104636))

def event104642 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9948⟩⟩) 0 ⟨6727⟩ 723

def event104643 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9948⟩⟩) 1 ⟨9904⟩ 90528

def event104644 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9948⟩⟩) (.tensor (.predecessor 0 104642 .coefficient) (.predecessor 1 104643 .coefficient) true false)

def event104645 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9948⟩⟩, .operator (⟨723, 0⟩, ⟨90528, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact104646RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact104646RawTermsValid :
    exact104646RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104646 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9948⟩⟩) exact104646RawTerms .large 104644 .exactZero (none)

def event104647 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9926⟩⟩) 0 ⟨9903⟩ 90398

def event104648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9926⟩⟩) 1 ⟨7292⟩ 15896

def event104649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9926⟩⟩) (.product (.predecessor 0 104647 .coefficient) (.predecessor 1 104648 .coefficient) (⟨false, false, none, none, none⟩))

def event104650 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9926⟩⟩, .operator (⟨90398, 0⟩, ⟨15896, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩)

def exact104651RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact104651RawTermsValid :
    exact104651RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104651 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9926⟩⟩) exact104651RawTerms .large 104649 .exactZero (none)

def event104652 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9949⟩⟩) 0 ⟨9926⟩ 104651

def event104653 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9949⟩⟩) 1 ⟨9948⟩ 104646

def event104654 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9949⟩⟩) (.sum [.predecessor 0 104652 .coefficient, .predecessor 1 104653 .coefficient])

def exact104655RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact104655RawTermsValid :
    exact104655RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104655 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9949⟩⟩) exact104655RawTerms .large 104654 .exactZero (none)

def event104656 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9950⟩⟩) 0 ⟨9949⟩ 104655

def event104657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9950⟩⟩) 1 ⟨118⟩ 31516

def event104658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9950⟩⟩) (.sum [.predecessor 0 104656 .coefficient, .predecessor 1 104657 .coefficient])

def event104659 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9950⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) [⟨.result 31516 .coefficient, false, none⟩])

def event104660 : Event := .survivorFold (1) 104659

def exact104661RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact104661RawTermsValid :
    exact104661RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104661 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9950⟩⟩) exact104661RawTerms .large 104658 (.finite 26) (some (104659))

def event104662 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9951⟩⟩) 0 ⟨9950⟩ 104661

def event104663 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9951⟩⟩) 1 ⟨9950⟩ 104661

def event104664 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9951⟩⟩) (.sum [.predecessor 0 104662 .coefficient, .predecessor 1 104663 .coefficient])

def event104665 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9951⟩⟩, .operator (⟨104661, 0⟩, ⟨104661, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event104666 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9951⟩⟩, .operator (⟨104661, 1⟩, ⟨104661, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (-1)⟩)

def event104667 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9951⟩⟩) (.sum [.result 104661 .summary, .result 104661 .summary])

def exact104668RawTerms : List Term := []

theorem exact104668RawTermsValid :
    exact104668RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104668 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9951⟩⟩) exact104668RawTerms .large 104664 (.finite 52) (some (104667))

def event104669 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17899⟩⟩) 0 ⟨9951⟩ 104668

def event104670 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17899⟩⟩) 1 ⟨17898⟩ 104641

def event104671 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17899⟩⟩) (.sum [.predecessor 0 104669 .coefficient, .predecessor 1 104670 .coefficient])

def event104672 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17899⟩⟩) (.sum [.result 104668 .summary, .result 104641 .summary])

def exact104673RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩]

theorem exact104673RawTermsValid :
    exact104673RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104673 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17899⟩⟩) exact104673RawTerms .large 104671 (.finite 345624685687166110058245054666339432529972) (some (104672))

def event104674 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20805⟩⟩) 0 ⟨17899⟩ 104673

def event104675 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20805⟩⟩) 1 ⟨20804⟩ 104429

def event104676 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20805⟩⟩) (.sum [.predecessor 0 104674 .coefficient, .predecessor 1 104675 .coefficient])

def event104677 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20805⟩⟩) (.sum [.result 104673 .summary, .result 104429 .summary])

def exact104678RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩]

theorem exact104678RawTermsValid :
    exact104678RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104678 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20805⟩⟩) exact104678RawTerms .large 104676 (.finite 691250426059631610003352154589745737891892) (some (104677))

def event104679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24025⟩⟩) 0 ⟨20805⟩ 104678

def event104680 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24025⟩⟩) 1 ⟨24024⟩ 104217

def event104681 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24025⟩⟩) (.sum [.predecessor 0 104679 .coefficient, .predecessor 1 104680 .coefficient])

def event104682 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24025⟩⟩) (.sum [.result 104678 .summary, .result 104217 .summary])

def exact104683RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩]

theorem exact104683RawTermsValid :
    exact104683RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104683 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24025⟩⟩) exact104683RawTerms .large 104681 (.finite 1036877221117396499835321299770218916085812) (some (104682))

def event104684 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34045⟩⟩) 0 ⟨24025⟩ 104683

def event104685 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34045⟩⟩) 1 ⟨34044⟩ 104005

def event104686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34045⟩⟩) (.sum [.predecessor 0 104684 .coefficient, .predecessor 1 104685 .coefficient])

def event104687 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34045⟩⟩) (.sum [.result 104683 .summary, .result 104005 .summary])

def exact104688RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨32196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩]

theorem exact104688RawTermsValid :
    exact104688RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104688 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34045⟩⟩) exact104688RawTerms .large 104686 (.finite 1382506125545760169441014535464825839943732) (some (104687))

def event104689 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53105⟩⟩) 0 ⟨34045⟩ 104688

def event104690 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53105⟩⟩) 1 ⟨53104⟩ 103793

def event104691 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53105⟩⟩) (.sum [.predecessor 0 104689 .coefficient, .predecessor 1 104690 .coefficient])

def event104692 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53105⟩⟩) (.sum [.result 104688 .summary, .result 103793 .summary])

def exact104693RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨32196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩]

theorem exact104693RawTermsValid :
    exact104693RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104693 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53105⟩⟩) exact104693RawTerms .large 104691 (.finite 1728139248715321398594155952187700255129652) (some (104692))

def event104694 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56085⟩⟩) 0 ⟨53105⟩ 104693

def event104695 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56085⟩⟩) 1 ⟨56084⟩ 103581

def event104696 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56085⟩⟩) (.sum [.predecessor 0 104694 .coefficient, .predecessor 1 104695 .coefficient])

def event104697 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56085⟩⟩) (.sum [.result 104693 .summary, .result 103581 .summary])

def exact104698RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨54240⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨32196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩]

theorem exact104698RawTermsValid :
    exact104698RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104698 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56085⟩⟩) exact104698RawTerms .large 104696 (.finite 2073774481255481407521021459424708415979572) (some (104697))

def event104699 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59065⟩⟩) 0 ⟨56085⟩ 104698

def event104700 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59065⟩⟩) 1 ⟨59064⟩ 103369

def event104701 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59065⟩⟩) (.sum [.predecessor 0 104699 .coefficient, .predecessor 1 104700 .coefficient])

def event104702 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59065⟩⟩) (.sum [.result 104698 .summary, .result 103369 .summary])

def exact104703RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨57220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨54240⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨32196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩]

theorem exact104703RawTermsValid :
    exact104703RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event104703 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59065⟩⟩) exact104703RawTerms .large 104701 (.finite 2419413932536838975995335147689984068157492) (some (104702))

def eventLeaf6528 : Array AnnotatedEvent := #[
  { event := event104448
    frameStart := 0 },
  { event := event104449
    frameStart := 0 },
  { event := event104450
    frameStart := 0 },
  { event := event104451
    frameStart := 0 },
  { event := event104452
    frameStart := 0 },
  { event := event104453
    frameStart := 0 },
  { event := event104454
    frameStart := 0 },
  { event := event104455
    frameStart := 0 },
  { event := event104456
    frameStart := 0 },
  { event := event104457
    frameStart := 0 },
  { event := event104458
    frameStart := 0 },
  { event := event104459
    frameStart := 0 },
  { event := event104460
    frameStart := 104460 },
  { event := event104461
    frameStart := 104460 },
  { event := event104462
    frameStart := 104460 },
  { event := event104463
    frameStart := 104460 }
]

def eventLeaf6529 : Array AnnotatedEvent := #[
  { event := event104464
    frameStart := 104460 },
  { event := event104465
    frameStart := 104460 },
  { event := event104466
    frameStart := 104460 },
  { event := event104467
    frameStart := 104460 },
  { event := event104468
    frameStart := 104460 },
  { event := event104469
    frameStart := 104460 },
  { event := event104470
    frameStart := 104460 },
  { event := event104471
    frameStart := 104460 },
  { event := event104472
    frameStart := 104460 },
  { event := event104473
    frameStart := 104460 },
  { event := event104474
    frameStart := 104460 },
  { event := event104475
    frameStart := 104460 },
  { event := event104476
    frameStart := 104460 },
  { event := event104477
    frameStart := 104460 },
  { event := event104478
    frameStart := 104460 },
  { event := event104479
    frameStart := 104460 }
]

def eventLeaf6530 : Array AnnotatedEvent := #[
  { event := event104480
    frameStart := 104460 },
  { event := event104481
    frameStart := 104460 },
  { event := event104482
    frameStart := 104460 },
  { event := event104483
    frameStart := 104460 },
  { event := event104484
    frameStart := 104460 },
  { event := event104485
    frameStart := 104460 },
  { event := event104486
    frameStart := 104460 },
  { event := event104487
    frameStart := 104460 },
  { event := event104488
    frameStart := 104460 },
  { event := event104489
    frameStart := 104460 },
  { event := event104490
    frameStart := 104460 },
  { event := event104491
    frameStart := 104460 },
  { event := event104492
    frameStart := 104460 },
  { event := event104493
    frameStart := 104460 },
  { event := event104494
    frameStart := 104460 },
  { event := event104495
    frameStart := 104460 }
]

def eventLeaf6531 : Array AnnotatedEvent := #[
  { event := event104496
    frameStart := 104460 },
  { event := event104497
    frameStart := 104460 },
  { event := event104498
    frameStart := 104460 },
  { event := event104499
    frameStart := 104460 },
  { event := event104500
    frameStart := 104460 },
  { event := event104501
    frameStart := 104460 },
  { event := event104502
    frameStart := 104460 },
  { event := event104503
    frameStart := 104460 },
  { event := event104504
    frameStart := 104460 },
  { event := event104505
    frameStart := 104460 },
  { event := event104506
    frameStart := 104460 },
  { event := event104507
    frameStart := 104460 },
  { event := event104508
    frameStart := 104460 },
  { event := event104509
    frameStart := 104460 },
  { event := event104510
    frameStart := 104460 },
  { event := event104511
    frameStart := 104460 }
]

def eventLeaf6532 : Array AnnotatedEvent := #[
  { event := event104512
    frameStart := 104460 },
  { event := event104513
    frameStart := 104460 },
  { event := event104514
    frameStart := 104514 },
  { event := event104515
    frameStart := 104514 },
  { event := event104516
    frameStart := 104514 },
  { event := event104517
    frameStart := 104514 },
  { event := event104518
    frameStart := 104514 },
  { event := event104519
    frameStart := 104514 },
  { event := event104520
    frameStart := 104514 },
  { event := event104521
    frameStart := 104514 },
  { event := event104522
    frameStart := 104514 },
  { event := event104523
    frameStart := 104514 },
  { event := event104524
    frameStart := 104514 },
  { event := event104525
    frameStart := 104514 },
  { event := event104526
    frameStart := 104514 },
  { event := event104527
    frameStart := 104514 }
]

def eventLeaf6533 : Array AnnotatedEvent := #[
  { event := event104528
    frameStart := 104514 },
  { event := event104529
    frameStart := 104514 },
  { event := event104530
    frameStart := 104514 },
  { event := event104531
    frameStart := 104514 },
  { event := event104532
    frameStart := 104514 },
  { event := event104533
    frameStart := 104514 },
  { event := event104534
    frameStart := 104514 },
  { event := event104535
    frameStart := 104514 },
  { event := event104536
    frameStart := 104514 },
  { event := event104537
    frameStart := 104514 },
  { event := event104538
    frameStart := 104514 },
  { event := event104539
    frameStart := 104514 },
  { event := event104540
    frameStart := 104514 },
  { event := event104541
    frameStart := 104514 },
  { event := event104542
    frameStart := 104514 },
  { event := event104543
    frameStart := 104514 }
]

def eventLeaf6534 : Array AnnotatedEvent := #[
  { event := event104544
    frameStart := 104514 },
  { event := event104545
    frameStart := 104514 },
  { event := event104546
    frameStart := 104514 },
  { event := event104547
    frameStart := 104514 },
  { event := event104548
    frameStart := 104514 },
  { event := event104549
    frameStart := 104514 },
  { event := event104550
    frameStart := 104514 },
  { event := event104551
    frameStart := 104514 },
  { event := event104552
    frameStart := 104514 },
  { event := event104553
    frameStart := 104514 },
  { event := event104554
    frameStart := 104514 },
  { event := event104555
    frameStart := 104514 },
  { event := event104556
    frameStart := 104514 },
  { event := event104557
    frameStart := 104514 },
  { event := event104558
    frameStart := 104514 },
  { event := event104559
    frameStart := 104514 }
]

def eventLeaf6535 : Array AnnotatedEvent := #[
  { event := event104560
    frameStart := 104514 },
  { event := event104561
    frameStart := 104514 },
  { event := event104562
    frameStart := 104514 },
  { event := event104563
    frameStart := 104514 },
  { event := event104564
    frameStart := 104514 },
  { event := event104565
    frameStart := 104514 },
  { event := event104566
    frameStart := 104514 },
  { event := event104567
    frameStart := 104514 },
  { event := event104568
    frameStart := 104514 },
  { event := event104569
    frameStart := 104514 },
  { event := event104570
    frameStart := 104514 },
  { event := event104571
    frameStart := 104514 },
  { event := event104572
    frameStart := 104514 },
  { event := event104573
    frameStart := 104514 },
  { event := event104574
    frameStart := 104514 },
  { event := event104575
    frameStart := 104514 }
]

def eventLeaf6536 : Array AnnotatedEvent := #[
  { event := event104576
    frameStart := 104514 },
  { event := event104577
    frameStart := 104514 },
  { event := event104578
    frameStart := 104514 },
  { event := event104579
    frameStart := 104514 },
  { event := event104580
    frameStart := 104514 },
  { event := event104581
    frameStart := 104514 },
  { event := event104582
    frameStart := 104514 },
  { event := event104583
    frameStart := 104514 },
  { event := event104584
    frameStart := 104514 },
  { event := event104585
    frameStart := 104514 },
  { event := event104586
    frameStart := 104514 },
  { event := event104587
    frameStart := 104514 },
  { event := event104588
    frameStart := 104514 },
  { event := event104589
    frameStart := 104514 },
  { event := event104590
    frameStart := 104514 },
  { event := event104591
    frameStart := 104514 }
]

def eventLeaf6537 : Array AnnotatedEvent := #[
  { event := event104592
    frameStart := 104514 },
  { event := event104593
    frameStart := 104514 },
  { event := event104594
    frameStart := 104514 },
  { event := event104595
    frameStart := 104514 },
  { event := event104596
    frameStart := 104514 },
  { event := event104597
    frameStart := 104514 },
  { event := event104598
    frameStart := 104514 },
  { event := event104599
    frameStart := 104514 },
  { event := event104600
    frameStart := 104514 },
  { event := event104601
    frameStart := 104514 },
  { event := event104602
    frameStart := 104514 },
  { event := event104603
    frameStart := 104514 },
  { event := event104604
    frameStart := 104514 },
  { event := event104605
    frameStart := 104514 },
  { event := event104606
    frameStart := 104514 },
  { event := event104607
    frameStart := 104514 }
]

def eventLeaf6538 : Array AnnotatedEvent := #[
  { event := event104608
    frameStart := 104514 },
  { event := event104609
    frameStart := 104514 },
  { event := event104610
    frameStart := 104514 },
  { event := event104611
    frameStart := 104514 },
  { event := event104612
    frameStart := 104514 },
  { event := event104613
    frameStart := 104514 },
  { event := event104614
    frameStart := 104514 },
  { event := event104615
    frameStart := 104514 },
  { event := event104616
    frameStart := 104514 },
  { event := event104617
    frameStart := 104514 },
  { event := event104618
    frameStart := 0 },
  { event := event104619
    frameStart := 0 },
  { event := event104620
    frameStart := 0 },
  { event := event104621
    frameStart := 0 },
  { event := event104622
    frameStart := 0 },
  { event := event104623
    frameStart := 0 }
]

def eventLeaf6539 : Array AnnotatedEvent := #[
  { event := event104624
    frameStart := 0 },
  { event := event104625
    frameStart := 0 },
  { event := event104626
    frameStart := 0 },
  { event := event104627
    frameStart := 0 },
  { event := event104628
    frameStart := 0 },
  { event := event104629
    frameStart := 0 },
  { event := event104630
    frameStart := 0 },
  { event := event104631
    frameStart := 0 },
  { event := event104632
    frameStart := 0 },
  { event := event104633
    frameStart := 0 },
  { event := event104634
    frameStart := 0 },
  { event := event104635
    frameStart := 0 },
  { event := event104636
    frameStart := 0 },
  { event := event104637
    frameStart := 0 },
  { event := event104638
    frameStart := 0 },
  { event := event104639
    frameStart := 0 }
]

def eventLeaf6540 : Array AnnotatedEvent := #[
  { event := event104640
    frameStart := 0 },
  { event := event104641
    frameStart := 0 },
  { event := event104642
    frameStart := 0 },
  { event := event104643
    frameStart := 0 },
  { event := event104644
    frameStart := 0 },
  { event := event104645
    frameStart := 0 },
  { event := event104646
    frameStart := 0 },
  { event := event104647
    frameStart := 0 },
  { event := event104648
    frameStart := 0 },
  { event := event104649
    frameStart := 0 },
  { event := event104650
    frameStart := 0 },
  { event := event104651
    frameStart := 0 },
  { event := event104652
    frameStart := 0 },
  { event := event104653
    frameStart := 0 },
  { event := event104654
    frameStart := 0 },
  { event := event104655
    frameStart := 0 }
]

def eventLeaf6541 : Array AnnotatedEvent := #[
  { event := event104656
    frameStart := 0 },
  { event := event104657
    frameStart := 0 },
  { event := event104658
    frameStart := 0 },
  { event := event104659
    frameStart := 0 },
  { event := event104660
    frameStart := 0 },
  { event := event104661
    frameStart := 0 },
  { event := event104662
    frameStart := 0 },
  { event := event104663
    frameStart := 0 },
  { event := event104664
    frameStart := 0 },
  { event := event104665
    frameStart := 0 },
  { event := event104666
    frameStart := 0 },
  { event := event104667
    frameStart := 0 },
  { event := event104668
    frameStart := 0 },
  { event := event104669
    frameStart := 0 },
  { event := event104670
    frameStart := 0 },
  { event := event104671
    frameStart := 0 }
]

def eventLeaf6542 : Array AnnotatedEvent := #[
  { event := event104672
    frameStart := 0 },
  { event := event104673
    frameStart := 0 },
  { event := event104674
    frameStart := 0 },
  { event := event104675
    frameStart := 0 },
  { event := event104676
    frameStart := 0 },
  { event := event104677
    frameStart := 0 },
  { event := event104678
    frameStart := 0 },
  { event := event104679
    frameStart := 0 },
  { event := event104680
    frameStart := 0 },
  { event := event104681
    frameStart := 0 },
  { event := event104682
    frameStart := 0 },
  { event := event104683
    frameStart := 0 },
  { event := event104684
    frameStart := 0 },
  { event := event104685
    frameStart := 0 },
  { event := event104686
    frameStart := 0 },
  { event := event104687
    frameStart := 0 }
]

def eventLeaf6543 : Array AnnotatedEvent := #[
  { event := event104688
    frameStart := 0 },
  { event := event104689
    frameStart := 0 },
  { event := event104690
    frameStart := 0 },
  { event := event104691
    frameStart := 0 },
  { event := event104692
    frameStart := 0 },
  { event := event104693
    frameStart := 0 },
  { event := event104694
    frameStart := 0 },
  { event := event104695
    frameStart := 0 },
  { event := event104696
    frameStart := 0 },
  { event := event104697
    frameStart := 0 },
  { event := event104698
    frameStart := 0 },
  { event := event104699
    frameStart := 0 },
  { event := event104700
    frameStart := 0 },
  { event := event104701
    frameStart := 0 },
  { event := event104702
    frameStart := 0 },
  { event := event104703
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events408
