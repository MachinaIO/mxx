import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1123

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event287488 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event287489 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 287488

def event287490 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 287474

def event287491 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 287490 .coefficient))

def event287492 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event287493 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24218⟩⟩) 0 ⟨5487⟩ 287492

def event287494 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24218⟩⟩) (.authority (.programFamilyFact))

def exact287495RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩], []⟩, (1)⟩]

theorem exact287495RawTermsValid :
    exact287495RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287495 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24218⟩⟩) exact287495RawTerms (.finite 6) 287494 .exactZero (none)

def event287496 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31323⟩⟩) 0 ⟨5487⟩ 287492

def event287497 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31323⟩⟩) (.authority (.programFamilyFact))

def exact287498RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31323⟩⟩], []⟩, (1)⟩]

theorem exact287498RawTermsValid :
    exact287498RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287498 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31323⟩⟩) exact287498RawTerms (.finite 6) 287497 .exactZero (none)

def event287499 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31324⟩⟩) 0 ⟨31323⟩ 287498

def event287500 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31324⟩⟩) 1 ⟨24218⟩ 287495

def event287501 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31324⟩⟩) (.product (.predecessor 0 287499 .coefficient) (.predecessor 1 287500 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event287502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31324⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], []⟩) [⟨.result 287498 .coefficient, true, some 1⟩, ⟨.result 287495 .coefficient, true, some 1⟩])

def event287503 : Event := .survivorFold (1) 287502

def exact287504RawTerms : List Term := []

theorem exact287504RawTermsValid :
    exact287504RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287504 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31324⟩⟩) exact287504RawTerms (.finite 36) 287501 (.finite 36) (some (287502))

def event287505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31325⟩⟩) 0 ⟨31324⟩ 287504

def event287506 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31325⟩⟩) (.identity (.predecessor 0 287505 .coefficient))

def event287507 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31325⟩⟩) (.finite 36)

def event287508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32329⟩⟩) 0 ⟨31325⟩ 287507

def event287509 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32329⟩⟩) (.authority (.relationPreimageSource ⟨39⟩))

def exact287510RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32329⟩⟩]⟩, (1)⟩]

theorem exact287510RawTermsValid :
    exact287510RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287510 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32329⟩⟩) exact287510RawTerms (.finite 5647228698) 287509 .exactZero (none)

def event287511 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact287512RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact287512RawTermsValid :
    exact287512RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287512 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact287512RawTerms .large 287511 .exactZero (none)

def event287513 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32330⟩⟩) 0 ⟨35⟩ 287512

def event287514 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32330⟩⟩) 1 ⟨32329⟩ 287510

def event287515 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32330⟩⟩) (.product (.predecessor 0 287513 .coefficient) (.predecessor 1 287514 .coefficient) (⟨false, false, none, none, none⟩))

def event287516 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32330⟩⟩, .operator (⟨287512, 0⟩, ⟨287510, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32329⟩⟩]⟩, (1)⟩)

def exact287517RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32329⟩⟩]⟩, (1)⟩]

theorem exact287517RawTermsValid :
    exact287517RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287517 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32330⟩⟩) exact287517RawTerms .large 287515 .exactZero (none)

def event287518 : Event := .preFoldPolynomial 287517 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32329⟩⟩]⟩, (1)⟩] .exactZero none

def exact287519RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32329⟩⟩]⟩, (1)⟩]

def event287519 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨32330⟩⟩) 287518 exact287519RawTerms .large 287515 .exactZero (none)

def event287520 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨33397⟩⟩)

def event287521 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event287522 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event287523 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event287524 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event287525 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event287526 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event287527 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event287528 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event287529 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 287528

def event287530 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 287526

def event287531 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 287529 .coefficient) (.value (.predecessor 1 287530 .coefficient)))

def event287532 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event287533 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 287532

def event287534 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 287524

def event287535 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 287533 .coefficient, .predecessor 1 287534 .coefficient])

def event287536 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event287537 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 287536

def event287538 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 287522

def event287539 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 287538 .coefficient))

def event287540 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event287541 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24218⟩⟩) 0 ⟨5487⟩ 287540

def event287542 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24218⟩⟩) (.authority (.programFamilyFact))

def exact287543RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩], []⟩, (1)⟩]

theorem exact287543RawTermsValid :
    exact287543RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287543 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24218⟩⟩) exact287543RawTerms (.finite 6) 287542 .exactZero (none)

def event287544 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31323⟩⟩) 0 ⟨5487⟩ 287540

def event287545 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31323⟩⟩) (.authority (.programFamilyFact))

def exact287546RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31323⟩⟩], []⟩, (1)⟩]

theorem exact287546RawTermsValid :
    exact287546RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287546 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31323⟩⟩) exact287546RawTerms (.finite 6) 287545 .exactZero (none)

def event287547 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31324⟩⟩) 0 ⟨31323⟩ 287546

def event287548 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31324⟩⟩) 1 ⟨24218⟩ 287543

def event287549 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31324⟩⟩) (.product (.predecessor 0 287547 .coefficient) (.predecessor 1 287548 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event287550 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31324⟩⟩, .operator (⟨287546, 0⟩, ⟨287543, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], []⟩, (1)⟩)

def exact287551RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], []⟩, (1)⟩]

theorem exact287551RawTermsValid :
    exact287551RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287551 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31324⟩⟩) exact287551RawTerms (.finite 36) 287549 .exactZero (none)

def event287552 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31325⟩⟩) 0 ⟨31324⟩ 287551

def event287553 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31325⟩⟩) (.identity (.predecessor 0 287552 .coefficient))

def event287554 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31325⟩⟩) (.finite 36)

def event287555 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32912⟩⟩) 0 ⟨31325⟩ 287554

def event287556 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32912⟩⟩) (.authority (.programFamilyFact))

def event287557 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨32912⟩⟩) (.finite 3720)

def event287558 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event287559 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32913⟩⟩) 0 ⟨7177⟩ 287558

def event287560 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32913⟩⟩) 1 ⟨32912⟩ 287557

def event287561 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32913⟩⟩) (.authority (.operator))

def exact287562RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32913⟩⟩]⟩, (1)⟩]

theorem exact287562RawTermsValid :
    exact287562RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287562 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32913⟩⟩) exact287562RawTerms .large 287561 .exactZero (none)

def event287563 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33393⟩⟩) 0 ⟨32913⟩ 287562

def event287564 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33393⟩⟩) (.authority (.operator))

def exact287565RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨33393⟩⟩]⟩, (1)⟩]

theorem exact287565RawTermsValid :
    exact287565RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287565 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33393⟩⟩) exact287565RawTerms (.finite 8192) 287564 .exactZero (none)

def event287566 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event287567 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event287568 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33202⟩⟩) 0 ⟨31325⟩ 287554

def event287569 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33202⟩⟩) 1 ⟨136⟩ 287567

def event287570 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33202⟩⟩) (.sum [.predecessor 0 287568 .coefficient, .predecessor 1 287569 .coefficient])

def event287571 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨33202⟩⟩) (.finite 36)

def event287572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33203⟩⟩) 0 ⟨33202⟩ 287571

def event287573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33203⟩⟩) (.identity (.predecessor 0 287572 .coefficient))

def exact287574RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], []⟩, (1)⟩]

theorem exact287574RawTermsValid :
    exact287574RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287574 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33203⟩⟩) exact287574RawTerms (.finite 36) 287573 .exactZero (none)

def event287575 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact287576RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact287576RawTermsValid :
    exact287576RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287576 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact287576RawTerms .large 287575 .exactZero (none)

def event287577 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33204⟩⟩) 0 ⟨6908⟩ 287576

def event287578 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33204⟩⟩) 1 ⟨33203⟩ 287574

def event287579 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33204⟩⟩) (.product (.predecessor 0 287577 .coefficient) (.predecessor 1 287578 .coefficient) (⟨false, false, none, none, none⟩))

def event287580 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33204⟩⟩, .operator (⟨287576, 0⟩, ⟨287574, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact287581RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact287581RawTermsValid :
    exact287581RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287581 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33204⟩⟩) exact287581RawTerms .large 287579 .exactZero (none)

def event287582 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 287558

def event287583 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact287584RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact287584RawTermsValid :
    exact287584RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287584 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact287584RawTerms .large 287583 .exactZero (none)

def event287585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7307⟩⟩) 0 ⟨7178⟩ 287584

def event287586 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7307⟩⟩) (.identity (.predecessor 0 287585 .coefficient))

def exact287587RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩]

theorem exact287587RawTermsValid :
    exact287587RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287587 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7307⟩⟩) exact287587RawTerms .large 287586 .exactZero (none)

def event287588 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9577⟩⟩) 0 ⟨7307⟩ 287587

def event287589 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9577⟩⟩) (.authority (.operator))

def exact287590RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩]

theorem exact287590RawTermsValid :
    exact287590RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287590 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9577⟩⟩) exact287590RawTerms (.finite 8192) 287589 .exactZero (none)

def event287591 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9578⟩⟩) 0 ⟨9577⟩ 287590

def event287592 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9578⟩⟩) 1 ⟨2370⟩ 287524

def event287593 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9578⟩⟩) (.scale (.predecessor 0 287591 .coefficient) (.value (.predecessor 1 287592 .coefficient)))

def exact287594RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩]

theorem exact287594RawTermsValid :
    exact287594RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287594 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9578⟩⟩) exact287594RawTerms (.finite 8192) 287593 .exactZero (none)

def event287595 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7287⟩⟩) 0 ⟨7178⟩ 287584

def event287596 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7287⟩⟩) (.identity (.predecessor 0 287595 .coefficient))

def exact287597RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩]⟩, (1)⟩]

theorem exact287597RawTermsValid :
    exact287597RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287597 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7287⟩⟩) exact287597RawTerms .large 287596 .exactZero (none)

def event287598 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9579⟩⟩) 0 ⟨7287⟩ 287597

def event287599 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9579⟩⟩) 1 ⟨9578⟩ 287594

def event287600 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9579⟩⟩) (.product (.predecessor 0 287598 .coefficient) (.predecessor 1 287599 .coefficient) (⟨false, false, none, none, none⟩))

def event287601 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9579⟩⟩, .operator (⟨287597, 0⟩, ⟨287594, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩)

def exact287602RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩]

theorem exact287602RawTermsValid :
    exact287602RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287602 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9579⟩⟩) exact287602RawTerms .large 287600 .exactZero (none)

def event287603 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33205⟩⟩) 0 ⟨9579⟩ 287602

def event287604 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33205⟩⟩) 1 ⟨33204⟩ 287581

def event287605 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33205⟩⟩) (.sum [.predecessor 0 287603 .coefficient, .predecessor 1 287604 .coefficient])

def exact287606RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact287606RawTermsValid :
    exact287606RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287606 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33205⟩⟩) exact287606RawTerms .large 287605 .exactZero (none)

def event287607 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33396⟩⟩) 0 ⟨33205⟩ 287606

def event287608 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33396⟩⟩) 1 ⟨33393⟩ 287565

def event287609 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33396⟩⟩) (.product (.predecessor 0 287607 .coefficient) (.predecessor 1 287608 .coefficient) (⟨false, false, none, none, none⟩))

def event287610 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33396⟩⟩, .operator (⟨287606, 0⟩, ⟨287565, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩]⟩, (1)⟩)

def event287611 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33396⟩⟩, .operator (⟨287606, 1⟩, ⟨287565, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩]⟩, (-1)⟩)

def event287612 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨33396⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33393⟩⟩) ⟨32913⟩ 287562)

def event287613 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33396⟩⟩, .relation 287612 0, ⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨32913⟩⟩]⟩, (-1)⟩)

def exact287614RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨32913⟩⟩]⟩, (-1)⟩]

theorem exact287614RawTermsValid :
    exact287614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287614 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33396⟩⟩) exact287614RawTerms .large 287609 .exactZero (none)

def event287615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31780⟩⟩) 0 ⟨31325⟩ 287554

def event287616 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31780⟩⟩) (.authority (.programFamilyFact))

def exact287617RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31780⟩⟩], []⟩, (1)⟩]

theorem exact287617RawTermsValid :
    exact287617RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287617 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31780⟩⟩) exact287617RawTerms (.finite 6) 287616 .exactZero (none)

def event287618 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31782⟩⟩) 0 ⟨6908⟩ 287576

def event287619 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31782⟩⟩) 1 ⟨31780⟩ 287617

def event287620 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31782⟩⟩) (.product (.predecessor 0 287618 .coefficient) (.predecessor 1 287619 .coefficient) (⟨false, true, none, none, some 1⟩))

def event287621 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31782⟩⟩, .operator (⟨287576, 0⟩, ⟨287617, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨31780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact287622RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact287622RawTermsValid :
    exact287622RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287622 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31782⟩⟩) exact287622RawTerms .large 287620 .exactZero (none)

def event287623 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7182⟩⟩) 0 ⟨7177⟩ 287558

def event287624 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7182⟩⟩) (.authority (.operator))

def exact287625RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩]

theorem exact287625RawTermsValid :
    exact287625RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287625 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7182⟩⟩) exact287625RawTerms .large 287624 .exactZero (none)

def event287626 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31783⟩⟩) 0 ⟨7182⟩ 287625

def event287627 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31783⟩⟩) 1 ⟨31782⟩ 287622

def event287628 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31783⟩⟩) (.sum [.predecessor 0 287626 .coefficient, .predecessor 1 287627 .coefficient])

def exact287629RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact287629RawTermsValid :
    exact287629RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287629 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31783⟩⟩) exact287629RawTerms .large 287628 .exactZero (none)

def event287630 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33397⟩⟩) 0 ⟨31783⟩ 287629

def event287631 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33397⟩⟩) 1 ⟨33396⟩ 287614

def event287632 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33397⟩⟩) (.sum [.predecessor 0 287630 .coefficient, .predecessor 1 287631 .coefficient])

def exact287633RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨32913⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact287633RawTermsValid :
    exact287633RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287633 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33397⟩⟩) exact287633RawTerms .large 287632 .exactZero (none)

def event287634 : Event := .preFoldPolynomial 287633 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨32913⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact287635RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨32913⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event287635 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨33397⟩⟩) 287634 exact287635RawTerms .large 287632 .exactZero (none)

def event287636 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨31325⟩⟩) ⟨⟨61⟩, ⟨39⟩, ⟨135⟩⟩ ⟨287472, 287636⟩

def event287637 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨32332⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32329⟩⟩]⟩) (1) 0 2 (.universal 287636 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32329⟩⟩]⟩) (none) 287635)

def event287638 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32332⟩⟩, .relation 287637 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩)

def event287639 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32332⟩⟩, .relation 287637 1, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩]⟩, (-1)⟩)

def event287640 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32332⟩⟩, .relation 287637 2, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨32913⟩⟩]⟩, (1)⟩)

def event287641 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32332⟩⟩, .relation 287637 3, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact287642RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨32913⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact287642RawTermsValid :
    exact287642RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287642 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32332⟩⟩) exact287642RawTerms .large 287468 (.finite 202072841853861888) (some (287470))

def event287643 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33395⟩⟩) 0 ⟨32332⟩ 287642

def event287644 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33395⟩⟩) 1 ⟨33394⟩ 287458

def event287645 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33395⟩⟩) (.sum [.predecessor 0 287643 .coefficient, .predecessor 1 287644 .coefficient])

def event287646 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33395⟩⟩, .operator (⟨287642, 2⟩, ⟨287458, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], [⟨.program ⟨257⟩, ⟨32913⟩⟩]⟩, (-1)⟩)

def event287647 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33395⟩⟩, .operator (⟨287642, 1⟩, ⟨287458, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33393⟩⟩]⟩, (1)⟩)

def event287648 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33395⟩⟩) (.sum [.result 287642 .summary, .result 287458 .summary])

def exact287649RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact287649RawTermsValid :
    exact287649RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287649 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33395⟩⟩) exact287649RawTerms .large 287645 (.finite 2997852872440114577408) (some (287648))

def event287650 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33708⟩⟩) 0 ⟨33395⟩ 287649

def event287651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33708⟩⟩) 1 ⟨33706⟩ 287374

def event287652 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33708⟩⟩) (.product (.predecessor 0 287650 .coefficient) (.predecessor 1 287651 .coefficient) (⟨false, false, none, none, none⟩))

def event287653 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33708⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨33706⟩⟩]⟩) [⟨.result 287374 .coefficient, false, none⟩])

def event287654 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33708⟩⟩) (.product (.result 287649 .summary) (.transfer 287653) (⟨false, false, none, none, none⟩))

def event287655 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33708⟩⟩, .operator (⟨287649, 0⟩, ⟨287374, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33706⟩⟩]⟩, (1)⟩)

def event287656 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33708⟩⟩, .operator (⟨287649, 1⟩, ⟨287374, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33706⟩⟩]⟩, (-1)⟩)

def event287657 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨33708⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33706⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33706⟩⟩) ⟨33047⟩ 287371)

def event287658 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33708⟩⟩, .relation 287657 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31780⟩⟩], [⟨.program ⟨257⟩, ⟨33047⟩⟩]⟩, (-1)⟩)

def exact287659RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33706⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨31780⟩⟩], [⟨.program ⟨257⟩, ⟨33047⟩⟩]⟩, (-1)⟩]

theorem exact287659RawTermsValid :
    exact287659RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287659 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33708⟩⟩) exact287659RawTerms .large 287652 (.finite 32189200113374879571150551121920) (some (287654))

def event287660 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32576⟩⟩) 0 ⟨31781⟩ 13891

def event287661 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32576⟩⟩) (.authority (.relationPreimageSource ⟨63⟩))

def exact287662RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32576⟩⟩]⟩, (1)⟩]

theorem exact287662RawTermsValid :
    exact287662RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287662 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32576⟩⟩) exact287662RawTerms (.finite 5647228698) 287661 .exactZero (none)

def event287663 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32578⟩⟩) 0 ⟨32576⟩ 287662

def event287664 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32578⟩⟩) 1 ⟨2370⟩ 4

def event287665 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32578⟩⟩) (.scale (.predecessor 0 287663 .coefficient) (.value (.predecessor 1 287664 .coefficient)))

def exact287666RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32576⟩⟩]⟩, (1)⟩]

theorem exact287666RawTermsValid :
    exact287666RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287666 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32578⟩⟩) exact287666RawTerms (.finite 5647228698) 287665 .exactZero (none)

def event287667 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32579⟩⟩) 0 ⟨5491⟩ 280745

def event287668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32579⟩⟩) 1 ⟨32578⟩ 287666

def event287669 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32579⟩⟩) (.product (.predecessor 0 287667 .coefficient) (.predecessor 1 287668 .coefficient) (⟨false, false, none, none, none⟩))

def event287670 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32579⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨32576⟩⟩]⟩) [⟨.result 287662 .coefficient, false, none⟩])

def event287671 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32579⟩⟩) (.product (.result 280745 .summary) (.transfer 287670) (⟨false, false, none, none, none⟩))

def event287672 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32579⟩⟩, .operator (⟨280745, 0⟩, ⟨287666, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32576⟩⟩]⟩, (1)⟩)

def event287673 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨32577⟩⟩)

def event287674 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event287675 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event287676 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event287677 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event287678 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event287679 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event287680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event287681 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event287682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 287681

def event287683 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 287679

def event287684 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 287682 .coefficient) (.value (.predecessor 1 287683 .coefficient)))

def event287685 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event287686 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 287685

def event287687 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 287677

def event287688 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 287686 .coefficient, .predecessor 1 287687 .coefficient])

def event287689 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event287690 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 287689

def event287691 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 287675

def event287692 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 287691 .coefficient))

def event287693 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event287694 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24218⟩⟩) 0 ⟨5487⟩ 287693

def event287695 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24218⟩⟩) (.authority (.programFamilyFact))

def exact287696RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩], []⟩, (1)⟩]

theorem exact287696RawTermsValid :
    exact287696RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287696 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24218⟩⟩) exact287696RawTerms (.finite 6) 287695 .exactZero (none)

def event287697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31323⟩⟩) 0 ⟨5487⟩ 287693

def event287698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31323⟩⟩) (.authority (.programFamilyFact))

def exact287699RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31323⟩⟩], []⟩, (1)⟩]

theorem exact287699RawTermsValid :
    exact287699RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287699 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31323⟩⟩) exact287699RawTerms (.finite 6) 287698 .exactZero (none)

def event287700 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31324⟩⟩) 0 ⟨31323⟩ 287699

def event287701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31324⟩⟩) 1 ⟨24218⟩ 287696

def event287702 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31324⟩⟩) (.product (.predecessor 0 287700 .coefficient) (.predecessor 1 287701 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event287703 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31324⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24218⟩⟩, ⟨.program ⟨257⟩, ⟨31323⟩⟩], []⟩) [⟨.result 287699 .coefficient, true, some 1⟩, ⟨.result 287696 .coefficient, true, some 1⟩])

def event287704 : Event := .survivorFold (1) 287703

def exact287705RawTerms : List Term := []

theorem exact287705RawTermsValid :
    exact287705RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287705 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31324⟩⟩) exact287705RawTerms (.finite 36) 287702 (.finite 36) (some (287703))

def event287706 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31325⟩⟩) 0 ⟨31324⟩ 287705

def event287707 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31325⟩⟩) (.identity (.predecessor 0 287706 .coefficient))

def event287708 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31325⟩⟩) (.finite 36)

def event287709 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31780⟩⟩) 0 ⟨31325⟩ 287708

def event287710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31780⟩⟩) (.authority (.programFamilyFact))

def exact287711RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31780⟩⟩], []⟩, (1)⟩]

theorem exact287711RawTermsValid :
    exact287711RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287711 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31780⟩⟩) exact287711RawTerms (.finite 6) 287710 .exactZero (none)

def event287712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31781⟩⟩) 0 ⟨31780⟩ 287711

def event287713 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31781⟩⟩) (.identity (.predecessor 0 287712 .coefficient))

def event287714 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31781⟩⟩) (.finite 6)

def event287715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32576⟩⟩) 0 ⟨31781⟩ 287714

def event287716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32576⟩⟩) (.authority (.relationPreimageSource ⟨63⟩))

def exact287717RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32576⟩⟩]⟩, (1)⟩]

theorem exact287717RawTermsValid :
    exact287717RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287717 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32576⟩⟩) exact287717RawTerms (.finite 5647228698) 287716 .exactZero (none)

def event287718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact287719RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact287719RawTermsValid :
    exact287719RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287719 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact287719RawTerms .large 287718 .exactZero (none)

def event287720 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32577⟩⟩) 0 ⟨35⟩ 287719

def event287721 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32577⟩⟩) 1 ⟨32576⟩ 287717

def event287722 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32577⟩⟩) (.product (.predecessor 0 287720 .coefficient) (.predecessor 1 287721 .coefficient) (⟨false, false, none, none, none⟩))

def event287723 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32577⟩⟩, .operator (⟨287719, 0⟩, ⟨287717, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32576⟩⟩]⟩, (1)⟩)

def exact287724RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32576⟩⟩]⟩, (1)⟩]

theorem exact287724RawTermsValid :
    exact287724RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event287724 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32577⟩⟩) exact287724RawTerms .large 287722 .exactZero (none)

def event287725 : Event := .preFoldPolynomial 287724 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32576⟩⟩]⟩, (1)⟩] .exactZero none

def exact287726RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32576⟩⟩]⟩, (1)⟩]

def event287726 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨32577⟩⟩) 287725 exact287726RawTerms .large 287722 .exactZero (none)

def event287727 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨33711⟩⟩)

def event287728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event287729 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event287730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event287731 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event287732 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event287733 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event287734 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event287735 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event287736 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 287735

def event287737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 287733

def event287738 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 287736 .coefficient) (.value (.predecessor 1 287737 .coefficient)))

def event287739 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event287740 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 287739

def event287741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 287731

def event287742 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 287740 .coefficient, .predecessor 1 287741 .coefficient])

def event287743 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def eventLeaf17968 : Array AnnotatedEvent := #[
  { event := event287488
    frameStart := 287472 },
  { event := event287489
    frameStart := 287472 },
  { event := event287490
    frameStart := 287472 },
  { event := event287491
    frameStart := 287472 },
  { event := event287492
    frameStart := 287472 },
  { event := event287493
    frameStart := 287472 },
  { event := event287494
    frameStart := 287472 },
  { event := event287495
    frameStart := 287472 },
  { event := event287496
    frameStart := 287472 },
  { event := event287497
    frameStart := 287472 },
  { event := event287498
    frameStart := 287472 },
  { event := event287499
    frameStart := 287472 },
  { event := event287500
    frameStart := 287472 },
  { event := event287501
    frameStart := 287472 },
  { event := event287502
    frameStart := 287472 },
  { event := event287503
    frameStart := 287472 }
]

def eventLeaf17969 : Array AnnotatedEvent := #[
  { event := event287504
    frameStart := 287472 },
  { event := event287505
    frameStart := 287472 },
  { event := event287506
    frameStart := 287472 },
  { event := event287507
    frameStart := 287472 },
  { event := event287508
    frameStart := 287472 },
  { event := event287509
    frameStart := 287472 },
  { event := event287510
    frameStart := 287472 },
  { event := event287511
    frameStart := 287472 },
  { event := event287512
    frameStart := 287472 },
  { event := event287513
    frameStart := 287472 },
  { event := event287514
    frameStart := 287472 },
  { event := event287515
    frameStart := 287472 },
  { event := event287516
    frameStart := 287472 },
  { event := event287517
    frameStart := 287472 },
  { event := event287518
    frameStart := 287472 },
  { event := event287519
    frameStart := 287472 }
]

def eventLeaf17970 : Array AnnotatedEvent := #[
  { event := event287520
    frameStart := 287520 },
  { event := event287521
    frameStart := 287520 },
  { event := event287522
    frameStart := 287520 },
  { event := event287523
    frameStart := 287520 },
  { event := event287524
    frameStart := 287520 },
  { event := event287525
    frameStart := 287520 },
  { event := event287526
    frameStart := 287520 },
  { event := event287527
    frameStart := 287520 },
  { event := event287528
    frameStart := 287520 },
  { event := event287529
    frameStart := 287520 },
  { event := event287530
    frameStart := 287520 },
  { event := event287531
    frameStart := 287520 },
  { event := event287532
    frameStart := 287520 },
  { event := event287533
    frameStart := 287520 },
  { event := event287534
    frameStart := 287520 },
  { event := event287535
    frameStart := 287520 }
]

def eventLeaf17971 : Array AnnotatedEvent := #[
  { event := event287536
    frameStart := 287520 },
  { event := event287537
    frameStart := 287520 },
  { event := event287538
    frameStart := 287520 },
  { event := event287539
    frameStart := 287520 },
  { event := event287540
    frameStart := 287520 },
  { event := event287541
    frameStart := 287520 },
  { event := event287542
    frameStart := 287520 },
  { event := event287543
    frameStart := 287520 },
  { event := event287544
    frameStart := 287520 },
  { event := event287545
    frameStart := 287520 },
  { event := event287546
    frameStart := 287520 },
  { event := event287547
    frameStart := 287520 },
  { event := event287548
    frameStart := 287520 },
  { event := event287549
    frameStart := 287520 },
  { event := event287550
    frameStart := 287520 },
  { event := event287551
    frameStart := 287520 }
]

def eventLeaf17972 : Array AnnotatedEvent := #[
  { event := event287552
    frameStart := 287520 },
  { event := event287553
    frameStart := 287520 },
  { event := event287554
    frameStart := 287520 },
  { event := event287555
    frameStart := 287520 },
  { event := event287556
    frameStart := 287520 },
  { event := event287557
    frameStart := 287520 },
  { event := event287558
    frameStart := 287520 },
  { event := event287559
    frameStart := 287520 },
  { event := event287560
    frameStart := 287520 },
  { event := event287561
    frameStart := 287520 },
  { event := event287562
    frameStart := 287520 },
  { event := event287563
    frameStart := 287520 },
  { event := event287564
    frameStart := 287520 },
  { event := event287565
    frameStart := 287520 },
  { event := event287566
    frameStart := 287520 },
  { event := event287567
    frameStart := 287520 }
]

def eventLeaf17973 : Array AnnotatedEvent := #[
  { event := event287568
    frameStart := 287520 },
  { event := event287569
    frameStart := 287520 },
  { event := event287570
    frameStart := 287520 },
  { event := event287571
    frameStart := 287520 },
  { event := event287572
    frameStart := 287520 },
  { event := event287573
    frameStart := 287520 },
  { event := event287574
    frameStart := 287520 },
  { event := event287575
    frameStart := 287520 },
  { event := event287576
    frameStart := 287520 },
  { event := event287577
    frameStart := 287520 },
  { event := event287578
    frameStart := 287520 },
  { event := event287579
    frameStart := 287520 },
  { event := event287580
    frameStart := 287520 },
  { event := event287581
    frameStart := 287520 },
  { event := event287582
    frameStart := 287520 },
  { event := event287583
    frameStart := 287520 }
]

def eventLeaf17974 : Array AnnotatedEvent := #[
  { event := event287584
    frameStart := 287520 },
  { event := event287585
    frameStart := 287520 },
  { event := event287586
    frameStart := 287520 },
  { event := event287587
    frameStart := 287520 },
  { event := event287588
    frameStart := 287520 },
  { event := event287589
    frameStart := 287520 },
  { event := event287590
    frameStart := 287520 },
  { event := event287591
    frameStart := 287520 },
  { event := event287592
    frameStart := 287520 },
  { event := event287593
    frameStart := 287520 },
  { event := event287594
    frameStart := 287520 },
  { event := event287595
    frameStart := 287520 },
  { event := event287596
    frameStart := 287520 },
  { event := event287597
    frameStart := 287520 },
  { event := event287598
    frameStart := 287520 },
  { event := event287599
    frameStart := 287520 }
]

def eventLeaf17975 : Array AnnotatedEvent := #[
  { event := event287600
    frameStart := 287520 },
  { event := event287601
    frameStart := 287520 },
  { event := event287602
    frameStart := 287520 },
  { event := event287603
    frameStart := 287520 },
  { event := event287604
    frameStart := 287520 },
  { event := event287605
    frameStart := 287520 },
  { event := event287606
    frameStart := 287520 },
  { event := event287607
    frameStart := 287520 },
  { event := event287608
    frameStart := 287520 },
  { event := event287609
    frameStart := 287520 },
  { event := event287610
    frameStart := 287520 },
  { event := event287611
    frameStart := 287520 },
  { event := event287612
    frameStart := 287520 },
  { event := event287613
    frameStart := 287520 },
  { event := event287614
    frameStart := 287520 },
  { event := event287615
    frameStart := 287520 }
]

def eventLeaf17976 : Array AnnotatedEvent := #[
  { event := event287616
    frameStart := 287520 },
  { event := event287617
    frameStart := 287520 },
  { event := event287618
    frameStart := 287520 },
  { event := event287619
    frameStart := 287520 },
  { event := event287620
    frameStart := 287520 },
  { event := event287621
    frameStart := 287520 },
  { event := event287622
    frameStart := 287520 },
  { event := event287623
    frameStart := 287520 },
  { event := event287624
    frameStart := 287520 },
  { event := event287625
    frameStart := 287520 },
  { event := event287626
    frameStart := 287520 },
  { event := event287627
    frameStart := 287520 },
  { event := event287628
    frameStart := 287520 },
  { event := event287629
    frameStart := 287520 },
  { event := event287630
    frameStart := 287520 },
  { event := event287631
    frameStart := 287520 }
]

def eventLeaf17977 : Array AnnotatedEvent := #[
  { event := event287632
    frameStart := 287520 },
  { event := event287633
    frameStart := 287520 },
  { event := event287634
    frameStart := 287520 },
  { event := event287635
    frameStart := 287520 },
  { event := event287636
    frameStart := 0 },
  { event := event287637
    frameStart := 0 },
  { event := event287638
    frameStart := 0 },
  { event := event287639
    frameStart := 0 },
  { event := event287640
    frameStart := 0 },
  { event := event287641
    frameStart := 0 },
  { event := event287642
    frameStart := 0 },
  { event := event287643
    frameStart := 0 },
  { event := event287644
    frameStart := 0 },
  { event := event287645
    frameStart := 0 },
  { event := event287646
    frameStart := 0 },
  { event := event287647
    frameStart := 0 }
]

def eventLeaf17978 : Array AnnotatedEvent := #[
  { event := event287648
    frameStart := 0 },
  { event := event287649
    frameStart := 0 },
  { event := event287650
    frameStart := 0 },
  { event := event287651
    frameStart := 0 },
  { event := event287652
    frameStart := 0 },
  { event := event287653
    frameStart := 0 },
  { event := event287654
    frameStart := 0 },
  { event := event287655
    frameStart := 0 },
  { event := event287656
    frameStart := 0 },
  { event := event287657
    frameStart := 0 },
  { event := event287658
    frameStart := 0 },
  { event := event287659
    frameStart := 0 },
  { event := event287660
    frameStart := 0 },
  { event := event287661
    frameStart := 0 },
  { event := event287662
    frameStart := 0 },
  { event := event287663
    frameStart := 0 }
]

def eventLeaf17979 : Array AnnotatedEvent := #[
  { event := event287664
    frameStart := 0 },
  { event := event287665
    frameStart := 0 },
  { event := event287666
    frameStart := 0 },
  { event := event287667
    frameStart := 0 },
  { event := event287668
    frameStart := 0 },
  { event := event287669
    frameStart := 0 },
  { event := event287670
    frameStart := 0 },
  { event := event287671
    frameStart := 0 },
  { event := event287672
    frameStart := 0 },
  { event := event287673
    frameStart := 287673 },
  { event := event287674
    frameStart := 287673 },
  { event := event287675
    frameStart := 287673 },
  { event := event287676
    frameStart := 287673 },
  { event := event287677
    frameStart := 287673 },
  { event := event287678
    frameStart := 287673 },
  { event := event287679
    frameStart := 287673 }
]

def eventLeaf17980 : Array AnnotatedEvent := #[
  { event := event287680
    frameStart := 287673 },
  { event := event287681
    frameStart := 287673 },
  { event := event287682
    frameStart := 287673 },
  { event := event287683
    frameStart := 287673 },
  { event := event287684
    frameStart := 287673 },
  { event := event287685
    frameStart := 287673 },
  { event := event287686
    frameStart := 287673 },
  { event := event287687
    frameStart := 287673 },
  { event := event287688
    frameStart := 287673 },
  { event := event287689
    frameStart := 287673 },
  { event := event287690
    frameStart := 287673 },
  { event := event287691
    frameStart := 287673 },
  { event := event287692
    frameStart := 287673 },
  { event := event287693
    frameStart := 287673 },
  { event := event287694
    frameStart := 287673 },
  { event := event287695
    frameStart := 287673 }
]

def eventLeaf17981 : Array AnnotatedEvent := #[
  { event := event287696
    frameStart := 287673 },
  { event := event287697
    frameStart := 287673 },
  { event := event287698
    frameStart := 287673 },
  { event := event287699
    frameStart := 287673 },
  { event := event287700
    frameStart := 287673 },
  { event := event287701
    frameStart := 287673 },
  { event := event287702
    frameStart := 287673 },
  { event := event287703
    frameStart := 287673 },
  { event := event287704
    frameStart := 287673 },
  { event := event287705
    frameStart := 287673 },
  { event := event287706
    frameStart := 287673 },
  { event := event287707
    frameStart := 287673 },
  { event := event287708
    frameStart := 287673 },
  { event := event287709
    frameStart := 287673 },
  { event := event287710
    frameStart := 287673 },
  { event := event287711
    frameStart := 287673 }
]

def eventLeaf17982 : Array AnnotatedEvent := #[
  { event := event287712
    frameStart := 287673 },
  { event := event287713
    frameStart := 287673 },
  { event := event287714
    frameStart := 287673 },
  { event := event287715
    frameStart := 287673 },
  { event := event287716
    frameStart := 287673 },
  { event := event287717
    frameStart := 287673 },
  { event := event287718
    frameStart := 287673 },
  { event := event287719
    frameStart := 287673 },
  { event := event287720
    frameStart := 287673 },
  { event := event287721
    frameStart := 287673 },
  { event := event287722
    frameStart := 287673 },
  { event := event287723
    frameStart := 287673 },
  { event := event287724
    frameStart := 287673 },
  { event := event287725
    frameStart := 287673 },
  { event := event287726
    frameStart := 287673 },
  { event := event287727
    frameStart := 287727 }
]

def eventLeaf17983 : Array AnnotatedEvent := #[
  { event := event287728
    frameStart := 287727 },
  { event := event287729
    frameStart := 287727 },
  { event := event287730
    frameStart := 287727 },
  { event := event287731
    frameStart := 287727 },
  { event := event287732
    frameStart := 287727 },
  { event := event287733
    frameStart := 287727 },
  { event := event287734
    frameStart := 287727 },
  { event := event287735
    frameStart := 287727 },
  { event := event287736
    frameStart := 287727 },
  { event := event287737
    frameStart := 287727 },
  { event := event287738
    frameStart := 287727 },
  { event := event287739
    frameStart := 287727 },
  { event := event287740
    frameStart := 287727 },
  { event := event287741
    frameStart := 287727 },
  { event := event287742
    frameStart := 287727 },
  { event := event287743
    frameStart := 287727 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1123
