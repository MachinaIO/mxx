import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events330

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event84480 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15619⟩⟩) 0 ⟨12471⟩ 84479

def event84481 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15619⟩⟩) 1 ⟨15618⟩ 84476

def event84482 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15619⟩⟩) (.product (.predecessor 0 84480 .coefficient) (.predecessor 1 84481 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event84483 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15619⟩⟩, .operator (⟨84479, 0⟩, ⟨84476, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], []⟩, (1)⟩)

def exact84484RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], []⟩, (1)⟩]

theorem exact84484RawTermsValid :
    exact84484RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84484 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15619⟩⟩) exact84484RawTerms (.finite 4) 84482 .exactZero (none)

def event84485 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15620⟩⟩) 0 ⟨15619⟩ 84484

def event84486 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15620⟩⟩) (.identity (.predecessor 0 84485 .coefficient))

def event84487 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15620⟩⟩) (.finite 4)

def event84488 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15836⟩⟩) 0 ⟨15620⟩ 84487

def event84489 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15836⟩⟩) (.authority (.programFamilyFact))

def exact84490RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], []⟩, (1)⟩]

theorem exact84490RawTermsValid :
    exact84490RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84490 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15836⟩⟩) exact84490RawTerms (.finite 2) 84489 .exactZero (none)

def event84491 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15837⟩⟩) 0 ⟨15836⟩ 84490

def event84492 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15837⟩⟩) (.identity (.predecessor 0 84491 .coefficient))

def event84493 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15837⟩⟩) (.finite 2)

def event84494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17053⟩⟩) 0 ⟨15837⟩ 84493

def event84495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17053⟩⟩) (.authority (.programFamilyFact))

def event84496 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17053⟩⟩) (.finite 3720)

def event84497 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event84498 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17055⟩⟩) 0 ⟨7177⟩ 84497

def event84499 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17055⟩⟩) 1 ⟨17053⟩ 84496

def event84500 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17055⟩⟩) (.authority (.operator))

def exact84501RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17055⟩⟩]⟩, (1)⟩]

theorem exact84501RawTermsValid :
    exact84501RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84501 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17055⟩⟩) exact84501RawTerms .large 84500 .exactZero (none)

def event84502 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17929⟩⟩) 0 ⟨17055⟩ 84501

def event84503 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17929⟩⟩) (.authority (.operator))

def exact84504RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17929⟩⟩]⟩, (1)⟩]

theorem exact84504RawTermsValid :
    exact84504RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84504 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17929⟩⟩) exact84504RawTerms (.finite 8192) 84503 .exactZero (none)

def event84505 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event84506 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event84507 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17230⟩⟩) 0 ⟨15837⟩ 84493

def event84508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17230⟩⟩) 1 ⟨136⟩ 84506

def event84509 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17230⟩⟩) (.sum [.predecessor 0 84507 .coefficient, .predecessor 1 84508 .coefficient])

def event84510 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17230⟩⟩) (.finite 2)

def event84511 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17231⟩⟩) 0 ⟨17230⟩ 84510

def event84512 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17231⟩⟩) (.identity (.predecessor 0 84511 .coefficient))

def exact84513RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], []⟩, (1)⟩]

theorem exact84513RawTermsValid :
    exact84513RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84513 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17231⟩⟩) exact84513RawTerms (.finite 2) 84512 .exactZero (none)

def event84514 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact84515RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact84515RawTermsValid :
    exact84515RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84515 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact84515RawTerms .large 84514 .exactZero (none)

def event84516 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17232⟩⟩) 0 ⟨6908⟩ 84515

def event84517 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17232⟩⟩) 1 ⟨17231⟩ 84513

def event84518 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17232⟩⟩) (.product (.predecessor 0 84516 .coefficient) (.predecessor 1 84517 .coefficient) (⟨false, false, none, none, none⟩))

def event84519 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17232⟩⟩, .operator (⟨84515, 0⟩, ⟨84513, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact84520RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact84520RawTermsValid :
    exact84520RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84520 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17232⟩⟩) exact84520RawTerms .large 84518 .exactZero (none)

def event84521 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 84497

def event84522 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact84523RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact84523RawTermsValid :
    exact84523RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84523 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact84523RawTerms .large 84522 .exactZero (none)

def event84524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17233⟩⟩) 0 ⟨7179⟩ 84523

def event84525 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17233⟩⟩) 1 ⟨17232⟩ 84520

def event84526 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17233⟩⟩) (.sum [.predecessor 0 84524 .coefficient, .predecessor 1 84525 .coefficient])

def exact84527RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84527RawTermsValid :
    exact84527RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84527 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17233⟩⟩) exact84527RawTerms .large 84526 .exactZero (none)

def event84528 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17930⟩⟩) 0 ⟨17233⟩ 84527

def event84529 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17930⟩⟩) 1 ⟨17929⟩ 84504

def event84530 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17930⟩⟩) (.product (.predecessor 0 84528 .coefficient) (.predecessor 1 84529 .coefficient) (⟨false, false, none, none, none⟩))

def event84531 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17930⟩⟩, .operator (⟨84527, 0⟩, ⟨84504, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17929⟩⟩]⟩, (1)⟩)

def event84532 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17930⟩⟩, .operator (⟨84527, 1⟩, ⟨84504, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17929⟩⟩]⟩, (-1)⟩)

def event84533 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17930⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17929⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17929⟩⟩) ⟨17055⟩ 84501)

def event84534 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17930⟩⟩, .relation 84533 0, ⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨17055⟩⟩]⟩, (-1)⟩)

def exact84535RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17929⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨17055⟩⟩]⟩, (-1)⟩]

theorem exact84535RawTermsValid :
    exact84535RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84535 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17930⟩⟩) exact84535RawTerms .large 84530 .exactZero (none)

def event84536 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16131⟩⟩) 0 ⟨15837⟩ 84493

def event84537 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16131⟩⟩) (.authority (.programFamilyFact))

def exact84538RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], []⟩, (1)⟩]

theorem exact84538RawTermsValid :
    exact84538RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84538 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16131⟩⟩) exact84538RawTerms (.finite 43) 84537 .exactZero (none)

def event84539 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16132⟩⟩) 0 ⟨6908⟩ 84515

def event84540 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16132⟩⟩) 1 ⟨16131⟩ 84538

def event84541 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16132⟩⟩) (.product (.predecessor 0 84539 .coefficient) (.predecessor 1 84540 .coefficient) (⟨false, true, none, none, some 1⟩))

def event84542 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16132⟩⟩, .operator (⟨84515, 0⟩, ⟨84538, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact84543RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact84543RawTermsValid :
    exact84543RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84543 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16132⟩⟩) exact84543RawTerms .large 84541 .exactZero (none)

def event84544 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 84497

def event84545 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact84546RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact84546RawTermsValid :
    exact84546RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84546 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact84546RawTerms .large 84545 .exactZero (none)

def event84547 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16133⟩⟩) 0 ⟨7198⟩ 84546

def event84548 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16133⟩⟩) 1 ⟨16132⟩ 84543

def event84549 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16133⟩⟩) (.sum [.predecessor 0 84547 .coefficient, .predecessor 1 84548 .coefficient])

def exact84550RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84550RawTermsValid :
    exact84550RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84550 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16133⟩⟩) exact84550RawTerms .large 84549 .exactZero (none)

def event84551 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17933⟩⟩) 0 ⟨16133⟩ 84550

def event84552 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17933⟩⟩) 1 ⟨17930⟩ 84535

def event84553 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17933⟩⟩) (.sum [.predecessor 0 84551 .coefficient, .predecessor 1 84552 .coefficient])

def exact84554RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17929⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨17055⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84554RawTermsValid :
    exact84554RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84554 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17933⟩⟩) exact84554RawTerms .large 84553 .exactZero (none)

def event84555 : Event := .preFoldPolynomial 84554 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17929⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨17055⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact84556RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17929⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨17055⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event84556 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17933⟩⟩) 84555 exact84556RawTerms .large 84553 .exactZero (none)

def event84557 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15837⟩⟩) ⟨⟨77⟩, ⟨57⟩, ⟨135⟩⟩ ⟨84399, 84557⟩

def event84558 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16719⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16716⟩⟩]⟩) (1) 0 2 (.universal 84557 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16716⟩⟩]⟩) (none) 84556)

def event84559 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16719⟩⟩, .relation 84558 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩)

def event84560 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16719⟩⟩, .relation 84558 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17929⟩⟩]⟩, (-1)⟩)

def event84561 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16719⟩⟩, .relation 84558 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨17055⟩⟩]⟩, (1)⟩)

def event84562 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16719⟩⟩, .relation 84558 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact84563RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17929⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨17055⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84563RawTermsValid :
    exact84563RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84563 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16719⟩⟩) exact84563RawTerms .large 84395 (.finite 202072841853861888) (some (84397))

def event84564 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17932⟩⟩) 0 ⟨16719⟩ 84563

def event84565 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17932⟩⟩) 1 ⟨17931⟩ 84385

def event84566 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17932⟩⟩) (.sum [.predecessor 0 84564 .coefficient, .predecessor 1 84565 .coefficient])

def event84567 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17932⟩⟩, .operator (⟨84563, 0⟩, ⟨84385, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17929⟩⟩]⟩, (1)⟩)

def event84568 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17932⟩⟩, .operator (⟨84563, 2⟩, ⟨84385, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨17055⟩⟩]⟩, (-1)⟩)

def event84569 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17932⟩⟩) (.sum [.result 84563 .summary, .result 84385 .summary])

def exact84570RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84570RawTermsValid :
    exact84570RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84570 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17932⟩⟩) exact84570RawTerms .large 84566 (.finite 32188807212483706889510625476608) (some (84569))

def event84571 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20842⟩⟩) 0 ⟨17932⟩ 84570

def event84572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20842⟩⟩) 1 ⟨20841⟩ 84088

def event84573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20842⟩⟩) (.sum [.predecessor 0 84571 .coefficient, .predecessor 1 84572 .coefficient])

def event84574 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20842⟩⟩) (.sum [.result 84570 .summary, .result 84088 .summary])

def exact84575RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84575RawTermsValid :
    exact84575RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84575 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20842⟩⟩) exact84575RawTerms .large 84573 (.finite 64377712650190257467641695830016) (some (84574))

def event84576 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24062⟩⟩) 0 ⟨20842⟩ 84575

def event84577 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24062⟩⟩) 1 ⟨24061⟩ 83606

def event84578 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24062⟩⟩) (.sum [.predecessor 0 84576 .coefficient, .predecessor 1 84577 .coefficient])

def event84579 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24062⟩⟩) (.sum [.result 84575 .summary, .result 83606 .summary])

def exact84580RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84580RawTermsValid :
    exact84580RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84580 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24062⟩⟩) exact84580RawTerms .large 84578 (.finite 96566716313119651734393211060224) (some (84579))

def event84581 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34082⟩⟩) 0 ⟨24062⟩ 84580

def event84582 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34082⟩⟩) 1 ⟨34081⟩ 83124

def event84583 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34082⟩⟩) (.sum [.predecessor 0 84581 .coefficient, .predecessor 1 84582 .coefficient])

def event84584 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34082⟩⟩) (.sum [.result 84580 .summary, .result 83124 .summary])

def exact84585RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84585RawTermsValid :
    exact84585RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84585 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34082⟩⟩) exact84585RawTerms .large 84583 (.finite 128755916426494733378385616044032) (some (84584))

def event84586 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53142⟩⟩) 0 ⟨34082⟩ 84585

def event84587 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53142⟩⟩) 1 ⟨53141⟩ 82642

def event84588 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53142⟩⟩) (.sum [.predecessor 0 84586 .coefficient, .predecessor 1 84587 .coefficient])

def event84589 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53142⟩⟩) (.sum [.result 84585 .summary, .result 82642 .summary])

def exact84590RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84590RawTermsValid :
    exact84590RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84590 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53142⟩⟩) exact84590RawTerms .large 84588 (.finite 160945509440761189776859800535040) (some (84589))

def event84591 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56122⟩⟩) 0 ⟨53142⟩ 84590

def event84592 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56122⟩⟩) 1 ⟨56121⟩ 82160

def event84593 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56122⟩⟩) (.sum [.predecessor 0 84591 .coefficient, .predecessor 1 84592 .coefficient])

def event84594 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56122⟩⟩) (.sum [.result 84590 .summary, .result 82160 .summary])

def exact84595RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84595RawTermsValid :
    exact84595RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84595 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56122⟩⟩) exact84595RawTerms .large 84593 (.finite 193135298905473333552574874779648) (some (84594))

def event84596 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59102⟩⟩) 0 ⟨56122⟩ 84595

def event84597 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59102⟩⟩) 1 ⟨59101⟩ 81678

def event84598 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59102⟩⟩) (.sum [.predecessor 0 84596 .coefficient, .predecessor 1 84597 .coefficient])

def event84599 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59102⟩⟩) (.sum [.result 84595 .summary, .result 81678 .summary])

def exact84600RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84600RawTermsValid :
    exact84600RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84600 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59102⟩⟩) exact84600RawTerms .large 84598 (.finite 225325481271076852082771728531456) (some (84599))

def event84601 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62082⟩⟩) 0 ⟨59102⟩ 84600

def event84602 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62082⟩⟩) 1 ⟨62081⟩ 81196

def event84603 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62082⟩⟩) (.sum [.predecessor 0 84601 .coefficient, .predecessor 1 84602 .coefficient])

def event84604 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62082⟩⟩) (.sum [.result 84600 .summary, .result 81196 .summary])

def exact84605RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84605RawTermsValid :
    exact84605RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84605 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62082⟩⟩) exact84605RawTerms .large 84603 (.finite 257515860087126057990209472036864) (some (84604))

def event84606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65062⟩⟩) 0 ⟨62082⟩ 84605

def event84607 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65062⟩⟩) 1 ⟨65061⟩ 80714

def event84608 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65062⟩⟩) (.sum [.predecessor 0 84606 .coefficient, .predecessor 1 84607 .coefficient])

def event84609 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65062⟩⟩) (.sum [.result 84605 .summary, .result 80714 .summary])

def exact84610RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84610RawTermsValid :
    exact84610RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84610 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65062⟩⟩) exact84610RawTerms .large 84608 (.finite 289706631804066638652128995049472) (some (84609))

def event84611 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70655⟩⟩) 0 ⟨65062⟩ 84610

def event84612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70655⟩⟩) 1 ⟨70654⟩ 80232

def event84613 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70655⟩⟩) (.sum [.predecessor 0 84611 .coefficient, .predecessor 1 84612 .coefficient])

def event84614 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70655⟩⟩) (.sum [.result 84610 .summary, .result 80232 .summary])

def exact84615RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84615RawTermsValid :
    exact84615RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84615 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70655⟩⟩) exact84615RawTerms .large 84613 (.finite 321897992872344281445771187322880) (some (84614))

def event84616 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70656⟩⟩) 0 ⟨70655⟩ 84615

def event84617 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70656⟩⟩) 1 ⟨28442⟩ 79750

def event84618 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70656⟩⟩) (.sum [.predecessor 0 84616 .coefficient, .predecessor 1 84617 .coefficient])

def event84619 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70656⟩⟩) (.sum [.result 84615 .summary, .result 79750 .summary])

def exact84620RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84620RawTermsValid :
    exact84620RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84620 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70656⟩⟩) exact84620RawTerms .large 84618 (.finite 354089550391067611616654269349888) (some (84619))

def event84621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70657⟩⟩) 0 ⟨70656⟩ 84620

def event84622 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70657⟩⟩) 1 ⟨31122⟩ 79268

def event84623 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70657⟩⟩) (.sum [.predecessor 0 84621 .coefficient, .predecessor 1 84622 .coefficient])

def event84624 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70657⟩⟩) (.sum [.result 84620 .summary, .result 79268 .summary])

def exact84625RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84625RawTermsValid :
    exact84625RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84625 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70657⟩⟩) exact84625RawTerms .large 84623 (.finite 386281697261128003919260020637696) (some (84624))

def event84626 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70658⟩⟩) 0 ⟨70657⟩ 84625

def event84627 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70658⟩⟩) 1 ⟨36782⟩ 78786

def event84628 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70658⟩⟩) (.sum [.predecessor 0 84626 .coefficient, .predecessor 1 84627 .coefficient])

def event84629 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70658⟩⟩) (.sum [.result 84625 .summary, .result 78786 .summary])

def exact84630RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84630RawTermsValid :
    exact84630RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84630 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70658⟩⟩) exact84630RawTerms .large 84628 (.finite 418474237032079770976347551432704) (some (84629))

def event84631 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70659⟩⟩) 0 ⟨70658⟩ 84630

def event84632 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70659⟩⟩) 1 ⟨39462⟩ 78304

def event84633 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70659⟩⟩) (.sum [.predecessor 0 84631 .coefficient, .predecessor 1 84632 .coefficient])

def event84634 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70659⟩⟩) (.sum [.result 84630 .summary, .result 78304 .summary])

def exact84635RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84635RawTermsValid :
    exact84635RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84635 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70659⟩⟩) exact84635RawTerms .large 84633 (.finite 450666973253477225410675971981312) (some (84634))

def event84636 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70660⟩⟩) 0 ⟨70659⟩ 84635

def event84637 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70660⟩⟩) 1 ⟨42142⟩ 77822

def event84638 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70660⟩⟩) (.sum [.predecessor 0 84636 .coefficient, .predecessor 1 84637 .coefficient])

def event84639 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70660⟩⟩) (.sum [.result 84635 .summary, .result 77822 .summary])

def exact84640RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84640RawTermsValid :
    exact84640RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84640 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70660⟩⟩) exact84640RawTerms .large 84638 (.finite 482860102375766054599486172037120) (some (84639))

def event84641 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70661⟩⟩) 0 ⟨70660⟩ 84640

def event84642 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70661⟩⟩) 1 ⟨44822⟩ 77340

def event84643 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70661⟩⟩) (.sum [.predecessor 0 84641 .coefficient, .predecessor 1 84642 .coefficient])

def event84644 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70661⟩⟩) (.sum [.result 84640 .summary, .result 77340 .summary])

def exact84645RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84645RawTermsValid :
    exact84645RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84645 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70661⟩⟩) exact84645RawTerms .large 84643 (.finite 515053820849391945920019041353728) (some (84644))

def event84646 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70662⟩⟩) 0 ⟨70661⟩ 84645

def event84647 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70662⟩⟩) 1 ⟨47502⟩ 76858

def event84648 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70662⟩⟩) (.sum [.predecessor 0 84646 .coefficient, .predecessor 1 84647 .coefficient])

def event84649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70662⟩⟩) (.sum [.result 84645 .summary, .result 76858 .summary])

def exact84650RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84650RawTermsValid :
    exact84650RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84650 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70662⟩⟩) exact84650RawTerms .large 84648 (.finite 547248128674354899372274579931136) (some (84649))

def event84651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70663⟩⟩) 0 ⟨70662⟩ 84650

def event84652 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70663⟩⟩) 1 ⟨50182⟩ 76376

def event84653 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70663⟩⟩) (.sum [.predecessor 0 84651 .coefficient, .predecessor 1 84652 .coefficient])

def event84654 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70663⟩⟩) (.sum [.result 84650 .summary, .result 76376 .summary])

def exact84655RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84655RawTermsValid :
    exact84655RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84655 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70663⟩⟩) exact84655RawTerms .large 84653 (.finite 579442632949763540201771008262144) (some (84654))

def event84656 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71439⟩⟩) 0 ⟨70663⟩ 84655

def event84657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71439⟩⟩) 1 ⟨71437⟩ 75878

def event84658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71439⟩⟩) (.product (.predecessor 0 84656 .coefficient) (.predecessor 1 84657 .coefficient) (⟨false, false, none, none, none⟩))

def event84659 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71439⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) [⟨.result 75878 .coefficient, false, none⟩])

def event84660 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71439⟩⟩) (.product (.result 84655 .summary) (.transfer 84659) (⟨false, false, none, none, none⟩))

def event84661 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 17⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def event84662 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 29⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (-1)⟩)

def event84663 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71439⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875)

def event84664 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .relation 84663 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩)

def event84665 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 16⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def event84666 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 28⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (-1)⟩)

def event84667 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71439⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875)

def event84668 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .relation 84667 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45761⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩)

def event84669 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 15⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def event84670 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 27⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (-1)⟩)

def event84671 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71439⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875)

def event84672 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .relation 84671 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩)

def event84673 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 14⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def event84674 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 26⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (-1)⟩)

def event84675 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71439⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875)

def event84676 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .relation 84675 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩)

def event84677 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 13⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def event84678 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 25⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (-1)⟩)

def event84679 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71439⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875)

def event84680 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .relation 84679 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩)

def event84681 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 12⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def event84682 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 24⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (-1)⟩)

def event84683 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71439⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875)

def event84684 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .relation 84683 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩)

def event84685 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 11⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def event84686 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 22⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (-1)⟩)

def event84687 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71439⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875)

def event84688 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .relation 84687 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩)

def event84689 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 10⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def event84690 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 21⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (-1)⟩)

def event84691 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71439⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875)

def event84692 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .relation 84691 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩)

def event84693 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 9⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def event84694 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 35⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (-1)⟩)

def event84695 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71439⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875)

def event84696 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .relation 84695 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩)

def event84697 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 8⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def event84698 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 34⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (-1)⟩)

def event84699 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71439⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875)

def event84700 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .relation 84699 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩)

def event84701 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 7⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def event84702 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 33⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (-1)⟩)

def event84703 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71439⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875)

def event84704 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .relation 84703 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩)

def event84705 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 6⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def event84706 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 32⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (-1)⟩)

def event84707 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71439⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875)

def event84708 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .relation 84707 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩)

def event84709 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 5⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def event84710 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 31⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (-1)⟩)

def event84711 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71439⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875)

def event84712 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .relation 84711 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩)

def event84713 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 4⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def event84714 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 30⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (-1)⟩)

def event84715 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71439⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875)

def event84716 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .relation 84715 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩)

def event84717 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 3⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def event84718 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 23⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (-1)⟩)

def event84719 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71439⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875)

def event84720 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .relation 84719 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩)

def event84721 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 2⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def event84722 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 20⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (-1)⟩)

def event84723 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71439⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875)

def event84724 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .relation 84723 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩)

def event84725 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 1⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def event84726 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 19⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (-1)⟩)

def event84727 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71439⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875)

def event84728 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .relation 84727 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩)

def event84729 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 0⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩)

def event84730 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .operator (⟨84655, 18⟩, ⟨75878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (-1)⟩)

def event84731 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71439⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875)

def event84732 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71439⟩⟩, .relation 84731 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩)

def exact84733RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16131⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18980⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45761⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67021⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (-1)⟩]

theorem exact84733RawTermsValid :
    exact84733RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84733 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71439⟩⟩) exact84733RawTerms .large 84658 (.finite 6221717896068416040249469304417135687106560) (some (84660))

def event84734 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68430⟩⟩) 0 ⟨67031⟩ 3568

def event84735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68430⟩⟩) (.authority (.relationPreimageSource ⟨95⟩))

def eventLeaf5280 : Array AnnotatedEvent := #[
  { event := event84480
    frameStart := 84453 },
  { event := event84481
    frameStart := 84453 },
  { event := event84482
    frameStart := 84453 },
  { event := event84483
    frameStart := 84453 },
  { event := event84484
    frameStart := 84453 },
  { event := event84485
    frameStart := 84453 },
  { event := event84486
    frameStart := 84453 },
  { event := event84487
    frameStart := 84453 },
  { event := event84488
    frameStart := 84453 },
  { event := event84489
    frameStart := 84453 },
  { event := event84490
    frameStart := 84453 },
  { event := event84491
    frameStart := 84453 },
  { event := event84492
    frameStart := 84453 },
  { event := event84493
    frameStart := 84453 },
  { event := event84494
    frameStart := 84453 },
  { event := event84495
    frameStart := 84453 }
]

def eventLeaf5281 : Array AnnotatedEvent := #[
  { event := event84496
    frameStart := 84453 },
  { event := event84497
    frameStart := 84453 },
  { event := event84498
    frameStart := 84453 },
  { event := event84499
    frameStart := 84453 },
  { event := event84500
    frameStart := 84453 },
  { event := event84501
    frameStart := 84453 },
  { event := event84502
    frameStart := 84453 },
  { event := event84503
    frameStart := 84453 },
  { event := event84504
    frameStart := 84453 },
  { event := event84505
    frameStart := 84453 },
  { event := event84506
    frameStart := 84453 },
  { event := event84507
    frameStart := 84453 },
  { event := event84508
    frameStart := 84453 },
  { event := event84509
    frameStart := 84453 },
  { event := event84510
    frameStart := 84453 },
  { event := event84511
    frameStart := 84453 }
]

def eventLeaf5282 : Array AnnotatedEvent := #[
  { event := event84512
    frameStart := 84453 },
  { event := event84513
    frameStart := 84453 },
  { event := event84514
    frameStart := 84453 },
  { event := event84515
    frameStart := 84453 },
  { event := event84516
    frameStart := 84453 },
  { event := event84517
    frameStart := 84453 },
  { event := event84518
    frameStart := 84453 },
  { event := event84519
    frameStart := 84453 },
  { event := event84520
    frameStart := 84453 },
  { event := event84521
    frameStart := 84453 },
  { event := event84522
    frameStart := 84453 },
  { event := event84523
    frameStart := 84453 },
  { event := event84524
    frameStart := 84453 },
  { event := event84525
    frameStart := 84453 },
  { event := event84526
    frameStart := 84453 },
  { event := event84527
    frameStart := 84453 }
]

def eventLeaf5283 : Array AnnotatedEvent := #[
  { event := event84528
    frameStart := 84453 },
  { event := event84529
    frameStart := 84453 },
  { event := event84530
    frameStart := 84453 },
  { event := event84531
    frameStart := 84453 },
  { event := event84532
    frameStart := 84453 },
  { event := event84533
    frameStart := 84453 },
  { event := event84534
    frameStart := 84453 },
  { event := event84535
    frameStart := 84453 },
  { event := event84536
    frameStart := 84453 },
  { event := event84537
    frameStart := 84453 },
  { event := event84538
    frameStart := 84453 },
  { event := event84539
    frameStart := 84453 },
  { event := event84540
    frameStart := 84453 },
  { event := event84541
    frameStart := 84453 },
  { event := event84542
    frameStart := 84453 },
  { event := event84543
    frameStart := 84453 }
]

def eventLeaf5284 : Array AnnotatedEvent := #[
  { event := event84544
    frameStart := 84453 },
  { event := event84545
    frameStart := 84453 },
  { event := event84546
    frameStart := 84453 },
  { event := event84547
    frameStart := 84453 },
  { event := event84548
    frameStart := 84453 },
  { event := event84549
    frameStart := 84453 },
  { event := event84550
    frameStart := 84453 },
  { event := event84551
    frameStart := 84453 },
  { event := event84552
    frameStart := 84453 },
  { event := event84553
    frameStart := 84453 },
  { event := event84554
    frameStart := 84453 },
  { event := event84555
    frameStart := 84453 },
  { event := event84556
    frameStart := 84453 },
  { event := event84557
    frameStart := 0 },
  { event := event84558
    frameStart := 0 },
  { event := event84559
    frameStart := 0 }
]

def eventLeaf5285 : Array AnnotatedEvent := #[
  { event := event84560
    frameStart := 0 },
  { event := event84561
    frameStart := 0 },
  { event := event84562
    frameStart := 0 },
  { event := event84563
    frameStart := 0 },
  { event := event84564
    frameStart := 0 },
  { event := event84565
    frameStart := 0 },
  { event := event84566
    frameStart := 0 },
  { event := event84567
    frameStart := 0 },
  { event := event84568
    frameStart := 0 },
  { event := event84569
    frameStart := 0 },
  { event := event84570
    frameStart := 0 },
  { event := event84571
    frameStart := 0 },
  { event := event84572
    frameStart := 0 },
  { event := event84573
    frameStart := 0 },
  { event := event84574
    frameStart := 0 },
  { event := event84575
    frameStart := 0 }
]

def eventLeaf5286 : Array AnnotatedEvent := #[
  { event := event84576
    frameStart := 0 },
  { event := event84577
    frameStart := 0 },
  { event := event84578
    frameStart := 0 },
  { event := event84579
    frameStart := 0 },
  { event := event84580
    frameStart := 0 },
  { event := event84581
    frameStart := 0 },
  { event := event84582
    frameStart := 0 },
  { event := event84583
    frameStart := 0 },
  { event := event84584
    frameStart := 0 },
  { event := event84585
    frameStart := 0 },
  { event := event84586
    frameStart := 0 },
  { event := event84587
    frameStart := 0 },
  { event := event84588
    frameStart := 0 },
  { event := event84589
    frameStart := 0 },
  { event := event84590
    frameStart := 0 },
  { event := event84591
    frameStart := 0 }
]

def eventLeaf5287 : Array AnnotatedEvent := #[
  { event := event84592
    frameStart := 0 },
  { event := event84593
    frameStart := 0 },
  { event := event84594
    frameStart := 0 },
  { event := event84595
    frameStart := 0 },
  { event := event84596
    frameStart := 0 },
  { event := event84597
    frameStart := 0 },
  { event := event84598
    frameStart := 0 },
  { event := event84599
    frameStart := 0 },
  { event := event84600
    frameStart := 0 },
  { event := event84601
    frameStart := 0 },
  { event := event84602
    frameStart := 0 },
  { event := event84603
    frameStart := 0 },
  { event := event84604
    frameStart := 0 },
  { event := event84605
    frameStart := 0 },
  { event := event84606
    frameStart := 0 },
  { event := event84607
    frameStart := 0 }
]

def eventLeaf5288 : Array AnnotatedEvent := #[
  { event := event84608
    frameStart := 0 },
  { event := event84609
    frameStart := 0 },
  { event := event84610
    frameStart := 0 },
  { event := event84611
    frameStart := 0 },
  { event := event84612
    frameStart := 0 },
  { event := event84613
    frameStart := 0 },
  { event := event84614
    frameStart := 0 },
  { event := event84615
    frameStart := 0 },
  { event := event84616
    frameStart := 0 },
  { event := event84617
    frameStart := 0 },
  { event := event84618
    frameStart := 0 },
  { event := event84619
    frameStart := 0 },
  { event := event84620
    frameStart := 0 },
  { event := event84621
    frameStart := 0 },
  { event := event84622
    frameStart := 0 },
  { event := event84623
    frameStart := 0 }
]

def eventLeaf5289 : Array AnnotatedEvent := #[
  { event := event84624
    frameStart := 0 },
  { event := event84625
    frameStart := 0 },
  { event := event84626
    frameStart := 0 },
  { event := event84627
    frameStart := 0 },
  { event := event84628
    frameStart := 0 },
  { event := event84629
    frameStart := 0 },
  { event := event84630
    frameStart := 0 },
  { event := event84631
    frameStart := 0 },
  { event := event84632
    frameStart := 0 },
  { event := event84633
    frameStart := 0 },
  { event := event84634
    frameStart := 0 },
  { event := event84635
    frameStart := 0 },
  { event := event84636
    frameStart := 0 },
  { event := event84637
    frameStart := 0 },
  { event := event84638
    frameStart := 0 },
  { event := event84639
    frameStart := 0 }
]

def eventLeaf5290 : Array AnnotatedEvent := #[
  { event := event84640
    frameStart := 0 },
  { event := event84641
    frameStart := 0 },
  { event := event84642
    frameStart := 0 },
  { event := event84643
    frameStart := 0 },
  { event := event84644
    frameStart := 0 },
  { event := event84645
    frameStart := 0 },
  { event := event84646
    frameStart := 0 },
  { event := event84647
    frameStart := 0 },
  { event := event84648
    frameStart := 0 },
  { event := event84649
    frameStart := 0 },
  { event := event84650
    frameStart := 0 },
  { event := event84651
    frameStart := 0 },
  { event := event84652
    frameStart := 0 },
  { event := event84653
    frameStart := 0 },
  { event := event84654
    frameStart := 0 },
  { event := event84655
    frameStart := 0 }
]

def eventLeaf5291 : Array AnnotatedEvent := #[
  { event := event84656
    frameStart := 0 },
  { event := event84657
    frameStart := 0 },
  { event := event84658
    frameStart := 0 },
  { event := event84659
    frameStart := 0 },
  { event := event84660
    frameStart := 0 },
  { event := event84661
    frameStart := 0 },
  { event := event84662
    frameStart := 0 },
  { event := event84663
    frameStart := 0 },
  { event := event84664
    frameStart := 0 },
  { event := event84665
    frameStart := 0 },
  { event := event84666
    frameStart := 0 },
  { event := event84667
    frameStart := 0 },
  { event := event84668
    frameStart := 0 },
  { event := event84669
    frameStart := 0 },
  { event := event84670
    frameStart := 0 },
  { event := event84671
    frameStart := 0 }
]

def eventLeaf5292 : Array AnnotatedEvent := #[
  { event := event84672
    frameStart := 0 },
  { event := event84673
    frameStart := 0 },
  { event := event84674
    frameStart := 0 },
  { event := event84675
    frameStart := 0 },
  { event := event84676
    frameStart := 0 },
  { event := event84677
    frameStart := 0 },
  { event := event84678
    frameStart := 0 },
  { event := event84679
    frameStart := 0 },
  { event := event84680
    frameStart := 0 },
  { event := event84681
    frameStart := 0 },
  { event := event84682
    frameStart := 0 },
  { event := event84683
    frameStart := 0 },
  { event := event84684
    frameStart := 0 },
  { event := event84685
    frameStart := 0 },
  { event := event84686
    frameStart := 0 },
  { event := event84687
    frameStart := 0 }
]

def eventLeaf5293 : Array AnnotatedEvent := #[
  { event := event84688
    frameStart := 0 },
  { event := event84689
    frameStart := 0 },
  { event := event84690
    frameStart := 0 },
  { event := event84691
    frameStart := 0 },
  { event := event84692
    frameStart := 0 },
  { event := event84693
    frameStart := 0 },
  { event := event84694
    frameStart := 0 },
  { event := event84695
    frameStart := 0 },
  { event := event84696
    frameStart := 0 },
  { event := event84697
    frameStart := 0 },
  { event := event84698
    frameStart := 0 },
  { event := event84699
    frameStart := 0 },
  { event := event84700
    frameStart := 0 },
  { event := event84701
    frameStart := 0 },
  { event := event84702
    frameStart := 0 },
  { event := event84703
    frameStart := 0 }
]

def eventLeaf5294 : Array AnnotatedEvent := #[
  { event := event84704
    frameStart := 0 },
  { event := event84705
    frameStart := 0 },
  { event := event84706
    frameStart := 0 },
  { event := event84707
    frameStart := 0 },
  { event := event84708
    frameStart := 0 },
  { event := event84709
    frameStart := 0 },
  { event := event84710
    frameStart := 0 },
  { event := event84711
    frameStart := 0 },
  { event := event84712
    frameStart := 0 },
  { event := event84713
    frameStart := 0 },
  { event := event84714
    frameStart := 0 },
  { event := event84715
    frameStart := 0 },
  { event := event84716
    frameStart := 0 },
  { event := event84717
    frameStart := 0 },
  { event := event84718
    frameStart := 0 },
  { event := event84719
    frameStart := 0 }
]

def eventLeaf5295 : Array AnnotatedEvent := #[
  { event := event84720
    frameStart := 0 },
  { event := event84721
    frameStart := 0 },
  { event := event84722
    frameStart := 0 },
  { event := event84723
    frameStart := 0 },
  { event := event84724
    frameStart := 0 },
  { event := event84725
    frameStart := 0 },
  { event := event84726
    frameStart := 0 },
  { event := event84727
    frameStart := 0 },
  { event := event84728
    frameStart := 0 },
  { event := event84729
    frameStart := 0 },
  { event := event84730
    frameStart := 0 },
  { event := event84731
    frameStart := 0 },
  { event := event84732
    frameStart := 0 },
  { event := event84733
    frameStart := 0 },
  { event := event84734
    frameStart := 0 },
  { event := event84735
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events330
