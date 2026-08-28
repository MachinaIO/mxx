import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events787

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event201472 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5905⟩⟩) (.identity (.predecessor 1 201471 .coefficient))

def event201473 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5905⟩⟩) (.finite 655360)

def event201474 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15522⟩⟩) 0 ⟨5905⟩ 201473

def event201475 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15522⟩⟩) (.authority (.programFamilyFact))

def exact201476RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15522⟩⟩], []⟩, (1)⟩]

theorem exact201476RawTermsValid :
    exact201476RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201476 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15522⟩⟩) exact201476RawTerms (.finite 2) 201475 .exactZero (none)

def event201477 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12411⟩⟩) 0 ⟨5905⟩ 201473

def event201478 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12411⟩⟩) (.authority (.programFamilyFact))

def exact201479RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12411⟩⟩], []⟩, (1)⟩]

theorem exact201479RawTermsValid :
    exact201479RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201479 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12411⟩⟩) exact201479RawTerms (.finite 2) 201478 .exactZero (none)

def event201480 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15523⟩⟩) 0 ⟨12411⟩ 201479

def event201481 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15523⟩⟩) 1 ⟨15522⟩ 201476

def event201482 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15523⟩⟩) (.product (.predecessor 0 201480 .coefficient) (.predecessor 1 201481 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event201483 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15523⟩⟩, .operator (⟨201479, 0⟩, ⟨201476, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], []⟩, (1)⟩)

def exact201484RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], []⟩, (1)⟩]

theorem exact201484RawTermsValid :
    exact201484RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201484 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15523⟩⟩) exact201484RawTerms (.finite 4) 201482 .exactZero (none)

def event201485 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15524⟩⟩) 0 ⟨15523⟩ 201484

def event201486 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15524⟩⟩) (.identity (.predecessor 0 201485 .coefficient))

def event201487 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15524⟩⟩) (.finite 4)

def event201488 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15804⟩⟩) 0 ⟨15524⟩ 201487

def event201489 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15804⟩⟩) (.authority (.programFamilyFact))

def exact201490RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], []⟩, (1)⟩]

theorem exact201490RawTermsValid :
    exact201490RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201490 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15804⟩⟩) exact201490RawTerms (.finite 2) 201489 .exactZero (none)

def event201491 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15805⟩⟩) 0 ⟨15804⟩ 201490

def event201492 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15805⟩⟩) (.identity (.predecessor 0 201491 .coefficient))

def event201493 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15805⟩⟩) (.finite 2)

def event201494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17017⟩⟩) 0 ⟨15805⟩ 201493

def event201495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17017⟩⟩) (.authority (.programFamilyFact))

def event201496 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17017⟩⟩) (.finite 3720)

def event201497 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event201498 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17019⟩⟩) 0 ⟨7177⟩ 201497

def event201499 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17019⟩⟩) 1 ⟨17017⟩ 201496

def event201500 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17019⟩⟩) (.authority (.operator))

def exact201501RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17019⟩⟩]⟩, (1)⟩]

theorem exact201501RawTermsValid :
    exact201501RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201501 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17019⟩⟩) exact201501RawTerms .large 201500 .exactZero (none)

def event201502 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17817⟩⟩) 0 ⟨17019⟩ 201501

def event201503 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17817⟩⟩) (.authority (.operator))

def exact201504RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17817⟩⟩]⟩, (1)⟩]

theorem exact201504RawTermsValid :
    exact201504RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201504 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17817⟩⟩) exact201504RawTerms (.finite 8192) 201503 .exactZero (none)

def event201505 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event201506 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event201507 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17214⟩⟩) 0 ⟨15805⟩ 201493

def event201508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17214⟩⟩) 1 ⟨136⟩ 201506

def event201509 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17214⟩⟩) (.sum [.predecessor 0 201507 .coefficient, .predecessor 1 201508 .coefficient])

def event201510 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17214⟩⟩) (.finite 2)

def event201511 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17215⟩⟩) 0 ⟨17214⟩ 201510

def event201512 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17215⟩⟩) (.identity (.predecessor 0 201511 .coefficient))

def exact201513RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], []⟩, (1)⟩]

theorem exact201513RawTermsValid :
    exact201513RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201513 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17215⟩⟩) exact201513RawTerms (.finite 2) 201512 .exactZero (none)

def event201514 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact201515RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact201515RawTermsValid :
    exact201515RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201515 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact201515RawTerms .large 201514 .exactZero (none)

def event201516 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17216⟩⟩) 0 ⟨6908⟩ 201515

def event201517 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17216⟩⟩) 1 ⟨17215⟩ 201513

def event201518 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17216⟩⟩) (.product (.predecessor 0 201516 .coefficient) (.predecessor 1 201517 .coefficient) (⟨false, false, none, none, none⟩))

def event201519 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17216⟩⟩, .operator (⟨201515, 0⟩, ⟨201513, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact201520RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact201520RawTermsValid :
    exact201520RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201520 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17216⟩⟩) exact201520RawTerms .large 201518 .exactZero (none)

def event201521 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 201497

def event201522 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact201523RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact201523RawTermsValid :
    exact201523RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201523 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact201523RawTerms .large 201522 .exactZero (none)

def event201524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17217⟩⟩) 0 ⟨7179⟩ 201523

def event201525 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17217⟩⟩) 1 ⟨17216⟩ 201520

def event201526 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17217⟩⟩) (.sum [.predecessor 0 201524 .coefficient, .predecessor 1 201525 .coefficient])

def exact201527RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201527RawTermsValid :
    exact201527RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201527 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17217⟩⟩) exact201527RawTerms .large 201526 .exactZero (none)

def event201528 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17818⟩⟩) 0 ⟨17217⟩ 201527

def event201529 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17818⟩⟩) 1 ⟨17817⟩ 201504

def event201530 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17818⟩⟩) (.product (.predecessor 0 201528 .coefficient) (.predecessor 1 201529 .coefficient) (⟨false, false, none, none, none⟩))

def event201531 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17818⟩⟩, .operator (⟨201527, 0⟩, ⟨201504, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩]⟩, (1)⟩)

def event201532 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17818⟩⟩, .operator (⟨201527, 1⟩, ⟨201504, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩]⟩, (-1)⟩)

def event201533 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17818⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17817⟩⟩) ⟨17019⟩ 201501)

def event201534 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17818⟩⟩, .relation 201533 0, ⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨17019⟩⟩]⟩, (-1)⟩)

def exact201535RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨17019⟩⟩]⟩, (-1)⟩]

theorem exact201535RawTermsValid :
    exact201535RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201535 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17818⟩⟩) exact201535RawTerms .large 201530 .exactZero (none)

def event201536 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16067⟩⟩) 0 ⟨15805⟩ 201493

def event201537 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16067⟩⟩) (.authority (.programFamilyFact))

def exact201538RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16067⟩⟩], []⟩, (1)⟩]

theorem exact201538RawTermsValid :
    exact201538RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201538 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16067⟩⟩) exact201538RawTerms (.finite 43) 201537 .exactZero (none)

def event201539 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16068⟩⟩) 0 ⟨6908⟩ 201515

def event201540 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16068⟩⟩) 1 ⟨16067⟩ 201538

def event201541 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16068⟩⟩) (.product (.predecessor 0 201539 .coefficient) (.predecessor 1 201540 .coefficient) (⟨false, true, none, none, some 1⟩))

def event201542 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16068⟩⟩, .operator (⟨201515, 0⟩, ⟨201538, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact201543RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact201543RawTermsValid :
    exact201543RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201543 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16068⟩⟩) exact201543RawTerms .large 201541 .exactZero (none)

def event201544 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 201497

def event201545 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact201546RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact201546RawTermsValid :
    exact201546RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201546 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact201546RawTerms .large 201545 .exactZero (none)

def event201547 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16069⟩⟩) 0 ⟨7198⟩ 201546

def event201548 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16069⟩⟩) 1 ⟨16068⟩ 201543

def event201549 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16069⟩⟩) (.sum [.predecessor 0 201547 .coefficient, .predecessor 1 201548 .coefficient])

def exact201550RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201550RawTermsValid :
    exact201550RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201550 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16069⟩⟩) exact201550RawTerms .large 201549 .exactZero (none)

def event201551 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17821⟩⟩) 0 ⟨16069⟩ 201550

def event201552 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17821⟩⟩) 1 ⟨17818⟩ 201535

def event201553 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17821⟩⟩) (.sum [.predecessor 0 201551 .coefficient, .predecessor 1 201552 .coefficient])

def exact201554RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨17019⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201554RawTermsValid :
    exact201554RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201554 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17821⟩⟩) exact201554RawTerms .large 201553 .exactZero (none)

def event201555 : Event := .preFoldPolynomial 201554 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨17019⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact201556RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨17019⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event201556 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17821⟩⟩) 201555 exact201556RawTerms .large 201553 .exactZero (none)

def event201557 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15805⟩⟩) ⟨⟨77⟩, ⟨57⟩, ⟨135⟩⟩ ⟨201399, 201557⟩

def event201558 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16639⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16636⟩⟩]⟩) (1) 0 2 (.universal 201557 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16636⟩⟩]⟩) (none) 201556)

def event201559 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16639⟩⟩, .relation 201558 1, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩)

def event201560 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16639⟩⟩, .relation 201558 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩]⟩, (-1)⟩)

def event201561 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16639⟩⟩, .relation 201558 2, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨17019⟩⟩]⟩, (1)⟩)

def event201562 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16639⟩⟩, .relation 201558 3, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact201563RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨17019⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201563RawTermsValid :
    exact201563RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201563 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16639⟩⟩) exact201563RawTerms .large 201395 (.finite 202072841853861888) (some (201397))

def event201564 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17820⟩⟩) 0 ⟨16639⟩ 201563

def event201565 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17820⟩⟩) 1 ⟨17819⟩ 201385

def event201566 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17820⟩⟩) (.sum [.predecessor 0 201564 .coefficient, .predecessor 1 201565 .coefficient])

def event201567 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17820⟩⟩, .operator (⟨201563, 0⟩, ⟨201385, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩]⟩, (1)⟩)

def event201568 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17820⟩⟩, .operator (⟨201563, 2⟩, ⟨201385, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨17019⟩⟩]⟩, (-1)⟩)

def event201569 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17820⟩⟩) (.sum [.result 201563 .summary, .result 201385 .summary])

def exact201570RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201570RawTermsValid :
    exact201570RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201570 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17820⟩⟩) exact201570RawTerms .large 201566 (.finite 32188807212483706889510625476608) (some (201569))

def event201571 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20718⟩⟩) 0 ⟨17820⟩ 201570

def event201572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20718⟩⟩) 1 ⟨20717⟩ 201088

def event201573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20718⟩⟩) (.sum [.predecessor 0 201571 .coefficient, .predecessor 1 201572 .coefficient])

def event201574 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20718⟩⟩) (.sum [.result 201570 .summary, .result 201088 .summary])

def exact201575RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201575RawTermsValid :
    exact201575RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201575 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20718⟩⟩) exact201575RawTerms .large 201573 (.finite 64377712650190257467641695830016) (some (201574))

def event201576 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23938⟩⟩) 0 ⟨20718⟩ 201575

def event201577 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23938⟩⟩) 1 ⟨23937⟩ 200606

def event201578 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23938⟩⟩) (.sum [.predecessor 0 201576 .coefficient, .predecessor 1 201577 .coefficient])

def event201579 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23938⟩⟩) (.sum [.result 201575 .summary, .result 200606 .summary])

def exact201580RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201580RawTermsValid :
    exact201580RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201580 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23938⟩⟩) exact201580RawTerms .large 201578 (.finite 96566716313119651734393211060224) (some (201579))

def event201581 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33958⟩⟩) 0 ⟨23938⟩ 201580

def event201582 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33958⟩⟩) 1 ⟨33957⟩ 200124

def event201583 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33958⟩⟩) (.sum [.predecessor 0 201581 .coefficient, .predecessor 1 201582 .coefficient])

def event201584 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33958⟩⟩) (.sum [.result 201580 .summary, .result 200124 .summary])

def exact201585RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201585RawTermsValid :
    exact201585RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201585 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33958⟩⟩) exact201585RawTerms .large 201583 (.finite 128755916426494733378385616044032) (some (201584))

def event201586 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53018⟩⟩) 0 ⟨33958⟩ 201585

def event201587 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53018⟩⟩) 1 ⟨53017⟩ 199642

def event201588 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53018⟩⟩) (.sum [.predecessor 0 201586 .coefficient, .predecessor 1 201587 .coefficient])

def event201589 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53018⟩⟩) (.sum [.result 201585 .summary, .result 199642 .summary])

def exact201590RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201590RawTermsValid :
    exact201590RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201590 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53018⟩⟩) exact201590RawTerms .large 201588 (.finite 160945509440761189776859800535040) (some (201589))

def event201591 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55998⟩⟩) 0 ⟨53018⟩ 201590

def event201592 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55998⟩⟩) 1 ⟨55997⟩ 199160

def event201593 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55998⟩⟩) (.sum [.predecessor 0 201591 .coefficient, .predecessor 1 201592 .coefficient])

def event201594 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55998⟩⟩) (.sum [.result 201590 .summary, .result 199160 .summary])

def exact201595RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201595RawTermsValid :
    exact201595RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201595 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55998⟩⟩) exact201595RawTerms .large 201593 (.finite 193135298905473333552574874779648) (some (201594))

def event201596 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58978⟩⟩) 0 ⟨55998⟩ 201595

def event201597 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58978⟩⟩) 1 ⟨58977⟩ 198678

def event201598 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58978⟩⟩) (.sum [.predecessor 0 201596 .coefficient, .predecessor 1 201597 .coefficient])

def event201599 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58978⟩⟩) (.sum [.result 201595 .summary, .result 198678 .summary])

def exact201600RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201600RawTermsValid :
    exact201600RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201600 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58978⟩⟩) exact201600RawTerms .large 201598 (.finite 225325481271076852082771728531456) (some (201599))

def event201601 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61958⟩⟩) 0 ⟨58978⟩ 201600

def event201602 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61958⟩⟩) 1 ⟨61957⟩ 198196

def event201603 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61958⟩⟩) (.sum [.predecessor 0 201601 .coefficient, .predecessor 1 201602 .coefficient])

def event201604 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61958⟩⟩) (.sum [.result 201600 .summary, .result 198196 .summary])

def exact201605RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨60139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201605RawTermsValid :
    exact201605RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201605 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61958⟩⟩) exact201605RawTerms .large 201603 (.finite 257515860087126057990209472036864) (some (201604))

def event201606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64938⟩⟩) 0 ⟨61958⟩ 201605

def event201607 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64938⟩⟩) 1 ⟨64937⟩ 197714

def event201608 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64938⟩⟩) (.sum [.predecessor 0 201606 .coefficient, .predecessor 1 201607 .coefficient])

def event201609 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64938⟩⟩) (.sum [.result 201605 .summary, .result 197714 .summary])

def exact201610RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨60139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201610RawTermsValid :
    exact201610RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201610 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64938⟩⟩) exact201610RawTerms .large 201608 (.finite 289706631804066638652128995049472) (some (201609))

def event201611 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70339⟩⟩) 0 ⟨64938⟩ 201610

def event201612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70339⟩⟩) 1 ⟨70338⟩ 197232

def event201613 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70339⟩⟩) (.sum [.predecessor 0 201611 .coefficient, .predecessor 1 201612 .coefficient])

def event201614 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70339⟩⟩) (.sum [.result 201610 .summary, .result 197232 .summary])

def exact201615RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨60139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66741⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201615RawTermsValid :
    exact201615RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201615 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70339⟩⟩) exact201615RawTerms .large 201613 (.finite 321897992872344281445771187322880) (some (201614))

def event201616 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70340⟩⟩) 0 ⟨70339⟩ 201615

def event201617 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70340⟩⟩) 1 ⟨28342⟩ 196750

def event201618 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70340⟩⟩) (.sum [.predecessor 0 201616 .coefficient, .predecessor 1 201617 .coefficient])

def event201619 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70340⟩⟩) (.sum [.result 201615 .summary, .result 196750 .summary])

def exact201620RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨60139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66741⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201620RawTermsValid :
    exact201620RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201620 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70340⟩⟩) exact201620RawTerms .large 201618 (.finite 354089550391067611616654269349888) (some (201619))

def event201621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70341⟩⟩) 0 ⟨70340⟩ 201620

def event201622 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70341⟩⟩) 1 ⟨31022⟩ 196268

def event201623 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70341⟩⟩) (.sum [.predecessor 0 201621 .coefficient, .predecessor 1 201622 .coefficient])

def event201624 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70341⟩⟩) (.sum [.result 201620 .summary, .result 196268 .summary])

def exact201625RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29325⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨60139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66741⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201625RawTermsValid :
    exact201625RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201625 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70341⟩⟩) exact201625RawTerms .large 201623 (.finite 386281697261128003919260020637696) (some (201624))

def event201626 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70342⟩⟩) 0 ⟨70341⟩ 201625

def event201627 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70342⟩⟩) 1 ⟨36682⟩ 195786

def event201628 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70342⟩⟩) (.sum [.predecessor 0 201626 .coefficient, .predecessor 1 201627 .coefficient])

def event201629 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70342⟩⟩) (.sum [.result 201625 .summary, .result 195786 .summary])

def exact201630RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29325⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34989⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨60139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66741⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201630RawTermsValid :
    exact201630RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201630 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70342⟩⟩) exact201630RawTerms .large 201628 (.finite 418474237032079770976347551432704) (some (201629))

def event201631 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70343⟩⟩) 0 ⟨70342⟩ 201630

def event201632 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70343⟩⟩) 1 ⟨39362⟩ 195304

def event201633 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70343⟩⟩) (.sum [.predecessor 0 201631 .coefficient, .predecessor 1 201632 .coefficient])

def event201634 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70343⟩⟩) (.sum [.result 201630 .summary, .result 195304 .summary])

def exact201635RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29325⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34989⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37669⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨60139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66741⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201635RawTermsValid :
    exact201635RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201635 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70343⟩⟩) exact201635RawTerms .large 201633 (.finite 450666973253477225410675971981312) (some (201634))

def event201636 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70344⟩⟩) 0 ⟨70343⟩ 201635

def event201637 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70344⟩⟩) 1 ⟨42042⟩ 194822

def event201638 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70344⟩⟩) (.sum [.predecessor 0 201636 .coefficient, .predecessor 1 201637 .coefficient])

def event201639 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70344⟩⟩) (.sum [.result 201635 .summary, .result 194822 .summary])

def exact201640RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29325⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34989⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37669⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40345⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨60139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66741⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201640RawTermsValid :
    exact201640RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201640 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70344⟩⟩) exact201640RawTerms .large 201638 (.finite 482860102375766054599486172037120) (some (201639))

def event201641 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70345⟩⟩) 0 ⟨70344⟩ 201640

def event201642 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70345⟩⟩) 1 ⟨44722⟩ 194340

def event201643 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70345⟩⟩) (.sum [.predecessor 0 201641 .coefficient, .predecessor 1 201642 .coefficient])

def event201644 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70345⟩⟩) (.sum [.result 201640 .summary, .result 194340 .summary])

def exact201645RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29325⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34989⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37669⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40345⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨43025⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨60139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66741⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201645RawTermsValid :
    exact201645RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201645 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70345⟩⟩) exact201645RawTerms .large 201643 (.finite 515053820849391945920019041353728) (some (201644))

def event201646 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70346⟩⟩) 0 ⟨70345⟩ 201645

def event201647 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70346⟩⟩) 1 ⟨47402⟩ 193858

def event201648 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70346⟩⟩) (.sum [.predecessor 0 201646 .coefficient, .predecessor 1 201647 .coefficient])

def event201649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70346⟩⟩) (.sum [.result 201645 .summary, .result 193858 .summary])

def exact201650RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29325⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34989⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37669⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40345⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨43025⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨45709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨60139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66741⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201650RawTermsValid :
    exact201650RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201650 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70346⟩⟩) exact201650RawTerms .large 201648 (.finite 547248128674354899372274579931136) (some (201649))

def event201651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70347⟩⟩) 0 ⟨70346⟩ 201650

def event201652 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70347⟩⟩) 1 ⟨50082⟩ 193376

def event201653 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70347⟩⟩) (.sum [.predecessor 0 201651 .coefficient, .predecessor 1 201652 .coefficient])

def event201654 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70347⟩⟩) (.sum [.result 201650 .summary, .result 193376 .summary])

def exact201655RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29325⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34989⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37669⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40345⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨43025⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨45709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48389⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨60139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66741⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact201655RawTermsValid :
    exact201655RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event201655 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70347⟩⟩) exact201655RawTerms .large 201653 (.finite 579442632949763540201771008262144) (some (201654))

def event201656 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71299⟩⟩) 0 ⟨70347⟩ 201655

def event201657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71299⟩⟩) 1 ⟨71297⟩ 192878

def event201658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71299⟩⟩) (.product (.predecessor 0 201656 .coefficient) (.predecessor 1 201657 .coefficient) (⟨false, false, none, none, none⟩))

def event201659 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71299⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) [⟨.result 192878 .coefficient, false, none⟩])

def event201660 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71299⟩⟩) (.product (.result 201655 .summary) (.transfer 201659) (⟨false, false, none, none, none⟩))

def event201661 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 17⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (1)⟩)

def event201662 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 29⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48389⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (-1)⟩)

def event201663 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71299⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48389⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 192875)

def event201664 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .relation 201663 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48389⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩, (-1)⟩)

def event201665 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 16⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (1)⟩)

def event201666 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 28⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨45709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (-1)⟩)

def event201667 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71299⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨45709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 192875)

def event201668 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .relation 201667 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨45709⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩, (-1)⟩)

def event201669 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 15⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (1)⟩)

def event201670 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 27⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨43025⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (-1)⟩)

def event201671 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71299⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨43025⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 192875)

def event201672 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .relation 201671 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨43025⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩, (-1)⟩)

def event201673 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 14⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (1)⟩)

def event201674 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 26⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40345⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (-1)⟩)

def event201675 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71299⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40345⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 192875)

def event201676 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .relation 201675 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40345⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩, (-1)⟩)

def event201677 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 13⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (1)⟩)

def event201678 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 25⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37669⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (-1)⟩)

def event201679 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71299⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37669⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 192875)

def event201680 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .relation 201679 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37669⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩, (-1)⟩)

def event201681 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 12⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (1)⟩)

def event201682 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 24⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34989⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (-1)⟩)

def event201683 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71299⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34989⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 192875)

def event201684 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .relation 201683 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34989⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩, (-1)⟩)

def event201685 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 11⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (1)⟩)

def event201686 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 22⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29325⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (-1)⟩)

def event201687 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71299⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29325⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 192875)

def event201688 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .relation 201687 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨29325⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩, (-1)⟩)

def event201689 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 10⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (1)⟩)

def event201690 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 21⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (-1)⟩)

def event201691 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71299⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 192875)

def event201692 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .relation 201691 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩, (-1)⟩)

def event201693 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 9⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (1)⟩)

def event201694 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 35⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66741⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (-1)⟩)

def event201695 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71299⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66741⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 192875)

def event201696 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .relation 201695 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨66741⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩, (-1)⟩)

def event201697 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 8⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (1)⟩)

def event201698 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 34⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (-1)⟩)

def event201699 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71299⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 192875)

def event201700 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .relation 201699 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩, (-1)⟩)

def event201701 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 7⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (1)⟩)

def event201702 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 33⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨60139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (-1)⟩)

def event201703 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71299⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨60139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 192875)

def event201704 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .relation 201703 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨60139⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩, (-1)⟩)

def event201705 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 6⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (1)⟩)

def event201706 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 32⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (-1)⟩)

def event201707 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71299⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 192875)

def event201708 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .relation 201707 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩, (-1)⟩)

def event201709 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 5⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (1)⟩)

def event201710 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 31⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (-1)⟩)

def event201711 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71299⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 192875)

def event201712 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .relation 201711 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨54179⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩, (-1)⟩)

def event201713 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 4⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (1)⟩)

def event201714 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 30⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (-1)⟩)

def event201715 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71299⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 192875)

def event201716 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .relation 201715 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩, (-1)⟩)

def event201717 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 3⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (1)⟩)

def event201718 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 23⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (-1)⟩)

def event201719 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71299⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 192875)

def event201720 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .relation 201719 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨32144⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩, (-1)⟩)

def event201721 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 2⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (1)⟩)

def event201722 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 20⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (-1)⟩)

def event201723 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71299⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 192875)

def event201724 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .relation 201723 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨22124⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩, (-1)⟩)

def event201725 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 1⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (1)⟩)

def event201726 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71299⟩⟩, .operator (⟨201655, 19⟩, ⟨192878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (-1)⟩)

def event201727 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71299⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨18904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 192875)

def eventLeaf12592 : Array AnnotatedEvent := #[
  { event := event201472
    frameStart := 201453 },
  { event := event201473
    frameStart := 201453 },
  { event := event201474
    frameStart := 201453 },
  { event := event201475
    frameStart := 201453 },
  { event := event201476
    frameStart := 201453 },
  { event := event201477
    frameStart := 201453 },
  { event := event201478
    frameStart := 201453 },
  { event := event201479
    frameStart := 201453 },
  { event := event201480
    frameStart := 201453 },
  { event := event201481
    frameStart := 201453 },
  { event := event201482
    frameStart := 201453 },
  { event := event201483
    frameStart := 201453 },
  { event := event201484
    frameStart := 201453 },
  { event := event201485
    frameStart := 201453 },
  { event := event201486
    frameStart := 201453 },
  { event := event201487
    frameStart := 201453 }
]

def eventLeaf12593 : Array AnnotatedEvent := #[
  { event := event201488
    frameStart := 201453 },
  { event := event201489
    frameStart := 201453 },
  { event := event201490
    frameStart := 201453 },
  { event := event201491
    frameStart := 201453 },
  { event := event201492
    frameStart := 201453 },
  { event := event201493
    frameStart := 201453 },
  { event := event201494
    frameStart := 201453 },
  { event := event201495
    frameStart := 201453 },
  { event := event201496
    frameStart := 201453 },
  { event := event201497
    frameStart := 201453 },
  { event := event201498
    frameStart := 201453 },
  { event := event201499
    frameStart := 201453 },
  { event := event201500
    frameStart := 201453 },
  { event := event201501
    frameStart := 201453 },
  { event := event201502
    frameStart := 201453 },
  { event := event201503
    frameStart := 201453 }
]

def eventLeaf12594 : Array AnnotatedEvent := #[
  { event := event201504
    frameStart := 201453 },
  { event := event201505
    frameStart := 201453 },
  { event := event201506
    frameStart := 201453 },
  { event := event201507
    frameStart := 201453 },
  { event := event201508
    frameStart := 201453 },
  { event := event201509
    frameStart := 201453 },
  { event := event201510
    frameStart := 201453 },
  { event := event201511
    frameStart := 201453 },
  { event := event201512
    frameStart := 201453 },
  { event := event201513
    frameStart := 201453 },
  { event := event201514
    frameStart := 201453 },
  { event := event201515
    frameStart := 201453 },
  { event := event201516
    frameStart := 201453 },
  { event := event201517
    frameStart := 201453 },
  { event := event201518
    frameStart := 201453 },
  { event := event201519
    frameStart := 201453 }
]

def eventLeaf12595 : Array AnnotatedEvent := #[
  { event := event201520
    frameStart := 201453 },
  { event := event201521
    frameStart := 201453 },
  { event := event201522
    frameStart := 201453 },
  { event := event201523
    frameStart := 201453 },
  { event := event201524
    frameStart := 201453 },
  { event := event201525
    frameStart := 201453 },
  { event := event201526
    frameStart := 201453 },
  { event := event201527
    frameStart := 201453 },
  { event := event201528
    frameStart := 201453 },
  { event := event201529
    frameStart := 201453 },
  { event := event201530
    frameStart := 201453 },
  { event := event201531
    frameStart := 201453 },
  { event := event201532
    frameStart := 201453 },
  { event := event201533
    frameStart := 201453 },
  { event := event201534
    frameStart := 201453 },
  { event := event201535
    frameStart := 201453 }
]

def eventLeaf12596 : Array AnnotatedEvent := #[
  { event := event201536
    frameStart := 201453 },
  { event := event201537
    frameStart := 201453 },
  { event := event201538
    frameStart := 201453 },
  { event := event201539
    frameStart := 201453 },
  { event := event201540
    frameStart := 201453 },
  { event := event201541
    frameStart := 201453 },
  { event := event201542
    frameStart := 201453 },
  { event := event201543
    frameStart := 201453 },
  { event := event201544
    frameStart := 201453 },
  { event := event201545
    frameStart := 201453 },
  { event := event201546
    frameStart := 201453 },
  { event := event201547
    frameStart := 201453 },
  { event := event201548
    frameStart := 201453 },
  { event := event201549
    frameStart := 201453 },
  { event := event201550
    frameStart := 201453 },
  { event := event201551
    frameStart := 201453 }
]

def eventLeaf12597 : Array AnnotatedEvent := #[
  { event := event201552
    frameStart := 201453 },
  { event := event201553
    frameStart := 201453 },
  { event := event201554
    frameStart := 201453 },
  { event := event201555
    frameStart := 201453 },
  { event := event201556
    frameStart := 201453 },
  { event := event201557
    frameStart := 0 },
  { event := event201558
    frameStart := 0 },
  { event := event201559
    frameStart := 0 },
  { event := event201560
    frameStart := 0 },
  { event := event201561
    frameStart := 0 },
  { event := event201562
    frameStart := 0 },
  { event := event201563
    frameStart := 0 },
  { event := event201564
    frameStart := 0 },
  { event := event201565
    frameStart := 0 },
  { event := event201566
    frameStart := 0 },
  { event := event201567
    frameStart := 0 }
]

def eventLeaf12598 : Array AnnotatedEvent := #[
  { event := event201568
    frameStart := 0 },
  { event := event201569
    frameStart := 0 },
  { event := event201570
    frameStart := 0 },
  { event := event201571
    frameStart := 0 },
  { event := event201572
    frameStart := 0 },
  { event := event201573
    frameStart := 0 },
  { event := event201574
    frameStart := 0 },
  { event := event201575
    frameStart := 0 },
  { event := event201576
    frameStart := 0 },
  { event := event201577
    frameStart := 0 },
  { event := event201578
    frameStart := 0 },
  { event := event201579
    frameStart := 0 },
  { event := event201580
    frameStart := 0 },
  { event := event201581
    frameStart := 0 },
  { event := event201582
    frameStart := 0 },
  { event := event201583
    frameStart := 0 }
]

def eventLeaf12599 : Array AnnotatedEvent := #[
  { event := event201584
    frameStart := 0 },
  { event := event201585
    frameStart := 0 },
  { event := event201586
    frameStart := 0 },
  { event := event201587
    frameStart := 0 },
  { event := event201588
    frameStart := 0 },
  { event := event201589
    frameStart := 0 },
  { event := event201590
    frameStart := 0 },
  { event := event201591
    frameStart := 0 },
  { event := event201592
    frameStart := 0 },
  { event := event201593
    frameStart := 0 },
  { event := event201594
    frameStart := 0 },
  { event := event201595
    frameStart := 0 },
  { event := event201596
    frameStart := 0 },
  { event := event201597
    frameStart := 0 },
  { event := event201598
    frameStart := 0 },
  { event := event201599
    frameStart := 0 }
]

def eventLeaf12600 : Array AnnotatedEvent := #[
  { event := event201600
    frameStart := 0 },
  { event := event201601
    frameStart := 0 },
  { event := event201602
    frameStart := 0 },
  { event := event201603
    frameStart := 0 },
  { event := event201604
    frameStart := 0 },
  { event := event201605
    frameStart := 0 },
  { event := event201606
    frameStart := 0 },
  { event := event201607
    frameStart := 0 },
  { event := event201608
    frameStart := 0 },
  { event := event201609
    frameStart := 0 },
  { event := event201610
    frameStart := 0 },
  { event := event201611
    frameStart := 0 },
  { event := event201612
    frameStart := 0 },
  { event := event201613
    frameStart := 0 },
  { event := event201614
    frameStart := 0 },
  { event := event201615
    frameStart := 0 }
]

def eventLeaf12601 : Array AnnotatedEvent := #[
  { event := event201616
    frameStart := 0 },
  { event := event201617
    frameStart := 0 },
  { event := event201618
    frameStart := 0 },
  { event := event201619
    frameStart := 0 },
  { event := event201620
    frameStart := 0 },
  { event := event201621
    frameStart := 0 },
  { event := event201622
    frameStart := 0 },
  { event := event201623
    frameStart := 0 },
  { event := event201624
    frameStart := 0 },
  { event := event201625
    frameStart := 0 },
  { event := event201626
    frameStart := 0 },
  { event := event201627
    frameStart := 0 },
  { event := event201628
    frameStart := 0 },
  { event := event201629
    frameStart := 0 },
  { event := event201630
    frameStart := 0 },
  { event := event201631
    frameStart := 0 }
]

def eventLeaf12602 : Array AnnotatedEvent := #[
  { event := event201632
    frameStart := 0 },
  { event := event201633
    frameStart := 0 },
  { event := event201634
    frameStart := 0 },
  { event := event201635
    frameStart := 0 },
  { event := event201636
    frameStart := 0 },
  { event := event201637
    frameStart := 0 },
  { event := event201638
    frameStart := 0 },
  { event := event201639
    frameStart := 0 },
  { event := event201640
    frameStart := 0 },
  { event := event201641
    frameStart := 0 },
  { event := event201642
    frameStart := 0 },
  { event := event201643
    frameStart := 0 },
  { event := event201644
    frameStart := 0 },
  { event := event201645
    frameStart := 0 },
  { event := event201646
    frameStart := 0 },
  { event := event201647
    frameStart := 0 }
]

def eventLeaf12603 : Array AnnotatedEvent := #[
  { event := event201648
    frameStart := 0 },
  { event := event201649
    frameStart := 0 },
  { event := event201650
    frameStart := 0 },
  { event := event201651
    frameStart := 0 },
  { event := event201652
    frameStart := 0 },
  { event := event201653
    frameStart := 0 },
  { event := event201654
    frameStart := 0 },
  { event := event201655
    frameStart := 0 },
  { event := event201656
    frameStart := 0 },
  { event := event201657
    frameStart := 0 },
  { event := event201658
    frameStart := 0 },
  { event := event201659
    frameStart := 0 },
  { event := event201660
    frameStart := 0 },
  { event := event201661
    frameStart := 0 },
  { event := event201662
    frameStart := 0 },
  { event := event201663
    frameStart := 0 }
]

def eventLeaf12604 : Array AnnotatedEvent := #[
  { event := event201664
    frameStart := 0 },
  { event := event201665
    frameStart := 0 },
  { event := event201666
    frameStart := 0 },
  { event := event201667
    frameStart := 0 },
  { event := event201668
    frameStart := 0 },
  { event := event201669
    frameStart := 0 },
  { event := event201670
    frameStart := 0 },
  { event := event201671
    frameStart := 0 },
  { event := event201672
    frameStart := 0 },
  { event := event201673
    frameStart := 0 },
  { event := event201674
    frameStart := 0 },
  { event := event201675
    frameStart := 0 },
  { event := event201676
    frameStart := 0 },
  { event := event201677
    frameStart := 0 },
  { event := event201678
    frameStart := 0 },
  { event := event201679
    frameStart := 0 }
]

def eventLeaf12605 : Array AnnotatedEvent := #[
  { event := event201680
    frameStart := 0 },
  { event := event201681
    frameStart := 0 },
  { event := event201682
    frameStart := 0 },
  { event := event201683
    frameStart := 0 },
  { event := event201684
    frameStart := 0 },
  { event := event201685
    frameStart := 0 },
  { event := event201686
    frameStart := 0 },
  { event := event201687
    frameStart := 0 },
  { event := event201688
    frameStart := 0 },
  { event := event201689
    frameStart := 0 },
  { event := event201690
    frameStart := 0 },
  { event := event201691
    frameStart := 0 },
  { event := event201692
    frameStart := 0 },
  { event := event201693
    frameStart := 0 },
  { event := event201694
    frameStart := 0 },
  { event := event201695
    frameStart := 0 }
]

def eventLeaf12606 : Array AnnotatedEvent := #[
  { event := event201696
    frameStart := 0 },
  { event := event201697
    frameStart := 0 },
  { event := event201698
    frameStart := 0 },
  { event := event201699
    frameStart := 0 },
  { event := event201700
    frameStart := 0 },
  { event := event201701
    frameStart := 0 },
  { event := event201702
    frameStart := 0 },
  { event := event201703
    frameStart := 0 },
  { event := event201704
    frameStart := 0 },
  { event := event201705
    frameStart := 0 },
  { event := event201706
    frameStart := 0 },
  { event := event201707
    frameStart := 0 },
  { event := event201708
    frameStart := 0 },
  { event := event201709
    frameStart := 0 },
  { event := event201710
    frameStart := 0 },
  { event := event201711
    frameStart := 0 }
]

def eventLeaf12607 : Array AnnotatedEvent := #[
  { event := event201712
    frameStart := 0 },
  { event := event201713
    frameStart := 0 },
  { event := event201714
    frameStart := 0 },
  { event := event201715
    frameStart := 0 },
  { event := event201716
    frameStart := 0 },
  { event := event201717
    frameStart := 0 },
  { event := event201718
    frameStart := 0 },
  { event := event201719
    frameStart := 0 },
  { event := event201720
    frameStart := 0 },
  { event := event201721
    frameStart := 0 },
  { event := event201722
    frameStart := 0 },
  { event := event201723
    frameStart := 0 },
  { event := event201724
    frameStart := 0 },
  { event := event201725
    frameStart := 0 },
  { event := event201726
    frameStart := 0 },
  { event := event201727
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events787
