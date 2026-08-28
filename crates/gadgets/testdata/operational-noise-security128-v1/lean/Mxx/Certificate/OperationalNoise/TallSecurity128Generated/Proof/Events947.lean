import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events947

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event242432 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57677⟩⟩, .operator (⟨242428, 0⟩, ⟨242426, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57676⟩⟩]⟩, (1)⟩)

def exact242433RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57676⟩⟩]⟩, (1)⟩]

theorem exact242433RawTermsValid :
    exact242433RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242433 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57677⟩⟩) exact242433RawTerms .large 242431 .exactZero (none)

def event242434 : Event := .preFoldPolynomial 242433 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57676⟩⟩]⟩, (1)⟩] .exactZero none

def exact242435RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57676⟩⟩]⟩, (1)⟩]

def event242435 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨57677⟩⟩) 242434 exact242435RawTerms .large 242431 .exactZero (none)

def event242436 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨58855⟩⟩)

def event242437 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event242438 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event242439 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.authority (.operator))

def event242440 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.finite 4)

def event242441 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event242442 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event242443 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event242444 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event242445 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 242444

def event242446 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 242442

def event242447 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 242445 .coefficient) (.value (.predecessor 1 242446 .coefficient)))

def event242448 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event242449 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 0 ⟨392⟩ 242448

def event242450 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 1 ⟨4740⟩ 242440

def event242451 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.sum [.predecessor 0 242449 .coefficient, .predecessor 1 242450 .coefficient])

def event242452 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.finite 655344)

def event242453 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 0 ⟨4742⟩ 242452

def event242454 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 1 ⟨5426⟩ 242438

def event242455 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.identity (.predecessor 1 242454 .coefficient))

def event242456 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.finite 655360)

def event242457 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24986⟩⟩) 0 ⟨5559⟩ 242456

def event242458 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24986⟩⟩) (.authority (.programFamilyFact))

def exact242459RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24986⟩⟩], []⟩, (1)⟩]

theorem exact242459RawTermsValid :
    exact242459RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242459 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24986⟩⟩) exact242459RawTerms (.finite 16) 242458 .exactZero (none)

def event242460 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56451⟩⟩) 0 ⟨5559⟩ 242456

def event242461 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56451⟩⟩) (.authority (.programFamilyFact))

def exact242462RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56451⟩⟩], []⟩, (1)⟩]

theorem exact242462RawTermsValid :
    exact242462RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242462 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56451⟩⟩) exact242462RawTerms (.finite 16) 242461 .exactZero (none)

def event242463 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56452⟩⟩) 0 ⟨56451⟩ 242462

def event242464 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56452⟩⟩) 1 ⟨24986⟩ 242459

def event242465 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56452⟩⟩) (.product (.predecessor 0 242463 .coefficient) (.predecessor 1 242464 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event242466 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56452⟩⟩, .operator (⟨242462, 0⟩, ⟨242459, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24986⟩⟩, ⟨.program ⟨257⟩, ⟨56451⟩⟩], []⟩, (1)⟩)

def exact242467RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24986⟩⟩, ⟨.program ⟨257⟩, ⟨56451⟩⟩], []⟩, (1)⟩]

theorem exact242467RawTermsValid :
    exact242467RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242467 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56452⟩⟩) exact242467RawTerms (.finite 256) 242465 .exactZero (none)

def event242468 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56453⟩⟩) 0 ⟨56452⟩ 242467

def event242469 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56453⟩⟩) (.identity (.predecessor 0 242468 .coefficient))

def event242470 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56453⟩⟩) (.finite 256)

def event242471 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56832⟩⟩) 0 ⟨56453⟩ 242470

def event242472 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56832⟩⟩) (.authority (.programFamilyFact))

def exact242473RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56832⟩⟩], []⟩, (1)⟩]

theorem exact242473RawTermsValid :
    exact242473RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242473 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56832⟩⟩) exact242473RawTerms (.finite 16) 242472 .exactZero (none)

def event242474 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56833⟩⟩) 0 ⟨56832⟩ 242473

def event242475 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56833⟩⟩) (.identity (.predecessor 0 242474 .coefficient))

def event242476 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56833⟩⟩) (.finite 16)

def event242477 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58101⟩⟩) 0 ⟨56833⟩ 242476

def event242478 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58101⟩⟩) (.authority (.programFamilyFact))

def event242479 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨58101⟩⟩) (.finite 3720)

def event242480 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event242481 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58103⟩⟩) 0 ⟨7177⟩ 242480

def event242482 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58103⟩⟩) 1 ⟨58101⟩ 242479

def event242483 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58103⟩⟩) (.authority (.operator))

def exact242484RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58103⟩⟩]⟩, (1)⟩]

theorem exact242484RawTermsValid :
    exact242484RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242484 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58103⟩⟩) exact242484RawTerms .large 242483 .exactZero (none)

def event242485 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58850⟩⟩) 0 ⟨58103⟩ 242484

def event242486 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58850⟩⟩) (.authority (.operator))

def exact242487RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58850⟩⟩]⟩, (1)⟩]

theorem exact242487RawTermsValid :
    exact242487RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242487 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58850⟩⟩) exact242487RawTerms (.finite 8192) 242486 .exactZero (none)

def event242488 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event242489 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event242490 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58318⟩⟩) 0 ⟨56833⟩ 242476

def event242491 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58318⟩⟩) 1 ⟨136⟩ 242489

def event242492 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58318⟩⟩) (.sum [.predecessor 0 242490 .coefficient, .predecessor 1 242491 .coefficient])

def event242493 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨58318⟩⟩) (.finite 16)

def event242494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58319⟩⟩) 0 ⟨58318⟩ 242493

def event242495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58319⟩⟩) (.identity (.predecessor 0 242494 .coefficient))

def exact242496RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56832⟩⟩], []⟩, (1)⟩]

theorem exact242496RawTermsValid :
    exact242496RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242496 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58319⟩⟩) exact242496RawTerms (.finite 16) 242495 .exactZero (none)

def event242497 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact242498RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact242498RawTermsValid :
    exact242498RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242498 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact242498RawTerms .large 242497 .exactZero (none)

def event242499 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58320⟩⟩) 0 ⟨6908⟩ 242498

def event242500 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58320⟩⟩) 1 ⟨58319⟩ 242496

def event242501 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58320⟩⟩) (.product (.predecessor 0 242499 .coefficient) (.predecessor 1 242500 .coefficient) (⟨false, false, none, none, none⟩))

def event242502 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58320⟩⟩, .operator (⟨242498, 0⟩, ⟨242496, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact242503RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact242503RawTermsValid :
    exact242503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242503 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58320⟩⟩) exact242503RawTerms .large 242501 .exactZero (none)

def event242504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7185⟩⟩) 0 ⟨7177⟩ 242480

def event242505 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7185⟩⟩) (.authority (.operator))

def exact242506RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩, (1)⟩]

theorem exact242506RawTermsValid :
    exact242506RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242506 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7185⟩⟩) exact242506RawTerms .large 242505 .exactZero (none)

def event242507 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58321⟩⟩) 0 ⟨7185⟩ 242506

def event242508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58321⟩⟩) 1 ⟨58320⟩ 242503

def event242509 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58321⟩⟩) (.sum [.predecessor 0 242507 .coefficient, .predecessor 1 242508 .coefficient])

def exact242510RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact242510RawTermsValid :
    exact242510RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242510 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58321⟩⟩) exact242510RawTerms .large 242509 .exactZero (none)

def event242511 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58851⟩⟩) 0 ⟨58321⟩ 242510

def event242512 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58851⟩⟩) 1 ⟨58850⟩ 242487

def event242513 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58851⟩⟩) (.product (.predecessor 0 242511 .coefficient) (.predecessor 1 242512 .coefficient) (⟨false, false, none, none, none⟩))

def event242514 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58851⟩⟩, .operator (⟨242510, 0⟩, ⟨242487, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩]⟩, (1)⟩)

def event242515 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58851⟩⟩, .operator (⟨242510, 1⟩, ⟨242487, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩]⟩, (-1)⟩)

def event242516 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨58851⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58850⟩⟩) ⟨58103⟩ 242484)

def event242517 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58851⟩⟩, .relation 242516 0, ⟨[⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨58103⟩⟩]⟩, (-1)⟩)

def exact242518RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨58103⟩⟩]⟩, (-1)⟩]

theorem exact242518RawTermsValid :
    exact242518RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242518 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58851⟩⟩) exact242518RawTerms .large 242513 .exactZero (none)

def event242519 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57083⟩⟩) 0 ⟨56833⟩ 242476

def event242520 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57083⟩⟩) (.authority (.programFamilyFact))

def exact242521RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨57083⟩⟩], []⟩, (1)⟩]

theorem exact242521RawTermsValid :
    exact242521RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242521 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57083⟩⟩) exact242521RawTerms (.finite 60) 242520 .exactZero (none)

def event242522 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57085⟩⟩) 0 ⟨6908⟩ 242498

def event242523 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57085⟩⟩) 1 ⟨57083⟩ 242521

def event242524 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57085⟩⟩) (.product (.predecessor 0 242522 .coefficient) (.predecessor 1 242523 .coefficient) (⟨false, true, none, none, some 1⟩))

def event242525 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57085⟩⟩, .operator (⟨242498, 0⟩, ⟨242521, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨57083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact242526RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨57083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact242526RawTermsValid :
    exact242526RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242526 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57085⟩⟩) exact242526RawTerms .large 242524 .exactZero (none)

def event242527 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7210⟩⟩) 0 ⟨7177⟩ 242480

def event242528 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7210⟩⟩) (.authority (.operator))

def exact242529RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩]

theorem exact242529RawTermsValid :
    exact242529RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242529 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7210⟩⟩) exact242529RawTerms .large 242528 .exactZero (none)

def event242530 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57086⟩⟩) 0 ⟨7210⟩ 242529

def event242531 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57086⟩⟩) 1 ⟨57085⟩ 242526

def event242532 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57086⟩⟩) (.sum [.predecessor 0 242530 .coefficient, .predecessor 1 242531 .coefficient])

def exact242533RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact242533RawTermsValid :
    exact242533RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242533 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57086⟩⟩) exact242533RawTerms .large 242532 .exactZero (none)

def event242534 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58855⟩⟩) 0 ⟨57086⟩ 242533

def event242535 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58855⟩⟩) 1 ⟨58851⟩ 242518

def event242536 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58855⟩⟩) (.sum [.predecessor 0 242534 .coefficient, .predecessor 1 242535 .coefficient])

def exact242537RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨58103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact242537RawTermsValid :
    exact242537RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242537 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58855⟩⟩) exact242537RawTerms .large 242536 .exactZero (none)

def event242538 : Event := .preFoldPolynomial 242537 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨58103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact242539RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨58103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨57083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event242539 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨58855⟩⟩) 242538 exact242539RawTerms .large 242536 .exactZero (none)

def event242540 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨56833⟩⟩) ⟨⟨89⟩, ⟨70⟩, ⟨135⟩⟩ ⟨242382, 242540⟩

def event242541 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨57679⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57676⟩⟩]⟩) (1) 0 2 (.universal 242540 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57676⟩⟩]⟩) (none) 242539)

def event242542 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57679⟩⟩, .relation 242541 1, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩)

def event242543 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57679⟩⟩, .relation 242541 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩]⟩, (-1)⟩)

def event242544 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57679⟩⟩, .relation 242541 2, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨58103⟩⟩]⟩, (1)⟩)

def event242545 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57679⟩⟩, .relation 242541 3, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨57083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact242546RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨58103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨57083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact242546RawTermsValid :
    exact242546RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242546 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57679⟩⟩) exact242546RawTerms .large 242378 (.finite 202072841853861888) (some (242380))

def event242547 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58853⟩⟩) 0 ⟨57679⟩ 242546

def event242548 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58853⟩⟩) 1 ⟨58852⟩ 242368

def event242549 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58853⟩⟩) (.sum [.predecessor 0 242547 .coefficient, .predecessor 1 242548 .coefficient])

def event242550 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58853⟩⟩, .operator (⟨242546, 0⟩, ⟨242368, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩]⟩, (1)⟩)

def event242551 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58853⟩⟩, .operator (⟨242546, 2⟩, ⟨242368, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨58103⟩⟩]⟩, (-1)⟩)

def event242552 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58853⟩⟩) (.sum [.result 242546 .summary, .result 242368 .summary])

def exact242553RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨57083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact242553RawTermsValid :
    exact242553RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242553 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58853⟩⟩) exact242553RawTerms .large 242549 (.finite 32190182365603518530196853751808) (some (242552))

def event242554 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55121⟩⟩) 0 ⟨53853⟩ 11607

def event242555 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55121⟩⟩) (.authority (.programFamilyFact))

def event242556 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨55121⟩⟩) (.finite 3720)

def event242557 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55123⟩⟩) 0 ⟨7177⟩ 15500

def event242558 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55123⟩⟩) 1 ⟨55121⟩ 242556

def event242559 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55123⟩⟩) (.authority (.operator))

def exact242560RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55123⟩⟩]⟩, (1)⟩]

theorem exact242560RawTermsValid :
    exact242560RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242560 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55123⟩⟩) exact242560RawTerms .large 242559 .exactZero (none)

def event242561 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55870⟩⟩) 0 ⟨55123⟩ 242560

def event242562 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55870⟩⟩) (.authority (.operator))

def exact242563RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55870⟩⟩]⟩, (1)⟩]

theorem exact242563RawTermsValid :
    exact242563RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242563 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55870⟩⟩) exact242563RawTerms (.finite 8192) 242562 .exactZero (none)

def event242564 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54976⟩⟩) 0 ⟨53473⟩ 11601

def event242565 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54976⟩⟩) (.authority (.programFamilyFact))

def event242566 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨54976⟩⟩) (.finite 3720)

def event242567 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54977⟩⟩) 0 ⟨7177⟩ 15500

def event242568 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54977⟩⟩) 1 ⟨54976⟩ 242566

def event242569 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54977⟩⟩) (.authority (.operator))

def exact242570RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54977⟩⟩]⟩, (1)⟩]

theorem exact242570RawTermsValid :
    exact242570RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242570 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54977⟩⟩) exact242570RawTerms .large 242569 .exactZero (none)

def event242571 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55477⟩⟩) 0 ⟨54977⟩ 242570

def event242572 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55477⟩⟩) (.authority (.operator))

def exact242573RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55477⟩⟩]⟩, (1)⟩]

theorem exact242573RawTermsValid :
    exact242573RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242573 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55477⟩⟩) exact242573RawTerms (.finite 8192) 242572 .exactZero (none)

def event242574 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24747⟩⟩) 0 ⟨24746⟩ 11590

def event242575 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24747⟩⟩) 1 ⟨6934⟩ 236778

def event242576 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24747⟩⟩) (.tensor (.predecessor 0 242574 .coefficient) (.predecessor 1 242575 .coefficient) true false)

def event242577 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24747⟩⟩, .operator (⟨11590, 0⟩, ⟨236778, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact242578RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact242578RawTermsValid :
    exact242578RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242578 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24747⟩⟩) exact242578RawTerms .large 242576 .exactZero (none)

def event242579 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8350⟩⟩) 0 ⟨5561⟩ 236648

def event242580 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8350⟩⟩) 1 ⟨7272⟩ 23092

def event242581 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8350⟩⟩) (.product (.predecessor 0 242579 .coefficient) (.predecessor 1 242580 .coefficient) (⟨false, false, none, none, none⟩))

def event242582 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8350⟩⟩, .operator (⟨236648, 0⟩, ⟨23092, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩)

def exact242583RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩]

theorem exact242583RawTermsValid :
    exact242583RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242583 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8350⟩⟩) exact242583RawTerms .large 242581 .exactZero (none)

def event242584 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24748⟩⟩) 0 ⟨8350⟩ 242583

def event242585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24748⟩⟩) 1 ⟨24747⟩ 242578

def event242586 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24748⟩⟩) (.sum [.predecessor 0 242584 .coefficient, .predecessor 1 242585 .coefficient])

def exact242587RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact242587RawTermsValid :
    exact242587RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242587 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24748⟩⟩) exact242587RawTerms .large 242586 .exactZero (none)

def event242588 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24749⟩⟩) 0 ⟨24748⟩ 242587

def event242589 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24749⟩⟩) 1 ⟨98⟩ 23084

def event242590 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24749⟩⟩) (.sum [.predecessor 0 242588 .coefficient, .predecessor 1 242589 .coefficient])

def event242591 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24749⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨98⟩⟩]⟩) [⟨.result 23084 .coefficient, false, none⟩])

def event242592 : Event := .survivorFold (1) 242591

def exact242593RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact242593RawTermsValid :
    exact242593RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242593 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24749⟩⟩) exact242593RawTerms .large 242590 (.finite 26) (some (242591))

def event242594 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53474⟩⟩) 0 ⟨24749⟩ 242593

def event242595 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53474⟩⟩) 1 ⟨53471⟩ 11593

def event242596 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53474⟩⟩) (.product (.predecessor 0 242594 .coefficient) (.predecessor 1 242595 .coefficient) (⟨false, true, none, none, some 1⟩))

def event242597 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53474⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨53471⟩⟩], []⟩) [⟨.result 11593 .coefficient, true, some 1⟩])

def event242598 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53474⟩⟩) (.product (.result 242593 .summary) (.transfer 242597) (⟨false, false, none, none, none⟩))

def event242599 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53474⟩⟩, .operator (⟨242593, 1⟩, ⟨11593, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event242600 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53474⟩⟩, .operator (⟨242593, 0⟩, ⟨11593, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩)

def exact242601RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩]

theorem exact242601RawTermsValid :
    exact242601RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242601 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53474⟩⟩) exact242601RawTerms .large 242596 (.finite 10223616) (some (242598))

def event242602 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53475⟩⟩) 0 ⟨53471⟩ 11593

def event242603 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53475⟩⟩) 1 ⟨6934⟩ 236778

def event242604 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53475⟩⟩) (.tensor (.predecessor 0 242602 .coefficient) (.predecessor 1 242603 .coefficient) true false)

def event242605 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53475⟩⟩, .operator (⟨11593, 0⟩, ⟨236778, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact242606RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact242606RawTermsValid :
    exact242606RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242606 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53475⟩⟩) exact242606RawTerms .large 242604 .exactZero (none)

def event242607 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8367⟩⟩) 0 ⟨5561⟩ 236648

def event242608 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8367⟩⟩) 1 ⟨7289⟩ 23133

def event242609 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8367⟩⟩) (.product (.predecessor 0 242607 .coefficient) (.predecessor 1 242608 .coefficient) (⟨false, false, none, none, none⟩))

def event242610 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8367⟩⟩, .operator (⟨236648, 0⟩, ⟨23133, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩]⟩, (1)⟩)

def exact242611RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩]⟩, (1)⟩]

theorem exact242611RawTermsValid :
    exact242611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242611 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8367⟩⟩) exact242611RawTerms .large 242609 .exactZero (none)

def event242612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53476⟩⟩) 0 ⟨8367⟩ 242611

def event242613 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53476⟩⟩) 1 ⟨53475⟩ 242606

def event242614 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53476⟩⟩) (.sum [.predecessor 0 242612 .coefficient, .predecessor 1 242613 .coefficient])

def exact242615RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact242615RawTermsValid :
    exact242615RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242615 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53476⟩⟩) exact242615RawTerms .large 242614 .exactZero (none)

def event242616 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53477⟩⟩) 0 ⟨53476⟩ 242615

def event242617 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53477⟩⟩) 1 ⟨115⟩ 23125

def event242618 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53477⟩⟩) (.sum [.predecessor 0 242616 .coefficient, .predecessor 1 242617 .coefficient])

def event242619 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53477⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨115⟩⟩]⟩) [⟨.result 23125 .coefficient, false, none⟩])

def event242620 : Event := .survivorFold (1) 242619

def exact242621RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact242621RawTermsValid :
    exact242621RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242621 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53477⟩⟩) exact242621RawTerms .large 242618 (.finite 26) (some (242619))

def event242622 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53478⟩⟩) 0 ⟨53477⟩ 242621

def event242623 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53478⟩⟩) 1 ⟨9530⟩ 23122

def event242624 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53478⟩⟩) (.product (.predecessor 0 242622 .coefficient) (.predecessor 1 242623 .coefficient) (⟨false, false, none, none, none⟩))

def event242625 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53478⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) [⟨.result 23118 .coefficient, false, none⟩])

def event242626 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53478⟩⟩) (.product (.result 242621 .summary) (.transfer 242625) (⟨false, false, none, none, none⟩))

def event242627 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53478⟩⟩, .operator (⟨242621, 1⟩, ⟨23122, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (-1)⟩)

def event242628 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨53478⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9529⟩⟩) ⟨7272⟩ 23092)

def event242629 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53478⟩⟩, .relation 242628 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (-1)⟩)

def event242630 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53478⟩⟩, .operator (⟨242621, 0⟩, ⟨23122, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (1)⟩)

def exact242631RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (-1)⟩]

theorem exact242631RawTermsValid :
    exact242631RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242631 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53478⟩⟩) exact242631RawTerms .large 242624 (.finite 279172874240) (some (242626))

def event242632 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53479⟩⟩) 0 ⟨53478⟩ 242631

def event242633 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53479⟩⟩) 1 ⟨53474⟩ 242601

def event242634 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53479⟩⟩) (.sum [.predecessor 0 242632 .coefficient, .predecessor 1 242633 .coefficient])

def event242635 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53479⟩⟩, .operator (⟨242631, 1⟩, ⟨242601, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩)

def event242636 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53479⟩⟩) (.sum [.result 242631 .summary, .result 242601 .summary])

def exact242637RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact242637RawTermsValid :
    exact242637RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242637 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53479⟩⟩) exact242637RawTerms .large 242634 (.finite 279183097856) (some (242636))

def event242638 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55478⟩⟩) 0 ⟨53479⟩ 242637

def event242639 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55478⟩⟩) 1 ⟨55477⟩ 242573

def event242640 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55478⟩⟩) (.product (.predecessor 0 242638 .coefficient) (.predecessor 1 242639 .coefficient) (⟨false, false, none, none, none⟩))

def event242641 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55478⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨55477⟩⟩]⟩) [⟨.result 242573 .coefficient, false, none⟩])

def event242642 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55478⟩⟩) (.product (.result 242637 .summary) (.transfer 242641) (⟨false, false, none, none, none⟩))

def event242643 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55478⟩⟩, .operator (⟨242637, 1⟩, ⟨242573, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩]⟩, (-1)⟩)

def event242644 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨55478⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55477⟩⟩) ⟨54977⟩ 242570)

def event242645 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55478⟩⟩, .relation 242644 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨54977⟩⟩]⟩, (-1)⟩)

def event242646 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55478⟩⟩, .operator (⟨242637, 0⟩, ⟨242573, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩]⟩, (1)⟩)

def exact242647RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55477⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24746⟩⟩, ⟨.program ⟨257⟩, ⟨53471⟩⟩], [⟨.program ⟨257⟩, ⟨54977⟩⟩]⟩, (-1)⟩]

theorem exact242647RawTermsValid :
    exact242647RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242647 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55478⟩⟩) exact242647RawTerms .large 242640 (.finite 2997705687218719293440) (some (242642))

def event242648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54409⟩⟩) 0 ⟨53473⟩ 11601

def event242649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54409⟩⟩) (.authority (.relationPreimageSource ⟨41⟩))

def exact242650RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54409⟩⟩]⟩, (1)⟩]

theorem exact242650RawTermsValid :
    exact242650RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242650 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54409⟩⟩) exact242650RawTerms (.finite 5647228698) 242649 .exactZero (none)

def event242651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54411⟩⟩) 0 ⟨54409⟩ 242650

def event242652 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54411⟩⟩) 1 ⟨2370⟩ 4

def event242653 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54411⟩⟩) (.scale (.predecessor 0 242651 .coefficient) (.value (.predecessor 1 242652 .coefficient)))

def exact242654RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54409⟩⟩]⟩, (1)⟩]

theorem exact242654RawTermsValid :
    exact242654RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242654 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54411⟩⟩) exact242654RawTerms (.finite 5647228698) 242653 .exactZero (none)

def event242655 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54412⟩⟩) 0 ⟨5563⟩ 236870

def event242656 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54412⟩⟩) 1 ⟨54411⟩ 242654

def event242657 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54412⟩⟩) (.product (.predecessor 0 242655 .coefficient) (.predecessor 1 242656 .coefficient) (⟨false, false, none, none, none⟩))

def event242658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54412⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨54409⟩⟩]⟩) [⟨.result 242650 .coefficient, false, none⟩])

def event242659 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54412⟩⟩) (.product (.result 236870 .summary) (.transfer 242658) (⟨false, false, none, none, none⟩))

def event242660 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54412⟩⟩, .operator (⟨236870, 0⟩, ⟨242654, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54409⟩⟩]⟩, (1)⟩)

def event242661 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨54410⟩⟩)

def event242662 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event242663 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event242664 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.authority (.operator))

def event242665 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.finite 4)

def event242666 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event242667 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event242668 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event242669 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event242670 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 242669

def event242671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 242667

def event242672 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 242670 .coefficient) (.value (.predecessor 1 242671 .coefficient)))

def event242673 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event242674 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 0 ⟨392⟩ 242673

def event242675 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 1 ⟨4740⟩ 242665

def event242676 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.sum [.predecessor 0 242674 .coefficient, .predecessor 1 242675 .coefficient])

def event242677 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.finite 655344)

def event242678 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 0 ⟨4742⟩ 242677

def event242679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 1 ⟨5426⟩ 242663

def event242680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.identity (.predecessor 1 242679 .coefficient))

def event242681 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.finite 655360)

def event242682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24746⟩⟩) 0 ⟨5559⟩ 242681

def event242683 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24746⟩⟩) (.authority (.programFamilyFact))

def exact242684RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24746⟩⟩], []⟩, (1)⟩]

theorem exact242684RawTermsValid :
    exact242684RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242684 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24746⟩⟩) exact242684RawTerms (.finite 12) 242683 .exactZero (none)

def event242685 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53471⟩⟩) 0 ⟨5559⟩ 242681

def event242686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53471⟩⟩) (.authority (.programFamilyFact))

def exact242687RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53471⟩⟩], []⟩, (1)⟩]

theorem exact242687RawTermsValid :
    exact242687RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event242687 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53471⟩⟩) exact242687RawTerms (.finite 12) 242686 .exactZero (none)

def eventLeaf15152 : Array AnnotatedEvent := #[
  { event := event242432
    frameStart := 242382 },
  { event := event242433
    frameStart := 242382 },
  { event := event242434
    frameStart := 242382 },
  { event := event242435
    frameStart := 242382 },
  { event := event242436
    frameStart := 242436 },
  { event := event242437
    frameStart := 242436 },
  { event := event242438
    frameStart := 242436 },
  { event := event242439
    frameStart := 242436 },
  { event := event242440
    frameStart := 242436 },
  { event := event242441
    frameStart := 242436 },
  { event := event242442
    frameStart := 242436 },
  { event := event242443
    frameStart := 242436 },
  { event := event242444
    frameStart := 242436 },
  { event := event242445
    frameStart := 242436 },
  { event := event242446
    frameStart := 242436 },
  { event := event242447
    frameStart := 242436 }
]

def eventLeaf15153 : Array AnnotatedEvent := #[
  { event := event242448
    frameStart := 242436 },
  { event := event242449
    frameStart := 242436 },
  { event := event242450
    frameStart := 242436 },
  { event := event242451
    frameStart := 242436 },
  { event := event242452
    frameStart := 242436 },
  { event := event242453
    frameStart := 242436 },
  { event := event242454
    frameStart := 242436 },
  { event := event242455
    frameStart := 242436 },
  { event := event242456
    frameStart := 242436 },
  { event := event242457
    frameStart := 242436 },
  { event := event242458
    frameStart := 242436 },
  { event := event242459
    frameStart := 242436 },
  { event := event242460
    frameStart := 242436 },
  { event := event242461
    frameStart := 242436 },
  { event := event242462
    frameStart := 242436 },
  { event := event242463
    frameStart := 242436 }
]

def eventLeaf15154 : Array AnnotatedEvent := #[
  { event := event242464
    frameStart := 242436 },
  { event := event242465
    frameStart := 242436 },
  { event := event242466
    frameStart := 242436 },
  { event := event242467
    frameStart := 242436 },
  { event := event242468
    frameStart := 242436 },
  { event := event242469
    frameStart := 242436 },
  { event := event242470
    frameStart := 242436 },
  { event := event242471
    frameStart := 242436 },
  { event := event242472
    frameStart := 242436 },
  { event := event242473
    frameStart := 242436 },
  { event := event242474
    frameStart := 242436 },
  { event := event242475
    frameStart := 242436 },
  { event := event242476
    frameStart := 242436 },
  { event := event242477
    frameStart := 242436 },
  { event := event242478
    frameStart := 242436 },
  { event := event242479
    frameStart := 242436 }
]

def eventLeaf15155 : Array AnnotatedEvent := #[
  { event := event242480
    frameStart := 242436 },
  { event := event242481
    frameStart := 242436 },
  { event := event242482
    frameStart := 242436 },
  { event := event242483
    frameStart := 242436 },
  { event := event242484
    frameStart := 242436 },
  { event := event242485
    frameStart := 242436 },
  { event := event242486
    frameStart := 242436 },
  { event := event242487
    frameStart := 242436 },
  { event := event242488
    frameStart := 242436 },
  { event := event242489
    frameStart := 242436 },
  { event := event242490
    frameStart := 242436 },
  { event := event242491
    frameStart := 242436 },
  { event := event242492
    frameStart := 242436 },
  { event := event242493
    frameStart := 242436 },
  { event := event242494
    frameStart := 242436 },
  { event := event242495
    frameStart := 242436 }
]

def eventLeaf15156 : Array AnnotatedEvent := #[
  { event := event242496
    frameStart := 242436 },
  { event := event242497
    frameStart := 242436 },
  { event := event242498
    frameStart := 242436 },
  { event := event242499
    frameStart := 242436 },
  { event := event242500
    frameStart := 242436 },
  { event := event242501
    frameStart := 242436 },
  { event := event242502
    frameStart := 242436 },
  { event := event242503
    frameStart := 242436 },
  { event := event242504
    frameStart := 242436 },
  { event := event242505
    frameStart := 242436 },
  { event := event242506
    frameStart := 242436 },
  { event := event242507
    frameStart := 242436 },
  { event := event242508
    frameStart := 242436 },
  { event := event242509
    frameStart := 242436 },
  { event := event242510
    frameStart := 242436 },
  { event := event242511
    frameStart := 242436 }
]

def eventLeaf15157 : Array AnnotatedEvent := #[
  { event := event242512
    frameStart := 242436 },
  { event := event242513
    frameStart := 242436 },
  { event := event242514
    frameStart := 242436 },
  { event := event242515
    frameStart := 242436 },
  { event := event242516
    frameStart := 242436 },
  { event := event242517
    frameStart := 242436 },
  { event := event242518
    frameStart := 242436 },
  { event := event242519
    frameStart := 242436 },
  { event := event242520
    frameStart := 242436 },
  { event := event242521
    frameStart := 242436 },
  { event := event242522
    frameStart := 242436 },
  { event := event242523
    frameStart := 242436 },
  { event := event242524
    frameStart := 242436 },
  { event := event242525
    frameStart := 242436 },
  { event := event242526
    frameStart := 242436 },
  { event := event242527
    frameStart := 242436 }
]

def eventLeaf15158 : Array AnnotatedEvent := #[
  { event := event242528
    frameStart := 242436 },
  { event := event242529
    frameStart := 242436 },
  { event := event242530
    frameStart := 242436 },
  { event := event242531
    frameStart := 242436 },
  { event := event242532
    frameStart := 242436 },
  { event := event242533
    frameStart := 242436 },
  { event := event242534
    frameStart := 242436 },
  { event := event242535
    frameStart := 242436 },
  { event := event242536
    frameStart := 242436 },
  { event := event242537
    frameStart := 242436 },
  { event := event242538
    frameStart := 242436 },
  { event := event242539
    frameStart := 242436 },
  { event := event242540
    frameStart := 0 },
  { event := event242541
    frameStart := 0 },
  { event := event242542
    frameStart := 0 },
  { event := event242543
    frameStart := 0 }
]

def eventLeaf15159 : Array AnnotatedEvent := #[
  { event := event242544
    frameStart := 0 },
  { event := event242545
    frameStart := 0 },
  { event := event242546
    frameStart := 0 },
  { event := event242547
    frameStart := 0 },
  { event := event242548
    frameStart := 0 },
  { event := event242549
    frameStart := 0 },
  { event := event242550
    frameStart := 0 },
  { event := event242551
    frameStart := 0 },
  { event := event242552
    frameStart := 0 },
  { event := event242553
    frameStart := 0 },
  { event := event242554
    frameStart := 0 },
  { event := event242555
    frameStart := 0 },
  { event := event242556
    frameStart := 0 },
  { event := event242557
    frameStart := 0 },
  { event := event242558
    frameStart := 0 },
  { event := event242559
    frameStart := 0 }
]

def eventLeaf15160 : Array AnnotatedEvent := #[
  { event := event242560
    frameStart := 0 },
  { event := event242561
    frameStart := 0 },
  { event := event242562
    frameStart := 0 },
  { event := event242563
    frameStart := 0 },
  { event := event242564
    frameStart := 0 },
  { event := event242565
    frameStart := 0 },
  { event := event242566
    frameStart := 0 },
  { event := event242567
    frameStart := 0 },
  { event := event242568
    frameStart := 0 },
  { event := event242569
    frameStart := 0 },
  { event := event242570
    frameStart := 0 },
  { event := event242571
    frameStart := 0 },
  { event := event242572
    frameStart := 0 },
  { event := event242573
    frameStart := 0 },
  { event := event242574
    frameStart := 0 },
  { event := event242575
    frameStart := 0 }
]

def eventLeaf15161 : Array AnnotatedEvent := #[
  { event := event242576
    frameStart := 0 },
  { event := event242577
    frameStart := 0 },
  { event := event242578
    frameStart := 0 },
  { event := event242579
    frameStart := 0 },
  { event := event242580
    frameStart := 0 },
  { event := event242581
    frameStart := 0 },
  { event := event242582
    frameStart := 0 },
  { event := event242583
    frameStart := 0 },
  { event := event242584
    frameStart := 0 },
  { event := event242585
    frameStart := 0 },
  { event := event242586
    frameStart := 0 },
  { event := event242587
    frameStart := 0 },
  { event := event242588
    frameStart := 0 },
  { event := event242589
    frameStart := 0 },
  { event := event242590
    frameStart := 0 },
  { event := event242591
    frameStart := 0 }
]

def eventLeaf15162 : Array AnnotatedEvent := #[
  { event := event242592
    frameStart := 0 },
  { event := event242593
    frameStart := 0 },
  { event := event242594
    frameStart := 0 },
  { event := event242595
    frameStart := 0 },
  { event := event242596
    frameStart := 0 },
  { event := event242597
    frameStart := 0 },
  { event := event242598
    frameStart := 0 },
  { event := event242599
    frameStart := 0 },
  { event := event242600
    frameStart := 0 },
  { event := event242601
    frameStart := 0 },
  { event := event242602
    frameStart := 0 },
  { event := event242603
    frameStart := 0 },
  { event := event242604
    frameStart := 0 },
  { event := event242605
    frameStart := 0 },
  { event := event242606
    frameStart := 0 },
  { event := event242607
    frameStart := 0 }
]

def eventLeaf15163 : Array AnnotatedEvent := #[
  { event := event242608
    frameStart := 0 },
  { event := event242609
    frameStart := 0 },
  { event := event242610
    frameStart := 0 },
  { event := event242611
    frameStart := 0 },
  { event := event242612
    frameStart := 0 },
  { event := event242613
    frameStart := 0 },
  { event := event242614
    frameStart := 0 },
  { event := event242615
    frameStart := 0 },
  { event := event242616
    frameStart := 0 },
  { event := event242617
    frameStart := 0 },
  { event := event242618
    frameStart := 0 },
  { event := event242619
    frameStart := 0 },
  { event := event242620
    frameStart := 0 },
  { event := event242621
    frameStart := 0 },
  { event := event242622
    frameStart := 0 },
  { event := event242623
    frameStart := 0 }
]

def eventLeaf15164 : Array AnnotatedEvent := #[
  { event := event242624
    frameStart := 0 },
  { event := event242625
    frameStart := 0 },
  { event := event242626
    frameStart := 0 },
  { event := event242627
    frameStart := 0 },
  { event := event242628
    frameStart := 0 },
  { event := event242629
    frameStart := 0 },
  { event := event242630
    frameStart := 0 },
  { event := event242631
    frameStart := 0 },
  { event := event242632
    frameStart := 0 },
  { event := event242633
    frameStart := 0 },
  { event := event242634
    frameStart := 0 },
  { event := event242635
    frameStart := 0 },
  { event := event242636
    frameStart := 0 },
  { event := event242637
    frameStart := 0 },
  { event := event242638
    frameStart := 0 },
  { event := event242639
    frameStart := 0 }
]

def eventLeaf15165 : Array AnnotatedEvent := #[
  { event := event242640
    frameStart := 0 },
  { event := event242641
    frameStart := 0 },
  { event := event242642
    frameStart := 0 },
  { event := event242643
    frameStart := 0 },
  { event := event242644
    frameStart := 0 },
  { event := event242645
    frameStart := 0 },
  { event := event242646
    frameStart := 0 },
  { event := event242647
    frameStart := 0 },
  { event := event242648
    frameStart := 0 },
  { event := event242649
    frameStart := 0 },
  { event := event242650
    frameStart := 0 },
  { event := event242651
    frameStart := 0 },
  { event := event242652
    frameStart := 0 },
  { event := event242653
    frameStart := 0 },
  { event := event242654
    frameStart := 0 },
  { event := event242655
    frameStart := 0 }
]

def eventLeaf15166 : Array AnnotatedEvent := #[
  { event := event242656
    frameStart := 0 },
  { event := event242657
    frameStart := 0 },
  { event := event242658
    frameStart := 0 },
  { event := event242659
    frameStart := 0 },
  { event := event242660
    frameStart := 0 },
  { event := event242661
    frameStart := 242661 },
  { event := event242662
    frameStart := 242661 },
  { event := event242663
    frameStart := 242661 },
  { event := event242664
    frameStart := 242661 },
  { event := event242665
    frameStart := 242661 },
  { event := event242666
    frameStart := 242661 },
  { event := event242667
    frameStart := 242661 },
  { event := event242668
    frameStart := 242661 },
  { event := event242669
    frameStart := 242661 },
  { event := event242670
    frameStart := 242661 },
  { event := event242671
    frameStart := 242661 }
]

def eventLeaf15167 : Array AnnotatedEvent := #[
  { event := event242672
    frameStart := 242661 },
  { event := event242673
    frameStart := 242661 },
  { event := event242674
    frameStart := 242661 },
  { event := event242675
    frameStart := 242661 },
  { event := event242676
    frameStart := 242661 },
  { event := event242677
    frameStart := 242661 },
  { event := event242678
    frameStart := 242661 },
  { event := event242679
    frameStart := 242661 },
  { event := event242680
    frameStart := 242661 },
  { event := event242681
    frameStart := 242661 },
  { event := event242682
    frameStart := 242661 },
  { event := event242683
    frameStart := 242661 },
  { event := event242684
    frameStart := 242661 },
  { event := event242685
    frameStart := 242661 },
  { event := event242686
    frameStart := 242661 },
  { event := event242687
    frameStart := 242661 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events947
