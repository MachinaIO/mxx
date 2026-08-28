import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events416

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact106496RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42498⟩⟩], []⟩, (1)⟩]

theorem exact106496RawTermsValid :
    exact106496RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106496 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42498⟩⟩) exact106496RawTerms (.finite 52) 106495 .exactZero (none)

def event106497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14496⟩⟩) 0 ⟨5766⟩ 106493

def event106498 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14496⟩⟩) (.authority (.programFamilyFact))

def exact106499RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩], []⟩, (1)⟩]

theorem exact106499RawTermsValid :
    exact106499RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106499 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14496⟩⟩) exact106499RawTerms (.finite 52) 106498 .exactZero (none)

def event106500 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42499⟩⟩) 0 ⟨14496⟩ 106499

def event106501 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42499⟩⟩) 1 ⟨42498⟩ 106496

def event106502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42499⟩⟩) (.product (.predecessor 0 106500 .coefficient) (.predecessor 1 106501 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event106503 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42499⟩⟩, .operator (⟨106499, 0⟩, ⟨106496, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], []⟩, (1)⟩)

def exact106504RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14496⟩⟩, ⟨.program ⟨257⟩, ⟨42498⟩⟩], []⟩, (1)⟩]

theorem exact106504RawTermsValid :
    exact106504RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106504 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42499⟩⟩) exact106504RawTerms (.finite 2704) 106502 .exactZero (none)

def event106505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42500⟩⟩) 0 ⟨42499⟩ 106504

def event106506 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42500⟩⟩) (.identity (.predecessor 0 106505 .coefficient))

def event106507 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42500⟩⟩) (.finite 2704)

def event106508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42796⟩⟩) 0 ⟨42500⟩ 106507

def event106509 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42796⟩⟩) (.authority (.programFamilyFact))

def exact106510RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], []⟩, (1)⟩]

theorem exact106510RawTermsValid :
    exact106510RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106510 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42796⟩⟩) exact106510RawTerms (.finite 52) 106509 .exactZero (none)

def event106511 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42797⟩⟩) 0 ⟨42796⟩ 106510

def event106512 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42797⟩⟩) (.identity (.predecessor 0 106511 .coefficient))

def event106513 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42797⟩⟩) (.finite 52)

def event106514 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43948⟩⟩) 0 ⟨42797⟩ 106513

def event106515 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43948⟩⟩) (.authority (.programFamilyFact))

def event106516 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨43948⟩⟩) (.finite 3720)

def event106517 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event106518 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43950⟩⟩) 0 ⟨7177⟩ 106517

def event106519 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43950⟩⟩) 1 ⟨43948⟩ 106516

def event106520 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43950⟩⟩) (.authority (.operator))

def exact106521RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43950⟩⟩]⟩, (1)⟩]

theorem exact106521RawTermsValid :
    exact106521RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106521 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43950⟩⟩) exact106521RawTerms .large 106520 .exactZero (none)

def event106522 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44694⟩⟩) 0 ⟨43950⟩ 106521

def event106523 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44694⟩⟩) (.authority (.operator))

def exact106524RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩, (1)⟩]

theorem exact106524RawTermsValid :
    exact106524RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106524 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44694⟩⟩) exact106524RawTerms (.finite 8192) 106523 .exactZero (none)

def event106525 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event106526 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event106527 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44150⟩⟩) 0 ⟨42797⟩ 106513

def event106528 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44150⟩⟩) 1 ⟨136⟩ 106526

def event106529 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44150⟩⟩) (.sum [.predecessor 0 106527 .coefficient, .predecessor 1 106528 .coefficient])

def event106530 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44150⟩⟩) (.finite 52)

def event106531 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44151⟩⟩) 0 ⟨44150⟩ 106530

def event106532 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44151⟩⟩) (.identity (.predecessor 0 106531 .coefficient))

def exact106533RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], []⟩, (1)⟩]

theorem exact106533RawTermsValid :
    exact106533RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106533 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44151⟩⟩) exact106533RawTerms (.finite 52) 106532 .exactZero (none)

def event106534 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact106535RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact106535RawTermsValid :
    exact106535RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106535 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact106535RawTerms .large 106534 .exactZero (none)

def event106536 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44152⟩⟩) 0 ⟨6908⟩ 106535

def event106537 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44152⟩⟩) 1 ⟨44151⟩ 106533

def event106538 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44152⟩⟩) (.product (.predecessor 0 106536 .coefficient) (.predecessor 1 106537 .coefficient) (⟨false, false, none, none, none⟩))

def event106539 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44152⟩⟩, .operator (⟨106535, 0⟩, ⟨106533, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact106540RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact106540RawTermsValid :
    exact106540RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106540 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44152⟩⟩) exact106540RawTerms .large 106538 .exactZero (none)

def event106541 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7194⟩⟩) 0 ⟨7177⟩ 106517

def event106542 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7194⟩⟩) (.authority (.operator))

def exact106543RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩]

theorem exact106543RawTermsValid :
    exact106543RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106543 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7194⟩⟩) exact106543RawTerms .large 106542 .exactZero (none)

def event106544 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44153⟩⟩) 0 ⟨7194⟩ 106543

def event106545 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44153⟩⟩) 1 ⟨44152⟩ 106540

def event106546 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44153⟩⟩) (.sum [.predecessor 0 106544 .coefficient, .predecessor 1 106545 .coefficient])

def exact106547RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106547RawTermsValid :
    exact106547RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106547 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44153⟩⟩) exact106547RawTerms .large 106546 .exactZero (none)

def event106548 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44695⟩⟩) 0 ⟨44153⟩ 106547

def event106549 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44695⟩⟩) 1 ⟨44694⟩ 106524

def event106550 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44695⟩⟩) (.product (.predecessor 0 106548 .coefficient) (.predecessor 1 106549 .coefficient) (⟨false, false, none, none, none⟩))

def event106551 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44695⟩⟩, .operator (⟨106547, 0⟩, ⟨106524, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩, (1)⟩)

def event106552 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44695⟩⟩, .operator (⟨106547, 1⟩, ⟨106524, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩, (-1)⟩)

def event106553 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44695⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44694⟩⟩) ⟨43950⟩ 106521)

def event106554 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44695⟩⟩, .relation 106553 0, ⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨43950⟩⟩]⟩, (-1)⟩)

def exact106555RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨43950⟩⟩]⟩, (-1)⟩]

theorem exact106555RawTermsValid :
    exact106555RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106555 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44695⟩⟩) exact106555RawTerms .large 106550 .exactZero (none)

def event106556 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43012⟩⟩) 0 ⟨42797⟩ 106513

def event106557 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43012⟩⟩) (.authority (.programFamilyFact))

def exact106558RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨43012⟩⟩], []⟩, (1)⟩]

theorem exact106558RawTermsValid :
    exact106558RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106558 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43012⟩⟩) exact106558RawTerms (.finite 63) 106557 .exactZero (none)

def event106559 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43013⟩⟩) 0 ⟨6908⟩ 106535

def event106560 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43013⟩⟩) 1 ⟨43012⟩ 106558

def event106561 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43013⟩⟩) (.product (.predecessor 0 106559 .coefficient) (.predecessor 1 106560 .coefficient) (⟨false, true, none, none, some 1⟩))

def event106562 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43013⟩⟩, .operator (⟨106535, 0⟩, ⟨106558, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨43012⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact106563RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨43012⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact106563RawTermsValid :
    exact106563RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106563 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43013⟩⟩) exact106563RawTerms .large 106561 .exactZero (none)

def event106564 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7228⟩⟩) 0 ⟨7177⟩ 106517

def event106565 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7228⟩⟩) (.authority (.operator))

def exact106566RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact106566RawTermsValid :
    exact106566RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106566 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7228⟩⟩) exact106566RawTerms .large 106565 .exactZero (none)

def event106567 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43014⟩⟩) 0 ⟨7228⟩ 106566

def event106568 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43014⟩⟩) 1 ⟨43013⟩ 106563

def event106569 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43014⟩⟩) (.sum [.predecessor 0 106567 .coefficient, .predecessor 1 106568 .coefficient])

def exact106570RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43012⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106570RawTermsValid :
    exact106570RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106570 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43014⟩⟩) exact106570RawTerms .large 106569 .exactZero (none)

def event106571 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44698⟩⟩) 0 ⟨43014⟩ 106570

def event106572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44698⟩⟩) 1 ⟨44695⟩ 106555

def event106573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44698⟩⟩) (.sum [.predecessor 0 106571 .coefficient, .predecessor 1 106572 .coefficient])

def exact106574RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨43950⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43012⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106574RawTermsValid :
    exact106574RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106574 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44698⟩⟩) exact106574RawTerms .large 106573 .exactZero (none)

def event106575 : Event := .preFoldPolynomial 106574 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨43950⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43012⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact106576RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨43950⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43012⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event106576 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨44698⟩⟩) 106575 exact106576RawTerms .large 106573 .exactZero (none)

def event106577 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨42797⟩⟩) ⟨⟨107⟩, ⟨90⟩, ⟨135⟩⟩ ⟨106419, 106577⟩

def event106578 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨43559⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43556⟩⟩]⟩) (1) 0 2 (.universal 106577 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43556⟩⟩]⟩) (none) 106576)

def event106579 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43559⟩⟩, .relation 106578 1, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩)

def event106580 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43559⟩⟩, .relation 106578 0, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩, (-1)⟩)

def event106581 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43559⟩⟩, .relation 106578 2, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨43950⟩⟩]⟩, (1)⟩)

def event106582 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43559⟩⟩, .relation 106578 3, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨43012⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact106583RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨43950⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨43012⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106583RawTermsValid :
    exact106583RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106583 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43559⟩⟩) exact106583RawTerms .large 106415 (.finite 202072841853861888) (some (106417))

def event106584 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44697⟩⟩) 0 ⟨43559⟩ 106583

def event106585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44697⟩⟩) 1 ⟨44696⟩ 106405

def event106586 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44697⟩⟩) (.sum [.predecessor 0 106584 .coefficient, .predecessor 1 106585 .coefficient])

def event106587 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44697⟩⟩, .operator (⟨106583, 0⟩, ⟨106405, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44694⟩⟩]⟩, (1)⟩)

def event106588 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44697⟩⟩, .operator (⟨106583, 2⟩, ⟨106405, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨42796⟩⟩], [⟨.program ⟨257⟩, ⟨43950⟩⟩]⟩, (-1)⟩)

def event106589 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44697⟩⟩) (.sum [.result 106583 .summary, .result 106405 .summary])

def exact106590RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨43012⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106590RawTermsValid :
    exact106590RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106590 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44697⟩⟩) exact106590RawTerms .large 106586 (.finite 32193718473625891320532869316608) (some (106589))

def event106591 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41268⟩⟩) 0 ⟨40117⟩ 4668

def event106592 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41268⟩⟩) (.authority (.programFamilyFact))

def event106593 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41268⟩⟩) (.finite 3720)

def event106594 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41270⟩⟩) 0 ⟨7177⟩ 15500

def event106595 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41270⟩⟩) 1 ⟨41268⟩ 106593

def event106596 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41270⟩⟩) (.authority (.operator))

def exact106597RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41270⟩⟩]⟩, (1)⟩]

theorem exact106597RawTermsValid :
    exact106597RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106597 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41270⟩⟩) exact106597RawTerms .large 106596 .exactZero (none)

def event106598 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42014⟩⟩) 0 ⟨41270⟩ 106597

def event106599 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42014⟩⟩) (.authority (.operator))

def exact106600RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨42014⟩⟩]⟩, (1)⟩]

theorem exact106600RawTermsValid :
    exact106600RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106600 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42014⟩⟩) exact106600RawTerms (.finite 8192) 106599 .exactZero (none)

def event106601 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41114⟩⟩) 0 ⟨39820⟩ 4662

def event106602 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41114⟩⟩) (.authority (.programFamilyFact))

def event106603 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41114⟩⟩) (.finite 3720)

def event106604 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41115⟩⟩) 0 ⟨7177⟩ 15500

def event106605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41115⟩⟩) 1 ⟨41114⟩ 106603

def event106606 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41115⟩⟩) (.authority (.operator))

def exact106607RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41115⟩⟩]⟩, (1)⟩]

theorem exact106607RawTermsValid :
    exact106607RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106607 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41115⟩⟩) exact106607RawTerms .large 106606 .exactZero (none)

def event106608 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41630⟩⟩) 0 ⟨41115⟩ 106607

def event106609 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41630⟩⟩) (.authority (.operator))

def exact106610RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41630⟩⟩]⟩, (1)⟩]

theorem exact106610RawTermsValid :
    exact106610RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106610 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41630⟩⟩) exact106610RawTerms (.finite 8192) 106609 .exactZero (none)

def event106611 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39821⟩⟩) 0 ⟨39818⟩ 4651

def event106612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39821⟩⟩) 1 ⟨6992⟩ 105153

def event106613 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39821⟩⟩) (.tensor (.predecessor 0 106611 .coefficient) (.predecessor 1 106612 .coefficient) true false)

def event106614 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39821⟩⟩, .operator (⟨4651, 0⟩, ⟨105153, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact106615RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact106615RawTermsValid :
    exact106615RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106615 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39821⟩⟩) exact106615RawTerms .large 106613 .exactZero (none)

def event106616 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8702⟩⟩) 0 ⟨5768⟩ 105023

def event106617 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8702⟩⟩) 1 ⟨7282⟩ 18583

def event106618 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8702⟩⟩) (.product (.predecessor 0 106616 .coefficient) (.predecessor 1 106617 .coefficient) (⟨false, false, none, none, none⟩))

def event106619 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8702⟩⟩, .operator (⟨105023, 0⟩, ⟨18583, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩)

def exact106620RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩]

theorem exact106620RawTermsValid :
    exact106620RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106620 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8702⟩⟩) exact106620RawTerms .large 106618 .exactZero (none)

def event106621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39822⟩⟩) 0 ⟨8702⟩ 106620

def event106622 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39822⟩⟩) 1 ⟨39821⟩ 106615

def event106623 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39822⟩⟩) (.sum [.predecessor 0 106621 .coefficient, .predecessor 1 106622 .coefficient])

def exact106624RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106624RawTermsValid :
    exact106624RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106624 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39822⟩⟩) exact106624RawTerms .large 106623 .exactZero (none)

def event106625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39823⟩⟩) 0 ⟨39822⟩ 106624

def event106626 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39823⟩⟩) 1 ⟨108⟩ 18575

def event106627 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39823⟩⟩) (.sum [.predecessor 0 106625 .coefficient, .predecessor 1 106626 .coefficient])

def event106628 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39823⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨108⟩⟩]⟩) [⟨.result 18575 .coefficient, false, none⟩])

def event106629 : Event := .survivorFold (1) 106628

def exact106630RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106630RawTermsValid :
    exact106630RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106630 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39823⟩⟩) exact106630RawTerms .large 106627 (.finite 26) (some (106628))

def event106631 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39824⟩⟩) 0 ⟨39823⟩ 106630

def event106632 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39824⟩⟩) 1 ⟨14196⟩ 4654

def event106633 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39824⟩⟩) (.product (.predecessor 0 106631 .coefficient) (.predecessor 1 106632 .coefficient) (⟨false, true, none, none, some 1⟩))

def event106634 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39824⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩], []⟩) [⟨.result 4654 .coefficient, true, some 1⟩])

def event106635 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39824⟩⟩) (.product (.result 106630 .summary) (.transfer 106634) (⟨false, false, none, none, none⟩))

def event106636 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39824⟩⟩, .operator (⟨106630, 1⟩, ⟨4654, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event106637 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39824⟩⟩, .operator (⟨106630, 0⟩, ⟨4654, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩)

def exact106638RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106638RawTermsValid :
    exact106638RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106638 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39824⟩⟩) exact106638RawTerms .large 106633 (.finite 39190528) (some (106635))

def event106639 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14197⟩⟩) 0 ⟨14196⟩ 4654

def event106640 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14197⟩⟩) 1 ⟨6992⟩ 105153

def event106641 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14197⟩⟩) (.tensor (.predecessor 0 106639 .coefficient) (.predecessor 1 106640 .coefficient) true false)

def event106642 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14197⟩⟩, .operator (⟨4654, 0⟩, ⟨105153, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact106643RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact106643RawTermsValid :
    exact106643RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106643 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14197⟩⟩) exact106643RawTerms .large 106641 .exactZero (none)

def event106644 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8719⟩⟩) 0 ⟨5768⟩ 105023

def event106645 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8719⟩⟩) 1 ⟨7299⟩ 18624

def event106646 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8719⟩⟩) (.product (.predecessor 0 106644 .coefficient) (.predecessor 1 106645 .coefficient) (⟨false, false, none, none, none⟩))

def event106647 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8719⟩⟩, .operator (⟨105023, 0⟩, ⟨18624, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩)

def exact106648RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩]

theorem exact106648RawTermsValid :
    exact106648RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106648 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8719⟩⟩) exact106648RawTerms .large 106646 .exactZero (none)

def event106649 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14198⟩⟩) 0 ⟨8719⟩ 106648

def event106650 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14198⟩⟩) 1 ⟨14197⟩ 106643

def event106651 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14198⟩⟩) (.sum [.predecessor 0 106649 .coefficient, .predecessor 1 106650 .coefficient])

def exact106652RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106652RawTermsValid :
    exact106652RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106652 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14198⟩⟩) exact106652RawTerms .large 106651 .exactZero (none)

def event106653 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14199⟩⟩) 0 ⟨14198⟩ 106652

def event106654 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14199⟩⟩) 1 ⟨125⟩ 18616

def event106655 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14199⟩⟩) (.sum [.predecessor 0 106653 .coefficient, .predecessor 1 106654 .coefficient])

def event106656 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14199⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨125⟩⟩]⟩) [⟨.result 18616 .coefficient, false, none⟩])

def event106657 : Event := .survivorFold (1) 106656

def exact106658RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106658RawTermsValid :
    exact106658RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106658 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14199⟩⟩) exact106658RawTerms .large 106655 (.finite 26) (some (106656))

def event106659 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14200⟩⟩) 0 ⟨14199⟩ 106658

def event106660 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14200⟩⟩) 1 ⟨9557⟩ 18613

def event106661 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14200⟩⟩) (.product (.predecessor 0 106659 .coefficient) (.predecessor 1 106660 .coefficient) (⟨false, false, none, none, none⟩))

def event106662 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14200⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) [⟨.result 18609 .coefficient, false, none⟩])

def event106663 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14200⟩⟩) (.product (.result 106658 .summary) (.transfer 106662) (⟨false, false, none, none, none⟩))

def event106664 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14200⟩⟩, .operator (⟨106658, 1⟩, ⟨18613, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (-1)⟩)

def event106665 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨14200⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9556⟩⟩) ⟨7282⟩ 18583)

def event106666 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14200⟩⟩, .relation 106665 0, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (-1)⟩)

def event106667 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14200⟩⟩, .operator (⟨106658, 0⟩, ⟨18613, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩)

def exact106668RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (-1)⟩]

theorem exact106668RawTermsValid :
    exact106668RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106668 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14200⟩⟩) exact106668RawTerms .large 106661 (.finite 279172874240) (some (106663))

def event106669 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39825⟩⟩) 0 ⟨14200⟩ 106668

def event106670 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39825⟩⟩) 1 ⟨39824⟩ 106638

def event106671 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39825⟩⟩) (.sum [.predecessor 0 106669 .coefficient, .predecessor 1 106670 .coefficient])

def event106672 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39825⟩⟩, .operator (⟨106668, 1⟩, ⟨106638, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩)

def event106673 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39825⟩⟩) (.sum [.result 106668 .summary, .result 106638 .summary])

def exact106674RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact106674RawTermsValid :
    exact106674RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106674 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39825⟩⟩) exact106674RawTerms .large 106671 (.finite 279212064768) (some (106673))

def event106675 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41631⟩⟩) 0 ⟨39825⟩ 106674

def event106676 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41631⟩⟩) 1 ⟨41630⟩ 106610

def event106677 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41631⟩⟩) (.product (.predecessor 0 106675 .coefficient) (.predecessor 1 106676 .coefficient) (⟨false, false, none, none, none⟩))

def event106678 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41631⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨41630⟩⟩]⟩) [⟨.result 106610 .coefficient, false, none⟩])

def event106679 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41631⟩⟩) (.product (.result 106674 .summary) (.transfer 106678) (⟨false, false, none, none, none⟩))

def event106680 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41631⟩⟩, .operator (⟨106674, 1⟩, ⟨106610, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41630⟩⟩]⟩, (-1)⟩)

def event106681 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41631⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41630⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41630⟩⟩) ⟨41115⟩ 106607)

def event106682 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41631⟩⟩, .relation 106681 0, ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨41115⟩⟩]⟩, (-1)⟩)

def event106683 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41631⟩⟩, .operator (⟨106674, 0⟩, ⟨106610, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41630⟩⟩]⟩, (1)⟩)

def exact106684RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41630⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], [⟨.program ⟨257⟩, ⟨41115⟩⟩]⟩, (-1)⟩]

theorem exact106684RawTermsValid :
    exact106684RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106684 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41631⟩⟩) exact106684RawTerms .large 106677 (.finite 2998016717067984568320) (some (106679))

def event106685 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40559⟩⟩) 0 ⟨39820⟩ 4662

def event106686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40559⟩⟩) (.authority (.relationPreimageSource ⟨51⟩))

def exact106687RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40559⟩⟩]⟩, (1)⟩]

theorem exact106687RawTermsValid :
    exact106687RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106687 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40559⟩⟩) exact106687RawTerms (.finite 5647228698) 106686 .exactZero (none)

def event106688 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40561⟩⟩) 0 ⟨40559⟩ 106687

def event106689 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40561⟩⟩) 1 ⟨2370⟩ 4

def event106690 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40561⟩⟩) (.scale (.predecessor 0 106688 .coefficient) (.value (.predecessor 1 106689 .coefficient)))

def exact106691RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40559⟩⟩]⟩, (1)⟩]

theorem exact106691RawTermsValid :
    exact106691RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106691 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40561⟩⟩) exact106691RawTerms (.finite 5647228698) 106690 .exactZero (none)

def event106692 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40562⟩⟩) 0 ⟨5770⟩ 105245

def event106693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40562⟩⟩) 1 ⟨40561⟩ 106691

def event106694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40562⟩⟩) (.product (.predecessor 0 106692 .coefficient) (.predecessor 1 106693 .coefficient) (⟨false, false, none, none, none⟩))

def event106695 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40562⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨40559⟩⟩]⟩) [⟨.result 106687 .coefficient, false, none⟩])

def event106696 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40562⟩⟩) (.product (.result 105245 .summary) (.transfer 106695) (⟨false, false, none, none, none⟩))

def event106697 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40562⟩⟩, .operator (⟨105245, 0⟩, ⟨106691, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40559⟩⟩]⟩, (1)⟩)

def event106698 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨40560⟩⟩)

def event106699 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event106700 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event106701 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.authority (.operator))

def event106702 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.finite 13)

def event106703 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event106704 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event106705 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event106706 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event106707 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 106706

def event106708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 106704

def event106709 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 106707 .coefficient) (.value (.predecessor 1 106708 .coefficient)))

def event106710 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event106711 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 0 ⟨392⟩ 106710

def event106712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5756⟩⟩) 1 ⟨5754⟩ 106702

def event106713 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.sum [.predecessor 0 106711 .coefficient, .predecessor 1 106712 .coefficient])

def event106714 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5756⟩⟩) (.finite 655353)

def event106715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 0 ⟨5756⟩ 106714

def event106716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5766⟩⟩) 1 ⟨5426⟩ 106700

def event106717 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.identity (.predecessor 1 106716 .coefficient))

def event106718 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5766⟩⟩) (.finite 655360)

def event106719 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39818⟩⟩) 0 ⟨5766⟩ 106718

def event106720 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39818⟩⟩) (.authority (.programFamilyFact))

def exact106721RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39818⟩⟩], []⟩, (1)⟩]

theorem exact106721RawTermsValid :
    exact106721RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106721 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39818⟩⟩) exact106721RawTerms (.finite 46) 106720 .exactZero (none)

def event106722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14196⟩⟩) 0 ⟨5766⟩ 106718

def event106723 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14196⟩⟩) (.authority (.programFamilyFact))

def exact106724RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩], []⟩, (1)⟩]

theorem exact106724RawTermsValid :
    exact106724RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106724 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14196⟩⟩) exact106724RawTerms (.finite 46) 106723 .exactZero (none)

def event106725 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39819⟩⟩) 0 ⟨14196⟩ 106724

def event106726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39819⟩⟩) 1 ⟨39818⟩ 106721

def event106727 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39819⟩⟩) (.product (.predecessor 0 106725 .coefficient) (.predecessor 1 106726 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event106728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39819⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14196⟩⟩, ⟨.program ⟨257⟩, ⟨39818⟩⟩], []⟩) [⟨.result 106724 .coefficient, true, some 1⟩, ⟨.result 106721 .coefficient, true, some 1⟩])

def event106729 : Event := .survivorFold (1) 106728

def exact106730RawTerms : List Term := []

theorem exact106730RawTermsValid :
    exact106730RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106730 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39819⟩⟩) exact106730RawTerms (.finite 2116) 106727 (.finite 2116) (some (106728))

def event106731 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39820⟩⟩) 0 ⟨39819⟩ 106730

def event106732 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39820⟩⟩) (.identity (.predecessor 0 106731 .coefficient))

def event106733 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39820⟩⟩) (.finite 2116)

def event106734 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40559⟩⟩) 0 ⟨39820⟩ 106733

def event106735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40559⟩⟩) (.authority (.relationPreimageSource ⟨51⟩))

def exact106736RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40559⟩⟩]⟩, (1)⟩]

theorem exact106736RawTermsValid :
    exact106736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106736 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40559⟩⟩) exact106736RawTerms (.finite 5647228698) 106735 .exactZero (none)

def event106737 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact106738RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact106738RawTermsValid :
    exact106738RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106738 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact106738RawTerms .large 106737 .exactZero (none)

def event106739 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40560⟩⟩) 0 ⟨35⟩ 106738

def event106740 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40560⟩⟩) 1 ⟨40559⟩ 106736

def event106741 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40560⟩⟩) (.product (.predecessor 0 106739 .coefficient) (.predecessor 1 106740 .coefficient) (⟨false, false, none, none, none⟩))

def event106742 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40560⟩⟩, .operator (⟨106738, 0⟩, ⟨106736, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40559⟩⟩]⟩, (1)⟩)

def exact106743RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40559⟩⟩]⟩, (1)⟩]

theorem exact106743RawTermsValid :
    exact106743RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event106743 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40560⟩⟩) exact106743RawTerms .large 106741 .exactZero (none)

def event106744 : Event := .preFoldPolynomial 106743 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40559⟩⟩]⟩, (1)⟩] .exactZero none

def exact106745RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40559⟩⟩]⟩, (1)⟩]

def event106745 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨40560⟩⟩) 106744 exact106745RawTerms .large 106741 .exactZero (none)

def event106746 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨41634⟩⟩)

def event106747 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event106748 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event106749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.authority (.operator))

def event106750 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5754⟩⟩) (.finite 13)

def event106751 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def eventLeaf6656 : Array AnnotatedEvent := #[
  { event := event106496
    frameStart := 106473 },
  { event := event106497
    frameStart := 106473 },
  { event := event106498
    frameStart := 106473 },
  { event := event106499
    frameStart := 106473 },
  { event := event106500
    frameStart := 106473 },
  { event := event106501
    frameStart := 106473 },
  { event := event106502
    frameStart := 106473 },
  { event := event106503
    frameStart := 106473 },
  { event := event106504
    frameStart := 106473 },
  { event := event106505
    frameStart := 106473 },
  { event := event106506
    frameStart := 106473 },
  { event := event106507
    frameStart := 106473 },
  { event := event106508
    frameStart := 106473 },
  { event := event106509
    frameStart := 106473 },
  { event := event106510
    frameStart := 106473 },
  { event := event106511
    frameStart := 106473 }
]

def eventLeaf6657 : Array AnnotatedEvent := #[
  { event := event106512
    frameStart := 106473 },
  { event := event106513
    frameStart := 106473 },
  { event := event106514
    frameStart := 106473 },
  { event := event106515
    frameStart := 106473 },
  { event := event106516
    frameStart := 106473 },
  { event := event106517
    frameStart := 106473 },
  { event := event106518
    frameStart := 106473 },
  { event := event106519
    frameStart := 106473 },
  { event := event106520
    frameStart := 106473 },
  { event := event106521
    frameStart := 106473 },
  { event := event106522
    frameStart := 106473 },
  { event := event106523
    frameStart := 106473 },
  { event := event106524
    frameStart := 106473 },
  { event := event106525
    frameStart := 106473 },
  { event := event106526
    frameStart := 106473 },
  { event := event106527
    frameStart := 106473 }
]

def eventLeaf6658 : Array AnnotatedEvent := #[
  { event := event106528
    frameStart := 106473 },
  { event := event106529
    frameStart := 106473 },
  { event := event106530
    frameStart := 106473 },
  { event := event106531
    frameStart := 106473 },
  { event := event106532
    frameStart := 106473 },
  { event := event106533
    frameStart := 106473 },
  { event := event106534
    frameStart := 106473 },
  { event := event106535
    frameStart := 106473 },
  { event := event106536
    frameStart := 106473 },
  { event := event106537
    frameStart := 106473 },
  { event := event106538
    frameStart := 106473 },
  { event := event106539
    frameStart := 106473 },
  { event := event106540
    frameStart := 106473 },
  { event := event106541
    frameStart := 106473 },
  { event := event106542
    frameStart := 106473 },
  { event := event106543
    frameStart := 106473 }
]

def eventLeaf6659 : Array AnnotatedEvent := #[
  { event := event106544
    frameStart := 106473 },
  { event := event106545
    frameStart := 106473 },
  { event := event106546
    frameStart := 106473 },
  { event := event106547
    frameStart := 106473 },
  { event := event106548
    frameStart := 106473 },
  { event := event106549
    frameStart := 106473 },
  { event := event106550
    frameStart := 106473 },
  { event := event106551
    frameStart := 106473 },
  { event := event106552
    frameStart := 106473 },
  { event := event106553
    frameStart := 106473 },
  { event := event106554
    frameStart := 106473 },
  { event := event106555
    frameStart := 106473 },
  { event := event106556
    frameStart := 106473 },
  { event := event106557
    frameStart := 106473 },
  { event := event106558
    frameStart := 106473 },
  { event := event106559
    frameStart := 106473 }
]

def eventLeaf6660 : Array AnnotatedEvent := #[
  { event := event106560
    frameStart := 106473 },
  { event := event106561
    frameStart := 106473 },
  { event := event106562
    frameStart := 106473 },
  { event := event106563
    frameStart := 106473 },
  { event := event106564
    frameStart := 106473 },
  { event := event106565
    frameStart := 106473 },
  { event := event106566
    frameStart := 106473 },
  { event := event106567
    frameStart := 106473 },
  { event := event106568
    frameStart := 106473 },
  { event := event106569
    frameStart := 106473 },
  { event := event106570
    frameStart := 106473 },
  { event := event106571
    frameStart := 106473 },
  { event := event106572
    frameStart := 106473 },
  { event := event106573
    frameStart := 106473 },
  { event := event106574
    frameStart := 106473 },
  { event := event106575
    frameStart := 106473 }
]

def eventLeaf6661 : Array AnnotatedEvent := #[
  { event := event106576
    frameStart := 106473 },
  { event := event106577
    frameStart := 0 },
  { event := event106578
    frameStart := 0 },
  { event := event106579
    frameStart := 0 },
  { event := event106580
    frameStart := 0 },
  { event := event106581
    frameStart := 0 },
  { event := event106582
    frameStart := 0 },
  { event := event106583
    frameStart := 0 },
  { event := event106584
    frameStart := 0 },
  { event := event106585
    frameStart := 0 },
  { event := event106586
    frameStart := 0 },
  { event := event106587
    frameStart := 0 },
  { event := event106588
    frameStart := 0 },
  { event := event106589
    frameStart := 0 },
  { event := event106590
    frameStart := 0 },
  { event := event106591
    frameStart := 0 }
]

def eventLeaf6662 : Array AnnotatedEvent := #[
  { event := event106592
    frameStart := 0 },
  { event := event106593
    frameStart := 0 },
  { event := event106594
    frameStart := 0 },
  { event := event106595
    frameStart := 0 },
  { event := event106596
    frameStart := 0 },
  { event := event106597
    frameStart := 0 },
  { event := event106598
    frameStart := 0 },
  { event := event106599
    frameStart := 0 },
  { event := event106600
    frameStart := 0 },
  { event := event106601
    frameStart := 0 },
  { event := event106602
    frameStart := 0 },
  { event := event106603
    frameStart := 0 },
  { event := event106604
    frameStart := 0 },
  { event := event106605
    frameStart := 0 },
  { event := event106606
    frameStart := 0 },
  { event := event106607
    frameStart := 0 }
]

def eventLeaf6663 : Array AnnotatedEvent := #[
  { event := event106608
    frameStart := 0 },
  { event := event106609
    frameStart := 0 },
  { event := event106610
    frameStart := 0 },
  { event := event106611
    frameStart := 0 },
  { event := event106612
    frameStart := 0 },
  { event := event106613
    frameStart := 0 },
  { event := event106614
    frameStart := 0 },
  { event := event106615
    frameStart := 0 },
  { event := event106616
    frameStart := 0 },
  { event := event106617
    frameStart := 0 },
  { event := event106618
    frameStart := 0 },
  { event := event106619
    frameStart := 0 },
  { event := event106620
    frameStart := 0 },
  { event := event106621
    frameStart := 0 },
  { event := event106622
    frameStart := 0 },
  { event := event106623
    frameStart := 0 }
]

def eventLeaf6664 : Array AnnotatedEvent := #[
  { event := event106624
    frameStart := 0 },
  { event := event106625
    frameStart := 0 },
  { event := event106626
    frameStart := 0 },
  { event := event106627
    frameStart := 0 },
  { event := event106628
    frameStart := 0 },
  { event := event106629
    frameStart := 0 },
  { event := event106630
    frameStart := 0 },
  { event := event106631
    frameStart := 0 },
  { event := event106632
    frameStart := 0 },
  { event := event106633
    frameStart := 0 },
  { event := event106634
    frameStart := 0 },
  { event := event106635
    frameStart := 0 },
  { event := event106636
    frameStart := 0 },
  { event := event106637
    frameStart := 0 },
  { event := event106638
    frameStart := 0 },
  { event := event106639
    frameStart := 0 }
]

def eventLeaf6665 : Array AnnotatedEvent := #[
  { event := event106640
    frameStart := 0 },
  { event := event106641
    frameStart := 0 },
  { event := event106642
    frameStart := 0 },
  { event := event106643
    frameStart := 0 },
  { event := event106644
    frameStart := 0 },
  { event := event106645
    frameStart := 0 },
  { event := event106646
    frameStart := 0 },
  { event := event106647
    frameStart := 0 },
  { event := event106648
    frameStart := 0 },
  { event := event106649
    frameStart := 0 },
  { event := event106650
    frameStart := 0 },
  { event := event106651
    frameStart := 0 },
  { event := event106652
    frameStart := 0 },
  { event := event106653
    frameStart := 0 },
  { event := event106654
    frameStart := 0 },
  { event := event106655
    frameStart := 0 }
]

def eventLeaf6666 : Array AnnotatedEvent := #[
  { event := event106656
    frameStart := 0 },
  { event := event106657
    frameStart := 0 },
  { event := event106658
    frameStart := 0 },
  { event := event106659
    frameStart := 0 },
  { event := event106660
    frameStart := 0 },
  { event := event106661
    frameStart := 0 },
  { event := event106662
    frameStart := 0 },
  { event := event106663
    frameStart := 0 },
  { event := event106664
    frameStart := 0 },
  { event := event106665
    frameStart := 0 },
  { event := event106666
    frameStart := 0 },
  { event := event106667
    frameStart := 0 },
  { event := event106668
    frameStart := 0 },
  { event := event106669
    frameStart := 0 },
  { event := event106670
    frameStart := 0 },
  { event := event106671
    frameStart := 0 }
]

def eventLeaf6667 : Array AnnotatedEvent := #[
  { event := event106672
    frameStart := 0 },
  { event := event106673
    frameStart := 0 },
  { event := event106674
    frameStart := 0 },
  { event := event106675
    frameStart := 0 },
  { event := event106676
    frameStart := 0 },
  { event := event106677
    frameStart := 0 },
  { event := event106678
    frameStart := 0 },
  { event := event106679
    frameStart := 0 },
  { event := event106680
    frameStart := 0 },
  { event := event106681
    frameStart := 0 },
  { event := event106682
    frameStart := 0 },
  { event := event106683
    frameStart := 0 },
  { event := event106684
    frameStart := 0 },
  { event := event106685
    frameStart := 0 },
  { event := event106686
    frameStart := 0 },
  { event := event106687
    frameStart := 0 }
]

def eventLeaf6668 : Array AnnotatedEvent := #[
  { event := event106688
    frameStart := 0 },
  { event := event106689
    frameStart := 0 },
  { event := event106690
    frameStart := 0 },
  { event := event106691
    frameStart := 0 },
  { event := event106692
    frameStart := 0 },
  { event := event106693
    frameStart := 0 },
  { event := event106694
    frameStart := 0 },
  { event := event106695
    frameStart := 0 },
  { event := event106696
    frameStart := 0 },
  { event := event106697
    frameStart := 0 },
  { event := event106698
    frameStart := 106698 },
  { event := event106699
    frameStart := 106698 },
  { event := event106700
    frameStart := 106698 },
  { event := event106701
    frameStart := 106698 },
  { event := event106702
    frameStart := 106698 },
  { event := event106703
    frameStart := 106698 }
]

def eventLeaf6669 : Array AnnotatedEvent := #[
  { event := event106704
    frameStart := 106698 },
  { event := event106705
    frameStart := 106698 },
  { event := event106706
    frameStart := 106698 },
  { event := event106707
    frameStart := 106698 },
  { event := event106708
    frameStart := 106698 },
  { event := event106709
    frameStart := 106698 },
  { event := event106710
    frameStart := 106698 },
  { event := event106711
    frameStart := 106698 },
  { event := event106712
    frameStart := 106698 },
  { event := event106713
    frameStart := 106698 },
  { event := event106714
    frameStart := 106698 },
  { event := event106715
    frameStart := 106698 },
  { event := event106716
    frameStart := 106698 },
  { event := event106717
    frameStart := 106698 },
  { event := event106718
    frameStart := 106698 },
  { event := event106719
    frameStart := 106698 }
]

def eventLeaf6670 : Array AnnotatedEvent := #[
  { event := event106720
    frameStart := 106698 },
  { event := event106721
    frameStart := 106698 },
  { event := event106722
    frameStart := 106698 },
  { event := event106723
    frameStart := 106698 },
  { event := event106724
    frameStart := 106698 },
  { event := event106725
    frameStart := 106698 },
  { event := event106726
    frameStart := 106698 },
  { event := event106727
    frameStart := 106698 },
  { event := event106728
    frameStart := 106698 },
  { event := event106729
    frameStart := 106698 },
  { event := event106730
    frameStart := 106698 },
  { event := event106731
    frameStart := 106698 },
  { event := event106732
    frameStart := 106698 },
  { event := event106733
    frameStart := 106698 },
  { event := event106734
    frameStart := 106698 },
  { event := event106735
    frameStart := 106698 }
]

def eventLeaf6671 : Array AnnotatedEvent := #[
  { event := event106736
    frameStart := 106698 },
  { event := event106737
    frameStart := 106698 },
  { event := event106738
    frameStart := 106698 },
  { event := event106739
    frameStart := 106698 },
  { event := event106740
    frameStart := 106698 },
  { event := event106741
    frameStart := 106698 },
  { event := event106742
    frameStart := 106698 },
  { event := event106743
    frameStart := 106698 },
  { event := event106744
    frameStart := 106698 },
  { event := event106745
    frameStart := 106698 },
  { event := event106746
    frameStart := 106746 },
  { event := event106747
    frameStart := 106746 },
  { event := event106748
    frameStart := 106746 },
  { event := event106749
    frameStart := 106746 },
  { event := event106750
    frameStart := 106746 },
  { event := event106751
    frameStart := 106746 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events416
