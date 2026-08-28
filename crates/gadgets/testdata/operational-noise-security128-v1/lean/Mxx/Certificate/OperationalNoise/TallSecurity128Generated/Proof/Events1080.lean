import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1080

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact276480RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩]

theorem exact276480RawTermsValid :
    exact276480RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276480 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7196⟩⟩) exact276480RawTerms .large 276479 .exactZero (none)

def event276481 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49477⟩⟩) 0 ⟨7196⟩ 276480

def event276482 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49477⟩⟩) 1 ⟨49476⟩ 276477

def event276483 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49477⟩⟩) (.sum [.predecessor 0 276481 .coefficient, .predecessor 1 276482 .coefficient])

def exact276484RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact276484RawTermsValid :
    exact276484RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276484 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49477⟩⟩) exact276484RawTerms .large 276483 .exactZero (none)

def event276485 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49817⟩⟩) 0 ⟨49477⟩ 276484

def event276486 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49817⟩⟩) 1 ⟨49816⟩ 276461

def event276487 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49817⟩⟩) (.product (.predecessor 0 276485 .coefficient) (.predecessor 1 276486 .coefficient) (⟨false, false, none, none, none⟩))

def event276488 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49817⟩⟩, .operator (⟨276484, 0⟩, ⟨276461, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49816⟩⟩]⟩, (1)⟩)

def event276489 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49817⟩⟩, .operator (⟨276484, 1⟩, ⟨276461, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49816⟩⟩]⟩, (-1)⟩)

def event276490 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49817⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49816⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49816⟩⟩) ⟨49225⟩ 276458)

def event276491 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49817⟩⟩, .relation 276490 0, ⟨[⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨49225⟩⟩]⟩, (-1)⟩)

def exact276492RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49816⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨49225⟩⟩]⟩, (-1)⟩]

theorem exact276492RawTermsValid :
    exact276492RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276492 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49817⟩⟩) exact276492RawTerms .large 276487 .exactZero (none)

def event276493 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48252⟩⟩) 0 ⟨48083⟩ 276450

def event276494 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48252⟩⟩) (.authority (.programFamilyFact))

def exact276495RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48252⟩⟩], []⟩, (1)⟩]

theorem exact276495RawTermsValid :
    exact276495RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276495 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48252⟩⟩) exact276495RawTerms (.finite 60) 276494 .exactZero (none)

def event276496 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48254⟩⟩) 0 ⟨6908⟩ 276472

def event276497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48254⟩⟩) 1 ⟨48252⟩ 276495

def event276498 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48254⟩⟩) (.product (.predecessor 0 276496 .coefficient) (.predecessor 1 276497 .coefficient) (⟨false, true, none, none, some 1⟩))

def event276499 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48254⟩⟩, .operator (⟨276472, 0⟩, ⟨276495, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact276500RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact276500RawTermsValid :
    exact276500RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276500 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48254⟩⟩) exact276500RawTerms .large 276498 .exactZero (none)

def event276501 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7231⟩⟩) 0 ⟨7177⟩ 276454

def event276502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7231⟩⟩) (.authority (.operator))

def exact276503RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩]

theorem exact276503RawTermsValid :
    exact276503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276503 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7231⟩⟩) exact276503RawTerms .large 276502 .exactZero (none)

def event276504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48255⟩⟩) 0 ⟨7231⟩ 276503

def event276505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48255⟩⟩) 1 ⟨48254⟩ 276500

def event276506 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48255⟩⟩) (.sum [.predecessor 0 276504 .coefficient, .predecessor 1 276505 .coefficient])

def exact276507RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact276507RawTermsValid :
    exact276507RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276507 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48255⟩⟩) exact276507RawTerms .large 276506 .exactZero (none)

def event276508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49821⟩⟩) 0 ⟨48255⟩ 276507

def event276509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49821⟩⟩) 1 ⟨49817⟩ 276492

def event276510 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49821⟩⟩) (.sum [.predecessor 0 276508 .coefficient, .predecessor 1 276509 .coefficient])

def exact276511RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49816⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨49225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact276511RawTermsValid :
    exact276511RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276511 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49821⟩⟩) exact276511RawTerms .large 276510 .exactZero (none)

def event276512 : Event := .preFoldPolynomial 276511 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49816⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨49225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact276513RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49816⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨49225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event276513 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨49821⟩⟩) 276512 exact276513RawTerms .large 276510 .exactZero (none)

def event276514 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨48083⟩⟩) ⟨⟨110⟩, ⟨93⟩, ⟨135⟩⟩ ⟨276356, 276514⟩

def event276515 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨48729⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48726⟩⟩]⟩) (1) 0 2 (.universal 276514 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48726⟩⟩]⟩) (none) 276513)

def event276516 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48729⟩⟩, .relation 276515 1, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩)

def event276517 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48729⟩⟩, .relation 276515 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49816⟩⟩]⟩, (-1)⟩)

def event276518 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48729⟩⟩, .relation 276515 2, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨49225⟩⟩]⟩, (1)⟩)

def event276519 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48729⟩⟩, .relation 276515 3, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact276520RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49816⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨49225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact276520RawTermsValid :
    exact276520RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276520 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48729⟩⟩) exact276520RawTerms .large 276352 (.finite 202072841853861888) (some (276354))

def event276521 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49819⟩⟩) 0 ⟨48729⟩ 276520

def event276522 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49819⟩⟩) 1 ⟨49818⟩ 276342

def event276523 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49819⟩⟩) (.sum [.predecessor 0 276521 .coefficient, .predecessor 1 276522 .coefficient])

def event276524 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49819⟩⟩, .operator (⟨276520, 0⟩, ⟨276342, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49816⟩⟩]⟩, (1)⟩)

def event276525 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49819⟩⟩, .operator (⟨276520, 2⟩, ⟨276342, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨49225⟩⟩]⟩, (-1)⟩)

def event276526 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49819⟩⟩) (.sum [.result 276520 .summary, .result 276342 .summary])

def exact276527RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact276527RawTermsValid :
    exact276527RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276527 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49819⟩⟩) exact276527RawTerms .large 276523 (.finite 32194504275408640829496428331008) (some (276526))

def event276528 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49820⟩⟩) 0 ⟨49819⟩ 276527

def event276529 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49820⟩⟩) 1 ⟨7148⟩ 15542

def event276530 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49820⟩⟩) (.product (.predecessor 0 276528 .coefficient) (.predecessor 1 276529 .coefficient) (⟨false, false, none, none, none⟩))

def event276531 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49820⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩) [⟨.result 15538 .coefficient, false, none⟩])

def event276532 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49820⟩⟩) (.product (.result 276527 .summary) (.transfer 276531) (⟨false, false, none, none, none⟩))

def event276533 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49820⟩⟩, .operator (⟨276527, 0⟩, ⟨15542, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩)

def event276534 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49820⟩⟩, .operator (⟨276527, 1⟩, ⟨15542, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (-1)⟩)

def event276535 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49820⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7147⟩⟩) ⟨7039⟩ 15535)

def event276536 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49820⟩⟩, .relation 276535 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact276537RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact276537RawTermsValid :
    exact276537RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276537 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49820⟩⟩) exact276537RawTerms .large 276530 (.finite 345685857434530723496243679576218056785920) (some (276532))

def event276538 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46545⟩⟩) 0 ⟨7177⟩ 15500

def event276539 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46545⟩⟩) 1 ⟨46544⟩ 266504

def event276540 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46545⟩⟩) (.authority (.operator))

def exact276541RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46545⟩⟩]⟩, (1)⟩]

theorem exact276541RawTermsValid :
    exact276541RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276541 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46545⟩⟩) exact276541RawTerms .large 276540 .exactZero (none)

def event276542 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47136⟩⟩) 0 ⟨46545⟩ 276541

def event276543 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47136⟩⟩) (.authority (.operator))

def exact276544RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47136⟩⟩]⟩, (1)⟩]

theorem exact276544RawTermsValid :
    exact276544RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276544 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47136⟩⟩) exact276544RawTerms (.finite 8192) 276543 .exactZero (none)

def event276545 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47138⟩⟩) 0 ⟨46890⟩ 266788

def event276546 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47138⟩⟩) 1 ⟨47136⟩ 276544

def event276547 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47138⟩⟩) (.product (.predecessor 0 276545 .coefficient) (.predecessor 1 276546 .coefficient) (⟨false, false, none, none, none⟩))

def event276548 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47138⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨47136⟩⟩]⟩) [⟨.result 276544 .coefficient, false, none⟩])

def event276549 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47138⟩⟩) (.product (.result 266788 .summary) (.transfer 276548) (⟨false, false, none, none, none⟩))

def event276550 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47138⟩⟩, .operator (⟨266788, 0⟩, ⟨276544, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47136⟩⟩]⟩, (1)⟩)

def event276551 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47138⟩⟩, .operator (⟨266788, 1⟩, ⟨276544, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47136⟩⟩]⟩, (-1)⟩)

def event276552 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47138⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47136⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47136⟩⟩) ⟨46545⟩ 276541)

def event276553 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47138⟩⟩, .relation 276552 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45402⟩⟩], [⟨.program ⟨257⟩, ⟨46545⟩⟩]⟩, (-1)⟩)

def exact276554RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47136⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45402⟩⟩], [⟨.program ⟨257⟩, ⟨46545⟩⟩]⟩, (-1)⟩]

theorem exact276554RawTermsValid :
    exact276554RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276554 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47138⟩⟩) exact276554RawTerms .large 276547 (.finite 32194307824962751379413684715520) (some (276549))

def event276555 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46046⟩⟩) 0 ⟨45403⟩ 12850

def event276556 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46046⟩⟩) (.authority (.relationPreimageSource ⟨91⟩))

def exact276557RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46046⟩⟩]⟩, (1)⟩]

theorem exact276557RawTermsValid :
    exact276557RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276557 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46046⟩⟩) exact276557RawTerms (.finite 5647228698) 276556 .exactZero (none)

def event276558 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46048⟩⟩) 0 ⟨46046⟩ 276557

def event276559 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46048⟩⟩) 1 ⟨2370⟩ 4

def event276560 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46048⟩⟩) (.scale (.predecessor 0 276558 .coefficient) (.value (.predecessor 1 276559 .coefficient)))

def exact276561RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46046⟩⟩]⟩, (1)⟩]

theorem exact276561RawTermsValid :
    exact276561RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276561 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46048⟩⟩) exact276561RawTerms (.finite 5647228698) 276560 .exactZero (none)

def event276562 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46049⟩⟩) 0 ⟨5449⟩ 266120

def event276563 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46049⟩⟩) 1 ⟨46048⟩ 276561

def event276564 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46049⟩⟩) (.product (.predecessor 0 276562 .coefficient) (.predecessor 1 276563 .coefficient) (⟨false, false, none, none, none⟩))

def event276565 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46049⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨46046⟩⟩]⟩) [⟨.result 276557 .coefficient, false, none⟩])

def event276566 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46049⟩⟩) (.product (.result 266120 .summary) (.transfer 276565) (⟨false, false, none, none, none⟩))

def event276567 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46049⟩⟩, .operator (⟨266120, 0⟩, ⟨276561, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46046⟩⟩]⟩, (1)⟩)

def event276568 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨46047⟩⟩)

def event276569 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event276570 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event276571 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨387⟩⟩) (.authority (.operator))

def event276572 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨387⟩⟩) (.finite 2)

def event276573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event276574 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event276575 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event276576 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event276577 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 276576

def event276578 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 276574

def event276579 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 276577 .coefficient) (.value (.predecessor 1 276578 .coefficient)))

def event276580 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event276581 : Event := .predecessor (⟨.program ⟨257⟩, ⟨394⟩⟩) 0 ⟨392⟩ 276580

def event276582 : Event := .predecessor (⟨.program ⟨257⟩, ⟨394⟩⟩) 1 ⟨387⟩ 276572

def event276583 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨394⟩⟩) (.sum [.predecessor 0 276581 .coefficient, .predecessor 1 276582 .coefficient])

def event276584 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨394⟩⟩) (.finite 655342)

def event276585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5445⟩⟩) 0 ⟨394⟩ 276584

def event276586 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5445⟩⟩) 1 ⟨5426⟩ 276570

def event276587 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5445⟩⟩) (.identity (.predecessor 1 276586 .coefficient))

def event276588 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5445⟩⟩) (.finite 655360)

def event276589 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44954⟩⟩) 0 ⟨5445⟩ 276588

def event276590 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44954⟩⟩) (.authority (.programFamilyFact))

def exact276591RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨44954⟩⟩], []⟩, (1)⟩]

theorem exact276591RawTermsValid :
    exact276591RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276591 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44954⟩⟩) exact276591RawTerms (.finite 58) 276590 .exactZero (none)

def event276592 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14656⟩⟩) 0 ⟨5445⟩ 276588

def event276593 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14656⟩⟩) (.authority (.programFamilyFact))

def exact276594RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14656⟩⟩], []⟩, (1)⟩]

theorem exact276594RawTermsValid :
    exact276594RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276594 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14656⟩⟩) exact276594RawTerms (.finite 58) 276593 .exactZero (none)

def event276595 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44955⟩⟩) 0 ⟨14656⟩ 276594

def event276596 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44955⟩⟩) 1 ⟨44954⟩ 276591

def event276597 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44955⟩⟩) (.product (.predecessor 0 276595 .coefficient) (.predecessor 1 276596 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event276598 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44955⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14656⟩⟩, ⟨.program ⟨257⟩, ⟨44954⟩⟩], []⟩) [⟨.result 276594 .coefficient, true, some 1⟩, ⟨.result 276591 .coefficient, true, some 1⟩])

def event276599 : Event := .survivorFold (1) 276598

def exact276600RawTerms : List Term := []

theorem exact276600RawTermsValid :
    exact276600RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276600 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44955⟩⟩) exact276600RawTerms (.finite 3364) 276597 (.finite 3364) (some (276598))

def event276601 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44956⟩⟩) 0 ⟨44955⟩ 276600

def event276602 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44956⟩⟩) (.identity (.predecessor 0 276601 .coefficient))

def event276603 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44956⟩⟩) (.finite 3364)

def event276604 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45402⟩⟩) 0 ⟨44956⟩ 276603

def event276605 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45402⟩⟩) (.authority (.programFamilyFact))

def exact276606RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45402⟩⟩], []⟩, (1)⟩]

theorem exact276606RawTermsValid :
    exact276606RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276606 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45402⟩⟩) exact276606RawTerms (.finite 58) 276605 .exactZero (none)

def event276607 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45403⟩⟩) 0 ⟨45402⟩ 276606

def event276608 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45403⟩⟩) (.identity (.predecessor 0 276607 .coefficient))

def event276609 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45403⟩⟩) (.finite 58)

def event276610 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46046⟩⟩) 0 ⟨45403⟩ 276609

def event276611 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46046⟩⟩) (.authority (.relationPreimageSource ⟨91⟩))

def exact276612RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46046⟩⟩]⟩, (1)⟩]

theorem exact276612RawTermsValid :
    exact276612RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276612 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46046⟩⟩) exact276612RawTerms (.finite 5647228698) 276611 .exactZero (none)

def event276613 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact276614RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact276614RawTermsValid :
    exact276614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276614 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact276614RawTerms .large 276613 .exactZero (none)

def event276615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46047⟩⟩) 0 ⟨35⟩ 276614

def event276616 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46047⟩⟩) 1 ⟨46046⟩ 276612

def event276617 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46047⟩⟩) (.product (.predecessor 0 276615 .coefficient) (.predecessor 1 276616 .coefficient) (⟨false, false, none, none, none⟩))

def event276618 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46047⟩⟩, .operator (⟨276614, 0⟩, ⟨276612, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46046⟩⟩]⟩, (1)⟩)

def exact276619RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46046⟩⟩]⟩, (1)⟩]

theorem exact276619RawTermsValid :
    exact276619RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276619 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46047⟩⟩) exact276619RawTerms .large 276617 .exactZero (none)

def event276620 : Event := .preFoldPolynomial 276619 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46046⟩⟩]⟩, (1)⟩] .exactZero none

def exact276621RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46046⟩⟩]⟩, (1)⟩]

def event276621 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨46047⟩⟩) 276620 exact276621RawTerms .large 276617 .exactZero (none)

def event276622 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨47141⟩⟩)

def event276623 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event276624 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event276625 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨387⟩⟩) (.authority (.operator))

def event276626 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨387⟩⟩) (.finite 2)

def event276627 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event276628 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event276629 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event276630 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event276631 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 276630

def event276632 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 276628

def event276633 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 276631 .coefficient) (.value (.predecessor 1 276632 .coefficient)))

def event276634 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event276635 : Event := .predecessor (⟨.program ⟨257⟩, ⟨394⟩⟩) 0 ⟨392⟩ 276634

def event276636 : Event := .predecessor (⟨.program ⟨257⟩, ⟨394⟩⟩) 1 ⟨387⟩ 276626

def event276637 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨394⟩⟩) (.sum [.predecessor 0 276635 .coefficient, .predecessor 1 276636 .coefficient])

def event276638 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨394⟩⟩) (.finite 655342)

def event276639 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5445⟩⟩) 0 ⟨394⟩ 276638

def event276640 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5445⟩⟩) 1 ⟨5426⟩ 276624

def event276641 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5445⟩⟩) (.identity (.predecessor 1 276640 .coefficient))

def event276642 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5445⟩⟩) (.finite 655360)

def event276643 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44954⟩⟩) 0 ⟨5445⟩ 276642

def event276644 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44954⟩⟩) (.authority (.programFamilyFact))

def exact276645RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨44954⟩⟩], []⟩, (1)⟩]

theorem exact276645RawTermsValid :
    exact276645RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276645 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44954⟩⟩) exact276645RawTerms (.finite 58) 276644 .exactZero (none)

def event276646 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14656⟩⟩) 0 ⟨5445⟩ 276642

def event276647 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14656⟩⟩) (.authority (.programFamilyFact))

def exact276648RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14656⟩⟩], []⟩, (1)⟩]

theorem exact276648RawTermsValid :
    exact276648RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276648 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14656⟩⟩) exact276648RawTerms (.finite 58) 276647 .exactZero (none)

def event276649 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44955⟩⟩) 0 ⟨14656⟩ 276648

def event276650 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44955⟩⟩) 1 ⟨44954⟩ 276645

def event276651 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44955⟩⟩) (.product (.predecessor 0 276649 .coefficient) (.predecessor 1 276650 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event276652 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44955⟩⟩, .operator (⟨276648, 0⟩, ⟨276645, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14656⟩⟩, ⟨.program ⟨257⟩, ⟨44954⟩⟩], []⟩, (1)⟩)

def exact276653RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14656⟩⟩, ⟨.program ⟨257⟩, ⟨44954⟩⟩], []⟩, (1)⟩]

theorem exact276653RawTermsValid :
    exact276653RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276653 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44955⟩⟩) exact276653RawTerms (.finite 3364) 276651 .exactZero (none)

def event276654 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44956⟩⟩) 0 ⟨44955⟩ 276653

def event276655 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44956⟩⟩) (.identity (.predecessor 0 276654 .coefficient))

def event276656 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44956⟩⟩) (.finite 3364)

def event276657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45402⟩⟩) 0 ⟨44956⟩ 276656

def event276658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45402⟩⟩) (.authority (.programFamilyFact))

def exact276659RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45402⟩⟩], []⟩, (1)⟩]

theorem exact276659RawTermsValid :
    exact276659RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276659 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45402⟩⟩) exact276659RawTerms (.finite 58) 276658 .exactZero (none)

def event276660 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45403⟩⟩) 0 ⟨45402⟩ 276659

def event276661 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45403⟩⟩) (.identity (.predecessor 0 276660 .coefficient))

def event276662 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45403⟩⟩) (.finite 58)

def event276663 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46544⟩⟩) 0 ⟨45403⟩ 276662

def event276664 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46544⟩⟩) (.authority (.programFamilyFact))

def event276665 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46544⟩⟩) (.finite 3720)

def event276666 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event276667 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46545⟩⟩) 0 ⟨7177⟩ 276666

def event276668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46545⟩⟩) 1 ⟨46544⟩ 276665

def event276669 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46545⟩⟩) (.authority (.operator))

def exact276670RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46545⟩⟩]⟩, (1)⟩]

theorem exact276670RawTermsValid :
    exact276670RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276670 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46545⟩⟩) exact276670RawTerms .large 276669 .exactZero (none)

def event276671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47136⟩⟩) 0 ⟨46545⟩ 276670

def event276672 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47136⟩⟩) (.authority (.operator))

def exact276673RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47136⟩⟩]⟩, (1)⟩]

theorem exact276673RawTermsValid :
    exact276673RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276673 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47136⟩⟩) exact276673RawTerms (.finite 8192) 276672 .exactZero (none)

def event276674 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event276675 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event276676 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46794⟩⟩) 0 ⟨45403⟩ 276662

def event276677 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46794⟩⟩) 1 ⟨136⟩ 276675

def event276678 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46794⟩⟩) (.sum [.predecessor 0 276676 .coefficient, .predecessor 1 276677 .coefficient])

def event276679 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46794⟩⟩) (.finite 58)

def event276680 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46795⟩⟩) 0 ⟨46794⟩ 276679

def event276681 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46795⟩⟩) (.identity (.predecessor 0 276680 .coefficient))

def exact276682RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45402⟩⟩], []⟩, (1)⟩]

theorem exact276682RawTermsValid :
    exact276682RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276682 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46795⟩⟩) exact276682RawTerms (.finite 58) 276681 .exactZero (none)

def event276683 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact276684RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact276684RawTermsValid :
    exact276684RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276684 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact276684RawTerms .large 276683 .exactZero (none)

def event276685 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46796⟩⟩) 0 ⟨6908⟩ 276684

def event276686 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46796⟩⟩) 1 ⟨46795⟩ 276682

def event276687 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46796⟩⟩) (.product (.predecessor 0 276685 .coefficient) (.predecessor 1 276686 .coefficient) (⟨false, false, none, none, none⟩))

def event276688 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46796⟩⟩, .operator (⟨276684, 0⟩, ⟨276682, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact276689RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact276689RawTermsValid :
    exact276689RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276689 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46796⟩⟩) exact276689RawTerms .large 276687 .exactZero (none)

def event276690 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7195⟩⟩) 0 ⟨7177⟩ 276666

def event276691 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7195⟩⟩) (.authority (.operator))

def exact276692RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩]

theorem exact276692RawTermsValid :
    exact276692RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276692 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7195⟩⟩) exact276692RawTerms .large 276691 .exactZero (none)

def event276693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46797⟩⟩) 0 ⟨7195⟩ 276692

def event276694 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46797⟩⟩) 1 ⟨46796⟩ 276689

def event276695 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46797⟩⟩) (.sum [.predecessor 0 276693 .coefficient, .predecessor 1 276694 .coefficient])

def exact276696RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact276696RawTermsValid :
    exact276696RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276696 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46797⟩⟩) exact276696RawTerms .large 276695 .exactZero (none)

def event276697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47137⟩⟩) 0 ⟨46797⟩ 276696

def event276698 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47137⟩⟩) 1 ⟨47136⟩ 276673

def event276699 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47137⟩⟩) (.product (.predecessor 0 276697 .coefficient) (.predecessor 1 276698 .coefficient) (⟨false, false, none, none, none⟩))

def event276700 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47137⟩⟩, .operator (⟨276696, 0⟩, ⟨276673, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47136⟩⟩]⟩, (1)⟩)

def event276701 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47137⟩⟩, .operator (⟨276696, 1⟩, ⟨276673, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47136⟩⟩]⟩, (-1)⟩)

def event276702 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47137⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨45402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47136⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47136⟩⟩) ⟨46545⟩ 276670)

def event276703 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47137⟩⟩, .relation 276702 0, ⟨[⟨.program ⟨257⟩, ⟨45402⟩⟩], [⟨.program ⟨257⟩, ⟨46545⟩⟩]⟩, (-1)⟩)

def exact276704RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47136⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45402⟩⟩], [⟨.program ⟨257⟩, ⟨46545⟩⟩]⟩, (-1)⟩]

theorem exact276704RawTermsValid :
    exact276704RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276704 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47137⟩⟩) exact276704RawTerms .large 276699 .exactZero (none)

def event276705 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45572⟩⟩) 0 ⟨45403⟩ 276662

def event276706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45572⟩⟩) (.authority (.programFamilyFact))

def exact276707RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45572⟩⟩], []⟩, (1)⟩]

theorem exact276707RawTermsValid :
    exact276707RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276707 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45572⟩⟩) exact276707RawTerms (.finite 58) 276706 .exactZero (none)

def event276708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45574⟩⟩) 0 ⟨6908⟩ 276684

def event276709 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45574⟩⟩) 1 ⟨45572⟩ 276707

def event276710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45574⟩⟩) (.product (.predecessor 0 276708 .coefficient) (.predecessor 1 276709 .coefficient) (⟨false, true, none, none, some 1⟩))

def event276711 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45574⟩⟩, .operator (⟨276684, 0⟩, ⟨276707, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45572⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact276712RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45572⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact276712RawTermsValid :
    exact276712RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276712 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45574⟩⟩) exact276712RawTerms .large 276710 .exactZero (none)

def event276713 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7229⟩⟩) 0 ⟨7177⟩ 276666

def event276714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7229⟩⟩) (.authority (.operator))

def exact276715RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩]

theorem exact276715RawTermsValid :
    exact276715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276715 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7229⟩⟩) exact276715RawTerms .large 276714 .exactZero (none)

def event276716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45575⟩⟩) 0 ⟨7229⟩ 276715

def event276717 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45575⟩⟩) 1 ⟨45574⟩ 276712

def event276718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45575⟩⟩) (.sum [.predecessor 0 276716 .coefficient, .predecessor 1 276717 .coefficient])

def exact276719RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45572⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact276719RawTermsValid :
    exact276719RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276719 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45575⟩⟩) exact276719RawTerms .large 276718 .exactZero (none)

def event276720 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47141⟩⟩) 0 ⟨45575⟩ 276719

def event276721 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47141⟩⟩) 1 ⟨47137⟩ 276704

def event276722 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47141⟩⟩) (.sum [.predecessor 0 276720 .coefficient, .predecessor 1 276721 .coefficient])

def exact276723RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47136⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45402⟩⟩], [⟨.program ⟨257⟩, ⟨46545⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45572⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact276723RawTermsValid :
    exact276723RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276723 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47141⟩⟩) exact276723RawTerms .large 276722 .exactZero (none)

def event276724 : Event := .preFoldPolynomial 276723 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47136⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45402⟩⟩], [⟨.program ⟨257⟩, ⟨46545⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45572⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact276725RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47136⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45402⟩⟩], [⟨.program ⟨257⟩, ⟨46545⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45572⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event276725 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨47141⟩⟩) 276724 exact276725RawTerms .large 276722 .exactZero (none)

def event276726 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨45403⟩⟩) ⟨⟨108⟩, ⟨91⟩, ⟨135⟩⟩ ⟨276568, 276726⟩

def event276727 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨46049⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46046⟩⟩]⟩) (1) 0 2 (.universal 276726 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46046⟩⟩]⟩) (none) 276725)

def event276728 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46049⟩⟩, .relation 276727 1, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩)

def event276729 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46049⟩⟩, .relation 276727 0, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47136⟩⟩]⟩, (-1)⟩)

def event276730 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46049⟩⟩, .relation 276727 2, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45402⟩⟩], [⟨.program ⟨257⟩, ⟨46545⟩⟩]⟩, (1)⟩)

def event276731 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46049⟩⟩, .relation 276727 3, ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45572⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact276732RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47136⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45402⟩⟩], [⟨.program ⟨257⟩, ⟨46545⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨45572⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact276732RawTermsValid :
    exact276732RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event276732 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46049⟩⟩) exact276732RawTerms .large 276564 (.finite 202072841853861888) (some (276566))

def event276733 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47139⟩⟩) 0 ⟨46049⟩ 276732

def event276734 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47139⟩⟩) 1 ⟨47138⟩ 276554

def event276735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47139⟩⟩) (.sum [.predecessor 0 276733 .coefficient, .predecessor 1 276734 .coefficient])

def eventLeaf17280 : Array AnnotatedEvent := #[
  { event := event276480
    frameStart := 276410 },
  { event := event276481
    frameStart := 276410 },
  { event := event276482
    frameStart := 276410 },
  { event := event276483
    frameStart := 276410 },
  { event := event276484
    frameStart := 276410 },
  { event := event276485
    frameStart := 276410 },
  { event := event276486
    frameStart := 276410 },
  { event := event276487
    frameStart := 276410 },
  { event := event276488
    frameStart := 276410 },
  { event := event276489
    frameStart := 276410 },
  { event := event276490
    frameStart := 276410 },
  { event := event276491
    frameStart := 276410 },
  { event := event276492
    frameStart := 276410 },
  { event := event276493
    frameStart := 276410 },
  { event := event276494
    frameStart := 276410 },
  { event := event276495
    frameStart := 276410 }
]

def eventLeaf17281 : Array AnnotatedEvent := #[
  { event := event276496
    frameStart := 276410 },
  { event := event276497
    frameStart := 276410 },
  { event := event276498
    frameStart := 276410 },
  { event := event276499
    frameStart := 276410 },
  { event := event276500
    frameStart := 276410 },
  { event := event276501
    frameStart := 276410 },
  { event := event276502
    frameStart := 276410 },
  { event := event276503
    frameStart := 276410 },
  { event := event276504
    frameStart := 276410 },
  { event := event276505
    frameStart := 276410 },
  { event := event276506
    frameStart := 276410 },
  { event := event276507
    frameStart := 276410 },
  { event := event276508
    frameStart := 276410 },
  { event := event276509
    frameStart := 276410 },
  { event := event276510
    frameStart := 276410 },
  { event := event276511
    frameStart := 276410 }
]

def eventLeaf17282 : Array AnnotatedEvent := #[
  { event := event276512
    frameStart := 276410 },
  { event := event276513
    frameStart := 276410 },
  { event := event276514
    frameStart := 0 },
  { event := event276515
    frameStart := 0 },
  { event := event276516
    frameStart := 0 },
  { event := event276517
    frameStart := 0 },
  { event := event276518
    frameStart := 0 },
  { event := event276519
    frameStart := 0 },
  { event := event276520
    frameStart := 0 },
  { event := event276521
    frameStart := 0 },
  { event := event276522
    frameStart := 0 },
  { event := event276523
    frameStart := 0 },
  { event := event276524
    frameStart := 0 },
  { event := event276525
    frameStart := 0 },
  { event := event276526
    frameStart := 0 },
  { event := event276527
    frameStart := 0 }
]

def eventLeaf17283 : Array AnnotatedEvent := #[
  { event := event276528
    frameStart := 0 },
  { event := event276529
    frameStart := 0 },
  { event := event276530
    frameStart := 0 },
  { event := event276531
    frameStart := 0 },
  { event := event276532
    frameStart := 0 },
  { event := event276533
    frameStart := 0 },
  { event := event276534
    frameStart := 0 },
  { event := event276535
    frameStart := 0 },
  { event := event276536
    frameStart := 0 },
  { event := event276537
    frameStart := 0 },
  { event := event276538
    frameStart := 0 },
  { event := event276539
    frameStart := 0 },
  { event := event276540
    frameStart := 0 },
  { event := event276541
    frameStart := 0 },
  { event := event276542
    frameStart := 0 },
  { event := event276543
    frameStart := 0 }
]

def eventLeaf17284 : Array AnnotatedEvent := #[
  { event := event276544
    frameStart := 0 },
  { event := event276545
    frameStart := 0 },
  { event := event276546
    frameStart := 0 },
  { event := event276547
    frameStart := 0 },
  { event := event276548
    frameStart := 0 },
  { event := event276549
    frameStart := 0 },
  { event := event276550
    frameStart := 0 },
  { event := event276551
    frameStart := 0 },
  { event := event276552
    frameStart := 0 },
  { event := event276553
    frameStart := 0 },
  { event := event276554
    frameStart := 0 },
  { event := event276555
    frameStart := 0 },
  { event := event276556
    frameStart := 0 },
  { event := event276557
    frameStart := 0 },
  { event := event276558
    frameStart := 0 },
  { event := event276559
    frameStart := 0 }
]

def eventLeaf17285 : Array AnnotatedEvent := #[
  { event := event276560
    frameStart := 0 },
  { event := event276561
    frameStart := 0 },
  { event := event276562
    frameStart := 0 },
  { event := event276563
    frameStart := 0 },
  { event := event276564
    frameStart := 0 },
  { event := event276565
    frameStart := 0 },
  { event := event276566
    frameStart := 0 },
  { event := event276567
    frameStart := 0 },
  { event := event276568
    frameStart := 276568 },
  { event := event276569
    frameStart := 276568 },
  { event := event276570
    frameStart := 276568 },
  { event := event276571
    frameStart := 276568 },
  { event := event276572
    frameStart := 276568 },
  { event := event276573
    frameStart := 276568 },
  { event := event276574
    frameStart := 276568 },
  { event := event276575
    frameStart := 276568 }
]

def eventLeaf17286 : Array AnnotatedEvent := #[
  { event := event276576
    frameStart := 276568 },
  { event := event276577
    frameStart := 276568 },
  { event := event276578
    frameStart := 276568 },
  { event := event276579
    frameStart := 276568 },
  { event := event276580
    frameStart := 276568 },
  { event := event276581
    frameStart := 276568 },
  { event := event276582
    frameStart := 276568 },
  { event := event276583
    frameStart := 276568 },
  { event := event276584
    frameStart := 276568 },
  { event := event276585
    frameStart := 276568 },
  { event := event276586
    frameStart := 276568 },
  { event := event276587
    frameStart := 276568 },
  { event := event276588
    frameStart := 276568 },
  { event := event276589
    frameStart := 276568 },
  { event := event276590
    frameStart := 276568 },
  { event := event276591
    frameStart := 276568 }
]

def eventLeaf17287 : Array AnnotatedEvent := #[
  { event := event276592
    frameStart := 276568 },
  { event := event276593
    frameStart := 276568 },
  { event := event276594
    frameStart := 276568 },
  { event := event276595
    frameStart := 276568 },
  { event := event276596
    frameStart := 276568 },
  { event := event276597
    frameStart := 276568 },
  { event := event276598
    frameStart := 276568 },
  { event := event276599
    frameStart := 276568 },
  { event := event276600
    frameStart := 276568 },
  { event := event276601
    frameStart := 276568 },
  { event := event276602
    frameStart := 276568 },
  { event := event276603
    frameStart := 276568 },
  { event := event276604
    frameStart := 276568 },
  { event := event276605
    frameStart := 276568 },
  { event := event276606
    frameStart := 276568 },
  { event := event276607
    frameStart := 276568 }
]

def eventLeaf17288 : Array AnnotatedEvent := #[
  { event := event276608
    frameStart := 276568 },
  { event := event276609
    frameStart := 276568 },
  { event := event276610
    frameStart := 276568 },
  { event := event276611
    frameStart := 276568 },
  { event := event276612
    frameStart := 276568 },
  { event := event276613
    frameStart := 276568 },
  { event := event276614
    frameStart := 276568 },
  { event := event276615
    frameStart := 276568 },
  { event := event276616
    frameStart := 276568 },
  { event := event276617
    frameStart := 276568 },
  { event := event276618
    frameStart := 276568 },
  { event := event276619
    frameStart := 276568 },
  { event := event276620
    frameStart := 276568 },
  { event := event276621
    frameStart := 276568 },
  { event := event276622
    frameStart := 276622 },
  { event := event276623
    frameStart := 276622 }
]

def eventLeaf17289 : Array AnnotatedEvent := #[
  { event := event276624
    frameStart := 276622 },
  { event := event276625
    frameStart := 276622 },
  { event := event276626
    frameStart := 276622 },
  { event := event276627
    frameStart := 276622 },
  { event := event276628
    frameStart := 276622 },
  { event := event276629
    frameStart := 276622 },
  { event := event276630
    frameStart := 276622 },
  { event := event276631
    frameStart := 276622 },
  { event := event276632
    frameStart := 276622 },
  { event := event276633
    frameStart := 276622 },
  { event := event276634
    frameStart := 276622 },
  { event := event276635
    frameStart := 276622 },
  { event := event276636
    frameStart := 276622 },
  { event := event276637
    frameStart := 276622 },
  { event := event276638
    frameStart := 276622 },
  { event := event276639
    frameStart := 276622 }
]

def eventLeaf17290 : Array AnnotatedEvent := #[
  { event := event276640
    frameStart := 276622 },
  { event := event276641
    frameStart := 276622 },
  { event := event276642
    frameStart := 276622 },
  { event := event276643
    frameStart := 276622 },
  { event := event276644
    frameStart := 276622 },
  { event := event276645
    frameStart := 276622 },
  { event := event276646
    frameStart := 276622 },
  { event := event276647
    frameStart := 276622 },
  { event := event276648
    frameStart := 276622 },
  { event := event276649
    frameStart := 276622 },
  { event := event276650
    frameStart := 276622 },
  { event := event276651
    frameStart := 276622 },
  { event := event276652
    frameStart := 276622 },
  { event := event276653
    frameStart := 276622 },
  { event := event276654
    frameStart := 276622 },
  { event := event276655
    frameStart := 276622 }
]

def eventLeaf17291 : Array AnnotatedEvent := #[
  { event := event276656
    frameStart := 276622 },
  { event := event276657
    frameStart := 276622 },
  { event := event276658
    frameStart := 276622 },
  { event := event276659
    frameStart := 276622 },
  { event := event276660
    frameStart := 276622 },
  { event := event276661
    frameStart := 276622 },
  { event := event276662
    frameStart := 276622 },
  { event := event276663
    frameStart := 276622 },
  { event := event276664
    frameStart := 276622 },
  { event := event276665
    frameStart := 276622 },
  { event := event276666
    frameStart := 276622 },
  { event := event276667
    frameStart := 276622 },
  { event := event276668
    frameStart := 276622 },
  { event := event276669
    frameStart := 276622 },
  { event := event276670
    frameStart := 276622 },
  { event := event276671
    frameStart := 276622 }
]

def eventLeaf17292 : Array AnnotatedEvent := #[
  { event := event276672
    frameStart := 276622 },
  { event := event276673
    frameStart := 276622 },
  { event := event276674
    frameStart := 276622 },
  { event := event276675
    frameStart := 276622 },
  { event := event276676
    frameStart := 276622 },
  { event := event276677
    frameStart := 276622 },
  { event := event276678
    frameStart := 276622 },
  { event := event276679
    frameStart := 276622 },
  { event := event276680
    frameStart := 276622 },
  { event := event276681
    frameStart := 276622 },
  { event := event276682
    frameStart := 276622 },
  { event := event276683
    frameStart := 276622 },
  { event := event276684
    frameStart := 276622 },
  { event := event276685
    frameStart := 276622 },
  { event := event276686
    frameStart := 276622 },
  { event := event276687
    frameStart := 276622 }
]

def eventLeaf17293 : Array AnnotatedEvent := #[
  { event := event276688
    frameStart := 276622 },
  { event := event276689
    frameStart := 276622 },
  { event := event276690
    frameStart := 276622 },
  { event := event276691
    frameStart := 276622 },
  { event := event276692
    frameStart := 276622 },
  { event := event276693
    frameStart := 276622 },
  { event := event276694
    frameStart := 276622 },
  { event := event276695
    frameStart := 276622 },
  { event := event276696
    frameStart := 276622 },
  { event := event276697
    frameStart := 276622 },
  { event := event276698
    frameStart := 276622 },
  { event := event276699
    frameStart := 276622 },
  { event := event276700
    frameStart := 276622 },
  { event := event276701
    frameStart := 276622 },
  { event := event276702
    frameStart := 276622 },
  { event := event276703
    frameStart := 276622 }
]

def eventLeaf17294 : Array AnnotatedEvent := #[
  { event := event276704
    frameStart := 276622 },
  { event := event276705
    frameStart := 276622 },
  { event := event276706
    frameStart := 276622 },
  { event := event276707
    frameStart := 276622 },
  { event := event276708
    frameStart := 276622 },
  { event := event276709
    frameStart := 276622 },
  { event := event276710
    frameStart := 276622 },
  { event := event276711
    frameStart := 276622 },
  { event := event276712
    frameStart := 276622 },
  { event := event276713
    frameStart := 276622 },
  { event := event276714
    frameStart := 276622 },
  { event := event276715
    frameStart := 276622 },
  { event := event276716
    frameStart := 276622 },
  { event := event276717
    frameStart := 276622 },
  { event := event276718
    frameStart := 276622 },
  { event := event276719
    frameStart := 276622 }
]

def eventLeaf17295 : Array AnnotatedEvent := #[
  { event := event276720
    frameStart := 276622 },
  { event := event276721
    frameStart := 276622 },
  { event := event276722
    frameStart := 276622 },
  { event := event276723
    frameStart := 276622 },
  { event := event276724
    frameStart := 276622 },
  { event := event276725
    frameStart := 276622 },
  { event := event276726
    frameStart := 0 },
  { event := event276727
    frameStart := 0 },
  { event := event276728
    frameStart := 0 },
  { event := event276729
    frameStart := 0 },
  { event := event276730
    frameStart := 0 },
  { event := event276731
    frameStart := 0 },
  { event := event276732
    frameStart := 0 },
  { event := event276733
    frameStart := 0 },
  { event := event276734
    frameStart := 0 },
  { event := event276735
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1080
