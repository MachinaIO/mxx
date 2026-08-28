import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events080

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event20480 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26830⟩⟩, .relation 20479 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact20481RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20481RawTermsValid :
    exact20481RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20481 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26830⟩⟩) exact20481RawTerms .large 20474 (.finite 4741336194231092170536779776) (some (20476))

def event20482 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23795⟩⟩) 0 ⟨6689⟩ 5477

def event20483 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23795⟩⟩) 1 ⟨23794⟩ 14460

def event20484 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23795⟩⟩) (.authority (.operator))

def exact20485RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23795⟩⟩]⟩, (1)⟩]

theorem exact20485RawTermsValid :
    exact20485RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20485 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23795⟩⟩) exact20485RawTerms .large 20484 .exactZero (none)

def event20486 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26609⟩⟩) 0 ⟨23795⟩ 20485

def event20487 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26609⟩⟩) (.authority (.operator))

def exact20488RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26609⟩⟩]⟩, (1)⟩]

theorem exact20488RawTermsValid :
    exact20488RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20488 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26609⟩⟩) exact20488RawTerms (.finite 8192) 20487 .exactZero (none)

def event20489 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26611⟩⟩) 0 ⟨25010⟩ 14763

def event20490 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26611⟩⟩) 1 ⟨26609⟩ 20488

def event20491 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26611⟩⟩) (.product (.predecessor 0 20489 .coefficient) (.predecessor 1 20490 .coefficient) (⟨false, false, none, none, none⟩))

def event20492 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26611⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26609⟩⟩]⟩) [⟨.result 20488 .coefficient, false, none⟩])

def event20493 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26611⟩⟩) (.product (.result 14763 .summary) (.transfer 20492) (⟨false, false, none, none, none⟩))

def event20494 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26611⟩⟩, .operator (⟨14763, 1⟩, ⟨20488, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26609⟩⟩]⟩, (-1)⟩)

def event20495 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26611⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26609⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26609⟩⟩) ⟨23795⟩ 20485)

def event20496 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26611⟩⟩, .relation 20495 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨23795⟩⟩]⟩, (-1)⟩)

def event20497 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26611⟩⟩, .operator (⟨14763, 0⟩, ⟨20488, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26609⟩⟩]⟩, (1)⟩)

def exact20498RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26609⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨23795⟩⟩]⟩, (-1)⟩]

theorem exact20498RawTermsValid :
    exact20498RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20498 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26611⟩⟩) exact20498RawTerms .large 20491 (.finite 1291900378790628425728) (some (20493))

def event20499 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20480⟩⟩) 0 ⟨14970⟩ 436

def event20500 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20480⟩⟩) (.authority (.relationPreimageSource ⟨29⟩))

def exact20501RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20480⟩⟩]⟩, (1)⟩]

theorem exact20501RawTermsValid :
    exact20501RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20501 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20480⟩⟩) exact20501RawTerms (.finite 136065468) 20500 .exactZero (none)

def event20502 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20482⟩⟩) 0 ⟨20480⟩ 20501

def event20503 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20482⟩⟩) 1 ⟨2348⟩ 4

def event20504 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20482⟩⟩) (.scale (.predecessor 0 20502 .coefficient) (.value (.predecessor 1 20503 .coefficient)))

def exact20505RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20480⟩⟩]⟩, (1)⟩]

theorem exact20505RawTermsValid :
    exact20505RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20505 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20482⟩⟩) exact20505RawTerms (.finite 136065468) 20504 .exactZero (none)

def event20506 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20483⟩⟩) 0 ⟨5565⟩ 6561

def event20507 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20483⟩⟩) 1 ⟨20482⟩ 20505

def event20508 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20483⟩⟩) (.product (.predecessor 0 20506 .coefficient) (.predecessor 1 20507 .coefficient) (⟨false, false, none, none, none⟩))

def event20509 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20483⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20480⟩⟩]⟩) [⟨.result 20501 .coefficient, false, none⟩])

def event20510 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20483⟩⟩) (.product (.result 6561 .summary) (.transfer 20509) (⟨false, false, none, none, none⟩))

def event20511 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20483⟩⟩, .operator (⟨6561, 0⟩, ⟨20505, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20480⟩⟩]⟩, (1)⟩)

def event20512 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20481⟩⟩)

def event20513 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event20514 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event20515 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.authority (.operator))

def event20516 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.finite 6)

def event20517 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event20518 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event20519 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event20520 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event20521 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 20520

def event20522 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 20518

def event20523 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 20521 .coefficient) (.value (.predecessor 1 20522 .coefficient)))

def event20524 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event20525 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 0 ⟨5503⟩ 20524

def event20526 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 1 ⟨5245⟩ 20516

def event20527 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.sum [.predecessor 0 20525 .coefficient, .predecessor 1 20526 .coefficient])

def event20528 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.finite 223)

def event20529 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 0 ⟨5516⟩ 20528

def event20530 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 1 ⟨961⟩ 20514

def event20531 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.identity (.predecessor 1 20530 .coefficient))

def event20532 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.finite 224)

def event20533 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10708⟩⟩) 0 ⟨5560⟩ 20532

def event20534 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10708⟩⟩) (.authority (.programFamilyFact))

def exact20535RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10708⟩⟩], []⟩, (1)⟩]

theorem exact20535RawTermsValid :
    exact20535RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20535 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10708⟩⟩) exact20535RawTerms (.finite 3) 20534 .exactZero (none)

def event20536 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9525⟩⟩) 0 ⟨5560⟩ 20532

def event20537 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9525⟩⟩) (.authority (.programFamilyFact))

def exact20538RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩], []⟩, (1)⟩]

theorem exact20538RawTermsValid :
    exact20538RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20538 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9525⟩⟩) exact20538RawTerms (.finite 3) 20537 .exactZero (none)

def event20539 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10709⟩⟩) 0 ⟨9525⟩ 20538

def event20540 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10709⟩⟩) 1 ⟨10708⟩ 20535

def event20541 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10709⟩⟩) (.product (.predecessor 0 20539 .coefficient) (.predecessor 1 20540 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event20542 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10709⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], []⟩) [⟨.result 20538 .coefficient, true, some 1⟩, ⟨.result 20535 .coefficient, true, some 1⟩])

def event20543 : Event := .survivorFold (1) 20542

def exact20544RawTerms : List Term := []

theorem exact20544RawTermsValid :
    exact20544RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20544 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10709⟩⟩) exact20544RawTerms (.finite 9) 20541 (.finite 9) (some (20542))

def event20545 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10710⟩⟩) 0 ⟨10709⟩ 20544

def event20546 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10710⟩⟩) (.identity (.predecessor 0 20545 .coefficient))

def event20547 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10710⟩⟩) (.finite 9)

def event20548 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14969⟩⟩) 0 ⟨10710⟩ 20547

def event20549 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14969⟩⟩) (.authority (.programFamilyFact))

def exact20550RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], []⟩, (1)⟩]

theorem exact20550RawTermsValid :
    exact20550RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20550 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14969⟩⟩) exact20550RawTerms (.finite 3) 20549 .exactZero (none)

def event20551 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14970⟩⟩) 0 ⟨14969⟩ 20550

def event20552 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14970⟩⟩) (.identity (.predecessor 0 20551 .coefficient))

def event20553 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14970⟩⟩) (.finite 3)

def event20554 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20480⟩⟩) 0 ⟨14970⟩ 20553

def event20555 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20480⟩⟩) (.authority (.relationPreimageSource ⟨29⟩))

def exact20556RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20480⟩⟩]⟩, (1)⟩]

theorem exact20556RawTermsValid :
    exact20556RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20556 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20480⟩⟩) exact20556RawTerms (.finite 136065468) 20555 .exactZero (none)

def event20557 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact20558RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact20558RawTermsValid :
    exact20558RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20558 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact20558RawTerms .large 20557 .exactZero (none)

def event20559 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20481⟩⟩) 0 ⟨6⟩ 20558

def event20560 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20481⟩⟩) 1 ⟨20480⟩ 20556

def event20561 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20481⟩⟩) (.product (.predecessor 0 20559 .coefficient) (.predecessor 1 20560 .coefficient) (⟨false, false, none, none, none⟩))

def event20562 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20481⟩⟩, .operator (⟨20558, 0⟩, ⟨20556, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20480⟩⟩]⟩, (1)⟩)

def exact20563RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20480⟩⟩]⟩, (1)⟩]

theorem exact20563RawTermsValid :
    exact20563RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20563 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20481⟩⟩) exact20563RawTerms .large 20561 .exactZero (none)

def event20564 : Event := .preFoldPolynomial 20563 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20480⟩⟩]⟩, (1)⟩] .exactZero none

def exact20565RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20480⟩⟩]⟩, (1)⟩]

def event20565 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20481⟩⟩) 20564 exact20565RawTerms .large 20561 .exactZero (none)

def event20566 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26615⟩⟩)

def event20567 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event20568 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event20569 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.authority (.operator))

def event20570 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.finite 6)

def event20571 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event20572 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event20573 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event20574 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event20575 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 20574

def event20576 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 20572

def event20577 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 20575 .coefficient) (.value (.predecessor 1 20576 .coefficient)))

def event20578 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event20579 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 0 ⟨5503⟩ 20578

def event20580 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 1 ⟨5245⟩ 20570

def event20581 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.sum [.predecessor 0 20579 .coefficient, .predecessor 1 20580 .coefficient])

def event20582 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.finite 223)

def event20583 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 0 ⟨5516⟩ 20582

def event20584 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 1 ⟨961⟩ 20568

def event20585 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.identity (.predecessor 1 20584 .coefficient))

def event20586 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.finite 224)

def event20587 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10708⟩⟩) 0 ⟨5560⟩ 20586

def event20588 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10708⟩⟩) (.authority (.programFamilyFact))

def exact20589RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10708⟩⟩], []⟩, (1)⟩]

theorem exact20589RawTermsValid :
    exact20589RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20589 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10708⟩⟩) exact20589RawTerms (.finite 3) 20588 .exactZero (none)

def event20590 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9525⟩⟩) 0 ⟨5560⟩ 20586

def event20591 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9525⟩⟩) (.authority (.programFamilyFact))

def exact20592RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩], []⟩, (1)⟩]

theorem exact20592RawTermsValid :
    exact20592RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20592 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9525⟩⟩) exact20592RawTerms (.finite 3) 20591 .exactZero (none)

def event20593 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10709⟩⟩) 0 ⟨9525⟩ 20592

def event20594 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10709⟩⟩) 1 ⟨10708⟩ 20589

def event20595 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10709⟩⟩) (.product (.predecessor 0 20593 .coefficient) (.predecessor 1 20594 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event20596 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10709⟩⟩, .operator (⟨20592, 0⟩, ⟨20589, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], []⟩, (1)⟩)

def exact20597RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], []⟩, (1)⟩]

theorem exact20597RawTermsValid :
    exact20597RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20597 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10709⟩⟩) exact20597RawTerms (.finite 9) 20595 .exactZero (none)

def event20598 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10710⟩⟩) 0 ⟨10709⟩ 20597

def event20599 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10710⟩⟩) (.identity (.predecessor 0 20598 .coefficient))

def event20600 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10710⟩⟩) (.finite 9)

def event20601 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14969⟩⟩) 0 ⟨10710⟩ 20600

def event20602 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14969⟩⟩) (.authority (.programFamilyFact))

def exact20603RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], []⟩, (1)⟩]

theorem exact20603RawTermsValid :
    exact20603RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20603 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14969⟩⟩) exact20603RawTerms (.finite 3) 20602 .exactZero (none)

def event20604 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14970⟩⟩) 0 ⟨14969⟩ 20603

def event20605 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14970⟩⟩) (.identity (.predecessor 0 20604 .coefficient))

def event20606 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14970⟩⟩) (.finite 3)

def event20607 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23794⟩⟩) 0 ⟨14970⟩ 20606

def event20608 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23794⟩⟩) (.authority (.programFamilyFact))

def event20609 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23794⟩⟩) (.finite 3720)

def event20610 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event20611 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23795⟩⟩) 0 ⟨6689⟩ 20610

def event20612 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23795⟩⟩) 1 ⟨23794⟩ 20609

def event20613 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23795⟩⟩) (.authority (.operator))

def exact20614RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23795⟩⟩]⟩, (1)⟩]

theorem exact20614RawTermsValid :
    exact20614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20614 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23795⟩⟩) exact20614RawTerms .large 20613 .exactZero (none)

def event20615 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26609⟩⟩) 0 ⟨23795⟩ 20614

def event20616 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26609⟩⟩) (.authority (.operator))

def exact20617RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26609⟩⟩]⟩, (1)⟩]

theorem exact20617RawTermsValid :
    exact20617RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20617 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26609⟩⟩) exact20617RawTerms (.finite 8192) 20616 .exactZero (none)

def event20618 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event20619 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event20620 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15009⟩⟩) 0 ⟨14970⟩ 20606

def event20621 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15009⟩⟩) 1 ⟨110⟩ 20619

def event20622 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15009⟩⟩) (.sum [.predecessor 0 20620 .coefficient, .predecessor 1 20621 .coefficient])

def event20623 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15009⟩⟩) (.finite 3)

def event20624 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15010⟩⟩) 0 ⟨15009⟩ 20623

def event20625 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15010⟩⟩) (.identity (.predecessor 0 20624 .coefficient))

def exact20626RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], []⟩, (1)⟩]

theorem exact20626RawTermsValid :
    exact20626RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20626 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15010⟩⟩) exact20626RawTerms (.finite 3) 20625 .exactZero (none)

def event20627 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact20628RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact20628RawTermsValid :
    exact20628RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20628 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact20628RawTerms .large 20627 .exactZero (none)

def event20629 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15011⟩⟩) 0 ⟨6544⟩ 20628

def event20630 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15011⟩⟩) 1 ⟨15010⟩ 20626

def event20631 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15011⟩⟩) (.product (.predecessor 0 20629 .coefficient) (.predecessor 1 20630 .coefficient) (⟨false, false, none, none, none⟩))

def event20632 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15011⟩⟩, .operator (⟨20628, 0⟩, ⟨20626, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact20633RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact20633RawTermsValid :
    exact20633RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20633 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15011⟩⟩) exact20633RawTerms .large 20631 .exactZero (none)

def event20634 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6691⟩⟩) 0 ⟨6689⟩ 20610

def event20635 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6691⟩⟩) (.authority (.operator))

def exact20636RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩]

theorem exact20636RawTermsValid :
    exact20636RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20636 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6691⟩⟩) exact20636RawTerms .large 20635 .exactZero (none)

def event20637 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15012⟩⟩) 0 ⟨6691⟩ 20636

def event20638 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15012⟩⟩) 1 ⟨15011⟩ 20633

def event20639 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15012⟩⟩) (.sum [.predecessor 0 20637 .coefficient, .predecessor 1 20638 .coefficient])

def exact20640RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20640RawTermsValid :
    exact20640RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20640 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15012⟩⟩) exact20640RawTerms .large 20639 .exactZero (none)

def event20641 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26610⟩⟩) 0 ⟨15012⟩ 20640

def event20642 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26610⟩⟩) 1 ⟨26609⟩ 20617

def event20643 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26610⟩⟩) (.product (.predecessor 0 20641 .coefficient) (.predecessor 1 20642 .coefficient) (⟨false, false, none, none, none⟩))

def event20644 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26610⟩⟩, .operator (⟨20640, 1⟩, ⟨20617, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26609⟩⟩]⟩, (-1)⟩)

def event20645 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26610⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26609⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26609⟩⟩) ⟨23795⟩ 20614)

def event20646 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26610⟩⟩, .relation 20645 0, ⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨23795⟩⟩]⟩, (-1)⟩)

def event20647 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26610⟩⟩, .operator (⟨20640, 0⟩, ⟨20617, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26609⟩⟩]⟩, (1)⟩)

def exact20648RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26609⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨23795⟩⟩]⟩, (-1)⟩]

theorem exact20648RawTermsValid :
    exact20648RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20648 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26610⟩⟩) exact20648RawTerms .large 20643 .exactZero (none)

def event20649 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15067⟩⟩) 0 ⟨14970⟩ 20606

def event20650 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15067⟩⟩) (.authority (.programFamilyFact))

def exact20651RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15067⟩⟩], []⟩, (1)⟩]

theorem exact20651RawTermsValid :
    exact20651RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20651 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15067⟩⟩) exact20651RawTerms (.finite 3) 20650 .exactZero (none)

def event20652 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15070⟩⟩) 0 ⟨6544⟩ 20628

def event20653 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15070⟩⟩) 1 ⟨15067⟩ 20651

def event20654 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15070⟩⟩) (.product (.predecessor 0 20652 .coefficient) (.predecessor 1 20653 .coefficient) (⟨false, true, none, none, some 1⟩))

def event20655 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15070⟩⟩, .operator (⟨20628, 0⟩, ⟨20651, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact20656RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact20656RawTermsValid :
    exact20656RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20656 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15070⟩⟩) exact20656RawTerms .large 20654 .exactZero (none)

def event20657 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6710⟩⟩) 0 ⟨6689⟩ 20610

def event20658 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6710⟩⟩) (.authority (.operator))

def exact20659RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩]

theorem exact20659RawTermsValid :
    exact20659RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20659 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6710⟩⟩) exact20659RawTerms .large 20658 .exactZero (none)

def event20660 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15071⟩⟩) 0 ⟨6710⟩ 20659

def event20661 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15071⟩⟩) 1 ⟨15070⟩ 20656

def event20662 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15071⟩⟩) (.sum [.predecessor 0 20660 .coefficient, .predecessor 1 20661 .coefficient])

def exact20663RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20663RawTermsValid :
    exact20663RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20663 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15071⟩⟩) exact20663RawTerms .large 20662 .exactZero (none)

def event20664 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26615⟩⟩) 0 ⟨15071⟩ 20663

def event20665 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26615⟩⟩) 1 ⟨26610⟩ 20648

def event20666 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26615⟩⟩) (.sum [.predecessor 0 20664 .coefficient, .predecessor 1 20665 .coefficient])

def exact20667RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26609⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨23795⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20667RawTermsValid :
    exact20667RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20667 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26615⟩⟩) exact20667RawTerms .large 20666 .exactZero (none)

def event20668 : Event := .preFoldPolynomial 20667 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26609⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨23795⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact20669RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26609⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨23795⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event20669 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26615⟩⟩) 20668 exact20669RawTerms .large 20666 .exactZero (none)

def event20670 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14970⟩⟩) ⟨⟨123⟩, ⟨29⟩, ⟨109⟩⟩ ⟨20512, 20670⟩

def event20671 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20483⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20480⟩⟩]⟩) (1) 0 2 (.universal 20670 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20480⟩⟩]⟩) (none) 20669)

def event20672 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20483⟩⟩, .relation 20671 1, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩)

def event20673 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20483⟩⟩, .relation 20671 2, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨23795⟩⟩]⟩, (1)⟩)

def event20674 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20483⟩⟩, .relation 20671 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26609⟩⟩]⟩, (-1)⟩)

def event20675 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20483⟩⟩, .relation 20671 3, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact20676RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26609⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨23795⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20676RawTermsValid :
    exact20676RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20676 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20483⟩⟩) exact20676RawTerms .large 20508 (.finite 1811303510016) (some (20510))

def event20677 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26612⟩⟩) 0 ⟨20483⟩ 20676

def event20678 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26612⟩⟩) 1 ⟨26611⟩ 20498

def event20679 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26612⟩⟩) (.sum [.predecessor 0 20677 .coefficient, .predecessor 1 20678 .coefficient])

def event20680 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26612⟩⟩, .operator (⟨20676, 2⟩, ⟨20498, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨23795⟩⟩]⟩, (-1)⟩)

def event20681 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26612⟩⟩, .operator (⟨20676, 0⟩, ⟨20498, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26609⟩⟩]⟩, (1)⟩)

def event20682 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26612⟩⟩) (.sum [.result 20676 .summary, .result 20498 .summary])

def exact20683RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20683RawTermsValid :
    exact20683RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20683 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26612⟩⟩) exact20683RawTerms .large 20679 (.finite 1291900380601931935744) (some (20682))

def event20684 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26613⟩⟩) 0 ⟨26612⟩ 20683

def event20685 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26613⟩⟩) 1 ⟨6672⟩ 5839

def event20686 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26613⟩⟩) (.product (.predecessor 0 20684 .coefficient) (.predecessor 1 20685 .coefficient) (⟨false, false, none, none, none⟩))

def event20687 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26613⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩) [⟨.result 5835 .coefficient, false, none⟩])

def event20688 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26613⟩⟩) (.product (.result 20683 .summary) (.transfer 20687) (⟨false, false, none, none, none⟩))

def event20689 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26613⟩⟩, .operator (⟨20683, 0⟩, ⟨5839, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩)

def event20690 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26613⟩⟩, .operator (⟨20683, 1⟩, ⟨5839, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (-1)⟩)

def event20691 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26613⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6671⟩⟩) ⟨6607⟩ 5832)

def event20692 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26613⟩⟩, .relation 20691 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact20693RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20693RawTermsValid :
    exact20693RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20693 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26613⟩⟩) exact20693RawTerms .large 20686 (.finite 4741295067215179835091451904) (some (20688))

def event20694 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23732⟩⟩) 0 ⟨6689⟩ 5477

def event20695 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23732⟩⟩) 1 ⟨23731⟩ 14961

def event20696 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23732⟩⟩) (.authority (.operator))

def exact20697RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23732⟩⟩]⟩, (1)⟩]

theorem exact20697RawTermsValid :
    exact20697RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20697 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23732⟩⟩) exact20697RawTerms .large 20696 .exactZero (none)

def event20698 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26399⟩⟩) 0 ⟨23732⟩ 20697

def event20699 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26399⟩⟩) (.authority (.operator))

def exact20700RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26399⟩⟩]⟩, (1)⟩]

theorem exact20700RawTermsValid :
    exact20700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20700 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26399⟩⟩) exact20700RawTerms (.finite 8192) 20699 .exactZero (none)

def event20701 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26401⟩⟩) 0 ⟨24933⟩ 15264

def event20702 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26401⟩⟩) 1 ⟨26399⟩ 20700

def event20703 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26401⟩⟩) (.product (.predecessor 0 20701 .coefficient) (.predecessor 1 20702 .coefficient) (⟨false, false, none, none, none⟩))

def event20704 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26401⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26399⟩⟩]⟩) [⟨.result 20700 .coefficient, false, none⟩])

def event20705 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26401⟩⟩) (.product (.result 15264 .summary) (.transfer 20704) (⟨false, false, none, none, none⟩))

def event20706 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26401⟩⟩, .operator (⟨15264, 1⟩, ⟨20700, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26399⟩⟩]⟩, (-1)⟩)

def event20707 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26401⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26399⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26399⟩⟩) ⟨23732⟩ 20697)

def event20708 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26401⟩⟩, .relation 20707 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨23732⟩⟩]⟩, (-1)⟩)

def event20709 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26401⟩⟩, .operator (⟨15264, 0⟩, ⟨20700, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26399⟩⟩]⟩, (1)⟩)

def exact20710RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26399⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨23732⟩⟩]⟩, (-1)⟩]

theorem exact20710RawTermsValid :
    exact20710RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20710 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26401⟩⟩) exact20710RawTerms .large 20703 (.finite 1291889172568118132736) (some (20705))

def event20711 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20336⟩⟩) 0 ⟨14809⟩ 459

def event20712 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20336⟩⟩) (.authority (.relationPreimageSource ⟨27⟩))

def exact20713RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20336⟩⟩]⟩, (1)⟩]

theorem exact20713RawTermsValid :
    exact20713RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20713 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20336⟩⟩) exact20713RawTerms (.finite 136065468) 20712 .exactZero (none)

def event20714 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20338⟩⟩) 0 ⟨20336⟩ 20713

def event20715 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20338⟩⟩) 1 ⟨2348⟩ 4

def event20716 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20338⟩⟩) (.scale (.predecessor 0 20714 .coefficient) (.value (.predecessor 1 20715 .coefficient)))

def exact20717RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20336⟩⟩]⟩, (1)⟩]

theorem exact20717RawTermsValid :
    exact20717RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20717 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20338⟩⟩) exact20717RawTerms (.finite 136065468) 20716 .exactZero (none)

def event20718 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20339⟩⟩) 0 ⟨5565⟩ 6561

def event20719 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20339⟩⟩) 1 ⟨20338⟩ 20717

def event20720 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20339⟩⟩) (.product (.predecessor 0 20718 .coefficient) (.predecessor 1 20719 .coefficient) (⟨false, false, none, none, none⟩))

def event20721 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20339⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20336⟩⟩]⟩) [⟨.result 20713 .coefficient, false, none⟩])

def event20722 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20339⟩⟩) (.product (.result 6561 .summary) (.transfer 20721) (⟨false, false, none, none, none⟩))

def event20723 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20339⟩⟩, .operator (⟨6561, 0⟩, ⟨20717, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20336⟩⟩]⟩, (1)⟩)

def event20724 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20337⟩⟩)

def event20725 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event20726 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event20727 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.authority (.operator))

def event20728 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.finite 6)

def event20729 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event20730 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event20731 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event20732 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event20733 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 20732

def event20734 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 20730

def event20735 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 20733 .coefficient) (.value (.predecessor 1 20734 .coefficient)))

def eventLeaf1280 : Array AnnotatedEvent := #[
  { event := event20480
    frameStart := 0 },
  { event := event20481
    frameStart := 0 },
  { event := event20482
    frameStart := 0 },
  { event := event20483
    frameStart := 0 },
  { event := event20484
    frameStart := 0 },
  { event := event20485
    frameStart := 0 },
  { event := event20486
    frameStart := 0 },
  { event := event20487
    frameStart := 0 },
  { event := event20488
    frameStart := 0 },
  { event := event20489
    frameStart := 0 },
  { event := event20490
    frameStart := 0 },
  { event := event20491
    frameStart := 0 },
  { event := event20492
    frameStart := 0 },
  { event := event20493
    frameStart := 0 },
  { event := event20494
    frameStart := 0 },
  { event := event20495
    frameStart := 0 }
]

def eventLeaf1281 : Array AnnotatedEvent := #[
  { event := event20496
    frameStart := 0 },
  { event := event20497
    frameStart := 0 },
  { event := event20498
    frameStart := 0 },
  { event := event20499
    frameStart := 0 },
  { event := event20500
    frameStart := 0 },
  { event := event20501
    frameStart := 0 },
  { event := event20502
    frameStart := 0 },
  { event := event20503
    frameStart := 0 },
  { event := event20504
    frameStart := 0 },
  { event := event20505
    frameStart := 0 },
  { event := event20506
    frameStart := 0 },
  { event := event20507
    frameStart := 0 },
  { event := event20508
    frameStart := 0 },
  { event := event20509
    frameStart := 0 },
  { event := event20510
    frameStart := 0 },
  { event := event20511
    frameStart := 0 }
]

def eventLeaf1282 : Array AnnotatedEvent := #[
  { event := event20512
    frameStart := 20512 },
  { event := event20513
    frameStart := 20512 },
  { event := event20514
    frameStart := 20512 },
  { event := event20515
    frameStart := 20512 },
  { event := event20516
    frameStart := 20512 },
  { event := event20517
    frameStart := 20512 },
  { event := event20518
    frameStart := 20512 },
  { event := event20519
    frameStart := 20512 },
  { event := event20520
    frameStart := 20512 },
  { event := event20521
    frameStart := 20512 },
  { event := event20522
    frameStart := 20512 },
  { event := event20523
    frameStart := 20512 },
  { event := event20524
    frameStart := 20512 },
  { event := event20525
    frameStart := 20512 },
  { event := event20526
    frameStart := 20512 },
  { event := event20527
    frameStart := 20512 }
]

def eventLeaf1283 : Array AnnotatedEvent := #[
  { event := event20528
    frameStart := 20512 },
  { event := event20529
    frameStart := 20512 },
  { event := event20530
    frameStart := 20512 },
  { event := event20531
    frameStart := 20512 },
  { event := event20532
    frameStart := 20512 },
  { event := event20533
    frameStart := 20512 },
  { event := event20534
    frameStart := 20512 },
  { event := event20535
    frameStart := 20512 },
  { event := event20536
    frameStart := 20512 },
  { event := event20537
    frameStart := 20512 },
  { event := event20538
    frameStart := 20512 },
  { event := event20539
    frameStart := 20512 },
  { event := event20540
    frameStart := 20512 },
  { event := event20541
    frameStart := 20512 },
  { event := event20542
    frameStart := 20512 },
  { event := event20543
    frameStart := 20512 }
]

def eventLeaf1284 : Array AnnotatedEvent := #[
  { event := event20544
    frameStart := 20512 },
  { event := event20545
    frameStart := 20512 },
  { event := event20546
    frameStart := 20512 },
  { event := event20547
    frameStart := 20512 },
  { event := event20548
    frameStart := 20512 },
  { event := event20549
    frameStart := 20512 },
  { event := event20550
    frameStart := 20512 },
  { event := event20551
    frameStart := 20512 },
  { event := event20552
    frameStart := 20512 },
  { event := event20553
    frameStart := 20512 },
  { event := event20554
    frameStart := 20512 },
  { event := event20555
    frameStart := 20512 },
  { event := event20556
    frameStart := 20512 },
  { event := event20557
    frameStart := 20512 },
  { event := event20558
    frameStart := 20512 },
  { event := event20559
    frameStart := 20512 }
]

def eventLeaf1285 : Array AnnotatedEvent := #[
  { event := event20560
    frameStart := 20512 },
  { event := event20561
    frameStart := 20512 },
  { event := event20562
    frameStart := 20512 },
  { event := event20563
    frameStart := 20512 },
  { event := event20564
    frameStart := 20512 },
  { event := event20565
    frameStart := 20512 },
  { event := event20566
    frameStart := 20566 },
  { event := event20567
    frameStart := 20566 },
  { event := event20568
    frameStart := 20566 },
  { event := event20569
    frameStart := 20566 },
  { event := event20570
    frameStart := 20566 },
  { event := event20571
    frameStart := 20566 },
  { event := event20572
    frameStart := 20566 },
  { event := event20573
    frameStart := 20566 },
  { event := event20574
    frameStart := 20566 },
  { event := event20575
    frameStart := 20566 }
]

def eventLeaf1286 : Array AnnotatedEvent := #[
  { event := event20576
    frameStart := 20566 },
  { event := event20577
    frameStart := 20566 },
  { event := event20578
    frameStart := 20566 },
  { event := event20579
    frameStart := 20566 },
  { event := event20580
    frameStart := 20566 },
  { event := event20581
    frameStart := 20566 },
  { event := event20582
    frameStart := 20566 },
  { event := event20583
    frameStart := 20566 },
  { event := event20584
    frameStart := 20566 },
  { event := event20585
    frameStart := 20566 },
  { event := event20586
    frameStart := 20566 },
  { event := event20587
    frameStart := 20566 },
  { event := event20588
    frameStart := 20566 },
  { event := event20589
    frameStart := 20566 },
  { event := event20590
    frameStart := 20566 },
  { event := event20591
    frameStart := 20566 }
]

def eventLeaf1287 : Array AnnotatedEvent := #[
  { event := event20592
    frameStart := 20566 },
  { event := event20593
    frameStart := 20566 },
  { event := event20594
    frameStart := 20566 },
  { event := event20595
    frameStart := 20566 },
  { event := event20596
    frameStart := 20566 },
  { event := event20597
    frameStart := 20566 },
  { event := event20598
    frameStart := 20566 },
  { event := event20599
    frameStart := 20566 },
  { event := event20600
    frameStart := 20566 },
  { event := event20601
    frameStart := 20566 },
  { event := event20602
    frameStart := 20566 },
  { event := event20603
    frameStart := 20566 },
  { event := event20604
    frameStart := 20566 },
  { event := event20605
    frameStart := 20566 },
  { event := event20606
    frameStart := 20566 },
  { event := event20607
    frameStart := 20566 }
]

def eventLeaf1288 : Array AnnotatedEvent := #[
  { event := event20608
    frameStart := 20566 },
  { event := event20609
    frameStart := 20566 },
  { event := event20610
    frameStart := 20566 },
  { event := event20611
    frameStart := 20566 },
  { event := event20612
    frameStart := 20566 },
  { event := event20613
    frameStart := 20566 },
  { event := event20614
    frameStart := 20566 },
  { event := event20615
    frameStart := 20566 },
  { event := event20616
    frameStart := 20566 },
  { event := event20617
    frameStart := 20566 },
  { event := event20618
    frameStart := 20566 },
  { event := event20619
    frameStart := 20566 },
  { event := event20620
    frameStart := 20566 },
  { event := event20621
    frameStart := 20566 },
  { event := event20622
    frameStart := 20566 },
  { event := event20623
    frameStart := 20566 }
]

def eventLeaf1289 : Array AnnotatedEvent := #[
  { event := event20624
    frameStart := 20566 },
  { event := event20625
    frameStart := 20566 },
  { event := event20626
    frameStart := 20566 },
  { event := event20627
    frameStart := 20566 },
  { event := event20628
    frameStart := 20566 },
  { event := event20629
    frameStart := 20566 },
  { event := event20630
    frameStart := 20566 },
  { event := event20631
    frameStart := 20566 },
  { event := event20632
    frameStart := 20566 },
  { event := event20633
    frameStart := 20566 },
  { event := event20634
    frameStart := 20566 },
  { event := event20635
    frameStart := 20566 },
  { event := event20636
    frameStart := 20566 },
  { event := event20637
    frameStart := 20566 },
  { event := event20638
    frameStart := 20566 },
  { event := event20639
    frameStart := 20566 }
]

def eventLeaf1290 : Array AnnotatedEvent := #[
  { event := event20640
    frameStart := 20566 },
  { event := event20641
    frameStart := 20566 },
  { event := event20642
    frameStart := 20566 },
  { event := event20643
    frameStart := 20566 },
  { event := event20644
    frameStart := 20566 },
  { event := event20645
    frameStart := 20566 },
  { event := event20646
    frameStart := 20566 },
  { event := event20647
    frameStart := 20566 },
  { event := event20648
    frameStart := 20566 },
  { event := event20649
    frameStart := 20566 },
  { event := event20650
    frameStart := 20566 },
  { event := event20651
    frameStart := 20566 },
  { event := event20652
    frameStart := 20566 },
  { event := event20653
    frameStart := 20566 },
  { event := event20654
    frameStart := 20566 },
  { event := event20655
    frameStart := 20566 }
]

def eventLeaf1291 : Array AnnotatedEvent := #[
  { event := event20656
    frameStart := 20566 },
  { event := event20657
    frameStart := 20566 },
  { event := event20658
    frameStart := 20566 },
  { event := event20659
    frameStart := 20566 },
  { event := event20660
    frameStart := 20566 },
  { event := event20661
    frameStart := 20566 },
  { event := event20662
    frameStart := 20566 },
  { event := event20663
    frameStart := 20566 },
  { event := event20664
    frameStart := 20566 },
  { event := event20665
    frameStart := 20566 },
  { event := event20666
    frameStart := 20566 },
  { event := event20667
    frameStart := 20566 },
  { event := event20668
    frameStart := 20566 },
  { event := event20669
    frameStart := 20566 },
  { event := event20670
    frameStart := 0 },
  { event := event20671
    frameStart := 0 }
]

def eventLeaf1292 : Array AnnotatedEvent := #[
  { event := event20672
    frameStart := 0 },
  { event := event20673
    frameStart := 0 },
  { event := event20674
    frameStart := 0 },
  { event := event20675
    frameStart := 0 },
  { event := event20676
    frameStart := 0 },
  { event := event20677
    frameStart := 0 },
  { event := event20678
    frameStart := 0 },
  { event := event20679
    frameStart := 0 },
  { event := event20680
    frameStart := 0 },
  { event := event20681
    frameStart := 0 },
  { event := event20682
    frameStart := 0 },
  { event := event20683
    frameStart := 0 },
  { event := event20684
    frameStart := 0 },
  { event := event20685
    frameStart := 0 },
  { event := event20686
    frameStart := 0 },
  { event := event20687
    frameStart := 0 }
]

def eventLeaf1293 : Array AnnotatedEvent := #[
  { event := event20688
    frameStart := 0 },
  { event := event20689
    frameStart := 0 },
  { event := event20690
    frameStart := 0 },
  { event := event20691
    frameStart := 0 },
  { event := event20692
    frameStart := 0 },
  { event := event20693
    frameStart := 0 },
  { event := event20694
    frameStart := 0 },
  { event := event20695
    frameStart := 0 },
  { event := event20696
    frameStart := 0 },
  { event := event20697
    frameStart := 0 },
  { event := event20698
    frameStart := 0 },
  { event := event20699
    frameStart := 0 },
  { event := event20700
    frameStart := 0 },
  { event := event20701
    frameStart := 0 },
  { event := event20702
    frameStart := 0 },
  { event := event20703
    frameStart := 0 }
]

def eventLeaf1294 : Array AnnotatedEvent := #[
  { event := event20704
    frameStart := 0 },
  { event := event20705
    frameStart := 0 },
  { event := event20706
    frameStart := 0 },
  { event := event20707
    frameStart := 0 },
  { event := event20708
    frameStart := 0 },
  { event := event20709
    frameStart := 0 },
  { event := event20710
    frameStart := 0 },
  { event := event20711
    frameStart := 0 },
  { event := event20712
    frameStart := 0 },
  { event := event20713
    frameStart := 0 },
  { event := event20714
    frameStart := 0 },
  { event := event20715
    frameStart := 0 },
  { event := event20716
    frameStart := 0 },
  { event := event20717
    frameStart := 0 },
  { event := event20718
    frameStart := 0 },
  { event := event20719
    frameStart := 0 }
]

def eventLeaf1295 : Array AnnotatedEvent := #[
  { event := event20720
    frameStart := 0 },
  { event := event20721
    frameStart := 0 },
  { event := event20722
    frameStart := 0 },
  { event := event20723
    frameStart := 0 },
  { event := event20724
    frameStart := 20724 },
  { event := event20725
    frameStart := 20724 },
  { event := event20726
    frameStart := 20724 },
  { event := event20727
    frameStart := 20724 },
  { event := event20728
    frameStart := 20724 },
  { event := event20729
    frameStart := 20724 },
  { event := event20730
    frameStart := 20724 },
  { event := event20731
    frameStart := 20724 },
  { event := event20732
    frameStart := 20724 },
  { event := event20733
    frameStart := 20724 },
  { event := event20734
    frameStart := 20724 },
  { event := event20735
    frameStart := 20724 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events080
