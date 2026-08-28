import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events666

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event170496 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32432⟩⟩) (.product (.predecessor 0 170494 .coefficient) (.predecessor 1 170495 .coefficient) (⟨false, false, none, none, none⟩))

def event170497 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32432⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨32429⟩⟩]⟩) [⟨.result 170489 .coefficient, false, none⟩])

def event170498 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32432⟩⟩) (.product (.result 163745 .summary) (.transfer 170497) (⟨false, false, none, none, none⟩))

def event170499 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32432⟩⟩, .operator (⟨163745, 0⟩, ⟨170493, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32429⟩⟩]⟩, (1)⟩)

def event170500 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨32430⟩⟩)

def event170501 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event170502 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event170503 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6449⟩⟩) (.authority (.operator))

def event170504 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6449⟩⟩) (.finite 9)

def event170505 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event170506 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event170507 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event170508 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event170509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 170508

def event170510 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 170506

def event170511 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 170509 .coefficient) (.value (.predecessor 1 170510 .coefficient)))

def event170512 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event170513 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6451⟩⟩) 0 ⟨392⟩ 170512

def event170514 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6451⟩⟩) 1 ⟨6449⟩ 170504

def event170515 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6451⟩⟩) (.sum [.predecessor 0 170513 .coefficient, .predecessor 1 170514 .coefficient])

def event170516 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6451⟩⟩) (.finite 655349)

def event170517 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6462⟩⟩) 0 ⟨6451⟩ 170516

def event170518 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6462⟩⟩) 1 ⟨5426⟩ 170502

def event170519 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6462⟩⟩) (.identity (.predecessor 1 170518 .coefficient))

def event170520 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6462⟩⟩) (.finite 655360)

def event170521 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24338⟩⟩) 0 ⟨6462⟩ 170520

def event170522 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24338⟩⟩) (.authority (.programFamilyFact))

def exact170523RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24338⟩⟩], []⟩, (1)⟩]

theorem exact170523RawTermsValid :
    exact170523RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170523 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24338⟩⟩) exact170523RawTerms (.finite 6) 170522 .exactZero (none)

def event170524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31593⟩⟩) 0 ⟨6462⟩ 170520

def event170525 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31593⟩⟩) (.authority (.programFamilyFact))

def exact170526RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31593⟩⟩], []⟩, (1)⟩]

theorem exact170526RawTermsValid :
    exact170526RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170526 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31593⟩⟩) exact170526RawTerms (.finite 6) 170525 .exactZero (none)

def event170527 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31594⟩⟩) 0 ⟨31593⟩ 170526

def event170528 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31594⟩⟩) 1 ⟨24338⟩ 170523

def event170529 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31594⟩⟩) (.product (.predecessor 0 170527 .coefficient) (.predecessor 1 170528 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event170530 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31594⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24338⟩⟩, ⟨.program ⟨257⟩, ⟨31593⟩⟩], []⟩) [⟨.result 170526 .coefficient, true, some 1⟩, ⟨.result 170523 .coefficient, true, some 1⟩])

def event170531 : Event := .survivorFold (1) 170530

def exact170532RawTerms : List Term := []

theorem exact170532RawTermsValid :
    exact170532RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170532 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31594⟩⟩) exact170532RawTerms (.finite 36) 170529 (.finite 36) (some (170530))

def event170533 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31595⟩⟩) 0 ⟨31594⟩ 170532

def event170534 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31595⟩⟩) (.identity (.predecessor 0 170533 .coefficient))

def event170535 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31595⟩⟩) (.finite 36)

def event170536 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32429⟩⟩) 0 ⟨31595⟩ 170535

def event170537 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32429⟩⟩) (.authority (.relationPreimageSource ⟨39⟩))

def exact170538RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32429⟩⟩]⟩, (1)⟩]

theorem exact170538RawTermsValid :
    exact170538RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170538 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32429⟩⟩) exact170538RawTerms (.finite 5647228698) 170537 .exactZero (none)

def event170539 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact170540RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact170540RawTermsValid :
    exact170540RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170540 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact170540RawTerms .large 170539 .exactZero (none)

def event170541 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32430⟩⟩) 0 ⟨35⟩ 170540

def event170542 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32430⟩⟩) 1 ⟨32429⟩ 170538

def event170543 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32430⟩⟩) (.product (.predecessor 0 170541 .coefficient) (.predecessor 1 170542 .coefficient) (⟨false, false, none, none, none⟩))

def event170544 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32430⟩⟩, .operator (⟨170540, 0⟩, ⟨170538, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32429⟩⟩]⟩, (1)⟩)

def exact170545RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32429⟩⟩]⟩, (1)⟩]

theorem exact170545RawTermsValid :
    exact170545RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170545 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32430⟩⟩) exact170545RawTerms .large 170543 .exactZero (none)

def event170546 : Event := .preFoldPolynomial 170545 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32429⟩⟩]⟩, (1)⟩] .exactZero none

def exact170547RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32429⟩⟩]⟩, (1)⟩]

def event170547 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨32430⟩⟩) 170546 exact170547RawTerms .large 170543 .exactZero (none)

def event170548 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨33507⟩⟩)

def event170549 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event170550 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event170551 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6449⟩⟩) (.authority (.operator))

def event170552 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6449⟩⟩) (.finite 9)

def event170553 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event170554 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event170555 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event170556 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event170557 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 170556

def event170558 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 170554

def event170559 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 170557 .coefficient) (.value (.predecessor 1 170558 .coefficient)))

def event170560 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event170561 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6451⟩⟩) 0 ⟨392⟩ 170560

def event170562 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6451⟩⟩) 1 ⟨6449⟩ 170552

def event170563 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6451⟩⟩) (.sum [.predecessor 0 170561 .coefficient, .predecessor 1 170562 .coefficient])

def event170564 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6451⟩⟩) (.finite 655349)

def event170565 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6462⟩⟩) 0 ⟨6451⟩ 170564

def event170566 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6462⟩⟩) 1 ⟨5426⟩ 170550

def event170567 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6462⟩⟩) (.identity (.predecessor 1 170566 .coefficient))

def event170568 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6462⟩⟩) (.finite 655360)

def event170569 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24338⟩⟩) 0 ⟨6462⟩ 170568

def event170570 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24338⟩⟩) (.authority (.programFamilyFact))

def exact170571RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24338⟩⟩], []⟩, (1)⟩]

theorem exact170571RawTermsValid :
    exact170571RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170571 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24338⟩⟩) exact170571RawTerms (.finite 6) 170570 .exactZero (none)

def event170572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31593⟩⟩) 0 ⟨6462⟩ 170568

def event170573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31593⟩⟩) (.authority (.programFamilyFact))

def exact170574RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31593⟩⟩], []⟩, (1)⟩]

theorem exact170574RawTermsValid :
    exact170574RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170574 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31593⟩⟩) exact170574RawTerms (.finite 6) 170573 .exactZero (none)

def event170575 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31594⟩⟩) 0 ⟨31593⟩ 170574

def event170576 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31594⟩⟩) 1 ⟨24338⟩ 170571

def event170577 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31594⟩⟩) (.product (.predecessor 0 170575 .coefficient) (.predecessor 1 170576 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event170578 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31594⟩⟩, .operator (⟨170574, 0⟩, ⟨170571, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24338⟩⟩, ⟨.program ⟨257⟩, ⟨31593⟩⟩], []⟩, (1)⟩)

def exact170579RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24338⟩⟩, ⟨.program ⟨257⟩, ⟨31593⟩⟩], []⟩, (1)⟩]

theorem exact170579RawTermsValid :
    exact170579RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170579 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31594⟩⟩) exact170579RawTerms (.finite 36) 170577 .exactZero (none)

def event170580 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31595⟩⟩) 0 ⟨31594⟩ 170579

def event170581 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31595⟩⟩) (.identity (.predecessor 0 170580 .coefficient))

def event170582 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31595⟩⟩) (.finite 36)

def event170583 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32972⟩⟩) 0 ⟨31595⟩ 170582

def event170584 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32972⟩⟩) (.authority (.programFamilyFact))

def event170585 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨32972⟩⟩) (.finite 3720)

def event170586 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event170587 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32973⟩⟩) 0 ⟨7177⟩ 170586

def event170588 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32973⟩⟩) 1 ⟨32972⟩ 170585

def event170589 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32973⟩⟩) (.authority (.operator))

def exact170590RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32973⟩⟩]⟩, (1)⟩]

theorem exact170590RawTermsValid :
    exact170590RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170590 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32973⟩⟩) exact170590RawTerms .large 170589 .exactZero (none)

def event170591 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33503⟩⟩) 0 ⟨32973⟩ 170590

def event170592 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33503⟩⟩) (.authority (.operator))

def exact170593RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨33503⟩⟩]⟩, (1)⟩]

theorem exact170593RawTermsValid :
    exact170593RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170593 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33503⟩⟩) exact170593RawTerms (.finite 8192) 170592 .exactZero (none)

def event170594 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event170595 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event170596 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33242⟩⟩) 0 ⟨31595⟩ 170582

def event170597 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33242⟩⟩) 1 ⟨136⟩ 170595

def event170598 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33242⟩⟩) (.sum [.predecessor 0 170596 .coefficient, .predecessor 1 170597 .coefficient])

def event170599 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨33242⟩⟩) (.finite 36)

def event170600 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33243⟩⟩) 0 ⟨33242⟩ 170599

def event170601 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33243⟩⟩) (.identity (.predecessor 0 170600 .coefficient))

def exact170602RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24338⟩⟩, ⟨.program ⟨257⟩, ⟨31593⟩⟩], []⟩, (1)⟩]

theorem exact170602RawTermsValid :
    exact170602RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170602 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33243⟩⟩) exact170602RawTerms (.finite 36) 170601 .exactZero (none)

def event170603 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact170604RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact170604RawTermsValid :
    exact170604RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170604 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact170604RawTerms .large 170603 .exactZero (none)

def event170605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33244⟩⟩) 0 ⟨6908⟩ 170604

def event170606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33244⟩⟩) 1 ⟨33243⟩ 170602

def event170607 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33244⟩⟩) (.product (.predecessor 0 170605 .coefficient) (.predecessor 1 170606 .coefficient) (⟨false, false, none, none, none⟩))

def event170608 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33244⟩⟩, .operator (⟨170604, 0⟩, ⟨170602, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24338⟩⟩, ⟨.program ⟨257⟩, ⟨31593⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact170609RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24338⟩⟩, ⟨.program ⟨257⟩, ⟨31593⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact170609RawTermsValid :
    exact170609RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170609 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33244⟩⟩) exact170609RawTerms .large 170607 .exactZero (none)

def event170610 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event170611 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event170612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 170586

def event170613 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact170614RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact170614RawTermsValid :
    exact170614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170614 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact170614RawTerms .large 170613 .exactZero (none)

def event170615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7307⟩⟩) 0 ⟨7178⟩ 170614

def event170616 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7307⟩⟩) (.identity (.predecessor 0 170615 .coefficient))

def exact170617RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩]

theorem exact170617RawTermsValid :
    exact170617RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170617 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7307⟩⟩) exact170617RawTerms .large 170616 .exactZero (none)

def event170618 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9577⟩⟩) 0 ⟨7307⟩ 170617

def event170619 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9577⟩⟩) (.authority (.operator))

def exact170620RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩]

theorem exact170620RawTermsValid :
    exact170620RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170620 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9577⟩⟩) exact170620RawTerms (.finite 8192) 170619 .exactZero (none)

def event170621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9578⟩⟩) 0 ⟨9577⟩ 170620

def event170622 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9578⟩⟩) 1 ⟨2370⟩ 170611

def event170623 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9578⟩⟩) (.scale (.predecessor 0 170621 .coefficient) (.value (.predecessor 1 170622 .coefficient)))

def exact170624RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩]

theorem exact170624RawTermsValid :
    exact170624RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170624 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9578⟩⟩) exact170624RawTerms (.finite 8192) 170623 .exactZero (none)

def event170625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7287⟩⟩) 0 ⟨7178⟩ 170614

def event170626 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7287⟩⟩) (.identity (.predecessor 0 170625 .coefficient))

def exact170627RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩]⟩, (1)⟩]

theorem exact170627RawTermsValid :
    exact170627RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170627 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7287⟩⟩) exact170627RawTerms .large 170626 .exactZero (none)

def event170628 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9579⟩⟩) 0 ⟨7287⟩ 170627

def event170629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9579⟩⟩) 1 ⟨9578⟩ 170624

def event170630 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9579⟩⟩) (.product (.predecessor 0 170628 .coefficient) (.predecessor 1 170629 .coefficient) (⟨false, false, none, none, none⟩))

def event170631 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9579⟩⟩, .operator (⟨170627, 0⟩, ⟨170624, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩)

def exact170632RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩]

theorem exact170632RawTermsValid :
    exact170632RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170632 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9579⟩⟩) exact170632RawTerms .large 170630 .exactZero (none)

def event170633 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33245⟩⟩) 0 ⟨9579⟩ 170632

def event170634 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33245⟩⟩) 1 ⟨33244⟩ 170609

def event170635 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33245⟩⟩) (.sum [.predecessor 0 170633 .coefficient, .predecessor 1 170634 .coefficient])

def exact170636RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24338⟩⟩, ⟨.program ⟨257⟩, ⟨31593⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact170636RawTermsValid :
    exact170636RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170636 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33245⟩⟩) exact170636RawTerms .large 170635 .exactZero (none)

def event170637 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33506⟩⟩) 0 ⟨33245⟩ 170636

def event170638 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33506⟩⟩) 1 ⟨33503⟩ 170593

def event170639 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33506⟩⟩) (.product (.predecessor 0 170637 .coefficient) (.predecessor 1 170638 .coefficient) (⟨false, false, none, none, none⟩))

def event170640 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33506⟩⟩, .operator (⟨170636, 0⟩, ⟨170593, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33503⟩⟩]⟩, (1)⟩)

def event170641 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33506⟩⟩, .operator (⟨170636, 1⟩, ⟨170593, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24338⟩⟩, ⟨.program ⟨257⟩, ⟨31593⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33503⟩⟩]⟩, (-1)⟩)

def event170642 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨33506⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨24338⟩⟩, ⟨.program ⟨257⟩, ⟨31593⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33503⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33503⟩⟩) ⟨32973⟩ 170590)

def event170643 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33506⟩⟩, .relation 170642 0, ⟨[⟨.program ⟨257⟩, ⟨24338⟩⟩, ⟨.program ⟨257⟩, ⟨31593⟩⟩], [⟨.program ⟨257⟩, ⟨32973⟩⟩]⟩, (-1)⟩)

def exact170644RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33503⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24338⟩⟩, ⟨.program ⟨257⟩, ⟨31593⟩⟩], [⟨.program ⟨257⟩, ⟨32973⟩⟩]⟩, (-1)⟩]

theorem exact170644RawTermsValid :
    exact170644RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170644 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33506⟩⟩) exact170644RawTerms .large 170639 .exactZero (none)

def event170645 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31860⟩⟩) 0 ⟨31595⟩ 170582

def event170646 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31860⟩⟩) (.authority (.programFamilyFact))

def exact170647RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31860⟩⟩], []⟩, (1)⟩]

theorem exact170647RawTermsValid :
    exact170647RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170647 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31860⟩⟩) exact170647RawTerms (.finite 6) 170646 .exactZero (none)

def event170648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31862⟩⟩) 0 ⟨6908⟩ 170604

def event170649 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31862⟩⟩) 1 ⟨31860⟩ 170647

def event170650 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31862⟩⟩) (.product (.predecessor 0 170648 .coefficient) (.predecessor 1 170649 .coefficient) (⟨false, true, none, none, some 1⟩))

def event170651 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31862⟩⟩, .operator (⟨170604, 0⟩, ⟨170647, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact170652RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact170652RawTermsValid :
    exact170652RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170652 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31862⟩⟩) exact170652RawTerms .large 170650 .exactZero (none)

def event170653 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7182⟩⟩) 0 ⟨7177⟩ 170586

def event170654 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7182⟩⟩) (.authority (.operator))

def exact170655RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩]

theorem exact170655RawTermsValid :
    exact170655RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170655 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7182⟩⟩) exact170655RawTerms .large 170654 .exactZero (none)

def event170656 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31863⟩⟩) 0 ⟨7182⟩ 170655

def event170657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31863⟩⟩) 1 ⟨31862⟩ 170652

def event170658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31863⟩⟩) (.sum [.predecessor 0 170656 .coefficient, .predecessor 1 170657 .coefficient])

def exact170659RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact170659RawTermsValid :
    exact170659RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170659 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31863⟩⟩) exact170659RawTerms .large 170658 .exactZero (none)

def event170660 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33507⟩⟩) 0 ⟨31863⟩ 170659

def event170661 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33507⟩⟩) 1 ⟨33506⟩ 170644

def event170662 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33507⟩⟩) (.sum [.predecessor 0 170660 .coefficient, .predecessor 1 170661 .coefficient])

def exact170663RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33503⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24338⟩⟩, ⟨.program ⟨257⟩, ⟨31593⟩⟩], [⟨.program ⟨257⟩, ⟨32973⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact170663RawTermsValid :
    exact170663RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170663 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33507⟩⟩) exact170663RawTerms .large 170662 .exactZero (none)

def event170664 : Event := .preFoldPolynomial 170663 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33503⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24338⟩⟩, ⟨.program ⟨257⟩, ⟨31593⟩⟩], [⟨.program ⟨257⟩, ⟨32973⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact170665RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33503⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24338⟩⟩, ⟨.program ⟨257⟩, ⟨31593⟩⟩], [⟨.program ⟨257⟩, ⟨32973⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event170665 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨33507⟩⟩) 170664 exact170665RawTerms .large 170662 .exactZero (none)

def event170666 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨31595⟩⟩) ⟨⟨61⟩, ⟨39⟩, ⟨135⟩⟩ ⟨170500, 170666⟩

def event170667 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨32432⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32429⟩⟩]⟩) (1) 0 2 (.universal 170666 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32429⟩⟩]⟩) (none) 170665)

def event170668 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32432⟩⟩, .relation 170667 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩)

def event170669 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32432⟩⟩, .relation 170667 1, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33503⟩⟩]⟩, (-1)⟩)

def event170670 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32432⟩⟩, .relation 170667 2, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨24338⟩⟩, ⟨.program ⟨257⟩, ⟨31593⟩⟩], [⟨.program ⟨257⟩, ⟨32973⟩⟩]⟩, (1)⟩)

def event170671 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32432⟩⟩, .relation 170667 3, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact170672RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33503⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨24338⟩⟩, ⟨.program ⟨257⟩, ⟨31593⟩⟩], [⟨.program ⟨257⟩, ⟨32973⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact170672RawTermsValid :
    exact170672RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170672 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32432⟩⟩) exact170672RawTerms .large 170496 (.finite 202072841853861888) (some (170498))

def event170673 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33505⟩⟩) 0 ⟨32432⟩ 170672

def event170674 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33505⟩⟩) 1 ⟨33504⟩ 170486

def event170675 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33505⟩⟩) (.sum [.predecessor 0 170673 .coefficient, .predecessor 1 170674 .coefficient])

def event170676 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33505⟩⟩, .operator (⟨170672, 2⟩, ⟨170486, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨24338⟩⟩, ⟨.program ⟨257⟩, ⟨31593⟩⟩], [⟨.program ⟨257⟩, ⟨32973⟩⟩]⟩, (-1)⟩)

def event170677 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33505⟩⟩, .operator (⟨170672, 1⟩, ⟨170486, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33503⟩⟩]⟩, (1)⟩)

def event170678 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33505⟩⟩) (.sum [.result 170672 .summary, .result 170486 .summary])

def exact170679RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact170679RawTermsValid :
    exact170679RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170679 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33505⟩⟩) exact170679RawTerms .large 170675 (.finite 2997852872440114577408) (some (170678))

def event170680 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34018⟩⟩) 0 ⟨33505⟩ 170679

def event170681 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34018⟩⟩) 1 ⟨34016⟩ 170402

def event170682 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34018⟩⟩) (.product (.predecessor 0 170680 .coefficient) (.predecessor 1 170681 .coefficient) (⟨false, false, none, none, none⟩))

def event170683 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34018⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨34016⟩⟩]⟩) [⟨.result 170402 .coefficient, false, none⟩])

def event170684 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34018⟩⟩) (.product (.result 170679 .summary) (.transfer 170683) (⟨false, false, none, none, none⟩))

def event170685 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34018⟩⟩, .operator (⟨170679, 0⟩, ⟨170402, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34016⟩⟩]⟩, (1)⟩)

def event170686 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34018⟩⟩, .operator (⟨170679, 1⟩, ⟨170402, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34016⟩⟩]⟩, (-1)⟩)

def event170687 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨34018⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34016⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨34016⟩⟩) ⟨33137⟩ 170399)

def event170688 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34018⟩⟩, .relation 170687 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨33137⟩⟩]⟩, (-1)⟩)

def exact170689RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34016⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨31860⟩⟩], [⟨.program ⟨257⟩, ⟨33137⟩⟩]⟩, (-1)⟩]

theorem exact170689RawTermsValid :
    exact170689RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170689 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34018⟩⟩) exact170689RawTerms .large 170682 (.finite 32189200113374879571150551121920) (some (170684))

def event170690 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32776⟩⟩) 0 ⟨31861⟩ 7913

def event170691 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32776⟩⟩) (.authority (.relationPreimageSource ⟨63⟩))

def exact170692RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32776⟩⟩]⟩, (1)⟩]

theorem exact170692RawTermsValid :
    exact170692RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170692 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32776⟩⟩) exact170692RawTerms (.finite 5647228698) 170691 .exactZero (none)

def event170693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32778⟩⟩) 0 ⟨32776⟩ 170692

def event170694 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32778⟩⟩) 1 ⟨2370⟩ 4

def event170695 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32778⟩⟩) (.scale (.predecessor 0 170693 .coefficient) (.value (.predecessor 1 170694 .coefficient)))

def exact170696RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32776⟩⟩]⟩, (1)⟩]

theorem exact170696RawTermsValid :
    exact170696RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170696 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32778⟩⟩) exact170696RawTerms (.finite 5647228698) 170695 .exactZero (none)

def event170697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32779⟩⟩) 0 ⟨6466⟩ 163745

def event170698 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32779⟩⟩) 1 ⟨32778⟩ 170696

def event170699 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32779⟩⟩) (.product (.predecessor 0 170697 .coefficient) (.predecessor 1 170698 .coefficient) (⟨false, false, none, none, none⟩))

def event170700 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32779⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨32776⟩⟩]⟩) [⟨.result 170692 .coefficient, false, none⟩])

def event170701 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32779⟩⟩) (.product (.result 163745 .summary) (.transfer 170700) (⟨false, false, none, none, none⟩))

def event170702 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32779⟩⟩, .operator (⟨163745, 0⟩, ⟨170696, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32776⟩⟩]⟩, (1)⟩)

def event170703 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨32777⟩⟩)

def event170704 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event170705 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event170706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6449⟩⟩) (.authority (.operator))

def event170707 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6449⟩⟩) (.finite 9)

def event170708 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event170709 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event170710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event170711 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event170712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 170711

def event170713 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 170709

def event170714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 170712 .coefficient) (.value (.predecessor 1 170713 .coefficient)))

def event170715 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event170716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6451⟩⟩) 0 ⟨392⟩ 170715

def event170717 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6451⟩⟩) 1 ⟨6449⟩ 170707

def event170718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6451⟩⟩) (.sum [.predecessor 0 170716 .coefficient, .predecessor 1 170717 .coefficient])

def event170719 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6451⟩⟩) (.finite 655349)

def event170720 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6462⟩⟩) 0 ⟨6451⟩ 170719

def event170721 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6462⟩⟩) 1 ⟨5426⟩ 170705

def event170722 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6462⟩⟩) (.identity (.predecessor 1 170721 .coefficient))

def event170723 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6462⟩⟩) (.finite 655360)

def event170724 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24338⟩⟩) 0 ⟨6462⟩ 170723

def event170725 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24338⟩⟩) (.authority (.programFamilyFact))

def exact170726RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24338⟩⟩], []⟩, (1)⟩]

theorem exact170726RawTermsValid :
    exact170726RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170726 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24338⟩⟩) exact170726RawTerms (.finite 6) 170725 .exactZero (none)

def event170727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31593⟩⟩) 0 ⟨6462⟩ 170723

def event170728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31593⟩⟩) (.authority (.programFamilyFact))

def exact170729RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31593⟩⟩], []⟩, (1)⟩]

theorem exact170729RawTermsValid :
    exact170729RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170729 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31593⟩⟩) exact170729RawTerms (.finite 6) 170728 .exactZero (none)

def event170730 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31594⟩⟩) 0 ⟨31593⟩ 170729

def event170731 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31594⟩⟩) 1 ⟨24338⟩ 170726

def event170732 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31594⟩⟩) (.product (.predecessor 0 170730 .coefficient) (.predecessor 1 170731 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event170733 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31594⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24338⟩⟩, ⟨.program ⟨257⟩, ⟨31593⟩⟩], []⟩) [⟨.result 170729 .coefficient, true, some 1⟩, ⟨.result 170726 .coefficient, true, some 1⟩])

def event170734 : Event := .survivorFold (1) 170733

def exact170735RawTerms : List Term := []

theorem exact170735RawTermsValid :
    exact170735RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170735 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31594⟩⟩) exact170735RawTerms (.finite 36) 170732 (.finite 36) (some (170733))

def event170736 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31595⟩⟩) 0 ⟨31594⟩ 170735

def event170737 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31595⟩⟩) (.identity (.predecessor 0 170736 .coefficient))

def event170738 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31595⟩⟩) (.finite 36)

def event170739 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31860⟩⟩) 0 ⟨31595⟩ 170738

def event170740 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31860⟩⟩) (.authority (.programFamilyFact))

def exact170741RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31860⟩⟩], []⟩, (1)⟩]

theorem exact170741RawTermsValid :
    exact170741RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170741 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31860⟩⟩) exact170741RawTerms (.finite 6) 170740 .exactZero (none)

def event170742 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31861⟩⟩) 0 ⟨31860⟩ 170741

def event170743 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31861⟩⟩) (.identity (.predecessor 0 170742 .coefficient))

def event170744 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31861⟩⟩) (.finite 6)

def event170745 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32776⟩⟩) 0 ⟨31861⟩ 170744

def event170746 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32776⟩⟩) (.authority (.relationPreimageSource ⟨63⟩))

def exact170747RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32776⟩⟩]⟩, (1)⟩]

theorem exact170747RawTermsValid :
    exact170747RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170747 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32776⟩⟩) exact170747RawTerms (.finite 5647228698) 170746 .exactZero (none)

def event170748 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact170749RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact170749RawTermsValid :
    exact170749RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event170749 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact170749RawTerms .large 170748 .exactZero (none)

def event170750 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32777⟩⟩) 0 ⟨35⟩ 170749

def event170751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32777⟩⟩) 1 ⟨32776⟩ 170747

def eventLeaf10656 : Array AnnotatedEvent := #[
  { event := event170496
    frameStart := 0 },
  { event := event170497
    frameStart := 0 },
  { event := event170498
    frameStart := 0 },
  { event := event170499
    frameStart := 0 },
  { event := event170500
    frameStart := 170500 },
  { event := event170501
    frameStart := 170500 },
  { event := event170502
    frameStart := 170500 },
  { event := event170503
    frameStart := 170500 },
  { event := event170504
    frameStart := 170500 },
  { event := event170505
    frameStart := 170500 },
  { event := event170506
    frameStart := 170500 },
  { event := event170507
    frameStart := 170500 },
  { event := event170508
    frameStart := 170500 },
  { event := event170509
    frameStart := 170500 },
  { event := event170510
    frameStart := 170500 },
  { event := event170511
    frameStart := 170500 }
]

def eventLeaf10657 : Array AnnotatedEvent := #[
  { event := event170512
    frameStart := 170500 },
  { event := event170513
    frameStart := 170500 },
  { event := event170514
    frameStart := 170500 },
  { event := event170515
    frameStart := 170500 },
  { event := event170516
    frameStart := 170500 },
  { event := event170517
    frameStart := 170500 },
  { event := event170518
    frameStart := 170500 },
  { event := event170519
    frameStart := 170500 },
  { event := event170520
    frameStart := 170500 },
  { event := event170521
    frameStart := 170500 },
  { event := event170522
    frameStart := 170500 },
  { event := event170523
    frameStart := 170500 },
  { event := event170524
    frameStart := 170500 },
  { event := event170525
    frameStart := 170500 },
  { event := event170526
    frameStart := 170500 },
  { event := event170527
    frameStart := 170500 }
]

def eventLeaf10658 : Array AnnotatedEvent := #[
  { event := event170528
    frameStart := 170500 },
  { event := event170529
    frameStart := 170500 },
  { event := event170530
    frameStart := 170500 },
  { event := event170531
    frameStart := 170500 },
  { event := event170532
    frameStart := 170500 },
  { event := event170533
    frameStart := 170500 },
  { event := event170534
    frameStart := 170500 },
  { event := event170535
    frameStart := 170500 },
  { event := event170536
    frameStart := 170500 },
  { event := event170537
    frameStart := 170500 },
  { event := event170538
    frameStart := 170500 },
  { event := event170539
    frameStart := 170500 },
  { event := event170540
    frameStart := 170500 },
  { event := event170541
    frameStart := 170500 },
  { event := event170542
    frameStart := 170500 },
  { event := event170543
    frameStart := 170500 }
]

def eventLeaf10659 : Array AnnotatedEvent := #[
  { event := event170544
    frameStart := 170500 },
  { event := event170545
    frameStart := 170500 },
  { event := event170546
    frameStart := 170500 },
  { event := event170547
    frameStart := 170500 },
  { event := event170548
    frameStart := 170548 },
  { event := event170549
    frameStart := 170548 },
  { event := event170550
    frameStart := 170548 },
  { event := event170551
    frameStart := 170548 },
  { event := event170552
    frameStart := 170548 },
  { event := event170553
    frameStart := 170548 },
  { event := event170554
    frameStart := 170548 },
  { event := event170555
    frameStart := 170548 },
  { event := event170556
    frameStart := 170548 },
  { event := event170557
    frameStart := 170548 },
  { event := event170558
    frameStart := 170548 },
  { event := event170559
    frameStart := 170548 }
]

def eventLeaf10660 : Array AnnotatedEvent := #[
  { event := event170560
    frameStart := 170548 },
  { event := event170561
    frameStart := 170548 },
  { event := event170562
    frameStart := 170548 },
  { event := event170563
    frameStart := 170548 },
  { event := event170564
    frameStart := 170548 },
  { event := event170565
    frameStart := 170548 },
  { event := event170566
    frameStart := 170548 },
  { event := event170567
    frameStart := 170548 },
  { event := event170568
    frameStart := 170548 },
  { event := event170569
    frameStart := 170548 },
  { event := event170570
    frameStart := 170548 },
  { event := event170571
    frameStart := 170548 },
  { event := event170572
    frameStart := 170548 },
  { event := event170573
    frameStart := 170548 },
  { event := event170574
    frameStart := 170548 },
  { event := event170575
    frameStart := 170548 }
]

def eventLeaf10661 : Array AnnotatedEvent := #[
  { event := event170576
    frameStart := 170548 },
  { event := event170577
    frameStart := 170548 },
  { event := event170578
    frameStart := 170548 },
  { event := event170579
    frameStart := 170548 },
  { event := event170580
    frameStart := 170548 },
  { event := event170581
    frameStart := 170548 },
  { event := event170582
    frameStart := 170548 },
  { event := event170583
    frameStart := 170548 },
  { event := event170584
    frameStart := 170548 },
  { event := event170585
    frameStart := 170548 },
  { event := event170586
    frameStart := 170548 },
  { event := event170587
    frameStart := 170548 },
  { event := event170588
    frameStart := 170548 },
  { event := event170589
    frameStart := 170548 },
  { event := event170590
    frameStart := 170548 },
  { event := event170591
    frameStart := 170548 }
]

def eventLeaf10662 : Array AnnotatedEvent := #[
  { event := event170592
    frameStart := 170548 },
  { event := event170593
    frameStart := 170548 },
  { event := event170594
    frameStart := 170548 },
  { event := event170595
    frameStart := 170548 },
  { event := event170596
    frameStart := 170548 },
  { event := event170597
    frameStart := 170548 },
  { event := event170598
    frameStart := 170548 },
  { event := event170599
    frameStart := 170548 },
  { event := event170600
    frameStart := 170548 },
  { event := event170601
    frameStart := 170548 },
  { event := event170602
    frameStart := 170548 },
  { event := event170603
    frameStart := 170548 },
  { event := event170604
    frameStart := 170548 },
  { event := event170605
    frameStart := 170548 },
  { event := event170606
    frameStart := 170548 },
  { event := event170607
    frameStart := 170548 }
]

def eventLeaf10663 : Array AnnotatedEvent := #[
  { event := event170608
    frameStart := 170548 },
  { event := event170609
    frameStart := 170548 },
  { event := event170610
    frameStart := 170548 },
  { event := event170611
    frameStart := 170548 },
  { event := event170612
    frameStart := 170548 },
  { event := event170613
    frameStart := 170548 },
  { event := event170614
    frameStart := 170548 },
  { event := event170615
    frameStart := 170548 },
  { event := event170616
    frameStart := 170548 },
  { event := event170617
    frameStart := 170548 },
  { event := event170618
    frameStart := 170548 },
  { event := event170619
    frameStart := 170548 },
  { event := event170620
    frameStart := 170548 },
  { event := event170621
    frameStart := 170548 },
  { event := event170622
    frameStart := 170548 },
  { event := event170623
    frameStart := 170548 }
]

def eventLeaf10664 : Array AnnotatedEvent := #[
  { event := event170624
    frameStart := 170548 },
  { event := event170625
    frameStart := 170548 },
  { event := event170626
    frameStart := 170548 },
  { event := event170627
    frameStart := 170548 },
  { event := event170628
    frameStart := 170548 },
  { event := event170629
    frameStart := 170548 },
  { event := event170630
    frameStart := 170548 },
  { event := event170631
    frameStart := 170548 },
  { event := event170632
    frameStart := 170548 },
  { event := event170633
    frameStart := 170548 },
  { event := event170634
    frameStart := 170548 },
  { event := event170635
    frameStart := 170548 },
  { event := event170636
    frameStart := 170548 },
  { event := event170637
    frameStart := 170548 },
  { event := event170638
    frameStart := 170548 },
  { event := event170639
    frameStart := 170548 }
]

def eventLeaf10665 : Array AnnotatedEvent := #[
  { event := event170640
    frameStart := 170548 },
  { event := event170641
    frameStart := 170548 },
  { event := event170642
    frameStart := 170548 },
  { event := event170643
    frameStart := 170548 },
  { event := event170644
    frameStart := 170548 },
  { event := event170645
    frameStart := 170548 },
  { event := event170646
    frameStart := 170548 },
  { event := event170647
    frameStart := 170548 },
  { event := event170648
    frameStart := 170548 },
  { event := event170649
    frameStart := 170548 },
  { event := event170650
    frameStart := 170548 },
  { event := event170651
    frameStart := 170548 },
  { event := event170652
    frameStart := 170548 },
  { event := event170653
    frameStart := 170548 },
  { event := event170654
    frameStart := 170548 },
  { event := event170655
    frameStart := 170548 }
]

def eventLeaf10666 : Array AnnotatedEvent := #[
  { event := event170656
    frameStart := 170548 },
  { event := event170657
    frameStart := 170548 },
  { event := event170658
    frameStart := 170548 },
  { event := event170659
    frameStart := 170548 },
  { event := event170660
    frameStart := 170548 },
  { event := event170661
    frameStart := 170548 },
  { event := event170662
    frameStart := 170548 },
  { event := event170663
    frameStart := 170548 },
  { event := event170664
    frameStart := 170548 },
  { event := event170665
    frameStart := 170548 },
  { event := event170666
    frameStart := 0 },
  { event := event170667
    frameStart := 0 },
  { event := event170668
    frameStart := 0 },
  { event := event170669
    frameStart := 0 },
  { event := event170670
    frameStart := 0 },
  { event := event170671
    frameStart := 0 }
]

def eventLeaf10667 : Array AnnotatedEvent := #[
  { event := event170672
    frameStart := 0 },
  { event := event170673
    frameStart := 0 },
  { event := event170674
    frameStart := 0 },
  { event := event170675
    frameStart := 0 },
  { event := event170676
    frameStart := 0 },
  { event := event170677
    frameStart := 0 },
  { event := event170678
    frameStart := 0 },
  { event := event170679
    frameStart := 0 },
  { event := event170680
    frameStart := 0 },
  { event := event170681
    frameStart := 0 },
  { event := event170682
    frameStart := 0 },
  { event := event170683
    frameStart := 0 },
  { event := event170684
    frameStart := 0 },
  { event := event170685
    frameStart := 0 },
  { event := event170686
    frameStart := 0 },
  { event := event170687
    frameStart := 0 }
]

def eventLeaf10668 : Array AnnotatedEvent := #[
  { event := event170688
    frameStart := 0 },
  { event := event170689
    frameStart := 0 },
  { event := event170690
    frameStart := 0 },
  { event := event170691
    frameStart := 0 },
  { event := event170692
    frameStart := 0 },
  { event := event170693
    frameStart := 0 },
  { event := event170694
    frameStart := 0 },
  { event := event170695
    frameStart := 0 },
  { event := event170696
    frameStart := 0 },
  { event := event170697
    frameStart := 0 },
  { event := event170698
    frameStart := 0 },
  { event := event170699
    frameStart := 0 },
  { event := event170700
    frameStart := 0 },
  { event := event170701
    frameStart := 0 },
  { event := event170702
    frameStart := 0 },
  { event := event170703
    frameStart := 170703 }
]

def eventLeaf10669 : Array AnnotatedEvent := #[
  { event := event170704
    frameStart := 170703 },
  { event := event170705
    frameStart := 170703 },
  { event := event170706
    frameStart := 170703 },
  { event := event170707
    frameStart := 170703 },
  { event := event170708
    frameStart := 170703 },
  { event := event170709
    frameStart := 170703 },
  { event := event170710
    frameStart := 170703 },
  { event := event170711
    frameStart := 170703 },
  { event := event170712
    frameStart := 170703 },
  { event := event170713
    frameStart := 170703 },
  { event := event170714
    frameStart := 170703 },
  { event := event170715
    frameStart := 170703 },
  { event := event170716
    frameStart := 170703 },
  { event := event170717
    frameStart := 170703 },
  { event := event170718
    frameStart := 170703 },
  { event := event170719
    frameStart := 170703 }
]

def eventLeaf10670 : Array AnnotatedEvent := #[
  { event := event170720
    frameStart := 170703 },
  { event := event170721
    frameStart := 170703 },
  { event := event170722
    frameStart := 170703 },
  { event := event170723
    frameStart := 170703 },
  { event := event170724
    frameStart := 170703 },
  { event := event170725
    frameStart := 170703 },
  { event := event170726
    frameStart := 170703 },
  { event := event170727
    frameStart := 170703 },
  { event := event170728
    frameStart := 170703 },
  { event := event170729
    frameStart := 170703 },
  { event := event170730
    frameStart := 170703 },
  { event := event170731
    frameStart := 170703 },
  { event := event170732
    frameStart := 170703 },
  { event := event170733
    frameStart := 170703 },
  { event := event170734
    frameStart := 170703 },
  { event := event170735
    frameStart := 170703 }
]

def eventLeaf10671 : Array AnnotatedEvent := #[
  { event := event170736
    frameStart := 170703 },
  { event := event170737
    frameStart := 170703 },
  { event := event170738
    frameStart := 170703 },
  { event := event170739
    frameStart := 170703 },
  { event := event170740
    frameStart := 170703 },
  { event := event170741
    frameStart := 170703 },
  { event := event170742
    frameStart := 170703 },
  { event := event170743
    frameStart := 170703 },
  { event := event170744
    frameStart := 170703 },
  { event := event170745
    frameStart := 170703 },
  { event := event170746
    frameStart := 170703 },
  { event := event170747
    frameStart := 170703 },
  { event := event170748
    frameStart := 170703 },
  { event := event170749
    frameStart := 170703 },
  { event := event170750
    frameStart := 170703 },
  { event := event170751
    frameStart := 170703 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events666
