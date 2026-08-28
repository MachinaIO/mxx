import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events084

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event21504 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5558⟩⟩) (.product (.predecessor 0 21502 .coefficient) (.predecessor 1 21503 .coefficient) (⟨false, false, none, none, none⟩))

def event21505 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨5558⟩⟩, .operator (⟨21290, 0⟩, ⟨6550, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩)

def exact21506RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact21506RawTermsValid :
    exact21506RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21506 : Event := .resultExact (⟨.program ⟨214⟩, ⟨5558⟩⟩) exact21506RawTerms .large 21504 .exactZero (none)

def event21507 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5559⟩⟩) 0 ⟨5558⟩ 21506

def event21508 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5559⟩⟩) 1 ⟨22⟩ 6548

def event21509 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5559⟩⟩) (.sum [.predecessor 0 21507 .coefficient, .predecessor 1 21508 .coefficient])

def event21510 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5559⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22⟩⟩]⟩) [⟨.result 6548 .coefficient, false, none⟩])

def event21511 : Event := .survivorFold (1) 21510

def exact21512RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact21512RawTermsValid :
    exact21512RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21512 : Event := .resultExact (⟨.program ⟨214⟩, ⟨5559⟩⟩) exact21512RawTerms .large 21509 (.finite 26) (some (21510))

def event21513 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20263⟩⟩) 0 ⟨5559⟩ 21512

def event21514 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20263⟩⟩) 1 ⟨20262⟩ 21501

def event21515 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20263⟩⟩) (.product (.predecessor 0 21513 .coefficient) (.predecessor 1 21514 .coefficient) (⟨false, false, none, none, none⟩))

def event21516 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20263⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20260⟩⟩]⟩) [⟨.result 21497 .coefficient, false, none⟩])

def event21517 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20263⟩⟩) (.product (.result 21512 .summary) (.transfer 21516) (⟨false, false, none, none, none⟩))

def event21518 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20263⟩⟩, .operator (⟨21512, 0⟩, ⟨21501, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20260⟩⟩]⟩, (1)⟩)

def event21519 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20261⟩⟩)

def event21520 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event21521 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event21522 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event21523 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event21524 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event21525 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event21526 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event21527 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event21528 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 21527

def event21529 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 21525

def event21530 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 21528 .coefficient) (.value (.predecessor 1 21529 .coefficient)))

def event21531 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event21532 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 21531

def event21533 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 21523

def event21534 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 21532 .coefficient, .predecessor 1 21533 .coefficient])

def event21535 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event21536 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 21535

def event21537 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 21521

def event21538 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 21537 .coefficient))

def event21539 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event21540 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13374⟩⟩) 0 ⟨5554⟩ 21539

def event21541 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13374⟩⟩) (.authority (.programFamilyFact))

def exact21542RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13374⟩⟩], []⟩, (1)⟩]

theorem exact21542RawTermsValid :
    exact21542RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21542 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13374⟩⟩) exact21542RawTerms (.finite 60) 21541 .exactZero (none)

def event21543 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10360⟩⟩) 0 ⟨5554⟩ 21539

def event21544 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10360⟩⟩) (.authority (.programFamilyFact))

def exact21545RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩], []⟩, (1)⟩]

theorem exact21545RawTermsValid :
    exact21545RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21545 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10360⟩⟩) exact21545RawTerms (.finite 60) 21544 .exactZero (none)

def event21546 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13375⟩⟩) 0 ⟨10360⟩ 21545

def event21547 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13375⟩⟩) 1 ⟨13374⟩ 21542

def event21548 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13375⟩⟩) (.product (.predecessor 0 21546 .coefficient) (.predecessor 1 21547 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event21549 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13375⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], []⟩) [⟨.result 21545 .coefficient, true, some 1⟩, ⟨.result 21542 .coefficient, true, some 1⟩])

def event21550 : Event := .survivorFold (1) 21549

def exact21551RawTerms : List Term := []

theorem exact21551RawTermsValid :
    exact21551RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21551 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13375⟩⟩) exact21551RawTerms (.finite 3600) 21548 (.finite 3600) (some (21549))

def event21552 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13376⟩⟩) 0 ⟨13375⟩ 21551

def event21553 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13376⟩⟩) (.identity (.predecessor 0 21552 .coefficient))

def event21554 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13376⟩⟩) (.finite 3600)

def event21555 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20260⟩⟩) 0 ⟨13376⟩ 21554

def event21556 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20260⟩⟩) (.authority (.relationPreimageSource ⟨26⟩))

def exact21557RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20260⟩⟩]⟩, (1)⟩]

theorem exact21557RawTermsValid :
    exact21557RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21557 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20260⟩⟩) exact21557RawTerms (.finite 136065468) 21556 .exactZero (none)

def event21558 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact21559RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact21559RawTermsValid :
    exact21559RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21559 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact21559RawTerms .large 21558 .exactZero (none)

def event21560 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20261⟩⟩) 0 ⟨6⟩ 21559

def event21561 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20261⟩⟩) 1 ⟨20260⟩ 21557

def event21562 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20261⟩⟩) (.product (.predecessor 0 21560 .coefficient) (.predecessor 1 21561 .coefficient) (⟨false, false, none, none, none⟩))

def event21563 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20261⟩⟩, .operator (⟨21559, 0⟩, ⟨21557, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20260⟩⟩]⟩, (1)⟩)

def exact21564RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20260⟩⟩]⟩, (1)⟩]

theorem exact21564RawTermsValid :
    exact21564RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21564 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20261⟩⟩) exact21564RawTerms .large 21562 .exactZero (none)

def event21565 : Event := .preFoldPolynomial 21564 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20260⟩⟩]⟩, (1)⟩] .exactZero none

def exact21566RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20260⟩⟩]⟩, (1)⟩]

def event21566 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20261⟩⟩) 21565 exact21566RawTerms .large 21562 .exactZero (none)

def event21567 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25777⟩⟩)

def event21568 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event21569 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event21570 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event21571 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event21572 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event21573 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event21574 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event21575 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event21576 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 21575

def event21577 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 21573

def event21578 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 21576 .coefficient) (.value (.predecessor 1 21577 .coefficient)))

def event21579 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event21580 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 21579

def event21581 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 21571

def event21582 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 21580 .coefficient, .predecessor 1 21581 .coefficient])

def event21583 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event21584 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 21583

def event21585 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 21569

def event21586 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 21585 .coefficient))

def event21587 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event21588 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13374⟩⟩) 0 ⟨5554⟩ 21587

def event21589 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13374⟩⟩) (.authority (.programFamilyFact))

def exact21590RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13374⟩⟩], []⟩, (1)⟩]

theorem exact21590RawTermsValid :
    exact21590RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21590 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13374⟩⟩) exact21590RawTerms (.finite 60) 21589 .exactZero (none)

def event21591 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10360⟩⟩) 0 ⟨5554⟩ 21587

def event21592 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10360⟩⟩) (.authority (.programFamilyFact))

def exact21593RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩], []⟩, (1)⟩]

theorem exact21593RawTermsValid :
    exact21593RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21593 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10360⟩⟩) exact21593RawTerms (.finite 60) 21592 .exactZero (none)

def event21594 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13375⟩⟩) 0 ⟨10360⟩ 21593

def event21595 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13375⟩⟩) 1 ⟨13374⟩ 21590

def event21596 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13375⟩⟩) (.product (.predecessor 0 21594 .coefficient) (.predecessor 1 21595 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event21597 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13375⟩⟩, .operator (⟨21593, 0⟩, ⟨21590, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], []⟩, (1)⟩)

def exact21598RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], []⟩, (1)⟩]

theorem exact21598RawTermsValid :
    exact21598RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21598 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13375⟩⟩) exact21598RawTerms (.finite 3600) 21596 .exactZero (none)

def event21599 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13376⟩⟩) 0 ⟨13375⟩ 21598

def event21600 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13376⟩⟩) (.identity (.predecessor 0 21599 .coefficient))

def event21601 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13376⟩⟩) (.finite 3600)

def event21602 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23421⟩⟩) 0 ⟨13376⟩ 21601

def event21603 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23421⟩⟩) (.authority (.programFamilyFact))

def event21604 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23421⟩⟩) (.finite 3720)

def event21605 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event21606 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23422⟩⟩) 0 ⟨6689⟩ 21605

def event21607 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23422⟩⟩) 1 ⟨23421⟩ 21604

def event21608 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23422⟩⟩) (.authority (.operator))

def exact21609RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23422⟩⟩]⟩, (1)⟩]

theorem exact21609RawTermsValid :
    exact21609RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21609 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23422⟩⟩) exact21609RawTerms .large 21608 .exactZero (none)

def event21610 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25773⟩⟩) 0 ⟨23422⟩ 21609

def event21611 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25773⟩⟩) (.authority (.operator))

def exact21612RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩, (1)⟩]

theorem exact21612RawTermsValid :
    exact21612RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21612 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25773⟩⟩) exact21612RawTerms (.finite 8192) 21611 .exactZero (none)

def event21613 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event21614 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event21615 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13458⟩⟩) 0 ⟨13376⟩ 21601

def event21616 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13458⟩⟩) 1 ⟨110⟩ 21614

def event21617 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13458⟩⟩) (.sum [.predecessor 0 21615 .coefficient, .predecessor 1 21616 .coefficient])

def event21618 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13458⟩⟩) (.finite 3600)

def event21619 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13459⟩⟩) 0 ⟨13458⟩ 21618

def event21620 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13459⟩⟩) (.identity (.predecessor 0 21619 .coefficient))

def exact21621RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], []⟩, (1)⟩]

theorem exact21621RawTermsValid :
    exact21621RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21621 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13459⟩⟩) exact21621RawTerms (.finite 3600) 21620 .exactZero (none)

def event21622 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact21623RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact21623RawTermsValid :
    exact21623RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21623 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact21623RawTerms .large 21622 .exactZero (none)

def event21624 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13460⟩⟩) 0 ⟨6544⟩ 21623

def event21625 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13460⟩⟩) 1 ⟨13459⟩ 21621

def event21626 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13460⟩⟩) (.product (.predecessor 0 21624 .coefficient) (.predecessor 1 21625 .coefficient) (⟨false, false, none, none, none⟩))

def event21627 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13460⟩⟩, .operator (⟨21623, 0⟩, ⟨21621, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact21628RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact21628RawTermsValid :
    exact21628RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21628 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13460⟩⟩) exact21628RawTerms .large 21626 .exactZero (none)

def event21629 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event21630 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event21631 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 21605

def event21632 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact21633RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact21633RawTermsValid :
    exact21633RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21633 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact21633RawTerms .large 21632 .exactZero (none)

def event21634 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6790⟩⟩) 0 ⟨6757⟩ 21633

def event21635 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6790⟩⟩) (.identity (.predecessor 0 21634 .coefficient))

def exact21636RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6790⟩⟩]⟩, (1)⟩]

theorem exact21636RawTermsValid :
    exact21636RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21636 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6790⟩⟩) exact21636RawTerms .large 21635 .exactZero (none)

def event21637 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7882⟩⟩) 0 ⟨6790⟩ 21636

def event21638 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7882⟩⟩) (.authority (.operator))

def exact21639RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩, (1)⟩]

theorem exact21639RawTermsValid :
    exact21639RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21639 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7882⟩⟩) exact21639RawTerms (.finite 8192) 21638 .exactZero (none)

def event21640 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7883⟩⟩) 0 ⟨7882⟩ 21639

def event21641 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7883⟩⟩) 1 ⟨2348⟩ 21630

def event21642 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7883⟩⟩) (.scale (.predecessor 0 21640 .coefficient) (.value (.predecessor 1 21641 .coefficient)))

def exact21643RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩, (1)⟩]

theorem exact21643RawTermsValid :
    exact21643RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21643 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7883⟩⟩) exact21643RawTerms (.finite 8192) 21642 .exactZero (none)

def event21644 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6770⟩⟩) 0 ⟨6757⟩ 21633

def event21645 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6770⟩⟩) (.identity (.predecessor 0 21644 .coefficient))

def exact21646RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩]⟩, (1)⟩]

theorem exact21646RawTermsValid :
    exact21646RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21646 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6770⟩⟩) exact21646RawTerms .large 21645 .exactZero (none)

def event21647 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7884⟩⟩) 0 ⟨6770⟩ 21646

def event21648 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7884⟩⟩) 1 ⟨7883⟩ 21643

def event21649 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7884⟩⟩) (.product (.predecessor 0 21647 .coefficient) (.predecessor 1 21648 .coefficient) (⟨false, false, none, none, none⟩))

def event21650 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7884⟩⟩, .operator (⟨21646, 0⟩, ⟨21643, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩, (1)⟩)

def exact21651RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩, (1)⟩]

theorem exact21651RawTermsValid :
    exact21651RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21651 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7884⟩⟩) exact21651RawTerms .large 21649 .exactZero (none)

def event21652 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13461⟩⟩) 0 ⟨7884⟩ 21651

def event21653 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13461⟩⟩) 1 ⟨13460⟩ 21628

def event21654 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13461⟩⟩) (.sum [.predecessor 0 21652 .coefficient, .predecessor 1 21653 .coefficient])

def exact21655RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact21655RawTermsValid :
    exact21655RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21655 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13461⟩⟩) exact21655RawTerms .large 21654 .exactZero (none)

def event21656 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25776⟩⟩) 0 ⟨13461⟩ 21655

def event21657 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25776⟩⟩) 1 ⟨25773⟩ 21612

def event21658 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25776⟩⟩) (.product (.predecessor 0 21656 .coefficient) (.predecessor 1 21657 .coefficient) (⟨false, false, none, none, none⟩))

def event21659 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25776⟩⟩, .operator (⟨21655, 0⟩, ⟨21612, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩, (1)⟩)

def event21660 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25776⟩⟩, .operator (⟨21655, 1⟩, ⟨21612, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩, (-1)⟩)

def event21661 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25776⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25773⟩⟩) ⟨23422⟩ 21609)

def event21662 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25776⟩⟩, .relation 21661 0, ⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨23422⟩⟩]⟩, (-1)⟩)

def exact21663RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨23422⟩⟩]⟩, (-1)⟩]

theorem exact21663RawTermsValid :
    exact21663RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21663 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25776⟩⟩) exact21663RawTerms .large 21658 .exactZero (none)

def event21664 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17023⟩⟩) 0 ⟨13376⟩ 21601

def event21665 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17023⟩⟩) (.authority (.programFamilyFact))

def exact21666RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17023⟩⟩], []⟩, (1)⟩]

theorem exact21666RawTermsValid :
    exact21666RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21666 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17023⟩⟩) exact21666RawTerms (.finite 60) 21665 .exactZero (none)

def event21667 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17025⟩⟩) 0 ⟨6544⟩ 21623

def event21668 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17025⟩⟩) 1 ⟨17023⟩ 21666

def event21669 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17025⟩⟩) (.product (.predecessor 0 21667 .coefficient) (.predecessor 1 21668 .coefficient) (⟨false, true, none, none, some 1⟩))

def event21670 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17025⟩⟩, .operator (⟨21623, 0⟩, ⟨21666, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17023⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact21671RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17023⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact21671RawTermsValid :
    exact21671RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21671 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17025⟩⟩) exact21671RawTerms .large 21669 .exactZero (none)

def event21672 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6707⟩⟩) 0 ⟨6689⟩ 21605

def event21673 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6707⟩⟩) (.authority (.operator))

def exact21674RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩]

theorem exact21674RawTermsValid :
    exact21674RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21674 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6707⟩⟩) exact21674RawTerms .large 21673 .exactZero (none)

def event21675 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17026⟩⟩) 0 ⟨6707⟩ 21674

def event21676 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17026⟩⟩) 1 ⟨17025⟩ 21671

def event21677 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17026⟩⟩) (.sum [.predecessor 0 21675 .coefficient, .predecessor 1 21676 .coefficient])

def exact21678RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17023⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact21678RawTermsValid :
    exact21678RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21678 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17026⟩⟩) exact21678RawTerms .large 21677 .exactZero (none)

def event21679 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25777⟩⟩) 0 ⟨17026⟩ 21678

def event21680 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25777⟩⟩) 1 ⟨25776⟩ 21663

def event21681 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25777⟩⟩) (.sum [.predecessor 0 21679 .coefficient, .predecessor 1 21680 .coefficient])

def exact21682RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨23422⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17023⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact21682RawTermsValid :
    exact21682RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21682 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25777⟩⟩) exact21682RawTerms .large 21681 .exactZero (none)

def event21683 : Event := .preFoldPolynomial 21682 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨23422⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17023⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact21684RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨23422⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17023⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event21684 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25777⟩⟩) 21683 exact21684RawTerms .large 21681 .exactZero (none)

def event21685 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨13376⟩⟩) ⟨⟨120⟩, ⟨26⟩, ⟨109⟩⟩ ⟨21519, 21685⟩

def event21686 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20263⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20260⟩⟩]⟩) (1) 0 2 (.universal 21685 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20260⟩⟩]⟩) (none) 21684)

def event21687 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20263⟩⟩, .relation 21686 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩)

def event21688 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20263⟩⟩, .relation 21686 1, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩, (-1)⟩)

def event21689 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20263⟩⟩, .relation 21686 2, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨23422⟩⟩]⟩, (1)⟩)

def event21690 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20263⟩⟩, .relation 21686 3, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17023⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact21691RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨23422⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17023⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact21691RawTermsValid :
    exact21691RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21691 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20263⟩⟩) exact21691RawTerms .large 21515 (.finite 1811303510016) (some (21517))

def event21692 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25775⟩⟩) 0 ⟨20263⟩ 21691

def event21693 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25775⟩⟩) 1 ⟨25774⟩ 21494

def event21694 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25775⟩⟩) (.sum [.predecessor 0 21692 .coefficient, .predecessor 1 21693 .coefficient])

def event21695 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25775⟩⟩, .operator (⟨21691, 2⟩, ⟨21494, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨23422⟩⟩]⟩, (-1)⟩)

def event21696 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25775⟩⟩, .operator (⟨21691, 1⟩, ⟨21494, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩, (1)⟩)

def event21697 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25775⟩⟩) (.sum [.result 21691 .summary, .result 21494 .summary])

def exact21698RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17023⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact21698RawTermsValid :
    exact21698RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21698 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25775⟩⟩) exact21698RawTerms .large 21694 (.finite 352188964155392) (some (21697))

def event21699 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30185⟩⟩) 0 ⟨25775⟩ 21698

def event21700 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30185⟩⟩) 1 ⟨30183⟩ 21405

def event21701 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30185⟩⟩) (.product (.predecessor 0 21699 .coefficient) (.predecessor 1 21700 .coefficient) (⟨false, false, none, none, none⟩))

def event21702 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30185⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨30183⟩⟩]⟩) [⟨.result 21405 .coefficient, false, none⟩])

def event21703 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30185⟩⟩) (.product (.result 21698 .summary) (.transfer 21702) (⟨false, false, none, none, none⟩))

def event21704 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30185⟩⟩, .operator (⟨21698, 0⟩, ⟨21405, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30183⟩⟩]⟩, (1)⟩)

def event21705 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30185⟩⟩, .operator (⟨21698, 1⟩, ⟨21405, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17023⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30183⟩⟩]⟩, (-1)⟩)

def event21706 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30185⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17023⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30183⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨30183⟩⟩) ⟨24801⟩ 21402)

def event21707 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30185⟩⟩, .relation 21706 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17023⟩⟩], [⟨.program ⟨214⟩, ⟨24801⟩⟩]⟩, (-1)⟩)

def exact21708RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17023⟩⟩], [⟨.program ⟨214⟩, ⟨24801⟩⟩]⟩, (-1)⟩]

theorem exact21708RawTermsValid :
    exact21708RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21708 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30185⟩⟩) exact21708RawTerms .large 21701 (.finite 1292539133473715126272) (some (21703))

def event21709 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22852⟩⟩) 0 ⟨17024⟩ 859

def event21710 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22852⟩⟩) (.authority (.relationPreimageSource ⟨65⟩))

def exact21711RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22852⟩⟩]⟩, (1)⟩]

theorem exact21711RawTermsValid :
    exact21711RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21711 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22852⟩⟩) exact21711RawTerms (.finite 136065468) 21710 .exactZero (none)

def event21712 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22854⟩⟩) 0 ⟨22852⟩ 21711

def event21713 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22854⟩⟩) 1 ⟨2348⟩ 4

def event21714 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22854⟩⟩) (.scale (.predecessor 0 21712 .coefficient) (.value (.predecessor 1 21713 .coefficient)))

def exact21715RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22852⟩⟩]⟩, (1)⟩]

theorem exact21715RawTermsValid :
    exact21715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21715 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22854⟩⟩) exact21715RawTerms (.finite 136065468) 21714 .exactZero (none)

def event21716 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22855⟩⟩) 0 ⟨5559⟩ 21512

def event21717 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22855⟩⟩) 1 ⟨22854⟩ 21715

def event21718 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22855⟩⟩) (.product (.predecessor 0 21716 .coefficient) (.predecessor 1 21717 .coefficient) (⟨false, false, none, none, none⟩))

def event21719 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22855⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22852⟩⟩]⟩) [⟨.result 21711 .coefficient, false, none⟩])

def event21720 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22855⟩⟩) (.product (.result 21512 .summary) (.transfer 21719) (⟨false, false, none, none, none⟩))

def event21721 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22855⟩⟩, .operator (⟨21512, 0⟩, ⟨21715, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22852⟩⟩]⟩, (1)⟩)

def event21722 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22853⟩⟩)

def event21723 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event21724 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event21725 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event21726 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event21727 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event21728 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event21729 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event21730 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event21731 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 21730

def event21732 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 21728

def event21733 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 21731 .coefficient) (.value (.predecessor 1 21732 .coefficient)))

def event21734 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event21735 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 21734

def event21736 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 21726

def event21737 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 21735 .coefficient, .predecessor 1 21736 .coefficient])

def event21738 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event21739 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 21738

def event21740 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 21724

def event21741 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 21740 .coefficient))

def event21742 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event21743 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13374⟩⟩) 0 ⟨5554⟩ 21742

def event21744 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13374⟩⟩) (.authority (.programFamilyFact))

def exact21745RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13374⟩⟩], []⟩, (1)⟩]

theorem exact21745RawTermsValid :
    exact21745RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21745 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13374⟩⟩) exact21745RawTerms (.finite 60) 21744 .exactZero (none)

def event21746 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10360⟩⟩) 0 ⟨5554⟩ 21742

def event21747 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10360⟩⟩) (.authority (.programFamilyFact))

def exact21748RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩], []⟩, (1)⟩]

theorem exact21748RawTermsValid :
    exact21748RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21748 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10360⟩⟩) exact21748RawTerms (.finite 60) 21747 .exactZero (none)

def event21749 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13375⟩⟩) 0 ⟨10360⟩ 21748

def event21750 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13375⟩⟩) 1 ⟨13374⟩ 21745

def event21751 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13375⟩⟩) (.product (.predecessor 0 21749 .coefficient) (.predecessor 1 21750 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event21752 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13375⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], []⟩) [⟨.result 21748 .coefficient, true, some 1⟩, ⟨.result 21745 .coefficient, true, some 1⟩])

def event21753 : Event := .survivorFold (1) 21752

def exact21754RawTerms : List Term := []

theorem exact21754RawTermsValid :
    exact21754RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event21754 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13375⟩⟩) exact21754RawTerms (.finite 3600) 21751 (.finite 3600) (some (21752))

def event21755 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13376⟩⟩) 0 ⟨13375⟩ 21754

def event21756 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13376⟩⟩) (.identity (.predecessor 0 21755 .coefficient))

def event21757 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13376⟩⟩) (.finite 3600)

def event21758 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17023⟩⟩) 0 ⟨13376⟩ 21757

def event21759 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17023⟩⟩) (.authority (.programFamilyFact))

def eventLeaf1344 : Array AnnotatedEvent := #[
  { event := event21504
    frameStart := 0 },
  { event := event21505
    frameStart := 0 },
  { event := event21506
    frameStart := 0 },
  { event := event21507
    frameStart := 0 },
  { event := event21508
    frameStart := 0 },
  { event := event21509
    frameStart := 0 },
  { event := event21510
    frameStart := 0 },
  { event := event21511
    frameStart := 0 },
  { event := event21512
    frameStart := 0 },
  { event := event21513
    frameStart := 0 },
  { event := event21514
    frameStart := 0 },
  { event := event21515
    frameStart := 0 },
  { event := event21516
    frameStart := 0 },
  { event := event21517
    frameStart := 0 },
  { event := event21518
    frameStart := 0 },
  { event := event21519
    frameStart := 21519 }
]

def eventLeaf1345 : Array AnnotatedEvent := #[
  { event := event21520
    frameStart := 21519 },
  { event := event21521
    frameStart := 21519 },
  { event := event21522
    frameStart := 21519 },
  { event := event21523
    frameStart := 21519 },
  { event := event21524
    frameStart := 21519 },
  { event := event21525
    frameStart := 21519 },
  { event := event21526
    frameStart := 21519 },
  { event := event21527
    frameStart := 21519 },
  { event := event21528
    frameStart := 21519 },
  { event := event21529
    frameStart := 21519 },
  { event := event21530
    frameStart := 21519 },
  { event := event21531
    frameStart := 21519 },
  { event := event21532
    frameStart := 21519 },
  { event := event21533
    frameStart := 21519 },
  { event := event21534
    frameStart := 21519 },
  { event := event21535
    frameStart := 21519 }
]

def eventLeaf1346 : Array AnnotatedEvent := #[
  { event := event21536
    frameStart := 21519 },
  { event := event21537
    frameStart := 21519 },
  { event := event21538
    frameStart := 21519 },
  { event := event21539
    frameStart := 21519 },
  { event := event21540
    frameStart := 21519 },
  { event := event21541
    frameStart := 21519 },
  { event := event21542
    frameStart := 21519 },
  { event := event21543
    frameStart := 21519 },
  { event := event21544
    frameStart := 21519 },
  { event := event21545
    frameStart := 21519 },
  { event := event21546
    frameStart := 21519 },
  { event := event21547
    frameStart := 21519 },
  { event := event21548
    frameStart := 21519 },
  { event := event21549
    frameStart := 21519 },
  { event := event21550
    frameStart := 21519 },
  { event := event21551
    frameStart := 21519 }
]

def eventLeaf1347 : Array AnnotatedEvent := #[
  { event := event21552
    frameStart := 21519 },
  { event := event21553
    frameStart := 21519 },
  { event := event21554
    frameStart := 21519 },
  { event := event21555
    frameStart := 21519 },
  { event := event21556
    frameStart := 21519 },
  { event := event21557
    frameStart := 21519 },
  { event := event21558
    frameStart := 21519 },
  { event := event21559
    frameStart := 21519 },
  { event := event21560
    frameStart := 21519 },
  { event := event21561
    frameStart := 21519 },
  { event := event21562
    frameStart := 21519 },
  { event := event21563
    frameStart := 21519 },
  { event := event21564
    frameStart := 21519 },
  { event := event21565
    frameStart := 21519 },
  { event := event21566
    frameStart := 21519 },
  { event := event21567
    frameStart := 21567 }
]

def eventLeaf1348 : Array AnnotatedEvent := #[
  { event := event21568
    frameStart := 21567 },
  { event := event21569
    frameStart := 21567 },
  { event := event21570
    frameStart := 21567 },
  { event := event21571
    frameStart := 21567 },
  { event := event21572
    frameStart := 21567 },
  { event := event21573
    frameStart := 21567 },
  { event := event21574
    frameStart := 21567 },
  { event := event21575
    frameStart := 21567 },
  { event := event21576
    frameStart := 21567 },
  { event := event21577
    frameStart := 21567 },
  { event := event21578
    frameStart := 21567 },
  { event := event21579
    frameStart := 21567 },
  { event := event21580
    frameStart := 21567 },
  { event := event21581
    frameStart := 21567 },
  { event := event21582
    frameStart := 21567 },
  { event := event21583
    frameStart := 21567 }
]

def eventLeaf1349 : Array AnnotatedEvent := #[
  { event := event21584
    frameStart := 21567 },
  { event := event21585
    frameStart := 21567 },
  { event := event21586
    frameStart := 21567 },
  { event := event21587
    frameStart := 21567 },
  { event := event21588
    frameStart := 21567 },
  { event := event21589
    frameStart := 21567 },
  { event := event21590
    frameStart := 21567 },
  { event := event21591
    frameStart := 21567 },
  { event := event21592
    frameStart := 21567 },
  { event := event21593
    frameStart := 21567 },
  { event := event21594
    frameStart := 21567 },
  { event := event21595
    frameStart := 21567 },
  { event := event21596
    frameStart := 21567 },
  { event := event21597
    frameStart := 21567 },
  { event := event21598
    frameStart := 21567 },
  { event := event21599
    frameStart := 21567 }
]

def eventLeaf1350 : Array AnnotatedEvent := #[
  { event := event21600
    frameStart := 21567 },
  { event := event21601
    frameStart := 21567 },
  { event := event21602
    frameStart := 21567 },
  { event := event21603
    frameStart := 21567 },
  { event := event21604
    frameStart := 21567 },
  { event := event21605
    frameStart := 21567 },
  { event := event21606
    frameStart := 21567 },
  { event := event21607
    frameStart := 21567 },
  { event := event21608
    frameStart := 21567 },
  { event := event21609
    frameStart := 21567 },
  { event := event21610
    frameStart := 21567 },
  { event := event21611
    frameStart := 21567 },
  { event := event21612
    frameStart := 21567 },
  { event := event21613
    frameStart := 21567 },
  { event := event21614
    frameStart := 21567 },
  { event := event21615
    frameStart := 21567 }
]

def eventLeaf1351 : Array AnnotatedEvent := #[
  { event := event21616
    frameStart := 21567 },
  { event := event21617
    frameStart := 21567 },
  { event := event21618
    frameStart := 21567 },
  { event := event21619
    frameStart := 21567 },
  { event := event21620
    frameStart := 21567 },
  { event := event21621
    frameStart := 21567 },
  { event := event21622
    frameStart := 21567 },
  { event := event21623
    frameStart := 21567 },
  { event := event21624
    frameStart := 21567 },
  { event := event21625
    frameStart := 21567 },
  { event := event21626
    frameStart := 21567 },
  { event := event21627
    frameStart := 21567 },
  { event := event21628
    frameStart := 21567 },
  { event := event21629
    frameStart := 21567 },
  { event := event21630
    frameStart := 21567 },
  { event := event21631
    frameStart := 21567 }
]

def eventLeaf1352 : Array AnnotatedEvent := #[
  { event := event21632
    frameStart := 21567 },
  { event := event21633
    frameStart := 21567 },
  { event := event21634
    frameStart := 21567 },
  { event := event21635
    frameStart := 21567 },
  { event := event21636
    frameStart := 21567 },
  { event := event21637
    frameStart := 21567 },
  { event := event21638
    frameStart := 21567 },
  { event := event21639
    frameStart := 21567 },
  { event := event21640
    frameStart := 21567 },
  { event := event21641
    frameStart := 21567 },
  { event := event21642
    frameStart := 21567 },
  { event := event21643
    frameStart := 21567 },
  { event := event21644
    frameStart := 21567 },
  { event := event21645
    frameStart := 21567 },
  { event := event21646
    frameStart := 21567 },
  { event := event21647
    frameStart := 21567 }
]

def eventLeaf1353 : Array AnnotatedEvent := #[
  { event := event21648
    frameStart := 21567 },
  { event := event21649
    frameStart := 21567 },
  { event := event21650
    frameStart := 21567 },
  { event := event21651
    frameStart := 21567 },
  { event := event21652
    frameStart := 21567 },
  { event := event21653
    frameStart := 21567 },
  { event := event21654
    frameStart := 21567 },
  { event := event21655
    frameStart := 21567 },
  { event := event21656
    frameStart := 21567 },
  { event := event21657
    frameStart := 21567 },
  { event := event21658
    frameStart := 21567 },
  { event := event21659
    frameStart := 21567 },
  { event := event21660
    frameStart := 21567 },
  { event := event21661
    frameStart := 21567 },
  { event := event21662
    frameStart := 21567 },
  { event := event21663
    frameStart := 21567 }
]

def eventLeaf1354 : Array AnnotatedEvent := #[
  { event := event21664
    frameStart := 21567 },
  { event := event21665
    frameStart := 21567 },
  { event := event21666
    frameStart := 21567 },
  { event := event21667
    frameStart := 21567 },
  { event := event21668
    frameStart := 21567 },
  { event := event21669
    frameStart := 21567 },
  { event := event21670
    frameStart := 21567 },
  { event := event21671
    frameStart := 21567 },
  { event := event21672
    frameStart := 21567 },
  { event := event21673
    frameStart := 21567 },
  { event := event21674
    frameStart := 21567 },
  { event := event21675
    frameStart := 21567 },
  { event := event21676
    frameStart := 21567 },
  { event := event21677
    frameStart := 21567 },
  { event := event21678
    frameStart := 21567 },
  { event := event21679
    frameStart := 21567 }
]

def eventLeaf1355 : Array AnnotatedEvent := #[
  { event := event21680
    frameStart := 21567 },
  { event := event21681
    frameStart := 21567 },
  { event := event21682
    frameStart := 21567 },
  { event := event21683
    frameStart := 21567 },
  { event := event21684
    frameStart := 21567 },
  { event := event21685
    frameStart := 0 },
  { event := event21686
    frameStart := 0 },
  { event := event21687
    frameStart := 0 },
  { event := event21688
    frameStart := 0 },
  { event := event21689
    frameStart := 0 },
  { event := event21690
    frameStart := 0 },
  { event := event21691
    frameStart := 0 },
  { event := event21692
    frameStart := 0 },
  { event := event21693
    frameStart := 0 },
  { event := event21694
    frameStart := 0 },
  { event := event21695
    frameStart := 0 }
]

def eventLeaf1356 : Array AnnotatedEvent := #[
  { event := event21696
    frameStart := 0 },
  { event := event21697
    frameStart := 0 },
  { event := event21698
    frameStart := 0 },
  { event := event21699
    frameStart := 0 },
  { event := event21700
    frameStart := 0 },
  { event := event21701
    frameStart := 0 },
  { event := event21702
    frameStart := 0 },
  { event := event21703
    frameStart := 0 },
  { event := event21704
    frameStart := 0 },
  { event := event21705
    frameStart := 0 },
  { event := event21706
    frameStart := 0 },
  { event := event21707
    frameStart := 0 },
  { event := event21708
    frameStart := 0 },
  { event := event21709
    frameStart := 0 },
  { event := event21710
    frameStart := 0 },
  { event := event21711
    frameStart := 0 }
]

def eventLeaf1357 : Array AnnotatedEvent := #[
  { event := event21712
    frameStart := 0 },
  { event := event21713
    frameStart := 0 },
  { event := event21714
    frameStart := 0 },
  { event := event21715
    frameStart := 0 },
  { event := event21716
    frameStart := 0 },
  { event := event21717
    frameStart := 0 },
  { event := event21718
    frameStart := 0 },
  { event := event21719
    frameStart := 0 },
  { event := event21720
    frameStart := 0 },
  { event := event21721
    frameStart := 0 },
  { event := event21722
    frameStart := 21722 },
  { event := event21723
    frameStart := 21722 },
  { event := event21724
    frameStart := 21722 },
  { event := event21725
    frameStart := 21722 },
  { event := event21726
    frameStart := 21722 },
  { event := event21727
    frameStart := 21722 }
]

def eventLeaf1358 : Array AnnotatedEvent := #[
  { event := event21728
    frameStart := 21722 },
  { event := event21729
    frameStart := 21722 },
  { event := event21730
    frameStart := 21722 },
  { event := event21731
    frameStart := 21722 },
  { event := event21732
    frameStart := 21722 },
  { event := event21733
    frameStart := 21722 },
  { event := event21734
    frameStart := 21722 },
  { event := event21735
    frameStart := 21722 },
  { event := event21736
    frameStart := 21722 },
  { event := event21737
    frameStart := 21722 },
  { event := event21738
    frameStart := 21722 },
  { event := event21739
    frameStart := 21722 },
  { event := event21740
    frameStart := 21722 },
  { event := event21741
    frameStart := 21722 },
  { event := event21742
    frameStart := 21722 },
  { event := event21743
    frameStart := 21722 }
]

def eventLeaf1359 : Array AnnotatedEvent := #[
  { event := event21744
    frameStart := 21722 },
  { event := event21745
    frameStart := 21722 },
  { event := event21746
    frameStart := 21722 },
  { event := event21747
    frameStart := 21722 },
  { event := event21748
    frameStart := 21722 },
  { event := event21749
    frameStart := 21722 },
  { event := event21750
    frameStart := 21722 },
  { event := event21751
    frameStart := 21722 },
  { event := event21752
    frameStart := 21722 },
  { event := event21753
    frameStart := 21722 },
  { event := event21754
    frameStart := 21722 },
  { event := event21755
    frameStart := 21722 },
  { event := event21756
    frameStart := 21722 },
  { event := event21757
    frameStart := 21722 },
  { event := event21758
    frameStart := 21722 },
  { event := event21759
    frameStart := 21722 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events084
