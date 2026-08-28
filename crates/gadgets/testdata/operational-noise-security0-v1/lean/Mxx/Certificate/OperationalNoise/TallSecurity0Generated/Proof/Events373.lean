import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events373

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event95488 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25593⟩⟩) (.sum [.predecessor 0 95486 .coefficient, .predecessor 1 95487 .coefficient])

def event95489 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25593⟩⟩, .operator (⟨95485, 2⟩, ⟨95323, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], [⟨.program ⟨214⟩, ⟨23326⟩⟩]⟩, (-1)⟩)

def event95490 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25593⟩⟩, .operator (⟨95485, 1⟩, ⟨95323, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25591⟩⟩]⟩, (1)⟩)

def event95491 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25593⟩⟩) (.sum [.result 95485 .summary, .result 95323 .summary])

def exact95492RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact95492RawTermsValid :
    exact95492RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95492 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25593⟩⟩) exact95492RawTerms .large 95488 (.finite 352164536528896) (some (95491))

def event95493 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29569⟩⟩) 0 ⟨25593⟩ 95492

def event95494 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29569⟩⟩) 1 ⟨29567⟩ 95239

def event95495 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29569⟩⟩) (.product (.predecessor 0 95493 .coefficient) (.predecessor 1 95494 .coefficient) (⟨false, false, none, none, none⟩))

def event95496 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29569⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩) [⟨.result 95239 .coefficient, false, none⟩])

def event95497 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29569⟩⟩) (.product (.result 95492 .summary) (.transfer 95496) (⟨false, false, none, none, none⟩))

def event95498 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29569⟩⟩, .operator (⟨95492, 0⟩, ⟨95239, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩, (1)⟩)

def event95499 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29569⟩⟩, .operator (⟨95492, 1⟩, ⟨95239, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩, (-1)⟩)

def event95500 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29569⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29567⟩⟩) ⟨24657⟩ 95236)

def event95501 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29569⟩⟩, .relation 95500 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨24657⟩⟩]⟩, (-1)⟩)

def exact95502RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨24657⟩⟩]⟩, (-1)⟩]

theorem exact95502RawTermsValid :
    exact95502RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95502 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29569⟩⟩) exact95502RawTerms .large 95495 (.finite 1292449483693632782336) (some (95497))

def event95503 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22541⟩⟩) 0 ⟨16743⟩ 4629

def event95504 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22541⟩⟩) (.authority (.relationPreimageSource ⟨61⟩))

def exact95505RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22541⟩⟩]⟩, (1)⟩]

theorem exact95505RawTermsValid :
    exact95505RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95505 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22541⟩⟩) exact95505RawTerms (.finite 136065468) 95504 .exactZero (none)

def event95506 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22543⟩⟩) 0 ⟨22541⟩ 95505

def event95507 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22543⟩⟩) 1 ⟨2348⟩ 4

def event95508 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22543⟩⟩) (.scale (.predecessor 0 95506 .coefficient) (.value (.predecessor 1 95507 .coefficient)))

def exact95509RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22541⟩⟩]⟩, (1)⟩]

theorem exact95509RawTermsValid :
    exact95509RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95509 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22543⟩⟩) exact95509RawTerms (.finite 136065468) 95508 .exactZero (none)

def event95510 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22544⟩⟩) 0 ⟨5509⟩ 94462

def event95511 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22544⟩⟩) 1 ⟨22543⟩ 95509

def event95512 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22544⟩⟩) (.product (.predecessor 0 95510 .coefficient) (.predecessor 1 95511 .coefficient) (⟨false, false, none, none, none⟩))

def event95513 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22544⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22541⟩⟩]⟩) [⟨.result 95505 .coefficient, false, none⟩])

def event95514 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22544⟩⟩) (.product (.result 94462 .summary) (.transfer 95513) (⟨false, false, none, none, none⟩))

def event95515 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22544⟩⟩, .operator (⟨94462, 0⟩, ⟨95509, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22541⟩⟩]⟩, (1)⟩)

def event95516 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22542⟩⟩)

def event95517 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event95518 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event95519 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event95520 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event95521 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 95520

def event95522 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 95518

def event95523 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 95521 .coefficient) (.value (.predecessor 1 95522 .coefficient)))

def event95524 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event95525 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12934⟩⟩) 0 ⟨5503⟩ 95524

def event95526 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12934⟩⟩) (.authority (.programFamilyFact))

def exact95527RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12934⟩⟩], []⟩, (1)⟩]

theorem exact95527RawTermsValid :
    exact95527RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95527 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12934⟩⟩) exact95527RawTerms (.finite 52) 95526 .exactZero (none)

def event95528 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10120⟩⟩) 0 ⟨5503⟩ 95524

def event95529 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10120⟩⟩) (.authority (.programFamilyFact))

def exact95530RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10120⟩⟩], []⟩, (1)⟩]

theorem exact95530RawTermsValid :
    exact95530RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95530 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10120⟩⟩) exact95530RawTerms (.finite 52) 95529 .exactZero (none)

def event95531 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12935⟩⟩) 0 ⟨10120⟩ 95530

def event95532 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12935⟩⟩) 1 ⟨12934⟩ 95527

def event95533 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12935⟩⟩) (.product (.predecessor 0 95531 .coefficient) (.predecessor 1 95532 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event95534 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12935⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], []⟩) [⟨.result 95530 .coefficient, true, some 1⟩, ⟨.result 95527 .coefficient, true, some 1⟩])

def event95535 : Event := .survivorFold (1) 95534

def exact95536RawTerms : List Term := []

theorem exact95536RawTermsValid :
    exact95536RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95536 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12935⟩⟩) exact95536RawTerms (.finite 2704) 95533 (.finite 2704) (some (95534))

def event95537 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12936⟩⟩) 0 ⟨12935⟩ 95536

def event95538 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12936⟩⟩) (.identity (.predecessor 0 95537 .coefficient))

def event95539 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12936⟩⟩) (.finite 2704)

def event95540 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16742⟩⟩) 0 ⟨12936⟩ 95539

def event95541 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16742⟩⟩) (.authority (.programFamilyFact))

def exact95542RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16742⟩⟩], []⟩, (1)⟩]

theorem exact95542RawTermsValid :
    exact95542RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95542 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16742⟩⟩) exact95542RawTerms (.finite 52) 95541 .exactZero (none)

def event95543 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16743⟩⟩) 0 ⟨16742⟩ 95542

def event95544 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16743⟩⟩) (.identity (.predecessor 0 95543 .coefficient))

def event95545 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16743⟩⟩) (.finite 52)

def event95546 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22541⟩⟩) 0 ⟨16743⟩ 95545

def event95547 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22541⟩⟩) (.authority (.relationPreimageSource ⟨61⟩))

def exact95548RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22541⟩⟩]⟩, (1)⟩]

theorem exact95548RawTermsValid :
    exact95548RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95548 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22541⟩⟩) exact95548RawTerms (.finite 136065468) 95547 .exactZero (none)

def event95549 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact95550RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact95550RawTermsValid :
    exact95550RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95550 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact95550RawTerms .large 95549 .exactZero (none)

def event95551 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22542⟩⟩) 0 ⟨6⟩ 95550

def event95552 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22542⟩⟩) 1 ⟨22541⟩ 95548

def event95553 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22542⟩⟩) (.product (.predecessor 0 95551 .coefficient) (.predecessor 1 95552 .coefficient) (⟨false, false, none, none, none⟩))

def event95554 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22542⟩⟩, .operator (⟨95550, 0⟩, ⟨95548, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22541⟩⟩]⟩, (1)⟩)

def exact95555RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22541⟩⟩]⟩, (1)⟩]

theorem exact95555RawTermsValid :
    exact95555RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95555 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22542⟩⟩) exact95555RawTerms .large 95553 .exactZero (none)

def event95556 : Event := .preFoldPolynomial 95555 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22541⟩⟩]⟩, (1)⟩] .exactZero none

def exact95557RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22541⟩⟩]⟩, (1)⟩]

def event95557 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨22542⟩⟩) 95556 exact95557RawTerms .large 95553 .exactZero (none)

def event95558 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨29572⟩⟩)

def event95559 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event95560 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event95561 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event95562 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event95563 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 95562

def event95564 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 95560

def event95565 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 95563 .coefficient) (.value (.predecessor 1 95564 .coefficient)))

def event95566 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event95567 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12934⟩⟩) 0 ⟨5503⟩ 95566

def event95568 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12934⟩⟩) (.authority (.programFamilyFact))

def exact95569RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12934⟩⟩], []⟩, (1)⟩]

theorem exact95569RawTermsValid :
    exact95569RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95569 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12934⟩⟩) exact95569RawTerms (.finite 52) 95568 .exactZero (none)

def event95570 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10120⟩⟩) 0 ⟨5503⟩ 95566

def event95571 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10120⟩⟩) (.authority (.programFamilyFact))

def exact95572RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10120⟩⟩], []⟩, (1)⟩]

theorem exact95572RawTermsValid :
    exact95572RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95572 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10120⟩⟩) exact95572RawTerms (.finite 52) 95571 .exactZero (none)

def event95573 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12935⟩⟩) 0 ⟨10120⟩ 95572

def event95574 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12935⟩⟩) 1 ⟨12934⟩ 95569

def event95575 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12935⟩⟩) (.product (.predecessor 0 95573 .coefficient) (.predecessor 1 95574 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event95576 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12935⟩⟩, .operator (⟨95572, 0⟩, ⟨95569, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], []⟩, (1)⟩)

def exact95577RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], []⟩, (1)⟩]

theorem exact95577RawTermsValid :
    exact95577RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95577 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12935⟩⟩) exact95577RawTerms (.finite 2704) 95575 .exactZero (none)

def event95578 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12936⟩⟩) 0 ⟨12935⟩ 95577

def event95579 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12936⟩⟩) (.identity (.predecessor 0 95578 .coefficient))

def event95580 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12936⟩⟩) (.finite 2704)

def event95581 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16742⟩⟩) 0 ⟨12936⟩ 95580

def event95582 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16742⟩⟩) (.authority (.programFamilyFact))

def exact95583RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16742⟩⟩], []⟩, (1)⟩]

theorem exact95583RawTermsValid :
    exact95583RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95583 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16742⟩⟩) exact95583RawTerms (.finite 52) 95582 .exactZero (none)

def event95584 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16743⟩⟩) 0 ⟨16742⟩ 95583

def event95585 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16743⟩⟩) (.identity (.predecessor 0 95584 .coefficient))

def event95586 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16743⟩⟩) (.finite 52)

def event95587 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24655⟩⟩) 0 ⟨16743⟩ 95586

def event95588 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24655⟩⟩) (.authority (.programFamilyFact))

def event95589 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24655⟩⟩) (.finite 3720)

def event95590 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event95591 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24657⟩⟩) 0 ⟨6689⟩ 95590

def event95592 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24657⟩⟩) 1 ⟨24655⟩ 95589

def event95593 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24657⟩⟩) (.authority (.operator))

def exact95594RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24657⟩⟩]⟩, (1)⟩]

theorem exact95594RawTermsValid :
    exact95594RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95594 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24657⟩⟩) exact95594RawTerms .large 95593 .exactZero (none)

def event95595 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29567⟩⟩) 0 ⟨24657⟩ 95594

def event95596 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29567⟩⟩) (.authority (.operator))

def exact95597RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩, (1)⟩]

theorem exact95597RawTermsValid :
    exact95597RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95597 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29567⟩⟩) exact95597RawTerms (.finite 8192) 95596 .exactZero (none)

def event95598 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event95599 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event95600 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16819⟩⟩) 0 ⟨16743⟩ 95586

def event95601 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16819⟩⟩) 1 ⟨110⟩ 95599

def event95602 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16819⟩⟩) (.sum [.predecessor 0 95600 .coefficient, .predecessor 1 95601 .coefficient])

def event95603 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16819⟩⟩) (.finite 52)

def event95604 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16820⟩⟩) 0 ⟨16819⟩ 95603

def event95605 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16820⟩⟩) (.identity (.predecessor 0 95604 .coefficient))

def exact95606RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16742⟩⟩], []⟩, (1)⟩]

theorem exact95606RawTermsValid :
    exact95606RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95606 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16820⟩⟩) exact95606RawTerms (.finite 52) 95605 .exactZero (none)

def event95607 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact95608RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact95608RawTermsValid :
    exact95608RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95608 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact95608RawTerms .large 95607 .exactZero (none)

def event95609 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16821⟩⟩) 0 ⟨6544⟩ 95608

def event95610 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16821⟩⟩) 1 ⟨16820⟩ 95606

def event95611 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16821⟩⟩) (.product (.predecessor 0 95609 .coefficient) (.predecessor 1 95610 .coefficient) (⟨false, false, none, none, none⟩))

def event95612 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16821⟩⟩, .operator (⟨95608, 0⟩, ⟨95606, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact95613RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact95613RawTermsValid :
    exact95613RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95613 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16821⟩⟩) exact95613RawTerms .large 95611 .exactZero (none)

def event95614 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6705⟩⟩) 0 ⟨6689⟩ 95590

def event95615 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6705⟩⟩) (.authority (.operator))

def exact95616RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩, (1)⟩]

theorem exact95616RawTermsValid :
    exact95616RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95616 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6705⟩⟩) exact95616RawTerms .large 95615 .exactZero (none)

def event95617 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16822⟩⟩) 0 ⟨6705⟩ 95616

def event95618 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16822⟩⟩) 1 ⟨16821⟩ 95613

def event95619 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16822⟩⟩) (.sum [.predecessor 0 95617 .coefficient, .predecessor 1 95618 .coefficient])

def exact95620RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact95620RawTermsValid :
    exact95620RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95620 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16822⟩⟩) exact95620RawTerms .large 95619 .exactZero (none)

def event95621 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29568⟩⟩) 0 ⟨16822⟩ 95620

def event95622 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29568⟩⟩) 1 ⟨29567⟩ 95597

def event95623 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29568⟩⟩) (.product (.predecessor 0 95621 .coefficient) (.predecessor 1 95622 .coefficient) (⟨false, false, none, none, none⟩))

def event95624 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29568⟩⟩, .operator (⟨95620, 0⟩, ⟨95597, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩, (1)⟩)

def event95625 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29568⟩⟩, .operator (⟨95620, 1⟩, ⟨95597, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩, (-1)⟩)

def event95626 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29568⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29567⟩⟩) ⟨24657⟩ 95594)

def event95627 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29568⟩⟩, .relation 95626 0, ⟨[⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨24657⟩⟩]⟩, (-1)⟩)

def exact95628RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨24657⟩⟩]⟩, (-1)⟩]

theorem exact95628RawTermsValid :
    exact95628RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95628 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29568⟩⟩) exact95628RawTerms .large 95623 .exactZero (none)

def event95629 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16791⟩⟩) 0 ⟨16743⟩ 95586

def event95630 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16791⟩⟩) (.authority (.programFamilyFact))

def exact95631RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], []⟩, (1)⟩]

theorem exact95631RawTermsValid :
    exact95631RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95631 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16791⟩⟩) exact95631RawTerms (.finite 63) 95630 .exactZero (none)

def event95632 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16792⟩⟩) 0 ⟨6544⟩ 95608

def event95633 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16792⟩⟩) 1 ⟨16791⟩ 95631

def event95634 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16792⟩⟩) (.product (.predecessor 0 95632 .coefficient) (.predecessor 1 95633 .coefficient) (⟨false, true, none, none, some 1⟩))

def event95635 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16792⟩⟩, .operator (⟨95608, 0⟩, ⟨95631, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact95636RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact95636RawTermsValid :
    exact95636RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95636 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16792⟩⟩) exact95636RawTerms .large 95634 .exactZero (none)

def event95637 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6739⟩⟩) 0 ⟨6689⟩ 95590

def event95638 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6739⟩⟩) (.authority (.operator))

def exact95639RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩]

theorem exact95639RawTermsValid :
    exact95639RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95639 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6739⟩⟩) exact95639RawTerms .large 95638 .exactZero (none)

def event95640 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16793⟩⟩) 0 ⟨6739⟩ 95639

def event95641 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16793⟩⟩) 1 ⟨16792⟩ 95636

def event95642 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16793⟩⟩) (.sum [.predecessor 0 95640 .coefficient, .predecessor 1 95641 .coefficient])

def exact95643RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact95643RawTermsValid :
    exact95643RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95643 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16793⟩⟩) exact95643RawTerms .large 95642 .exactZero (none)

def event95644 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29572⟩⟩) 0 ⟨16793⟩ 95643

def event95645 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29572⟩⟩) 1 ⟨29568⟩ 95628

def event95646 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29572⟩⟩) (.sum [.predecessor 0 95644 .coefficient, .predecessor 1 95645 .coefficient])

def exact95647RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨24657⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact95647RawTermsValid :
    exact95647RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95647 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29572⟩⟩) exact95647RawTerms .large 95646 .exactZero (none)

def event95648 : Event := .preFoldPolynomial 95647 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨24657⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact95649RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨24657⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event95649 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨29572⟩⟩) 95648 exact95649RawTerms .large 95646 .exactZero (none)

def event95650 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16743⟩⟩) ⟨⟨152⟩, ⟨61⟩, ⟨109⟩⟩ ⟨95516, 95650⟩

def event95651 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨22544⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22541⟩⟩]⟩) (1) 0 2 (.universal 95650 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22541⟩⟩]⟩) (none) 95649)

def event95652 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22544⟩⟩, .relation 95651 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩)

def event95653 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22544⟩⟩, .relation 95651 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩, (-1)⟩)

def event95654 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22544⟩⟩, .relation 95651 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨24657⟩⟩]⟩, (1)⟩)

def event95655 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22544⟩⟩, .relation 95651 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact95656RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨24657⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact95656RawTermsValid :
    exact95656RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95656 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22544⟩⟩) exact95656RawTerms .large 95512 (.finite 1811303510016) (some (95514))

def event95657 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29570⟩⟩) 0 ⟨22544⟩ 95656

def event95658 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29570⟩⟩) 1 ⟨29569⟩ 95502

def event95659 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29570⟩⟩) (.sum [.predecessor 0 95657 .coefficient, .predecessor 1 95658 .coefficient])

def event95660 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29570⟩⟩, .operator (⟨95656, 0⟩, ⟨95502, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩, (1)⟩)

def event95661 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29570⟩⟩, .operator (⟨95656, 2⟩, ⟨95502, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16742⟩⟩], [⟨.program ⟨214⟩, ⟨24657⟩⟩]⟩, (-1)⟩)

def event95662 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29570⟩⟩) (.sum [.result 95656 .summary, .result 95502 .summary])

def exact95663RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact95663RawTermsValid :
    exact95663RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95663 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29570⟩⟩) exact95663RawTerms .large 95659 (.finite 1292449485504936292352) (some (95662))

def event95664 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24592⟩⟩) 0 ⟨16624⟩ 4652

def event95665 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24592⟩⟩) (.authority (.programFamilyFact))

def event95666 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24592⟩⟩) (.finite 3720)

def event95667 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24594⟩⟩) 0 ⟨6689⟩ 5477

def event95668 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24594⟩⟩) 1 ⟨24592⟩ 95666

def event95669 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24594⟩⟩) (.authority (.operator))

def exact95670RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24594⟩⟩]⟩, (1)⟩]

theorem exact95670RawTermsValid :
    exact95670RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95670 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24594⟩⟩) exact95670RawTerms .large 95669 .exactZero (none)

def event95671 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29350⟩⟩) 0 ⟨24594⟩ 95670

def event95672 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29350⟩⟩) (.authority (.operator))

def exact95673RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29350⟩⟩]⟩, (1)⟩]

theorem exact95673RawTermsValid :
    exact95673RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95673 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29350⟩⟩) exact95673RawTerms (.finite 8192) 95672 .exactZero (none)

def event95674 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23283⟩⟩) 0 ⟨12740⟩ 4646

def event95675 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23283⟩⟩) (.authority (.programFamilyFact))

def event95676 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23283⟩⟩) (.finite 3720)

def event95677 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23284⟩⟩) 0 ⟨6689⟩ 5477

def event95678 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23284⟩⟩) 1 ⟨23283⟩ 95676

def event95679 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23284⟩⟩) (.authority (.operator))

def exact95680RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23284⟩⟩]⟩, (1)⟩]

theorem exact95680RawTermsValid :
    exact95680RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95680 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23284⟩⟩) exact95680RawTerms .large 95679 .exactZero (none)

def event95681 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25514⟩⟩) 0 ⟨23284⟩ 95680

def event95682 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25514⟩⟩) (.authority (.operator))

def exact95683RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25514⟩⟩]⟩, (1)⟩]

theorem exact95683RawTermsValid :
    exact95683RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95683 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25514⟩⟩) exact95683RawTerms (.finite 8192) 95682 .exactZero (none)

def event95684 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12741⟩⟩) 0 ⟨12738⟩ 4635

def event95685 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12741⟩⟩) 1 ⟨6564⟩ 32

def event95686 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12741⟩⟩) (.tensor (.predecessor 0 95684 .coefficient) (.predecessor 1 95685 .coefficient) true false)

def event95687 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12741⟩⟩, .operator (⟨4635, 0⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact95688RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact95688RawTermsValid :
    exact95688RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95688 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12741⟩⟩) exact95688RawTerms .large 95686 .exactZero (none)

def event95689 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7124⟩⟩) 0 ⟨5506⟩ 27

def event95690 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7124⟩⟩) 1 ⟨6787⟩ 7975

def event95691 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7124⟩⟩) (.product (.predecessor 0 95689 .coefficient) (.predecessor 1 95690 .coefficient) (⟨false, false, none, none, none⟩))

def event95692 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7124⟩⟩, .operator (⟨27, 0⟩, ⟨7975, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (1)⟩)

def exact95693RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (1)⟩]

theorem exact95693RawTermsValid :
    exact95693RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95693 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7124⟩⟩) exact95693RawTerms .large 95691 .exactZero (none)

def event95694 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12742⟩⟩) 0 ⟨7124⟩ 95693

def event95695 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12742⟩⟩) 1 ⟨12741⟩ 95688

def event95696 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12742⟩⟩) (.sum [.predecessor 0 95694 .coefficient, .predecessor 1 95695 .coefficient])

def exact95697RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact95697RawTermsValid :
    exact95697RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95697 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12742⟩⟩) exact95697RawTerms .large 95696 .exactZero (none)

def event95698 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12743⟩⟩) 0 ⟨12742⟩ 95697

def event95699 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12743⟩⟩) 1 ⟨101⟩ 7967

def event95700 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12743⟩⟩) (.sum [.predecessor 0 95698 .coefficient, .predecessor 1 95699 .coefficient])

def event95701 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12743⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨101⟩⟩]⟩) [⟨.result 7967 .coefficient, false, none⟩])

def event95702 : Event := .survivorFold (1) 95701

def exact95703RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact95703RawTermsValid :
    exact95703RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95703 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12743⟩⟩) exact95703RawTerms .large 95700 (.finite 26) (some (95701))

def event95704 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12744⟩⟩) 0 ⟨12743⟩ 95703

def event95705 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12744⟩⟩) 1 ⟨10015⟩ 4638

def event95706 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12744⟩⟩) (.product (.predecessor 0 95704 .coefficient) (.predecessor 1 95705 .coefficient) (⟨false, true, none, none, some 1⟩))

def event95707 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12744⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10015⟩⟩], []⟩) [⟨.result 4638 .coefficient, true, some 1⟩])

def event95708 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12744⟩⟩) (.product (.result 95703 .summary) (.transfer 95707) (⟨false, false, none, none, none⟩))

def event95709 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12744⟩⟩, .operator (⟨95703, 1⟩, ⟨4638, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event95710 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12744⟩⟩, .operator (⟨95703, 0⟩, ⟨4638, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (1)⟩)

def exact95711RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact95711RawTermsValid :
    exact95711RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95711 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12744⟩⟩) exact95711RawTerms .large 95706 (.finite 38272) (some (95708))

def event95712 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10016⟩⟩) 0 ⟨10015⟩ 4638

def event95713 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10016⟩⟩) 1 ⟨6564⟩ 32

def event95714 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10016⟩⟩) (.tensor (.predecessor 0 95712 .coefficient) (.predecessor 1 95713 .coefficient) true false)

def event95715 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10016⟩⟩, .operator (⟨4638, 0⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact95716RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact95716RawTermsValid :
    exact95716RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95716 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10016⟩⟩) exact95716RawTerms .large 95714 .exactZero (none)

def event95717 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7104⟩⟩) 0 ⟨5506⟩ 27

def event95718 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7104⟩⟩) 1 ⟨6767⟩ 8016

def event95719 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7104⟩⟩) (.product (.predecessor 0 95717 .coefficient) (.predecessor 1 95718 .coefficient) (⟨false, false, none, none, none⟩))

def event95720 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7104⟩⟩, .operator (⟨27, 0⟩, ⟨8016, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩]⟩, (1)⟩)

def exact95721RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩]⟩, (1)⟩]

theorem exact95721RawTermsValid :
    exact95721RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95721 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7104⟩⟩) exact95721RawTerms .large 95719 .exactZero (none)

def event95722 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10017⟩⟩) 0 ⟨7104⟩ 95721

def event95723 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10017⟩⟩) 1 ⟨10016⟩ 95716

def event95724 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10017⟩⟩) (.sum [.predecessor 0 95722 .coefficient, .predecessor 1 95723 .coefficient])

def exact95725RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact95725RawTermsValid :
    exact95725RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95725 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10017⟩⟩) exact95725RawTerms .large 95724 .exactZero (none)

def event95726 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10018⟩⟩) 0 ⟨10017⟩ 95725

def event95727 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10018⟩⟩) 1 ⟨81⟩ 8008

def event95728 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10018⟩⟩) (.sum [.predecessor 0 95726 .coefficient, .predecessor 1 95727 .coefficient])

def event95729 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10018⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨81⟩⟩]⟩) [⟨.result 8008 .coefficient, false, none⟩])

def event95730 : Event := .survivorFold (1) 95729

def exact95731RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact95731RawTermsValid :
    exact95731RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95731 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10018⟩⟩) exact95731RawTerms .large 95728 (.finite 26) (some (95729))

def event95732 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10019⟩⟩) 0 ⟨10018⟩ 95731

def event95733 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10019⟩⟩) 1 ⟨7874⟩ 8005

def event95734 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10019⟩⟩) (.product (.predecessor 0 95732 .coefficient) (.predecessor 1 95733 .coefficient) (⟨false, false, none, none, none⟩))

def event95735 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10019⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩) [⟨.result 8001 .coefficient, false, none⟩])

def event95736 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10019⟩⟩) (.product (.result 95731 .summary) (.transfer 95735) (⟨false, false, none, none, none⟩))

def event95737 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10019⟩⟩, .operator (⟨95731, 1⟩, ⟨8005, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩, (-1)⟩)

def event95738 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨10019⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7873⟩⟩) ⟨6787⟩ 7975)

def event95739 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10019⟩⟩, .relation 95738 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (-1)⟩)

def event95740 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10019⟩⟩, .operator (⟨95731, 0⟩, ⟨8005, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩, (1)⟩)

def exact95741RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (-1)⟩]

theorem exact95741RawTermsValid :
    exact95741RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event95741 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10019⟩⟩) exact95741RawTerms .large 95734 (.finite 95420416) (some (95736))

def event95742 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12745⟩⟩) 0 ⟨10019⟩ 95741

def event95743 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12745⟩⟩) 1 ⟨12744⟩ 95711

def eventLeaf5968 : Array AnnotatedEvent := #[
  { event := event95488
    frameStart := 0 },
  { event := event95489
    frameStart := 0 },
  { event := event95490
    frameStart := 0 },
  { event := event95491
    frameStart := 0 },
  { event := event95492
    frameStart := 0 },
  { event := event95493
    frameStart := 0 },
  { event := event95494
    frameStart := 0 },
  { event := event95495
    frameStart := 0 },
  { event := event95496
    frameStart := 0 },
  { event := event95497
    frameStart := 0 },
  { event := event95498
    frameStart := 0 },
  { event := event95499
    frameStart := 0 },
  { event := event95500
    frameStart := 0 },
  { event := event95501
    frameStart := 0 },
  { event := event95502
    frameStart := 0 },
  { event := event95503
    frameStart := 0 }
]

def eventLeaf5969 : Array AnnotatedEvent := #[
  { event := event95504
    frameStart := 0 },
  { event := event95505
    frameStart := 0 },
  { event := event95506
    frameStart := 0 },
  { event := event95507
    frameStart := 0 },
  { event := event95508
    frameStart := 0 },
  { event := event95509
    frameStart := 0 },
  { event := event95510
    frameStart := 0 },
  { event := event95511
    frameStart := 0 },
  { event := event95512
    frameStart := 0 },
  { event := event95513
    frameStart := 0 },
  { event := event95514
    frameStart := 0 },
  { event := event95515
    frameStart := 0 },
  { event := event95516
    frameStart := 95516 },
  { event := event95517
    frameStart := 95516 },
  { event := event95518
    frameStart := 95516 },
  { event := event95519
    frameStart := 95516 }
]

def eventLeaf5970 : Array AnnotatedEvent := #[
  { event := event95520
    frameStart := 95516 },
  { event := event95521
    frameStart := 95516 },
  { event := event95522
    frameStart := 95516 },
  { event := event95523
    frameStart := 95516 },
  { event := event95524
    frameStart := 95516 },
  { event := event95525
    frameStart := 95516 },
  { event := event95526
    frameStart := 95516 },
  { event := event95527
    frameStart := 95516 },
  { event := event95528
    frameStart := 95516 },
  { event := event95529
    frameStart := 95516 },
  { event := event95530
    frameStart := 95516 },
  { event := event95531
    frameStart := 95516 },
  { event := event95532
    frameStart := 95516 },
  { event := event95533
    frameStart := 95516 },
  { event := event95534
    frameStart := 95516 },
  { event := event95535
    frameStart := 95516 }
]

def eventLeaf5971 : Array AnnotatedEvent := #[
  { event := event95536
    frameStart := 95516 },
  { event := event95537
    frameStart := 95516 },
  { event := event95538
    frameStart := 95516 },
  { event := event95539
    frameStart := 95516 },
  { event := event95540
    frameStart := 95516 },
  { event := event95541
    frameStart := 95516 },
  { event := event95542
    frameStart := 95516 },
  { event := event95543
    frameStart := 95516 },
  { event := event95544
    frameStart := 95516 },
  { event := event95545
    frameStart := 95516 },
  { event := event95546
    frameStart := 95516 },
  { event := event95547
    frameStart := 95516 },
  { event := event95548
    frameStart := 95516 },
  { event := event95549
    frameStart := 95516 },
  { event := event95550
    frameStart := 95516 },
  { event := event95551
    frameStart := 95516 }
]

def eventLeaf5972 : Array AnnotatedEvent := #[
  { event := event95552
    frameStart := 95516 },
  { event := event95553
    frameStart := 95516 },
  { event := event95554
    frameStart := 95516 },
  { event := event95555
    frameStart := 95516 },
  { event := event95556
    frameStart := 95516 },
  { event := event95557
    frameStart := 95516 },
  { event := event95558
    frameStart := 95558 },
  { event := event95559
    frameStart := 95558 },
  { event := event95560
    frameStart := 95558 },
  { event := event95561
    frameStart := 95558 },
  { event := event95562
    frameStart := 95558 },
  { event := event95563
    frameStart := 95558 },
  { event := event95564
    frameStart := 95558 },
  { event := event95565
    frameStart := 95558 },
  { event := event95566
    frameStart := 95558 },
  { event := event95567
    frameStart := 95558 }
]

def eventLeaf5973 : Array AnnotatedEvent := #[
  { event := event95568
    frameStart := 95558 },
  { event := event95569
    frameStart := 95558 },
  { event := event95570
    frameStart := 95558 },
  { event := event95571
    frameStart := 95558 },
  { event := event95572
    frameStart := 95558 },
  { event := event95573
    frameStart := 95558 },
  { event := event95574
    frameStart := 95558 },
  { event := event95575
    frameStart := 95558 },
  { event := event95576
    frameStart := 95558 },
  { event := event95577
    frameStart := 95558 },
  { event := event95578
    frameStart := 95558 },
  { event := event95579
    frameStart := 95558 },
  { event := event95580
    frameStart := 95558 },
  { event := event95581
    frameStart := 95558 },
  { event := event95582
    frameStart := 95558 },
  { event := event95583
    frameStart := 95558 }
]

def eventLeaf5974 : Array AnnotatedEvent := #[
  { event := event95584
    frameStart := 95558 },
  { event := event95585
    frameStart := 95558 },
  { event := event95586
    frameStart := 95558 },
  { event := event95587
    frameStart := 95558 },
  { event := event95588
    frameStart := 95558 },
  { event := event95589
    frameStart := 95558 },
  { event := event95590
    frameStart := 95558 },
  { event := event95591
    frameStart := 95558 },
  { event := event95592
    frameStart := 95558 },
  { event := event95593
    frameStart := 95558 },
  { event := event95594
    frameStart := 95558 },
  { event := event95595
    frameStart := 95558 },
  { event := event95596
    frameStart := 95558 },
  { event := event95597
    frameStart := 95558 },
  { event := event95598
    frameStart := 95558 },
  { event := event95599
    frameStart := 95558 }
]

def eventLeaf5975 : Array AnnotatedEvent := #[
  { event := event95600
    frameStart := 95558 },
  { event := event95601
    frameStart := 95558 },
  { event := event95602
    frameStart := 95558 },
  { event := event95603
    frameStart := 95558 },
  { event := event95604
    frameStart := 95558 },
  { event := event95605
    frameStart := 95558 },
  { event := event95606
    frameStart := 95558 },
  { event := event95607
    frameStart := 95558 },
  { event := event95608
    frameStart := 95558 },
  { event := event95609
    frameStart := 95558 },
  { event := event95610
    frameStart := 95558 },
  { event := event95611
    frameStart := 95558 },
  { event := event95612
    frameStart := 95558 },
  { event := event95613
    frameStart := 95558 },
  { event := event95614
    frameStart := 95558 },
  { event := event95615
    frameStart := 95558 }
]

def eventLeaf5976 : Array AnnotatedEvent := #[
  { event := event95616
    frameStart := 95558 },
  { event := event95617
    frameStart := 95558 },
  { event := event95618
    frameStart := 95558 },
  { event := event95619
    frameStart := 95558 },
  { event := event95620
    frameStart := 95558 },
  { event := event95621
    frameStart := 95558 },
  { event := event95622
    frameStart := 95558 },
  { event := event95623
    frameStart := 95558 },
  { event := event95624
    frameStart := 95558 },
  { event := event95625
    frameStart := 95558 },
  { event := event95626
    frameStart := 95558 },
  { event := event95627
    frameStart := 95558 },
  { event := event95628
    frameStart := 95558 },
  { event := event95629
    frameStart := 95558 },
  { event := event95630
    frameStart := 95558 },
  { event := event95631
    frameStart := 95558 }
]

def eventLeaf5977 : Array AnnotatedEvent := #[
  { event := event95632
    frameStart := 95558 },
  { event := event95633
    frameStart := 95558 },
  { event := event95634
    frameStart := 95558 },
  { event := event95635
    frameStart := 95558 },
  { event := event95636
    frameStart := 95558 },
  { event := event95637
    frameStart := 95558 },
  { event := event95638
    frameStart := 95558 },
  { event := event95639
    frameStart := 95558 },
  { event := event95640
    frameStart := 95558 },
  { event := event95641
    frameStart := 95558 },
  { event := event95642
    frameStart := 95558 },
  { event := event95643
    frameStart := 95558 },
  { event := event95644
    frameStart := 95558 },
  { event := event95645
    frameStart := 95558 },
  { event := event95646
    frameStart := 95558 },
  { event := event95647
    frameStart := 95558 }
]

def eventLeaf5978 : Array AnnotatedEvent := #[
  { event := event95648
    frameStart := 95558 },
  { event := event95649
    frameStart := 95558 },
  { event := event95650
    frameStart := 0 },
  { event := event95651
    frameStart := 0 },
  { event := event95652
    frameStart := 0 },
  { event := event95653
    frameStart := 0 },
  { event := event95654
    frameStart := 0 },
  { event := event95655
    frameStart := 0 },
  { event := event95656
    frameStart := 0 },
  { event := event95657
    frameStart := 0 },
  { event := event95658
    frameStart := 0 },
  { event := event95659
    frameStart := 0 },
  { event := event95660
    frameStart := 0 },
  { event := event95661
    frameStart := 0 },
  { event := event95662
    frameStart := 0 },
  { event := event95663
    frameStart := 0 }
]

def eventLeaf5979 : Array AnnotatedEvent := #[
  { event := event95664
    frameStart := 0 },
  { event := event95665
    frameStart := 0 },
  { event := event95666
    frameStart := 0 },
  { event := event95667
    frameStart := 0 },
  { event := event95668
    frameStart := 0 },
  { event := event95669
    frameStart := 0 },
  { event := event95670
    frameStart := 0 },
  { event := event95671
    frameStart := 0 },
  { event := event95672
    frameStart := 0 },
  { event := event95673
    frameStart := 0 },
  { event := event95674
    frameStart := 0 },
  { event := event95675
    frameStart := 0 },
  { event := event95676
    frameStart := 0 },
  { event := event95677
    frameStart := 0 },
  { event := event95678
    frameStart := 0 },
  { event := event95679
    frameStart := 0 }
]

def eventLeaf5980 : Array AnnotatedEvent := #[
  { event := event95680
    frameStart := 0 },
  { event := event95681
    frameStart := 0 },
  { event := event95682
    frameStart := 0 },
  { event := event95683
    frameStart := 0 },
  { event := event95684
    frameStart := 0 },
  { event := event95685
    frameStart := 0 },
  { event := event95686
    frameStart := 0 },
  { event := event95687
    frameStart := 0 },
  { event := event95688
    frameStart := 0 },
  { event := event95689
    frameStart := 0 },
  { event := event95690
    frameStart := 0 },
  { event := event95691
    frameStart := 0 },
  { event := event95692
    frameStart := 0 },
  { event := event95693
    frameStart := 0 },
  { event := event95694
    frameStart := 0 },
  { event := event95695
    frameStart := 0 }
]

def eventLeaf5981 : Array AnnotatedEvent := #[
  { event := event95696
    frameStart := 0 },
  { event := event95697
    frameStart := 0 },
  { event := event95698
    frameStart := 0 },
  { event := event95699
    frameStart := 0 },
  { event := event95700
    frameStart := 0 },
  { event := event95701
    frameStart := 0 },
  { event := event95702
    frameStart := 0 },
  { event := event95703
    frameStart := 0 },
  { event := event95704
    frameStart := 0 },
  { event := event95705
    frameStart := 0 },
  { event := event95706
    frameStart := 0 },
  { event := event95707
    frameStart := 0 },
  { event := event95708
    frameStart := 0 },
  { event := event95709
    frameStart := 0 },
  { event := event95710
    frameStart := 0 },
  { event := event95711
    frameStart := 0 }
]

def eventLeaf5982 : Array AnnotatedEvent := #[
  { event := event95712
    frameStart := 0 },
  { event := event95713
    frameStart := 0 },
  { event := event95714
    frameStart := 0 },
  { event := event95715
    frameStart := 0 },
  { event := event95716
    frameStart := 0 },
  { event := event95717
    frameStart := 0 },
  { event := event95718
    frameStart := 0 },
  { event := event95719
    frameStart := 0 },
  { event := event95720
    frameStart := 0 },
  { event := event95721
    frameStart := 0 },
  { event := event95722
    frameStart := 0 },
  { event := event95723
    frameStart := 0 },
  { event := event95724
    frameStart := 0 },
  { event := event95725
    frameStart := 0 },
  { event := event95726
    frameStart := 0 },
  { event := event95727
    frameStart := 0 }
]

def eventLeaf5983 : Array AnnotatedEvent := #[
  { event := event95728
    frameStart := 0 },
  { event := event95729
    frameStart := 0 },
  { event := event95730
    frameStart := 0 },
  { event := event95731
    frameStart := 0 },
  { event := event95732
    frameStart := 0 },
  { event := event95733
    frameStart := 0 },
  { event := event95734
    frameStart := 0 },
  { event := event95735
    frameStart := 0 },
  { event := event95736
    frameStart := 0 },
  { event := event95737
    frameStart := 0 },
  { event := event95738
    frameStart := 0 },
  { event := event95739
    frameStart := 0 },
  { event := event95740
    frameStart := 0 },
  { event := event95741
    frameStart := 0 },
  { event := event95742
    frameStart := 0 },
  { event := event95743
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events373
