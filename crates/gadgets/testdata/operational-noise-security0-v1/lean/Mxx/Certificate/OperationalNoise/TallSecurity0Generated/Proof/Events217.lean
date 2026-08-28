import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events217

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event55552 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14223⟩⟩) (.product (.predecessor 0 55550 .coefficient) (.predecessor 1 55551 .coefficient) (⟨false, false, none, none, none⟩))

def event55553 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14223⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩) [⟨.result 11508 .coefficient, false, none⟩])

def event55554 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14223⟩⟩) (.product (.result 55549 .summary) (.transfer 55553) (⟨false, false, none, none, none⟩))

def event55555 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14223⟩⟩, .operator (⟨55549, 1⟩, ⟨11512, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (-1)⟩)

def event55556 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨14223⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7852⟩⟩) ⟨6779⟩ 11482)

def event55557 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14223⟩⟩, .relation 55556 0, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (-1)⟩)

def event55558 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14223⟩⟩, .operator (⟨55549, 0⟩, ⟨11512, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩)

def exact55559RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (-1)⟩]

theorem exact55559RawTermsValid :
    exact55559RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55559 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14223⟩⟩) exact55559RawTerms .large 55552 (.finite 95420416) (some (55554))

def event55560 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14224⟩⟩) 0 ⟨14223⟩ 55559

def event55561 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14224⟩⟩) 1 ⟨14219⟩ 55529

def event55562 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14224⟩⟩) (.sum [.predecessor 0 55560 .coefficient, .predecessor 1 55561 .coefficient])

def event55563 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14224⟩⟩, .operator (⟨55559, 1⟩, ⟨55529, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩)

def event55564 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14224⟩⟩) (.sum [.result 55559 .summary, .result 55529 .summary])

def exact55565RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact55565RawTermsValid :
    exact55565RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55565 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14224⟩⟩) exact55565RawTerms .large 55562 (.finite 95435392) (some (55564))

def event55566 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26072⟩⟩) 0 ⟨14224⟩ 55565

def event55567 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26072⟩⟩) 1 ⟨26071⟩ 55501

def event55568 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26072⟩⟩) (.product (.predecessor 0 55566 .coefficient) (.predecessor 1 55567 .coefficient) (⟨false, false, none, none, none⟩))

def event55569 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26072⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩) [⟨.result 55501 .coefficient, false, none⟩])

def event55570 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26072⟩⟩) (.product (.result 55565 .summary) (.transfer 55569) (⟨false, false, none, none, none⟩))

def event55571 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26072⟩⟩, .operator (⟨55565, 1⟩, ⟨55501, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩, (-1)⟩)

def event55572 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26072⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26071⟩⟩) ⟨23586⟩ 55498)

def event55573 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26072⟩⟩, .relation 55572 0, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨23586⟩⟩]⟩, (-1)⟩)

def event55574 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26072⟩⟩, .operator (⟨55565, 0⟩, ⟨55501, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩, (1)⟩)

def exact55575RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨23586⟩⟩]⟩, (-1)⟩]

theorem exact55575RawTermsValid :
    exact55575RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55575 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26072⟩⟩) exact55575RawTerms .large 55568 (.finite 350249415606272) (some (55570))

def event55576 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19532⟩⟩) 0 ⟨14218⟩ 2579

def event55577 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19532⟩⟩) (.authority (.relationPreimageSource ⟨15⟩))

def exact55578RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19532⟩⟩]⟩, (1)⟩]

theorem exact55578RawTermsValid :
    exact55578RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55578 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19532⟩⟩) exact55578RawTerms (.finite 136065468) 55577 .exactZero (none)

def event55579 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19534⟩⟩) 0 ⟨19532⟩ 55578

def event55580 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19534⟩⟩) 1 ⟨2348⟩ 4

def event55581 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19534⟩⟩) (.scale (.predecessor 0 55579 .coefficient) (.value (.predecessor 1 55580 .coefficient)))

def exact55582RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19532⟩⟩]⟩, (1)⟩]

theorem exact55582RawTermsValid :
    exact55582RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55582 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19534⟩⟩) exact55582RawTerms (.finite 136065468) 55581 .exactZero (none)

def event55583 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19535⟩⟩) 0 ⟨5547⟩ 50762

def event55584 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19535⟩⟩) 1 ⟨19534⟩ 55582

def event55585 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19535⟩⟩) (.product (.predecessor 0 55583 .coefficient) (.predecessor 1 55584 .coefficient) (⟨false, false, none, none, none⟩))

def event55586 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19535⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19532⟩⟩]⟩) [⟨.result 55578 .coefficient, false, none⟩])

def event55587 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19535⟩⟩) (.product (.result 50762 .summary) (.transfer 55586) (⟨false, false, none, none, none⟩))

def event55588 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19535⟩⟩, .operator (⟨50762, 0⟩, ⟨55582, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19532⟩⟩]⟩, (1)⟩)

def event55589 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19533⟩⟩)

def event55590 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event55591 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event55592 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.authority (.operator))

def event55593 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.finite 3)

def event55594 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event55595 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event55596 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event55597 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event55598 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 55597

def event55599 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 55595

def event55600 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 55598 .coefficient) (.value (.predecessor 1 55599 .coefficient)))

def event55601 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event55602 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 0 ⟨5503⟩ 55601

def event55603 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 1 ⟨2875⟩ 55593

def event55604 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.sum [.predecessor 0 55602 .coefficient, .predecessor 1 55603 .coefficient])

def event55605 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.finite 220)

def event55606 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 0 ⟨5510⟩ 55605

def event55607 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 1 ⟨961⟩ 55591

def event55608 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.identity (.predecessor 1 55607 .coefficient))

def event55609 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.finite 224)

def event55610 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11473⟩⟩) 0 ⟨5542⟩ 55609

def event55611 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11473⟩⟩) (.authority (.programFamilyFact))

def exact55612RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11473⟩⟩], []⟩, (1)⟩]

theorem exact55612RawTermsValid :
    exact55612RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55612 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11473⟩⟩) exact55612RawTerms (.finite 18) 55611 .exactZero (none)

def event55613 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14216⟩⟩) 0 ⟨5542⟩ 55609

def event55614 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14216⟩⟩) (.authority (.programFamilyFact))

def exact55615RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14216⟩⟩], []⟩, (1)⟩]

theorem exact55615RawTermsValid :
    exact55615RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55615 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14216⟩⟩) exact55615RawTerms (.finite 18) 55614 .exactZero (none)

def event55616 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14217⟩⟩) 0 ⟨14216⟩ 55615

def event55617 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14217⟩⟩) 1 ⟨11473⟩ 55612

def event55618 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14217⟩⟩) (.product (.predecessor 0 55616 .coefficient) (.predecessor 1 55617 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event55619 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14217⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], []⟩) [⟨.result 55615 .coefficient, true, some 1⟩, ⟨.result 55612 .coefficient, true, some 1⟩])

def event55620 : Event := .survivorFold (1) 55619

def exact55621RawTerms : List Term := []

theorem exact55621RawTermsValid :
    exact55621RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55621 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14217⟩⟩) exact55621RawTerms (.finite 324) 55618 (.finite 324) (some (55619))

def event55622 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14218⟩⟩) 0 ⟨14217⟩ 55621

def event55623 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14218⟩⟩) (.identity (.predecessor 0 55622 .coefficient))

def event55624 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14218⟩⟩) (.finite 324)

def event55625 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19532⟩⟩) 0 ⟨14218⟩ 55624

def event55626 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19532⟩⟩) (.authority (.relationPreimageSource ⟨15⟩))

def exact55627RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19532⟩⟩]⟩, (1)⟩]

theorem exact55627RawTermsValid :
    exact55627RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55627 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19532⟩⟩) exact55627RawTerms (.finite 136065468) 55626 .exactZero (none)

def event55628 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact55629RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact55629RawTermsValid :
    exact55629RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55629 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact55629RawTerms .large 55628 .exactZero (none)

def event55630 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19533⟩⟩) 0 ⟨6⟩ 55629

def event55631 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19533⟩⟩) 1 ⟨19532⟩ 55627

def event55632 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19533⟩⟩) (.product (.predecessor 0 55630 .coefficient) (.predecessor 1 55631 .coefficient) (⟨false, false, none, none, none⟩))

def event55633 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19533⟩⟩, .operator (⟨55629, 0⟩, ⟨55627, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19532⟩⟩]⟩, (1)⟩)

def exact55634RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19532⟩⟩]⟩, (1)⟩]

theorem exact55634RawTermsValid :
    exact55634RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55634 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19533⟩⟩) exact55634RawTerms .large 55632 .exactZero (none)

def event55635 : Event := .preFoldPolynomial 55634 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19532⟩⟩]⟩, (1)⟩] .exactZero none

def exact55636RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19532⟩⟩]⟩, (1)⟩]

def event55636 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19533⟩⟩) 55635 exact55636RawTerms .large 55632 .exactZero (none)

def event55637 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26075⟩⟩)

def event55638 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event55639 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event55640 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.authority (.operator))

def event55641 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.finite 3)

def event55642 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event55643 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event55644 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event55645 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event55646 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 55645

def event55647 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 55643

def event55648 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 55646 .coefficient) (.value (.predecessor 1 55647 .coefficient)))

def event55649 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event55650 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 0 ⟨5503⟩ 55649

def event55651 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 1 ⟨2875⟩ 55641

def event55652 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.sum [.predecessor 0 55650 .coefficient, .predecessor 1 55651 .coefficient])

def event55653 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.finite 220)

def event55654 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 0 ⟨5510⟩ 55653

def event55655 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 1 ⟨961⟩ 55639

def event55656 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.identity (.predecessor 1 55655 .coefficient))

def event55657 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.finite 224)

def event55658 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11473⟩⟩) 0 ⟨5542⟩ 55657

def event55659 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11473⟩⟩) (.authority (.programFamilyFact))

def exact55660RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11473⟩⟩], []⟩, (1)⟩]

theorem exact55660RawTermsValid :
    exact55660RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55660 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11473⟩⟩) exact55660RawTerms (.finite 18) 55659 .exactZero (none)

def event55661 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14216⟩⟩) 0 ⟨5542⟩ 55657

def event55662 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14216⟩⟩) (.authority (.programFamilyFact))

def exact55663RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14216⟩⟩], []⟩, (1)⟩]

theorem exact55663RawTermsValid :
    exact55663RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55663 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14216⟩⟩) exact55663RawTerms (.finite 18) 55662 .exactZero (none)

def event55664 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14217⟩⟩) 0 ⟨14216⟩ 55663

def event55665 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14217⟩⟩) 1 ⟨11473⟩ 55660

def event55666 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14217⟩⟩) (.product (.predecessor 0 55664 .coefficient) (.predecessor 1 55665 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event55667 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14217⟩⟩, .operator (⟨55663, 0⟩, ⟨55660, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], []⟩, (1)⟩)

def exact55668RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], []⟩, (1)⟩]

theorem exact55668RawTermsValid :
    exact55668RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55668 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14217⟩⟩) exact55668RawTerms (.finite 324) 55666 .exactZero (none)

def event55669 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14218⟩⟩) 0 ⟨14217⟩ 55668

def event55670 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14218⟩⟩) (.identity (.predecessor 0 55669 .coefficient))

def event55671 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14218⟩⟩) (.finite 324)

def event55672 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23585⟩⟩) 0 ⟨14218⟩ 55671

def event55673 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23585⟩⟩) (.authority (.programFamilyFact))

def event55674 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23585⟩⟩) (.finite 3720)

def event55675 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event55676 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23586⟩⟩) 0 ⟨6689⟩ 55675

def event55677 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23586⟩⟩) 1 ⟨23585⟩ 55674

def event55678 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23586⟩⟩) (.authority (.operator))

def exact55679RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23586⟩⟩]⟩, (1)⟩]

theorem exact55679RawTermsValid :
    exact55679RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55679 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23586⟩⟩) exact55679RawTerms .large 55678 .exactZero (none)

def event55680 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26071⟩⟩) 0 ⟨23586⟩ 55679

def event55681 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26071⟩⟩) (.authority (.operator))

def exact55682RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩, (1)⟩]

theorem exact55682RawTermsValid :
    exact55682RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55682 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26071⟩⟩) exact55682RawTerms (.finite 8192) 55681 .exactZero (none)

def event55683 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event55684 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event55685 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14318⟩⟩) 0 ⟨14218⟩ 55671

def event55686 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14318⟩⟩) 1 ⟨110⟩ 55684

def event55687 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14318⟩⟩) (.sum [.predecessor 0 55685 .coefficient, .predecessor 1 55686 .coefficient])

def event55688 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14318⟩⟩) (.finite 324)

def event55689 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14319⟩⟩) 0 ⟨14318⟩ 55688

def event55690 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14319⟩⟩) (.identity (.predecessor 0 55689 .coefficient))

def exact55691RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], []⟩, (1)⟩]

theorem exact55691RawTermsValid :
    exact55691RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55691 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14319⟩⟩) exact55691RawTerms (.finite 324) 55690 .exactZero (none)

def event55692 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact55693RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact55693RawTermsValid :
    exact55693RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55693 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact55693RawTerms .large 55692 .exactZero (none)

def event55694 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14320⟩⟩) 0 ⟨6544⟩ 55693

def event55695 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14320⟩⟩) 1 ⟨14319⟩ 55691

def event55696 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14320⟩⟩) (.product (.predecessor 0 55694 .coefficient) (.predecessor 1 55695 .coefficient) (⟨false, false, none, none, none⟩))

def event55697 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14320⟩⟩, .operator (⟨55693, 0⟩, ⟨55691, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact55698RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact55698RawTermsValid :
    exact55698RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55698 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14320⟩⟩) exact55698RawTerms .large 55696 .exactZero (none)

def event55699 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event55700 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event55701 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 55675

def event55702 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact55703RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact55703RawTermsValid :
    exact55703RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55703 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact55703RawTerms .large 55702 .exactZero (none)

def event55704 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6779⟩⟩) 0 ⟨6757⟩ 55703

def event55705 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6779⟩⟩) (.identity (.predecessor 0 55704 .coefficient))

def exact55706RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩, (1)⟩]

theorem exact55706RawTermsValid :
    exact55706RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55706 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6779⟩⟩) exact55706RawTerms .large 55705 .exactZero (none)

def event55707 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7852⟩⟩) 0 ⟨6779⟩ 55706

def event55708 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7852⟩⟩) (.authority (.operator))

def exact55709RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩]

theorem exact55709RawTermsValid :
    exact55709RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55709 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7852⟩⟩) exact55709RawTerms (.finite 8192) 55708 .exactZero (none)

def event55710 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7853⟩⟩) 0 ⟨7852⟩ 55709

def event55711 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7853⟩⟩) 1 ⟨2348⟩ 55700

def event55712 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7853⟩⟩) (.scale (.predecessor 0 55710 .coefficient) (.value (.predecessor 1 55711 .coefficient)))

def exact55713RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩]

theorem exact55713RawTermsValid :
    exact55713RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55713 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7853⟩⟩) exact55713RawTerms (.finite 8192) 55712 .exactZero (none)

def event55714 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6759⟩⟩) 0 ⟨6757⟩ 55703

def event55715 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6759⟩⟩) (.identity (.predecessor 0 55714 .coefficient))

def exact55716RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩]⟩, (1)⟩]

theorem exact55716RawTermsValid :
    exact55716RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55716 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6759⟩⟩) exact55716RawTerms .large 55715 .exactZero (none)

def event55717 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7854⟩⟩) 0 ⟨6759⟩ 55716

def event55718 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7854⟩⟩) 1 ⟨7853⟩ 55713

def event55719 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7854⟩⟩) (.product (.predecessor 0 55717 .coefficient) (.predecessor 1 55718 .coefficient) (⟨false, false, none, none, none⟩))

def event55720 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7854⟩⟩, .operator (⟨55716, 0⟩, ⟨55713, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩)

def exact55721RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩]

theorem exact55721RawTermsValid :
    exact55721RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55721 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7854⟩⟩) exact55721RawTerms .large 55719 .exactZero (none)

def event55722 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14321⟩⟩) 0 ⟨7854⟩ 55721

def event55723 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14321⟩⟩) 1 ⟨14320⟩ 55698

def event55724 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14321⟩⟩) (.sum [.predecessor 0 55722 .coefficient, .predecessor 1 55723 .coefficient])

def exact55725RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact55725RawTermsValid :
    exact55725RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55725 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14321⟩⟩) exact55725RawTerms .large 55724 .exactZero (none)

def event55726 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26074⟩⟩) 0 ⟨14321⟩ 55725

def event55727 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26074⟩⟩) 1 ⟨26071⟩ 55682

def event55728 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26074⟩⟩) (.product (.predecessor 0 55726 .coefficient) (.predecessor 1 55727 .coefficient) (⟨false, false, none, none, none⟩))

def event55729 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26074⟩⟩, .operator (⟨55725, 0⟩, ⟨55682, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩, (1)⟩)

def event55730 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26074⟩⟩, .operator (⟨55725, 1⟩, ⟨55682, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩, (-1)⟩)

def event55731 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26074⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26071⟩⟩) ⟨23586⟩ 55679)

def event55732 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26074⟩⟩, .relation 55731 0, ⟨[⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨23586⟩⟩]⟩, (-1)⟩)

def exact55733RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨23586⟩⟩]⟩, (-1)⟩]

theorem exact55733RawTermsValid :
    exact55733RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55733 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26074⟩⟩) exact55733RawTerms .large 55728 .exactZero (none)

def event55734 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15944⟩⟩) 0 ⟨14218⟩ 55671

def event55735 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15944⟩⟩) (.authority (.programFamilyFact))

def exact55736RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15944⟩⟩], []⟩, (1)⟩]

theorem exact55736RawTermsValid :
    exact55736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55736 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15944⟩⟩) exact55736RawTerms (.finite 18) 55735 .exactZero (none)

def event55737 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15946⟩⟩) 0 ⟨6544⟩ 55693

def event55738 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15946⟩⟩) 1 ⟨15944⟩ 55736

def event55739 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15946⟩⟩) (.product (.predecessor 0 55737 .coefficient) (.predecessor 1 55738 .coefficient) (⟨false, true, none, none, some 1⟩))

def event55740 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15946⟩⟩, .operator (⟨55693, 0⟩, ⟨55736, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15944⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact55741RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15944⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact55741RawTermsValid :
    exact55741RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55741 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15946⟩⟩) exact55741RawTerms .large 55739 .exactZero (none)

def event55742 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6697⟩⟩) 0 ⟨6689⟩ 55675

def event55743 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6697⟩⟩) (.authority (.operator))

def exact55744RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩]

theorem exact55744RawTermsValid :
    exact55744RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55744 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6697⟩⟩) exact55744RawTerms .large 55743 .exactZero (none)

def event55745 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15947⟩⟩) 0 ⟨6697⟩ 55744

def event55746 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15947⟩⟩) 1 ⟨15946⟩ 55741

def event55747 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15947⟩⟩) (.sum [.predecessor 0 55745 .coefficient, .predecessor 1 55746 .coefficient])

def exact55748RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15944⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact55748RawTermsValid :
    exact55748RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55748 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15947⟩⟩) exact55748RawTerms .large 55747 .exactZero (none)

def event55749 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26075⟩⟩) 0 ⟨15947⟩ 55748

def event55750 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26075⟩⟩) 1 ⟨26074⟩ 55733

def event55751 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26075⟩⟩) (.sum [.predecessor 0 55749 .coefficient, .predecessor 1 55750 .coefficient])

def exact55752RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨23586⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15944⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact55752RawTermsValid :
    exact55752RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55752 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26075⟩⟩) exact55752RawTerms .large 55751 .exactZero (none)

def event55753 : Event := .preFoldPolynomial 55752 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨23586⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15944⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact55754RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨23586⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15944⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event55754 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26075⟩⟩) 55753 exact55754RawTerms .large 55751 .exactZero (none)

def event55755 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14218⟩⟩) ⟨⟨110⟩, ⟨15⟩, ⟨109⟩⟩ ⟨55589, 55755⟩

def event55756 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19535⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19532⟩⟩]⟩) (1) 0 2 (.universal 55755 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19532⟩⟩]⟩) (none) 55754)

def event55757 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19535⟩⟩, .relation 55756 0, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩)

def event55758 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19535⟩⟩, .relation 55756 1, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩, (-1)⟩)

def event55759 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19535⟩⟩, .relation 55756 2, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨23586⟩⟩]⟩, (1)⟩)

def event55760 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19535⟩⟩, .relation 55756 3, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15944⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact55761RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨23586⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15944⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact55761RawTermsValid :
    exact55761RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55761 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19535⟩⟩) exact55761RawTerms .large 55585 (.finite 1811303510016) (some (55587))

def event55762 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26073⟩⟩) 0 ⟨19535⟩ 55761

def event55763 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26073⟩⟩) 1 ⟨26072⟩ 55575

def event55764 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26073⟩⟩) (.sum [.predecessor 0 55762 .coefficient, .predecessor 1 55763 .coefficient])

def event55765 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26073⟩⟩, .operator (⟨55761, 2⟩, ⟨55575, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨23586⟩⟩]⟩, (-1)⟩)

def event55766 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26073⟩⟩, .operator (⟨55761, 1⟩, ⟨55575, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩, (1)⟩)

def event55767 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26073⟩⟩) (.sum [.result 55761 .summary, .result 55575 .summary])

def exact55768RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15944⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact55768RawTermsValid :
    exact55768RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55768 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26073⟩⟩) exact55768RawTerms .large 55764 (.finite 352060719116288) (some (55767))

def event55769 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27881⟩⟩) 0 ⟨26073⟩ 55768

def event55770 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27881⟩⟩) 1 ⟨27879⟩ 55491

def event55771 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27881⟩⟩) (.product (.predecessor 0 55769 .coefficient) (.predecessor 1 55770 .coefficient) (⟨false, false, none, none, none⟩))

def event55772 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27881⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27879⟩⟩]⟩) [⟨.result 55491 .coefficient, false, none⟩])

def event55773 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27881⟩⟩) (.product (.result 55768 .summary) (.transfer 55772) (⟨false, false, none, none, none⟩))

def event55774 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27881⟩⟩, .operator (⟨55768, 0⟩, ⟨55491, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27879⟩⟩]⟩, (1)⟩)

def event55775 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27881⟩⟩, .operator (⟨55768, 1⟩, ⟨55491, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15944⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27879⟩⟩]⟩, (-1)⟩)

def event55776 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27881⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15944⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27879⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27879⟩⟩) ⟨24165⟩ 55488)

def event55777 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27881⟩⟩, .relation 55776 0, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15944⟩⟩], [⟨.program ⟨214⟩, ⟨24165⟩⟩]⟩, (-1)⟩)

def exact55778RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27879⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨15944⟩⟩], [⟨.program ⟨214⟩, ⟨24165⟩⟩]⟩, (-1)⟩]

theorem exact55778RawTermsValid :
    exact55778RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55778 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27881⟩⟩) exact55778RawTerms .large 55771 (.finite 1292068472128282820608) (some (55773))

def event55779 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21404⟩⟩) 0 ⟨15945⟩ 2585

def event55780 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21404⟩⟩) (.authority (.relationPreimageSource ⟨43⟩))

def exact55781RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21404⟩⟩]⟩, (1)⟩]

theorem exact55781RawTermsValid :
    exact55781RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55781 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21404⟩⟩) exact55781RawTerms (.finite 136065468) 55780 .exactZero (none)

def event55782 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21406⟩⟩) 0 ⟨21404⟩ 55781

def event55783 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21406⟩⟩) 1 ⟨2348⟩ 4

def event55784 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21406⟩⟩) (.scale (.predecessor 0 55782 .coefficient) (.value (.predecessor 1 55783 .coefficient)))

def exact55785RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21404⟩⟩]⟩, (1)⟩]

theorem exact55785RawTermsValid :
    exact55785RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55785 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21406⟩⟩) exact55785RawTerms (.finite 136065468) 55784 .exactZero (none)

def event55786 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21407⟩⟩) 0 ⟨5547⟩ 50762

def event55787 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21407⟩⟩) 1 ⟨21406⟩ 55785

def event55788 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21407⟩⟩) (.product (.predecessor 0 55786 .coefficient) (.predecessor 1 55787 .coefficient) (⟨false, false, none, none, none⟩))

def event55789 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21407⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21404⟩⟩]⟩) [⟨.result 55781 .coefficient, false, none⟩])

def event55790 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21407⟩⟩) (.product (.result 50762 .summary) (.transfer 55789) (⟨false, false, none, none, none⟩))

def event55791 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21407⟩⟩, .operator (⟨50762, 0⟩, ⟨55785, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21404⟩⟩]⟩, (1)⟩)

def event55792 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21405⟩⟩)

def event55793 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event55794 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event55795 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.authority (.operator))

def event55796 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.finite 3)

def event55797 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event55798 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event55799 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event55800 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event55801 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 55800

def event55802 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 55798

def event55803 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 55801 .coefficient) (.value (.predecessor 1 55802 .coefficient)))

def event55804 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event55805 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 0 ⟨5503⟩ 55804

def event55806 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 1 ⟨2875⟩ 55796

def event55807 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.sum [.predecessor 0 55805 .coefficient, .predecessor 1 55806 .coefficient])

def eventLeaf3472 : Array AnnotatedEvent := #[
  { event := event55552
    frameStart := 0 },
  { event := event55553
    frameStart := 0 },
  { event := event55554
    frameStart := 0 },
  { event := event55555
    frameStart := 0 },
  { event := event55556
    frameStart := 0 },
  { event := event55557
    frameStart := 0 },
  { event := event55558
    frameStart := 0 },
  { event := event55559
    frameStart := 0 },
  { event := event55560
    frameStart := 0 },
  { event := event55561
    frameStart := 0 },
  { event := event55562
    frameStart := 0 },
  { event := event55563
    frameStart := 0 },
  { event := event55564
    frameStart := 0 },
  { event := event55565
    frameStart := 0 },
  { event := event55566
    frameStart := 0 },
  { event := event55567
    frameStart := 0 }
]

def eventLeaf3473 : Array AnnotatedEvent := #[
  { event := event55568
    frameStart := 0 },
  { event := event55569
    frameStart := 0 },
  { event := event55570
    frameStart := 0 },
  { event := event55571
    frameStart := 0 },
  { event := event55572
    frameStart := 0 },
  { event := event55573
    frameStart := 0 },
  { event := event55574
    frameStart := 0 },
  { event := event55575
    frameStart := 0 },
  { event := event55576
    frameStart := 0 },
  { event := event55577
    frameStart := 0 },
  { event := event55578
    frameStart := 0 },
  { event := event55579
    frameStart := 0 },
  { event := event55580
    frameStart := 0 },
  { event := event55581
    frameStart := 0 },
  { event := event55582
    frameStart := 0 },
  { event := event55583
    frameStart := 0 }
]

def eventLeaf3474 : Array AnnotatedEvent := #[
  { event := event55584
    frameStart := 0 },
  { event := event55585
    frameStart := 0 },
  { event := event55586
    frameStart := 0 },
  { event := event55587
    frameStart := 0 },
  { event := event55588
    frameStart := 0 },
  { event := event55589
    frameStart := 55589 },
  { event := event55590
    frameStart := 55589 },
  { event := event55591
    frameStart := 55589 },
  { event := event55592
    frameStart := 55589 },
  { event := event55593
    frameStart := 55589 },
  { event := event55594
    frameStart := 55589 },
  { event := event55595
    frameStart := 55589 },
  { event := event55596
    frameStart := 55589 },
  { event := event55597
    frameStart := 55589 },
  { event := event55598
    frameStart := 55589 },
  { event := event55599
    frameStart := 55589 }
]

def eventLeaf3475 : Array AnnotatedEvent := #[
  { event := event55600
    frameStart := 55589 },
  { event := event55601
    frameStart := 55589 },
  { event := event55602
    frameStart := 55589 },
  { event := event55603
    frameStart := 55589 },
  { event := event55604
    frameStart := 55589 },
  { event := event55605
    frameStart := 55589 },
  { event := event55606
    frameStart := 55589 },
  { event := event55607
    frameStart := 55589 },
  { event := event55608
    frameStart := 55589 },
  { event := event55609
    frameStart := 55589 },
  { event := event55610
    frameStart := 55589 },
  { event := event55611
    frameStart := 55589 },
  { event := event55612
    frameStart := 55589 },
  { event := event55613
    frameStart := 55589 },
  { event := event55614
    frameStart := 55589 },
  { event := event55615
    frameStart := 55589 }
]

def eventLeaf3476 : Array AnnotatedEvent := #[
  { event := event55616
    frameStart := 55589 },
  { event := event55617
    frameStart := 55589 },
  { event := event55618
    frameStart := 55589 },
  { event := event55619
    frameStart := 55589 },
  { event := event55620
    frameStart := 55589 },
  { event := event55621
    frameStart := 55589 },
  { event := event55622
    frameStart := 55589 },
  { event := event55623
    frameStart := 55589 },
  { event := event55624
    frameStart := 55589 },
  { event := event55625
    frameStart := 55589 },
  { event := event55626
    frameStart := 55589 },
  { event := event55627
    frameStart := 55589 },
  { event := event55628
    frameStart := 55589 },
  { event := event55629
    frameStart := 55589 },
  { event := event55630
    frameStart := 55589 },
  { event := event55631
    frameStart := 55589 }
]

def eventLeaf3477 : Array AnnotatedEvent := #[
  { event := event55632
    frameStart := 55589 },
  { event := event55633
    frameStart := 55589 },
  { event := event55634
    frameStart := 55589 },
  { event := event55635
    frameStart := 55589 },
  { event := event55636
    frameStart := 55589 },
  { event := event55637
    frameStart := 55637 },
  { event := event55638
    frameStart := 55637 },
  { event := event55639
    frameStart := 55637 },
  { event := event55640
    frameStart := 55637 },
  { event := event55641
    frameStart := 55637 },
  { event := event55642
    frameStart := 55637 },
  { event := event55643
    frameStart := 55637 },
  { event := event55644
    frameStart := 55637 },
  { event := event55645
    frameStart := 55637 },
  { event := event55646
    frameStart := 55637 },
  { event := event55647
    frameStart := 55637 }
]

def eventLeaf3478 : Array AnnotatedEvent := #[
  { event := event55648
    frameStart := 55637 },
  { event := event55649
    frameStart := 55637 },
  { event := event55650
    frameStart := 55637 },
  { event := event55651
    frameStart := 55637 },
  { event := event55652
    frameStart := 55637 },
  { event := event55653
    frameStart := 55637 },
  { event := event55654
    frameStart := 55637 },
  { event := event55655
    frameStart := 55637 },
  { event := event55656
    frameStart := 55637 },
  { event := event55657
    frameStart := 55637 },
  { event := event55658
    frameStart := 55637 },
  { event := event55659
    frameStart := 55637 },
  { event := event55660
    frameStart := 55637 },
  { event := event55661
    frameStart := 55637 },
  { event := event55662
    frameStart := 55637 },
  { event := event55663
    frameStart := 55637 }
]

def eventLeaf3479 : Array AnnotatedEvent := #[
  { event := event55664
    frameStart := 55637 },
  { event := event55665
    frameStart := 55637 },
  { event := event55666
    frameStart := 55637 },
  { event := event55667
    frameStart := 55637 },
  { event := event55668
    frameStart := 55637 },
  { event := event55669
    frameStart := 55637 },
  { event := event55670
    frameStart := 55637 },
  { event := event55671
    frameStart := 55637 },
  { event := event55672
    frameStart := 55637 },
  { event := event55673
    frameStart := 55637 },
  { event := event55674
    frameStart := 55637 },
  { event := event55675
    frameStart := 55637 },
  { event := event55676
    frameStart := 55637 },
  { event := event55677
    frameStart := 55637 },
  { event := event55678
    frameStart := 55637 },
  { event := event55679
    frameStart := 55637 }
]

def eventLeaf3480 : Array AnnotatedEvent := #[
  { event := event55680
    frameStart := 55637 },
  { event := event55681
    frameStart := 55637 },
  { event := event55682
    frameStart := 55637 },
  { event := event55683
    frameStart := 55637 },
  { event := event55684
    frameStart := 55637 },
  { event := event55685
    frameStart := 55637 },
  { event := event55686
    frameStart := 55637 },
  { event := event55687
    frameStart := 55637 },
  { event := event55688
    frameStart := 55637 },
  { event := event55689
    frameStart := 55637 },
  { event := event55690
    frameStart := 55637 },
  { event := event55691
    frameStart := 55637 },
  { event := event55692
    frameStart := 55637 },
  { event := event55693
    frameStart := 55637 },
  { event := event55694
    frameStart := 55637 },
  { event := event55695
    frameStart := 55637 }
]

def eventLeaf3481 : Array AnnotatedEvent := #[
  { event := event55696
    frameStart := 55637 },
  { event := event55697
    frameStart := 55637 },
  { event := event55698
    frameStart := 55637 },
  { event := event55699
    frameStart := 55637 },
  { event := event55700
    frameStart := 55637 },
  { event := event55701
    frameStart := 55637 },
  { event := event55702
    frameStart := 55637 },
  { event := event55703
    frameStart := 55637 },
  { event := event55704
    frameStart := 55637 },
  { event := event55705
    frameStart := 55637 },
  { event := event55706
    frameStart := 55637 },
  { event := event55707
    frameStart := 55637 },
  { event := event55708
    frameStart := 55637 },
  { event := event55709
    frameStart := 55637 },
  { event := event55710
    frameStart := 55637 },
  { event := event55711
    frameStart := 55637 }
]

def eventLeaf3482 : Array AnnotatedEvent := #[
  { event := event55712
    frameStart := 55637 },
  { event := event55713
    frameStart := 55637 },
  { event := event55714
    frameStart := 55637 },
  { event := event55715
    frameStart := 55637 },
  { event := event55716
    frameStart := 55637 },
  { event := event55717
    frameStart := 55637 },
  { event := event55718
    frameStart := 55637 },
  { event := event55719
    frameStart := 55637 },
  { event := event55720
    frameStart := 55637 },
  { event := event55721
    frameStart := 55637 },
  { event := event55722
    frameStart := 55637 },
  { event := event55723
    frameStart := 55637 },
  { event := event55724
    frameStart := 55637 },
  { event := event55725
    frameStart := 55637 },
  { event := event55726
    frameStart := 55637 },
  { event := event55727
    frameStart := 55637 }
]

def eventLeaf3483 : Array AnnotatedEvent := #[
  { event := event55728
    frameStart := 55637 },
  { event := event55729
    frameStart := 55637 },
  { event := event55730
    frameStart := 55637 },
  { event := event55731
    frameStart := 55637 },
  { event := event55732
    frameStart := 55637 },
  { event := event55733
    frameStart := 55637 },
  { event := event55734
    frameStart := 55637 },
  { event := event55735
    frameStart := 55637 },
  { event := event55736
    frameStart := 55637 },
  { event := event55737
    frameStart := 55637 },
  { event := event55738
    frameStart := 55637 },
  { event := event55739
    frameStart := 55637 },
  { event := event55740
    frameStart := 55637 },
  { event := event55741
    frameStart := 55637 },
  { event := event55742
    frameStart := 55637 },
  { event := event55743
    frameStart := 55637 }
]

def eventLeaf3484 : Array AnnotatedEvent := #[
  { event := event55744
    frameStart := 55637 },
  { event := event55745
    frameStart := 55637 },
  { event := event55746
    frameStart := 55637 },
  { event := event55747
    frameStart := 55637 },
  { event := event55748
    frameStart := 55637 },
  { event := event55749
    frameStart := 55637 },
  { event := event55750
    frameStart := 55637 },
  { event := event55751
    frameStart := 55637 },
  { event := event55752
    frameStart := 55637 },
  { event := event55753
    frameStart := 55637 },
  { event := event55754
    frameStart := 55637 },
  { event := event55755
    frameStart := 0 },
  { event := event55756
    frameStart := 0 },
  { event := event55757
    frameStart := 0 },
  { event := event55758
    frameStart := 0 },
  { event := event55759
    frameStart := 0 }
]

def eventLeaf3485 : Array AnnotatedEvent := #[
  { event := event55760
    frameStart := 0 },
  { event := event55761
    frameStart := 0 },
  { event := event55762
    frameStart := 0 },
  { event := event55763
    frameStart := 0 },
  { event := event55764
    frameStart := 0 },
  { event := event55765
    frameStart := 0 },
  { event := event55766
    frameStart := 0 },
  { event := event55767
    frameStart := 0 },
  { event := event55768
    frameStart := 0 },
  { event := event55769
    frameStart := 0 },
  { event := event55770
    frameStart := 0 },
  { event := event55771
    frameStart := 0 },
  { event := event55772
    frameStart := 0 },
  { event := event55773
    frameStart := 0 },
  { event := event55774
    frameStart := 0 },
  { event := event55775
    frameStart := 0 }
]

def eventLeaf3486 : Array AnnotatedEvent := #[
  { event := event55776
    frameStart := 0 },
  { event := event55777
    frameStart := 0 },
  { event := event55778
    frameStart := 0 },
  { event := event55779
    frameStart := 0 },
  { event := event55780
    frameStart := 0 },
  { event := event55781
    frameStart := 0 },
  { event := event55782
    frameStart := 0 },
  { event := event55783
    frameStart := 0 },
  { event := event55784
    frameStart := 0 },
  { event := event55785
    frameStart := 0 },
  { event := event55786
    frameStart := 0 },
  { event := event55787
    frameStart := 0 },
  { event := event55788
    frameStart := 0 },
  { event := event55789
    frameStart := 0 },
  { event := event55790
    frameStart := 0 },
  { event := event55791
    frameStart := 0 }
]

def eventLeaf3487 : Array AnnotatedEvent := #[
  { event := event55792
    frameStart := 55792 },
  { event := event55793
    frameStart := 55792 },
  { event := event55794
    frameStart := 55792 },
  { event := event55795
    frameStart := 55792 },
  { event := event55796
    frameStart := 55792 },
  { event := event55797
    frameStart := 55792 },
  { event := event55798
    frameStart := 55792 },
  { event := event55799
    frameStart := 55792 },
  { event := event55800
    frameStart := 55792 },
  { event := event55801
    frameStart := 55792 },
  { event := event55802
    frameStart := 55792 },
  { event := event55803
    frameStart := 55792 },
  { event := event55804
    frameStart := 55792 },
  { event := event55805
    frameStart := 55792 },
  { event := event55806
    frameStart := 55792 },
  { event := event55807
    frameStart := 55792 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events217
