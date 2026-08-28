import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events174

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event44544 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.authority (.operator))

def event44545 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.finite 4)

def event44546 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event44547 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event44548 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event44549 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event44550 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 44549

def event44551 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 44547

def event44552 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 44550 .coefficient) (.value (.predecessor 1 44551 .coefficient)))

def event44553 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event44554 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 0 ⟨5503⟩ 44553

def event44555 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 1 ⟨4733⟩ 44545

def event44556 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.sum [.predecessor 0 44554 .coefficient, .predecessor 1 44555 .coefficient])

def event44557 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.finite 221)

def event44558 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 0 ⟨5512⟩ 44557

def event44559 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 1 ⟨961⟩ 44543

def event44560 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.identity (.predecessor 1 44559 .coefficient))

def event44561 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.finite 224)

def event44562 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10496⟩⟩) 0 ⟨5548⟩ 44561

def event44563 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10496⟩⟩) (.authority (.programFamilyFact))

def exact44564RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10496⟩⟩], []⟩, (1)⟩]

theorem exact44564RawTermsValid :
    exact44564RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44564 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10496⟩⟩) exact44564RawTerms (.finite 2) 44563 .exactZero (none)

def event44565 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9410⟩⟩) 0 ⟨5548⟩ 44561

def event44566 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9410⟩⟩) (.authority (.programFamilyFact))

def exact44567RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9410⟩⟩], []⟩, (1)⟩]

theorem exact44567RawTermsValid :
    exact44567RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44567 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9410⟩⟩) exact44567RawTerms (.finite 2) 44566 .exactZero (none)

def event44568 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10497⟩⟩) 0 ⟨9410⟩ 44567

def event44569 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10497⟩⟩) 1 ⟨10496⟩ 44564

def event44570 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10497⟩⟩) (.product (.predecessor 0 44568 .coefficient) (.predecessor 1 44569 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event44571 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10497⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], []⟩) [⟨.result 44567 .coefficient, true, some 1⟩, ⟨.result 44564 .coefficient, true, some 1⟩])

def event44572 : Event := .survivorFold (1) 44571

def exact44573RawTerms : List Term := []

theorem exact44573RawTermsValid :
    exact44573RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44573 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10497⟩⟩) exact44573RawTerms (.finite 4) 44570 (.finite 4) (some (44571))

def event44574 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10498⟩⟩) 0 ⟨10497⟩ 44573

def event44575 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10498⟩⟩) (.identity (.predecessor 0 44574 .coefficient))

def event44576 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10498⟩⟩) (.finite 4)

def event44577 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14800⟩⟩) 0 ⟨10498⟩ 44576

def event44578 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14800⟩⟩) (.authority (.programFamilyFact))

def exact44579RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], []⟩, (1)⟩]

theorem exact44579RawTermsValid :
    exact44579RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44579 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14800⟩⟩) exact44579RawTerms (.finite 2) 44578 .exactZero (none)

def event44580 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14801⟩⟩) 0 ⟨14800⟩ 44579

def event44581 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14801⟩⟩) (.identity (.predecessor 0 44580 .coefficient))

def event44582 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14801⟩⟩) (.finite 2)

def event44583 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20400⟩⟩) 0 ⟨14801⟩ 44582

def event44584 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20400⟩⟩) (.authority (.relationPreimageSource ⟨28⟩))

def exact44585RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20400⟩⟩]⟩, (1)⟩]

theorem exact44585RawTermsValid :
    exact44585RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44585 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20400⟩⟩) exact44585RawTerms (.finite 136065468) 44584 .exactZero (none)

def event44586 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact44587RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact44587RawTermsValid :
    exact44587RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44587 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact44587RawTerms .large 44586 .exactZero (none)

def event44588 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20401⟩⟩) 0 ⟨6⟩ 44587

def event44589 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20401⟩⟩) 1 ⟨20400⟩ 44585

def event44590 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20401⟩⟩) (.product (.predecessor 0 44588 .coefficient) (.predecessor 1 44589 .coefficient) (⟨false, false, none, none, none⟩))

def event44591 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20401⟩⟩, .operator (⟨44587, 0⟩, ⟨44585, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20400⟩⟩]⟩, (1)⟩)

def exact44592RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20400⟩⟩]⟩, (1)⟩]

theorem exact44592RawTermsValid :
    exact44592RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44592 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20401⟩⟩) exact44592RawTerms .large 44590 .exactZero (none)

def event44593 : Event := .preFoldPolynomial 44592 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20400⟩⟩]⟩, (1)⟩] .exactZero none

def exact44594RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20400⟩⟩]⟩, (1)⟩]

def event44594 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20401⟩⟩) 44593 exact44594RawTerms .large 44590 .exactZero (none)

def event44595 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26386⟩⟩)

def event44596 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event44597 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event44598 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.authority (.operator))

def event44599 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.finite 4)

def event44600 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event44601 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event44602 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event44603 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event44604 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 44603

def event44605 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 44601

def event44606 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 44604 .coefficient) (.value (.predecessor 1 44605 .coefficient)))

def event44607 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event44608 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 0 ⟨5503⟩ 44607

def event44609 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 1 ⟨4733⟩ 44599

def event44610 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.sum [.predecessor 0 44608 .coefficient, .predecessor 1 44609 .coefficient])

def event44611 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.finite 221)

def event44612 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 0 ⟨5512⟩ 44611

def event44613 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 1 ⟨961⟩ 44597

def event44614 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.identity (.predecessor 1 44613 .coefficient))

def event44615 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.finite 224)

def event44616 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10496⟩⟩) 0 ⟨5548⟩ 44615

def event44617 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10496⟩⟩) (.authority (.programFamilyFact))

def exact44618RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10496⟩⟩], []⟩, (1)⟩]

theorem exact44618RawTermsValid :
    exact44618RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44618 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10496⟩⟩) exact44618RawTerms (.finite 2) 44617 .exactZero (none)

def event44619 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9410⟩⟩) 0 ⟨5548⟩ 44615

def event44620 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9410⟩⟩) (.authority (.programFamilyFact))

def exact44621RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9410⟩⟩], []⟩, (1)⟩]

theorem exact44621RawTermsValid :
    exact44621RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44621 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9410⟩⟩) exact44621RawTerms (.finite 2) 44620 .exactZero (none)

def event44622 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10497⟩⟩) 0 ⟨9410⟩ 44621

def event44623 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10497⟩⟩) 1 ⟨10496⟩ 44618

def event44624 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10497⟩⟩) (.product (.predecessor 0 44622 .coefficient) (.predecessor 1 44623 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event44625 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10497⟩⟩, .operator (⟨44621, 0⟩, ⟨44618, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], []⟩, (1)⟩)

def exact44626RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], []⟩, (1)⟩]

theorem exact44626RawTermsValid :
    exact44626RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44626 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10497⟩⟩) exact44626RawTerms (.finite 4) 44624 .exactZero (none)

def event44627 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10498⟩⟩) 0 ⟨10497⟩ 44626

def event44628 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10498⟩⟩) (.identity (.predecessor 0 44627 .coefficient))

def event44629 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10498⟩⟩) (.finite 4)

def event44630 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14800⟩⟩) 0 ⟨10498⟩ 44629

def event44631 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14800⟩⟩) (.authority (.programFamilyFact))

def exact44632RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], []⟩, (1)⟩]

theorem exact44632RawTermsValid :
    exact44632RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44632 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14800⟩⟩) exact44632RawTerms (.finite 2) 44631 .exactZero (none)

def event44633 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14801⟩⟩) 0 ⟨14800⟩ 44632

def event44634 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14801⟩⟩) (.identity (.predecessor 0 44633 .coefficient))

def event44635 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14801⟩⟩) (.finite 2)

def event44636 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23725⟩⟩) 0 ⟨14801⟩ 44635

def event44637 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23725⟩⟩) (.authority (.programFamilyFact))

def event44638 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23725⟩⟩) (.finite 3720)

def event44639 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event44640 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23727⟩⟩) 0 ⟨6689⟩ 44639

def event44641 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23727⟩⟩) 1 ⟨23725⟩ 44638

def event44642 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23727⟩⟩) (.authority (.operator))

def exact44643RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23727⟩⟩]⟩, (1)⟩]

theorem exact44643RawTermsValid :
    exact44643RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44643 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23727⟩⟩) exact44643RawTerms .large 44642 .exactZero (none)

def event44644 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26382⟩⟩) 0 ⟨23727⟩ 44643

def event44645 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26382⟩⟩) (.authority (.operator))

def exact44646RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩, (1)⟩]

theorem exact44646RawTermsValid :
    exact44646RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44646 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26382⟩⟩) exact44646RawTerms (.finite 8192) 44645 .exactZero (none)

def event44647 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event44648 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event44649 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14840⟩⟩) 0 ⟨14801⟩ 44635

def event44650 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14840⟩⟩) 1 ⟨110⟩ 44648

def event44651 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14840⟩⟩) (.sum [.predecessor 0 44649 .coefficient, .predecessor 1 44650 .coefficient])

def event44652 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14840⟩⟩) (.finite 2)

def event44653 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14841⟩⟩) 0 ⟨14840⟩ 44652

def event44654 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14841⟩⟩) (.identity (.predecessor 0 44653 .coefficient))

def exact44655RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], []⟩, (1)⟩]

theorem exact44655RawTermsValid :
    exact44655RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44655 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14841⟩⟩) exact44655RawTerms (.finite 2) 44654 .exactZero (none)

def event44656 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact44657RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact44657RawTermsValid :
    exact44657RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44657 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact44657RawTerms .large 44656 .exactZero (none)

def event44658 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14842⟩⟩) 0 ⟨6544⟩ 44657

def event44659 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14842⟩⟩) 1 ⟨14841⟩ 44655

def event44660 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14842⟩⟩) (.product (.predecessor 0 44658 .coefficient) (.predecessor 1 44659 .coefficient) (⟨false, false, none, none, none⟩))

def event44661 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14842⟩⟩, .operator (⟨44657, 0⟩, ⟨44655, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact44662RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact44662RawTermsValid :
    exact44662RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44662 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14842⟩⟩) exact44662RawTerms .large 44660 .exactZero (none)

def event44663 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6690⟩⟩) 0 ⟨6689⟩ 44639

def event44664 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6690⟩⟩) (.authority (.operator))

def exact44665RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩]

theorem exact44665RawTermsValid :
    exact44665RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44665 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6690⟩⟩) exact44665RawTerms .large 44664 .exactZero (none)

def event44666 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14843⟩⟩) 0 ⟨6690⟩ 44665

def event44667 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14843⟩⟩) 1 ⟨14842⟩ 44662

def event44668 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14843⟩⟩) (.sum [.predecessor 0 44666 .coefficient, .predecessor 1 44667 .coefficient])

def exact44669RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44669RawTermsValid :
    exact44669RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44669 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14843⟩⟩) exact44669RawTerms .large 44668 .exactZero (none)

def event44670 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26383⟩⟩) 0 ⟨14843⟩ 44669

def event44671 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26383⟩⟩) 1 ⟨26382⟩ 44646

def event44672 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26383⟩⟩) (.product (.predecessor 0 44670 .coefficient) (.predecessor 1 44671 .coefficient) (⟨false, false, none, none, none⟩))

def event44673 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26383⟩⟩, .operator (⟨44669, 0⟩, ⟨44646, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩, (1)⟩)

def event44674 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26383⟩⟩, .operator (⟨44669, 1⟩, ⟨44646, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩, (-1)⟩)

def event44675 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26383⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26382⟩⟩) ⟨23727⟩ 44643)

def event44676 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26383⟩⟩, .relation 44675 0, ⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨23727⟩⟩]⟩, (-1)⟩)

def exact44677RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨23727⟩⟩]⟩, (-1)⟩]

theorem exact44677RawTermsValid :
    exact44677RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44677 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26383⟩⟩) exact44677RawTerms .large 44672 .exactZero (none)

def event44678 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15271⟩⟩) 0 ⟨14801⟩ 44635

def event44679 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15271⟩⟩) (.authority (.programFamilyFact))

def exact44680RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15271⟩⟩], []⟩, (1)⟩]

theorem exact44680RawTermsValid :
    exact44680RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44680 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15271⟩⟩) exact44680RawTerms (.finite 43) 44679 .exactZero (none)

def event44681 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15272⟩⟩) 0 ⟨6544⟩ 44657

def event44682 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15272⟩⟩) 1 ⟨15271⟩ 44680

def event44683 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15272⟩⟩) (.product (.predecessor 0 44681 .coefficient) (.predecessor 1 44682 .coefficient) (⟨false, true, none, none, some 1⟩))

def event44684 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15272⟩⟩, .operator (⟨44657, 0⟩, ⟨44680, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact44685RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact44685RawTermsValid :
    exact44685RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44685 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15272⟩⟩) exact44685RawTerms .large 44683 .exactZero (none)

def event44686 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6709⟩⟩) 0 ⟨6689⟩ 44639

def event44687 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6709⟩⟩) (.authority (.operator))

def exact44688RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩]

theorem exact44688RawTermsValid :
    exact44688RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44688 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6709⟩⟩) exact44688RawTerms .large 44687 .exactZero (none)

def event44689 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15273⟩⟩) 0 ⟨6709⟩ 44688

def event44690 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15273⟩⟩) 1 ⟨15272⟩ 44685

def event44691 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15273⟩⟩) (.sum [.predecessor 0 44689 .coefficient, .predecessor 1 44690 .coefficient])

def exact44692RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44692RawTermsValid :
    exact44692RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44692 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15273⟩⟩) exact44692RawTerms .large 44691 .exactZero (none)

def event44693 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26386⟩⟩) 0 ⟨15273⟩ 44692

def event44694 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26386⟩⟩) 1 ⟨26383⟩ 44677

def event44695 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26386⟩⟩) (.sum [.predecessor 0 44693 .coefficient, .predecessor 1 44694 .coefficient])

def exact44696RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨23727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44696RawTermsValid :
    exact44696RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44696 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26386⟩⟩) exact44696RawTerms .large 44695 .exactZero (none)

def event44697 : Event := .preFoldPolynomial 44696 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨23727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact44698RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨23727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event44698 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26386⟩⟩) 44697 exact44698RawTerms .large 44695 .exactZero (none)

def event44699 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14801⟩⟩) ⟨⟨122⟩, ⟨28⟩, ⟨109⟩⟩ ⟨44541, 44699⟩

def event44700 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20403⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20400⟩⟩]⟩) (1) 0 2 (.universal 44699 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20400⟩⟩]⟩) (none) 44698)

def event44701 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20403⟩⟩, .relation 44700 1, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩)

def event44702 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20403⟩⟩, .relation 44700 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩, (-1)⟩)

def event44703 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20403⟩⟩, .relation 44700 2, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨23727⟩⟩]⟩, (1)⟩)

def event44704 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20403⟩⟩, .relation 44700 3, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact44705RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨23727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44705RawTermsValid :
    exact44705RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44705 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20403⟩⟩) exact44705RawTerms .large 44537 (.finite 1811303510016) (some (44539))

def event44706 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26385⟩⟩) 0 ⟨20403⟩ 44705

def event44707 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26385⟩⟩) 1 ⟨26384⟩ 44527

def event44708 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26385⟩⟩) (.sum [.predecessor 0 44706 .coefficient, .predecessor 1 44707 .coefficient])

def event44709 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26385⟩⟩, .operator (⟨44705, 0⟩, ⟨44527, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩, (1)⟩)

def event44710 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26385⟩⟩, .operator (⟨44705, 2⟩, ⟨44527, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨23727⟩⟩]⟩, (-1)⟩)

def event44711 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26385⟩⟩) (.sum [.result 44705 .summary, .result 44527 .summary])

def exact44712RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44712RawTermsValid :
    exact44712RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44712 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26385⟩⟩) exact44712RawTerms .large 44708 (.finite 1291889174379421642752) (some (44711))

def event44713 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26594⟩⟩) 0 ⟨26385⟩ 44712

def event44714 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26594⟩⟩) 1 ⟨26593⟩ 44230

def event44715 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26594⟩⟩) (.sum [.predecessor 0 44713 .coefficient, .predecessor 1 44714 .coefficient])

def event44716 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26594⟩⟩) (.sum [.result 44712 .summary, .result 44230 .summary])

def exact44717RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44717RawTermsValid :
    exact44717RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44717 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26594⟩⟩) exact44717RawTerms .large 44715 (.finite 2583789554981353578496) (some (44716))

def event44718 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26811⟩⟩) 0 ⟨26594⟩ 44717

def event44719 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26811⟩⟩) 1 ⟨26810⟩ 43748

def event44720 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26811⟩⟩) (.sum [.predecessor 0 44718 .coefficient, .predecessor 1 44719 .coefficient])

def event44721 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26811⟩⟩) (.sum [.result 44717 .summary, .result 43748 .summary])

def exact44722RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44722RawTermsValid :
    exact44722RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44722 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26811⟩⟩) exact44722RawTerms .large 44720 (.finite 3875701141805795807232) (some (44721))

def event44723 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27028⟩⟩) 0 ⟨26811⟩ 44722

def event44724 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27028⟩⟩) 1 ⟨27027⟩ 43266

def event44725 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27028⟩⟩) (.sum [.predecessor 0 44723 .coefficient, .predecessor 1 44724 .coefficient])

def event44726 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27028⟩⟩) (.sum [.result 44722 .summary, .result 43266 .summary])

def exact44727RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17345⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44727RawTermsValid :
    exact44727RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44727 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27028⟩⟩) exact44727RawTerms .large 44725 (.finite 5167635141075258621952) (some (44726))

def event44728 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27245⟩⟩) 0 ⟨27028⟩ 44727

def event44729 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27245⟩⟩) 1 ⟨27244⟩ 42784

def event44730 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27245⟩⟩) (.sum [.predecessor 0 44728 .coefficient, .predecessor 1 44729 .coefficient])

def event44731 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27245⟩⟩) (.sum [.result 44727 .summary, .result 42784 .summary])

def exact44732RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15635⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17345⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44732RawTermsValid :
    exact44732RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44732 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27245⟩⟩) exact44732RawTerms .large 44730 (.finite 6459613965234762608640) (some (44731))

def event44733 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27462⟩⟩) 0 ⟨27245⟩ 44732

def event44734 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27462⟩⟩) 1 ⟨27461⟩ 42302

def event44735 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27462⟩⟩) (.sum [.predecessor 0 44733 .coefficient, .predecessor 1 44734 .coefficient])

def event44736 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27462⟩⟩) (.sum [.result 44732 .summary, .result 42302 .summary])

def exact44737RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15635⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17345⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44737RawTermsValid :
    exact44737RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44737 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27462⟩⟩) exact44737RawTerms .large 44735 (.finite 7751615201839287181312) (some (44736))

def event44738 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27679⟩⟩) 0 ⟨27462⟩ 44737

def event44739 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27679⟩⟩) 1 ⟨27678⟩ 41820

def event44740 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27679⟩⟩) (.sum [.predecessor 0 44738 .coefficient, .predecessor 1 44739 .coefficient])

def event44741 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27679⟩⟩) (.sum [.result 44737 .summary, .result 41820 .summary])

def exact44742RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15635⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17345⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44742RawTermsValid :
    exact44742RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44742 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27679⟩⟩) exact44742RawTerms .large 44740 (.finite 9043661263333852925952) (some (44741))

def event44743 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27896⟩⟩) 0 ⟨27679⟩ 44742

def event44744 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27896⟩⟩) 1 ⟨27895⟩ 41338

def event44745 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27896⟩⟩) (.sum [.predecessor 0 44743 .coefficient, .predecessor 1 44744 .coefficient])

def event44746 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27896⟩⟩) (.sum [.result 44742 .summary, .result 41338 .summary])

def exact44747RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15635⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15992⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17345⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44747RawTermsValid :
    exact44747RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44747 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27896⟩⟩) exact44747RawTerms .large 44745 (.finite 10335729737273439256576) (some (44746))

def event44748 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28113⟩⟩) 0 ⟨27896⟩ 44747

def event44749 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28113⟩⟩) 1 ⟨28112⟩ 40856

def event44750 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28113⟩⟩) (.sum [.predecessor 0 44748 .coefficient, .predecessor 1 44749 .coefficient])

def event44751 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28113⟩⟩) (.sum [.result 44747 .summary, .result 40856 .summary])

def exact44752RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15635⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15992⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16111⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17345⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44752RawTermsValid :
    exact44752RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44752 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28113⟩⟩) exact44752RawTerms .large 44750 (.finite 11627843036103066759168) (some (44751))

def event44753 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28330⟩⟩) 0 ⟨28113⟩ 44752

def event44754 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28330⟩⟩) 1 ⟨28329⟩ 40374

def event44755 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28330⟩⟩) (.sum [.predecessor 0 44753 .coefficient, .predecessor 1 44754 .coefficient])

def event44756 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28330⟩⟩) (.sum [.result 44752 .summary, .result 40374 .summary])

def exact44757RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15635⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15992⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16111⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17345⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44757RawTermsValid :
    exact44757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44757 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28330⟩⟩) exact44757RawTerms .large 44755 (.finite 12920023572267756019712) (some (44756))

def event44758 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28547⟩⟩) 0 ⟨28330⟩ 44757

def event44759 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28547⟩⟩) 1 ⟨28546⟩ 39892

def event44760 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28547⟩⟩) (.sum [.predecessor 0 44758 .coefficient, .predecessor 1 44759 .coefficient])

def event44761 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28547⟩⟩) (.sum [.result 44757 .summary, .result 39892 .summary])

def exact44762RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15635⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15992⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16111⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16314⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17345⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44762RawTermsValid :
    exact44762RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44762 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28547⟩⟩) exact44762RawTerms .large 44760 (.finite 14212226520877465866240) (some (44761))

def event44763 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28764⟩⟩) 0 ⟨28547⟩ 44762

def event44764 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28764⟩⟩) 1 ⟨28763⟩ 39410

def event44765 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28764⟩⟩) (.sum [.predecessor 0 44763 .coefficient, .predecessor 1 44764 .coefficient])

def event44766 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28764⟩⟩) (.sum [.result 44762 .summary, .result 39410 .summary])

def exact44767RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15635⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15992⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16111⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16314⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17126⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17345⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44767RawTermsValid :
    exact44767RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44767 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28764⟩⟩) exact44767RawTerms .large 44765 (.finite 15504496706822237470720) (some (44766))

def event44768 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28981⟩⟩) 0 ⟨28764⟩ 44767

def event44769 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28981⟩⟩) 1 ⟨28980⟩ 38928

def event44770 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28981⟩⟩) (.sum [.predecessor 0 44768 .coefficient, .predecessor 1 44769 .coefficient])

def event44771 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28981⟩⟩) (.sum [.result 44767 .summary, .result 38928 .summary])

def exact44772RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15635⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15992⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16111⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16314⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17126⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17345⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17910⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44772RawTermsValid :
    exact44772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44772 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28981⟩⟩) exact44772RawTerms .large 44770 (.finite 16796811717657050247168) (some (44771))

def event44773 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29198⟩⟩) 0 ⟨28981⟩ 44772

def event44774 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29198⟩⟩) 1 ⟨29197⟩ 38446

def event44775 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29198⟩⟩) (.sum [.predecessor 0 44773 .coefficient, .predecessor 1 44774 .coefficient])

def event44776 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29198⟩⟩) (.sum [.result 44772 .summary, .result 38446 .summary])

def exact44777RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15635⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15992⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16111⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16314⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17126⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17345⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17910⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44777RawTermsValid :
    exact44777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44777 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29198⟩⟩) exact44777RawTerms .large 44775 (.finite 18089149140936883609600) (some (44776))

def event44778 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29415⟩⟩) 0 ⟨29198⟩ 44777

def event44779 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29415⟩⟩) 1 ⟨29414⟩ 37964

def event44780 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29415⟩⟩) (.sum [.predecessor 0 44778 .coefficient, .predecessor 1 44779 .coefficient])

def event44781 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29415⟩⟩) (.sum [.result 44777 .summary, .result 37964 .summary])

def exact44782RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15635⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15992⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16111⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16314⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16685⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17126⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17345⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17910⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44782RawTermsValid :
    exact44782RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44782 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29415⟩⟩) exact44782RawTerms .large 44780 (.finite 19381531389106758144000) (some (44781))

def event44783 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29632⟩⟩) 0 ⟨29415⟩ 44782

def event44784 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29632⟩⟩) 1 ⟨29631⟩ 37482

def event44785 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29632⟩⟩) (.sum [.predecessor 0 44783 .coefficient, .predecessor 1 44784 .coefficient])

def event44786 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29632⟩⟩) (.sum [.result 44782 .summary, .result 37482 .summary])

def exact44787RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15635⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15992⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16111⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16314⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16685⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16804⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17126⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17345⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17910⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44787RawTermsValid :
    exact44787RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44787 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29632⟩⟩) exact44787RawTerms .large 44785 (.finite 20673980874611694436352) (some (44786))

def event44788 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29849⟩⟩) 0 ⟨29632⟩ 44787

def event44789 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29849⟩⟩) 1 ⟨29848⟩ 37000

def event44790 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29849⟩⟩) (.sum [.predecessor 0 44788 .coefficient, .predecessor 1 44789 .coefficient])

def event44791 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29849⟩⟩) (.sum [.result 44787 .summary, .result 37000 .summary])

def exact44792RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15635⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15992⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16111⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16314⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16685⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16804⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17091⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17126⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17345⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17910⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44792RawTermsValid :
    exact44792RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44792 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29849⟩⟩) exact44792RawTerms .large 44790 (.finite 21966497597451692486656) (some (44791))

def event44793 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30165⟩⟩) 0 ⟨29849⟩ 44792

def event44794 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30165⟩⟩) 1 ⟨30164⟩ 36518

def event44795 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30165⟩⟩) (.sum [.predecessor 0 44793 .coefficient, .predecessor 1 44794 .coefficient])

def event44796 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30165⟩⟩) (.sum [.result 44792 .summary, .result 36518 .summary])

def exact44797RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15635⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15992⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16111⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16314⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16685⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16804⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17091⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17126⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17345⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17910⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18176⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44797RawTermsValid :
    exact44797RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44797 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30165⟩⟩) exact44797RawTerms .large 44795 (.finite 23259036732736711122944) (some (44796))

def event44798 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30166⟩⟩) 0 ⟨30165⟩ 44797

def event44799 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30166⟩⟩) 1 ⟨18687⟩ 36020

def eventLeaf2784 : Array AnnotatedEvent := #[
  { event := event44544
    frameStart := 44541 },
  { event := event44545
    frameStart := 44541 },
  { event := event44546
    frameStart := 44541 },
  { event := event44547
    frameStart := 44541 },
  { event := event44548
    frameStart := 44541 },
  { event := event44549
    frameStart := 44541 },
  { event := event44550
    frameStart := 44541 },
  { event := event44551
    frameStart := 44541 },
  { event := event44552
    frameStart := 44541 },
  { event := event44553
    frameStart := 44541 },
  { event := event44554
    frameStart := 44541 },
  { event := event44555
    frameStart := 44541 },
  { event := event44556
    frameStart := 44541 },
  { event := event44557
    frameStart := 44541 },
  { event := event44558
    frameStart := 44541 },
  { event := event44559
    frameStart := 44541 }
]

def eventLeaf2785 : Array AnnotatedEvent := #[
  { event := event44560
    frameStart := 44541 },
  { event := event44561
    frameStart := 44541 },
  { event := event44562
    frameStart := 44541 },
  { event := event44563
    frameStart := 44541 },
  { event := event44564
    frameStart := 44541 },
  { event := event44565
    frameStart := 44541 },
  { event := event44566
    frameStart := 44541 },
  { event := event44567
    frameStart := 44541 },
  { event := event44568
    frameStart := 44541 },
  { event := event44569
    frameStart := 44541 },
  { event := event44570
    frameStart := 44541 },
  { event := event44571
    frameStart := 44541 },
  { event := event44572
    frameStart := 44541 },
  { event := event44573
    frameStart := 44541 },
  { event := event44574
    frameStart := 44541 },
  { event := event44575
    frameStart := 44541 }
]

def eventLeaf2786 : Array AnnotatedEvent := #[
  { event := event44576
    frameStart := 44541 },
  { event := event44577
    frameStart := 44541 },
  { event := event44578
    frameStart := 44541 },
  { event := event44579
    frameStart := 44541 },
  { event := event44580
    frameStart := 44541 },
  { event := event44581
    frameStart := 44541 },
  { event := event44582
    frameStart := 44541 },
  { event := event44583
    frameStart := 44541 },
  { event := event44584
    frameStart := 44541 },
  { event := event44585
    frameStart := 44541 },
  { event := event44586
    frameStart := 44541 },
  { event := event44587
    frameStart := 44541 },
  { event := event44588
    frameStart := 44541 },
  { event := event44589
    frameStart := 44541 },
  { event := event44590
    frameStart := 44541 },
  { event := event44591
    frameStart := 44541 }
]

def eventLeaf2787 : Array AnnotatedEvent := #[
  { event := event44592
    frameStart := 44541 },
  { event := event44593
    frameStart := 44541 },
  { event := event44594
    frameStart := 44541 },
  { event := event44595
    frameStart := 44595 },
  { event := event44596
    frameStart := 44595 },
  { event := event44597
    frameStart := 44595 },
  { event := event44598
    frameStart := 44595 },
  { event := event44599
    frameStart := 44595 },
  { event := event44600
    frameStart := 44595 },
  { event := event44601
    frameStart := 44595 },
  { event := event44602
    frameStart := 44595 },
  { event := event44603
    frameStart := 44595 },
  { event := event44604
    frameStart := 44595 },
  { event := event44605
    frameStart := 44595 },
  { event := event44606
    frameStart := 44595 },
  { event := event44607
    frameStart := 44595 }
]

def eventLeaf2788 : Array AnnotatedEvent := #[
  { event := event44608
    frameStart := 44595 },
  { event := event44609
    frameStart := 44595 },
  { event := event44610
    frameStart := 44595 },
  { event := event44611
    frameStart := 44595 },
  { event := event44612
    frameStart := 44595 },
  { event := event44613
    frameStart := 44595 },
  { event := event44614
    frameStart := 44595 },
  { event := event44615
    frameStart := 44595 },
  { event := event44616
    frameStart := 44595 },
  { event := event44617
    frameStart := 44595 },
  { event := event44618
    frameStart := 44595 },
  { event := event44619
    frameStart := 44595 },
  { event := event44620
    frameStart := 44595 },
  { event := event44621
    frameStart := 44595 },
  { event := event44622
    frameStart := 44595 },
  { event := event44623
    frameStart := 44595 }
]

def eventLeaf2789 : Array AnnotatedEvent := #[
  { event := event44624
    frameStart := 44595 },
  { event := event44625
    frameStart := 44595 },
  { event := event44626
    frameStart := 44595 },
  { event := event44627
    frameStart := 44595 },
  { event := event44628
    frameStart := 44595 },
  { event := event44629
    frameStart := 44595 },
  { event := event44630
    frameStart := 44595 },
  { event := event44631
    frameStart := 44595 },
  { event := event44632
    frameStart := 44595 },
  { event := event44633
    frameStart := 44595 },
  { event := event44634
    frameStart := 44595 },
  { event := event44635
    frameStart := 44595 },
  { event := event44636
    frameStart := 44595 },
  { event := event44637
    frameStart := 44595 },
  { event := event44638
    frameStart := 44595 },
  { event := event44639
    frameStart := 44595 }
]

def eventLeaf2790 : Array AnnotatedEvent := #[
  { event := event44640
    frameStart := 44595 },
  { event := event44641
    frameStart := 44595 },
  { event := event44642
    frameStart := 44595 },
  { event := event44643
    frameStart := 44595 },
  { event := event44644
    frameStart := 44595 },
  { event := event44645
    frameStart := 44595 },
  { event := event44646
    frameStart := 44595 },
  { event := event44647
    frameStart := 44595 },
  { event := event44648
    frameStart := 44595 },
  { event := event44649
    frameStart := 44595 },
  { event := event44650
    frameStart := 44595 },
  { event := event44651
    frameStart := 44595 },
  { event := event44652
    frameStart := 44595 },
  { event := event44653
    frameStart := 44595 },
  { event := event44654
    frameStart := 44595 },
  { event := event44655
    frameStart := 44595 }
]

def eventLeaf2791 : Array AnnotatedEvent := #[
  { event := event44656
    frameStart := 44595 },
  { event := event44657
    frameStart := 44595 },
  { event := event44658
    frameStart := 44595 },
  { event := event44659
    frameStart := 44595 },
  { event := event44660
    frameStart := 44595 },
  { event := event44661
    frameStart := 44595 },
  { event := event44662
    frameStart := 44595 },
  { event := event44663
    frameStart := 44595 },
  { event := event44664
    frameStart := 44595 },
  { event := event44665
    frameStart := 44595 },
  { event := event44666
    frameStart := 44595 },
  { event := event44667
    frameStart := 44595 },
  { event := event44668
    frameStart := 44595 },
  { event := event44669
    frameStart := 44595 },
  { event := event44670
    frameStart := 44595 },
  { event := event44671
    frameStart := 44595 }
]

def eventLeaf2792 : Array AnnotatedEvent := #[
  { event := event44672
    frameStart := 44595 },
  { event := event44673
    frameStart := 44595 },
  { event := event44674
    frameStart := 44595 },
  { event := event44675
    frameStart := 44595 },
  { event := event44676
    frameStart := 44595 },
  { event := event44677
    frameStart := 44595 },
  { event := event44678
    frameStart := 44595 },
  { event := event44679
    frameStart := 44595 },
  { event := event44680
    frameStart := 44595 },
  { event := event44681
    frameStart := 44595 },
  { event := event44682
    frameStart := 44595 },
  { event := event44683
    frameStart := 44595 },
  { event := event44684
    frameStart := 44595 },
  { event := event44685
    frameStart := 44595 },
  { event := event44686
    frameStart := 44595 },
  { event := event44687
    frameStart := 44595 }
]

def eventLeaf2793 : Array AnnotatedEvent := #[
  { event := event44688
    frameStart := 44595 },
  { event := event44689
    frameStart := 44595 },
  { event := event44690
    frameStart := 44595 },
  { event := event44691
    frameStart := 44595 },
  { event := event44692
    frameStart := 44595 },
  { event := event44693
    frameStart := 44595 },
  { event := event44694
    frameStart := 44595 },
  { event := event44695
    frameStart := 44595 },
  { event := event44696
    frameStart := 44595 },
  { event := event44697
    frameStart := 44595 },
  { event := event44698
    frameStart := 44595 },
  { event := event44699
    frameStart := 0 },
  { event := event44700
    frameStart := 0 },
  { event := event44701
    frameStart := 0 },
  { event := event44702
    frameStart := 0 },
  { event := event44703
    frameStart := 0 }
]

def eventLeaf2794 : Array AnnotatedEvent := #[
  { event := event44704
    frameStart := 0 },
  { event := event44705
    frameStart := 0 },
  { event := event44706
    frameStart := 0 },
  { event := event44707
    frameStart := 0 },
  { event := event44708
    frameStart := 0 },
  { event := event44709
    frameStart := 0 },
  { event := event44710
    frameStart := 0 },
  { event := event44711
    frameStart := 0 },
  { event := event44712
    frameStart := 0 },
  { event := event44713
    frameStart := 0 },
  { event := event44714
    frameStart := 0 },
  { event := event44715
    frameStart := 0 },
  { event := event44716
    frameStart := 0 },
  { event := event44717
    frameStart := 0 },
  { event := event44718
    frameStart := 0 },
  { event := event44719
    frameStart := 0 }
]

def eventLeaf2795 : Array AnnotatedEvent := #[
  { event := event44720
    frameStart := 0 },
  { event := event44721
    frameStart := 0 },
  { event := event44722
    frameStart := 0 },
  { event := event44723
    frameStart := 0 },
  { event := event44724
    frameStart := 0 },
  { event := event44725
    frameStart := 0 },
  { event := event44726
    frameStart := 0 },
  { event := event44727
    frameStart := 0 },
  { event := event44728
    frameStart := 0 },
  { event := event44729
    frameStart := 0 },
  { event := event44730
    frameStart := 0 },
  { event := event44731
    frameStart := 0 },
  { event := event44732
    frameStart := 0 },
  { event := event44733
    frameStart := 0 },
  { event := event44734
    frameStart := 0 },
  { event := event44735
    frameStart := 0 }
]

def eventLeaf2796 : Array AnnotatedEvent := #[
  { event := event44736
    frameStart := 0 },
  { event := event44737
    frameStart := 0 },
  { event := event44738
    frameStart := 0 },
  { event := event44739
    frameStart := 0 },
  { event := event44740
    frameStart := 0 },
  { event := event44741
    frameStart := 0 },
  { event := event44742
    frameStart := 0 },
  { event := event44743
    frameStart := 0 },
  { event := event44744
    frameStart := 0 },
  { event := event44745
    frameStart := 0 },
  { event := event44746
    frameStart := 0 },
  { event := event44747
    frameStart := 0 },
  { event := event44748
    frameStart := 0 },
  { event := event44749
    frameStart := 0 },
  { event := event44750
    frameStart := 0 },
  { event := event44751
    frameStart := 0 }
]

def eventLeaf2797 : Array AnnotatedEvent := #[
  { event := event44752
    frameStart := 0 },
  { event := event44753
    frameStart := 0 },
  { event := event44754
    frameStart := 0 },
  { event := event44755
    frameStart := 0 },
  { event := event44756
    frameStart := 0 },
  { event := event44757
    frameStart := 0 },
  { event := event44758
    frameStart := 0 },
  { event := event44759
    frameStart := 0 },
  { event := event44760
    frameStart := 0 },
  { event := event44761
    frameStart := 0 },
  { event := event44762
    frameStart := 0 },
  { event := event44763
    frameStart := 0 },
  { event := event44764
    frameStart := 0 },
  { event := event44765
    frameStart := 0 },
  { event := event44766
    frameStart := 0 },
  { event := event44767
    frameStart := 0 }
]

def eventLeaf2798 : Array AnnotatedEvent := #[
  { event := event44768
    frameStart := 0 },
  { event := event44769
    frameStart := 0 },
  { event := event44770
    frameStart := 0 },
  { event := event44771
    frameStart := 0 },
  { event := event44772
    frameStart := 0 },
  { event := event44773
    frameStart := 0 },
  { event := event44774
    frameStart := 0 },
  { event := event44775
    frameStart := 0 },
  { event := event44776
    frameStart := 0 },
  { event := event44777
    frameStart := 0 },
  { event := event44778
    frameStart := 0 },
  { event := event44779
    frameStart := 0 },
  { event := event44780
    frameStart := 0 },
  { event := event44781
    frameStart := 0 },
  { event := event44782
    frameStart := 0 },
  { event := event44783
    frameStart := 0 }
]

def eventLeaf2799 : Array AnnotatedEvent := #[
  { event := event44784
    frameStart := 0 },
  { event := event44785
    frameStart := 0 },
  { event := event44786
    frameStart := 0 },
  { event := event44787
    frameStart := 0 },
  { event := event44788
    frameStart := 0 },
  { event := event44789
    frameStart := 0 },
  { event := event44790
    frameStart := 0 },
  { event := event44791
    frameStart := 0 },
  { event := event44792
    frameStart := 0 },
  { event := event44793
    frameStart := 0 },
  { event := event44794
    frameStart := 0 },
  { event := event44795
    frameStart := 0 },
  { event := event44796
    frameStart := 0 },
  { event := event44797
    frameStart := 0 },
  { event := event44798
    frameStart := 0 },
  { event := event44799
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events174
