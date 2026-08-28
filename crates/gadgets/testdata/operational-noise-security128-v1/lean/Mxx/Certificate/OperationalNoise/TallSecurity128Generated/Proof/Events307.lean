import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events307

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event78592 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36781⟩⟩) 0 ⟨36327⟩ 78591

def event78593 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36781⟩⟩) 1 ⟨36779⟩ 78314

def event78594 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36781⟩⟩) (.product (.predecessor 0 78592 .coefficient) (.predecessor 1 78593 .coefficient) (⟨false, false, none, none, none⟩))

def event78595 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36781⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨36779⟩⟩]⟩) [⟨.result 78314 .coefficient, false, none⟩])

def event78596 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36781⟩⟩) (.product (.result 78591 .summary) (.transfer 78595) (⟨false, false, none, none, none⟩))

def event78597 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36781⟩⟩, .operator (⟨78591, 0⟩, ⟨78314, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36779⟩⟩]⟩, (1)⟩)

def event78598 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36781⟩⟩, .operator (⟨78591, 1⟩, ⟨78314, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36779⟩⟩]⟩, (-1)⟩)

def event78599 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36781⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36779⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36779⟩⟩) ⟨35955⟩ 78311)

def event78600 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36781⟩⟩, .relation 78599 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35955⟩⟩]⟩, (-1)⟩)

def exact78601RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36779⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35955⟩⟩]⟩, (-1)⟩]

theorem exact78601RawTermsValid :
    exact78601RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78601 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36781⟩⟩) exact78601RawTerms .large 78594 (.finite 32192539770951564984245676933120) (some (78596))

def event78602 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35616⟩⟩) 0 ⟨34797⟩ 3218

def event78603 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35616⟩⟩) (.authority (.relationPreimageSource ⟨83⟩))

def exact78604RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35616⟩⟩]⟩, (1)⟩]

theorem exact78604RawTermsValid :
    exact78604RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78604 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35616⟩⟩) exact78604RawTerms (.finite 5647228698) 78603 .exactZero (none)

def event78605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35618⟩⟩) 0 ⟨35616⟩ 78604

def event78606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35618⟩⟩) 1 ⟨2370⟩ 4

def event78607 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35618⟩⟩) (.scale (.predecessor 0 78605 .coefficient) (.value (.predecessor 1 78606 .coefficient)))

def exact78608RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35616⟩⟩]⟩, (1)⟩]

theorem exact78608RawTermsValid :
    exact78608RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78608 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35618⟩⟩) exact78608RawTerms (.finite 5647228698) 78607 .exactZero (none)

def event78609 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35619⟩⟩) 0 ⟨10368⟩ 75995

def event78610 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35619⟩⟩) 1 ⟨35618⟩ 78608

def event78611 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35619⟩⟩) (.product (.predecessor 0 78609 .coefficient) (.predecessor 1 78610 .coefficient) (⟨false, false, none, none, none⟩))

def event78612 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35619⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨35616⟩⟩]⟩) [⟨.result 78604 .coefficient, false, none⟩])

def event78613 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35619⟩⟩) (.product (.result 75995 .summary) (.transfer 78612) (⟨false, false, none, none, none⟩))

def event78614 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35619⟩⟩, .operator (⟨75995, 0⟩, ⟨78608, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35616⟩⟩]⟩, (1)⟩)

def event78615 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨35617⟩⟩)

def event78616 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event78617 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event78618 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event78619 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event78620 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event78621 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event78622 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event78623 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event78624 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 78623

def event78625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 78621

def event78626 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 78624 .coefficient) (.value (.predecessor 1 78625 .coefficient)))

def event78627 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event78628 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 78627

def event78629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 78619

def event78630 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 78628 .coefficient, .predecessor 1 78629 .coefficient])

def event78631 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event78632 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 78631

def event78633 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 78617

def event78634 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 78633 .coefficient))

def event78635 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event78636 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34578⟩⟩) 0 ⟨10325⟩ 78635

def event78637 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34578⟩⟩) (.authority (.programFamilyFact))

def exact78638RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34578⟩⟩], []⟩, (1)⟩]

theorem exact78638RawTermsValid :
    exact78638RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78638 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34578⟩⟩) exact78638RawTerms (.finite 40) 78637 .exactZero (none)

def event78639 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13671⟩⟩) 0 ⟨10325⟩ 78635

def event78640 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13671⟩⟩) (.authority (.programFamilyFact))

def exact78641RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩], []⟩, (1)⟩]

theorem exact78641RawTermsValid :
    exact78641RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78641 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13671⟩⟩) exact78641RawTerms (.finite 40) 78640 .exactZero (none)

def event78642 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34579⟩⟩) 0 ⟨13671⟩ 78641

def event78643 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34579⟩⟩) 1 ⟨34578⟩ 78638

def event78644 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34579⟩⟩) (.product (.predecessor 0 78642 .coefficient) (.predecessor 1 78643 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event78645 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34579⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], []⟩) [⟨.result 78641 .coefficient, true, some 1⟩, ⟨.result 78638 .coefficient, true, some 1⟩])

def event78646 : Event := .survivorFold (1) 78645

def exact78647RawTerms : List Term := []

theorem exact78647RawTermsValid :
    exact78647RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78647 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34579⟩⟩) exact78647RawTerms (.finite 1600) 78644 (.finite 1600) (some (78645))

def event78648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34580⟩⟩) 0 ⟨34579⟩ 78647

def event78649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34580⟩⟩) (.identity (.predecessor 0 78648 .coefficient))

def event78650 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34580⟩⟩) (.finite 1600)

def event78651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34796⟩⟩) 0 ⟨34580⟩ 78650

def event78652 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34796⟩⟩) (.authority (.programFamilyFact))

def exact78653RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], []⟩, (1)⟩]

theorem exact78653RawTermsValid :
    exact78653RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78653 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34796⟩⟩) exact78653RawTerms (.finite 40) 78652 .exactZero (none)

def event78654 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34797⟩⟩) 0 ⟨34796⟩ 78653

def event78655 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34797⟩⟩) (.identity (.predecessor 0 78654 .coefficient))

def event78656 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34797⟩⟩) (.finite 40)

def event78657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35616⟩⟩) 0 ⟨34797⟩ 78656

def event78658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35616⟩⟩) (.authority (.relationPreimageSource ⟨83⟩))

def exact78659RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35616⟩⟩]⟩, (1)⟩]

theorem exact78659RawTermsValid :
    exact78659RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78659 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35616⟩⟩) exact78659RawTerms (.finite 5647228698) 78658 .exactZero (none)

def event78660 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact78661RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact78661RawTermsValid :
    exact78661RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78661 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact78661RawTerms .large 78660 .exactZero (none)

def event78662 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35617⟩⟩) 0 ⟨35⟩ 78661

def event78663 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35617⟩⟩) 1 ⟨35616⟩ 78659

def event78664 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35617⟩⟩) (.product (.predecessor 0 78662 .coefficient) (.predecessor 1 78663 .coefficient) (⟨false, false, none, none, none⟩))

def event78665 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35617⟩⟩, .operator (⟨78661, 0⟩, ⟨78659, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35616⟩⟩]⟩, (1)⟩)

def exact78666RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35616⟩⟩]⟩, (1)⟩]

theorem exact78666RawTermsValid :
    exact78666RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78666 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35617⟩⟩) exact78666RawTerms .large 78664 .exactZero (none)

def event78667 : Event := .preFoldPolynomial 78666 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35616⟩⟩]⟩, (1)⟩] .exactZero none

def exact78668RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35616⟩⟩]⟩, (1)⟩]

def event78668 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨35617⟩⟩) 78667 exact78668RawTerms .large 78664 .exactZero (none)

def event78669 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨36783⟩⟩)

def event78670 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event78671 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event78672 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event78673 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event78674 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event78675 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event78676 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event78677 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event78678 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 78677

def event78679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 78675

def event78680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 78678 .coefficient) (.value (.predecessor 1 78679 .coefficient)))

def event78681 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event78682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 78681

def event78683 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 78673

def event78684 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 78682 .coefficient, .predecessor 1 78683 .coefficient])

def event78685 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event78686 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 78685

def event78687 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 78671

def event78688 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 78687 .coefficient))

def event78689 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event78690 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34578⟩⟩) 0 ⟨10325⟩ 78689

def event78691 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34578⟩⟩) (.authority (.programFamilyFact))

def exact78692RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34578⟩⟩], []⟩, (1)⟩]

theorem exact78692RawTermsValid :
    exact78692RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78692 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34578⟩⟩) exact78692RawTerms (.finite 40) 78691 .exactZero (none)

def event78693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13671⟩⟩) 0 ⟨10325⟩ 78689

def event78694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13671⟩⟩) (.authority (.programFamilyFact))

def exact78695RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩], []⟩, (1)⟩]

theorem exact78695RawTermsValid :
    exact78695RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78695 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13671⟩⟩) exact78695RawTerms (.finite 40) 78694 .exactZero (none)

def event78696 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34579⟩⟩) 0 ⟨13671⟩ 78695

def event78697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34579⟩⟩) 1 ⟨34578⟩ 78692

def event78698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34579⟩⟩) (.product (.predecessor 0 78696 .coefficient) (.predecessor 1 78697 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event78699 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34579⟩⟩, .operator (⟨78695, 0⟩, ⟨78692, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], []⟩, (1)⟩)

def exact78700RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩, ⟨.program ⟨257⟩, ⟨34578⟩⟩], []⟩, (1)⟩]

theorem exact78700RawTermsValid :
    exact78700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78700 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34579⟩⟩) exact78700RawTerms (.finite 1600) 78698 .exactZero (none)

def event78701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34580⟩⟩) 0 ⟨34579⟩ 78700

def event78702 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34580⟩⟩) (.identity (.predecessor 0 78701 .coefficient))

def event78703 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34580⟩⟩) (.finite 1600)

def event78704 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34796⟩⟩) 0 ⟨34580⟩ 78703

def event78705 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34796⟩⟩) (.authority (.programFamilyFact))

def exact78706RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], []⟩, (1)⟩]

theorem exact78706RawTermsValid :
    exact78706RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78706 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34796⟩⟩) exact78706RawTerms (.finite 40) 78705 .exactZero (none)

def event78707 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34797⟩⟩) 0 ⟨34796⟩ 78706

def event78708 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34797⟩⟩) (.identity (.predecessor 0 78707 .coefficient))

def event78709 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34797⟩⟩) (.finite 40)

def event78710 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35953⟩⟩) 0 ⟨34797⟩ 78709

def event78711 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35953⟩⟩) (.authority (.programFamilyFact))

def event78712 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨35953⟩⟩) (.finite 3720)

def event78713 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event78714 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35955⟩⟩) 0 ⟨7177⟩ 78713

def event78715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35955⟩⟩) 1 ⟨35953⟩ 78712

def event78716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35955⟩⟩) (.authority (.operator))

def exact78717RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35955⟩⟩]⟩, (1)⟩]

theorem exact78717RawTermsValid :
    exact78717RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78717 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35955⟩⟩) exact78717RawTerms .large 78716 .exactZero (none)

def event78718 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36779⟩⟩) 0 ⟨35955⟩ 78717

def event78719 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36779⟩⟩) (.authority (.operator))

def exact78720RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨36779⟩⟩]⟩, (1)⟩]

theorem exact78720RawTermsValid :
    exact78720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78720 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36779⟩⟩) exact78720RawTerms (.finite 8192) 78719 .exactZero (none)

def event78721 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event78722 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event78723 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36130⟩⟩) 0 ⟨34797⟩ 78709

def event78724 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36130⟩⟩) 1 ⟨136⟩ 78722

def event78725 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36130⟩⟩) (.sum [.predecessor 0 78723 .coefficient, .predecessor 1 78724 .coefficient])

def event78726 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨36130⟩⟩) (.finite 40)

def event78727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36131⟩⟩) 0 ⟨36130⟩ 78726

def event78728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36131⟩⟩) (.identity (.predecessor 0 78727 .coefficient))

def exact78729RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], []⟩, (1)⟩]

theorem exact78729RawTermsValid :
    exact78729RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78729 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36131⟩⟩) exact78729RawTerms (.finite 40) 78728 .exactZero (none)

def event78730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact78731RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact78731RawTermsValid :
    exact78731RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78731 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact78731RawTerms .large 78730 .exactZero (none)

def event78732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36132⟩⟩) 0 ⟨6908⟩ 78731

def event78733 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36132⟩⟩) 1 ⟨36131⟩ 78729

def event78734 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36132⟩⟩) (.product (.predecessor 0 78732 .coefficient) (.predecessor 1 78733 .coefficient) (⟨false, false, none, none, none⟩))

def event78735 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36132⟩⟩, .operator (⟨78731, 0⟩, ⟨78729, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact78736RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact78736RawTermsValid :
    exact78736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78736 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36132⟩⟩) exact78736RawTerms .large 78734 .exactZero (none)

def event78737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7191⟩⟩) 0 ⟨7177⟩ 78713

def event78738 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7191⟩⟩) (.authority (.operator))

def exact78739RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩]

theorem exact78739RawTermsValid :
    exact78739RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78739 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7191⟩⟩) exact78739RawTerms .large 78738 .exactZero (none)

def event78740 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36133⟩⟩) 0 ⟨7191⟩ 78739

def event78741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36133⟩⟩) 1 ⟨36132⟩ 78736

def event78742 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36133⟩⟩) (.sum [.predecessor 0 78740 .coefficient, .predecessor 1 78741 .coefficient])

def exact78743RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78743RawTermsValid :
    exact78743RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78743 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36133⟩⟩) exact78743RawTerms .large 78742 .exactZero (none)

def event78744 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36780⟩⟩) 0 ⟨36133⟩ 78743

def event78745 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36780⟩⟩) 1 ⟨36779⟩ 78720

def event78746 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36780⟩⟩) (.product (.predecessor 0 78744 .coefficient) (.predecessor 1 78745 .coefficient) (⟨false, false, none, none, none⟩))

def event78747 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36780⟩⟩, .operator (⟨78743, 0⟩, ⟨78720, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36779⟩⟩]⟩, (1)⟩)

def event78748 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36780⟩⟩, .operator (⟨78743, 1⟩, ⟨78720, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36779⟩⟩]⟩, (-1)⟩)

def event78749 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36780⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36779⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36779⟩⟩) ⟨35955⟩ 78717)

def event78750 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36780⟩⟩, .relation 78749 0, ⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35955⟩⟩]⟩, (-1)⟩)

def exact78751RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36779⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35955⟩⟩]⟩, (-1)⟩]

theorem exact78751RawTermsValid :
    exact78751RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78751 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36780⟩⟩) exact78751RawTerms .large 78746 .exactZero (none)

def event78752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35041⟩⟩) 0 ⟨34797⟩ 78709

def event78753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35041⟩⟩) (.authority (.programFamilyFact))

def exact78754RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], []⟩, (1)⟩]

theorem exact78754RawTermsValid :
    exact78754RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78754 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35041⟩⟩) exact78754RawTerms (.finite 62) 78753 .exactZero (none)

def event78755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35042⟩⟩) 0 ⟨6908⟩ 78731

def event78756 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35042⟩⟩) 1 ⟨35041⟩ 78754

def event78757 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35042⟩⟩) (.product (.predecessor 0 78755 .coefficient) (.predecessor 1 78756 .coefficient) (⟨false, true, none, none, some 1⟩))

def event78758 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35042⟩⟩, .operator (⟨78731, 0⟩, ⟨78754, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact78759RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact78759RawTermsValid :
    exact78759RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78759 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35042⟩⟩) exact78759RawTerms .large 78757 .exactZero (none)

def event78760 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7222⟩⟩) 0 ⟨7177⟩ 78713

def event78761 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7222⟩⟩) (.authority (.operator))

def exact78762RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩]

theorem exact78762RawTermsValid :
    exact78762RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78762 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7222⟩⟩) exact78762RawTerms .large 78761 .exactZero (none)

def event78763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35043⟩⟩) 0 ⟨7222⟩ 78762

def event78764 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35043⟩⟩) 1 ⟨35042⟩ 78759

def event78765 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35043⟩⟩) (.sum [.predecessor 0 78763 .coefficient, .predecessor 1 78764 .coefficient])

def exact78766RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78766RawTermsValid :
    exact78766RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78766 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35043⟩⟩) exact78766RawTerms .large 78765 .exactZero (none)

def event78767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36783⟩⟩) 0 ⟨35043⟩ 78766

def event78768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36783⟩⟩) 1 ⟨36780⟩ 78751

def event78769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36783⟩⟩) (.sum [.predecessor 0 78767 .coefficient, .predecessor 1 78768 .coefficient])

def exact78770RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36779⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35955⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78770RawTermsValid :
    exact78770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78770 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36783⟩⟩) exact78770RawTerms .large 78769 .exactZero (none)

def event78771 : Event := .preFoldPolynomial 78770 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36779⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35955⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact78772RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36779⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35955⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event78772 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨36783⟩⟩) 78771 exact78772RawTerms .large 78769 .exactZero (none)

def event78773 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨34797⟩⟩) ⟨⟨101⟩, ⟨83⟩, ⟨135⟩⟩ ⟨78615, 78773⟩

def event78774 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨35619⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35616⟩⟩]⟩) (1) 0 2 (.universal 78773 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35616⟩⟩]⟩) (none) 78772)

def event78775 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35619⟩⟩, .relation 78774 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩)

def event78776 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35619⟩⟩, .relation 78774 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36779⟩⟩]⟩, (-1)⟩)

def event78777 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35619⟩⟩, .relation 78774 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35955⟩⟩]⟩, (1)⟩)

def event78778 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35619⟩⟩, .relation 78774 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact78779RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36779⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35955⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78779RawTermsValid :
    exact78779RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78779 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35619⟩⟩) exact78779RawTerms .large 78611 (.finite 202072841853861888) (some (78613))

def event78780 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36782⟩⟩) 0 ⟨35619⟩ 78779

def event78781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36782⟩⟩) 1 ⟨36781⟩ 78601

def event78782 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36782⟩⟩) (.sum [.predecessor 0 78780 .coefficient, .predecessor 1 78781 .coefficient])

def event78783 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36782⟩⟩, .operator (⟨78779, 0⟩, ⟨78601, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36779⟩⟩]⟩, (1)⟩)

def event78784 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36782⟩⟩, .operator (⟨78779, 2⟩, ⟨78601, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35955⟩⟩]⟩, (-1)⟩)

def event78785 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36782⟩⟩) (.sum [.result 78779 .summary, .result 78601 .summary])

def exact78786RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78786RawTermsValid :
    exact78786RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78786 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36782⟩⟩) exact78786RawTerms .large 78782 (.finite 32192539770951767057087530795008) (some (78785))

def event78787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30293⟩⟩) 0 ⟨29137⟩ 3241

def event78788 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30293⟩⟩) (.authority (.programFamilyFact))

def event78789 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨30293⟩⟩) (.finite 3720)

def event78790 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30295⟩⟩) 0 ⟨7177⟩ 15500

def event78791 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30295⟩⟩) 1 ⟨30293⟩ 78789

def event78792 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30295⟩⟩) (.authority (.operator))

def exact78793RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨30295⟩⟩]⟩, (1)⟩]

theorem exact78793RawTermsValid :
    exact78793RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78793 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30295⟩⟩) exact78793RawTerms .large 78792 .exactZero (none)

def event78794 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31119⟩⟩) 0 ⟨30295⟩ 78793

def event78795 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31119⟩⟩) (.authority (.operator))

def exact78796RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨31119⟩⟩]⟩, (1)⟩]

theorem exact78796RawTermsValid :
    exact78796RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78796 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31119⟩⟩) exact78796RawTerms (.finite 8192) 78795 .exactZero (none)

def event78797 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30124⟩⟩) 0 ⟨28920⟩ 3235

def event78798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30124⟩⟩) (.authority (.programFamilyFact))

def event78799 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨30124⟩⟩) (.finite 3720)

def event78800 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30125⟩⟩) 0 ⟨7177⟩ 15500

def event78801 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30125⟩⟩) 1 ⟨30124⟩ 78799

def event78802 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30125⟩⟩) (.authority (.operator))

def exact78803RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨30125⟩⟩]⟩, (1)⟩]

theorem exact78803RawTermsValid :
    exact78803RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78803 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30125⟩⟩) exact78803RawTerms .large 78802 .exactZero (none)

def event78804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨30665⟩⟩) 0 ⟨30125⟩ 78803

def event78805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨30665⟩⟩) (.authority (.operator))

def exact78806RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨30665⟩⟩]⟩, (1)⟩]

theorem exact78806RawTermsValid :
    exact78806RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78806 : Event := .resultExact (⟨.program ⟨257⟩, ⟨30665⟩⟩) exact78806RawTerms (.finite 8192) 78805 .exactZero (none)

def event78807 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28921⟩⟩) 0 ⟨28918⟩ 3224

def event78808 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28921⟩⟩) 1 ⟨10328⟩ 75903

def event78809 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28921⟩⟩) (.tensor (.predecessor 0 78807 .coefficient) (.predecessor 1 78808 .coefficient) true false)

def event78810 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28921⟩⟩, .operator (⟨3224, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact78811RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact78811RawTermsValid :
    exact78811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78811 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28921⟩⟩) exact78811RawTerms .large 78809 .exactZero (none)

def event78812 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10337⟩⟩) 0 ⟨10327⟩ 75773

def event78813 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10337⟩⟩) 1 ⟨7279⟩ 20086

def event78814 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10337⟩⟩) (.product (.predecessor 0 78812 .coefficient) (.predecessor 1 78813 .coefficient) (⟨false, false, none, none, none⟩))

def event78815 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10337⟩⟩, .operator (⟨75773, 0⟩, ⟨20086, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩, (1)⟩)

def exact78816RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩, (1)⟩]

theorem exact78816RawTermsValid :
    exact78816RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78816 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10337⟩⟩) exact78816RawTerms .large 78814 .exactZero (none)

def event78817 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28922⟩⟩) 0 ⟨10337⟩ 78816

def event78818 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28922⟩⟩) 1 ⟨28921⟩ 78811

def event78819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28922⟩⟩) (.sum [.predecessor 0 78817 .coefficient, .predecessor 1 78818 .coefficient])

def exact78820RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78820RawTermsValid :
    exact78820RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78820 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28922⟩⟩) exact78820RawTerms .large 78819 .exactZero (none)

def event78821 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28923⟩⟩) 0 ⟨28922⟩ 78820

def event78822 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28923⟩⟩) 1 ⟨105⟩ 20078

def event78823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28923⟩⟩) (.sum [.predecessor 0 78821 .coefficient, .predecessor 1 78822 .coefficient])

def event78824 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28923⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨105⟩⟩]⟩) [⟨.result 20078 .coefficient, false, none⟩])

def event78825 : Event := .survivorFold (1) 78824

def exact78826RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78826RawTermsValid :
    exact78826RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78826 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28923⟩⟩) exact78826RawTerms .large 78823 (.finite 26) (some (78824))

def event78827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28924⟩⟩) 0 ⟨28923⟩ 78826

def event78828 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28924⟩⟩) 1 ⟨13371⟩ 3227

def event78829 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28924⟩⟩) (.product (.predecessor 0 78827 .coefficient) (.predecessor 1 78828 .coefficient) (⟨false, true, none, none, some 1⟩))

def event78830 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28924⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13371⟩⟩], []⟩) [⟨.result 3227 .coefficient, true, some 1⟩])

def event78831 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28924⟩⟩) (.product (.result 78826 .summary) (.transfer 78830) (⟨false, false, none, none, none⟩))

def event78832 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28924⟩⟩, .operator (⟨78826, 1⟩, ⟨3227, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event78833 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28924⟩⟩, .operator (⟨78826, 0⟩, ⟨3227, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩, (1)⟩)

def exact78834RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩, ⟨.program ⟨257⟩, ⟨28918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78834RawTermsValid :
    exact78834RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78834 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28924⟩⟩) exact78834RawTerms .large 78829 (.finite 30670848) (some (78831))

def event78835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13372⟩⟩) 0 ⟨13371⟩ 3227

def event78836 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13372⟩⟩) 1 ⟨10328⟩ 75903

def event78837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13372⟩⟩) (.tensor (.predecessor 0 78835 .coefficient) (.predecessor 1 78836 .coefficient) true false)

def event78838 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13372⟩⟩, .operator (⟨3227, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact78839RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13371⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact78839RawTermsValid :
    exact78839RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78839 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13372⟩⟩) exact78839RawTerms .large 78837 .exactZero (none)

def event78840 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10354⟩⟩) 0 ⟨10327⟩ 75773

def event78841 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10354⟩⟩) 1 ⟨7296⟩ 20127

def event78842 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10354⟩⟩) (.product (.predecessor 0 78840 .coefficient) (.predecessor 1 78841 .coefficient) (⟨false, false, none, none, none⟩))

def event78843 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10354⟩⟩, .operator (⟨75773, 0⟩, ⟨20127, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩]⟩, (1)⟩)

def exact78844RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩]⟩, (1)⟩]

theorem exact78844RawTermsValid :
    exact78844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78844 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10354⟩⟩) exact78844RawTerms .large 78842 .exactZero (none)

def event78845 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13373⟩⟩) 0 ⟨10354⟩ 78844

def event78846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13373⟩⟩) 1 ⟨13372⟩ 78839

def event78847 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13373⟩⟩) (.sum [.predecessor 0 78845 .coefficient, .predecessor 1 78846 .coefficient])

def eventLeaf4912 : Array AnnotatedEvent := #[
  { event := event78592
    frameStart := 0 },
  { event := event78593
    frameStart := 0 },
  { event := event78594
    frameStart := 0 },
  { event := event78595
    frameStart := 0 },
  { event := event78596
    frameStart := 0 },
  { event := event78597
    frameStart := 0 },
  { event := event78598
    frameStart := 0 },
  { event := event78599
    frameStart := 0 },
  { event := event78600
    frameStart := 0 },
  { event := event78601
    frameStart := 0 },
  { event := event78602
    frameStart := 0 },
  { event := event78603
    frameStart := 0 },
  { event := event78604
    frameStart := 0 },
  { event := event78605
    frameStart := 0 },
  { event := event78606
    frameStart := 0 },
  { event := event78607
    frameStart := 0 }
]

def eventLeaf4913 : Array AnnotatedEvent := #[
  { event := event78608
    frameStart := 0 },
  { event := event78609
    frameStart := 0 },
  { event := event78610
    frameStart := 0 },
  { event := event78611
    frameStart := 0 },
  { event := event78612
    frameStart := 0 },
  { event := event78613
    frameStart := 0 },
  { event := event78614
    frameStart := 0 },
  { event := event78615
    frameStart := 78615 },
  { event := event78616
    frameStart := 78615 },
  { event := event78617
    frameStart := 78615 },
  { event := event78618
    frameStart := 78615 },
  { event := event78619
    frameStart := 78615 },
  { event := event78620
    frameStart := 78615 },
  { event := event78621
    frameStart := 78615 },
  { event := event78622
    frameStart := 78615 },
  { event := event78623
    frameStart := 78615 }
]

def eventLeaf4914 : Array AnnotatedEvent := #[
  { event := event78624
    frameStart := 78615 },
  { event := event78625
    frameStart := 78615 },
  { event := event78626
    frameStart := 78615 },
  { event := event78627
    frameStart := 78615 },
  { event := event78628
    frameStart := 78615 },
  { event := event78629
    frameStart := 78615 },
  { event := event78630
    frameStart := 78615 },
  { event := event78631
    frameStart := 78615 },
  { event := event78632
    frameStart := 78615 },
  { event := event78633
    frameStart := 78615 },
  { event := event78634
    frameStart := 78615 },
  { event := event78635
    frameStart := 78615 },
  { event := event78636
    frameStart := 78615 },
  { event := event78637
    frameStart := 78615 },
  { event := event78638
    frameStart := 78615 },
  { event := event78639
    frameStart := 78615 }
]

def eventLeaf4915 : Array AnnotatedEvent := #[
  { event := event78640
    frameStart := 78615 },
  { event := event78641
    frameStart := 78615 },
  { event := event78642
    frameStart := 78615 },
  { event := event78643
    frameStart := 78615 },
  { event := event78644
    frameStart := 78615 },
  { event := event78645
    frameStart := 78615 },
  { event := event78646
    frameStart := 78615 },
  { event := event78647
    frameStart := 78615 },
  { event := event78648
    frameStart := 78615 },
  { event := event78649
    frameStart := 78615 },
  { event := event78650
    frameStart := 78615 },
  { event := event78651
    frameStart := 78615 },
  { event := event78652
    frameStart := 78615 },
  { event := event78653
    frameStart := 78615 },
  { event := event78654
    frameStart := 78615 },
  { event := event78655
    frameStart := 78615 }
]

def eventLeaf4916 : Array AnnotatedEvent := #[
  { event := event78656
    frameStart := 78615 },
  { event := event78657
    frameStart := 78615 },
  { event := event78658
    frameStart := 78615 },
  { event := event78659
    frameStart := 78615 },
  { event := event78660
    frameStart := 78615 },
  { event := event78661
    frameStart := 78615 },
  { event := event78662
    frameStart := 78615 },
  { event := event78663
    frameStart := 78615 },
  { event := event78664
    frameStart := 78615 },
  { event := event78665
    frameStart := 78615 },
  { event := event78666
    frameStart := 78615 },
  { event := event78667
    frameStart := 78615 },
  { event := event78668
    frameStart := 78615 },
  { event := event78669
    frameStart := 78669 },
  { event := event78670
    frameStart := 78669 },
  { event := event78671
    frameStart := 78669 }
]

def eventLeaf4917 : Array AnnotatedEvent := #[
  { event := event78672
    frameStart := 78669 },
  { event := event78673
    frameStart := 78669 },
  { event := event78674
    frameStart := 78669 },
  { event := event78675
    frameStart := 78669 },
  { event := event78676
    frameStart := 78669 },
  { event := event78677
    frameStart := 78669 },
  { event := event78678
    frameStart := 78669 },
  { event := event78679
    frameStart := 78669 },
  { event := event78680
    frameStart := 78669 },
  { event := event78681
    frameStart := 78669 },
  { event := event78682
    frameStart := 78669 },
  { event := event78683
    frameStart := 78669 },
  { event := event78684
    frameStart := 78669 },
  { event := event78685
    frameStart := 78669 },
  { event := event78686
    frameStart := 78669 },
  { event := event78687
    frameStart := 78669 }
]

def eventLeaf4918 : Array AnnotatedEvent := #[
  { event := event78688
    frameStart := 78669 },
  { event := event78689
    frameStart := 78669 },
  { event := event78690
    frameStart := 78669 },
  { event := event78691
    frameStart := 78669 },
  { event := event78692
    frameStart := 78669 },
  { event := event78693
    frameStart := 78669 },
  { event := event78694
    frameStart := 78669 },
  { event := event78695
    frameStart := 78669 },
  { event := event78696
    frameStart := 78669 },
  { event := event78697
    frameStart := 78669 },
  { event := event78698
    frameStart := 78669 },
  { event := event78699
    frameStart := 78669 },
  { event := event78700
    frameStart := 78669 },
  { event := event78701
    frameStart := 78669 },
  { event := event78702
    frameStart := 78669 },
  { event := event78703
    frameStart := 78669 }
]

def eventLeaf4919 : Array AnnotatedEvent := #[
  { event := event78704
    frameStart := 78669 },
  { event := event78705
    frameStart := 78669 },
  { event := event78706
    frameStart := 78669 },
  { event := event78707
    frameStart := 78669 },
  { event := event78708
    frameStart := 78669 },
  { event := event78709
    frameStart := 78669 },
  { event := event78710
    frameStart := 78669 },
  { event := event78711
    frameStart := 78669 },
  { event := event78712
    frameStart := 78669 },
  { event := event78713
    frameStart := 78669 },
  { event := event78714
    frameStart := 78669 },
  { event := event78715
    frameStart := 78669 },
  { event := event78716
    frameStart := 78669 },
  { event := event78717
    frameStart := 78669 },
  { event := event78718
    frameStart := 78669 },
  { event := event78719
    frameStart := 78669 }
]

def eventLeaf4920 : Array AnnotatedEvent := #[
  { event := event78720
    frameStart := 78669 },
  { event := event78721
    frameStart := 78669 },
  { event := event78722
    frameStart := 78669 },
  { event := event78723
    frameStart := 78669 },
  { event := event78724
    frameStart := 78669 },
  { event := event78725
    frameStart := 78669 },
  { event := event78726
    frameStart := 78669 },
  { event := event78727
    frameStart := 78669 },
  { event := event78728
    frameStart := 78669 },
  { event := event78729
    frameStart := 78669 },
  { event := event78730
    frameStart := 78669 },
  { event := event78731
    frameStart := 78669 },
  { event := event78732
    frameStart := 78669 },
  { event := event78733
    frameStart := 78669 },
  { event := event78734
    frameStart := 78669 },
  { event := event78735
    frameStart := 78669 }
]

def eventLeaf4921 : Array AnnotatedEvent := #[
  { event := event78736
    frameStart := 78669 },
  { event := event78737
    frameStart := 78669 },
  { event := event78738
    frameStart := 78669 },
  { event := event78739
    frameStart := 78669 },
  { event := event78740
    frameStart := 78669 },
  { event := event78741
    frameStart := 78669 },
  { event := event78742
    frameStart := 78669 },
  { event := event78743
    frameStart := 78669 },
  { event := event78744
    frameStart := 78669 },
  { event := event78745
    frameStart := 78669 },
  { event := event78746
    frameStart := 78669 },
  { event := event78747
    frameStart := 78669 },
  { event := event78748
    frameStart := 78669 },
  { event := event78749
    frameStart := 78669 },
  { event := event78750
    frameStart := 78669 },
  { event := event78751
    frameStart := 78669 }
]

def eventLeaf4922 : Array AnnotatedEvent := #[
  { event := event78752
    frameStart := 78669 },
  { event := event78753
    frameStart := 78669 },
  { event := event78754
    frameStart := 78669 },
  { event := event78755
    frameStart := 78669 },
  { event := event78756
    frameStart := 78669 },
  { event := event78757
    frameStart := 78669 },
  { event := event78758
    frameStart := 78669 },
  { event := event78759
    frameStart := 78669 },
  { event := event78760
    frameStart := 78669 },
  { event := event78761
    frameStart := 78669 },
  { event := event78762
    frameStart := 78669 },
  { event := event78763
    frameStart := 78669 },
  { event := event78764
    frameStart := 78669 },
  { event := event78765
    frameStart := 78669 },
  { event := event78766
    frameStart := 78669 },
  { event := event78767
    frameStart := 78669 }
]

def eventLeaf4923 : Array AnnotatedEvent := #[
  { event := event78768
    frameStart := 78669 },
  { event := event78769
    frameStart := 78669 },
  { event := event78770
    frameStart := 78669 },
  { event := event78771
    frameStart := 78669 },
  { event := event78772
    frameStart := 78669 },
  { event := event78773
    frameStart := 0 },
  { event := event78774
    frameStart := 0 },
  { event := event78775
    frameStart := 0 },
  { event := event78776
    frameStart := 0 },
  { event := event78777
    frameStart := 0 },
  { event := event78778
    frameStart := 0 },
  { event := event78779
    frameStart := 0 },
  { event := event78780
    frameStart := 0 },
  { event := event78781
    frameStart := 0 },
  { event := event78782
    frameStart := 0 },
  { event := event78783
    frameStart := 0 }
]

def eventLeaf4924 : Array AnnotatedEvent := #[
  { event := event78784
    frameStart := 0 },
  { event := event78785
    frameStart := 0 },
  { event := event78786
    frameStart := 0 },
  { event := event78787
    frameStart := 0 },
  { event := event78788
    frameStart := 0 },
  { event := event78789
    frameStart := 0 },
  { event := event78790
    frameStart := 0 },
  { event := event78791
    frameStart := 0 },
  { event := event78792
    frameStart := 0 },
  { event := event78793
    frameStart := 0 },
  { event := event78794
    frameStart := 0 },
  { event := event78795
    frameStart := 0 },
  { event := event78796
    frameStart := 0 },
  { event := event78797
    frameStart := 0 },
  { event := event78798
    frameStart := 0 },
  { event := event78799
    frameStart := 0 }
]

def eventLeaf4925 : Array AnnotatedEvent := #[
  { event := event78800
    frameStart := 0 },
  { event := event78801
    frameStart := 0 },
  { event := event78802
    frameStart := 0 },
  { event := event78803
    frameStart := 0 },
  { event := event78804
    frameStart := 0 },
  { event := event78805
    frameStart := 0 },
  { event := event78806
    frameStart := 0 },
  { event := event78807
    frameStart := 0 },
  { event := event78808
    frameStart := 0 },
  { event := event78809
    frameStart := 0 },
  { event := event78810
    frameStart := 0 },
  { event := event78811
    frameStart := 0 },
  { event := event78812
    frameStart := 0 },
  { event := event78813
    frameStart := 0 },
  { event := event78814
    frameStart := 0 },
  { event := event78815
    frameStart := 0 }
]

def eventLeaf4926 : Array AnnotatedEvent := #[
  { event := event78816
    frameStart := 0 },
  { event := event78817
    frameStart := 0 },
  { event := event78818
    frameStart := 0 },
  { event := event78819
    frameStart := 0 },
  { event := event78820
    frameStart := 0 },
  { event := event78821
    frameStart := 0 },
  { event := event78822
    frameStart := 0 },
  { event := event78823
    frameStart := 0 },
  { event := event78824
    frameStart := 0 },
  { event := event78825
    frameStart := 0 },
  { event := event78826
    frameStart := 0 },
  { event := event78827
    frameStart := 0 },
  { event := event78828
    frameStart := 0 },
  { event := event78829
    frameStart := 0 },
  { event := event78830
    frameStart := 0 },
  { event := event78831
    frameStart := 0 }
]

def eventLeaf4927 : Array AnnotatedEvent := #[
  { event := event78832
    frameStart := 0 },
  { event := event78833
    frameStart := 0 },
  { event := event78834
    frameStart := 0 },
  { event := event78835
    frameStart := 0 },
  { event := event78836
    frameStart := 0 },
  { event := event78837
    frameStart := 0 },
  { event := event78838
    frameStart := 0 },
  { event := event78839
    frameStart := 0 },
  { event := event78840
    frameStart := 0 },
  { event := event78841
    frameStart := 0 },
  { event := event78842
    frameStart := 0 },
  { event := event78843
    frameStart := 0 },
  { event := event78844
    frameStart := 0 },
  { event := event78845
    frameStart := 0 },
  { event := event78846
    frameStart := 0 },
  { event := event78847
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events307
