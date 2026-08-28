import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events307

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event78592 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event78593 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event78594 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event78595 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event78596 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event78597 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event78598 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event78599 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event78600 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 78599

def event78601 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 78597

def event78602 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 78600 .coefficient) (.value (.predecessor 1 78601 .coefficient)))

def event78603 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event78604 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 78603

def event78605 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 78595

def event78606 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 78604 .coefficient, .predecessor 1 78605 .coefficient])

def event78607 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event78608 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 78607

def event78609 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 78593

def event78610 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 78609 .coefficient))

def event78611 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event78612 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11129⟩⟩) 0 ⟨5530⟩ 78611

def event78613 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11129⟩⟩) (.authority (.programFamilyFact))

def exact78614RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩], []⟩, (1)⟩]

theorem exact78614RawTermsValid :
    exact78614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78614 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11129⟩⟩) exact78614RawTerms (.finite 6) 78613 .exactZero (none)

def event78615 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12154⟩⟩) 0 ⟨5530⟩ 78611

def event78616 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12154⟩⟩) (.authority (.programFamilyFact))

def exact78617RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12154⟩⟩], []⟩, (1)⟩]

theorem exact78617RawTermsValid :
    exact78617RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78617 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12154⟩⟩) exact78617RawTerms (.finite 6) 78616 .exactZero (none)

def event78618 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12155⟩⟩) 0 ⟨12154⟩ 78617

def event78619 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12155⟩⟩) 1 ⟨11129⟩ 78614

def event78620 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12155⟩⟩) (.product (.predecessor 0 78618 .coefficient) (.predecessor 1 78619 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event78621 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12155⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], []⟩) [⟨.result 78617 .coefficient, true, some 1⟩, ⟨.result 78614 .coefficient, true, some 1⟩])

def event78622 : Event := .survivorFold (1) 78621

def exact78623RawTerms : List Term := []

theorem exact78623RawTermsValid :
    exact78623RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78623 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12155⟩⟩) exact78623RawTerms (.finite 36) 78620 (.finite 36) (some (78621))

def event78624 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12156⟩⟩) 0 ⟨12155⟩ 78623

def event78625 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12156⟩⟩) (.identity (.predecessor 0 78624 .coefficient))

def event78626 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12156⟩⟩) (.finite 36)

def event78627 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15418⟩⟩) 0 ⟨12156⟩ 78626

def event78628 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15418⟩⟩) (.authority (.programFamilyFact))

def exact78629RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], []⟩, (1)⟩]

theorem exact78629RawTermsValid :
    exact78629RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78629 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15418⟩⟩) exact78629RawTerms (.finite 6) 78628 .exactZero (none)

def event78630 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15419⟩⟩) 0 ⟨15418⟩ 78629

def event78631 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15419⟩⟩) (.identity (.predecessor 0 78630 .coefficient))

def event78632 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15419⟩⟩) (.finite 6)

def event78633 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20748⟩⟩) 0 ⟨15419⟩ 78632

def event78634 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20748⟩⟩) (.authority (.relationPreimageSource ⟨34⟩))

def exact78635RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20748⟩⟩]⟩, (1)⟩]

theorem exact78635RawTermsValid :
    exact78635RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78635 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20748⟩⟩) exact78635RawTerms (.finite 136065468) 78634 .exactZero (none)

def event78636 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact78637RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact78637RawTermsValid :
    exact78637RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78637 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact78637RawTerms .large 78636 .exactZero (none)

def event78638 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20749⟩⟩) 0 ⟨6⟩ 78637

def event78639 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20749⟩⟩) 1 ⟨20748⟩ 78635

def event78640 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20749⟩⟩) (.product (.predecessor 0 78638 .coefficient) (.predecessor 1 78639 .coefficient) (⟨false, false, none, none, none⟩))

def event78641 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20749⟩⟩, .operator (⟨78637, 0⟩, ⟨78635, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20748⟩⟩]⟩, (1)⟩)

def exact78642RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20748⟩⟩]⟩, (1)⟩]

theorem exact78642RawTermsValid :
    exact78642RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78642 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20749⟩⟩) exact78642RawTerms .large 78640 .exactZero (none)

def event78643 : Event := .preFoldPolynomial 78642 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20748⟩⟩]⟩, (1)⟩] .exactZero none

def exact78644RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20748⟩⟩]⟩, (1)⟩]

def event78644 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20749⟩⟩) 78643 exact78644RawTerms .large 78640 .exactZero (none)

def event78645 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26984⟩⟩)

def event78646 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event78647 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event78648 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event78649 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event78650 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event78651 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event78652 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event78653 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event78654 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 78653

def event78655 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 78651

def event78656 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 78654 .coefficient) (.value (.predecessor 1 78655 .coefficient)))

def event78657 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event78658 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 78657

def event78659 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 78649

def event78660 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 78658 .coefficient, .predecessor 1 78659 .coefficient])

def event78661 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event78662 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 78661

def event78663 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 78647

def event78664 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 78663 .coefficient))

def event78665 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event78666 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11129⟩⟩) 0 ⟨5530⟩ 78665

def event78667 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11129⟩⟩) (.authority (.programFamilyFact))

def exact78668RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩], []⟩, (1)⟩]

theorem exact78668RawTermsValid :
    exact78668RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78668 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11129⟩⟩) exact78668RawTerms (.finite 6) 78667 .exactZero (none)

def event78669 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12154⟩⟩) 0 ⟨5530⟩ 78665

def event78670 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12154⟩⟩) (.authority (.programFamilyFact))

def exact78671RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12154⟩⟩], []⟩, (1)⟩]

theorem exact78671RawTermsValid :
    exact78671RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78671 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12154⟩⟩) exact78671RawTerms (.finite 6) 78670 .exactZero (none)

def event78672 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12155⟩⟩) 0 ⟨12154⟩ 78671

def event78673 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12155⟩⟩) 1 ⟨11129⟩ 78668

def event78674 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12155⟩⟩) (.product (.predecessor 0 78672 .coefficient) (.predecessor 1 78673 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event78675 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12155⟩⟩, .operator (⟨78671, 0⟩, ⟨78668, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], []⟩, (1)⟩)

def exact78676RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11129⟩⟩, ⟨.program ⟨214⟩, ⟨12154⟩⟩], []⟩, (1)⟩]

theorem exact78676RawTermsValid :
    exact78676RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78676 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12155⟩⟩) exact78676RawTerms (.finite 36) 78674 .exactZero (none)

def event78677 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12156⟩⟩) 0 ⟨12155⟩ 78676

def event78678 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12156⟩⟩) (.identity (.predecessor 0 78677 .coefficient))

def event78679 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12156⟩⟩) (.finite 36)

def event78680 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15418⟩⟩) 0 ⟨12156⟩ 78679

def event78681 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15418⟩⟩) (.authority (.programFamilyFact))

def exact78682RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], []⟩, (1)⟩]

theorem exact78682RawTermsValid :
    exact78682RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78682 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15418⟩⟩) exact78682RawTerms (.finite 6) 78681 .exactZero (none)

def event78683 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15419⟩⟩) 0 ⟨15418⟩ 78682

def event78684 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15419⟩⟩) (.identity (.predecessor 0 78683 .coefficient))

def event78685 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15419⟩⟩) (.finite 6)

def event78686 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23905⟩⟩) 0 ⟨15419⟩ 78685

def event78687 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23905⟩⟩) (.authority (.programFamilyFact))

def event78688 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23905⟩⟩) (.finite 3720)

def event78689 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event78690 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23906⟩⟩) 0 ⟨6689⟩ 78689

def event78691 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23906⟩⟩) 1 ⟨23905⟩ 78688

def event78692 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23906⟩⟩) (.authority (.operator))

def exact78693RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23906⟩⟩]⟩, (1)⟩]

theorem exact78693RawTermsValid :
    exact78693RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78693 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23906⟩⟩) exact78693RawTerms .large 78692 .exactZero (none)

def event78694 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26978⟩⟩) 0 ⟨23906⟩ 78693

def event78695 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26978⟩⟩) (.authority (.operator))

def exact78696RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26978⟩⟩]⟩, (1)⟩]

theorem exact78696RawTermsValid :
    exact78696RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78696 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26978⟩⟩) exact78696RawTerms (.finite 8192) 78695 .exactZero (none)

def event78697 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event78698 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event78699 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15458⟩⟩) 0 ⟨15419⟩ 78685

def event78700 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15458⟩⟩) 1 ⟨110⟩ 78698

def event78701 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15458⟩⟩) (.sum [.predecessor 0 78699 .coefficient, .predecessor 1 78700 .coefficient])

def event78702 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15458⟩⟩) (.finite 6)

def event78703 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15459⟩⟩) 0 ⟨15458⟩ 78702

def event78704 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15459⟩⟩) (.identity (.predecessor 0 78703 .coefficient))

def exact78705RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], []⟩, (1)⟩]

theorem exact78705RawTermsValid :
    exact78705RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78705 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15459⟩⟩) exact78705RawTerms (.finite 6) 78704 .exactZero (none)

def event78706 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact78707RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact78707RawTermsValid :
    exact78707RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78707 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact78707RawTerms .large 78706 .exactZero (none)

def event78708 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15460⟩⟩) 0 ⟨6544⟩ 78707

def event78709 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15460⟩⟩) 1 ⟨15459⟩ 78705

def event78710 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15460⟩⟩) (.product (.predecessor 0 78708 .coefficient) (.predecessor 1 78709 .coefficient) (⟨false, false, none, none, none⟩))

def event78711 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15460⟩⟩, .operator (⟨78707, 0⟩, ⟨78705, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact78712RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact78712RawTermsValid :
    exact78712RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78712 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15460⟩⟩) exact78712RawTerms .large 78710 .exactZero (none)

def event78713 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6693⟩⟩) 0 ⟨6689⟩ 78689

def event78714 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6693⟩⟩) (.authority (.operator))

def exact78715RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩]

theorem exact78715RawTermsValid :
    exact78715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78715 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6693⟩⟩) exact78715RawTerms .large 78714 .exactZero (none)

def event78716 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15461⟩⟩) 0 ⟨6693⟩ 78715

def event78717 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15461⟩⟩) 1 ⟨15460⟩ 78712

def event78718 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15461⟩⟩) (.sum [.predecessor 0 78716 .coefficient, .predecessor 1 78717 .coefficient])

def exact78719RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78719RawTermsValid :
    exact78719RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78719 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15461⟩⟩) exact78719RawTerms .large 78718 .exactZero (none)

def event78720 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26979⟩⟩) 0 ⟨15461⟩ 78719

def event78721 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26979⟩⟩) 1 ⟨26978⟩ 78696

def event78722 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26979⟩⟩) (.product (.predecessor 0 78720 .coefficient) (.predecessor 1 78721 .coefficient) (⟨false, false, none, none, none⟩))

def event78723 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26979⟩⟩, .operator (⟨78719, 0⟩, ⟨78696, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26978⟩⟩]⟩, (1)⟩)

def event78724 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26979⟩⟩, .operator (⟨78719, 1⟩, ⟨78696, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26978⟩⟩]⟩, (-1)⟩)

def event78725 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26979⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26978⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26978⟩⟩) ⟨23906⟩ 78693)

def event78726 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26979⟩⟩, .relation 78725 0, ⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨23906⟩⟩]⟩, (-1)⟩)

def exact78727RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26978⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨23906⟩⟩]⟩, (-1)⟩]

theorem exact78727RawTermsValid :
    exact78727RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78727 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26979⟩⟩) exact78727RawTerms .large 78722 .exactZero (none)

def event78728 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15511⟩⟩) 0 ⟨15419⟩ 78685

def event78729 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15511⟩⟩) (.authority (.programFamilyFact))

def exact78730RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15511⟩⟩], []⟩, (1)⟩]

theorem exact78730RawTermsValid :
    exact78730RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78730 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15511⟩⟩) exact78730RawTerms (.finite 6) 78729 .exactZero (none)

def event78731 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15514⟩⟩) 0 ⟨6544⟩ 78707

def event78732 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15514⟩⟩) 1 ⟨15511⟩ 78730

def event78733 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15514⟩⟩) (.product (.predecessor 0 78731 .coefficient) (.predecessor 1 78732 .coefficient) (⟨false, true, none, none, some 1⟩))

def event78734 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15514⟩⟩, .operator (⟨78707, 0⟩, ⟨78730, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact78735RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact78735RawTermsValid :
    exact78735RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78735 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15514⟩⟩) exact78735RawTerms .large 78733 .exactZero (none)

def event78736 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6714⟩⟩) 0 ⟨6689⟩ 78689

def event78737 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6714⟩⟩) (.authority (.operator))

def exact78738RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩]⟩, (1)⟩]

theorem exact78738RawTermsValid :
    exact78738RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78738 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6714⟩⟩) exact78738RawTerms .large 78737 .exactZero (none)

def event78739 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15515⟩⟩) 0 ⟨6714⟩ 78738

def event78740 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15515⟩⟩) 1 ⟨15514⟩ 78735

def event78741 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15515⟩⟩) (.sum [.predecessor 0 78739 .coefficient, .predecessor 1 78740 .coefficient])

def exact78742RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78742RawTermsValid :
    exact78742RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78742 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15515⟩⟩) exact78742RawTerms .large 78741 .exactZero (none)

def event78743 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26984⟩⟩) 0 ⟨15515⟩ 78742

def event78744 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26984⟩⟩) 1 ⟨26979⟩ 78727

def event78745 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26984⟩⟩) (.sum [.predecessor 0 78743 .coefficient, .predecessor 1 78744 .coefficient])

def exact78746RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26978⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨23906⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78746RawTermsValid :
    exact78746RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78746 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26984⟩⟩) exact78746RawTerms .large 78745 .exactZero (none)

def event78747 : Event := .preFoldPolynomial 78746 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26978⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨23906⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact78748RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26978⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨23906⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event78748 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26984⟩⟩) 78747 exact78748RawTerms .large 78745 .exactZero (none)

def event78749 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15419⟩⟩) ⟨⟨127⟩, ⟨34⟩, ⟨109⟩⟩ ⟨78591, 78749⟩

def event78750 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20751⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20748⟩⟩]⟩) (1) 0 2 (.universal 78749 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20748⟩⟩]⟩) (none) 78748)

def event78751 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20751⟩⟩, .relation 78750 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩]⟩, (1)⟩)

def event78752 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20751⟩⟩, .relation 78750 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26978⟩⟩]⟩, (-1)⟩)

def event78753 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20751⟩⟩, .relation 78750 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨23906⟩⟩]⟩, (1)⟩)

def event78754 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20751⟩⟩, .relation 78750 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact78755RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26978⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨23906⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78755RawTermsValid :
    exact78755RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78755 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20751⟩⟩) exact78755RawTerms .large 78587 (.finite 1811303510016) (some (78589))

def event78756 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26981⟩⟩) 0 ⟨20751⟩ 78755

def event78757 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26981⟩⟩) 1 ⟨26980⟩ 78577

def event78758 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26981⟩⟩) (.sum [.predecessor 0 78756 .coefficient, .predecessor 1 78757 .coefficient])

def event78759 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26981⟩⟩, .operator (⟨78755, 0⟩, ⟨78577, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨26978⟩⟩]⟩, (1)⟩)

def event78760 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26981⟩⟩, .operator (⟨78755, 2⟩, ⟨78577, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15418⟩⟩], [⟨.program ⟨214⟩, ⟨23906⟩⟩]⟩, (-1)⟩)

def event78761 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26981⟩⟩) (.sum [.result 78755 .summary, .result 78577 .summary])

def exact78762RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78762RawTermsValid :
    exact78762RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78762 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26981⟩⟩) exact78762RawTerms .large 78758 (.finite 1291933999269462814720) (some (78761))

def event78763 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26982⟩⟩) 0 ⟨26981⟩ 78762

def event78764 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26982⟩⟩) 1 ⟨6656⟩ 5799

def event78765 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26982⟩⟩) (.product (.predecessor 0 78763 .coefficient) (.predecessor 1 78764 .coefficient) (⟨false, false, none, none, none⟩))

def event78766 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26982⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩) [⟨.result 5795 .coefficient, false, none⟩])

def event78767 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26982⟩⟩) (.product (.result 78762 .summary) (.transfer 78766) (⟨false, false, none, none, none⟩))

def event78768 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26982⟩⟩, .operator (⟨78762, 0⟩, ⟨5799, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩)

def event78769 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26982⟩⟩, .operator (⟨78762, 1⟩, ⟨5799, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (-1)⟩)

def event78770 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26982⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6655⟩⟩) ⟨6599⟩ 5792)

def event78771 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26982⟩⟩, .relation 78770 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact78772RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact78772RawTermsValid :
    exact78772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78772 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26982⟩⟩) exact78772RawTerms .large 78765 (.finite 4741418448262916841427435520) (some (78767))

def event78773 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23843⟩⟩) 0 ⟨6689⟩ 5477

def event78774 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23843⟩⟩) 1 ⟨23842⟩ 72519

def event78775 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23843⟩⟩) (.authority (.operator))

def exact78776RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23843⟩⟩]⟩, (1)⟩]

theorem exact78776RawTermsValid :
    exact78776RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78776 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23843⟩⟩) exact78776RawTerms .large 78775 .exactZero (none)

def event78777 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26761⟩⟩) 0 ⟨23843⟩ 78776

def event78778 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26761⟩⟩) (.authority (.operator))

def exact78779RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26761⟩⟩]⟩, (1)⟩]

theorem exact78779RawTermsValid :
    exact78779RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78779 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26761⟩⟩) exact78779RawTerms (.finite 8192) 78778 .exactZero (none)

def event78780 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26763⟩⟩) 0 ⟨25062⟩ 72803

def event78781 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26763⟩⟩) 1 ⟨26761⟩ 78779

def event78782 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26763⟩⟩) (.product (.predecessor 0 78780 .coefficient) (.predecessor 1 78781 .coefficient) (⟨false, false, none, none, none⟩))

def event78783 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26763⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26761⟩⟩]⟩) [⟨.result 78779 .coefficient, false, none⟩])

def event78784 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26763⟩⟩) (.product (.result 72803 .summary) (.transfer 78783) (⟨false, false, none, none, none⟩))

def event78785 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26763⟩⟩, .operator (⟨72803, 0⟩, ⟨78779, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26761⟩⟩]⟩, (1)⟩)

def event78786 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26763⟩⟩, .operator (⟨72803, 1⟩, ⟨78779, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26761⟩⟩]⟩, (-1)⟩)

def event78787 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26763⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26761⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26761⟩⟩) ⟨23843⟩ 78776)

def event78788 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26763⟩⟩, .relation 78787 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23843⟩⟩]⟩, (-1)⟩)

def exact78789RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26761⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23843⟩⟩]⟩, (-1)⟩]

theorem exact78789RawTermsValid :
    exact78789RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78789 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26763⟩⟩) exact78789RawTerms .large 78782 (.finite 1291911585013138718720) (some (78784))

def event78790 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20604⟩⟩) 0 ⟨15111⟩ 3448

def event78791 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20604⟩⟩) (.authority (.relationPreimageSource ⟨31⟩))

def exact78792RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20604⟩⟩]⟩, (1)⟩]

theorem exact78792RawTermsValid :
    exact78792RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78792 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20604⟩⟩) exact78792RawTerms (.finite 136065468) 78791 .exactZero (none)

def event78793 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20606⟩⟩) 0 ⟨20604⟩ 78792

def event78794 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20606⟩⟩) 1 ⟨2348⟩ 4

def event78795 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20606⟩⟩) (.scale (.predecessor 0 78793 .coefficient) (.value (.predecessor 1 78794 .coefficient)))

def exact78796RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20604⟩⟩]⟩, (1)⟩]

theorem exact78796RawTermsValid :
    exact78796RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78796 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20606⟩⟩) exact78796RawTerms (.finite 136065468) 78795 .exactZero (none)

def event78797 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20607⟩⟩) 0 ⟨5535⟩ 65387

def event78798 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20607⟩⟩) 1 ⟨20606⟩ 78796

def event78799 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20607⟩⟩) (.product (.predecessor 0 78797 .coefficient) (.predecessor 1 78798 .coefficient) (⟨false, false, none, none, none⟩))

def event78800 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20607⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20604⟩⟩]⟩) [⟨.result 78792 .coefficient, false, none⟩])

def event78801 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20607⟩⟩) (.product (.result 65387 .summary) (.transfer 78800) (⟨false, false, none, none, none⟩))

def event78802 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20607⟩⟩, .operator (⟨65387, 0⟩, ⟨78796, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20604⟩⟩]⟩, (1)⟩)

def event78803 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20605⟩⟩)

def event78804 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event78805 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event78806 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event78807 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event78808 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event78809 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event78810 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event78811 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event78812 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 78811

def event78813 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 78809

def event78814 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 78812 .coefficient) (.value (.predecessor 1 78813 .coefficient)))

def event78815 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event78816 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 78815

def event78817 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 78807

def event78818 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 78816 .coefficient, .predecessor 1 78817 .coefficient])

def event78819 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event78820 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 78819

def event78821 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 78805

def event78822 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 78821 .coefficient))

def event78823 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event78824 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10969⟩⟩) 0 ⟨5530⟩ 78823

def event78825 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10969⟩⟩) (.authority (.programFamilyFact))

def exact78826RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10969⟩⟩], []⟩, (1)⟩]

theorem exact78826RawTermsValid :
    exact78826RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78826 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10969⟩⟩) exact78826RawTerms (.finite 4) 78825 .exactZero (none)

def event78827 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10837⟩⟩) 0 ⟨5530⟩ 78823

def event78828 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10837⟩⟩) (.authority (.programFamilyFact))

def exact78829RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩], []⟩, (1)⟩]

theorem exact78829RawTermsValid :
    exact78829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78829 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10837⟩⟩) exact78829RawTerms (.finite 4) 78828 .exactZero (none)

def event78830 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10970⟩⟩) 0 ⟨10837⟩ 78829

def event78831 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10970⟩⟩) 1 ⟨10969⟩ 78826

def event78832 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10970⟩⟩) (.product (.predecessor 0 78830 .coefficient) (.predecessor 1 78831 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event78833 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10970⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], []⟩) [⟨.result 78829 .coefficient, true, some 1⟩, ⟨.result 78826 .coefficient, true, some 1⟩])

def event78834 : Event := .survivorFold (1) 78833

def exact78835RawTerms : List Term := []

theorem exact78835RawTermsValid :
    exact78835RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78835 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10970⟩⟩) exact78835RawTerms (.finite 16) 78832 (.finite 16) (some (78833))

def event78836 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10971⟩⟩) 0 ⟨10970⟩ 78835

def event78837 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10971⟩⟩) (.identity (.predecessor 0 78836 .coefficient))

def event78838 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10971⟩⟩) (.finite 16)

def event78839 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15110⟩⟩) 0 ⟨10971⟩ 78838

def event78840 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15110⟩⟩) (.authority (.programFamilyFact))

def exact78841RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], []⟩, (1)⟩]

theorem exact78841RawTermsValid :
    exact78841RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78841 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15110⟩⟩) exact78841RawTerms (.finite 4) 78840 .exactZero (none)

def event78842 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15111⟩⟩) 0 ⟨15110⟩ 78841

def event78843 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15111⟩⟩) (.identity (.predecessor 0 78842 .coefficient))

def event78844 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15111⟩⟩) (.finite 4)

def event78845 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20604⟩⟩) 0 ⟨15111⟩ 78844

def event78846 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20604⟩⟩) (.authority (.relationPreimageSource ⟨31⟩))

def exact78847RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20604⟩⟩]⟩, (1)⟩]

theorem exact78847RawTermsValid :
    exact78847RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78847 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20604⟩⟩) exact78847RawTerms (.finite 136065468) 78846 .exactZero (none)

def eventLeaf4912 : Array AnnotatedEvent := #[
  { event := event78592
    frameStart := 78591 },
  { event := event78593
    frameStart := 78591 },
  { event := event78594
    frameStart := 78591 },
  { event := event78595
    frameStart := 78591 },
  { event := event78596
    frameStart := 78591 },
  { event := event78597
    frameStart := 78591 },
  { event := event78598
    frameStart := 78591 },
  { event := event78599
    frameStart := 78591 },
  { event := event78600
    frameStart := 78591 },
  { event := event78601
    frameStart := 78591 },
  { event := event78602
    frameStart := 78591 },
  { event := event78603
    frameStart := 78591 },
  { event := event78604
    frameStart := 78591 },
  { event := event78605
    frameStart := 78591 },
  { event := event78606
    frameStart := 78591 },
  { event := event78607
    frameStart := 78591 }
]

def eventLeaf4913 : Array AnnotatedEvent := #[
  { event := event78608
    frameStart := 78591 },
  { event := event78609
    frameStart := 78591 },
  { event := event78610
    frameStart := 78591 },
  { event := event78611
    frameStart := 78591 },
  { event := event78612
    frameStart := 78591 },
  { event := event78613
    frameStart := 78591 },
  { event := event78614
    frameStart := 78591 },
  { event := event78615
    frameStart := 78591 },
  { event := event78616
    frameStart := 78591 },
  { event := event78617
    frameStart := 78591 },
  { event := event78618
    frameStart := 78591 },
  { event := event78619
    frameStart := 78591 },
  { event := event78620
    frameStart := 78591 },
  { event := event78621
    frameStart := 78591 },
  { event := event78622
    frameStart := 78591 },
  { event := event78623
    frameStart := 78591 }
]

def eventLeaf4914 : Array AnnotatedEvent := #[
  { event := event78624
    frameStart := 78591 },
  { event := event78625
    frameStart := 78591 },
  { event := event78626
    frameStart := 78591 },
  { event := event78627
    frameStart := 78591 },
  { event := event78628
    frameStart := 78591 },
  { event := event78629
    frameStart := 78591 },
  { event := event78630
    frameStart := 78591 },
  { event := event78631
    frameStart := 78591 },
  { event := event78632
    frameStart := 78591 },
  { event := event78633
    frameStart := 78591 },
  { event := event78634
    frameStart := 78591 },
  { event := event78635
    frameStart := 78591 },
  { event := event78636
    frameStart := 78591 },
  { event := event78637
    frameStart := 78591 },
  { event := event78638
    frameStart := 78591 },
  { event := event78639
    frameStart := 78591 }
]

def eventLeaf4915 : Array AnnotatedEvent := #[
  { event := event78640
    frameStart := 78591 },
  { event := event78641
    frameStart := 78591 },
  { event := event78642
    frameStart := 78591 },
  { event := event78643
    frameStart := 78591 },
  { event := event78644
    frameStart := 78591 },
  { event := event78645
    frameStart := 78645 },
  { event := event78646
    frameStart := 78645 },
  { event := event78647
    frameStart := 78645 },
  { event := event78648
    frameStart := 78645 },
  { event := event78649
    frameStart := 78645 },
  { event := event78650
    frameStart := 78645 },
  { event := event78651
    frameStart := 78645 },
  { event := event78652
    frameStart := 78645 },
  { event := event78653
    frameStart := 78645 },
  { event := event78654
    frameStart := 78645 },
  { event := event78655
    frameStart := 78645 }
]

def eventLeaf4916 : Array AnnotatedEvent := #[
  { event := event78656
    frameStart := 78645 },
  { event := event78657
    frameStart := 78645 },
  { event := event78658
    frameStart := 78645 },
  { event := event78659
    frameStart := 78645 },
  { event := event78660
    frameStart := 78645 },
  { event := event78661
    frameStart := 78645 },
  { event := event78662
    frameStart := 78645 },
  { event := event78663
    frameStart := 78645 },
  { event := event78664
    frameStart := 78645 },
  { event := event78665
    frameStart := 78645 },
  { event := event78666
    frameStart := 78645 },
  { event := event78667
    frameStart := 78645 },
  { event := event78668
    frameStart := 78645 },
  { event := event78669
    frameStart := 78645 },
  { event := event78670
    frameStart := 78645 },
  { event := event78671
    frameStart := 78645 }
]

def eventLeaf4917 : Array AnnotatedEvent := #[
  { event := event78672
    frameStart := 78645 },
  { event := event78673
    frameStart := 78645 },
  { event := event78674
    frameStart := 78645 },
  { event := event78675
    frameStart := 78645 },
  { event := event78676
    frameStart := 78645 },
  { event := event78677
    frameStart := 78645 },
  { event := event78678
    frameStart := 78645 },
  { event := event78679
    frameStart := 78645 },
  { event := event78680
    frameStart := 78645 },
  { event := event78681
    frameStart := 78645 },
  { event := event78682
    frameStart := 78645 },
  { event := event78683
    frameStart := 78645 },
  { event := event78684
    frameStart := 78645 },
  { event := event78685
    frameStart := 78645 },
  { event := event78686
    frameStart := 78645 },
  { event := event78687
    frameStart := 78645 }
]

def eventLeaf4918 : Array AnnotatedEvent := #[
  { event := event78688
    frameStart := 78645 },
  { event := event78689
    frameStart := 78645 },
  { event := event78690
    frameStart := 78645 },
  { event := event78691
    frameStart := 78645 },
  { event := event78692
    frameStart := 78645 },
  { event := event78693
    frameStart := 78645 },
  { event := event78694
    frameStart := 78645 },
  { event := event78695
    frameStart := 78645 },
  { event := event78696
    frameStart := 78645 },
  { event := event78697
    frameStart := 78645 },
  { event := event78698
    frameStart := 78645 },
  { event := event78699
    frameStart := 78645 },
  { event := event78700
    frameStart := 78645 },
  { event := event78701
    frameStart := 78645 },
  { event := event78702
    frameStart := 78645 },
  { event := event78703
    frameStart := 78645 }
]

def eventLeaf4919 : Array AnnotatedEvent := #[
  { event := event78704
    frameStart := 78645 },
  { event := event78705
    frameStart := 78645 },
  { event := event78706
    frameStart := 78645 },
  { event := event78707
    frameStart := 78645 },
  { event := event78708
    frameStart := 78645 },
  { event := event78709
    frameStart := 78645 },
  { event := event78710
    frameStart := 78645 },
  { event := event78711
    frameStart := 78645 },
  { event := event78712
    frameStart := 78645 },
  { event := event78713
    frameStart := 78645 },
  { event := event78714
    frameStart := 78645 },
  { event := event78715
    frameStart := 78645 },
  { event := event78716
    frameStart := 78645 },
  { event := event78717
    frameStart := 78645 },
  { event := event78718
    frameStart := 78645 },
  { event := event78719
    frameStart := 78645 }
]

def eventLeaf4920 : Array AnnotatedEvent := #[
  { event := event78720
    frameStart := 78645 },
  { event := event78721
    frameStart := 78645 },
  { event := event78722
    frameStart := 78645 },
  { event := event78723
    frameStart := 78645 },
  { event := event78724
    frameStart := 78645 },
  { event := event78725
    frameStart := 78645 },
  { event := event78726
    frameStart := 78645 },
  { event := event78727
    frameStart := 78645 },
  { event := event78728
    frameStart := 78645 },
  { event := event78729
    frameStart := 78645 },
  { event := event78730
    frameStart := 78645 },
  { event := event78731
    frameStart := 78645 },
  { event := event78732
    frameStart := 78645 },
  { event := event78733
    frameStart := 78645 },
  { event := event78734
    frameStart := 78645 },
  { event := event78735
    frameStart := 78645 }
]

def eventLeaf4921 : Array AnnotatedEvent := #[
  { event := event78736
    frameStart := 78645 },
  { event := event78737
    frameStart := 78645 },
  { event := event78738
    frameStart := 78645 },
  { event := event78739
    frameStart := 78645 },
  { event := event78740
    frameStart := 78645 },
  { event := event78741
    frameStart := 78645 },
  { event := event78742
    frameStart := 78645 },
  { event := event78743
    frameStart := 78645 },
  { event := event78744
    frameStart := 78645 },
  { event := event78745
    frameStart := 78645 },
  { event := event78746
    frameStart := 78645 },
  { event := event78747
    frameStart := 78645 },
  { event := event78748
    frameStart := 78645 },
  { event := event78749
    frameStart := 0 },
  { event := event78750
    frameStart := 0 },
  { event := event78751
    frameStart := 0 }
]

def eventLeaf4922 : Array AnnotatedEvent := #[
  { event := event78752
    frameStart := 0 },
  { event := event78753
    frameStart := 0 },
  { event := event78754
    frameStart := 0 },
  { event := event78755
    frameStart := 0 },
  { event := event78756
    frameStart := 0 },
  { event := event78757
    frameStart := 0 },
  { event := event78758
    frameStart := 0 },
  { event := event78759
    frameStart := 0 },
  { event := event78760
    frameStart := 0 },
  { event := event78761
    frameStart := 0 },
  { event := event78762
    frameStart := 0 },
  { event := event78763
    frameStart := 0 },
  { event := event78764
    frameStart := 0 },
  { event := event78765
    frameStart := 0 },
  { event := event78766
    frameStart := 0 },
  { event := event78767
    frameStart := 0 }
]

def eventLeaf4923 : Array AnnotatedEvent := #[
  { event := event78768
    frameStart := 0 },
  { event := event78769
    frameStart := 0 },
  { event := event78770
    frameStart := 0 },
  { event := event78771
    frameStart := 0 },
  { event := event78772
    frameStart := 0 },
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
    frameStart := 78803 },
  { event := event78804
    frameStart := 78803 },
  { event := event78805
    frameStart := 78803 },
  { event := event78806
    frameStart := 78803 },
  { event := event78807
    frameStart := 78803 },
  { event := event78808
    frameStart := 78803 },
  { event := event78809
    frameStart := 78803 },
  { event := event78810
    frameStart := 78803 },
  { event := event78811
    frameStart := 78803 },
  { event := event78812
    frameStart := 78803 },
  { event := event78813
    frameStart := 78803 },
  { event := event78814
    frameStart := 78803 },
  { event := event78815
    frameStart := 78803 }
]

def eventLeaf4926 : Array AnnotatedEvent := #[
  { event := event78816
    frameStart := 78803 },
  { event := event78817
    frameStart := 78803 },
  { event := event78818
    frameStart := 78803 },
  { event := event78819
    frameStart := 78803 },
  { event := event78820
    frameStart := 78803 },
  { event := event78821
    frameStart := 78803 },
  { event := event78822
    frameStart := 78803 },
  { event := event78823
    frameStart := 78803 },
  { event := event78824
    frameStart := 78803 },
  { event := event78825
    frameStart := 78803 },
  { event := event78826
    frameStart := 78803 },
  { event := event78827
    frameStart := 78803 },
  { event := event78828
    frameStart := 78803 },
  { event := event78829
    frameStart := 78803 },
  { event := event78830
    frameStart := 78803 },
  { event := event78831
    frameStart := 78803 }
]

def eventLeaf4927 : Array AnnotatedEvent := #[
  { event := event78832
    frameStart := 78803 },
  { event := event78833
    frameStart := 78803 },
  { event := event78834
    frameStart := 78803 },
  { event := event78835
    frameStart := 78803 },
  { event := event78836
    frameStart := 78803 },
  { event := event78837
    frameStart := 78803 },
  { event := event78838
    frameStart := 78803 },
  { event := event78839
    frameStart := 78803 },
  { event := event78840
    frameStart := 78803 },
  { event := event78841
    frameStart := 78803 },
  { event := event78842
    frameStart := 78803 },
  { event := event78843
    frameStart := 78803 },
  { event := event78844
    frameStart := 78803 },
  { event := event78845
    frameStart := 78803 },
  { event := event78846
    frameStart := 78803 },
  { event := event78847
    frameStart := 78803 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events307
