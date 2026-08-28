import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events057

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event14592 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event14593 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 14592

def event14594 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 14590

def event14595 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 14593 .coefficient) (.value (.predecessor 1 14594 .coefficient)))

def event14596 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event14597 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 0 ⟨5503⟩ 14596

def event14598 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 1 ⟨5245⟩ 14588

def event14599 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.sum [.predecessor 0 14597 .coefficient, .predecessor 1 14598 .coefficient])

def event14600 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.finite 223)

def event14601 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 0 ⟨5516⟩ 14600

def event14602 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 1 ⟨961⟩ 14586

def event14603 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.identity (.predecessor 1 14602 .coefficient))

def event14604 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.finite 224)

def event14605 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10708⟩⟩) 0 ⟨5560⟩ 14604

def event14606 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10708⟩⟩) (.authority (.programFamilyFact))

def exact14607RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10708⟩⟩], []⟩, (1)⟩]

theorem exact14607RawTermsValid :
    exact14607RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14607 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10708⟩⟩) exact14607RawTerms (.finite 3) 14606 .exactZero (none)

def event14608 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9525⟩⟩) 0 ⟨5560⟩ 14604

def event14609 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9525⟩⟩) (.authority (.programFamilyFact))

def exact14610RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩], []⟩, (1)⟩]

theorem exact14610RawTermsValid :
    exact14610RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14610 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9525⟩⟩) exact14610RawTerms (.finite 3) 14609 .exactZero (none)

def event14611 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10709⟩⟩) 0 ⟨9525⟩ 14610

def event14612 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10709⟩⟩) 1 ⟨10708⟩ 14607

def event14613 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10709⟩⟩) (.product (.predecessor 0 14611 .coefficient) (.predecessor 1 14612 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14614 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10709⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], []⟩) [⟨.result 14610 .coefficient, true, some 1⟩, ⟨.result 14607 .coefficient, true, some 1⟩])

def event14615 : Event := .survivorFold (1) 14614

def exact14616RawTerms : List Term := []

theorem exact14616RawTermsValid :
    exact14616RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14616 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10709⟩⟩) exact14616RawTerms (.finite 9) 14613 (.finite 9) (some (14614))

def event14617 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10710⟩⟩) 0 ⟨10709⟩ 14616

def event14618 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10710⟩⟩) (.identity (.predecessor 0 14617 .coefficient))

def event14619 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10710⟩⟩) (.finite 9)

def event14620 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19112⟩⟩) 0 ⟨10710⟩ 14619

def event14621 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19112⟩⟩) (.authority (.relationPreimageSource ⟨8⟩))

def exact14622RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19112⟩⟩]⟩, (1)⟩]

theorem exact14622RawTermsValid :
    exact14622RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14622 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19112⟩⟩) exact14622RawTerms (.finite 136065468) 14621 .exactZero (none)

def event14623 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact14624RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact14624RawTermsValid :
    exact14624RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14624 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact14624RawTerms .large 14623 .exactZero (none)

def event14625 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19113⟩⟩) 0 ⟨6⟩ 14624

def event14626 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19113⟩⟩) 1 ⟨19112⟩ 14622

def event14627 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19113⟩⟩) (.product (.predecessor 0 14625 .coefficient) (.predecessor 1 14626 .coefficient) (⟨false, false, none, none, none⟩))

def event14628 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19113⟩⟩, .operator (⟨14624, 0⟩, ⟨14622, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19112⟩⟩]⟩, (1)⟩)

def exact14629RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19112⟩⟩]⟩, (1)⟩]

theorem exact14629RawTermsValid :
    exact14629RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14629 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19113⟩⟩) exact14629RawTerms .large 14627 .exactZero (none)

def event14630 : Event := .preFoldPolynomial 14629 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19112⟩⟩]⟩, (1)⟩] .exactZero none

def exact14631RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19112⟩⟩]⟩, (1)⟩]

def event14631 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19113⟩⟩) 14630 exact14631RawTerms .large 14627 .exactZero (none)

def event14632 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25012⟩⟩)

def event14633 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event14634 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event14635 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.authority (.operator))

def event14636 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.finite 6)

def event14637 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event14638 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event14639 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event14640 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event14641 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 14640

def event14642 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 14638

def event14643 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 14641 .coefficient) (.value (.predecessor 1 14642 .coefficient)))

def event14644 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event14645 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 0 ⟨5503⟩ 14644

def event14646 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 1 ⟨5245⟩ 14636

def event14647 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.sum [.predecessor 0 14645 .coefficient, .predecessor 1 14646 .coefficient])

def event14648 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.finite 223)

def event14649 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 0 ⟨5516⟩ 14648

def event14650 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 1 ⟨961⟩ 14634

def event14651 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.identity (.predecessor 1 14650 .coefficient))

def event14652 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.finite 224)

def event14653 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10708⟩⟩) 0 ⟨5560⟩ 14652

def event14654 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10708⟩⟩) (.authority (.programFamilyFact))

def exact14655RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10708⟩⟩], []⟩, (1)⟩]

theorem exact14655RawTermsValid :
    exact14655RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14655 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10708⟩⟩) exact14655RawTerms (.finite 3) 14654 .exactZero (none)

def event14656 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9525⟩⟩) 0 ⟨5560⟩ 14652

def event14657 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9525⟩⟩) (.authority (.programFamilyFact))

def exact14658RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩], []⟩, (1)⟩]

theorem exact14658RawTermsValid :
    exact14658RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14658 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9525⟩⟩) exact14658RawTerms (.finite 3) 14657 .exactZero (none)

def event14659 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10709⟩⟩) 0 ⟨9525⟩ 14658

def event14660 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10709⟩⟩) 1 ⟨10708⟩ 14655

def event14661 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10709⟩⟩) (.product (.predecessor 0 14659 .coefficient) (.predecessor 1 14660 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14662 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10709⟩⟩, .operator (⟨14658, 0⟩, ⟨14655, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], []⟩, (1)⟩)

def exact14663RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], []⟩, (1)⟩]

theorem exact14663RawTermsValid :
    exact14663RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14663 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10709⟩⟩) exact14663RawTerms (.finite 9) 14661 .exactZero (none)

def event14664 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10710⟩⟩) 0 ⟨10709⟩ 14663

def event14665 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10710⟩⟩) (.identity (.predecessor 0 14664 .coefficient))

def event14666 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10710⟩⟩) (.finite 9)

def event14667 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23003⟩⟩) 0 ⟨10710⟩ 14666

def event14668 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23003⟩⟩) (.authority (.programFamilyFact))

def event14669 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23003⟩⟩) (.finite 3720)

def event14670 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event14671 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23004⟩⟩) 0 ⟨6689⟩ 14670

def event14672 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23004⟩⟩) 1 ⟨23003⟩ 14669

def event14673 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23004⟩⟩) (.authority (.operator))

def exact14674RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23004⟩⟩]⟩, (1)⟩]

theorem exact14674RawTermsValid :
    exact14674RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14674 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23004⟩⟩) exact14674RawTerms .large 14673 .exactZero (none)

def event14675 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25008⟩⟩) 0 ⟨23004⟩ 14674

def event14676 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25008⟩⟩) (.authority (.operator))

def exact14677RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25008⟩⟩]⟩, (1)⟩]

theorem exact14677RawTermsValid :
    exact14677RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14677 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25008⟩⟩) exact14677RawTerms (.finite 8192) 14676 .exactZero (none)

def event14678 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event14679 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event14680 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10788⟩⟩) 0 ⟨10710⟩ 14666

def event14681 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10788⟩⟩) 1 ⟨110⟩ 14679

def event14682 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10788⟩⟩) (.sum [.predecessor 0 14680 .coefficient, .predecessor 1 14681 .coefficient])

def event14683 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10788⟩⟩) (.finite 9)

def event14684 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10789⟩⟩) 0 ⟨10788⟩ 14683

def event14685 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10789⟩⟩) (.identity (.predecessor 0 14684 .coefficient))

def exact14686RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], []⟩, (1)⟩]

theorem exact14686RawTermsValid :
    exact14686RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14686 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10789⟩⟩) exact14686RawTerms (.finite 9) 14685 .exactZero (none)

def event14687 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact14688RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact14688RawTermsValid :
    exact14688RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14688 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact14688RawTerms .large 14687 .exactZero (none)

def event14689 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10790⟩⟩) 0 ⟨6544⟩ 14688

def event14690 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10790⟩⟩) 1 ⟨10789⟩ 14686

def event14691 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10790⟩⟩) (.product (.predecessor 0 14689 .coefficient) (.predecessor 1 14690 .coefficient) (⟨false, false, none, none, none⟩))

def event14692 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10790⟩⟩, .operator (⟨14688, 0⟩, ⟨14686, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact14693RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact14693RawTermsValid :
    exact14693RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14693 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10790⟩⟩) exact14693RawTerms .large 14691 .exactZero (none)

def event14694 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event14695 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event14696 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 14670

def event14697 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact14698RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact14698RawTermsValid :
    exact14698RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14698 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact14698RawTerms .large 14697 .exactZero (none)

def event14699 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6773⟩⟩) 0 ⟨6757⟩ 14698

def event14700 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6773⟩⟩) (.identity (.predecessor 0 14699 .coefficient))

def exact14701RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩, (1)⟩]

theorem exact14701RawTermsValid :
    exact14701RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14701 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6773⟩⟩) exact14701RawTerms .large 14700 .exactZero (none)

def event14702 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7834⟩⟩) 0 ⟨6773⟩ 14701

def event14703 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7834⟩⟩) (.authority (.operator))

def exact14704RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩, (1)⟩]

theorem exact14704RawTermsValid :
    exact14704RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14704 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7834⟩⟩) exact14704RawTerms (.finite 8192) 14703 .exactZero (none)

def event14705 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7835⟩⟩) 0 ⟨7834⟩ 14704

def event14706 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7835⟩⟩) 1 ⟨2348⟩ 14695

def event14707 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7835⟩⟩) (.scale (.predecessor 0 14705 .coefficient) (.value (.predecessor 1 14706 .coefficient)))

def exact14708RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩, (1)⟩]

theorem exact14708RawTermsValid :
    exact14708RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14708 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7835⟩⟩) exact14708RawTerms (.finite 8192) 14707 .exactZero (none)

def event14709 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6782⟩⟩) 0 ⟨6757⟩ 14698

def event14710 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6782⟩⟩) (.identity (.predecessor 0 14709 .coefficient))

def exact14711RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩]⟩, (1)⟩]

theorem exact14711RawTermsValid :
    exact14711RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14711 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6782⟩⟩) exact14711RawTerms .large 14710 .exactZero (none)

def event14712 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7836⟩⟩) 0 ⟨6782⟩ 14711

def event14713 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7836⟩⟩) 1 ⟨7835⟩ 14708

def event14714 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7836⟩⟩) (.product (.predecessor 0 14712 .coefficient) (.predecessor 1 14713 .coefficient) (⟨false, false, none, none, none⟩))

def event14715 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7836⟩⟩, .operator (⟨14711, 0⟩, ⟨14708, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩, (1)⟩)

def exact14716RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩, (1)⟩]

theorem exact14716RawTermsValid :
    exact14716RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14716 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7836⟩⟩) exact14716RawTerms .large 14714 .exactZero (none)

def event14717 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10791⟩⟩) 0 ⟨7836⟩ 14716

def event14718 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10791⟩⟩) 1 ⟨10790⟩ 14693

def event14719 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10791⟩⟩) (.sum [.predecessor 0 14717 .coefficient, .predecessor 1 14718 .coefficient])

def exact14720RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact14720RawTermsValid :
    exact14720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14720 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10791⟩⟩) exact14720RawTerms .large 14719 .exactZero (none)

def event14721 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25011⟩⟩) 0 ⟨10791⟩ 14720

def event14722 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25011⟩⟩) 1 ⟨25008⟩ 14677

def event14723 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25011⟩⟩) (.product (.predecessor 0 14721 .coefficient) (.predecessor 1 14722 .coefficient) (⟨false, false, none, none, none⟩))

def event14724 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25011⟩⟩, .operator (⟨14720, 1⟩, ⟨14677, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25008⟩⟩]⟩, (-1)⟩)

def event14725 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25011⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25008⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25008⟩⟩) ⟨23004⟩ 14674)

def event14726 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25011⟩⟩, .relation 14725 0, ⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], [⟨.program ⟨214⟩, ⟨23004⟩⟩]⟩, (-1)⟩)

def event14727 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25011⟩⟩, .operator (⟨14720, 0⟩, ⟨14677, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25008⟩⟩]⟩, (1)⟩)

def exact14728RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25008⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], [⟨.program ⟨214⟩, ⟨23004⟩⟩]⟩, (-1)⟩]

theorem exact14728RawTermsValid :
    exact14728RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14728 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25011⟩⟩) exact14728RawTerms .large 14723 .exactZero (none)

def event14729 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14969⟩⟩) 0 ⟨10710⟩ 14666

def event14730 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14969⟩⟩) (.authority (.programFamilyFact))

def exact14731RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], []⟩, (1)⟩]

theorem exact14731RawTermsValid :
    exact14731RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14731 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14969⟩⟩) exact14731RawTerms (.finite 3) 14730 .exactZero (none)

def event14732 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14971⟩⟩) 0 ⟨6544⟩ 14688

def event14733 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14971⟩⟩) 1 ⟨14969⟩ 14731

def event14734 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14971⟩⟩) (.product (.predecessor 0 14732 .coefficient) (.predecessor 1 14733 .coefficient) (⟨false, true, none, none, some 1⟩))

def event14735 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14971⟩⟩, .operator (⟨14688, 0⟩, ⟨14731, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact14736RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact14736RawTermsValid :
    exact14736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14736 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14971⟩⟩) exact14736RawTerms .large 14734 .exactZero (none)

def event14737 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6691⟩⟩) 0 ⟨6689⟩ 14670

def event14738 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6691⟩⟩) (.authority (.operator))

def exact14739RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩]

theorem exact14739RawTermsValid :
    exact14739RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14739 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6691⟩⟩) exact14739RawTerms .large 14738 .exactZero (none)

def event14740 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14972⟩⟩) 0 ⟨6691⟩ 14739

def event14741 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14972⟩⟩) 1 ⟨14971⟩ 14736

def event14742 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14972⟩⟩) (.sum [.predecessor 0 14740 .coefficient, .predecessor 1 14741 .coefficient])

def exact14743RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact14743RawTermsValid :
    exact14743RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14743 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14972⟩⟩) exact14743RawTerms .large 14742 .exactZero (none)

def event14744 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25012⟩⟩) 0 ⟨14972⟩ 14743

def event14745 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25012⟩⟩) 1 ⟨25011⟩ 14728

def event14746 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25012⟩⟩) (.sum [.predecessor 0 14744 .coefficient, .predecessor 1 14745 .coefficient])

def exact14747RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25008⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], [⟨.program ⟨214⟩, ⟨23004⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact14747RawTermsValid :
    exact14747RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14747 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25012⟩⟩) exact14747RawTerms .large 14746 .exactZero (none)

def event14748 : Event := .preFoldPolynomial 14747 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25008⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], [⟨.program ⟨214⟩, ⟨23004⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact14749RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25008⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], [⟨.program ⟨214⟩, ⟨23004⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event14749 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25012⟩⟩) 14748 exact14749RawTerms .large 14746 .exactZero (none)

def event14750 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨10710⟩⟩) ⟨⟨104⟩, ⟨8⟩, ⟨109⟩⟩ ⟨14584, 14750⟩

def event14751 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19115⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19112⟩⟩]⟩) (1) 0 2 (.universal 14750 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19112⟩⟩]⟩) (none) 14749)

def event14752 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19115⟩⟩, .relation 14751 2, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], [⟨.program ⟨214⟩, ⟨23004⟩⟩]⟩, (1)⟩)

def event14753 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19115⟩⟩, .relation 14751 1, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25008⟩⟩]⟩, (-1)⟩)

def event14754 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19115⟩⟩, .relation 14751 3, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event14755 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19115⟩⟩, .relation 14751 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩)

def exact14756RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25008⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], [⟨.program ⟨214⟩, ⟨23004⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact14756RawTermsValid :
    exact14756RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14756 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19115⟩⟩) exact14756RawTerms .large 14580 (.finite 1811303510016) (some (14582))

def event14757 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25010⟩⟩) 0 ⟨19115⟩ 14756

def event14758 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25010⟩⟩) 1 ⟨25009⟩ 14570

def event14759 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25010⟩⟩) (.sum [.predecessor 0 14757 .coefficient, .predecessor 1 14758 .coefficient])

def event14760 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25010⟩⟩, .operator (⟨14756, 2⟩, ⟨14570, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], [⟨.program ⟨214⟩, ⟨23004⟩⟩]⟩, (-1)⟩)

def event14761 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25010⟩⟩, .operator (⟨14756, 1⟩, ⟨14570, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25008⟩⟩]⟩, (1)⟩)

def event14762 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25010⟩⟩) (.sum [.result 14756 .summary, .result 14570 .summary])

def exact14763RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact14763RawTermsValid :
    exact14763RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14763 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25010⟩⟩) exact14763RawTerms .large 14759 (.finite 352014917316608) (some (14762))

def event14764 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26618⟩⟩) 0 ⟨25010⟩ 14763

def event14765 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26618⟩⟩) 1 ⟨26616⟩ 14467

def event14766 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26618⟩⟩) (.product (.predecessor 0 14764 .coefficient) (.predecessor 1 14765 .coefficient) (⟨false, false, none, none, none⟩))

def event14767 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26618⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26616⟩⟩]⟩) [⟨.result 14467 .coefficient, false, none⟩])

def event14768 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26618⟩⟩) (.product (.result 14763 .summary) (.transfer 14767) (⟨false, false, none, none, none⟩))

def event14769 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26618⟩⟩, .operator (⟨14763, 1⟩, ⟨14467, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26616⟩⟩]⟩, (-1)⟩)

def event14770 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26618⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26616⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26616⟩⟩) ⟨23796⟩ 14464)

def event14771 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26618⟩⟩, .relation 14770 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨23796⟩⟩]⟩, (-1)⟩)

def event14772 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26618⟩⟩, .operator (⟨14763, 0⟩, ⟨14467, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26616⟩⟩]⟩, (1)⟩)

def exact14773RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26616⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨23796⟩⟩]⟩, (-1)⟩]

theorem exact14773RawTermsValid :
    exact14773RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14773 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26618⟩⟩) exact14773RawTerms .large 14766 (.finite 1291900378790628425728) (some (14768))

def event14774 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20552⟩⟩) 0 ⟨14970⟩ 436

def event14775 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20552⟩⟩) (.authority (.relationPreimageSource ⟨30⟩))

def exact14776RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩, (1)⟩]

theorem exact14776RawTermsValid :
    exact14776RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14776 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20552⟩⟩) exact14776RawTerms (.finite 136065468) 14775 .exactZero (none)

def event14777 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20554⟩⟩) 0 ⟨20552⟩ 14776

def event14778 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20554⟩⟩) 1 ⟨2348⟩ 4

def event14779 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20554⟩⟩) (.scale (.predecessor 0 14777 .coefficient) (.value (.predecessor 1 14778 .coefficient)))

def exact14780RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩, (1)⟩]

theorem exact14780RawTermsValid :
    exact14780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14780 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20554⟩⟩) exact14780RawTerms (.finite 136065468) 14779 .exactZero (none)

def event14781 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20555⟩⟩) 0 ⟨5565⟩ 6561

def event14782 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20555⟩⟩) 1 ⟨20554⟩ 14780

def event14783 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20555⟩⟩) (.product (.predecessor 0 14781 .coefficient) (.predecessor 1 14782 .coefficient) (⟨false, false, none, none, none⟩))

def event14784 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20555⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩) [⟨.result 14776 .coefficient, false, none⟩])

def event14785 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20555⟩⟩) (.product (.result 6561 .summary) (.transfer 14784) (⟨false, false, none, none, none⟩))

def event14786 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20555⟩⟩, .operator (⟨6561, 0⟩, ⟨14780, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩, (1)⟩)

def event14787 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20553⟩⟩)

def event14788 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event14789 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event14790 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.authority (.operator))

def event14791 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.finite 6)

def event14792 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event14793 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event14794 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event14795 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event14796 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 14795

def event14797 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 14793

def event14798 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 14796 .coefficient) (.value (.predecessor 1 14797 .coefficient)))

def event14799 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event14800 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 0 ⟨5503⟩ 14799

def event14801 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 1 ⟨5245⟩ 14791

def event14802 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.sum [.predecessor 0 14800 .coefficient, .predecessor 1 14801 .coefficient])

def event14803 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.finite 223)

def event14804 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 0 ⟨5516⟩ 14803

def event14805 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 1 ⟨961⟩ 14789

def event14806 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.identity (.predecessor 1 14805 .coefficient))

def event14807 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.finite 224)

def event14808 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10708⟩⟩) 0 ⟨5560⟩ 14807

def event14809 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10708⟩⟩) (.authority (.programFamilyFact))

def exact14810RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10708⟩⟩], []⟩, (1)⟩]

theorem exact14810RawTermsValid :
    exact14810RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14810 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10708⟩⟩) exact14810RawTerms (.finite 3) 14809 .exactZero (none)

def event14811 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9525⟩⟩) 0 ⟨5560⟩ 14807

def event14812 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9525⟩⟩) (.authority (.programFamilyFact))

def exact14813RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩], []⟩, (1)⟩]

theorem exact14813RawTermsValid :
    exact14813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14813 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9525⟩⟩) exact14813RawTerms (.finite 3) 14812 .exactZero (none)

def event14814 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10709⟩⟩) 0 ⟨9525⟩ 14813

def event14815 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10709⟩⟩) 1 ⟨10708⟩ 14810

def event14816 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10709⟩⟩) (.product (.predecessor 0 14814 .coefficient) (.predecessor 1 14815 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event14817 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10709⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], []⟩) [⟨.result 14813 .coefficient, true, some 1⟩, ⟨.result 14810 .coefficient, true, some 1⟩])

def event14818 : Event := .survivorFold (1) 14817

def exact14819RawTerms : List Term := []

theorem exact14819RawTermsValid :
    exact14819RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14819 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10709⟩⟩) exact14819RawTerms (.finite 9) 14816 (.finite 9) (some (14817))

def event14820 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10710⟩⟩) 0 ⟨10709⟩ 14819

def event14821 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10710⟩⟩) (.identity (.predecessor 0 14820 .coefficient))

def event14822 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10710⟩⟩) (.finite 9)

def event14823 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14969⟩⟩) 0 ⟨10710⟩ 14822

def event14824 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14969⟩⟩) (.authority (.programFamilyFact))

def exact14825RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], []⟩, (1)⟩]

theorem exact14825RawTermsValid :
    exact14825RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14825 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14969⟩⟩) exact14825RawTerms (.finite 3) 14824 .exactZero (none)

def event14826 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14970⟩⟩) 0 ⟨14969⟩ 14825

def event14827 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14970⟩⟩) (.identity (.predecessor 0 14826 .coefficient))

def event14828 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14970⟩⟩) (.finite 3)

def event14829 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20552⟩⟩) 0 ⟨14970⟩ 14828

def event14830 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20552⟩⟩) (.authority (.relationPreimageSource ⟨30⟩))

def exact14831RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩, (1)⟩]

theorem exact14831RawTermsValid :
    exact14831RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14831 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20552⟩⟩) exact14831RawTerms (.finite 136065468) 14830 .exactZero (none)

def event14832 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact14833RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact14833RawTermsValid :
    exact14833RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14833 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact14833RawTerms .large 14832 .exactZero (none)

def event14834 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20553⟩⟩) 0 ⟨6⟩ 14833

def event14835 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20553⟩⟩) 1 ⟨20552⟩ 14831

def event14836 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20553⟩⟩) (.product (.predecessor 0 14834 .coefficient) (.predecessor 1 14835 .coefficient) (⟨false, false, none, none, none⟩))

def event14837 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20553⟩⟩, .operator (⟨14833, 0⟩, ⟨14831, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩, (1)⟩)

def exact14838RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩, (1)⟩]

theorem exact14838RawTermsValid :
    exact14838RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event14838 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20553⟩⟩) exact14838RawTerms .large 14836 .exactZero (none)

def event14839 : Event := .preFoldPolynomial 14838 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩, (1)⟩] .exactZero none

def exact14840RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩, (1)⟩]

def event14840 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20553⟩⟩) 14839 exact14840RawTerms .large 14836 .exactZero (none)

def event14841 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26621⟩⟩)

def event14842 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event14843 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event14844 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.authority (.operator))

def event14845 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.finite 6)

def event14846 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event14847 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def eventLeaf912 : Array AnnotatedEvent := #[
  { event := event14592
    frameStart := 14584 },
  { event := event14593
    frameStart := 14584 },
  { event := event14594
    frameStart := 14584 },
  { event := event14595
    frameStart := 14584 },
  { event := event14596
    frameStart := 14584 },
  { event := event14597
    frameStart := 14584 },
  { event := event14598
    frameStart := 14584 },
  { event := event14599
    frameStart := 14584 },
  { event := event14600
    frameStart := 14584 },
  { event := event14601
    frameStart := 14584 },
  { event := event14602
    frameStart := 14584 },
  { event := event14603
    frameStart := 14584 },
  { event := event14604
    frameStart := 14584 },
  { event := event14605
    frameStart := 14584 },
  { event := event14606
    frameStart := 14584 },
  { event := event14607
    frameStart := 14584 }
]

def eventLeaf913 : Array AnnotatedEvent := #[
  { event := event14608
    frameStart := 14584 },
  { event := event14609
    frameStart := 14584 },
  { event := event14610
    frameStart := 14584 },
  { event := event14611
    frameStart := 14584 },
  { event := event14612
    frameStart := 14584 },
  { event := event14613
    frameStart := 14584 },
  { event := event14614
    frameStart := 14584 },
  { event := event14615
    frameStart := 14584 },
  { event := event14616
    frameStart := 14584 },
  { event := event14617
    frameStart := 14584 },
  { event := event14618
    frameStart := 14584 },
  { event := event14619
    frameStart := 14584 },
  { event := event14620
    frameStart := 14584 },
  { event := event14621
    frameStart := 14584 },
  { event := event14622
    frameStart := 14584 },
  { event := event14623
    frameStart := 14584 }
]

def eventLeaf914 : Array AnnotatedEvent := #[
  { event := event14624
    frameStart := 14584 },
  { event := event14625
    frameStart := 14584 },
  { event := event14626
    frameStart := 14584 },
  { event := event14627
    frameStart := 14584 },
  { event := event14628
    frameStart := 14584 },
  { event := event14629
    frameStart := 14584 },
  { event := event14630
    frameStart := 14584 },
  { event := event14631
    frameStart := 14584 },
  { event := event14632
    frameStart := 14632 },
  { event := event14633
    frameStart := 14632 },
  { event := event14634
    frameStart := 14632 },
  { event := event14635
    frameStart := 14632 },
  { event := event14636
    frameStart := 14632 },
  { event := event14637
    frameStart := 14632 },
  { event := event14638
    frameStart := 14632 },
  { event := event14639
    frameStart := 14632 }
]

def eventLeaf915 : Array AnnotatedEvent := #[
  { event := event14640
    frameStart := 14632 },
  { event := event14641
    frameStart := 14632 },
  { event := event14642
    frameStart := 14632 },
  { event := event14643
    frameStart := 14632 },
  { event := event14644
    frameStart := 14632 },
  { event := event14645
    frameStart := 14632 },
  { event := event14646
    frameStart := 14632 },
  { event := event14647
    frameStart := 14632 },
  { event := event14648
    frameStart := 14632 },
  { event := event14649
    frameStart := 14632 },
  { event := event14650
    frameStart := 14632 },
  { event := event14651
    frameStart := 14632 },
  { event := event14652
    frameStart := 14632 },
  { event := event14653
    frameStart := 14632 },
  { event := event14654
    frameStart := 14632 },
  { event := event14655
    frameStart := 14632 }
]

def eventLeaf916 : Array AnnotatedEvent := #[
  { event := event14656
    frameStart := 14632 },
  { event := event14657
    frameStart := 14632 },
  { event := event14658
    frameStart := 14632 },
  { event := event14659
    frameStart := 14632 },
  { event := event14660
    frameStart := 14632 },
  { event := event14661
    frameStart := 14632 },
  { event := event14662
    frameStart := 14632 },
  { event := event14663
    frameStart := 14632 },
  { event := event14664
    frameStart := 14632 },
  { event := event14665
    frameStart := 14632 },
  { event := event14666
    frameStart := 14632 },
  { event := event14667
    frameStart := 14632 },
  { event := event14668
    frameStart := 14632 },
  { event := event14669
    frameStart := 14632 },
  { event := event14670
    frameStart := 14632 },
  { event := event14671
    frameStart := 14632 }
]

def eventLeaf917 : Array AnnotatedEvent := #[
  { event := event14672
    frameStart := 14632 },
  { event := event14673
    frameStart := 14632 },
  { event := event14674
    frameStart := 14632 },
  { event := event14675
    frameStart := 14632 },
  { event := event14676
    frameStart := 14632 },
  { event := event14677
    frameStart := 14632 },
  { event := event14678
    frameStart := 14632 },
  { event := event14679
    frameStart := 14632 },
  { event := event14680
    frameStart := 14632 },
  { event := event14681
    frameStart := 14632 },
  { event := event14682
    frameStart := 14632 },
  { event := event14683
    frameStart := 14632 },
  { event := event14684
    frameStart := 14632 },
  { event := event14685
    frameStart := 14632 },
  { event := event14686
    frameStart := 14632 },
  { event := event14687
    frameStart := 14632 }
]

def eventLeaf918 : Array AnnotatedEvent := #[
  { event := event14688
    frameStart := 14632 },
  { event := event14689
    frameStart := 14632 },
  { event := event14690
    frameStart := 14632 },
  { event := event14691
    frameStart := 14632 },
  { event := event14692
    frameStart := 14632 },
  { event := event14693
    frameStart := 14632 },
  { event := event14694
    frameStart := 14632 },
  { event := event14695
    frameStart := 14632 },
  { event := event14696
    frameStart := 14632 },
  { event := event14697
    frameStart := 14632 },
  { event := event14698
    frameStart := 14632 },
  { event := event14699
    frameStart := 14632 },
  { event := event14700
    frameStart := 14632 },
  { event := event14701
    frameStart := 14632 },
  { event := event14702
    frameStart := 14632 },
  { event := event14703
    frameStart := 14632 }
]

def eventLeaf919 : Array AnnotatedEvent := #[
  { event := event14704
    frameStart := 14632 },
  { event := event14705
    frameStart := 14632 },
  { event := event14706
    frameStart := 14632 },
  { event := event14707
    frameStart := 14632 },
  { event := event14708
    frameStart := 14632 },
  { event := event14709
    frameStart := 14632 },
  { event := event14710
    frameStart := 14632 },
  { event := event14711
    frameStart := 14632 },
  { event := event14712
    frameStart := 14632 },
  { event := event14713
    frameStart := 14632 },
  { event := event14714
    frameStart := 14632 },
  { event := event14715
    frameStart := 14632 },
  { event := event14716
    frameStart := 14632 },
  { event := event14717
    frameStart := 14632 },
  { event := event14718
    frameStart := 14632 },
  { event := event14719
    frameStart := 14632 }
]

def eventLeaf920 : Array AnnotatedEvent := #[
  { event := event14720
    frameStart := 14632 },
  { event := event14721
    frameStart := 14632 },
  { event := event14722
    frameStart := 14632 },
  { event := event14723
    frameStart := 14632 },
  { event := event14724
    frameStart := 14632 },
  { event := event14725
    frameStart := 14632 },
  { event := event14726
    frameStart := 14632 },
  { event := event14727
    frameStart := 14632 },
  { event := event14728
    frameStart := 14632 },
  { event := event14729
    frameStart := 14632 },
  { event := event14730
    frameStart := 14632 },
  { event := event14731
    frameStart := 14632 },
  { event := event14732
    frameStart := 14632 },
  { event := event14733
    frameStart := 14632 },
  { event := event14734
    frameStart := 14632 },
  { event := event14735
    frameStart := 14632 }
]

def eventLeaf921 : Array AnnotatedEvent := #[
  { event := event14736
    frameStart := 14632 },
  { event := event14737
    frameStart := 14632 },
  { event := event14738
    frameStart := 14632 },
  { event := event14739
    frameStart := 14632 },
  { event := event14740
    frameStart := 14632 },
  { event := event14741
    frameStart := 14632 },
  { event := event14742
    frameStart := 14632 },
  { event := event14743
    frameStart := 14632 },
  { event := event14744
    frameStart := 14632 },
  { event := event14745
    frameStart := 14632 },
  { event := event14746
    frameStart := 14632 },
  { event := event14747
    frameStart := 14632 },
  { event := event14748
    frameStart := 14632 },
  { event := event14749
    frameStart := 14632 },
  { event := event14750
    frameStart := 0 },
  { event := event14751
    frameStart := 0 }
]

def eventLeaf922 : Array AnnotatedEvent := #[
  { event := event14752
    frameStart := 0 },
  { event := event14753
    frameStart := 0 },
  { event := event14754
    frameStart := 0 },
  { event := event14755
    frameStart := 0 },
  { event := event14756
    frameStart := 0 },
  { event := event14757
    frameStart := 0 },
  { event := event14758
    frameStart := 0 },
  { event := event14759
    frameStart := 0 },
  { event := event14760
    frameStart := 0 },
  { event := event14761
    frameStart := 0 },
  { event := event14762
    frameStart := 0 },
  { event := event14763
    frameStart := 0 },
  { event := event14764
    frameStart := 0 },
  { event := event14765
    frameStart := 0 },
  { event := event14766
    frameStart := 0 },
  { event := event14767
    frameStart := 0 }
]

def eventLeaf923 : Array AnnotatedEvent := #[
  { event := event14768
    frameStart := 0 },
  { event := event14769
    frameStart := 0 },
  { event := event14770
    frameStart := 0 },
  { event := event14771
    frameStart := 0 },
  { event := event14772
    frameStart := 0 },
  { event := event14773
    frameStart := 0 },
  { event := event14774
    frameStart := 0 },
  { event := event14775
    frameStart := 0 },
  { event := event14776
    frameStart := 0 },
  { event := event14777
    frameStart := 0 },
  { event := event14778
    frameStart := 0 },
  { event := event14779
    frameStart := 0 },
  { event := event14780
    frameStart := 0 },
  { event := event14781
    frameStart := 0 },
  { event := event14782
    frameStart := 0 },
  { event := event14783
    frameStart := 0 }
]

def eventLeaf924 : Array AnnotatedEvent := #[
  { event := event14784
    frameStart := 0 },
  { event := event14785
    frameStart := 0 },
  { event := event14786
    frameStart := 0 },
  { event := event14787
    frameStart := 14787 },
  { event := event14788
    frameStart := 14787 },
  { event := event14789
    frameStart := 14787 },
  { event := event14790
    frameStart := 14787 },
  { event := event14791
    frameStart := 14787 },
  { event := event14792
    frameStart := 14787 },
  { event := event14793
    frameStart := 14787 },
  { event := event14794
    frameStart := 14787 },
  { event := event14795
    frameStart := 14787 },
  { event := event14796
    frameStart := 14787 },
  { event := event14797
    frameStart := 14787 },
  { event := event14798
    frameStart := 14787 },
  { event := event14799
    frameStart := 14787 }
]

def eventLeaf925 : Array AnnotatedEvent := #[
  { event := event14800
    frameStart := 14787 },
  { event := event14801
    frameStart := 14787 },
  { event := event14802
    frameStart := 14787 },
  { event := event14803
    frameStart := 14787 },
  { event := event14804
    frameStart := 14787 },
  { event := event14805
    frameStart := 14787 },
  { event := event14806
    frameStart := 14787 },
  { event := event14807
    frameStart := 14787 },
  { event := event14808
    frameStart := 14787 },
  { event := event14809
    frameStart := 14787 },
  { event := event14810
    frameStart := 14787 },
  { event := event14811
    frameStart := 14787 },
  { event := event14812
    frameStart := 14787 },
  { event := event14813
    frameStart := 14787 },
  { event := event14814
    frameStart := 14787 },
  { event := event14815
    frameStart := 14787 }
]

def eventLeaf926 : Array AnnotatedEvent := #[
  { event := event14816
    frameStart := 14787 },
  { event := event14817
    frameStart := 14787 },
  { event := event14818
    frameStart := 14787 },
  { event := event14819
    frameStart := 14787 },
  { event := event14820
    frameStart := 14787 },
  { event := event14821
    frameStart := 14787 },
  { event := event14822
    frameStart := 14787 },
  { event := event14823
    frameStart := 14787 },
  { event := event14824
    frameStart := 14787 },
  { event := event14825
    frameStart := 14787 },
  { event := event14826
    frameStart := 14787 },
  { event := event14827
    frameStart := 14787 },
  { event := event14828
    frameStart := 14787 },
  { event := event14829
    frameStart := 14787 },
  { event := event14830
    frameStart := 14787 },
  { event := event14831
    frameStart := 14787 }
]

def eventLeaf927 : Array AnnotatedEvent := #[
  { event := event14832
    frameStart := 14787 },
  { event := event14833
    frameStart := 14787 },
  { event := event14834
    frameStart := 14787 },
  { event := event14835
    frameStart := 14787 },
  { event := event14836
    frameStart := 14787 },
  { event := event14837
    frameStart := 14787 },
  { event := event14838
    frameStart := 14787 },
  { event := event14839
    frameStart := 14787 },
  { event := event14840
    frameStart := 14787 },
  { event := event14841
    frameStart := 14841 },
  { event := event14842
    frameStart := 14841 },
  { event := event14843
    frameStart := 14841 },
  { event := event14844
    frameStart := 14841 },
  { event := event14845
    frameStart := 14841 },
  { event := event14846
    frameStart := 14841 },
  { event := event14847
    frameStart := 14841 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events057
