import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events104

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event26624 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14235⟩⟩) 1 ⟨11481⟩ 26619

def event26625 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14235⟩⟩) (.product (.predecessor 0 26623 .coefficient) (.predecessor 1 26624 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event26626 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14235⟩⟩, .operator (⟨26622, 0⟩, ⟨26619, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], []⟩, (1)⟩)

def exact26627RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], []⟩, (1)⟩]

theorem exact26627RawTermsValid :
    exact26627RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26627 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14235⟩⟩) exact26627RawTerms (.finite 324) 26625 .exactZero (none)

def event26628 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14236⟩⟩) 0 ⟨14235⟩ 26627

def event26629 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14236⟩⟩) (.identity (.predecessor 0 26628 .coefficient))

def event26630 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14236⟩⟩) (.finite 324)

def event26631 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15952⟩⟩) 0 ⟨14236⟩ 26630

def event26632 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15952⟩⟩) (.authority (.programFamilyFact))

def exact26633RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], []⟩, (1)⟩]

theorem exact26633RawTermsValid :
    exact26633RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26633 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15952⟩⟩) exact26633RawTerms (.finite 18) 26632 .exactZero (none)

def event26634 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15953⟩⟩) 0 ⟨15952⟩ 26633

def event26635 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15953⟩⟩) (.identity (.predecessor 0 26634 .coefficient))

def event26636 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15953⟩⟩) (.finite 18)

def event26637 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24169⟩⟩) 0 ⟨15953⟩ 26636

def event26638 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24169⟩⟩) (.authority (.programFamilyFact))

def event26639 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24169⟩⟩) (.finite 3720)

def event26640 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event26641 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24171⟩⟩) 0 ⟨6689⟩ 26640

def event26642 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24171⟩⟩) 1 ⟨24169⟩ 26639

def event26643 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24171⟩⟩) (.authority (.operator))

def exact26644RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24171⟩⟩]⟩, (1)⟩]

theorem exact26644RawTermsValid :
    exact26644RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26644 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24171⟩⟩) exact26644RawTerms .large 26643 .exactZero (none)

def event26645 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27905⟩⟩) 0 ⟨24171⟩ 26644

def event26646 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27905⟩⟩) (.authority (.operator))

def exact26647RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27905⟩⟩]⟩, (1)⟩]

theorem exact26647RawTermsValid :
    exact26647RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26647 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27905⟩⟩) exact26647RawTerms (.finite 8192) 26646 .exactZero (none)

def event26648 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event26649 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event26650 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16027⟩⟩) 0 ⟨15953⟩ 26636

def event26651 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16027⟩⟩) 1 ⟨110⟩ 26649

def event26652 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16027⟩⟩) (.sum [.predecessor 0 26650 .coefficient, .predecessor 1 26651 .coefficient])

def event26653 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16027⟩⟩) (.finite 18)

def event26654 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16028⟩⟩) 0 ⟨16027⟩ 26653

def event26655 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16028⟩⟩) (.identity (.predecessor 0 26654 .coefficient))

def exact26656RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], []⟩, (1)⟩]

theorem exact26656RawTermsValid :
    exact26656RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26656 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16028⟩⟩) exact26656RawTerms (.finite 18) 26655 .exactZero (none)

def event26657 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact26658RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact26658RawTermsValid :
    exact26658RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26658 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact26658RawTerms .large 26657 .exactZero (none)

def event26659 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16029⟩⟩) 0 ⟨6544⟩ 26658

def event26660 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16029⟩⟩) 1 ⟨16028⟩ 26656

def event26661 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16029⟩⟩) (.product (.predecessor 0 26659 .coefficient) (.predecessor 1 26660 .coefficient) (⟨false, false, none, none, none⟩))

def event26662 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16029⟩⟩, .operator (⟨26658, 0⟩, ⟨26656, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact26663RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact26663RawTermsValid :
    exact26663RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26663 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16029⟩⟩) exact26663RawTerms .large 26661 .exactZero (none)

def event26664 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6697⟩⟩) 0 ⟨6689⟩ 26640

def event26665 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6697⟩⟩) (.authority (.operator))

def exact26666RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩]

theorem exact26666RawTermsValid :
    exact26666RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26666 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6697⟩⟩) exact26666RawTerms .large 26665 .exactZero (none)

def event26667 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16030⟩⟩) 0 ⟨6697⟩ 26666

def event26668 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16030⟩⟩) 1 ⟨16029⟩ 26663

def event26669 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16030⟩⟩) (.sum [.predecessor 0 26667 .coefficient, .predecessor 1 26668 .coefficient])

def exact26670RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26670RawTermsValid :
    exact26670RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26670 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16030⟩⟩) exact26670RawTerms .large 26669 .exactZero (none)

def event26671 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27906⟩⟩) 0 ⟨16030⟩ 26670

def event26672 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27906⟩⟩) 1 ⟨27905⟩ 26647

def event26673 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27906⟩⟩) (.product (.predecessor 0 26671 .coefficient) (.predecessor 1 26672 .coefficient) (⟨false, false, none, none, none⟩))

def event26674 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27906⟩⟩, .operator (⟨26670, 0⟩, ⟨26647, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27905⟩⟩]⟩, (1)⟩)

def event26675 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27906⟩⟩, .operator (⟨26670, 1⟩, ⟨26647, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27905⟩⟩]⟩, (-1)⟩)

def event26676 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27906⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27905⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27905⟩⟩) ⟨24171⟩ 26644)

def event26677 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27906⟩⟩, .relation 26676 0, ⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨24171⟩⟩]⟩, (-1)⟩)

def exact26678RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27905⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨24171⟩⟩]⟩, (-1)⟩]

theorem exact26678RawTermsValid :
    exact26678RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26678 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27906⟩⟩) exact26678RawTerms .large 26673 .exactZero (none)

def event26679 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15995⟩⟩) 0 ⟨15953⟩ 26636

def event26680 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15995⟩⟩) (.authority (.programFamilyFact))

def exact26681RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15995⟩⟩], []⟩, (1)⟩]

theorem exact26681RawTermsValid :
    exact26681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26681 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15995⟩⟩) exact26681RawTerms (.finite 61) 26680 .exactZero (none)

def event26682 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15996⟩⟩) 0 ⟨6544⟩ 26658

def event26683 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15996⟩⟩) 1 ⟨15995⟩ 26681

def event26684 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15996⟩⟩) (.product (.predecessor 0 26682 .coefficient) (.predecessor 1 26683 .coefficient) (⟨false, true, none, none, some 1⟩))

def event26685 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15996⟩⟩, .operator (⟨26658, 0⟩, ⟨26681, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact26686RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact26686RawTermsValid :
    exact26686RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26686 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15996⟩⟩) exact26686RawTerms .large 26684 .exactZero (none)

def event26687 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6723⟩⟩) 0 ⟨6689⟩ 26640

def event26688 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6723⟩⟩) (.authority (.operator))

def exact26689RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩]

theorem exact26689RawTermsValid :
    exact26689RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26689 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6723⟩⟩) exact26689RawTerms .large 26688 .exactZero (none)

def event26690 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15997⟩⟩) 0 ⟨6723⟩ 26689

def event26691 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15997⟩⟩) 1 ⟨15996⟩ 26686

def event26692 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15997⟩⟩) (.sum [.predecessor 0 26690 .coefficient, .predecessor 1 26691 .coefficient])

def exact26693RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26693RawTermsValid :
    exact26693RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26693 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15997⟩⟩) exact26693RawTerms .large 26692 .exactZero (none)

def event26694 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27910⟩⟩) 0 ⟨15997⟩ 26693

def event26695 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27910⟩⟩) 1 ⟨27906⟩ 26678

def event26696 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27910⟩⟩) (.sum [.predecessor 0 26694 .coefficient, .predecessor 1 26695 .coefficient])

def exact26697RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27905⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨24171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26697RawTermsValid :
    exact26697RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26697 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27910⟩⟩) exact26697RawTerms .large 26696 .exactZero (none)

def event26698 : Event := .preFoldPolynomial 26697 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27905⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨24171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact26699RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27905⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨24171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event26699 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27910⟩⟩) 26698 exact26699RawTerms .large 26696 .exactZero (none)

def event26700 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15953⟩⟩) ⟨⟨136⟩, ⟨43⟩, ⟨109⟩⟩ ⟨26542, 26700⟩

def event26701 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21415⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21412⟩⟩]⟩) (1) 0 2 (.universal 26700 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21412⟩⟩]⟩) (none) 26699)

def event26702 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21415⟩⟩, .relation 26701 1, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩)

def event26703 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21415⟩⟩, .relation 26701 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27905⟩⟩]⟩, (-1)⟩)

def event26704 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21415⟩⟩, .relation 26701 2, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨24171⟩⟩]⟩, (1)⟩)

def event26705 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21415⟩⟩, .relation 26701 3, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact26706RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27905⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨24171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26706RawTermsValid :
    exact26706RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26706 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21415⟩⟩) exact26706RawTerms .large 26538 (.finite 1811303510016) (some (26540))

def event26707 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27908⟩⟩) 0 ⟨21415⟩ 26706

def event26708 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27908⟩⟩) 1 ⟨27907⟩ 26528

def event26709 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27908⟩⟩) (.sum [.predecessor 0 26707 .coefficient, .predecessor 1 26708 .coefficient])

def event26710 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27908⟩⟩, .operator (⟨26706, 0⟩, ⟨26528, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27905⟩⟩]⟩, (1)⟩)

def event26711 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27908⟩⟩, .operator (⟨26706, 2⟩, ⟨26528, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨24171⟩⟩]⟩, (-1)⟩)

def event26712 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27908⟩⟩) (.sum [.result 26706 .summary, .result 26528 .summary])

def exact26713RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26713RawTermsValid :
    exact26713RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26713 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27908⟩⟩) exact26713RawTerms .large 26709 (.finite 1292068473939586330624) (some (26712))

def event26714 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24106⟩⟩) 0 ⟨15834⟩ 1112

def event26715 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24106⟩⟩) (.authority (.programFamilyFact))

def event26716 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24106⟩⟩) (.finite 3720)

def event26717 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24108⟩⟩) 0 ⟨6689⟩ 5477

def event26718 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24108⟩⟩) 1 ⟨24106⟩ 26716

def event26719 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24108⟩⟩) (.authority (.operator))

def exact26720RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24108⟩⟩]⟩, (1)⟩]

theorem exact26720RawTermsValid :
    exact26720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26720 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24108⟩⟩) exact26720RawTerms .large 26719 .exactZero (none)

def event26721 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27688⟩⟩) 0 ⟨24108⟩ 26720

def event26722 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27688⟩⟩) (.authority (.operator))

def exact26723RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩, (1)⟩]

theorem exact26723RawTermsValid :
    exact26723RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26723 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27688⟩⟩) exact26723RawTerms (.finite 8192) 26722 .exactZero (none)

def event26724 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23547⟩⟩) 0 ⟨14019⟩ 1106

def event26725 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23547⟩⟩) (.authority (.programFamilyFact))

def event26726 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23547⟩⟩) (.finite 3720)

def event26727 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23548⟩⟩) 0 ⟨6689⟩ 5477

def event26728 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23548⟩⟩) 1 ⟨23547⟩ 26726

def event26729 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23548⟩⟩) (.authority (.operator))

def exact26730RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23548⟩⟩]⟩, (1)⟩]

theorem exact26730RawTermsValid :
    exact26730RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26730 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23548⟩⟩) exact26730RawTerms .large 26729 .exactZero (none)

def event26731 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26004⟩⟩) 0 ⟨23548⟩ 26730

def event26732 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26004⟩⟩) (.authority (.operator))

def exact26733RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26004⟩⟩]⟩, (1)⟩]

theorem exact26733RawTermsValid :
    exact26733RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26733 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26004⟩⟩) exact26733RawTerms (.finite 8192) 26732 .exactZero (none)

def event26734 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11398⟩⟩) 0 ⟨11397⟩ 1095

def event26735 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11398⟩⟩) 1 ⟨6570⟩ 21420

def event26736 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11398⟩⟩) (.tensor (.predecessor 0 26734 .coefficient) (.predecessor 1 26735 .coefficient) true false)

def event26737 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11398⟩⟩, .operator (⟨1095, 0⟩, ⟨21420, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11397⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact26738RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11397⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact26738RawTermsValid :
    exact26738RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26738 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11398⟩⟩) exact26738RawTerms .large 26736 .exactZero (none)

def event26739 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7348⟩⟩) 0 ⟨5557⟩ 21290

def event26740 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7348⟩⟩) 1 ⟨6778⟩ 11983

def event26741 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7348⟩⟩) (.product (.predecessor 0 26739 .coefficient) (.predecessor 1 26740 .coefficient) (⟨false, false, none, none, none⟩))

def event26742 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7348⟩⟩, .operator (⟨21290, 0⟩, ⟨11983, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩)

def exact26743RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩]

theorem exact26743RawTermsValid :
    exact26743RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26743 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7348⟩⟩) exact26743RawTerms .large 26741 .exactZero (none)

def event26744 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11399⟩⟩) 0 ⟨7348⟩ 26743

def event26745 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11399⟩⟩) 1 ⟨11398⟩ 26738

def event26746 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11399⟩⟩) (.sum [.predecessor 0 26744 .coefficient, .predecessor 1 26745 .coefficient])

def exact26747RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11397⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26747RawTermsValid :
    exact26747RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26747 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11399⟩⟩) exact26747RawTerms .large 26746 .exactZero (none)

def event26748 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11400⟩⟩) 0 ⟨11399⟩ 26747

def event26749 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11400⟩⟩) 1 ⟨92⟩ 11975

def event26750 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11400⟩⟩) (.sum [.predecessor 0 26748 .coefficient, .predecessor 1 26749 .coefficient])

def event26751 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11400⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨92⟩⟩]⟩) [⟨.result 11975 .coefficient, false, none⟩])

def event26752 : Event := .survivorFold (1) 26751

def exact26753RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11397⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26753RawTermsValid :
    exact26753RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26753 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11400⟩⟩) exact26753RawTerms .large 26750 (.finite 26) (some (26751))

def event26754 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14020⟩⟩) 0 ⟨11400⟩ 26753

def event26755 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14020⟩⟩) 1 ⟨14017⟩ 1098

def event26756 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14020⟩⟩) (.product (.predecessor 0 26754 .coefficient) (.predecessor 1 26755 .coefficient) (⟨false, true, none, none, some 1⟩))

def event26757 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14020⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨14017⟩⟩], []⟩) [⟨.result 1098 .coefficient, true, some 1⟩])

def event26758 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14020⟩⟩) (.product (.result 26753 .summary) (.transfer 26757) (⟨false, false, none, none, none⟩))

def event26759 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14020⟩⟩, .operator (⟨26753, 1⟩, ⟨1098, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event26760 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14020⟩⟩, .operator (⟨26753, 0⟩, ⟨1098, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩)

def exact26761RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩]

theorem exact26761RawTermsValid :
    exact26761RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26761 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14020⟩⟩) exact26761RawTerms .large 26756 (.finite 13312) (some (26758))

def event26762 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14021⟩⟩) 0 ⟨14017⟩ 1098

def event26763 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14021⟩⟩) 1 ⟨6570⟩ 21420

def event26764 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14021⟩⟩) (.tensor (.predecessor 0 26762 .coefficient) (.predecessor 1 26763 .coefficient) true false)

def event26765 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14021⟩⟩, .operator (⟨1098, 0⟩, ⟨21420, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact26766RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact26766RawTermsValid :
    exact26766RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26766 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14021⟩⟩) exact26766RawTerms .large 26764 .exactZero (none)

def event26767 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7328⟩⟩) 0 ⟨5557⟩ 21290

def event26768 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7328⟩⟩) 1 ⟨6758⟩ 12024

def event26769 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7328⟩⟩) (.product (.predecessor 0 26767 .coefficient) (.predecessor 1 26768 .coefficient) (⟨false, false, none, none, none⟩))

def event26770 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7328⟩⟩, .operator (⟨21290, 0⟩, ⟨12024, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩]⟩, (1)⟩)

def exact26771RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩]⟩, (1)⟩]

theorem exact26771RawTermsValid :
    exact26771RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26771 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7328⟩⟩) exact26771RawTerms .large 26769 .exactZero (none)

def event26772 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14022⟩⟩) 0 ⟨7328⟩ 26771

def event26773 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14022⟩⟩) 1 ⟨14021⟩ 26766

def event26774 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14022⟩⟩) (.sum [.predecessor 0 26772 .coefficient, .predecessor 1 26773 .coefficient])

def exact26775RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26775RawTermsValid :
    exact26775RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26775 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14022⟩⟩) exact26775RawTerms .large 26774 .exactZero (none)

def event26776 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14023⟩⟩) 0 ⟨14022⟩ 26775

def event26777 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14023⟩⟩) 1 ⟨72⟩ 12016

def event26778 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14023⟩⟩) (.sum [.predecessor 0 26776 .coefficient, .predecessor 1 26777 .coefficient])

def event26779 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14023⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨72⟩⟩]⟩) [⟨.result 12016 .coefficient, false, none⟩])

def event26780 : Event := .survivorFold (1) 26779

def exact26781RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26781RawTermsValid :
    exact26781RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26781 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14023⟩⟩) exact26781RawTerms .large 26778 (.finite 26) (some (26779))

def event26782 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14024⟩⟩) 0 ⟨14023⟩ 26781

def event26783 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14024⟩⟩) 1 ⟨7850⟩ 12013

def event26784 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14024⟩⟩) (.product (.predecessor 0 26782 .coefficient) (.predecessor 1 26783 .coefficient) (⟨false, false, none, none, none⟩))

def event26785 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14024⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩) [⟨.result 12009 .coefficient, false, none⟩])

def event26786 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14024⟩⟩) (.product (.result 26781 .summary) (.transfer 26785) (⟨false, false, none, none, none⟩))

def event26787 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14024⟩⟩, .operator (⟨26781, 1⟩, ⟨12013, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (-1)⟩)

def event26788 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨14024⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7849⟩⟩) ⟨6778⟩ 11983)

def event26789 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14024⟩⟩, .relation 26788 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (-1)⟩)

def event26790 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14024⟩⟩, .operator (⟨26781, 0⟩, ⟨12013, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩)

def exact26791RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (-1)⟩]

theorem exact26791RawTermsValid :
    exact26791RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26791 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14024⟩⟩) exact26791RawTerms .large 26784 (.finite 95420416) (some (26786))

def event26792 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14025⟩⟩) 0 ⟨14024⟩ 26791

def event26793 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14025⟩⟩) 1 ⟨14020⟩ 26761

def event26794 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14025⟩⟩) (.sum [.predecessor 0 26792 .coefficient, .predecessor 1 26793 .coefficient])

def event26795 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14025⟩⟩, .operator (⟨26791, 1⟩, ⟨26761, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩)

def event26796 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14025⟩⟩) (.sum [.result 26791 .summary, .result 26761 .summary])

def exact26797RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26797RawTermsValid :
    exact26797RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26797 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14025⟩⟩) exact26797RawTerms .large 26794 (.finite 95433728) (some (26796))

def event26798 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26005⟩⟩) 0 ⟨14025⟩ 26797

def event26799 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26005⟩⟩) 1 ⟨26004⟩ 26733

def event26800 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26005⟩⟩) (.product (.predecessor 0 26798 .coefficient) (.predecessor 1 26799 .coefficient) (⟨false, false, none, none, none⟩))

def event26801 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26005⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26004⟩⟩]⟩) [⟨.result 26733 .coefficient, false, none⟩])

def event26802 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26005⟩⟩) (.product (.result 26797 .summary) (.transfer 26801) (⟨false, false, none, none, none⟩))

def event26803 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26005⟩⟩, .operator (⟨26797, 1⟩, ⟨26733, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26004⟩⟩]⟩, (-1)⟩)

def event26804 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26005⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26004⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26004⟩⟩) ⟨23548⟩ 26730)

def event26805 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26005⟩⟩, .relation 26804 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨23548⟩⟩]⟩, (-1)⟩)

def event26806 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26005⟩⟩, .operator (⟨26797, 0⟩, ⟨26733, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨26004⟩⟩]⟩, (1)⟩)

def exact26807RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨26004⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨23548⟩⟩]⟩, (-1)⟩]

theorem exact26807RawTermsValid :
    exact26807RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26807 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26005⟩⟩) exact26807RawTerms .large 26800 (.finite 350243308699648) (some (26802))

def event26808 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19468⟩⟩) 0 ⟨14019⟩ 1106

def event26809 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19468⟩⟩) (.authority (.relationPreimageSource ⟨14⟩))

def exact26810RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19468⟩⟩]⟩, (1)⟩]

theorem exact26810RawTermsValid :
    exact26810RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26810 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19468⟩⟩) exact26810RawTerms (.finite 136065468) 26809 .exactZero (none)

def event26811 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19470⟩⟩) 0 ⟨19468⟩ 26810

def event26812 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19470⟩⟩) 1 ⟨2348⟩ 4

def event26813 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19470⟩⟩) (.scale (.predecessor 0 26811 .coefficient) (.value (.predecessor 1 26812 .coefficient)))

def exact26814RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19468⟩⟩]⟩, (1)⟩]

theorem exact26814RawTermsValid :
    exact26814RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26814 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19470⟩⟩) exact26814RawTerms (.finite 136065468) 26813 .exactZero (none)

def event26815 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19471⟩⟩) 0 ⟨5559⟩ 21512

def event26816 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19471⟩⟩) 1 ⟨19470⟩ 26814

def event26817 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19471⟩⟩) (.product (.predecessor 0 26815 .coefficient) (.predecessor 1 26816 .coefficient) (⟨false, false, none, none, none⟩))

def event26818 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19471⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19468⟩⟩]⟩) [⟨.result 26810 .coefficient, false, none⟩])

def event26819 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19471⟩⟩) (.product (.result 21512 .summary) (.transfer 26818) (⟨false, false, none, none, none⟩))

def event26820 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19471⟩⟩, .operator (⟨21512, 0⟩, ⟨26814, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19468⟩⟩]⟩, (1)⟩)

def event26821 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19469⟩⟩)

def event26822 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event26823 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event26824 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event26825 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event26826 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event26827 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event26828 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event26829 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event26830 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 26829

def event26831 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 26827

def event26832 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 26830 .coefficient) (.value (.predecessor 1 26831 .coefficient)))

def event26833 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event26834 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 26833

def event26835 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 26825

def event26836 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 26834 .coefficient, .predecessor 1 26835 .coefficient])

def event26837 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event26838 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 26837

def event26839 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 26823

def event26840 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 26839 .coefficient))

def event26841 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event26842 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11397⟩⟩) 0 ⟨5554⟩ 26841

def event26843 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11397⟩⟩) (.authority (.programFamilyFact))

def exact26844RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩], []⟩, (1)⟩]

theorem exact26844RawTermsValid :
    exact26844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26844 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11397⟩⟩) exact26844RawTerms (.finite 16) 26843 .exactZero (none)

def event26845 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14017⟩⟩) 0 ⟨5554⟩ 26841

def event26846 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14017⟩⟩) (.authority (.programFamilyFact))

def exact26847RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14017⟩⟩], []⟩, (1)⟩]

theorem exact26847RawTermsValid :
    exact26847RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26847 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14017⟩⟩) exact26847RawTerms (.finite 16) 26846 .exactZero (none)

def event26848 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14018⟩⟩) 0 ⟨14017⟩ 26847

def event26849 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14018⟩⟩) 1 ⟨11397⟩ 26844

def event26850 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14018⟩⟩) (.product (.predecessor 0 26848 .coefficient) (.predecessor 1 26849 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event26851 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14018⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], []⟩) [⟨.result 26847 .coefficient, true, some 1⟩, ⟨.result 26844 .coefficient, true, some 1⟩])

def event26852 : Event := .survivorFold (1) 26851

def exact26853RawTerms : List Term := []

theorem exact26853RawTermsValid :
    exact26853RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26853 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14018⟩⟩) exact26853RawTerms (.finite 256) 26850 (.finite 256) (some (26851))

def event26854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14019⟩⟩) 0 ⟨14018⟩ 26853

def event26855 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14019⟩⟩) (.identity (.predecessor 0 26854 .coefficient))

def event26856 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14019⟩⟩) (.finite 256)

def event26857 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19468⟩⟩) 0 ⟨14019⟩ 26856

def event26858 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19468⟩⟩) (.authority (.relationPreimageSource ⟨14⟩))

def exact26859RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19468⟩⟩]⟩, (1)⟩]

theorem exact26859RawTermsValid :
    exact26859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26859 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19468⟩⟩) exact26859RawTerms (.finite 136065468) 26858 .exactZero (none)

def event26860 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact26861RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact26861RawTermsValid :
    exact26861RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26861 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact26861RawTerms .large 26860 .exactZero (none)

def event26862 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19469⟩⟩) 0 ⟨6⟩ 26861

def event26863 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19469⟩⟩) 1 ⟨19468⟩ 26859

def event26864 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19469⟩⟩) (.product (.predecessor 0 26862 .coefficient) (.predecessor 1 26863 .coefficient) (⟨false, false, none, none, none⟩))

def event26865 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19469⟩⟩, .operator (⟨26861, 0⟩, ⟨26859, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19468⟩⟩]⟩, (1)⟩)

def exact26866RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19468⟩⟩]⟩, (1)⟩]

theorem exact26866RawTermsValid :
    exact26866RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26866 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19469⟩⟩) exact26866RawTerms .large 26864 .exactZero (none)

def event26867 : Event := .preFoldPolynomial 26866 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19468⟩⟩]⟩, (1)⟩] .exactZero none

def exact26868RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19468⟩⟩]⟩, (1)⟩]

def event26868 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19469⟩⟩) 26867 exact26868RawTerms .large 26864 .exactZero (none)

def event26869 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26008⟩⟩)

def event26870 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event26871 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event26872 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event26873 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event26874 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event26875 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event26876 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event26877 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event26878 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 26877

def event26879 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 26875

def eventLeaf1664 : Array AnnotatedEvent := #[
  { event := event26624
    frameStart := 26596 },
  { event := event26625
    frameStart := 26596 },
  { event := event26626
    frameStart := 26596 },
  { event := event26627
    frameStart := 26596 },
  { event := event26628
    frameStart := 26596 },
  { event := event26629
    frameStart := 26596 },
  { event := event26630
    frameStart := 26596 },
  { event := event26631
    frameStart := 26596 },
  { event := event26632
    frameStart := 26596 },
  { event := event26633
    frameStart := 26596 },
  { event := event26634
    frameStart := 26596 },
  { event := event26635
    frameStart := 26596 },
  { event := event26636
    frameStart := 26596 },
  { event := event26637
    frameStart := 26596 },
  { event := event26638
    frameStart := 26596 },
  { event := event26639
    frameStart := 26596 }
]

def eventLeaf1665 : Array AnnotatedEvent := #[
  { event := event26640
    frameStart := 26596 },
  { event := event26641
    frameStart := 26596 },
  { event := event26642
    frameStart := 26596 },
  { event := event26643
    frameStart := 26596 },
  { event := event26644
    frameStart := 26596 },
  { event := event26645
    frameStart := 26596 },
  { event := event26646
    frameStart := 26596 },
  { event := event26647
    frameStart := 26596 },
  { event := event26648
    frameStart := 26596 },
  { event := event26649
    frameStart := 26596 },
  { event := event26650
    frameStart := 26596 },
  { event := event26651
    frameStart := 26596 },
  { event := event26652
    frameStart := 26596 },
  { event := event26653
    frameStart := 26596 },
  { event := event26654
    frameStart := 26596 },
  { event := event26655
    frameStart := 26596 }
]

def eventLeaf1666 : Array AnnotatedEvent := #[
  { event := event26656
    frameStart := 26596 },
  { event := event26657
    frameStart := 26596 },
  { event := event26658
    frameStart := 26596 },
  { event := event26659
    frameStart := 26596 },
  { event := event26660
    frameStart := 26596 },
  { event := event26661
    frameStart := 26596 },
  { event := event26662
    frameStart := 26596 },
  { event := event26663
    frameStart := 26596 },
  { event := event26664
    frameStart := 26596 },
  { event := event26665
    frameStart := 26596 },
  { event := event26666
    frameStart := 26596 },
  { event := event26667
    frameStart := 26596 },
  { event := event26668
    frameStart := 26596 },
  { event := event26669
    frameStart := 26596 },
  { event := event26670
    frameStart := 26596 },
  { event := event26671
    frameStart := 26596 }
]

def eventLeaf1667 : Array AnnotatedEvent := #[
  { event := event26672
    frameStart := 26596 },
  { event := event26673
    frameStart := 26596 },
  { event := event26674
    frameStart := 26596 },
  { event := event26675
    frameStart := 26596 },
  { event := event26676
    frameStart := 26596 },
  { event := event26677
    frameStart := 26596 },
  { event := event26678
    frameStart := 26596 },
  { event := event26679
    frameStart := 26596 },
  { event := event26680
    frameStart := 26596 },
  { event := event26681
    frameStart := 26596 },
  { event := event26682
    frameStart := 26596 },
  { event := event26683
    frameStart := 26596 },
  { event := event26684
    frameStart := 26596 },
  { event := event26685
    frameStart := 26596 },
  { event := event26686
    frameStart := 26596 },
  { event := event26687
    frameStart := 26596 }
]

def eventLeaf1668 : Array AnnotatedEvent := #[
  { event := event26688
    frameStart := 26596 },
  { event := event26689
    frameStart := 26596 },
  { event := event26690
    frameStart := 26596 },
  { event := event26691
    frameStart := 26596 },
  { event := event26692
    frameStart := 26596 },
  { event := event26693
    frameStart := 26596 },
  { event := event26694
    frameStart := 26596 },
  { event := event26695
    frameStart := 26596 },
  { event := event26696
    frameStart := 26596 },
  { event := event26697
    frameStart := 26596 },
  { event := event26698
    frameStart := 26596 },
  { event := event26699
    frameStart := 26596 },
  { event := event26700
    frameStart := 0 },
  { event := event26701
    frameStart := 0 },
  { event := event26702
    frameStart := 0 },
  { event := event26703
    frameStart := 0 }
]

def eventLeaf1669 : Array AnnotatedEvent := #[
  { event := event26704
    frameStart := 0 },
  { event := event26705
    frameStart := 0 },
  { event := event26706
    frameStart := 0 },
  { event := event26707
    frameStart := 0 },
  { event := event26708
    frameStart := 0 },
  { event := event26709
    frameStart := 0 },
  { event := event26710
    frameStart := 0 },
  { event := event26711
    frameStart := 0 },
  { event := event26712
    frameStart := 0 },
  { event := event26713
    frameStart := 0 },
  { event := event26714
    frameStart := 0 },
  { event := event26715
    frameStart := 0 },
  { event := event26716
    frameStart := 0 },
  { event := event26717
    frameStart := 0 },
  { event := event26718
    frameStart := 0 },
  { event := event26719
    frameStart := 0 }
]

def eventLeaf1670 : Array AnnotatedEvent := #[
  { event := event26720
    frameStart := 0 },
  { event := event26721
    frameStart := 0 },
  { event := event26722
    frameStart := 0 },
  { event := event26723
    frameStart := 0 },
  { event := event26724
    frameStart := 0 },
  { event := event26725
    frameStart := 0 },
  { event := event26726
    frameStart := 0 },
  { event := event26727
    frameStart := 0 },
  { event := event26728
    frameStart := 0 },
  { event := event26729
    frameStart := 0 },
  { event := event26730
    frameStart := 0 },
  { event := event26731
    frameStart := 0 },
  { event := event26732
    frameStart := 0 },
  { event := event26733
    frameStart := 0 },
  { event := event26734
    frameStart := 0 },
  { event := event26735
    frameStart := 0 }
]

def eventLeaf1671 : Array AnnotatedEvent := #[
  { event := event26736
    frameStart := 0 },
  { event := event26737
    frameStart := 0 },
  { event := event26738
    frameStart := 0 },
  { event := event26739
    frameStart := 0 },
  { event := event26740
    frameStart := 0 },
  { event := event26741
    frameStart := 0 },
  { event := event26742
    frameStart := 0 },
  { event := event26743
    frameStart := 0 },
  { event := event26744
    frameStart := 0 },
  { event := event26745
    frameStart := 0 },
  { event := event26746
    frameStart := 0 },
  { event := event26747
    frameStart := 0 },
  { event := event26748
    frameStart := 0 },
  { event := event26749
    frameStart := 0 },
  { event := event26750
    frameStart := 0 },
  { event := event26751
    frameStart := 0 }
]

def eventLeaf1672 : Array AnnotatedEvent := #[
  { event := event26752
    frameStart := 0 },
  { event := event26753
    frameStart := 0 },
  { event := event26754
    frameStart := 0 },
  { event := event26755
    frameStart := 0 },
  { event := event26756
    frameStart := 0 },
  { event := event26757
    frameStart := 0 },
  { event := event26758
    frameStart := 0 },
  { event := event26759
    frameStart := 0 },
  { event := event26760
    frameStart := 0 },
  { event := event26761
    frameStart := 0 },
  { event := event26762
    frameStart := 0 },
  { event := event26763
    frameStart := 0 },
  { event := event26764
    frameStart := 0 },
  { event := event26765
    frameStart := 0 },
  { event := event26766
    frameStart := 0 },
  { event := event26767
    frameStart := 0 }
]

def eventLeaf1673 : Array AnnotatedEvent := #[
  { event := event26768
    frameStart := 0 },
  { event := event26769
    frameStart := 0 },
  { event := event26770
    frameStart := 0 },
  { event := event26771
    frameStart := 0 },
  { event := event26772
    frameStart := 0 },
  { event := event26773
    frameStart := 0 },
  { event := event26774
    frameStart := 0 },
  { event := event26775
    frameStart := 0 },
  { event := event26776
    frameStart := 0 },
  { event := event26777
    frameStart := 0 },
  { event := event26778
    frameStart := 0 },
  { event := event26779
    frameStart := 0 },
  { event := event26780
    frameStart := 0 },
  { event := event26781
    frameStart := 0 },
  { event := event26782
    frameStart := 0 },
  { event := event26783
    frameStart := 0 }
]

def eventLeaf1674 : Array AnnotatedEvent := #[
  { event := event26784
    frameStart := 0 },
  { event := event26785
    frameStart := 0 },
  { event := event26786
    frameStart := 0 },
  { event := event26787
    frameStart := 0 },
  { event := event26788
    frameStart := 0 },
  { event := event26789
    frameStart := 0 },
  { event := event26790
    frameStart := 0 },
  { event := event26791
    frameStart := 0 },
  { event := event26792
    frameStart := 0 },
  { event := event26793
    frameStart := 0 },
  { event := event26794
    frameStart := 0 },
  { event := event26795
    frameStart := 0 },
  { event := event26796
    frameStart := 0 },
  { event := event26797
    frameStart := 0 },
  { event := event26798
    frameStart := 0 },
  { event := event26799
    frameStart := 0 }
]

def eventLeaf1675 : Array AnnotatedEvent := #[
  { event := event26800
    frameStart := 0 },
  { event := event26801
    frameStart := 0 },
  { event := event26802
    frameStart := 0 },
  { event := event26803
    frameStart := 0 },
  { event := event26804
    frameStart := 0 },
  { event := event26805
    frameStart := 0 },
  { event := event26806
    frameStart := 0 },
  { event := event26807
    frameStart := 0 },
  { event := event26808
    frameStart := 0 },
  { event := event26809
    frameStart := 0 },
  { event := event26810
    frameStart := 0 },
  { event := event26811
    frameStart := 0 },
  { event := event26812
    frameStart := 0 },
  { event := event26813
    frameStart := 0 },
  { event := event26814
    frameStart := 0 },
  { event := event26815
    frameStart := 0 }
]

def eventLeaf1676 : Array AnnotatedEvent := #[
  { event := event26816
    frameStart := 0 },
  { event := event26817
    frameStart := 0 },
  { event := event26818
    frameStart := 0 },
  { event := event26819
    frameStart := 0 },
  { event := event26820
    frameStart := 0 },
  { event := event26821
    frameStart := 26821 },
  { event := event26822
    frameStart := 26821 },
  { event := event26823
    frameStart := 26821 },
  { event := event26824
    frameStart := 26821 },
  { event := event26825
    frameStart := 26821 },
  { event := event26826
    frameStart := 26821 },
  { event := event26827
    frameStart := 26821 },
  { event := event26828
    frameStart := 26821 },
  { event := event26829
    frameStart := 26821 },
  { event := event26830
    frameStart := 26821 },
  { event := event26831
    frameStart := 26821 }
]

def eventLeaf1677 : Array AnnotatedEvent := #[
  { event := event26832
    frameStart := 26821 },
  { event := event26833
    frameStart := 26821 },
  { event := event26834
    frameStart := 26821 },
  { event := event26835
    frameStart := 26821 },
  { event := event26836
    frameStart := 26821 },
  { event := event26837
    frameStart := 26821 },
  { event := event26838
    frameStart := 26821 },
  { event := event26839
    frameStart := 26821 },
  { event := event26840
    frameStart := 26821 },
  { event := event26841
    frameStart := 26821 },
  { event := event26842
    frameStart := 26821 },
  { event := event26843
    frameStart := 26821 },
  { event := event26844
    frameStart := 26821 },
  { event := event26845
    frameStart := 26821 },
  { event := event26846
    frameStart := 26821 },
  { event := event26847
    frameStart := 26821 }
]

def eventLeaf1678 : Array AnnotatedEvent := #[
  { event := event26848
    frameStart := 26821 },
  { event := event26849
    frameStart := 26821 },
  { event := event26850
    frameStart := 26821 },
  { event := event26851
    frameStart := 26821 },
  { event := event26852
    frameStart := 26821 },
  { event := event26853
    frameStart := 26821 },
  { event := event26854
    frameStart := 26821 },
  { event := event26855
    frameStart := 26821 },
  { event := event26856
    frameStart := 26821 },
  { event := event26857
    frameStart := 26821 },
  { event := event26858
    frameStart := 26821 },
  { event := event26859
    frameStart := 26821 },
  { event := event26860
    frameStart := 26821 },
  { event := event26861
    frameStart := 26821 },
  { event := event26862
    frameStart := 26821 },
  { event := event26863
    frameStart := 26821 }
]

def eventLeaf1679 : Array AnnotatedEvent := #[
  { event := event26864
    frameStart := 26821 },
  { event := event26865
    frameStart := 26821 },
  { event := event26866
    frameStart := 26821 },
  { event := event26867
    frameStart := 26821 },
  { event := event26868
    frameStart := 26821 },
  { event := event26869
    frameStart := 26869 },
  { event := event26870
    frameStart := 26869 },
  { event := event26871
    frameStart := 26869 },
  { event := event26872
    frameStart := 26869 },
  { event := event26873
    frameStart := 26869 },
  { event := event26874
    frameStart := 26869 },
  { event := event26875
    frameStart := 26869 },
  { event := event26876
    frameStart := 26869 },
  { event := event26877
    frameStart := 26869 },
  { event := event26878
    frameStart := 26869 },
  { event := event26879
    frameStart := 26869 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events104
