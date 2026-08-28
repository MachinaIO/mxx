import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events315

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event80640 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25684⟩⟩, .relation 80639 0, ⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨23374⟩⟩]⟩, (-1)⟩)

def exact80641RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨23374⟩⟩]⟩, (-1)⟩]

theorem exact80641RawTermsValid :
    exact80641RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80641 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25684⟩⟩) exact80641RawTerms .large 80636 .exactZero (none)

def event80642 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16871⟩⟩) 0 ⟨13156⟩ 80581

def event80643 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16871⟩⟩) (.authority (.programFamilyFact))

def exact80644RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], []⟩, (1)⟩]

theorem exact80644RawTermsValid :
    exact80644RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80644 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16871⟩⟩) exact80644RawTerms (.finite 58) 80643 .exactZero (none)

def event80645 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16873⟩⟩) 0 ⟨6544⟩ 80603

def event80646 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16873⟩⟩) 1 ⟨16871⟩ 80644

def event80647 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16873⟩⟩) (.product (.predecessor 0 80645 .coefficient) (.predecessor 1 80646 .coefficient) (⟨false, true, none, none, some 1⟩))

def event80648 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16873⟩⟩, .operator (⟨80603, 0⟩, ⟨80644, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact80649RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact80649RawTermsValid :
    exact80649RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80649 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16873⟩⟩) exact80649RawTerms .large 80647 .exactZero (none)

def event80650 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6706⟩⟩) 0 ⟨6689⟩ 80585

def event80651 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6706⟩⟩) (.authority (.operator))

def exact80652RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩]⟩, (1)⟩]

theorem exact80652RawTermsValid :
    exact80652RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80652 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6706⟩⟩) exact80652RawTerms .large 80651 .exactZero (none)

def event80653 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16874⟩⟩) 0 ⟨6706⟩ 80652

def event80654 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16874⟩⟩) 1 ⟨16873⟩ 80649

def event80655 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16874⟩⟩) (.sum [.predecessor 0 80653 .coefficient, .predecessor 1 80654 .coefficient])

def exact80656RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80656RawTermsValid :
    exact80656RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80656 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16874⟩⟩) exact80656RawTerms .large 80655 .exactZero (none)

def event80657 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25685⟩⟩) 0 ⟨16874⟩ 80656

def event80658 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25685⟩⟩) 1 ⟨25684⟩ 80641

def event80659 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25685⟩⟩) (.sum [.predecessor 0 80657 .coefficient, .predecessor 1 80658 .coefficient])

def exact80660RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨23374⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80660RawTermsValid :
    exact80660RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80660 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25685⟩⟩) exact80660RawTerms .large 80659 .exactZero (none)

def event80661 : Event := .preFoldPolynomial 80660 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨23374⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact80662RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨23374⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event80662 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25685⟩⟩) 80661 exact80662RawTerms .large 80659 .exactZero (none)

def event80663 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨13156⟩⟩) ⟨⟨119⟩, ⟨25⟩, ⟨109⟩⟩ ⟨80499, 80663⟩

def event80664 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20179⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩) (1) 0 2 (.universal 80663 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20176⟩⟩]⟩) (none) 80662)

def event80665 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20179⟩⟩, .relation 80664 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩]⟩, (1)⟩)

def event80666 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20179⟩⟩, .relation 80664 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩, (-1)⟩)

def event80667 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20179⟩⟩, .relation 80664 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨23374⟩⟩]⟩, (1)⟩)

def event80668 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20179⟩⟩, .relation 80664 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact80669RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨23374⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80669RawTermsValid :
    exact80669RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80669 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20179⟩⟩) exact80669RawTerms .large 80495 (.finite 1811303510016) (some (80497))

def event80670 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25683⟩⟩) 0 ⟨20179⟩ 80669

def event80671 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25683⟩⟩) 1 ⟨25682⟩ 80485

def event80672 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25683⟩⟩) (.sum [.predecessor 0 80670 .coefficient, .predecessor 1 80671 .coefficient])

def event80673 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25683⟩⟩, .operator (⟨80669, 2⟩, ⟨80485, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨23374⟩⟩]⟩, (-1)⟩)

def event80674 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25683⟩⟩, .operator (⟨80669, 1⟩, ⟨80485, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩, (1)⟩)

def event80675 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25683⟩⟩) (.sum [.result 80669 .summary, .result 80485 .summary])

def exact80676RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80676RawTermsValid :
    exact80676RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80676 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25683⟩⟩) exact80676RawTerms .large 80672 (.finite 352182857248768) (some (80675))

def event80677 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29821⟩⟩) 0 ⟨25683⟩ 80676

def event80678 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29821⟩⟩) 1 ⟨29819⟩ 80401

def event80679 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29821⟩⟩) (.product (.predecessor 0 80677 .coefficient) (.predecessor 1 80678 .coefficient) (⟨false, false, none, none, none⟩))

def event80680 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29821⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩) [⟨.result 80401 .coefficient, false, none⟩])

def event80681 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29821⟩⟩) (.product (.result 80676 .summary) (.transfer 80680) (⟨false, false, none, none, none⟩))

def event80682 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29821⟩⟩, .operator (⟨80676, 0⟩, ⟨80401, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩, (1)⟩)

def event80683 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29821⟩⟩, .operator (⟨80676, 1⟩, ⟨80401, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩, (-1)⟩)

def event80684 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29821⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29819⟩⟩) ⟨24729⟩ 80398)

def event80685 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29821⟩⟩, .relation 80684 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨24729⟩⟩]⟩, (-1)⟩)

def exact80686RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨24729⟩⟩]⟩, (-1)⟩]

theorem exact80686RawTermsValid :
    exact80686RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80686 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29821⟩⟩) exact80686RawTerms .large 80679 (.finite 1292516721028694540288) (some (80681))

def event80687 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22696⟩⟩) 0 ⟨16872⟩ 3868

def event80688 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22696⟩⟩) (.authority (.relationPreimageSource ⟨63⟩))

def exact80689RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩, (1)⟩]

theorem exact80689RawTermsValid :
    exact80689RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80689 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22696⟩⟩) exact80689RawTerms (.finite 136065468) 80688 .exactZero (none)

def event80690 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22698⟩⟩) 0 ⟨22696⟩ 80689

def event80691 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22698⟩⟩) 1 ⟨2348⟩ 4

def event80692 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22698⟩⟩) (.scale (.predecessor 0 80690 .coefficient) (.value (.predecessor 1 80691 .coefficient)))

def exact80693RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩, (1)⟩]

theorem exact80693RawTermsValid :
    exact80693RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80693 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22698⟩⟩) exact80693RawTerms (.finite 136065468) 80692 .exactZero (none)

def event80694 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22699⟩⟩) 0 ⟨5541⟩ 80012

def event80695 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22699⟩⟩) 1 ⟨22698⟩ 80693

def event80696 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22699⟩⟩) (.product (.predecessor 0 80694 .coefficient) (.predecessor 1 80695 .coefficient) (⟨false, false, none, none, none⟩))

def event80697 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22699⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩) [⟨.result 80689 .coefficient, false, none⟩])

def event80698 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22699⟩⟩) (.product (.result 80012 .summary) (.transfer 80697) (⟨false, false, none, none, none⟩))

def event80699 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22699⟩⟩, .operator (⟨80012, 0⟩, ⟨80693, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩, (1)⟩)

def event80700 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22697⟩⟩)

def event80701 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event80702 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event80703 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event80704 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event80705 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event80706 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event80707 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event80708 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event80709 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 80708

def event80710 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 80706

def event80711 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 80709 .coefficient) (.value (.predecessor 1 80710 .coefficient)))

def event80712 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event80713 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 80712

def event80714 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 80704

def event80715 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 80713 .coefficient, .predecessor 1 80714 .coefficient])

def event80716 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event80717 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 80716

def event80718 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 80702

def event80719 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 80718 .coefficient))

def event80720 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event80721 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13154⟩⟩) 0 ⟨5536⟩ 80720

def event80722 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13154⟩⟩) (.authority (.programFamilyFact))

def exact80723RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13154⟩⟩], []⟩, (1)⟩]

theorem exact80723RawTermsValid :
    exact80723RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80723 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13154⟩⟩) exact80723RawTerms (.finite 58) 80722 .exactZero (none)

def event80724 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10240⟩⟩) 0 ⟨5536⟩ 80720

def event80725 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10240⟩⟩) (.authority (.programFamilyFact))

def exact80726RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩], []⟩, (1)⟩]

theorem exact80726RawTermsValid :
    exact80726RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80726 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10240⟩⟩) exact80726RawTerms (.finite 58) 80725 .exactZero (none)

def event80727 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13155⟩⟩) 0 ⟨10240⟩ 80726

def event80728 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13155⟩⟩) 1 ⟨13154⟩ 80723

def event80729 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13155⟩⟩) (.product (.predecessor 0 80727 .coefficient) (.predecessor 1 80728 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event80730 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13155⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], []⟩) [⟨.result 80726 .coefficient, true, some 1⟩, ⟨.result 80723 .coefficient, true, some 1⟩])

def event80731 : Event := .survivorFold (1) 80730

def exact80732RawTerms : List Term := []

theorem exact80732RawTermsValid :
    exact80732RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80732 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13155⟩⟩) exact80732RawTerms (.finite 3364) 80729 (.finite 3364) (some (80730))

def event80733 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13156⟩⟩) 0 ⟨13155⟩ 80732

def event80734 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13156⟩⟩) (.identity (.predecessor 0 80733 .coefficient))

def event80735 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13156⟩⟩) (.finite 3364)

def event80736 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16871⟩⟩) 0 ⟨13156⟩ 80735

def event80737 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16871⟩⟩) (.authority (.programFamilyFact))

def exact80738RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], []⟩, (1)⟩]

theorem exact80738RawTermsValid :
    exact80738RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80738 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16871⟩⟩) exact80738RawTerms (.finite 58) 80737 .exactZero (none)

def event80739 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16872⟩⟩) 0 ⟨16871⟩ 80738

def event80740 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16872⟩⟩) (.identity (.predecessor 0 80739 .coefficient))

def event80741 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16872⟩⟩) (.finite 58)

def event80742 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22696⟩⟩) 0 ⟨16872⟩ 80741

def event80743 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22696⟩⟩) (.authority (.relationPreimageSource ⟨63⟩))

def exact80744RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩, (1)⟩]

theorem exact80744RawTermsValid :
    exact80744RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80744 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22696⟩⟩) exact80744RawTerms (.finite 136065468) 80743 .exactZero (none)

def event80745 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact80746RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact80746RawTermsValid :
    exact80746RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80746 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact80746RawTerms .large 80745 .exactZero (none)

def event80747 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22697⟩⟩) 0 ⟨6⟩ 80746

def event80748 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22697⟩⟩) 1 ⟨22696⟩ 80744

def event80749 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22697⟩⟩) (.product (.predecessor 0 80747 .coefficient) (.predecessor 1 80748 .coefficient) (⟨false, false, none, none, none⟩))

def event80750 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22697⟩⟩, .operator (⟨80746, 0⟩, ⟨80744, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩, (1)⟩)

def exact80751RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩, (1)⟩]

theorem exact80751RawTermsValid :
    exact80751RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80751 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22697⟩⟩) exact80751RawTerms .large 80749 .exactZero (none)

def event80752 : Event := .preFoldPolynomial 80751 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩, (1)⟩] .exactZero none

def exact80753RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩, (1)⟩]

def event80753 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨22697⟩⟩) 80752 exact80753RawTerms .large 80749 .exactZero (none)

def event80754 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨29824⟩⟩)

def event80755 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event80756 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event80757 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event80758 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event80759 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event80760 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event80761 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event80762 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event80763 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 80762

def event80764 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 80760

def event80765 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 80763 .coefficient) (.value (.predecessor 1 80764 .coefficient)))

def event80766 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event80767 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 80766

def event80768 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 80758

def event80769 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 80767 .coefficient, .predecessor 1 80768 .coefficient])

def event80770 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event80771 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 80770

def event80772 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 80756

def event80773 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 80772 .coefficient))

def event80774 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event80775 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13154⟩⟩) 0 ⟨5536⟩ 80774

def event80776 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13154⟩⟩) (.authority (.programFamilyFact))

def exact80777RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13154⟩⟩], []⟩, (1)⟩]

theorem exact80777RawTermsValid :
    exact80777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80777 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13154⟩⟩) exact80777RawTerms (.finite 58) 80776 .exactZero (none)

def event80778 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10240⟩⟩) 0 ⟨5536⟩ 80774

def event80779 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10240⟩⟩) (.authority (.programFamilyFact))

def exact80780RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩], []⟩, (1)⟩]

theorem exact80780RawTermsValid :
    exact80780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80780 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10240⟩⟩) exact80780RawTerms (.finite 58) 80779 .exactZero (none)

def event80781 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13155⟩⟩) 0 ⟨10240⟩ 80780

def event80782 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13155⟩⟩) 1 ⟨13154⟩ 80777

def event80783 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13155⟩⟩) (.product (.predecessor 0 80781 .coefficient) (.predecessor 1 80782 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event80784 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13155⟩⟩, .operator (⟨80780, 0⟩, ⟨80777, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], []⟩, (1)⟩)

def exact80785RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], []⟩, (1)⟩]

theorem exact80785RawTermsValid :
    exact80785RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80785 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13155⟩⟩) exact80785RawTerms (.finite 3364) 80783 .exactZero (none)

def event80786 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13156⟩⟩) 0 ⟨13155⟩ 80785

def event80787 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13156⟩⟩) (.identity (.predecessor 0 80786 .coefficient))

def event80788 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13156⟩⟩) (.finite 3364)

def event80789 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16871⟩⟩) 0 ⟨13156⟩ 80788

def event80790 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16871⟩⟩) (.authority (.programFamilyFact))

def exact80791RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], []⟩, (1)⟩]

theorem exact80791RawTermsValid :
    exact80791RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80791 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16871⟩⟩) exact80791RawTerms (.finite 58) 80790 .exactZero (none)

def event80792 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16872⟩⟩) 0 ⟨16871⟩ 80791

def event80793 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16872⟩⟩) (.identity (.predecessor 0 80792 .coefficient))

def event80794 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16872⟩⟩) (.finite 58)

def event80795 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24727⟩⟩) 0 ⟨16872⟩ 80794

def event80796 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24727⟩⟩) (.authority (.programFamilyFact))

def event80797 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24727⟩⟩) (.finite 3720)

def event80798 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event80799 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24729⟩⟩) 0 ⟨6689⟩ 80798

def event80800 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24729⟩⟩) 1 ⟨24727⟩ 80797

def event80801 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24729⟩⟩) (.authority (.operator))

def exact80802RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24729⟩⟩]⟩, (1)⟩]

theorem exact80802RawTermsValid :
    exact80802RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80802 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24729⟩⟩) exact80802RawTerms .large 80801 .exactZero (none)

def event80803 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29819⟩⟩) 0 ⟨24729⟩ 80802

def event80804 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29819⟩⟩) (.authority (.operator))

def exact80805RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩, (1)⟩]

theorem exact80805RawTermsValid :
    exact80805RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80805 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29819⟩⟩) exact80805RawTerms (.finite 8192) 80804 .exactZero (none)

def event80806 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event80807 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event80808 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16967⟩⟩) 0 ⟨16872⟩ 80794

def event80809 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16967⟩⟩) 1 ⟨110⟩ 80807

def event80810 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16967⟩⟩) (.sum [.predecessor 0 80808 .coefficient, .predecessor 1 80809 .coefficient])

def event80811 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16967⟩⟩) (.finite 58)

def event80812 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16968⟩⟩) 0 ⟨16967⟩ 80811

def event80813 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16968⟩⟩) (.identity (.predecessor 0 80812 .coefficient))

def exact80814RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], []⟩, (1)⟩]

theorem exact80814RawTermsValid :
    exact80814RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80814 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16968⟩⟩) exact80814RawTerms (.finite 58) 80813 .exactZero (none)

def event80815 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact80816RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact80816RawTermsValid :
    exact80816RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80816 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact80816RawTerms .large 80815 .exactZero (none)

def event80817 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16969⟩⟩) 0 ⟨6544⟩ 80816

def event80818 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16969⟩⟩) 1 ⟨16968⟩ 80814

def event80819 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16969⟩⟩) (.product (.predecessor 0 80817 .coefficient) (.predecessor 1 80818 .coefficient) (⟨false, false, none, none, none⟩))

def event80820 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16969⟩⟩, .operator (⟨80816, 0⟩, ⟨80814, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact80821RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact80821RawTermsValid :
    exact80821RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80821 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16969⟩⟩) exact80821RawTerms .large 80819 .exactZero (none)

def event80822 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6706⟩⟩) 0 ⟨6689⟩ 80798

def event80823 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6706⟩⟩) (.authority (.operator))

def exact80824RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩]⟩, (1)⟩]

theorem exact80824RawTermsValid :
    exact80824RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80824 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6706⟩⟩) exact80824RawTerms .large 80823 .exactZero (none)

def event80825 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16970⟩⟩) 0 ⟨6706⟩ 80824

def event80826 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16970⟩⟩) 1 ⟨16969⟩ 80821

def event80827 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16970⟩⟩) (.sum [.predecessor 0 80825 .coefficient, .predecessor 1 80826 .coefficient])

def exact80828RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80828RawTermsValid :
    exact80828RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80828 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16970⟩⟩) exact80828RawTerms .large 80827 .exactZero (none)

def event80829 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29820⟩⟩) 0 ⟨16970⟩ 80828

def event80830 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29820⟩⟩) 1 ⟨29819⟩ 80805

def event80831 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29820⟩⟩) (.product (.predecessor 0 80829 .coefficient) (.predecessor 1 80830 .coefficient) (⟨false, false, none, none, none⟩))

def event80832 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29820⟩⟩, .operator (⟨80828, 0⟩, ⟨80805, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩, (1)⟩)

def event80833 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29820⟩⟩, .operator (⟨80828, 1⟩, ⟨80805, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩, (-1)⟩)

def event80834 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29820⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29819⟩⟩) ⟨24729⟩ 80802)

def event80835 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29820⟩⟩, .relation 80834 0, ⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨24729⟩⟩]⟩, (-1)⟩)

def exact80836RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨24729⟩⟩]⟩, (-1)⟩]

theorem exact80836RawTermsValid :
    exact80836RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80836 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29820⟩⟩) exact80836RawTerms .large 80831 .exactZero (none)

def event80837 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17085⟩⟩) 0 ⟨16872⟩ 80794

def event80838 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17085⟩⟩) (.authority (.programFamilyFact))

def exact80839RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17085⟩⟩], []⟩, (1)⟩]

theorem exact80839RawTermsValid :
    exact80839RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80839 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17085⟩⟩) exact80839RawTerms (.finite 63) 80838 .exactZero (none)

def event80840 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17086⟩⟩) 0 ⟨6544⟩ 80816

def event80841 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17086⟩⟩) 1 ⟨17085⟩ 80839

def event80842 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17086⟩⟩) (.product (.predecessor 0 80840 .coefficient) (.predecessor 1 80841 .coefficient) (⟨false, true, none, none, some 1⟩))

def event80843 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17086⟩⟩, .operator (⟨80816, 0⟩, ⟨80839, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact80844RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact80844RawTermsValid :
    exact80844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80844 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17086⟩⟩) exact80844RawTerms .large 80842 .exactZero (none)

def event80845 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6741⟩⟩) 0 ⟨6689⟩ 80798

def event80846 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6741⟩⟩) (.authority (.operator))

def exact80847RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩]

theorem exact80847RawTermsValid :
    exact80847RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80847 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6741⟩⟩) exact80847RawTerms .large 80846 .exactZero (none)

def event80848 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17087⟩⟩) 0 ⟨6741⟩ 80847

def event80849 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17087⟩⟩) 1 ⟨17086⟩ 80844

def event80850 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17087⟩⟩) (.sum [.predecessor 0 80848 .coefficient, .predecessor 1 80849 .coefficient])

def exact80851RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80851RawTermsValid :
    exact80851RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80851 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17087⟩⟩) exact80851RawTerms .large 80850 .exactZero (none)

def event80852 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29824⟩⟩) 0 ⟨17087⟩ 80851

def event80853 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29824⟩⟩) 1 ⟨29820⟩ 80836

def event80854 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29824⟩⟩) (.sum [.predecessor 0 80852 .coefficient, .predecessor 1 80853 .coefficient])

def exact80855RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨24729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80855RawTermsValid :
    exact80855RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80855 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29824⟩⟩) exact80855RawTerms .large 80854 .exactZero (none)

def event80856 : Event := .preFoldPolynomial 80855 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨24729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact80857RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨24729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event80857 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨29824⟩⟩) 80856 exact80857RawTerms .large 80854 .exactZero (none)

def event80858 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16872⟩⟩) ⟨⟨154⟩, ⟨63⟩, ⟨109⟩⟩ ⟨80700, 80858⟩

def event80859 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨22699⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩) (1) 0 2 (.universal 80858 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22696⟩⟩]⟩) (none) 80857)

def event80860 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22699⟩⟩, .relation 80859 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩)

def event80861 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22699⟩⟩, .relation 80859 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩, (-1)⟩)

def event80862 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22699⟩⟩, .relation 80859 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨24729⟩⟩]⟩, (1)⟩)

def event80863 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22699⟩⟩, .relation 80859 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact80864RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨24729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80864RawTermsValid :
    exact80864RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80864 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22699⟩⟩) exact80864RawTerms .large 80696 (.finite 1811303510016) (some (80698))

def event80865 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29822⟩⟩) 0 ⟨22699⟩ 80864

def event80866 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29822⟩⟩) 1 ⟨29821⟩ 80686

def event80867 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29822⟩⟩) (.sum [.predecessor 0 80865 .coefficient, .predecessor 1 80866 .coefficient])

def event80868 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29822⟩⟩, .operator (⟨80864, 0⟩, ⟨80686, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩, (1)⟩)

def event80869 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29822⟩⟩, .operator (⟨80864, 2⟩, ⟨80686, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16871⟩⟩], [⟨.program ⟨214⟩, ⟨24729⟩⟩]⟩, (-1)⟩)

def event80870 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29822⟩⟩) (.sum [.result 80864 .summary, .result 80686 .summary])

def exact80871RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact80871RawTermsValid :
    exact80871RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80871 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29822⟩⟩) exact80871RawTerms .large 80867 (.finite 1292516722839998050304) (some (80870))

def event80872 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24664⟩⟩) 0 ⟨16753⟩ 3891

def event80873 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24664⟩⟩) (.authority (.programFamilyFact))

def event80874 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24664⟩⟩) (.finite 3720)

def event80875 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24666⟩⟩) 0 ⟨6689⟩ 5477

def event80876 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24666⟩⟩) 1 ⟨24664⟩ 80874

def event80877 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24666⟩⟩) (.authority (.operator))

def exact80878RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24666⟩⟩]⟩, (1)⟩]

theorem exact80878RawTermsValid :
    exact80878RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80878 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24666⟩⟩) exact80878RawTerms .large 80877 .exactZero (none)

def event80879 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29602⟩⟩) 0 ⟨24666⟩ 80878

def event80880 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29602⟩⟩) (.authority (.operator))

def exact80881RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29602⟩⟩]⟩, (1)⟩]

theorem exact80881RawTermsValid :
    exact80881RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80881 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29602⟩⟩) exact80881RawTerms (.finite 8192) 80880 .exactZero (none)

def event80882 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23331⟩⟩) 0 ⟨12960⟩ 3885

def event80883 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23331⟩⟩) (.authority (.programFamilyFact))

def event80884 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23331⟩⟩) (.finite 3720)

def event80885 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23332⟩⟩) 0 ⟨6689⟩ 5477

def event80886 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23332⟩⟩) 1 ⟨23331⟩ 80884

def event80887 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23332⟩⟩) (.authority (.operator))

def exact80888RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23332⟩⟩]⟩, (1)⟩]

theorem exact80888RawTermsValid :
    exact80888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80888 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23332⟩⟩) exact80888RawTerms .large 80887 .exactZero (none)

def event80889 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25604⟩⟩) 0 ⟨23332⟩ 80888

def event80890 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25604⟩⟩) (.authority (.operator))

def exact80891RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩, (1)⟩]

theorem exact80891RawTermsValid :
    exact80891RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80891 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25604⟩⟩) exact80891RawTerms (.finite 8192) 80890 .exactZero (none)

def event80892 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12961⟩⟩) 0 ⟨12958⟩ 3874

def event80893 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12961⟩⟩) 1 ⟨6567⟩ 79920

def event80894 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12961⟩⟩) (.tensor (.predecessor 0 80892 .coefficient) (.predecessor 1 80893 .coefficient) true false)

def event80895 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12961⟩⟩, .operator (⟨3874, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def eventLeaf5040 : Array AnnotatedEvent := #[
  { event := event80640
    frameStart := 80547 },
  { event := event80641
    frameStart := 80547 },
  { event := event80642
    frameStart := 80547 },
  { event := event80643
    frameStart := 80547 },
  { event := event80644
    frameStart := 80547 },
  { event := event80645
    frameStart := 80547 },
  { event := event80646
    frameStart := 80547 },
  { event := event80647
    frameStart := 80547 },
  { event := event80648
    frameStart := 80547 },
  { event := event80649
    frameStart := 80547 },
  { event := event80650
    frameStart := 80547 },
  { event := event80651
    frameStart := 80547 },
  { event := event80652
    frameStart := 80547 },
  { event := event80653
    frameStart := 80547 },
  { event := event80654
    frameStart := 80547 },
  { event := event80655
    frameStart := 80547 }
]

def eventLeaf5041 : Array AnnotatedEvent := #[
  { event := event80656
    frameStart := 80547 },
  { event := event80657
    frameStart := 80547 },
  { event := event80658
    frameStart := 80547 },
  { event := event80659
    frameStart := 80547 },
  { event := event80660
    frameStart := 80547 },
  { event := event80661
    frameStart := 80547 },
  { event := event80662
    frameStart := 80547 },
  { event := event80663
    frameStart := 0 },
  { event := event80664
    frameStart := 0 },
  { event := event80665
    frameStart := 0 },
  { event := event80666
    frameStart := 0 },
  { event := event80667
    frameStart := 0 },
  { event := event80668
    frameStart := 0 },
  { event := event80669
    frameStart := 0 },
  { event := event80670
    frameStart := 0 },
  { event := event80671
    frameStart := 0 }
]

def eventLeaf5042 : Array AnnotatedEvent := #[
  { event := event80672
    frameStart := 0 },
  { event := event80673
    frameStart := 0 },
  { event := event80674
    frameStart := 0 },
  { event := event80675
    frameStart := 0 },
  { event := event80676
    frameStart := 0 },
  { event := event80677
    frameStart := 0 },
  { event := event80678
    frameStart := 0 },
  { event := event80679
    frameStart := 0 },
  { event := event80680
    frameStart := 0 },
  { event := event80681
    frameStart := 0 },
  { event := event80682
    frameStart := 0 },
  { event := event80683
    frameStart := 0 },
  { event := event80684
    frameStart := 0 },
  { event := event80685
    frameStart := 0 },
  { event := event80686
    frameStart := 0 },
  { event := event80687
    frameStart := 0 }
]

def eventLeaf5043 : Array AnnotatedEvent := #[
  { event := event80688
    frameStart := 0 },
  { event := event80689
    frameStart := 0 },
  { event := event80690
    frameStart := 0 },
  { event := event80691
    frameStart := 0 },
  { event := event80692
    frameStart := 0 },
  { event := event80693
    frameStart := 0 },
  { event := event80694
    frameStart := 0 },
  { event := event80695
    frameStart := 0 },
  { event := event80696
    frameStart := 0 },
  { event := event80697
    frameStart := 0 },
  { event := event80698
    frameStart := 0 },
  { event := event80699
    frameStart := 0 },
  { event := event80700
    frameStart := 80700 },
  { event := event80701
    frameStart := 80700 },
  { event := event80702
    frameStart := 80700 },
  { event := event80703
    frameStart := 80700 }
]

def eventLeaf5044 : Array AnnotatedEvent := #[
  { event := event80704
    frameStart := 80700 },
  { event := event80705
    frameStart := 80700 },
  { event := event80706
    frameStart := 80700 },
  { event := event80707
    frameStart := 80700 },
  { event := event80708
    frameStart := 80700 },
  { event := event80709
    frameStart := 80700 },
  { event := event80710
    frameStart := 80700 },
  { event := event80711
    frameStart := 80700 },
  { event := event80712
    frameStart := 80700 },
  { event := event80713
    frameStart := 80700 },
  { event := event80714
    frameStart := 80700 },
  { event := event80715
    frameStart := 80700 },
  { event := event80716
    frameStart := 80700 },
  { event := event80717
    frameStart := 80700 },
  { event := event80718
    frameStart := 80700 },
  { event := event80719
    frameStart := 80700 }
]

def eventLeaf5045 : Array AnnotatedEvent := #[
  { event := event80720
    frameStart := 80700 },
  { event := event80721
    frameStart := 80700 },
  { event := event80722
    frameStart := 80700 },
  { event := event80723
    frameStart := 80700 },
  { event := event80724
    frameStart := 80700 },
  { event := event80725
    frameStart := 80700 },
  { event := event80726
    frameStart := 80700 },
  { event := event80727
    frameStart := 80700 },
  { event := event80728
    frameStart := 80700 },
  { event := event80729
    frameStart := 80700 },
  { event := event80730
    frameStart := 80700 },
  { event := event80731
    frameStart := 80700 },
  { event := event80732
    frameStart := 80700 },
  { event := event80733
    frameStart := 80700 },
  { event := event80734
    frameStart := 80700 },
  { event := event80735
    frameStart := 80700 }
]

def eventLeaf5046 : Array AnnotatedEvent := #[
  { event := event80736
    frameStart := 80700 },
  { event := event80737
    frameStart := 80700 },
  { event := event80738
    frameStart := 80700 },
  { event := event80739
    frameStart := 80700 },
  { event := event80740
    frameStart := 80700 },
  { event := event80741
    frameStart := 80700 },
  { event := event80742
    frameStart := 80700 },
  { event := event80743
    frameStart := 80700 },
  { event := event80744
    frameStart := 80700 },
  { event := event80745
    frameStart := 80700 },
  { event := event80746
    frameStart := 80700 },
  { event := event80747
    frameStart := 80700 },
  { event := event80748
    frameStart := 80700 },
  { event := event80749
    frameStart := 80700 },
  { event := event80750
    frameStart := 80700 },
  { event := event80751
    frameStart := 80700 }
]

def eventLeaf5047 : Array AnnotatedEvent := #[
  { event := event80752
    frameStart := 80700 },
  { event := event80753
    frameStart := 80700 },
  { event := event80754
    frameStart := 80754 },
  { event := event80755
    frameStart := 80754 },
  { event := event80756
    frameStart := 80754 },
  { event := event80757
    frameStart := 80754 },
  { event := event80758
    frameStart := 80754 },
  { event := event80759
    frameStart := 80754 },
  { event := event80760
    frameStart := 80754 },
  { event := event80761
    frameStart := 80754 },
  { event := event80762
    frameStart := 80754 },
  { event := event80763
    frameStart := 80754 },
  { event := event80764
    frameStart := 80754 },
  { event := event80765
    frameStart := 80754 },
  { event := event80766
    frameStart := 80754 },
  { event := event80767
    frameStart := 80754 }
]

def eventLeaf5048 : Array AnnotatedEvent := #[
  { event := event80768
    frameStart := 80754 },
  { event := event80769
    frameStart := 80754 },
  { event := event80770
    frameStart := 80754 },
  { event := event80771
    frameStart := 80754 },
  { event := event80772
    frameStart := 80754 },
  { event := event80773
    frameStart := 80754 },
  { event := event80774
    frameStart := 80754 },
  { event := event80775
    frameStart := 80754 },
  { event := event80776
    frameStart := 80754 },
  { event := event80777
    frameStart := 80754 },
  { event := event80778
    frameStart := 80754 },
  { event := event80779
    frameStart := 80754 },
  { event := event80780
    frameStart := 80754 },
  { event := event80781
    frameStart := 80754 },
  { event := event80782
    frameStart := 80754 },
  { event := event80783
    frameStart := 80754 }
]

def eventLeaf5049 : Array AnnotatedEvent := #[
  { event := event80784
    frameStart := 80754 },
  { event := event80785
    frameStart := 80754 },
  { event := event80786
    frameStart := 80754 },
  { event := event80787
    frameStart := 80754 },
  { event := event80788
    frameStart := 80754 },
  { event := event80789
    frameStart := 80754 },
  { event := event80790
    frameStart := 80754 },
  { event := event80791
    frameStart := 80754 },
  { event := event80792
    frameStart := 80754 },
  { event := event80793
    frameStart := 80754 },
  { event := event80794
    frameStart := 80754 },
  { event := event80795
    frameStart := 80754 },
  { event := event80796
    frameStart := 80754 },
  { event := event80797
    frameStart := 80754 },
  { event := event80798
    frameStart := 80754 },
  { event := event80799
    frameStart := 80754 }
]

def eventLeaf5050 : Array AnnotatedEvent := #[
  { event := event80800
    frameStart := 80754 },
  { event := event80801
    frameStart := 80754 },
  { event := event80802
    frameStart := 80754 },
  { event := event80803
    frameStart := 80754 },
  { event := event80804
    frameStart := 80754 },
  { event := event80805
    frameStart := 80754 },
  { event := event80806
    frameStart := 80754 },
  { event := event80807
    frameStart := 80754 },
  { event := event80808
    frameStart := 80754 },
  { event := event80809
    frameStart := 80754 },
  { event := event80810
    frameStart := 80754 },
  { event := event80811
    frameStart := 80754 },
  { event := event80812
    frameStart := 80754 },
  { event := event80813
    frameStart := 80754 },
  { event := event80814
    frameStart := 80754 },
  { event := event80815
    frameStart := 80754 }
]

def eventLeaf5051 : Array AnnotatedEvent := #[
  { event := event80816
    frameStart := 80754 },
  { event := event80817
    frameStart := 80754 },
  { event := event80818
    frameStart := 80754 },
  { event := event80819
    frameStart := 80754 },
  { event := event80820
    frameStart := 80754 },
  { event := event80821
    frameStart := 80754 },
  { event := event80822
    frameStart := 80754 },
  { event := event80823
    frameStart := 80754 },
  { event := event80824
    frameStart := 80754 },
  { event := event80825
    frameStart := 80754 },
  { event := event80826
    frameStart := 80754 },
  { event := event80827
    frameStart := 80754 },
  { event := event80828
    frameStart := 80754 },
  { event := event80829
    frameStart := 80754 },
  { event := event80830
    frameStart := 80754 },
  { event := event80831
    frameStart := 80754 }
]

def eventLeaf5052 : Array AnnotatedEvent := #[
  { event := event80832
    frameStart := 80754 },
  { event := event80833
    frameStart := 80754 },
  { event := event80834
    frameStart := 80754 },
  { event := event80835
    frameStart := 80754 },
  { event := event80836
    frameStart := 80754 },
  { event := event80837
    frameStart := 80754 },
  { event := event80838
    frameStart := 80754 },
  { event := event80839
    frameStart := 80754 },
  { event := event80840
    frameStart := 80754 },
  { event := event80841
    frameStart := 80754 },
  { event := event80842
    frameStart := 80754 },
  { event := event80843
    frameStart := 80754 },
  { event := event80844
    frameStart := 80754 },
  { event := event80845
    frameStart := 80754 },
  { event := event80846
    frameStart := 80754 },
  { event := event80847
    frameStart := 80754 }
]

def eventLeaf5053 : Array AnnotatedEvent := #[
  { event := event80848
    frameStart := 80754 },
  { event := event80849
    frameStart := 80754 },
  { event := event80850
    frameStart := 80754 },
  { event := event80851
    frameStart := 80754 },
  { event := event80852
    frameStart := 80754 },
  { event := event80853
    frameStart := 80754 },
  { event := event80854
    frameStart := 80754 },
  { event := event80855
    frameStart := 80754 },
  { event := event80856
    frameStart := 80754 },
  { event := event80857
    frameStart := 80754 },
  { event := event80858
    frameStart := 0 },
  { event := event80859
    frameStart := 0 },
  { event := event80860
    frameStart := 0 },
  { event := event80861
    frameStart := 0 },
  { event := event80862
    frameStart := 0 },
  { event := event80863
    frameStart := 0 }
]

def eventLeaf5054 : Array AnnotatedEvent := #[
  { event := event80864
    frameStart := 0 },
  { event := event80865
    frameStart := 0 },
  { event := event80866
    frameStart := 0 },
  { event := event80867
    frameStart := 0 },
  { event := event80868
    frameStart := 0 },
  { event := event80869
    frameStart := 0 },
  { event := event80870
    frameStart := 0 },
  { event := event80871
    frameStart := 0 },
  { event := event80872
    frameStart := 0 },
  { event := event80873
    frameStart := 0 },
  { event := event80874
    frameStart := 0 },
  { event := event80875
    frameStart := 0 },
  { event := event80876
    frameStart := 0 },
  { event := event80877
    frameStart := 0 },
  { event := event80878
    frameStart := 0 },
  { event := event80879
    frameStart := 0 }
]

def eventLeaf5055 : Array AnnotatedEvent := #[
  { event := event80880
    frameStart := 0 },
  { event := event80881
    frameStart := 0 },
  { event := event80882
    frameStart := 0 },
  { event := event80883
    frameStart := 0 },
  { event := event80884
    frameStart := 0 },
  { event := event80885
    frameStart := 0 },
  { event := event80886
    frameStart := 0 },
  { event := event80887
    frameStart := 0 },
  { event := event80888
    frameStart := 0 },
  { event := event80889
    frameStart := 0 },
  { event := event80890
    frameStart := 0 },
  { event := event80891
    frameStart := 0 },
  { event := event80892
    frameStart := 0 },
  { event := event80893
    frameStart := 0 },
  { event := event80894
    frameStart := 0 },
  { event := event80895
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events315
