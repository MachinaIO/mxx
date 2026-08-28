import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events315

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event80640 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨64133⟩⟩) (.finite 3720)

def event80641 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event80642 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64135⟩⟩) 0 ⟨7177⟩ 80641

def event80643 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64135⟩⟩) 1 ⟨64133⟩ 80640

def event80644 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64135⟩⟩) (.authority (.operator))

def exact80645RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨64135⟩⟩]⟩, (1)⟩]

theorem exact80645RawTermsValid :
    exact80645RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80645 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64135⟩⟩) exact80645RawTerms .large 80644 .exactZero (none)

def event80646 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65058⟩⟩) 0 ⟨64135⟩ 80645

def event80647 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65058⟩⟩) (.authority (.operator))

def exact80648RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨65058⟩⟩]⟩, (1)⟩]

theorem exact80648RawTermsValid :
    exact80648RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80648 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65058⟩⟩) exact80648RawTerms (.finite 8192) 80647 .exactZero (none)

def event80649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event80650 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event80651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64310⟩⟩) 0 ⟨62857⟩ 80637

def event80652 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64310⟩⟩) 1 ⟨136⟩ 80650

def event80653 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64310⟩⟩) (.sum [.predecessor 0 80651 .coefficient, .predecessor 1 80652 .coefficient])

def event80654 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨64310⟩⟩) (.finite 22)

def event80655 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64311⟩⟩) 0 ⟨64310⟩ 80654

def event80656 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64311⟩⟩) (.identity (.predecessor 0 80655 .coefficient))

def exact80657RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], []⟩, (1)⟩]

theorem exact80657RawTermsValid :
    exact80657RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80657 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64311⟩⟩) exact80657RawTerms (.finite 22) 80656 .exactZero (none)

def event80658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact80659RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact80659RawTermsValid :
    exact80659RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80659 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact80659RawTerms .large 80658 .exactZero (none)

def event80660 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64312⟩⟩) 0 ⟨6908⟩ 80659

def event80661 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64312⟩⟩) 1 ⟨64311⟩ 80657

def event80662 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64312⟩⟩) (.product (.predecessor 0 80660 .coefficient) (.predecessor 1 80661 .coefficient) (⟨false, false, none, none, none⟩))

def event80663 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64312⟩⟩, .operator (⟨80659, 0⟩, ⟨80657, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact80664RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact80664RawTermsValid :
    exact80664RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80664 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64312⟩⟩) exact80664RawTerms .large 80662 .exactZero (none)

def event80665 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7187⟩⟩) 0 ⟨7177⟩ 80641

def event80666 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7187⟩⟩) (.authority (.operator))

def exact80667RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩, (1)⟩]

theorem exact80667RawTermsValid :
    exact80667RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80667 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7187⟩⟩) exact80667RawTerms .large 80666 .exactZero (none)

def event80668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64313⟩⟩) 0 ⟨7187⟩ 80667

def event80669 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64313⟩⟩) 1 ⟨64312⟩ 80664

def event80670 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64313⟩⟩) (.sum [.predecessor 0 80668 .coefficient, .predecessor 1 80669 .coefficient])

def exact80671RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80671RawTermsValid :
    exact80671RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80671 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64313⟩⟩) exact80671RawTerms .large 80670 .exactZero (none)

def event80672 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65059⟩⟩) 0 ⟨64313⟩ 80671

def event80673 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65059⟩⟩) 1 ⟨65058⟩ 80648

def event80674 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65059⟩⟩) (.product (.predecessor 0 80672 .coefficient) (.predecessor 1 80673 .coefficient) (⟨false, false, none, none, none⟩))

def event80675 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65059⟩⟩, .operator (⟨80671, 0⟩, ⟨80648, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65058⟩⟩]⟩, (1)⟩)

def event80676 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65059⟩⟩, .operator (⟨80671, 1⟩, ⟨80648, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65058⟩⟩]⟩, (-1)⟩)

def event80677 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨65059⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65058⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨65058⟩⟩) ⟨64135⟩ 80645)

def event80678 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65059⟩⟩, .relation 80677 0, ⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨64135⟩⟩]⟩, (-1)⟩)

def exact80679RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65058⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨64135⟩⟩]⟩, (-1)⟩]

theorem exact80679RawTermsValid :
    exact80679RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80679 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65059⟩⟩) exact80679RawTerms .large 80674 .exactZero (none)

def event80680 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63195⟩⟩) 0 ⟨62857⟩ 80637

def event80681 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63195⟩⟩) (.authority (.programFamilyFact))

def exact80682RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], []⟩, (1)⟩]

theorem exact80682RawTermsValid :
    exact80682RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80682 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63195⟩⟩) exact80682RawTerms (.finite 61) 80681 .exactZero (none)

def event80683 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63197⟩⟩) 0 ⟨6908⟩ 80659

def event80684 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63197⟩⟩) 1 ⟨63195⟩ 80682

def event80685 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63197⟩⟩) (.product (.predecessor 0 80683 .coefficient) (.predecessor 1 80684 .coefficient) (⟨false, true, none, none, some 1⟩))

def event80686 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63197⟩⟩, .operator (⟨80659, 0⟩, ⟨80682, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact80687RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact80687RawTermsValid :
    exact80687RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80687 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63197⟩⟩) exact80687RawTerms .large 80685 .exactZero (none)

def event80688 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7214⟩⟩) 0 ⟨7177⟩ 80641

def event80689 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7214⟩⟩) (.authority (.operator))

def exact80690RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩]

theorem exact80690RawTermsValid :
    exact80690RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80690 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7214⟩⟩) exact80690RawTerms .large 80689 .exactZero (none)

def event80691 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63198⟩⟩) 0 ⟨7214⟩ 80690

def event80692 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63198⟩⟩) 1 ⟨63197⟩ 80687

def event80693 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63198⟩⟩) (.sum [.predecessor 0 80691 .coefficient, .predecessor 1 80692 .coefficient])

def exact80694RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80694RawTermsValid :
    exact80694RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80694 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63198⟩⟩) exact80694RawTerms .large 80693 .exactZero (none)

def event80695 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65063⟩⟩) 0 ⟨63198⟩ 80694

def event80696 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65063⟩⟩) 1 ⟨65059⟩ 80679

def event80697 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65063⟩⟩) (.sum [.predecessor 0 80695 .coefficient, .predecessor 1 80696 .coefficient])

def exact80698RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65058⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨64135⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80698RawTermsValid :
    exact80698RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80698 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65063⟩⟩) exact80698RawTerms .large 80697 .exactZero (none)

def event80699 : Event := .preFoldPolynomial 80698 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65058⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨64135⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact80700RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65058⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨64135⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event80700 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨65063⟩⟩) 80699 exact80700RawTerms .large 80697 .exactZero (none)

def event80701 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨62857⟩⟩) ⟨⟨93⟩, ⟨74⟩, ⟨135⟩⟩ ⟨80543, 80701⟩

def event80702 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨63799⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63796⟩⟩]⟩) (1) 0 2 (.universal 80701 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63796⟩⟩]⟩) (none) 80700)

def event80703 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63799⟩⟩, .relation 80702 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩)

def event80704 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63799⟩⟩, .relation 80702 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65058⟩⟩]⟩, (-1)⟩)

def event80705 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63799⟩⟩, .relation 80702 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨64135⟩⟩]⟩, (1)⟩)

def event80706 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63799⟩⟩, .relation 80702 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact80707RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65058⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨64135⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80707RawTermsValid :
    exact80707RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80707 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63799⟩⟩) exact80707RawTerms .large 80539 (.finite 202072841853861888) (some (80541))

def event80708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65061⟩⟩) 0 ⟨63799⟩ 80707

def event80709 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65061⟩⟩) 1 ⟨65060⟩ 80529

def event80710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65061⟩⟩) (.sum [.predecessor 0 80708 .coefficient, .predecessor 1 80709 .coefficient])

def event80711 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65061⟩⟩, .operator (⟨80707, 0⟩, ⟨80529, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65058⟩⟩]⟩, (1)⟩)

def event80712 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65061⟩⟩, .operator (⟨80707, 2⟩, ⟨80529, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨62856⟩⟩], [⟨.program ⟨257⟩, ⟨64135⟩⟩]⟩, (-1)⟩)

def event80713 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65061⟩⟩) (.sum [.result 80707 .summary, .result 80529 .summary])

def exact80714RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80714RawTermsValid :
    exact80714RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80714 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65061⟩⟩) exact80714RawTerms .large 80710 (.finite 32190771716940580661919523012608) (some (80713))

def event80715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61153⟩⟩) 0 ⟨59877⟩ 3333

def event80716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61153⟩⟩) (.authority (.programFamilyFact))

def event80717 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨61153⟩⟩) (.finite 3720)

def event80718 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61155⟩⟩) 0 ⟨7177⟩ 15500

def event80719 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61155⟩⟩) 1 ⟨61153⟩ 80717

def event80720 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61155⟩⟩) (.authority (.operator))

def exact80721RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨61155⟩⟩]⟩, (1)⟩]

theorem exact80721RawTermsValid :
    exact80721RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80721 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61155⟩⟩) exact80721RawTerms .large 80720 .exactZero (none)

def event80722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62078⟩⟩) 0 ⟨61155⟩ 80721

def event80723 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62078⟩⟩) (.authority (.operator))

def exact80724RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨62078⟩⟩]⟩, (1)⟩]

theorem exact80724RawTermsValid :
    exact80724RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80724 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62078⟩⟩) exact80724RawTerms (.finite 8192) 80723 .exactZero (none)

def event80725 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60984⟩⟩) 0 ⟨59649⟩ 3327

def event80726 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60984⟩⟩) (.authority (.programFamilyFact))

def event80727 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨60984⟩⟩) (.finite 3720)

def event80728 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60985⟩⟩) 0 ⟨7177⟩ 15500

def event80729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60985⟩⟩) 1 ⟨60984⟩ 80727

def event80730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60985⟩⟩) (.authority (.operator))

def exact80731RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨60985⟩⟩]⟩, (1)⟩]

theorem exact80731RawTermsValid :
    exact80731RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80731 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60985⟩⟩) exact80731RawTerms .large 80730 .exactZero (none)

def event80732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61525⟩⟩) 0 ⟨60985⟩ 80731

def event80733 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61525⟩⟩) (.authority (.operator))

def exact80734RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩, (1)⟩]

theorem exact80734RawTermsValid :
    exact80734RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80734 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61525⟩⟩) exact80734RawTerms (.finite 8192) 80733 .exactZero (none)

def event80735 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25323⟩⟩) 0 ⟨25322⟩ 3316

def event80736 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25323⟩⟩) 1 ⟨10328⟩ 75903

def event80737 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25323⟩⟩) (.tensor (.predecessor 0 80735 .coefficient) (.predecessor 1 80736 .coefficient) true false)

def event80738 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨25323⟩⟩, .operator (⟨3316, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact80739RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact80739RawTermsValid :
    exact80739RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80739 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25323⟩⟩) exact80739RawTerms .large 80737 .exactZero (none)

def event80740 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10332⟩⟩) 0 ⟨10327⟩ 75773

def event80741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10332⟩⟩) 1 ⟨7274⟩ 22090

def event80742 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10332⟩⟩) (.product (.predecessor 0 80740 .coefficient) (.predecessor 1 80741 .coefficient) (⟨false, false, none, none, none⟩))

def event80743 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10332⟩⟩, .operator (⟨75773, 0⟩, ⟨22090, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩, (1)⟩)

def exact80744RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩, (1)⟩]

theorem exact80744RawTermsValid :
    exact80744RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80744 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10332⟩⟩) exact80744RawTerms .large 80742 .exactZero (none)

def event80745 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25324⟩⟩) 0 ⟨10332⟩ 80744

def event80746 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25324⟩⟩) 1 ⟨25323⟩ 80739

def event80747 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25324⟩⟩) (.sum [.predecessor 0 80745 .coefficient, .predecessor 1 80746 .coefficient])

def exact80748RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80748RawTermsValid :
    exact80748RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80748 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25324⟩⟩) exact80748RawTerms .large 80747 .exactZero (none)

def event80749 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25325⟩⟩) 0 ⟨25324⟩ 80748

def event80750 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25325⟩⟩) 1 ⟨100⟩ 22082

def event80751 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25325⟩⟩) (.sum [.predecessor 0 80749 .coefficient, .predecessor 1 80750 .coefficient])

def event80752 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25325⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨100⟩⟩]⟩) [⟨.result 22082 .coefficient, false, none⟩])

def event80753 : Event := .survivorFold (1) 80752

def exact80754RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80754RawTermsValid :
    exact80754RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80754 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25325⟩⟩) exact80754RawTerms .large 80751 (.finite 26) (some (80752))

def event80755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59650⟩⟩) 0 ⟨25325⟩ 80754

def event80756 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59650⟩⟩) 1 ⟨59647⟩ 3319

def event80757 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59650⟩⟩) (.product (.predecessor 0 80755 .coefficient) (.predecessor 1 80756 .coefficient) (⟨false, true, none, none, some 1⟩))

def event80758 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59650⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨59647⟩⟩], []⟩) [⟨.result 3319 .coefficient, true, some 1⟩])

def event80759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59650⟩⟩) (.product (.result 80754 .summary) (.transfer 80758) (⟨false, false, none, none, none⟩))

def event80760 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59650⟩⟩, .operator (⟨80754, 1⟩, ⟨3319, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event80761 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59650⟩⟩, .operator (⟨80754, 0⟩, ⟨3319, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩, (1)⟩)

def exact80762RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩, (1)⟩]

theorem exact80762RawTermsValid :
    exact80762RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80762 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59650⟩⟩) exact80762RawTerms .large 80757 (.finite 15335424) (some (80759))

def event80763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59651⟩⟩) 0 ⟨59647⟩ 3319

def event80764 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59651⟩⟩) 1 ⟨10328⟩ 75903

def event80765 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59651⟩⟩) (.tensor (.predecessor 0 80763 .coefficient) (.predecessor 1 80764 .coefficient) true false)

def event80766 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59651⟩⟩, .operator (⟨3319, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact80767RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact80767RawTermsValid :
    exact80767RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80767 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59651⟩⟩) exact80767RawTerms .large 80765 .exactZero (none)

def event80768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10349⟩⟩) 0 ⟨10327⟩ 75773

def event80769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10349⟩⟩) 1 ⟨7291⟩ 22131

def event80770 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10349⟩⟩) (.product (.predecessor 0 80768 .coefficient) (.predecessor 1 80769 .coefficient) (⟨false, false, none, none, none⟩))

def event80771 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10349⟩⟩, .operator (⟨75773, 0⟩, ⟨22131, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩]⟩, (1)⟩)

def exact80772RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩]⟩, (1)⟩]

theorem exact80772RawTermsValid :
    exact80772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80772 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10349⟩⟩) exact80772RawTerms .large 80770 .exactZero (none)

def event80773 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59652⟩⟩) 0 ⟨10349⟩ 80772

def event80774 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59652⟩⟩) 1 ⟨59651⟩ 80767

def event80775 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59652⟩⟩) (.sum [.predecessor 0 80773 .coefficient, .predecessor 1 80774 .coefficient])

def exact80776RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80776RawTermsValid :
    exact80776RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80776 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59652⟩⟩) exact80776RawTerms .large 80775 .exactZero (none)

def event80777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59653⟩⟩) 0 ⟨59652⟩ 80776

def event80778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59653⟩⟩) 1 ⟨117⟩ 22123

def event80779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59653⟩⟩) (.sum [.predecessor 0 80777 .coefficient, .predecessor 1 80778 .coefficient])

def event80780 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59653⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨117⟩⟩]⟩) [⟨.result 22123 .coefficient, false, none⟩])

def event80781 : Event := .survivorFold (1) 80780

def exact80782RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80782RawTermsValid :
    exact80782RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80782 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59653⟩⟩) exact80782RawTerms .large 80779 (.finite 26) (some (80780))

def event80783 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59654⟩⟩) 0 ⟨59653⟩ 80782

def event80784 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59654⟩⟩) 1 ⟨9536⟩ 22120

def event80785 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59654⟩⟩) (.product (.predecessor 0 80783 .coefficient) (.predecessor 1 80784 .coefficient) (⟨false, false, none, none, none⟩))

def event80786 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59654⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) [⟨.result 22116 .coefficient, false, none⟩])

def event80787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59654⟩⟩) (.product (.result 80782 .summary) (.transfer 80786) (⟨false, false, none, none, none⟩))

def event80788 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59654⟩⟩, .operator (⟨80782, 1⟩, ⟨22120, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩, (-1)⟩)

def event80789 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨59654⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9535⟩⟩) ⟨7274⟩ 22090)

def event80790 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59654⟩⟩, .relation 80789 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩, (-1)⟩)

def event80791 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59654⟩⟩, .operator (⟨80782, 0⟩, ⟨22120, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩, (1)⟩)

def exact80792RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩, (-1)⟩]

theorem exact80792RawTermsValid :
    exact80792RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80792 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59654⟩⟩) exact80792RawTerms .large 80785 (.finite 279172874240) (some (80787))

def event80793 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59655⟩⟩) 0 ⟨59654⟩ 80792

def event80794 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59655⟩⟩) 1 ⟨59650⟩ 80762

def event80795 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59655⟩⟩) (.sum [.predecessor 0 80793 .coefficient, .predecessor 1 80794 .coefficient])

def event80796 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59655⟩⟩, .operator (⟨80792, 1⟩, ⟨80762, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩, (1)⟩)

def event80797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59655⟩⟩) (.sum [.result 80792 .summary, .result 80762 .summary])

def exact80798RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80798RawTermsValid :
    exact80798RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80798 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59655⟩⟩) exact80798RawTerms .large 80795 (.finite 279188209664) (some (80797))

def event80799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61526⟩⟩) 0 ⟨59655⟩ 80798

def event80800 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61526⟩⟩) 1 ⟨61525⟩ 80734

def event80801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61526⟩⟩) (.product (.predecessor 0 80799 .coefficient) (.predecessor 1 80800 .coefficient) (⟨false, false, none, none, none⟩))

def event80802 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61526⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩) [⟨.result 80734 .coefficient, false, none⟩])

def event80803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61526⟩⟩) (.product (.result 80798 .summary) (.transfer 80802) (⟨false, false, none, none, none⟩))

def event80804 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61526⟩⟩, .operator (⟨80798, 1⟩, ⟨80734, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩, (-1)⟩)

def event80805 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨61526⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61525⟩⟩) ⟨60985⟩ 80731)

def event80806 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61526⟩⟩, .relation 80805 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨60985⟩⟩]⟩, (-1)⟩)

def event80807 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61526⟩⟩, .operator (⟨80798, 0⟩, ⟨80734, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩, (1)⟩)

def exact80808RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨60985⟩⟩]⟩, (-1)⟩]

theorem exact80808RawTermsValid :
    exact80808RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80808 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61526⟩⟩) exact80808RawTerms .large 80801 (.finite 2997760574839177871360) (some (80803))

def event80809 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60449⟩⟩) 0 ⟨59649⟩ 3327

def event80810 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60449⟩⟩) (.authority (.relationPreimageSource ⟨43⟩))

def exact80811RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩, (1)⟩]

theorem exact80811RawTermsValid :
    exact80811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80811 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60449⟩⟩) exact80811RawTerms (.finite 5647228698) 80810 .exactZero (none)

def event80812 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60451⟩⟩) 0 ⟨60449⟩ 80811

def event80813 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60451⟩⟩) 1 ⟨2370⟩ 4

def event80814 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60451⟩⟩) (.scale (.predecessor 0 80812 .coefficient) (.value (.predecessor 1 80813 .coefficient)))

def exact80815RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩, (1)⟩]

theorem exact80815RawTermsValid :
    exact80815RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80815 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60451⟩⟩) exact80815RawTerms (.finite 5647228698) 80814 .exactZero (none)

def event80816 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60452⟩⟩) 0 ⟨10368⟩ 75995

def event80817 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60452⟩⟩) 1 ⟨60451⟩ 80815

def event80818 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60452⟩⟩) (.product (.predecessor 0 80816 .coefficient) (.predecessor 1 80817 .coefficient) (⟨false, false, none, none, none⟩))

def event80819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60452⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩) [⟨.result 80811 .coefficient, false, none⟩])

def event80820 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60452⟩⟩) (.product (.result 75995 .summary) (.transfer 80819) (⟨false, false, none, none, none⟩))

def event80821 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60452⟩⟩, .operator (⟨75995, 0⟩, ⟨80815, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩, (1)⟩)

def event80822 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨60450⟩⟩)

def event80823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event80824 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event80825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event80826 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event80827 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event80828 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event80829 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event80830 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event80831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 80830

def event80832 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 80828

def event80833 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 80831 .coefficient) (.value (.predecessor 1 80832 .coefficient)))

def event80834 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event80835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 80834

def event80836 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 80826

def event80837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 80835 .coefficient, .predecessor 1 80836 .coefficient])

def event80838 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event80839 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 80838

def event80840 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 80824

def event80841 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 80840 .coefficient))

def event80842 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event80843 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25322⟩⟩) 0 ⟨10325⟩ 80842

def event80844 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25322⟩⟩) (.authority (.programFamilyFact))

def exact80845RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩], []⟩, (1)⟩]

theorem exact80845RawTermsValid :
    exact80845RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80845 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25322⟩⟩) exact80845RawTerms (.finite 18) 80844 .exactZero (none)

def event80846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59647⟩⟩) 0 ⟨10325⟩ 80842

def event80847 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59647⟩⟩) (.authority (.programFamilyFact))

def exact80848RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59647⟩⟩], []⟩, (1)⟩]

theorem exact80848RawTermsValid :
    exact80848RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80848 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59647⟩⟩) exact80848RawTerms (.finite 18) 80847 .exactZero (none)

def event80849 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59648⟩⟩) 0 ⟨59647⟩ 80848

def event80850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59648⟩⟩) 1 ⟨25322⟩ 80845

def event80851 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59648⟩⟩) (.product (.predecessor 0 80849 .coefficient) (.predecessor 1 80850 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event80852 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59648⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], []⟩) [⟨.result 80848 .coefficient, true, some 1⟩, ⟨.result 80845 .coefficient, true, some 1⟩])

def event80853 : Event := .survivorFold (1) 80852

def exact80854RawTerms : List Term := []

theorem exact80854RawTermsValid :
    exact80854RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80854 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59648⟩⟩) exact80854RawTerms (.finite 324) 80851 (.finite 324) (some (80852))

def event80855 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59649⟩⟩) 0 ⟨59648⟩ 80854

def event80856 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59649⟩⟩) (.identity (.predecessor 0 80855 .coefficient))

def event80857 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨59649⟩⟩) (.finite 324)

def event80858 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60449⟩⟩) 0 ⟨59649⟩ 80857

def event80859 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60449⟩⟩) (.authority (.relationPreimageSource ⟨43⟩))

def exact80860RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩, (1)⟩]

theorem exact80860RawTermsValid :
    exact80860RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80860 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60449⟩⟩) exact80860RawTerms (.finite 5647228698) 80859 .exactZero (none)

def event80861 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact80862RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact80862RawTermsValid :
    exact80862RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80862 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact80862RawTerms .large 80861 .exactZero (none)

def event80863 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60450⟩⟩) 0 ⟨35⟩ 80862

def event80864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60450⟩⟩) 1 ⟨60449⟩ 80860

def event80865 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60450⟩⟩) (.product (.predecessor 0 80863 .coefficient) (.predecessor 1 80864 .coefficient) (⟨false, false, none, none, none⟩))

def event80866 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60450⟩⟩, .operator (⟨80862, 0⟩, ⟨80860, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩, (1)⟩)

def exact80867RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩, (1)⟩]

theorem exact80867RawTermsValid :
    exact80867RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80867 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60450⟩⟩) exact80867RawTerms .large 80865 .exactZero (none)

def event80868 : Event := .preFoldPolynomial 80867 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩, (1)⟩] .exactZero none

def exact80869RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩, (1)⟩]

def event80869 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨60450⟩⟩) 80868 exact80869RawTerms .large 80865 .exactZero (none)

def event80870 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨61529⟩⟩)

def event80871 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event80872 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event80873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event80874 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event80875 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event80876 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event80877 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event80878 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event80879 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 80878

def event80880 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 80876

def event80881 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 80879 .coefficient) (.value (.predecessor 1 80880 .coefficient)))

def event80882 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event80883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 80882

def event80884 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 80874

def event80885 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 80883 .coefficient, .predecessor 1 80884 .coefficient])

def event80886 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event80887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 80886

def event80888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 80872

def event80889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 80888 .coefficient))

def event80890 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event80891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25322⟩⟩) 0 ⟨10325⟩ 80890

def event80892 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25322⟩⟩) (.authority (.programFamilyFact))

def exact80893RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩], []⟩, (1)⟩]

theorem exact80893RawTermsValid :
    exact80893RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80893 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25322⟩⟩) exact80893RawTerms (.finite 18) 80892 .exactZero (none)

def event80894 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59647⟩⟩) 0 ⟨10325⟩ 80890

def event80895 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59647⟩⟩) (.authority (.programFamilyFact))

def eventLeaf5040 : Array AnnotatedEvent := #[
  { event := event80640
    frameStart := 80597 },
  { event := event80641
    frameStart := 80597 },
  { event := event80642
    frameStart := 80597 },
  { event := event80643
    frameStart := 80597 },
  { event := event80644
    frameStart := 80597 },
  { event := event80645
    frameStart := 80597 },
  { event := event80646
    frameStart := 80597 },
  { event := event80647
    frameStart := 80597 },
  { event := event80648
    frameStart := 80597 },
  { event := event80649
    frameStart := 80597 },
  { event := event80650
    frameStart := 80597 },
  { event := event80651
    frameStart := 80597 },
  { event := event80652
    frameStart := 80597 },
  { event := event80653
    frameStart := 80597 },
  { event := event80654
    frameStart := 80597 },
  { event := event80655
    frameStart := 80597 }
]

def eventLeaf5041 : Array AnnotatedEvent := #[
  { event := event80656
    frameStart := 80597 },
  { event := event80657
    frameStart := 80597 },
  { event := event80658
    frameStart := 80597 },
  { event := event80659
    frameStart := 80597 },
  { event := event80660
    frameStart := 80597 },
  { event := event80661
    frameStart := 80597 },
  { event := event80662
    frameStart := 80597 },
  { event := event80663
    frameStart := 80597 },
  { event := event80664
    frameStart := 80597 },
  { event := event80665
    frameStart := 80597 },
  { event := event80666
    frameStart := 80597 },
  { event := event80667
    frameStart := 80597 },
  { event := event80668
    frameStart := 80597 },
  { event := event80669
    frameStart := 80597 },
  { event := event80670
    frameStart := 80597 },
  { event := event80671
    frameStart := 80597 }
]

def eventLeaf5042 : Array AnnotatedEvent := #[
  { event := event80672
    frameStart := 80597 },
  { event := event80673
    frameStart := 80597 },
  { event := event80674
    frameStart := 80597 },
  { event := event80675
    frameStart := 80597 },
  { event := event80676
    frameStart := 80597 },
  { event := event80677
    frameStart := 80597 },
  { event := event80678
    frameStart := 80597 },
  { event := event80679
    frameStart := 80597 },
  { event := event80680
    frameStart := 80597 },
  { event := event80681
    frameStart := 80597 },
  { event := event80682
    frameStart := 80597 },
  { event := event80683
    frameStart := 80597 },
  { event := event80684
    frameStart := 80597 },
  { event := event80685
    frameStart := 80597 },
  { event := event80686
    frameStart := 80597 },
  { event := event80687
    frameStart := 80597 }
]

def eventLeaf5043 : Array AnnotatedEvent := #[
  { event := event80688
    frameStart := 80597 },
  { event := event80689
    frameStart := 80597 },
  { event := event80690
    frameStart := 80597 },
  { event := event80691
    frameStart := 80597 },
  { event := event80692
    frameStart := 80597 },
  { event := event80693
    frameStart := 80597 },
  { event := event80694
    frameStart := 80597 },
  { event := event80695
    frameStart := 80597 },
  { event := event80696
    frameStart := 80597 },
  { event := event80697
    frameStart := 80597 },
  { event := event80698
    frameStart := 80597 },
  { event := event80699
    frameStart := 80597 },
  { event := event80700
    frameStart := 80597 },
  { event := event80701
    frameStart := 0 },
  { event := event80702
    frameStart := 0 },
  { event := event80703
    frameStart := 0 }
]

def eventLeaf5044 : Array AnnotatedEvent := #[
  { event := event80704
    frameStart := 0 },
  { event := event80705
    frameStart := 0 },
  { event := event80706
    frameStart := 0 },
  { event := event80707
    frameStart := 0 },
  { event := event80708
    frameStart := 0 },
  { event := event80709
    frameStart := 0 },
  { event := event80710
    frameStart := 0 },
  { event := event80711
    frameStart := 0 },
  { event := event80712
    frameStart := 0 },
  { event := event80713
    frameStart := 0 },
  { event := event80714
    frameStart := 0 },
  { event := event80715
    frameStart := 0 },
  { event := event80716
    frameStart := 0 },
  { event := event80717
    frameStart := 0 },
  { event := event80718
    frameStart := 0 },
  { event := event80719
    frameStart := 0 }
]

def eventLeaf5045 : Array AnnotatedEvent := #[
  { event := event80720
    frameStart := 0 },
  { event := event80721
    frameStart := 0 },
  { event := event80722
    frameStart := 0 },
  { event := event80723
    frameStart := 0 },
  { event := event80724
    frameStart := 0 },
  { event := event80725
    frameStart := 0 },
  { event := event80726
    frameStart := 0 },
  { event := event80727
    frameStart := 0 },
  { event := event80728
    frameStart := 0 },
  { event := event80729
    frameStart := 0 },
  { event := event80730
    frameStart := 0 },
  { event := event80731
    frameStart := 0 },
  { event := event80732
    frameStart := 0 },
  { event := event80733
    frameStart := 0 },
  { event := event80734
    frameStart := 0 },
  { event := event80735
    frameStart := 0 }
]

def eventLeaf5046 : Array AnnotatedEvent := #[
  { event := event80736
    frameStart := 0 },
  { event := event80737
    frameStart := 0 },
  { event := event80738
    frameStart := 0 },
  { event := event80739
    frameStart := 0 },
  { event := event80740
    frameStart := 0 },
  { event := event80741
    frameStart := 0 },
  { event := event80742
    frameStart := 0 },
  { event := event80743
    frameStart := 0 },
  { event := event80744
    frameStart := 0 },
  { event := event80745
    frameStart := 0 },
  { event := event80746
    frameStart := 0 },
  { event := event80747
    frameStart := 0 },
  { event := event80748
    frameStart := 0 },
  { event := event80749
    frameStart := 0 },
  { event := event80750
    frameStart := 0 },
  { event := event80751
    frameStart := 0 }
]

def eventLeaf5047 : Array AnnotatedEvent := #[
  { event := event80752
    frameStart := 0 },
  { event := event80753
    frameStart := 0 },
  { event := event80754
    frameStart := 0 },
  { event := event80755
    frameStart := 0 },
  { event := event80756
    frameStart := 0 },
  { event := event80757
    frameStart := 0 },
  { event := event80758
    frameStart := 0 },
  { event := event80759
    frameStart := 0 },
  { event := event80760
    frameStart := 0 },
  { event := event80761
    frameStart := 0 },
  { event := event80762
    frameStart := 0 },
  { event := event80763
    frameStart := 0 },
  { event := event80764
    frameStart := 0 },
  { event := event80765
    frameStart := 0 },
  { event := event80766
    frameStart := 0 },
  { event := event80767
    frameStart := 0 }
]

def eventLeaf5048 : Array AnnotatedEvent := #[
  { event := event80768
    frameStart := 0 },
  { event := event80769
    frameStart := 0 },
  { event := event80770
    frameStart := 0 },
  { event := event80771
    frameStart := 0 },
  { event := event80772
    frameStart := 0 },
  { event := event80773
    frameStart := 0 },
  { event := event80774
    frameStart := 0 },
  { event := event80775
    frameStart := 0 },
  { event := event80776
    frameStart := 0 },
  { event := event80777
    frameStart := 0 },
  { event := event80778
    frameStart := 0 },
  { event := event80779
    frameStart := 0 },
  { event := event80780
    frameStart := 0 },
  { event := event80781
    frameStart := 0 },
  { event := event80782
    frameStart := 0 },
  { event := event80783
    frameStart := 0 }
]

def eventLeaf5049 : Array AnnotatedEvent := #[
  { event := event80784
    frameStart := 0 },
  { event := event80785
    frameStart := 0 },
  { event := event80786
    frameStart := 0 },
  { event := event80787
    frameStart := 0 },
  { event := event80788
    frameStart := 0 },
  { event := event80789
    frameStart := 0 },
  { event := event80790
    frameStart := 0 },
  { event := event80791
    frameStart := 0 },
  { event := event80792
    frameStart := 0 },
  { event := event80793
    frameStart := 0 },
  { event := event80794
    frameStart := 0 },
  { event := event80795
    frameStart := 0 },
  { event := event80796
    frameStart := 0 },
  { event := event80797
    frameStart := 0 },
  { event := event80798
    frameStart := 0 },
  { event := event80799
    frameStart := 0 }
]

def eventLeaf5050 : Array AnnotatedEvent := #[
  { event := event80800
    frameStart := 0 },
  { event := event80801
    frameStart := 0 },
  { event := event80802
    frameStart := 0 },
  { event := event80803
    frameStart := 0 },
  { event := event80804
    frameStart := 0 },
  { event := event80805
    frameStart := 0 },
  { event := event80806
    frameStart := 0 },
  { event := event80807
    frameStart := 0 },
  { event := event80808
    frameStart := 0 },
  { event := event80809
    frameStart := 0 },
  { event := event80810
    frameStart := 0 },
  { event := event80811
    frameStart := 0 },
  { event := event80812
    frameStart := 0 },
  { event := event80813
    frameStart := 0 },
  { event := event80814
    frameStart := 0 },
  { event := event80815
    frameStart := 0 }
]

def eventLeaf5051 : Array AnnotatedEvent := #[
  { event := event80816
    frameStart := 0 },
  { event := event80817
    frameStart := 0 },
  { event := event80818
    frameStart := 0 },
  { event := event80819
    frameStart := 0 },
  { event := event80820
    frameStart := 0 },
  { event := event80821
    frameStart := 0 },
  { event := event80822
    frameStart := 80822 },
  { event := event80823
    frameStart := 80822 },
  { event := event80824
    frameStart := 80822 },
  { event := event80825
    frameStart := 80822 },
  { event := event80826
    frameStart := 80822 },
  { event := event80827
    frameStart := 80822 },
  { event := event80828
    frameStart := 80822 },
  { event := event80829
    frameStart := 80822 },
  { event := event80830
    frameStart := 80822 },
  { event := event80831
    frameStart := 80822 }
]

def eventLeaf5052 : Array AnnotatedEvent := #[
  { event := event80832
    frameStart := 80822 },
  { event := event80833
    frameStart := 80822 },
  { event := event80834
    frameStart := 80822 },
  { event := event80835
    frameStart := 80822 },
  { event := event80836
    frameStart := 80822 },
  { event := event80837
    frameStart := 80822 },
  { event := event80838
    frameStart := 80822 },
  { event := event80839
    frameStart := 80822 },
  { event := event80840
    frameStart := 80822 },
  { event := event80841
    frameStart := 80822 },
  { event := event80842
    frameStart := 80822 },
  { event := event80843
    frameStart := 80822 },
  { event := event80844
    frameStart := 80822 },
  { event := event80845
    frameStart := 80822 },
  { event := event80846
    frameStart := 80822 },
  { event := event80847
    frameStart := 80822 }
]

def eventLeaf5053 : Array AnnotatedEvent := #[
  { event := event80848
    frameStart := 80822 },
  { event := event80849
    frameStart := 80822 },
  { event := event80850
    frameStart := 80822 },
  { event := event80851
    frameStart := 80822 },
  { event := event80852
    frameStart := 80822 },
  { event := event80853
    frameStart := 80822 },
  { event := event80854
    frameStart := 80822 },
  { event := event80855
    frameStart := 80822 },
  { event := event80856
    frameStart := 80822 },
  { event := event80857
    frameStart := 80822 },
  { event := event80858
    frameStart := 80822 },
  { event := event80859
    frameStart := 80822 },
  { event := event80860
    frameStart := 80822 },
  { event := event80861
    frameStart := 80822 },
  { event := event80862
    frameStart := 80822 },
  { event := event80863
    frameStart := 80822 }
]

def eventLeaf5054 : Array AnnotatedEvent := #[
  { event := event80864
    frameStart := 80822 },
  { event := event80865
    frameStart := 80822 },
  { event := event80866
    frameStart := 80822 },
  { event := event80867
    frameStart := 80822 },
  { event := event80868
    frameStart := 80822 },
  { event := event80869
    frameStart := 80822 },
  { event := event80870
    frameStart := 80870 },
  { event := event80871
    frameStart := 80870 },
  { event := event80872
    frameStart := 80870 },
  { event := event80873
    frameStart := 80870 },
  { event := event80874
    frameStart := 80870 },
  { event := event80875
    frameStart := 80870 },
  { event := event80876
    frameStart := 80870 },
  { event := event80877
    frameStart := 80870 },
  { event := event80878
    frameStart := 80870 },
  { event := event80879
    frameStart := 80870 }
]

def eventLeaf5055 : Array AnnotatedEvent := #[
  { event := event80880
    frameStart := 80870 },
  { event := event80881
    frameStart := 80870 },
  { event := event80882
    frameStart := 80870 },
  { event := event80883
    frameStart := 80870 },
  { event := event80884
    frameStart := 80870 },
  { event := event80885
    frameStart := 80870 },
  { event := event80886
    frameStart := 80870 },
  { event := event80887
    frameStart := 80870 },
  { event := event80888
    frameStart := 80870 },
  { event := event80889
    frameStart := 80870 },
  { event := event80890
    frameStart := 80870 },
  { event := event80891
    frameStart := 80870 },
  { event := event80892
    frameStart := 80870 },
  { event := event80893
    frameStart := 80870 },
  { event := event80894
    frameStart := 80870 },
  { event := event80895
    frameStart := 80870 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events315
