import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events284

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event72704 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10971⟩⟩) 0 ⟨10970⟩ 72703

def event72705 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10971⟩⟩) (.identity (.predecessor 0 72704 .coefficient))

def event72706 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10971⟩⟩) (.finite 16)

def event72707 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23035⟩⟩) 0 ⟨10971⟩ 72706

def event72708 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23035⟩⟩) (.authority (.programFamilyFact))

def event72709 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23035⟩⟩) (.finite 3720)

def event72710 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event72711 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23036⟩⟩) 0 ⟨6689⟩ 72710

def event72712 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23036⟩⟩) 1 ⟨23035⟩ 72709

def event72713 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23036⟩⟩) (.authority (.operator))

def exact72714RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23036⟩⟩]⟩, (1)⟩]

theorem exact72714RawTermsValid :
    exact72714RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72714 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23036⟩⟩) exact72714RawTerms .large 72713 .exactZero (none)

def event72715 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25060⟩⟩) 0 ⟨23036⟩ 72714

def event72716 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25060⟩⟩) (.authority (.operator))

def exact72717RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25060⟩⟩]⟩, (1)⟩]

theorem exact72717RawTermsValid :
    exact72717RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72717 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25060⟩⟩) exact72717RawTerms (.finite 8192) 72716 .exactZero (none)

def event72718 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event72719 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event72720 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11069⟩⟩) 0 ⟨10971⟩ 72706

def event72721 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11069⟩⟩) 1 ⟨110⟩ 72719

def event72722 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11069⟩⟩) (.sum [.predecessor 0 72720 .coefficient, .predecessor 1 72721 .coefficient])

def event72723 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11069⟩⟩) (.finite 16)

def event72724 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11070⟩⟩) 0 ⟨11069⟩ 72723

def event72725 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11070⟩⟩) (.identity (.predecessor 0 72724 .coefficient))

def exact72726RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], []⟩, (1)⟩]

theorem exact72726RawTermsValid :
    exact72726RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72726 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11070⟩⟩) exact72726RawTerms (.finite 16) 72725 .exactZero (none)

def event72727 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact72728RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact72728RawTermsValid :
    exact72728RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72728 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact72728RawTerms .large 72727 .exactZero (none)

def event72729 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11071⟩⟩) 0 ⟨6544⟩ 72728

def event72730 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11071⟩⟩) 1 ⟨11070⟩ 72726

def event72731 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11071⟩⟩) (.product (.predecessor 0 72729 .coefficient) (.predecessor 1 72730 .coefficient) (⟨false, false, none, none, none⟩))

def event72732 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11071⟩⟩, .operator (⟨72728, 0⟩, ⟨72726, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact72733RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact72733RawTermsValid :
    exact72733RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72733 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11071⟩⟩) exact72733RawTerms .large 72731 .exactZero (none)

def event72734 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event72735 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event72736 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 72710

def event72737 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact72738RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact72738RawTermsValid :
    exact72738RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72738 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact72738RawTerms .large 72737 .exactZero (none)

def event72739 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6774⟩⟩) 0 ⟨6757⟩ 72738

def event72740 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6774⟩⟩) (.identity (.predecessor 0 72739 .coefficient))

def exact72741RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6774⟩⟩]⟩, (1)⟩]

theorem exact72741RawTermsValid :
    exact72741RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72741 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6774⟩⟩) exact72741RawTerms .large 72740 .exactZero (none)

def event72742 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7837⟩⟩) 0 ⟨6774⟩ 72741

def event72743 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7837⟩⟩) (.authority (.operator))

def exact72744RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩, (1)⟩]

theorem exact72744RawTermsValid :
    exact72744RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72744 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7837⟩⟩) exact72744RawTerms (.finite 8192) 72743 .exactZero (none)

def event72745 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7838⟩⟩) 0 ⟨7837⟩ 72744

def event72746 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7838⟩⟩) 1 ⟨2348⟩ 72735

def event72747 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7838⟩⟩) (.scale (.predecessor 0 72745 .coefficient) (.value (.predecessor 1 72746 .coefficient)))

def exact72748RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩, (1)⟩]

theorem exact72748RawTermsValid :
    exact72748RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72748 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7838⟩⟩) exact72748RawTerms (.finite 8192) 72747 .exactZero (none)

def event72749 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6791⟩⟩) 0 ⟨6757⟩ 72738

def event72750 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6791⟩⟩) (.identity (.predecessor 0 72749 .coefficient))

def exact72751RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩]⟩, (1)⟩]

theorem exact72751RawTermsValid :
    exact72751RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72751 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6791⟩⟩) exact72751RawTerms .large 72750 .exactZero (none)

def event72752 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7839⟩⟩) 0 ⟨6791⟩ 72751

def event72753 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7839⟩⟩) 1 ⟨7838⟩ 72748

def event72754 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7839⟩⟩) (.product (.predecessor 0 72752 .coefficient) (.predecessor 1 72753 .coefficient) (⟨false, false, none, none, none⟩))

def event72755 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7839⟩⟩, .operator (⟨72751, 0⟩, ⟨72748, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩, (1)⟩)

def exact72756RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩, (1)⟩]

theorem exact72756RawTermsValid :
    exact72756RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72756 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7839⟩⟩) exact72756RawTerms .large 72754 .exactZero (none)

def event72757 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11072⟩⟩) 0 ⟨7839⟩ 72756

def event72758 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11072⟩⟩) 1 ⟨11071⟩ 72733

def event72759 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11072⟩⟩) (.sum [.predecessor 0 72757 .coefficient, .predecessor 1 72758 .coefficient])

def exact72760RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact72760RawTermsValid :
    exact72760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72760 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11072⟩⟩) exact72760RawTerms .large 72759 .exactZero (none)

def event72761 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25063⟩⟩) 0 ⟨11072⟩ 72760

def event72762 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25063⟩⟩) 1 ⟨25060⟩ 72717

def event72763 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25063⟩⟩) (.product (.predecessor 0 72761 .coefficient) (.predecessor 1 72762 .coefficient) (⟨false, false, none, none, none⟩))

def event72764 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25063⟩⟩, .operator (⟨72760, 0⟩, ⟨72717, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25060⟩⟩]⟩, (1)⟩)

def event72765 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25063⟩⟩, .operator (⟨72760, 1⟩, ⟨72717, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25060⟩⟩]⟩, (-1)⟩)

def event72766 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25063⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25060⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25060⟩⟩) ⟨23036⟩ 72714)

def event72767 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25063⟩⟩, .relation 72766 0, ⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], [⟨.program ⟨214⟩, ⟨23036⟩⟩]⟩, (-1)⟩)

def exact72768RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25060⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], [⟨.program ⟨214⟩, ⟨23036⟩⟩]⟩, (-1)⟩]

theorem exact72768RawTermsValid :
    exact72768RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72768 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25063⟩⟩) exact72768RawTerms .large 72763 .exactZero (none)

def event72769 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15110⟩⟩) 0 ⟨10971⟩ 72706

def event72770 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15110⟩⟩) (.authority (.programFamilyFact))

def exact72771RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], []⟩, (1)⟩]

theorem exact72771RawTermsValid :
    exact72771RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72771 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15110⟩⟩) exact72771RawTerms (.finite 4) 72770 .exactZero (none)

def event72772 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15112⟩⟩) 0 ⟨6544⟩ 72728

def event72773 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15112⟩⟩) 1 ⟨15110⟩ 72771

def event72774 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15112⟩⟩) (.product (.predecessor 0 72772 .coefficient) (.predecessor 1 72773 .coefficient) (⟨false, true, none, none, some 1⟩))

def event72775 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15112⟩⟩, .operator (⟨72728, 0⟩, ⟨72771, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact72776RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact72776RawTermsValid :
    exact72776RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72776 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15112⟩⟩) exact72776RawTerms .large 72774 .exactZero (none)

def event72777 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6692⟩⟩) 0 ⟨6689⟩ 72710

def event72778 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6692⟩⟩) (.authority (.operator))

def exact72779RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩]

theorem exact72779RawTermsValid :
    exact72779RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72779 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6692⟩⟩) exact72779RawTerms .large 72778 .exactZero (none)

def event72780 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15113⟩⟩) 0 ⟨6692⟩ 72779

def event72781 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15113⟩⟩) 1 ⟨15112⟩ 72776

def event72782 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15113⟩⟩) (.sum [.predecessor 0 72780 .coefficient, .predecessor 1 72781 .coefficient])

def exact72783RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact72783RawTermsValid :
    exact72783RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72783 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15113⟩⟩) exact72783RawTerms .large 72782 .exactZero (none)

def event72784 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25064⟩⟩) 0 ⟨15113⟩ 72783

def event72785 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25064⟩⟩) 1 ⟨25063⟩ 72768

def event72786 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25064⟩⟩) (.sum [.predecessor 0 72784 .coefficient, .predecessor 1 72785 .coefficient])

def exact72787RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25060⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], [⟨.program ⟨214⟩, ⟨23036⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact72787RawTermsValid :
    exact72787RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72787 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25064⟩⟩) exact72787RawTerms .large 72786 .exactZero (none)

def event72788 : Event := .preFoldPolynomial 72787 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25060⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], [⟨.program ⟨214⟩, ⟨23036⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact72789RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25060⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], [⟨.program ⟨214⟩, ⟨23036⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event72789 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25064⟩⟩) 72788 exact72789RawTerms .large 72786 .exactZero (none)

def event72790 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨10971⟩⟩) ⟨⟨105⟩, ⟨9⟩, ⟨109⟩⟩ ⟨72624, 72790⟩

def event72791 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19167⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19164⟩⟩]⟩) (1) 0 2 (.universal 72790 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19164⟩⟩]⟩) (none) 72789)

def event72792 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19167⟩⟩, .relation 72791 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩)

def event72793 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19167⟩⟩, .relation 72791 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25060⟩⟩]⟩, (-1)⟩)

def event72794 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19167⟩⟩, .relation 72791 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], [⟨.program ⟨214⟩, ⟨23036⟩⟩]⟩, (1)⟩)

def event72795 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19167⟩⟩, .relation 72791 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact72796RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25060⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], [⟨.program ⟨214⟩, ⟨23036⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact72796RawTermsValid :
    exact72796RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72796 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19167⟩⟩) exact72796RawTerms .large 72620 (.finite 1811303510016) (some (72622))

def event72797 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25062⟩⟩) 0 ⟨19167⟩ 72796

def event72798 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25062⟩⟩) 1 ⟨25061⟩ 72610

def event72799 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25062⟩⟩) (.sum [.predecessor 0 72797 .coefficient, .predecessor 1 72798 .coefficient])

def event72800 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25062⟩⟩, .operator (⟨72796, 2⟩, ⟨72610, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], [⟨.program ⟨214⟩, ⟨23036⟩⟩]⟩, (-1)⟩)

def event72801 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25062⟩⟩, .operator (⟨72796, 1⟩, ⟨72610, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25060⟩⟩]⟩, (1)⟩)

def event72802 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25062⟩⟩) (.sum [.result 72796 .summary, .result 72610 .summary])

def exact72803RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact72803RawTermsValid :
    exact72803RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72803 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25062⟩⟩) exact72803RawTerms .large 72799 (.finite 352017970769920) (some (72802))

def event72804 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26770⟩⟩) 0 ⟨25062⟩ 72803

def event72805 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26770⟩⟩) 1 ⟨26768⟩ 72526

def event72806 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26770⟩⟩) (.product (.predecessor 0 72804 .coefficient) (.predecessor 1 72805 .coefficient) (⟨false, false, none, none, none⟩))

def event72807 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26770⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩) [⟨.result 72526 .coefficient, false, none⟩])

def event72808 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26770⟩⟩) (.product (.result 72803 .summary) (.transfer 72807) (⟨false, false, none, none, none⟩))

def event72809 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26770⟩⟩, .operator (⟨72803, 0⟩, ⟨72526, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩, (1)⟩)

def event72810 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26770⟩⟩, .operator (⟨72803, 1⟩, ⟨72526, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩, (-1)⟩)

def event72811 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26770⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26768⟩⟩) ⟨23844⟩ 72523)

def event72812 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26770⟩⟩, .relation 72811 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23844⟩⟩]⟩, (-1)⟩)

def exact72813RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨23844⟩⟩]⟩, (-1)⟩]

theorem exact72813RawTermsValid :
    exact72813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72813 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26770⟩⟩) exact72813RawTerms .large 72806 (.finite 1291911585013138718720) (some (72808))

def event72814 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20676⟩⟩) 0 ⟨15111⟩ 3448

def event72815 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20676⟩⟩) (.authority (.relationPreimageSource ⟨32⟩))

def exact72816RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩, (1)⟩]

theorem exact72816RawTermsValid :
    exact72816RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72816 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20676⟩⟩) exact72816RawTerms (.finite 136065468) 72815 .exactZero (none)

def event72817 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20678⟩⟩) 0 ⟨20676⟩ 72816

def event72818 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20678⟩⟩) 1 ⟨2348⟩ 4

def event72819 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20678⟩⟩) (.scale (.predecessor 0 72817 .coefficient) (.value (.predecessor 1 72818 .coefficient)))

def exact72820RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩, (1)⟩]

theorem exact72820RawTermsValid :
    exact72820RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72820 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20678⟩⟩) exact72820RawTerms (.finite 136065468) 72819 .exactZero (none)

def event72821 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20679⟩⟩) 0 ⟨5535⟩ 65387

def event72822 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20679⟩⟩) 1 ⟨20678⟩ 72820

def event72823 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20679⟩⟩) (.product (.predecessor 0 72821 .coefficient) (.predecessor 1 72822 .coefficient) (⟨false, false, none, none, none⟩))

def event72824 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20679⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩) [⟨.result 72816 .coefficient, false, none⟩])

def event72825 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20679⟩⟩) (.product (.result 65387 .summary) (.transfer 72824) (⟨false, false, none, none, none⟩))

def event72826 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20679⟩⟩, .operator (⟨65387, 0⟩, ⟨72820, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩, (1)⟩)

def event72827 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20677⟩⟩)

def event72828 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event72829 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event72830 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event72831 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event72832 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event72833 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event72834 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event72835 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event72836 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 72835

def event72837 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 72833

def event72838 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 72836 .coefficient) (.value (.predecessor 1 72837 .coefficient)))

def event72839 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event72840 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 72839

def event72841 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 72831

def event72842 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 72840 .coefficient, .predecessor 1 72841 .coefficient])

def event72843 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event72844 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 72843

def event72845 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 72829

def event72846 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 72845 .coefficient))

def event72847 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event72848 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10969⟩⟩) 0 ⟨5530⟩ 72847

def event72849 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10969⟩⟩) (.authority (.programFamilyFact))

def exact72850RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10969⟩⟩], []⟩, (1)⟩]

theorem exact72850RawTermsValid :
    exact72850RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72850 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10969⟩⟩) exact72850RawTerms (.finite 4) 72849 .exactZero (none)

def event72851 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10837⟩⟩) 0 ⟨5530⟩ 72847

def event72852 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10837⟩⟩) (.authority (.programFamilyFact))

def exact72853RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩], []⟩, (1)⟩]

theorem exact72853RawTermsValid :
    exact72853RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72853 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10837⟩⟩) exact72853RawTerms (.finite 4) 72852 .exactZero (none)

def event72854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10970⟩⟩) 0 ⟨10837⟩ 72853

def event72855 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10970⟩⟩) 1 ⟨10969⟩ 72850

def event72856 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10970⟩⟩) (.product (.predecessor 0 72854 .coefficient) (.predecessor 1 72855 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event72857 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10970⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], []⟩) [⟨.result 72853 .coefficient, true, some 1⟩, ⟨.result 72850 .coefficient, true, some 1⟩])

def event72858 : Event := .survivorFold (1) 72857

def exact72859RawTerms : List Term := []

theorem exact72859RawTermsValid :
    exact72859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72859 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10970⟩⟩) exact72859RawTerms (.finite 16) 72856 (.finite 16) (some (72857))

def event72860 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10971⟩⟩) 0 ⟨10970⟩ 72859

def event72861 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10971⟩⟩) (.identity (.predecessor 0 72860 .coefficient))

def event72862 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10971⟩⟩) (.finite 16)

def event72863 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15110⟩⟩) 0 ⟨10971⟩ 72862

def event72864 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15110⟩⟩) (.authority (.programFamilyFact))

def exact72865RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], []⟩, (1)⟩]

theorem exact72865RawTermsValid :
    exact72865RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72865 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15110⟩⟩) exact72865RawTerms (.finite 4) 72864 .exactZero (none)

def event72866 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15111⟩⟩) 0 ⟨15110⟩ 72865

def event72867 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15111⟩⟩) (.identity (.predecessor 0 72866 .coefficient))

def event72868 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15111⟩⟩) (.finite 4)

def event72869 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20676⟩⟩) 0 ⟨15111⟩ 72868

def event72870 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20676⟩⟩) (.authority (.relationPreimageSource ⟨32⟩))

def exact72871RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩, (1)⟩]

theorem exact72871RawTermsValid :
    exact72871RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72871 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20676⟩⟩) exact72871RawTerms (.finite 136065468) 72870 .exactZero (none)

def event72872 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact72873RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact72873RawTermsValid :
    exact72873RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72873 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact72873RawTerms .large 72872 .exactZero (none)

def event72874 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20677⟩⟩) 0 ⟨6⟩ 72873

def event72875 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20677⟩⟩) 1 ⟨20676⟩ 72871

def event72876 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20677⟩⟩) (.product (.predecessor 0 72874 .coefficient) (.predecessor 1 72875 .coefficient) (⟨false, false, none, none, none⟩))

def event72877 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20677⟩⟩, .operator (⟨72873, 0⟩, ⟨72871, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩, (1)⟩)

def exact72878RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩, (1)⟩]

theorem exact72878RawTermsValid :
    exact72878RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72878 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20677⟩⟩) exact72878RawTerms .large 72876 .exactZero (none)

def event72879 : Event := .preFoldPolynomial 72878 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩, (1)⟩] .exactZero none

def exact72880RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩, (1)⟩]

def event72880 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20677⟩⟩) 72879 exact72880RawTerms .large 72876 .exactZero (none)

def event72881 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26773⟩⟩)

def event72882 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event72883 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event72884 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event72885 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event72886 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event72887 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event72888 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event72889 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event72890 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 72889

def event72891 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 72887

def event72892 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 72890 .coefficient) (.value (.predecessor 1 72891 .coefficient)))

def event72893 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event72894 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 72893

def event72895 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 72885

def event72896 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 72894 .coefficient, .predecessor 1 72895 .coefficient])

def event72897 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event72898 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 72897

def event72899 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 72883

def event72900 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 72899 .coefficient))

def event72901 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event72902 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10969⟩⟩) 0 ⟨5530⟩ 72901

def event72903 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10969⟩⟩) (.authority (.programFamilyFact))

def exact72904RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10969⟩⟩], []⟩, (1)⟩]

theorem exact72904RawTermsValid :
    exact72904RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72904 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10969⟩⟩) exact72904RawTerms (.finite 4) 72903 .exactZero (none)

def event72905 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10837⟩⟩) 0 ⟨5530⟩ 72901

def event72906 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10837⟩⟩) (.authority (.programFamilyFact))

def exact72907RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩], []⟩, (1)⟩]

theorem exact72907RawTermsValid :
    exact72907RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72907 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10837⟩⟩) exact72907RawTerms (.finite 4) 72906 .exactZero (none)

def event72908 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10970⟩⟩) 0 ⟨10837⟩ 72907

def event72909 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10970⟩⟩) 1 ⟨10969⟩ 72904

def event72910 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10970⟩⟩) (.product (.predecessor 0 72908 .coefficient) (.predecessor 1 72909 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event72911 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10970⟩⟩, .operator (⟨72907, 0⟩, ⟨72904, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], []⟩, (1)⟩)

def exact72912RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], []⟩, (1)⟩]

theorem exact72912RawTermsValid :
    exact72912RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72912 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10970⟩⟩) exact72912RawTerms (.finite 16) 72910 .exactZero (none)

def event72913 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10971⟩⟩) 0 ⟨10970⟩ 72912

def event72914 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10971⟩⟩) (.identity (.predecessor 0 72913 .coefficient))

def event72915 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10971⟩⟩) (.finite 16)

def event72916 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15110⟩⟩) 0 ⟨10971⟩ 72915

def event72917 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15110⟩⟩) (.authority (.programFamilyFact))

def exact72918RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], []⟩, (1)⟩]

theorem exact72918RawTermsValid :
    exact72918RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72918 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15110⟩⟩) exact72918RawTerms (.finite 4) 72917 .exactZero (none)

def event72919 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15111⟩⟩) 0 ⟨15110⟩ 72918

def event72920 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15111⟩⟩) (.identity (.predecessor 0 72919 .coefficient))

def event72921 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15111⟩⟩) (.finite 4)

def event72922 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23842⟩⟩) 0 ⟨15111⟩ 72921

def event72923 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23842⟩⟩) (.authority (.programFamilyFact))

def event72924 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23842⟩⟩) (.finite 3720)

def event72925 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event72926 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23844⟩⟩) 0 ⟨6689⟩ 72925

def event72927 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23844⟩⟩) 1 ⟨23842⟩ 72924

def event72928 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23844⟩⟩) (.authority (.operator))

def exact72929RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23844⟩⟩]⟩, (1)⟩]

theorem exact72929RawTermsValid :
    exact72929RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72929 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23844⟩⟩) exact72929RawTerms .large 72928 .exactZero (none)

def event72930 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26768⟩⟩) 0 ⟨23844⟩ 72929

def event72931 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26768⟩⟩) (.authority (.operator))

def exact72932RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩, (1)⟩]

theorem exact72932RawTermsValid :
    exact72932RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72932 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26768⟩⟩) exact72932RawTerms (.finite 8192) 72931 .exactZero (none)

def event72933 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event72934 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event72935 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15150⟩⟩) 0 ⟨15111⟩ 72921

def event72936 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15150⟩⟩) 1 ⟨110⟩ 72934

def event72937 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15150⟩⟩) (.sum [.predecessor 0 72935 .coefficient, .predecessor 1 72936 .coefficient])

def event72938 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15150⟩⟩) (.finite 4)

def event72939 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15151⟩⟩) 0 ⟨15150⟩ 72938

def event72940 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15151⟩⟩) (.identity (.predecessor 0 72939 .coefficient))

def exact72941RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], []⟩, (1)⟩]

theorem exact72941RawTermsValid :
    exact72941RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72941 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15151⟩⟩) exact72941RawTerms (.finite 4) 72940 .exactZero (none)

def event72942 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact72943RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact72943RawTermsValid :
    exact72943RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72943 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact72943RawTerms .large 72942 .exactZero (none)

def event72944 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15152⟩⟩) 0 ⟨6544⟩ 72943

def event72945 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15152⟩⟩) 1 ⟨15151⟩ 72941

def event72946 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15152⟩⟩) (.product (.predecessor 0 72944 .coefficient) (.predecessor 1 72945 .coefficient) (⟨false, false, none, none, none⟩))

def event72947 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15152⟩⟩, .operator (⟨72943, 0⟩, ⟨72941, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact72948RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact72948RawTermsValid :
    exact72948RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72948 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15152⟩⟩) exact72948RawTerms .large 72946 .exactZero (none)

def event72949 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6692⟩⟩) 0 ⟨6689⟩ 72925

def event72950 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6692⟩⟩) (.authority (.operator))

def exact72951RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩]

theorem exact72951RawTermsValid :
    exact72951RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72951 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6692⟩⟩) exact72951RawTerms .large 72950 .exactZero (none)

def event72952 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15153⟩⟩) 0 ⟨6692⟩ 72951

def event72953 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15153⟩⟩) 1 ⟨15152⟩ 72948

def event72954 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15153⟩⟩) (.sum [.predecessor 0 72952 .coefficient, .predecessor 1 72953 .coefficient])

def exact72955RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15110⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact72955RawTermsValid :
    exact72955RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event72955 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15153⟩⟩) exact72955RawTerms .large 72954 .exactZero (none)

def event72956 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26769⟩⟩) 0 ⟨15153⟩ 72955

def event72957 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26769⟩⟩) 1 ⟨26768⟩ 72932

def event72958 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26769⟩⟩) (.product (.predecessor 0 72956 .coefficient) (.predecessor 1 72957 .coefficient) (⟨false, false, none, none, none⟩))

def event72959 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26769⟩⟩, .operator (⟨72955, 0⟩, ⟨72932, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩, (1)⟩)

def eventLeaf4544 : Array AnnotatedEvent := #[
  { event := event72704
    frameStart := 72672 },
  { event := event72705
    frameStart := 72672 },
  { event := event72706
    frameStart := 72672 },
  { event := event72707
    frameStart := 72672 },
  { event := event72708
    frameStart := 72672 },
  { event := event72709
    frameStart := 72672 },
  { event := event72710
    frameStart := 72672 },
  { event := event72711
    frameStart := 72672 },
  { event := event72712
    frameStart := 72672 },
  { event := event72713
    frameStart := 72672 },
  { event := event72714
    frameStart := 72672 },
  { event := event72715
    frameStart := 72672 },
  { event := event72716
    frameStart := 72672 },
  { event := event72717
    frameStart := 72672 },
  { event := event72718
    frameStart := 72672 },
  { event := event72719
    frameStart := 72672 }
]

def eventLeaf4545 : Array AnnotatedEvent := #[
  { event := event72720
    frameStart := 72672 },
  { event := event72721
    frameStart := 72672 },
  { event := event72722
    frameStart := 72672 },
  { event := event72723
    frameStart := 72672 },
  { event := event72724
    frameStart := 72672 },
  { event := event72725
    frameStart := 72672 },
  { event := event72726
    frameStart := 72672 },
  { event := event72727
    frameStart := 72672 },
  { event := event72728
    frameStart := 72672 },
  { event := event72729
    frameStart := 72672 },
  { event := event72730
    frameStart := 72672 },
  { event := event72731
    frameStart := 72672 },
  { event := event72732
    frameStart := 72672 },
  { event := event72733
    frameStart := 72672 },
  { event := event72734
    frameStart := 72672 },
  { event := event72735
    frameStart := 72672 }
]

def eventLeaf4546 : Array AnnotatedEvent := #[
  { event := event72736
    frameStart := 72672 },
  { event := event72737
    frameStart := 72672 },
  { event := event72738
    frameStart := 72672 },
  { event := event72739
    frameStart := 72672 },
  { event := event72740
    frameStart := 72672 },
  { event := event72741
    frameStart := 72672 },
  { event := event72742
    frameStart := 72672 },
  { event := event72743
    frameStart := 72672 },
  { event := event72744
    frameStart := 72672 },
  { event := event72745
    frameStart := 72672 },
  { event := event72746
    frameStart := 72672 },
  { event := event72747
    frameStart := 72672 },
  { event := event72748
    frameStart := 72672 },
  { event := event72749
    frameStart := 72672 },
  { event := event72750
    frameStart := 72672 },
  { event := event72751
    frameStart := 72672 }
]

def eventLeaf4547 : Array AnnotatedEvent := #[
  { event := event72752
    frameStart := 72672 },
  { event := event72753
    frameStart := 72672 },
  { event := event72754
    frameStart := 72672 },
  { event := event72755
    frameStart := 72672 },
  { event := event72756
    frameStart := 72672 },
  { event := event72757
    frameStart := 72672 },
  { event := event72758
    frameStart := 72672 },
  { event := event72759
    frameStart := 72672 },
  { event := event72760
    frameStart := 72672 },
  { event := event72761
    frameStart := 72672 },
  { event := event72762
    frameStart := 72672 },
  { event := event72763
    frameStart := 72672 },
  { event := event72764
    frameStart := 72672 },
  { event := event72765
    frameStart := 72672 },
  { event := event72766
    frameStart := 72672 },
  { event := event72767
    frameStart := 72672 }
]

def eventLeaf4548 : Array AnnotatedEvent := #[
  { event := event72768
    frameStart := 72672 },
  { event := event72769
    frameStart := 72672 },
  { event := event72770
    frameStart := 72672 },
  { event := event72771
    frameStart := 72672 },
  { event := event72772
    frameStart := 72672 },
  { event := event72773
    frameStart := 72672 },
  { event := event72774
    frameStart := 72672 },
  { event := event72775
    frameStart := 72672 },
  { event := event72776
    frameStart := 72672 },
  { event := event72777
    frameStart := 72672 },
  { event := event72778
    frameStart := 72672 },
  { event := event72779
    frameStart := 72672 },
  { event := event72780
    frameStart := 72672 },
  { event := event72781
    frameStart := 72672 },
  { event := event72782
    frameStart := 72672 },
  { event := event72783
    frameStart := 72672 }
]

def eventLeaf4549 : Array AnnotatedEvent := #[
  { event := event72784
    frameStart := 72672 },
  { event := event72785
    frameStart := 72672 },
  { event := event72786
    frameStart := 72672 },
  { event := event72787
    frameStart := 72672 },
  { event := event72788
    frameStart := 72672 },
  { event := event72789
    frameStart := 72672 },
  { event := event72790
    frameStart := 0 },
  { event := event72791
    frameStart := 0 },
  { event := event72792
    frameStart := 0 },
  { event := event72793
    frameStart := 0 },
  { event := event72794
    frameStart := 0 },
  { event := event72795
    frameStart := 0 },
  { event := event72796
    frameStart := 0 },
  { event := event72797
    frameStart := 0 },
  { event := event72798
    frameStart := 0 },
  { event := event72799
    frameStart := 0 }
]

def eventLeaf4550 : Array AnnotatedEvent := #[
  { event := event72800
    frameStart := 0 },
  { event := event72801
    frameStart := 0 },
  { event := event72802
    frameStart := 0 },
  { event := event72803
    frameStart := 0 },
  { event := event72804
    frameStart := 0 },
  { event := event72805
    frameStart := 0 },
  { event := event72806
    frameStart := 0 },
  { event := event72807
    frameStart := 0 },
  { event := event72808
    frameStart := 0 },
  { event := event72809
    frameStart := 0 },
  { event := event72810
    frameStart := 0 },
  { event := event72811
    frameStart := 0 },
  { event := event72812
    frameStart := 0 },
  { event := event72813
    frameStart := 0 },
  { event := event72814
    frameStart := 0 },
  { event := event72815
    frameStart := 0 }
]

def eventLeaf4551 : Array AnnotatedEvent := #[
  { event := event72816
    frameStart := 0 },
  { event := event72817
    frameStart := 0 },
  { event := event72818
    frameStart := 0 },
  { event := event72819
    frameStart := 0 },
  { event := event72820
    frameStart := 0 },
  { event := event72821
    frameStart := 0 },
  { event := event72822
    frameStart := 0 },
  { event := event72823
    frameStart := 0 },
  { event := event72824
    frameStart := 0 },
  { event := event72825
    frameStart := 0 },
  { event := event72826
    frameStart := 0 },
  { event := event72827
    frameStart := 72827 },
  { event := event72828
    frameStart := 72827 },
  { event := event72829
    frameStart := 72827 },
  { event := event72830
    frameStart := 72827 },
  { event := event72831
    frameStart := 72827 }
]

def eventLeaf4552 : Array AnnotatedEvent := #[
  { event := event72832
    frameStart := 72827 },
  { event := event72833
    frameStart := 72827 },
  { event := event72834
    frameStart := 72827 },
  { event := event72835
    frameStart := 72827 },
  { event := event72836
    frameStart := 72827 },
  { event := event72837
    frameStart := 72827 },
  { event := event72838
    frameStart := 72827 },
  { event := event72839
    frameStart := 72827 },
  { event := event72840
    frameStart := 72827 },
  { event := event72841
    frameStart := 72827 },
  { event := event72842
    frameStart := 72827 },
  { event := event72843
    frameStart := 72827 },
  { event := event72844
    frameStart := 72827 },
  { event := event72845
    frameStart := 72827 },
  { event := event72846
    frameStart := 72827 },
  { event := event72847
    frameStart := 72827 }
]

def eventLeaf4553 : Array AnnotatedEvent := #[
  { event := event72848
    frameStart := 72827 },
  { event := event72849
    frameStart := 72827 },
  { event := event72850
    frameStart := 72827 },
  { event := event72851
    frameStart := 72827 },
  { event := event72852
    frameStart := 72827 },
  { event := event72853
    frameStart := 72827 },
  { event := event72854
    frameStart := 72827 },
  { event := event72855
    frameStart := 72827 },
  { event := event72856
    frameStart := 72827 },
  { event := event72857
    frameStart := 72827 },
  { event := event72858
    frameStart := 72827 },
  { event := event72859
    frameStart := 72827 },
  { event := event72860
    frameStart := 72827 },
  { event := event72861
    frameStart := 72827 },
  { event := event72862
    frameStart := 72827 },
  { event := event72863
    frameStart := 72827 }
]

def eventLeaf4554 : Array AnnotatedEvent := #[
  { event := event72864
    frameStart := 72827 },
  { event := event72865
    frameStart := 72827 },
  { event := event72866
    frameStart := 72827 },
  { event := event72867
    frameStart := 72827 },
  { event := event72868
    frameStart := 72827 },
  { event := event72869
    frameStart := 72827 },
  { event := event72870
    frameStart := 72827 },
  { event := event72871
    frameStart := 72827 },
  { event := event72872
    frameStart := 72827 },
  { event := event72873
    frameStart := 72827 },
  { event := event72874
    frameStart := 72827 },
  { event := event72875
    frameStart := 72827 },
  { event := event72876
    frameStart := 72827 },
  { event := event72877
    frameStart := 72827 },
  { event := event72878
    frameStart := 72827 },
  { event := event72879
    frameStart := 72827 }
]

def eventLeaf4555 : Array AnnotatedEvent := #[
  { event := event72880
    frameStart := 72827 },
  { event := event72881
    frameStart := 72881 },
  { event := event72882
    frameStart := 72881 },
  { event := event72883
    frameStart := 72881 },
  { event := event72884
    frameStart := 72881 },
  { event := event72885
    frameStart := 72881 },
  { event := event72886
    frameStart := 72881 },
  { event := event72887
    frameStart := 72881 },
  { event := event72888
    frameStart := 72881 },
  { event := event72889
    frameStart := 72881 },
  { event := event72890
    frameStart := 72881 },
  { event := event72891
    frameStart := 72881 },
  { event := event72892
    frameStart := 72881 },
  { event := event72893
    frameStart := 72881 },
  { event := event72894
    frameStart := 72881 },
  { event := event72895
    frameStart := 72881 }
]

def eventLeaf4556 : Array AnnotatedEvent := #[
  { event := event72896
    frameStart := 72881 },
  { event := event72897
    frameStart := 72881 },
  { event := event72898
    frameStart := 72881 },
  { event := event72899
    frameStart := 72881 },
  { event := event72900
    frameStart := 72881 },
  { event := event72901
    frameStart := 72881 },
  { event := event72902
    frameStart := 72881 },
  { event := event72903
    frameStart := 72881 },
  { event := event72904
    frameStart := 72881 },
  { event := event72905
    frameStart := 72881 },
  { event := event72906
    frameStart := 72881 },
  { event := event72907
    frameStart := 72881 },
  { event := event72908
    frameStart := 72881 },
  { event := event72909
    frameStart := 72881 },
  { event := event72910
    frameStart := 72881 },
  { event := event72911
    frameStart := 72881 }
]

def eventLeaf4557 : Array AnnotatedEvent := #[
  { event := event72912
    frameStart := 72881 },
  { event := event72913
    frameStart := 72881 },
  { event := event72914
    frameStart := 72881 },
  { event := event72915
    frameStart := 72881 },
  { event := event72916
    frameStart := 72881 },
  { event := event72917
    frameStart := 72881 },
  { event := event72918
    frameStart := 72881 },
  { event := event72919
    frameStart := 72881 },
  { event := event72920
    frameStart := 72881 },
  { event := event72921
    frameStart := 72881 },
  { event := event72922
    frameStart := 72881 },
  { event := event72923
    frameStart := 72881 },
  { event := event72924
    frameStart := 72881 },
  { event := event72925
    frameStart := 72881 },
  { event := event72926
    frameStart := 72881 },
  { event := event72927
    frameStart := 72881 }
]

def eventLeaf4558 : Array AnnotatedEvent := #[
  { event := event72928
    frameStart := 72881 },
  { event := event72929
    frameStart := 72881 },
  { event := event72930
    frameStart := 72881 },
  { event := event72931
    frameStart := 72881 },
  { event := event72932
    frameStart := 72881 },
  { event := event72933
    frameStart := 72881 },
  { event := event72934
    frameStart := 72881 },
  { event := event72935
    frameStart := 72881 },
  { event := event72936
    frameStart := 72881 },
  { event := event72937
    frameStart := 72881 },
  { event := event72938
    frameStart := 72881 },
  { event := event72939
    frameStart := 72881 },
  { event := event72940
    frameStart := 72881 },
  { event := event72941
    frameStart := 72881 },
  { event := event72942
    frameStart := 72881 },
  { event := event72943
    frameStart := 72881 }
]

def eventLeaf4559 : Array AnnotatedEvent := #[
  { event := event72944
    frameStart := 72881 },
  { event := event72945
    frameStart := 72881 },
  { event := event72946
    frameStart := 72881 },
  { event := event72947
    frameStart := 72881 },
  { event := event72948
    frameStart := 72881 },
  { event := event72949
    frameStart := 72881 },
  { event := event72950
    frameStart := 72881 },
  { event := event72951
    frameStart := 72881 },
  { event := event72952
    frameStart := 72881 },
  { event := event72953
    frameStart := 72881 },
  { event := event72954
    frameStart := 72881 },
  { event := event72955
    frameStart := 72881 },
  { event := event72956
    frameStart := 72881 },
  { event := event72957
    frameStart := 72881 },
  { event := event72958
    frameStart := 72881 },
  { event := event72959
    frameStart := 72881 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events284
