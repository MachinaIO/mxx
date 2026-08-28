import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events081

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event20736 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event20737 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 0 ⟨5503⟩ 20736

def event20738 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 1 ⟨5245⟩ 20728

def event20739 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.sum [.predecessor 0 20737 .coefficient, .predecessor 1 20738 .coefficient])

def event20740 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.finite 223)

def event20741 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 0 ⟨5516⟩ 20740

def event20742 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 1 ⟨961⟩ 20726

def event20743 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.identity (.predecessor 1 20742 .coefficient))

def event20744 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.finite 224)

def event20745 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10512⟩⟩) 0 ⟨5560⟩ 20744

def event20746 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10512⟩⟩) (.authority (.programFamilyFact))

def exact20747RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10512⟩⟩], []⟩, (1)⟩]

theorem exact20747RawTermsValid :
    exact20747RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20747 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10512⟩⟩) exact20747RawTerms (.finite 2) 20746 .exactZero (none)

def event20748 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9420⟩⟩) 0 ⟨5560⟩ 20744

def event20749 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9420⟩⟩) (.authority (.programFamilyFact))

def exact20750RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9420⟩⟩], []⟩, (1)⟩]

theorem exact20750RawTermsValid :
    exact20750RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20750 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9420⟩⟩) exact20750RawTerms (.finite 2) 20749 .exactZero (none)

def event20751 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10513⟩⟩) 0 ⟨9420⟩ 20750

def event20752 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10513⟩⟩) 1 ⟨10512⟩ 20747

def event20753 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10513⟩⟩) (.product (.predecessor 0 20751 .coefficient) (.predecessor 1 20752 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event20754 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10513⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9420⟩⟩, ⟨.program ⟨214⟩, ⟨10512⟩⟩], []⟩) [⟨.result 20750 .coefficient, true, some 1⟩, ⟨.result 20747 .coefficient, true, some 1⟩])

def event20755 : Event := .survivorFold (1) 20754

def exact20756RawTerms : List Term := []

theorem exact20756RawTermsValid :
    exact20756RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20756 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10513⟩⟩) exact20756RawTerms (.finite 4) 20753 (.finite 4) (some (20754))

def event20757 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10514⟩⟩) 0 ⟨10513⟩ 20756

def event20758 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10514⟩⟩) (.identity (.predecessor 0 20757 .coefficient))

def event20759 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10514⟩⟩) (.finite 4)

def event20760 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14808⟩⟩) 0 ⟨10514⟩ 20759

def event20761 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14808⟩⟩) (.authority (.programFamilyFact))

def exact20762RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], []⟩, (1)⟩]

theorem exact20762RawTermsValid :
    exact20762RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20762 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14808⟩⟩) exact20762RawTerms (.finite 2) 20761 .exactZero (none)

def event20763 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14809⟩⟩) 0 ⟨14808⟩ 20762

def event20764 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14809⟩⟩) (.identity (.predecessor 0 20763 .coefficient))

def event20765 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14809⟩⟩) (.finite 2)

def event20766 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20336⟩⟩) 0 ⟨14809⟩ 20765

def event20767 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20336⟩⟩) (.authority (.relationPreimageSource ⟨27⟩))

def exact20768RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20336⟩⟩]⟩, (1)⟩]

theorem exact20768RawTermsValid :
    exact20768RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20768 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20336⟩⟩) exact20768RawTerms (.finite 136065468) 20767 .exactZero (none)

def event20769 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact20770RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact20770RawTermsValid :
    exact20770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20770 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact20770RawTerms .large 20769 .exactZero (none)

def event20771 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20337⟩⟩) 0 ⟨6⟩ 20770

def event20772 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20337⟩⟩) 1 ⟨20336⟩ 20768

def event20773 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20337⟩⟩) (.product (.predecessor 0 20771 .coefficient) (.predecessor 1 20772 .coefficient) (⟨false, false, none, none, none⟩))

def event20774 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20337⟩⟩, .operator (⟨20770, 0⟩, ⟨20768, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20336⟩⟩]⟩, (1)⟩)

def exact20775RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20336⟩⟩]⟩, (1)⟩]

theorem exact20775RawTermsValid :
    exact20775RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20775 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20337⟩⟩) exact20775RawTerms .large 20773 .exactZero (none)

def event20776 : Event := .preFoldPolynomial 20775 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20336⟩⟩]⟩, (1)⟩] .exactZero none

def exact20777RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20336⟩⟩]⟩, (1)⟩]

def event20777 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20337⟩⟩) 20776 exact20777RawTerms .large 20773 .exactZero (none)

def event20778 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26405⟩⟩)

def event20779 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event20780 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event20781 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.authority (.operator))

def event20782 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.finite 6)

def event20783 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event20784 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event20785 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event20786 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event20787 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 20786

def event20788 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 20784

def event20789 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 20787 .coefficient) (.value (.predecessor 1 20788 .coefficient)))

def event20790 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event20791 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 0 ⟨5503⟩ 20790

def event20792 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 1 ⟨5245⟩ 20782

def event20793 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.sum [.predecessor 0 20791 .coefficient, .predecessor 1 20792 .coefficient])

def event20794 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.finite 223)

def event20795 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 0 ⟨5516⟩ 20794

def event20796 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 1 ⟨961⟩ 20780

def event20797 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.identity (.predecessor 1 20796 .coefficient))

def event20798 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.finite 224)

def event20799 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10512⟩⟩) 0 ⟨5560⟩ 20798

def event20800 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10512⟩⟩) (.authority (.programFamilyFact))

def exact20801RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10512⟩⟩], []⟩, (1)⟩]

theorem exact20801RawTermsValid :
    exact20801RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20801 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10512⟩⟩) exact20801RawTerms (.finite 2) 20800 .exactZero (none)

def event20802 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9420⟩⟩) 0 ⟨5560⟩ 20798

def event20803 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9420⟩⟩) (.authority (.programFamilyFact))

def exact20804RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9420⟩⟩], []⟩, (1)⟩]

theorem exact20804RawTermsValid :
    exact20804RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20804 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9420⟩⟩) exact20804RawTerms (.finite 2) 20803 .exactZero (none)

def event20805 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10513⟩⟩) 0 ⟨9420⟩ 20804

def event20806 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10513⟩⟩) 1 ⟨10512⟩ 20801

def event20807 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10513⟩⟩) (.product (.predecessor 0 20805 .coefficient) (.predecessor 1 20806 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event20808 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10513⟩⟩, .operator (⟨20804, 0⟩, ⟨20801, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9420⟩⟩, ⟨.program ⟨214⟩, ⟨10512⟩⟩], []⟩, (1)⟩)

def exact20809RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9420⟩⟩, ⟨.program ⟨214⟩, ⟨10512⟩⟩], []⟩, (1)⟩]

theorem exact20809RawTermsValid :
    exact20809RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20809 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10513⟩⟩) exact20809RawTerms (.finite 4) 20807 .exactZero (none)

def event20810 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10514⟩⟩) 0 ⟨10513⟩ 20809

def event20811 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10514⟩⟩) (.identity (.predecessor 0 20810 .coefficient))

def event20812 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10514⟩⟩) (.finite 4)

def event20813 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14808⟩⟩) 0 ⟨10514⟩ 20812

def event20814 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14808⟩⟩) (.authority (.programFamilyFact))

def exact20815RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], []⟩, (1)⟩]

theorem exact20815RawTermsValid :
    exact20815RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20815 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14808⟩⟩) exact20815RawTerms (.finite 2) 20814 .exactZero (none)

def event20816 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14809⟩⟩) 0 ⟨14808⟩ 20815

def event20817 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14809⟩⟩) (.identity (.predecessor 0 20816 .coefficient))

def event20818 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14809⟩⟩) (.finite 2)

def event20819 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23731⟩⟩) 0 ⟨14809⟩ 20818

def event20820 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23731⟩⟩) (.authority (.programFamilyFact))

def event20821 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23731⟩⟩) (.finite 3720)

def event20822 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event20823 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23732⟩⟩) 0 ⟨6689⟩ 20822

def event20824 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23732⟩⟩) 1 ⟨23731⟩ 20821

def event20825 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23732⟩⟩) (.authority (.operator))

def exact20826RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23732⟩⟩]⟩, (1)⟩]

theorem exact20826RawTermsValid :
    exact20826RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20826 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23732⟩⟩) exact20826RawTerms .large 20825 .exactZero (none)

def event20827 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26399⟩⟩) 0 ⟨23732⟩ 20826

def event20828 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26399⟩⟩) (.authority (.operator))

def exact20829RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26399⟩⟩]⟩, (1)⟩]

theorem exact20829RawTermsValid :
    exact20829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20829 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26399⟩⟩) exact20829RawTerms (.finite 8192) 20828 .exactZero (none)

def event20830 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event20831 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event20832 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14848⟩⟩) 0 ⟨14809⟩ 20818

def event20833 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14848⟩⟩) 1 ⟨110⟩ 20831

def event20834 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14848⟩⟩) (.sum [.predecessor 0 20832 .coefficient, .predecessor 1 20833 .coefficient])

def event20835 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14848⟩⟩) (.finite 2)

def event20836 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14849⟩⟩) 0 ⟨14848⟩ 20835

def event20837 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14849⟩⟩) (.identity (.predecessor 0 20836 .coefficient))

def exact20838RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], []⟩, (1)⟩]

theorem exact20838RawTermsValid :
    exact20838RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20838 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14849⟩⟩) exact20838RawTerms (.finite 2) 20837 .exactZero (none)

def event20839 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact20840RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact20840RawTermsValid :
    exact20840RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20840 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact20840RawTerms .large 20839 .exactZero (none)

def event20841 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14850⟩⟩) 0 ⟨6544⟩ 20840

def event20842 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14850⟩⟩) 1 ⟨14849⟩ 20838

def event20843 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14850⟩⟩) (.product (.predecessor 0 20841 .coefficient) (.predecessor 1 20842 .coefficient) (⟨false, false, none, none, none⟩))

def event20844 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14850⟩⟩, .operator (⟨20840, 0⟩, ⟨20838, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact20845RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact20845RawTermsValid :
    exact20845RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20845 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14850⟩⟩) exact20845RawTerms .large 20843 .exactZero (none)

def event20846 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6690⟩⟩) 0 ⟨6689⟩ 20822

def event20847 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6690⟩⟩) (.authority (.operator))

def exact20848RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩]

theorem exact20848RawTermsValid :
    exact20848RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20848 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6690⟩⟩) exact20848RawTerms .large 20847 .exactZero (none)

def event20849 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14851⟩⟩) 0 ⟨6690⟩ 20848

def event20850 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14851⟩⟩) 1 ⟨14850⟩ 20845

def event20851 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14851⟩⟩) (.sum [.predecessor 0 20849 .coefficient, .predecessor 1 20850 .coefficient])

def exact20852RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20852RawTermsValid :
    exact20852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20852 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14851⟩⟩) exact20852RawTerms .large 20851 .exactZero (none)

def event20853 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26400⟩⟩) 0 ⟨14851⟩ 20852

def event20854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26400⟩⟩) 1 ⟨26399⟩ 20829

def event20855 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26400⟩⟩) (.product (.predecessor 0 20853 .coefficient) (.predecessor 1 20854 .coefficient) (⟨false, false, none, none, none⟩))

def event20856 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26400⟩⟩, .operator (⟨20852, 1⟩, ⟨20829, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26399⟩⟩]⟩, (-1)⟩)

def event20857 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26400⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26399⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26399⟩⟩) ⟨23732⟩ 20826)

def event20858 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26400⟩⟩, .relation 20857 0, ⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨23732⟩⟩]⟩, (-1)⟩)

def event20859 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26400⟩⟩, .operator (⟨20852, 0⟩, ⟨20829, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26399⟩⟩]⟩, (1)⟩)

def exact20860RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26399⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨23732⟩⟩]⟩, (-1)⟩]

theorem exact20860RawTermsValid :
    exact20860RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20860 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26400⟩⟩) exact20860RawTerms .large 20855 .exactZero (none)

def event20861 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14906⟩⟩) 0 ⟨14809⟩ 20818

def event20862 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14906⟩⟩) (.authority (.programFamilyFact))

def exact20863RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14906⟩⟩], []⟩, (1)⟩]

theorem exact20863RawTermsValid :
    exact20863RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20863 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14906⟩⟩) exact20863RawTerms (.finite 2) 20862 .exactZero (none)

def event20864 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14909⟩⟩) 0 ⟨6544⟩ 20840

def event20865 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14909⟩⟩) 1 ⟨14906⟩ 20863

def event20866 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14909⟩⟩) (.product (.predecessor 0 20864 .coefficient) (.predecessor 1 20865 .coefficient) (⟨false, true, none, none, some 1⟩))

def event20867 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14909⟩⟩, .operator (⟨20840, 0⟩, ⟨20863, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact20868RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact20868RawTermsValid :
    exact20868RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20868 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14909⟩⟩) exact20868RawTerms .large 20866 .exactZero (none)

def event20869 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6708⟩⟩) 0 ⟨6689⟩ 20822

def event20870 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6708⟩⟩) (.authority (.operator))

def exact20871RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩]⟩, (1)⟩]

theorem exact20871RawTermsValid :
    exact20871RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20871 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6708⟩⟩) exact20871RawTerms .large 20870 .exactZero (none)

def event20872 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14910⟩⟩) 0 ⟨6708⟩ 20871

def event20873 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14910⟩⟩) 1 ⟨14909⟩ 20868

def event20874 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14910⟩⟩) (.sum [.predecessor 0 20872 .coefficient, .predecessor 1 20873 .coefficient])

def exact20875RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20875RawTermsValid :
    exact20875RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20875 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14910⟩⟩) exact20875RawTerms .large 20874 .exactZero (none)

def event20876 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26405⟩⟩) 0 ⟨14910⟩ 20875

def event20877 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26405⟩⟩) 1 ⟨26400⟩ 20860

def event20878 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26405⟩⟩) (.sum [.predecessor 0 20876 .coefficient, .predecessor 1 20877 .coefficient])

def exact20879RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26399⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨23732⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20879RawTermsValid :
    exact20879RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20879 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26405⟩⟩) exact20879RawTerms .large 20878 .exactZero (none)

def event20880 : Event := .preFoldPolynomial 20879 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26399⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨23732⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact20881RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26399⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨23732⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event20881 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26405⟩⟩) 20880 exact20881RawTerms .large 20878 .exactZero (none)

def event20882 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14809⟩⟩) ⟨⟨121⟩, ⟨27⟩, ⟨109⟩⟩ ⟨20724, 20882⟩

def event20883 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20339⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20336⟩⟩]⟩) (1) 0 2 (.universal 20882 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20336⟩⟩]⟩) (none) 20881)

def event20884 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20339⟩⟩, .relation 20883 1, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩]⟩, (1)⟩)

def event20885 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20339⟩⟩, .relation 20883 2, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨23732⟩⟩]⟩, (1)⟩)

def event20886 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20339⟩⟩, .relation 20883 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26399⟩⟩]⟩, (-1)⟩)

def event20887 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20339⟩⟩, .relation 20883 3, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact20888RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26399⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨23732⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20888RawTermsValid :
    exact20888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20888 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20339⟩⟩) exact20888RawTerms .large 20720 (.finite 1811303510016) (some (20722))

def event20889 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26402⟩⟩) 0 ⟨20339⟩ 20888

def event20890 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26402⟩⟩) 1 ⟨26401⟩ 20710

def event20891 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26402⟩⟩) (.sum [.predecessor 0 20889 .coefficient, .predecessor 1 20890 .coefficient])

def event20892 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26402⟩⟩, .operator (⟨20888, 2⟩, ⟨20710, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨23732⟩⟩]⟩, (-1)⟩)

def event20893 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26402⟩⟩, .operator (⟨20888, 0⟩, ⟨20710, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26399⟩⟩]⟩, (1)⟩)

def event20894 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26402⟩⟩) (.sum [.result 20888 .summary, .result 20710 .summary])

def exact20895RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20895RawTermsValid :
    exact20895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20895 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26402⟩⟩) exact20895RawTerms .large 20891 (.finite 1291889174379421642752) (some (20894))

def event20896 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26403⟩⟩) 0 ⟨26402⟩ 20895

def event20897 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26403⟩⟩) 1 ⟨6680⟩ 5859

def event20898 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26403⟩⟩) (.product (.predecessor 0 20896 .coefficient) (.predecessor 1 20897 .coefficient) (⟨false, false, none, none, none⟩))

def event20899 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26403⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩) [⟨.result 5855 .coefficient, false, none⟩])

def event20900 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26403⟩⟩) (.product (.result 20895 .summary) (.transfer 20899) (⟨false, false, none, none, none⟩))

def event20901 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26403⟩⟩, .operator (⟨20895, 0⟩, ⟨5859, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩)

def event20902 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26403⟩⟩, .operator (⟨20895, 1⟩, ⟨5859, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (-1)⟩)

def event20903 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26403⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6679⟩⟩) ⟨6611⟩ 5852)

def event20904 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26403⟩⟩, .relation 20903 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact20905RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20905RawTermsValid :
    exact20905RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20905 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26403⟩⟩) exact20905RawTerms .large 20898 (.finite 4741253940199267499646124032) (some (20900))

def event20906 : Event := .predecessor (⟨.program ⟨214⟩, ⟨74⟩⟩) 0 ⟨11⟩ 6441

def event20907 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨74⟩⟩) (.identity (.predecessor 0 20906 .coefficient))

def exact20908RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨74⟩⟩]⟩, (1)⟩]

theorem exact20908RawTermsValid :
    exact20908RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20908 : Event := .resultExact (⟨.program ⟨214⟩, ⟨74⟩⟩) exact20908RawTerms (.finite 26) 20907 .exactZero (none)

def event20909 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6630⟩⟩) 0 ⟨6378⟩ 723

def event20910 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6630⟩⟩) 1 ⟨6571⟩ 6449

def event20911 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6630⟩⟩) (.tensor (.predecessor 0 20909 .coefficient) (.predecessor 1 20910 .coefficient) true false)

def event20912 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨6630⟩⟩, .operator (⟨723, 0⟩, ⟨6449, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact20913RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact20913RawTermsValid :
    exact20913RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20913 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6630⟩⟩) exact20913RawTerms .large 20911 .exactZero (none)

def event20914 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7368⟩⟩) 0 ⟨5563⟩ 6314

def event20915 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7368⟩⟩) 1 ⟨6760⟩ 5873

def event20916 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7368⟩⟩) (.product (.predecessor 0 20914 .coefficient) (.predecessor 1 20915 .coefficient) (⟨false, false, none, none, none⟩))

def event20917 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7368⟩⟩, .operator (⟨6314, 0⟩, ⟨5873, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6760⟩⟩]⟩, (1)⟩)

def exact20918RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6760⟩⟩]⟩, (1)⟩]

theorem exact20918RawTermsValid :
    exact20918RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20918 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7368⟩⟩) exact20918RawTerms .large 20916 .exactZero (none)

def event20919 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7769⟩⟩) 0 ⟨7368⟩ 20918

def event20920 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7769⟩⟩) 1 ⟨6630⟩ 20913

def event20921 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7769⟩⟩) (.sum [.predecessor 0 20919 .coefficient, .predecessor 1 20920 .coefficient])

def exact20922RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6760⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20922RawTermsValid :
    exact20922RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20922 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7769⟩⟩) exact20922RawTerms .large 20921 .exactZero (none)

def event20923 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7770⟩⟩) 0 ⟨7769⟩ 20922

def event20924 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7770⟩⟩) 1 ⟨74⟩ 20908

def event20925 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7770⟩⟩) (.sum [.predecessor 0 20923 .coefficient, .predecessor 1 20924 .coefficient])

def event20926 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7770⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨74⟩⟩]⟩) [⟨.result 20908 .coefficient, false, none⟩])

def event20927 : Event := .survivorFold (1) 20926

def exact20928RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6760⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20928RawTermsValid :
    exact20928RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20928 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7770⟩⟩) exact20928RawTerms .large 20925 (.finite 26) (some (20926))

def event20929 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7812⟩⟩) 0 ⟨7770⟩ 20928

def event20930 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7812⟩⟩) 1 ⟨7770⟩ 20928

def event20931 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7812⟩⟩) (.sum [.predecessor 0 20929 .coefficient, .predecessor 1 20930 .coefficient])

def event20932 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7812⟩⟩, .operator (⟨20928, 1⟩, ⟨20928, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event20933 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7812⟩⟩, .operator (⟨20928, 0⟩, ⟨20928, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6760⟩⟩]⟩, (-1)⟩)

def event20934 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7812⟩⟩) (.sum [.result 20928 .summary, .result 20928 .summary])

def exact20935RawTerms : List Term := []

theorem exact20935RawTermsValid :
    exact20935RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20935 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7812⟩⟩) exact20935RawTerms .large 20931 (.finite 52) (some (20934))

def event20936 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26404⟩⟩) 0 ⟨7812⟩ 20935

def event20937 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26404⟩⟩) 1 ⟨26403⟩ 20905

def event20938 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26404⟩⟩) (.sum [.predecessor 0 20936 .coefficient, .predecessor 1 20937 .coefficient])

def event20939 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26404⟩⟩) (.sum [.result 20935 .summary, .result 20905 .summary])

def exact20940RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20940RawTermsValid :
    exact20940RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20940 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26404⟩⟩) exact20940RawTerms .large 20938 (.finite 4741253940199267499646124084) (some (20939))

def event20941 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26614⟩⟩) 0 ⟨26404⟩ 20940

def event20942 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26614⟩⟩) 1 ⟨26613⟩ 20693

def event20943 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26614⟩⟩) (.sum [.predecessor 0 20941 .coefficient, .predecessor 1 20942 .coefficient])

def event20944 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26614⟩⟩) (.sum [.result 20940 .summary, .result 20693 .summary])

def exact20945RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20945RawTermsValid :
    exact20945RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20945 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26614⟩⟩) exact20945RawTerms .large 20943 (.finite 9482549007414447334737575988) (some (20944))

def event20946 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26831⟩⟩) 0 ⟨26614⟩ 20945

def event20947 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26831⟩⟩) 1 ⟨26830⟩ 20481

def event20948 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26831⟩⟩) (.sum [.predecessor 0 20946 .coefficient, .predecessor 1 20947 .coefficient])

def event20949 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26831⟩⟩) (.sum [.result 20945 .summary, .result 20481 .summary])

def exact20950RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20950RawTermsValid :
    exact20950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20950 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26831⟩⟩) exact20950RawTerms .large 20948 (.finite 14223885201645539505274355764) (some (20949))

def event20951 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27048⟩⟩) 0 ⟨26831⟩ 20950

def event20952 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27048⟩⟩) 1 ⟨27047⟩ 20269

def event20953 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27048⟩⟩) (.sum [.predecessor 0 20951 .coefficient, .predecessor 1 20952 .coefficient])

def event20954 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27048⟩⟩) (.sum [.result 20950 .summary, .result 20269 .summary])

def exact20955RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20955RawTermsValid :
    exact20955RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20955 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27048⟩⟩) exact20955RawTerms .large 20953 (.finite 18965303649908456346701791284) (some (20954))

def event20956 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27265⟩⟩) 0 ⟨27048⟩ 20955

def event20957 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27265⟩⟩) 1 ⟨27264⟩ 20057

def event20958 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27265⟩⟩) (.sum [.predecessor 0 20956 .coefficient, .predecessor 1 20957 .coefficient])

def event20959 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27265⟩⟩) (.sum [.result 20955 .summary, .result 20057 .summary])

def exact20960RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20960RawTermsValid :
    exact20960RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20960 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27265⟩⟩) exact20960RawTerms .large 20958 (.finite 23706886606235022529910538292) (some (20959))

def event20961 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27482⟩⟩) 0 ⟨27265⟩ 20960

def event20962 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27482⟩⟩) 1 ⟨27481⟩ 19845

def event20963 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27482⟩⟩) (.sum [.predecessor 0 20961 .coefficient, .predecessor 1 20962 .coefficient])

def event20964 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27482⟩⟩) (.sum [.result 20960 .summary, .result 19845 .summary])

def exact20965RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20965RawTermsValid :
    exact20965RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20965 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27482⟩⟩) exact20965RawTerms .large 20963 (.finite 28448551816593413384009941044) (some (20964))

def event20966 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27699⟩⟩) 0 ⟨27482⟩ 20965

def event20967 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27699⟩⟩) 1 ⟨27698⟩ 19633

def event20968 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27699⟩⟩) (.sum [.predecessor 0 20966 .coefficient, .predecessor 1 20967 .coefficient])

def event20969 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27699⟩⟩) (.sum [.result 20965 .summary, .result 19633 .summary])

def exact20970RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20970RawTermsValid :
    exact20970RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20970 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27699⟩⟩) exact20970RawTerms .large 20968 (.finite 33190381535015453579890655284) (some (20969))

def event20971 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27916⟩⟩) 0 ⟨27699⟩ 20970

def event20972 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27916⟩⟩) 1 ⟨27915⟩ 19421

def event20973 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27916⟩⟩) (.sum [.predecessor 0 20971 .coefficient, .predecessor 1 20972 .coefficient])

def event20974 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27916⟩⟩) (.sum [.result 20970 .summary, .result 19421 .summary])

def exact20975RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20975RawTermsValid :
    exact20975RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20975 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27916⟩⟩) exact20975RawTerms .large 20973 (.finite 37932293507469318446662025268) (some (20974))

def event20976 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28133⟩⟩) 0 ⟨27916⟩ 20975

def event20977 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28133⟩⟩) 1 ⟨28132⟩ 19209

def event20978 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28133⟩⟩) (.sum [.predecessor 0 20976 .coefficient, .predecessor 1 20977 .coefficient])

def event20979 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28133⟩⟩) (.sum [.result 20975 .summary, .result 19209 .summary])

def exact20980RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20980RawTermsValid :
    exact20980RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20980 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28133⟩⟩) exact20980RawTerms .large 20978 (.finite 42674369987986832655214706740) (some (20979))

def event20981 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28350⟩⟩) 0 ⟨28133⟩ 20980

def event20982 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28350⟩⟩) 1 ⟨28349⟩ 18997

def event20983 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28350⟩⟩) (.sum [.predecessor 0 20981 .coefficient, .predecessor 1 20982 .coefficient])

def event20984 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28350⟩⟩) (.sum [.result 20980 .summary, .result 18997 .summary])

def exact20985RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20985RawTermsValid :
    exact20985RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20985 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28350⟩⟩) exact20985RawTerms .large 20983 (.finite 47416693230599820876439355444) (some (20984))

def event20986 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28567⟩⟩) 0 ⟨28350⟩ 20985

def event20987 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28567⟩⟩) 1 ⟨28566⟩ 18785

def event20988 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28567⟩⟩) (.sum [.predecessor 0 20986 .coefficient, .predecessor 1 20987 .coefficient])

def event20989 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28567⟩⟩) (.sum [.result 20985 .summary, .result 18785 .summary])

def exact20990RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6718⟩⟩, ⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6720⟩⟩, ⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6722⟩⟩, ⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6724⟩⟩, ⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6726⟩⟩, ⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18063⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17181⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6391⟩⟩, ⟨.program ⟨214⟩, ⟨17237⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6398⟩⟩, ⟨.program ⟨214⟩, ⟨17454⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17846⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6427⟩⟩, ⟨.program ⟨214⟩, ⟨15536⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6452⟩⟩, ⟨.program ⟨214⟩, ⟨15228⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15067⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17678⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact20990RawTermsValid :
    exact20990RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event20990 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28567⟩⟩) exact20990RawTerms .large 20988 (.finite 52159098727244633768554659892) (some (20989))

def event20991 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28784⟩⟩) 0 ⟨28567⟩ 20990

def eventLeaf1296 : Array AnnotatedEvent := #[
  { event := event20736
    frameStart := 20724 },
  { event := event20737
    frameStart := 20724 },
  { event := event20738
    frameStart := 20724 },
  { event := event20739
    frameStart := 20724 },
  { event := event20740
    frameStart := 20724 },
  { event := event20741
    frameStart := 20724 },
  { event := event20742
    frameStart := 20724 },
  { event := event20743
    frameStart := 20724 },
  { event := event20744
    frameStart := 20724 },
  { event := event20745
    frameStart := 20724 },
  { event := event20746
    frameStart := 20724 },
  { event := event20747
    frameStart := 20724 },
  { event := event20748
    frameStart := 20724 },
  { event := event20749
    frameStart := 20724 },
  { event := event20750
    frameStart := 20724 },
  { event := event20751
    frameStart := 20724 }
]

def eventLeaf1297 : Array AnnotatedEvent := #[
  { event := event20752
    frameStart := 20724 },
  { event := event20753
    frameStart := 20724 },
  { event := event20754
    frameStart := 20724 },
  { event := event20755
    frameStart := 20724 },
  { event := event20756
    frameStart := 20724 },
  { event := event20757
    frameStart := 20724 },
  { event := event20758
    frameStart := 20724 },
  { event := event20759
    frameStart := 20724 },
  { event := event20760
    frameStart := 20724 },
  { event := event20761
    frameStart := 20724 },
  { event := event20762
    frameStart := 20724 },
  { event := event20763
    frameStart := 20724 },
  { event := event20764
    frameStart := 20724 },
  { event := event20765
    frameStart := 20724 },
  { event := event20766
    frameStart := 20724 },
  { event := event20767
    frameStart := 20724 }
]

def eventLeaf1298 : Array AnnotatedEvent := #[
  { event := event20768
    frameStart := 20724 },
  { event := event20769
    frameStart := 20724 },
  { event := event20770
    frameStart := 20724 },
  { event := event20771
    frameStart := 20724 },
  { event := event20772
    frameStart := 20724 },
  { event := event20773
    frameStart := 20724 },
  { event := event20774
    frameStart := 20724 },
  { event := event20775
    frameStart := 20724 },
  { event := event20776
    frameStart := 20724 },
  { event := event20777
    frameStart := 20724 },
  { event := event20778
    frameStart := 20778 },
  { event := event20779
    frameStart := 20778 },
  { event := event20780
    frameStart := 20778 },
  { event := event20781
    frameStart := 20778 },
  { event := event20782
    frameStart := 20778 },
  { event := event20783
    frameStart := 20778 }
]

def eventLeaf1299 : Array AnnotatedEvent := #[
  { event := event20784
    frameStart := 20778 },
  { event := event20785
    frameStart := 20778 },
  { event := event20786
    frameStart := 20778 },
  { event := event20787
    frameStart := 20778 },
  { event := event20788
    frameStart := 20778 },
  { event := event20789
    frameStart := 20778 },
  { event := event20790
    frameStart := 20778 },
  { event := event20791
    frameStart := 20778 },
  { event := event20792
    frameStart := 20778 },
  { event := event20793
    frameStart := 20778 },
  { event := event20794
    frameStart := 20778 },
  { event := event20795
    frameStart := 20778 },
  { event := event20796
    frameStart := 20778 },
  { event := event20797
    frameStart := 20778 },
  { event := event20798
    frameStart := 20778 },
  { event := event20799
    frameStart := 20778 }
]

def eventLeaf1300 : Array AnnotatedEvent := #[
  { event := event20800
    frameStart := 20778 },
  { event := event20801
    frameStart := 20778 },
  { event := event20802
    frameStart := 20778 },
  { event := event20803
    frameStart := 20778 },
  { event := event20804
    frameStart := 20778 },
  { event := event20805
    frameStart := 20778 },
  { event := event20806
    frameStart := 20778 },
  { event := event20807
    frameStart := 20778 },
  { event := event20808
    frameStart := 20778 },
  { event := event20809
    frameStart := 20778 },
  { event := event20810
    frameStart := 20778 },
  { event := event20811
    frameStart := 20778 },
  { event := event20812
    frameStart := 20778 },
  { event := event20813
    frameStart := 20778 },
  { event := event20814
    frameStart := 20778 },
  { event := event20815
    frameStart := 20778 }
]

def eventLeaf1301 : Array AnnotatedEvent := #[
  { event := event20816
    frameStart := 20778 },
  { event := event20817
    frameStart := 20778 },
  { event := event20818
    frameStart := 20778 },
  { event := event20819
    frameStart := 20778 },
  { event := event20820
    frameStart := 20778 },
  { event := event20821
    frameStart := 20778 },
  { event := event20822
    frameStart := 20778 },
  { event := event20823
    frameStart := 20778 },
  { event := event20824
    frameStart := 20778 },
  { event := event20825
    frameStart := 20778 },
  { event := event20826
    frameStart := 20778 },
  { event := event20827
    frameStart := 20778 },
  { event := event20828
    frameStart := 20778 },
  { event := event20829
    frameStart := 20778 },
  { event := event20830
    frameStart := 20778 },
  { event := event20831
    frameStart := 20778 }
]

def eventLeaf1302 : Array AnnotatedEvent := #[
  { event := event20832
    frameStart := 20778 },
  { event := event20833
    frameStart := 20778 },
  { event := event20834
    frameStart := 20778 },
  { event := event20835
    frameStart := 20778 },
  { event := event20836
    frameStart := 20778 },
  { event := event20837
    frameStart := 20778 },
  { event := event20838
    frameStart := 20778 },
  { event := event20839
    frameStart := 20778 },
  { event := event20840
    frameStart := 20778 },
  { event := event20841
    frameStart := 20778 },
  { event := event20842
    frameStart := 20778 },
  { event := event20843
    frameStart := 20778 },
  { event := event20844
    frameStart := 20778 },
  { event := event20845
    frameStart := 20778 },
  { event := event20846
    frameStart := 20778 },
  { event := event20847
    frameStart := 20778 }
]

def eventLeaf1303 : Array AnnotatedEvent := #[
  { event := event20848
    frameStart := 20778 },
  { event := event20849
    frameStart := 20778 },
  { event := event20850
    frameStart := 20778 },
  { event := event20851
    frameStart := 20778 },
  { event := event20852
    frameStart := 20778 },
  { event := event20853
    frameStart := 20778 },
  { event := event20854
    frameStart := 20778 },
  { event := event20855
    frameStart := 20778 },
  { event := event20856
    frameStart := 20778 },
  { event := event20857
    frameStart := 20778 },
  { event := event20858
    frameStart := 20778 },
  { event := event20859
    frameStart := 20778 },
  { event := event20860
    frameStart := 20778 },
  { event := event20861
    frameStart := 20778 },
  { event := event20862
    frameStart := 20778 },
  { event := event20863
    frameStart := 20778 }
]

def eventLeaf1304 : Array AnnotatedEvent := #[
  { event := event20864
    frameStart := 20778 },
  { event := event20865
    frameStart := 20778 },
  { event := event20866
    frameStart := 20778 },
  { event := event20867
    frameStart := 20778 },
  { event := event20868
    frameStart := 20778 },
  { event := event20869
    frameStart := 20778 },
  { event := event20870
    frameStart := 20778 },
  { event := event20871
    frameStart := 20778 },
  { event := event20872
    frameStart := 20778 },
  { event := event20873
    frameStart := 20778 },
  { event := event20874
    frameStart := 20778 },
  { event := event20875
    frameStart := 20778 },
  { event := event20876
    frameStart := 20778 },
  { event := event20877
    frameStart := 20778 },
  { event := event20878
    frameStart := 20778 },
  { event := event20879
    frameStart := 20778 }
]

def eventLeaf1305 : Array AnnotatedEvent := #[
  { event := event20880
    frameStart := 20778 },
  { event := event20881
    frameStart := 20778 },
  { event := event20882
    frameStart := 0 },
  { event := event20883
    frameStart := 0 },
  { event := event20884
    frameStart := 0 },
  { event := event20885
    frameStart := 0 },
  { event := event20886
    frameStart := 0 },
  { event := event20887
    frameStart := 0 },
  { event := event20888
    frameStart := 0 },
  { event := event20889
    frameStart := 0 },
  { event := event20890
    frameStart := 0 },
  { event := event20891
    frameStart := 0 },
  { event := event20892
    frameStart := 0 },
  { event := event20893
    frameStart := 0 },
  { event := event20894
    frameStart := 0 },
  { event := event20895
    frameStart := 0 }
]

def eventLeaf1306 : Array AnnotatedEvent := #[
  { event := event20896
    frameStart := 0 },
  { event := event20897
    frameStart := 0 },
  { event := event20898
    frameStart := 0 },
  { event := event20899
    frameStart := 0 },
  { event := event20900
    frameStart := 0 },
  { event := event20901
    frameStart := 0 },
  { event := event20902
    frameStart := 0 },
  { event := event20903
    frameStart := 0 },
  { event := event20904
    frameStart := 0 },
  { event := event20905
    frameStart := 0 },
  { event := event20906
    frameStart := 0 },
  { event := event20907
    frameStart := 0 },
  { event := event20908
    frameStart := 0 },
  { event := event20909
    frameStart := 0 },
  { event := event20910
    frameStart := 0 },
  { event := event20911
    frameStart := 0 }
]

def eventLeaf1307 : Array AnnotatedEvent := #[
  { event := event20912
    frameStart := 0 },
  { event := event20913
    frameStart := 0 },
  { event := event20914
    frameStart := 0 },
  { event := event20915
    frameStart := 0 },
  { event := event20916
    frameStart := 0 },
  { event := event20917
    frameStart := 0 },
  { event := event20918
    frameStart := 0 },
  { event := event20919
    frameStart := 0 },
  { event := event20920
    frameStart := 0 },
  { event := event20921
    frameStart := 0 },
  { event := event20922
    frameStart := 0 },
  { event := event20923
    frameStart := 0 },
  { event := event20924
    frameStart := 0 },
  { event := event20925
    frameStart := 0 },
  { event := event20926
    frameStart := 0 },
  { event := event20927
    frameStart := 0 }
]

def eventLeaf1308 : Array AnnotatedEvent := #[
  { event := event20928
    frameStart := 0 },
  { event := event20929
    frameStart := 0 },
  { event := event20930
    frameStart := 0 },
  { event := event20931
    frameStart := 0 },
  { event := event20932
    frameStart := 0 },
  { event := event20933
    frameStart := 0 },
  { event := event20934
    frameStart := 0 },
  { event := event20935
    frameStart := 0 },
  { event := event20936
    frameStart := 0 },
  { event := event20937
    frameStart := 0 },
  { event := event20938
    frameStart := 0 },
  { event := event20939
    frameStart := 0 },
  { event := event20940
    frameStart := 0 },
  { event := event20941
    frameStart := 0 },
  { event := event20942
    frameStart := 0 },
  { event := event20943
    frameStart := 0 }
]

def eventLeaf1309 : Array AnnotatedEvent := #[
  { event := event20944
    frameStart := 0 },
  { event := event20945
    frameStart := 0 },
  { event := event20946
    frameStart := 0 },
  { event := event20947
    frameStart := 0 },
  { event := event20948
    frameStart := 0 },
  { event := event20949
    frameStart := 0 },
  { event := event20950
    frameStart := 0 },
  { event := event20951
    frameStart := 0 },
  { event := event20952
    frameStart := 0 },
  { event := event20953
    frameStart := 0 },
  { event := event20954
    frameStart := 0 },
  { event := event20955
    frameStart := 0 },
  { event := event20956
    frameStart := 0 },
  { event := event20957
    frameStart := 0 },
  { event := event20958
    frameStart := 0 },
  { event := event20959
    frameStart := 0 }
]

def eventLeaf1310 : Array AnnotatedEvent := #[
  { event := event20960
    frameStart := 0 },
  { event := event20961
    frameStart := 0 },
  { event := event20962
    frameStart := 0 },
  { event := event20963
    frameStart := 0 },
  { event := event20964
    frameStart := 0 },
  { event := event20965
    frameStart := 0 },
  { event := event20966
    frameStart := 0 },
  { event := event20967
    frameStart := 0 },
  { event := event20968
    frameStart := 0 },
  { event := event20969
    frameStart := 0 },
  { event := event20970
    frameStart := 0 },
  { event := event20971
    frameStart := 0 },
  { event := event20972
    frameStart := 0 },
  { event := event20973
    frameStart := 0 },
  { event := event20974
    frameStart := 0 },
  { event := event20975
    frameStart := 0 }
]

def eventLeaf1311 : Array AnnotatedEvent := #[
  { event := event20976
    frameStart := 0 },
  { event := event20977
    frameStart := 0 },
  { event := event20978
    frameStart := 0 },
  { event := event20979
    frameStart := 0 },
  { event := event20980
    frameStart := 0 },
  { event := event20981
    frameStart := 0 },
  { event := event20982
    frameStart := 0 },
  { event := event20983
    frameStart := 0 },
  { event := event20984
    frameStart := 0 },
  { event := event20985
    frameStart := 0 },
  { event := event20986
    frameStart := 0 },
  { event := event20987
    frameStart := 0 },
  { event := event20988
    frameStart := 0 },
  { event := event20989
    frameStart := 0 },
  { event := event20990
    frameStart := 0 },
  { event := event20991
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events081
