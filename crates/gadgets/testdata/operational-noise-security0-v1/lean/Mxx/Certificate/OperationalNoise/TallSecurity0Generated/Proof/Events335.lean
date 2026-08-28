import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events335

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event85760 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25913⟩⟩) (.product (.result 85755 .summary) (.transfer 85759) (⟨false, false, none, none, none⟩))

def event85761 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25913⟩⟩, .operator (⟨85755, 1⟩, ⟨85691, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩, (-1)⟩)

def event85762 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25913⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25912⟩⟩) ⟨23500⟩ 85688)

def event85763 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25913⟩⟩, .relation 85762 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨23500⟩⟩]⟩, (-1)⟩)

def event85764 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25913⟩⟩, .operator (⟨85755, 0⟩, ⟨85691, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩, (1)⟩)

def exact85765RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨23500⟩⟩]⟩, (-1)⟩]

theorem exact85765RawTermsValid :
    exact85765RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85765 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25913⟩⟩) exact85765RawTerms .large 85758 (.finite 350231094886400) (some (85760))

def event85766 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19384⟩⟩) 0 ⟨13775⟩ 4115

def event85767 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19384⟩⟩) (.authority (.relationPreimageSource ⟨13⟩))

def exact85768RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩, (1)⟩]

theorem exact85768RawTermsValid :
    exact85768RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85768 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19384⟩⟩) exact85768RawTerms (.finite 136065468) 85767 .exactZero (none)

def event85769 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19386⟩⟩) 0 ⟨19384⟩ 85768

def event85770 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19386⟩⟩) 1 ⟨2348⟩ 4

def event85771 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19386⟩⟩) (.scale (.predecessor 0 85769 .coefficient) (.value (.predecessor 1 85770 .coefficient)))

def exact85772RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩, (1)⟩]

theorem exact85772RawTermsValid :
    exact85772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85772 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19386⟩⟩) exact85772RawTerms (.finite 136065468) 85771 .exactZero (none)

def event85773 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19387⟩⟩) 0 ⟨5541⟩ 80012

def event85774 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19387⟩⟩) 1 ⟨19386⟩ 85772

def event85775 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19387⟩⟩) (.product (.predecessor 0 85773 .coefficient) (.predecessor 1 85774 .coefficient) (⟨false, false, none, none, none⟩))

def event85776 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19387⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩) [⟨.result 85768 .coefficient, false, none⟩])

def event85777 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19387⟩⟩) (.product (.result 80012 .summary) (.transfer 85776) (⟨false, false, none, none, none⟩))

def event85778 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19387⟩⟩, .operator (⟨80012, 0⟩, ⟨85772, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩, (1)⟩)

def event85779 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19385⟩⟩)

def event85780 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event85781 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event85782 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event85783 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event85784 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event85785 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event85786 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event85787 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event85788 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 85787

def event85789 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 85785

def event85790 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 85788 .coefficient) (.value (.predecessor 1 85789 .coefficient)))

def event85791 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event85792 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 85791

def event85793 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 85783

def event85794 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 85792 .coefficient, .predecessor 1 85793 .coefficient])

def event85795 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event85796 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 85795

def event85797 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 85781

def event85798 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 85797 .coefficient))

def event85799 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event85800 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11301⟩⟩) 0 ⟨5536⟩ 85799

def event85801 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11301⟩⟩) (.authority (.programFamilyFact))

def exact85802RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩], []⟩, (1)⟩]

theorem exact85802RawTermsValid :
    exact85802RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85802 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11301⟩⟩) exact85802RawTerms (.finite 12) 85801 .exactZero (none)

def event85803 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13773⟩⟩) 0 ⟨5536⟩ 85799

def event85804 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13773⟩⟩) (.authority (.programFamilyFact))

def exact85805RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13773⟩⟩], []⟩, (1)⟩]

theorem exact85805RawTermsValid :
    exact85805RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85805 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13773⟩⟩) exact85805RawTerms (.finite 12) 85804 .exactZero (none)

def event85806 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13774⟩⟩) 0 ⟨13773⟩ 85805

def event85807 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13774⟩⟩) 1 ⟨11301⟩ 85802

def event85808 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13774⟩⟩) (.product (.predecessor 0 85806 .coefficient) (.predecessor 1 85807 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event85809 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13774⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], []⟩) [⟨.result 85805 .coefficient, true, some 1⟩, ⟨.result 85802 .coefficient, true, some 1⟩])

def event85810 : Event := .survivorFold (1) 85809

def exact85811RawTerms : List Term := []

theorem exact85811RawTermsValid :
    exact85811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85811 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13774⟩⟩) exact85811RawTerms (.finite 144) 85808 (.finite 144) (some (85809))

def event85812 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13775⟩⟩) 0 ⟨13774⟩ 85811

def event85813 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13775⟩⟩) (.identity (.predecessor 0 85812 .coefficient))

def event85814 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13775⟩⟩) (.finite 144)

def event85815 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19384⟩⟩) 0 ⟨13775⟩ 85814

def event85816 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19384⟩⟩) (.authority (.relationPreimageSource ⟨13⟩))

def exact85817RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩, (1)⟩]

theorem exact85817RawTermsValid :
    exact85817RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85817 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19384⟩⟩) exact85817RawTerms (.finite 136065468) 85816 .exactZero (none)

def event85818 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact85819RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact85819RawTermsValid :
    exact85819RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85819 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact85819RawTerms .large 85818 .exactZero (none)

def event85820 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19385⟩⟩) 0 ⟨6⟩ 85819

def event85821 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19385⟩⟩) 1 ⟨19384⟩ 85817

def event85822 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19385⟩⟩) (.product (.predecessor 0 85820 .coefficient) (.predecessor 1 85821 .coefficient) (⟨false, false, none, none, none⟩))

def event85823 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19385⟩⟩, .operator (⟨85819, 0⟩, ⟨85817, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩, (1)⟩)

def exact85824RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩, (1)⟩]

theorem exact85824RawTermsValid :
    exact85824RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85824 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19385⟩⟩) exact85824RawTerms .large 85822 .exactZero (none)

def event85825 : Event := .preFoldPolynomial 85824 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩, (1)⟩] .exactZero none

def exact85826RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩, (1)⟩]

def event85826 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19385⟩⟩) 85825 exact85826RawTerms .large 85822 .exactZero (none)

def event85827 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25916⟩⟩)

def event85828 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event85829 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event85830 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event85831 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event85832 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event85833 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event85834 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event85835 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event85836 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 85835

def event85837 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 85833

def event85838 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 85836 .coefficient) (.value (.predecessor 1 85837 .coefficient)))

def event85839 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event85840 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 85839

def event85841 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 85831

def event85842 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 85840 .coefficient, .predecessor 1 85841 .coefficient])

def event85843 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event85844 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 85843

def event85845 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 85829

def event85846 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 85845 .coefficient))

def event85847 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event85848 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11301⟩⟩) 0 ⟨5536⟩ 85847

def event85849 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11301⟩⟩) (.authority (.programFamilyFact))

def exact85850RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩], []⟩, (1)⟩]

theorem exact85850RawTermsValid :
    exact85850RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85850 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11301⟩⟩) exact85850RawTerms (.finite 12) 85849 .exactZero (none)

def event85851 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13773⟩⟩) 0 ⟨5536⟩ 85847

def event85852 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13773⟩⟩) (.authority (.programFamilyFact))

def exact85853RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13773⟩⟩], []⟩, (1)⟩]

theorem exact85853RawTermsValid :
    exact85853RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85853 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13773⟩⟩) exact85853RawTerms (.finite 12) 85852 .exactZero (none)

def event85854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13774⟩⟩) 0 ⟨13773⟩ 85853

def event85855 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13774⟩⟩) 1 ⟨11301⟩ 85850

def event85856 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13774⟩⟩) (.product (.predecessor 0 85854 .coefficient) (.predecessor 1 85855 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event85857 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13774⟩⟩, .operator (⟨85853, 0⟩, ⟨85850, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], []⟩, (1)⟩)

def exact85858RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], []⟩, (1)⟩]

theorem exact85858RawTermsValid :
    exact85858RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85858 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13774⟩⟩) exact85858RawTerms (.finite 144) 85856 .exactZero (none)

def event85859 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13775⟩⟩) 0 ⟨13774⟩ 85858

def event85860 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13775⟩⟩) (.identity (.predecessor 0 85859 .coefficient))

def event85861 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13775⟩⟩) (.finite 144)

def event85862 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23499⟩⟩) 0 ⟨13775⟩ 85861

def event85863 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23499⟩⟩) (.authority (.programFamilyFact))

def event85864 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23499⟩⟩) (.finite 3720)

def event85865 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event85866 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23500⟩⟩) 0 ⟨6689⟩ 85865

def event85867 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23500⟩⟩) 1 ⟨23499⟩ 85864

def event85868 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23500⟩⟩) (.authority (.operator))

def exact85869RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23500⟩⟩]⟩, (1)⟩]

theorem exact85869RawTermsValid :
    exact85869RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85869 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23500⟩⟩) exact85869RawTerms .large 85868 .exactZero (none)

def event85870 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25912⟩⟩) 0 ⟨23500⟩ 85869

def event85871 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25912⟩⟩) (.authority (.operator))

def exact85872RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩, (1)⟩]

theorem exact85872RawTermsValid :
    exact85872RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85872 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25912⟩⟩) exact85872RawTerms (.finite 8192) 85871 .exactZero (none)

def event85873 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event85874 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event85875 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13880⟩⟩) 0 ⟨13775⟩ 85861

def event85876 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13880⟩⟩) 1 ⟨110⟩ 85874

def event85877 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13880⟩⟩) (.sum [.predecessor 0 85875 .coefficient, .predecessor 1 85876 .coefficient])

def event85878 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13880⟩⟩) (.finite 144)

def event85879 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13881⟩⟩) 0 ⟨13880⟩ 85878

def event85880 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13881⟩⟩) (.identity (.predecessor 0 85879 .coefficient))

def exact85881RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], []⟩, (1)⟩]

theorem exact85881RawTermsValid :
    exact85881RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85881 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13881⟩⟩) exact85881RawTerms (.finite 144) 85880 .exactZero (none)

def event85882 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact85883RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact85883RawTermsValid :
    exact85883RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85883 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact85883RawTerms .large 85882 .exactZero (none)

def event85884 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13882⟩⟩) 0 ⟨6544⟩ 85883

def event85885 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13882⟩⟩) 1 ⟨13881⟩ 85881

def event85886 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13882⟩⟩) (.product (.predecessor 0 85884 .coefficient) (.predecessor 1 85885 .coefficient) (⟨false, false, none, none, none⟩))

def event85887 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13882⟩⟩, .operator (⟨85883, 0⟩, ⟨85881, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact85888RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact85888RawTermsValid :
    exact85888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85888 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13882⟩⟩) exact85888RawTerms .large 85886 .exactZero (none)

def event85889 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 85865

def event85890 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact85891RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact85891RawTermsValid :
    exact85891RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85891 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact85891RawTerms .large 85890 .exactZero (none)

def event85892 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6777⟩⟩) 0 ⟨6757⟩ 85891

def event85893 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6777⟩⟩) (.identity (.predecessor 0 85892 .coefficient))

def exact85894RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩]

theorem exact85894RawTermsValid :
    exact85894RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85894 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6777⟩⟩) exact85894RawTerms .large 85893 .exactZero (none)

def event85895 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7846⟩⟩) 0 ⟨6777⟩ 85894

def event85896 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7846⟩⟩) (.authority (.operator))

def exact85897RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (1)⟩]

theorem exact85897RawTermsValid :
    exact85897RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85897 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7846⟩⟩) exact85897RawTerms (.finite 8192) 85896 .exactZero (none)

def event85898 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7847⟩⟩) 0 ⟨7846⟩ 85897

def event85899 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7847⟩⟩) 1 ⟨2348⟩ 85831

def event85900 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7847⟩⟩) (.scale (.predecessor 0 85898 .coefficient) (.value (.predecessor 1 85899 .coefficient)))

def exact85901RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (1)⟩]

theorem exact85901RawTermsValid :
    exact85901RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85901 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7847⟩⟩) exact85901RawTerms (.finite 8192) 85900 .exactZero (none)

def event85902 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6794⟩⟩) 0 ⟨6757⟩ 85891

def event85903 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6794⟩⟩) (.identity (.predecessor 0 85902 .coefficient))

def exact85904RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩]⟩, (1)⟩]

theorem exact85904RawTermsValid :
    exact85904RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85904 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6794⟩⟩) exact85904RawTerms .large 85903 .exactZero (none)

def event85905 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7848⟩⟩) 0 ⟨6794⟩ 85904

def event85906 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7848⟩⟩) 1 ⟨7847⟩ 85901

def event85907 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7848⟩⟩) (.product (.predecessor 0 85905 .coefficient) (.predecessor 1 85906 .coefficient) (⟨false, false, none, none, none⟩))

def event85908 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7848⟩⟩, .operator (⟨85904, 0⟩, ⟨85901, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (1)⟩)

def exact85909RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (1)⟩]

theorem exact85909RawTermsValid :
    exact85909RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85909 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7848⟩⟩) exact85909RawTerms .large 85907 .exactZero (none)

def event85910 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13883⟩⟩) 0 ⟨7848⟩ 85909

def event85911 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13883⟩⟩) 1 ⟨13882⟩ 85888

def event85912 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13883⟩⟩) (.sum [.predecessor 0 85910 .coefficient, .predecessor 1 85911 .coefficient])

def exact85913RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact85913RawTermsValid :
    exact85913RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85913 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13883⟩⟩) exact85913RawTerms .large 85912 .exactZero (none)

def event85914 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25915⟩⟩) 0 ⟨13883⟩ 85913

def event85915 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25915⟩⟩) 1 ⟨25912⟩ 85872

def event85916 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25915⟩⟩) (.product (.predecessor 0 85914 .coefficient) (.predecessor 1 85915 .coefficient) (⟨false, false, none, none, none⟩))

def event85917 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25915⟩⟩, .operator (⟨85913, 0⟩, ⟨85872, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩, (1)⟩)

def event85918 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25915⟩⟩, .operator (⟨85913, 1⟩, ⟨85872, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩, (-1)⟩)

def event85919 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25915⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25912⟩⟩) ⟨23500⟩ 85869)

def event85920 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25915⟩⟩, .relation 85919 0, ⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨23500⟩⟩]⟩, (-1)⟩)

def exact85921RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨23500⟩⟩]⟩, (-1)⟩]

theorem exact85921RawTermsValid :
    exact85921RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85921 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25915⟩⟩) exact85921RawTerms .large 85916 .exactZero (none)

def event85922 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15702⟩⟩) 0 ⟨13775⟩ 85861

def event85923 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15702⟩⟩) (.authority (.programFamilyFact))

def exact85924RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], []⟩, (1)⟩]

theorem exact85924RawTermsValid :
    exact85924RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85924 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15702⟩⟩) exact85924RawTerms (.finite 12) 85923 .exactZero (none)

def event85925 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15704⟩⟩) 0 ⟨6544⟩ 85883

def event85926 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15704⟩⟩) 1 ⟨15702⟩ 85924

def event85927 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15704⟩⟩) (.product (.predecessor 0 85925 .coefficient) (.predecessor 1 85926 .coefficient) (⟨false, true, none, none, some 1⟩))

def event85928 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15704⟩⟩, .operator (⟨85883, 0⟩, ⟨85924, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact85929RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact85929RawTermsValid :
    exact85929RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85929 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15704⟩⟩) exact85929RawTerms .large 85927 .exactZero (none)

def event85930 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6695⟩⟩) 0 ⟨6689⟩ 85865

def event85931 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6695⟩⟩) (.authority (.operator))

def exact85932RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩]

theorem exact85932RawTermsValid :
    exact85932RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85932 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6695⟩⟩) exact85932RawTerms .large 85931 .exactZero (none)

def event85933 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15705⟩⟩) 0 ⟨6695⟩ 85932

def event85934 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15705⟩⟩) 1 ⟨15704⟩ 85929

def event85935 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15705⟩⟩) (.sum [.predecessor 0 85933 .coefficient, .predecessor 1 85934 .coefficient])

def exact85936RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact85936RawTermsValid :
    exact85936RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85936 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15705⟩⟩) exact85936RawTerms .large 85935 .exactZero (none)

def event85937 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25916⟩⟩) 0 ⟨15705⟩ 85936

def event85938 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25916⟩⟩) 1 ⟨25915⟩ 85921

def event85939 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25916⟩⟩) (.sum [.predecessor 0 85937 .coefficient, .predecessor 1 85938 .coefficient])

def exact85940RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨23500⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact85940RawTermsValid :
    exact85940RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85940 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25916⟩⟩) exact85940RawTerms .large 85939 .exactZero (none)

def event85941 : Event := .preFoldPolynomial 85940 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨23500⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact85942RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨23500⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event85942 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25916⟩⟩) 85941 exact85942RawTerms .large 85939 .exactZero (none)

def event85943 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨13775⟩⟩) ⟨⟨108⟩, ⟨13⟩, ⟨109⟩⟩ ⟨85779, 85943⟩

def event85944 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19387⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩) (1) 0 2 (.universal 85943 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩) (none) 85942)

def event85945 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19387⟩⟩, .relation 85944 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩)

def event85946 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19387⟩⟩, .relation 85944 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩, (-1)⟩)

def event85947 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19387⟩⟩, .relation 85944 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨23500⟩⟩]⟩, (1)⟩)

def event85948 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19387⟩⟩, .relation 85944 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact85949RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨23500⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact85949RawTermsValid :
    exact85949RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85949 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19387⟩⟩) exact85949RawTerms .large 85775 (.finite 1811303510016) (some (85777))

def event85950 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25914⟩⟩) 0 ⟨19387⟩ 85949

def event85951 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25914⟩⟩) 1 ⟨25913⟩ 85765

def event85952 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25914⟩⟩) (.sum [.predecessor 0 85950 .coefficient, .predecessor 1 85951 .coefficient])

def event85953 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25914⟩⟩, .operator (⟨85949, 2⟩, ⟨85765, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨23500⟩⟩]⟩, (-1)⟩)

def event85954 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25914⟩⟩, .operator (⟨85949, 1⟩, ⟨85765, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩, (1)⟩)

def event85955 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25914⟩⟩) (.sum [.result 85949 .summary, .result 85765 .summary])

def exact85956RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact85956RawTermsValid :
    exact85956RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85956 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25914⟩⟩) exact85956RawTerms .large 85952 (.finite 352042398396416) (some (85955))

def event85957 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27434⟩⟩) 0 ⟨25914⟩ 85956

def event85958 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27434⟩⟩) 1 ⟨27432⟩ 85681

def event85959 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27434⟩⟩) (.product (.predecessor 0 85957 .coefficient) (.predecessor 1 85958 .coefficient) (⟨false, false, none, none, none⟩))

def event85960 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27434⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩) [⟨.result 85681 .coefficient, false, none⟩])

def event85961 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27434⟩⟩) (.product (.result 85956 .summary) (.transfer 85960) (⟨false, false, none, none, none⟩))

def event85962 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27434⟩⟩, .operator (⟨85956, 0⟩, ⟨85681, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩, (1)⟩)

def event85963 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27434⟩⟩, .operator (⟨85956, 1⟩, ⟨85681, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩, (-1)⟩)

def event85964 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27434⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27432⟩⟩) ⟨24036⟩ 85678)

def event85965 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27434⟩⟩, .relation 85964 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨24036⟩⟩]⟩, (-1)⟩)

def exact85966RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨24036⟩⟩]⟩, (-1)⟩]

theorem exact85966RawTermsValid :
    exact85966RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85966 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27434⟩⟩) exact85966RawTerms .large 85959 (.finite 1292001234793221062656) (some (85961))

def event85967 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21112⟩⟩) 0 ⟨15703⟩ 4121

def event85968 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21112⟩⟩) (.authority (.relationPreimageSource ⟨39⟩))

def exact85969RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21112⟩⟩]⟩, (1)⟩]

theorem exact85969RawTermsValid :
    exact85969RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85969 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21112⟩⟩) exact85969RawTerms (.finite 136065468) 85968 .exactZero (none)

def event85970 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21114⟩⟩) 0 ⟨21112⟩ 85969

def event85971 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21114⟩⟩) 1 ⟨2348⟩ 4

def event85972 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21114⟩⟩) (.scale (.predecessor 0 85970 .coefficient) (.value (.predecessor 1 85971 .coefficient)))

def exact85973RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21112⟩⟩]⟩, (1)⟩]

theorem exact85973RawTermsValid :
    exact85973RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event85973 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21114⟩⟩) exact85973RawTerms (.finite 136065468) 85972 .exactZero (none)

def event85974 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21115⟩⟩) 0 ⟨5541⟩ 80012

def event85975 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21115⟩⟩) 1 ⟨21114⟩ 85973

def event85976 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21115⟩⟩) (.product (.predecessor 0 85974 .coefficient) (.predecessor 1 85975 .coefficient) (⟨false, false, none, none, none⟩))

def event85977 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21115⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21112⟩⟩]⟩) [⟨.result 85969 .coefficient, false, none⟩])

def event85978 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21115⟩⟩) (.product (.result 80012 .summary) (.transfer 85977) (⟨false, false, none, none, none⟩))

def event85979 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21115⟩⟩, .operator (⟨80012, 0⟩, ⟨85973, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21112⟩⟩]⟩, (1)⟩)

def event85980 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21113⟩⟩)

def event85981 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event85982 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event85983 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event85984 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event85985 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event85986 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event85987 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event85988 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event85989 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 85988

def event85990 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 85986

def event85991 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 85989 .coefficient) (.value (.predecessor 1 85990 .coefficient)))

def event85992 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event85993 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 85992

def event85994 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 85984

def event85995 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 85993 .coefficient, .predecessor 1 85994 .coefficient])

def event85996 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event85997 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 85996

def event85998 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 85982

def event85999 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 85998 .coefficient))

def event86000 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event86001 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11301⟩⟩) 0 ⟨5536⟩ 86000

def event86002 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11301⟩⟩) (.authority (.programFamilyFact))

def exact86003RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩], []⟩, (1)⟩]

theorem exact86003RawTermsValid :
    exact86003RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86003 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11301⟩⟩) exact86003RawTerms (.finite 12) 86002 .exactZero (none)

def event86004 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13773⟩⟩) 0 ⟨5536⟩ 86000

def event86005 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13773⟩⟩) (.authority (.programFamilyFact))

def exact86006RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13773⟩⟩], []⟩, (1)⟩]

theorem exact86006RawTermsValid :
    exact86006RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86006 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13773⟩⟩) exact86006RawTerms (.finite 12) 86005 .exactZero (none)

def event86007 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13774⟩⟩) 0 ⟨13773⟩ 86006

def event86008 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13774⟩⟩) 1 ⟨11301⟩ 86003

def event86009 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13774⟩⟩) (.product (.predecessor 0 86007 .coefficient) (.predecessor 1 86008 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event86010 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13774⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], []⟩) [⟨.result 86006 .coefficient, true, some 1⟩, ⟨.result 86003 .coefficient, true, some 1⟩])

def event86011 : Event := .survivorFold (1) 86010

def exact86012RawTerms : List Term := []

theorem exact86012RawTermsValid :
    exact86012RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86012 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13774⟩⟩) exact86012RawTerms (.finite 144) 86009 (.finite 144) (some (86010))

def event86013 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13775⟩⟩) 0 ⟨13774⟩ 86012

def event86014 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13775⟩⟩) (.identity (.predecessor 0 86013 .coefficient))

def event86015 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13775⟩⟩) (.finite 144)

def eventLeaf5360 : Array AnnotatedEvent := #[
  { event := event85760
    frameStart := 0 },
  { event := event85761
    frameStart := 0 },
  { event := event85762
    frameStart := 0 },
  { event := event85763
    frameStart := 0 },
  { event := event85764
    frameStart := 0 },
  { event := event85765
    frameStart := 0 },
  { event := event85766
    frameStart := 0 },
  { event := event85767
    frameStart := 0 },
  { event := event85768
    frameStart := 0 },
  { event := event85769
    frameStart := 0 },
  { event := event85770
    frameStart := 0 },
  { event := event85771
    frameStart := 0 },
  { event := event85772
    frameStart := 0 },
  { event := event85773
    frameStart := 0 },
  { event := event85774
    frameStart := 0 },
  { event := event85775
    frameStart := 0 }
]

def eventLeaf5361 : Array AnnotatedEvent := #[
  { event := event85776
    frameStart := 0 },
  { event := event85777
    frameStart := 0 },
  { event := event85778
    frameStart := 0 },
  { event := event85779
    frameStart := 85779 },
  { event := event85780
    frameStart := 85779 },
  { event := event85781
    frameStart := 85779 },
  { event := event85782
    frameStart := 85779 },
  { event := event85783
    frameStart := 85779 },
  { event := event85784
    frameStart := 85779 },
  { event := event85785
    frameStart := 85779 },
  { event := event85786
    frameStart := 85779 },
  { event := event85787
    frameStart := 85779 },
  { event := event85788
    frameStart := 85779 },
  { event := event85789
    frameStart := 85779 },
  { event := event85790
    frameStart := 85779 },
  { event := event85791
    frameStart := 85779 }
]

def eventLeaf5362 : Array AnnotatedEvent := #[
  { event := event85792
    frameStart := 85779 },
  { event := event85793
    frameStart := 85779 },
  { event := event85794
    frameStart := 85779 },
  { event := event85795
    frameStart := 85779 },
  { event := event85796
    frameStart := 85779 },
  { event := event85797
    frameStart := 85779 },
  { event := event85798
    frameStart := 85779 },
  { event := event85799
    frameStart := 85779 },
  { event := event85800
    frameStart := 85779 },
  { event := event85801
    frameStart := 85779 },
  { event := event85802
    frameStart := 85779 },
  { event := event85803
    frameStart := 85779 },
  { event := event85804
    frameStart := 85779 },
  { event := event85805
    frameStart := 85779 },
  { event := event85806
    frameStart := 85779 },
  { event := event85807
    frameStart := 85779 }
]

def eventLeaf5363 : Array AnnotatedEvent := #[
  { event := event85808
    frameStart := 85779 },
  { event := event85809
    frameStart := 85779 },
  { event := event85810
    frameStart := 85779 },
  { event := event85811
    frameStart := 85779 },
  { event := event85812
    frameStart := 85779 },
  { event := event85813
    frameStart := 85779 },
  { event := event85814
    frameStart := 85779 },
  { event := event85815
    frameStart := 85779 },
  { event := event85816
    frameStart := 85779 },
  { event := event85817
    frameStart := 85779 },
  { event := event85818
    frameStart := 85779 },
  { event := event85819
    frameStart := 85779 },
  { event := event85820
    frameStart := 85779 },
  { event := event85821
    frameStart := 85779 },
  { event := event85822
    frameStart := 85779 },
  { event := event85823
    frameStart := 85779 }
]

def eventLeaf5364 : Array AnnotatedEvent := #[
  { event := event85824
    frameStart := 85779 },
  { event := event85825
    frameStart := 85779 },
  { event := event85826
    frameStart := 85779 },
  { event := event85827
    frameStart := 85827 },
  { event := event85828
    frameStart := 85827 },
  { event := event85829
    frameStart := 85827 },
  { event := event85830
    frameStart := 85827 },
  { event := event85831
    frameStart := 85827 },
  { event := event85832
    frameStart := 85827 },
  { event := event85833
    frameStart := 85827 },
  { event := event85834
    frameStart := 85827 },
  { event := event85835
    frameStart := 85827 },
  { event := event85836
    frameStart := 85827 },
  { event := event85837
    frameStart := 85827 },
  { event := event85838
    frameStart := 85827 },
  { event := event85839
    frameStart := 85827 }
]

def eventLeaf5365 : Array AnnotatedEvent := #[
  { event := event85840
    frameStart := 85827 },
  { event := event85841
    frameStart := 85827 },
  { event := event85842
    frameStart := 85827 },
  { event := event85843
    frameStart := 85827 },
  { event := event85844
    frameStart := 85827 },
  { event := event85845
    frameStart := 85827 },
  { event := event85846
    frameStart := 85827 },
  { event := event85847
    frameStart := 85827 },
  { event := event85848
    frameStart := 85827 },
  { event := event85849
    frameStart := 85827 },
  { event := event85850
    frameStart := 85827 },
  { event := event85851
    frameStart := 85827 },
  { event := event85852
    frameStart := 85827 },
  { event := event85853
    frameStart := 85827 },
  { event := event85854
    frameStart := 85827 },
  { event := event85855
    frameStart := 85827 }
]

def eventLeaf5366 : Array AnnotatedEvent := #[
  { event := event85856
    frameStart := 85827 },
  { event := event85857
    frameStart := 85827 },
  { event := event85858
    frameStart := 85827 },
  { event := event85859
    frameStart := 85827 },
  { event := event85860
    frameStart := 85827 },
  { event := event85861
    frameStart := 85827 },
  { event := event85862
    frameStart := 85827 },
  { event := event85863
    frameStart := 85827 },
  { event := event85864
    frameStart := 85827 },
  { event := event85865
    frameStart := 85827 },
  { event := event85866
    frameStart := 85827 },
  { event := event85867
    frameStart := 85827 },
  { event := event85868
    frameStart := 85827 },
  { event := event85869
    frameStart := 85827 },
  { event := event85870
    frameStart := 85827 },
  { event := event85871
    frameStart := 85827 }
]

def eventLeaf5367 : Array AnnotatedEvent := #[
  { event := event85872
    frameStart := 85827 },
  { event := event85873
    frameStart := 85827 },
  { event := event85874
    frameStart := 85827 },
  { event := event85875
    frameStart := 85827 },
  { event := event85876
    frameStart := 85827 },
  { event := event85877
    frameStart := 85827 },
  { event := event85878
    frameStart := 85827 },
  { event := event85879
    frameStart := 85827 },
  { event := event85880
    frameStart := 85827 },
  { event := event85881
    frameStart := 85827 },
  { event := event85882
    frameStart := 85827 },
  { event := event85883
    frameStart := 85827 },
  { event := event85884
    frameStart := 85827 },
  { event := event85885
    frameStart := 85827 },
  { event := event85886
    frameStart := 85827 },
  { event := event85887
    frameStart := 85827 }
]

def eventLeaf5368 : Array AnnotatedEvent := #[
  { event := event85888
    frameStart := 85827 },
  { event := event85889
    frameStart := 85827 },
  { event := event85890
    frameStart := 85827 },
  { event := event85891
    frameStart := 85827 },
  { event := event85892
    frameStart := 85827 },
  { event := event85893
    frameStart := 85827 },
  { event := event85894
    frameStart := 85827 },
  { event := event85895
    frameStart := 85827 },
  { event := event85896
    frameStart := 85827 },
  { event := event85897
    frameStart := 85827 },
  { event := event85898
    frameStart := 85827 },
  { event := event85899
    frameStart := 85827 },
  { event := event85900
    frameStart := 85827 },
  { event := event85901
    frameStart := 85827 },
  { event := event85902
    frameStart := 85827 },
  { event := event85903
    frameStart := 85827 }
]

def eventLeaf5369 : Array AnnotatedEvent := #[
  { event := event85904
    frameStart := 85827 },
  { event := event85905
    frameStart := 85827 },
  { event := event85906
    frameStart := 85827 },
  { event := event85907
    frameStart := 85827 },
  { event := event85908
    frameStart := 85827 },
  { event := event85909
    frameStart := 85827 },
  { event := event85910
    frameStart := 85827 },
  { event := event85911
    frameStart := 85827 },
  { event := event85912
    frameStart := 85827 },
  { event := event85913
    frameStart := 85827 },
  { event := event85914
    frameStart := 85827 },
  { event := event85915
    frameStart := 85827 },
  { event := event85916
    frameStart := 85827 },
  { event := event85917
    frameStart := 85827 },
  { event := event85918
    frameStart := 85827 },
  { event := event85919
    frameStart := 85827 }
]

def eventLeaf5370 : Array AnnotatedEvent := #[
  { event := event85920
    frameStart := 85827 },
  { event := event85921
    frameStart := 85827 },
  { event := event85922
    frameStart := 85827 },
  { event := event85923
    frameStart := 85827 },
  { event := event85924
    frameStart := 85827 },
  { event := event85925
    frameStart := 85827 },
  { event := event85926
    frameStart := 85827 },
  { event := event85927
    frameStart := 85827 },
  { event := event85928
    frameStart := 85827 },
  { event := event85929
    frameStart := 85827 },
  { event := event85930
    frameStart := 85827 },
  { event := event85931
    frameStart := 85827 },
  { event := event85932
    frameStart := 85827 },
  { event := event85933
    frameStart := 85827 },
  { event := event85934
    frameStart := 85827 },
  { event := event85935
    frameStart := 85827 }
]

def eventLeaf5371 : Array AnnotatedEvent := #[
  { event := event85936
    frameStart := 85827 },
  { event := event85937
    frameStart := 85827 },
  { event := event85938
    frameStart := 85827 },
  { event := event85939
    frameStart := 85827 },
  { event := event85940
    frameStart := 85827 },
  { event := event85941
    frameStart := 85827 },
  { event := event85942
    frameStart := 85827 },
  { event := event85943
    frameStart := 0 },
  { event := event85944
    frameStart := 0 },
  { event := event85945
    frameStart := 0 },
  { event := event85946
    frameStart := 0 },
  { event := event85947
    frameStart := 0 },
  { event := event85948
    frameStart := 0 },
  { event := event85949
    frameStart := 0 },
  { event := event85950
    frameStart := 0 },
  { event := event85951
    frameStart := 0 }
]

def eventLeaf5372 : Array AnnotatedEvent := #[
  { event := event85952
    frameStart := 0 },
  { event := event85953
    frameStart := 0 },
  { event := event85954
    frameStart := 0 },
  { event := event85955
    frameStart := 0 },
  { event := event85956
    frameStart := 0 },
  { event := event85957
    frameStart := 0 },
  { event := event85958
    frameStart := 0 },
  { event := event85959
    frameStart := 0 },
  { event := event85960
    frameStart := 0 },
  { event := event85961
    frameStart := 0 },
  { event := event85962
    frameStart := 0 },
  { event := event85963
    frameStart := 0 },
  { event := event85964
    frameStart := 0 },
  { event := event85965
    frameStart := 0 },
  { event := event85966
    frameStart := 0 },
  { event := event85967
    frameStart := 0 }
]

def eventLeaf5373 : Array AnnotatedEvent := #[
  { event := event85968
    frameStart := 0 },
  { event := event85969
    frameStart := 0 },
  { event := event85970
    frameStart := 0 },
  { event := event85971
    frameStart := 0 },
  { event := event85972
    frameStart := 0 },
  { event := event85973
    frameStart := 0 },
  { event := event85974
    frameStart := 0 },
  { event := event85975
    frameStart := 0 },
  { event := event85976
    frameStart := 0 },
  { event := event85977
    frameStart := 0 },
  { event := event85978
    frameStart := 0 },
  { event := event85979
    frameStart := 0 },
  { event := event85980
    frameStart := 85980 },
  { event := event85981
    frameStart := 85980 },
  { event := event85982
    frameStart := 85980 },
  { event := event85983
    frameStart := 85980 }
]

def eventLeaf5374 : Array AnnotatedEvent := #[
  { event := event85984
    frameStart := 85980 },
  { event := event85985
    frameStart := 85980 },
  { event := event85986
    frameStart := 85980 },
  { event := event85987
    frameStart := 85980 },
  { event := event85988
    frameStart := 85980 },
  { event := event85989
    frameStart := 85980 },
  { event := event85990
    frameStart := 85980 },
  { event := event85991
    frameStart := 85980 },
  { event := event85992
    frameStart := 85980 },
  { event := event85993
    frameStart := 85980 },
  { event := event85994
    frameStart := 85980 },
  { event := event85995
    frameStart := 85980 },
  { event := event85996
    frameStart := 85980 },
  { event := event85997
    frameStart := 85980 },
  { event := event85998
    frameStart := 85980 },
  { event := event85999
    frameStart := 85980 }
]

def eventLeaf5375 : Array AnnotatedEvent := #[
  { event := event86000
    frameStart := 85980 },
  { event := event86001
    frameStart := 85980 },
  { event := event86002
    frameStart := 85980 },
  { event := event86003
    frameStart := 85980 },
  { event := event86004
    frameStart := 85980 },
  { event := event86005
    frameStart := 85980 },
  { event := event86006
    frameStart := 85980 },
  { event := event86007
    frameStart := 85980 },
  { event := event86008
    frameStart := 85980 },
  { event := event86009
    frameStart := 85980 },
  { event := event86010
    frameStart := 85980 },
  { event := event86011
    frameStart := 85980 },
  { event := event86012
    frameStart := 85980 },
  { event := event86013
    frameStart := 85980 },
  { event := event86014
    frameStart := 85980 },
  { event := event86015
    frameStart := 85980 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events335
