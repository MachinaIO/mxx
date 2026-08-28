import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events167

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event42752 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44021⟩⟩) (.authority (.operator))

def exact42753RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44021⟩⟩]⟩, (1)⟩]

theorem exact42753RawTermsValid :
    exact42753RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42753 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44021⟩⟩) exact42753RawTerms .large 42752 .exactZero (none)

def event42754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44888⟩⟩) 0 ⟨44021⟩ 42753

def event42755 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44888⟩⟩) (.authority (.operator))

def exact42756RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44888⟩⟩]⟩, (1)⟩]

theorem exact42756RawTermsValid :
    exact42756RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42756 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44888⟩⟩) exact42756RawTerms (.finite 8192) 42755 .exactZero (none)

def event42757 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44890⟩⟩) 0 ⟨44400⟩ 33270

def event42758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44890⟩⟩) 1 ⟨44888⟩ 42756

def event42759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44890⟩⟩) (.product (.predecessor 0 42757 .coefficient) (.predecessor 1 42758 .coefficient) (⟨false, false, none, none, none⟩))

def event42760 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44890⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨44888⟩⟩]⟩) [⟨.result 42756 .coefficient, false, none⟩])

def event42761 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44890⟩⟩) (.product (.result 33270 .summary) (.transfer 42760) (⟨false, false, none, none, none⟩))

def event42762 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44890⟩⟩, .operator (⟨33270, 0⟩, ⟨42756, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44888⟩⟩]⟩, (1)⟩)

def event42763 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44890⟩⟩, .operator (⟨33270, 1⟩, ⟨42756, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨42860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44888⟩⟩]⟩, (-1)⟩)

def event42764 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44890⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨42860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44888⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44888⟩⟩) ⟨44021⟩ 42753)

def event42765 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44890⟩⟩, .relation 42764 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨42860⟩⟩], [⟨.program ⟨257⟩, ⟨44021⟩⟩]⟩, (-1)⟩)

def exact42766RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44888⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨42860⟩⟩], [⟨.program ⟨257⟩, ⟨44021⟩⟩]⟩, (-1)⟩]

theorem exact42766RawTermsValid :
    exact42766RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42766 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44890⟩⟩) exact42766RawTerms .large 42759 (.finite 32193718473625689247691015454720) (some (42761))

def event42767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43712⟩⟩) 0 ⟨42861⟩ 905

def event42768 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43712⟩⟩) (.authority (.relationPreimageSource ⟨89⟩))

def exact42769RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43712⟩⟩]⟩, (1)⟩]

theorem exact42769RawTermsValid :
    exact42769RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42769 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43712⟩⟩) exact42769RawTerms (.finite 5647228698) 42768 .exactZero (none)

def event42770 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43714⟩⟩) 0 ⟨43712⟩ 42769

def event42771 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43714⟩⟩) 1 ⟨2370⟩ 4

def event42772 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43714⟩⟩) (.scale (.predecessor 0 42770 .coefficient) (.value (.predecessor 1 42771 .coefficient)))

def exact42773RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43712⟩⟩]⟩, (1)⟩]

theorem exact42773RawTermsValid :
    exact42773RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42773 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43714⟩⟩) exact42773RawTerms (.finite 5647228698) 42772 .exactZero (none)

def event42774 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43715⟩⟩) 0 ⟨11643⟩ 32120

def event42775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43715⟩⟩) 1 ⟨43714⟩ 42773

def event42776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43715⟩⟩) (.product (.predecessor 0 42774 .coefficient) (.predecessor 1 42775 .coefficient) (⟨false, false, none, none, none⟩))

def event42777 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43715⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨43712⟩⟩]⟩) [⟨.result 42769 .coefficient, false, none⟩])

def event42778 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43715⟩⟩) (.product (.result 32120 .summary) (.transfer 42777) (⟨false, false, none, none, none⟩))

def event42779 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43715⟩⟩, .operator (⟨32120, 0⟩, ⟨42773, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43712⟩⟩]⟩, (1)⟩)

def event42780 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨43713⟩⟩)

def event42781 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event42782 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event42783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.authority (.operator))

def event42784 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.finite 18)

def event42785 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event42786 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event42787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event42788 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event42789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 42788

def event42790 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 42786

def event42791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 42789 .coefficient) (.value (.predecessor 1 42790 .coefficient)))

def event42792 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event42793 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 0 ⟨392⟩ 42792

def event42794 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 1 ⟨11541⟩ 42784

def event42795 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.sum [.predecessor 0 42793 .coefficient, .predecessor 1 42794 .coefficient])

def event42796 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.finite 655358)

def event42797 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 0 ⟨11543⟩ 42796

def event42798 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 1 ⟨5426⟩ 42782

def event42799 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.identity (.predecessor 1 42798 .coefficient))

def event42800 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.finite 655360)

def event42801 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42690⟩⟩) 0 ⟨11600⟩ 42800

def event42802 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42690⟩⟩) (.authority (.programFamilyFact))

def exact42803RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42690⟩⟩], []⟩, (1)⟩]

theorem exact42803RawTermsValid :
    exact42803RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42803 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42690⟩⟩) exact42803RawTerms (.finite 52) 42802 .exactZero (none)

def event42804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14616⟩⟩) 0 ⟨11600⟩ 42800

def event42805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14616⟩⟩) (.authority (.programFamilyFact))

def exact42806RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14616⟩⟩], []⟩, (1)⟩]

theorem exact42806RawTermsValid :
    exact42806RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42806 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14616⟩⟩) exact42806RawTerms (.finite 52) 42805 .exactZero (none)

def event42807 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42691⟩⟩) 0 ⟨14616⟩ 42806

def event42808 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42691⟩⟩) 1 ⟨42690⟩ 42803

def event42809 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42691⟩⟩) (.product (.predecessor 0 42807 .coefficient) (.predecessor 1 42808 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event42810 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42691⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14616⟩⟩, ⟨.program ⟨257⟩, ⟨42690⟩⟩], []⟩) [⟨.result 42806 .coefficient, true, some 1⟩, ⟨.result 42803 .coefficient, true, some 1⟩])

def event42811 : Event := .survivorFold (1) 42810

def exact42812RawTerms : List Term := []

theorem exact42812RawTermsValid :
    exact42812RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42812 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42691⟩⟩) exact42812RawTerms (.finite 2704) 42809 (.finite 2704) (some (42810))

def event42813 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42692⟩⟩) 0 ⟨42691⟩ 42812

def event42814 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42692⟩⟩) (.identity (.predecessor 0 42813 .coefficient))

def event42815 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42692⟩⟩) (.finite 2704)

def event42816 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42860⟩⟩) 0 ⟨42692⟩ 42815

def event42817 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42860⟩⟩) (.authority (.programFamilyFact))

def exact42818RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42860⟩⟩], []⟩, (1)⟩]

theorem exact42818RawTermsValid :
    exact42818RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42818 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42860⟩⟩) exact42818RawTerms (.finite 52) 42817 .exactZero (none)

def event42819 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42861⟩⟩) 0 ⟨42860⟩ 42818

def event42820 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42861⟩⟩) (.identity (.predecessor 0 42819 .coefficient))

def event42821 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42861⟩⟩) (.finite 52)

def event42822 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43712⟩⟩) 0 ⟨42861⟩ 42821

def event42823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43712⟩⟩) (.authority (.relationPreimageSource ⟨89⟩))

def exact42824RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43712⟩⟩]⟩, (1)⟩]

theorem exact42824RawTermsValid :
    exact42824RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42824 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43712⟩⟩) exact42824RawTerms (.finite 5647228698) 42823 .exactZero (none)

def event42825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact42826RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact42826RawTermsValid :
    exact42826RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42826 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact42826RawTerms .large 42825 .exactZero (none)

def event42827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43713⟩⟩) 0 ⟨35⟩ 42826

def event42828 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43713⟩⟩) 1 ⟨43712⟩ 42824

def event42829 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43713⟩⟩) (.product (.predecessor 0 42827 .coefficient) (.predecessor 1 42828 .coefficient) (⟨false, false, none, none, none⟩))

def event42830 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43713⟩⟩, .operator (⟨42826, 0⟩, ⟨42824, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43712⟩⟩]⟩, (1)⟩)

def exact42831RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43712⟩⟩]⟩, (1)⟩]

theorem exact42831RawTermsValid :
    exact42831RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42831 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43713⟩⟩) exact42831RawTerms .large 42829 .exactZero (none)

def event42832 : Event := .preFoldPolynomial 42831 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43712⟩⟩]⟩, (1)⟩] .exactZero none

def exact42833RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43712⟩⟩]⟩, (1)⟩]

def event42833 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨43713⟩⟩) 42832 exact42833RawTerms .large 42829 .exactZero (none)

def event42834 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨44893⟩⟩)

def event42835 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event42836 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event42837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.authority (.operator))

def event42838 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.finite 18)

def event42839 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event42840 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event42841 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event42842 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event42843 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 42842

def event42844 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 42840

def event42845 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 42843 .coefficient) (.value (.predecessor 1 42844 .coefficient)))

def event42846 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event42847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 0 ⟨392⟩ 42846

def event42848 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 1 ⟨11541⟩ 42838

def event42849 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.sum [.predecessor 0 42847 .coefficient, .predecessor 1 42848 .coefficient])

def event42850 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.finite 655358)

def event42851 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 0 ⟨11543⟩ 42850

def event42852 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11600⟩⟩) 1 ⟨5426⟩ 42836

def event42853 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.identity (.predecessor 1 42852 .coefficient))

def event42854 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11600⟩⟩) (.finite 655360)

def event42855 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42690⟩⟩) 0 ⟨11600⟩ 42854

def event42856 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42690⟩⟩) (.authority (.programFamilyFact))

def exact42857RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42690⟩⟩], []⟩, (1)⟩]

theorem exact42857RawTermsValid :
    exact42857RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42857 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42690⟩⟩) exact42857RawTerms (.finite 52) 42856 .exactZero (none)

def event42858 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14616⟩⟩) 0 ⟨11600⟩ 42854

def event42859 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14616⟩⟩) (.authority (.programFamilyFact))

def exact42860RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14616⟩⟩], []⟩, (1)⟩]

theorem exact42860RawTermsValid :
    exact42860RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42860 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14616⟩⟩) exact42860RawTerms (.finite 52) 42859 .exactZero (none)

def event42861 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42691⟩⟩) 0 ⟨14616⟩ 42860

def event42862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42691⟩⟩) 1 ⟨42690⟩ 42857

def event42863 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42691⟩⟩) (.product (.predecessor 0 42861 .coefficient) (.predecessor 1 42862 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event42864 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42691⟩⟩, .operator (⟨42860, 0⟩, ⟨42857, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14616⟩⟩, ⟨.program ⟨257⟩, ⟨42690⟩⟩], []⟩, (1)⟩)

def exact42865RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14616⟩⟩, ⟨.program ⟨257⟩, ⟨42690⟩⟩], []⟩, (1)⟩]

theorem exact42865RawTermsValid :
    exact42865RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42865 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42691⟩⟩) exact42865RawTerms (.finite 2704) 42863 .exactZero (none)

def event42866 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42692⟩⟩) 0 ⟨42691⟩ 42865

def event42867 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42692⟩⟩) (.identity (.predecessor 0 42866 .coefficient))

def event42868 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42692⟩⟩) (.finite 2704)

def event42869 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42860⟩⟩) 0 ⟨42692⟩ 42868

def event42870 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42860⟩⟩) (.authority (.programFamilyFact))

def exact42871RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42860⟩⟩], []⟩, (1)⟩]

theorem exact42871RawTermsValid :
    exact42871RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42871 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42860⟩⟩) exact42871RawTerms (.finite 52) 42870 .exactZero (none)

def event42872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42861⟩⟩) 0 ⟨42860⟩ 42871

def event42873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42861⟩⟩) (.identity (.predecessor 0 42872 .coefficient))

def event42874 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42861⟩⟩) (.finite 52)

def event42875 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44020⟩⟩) 0 ⟨42861⟩ 42874

def event42876 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44020⟩⟩) (.authority (.programFamilyFact))

def event42877 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44020⟩⟩) (.finite 3720)

def event42878 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event42879 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44021⟩⟩) 0 ⟨7177⟩ 42878

def event42880 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44021⟩⟩) 1 ⟨44020⟩ 42877

def event42881 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44021⟩⟩) (.authority (.operator))

def exact42882RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44021⟩⟩]⟩, (1)⟩]

theorem exact42882RawTermsValid :
    exact42882RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42882 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44021⟩⟩) exact42882RawTerms .large 42881 .exactZero (none)

def event42883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44888⟩⟩) 0 ⟨44021⟩ 42882

def event42884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44888⟩⟩) (.authority (.operator))

def exact42885RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44888⟩⟩]⟩, (1)⟩]

theorem exact42885RawTermsValid :
    exact42885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42885 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44888⟩⟩) exact42885RawTerms (.finite 8192) 42884 .exactZero (none)

def event42886 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event42887 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event42888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44182⟩⟩) 0 ⟨42861⟩ 42874

def event42889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44182⟩⟩) 1 ⟨136⟩ 42887

def event42890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44182⟩⟩) (.sum [.predecessor 0 42888 .coefficient, .predecessor 1 42889 .coefficient])

def event42891 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44182⟩⟩) (.finite 52)

def event42892 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44183⟩⟩) 0 ⟨44182⟩ 42891

def event42893 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44183⟩⟩) (.identity (.predecessor 0 42892 .coefficient))

def exact42894RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42860⟩⟩], []⟩, (1)⟩]

theorem exact42894RawTermsValid :
    exact42894RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42894 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44183⟩⟩) exact42894RawTerms (.finite 52) 42893 .exactZero (none)

def event42895 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact42896RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact42896RawTermsValid :
    exact42896RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42896 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact42896RawTerms .large 42895 .exactZero (none)

def event42897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44184⟩⟩) 0 ⟨6908⟩ 42896

def event42898 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44184⟩⟩) 1 ⟨44183⟩ 42894

def event42899 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44184⟩⟩) (.product (.predecessor 0 42897 .coefficient) (.predecessor 1 42898 .coefficient) (⟨false, false, none, none, none⟩))

def event42900 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44184⟩⟩, .operator (⟨42896, 0⟩, ⟨42894, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact42901RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact42901RawTermsValid :
    exact42901RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42901 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44184⟩⟩) exact42901RawTerms .large 42899 .exactZero (none)

def event42902 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7194⟩⟩) 0 ⟨7177⟩ 42878

def event42903 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7194⟩⟩) (.authority (.operator))

def exact42904RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩]

theorem exact42904RawTermsValid :
    exact42904RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42904 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7194⟩⟩) exact42904RawTerms .large 42903 .exactZero (none)

def event42905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44185⟩⟩) 0 ⟨7194⟩ 42904

def event42906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44185⟩⟩) 1 ⟨44184⟩ 42901

def event42907 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44185⟩⟩) (.sum [.predecessor 0 42905 .coefficient, .predecessor 1 42906 .coefficient])

def exact42908RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact42908RawTermsValid :
    exact42908RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42908 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44185⟩⟩) exact42908RawTerms .large 42907 .exactZero (none)

def event42909 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44889⟩⟩) 0 ⟨44185⟩ 42908

def event42910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44889⟩⟩) 1 ⟨44888⟩ 42885

def event42911 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44889⟩⟩) (.product (.predecessor 0 42909 .coefficient) (.predecessor 1 42910 .coefficient) (⟨false, false, none, none, none⟩))

def event42912 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44889⟩⟩, .operator (⟨42908, 0⟩, ⟨42885, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44888⟩⟩]⟩, (1)⟩)

def event42913 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44889⟩⟩, .operator (⟨42908, 1⟩, ⟨42885, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44888⟩⟩]⟩, (-1)⟩)

def event42914 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44889⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨42860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44888⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44888⟩⟩) ⟨44021⟩ 42882)

def event42915 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44889⟩⟩, .relation 42914 0, ⟨[⟨.program ⟨257⟩, ⟨42860⟩⟩], [⟨.program ⟨257⟩, ⟨44021⟩⟩]⟩, (-1)⟩)

def exact42916RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44888⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42860⟩⟩], [⟨.program ⟨257⟩, ⟨44021⟩⟩]⟩, (-1)⟩]

theorem exact42916RawTermsValid :
    exact42916RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42916 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44889⟩⟩) exact42916RawTerms .large 42911 .exactZero (none)

def event42917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43119⟩⟩) 0 ⟨42861⟩ 42874

def event42918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43119⟩⟩) (.authority (.programFamilyFact))

def exact42919RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨43119⟩⟩], []⟩, (1)⟩]

theorem exact42919RawTermsValid :
    exact42919RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42919 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43119⟩⟩) exact42919RawTerms (.finite 52) 42918 .exactZero (none)

def event42920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43121⟩⟩) 0 ⟨6908⟩ 42896

def event42921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43121⟩⟩) 1 ⟨43119⟩ 42919

def event42922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43121⟩⟩) (.product (.predecessor 0 42920 .coefficient) (.predecessor 1 42921 .coefficient) (⟨false, true, none, none, some 1⟩))

def event42923 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43121⟩⟩, .operator (⟨42896, 0⟩, ⟨42919, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨43119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact42924RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨43119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact42924RawTermsValid :
    exact42924RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42924 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43121⟩⟩) exact42924RawTerms .large 42922 .exactZero (none)

def event42925 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7227⟩⟩) 0 ⟨7177⟩ 42878

def event42926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7227⟩⟩) (.authority (.operator))

def exact42927RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩]

theorem exact42927RawTermsValid :
    exact42927RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42927 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7227⟩⟩) exact42927RawTerms .large 42926 .exactZero (none)

def event42928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43122⟩⟩) 0 ⟨7227⟩ 42927

def event42929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43122⟩⟩) 1 ⟨43121⟩ 42924

def event42930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43122⟩⟩) (.sum [.predecessor 0 42928 .coefficient, .predecessor 1 42929 .coefficient])

def exact42931RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact42931RawTermsValid :
    exact42931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42931 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43122⟩⟩) exact42931RawTerms .large 42930 .exactZero (none)

def event42932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44893⟩⟩) 0 ⟨43122⟩ 42931

def event42933 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44893⟩⟩) 1 ⟨44889⟩ 42916

def event42934 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44893⟩⟩) (.sum [.predecessor 0 42932 .coefficient, .predecessor 1 42933 .coefficient])

def exact42935RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44888⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42860⟩⟩], [⟨.program ⟨257⟩, ⟨44021⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact42935RawTermsValid :
    exact42935RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42935 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44893⟩⟩) exact42935RawTerms .large 42934 .exactZero (none)

def event42936 : Event := .preFoldPolynomial 42935 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44888⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42860⟩⟩], [⟨.program ⟨257⟩, ⟨44021⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact42937RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44888⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42860⟩⟩], [⟨.program ⟨257⟩, ⟨44021⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event42937 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨44893⟩⟩) 42936 exact42937RawTerms .large 42934 .exactZero (none)

def event42938 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨42861⟩⟩) ⟨⟨106⟩, ⟨89⟩, ⟨135⟩⟩ ⟨42780, 42938⟩

def event42939 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨43715⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43712⟩⟩]⟩) (1) 0 2 (.universal 42938 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43712⟩⟩]⟩) (none) 42937)

def event42940 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43715⟩⟩, .relation 42939 1, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩)

def event42941 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43715⟩⟩, .relation 42939 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44888⟩⟩]⟩, (-1)⟩)

def event42942 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43715⟩⟩, .relation 42939 2, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨42860⟩⟩], [⟨.program ⟨257⟩, ⟨44021⟩⟩]⟩, (1)⟩)

def event42943 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43715⟩⟩, .relation 42939 3, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact42944RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44888⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨42860⟩⟩], [⟨.program ⟨257⟩, ⟨44021⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact42944RawTermsValid :
    exact42944RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42944 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43715⟩⟩) exact42944RawTerms .large 42776 (.finite 202072841853861888) (some (42778))

def event42945 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44891⟩⟩) 0 ⟨43715⟩ 42944

def event42946 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44891⟩⟩) 1 ⟨44890⟩ 42766

def event42947 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44891⟩⟩) (.sum [.predecessor 0 42945 .coefficient, .predecessor 1 42946 .coefficient])

def event42948 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44891⟩⟩, .operator (⟨42944, 0⟩, ⟨42766, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44888⟩⟩]⟩, (1)⟩)

def event42949 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44891⟩⟩, .operator (⟨42944, 2⟩, ⟨42766, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨42860⟩⟩], [⟨.program ⟨257⟩, ⟨44021⟩⟩]⟩, (-1)⟩)

def event42950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44891⟩⟩) (.sum [.result 42944 .summary, .result 42766 .summary])

def exact42951RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact42951RawTermsValid :
    exact42951RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42951 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44891⟩⟩) exact42951RawTerms .large 42947 (.finite 32193718473625891320532869316608) (some (42950))

def event42952 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44892⟩⟩) 0 ⟨44891⟩ 42951

def event42953 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44892⟩⟩) 1 ⟨7154⟩ 15582

def event42954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44892⟩⟩) (.product (.predecessor 0 42952 .coefficient) (.predecessor 1 42953 .coefficient) (⟨false, false, none, none, none⟩))

def event42955 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44892⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩) [⟨.result 15578 .coefficient, false, none⟩])

def event42956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44892⟩⟩) (.product (.result 42951 .summary) (.transfer 42955) (⟨false, false, none, none, none⟩))

def event42957 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44892⟩⟩, .operator (⟨42951, 0⟩, ⟨15582, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩)

def event42958 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44892⟩⟩, .operator (⟨42951, 1⟩, ⟨15582, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (-1)⟩)

def event42959 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44892⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7153⟩⟩) ⟨7042⟩ 15575)

def event42960 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44892⟩⟩, .relation 42959 0, ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact42961RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨43119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩]

theorem exact42961RawTermsValid :
    exact42961RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42961 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44892⟩⟩) exact42961RawTerms .large 42954 (.finite 345677419952135604401347317519683074129920) (some (42956))

def event42962 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41341⟩⟩) 0 ⟨7177⟩ 15500

def event42963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41341⟩⟩) 1 ⟨41340⟩ 33468

def event42964 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41341⟩⟩) (.authority (.operator))

def exact42965RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41341⟩⟩]⟩, (1)⟩]

theorem exact42965RawTermsValid :
    exact42965RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42965 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41341⟩⟩) exact42965RawTerms .large 42964 .exactZero (none)

def event42966 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42208⟩⟩) 0 ⟨41341⟩ 42965

def event42967 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42208⟩⟩) (.authority (.operator))

def exact42968RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨42208⟩⟩]⟩, (1)⟩]

theorem exact42968RawTermsValid :
    exact42968RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42968 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42208⟩⟩) exact42968RawTerms (.finite 8192) 42967 .exactZero (none)

def event42969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42210⟩⟩) 0 ⟨41720⟩ 33752

def event42970 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42210⟩⟩) 1 ⟨42208⟩ 42968

def event42971 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42210⟩⟩) (.product (.predecessor 0 42969 .coefficient) (.predecessor 1 42970 .coefficient) (⟨false, false, none, none, none⟩))

def event42972 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42210⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨42208⟩⟩]⟩) [⟨.result 42968 .coefficient, false, none⟩])

def event42973 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42210⟩⟩) (.product (.result 33752 .summary) (.transfer 42972) (⟨false, false, none, none, none⟩))

def event42974 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42210⟩⟩, .operator (⟨33752, 0⟩, ⟨42968, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42208⟩⟩]⟩, (1)⟩)

def event42975 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42210⟩⟩, .operator (⟨33752, 1⟩, ⟨42968, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42208⟩⟩]⟩, (-1)⟩)

def event42976 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨42210⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42208⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42208⟩⟩) ⟨41341⟩ 42965)

def event42977 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42210⟩⟩, .relation 42976 0, ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨41341⟩⟩]⟩, (-1)⟩)

def exact42978RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨41341⟩⟩]⟩, (-1)⟩]

theorem exact42978RawTermsValid :
    exact42978RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42978 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42210⟩⟩) exact42978RawTerms .large 42971 (.finite 32193129122288627115968346193920) (some (42973))

def event42979 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41032⟩⟩) 0 ⟨40181⟩ 928

def event42980 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41032⟩⟩) (.authority (.relationPreimageSource ⟨86⟩))

def exact42981RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41032⟩⟩]⟩, (1)⟩]

theorem exact42981RawTermsValid :
    exact42981RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42981 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41032⟩⟩) exact42981RawTerms (.finite 5647228698) 42980 .exactZero (none)

def event42982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41034⟩⟩) 0 ⟨41032⟩ 42981

def event42983 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41034⟩⟩) 1 ⟨2370⟩ 4

def event42984 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41034⟩⟩) (.scale (.predecessor 0 42982 .coefficient) (.value (.predecessor 1 42983 .coefficient)))

def exact42985RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41032⟩⟩]⟩, (1)⟩]

theorem exact42985RawTermsValid :
    exact42985RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event42985 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41034⟩⟩) exact42985RawTerms (.finite 5647228698) 42984 .exactZero (none)

def event42986 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41035⟩⟩) 0 ⟨11643⟩ 32120

def event42987 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41035⟩⟩) 1 ⟨41034⟩ 42985

def event42988 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41035⟩⟩) (.product (.predecessor 0 42986 .coefficient) (.predecessor 1 42987 .coefficient) (⟨false, false, none, none, none⟩))

def event42989 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41035⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨41032⟩⟩]⟩) [⟨.result 42981 .coefficient, false, none⟩])

def event42990 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41035⟩⟩) (.product (.result 32120 .summary) (.transfer 42989) (⟨false, false, none, none, none⟩))

def event42991 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41035⟩⟩, .operator (⟨32120, 0⟩, ⟨42985, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41032⟩⟩]⟩, (1)⟩)

def event42992 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨41033⟩⟩)

def event42993 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event42994 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event42995 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.authority (.operator))

def event42996 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11541⟩⟩) (.finite 18)

def event42997 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event42998 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event42999 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event43000 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event43001 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 43000

def event43002 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 42998

def event43003 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 43001 .coefficient) (.value (.predecessor 1 43002 .coefficient)))

def event43004 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event43005 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 0 ⟨392⟩ 43004

def event43006 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11543⟩⟩) 1 ⟨11541⟩ 42996

def event43007 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11543⟩⟩) (.sum [.predecessor 0 43005 .coefficient, .predecessor 1 43006 .coefficient])

def eventLeaf2672 : Array AnnotatedEvent := #[
  { event := event42752
    frameStart := 0 },
  { event := event42753
    frameStart := 0 },
  { event := event42754
    frameStart := 0 },
  { event := event42755
    frameStart := 0 },
  { event := event42756
    frameStart := 0 },
  { event := event42757
    frameStart := 0 },
  { event := event42758
    frameStart := 0 },
  { event := event42759
    frameStart := 0 },
  { event := event42760
    frameStart := 0 },
  { event := event42761
    frameStart := 0 },
  { event := event42762
    frameStart := 0 },
  { event := event42763
    frameStart := 0 },
  { event := event42764
    frameStart := 0 },
  { event := event42765
    frameStart := 0 },
  { event := event42766
    frameStart := 0 },
  { event := event42767
    frameStart := 0 }
]

def eventLeaf2673 : Array AnnotatedEvent := #[
  { event := event42768
    frameStart := 0 },
  { event := event42769
    frameStart := 0 },
  { event := event42770
    frameStart := 0 },
  { event := event42771
    frameStart := 0 },
  { event := event42772
    frameStart := 0 },
  { event := event42773
    frameStart := 0 },
  { event := event42774
    frameStart := 0 },
  { event := event42775
    frameStart := 0 },
  { event := event42776
    frameStart := 0 },
  { event := event42777
    frameStart := 0 },
  { event := event42778
    frameStart := 0 },
  { event := event42779
    frameStart := 0 },
  { event := event42780
    frameStart := 42780 },
  { event := event42781
    frameStart := 42780 },
  { event := event42782
    frameStart := 42780 },
  { event := event42783
    frameStart := 42780 }
]

def eventLeaf2674 : Array AnnotatedEvent := #[
  { event := event42784
    frameStart := 42780 },
  { event := event42785
    frameStart := 42780 },
  { event := event42786
    frameStart := 42780 },
  { event := event42787
    frameStart := 42780 },
  { event := event42788
    frameStart := 42780 },
  { event := event42789
    frameStart := 42780 },
  { event := event42790
    frameStart := 42780 },
  { event := event42791
    frameStart := 42780 },
  { event := event42792
    frameStart := 42780 },
  { event := event42793
    frameStart := 42780 },
  { event := event42794
    frameStart := 42780 },
  { event := event42795
    frameStart := 42780 },
  { event := event42796
    frameStart := 42780 },
  { event := event42797
    frameStart := 42780 },
  { event := event42798
    frameStart := 42780 },
  { event := event42799
    frameStart := 42780 }
]

def eventLeaf2675 : Array AnnotatedEvent := #[
  { event := event42800
    frameStart := 42780 },
  { event := event42801
    frameStart := 42780 },
  { event := event42802
    frameStart := 42780 },
  { event := event42803
    frameStart := 42780 },
  { event := event42804
    frameStart := 42780 },
  { event := event42805
    frameStart := 42780 },
  { event := event42806
    frameStart := 42780 },
  { event := event42807
    frameStart := 42780 },
  { event := event42808
    frameStart := 42780 },
  { event := event42809
    frameStart := 42780 },
  { event := event42810
    frameStart := 42780 },
  { event := event42811
    frameStart := 42780 },
  { event := event42812
    frameStart := 42780 },
  { event := event42813
    frameStart := 42780 },
  { event := event42814
    frameStart := 42780 },
  { event := event42815
    frameStart := 42780 }
]

def eventLeaf2676 : Array AnnotatedEvent := #[
  { event := event42816
    frameStart := 42780 },
  { event := event42817
    frameStart := 42780 },
  { event := event42818
    frameStart := 42780 },
  { event := event42819
    frameStart := 42780 },
  { event := event42820
    frameStart := 42780 },
  { event := event42821
    frameStart := 42780 },
  { event := event42822
    frameStart := 42780 },
  { event := event42823
    frameStart := 42780 },
  { event := event42824
    frameStart := 42780 },
  { event := event42825
    frameStart := 42780 },
  { event := event42826
    frameStart := 42780 },
  { event := event42827
    frameStart := 42780 },
  { event := event42828
    frameStart := 42780 },
  { event := event42829
    frameStart := 42780 },
  { event := event42830
    frameStart := 42780 },
  { event := event42831
    frameStart := 42780 }
]

def eventLeaf2677 : Array AnnotatedEvent := #[
  { event := event42832
    frameStart := 42780 },
  { event := event42833
    frameStart := 42780 },
  { event := event42834
    frameStart := 42834 },
  { event := event42835
    frameStart := 42834 },
  { event := event42836
    frameStart := 42834 },
  { event := event42837
    frameStart := 42834 },
  { event := event42838
    frameStart := 42834 },
  { event := event42839
    frameStart := 42834 },
  { event := event42840
    frameStart := 42834 },
  { event := event42841
    frameStart := 42834 },
  { event := event42842
    frameStart := 42834 },
  { event := event42843
    frameStart := 42834 },
  { event := event42844
    frameStart := 42834 },
  { event := event42845
    frameStart := 42834 },
  { event := event42846
    frameStart := 42834 },
  { event := event42847
    frameStart := 42834 }
]

def eventLeaf2678 : Array AnnotatedEvent := #[
  { event := event42848
    frameStart := 42834 },
  { event := event42849
    frameStart := 42834 },
  { event := event42850
    frameStart := 42834 },
  { event := event42851
    frameStart := 42834 },
  { event := event42852
    frameStart := 42834 },
  { event := event42853
    frameStart := 42834 },
  { event := event42854
    frameStart := 42834 },
  { event := event42855
    frameStart := 42834 },
  { event := event42856
    frameStart := 42834 },
  { event := event42857
    frameStart := 42834 },
  { event := event42858
    frameStart := 42834 },
  { event := event42859
    frameStart := 42834 },
  { event := event42860
    frameStart := 42834 },
  { event := event42861
    frameStart := 42834 },
  { event := event42862
    frameStart := 42834 },
  { event := event42863
    frameStart := 42834 }
]

def eventLeaf2679 : Array AnnotatedEvent := #[
  { event := event42864
    frameStart := 42834 },
  { event := event42865
    frameStart := 42834 },
  { event := event42866
    frameStart := 42834 },
  { event := event42867
    frameStart := 42834 },
  { event := event42868
    frameStart := 42834 },
  { event := event42869
    frameStart := 42834 },
  { event := event42870
    frameStart := 42834 },
  { event := event42871
    frameStart := 42834 },
  { event := event42872
    frameStart := 42834 },
  { event := event42873
    frameStart := 42834 },
  { event := event42874
    frameStart := 42834 },
  { event := event42875
    frameStart := 42834 },
  { event := event42876
    frameStart := 42834 },
  { event := event42877
    frameStart := 42834 },
  { event := event42878
    frameStart := 42834 },
  { event := event42879
    frameStart := 42834 }
]

def eventLeaf2680 : Array AnnotatedEvent := #[
  { event := event42880
    frameStart := 42834 },
  { event := event42881
    frameStart := 42834 },
  { event := event42882
    frameStart := 42834 },
  { event := event42883
    frameStart := 42834 },
  { event := event42884
    frameStart := 42834 },
  { event := event42885
    frameStart := 42834 },
  { event := event42886
    frameStart := 42834 },
  { event := event42887
    frameStart := 42834 },
  { event := event42888
    frameStart := 42834 },
  { event := event42889
    frameStart := 42834 },
  { event := event42890
    frameStart := 42834 },
  { event := event42891
    frameStart := 42834 },
  { event := event42892
    frameStart := 42834 },
  { event := event42893
    frameStart := 42834 },
  { event := event42894
    frameStart := 42834 },
  { event := event42895
    frameStart := 42834 }
]

def eventLeaf2681 : Array AnnotatedEvent := #[
  { event := event42896
    frameStart := 42834 },
  { event := event42897
    frameStart := 42834 },
  { event := event42898
    frameStart := 42834 },
  { event := event42899
    frameStart := 42834 },
  { event := event42900
    frameStart := 42834 },
  { event := event42901
    frameStart := 42834 },
  { event := event42902
    frameStart := 42834 },
  { event := event42903
    frameStart := 42834 },
  { event := event42904
    frameStart := 42834 },
  { event := event42905
    frameStart := 42834 },
  { event := event42906
    frameStart := 42834 },
  { event := event42907
    frameStart := 42834 },
  { event := event42908
    frameStart := 42834 },
  { event := event42909
    frameStart := 42834 },
  { event := event42910
    frameStart := 42834 },
  { event := event42911
    frameStart := 42834 }
]

def eventLeaf2682 : Array AnnotatedEvent := #[
  { event := event42912
    frameStart := 42834 },
  { event := event42913
    frameStart := 42834 },
  { event := event42914
    frameStart := 42834 },
  { event := event42915
    frameStart := 42834 },
  { event := event42916
    frameStart := 42834 },
  { event := event42917
    frameStart := 42834 },
  { event := event42918
    frameStart := 42834 },
  { event := event42919
    frameStart := 42834 },
  { event := event42920
    frameStart := 42834 },
  { event := event42921
    frameStart := 42834 },
  { event := event42922
    frameStart := 42834 },
  { event := event42923
    frameStart := 42834 },
  { event := event42924
    frameStart := 42834 },
  { event := event42925
    frameStart := 42834 },
  { event := event42926
    frameStart := 42834 },
  { event := event42927
    frameStart := 42834 }
]

def eventLeaf2683 : Array AnnotatedEvent := #[
  { event := event42928
    frameStart := 42834 },
  { event := event42929
    frameStart := 42834 },
  { event := event42930
    frameStart := 42834 },
  { event := event42931
    frameStart := 42834 },
  { event := event42932
    frameStart := 42834 },
  { event := event42933
    frameStart := 42834 },
  { event := event42934
    frameStart := 42834 },
  { event := event42935
    frameStart := 42834 },
  { event := event42936
    frameStart := 42834 },
  { event := event42937
    frameStart := 42834 },
  { event := event42938
    frameStart := 0 },
  { event := event42939
    frameStart := 0 },
  { event := event42940
    frameStart := 0 },
  { event := event42941
    frameStart := 0 },
  { event := event42942
    frameStart := 0 },
  { event := event42943
    frameStart := 0 }
]

def eventLeaf2684 : Array AnnotatedEvent := #[
  { event := event42944
    frameStart := 0 },
  { event := event42945
    frameStart := 0 },
  { event := event42946
    frameStart := 0 },
  { event := event42947
    frameStart := 0 },
  { event := event42948
    frameStart := 0 },
  { event := event42949
    frameStart := 0 },
  { event := event42950
    frameStart := 0 },
  { event := event42951
    frameStart := 0 },
  { event := event42952
    frameStart := 0 },
  { event := event42953
    frameStart := 0 },
  { event := event42954
    frameStart := 0 },
  { event := event42955
    frameStart := 0 },
  { event := event42956
    frameStart := 0 },
  { event := event42957
    frameStart := 0 },
  { event := event42958
    frameStart := 0 },
  { event := event42959
    frameStart := 0 }
]

def eventLeaf2685 : Array AnnotatedEvent := #[
  { event := event42960
    frameStart := 0 },
  { event := event42961
    frameStart := 0 },
  { event := event42962
    frameStart := 0 },
  { event := event42963
    frameStart := 0 },
  { event := event42964
    frameStart := 0 },
  { event := event42965
    frameStart := 0 },
  { event := event42966
    frameStart := 0 },
  { event := event42967
    frameStart := 0 },
  { event := event42968
    frameStart := 0 },
  { event := event42969
    frameStart := 0 },
  { event := event42970
    frameStart := 0 },
  { event := event42971
    frameStart := 0 },
  { event := event42972
    frameStart := 0 },
  { event := event42973
    frameStart := 0 },
  { event := event42974
    frameStart := 0 },
  { event := event42975
    frameStart := 0 }
]

def eventLeaf2686 : Array AnnotatedEvent := #[
  { event := event42976
    frameStart := 0 },
  { event := event42977
    frameStart := 0 },
  { event := event42978
    frameStart := 0 },
  { event := event42979
    frameStart := 0 },
  { event := event42980
    frameStart := 0 },
  { event := event42981
    frameStart := 0 },
  { event := event42982
    frameStart := 0 },
  { event := event42983
    frameStart := 0 },
  { event := event42984
    frameStart := 0 },
  { event := event42985
    frameStart := 0 },
  { event := event42986
    frameStart := 0 },
  { event := event42987
    frameStart := 0 },
  { event := event42988
    frameStart := 0 },
  { event := event42989
    frameStart := 0 },
  { event := event42990
    frameStart := 0 },
  { event := event42991
    frameStart := 0 }
]

def eventLeaf2687 : Array AnnotatedEvent := #[
  { event := event42992
    frameStart := 42992 },
  { event := event42993
    frameStart := 42992 },
  { event := event42994
    frameStart := 42992 },
  { event := event42995
    frameStart := 42992 },
  { event := event42996
    frameStart := 42992 },
  { event := event42997
    frameStart := 42992 },
  { event := event42998
    frameStart := 42992 },
  { event := event42999
    frameStart := 42992 },
  { event := event43000
    frameStart := 42992 },
  { event := event43001
    frameStart := 42992 },
  { event := event43002
    frameStart := 42992 },
  { event := event43003
    frameStart := 42992 },
  { event := event43004
    frameStart := 42992 },
  { event := event43005
    frameStart := 42992 },
  { event := event43006
    frameStart := 42992 },
  { event := event43007
    frameStart := 42992 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events167
