import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events042

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event10752 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26242⟩⟩, .operator (⟨10748, 2⟩, ⟨10562, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11653⟩⟩, ⟨.program ⟨214⟩, ⟨14677⟩⟩], [⟨.program ⟨214⟩, ⟨23676⟩⟩]⟩, (-1)⟩)

def event10753 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26242⟩⟩, .operator (⟨10748, 1⟩, ⟨10562, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26240⟩⟩]⟩, (1)⟩)

def event10754 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26242⟩⟩) (.sum [.result 10748 .summary, .result 10562 .summary])

def exact10755RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact10755RawTermsValid :
    exact10755RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10755 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26242⟩⟩) exact10755RawTerms .large 10751 (.finite 352091253649408) (some (10754))

def event10756 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28354⟩⟩) 0 ⟨26242⟩ 10755

def event10757 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28354⟩⟩) 1 ⟨28352⟩ 10459

def event10758 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28354⟩⟩) (.product (.predecessor 0 10756 .coefficient) (.predecessor 1 10757 .coefficient) (⟨false, false, none, none, none⟩))

def event10759 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28354⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩) [⟨.result 10459 .coefficient, false, none⟩])

def event10760 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28354⟩⟩) (.product (.result 10755 .summary) (.transfer 10759) (⟨false, false, none, none, none⟩))

def event10761 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28354⟩⟩, .operator (⟨10755, 1⟩, ⟨10459, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩, (-1)⟩)

def event10762 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28354⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28352⟩⟩) ⟨24300⟩ 10456)

def event10763 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28354⟩⟩, .relation 10762 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨24300⟩⟩]⟩, (-1)⟩)

def event10764 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28354⟩⟩, .operator (⟨10755, 0⟩, ⟨10459, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩, (1)⟩)

def exact10765RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨24300⟩⟩]⟩, (-1)⟩]

theorem exact10765RawTermsValid :
    exact10765RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10765 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28354⟩⟩) exact10765RawTerms .large 10758 (.finite 1292180534353385750528) (some (10760))

def event10766 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21704⟩⟩) 0 ⟨16195⟩ 252

def event10767 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21704⟩⟩) (.authority (.relationPreimageSource ⟨48⟩))

def exact10768RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩, (1)⟩]

theorem exact10768RawTermsValid :
    exact10768RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10768 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21704⟩⟩) exact10768RawTerms (.finite 136065468) 10767 .exactZero (none)

def event10769 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21706⟩⟩) 0 ⟨21704⟩ 10768

def event10770 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21706⟩⟩) 1 ⟨2348⟩ 4

def event10771 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21706⟩⟩) (.scale (.predecessor 0 10769 .coefficient) (.value (.predecessor 1 10770 .coefficient)))

def exact10772RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩, (1)⟩]

theorem exact10772RawTermsValid :
    exact10772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10772 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21706⟩⟩) exact10772RawTerms (.finite 136065468) 10771 .exactZero (none)

def event10773 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21707⟩⟩) 0 ⟨5565⟩ 6561

def event10774 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21707⟩⟩) 1 ⟨21706⟩ 10772

def event10775 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21707⟩⟩) (.product (.predecessor 0 10773 .coefficient) (.predecessor 1 10774 .coefficient) (⟨false, false, none, none, none⟩))

def event10776 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21707⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩) [⟨.result 10768 .coefficient, false, none⟩])

def event10777 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21707⟩⟩) (.product (.result 6561 .summary) (.transfer 10776) (⟨false, false, none, none, none⟩))

def event10778 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21707⟩⟩, .operator (⟨6561, 0⟩, ⟨10772, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩, (1)⟩)

def event10779 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21705⟩⟩)

def event10780 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event10781 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event10782 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.authority (.operator))

def event10783 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.finite 6)

def event10784 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event10785 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event10786 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event10787 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event10788 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 10787

def event10789 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 10785

def event10790 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 10788 .coefficient) (.value (.predecessor 1 10789 .coefficient)))

def event10791 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event10792 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 0 ⟨5503⟩ 10791

def event10793 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 1 ⟨5245⟩ 10783

def event10794 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.sum [.predecessor 0 10792 .coefficient, .predecessor 1 10793 .coefficient])

def event10795 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.finite 223)

def event10796 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 0 ⟨5516⟩ 10795

def event10797 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 1 ⟨961⟩ 10781

def event10798 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.identity (.predecessor 1 10797 .coefficient))

def event10799 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.finite 224)

def event10800 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11653⟩⟩) 0 ⟨5560⟩ 10799

def event10801 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11653⟩⟩) (.authority (.programFamilyFact))

def exact10802RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11653⟩⟩], []⟩, (1)⟩]

theorem exact10802RawTermsValid :
    exact10802RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10802 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11653⟩⟩) exact10802RawTerms (.finite 28) 10801 .exactZero (none)

def event10803 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14677⟩⟩) 0 ⟨5560⟩ 10799

def event10804 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14677⟩⟩) (.authority (.programFamilyFact))

def exact10805RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14677⟩⟩], []⟩, (1)⟩]

theorem exact10805RawTermsValid :
    exact10805RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10805 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14677⟩⟩) exact10805RawTerms (.finite 28) 10804 .exactZero (none)

def event10806 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14678⟩⟩) 0 ⟨14677⟩ 10805

def event10807 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14678⟩⟩) 1 ⟨11653⟩ 10802

def event10808 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14678⟩⟩) (.product (.predecessor 0 10806 .coefficient) (.predecessor 1 10807 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event10809 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14678⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11653⟩⟩, ⟨.program ⟨214⟩, ⟨14677⟩⟩], []⟩) [⟨.result 10805 .coefficient, true, some 1⟩, ⟨.result 10802 .coefficient, true, some 1⟩])

def event10810 : Event := .survivorFold (1) 10809

def exact10811RawTerms : List Term := []

theorem exact10811RawTermsValid :
    exact10811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10811 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14678⟩⟩) exact10811RawTerms (.finite 784) 10808 (.finite 784) (some (10809))

def event10812 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14679⟩⟩) 0 ⟨14678⟩ 10811

def event10813 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14679⟩⟩) (.identity (.predecessor 0 10812 .coefficient))

def event10814 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14679⟩⟩) (.finite 784)

def event10815 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16194⟩⟩) 0 ⟨14679⟩ 10814

def event10816 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16194⟩⟩) (.authority (.programFamilyFact))

def exact10817RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16194⟩⟩], []⟩, (1)⟩]

theorem exact10817RawTermsValid :
    exact10817RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10817 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16194⟩⟩) exact10817RawTerms (.finite 28) 10816 .exactZero (none)

def event10818 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16195⟩⟩) 0 ⟨16194⟩ 10817

def event10819 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16195⟩⟩) (.identity (.predecessor 0 10818 .coefficient))

def event10820 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16195⟩⟩) (.finite 28)

def event10821 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21704⟩⟩) 0 ⟨16195⟩ 10820

def event10822 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21704⟩⟩) (.authority (.relationPreimageSource ⟨48⟩))

def exact10823RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩, (1)⟩]

theorem exact10823RawTermsValid :
    exact10823RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10823 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21704⟩⟩) exact10823RawTerms (.finite 136065468) 10822 .exactZero (none)

def event10824 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact10825RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact10825RawTermsValid :
    exact10825RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10825 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact10825RawTerms .large 10824 .exactZero (none)

def event10826 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21705⟩⟩) 0 ⟨6⟩ 10825

def event10827 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21705⟩⟩) 1 ⟨21704⟩ 10823

def event10828 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21705⟩⟩) (.product (.predecessor 0 10826 .coefficient) (.predecessor 1 10827 .coefficient) (⟨false, false, none, none, none⟩))

def event10829 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21705⟩⟩, .operator (⟨10825, 0⟩, ⟨10823, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩, (1)⟩)

def exact10830RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩, (1)⟩]

theorem exact10830RawTermsValid :
    exact10830RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10830 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21705⟩⟩) exact10830RawTerms .large 10828 .exactZero (none)

def event10831 : Event := .preFoldPolynomial 10830 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩, (1)⟩] .exactZero none

def exact10832RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩, (1)⟩]

def event10832 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21705⟩⟩) 10831 exact10832RawTerms .large 10828 .exactZero (none)

def event10833 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28357⟩⟩)

def event10834 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event10835 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event10836 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.authority (.operator))

def event10837 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5245⟩⟩) (.finite 6)

def event10838 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event10839 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event10840 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event10841 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event10842 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 10841

def event10843 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 10839

def event10844 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 10842 .coefficient) (.value (.predecessor 1 10843 .coefficient)))

def event10845 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event10846 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 0 ⟨5503⟩ 10845

def event10847 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5516⟩⟩) 1 ⟨5245⟩ 10837

def event10848 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.sum [.predecessor 0 10846 .coefficient, .predecessor 1 10847 .coefficient])

def event10849 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5516⟩⟩) (.finite 223)

def event10850 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 0 ⟨5516⟩ 10849

def event10851 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5560⟩⟩) 1 ⟨961⟩ 10835

def event10852 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.identity (.predecessor 1 10851 .coefficient))

def event10853 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5560⟩⟩) (.finite 224)

def event10854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11653⟩⟩) 0 ⟨5560⟩ 10853

def event10855 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11653⟩⟩) (.authority (.programFamilyFact))

def exact10856RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11653⟩⟩], []⟩, (1)⟩]

theorem exact10856RawTermsValid :
    exact10856RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10856 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11653⟩⟩) exact10856RawTerms (.finite 28) 10855 .exactZero (none)

def event10857 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14677⟩⟩) 0 ⟨5560⟩ 10853

def event10858 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14677⟩⟩) (.authority (.programFamilyFact))

def exact10859RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14677⟩⟩], []⟩, (1)⟩]

theorem exact10859RawTermsValid :
    exact10859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10859 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14677⟩⟩) exact10859RawTerms (.finite 28) 10858 .exactZero (none)

def event10860 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14678⟩⟩) 0 ⟨14677⟩ 10859

def event10861 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14678⟩⟩) 1 ⟨11653⟩ 10856

def event10862 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14678⟩⟩) (.product (.predecessor 0 10860 .coefficient) (.predecessor 1 10861 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event10863 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14678⟩⟩, .operator (⟨10859, 0⟩, ⟨10856, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11653⟩⟩, ⟨.program ⟨214⟩, ⟨14677⟩⟩], []⟩, (1)⟩)

def exact10864RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11653⟩⟩, ⟨.program ⟨214⟩, ⟨14677⟩⟩], []⟩, (1)⟩]

theorem exact10864RawTermsValid :
    exact10864RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10864 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14678⟩⟩) exact10864RawTerms (.finite 784) 10862 .exactZero (none)

def event10865 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14679⟩⟩) 0 ⟨14678⟩ 10864

def event10866 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14679⟩⟩) (.identity (.predecessor 0 10865 .coefficient))

def event10867 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14679⟩⟩) (.finite 784)

def event10868 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16194⟩⟩) 0 ⟨14679⟩ 10867

def event10869 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16194⟩⟩) (.authority (.programFamilyFact))

def exact10870RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16194⟩⟩], []⟩, (1)⟩]

theorem exact10870RawTermsValid :
    exact10870RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10870 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16194⟩⟩) exact10870RawTerms (.finite 28) 10869 .exactZero (none)

def event10871 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16195⟩⟩) 0 ⟨16194⟩ 10870

def event10872 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16195⟩⟩) (.identity (.predecessor 0 10871 .coefficient))

def event10873 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16195⟩⟩) (.finite 28)

def event10874 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24298⟩⟩) 0 ⟨16195⟩ 10873

def event10875 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24298⟩⟩) (.authority (.programFamilyFact))

def event10876 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24298⟩⟩) (.finite 3720)

def event10877 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event10878 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24300⟩⟩) 0 ⟨6689⟩ 10877

def event10879 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24300⟩⟩) 1 ⟨24298⟩ 10876

def event10880 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24300⟩⟩) (.authority (.operator))

def exact10881RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24300⟩⟩]⟩, (1)⟩]

theorem exact10881RawTermsValid :
    exact10881RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10881 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24300⟩⟩) exact10881RawTerms .large 10880 .exactZero (none)

def event10882 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28352⟩⟩) 0 ⟨24300⟩ 10881

def event10883 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28352⟩⟩) (.authority (.operator))

def exact10884RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩, (1)⟩]

theorem exact10884RawTermsValid :
    exact10884RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10884 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28352⟩⟩) exact10884RawTerms (.finite 8192) 10883 .exactZero (none)

def event10885 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event10886 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event10887 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16234⟩⟩) 0 ⟨16195⟩ 10873

def event10888 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16234⟩⟩) 1 ⟨110⟩ 10886

def event10889 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16234⟩⟩) (.sum [.predecessor 0 10887 .coefficient, .predecessor 1 10888 .coefficient])

def event10890 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16234⟩⟩) (.finite 28)

def event10891 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16235⟩⟩) 0 ⟨16234⟩ 10890

def event10892 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16235⟩⟩) (.identity (.predecessor 0 10891 .coefficient))

def exact10893RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16194⟩⟩], []⟩, (1)⟩]

theorem exact10893RawTermsValid :
    exact10893RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10893 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16235⟩⟩) exact10893RawTerms (.finite 28) 10892 .exactZero (none)

def event10894 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact10895RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact10895RawTermsValid :
    exact10895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10895 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact10895RawTerms .large 10894 .exactZero (none)

def event10896 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16236⟩⟩) 0 ⟨6544⟩ 10895

def event10897 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16236⟩⟩) 1 ⟨16235⟩ 10893

def event10898 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16236⟩⟩) (.product (.predecessor 0 10896 .coefficient) (.predecessor 1 10897 .coefficient) (⟨false, false, none, none, none⟩))

def event10899 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16236⟩⟩, .operator (⟨10895, 0⟩, ⟨10893, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact10900RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact10900RawTermsValid :
    exact10900RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10900 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16236⟩⟩) exact10900RawTerms .large 10898 .exactZero (none)

def event10901 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6699⟩⟩) 0 ⟨6689⟩ 10877

def event10902 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6699⟩⟩) (.authority (.operator))

def exact10903RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩]

theorem exact10903RawTermsValid :
    exact10903RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10903 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6699⟩⟩) exact10903RawTerms .large 10902 .exactZero (none)

def event10904 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16237⟩⟩) 0 ⟨6699⟩ 10903

def event10905 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16237⟩⟩) 1 ⟨16236⟩ 10900

def event10906 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16237⟩⟩) (.sum [.predecessor 0 10904 .coefficient, .predecessor 1 10905 .coefficient])

def exact10907RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact10907RawTermsValid :
    exact10907RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10907 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16237⟩⟩) exact10907RawTerms .large 10906 .exactZero (none)

def event10908 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28353⟩⟩) 0 ⟨16237⟩ 10907

def event10909 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28353⟩⟩) 1 ⟨28352⟩ 10884

def event10910 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28353⟩⟩) (.product (.predecessor 0 10908 .coefficient) (.predecessor 1 10909 .coefficient) (⟨false, false, none, none, none⟩))

def event10911 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28353⟩⟩, .operator (⟨10907, 1⟩, ⟨10884, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩, (-1)⟩)

def event10912 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28353⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28352⟩⟩) ⟨24300⟩ 10881)

def event10913 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28353⟩⟩, .relation 10912 0, ⟨[⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨24300⟩⟩]⟩, (-1)⟩)

def event10914 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28353⟩⟩, .operator (⟨10907, 0⟩, ⟨10884, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩, (1)⟩)

def exact10915RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨24300⟩⟩]⟩, (-1)⟩]

theorem exact10915RawTermsValid :
    exact10915RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10915 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28353⟩⟩) exact10915RawTerms .large 10910 .exactZero (none)

def event10916 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18392⟩⟩) 0 ⟨16195⟩ 10873

def event10917 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18392⟩⟩) (.authority (.programFamilyFact))

def exact10918RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18392⟩⟩], []⟩, (1)⟩]

theorem exact10918RawTermsValid :
    exact10918RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10918 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18392⟩⟩) exact10918RawTerms (.finite 62) 10917 .exactZero (none)

def event10919 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18403⟩⟩) 0 ⟨6544⟩ 10895

def event10920 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18403⟩⟩) 1 ⟨18392⟩ 10918

def event10921 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18403⟩⟩) (.product (.predecessor 0 10919 .coefficient) (.predecessor 1 10920 .coefficient) (⟨false, true, none, none, some 1⟩))

def event10922 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18403⟩⟩, .operator (⟨10895, 0⟩, ⟨10918, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact10923RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact10923RawTermsValid :
    exact10923RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10923 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18403⟩⟩) exact10923RawTerms .large 10921 .exactZero (none)

def event10924 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6727⟩⟩) 0 ⟨6689⟩ 10877

def event10925 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6727⟩⟩) (.authority (.operator))

def exact10926RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩]

theorem exact10926RawTermsValid :
    exact10926RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10926 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6727⟩⟩) exact10926RawTerms .large 10925 .exactZero (none)

def event10927 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18404⟩⟩) 0 ⟨6727⟩ 10926

def event10928 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18404⟩⟩) 1 ⟨18403⟩ 10923

def event10929 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18404⟩⟩) (.sum [.predecessor 0 10927 .coefficient, .predecessor 1 10928 .coefficient])

def exact10930RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact10930RawTermsValid :
    exact10930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10930 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18404⟩⟩) exact10930RawTerms .large 10929 .exactZero (none)

def event10931 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28357⟩⟩) 0 ⟨18404⟩ 10930

def event10932 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28357⟩⟩) 1 ⟨28353⟩ 10915

def event10933 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28357⟩⟩) (.sum [.predecessor 0 10931 .coefficient, .predecessor 1 10932 .coefficient])

def exact10934RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨24300⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact10934RawTermsValid :
    exact10934RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10934 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28357⟩⟩) exact10934RawTerms .large 10933 .exactZero (none)

def event10935 : Event := .preFoldPolynomial 10934 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨24300⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact10936RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨24300⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event10936 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28357⟩⟩) 10935 exact10936RawTerms .large 10933 .exactZero (none)

def event10937 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16195⟩⟩) ⟨⟨140⟩, ⟨48⟩, ⟨109⟩⟩ ⟨10779, 10937⟩

def event10938 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21707⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩) (1) 0 2 (.universal 10937 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21704⟩⟩]⟩) (none) 10936)

def event10939 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21707⟩⟩, .relation 10938 2, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨24300⟩⟩]⟩, (1)⟩)

def event10940 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21707⟩⟩, .relation 10938 0, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩, (-1)⟩)

def event10941 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21707⟩⟩, .relation 10938 3, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event10942 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21707⟩⟩, .relation 10938 1, ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩)

def exact10943RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨24300⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact10943RawTermsValid :
    exact10943RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10943 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21707⟩⟩) exact10943RawTerms .large 10775 (.finite 1811303510016) (some (10777))

def event10944 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28355⟩⟩) 0 ⟨21707⟩ 10943

def event10945 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28355⟩⟩) 1 ⟨28354⟩ 10765

def event10946 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28355⟩⟩) (.sum [.predecessor 0 10944 .coefficient, .predecessor 1 10945 .coefficient])

def event10947 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28355⟩⟩, .operator (⟨10943, 2⟩, ⟨10765, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16194⟩⟩], [⟨.program ⟨214⟩, ⟨24300⟩⟩]⟩, (-1)⟩)

def event10948 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28355⟩⟩, .operator (⟨10943, 0⟩, ⟨10765, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28352⟩⟩]⟩, (1)⟩)

def event10949 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28355⟩⟩) (.sum [.result 10943 .summary, .result 10765 .summary])

def exact10950RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨18392⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact10950RawTermsValid :
    exact10950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10950 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28355⟩⟩) exact10950RawTerms .large 10946 (.finite 1292180536164689260544) (some (10949))

def event10951 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24235⟩⟩) 0 ⟨16076⟩ 275

def event10952 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24235⟩⟩) (.authority (.programFamilyFact))

def event10953 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24235⟩⟩) (.finite 3720)

def event10954 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24237⟩⟩) 0 ⟨6689⟩ 5477

def event10955 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24237⟩⟩) 1 ⟨24235⟩ 10953

def event10956 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24237⟩⟩) (.authority (.operator))

def exact10957RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24237⟩⟩]⟩, (1)⟩]

theorem exact10957RawTermsValid :
    exact10957RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10957 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24237⟩⟩) exact10957RawTerms .large 10956 .exactZero (none)

def event10958 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28135⟩⟩) 0 ⟨24237⟩ 10957

def event10959 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28135⟩⟩) (.authority (.operator))

def exact10960RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28135⟩⟩]⟩, (1)⟩]

theorem exact10960RawTermsValid :
    exact10960RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10960 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28135⟩⟩) exact10960RawTerms (.finite 8192) 10959 .exactZero (none)

def event10961 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23633⟩⟩) 0 ⟨14462⟩ 269

def event10962 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23633⟩⟩) (.authority (.programFamilyFact))

def event10963 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23633⟩⟩) (.finite 3720)

def event10964 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23634⟩⟩) 0 ⟨6689⟩ 5477

def event10965 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23634⟩⟩) 1 ⟨23633⟩ 10963

def event10966 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23634⟩⟩) (.authority (.operator))

def exact10967RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23634⟩⟩]⟩, (1)⟩]

theorem exact10967RawTermsValid :
    exact10967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10967 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23634⟩⟩) exact10967RawTerms .large 10966 .exactZero (none)

def event10968 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26163⟩⟩) 0 ⟨23634⟩ 10967

def event10969 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26163⟩⟩) (.authority (.operator))

def exact10970RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26163⟩⟩]⟩, (1)⟩]

theorem exact10970RawTermsValid :
    exact10970RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10970 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26163⟩⟩) exact10970RawTerms (.finite 8192) 10969 .exactZero (none)

def event10971 : Event := .predecessor (⟨.program ⟨214⟩, ⟨94⟩⟩) 0 ⟨11⟩ 6441

def event10972 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨94⟩⟩) (.identity (.predecessor 0 10971 .coefficient))

def exact10973RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨94⟩⟩]⟩, (1)⟩]

theorem exact10973RawTermsValid :
    exact10973RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10973 : Event := .resultExact (⟨.program ⟨214⟩, ⟨94⟩⟩) exact10973RawTerms (.finite 26) 10972 .exactZero (none)

def event10974 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11570⟩⟩) 0 ⟨11569⟩ 258

def event10975 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11570⟩⟩) 1 ⟨6571⟩ 6449

def event10976 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11570⟩⟩) (.tensor (.predecessor 0 10974 .coefficient) (.predecessor 1 10975 .coefficient) true false)

def event10977 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11570⟩⟩, .operator (⟨258, 0⟩, ⟨6449, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11569⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact10978RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11569⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact10978RawTermsValid :
    exact10978RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10978 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11570⟩⟩) exact10978RawTerms .large 10976 .exactZero (none)

def event10979 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6780⟩⟩) 0 ⟨6757⟩ 5870

def event10980 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6780⟩⟩) (.identity (.predecessor 0 10979 .coefficient))

def exact10981RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩]

theorem exact10981RawTermsValid :
    exact10981RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10981 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6780⟩⟩) exact10981RawTerms .large 10980 .exactZero (none)

def event10982 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7388⟩⟩) 0 ⟨5563⟩ 6314

def event10983 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7388⟩⟩) 1 ⟨6780⟩ 10981

def event10984 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7388⟩⟩) (.product (.predecessor 0 10982 .coefficient) (.predecessor 1 10983 .coefficient) (⟨false, false, none, none, none⟩))

def event10985 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7388⟩⟩, .operator (⟨6314, 0⟩, ⟨10981, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩)

def exact10986RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩]

theorem exact10986RawTermsValid :
    exact10986RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10986 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7388⟩⟩) exact10986RawTerms .large 10984 .exactZero (none)

def event10987 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11571⟩⟩) 0 ⟨7388⟩ 10986

def event10988 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11571⟩⟩) 1 ⟨11570⟩ 10978

def event10989 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11571⟩⟩) (.sum [.predecessor 0 10987 .coefficient, .predecessor 1 10988 .coefficient])

def exact10990RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11569⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact10990RawTermsValid :
    exact10990RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10990 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11571⟩⟩) exact10990RawTerms .large 10989 .exactZero (none)

def event10991 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11572⟩⟩) 0 ⟨11571⟩ 10990

def event10992 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11572⟩⟩) 1 ⟨94⟩ 10973

def event10993 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11572⟩⟩) (.sum [.predecessor 0 10991 .coefficient, .predecessor 1 10992 .coefficient])

def event10994 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11572⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨94⟩⟩]⟩) [⟨.result 10973 .coefficient, false, none⟩])

def event10995 : Event := .survivorFold (1) 10994

def exact10996RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11569⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact10996RawTermsValid :
    exact10996RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event10996 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11572⟩⟩) exact10996RawTerms .large 10993 (.finite 26) (some (10994))

def event10997 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14463⟩⟩) 0 ⟨11572⟩ 10996

def event10998 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14463⟩⟩) 1 ⟨14460⟩ 261

def event10999 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14463⟩⟩) (.product (.predecessor 0 10997 .coefficient) (.predecessor 1 10998 .coefficient) (⟨false, true, none, none, some 1⟩))

def event11000 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨14460⟩⟩], []⟩) [⟨.result 261 .coefficient, true, some 1⟩])

def event11001 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14463⟩⟩) (.product (.result 10996 .summary) (.transfer 11000) (⟨false, false, none, none, none⟩))

def event11002 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14463⟩⟩, .operator (⟨10996, 1⟩, ⟨261, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11569⟩⟩, ⟨.program ⟨214⟩, ⟨14460⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event11003 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14463⟩⟩, .operator (⟨10996, 0⟩, ⟨261, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14460⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩)

def exact11004RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11569⟩⟩, ⟨.program ⟨214⟩, ⟨14460⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14460⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩]

theorem exact11004RawTermsValid :
    exact11004RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event11004 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14463⟩⟩) exact11004RawTerms .large 10999 (.finite 18304) (some (11001))

def event11005 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7855⟩⟩) 0 ⟨6780⟩ 10981

def event11006 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7855⟩⟩) (.authority (.operator))

def exact11007RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩, (1)⟩]

theorem exact11007RawTermsValid :
    exact11007RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event11007 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7855⟩⟩) exact11007RawTerms (.finite 8192) 11006 .exactZero (none)

def eventLeaf672 : Array AnnotatedEvent := #[
  { event := event10752
    frameStart := 0 },
  { event := event10753
    frameStart := 0 },
  { event := event10754
    frameStart := 0 },
  { event := event10755
    frameStart := 0 },
  { event := event10756
    frameStart := 0 },
  { event := event10757
    frameStart := 0 },
  { event := event10758
    frameStart := 0 },
  { event := event10759
    frameStart := 0 },
  { event := event10760
    frameStart := 0 },
  { event := event10761
    frameStart := 0 },
  { event := event10762
    frameStart := 0 },
  { event := event10763
    frameStart := 0 },
  { event := event10764
    frameStart := 0 },
  { event := event10765
    frameStart := 0 },
  { event := event10766
    frameStart := 0 },
  { event := event10767
    frameStart := 0 }
]

def eventLeaf673 : Array AnnotatedEvent := #[
  { event := event10768
    frameStart := 0 },
  { event := event10769
    frameStart := 0 },
  { event := event10770
    frameStart := 0 },
  { event := event10771
    frameStart := 0 },
  { event := event10772
    frameStart := 0 },
  { event := event10773
    frameStart := 0 },
  { event := event10774
    frameStart := 0 },
  { event := event10775
    frameStart := 0 },
  { event := event10776
    frameStart := 0 },
  { event := event10777
    frameStart := 0 },
  { event := event10778
    frameStart := 0 },
  { event := event10779
    frameStart := 10779 },
  { event := event10780
    frameStart := 10779 },
  { event := event10781
    frameStart := 10779 },
  { event := event10782
    frameStart := 10779 },
  { event := event10783
    frameStart := 10779 }
]

def eventLeaf674 : Array AnnotatedEvent := #[
  { event := event10784
    frameStart := 10779 },
  { event := event10785
    frameStart := 10779 },
  { event := event10786
    frameStart := 10779 },
  { event := event10787
    frameStart := 10779 },
  { event := event10788
    frameStart := 10779 },
  { event := event10789
    frameStart := 10779 },
  { event := event10790
    frameStart := 10779 },
  { event := event10791
    frameStart := 10779 },
  { event := event10792
    frameStart := 10779 },
  { event := event10793
    frameStart := 10779 },
  { event := event10794
    frameStart := 10779 },
  { event := event10795
    frameStart := 10779 },
  { event := event10796
    frameStart := 10779 },
  { event := event10797
    frameStart := 10779 },
  { event := event10798
    frameStart := 10779 },
  { event := event10799
    frameStart := 10779 }
]

def eventLeaf675 : Array AnnotatedEvent := #[
  { event := event10800
    frameStart := 10779 },
  { event := event10801
    frameStart := 10779 },
  { event := event10802
    frameStart := 10779 },
  { event := event10803
    frameStart := 10779 },
  { event := event10804
    frameStart := 10779 },
  { event := event10805
    frameStart := 10779 },
  { event := event10806
    frameStart := 10779 },
  { event := event10807
    frameStart := 10779 },
  { event := event10808
    frameStart := 10779 },
  { event := event10809
    frameStart := 10779 },
  { event := event10810
    frameStart := 10779 },
  { event := event10811
    frameStart := 10779 },
  { event := event10812
    frameStart := 10779 },
  { event := event10813
    frameStart := 10779 },
  { event := event10814
    frameStart := 10779 },
  { event := event10815
    frameStart := 10779 }
]

def eventLeaf676 : Array AnnotatedEvent := #[
  { event := event10816
    frameStart := 10779 },
  { event := event10817
    frameStart := 10779 },
  { event := event10818
    frameStart := 10779 },
  { event := event10819
    frameStart := 10779 },
  { event := event10820
    frameStart := 10779 },
  { event := event10821
    frameStart := 10779 },
  { event := event10822
    frameStart := 10779 },
  { event := event10823
    frameStart := 10779 },
  { event := event10824
    frameStart := 10779 },
  { event := event10825
    frameStart := 10779 },
  { event := event10826
    frameStart := 10779 },
  { event := event10827
    frameStart := 10779 },
  { event := event10828
    frameStart := 10779 },
  { event := event10829
    frameStart := 10779 },
  { event := event10830
    frameStart := 10779 },
  { event := event10831
    frameStart := 10779 }
]

def eventLeaf677 : Array AnnotatedEvent := #[
  { event := event10832
    frameStart := 10779 },
  { event := event10833
    frameStart := 10833 },
  { event := event10834
    frameStart := 10833 },
  { event := event10835
    frameStart := 10833 },
  { event := event10836
    frameStart := 10833 },
  { event := event10837
    frameStart := 10833 },
  { event := event10838
    frameStart := 10833 },
  { event := event10839
    frameStart := 10833 },
  { event := event10840
    frameStart := 10833 },
  { event := event10841
    frameStart := 10833 },
  { event := event10842
    frameStart := 10833 },
  { event := event10843
    frameStart := 10833 },
  { event := event10844
    frameStart := 10833 },
  { event := event10845
    frameStart := 10833 },
  { event := event10846
    frameStart := 10833 },
  { event := event10847
    frameStart := 10833 }
]

def eventLeaf678 : Array AnnotatedEvent := #[
  { event := event10848
    frameStart := 10833 },
  { event := event10849
    frameStart := 10833 },
  { event := event10850
    frameStart := 10833 },
  { event := event10851
    frameStart := 10833 },
  { event := event10852
    frameStart := 10833 },
  { event := event10853
    frameStart := 10833 },
  { event := event10854
    frameStart := 10833 },
  { event := event10855
    frameStart := 10833 },
  { event := event10856
    frameStart := 10833 },
  { event := event10857
    frameStart := 10833 },
  { event := event10858
    frameStart := 10833 },
  { event := event10859
    frameStart := 10833 },
  { event := event10860
    frameStart := 10833 },
  { event := event10861
    frameStart := 10833 },
  { event := event10862
    frameStart := 10833 },
  { event := event10863
    frameStart := 10833 }
]

def eventLeaf679 : Array AnnotatedEvent := #[
  { event := event10864
    frameStart := 10833 },
  { event := event10865
    frameStart := 10833 },
  { event := event10866
    frameStart := 10833 },
  { event := event10867
    frameStart := 10833 },
  { event := event10868
    frameStart := 10833 },
  { event := event10869
    frameStart := 10833 },
  { event := event10870
    frameStart := 10833 },
  { event := event10871
    frameStart := 10833 },
  { event := event10872
    frameStart := 10833 },
  { event := event10873
    frameStart := 10833 },
  { event := event10874
    frameStart := 10833 },
  { event := event10875
    frameStart := 10833 },
  { event := event10876
    frameStart := 10833 },
  { event := event10877
    frameStart := 10833 },
  { event := event10878
    frameStart := 10833 },
  { event := event10879
    frameStart := 10833 }
]

def eventLeaf680 : Array AnnotatedEvent := #[
  { event := event10880
    frameStart := 10833 },
  { event := event10881
    frameStart := 10833 },
  { event := event10882
    frameStart := 10833 },
  { event := event10883
    frameStart := 10833 },
  { event := event10884
    frameStart := 10833 },
  { event := event10885
    frameStart := 10833 },
  { event := event10886
    frameStart := 10833 },
  { event := event10887
    frameStart := 10833 },
  { event := event10888
    frameStart := 10833 },
  { event := event10889
    frameStart := 10833 },
  { event := event10890
    frameStart := 10833 },
  { event := event10891
    frameStart := 10833 },
  { event := event10892
    frameStart := 10833 },
  { event := event10893
    frameStart := 10833 },
  { event := event10894
    frameStart := 10833 },
  { event := event10895
    frameStart := 10833 }
]

def eventLeaf681 : Array AnnotatedEvent := #[
  { event := event10896
    frameStart := 10833 },
  { event := event10897
    frameStart := 10833 },
  { event := event10898
    frameStart := 10833 },
  { event := event10899
    frameStart := 10833 },
  { event := event10900
    frameStart := 10833 },
  { event := event10901
    frameStart := 10833 },
  { event := event10902
    frameStart := 10833 },
  { event := event10903
    frameStart := 10833 },
  { event := event10904
    frameStart := 10833 },
  { event := event10905
    frameStart := 10833 },
  { event := event10906
    frameStart := 10833 },
  { event := event10907
    frameStart := 10833 },
  { event := event10908
    frameStart := 10833 },
  { event := event10909
    frameStart := 10833 },
  { event := event10910
    frameStart := 10833 },
  { event := event10911
    frameStart := 10833 }
]

def eventLeaf682 : Array AnnotatedEvent := #[
  { event := event10912
    frameStart := 10833 },
  { event := event10913
    frameStart := 10833 },
  { event := event10914
    frameStart := 10833 },
  { event := event10915
    frameStart := 10833 },
  { event := event10916
    frameStart := 10833 },
  { event := event10917
    frameStart := 10833 },
  { event := event10918
    frameStart := 10833 },
  { event := event10919
    frameStart := 10833 },
  { event := event10920
    frameStart := 10833 },
  { event := event10921
    frameStart := 10833 },
  { event := event10922
    frameStart := 10833 },
  { event := event10923
    frameStart := 10833 },
  { event := event10924
    frameStart := 10833 },
  { event := event10925
    frameStart := 10833 },
  { event := event10926
    frameStart := 10833 },
  { event := event10927
    frameStart := 10833 }
]

def eventLeaf683 : Array AnnotatedEvent := #[
  { event := event10928
    frameStart := 10833 },
  { event := event10929
    frameStart := 10833 },
  { event := event10930
    frameStart := 10833 },
  { event := event10931
    frameStart := 10833 },
  { event := event10932
    frameStart := 10833 },
  { event := event10933
    frameStart := 10833 },
  { event := event10934
    frameStart := 10833 },
  { event := event10935
    frameStart := 10833 },
  { event := event10936
    frameStart := 10833 },
  { event := event10937
    frameStart := 0 },
  { event := event10938
    frameStart := 0 },
  { event := event10939
    frameStart := 0 },
  { event := event10940
    frameStart := 0 },
  { event := event10941
    frameStart := 0 },
  { event := event10942
    frameStart := 0 },
  { event := event10943
    frameStart := 0 }
]

def eventLeaf684 : Array AnnotatedEvent := #[
  { event := event10944
    frameStart := 0 },
  { event := event10945
    frameStart := 0 },
  { event := event10946
    frameStart := 0 },
  { event := event10947
    frameStart := 0 },
  { event := event10948
    frameStart := 0 },
  { event := event10949
    frameStart := 0 },
  { event := event10950
    frameStart := 0 },
  { event := event10951
    frameStart := 0 },
  { event := event10952
    frameStart := 0 },
  { event := event10953
    frameStart := 0 },
  { event := event10954
    frameStart := 0 },
  { event := event10955
    frameStart := 0 },
  { event := event10956
    frameStart := 0 },
  { event := event10957
    frameStart := 0 },
  { event := event10958
    frameStart := 0 },
  { event := event10959
    frameStart := 0 }
]

def eventLeaf685 : Array AnnotatedEvent := #[
  { event := event10960
    frameStart := 0 },
  { event := event10961
    frameStart := 0 },
  { event := event10962
    frameStart := 0 },
  { event := event10963
    frameStart := 0 },
  { event := event10964
    frameStart := 0 },
  { event := event10965
    frameStart := 0 },
  { event := event10966
    frameStart := 0 },
  { event := event10967
    frameStart := 0 },
  { event := event10968
    frameStart := 0 },
  { event := event10969
    frameStart := 0 },
  { event := event10970
    frameStart := 0 },
  { event := event10971
    frameStart := 0 },
  { event := event10972
    frameStart := 0 },
  { event := event10973
    frameStart := 0 },
  { event := event10974
    frameStart := 0 },
  { event := event10975
    frameStart := 0 }
]

def eventLeaf686 : Array AnnotatedEvent := #[
  { event := event10976
    frameStart := 0 },
  { event := event10977
    frameStart := 0 },
  { event := event10978
    frameStart := 0 },
  { event := event10979
    frameStart := 0 },
  { event := event10980
    frameStart := 0 },
  { event := event10981
    frameStart := 0 },
  { event := event10982
    frameStart := 0 },
  { event := event10983
    frameStart := 0 },
  { event := event10984
    frameStart := 0 },
  { event := event10985
    frameStart := 0 },
  { event := event10986
    frameStart := 0 },
  { event := event10987
    frameStart := 0 },
  { event := event10988
    frameStart := 0 },
  { event := event10989
    frameStart := 0 },
  { event := event10990
    frameStart := 0 },
  { event := event10991
    frameStart := 0 }
]

def eventLeaf687 : Array AnnotatedEvent := #[
  { event := event10992
    frameStart := 0 },
  { event := event10993
    frameStart := 0 },
  { event := event10994
    frameStart := 0 },
  { event := event10995
    frameStart := 0 },
  { event := event10996
    frameStart := 0 },
  { event := event10997
    frameStart := 0 },
  { event := event10998
    frameStart := 0 },
  { event := event10999
    frameStart := 0 },
  { event := event11000
    frameStart := 0 },
  { event := event11001
    frameStart := 0 },
  { event := event11002
    frameStart := 0 },
  { event := event11003
    frameStart := 0 },
  { event := event11004
    frameStart := 0 },
  { event := event11005
    frameStart := 0 },
  { event := event11006
    frameStart := 0 },
  { event := event11007
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events042
