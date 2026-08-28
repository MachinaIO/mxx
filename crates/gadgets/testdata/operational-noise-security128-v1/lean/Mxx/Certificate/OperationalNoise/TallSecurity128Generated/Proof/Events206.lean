import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events206

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event52736 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54899⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨54896⟩⟩]⟩) [⟨.result 52728 .coefficient, false, none⟩])

def event52737 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54899⟩⟩) (.product (.result 46745 .summary) (.transfer 52736) (⟨false, false, none, none, none⟩))

def event52738 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54899⟩⟩, .operator (⟨46745, 0⟩, ⟨52732, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54896⟩⟩]⟩, (1)⟩)

def event52739 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨54897⟩⟩)

def event52740 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event52741 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event52742 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event52743 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event52744 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event52745 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event52746 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event52747 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event52748 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 52747

def event52749 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 52745

def event52750 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 52748 .coefficient) (.value (.predecessor 1 52749 .coefficient)))

def event52751 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event52752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 52751

def event52753 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 52743

def event52754 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 52752 .coefficient, .predecessor 1 52753 .coefficient])

def event52755 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event52756 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 52755

def event52757 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 52741

def event52758 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 52757 .coefficient))

def event52759 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event52760 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24866⟩⟩) 0 ⟨11173⟩ 52759

def event52761 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24866⟩⟩) (.authority (.programFamilyFact))

def exact52762RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24866⟩⟩], []⟩, (1)⟩]

theorem exact52762RawTermsValid :
    exact52762RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52762 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24866⟩⟩) exact52762RawTerms (.finite 12) 52761 .exactZero (none)

def event52763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53741⟩⟩) 0 ⟨11173⟩ 52759

def event52764 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53741⟩⟩) (.authority (.programFamilyFact))

def exact52765RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53741⟩⟩], []⟩, (1)⟩]

theorem exact52765RawTermsValid :
    exact52765RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52765 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53741⟩⟩) exact52765RawTerms (.finite 12) 52764 .exactZero (none)

def event52766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53742⟩⟩) 0 ⟨53741⟩ 52765

def event52767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53742⟩⟩) 1 ⟨24866⟩ 52762

def event52768 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53742⟩⟩) (.product (.predecessor 0 52766 .coefficient) (.predecessor 1 52767 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event52769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53742⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24866⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], []⟩) [⟨.result 52765 .coefficient, true, some 1⟩, ⟨.result 52762 .coefficient, true, some 1⟩])

def event52770 : Event := .survivorFold (1) 52769

def exact52771RawTerms : List Term := []

theorem exact52771RawTermsValid :
    exact52771RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52771 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53742⟩⟩) exact52771RawTerms (.finite 144) 52768 (.finite 144) (some (52769))

def event52772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53743⟩⟩) 0 ⟨53742⟩ 52771

def event52773 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53743⟩⟩) (.identity (.predecessor 0 52772 .coefficient))

def event52774 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53743⟩⟩) (.finite 144)

def event52775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53932⟩⟩) 0 ⟨53743⟩ 52774

def event52776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53932⟩⟩) (.authority (.programFamilyFact))

def exact52777RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53932⟩⟩], []⟩, (1)⟩]

theorem exact52777RawTermsValid :
    exact52777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52777 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53932⟩⟩) exact52777RawTerms (.finite 12) 52776 .exactZero (none)

def event52778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53933⟩⟩) 0 ⟨53932⟩ 52777

def event52779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53933⟩⟩) (.identity (.predecessor 0 52778 .coefficient))

def event52780 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53933⟩⟩) (.finite 12)

def event52781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54896⟩⟩) 0 ⟨53933⟩ 52780

def event52782 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54896⟩⟩) (.authority (.relationPreimageSource ⟨68⟩))

def exact52783RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54896⟩⟩]⟩, (1)⟩]

theorem exact52783RawTermsValid :
    exact52783RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52783 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54896⟩⟩) exact52783RawTerms (.finite 5647228698) 52782 .exactZero (none)

def event52784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact52785RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact52785RawTermsValid :
    exact52785RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52785 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact52785RawTerms .large 52784 .exactZero (none)

def event52786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54897⟩⟩) 0 ⟨35⟩ 52785

def event52787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54897⟩⟩) 1 ⟨54896⟩ 52783

def event52788 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54897⟩⟩) (.product (.predecessor 0 52786 .coefficient) (.predecessor 1 52787 .coefficient) (⟨false, false, none, none, none⟩))

def event52789 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54897⟩⟩, .operator (⟨52785, 0⟩, ⟨52783, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54896⟩⟩]⟩, (1)⟩)

def exact52790RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54896⟩⟩]⟩, (1)⟩]

theorem exact52790RawTermsValid :
    exact52790RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52790 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54897⟩⟩) exact52790RawTerms .large 52788 .exactZero (none)

def event52791 : Event := .preFoldPolynomial 52790 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54896⟩⟩]⟩, (1)⟩] .exactZero none

def exact52792RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54896⟩⟩]⟩, (1)⟩]

def event52792 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨54897⟩⟩) 52791 exact52792RawTerms .large 52788 .exactZero (none)

def event52793 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨56185⟩⟩)

def event52794 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event52795 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event52796 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event52797 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event52798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event52799 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event52800 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event52801 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event52802 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 52801

def event52803 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 52799

def event52804 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 52802 .coefficient) (.value (.predecessor 1 52803 .coefficient)))

def event52805 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event52806 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 52805

def event52807 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 52797

def event52808 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 52806 .coefficient, .predecessor 1 52807 .coefficient])

def event52809 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event52810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 52809

def event52811 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 52795

def event52812 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 52811 .coefficient))

def event52813 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event52814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24866⟩⟩) 0 ⟨11173⟩ 52813

def event52815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24866⟩⟩) (.authority (.programFamilyFact))

def exact52816RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24866⟩⟩], []⟩, (1)⟩]

theorem exact52816RawTermsValid :
    exact52816RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52816 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24866⟩⟩) exact52816RawTerms (.finite 12) 52815 .exactZero (none)

def event52817 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53741⟩⟩) 0 ⟨11173⟩ 52813

def event52818 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53741⟩⟩) (.authority (.programFamilyFact))

def exact52819RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53741⟩⟩], []⟩, (1)⟩]

theorem exact52819RawTermsValid :
    exact52819RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52819 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53741⟩⟩) exact52819RawTerms (.finite 12) 52818 .exactZero (none)

def event52820 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53742⟩⟩) 0 ⟨53741⟩ 52819

def event52821 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53742⟩⟩) 1 ⟨24866⟩ 52816

def event52822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53742⟩⟩) (.product (.predecessor 0 52820 .coefficient) (.predecessor 1 52821 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event52823 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53742⟩⟩, .operator (⟨52819, 0⟩, ⟨52816, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24866⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], []⟩, (1)⟩)

def exact52824RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24866⟩⟩, ⟨.program ⟨257⟩, ⟨53741⟩⟩], []⟩, (1)⟩]

theorem exact52824RawTermsValid :
    exact52824RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52824 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53742⟩⟩) exact52824RawTerms (.finite 144) 52822 .exactZero (none)

def event52825 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53743⟩⟩) 0 ⟨53742⟩ 52824

def event52826 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53743⟩⟩) (.identity (.predecessor 0 52825 .coefficient))

def event52827 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53743⟩⟩) (.finite 144)

def event52828 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53932⟩⟩) 0 ⟨53743⟩ 52827

def event52829 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53932⟩⟩) (.authority (.programFamilyFact))

def exact52830RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53932⟩⟩], []⟩, (1)⟩]

theorem exact52830RawTermsValid :
    exact52830RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52830 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53932⟩⟩) exact52830RawTerms (.finite 12) 52829 .exactZero (none)

def event52831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53933⟩⟩) 0 ⟨53932⟩ 52830

def event52832 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53933⟩⟩) (.identity (.predecessor 0 52831 .coefficient))

def event52833 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53933⟩⟩) (.finite 12)

def event52834 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55211⟩⟩) 0 ⟨53933⟩ 52833

def event52835 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55211⟩⟩) (.authority (.programFamilyFact))

def event52836 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨55211⟩⟩) (.finite 3720)

def event52837 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event52838 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55213⟩⟩) 0 ⟨7177⟩ 52837

def event52839 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55213⟩⟩) 1 ⟨55211⟩ 52836

def event52840 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55213⟩⟩) (.authority (.operator))

def exact52841RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55213⟩⟩]⟩, (1)⟩]

theorem exact52841RawTermsValid :
    exact52841RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52841 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55213⟩⟩) exact52841RawTerms .large 52840 .exactZero (none)

def event52842 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56180⟩⟩) 0 ⟨55213⟩ 52841

def event52843 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56180⟩⟩) (.authority (.operator))

def exact52844RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨56180⟩⟩]⟩, (1)⟩]

theorem exact52844RawTermsValid :
    exact52844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52844 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56180⟩⟩) exact52844RawTerms (.finite 8192) 52843 .exactZero (none)

def event52845 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event52846 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event52847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55378⟩⟩) 0 ⟨53933⟩ 52833

def event52848 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55378⟩⟩) 1 ⟨136⟩ 52846

def event52849 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55378⟩⟩) (.sum [.predecessor 0 52847 .coefficient, .predecessor 1 52848 .coefficient])

def event52850 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨55378⟩⟩) (.finite 12)

def event52851 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55379⟩⟩) 0 ⟨55378⟩ 52850

def event52852 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55379⟩⟩) (.identity (.predecessor 0 52851 .coefficient))

def exact52853RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53932⟩⟩], []⟩, (1)⟩]

theorem exact52853RawTermsValid :
    exact52853RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52853 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55379⟩⟩) exact52853RawTerms (.finite 12) 52852 .exactZero (none)

def event52854 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact52855RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact52855RawTermsValid :
    exact52855RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52855 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact52855RawTerms .large 52854 .exactZero (none)

def event52856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55380⟩⟩) 0 ⟨6908⟩ 52855

def event52857 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55380⟩⟩) 1 ⟨55379⟩ 52853

def event52858 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55380⟩⟩) (.product (.predecessor 0 52856 .coefficient) (.predecessor 1 52857 .coefficient) (⟨false, false, none, none, none⟩))

def event52859 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55380⟩⟩, .operator (⟨52855, 0⟩, ⟨52853, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact52860RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact52860RawTermsValid :
    exact52860RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52860 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55380⟩⟩) exact52860RawTerms .large 52858 .exactZero (none)

def event52861 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7184⟩⟩) 0 ⟨7177⟩ 52837

def event52862 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7184⟩⟩) (.authority (.operator))

def exact52863RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩]

theorem exact52863RawTermsValid :
    exact52863RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52863 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7184⟩⟩) exact52863RawTerms .large 52862 .exactZero (none)

def event52864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55381⟩⟩) 0 ⟨7184⟩ 52863

def event52865 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55381⟩⟩) 1 ⟨55380⟩ 52860

def event52866 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55381⟩⟩) (.sum [.predecessor 0 52864 .coefficient, .predecessor 1 52865 .coefficient])

def exact52867RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact52867RawTermsValid :
    exact52867RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52867 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55381⟩⟩) exact52867RawTerms .large 52866 .exactZero (none)

def event52868 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56181⟩⟩) 0 ⟨55381⟩ 52867

def event52869 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56181⟩⟩) 1 ⟨56180⟩ 52844

def event52870 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56181⟩⟩) (.product (.predecessor 0 52868 .coefficient) (.predecessor 1 52869 .coefficient) (⟨false, false, none, none, none⟩))

def event52871 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56181⟩⟩, .operator (⟨52867, 0⟩, ⟨52844, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56180⟩⟩]⟩, (1)⟩)

def event52872 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56181⟩⟩, .operator (⟨52867, 1⟩, ⟨52844, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56180⟩⟩]⟩, (-1)⟩)

def event52873 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨56181⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56180⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨56180⟩⟩) ⟨55213⟩ 52841)

def event52874 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56181⟩⟩, .relation 52873 0, ⟨[⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨55213⟩⟩]⟩, (-1)⟩)

def exact52875RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56180⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨55213⟩⟩]⟩, (-1)⟩]

theorem exact52875RawTermsValid :
    exact52875RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52875 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56181⟩⟩) exact52875RawTerms .large 52870 .exactZero (none)

def event52876 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54293⟩⟩) 0 ⟨53933⟩ 52833

def event52877 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54293⟩⟩) (.authority (.programFamilyFact))

def exact52878RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨54293⟩⟩], []⟩, (1)⟩]

theorem exact52878RawTermsValid :
    exact52878RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52878 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54293⟩⟩) exact52878RawTerms (.finite 59) 52877 .exactZero (none)

def event52879 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54295⟩⟩) 0 ⟨6908⟩ 52855

def event52880 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54295⟩⟩) 1 ⟨54293⟩ 52878

def event52881 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54295⟩⟩) (.product (.predecessor 0 52879 .coefficient) (.predecessor 1 52880 .coefficient) (⟨false, true, none, none, some 1⟩))

def event52882 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54295⟩⟩, .operator (⟨52855, 0⟩, ⟨52878, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact52883RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact52883RawTermsValid :
    exact52883RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52883 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54295⟩⟩) exact52883RawTerms .large 52881 .exactZero (none)

def event52884 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7208⟩⟩) 0 ⟨7177⟩ 52837

def event52885 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7208⟩⟩) (.authority (.operator))

def exact52886RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩]

theorem exact52886RawTermsValid :
    exact52886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52886 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7208⟩⟩) exact52886RawTerms .large 52885 .exactZero (none)

def event52887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54296⟩⟩) 0 ⟨7208⟩ 52886

def event52888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54296⟩⟩) 1 ⟨54295⟩ 52883

def event52889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54296⟩⟩) (.sum [.predecessor 0 52887 .coefficient, .predecessor 1 52888 .coefficient])

def exact52890RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact52890RawTermsValid :
    exact52890RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52890 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54296⟩⟩) exact52890RawTerms .large 52889 .exactZero (none)

def event52891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56185⟩⟩) 0 ⟨54296⟩ 52890

def event52892 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56185⟩⟩) 1 ⟨56181⟩ 52875

def event52893 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56185⟩⟩) (.sum [.predecessor 0 52891 .coefficient, .predecessor 1 52892 .coefficient])

def exact52894RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56180⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨55213⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact52894RawTermsValid :
    exact52894RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52894 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56185⟩⟩) exact52894RawTerms .large 52893 .exactZero (none)

def event52895 : Event := .preFoldPolynomial 52894 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56180⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨55213⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact52896RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56180⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨55213⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event52896 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨56185⟩⟩) 52895 exact52896RawTerms .large 52893 .exactZero (none)

def event52897 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨53933⟩⟩) ⟨⟨87⟩, ⟨68⟩, ⟨135⟩⟩ ⟨52739, 52897⟩

def event52898 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨54899⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54896⟩⟩]⟩) (1) 0 2 (.universal 52897 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54896⟩⟩]⟩) (none) 52896)

def event52899 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54899⟩⟩, .relation 52898 1, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩)

def event52900 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54899⟩⟩, .relation 52898 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56180⟩⟩]⟩, (-1)⟩)

def event52901 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54899⟩⟩, .relation 52898 2, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨55213⟩⟩]⟩, (1)⟩)

def event52902 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54899⟩⟩, .relation 52898 3, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact52903RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56180⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨55213⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact52903RawTermsValid :
    exact52903RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52903 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54899⟩⟩) exact52903RawTerms .large 52735 (.finite 202072841853861888) (some (52737))

def event52904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56183⟩⟩) 0 ⟨54899⟩ 52903

def event52905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56183⟩⟩) 1 ⟨56182⟩ 52725

def event52906 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56183⟩⟩) (.sum [.predecessor 0 52904 .coefficient, .predecessor 1 52905 .coefficient])

def event52907 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56183⟩⟩, .operator (⟨52903, 0⟩, ⟨52725, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56180⟩⟩]⟩, (1)⟩)

def event52908 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56183⟩⟩, .operator (⟨52903, 2⟩, ⟨52725, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨53932⟩⟩], [⟨.program ⟨257⟩, ⟨55213⟩⟩]⟩, (-1)⟩)

def event52909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56183⟩⟩) (.sum [.result 52903 .summary, .result 52725 .summary])

def exact52910RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact52910RawTermsValid :
    exact52910RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52910 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56183⟩⟩) exact52910RawTerms .large 52906 (.finite 32189789464712143775715074244608) (some (52909))

def event52911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52231⟩⟩) 0 ⟨50953⟩ 1906

def event52912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52231⟩⟩) (.authority (.programFamilyFact))

def event52913 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52231⟩⟩) (.finite 3720)

def event52914 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52233⟩⟩) 0 ⟨7177⟩ 15500

def event52915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52233⟩⟩) 1 ⟨52231⟩ 52913

def event52916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52233⟩⟩) (.authority (.operator))

def exact52917RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52233⟩⟩]⟩, (1)⟩]

theorem exact52917RawTermsValid :
    exact52917RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52917 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52233⟩⟩) exact52917RawTerms .large 52916 .exactZero (none)

def event52918 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53200⟩⟩) 0 ⟨52233⟩ 52917

def event52919 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53200⟩⟩) (.authority (.operator))

def exact52920RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨53200⟩⟩]⟩, (1)⟩]

theorem exact52920RawTermsValid :
    exact52920RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52920 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53200⟩⟩) exact52920RawTerms (.finite 8192) 52919 .exactZero (none)

def event52921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52056⟩⟩) 0 ⟨50763⟩ 1900

def event52922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52056⟩⟩) (.authority (.programFamilyFact))

def event52923 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52056⟩⟩) (.finite 3720)

def event52924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52057⟩⟩) 0 ⟨7177⟩ 15500

def event52925 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52057⟩⟩) 1 ⟨52056⟩ 52923

def event52926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52057⟩⟩) (.authority (.operator))

def exact52927RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52057⟩⟩]⟩, (1)⟩]

theorem exact52927RawTermsValid :
    exact52927RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52927 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52057⟩⟩) exact52927RawTerms .large 52926 .exactZero (none)

def event52928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52607⟩⟩) 0 ⟨52057⟩ 52927

def event52929 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52607⟩⟩) (.authority (.operator))

def exact52930RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52607⟩⟩]⟩, (1)⟩]

theorem exact52930RawTermsValid :
    exact52930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52930 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52607⟩⟩) exact52930RawTerms (.finite 8192) 52929 .exactZero (none)

def event52931 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24627⟩⟩) 0 ⟨24626⟩ 1889

def event52932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24627⟩⟩) 1 ⟨11176⟩ 46653

def event52933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24627⟩⟩) (.tensor (.predecessor 0 52931 .coefficient) (.predecessor 1 52932 .coefficient) true false)

def event52934 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24627⟩⟩, .operator (⟨1889, 0⟩, ⟨46653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact52935RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact52935RawTermsValid :
    exact52935RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52935 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24627⟩⟩) exact52935RawTerms .large 52933 .exactZero (none)

def event52936 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11214⟩⟩) 0 ⟨11175⟩ 46523

def event52937 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11214⟩⟩) 1 ⟨7308⟩ 23593

def event52938 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11214⟩⟩) (.product (.predecessor 0 52936 .coefficient) (.predecessor 1 52937 .coefficient) (⟨false, false, none, none, none⟩))

def event52939 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11214⟩⟩, .operator (⟨46523, 0⟩, ⟨23593, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩)

def exact52940RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩]

theorem exact52940RawTermsValid :
    exact52940RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52940 : Event := .resultExact (⟨.program ⟨257⟩, ⟨11214⟩⟩) exact52940RawTerms .large 52938 .exactZero (none)

def event52941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24628⟩⟩) 0 ⟨11214⟩ 52940

def event52942 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24628⟩⟩) 1 ⟨24627⟩ 52935

def event52943 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24628⟩⟩) (.sum [.predecessor 0 52941 .coefficient, .predecessor 1 52942 .coefficient])

def exact52944RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact52944RawTermsValid :
    exact52944RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52944 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24628⟩⟩) exact52944RawTerms .large 52943 .exactZero (none)

def event52945 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24629⟩⟩) 0 ⟨24628⟩ 52944

def event52946 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24629⟩⟩) 1 ⟨134⟩ 23585

def event52947 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24629⟩⟩) (.sum [.predecessor 0 52945 .coefficient, .predecessor 1 52946 .coefficient])

def event52948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24629⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨134⟩⟩]⟩) [⟨.result 23585 .coefficient, false, none⟩])

def event52949 : Event := .survivorFold (1) 52948

def exact52950RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact52950RawTermsValid :
    exact52950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52950 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24629⟩⟩) exact52950RawTerms .large 52947 (.finite 26) (some (52948))

def event52951 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50764⟩⟩) 0 ⟨24629⟩ 52950

def event52952 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50764⟩⟩) 1 ⟨50761⟩ 1892

def event52953 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50764⟩⟩) (.product (.predecessor 0 52951 .coefficient) (.predecessor 1 52952 .coefficient) (⟨false, true, none, none, some 1⟩))

def event52954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50764⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨50761⟩⟩], []⟩) [⟨.result 1892 .coefficient, true, some 1⟩])

def event52955 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50764⟩⟩) (.product (.result 52950 .summary) (.transfer 52954) (⟨false, false, none, none, none⟩))

def event52956 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50764⟩⟩, .operator (⟨52950, 1⟩, ⟨1892, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event52957 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50764⟩⟩, .operator (⟨52950, 0⟩, ⟨1892, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩)

def exact52958RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨24626⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (1)⟩]

theorem exact52958RawTermsValid :
    exact52958RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52958 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50764⟩⟩) exact52958RawTerms .large 52953 (.finite 8519680) (some (52955))

def event52959 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50765⟩⟩) 0 ⟨50761⟩ 1892

def event52960 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50765⟩⟩) 1 ⟨11176⟩ 46653

def event52961 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50765⟩⟩) (.tensor (.predecessor 0 52959 .coefficient) (.predecessor 1 52960 .coefficient) true false)

def event52962 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50765⟩⟩, .operator (⟨1892, 0⟩, ⟨46653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact52963RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact52963RawTermsValid :
    exact52963RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52963 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50765⟩⟩) exact52963RawTerms .large 52961 .exactZero (none)

def event52964 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11194⟩⟩) 0 ⟨11175⟩ 46523

def event52965 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11194⟩⟩) 1 ⟨7288⟩ 23634

def event52966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11194⟩⟩) (.product (.predecessor 0 52964 .coefficient) (.predecessor 1 52965 .coefficient) (⟨false, false, none, none, none⟩))

def event52967 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11194⟩⟩, .operator (⟨46523, 0⟩, ⟨23634, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩)

def exact52968RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩]

theorem exact52968RawTermsValid :
    exact52968RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52968 : Event := .resultExact (⟨.program ⟨257⟩, ⟨11194⟩⟩) exact52968RawTerms .large 52966 .exactZero (none)

def event52969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50766⟩⟩) 0 ⟨11194⟩ 52968

def event52970 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50766⟩⟩) 1 ⟨50765⟩ 52963

def event52971 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50766⟩⟩) (.sum [.predecessor 0 52969 .coefficient, .predecessor 1 52970 .coefficient])

def exact52972RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact52972RawTermsValid :
    exact52972RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52972 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50766⟩⟩) exact52972RawTerms .large 52971 .exactZero (none)

def event52973 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50767⟩⟩) 0 ⟨50766⟩ 52972

def event52974 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50767⟩⟩) 1 ⟨114⟩ 23626

def event52975 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50767⟩⟩) (.sum [.predecessor 0 52973 .coefficient, .predecessor 1 52974 .coefficient])

def event52976 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50767⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨114⟩⟩]⟩) [⟨.result 23626 .coefficient, false, none⟩])

def event52977 : Event := .survivorFold (1) 52976

def exact52978RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact52978RawTermsValid :
    exact52978RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52978 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50767⟩⟩) exact52978RawTerms .large 52975 (.finite 26) (some (52976))

def event52979 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50768⟩⟩) 0 ⟨50767⟩ 52978

def event52980 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50768⟩⟩) 1 ⟨9581⟩ 23623

def event52981 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50768⟩⟩) (.product (.predecessor 0 52979 .coefficient) (.predecessor 1 52980 .coefficient) (⟨false, false, none, none, none⟩))

def event52982 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50768⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) [⟨.result 23619 .coefficient, false, none⟩])

def event52983 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50768⟩⟩) (.product (.result 52978 .summary) (.transfer 52982) (⟨false, false, none, none, none⟩))

def event52984 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50768⟩⟩, .operator (⟨52978, 1⟩, ⟨23623, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (-1)⟩)

def event52985 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨50768⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9580⟩⟩) ⟨7308⟩ 23593)

def event52986 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50768⟩⟩, .relation 52985 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (-1)⟩)

def event52987 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50768⟩⟩, .operator (⟨52978, 0⟩, ⟨23623, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩)

def exact52988RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨50761⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩, (-1)⟩]

theorem exact52988RawTermsValid :
    exact52988RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event52988 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50768⟩⟩) exact52988RawTerms .large 52981 (.finite 279172874240) (some (52983))

def event52989 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50769⟩⟩) 0 ⟨50768⟩ 52988

def event52990 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50769⟩⟩) 1 ⟨50764⟩ 52958

def event52991 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50769⟩⟩) (.sum [.predecessor 0 52989 .coefficient, .predecessor 1 52990 .coefficient])

def eventLeaf3296 : Array AnnotatedEvent := #[
  { event := event52736
    frameStart := 0 },
  { event := event52737
    frameStart := 0 },
  { event := event52738
    frameStart := 0 },
  { event := event52739
    frameStart := 52739 },
  { event := event52740
    frameStart := 52739 },
  { event := event52741
    frameStart := 52739 },
  { event := event52742
    frameStart := 52739 },
  { event := event52743
    frameStart := 52739 },
  { event := event52744
    frameStart := 52739 },
  { event := event52745
    frameStart := 52739 },
  { event := event52746
    frameStart := 52739 },
  { event := event52747
    frameStart := 52739 },
  { event := event52748
    frameStart := 52739 },
  { event := event52749
    frameStart := 52739 },
  { event := event52750
    frameStart := 52739 },
  { event := event52751
    frameStart := 52739 }
]

def eventLeaf3297 : Array AnnotatedEvent := #[
  { event := event52752
    frameStart := 52739 },
  { event := event52753
    frameStart := 52739 },
  { event := event52754
    frameStart := 52739 },
  { event := event52755
    frameStart := 52739 },
  { event := event52756
    frameStart := 52739 },
  { event := event52757
    frameStart := 52739 },
  { event := event52758
    frameStart := 52739 },
  { event := event52759
    frameStart := 52739 },
  { event := event52760
    frameStart := 52739 },
  { event := event52761
    frameStart := 52739 },
  { event := event52762
    frameStart := 52739 },
  { event := event52763
    frameStart := 52739 },
  { event := event52764
    frameStart := 52739 },
  { event := event52765
    frameStart := 52739 },
  { event := event52766
    frameStart := 52739 },
  { event := event52767
    frameStart := 52739 }
]

def eventLeaf3298 : Array AnnotatedEvent := #[
  { event := event52768
    frameStart := 52739 },
  { event := event52769
    frameStart := 52739 },
  { event := event52770
    frameStart := 52739 },
  { event := event52771
    frameStart := 52739 },
  { event := event52772
    frameStart := 52739 },
  { event := event52773
    frameStart := 52739 },
  { event := event52774
    frameStart := 52739 },
  { event := event52775
    frameStart := 52739 },
  { event := event52776
    frameStart := 52739 },
  { event := event52777
    frameStart := 52739 },
  { event := event52778
    frameStart := 52739 },
  { event := event52779
    frameStart := 52739 },
  { event := event52780
    frameStart := 52739 },
  { event := event52781
    frameStart := 52739 },
  { event := event52782
    frameStart := 52739 },
  { event := event52783
    frameStart := 52739 }
]

def eventLeaf3299 : Array AnnotatedEvent := #[
  { event := event52784
    frameStart := 52739 },
  { event := event52785
    frameStart := 52739 },
  { event := event52786
    frameStart := 52739 },
  { event := event52787
    frameStart := 52739 },
  { event := event52788
    frameStart := 52739 },
  { event := event52789
    frameStart := 52739 },
  { event := event52790
    frameStart := 52739 },
  { event := event52791
    frameStart := 52739 },
  { event := event52792
    frameStart := 52739 },
  { event := event52793
    frameStart := 52793 },
  { event := event52794
    frameStart := 52793 },
  { event := event52795
    frameStart := 52793 },
  { event := event52796
    frameStart := 52793 },
  { event := event52797
    frameStart := 52793 },
  { event := event52798
    frameStart := 52793 },
  { event := event52799
    frameStart := 52793 }
]

def eventLeaf3300 : Array AnnotatedEvent := #[
  { event := event52800
    frameStart := 52793 },
  { event := event52801
    frameStart := 52793 },
  { event := event52802
    frameStart := 52793 },
  { event := event52803
    frameStart := 52793 },
  { event := event52804
    frameStart := 52793 },
  { event := event52805
    frameStart := 52793 },
  { event := event52806
    frameStart := 52793 },
  { event := event52807
    frameStart := 52793 },
  { event := event52808
    frameStart := 52793 },
  { event := event52809
    frameStart := 52793 },
  { event := event52810
    frameStart := 52793 },
  { event := event52811
    frameStart := 52793 },
  { event := event52812
    frameStart := 52793 },
  { event := event52813
    frameStart := 52793 },
  { event := event52814
    frameStart := 52793 },
  { event := event52815
    frameStart := 52793 }
]

def eventLeaf3301 : Array AnnotatedEvent := #[
  { event := event52816
    frameStart := 52793 },
  { event := event52817
    frameStart := 52793 },
  { event := event52818
    frameStart := 52793 },
  { event := event52819
    frameStart := 52793 },
  { event := event52820
    frameStart := 52793 },
  { event := event52821
    frameStart := 52793 },
  { event := event52822
    frameStart := 52793 },
  { event := event52823
    frameStart := 52793 },
  { event := event52824
    frameStart := 52793 },
  { event := event52825
    frameStart := 52793 },
  { event := event52826
    frameStart := 52793 },
  { event := event52827
    frameStart := 52793 },
  { event := event52828
    frameStart := 52793 },
  { event := event52829
    frameStart := 52793 },
  { event := event52830
    frameStart := 52793 },
  { event := event52831
    frameStart := 52793 }
]

def eventLeaf3302 : Array AnnotatedEvent := #[
  { event := event52832
    frameStart := 52793 },
  { event := event52833
    frameStart := 52793 },
  { event := event52834
    frameStart := 52793 },
  { event := event52835
    frameStart := 52793 },
  { event := event52836
    frameStart := 52793 },
  { event := event52837
    frameStart := 52793 },
  { event := event52838
    frameStart := 52793 },
  { event := event52839
    frameStart := 52793 },
  { event := event52840
    frameStart := 52793 },
  { event := event52841
    frameStart := 52793 },
  { event := event52842
    frameStart := 52793 },
  { event := event52843
    frameStart := 52793 },
  { event := event52844
    frameStart := 52793 },
  { event := event52845
    frameStart := 52793 },
  { event := event52846
    frameStart := 52793 },
  { event := event52847
    frameStart := 52793 }
]

def eventLeaf3303 : Array AnnotatedEvent := #[
  { event := event52848
    frameStart := 52793 },
  { event := event52849
    frameStart := 52793 },
  { event := event52850
    frameStart := 52793 },
  { event := event52851
    frameStart := 52793 },
  { event := event52852
    frameStart := 52793 },
  { event := event52853
    frameStart := 52793 },
  { event := event52854
    frameStart := 52793 },
  { event := event52855
    frameStart := 52793 },
  { event := event52856
    frameStart := 52793 },
  { event := event52857
    frameStart := 52793 },
  { event := event52858
    frameStart := 52793 },
  { event := event52859
    frameStart := 52793 },
  { event := event52860
    frameStart := 52793 },
  { event := event52861
    frameStart := 52793 },
  { event := event52862
    frameStart := 52793 },
  { event := event52863
    frameStart := 52793 }
]

def eventLeaf3304 : Array AnnotatedEvent := #[
  { event := event52864
    frameStart := 52793 },
  { event := event52865
    frameStart := 52793 },
  { event := event52866
    frameStart := 52793 },
  { event := event52867
    frameStart := 52793 },
  { event := event52868
    frameStart := 52793 },
  { event := event52869
    frameStart := 52793 },
  { event := event52870
    frameStart := 52793 },
  { event := event52871
    frameStart := 52793 },
  { event := event52872
    frameStart := 52793 },
  { event := event52873
    frameStart := 52793 },
  { event := event52874
    frameStart := 52793 },
  { event := event52875
    frameStart := 52793 },
  { event := event52876
    frameStart := 52793 },
  { event := event52877
    frameStart := 52793 },
  { event := event52878
    frameStart := 52793 },
  { event := event52879
    frameStart := 52793 }
]

def eventLeaf3305 : Array AnnotatedEvent := #[
  { event := event52880
    frameStart := 52793 },
  { event := event52881
    frameStart := 52793 },
  { event := event52882
    frameStart := 52793 },
  { event := event52883
    frameStart := 52793 },
  { event := event52884
    frameStart := 52793 },
  { event := event52885
    frameStart := 52793 },
  { event := event52886
    frameStart := 52793 },
  { event := event52887
    frameStart := 52793 },
  { event := event52888
    frameStart := 52793 },
  { event := event52889
    frameStart := 52793 },
  { event := event52890
    frameStart := 52793 },
  { event := event52891
    frameStart := 52793 },
  { event := event52892
    frameStart := 52793 },
  { event := event52893
    frameStart := 52793 },
  { event := event52894
    frameStart := 52793 },
  { event := event52895
    frameStart := 52793 }
]

def eventLeaf3306 : Array AnnotatedEvent := #[
  { event := event52896
    frameStart := 52793 },
  { event := event52897
    frameStart := 0 },
  { event := event52898
    frameStart := 0 },
  { event := event52899
    frameStart := 0 },
  { event := event52900
    frameStart := 0 },
  { event := event52901
    frameStart := 0 },
  { event := event52902
    frameStart := 0 },
  { event := event52903
    frameStart := 0 },
  { event := event52904
    frameStart := 0 },
  { event := event52905
    frameStart := 0 },
  { event := event52906
    frameStart := 0 },
  { event := event52907
    frameStart := 0 },
  { event := event52908
    frameStart := 0 },
  { event := event52909
    frameStart := 0 },
  { event := event52910
    frameStart := 0 },
  { event := event52911
    frameStart := 0 }
]

def eventLeaf3307 : Array AnnotatedEvent := #[
  { event := event52912
    frameStart := 0 },
  { event := event52913
    frameStart := 0 },
  { event := event52914
    frameStart := 0 },
  { event := event52915
    frameStart := 0 },
  { event := event52916
    frameStart := 0 },
  { event := event52917
    frameStart := 0 },
  { event := event52918
    frameStart := 0 },
  { event := event52919
    frameStart := 0 },
  { event := event52920
    frameStart := 0 },
  { event := event52921
    frameStart := 0 },
  { event := event52922
    frameStart := 0 },
  { event := event52923
    frameStart := 0 },
  { event := event52924
    frameStart := 0 },
  { event := event52925
    frameStart := 0 },
  { event := event52926
    frameStart := 0 },
  { event := event52927
    frameStart := 0 }
]

def eventLeaf3308 : Array AnnotatedEvent := #[
  { event := event52928
    frameStart := 0 },
  { event := event52929
    frameStart := 0 },
  { event := event52930
    frameStart := 0 },
  { event := event52931
    frameStart := 0 },
  { event := event52932
    frameStart := 0 },
  { event := event52933
    frameStart := 0 },
  { event := event52934
    frameStart := 0 },
  { event := event52935
    frameStart := 0 },
  { event := event52936
    frameStart := 0 },
  { event := event52937
    frameStart := 0 },
  { event := event52938
    frameStart := 0 },
  { event := event52939
    frameStart := 0 },
  { event := event52940
    frameStart := 0 },
  { event := event52941
    frameStart := 0 },
  { event := event52942
    frameStart := 0 },
  { event := event52943
    frameStart := 0 }
]

def eventLeaf3309 : Array AnnotatedEvent := #[
  { event := event52944
    frameStart := 0 },
  { event := event52945
    frameStart := 0 },
  { event := event52946
    frameStart := 0 },
  { event := event52947
    frameStart := 0 },
  { event := event52948
    frameStart := 0 },
  { event := event52949
    frameStart := 0 },
  { event := event52950
    frameStart := 0 },
  { event := event52951
    frameStart := 0 },
  { event := event52952
    frameStart := 0 },
  { event := event52953
    frameStart := 0 },
  { event := event52954
    frameStart := 0 },
  { event := event52955
    frameStart := 0 },
  { event := event52956
    frameStart := 0 },
  { event := event52957
    frameStart := 0 },
  { event := event52958
    frameStart := 0 },
  { event := event52959
    frameStart := 0 }
]

def eventLeaf3310 : Array AnnotatedEvent := #[
  { event := event52960
    frameStart := 0 },
  { event := event52961
    frameStart := 0 },
  { event := event52962
    frameStart := 0 },
  { event := event52963
    frameStart := 0 },
  { event := event52964
    frameStart := 0 },
  { event := event52965
    frameStart := 0 },
  { event := event52966
    frameStart := 0 },
  { event := event52967
    frameStart := 0 },
  { event := event52968
    frameStart := 0 },
  { event := event52969
    frameStart := 0 },
  { event := event52970
    frameStart := 0 },
  { event := event52971
    frameStart := 0 },
  { event := event52972
    frameStart := 0 },
  { event := event52973
    frameStart := 0 },
  { event := event52974
    frameStart := 0 },
  { event := event52975
    frameStart := 0 }
]

def eventLeaf3311 : Array AnnotatedEvent := #[
  { event := event52976
    frameStart := 0 },
  { event := event52977
    frameStart := 0 },
  { event := event52978
    frameStart := 0 },
  { event := event52979
    frameStart := 0 },
  { event := event52980
    frameStart := 0 },
  { event := event52981
    frameStart := 0 },
  { event := event52982
    frameStart := 0 },
  { event := event52983
    frameStart := 0 },
  { event := event52984
    frameStart := 0 },
  { event := event52985
    frameStart := 0 },
  { event := event52986
    frameStart := 0 },
  { event := event52987
    frameStart := 0 },
  { event := event52988
    frameStart := 0 },
  { event := event52989
    frameStart := 0 },
  { event := event52990
    frameStart := 0 },
  { event := event52991
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events206
