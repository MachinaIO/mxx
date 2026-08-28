import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events296

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event75776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10329⟩⟩) (.product (.predecessor 0 75774 .coefficient) (.predecessor 1 75775 .coefficient) (⟨false, false, none, none, none⟩))

def event75777 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10329⟩⟩, .operator (⟨75773, 0⟩, ⟨16137, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def exact75778RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩]

theorem exact75778RawTermsValid :
    exact75778RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75778 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10329⟩⟩) exact75778RawTerms .large 75776 .exactZero (none)

def event75779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10439⟩⟩) 0 ⟨10329⟩ 75778

def event75780 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10439⟩⟩) 1 ⟨10438⟩ 75762

def event75781 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10439⟩⟩) (.sum [.predecessor 0 75779 .coefficient, .predecessor 1 75780 .coefficient])

def exact75782RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩]

theorem exact75782RawTermsValid :
    exact75782RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75782 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10439⟩⟩) exact75782RawTerms .large 75781 .exactZero (none)

def event75783 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10440⟩⟩) 0 ⟨10439⟩ 75782

def event75784 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10440⟩⟩) 1 ⟨25⟩ 75736

def event75785 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10440⟩⟩) (.sum [.predecessor 0 75783 .coefficient, .predecessor 1 75784 .coefficient])

def event75786 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10440⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨25⟩⟩]⟩) [⟨.result 75736 .coefficient, false, none⟩])

def event75787 : Event := .survivorFold (1) 75786

def exact75788RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩]

theorem exact75788RawTermsValid :
    exact75788RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75788 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10440⟩⟩) exact75788RawTerms .large 75785 (.finite 26) (some (75786))

def event75789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67592⟩⟩) 0 ⟨10440⟩ 75788

def event75790 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67592⟩⟩) 1 ⟨67589⟩ 3796

def event75791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.product (.predecessor 0 75789 .coefficient) (.predecessor 1 75790 .coefficient) (⟨false, false, none, none, none⟩))

def event75792 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], []⟩) [⟨.result 36 .coefficient, true, some 1⟩, ⟨.result 3571 .coefficient, true, some 1⟩])

def event75793 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], []⟩) [⟨.result 543 .coefficient, true, some 1⟩, ⟨.result 3579 .coefficient, true, some 1⟩])

def event75794 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.sum [.transfer 75792, .transfer 75793])

def event75795 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], []⟩) [⟨.result 553 .coefficient, true, some 1⟩, ⟨.result 3587 .coefficient, true, some 1⟩])

def event75796 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.sum [.transfer 75794, .transfer 75795])

def event75797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], []⟩) [⟨.result 563 .coefficient, true, some 1⟩, ⟨.result 3595 .coefficient, true, some 1⟩])

def event75798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.sum [.transfer 75796, .transfer 75797])

def event75799 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], []⟩) [⟨.result 573 .coefficient, true, some 1⟩, ⟨.result 3603 .coefficient, true, some 1⟩])

def event75800 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.sum [.transfer 75798, .transfer 75799])

def event75801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], []⟩) [⟨.result 583 .coefficient, true, some 1⟩, ⟨.result 3611 .coefficient, true, some 1⟩])

def event75802 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.sum [.transfer 75800, .transfer 75801])

def event75803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], []⟩) [⟨.result 593 .coefficient, true, some 1⟩, ⟨.result 3619 .coefficient, true, some 1⟩])

def event75804 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.sum [.transfer 75802, .transfer 75803])

def event75805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], []⟩) [⟨.result 603 .coefficient, true, some 1⟩, ⟨.result 3627 .coefficient, true, some 1⟩])

def event75806 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.sum [.transfer 75804, .transfer 75805])

def event75807 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], []⟩) [⟨.result 613 .coefficient, true, some 1⟩, ⟨.result 3635 .coefficient, true, some 1⟩])

def event75808 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.sum [.transfer 75806, .transfer 75807])

def event75809 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], []⟩) [⟨.result 623 .coefficient, true, some 1⟩, ⟨.result 3643 .coefficient, true, some 1⟩])

def event75810 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.sum [.transfer 75808, .transfer 75809])

def event75811 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], []⟩) [⟨.result 633 .coefficient, true, some 1⟩, ⟨.result 3651 .coefficient, true, some 1⟩])

def event75812 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.sum [.transfer 75810, .transfer 75811])

def event75813 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩) [⟨.result 643 .coefficient, true, some 1⟩, ⟨.result 3659 .coefficient, true, some 1⟩])

def event75814 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.sum [.transfer 75812, .transfer 75813])

def event75815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩) [⟨.result 653 .coefficient, true, some 1⟩, ⟨.result 3667 .coefficient, true, some 1⟩])

def event75816 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.sum [.transfer 75814, .transfer 75815])

def event75817 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩) [⟨.result 663 .coefficient, true, some 1⟩, ⟨.result 3675 .coefficient, true, some 1⟩])

def event75818 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.sum [.transfer 75816, .transfer 75817])

def event75819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩) [⟨.result 673 .coefficient, true, some 1⟩, ⟨.result 3683 .coefficient, true, some 1⟩])

def event75820 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.sum [.transfer 75818, .transfer 75819])

def event75821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩) [⟨.result 683 .coefficient, true, some 1⟩, ⟨.result 3691 .coefficient, true, some 1⟩])

def event75822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.sum [.transfer 75820, .transfer 75821])

def event75823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩) [⟨.result 693 .coefficient, true, some 1⟩, ⟨.result 3699 .coefficient, true, some 1⟩])

def event75824 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.sum [.transfer 75822, .transfer 75823])

def event75825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩) [⟨.result 703 .coefficient, true, some 1⟩, ⟨.result 3707 .coefficient, true, some 1⟩])

def event75826 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.sum [.transfer 75824, .transfer 75825])

def event75827 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩) [⟨.result 713 .coefficient, true, some 1⟩, ⟨.result 3715 .coefficient, true, some 1⟩])

def event75828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.sum [.transfer 75826, .transfer 75827])

def event75829 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67592⟩⟩) (.product (.result 75788 .summary) (.transfer 75828) (⟨false, false, none, none, none⟩))

def event75830 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event75831 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75832 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75833 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75834 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75835 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75836 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75837 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75838 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75839 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 18⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75840 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75841 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75842 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75843 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75844 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75845 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75846 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75847 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75848 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 0⟩, ⟨3796, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75849 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (-1)⟩)

def event75850 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def event75851 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def event75852 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def event75853 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def event75854 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def event75855 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def event75856 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def event75857 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def event75858 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 18⟩), ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def event75859 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def event75860 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def event75861 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def event75862 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def event75863 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def event75864 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def event75865 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def event75866 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def event75867 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67592⟩⟩, .operator (⟨75788, 1⟩, ⟨3796, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩)

def exact75868RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩, (1)⟩]

theorem exact75868RawTermsValid :
    exact75868RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75868 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67592⟩⟩) exact75868RawTerms .large 75791 (.finite 6902113630329048043564518670336) (some (75829))

def event75869 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68865⟩⟩) 0 ⟨67031⟩ 3568

def event75870 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68865⟩⟩) (.authority (.programFamilyFact))

def event75871 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68865⟩⟩) (.finite 1152)

def event75872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68866⟩⟩) 0 ⟨7177⟩ 15500

def event75873 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68866⟩⟩) 1 ⟨68865⟩ 75871

def event75874 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68866⟩⟩) (.authority (.operator))

def exact75875RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩, (1)⟩]

theorem exact75875RawTermsValid :
    exact75875RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75875 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68866⟩⟩) exact75875RawTerms .large 75874 .exactZero (none)

def event75876 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71437⟩⟩) 0 ⟨68866⟩ 75875

def event75877 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71437⟩⟩) (.authority (.operator))

def exact75878RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩, (1)⟩]

theorem exact75878RawTermsValid :
    exact75878RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75878 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71437⟩⟩) exact75878RawTerms (.finite 8192) 75877 .exactZero (none)

def event75879 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49353⟩⟩) 0 ⟨48197⟩ 3103

def event75880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49353⟩⟩) (.authority (.programFamilyFact))

def event75881 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49353⟩⟩) (.finite 3720)

def event75882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49355⟩⟩) 0 ⟨7177⟩ 15500

def event75883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49355⟩⟩) 1 ⟨49353⟩ 75881

def event75884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49355⟩⟩) (.authority (.operator))

def exact75885RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49355⟩⟩]⟩, (1)⟩]

theorem exact75885RawTermsValid :
    exact75885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75885 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49355⟩⟩) exact75885RawTerms .large 75884 .exactZero (none)

def event75886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50179⟩⟩) 0 ⟨49355⟩ 75885

def event75887 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50179⟩⟩) (.authority (.operator))

def exact75888RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨50179⟩⟩]⟩, (1)⟩]

theorem exact75888RawTermsValid :
    exact75888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75888 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50179⟩⟩) exact75888RawTerms (.finite 8192) 75887 .exactZero (none)

def event75889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49184⟩⟩) 0 ⟨47980⟩ 3097

def event75890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49184⟩⟩) (.authority (.programFamilyFact))

def event75891 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49184⟩⟩) (.finite 3720)

def event75892 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49185⟩⟩) 0 ⟨7177⟩ 15500

def event75893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49185⟩⟩) 1 ⟨49184⟩ 75891

def event75894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49185⟩⟩) (.authority (.operator))

def exact75895RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49185⟩⟩]⟩, (1)⟩]

theorem exact75895RawTermsValid :
    exact75895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49185⟩⟩) exact75895RawTerms .large 75894 .exactZero (none)

def event75896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49725⟩⟩) 0 ⟨49185⟩ 75895

def event75897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49725⟩⟩) (.authority (.operator))

def exact75898RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49725⟩⟩]⟩, (1)⟩]

theorem exact75898RawTermsValid :
    exact75898RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75898 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49725⟩⟩) exact75898RawTerms (.finite 8192) 75897 .exactZero (none)

def event75899 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10328⟩⟩) 0 ⟨10327⟩ 75773

def event75900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10328⟩⟩) 1 ⟨6908⟩ 2

def event75901 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10328⟩⟩) (.product (.predecessor 0 75899 .coefficient) (.predecessor 1 75900 .coefficient) (⟨false, false, none, none, none⟩))

def event75902 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10328⟩⟩, .operator (⟨75773, 0⟩, ⟨2, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact75903RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact75903RawTermsValid :
    exact75903RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75903 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10328⟩⟩) exact75903RawTerms .large 75901 .exactZero (none)

def event75904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47981⟩⟩) 0 ⟨47978⟩ 3086

def event75905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47981⟩⟩) 1 ⟨10328⟩ 75903

def event75906 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47981⟩⟩) (.tensor (.predecessor 0 75904 .coefficient) (.predecessor 1 75905 .coefficient) true false)

def event75907 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47981⟩⟩, .operator (⟨3086, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact75908RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact75908RawTermsValid :
    exact75908RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75908 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47981⟩⟩) exact75908RawTerms .large 75906 .exactZero (none)

def event75909 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10343⟩⟩) 0 ⟨10327⟩ 75773

def event75910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10343⟩⟩) 1 ⟨7285⟩ 17065

def event75911 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10343⟩⟩) (.product (.predecessor 0 75909 .coefficient) (.predecessor 1 75910 .coefficient) (⟨false, false, none, none, none⟩))

def event75912 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10343⟩⟩, .operator (⟨75773, 0⟩, ⟨17065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩)

def exact75913RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩]

theorem exact75913RawTermsValid :
    exact75913RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75913 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10343⟩⟩) exact75913RawTerms .large 75911 .exactZero (none)

def event75914 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47982⟩⟩) 0 ⟨10343⟩ 75913

def event75915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47982⟩⟩) 1 ⟨47981⟩ 75908

def event75916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47982⟩⟩) (.sum [.predecessor 0 75914 .coefficient, .predecessor 1 75915 .coefficient])

def exact75917RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact75917RawTermsValid :
    exact75917RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75917 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47982⟩⟩) exact75917RawTerms .large 75916 .exactZero (none)

def event75918 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47983⟩⟩) 0 ⟨47982⟩ 75917

def event75919 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47983⟩⟩) 1 ⟨111⟩ 17052

def event75920 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47983⟩⟩) (.sum [.predecessor 0 75918 .coefficient, .predecessor 1 75919 .coefficient])

def event75921 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47983⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨111⟩⟩]⟩) [⟨.result 17052 .coefficient, false, none⟩])

def event75922 : Event := .survivorFold (1) 75921

def exact75923RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact75923RawTermsValid :
    exact75923RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75923 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47983⟩⟩) exact75923RawTerms .large 75920 (.finite 26) (some (75921))

def event75924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47984⟩⟩) 0 ⟨47983⟩ 75923

def event75925 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47984⟩⟩) 1 ⟨15171⟩ 3089

def event75926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47984⟩⟩) (.product (.predecessor 0 75924 .coefficient) (.predecessor 1 75925 .coefficient) (⟨false, true, none, none, some 1⟩))

def event75927 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47984⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨15171⟩⟩], []⟩) [⟨.result 3089 .coefficient, true, some 1⟩])

def event75928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47984⟩⟩) (.product (.result 75923 .summary) (.transfer 75927) (⟨false, false, none, none, none⟩))

def event75929 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47984⟩⟩, .operator (⟨75923, 1⟩, ⟨3089, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event75930 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47984⟩⟩, .operator (⟨75923, 0⟩, ⟨3089, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩)

def exact75931RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact75931RawTermsValid :
    exact75931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75931 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47984⟩⟩) exact75931RawTerms .large 75926 (.finite 51118080) (some (75928))

def event75932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15172⟩⟩) 0 ⟨15171⟩ 3089

def event75933 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15172⟩⟩) 1 ⟨10328⟩ 75903

def event75934 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15172⟩⟩) (.tensor (.predecessor 0 75932 .coefficient) (.predecessor 1 75933 .coefficient) true false)

def event75935 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15172⟩⟩, .operator (⟨3089, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact75936RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact75936RawTermsValid :
    exact75936RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75936 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15172⟩⟩) exact75936RawTerms .large 75934 .exactZero (none)

def event75937 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10360⟩⟩) 0 ⟨10327⟩ 75773

def event75938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10360⟩⟩) 1 ⟨7302⟩ 17106

def event75939 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10360⟩⟩) (.product (.predecessor 0 75937 .coefficient) (.predecessor 1 75938 .coefficient) (⟨false, false, none, none, none⟩))

def event75940 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10360⟩⟩, .operator (⟨75773, 0⟩, ⟨17106, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩)

def exact75941RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩]

theorem exact75941RawTermsValid :
    exact75941RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75941 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10360⟩⟩) exact75941RawTerms .large 75939 .exactZero (none)

def event75942 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15173⟩⟩) 0 ⟨10360⟩ 75941

def event75943 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15173⟩⟩) 1 ⟨15172⟩ 75936

def event75944 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15173⟩⟩) (.sum [.predecessor 0 75942 .coefficient, .predecessor 1 75943 .coefficient])

def exact75945RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact75945RawTermsValid :
    exact75945RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75945 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15173⟩⟩) exact75945RawTerms .large 75944 .exactZero (none)

def event75946 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15174⟩⟩) 0 ⟨15173⟩ 75945

def event75947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15174⟩⟩) 1 ⟨128⟩ 17098

def event75948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15174⟩⟩) (.sum [.predecessor 0 75946 .coefficient, .predecessor 1 75947 .coefficient])

def event75949 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15174⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨128⟩⟩]⟩) [⟨.result 17098 .coefficient, false, none⟩])

def event75950 : Event := .survivorFold (1) 75949

def exact75951RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact75951RawTermsValid :
    exact75951RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75951 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15174⟩⟩) exact75951RawTerms .large 75948 (.finite 26) (some (75949))

def event75952 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15175⟩⟩) 0 ⟨15174⟩ 75951

def event75953 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15175⟩⟩) 1 ⟨9566⟩ 17095

def event75954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15175⟩⟩) (.product (.predecessor 0 75952 .coefficient) (.predecessor 1 75953 .coefficient) (⟨false, false, none, none, none⟩))

def event75955 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15175⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) [⟨.result 17091 .coefficient, false, none⟩])

def event75956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15175⟩⟩) (.product (.result 75951 .summary) (.transfer 75955) (⟨false, false, none, none, none⟩))

def event75957 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15175⟩⟩, .operator (⟨75951, 1⟩, ⟨17095, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (-1)⟩)

def event75958 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨15175⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9565⟩⟩) ⟨7285⟩ 17065)

def event75959 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15175⟩⟩, .relation 75958 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (-1)⟩)

def event75960 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15175⟩⟩, .operator (⟨75951, 0⟩, ⟨17095, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩)

def exact75961RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (-1)⟩]

theorem exact75961RawTermsValid :
    exact75961RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75961 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15175⟩⟩) exact75961RawTerms .large 75954 (.finite 279172874240) (some (75956))

def event75962 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47985⟩⟩) 0 ⟨15175⟩ 75961

def event75963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47985⟩⟩) 1 ⟨47984⟩ 75931

def event75964 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47985⟩⟩) (.sum [.predecessor 0 75962 .coefficient, .predecessor 1 75963 .coefficient])

def event75965 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47985⟩⟩, .operator (⟨75961, 1⟩, ⟨75931, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩)

def event75966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47985⟩⟩) (.sum [.result 75961 .summary, .result 75931 .summary])

def exact75967RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact75967RawTermsValid :
    exact75967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75967 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47985⟩⟩) exact75967RawTerms .large 75964 (.finite 279223992320) (some (75966))

def event75968 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49726⟩⟩) 0 ⟨47985⟩ 75967

def event75969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49726⟩⟩) 1 ⟨49725⟩ 75898

def event75970 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49726⟩⟩) (.product (.predecessor 0 75968 .coefficient) (.predecessor 1 75969 .coefficient) (⟨false, false, none, none, none⟩))

def event75971 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49726⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨49725⟩⟩]⟩) [⟨.result 75898 .coefficient, false, none⟩])

def event75972 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49726⟩⟩) (.product (.result 75967 .summary) (.transfer 75971) (⟨false, false, none, none, none⟩))

def event75973 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49726⟩⟩, .operator (⟨75967, 1⟩, ⟨75898, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49725⟩⟩]⟩, (-1)⟩)

def event75974 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49726⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49725⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49725⟩⟩) ⟨49185⟩ 75895)

def event75975 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49726⟩⟩, .relation 75974 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], [⟨.program ⟨257⟩, ⟨49185⟩⟩]⟩, (-1)⟩)

def event75976 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49726⟩⟩, .operator (⟨75967, 0⟩, ⟨75898, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49725⟩⟩]⟩, (1)⟩)

def exact75977RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15171⟩⟩, ⟨.program ⟨257⟩, ⟨47978⟩⟩], [⟨.program ⟨257⟩, ⟨49185⟩⟩]⟩, (-1)⟩]

theorem exact75977RawTermsValid :
    exact75977RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75977 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49726⟩⟩) exact75977RawTerms .large 75970 (.finite 2998144788182387916800) (some (75972))

def event75978 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48649⟩⟩) 0 ⟨47980⟩ 3097

def event75979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48649⟩⟩) (.authority (.relationPreimageSource ⟨54⟩))

def exact75980RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48649⟩⟩]⟩, (1)⟩]

theorem exact75980RawTermsValid :
    exact75980RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75980 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48649⟩⟩) exact75980RawTerms (.finite 5647228698) 75979 .exactZero (none)

def event75981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48651⟩⟩) 0 ⟨48649⟩ 75980

def event75982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48651⟩⟩) 1 ⟨2370⟩ 4

def event75983 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48651⟩⟩) (.scale (.predecessor 0 75981 .coefficient) (.value (.predecessor 1 75982 .coefficient)))

def exact75984RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48649⟩⟩]⟩, (1)⟩]

theorem exact75984RawTermsValid :
    exact75984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75984 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48651⟩⟩) exact75984RawTerms (.finite 5647228698) 75983 .exactZero (none)

def event75985 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10367⟩⟩) 0 ⟨10327⟩ 75773

def event75986 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10367⟩⟩) 1 ⟨35⟩ 17158

def event75987 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10367⟩⟩) (.product (.predecessor 0 75985 .coefficient) (.predecessor 1 75986 .coefficient) (⟨false, false, none, none, none⟩))

def event75988 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10367⟩⟩, .operator (⟨75773, 0⟩, ⟨17158, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩)

def exact75989RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact75989RawTermsValid :
    exact75989RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75989 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10367⟩⟩) exact75989RawTerms .large 75987 .exactZero (none)

def event75990 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10368⟩⟩) 0 ⟨10367⟩ 75989

def event75991 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10368⟩⟩) 1 ⟨22⟩ 17156

def event75992 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10368⟩⟩) (.sum [.predecessor 0 75990 .coefficient, .predecessor 1 75991 .coefficient])

def event75993 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10368⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨22⟩⟩]⟩) [⟨.result 17156 .coefficient, false, none⟩])

def event75994 : Event := .survivorFold (1) 75993

def exact75995RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact75995RawTermsValid :
    exact75995RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event75995 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10368⟩⟩) exact75995RawTerms .large 75992 (.finite 26) (some (75993))

def event75996 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48652⟩⟩) 0 ⟨10368⟩ 75995

def event75997 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48652⟩⟩) 1 ⟨48651⟩ 75984

def event75998 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48652⟩⟩) (.product (.predecessor 0 75996 .coefficient) (.predecessor 1 75997 .coefficient) (⟨false, false, none, none, none⟩))

def event75999 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48652⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨48649⟩⟩]⟩) [⟨.result 75980 .coefficient, false, none⟩])

def event76000 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48652⟩⟩) (.product (.result 75995 .summary) (.transfer 75999) (⟨false, false, none, none, none⟩))

def event76001 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48652⟩⟩, .operator (⟨75995, 0⟩, ⟨75984, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48649⟩⟩]⟩, (1)⟩)

def event76002 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨48650⟩⟩)

def event76003 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event76004 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event76005 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event76006 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event76007 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event76008 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event76009 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event76010 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event76011 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 76010

def event76012 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 76008

def event76013 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 76011 .coefficient) (.value (.predecessor 1 76012 .coefficient)))

def event76014 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event76015 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 76014

def event76016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 76006

def event76017 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 76015 .coefficient, .predecessor 1 76016 .coefficient])

def event76018 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event76019 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 76018

def event76020 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 76004

def event76021 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 76020 .coefficient))

def event76022 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event76023 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47978⟩⟩) 0 ⟨10325⟩ 76022

def event76024 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47978⟩⟩) (.authority (.programFamilyFact))

def exact76025RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47978⟩⟩], []⟩, (1)⟩]

theorem exact76025RawTermsValid :
    exact76025RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76025 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47978⟩⟩) exact76025RawTerms (.finite 60) 76024 .exactZero (none)

def event76026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15171⟩⟩) 0 ⟨10325⟩ 76022

def event76027 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15171⟩⟩) (.authority (.programFamilyFact))

def exact76028RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15171⟩⟩], []⟩, (1)⟩]

theorem exact76028RawTermsValid :
    exact76028RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76028 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15171⟩⟩) exact76028RawTerms (.finite 60) 76027 .exactZero (none)

def event76029 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47979⟩⟩) 0 ⟨15171⟩ 76028

def event76030 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47979⟩⟩) 1 ⟨47978⟩ 76025

def event76031 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47979⟩⟩) (.product (.predecessor 0 76029 .coefficient) (.predecessor 1 76030 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def eventLeaf4736 : Array AnnotatedEvent := #[
  { event := event75776
    frameStart := 0 },
  { event := event75777
    frameStart := 0 },
  { event := event75778
    frameStart := 0 },
  { event := event75779
    frameStart := 0 },
  { event := event75780
    frameStart := 0 },
  { event := event75781
    frameStart := 0 },
  { event := event75782
    frameStart := 0 },
  { event := event75783
    frameStart := 0 },
  { event := event75784
    frameStart := 0 },
  { event := event75785
    frameStart := 0 },
  { event := event75786
    frameStart := 0 },
  { event := event75787
    frameStart := 0 },
  { event := event75788
    frameStart := 0 },
  { event := event75789
    frameStart := 0 },
  { event := event75790
    frameStart := 0 },
  { event := event75791
    frameStart := 0 }
]

def eventLeaf4737 : Array AnnotatedEvent := #[
  { event := event75792
    frameStart := 0 },
  { event := event75793
    frameStart := 0 },
  { event := event75794
    frameStart := 0 },
  { event := event75795
    frameStart := 0 },
  { event := event75796
    frameStart := 0 },
  { event := event75797
    frameStart := 0 },
  { event := event75798
    frameStart := 0 },
  { event := event75799
    frameStart := 0 },
  { event := event75800
    frameStart := 0 },
  { event := event75801
    frameStart := 0 },
  { event := event75802
    frameStart := 0 },
  { event := event75803
    frameStart := 0 },
  { event := event75804
    frameStart := 0 },
  { event := event75805
    frameStart := 0 },
  { event := event75806
    frameStart := 0 },
  { event := event75807
    frameStart := 0 }
]

def eventLeaf4738 : Array AnnotatedEvent := #[
  { event := event75808
    frameStart := 0 },
  { event := event75809
    frameStart := 0 },
  { event := event75810
    frameStart := 0 },
  { event := event75811
    frameStart := 0 },
  { event := event75812
    frameStart := 0 },
  { event := event75813
    frameStart := 0 },
  { event := event75814
    frameStart := 0 },
  { event := event75815
    frameStart := 0 },
  { event := event75816
    frameStart := 0 },
  { event := event75817
    frameStart := 0 },
  { event := event75818
    frameStart := 0 },
  { event := event75819
    frameStart := 0 },
  { event := event75820
    frameStart := 0 },
  { event := event75821
    frameStart := 0 },
  { event := event75822
    frameStart := 0 },
  { event := event75823
    frameStart := 0 }
]

def eventLeaf4739 : Array AnnotatedEvent := #[
  { event := event75824
    frameStart := 0 },
  { event := event75825
    frameStart := 0 },
  { event := event75826
    frameStart := 0 },
  { event := event75827
    frameStart := 0 },
  { event := event75828
    frameStart := 0 },
  { event := event75829
    frameStart := 0 },
  { event := event75830
    frameStart := 0 },
  { event := event75831
    frameStart := 0 },
  { event := event75832
    frameStart := 0 },
  { event := event75833
    frameStart := 0 },
  { event := event75834
    frameStart := 0 },
  { event := event75835
    frameStart := 0 },
  { event := event75836
    frameStart := 0 },
  { event := event75837
    frameStart := 0 },
  { event := event75838
    frameStart := 0 },
  { event := event75839
    frameStart := 0 }
]

def eventLeaf4740 : Array AnnotatedEvent := #[
  { event := event75840
    frameStart := 0 },
  { event := event75841
    frameStart := 0 },
  { event := event75842
    frameStart := 0 },
  { event := event75843
    frameStart := 0 },
  { event := event75844
    frameStart := 0 },
  { event := event75845
    frameStart := 0 },
  { event := event75846
    frameStart := 0 },
  { event := event75847
    frameStart := 0 },
  { event := event75848
    frameStart := 0 },
  { event := event75849
    frameStart := 0 },
  { event := event75850
    frameStart := 0 },
  { event := event75851
    frameStart := 0 },
  { event := event75852
    frameStart := 0 },
  { event := event75853
    frameStart := 0 },
  { event := event75854
    frameStart := 0 },
  { event := event75855
    frameStart := 0 }
]

def eventLeaf4741 : Array AnnotatedEvent := #[
  { event := event75856
    frameStart := 0 },
  { event := event75857
    frameStart := 0 },
  { event := event75858
    frameStart := 0 },
  { event := event75859
    frameStart := 0 },
  { event := event75860
    frameStart := 0 },
  { event := event75861
    frameStart := 0 },
  { event := event75862
    frameStart := 0 },
  { event := event75863
    frameStart := 0 },
  { event := event75864
    frameStart := 0 },
  { event := event75865
    frameStart := 0 },
  { event := event75866
    frameStart := 0 },
  { event := event75867
    frameStart := 0 },
  { event := event75868
    frameStart := 0 },
  { event := event75869
    frameStart := 0 },
  { event := event75870
    frameStart := 0 },
  { event := event75871
    frameStart := 0 }
]

def eventLeaf4742 : Array AnnotatedEvent := #[
  { event := event75872
    frameStart := 0 },
  { event := event75873
    frameStart := 0 },
  { event := event75874
    frameStart := 0 },
  { event := event75875
    frameStart := 0 },
  { event := event75876
    frameStart := 0 },
  { event := event75877
    frameStart := 0 },
  { event := event75878
    frameStart := 0 },
  { event := event75879
    frameStart := 0 },
  { event := event75880
    frameStart := 0 },
  { event := event75881
    frameStart := 0 },
  { event := event75882
    frameStart := 0 },
  { event := event75883
    frameStart := 0 },
  { event := event75884
    frameStart := 0 },
  { event := event75885
    frameStart := 0 },
  { event := event75886
    frameStart := 0 },
  { event := event75887
    frameStart := 0 }
]

def eventLeaf4743 : Array AnnotatedEvent := #[
  { event := event75888
    frameStart := 0 },
  { event := event75889
    frameStart := 0 },
  { event := event75890
    frameStart := 0 },
  { event := event75891
    frameStart := 0 },
  { event := event75892
    frameStart := 0 },
  { event := event75893
    frameStart := 0 },
  { event := event75894
    frameStart := 0 },
  { event := event75895
    frameStart := 0 },
  { event := event75896
    frameStart := 0 },
  { event := event75897
    frameStart := 0 },
  { event := event75898
    frameStart := 0 },
  { event := event75899
    frameStart := 0 },
  { event := event75900
    frameStart := 0 },
  { event := event75901
    frameStart := 0 },
  { event := event75902
    frameStart := 0 },
  { event := event75903
    frameStart := 0 }
]

def eventLeaf4744 : Array AnnotatedEvent := #[
  { event := event75904
    frameStart := 0 },
  { event := event75905
    frameStart := 0 },
  { event := event75906
    frameStart := 0 },
  { event := event75907
    frameStart := 0 },
  { event := event75908
    frameStart := 0 },
  { event := event75909
    frameStart := 0 },
  { event := event75910
    frameStart := 0 },
  { event := event75911
    frameStart := 0 },
  { event := event75912
    frameStart := 0 },
  { event := event75913
    frameStart := 0 },
  { event := event75914
    frameStart := 0 },
  { event := event75915
    frameStart := 0 },
  { event := event75916
    frameStart := 0 },
  { event := event75917
    frameStart := 0 },
  { event := event75918
    frameStart := 0 },
  { event := event75919
    frameStart := 0 }
]

def eventLeaf4745 : Array AnnotatedEvent := #[
  { event := event75920
    frameStart := 0 },
  { event := event75921
    frameStart := 0 },
  { event := event75922
    frameStart := 0 },
  { event := event75923
    frameStart := 0 },
  { event := event75924
    frameStart := 0 },
  { event := event75925
    frameStart := 0 },
  { event := event75926
    frameStart := 0 },
  { event := event75927
    frameStart := 0 },
  { event := event75928
    frameStart := 0 },
  { event := event75929
    frameStart := 0 },
  { event := event75930
    frameStart := 0 },
  { event := event75931
    frameStart := 0 },
  { event := event75932
    frameStart := 0 },
  { event := event75933
    frameStart := 0 },
  { event := event75934
    frameStart := 0 },
  { event := event75935
    frameStart := 0 }
]

def eventLeaf4746 : Array AnnotatedEvent := #[
  { event := event75936
    frameStart := 0 },
  { event := event75937
    frameStart := 0 },
  { event := event75938
    frameStart := 0 },
  { event := event75939
    frameStart := 0 },
  { event := event75940
    frameStart := 0 },
  { event := event75941
    frameStart := 0 },
  { event := event75942
    frameStart := 0 },
  { event := event75943
    frameStart := 0 },
  { event := event75944
    frameStart := 0 },
  { event := event75945
    frameStart := 0 },
  { event := event75946
    frameStart := 0 },
  { event := event75947
    frameStart := 0 },
  { event := event75948
    frameStart := 0 },
  { event := event75949
    frameStart := 0 },
  { event := event75950
    frameStart := 0 },
  { event := event75951
    frameStart := 0 }
]

def eventLeaf4747 : Array AnnotatedEvent := #[
  { event := event75952
    frameStart := 0 },
  { event := event75953
    frameStart := 0 },
  { event := event75954
    frameStart := 0 },
  { event := event75955
    frameStart := 0 },
  { event := event75956
    frameStart := 0 },
  { event := event75957
    frameStart := 0 },
  { event := event75958
    frameStart := 0 },
  { event := event75959
    frameStart := 0 },
  { event := event75960
    frameStart := 0 },
  { event := event75961
    frameStart := 0 },
  { event := event75962
    frameStart := 0 },
  { event := event75963
    frameStart := 0 },
  { event := event75964
    frameStart := 0 },
  { event := event75965
    frameStart := 0 },
  { event := event75966
    frameStart := 0 },
  { event := event75967
    frameStart := 0 }
]

def eventLeaf4748 : Array AnnotatedEvent := #[
  { event := event75968
    frameStart := 0 },
  { event := event75969
    frameStart := 0 },
  { event := event75970
    frameStart := 0 },
  { event := event75971
    frameStart := 0 },
  { event := event75972
    frameStart := 0 },
  { event := event75973
    frameStart := 0 },
  { event := event75974
    frameStart := 0 },
  { event := event75975
    frameStart := 0 },
  { event := event75976
    frameStart := 0 },
  { event := event75977
    frameStart := 0 },
  { event := event75978
    frameStart := 0 },
  { event := event75979
    frameStart := 0 },
  { event := event75980
    frameStart := 0 },
  { event := event75981
    frameStart := 0 },
  { event := event75982
    frameStart := 0 },
  { event := event75983
    frameStart := 0 }
]

def eventLeaf4749 : Array AnnotatedEvent := #[
  { event := event75984
    frameStart := 0 },
  { event := event75985
    frameStart := 0 },
  { event := event75986
    frameStart := 0 },
  { event := event75987
    frameStart := 0 },
  { event := event75988
    frameStart := 0 },
  { event := event75989
    frameStart := 0 },
  { event := event75990
    frameStart := 0 },
  { event := event75991
    frameStart := 0 },
  { event := event75992
    frameStart := 0 },
  { event := event75993
    frameStart := 0 },
  { event := event75994
    frameStart := 0 },
  { event := event75995
    frameStart := 0 },
  { event := event75996
    frameStart := 0 },
  { event := event75997
    frameStart := 0 },
  { event := event75998
    frameStart := 0 },
  { event := event75999
    frameStart := 0 }
]

def eventLeaf4750 : Array AnnotatedEvent := #[
  { event := event76000
    frameStart := 0 },
  { event := event76001
    frameStart := 0 },
  { event := event76002
    frameStart := 76002 },
  { event := event76003
    frameStart := 76002 },
  { event := event76004
    frameStart := 76002 },
  { event := event76005
    frameStart := 76002 },
  { event := event76006
    frameStart := 76002 },
  { event := event76007
    frameStart := 76002 },
  { event := event76008
    frameStart := 76002 },
  { event := event76009
    frameStart := 76002 },
  { event := event76010
    frameStart := 76002 },
  { event := event76011
    frameStart := 76002 },
  { event := event76012
    frameStart := 76002 },
  { event := event76013
    frameStart := 76002 },
  { event := event76014
    frameStart := 76002 },
  { event := event76015
    frameStart := 76002 }
]

def eventLeaf4751 : Array AnnotatedEvent := #[
  { event := event76016
    frameStart := 76002 },
  { event := event76017
    frameStart := 76002 },
  { event := event76018
    frameStart := 76002 },
  { event := event76019
    frameStart := 76002 },
  { event := event76020
    frameStart := 76002 },
  { event := event76021
    frameStart := 76002 },
  { event := event76022
    frameStart := 76002 },
  { event := event76023
    frameStart := 76002 },
  { event := event76024
    frameStart := 76002 },
  { event := event76025
    frameStart := 76002 },
  { event := event76026
    frameStart := 76002 },
  { event := event76027
    frameStart := 76002 },
  { event := event76028
    frameStart := 76002 },
  { event := event76029
    frameStart := 76002 },
  { event := event76030
    frameStart := 76002 },
  { event := event76031
    frameStart := 76002 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events296
