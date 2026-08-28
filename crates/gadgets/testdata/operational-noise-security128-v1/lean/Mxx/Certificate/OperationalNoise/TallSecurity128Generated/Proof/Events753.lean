import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events753

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event192768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5907⟩⟩) 1 ⟨5905⟩ 9067

def event192769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5907⟩⟩) 2 ⟨5906⟩ 192766

def event192770 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5907⟩⟩) 3 ⟨136⟩ 6

def event192771 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5907⟩⟩) 4 ⟨2370⟩ 4

def event192772 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5907⟩⟩) (.identity (.predecessor 0 192767 .coefficient))

def exact192773RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6173⟩⟩]⟩, (1)⟩]

theorem exact192773RawTermsValid :
    exact192773RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192773 : Event := .resultExact (⟨.program ⟨257⟩, ⟨5907⟩⟩) exact192773RawTerms (.finite 1) 192772 .exactZero (none)

def event192774 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8805⟩⟩) 0 ⟨5907⟩ 192773

def event192775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8805⟩⟩) 1 ⟨7259⟩ 16457

def event192776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8805⟩⟩) (.product (.predecessor 0 192774 .coefficient) (.predecessor 1 192775 .coefficient) (⟨false, false, none, none, none⟩))

def event192777 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8805⟩⟩, .operator (⟨192773, 0⟩, ⟨16457, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def exact192778RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩]

theorem exact192778RawTermsValid :
    exact192778RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192778 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8805⟩⟩) exact192778RawTerms .large 192776 .exactZero (none)

def event192779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9411⟩⟩) 0 ⟨8805⟩ 192778

def event192780 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9411⟩⟩) 1 ⟨7001⟩ 192762

def event192781 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9411⟩⟩) (.sum [.predecessor 0 192779 .coefficient, .predecessor 1 192780 .coefficient])

def exact192782RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩]

theorem exact192782RawTermsValid :
    exact192782RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192782 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9411⟩⟩) exact192782RawTerms .large 192781 .exactZero (none)

def event192783 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9412⟩⟩) 0 ⟨9411⟩ 192782

def event192784 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9412⟩⟩) 1 ⟨7⟩ 192736

def event192785 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9412⟩⟩) (.sum [.predecessor 0 192783 .coefficient, .predecessor 1 192784 .coefficient])

def event192786 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9412⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7⟩⟩]⟩) [⟨.result 192736 .coefficient, false, none⟩])

def event192787 : Event := .survivorFold (1) 192786

def exact192788RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩]

theorem exact192788RawTermsValid :
    exact192788RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192788 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9412⟩⟩) exact192788RawTerms .large 192785 (.finite 26) (some (192786))

def event192789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67500⟩⟩) 0 ⟨9412⟩ 192788

def event192790 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67500⟩⟩) 1 ⟨67497⟩ 9780

def event192791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.product (.predecessor 0 192789 .coefficient) (.predecessor 1 192790 .coefficient) (⟨false, false, none, none, none⟩))

def event192792 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67494⟩⟩], []⟩) [⟨.result 36 .coefficient, true, some 1⟩, ⟨.result 9555 .coefficient, true, some 1⟩])

def event192793 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48385⟩⟩], []⟩) [⟨.result 543 .coefficient, true, some 1⟩, ⟨.result 9563 .coefficient, true, some 1⟩])

def event192794 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.sum [.transfer 192792, .transfer 192793])

def event192795 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45705⟩⟩], []⟩) [⟨.result 553 .coefficient, true, some 1⟩, ⟨.result 9571 .coefficient, true, some 1⟩])

def event192796 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.sum [.transfer 192794, .transfer 192795])

def event192797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43028⟩⟩], []⟩) [⟨.result 563 .coefficient, true, some 1⟩, ⟨.result 9579 .coefficient, true, some 1⟩])

def event192798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.sum [.transfer 192796, .transfer 192797])

def event192799 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], []⟩) [⟨.result 573 .coefficient, true, some 1⟩, ⟨.result 9587 .coefficient, true, some 1⟩])

def event192800 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.sum [.transfer 192798, .transfer 192799])

def event192801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], []⟩) [⟨.result 583 .coefficient, true, some 1⟩, ⟨.result 9595 .coefficient, true, some 1⟩])

def event192802 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.sum [.transfer 192800, .transfer 192801])

def event192803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34985⟩⟩], []⟩) [⟨.result 593 .coefficient, true, some 1⟩, ⟨.result 9603 .coefficient, true, some 1⟩])

def event192804 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.sum [.transfer 192802, .transfer 192803])

def event192805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], []⟩) [⟨.result 603 .coefficient, true, some 1⟩, ⟨.result 9611 .coefficient, true, some 1⟩])

def event192806 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.sum [.transfer 192804, .transfer 192805])

def event192807 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26648⟩⟩], []⟩) [⟨.result 613 .coefficient, true, some 1⟩, ⟨.result 9619 .coefficient, true, some 1⟩])

def event192808 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.sum [.transfer 192806, .transfer 192807])

def event192809 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], []⟩) [⟨.result 623 .coefficient, true, some 1⟩, ⟨.result 9627 .coefficient, true, some 1⟩])

def event192810 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.sum [.transfer 192808, .transfer 192809])

def event192811 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], []⟩) [⟨.result 633 .coefficient, true, some 1⟩, ⟨.result 9635 .coefficient, true, some 1⟩])

def event192812 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.sum [.transfer 192810, .transfer 192811])

def event192813 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], []⟩) [⟨.result 643 .coefficient, true, some 1⟩, ⟨.result 9643 .coefficient, true, some 1⟩])

def event192814 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.sum [.transfer 192812, .transfer 192813])

def event192815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57163⟩⟩], []⟩) [⟨.result 653 .coefficient, true, some 1⟩, ⟨.result 9651 .coefficient, true, some 1⟩])

def event192816 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.sum [.transfer 192814, .transfer 192815])

def event192817 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], []⟩) [⟨.result 663 .coefficient, true, some 1⟩, ⟨.result 9659 .coefficient, true, some 1⟩])

def event192818 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.sum [.transfer 192816, .transfer 192817])

def event192819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], []⟩) [⟨.result 673 .coefficient, true, some 1⟩, ⟨.result 9667 .coefficient, true, some 1⟩])

def event192820 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.sum [.transfer 192818, .transfer 192819])

def event192821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], []⟩) [⟨.result 683 .coefficient, true, some 1⟩, ⟨.result 9675 .coefficient, true, some 1⟩])

def event192822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.sum [.transfer 192820, .transfer 192821])

def event192823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], []⟩) [⟨.result 693 .coefficient, true, some 1⟩, ⟨.result 9683 .coefficient, true, some 1⟩])

def event192824 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.sum [.transfer 192822, .transfer 192823])

def event192825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], []⟩) [⟨.result 703 .coefficient, true, some 1⟩, ⟨.result 9691 .coefficient, true, some 1⟩])

def event192826 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.sum [.transfer 192824, .transfer 192825])

def event192827 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], []⟩) [⟨.result 713 .coefficient, true, some 1⟩, ⟨.result 9699 .coefficient, true, some 1⟩])

def event192828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.sum [.transfer 192826, .transfer 192827])

def event192829 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67500⟩⟩) (.product (.result 192788 .summary) (.transfer 192828) (⟨false, false, none, none, none⟩))

def event192830 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67494⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event192831 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48385⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192832 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45705⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192833 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192834 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192835 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192836 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34985⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192837 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192838 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26648⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192839 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 18⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192840 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192841 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192842 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192843 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192844 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192845 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192846 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192847 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192848 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 0⟩, ⟨9780, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192849 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67494⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (-1)⟩)

def event192850 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48385⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def event192851 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45705⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def event192852 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43028⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def event192853 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def event192854 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def event192855 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34985⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def event192856 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def event192857 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26648⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def event192858 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 18⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def event192859 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def event192860 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def event192861 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57163⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def event192862 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def event192863 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def event192864 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def event192865 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def event192866 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def event192867 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67500⟩⟩, .operator (⟨192788, 1⟩, ⟨9780, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩)

def exact192868RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67494⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48385⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45705⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34985⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26648⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63123⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60143⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57163⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54183⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51203⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67494⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32139⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48385⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45705⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43028⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22119⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40348⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34985⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18899⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29328⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26648⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16062⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66728⟩⟩], [⟨.program ⟨257⟩, ⟨7259⟩⟩]⟩, (1)⟩]

theorem exact192868RawTermsValid :
    exact192868RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192868 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67500⟩⟩) exact192868RawTerms .large 192791 (.finite 6902113630329048043564518670336) (some (192829))

def event192869 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68841⟩⟩) 0 ⟨66751⟩ 9552

def event192870 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68841⟩⟩) (.authority (.programFamilyFact))

def event192871 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68841⟩⟩) (.finite 1152)

def event192872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68842⟩⟩) 0 ⟨7177⟩ 15500

def event192873 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68842⟩⟩) 1 ⟨68841⟩ 192871

def event192874 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68842⟩⟩) (.authority (.operator))

def exact192875RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩, (1)⟩]

theorem exact192875RawTermsValid :
    exact192875RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192875 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68842⟩⟩) exact192875RawTerms .large 192874 .exactZero (none)

def event192876 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71297⟩⟩) 0 ⟨68842⟩ 192875

def event192877 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71297⟩⟩) (.authority (.operator))

def exact192878RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩, (1)⟩]

theorem exact192878RawTermsValid :
    exact192878RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192878 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71297⟩⟩) exact192878RawTerms (.finite 8192) 192877 .exactZero (none)

def event192879 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49317⟩⟩) 0 ⟨48165⟩ 9087

def event192880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49317⟩⟩) (.authority (.programFamilyFact))

def event192881 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49317⟩⟩) (.finite 3720)

def event192882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49319⟩⟩) 0 ⟨7177⟩ 15500

def event192883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49319⟩⟩) 1 ⟨49317⟩ 192881

def event192884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49319⟩⟩) (.authority (.operator))

def exact192885RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49319⟩⟩]⟩, (1)⟩]

theorem exact192885RawTermsValid :
    exact192885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192885 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49319⟩⟩) exact192885RawTerms .large 192884 .exactZero (none)

def event192886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50079⟩⟩) 0 ⟨49319⟩ 192885

def event192887 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50079⟩⟩) (.authority (.operator))

def exact192888RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨50079⟩⟩]⟩, (1)⟩]

theorem exact192888RawTermsValid :
    exact192888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192888 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50079⟩⟩) exact192888RawTerms (.finite 8192) 192887 .exactZero (none)

def event192889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49160⟩⟩) 0 ⟨47884⟩ 9081

def event192890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49160⟩⟩) (.authority (.programFamilyFact))

def event192891 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49160⟩⟩) (.finite 3720)

def event192892 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49161⟩⟩) 0 ⟨7177⟩ 15500

def event192893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49161⟩⟩) 1 ⟨49160⟩ 192891

def event192894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49161⟩⟩) (.authority (.operator))

def exact192895RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49161⟩⟩]⟩, (1)⟩]

theorem exact192895RawTermsValid :
    exact192895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49161⟩⟩) exact192895RawTerms .large 192894 .exactZero (none)

def event192896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49681⟩⟩) 0 ⟨49161⟩ 192895

def event192897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49681⟩⟩) (.authority (.operator))

def exact192898RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩, (1)⟩]

theorem exact192898RawTermsValid :
    exact192898RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192898 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49681⟩⟩) exact192898RawTerms (.finite 8192) 192897 .exactZero (none)

def event192899 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6998⟩⟩) 0 ⟨5907⟩ 192773

def event192900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6998⟩⟩) 1 ⟨6908⟩ 2

def event192901 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6998⟩⟩) (.product (.predecessor 0 192899 .coefficient) (.predecessor 1 192900 .coefficient) (⟨false, false, none, none, none⟩))

def event192902 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨6998⟩⟩, .operator (⟨192773, 0⟩, ⟨2, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact192903RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact192903RawTermsValid :
    exact192903RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192903 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6998⟩⟩) exact192903RawTerms .large 192901 .exactZero (none)

def event192904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47885⟩⟩) 0 ⟨47882⟩ 9070

def event192905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47885⟩⟩) 1 ⟨6998⟩ 192903

def event192906 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47885⟩⟩) (.tensor (.predecessor 0 192904 .coefficient) (.predecessor 1 192905 .coefficient) true false)

def event192907 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47885⟩⟩, .operator (⟨9070, 0⟩, ⟨192903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact192908RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact192908RawTermsValid :
    exact192908RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192908 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47885⟩⟩) exact192908RawTerms .large 192906 .exactZero (none)

def event192909 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8819⟩⟩) 0 ⟨5907⟩ 192773

def event192910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8819⟩⟩) 1 ⟨7285⟩ 17065

def event192911 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8819⟩⟩) (.product (.predecessor 0 192909 .coefficient) (.predecessor 1 192910 .coefficient) (⟨false, false, none, none, none⟩))

def event192912 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8819⟩⟩, .operator (⟨192773, 0⟩, ⟨17065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩)

def exact192913RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩]

theorem exact192913RawTermsValid :
    exact192913RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192913 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8819⟩⟩) exact192913RawTerms .large 192911 .exactZero (none)

def event192914 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47886⟩⟩) 0 ⟨8819⟩ 192913

def event192915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47886⟩⟩) 1 ⟨47885⟩ 192908

def event192916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47886⟩⟩) (.sum [.predecessor 0 192914 .coefficient, .predecessor 1 192915 .coefficient])

def exact192917RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192917RawTermsValid :
    exact192917RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192917 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47886⟩⟩) exact192917RawTerms .large 192916 .exactZero (none)

def event192918 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47887⟩⟩) 0 ⟨47886⟩ 192917

def event192919 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47887⟩⟩) 1 ⟨111⟩ 17052

def event192920 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47887⟩⟩) (.sum [.predecessor 0 192918 .coefficient, .predecessor 1 192919 .coefficient])

def event192921 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47887⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨111⟩⟩]⟩) [⟨.result 17052 .coefficient, false, none⟩])

def event192922 : Event := .survivorFold (1) 192921

def exact192923RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192923RawTermsValid :
    exact192923RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192923 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47887⟩⟩) exact192923RawTerms .large 192920 (.finite 26) (some (192921))

def event192924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47888⟩⟩) 0 ⟨47887⟩ 192923

def event192925 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47888⟩⟩) 1 ⟨15111⟩ 9073

def event192926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47888⟩⟩) (.product (.predecessor 0 192924 .coefficient) (.predecessor 1 192925 .coefficient) (⟨false, true, none, none, some 1⟩))

def event192927 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47888⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩], []⟩) [⟨.result 9073 .coefficient, true, some 1⟩])

def event192928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47888⟩⟩) (.product (.result 192923 .summary) (.transfer 192927) (⟨false, false, none, none, none⟩))

def event192929 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47888⟩⟩, .operator (⟨192923, 1⟩, ⟨9073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event192930 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47888⟩⟩, .operator (⟨192923, 0⟩, ⟨9073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩)

def exact192931RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192931RawTermsValid :
    exact192931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192931 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47888⟩⟩) exact192931RawTerms .large 192926 (.finite 51118080) (some (192928))

def event192932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15112⟩⟩) 0 ⟨15111⟩ 9073

def event192933 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15112⟩⟩) 1 ⟨6998⟩ 192903

def event192934 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15112⟩⟩) (.tensor (.predecessor 0 192932 .coefficient) (.predecessor 1 192933 .coefficient) true false)

def event192935 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15112⟩⟩, .operator (⟨9073, 0⟩, ⟨192903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact192936RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact192936RawTermsValid :
    exact192936RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192936 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15112⟩⟩) exact192936RawTerms .large 192934 .exactZero (none)

def event192937 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8836⟩⟩) 0 ⟨5907⟩ 192773

def event192938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8836⟩⟩) 1 ⟨7302⟩ 17106

def event192939 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8836⟩⟩) (.product (.predecessor 0 192937 .coefficient) (.predecessor 1 192938 .coefficient) (⟨false, false, none, none, none⟩))

def event192940 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8836⟩⟩, .operator (⟨192773, 0⟩, ⟨17106, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩)

def exact192941RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩]

theorem exact192941RawTermsValid :
    exact192941RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192941 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8836⟩⟩) exact192941RawTerms .large 192939 .exactZero (none)

def event192942 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15113⟩⟩) 0 ⟨8836⟩ 192941

def event192943 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15113⟩⟩) 1 ⟨15112⟩ 192936

def event192944 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15113⟩⟩) (.sum [.predecessor 0 192942 .coefficient, .predecessor 1 192943 .coefficient])

def exact192945RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192945RawTermsValid :
    exact192945RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192945 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15113⟩⟩) exact192945RawTerms .large 192944 .exactZero (none)

def event192946 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15114⟩⟩) 0 ⟨15113⟩ 192945

def event192947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15114⟩⟩) 1 ⟨128⟩ 17098

def event192948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15114⟩⟩) (.sum [.predecessor 0 192946 .coefficient, .predecessor 1 192947 .coefficient])

def event192949 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15114⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨128⟩⟩]⟩) [⟨.result 17098 .coefficient, false, none⟩])

def event192950 : Event := .survivorFold (1) 192949

def exact192951RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192951RawTermsValid :
    exact192951RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192951 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15114⟩⟩) exact192951RawTerms .large 192948 (.finite 26) (some (192949))

def event192952 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15115⟩⟩) 0 ⟨15114⟩ 192951

def event192953 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15115⟩⟩) 1 ⟨9566⟩ 17095

def event192954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15115⟩⟩) (.product (.predecessor 0 192952 .coefficient) (.predecessor 1 192953 .coefficient) (⟨false, false, none, none, none⟩))

def event192955 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15115⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) [⟨.result 17091 .coefficient, false, none⟩])

def event192956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15115⟩⟩) (.product (.result 192951 .summary) (.transfer 192955) (⟨false, false, none, none, none⟩))

def event192957 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15115⟩⟩, .operator (⟨192951, 1⟩, ⟨17095, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (-1)⟩)

def event192958 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨15115⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9565⟩⟩) ⟨7285⟩ 17065)

def event192959 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15115⟩⟩, .relation 192958 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (-1)⟩)

def event192960 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15115⟩⟩, .operator (⟨192951, 0⟩, ⟨17095, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩)

def exact192961RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (-1)⟩]

theorem exact192961RawTermsValid :
    exact192961RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192961 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15115⟩⟩) exact192961RawTerms .large 192954 (.finite 279172874240) (some (192956))

def event192962 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47889⟩⟩) 0 ⟨15115⟩ 192961

def event192963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47889⟩⟩) 1 ⟨47888⟩ 192931

def event192964 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47889⟩⟩) (.sum [.predecessor 0 192962 .coefficient, .predecessor 1 192963 .coefficient])

def event192965 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47889⟩⟩, .operator (⟨192961, 1⟩, ⟨192931, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩)

def event192966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47889⟩⟩) (.sum [.result 192961 .summary, .result 192931 .summary])

def exact192967RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact192967RawTermsValid :
    exact192967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192967 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47889⟩⟩) exact192967RawTerms .large 192964 (.finite 279223992320) (some (192966))

def event192968 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49682⟩⟩) 0 ⟨47889⟩ 192967

def event192969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49682⟩⟩) 1 ⟨49681⟩ 192898

def event192970 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49682⟩⟩) (.product (.predecessor 0 192968 .coefficient) (.predecessor 1 192969 .coefficient) (⟨false, false, none, none, none⟩))

def event192971 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49682⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩) [⟨.result 192898 .coefficient, false, none⟩])

def event192972 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49682⟩⟩) (.product (.result 192967 .summary) (.transfer 192971) (⟨false, false, none, none, none⟩))

def event192973 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49682⟩⟩, .operator (⟨192967, 1⟩, ⟨192898, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩, (-1)⟩)

def event192974 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49682⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49681⟩⟩) ⟨49161⟩ 192895)

def event192975 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49682⟩⟩, .relation 192974 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨49161⟩⟩]⟩, (-1)⟩)

def event192976 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49682⟩⟩, .operator (⟨192967, 0⟩, ⟨192898, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩, (1)⟩)

def exact192977RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨49161⟩⟩]⟩, (-1)⟩]

theorem exact192977RawTermsValid :
    exact192977RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192977 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49682⟩⟩) exact192977RawTerms .large 192970 (.finite 2998144788182387916800) (some (192972))

def event192978 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48609⟩⟩) 0 ⟨47884⟩ 9081

def event192979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48609⟩⟩) (.authority (.relationPreimageSource ⟨54⟩))

def exact192980RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48609⟩⟩]⟩, (1)⟩]

theorem exact192980RawTermsValid :
    exact192980RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192980 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48609⟩⟩) exact192980RawTerms (.finite 5647228698) 192979 .exactZero (none)

def event192981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48611⟩⟩) 0 ⟨48609⟩ 192980

def event192982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48611⟩⟩) 1 ⟨2370⟩ 4

def event192983 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48611⟩⟩) (.scale (.predecessor 0 192981 .coefficient) (.value (.predecessor 1 192982 .coefficient)))

def exact192984RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48609⟩⟩]⟩, (1)⟩]

theorem exact192984RawTermsValid :
    exact192984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192984 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48611⟩⟩) exact192984RawTerms (.finite 5647228698) 192983 .exactZero (none)

def event192985 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5908⟩⟩) 0 ⟨5907⟩ 192773

def event192986 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5908⟩⟩) 1 ⟨35⟩ 17158

def event192987 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5908⟩⟩) (.product (.predecessor 0 192985 .coefficient) (.predecessor 1 192986 .coefficient) (⟨false, false, none, none, none⟩))

def event192988 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨5908⟩⟩, .operator (⟨192773, 0⟩, ⟨17158, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩)

def exact192989RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact192989RawTermsValid :
    exact192989RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192989 : Event := .resultExact (⟨.program ⟨257⟩, ⟨5908⟩⟩) exact192989RawTerms .large 192987 .exactZero (none)

def event192990 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5909⟩⟩) 0 ⟨5908⟩ 192989

def event192991 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5909⟩⟩) 1 ⟨22⟩ 17156

def event192992 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5909⟩⟩) (.sum [.predecessor 0 192990 .coefficient, .predecessor 1 192991 .coefficient])

def event192993 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5909⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨22⟩⟩]⟩) [⟨.result 17156 .coefficient, false, none⟩])

def event192994 : Event := .survivorFold (1) 192993

def exact192995RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact192995RawTermsValid :
    exact192995RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event192995 : Event := .resultExact (⟨.program ⟨257⟩, ⟨5909⟩⟩) exact192995RawTerms .large 192992 (.finite 26) (some (192993))

def event192996 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48612⟩⟩) 0 ⟨5909⟩ 192995

def event192997 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48612⟩⟩) 1 ⟨48611⟩ 192984

def event192998 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48612⟩⟩) (.product (.predecessor 0 192996 .coefficient) (.predecessor 1 192997 .coefficient) (⟨false, false, none, none, none⟩))

def event192999 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48612⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨48609⟩⟩]⟩) [⟨.result 192980 .coefficient, false, none⟩])

def event193000 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48612⟩⟩) (.product (.result 192995 .summary) (.transfer 192999) (⟨false, false, none, none, none⟩))

def event193001 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48612⟩⟩, .operator (⟨192995, 0⟩, ⟨192984, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48609⟩⟩]⟩, (1)⟩)

def event193002 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨48610⟩⟩)

def event193003 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event193004 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event193005 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5893⟩⟩) (.authority (.operator))

def event193006 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5893⟩⟩) (.finite 7)

def event193007 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event193008 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event193009 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event193010 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event193011 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 193010

def event193012 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 193008

def event193013 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 193011 .coefficient) (.value (.predecessor 1 193012 .coefficient)))

def event193014 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event193015 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5895⟩⟩) 0 ⟨392⟩ 193014

def event193016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5895⟩⟩) 1 ⟨5893⟩ 193006

def event193017 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5895⟩⟩) (.sum [.predecessor 0 193015 .coefficient, .predecessor 1 193016 .coefficient])

def event193018 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5895⟩⟩) (.finite 655347)

def event193019 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5905⟩⟩) 0 ⟨5895⟩ 193018

def event193020 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5905⟩⟩) 1 ⟨5426⟩ 193004

def event193021 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5905⟩⟩) (.identity (.predecessor 1 193020 .coefficient))

def event193022 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5905⟩⟩) (.finite 655360)

def event193023 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47882⟩⟩) 0 ⟨5905⟩ 193022

def eventLeaf12048 : Array AnnotatedEvent := #[
  { event := event192768
    frameStart := 0 },
  { event := event192769
    frameStart := 0 },
  { event := event192770
    frameStart := 0 },
  { event := event192771
    frameStart := 0 },
  { event := event192772
    frameStart := 0 },
  { event := event192773
    frameStart := 0 },
  { event := event192774
    frameStart := 0 },
  { event := event192775
    frameStart := 0 },
  { event := event192776
    frameStart := 0 },
  { event := event192777
    frameStart := 0 },
  { event := event192778
    frameStart := 0 },
  { event := event192779
    frameStart := 0 },
  { event := event192780
    frameStart := 0 },
  { event := event192781
    frameStart := 0 },
  { event := event192782
    frameStart := 0 },
  { event := event192783
    frameStart := 0 }
]

def eventLeaf12049 : Array AnnotatedEvent := #[
  { event := event192784
    frameStart := 0 },
  { event := event192785
    frameStart := 0 },
  { event := event192786
    frameStart := 0 },
  { event := event192787
    frameStart := 0 },
  { event := event192788
    frameStart := 0 },
  { event := event192789
    frameStart := 0 },
  { event := event192790
    frameStart := 0 },
  { event := event192791
    frameStart := 0 },
  { event := event192792
    frameStart := 0 },
  { event := event192793
    frameStart := 0 },
  { event := event192794
    frameStart := 0 },
  { event := event192795
    frameStart := 0 },
  { event := event192796
    frameStart := 0 },
  { event := event192797
    frameStart := 0 },
  { event := event192798
    frameStart := 0 },
  { event := event192799
    frameStart := 0 }
]

def eventLeaf12050 : Array AnnotatedEvent := #[
  { event := event192800
    frameStart := 0 },
  { event := event192801
    frameStart := 0 },
  { event := event192802
    frameStart := 0 },
  { event := event192803
    frameStart := 0 },
  { event := event192804
    frameStart := 0 },
  { event := event192805
    frameStart := 0 },
  { event := event192806
    frameStart := 0 },
  { event := event192807
    frameStart := 0 },
  { event := event192808
    frameStart := 0 },
  { event := event192809
    frameStart := 0 },
  { event := event192810
    frameStart := 0 },
  { event := event192811
    frameStart := 0 },
  { event := event192812
    frameStart := 0 },
  { event := event192813
    frameStart := 0 },
  { event := event192814
    frameStart := 0 },
  { event := event192815
    frameStart := 0 }
]

def eventLeaf12051 : Array AnnotatedEvent := #[
  { event := event192816
    frameStart := 0 },
  { event := event192817
    frameStart := 0 },
  { event := event192818
    frameStart := 0 },
  { event := event192819
    frameStart := 0 },
  { event := event192820
    frameStart := 0 },
  { event := event192821
    frameStart := 0 },
  { event := event192822
    frameStart := 0 },
  { event := event192823
    frameStart := 0 },
  { event := event192824
    frameStart := 0 },
  { event := event192825
    frameStart := 0 },
  { event := event192826
    frameStart := 0 },
  { event := event192827
    frameStart := 0 },
  { event := event192828
    frameStart := 0 },
  { event := event192829
    frameStart := 0 },
  { event := event192830
    frameStart := 0 },
  { event := event192831
    frameStart := 0 }
]

def eventLeaf12052 : Array AnnotatedEvent := #[
  { event := event192832
    frameStart := 0 },
  { event := event192833
    frameStart := 0 },
  { event := event192834
    frameStart := 0 },
  { event := event192835
    frameStart := 0 },
  { event := event192836
    frameStart := 0 },
  { event := event192837
    frameStart := 0 },
  { event := event192838
    frameStart := 0 },
  { event := event192839
    frameStart := 0 },
  { event := event192840
    frameStart := 0 },
  { event := event192841
    frameStart := 0 },
  { event := event192842
    frameStart := 0 },
  { event := event192843
    frameStart := 0 },
  { event := event192844
    frameStart := 0 },
  { event := event192845
    frameStart := 0 },
  { event := event192846
    frameStart := 0 },
  { event := event192847
    frameStart := 0 }
]

def eventLeaf12053 : Array AnnotatedEvent := #[
  { event := event192848
    frameStart := 0 },
  { event := event192849
    frameStart := 0 },
  { event := event192850
    frameStart := 0 },
  { event := event192851
    frameStart := 0 },
  { event := event192852
    frameStart := 0 },
  { event := event192853
    frameStart := 0 },
  { event := event192854
    frameStart := 0 },
  { event := event192855
    frameStart := 0 },
  { event := event192856
    frameStart := 0 },
  { event := event192857
    frameStart := 0 },
  { event := event192858
    frameStart := 0 },
  { event := event192859
    frameStart := 0 },
  { event := event192860
    frameStart := 0 },
  { event := event192861
    frameStart := 0 },
  { event := event192862
    frameStart := 0 },
  { event := event192863
    frameStart := 0 }
]

def eventLeaf12054 : Array AnnotatedEvent := #[
  { event := event192864
    frameStart := 0 },
  { event := event192865
    frameStart := 0 },
  { event := event192866
    frameStart := 0 },
  { event := event192867
    frameStart := 0 },
  { event := event192868
    frameStart := 0 },
  { event := event192869
    frameStart := 0 },
  { event := event192870
    frameStart := 0 },
  { event := event192871
    frameStart := 0 },
  { event := event192872
    frameStart := 0 },
  { event := event192873
    frameStart := 0 },
  { event := event192874
    frameStart := 0 },
  { event := event192875
    frameStart := 0 },
  { event := event192876
    frameStart := 0 },
  { event := event192877
    frameStart := 0 },
  { event := event192878
    frameStart := 0 },
  { event := event192879
    frameStart := 0 }
]

def eventLeaf12055 : Array AnnotatedEvent := #[
  { event := event192880
    frameStart := 0 },
  { event := event192881
    frameStart := 0 },
  { event := event192882
    frameStart := 0 },
  { event := event192883
    frameStart := 0 },
  { event := event192884
    frameStart := 0 },
  { event := event192885
    frameStart := 0 },
  { event := event192886
    frameStart := 0 },
  { event := event192887
    frameStart := 0 },
  { event := event192888
    frameStart := 0 },
  { event := event192889
    frameStart := 0 },
  { event := event192890
    frameStart := 0 },
  { event := event192891
    frameStart := 0 },
  { event := event192892
    frameStart := 0 },
  { event := event192893
    frameStart := 0 },
  { event := event192894
    frameStart := 0 },
  { event := event192895
    frameStart := 0 }
]

def eventLeaf12056 : Array AnnotatedEvent := #[
  { event := event192896
    frameStart := 0 },
  { event := event192897
    frameStart := 0 },
  { event := event192898
    frameStart := 0 },
  { event := event192899
    frameStart := 0 },
  { event := event192900
    frameStart := 0 },
  { event := event192901
    frameStart := 0 },
  { event := event192902
    frameStart := 0 },
  { event := event192903
    frameStart := 0 },
  { event := event192904
    frameStart := 0 },
  { event := event192905
    frameStart := 0 },
  { event := event192906
    frameStart := 0 },
  { event := event192907
    frameStart := 0 },
  { event := event192908
    frameStart := 0 },
  { event := event192909
    frameStart := 0 },
  { event := event192910
    frameStart := 0 },
  { event := event192911
    frameStart := 0 }
]

def eventLeaf12057 : Array AnnotatedEvent := #[
  { event := event192912
    frameStart := 0 },
  { event := event192913
    frameStart := 0 },
  { event := event192914
    frameStart := 0 },
  { event := event192915
    frameStart := 0 },
  { event := event192916
    frameStart := 0 },
  { event := event192917
    frameStart := 0 },
  { event := event192918
    frameStart := 0 },
  { event := event192919
    frameStart := 0 },
  { event := event192920
    frameStart := 0 },
  { event := event192921
    frameStart := 0 },
  { event := event192922
    frameStart := 0 },
  { event := event192923
    frameStart := 0 },
  { event := event192924
    frameStart := 0 },
  { event := event192925
    frameStart := 0 },
  { event := event192926
    frameStart := 0 },
  { event := event192927
    frameStart := 0 }
]

def eventLeaf12058 : Array AnnotatedEvent := #[
  { event := event192928
    frameStart := 0 },
  { event := event192929
    frameStart := 0 },
  { event := event192930
    frameStart := 0 },
  { event := event192931
    frameStart := 0 },
  { event := event192932
    frameStart := 0 },
  { event := event192933
    frameStart := 0 },
  { event := event192934
    frameStart := 0 },
  { event := event192935
    frameStart := 0 },
  { event := event192936
    frameStart := 0 },
  { event := event192937
    frameStart := 0 },
  { event := event192938
    frameStart := 0 },
  { event := event192939
    frameStart := 0 },
  { event := event192940
    frameStart := 0 },
  { event := event192941
    frameStart := 0 },
  { event := event192942
    frameStart := 0 },
  { event := event192943
    frameStart := 0 }
]

def eventLeaf12059 : Array AnnotatedEvent := #[
  { event := event192944
    frameStart := 0 },
  { event := event192945
    frameStart := 0 },
  { event := event192946
    frameStart := 0 },
  { event := event192947
    frameStart := 0 },
  { event := event192948
    frameStart := 0 },
  { event := event192949
    frameStart := 0 },
  { event := event192950
    frameStart := 0 },
  { event := event192951
    frameStart := 0 },
  { event := event192952
    frameStart := 0 },
  { event := event192953
    frameStart := 0 },
  { event := event192954
    frameStart := 0 },
  { event := event192955
    frameStart := 0 },
  { event := event192956
    frameStart := 0 },
  { event := event192957
    frameStart := 0 },
  { event := event192958
    frameStart := 0 },
  { event := event192959
    frameStart := 0 }
]

def eventLeaf12060 : Array AnnotatedEvent := #[
  { event := event192960
    frameStart := 0 },
  { event := event192961
    frameStart := 0 },
  { event := event192962
    frameStart := 0 },
  { event := event192963
    frameStart := 0 },
  { event := event192964
    frameStart := 0 },
  { event := event192965
    frameStart := 0 },
  { event := event192966
    frameStart := 0 },
  { event := event192967
    frameStart := 0 },
  { event := event192968
    frameStart := 0 },
  { event := event192969
    frameStart := 0 },
  { event := event192970
    frameStart := 0 },
  { event := event192971
    frameStart := 0 },
  { event := event192972
    frameStart := 0 },
  { event := event192973
    frameStart := 0 },
  { event := event192974
    frameStart := 0 },
  { event := event192975
    frameStart := 0 }
]

def eventLeaf12061 : Array AnnotatedEvent := #[
  { event := event192976
    frameStart := 0 },
  { event := event192977
    frameStart := 0 },
  { event := event192978
    frameStart := 0 },
  { event := event192979
    frameStart := 0 },
  { event := event192980
    frameStart := 0 },
  { event := event192981
    frameStart := 0 },
  { event := event192982
    frameStart := 0 },
  { event := event192983
    frameStart := 0 },
  { event := event192984
    frameStart := 0 },
  { event := event192985
    frameStart := 0 },
  { event := event192986
    frameStart := 0 },
  { event := event192987
    frameStart := 0 },
  { event := event192988
    frameStart := 0 },
  { event := event192989
    frameStart := 0 },
  { event := event192990
    frameStart := 0 },
  { event := event192991
    frameStart := 0 }
]

def eventLeaf12062 : Array AnnotatedEvent := #[
  { event := event192992
    frameStart := 0 },
  { event := event192993
    frameStart := 0 },
  { event := event192994
    frameStart := 0 },
  { event := event192995
    frameStart := 0 },
  { event := event192996
    frameStart := 0 },
  { event := event192997
    frameStart := 0 },
  { event := event192998
    frameStart := 0 },
  { event := event192999
    frameStart := 0 },
  { event := event193000
    frameStart := 0 },
  { event := event193001
    frameStart := 0 },
  { event := event193002
    frameStart := 193002 },
  { event := event193003
    frameStart := 193002 },
  { event := event193004
    frameStart := 193002 },
  { event := event193005
    frameStart := 193002 },
  { event := event193006
    frameStart := 193002 },
  { event := event193007
    frameStart := 193002 }
]

def eventLeaf12063 : Array AnnotatedEvent := #[
  { event := event193008
    frameStart := 193002 },
  { event := event193009
    frameStart := 193002 },
  { event := event193010
    frameStart := 193002 },
  { event := event193011
    frameStart := 193002 },
  { event := event193012
    frameStart := 193002 },
  { event := event193013
    frameStart := 193002 },
  { event := event193014
    frameStart := 193002 },
  { event := event193015
    frameStart := 193002 },
  { event := event193016
    frameStart := 193002 },
  { event := event193017
    frameStart := 193002 },
  { event := event193018
    frameStart := 193002 },
  { event := event193019
    frameStart := 193002 },
  { event := event193020
    frameStart := 193002 },
  { event := event193021
    frameStart := 193002 },
  { event := event193022
    frameStart := 193002 },
  { event := event193023
    frameStart := 193002 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events753
