import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1128

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event288768 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20467⟩⟩, .relation 288767 0, ⟨[⟨.program ⟨257⟩, ⟨18540⟩⟩], [⟨.program ⟨257⟩, ⟨19807⟩⟩]⟩, (-1)⟩)

def exact288769RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18540⟩⟩], [⟨.program ⟨257⟩, ⟨19807⟩⟩]⟩, (-1)⟩]

theorem exact288769RawTermsValid :
    exact288769RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288769 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20467⟩⟩) exact288769RawTerms .large 288764 .exactZero (none)

def event288770 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18752⟩⟩) 0 ⟨18541⟩ 288727

def event288771 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18752⟩⟩) (.authority (.programFamilyFact))

def exact288772RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], []⟩, (1)⟩]

theorem exact288772RawTermsValid :
    exact288772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288772 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18752⟩⟩) exact288772RawTerms (.finite 48) 288771 .exactZero (none)

def event288773 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18754⟩⟩) 0 ⟨6908⟩ 288749

def event288774 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18754⟩⟩) 1 ⟨18752⟩ 288772

def event288775 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18754⟩⟩) (.product (.predecessor 0 288773 .coefficient) (.predecessor 1 288774 .coefficient) (⟨false, true, none, none, some 1⟩))

def event288776 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18754⟩⟩, .operator (⟨288749, 0⟩, ⟨288772, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact288777RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact288777RawTermsValid :
    exact288777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288777 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18754⟩⟩) exact288777RawTerms .large 288775 .exactZero (none)

def event288778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7200⟩⟩) 0 ⟨7177⟩ 288731

def event288779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7200⟩⟩) (.authority (.operator))

def exact288780RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩]

theorem exact288780RawTermsValid :
    exact288780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288780 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7200⟩⟩) exact288780RawTerms .large 288779 .exactZero (none)

def event288781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18755⟩⟩) 0 ⟨7200⟩ 288780

def event288782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18755⟩⟩) 1 ⟨18754⟩ 288777

def event288783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18755⟩⟩) (.sum [.predecessor 0 288781 .coefficient, .predecessor 1 288782 .coefficient])

def exact288784RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact288784RawTermsValid :
    exact288784RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288784 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18755⟩⟩) exact288784RawTerms .large 288783 .exactZero (none)

def event288785 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20471⟩⟩) 0 ⟨18755⟩ 288784

def event288786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20471⟩⟩) 1 ⟨20467⟩ 288769

def event288787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20471⟩⟩) (.sum [.predecessor 0 288785 .coefficient, .predecessor 1 288786 .coefficient])

def exact288788RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18540⟩⟩], [⟨.program ⟨257⟩, ⟨19807⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact288788RawTermsValid :
    exact288788RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288788 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20471⟩⟩) exact288788RawTerms .large 288787 .exactZero (none)

def event288789 : Event := .preFoldPolynomial 288788 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18540⟩⟩], [⟨.program ⟨257⟩, ⟨19807⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact288790RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18540⟩⟩], [⟨.program ⟨257⟩, ⟨19807⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event288790 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨20471⟩⟩) 288789 exact288790RawTerms .large 288787 .exactZero (none)

def event288791 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨18541⟩⟩) ⟨⟨79⟩, ⟨59⟩, ⟨135⟩⟩ ⟨288633, 288791⟩

def event288792 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨19339⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19336⟩⟩]⟩) (1) 0 2 (.universal 288791 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19336⟩⟩]⟩) (none) 288790)

def event288793 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19339⟩⟩, .relation 288792 1, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩)

def event288794 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19339⟩⟩, .relation 288792 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩]⟩, (-1)⟩)

def event288795 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19339⟩⟩, .relation 288792 2, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18540⟩⟩], [⟨.program ⟨257⟩, ⟨19807⟩⟩]⟩, (1)⟩)

def event288796 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨19339⟩⟩, .relation 288792 3, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact288797RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18540⟩⟩], [⟨.program ⟨257⟩, ⟨19807⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact288797RawTermsValid :
    exact288797RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288797 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19339⟩⟩) exact288797RawTerms .large 288629 (.finite 202072841853861888) (some (288631))

def event288798 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20469⟩⟩) 0 ⟨19339⟩ 288797

def event288799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20469⟩⟩) 1 ⟨20468⟩ 288619

def event288800 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20469⟩⟩) (.sum [.predecessor 0 288798 .coefficient, .predecessor 1 288799 .coefficient])

def event288801 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20469⟩⟩, .operator (⟨288797, 0⟩, ⟨288619, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩]⟩, (1)⟩)

def event288802 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨20469⟩⟩, .operator (⟨288797, 2⟩, ⟨288619, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18540⟩⟩], [⟨.program ⟨257⟩, ⟨19807⟩⟩]⟩, (-1)⟩)

def event288803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20469⟩⟩) (.sum [.result 288797 .summary, .result 288619 .summary])

def exact288804RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact288804RawTermsValid :
    exact288804RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288804 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20469⟩⟩) exact288804RawTerms .large 288800 (.finite 32188905437706550578131070353408) (some (288803))

def event288805 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16945⟩⟩) 0 ⟨15741⟩ 13960

def event288806 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16945⟩⟩) (.authority (.programFamilyFact))

def event288807 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨16945⟩⟩) (.finite 3720)

def event288808 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16947⟩⟩) 0 ⟨7177⟩ 15500

def event288809 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16947⟩⟩) 1 ⟨16945⟩ 288807

def event288810 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16947⟩⟩) (.authority (.operator))

def exact288811RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16947⟩⟩]⟩, (1)⟩]

theorem exact288811RawTermsValid :
    exact288811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288811 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16947⟩⟩) exact288811RawTerms .large 288810 .exactZero (none)

def event288812 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17593⟩⟩) 0 ⟨16947⟩ 288811

def event288813 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17593⟩⟩) (.authority (.operator))

def exact288814RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17593⟩⟩]⟩, (1)⟩]

theorem exact288814RawTermsValid :
    exact288814RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288814 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17593⟩⟩) exact288814RawTerms (.finite 8192) 288813 .exactZero (none)

def event288815 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16812⟩⟩) 0 ⟨15332⟩ 13954

def event288816 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16812⟩⟩) (.authority (.programFamilyFact))

def event288817 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨16812⟩⟩) (.finite 3720)

def event288818 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16813⟩⟩) 0 ⟨7177⟩ 15500

def event288819 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16813⟩⟩) 1 ⟨16812⟩ 288817

def event288820 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16813⟩⟩) (.authority (.operator))

def exact288821RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16813⟩⟩]⟩, (1)⟩]

theorem exact288821RawTermsValid :
    exact288821RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288821 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16813⟩⟩) exact288821RawTerms .large 288820 .exactZero (none)

def event288822 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17293⟩⟩) 0 ⟨16813⟩ 288821

def event288823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17293⟩⟩) (.authority (.operator))

def exact288824RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17293⟩⟩]⟩, (1)⟩]

theorem exact288824RawTermsValid :
    exact288824RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288824 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17293⟩⟩) exact288824RawTerms (.finite 8192) 288823 .exactZero (none)

def event288825 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15333⟩⟩) 0 ⟨15330⟩ 13943

def event288826 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15333⟩⟩) 1 ⟨6922⟩ 280653

def event288827 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15333⟩⟩) (.tensor (.predecessor 0 288825 .coefficient) (.predecessor 1 288826 .coefficient) true false)

def event288828 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15333⟩⟩, .operator (⟨13943, 0⟩, ⟨280653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact288829RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact288829RawTermsValid :
    exact288829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288829 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15333⟩⟩) exact288829RawTerms .large 288827 .exactZero (none)

def event288830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7926⟩⟩) 0 ⟨5489⟩ 280523

def event288831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7926⟩⟩) 1 ⟨7304⟩ 25597

def event288832 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7926⟩⟩) (.product (.predecessor 0 288830 .coefficient) (.predecessor 1 288831 .coefficient) (⟨false, false, none, none, none⟩))

def event288833 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7926⟩⟩, .operator (⟨280523, 0⟩, ⟨25597, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩)

def exact288834RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩]

theorem exact288834RawTermsValid :
    exact288834RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288834 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7926⟩⟩) exact288834RawTerms .large 288832 .exactZero (none)

def event288835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15334⟩⟩) 0 ⟨7926⟩ 288834

def event288836 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15334⟩⟩) 1 ⟨15333⟩ 288829

def event288837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15334⟩⟩) (.sum [.predecessor 0 288835 .coefficient, .predecessor 1 288836 .coefficient])

def exact288838RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact288838RawTermsValid :
    exact288838RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288838 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15334⟩⟩) exact288838RawTerms .large 288837 .exactZero (none)

def event288839 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15335⟩⟩) 0 ⟨15334⟩ 288838

def event288840 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15335⟩⟩) 1 ⟨130⟩ 25589

def event288841 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15335⟩⟩) (.sum [.predecessor 0 288839 .coefficient, .predecessor 1 288840 .coefficient])

def event288842 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15335⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨130⟩⟩]⟩) [⟨.result 25589 .coefficient, false, none⟩])

def event288843 : Event := .survivorFold (1) 288842

def exact288844RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact288844RawTermsValid :
    exact288844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288844 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15335⟩⟩) exact288844RawTerms .large 288841 (.finite 26) (some (288842))

def event288845 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15336⟩⟩) 0 ⟨15335⟩ 288844

def event288846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15336⟩⟩) 1 ⟨12291⟩ 13946

def event288847 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15336⟩⟩) (.product (.predecessor 0 288845 .coefficient) (.predecessor 1 288846 .coefficient) (⟨false, true, none, none, some 1⟩))

def event288848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15336⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12291⟩⟩], []⟩) [⟨.result 13946 .coefficient, true, some 1⟩])

def event288849 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15336⟩⟩) (.product (.result 288844 .summary) (.transfer 288848) (⟨false, false, none, none, none⟩))

def event288850 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15336⟩⟩, .operator (⟨288844, 1⟩, ⟨13946, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event288851 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15336⟩⟩, .operator (⟨288844, 0⟩, ⟨13946, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩)

def exact288852RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact288852RawTermsValid :
    exact288852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288852 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15336⟩⟩) exact288852RawTerms .large 288847 (.finite 1703936) (some (288849))

def event288853 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12292⟩⟩) 0 ⟨12291⟩ 13946

def event288854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12292⟩⟩) 1 ⟨6922⟩ 280653

def event288855 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12292⟩⟩) (.tensor (.predecessor 0 288853 .coefficient) (.predecessor 1 288854 .coefficient) true false)

def event288856 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨12292⟩⟩, .operator (⟨13946, 0⟩, ⟨280653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact288857RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact288857RawTermsValid :
    exact288857RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288857 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12292⟩⟩) exact288857RawTerms .large 288855 .exactZero (none)

def event288858 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7925⟩⟩) 0 ⟨5489⟩ 280523

def event288859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7925⟩⟩) 1 ⟨7303⟩ 25638

def event288860 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7925⟩⟩) (.product (.predecessor 0 288858 .coefficient) (.predecessor 1 288859 .coefficient) (⟨false, false, none, none, none⟩))

def event288861 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7925⟩⟩, .operator (⟨280523, 0⟩, ⟨25638, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩]⟩, (1)⟩)

def exact288862RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩]⟩, (1)⟩]

theorem exact288862RawTermsValid :
    exact288862RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288862 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7925⟩⟩) exact288862RawTerms .large 288860 .exactZero (none)

def event288863 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12293⟩⟩) 0 ⟨7925⟩ 288862

def event288864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12293⟩⟩) 1 ⟨12292⟩ 288857

def event288865 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12293⟩⟩) (.sum [.predecessor 0 288863 .coefficient, .predecessor 1 288864 .coefficient])

def exact288866RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact288866RawTermsValid :
    exact288866RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288866 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12293⟩⟩) exact288866RawTerms .large 288865 .exactZero (none)

def event288867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12294⟩⟩) 0 ⟨12293⟩ 288866

def event288868 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12294⟩⟩) 1 ⟨129⟩ 25630

def event288869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12294⟩⟩) (.sum [.predecessor 0 288867 .coefficient, .predecessor 1 288868 .coefficient])

def event288870 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12294⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨129⟩⟩]⟩) [⟨.result 25630 .coefficient, false, none⟩])

def event288871 : Event := .survivorFold (1) 288870

def exact288872RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact288872RawTermsValid :
    exact288872RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288872 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12294⟩⟩) exact288872RawTerms .large 288869 (.finite 26) (some (288870))

def event288873 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12295⟩⟩) 0 ⟨12294⟩ 288872

def event288874 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12295⟩⟩) 1 ⟨9569⟩ 25627

def event288875 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12295⟩⟩) (.product (.predecessor 0 288873 .coefficient) (.predecessor 1 288874 .coefficient) (⟨false, false, none, none, none⟩))

def event288876 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12295⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) [⟨.result 25623 .coefficient, false, none⟩])

def event288877 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12295⟩⟩) (.product (.result 288872 .summary) (.transfer 288876) (⟨false, false, none, none, none⟩))

def event288878 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨12295⟩⟩, .operator (⟨288872, 1⟩, ⟨25627, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (-1)⟩)

def event288879 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨12295⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9568⟩⟩) ⟨7304⟩ 25597)

def event288880 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨12295⟩⟩, .relation 288879 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (-1)⟩)

def event288881 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨12295⟩⟩, .operator (⟨288872, 0⟩, ⟨25627, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩)

def exact288882RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (-1)⟩]

theorem exact288882RawTermsValid :
    exact288882RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288882 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12295⟩⟩) exact288882RawTerms .large 288875 (.finite 279172874240) (some (288877))

def event288883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15337⟩⟩) 0 ⟨12295⟩ 288882

def event288884 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15337⟩⟩) 1 ⟨15336⟩ 288852

def event288885 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15337⟩⟩) (.sum [.predecessor 0 288883 .coefficient, .predecessor 1 288884 .coefficient])

def event288886 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15337⟩⟩, .operator (⟨288882, 1⟩, ⟨288852, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩)

def event288887 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15337⟩⟩) (.sum [.result 288882 .summary, .result 288852 .summary])

def exact288888RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact288888RawTermsValid :
    exact288888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288888 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15337⟩⟩) exact288888RawTerms .large 288885 (.finite 279174578176) (some (288887))

def event288889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17294⟩⟩) 0 ⟨15337⟩ 288888

def event288890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17294⟩⟩) 1 ⟨17293⟩ 288824

def event288891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17294⟩⟩) (.product (.predecessor 0 288889 .coefficient) (.predecessor 1 288890 .coefficient) (⟨false, false, none, none, none⟩))

def event288892 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17294⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨17293⟩⟩]⟩) [⟨.result 288824 .coefficient, false, none⟩])

def event288893 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17294⟩⟩) (.product (.result 288888 .summary) (.transfer 288892) (⟨false, false, none, none, none⟩))

def event288894 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17294⟩⟩, .operator (⟨288888, 1⟩, ⟨288824, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17293⟩⟩]⟩, (-1)⟩)

def event288895 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17294⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17293⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17293⟩⟩) ⟨16813⟩ 288821)

def event288896 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17294⟩⟩, .relation 288895 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨16813⟩⟩]⟩, (-1)⟩)

def event288897 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17294⟩⟩, .operator (⟨288888, 0⟩, ⟨288824, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17293⟩⟩]⟩, (1)⟩)

def exact288898RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17293⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨16813⟩⟩]⟩, (-1)⟩]

theorem exact288898RawTermsValid :
    exact288898RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288898 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17294⟩⟩) exact288898RawTerms .large 288891 (.finite 2997614207851288330240) (some (288893))

def event288899 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16229⟩⟩) 0 ⟨15332⟩ 13954

def event288900 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16229⟩⟩) (.authority (.relationPreimageSource ⟨36⟩))

def exact288901RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16229⟩⟩]⟩, (1)⟩]

theorem exact288901RawTermsValid :
    exact288901RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288901 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16229⟩⟩) exact288901RawTerms (.finite 5647228698) 288900 .exactZero (none)

def event288902 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16231⟩⟩) 0 ⟨16229⟩ 288901

def event288903 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16231⟩⟩) 1 ⟨2370⟩ 4

def event288904 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16231⟩⟩) (.scale (.predecessor 0 288902 .coefficient) (.value (.predecessor 1 288903 .coefficient)))

def exact288905RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16229⟩⟩]⟩, (1)⟩]

theorem exact288905RawTermsValid :
    exact288905RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288905 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16231⟩⟩) exact288905RawTerms (.finite 5647228698) 288904 .exactZero (none)

def event288906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16232⟩⟩) 0 ⟨5491⟩ 280745

def event288907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16232⟩⟩) 1 ⟨16231⟩ 288905

def event288908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16232⟩⟩) (.product (.predecessor 0 288906 .coefficient) (.predecessor 1 288907 .coefficient) (⟨false, false, none, none, none⟩))

def event288909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16232⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨16229⟩⟩]⟩) [⟨.result 288901 .coefficient, false, none⟩])

def event288910 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16232⟩⟩) (.product (.result 280745 .summary) (.transfer 288909) (⟨false, false, none, none, none⟩))

def event288911 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16232⟩⟩, .operator (⟨280745, 0⟩, ⟨288905, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16229⟩⟩]⟩, (1)⟩)

def event288912 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨16230⟩⟩)

def event288913 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event288914 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event288915 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event288916 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event288917 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event288918 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event288919 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event288920 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event288921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 288920

def event288922 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 288918

def event288923 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 288921 .coefficient) (.value (.predecessor 1 288922 .coefficient)))

def event288924 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event288925 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 288924

def event288926 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 288916

def event288927 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 288925 .coefficient, .predecessor 1 288926 .coefficient])

def event288928 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event288929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 288928

def event288930 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 288914

def event288931 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 288930 .coefficient))

def event288932 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event288933 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15330⟩⟩) 0 ⟨5487⟩ 288932

def event288934 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15330⟩⟩) (.authority (.programFamilyFact))

def exact288935RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15330⟩⟩], []⟩, (1)⟩]

theorem exact288935RawTermsValid :
    exact288935RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288935 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15330⟩⟩) exact288935RawTerms (.finite 2) 288934 .exactZero (none)

def event288936 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12291⟩⟩) 0 ⟨5487⟩ 288932

def event288937 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12291⟩⟩) (.authority (.programFamilyFact))

def exact288938RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12291⟩⟩], []⟩, (1)⟩]

theorem exact288938RawTermsValid :
    exact288938RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288938 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12291⟩⟩) exact288938RawTerms (.finite 2) 288937 .exactZero (none)

def event288939 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15331⟩⟩) 0 ⟨12291⟩ 288938

def event288940 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15331⟩⟩) 1 ⟨15330⟩ 288935

def event288941 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15331⟩⟩) (.product (.predecessor 0 288939 .coefficient) (.predecessor 1 288940 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event288942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15331⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], []⟩) [⟨.result 288938 .coefficient, true, some 1⟩, ⟨.result 288935 .coefficient, true, some 1⟩])

def event288943 : Event := .survivorFold (1) 288942

def exact288944RawTerms : List Term := []

theorem exact288944RawTermsValid :
    exact288944RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288944 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15331⟩⟩) exact288944RawTerms (.finite 4) 288941 (.finite 4) (some (288942))

def event288945 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15332⟩⟩) 0 ⟨15331⟩ 288944

def event288946 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15332⟩⟩) (.identity (.predecessor 0 288945 .coefficient))

def event288947 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15332⟩⟩) (.finite 4)

def event288948 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16229⟩⟩) 0 ⟨15332⟩ 288947

def event288949 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16229⟩⟩) (.authority (.relationPreimageSource ⟨36⟩))

def exact288950RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16229⟩⟩]⟩, (1)⟩]

theorem exact288950RawTermsValid :
    exact288950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288950 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16229⟩⟩) exact288950RawTerms (.finite 5647228698) 288949 .exactZero (none)

def event288951 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact288952RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact288952RawTermsValid :
    exact288952RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288952 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact288952RawTerms .large 288951 .exactZero (none)

def event288953 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16230⟩⟩) 0 ⟨35⟩ 288952

def event288954 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16230⟩⟩) 1 ⟨16229⟩ 288950

def event288955 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16230⟩⟩) (.product (.predecessor 0 288953 .coefficient) (.predecessor 1 288954 .coefficient) (⟨false, false, none, none, none⟩))

def event288956 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16230⟩⟩, .operator (⟨288952, 0⟩, ⟨288950, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16229⟩⟩]⟩, (1)⟩)

def exact288957RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16229⟩⟩]⟩, (1)⟩]

theorem exact288957RawTermsValid :
    exact288957RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288957 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16230⟩⟩) exact288957RawTerms .large 288955 .exactZero (none)

def event288958 : Event := .preFoldPolynomial 288957 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16229⟩⟩]⟩, (1)⟩] .exactZero none

def exact288959RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16229⟩⟩]⟩, (1)⟩]

def event288959 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨16230⟩⟩) 288958 exact288959RawTerms .large 288955 .exactZero (none)

def event288960 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨17297⟩⟩)

def event288961 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event288962 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event288963 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event288964 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event288965 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event288966 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event288967 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event288968 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event288969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 288968

def event288970 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 288966

def event288971 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 288969 .coefficient) (.value (.predecessor 1 288970 .coefficient)))

def event288972 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event288973 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 288972

def event288974 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 288964

def event288975 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 288973 .coefficient, .predecessor 1 288974 .coefficient])

def event288976 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event288977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 288976

def event288978 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 288962

def event288979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 288978 .coefficient))

def event288980 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event288981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15330⟩⟩) 0 ⟨5487⟩ 288980

def event288982 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15330⟩⟩) (.authority (.programFamilyFact))

def exact288983RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15330⟩⟩], []⟩, (1)⟩]

theorem exact288983RawTermsValid :
    exact288983RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288983 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15330⟩⟩) exact288983RawTerms (.finite 2) 288982 .exactZero (none)

def event288984 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12291⟩⟩) 0 ⟨5487⟩ 288980

def event288985 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12291⟩⟩) (.authority (.programFamilyFact))

def exact288986RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12291⟩⟩], []⟩, (1)⟩]

theorem exact288986RawTermsValid :
    exact288986RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288986 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12291⟩⟩) exact288986RawTerms (.finite 2) 288985 .exactZero (none)

def event288987 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15331⟩⟩) 0 ⟨12291⟩ 288986

def event288988 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15331⟩⟩) 1 ⟨15330⟩ 288983

def event288989 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15331⟩⟩) (.product (.predecessor 0 288987 .coefficient) (.predecessor 1 288988 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event288990 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15331⟩⟩, .operator (⟨288986, 0⟩, ⟨288983, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], []⟩, (1)⟩)

def exact288991RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], []⟩, (1)⟩]

theorem exact288991RawTermsValid :
    exact288991RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event288991 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15331⟩⟩) exact288991RawTerms (.finite 4) 288989 .exactZero (none)

def event288992 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15332⟩⟩) 0 ⟨15331⟩ 288991

def event288993 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15332⟩⟩) (.identity (.predecessor 0 288992 .coefficient))

def event288994 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15332⟩⟩) (.finite 4)

def event288995 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16812⟩⟩) 0 ⟨15332⟩ 288994

def event288996 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16812⟩⟩) (.authority (.programFamilyFact))

def event288997 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨16812⟩⟩) (.finite 3720)

def event288998 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event288999 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16813⟩⟩) 0 ⟨7177⟩ 288998

def event289000 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16813⟩⟩) 1 ⟨16812⟩ 288997

def event289001 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16813⟩⟩) (.authority (.operator))

def exact289002RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16813⟩⟩]⟩, (1)⟩]

theorem exact289002RawTermsValid :
    exact289002RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289002 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16813⟩⟩) exact289002RawTerms .large 289001 .exactZero (none)

def event289003 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17293⟩⟩) 0 ⟨16813⟩ 289002

def event289004 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17293⟩⟩) (.authority (.operator))

def exact289005RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17293⟩⟩]⟩, (1)⟩]

theorem exact289005RawTermsValid :
    exact289005RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289005 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17293⟩⟩) exact289005RawTerms (.finite 8192) 289004 .exactZero (none)

def event289006 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event289007 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event289008 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17102⟩⟩) 0 ⟨15332⟩ 288994

def event289009 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17102⟩⟩) 1 ⟨136⟩ 289007

def event289010 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17102⟩⟩) (.sum [.predecessor 0 289008 .coefficient, .predecessor 1 289009 .coefficient])

def event289011 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17102⟩⟩) (.finite 4)

def event289012 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17103⟩⟩) 0 ⟨17102⟩ 289011

def event289013 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17103⟩⟩) (.identity (.predecessor 0 289012 .coefficient))

def exact289014RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], []⟩, (1)⟩]

theorem exact289014RawTermsValid :
    exact289014RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289014 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17103⟩⟩) exact289014RawTerms (.finite 4) 289013 .exactZero (none)

def event289015 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact289016RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact289016RawTermsValid :
    exact289016RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289016 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact289016RawTerms .large 289015 .exactZero (none)

def event289017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17104⟩⟩) 0 ⟨6908⟩ 289016

def event289018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17104⟩⟩) 1 ⟨17103⟩ 289014

def event289019 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17104⟩⟩) (.product (.predecessor 0 289017 .coefficient) (.predecessor 1 289018 .coefficient) (⟨false, false, none, none, none⟩))

def event289020 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17104⟩⟩, .operator (⟨289016, 0⟩, ⟨289014, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact289021RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12291⟩⟩, ⟨.program ⟨257⟩, ⟨15330⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact289021RawTermsValid :
    exact289021RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event289021 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17104⟩⟩) exact289021RawTerms .large 289019 .exactZero (none)

def event289022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 288998

def event289023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def eventLeaf18048 : Array AnnotatedEvent := #[
  { event := event288768
    frameStart := 288687 },
  { event := event288769
    frameStart := 288687 },
  { event := event288770
    frameStart := 288687 },
  { event := event288771
    frameStart := 288687 },
  { event := event288772
    frameStart := 288687 },
  { event := event288773
    frameStart := 288687 },
  { event := event288774
    frameStart := 288687 },
  { event := event288775
    frameStart := 288687 },
  { event := event288776
    frameStart := 288687 },
  { event := event288777
    frameStart := 288687 },
  { event := event288778
    frameStart := 288687 },
  { event := event288779
    frameStart := 288687 },
  { event := event288780
    frameStart := 288687 },
  { event := event288781
    frameStart := 288687 },
  { event := event288782
    frameStart := 288687 },
  { event := event288783
    frameStart := 288687 }
]

def eventLeaf18049 : Array AnnotatedEvent := #[
  { event := event288784
    frameStart := 288687 },
  { event := event288785
    frameStart := 288687 },
  { event := event288786
    frameStart := 288687 },
  { event := event288787
    frameStart := 288687 },
  { event := event288788
    frameStart := 288687 },
  { event := event288789
    frameStart := 288687 },
  { event := event288790
    frameStart := 288687 },
  { event := event288791
    frameStart := 0 },
  { event := event288792
    frameStart := 0 },
  { event := event288793
    frameStart := 0 },
  { event := event288794
    frameStart := 0 },
  { event := event288795
    frameStart := 0 },
  { event := event288796
    frameStart := 0 },
  { event := event288797
    frameStart := 0 },
  { event := event288798
    frameStart := 0 },
  { event := event288799
    frameStart := 0 }
]

def eventLeaf18050 : Array AnnotatedEvent := #[
  { event := event288800
    frameStart := 0 },
  { event := event288801
    frameStart := 0 },
  { event := event288802
    frameStart := 0 },
  { event := event288803
    frameStart := 0 },
  { event := event288804
    frameStart := 0 },
  { event := event288805
    frameStart := 0 },
  { event := event288806
    frameStart := 0 },
  { event := event288807
    frameStart := 0 },
  { event := event288808
    frameStart := 0 },
  { event := event288809
    frameStart := 0 },
  { event := event288810
    frameStart := 0 },
  { event := event288811
    frameStart := 0 },
  { event := event288812
    frameStart := 0 },
  { event := event288813
    frameStart := 0 },
  { event := event288814
    frameStart := 0 },
  { event := event288815
    frameStart := 0 }
]

def eventLeaf18051 : Array AnnotatedEvent := #[
  { event := event288816
    frameStart := 0 },
  { event := event288817
    frameStart := 0 },
  { event := event288818
    frameStart := 0 },
  { event := event288819
    frameStart := 0 },
  { event := event288820
    frameStart := 0 },
  { event := event288821
    frameStart := 0 },
  { event := event288822
    frameStart := 0 },
  { event := event288823
    frameStart := 0 },
  { event := event288824
    frameStart := 0 },
  { event := event288825
    frameStart := 0 },
  { event := event288826
    frameStart := 0 },
  { event := event288827
    frameStart := 0 },
  { event := event288828
    frameStart := 0 },
  { event := event288829
    frameStart := 0 },
  { event := event288830
    frameStart := 0 },
  { event := event288831
    frameStart := 0 }
]

def eventLeaf18052 : Array AnnotatedEvent := #[
  { event := event288832
    frameStart := 0 },
  { event := event288833
    frameStart := 0 },
  { event := event288834
    frameStart := 0 },
  { event := event288835
    frameStart := 0 },
  { event := event288836
    frameStart := 0 },
  { event := event288837
    frameStart := 0 },
  { event := event288838
    frameStart := 0 },
  { event := event288839
    frameStart := 0 },
  { event := event288840
    frameStart := 0 },
  { event := event288841
    frameStart := 0 },
  { event := event288842
    frameStart := 0 },
  { event := event288843
    frameStart := 0 },
  { event := event288844
    frameStart := 0 },
  { event := event288845
    frameStart := 0 },
  { event := event288846
    frameStart := 0 },
  { event := event288847
    frameStart := 0 }
]

def eventLeaf18053 : Array AnnotatedEvent := #[
  { event := event288848
    frameStart := 0 },
  { event := event288849
    frameStart := 0 },
  { event := event288850
    frameStart := 0 },
  { event := event288851
    frameStart := 0 },
  { event := event288852
    frameStart := 0 },
  { event := event288853
    frameStart := 0 },
  { event := event288854
    frameStart := 0 },
  { event := event288855
    frameStart := 0 },
  { event := event288856
    frameStart := 0 },
  { event := event288857
    frameStart := 0 },
  { event := event288858
    frameStart := 0 },
  { event := event288859
    frameStart := 0 },
  { event := event288860
    frameStart := 0 },
  { event := event288861
    frameStart := 0 },
  { event := event288862
    frameStart := 0 },
  { event := event288863
    frameStart := 0 }
]

def eventLeaf18054 : Array AnnotatedEvent := #[
  { event := event288864
    frameStart := 0 },
  { event := event288865
    frameStart := 0 },
  { event := event288866
    frameStart := 0 },
  { event := event288867
    frameStart := 0 },
  { event := event288868
    frameStart := 0 },
  { event := event288869
    frameStart := 0 },
  { event := event288870
    frameStart := 0 },
  { event := event288871
    frameStart := 0 },
  { event := event288872
    frameStart := 0 },
  { event := event288873
    frameStart := 0 },
  { event := event288874
    frameStart := 0 },
  { event := event288875
    frameStart := 0 },
  { event := event288876
    frameStart := 0 },
  { event := event288877
    frameStart := 0 },
  { event := event288878
    frameStart := 0 },
  { event := event288879
    frameStart := 0 }
]

def eventLeaf18055 : Array AnnotatedEvent := #[
  { event := event288880
    frameStart := 0 },
  { event := event288881
    frameStart := 0 },
  { event := event288882
    frameStart := 0 },
  { event := event288883
    frameStart := 0 },
  { event := event288884
    frameStart := 0 },
  { event := event288885
    frameStart := 0 },
  { event := event288886
    frameStart := 0 },
  { event := event288887
    frameStart := 0 },
  { event := event288888
    frameStart := 0 },
  { event := event288889
    frameStart := 0 },
  { event := event288890
    frameStart := 0 },
  { event := event288891
    frameStart := 0 },
  { event := event288892
    frameStart := 0 },
  { event := event288893
    frameStart := 0 },
  { event := event288894
    frameStart := 0 },
  { event := event288895
    frameStart := 0 }
]

def eventLeaf18056 : Array AnnotatedEvent := #[
  { event := event288896
    frameStart := 0 },
  { event := event288897
    frameStart := 0 },
  { event := event288898
    frameStart := 0 },
  { event := event288899
    frameStart := 0 },
  { event := event288900
    frameStart := 0 },
  { event := event288901
    frameStart := 0 },
  { event := event288902
    frameStart := 0 },
  { event := event288903
    frameStart := 0 },
  { event := event288904
    frameStart := 0 },
  { event := event288905
    frameStart := 0 },
  { event := event288906
    frameStart := 0 },
  { event := event288907
    frameStart := 0 },
  { event := event288908
    frameStart := 0 },
  { event := event288909
    frameStart := 0 },
  { event := event288910
    frameStart := 0 },
  { event := event288911
    frameStart := 0 }
]

def eventLeaf18057 : Array AnnotatedEvent := #[
  { event := event288912
    frameStart := 288912 },
  { event := event288913
    frameStart := 288912 },
  { event := event288914
    frameStart := 288912 },
  { event := event288915
    frameStart := 288912 },
  { event := event288916
    frameStart := 288912 },
  { event := event288917
    frameStart := 288912 },
  { event := event288918
    frameStart := 288912 },
  { event := event288919
    frameStart := 288912 },
  { event := event288920
    frameStart := 288912 },
  { event := event288921
    frameStart := 288912 },
  { event := event288922
    frameStart := 288912 },
  { event := event288923
    frameStart := 288912 },
  { event := event288924
    frameStart := 288912 },
  { event := event288925
    frameStart := 288912 },
  { event := event288926
    frameStart := 288912 },
  { event := event288927
    frameStart := 288912 }
]

def eventLeaf18058 : Array AnnotatedEvent := #[
  { event := event288928
    frameStart := 288912 },
  { event := event288929
    frameStart := 288912 },
  { event := event288930
    frameStart := 288912 },
  { event := event288931
    frameStart := 288912 },
  { event := event288932
    frameStart := 288912 },
  { event := event288933
    frameStart := 288912 },
  { event := event288934
    frameStart := 288912 },
  { event := event288935
    frameStart := 288912 },
  { event := event288936
    frameStart := 288912 },
  { event := event288937
    frameStart := 288912 },
  { event := event288938
    frameStart := 288912 },
  { event := event288939
    frameStart := 288912 },
  { event := event288940
    frameStart := 288912 },
  { event := event288941
    frameStart := 288912 },
  { event := event288942
    frameStart := 288912 },
  { event := event288943
    frameStart := 288912 }
]

def eventLeaf18059 : Array AnnotatedEvent := #[
  { event := event288944
    frameStart := 288912 },
  { event := event288945
    frameStart := 288912 },
  { event := event288946
    frameStart := 288912 },
  { event := event288947
    frameStart := 288912 },
  { event := event288948
    frameStart := 288912 },
  { event := event288949
    frameStart := 288912 },
  { event := event288950
    frameStart := 288912 },
  { event := event288951
    frameStart := 288912 },
  { event := event288952
    frameStart := 288912 },
  { event := event288953
    frameStart := 288912 },
  { event := event288954
    frameStart := 288912 },
  { event := event288955
    frameStart := 288912 },
  { event := event288956
    frameStart := 288912 },
  { event := event288957
    frameStart := 288912 },
  { event := event288958
    frameStart := 288912 },
  { event := event288959
    frameStart := 288912 }
]

def eventLeaf18060 : Array AnnotatedEvent := #[
  { event := event288960
    frameStart := 288960 },
  { event := event288961
    frameStart := 288960 },
  { event := event288962
    frameStart := 288960 },
  { event := event288963
    frameStart := 288960 },
  { event := event288964
    frameStart := 288960 },
  { event := event288965
    frameStart := 288960 },
  { event := event288966
    frameStart := 288960 },
  { event := event288967
    frameStart := 288960 },
  { event := event288968
    frameStart := 288960 },
  { event := event288969
    frameStart := 288960 },
  { event := event288970
    frameStart := 288960 },
  { event := event288971
    frameStart := 288960 },
  { event := event288972
    frameStart := 288960 },
  { event := event288973
    frameStart := 288960 },
  { event := event288974
    frameStart := 288960 },
  { event := event288975
    frameStart := 288960 }
]

def eventLeaf18061 : Array AnnotatedEvent := #[
  { event := event288976
    frameStart := 288960 },
  { event := event288977
    frameStart := 288960 },
  { event := event288978
    frameStart := 288960 },
  { event := event288979
    frameStart := 288960 },
  { event := event288980
    frameStart := 288960 },
  { event := event288981
    frameStart := 288960 },
  { event := event288982
    frameStart := 288960 },
  { event := event288983
    frameStart := 288960 },
  { event := event288984
    frameStart := 288960 },
  { event := event288985
    frameStart := 288960 },
  { event := event288986
    frameStart := 288960 },
  { event := event288987
    frameStart := 288960 },
  { event := event288988
    frameStart := 288960 },
  { event := event288989
    frameStart := 288960 },
  { event := event288990
    frameStart := 288960 },
  { event := event288991
    frameStart := 288960 }
]

def eventLeaf18062 : Array AnnotatedEvent := #[
  { event := event288992
    frameStart := 288960 },
  { event := event288993
    frameStart := 288960 },
  { event := event288994
    frameStart := 288960 },
  { event := event288995
    frameStart := 288960 },
  { event := event288996
    frameStart := 288960 },
  { event := event288997
    frameStart := 288960 },
  { event := event288998
    frameStart := 288960 },
  { event := event288999
    frameStart := 288960 },
  { event := event289000
    frameStart := 288960 },
  { event := event289001
    frameStart := 288960 },
  { event := event289002
    frameStart := 288960 },
  { event := event289003
    frameStart := 288960 },
  { event := event289004
    frameStart := 288960 },
  { event := event289005
    frameStart := 288960 },
  { event := event289006
    frameStart := 288960 },
  { event := event289007
    frameStart := 288960 }
]

def eventLeaf18063 : Array AnnotatedEvent := #[
  { event := event289008
    frameStart := 288960 },
  { event := event289009
    frameStart := 288960 },
  { event := event289010
    frameStart := 288960 },
  { event := event289011
    frameStart := 288960 },
  { event := event289012
    frameStart := 288960 },
  { event := event289013
    frameStart := 288960 },
  { event := event289014
    frameStart := 288960 },
  { event := event289015
    frameStart := 288960 },
  { event := event289016
    frameStart := 288960 },
  { event := event289017
    frameStart := 288960 },
  { event := event289018
    frameStart := 288960 },
  { event := event289019
    frameStart := 288960 },
  { event := event289020
    frameStart := 288960 },
  { event := event289021
    frameStart := 288960 },
  { event := event289022
    frameStart := 288960 },
  { event := event289023
    frameStart := 288960 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1128
