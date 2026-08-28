import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1190

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact304640RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact304640RawTermsValid :
    exact304640RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304640 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49468⟩⟩) exact304640RawTerms .large 304638 .exactZero (none)

def event304641 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7196⟩⟩) 0 ⟨7177⟩ 304617

def event304642 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7196⟩⟩) (.authority (.operator))

def exact304643RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩]

theorem exact304643RawTermsValid :
    exact304643RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304643 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7196⟩⟩) exact304643RawTerms .large 304642 .exactZero (none)

def event304644 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49469⟩⟩) 0 ⟨7196⟩ 304643

def event304645 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49469⟩⟩) 1 ⟨49468⟩ 304640

def event304646 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49469⟩⟩) (.sum [.predecessor 0 304644 .coefficient, .predecessor 1 304645 .coefficient])

def exact304647RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact304647RawTermsValid :
    exact304647RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304647 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49469⟩⟩) exact304647RawTerms .large 304646 .exactZero (none)

def event304648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49774⟩⟩) 0 ⟨49469⟩ 304647

def event304649 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49774⟩⟩) 1 ⟨49773⟩ 304624

def event304650 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49774⟩⟩) (.product (.predecessor 0 304648 .coefficient) (.predecessor 1 304649 .coefficient) (⟨false, false, none, none, none⟩))

def event304651 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49774⟩⟩, .operator (⟨304647, 0⟩, ⟨304624, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49773⟩⟩]⟩, (1)⟩)

def event304652 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49774⟩⟩, .operator (⟨304647, 1⟩, ⟨304624, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49773⟩⟩]⟩, (-1)⟩)

def event304653 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49774⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49773⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49773⟩⟩) ⟨49210⟩ 304621)

def event304654 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49774⟩⟩, .relation 304653 0, ⟨[⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨49210⟩⟩]⟩, (-1)⟩)

def exact304655RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49773⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨49210⟩⟩]⟩, (-1)⟩]

theorem exact304655RawTermsValid :
    exact304655RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304655 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49774⟩⟩) exact304655RawTerms .large 304650 .exactZero (none)

def event304656 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48229⟩⟩) 0 ⟨48069⟩ 304613

def event304657 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48229⟩⟩) (.authority (.programFamilyFact))

def exact304658RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48229⟩⟩], []⟩, (1)⟩]

theorem exact304658RawTermsValid :
    exact304658RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304658 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48229⟩⟩) exact304658RawTerms (.finite 60) 304657 .exactZero (none)

def event304659 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48231⟩⟩) 0 ⟨6908⟩ 304635

def event304660 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48231⟩⟩) 1 ⟨48229⟩ 304658

def event304661 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48231⟩⟩) (.product (.predecessor 0 304659 .coefficient) (.predecessor 1 304660 .coefficient) (⟨false, true, none, none, some 1⟩))

def event304662 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48231⟩⟩, .operator (⟨304635, 0⟩, ⟨304658, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48229⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact304663RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48229⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact304663RawTermsValid :
    exact304663RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304663 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48231⟩⟩) exact304663RawTerms .large 304661 .exactZero (none)

def event304664 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7231⟩⟩) 0 ⟨7177⟩ 304617

def event304665 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7231⟩⟩) (.authority (.operator))

def exact304666RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩]

theorem exact304666RawTermsValid :
    exact304666RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304666 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7231⟩⟩) exact304666RawTerms .large 304665 .exactZero (none)

def event304667 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48232⟩⟩) 0 ⟨7231⟩ 304666

def event304668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48232⟩⟩) 1 ⟨48231⟩ 304663

def event304669 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48232⟩⟩) (.sum [.predecessor 0 304667 .coefficient, .predecessor 1 304668 .coefficient])

def exact304670RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48229⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact304670RawTermsValid :
    exact304670RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304670 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48232⟩⟩) exact304670RawTerms .large 304669 .exactZero (none)

def event304671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49778⟩⟩) 0 ⟨48232⟩ 304670

def event304672 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49778⟩⟩) 1 ⟨49774⟩ 304655

def event304673 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49778⟩⟩) (.sum [.predecessor 0 304671 .coefficient, .predecessor 1 304672 .coefficient])

def exact304674RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49773⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨49210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48229⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact304674RawTermsValid :
    exact304674RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304674 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49778⟩⟩) exact304674RawTerms .large 304673 .exactZero (none)

def event304675 : Event := .preFoldPolynomial 304674 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49773⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨49210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48229⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact304676RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49773⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨49210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48229⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event304676 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨49778⟩⟩) 304675 exact304676RawTerms .large 304673 .exactZero (none)

def event304677 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨48069⟩⟩) ⟨⟨110⟩, ⟨93⟩, ⟨135⟩⟩ ⟨304543, 304677⟩

def event304678 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨48695⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48692⟩⟩]⟩) (1) 0 2 (.universal 304677 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48692⟩⟩]⟩) (none) 304676)

def event304679 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48695⟩⟩, .relation 304678 1, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩)

def event304680 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48695⟩⟩, .relation 304678 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49773⟩⟩]⟩, (-1)⟩)

def event304681 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48695⟩⟩, .relation 304678 2, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨49210⟩⟩]⟩, (1)⟩)

def event304682 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48695⟩⟩, .relation 304678 3, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48229⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact304683RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49773⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨49210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48229⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact304683RawTermsValid :
    exact304683RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304683 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48695⟩⟩) exact304683RawTerms .large 304539 (.finite 202072841853861888) (some (304541))

def event304684 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49776⟩⟩) 0 ⟨48695⟩ 304683

def event304685 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49776⟩⟩) 1 ⟨49775⟩ 304529

def event304686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49776⟩⟩) (.sum [.predecessor 0 304684 .coefficient, .predecessor 1 304685 .coefficient])

def event304687 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49776⟩⟩, .operator (⟨304683, 0⟩, ⟨304529, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49773⟩⟩]⟩, (1)⟩)

def event304688 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49776⟩⟩, .operator (⟨304683, 2⟩, ⟨304529, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48068⟩⟩], [⟨.program ⟨257⟩, ⟨49210⟩⟩]⟩, (-1)⟩)

def event304689 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49776⟩⟩) (.sum [.result 304683 .summary, .result 304529 .summary])

def exact304690RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48229⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact304690RawTermsValid :
    exact304690RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304690 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49776⟩⟩) exact304690RawTerms .large 304686 (.finite 32194504275408640829496428331008) (some (304689))

def event304691 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49777⟩⟩) 0 ⟨49776⟩ 304690

def event304692 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49777⟩⟩) 1 ⟨7148⟩ 15542

def event304693 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49777⟩⟩) (.product (.predecessor 0 304691 .coefficient) (.predecessor 1 304692 .coefficient) (⟨false, false, none, none, none⟩))

def event304694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49777⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩) [⟨.result 15538 .coefficient, false, none⟩])

def event304695 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49777⟩⟩) (.product (.result 304690 .summary) (.transfer 304694) (⟨false, false, none, none, none⟩))

def event304696 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49777⟩⟩, .operator (⟨304690, 0⟩, ⟨15542, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩)

def event304697 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49777⟩⟩, .operator (⟨304690, 1⟩, ⟨15542, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48229⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (-1)⟩)

def event304698 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49777⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48229⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7147⟩⟩) ⟨7039⟩ 15535)

def event304699 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49777⟩⟩, .relation 304698 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48229⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact304700RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48229⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact304700RawTermsValid :
    exact304700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304700 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49777⟩⟩) exact304700RawTerms .large 304693 (.finite 345685857434530723496243679576218056785920) (some (304695))

def event304701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46530⟩⟩) 0 ⟨7177⟩ 15500

def event304702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46530⟩⟩) 1 ⟨46529⟩ 295531

def event304703 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46530⟩⟩) (.authority (.operator))

def exact304704RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46530⟩⟩]⟩, (1)⟩]

theorem exact304704RawTermsValid :
    exact304704RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304704 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46530⟩⟩) exact304704RawTerms .large 304703 .exactZero (none)

def event304705 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47093⟩⟩) 0 ⟨46530⟩ 304704

def event304706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47093⟩⟩) (.authority (.operator))

def exact304707RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩, (1)⟩]

theorem exact304707RawTermsValid :
    exact304707RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304707 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47093⟩⟩) exact304707RawTerms (.finite 8192) 304706 .exactZero (none)

def event304708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47095⟩⟩) 0 ⟨46871⟩ 295791

def event304709 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47095⟩⟩) 1 ⟨47093⟩ 304707

def event304710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47095⟩⟩) (.product (.predecessor 0 304708 .coefficient) (.predecessor 1 304709 .coefficient) (⟨false, false, none, none, none⟩))

def event304711 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47095⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩) [⟨.result 304707 .coefficient, false, none⟩])

def event304712 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47095⟩⟩) (.product (.result 295791 .summary) (.transfer 304711) (⟨false, false, none, none, none⟩))

def event304713 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47095⟩⟩, .operator (⟨295791, 0⟩, ⟨304707, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩, (1)⟩)

def event304714 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47095⟩⟩, .operator (⟨295791, 1⟩, ⟨304707, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩, (-1)⟩)

def event304715 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47095⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47093⟩⟩) ⟨46530⟩ 304704)

def event304716 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47095⟩⟩, .relation 304715 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨46530⟩⟩]⟩, (-1)⟩)

def exact304717RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨46530⟩⟩]⟩, (-1)⟩]

theorem exact304717RawTermsValid :
    exact304717RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304717 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47095⟩⟩) exact304717RawTerms .large 304710 (.finite 32194307824962751379413684715520) (some (304712))

def event304718 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46012⟩⟩) 0 ⟨45389⟩ 14330

def event304719 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46012⟩⟩) (.authority (.relationPreimageSource ⟨91⟩))

def exact304720RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩, (1)⟩]

theorem exact304720RawTermsValid :
    exact304720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304720 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46012⟩⟩) exact304720RawTerms (.finite 5647228698) 304719 .exactZero (none)

def event304721 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46014⟩⟩) 0 ⟨46012⟩ 304720

def event304722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46014⟩⟩) 1 ⟨2370⟩ 4

def event304723 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46014⟩⟩) (.scale (.predecessor 0 304721 .coefficient) (.value (.predecessor 1 304722 .coefficient)))

def exact304724RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩, (1)⟩]

theorem exact304724RawTermsValid :
    exact304724RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304724 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46014⟩⟩) exact304724RawTerms (.finite 5647228698) 304723 .exactZero (none)

def event304725 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46015⟩⟩) 0 ⟨2380⟩ 295195

def event304726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46015⟩⟩) 1 ⟨46014⟩ 304724

def event304727 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46015⟩⟩) (.product (.predecessor 0 304725 .coefficient) (.predecessor 1 304726 .coefficient) (⟨false, false, none, none, none⟩))

def event304728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46015⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩) [⟨.result 304720 .coefficient, false, none⟩])

def event304729 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46015⟩⟩) (.product (.result 295195 .summary) (.transfer 304728) (⟨false, false, none, none, none⟩))

def event304730 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46015⟩⟩, .operator (⟨295195, 0⟩, ⟨304724, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩, (1)⟩)

def event304731 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨46013⟩⟩)

def event304732 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event304733 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event304734 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event304735 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event304736 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 304735

def event304737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 304733

def event304738 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 304736 .coefficient) (.value (.predecessor 1 304737 .coefficient)))

def event304739 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event304740 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44914⟩⟩) 0 ⟨392⟩ 304739

def event304741 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44914⟩⟩) (.authority (.programFamilyFact))

def exact304742RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨44914⟩⟩], []⟩, (1)⟩]

theorem exact304742RawTermsValid :
    exact304742RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304742 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44914⟩⟩) exact304742RawTerms (.finite 58) 304741 .exactZero (none)

def event304743 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14631⟩⟩) 0 ⟨392⟩ 304739

def event304744 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14631⟩⟩) (.authority (.programFamilyFact))

def exact304745RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14631⟩⟩], []⟩, (1)⟩]

theorem exact304745RawTermsValid :
    exact304745RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304745 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14631⟩⟩) exact304745RawTerms (.finite 58) 304744 .exactZero (none)

def event304746 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44915⟩⟩) 0 ⟨14631⟩ 304745

def event304747 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44915⟩⟩) 1 ⟨44914⟩ 304742

def event304748 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44915⟩⟩) (.product (.predecessor 0 304746 .coefficient) (.predecessor 1 304747 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event304749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44915⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], []⟩) [⟨.result 304745 .coefficient, true, some 1⟩, ⟨.result 304742 .coefficient, true, some 1⟩])

def event304750 : Event := .survivorFold (1) 304749

def exact304751RawTerms : List Term := []

theorem exact304751RawTermsValid :
    exact304751RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304751 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44915⟩⟩) exact304751RawTerms (.finite 3364) 304748 (.finite 3364) (some (304749))

def event304752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44916⟩⟩) 0 ⟨44915⟩ 304751

def event304753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44916⟩⟩) (.identity (.predecessor 0 304752 .coefficient))

def event304754 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44916⟩⟩) (.finite 3364)

def event304755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45388⟩⟩) 0 ⟨44916⟩ 304754

def event304756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45388⟩⟩) (.authority (.programFamilyFact))

def exact304757RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], []⟩, (1)⟩]

theorem exact304757RawTermsValid :
    exact304757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304757 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45388⟩⟩) exact304757RawTerms (.finite 58) 304756 .exactZero (none)

def event304758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45389⟩⟩) 0 ⟨45388⟩ 304757

def event304759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45389⟩⟩) (.identity (.predecessor 0 304758 .coefficient))

def event304760 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45389⟩⟩) (.finite 58)

def event304761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46012⟩⟩) 0 ⟨45389⟩ 304760

def event304762 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46012⟩⟩) (.authority (.relationPreimageSource ⟨91⟩))

def exact304763RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩, (1)⟩]

theorem exact304763RawTermsValid :
    exact304763RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304763 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46012⟩⟩) exact304763RawTerms (.finite 5647228698) 304762 .exactZero (none)

def event304764 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact304765RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact304765RawTermsValid :
    exact304765RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304765 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact304765RawTerms .large 304764 .exactZero (none)

def event304766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46013⟩⟩) 0 ⟨35⟩ 304765

def event304767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46013⟩⟩) 1 ⟨46012⟩ 304763

def event304768 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46013⟩⟩) (.product (.predecessor 0 304766 .coefficient) (.predecessor 1 304767 .coefficient) (⟨false, false, none, none, none⟩))

def event304769 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46013⟩⟩, .operator (⟨304765, 0⟩, ⟨304763, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩, (1)⟩)

def exact304770RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩, (1)⟩]

theorem exact304770RawTermsValid :
    exact304770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304770 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46013⟩⟩) exact304770RawTerms .large 304768 .exactZero (none)

def event304771 : Event := .preFoldPolynomial 304770 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩, (1)⟩] .exactZero none

def exact304772RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩, (1)⟩]

def event304772 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨46013⟩⟩) 304771 exact304772RawTerms .large 304768 .exactZero (none)

def event304773 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨47098⟩⟩)

def event304774 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event304775 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event304776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event304777 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event304778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 304777

def event304779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 304775

def event304780 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 304778 .coefficient) (.value (.predecessor 1 304779 .coefficient)))

def event304781 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event304782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44914⟩⟩) 0 ⟨392⟩ 304781

def event304783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44914⟩⟩) (.authority (.programFamilyFact))

def exact304784RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨44914⟩⟩], []⟩, (1)⟩]

theorem exact304784RawTermsValid :
    exact304784RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304784 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44914⟩⟩) exact304784RawTerms (.finite 58) 304783 .exactZero (none)

def event304785 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14631⟩⟩) 0 ⟨392⟩ 304781

def event304786 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14631⟩⟩) (.authority (.programFamilyFact))

def exact304787RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14631⟩⟩], []⟩, (1)⟩]

theorem exact304787RawTermsValid :
    exact304787RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304787 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14631⟩⟩) exact304787RawTerms (.finite 58) 304786 .exactZero (none)

def event304788 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44915⟩⟩) 0 ⟨14631⟩ 304787

def event304789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44915⟩⟩) 1 ⟨44914⟩ 304784

def event304790 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44915⟩⟩) (.product (.predecessor 0 304788 .coefficient) (.predecessor 1 304789 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event304791 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44915⟩⟩, .operator (⟨304787, 0⟩, ⟨304784, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], []⟩, (1)⟩)

def exact304792RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14631⟩⟩, ⟨.program ⟨257⟩, ⟨44914⟩⟩], []⟩, (1)⟩]

theorem exact304792RawTermsValid :
    exact304792RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304792 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44915⟩⟩) exact304792RawTerms (.finite 3364) 304790 .exactZero (none)

def event304793 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44916⟩⟩) 0 ⟨44915⟩ 304792

def event304794 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44916⟩⟩) (.identity (.predecessor 0 304793 .coefficient))

def event304795 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44916⟩⟩) (.finite 3364)

def event304796 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45388⟩⟩) 0 ⟨44916⟩ 304795

def event304797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45388⟩⟩) (.authority (.programFamilyFact))

def exact304798RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], []⟩, (1)⟩]

theorem exact304798RawTermsValid :
    exact304798RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304798 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45388⟩⟩) exact304798RawTerms (.finite 58) 304797 .exactZero (none)

def event304799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45389⟩⟩) 0 ⟨45388⟩ 304798

def event304800 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45389⟩⟩) (.identity (.predecessor 0 304799 .coefficient))

def event304801 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45389⟩⟩) (.finite 58)

def event304802 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46529⟩⟩) 0 ⟨45389⟩ 304801

def event304803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46529⟩⟩) (.authority (.programFamilyFact))

def event304804 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46529⟩⟩) (.finite 3720)

def event304805 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event304806 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46530⟩⟩) 0 ⟨7177⟩ 304805

def event304807 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46530⟩⟩) 1 ⟨46529⟩ 304804

def event304808 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46530⟩⟩) (.authority (.operator))

def exact304809RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46530⟩⟩]⟩, (1)⟩]

theorem exact304809RawTermsValid :
    exact304809RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304809 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46530⟩⟩) exact304809RawTerms .large 304808 .exactZero (none)

def event304810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47093⟩⟩) 0 ⟨46530⟩ 304809

def event304811 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47093⟩⟩) (.authority (.operator))

def exact304812RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩, (1)⟩]

theorem exact304812RawTermsValid :
    exact304812RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304812 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47093⟩⟩) exact304812RawTerms (.finite 8192) 304811 .exactZero (none)

def event304813 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event304814 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event304815 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46786⟩⟩) 0 ⟨45389⟩ 304801

def event304816 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46786⟩⟩) 1 ⟨136⟩ 304814

def event304817 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46786⟩⟩) (.sum [.predecessor 0 304815 .coefficient, .predecessor 1 304816 .coefficient])

def event304818 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46786⟩⟩) (.finite 58)

def event304819 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46787⟩⟩) 0 ⟨46786⟩ 304818

def event304820 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46787⟩⟩) (.identity (.predecessor 0 304819 .coefficient))

def exact304821RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], []⟩, (1)⟩]

theorem exact304821RawTermsValid :
    exact304821RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304821 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46787⟩⟩) exact304821RawTerms (.finite 58) 304820 .exactZero (none)

def event304822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact304823RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact304823RawTermsValid :
    exact304823RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304823 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact304823RawTerms .large 304822 .exactZero (none)

def event304824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46788⟩⟩) 0 ⟨6908⟩ 304823

def event304825 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46788⟩⟩) 1 ⟨46787⟩ 304821

def event304826 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46788⟩⟩) (.product (.predecessor 0 304824 .coefficient) (.predecessor 1 304825 .coefficient) (⟨false, false, none, none, none⟩))

def event304827 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46788⟩⟩, .operator (⟨304823, 0⟩, ⟨304821, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact304828RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact304828RawTermsValid :
    exact304828RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304828 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46788⟩⟩) exact304828RawTerms .large 304826 .exactZero (none)

def event304829 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7195⟩⟩) 0 ⟨7177⟩ 304805

def event304830 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7195⟩⟩) (.authority (.operator))

def exact304831RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩]

theorem exact304831RawTermsValid :
    exact304831RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304831 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7195⟩⟩) exact304831RawTerms .large 304830 .exactZero (none)

def event304832 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46789⟩⟩) 0 ⟨7195⟩ 304831

def event304833 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46789⟩⟩) 1 ⟨46788⟩ 304828

def event304834 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46789⟩⟩) (.sum [.predecessor 0 304832 .coefficient, .predecessor 1 304833 .coefficient])

def exact304835RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact304835RawTermsValid :
    exact304835RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304835 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46789⟩⟩) exact304835RawTerms .large 304834 .exactZero (none)

def event304836 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47094⟩⟩) 0 ⟨46789⟩ 304835

def event304837 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47094⟩⟩) 1 ⟨47093⟩ 304812

def event304838 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47094⟩⟩) (.product (.predecessor 0 304836 .coefficient) (.predecessor 1 304837 .coefficient) (⟨false, false, none, none, none⟩))

def event304839 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47094⟩⟩, .operator (⟨304835, 0⟩, ⟨304812, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩, (1)⟩)

def event304840 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47094⟩⟩, .operator (⟨304835, 1⟩, ⟨304812, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩, (-1)⟩)

def event304841 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47094⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47093⟩⟩) ⟨46530⟩ 304809)

def event304842 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47094⟩⟩, .relation 304841 0, ⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨46530⟩⟩]⟩, (-1)⟩)

def exact304843RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨46530⟩⟩]⟩, (-1)⟩]

theorem exact304843RawTermsValid :
    exact304843RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304843 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47094⟩⟩) exact304843RawTerms .large 304838 .exactZero (none)

def event304844 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45549⟩⟩) 0 ⟨45389⟩ 304801

def event304845 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45549⟩⟩) (.authority (.programFamilyFact))

def exact304846RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45549⟩⟩], []⟩, (1)⟩]

theorem exact304846RawTermsValid :
    exact304846RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304846 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45549⟩⟩) exact304846RawTerms (.finite 58) 304845 .exactZero (none)

def event304847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45551⟩⟩) 0 ⟨6908⟩ 304823

def event304848 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45551⟩⟩) 1 ⟨45549⟩ 304846

def event304849 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45551⟩⟩) (.product (.predecessor 0 304847 .coefficient) (.predecessor 1 304848 .coefficient) (⟨false, true, none, none, some 1⟩))

def event304850 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45551⟩⟩, .operator (⟨304823, 0⟩, ⟨304846, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45549⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact304851RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45549⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact304851RawTermsValid :
    exact304851RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304851 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45551⟩⟩) exact304851RawTerms .large 304849 .exactZero (none)

def event304852 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7229⟩⟩) 0 ⟨7177⟩ 304805

def event304853 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7229⟩⟩) (.authority (.operator))

def exact304854RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩]

theorem exact304854RawTermsValid :
    exact304854RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304854 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7229⟩⟩) exact304854RawTerms .large 304853 .exactZero (none)

def event304855 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45552⟩⟩) 0 ⟨7229⟩ 304854

def event304856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45552⟩⟩) 1 ⟨45551⟩ 304851

def event304857 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45552⟩⟩) (.sum [.predecessor 0 304855 .coefficient, .predecessor 1 304856 .coefficient])

def exact304858RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45549⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact304858RawTermsValid :
    exact304858RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304858 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45552⟩⟩) exact304858RawTerms .large 304857 .exactZero (none)

def event304859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47098⟩⟩) 0 ⟨45552⟩ 304858

def event304860 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47098⟩⟩) 1 ⟨47094⟩ 304843

def event304861 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47098⟩⟩) (.sum [.predecessor 0 304859 .coefficient, .predecessor 1 304860 .coefficient])

def exact304862RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨46530⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45549⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact304862RawTermsValid :
    exact304862RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304862 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47098⟩⟩) exact304862RawTerms .large 304861 .exactZero (none)

def event304863 : Event := .preFoldPolynomial 304862 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨46530⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45549⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact304864RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨46530⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45549⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event304864 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨47098⟩⟩) 304863 exact304864RawTerms .large 304861 .exactZero (none)

def event304865 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨45389⟩⟩) ⟨⟨108⟩, ⟨91⟩, ⟨135⟩⟩ ⟨304731, 304865⟩

def event304866 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨46015⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩) (1) 0 2 (.universal 304865 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46012⟩⟩]⟩) (none) 304864)

def event304867 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46015⟩⟩, .relation 304866 1, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩)

def event304868 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46015⟩⟩, .relation 304866 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩, (-1)⟩)

def event304869 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46015⟩⟩, .relation 304866 2, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨46530⟩⟩]⟩, (1)⟩)

def event304870 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46015⟩⟩, .relation 304866 3, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact304871RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨46530⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact304871RawTermsValid :
    exact304871RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304871 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46015⟩⟩) exact304871RawTerms .large 304727 (.finite 202072841853861888) (some (304729))

def event304872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47096⟩⟩) 0 ⟨46015⟩ 304871

def event304873 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47096⟩⟩) 1 ⟨47095⟩ 304717

def event304874 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47096⟩⟩) (.sum [.predecessor 0 304872 .coefficient, .predecessor 1 304873 .coefficient])

def event304875 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47096⟩⟩, .operator (⟨304871, 0⟩, ⟨304717, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47093⟩⟩]⟩, (1)⟩)

def event304876 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47096⟩⟩, .operator (⟨304871, 2⟩, ⟨304717, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45388⟩⟩], [⟨.program ⟨257⟩, ⟨46530⟩⟩]⟩, (-1)⟩)

def event304877 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47096⟩⟩) (.sum [.result 304871 .summary, .result 304717 .summary])

def exact304878RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact304878RawTermsValid :
    exact304878RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304878 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47096⟩⟩) exact304878RawTerms .large 304874 (.finite 32194307824962953452255538577408) (some (304877))

def event304879 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47097⟩⟩) 0 ⟨47096⟩ 304878

def event304880 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47097⟩⟩) 1 ⟨7152⟩ 15562

def event304881 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47097⟩⟩) (.product (.predecessor 0 304879 .coefficient) (.predecessor 1 304880 .coefficient) (⟨false, false, none, none, none⟩))

def event304882 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47097⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩) [⟨.result 15558 .coefficient, false, none⟩])

def event304883 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47097⟩⟩) (.product (.result 304878 .summary) (.transfer 304882) (⟨false, false, none, none, none⟩))

def event304884 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47097⟩⟩, .operator (⟨304878, 0⟩, ⟨15562, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩)

def event304885 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47097⟩⟩, .operator (⟨304878, 1⟩, ⟨15562, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (-1)⟩)

def event304886 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47097⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7151⟩⟩) ⟨7041⟩ 15555)

def event304887 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47097⟩⟩, .relation 304886 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact304888RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45549⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact304888RawTermsValid :
    exact304888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304888 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47097⟩⟩) exact304888RawTerms .large 304881 (.finite 345683748063931943722519589062084311121920) (some (304883))

def event304889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43850⟩⟩) 0 ⟨7177⟩ 15500

def event304890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43850⟩⟩) 1 ⟨43849⟩ 295965

def event304891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43850⟩⟩) (.authority (.operator))

def exact304892RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43850⟩⟩]⟩, (1)⟩]

theorem exact304892RawTermsValid :
    exact304892RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304892 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43850⟩⟩) exact304892RawTerms .large 304891 .exactZero (none)

def event304893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44413⟩⟩) 0 ⟨43850⟩ 304892

def event304894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44413⟩⟩) (.authority (.operator))

def exact304895RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44413⟩⟩]⟩, (1)⟩]

theorem exact304895RawTermsValid :
    exact304895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event304895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44413⟩⟩) exact304895RawTerms (.finite 8192) 304894 .exactZero (none)

def eventLeaf19040 : Array AnnotatedEvent := #[
  { event := event304640
    frameStart := 304585 },
  { event := event304641
    frameStart := 304585 },
  { event := event304642
    frameStart := 304585 },
  { event := event304643
    frameStart := 304585 },
  { event := event304644
    frameStart := 304585 },
  { event := event304645
    frameStart := 304585 },
  { event := event304646
    frameStart := 304585 },
  { event := event304647
    frameStart := 304585 },
  { event := event304648
    frameStart := 304585 },
  { event := event304649
    frameStart := 304585 },
  { event := event304650
    frameStart := 304585 },
  { event := event304651
    frameStart := 304585 },
  { event := event304652
    frameStart := 304585 },
  { event := event304653
    frameStart := 304585 },
  { event := event304654
    frameStart := 304585 },
  { event := event304655
    frameStart := 304585 }
]

def eventLeaf19041 : Array AnnotatedEvent := #[
  { event := event304656
    frameStart := 304585 },
  { event := event304657
    frameStart := 304585 },
  { event := event304658
    frameStart := 304585 },
  { event := event304659
    frameStart := 304585 },
  { event := event304660
    frameStart := 304585 },
  { event := event304661
    frameStart := 304585 },
  { event := event304662
    frameStart := 304585 },
  { event := event304663
    frameStart := 304585 },
  { event := event304664
    frameStart := 304585 },
  { event := event304665
    frameStart := 304585 },
  { event := event304666
    frameStart := 304585 },
  { event := event304667
    frameStart := 304585 },
  { event := event304668
    frameStart := 304585 },
  { event := event304669
    frameStart := 304585 },
  { event := event304670
    frameStart := 304585 },
  { event := event304671
    frameStart := 304585 }
]

def eventLeaf19042 : Array AnnotatedEvent := #[
  { event := event304672
    frameStart := 304585 },
  { event := event304673
    frameStart := 304585 },
  { event := event304674
    frameStart := 304585 },
  { event := event304675
    frameStart := 304585 },
  { event := event304676
    frameStart := 304585 },
  { event := event304677
    frameStart := 0 },
  { event := event304678
    frameStart := 0 },
  { event := event304679
    frameStart := 0 },
  { event := event304680
    frameStart := 0 },
  { event := event304681
    frameStart := 0 },
  { event := event304682
    frameStart := 0 },
  { event := event304683
    frameStart := 0 },
  { event := event304684
    frameStart := 0 },
  { event := event304685
    frameStart := 0 },
  { event := event304686
    frameStart := 0 },
  { event := event304687
    frameStart := 0 }
]

def eventLeaf19043 : Array AnnotatedEvent := #[
  { event := event304688
    frameStart := 0 },
  { event := event304689
    frameStart := 0 },
  { event := event304690
    frameStart := 0 },
  { event := event304691
    frameStart := 0 },
  { event := event304692
    frameStart := 0 },
  { event := event304693
    frameStart := 0 },
  { event := event304694
    frameStart := 0 },
  { event := event304695
    frameStart := 0 },
  { event := event304696
    frameStart := 0 },
  { event := event304697
    frameStart := 0 },
  { event := event304698
    frameStart := 0 },
  { event := event304699
    frameStart := 0 },
  { event := event304700
    frameStart := 0 },
  { event := event304701
    frameStart := 0 },
  { event := event304702
    frameStart := 0 },
  { event := event304703
    frameStart := 0 }
]

def eventLeaf19044 : Array AnnotatedEvent := #[
  { event := event304704
    frameStart := 0 },
  { event := event304705
    frameStart := 0 },
  { event := event304706
    frameStart := 0 },
  { event := event304707
    frameStart := 0 },
  { event := event304708
    frameStart := 0 },
  { event := event304709
    frameStart := 0 },
  { event := event304710
    frameStart := 0 },
  { event := event304711
    frameStart := 0 },
  { event := event304712
    frameStart := 0 },
  { event := event304713
    frameStart := 0 },
  { event := event304714
    frameStart := 0 },
  { event := event304715
    frameStart := 0 },
  { event := event304716
    frameStart := 0 },
  { event := event304717
    frameStart := 0 },
  { event := event304718
    frameStart := 0 },
  { event := event304719
    frameStart := 0 }
]

def eventLeaf19045 : Array AnnotatedEvent := #[
  { event := event304720
    frameStart := 0 },
  { event := event304721
    frameStart := 0 },
  { event := event304722
    frameStart := 0 },
  { event := event304723
    frameStart := 0 },
  { event := event304724
    frameStart := 0 },
  { event := event304725
    frameStart := 0 },
  { event := event304726
    frameStart := 0 },
  { event := event304727
    frameStart := 0 },
  { event := event304728
    frameStart := 0 },
  { event := event304729
    frameStart := 0 },
  { event := event304730
    frameStart := 0 },
  { event := event304731
    frameStart := 304731 },
  { event := event304732
    frameStart := 304731 },
  { event := event304733
    frameStart := 304731 },
  { event := event304734
    frameStart := 304731 },
  { event := event304735
    frameStart := 304731 }
]

def eventLeaf19046 : Array AnnotatedEvent := #[
  { event := event304736
    frameStart := 304731 },
  { event := event304737
    frameStart := 304731 },
  { event := event304738
    frameStart := 304731 },
  { event := event304739
    frameStart := 304731 },
  { event := event304740
    frameStart := 304731 },
  { event := event304741
    frameStart := 304731 },
  { event := event304742
    frameStart := 304731 },
  { event := event304743
    frameStart := 304731 },
  { event := event304744
    frameStart := 304731 },
  { event := event304745
    frameStart := 304731 },
  { event := event304746
    frameStart := 304731 },
  { event := event304747
    frameStart := 304731 },
  { event := event304748
    frameStart := 304731 },
  { event := event304749
    frameStart := 304731 },
  { event := event304750
    frameStart := 304731 },
  { event := event304751
    frameStart := 304731 }
]

def eventLeaf19047 : Array AnnotatedEvent := #[
  { event := event304752
    frameStart := 304731 },
  { event := event304753
    frameStart := 304731 },
  { event := event304754
    frameStart := 304731 },
  { event := event304755
    frameStart := 304731 },
  { event := event304756
    frameStart := 304731 },
  { event := event304757
    frameStart := 304731 },
  { event := event304758
    frameStart := 304731 },
  { event := event304759
    frameStart := 304731 },
  { event := event304760
    frameStart := 304731 },
  { event := event304761
    frameStart := 304731 },
  { event := event304762
    frameStart := 304731 },
  { event := event304763
    frameStart := 304731 },
  { event := event304764
    frameStart := 304731 },
  { event := event304765
    frameStart := 304731 },
  { event := event304766
    frameStart := 304731 },
  { event := event304767
    frameStart := 304731 }
]

def eventLeaf19048 : Array AnnotatedEvent := #[
  { event := event304768
    frameStart := 304731 },
  { event := event304769
    frameStart := 304731 },
  { event := event304770
    frameStart := 304731 },
  { event := event304771
    frameStart := 304731 },
  { event := event304772
    frameStart := 304731 },
  { event := event304773
    frameStart := 304773 },
  { event := event304774
    frameStart := 304773 },
  { event := event304775
    frameStart := 304773 },
  { event := event304776
    frameStart := 304773 },
  { event := event304777
    frameStart := 304773 },
  { event := event304778
    frameStart := 304773 },
  { event := event304779
    frameStart := 304773 },
  { event := event304780
    frameStart := 304773 },
  { event := event304781
    frameStart := 304773 },
  { event := event304782
    frameStart := 304773 },
  { event := event304783
    frameStart := 304773 }
]

def eventLeaf19049 : Array AnnotatedEvent := #[
  { event := event304784
    frameStart := 304773 },
  { event := event304785
    frameStart := 304773 },
  { event := event304786
    frameStart := 304773 },
  { event := event304787
    frameStart := 304773 },
  { event := event304788
    frameStart := 304773 },
  { event := event304789
    frameStart := 304773 },
  { event := event304790
    frameStart := 304773 },
  { event := event304791
    frameStart := 304773 },
  { event := event304792
    frameStart := 304773 },
  { event := event304793
    frameStart := 304773 },
  { event := event304794
    frameStart := 304773 },
  { event := event304795
    frameStart := 304773 },
  { event := event304796
    frameStart := 304773 },
  { event := event304797
    frameStart := 304773 },
  { event := event304798
    frameStart := 304773 },
  { event := event304799
    frameStart := 304773 }
]

def eventLeaf19050 : Array AnnotatedEvent := #[
  { event := event304800
    frameStart := 304773 },
  { event := event304801
    frameStart := 304773 },
  { event := event304802
    frameStart := 304773 },
  { event := event304803
    frameStart := 304773 },
  { event := event304804
    frameStart := 304773 },
  { event := event304805
    frameStart := 304773 },
  { event := event304806
    frameStart := 304773 },
  { event := event304807
    frameStart := 304773 },
  { event := event304808
    frameStart := 304773 },
  { event := event304809
    frameStart := 304773 },
  { event := event304810
    frameStart := 304773 },
  { event := event304811
    frameStart := 304773 },
  { event := event304812
    frameStart := 304773 },
  { event := event304813
    frameStart := 304773 },
  { event := event304814
    frameStart := 304773 },
  { event := event304815
    frameStart := 304773 }
]

def eventLeaf19051 : Array AnnotatedEvent := #[
  { event := event304816
    frameStart := 304773 },
  { event := event304817
    frameStart := 304773 },
  { event := event304818
    frameStart := 304773 },
  { event := event304819
    frameStart := 304773 },
  { event := event304820
    frameStart := 304773 },
  { event := event304821
    frameStart := 304773 },
  { event := event304822
    frameStart := 304773 },
  { event := event304823
    frameStart := 304773 },
  { event := event304824
    frameStart := 304773 },
  { event := event304825
    frameStart := 304773 },
  { event := event304826
    frameStart := 304773 },
  { event := event304827
    frameStart := 304773 },
  { event := event304828
    frameStart := 304773 },
  { event := event304829
    frameStart := 304773 },
  { event := event304830
    frameStart := 304773 },
  { event := event304831
    frameStart := 304773 }
]

def eventLeaf19052 : Array AnnotatedEvent := #[
  { event := event304832
    frameStart := 304773 },
  { event := event304833
    frameStart := 304773 },
  { event := event304834
    frameStart := 304773 },
  { event := event304835
    frameStart := 304773 },
  { event := event304836
    frameStart := 304773 },
  { event := event304837
    frameStart := 304773 },
  { event := event304838
    frameStart := 304773 },
  { event := event304839
    frameStart := 304773 },
  { event := event304840
    frameStart := 304773 },
  { event := event304841
    frameStart := 304773 },
  { event := event304842
    frameStart := 304773 },
  { event := event304843
    frameStart := 304773 },
  { event := event304844
    frameStart := 304773 },
  { event := event304845
    frameStart := 304773 },
  { event := event304846
    frameStart := 304773 },
  { event := event304847
    frameStart := 304773 }
]

def eventLeaf19053 : Array AnnotatedEvent := #[
  { event := event304848
    frameStart := 304773 },
  { event := event304849
    frameStart := 304773 },
  { event := event304850
    frameStart := 304773 },
  { event := event304851
    frameStart := 304773 },
  { event := event304852
    frameStart := 304773 },
  { event := event304853
    frameStart := 304773 },
  { event := event304854
    frameStart := 304773 },
  { event := event304855
    frameStart := 304773 },
  { event := event304856
    frameStart := 304773 },
  { event := event304857
    frameStart := 304773 },
  { event := event304858
    frameStart := 304773 },
  { event := event304859
    frameStart := 304773 },
  { event := event304860
    frameStart := 304773 },
  { event := event304861
    frameStart := 304773 },
  { event := event304862
    frameStart := 304773 },
  { event := event304863
    frameStart := 304773 }
]

def eventLeaf19054 : Array AnnotatedEvent := #[
  { event := event304864
    frameStart := 304773 },
  { event := event304865
    frameStart := 0 },
  { event := event304866
    frameStart := 0 },
  { event := event304867
    frameStart := 0 },
  { event := event304868
    frameStart := 0 },
  { event := event304869
    frameStart := 0 },
  { event := event304870
    frameStart := 0 },
  { event := event304871
    frameStart := 0 },
  { event := event304872
    frameStart := 0 },
  { event := event304873
    frameStart := 0 },
  { event := event304874
    frameStart := 0 },
  { event := event304875
    frameStart := 0 },
  { event := event304876
    frameStart := 0 },
  { event := event304877
    frameStart := 0 },
  { event := event304878
    frameStart := 0 },
  { event := event304879
    frameStart := 0 }
]

def eventLeaf19055 : Array AnnotatedEvent := #[
  { event := event304880
    frameStart := 0 },
  { event := event304881
    frameStart := 0 },
  { event := event304882
    frameStart := 0 },
  { event := event304883
    frameStart := 0 },
  { event := event304884
    frameStart := 0 },
  { event := event304885
    frameStart := 0 },
  { event := event304886
    frameStart := 0 },
  { event := event304887
    frameStart := 0 },
  { event := event304888
    frameStart := 0 },
  { event := event304889
    frameStart := 0 },
  { event := event304890
    frameStart := 0 },
  { event := event304891
    frameStart := 0 },
  { event := event304892
    frameStart := 0 },
  { event := event304893
    frameStart := 0 },
  { event := event304894
    frameStart := 0 },
  { event := event304895
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1190
