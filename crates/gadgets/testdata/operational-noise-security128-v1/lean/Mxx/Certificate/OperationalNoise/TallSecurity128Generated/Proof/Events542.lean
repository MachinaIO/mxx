import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events542

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact138752RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩, (1)⟩]

theorem exact138752RawTermsValid :
    exact138752RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138752 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64362⟩⟩) exact138752RawTerms (.finite 8192) 138751 .exactZero (none)

def event138753 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25407⟩⟩) 0 ⟨25406⟩ 6285

def event138754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25407⟩⟩) 1 ⟨6919⟩ 134403

def event138755 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25407⟩⟩) (.tensor (.predecessor 0 138753 .coefficient) (.predecessor 1 138754 .coefficient) true false)

def event138756 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨25407⟩⟩, .operator (⟨6285, 0⟩, ⟨134403, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact138757RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact138757RawTermsValid :
    exact138757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138757 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25407⟩⟩) exact138757RawTerms .large 138755 .exactZero (none)

def event138758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7783⟩⟩) 0 ⟨5471⟩ 134273

def event138759 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7783⟩⟩) 1 ⟨7275⟩ 21589

def event138760 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7783⟩⟩) (.product (.predecessor 0 138758 .coefficient) (.predecessor 1 138759 .coefficient) (⟨false, false, none, none, none⟩))

def event138761 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7783⟩⟩, .operator (⟨134273, 0⟩, ⟨21589, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩)

def exact138762RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩]

theorem exact138762RawTermsValid :
    exact138762RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138762 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7783⟩⟩) exact138762RawTerms .large 138760 .exactZero (none)

def event138763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25408⟩⟩) 0 ⟨7783⟩ 138762

def event138764 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25408⟩⟩) 1 ⟨25407⟩ 138757

def event138765 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25408⟩⟩) (.sum [.predecessor 0 138763 .coefficient, .predecessor 1 138764 .coefficient])

def exact138766RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact138766RawTermsValid :
    exact138766RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138766 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25408⟩⟩) exact138766RawTerms .large 138765 .exactZero (none)

def event138767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25409⟩⟩) 0 ⟨25408⟩ 138766

def event138768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25409⟩⟩) 1 ⟨101⟩ 21581

def event138769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25409⟩⟩) (.sum [.predecessor 0 138767 .coefficient, .predecessor 1 138768 .coefficient])

def event138770 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25409⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨101⟩⟩]⟩) [⟨.result 21581 .coefficient, false, none⟩])

def event138771 : Event := .survivorFold (1) 138770

def exact138772RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact138772RawTermsValid :
    exact138772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138772 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25409⟩⟩) exact138772RawTerms .large 138769 (.finite 26) (some (138770))

def event138773 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62279⟩⟩) 0 ⟨25409⟩ 138772

def event138774 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62279⟩⟩) 1 ⟨62276⟩ 6288

def event138775 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62279⟩⟩) (.product (.predecessor 0 138773 .coefficient) (.predecessor 1 138774 .coefficient) (⟨false, true, none, none, some 1⟩))

def event138776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62279⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨62276⟩⟩], []⟩) [⟨.result 6288 .coefficient, true, some 1⟩])

def event138777 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62279⟩⟩) (.product (.result 138772 .summary) (.transfer 138776) (⟨false, false, none, none, none⟩))

def event138778 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62279⟩⟩, .operator (⟨138772, 1⟩, ⟨6288, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event138779 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62279⟩⟩, .operator (⟨138772, 0⟩, ⟨6288, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩)

def exact138780RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩]

theorem exact138780RawTermsValid :
    exact138780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138780 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62279⟩⟩) exact138780RawTerms .large 138775 (.finite 18743296) (some (138777))

def event138781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62280⟩⟩) 0 ⟨62276⟩ 6288

def event138782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62280⟩⟩) 1 ⟨6919⟩ 134403

def event138783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62280⟩⟩) (.tensor (.predecessor 0 138781 .coefficient) (.predecessor 1 138782 .coefficient) true false)

def event138784 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62280⟩⟩, .operator (⟨6288, 0⟩, ⟨134403, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact138785RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact138785RawTermsValid :
    exact138785RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138785 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62280⟩⟩) exact138785RawTerms .large 138783 .exactZero (none)

def event138786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7801⟩⟩) 0 ⟨5471⟩ 134273

def event138787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7801⟩⟩) 1 ⟨7293⟩ 21630

def event138788 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7801⟩⟩) (.product (.predecessor 0 138786 .coefficient) (.predecessor 1 138787 .coefficient) (⟨false, false, none, none, none⟩))

def event138789 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7801⟩⟩, .operator (⟨134273, 0⟩, ⟨21630, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩]⟩, (1)⟩)

def exact138790RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩]⟩, (1)⟩]

theorem exact138790RawTermsValid :
    exact138790RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138790 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7801⟩⟩) exact138790RawTerms .large 138788 .exactZero (none)

def event138791 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62281⟩⟩) 0 ⟨7801⟩ 138790

def event138792 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62281⟩⟩) 1 ⟨62280⟩ 138785

def event138793 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62281⟩⟩) (.sum [.predecessor 0 138791 .coefficient, .predecessor 1 138792 .coefficient])

def exact138794RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact138794RawTermsValid :
    exact138794RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138794 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62281⟩⟩) exact138794RawTerms .large 138793 .exactZero (none)

def event138795 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62282⟩⟩) 0 ⟨62281⟩ 138794

def event138796 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62282⟩⟩) 1 ⟨119⟩ 21622

def event138797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62282⟩⟩) (.sum [.predecessor 0 138795 .coefficient, .predecessor 1 138796 .coefficient])

def event138798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62282⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨119⟩⟩]⟩) [⟨.result 21622 .coefficient, false, none⟩])

def event138799 : Event := .survivorFold (1) 138798

def exact138800RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact138800RawTermsValid :
    exact138800RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138800 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62282⟩⟩) exact138800RawTerms .large 138797 (.finite 26) (some (138798))

def event138801 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62283⟩⟩) 0 ⟨62282⟩ 138800

def event138802 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62283⟩⟩) 1 ⟨9539⟩ 21619

def event138803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62283⟩⟩) (.product (.predecessor 0 138801 .coefficient) (.predecessor 1 138802 .coefficient) (⟨false, false, none, none, none⟩))

def event138804 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62283⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) [⟨.result 21615 .coefficient, false, none⟩])

def event138805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62283⟩⟩) (.product (.result 138800 .summary) (.transfer 138804) (⟨false, false, none, none, none⟩))

def event138806 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62283⟩⟩, .operator (⟨138800, 1⟩, ⟨21619, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (-1)⟩)

def event138807 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨62283⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9538⟩⟩) ⟨7275⟩ 21589)

def event138808 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62283⟩⟩, .relation 138807 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (-1)⟩)

def event138809 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62283⟩⟩, .operator (⟨138800, 0⟩, ⟨21619, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (1)⟩)

def exact138810RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (-1)⟩]

theorem exact138810RawTermsValid :
    exact138810RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138810 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62283⟩⟩) exact138810RawTerms .large 138803 (.finite 279172874240) (some (138805))

def event138811 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62284⟩⟩) 0 ⟨62283⟩ 138810

def event138812 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62284⟩⟩) 1 ⟨62279⟩ 138780

def event138813 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62284⟩⟩) (.sum [.predecessor 0 138811 .coefficient, .predecessor 1 138812 .coefficient])

def event138814 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62284⟩⟩, .operator (⟨138810, 1⟩, ⟨138780, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩)

def event138815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62284⟩⟩) (.sum [.result 138810 .summary, .result 138780 .summary])

def exact138816RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact138816RawTermsValid :
    exact138816RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138816 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62284⟩⟩) exact138816RawTerms .large 138813 (.finite 279191617536) (some (138815))

def event138817 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64363⟩⟩) 0 ⟨62284⟩ 138816

def event138818 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64363⟩⟩) 1 ⟨64362⟩ 138752

def event138819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64363⟩⟩) (.product (.predecessor 0 138817 .coefficient) (.predecessor 1 138818 .coefficient) (⟨false, false, none, none, none⟩))

def event138820 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64363⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩) [⟨.result 138752 .coefficient, false, none⟩])

def event138821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64363⟩⟩) (.product (.result 138816 .summary) (.transfer 138820) (⟨false, false, none, none, none⟩))

def event138822 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64363⟩⟩, .operator (⟨138816, 1⟩, ⟨138752, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩, (-1)⟩)

def event138823 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨64363⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64362⟩⟩) ⟨63887⟩ 138749)

def event138824 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64363⟩⟩, .relation 138823 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨63887⟩⟩]⟩, (-1)⟩)

def event138825 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64363⟩⟩, .operator (⟨138816, 0⟩, ⟨138752, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩, (1)⟩)

def exact138826RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨63887⟩⟩]⟩, (-1)⟩]

theorem exact138826RawTermsValid :
    exact138826RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138826 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64363⟩⟩) exact138826RawTerms .large 138819 (.finite 2997797166586150256640) (some (138821))

def event138827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63299⟩⟩) 0 ⟨62278⟩ 6296

def event138828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63299⟩⟩) (.authority (.relationPreimageSource ⟨45⟩))

def exact138829RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩, (1)⟩]

theorem exact138829RawTermsValid :
    exact138829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138829 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63299⟩⟩) exact138829RawTerms (.finite 5647228698) 138828 .exactZero (none)

def event138830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63301⟩⟩) 0 ⟨63299⟩ 138829

def event138831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63301⟩⟩) 1 ⟨2370⟩ 4

def event138832 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63301⟩⟩) (.scale (.predecessor 0 138830 .coefficient) (.value (.predecessor 1 138831 .coefficient)))

def exact138833RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩, (1)⟩]

theorem exact138833RawTermsValid :
    exact138833RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138833 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63301⟩⟩) exact138833RawTerms (.finite 5647228698) 138832 .exactZero (none)

def event138834 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63302⟩⟩) 0 ⟨5473⟩ 134495

def event138835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63302⟩⟩) 1 ⟨63301⟩ 138833

def event138836 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63302⟩⟩) (.product (.predecessor 0 138834 .coefficient) (.predecessor 1 138835 .coefficient) (⟨false, false, none, none, none⟩))

def event138837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63302⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩) [⟨.result 138829 .coefficient, false, none⟩])

def event138838 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63302⟩⟩) (.product (.result 134495 .summary) (.transfer 138837) (⟨false, false, none, none, none⟩))

def event138839 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63302⟩⟩, .operator (⟨134495, 0⟩, ⟨138833, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩, (1)⟩)

def event138840 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨63300⟩⟩)

def event138841 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event138842 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event138843 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event138844 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event138845 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event138846 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event138847 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event138848 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event138849 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 138848

def event138850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 138846

def event138851 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 138849 .coefficient) (.value (.predecessor 1 138850 .coefficient)))

def event138852 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event138853 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 138852

def event138854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 138844

def event138855 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 138853 .coefficient, .predecessor 1 138854 .coefficient])

def event138856 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event138857 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 138856

def event138858 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 138842

def event138859 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 138858 .coefficient))

def event138860 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event138861 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25406⟩⟩) 0 ⟨5469⟩ 138860

def event138862 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25406⟩⟩) (.authority (.programFamilyFact))

def exact138863RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩], []⟩, (1)⟩]

theorem exact138863RawTermsValid :
    exact138863RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138863 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25406⟩⟩) exact138863RawTerms (.finite 22) 138862 .exactZero (none)

def event138864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62276⟩⟩) 0 ⟨5469⟩ 138860

def event138865 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62276⟩⟩) (.authority (.programFamilyFact))

def exact138866RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62276⟩⟩], []⟩, (1)⟩]

theorem exact138866RawTermsValid :
    exact138866RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138866 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62276⟩⟩) exact138866RawTerms (.finite 22) 138865 .exactZero (none)

def event138867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62277⟩⟩) 0 ⟨62276⟩ 138866

def event138868 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62277⟩⟩) 1 ⟨25406⟩ 138863

def event138869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62277⟩⟩) (.product (.predecessor 0 138867 .coefficient) (.predecessor 1 138868 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event138870 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62277⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], []⟩) [⟨.result 138866 .coefficient, true, some 1⟩, ⟨.result 138863 .coefficient, true, some 1⟩])

def event138871 : Event := .survivorFold (1) 138870

def exact138872RawTerms : List Term := []

theorem exact138872RawTermsValid :
    exact138872RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138872 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62277⟩⟩) exact138872RawTerms (.finite 484) 138869 (.finite 484) (some (138870))

def event138873 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62278⟩⟩) 0 ⟨62277⟩ 138872

def event138874 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62278⟩⟩) (.identity (.predecessor 0 138873 .coefficient))

def event138875 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨62278⟩⟩) (.finite 484)

def event138876 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63299⟩⟩) 0 ⟨62278⟩ 138875

def event138877 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63299⟩⟩) (.authority (.relationPreimageSource ⟨45⟩))

def exact138878RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩, (1)⟩]

theorem exact138878RawTermsValid :
    exact138878RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138878 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63299⟩⟩) exact138878RawTerms (.finite 5647228698) 138877 .exactZero (none)

def event138879 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact138880RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact138880RawTermsValid :
    exact138880RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138880 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact138880RawTerms .large 138879 .exactZero (none)

def event138881 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63300⟩⟩) 0 ⟨35⟩ 138880

def event138882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63300⟩⟩) 1 ⟨63299⟩ 138878

def event138883 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63300⟩⟩) (.product (.predecessor 0 138881 .coefficient) (.predecessor 1 138882 .coefficient) (⟨false, false, none, none, none⟩))

def event138884 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63300⟩⟩, .operator (⟨138880, 0⟩, ⟨138878, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩, (1)⟩)

def exact138885RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩, (1)⟩]

theorem exact138885RawTermsValid :
    exact138885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138885 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63300⟩⟩) exact138885RawTerms .large 138883 .exactZero (none)

def event138886 : Event := .preFoldPolynomial 138885 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩, (1)⟩] .exactZero none

def exact138887RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩, (1)⟩]

def event138887 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨63300⟩⟩) 138886 exact138887RawTerms .large 138883 .exactZero (none)

def event138888 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨64366⟩⟩)

def event138889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event138890 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event138891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event138892 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event138893 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event138894 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event138895 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event138896 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event138897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 138896

def event138898 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 138894

def event138899 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 138897 .coefficient) (.value (.predecessor 1 138898 .coefficient)))

def event138900 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event138901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 138900

def event138902 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 138892

def event138903 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 138901 .coefficient, .predecessor 1 138902 .coefficient])

def event138904 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event138905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 138904

def event138906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 138890

def event138907 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 138906 .coefficient))

def event138908 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event138909 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25406⟩⟩) 0 ⟨5469⟩ 138908

def event138910 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25406⟩⟩) (.authority (.programFamilyFact))

def exact138911RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩], []⟩, (1)⟩]

theorem exact138911RawTermsValid :
    exact138911RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138911 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25406⟩⟩) exact138911RawTerms (.finite 22) 138910 .exactZero (none)

def event138912 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62276⟩⟩) 0 ⟨5469⟩ 138908

def event138913 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62276⟩⟩) (.authority (.programFamilyFact))

def exact138914RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62276⟩⟩], []⟩, (1)⟩]

theorem exact138914RawTermsValid :
    exact138914RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138914 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62276⟩⟩) exact138914RawTerms (.finite 22) 138913 .exactZero (none)

def event138915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62277⟩⟩) 0 ⟨62276⟩ 138914

def event138916 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62277⟩⟩) 1 ⟨25406⟩ 138911

def event138917 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62277⟩⟩) (.product (.predecessor 0 138915 .coefficient) (.predecessor 1 138916 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event138918 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62277⟩⟩, .operator (⟨138914, 0⟩, ⟨138911, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], []⟩, (1)⟩)

def exact138919RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], []⟩, (1)⟩]

theorem exact138919RawTermsValid :
    exact138919RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138919 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62277⟩⟩) exact138919RawTerms (.finite 484) 138917 .exactZero (none)

def event138920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62278⟩⟩) 0 ⟨62277⟩ 138919

def event138921 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62278⟩⟩) (.identity (.predecessor 0 138920 .coefficient))

def event138922 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨62278⟩⟩) (.finite 484)

def event138923 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63886⟩⟩) 0 ⟨62278⟩ 138922

def event138924 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63886⟩⟩) (.authority (.programFamilyFact))

def event138925 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨63886⟩⟩) (.finite 3720)

def event138926 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event138927 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63887⟩⟩) 0 ⟨7177⟩ 138926

def event138928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63887⟩⟩) 1 ⟨63886⟩ 138925

def event138929 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63887⟩⟩) (.authority (.operator))

def exact138930RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨63887⟩⟩]⟩, (1)⟩]

theorem exact138930RawTermsValid :
    exact138930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138930 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63887⟩⟩) exact138930RawTerms .large 138929 .exactZero (none)

def event138931 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64362⟩⟩) 0 ⟨63887⟩ 138930

def event138932 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64362⟩⟩) (.authority (.operator))

def exact138933RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩, (1)⟩]

theorem exact138933RawTermsValid :
    exact138933RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138933 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64362⟩⟩) exact138933RawTerms (.finite 8192) 138932 .exactZero (none)

def event138934 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event138935 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event138936 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64178⟩⟩) 0 ⟨62278⟩ 138922

def event138937 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64178⟩⟩) 1 ⟨136⟩ 138935

def event138938 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64178⟩⟩) (.sum [.predecessor 0 138936 .coefficient, .predecessor 1 138937 .coefficient])

def event138939 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨64178⟩⟩) (.finite 484)

def event138940 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64179⟩⟩) 0 ⟨64178⟩ 138939

def event138941 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64179⟩⟩) (.identity (.predecessor 0 138940 .coefficient))

def exact138942RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], []⟩, (1)⟩]

theorem exact138942RawTermsValid :
    exact138942RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138942 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64179⟩⟩) exact138942RawTerms (.finite 484) 138941 .exactZero (none)

def event138943 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact138944RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact138944RawTermsValid :
    exact138944RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138944 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact138944RawTerms .large 138943 .exactZero (none)

def event138945 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64180⟩⟩) 0 ⟨6908⟩ 138944

def event138946 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64180⟩⟩) 1 ⟨64179⟩ 138942

def event138947 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64180⟩⟩) (.product (.predecessor 0 138945 .coefficient) (.predecessor 1 138946 .coefficient) (⟨false, false, none, none, none⟩))

def event138948 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64180⟩⟩, .operator (⟨138944, 0⟩, ⟨138942, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact138949RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact138949RawTermsValid :
    exact138949RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138949 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64180⟩⟩) exact138949RawTerms .large 138947 .exactZero (none)

def event138950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event138951 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event138952 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 138926

def event138953 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact138954RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact138954RawTermsValid :
    exact138954RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138954 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact138954RawTerms .large 138953 .exactZero (none)

def event138955 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7275⟩⟩) 0 ⟨7178⟩ 138954

def event138956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7275⟩⟩) (.identity (.predecessor 0 138955 .coefficient))

def exact138957RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩, (1)⟩]

theorem exact138957RawTermsValid :
    exact138957RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138957 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7275⟩⟩) exact138957RawTerms .large 138956 .exactZero (none)

def event138958 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9538⟩⟩) 0 ⟨7275⟩ 138957

def event138959 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9538⟩⟩) (.authority (.operator))

def exact138960RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (1)⟩]

theorem exact138960RawTermsValid :
    exact138960RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138960 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9538⟩⟩) exact138960RawTerms (.finite 8192) 138959 .exactZero (none)

def event138961 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9539⟩⟩) 0 ⟨9538⟩ 138960

def event138962 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9539⟩⟩) 1 ⟨2370⟩ 138951

def event138963 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9539⟩⟩) (.scale (.predecessor 0 138961 .coefficient) (.value (.predecessor 1 138962 .coefficient)))

def exact138964RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (1)⟩]

theorem exact138964RawTermsValid :
    exact138964RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138964 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9539⟩⟩) exact138964RawTerms (.finite 8192) 138963 .exactZero (none)

def event138965 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7293⟩⟩) 0 ⟨7178⟩ 138954

def event138966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7293⟩⟩) (.identity (.predecessor 0 138965 .coefficient))

def exact138967RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩]⟩, (1)⟩]

theorem exact138967RawTermsValid :
    exact138967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138967 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7293⟩⟩) exact138967RawTerms .large 138966 .exactZero (none)

def event138968 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9540⟩⟩) 0 ⟨7293⟩ 138967

def event138969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9540⟩⟩) 1 ⟨9539⟩ 138964

def event138970 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9540⟩⟩) (.product (.predecessor 0 138968 .coefficient) (.predecessor 1 138969 .coefficient) (⟨false, false, none, none, none⟩))

def event138971 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9540⟩⟩, .operator (⟨138967, 0⟩, ⟨138964, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (1)⟩)

def exact138972RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (1)⟩]

theorem exact138972RawTermsValid :
    exact138972RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138972 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9540⟩⟩) exact138972RawTerms .large 138970 .exactZero (none)

def event138973 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64181⟩⟩) 0 ⟨9540⟩ 138972

def event138974 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64181⟩⟩) 1 ⟨64180⟩ 138949

def event138975 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64181⟩⟩) (.sum [.predecessor 0 138973 .coefficient, .predecessor 1 138974 .coefficient])

def exact138976RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact138976RawTermsValid :
    exact138976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138976 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64181⟩⟩) exact138976RawTerms .large 138975 .exactZero (none)

def event138977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64365⟩⟩) 0 ⟨64181⟩ 138976

def event138978 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64365⟩⟩) 1 ⟨64362⟩ 138933

def event138979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64365⟩⟩) (.product (.predecessor 0 138977 .coefficient) (.predecessor 1 138978 .coefficient) (⟨false, false, none, none, none⟩))

def event138980 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64365⟩⟩, .operator (⟨138976, 0⟩, ⟨138933, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩, (1)⟩)

def event138981 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64365⟩⟩, .operator (⟨138976, 1⟩, ⟨138933, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩, (-1)⟩)

def event138982 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨64365⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64362⟩⟩) ⟨63887⟩ 138930)

def event138983 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨64365⟩⟩, .relation 138982 0, ⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨63887⟩⟩]⟩, (-1)⟩)

def exact138984RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨63887⟩⟩]⟩, (-1)⟩]

theorem exact138984RawTermsValid :
    exact138984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138984 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64365⟩⟩) exact138984RawTerms .large 138979 .exactZero (none)

def event138985 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62752⟩⟩) 0 ⟨62278⟩ 138922

def event138986 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62752⟩⟩) (.authority (.programFamilyFact))

def exact138987RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62752⟩⟩], []⟩, (1)⟩]

theorem exact138987RawTermsValid :
    exact138987RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138987 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62752⟩⟩) exact138987RawTerms (.finite 22) 138986 .exactZero (none)

def event138988 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62754⟩⟩) 0 ⟨6908⟩ 138944

def event138989 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62754⟩⟩) 1 ⟨62752⟩ 138987

def event138990 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62754⟩⟩) (.product (.predecessor 0 138988 .coefficient) (.predecessor 1 138989 .coefficient) (⟨false, true, none, none, some 1⟩))

def event138991 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62754⟩⟩, .operator (⟨138944, 0⟩, ⟨138987, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨62752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact138992RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨62752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact138992RawTermsValid :
    exact138992RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138992 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62754⟩⟩) exact138992RawTerms .large 138990 .exactZero (none)

def event138993 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7187⟩⟩) 0 ⟨7177⟩ 138926

def event138994 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7187⟩⟩) (.authority (.operator))

def exact138995RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩, (1)⟩]

theorem exact138995RawTermsValid :
    exact138995RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138995 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7187⟩⟩) exact138995RawTerms .large 138994 .exactZero (none)

def event138996 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62755⟩⟩) 0 ⟨7187⟩ 138995

def event138997 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62755⟩⟩) 1 ⟨62754⟩ 138992

def event138998 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62755⟩⟩) (.sum [.predecessor 0 138996 .coefficient, .predecessor 1 138997 .coefficient])

def exact138999RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact138999RawTermsValid :
    exact138999RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event138999 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62755⟩⟩) exact138999RawTerms .large 138998 .exactZero (none)

def event139000 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64366⟩⟩) 0 ⟨62755⟩ 138999

def event139001 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64366⟩⟩) 1 ⟨64365⟩ 138984

def event139002 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64366⟩⟩) (.sum [.predecessor 0 139000 .coefficient, .predecessor 1 139001 .coefficient])

def exact139003RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨63887⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact139003RawTermsValid :
    exact139003RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event139003 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64366⟩⟩) exact139003RawTerms .large 139002 .exactZero (none)

def event139004 : Event := .preFoldPolynomial 139003 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨63887⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact139005RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64362⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25406⟩⟩, ⟨.program ⟨257⟩, ⟨62276⟩⟩], [⟨.program ⟨257⟩, ⟨63887⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨62752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event139005 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨64366⟩⟩) 139004 exact139005RawTerms .large 139002 .exactZero (none)

def event139006 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨62278⟩⟩) ⟨⟨66⟩, ⟨45⟩, ⟨135⟩⟩ ⟨138840, 139006⟩

def event139007 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨63302⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩) (1) 0 2 (.universal 139006 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63299⟩⟩]⟩) (none) 139005)

def eventLeaf8672 : Array AnnotatedEvent := #[
  { event := event138752
    frameStart := 0 },
  { event := event138753
    frameStart := 0 },
  { event := event138754
    frameStart := 0 },
  { event := event138755
    frameStart := 0 },
  { event := event138756
    frameStart := 0 },
  { event := event138757
    frameStart := 0 },
  { event := event138758
    frameStart := 0 },
  { event := event138759
    frameStart := 0 },
  { event := event138760
    frameStart := 0 },
  { event := event138761
    frameStart := 0 },
  { event := event138762
    frameStart := 0 },
  { event := event138763
    frameStart := 0 },
  { event := event138764
    frameStart := 0 },
  { event := event138765
    frameStart := 0 },
  { event := event138766
    frameStart := 0 },
  { event := event138767
    frameStart := 0 }
]

def eventLeaf8673 : Array AnnotatedEvent := #[
  { event := event138768
    frameStart := 0 },
  { event := event138769
    frameStart := 0 },
  { event := event138770
    frameStart := 0 },
  { event := event138771
    frameStart := 0 },
  { event := event138772
    frameStart := 0 },
  { event := event138773
    frameStart := 0 },
  { event := event138774
    frameStart := 0 },
  { event := event138775
    frameStart := 0 },
  { event := event138776
    frameStart := 0 },
  { event := event138777
    frameStart := 0 },
  { event := event138778
    frameStart := 0 },
  { event := event138779
    frameStart := 0 },
  { event := event138780
    frameStart := 0 },
  { event := event138781
    frameStart := 0 },
  { event := event138782
    frameStart := 0 },
  { event := event138783
    frameStart := 0 }
]

def eventLeaf8674 : Array AnnotatedEvent := #[
  { event := event138784
    frameStart := 0 },
  { event := event138785
    frameStart := 0 },
  { event := event138786
    frameStart := 0 },
  { event := event138787
    frameStart := 0 },
  { event := event138788
    frameStart := 0 },
  { event := event138789
    frameStart := 0 },
  { event := event138790
    frameStart := 0 },
  { event := event138791
    frameStart := 0 },
  { event := event138792
    frameStart := 0 },
  { event := event138793
    frameStart := 0 },
  { event := event138794
    frameStart := 0 },
  { event := event138795
    frameStart := 0 },
  { event := event138796
    frameStart := 0 },
  { event := event138797
    frameStart := 0 },
  { event := event138798
    frameStart := 0 },
  { event := event138799
    frameStart := 0 }
]

def eventLeaf8675 : Array AnnotatedEvent := #[
  { event := event138800
    frameStart := 0 },
  { event := event138801
    frameStart := 0 },
  { event := event138802
    frameStart := 0 },
  { event := event138803
    frameStart := 0 },
  { event := event138804
    frameStart := 0 },
  { event := event138805
    frameStart := 0 },
  { event := event138806
    frameStart := 0 },
  { event := event138807
    frameStart := 0 },
  { event := event138808
    frameStart := 0 },
  { event := event138809
    frameStart := 0 },
  { event := event138810
    frameStart := 0 },
  { event := event138811
    frameStart := 0 },
  { event := event138812
    frameStart := 0 },
  { event := event138813
    frameStart := 0 },
  { event := event138814
    frameStart := 0 },
  { event := event138815
    frameStart := 0 }
]

def eventLeaf8676 : Array AnnotatedEvent := #[
  { event := event138816
    frameStart := 0 },
  { event := event138817
    frameStart := 0 },
  { event := event138818
    frameStart := 0 },
  { event := event138819
    frameStart := 0 },
  { event := event138820
    frameStart := 0 },
  { event := event138821
    frameStart := 0 },
  { event := event138822
    frameStart := 0 },
  { event := event138823
    frameStart := 0 },
  { event := event138824
    frameStart := 0 },
  { event := event138825
    frameStart := 0 },
  { event := event138826
    frameStart := 0 },
  { event := event138827
    frameStart := 0 },
  { event := event138828
    frameStart := 0 },
  { event := event138829
    frameStart := 0 },
  { event := event138830
    frameStart := 0 },
  { event := event138831
    frameStart := 0 }
]

def eventLeaf8677 : Array AnnotatedEvent := #[
  { event := event138832
    frameStart := 0 },
  { event := event138833
    frameStart := 0 },
  { event := event138834
    frameStart := 0 },
  { event := event138835
    frameStart := 0 },
  { event := event138836
    frameStart := 0 },
  { event := event138837
    frameStart := 0 },
  { event := event138838
    frameStart := 0 },
  { event := event138839
    frameStart := 0 },
  { event := event138840
    frameStart := 138840 },
  { event := event138841
    frameStart := 138840 },
  { event := event138842
    frameStart := 138840 },
  { event := event138843
    frameStart := 138840 },
  { event := event138844
    frameStart := 138840 },
  { event := event138845
    frameStart := 138840 },
  { event := event138846
    frameStart := 138840 },
  { event := event138847
    frameStart := 138840 }
]

def eventLeaf8678 : Array AnnotatedEvent := #[
  { event := event138848
    frameStart := 138840 },
  { event := event138849
    frameStart := 138840 },
  { event := event138850
    frameStart := 138840 },
  { event := event138851
    frameStart := 138840 },
  { event := event138852
    frameStart := 138840 },
  { event := event138853
    frameStart := 138840 },
  { event := event138854
    frameStart := 138840 },
  { event := event138855
    frameStart := 138840 },
  { event := event138856
    frameStart := 138840 },
  { event := event138857
    frameStart := 138840 },
  { event := event138858
    frameStart := 138840 },
  { event := event138859
    frameStart := 138840 },
  { event := event138860
    frameStart := 138840 },
  { event := event138861
    frameStart := 138840 },
  { event := event138862
    frameStart := 138840 },
  { event := event138863
    frameStart := 138840 }
]

def eventLeaf8679 : Array AnnotatedEvent := #[
  { event := event138864
    frameStart := 138840 },
  { event := event138865
    frameStart := 138840 },
  { event := event138866
    frameStart := 138840 },
  { event := event138867
    frameStart := 138840 },
  { event := event138868
    frameStart := 138840 },
  { event := event138869
    frameStart := 138840 },
  { event := event138870
    frameStart := 138840 },
  { event := event138871
    frameStart := 138840 },
  { event := event138872
    frameStart := 138840 },
  { event := event138873
    frameStart := 138840 },
  { event := event138874
    frameStart := 138840 },
  { event := event138875
    frameStart := 138840 },
  { event := event138876
    frameStart := 138840 },
  { event := event138877
    frameStart := 138840 },
  { event := event138878
    frameStart := 138840 },
  { event := event138879
    frameStart := 138840 }
]

def eventLeaf8680 : Array AnnotatedEvent := #[
  { event := event138880
    frameStart := 138840 },
  { event := event138881
    frameStart := 138840 },
  { event := event138882
    frameStart := 138840 },
  { event := event138883
    frameStart := 138840 },
  { event := event138884
    frameStart := 138840 },
  { event := event138885
    frameStart := 138840 },
  { event := event138886
    frameStart := 138840 },
  { event := event138887
    frameStart := 138840 },
  { event := event138888
    frameStart := 138888 },
  { event := event138889
    frameStart := 138888 },
  { event := event138890
    frameStart := 138888 },
  { event := event138891
    frameStart := 138888 },
  { event := event138892
    frameStart := 138888 },
  { event := event138893
    frameStart := 138888 },
  { event := event138894
    frameStart := 138888 },
  { event := event138895
    frameStart := 138888 }
]

def eventLeaf8681 : Array AnnotatedEvent := #[
  { event := event138896
    frameStart := 138888 },
  { event := event138897
    frameStart := 138888 },
  { event := event138898
    frameStart := 138888 },
  { event := event138899
    frameStart := 138888 },
  { event := event138900
    frameStart := 138888 },
  { event := event138901
    frameStart := 138888 },
  { event := event138902
    frameStart := 138888 },
  { event := event138903
    frameStart := 138888 },
  { event := event138904
    frameStart := 138888 },
  { event := event138905
    frameStart := 138888 },
  { event := event138906
    frameStart := 138888 },
  { event := event138907
    frameStart := 138888 },
  { event := event138908
    frameStart := 138888 },
  { event := event138909
    frameStart := 138888 },
  { event := event138910
    frameStart := 138888 },
  { event := event138911
    frameStart := 138888 }
]

def eventLeaf8682 : Array AnnotatedEvent := #[
  { event := event138912
    frameStart := 138888 },
  { event := event138913
    frameStart := 138888 },
  { event := event138914
    frameStart := 138888 },
  { event := event138915
    frameStart := 138888 },
  { event := event138916
    frameStart := 138888 },
  { event := event138917
    frameStart := 138888 },
  { event := event138918
    frameStart := 138888 },
  { event := event138919
    frameStart := 138888 },
  { event := event138920
    frameStart := 138888 },
  { event := event138921
    frameStart := 138888 },
  { event := event138922
    frameStart := 138888 },
  { event := event138923
    frameStart := 138888 },
  { event := event138924
    frameStart := 138888 },
  { event := event138925
    frameStart := 138888 },
  { event := event138926
    frameStart := 138888 },
  { event := event138927
    frameStart := 138888 }
]

def eventLeaf8683 : Array AnnotatedEvent := #[
  { event := event138928
    frameStart := 138888 },
  { event := event138929
    frameStart := 138888 },
  { event := event138930
    frameStart := 138888 },
  { event := event138931
    frameStart := 138888 },
  { event := event138932
    frameStart := 138888 },
  { event := event138933
    frameStart := 138888 },
  { event := event138934
    frameStart := 138888 },
  { event := event138935
    frameStart := 138888 },
  { event := event138936
    frameStart := 138888 },
  { event := event138937
    frameStart := 138888 },
  { event := event138938
    frameStart := 138888 },
  { event := event138939
    frameStart := 138888 },
  { event := event138940
    frameStart := 138888 },
  { event := event138941
    frameStart := 138888 },
  { event := event138942
    frameStart := 138888 },
  { event := event138943
    frameStart := 138888 }
]

def eventLeaf8684 : Array AnnotatedEvent := #[
  { event := event138944
    frameStart := 138888 },
  { event := event138945
    frameStart := 138888 },
  { event := event138946
    frameStart := 138888 },
  { event := event138947
    frameStart := 138888 },
  { event := event138948
    frameStart := 138888 },
  { event := event138949
    frameStart := 138888 },
  { event := event138950
    frameStart := 138888 },
  { event := event138951
    frameStart := 138888 },
  { event := event138952
    frameStart := 138888 },
  { event := event138953
    frameStart := 138888 },
  { event := event138954
    frameStart := 138888 },
  { event := event138955
    frameStart := 138888 },
  { event := event138956
    frameStart := 138888 },
  { event := event138957
    frameStart := 138888 },
  { event := event138958
    frameStart := 138888 },
  { event := event138959
    frameStart := 138888 }
]

def eventLeaf8685 : Array AnnotatedEvent := #[
  { event := event138960
    frameStart := 138888 },
  { event := event138961
    frameStart := 138888 },
  { event := event138962
    frameStart := 138888 },
  { event := event138963
    frameStart := 138888 },
  { event := event138964
    frameStart := 138888 },
  { event := event138965
    frameStart := 138888 },
  { event := event138966
    frameStart := 138888 },
  { event := event138967
    frameStart := 138888 },
  { event := event138968
    frameStart := 138888 },
  { event := event138969
    frameStart := 138888 },
  { event := event138970
    frameStart := 138888 },
  { event := event138971
    frameStart := 138888 },
  { event := event138972
    frameStart := 138888 },
  { event := event138973
    frameStart := 138888 },
  { event := event138974
    frameStart := 138888 },
  { event := event138975
    frameStart := 138888 }
]

def eventLeaf8686 : Array AnnotatedEvent := #[
  { event := event138976
    frameStart := 138888 },
  { event := event138977
    frameStart := 138888 },
  { event := event138978
    frameStart := 138888 },
  { event := event138979
    frameStart := 138888 },
  { event := event138980
    frameStart := 138888 },
  { event := event138981
    frameStart := 138888 },
  { event := event138982
    frameStart := 138888 },
  { event := event138983
    frameStart := 138888 },
  { event := event138984
    frameStart := 138888 },
  { event := event138985
    frameStart := 138888 },
  { event := event138986
    frameStart := 138888 },
  { event := event138987
    frameStart := 138888 },
  { event := event138988
    frameStart := 138888 },
  { event := event138989
    frameStart := 138888 },
  { event := event138990
    frameStart := 138888 },
  { event := event138991
    frameStart := 138888 }
]

def eventLeaf8687 : Array AnnotatedEvent := #[
  { event := event138992
    frameStart := 138888 },
  { event := event138993
    frameStart := 138888 },
  { event := event138994
    frameStart := 138888 },
  { event := event138995
    frameStart := 138888 },
  { event := event138996
    frameStart := 138888 },
  { event := event138997
    frameStart := 138888 },
  { event := event138998
    frameStart := 138888 },
  { event := event138999
    frameStart := 138888 },
  { event := event139000
    frameStart := 138888 },
  { event := event139001
    frameStart := 138888 },
  { event := event139002
    frameStart := 138888 },
  { event := event139003
    frameStart := 138888 },
  { event := event139004
    frameStart := 138888 },
  { event := event139005
    frameStart := 138888 },
  { event := event139006
    frameStart := 0 },
  { event := event139007
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events542
