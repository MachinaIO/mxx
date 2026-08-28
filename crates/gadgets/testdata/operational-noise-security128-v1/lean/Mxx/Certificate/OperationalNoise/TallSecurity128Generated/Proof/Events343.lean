import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events343

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event87808 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26457⟩⟩) (.identity (.predecessor 0 87807 .coefficient))

def event87809 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26457⟩⟩) (.finite 30)

def event87810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27613⟩⟩) 0 ⟨26457⟩ 87809

def event87811 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27613⟩⟩) (.authority (.programFamilyFact))

def event87812 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27613⟩⟩) (.finite 3720)

def event87813 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event87814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27614⟩⟩) 0 ⟨7177⟩ 87813

def event87815 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27614⟩⟩) 1 ⟨27613⟩ 87812

def event87816 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27614⟩⟩) (.authority (.operator))

def exact87817RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27614⟩⟩]⟩, (1)⟩]

theorem exact87817RawTermsValid :
    exact87817RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87817 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27614⟩⟩) exact87817RawTerms .large 87816 .exactZero (none)

def event87818 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28433⟩⟩) 0 ⟨27614⟩ 87817

def event87819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28433⟩⟩) (.authority (.operator))

def exact87820RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨28433⟩⟩]⟩, (1)⟩]

theorem exact87820RawTermsValid :
    exact87820RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87820 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28433⟩⟩) exact87820RawTerms (.finite 8192) 87819 .exactZero (none)

def event87821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event87822 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event87823 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27790⟩⟩) 0 ⟨26457⟩ 87809

def event87824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27790⟩⟩) 1 ⟨136⟩ 87822

def event87825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27790⟩⟩) (.sum [.predecessor 0 87823 .coefficient, .predecessor 1 87824 .coefficient])

def event87826 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27790⟩⟩) (.finite 30)

def event87827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27791⟩⟩) 0 ⟨27790⟩ 87826

def event87828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27791⟩⟩) (.identity (.predecessor 0 87827 .coefficient))

def exact87829RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], []⟩, (1)⟩]

theorem exact87829RawTermsValid :
    exact87829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87829 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27791⟩⟩) exact87829RawTerms (.finite 30) 87828 .exactZero (none)

def event87830 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact87831RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact87831RawTermsValid :
    exact87831RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87831 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact87831RawTerms .large 87830 .exactZero (none)

def event87832 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27792⟩⟩) 0 ⟨6908⟩ 87831

def event87833 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27792⟩⟩) 1 ⟨27791⟩ 87829

def event87834 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27792⟩⟩) (.product (.predecessor 0 87832 .coefficient) (.predecessor 1 87833 .coefficient) (⟨false, false, none, none, none⟩))

def event87835 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27792⟩⟩, .operator (⟨87831, 0⟩, ⟨87829, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact87836RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact87836RawTermsValid :
    exact87836RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87836 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27792⟩⟩) exact87836RawTerms .large 87834 .exactZero (none)

def event87837 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7189⟩⟩) 0 ⟨7177⟩ 87813

def event87838 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7189⟩⟩) (.authority (.operator))

def exact87839RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩]

theorem exact87839RawTermsValid :
    exact87839RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87839 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7189⟩⟩) exact87839RawTerms .large 87838 .exactZero (none)

def event87840 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27793⟩⟩) 0 ⟨7189⟩ 87839

def event87841 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27793⟩⟩) 1 ⟨27792⟩ 87836

def event87842 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27793⟩⟩) (.sum [.predecessor 0 87840 .coefficient, .predecessor 1 87841 .coefficient])

def exact87843RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact87843RawTermsValid :
    exact87843RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87843 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27793⟩⟩) exact87843RawTerms .large 87842 .exactZero (none)

def event87844 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28434⟩⟩) 0 ⟨27793⟩ 87843

def event87845 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28434⟩⟩) 1 ⟨28433⟩ 87820

def event87846 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28434⟩⟩) (.product (.predecessor 0 87844 .coefficient) (.predecessor 1 87845 .coefficient) (⟨false, false, none, none, none⟩))

def event87847 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28434⟩⟩, .operator (⟨87843, 0⟩, ⟨87820, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28433⟩⟩]⟩, (1)⟩)

def event87848 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28434⟩⟩, .operator (⟨87843, 1⟩, ⟨87820, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28433⟩⟩]⟩, (-1)⟩)

def event87849 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨28434⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28433⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28433⟩⟩) ⟨27614⟩ 87817)

def event87850 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28434⟩⟩, .relation 87849 0, ⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27614⟩⟩]⟩, (-1)⟩)

def exact87851RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28433⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27614⟩⟩]⟩, (-1)⟩]

theorem exact87851RawTermsValid :
    exact87851RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87851 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28434⟩⟩) exact87851RawTerms .large 87846 .exactZero (none)

def event87852 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26700⟩⟩) 0 ⟨26457⟩ 87809

def event87853 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26700⟩⟩) (.authority (.programFamilyFact))

def exact87854RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26700⟩⟩], []⟩, (1)⟩]

theorem exact87854RawTermsValid :
    exact87854RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87854 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26700⟩⟩) exact87854RawTerms (.finite 30) 87853 .exactZero (none)

def event87855 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26702⟩⟩) 0 ⟨6908⟩ 87831

def event87856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26702⟩⟩) 1 ⟨26700⟩ 87854

def event87857 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26702⟩⟩) (.product (.predecessor 0 87855 .coefficient) (.predecessor 1 87856 .coefficient) (⟨false, true, none, none, some 1⟩))

def event87858 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26702⟩⟩, .operator (⟨87831, 0⟩, ⟨87854, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact87859RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact87859RawTermsValid :
    exact87859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87859 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26702⟩⟩) exact87859RawTerms .large 87857 .exactZero (none)

def event87860 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7217⟩⟩) 0 ⟨7177⟩ 87813

def event87861 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7217⟩⟩) (.authority (.operator))

def exact87862RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩]

theorem exact87862RawTermsValid :
    exact87862RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87862 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7217⟩⟩) exact87862RawTerms .large 87861 .exactZero (none)

def event87863 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26703⟩⟩) 0 ⟨7217⟩ 87862

def event87864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26703⟩⟩) 1 ⟨26702⟩ 87859

def event87865 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26703⟩⟩) (.sum [.predecessor 0 87863 .coefficient, .predecessor 1 87864 .coefficient])

def exact87866RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact87866RawTermsValid :
    exact87866RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87866 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26703⟩⟩) exact87866RawTerms .large 87865 .exactZero (none)

def event87867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28438⟩⟩) 0 ⟨26703⟩ 87866

def event87868 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28438⟩⟩) 1 ⟨28434⟩ 87851

def event87869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28438⟩⟩) (.sum [.predecessor 0 87867 .coefficient, .predecessor 1 87868 .coefficient])

def exact87870RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28433⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27614⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact87870RawTermsValid :
    exact87870RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87870 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28438⟩⟩) exact87870RawTerms .large 87869 .exactZero (none)

def event87871 : Event := .preFoldPolynomial 87870 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28433⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27614⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact87872RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28433⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27614⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event87872 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨28438⟩⟩) 87871 exact87872RawTerms .large 87869 .exactZero (none)

def event87873 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨26457⟩⟩) ⟨⟨96⟩, ⟨78⟩, ⟨135⟩⟩ ⟨87715, 87873⟩

def event87874 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨27275⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27272⟩⟩]⟩) (1) 0 2 (.universal 87873 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27272⟩⟩]⟩) (none) 87872)

def event87875 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27275⟩⟩, .relation 87874 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩)

def event87876 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27275⟩⟩, .relation 87874 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28433⟩⟩]⟩, (-1)⟩)

def event87877 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27275⟩⟩, .relation 87874 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27614⟩⟩]⟩, (1)⟩)

def event87878 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27275⟩⟩, .relation 87874 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact87879RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28433⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27614⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact87879RawTermsValid :
    exact87879RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87879 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27275⟩⟩) exact87879RawTerms .large 87711 (.finite 202072841853861888) (some (87713))

def event87880 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28436⟩⟩) 0 ⟨27275⟩ 87879

def event87881 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28436⟩⟩) 1 ⟨28435⟩ 87701

def event87882 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28436⟩⟩) (.sum [.predecessor 0 87880 .coefficient, .predecessor 1 87881 .coefficient])

def event87883 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28436⟩⟩, .operator (⟨87879, 0⟩, ⟨87701, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28433⟩⟩]⟩, (1)⟩)

def event87884 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28436⟩⟩, .operator (⟨87879, 2⟩, ⟨87701, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27614⟩⟩]⟩, (-1)⟩)

def event87885 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28436⟩⟩) (.sum [.result 87879 .summary, .result 87701 .summary])

def exact87886RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact87886RawTermsValid :
    exact87886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87886 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28436⟩⟩) exact87886RawTerms .large 87882 (.finite 32191557518723330170883082027008) (some (87885))

def event87887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28437⟩⟩) 0 ⟨28436⟩ 87886

def event87888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28437⟩⟩) 1 ⟨7170⟩ 15682

def event87889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28437⟩⟩) (.product (.predecessor 0 87887 .coefficient) (.predecessor 1 87888 .coefficient) (⟨false, false, none, none, none⟩))

def event87890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28437⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩) [⟨.result 15678 .coefficient, false, none⟩])

def event87891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28437⟩⟩) (.product (.result 87886 .summary) (.transfer 87890) (⟨false, false, none, none, none⟩))

def event87892 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28437⟩⟩, .operator (⟨87886, 0⟩, ⟨15682, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩)

def event87893 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28437⟩⟩, .operator (⟨87886, 1⟩, ⟨15682, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (-1)⟩)

def event87894 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨28437⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7169⟩⟩) ⟨7050⟩ 15675)

def event87895 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28437⟩⟩, .relation 87894 0, ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact87896RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩]

theorem exact87896RawTermsValid :
    exact87896RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87896 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28437⟩⟩) exact87896RawTerms .large 87889 (.finite 345654216875549026890382321864211871825920) (some (87891))

def event87897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68735⟩⟩) 0 ⟨7177⟩ 15500

def event87898 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68735⟩⟩) 1 ⟨68734⟩ 79753

def event87899 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68735⟩⟩) (.authority (.operator))

def exact87900RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68735⟩⟩]⟩, (1)⟩]

theorem exact87900RawTermsValid :
    exact87900RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87900 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68735⟩⟩) exact87900RawTerms .large 87899 .exactZero (none)

def event87901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70636⟩⟩) 0 ⟨68735⟩ 87900

def event87902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70636⟩⟩) (.authority (.operator))

def exact87903RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩, (1)⟩]

theorem exact87903RawTermsValid :
    exact87903RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87903 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70636⟩⟩) exact87903RawTerms (.finite 8192) 87902 .exactZero (none)

def event87904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70638⟩⟩) 0 ⟨69308⟩ 80037

def event87905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70638⟩⟩) 1 ⟨70636⟩ 87903

def event87906 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70638⟩⟩) (.product (.predecessor 0 87904 .coefficient) (.predecessor 1 87905 .coefficient) (⟨false, false, none, none, none⟩))

def event87907 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70638⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩) [⟨.result 87903 .coefficient, false, none⟩])

def event87908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70638⟩⟩) (.product (.result 80037 .summary) (.transfer 87907) (⟨false, false, none, none, none⟩))

def event87909 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70638⟩⟩, .operator (⟨80037, 0⟩, ⟨87903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩, (1)⟩)

def event87910 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70638⟩⟩, .operator (⟨80037, 1⟩, ⟨87903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩, (-1)⟩)

def event87911 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70638⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70636⟩⟩) ⟨68735⟩ 87900)

def event87912 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70638⟩⟩, .relation 87911 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨68735⟩⟩]⟩, (-1)⟩)

def exact87913RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨68735⟩⟩]⟩, (-1)⟩]

theorem exact87913RawTermsValid :
    exact87913RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87913 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70638⟩⟩) exact87913RawTerms .large 87906 (.finite 32191361068277440720800338411520) (some (87908))

def event87914 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68193⟩⟩) 0 ⟨65837⟩ 3287

def event87915 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68193⟩⟩) (.authority (.relationPreimageSource ⟨75⟩))

def exact87916RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩, (1)⟩]

theorem exact87916RawTermsValid :
    exact87916RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87916 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68193⟩⟩) exact87916RawTerms (.finite 5647228698) 87915 .exactZero (none)

def event87917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68195⟩⟩) 0 ⟨68193⟩ 87916

def event87918 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68195⟩⟩) 1 ⟨2370⟩ 4

def event87919 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68195⟩⟩) (.scale (.predecessor 0 87917 .coefficient) (.value (.predecessor 1 87918 .coefficient)))

def exact87920RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩, (1)⟩]

theorem exact87920RawTermsValid :
    exact87920RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87920 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68195⟩⟩) exact87920RawTerms (.finite 5647228698) 87919 .exactZero (none)

def event87921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68196⟩⟩) 0 ⟨10368⟩ 75995

def event87922 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68196⟩⟩) 1 ⟨68195⟩ 87920

def event87923 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68196⟩⟩) (.product (.predecessor 0 87921 .coefficient) (.predecessor 1 87922 .coefficient) (⟨false, false, none, none, none⟩))

def event87924 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68196⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩) [⟨.result 87916 .coefficient, false, none⟩])

def event87925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68196⟩⟩) (.product (.result 75995 .summary) (.transfer 87924) (⟨false, false, none, none, none⟩))

def event87926 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68196⟩⟩, .operator (⟨75995, 0⟩, ⟨87920, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩, (1)⟩)

def event87927 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨68194⟩⟩)

def event87928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event87929 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event87930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event87931 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event87932 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event87933 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event87934 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event87935 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event87936 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 87935

def event87937 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 87933

def event87938 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 87936 .coefficient) (.value (.predecessor 1 87937 .coefficient)))

def event87939 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event87940 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 87939

def event87941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 87931

def event87942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 87940 .coefficient, .predecessor 1 87941 .coefficient])

def event87943 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event87944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 87943

def event87945 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 87929

def event87946 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 87945 .coefficient))

def event87947 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event87948 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25802⟩⟩) 0 ⟨10325⟩ 87947

def event87949 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25802⟩⟩) (.authority (.programFamilyFact))

def exact87950RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩], []⟩, (1)⟩]

theorem exact87950RawTermsValid :
    exact87950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87950 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25802⟩⟩) exact87950RawTerms (.finite 28) 87949 .exactZero (none)

def event87951 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65607⟩⟩) 0 ⟨10325⟩ 87947

def event87952 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65607⟩⟩) (.authority (.programFamilyFact))

def exact87953RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65607⟩⟩], []⟩, (1)⟩]

theorem exact87953RawTermsValid :
    exact87953RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87953 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65607⟩⟩) exact87953RawTerms (.finite 28) 87952 .exactZero (none)

def event87954 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65608⟩⟩) 0 ⟨65607⟩ 87953

def event87955 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65608⟩⟩) 1 ⟨25802⟩ 87950

def event87956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65608⟩⟩) (.product (.predecessor 0 87954 .coefficient) (.predecessor 1 87955 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event87957 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65608⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], []⟩) [⟨.result 87953 .coefficient, true, some 1⟩, ⟨.result 87950 .coefficient, true, some 1⟩])

def event87958 : Event := .survivorFold (1) 87957

def exact87959RawTerms : List Term := []

theorem exact87959RawTermsValid :
    exact87959RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87959 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65608⟩⟩) exact87959RawTerms (.finite 784) 87956 (.finite 784) (some (87957))

def event87960 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65609⟩⟩) 0 ⟨65608⟩ 87959

def event87961 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65609⟩⟩) (.identity (.predecessor 0 87960 .coefficient))

def event87962 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65609⟩⟩) (.finite 784)

def event87963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65836⟩⟩) 0 ⟨65609⟩ 87962

def event87964 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65836⟩⟩) (.authority (.programFamilyFact))

def exact87965RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], []⟩, (1)⟩]

theorem exact87965RawTermsValid :
    exact87965RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87965 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65836⟩⟩) exact87965RawTerms (.finite 28) 87964 .exactZero (none)

def event87966 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65837⟩⟩) 0 ⟨65836⟩ 87965

def event87967 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65837⟩⟩) (.identity (.predecessor 0 87966 .coefficient))

def event87968 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65837⟩⟩) (.finite 28)

def event87969 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68193⟩⟩) 0 ⟨65837⟩ 87968

def event87970 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68193⟩⟩) (.authority (.relationPreimageSource ⟨75⟩))

def exact87971RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩, (1)⟩]

theorem exact87971RawTermsValid :
    exact87971RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87971 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68193⟩⟩) exact87971RawTerms (.finite 5647228698) 87970 .exactZero (none)

def event87972 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact87973RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact87973RawTermsValid :
    exact87973RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87973 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact87973RawTerms .large 87972 .exactZero (none)

def event87974 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68194⟩⟩) 0 ⟨35⟩ 87973

def event87975 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68194⟩⟩) 1 ⟨68193⟩ 87971

def event87976 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68194⟩⟩) (.product (.predecessor 0 87974 .coefficient) (.predecessor 1 87975 .coefficient) (⟨false, false, none, none, none⟩))

def event87977 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68194⟩⟩, .operator (⟨87973, 0⟩, ⟨87971, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩, (1)⟩)

def exact87978RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩, (1)⟩]

theorem exact87978RawTermsValid :
    exact87978RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event87978 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68194⟩⟩) exact87978RawTerms .large 87976 .exactZero (none)

def event87979 : Event := .preFoldPolynomial 87978 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩, (1)⟩] .exactZero none

def exact87980RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩, (1)⟩]

def event87980 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨68194⟩⟩) 87979 exact87980RawTerms .large 87976 .exactZero (none)

def event87981 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨70650⟩⟩)

def event87982 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event87983 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event87984 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event87985 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event87986 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event87987 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event87988 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event87989 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event87990 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 87989

def event87991 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 87987

def event87992 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 87990 .coefficient) (.value (.predecessor 1 87991 .coefficient)))

def event87993 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event87994 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 87993

def event87995 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 87985

def event87996 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 87994 .coefficient, .predecessor 1 87995 .coefficient])

def event87997 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event87998 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 87997

def event87999 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 87983

def event88000 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 87999 .coefficient))

def event88001 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event88002 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25802⟩⟩) 0 ⟨10325⟩ 88001

def event88003 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25802⟩⟩) (.authority (.programFamilyFact))

def exact88004RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩], []⟩, (1)⟩]

theorem exact88004RawTermsValid :
    exact88004RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88004 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25802⟩⟩) exact88004RawTerms (.finite 28) 88003 .exactZero (none)

def event88005 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65607⟩⟩) 0 ⟨10325⟩ 88001

def event88006 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65607⟩⟩) (.authority (.programFamilyFact))

def exact88007RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65607⟩⟩], []⟩, (1)⟩]

theorem exact88007RawTermsValid :
    exact88007RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88007 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65607⟩⟩) exact88007RawTerms (.finite 28) 88006 .exactZero (none)

def event88008 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65608⟩⟩) 0 ⟨65607⟩ 88007

def event88009 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65608⟩⟩) 1 ⟨25802⟩ 88004

def event88010 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65608⟩⟩) (.product (.predecessor 0 88008 .coefficient) (.predecessor 1 88009 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event88011 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65608⟩⟩, .operator (⟨88007, 0⟩, ⟨88004, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], []⟩, (1)⟩)

def exact88012RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], []⟩, (1)⟩]

theorem exact88012RawTermsValid :
    exact88012RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88012 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65608⟩⟩) exact88012RawTerms (.finite 784) 88010 .exactZero (none)

def event88013 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65609⟩⟩) 0 ⟨65608⟩ 88012

def event88014 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65609⟩⟩) (.identity (.predecessor 0 88013 .coefficient))

def event88015 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65609⟩⟩) (.finite 784)

def event88016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65836⟩⟩) 0 ⟨65609⟩ 88015

def event88017 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65836⟩⟩) (.authority (.programFamilyFact))

def exact88018RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], []⟩, (1)⟩]

theorem exact88018RawTermsValid :
    exact88018RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88018 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65836⟩⟩) exact88018RawTerms (.finite 28) 88017 .exactZero (none)

def event88019 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65837⟩⟩) 0 ⟨65836⟩ 88018

def event88020 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65837⟩⟩) (.identity (.predecessor 0 88019 .coefficient))

def event88021 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65837⟩⟩) (.finite 28)

def event88022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68734⟩⟩) 0 ⟨65837⟩ 88021

def event88023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68734⟩⟩) (.authority (.programFamilyFact))

def event88024 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68734⟩⟩) (.finite 3720)

def event88025 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event88026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68735⟩⟩) 0 ⟨7177⟩ 88025

def event88027 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68735⟩⟩) 1 ⟨68734⟩ 88024

def event88028 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68735⟩⟩) (.authority (.operator))

def exact88029RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68735⟩⟩]⟩, (1)⟩]

theorem exact88029RawTermsValid :
    exact88029RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88029 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68735⟩⟩) exact88029RawTerms .large 88028 .exactZero (none)

def event88030 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70636⟩⟩) 0 ⟨68735⟩ 88029

def event88031 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70636⟩⟩) (.authority (.operator))

def exact88032RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩, (1)⟩]

theorem exact88032RawTermsValid :
    exact88032RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88032 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70636⟩⟩) exact88032RawTerms (.finite 8192) 88031 .exactZero (none)

def event88033 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event88034 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event88035 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69031⟩⟩) 0 ⟨65837⟩ 88021

def event88036 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69031⟩⟩) 1 ⟨136⟩ 88034

def event88037 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69031⟩⟩) (.sum [.predecessor 0 88035 .coefficient, .predecessor 1 88036 .coefficient])

def event88038 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨69031⟩⟩) (.finite 28)

def event88039 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69032⟩⟩) 0 ⟨69031⟩ 88038

def event88040 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69032⟩⟩) (.identity (.predecessor 0 88039 .coefficient))

def exact88041RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], []⟩, (1)⟩]

theorem exact88041RawTermsValid :
    exact88041RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88041 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69032⟩⟩) exact88041RawTerms (.finite 28) 88040 .exactZero (none)

def event88042 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact88043RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact88043RawTermsValid :
    exact88043RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88043 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact88043RawTerms .large 88042 .exactZero (none)

def event88044 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69033⟩⟩) 0 ⟨6908⟩ 88043

def event88045 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69033⟩⟩) 1 ⟨69032⟩ 88041

def event88046 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69033⟩⟩) (.product (.predecessor 0 88044 .coefficient) (.predecessor 1 88045 .coefficient) (⟨false, false, none, none, none⟩))

def event88047 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69033⟩⟩, .operator (⟨88043, 0⟩, ⟨88041, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact88048RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact88048RawTermsValid :
    exact88048RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88048 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69033⟩⟩) exact88048RawTerms .large 88046 .exactZero (none)

def event88049 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7188⟩⟩) 0 ⟨7177⟩ 88025

def event88050 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7188⟩⟩) (.authority (.operator))

def exact88051RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩]

theorem exact88051RawTermsValid :
    exact88051RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88051 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7188⟩⟩) exact88051RawTerms .large 88050 .exactZero (none)

def event88052 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69034⟩⟩) 0 ⟨7188⟩ 88051

def event88053 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69034⟩⟩) 1 ⟨69033⟩ 88048

def event88054 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69034⟩⟩) (.sum [.predecessor 0 88052 .coefficient, .predecessor 1 88053 .coefficient])

def exact88055RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact88055RawTermsValid :
    exact88055RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88055 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69034⟩⟩) exact88055RawTerms .large 88054 .exactZero (none)

def event88056 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70637⟩⟩) 0 ⟨69034⟩ 88055

def event88057 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70637⟩⟩) 1 ⟨70636⟩ 88032

def event88058 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70637⟩⟩) (.product (.predecessor 0 88056 .coefficient) (.predecessor 1 88057 .coefficient) (⟨false, false, none, none, none⟩))

def event88059 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70637⟩⟩, .operator (⟨88055, 0⟩, ⟨88032, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩, (1)⟩)

def event88060 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70637⟩⟩, .operator (⟨88055, 1⟩, ⟨88032, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩, (-1)⟩)

def event88061 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70637⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70636⟩⟩) ⟨68735⟩ 88029)

def event88062 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70637⟩⟩, .relation 88061 0, ⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨68735⟩⟩]⟩, (-1)⟩)

def exact88063RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨68735⟩⟩]⟩, (-1)⟩]

theorem exact88063RawTermsValid :
    exact88063RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88063 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70637⟩⟩) exact88063RawTerms .large 88058 .exactZero (none)

def eventLeaf5488 : Array AnnotatedEvent := #[
  { event := event87808
    frameStart := 87769 },
  { event := event87809
    frameStart := 87769 },
  { event := event87810
    frameStart := 87769 },
  { event := event87811
    frameStart := 87769 },
  { event := event87812
    frameStart := 87769 },
  { event := event87813
    frameStart := 87769 },
  { event := event87814
    frameStart := 87769 },
  { event := event87815
    frameStart := 87769 },
  { event := event87816
    frameStart := 87769 },
  { event := event87817
    frameStart := 87769 },
  { event := event87818
    frameStart := 87769 },
  { event := event87819
    frameStart := 87769 },
  { event := event87820
    frameStart := 87769 },
  { event := event87821
    frameStart := 87769 },
  { event := event87822
    frameStart := 87769 },
  { event := event87823
    frameStart := 87769 }
]

def eventLeaf5489 : Array AnnotatedEvent := #[
  { event := event87824
    frameStart := 87769 },
  { event := event87825
    frameStart := 87769 },
  { event := event87826
    frameStart := 87769 },
  { event := event87827
    frameStart := 87769 },
  { event := event87828
    frameStart := 87769 },
  { event := event87829
    frameStart := 87769 },
  { event := event87830
    frameStart := 87769 },
  { event := event87831
    frameStart := 87769 },
  { event := event87832
    frameStart := 87769 },
  { event := event87833
    frameStart := 87769 },
  { event := event87834
    frameStart := 87769 },
  { event := event87835
    frameStart := 87769 },
  { event := event87836
    frameStart := 87769 },
  { event := event87837
    frameStart := 87769 },
  { event := event87838
    frameStart := 87769 },
  { event := event87839
    frameStart := 87769 }
]

def eventLeaf5490 : Array AnnotatedEvent := #[
  { event := event87840
    frameStart := 87769 },
  { event := event87841
    frameStart := 87769 },
  { event := event87842
    frameStart := 87769 },
  { event := event87843
    frameStart := 87769 },
  { event := event87844
    frameStart := 87769 },
  { event := event87845
    frameStart := 87769 },
  { event := event87846
    frameStart := 87769 },
  { event := event87847
    frameStart := 87769 },
  { event := event87848
    frameStart := 87769 },
  { event := event87849
    frameStart := 87769 },
  { event := event87850
    frameStart := 87769 },
  { event := event87851
    frameStart := 87769 },
  { event := event87852
    frameStart := 87769 },
  { event := event87853
    frameStart := 87769 },
  { event := event87854
    frameStart := 87769 },
  { event := event87855
    frameStart := 87769 }
]

def eventLeaf5491 : Array AnnotatedEvent := #[
  { event := event87856
    frameStart := 87769 },
  { event := event87857
    frameStart := 87769 },
  { event := event87858
    frameStart := 87769 },
  { event := event87859
    frameStart := 87769 },
  { event := event87860
    frameStart := 87769 },
  { event := event87861
    frameStart := 87769 },
  { event := event87862
    frameStart := 87769 },
  { event := event87863
    frameStart := 87769 },
  { event := event87864
    frameStart := 87769 },
  { event := event87865
    frameStart := 87769 },
  { event := event87866
    frameStart := 87769 },
  { event := event87867
    frameStart := 87769 },
  { event := event87868
    frameStart := 87769 },
  { event := event87869
    frameStart := 87769 },
  { event := event87870
    frameStart := 87769 },
  { event := event87871
    frameStart := 87769 }
]

def eventLeaf5492 : Array AnnotatedEvent := #[
  { event := event87872
    frameStart := 87769 },
  { event := event87873
    frameStart := 0 },
  { event := event87874
    frameStart := 0 },
  { event := event87875
    frameStart := 0 },
  { event := event87876
    frameStart := 0 },
  { event := event87877
    frameStart := 0 },
  { event := event87878
    frameStart := 0 },
  { event := event87879
    frameStart := 0 },
  { event := event87880
    frameStart := 0 },
  { event := event87881
    frameStart := 0 },
  { event := event87882
    frameStart := 0 },
  { event := event87883
    frameStart := 0 },
  { event := event87884
    frameStart := 0 },
  { event := event87885
    frameStart := 0 },
  { event := event87886
    frameStart := 0 },
  { event := event87887
    frameStart := 0 }
]

def eventLeaf5493 : Array AnnotatedEvent := #[
  { event := event87888
    frameStart := 0 },
  { event := event87889
    frameStart := 0 },
  { event := event87890
    frameStart := 0 },
  { event := event87891
    frameStart := 0 },
  { event := event87892
    frameStart := 0 },
  { event := event87893
    frameStart := 0 },
  { event := event87894
    frameStart := 0 },
  { event := event87895
    frameStart := 0 },
  { event := event87896
    frameStart := 0 },
  { event := event87897
    frameStart := 0 },
  { event := event87898
    frameStart := 0 },
  { event := event87899
    frameStart := 0 },
  { event := event87900
    frameStart := 0 },
  { event := event87901
    frameStart := 0 },
  { event := event87902
    frameStart := 0 },
  { event := event87903
    frameStart := 0 }
]

def eventLeaf5494 : Array AnnotatedEvent := #[
  { event := event87904
    frameStart := 0 },
  { event := event87905
    frameStart := 0 },
  { event := event87906
    frameStart := 0 },
  { event := event87907
    frameStart := 0 },
  { event := event87908
    frameStart := 0 },
  { event := event87909
    frameStart := 0 },
  { event := event87910
    frameStart := 0 },
  { event := event87911
    frameStart := 0 },
  { event := event87912
    frameStart := 0 },
  { event := event87913
    frameStart := 0 },
  { event := event87914
    frameStart := 0 },
  { event := event87915
    frameStart := 0 },
  { event := event87916
    frameStart := 0 },
  { event := event87917
    frameStart := 0 },
  { event := event87918
    frameStart := 0 },
  { event := event87919
    frameStart := 0 }
]

def eventLeaf5495 : Array AnnotatedEvent := #[
  { event := event87920
    frameStart := 0 },
  { event := event87921
    frameStart := 0 },
  { event := event87922
    frameStart := 0 },
  { event := event87923
    frameStart := 0 },
  { event := event87924
    frameStart := 0 },
  { event := event87925
    frameStart := 0 },
  { event := event87926
    frameStart := 0 },
  { event := event87927
    frameStart := 87927 },
  { event := event87928
    frameStart := 87927 },
  { event := event87929
    frameStart := 87927 },
  { event := event87930
    frameStart := 87927 },
  { event := event87931
    frameStart := 87927 },
  { event := event87932
    frameStart := 87927 },
  { event := event87933
    frameStart := 87927 },
  { event := event87934
    frameStart := 87927 },
  { event := event87935
    frameStart := 87927 }
]

def eventLeaf5496 : Array AnnotatedEvent := #[
  { event := event87936
    frameStart := 87927 },
  { event := event87937
    frameStart := 87927 },
  { event := event87938
    frameStart := 87927 },
  { event := event87939
    frameStart := 87927 },
  { event := event87940
    frameStart := 87927 },
  { event := event87941
    frameStart := 87927 },
  { event := event87942
    frameStart := 87927 },
  { event := event87943
    frameStart := 87927 },
  { event := event87944
    frameStart := 87927 },
  { event := event87945
    frameStart := 87927 },
  { event := event87946
    frameStart := 87927 },
  { event := event87947
    frameStart := 87927 },
  { event := event87948
    frameStart := 87927 },
  { event := event87949
    frameStart := 87927 },
  { event := event87950
    frameStart := 87927 },
  { event := event87951
    frameStart := 87927 }
]

def eventLeaf5497 : Array AnnotatedEvent := #[
  { event := event87952
    frameStart := 87927 },
  { event := event87953
    frameStart := 87927 },
  { event := event87954
    frameStart := 87927 },
  { event := event87955
    frameStart := 87927 },
  { event := event87956
    frameStart := 87927 },
  { event := event87957
    frameStart := 87927 },
  { event := event87958
    frameStart := 87927 },
  { event := event87959
    frameStart := 87927 },
  { event := event87960
    frameStart := 87927 },
  { event := event87961
    frameStart := 87927 },
  { event := event87962
    frameStart := 87927 },
  { event := event87963
    frameStart := 87927 },
  { event := event87964
    frameStart := 87927 },
  { event := event87965
    frameStart := 87927 },
  { event := event87966
    frameStart := 87927 },
  { event := event87967
    frameStart := 87927 }
]

def eventLeaf5498 : Array AnnotatedEvent := #[
  { event := event87968
    frameStart := 87927 },
  { event := event87969
    frameStart := 87927 },
  { event := event87970
    frameStart := 87927 },
  { event := event87971
    frameStart := 87927 },
  { event := event87972
    frameStart := 87927 },
  { event := event87973
    frameStart := 87927 },
  { event := event87974
    frameStart := 87927 },
  { event := event87975
    frameStart := 87927 },
  { event := event87976
    frameStart := 87927 },
  { event := event87977
    frameStart := 87927 },
  { event := event87978
    frameStart := 87927 },
  { event := event87979
    frameStart := 87927 },
  { event := event87980
    frameStart := 87927 },
  { event := event87981
    frameStart := 87981 },
  { event := event87982
    frameStart := 87981 },
  { event := event87983
    frameStart := 87981 }
]

def eventLeaf5499 : Array AnnotatedEvent := #[
  { event := event87984
    frameStart := 87981 },
  { event := event87985
    frameStart := 87981 },
  { event := event87986
    frameStart := 87981 },
  { event := event87987
    frameStart := 87981 },
  { event := event87988
    frameStart := 87981 },
  { event := event87989
    frameStart := 87981 },
  { event := event87990
    frameStart := 87981 },
  { event := event87991
    frameStart := 87981 },
  { event := event87992
    frameStart := 87981 },
  { event := event87993
    frameStart := 87981 },
  { event := event87994
    frameStart := 87981 },
  { event := event87995
    frameStart := 87981 },
  { event := event87996
    frameStart := 87981 },
  { event := event87997
    frameStart := 87981 },
  { event := event87998
    frameStart := 87981 },
  { event := event87999
    frameStart := 87981 }
]

def eventLeaf5500 : Array AnnotatedEvent := #[
  { event := event88000
    frameStart := 87981 },
  { event := event88001
    frameStart := 87981 },
  { event := event88002
    frameStart := 87981 },
  { event := event88003
    frameStart := 87981 },
  { event := event88004
    frameStart := 87981 },
  { event := event88005
    frameStart := 87981 },
  { event := event88006
    frameStart := 87981 },
  { event := event88007
    frameStart := 87981 },
  { event := event88008
    frameStart := 87981 },
  { event := event88009
    frameStart := 87981 },
  { event := event88010
    frameStart := 87981 },
  { event := event88011
    frameStart := 87981 },
  { event := event88012
    frameStart := 87981 },
  { event := event88013
    frameStart := 87981 },
  { event := event88014
    frameStart := 87981 },
  { event := event88015
    frameStart := 87981 }
]

def eventLeaf5501 : Array AnnotatedEvent := #[
  { event := event88016
    frameStart := 87981 },
  { event := event88017
    frameStart := 87981 },
  { event := event88018
    frameStart := 87981 },
  { event := event88019
    frameStart := 87981 },
  { event := event88020
    frameStart := 87981 },
  { event := event88021
    frameStart := 87981 },
  { event := event88022
    frameStart := 87981 },
  { event := event88023
    frameStart := 87981 },
  { event := event88024
    frameStart := 87981 },
  { event := event88025
    frameStart := 87981 },
  { event := event88026
    frameStart := 87981 },
  { event := event88027
    frameStart := 87981 },
  { event := event88028
    frameStart := 87981 },
  { event := event88029
    frameStart := 87981 },
  { event := event88030
    frameStart := 87981 },
  { event := event88031
    frameStart := 87981 }
]

def eventLeaf5502 : Array AnnotatedEvent := #[
  { event := event88032
    frameStart := 87981 },
  { event := event88033
    frameStart := 87981 },
  { event := event88034
    frameStart := 87981 },
  { event := event88035
    frameStart := 87981 },
  { event := event88036
    frameStart := 87981 },
  { event := event88037
    frameStart := 87981 },
  { event := event88038
    frameStart := 87981 },
  { event := event88039
    frameStart := 87981 },
  { event := event88040
    frameStart := 87981 },
  { event := event88041
    frameStart := 87981 },
  { event := event88042
    frameStart := 87981 },
  { event := event88043
    frameStart := 87981 },
  { event := event88044
    frameStart := 87981 },
  { event := event88045
    frameStart := 87981 },
  { event := event88046
    frameStart := 87981 },
  { event := event88047
    frameStart := 87981 }
]

def eventLeaf5503 : Array AnnotatedEvent := #[
  { event := event88048
    frameStart := 87981 },
  { event := event88049
    frameStart := 87981 },
  { event := event88050
    frameStart := 87981 },
  { event := event88051
    frameStart := 87981 },
  { event := event88052
    frameStart := 87981 },
  { event := event88053
    frameStart := 87981 },
  { event := event88054
    frameStart := 87981 },
  { event := event88055
    frameStart := 87981 },
  { event := event88056
    frameStart := 87981 },
  { event := event88057
    frameStart := 87981 },
  { event := event88058
    frameStart := 87981 },
  { event := event88059
    frameStart := 87981 },
  { event := event88060
    frameStart := 87981 },
  { event := event88061
    frameStart := 87981 },
  { event := event88062
    frameStart := 87981 },
  { event := event88063
    frameStart := 87981 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events343
