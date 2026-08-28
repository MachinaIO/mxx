import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1183

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event302848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17167⟩⟩) (.identity (.predecessor 0 302847 .coefficient))

def exact302849RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], []⟩, (1)⟩]

theorem exact302849RawTermsValid :
    exact302849RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302849 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17167⟩⟩) exact302849RawTerms (.finite 2) 302848 .exactZero (none)

def event302850 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact302851RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact302851RawTermsValid :
    exact302851RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302851 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact302851RawTerms .large 302850 .exactZero (none)

def event302852 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17168⟩⟩) 0 ⟨6908⟩ 302851

def event302853 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17168⟩⟩) 1 ⟨17167⟩ 302849

def event302854 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17168⟩⟩) (.product (.predecessor 0 302852 .coefficient) (.predecessor 1 302853 .coefficient) (⟨false, false, none, none, none⟩))

def event302855 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17168⟩⟩, .operator (⟨302851, 0⟩, ⟨302849, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact302856RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact302856RawTermsValid :
    exact302856RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302856 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17168⟩⟩) exact302856RawTerms .large 302854 .exactZero (none)

def event302857 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 302833

def event302858 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact302859RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact302859RawTermsValid :
    exact302859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302859 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact302859RawTerms .large 302858 .exactZero (none)

def event302860 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17169⟩⟩) 0 ⟨7179⟩ 302859

def event302861 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17169⟩⟩) 1 ⟨17168⟩ 302856

def event302862 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17169⟩⟩) (.sum [.predecessor 0 302860 .coefficient, .predecessor 1 302861 .coefficient])

def exact302863RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302863RawTermsValid :
    exact302863RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302863 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17169⟩⟩) exact302863RawTerms .large 302862 .exactZero (none)

def event302864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17482⟩⟩) 0 ⟨17169⟩ 302863

def event302865 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17482⟩⟩) 1 ⟨17481⟩ 302840

def event302866 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17482⟩⟩) (.product (.predecessor 0 302864 .coefficient) (.predecessor 1 302865 .coefficient) (⟨false, false, none, none, none⟩))

def event302867 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17482⟩⟩, .operator (⟨302863, 0⟩, ⟨302840, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17481⟩⟩]⟩, (1)⟩)

def event302868 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17482⟩⟩, .operator (⟨302863, 1⟩, ⟨302840, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17481⟩⟩]⟩, (-1)⟩)

def event302869 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17482⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17481⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17481⟩⟩) ⟨16911⟩ 302837)

def event302870 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17482⟩⟩, .relation 302869 0, ⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨16911⟩⟩]⟩, (-1)⟩)

def exact302871RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17481⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨16911⟩⟩]⟩, (-1)⟩]

theorem exact302871RawTermsValid :
    exact302871RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302871 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17482⟩⟩) exact302871RawTerms .large 302866 .exactZero (none)

def event302872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15875⟩⟩) 0 ⟨15709⟩ 302829

def event302873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15875⟩⟩) (.authority (.programFamilyFact))

def exact302874RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15875⟩⟩], []⟩, (1)⟩]

theorem exact302874RawTermsValid :
    exact302874RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302874 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15875⟩⟩) exact302874RawTerms (.finite 43) 302873 .exactZero (none)

def event302875 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15876⟩⟩) 0 ⟨6908⟩ 302851

def event302876 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15876⟩⟩) 1 ⟨15875⟩ 302874

def event302877 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15876⟩⟩) (.product (.predecessor 0 302875 .coefficient) (.predecessor 1 302876 .coefficient) (⟨false, true, none, none, some 1⟩))

def event302878 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15876⟩⟩, .operator (⟨302851, 0⟩, ⟨302874, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact302879RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact302879RawTermsValid :
    exact302879RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302879 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15876⟩⟩) exact302879RawTerms .large 302877 .exactZero (none)

def event302880 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 302833

def event302881 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact302882RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact302882RawTermsValid :
    exact302882RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302882 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact302882RawTerms .large 302881 .exactZero (none)

def event302883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15877⟩⟩) 0 ⟨7198⟩ 302882

def event302884 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15877⟩⟩) 1 ⟨15876⟩ 302879

def event302885 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15877⟩⟩) (.sum [.predecessor 0 302883 .coefficient, .predecessor 1 302884 .coefficient])

def exact302886RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302886RawTermsValid :
    exact302886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302886 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15877⟩⟩) exact302886RawTerms .large 302885 .exactZero (none)

def event302887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17485⟩⟩) 0 ⟨15877⟩ 302886

def event302888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17485⟩⟩) 1 ⟨17482⟩ 302871

def event302889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17485⟩⟩) (.sum [.predecessor 0 302887 .coefficient, .predecessor 1 302888 .coefficient])

def exact302890RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17481⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨16911⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302890RawTermsValid :
    exact302890RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302890 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17485⟩⟩) exact302890RawTerms .large 302889 .exactZero (none)

def event302891 : Event := .preFoldPolynomial 302890 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17481⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨16911⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact302892RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17481⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨16911⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event302892 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17485⟩⟩) 302891 exact302892RawTerms .large 302889 .exactZero (none)

def event302893 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15709⟩⟩) ⟨⟨77⟩, ⟨57⟩, ⟨135⟩⟩ ⟨302759, 302893⟩

def event302894 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16399⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16396⟩⟩]⟩) (1) 0 2 (.universal 302893 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16396⟩⟩]⟩) (none) 302892)

def event302895 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16399⟩⟩, .relation 302894 1, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩)

def event302896 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16399⟩⟩, .relation 302894 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17481⟩⟩]⟩, (-1)⟩)

def event302897 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16399⟩⟩, .relation 302894 2, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨16911⟩⟩]⟩, (1)⟩)

def event302898 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16399⟩⟩, .relation 302894 3, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact302899RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17481⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨16911⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302899RawTermsValid :
    exact302899RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302899 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16399⟩⟩) exact302899RawTerms .large 302755 (.finite 202072841853861888) (some (302757))

def event302900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17484⟩⟩) 0 ⟨16399⟩ 302899

def event302901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17484⟩⟩) 1 ⟨17483⟩ 302745

def event302902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17484⟩⟩) (.sum [.predecessor 0 302900 .coefficient, .predecessor 1 302901 .coefficient])

def event302903 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17484⟩⟩, .operator (⟨302899, 0⟩, ⟨302745, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17481⟩⟩]⟩, (1)⟩)

def event302904 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17484⟩⟩, .operator (⟨302899, 2⟩, ⟨302745, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨16911⟩⟩]⟩, (-1)⟩)

def event302905 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17484⟩⟩) (.sum [.result 302899 .summary, .result 302745 .summary])

def exact302906RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302906RawTermsValid :
    exact302906RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302906 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17484⟩⟩) exact302906RawTerms .large 302902 (.finite 32188807212483706889510625476608) (some (302905))

def event302907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20346⟩⟩) 0 ⟨17484⟩ 302906

def event302908 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20346⟩⟩) 1 ⟨20345⟩ 302472

def event302909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20346⟩⟩) (.sum [.predecessor 0 302907 .coefficient, .predecessor 1 302908 .coefficient])

def event302910 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20346⟩⟩) (.sum [.result 302906 .summary, .result 302472 .summary])

def exact302911RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302911RawTermsValid :
    exact302911RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302911 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20346⟩⟩) exact302911RawTerms .large 302909 (.finite 64377712650190257467641695830016) (some (302910))

def event302912 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23566⟩⟩) 0 ⟨20346⟩ 302911

def event302913 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23566⟩⟩) 1 ⟨23565⟩ 302038

def event302914 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23566⟩⟩) (.sum [.predecessor 0 302912 .coefficient, .predecessor 1 302913 .coefficient])

def event302915 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23566⟩⟩) (.sum [.result 302911 .summary, .result 302038 .summary])

def exact302916RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302916RawTermsValid :
    exact302916RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302916 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23566⟩⟩) exact302916RawTerms .large 302914 (.finite 96566716313119651734393211060224) (some (302915))

def event302917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33586⟩⟩) 0 ⟨23566⟩ 302916

def event302918 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33586⟩⟩) 1 ⟨33585⟩ 301604

def event302919 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33586⟩⟩) (.sum [.predecessor 0 302917 .coefficient, .predecessor 1 302918 .coefficient])

def event302920 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33586⟩⟩) (.sum [.result 302916 .summary, .result 301604 .summary])

def exact302921RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302921RawTermsValid :
    exact302921RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302921 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33586⟩⟩) exact302921RawTerms .large 302919 (.finite 128755916426494733378385616044032) (some (302920))

def event302922 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52646⟩⟩) 0 ⟨33586⟩ 302921

def event302923 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52646⟩⟩) 1 ⟨52645⟩ 301170

def event302924 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52646⟩⟩) (.sum [.predecessor 0 302922 .coefficient, .predecessor 1 302923 .coefficient])

def event302925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52646⟩⟩) (.sum [.result 302921 .summary, .result 301170 .summary])

def exact302926RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302926RawTermsValid :
    exact302926RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302926 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52646⟩⟩) exact302926RawTerms .large 302924 (.finite 160945509440761189776859800535040) (some (302925))

def event302927 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55626⟩⟩) 0 ⟨52646⟩ 302926

def event302928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55626⟩⟩) 1 ⟨55625⟩ 300736

def event302929 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55626⟩⟩) (.sum [.predecessor 0 302927 .coefficient, .predecessor 1 302928 .coefficient])

def event302930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55626⟩⟩) (.sum [.result 302926 .summary, .result 300736 .summary])

def exact302931RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302931RawTermsValid :
    exact302931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302931 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55626⟩⟩) exact302931RawTerms .large 302929 (.finite 193135298905473333552574874779648) (some (302930))

def event302932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58606⟩⟩) 0 ⟨55626⟩ 302931

def event302933 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58606⟩⟩) 1 ⟨58605⟩ 300302

def event302934 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58606⟩⟩) (.sum [.predecessor 0 302932 .coefficient, .predecessor 1 302933 .coefficient])

def event302935 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58606⟩⟩) (.sum [.result 302931 .summary, .result 300302 .summary])

def exact302936RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56931⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302936RawTermsValid :
    exact302936RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302936 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58606⟩⟩) exact302936RawTerms .large 302934 (.finite 225325481271076852082771728531456) (some (302935))

def event302937 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61586⟩⟩) 0 ⟨58606⟩ 302936

def event302938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61586⟩⟩) 1 ⟨61585⟩ 299868

def event302939 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61586⟩⟩) (.sum [.predecessor 0 302937 .coefficient, .predecessor 1 302938 .coefficient])

def event302940 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61586⟩⟩) (.sum [.result 302936 .summary, .result 299868 .summary])

def exact302941RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56931⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302941RawTermsValid :
    exact302941RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302941 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61586⟩⟩) exact302941RawTerms .large 302939 (.finite 257515860087126057990209472036864) (some (302940))

def event302942 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64566⟩⟩) 0 ⟨61586⟩ 302941

def event302943 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64566⟩⟩) 1 ⟨64565⟩ 299434

def event302944 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64566⟩⟩) (.sum [.predecessor 0 302942 .coefficient, .predecessor 1 302943 .coefficient])

def event302945 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64566⟩⟩) (.sum [.result 302941 .summary, .result 299434 .summary])

def exact302946RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56931⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨62891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302946RawTermsValid :
    exact302946RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302946 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64566⟩⟩) exact302946RawTerms .large 302944 (.finite 289706631804066638652128995049472) (some (302945))

def event302947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69391⟩⟩) 0 ⟨64566⟩ 302946

def event302948 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69391⟩⟩) 1 ⟨69390⟩ 299000

def event302949 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69391⟩⟩) (.sum [.predecessor 0 302947 .coefficient, .predecessor 1 302948 .coefficient])

def event302950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69391⟩⟩) (.sum [.result 302946 .summary, .result 299000 .summary])

def exact302951RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56931⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨62891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302951RawTermsValid :
    exact302951RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302951 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69391⟩⟩) exact302951RawTerms .large 302949 (.finite 321897992872344281445771187322880) (some (302950))

def event302952 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69392⟩⟩) 0 ⟨69391⟩ 302951

def event302953 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69392⟩⟩) 1 ⟨28042⟩ 298566

def event302954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69392⟩⟩) (.sum [.predecessor 0 302952 .coefficient, .predecessor 1 302953 .coefficient])

def event302955 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69392⟩⟩) (.sum [.result 302951 .summary, .result 298566 .summary])

def exact302956RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26489⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56931⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨62891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302956RawTermsValid :
    exact302956RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302956 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69392⟩⟩) exact302956RawTerms .large 302954 (.finite 354089550391067611616654269349888) (some (302955))

def event302957 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69393⟩⟩) 0 ⟨69392⟩ 302956

def event302958 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69393⟩⟩) 1 ⟨30722⟩ 298132

def event302959 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69393⟩⟩) (.sum [.predecessor 0 302957 .coefficient, .predecessor 1 302958 .coefficient])

def event302960 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69393⟩⟩) (.sum [.result 302956 .summary, .result 298132 .summary])

def exact302961RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26489⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨29169⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56931⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨62891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302961RawTermsValid :
    exact302961RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302961 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69393⟩⟩) exact302961RawTerms .large 302959 (.finite 386281697261128003919260020637696) (some (302960))

def event302962 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69394⟩⟩) 0 ⟨69393⟩ 302961

def event302963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69394⟩⟩) 1 ⟨36382⟩ 297698

def event302964 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69394⟩⟩) (.sum [.predecessor 0 302962 .coefficient, .predecessor 1 302963 .coefficient])

def event302965 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69394⟩⟩) (.sum [.result 302961 .summary, .result 297698 .summary])

def exact302966RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26489⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨29169⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34833⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56931⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨62891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302966RawTermsValid :
    exact302966RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302966 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69394⟩⟩) exact302966RawTerms .large 302964 (.finite 418474237032079770976347551432704) (some (302965))

def event302967 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69395⟩⟩) 0 ⟨69394⟩ 302966

def event302968 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69395⟩⟩) 1 ⟨39062⟩ 297264

def event302969 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69395⟩⟩) (.sum [.predecessor 0 302967 .coefficient, .predecessor 1 302968 .coefficient])

def event302970 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69395⟩⟩) (.sum [.result 302966 .summary, .result 297264 .summary])

def exact302971RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26489⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨29169⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34833⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56931⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨62891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302971RawTermsValid :
    exact302971RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302971 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69395⟩⟩) exact302971RawTerms .large 302969 (.finite 450666973253477225410675971981312) (some (302970))

def event302972 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69396⟩⟩) 0 ⟨69395⟩ 302971

def event302973 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69396⟩⟩) 1 ⟨41742⟩ 296830

def event302974 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69396⟩⟩) (.sum [.predecessor 0 302972 .coefficient, .predecessor 1 302973 .coefficient])

def event302975 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69396⟩⟩) (.sum [.result 302971 .summary, .result 296830 .summary])

def exact302976RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26489⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨29169⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34833⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40189⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56931⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨62891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302976RawTermsValid :
    exact302976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302976 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69396⟩⟩) exact302976RawTerms .large 302974 (.finite 482860102375766054599486172037120) (some (302975))

def event302977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69397⟩⟩) 0 ⟨69396⟩ 302976

def event302978 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69397⟩⟩) 1 ⟨44422⟩ 296396

def event302979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69397⟩⟩) (.sum [.predecessor 0 302977 .coefficient, .predecessor 1 302978 .coefficient])

def event302980 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69397⟩⟩) (.sum [.result 302976 .summary, .result 296396 .summary])

def exact302981RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26489⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨29169⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34833⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40189⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42869⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56931⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨62891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302981RawTermsValid :
    exact302981RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302981 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69397⟩⟩) exact302981RawTerms .large 302979 (.finite 515053820849391945920019041353728) (some (302980))

def event302982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69398⟩⟩) 0 ⟨69397⟩ 302981

def event302983 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69398⟩⟩) 1 ⟨47102⟩ 295962

def event302984 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69398⟩⟩) (.sum [.predecessor 0 302982 .coefficient, .predecessor 1 302983 .coefficient])

def event302985 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69398⟩⟩) (.sum [.result 302981 .summary, .result 295962 .summary])

def exact302986RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26489⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨29169⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34833⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40189⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42869⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45553⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56931⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨62891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302986RawTermsValid :
    exact302986RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302986 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69398⟩⟩) exact302986RawTerms .large 302984 (.finite 547248128674354899372274579931136) (some (302985))

def event302987 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69399⟩⟩) 0 ⟨69398⟩ 302986

def event302988 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69399⟩⟩) 1 ⟨49782⟩ 295528

def event302989 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69399⟩⟩) (.sum [.predecessor 0 302987 .coefficient, .predecessor 1 302988 .coefficient])

def event302990 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69399⟩⟩) (.sum [.result 302986 .summary, .result 295528 .summary])

def exact302991RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26489⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨29169⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34833⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40189⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42869⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45553⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56931⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨62891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact302991RawTermsValid :
    exact302991RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event302991 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69399⟩⟩) exact302991RawTerms .large 302989 (.finite 579442632949763540201771008262144) (some (302990))

def event302992 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70936⟩⟩) 0 ⟨69399⟩ 302991

def event302993 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70936⟩⟩) 1 ⟨70934⟩ 295083

def event302994 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70936⟩⟩) (.product (.predecessor 0 302992 .coefficient) (.predecessor 1 302993 .coefficient) (⟨false, false, none, none, none⟩))

def event302995 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70936⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) [⟨.result 295083 .coefficient, false, none⟩])

def event302996 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70936⟩⟩) (.product (.result 302991 .summary) (.transfer 302995) (⟨false, false, none, none, none⟩))

def event302997 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 17⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩)

def event302998 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 29⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (-1)⟩)

def event302999 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70936⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080)

def event303000 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .relation 302999 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48233⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩)

def event303001 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 16⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩)

def event303002 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 28⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45553⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (-1)⟩)

def event303003 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70936⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45553⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080)

def event303004 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .relation 303003 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45553⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩)

def event303005 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 15⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩)

def event303006 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 27⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42869⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (-1)⟩)

def event303007 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70936⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42869⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080)

def event303008 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .relation 303007 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42869⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩)

def event303009 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 14⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩)

def event303010 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 26⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40189⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (-1)⟩)

def event303011 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70936⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40189⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080)

def event303012 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .relation 303011 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40189⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩)

def event303013 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 13⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩)

def event303014 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 25⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (-1)⟩)

def event303015 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70936⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080)

def event303016 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .relation 303015 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩)

def event303017 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 12⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩)

def event303018 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 24⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34833⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (-1)⟩)

def event303019 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70936⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34833⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080)

def event303020 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .relation 303019 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34833⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩)

def event303021 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 11⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩)

def event303022 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 22⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨29169⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (-1)⟩)

def event303023 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70936⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨29169⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080)

def event303024 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .relation 303023 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨29169⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩)

def event303025 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 10⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩)

def event303026 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 21⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26489⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (-1)⟩)

def event303027 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70936⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26489⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080)

def event303028 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .relation 303027 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26489⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩)

def event303029 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 9⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩)

def event303030 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 35⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (-1)⟩)

def event303031 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70936⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080)

def event303032 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .relation 303031 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩)

def event303033 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 8⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩)

def event303034 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 34⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨62891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (-1)⟩)

def event303035 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70936⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨62891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080)

def event303036 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .relation 303035 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨62891⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩)

def event303037 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 7⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩)

def event303038 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 33⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (-1)⟩)

def event303039 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70936⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080)

def event303040 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .relation 303039 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩)

def event303041 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 6⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩)

def event303042 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 32⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56931⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (-1)⟩)

def event303043 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70936⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56931⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080)

def event303044 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .relation 303043 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56931⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩)

def event303045 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 5⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩)

def event303046 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 31⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (-1)⟩)

def event303047 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70936⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53951⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080)

def event303048 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .relation 303047 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53951⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩)

def event303049 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 4⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩)

def event303050 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 30⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (-1)⟩)

def event303051 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70936⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080)

def event303052 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .relation 303051 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩)

def event303053 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 3⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩)

def event303054 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 23⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (-1)⟩)

def event303055 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70936⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080)

def event303056 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .relation 303055 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩)

def event303057 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 2⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩)

def event303058 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 20⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (-1)⟩)

def event303059 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70936⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080)

def event303060 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .relation 303059 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩)

def event303061 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 1⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩)

def event303062 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 19⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (-1)⟩)

def event303063 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70936⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080)

def event303064 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .relation 303063 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩)

def event303065 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 0⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩)

def event303066 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .operator (⟨302991, 18⟩, ⟨295083, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (-1)⟩)

def event303067 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70936⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080)

def event303068 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70936⟩⟩, .relation 303067 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩)

def exact303069RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨26489⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨29169⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34833⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40189⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42869⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45553⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48233⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨50971⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨53951⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨56931⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨62891⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩, (-1)⟩]

theorem exact303069RawTermsValid :
    exact303069RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event303069 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70936⟩⟩) exact303069RawTerms .large 302994 (.finite 6221717896068416040249469304417135687106560) (some (302996))

def event303070 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68270⟩⟩) 0 ⟨65911⟩ 14772

def event303071 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68270⟩⟩) (.authority (.relationPreimageSource ⟨95⟩))

def exact303072RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68270⟩⟩]⟩, (1)⟩]

theorem exact303072RawTermsValid :
    exact303072RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event303072 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68270⟩⟩) exact303072RawTerms (.finite 5647228698) 303071 .exactZero (none)

def event303073 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68272⟩⟩) 0 ⟨68270⟩ 303072

def event303074 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68272⟩⟩) 1 ⟨2370⟩ 4

def event303075 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68272⟩⟩) (.scale (.predecessor 0 303073 .coefficient) (.value (.predecessor 1 303074 .coefficient)))

def exact303076RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68270⟩⟩]⟩, (1)⟩]

theorem exact303076RawTermsValid :
    exact303076RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event303076 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68272⟩⟩) exact303076RawTerms (.finite 5647228698) 303075 .exactZero (none)

def event303077 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68273⟩⟩) 0 ⟨2380⟩ 295195

def event303078 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68273⟩⟩) 1 ⟨68272⟩ 303076

def event303079 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68273⟩⟩) (.product (.predecessor 0 303077 .coefficient) (.predecessor 1 303078 .coefficient) (⟨false, false, none, none, none⟩))

def event303080 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68273⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨68270⟩⟩]⟩) [⟨.result 303072 .coefficient, false, none⟩])

def event303081 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68273⟩⟩) (.product (.result 295195 .summary) (.transfer 303080) (⟨false, false, none, none, none⟩))

def event303082 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68273⟩⟩, .operator (⟨295195, 0⟩, ⟨303076, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68270⟩⟩]⟩, (1)⟩)

def event303083 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨68271⟩⟩)

def event303084 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event303085 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event303086 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event303087 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event303088 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 303087

def event303089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 303085

def event303090 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 303088 .coefficient) (.value (.predecessor 1 303089 .coefficient)))

def event303091 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event303092 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47594⟩⟩) 0 ⟨392⟩ 303091

def event303093 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47594⟩⟩) (.authority (.programFamilyFact))

def exact303094RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47594⟩⟩], []⟩, (1)⟩]

theorem exact303094RawTermsValid :
    exact303094RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event303094 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47594⟩⟩) exact303094RawTerms (.finite 60) 303093 .exactZero (none)

def event303095 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14931⟩⟩) 0 ⟨392⟩ 303091

def event303096 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14931⟩⟩) (.authority (.programFamilyFact))

def exact303097RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14931⟩⟩], []⟩, (1)⟩]

theorem exact303097RawTermsValid :
    exact303097RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event303097 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14931⟩⟩) exact303097RawTerms (.finite 60) 303096 .exactZero (none)

def event303098 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47595⟩⟩) 0 ⟨14931⟩ 303097

def event303099 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47595⟩⟩) 1 ⟨47594⟩ 303094

def event303100 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47595⟩⟩) (.product (.predecessor 0 303098 .coefficient) (.predecessor 1 303099 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event303101 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47595⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14931⟩⟩, ⟨.program ⟨257⟩, ⟨47594⟩⟩], []⟩) [⟨.result 303097 .coefficient, true, some 1⟩, ⟨.result 303094 .coefficient, true, some 1⟩])

def event303102 : Event := .survivorFold (1) 303101

def exact303103RawTerms : List Term := []

theorem exact303103RawTermsValid :
    exact303103RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event303103 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47595⟩⟩) exact303103RawTerms (.finite 3600) 303100 (.finite 3600) (some (303101))

def eventLeaf18928 : Array AnnotatedEvent := #[
  { event := event302848
    frameStart := 302801 },
  { event := event302849
    frameStart := 302801 },
  { event := event302850
    frameStart := 302801 },
  { event := event302851
    frameStart := 302801 },
  { event := event302852
    frameStart := 302801 },
  { event := event302853
    frameStart := 302801 },
  { event := event302854
    frameStart := 302801 },
  { event := event302855
    frameStart := 302801 },
  { event := event302856
    frameStart := 302801 },
  { event := event302857
    frameStart := 302801 },
  { event := event302858
    frameStart := 302801 },
  { event := event302859
    frameStart := 302801 },
  { event := event302860
    frameStart := 302801 },
  { event := event302861
    frameStart := 302801 },
  { event := event302862
    frameStart := 302801 },
  { event := event302863
    frameStart := 302801 }
]

def eventLeaf18929 : Array AnnotatedEvent := #[
  { event := event302864
    frameStart := 302801 },
  { event := event302865
    frameStart := 302801 },
  { event := event302866
    frameStart := 302801 },
  { event := event302867
    frameStart := 302801 },
  { event := event302868
    frameStart := 302801 },
  { event := event302869
    frameStart := 302801 },
  { event := event302870
    frameStart := 302801 },
  { event := event302871
    frameStart := 302801 },
  { event := event302872
    frameStart := 302801 },
  { event := event302873
    frameStart := 302801 },
  { event := event302874
    frameStart := 302801 },
  { event := event302875
    frameStart := 302801 },
  { event := event302876
    frameStart := 302801 },
  { event := event302877
    frameStart := 302801 },
  { event := event302878
    frameStart := 302801 },
  { event := event302879
    frameStart := 302801 }
]

def eventLeaf18930 : Array AnnotatedEvent := #[
  { event := event302880
    frameStart := 302801 },
  { event := event302881
    frameStart := 302801 },
  { event := event302882
    frameStart := 302801 },
  { event := event302883
    frameStart := 302801 },
  { event := event302884
    frameStart := 302801 },
  { event := event302885
    frameStart := 302801 },
  { event := event302886
    frameStart := 302801 },
  { event := event302887
    frameStart := 302801 },
  { event := event302888
    frameStart := 302801 },
  { event := event302889
    frameStart := 302801 },
  { event := event302890
    frameStart := 302801 },
  { event := event302891
    frameStart := 302801 },
  { event := event302892
    frameStart := 302801 },
  { event := event302893
    frameStart := 0 },
  { event := event302894
    frameStart := 0 },
  { event := event302895
    frameStart := 0 }
]

def eventLeaf18931 : Array AnnotatedEvent := #[
  { event := event302896
    frameStart := 0 },
  { event := event302897
    frameStart := 0 },
  { event := event302898
    frameStart := 0 },
  { event := event302899
    frameStart := 0 },
  { event := event302900
    frameStart := 0 },
  { event := event302901
    frameStart := 0 },
  { event := event302902
    frameStart := 0 },
  { event := event302903
    frameStart := 0 },
  { event := event302904
    frameStart := 0 },
  { event := event302905
    frameStart := 0 },
  { event := event302906
    frameStart := 0 },
  { event := event302907
    frameStart := 0 },
  { event := event302908
    frameStart := 0 },
  { event := event302909
    frameStart := 0 },
  { event := event302910
    frameStart := 0 },
  { event := event302911
    frameStart := 0 }
]

def eventLeaf18932 : Array AnnotatedEvent := #[
  { event := event302912
    frameStart := 0 },
  { event := event302913
    frameStart := 0 },
  { event := event302914
    frameStart := 0 },
  { event := event302915
    frameStart := 0 },
  { event := event302916
    frameStart := 0 },
  { event := event302917
    frameStart := 0 },
  { event := event302918
    frameStart := 0 },
  { event := event302919
    frameStart := 0 },
  { event := event302920
    frameStart := 0 },
  { event := event302921
    frameStart := 0 },
  { event := event302922
    frameStart := 0 },
  { event := event302923
    frameStart := 0 },
  { event := event302924
    frameStart := 0 },
  { event := event302925
    frameStart := 0 },
  { event := event302926
    frameStart := 0 },
  { event := event302927
    frameStart := 0 }
]

def eventLeaf18933 : Array AnnotatedEvent := #[
  { event := event302928
    frameStart := 0 },
  { event := event302929
    frameStart := 0 },
  { event := event302930
    frameStart := 0 },
  { event := event302931
    frameStart := 0 },
  { event := event302932
    frameStart := 0 },
  { event := event302933
    frameStart := 0 },
  { event := event302934
    frameStart := 0 },
  { event := event302935
    frameStart := 0 },
  { event := event302936
    frameStart := 0 },
  { event := event302937
    frameStart := 0 },
  { event := event302938
    frameStart := 0 },
  { event := event302939
    frameStart := 0 },
  { event := event302940
    frameStart := 0 },
  { event := event302941
    frameStart := 0 },
  { event := event302942
    frameStart := 0 },
  { event := event302943
    frameStart := 0 }
]

def eventLeaf18934 : Array AnnotatedEvent := #[
  { event := event302944
    frameStart := 0 },
  { event := event302945
    frameStart := 0 },
  { event := event302946
    frameStart := 0 },
  { event := event302947
    frameStart := 0 },
  { event := event302948
    frameStart := 0 },
  { event := event302949
    frameStart := 0 },
  { event := event302950
    frameStart := 0 },
  { event := event302951
    frameStart := 0 },
  { event := event302952
    frameStart := 0 },
  { event := event302953
    frameStart := 0 },
  { event := event302954
    frameStart := 0 },
  { event := event302955
    frameStart := 0 },
  { event := event302956
    frameStart := 0 },
  { event := event302957
    frameStart := 0 },
  { event := event302958
    frameStart := 0 },
  { event := event302959
    frameStart := 0 }
]

def eventLeaf18935 : Array AnnotatedEvent := #[
  { event := event302960
    frameStart := 0 },
  { event := event302961
    frameStart := 0 },
  { event := event302962
    frameStart := 0 },
  { event := event302963
    frameStart := 0 },
  { event := event302964
    frameStart := 0 },
  { event := event302965
    frameStart := 0 },
  { event := event302966
    frameStart := 0 },
  { event := event302967
    frameStart := 0 },
  { event := event302968
    frameStart := 0 },
  { event := event302969
    frameStart := 0 },
  { event := event302970
    frameStart := 0 },
  { event := event302971
    frameStart := 0 },
  { event := event302972
    frameStart := 0 },
  { event := event302973
    frameStart := 0 },
  { event := event302974
    frameStart := 0 },
  { event := event302975
    frameStart := 0 }
]

def eventLeaf18936 : Array AnnotatedEvent := #[
  { event := event302976
    frameStart := 0 },
  { event := event302977
    frameStart := 0 },
  { event := event302978
    frameStart := 0 },
  { event := event302979
    frameStart := 0 },
  { event := event302980
    frameStart := 0 },
  { event := event302981
    frameStart := 0 },
  { event := event302982
    frameStart := 0 },
  { event := event302983
    frameStart := 0 },
  { event := event302984
    frameStart := 0 },
  { event := event302985
    frameStart := 0 },
  { event := event302986
    frameStart := 0 },
  { event := event302987
    frameStart := 0 },
  { event := event302988
    frameStart := 0 },
  { event := event302989
    frameStart := 0 },
  { event := event302990
    frameStart := 0 },
  { event := event302991
    frameStart := 0 }
]

def eventLeaf18937 : Array AnnotatedEvent := #[
  { event := event302992
    frameStart := 0 },
  { event := event302993
    frameStart := 0 },
  { event := event302994
    frameStart := 0 },
  { event := event302995
    frameStart := 0 },
  { event := event302996
    frameStart := 0 },
  { event := event302997
    frameStart := 0 },
  { event := event302998
    frameStart := 0 },
  { event := event302999
    frameStart := 0 },
  { event := event303000
    frameStart := 0 },
  { event := event303001
    frameStart := 0 },
  { event := event303002
    frameStart := 0 },
  { event := event303003
    frameStart := 0 },
  { event := event303004
    frameStart := 0 },
  { event := event303005
    frameStart := 0 },
  { event := event303006
    frameStart := 0 },
  { event := event303007
    frameStart := 0 }
]

def eventLeaf18938 : Array AnnotatedEvent := #[
  { event := event303008
    frameStart := 0 },
  { event := event303009
    frameStart := 0 },
  { event := event303010
    frameStart := 0 },
  { event := event303011
    frameStart := 0 },
  { event := event303012
    frameStart := 0 },
  { event := event303013
    frameStart := 0 },
  { event := event303014
    frameStart := 0 },
  { event := event303015
    frameStart := 0 },
  { event := event303016
    frameStart := 0 },
  { event := event303017
    frameStart := 0 },
  { event := event303018
    frameStart := 0 },
  { event := event303019
    frameStart := 0 },
  { event := event303020
    frameStart := 0 },
  { event := event303021
    frameStart := 0 },
  { event := event303022
    frameStart := 0 },
  { event := event303023
    frameStart := 0 }
]

def eventLeaf18939 : Array AnnotatedEvent := #[
  { event := event303024
    frameStart := 0 },
  { event := event303025
    frameStart := 0 },
  { event := event303026
    frameStart := 0 },
  { event := event303027
    frameStart := 0 },
  { event := event303028
    frameStart := 0 },
  { event := event303029
    frameStart := 0 },
  { event := event303030
    frameStart := 0 },
  { event := event303031
    frameStart := 0 },
  { event := event303032
    frameStart := 0 },
  { event := event303033
    frameStart := 0 },
  { event := event303034
    frameStart := 0 },
  { event := event303035
    frameStart := 0 },
  { event := event303036
    frameStart := 0 },
  { event := event303037
    frameStart := 0 },
  { event := event303038
    frameStart := 0 },
  { event := event303039
    frameStart := 0 }
]

def eventLeaf18940 : Array AnnotatedEvent := #[
  { event := event303040
    frameStart := 0 },
  { event := event303041
    frameStart := 0 },
  { event := event303042
    frameStart := 0 },
  { event := event303043
    frameStart := 0 },
  { event := event303044
    frameStart := 0 },
  { event := event303045
    frameStart := 0 },
  { event := event303046
    frameStart := 0 },
  { event := event303047
    frameStart := 0 },
  { event := event303048
    frameStart := 0 },
  { event := event303049
    frameStart := 0 },
  { event := event303050
    frameStart := 0 },
  { event := event303051
    frameStart := 0 },
  { event := event303052
    frameStart := 0 },
  { event := event303053
    frameStart := 0 },
  { event := event303054
    frameStart := 0 },
  { event := event303055
    frameStart := 0 }
]

def eventLeaf18941 : Array AnnotatedEvent := #[
  { event := event303056
    frameStart := 0 },
  { event := event303057
    frameStart := 0 },
  { event := event303058
    frameStart := 0 },
  { event := event303059
    frameStart := 0 },
  { event := event303060
    frameStart := 0 },
  { event := event303061
    frameStart := 0 },
  { event := event303062
    frameStart := 0 },
  { event := event303063
    frameStart := 0 },
  { event := event303064
    frameStart := 0 },
  { event := event303065
    frameStart := 0 },
  { event := event303066
    frameStart := 0 },
  { event := event303067
    frameStart := 0 },
  { event := event303068
    frameStart := 0 },
  { event := event303069
    frameStart := 0 },
  { event := event303070
    frameStart := 0 },
  { event := event303071
    frameStart := 0 }
]

def eventLeaf18942 : Array AnnotatedEvent := #[
  { event := event303072
    frameStart := 0 },
  { event := event303073
    frameStart := 0 },
  { event := event303074
    frameStart := 0 },
  { event := event303075
    frameStart := 0 },
  { event := event303076
    frameStart := 0 },
  { event := event303077
    frameStart := 0 },
  { event := event303078
    frameStart := 0 },
  { event := event303079
    frameStart := 0 },
  { event := event303080
    frameStart := 0 },
  { event := event303081
    frameStart := 0 },
  { event := event303082
    frameStart := 0 },
  { event := event303083
    frameStart := 303083 },
  { event := event303084
    frameStart := 303083 },
  { event := event303085
    frameStart := 303083 },
  { event := event303086
    frameStart := 303083 },
  { event := event303087
    frameStart := 303083 }
]

def eventLeaf18943 : Array AnnotatedEvent := #[
  { event := event303088
    frameStart := 303083 },
  { event := event303089
    frameStart := 303083 },
  { event := event303090
    frameStart := 303083 },
  { event := event303091
    frameStart := 303083 },
  { event := event303092
    frameStart := 303083 },
  { event := event303093
    frameStart := 303083 },
  { event := event303094
    frameStart := 303083 },
  { event := event303095
    frameStart := 303083 },
  { event := event303096
    frameStart := 303083 },
  { event := event303097
    frameStart := 303083 },
  { event := event303098
    frameStart := 303083 },
  { event := event303099
    frameStart := 303083 },
  { event := event303100
    frameStart := 303083 },
  { event := event303101
    frameStart := 303083 },
  { event := event303102
    frameStart := 303083 },
  { event := event303103
    frameStart := 303083 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1183
