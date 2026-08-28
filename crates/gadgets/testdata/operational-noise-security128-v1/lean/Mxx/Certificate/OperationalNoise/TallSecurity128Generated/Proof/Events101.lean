import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events101

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact25856RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17263⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12251⟩⟩, ⟨.program ⟨257⟩, ⟨15266⟩⟩], [⟨.program ⟨257⟩, ⟨16797⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact25856RawTermsValid :
    exact25856RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25856 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17267⟩⟩) exact25856RawTerms .large 25855 .exactZero (none)

def event25857 : Event := .preFoldPolynomial 25856 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17263⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12251⟩⟩, ⟨.program ⟨257⟩, ⟨15266⟩⟩], [⟨.program ⟨257⟩, ⟨16797⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact25858RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17263⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12251⟩⟩, ⟨.program ⟨257⟩, ⟨15266⟩⟩], [⟨.program ⟨257⟩, ⟨16797⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event25858 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17267⟩⟩) 25857 exact25858RawTerms .large 25855 .exactZero (none)

def event25859 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15268⟩⟩) ⟨⟨58⟩, ⟨36⟩, ⟨135⟩⟩ ⟨25693, 25859⟩

def event25860 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16205⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16202⟩⟩]⟩) (1) 0 2 (.universal 25859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16202⟩⟩]⟩) (none) 25858)

def event25861 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16205⟩⟩, .relation 25860 2, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12251⟩⟩, ⟨.program ⟨257⟩, ⟨15266⟩⟩], [⟨.program ⟨257⟩, ⟨16797⟩⟩]⟩, (1)⟩)

def event25862 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16205⟩⟩, .relation 25860 1, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17263⟩⟩]⟩, (-1)⟩)

def event25863 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16205⟩⟩, .relation 25860 3, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event25864 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16205⟩⟩, .relation 25860 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩)

def exact25865RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17263⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12251⟩⟩, ⟨.program ⟨257⟩, ⟨15266⟩⟩], [⟨.program ⟨257⟩, ⟨16797⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact25865RawTermsValid :
    exact25865RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25865 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16205⟩⟩) exact25865RawTerms .large 25689 (.finite 202072841853861888) (some (25691))

def event25866 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17265⟩⟩) 0 ⟨16205⟩ 25865

def event25867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17265⟩⟩) 1 ⟨17264⟩ 25679

def event25868 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17265⟩⟩) (.sum [.predecessor 0 25866 .coefficient, .predecessor 1 25867 .coefficient])

def event25869 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17265⟩⟩, .operator (⟨25865, 2⟩, ⟨25679, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12251⟩⟩, ⟨.program ⟨257⟩, ⟨15266⟩⟩], [⟨.program ⟨257⟩, ⟨16797⟩⟩]⟩, (-1)⟩)

def event25870 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17265⟩⟩, .operator (⟨25865, 1⟩, ⟨25679, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17263⟩⟩]⟩, (1)⟩)

def event25871 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17265⟩⟩) (.sum [.result 25865 .summary, .result 25679 .summary])

def exact25872RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact25872RawTermsValid :
    exact25872RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25872 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17265⟩⟩) exact25872RawTerms .large 25868 (.finite 2997816280693142192128) (some (25871))

def event25873 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17519⟩⟩) 0 ⟨17265⟩ 25872

def event25874 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17519⟩⟩) 1 ⟨17517⟩ 25576

def event25875 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17519⟩⟩) (.product (.predecessor 0 25873 .coefficient) (.predecessor 1 25874 .coefficient) (⟨false, false, none, none, none⟩))

def event25876 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17519⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩) [⟨.result 25576 .coefficient, false, none⟩])

def event25877 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17519⟩⟩) (.product (.result 25872 .summary) (.transfer 25876) (⟨false, false, none, none, none⟩))

def event25878 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17519⟩⟩, .operator (⟨25872, 1⟩, ⟨25576, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩, (-1)⟩)

def event25879 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17519⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17517⟩⟩) ⟨16923⟩ 25573)

def event25880 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17519⟩⟩, .relation 25879 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16923⟩⟩]⟩, (-1)⟩)

def event25881 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17519⟩⟩, .operator (⟨25872, 0⟩, ⟨25576, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩, (1)⟩)

def exact25882RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16923⟩⟩]⟩, (-1)⟩]

theorem exact25882RawTermsValid :
    exact25882RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25882 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17519⟩⟩) exact25882RawTerms .large 25875 (.finite 32188807212483504816668771614720) (some (25877))

def event25883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16422⟩⟩) 0 ⟨15719⟩ 459

def event25884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16422⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact25885RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩, (1)⟩]

theorem exact25885RawTermsValid :
    exact25885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25885 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16422⟩⟩) exact25885RawTerms (.finite 5647228698) 25884 .exactZero (none)

def event25886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16424⟩⟩) 0 ⟨16422⟩ 25885

def event25887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16424⟩⟩) 1 ⟨2370⟩ 4

def event25888 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16424⟩⟩) (.scale (.predecessor 0 25886 .coefficient) (.value (.predecessor 1 25887 .coefficient)))

def exact25889RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩, (1)⟩]

theorem exact25889RawTermsValid :
    exact25889RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25889 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16424⟩⟩) exact25889RawTerms (.finite 5647228698) 25888 .exactZero (none)

def event25890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16425⟩⟩) 0 ⟨5443⟩ 17169

def event25891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16425⟩⟩) 1 ⟨16424⟩ 25889

def event25892 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16425⟩⟩) (.product (.predecessor 0 25890 .coefficient) (.predecessor 1 25891 .coefficient) (⟨false, false, none, none, none⟩))

def event25893 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16425⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩) [⟨.result 25885 .coefficient, false, none⟩])

def event25894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16425⟩⟩) (.product (.result 17169 .summary) (.transfer 25893) (⟨false, false, none, none, none⟩))

def event25895 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16425⟩⟩, .operator (⟨17169, 0⟩, ⟨25889, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩, (1)⟩)

def event25896 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨16423⟩⟩)

def event25897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event25898 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event25899 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨140⟩⟩) (.authority (.operator))

def event25900 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨140⟩⟩) (.finite 19)

def event25901 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event25902 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event25903 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event25904 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event25905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 25904

def event25906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 25902

def event25907 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 25905 .coefficient) (.value (.predecessor 1 25906 .coefficient)))

def event25908 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event25909 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 0 ⟨392⟩ 25908

def event25910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 1 ⟨140⟩ 25900

def event25911 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨393⟩⟩) (.sum [.predecessor 0 25909 .coefficient, .predecessor 1 25910 .coefficient])

def event25912 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨393⟩⟩) (.finite 655359)

def event25913 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 0 ⟨393⟩ 25912

def event25914 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 1 ⟨5426⟩ 25898

def event25915 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.identity (.predecessor 1 25914 .coefficient))

def event25916 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.finite 655360)

def event25917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15266⟩⟩) 0 ⟨5439⟩ 25916

def event25918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15266⟩⟩) (.authority (.programFamilyFact))

def exact25919RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15266⟩⟩], []⟩, (1)⟩]

theorem exact25919RawTermsValid :
    exact25919RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25919 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15266⟩⟩) exact25919RawTerms (.finite 2) 25918 .exactZero (none)

def event25920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12251⟩⟩) 0 ⟨5439⟩ 25916

def event25921 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12251⟩⟩) (.authority (.programFamilyFact))

def exact25922RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12251⟩⟩], []⟩, (1)⟩]

theorem exact25922RawTermsValid :
    exact25922RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25922 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12251⟩⟩) exact25922RawTerms (.finite 2) 25921 .exactZero (none)

def event25923 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15267⟩⟩) 0 ⟨12251⟩ 25922

def event25924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15267⟩⟩) 1 ⟨15266⟩ 25919

def event25925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15267⟩⟩) (.product (.predecessor 0 25923 .coefficient) (.predecessor 1 25924 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event25926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15267⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12251⟩⟩, ⟨.program ⟨257⟩, ⟨15266⟩⟩], []⟩) [⟨.result 25922 .coefficient, true, some 1⟩, ⟨.result 25919 .coefficient, true, some 1⟩])

def event25927 : Event := .survivorFold (1) 25926

def exact25928RawTerms : List Term := []

theorem exact25928RawTermsValid :
    exact25928RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25928 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15267⟩⟩) exact25928RawTerms (.finite 4) 25925 (.finite 4) (some (25926))

def event25929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15268⟩⟩) 0 ⟨15267⟩ 25928

def event25930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15268⟩⟩) (.identity (.predecessor 0 25929 .coefficient))

def event25931 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15268⟩⟩) (.finite 4)

def event25932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15718⟩⟩) 0 ⟨15268⟩ 25931

def event25933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15718⟩⟩) (.authority (.programFamilyFact))

def exact25934RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], []⟩, (1)⟩]

theorem exact25934RawTermsValid :
    exact25934RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25934 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15718⟩⟩) exact25934RawTerms (.finite 2) 25933 .exactZero (none)

def event25935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15719⟩⟩) 0 ⟨15718⟩ 25934

def event25936 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15719⟩⟩) (.identity (.predecessor 0 25935 .coefficient))

def event25937 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15719⟩⟩) (.finite 2)

def event25938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16422⟩⟩) 0 ⟨15719⟩ 25937

def event25939 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16422⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact25940RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩, (1)⟩]

theorem exact25940RawTermsValid :
    exact25940RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25940 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16422⟩⟩) exact25940RawTerms (.finite 5647228698) 25939 .exactZero (none)

def event25941 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact25942RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact25942RawTermsValid :
    exact25942RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25942 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact25942RawTerms .large 25941 .exactZero (none)

def event25943 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16423⟩⟩) 0 ⟨35⟩ 25942

def event25944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16423⟩⟩) 1 ⟨16422⟩ 25940

def event25945 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16423⟩⟩) (.product (.predecessor 0 25943 .coefficient) (.predecessor 1 25944 .coefficient) (⟨false, false, none, none, none⟩))

def event25946 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16423⟩⟩, .operator (⟨25942, 0⟩, ⟨25940, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩, (1)⟩)

def exact25947RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩, (1)⟩]

theorem exact25947RawTermsValid :
    exact25947RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25947 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16423⟩⟩) exact25947RawTerms .large 25945 .exactZero (none)

def event25948 : Event := .preFoldPolynomial 25947 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩, (1)⟩] .exactZero none

def exact25949RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩, (1)⟩]

def event25949 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨16423⟩⟩) 25948 exact25949RawTerms .large 25945 .exactZero (none)

def event25950 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨17521⟩⟩)

def event25951 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event25952 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event25953 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨140⟩⟩) (.authority (.operator))

def event25954 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨140⟩⟩) (.finite 19)

def event25955 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event25956 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event25957 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event25958 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event25959 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 25958

def event25960 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 25956

def event25961 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 25959 .coefficient) (.value (.predecessor 1 25960 .coefficient)))

def event25962 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event25963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 0 ⟨392⟩ 25962

def event25964 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 1 ⟨140⟩ 25954

def event25965 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨393⟩⟩) (.sum [.predecessor 0 25963 .coefficient, .predecessor 1 25964 .coefficient])

def event25966 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨393⟩⟩) (.finite 655359)

def event25967 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 0 ⟨393⟩ 25966

def event25968 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 1 ⟨5426⟩ 25952

def event25969 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.identity (.predecessor 1 25968 .coefficient))

def event25970 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.finite 655360)

def event25971 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15266⟩⟩) 0 ⟨5439⟩ 25970

def event25972 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15266⟩⟩) (.authority (.programFamilyFact))

def exact25973RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15266⟩⟩], []⟩, (1)⟩]

theorem exact25973RawTermsValid :
    exact25973RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25973 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15266⟩⟩) exact25973RawTerms (.finite 2) 25972 .exactZero (none)

def event25974 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12251⟩⟩) 0 ⟨5439⟩ 25970

def event25975 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12251⟩⟩) (.authority (.programFamilyFact))

def exact25976RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12251⟩⟩], []⟩, (1)⟩]

theorem exact25976RawTermsValid :
    exact25976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25976 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12251⟩⟩) exact25976RawTerms (.finite 2) 25975 .exactZero (none)

def event25977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15267⟩⟩) 0 ⟨12251⟩ 25976

def event25978 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15267⟩⟩) 1 ⟨15266⟩ 25973

def event25979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15267⟩⟩) (.product (.predecessor 0 25977 .coefficient) (.predecessor 1 25978 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event25980 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15267⟩⟩, .operator (⟨25976, 0⟩, ⟨25973, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12251⟩⟩, ⟨.program ⟨257⟩, ⟨15266⟩⟩], []⟩, (1)⟩)

def exact25981RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12251⟩⟩, ⟨.program ⟨257⟩, ⟨15266⟩⟩], []⟩, (1)⟩]

theorem exact25981RawTermsValid :
    exact25981RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25981 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15267⟩⟩) exact25981RawTerms (.finite 4) 25979 .exactZero (none)

def event25982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15268⟩⟩) 0 ⟨15267⟩ 25981

def event25983 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15268⟩⟩) (.identity (.predecessor 0 25982 .coefficient))

def event25984 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15268⟩⟩) (.finite 4)

def event25985 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15718⟩⟩) 0 ⟨15268⟩ 25984

def event25986 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15718⟩⟩) (.authority (.programFamilyFact))

def exact25987RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], []⟩, (1)⟩]

theorem exact25987RawTermsValid :
    exact25987RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25987 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15718⟩⟩) exact25987RawTerms (.finite 2) 25986 .exactZero (none)

def event25988 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15719⟩⟩) 0 ⟨15718⟩ 25987

def event25989 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15719⟩⟩) (.identity (.predecessor 0 25988 .coefficient))

def event25990 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15719⟩⟩) (.finite 2)

def event25991 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16921⟩⟩) 0 ⟨15719⟩ 25990

def event25992 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16921⟩⟩) (.authority (.programFamilyFact))

def event25993 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨16921⟩⟩) (.finite 3720)

def event25994 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event25995 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16923⟩⟩) 0 ⟨7177⟩ 25994

def event25996 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16923⟩⟩) 1 ⟨16921⟩ 25993

def event25997 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16923⟩⟩) (.authority (.operator))

def exact25998RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16923⟩⟩]⟩, (1)⟩]

theorem exact25998RawTermsValid :
    exact25998RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event25998 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16923⟩⟩) exact25998RawTerms .large 25997 .exactZero (none)

def event25999 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17517⟩⟩) 0 ⟨16923⟩ 25998

def event26000 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17517⟩⟩) (.authority (.operator))

def exact26001RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩, (1)⟩]

theorem exact26001RawTermsValid :
    exact26001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26001 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17517⟩⟩) exact26001RawTerms (.finite 8192) 26000 .exactZero (none)

def event26002 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event26003 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event26004 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17170⟩⟩) 0 ⟨15719⟩ 25990

def event26005 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17170⟩⟩) 1 ⟨136⟩ 26003

def event26006 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17170⟩⟩) (.sum [.predecessor 0 26004 .coefficient, .predecessor 1 26005 .coefficient])

def event26007 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17170⟩⟩) (.finite 2)

def event26008 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17171⟩⟩) 0 ⟨17170⟩ 26007

def event26009 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17171⟩⟩) (.identity (.predecessor 0 26008 .coefficient))

def exact26010RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], []⟩, (1)⟩]

theorem exact26010RawTermsValid :
    exact26010RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26010 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17171⟩⟩) exact26010RawTerms (.finite 2) 26009 .exactZero (none)

def event26011 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact26012RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact26012RawTermsValid :
    exact26012RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26012 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact26012RawTerms .large 26011 .exactZero (none)

def event26013 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17172⟩⟩) 0 ⟨6908⟩ 26012

def event26014 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17172⟩⟩) 1 ⟨17171⟩ 26010

def event26015 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17172⟩⟩) (.product (.predecessor 0 26013 .coefficient) (.predecessor 1 26014 .coefficient) (⟨false, false, none, none, none⟩))

def event26016 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17172⟩⟩, .operator (⟨26012, 0⟩, ⟨26010, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact26017RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact26017RawTermsValid :
    exact26017RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26017 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17172⟩⟩) exact26017RawTerms .large 26015 .exactZero (none)

def event26018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 25994

def event26019 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact26020RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact26020RawTermsValid :
    exact26020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26020 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact26020RawTerms .large 26019 .exactZero (none)

def event26021 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17173⟩⟩) 0 ⟨7179⟩ 26020

def event26022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17173⟩⟩) 1 ⟨17172⟩ 26017

def event26023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17173⟩⟩) (.sum [.predecessor 0 26021 .coefficient, .predecessor 1 26022 .coefficient])

def exact26024RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26024RawTermsValid :
    exact26024RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26024 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17173⟩⟩) exact26024RawTerms .large 26023 .exactZero (none)

def event26025 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17518⟩⟩) 0 ⟨17173⟩ 26024

def event26026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17518⟩⟩) 1 ⟨17517⟩ 26001

def event26027 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17518⟩⟩) (.product (.predecessor 0 26025 .coefficient) (.predecessor 1 26026 .coefficient) (⟨false, false, none, none, none⟩))

def event26028 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17518⟩⟩, .operator (⟨26024, 1⟩, ⟨26001, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩, (-1)⟩)

def event26029 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17518⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17517⟩⟩) ⟨16923⟩ 25998)

def event26030 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17518⟩⟩, .relation 26029 0, ⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16923⟩⟩]⟩, (-1)⟩)

def event26031 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17518⟩⟩, .operator (⟨26024, 0⟩, ⟨26001, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩, (1)⟩)

def exact26032RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16923⟩⟩]⟩, (-1)⟩]

theorem exact26032RawTermsValid :
    exact26032RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26032 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17518⟩⟩) exact26032RawTerms .large 26027 .exactZero (none)

def event26033 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15895⟩⟩) 0 ⟨15719⟩ 25990

def event26034 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15895⟩⟩) (.authority (.programFamilyFact))

def exact26035RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15895⟩⟩], []⟩, (1)⟩]

theorem exact26035RawTermsValid :
    exact26035RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26035 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15895⟩⟩) exact26035RawTerms (.finite 43) 26034 .exactZero (none)

def event26036 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15896⟩⟩) 0 ⟨6908⟩ 26012

def event26037 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15896⟩⟩) 1 ⟨15895⟩ 26035

def event26038 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15896⟩⟩) (.product (.predecessor 0 26036 .coefficient) (.predecessor 1 26037 .coefficient) (⟨false, true, none, none, some 1⟩))

def event26039 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15896⟩⟩, .operator (⟨26012, 0⟩, ⟨26035, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact26040RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact26040RawTermsValid :
    exact26040RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26040 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15896⟩⟩) exact26040RawTerms .large 26038 .exactZero (none)

def event26041 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 25994

def event26042 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact26043RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact26043RawTermsValid :
    exact26043RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26043 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact26043RawTerms .large 26042 .exactZero (none)

def event26044 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15897⟩⟩) 0 ⟨7198⟩ 26043

def event26045 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15897⟩⟩) 1 ⟨15896⟩ 26040

def event26046 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15897⟩⟩) (.sum [.predecessor 0 26044 .coefficient, .predecessor 1 26045 .coefficient])

def exact26047RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26047RawTermsValid :
    exact26047RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26047 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15897⟩⟩) exact26047RawTerms .large 26046 .exactZero (none)

def event26048 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17521⟩⟩) 0 ⟨15897⟩ 26047

def event26049 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17521⟩⟩) 1 ⟨17518⟩ 26032

def event26050 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17521⟩⟩) (.sum [.predecessor 0 26048 .coefficient, .predecessor 1 26049 .coefficient])

def exact26051RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16923⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26051RawTermsValid :
    exact26051RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26051 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17521⟩⟩) exact26051RawTerms .large 26050 .exactZero (none)

def event26052 : Event := .preFoldPolynomial 26051 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16923⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact26053RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16923⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event26053 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17521⟩⟩) 26052 exact26053RawTerms .large 26050 .exactZero (none)

def event26054 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15719⟩⟩) ⟨⟨77⟩, ⟨57⟩, ⟨135⟩⟩ ⟨25896, 26054⟩

def event26055 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16425⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩) (1) 0 2 (.universal 26054 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩) (none) 26053)

def event26056 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16425⟩⟩, .relation 26055 2, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16923⟩⟩]⟩, (1)⟩)

def event26057 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16425⟩⟩, .relation 26055 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩, (-1)⟩)

def event26058 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16425⟩⟩, .relation 26055 3, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event26059 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16425⟩⟩, .relation 26055 1, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩)

def exact26060RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16923⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26060RawTermsValid :
    exact26060RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26060 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16425⟩⟩) exact26060RawTerms .large 25892 (.finite 202072841853861888) (some (25894))

def event26061 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17520⟩⟩) 0 ⟨16425⟩ 26060

def event26062 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17520⟩⟩) 1 ⟨17519⟩ 25882

def event26063 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17520⟩⟩) (.sum [.predecessor 0 26061 .coefficient, .predecessor 1 26062 .coefficient])

def event26064 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17520⟩⟩, .operator (⟨26060, 2⟩, ⟨25882, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16923⟩⟩]⟩, (-1)⟩)

def event26065 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17520⟩⟩, .operator (⟨26060, 0⟩, ⟨25882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩, (1)⟩)

def event26066 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17520⟩⟩) (.sum [.result 26060 .summary, .result 25882 .summary])

def exact26067RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26067RawTermsValid :
    exact26067RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26067 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17520⟩⟩) exact26067RawTerms .large 26063 (.finite 32188807212483706889510625476608) (some (26066))

def event26068 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20386⟩⟩) 0 ⟨17520⟩ 26067

def event26069 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20386⟩⟩) 1 ⟨20385⟩ 25566

def event26070 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20386⟩⟩) (.sum [.predecessor 0 26068 .coefficient, .predecessor 1 26069 .coefficient])

def event26071 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20386⟩⟩) (.sum [.result 26067 .summary, .result 25566 .summary])

def exact26072RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26072RawTermsValid :
    exact26072RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26072 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20386⟩⟩) exact26072RawTerms .large 26070 (.finite 64377712650190257467641695830016) (some (26071))

def event26073 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23606⟩⟩) 0 ⟨20386⟩ 26072

def event26074 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23606⟩⟩) 1 ⟨23605⟩ 25065

def event26075 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23606⟩⟩) (.sum [.predecessor 0 26073 .coefficient, .predecessor 1 26074 .coefficient])

def event26076 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23606⟩⟩) (.sum [.result 26072 .summary, .result 25065 .summary])

def exact26077RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26077RawTermsValid :
    exact26077RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26077 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23606⟩⟩) exact26077RawTerms .large 26075 (.finite 96566716313119651734393211060224) (some (26076))

def event26078 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33626⟩⟩) 0 ⟨23606⟩ 26077

def event26079 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33626⟩⟩) 1 ⟨33625⟩ 24564

def event26080 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33626⟩⟩) (.sum [.predecessor 0 26078 .coefficient, .predecessor 1 26079 .coefficient])

def event26081 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33626⟩⟩) (.sum [.result 26077 .summary, .result 24564 .summary])

def exact26082RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26082RawTermsValid :
    exact26082RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26082 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33626⟩⟩) exact26082RawTerms .large 26080 (.finite 128755916426494733378385616044032) (some (26081))

def event26083 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52686⟩⟩) 0 ⟨33626⟩ 26082

def event26084 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52686⟩⟩) 1 ⟨52685⟩ 24063

def event26085 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52686⟩⟩) (.sum [.predecessor 0 26083 .coefficient, .predecessor 1 26084 .coefficient])

def event26086 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52686⟩⟩) (.sum [.result 26082 .summary, .result 24063 .summary])

def exact26087RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26087RawTermsValid :
    exact26087RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26087 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52686⟩⟩) exact26087RawTerms .large 26085 (.finite 160945509440761189776859800535040) (some (26086))

def event26088 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55666⟩⟩) 0 ⟨52686⟩ 26087

def event26089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55666⟩⟩) 1 ⟨55665⟩ 23562

def event26090 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55666⟩⟩) (.sum [.predecessor 0 26088 .coefficient, .predecessor 1 26089 .coefficient])

def event26091 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55666⟩⟩) (.sum [.result 26087 .summary, .result 23562 .summary])

def exact26092RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26092RawTermsValid :
    exact26092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26092 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55666⟩⟩) exact26092RawTerms .large 26090 (.finite 193135298905473333552574874779648) (some (26091))

def event26093 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58646⟩⟩) 0 ⟨55666⟩ 26092

def event26094 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58646⟩⟩) 1 ⟨58645⟩ 23061

def event26095 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58646⟩⟩) (.sum [.predecessor 0 26093 .coefficient, .predecessor 1 26094 .coefficient])

def event26096 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58646⟩⟩) (.sum [.result 26092 .summary, .result 23061 .summary])

def exact26097RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26097RawTermsValid :
    exact26097RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26097 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58646⟩⟩) exact26097RawTerms .large 26095 (.finite 225325481271076852082771728531456) (some (26096))

def event26098 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61626⟩⟩) 0 ⟨58646⟩ 26097

def event26099 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61626⟩⟩) 1 ⟨61625⟩ 22560

def event26100 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61626⟩⟩) (.sum [.predecessor 0 26098 .coefficient, .predecessor 1 26099 .coefficient])

def event26101 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61626⟩⟩) (.sum [.result 26097 .summary, .result 22560 .summary])

def exact26102RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26102RawTermsValid :
    exact26102RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26102 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61626⟩⟩) exact26102RawTerms .large 26100 (.finite 257515860087126057990209472036864) (some (26101))

def event26103 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64606⟩⟩) 0 ⟨61626⟩ 26102

def event26104 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64606⟩⟩) 1 ⟨64605⟩ 22059

def event26105 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64606⟩⟩) (.sum [.predecessor 0 26103 .coefficient, .predecessor 1 26104 .coefficient])

def event26106 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64606⟩⟩) (.sum [.result 26102 .summary, .result 22059 .summary])

def exact26107RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨53975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨56955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨59935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact26107RawTermsValid :
    exact26107RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26107 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64606⟩⟩) exact26107RawTerms .large 26105 (.finite 289706631804066638652128995049472) (some (26106))

def event26108 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69495⟩⟩) 0 ⟨64606⟩ 26107

def event26109 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69495⟩⟩) 1 ⟨69494⟩ 21558

def event26110 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69495⟩⟩) (.sum [.predecessor 0 26108 .coefficient, .predecessor 1 26109 .coefficient])

def event26111 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69495⟩⟩) (.sum [.result 26107 .summary, .result 21558 .summary])

def eventLeaf1616 : Array AnnotatedEvent := #[
  { event := event25856
    frameStart := 25741 },
  { event := event25857
    frameStart := 25741 },
  { event := event25858
    frameStart := 25741 },
  { event := event25859
    frameStart := 0 },
  { event := event25860
    frameStart := 0 },
  { event := event25861
    frameStart := 0 },
  { event := event25862
    frameStart := 0 },
  { event := event25863
    frameStart := 0 },
  { event := event25864
    frameStart := 0 },
  { event := event25865
    frameStart := 0 },
  { event := event25866
    frameStart := 0 },
  { event := event25867
    frameStart := 0 },
  { event := event25868
    frameStart := 0 },
  { event := event25869
    frameStart := 0 },
  { event := event25870
    frameStart := 0 },
  { event := event25871
    frameStart := 0 }
]

def eventLeaf1617 : Array AnnotatedEvent := #[
  { event := event25872
    frameStart := 0 },
  { event := event25873
    frameStart := 0 },
  { event := event25874
    frameStart := 0 },
  { event := event25875
    frameStart := 0 },
  { event := event25876
    frameStart := 0 },
  { event := event25877
    frameStart := 0 },
  { event := event25878
    frameStart := 0 },
  { event := event25879
    frameStart := 0 },
  { event := event25880
    frameStart := 0 },
  { event := event25881
    frameStart := 0 },
  { event := event25882
    frameStart := 0 },
  { event := event25883
    frameStart := 0 },
  { event := event25884
    frameStart := 0 },
  { event := event25885
    frameStart := 0 },
  { event := event25886
    frameStart := 0 },
  { event := event25887
    frameStart := 0 }
]

def eventLeaf1618 : Array AnnotatedEvent := #[
  { event := event25888
    frameStart := 0 },
  { event := event25889
    frameStart := 0 },
  { event := event25890
    frameStart := 0 },
  { event := event25891
    frameStart := 0 },
  { event := event25892
    frameStart := 0 },
  { event := event25893
    frameStart := 0 },
  { event := event25894
    frameStart := 0 },
  { event := event25895
    frameStart := 0 },
  { event := event25896
    frameStart := 25896 },
  { event := event25897
    frameStart := 25896 },
  { event := event25898
    frameStart := 25896 },
  { event := event25899
    frameStart := 25896 },
  { event := event25900
    frameStart := 25896 },
  { event := event25901
    frameStart := 25896 },
  { event := event25902
    frameStart := 25896 },
  { event := event25903
    frameStart := 25896 }
]

def eventLeaf1619 : Array AnnotatedEvent := #[
  { event := event25904
    frameStart := 25896 },
  { event := event25905
    frameStart := 25896 },
  { event := event25906
    frameStart := 25896 },
  { event := event25907
    frameStart := 25896 },
  { event := event25908
    frameStart := 25896 },
  { event := event25909
    frameStart := 25896 },
  { event := event25910
    frameStart := 25896 },
  { event := event25911
    frameStart := 25896 },
  { event := event25912
    frameStart := 25896 },
  { event := event25913
    frameStart := 25896 },
  { event := event25914
    frameStart := 25896 },
  { event := event25915
    frameStart := 25896 },
  { event := event25916
    frameStart := 25896 },
  { event := event25917
    frameStart := 25896 },
  { event := event25918
    frameStart := 25896 },
  { event := event25919
    frameStart := 25896 }
]

def eventLeaf1620 : Array AnnotatedEvent := #[
  { event := event25920
    frameStart := 25896 },
  { event := event25921
    frameStart := 25896 },
  { event := event25922
    frameStart := 25896 },
  { event := event25923
    frameStart := 25896 },
  { event := event25924
    frameStart := 25896 },
  { event := event25925
    frameStart := 25896 },
  { event := event25926
    frameStart := 25896 },
  { event := event25927
    frameStart := 25896 },
  { event := event25928
    frameStart := 25896 },
  { event := event25929
    frameStart := 25896 },
  { event := event25930
    frameStart := 25896 },
  { event := event25931
    frameStart := 25896 },
  { event := event25932
    frameStart := 25896 },
  { event := event25933
    frameStart := 25896 },
  { event := event25934
    frameStart := 25896 },
  { event := event25935
    frameStart := 25896 }
]

def eventLeaf1621 : Array AnnotatedEvent := #[
  { event := event25936
    frameStart := 25896 },
  { event := event25937
    frameStart := 25896 },
  { event := event25938
    frameStart := 25896 },
  { event := event25939
    frameStart := 25896 },
  { event := event25940
    frameStart := 25896 },
  { event := event25941
    frameStart := 25896 },
  { event := event25942
    frameStart := 25896 },
  { event := event25943
    frameStart := 25896 },
  { event := event25944
    frameStart := 25896 },
  { event := event25945
    frameStart := 25896 },
  { event := event25946
    frameStart := 25896 },
  { event := event25947
    frameStart := 25896 },
  { event := event25948
    frameStart := 25896 },
  { event := event25949
    frameStart := 25896 },
  { event := event25950
    frameStart := 25950 },
  { event := event25951
    frameStart := 25950 }
]

def eventLeaf1622 : Array AnnotatedEvent := #[
  { event := event25952
    frameStart := 25950 },
  { event := event25953
    frameStart := 25950 },
  { event := event25954
    frameStart := 25950 },
  { event := event25955
    frameStart := 25950 },
  { event := event25956
    frameStart := 25950 },
  { event := event25957
    frameStart := 25950 },
  { event := event25958
    frameStart := 25950 },
  { event := event25959
    frameStart := 25950 },
  { event := event25960
    frameStart := 25950 },
  { event := event25961
    frameStart := 25950 },
  { event := event25962
    frameStart := 25950 },
  { event := event25963
    frameStart := 25950 },
  { event := event25964
    frameStart := 25950 },
  { event := event25965
    frameStart := 25950 },
  { event := event25966
    frameStart := 25950 },
  { event := event25967
    frameStart := 25950 }
]

def eventLeaf1623 : Array AnnotatedEvent := #[
  { event := event25968
    frameStart := 25950 },
  { event := event25969
    frameStart := 25950 },
  { event := event25970
    frameStart := 25950 },
  { event := event25971
    frameStart := 25950 },
  { event := event25972
    frameStart := 25950 },
  { event := event25973
    frameStart := 25950 },
  { event := event25974
    frameStart := 25950 },
  { event := event25975
    frameStart := 25950 },
  { event := event25976
    frameStart := 25950 },
  { event := event25977
    frameStart := 25950 },
  { event := event25978
    frameStart := 25950 },
  { event := event25979
    frameStart := 25950 },
  { event := event25980
    frameStart := 25950 },
  { event := event25981
    frameStart := 25950 },
  { event := event25982
    frameStart := 25950 },
  { event := event25983
    frameStart := 25950 }
]

def eventLeaf1624 : Array AnnotatedEvent := #[
  { event := event25984
    frameStart := 25950 },
  { event := event25985
    frameStart := 25950 },
  { event := event25986
    frameStart := 25950 },
  { event := event25987
    frameStart := 25950 },
  { event := event25988
    frameStart := 25950 },
  { event := event25989
    frameStart := 25950 },
  { event := event25990
    frameStart := 25950 },
  { event := event25991
    frameStart := 25950 },
  { event := event25992
    frameStart := 25950 },
  { event := event25993
    frameStart := 25950 },
  { event := event25994
    frameStart := 25950 },
  { event := event25995
    frameStart := 25950 },
  { event := event25996
    frameStart := 25950 },
  { event := event25997
    frameStart := 25950 },
  { event := event25998
    frameStart := 25950 },
  { event := event25999
    frameStart := 25950 }
]

def eventLeaf1625 : Array AnnotatedEvent := #[
  { event := event26000
    frameStart := 25950 },
  { event := event26001
    frameStart := 25950 },
  { event := event26002
    frameStart := 25950 },
  { event := event26003
    frameStart := 25950 },
  { event := event26004
    frameStart := 25950 },
  { event := event26005
    frameStart := 25950 },
  { event := event26006
    frameStart := 25950 },
  { event := event26007
    frameStart := 25950 },
  { event := event26008
    frameStart := 25950 },
  { event := event26009
    frameStart := 25950 },
  { event := event26010
    frameStart := 25950 },
  { event := event26011
    frameStart := 25950 },
  { event := event26012
    frameStart := 25950 },
  { event := event26013
    frameStart := 25950 },
  { event := event26014
    frameStart := 25950 },
  { event := event26015
    frameStart := 25950 }
]

def eventLeaf1626 : Array AnnotatedEvent := #[
  { event := event26016
    frameStart := 25950 },
  { event := event26017
    frameStart := 25950 },
  { event := event26018
    frameStart := 25950 },
  { event := event26019
    frameStart := 25950 },
  { event := event26020
    frameStart := 25950 },
  { event := event26021
    frameStart := 25950 },
  { event := event26022
    frameStart := 25950 },
  { event := event26023
    frameStart := 25950 },
  { event := event26024
    frameStart := 25950 },
  { event := event26025
    frameStart := 25950 },
  { event := event26026
    frameStart := 25950 },
  { event := event26027
    frameStart := 25950 },
  { event := event26028
    frameStart := 25950 },
  { event := event26029
    frameStart := 25950 },
  { event := event26030
    frameStart := 25950 },
  { event := event26031
    frameStart := 25950 }
]

def eventLeaf1627 : Array AnnotatedEvent := #[
  { event := event26032
    frameStart := 25950 },
  { event := event26033
    frameStart := 25950 },
  { event := event26034
    frameStart := 25950 },
  { event := event26035
    frameStart := 25950 },
  { event := event26036
    frameStart := 25950 },
  { event := event26037
    frameStart := 25950 },
  { event := event26038
    frameStart := 25950 },
  { event := event26039
    frameStart := 25950 },
  { event := event26040
    frameStart := 25950 },
  { event := event26041
    frameStart := 25950 },
  { event := event26042
    frameStart := 25950 },
  { event := event26043
    frameStart := 25950 },
  { event := event26044
    frameStart := 25950 },
  { event := event26045
    frameStart := 25950 },
  { event := event26046
    frameStart := 25950 },
  { event := event26047
    frameStart := 25950 }
]

def eventLeaf1628 : Array AnnotatedEvent := #[
  { event := event26048
    frameStart := 25950 },
  { event := event26049
    frameStart := 25950 },
  { event := event26050
    frameStart := 25950 },
  { event := event26051
    frameStart := 25950 },
  { event := event26052
    frameStart := 25950 },
  { event := event26053
    frameStart := 25950 },
  { event := event26054
    frameStart := 0 },
  { event := event26055
    frameStart := 0 },
  { event := event26056
    frameStart := 0 },
  { event := event26057
    frameStart := 0 },
  { event := event26058
    frameStart := 0 },
  { event := event26059
    frameStart := 0 },
  { event := event26060
    frameStart := 0 },
  { event := event26061
    frameStart := 0 },
  { event := event26062
    frameStart := 0 },
  { event := event26063
    frameStart := 0 }
]

def eventLeaf1629 : Array AnnotatedEvent := #[
  { event := event26064
    frameStart := 0 },
  { event := event26065
    frameStart := 0 },
  { event := event26066
    frameStart := 0 },
  { event := event26067
    frameStart := 0 },
  { event := event26068
    frameStart := 0 },
  { event := event26069
    frameStart := 0 },
  { event := event26070
    frameStart := 0 },
  { event := event26071
    frameStart := 0 },
  { event := event26072
    frameStart := 0 },
  { event := event26073
    frameStart := 0 },
  { event := event26074
    frameStart := 0 },
  { event := event26075
    frameStart := 0 },
  { event := event26076
    frameStart := 0 },
  { event := event26077
    frameStart := 0 },
  { event := event26078
    frameStart := 0 },
  { event := event26079
    frameStart := 0 }
]

def eventLeaf1630 : Array AnnotatedEvent := #[
  { event := event26080
    frameStart := 0 },
  { event := event26081
    frameStart := 0 },
  { event := event26082
    frameStart := 0 },
  { event := event26083
    frameStart := 0 },
  { event := event26084
    frameStart := 0 },
  { event := event26085
    frameStart := 0 },
  { event := event26086
    frameStart := 0 },
  { event := event26087
    frameStart := 0 },
  { event := event26088
    frameStart := 0 },
  { event := event26089
    frameStart := 0 },
  { event := event26090
    frameStart := 0 },
  { event := event26091
    frameStart := 0 },
  { event := event26092
    frameStart := 0 },
  { event := event26093
    frameStart := 0 },
  { event := event26094
    frameStart := 0 },
  { event := event26095
    frameStart := 0 }
]

def eventLeaf1631 : Array AnnotatedEvent := #[
  { event := event26096
    frameStart := 0 },
  { event := event26097
    frameStart := 0 },
  { event := event26098
    frameStart := 0 },
  { event := event26099
    frameStart := 0 },
  { event := event26100
    frameStart := 0 },
  { event := event26101
    frameStart := 0 },
  { event := event26102
    frameStart := 0 },
  { event := event26103
    frameStart := 0 },
  { event := event26104
    frameStart := 0 },
  { event := event26105
    frameStart := 0 },
  { event := event26106
    frameStart := 0 },
  { event := event26107
    frameStart := 0 },
  { event := event26108
    frameStart := 0 },
  { event := event26109
    frameStart := 0 },
  { event := event26110
    frameStart := 0 },
  { event := event26111
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events101
