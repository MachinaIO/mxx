import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events187

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event47872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42855⟩⟩) 0 ⟨7194⟩ 47871

def event47873 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42855⟩⟩) 1 ⟨42854⟩ 47868

def event47874 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42855⟩⟩) (.sum [.predecessor 0 47872 .coefficient, .predecessor 1 47873 .coefficient])

def exact47875RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47875RawTermsValid :
    exact47875RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47875 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42855⟩⟩) exact47875RawTerms .large 47874 .exactZero (none)

def event47876 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44391⟩⟩) 0 ⟨42855⟩ 47875

def event47877 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44391⟩⟩) 1 ⟨44390⟩ 47860

def event47878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44391⟩⟩) (.sum [.predecessor 0 47876 .coefficient, .predecessor 1 47877 .coefficient])

def exact47879RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨43837⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47879RawTermsValid :
    exact47879RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47879 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44391⟩⟩) exact47879RawTerms .large 47878 .exactZero (none)

def event47880 : Event := .preFoldPolynomial 47879 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨43837⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact47881RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨43837⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event47881 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨44391⟩⟩) 47880 exact47881RawTerms .large 47878 .exactZero (none)

def event47882 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨42668⟩⟩) ⟨⟨73⟩, ⟨52⟩, ⟨135⟩⟩ ⟨47716, 47882⟩

def event47883 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨43312⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩) (1) 0 2 (.universal 47882 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩) (none) 47881)

def event47884 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43312⟩⟩, .relation 47883 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩)

def event47885 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43312⟩⟩, .relation 47883 1, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩, (-1)⟩)

def event47886 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43312⟩⟩, .relation 47883 2, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨43837⟩⟩]⟩, (1)⟩)

def event47887 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43312⟩⟩, .relation 47883 3, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact47888RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨43837⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47888RawTermsValid :
    exact47888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47888 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43312⟩⟩) exact47888RawTerms .large 47712 (.finite 202072841853861888) (some (47714))

def event47889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44389⟩⟩) 0 ⟨43312⟩ 47888

def event47890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44389⟩⟩) 1 ⟨44388⟩ 47702

def event47891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44389⟩⟩) (.sum [.predecessor 0 47889 .coefficient, .predecessor 1 47890 .coefficient])

def event47892 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44389⟩⟩, .operator (⟨47888, 2⟩, ⟨47702, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨43837⟩⟩]⟩, (-1)⟩)

def event47893 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44389⟩⟩, .operator (⟨47888, 1⟩, ⟨47702, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩, (1)⟩)

def event47894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44389⟩⟩) (.sum [.result 47888 .summary, .result 47702 .summary])

def exact47895RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47895RawTermsValid :
    exact47895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44389⟩⟩) exact47895RawTerms .large 47891 (.finite 2998273677530297008128) (some (47894))

def event47896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44871⟩⟩) 0 ⟨44389⟩ 47895

def event47897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44871⟩⟩) 1 ⟨44869⟩ 47618

def event47898 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44871⟩⟩) (.product (.predecessor 0 47896 .coefficient) (.predecessor 1 47897 .coefficient) (⟨false, false, none, none, none⟩))

def event47899 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44871⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩) [⟨.result 47618 .coefficient, false, none⟩])

def event47900 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44871⟩⟩) (.product (.result 47895 .summary) (.transfer 47899) (⟨false, false, none, none, none⟩))

def event47901 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44871⟩⟩, .operator (⟨47895, 0⟩, ⟨47618, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩, (1)⟩)

def event47902 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44871⟩⟩, .operator (⟨47895, 1⟩, ⟨47618, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩, (-1)⟩)

def event47903 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44871⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44869⟩⟩) ⟨44013⟩ 47615)

def event47904 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44871⟩⟩, .relation 47903 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨44013⟩⟩]⟩, (-1)⟩)

def exact47905RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨44013⟩⟩]⟩, (-1)⟩]

theorem exact47905RawTermsValid :
    exact47905RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47905 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44871⟩⟩) exact47905RawTerms .large 47898 (.finite 32193718473625689247691015454720) (some (47900))

def event47906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43696⟩⟩) 0 ⟨42853⟩ 1653

def event47907 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43696⟩⟩) (.authority (.relationPreimageSource ⟨90⟩))

def exact47908RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43696⟩⟩]⟩, (1)⟩]

theorem exact47908RawTermsValid :
    exact47908RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47908 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43696⟩⟩) exact47908RawTerms (.finite 5647228698) 47907 .exactZero (none)

def event47909 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43698⟩⟩) 0 ⟨43696⟩ 47908

def event47910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43698⟩⟩) 1 ⟨2370⟩ 4

def event47911 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43698⟩⟩) (.scale (.predecessor 0 47909 .coefficient) (.value (.predecessor 1 47910 .coefficient)))

def exact47912RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43696⟩⟩]⟩, (1)⟩]

theorem exact47912RawTermsValid :
    exact47912RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47912 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43698⟩⟩) exact47912RawTerms (.finite 5647228698) 47911 .exactZero (none)

def event47913 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43699⟩⟩) 0 ⟨11216⟩ 46745

def event47914 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43699⟩⟩) 1 ⟨43698⟩ 47912

def event47915 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43699⟩⟩) (.product (.predecessor 0 47913 .coefficient) (.predecessor 1 47914 .coefficient) (⟨false, false, none, none, none⟩))

def event47916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43699⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨43696⟩⟩]⟩) [⟨.result 47908 .coefficient, false, none⟩])

def event47917 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43699⟩⟩) (.product (.result 46745 .summary) (.transfer 47916) (⟨false, false, none, none, none⟩))

def event47918 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43699⟩⟩, .operator (⟨46745, 0⟩, ⟨47912, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43696⟩⟩]⟩, (1)⟩)

def event47919 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨43697⟩⟩)

def event47920 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event47921 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event47922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event47923 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event47924 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event47925 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event47926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event47927 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event47928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 47927

def event47929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 47925

def event47930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 47928 .coefficient) (.value (.predecessor 1 47929 .coefficient)))

def event47931 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event47932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 47931

def event47933 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 47923

def event47934 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 47932 .coefficient, .predecessor 1 47933 .coefficient])

def event47935 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event47936 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 47935

def event47937 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 47921

def event47938 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 47937 .coefficient))

def event47939 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event47940 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42666⟩⟩) 0 ⟨11173⟩ 47939

def event47941 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42666⟩⟩) (.authority (.programFamilyFact))

def exact47942RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42666⟩⟩], []⟩, (1)⟩]

theorem exact47942RawTermsValid :
    exact47942RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47942 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42666⟩⟩) exact47942RawTerms (.finite 52) 47941 .exactZero (none)

def event47943 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14601⟩⟩) 0 ⟨11173⟩ 47939

def event47944 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14601⟩⟩) (.authority (.programFamilyFact))

def exact47945RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩], []⟩, (1)⟩]

theorem exact47945RawTermsValid :
    exact47945RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47945 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14601⟩⟩) exact47945RawTerms (.finite 52) 47944 .exactZero (none)

def event47946 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42667⟩⟩) 0 ⟨14601⟩ 47945

def event47947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42667⟩⟩) 1 ⟨42666⟩ 47942

def event47948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42667⟩⟩) (.product (.predecessor 0 47946 .coefficient) (.predecessor 1 47947 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event47949 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42667⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], []⟩) [⟨.result 47945 .coefficient, true, some 1⟩, ⟨.result 47942 .coefficient, true, some 1⟩])

def event47950 : Event := .survivorFold (1) 47949

def exact47951RawTerms : List Term := []

theorem exact47951RawTermsValid :
    exact47951RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47951 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42667⟩⟩) exact47951RawTerms (.finite 2704) 47948 (.finite 2704) (some (47949))

def event47952 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42668⟩⟩) 0 ⟨42667⟩ 47951

def event47953 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42668⟩⟩) (.identity (.predecessor 0 47952 .coefficient))

def event47954 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42668⟩⟩) (.finite 2704)

def event47955 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42852⟩⟩) 0 ⟨42668⟩ 47954

def event47956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42852⟩⟩) (.authority (.programFamilyFact))

def exact47957RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], []⟩, (1)⟩]

theorem exact47957RawTermsValid :
    exact47957RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47957 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42852⟩⟩) exact47957RawTerms (.finite 52) 47956 .exactZero (none)

def event47958 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42853⟩⟩) 0 ⟨42852⟩ 47957

def event47959 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42853⟩⟩) (.identity (.predecessor 0 47958 .coefficient))

def event47960 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42853⟩⟩) (.finite 52)

def event47961 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43696⟩⟩) 0 ⟨42853⟩ 47960

def event47962 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43696⟩⟩) (.authority (.relationPreimageSource ⟨90⟩))

def exact47963RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43696⟩⟩]⟩, (1)⟩]

theorem exact47963RawTermsValid :
    exact47963RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47963 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43696⟩⟩) exact47963RawTerms (.finite 5647228698) 47962 .exactZero (none)

def event47964 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact47965RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact47965RawTermsValid :
    exact47965RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47965 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact47965RawTerms .large 47964 .exactZero (none)

def event47966 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43697⟩⟩) 0 ⟨35⟩ 47965

def event47967 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43697⟩⟩) 1 ⟨43696⟩ 47963

def event47968 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43697⟩⟩) (.product (.predecessor 0 47966 .coefficient) (.predecessor 1 47967 .coefficient) (⟨false, false, none, none, none⟩))

def event47969 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43697⟩⟩, .operator (⟨47965, 0⟩, ⟨47963, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43696⟩⟩]⟩, (1)⟩)

def exact47970RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43696⟩⟩]⟩, (1)⟩]

theorem exact47970RawTermsValid :
    exact47970RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47970 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43697⟩⟩) exact47970RawTerms .large 47968 .exactZero (none)

def event47971 : Event := .preFoldPolynomial 47970 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43696⟩⟩]⟩, (1)⟩] .exactZero none

def exact47972RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43696⟩⟩]⟩, (1)⟩]

def event47972 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨43697⟩⟩) 47971 exact47972RawTerms .large 47968 .exactZero (none)

def event47973 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨44873⟩⟩)

def event47974 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event47975 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event47976 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event47977 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event47978 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event47979 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event47980 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event47981 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event47982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 47981

def event47983 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 47979

def event47984 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 47982 .coefficient) (.value (.predecessor 1 47983 .coefficient)))

def event47985 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event47986 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 47985

def event47987 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 47977

def event47988 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 47986 .coefficient, .predecessor 1 47987 .coefficient])

def event47989 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event47990 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 47989

def event47991 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 47975

def event47992 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 47991 .coefficient))

def event47993 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event47994 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42666⟩⟩) 0 ⟨11173⟩ 47993

def event47995 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42666⟩⟩) (.authority (.programFamilyFact))

def exact47996RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42666⟩⟩], []⟩, (1)⟩]

theorem exact47996RawTermsValid :
    exact47996RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47996 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42666⟩⟩) exact47996RawTerms (.finite 52) 47995 .exactZero (none)

def event47997 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14601⟩⟩) 0 ⟨11173⟩ 47993

def event47998 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14601⟩⟩) (.authority (.programFamilyFact))

def exact47999RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩], []⟩, (1)⟩]

theorem exact47999RawTermsValid :
    exact47999RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47999 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14601⟩⟩) exact47999RawTerms (.finite 52) 47998 .exactZero (none)

def event48000 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42667⟩⟩) 0 ⟨14601⟩ 47999

def event48001 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42667⟩⟩) 1 ⟨42666⟩ 47996

def event48002 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42667⟩⟩) (.product (.predecessor 0 48000 .coefficient) (.predecessor 1 48001 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event48003 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42667⟩⟩, .operator (⟨47999, 0⟩, ⟨47996, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], []⟩, (1)⟩)

def exact48004RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], []⟩, (1)⟩]

theorem exact48004RawTermsValid :
    exact48004RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48004 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42667⟩⟩) exact48004RawTerms (.finite 2704) 48002 .exactZero (none)

def event48005 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42668⟩⟩) 0 ⟨42667⟩ 48004

def event48006 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42668⟩⟩) (.identity (.predecessor 0 48005 .coefficient))

def event48007 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42668⟩⟩) (.finite 2704)

def event48008 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42852⟩⟩) 0 ⟨42668⟩ 48007

def event48009 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42852⟩⟩) (.authority (.programFamilyFact))

def exact48010RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], []⟩, (1)⟩]

theorem exact48010RawTermsValid :
    exact48010RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48010 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42852⟩⟩) exact48010RawTerms (.finite 52) 48009 .exactZero (none)

def event48011 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42853⟩⟩) 0 ⟨42852⟩ 48010

def event48012 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42853⟩⟩) (.identity (.predecessor 0 48011 .coefficient))

def event48013 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42853⟩⟩) (.finite 52)

def event48014 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44011⟩⟩) 0 ⟨42853⟩ 48013

def event48015 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44011⟩⟩) (.authority (.programFamilyFact))

def event48016 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44011⟩⟩) (.finite 3720)

def event48017 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event48018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44013⟩⟩) 0 ⟨7177⟩ 48017

def event48019 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44013⟩⟩) 1 ⟨44011⟩ 48016

def event48020 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44013⟩⟩) (.authority (.operator))

def exact48021RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44013⟩⟩]⟩, (1)⟩]

theorem exact48021RawTermsValid :
    exact48021RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48021 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44013⟩⟩) exact48021RawTerms .large 48020 .exactZero (none)

def event48022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44869⟩⟩) 0 ⟨44013⟩ 48021

def event48023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44869⟩⟩) (.authority (.operator))

def exact48024RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩, (1)⟩]

theorem exact48024RawTermsValid :
    exact48024RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48024 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44869⟩⟩) exact48024RawTerms (.finite 8192) 48023 .exactZero (none)

def event48025 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event48026 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event48027 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44178⟩⟩) 0 ⟨42853⟩ 48013

def event48028 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44178⟩⟩) 1 ⟨136⟩ 48026

def event48029 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44178⟩⟩) (.sum [.predecessor 0 48027 .coefficient, .predecessor 1 48028 .coefficient])

def event48030 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44178⟩⟩) (.finite 52)

def event48031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44179⟩⟩) 0 ⟨44178⟩ 48030

def event48032 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44179⟩⟩) (.identity (.predecessor 0 48031 .coefficient))

def exact48033RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], []⟩, (1)⟩]

theorem exact48033RawTermsValid :
    exact48033RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48033 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44179⟩⟩) exact48033RawTerms (.finite 52) 48032 .exactZero (none)

def event48034 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact48035RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact48035RawTermsValid :
    exact48035RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48035 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact48035RawTerms .large 48034 .exactZero (none)

def event48036 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44180⟩⟩) 0 ⟨6908⟩ 48035

def event48037 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44180⟩⟩) 1 ⟨44179⟩ 48033

def event48038 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44180⟩⟩) (.product (.predecessor 0 48036 .coefficient) (.predecessor 1 48037 .coefficient) (⟨false, false, none, none, none⟩))

def event48039 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44180⟩⟩, .operator (⟨48035, 0⟩, ⟨48033, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact48040RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact48040RawTermsValid :
    exact48040RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48040 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44180⟩⟩) exact48040RawTerms .large 48038 .exactZero (none)

def event48041 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7194⟩⟩) 0 ⟨7177⟩ 48017

def event48042 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7194⟩⟩) (.authority (.operator))

def exact48043RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩]

theorem exact48043RawTermsValid :
    exact48043RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48043 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7194⟩⟩) exact48043RawTerms .large 48042 .exactZero (none)

def event48044 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44181⟩⟩) 0 ⟨7194⟩ 48043

def event48045 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44181⟩⟩) 1 ⟨44180⟩ 48040

def event48046 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44181⟩⟩) (.sum [.predecessor 0 48044 .coefficient, .predecessor 1 48045 .coefficient])

def exact48047RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48047RawTermsValid :
    exact48047RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48047 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44181⟩⟩) exact48047RawTerms .large 48046 .exactZero (none)

def event48048 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44870⟩⟩) 0 ⟨44181⟩ 48047

def event48049 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44870⟩⟩) 1 ⟨44869⟩ 48024

def event48050 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44870⟩⟩) (.product (.predecessor 0 48048 .coefficient) (.predecessor 1 48049 .coefficient) (⟨false, false, none, none, none⟩))

def event48051 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44870⟩⟩, .operator (⟨48047, 0⟩, ⟨48024, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩, (1)⟩)

def event48052 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44870⟩⟩, .operator (⟨48047, 1⟩, ⟨48024, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩, (-1)⟩)

def event48053 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44870⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44869⟩⟩) ⟨44013⟩ 48021)

def event48054 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44870⟩⟩, .relation 48053 0, ⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨44013⟩⟩]⟩, (-1)⟩)

def exact48055RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨44013⟩⟩]⟩, (-1)⟩]

theorem exact48055RawTermsValid :
    exact48055RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48055 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44870⟩⟩) exact48055RawTerms .large 48050 .exactZero (none)

def event48056 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43103⟩⟩) 0 ⟨42853⟩ 48013

def event48057 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43103⟩⟩) (.authority (.programFamilyFact))

def exact48058RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨43103⟩⟩], []⟩, (1)⟩]

theorem exact48058RawTermsValid :
    exact48058RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48058 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43103⟩⟩) exact48058RawTerms (.finite 63) 48057 .exactZero (none)

def event48059 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43104⟩⟩) 0 ⟨6908⟩ 48035

def event48060 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43104⟩⟩) 1 ⟨43103⟩ 48058

def event48061 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43104⟩⟩) (.product (.predecessor 0 48059 .coefficient) (.predecessor 1 48060 .coefficient) (⟨false, true, none, none, some 1⟩))

def event48062 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43104⟩⟩, .operator (⟨48035, 0⟩, ⟨48058, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact48063RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact48063RawTermsValid :
    exact48063RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48063 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43104⟩⟩) exact48063RawTerms .large 48061 .exactZero (none)

def event48064 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7228⟩⟩) 0 ⟨7177⟩ 48017

def event48065 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7228⟩⟩) (.authority (.operator))

def exact48066RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩]

theorem exact48066RawTermsValid :
    exact48066RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48066 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7228⟩⟩) exact48066RawTerms .large 48065 .exactZero (none)

def event48067 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43105⟩⟩) 0 ⟨7228⟩ 48066

def event48068 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43105⟩⟩) 1 ⟨43104⟩ 48063

def event48069 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43105⟩⟩) (.sum [.predecessor 0 48067 .coefficient, .predecessor 1 48068 .coefficient])

def exact48070RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48070RawTermsValid :
    exact48070RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48070 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43105⟩⟩) exact48070RawTerms .large 48069 .exactZero (none)

def event48071 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44873⟩⟩) 0 ⟨43105⟩ 48070

def event48072 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44873⟩⟩) 1 ⟨44870⟩ 48055

def event48073 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44873⟩⟩) (.sum [.predecessor 0 48071 .coefficient, .predecessor 1 48072 .coefficient])

def exact48074RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨44013⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48074RawTermsValid :
    exact48074RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48074 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44873⟩⟩) exact48074RawTerms .large 48073 .exactZero (none)

def event48075 : Event := .preFoldPolynomial 48074 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨44013⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact48076RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨44013⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event48076 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨44873⟩⟩) 48075 exact48076RawTerms .large 48073 .exactZero (none)

def event48077 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨42853⟩⟩) ⟨⟨107⟩, ⟨90⟩, ⟨135⟩⟩ ⟨47919, 48077⟩

def event48078 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨43699⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43696⟩⟩]⟩) (1) 0 2 (.universal 48077 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43696⟩⟩]⟩) (none) 48076)

def event48079 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43699⟩⟩, .relation 48078 1, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩)

def event48080 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43699⟩⟩, .relation 48078 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩, (-1)⟩)

def event48081 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43699⟩⟩, .relation 48078 2, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨44013⟩⟩]⟩, (1)⟩)

def event48082 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43699⟩⟩, .relation 48078 3, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact48083RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨44013⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48083RawTermsValid :
    exact48083RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48083 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43699⟩⟩) exact48083RawTerms .large 47915 (.finite 202072841853861888) (some (47917))

def event48084 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44872⟩⟩) 0 ⟨43699⟩ 48083

def event48085 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44872⟩⟩) 1 ⟨44871⟩ 47905

def event48086 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44872⟩⟩) (.sum [.predecessor 0 48084 .coefficient, .predecessor 1 48085 .coefficient])

def event48087 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44872⟩⟩, .operator (⟨48083, 0⟩, ⟨47905, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩, (1)⟩)

def event48088 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44872⟩⟩, .operator (⟨48083, 2⟩, ⟨47905, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨44013⟩⟩]⟩, (-1)⟩)

def event48089 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44872⟩⟩) (.sum [.result 48083 .summary, .result 47905 .summary])

def exact48090RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48090RawTermsValid :
    exact48090RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48090 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44872⟩⟩) exact48090RawTerms .large 48086 (.finite 32193718473625891320532869316608) (some (48089))

def event48091 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41331⟩⟩) 0 ⟨40173⟩ 1676

def event48092 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41331⟩⟩) (.authority (.programFamilyFact))

def event48093 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41331⟩⟩) (.finite 3720)

def event48094 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41333⟩⟩) 0 ⟨7177⟩ 15500

def event48095 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41333⟩⟩) 1 ⟨41331⟩ 48093

def event48096 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41333⟩⟩) (.authority (.operator))

def exact48097RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41333⟩⟩]⟩, (1)⟩]

theorem exact48097RawTermsValid :
    exact48097RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48097 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41333⟩⟩) exact48097RawTerms .large 48096 .exactZero (none)

def event48098 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42189⟩⟩) 0 ⟨41333⟩ 48097

def event48099 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42189⟩⟩) (.authority (.operator))

def exact48100RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨42189⟩⟩]⟩, (1)⟩]

theorem exact48100RawTermsValid :
    exact48100RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48100 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42189⟩⟩) exact48100RawTerms (.finite 8192) 48099 .exactZero (none)

def event48101 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41156⟩⟩) 0 ⟨39988⟩ 1670

def event48102 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41156⟩⟩) (.authority (.programFamilyFact))

def event48103 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41156⟩⟩) (.finite 3720)

def event48104 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41157⟩⟩) 0 ⟨7177⟩ 15500

def event48105 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41157⟩⟩) 1 ⟨41156⟩ 48103

def event48106 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41157⟩⟩) (.authority (.operator))

def exact48107RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41157⟩⟩]⟩, (1)⟩]

theorem exact48107RawTermsValid :
    exact48107RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48107 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41157⟩⟩) exact48107RawTerms .large 48106 .exactZero (none)

def event48108 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41707⟩⟩) 0 ⟨41157⟩ 48107

def event48109 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41707⟩⟩) (.authority (.operator))

def exact48110RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41707⟩⟩]⟩, (1)⟩]

theorem exact48110RawTermsValid :
    exact48110RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48110 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41707⟩⟩) exact48110RawTerms (.finite 8192) 48109 .exactZero (none)

def event48111 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39989⟩⟩) 0 ⟨39986⟩ 1659

def event48112 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39989⟩⟩) 1 ⟨11176⟩ 46653

def event48113 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39989⟩⟩) (.tensor (.predecessor 0 48111 .coefficient) (.predecessor 1 48112 .coefficient) true false)

def event48114 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39989⟩⟩, .operator (⟨1659, 0⟩, ⟨46653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact48115RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact48115RawTermsValid :
    exact48115RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48115 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39989⟩⟩) exact48115RawTerms .large 48113 .exactZero (none)

def event48116 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11188⟩⟩) 0 ⟨11175⟩ 46523

def event48117 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11188⟩⟩) 1 ⟨7282⟩ 18583

def event48118 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11188⟩⟩) (.product (.predecessor 0 48116 .coefficient) (.predecessor 1 48117 .coefficient) (⟨false, false, none, none, none⟩))

def event48119 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11188⟩⟩, .operator (⟨46523, 0⟩, ⟨18583, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩)

def exact48120RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩]

theorem exact48120RawTermsValid :
    exact48120RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48120 : Event := .resultExact (⟨.program ⟨257⟩, ⟨11188⟩⟩) exact48120RawTerms .large 48118 .exactZero (none)

def event48121 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39990⟩⟩) 0 ⟨11188⟩ 48120

def event48122 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39990⟩⟩) 1 ⟨39989⟩ 48115

def event48123 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39990⟩⟩) (.sum [.predecessor 0 48121 .coefficient, .predecessor 1 48122 .coefficient])

def exact48124RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48124RawTermsValid :
    exact48124RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48124 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39990⟩⟩) exact48124RawTerms .large 48123 .exactZero (none)

def event48125 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39991⟩⟩) 0 ⟨39990⟩ 48124

def event48126 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39991⟩⟩) 1 ⟨108⟩ 18575

def event48127 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39991⟩⟩) (.sum [.predecessor 0 48125 .coefficient, .predecessor 1 48126 .coefficient])

def eventLeaf2992 : Array AnnotatedEvent := #[
  { event := event47872
    frameStart := 47764 },
  { event := event47873
    frameStart := 47764 },
  { event := event47874
    frameStart := 47764 },
  { event := event47875
    frameStart := 47764 },
  { event := event47876
    frameStart := 47764 },
  { event := event47877
    frameStart := 47764 },
  { event := event47878
    frameStart := 47764 },
  { event := event47879
    frameStart := 47764 },
  { event := event47880
    frameStart := 47764 },
  { event := event47881
    frameStart := 47764 },
  { event := event47882
    frameStart := 0 },
  { event := event47883
    frameStart := 0 },
  { event := event47884
    frameStart := 0 },
  { event := event47885
    frameStart := 0 },
  { event := event47886
    frameStart := 0 },
  { event := event47887
    frameStart := 0 }
]

def eventLeaf2993 : Array AnnotatedEvent := #[
  { event := event47888
    frameStart := 0 },
  { event := event47889
    frameStart := 0 },
  { event := event47890
    frameStart := 0 },
  { event := event47891
    frameStart := 0 },
  { event := event47892
    frameStart := 0 },
  { event := event47893
    frameStart := 0 },
  { event := event47894
    frameStart := 0 },
  { event := event47895
    frameStart := 0 },
  { event := event47896
    frameStart := 0 },
  { event := event47897
    frameStart := 0 },
  { event := event47898
    frameStart := 0 },
  { event := event47899
    frameStart := 0 },
  { event := event47900
    frameStart := 0 },
  { event := event47901
    frameStart := 0 },
  { event := event47902
    frameStart := 0 },
  { event := event47903
    frameStart := 0 }
]

def eventLeaf2994 : Array AnnotatedEvent := #[
  { event := event47904
    frameStart := 0 },
  { event := event47905
    frameStart := 0 },
  { event := event47906
    frameStart := 0 },
  { event := event47907
    frameStart := 0 },
  { event := event47908
    frameStart := 0 },
  { event := event47909
    frameStart := 0 },
  { event := event47910
    frameStart := 0 },
  { event := event47911
    frameStart := 0 },
  { event := event47912
    frameStart := 0 },
  { event := event47913
    frameStart := 0 },
  { event := event47914
    frameStart := 0 },
  { event := event47915
    frameStart := 0 },
  { event := event47916
    frameStart := 0 },
  { event := event47917
    frameStart := 0 },
  { event := event47918
    frameStart := 0 },
  { event := event47919
    frameStart := 47919 }
]

def eventLeaf2995 : Array AnnotatedEvent := #[
  { event := event47920
    frameStart := 47919 },
  { event := event47921
    frameStart := 47919 },
  { event := event47922
    frameStart := 47919 },
  { event := event47923
    frameStart := 47919 },
  { event := event47924
    frameStart := 47919 },
  { event := event47925
    frameStart := 47919 },
  { event := event47926
    frameStart := 47919 },
  { event := event47927
    frameStart := 47919 },
  { event := event47928
    frameStart := 47919 },
  { event := event47929
    frameStart := 47919 },
  { event := event47930
    frameStart := 47919 },
  { event := event47931
    frameStart := 47919 },
  { event := event47932
    frameStart := 47919 },
  { event := event47933
    frameStart := 47919 },
  { event := event47934
    frameStart := 47919 },
  { event := event47935
    frameStart := 47919 }
]

def eventLeaf2996 : Array AnnotatedEvent := #[
  { event := event47936
    frameStart := 47919 },
  { event := event47937
    frameStart := 47919 },
  { event := event47938
    frameStart := 47919 },
  { event := event47939
    frameStart := 47919 },
  { event := event47940
    frameStart := 47919 },
  { event := event47941
    frameStart := 47919 },
  { event := event47942
    frameStart := 47919 },
  { event := event47943
    frameStart := 47919 },
  { event := event47944
    frameStart := 47919 },
  { event := event47945
    frameStart := 47919 },
  { event := event47946
    frameStart := 47919 },
  { event := event47947
    frameStart := 47919 },
  { event := event47948
    frameStart := 47919 },
  { event := event47949
    frameStart := 47919 },
  { event := event47950
    frameStart := 47919 },
  { event := event47951
    frameStart := 47919 }
]

def eventLeaf2997 : Array AnnotatedEvent := #[
  { event := event47952
    frameStart := 47919 },
  { event := event47953
    frameStart := 47919 },
  { event := event47954
    frameStart := 47919 },
  { event := event47955
    frameStart := 47919 },
  { event := event47956
    frameStart := 47919 },
  { event := event47957
    frameStart := 47919 },
  { event := event47958
    frameStart := 47919 },
  { event := event47959
    frameStart := 47919 },
  { event := event47960
    frameStart := 47919 },
  { event := event47961
    frameStart := 47919 },
  { event := event47962
    frameStart := 47919 },
  { event := event47963
    frameStart := 47919 },
  { event := event47964
    frameStart := 47919 },
  { event := event47965
    frameStart := 47919 },
  { event := event47966
    frameStart := 47919 },
  { event := event47967
    frameStart := 47919 }
]

def eventLeaf2998 : Array AnnotatedEvent := #[
  { event := event47968
    frameStart := 47919 },
  { event := event47969
    frameStart := 47919 },
  { event := event47970
    frameStart := 47919 },
  { event := event47971
    frameStart := 47919 },
  { event := event47972
    frameStart := 47919 },
  { event := event47973
    frameStart := 47973 },
  { event := event47974
    frameStart := 47973 },
  { event := event47975
    frameStart := 47973 },
  { event := event47976
    frameStart := 47973 },
  { event := event47977
    frameStart := 47973 },
  { event := event47978
    frameStart := 47973 },
  { event := event47979
    frameStart := 47973 },
  { event := event47980
    frameStart := 47973 },
  { event := event47981
    frameStart := 47973 },
  { event := event47982
    frameStart := 47973 },
  { event := event47983
    frameStart := 47973 }
]

def eventLeaf2999 : Array AnnotatedEvent := #[
  { event := event47984
    frameStart := 47973 },
  { event := event47985
    frameStart := 47973 },
  { event := event47986
    frameStart := 47973 },
  { event := event47987
    frameStart := 47973 },
  { event := event47988
    frameStart := 47973 },
  { event := event47989
    frameStart := 47973 },
  { event := event47990
    frameStart := 47973 },
  { event := event47991
    frameStart := 47973 },
  { event := event47992
    frameStart := 47973 },
  { event := event47993
    frameStart := 47973 },
  { event := event47994
    frameStart := 47973 },
  { event := event47995
    frameStart := 47973 },
  { event := event47996
    frameStart := 47973 },
  { event := event47997
    frameStart := 47973 },
  { event := event47998
    frameStart := 47973 },
  { event := event47999
    frameStart := 47973 }
]

def eventLeaf3000 : Array AnnotatedEvent := #[
  { event := event48000
    frameStart := 47973 },
  { event := event48001
    frameStart := 47973 },
  { event := event48002
    frameStart := 47973 },
  { event := event48003
    frameStart := 47973 },
  { event := event48004
    frameStart := 47973 },
  { event := event48005
    frameStart := 47973 },
  { event := event48006
    frameStart := 47973 },
  { event := event48007
    frameStart := 47973 },
  { event := event48008
    frameStart := 47973 },
  { event := event48009
    frameStart := 47973 },
  { event := event48010
    frameStart := 47973 },
  { event := event48011
    frameStart := 47973 },
  { event := event48012
    frameStart := 47973 },
  { event := event48013
    frameStart := 47973 },
  { event := event48014
    frameStart := 47973 },
  { event := event48015
    frameStart := 47973 }
]

def eventLeaf3001 : Array AnnotatedEvent := #[
  { event := event48016
    frameStart := 47973 },
  { event := event48017
    frameStart := 47973 },
  { event := event48018
    frameStart := 47973 },
  { event := event48019
    frameStart := 47973 },
  { event := event48020
    frameStart := 47973 },
  { event := event48021
    frameStart := 47973 },
  { event := event48022
    frameStart := 47973 },
  { event := event48023
    frameStart := 47973 },
  { event := event48024
    frameStart := 47973 },
  { event := event48025
    frameStart := 47973 },
  { event := event48026
    frameStart := 47973 },
  { event := event48027
    frameStart := 47973 },
  { event := event48028
    frameStart := 47973 },
  { event := event48029
    frameStart := 47973 },
  { event := event48030
    frameStart := 47973 },
  { event := event48031
    frameStart := 47973 }
]

def eventLeaf3002 : Array AnnotatedEvent := #[
  { event := event48032
    frameStart := 47973 },
  { event := event48033
    frameStart := 47973 },
  { event := event48034
    frameStart := 47973 },
  { event := event48035
    frameStart := 47973 },
  { event := event48036
    frameStart := 47973 },
  { event := event48037
    frameStart := 47973 },
  { event := event48038
    frameStart := 47973 },
  { event := event48039
    frameStart := 47973 },
  { event := event48040
    frameStart := 47973 },
  { event := event48041
    frameStart := 47973 },
  { event := event48042
    frameStart := 47973 },
  { event := event48043
    frameStart := 47973 },
  { event := event48044
    frameStart := 47973 },
  { event := event48045
    frameStart := 47973 },
  { event := event48046
    frameStart := 47973 },
  { event := event48047
    frameStart := 47973 }
]

def eventLeaf3003 : Array AnnotatedEvent := #[
  { event := event48048
    frameStart := 47973 },
  { event := event48049
    frameStart := 47973 },
  { event := event48050
    frameStart := 47973 },
  { event := event48051
    frameStart := 47973 },
  { event := event48052
    frameStart := 47973 },
  { event := event48053
    frameStart := 47973 },
  { event := event48054
    frameStart := 47973 },
  { event := event48055
    frameStart := 47973 },
  { event := event48056
    frameStart := 47973 },
  { event := event48057
    frameStart := 47973 },
  { event := event48058
    frameStart := 47973 },
  { event := event48059
    frameStart := 47973 },
  { event := event48060
    frameStart := 47973 },
  { event := event48061
    frameStart := 47973 },
  { event := event48062
    frameStart := 47973 },
  { event := event48063
    frameStart := 47973 }
]

def eventLeaf3004 : Array AnnotatedEvent := #[
  { event := event48064
    frameStart := 47973 },
  { event := event48065
    frameStart := 47973 },
  { event := event48066
    frameStart := 47973 },
  { event := event48067
    frameStart := 47973 },
  { event := event48068
    frameStart := 47973 },
  { event := event48069
    frameStart := 47973 },
  { event := event48070
    frameStart := 47973 },
  { event := event48071
    frameStart := 47973 },
  { event := event48072
    frameStart := 47973 },
  { event := event48073
    frameStart := 47973 },
  { event := event48074
    frameStart := 47973 },
  { event := event48075
    frameStart := 47973 },
  { event := event48076
    frameStart := 47973 },
  { event := event48077
    frameStart := 0 },
  { event := event48078
    frameStart := 0 },
  { event := event48079
    frameStart := 0 }
]

def eventLeaf3005 : Array AnnotatedEvent := #[
  { event := event48080
    frameStart := 0 },
  { event := event48081
    frameStart := 0 },
  { event := event48082
    frameStart := 0 },
  { event := event48083
    frameStart := 0 },
  { event := event48084
    frameStart := 0 },
  { event := event48085
    frameStart := 0 },
  { event := event48086
    frameStart := 0 },
  { event := event48087
    frameStart := 0 },
  { event := event48088
    frameStart := 0 },
  { event := event48089
    frameStart := 0 },
  { event := event48090
    frameStart := 0 },
  { event := event48091
    frameStart := 0 },
  { event := event48092
    frameStart := 0 },
  { event := event48093
    frameStart := 0 },
  { event := event48094
    frameStart := 0 },
  { event := event48095
    frameStart := 0 }
]

def eventLeaf3006 : Array AnnotatedEvent := #[
  { event := event48096
    frameStart := 0 },
  { event := event48097
    frameStart := 0 },
  { event := event48098
    frameStart := 0 },
  { event := event48099
    frameStart := 0 },
  { event := event48100
    frameStart := 0 },
  { event := event48101
    frameStart := 0 },
  { event := event48102
    frameStart := 0 },
  { event := event48103
    frameStart := 0 },
  { event := event48104
    frameStart := 0 },
  { event := event48105
    frameStart := 0 },
  { event := event48106
    frameStart := 0 },
  { event := event48107
    frameStart := 0 },
  { event := event48108
    frameStart := 0 },
  { event := event48109
    frameStart := 0 },
  { event := event48110
    frameStart := 0 },
  { event := event48111
    frameStart := 0 }
]

def eventLeaf3007 : Array AnnotatedEvent := #[
  { event := event48112
    frameStart := 0 },
  { event := event48113
    frameStart := 0 },
  { event := event48114
    frameStart := 0 },
  { event := event48115
    frameStart := 0 },
  { event := event48116
    frameStart := 0 },
  { event := event48117
    frameStart := 0 },
  { event := event48118
    frameStart := 0 },
  { event := event48119
    frameStart := 0 },
  { event := event48120
    frameStart := 0 },
  { event := event48121
    frameStart := 0 },
  { event := event48122
    frameStart := 0 },
  { event := event48123
    frameStart := 0 },
  { event := event48124
    frameStart := 0 },
  { event := event48125
    frameStart := 0 },
  { event := event48126
    frameStart := 0 },
  { event := event48127
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events187
