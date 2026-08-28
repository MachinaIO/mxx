import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events730

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event186880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event186881 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event186882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17218⟩⟩) 0 ⟨15813⟩ 186868

def event186883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17218⟩⟩) 1 ⟨136⟩ 186881

def event186884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17218⟩⟩) (.sum [.predecessor 0 186882 .coefficient, .predecessor 1 186883 .coefficient])

def event186885 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17218⟩⟩) (.finite 2)

def event186886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17219⟩⟩) 0 ⟨17218⟩ 186885

def event186887 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17219⟩⟩) (.identity (.predecessor 0 186886 .coefficient))

def exact186888RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], []⟩, (1)⟩]

theorem exact186888RawTermsValid :
    exact186888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186888 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17219⟩⟩) exact186888RawTerms (.finite 2) 186887 .exactZero (none)

def event186889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact186890RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact186890RawTermsValid :
    exact186890RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186890 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact186890RawTerms .large 186889 .exactZero (none)

def event186891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17220⟩⟩) 0 ⟨6908⟩ 186890

def event186892 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17220⟩⟩) 1 ⟨17219⟩ 186888

def event186893 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17220⟩⟩) (.product (.predecessor 0 186891 .coefficient) (.predecessor 1 186892 .coefficient) (⟨false, false, none, none, none⟩))

def event186894 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17220⟩⟩, .operator (⟨186890, 0⟩, ⟨186888, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact186895RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact186895RawTermsValid :
    exact186895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17220⟩⟩) exact186895RawTerms .large 186893 .exactZero (none)

def event186896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 186872

def event186897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact186898RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact186898RawTermsValid :
    exact186898RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186898 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact186898RawTerms .large 186897 .exactZero (none)

def event186899 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17221⟩⟩) 0 ⟨7179⟩ 186898

def event186900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17221⟩⟩) 1 ⟨17220⟩ 186895

def event186901 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17221⟩⟩) (.sum [.predecessor 0 186899 .coefficient, .predecessor 1 186900 .coefficient])

def exact186902RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact186902RawTermsValid :
    exact186902RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186902 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17221⟩⟩) exact186902RawTerms .large 186901 .exactZero (none)

def event186903 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17846⟩⟩) 0 ⟨17221⟩ 186902

def event186904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17846⟩⟩) 1 ⟨17845⟩ 186879

def event186905 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17846⟩⟩) (.product (.predecessor 0 186903 .coefficient) (.predecessor 1 186904 .coefficient) (⟨false, false, none, none, none⟩))

def event186906 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17846⟩⟩, .operator (⟨186902, 0⟩, ⟨186879, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17845⟩⟩]⟩, (1)⟩)

def event186907 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17846⟩⟩, .operator (⟨186902, 1⟩, ⟨186879, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17845⟩⟩]⟩, (-1)⟩)

def event186908 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17846⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17845⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17845⟩⟩) ⟨17028⟩ 186876)

def event186909 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17846⟩⟩, .relation 186908 0, ⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨17028⟩⟩]⟩, (-1)⟩)

def exact186910RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17845⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨17028⟩⟩]⟩, (-1)⟩]

theorem exact186910RawTermsValid :
    exact186910RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186910 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17846⟩⟩) exact186910RawTerms .large 186905 .exactZero (none)

def event186911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16083⟩⟩) 0 ⟨15813⟩ 186868

def event186912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16083⟩⟩) (.authority (.programFamilyFact))

def exact186913RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], []⟩, (1)⟩]

theorem exact186913RawTermsValid :
    exact186913RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186913 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16083⟩⟩) exact186913RawTerms (.finite 43) 186912 .exactZero (none)

def event186914 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16084⟩⟩) 0 ⟨6908⟩ 186890

def event186915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16084⟩⟩) 1 ⟨16083⟩ 186913

def event186916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16084⟩⟩) (.product (.predecessor 0 186914 .coefficient) (.predecessor 1 186915 .coefficient) (⟨false, true, none, none, some 1⟩))

def event186917 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16084⟩⟩, .operator (⟨186890, 0⟩, ⟨186913, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact186918RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact186918RawTermsValid :
    exact186918RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186918 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16084⟩⟩) exact186918RawTerms .large 186916 .exactZero (none)

def event186919 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 186872

def event186920 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact186921RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact186921RawTermsValid :
    exact186921RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186921 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact186921RawTerms .large 186920 .exactZero (none)

def event186922 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16085⟩⟩) 0 ⟨7198⟩ 186921

def event186923 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16085⟩⟩) 1 ⟨16084⟩ 186918

def event186924 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16085⟩⟩) (.sum [.predecessor 0 186922 .coefficient, .predecessor 1 186923 .coefficient])

def exact186925RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact186925RawTermsValid :
    exact186925RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186925 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16085⟩⟩) exact186925RawTerms .large 186924 .exactZero (none)

def event186926 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17849⟩⟩) 0 ⟨16085⟩ 186925

def event186927 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17849⟩⟩) 1 ⟨17846⟩ 186910

def event186928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17849⟩⟩) (.sum [.predecessor 0 186926 .coefficient, .predecessor 1 186927 .coefficient])

def exact186929RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17845⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨17028⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact186929RawTermsValid :
    exact186929RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186929 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17849⟩⟩) exact186929RawTerms .large 186928 .exactZero (none)

def event186930 : Event := .preFoldPolynomial 186929 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17845⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨17028⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact186931RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17845⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨17028⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event186931 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17849⟩⟩) 186930 exact186931RawTerms .large 186928 .exactZero (none)

def event186932 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15813⟩⟩) ⟨⟨77⟩, ⟨57⟩, ⟨135⟩⟩ ⟨186774, 186932⟩

def event186933 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16659⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16656⟩⟩]⟩) (1) 0 2 (.universal 186932 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16656⟩⟩]⟩) (none) 186931)

def event186934 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16659⟩⟩, .relation 186933 1, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩)

def event186935 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16659⟩⟩, .relation 186933 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17845⟩⟩]⟩, (-1)⟩)

def event186936 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16659⟩⟩, .relation 186933 2, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨17028⟩⟩]⟩, (1)⟩)

def event186937 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16659⟩⟩, .relation 186933 3, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact186938RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17845⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨17028⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact186938RawTermsValid :
    exact186938RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186938 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16659⟩⟩) exact186938RawTerms .large 186770 (.finite 202072841853861888) (some (186772))

def event186939 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17848⟩⟩) 0 ⟨16659⟩ 186938

def event186940 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17848⟩⟩) 1 ⟨17847⟩ 186760

def event186941 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17848⟩⟩) (.sum [.predecessor 0 186939 .coefficient, .predecessor 1 186940 .coefficient])

def event186942 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17848⟩⟩, .operator (⟨186938, 0⟩, ⟨186760, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17845⟩⟩]⟩, (1)⟩)

def event186943 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17848⟩⟩, .operator (⟨186938, 2⟩, ⟨186760, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨17028⟩⟩]⟩, (-1)⟩)

def event186944 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17848⟩⟩) (.sum [.result 186938 .summary, .result 186760 .summary])

def exact186945RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact186945RawTermsValid :
    exact186945RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186945 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17848⟩⟩) exact186945RawTerms .large 186941 (.finite 32188807212483706889510625476608) (some (186944))

def event186946 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20749⟩⟩) 0 ⟨17848⟩ 186945

def event186947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20749⟩⟩) 1 ⟨20748⟩ 186463

def event186948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20749⟩⟩) (.sum [.predecessor 0 186946 .coefficient, .predecessor 1 186947 .coefficient])

def event186949 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20749⟩⟩) (.sum [.result 186945 .summary, .result 186463 .summary])

def exact186950RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact186950RawTermsValid :
    exact186950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186950 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20749⟩⟩) exact186950RawTerms .large 186948 (.finite 64377712650190257467641695830016) (some (186949))

def event186951 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23969⟩⟩) 0 ⟨20749⟩ 186950

def event186952 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23969⟩⟩) 1 ⟨23968⟩ 185981

def event186953 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23969⟩⟩) (.sum [.predecessor 0 186951 .coefficient, .predecessor 1 186952 .coefficient])

def event186954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23969⟩⟩) (.sum [.result 186950 .summary, .result 185981 .summary])

def exact186955RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact186955RawTermsValid :
    exact186955RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186955 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23969⟩⟩) exact186955RawTerms .large 186953 (.finite 96566716313119651734393211060224) (some (186954))

def event186956 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33989⟩⟩) 0 ⟨23969⟩ 186955

def event186957 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33989⟩⟩) 1 ⟨33988⟩ 185499

def event186958 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33989⟩⟩) (.sum [.predecessor 0 186956 .coefficient, .predecessor 1 186957 .coefficient])

def event186959 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33989⟩⟩) (.sum [.result 186955 .summary, .result 185499 .summary])

def exact186960RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact186960RawTermsValid :
    exact186960RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186960 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33989⟩⟩) exact186960RawTerms .large 186958 (.finite 128755916426494733378385616044032) (some (186959))

def event186961 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53049⟩⟩) 0 ⟨33989⟩ 186960

def event186962 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53049⟩⟩) 1 ⟨53048⟩ 185017

def event186963 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53049⟩⟩) (.sum [.predecessor 0 186961 .coefficient, .predecessor 1 186962 .coefficient])

def event186964 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53049⟩⟩) (.sum [.result 186960 .summary, .result 185017 .summary])

def exact186965RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact186965RawTermsValid :
    exact186965RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186965 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53049⟩⟩) exact186965RawTerms .large 186963 (.finite 160945509440761189776859800535040) (some (186964))

def event186966 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56029⟩⟩) 0 ⟨53049⟩ 186965

def event186967 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56029⟩⟩) 1 ⟨56028⟩ 184535

def event186968 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56029⟩⟩) (.sum [.predecessor 0 186966 .coefficient, .predecessor 1 186967 .coefficient])

def event186969 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56029⟩⟩) (.sum [.result 186965 .summary, .result 184535 .summary])

def exact186970RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact186970RawTermsValid :
    exact186970RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186970 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56029⟩⟩) exact186970RawTerms .large 186968 (.finite 193135298905473333552574874779648) (some (186969))

def event186971 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59009⟩⟩) 0 ⟨56029⟩ 186970

def event186972 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59009⟩⟩) 1 ⟨59008⟩ 184053

def event186973 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59009⟩⟩) (.sum [.predecessor 0 186971 .coefficient, .predecessor 1 186972 .coefficient])

def event186974 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59009⟩⟩) (.sum [.result 186970 .summary, .result 184053 .summary])

def exact186975RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact186975RawTermsValid :
    exact186975RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186975 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59009⟩⟩) exact186975RawTerms .large 186973 (.finite 225325481271076852082771728531456) (some (186974))

def event186976 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61989⟩⟩) 0 ⟨59009⟩ 186975

def event186977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61989⟩⟩) 1 ⟨61988⟩ 183571

def event186978 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61989⟩⟩) (.sum [.predecessor 0 186976 .coefficient, .predecessor 1 186977 .coefficient])

def event186979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61989⟩⟩) (.sum [.result 186975 .summary, .result 183571 .summary])

def exact186980RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact186980RawTermsValid :
    exact186980RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186980 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61989⟩⟩) exact186980RawTerms .large 186978 (.finite 257515860087126057990209472036864) (some (186979))

def event186981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64969⟩⟩) 0 ⟨61989⟩ 186980

def event186982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64969⟩⟩) 1 ⟨64968⟩ 183089

def event186983 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64969⟩⟩) (.sum [.predecessor 0 186981 .coefficient, .predecessor 1 186982 .coefficient])

def event186984 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64969⟩⟩) (.sum [.result 186980 .summary, .result 183089 .summary])

def exact186985RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact186985RawTermsValid :
    exact186985RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186985 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64969⟩⟩) exact186985RawTerms .large 186983 (.finite 289706631804066638652128995049472) (some (186984))

def event186986 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70418⟩⟩) 0 ⟨64969⟩ 186985

def event186987 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70418⟩⟩) 1 ⟨70417⟩ 182607

def event186988 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70418⟩⟩) (.sum [.predecessor 0 186986 .coefficient, .predecessor 1 186987 .coefficient])

def event186989 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70418⟩⟩) (.sum [.result 186985 .summary, .result 182607 .summary])

def exact186990RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact186990RawTermsValid :
    exact186990RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186990 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70418⟩⟩) exact186990RawTerms .large 186988 (.finite 321897992872344281445771187322880) (some (186989))

def event186991 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70419⟩⟩) 0 ⟨70418⟩ 186990

def event186992 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70419⟩⟩) 1 ⟨28367⟩ 182125

def event186993 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70419⟩⟩) (.sum [.predecessor 0 186991 .coefficient, .predecessor 1 186992 .coefficient])

def event186994 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70419⟩⟩) (.sum [.result 186990 .summary, .result 182125 .summary])

def exact186995RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact186995RawTermsValid :
    exact186995RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event186995 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70419⟩⟩) exact186995RawTerms .large 186993 (.finite 354089550391067611616654269349888) (some (186994))

def event186996 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70420⟩⟩) 0 ⟨70419⟩ 186995

def event186997 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70420⟩⟩) 1 ⟨31047⟩ 181643

def event186998 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70420⟩⟩) (.sum [.predecessor 0 186996 .coefficient, .predecessor 1 186997 .coefficient])

def event186999 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70420⟩⟩) (.sum [.result 186995 .summary, .result 181643 .summary])

def exact187000RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨29338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact187000RawTermsValid :
    exact187000RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event187000 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70420⟩⟩) exact187000RawTerms .large 186998 (.finite 386281697261128003919260020637696) (some (186999))

def event187001 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70421⟩⟩) 0 ⟨70420⟩ 187000

def event187002 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70421⟩⟩) 1 ⟨36707⟩ 181161

def event187003 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70421⟩⟩) (.sum [.predecessor 0 187001 .coefficient, .predecessor 1 187002 .coefficient])

def event187004 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70421⟩⟩) (.sum [.result 187000 .summary, .result 181161 .summary])

def exact187005RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨29338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨35002⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact187005RawTermsValid :
    exact187005RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event187005 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70421⟩⟩) exact187005RawTerms .large 187003 (.finite 418474237032079770976347551432704) (some (187004))

def event187006 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70422⟩⟩) 0 ⟨70421⟩ 187005

def event187007 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70422⟩⟩) 1 ⟨39387⟩ 180679

def event187008 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70422⟩⟩) (.sum [.predecessor 0 187006 .coefficient, .predecessor 1 187007 .coefficient])

def event187009 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70422⟩⟩) (.sum [.result 187005 .summary, .result 180679 .summary])

def exact187010RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨29338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨35002⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37682⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact187010RawTermsValid :
    exact187010RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event187010 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70422⟩⟩) exact187010RawTerms .large 187008 (.finite 450666973253477225410675971981312) (some (187009))

def event187011 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70423⟩⟩) 0 ⟨70422⟩ 187010

def event187012 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70423⟩⟩) 1 ⟨42067⟩ 180197

def event187013 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70423⟩⟩) (.sum [.predecessor 0 187011 .coefficient, .predecessor 1 187012 .coefficient])

def event187014 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70423⟩⟩) (.sum [.result 187010 .summary, .result 180197 .summary])

def exact187015RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨29338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨35002⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37682⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨40358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact187015RawTermsValid :
    exact187015RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event187015 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70423⟩⟩) exact187015RawTerms .large 187013 (.finite 482860102375766054599486172037120) (some (187014))

def event187016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70424⟩⟩) 0 ⟨70423⟩ 187015

def event187017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70424⟩⟩) 1 ⟨44747⟩ 179715

def event187018 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70424⟩⟩) (.sum [.predecessor 0 187016 .coefficient, .predecessor 1 187017 .coefficient])

def event187019 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70424⟩⟩) (.sum [.result 187015 .summary, .result 179715 .summary])

def exact187020RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨29338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨35002⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37682⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨40358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨43038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact187020RawTermsValid :
    exact187020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event187020 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70424⟩⟩) exact187020RawTerms .large 187018 (.finite 515053820849391945920019041353728) (some (187019))

def event187021 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70425⟩⟩) 0 ⟨70424⟩ 187020

def event187022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70425⟩⟩) 1 ⟨47427⟩ 179233

def event187023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70425⟩⟩) (.sum [.predecessor 0 187021 .coefficient, .predecessor 1 187022 .coefficient])

def event187024 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70425⟩⟩) (.sum [.result 187020 .summary, .result 179233 .summary])

def exact187025RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨29338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨35002⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37682⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨40358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨43038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact187025RawTermsValid :
    exact187025RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event187025 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70425⟩⟩) exact187025RawTerms .large 187023 (.finite 547248128674354899372274579931136) (some (187024))

def event187026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70426⟩⟩) 0 ⟨70425⟩ 187025

def event187027 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70426⟩⟩) 1 ⟨50107⟩ 178751

def event187028 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70426⟩⟩) (.sum [.predecessor 0 187026 .coefficient, .predecessor 1 187027 .coefficient])

def event187029 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70426⟩⟩) (.sum [.result 187025 .summary, .result 178751 .summary])

def exact187030RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨29338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨35002⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37682⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨40358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨43038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨48402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact187030RawTermsValid :
    exact187030RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event187030 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70426⟩⟩) exact187030RawTerms .large 187028 (.finite 579442632949763540201771008262144) (some (187029))

def event187031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71331⟩⟩) 0 ⟨70426⟩ 187030

def event187032 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71331⟩⟩) 1 ⟨71329⟩ 178253

def event187033 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71331⟩⟩) (.product (.predecessor 0 187031 .coefficient) (.predecessor 1 187032 .coefficient) (⟨false, false, none, none, none⟩))

def event187034 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71331⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) [⟨.result 178253 .coefficient, false, none⟩])

def event187035 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71331⟩⟩) (.product (.result 187030 .summary) (.transfer 187034) (⟨false, false, none, none, none⟩))

def event187036 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 17⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event187037 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 29⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨48402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event187038 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71331⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨48402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 178250)

def event187039 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .relation 187038 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨48402⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event187040 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 16⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event187041 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 28⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event187042 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71331⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 178250)

def event187043 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .relation 187042 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45722⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event187044 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 15⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event187045 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 27⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨43038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event187046 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71331⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨43038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 178250)

def event187047 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .relation 187046 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨43038⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event187048 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 14⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event187049 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 26⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨40358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event187050 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71331⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨40358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 178250)

def event187051 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .relation 187050 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨40358⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event187052 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 13⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event187053 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 25⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37682⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event187054 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71331⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37682⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 178250)

def event187055 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .relation 187054 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37682⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event187056 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 12⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event187057 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 24⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨35002⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event187058 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71331⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨35002⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 178250)

def event187059 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .relation 187058 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨35002⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event187060 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 11⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event187061 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 22⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨29338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event187062 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71331⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨29338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 178250)

def event187063 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .relation 187062 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨29338⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event187064 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 10⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event187065 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 21⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event187066 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71331⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 178250)

def event187067 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .relation 187066 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event187068 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 9⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event187069 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 35⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event187070 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71331⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 178250)

def event187071 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .relation 187070 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event187072 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 8⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event187073 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 34⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event187074 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71331⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 178250)

def event187075 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .relation 187074 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event187076 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 7⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event187077 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 33⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event187078 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71331⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 178250)

def event187079 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .relation 187078 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event187080 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 6⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event187081 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 32⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event187082 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71331⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 178250)

def event187083 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .relation 187082 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event187084 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 5⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event187085 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 31⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event187086 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71331⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 178250)

def event187087 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .relation 187086 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event187088 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 4⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event187089 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 30⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event187090 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71331⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 178250)

def event187091 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .relation 187090 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event187092 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 3⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event187093 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 23⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event187094 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71331⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 178250)

def event187095 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .relation 187094 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event187096 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 2⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event187097 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 20⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event187098 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71331⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 178250)

def event187099 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .relation 187098 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event187100 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 1⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event187101 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 19⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event187102 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71331⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 178250)

def event187103 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .relation 187102 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def event187104 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 0⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩)

def event187105 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .operator (⟨187030, 18⟩, ⟨178253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (-1)⟩)

def event187106 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71331⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 178250)

def event187107 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71331⟩⟩, .relation 187106 0, ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩)

def exact187108RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨16083⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18923⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨29338⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨32163⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨35002⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37682⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨40358⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨43038⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨45722⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨48402⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨51218⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨54198⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57178⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩, (-1)⟩]

theorem exact187108RawTermsValid :
    exact187108RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event187108 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71331⟩⟩) exact187108RawTerms .large 187033 (.finite 6221717896068416040249469304417135687106560) (some (187035))

def event187109 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68400⟩⟩) 0 ⟨66821⟩ 8804

def event187110 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68400⟩⟩) (.authority (.relationPreimageSource ⟨95⟩))

def exact187111RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68400⟩⟩]⟩, (1)⟩]

theorem exact187111RawTermsValid :
    exact187111RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event187111 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68400⟩⟩) exact187111RawTerms (.finite 5647228698) 187110 .exactZero (none)

def event187112 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68402⟩⟩) 0 ⟨68400⟩ 187111

def event187113 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68402⟩⟩) 1 ⟨2370⟩ 4

def event187114 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68402⟩⟩) (.scale (.predecessor 0 187112 .coefficient) (.value (.predecessor 1 187113 .coefficient)))

def exact187115RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68400⟩⟩]⟩, (1)⟩]

theorem exact187115RawTermsValid :
    exact187115RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event187115 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68402⟩⟩) exact187115RawTerms (.finite 5647228698) 187114 .exactZero (none)

def event187116 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68403⟩⟩) 0 ⟨6186⟩ 178370

def event187117 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68403⟩⟩) 1 ⟨68402⟩ 187115

def event187118 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68403⟩⟩) (.product (.predecessor 0 187116 .coefficient) (.predecessor 1 187117 .coefficient) (⟨false, false, none, none, none⟩))

def event187119 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68403⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨68400⟩⟩]⟩) [⟨.result 187111 .coefficient, false, none⟩])

def event187120 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68403⟩⟩) (.product (.result 178370 .summary) (.transfer 187119) (⟨false, false, none, none, none⟩))

def event187121 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68403⟩⟩, .operator (⟨178370, 0⟩, ⟨187115, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68400⟩⟩]⟩, (1)⟩)

def event187122 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨68401⟩⟩)

def event187123 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event187124 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event187125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6170⟩⟩) (.authority (.operator))

def event187126 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6170⟩⟩) (.finite 8)

def event187127 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event187128 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event187129 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event187130 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event187131 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 187130

def event187132 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 187128

def event187133 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 187131 .coefficient) (.value (.predecessor 1 187132 .coefficient)))

def event187134 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event187135 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6172⟩⟩) 0 ⟨392⟩ 187134

def eventLeaf11680 : Array AnnotatedEvent := #[
  { event := event186880
    frameStart := 186828 },
  { event := event186881
    frameStart := 186828 },
  { event := event186882
    frameStart := 186828 },
  { event := event186883
    frameStart := 186828 },
  { event := event186884
    frameStart := 186828 },
  { event := event186885
    frameStart := 186828 },
  { event := event186886
    frameStart := 186828 },
  { event := event186887
    frameStart := 186828 },
  { event := event186888
    frameStart := 186828 },
  { event := event186889
    frameStart := 186828 },
  { event := event186890
    frameStart := 186828 },
  { event := event186891
    frameStart := 186828 },
  { event := event186892
    frameStart := 186828 },
  { event := event186893
    frameStart := 186828 },
  { event := event186894
    frameStart := 186828 },
  { event := event186895
    frameStart := 186828 }
]

def eventLeaf11681 : Array AnnotatedEvent := #[
  { event := event186896
    frameStart := 186828 },
  { event := event186897
    frameStart := 186828 },
  { event := event186898
    frameStart := 186828 },
  { event := event186899
    frameStart := 186828 },
  { event := event186900
    frameStart := 186828 },
  { event := event186901
    frameStart := 186828 },
  { event := event186902
    frameStart := 186828 },
  { event := event186903
    frameStart := 186828 },
  { event := event186904
    frameStart := 186828 },
  { event := event186905
    frameStart := 186828 },
  { event := event186906
    frameStart := 186828 },
  { event := event186907
    frameStart := 186828 },
  { event := event186908
    frameStart := 186828 },
  { event := event186909
    frameStart := 186828 },
  { event := event186910
    frameStart := 186828 },
  { event := event186911
    frameStart := 186828 }
]

def eventLeaf11682 : Array AnnotatedEvent := #[
  { event := event186912
    frameStart := 186828 },
  { event := event186913
    frameStart := 186828 },
  { event := event186914
    frameStart := 186828 },
  { event := event186915
    frameStart := 186828 },
  { event := event186916
    frameStart := 186828 },
  { event := event186917
    frameStart := 186828 },
  { event := event186918
    frameStart := 186828 },
  { event := event186919
    frameStart := 186828 },
  { event := event186920
    frameStart := 186828 },
  { event := event186921
    frameStart := 186828 },
  { event := event186922
    frameStart := 186828 },
  { event := event186923
    frameStart := 186828 },
  { event := event186924
    frameStart := 186828 },
  { event := event186925
    frameStart := 186828 },
  { event := event186926
    frameStart := 186828 },
  { event := event186927
    frameStart := 186828 }
]

def eventLeaf11683 : Array AnnotatedEvent := #[
  { event := event186928
    frameStart := 186828 },
  { event := event186929
    frameStart := 186828 },
  { event := event186930
    frameStart := 186828 },
  { event := event186931
    frameStart := 186828 },
  { event := event186932
    frameStart := 0 },
  { event := event186933
    frameStart := 0 },
  { event := event186934
    frameStart := 0 },
  { event := event186935
    frameStart := 0 },
  { event := event186936
    frameStart := 0 },
  { event := event186937
    frameStart := 0 },
  { event := event186938
    frameStart := 0 },
  { event := event186939
    frameStart := 0 },
  { event := event186940
    frameStart := 0 },
  { event := event186941
    frameStart := 0 },
  { event := event186942
    frameStart := 0 },
  { event := event186943
    frameStart := 0 }
]

def eventLeaf11684 : Array AnnotatedEvent := #[
  { event := event186944
    frameStart := 0 },
  { event := event186945
    frameStart := 0 },
  { event := event186946
    frameStart := 0 },
  { event := event186947
    frameStart := 0 },
  { event := event186948
    frameStart := 0 },
  { event := event186949
    frameStart := 0 },
  { event := event186950
    frameStart := 0 },
  { event := event186951
    frameStart := 0 },
  { event := event186952
    frameStart := 0 },
  { event := event186953
    frameStart := 0 },
  { event := event186954
    frameStart := 0 },
  { event := event186955
    frameStart := 0 },
  { event := event186956
    frameStart := 0 },
  { event := event186957
    frameStart := 0 },
  { event := event186958
    frameStart := 0 },
  { event := event186959
    frameStart := 0 }
]

def eventLeaf11685 : Array AnnotatedEvent := #[
  { event := event186960
    frameStart := 0 },
  { event := event186961
    frameStart := 0 },
  { event := event186962
    frameStart := 0 },
  { event := event186963
    frameStart := 0 },
  { event := event186964
    frameStart := 0 },
  { event := event186965
    frameStart := 0 },
  { event := event186966
    frameStart := 0 },
  { event := event186967
    frameStart := 0 },
  { event := event186968
    frameStart := 0 },
  { event := event186969
    frameStart := 0 },
  { event := event186970
    frameStart := 0 },
  { event := event186971
    frameStart := 0 },
  { event := event186972
    frameStart := 0 },
  { event := event186973
    frameStart := 0 },
  { event := event186974
    frameStart := 0 },
  { event := event186975
    frameStart := 0 }
]

def eventLeaf11686 : Array AnnotatedEvent := #[
  { event := event186976
    frameStart := 0 },
  { event := event186977
    frameStart := 0 },
  { event := event186978
    frameStart := 0 },
  { event := event186979
    frameStart := 0 },
  { event := event186980
    frameStart := 0 },
  { event := event186981
    frameStart := 0 },
  { event := event186982
    frameStart := 0 },
  { event := event186983
    frameStart := 0 },
  { event := event186984
    frameStart := 0 },
  { event := event186985
    frameStart := 0 },
  { event := event186986
    frameStart := 0 },
  { event := event186987
    frameStart := 0 },
  { event := event186988
    frameStart := 0 },
  { event := event186989
    frameStart := 0 },
  { event := event186990
    frameStart := 0 },
  { event := event186991
    frameStart := 0 }
]

def eventLeaf11687 : Array AnnotatedEvent := #[
  { event := event186992
    frameStart := 0 },
  { event := event186993
    frameStart := 0 },
  { event := event186994
    frameStart := 0 },
  { event := event186995
    frameStart := 0 },
  { event := event186996
    frameStart := 0 },
  { event := event186997
    frameStart := 0 },
  { event := event186998
    frameStart := 0 },
  { event := event186999
    frameStart := 0 },
  { event := event187000
    frameStart := 0 },
  { event := event187001
    frameStart := 0 },
  { event := event187002
    frameStart := 0 },
  { event := event187003
    frameStart := 0 },
  { event := event187004
    frameStart := 0 },
  { event := event187005
    frameStart := 0 },
  { event := event187006
    frameStart := 0 },
  { event := event187007
    frameStart := 0 }
]

def eventLeaf11688 : Array AnnotatedEvent := #[
  { event := event187008
    frameStart := 0 },
  { event := event187009
    frameStart := 0 },
  { event := event187010
    frameStart := 0 },
  { event := event187011
    frameStart := 0 },
  { event := event187012
    frameStart := 0 },
  { event := event187013
    frameStart := 0 },
  { event := event187014
    frameStart := 0 },
  { event := event187015
    frameStart := 0 },
  { event := event187016
    frameStart := 0 },
  { event := event187017
    frameStart := 0 },
  { event := event187018
    frameStart := 0 },
  { event := event187019
    frameStart := 0 },
  { event := event187020
    frameStart := 0 },
  { event := event187021
    frameStart := 0 },
  { event := event187022
    frameStart := 0 },
  { event := event187023
    frameStart := 0 }
]

def eventLeaf11689 : Array AnnotatedEvent := #[
  { event := event187024
    frameStart := 0 },
  { event := event187025
    frameStart := 0 },
  { event := event187026
    frameStart := 0 },
  { event := event187027
    frameStart := 0 },
  { event := event187028
    frameStart := 0 },
  { event := event187029
    frameStart := 0 },
  { event := event187030
    frameStart := 0 },
  { event := event187031
    frameStart := 0 },
  { event := event187032
    frameStart := 0 },
  { event := event187033
    frameStart := 0 },
  { event := event187034
    frameStart := 0 },
  { event := event187035
    frameStart := 0 },
  { event := event187036
    frameStart := 0 },
  { event := event187037
    frameStart := 0 },
  { event := event187038
    frameStart := 0 },
  { event := event187039
    frameStart := 0 }
]

def eventLeaf11690 : Array AnnotatedEvent := #[
  { event := event187040
    frameStart := 0 },
  { event := event187041
    frameStart := 0 },
  { event := event187042
    frameStart := 0 },
  { event := event187043
    frameStart := 0 },
  { event := event187044
    frameStart := 0 },
  { event := event187045
    frameStart := 0 },
  { event := event187046
    frameStart := 0 },
  { event := event187047
    frameStart := 0 },
  { event := event187048
    frameStart := 0 },
  { event := event187049
    frameStart := 0 },
  { event := event187050
    frameStart := 0 },
  { event := event187051
    frameStart := 0 },
  { event := event187052
    frameStart := 0 },
  { event := event187053
    frameStart := 0 },
  { event := event187054
    frameStart := 0 },
  { event := event187055
    frameStart := 0 }
]

def eventLeaf11691 : Array AnnotatedEvent := #[
  { event := event187056
    frameStart := 0 },
  { event := event187057
    frameStart := 0 },
  { event := event187058
    frameStart := 0 },
  { event := event187059
    frameStart := 0 },
  { event := event187060
    frameStart := 0 },
  { event := event187061
    frameStart := 0 },
  { event := event187062
    frameStart := 0 },
  { event := event187063
    frameStart := 0 },
  { event := event187064
    frameStart := 0 },
  { event := event187065
    frameStart := 0 },
  { event := event187066
    frameStart := 0 },
  { event := event187067
    frameStart := 0 },
  { event := event187068
    frameStart := 0 },
  { event := event187069
    frameStart := 0 },
  { event := event187070
    frameStart := 0 },
  { event := event187071
    frameStart := 0 }
]

def eventLeaf11692 : Array AnnotatedEvent := #[
  { event := event187072
    frameStart := 0 },
  { event := event187073
    frameStart := 0 },
  { event := event187074
    frameStart := 0 },
  { event := event187075
    frameStart := 0 },
  { event := event187076
    frameStart := 0 },
  { event := event187077
    frameStart := 0 },
  { event := event187078
    frameStart := 0 },
  { event := event187079
    frameStart := 0 },
  { event := event187080
    frameStart := 0 },
  { event := event187081
    frameStart := 0 },
  { event := event187082
    frameStart := 0 },
  { event := event187083
    frameStart := 0 },
  { event := event187084
    frameStart := 0 },
  { event := event187085
    frameStart := 0 },
  { event := event187086
    frameStart := 0 },
  { event := event187087
    frameStart := 0 }
]

def eventLeaf11693 : Array AnnotatedEvent := #[
  { event := event187088
    frameStart := 0 },
  { event := event187089
    frameStart := 0 },
  { event := event187090
    frameStart := 0 },
  { event := event187091
    frameStart := 0 },
  { event := event187092
    frameStart := 0 },
  { event := event187093
    frameStart := 0 },
  { event := event187094
    frameStart := 0 },
  { event := event187095
    frameStart := 0 },
  { event := event187096
    frameStart := 0 },
  { event := event187097
    frameStart := 0 },
  { event := event187098
    frameStart := 0 },
  { event := event187099
    frameStart := 0 },
  { event := event187100
    frameStart := 0 },
  { event := event187101
    frameStart := 0 },
  { event := event187102
    frameStart := 0 },
  { event := event187103
    frameStart := 0 }
]

def eventLeaf11694 : Array AnnotatedEvent := #[
  { event := event187104
    frameStart := 0 },
  { event := event187105
    frameStart := 0 },
  { event := event187106
    frameStart := 0 },
  { event := event187107
    frameStart := 0 },
  { event := event187108
    frameStart := 0 },
  { event := event187109
    frameStart := 0 },
  { event := event187110
    frameStart := 0 },
  { event := event187111
    frameStart := 0 },
  { event := event187112
    frameStart := 0 },
  { event := event187113
    frameStart := 0 },
  { event := event187114
    frameStart := 0 },
  { event := event187115
    frameStart := 0 },
  { event := event187116
    frameStart := 0 },
  { event := event187117
    frameStart := 0 },
  { event := event187118
    frameStart := 0 },
  { event := event187119
    frameStart := 0 }
]

def eventLeaf11695 : Array AnnotatedEvent := #[
  { event := event187120
    frameStart := 0 },
  { event := event187121
    frameStart := 0 },
  { event := event187122
    frameStart := 187122 },
  { event := event187123
    frameStart := 187122 },
  { event := event187124
    frameStart := 187122 },
  { event := event187125
    frameStart := 187122 },
  { event := event187126
    frameStart := 187122 },
  { event := event187127
    frameStart := 187122 },
  { event := event187128
    frameStart := 187122 },
  { event := event187129
    frameStart := 187122 },
  { event := event187130
    frameStart := 187122 },
  { event := event187131
    frameStart := 187122 },
  { event := event187132
    frameStart := 187122 },
  { event := event187133
    frameStart := 187122 },
  { event := event187134
    frameStart := 187122 },
  { event := event187135
    frameStart := 187122 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events730
