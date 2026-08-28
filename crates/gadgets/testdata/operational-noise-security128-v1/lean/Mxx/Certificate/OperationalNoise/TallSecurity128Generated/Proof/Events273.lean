import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events273

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact69888RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], []⟩, (1)⟩]

theorem exact69888RawTermsValid :
    exact69888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69888 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17235⟩⟩) exact69888RawTerms (.finite 2) 69887 .exactZero (none)

def event69889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact69890RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact69890RawTermsValid :
    exact69890RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69890 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact69890RawTerms .large 69889 .exactZero (none)

def event69891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17236⟩⟩) 0 ⟨6908⟩ 69890

def event69892 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17236⟩⟩) 1 ⟨17235⟩ 69888

def event69893 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17236⟩⟩) (.product (.predecessor 0 69891 .coefficient) (.predecessor 1 69892 .coefficient) (⟨false, false, none, none, none⟩))

def event69894 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17236⟩⟩, .operator (⟨69890, 0⟩, ⟨69888, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact69895RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact69895RawTermsValid :
    exact69895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17236⟩⟩) exact69895RawTerms .large 69893 .exactZero (none)

def event69896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 69872

def event69897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact69898RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact69898RawTermsValid :
    exact69898RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69898 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact69898RawTerms .large 69897 .exactZero (none)

def event69899 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17237⟩⟩) 0 ⟨7179⟩ 69898

def event69900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17237⟩⟩) 1 ⟨17236⟩ 69895

def event69901 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17237⟩⟩) (.sum [.predecessor 0 69899 .coefficient, .predecessor 1 69900 .coefficient])

def exact69902RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact69902RawTermsValid :
    exact69902RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69902 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17237⟩⟩) exact69902RawTerms .large 69901 .exactZero (none)

def event69903 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17958⟩⟩) 0 ⟨17237⟩ 69902

def event69904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17958⟩⟩) 1 ⟨17957⟩ 69879

def event69905 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17958⟩⟩) (.product (.predecessor 0 69903 .coefficient) (.predecessor 1 69904 .coefficient) (⟨false, false, none, none, none⟩))

def event69906 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17958⟩⟩, .operator (⟨69902, 0⟩, ⟨69879, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩]⟩, (1)⟩)

def event69907 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17958⟩⟩, .operator (⟨69902, 1⟩, ⟨69879, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩]⟩, (-1)⟩)

def event69908 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17958⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17957⟩⟩) ⟨17064⟩ 69876)

def event69909 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17958⟩⟩, .relation 69908 0, ⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨17064⟩⟩]⟩, (-1)⟩)

def exact69910RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨17064⟩⟩]⟩, (-1)⟩]

theorem exact69910RawTermsValid :
    exact69910RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69910 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17958⟩⟩) exact69910RawTerms .large 69905 .exactZero (none)

def event69911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16147⟩⟩) 0 ⟨15845⟩ 69868

def event69912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16147⟩⟩) (.authority (.programFamilyFact))

def exact69913RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], []⟩, (1)⟩]

theorem exact69913RawTermsValid :
    exact69913RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69913 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16147⟩⟩) exact69913RawTerms (.finite 43) 69912 .exactZero (none)

def event69914 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16148⟩⟩) 0 ⟨6908⟩ 69890

def event69915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16148⟩⟩) 1 ⟨16147⟩ 69913

def event69916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16148⟩⟩) (.product (.predecessor 0 69914 .coefficient) (.predecessor 1 69915 .coefficient) (⟨false, true, none, none, some 1⟩))

def event69917 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16148⟩⟩, .operator (⟨69890, 0⟩, ⟨69913, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact69918RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact69918RawTermsValid :
    exact69918RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69918 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16148⟩⟩) exact69918RawTerms .large 69916 .exactZero (none)

def event69919 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 69872

def event69920 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact69921RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact69921RawTermsValid :
    exact69921RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69921 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact69921RawTerms .large 69920 .exactZero (none)

def event69922 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16149⟩⟩) 0 ⟨7198⟩ 69921

def event69923 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16149⟩⟩) 1 ⟨16148⟩ 69918

def event69924 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16149⟩⟩) (.sum [.predecessor 0 69922 .coefficient, .predecessor 1 69923 .coefficient])

def exact69925RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact69925RawTermsValid :
    exact69925RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69925 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16149⟩⟩) exact69925RawTerms .large 69924 .exactZero (none)

def event69926 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17961⟩⟩) 0 ⟨16149⟩ 69925

def event69927 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17961⟩⟩) 1 ⟨17958⟩ 69910

def event69928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17961⟩⟩) (.sum [.predecessor 0 69926 .coefficient, .predecessor 1 69927 .coefficient])

def exact69929RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨17064⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact69929RawTermsValid :
    exact69929RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69929 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17961⟩⟩) exact69929RawTerms .large 69928 .exactZero (none)

def event69930 : Event := .preFoldPolynomial 69929 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨17064⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact69931RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨17064⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event69931 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17961⟩⟩) 69930 exact69931RawTerms .large 69928 .exactZero (none)

def event69932 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15845⟩⟩) ⟨⟨77⟩, ⟨57⟩, ⟨135⟩⟩ ⟨69774, 69932⟩

def event69933 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16739⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16736⟩⟩]⟩) (1) 0 2 (.universal 69932 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16736⟩⟩]⟩) (none) 69931)

def event69934 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16739⟩⟩, .relation 69933 1, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩)

def event69935 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16739⟩⟩, .relation 69933 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩]⟩, (-1)⟩)

def event69936 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16739⟩⟩, .relation 69933 2, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨17064⟩⟩]⟩, (1)⟩)

def event69937 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16739⟩⟩, .relation 69933 3, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact69938RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨17064⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact69938RawTermsValid :
    exact69938RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69938 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16739⟩⟩) exact69938RawTerms .large 69770 (.finite 202072841853861888) (some (69772))

def event69939 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17960⟩⟩) 0 ⟨16739⟩ 69938

def event69940 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17960⟩⟩) 1 ⟨17959⟩ 69760

def event69941 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17960⟩⟩) (.sum [.predecessor 0 69939 .coefficient, .predecessor 1 69940 .coefficient])

def event69942 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17960⟩⟩, .operator (⟨69938, 0⟩, ⟨69760, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩]⟩, (1)⟩)

def event69943 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17960⟩⟩, .operator (⟨69938, 2⟩, ⟨69760, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨17064⟩⟩]⟩, (-1)⟩)

def event69944 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17960⟩⟩) (.sum [.result 69938 .summary, .result 69760 .summary])

def exact69945RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact69945RawTermsValid :
    exact69945RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69945 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17960⟩⟩) exact69945RawTerms .large 69941 (.finite 32188807212483706889510625476608) (some (69944))

def event69946 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20873⟩⟩) 0 ⟨17960⟩ 69945

def event69947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20873⟩⟩) 1 ⟨20872⟩ 69463

def event69948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20873⟩⟩) (.sum [.predecessor 0 69946 .coefficient, .predecessor 1 69947 .coefficient])

def event69949 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20873⟩⟩) (.sum [.result 69945 .summary, .result 69463 .summary])

def exact69950RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact69950RawTermsValid :
    exact69950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69950 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20873⟩⟩) exact69950RawTerms .large 69948 (.finite 64377712650190257467641695830016) (some (69949))

def event69951 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24093⟩⟩) 0 ⟨20873⟩ 69950

def event69952 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24093⟩⟩) 1 ⟨24092⟩ 68981

def event69953 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24093⟩⟩) (.sum [.predecessor 0 69951 .coefficient, .predecessor 1 69952 .coefficient])

def event69954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24093⟩⟩) (.sum [.result 69950 .summary, .result 68981 .summary])

def exact69955RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact69955RawTermsValid :
    exact69955RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69955 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24093⟩⟩) exact69955RawTerms .large 69953 (.finite 96566716313119651734393211060224) (some (69954))

def event69956 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34113⟩⟩) 0 ⟨24093⟩ 69955

def event69957 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34113⟩⟩) 1 ⟨34112⟩ 68499

def event69958 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34113⟩⟩) (.sum [.predecessor 0 69956 .coefficient, .predecessor 1 69957 .coefficient])

def event69959 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34113⟩⟩) (.sum [.result 69955 .summary, .result 68499 .summary])

def exact69960RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact69960RawTermsValid :
    exact69960RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69960 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34113⟩⟩) exact69960RawTerms .large 69958 (.finite 128755916426494733378385616044032) (some (69959))

def event69961 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53173⟩⟩) 0 ⟨34113⟩ 69960

def event69962 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53173⟩⟩) 1 ⟨53172⟩ 68017

def event69963 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53173⟩⟩) (.sum [.predecessor 0 69961 .coefficient, .predecessor 1 69962 .coefficient])

def event69964 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53173⟩⟩) (.sum [.result 69960 .summary, .result 68017 .summary])

def exact69965RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact69965RawTermsValid :
    exact69965RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69965 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53173⟩⟩) exact69965RawTerms .large 69963 (.finite 160945509440761189776859800535040) (some (69964))

def event69966 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56153⟩⟩) 0 ⟨53173⟩ 69965

def event69967 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56153⟩⟩) 1 ⟨56152⟩ 67535

def event69968 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56153⟩⟩) (.sum [.predecessor 0 69966 .coefficient, .predecessor 1 69967 .coefficient])

def event69969 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56153⟩⟩) (.sum [.result 69965 .summary, .result 67535 .summary])

def exact69970RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact69970RawTermsValid :
    exact69970RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69970 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56153⟩⟩) exact69970RawTerms .large 69968 (.finite 193135298905473333552574874779648) (some (69969))

def event69971 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59133⟩⟩) 0 ⟨56153⟩ 69970

def event69972 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59133⟩⟩) 1 ⟨59132⟩ 67053

def event69973 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59133⟩⟩) (.sum [.predecessor 0 69971 .coefficient, .predecessor 1 69972 .coefficient])

def event69974 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59133⟩⟩) (.sum [.result 69970 .summary, .result 67053 .summary])

def exact69975RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact69975RawTermsValid :
    exact69975RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69975 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59133⟩⟩) exact69975RawTerms .large 69973 (.finite 225325481271076852082771728531456) (some (69974))

def event69976 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62113⟩⟩) 0 ⟨59133⟩ 69975

def event69977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62113⟩⟩) 1 ⟨62112⟩ 66571

def event69978 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62113⟩⟩) (.sum [.predecessor 0 69976 .coefficient, .predecessor 1 69977 .coefficient])

def event69979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62113⟩⟩) (.sum [.result 69975 .summary, .result 66571 .summary])

def exact69980RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact69980RawTermsValid :
    exact69980RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69980 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62113⟩⟩) exact69980RawTerms .large 69978 (.finite 257515860087126057990209472036864) (some (69979))

def event69981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65093⟩⟩) 0 ⟨62113⟩ 69980

def event69982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65093⟩⟩) 1 ⟨65092⟩ 66089

def event69983 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65093⟩⟩) (.sum [.predecessor 0 69981 .coefficient, .predecessor 1 69982 .coefficient])

def event69984 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65093⟩⟩) (.sum [.result 69980 .summary, .result 66089 .summary])

def exact69985RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact69985RawTermsValid :
    exact69985RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69985 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65093⟩⟩) exact69985RawTerms .large 69983 (.finite 289706631804066638652128995049472) (some (69984))

def event69986 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70734⟩⟩) 0 ⟨65093⟩ 69985

def event69987 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70734⟩⟩) 1 ⟨70733⟩ 65607

def event69988 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70734⟩⟩) (.sum [.predecessor 0 69986 .coefficient, .predecessor 1 69987 .coefficient])

def event69989 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70734⟩⟩) (.sum [.result 69985 .summary, .result 65607 .summary])

def exact69990RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact69990RawTermsValid :
    exact69990RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69990 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70734⟩⟩) exact69990RawTerms .large 69988 (.finite 321897992872344281445771187322880) (some (69989))

def event69991 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70735⟩⟩) 0 ⟨70734⟩ 69990

def event69992 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70735⟩⟩) 1 ⟨28467⟩ 65125

def event69993 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70735⟩⟩) (.sum [.predecessor 0 69991 .coefficient, .predecessor 1 69992 .coefficient])

def event69994 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70735⟩⟩) (.sum [.result 69990 .summary, .result 65125 .summary])

def exact69995RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact69995RawTermsValid :
    exact69995RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69995 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70735⟩⟩) exact69995RawTerms .large 69993 (.finite 354089550391067611616654269349888) (some (69994))

def event69996 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70736⟩⟩) 0 ⟨70735⟩ 69995

def event69997 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70736⟩⟩) 1 ⟨31147⟩ 64643

def event69998 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70736⟩⟩) (.sum [.predecessor 0 69996 .coefficient, .predecessor 1 69997 .coefficient])

def event69999 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70736⟩⟩) (.sum [.result 69995 .summary, .result 64643 .summary])

def exact70000RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact70000RawTermsValid :
    exact70000RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70000 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70736⟩⟩) exact70000RawTerms .large 69998 (.finite 386281697261128003919260020637696) (some (69999))

def event70001 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70737⟩⟩) 0 ⟨70736⟩ 70000

def event70002 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70737⟩⟩) 1 ⟨36807⟩ 64161

def event70003 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70737⟩⟩) (.sum [.predecessor 0 70001 .coefficient, .predecessor 1 70002 .coefficient])

def event70004 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70737⟩⟩) (.sum [.result 70000 .summary, .result 64161 .summary])

def exact70005RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact70005RawTermsValid :
    exact70005RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70005 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70737⟩⟩) exact70005RawTerms .large 70003 (.finite 418474237032079770976347551432704) (some (70004))

def event70006 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70738⟩⟩) 0 ⟨70737⟩ 70005

def event70007 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70738⟩⟩) 1 ⟨39487⟩ 63679

def event70008 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70738⟩⟩) (.sum [.predecessor 0 70006 .coefficient, .predecessor 1 70007 .coefficient])

def event70009 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70738⟩⟩) (.sum [.result 70005 .summary, .result 63679 .summary])

def exact70010RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact70010RawTermsValid :
    exact70010RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70010 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70738⟩⟩) exact70010RawTerms .large 70008 (.finite 450666973253477225410675971981312) (some (70009))

def event70011 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70739⟩⟩) 0 ⟨70738⟩ 70010

def event70012 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70739⟩⟩) 1 ⟨42167⟩ 63197

def event70013 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70739⟩⟩) (.sum [.predecessor 0 70011 .coefficient, .predecessor 1 70012 .coefficient])

def event70014 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70739⟩⟩) (.sum [.result 70010 .summary, .result 63197 .summary])

def exact70015RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact70015RawTermsValid :
    exact70015RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70015 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70739⟩⟩) exact70015RawTerms .large 70013 (.finite 482860102375766054599486172037120) (some (70014))

def event70016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70740⟩⟩) 0 ⟨70739⟩ 70015

def event70017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70740⟩⟩) 1 ⟨44847⟩ 62715

def event70018 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70740⟩⟩) (.sum [.predecessor 0 70016 .coefficient, .predecessor 1 70017 .coefficient])

def event70019 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70740⟩⟩) (.sum [.result 70015 .summary, .result 62715 .summary])

def exact70020RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43090⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact70020RawTermsValid :
    exact70020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70020 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70740⟩⟩) exact70020RawTerms .large 70018 (.finite 515053820849391945920019041353728) (some (70019))

def event70021 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70741⟩⟩) 0 ⟨70740⟩ 70020

def event70022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70741⟩⟩) 1 ⟨47527⟩ 62233

def event70023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70741⟩⟩) (.sum [.predecessor 0 70021 .coefficient, .predecessor 1 70022 .coefficient])

def event70024 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70741⟩⟩) (.sum [.result 70020 .summary, .result 62233 .summary])

def exact70025RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43090⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45774⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact70025RawTermsValid :
    exact70025RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70025 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70741⟩⟩) exact70025RawTerms .large 70023 (.finite 547248128674354899372274579931136) (some (70024))

def event70026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70742⟩⟩) 0 ⟨70741⟩ 70025

def event70027 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70742⟩⟩) 1 ⟨50207⟩ 61751

def event70028 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70742⟩⟩) (.sum [.predecessor 0 70026 .coefficient, .predecessor 1 70027 .coefficient])

def event70029 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70742⟩⟩) (.sum [.result 70025 .summary, .result 61751 .summary])

def exact70030RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43090⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45774⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48454⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact70030RawTermsValid :
    exact70030RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70030 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70742⟩⟩) exact70030RawTerms .large 70028 (.finite 579442632949763540201771008262144) (some (70029))

def event70031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71471⟩⟩) 0 ⟨70742⟩ 70030

def event70032 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71471⟩⟩) 1 ⟨71469⟩ 61253

def event70033 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71471⟩⟩) (.product (.predecessor 0 70031 .coefficient) (.predecessor 1 70032 .coefficient) (⟨false, false, none, none, none⟩))

def event70034 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71471⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) [⟨.result 61253 .coefficient, false, none⟩])

def event70035 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71471⟩⟩) (.product (.result 70030 .summary) (.transfer 70034) (⟨false, false, none, none, none⟩))

def event70036 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 17⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event70037 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 29⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48454⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event70038 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71471⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48454⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250)

def event70039 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .relation 70038 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48454⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event70040 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 16⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event70041 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 28⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45774⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event70042 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71471⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45774⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250)

def event70043 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .relation 70042 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45774⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event70044 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 15⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event70045 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 27⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43090⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event70046 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71471⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43090⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250)

def event70047 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .relation 70046 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43090⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event70048 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 14⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event70049 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 26⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event70050 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71471⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250)

def event70051 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .relation 70050 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40410⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event70052 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 13⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event70053 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 25⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event70054 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71471⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250)

def event70055 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .relation 70054 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event70056 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 12⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event70057 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 24⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event70058 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71471⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250)

def event70059 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .relation 70058 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event70060 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 11⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event70061 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 22⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event70062 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71471⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250)

def event70063 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .relation 70062 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event70064 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 10⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event70065 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 21⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event70066 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71471⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250)

def event70067 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .relation 70066 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event70068 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 9⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event70069 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 35⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event70070 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71471⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250)

def event70071 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .relation 70070 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event70072 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 8⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event70073 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 34⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event70074 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71471⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250)

def event70075 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .relation 70074 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event70076 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 7⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event70077 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 33⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event70078 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71471⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250)

def event70079 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .relation 70078 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event70080 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 6⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event70081 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 32⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event70082 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71471⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250)

def event70083 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .relation 70082 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event70084 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 5⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event70085 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 31⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event70086 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71471⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250)

def event70087 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .relation 70086 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event70088 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 4⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event70089 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 30⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event70090 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71471⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250)

def event70091 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .relation 70090 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event70092 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 3⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event70093 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 23⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event70094 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71471⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250)

def event70095 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .relation 70094 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event70096 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 2⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event70097 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 20⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event70098 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71471⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250)

def event70099 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .relation 70098 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event70100 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 1⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event70101 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 19⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event70102 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71471⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250)

def event70103 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .relation 70102 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def event70104 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 0⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩)

def event70105 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .operator (⟨70030, 18⟩, ⟨61253, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (-1)⟩)

def event70106 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨71471⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71469⟩⟩) ⟨68872⟩ 61250)

def event70107 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71471⟩⟩, .relation 70106 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩)

def exact70108RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71469⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨16147⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18999⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨22219⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨26710⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨29390⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨35054⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨40410⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨43090⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45774⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48454⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨54274⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨57254⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨63214⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨67091⟩⟩], [⟨.program ⟨257⟩, ⟨68872⟩⟩]⟩, (-1)⟩]

theorem exact70108RawTermsValid :
    exact70108RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70108 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71471⟩⟩) exact70108RawTerms .large 70033 (.finite 6221717896068416040249469304417135687106560) (some (70035))

def event70109 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68440⟩⟩) 0 ⟨67101⟩ 2820

def event70110 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68440⟩⟩) (.authority (.relationPreimageSource ⟨95⟩))

def exact70111RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68440⟩⟩]⟩, (1)⟩]

theorem exact70111RawTermsValid :
    exact70111RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70111 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68440⟩⟩) exact70111RawTerms (.finite 5647228698) 70110 .exactZero (none)

def event70112 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68442⟩⟩) 0 ⟨68440⟩ 70111

def event70113 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68442⟩⟩) 1 ⟨2370⟩ 4

def event70114 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68442⟩⟩) (.scale (.predecessor 0 70112 .coefficient) (.value (.predecessor 1 70113 .coefficient)))

def exact70115RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68440⟩⟩]⟩, (1)⟩]

theorem exact70115RawTermsValid :
    exact70115RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70115 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68442⟩⟩) exact70115RawTerms (.finite 5647228698) 70114 .exactZero (none)

def event70116 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68443⟩⟩) 0 ⟨10792⟩ 61370

def event70117 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68443⟩⟩) 1 ⟨68442⟩ 70115

def event70118 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68443⟩⟩) (.product (.predecessor 0 70116 .coefficient) (.predecessor 1 70117 .coefficient) (⟨false, false, none, none, none⟩))

def event70119 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68443⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨68440⟩⟩]⟩) [⟨.result 70111 .coefficient, false, none⟩])

def event70120 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68443⟩⟩) (.product (.result 61370 .summary) (.transfer 70119) (⟨false, false, none, none, none⟩))

def event70121 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68443⟩⟩, .operator (⟨61370, 0⟩, ⟨70115, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68440⟩⟩]⟩, (1)⟩)

def event70122 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨68441⟩⟩)

def event70123 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event70124 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event70125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.authority (.operator))

def event70126 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.finite 16)

def event70127 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event70128 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event70129 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event70130 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event70131 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 70130

def event70132 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 70128

def event70133 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 70131 .coefficient) (.value (.predecessor 1 70132 .coefficient)))

def event70134 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event70135 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 0 ⟨392⟩ 70134

def event70136 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 1 ⟨10691⟩ 70126

def event70137 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.sum [.predecessor 0 70135 .coefficient, .predecessor 1 70136 .coefficient])

def event70138 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.finite 655356)

def event70139 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 0 ⟨10693⟩ 70138

def event70140 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 1 ⟨5426⟩ 70124

def event70141 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.identity (.predecessor 1 70140 .coefficient))

def event70142 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.finite 655360)

def event70143 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48002⟩⟩) 0 ⟨10749⟩ 70142

def eventLeaf4368 : Array AnnotatedEvent := #[
  { event := event69888
    frameStart := 69828 },
  { event := event69889
    frameStart := 69828 },
  { event := event69890
    frameStart := 69828 },
  { event := event69891
    frameStart := 69828 },
  { event := event69892
    frameStart := 69828 },
  { event := event69893
    frameStart := 69828 },
  { event := event69894
    frameStart := 69828 },
  { event := event69895
    frameStart := 69828 },
  { event := event69896
    frameStart := 69828 },
  { event := event69897
    frameStart := 69828 },
  { event := event69898
    frameStart := 69828 },
  { event := event69899
    frameStart := 69828 },
  { event := event69900
    frameStart := 69828 },
  { event := event69901
    frameStart := 69828 },
  { event := event69902
    frameStart := 69828 },
  { event := event69903
    frameStart := 69828 }
]

def eventLeaf4369 : Array AnnotatedEvent := #[
  { event := event69904
    frameStart := 69828 },
  { event := event69905
    frameStart := 69828 },
  { event := event69906
    frameStart := 69828 },
  { event := event69907
    frameStart := 69828 },
  { event := event69908
    frameStart := 69828 },
  { event := event69909
    frameStart := 69828 },
  { event := event69910
    frameStart := 69828 },
  { event := event69911
    frameStart := 69828 },
  { event := event69912
    frameStart := 69828 },
  { event := event69913
    frameStart := 69828 },
  { event := event69914
    frameStart := 69828 },
  { event := event69915
    frameStart := 69828 },
  { event := event69916
    frameStart := 69828 },
  { event := event69917
    frameStart := 69828 },
  { event := event69918
    frameStart := 69828 },
  { event := event69919
    frameStart := 69828 }
]

def eventLeaf4370 : Array AnnotatedEvent := #[
  { event := event69920
    frameStart := 69828 },
  { event := event69921
    frameStart := 69828 },
  { event := event69922
    frameStart := 69828 },
  { event := event69923
    frameStart := 69828 },
  { event := event69924
    frameStart := 69828 },
  { event := event69925
    frameStart := 69828 },
  { event := event69926
    frameStart := 69828 },
  { event := event69927
    frameStart := 69828 },
  { event := event69928
    frameStart := 69828 },
  { event := event69929
    frameStart := 69828 },
  { event := event69930
    frameStart := 69828 },
  { event := event69931
    frameStart := 69828 },
  { event := event69932
    frameStart := 0 },
  { event := event69933
    frameStart := 0 },
  { event := event69934
    frameStart := 0 },
  { event := event69935
    frameStart := 0 }
]

def eventLeaf4371 : Array AnnotatedEvent := #[
  { event := event69936
    frameStart := 0 },
  { event := event69937
    frameStart := 0 },
  { event := event69938
    frameStart := 0 },
  { event := event69939
    frameStart := 0 },
  { event := event69940
    frameStart := 0 },
  { event := event69941
    frameStart := 0 },
  { event := event69942
    frameStart := 0 },
  { event := event69943
    frameStart := 0 },
  { event := event69944
    frameStart := 0 },
  { event := event69945
    frameStart := 0 },
  { event := event69946
    frameStart := 0 },
  { event := event69947
    frameStart := 0 },
  { event := event69948
    frameStart := 0 },
  { event := event69949
    frameStart := 0 },
  { event := event69950
    frameStart := 0 },
  { event := event69951
    frameStart := 0 }
]

def eventLeaf4372 : Array AnnotatedEvent := #[
  { event := event69952
    frameStart := 0 },
  { event := event69953
    frameStart := 0 },
  { event := event69954
    frameStart := 0 },
  { event := event69955
    frameStart := 0 },
  { event := event69956
    frameStart := 0 },
  { event := event69957
    frameStart := 0 },
  { event := event69958
    frameStart := 0 },
  { event := event69959
    frameStart := 0 },
  { event := event69960
    frameStart := 0 },
  { event := event69961
    frameStart := 0 },
  { event := event69962
    frameStart := 0 },
  { event := event69963
    frameStart := 0 },
  { event := event69964
    frameStart := 0 },
  { event := event69965
    frameStart := 0 },
  { event := event69966
    frameStart := 0 },
  { event := event69967
    frameStart := 0 }
]

def eventLeaf4373 : Array AnnotatedEvent := #[
  { event := event69968
    frameStart := 0 },
  { event := event69969
    frameStart := 0 },
  { event := event69970
    frameStart := 0 },
  { event := event69971
    frameStart := 0 },
  { event := event69972
    frameStart := 0 },
  { event := event69973
    frameStart := 0 },
  { event := event69974
    frameStart := 0 },
  { event := event69975
    frameStart := 0 },
  { event := event69976
    frameStart := 0 },
  { event := event69977
    frameStart := 0 },
  { event := event69978
    frameStart := 0 },
  { event := event69979
    frameStart := 0 },
  { event := event69980
    frameStart := 0 },
  { event := event69981
    frameStart := 0 },
  { event := event69982
    frameStart := 0 },
  { event := event69983
    frameStart := 0 }
]

def eventLeaf4374 : Array AnnotatedEvent := #[
  { event := event69984
    frameStart := 0 },
  { event := event69985
    frameStart := 0 },
  { event := event69986
    frameStart := 0 },
  { event := event69987
    frameStart := 0 },
  { event := event69988
    frameStart := 0 },
  { event := event69989
    frameStart := 0 },
  { event := event69990
    frameStart := 0 },
  { event := event69991
    frameStart := 0 },
  { event := event69992
    frameStart := 0 },
  { event := event69993
    frameStart := 0 },
  { event := event69994
    frameStart := 0 },
  { event := event69995
    frameStart := 0 },
  { event := event69996
    frameStart := 0 },
  { event := event69997
    frameStart := 0 },
  { event := event69998
    frameStart := 0 },
  { event := event69999
    frameStart := 0 }
]

def eventLeaf4375 : Array AnnotatedEvent := #[
  { event := event70000
    frameStart := 0 },
  { event := event70001
    frameStart := 0 },
  { event := event70002
    frameStart := 0 },
  { event := event70003
    frameStart := 0 },
  { event := event70004
    frameStart := 0 },
  { event := event70005
    frameStart := 0 },
  { event := event70006
    frameStart := 0 },
  { event := event70007
    frameStart := 0 },
  { event := event70008
    frameStart := 0 },
  { event := event70009
    frameStart := 0 },
  { event := event70010
    frameStart := 0 },
  { event := event70011
    frameStart := 0 },
  { event := event70012
    frameStart := 0 },
  { event := event70013
    frameStart := 0 },
  { event := event70014
    frameStart := 0 },
  { event := event70015
    frameStart := 0 }
]

def eventLeaf4376 : Array AnnotatedEvent := #[
  { event := event70016
    frameStart := 0 },
  { event := event70017
    frameStart := 0 },
  { event := event70018
    frameStart := 0 },
  { event := event70019
    frameStart := 0 },
  { event := event70020
    frameStart := 0 },
  { event := event70021
    frameStart := 0 },
  { event := event70022
    frameStart := 0 },
  { event := event70023
    frameStart := 0 },
  { event := event70024
    frameStart := 0 },
  { event := event70025
    frameStart := 0 },
  { event := event70026
    frameStart := 0 },
  { event := event70027
    frameStart := 0 },
  { event := event70028
    frameStart := 0 },
  { event := event70029
    frameStart := 0 },
  { event := event70030
    frameStart := 0 },
  { event := event70031
    frameStart := 0 }
]

def eventLeaf4377 : Array AnnotatedEvent := #[
  { event := event70032
    frameStart := 0 },
  { event := event70033
    frameStart := 0 },
  { event := event70034
    frameStart := 0 },
  { event := event70035
    frameStart := 0 },
  { event := event70036
    frameStart := 0 },
  { event := event70037
    frameStart := 0 },
  { event := event70038
    frameStart := 0 },
  { event := event70039
    frameStart := 0 },
  { event := event70040
    frameStart := 0 },
  { event := event70041
    frameStart := 0 },
  { event := event70042
    frameStart := 0 },
  { event := event70043
    frameStart := 0 },
  { event := event70044
    frameStart := 0 },
  { event := event70045
    frameStart := 0 },
  { event := event70046
    frameStart := 0 },
  { event := event70047
    frameStart := 0 }
]

def eventLeaf4378 : Array AnnotatedEvent := #[
  { event := event70048
    frameStart := 0 },
  { event := event70049
    frameStart := 0 },
  { event := event70050
    frameStart := 0 },
  { event := event70051
    frameStart := 0 },
  { event := event70052
    frameStart := 0 },
  { event := event70053
    frameStart := 0 },
  { event := event70054
    frameStart := 0 },
  { event := event70055
    frameStart := 0 },
  { event := event70056
    frameStart := 0 },
  { event := event70057
    frameStart := 0 },
  { event := event70058
    frameStart := 0 },
  { event := event70059
    frameStart := 0 },
  { event := event70060
    frameStart := 0 },
  { event := event70061
    frameStart := 0 },
  { event := event70062
    frameStart := 0 },
  { event := event70063
    frameStart := 0 }
]

def eventLeaf4379 : Array AnnotatedEvent := #[
  { event := event70064
    frameStart := 0 },
  { event := event70065
    frameStart := 0 },
  { event := event70066
    frameStart := 0 },
  { event := event70067
    frameStart := 0 },
  { event := event70068
    frameStart := 0 },
  { event := event70069
    frameStart := 0 },
  { event := event70070
    frameStart := 0 },
  { event := event70071
    frameStart := 0 },
  { event := event70072
    frameStart := 0 },
  { event := event70073
    frameStart := 0 },
  { event := event70074
    frameStart := 0 },
  { event := event70075
    frameStart := 0 },
  { event := event70076
    frameStart := 0 },
  { event := event70077
    frameStart := 0 },
  { event := event70078
    frameStart := 0 },
  { event := event70079
    frameStart := 0 }
]

def eventLeaf4380 : Array AnnotatedEvent := #[
  { event := event70080
    frameStart := 0 },
  { event := event70081
    frameStart := 0 },
  { event := event70082
    frameStart := 0 },
  { event := event70083
    frameStart := 0 },
  { event := event70084
    frameStart := 0 },
  { event := event70085
    frameStart := 0 },
  { event := event70086
    frameStart := 0 },
  { event := event70087
    frameStart := 0 },
  { event := event70088
    frameStart := 0 },
  { event := event70089
    frameStart := 0 },
  { event := event70090
    frameStart := 0 },
  { event := event70091
    frameStart := 0 },
  { event := event70092
    frameStart := 0 },
  { event := event70093
    frameStart := 0 },
  { event := event70094
    frameStart := 0 },
  { event := event70095
    frameStart := 0 }
]

def eventLeaf4381 : Array AnnotatedEvent := #[
  { event := event70096
    frameStart := 0 },
  { event := event70097
    frameStart := 0 },
  { event := event70098
    frameStart := 0 },
  { event := event70099
    frameStart := 0 },
  { event := event70100
    frameStart := 0 },
  { event := event70101
    frameStart := 0 },
  { event := event70102
    frameStart := 0 },
  { event := event70103
    frameStart := 0 },
  { event := event70104
    frameStart := 0 },
  { event := event70105
    frameStart := 0 },
  { event := event70106
    frameStart := 0 },
  { event := event70107
    frameStart := 0 },
  { event := event70108
    frameStart := 0 },
  { event := event70109
    frameStart := 0 },
  { event := event70110
    frameStart := 0 },
  { event := event70111
    frameStart := 0 }
]

def eventLeaf4382 : Array AnnotatedEvent := #[
  { event := event70112
    frameStart := 0 },
  { event := event70113
    frameStart := 0 },
  { event := event70114
    frameStart := 0 },
  { event := event70115
    frameStart := 0 },
  { event := event70116
    frameStart := 0 },
  { event := event70117
    frameStart := 0 },
  { event := event70118
    frameStart := 0 },
  { event := event70119
    frameStart := 0 },
  { event := event70120
    frameStart := 0 },
  { event := event70121
    frameStart := 0 },
  { event := event70122
    frameStart := 70122 },
  { event := event70123
    frameStart := 70122 },
  { event := event70124
    frameStart := 70122 },
  { event := event70125
    frameStart := 70122 },
  { event := event70126
    frameStart := 70122 },
  { event := event70127
    frameStart := 70122 }
]

def eventLeaf4383 : Array AnnotatedEvent := #[
  { event := event70128
    frameStart := 70122 },
  { event := event70129
    frameStart := 70122 },
  { event := event70130
    frameStart := 70122 },
  { event := event70131
    frameStart := 70122 },
  { event := event70132
    frameStart := 70122 },
  { event := event70133
    frameStart := 70122 },
  { event := event70134
    frameStart := 70122 },
  { event := event70135
    frameStart := 70122 },
  { event := event70136
    frameStart := 70122 },
  { event := event70137
    frameStart := 70122 },
  { event := event70138
    frameStart := 70122 },
  { event := event70139
    frameStart := 70122 },
  { event := event70140
    frameStart := 70122 },
  { event := event70141
    frameStart := 70122 },
  { event := event70142
    frameStart := 70122 },
  { event := event70143
    frameStart := 70122 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events273
