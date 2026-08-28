import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events105

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event26880 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 26878 .coefficient) (.value (.predecessor 1 26879 .coefficient)))

def event26881 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event26882 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 26881

def event26883 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 26873

def event26884 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 26882 .coefficient, .predecessor 1 26883 .coefficient])

def event26885 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event26886 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 26885

def event26887 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 26871

def event26888 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 26887 .coefficient))

def event26889 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event26890 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11397⟩⟩) 0 ⟨5554⟩ 26889

def event26891 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11397⟩⟩) (.authority (.programFamilyFact))

def exact26892RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩], []⟩, (1)⟩]

theorem exact26892RawTermsValid :
    exact26892RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26892 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11397⟩⟩) exact26892RawTerms (.finite 16) 26891 .exactZero (none)

def event26893 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14017⟩⟩) 0 ⟨5554⟩ 26889

def event26894 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14017⟩⟩) (.authority (.programFamilyFact))

def exact26895RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14017⟩⟩], []⟩, (1)⟩]

theorem exact26895RawTermsValid :
    exact26895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26895 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14017⟩⟩) exact26895RawTerms (.finite 16) 26894 .exactZero (none)

def event26896 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14018⟩⟩) 0 ⟨14017⟩ 26895

def event26897 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14018⟩⟩) 1 ⟨11397⟩ 26892

def event26898 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14018⟩⟩) (.product (.predecessor 0 26896 .coefficient) (.predecessor 1 26897 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event26899 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14018⟩⟩, .operator (⟨26895, 0⟩, ⟨26892, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], []⟩, (1)⟩)

def exact26900RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], []⟩, (1)⟩]

theorem exact26900RawTermsValid :
    exact26900RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26900 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14018⟩⟩) exact26900RawTerms (.finite 256) 26898 .exactZero (none)

def event26901 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14019⟩⟩) 0 ⟨14018⟩ 26900

def event26902 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14019⟩⟩) (.identity (.predecessor 0 26901 .coefficient))

def event26903 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14019⟩⟩) (.finite 256)

def event26904 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23547⟩⟩) 0 ⟨14019⟩ 26903

def event26905 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23547⟩⟩) (.authority (.programFamilyFact))

def event26906 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23547⟩⟩) (.finite 3720)

def event26907 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event26908 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23548⟩⟩) 0 ⟨6689⟩ 26907

def event26909 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23548⟩⟩) 1 ⟨23547⟩ 26906

def event26910 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23548⟩⟩) (.authority (.operator))

def exact26911RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23548⟩⟩]⟩, (1)⟩]

theorem exact26911RawTermsValid :
    exact26911RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26911 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23548⟩⟩) exact26911RawTerms .large 26910 .exactZero (none)

def event26912 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26004⟩⟩) 0 ⟨23548⟩ 26911

def event26913 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26004⟩⟩) (.authority (.operator))

def exact26914RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26004⟩⟩]⟩, (1)⟩]

theorem exact26914RawTermsValid :
    exact26914RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26914 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26004⟩⟩) exact26914RawTerms (.finite 8192) 26913 .exactZero (none)

def event26915 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event26916 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event26917 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14109⟩⟩) 0 ⟨14019⟩ 26903

def event26918 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14109⟩⟩) 1 ⟨110⟩ 26916

def event26919 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14109⟩⟩) (.sum [.predecessor 0 26917 .coefficient, .predecessor 1 26918 .coefficient])

def event26920 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14109⟩⟩) (.finite 256)

def event26921 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14110⟩⟩) 0 ⟨14109⟩ 26920

def event26922 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14110⟩⟩) (.identity (.predecessor 0 26921 .coefficient))

def exact26923RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], []⟩, (1)⟩]

theorem exact26923RawTermsValid :
    exact26923RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26923 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14110⟩⟩) exact26923RawTerms (.finite 256) 26922 .exactZero (none)

def event26924 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact26925RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact26925RawTermsValid :
    exact26925RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26925 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact26925RawTerms .large 26924 .exactZero (none)

def event26926 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14111⟩⟩) 0 ⟨6544⟩ 26925

def event26927 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14111⟩⟩) 1 ⟨14110⟩ 26923

def event26928 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14111⟩⟩) (.product (.predecessor 0 26926 .coefficient) (.predecessor 1 26927 .coefficient) (⟨false, false, none, none, none⟩))

def event26929 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14111⟩⟩, .operator (⟨26925, 0⟩, ⟨26923, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact26930RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact26930RawTermsValid :
    exact26930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26930 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14111⟩⟩) exact26930RawTerms .large 26928 .exactZero (none)

def event26931 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event26932 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event26933 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 26907

def event26934 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact26935RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact26935RawTermsValid :
    exact26935RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26935 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact26935RawTerms .large 26934 .exactZero (none)

def event26936 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6778⟩⟩) 0 ⟨6757⟩ 26935

def event26937 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6778⟩⟩) (.identity (.predecessor 0 26936 .coefficient))

def exact26938RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩]

theorem exact26938RawTermsValid :
    exact26938RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26938 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6778⟩⟩) exact26938RawTerms .large 26937 .exactZero (none)

def event26939 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7849⟩⟩) 0 ⟨6778⟩ 26938

def event26940 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7849⟩⟩) (.authority (.operator))

def exact26941RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩]

theorem exact26941RawTermsValid :
    exact26941RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26941 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7849⟩⟩) exact26941RawTerms (.finite 8192) 26940 .exactZero (none)

def event26942 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7850⟩⟩) 0 ⟨7849⟩ 26941

def event26943 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7850⟩⟩) 1 ⟨2348⟩ 26932

def event26944 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7850⟩⟩) (.scale (.predecessor 0 26942 .coefficient) (.value (.predecessor 1 26943 .coefficient)))

def exact26945RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩]

theorem exact26945RawTermsValid :
    exact26945RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26945 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7850⟩⟩) exact26945RawTerms (.finite 8192) 26944 .exactZero (none)

def event26946 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6758⟩⟩) 0 ⟨6757⟩ 26935

def event26947 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6758⟩⟩) (.identity (.predecessor 0 26946 .coefficient))

def exact26948RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩]⟩, (1)⟩]

theorem exact26948RawTermsValid :
    exact26948RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26948 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6758⟩⟩) exact26948RawTerms .large 26947 .exactZero (none)

def event26949 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7851⟩⟩) 0 ⟨6758⟩ 26948

def event26950 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7851⟩⟩) 1 ⟨7850⟩ 26945

def event26951 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7851⟩⟩) (.product (.predecessor 0 26949 .coefficient) (.predecessor 1 26950 .coefficient) (⟨false, false, none, none, none⟩))

def event26952 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7851⟩⟩, .operator (⟨26948, 0⟩, ⟨26945, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩)

def exact26953RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩]

theorem exact26953RawTermsValid :
    exact26953RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26953 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7851⟩⟩) exact26953RawTerms .large 26951 .exactZero (none)

def event26954 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14112⟩⟩) 0 ⟨7851⟩ 26953

def event26955 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14112⟩⟩) 1 ⟨14111⟩ 26930

def event26956 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14112⟩⟩) (.sum [.predecessor 0 26954 .coefficient, .predecessor 1 26955 .coefficient])

def exact26957RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26957RawTermsValid :
    exact26957RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26957 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14112⟩⟩) exact26957RawTerms .large 26956 .exactZero (none)

def event26958 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26007⟩⟩) 0 ⟨14112⟩ 26957

def event26959 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26007⟩⟩) 1 ⟨26004⟩ 26914

def event26960 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26007⟩⟩) (.product (.predecessor 0 26958 .coefficient) (.predecessor 1 26959 .coefficient) (⟨false, false, none, none, none⟩))

def event26961 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26007⟩⟩, .operator (⟨26957, 0⟩, ⟨26914, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨26004⟩⟩]⟩, (1)⟩)

def event26962 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26007⟩⟩, .operator (⟨26957, 1⟩, ⟨26914, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26004⟩⟩]⟩, (-1)⟩)

def event26963 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26007⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26004⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26004⟩⟩) ⟨23548⟩ 26911)

def event26964 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26007⟩⟩, .relation 26963 0, ⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨23548⟩⟩]⟩, (-1)⟩)

def exact26965RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨26004⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨23548⟩⟩]⟩, (-1)⟩]

theorem exact26965RawTermsValid :
    exact26965RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26965 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26007⟩⟩) exact26965RawTerms .large 26960 .exactZero (none)

def event26966 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15833⟩⟩) 0 ⟨14019⟩ 26903

def event26967 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15833⟩⟩) (.authority (.programFamilyFact))

def exact26968RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], []⟩, (1)⟩]

theorem exact26968RawTermsValid :
    exact26968RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26968 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15833⟩⟩) exact26968RawTerms (.finite 16) 26967 .exactZero (none)

def event26969 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15835⟩⟩) 0 ⟨6544⟩ 26925

def event26970 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15835⟩⟩) 1 ⟨15833⟩ 26968

def event26971 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15835⟩⟩) (.product (.predecessor 0 26969 .coefficient) (.predecessor 1 26970 .coefficient) (⟨false, true, none, none, some 1⟩))

def event26972 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15835⟩⟩, .operator (⟨26925, 0⟩, ⟨26968, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact26973RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact26973RawTermsValid :
    exact26973RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26973 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15835⟩⟩) exact26973RawTerms .large 26971 .exactZero (none)

def event26974 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6696⟩⟩) 0 ⟨6689⟩ 26907

def event26975 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6696⟩⟩) (.authority (.operator))

def exact26976RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩]

theorem exact26976RawTermsValid :
    exact26976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26976 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6696⟩⟩) exact26976RawTerms .large 26975 .exactZero (none)

def event26977 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15836⟩⟩) 0 ⟨6696⟩ 26976

def event26978 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15836⟩⟩) 1 ⟨15835⟩ 26973

def event26979 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15836⟩⟩) (.sum [.predecessor 0 26977 .coefficient, .predecessor 1 26978 .coefficient])

def exact26980RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26980RawTermsValid :
    exact26980RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26980 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15836⟩⟩) exact26980RawTerms .large 26979 .exactZero (none)

def event26981 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26008⟩⟩) 0 ⟨15836⟩ 26980

def event26982 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26008⟩⟩) 1 ⟨26007⟩ 26965

def event26983 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26008⟩⟩) (.sum [.predecessor 0 26981 .coefficient, .predecessor 1 26982 .coefficient])

def exact26984RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨26004⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨23548⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26984RawTermsValid :
    exact26984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26984 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26008⟩⟩) exact26984RawTerms .large 26983 .exactZero (none)

def event26985 : Event := .preFoldPolynomial 26984 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨26004⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨23548⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact26986RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨26004⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨23548⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event26986 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26008⟩⟩) 26985 exact26986RawTerms .large 26983 .exactZero (none)

def event26987 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14019⟩⟩) ⟨⟨109⟩, ⟨14⟩, ⟨109⟩⟩ ⟨26821, 26987⟩

def event26988 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19471⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19468⟩⟩]⟩) (1) 0 2 (.universal 26987 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19468⟩⟩]⟩) (none) 26986)

def event26989 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19471⟩⟩, .relation 26988 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩)

def event26990 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19471⟩⟩, .relation 26988 1, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨26004⟩⟩]⟩, (-1)⟩)

def event26991 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19471⟩⟩, .relation 26988 2, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨23548⟩⟩]⟩, (1)⟩)

def event26992 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19471⟩⟩, .relation 26988 3, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact26993RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨26004⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨23548⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact26993RawTermsValid :
    exact26993RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event26993 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19471⟩⟩) exact26993RawTerms .large 26817 (.finite 1811303510016) (some (26819))

def event26994 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26006⟩⟩) 0 ⟨19471⟩ 26993

def event26995 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26006⟩⟩) 1 ⟨26005⟩ 26807

def event26996 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26006⟩⟩) (.sum [.predecessor 0 26994 .coefficient, .predecessor 1 26995 .coefficient])

def event26997 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26006⟩⟩, .operator (⟨26993, 2⟩, ⟨26807, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], [⟨.program ⟨214⟩, ⟨23548⟩⟩]⟩, (-1)⟩)

def event26998 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26006⟩⟩, .operator (⟨26993, 1⟩, ⟨26807, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨26004⟩⟩]⟩, (1)⟩)

def event26999 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26006⟩⟩) (.sum [.result 26993 .summary, .result 26807 .summary])

def exact27000RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27000RawTermsValid :
    exact27000RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27000 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26006⟩⟩) exact27000RawTerms .large 26996 (.finite 352054612209664) (some (26999))

def event27001 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27690⟩⟩) 0 ⟨26006⟩ 27000

def event27002 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27690⟩⟩) 1 ⟨27688⟩ 26723

def event27003 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27690⟩⟩) (.product (.predecessor 0 27001 .coefficient) (.predecessor 1 27002 .coefficient) (⟨false, false, none, none, none⟩))

def event27004 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27690⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩) [⟨.result 26723 .coefficient, false, none⟩])

def event27005 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27690⟩⟩) (.product (.result 27000 .summary) (.transfer 27004) (⟨false, false, none, none, none⟩))

def event27006 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27690⟩⟩, .operator (⟨27000, 0⟩, ⟨26723, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩, (1)⟩)

def event27007 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27690⟩⟩, .operator (⟨27000, 1⟩, ⟨26723, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩, (-1)⟩)

def event27008 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27690⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27688⟩⟩) ⟨24108⟩ 26720)

def event27009 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27690⟩⟩, .relation 27008 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨24108⟩⟩]⟩, (-1)⟩)

def exact27010RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨24108⟩⟩]⟩, (-1)⟩]

theorem exact27010RawTermsValid :
    exact27010RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27010 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27690⟩⟩) exact27010RawTerms .large 27003 (.finite 1292046059683262234624) (some (27005))

def event27011 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21268⟩⟩) 0 ⟨15834⟩ 1112

def event27012 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21268⟩⟩) (.authority (.relationPreimageSource ⟨41⟩))

def exact27013RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩, (1)⟩]

theorem exact27013RawTermsValid :
    exact27013RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27013 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21268⟩⟩) exact27013RawTerms (.finite 136065468) 27012 .exactZero (none)

def event27014 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21270⟩⟩) 0 ⟨21268⟩ 27013

def event27015 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21270⟩⟩) 1 ⟨2348⟩ 4

def event27016 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21270⟩⟩) (.scale (.predecessor 0 27014 .coefficient) (.value (.predecessor 1 27015 .coefficient)))

def exact27017RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩, (1)⟩]

theorem exact27017RawTermsValid :
    exact27017RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27017 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21270⟩⟩) exact27017RawTerms (.finite 136065468) 27016 .exactZero (none)

def event27018 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21271⟩⟩) 0 ⟨5559⟩ 21512

def event27019 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21271⟩⟩) 1 ⟨21270⟩ 27017

def event27020 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21271⟩⟩) (.product (.predecessor 0 27018 .coefficient) (.predecessor 1 27019 .coefficient) (⟨false, false, none, none, none⟩))

def event27021 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21271⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩) [⟨.result 27013 .coefficient, false, none⟩])

def event27022 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21271⟩⟩) (.product (.result 21512 .summary) (.transfer 27021) (⟨false, false, none, none, none⟩))

def event27023 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21271⟩⟩, .operator (⟨21512, 0⟩, ⟨27017, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩, (1)⟩)

def event27024 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21269⟩⟩)

def event27025 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event27026 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event27027 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event27028 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event27029 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event27030 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event27031 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event27032 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event27033 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 27032

def event27034 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 27030

def event27035 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 27033 .coefficient) (.value (.predecessor 1 27034 .coefficient)))

def event27036 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event27037 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 27036

def event27038 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 27028

def event27039 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 27037 .coefficient, .predecessor 1 27038 .coefficient])

def event27040 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event27041 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 27040

def event27042 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 27026

def event27043 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 27042 .coefficient))

def event27044 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event27045 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11397⟩⟩) 0 ⟨5554⟩ 27044

def event27046 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11397⟩⟩) (.authority (.programFamilyFact))

def exact27047RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩], []⟩, (1)⟩]

theorem exact27047RawTermsValid :
    exact27047RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27047 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11397⟩⟩) exact27047RawTerms (.finite 16) 27046 .exactZero (none)

def event27048 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14017⟩⟩) 0 ⟨5554⟩ 27044

def event27049 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14017⟩⟩) (.authority (.programFamilyFact))

def exact27050RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14017⟩⟩], []⟩, (1)⟩]

theorem exact27050RawTermsValid :
    exact27050RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27050 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14017⟩⟩) exact27050RawTerms (.finite 16) 27049 .exactZero (none)

def event27051 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14018⟩⟩) 0 ⟨14017⟩ 27050

def event27052 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14018⟩⟩) 1 ⟨11397⟩ 27047

def event27053 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14018⟩⟩) (.product (.predecessor 0 27051 .coefficient) (.predecessor 1 27052 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event27054 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14018⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], []⟩) [⟨.result 27050 .coefficient, true, some 1⟩, ⟨.result 27047 .coefficient, true, some 1⟩])

def event27055 : Event := .survivorFold (1) 27054

def exact27056RawTerms : List Term := []

theorem exact27056RawTermsValid :
    exact27056RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27056 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14018⟩⟩) exact27056RawTerms (.finite 256) 27053 (.finite 256) (some (27054))

def event27057 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14019⟩⟩) 0 ⟨14018⟩ 27056

def event27058 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14019⟩⟩) (.identity (.predecessor 0 27057 .coefficient))

def event27059 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14019⟩⟩) (.finite 256)

def event27060 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15833⟩⟩) 0 ⟨14019⟩ 27059

def event27061 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15833⟩⟩) (.authority (.programFamilyFact))

def exact27062RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], []⟩, (1)⟩]

theorem exact27062RawTermsValid :
    exact27062RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27062 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15833⟩⟩) exact27062RawTerms (.finite 16) 27061 .exactZero (none)

def event27063 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15834⟩⟩) 0 ⟨15833⟩ 27062

def event27064 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15834⟩⟩) (.identity (.predecessor 0 27063 .coefficient))

def event27065 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15834⟩⟩) (.finite 16)

def event27066 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21268⟩⟩) 0 ⟨15834⟩ 27065

def event27067 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21268⟩⟩) (.authority (.relationPreimageSource ⟨41⟩))

def exact27068RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩, (1)⟩]

theorem exact27068RawTermsValid :
    exact27068RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27068 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21268⟩⟩) exact27068RawTerms (.finite 136065468) 27067 .exactZero (none)

def event27069 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact27070RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact27070RawTermsValid :
    exact27070RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27070 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact27070RawTerms .large 27069 .exactZero (none)

def event27071 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21269⟩⟩) 0 ⟨6⟩ 27070

def event27072 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21269⟩⟩) 1 ⟨21268⟩ 27068

def event27073 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21269⟩⟩) (.product (.predecessor 0 27071 .coefficient) (.predecessor 1 27072 .coefficient) (⟨false, false, none, none, none⟩))

def event27074 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21269⟩⟩, .operator (⟨27070, 0⟩, ⟨27068, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩, (1)⟩)

def exact27075RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩, (1)⟩]

theorem exact27075RawTermsValid :
    exact27075RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27075 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21269⟩⟩) exact27075RawTerms .large 27073 .exactZero (none)

def event27076 : Event := .preFoldPolynomial 27075 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩, (1)⟩] .exactZero none

def exact27077RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩, (1)⟩]

def event27077 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21269⟩⟩) 27076 exact27077RawTerms .large 27073 .exactZero (none)

def event27078 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27693⟩⟩)

def event27079 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event27080 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event27081 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event27082 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event27083 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event27084 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event27085 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event27086 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event27087 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 27086

def event27088 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 27084

def event27089 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 27087 .coefficient) (.value (.predecessor 1 27088 .coefficient)))

def event27090 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event27091 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 27090

def event27092 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 27082

def event27093 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 27091 .coefficient, .predecessor 1 27092 .coefficient])

def event27094 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event27095 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 27094

def event27096 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 27080

def event27097 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 27096 .coefficient))

def event27098 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event27099 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11397⟩⟩) 0 ⟨5554⟩ 27098

def event27100 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11397⟩⟩) (.authority (.programFamilyFact))

def exact27101RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩], []⟩, (1)⟩]

theorem exact27101RawTermsValid :
    exact27101RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27101 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11397⟩⟩) exact27101RawTerms (.finite 16) 27100 .exactZero (none)

def event27102 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14017⟩⟩) 0 ⟨5554⟩ 27098

def event27103 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14017⟩⟩) (.authority (.programFamilyFact))

def exact27104RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14017⟩⟩], []⟩, (1)⟩]

theorem exact27104RawTermsValid :
    exact27104RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27104 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14017⟩⟩) exact27104RawTerms (.finite 16) 27103 .exactZero (none)

def event27105 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14018⟩⟩) 0 ⟨14017⟩ 27104

def event27106 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14018⟩⟩) 1 ⟨11397⟩ 27101

def event27107 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14018⟩⟩) (.product (.predecessor 0 27105 .coefficient) (.predecessor 1 27106 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event27108 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14018⟩⟩, .operator (⟨27104, 0⟩, ⟨27101, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], []⟩, (1)⟩)

def exact27109RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11397⟩⟩, ⟨.program ⟨214⟩, ⟨14017⟩⟩], []⟩, (1)⟩]

theorem exact27109RawTermsValid :
    exact27109RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27109 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14018⟩⟩) exact27109RawTerms (.finite 256) 27107 .exactZero (none)

def event27110 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14019⟩⟩) 0 ⟨14018⟩ 27109

def event27111 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14019⟩⟩) (.identity (.predecessor 0 27110 .coefficient))

def event27112 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14019⟩⟩) (.finite 256)

def event27113 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15833⟩⟩) 0 ⟨14019⟩ 27112

def event27114 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15833⟩⟩) (.authority (.programFamilyFact))

def exact27115RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], []⟩, (1)⟩]

theorem exact27115RawTermsValid :
    exact27115RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27115 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15833⟩⟩) exact27115RawTerms (.finite 16) 27114 .exactZero (none)

def event27116 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15834⟩⟩) 0 ⟨15833⟩ 27115

def event27117 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15834⟩⟩) (.identity (.predecessor 0 27116 .coefficient))

def event27118 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15834⟩⟩) (.finite 16)

def event27119 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24106⟩⟩) 0 ⟨15834⟩ 27118

def event27120 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24106⟩⟩) (.authority (.programFamilyFact))

def event27121 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24106⟩⟩) (.finite 3720)

def event27122 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event27123 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24108⟩⟩) 0 ⟨6689⟩ 27122

def event27124 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24108⟩⟩) 1 ⟨24106⟩ 27121

def event27125 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24108⟩⟩) (.authority (.operator))

def exact27126RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24108⟩⟩]⟩, (1)⟩]

theorem exact27126RawTermsValid :
    exact27126RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27126 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24108⟩⟩) exact27126RawTerms .large 27125 .exactZero (none)

def event27127 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27688⟩⟩) 0 ⟨24108⟩ 27126

def event27128 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27688⟩⟩) (.authority (.operator))

def exact27129RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩, (1)⟩]

theorem exact27129RawTermsValid :
    exact27129RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27129 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27688⟩⟩) exact27129RawTerms (.finite 8192) 27128 .exactZero (none)

def event27130 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event27131 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event27132 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15908⟩⟩) 0 ⟨15834⟩ 27118

def event27133 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15908⟩⟩) 1 ⟨110⟩ 27131

def event27134 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15908⟩⟩) (.sum [.predecessor 0 27132 .coefficient, .predecessor 1 27133 .coefficient])

def event27135 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15908⟩⟩) (.finite 16)

def eventLeaf1680 : Array AnnotatedEvent := #[
  { event := event26880
    frameStart := 26869 },
  { event := event26881
    frameStart := 26869 },
  { event := event26882
    frameStart := 26869 },
  { event := event26883
    frameStart := 26869 },
  { event := event26884
    frameStart := 26869 },
  { event := event26885
    frameStart := 26869 },
  { event := event26886
    frameStart := 26869 },
  { event := event26887
    frameStart := 26869 },
  { event := event26888
    frameStart := 26869 },
  { event := event26889
    frameStart := 26869 },
  { event := event26890
    frameStart := 26869 },
  { event := event26891
    frameStart := 26869 },
  { event := event26892
    frameStart := 26869 },
  { event := event26893
    frameStart := 26869 },
  { event := event26894
    frameStart := 26869 },
  { event := event26895
    frameStart := 26869 }
]

def eventLeaf1681 : Array AnnotatedEvent := #[
  { event := event26896
    frameStart := 26869 },
  { event := event26897
    frameStart := 26869 },
  { event := event26898
    frameStart := 26869 },
  { event := event26899
    frameStart := 26869 },
  { event := event26900
    frameStart := 26869 },
  { event := event26901
    frameStart := 26869 },
  { event := event26902
    frameStart := 26869 },
  { event := event26903
    frameStart := 26869 },
  { event := event26904
    frameStart := 26869 },
  { event := event26905
    frameStart := 26869 },
  { event := event26906
    frameStart := 26869 },
  { event := event26907
    frameStart := 26869 },
  { event := event26908
    frameStart := 26869 },
  { event := event26909
    frameStart := 26869 },
  { event := event26910
    frameStart := 26869 },
  { event := event26911
    frameStart := 26869 }
]

def eventLeaf1682 : Array AnnotatedEvent := #[
  { event := event26912
    frameStart := 26869 },
  { event := event26913
    frameStart := 26869 },
  { event := event26914
    frameStart := 26869 },
  { event := event26915
    frameStart := 26869 },
  { event := event26916
    frameStart := 26869 },
  { event := event26917
    frameStart := 26869 },
  { event := event26918
    frameStart := 26869 },
  { event := event26919
    frameStart := 26869 },
  { event := event26920
    frameStart := 26869 },
  { event := event26921
    frameStart := 26869 },
  { event := event26922
    frameStart := 26869 },
  { event := event26923
    frameStart := 26869 },
  { event := event26924
    frameStart := 26869 },
  { event := event26925
    frameStart := 26869 },
  { event := event26926
    frameStart := 26869 },
  { event := event26927
    frameStart := 26869 }
]

def eventLeaf1683 : Array AnnotatedEvent := #[
  { event := event26928
    frameStart := 26869 },
  { event := event26929
    frameStart := 26869 },
  { event := event26930
    frameStart := 26869 },
  { event := event26931
    frameStart := 26869 },
  { event := event26932
    frameStart := 26869 },
  { event := event26933
    frameStart := 26869 },
  { event := event26934
    frameStart := 26869 },
  { event := event26935
    frameStart := 26869 },
  { event := event26936
    frameStart := 26869 },
  { event := event26937
    frameStart := 26869 },
  { event := event26938
    frameStart := 26869 },
  { event := event26939
    frameStart := 26869 },
  { event := event26940
    frameStart := 26869 },
  { event := event26941
    frameStart := 26869 },
  { event := event26942
    frameStart := 26869 },
  { event := event26943
    frameStart := 26869 }
]

def eventLeaf1684 : Array AnnotatedEvent := #[
  { event := event26944
    frameStart := 26869 },
  { event := event26945
    frameStart := 26869 },
  { event := event26946
    frameStart := 26869 },
  { event := event26947
    frameStart := 26869 },
  { event := event26948
    frameStart := 26869 },
  { event := event26949
    frameStart := 26869 },
  { event := event26950
    frameStart := 26869 },
  { event := event26951
    frameStart := 26869 },
  { event := event26952
    frameStart := 26869 },
  { event := event26953
    frameStart := 26869 },
  { event := event26954
    frameStart := 26869 },
  { event := event26955
    frameStart := 26869 },
  { event := event26956
    frameStart := 26869 },
  { event := event26957
    frameStart := 26869 },
  { event := event26958
    frameStart := 26869 },
  { event := event26959
    frameStart := 26869 }
]

def eventLeaf1685 : Array AnnotatedEvent := #[
  { event := event26960
    frameStart := 26869 },
  { event := event26961
    frameStart := 26869 },
  { event := event26962
    frameStart := 26869 },
  { event := event26963
    frameStart := 26869 },
  { event := event26964
    frameStart := 26869 },
  { event := event26965
    frameStart := 26869 },
  { event := event26966
    frameStart := 26869 },
  { event := event26967
    frameStart := 26869 },
  { event := event26968
    frameStart := 26869 },
  { event := event26969
    frameStart := 26869 },
  { event := event26970
    frameStart := 26869 },
  { event := event26971
    frameStart := 26869 },
  { event := event26972
    frameStart := 26869 },
  { event := event26973
    frameStart := 26869 },
  { event := event26974
    frameStart := 26869 },
  { event := event26975
    frameStart := 26869 }
]

def eventLeaf1686 : Array AnnotatedEvent := #[
  { event := event26976
    frameStart := 26869 },
  { event := event26977
    frameStart := 26869 },
  { event := event26978
    frameStart := 26869 },
  { event := event26979
    frameStart := 26869 },
  { event := event26980
    frameStart := 26869 },
  { event := event26981
    frameStart := 26869 },
  { event := event26982
    frameStart := 26869 },
  { event := event26983
    frameStart := 26869 },
  { event := event26984
    frameStart := 26869 },
  { event := event26985
    frameStart := 26869 },
  { event := event26986
    frameStart := 26869 },
  { event := event26987
    frameStart := 0 },
  { event := event26988
    frameStart := 0 },
  { event := event26989
    frameStart := 0 },
  { event := event26990
    frameStart := 0 },
  { event := event26991
    frameStart := 0 }
]

def eventLeaf1687 : Array AnnotatedEvent := #[
  { event := event26992
    frameStart := 0 },
  { event := event26993
    frameStart := 0 },
  { event := event26994
    frameStart := 0 },
  { event := event26995
    frameStart := 0 },
  { event := event26996
    frameStart := 0 },
  { event := event26997
    frameStart := 0 },
  { event := event26998
    frameStart := 0 },
  { event := event26999
    frameStart := 0 },
  { event := event27000
    frameStart := 0 },
  { event := event27001
    frameStart := 0 },
  { event := event27002
    frameStart := 0 },
  { event := event27003
    frameStart := 0 },
  { event := event27004
    frameStart := 0 },
  { event := event27005
    frameStart := 0 },
  { event := event27006
    frameStart := 0 },
  { event := event27007
    frameStart := 0 }
]

def eventLeaf1688 : Array AnnotatedEvent := #[
  { event := event27008
    frameStart := 0 },
  { event := event27009
    frameStart := 0 },
  { event := event27010
    frameStart := 0 },
  { event := event27011
    frameStart := 0 },
  { event := event27012
    frameStart := 0 },
  { event := event27013
    frameStart := 0 },
  { event := event27014
    frameStart := 0 },
  { event := event27015
    frameStart := 0 },
  { event := event27016
    frameStart := 0 },
  { event := event27017
    frameStart := 0 },
  { event := event27018
    frameStart := 0 },
  { event := event27019
    frameStart := 0 },
  { event := event27020
    frameStart := 0 },
  { event := event27021
    frameStart := 0 },
  { event := event27022
    frameStart := 0 },
  { event := event27023
    frameStart := 0 }
]

def eventLeaf1689 : Array AnnotatedEvent := #[
  { event := event27024
    frameStart := 27024 },
  { event := event27025
    frameStart := 27024 },
  { event := event27026
    frameStart := 27024 },
  { event := event27027
    frameStart := 27024 },
  { event := event27028
    frameStart := 27024 },
  { event := event27029
    frameStart := 27024 },
  { event := event27030
    frameStart := 27024 },
  { event := event27031
    frameStart := 27024 },
  { event := event27032
    frameStart := 27024 },
  { event := event27033
    frameStart := 27024 },
  { event := event27034
    frameStart := 27024 },
  { event := event27035
    frameStart := 27024 },
  { event := event27036
    frameStart := 27024 },
  { event := event27037
    frameStart := 27024 },
  { event := event27038
    frameStart := 27024 },
  { event := event27039
    frameStart := 27024 }
]

def eventLeaf1690 : Array AnnotatedEvent := #[
  { event := event27040
    frameStart := 27024 },
  { event := event27041
    frameStart := 27024 },
  { event := event27042
    frameStart := 27024 },
  { event := event27043
    frameStart := 27024 },
  { event := event27044
    frameStart := 27024 },
  { event := event27045
    frameStart := 27024 },
  { event := event27046
    frameStart := 27024 },
  { event := event27047
    frameStart := 27024 },
  { event := event27048
    frameStart := 27024 },
  { event := event27049
    frameStart := 27024 },
  { event := event27050
    frameStart := 27024 },
  { event := event27051
    frameStart := 27024 },
  { event := event27052
    frameStart := 27024 },
  { event := event27053
    frameStart := 27024 },
  { event := event27054
    frameStart := 27024 },
  { event := event27055
    frameStart := 27024 }
]

def eventLeaf1691 : Array AnnotatedEvent := #[
  { event := event27056
    frameStart := 27024 },
  { event := event27057
    frameStart := 27024 },
  { event := event27058
    frameStart := 27024 },
  { event := event27059
    frameStart := 27024 },
  { event := event27060
    frameStart := 27024 },
  { event := event27061
    frameStart := 27024 },
  { event := event27062
    frameStart := 27024 },
  { event := event27063
    frameStart := 27024 },
  { event := event27064
    frameStart := 27024 },
  { event := event27065
    frameStart := 27024 },
  { event := event27066
    frameStart := 27024 },
  { event := event27067
    frameStart := 27024 },
  { event := event27068
    frameStart := 27024 },
  { event := event27069
    frameStart := 27024 },
  { event := event27070
    frameStart := 27024 },
  { event := event27071
    frameStart := 27024 }
]

def eventLeaf1692 : Array AnnotatedEvent := #[
  { event := event27072
    frameStart := 27024 },
  { event := event27073
    frameStart := 27024 },
  { event := event27074
    frameStart := 27024 },
  { event := event27075
    frameStart := 27024 },
  { event := event27076
    frameStart := 27024 },
  { event := event27077
    frameStart := 27024 },
  { event := event27078
    frameStart := 27078 },
  { event := event27079
    frameStart := 27078 },
  { event := event27080
    frameStart := 27078 },
  { event := event27081
    frameStart := 27078 },
  { event := event27082
    frameStart := 27078 },
  { event := event27083
    frameStart := 27078 },
  { event := event27084
    frameStart := 27078 },
  { event := event27085
    frameStart := 27078 },
  { event := event27086
    frameStart := 27078 },
  { event := event27087
    frameStart := 27078 }
]

def eventLeaf1693 : Array AnnotatedEvent := #[
  { event := event27088
    frameStart := 27078 },
  { event := event27089
    frameStart := 27078 },
  { event := event27090
    frameStart := 27078 },
  { event := event27091
    frameStart := 27078 },
  { event := event27092
    frameStart := 27078 },
  { event := event27093
    frameStart := 27078 },
  { event := event27094
    frameStart := 27078 },
  { event := event27095
    frameStart := 27078 },
  { event := event27096
    frameStart := 27078 },
  { event := event27097
    frameStart := 27078 },
  { event := event27098
    frameStart := 27078 },
  { event := event27099
    frameStart := 27078 },
  { event := event27100
    frameStart := 27078 },
  { event := event27101
    frameStart := 27078 },
  { event := event27102
    frameStart := 27078 },
  { event := event27103
    frameStart := 27078 }
]

def eventLeaf1694 : Array AnnotatedEvent := #[
  { event := event27104
    frameStart := 27078 },
  { event := event27105
    frameStart := 27078 },
  { event := event27106
    frameStart := 27078 },
  { event := event27107
    frameStart := 27078 },
  { event := event27108
    frameStart := 27078 },
  { event := event27109
    frameStart := 27078 },
  { event := event27110
    frameStart := 27078 },
  { event := event27111
    frameStart := 27078 },
  { event := event27112
    frameStart := 27078 },
  { event := event27113
    frameStart := 27078 },
  { event := event27114
    frameStart := 27078 },
  { event := event27115
    frameStart := 27078 },
  { event := event27116
    frameStart := 27078 },
  { event := event27117
    frameStart := 27078 },
  { event := event27118
    frameStart := 27078 },
  { event := event27119
    frameStart := 27078 }
]

def eventLeaf1695 : Array AnnotatedEvent := #[
  { event := event27120
    frameStart := 27078 },
  { event := event27121
    frameStart := 27078 },
  { event := event27122
    frameStart := 27078 },
  { event := event27123
    frameStart := 27078 },
  { event := event27124
    frameStart := 27078 },
  { event := event27125
    frameStart := 27078 },
  { event := event27126
    frameStart := 27078 },
  { event := event27127
    frameStart := 27078 },
  { event := event27128
    frameStart := 27078 },
  { event := event27129
    frameStart := 27078 },
  { event := event27130
    frameStart := 27078 },
  { event := event27131
    frameStart := 27078 },
  { event := event27132
    frameStart := 27078 },
  { event := event27133
    frameStart := 27078 },
  { event := event27134
    frameStart := 27078 },
  { event := event27135
    frameStart := 27078 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events105
