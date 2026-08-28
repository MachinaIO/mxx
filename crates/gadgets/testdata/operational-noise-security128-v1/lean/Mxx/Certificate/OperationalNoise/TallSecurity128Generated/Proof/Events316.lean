import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events316

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact80896RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59647⟩⟩], []⟩, (1)⟩]

theorem exact80896RawTermsValid :
    exact80896RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80896 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59647⟩⟩) exact80896RawTerms (.finite 18) 80895 .exactZero (none)

def event80897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59648⟩⟩) 0 ⟨59647⟩ 80896

def event80898 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59648⟩⟩) 1 ⟨25322⟩ 80893

def event80899 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59648⟩⟩) (.product (.predecessor 0 80897 .coefficient) (.predecessor 1 80898 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event80900 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59648⟩⟩, .operator (⟨80896, 0⟩, ⟨80893, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], []⟩, (1)⟩)

def exact80901RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], []⟩, (1)⟩]

theorem exact80901RawTermsValid :
    exact80901RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80901 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59648⟩⟩) exact80901RawTerms (.finite 324) 80899 .exactZero (none)

def event80902 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59649⟩⟩) 0 ⟨59648⟩ 80901

def event80903 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59649⟩⟩) (.identity (.predecessor 0 80902 .coefficient))

def event80904 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨59649⟩⟩) (.finite 324)

def event80905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60984⟩⟩) 0 ⟨59649⟩ 80904

def event80906 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60984⟩⟩) (.authority (.programFamilyFact))

def event80907 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨60984⟩⟩) (.finite 3720)

def event80908 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event80909 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60985⟩⟩) 0 ⟨7177⟩ 80908

def event80910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60985⟩⟩) 1 ⟨60984⟩ 80907

def event80911 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60985⟩⟩) (.authority (.operator))

def exact80912RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨60985⟩⟩]⟩, (1)⟩]

theorem exact80912RawTermsValid :
    exact80912RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80912 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60985⟩⟩) exact80912RawTerms .large 80911 .exactZero (none)

def event80913 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61525⟩⟩) 0 ⟨60985⟩ 80912

def event80914 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61525⟩⟩) (.authority (.operator))

def exact80915RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩, (1)⟩]

theorem exact80915RawTermsValid :
    exact80915RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80915 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61525⟩⟩) exact80915RawTerms (.finite 8192) 80914 .exactZero (none)

def event80916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event80917 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event80918 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61250⟩⟩) 0 ⟨59649⟩ 80904

def event80919 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61250⟩⟩) 1 ⟨136⟩ 80917

def event80920 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61250⟩⟩) (.sum [.predecessor 0 80918 .coefficient, .predecessor 1 80919 .coefficient])

def event80921 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨61250⟩⟩) (.finite 324)

def event80922 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61251⟩⟩) 0 ⟨61250⟩ 80921

def event80923 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61251⟩⟩) (.identity (.predecessor 0 80922 .coefficient))

def exact80924RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], []⟩, (1)⟩]

theorem exact80924RawTermsValid :
    exact80924RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80924 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61251⟩⟩) exact80924RawTerms (.finite 324) 80923 .exactZero (none)

def event80925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact80926RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact80926RawTermsValid :
    exact80926RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80926 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact80926RawTerms .large 80925 .exactZero (none)

def event80927 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61252⟩⟩) 0 ⟨6908⟩ 80926

def event80928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61252⟩⟩) 1 ⟨61251⟩ 80924

def event80929 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61252⟩⟩) (.product (.predecessor 0 80927 .coefficient) (.predecessor 1 80928 .coefficient) (⟨false, false, none, none, none⟩))

def event80930 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61252⟩⟩, .operator (⟨80926, 0⟩, ⟨80924, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact80931RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact80931RawTermsValid :
    exact80931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80931 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61252⟩⟩) exact80931RawTerms .large 80929 .exactZero (none)

def event80932 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event80933 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event80934 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 80908

def event80935 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact80936RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact80936RawTermsValid :
    exact80936RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80936 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact80936RawTerms .large 80935 .exactZero (none)

def event80937 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7274⟩⟩) 0 ⟨7178⟩ 80936

def event80938 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7274⟩⟩) (.identity (.predecessor 0 80937 .coefficient))

def exact80939RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩, (1)⟩]

theorem exact80939RawTermsValid :
    exact80939RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80939 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7274⟩⟩) exact80939RawTerms .large 80938 .exactZero (none)

def event80940 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9535⟩⟩) 0 ⟨7274⟩ 80939

def event80941 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9535⟩⟩) (.authority (.operator))

def exact80942RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩, (1)⟩]

theorem exact80942RawTermsValid :
    exact80942RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80942 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9535⟩⟩) exact80942RawTerms (.finite 8192) 80941 .exactZero (none)

def event80943 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9536⟩⟩) 0 ⟨9535⟩ 80942

def event80944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9536⟩⟩) 1 ⟨2370⟩ 80933

def event80945 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9536⟩⟩) (.scale (.predecessor 0 80943 .coefficient) (.value (.predecessor 1 80944 .coefficient)))

def exact80946RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩, (1)⟩]

theorem exact80946RawTermsValid :
    exact80946RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80946 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9536⟩⟩) exact80946RawTerms (.finite 8192) 80945 .exactZero (none)

def event80947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7291⟩⟩) 0 ⟨7178⟩ 80936

def event80948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7291⟩⟩) (.identity (.predecessor 0 80947 .coefficient))

def exact80949RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩]⟩, (1)⟩]

theorem exact80949RawTermsValid :
    exact80949RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80949 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7291⟩⟩) exact80949RawTerms .large 80948 .exactZero (none)

def event80950 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9537⟩⟩) 0 ⟨7291⟩ 80949

def event80951 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9537⟩⟩) 1 ⟨9536⟩ 80946

def event80952 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9537⟩⟩) (.product (.predecessor 0 80950 .coefficient) (.predecessor 1 80951 .coefficient) (⟨false, false, none, none, none⟩))

def event80953 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9537⟩⟩, .operator (⟨80949, 0⟩, ⟨80946, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩, (1)⟩)

def exact80954RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩, (1)⟩]

theorem exact80954RawTermsValid :
    exact80954RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80954 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9537⟩⟩) exact80954RawTerms .large 80952 .exactZero (none)

def event80955 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61253⟩⟩) 0 ⟨9537⟩ 80954

def event80956 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61253⟩⟩) 1 ⟨61252⟩ 80931

def event80957 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61253⟩⟩) (.sum [.predecessor 0 80955 .coefficient, .predecessor 1 80956 .coefficient])

def exact80958RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80958RawTermsValid :
    exact80958RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80958 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61253⟩⟩) exact80958RawTerms .large 80957 .exactZero (none)

def event80959 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61528⟩⟩) 0 ⟨61253⟩ 80958

def event80960 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61528⟩⟩) 1 ⟨61525⟩ 80915

def event80961 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61528⟩⟩) (.product (.predecessor 0 80959 .coefficient) (.predecessor 1 80960 .coefficient) (⟨false, false, none, none, none⟩))

def event80962 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61528⟩⟩, .operator (⟨80958, 0⟩, ⟨80915, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩, (1)⟩)

def event80963 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61528⟩⟩, .operator (⟨80958, 1⟩, ⟨80915, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩, (-1)⟩)

def event80964 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨61528⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61525⟩⟩) ⟨60985⟩ 80912)

def event80965 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61528⟩⟩, .relation 80964 0, ⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨60985⟩⟩]⟩, (-1)⟩)

def exact80966RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨60985⟩⟩]⟩, (-1)⟩]

theorem exact80966RawTermsValid :
    exact80966RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80966 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61528⟩⟩) exact80966RawTerms .large 80961 .exactZero (none)

def event80967 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59876⟩⟩) 0 ⟨59649⟩ 80904

def event80968 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59876⟩⟩) (.authority (.programFamilyFact))

def exact80969RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59876⟩⟩], []⟩, (1)⟩]

theorem exact80969RawTermsValid :
    exact80969RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80969 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59876⟩⟩) exact80969RawTerms (.finite 18) 80968 .exactZero (none)

def event80970 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59878⟩⟩) 0 ⟨6908⟩ 80926

def event80971 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59878⟩⟩) 1 ⟨59876⟩ 80969

def event80972 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59878⟩⟩) (.product (.predecessor 0 80970 .coefficient) (.predecessor 1 80971 .coefficient) (⟨false, true, none, none, some 1⟩))

def event80973 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59878⟩⟩, .operator (⟨80926, 0⟩, ⟨80969, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨59876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact80974RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact80974RawTermsValid :
    exact80974RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80974 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59878⟩⟩) exact80974RawTerms .large 80972 .exactZero (none)

def event80975 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7186⟩⟩) 0 ⟨7177⟩ 80908

def event80976 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7186⟩⟩) (.authority (.operator))

def exact80977RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩, (1)⟩]

theorem exact80977RawTermsValid :
    exact80977RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80977 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7186⟩⟩) exact80977RawTerms .large 80976 .exactZero (none)

def event80978 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59879⟩⟩) 0 ⟨7186⟩ 80977

def event80979 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59879⟩⟩) 1 ⟨59878⟩ 80974

def event80980 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59879⟩⟩) (.sum [.predecessor 0 80978 .coefficient, .predecessor 1 80979 .coefficient])

def exact80981RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80981RawTermsValid :
    exact80981RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80981 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59879⟩⟩) exact80981RawTerms .large 80980 .exactZero (none)

def event80982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61529⟩⟩) 0 ⟨59879⟩ 80981

def event80983 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61529⟩⟩) 1 ⟨61528⟩ 80966

def event80984 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61529⟩⟩) (.sum [.predecessor 0 80982 .coefficient, .predecessor 1 80983 .coefficient])

def exact80985RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨60985⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80985RawTermsValid :
    exact80985RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80985 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61529⟩⟩) exact80985RawTerms .large 80984 .exactZero (none)

def event80986 : Event := .preFoldPolynomial 80985 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨60985⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact80987RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨60985⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event80987 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨61529⟩⟩) 80986 exact80987RawTerms .large 80984 .exactZero (none)

def event80988 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨59649⟩⟩) ⟨⟨65⟩, ⟨43⟩, ⟨135⟩⟩ ⟨80822, 80988⟩

def event80989 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨60452⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩) (1) 0 2 (.universal 80988 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60449⟩⟩]⟩) (none) 80987)

def event80990 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60452⟩⟩, .relation 80989 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩, (1)⟩)

def event80991 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60452⟩⟩, .relation 80989 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩, (-1)⟩)

def event80992 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60452⟩⟩, .relation 80989 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨60985⟩⟩]⟩, (1)⟩)

def event80993 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60452⟩⟩, .relation 80989 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact80994RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨60985⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80994RawTermsValid :
    exact80994RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80994 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60452⟩⟩) exact80994RawTerms .large 80818 (.finite 202072841853861888) (some (80820))

def event80995 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61527⟩⟩) 0 ⟨60452⟩ 80994

def event80996 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61527⟩⟩) 1 ⟨61526⟩ 80808

def event80997 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61527⟩⟩) (.sum [.predecessor 0 80995 .coefficient, .predecessor 1 80996 .coefficient])

def event80998 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61527⟩⟩, .operator (⟨80994, 2⟩, ⟨80808, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], [⟨.program ⟨257⟩, ⟨60985⟩⟩]⟩, (-1)⟩)

def event80999 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61527⟩⟩, .operator (⟨80994, 1⟩, ⟨80808, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61525⟩⟩]⟩, (1)⟩)

def event81000 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61527⟩⟩) (.sum [.result 80994 .summary, .result 80808 .summary])

def exact81001RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact81001RawTermsValid :
    exact81001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81001 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61527⟩⟩) exact81001RawTerms .large 80997 (.finite 2997962647681031733248) (some (81000))

def event81002 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62080⟩⟩) 0 ⟨61527⟩ 81001

def event81003 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62080⟩⟩) 1 ⟨62078⟩ 80724

def event81004 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62080⟩⟩) (.product (.predecessor 0 81002 .coefficient) (.predecessor 1 81003 .coefficient) (⟨false, false, none, none, none⟩))

def event81005 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62080⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨62078⟩⟩]⟩) [⟨.result 80724 .coefficient, false, none⟩])

def event81006 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62080⟩⟩) (.product (.result 81001 .summary) (.transfer 81005) (⟨false, false, none, none, none⟩))

def event81007 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62080⟩⟩, .operator (⟨81001, 0⟩, ⟨80724, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62078⟩⟩]⟩, (1)⟩)

def event81008 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62080⟩⟩, .operator (⟨81001, 1⟩, ⟨80724, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62078⟩⟩]⟩, (-1)⟩)

def event81009 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨62080⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62078⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨62078⟩⟩) ⟨61155⟩ 80721)

def event81010 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62080⟩⟩, .relation 81009 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59876⟩⟩], [⟨.program ⟨257⟩, ⟨61155⟩⟩]⟩, (-1)⟩)

def exact81011RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62078⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨59876⟩⟩], [⟨.program ⟨257⟩, ⟨61155⟩⟩]⟩, (-1)⟩]

theorem exact81011RawTermsValid :
    exact81011RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81011 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62080⟩⟩) exact81011RawTerms .large 81004 (.finite 32190378816049003834595889643520) (some (81006))

def event81012 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60816⟩⟩) 0 ⟨59877⟩ 3333

def event81013 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60816⟩⟩) (.authority (.relationPreimageSource ⟨72⟩))

def exact81014RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨60816⟩⟩]⟩, (1)⟩]

theorem exact81014RawTermsValid :
    exact81014RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81014 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60816⟩⟩) exact81014RawTerms (.finite 5647228698) 81013 .exactZero (none)

def event81015 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60818⟩⟩) 0 ⟨60816⟩ 81014

def event81016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60818⟩⟩) 1 ⟨2370⟩ 4

def event81017 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60818⟩⟩) (.scale (.predecessor 0 81015 .coefficient) (.value (.predecessor 1 81016 .coefficient)))

def exact81018RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨60816⟩⟩]⟩, (1)⟩]

theorem exact81018RawTermsValid :
    exact81018RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81018 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60818⟩⟩) exact81018RawTerms (.finite 5647228698) 81017 .exactZero (none)

def event81019 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60819⟩⟩) 0 ⟨10368⟩ 75995

def event81020 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60819⟩⟩) 1 ⟨60818⟩ 81018

def event81021 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60819⟩⟩) (.product (.predecessor 0 81019 .coefficient) (.predecessor 1 81020 .coefficient) (⟨false, false, none, none, none⟩))

def event81022 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60819⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨60816⟩⟩]⟩) [⟨.result 81014 .coefficient, false, none⟩])

def event81023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60819⟩⟩) (.product (.result 75995 .summary) (.transfer 81022) (⟨false, false, none, none, none⟩))

def event81024 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60819⟩⟩, .operator (⟨75995, 0⟩, ⟨81018, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60816⟩⟩]⟩, (1)⟩)

def event81025 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨60817⟩⟩)

def event81026 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event81027 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event81028 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event81029 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event81030 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event81031 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event81032 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event81033 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event81034 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 81033

def event81035 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 81031

def event81036 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 81034 .coefficient) (.value (.predecessor 1 81035 .coefficient)))

def event81037 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event81038 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 81037

def event81039 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 81029

def event81040 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 81038 .coefficient, .predecessor 1 81039 .coefficient])

def event81041 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event81042 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 81041

def event81043 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 81027

def event81044 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 81043 .coefficient))

def event81045 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event81046 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25322⟩⟩) 0 ⟨10325⟩ 81045

def event81047 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25322⟩⟩) (.authority (.programFamilyFact))

def exact81048RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩], []⟩, (1)⟩]

theorem exact81048RawTermsValid :
    exact81048RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81048 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25322⟩⟩) exact81048RawTerms (.finite 18) 81047 .exactZero (none)

def event81049 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59647⟩⟩) 0 ⟨10325⟩ 81045

def event81050 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59647⟩⟩) (.authority (.programFamilyFact))

def exact81051RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59647⟩⟩], []⟩, (1)⟩]

theorem exact81051RawTermsValid :
    exact81051RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81051 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59647⟩⟩) exact81051RawTerms (.finite 18) 81050 .exactZero (none)

def event81052 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59648⟩⟩) 0 ⟨59647⟩ 81051

def event81053 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59648⟩⟩) 1 ⟨25322⟩ 81048

def event81054 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59648⟩⟩) (.product (.predecessor 0 81052 .coefficient) (.predecessor 1 81053 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event81055 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59648⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], []⟩) [⟨.result 81051 .coefficient, true, some 1⟩, ⟨.result 81048 .coefficient, true, some 1⟩])

def event81056 : Event := .survivorFold (1) 81055

def exact81057RawTerms : List Term := []

theorem exact81057RawTermsValid :
    exact81057RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81057 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59648⟩⟩) exact81057RawTerms (.finite 324) 81054 (.finite 324) (some (81055))

def event81058 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59649⟩⟩) 0 ⟨59648⟩ 81057

def event81059 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59649⟩⟩) (.identity (.predecessor 0 81058 .coefficient))

def event81060 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨59649⟩⟩) (.finite 324)

def event81061 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59876⟩⟩) 0 ⟨59649⟩ 81060

def event81062 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59876⟩⟩) (.authority (.programFamilyFact))

def exact81063RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59876⟩⟩], []⟩, (1)⟩]

theorem exact81063RawTermsValid :
    exact81063RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81063 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59876⟩⟩) exact81063RawTerms (.finite 18) 81062 .exactZero (none)

def event81064 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59877⟩⟩) 0 ⟨59876⟩ 81063

def event81065 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59877⟩⟩) (.identity (.predecessor 0 81064 .coefficient))

def event81066 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨59877⟩⟩) (.finite 18)

def event81067 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60816⟩⟩) 0 ⟨59877⟩ 81066

def event81068 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60816⟩⟩) (.authority (.relationPreimageSource ⟨72⟩))

def exact81069RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨60816⟩⟩]⟩, (1)⟩]

theorem exact81069RawTermsValid :
    exact81069RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81069 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60816⟩⟩) exact81069RawTerms (.finite 5647228698) 81068 .exactZero (none)

def event81070 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact81071RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact81071RawTermsValid :
    exact81071RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81071 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact81071RawTerms .large 81070 .exactZero (none)

def event81072 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60817⟩⟩) 0 ⟨35⟩ 81071

def event81073 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60817⟩⟩) 1 ⟨60816⟩ 81069

def event81074 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60817⟩⟩) (.product (.predecessor 0 81072 .coefficient) (.predecessor 1 81073 .coefficient) (⟨false, false, none, none, none⟩))

def event81075 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60817⟩⟩, .operator (⟨81071, 0⟩, ⟨81069, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60816⟩⟩]⟩, (1)⟩)

def exact81076RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60816⟩⟩]⟩, (1)⟩]

theorem exact81076RawTermsValid :
    exact81076RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81076 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60817⟩⟩) exact81076RawTerms .large 81074 .exactZero (none)

def event81077 : Event := .preFoldPolynomial 81076 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60816⟩⟩]⟩, (1)⟩] .exactZero none

def exact81078RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60816⟩⟩]⟩, (1)⟩]

def event81078 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨60817⟩⟩) 81077 exact81078RawTerms .large 81074 .exactZero (none)

def event81079 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨62083⟩⟩)

def event81080 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event81081 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event81082 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event81083 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event81084 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event81085 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event81086 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event81087 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event81088 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 81087

def event81089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 81085

def event81090 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 81088 .coefficient) (.value (.predecessor 1 81089 .coefficient)))

def event81091 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event81092 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 81091

def event81093 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 81083

def event81094 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 81092 .coefficient, .predecessor 1 81093 .coefficient])

def event81095 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event81096 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 81095

def event81097 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 81081

def event81098 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 81097 .coefficient))

def event81099 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event81100 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25322⟩⟩) 0 ⟨10325⟩ 81099

def event81101 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25322⟩⟩) (.authority (.programFamilyFact))

def exact81102RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩], []⟩, (1)⟩]

theorem exact81102RawTermsValid :
    exact81102RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81102 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25322⟩⟩) exact81102RawTerms (.finite 18) 81101 .exactZero (none)

def event81103 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59647⟩⟩) 0 ⟨10325⟩ 81099

def event81104 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59647⟩⟩) (.authority (.programFamilyFact))

def exact81105RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59647⟩⟩], []⟩, (1)⟩]

theorem exact81105RawTermsValid :
    exact81105RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81105 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59647⟩⟩) exact81105RawTerms (.finite 18) 81104 .exactZero (none)

def event81106 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59648⟩⟩) 0 ⟨59647⟩ 81105

def event81107 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59648⟩⟩) 1 ⟨25322⟩ 81102

def event81108 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59648⟩⟩) (.product (.predecessor 0 81106 .coefficient) (.predecessor 1 81107 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event81109 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59648⟩⟩, .operator (⟨81105, 0⟩, ⟨81102, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], []⟩, (1)⟩)

def exact81110RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25322⟩⟩, ⟨.program ⟨257⟩, ⟨59647⟩⟩], []⟩, (1)⟩]

theorem exact81110RawTermsValid :
    exact81110RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81110 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59648⟩⟩) exact81110RawTerms (.finite 324) 81108 .exactZero (none)

def event81111 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59649⟩⟩) 0 ⟨59648⟩ 81110

def event81112 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59649⟩⟩) (.identity (.predecessor 0 81111 .coefficient))

def event81113 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨59649⟩⟩) (.finite 324)

def event81114 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59876⟩⟩) 0 ⟨59649⟩ 81113

def event81115 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59876⟩⟩) (.authority (.programFamilyFact))

def exact81116RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59876⟩⟩], []⟩, (1)⟩]

theorem exact81116RawTermsValid :
    exact81116RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81116 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59876⟩⟩) exact81116RawTerms (.finite 18) 81115 .exactZero (none)

def event81117 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59877⟩⟩) 0 ⟨59876⟩ 81116

def event81118 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59877⟩⟩) (.identity (.predecessor 0 81117 .coefficient))

def event81119 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨59877⟩⟩) (.finite 18)

def event81120 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61153⟩⟩) 0 ⟨59877⟩ 81119

def event81121 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61153⟩⟩) (.authority (.programFamilyFact))

def event81122 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨61153⟩⟩) (.finite 3720)

def event81123 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event81124 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61155⟩⟩) 0 ⟨7177⟩ 81123

def event81125 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61155⟩⟩) 1 ⟨61153⟩ 81122

def event81126 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61155⟩⟩) (.authority (.operator))

def exact81127RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨61155⟩⟩]⟩, (1)⟩]

theorem exact81127RawTermsValid :
    exact81127RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81127 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61155⟩⟩) exact81127RawTerms .large 81126 .exactZero (none)

def event81128 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62078⟩⟩) 0 ⟨61155⟩ 81127

def event81129 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62078⟩⟩) (.authority (.operator))

def exact81130RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨62078⟩⟩]⟩, (1)⟩]

theorem exact81130RawTermsValid :
    exact81130RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81130 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62078⟩⟩) exact81130RawTerms (.finite 8192) 81129 .exactZero (none)

def event81131 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event81132 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event81133 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61330⟩⟩) 0 ⟨59877⟩ 81119

def event81134 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61330⟩⟩) 1 ⟨136⟩ 81132

def event81135 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61330⟩⟩) (.sum [.predecessor 0 81133 .coefficient, .predecessor 1 81134 .coefficient])

def event81136 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨61330⟩⟩) (.finite 18)

def event81137 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61331⟩⟩) 0 ⟨61330⟩ 81136

def event81138 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61331⟩⟩) (.identity (.predecessor 0 81137 .coefficient))

def exact81139RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59876⟩⟩], []⟩, (1)⟩]

theorem exact81139RawTermsValid :
    exact81139RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81139 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61331⟩⟩) exact81139RawTerms (.finite 18) 81138 .exactZero (none)

def event81140 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact81141RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact81141RawTermsValid :
    exact81141RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81141 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact81141RawTerms .large 81140 .exactZero (none)

def event81142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61332⟩⟩) 0 ⟨6908⟩ 81141

def event81143 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61332⟩⟩) 1 ⟨61331⟩ 81139

def event81144 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61332⟩⟩) (.product (.predecessor 0 81142 .coefficient) (.predecessor 1 81143 .coefficient) (⟨false, false, none, none, none⟩))

def event81145 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61332⟩⟩, .operator (⟨81141, 0⟩, ⟨81139, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨59876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact81146RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact81146RawTermsValid :
    exact81146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81146 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61332⟩⟩) exact81146RawTerms .large 81144 .exactZero (none)

def event81147 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7186⟩⟩) 0 ⟨7177⟩ 81123

def event81148 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7186⟩⟩) (.authority (.operator))

def exact81149RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩, (1)⟩]

theorem exact81149RawTermsValid :
    exact81149RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event81149 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7186⟩⟩) exact81149RawTerms .large 81148 .exactZero (none)

def event81150 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61333⟩⟩) 0 ⟨7186⟩ 81149

def event81151 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61333⟩⟩) 1 ⟨61332⟩ 81146

def eventLeaf5056 : Array AnnotatedEvent := #[
  { event := event80896
    frameStart := 80870 },
  { event := event80897
    frameStart := 80870 },
  { event := event80898
    frameStart := 80870 },
  { event := event80899
    frameStart := 80870 },
  { event := event80900
    frameStart := 80870 },
  { event := event80901
    frameStart := 80870 },
  { event := event80902
    frameStart := 80870 },
  { event := event80903
    frameStart := 80870 },
  { event := event80904
    frameStart := 80870 },
  { event := event80905
    frameStart := 80870 },
  { event := event80906
    frameStart := 80870 },
  { event := event80907
    frameStart := 80870 },
  { event := event80908
    frameStart := 80870 },
  { event := event80909
    frameStart := 80870 },
  { event := event80910
    frameStart := 80870 },
  { event := event80911
    frameStart := 80870 }
]

def eventLeaf5057 : Array AnnotatedEvent := #[
  { event := event80912
    frameStart := 80870 },
  { event := event80913
    frameStart := 80870 },
  { event := event80914
    frameStart := 80870 },
  { event := event80915
    frameStart := 80870 },
  { event := event80916
    frameStart := 80870 },
  { event := event80917
    frameStart := 80870 },
  { event := event80918
    frameStart := 80870 },
  { event := event80919
    frameStart := 80870 },
  { event := event80920
    frameStart := 80870 },
  { event := event80921
    frameStart := 80870 },
  { event := event80922
    frameStart := 80870 },
  { event := event80923
    frameStart := 80870 },
  { event := event80924
    frameStart := 80870 },
  { event := event80925
    frameStart := 80870 },
  { event := event80926
    frameStart := 80870 },
  { event := event80927
    frameStart := 80870 }
]

def eventLeaf5058 : Array AnnotatedEvent := #[
  { event := event80928
    frameStart := 80870 },
  { event := event80929
    frameStart := 80870 },
  { event := event80930
    frameStart := 80870 },
  { event := event80931
    frameStart := 80870 },
  { event := event80932
    frameStart := 80870 },
  { event := event80933
    frameStart := 80870 },
  { event := event80934
    frameStart := 80870 },
  { event := event80935
    frameStart := 80870 },
  { event := event80936
    frameStart := 80870 },
  { event := event80937
    frameStart := 80870 },
  { event := event80938
    frameStart := 80870 },
  { event := event80939
    frameStart := 80870 },
  { event := event80940
    frameStart := 80870 },
  { event := event80941
    frameStart := 80870 },
  { event := event80942
    frameStart := 80870 },
  { event := event80943
    frameStart := 80870 }
]

def eventLeaf5059 : Array AnnotatedEvent := #[
  { event := event80944
    frameStart := 80870 },
  { event := event80945
    frameStart := 80870 },
  { event := event80946
    frameStart := 80870 },
  { event := event80947
    frameStart := 80870 },
  { event := event80948
    frameStart := 80870 },
  { event := event80949
    frameStart := 80870 },
  { event := event80950
    frameStart := 80870 },
  { event := event80951
    frameStart := 80870 },
  { event := event80952
    frameStart := 80870 },
  { event := event80953
    frameStart := 80870 },
  { event := event80954
    frameStart := 80870 },
  { event := event80955
    frameStart := 80870 },
  { event := event80956
    frameStart := 80870 },
  { event := event80957
    frameStart := 80870 },
  { event := event80958
    frameStart := 80870 },
  { event := event80959
    frameStart := 80870 }
]

def eventLeaf5060 : Array AnnotatedEvent := #[
  { event := event80960
    frameStart := 80870 },
  { event := event80961
    frameStart := 80870 },
  { event := event80962
    frameStart := 80870 },
  { event := event80963
    frameStart := 80870 },
  { event := event80964
    frameStart := 80870 },
  { event := event80965
    frameStart := 80870 },
  { event := event80966
    frameStart := 80870 },
  { event := event80967
    frameStart := 80870 },
  { event := event80968
    frameStart := 80870 },
  { event := event80969
    frameStart := 80870 },
  { event := event80970
    frameStart := 80870 },
  { event := event80971
    frameStart := 80870 },
  { event := event80972
    frameStart := 80870 },
  { event := event80973
    frameStart := 80870 },
  { event := event80974
    frameStart := 80870 },
  { event := event80975
    frameStart := 80870 }
]

def eventLeaf5061 : Array AnnotatedEvent := #[
  { event := event80976
    frameStart := 80870 },
  { event := event80977
    frameStart := 80870 },
  { event := event80978
    frameStart := 80870 },
  { event := event80979
    frameStart := 80870 },
  { event := event80980
    frameStart := 80870 },
  { event := event80981
    frameStart := 80870 },
  { event := event80982
    frameStart := 80870 },
  { event := event80983
    frameStart := 80870 },
  { event := event80984
    frameStart := 80870 },
  { event := event80985
    frameStart := 80870 },
  { event := event80986
    frameStart := 80870 },
  { event := event80987
    frameStart := 80870 },
  { event := event80988
    frameStart := 0 },
  { event := event80989
    frameStart := 0 },
  { event := event80990
    frameStart := 0 },
  { event := event80991
    frameStart := 0 }
]

def eventLeaf5062 : Array AnnotatedEvent := #[
  { event := event80992
    frameStart := 0 },
  { event := event80993
    frameStart := 0 },
  { event := event80994
    frameStart := 0 },
  { event := event80995
    frameStart := 0 },
  { event := event80996
    frameStart := 0 },
  { event := event80997
    frameStart := 0 },
  { event := event80998
    frameStart := 0 },
  { event := event80999
    frameStart := 0 },
  { event := event81000
    frameStart := 0 },
  { event := event81001
    frameStart := 0 },
  { event := event81002
    frameStart := 0 },
  { event := event81003
    frameStart := 0 },
  { event := event81004
    frameStart := 0 },
  { event := event81005
    frameStart := 0 },
  { event := event81006
    frameStart := 0 },
  { event := event81007
    frameStart := 0 }
]

def eventLeaf5063 : Array AnnotatedEvent := #[
  { event := event81008
    frameStart := 0 },
  { event := event81009
    frameStart := 0 },
  { event := event81010
    frameStart := 0 },
  { event := event81011
    frameStart := 0 },
  { event := event81012
    frameStart := 0 },
  { event := event81013
    frameStart := 0 },
  { event := event81014
    frameStart := 0 },
  { event := event81015
    frameStart := 0 },
  { event := event81016
    frameStart := 0 },
  { event := event81017
    frameStart := 0 },
  { event := event81018
    frameStart := 0 },
  { event := event81019
    frameStart := 0 },
  { event := event81020
    frameStart := 0 },
  { event := event81021
    frameStart := 0 },
  { event := event81022
    frameStart := 0 },
  { event := event81023
    frameStart := 0 }
]

def eventLeaf5064 : Array AnnotatedEvent := #[
  { event := event81024
    frameStart := 0 },
  { event := event81025
    frameStart := 81025 },
  { event := event81026
    frameStart := 81025 },
  { event := event81027
    frameStart := 81025 },
  { event := event81028
    frameStart := 81025 },
  { event := event81029
    frameStart := 81025 },
  { event := event81030
    frameStart := 81025 },
  { event := event81031
    frameStart := 81025 },
  { event := event81032
    frameStart := 81025 },
  { event := event81033
    frameStart := 81025 },
  { event := event81034
    frameStart := 81025 },
  { event := event81035
    frameStart := 81025 },
  { event := event81036
    frameStart := 81025 },
  { event := event81037
    frameStart := 81025 },
  { event := event81038
    frameStart := 81025 },
  { event := event81039
    frameStart := 81025 }
]

def eventLeaf5065 : Array AnnotatedEvent := #[
  { event := event81040
    frameStart := 81025 },
  { event := event81041
    frameStart := 81025 },
  { event := event81042
    frameStart := 81025 },
  { event := event81043
    frameStart := 81025 },
  { event := event81044
    frameStart := 81025 },
  { event := event81045
    frameStart := 81025 },
  { event := event81046
    frameStart := 81025 },
  { event := event81047
    frameStart := 81025 },
  { event := event81048
    frameStart := 81025 },
  { event := event81049
    frameStart := 81025 },
  { event := event81050
    frameStart := 81025 },
  { event := event81051
    frameStart := 81025 },
  { event := event81052
    frameStart := 81025 },
  { event := event81053
    frameStart := 81025 },
  { event := event81054
    frameStart := 81025 },
  { event := event81055
    frameStart := 81025 }
]

def eventLeaf5066 : Array AnnotatedEvent := #[
  { event := event81056
    frameStart := 81025 },
  { event := event81057
    frameStart := 81025 },
  { event := event81058
    frameStart := 81025 },
  { event := event81059
    frameStart := 81025 },
  { event := event81060
    frameStart := 81025 },
  { event := event81061
    frameStart := 81025 },
  { event := event81062
    frameStart := 81025 },
  { event := event81063
    frameStart := 81025 },
  { event := event81064
    frameStart := 81025 },
  { event := event81065
    frameStart := 81025 },
  { event := event81066
    frameStart := 81025 },
  { event := event81067
    frameStart := 81025 },
  { event := event81068
    frameStart := 81025 },
  { event := event81069
    frameStart := 81025 },
  { event := event81070
    frameStart := 81025 },
  { event := event81071
    frameStart := 81025 }
]

def eventLeaf5067 : Array AnnotatedEvent := #[
  { event := event81072
    frameStart := 81025 },
  { event := event81073
    frameStart := 81025 },
  { event := event81074
    frameStart := 81025 },
  { event := event81075
    frameStart := 81025 },
  { event := event81076
    frameStart := 81025 },
  { event := event81077
    frameStart := 81025 },
  { event := event81078
    frameStart := 81025 },
  { event := event81079
    frameStart := 81079 },
  { event := event81080
    frameStart := 81079 },
  { event := event81081
    frameStart := 81079 },
  { event := event81082
    frameStart := 81079 },
  { event := event81083
    frameStart := 81079 },
  { event := event81084
    frameStart := 81079 },
  { event := event81085
    frameStart := 81079 },
  { event := event81086
    frameStart := 81079 },
  { event := event81087
    frameStart := 81079 }
]

def eventLeaf5068 : Array AnnotatedEvent := #[
  { event := event81088
    frameStart := 81079 },
  { event := event81089
    frameStart := 81079 },
  { event := event81090
    frameStart := 81079 },
  { event := event81091
    frameStart := 81079 },
  { event := event81092
    frameStart := 81079 },
  { event := event81093
    frameStart := 81079 },
  { event := event81094
    frameStart := 81079 },
  { event := event81095
    frameStart := 81079 },
  { event := event81096
    frameStart := 81079 },
  { event := event81097
    frameStart := 81079 },
  { event := event81098
    frameStart := 81079 },
  { event := event81099
    frameStart := 81079 },
  { event := event81100
    frameStart := 81079 },
  { event := event81101
    frameStart := 81079 },
  { event := event81102
    frameStart := 81079 },
  { event := event81103
    frameStart := 81079 }
]

def eventLeaf5069 : Array AnnotatedEvent := #[
  { event := event81104
    frameStart := 81079 },
  { event := event81105
    frameStart := 81079 },
  { event := event81106
    frameStart := 81079 },
  { event := event81107
    frameStart := 81079 },
  { event := event81108
    frameStart := 81079 },
  { event := event81109
    frameStart := 81079 },
  { event := event81110
    frameStart := 81079 },
  { event := event81111
    frameStart := 81079 },
  { event := event81112
    frameStart := 81079 },
  { event := event81113
    frameStart := 81079 },
  { event := event81114
    frameStart := 81079 },
  { event := event81115
    frameStart := 81079 },
  { event := event81116
    frameStart := 81079 },
  { event := event81117
    frameStart := 81079 },
  { event := event81118
    frameStart := 81079 },
  { event := event81119
    frameStart := 81079 }
]

def eventLeaf5070 : Array AnnotatedEvent := #[
  { event := event81120
    frameStart := 81079 },
  { event := event81121
    frameStart := 81079 },
  { event := event81122
    frameStart := 81079 },
  { event := event81123
    frameStart := 81079 },
  { event := event81124
    frameStart := 81079 },
  { event := event81125
    frameStart := 81079 },
  { event := event81126
    frameStart := 81079 },
  { event := event81127
    frameStart := 81079 },
  { event := event81128
    frameStart := 81079 },
  { event := event81129
    frameStart := 81079 },
  { event := event81130
    frameStart := 81079 },
  { event := event81131
    frameStart := 81079 },
  { event := event81132
    frameStart := 81079 },
  { event := event81133
    frameStart := 81079 },
  { event := event81134
    frameStart := 81079 },
  { event := event81135
    frameStart := 81079 }
]

def eventLeaf5071 : Array AnnotatedEvent := #[
  { event := event81136
    frameStart := 81079 },
  { event := event81137
    frameStart := 81079 },
  { event := event81138
    frameStart := 81079 },
  { event := event81139
    frameStart := 81079 },
  { event := event81140
    frameStart := 81079 },
  { event := event81141
    frameStart := 81079 },
  { event := event81142
    frameStart := 81079 },
  { event := event81143
    frameStart := 81079 },
  { event := event81144
    frameStart := 81079 },
  { event := event81145
    frameStart := 81079 },
  { event := event81146
    frameStart := 81079 },
  { event := event81147
    frameStart := 81079 },
  { event := event81148
    frameStart := 81079 },
  { event := event81149
    frameStart := 81079 },
  { event := event81150
    frameStart := 81079 },
  { event := event81151
    frameStart := 81079 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events316
