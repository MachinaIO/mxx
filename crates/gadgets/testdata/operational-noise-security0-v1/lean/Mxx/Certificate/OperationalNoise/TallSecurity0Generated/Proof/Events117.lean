import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events117

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event29952 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14804⟩⟩) 0 ⟨10506⟩ 29951

def event29953 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14804⟩⟩) (.authority (.programFamilyFact))

def exact29954RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14804⟩⟩], []⟩, (1)⟩]

theorem exact29954RawTermsValid :
    exact29954RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event29954 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14804⟩⟩) exact29954RawTerms (.finite 2) 29953 .exactZero (none)

def event29955 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14805⟩⟩) 0 ⟨14804⟩ 29954

def event29956 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14805⟩⟩) (.identity (.predecessor 0 29955 .coefficient))

def event29957 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14805⟩⟩) (.finite 2)

def event29958 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20404⟩⟩) 0 ⟨14805⟩ 29957

def event29959 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20404⟩⟩) (.authority (.relationPreimageSource ⟨28⟩))

def exact29960RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20404⟩⟩]⟩, (1)⟩]

theorem exact29960RawTermsValid :
    exact29960RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event29960 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20404⟩⟩) exact29960RawTerms (.finite 136065468) 29959 .exactZero (none)

def event29961 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact29962RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact29962RawTermsValid :
    exact29962RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event29962 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact29962RawTerms .large 29961 .exactZero (none)

def event29963 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20405⟩⟩) 0 ⟨6⟩ 29962

def event29964 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20405⟩⟩) 1 ⟨20404⟩ 29960

def event29965 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20405⟩⟩) (.product (.predecessor 0 29963 .coefficient) (.predecessor 1 29964 .coefficient) (⟨false, false, none, none, none⟩))

def event29966 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20405⟩⟩, .operator (⟨29962, 0⟩, ⟨29960, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20404⟩⟩]⟩, (1)⟩)

def exact29967RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20404⟩⟩]⟩, (1)⟩]

theorem exact29967RawTermsValid :
    exact29967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event29967 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20405⟩⟩) exact29967RawTerms .large 29965 .exactZero (none)

def event29968 : Event := .preFoldPolynomial 29967 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20404⟩⟩]⟩, (1)⟩] .exactZero none

def exact29969RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20404⟩⟩]⟩, (1)⟩]

def event29969 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20405⟩⟩) 29968 exact29969RawTerms .large 29965 .exactZero (none)

def event29970 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26398⟩⟩)

def event29971 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event29972 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event29973 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event29974 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event29975 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event29976 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event29977 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event29978 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event29979 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 29978

def event29980 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 29976

def event29981 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 29979 .coefficient) (.value (.predecessor 1 29980 .coefficient)))

def event29982 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event29983 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 29982

def event29984 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 29974

def event29985 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 29983 .coefficient, .predecessor 1 29984 .coefficient])

def event29986 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event29987 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 29986

def event29988 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 29972

def event29989 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 29988 .coefficient))

def event29990 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event29991 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10504⟩⟩) 0 ⟨5554⟩ 29990

def event29992 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10504⟩⟩) (.authority (.programFamilyFact))

def exact29993RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10504⟩⟩], []⟩, (1)⟩]

theorem exact29993RawTermsValid :
    exact29993RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event29993 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10504⟩⟩) exact29993RawTerms (.finite 2) 29992 .exactZero (none)

def event29994 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9415⟩⟩) 0 ⟨5554⟩ 29990

def event29995 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9415⟩⟩) (.authority (.programFamilyFact))

def exact29996RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9415⟩⟩], []⟩, (1)⟩]

theorem exact29996RawTermsValid :
    exact29996RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event29996 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9415⟩⟩) exact29996RawTerms (.finite 2) 29995 .exactZero (none)

def event29997 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10505⟩⟩) 0 ⟨9415⟩ 29996

def event29998 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10505⟩⟩) 1 ⟨10504⟩ 29993

def event29999 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10505⟩⟩) (.product (.predecessor 0 29997 .coefficient) (.predecessor 1 29998 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event30000 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10505⟩⟩, .operator (⟨29996, 0⟩, ⟨29993, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9415⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], []⟩, (1)⟩)

def exact30001RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9415⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], []⟩, (1)⟩]

theorem exact30001RawTermsValid :
    exact30001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30001 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10505⟩⟩) exact30001RawTerms (.finite 4) 29999 .exactZero (none)

def event30002 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10506⟩⟩) 0 ⟨10505⟩ 30001

def event30003 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10506⟩⟩) (.identity (.predecessor 0 30002 .coefficient))

def event30004 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10506⟩⟩) (.finite 4)

def event30005 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14804⟩⟩) 0 ⟨10506⟩ 30004

def event30006 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14804⟩⟩) (.authority (.programFamilyFact))

def exact30007RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14804⟩⟩], []⟩, (1)⟩]

theorem exact30007RawTermsValid :
    exact30007RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30007 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14804⟩⟩) exact30007RawTerms (.finite 2) 30006 .exactZero (none)

def event30008 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14805⟩⟩) 0 ⟨14804⟩ 30007

def event30009 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14805⟩⟩) (.identity (.predecessor 0 30008 .coefficient))

def event30010 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14805⟩⟩) (.finite 2)

def event30011 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23728⟩⟩) 0 ⟨14805⟩ 30010

def event30012 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23728⟩⟩) (.authority (.programFamilyFact))

def event30013 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23728⟩⟩) (.finite 3720)

def event30014 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event30015 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23730⟩⟩) 0 ⟨6689⟩ 30014

def event30016 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23730⟩⟩) 1 ⟨23728⟩ 30013

def event30017 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23730⟩⟩) (.authority (.operator))

def exact30018RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23730⟩⟩]⟩, (1)⟩]

theorem exact30018RawTermsValid :
    exact30018RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30018 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23730⟩⟩) exact30018RawTerms .large 30017 .exactZero (none)

def event30019 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26394⟩⟩) 0 ⟨23730⟩ 30018

def event30020 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26394⟩⟩) (.authority (.operator))

def exact30021RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26394⟩⟩]⟩, (1)⟩]

theorem exact30021RawTermsValid :
    exact30021RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30021 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26394⟩⟩) exact30021RawTerms (.finite 8192) 30020 .exactZero (none)

def event30022 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event30023 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event30024 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14844⟩⟩) 0 ⟨14805⟩ 30010

def event30025 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14844⟩⟩) 1 ⟨110⟩ 30023

def event30026 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14844⟩⟩) (.sum [.predecessor 0 30024 .coefficient, .predecessor 1 30025 .coefficient])

def event30027 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14844⟩⟩) (.finite 2)

def event30028 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14845⟩⟩) 0 ⟨14844⟩ 30027

def event30029 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14845⟩⟩) (.identity (.predecessor 0 30028 .coefficient))

def exact30030RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14804⟩⟩], []⟩, (1)⟩]

theorem exact30030RawTermsValid :
    exact30030RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30030 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14845⟩⟩) exact30030RawTerms (.finite 2) 30029 .exactZero (none)

def event30031 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact30032RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact30032RawTermsValid :
    exact30032RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30032 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact30032RawTerms .large 30031 .exactZero (none)

def event30033 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14846⟩⟩) 0 ⟨6544⟩ 30032

def event30034 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14846⟩⟩) 1 ⟨14845⟩ 30030

def event30035 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14846⟩⟩) (.product (.predecessor 0 30033 .coefficient) (.predecessor 1 30034 .coefficient) (⟨false, false, none, none, none⟩))

def event30036 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14846⟩⟩, .operator (⟨30032, 0⟩, ⟨30030, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14804⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact30037RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14804⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact30037RawTermsValid :
    exact30037RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30037 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14846⟩⟩) exact30037RawTerms .large 30035 .exactZero (none)

def event30038 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6690⟩⟩) 0 ⟨6689⟩ 30014

def event30039 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6690⟩⟩) (.authority (.operator))

def exact30040RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩]

theorem exact30040RawTermsValid :
    exact30040RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30040 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6690⟩⟩) exact30040RawTerms .large 30039 .exactZero (none)

def event30041 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14847⟩⟩) 0 ⟨6690⟩ 30040

def event30042 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14847⟩⟩) 1 ⟨14846⟩ 30037

def event30043 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14847⟩⟩) (.sum [.predecessor 0 30041 .coefficient, .predecessor 1 30042 .coefficient])

def exact30044RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14804⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30044RawTermsValid :
    exact30044RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30044 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14847⟩⟩) exact30044RawTerms .large 30043 .exactZero (none)

def event30045 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26395⟩⟩) 0 ⟨14847⟩ 30044

def event30046 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26395⟩⟩) 1 ⟨26394⟩ 30021

def event30047 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26395⟩⟩) (.product (.predecessor 0 30045 .coefficient) (.predecessor 1 30046 .coefficient) (⟨false, false, none, none, none⟩))

def event30048 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26395⟩⟩, .operator (⟨30044, 0⟩, ⟨30021, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26394⟩⟩]⟩, (1)⟩)

def event30049 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26395⟩⟩, .operator (⟨30044, 1⟩, ⟨30021, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14804⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26394⟩⟩]⟩, (-1)⟩)

def event30050 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26395⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨14804⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26394⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26394⟩⟩) ⟨23730⟩ 30018)

def event30051 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26395⟩⟩, .relation 30050 0, ⟨[⟨.program ⟨214⟩, ⟨14804⟩⟩], [⟨.program ⟨214⟩, ⟨23730⟩⟩]⟩, (-1)⟩)

def exact30052RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26394⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14804⟩⟩], [⟨.program ⟨214⟩, ⟨23730⟩⟩]⟩, (-1)⟩]

theorem exact30052RawTermsValid :
    exact30052RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30052 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26395⟩⟩) exact30052RawTerms .large 30047 .exactZero (none)

def event30053 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15274⟩⟩) 0 ⟨14805⟩ 30010

def event30054 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15274⟩⟩) (.authority (.programFamilyFact))

def exact30055RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15274⟩⟩], []⟩, (1)⟩]

theorem exact30055RawTermsValid :
    exact30055RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30055 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15274⟩⟩) exact30055RawTerms (.finite 43) 30054 .exactZero (none)

def event30056 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15275⟩⟩) 0 ⟨6544⟩ 30032

def event30057 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15275⟩⟩) 1 ⟨15274⟩ 30055

def event30058 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15275⟩⟩) (.product (.predecessor 0 30056 .coefficient) (.predecessor 1 30057 .coefficient) (⟨false, true, none, none, some 1⟩))

def event30059 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15275⟩⟩, .operator (⟨30032, 0⟩, ⟨30055, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact30060RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact30060RawTermsValid :
    exact30060RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30060 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15275⟩⟩) exact30060RawTerms .large 30058 .exactZero (none)

def event30061 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6709⟩⟩) 0 ⟨6689⟩ 30014

def event30062 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6709⟩⟩) (.authority (.operator))

def exact30063RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩]

theorem exact30063RawTermsValid :
    exact30063RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30063 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6709⟩⟩) exact30063RawTerms .large 30062 .exactZero (none)

def event30064 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15276⟩⟩) 0 ⟨6709⟩ 30063

def event30065 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15276⟩⟩) 1 ⟨15275⟩ 30060

def event30066 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15276⟩⟩) (.sum [.predecessor 0 30064 .coefficient, .predecessor 1 30065 .coefficient])

def exact30067RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30067RawTermsValid :
    exact30067RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30067 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15276⟩⟩) exact30067RawTerms .large 30066 .exactZero (none)

def event30068 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26398⟩⟩) 0 ⟨15276⟩ 30067

def event30069 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26398⟩⟩) 1 ⟨26395⟩ 30052

def event30070 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26398⟩⟩) (.sum [.predecessor 0 30068 .coefficient, .predecessor 1 30069 .coefficient])

def exact30071RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26394⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14804⟩⟩], [⟨.program ⟨214⟩, ⟨23730⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30071RawTermsValid :
    exact30071RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30071 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26398⟩⟩) exact30071RawTerms .large 30070 .exactZero (none)

def event30072 : Event := .preFoldPolynomial 30071 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26394⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14804⟩⟩], [⟨.program ⟨214⟩, ⟨23730⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact30073RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26394⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14804⟩⟩], [⟨.program ⟨214⟩, ⟨23730⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event30073 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26398⟩⟩) 30072 exact30073RawTerms .large 30070 .exactZero (none)

def event30074 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14805⟩⟩) ⟨⟨122⟩, ⟨28⟩, ⟨109⟩⟩ ⟨29916, 30074⟩

def event30075 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20407⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20404⟩⟩]⟩) (1) 0 2 (.universal 30074 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20404⟩⟩]⟩) (none) 30073)

def event30076 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20407⟩⟩, .relation 30075 1, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩)

def event30077 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20407⟩⟩, .relation 30075 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26394⟩⟩]⟩, (-1)⟩)

def event30078 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20407⟩⟩, .relation 30075 2, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14804⟩⟩], [⟨.program ⟨214⟩, ⟨23730⟩⟩]⟩, (1)⟩)

def event30079 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20407⟩⟩, .relation 30075 3, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact30080RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26394⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14804⟩⟩], [⟨.program ⟨214⟩, ⟨23730⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30080RawTermsValid :
    exact30080RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30080 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20407⟩⟩) exact30080RawTerms .large 29912 (.finite 1811303510016) (some (29914))

def event30081 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26397⟩⟩) 0 ⟨20407⟩ 30080

def event30082 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26397⟩⟩) 1 ⟨26396⟩ 29902

def event30083 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26397⟩⟩) (.sum [.predecessor 0 30081 .coefficient, .predecessor 1 30082 .coefficient])

def event30084 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26397⟩⟩, .operator (⟨30080, 0⟩, ⟨29902, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26394⟩⟩]⟩, (1)⟩)

def event30085 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26397⟩⟩, .operator (⟨30080, 2⟩, ⟨29902, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14804⟩⟩], [⟨.program ⟨214⟩, ⟨23730⟩⟩]⟩, (-1)⟩)

def event30086 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26397⟩⟩) (.sum [.result 30080 .summary, .result 29902 .summary])

def exact30087RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30087RawTermsValid :
    exact30087RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30087 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26397⟩⟩) exact30087RawTerms .large 30083 (.finite 1291889174379421642752) (some (30086))

def event30088 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26607⟩⟩) 0 ⟨26397⟩ 30087

def event30089 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26607⟩⟩) 1 ⟨26606⟩ 29605

def event30090 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26607⟩⟩) (.sum [.predecessor 0 30088 .coefficient, .predecessor 1 30089 .coefficient])

def event30091 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26607⟩⟩) (.sum [.result 30087 .summary, .result 29605 .summary])

def exact30092RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30092RawTermsValid :
    exact30092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30092 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26607⟩⟩) exact30092RawTerms .large 30090 (.finite 2583789554981353578496) (some (30091))

def event30093 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26824⟩⟩) 0 ⟨26607⟩ 30092

def event30094 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26824⟩⟩) 1 ⟨26823⟩ 29123

def event30095 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26824⟩⟩) (.sum [.predecessor 0 30093 .coefficient, .predecessor 1 30094 .coefficient])

def event30096 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26824⟩⟩) (.sum [.result 30092 .summary, .result 29123 .summary])

def exact30097RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30097RawTermsValid :
    exact30097RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30097 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26824⟩⟩) exact30097RawTerms .large 30095 (.finite 3875701141805795807232) (some (30096))

def event30098 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27041⟩⟩) 0 ⟨26824⟩ 30097

def event30099 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27041⟩⟩) 1 ⟨27040⟩ 28641

def event30100 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27041⟩⟩) (.sum [.predecessor 0 30098 .coefficient, .predecessor 1 30099 .coefficient])

def event30101 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27041⟩⟩) (.sum [.result 30097 .summary, .result 28641 .summary])

def exact30102RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30102RawTermsValid :
    exact30102RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30102 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27041⟩⟩) exact30102RawTerms .large 30100 (.finite 5167635141075258621952) (some (30101))

def event30103 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27258⟩⟩) 0 ⟨27041⟩ 30102

def event30104 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27258⟩⟩) 1 ⟨27257⟩ 28159

def event30105 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27258⟩⟩) (.sum [.predecessor 0 30103 .coefficient, .predecessor 1 30104 .coefficient])

def event30106 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27258⟩⟩) (.sum [.result 30102 .summary, .result 28159 .summary])

def exact30107RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30107RawTermsValid :
    exact30107RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30107 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27258⟩⟩) exact30107RawTerms .large 30105 (.finite 6459613965234762608640) (some (30106))

def event30108 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27475⟩⟩) 0 ⟨27258⟩ 30107

def event30109 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27475⟩⟩) 1 ⟨27474⟩ 27677

def event30110 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27475⟩⟩) (.sum [.predecessor 0 30108 .coefficient, .predecessor 1 30109 .coefficient])

def event30111 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27475⟩⟩) (.sum [.result 30107 .summary, .result 27677 .summary])

def exact30112RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30112RawTermsValid :
    exact30112RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30112 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27475⟩⟩) exact30112RawTerms .large 30110 (.finite 7751615201839287181312) (some (30111))

def event30113 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27692⟩⟩) 0 ⟨27475⟩ 30112

def event30114 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27692⟩⟩) 1 ⟨27691⟩ 27195

def event30115 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27692⟩⟩) (.sum [.predecessor 0 30113 .coefficient, .predecessor 1 30114 .coefficient])

def event30116 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27692⟩⟩) (.sum [.result 30112 .summary, .result 27195 .summary])

def exact30117RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30117RawTermsValid :
    exact30117RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30117 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27692⟩⟩) exact30117RawTerms .large 30115 (.finite 9043661263333852925952) (some (30116))

def event30118 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27909⟩⟩) 0 ⟨27692⟩ 30117

def event30119 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27909⟩⟩) 1 ⟨27908⟩ 26713

def event30120 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27909⟩⟩) (.sum [.predecessor 0 30118 .coefficient, .predecessor 1 30119 .coefficient])

def event30121 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27909⟩⟩) (.sum [.result 30117 .summary, .result 26713 .summary])

def exact30122RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30122RawTermsValid :
    exact30122RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30122 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27909⟩⟩) exact30122RawTerms .large 30120 (.finite 10335729737273439256576) (some (30121))

def event30123 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28126⟩⟩) 0 ⟨27909⟩ 30122

def event30124 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28126⟩⟩) 1 ⟨28125⟩ 26231

def event30125 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28126⟩⟩) (.sum [.predecessor 0 30123 .coefficient, .predecessor 1 30124 .coefficient])

def event30126 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28126⟩⟩) (.sum [.result 30122 .summary, .result 26231 .summary])

def exact30127RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30127RawTermsValid :
    exact30127RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30127 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28126⟩⟩) exact30127RawTerms .large 30125 (.finite 11627843036103066759168) (some (30126))

def event30128 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28343⟩⟩) 0 ⟨28126⟩ 30127

def event30129 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28343⟩⟩) 1 ⟨28342⟩ 25749

def event30130 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28343⟩⟩) (.sum [.predecessor 0 30128 .coefficient, .predecessor 1 30129 .coefficient])

def event30131 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28343⟩⟩) (.sum [.result 30127 .summary, .result 25749 .summary])

def exact30132RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18379⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30132RawTermsValid :
    exact30132RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30132 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28343⟩⟩) exact30132RawTerms .large 30130 (.finite 12920023572267756019712) (some (30131))

def event30133 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28560⟩⟩) 0 ⟨28343⟩ 30132

def event30134 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28560⟩⟩) 1 ⟨28559⟩ 25267

def event30135 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28560⟩⟩) (.sum [.predecessor 0 30133 .coefficient, .predecessor 1 30134 .coefficient])

def event30136 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28560⟩⟩) (.sum [.result 30132 .summary, .result 25267 .summary])

def exact30137RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16317⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18379⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30137RawTermsValid :
    exact30137RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30137 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28560⟩⟩) exact30137RawTerms .large 30135 (.finite 14212226520877465866240) (some (30136))

def event30138 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28777⟩⟩) 0 ⟨28560⟩ 30137

def event30139 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28777⟩⟩) 1 ⟨28776⟩ 24785

def event30140 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28777⟩⟩) (.sum [.predecessor 0 30138 .coefficient, .predecessor 1 30139 .coefficient])

def event30141 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28777⟩⟩) (.sum [.result 30137 .summary, .result 24785 .summary])

def exact30142RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16317⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18379⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30142RawTermsValid :
    exact30142RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30142 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28777⟩⟩) exact30142RawTerms .large 30140 (.finite 15504496706822237470720) (some (30141))

def event30143 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28994⟩⟩) 0 ⟨28777⟩ 30142

def event30144 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28994⟩⟩) 1 ⟨28993⟩ 24303

def event30145 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28994⟩⟩) (.sum [.predecessor 0 30143 .coefficient, .predecessor 1 30144 .coefficient])

def event30146 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28994⟩⟩) (.sum [.result 30142 .summary, .result 24303 .summary])

def exact30147RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16317⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17913⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18379⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30147RawTermsValid :
    exact30147RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30147 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28994⟩⟩) exact30147RawTerms .large 30145 (.finite 16796811717657050247168) (some (30146))

def event30148 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29211⟩⟩) 0 ⟨28994⟩ 30147

def event30149 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29211⟩⟩) 1 ⟨29210⟩ 23821

def event30150 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29211⟩⟩) (.sum [.predecessor 0 30148 .coefficient, .predecessor 1 30149 .coefficient])

def event30151 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29211⟩⟩) (.sum [.result 30147 .summary, .result 23821 .summary])

def exact30152RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16317⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17913⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18214⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18379⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30152RawTermsValid :
    exact30152RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30152 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29211⟩⟩) exact30152RawTerms .large 30150 (.finite 18089149140936883609600) (some (30151))

def event30153 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29428⟩⟩) 0 ⟨29211⟩ 30152

def event30154 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29428⟩⟩) 1 ⟨29427⟩ 23339

def event30155 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29428⟩⟩) (.sum [.predecessor 0 30153 .coefficient, .predecessor 1 30154 .coefficient])

def event30156 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29428⟩⟩) (.sum [.result 30152 .summary, .result 23339 .summary])

def exact30157RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16317⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16688⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17913⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18214⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18379⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30157RawTermsValid :
    exact30157RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30157 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29428⟩⟩) exact30157RawTerms .large 30155 (.finite 19381531389106758144000) (some (30156))

def event30158 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29645⟩⟩) 0 ⟨29428⟩ 30157

def event30159 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29645⟩⟩) 1 ⟨29644⟩ 22857

def event30160 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29645⟩⟩) (.sum [.predecessor 0 30158 .coefficient, .predecessor 1 30159 .coefficient])

def event30161 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29645⟩⟩) (.sum [.result 30157 .summary, .result 22857 .summary])

def exact30162RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16317⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16688⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16807⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17913⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18214⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18379⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30162RawTermsValid :
    exact30162RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30162 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29645⟩⟩) exact30162RawTerms .large 30160 (.finite 20673980874611694436352) (some (30161))

def event30163 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29862⟩⟩) 0 ⟨29645⟩ 30162

def event30164 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29862⟩⟩) 1 ⟨29861⟩ 22375

def event30165 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29862⟩⟩) (.sum [.predecessor 0 30163 .coefficient, .predecessor 1 30164 .coefficient])

def event30166 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29862⟩⟩) (.sum [.result 30162 .summary, .result 22375 .summary])

def exact30167RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16317⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16688⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16807⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17094⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17913⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18214⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18379⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30167RawTermsValid :
    exact30167RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30167 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29862⟩⟩) exact30167RawTerms .large 30165 (.finite 21966497597451692486656) (some (30166))

def event30168 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30187⟩⟩) 0 ⟨29862⟩ 30167

def event30169 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30187⟩⟩) 1 ⟨30186⟩ 21893

def event30170 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30187⟩⟩) (.sum [.predecessor 0 30168 .coefficient, .predecessor 1 30169 .coefficient])

def event30171 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30187⟩⟩) (.sum [.result 30167 .summary, .result 21893 .summary])

def exact30172RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15378⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15638⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15757⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15995⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16114⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16317⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16688⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16807⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17094⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17354⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17913⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18179⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18214⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18379⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact30172RawTermsValid :
    exact30172RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event30172 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30187⟩⟩) exact30172RawTerms .large 30170 (.finite 23259036732736711122944) (some (30171))

def event30173 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30188⟩⟩) 0 ⟨30187⟩ 30172

def event30174 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30188⟩⟩) 1 ⟨18690⟩ 21395

def event30175 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30188⟩⟩) (.product (.predecessor 0 30173 .coefficient) (.predecessor 1 30174 .coefficient) (⟨false, false, none, none, none⟩))

def event30176 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30188⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) [⟨.result 21395 .coefficient, false, none⟩])

def event30177 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30188⟩⟩) (.product (.result 30172 .summary) (.transfer 30176) (⟨false, false, none, none, none⟩))

def event30178 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .operator (⟨30172, 17⟩, ⟨21395, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩, (1)⟩)

def event30179 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .operator (⟨30172, 33⟩, ⟨21395, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18179⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩, (-1)⟩)

def event30180 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30188⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18179⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 21392)

def event30181 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .relation 30180 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18179⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩, (-1)⟩)

def event30182 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .operator (⟨30172, 16⟩, ⟨21395, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩, (1)⟩)

def event30183 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .operator (⟨30172, 29⟩, ⟨21395, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17094⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩, (-1)⟩)

def event30184 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30188⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17094⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 21392)

def event30185 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .relation 30184 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17094⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩, (-1)⟩)

def event30186 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .operator (⟨30172, 15⟩, ⟨21395, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩, (1)⟩)

def event30187 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .operator (⟨30172, 28⟩, ⟨21395, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16807⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩, (-1)⟩)

def event30188 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30188⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16807⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 21392)

def event30189 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .relation 30188 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16807⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩, (-1)⟩)

def event30190 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .operator (⟨30172, 14⟩, ⟨21395, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩, (1)⟩)

def event30191 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .operator (⟨30172, 27⟩, ⟨21395, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16688⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩, (-1)⟩)

def event30192 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30188⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16688⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 21392)

def event30193 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .relation 30192 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16688⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩, (-1)⟩)

def event30194 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .operator (⟨30172, 13⟩, ⟨21395, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩, (1)⟩)

def event30195 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .operator (⟨30172, 34⟩, ⟨21395, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18214⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩, (-1)⟩)

def event30196 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30188⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18214⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 21392)

def event30197 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .relation 30196 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18214⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩, (-1)⟩)

def event30198 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .operator (⟨30172, 12⟩, ⟨21395, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩, (1)⟩)

def event30199 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .operator (⟨30172, 32⟩, ⟨21395, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17913⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩, (-1)⟩)

def event30200 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30188⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17913⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 21392)

def event30201 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .relation 30200 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17913⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩, (-1)⟩)

def event30202 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .operator (⟨30172, 11⟩, ⟨21395, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩, (1)⟩)

def event30203 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .operator (⟨30172, 30⟩, ⟨21395, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩, (-1)⟩)

def event30204 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30188⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18690⟩⟩) ⟨18624⟩ 21392)

def event30205 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .relation 30204 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨18624⟩⟩]⟩, (-1)⟩)

def event30206 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .operator (⟨30172, 10⟩, ⟨21395, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩, (1)⟩)

def event30207 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30188⟩⟩, .operator (⟨30172, 26⟩, ⟨21395, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16317⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩, (-1)⟩)

def eventLeaf1872 : Array AnnotatedEvent := #[
  { event := event29952
    frameStart := 29916 },
  { event := event29953
    frameStart := 29916 },
  { event := event29954
    frameStart := 29916 },
  { event := event29955
    frameStart := 29916 },
  { event := event29956
    frameStart := 29916 },
  { event := event29957
    frameStart := 29916 },
  { event := event29958
    frameStart := 29916 },
  { event := event29959
    frameStart := 29916 },
  { event := event29960
    frameStart := 29916 },
  { event := event29961
    frameStart := 29916 },
  { event := event29962
    frameStart := 29916 },
  { event := event29963
    frameStart := 29916 },
  { event := event29964
    frameStart := 29916 },
  { event := event29965
    frameStart := 29916 },
  { event := event29966
    frameStart := 29916 },
  { event := event29967
    frameStart := 29916 }
]

def eventLeaf1873 : Array AnnotatedEvent := #[
  { event := event29968
    frameStart := 29916 },
  { event := event29969
    frameStart := 29916 },
  { event := event29970
    frameStart := 29970 },
  { event := event29971
    frameStart := 29970 },
  { event := event29972
    frameStart := 29970 },
  { event := event29973
    frameStart := 29970 },
  { event := event29974
    frameStart := 29970 },
  { event := event29975
    frameStart := 29970 },
  { event := event29976
    frameStart := 29970 },
  { event := event29977
    frameStart := 29970 },
  { event := event29978
    frameStart := 29970 },
  { event := event29979
    frameStart := 29970 },
  { event := event29980
    frameStart := 29970 },
  { event := event29981
    frameStart := 29970 },
  { event := event29982
    frameStart := 29970 },
  { event := event29983
    frameStart := 29970 }
]

def eventLeaf1874 : Array AnnotatedEvent := #[
  { event := event29984
    frameStart := 29970 },
  { event := event29985
    frameStart := 29970 },
  { event := event29986
    frameStart := 29970 },
  { event := event29987
    frameStart := 29970 },
  { event := event29988
    frameStart := 29970 },
  { event := event29989
    frameStart := 29970 },
  { event := event29990
    frameStart := 29970 },
  { event := event29991
    frameStart := 29970 },
  { event := event29992
    frameStart := 29970 },
  { event := event29993
    frameStart := 29970 },
  { event := event29994
    frameStart := 29970 },
  { event := event29995
    frameStart := 29970 },
  { event := event29996
    frameStart := 29970 },
  { event := event29997
    frameStart := 29970 },
  { event := event29998
    frameStart := 29970 },
  { event := event29999
    frameStart := 29970 }
]

def eventLeaf1875 : Array AnnotatedEvent := #[
  { event := event30000
    frameStart := 29970 },
  { event := event30001
    frameStart := 29970 },
  { event := event30002
    frameStart := 29970 },
  { event := event30003
    frameStart := 29970 },
  { event := event30004
    frameStart := 29970 },
  { event := event30005
    frameStart := 29970 },
  { event := event30006
    frameStart := 29970 },
  { event := event30007
    frameStart := 29970 },
  { event := event30008
    frameStart := 29970 },
  { event := event30009
    frameStart := 29970 },
  { event := event30010
    frameStart := 29970 },
  { event := event30011
    frameStart := 29970 },
  { event := event30012
    frameStart := 29970 },
  { event := event30013
    frameStart := 29970 },
  { event := event30014
    frameStart := 29970 },
  { event := event30015
    frameStart := 29970 }
]

def eventLeaf1876 : Array AnnotatedEvent := #[
  { event := event30016
    frameStart := 29970 },
  { event := event30017
    frameStart := 29970 },
  { event := event30018
    frameStart := 29970 },
  { event := event30019
    frameStart := 29970 },
  { event := event30020
    frameStart := 29970 },
  { event := event30021
    frameStart := 29970 },
  { event := event30022
    frameStart := 29970 },
  { event := event30023
    frameStart := 29970 },
  { event := event30024
    frameStart := 29970 },
  { event := event30025
    frameStart := 29970 },
  { event := event30026
    frameStart := 29970 },
  { event := event30027
    frameStart := 29970 },
  { event := event30028
    frameStart := 29970 },
  { event := event30029
    frameStart := 29970 },
  { event := event30030
    frameStart := 29970 },
  { event := event30031
    frameStart := 29970 }
]

def eventLeaf1877 : Array AnnotatedEvent := #[
  { event := event30032
    frameStart := 29970 },
  { event := event30033
    frameStart := 29970 },
  { event := event30034
    frameStart := 29970 },
  { event := event30035
    frameStart := 29970 },
  { event := event30036
    frameStart := 29970 },
  { event := event30037
    frameStart := 29970 },
  { event := event30038
    frameStart := 29970 },
  { event := event30039
    frameStart := 29970 },
  { event := event30040
    frameStart := 29970 },
  { event := event30041
    frameStart := 29970 },
  { event := event30042
    frameStart := 29970 },
  { event := event30043
    frameStart := 29970 },
  { event := event30044
    frameStart := 29970 },
  { event := event30045
    frameStart := 29970 },
  { event := event30046
    frameStart := 29970 },
  { event := event30047
    frameStart := 29970 }
]

def eventLeaf1878 : Array AnnotatedEvent := #[
  { event := event30048
    frameStart := 29970 },
  { event := event30049
    frameStart := 29970 },
  { event := event30050
    frameStart := 29970 },
  { event := event30051
    frameStart := 29970 },
  { event := event30052
    frameStart := 29970 },
  { event := event30053
    frameStart := 29970 },
  { event := event30054
    frameStart := 29970 },
  { event := event30055
    frameStart := 29970 },
  { event := event30056
    frameStart := 29970 },
  { event := event30057
    frameStart := 29970 },
  { event := event30058
    frameStart := 29970 },
  { event := event30059
    frameStart := 29970 },
  { event := event30060
    frameStart := 29970 },
  { event := event30061
    frameStart := 29970 },
  { event := event30062
    frameStart := 29970 },
  { event := event30063
    frameStart := 29970 }
]

def eventLeaf1879 : Array AnnotatedEvent := #[
  { event := event30064
    frameStart := 29970 },
  { event := event30065
    frameStart := 29970 },
  { event := event30066
    frameStart := 29970 },
  { event := event30067
    frameStart := 29970 },
  { event := event30068
    frameStart := 29970 },
  { event := event30069
    frameStart := 29970 },
  { event := event30070
    frameStart := 29970 },
  { event := event30071
    frameStart := 29970 },
  { event := event30072
    frameStart := 29970 },
  { event := event30073
    frameStart := 29970 },
  { event := event30074
    frameStart := 0 },
  { event := event30075
    frameStart := 0 },
  { event := event30076
    frameStart := 0 },
  { event := event30077
    frameStart := 0 },
  { event := event30078
    frameStart := 0 },
  { event := event30079
    frameStart := 0 }
]

def eventLeaf1880 : Array AnnotatedEvent := #[
  { event := event30080
    frameStart := 0 },
  { event := event30081
    frameStart := 0 },
  { event := event30082
    frameStart := 0 },
  { event := event30083
    frameStart := 0 },
  { event := event30084
    frameStart := 0 },
  { event := event30085
    frameStart := 0 },
  { event := event30086
    frameStart := 0 },
  { event := event30087
    frameStart := 0 },
  { event := event30088
    frameStart := 0 },
  { event := event30089
    frameStart := 0 },
  { event := event30090
    frameStart := 0 },
  { event := event30091
    frameStart := 0 },
  { event := event30092
    frameStart := 0 },
  { event := event30093
    frameStart := 0 },
  { event := event30094
    frameStart := 0 },
  { event := event30095
    frameStart := 0 }
]

def eventLeaf1881 : Array AnnotatedEvent := #[
  { event := event30096
    frameStart := 0 },
  { event := event30097
    frameStart := 0 },
  { event := event30098
    frameStart := 0 },
  { event := event30099
    frameStart := 0 },
  { event := event30100
    frameStart := 0 },
  { event := event30101
    frameStart := 0 },
  { event := event30102
    frameStart := 0 },
  { event := event30103
    frameStart := 0 },
  { event := event30104
    frameStart := 0 },
  { event := event30105
    frameStart := 0 },
  { event := event30106
    frameStart := 0 },
  { event := event30107
    frameStart := 0 },
  { event := event30108
    frameStart := 0 },
  { event := event30109
    frameStart := 0 },
  { event := event30110
    frameStart := 0 },
  { event := event30111
    frameStart := 0 }
]

def eventLeaf1882 : Array AnnotatedEvent := #[
  { event := event30112
    frameStart := 0 },
  { event := event30113
    frameStart := 0 },
  { event := event30114
    frameStart := 0 },
  { event := event30115
    frameStart := 0 },
  { event := event30116
    frameStart := 0 },
  { event := event30117
    frameStart := 0 },
  { event := event30118
    frameStart := 0 },
  { event := event30119
    frameStart := 0 },
  { event := event30120
    frameStart := 0 },
  { event := event30121
    frameStart := 0 },
  { event := event30122
    frameStart := 0 },
  { event := event30123
    frameStart := 0 },
  { event := event30124
    frameStart := 0 },
  { event := event30125
    frameStart := 0 },
  { event := event30126
    frameStart := 0 },
  { event := event30127
    frameStart := 0 }
]

def eventLeaf1883 : Array AnnotatedEvent := #[
  { event := event30128
    frameStart := 0 },
  { event := event30129
    frameStart := 0 },
  { event := event30130
    frameStart := 0 },
  { event := event30131
    frameStart := 0 },
  { event := event30132
    frameStart := 0 },
  { event := event30133
    frameStart := 0 },
  { event := event30134
    frameStart := 0 },
  { event := event30135
    frameStart := 0 },
  { event := event30136
    frameStart := 0 },
  { event := event30137
    frameStart := 0 },
  { event := event30138
    frameStart := 0 },
  { event := event30139
    frameStart := 0 },
  { event := event30140
    frameStart := 0 },
  { event := event30141
    frameStart := 0 },
  { event := event30142
    frameStart := 0 },
  { event := event30143
    frameStart := 0 }
]

def eventLeaf1884 : Array AnnotatedEvent := #[
  { event := event30144
    frameStart := 0 },
  { event := event30145
    frameStart := 0 },
  { event := event30146
    frameStart := 0 },
  { event := event30147
    frameStart := 0 },
  { event := event30148
    frameStart := 0 },
  { event := event30149
    frameStart := 0 },
  { event := event30150
    frameStart := 0 },
  { event := event30151
    frameStart := 0 },
  { event := event30152
    frameStart := 0 },
  { event := event30153
    frameStart := 0 },
  { event := event30154
    frameStart := 0 },
  { event := event30155
    frameStart := 0 },
  { event := event30156
    frameStart := 0 },
  { event := event30157
    frameStart := 0 },
  { event := event30158
    frameStart := 0 },
  { event := event30159
    frameStart := 0 }
]

def eventLeaf1885 : Array AnnotatedEvent := #[
  { event := event30160
    frameStart := 0 },
  { event := event30161
    frameStart := 0 },
  { event := event30162
    frameStart := 0 },
  { event := event30163
    frameStart := 0 },
  { event := event30164
    frameStart := 0 },
  { event := event30165
    frameStart := 0 },
  { event := event30166
    frameStart := 0 },
  { event := event30167
    frameStart := 0 },
  { event := event30168
    frameStart := 0 },
  { event := event30169
    frameStart := 0 },
  { event := event30170
    frameStart := 0 },
  { event := event30171
    frameStart := 0 },
  { event := event30172
    frameStart := 0 },
  { event := event30173
    frameStart := 0 },
  { event := event30174
    frameStart := 0 },
  { event := event30175
    frameStart := 0 }
]

def eventLeaf1886 : Array AnnotatedEvent := #[
  { event := event30176
    frameStart := 0 },
  { event := event30177
    frameStart := 0 },
  { event := event30178
    frameStart := 0 },
  { event := event30179
    frameStart := 0 },
  { event := event30180
    frameStart := 0 },
  { event := event30181
    frameStart := 0 },
  { event := event30182
    frameStart := 0 },
  { event := event30183
    frameStart := 0 },
  { event := event30184
    frameStart := 0 },
  { event := event30185
    frameStart := 0 },
  { event := event30186
    frameStart := 0 },
  { event := event30187
    frameStart := 0 },
  { event := event30188
    frameStart := 0 },
  { event := event30189
    frameStart := 0 },
  { event := event30190
    frameStart := 0 },
  { event := event30191
    frameStart := 0 }
]

def eventLeaf1887 : Array AnnotatedEvent := #[
  { event := event30192
    frameStart := 0 },
  { event := event30193
    frameStart := 0 },
  { event := event30194
    frameStart := 0 },
  { event := event30195
    frameStart := 0 },
  { event := event30196
    frameStart := 0 },
  { event := event30197
    frameStart := 0 },
  { event := event30198
    frameStart := 0 },
  { event := event30199
    frameStart := 0 },
  { event := event30200
    frameStart := 0 },
  { event := event30201
    frameStart := 0 },
  { event := event30202
    frameStart := 0 },
  { event := event30203
    frameStart := 0 },
  { event := event30204
    frameStart := 0 },
  { event := event30205
    frameStart := 0 },
  { event := event30206
    frameStart := 0 },
  { event := event30207
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events117
