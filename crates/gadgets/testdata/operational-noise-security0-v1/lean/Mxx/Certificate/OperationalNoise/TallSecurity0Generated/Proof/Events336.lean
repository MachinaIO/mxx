import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events336

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event86016 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15702⟩⟩) 0 ⟨13775⟩ 86015

def event86017 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15702⟩⟩) (.authority (.programFamilyFact))

def exact86018RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], []⟩, (1)⟩]

theorem exact86018RawTermsValid :
    exact86018RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86018 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15702⟩⟩) exact86018RawTerms (.finite 12) 86017 .exactZero (none)

def event86019 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15703⟩⟩) 0 ⟨15702⟩ 86018

def event86020 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15703⟩⟩) (.identity (.predecessor 0 86019 .coefficient))

def event86021 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15703⟩⟩) (.finite 12)

def event86022 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21112⟩⟩) 0 ⟨15703⟩ 86021

def event86023 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21112⟩⟩) (.authority (.relationPreimageSource ⟨39⟩))

def exact86024RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21112⟩⟩]⟩, (1)⟩]

theorem exact86024RawTermsValid :
    exact86024RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86024 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21112⟩⟩) exact86024RawTerms (.finite 136065468) 86023 .exactZero (none)

def event86025 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact86026RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact86026RawTermsValid :
    exact86026RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86026 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact86026RawTerms .large 86025 .exactZero (none)

def event86027 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21113⟩⟩) 0 ⟨6⟩ 86026

def event86028 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21113⟩⟩) 1 ⟨21112⟩ 86024

def event86029 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21113⟩⟩) (.product (.predecessor 0 86027 .coefficient) (.predecessor 1 86028 .coefficient) (⟨false, false, none, none, none⟩))

def event86030 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21113⟩⟩, .operator (⟨86026, 0⟩, ⟨86024, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21112⟩⟩]⟩, (1)⟩)

def exact86031RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21112⟩⟩]⟩, (1)⟩]

theorem exact86031RawTermsValid :
    exact86031RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86031 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21113⟩⟩) exact86031RawTerms .large 86029 .exactZero (none)

def event86032 : Event := .preFoldPolynomial 86031 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21112⟩⟩]⟩, (1)⟩] .exactZero none

def exact86033RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21112⟩⟩]⟩, (1)⟩]

def event86033 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21113⟩⟩) 86032 exact86033RawTerms .large 86029 .exactZero (none)

def event86034 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨27437⟩⟩)

def event86035 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event86036 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event86037 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event86038 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event86039 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event86040 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event86041 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event86042 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event86043 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 86042

def event86044 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 86040

def event86045 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 86043 .coefficient) (.value (.predecessor 1 86044 .coefficient)))

def event86046 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event86047 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 86046

def event86048 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 86038

def event86049 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 86047 .coefficient, .predecessor 1 86048 .coefficient])

def event86050 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event86051 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 86050

def event86052 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 86036

def event86053 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 86052 .coefficient))

def event86054 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event86055 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11301⟩⟩) 0 ⟨5536⟩ 86054

def event86056 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11301⟩⟩) (.authority (.programFamilyFact))

def exact86057RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩], []⟩, (1)⟩]

theorem exact86057RawTermsValid :
    exact86057RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86057 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11301⟩⟩) exact86057RawTerms (.finite 12) 86056 .exactZero (none)

def event86058 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13773⟩⟩) 0 ⟨5536⟩ 86054

def event86059 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13773⟩⟩) (.authority (.programFamilyFact))

def exact86060RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13773⟩⟩], []⟩, (1)⟩]

theorem exact86060RawTermsValid :
    exact86060RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86060 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13773⟩⟩) exact86060RawTerms (.finite 12) 86059 .exactZero (none)

def event86061 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13774⟩⟩) 0 ⟨13773⟩ 86060

def event86062 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13774⟩⟩) 1 ⟨11301⟩ 86057

def event86063 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13774⟩⟩) (.product (.predecessor 0 86061 .coefficient) (.predecessor 1 86062 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event86064 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13774⟩⟩, .operator (⟨86060, 0⟩, ⟨86057, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], []⟩, (1)⟩)

def exact86065RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], []⟩, (1)⟩]

theorem exact86065RawTermsValid :
    exact86065RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86065 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13774⟩⟩) exact86065RawTerms (.finite 144) 86063 .exactZero (none)

def event86066 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13775⟩⟩) 0 ⟨13774⟩ 86065

def event86067 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13775⟩⟩) (.identity (.predecessor 0 86066 .coefficient))

def event86068 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13775⟩⟩) (.finite 144)

def event86069 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15702⟩⟩) 0 ⟨13775⟩ 86068

def event86070 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15702⟩⟩) (.authority (.programFamilyFact))

def exact86071RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], []⟩, (1)⟩]

theorem exact86071RawTermsValid :
    exact86071RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86071 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15702⟩⟩) exact86071RawTerms (.finite 12) 86070 .exactZero (none)

def event86072 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15703⟩⟩) 0 ⟨15702⟩ 86071

def event86073 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15703⟩⟩) (.identity (.predecessor 0 86072 .coefficient))

def event86074 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15703⟩⟩) (.finite 12)

def event86075 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24034⟩⟩) 0 ⟨15703⟩ 86074

def event86076 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24034⟩⟩) (.authority (.programFamilyFact))

def event86077 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24034⟩⟩) (.finite 3720)

def event86078 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event86079 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24036⟩⟩) 0 ⟨6689⟩ 86078

def event86080 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24036⟩⟩) 1 ⟨24034⟩ 86077

def event86081 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24036⟩⟩) (.authority (.operator))

def exact86082RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24036⟩⟩]⟩, (1)⟩]

theorem exact86082RawTermsValid :
    exact86082RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86082 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24036⟩⟩) exact86082RawTerms .large 86081 .exactZero (none)

def event86083 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27432⟩⟩) 0 ⟨24036⟩ 86082

def event86084 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27432⟩⟩) (.authority (.operator))

def exact86085RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩, (1)⟩]

theorem exact86085RawTermsValid :
    exact86085RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86085 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27432⟩⟩) exact86085RawTerms (.finite 8192) 86084 .exactZero (none)

def event86086 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event86087 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event86088 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15777⟩⟩) 0 ⟨15703⟩ 86074

def event86089 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15777⟩⟩) 1 ⟨110⟩ 86087

def event86090 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15777⟩⟩) (.sum [.predecessor 0 86088 .coefficient, .predecessor 1 86089 .coefficient])

def event86091 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15777⟩⟩) (.finite 12)

def event86092 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15778⟩⟩) 0 ⟨15777⟩ 86091

def event86093 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15778⟩⟩) (.identity (.predecessor 0 86092 .coefficient))

def exact86094RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], []⟩, (1)⟩]

theorem exact86094RawTermsValid :
    exact86094RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86094 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15778⟩⟩) exact86094RawTerms (.finite 12) 86093 .exactZero (none)

def event86095 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact86096RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact86096RawTermsValid :
    exact86096RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86096 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact86096RawTerms .large 86095 .exactZero (none)

def event86097 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15779⟩⟩) 0 ⟨6544⟩ 86096

def event86098 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15779⟩⟩) 1 ⟨15778⟩ 86094

def event86099 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15779⟩⟩) (.product (.predecessor 0 86097 .coefficient) (.predecessor 1 86098 .coefficient) (⟨false, false, none, none, none⟩))

def event86100 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15779⟩⟩, .operator (⟨86096, 0⟩, ⟨86094, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact86101RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact86101RawTermsValid :
    exact86101RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86101 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15779⟩⟩) exact86101RawTerms .large 86099 .exactZero (none)

def event86102 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6695⟩⟩) 0 ⟨6689⟩ 86078

def event86103 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6695⟩⟩) (.authority (.operator))

def exact86104RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩]

theorem exact86104RawTermsValid :
    exact86104RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86104 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6695⟩⟩) exact86104RawTerms .large 86103 .exactZero (none)

def event86105 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15780⟩⟩) 0 ⟨6695⟩ 86104

def event86106 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15780⟩⟩) 1 ⟨15779⟩ 86101

def event86107 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15780⟩⟩) (.sum [.predecessor 0 86105 .coefficient, .predecessor 1 86106 .coefficient])

def exact86108RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86108RawTermsValid :
    exact86108RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86108 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15780⟩⟩) exact86108RawTerms .large 86107 .exactZero (none)

def event86109 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27433⟩⟩) 0 ⟨15780⟩ 86108

def event86110 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27433⟩⟩) 1 ⟨27432⟩ 86085

def event86111 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27433⟩⟩) (.product (.predecessor 0 86109 .coefficient) (.predecessor 1 86110 .coefficient) (⟨false, false, none, none, none⟩))

def event86112 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27433⟩⟩, .operator (⟨86108, 0⟩, ⟨86085, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩, (1)⟩)

def event86113 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27433⟩⟩, .operator (⟨86108, 1⟩, ⟨86085, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩, (-1)⟩)

def event86114 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27433⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27432⟩⟩) ⟨24036⟩ 86082)

def event86115 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27433⟩⟩, .relation 86114 0, ⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨24036⟩⟩]⟩, (-1)⟩)

def exact86116RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨24036⟩⟩]⟩, (-1)⟩]

theorem exact86116RawTermsValid :
    exact86116RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86116 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27433⟩⟩) exact86116RawTerms .large 86111 .exactZero (none)

def event86117 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15748⟩⟩) 0 ⟨15703⟩ 86074

def event86118 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15748⟩⟩) (.authority (.programFamilyFact))

def exact86119RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15748⟩⟩], []⟩, (1)⟩]

theorem exact86119RawTermsValid :
    exact86119RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86119 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15748⟩⟩) exact86119RawTerms (.finite 59) 86118 .exactZero (none)

def event86120 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15749⟩⟩) 0 ⟨6544⟩ 86096

def event86121 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15749⟩⟩) 1 ⟨15748⟩ 86119

def event86122 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15749⟩⟩) (.product (.predecessor 0 86120 .coefficient) (.predecessor 1 86121 .coefficient) (⟨false, true, none, none, some 1⟩))

def event86123 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15749⟩⟩, .operator (⟨86096, 0⟩, ⟨86119, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact86124RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact86124RawTermsValid :
    exact86124RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86124 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15749⟩⟩) exact86124RawTerms .large 86122 .exactZero (none)

def event86125 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6719⟩⟩) 0 ⟨6689⟩ 86078

def event86126 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6719⟩⟩) (.authority (.operator))

def exact86127RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩]

theorem exact86127RawTermsValid :
    exact86127RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86127 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6719⟩⟩) exact86127RawTerms .large 86126 .exactZero (none)

def event86128 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15750⟩⟩) 0 ⟨6719⟩ 86127

def event86129 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15750⟩⟩) 1 ⟨15749⟩ 86124

def event86130 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15750⟩⟩) (.sum [.predecessor 0 86128 .coefficient, .predecessor 1 86129 .coefficient])

def exact86131RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86131RawTermsValid :
    exact86131RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86131 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15750⟩⟩) exact86131RawTerms .large 86130 .exactZero (none)

def event86132 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27437⟩⟩) 0 ⟨15750⟩ 86131

def event86133 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27437⟩⟩) 1 ⟨27433⟩ 86116

def event86134 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27437⟩⟩) (.sum [.predecessor 0 86132 .coefficient, .predecessor 1 86133 .coefficient])

def exact86135RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨24036⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86135RawTermsValid :
    exact86135RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86135 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27437⟩⟩) exact86135RawTerms .large 86134 .exactZero (none)

def event86136 : Event := .preFoldPolynomial 86135 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨24036⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact86137RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨24036⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event86137 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27437⟩⟩) 86136 exact86137RawTerms .large 86134 .exactZero (none)

def event86138 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15703⟩⟩) ⟨⟨132⟩, ⟨39⟩, ⟨109⟩⟩ ⟨85980, 86138⟩

def event86139 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21115⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21112⟩⟩]⟩) (1) 0 2 (.universal 86138 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21112⟩⟩]⟩) (none) 86137)

def event86140 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21115⟩⟩, .relation 86139 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩)

def event86141 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21115⟩⟩, .relation 86139 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩, (-1)⟩)

def event86142 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21115⟩⟩, .relation 86139 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨24036⟩⟩]⟩, (1)⟩)

def event86143 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21115⟩⟩, .relation 86139 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact86144RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨24036⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86144RawTermsValid :
    exact86144RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86144 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21115⟩⟩) exact86144RawTerms .large 85976 (.finite 1811303510016) (some (85978))

def event86145 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27435⟩⟩) 0 ⟨21115⟩ 86144

def event86146 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27435⟩⟩) 1 ⟨27434⟩ 85966

def event86147 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27435⟩⟩) (.sum [.predecessor 0 86145 .coefficient, .predecessor 1 86146 .coefficient])

def event86148 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27435⟩⟩, .operator (⟨86144, 0⟩, ⟨85966, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩, (1)⟩)

def event86149 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27435⟩⟩, .operator (⟨86144, 2⟩, ⟨85966, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨24036⟩⟩]⟩, (-1)⟩)

def event86150 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27435⟩⟩) (.sum [.result 86144 .summary, .result 85966 .summary])

def exact86151RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86151RawTermsValid :
    exact86151RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86151 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27435⟩⟩) exact86151RawTerms .large 86147 (.finite 1292001236604524572672) (some (86150))

def event86152 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23971⟩⟩) 0 ⟨15584⟩ 4144

def event86153 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23971⟩⟩) (.authority (.programFamilyFact))

def event86154 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23971⟩⟩) (.finite 3720)

def event86155 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23973⟩⟩) 0 ⟨6689⟩ 5477

def event86156 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23973⟩⟩) 1 ⟨23971⟩ 86154

def event86157 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23973⟩⟩) (.authority (.operator))

def exact86158RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23973⟩⟩]⟩, (1)⟩]

theorem exact86158RawTermsValid :
    exact86158RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86158 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23973⟩⟩) exact86158RawTerms .large 86157 .exactZero (none)

def event86159 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27215⟩⟩) 0 ⟨23973⟩ 86158

def event86160 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27215⟩⟩) (.authority (.operator))

def exact86161RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27215⟩⟩]⟩, (1)⟩]

theorem exact86161RawTermsValid :
    exact86161RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86161 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27215⟩⟩) exact86161RawTerms (.finite 8192) 86160 .exactZero (none)

def event86162 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23457⟩⟩) 0 ⟨13558⟩ 4138

def event86163 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23457⟩⟩) (.authority (.programFamilyFact))

def event86164 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23457⟩⟩) (.finite 3720)

def event86165 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23458⟩⟩) 0 ⟨6689⟩ 5477

def event86166 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23458⟩⟩) 1 ⟨23457⟩ 86164

def event86167 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23458⟩⟩) (.authority (.operator))

def exact86168RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23458⟩⟩]⟩, (1)⟩]

theorem exact86168RawTermsValid :
    exact86168RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86168 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23458⟩⟩) exact86168RawTerms .large 86167 .exactZero (none)

def event86169 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25835⟩⟩) 0 ⟨23458⟩ 86168

def event86170 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25835⟩⟩) (.authority (.operator))

def exact86171RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25835⟩⟩]⟩, (1)⟩]

theorem exact86171RawTermsValid :
    exact86171RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86171 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25835⟩⟩) exact86171RawTerms (.finite 8192) 86170 .exactZero (none)

def event86172 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11218⟩⟩) 0 ⟨11217⟩ 4127

def event86173 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11218⟩⟩) 1 ⟨6567⟩ 79920

def event86174 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11218⟩⟩) (.tensor (.predecessor 0 86172 .coefficient) (.predecessor 1 86173 .coefficient) true false)

def event86175 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11218⟩⟩, .operator (⟨4127, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact86176RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact86176RawTermsValid :
    exact86176RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86176 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11218⟩⟩) exact86176RawTerms .large 86174 .exactZero (none)

def event86177 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7232⟩⟩) 0 ⟨5539⟩ 79790

def event86178 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7232⟩⟩) 1 ⟨6776⟩ 12985

def event86179 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7232⟩⟩) (.product (.predecessor 0 86177 .coefficient) (.predecessor 1 86178 .coefficient) (⟨false, false, none, none, none⟩))

def event86180 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7232⟩⟩, .operator (⟨79790, 0⟩, ⟨12985, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩)

def exact86181RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩]

theorem exact86181RawTermsValid :
    exact86181RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86181 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7232⟩⟩) exact86181RawTerms .large 86179 .exactZero (none)

def event86182 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11219⟩⟩) 0 ⟨7232⟩ 86181

def event86183 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11219⟩⟩) 1 ⟨11218⟩ 86176

def event86184 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11219⟩⟩) (.sum [.predecessor 0 86182 .coefficient, .predecessor 1 86183 .coefficient])

def exact86185RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86185RawTermsValid :
    exact86185RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86185 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11219⟩⟩) exact86185RawTerms .large 86184 .exactZero (none)

def event86186 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11220⟩⟩) 0 ⟨11219⟩ 86185

def event86187 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11220⟩⟩) 1 ⟨90⟩ 12977

def event86188 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11220⟩⟩) (.sum [.predecessor 0 86186 .coefficient, .predecessor 1 86187 .coefficient])

def event86189 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11220⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨90⟩⟩]⟩) [⟨.result 12977 .coefficient, false, none⟩])

def event86190 : Event := .survivorFold (1) 86189

def exact86191RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11217⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86191RawTermsValid :
    exact86191RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86191 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11220⟩⟩) exact86191RawTerms .large 86188 (.finite 26) (some (86189))

def event86192 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13559⟩⟩) 0 ⟨11220⟩ 86191

def event86193 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13559⟩⟩) 1 ⟨13556⟩ 4130

def event86194 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13559⟩⟩) (.product (.predecessor 0 86192 .coefficient) (.predecessor 1 86193 .coefficient) (⟨false, true, none, none, some 1⟩))

def event86195 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13559⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨13556⟩⟩], []⟩) [⟨.result 4130 .coefficient, true, some 1⟩])

def event86196 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13559⟩⟩) (.product (.result 86191 .summary) (.transfer 86195) (⟨false, false, none, none, none⟩))

def event86197 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13559⟩⟩, .operator (⟨86191, 1⟩, ⟨4130, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11217⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event86198 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13559⟩⟩, .operator (⟨86191, 0⟩, ⟨4130, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩)

def exact86199RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11217⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩]

theorem exact86199RawTermsValid :
    exact86199RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86199 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13559⟩⟩) exact86199RawTerms .large 86194 (.finite 8320) (some (86196))

def event86200 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13560⟩⟩) 0 ⟨13556⟩ 4130

def event86201 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13560⟩⟩) 1 ⟨6567⟩ 79920

def event86202 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13560⟩⟩) (.tensor (.predecessor 0 86200 .coefficient) (.predecessor 1 86201 .coefficient) true false)

def event86203 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13560⟩⟩, .operator (⟨4130, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact86204RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact86204RawTermsValid :
    exact86204RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86204 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13560⟩⟩) exact86204RawTerms .large 86202 .exactZero (none)

def event86205 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7249⟩⟩) 0 ⟨5539⟩ 79790

def event86206 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7249⟩⟩) 1 ⟨6793⟩ 13026

def event86207 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7249⟩⟩) (.product (.predecessor 0 86205 .coefficient) (.predecessor 1 86206 .coefficient) (⟨false, false, none, none, none⟩))

def event86208 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7249⟩⟩, .operator (⟨79790, 0⟩, ⟨13026, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩]⟩, (1)⟩)

def exact86209RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩]⟩, (1)⟩]

theorem exact86209RawTermsValid :
    exact86209RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86209 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7249⟩⟩) exact86209RawTerms .large 86207 .exactZero (none)

def event86210 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13561⟩⟩) 0 ⟨7249⟩ 86209

def event86211 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13561⟩⟩) 1 ⟨13560⟩ 86204

def event86212 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13561⟩⟩) (.sum [.predecessor 0 86210 .coefficient, .predecessor 1 86211 .coefficient])

def exact86213RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86213RawTermsValid :
    exact86213RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86213 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13561⟩⟩) exact86213RawTerms .large 86212 .exactZero (none)

def event86214 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13562⟩⟩) 0 ⟨13561⟩ 86213

def event86215 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13562⟩⟩) 1 ⟨107⟩ 13018

def event86216 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13562⟩⟩) (.sum [.predecessor 0 86214 .coefficient, .predecessor 1 86215 .coefficient])

def event86217 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13562⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨107⟩⟩]⟩) [⟨.result 13018 .coefficient, false, none⟩])

def event86218 : Event := .survivorFold (1) 86217

def exact86219RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86219RawTermsValid :
    exact86219RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86219 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13562⟩⟩) exact86219RawTerms .large 86216 (.finite 26) (some (86217))

def event86220 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13563⟩⟩) 0 ⟨13562⟩ 86219

def event86221 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13563⟩⟩) 1 ⟨7844⟩ 13015

def event86222 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13563⟩⟩) (.product (.predecessor 0 86220 .coefficient) (.predecessor 1 86221 .coefficient) (⟨false, false, none, none, none⟩))

def event86223 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13563⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩) [⟨.result 13011 .coefficient, false, none⟩])

def event86224 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13563⟩⟩) (.product (.result 86219 .summary) (.transfer 86223) (⟨false, false, none, none, none⟩))

def event86225 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13563⟩⟩, .operator (⟨86219, 1⟩, ⟨13015, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (-1)⟩)

def event86226 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨13563⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7843⟩⟩) ⟨6776⟩ 12985)

def event86227 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13563⟩⟩, .relation 86226 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (-1)⟩)

def event86228 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13563⟩⟩, .operator (⟨86219, 0⟩, ⟨13015, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩)

def exact86229RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (-1)⟩]

theorem exact86229RawTermsValid :
    exact86229RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86229 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13563⟩⟩) exact86229RawTerms .large 86222 (.finite 95420416) (some (86224))

def event86230 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13564⟩⟩) 0 ⟨13563⟩ 86229

def event86231 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13564⟩⟩) 1 ⟨13559⟩ 86199

def event86232 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13564⟩⟩) (.sum [.predecessor 0 86230 .coefficient, .predecessor 1 86231 .coefficient])

def event86233 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13564⟩⟩, .operator (⟨86229, 1⟩, ⟨86199, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩, (1)⟩)

def event86234 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13564⟩⟩) (.sum [.result 86229 .summary, .result 86199 .summary])

def exact86235RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11217⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact86235RawTermsValid :
    exact86235RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86235 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13564⟩⟩) exact86235RawTerms .large 86232 (.finite 95428736) (some (86234))

def event86236 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25836⟩⟩) 0 ⟨13564⟩ 86235

def event86237 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25836⟩⟩) 1 ⟨25835⟩ 86171

def event86238 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25836⟩⟩) (.product (.predecessor 0 86236 .coefficient) (.predecessor 1 86237 .coefficient) (⟨false, false, none, none, none⟩))

def event86239 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25836⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25835⟩⟩]⟩) [⟨.result 86171 .coefficient, false, none⟩])

def event86240 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25836⟩⟩) (.product (.result 86235 .summary) (.transfer 86239) (⟨false, false, none, none, none⟩))

def event86241 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25836⟩⟩, .operator (⟨86235, 1⟩, ⟨86171, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11217⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25835⟩⟩]⟩, (-1)⟩)

def event86242 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25836⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11217⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25835⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25835⟩⟩) ⟨23458⟩ 86168)

def event86243 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25836⟩⟩, .relation 86242 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11217⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], [⟨.program ⟨214⟩, ⟨23458⟩⟩]⟩, (-1)⟩)

def event86244 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25836⟩⟩, .operator (⟨86235, 0⟩, ⟨86171, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25835⟩⟩]⟩, (1)⟩)

def exact86245RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25835⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11217⟩⟩, ⟨.program ⟨214⟩, ⟨13556⟩⟩], [⟨.program ⟨214⟩, ⟨23458⟩⟩]⟩, (-1)⟩]

theorem exact86245RawTermsValid :
    exact86245RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86245 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25836⟩⟩) exact86245RawTerms .large 86238 (.finite 350224987979776) (some (86240))

def event86246 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19312⟩⟩) 0 ⟨13558⟩ 4138

def event86247 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19312⟩⟩) (.authority (.relationPreimageSource ⟨12⟩))

def exact86248RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19312⟩⟩]⟩, (1)⟩]

theorem exact86248RawTermsValid :
    exact86248RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86248 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19312⟩⟩) exact86248RawTerms (.finite 136065468) 86247 .exactZero (none)

def event86249 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19314⟩⟩) 0 ⟨19312⟩ 86248

def event86250 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19314⟩⟩) 1 ⟨2348⟩ 4

def event86251 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19314⟩⟩) (.scale (.predecessor 0 86249 .coefficient) (.value (.predecessor 1 86250 .coefficient)))

def exact86252RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19312⟩⟩]⟩, (1)⟩]

theorem exact86252RawTermsValid :
    exact86252RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event86252 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19314⟩⟩) exact86252RawTerms (.finite 136065468) 86251 .exactZero (none)

def event86253 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19315⟩⟩) 0 ⟨5541⟩ 80012

def event86254 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19315⟩⟩) 1 ⟨19314⟩ 86252

def event86255 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19315⟩⟩) (.product (.predecessor 0 86253 .coefficient) (.predecessor 1 86254 .coefficient) (⟨false, false, none, none, none⟩))

def event86256 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19315⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19312⟩⟩]⟩) [⟨.result 86248 .coefficient, false, none⟩])

def event86257 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19315⟩⟩) (.product (.result 80012 .summary) (.transfer 86256) (⟨false, false, none, none, none⟩))

def event86258 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19315⟩⟩, .operator (⟨80012, 0⟩, ⟨86252, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19312⟩⟩]⟩, (1)⟩)

def event86259 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19313⟩⟩)

def event86260 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event86261 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event86262 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event86263 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event86264 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event86265 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event86266 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event86267 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event86268 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 86267

def event86269 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 86265

def event86270 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 86268 .coefficient) (.value (.predecessor 1 86269 .coefficient)))

def event86271 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def eventLeaf5376 : Array AnnotatedEvent := #[
  { event := event86016
    frameStart := 85980 },
  { event := event86017
    frameStart := 85980 },
  { event := event86018
    frameStart := 85980 },
  { event := event86019
    frameStart := 85980 },
  { event := event86020
    frameStart := 85980 },
  { event := event86021
    frameStart := 85980 },
  { event := event86022
    frameStart := 85980 },
  { event := event86023
    frameStart := 85980 },
  { event := event86024
    frameStart := 85980 },
  { event := event86025
    frameStart := 85980 },
  { event := event86026
    frameStart := 85980 },
  { event := event86027
    frameStart := 85980 },
  { event := event86028
    frameStart := 85980 },
  { event := event86029
    frameStart := 85980 },
  { event := event86030
    frameStart := 85980 },
  { event := event86031
    frameStart := 85980 }
]

def eventLeaf5377 : Array AnnotatedEvent := #[
  { event := event86032
    frameStart := 85980 },
  { event := event86033
    frameStart := 85980 },
  { event := event86034
    frameStart := 86034 },
  { event := event86035
    frameStart := 86034 },
  { event := event86036
    frameStart := 86034 },
  { event := event86037
    frameStart := 86034 },
  { event := event86038
    frameStart := 86034 },
  { event := event86039
    frameStart := 86034 },
  { event := event86040
    frameStart := 86034 },
  { event := event86041
    frameStart := 86034 },
  { event := event86042
    frameStart := 86034 },
  { event := event86043
    frameStart := 86034 },
  { event := event86044
    frameStart := 86034 },
  { event := event86045
    frameStart := 86034 },
  { event := event86046
    frameStart := 86034 },
  { event := event86047
    frameStart := 86034 }
]

def eventLeaf5378 : Array AnnotatedEvent := #[
  { event := event86048
    frameStart := 86034 },
  { event := event86049
    frameStart := 86034 },
  { event := event86050
    frameStart := 86034 },
  { event := event86051
    frameStart := 86034 },
  { event := event86052
    frameStart := 86034 },
  { event := event86053
    frameStart := 86034 },
  { event := event86054
    frameStart := 86034 },
  { event := event86055
    frameStart := 86034 },
  { event := event86056
    frameStart := 86034 },
  { event := event86057
    frameStart := 86034 },
  { event := event86058
    frameStart := 86034 },
  { event := event86059
    frameStart := 86034 },
  { event := event86060
    frameStart := 86034 },
  { event := event86061
    frameStart := 86034 },
  { event := event86062
    frameStart := 86034 },
  { event := event86063
    frameStart := 86034 }
]

def eventLeaf5379 : Array AnnotatedEvent := #[
  { event := event86064
    frameStart := 86034 },
  { event := event86065
    frameStart := 86034 },
  { event := event86066
    frameStart := 86034 },
  { event := event86067
    frameStart := 86034 },
  { event := event86068
    frameStart := 86034 },
  { event := event86069
    frameStart := 86034 },
  { event := event86070
    frameStart := 86034 },
  { event := event86071
    frameStart := 86034 },
  { event := event86072
    frameStart := 86034 },
  { event := event86073
    frameStart := 86034 },
  { event := event86074
    frameStart := 86034 },
  { event := event86075
    frameStart := 86034 },
  { event := event86076
    frameStart := 86034 },
  { event := event86077
    frameStart := 86034 },
  { event := event86078
    frameStart := 86034 },
  { event := event86079
    frameStart := 86034 }
]

def eventLeaf5380 : Array AnnotatedEvent := #[
  { event := event86080
    frameStart := 86034 },
  { event := event86081
    frameStart := 86034 },
  { event := event86082
    frameStart := 86034 },
  { event := event86083
    frameStart := 86034 },
  { event := event86084
    frameStart := 86034 },
  { event := event86085
    frameStart := 86034 },
  { event := event86086
    frameStart := 86034 },
  { event := event86087
    frameStart := 86034 },
  { event := event86088
    frameStart := 86034 },
  { event := event86089
    frameStart := 86034 },
  { event := event86090
    frameStart := 86034 },
  { event := event86091
    frameStart := 86034 },
  { event := event86092
    frameStart := 86034 },
  { event := event86093
    frameStart := 86034 },
  { event := event86094
    frameStart := 86034 },
  { event := event86095
    frameStart := 86034 }
]

def eventLeaf5381 : Array AnnotatedEvent := #[
  { event := event86096
    frameStart := 86034 },
  { event := event86097
    frameStart := 86034 },
  { event := event86098
    frameStart := 86034 },
  { event := event86099
    frameStart := 86034 },
  { event := event86100
    frameStart := 86034 },
  { event := event86101
    frameStart := 86034 },
  { event := event86102
    frameStart := 86034 },
  { event := event86103
    frameStart := 86034 },
  { event := event86104
    frameStart := 86034 },
  { event := event86105
    frameStart := 86034 },
  { event := event86106
    frameStart := 86034 },
  { event := event86107
    frameStart := 86034 },
  { event := event86108
    frameStart := 86034 },
  { event := event86109
    frameStart := 86034 },
  { event := event86110
    frameStart := 86034 },
  { event := event86111
    frameStart := 86034 }
]

def eventLeaf5382 : Array AnnotatedEvent := #[
  { event := event86112
    frameStart := 86034 },
  { event := event86113
    frameStart := 86034 },
  { event := event86114
    frameStart := 86034 },
  { event := event86115
    frameStart := 86034 },
  { event := event86116
    frameStart := 86034 },
  { event := event86117
    frameStart := 86034 },
  { event := event86118
    frameStart := 86034 },
  { event := event86119
    frameStart := 86034 },
  { event := event86120
    frameStart := 86034 },
  { event := event86121
    frameStart := 86034 },
  { event := event86122
    frameStart := 86034 },
  { event := event86123
    frameStart := 86034 },
  { event := event86124
    frameStart := 86034 },
  { event := event86125
    frameStart := 86034 },
  { event := event86126
    frameStart := 86034 },
  { event := event86127
    frameStart := 86034 }
]

def eventLeaf5383 : Array AnnotatedEvent := #[
  { event := event86128
    frameStart := 86034 },
  { event := event86129
    frameStart := 86034 },
  { event := event86130
    frameStart := 86034 },
  { event := event86131
    frameStart := 86034 },
  { event := event86132
    frameStart := 86034 },
  { event := event86133
    frameStart := 86034 },
  { event := event86134
    frameStart := 86034 },
  { event := event86135
    frameStart := 86034 },
  { event := event86136
    frameStart := 86034 },
  { event := event86137
    frameStart := 86034 },
  { event := event86138
    frameStart := 0 },
  { event := event86139
    frameStart := 0 },
  { event := event86140
    frameStart := 0 },
  { event := event86141
    frameStart := 0 },
  { event := event86142
    frameStart := 0 },
  { event := event86143
    frameStart := 0 }
]

def eventLeaf5384 : Array AnnotatedEvent := #[
  { event := event86144
    frameStart := 0 },
  { event := event86145
    frameStart := 0 },
  { event := event86146
    frameStart := 0 },
  { event := event86147
    frameStart := 0 },
  { event := event86148
    frameStart := 0 },
  { event := event86149
    frameStart := 0 },
  { event := event86150
    frameStart := 0 },
  { event := event86151
    frameStart := 0 },
  { event := event86152
    frameStart := 0 },
  { event := event86153
    frameStart := 0 },
  { event := event86154
    frameStart := 0 },
  { event := event86155
    frameStart := 0 },
  { event := event86156
    frameStart := 0 },
  { event := event86157
    frameStart := 0 },
  { event := event86158
    frameStart := 0 },
  { event := event86159
    frameStart := 0 }
]

def eventLeaf5385 : Array AnnotatedEvent := #[
  { event := event86160
    frameStart := 0 },
  { event := event86161
    frameStart := 0 },
  { event := event86162
    frameStart := 0 },
  { event := event86163
    frameStart := 0 },
  { event := event86164
    frameStart := 0 },
  { event := event86165
    frameStart := 0 },
  { event := event86166
    frameStart := 0 },
  { event := event86167
    frameStart := 0 },
  { event := event86168
    frameStart := 0 },
  { event := event86169
    frameStart := 0 },
  { event := event86170
    frameStart := 0 },
  { event := event86171
    frameStart := 0 },
  { event := event86172
    frameStart := 0 },
  { event := event86173
    frameStart := 0 },
  { event := event86174
    frameStart := 0 },
  { event := event86175
    frameStart := 0 }
]

def eventLeaf5386 : Array AnnotatedEvent := #[
  { event := event86176
    frameStart := 0 },
  { event := event86177
    frameStart := 0 },
  { event := event86178
    frameStart := 0 },
  { event := event86179
    frameStart := 0 },
  { event := event86180
    frameStart := 0 },
  { event := event86181
    frameStart := 0 },
  { event := event86182
    frameStart := 0 },
  { event := event86183
    frameStart := 0 },
  { event := event86184
    frameStart := 0 },
  { event := event86185
    frameStart := 0 },
  { event := event86186
    frameStart := 0 },
  { event := event86187
    frameStart := 0 },
  { event := event86188
    frameStart := 0 },
  { event := event86189
    frameStart := 0 },
  { event := event86190
    frameStart := 0 },
  { event := event86191
    frameStart := 0 }
]

def eventLeaf5387 : Array AnnotatedEvent := #[
  { event := event86192
    frameStart := 0 },
  { event := event86193
    frameStart := 0 },
  { event := event86194
    frameStart := 0 },
  { event := event86195
    frameStart := 0 },
  { event := event86196
    frameStart := 0 },
  { event := event86197
    frameStart := 0 },
  { event := event86198
    frameStart := 0 },
  { event := event86199
    frameStart := 0 },
  { event := event86200
    frameStart := 0 },
  { event := event86201
    frameStart := 0 },
  { event := event86202
    frameStart := 0 },
  { event := event86203
    frameStart := 0 },
  { event := event86204
    frameStart := 0 },
  { event := event86205
    frameStart := 0 },
  { event := event86206
    frameStart := 0 },
  { event := event86207
    frameStart := 0 }
]

def eventLeaf5388 : Array AnnotatedEvent := #[
  { event := event86208
    frameStart := 0 },
  { event := event86209
    frameStart := 0 },
  { event := event86210
    frameStart := 0 },
  { event := event86211
    frameStart := 0 },
  { event := event86212
    frameStart := 0 },
  { event := event86213
    frameStart := 0 },
  { event := event86214
    frameStart := 0 },
  { event := event86215
    frameStart := 0 },
  { event := event86216
    frameStart := 0 },
  { event := event86217
    frameStart := 0 },
  { event := event86218
    frameStart := 0 },
  { event := event86219
    frameStart := 0 },
  { event := event86220
    frameStart := 0 },
  { event := event86221
    frameStart := 0 },
  { event := event86222
    frameStart := 0 },
  { event := event86223
    frameStart := 0 }
]

def eventLeaf5389 : Array AnnotatedEvent := #[
  { event := event86224
    frameStart := 0 },
  { event := event86225
    frameStart := 0 },
  { event := event86226
    frameStart := 0 },
  { event := event86227
    frameStart := 0 },
  { event := event86228
    frameStart := 0 },
  { event := event86229
    frameStart := 0 },
  { event := event86230
    frameStart := 0 },
  { event := event86231
    frameStart := 0 },
  { event := event86232
    frameStart := 0 },
  { event := event86233
    frameStart := 0 },
  { event := event86234
    frameStart := 0 },
  { event := event86235
    frameStart := 0 },
  { event := event86236
    frameStart := 0 },
  { event := event86237
    frameStart := 0 },
  { event := event86238
    frameStart := 0 },
  { event := event86239
    frameStart := 0 }
]

def eventLeaf5390 : Array AnnotatedEvent := #[
  { event := event86240
    frameStart := 0 },
  { event := event86241
    frameStart := 0 },
  { event := event86242
    frameStart := 0 },
  { event := event86243
    frameStart := 0 },
  { event := event86244
    frameStart := 0 },
  { event := event86245
    frameStart := 0 },
  { event := event86246
    frameStart := 0 },
  { event := event86247
    frameStart := 0 },
  { event := event86248
    frameStart := 0 },
  { event := event86249
    frameStart := 0 },
  { event := event86250
    frameStart := 0 },
  { event := event86251
    frameStart := 0 },
  { event := event86252
    frameStart := 0 },
  { event := event86253
    frameStart := 0 },
  { event := event86254
    frameStart := 0 },
  { event := event86255
    frameStart := 0 }
]

def eventLeaf5391 : Array AnnotatedEvent := #[
  { event := event86256
    frameStart := 0 },
  { event := event86257
    frameStart := 0 },
  { event := event86258
    frameStart := 0 },
  { event := event86259
    frameStart := 86259 },
  { event := event86260
    frameStart := 86259 },
  { event := event86261
    frameStart := 86259 },
  { event := event86262
    frameStart := 86259 },
  { event := event86263
    frameStart := 86259 },
  { event := event86264
    frameStart := 86259 },
  { event := event86265
    frameStart := 86259 },
  { event := event86266
    frameStart := 86259 },
  { event := event86267
    frameStart := 86259 },
  { event := event86268
    frameStart := 86259 },
  { event := event86269
    frameStart := 86259 },
  { event := event86270
    frameStart := 86259 },
  { event := event86271
    frameStart := 86259 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events336
