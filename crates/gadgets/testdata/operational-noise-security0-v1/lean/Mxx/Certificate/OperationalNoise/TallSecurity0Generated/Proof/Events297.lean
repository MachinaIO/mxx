import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events297

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event76032 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29584⟩⟩, .relation 76031 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24662⟩⟩]⟩, (-1)⟩)

def exact76033RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29582⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24662⟩⟩]⟩, (-1)⟩]

theorem exact76033RawTermsValid :
    exact76033RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76033 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29584⟩⟩) exact76033RawTerms .large 76026 (.finite 1292449483693632782336) (some (76028))

def event76034 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22476⟩⟩) 0 ⟨16749⟩ 3149

def event76035 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22476⟩⟩) (.authority (.relationPreimageSource ⟨60⟩))

def exact76036RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22476⟩⟩]⟩, (1)⟩]

theorem exact76036RawTermsValid :
    exact76036RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76036 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22476⟩⟩) exact76036RawTerms (.finite 136065468) 76035 .exactZero (none)

def event76037 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22478⟩⟩) 0 ⟨22476⟩ 76036

def event76038 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22478⟩⟩) 1 ⟨2348⟩ 4

def event76039 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22478⟩⟩) (.scale (.predecessor 0 76037 .coefficient) (.value (.predecessor 1 76038 .coefficient)))

def exact76040RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22476⟩⟩]⟩, (1)⟩]

theorem exact76040RawTermsValid :
    exact76040RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76040 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22478⟩⟩) exact76040RawTerms (.finite 136065468) 76039 .exactZero (none)

def event76041 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22479⟩⟩) 0 ⟨5535⟩ 65387

def event76042 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22479⟩⟩) 1 ⟨22478⟩ 76040

def event76043 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22479⟩⟩) (.product (.predecessor 0 76041 .coefficient) (.predecessor 1 76042 .coefficient) (⟨false, false, none, none, none⟩))

def event76044 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22479⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22476⟩⟩]⟩) [⟨.result 76036 .coefficient, false, none⟩])

def event76045 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22479⟩⟩) (.product (.result 65387 .summary) (.transfer 76044) (⟨false, false, none, none, none⟩))

def event76046 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22479⟩⟩, .operator (⟨65387, 0⟩, ⟨76040, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22476⟩⟩]⟩, (1)⟩)

def event76047 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22477⟩⟩)

def event76048 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event76049 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event76050 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event76051 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event76052 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event76053 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event76054 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event76055 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event76056 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 76055

def event76057 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 76053

def event76058 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 76056 .coefficient) (.value (.predecessor 1 76057 .coefficient)))

def event76059 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event76060 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 76059

def event76061 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 76051

def event76062 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 76060 .coefficient, .predecessor 1 76061 .coefficient])

def event76063 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event76064 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 76063

def event76065 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 76049

def event76066 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 76065 .coefficient))

def event76067 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event76068 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12950⟩⟩) 0 ⟨5530⟩ 76067

def event76069 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12950⟩⟩) (.authority (.programFamilyFact))

def exact76070RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12950⟩⟩], []⟩, (1)⟩]

theorem exact76070RawTermsValid :
    exact76070RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76070 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12950⟩⟩) exact76070RawTerms (.finite 52) 76069 .exactZero (none)

def event76071 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10130⟩⟩) 0 ⟨5530⟩ 76067

def event76072 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10130⟩⟩) (.authority (.programFamilyFact))

def exact76073RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10130⟩⟩], []⟩, (1)⟩]

theorem exact76073RawTermsValid :
    exact76073RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76073 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10130⟩⟩) exact76073RawTerms (.finite 52) 76072 .exactZero (none)

def event76074 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12951⟩⟩) 0 ⟨10130⟩ 76073

def event76075 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12951⟩⟩) 1 ⟨12950⟩ 76070

def event76076 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12951⟩⟩) (.product (.predecessor 0 76074 .coefficient) (.predecessor 1 76075 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event76077 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12951⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], []⟩) [⟨.result 76073 .coefficient, true, some 1⟩, ⟨.result 76070 .coefficient, true, some 1⟩])

def event76078 : Event := .survivorFold (1) 76077

def exact76079RawTerms : List Term := []

theorem exact76079RawTermsValid :
    exact76079RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76079 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12951⟩⟩) exact76079RawTerms (.finite 2704) 76076 (.finite 2704) (some (76077))

def event76080 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12952⟩⟩) 0 ⟨12951⟩ 76079

def event76081 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12952⟩⟩) (.identity (.predecessor 0 76080 .coefficient))

def event76082 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12952⟩⟩) (.finite 2704)

def event76083 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16748⟩⟩) 0 ⟨12952⟩ 76082

def event76084 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16748⟩⟩) (.authority (.programFamilyFact))

def exact76085RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], []⟩, (1)⟩]

theorem exact76085RawTermsValid :
    exact76085RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76085 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16748⟩⟩) exact76085RawTerms (.finite 52) 76084 .exactZero (none)

def event76086 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16749⟩⟩) 0 ⟨16748⟩ 76085

def event76087 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16749⟩⟩) (.identity (.predecessor 0 76086 .coefficient))

def event76088 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16749⟩⟩) (.finite 52)

def event76089 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22476⟩⟩) 0 ⟨16749⟩ 76088

def event76090 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22476⟩⟩) (.authority (.relationPreimageSource ⟨60⟩))

def exact76091RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22476⟩⟩]⟩, (1)⟩]

theorem exact76091RawTermsValid :
    exact76091RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76091 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22476⟩⟩) exact76091RawTerms (.finite 136065468) 76090 .exactZero (none)

def event76092 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact76093RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact76093RawTermsValid :
    exact76093RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76093 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact76093RawTerms .large 76092 .exactZero (none)

def event76094 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22477⟩⟩) 0 ⟨6⟩ 76093

def event76095 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22477⟩⟩) 1 ⟨22476⟩ 76091

def event76096 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22477⟩⟩) (.product (.predecessor 0 76094 .coefficient) (.predecessor 1 76095 .coefficient) (⟨false, false, none, none, none⟩))

def event76097 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22477⟩⟩, .operator (⟨76093, 0⟩, ⟨76091, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22476⟩⟩]⟩, (1)⟩)

def exact76098RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22476⟩⟩]⟩, (1)⟩]

theorem exact76098RawTermsValid :
    exact76098RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76098 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22477⟩⟩) exact76098RawTerms .large 76096 .exactZero (none)

def event76099 : Event := .preFoldPolynomial 76098 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22476⟩⟩]⟩, (1)⟩] .exactZero none

def exact76100RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22476⟩⟩]⟩, (1)⟩]

def event76100 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨22477⟩⟩) 76099 exact76100RawTerms .large 76096 .exactZero (none)

def event76101 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨29588⟩⟩)

def event76102 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event76103 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event76104 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event76105 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event76106 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event76107 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event76108 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event76109 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event76110 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 76109

def event76111 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 76107

def event76112 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 76110 .coefficient) (.value (.predecessor 1 76111 .coefficient)))

def event76113 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event76114 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 76113

def event76115 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 76105

def event76116 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 76114 .coefficient, .predecessor 1 76115 .coefficient])

def event76117 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event76118 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 76117

def event76119 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 76103

def event76120 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 76119 .coefficient))

def event76121 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event76122 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12950⟩⟩) 0 ⟨5530⟩ 76121

def event76123 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12950⟩⟩) (.authority (.programFamilyFact))

def exact76124RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12950⟩⟩], []⟩, (1)⟩]

theorem exact76124RawTermsValid :
    exact76124RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76124 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12950⟩⟩) exact76124RawTerms (.finite 52) 76123 .exactZero (none)

def event76125 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10130⟩⟩) 0 ⟨5530⟩ 76121

def event76126 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10130⟩⟩) (.authority (.programFamilyFact))

def exact76127RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10130⟩⟩], []⟩, (1)⟩]

theorem exact76127RawTermsValid :
    exact76127RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76127 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10130⟩⟩) exact76127RawTerms (.finite 52) 76126 .exactZero (none)

def event76128 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12951⟩⟩) 0 ⟨10130⟩ 76127

def event76129 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12951⟩⟩) 1 ⟨12950⟩ 76124

def event76130 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12951⟩⟩) (.product (.predecessor 0 76128 .coefficient) (.predecessor 1 76129 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event76131 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12951⟩⟩, .operator (⟨76127, 0⟩, ⟨76124, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], []⟩, (1)⟩)

def exact76132RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], []⟩, (1)⟩]

theorem exact76132RawTermsValid :
    exact76132RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76132 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12951⟩⟩) exact76132RawTerms (.finite 2704) 76130 .exactZero (none)

def event76133 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12952⟩⟩) 0 ⟨12951⟩ 76132

def event76134 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12952⟩⟩) (.identity (.predecessor 0 76133 .coefficient))

def event76135 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12952⟩⟩) (.finite 2704)

def event76136 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16748⟩⟩) 0 ⟨12952⟩ 76135

def event76137 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16748⟩⟩) (.authority (.programFamilyFact))

def exact76138RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], []⟩, (1)⟩]

theorem exact76138RawTermsValid :
    exact76138RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76138 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16748⟩⟩) exact76138RawTerms (.finite 52) 76137 .exactZero (none)

def event76139 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16749⟩⟩) 0 ⟨16748⟩ 76138

def event76140 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16749⟩⟩) (.identity (.predecessor 0 76139 .coefficient))

def event76141 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16749⟩⟩) (.finite 52)

def event76142 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24661⟩⟩) 0 ⟨16749⟩ 76141

def event76143 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24661⟩⟩) (.authority (.programFamilyFact))

def event76144 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24661⟩⟩) (.finite 3720)

def event76145 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event76146 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24662⟩⟩) 0 ⟨6689⟩ 76145

def event76147 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24662⟩⟩) 1 ⟨24661⟩ 76144

def event76148 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24662⟩⟩) (.authority (.operator))

def exact76149RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24662⟩⟩]⟩, (1)⟩]

theorem exact76149RawTermsValid :
    exact76149RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76149 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24662⟩⟩) exact76149RawTerms .large 76148 .exactZero (none)

def event76150 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29582⟩⟩) 0 ⟨24662⟩ 76149

def event76151 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29582⟩⟩) (.authority (.operator))

def exact76152RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29582⟩⟩]⟩, (1)⟩]

theorem exact76152RawTermsValid :
    exact76152RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76152 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29582⟩⟩) exact76152RawTerms (.finite 8192) 76151 .exactZero (none)

def event76153 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event76154 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event76155 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16823⟩⟩) 0 ⟨16749⟩ 76141

def event76156 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16823⟩⟩) 1 ⟨110⟩ 76154

def event76157 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16823⟩⟩) (.sum [.predecessor 0 76155 .coefficient, .predecessor 1 76156 .coefficient])

def event76158 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16823⟩⟩) (.finite 52)

def event76159 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16824⟩⟩) 0 ⟨16823⟩ 76158

def event76160 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16824⟩⟩) (.identity (.predecessor 0 76159 .coefficient))

def exact76161RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], []⟩, (1)⟩]

theorem exact76161RawTermsValid :
    exact76161RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76161 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16824⟩⟩) exact76161RawTerms (.finite 52) 76160 .exactZero (none)

def event76162 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact76163RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact76163RawTermsValid :
    exact76163RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76163 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact76163RawTerms .large 76162 .exactZero (none)

def event76164 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16825⟩⟩) 0 ⟨6544⟩ 76163

def event76165 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16825⟩⟩) 1 ⟨16824⟩ 76161

def event76166 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16825⟩⟩) (.product (.predecessor 0 76164 .coefficient) (.predecessor 1 76165 .coefficient) (⟨false, false, none, none, none⟩))

def event76167 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16825⟩⟩, .operator (⟨76163, 0⟩, ⟨76161, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact76168RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact76168RawTermsValid :
    exact76168RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76168 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16825⟩⟩) exact76168RawTerms .large 76166 .exactZero (none)

def event76169 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6705⟩⟩) 0 ⟨6689⟩ 76145

def event76170 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6705⟩⟩) (.authority (.operator))

def exact76171RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩, (1)⟩]

theorem exact76171RawTermsValid :
    exact76171RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76171 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6705⟩⟩) exact76171RawTerms .large 76170 .exactZero (none)

def event76172 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16826⟩⟩) 0 ⟨6705⟩ 76171

def event76173 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16826⟩⟩) 1 ⟨16825⟩ 76168

def event76174 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16826⟩⟩) (.sum [.predecessor 0 76172 .coefficient, .predecessor 1 76173 .coefficient])

def exact76175RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact76175RawTermsValid :
    exact76175RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76175 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16826⟩⟩) exact76175RawTerms .large 76174 .exactZero (none)

def event76176 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29583⟩⟩) 0 ⟨16826⟩ 76175

def event76177 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29583⟩⟩) 1 ⟨29582⟩ 76152

def event76178 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29583⟩⟩) (.product (.predecessor 0 76176 .coefficient) (.predecessor 1 76177 .coefficient) (⟨false, false, none, none, none⟩))

def event76179 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29583⟩⟩, .operator (⟨76175, 0⟩, ⟨76152, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29582⟩⟩]⟩, (1)⟩)

def event76180 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29583⟩⟩, .operator (⟨76175, 1⟩, ⟨76152, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29582⟩⟩]⟩, (-1)⟩)

def event76181 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29583⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29582⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29582⟩⟩) ⟨24662⟩ 76149)

def event76182 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29583⟩⟩, .relation 76181 0, ⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24662⟩⟩]⟩, (-1)⟩)

def exact76183RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29582⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24662⟩⟩]⟩, (-1)⟩]

theorem exact76183RawTermsValid :
    exact76183RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76183 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29583⟩⟩) exact76183RawTerms .large 76178 .exactZero (none)

def event76184 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17490⟩⟩) 0 ⟨16749⟩ 76141

def event76185 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17490⟩⟩) (.authority (.programFamilyFact))

def exact76186RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17490⟩⟩], []⟩, (1)⟩]

theorem exact76186RawTermsValid :
    exact76186RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76186 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17490⟩⟩) exact76186RawTerms (.finite 52) 76185 .exactZero (none)

def event76187 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17492⟩⟩) 0 ⟨6544⟩ 76163

def event76188 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17492⟩⟩) 1 ⟨17490⟩ 76186

def event76189 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17492⟩⟩) (.product (.predecessor 0 76187 .coefficient) (.predecessor 1 76188 .coefficient) (⟨false, true, none, none, some 1⟩))

def event76190 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17492⟩⟩, .operator (⟨76163, 0⟩, ⟨76186, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17490⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact76191RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17490⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact76191RawTermsValid :
    exact76191RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76191 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17492⟩⟩) exact76191RawTerms .large 76189 .exactZero (none)

def event76192 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6738⟩⟩) 0 ⟨6689⟩ 76145

def event76193 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6738⟩⟩) (.authority (.operator))

def exact76194RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩]⟩, (1)⟩]

theorem exact76194RawTermsValid :
    exact76194RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76194 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6738⟩⟩) exact76194RawTerms .large 76193 .exactZero (none)

def event76195 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17493⟩⟩) 0 ⟨6738⟩ 76194

def event76196 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17493⟩⟩) 1 ⟨17492⟩ 76191

def event76197 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17493⟩⟩) (.sum [.predecessor 0 76195 .coefficient, .predecessor 1 76196 .coefficient])

def exact76198RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17490⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact76198RawTermsValid :
    exact76198RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76198 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17493⟩⟩) exact76198RawTerms .large 76197 .exactZero (none)

def event76199 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29588⟩⟩) 0 ⟨17493⟩ 76198

def event76200 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29588⟩⟩) 1 ⟨29583⟩ 76183

def event76201 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29588⟩⟩) (.sum [.predecessor 0 76199 .coefficient, .predecessor 1 76200 .coefficient])

def exact76202RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29582⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24662⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17490⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact76202RawTermsValid :
    exact76202RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76202 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29588⟩⟩) exact76202RawTerms .large 76201 .exactZero (none)

def event76203 : Event := .preFoldPolynomial 76202 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29582⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24662⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17490⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact76204RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29582⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24662⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17490⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event76204 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨29588⟩⟩) 76203 exact76204RawTerms .large 76201 .exactZero (none)

def event76205 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16749⟩⟩) ⟨⟨151⟩, ⟨60⟩, ⟨109⟩⟩ ⟨76047, 76205⟩

def event76206 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨22479⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22476⟩⟩]⟩) (1) 0 2 (.universal 76205 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22476⟩⟩]⟩) (none) 76204)

def event76207 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22479⟩⟩, .relation 76206 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6738⟩⟩]⟩, (1)⟩)

def event76208 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22479⟩⟩, .relation 76206 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29582⟩⟩]⟩, (-1)⟩)

def event76209 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22479⟩⟩, .relation 76206 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24662⟩⟩]⟩, (1)⟩)

def event76210 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22479⟩⟩, .relation 76206 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact76211RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29582⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6738⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24662⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact76211RawTermsValid :
    exact76211RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76211 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22479⟩⟩) exact76211RawTerms .large 76043 (.finite 1811303510016) (some (76045))

def event76212 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29585⟩⟩) 0 ⟨22479⟩ 76211

def event76213 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29585⟩⟩) 1 ⟨29584⟩ 76033

def event76214 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29585⟩⟩) (.sum [.predecessor 0 76212 .coefficient, .predecessor 1 76213 .coefficient])

def event76215 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29585⟩⟩, .operator (⟨76211, 0⟩, ⟨76033, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29582⟩⟩]⟩, (1)⟩)

def event76216 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29585⟩⟩, .operator (⟨76211, 2⟩, ⟨76033, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24662⟩⟩]⟩, (-1)⟩)

def event76217 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29585⟩⟩) (.sum [.result 76211 .summary, .result 76033 .summary])

def exact76218RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6738⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact76218RawTermsValid :
    exact76218RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76218 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29585⟩⟩) exact76218RawTerms .large 76214 (.finite 1292449485504936292352) (some (76217))

def event76219 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29586⟩⟩) 0 ⟨29585⟩ 76218

def event76220 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29586⟩⟩) 1 ⟨6662⟩ 5559

def event76221 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29586⟩⟩) (.product (.predecessor 0 76219 .coefficient) (.predecessor 1 76220 .coefficient) (⟨false, false, none, none, none⟩))

def event76222 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29586⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩) [⟨.result 5555 .coefficient, false, none⟩])

def event76223 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29586⟩⟩) (.product (.result 76218 .summary) (.transfer 76222) (⟨false, false, none, none, none⟩))

def event76224 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29586⟩⟩, .operator (⟨76218, 0⟩, ⟨5559, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩, (1)⟩)

def event76225 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29586⟩⟩, .operator (⟨76218, 1⟩, ⟨5559, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩, (-1)⟩)

def event76226 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29586⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6661⟩⟩) ⟨6602⟩ 5552)

def event76227 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29586⟩⟩, .relation 76226 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact76228RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17490⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact76228RawTermsValid :
    exact76228RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76228 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29586⟩⟩) exact76228RawTerms .large 76221 (.finite 4743310290994884271912517632) (some (76223))

def event76229 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24599⟩⟩) 0 ⟨6689⟩ 5477

def event76230 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24599⟩⟩) 1 ⟨24598⟩ 66735

def event76231 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24599⟩⟩) (.authority (.operator))

def exact76232RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24599⟩⟩]⟩, (1)⟩]

theorem exact76232RawTermsValid :
    exact76232RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76232 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24599⟩⟩) exact76232RawTerms .large 76231 .exactZero (none)

def event76233 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29365⟩⟩) 0 ⟨24599⟩ 76232

def event76234 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29365⟩⟩) (.authority (.operator))

def exact76235RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29365⟩⟩]⟩, (1)⟩]

theorem exact76235RawTermsValid :
    exact76235RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76235 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29365⟩⟩) exact76235RawTerms (.finite 8192) 76234 .exactZero (none)

def event76236 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29367⟩⟩) 0 ⟨25524⟩ 67019

def event76237 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29367⟩⟩) 1 ⟨29365⟩ 76235

def event76238 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29367⟩⟩) (.product (.predecessor 0 76236 .coefficient) (.predecessor 1 76237 .coefficient) (⟨false, false, none, none, none⟩))

def event76239 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29367⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨29365⟩⟩]⟩) [⟨.result 76235 .coefficient, false, none⟩])

def event76240 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29367⟩⟩) (.product (.result 67019 .summary) (.transfer 76239) (⟨false, false, none, none, none⟩))

def event76241 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29367⟩⟩, .operator (⟨67019, 0⟩, ⟨76235, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29365⟩⟩]⟩, (1)⟩)

def event76242 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29367⟩⟩, .operator (⟨67019, 1⟩, ⟨76235, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29365⟩⟩]⟩, (-1)⟩)

def event76243 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29367⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29365⟩⟩) ⟨24599⟩ 76232)

def event76244 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29367⟩⟩, .relation 76243 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨24599⟩⟩]⟩, (-1)⟩)

def exact76245RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨24599⟩⟩]⟩, (-1)⟩]

theorem exact76245RawTermsValid :
    exact76245RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76245 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29367⟩⟩) exact76245RawTerms .large 76238 (.finite 1292382246358571024384) (some (76240))

def event76246 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22332⟩⟩) 0 ⟨16630⟩ 3172

def event76247 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22332⟩⟩) (.authority (.relationPreimageSource ⟨58⟩))

def exact76248RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22332⟩⟩]⟩, (1)⟩]

theorem exact76248RawTermsValid :
    exact76248RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76248 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22332⟩⟩) exact76248RawTerms (.finite 136065468) 76247 .exactZero (none)

def event76249 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22334⟩⟩) 0 ⟨22332⟩ 76248

def event76250 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22334⟩⟩) 1 ⟨2348⟩ 4

def event76251 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22334⟩⟩) (.scale (.predecessor 0 76249 .coefficient) (.value (.predecessor 1 76250 .coefficient)))

def exact76252RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22332⟩⟩]⟩, (1)⟩]

theorem exact76252RawTermsValid :
    exact76252RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76252 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22334⟩⟩) exact76252RawTerms (.finite 136065468) 76251 .exactZero (none)

def event76253 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22335⟩⟩) 0 ⟨5535⟩ 65387

def event76254 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22335⟩⟩) 1 ⟨22334⟩ 76252

def event76255 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22335⟩⟩) (.product (.predecessor 0 76253 .coefficient) (.predecessor 1 76254 .coefficient) (⟨false, false, none, none, none⟩))

def event76256 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22335⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22332⟩⟩]⟩) [⟨.result 76248 .coefficient, false, none⟩])

def event76257 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22335⟩⟩) (.product (.result 65387 .summary) (.transfer 76256) (⟨false, false, none, none, none⟩))

def event76258 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22335⟩⟩, .operator (⟨65387, 0⟩, ⟨76252, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22332⟩⟩]⟩, (1)⟩)

def event76259 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22333⟩⟩)

def event76260 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event76261 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event76262 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event76263 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event76264 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event76265 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event76266 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event76267 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event76268 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 76267

def event76269 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 76265

def event76270 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 76268 .coefficient) (.value (.predecessor 1 76269 .coefficient)))

def event76271 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event76272 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 76271

def event76273 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 76263

def event76274 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 76272 .coefficient, .predecessor 1 76273 .coefficient])

def event76275 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event76276 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 76275

def event76277 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 76261

def event76278 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 76277 .coefficient))

def event76279 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event76280 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12754⟩⟩) 0 ⟨5530⟩ 76279

def event76281 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12754⟩⟩) (.authority (.programFamilyFact))

def exact76282RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12754⟩⟩], []⟩, (1)⟩]

theorem exact76282RawTermsValid :
    exact76282RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76282 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12754⟩⟩) exact76282RawTerms (.finite 46) 76281 .exactZero (none)

def event76283 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10025⟩⟩) 0 ⟨5530⟩ 76279

def event76284 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10025⟩⟩) (.authority (.programFamilyFact))

def exact76285RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩], []⟩, (1)⟩]

theorem exact76285RawTermsValid :
    exact76285RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76285 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10025⟩⟩) exact76285RawTerms (.finite 46) 76284 .exactZero (none)

def event76286 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12755⟩⟩) 0 ⟨10025⟩ 76285

def event76287 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12755⟩⟩) 1 ⟨12754⟩ 76282

def eventLeaf4752 : Array AnnotatedEvent := #[
  { event := event76032
    frameStart := 0 },
  { event := event76033
    frameStart := 0 },
  { event := event76034
    frameStart := 0 },
  { event := event76035
    frameStart := 0 },
  { event := event76036
    frameStart := 0 },
  { event := event76037
    frameStart := 0 },
  { event := event76038
    frameStart := 0 },
  { event := event76039
    frameStart := 0 },
  { event := event76040
    frameStart := 0 },
  { event := event76041
    frameStart := 0 },
  { event := event76042
    frameStart := 0 },
  { event := event76043
    frameStart := 0 },
  { event := event76044
    frameStart := 0 },
  { event := event76045
    frameStart := 0 },
  { event := event76046
    frameStart := 0 },
  { event := event76047
    frameStart := 76047 }
]

def eventLeaf4753 : Array AnnotatedEvent := #[
  { event := event76048
    frameStart := 76047 },
  { event := event76049
    frameStart := 76047 },
  { event := event76050
    frameStart := 76047 },
  { event := event76051
    frameStart := 76047 },
  { event := event76052
    frameStart := 76047 },
  { event := event76053
    frameStart := 76047 },
  { event := event76054
    frameStart := 76047 },
  { event := event76055
    frameStart := 76047 },
  { event := event76056
    frameStart := 76047 },
  { event := event76057
    frameStart := 76047 },
  { event := event76058
    frameStart := 76047 },
  { event := event76059
    frameStart := 76047 },
  { event := event76060
    frameStart := 76047 },
  { event := event76061
    frameStart := 76047 },
  { event := event76062
    frameStart := 76047 },
  { event := event76063
    frameStart := 76047 }
]

def eventLeaf4754 : Array AnnotatedEvent := #[
  { event := event76064
    frameStart := 76047 },
  { event := event76065
    frameStart := 76047 },
  { event := event76066
    frameStart := 76047 },
  { event := event76067
    frameStart := 76047 },
  { event := event76068
    frameStart := 76047 },
  { event := event76069
    frameStart := 76047 },
  { event := event76070
    frameStart := 76047 },
  { event := event76071
    frameStart := 76047 },
  { event := event76072
    frameStart := 76047 },
  { event := event76073
    frameStart := 76047 },
  { event := event76074
    frameStart := 76047 },
  { event := event76075
    frameStart := 76047 },
  { event := event76076
    frameStart := 76047 },
  { event := event76077
    frameStart := 76047 },
  { event := event76078
    frameStart := 76047 },
  { event := event76079
    frameStart := 76047 }
]

def eventLeaf4755 : Array AnnotatedEvent := #[
  { event := event76080
    frameStart := 76047 },
  { event := event76081
    frameStart := 76047 },
  { event := event76082
    frameStart := 76047 },
  { event := event76083
    frameStart := 76047 },
  { event := event76084
    frameStart := 76047 },
  { event := event76085
    frameStart := 76047 },
  { event := event76086
    frameStart := 76047 },
  { event := event76087
    frameStart := 76047 },
  { event := event76088
    frameStart := 76047 },
  { event := event76089
    frameStart := 76047 },
  { event := event76090
    frameStart := 76047 },
  { event := event76091
    frameStart := 76047 },
  { event := event76092
    frameStart := 76047 },
  { event := event76093
    frameStart := 76047 },
  { event := event76094
    frameStart := 76047 },
  { event := event76095
    frameStart := 76047 }
]

def eventLeaf4756 : Array AnnotatedEvent := #[
  { event := event76096
    frameStart := 76047 },
  { event := event76097
    frameStart := 76047 },
  { event := event76098
    frameStart := 76047 },
  { event := event76099
    frameStart := 76047 },
  { event := event76100
    frameStart := 76047 },
  { event := event76101
    frameStart := 76101 },
  { event := event76102
    frameStart := 76101 },
  { event := event76103
    frameStart := 76101 },
  { event := event76104
    frameStart := 76101 },
  { event := event76105
    frameStart := 76101 },
  { event := event76106
    frameStart := 76101 },
  { event := event76107
    frameStart := 76101 },
  { event := event76108
    frameStart := 76101 },
  { event := event76109
    frameStart := 76101 },
  { event := event76110
    frameStart := 76101 },
  { event := event76111
    frameStart := 76101 }
]

def eventLeaf4757 : Array AnnotatedEvent := #[
  { event := event76112
    frameStart := 76101 },
  { event := event76113
    frameStart := 76101 },
  { event := event76114
    frameStart := 76101 },
  { event := event76115
    frameStart := 76101 },
  { event := event76116
    frameStart := 76101 },
  { event := event76117
    frameStart := 76101 },
  { event := event76118
    frameStart := 76101 },
  { event := event76119
    frameStart := 76101 },
  { event := event76120
    frameStart := 76101 },
  { event := event76121
    frameStart := 76101 },
  { event := event76122
    frameStart := 76101 },
  { event := event76123
    frameStart := 76101 },
  { event := event76124
    frameStart := 76101 },
  { event := event76125
    frameStart := 76101 },
  { event := event76126
    frameStart := 76101 },
  { event := event76127
    frameStart := 76101 }
]

def eventLeaf4758 : Array AnnotatedEvent := #[
  { event := event76128
    frameStart := 76101 },
  { event := event76129
    frameStart := 76101 },
  { event := event76130
    frameStart := 76101 },
  { event := event76131
    frameStart := 76101 },
  { event := event76132
    frameStart := 76101 },
  { event := event76133
    frameStart := 76101 },
  { event := event76134
    frameStart := 76101 },
  { event := event76135
    frameStart := 76101 },
  { event := event76136
    frameStart := 76101 },
  { event := event76137
    frameStart := 76101 },
  { event := event76138
    frameStart := 76101 },
  { event := event76139
    frameStart := 76101 },
  { event := event76140
    frameStart := 76101 },
  { event := event76141
    frameStart := 76101 },
  { event := event76142
    frameStart := 76101 },
  { event := event76143
    frameStart := 76101 }
]

def eventLeaf4759 : Array AnnotatedEvent := #[
  { event := event76144
    frameStart := 76101 },
  { event := event76145
    frameStart := 76101 },
  { event := event76146
    frameStart := 76101 },
  { event := event76147
    frameStart := 76101 },
  { event := event76148
    frameStart := 76101 },
  { event := event76149
    frameStart := 76101 },
  { event := event76150
    frameStart := 76101 },
  { event := event76151
    frameStart := 76101 },
  { event := event76152
    frameStart := 76101 },
  { event := event76153
    frameStart := 76101 },
  { event := event76154
    frameStart := 76101 },
  { event := event76155
    frameStart := 76101 },
  { event := event76156
    frameStart := 76101 },
  { event := event76157
    frameStart := 76101 },
  { event := event76158
    frameStart := 76101 },
  { event := event76159
    frameStart := 76101 }
]

def eventLeaf4760 : Array AnnotatedEvent := #[
  { event := event76160
    frameStart := 76101 },
  { event := event76161
    frameStart := 76101 },
  { event := event76162
    frameStart := 76101 },
  { event := event76163
    frameStart := 76101 },
  { event := event76164
    frameStart := 76101 },
  { event := event76165
    frameStart := 76101 },
  { event := event76166
    frameStart := 76101 },
  { event := event76167
    frameStart := 76101 },
  { event := event76168
    frameStart := 76101 },
  { event := event76169
    frameStart := 76101 },
  { event := event76170
    frameStart := 76101 },
  { event := event76171
    frameStart := 76101 },
  { event := event76172
    frameStart := 76101 },
  { event := event76173
    frameStart := 76101 },
  { event := event76174
    frameStart := 76101 },
  { event := event76175
    frameStart := 76101 }
]

def eventLeaf4761 : Array AnnotatedEvent := #[
  { event := event76176
    frameStart := 76101 },
  { event := event76177
    frameStart := 76101 },
  { event := event76178
    frameStart := 76101 },
  { event := event76179
    frameStart := 76101 },
  { event := event76180
    frameStart := 76101 },
  { event := event76181
    frameStart := 76101 },
  { event := event76182
    frameStart := 76101 },
  { event := event76183
    frameStart := 76101 },
  { event := event76184
    frameStart := 76101 },
  { event := event76185
    frameStart := 76101 },
  { event := event76186
    frameStart := 76101 },
  { event := event76187
    frameStart := 76101 },
  { event := event76188
    frameStart := 76101 },
  { event := event76189
    frameStart := 76101 },
  { event := event76190
    frameStart := 76101 },
  { event := event76191
    frameStart := 76101 }
]

def eventLeaf4762 : Array AnnotatedEvent := #[
  { event := event76192
    frameStart := 76101 },
  { event := event76193
    frameStart := 76101 },
  { event := event76194
    frameStart := 76101 },
  { event := event76195
    frameStart := 76101 },
  { event := event76196
    frameStart := 76101 },
  { event := event76197
    frameStart := 76101 },
  { event := event76198
    frameStart := 76101 },
  { event := event76199
    frameStart := 76101 },
  { event := event76200
    frameStart := 76101 },
  { event := event76201
    frameStart := 76101 },
  { event := event76202
    frameStart := 76101 },
  { event := event76203
    frameStart := 76101 },
  { event := event76204
    frameStart := 76101 },
  { event := event76205
    frameStart := 0 },
  { event := event76206
    frameStart := 0 },
  { event := event76207
    frameStart := 0 }
]

def eventLeaf4763 : Array AnnotatedEvent := #[
  { event := event76208
    frameStart := 0 },
  { event := event76209
    frameStart := 0 },
  { event := event76210
    frameStart := 0 },
  { event := event76211
    frameStart := 0 },
  { event := event76212
    frameStart := 0 },
  { event := event76213
    frameStart := 0 },
  { event := event76214
    frameStart := 0 },
  { event := event76215
    frameStart := 0 },
  { event := event76216
    frameStart := 0 },
  { event := event76217
    frameStart := 0 },
  { event := event76218
    frameStart := 0 },
  { event := event76219
    frameStart := 0 },
  { event := event76220
    frameStart := 0 },
  { event := event76221
    frameStart := 0 },
  { event := event76222
    frameStart := 0 },
  { event := event76223
    frameStart := 0 }
]

def eventLeaf4764 : Array AnnotatedEvent := #[
  { event := event76224
    frameStart := 0 },
  { event := event76225
    frameStart := 0 },
  { event := event76226
    frameStart := 0 },
  { event := event76227
    frameStart := 0 },
  { event := event76228
    frameStart := 0 },
  { event := event76229
    frameStart := 0 },
  { event := event76230
    frameStart := 0 },
  { event := event76231
    frameStart := 0 },
  { event := event76232
    frameStart := 0 },
  { event := event76233
    frameStart := 0 },
  { event := event76234
    frameStart := 0 },
  { event := event76235
    frameStart := 0 },
  { event := event76236
    frameStart := 0 },
  { event := event76237
    frameStart := 0 },
  { event := event76238
    frameStart := 0 },
  { event := event76239
    frameStart := 0 }
]

def eventLeaf4765 : Array AnnotatedEvent := #[
  { event := event76240
    frameStart := 0 },
  { event := event76241
    frameStart := 0 },
  { event := event76242
    frameStart := 0 },
  { event := event76243
    frameStart := 0 },
  { event := event76244
    frameStart := 0 },
  { event := event76245
    frameStart := 0 },
  { event := event76246
    frameStart := 0 },
  { event := event76247
    frameStart := 0 },
  { event := event76248
    frameStart := 0 },
  { event := event76249
    frameStart := 0 },
  { event := event76250
    frameStart := 0 },
  { event := event76251
    frameStart := 0 },
  { event := event76252
    frameStart := 0 },
  { event := event76253
    frameStart := 0 },
  { event := event76254
    frameStart := 0 },
  { event := event76255
    frameStart := 0 }
]

def eventLeaf4766 : Array AnnotatedEvent := #[
  { event := event76256
    frameStart := 0 },
  { event := event76257
    frameStart := 0 },
  { event := event76258
    frameStart := 0 },
  { event := event76259
    frameStart := 76259 },
  { event := event76260
    frameStart := 76259 },
  { event := event76261
    frameStart := 76259 },
  { event := event76262
    frameStart := 76259 },
  { event := event76263
    frameStart := 76259 },
  { event := event76264
    frameStart := 76259 },
  { event := event76265
    frameStart := 76259 },
  { event := event76266
    frameStart := 76259 },
  { event := event76267
    frameStart := 76259 },
  { event := event76268
    frameStart := 76259 },
  { event := event76269
    frameStart := 76259 },
  { event := event76270
    frameStart := 76259 },
  { event := event76271
    frameStart := 76259 }
]

def eventLeaf4767 : Array AnnotatedEvent := #[
  { event := event76272
    frameStart := 76259 },
  { event := event76273
    frameStart := 76259 },
  { event := event76274
    frameStart := 76259 },
  { event := event76275
    frameStart := 76259 },
  { event := event76276
    frameStart := 76259 },
  { event := event76277
    frameStart := 76259 },
  { event := event76278
    frameStart := 76259 },
  { event := event76279
    frameStart := 76259 },
  { event := event76280
    frameStart := 76259 },
  { event := event76281
    frameStart := 76259 },
  { event := event76282
    frameStart := 76259 },
  { event := event76283
    frameStart := 76259 },
  { event := event76284
    frameStart := 76259 },
  { event := event76285
    frameStart := 76259 },
  { event := event76286
    frameStart := 76259 },
  { event := event76287
    frameStart := 76259 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events297
