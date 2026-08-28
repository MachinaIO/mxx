import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events637

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event163072 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17195⟩⟩) 0 ⟨17194⟩ 163071

def event163073 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17195⟩⟩) (.identity (.predecessor 0 163072 .coefficient))

def exact163074RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], []⟩, (1)⟩]

theorem exact163074RawTermsValid :
    exact163074RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163074 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17195⟩⟩) exact163074RawTerms (.finite 2) 163073 .exactZero (none)

def event163075 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact163076RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact163076RawTermsValid :
    exact163076RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163076 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact163076RawTerms .large 163075 .exactZero (none)

def event163077 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17196⟩⟩) 0 ⟨6908⟩ 163076

def event163078 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17196⟩⟩) 1 ⟨17195⟩ 163074

def event163079 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17196⟩⟩) (.product (.predecessor 0 163077 .coefficient) (.predecessor 1 163078 .coefficient) (⟨false, false, none, none, none⟩))

def event163080 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17196⟩⟩, .operator (⟨163076, 0⟩, ⟨163074, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact163081RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact163081RawTermsValid :
    exact163081RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163081 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17196⟩⟩) exact163081RawTerms .large 163079 .exactZero (none)

def event163082 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 163058

def event163083 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact163084RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact163084RawTermsValid :
    exact163084RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163084 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact163084RawTerms .large 163083 .exactZero (none)

def event163085 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17197⟩⟩) 0 ⟨7179⟩ 163084

def event163086 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17197⟩⟩) 1 ⟨17196⟩ 163081

def event163087 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17197⟩⟩) (.sum [.predecessor 0 163085 .coefficient, .predecessor 1 163086 .coefficient])

def exact163088RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163088RawTermsValid :
    exact163088RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163088 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17197⟩⟩) exact163088RawTerms .large 163087 .exactZero (none)

def event163089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17671⟩⟩) 0 ⟨17197⟩ 163088

def event163090 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17671⟩⟩) 1 ⟨17670⟩ 163065

def event163091 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17671⟩⟩) (.product (.predecessor 0 163089 .coefficient) (.predecessor 1 163090 .coefficient) (⟨false, false, none, none, none⟩))

def event163092 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17671⟩⟩, .operator (⟨163088, 0⟩, ⟨163065, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩]⟩, (1)⟩)

def event163093 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17671⟩⟩, .operator (⟨163088, 1⟩, ⟨163065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩]⟩, (-1)⟩)

def event163094 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17671⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17670⟩⟩) ⟨16973⟩ 163062)

def event163095 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17671⟩⟩, .relation 163094 0, ⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨16973⟩⟩]⟩, (-1)⟩)

def exact163096RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨16973⟩⟩]⟩, (-1)⟩]

theorem exact163096RawTermsValid :
    exact163096RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163096 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17671⟩⟩) exact163096RawTerms .large 163091 .exactZero (none)

def event163097 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15982⟩⟩) 0 ⟨15765⟩ 163054

def event163098 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15982⟩⟩) (.authority (.programFamilyFact))

def exact163099RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩]

theorem exact163099RawTermsValid :
    exact163099RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163099 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15982⟩⟩) exact163099RawTerms (.finite 2) 163098 .exactZero (none)

def event163100 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15985⟩⟩) 0 ⟨6908⟩ 163076

def event163101 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15985⟩⟩) 1 ⟨15982⟩ 163099

def event163102 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15985⟩⟩) (.product (.predecessor 0 163100 .coefficient) (.predecessor 1 163101 .coefficient) (⟨false, true, none, none, some 1⟩))

def event163103 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15985⟩⟩, .operator (⟨163076, 0⟩, ⟨163099, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact163104RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact163104RawTermsValid :
    exact163104RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163104 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15985⟩⟩) exact163104RawTerms .large 163102 .exactZero (none)

def event163105 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7197⟩⟩) 0 ⟨7177⟩ 163058

def event163106 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7197⟩⟩) (.authority (.operator))

def exact163107RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩]

theorem exact163107RawTermsValid :
    exact163107RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163107 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7197⟩⟩) exact163107RawTerms .large 163106 .exactZero (none)

def event163108 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15986⟩⟩) 0 ⟨7197⟩ 163107

def event163109 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15986⟩⟩) 1 ⟨15985⟩ 163104

def event163110 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15986⟩⟩) (.sum [.predecessor 0 163108 .coefficient, .predecessor 1 163109 .coefficient])

def exact163111RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163111RawTermsValid :
    exact163111RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163111 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15986⟩⟩) exact163111RawTerms .large 163110 .exactZero (none)

def event163112 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17676⟩⟩) 0 ⟨15986⟩ 163111

def event163113 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17676⟩⟩) 1 ⟨17671⟩ 163096

def event163114 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17676⟩⟩) (.sum [.predecessor 0 163112 .coefficient, .predecessor 1 163113 .coefficient])

def exact163115RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨16973⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163115RawTermsValid :
    exact163115RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163115 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17676⟩⟩) exact163115RawTerms .large 163114 .exactZero (none)

def event163116 : Event := .preFoldPolynomial 163115 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨16973⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact163117RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨16973⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event163117 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17676⟩⟩) 163116 exact163117RawTerms .large 163114 .exactZero (none)

def event163118 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15765⟩⟩) ⟨⟨76⟩, ⟨56⟩, ⟨135⟩⟩ ⟨162960, 163118⟩

def event163119 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16535⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16532⟩⟩]⟩) (1) 0 2 (.universal 163118 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16532⟩⟩]⟩) (none) 163117)

def event163120 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16535⟩⟩, .relation 163119 1, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩)

def event163121 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16535⟩⟩, .relation 163119 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩]⟩, (-1)⟩)

def event163122 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16535⟩⟩, .relation 163119 2, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨16973⟩⟩]⟩, (1)⟩)

def event163123 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16535⟩⟩, .relation 163119 3, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact163124RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨16973⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163124RawTermsValid :
    exact163124RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163124 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16535⟩⟩) exact163124RawTerms .large 162956 (.finite 202072841853861888) (some (162958))

def event163125 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17673⟩⟩) 0 ⟨16535⟩ 163124

def event163126 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17673⟩⟩) 1 ⟨17672⟩ 162946

def event163127 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17673⟩⟩) (.sum [.predecessor 0 163125 .coefficient, .predecessor 1 163126 .coefficient])

def event163128 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17673⟩⟩, .operator (⟨163124, 0⟩, ⟨162946, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17670⟩⟩]⟩, (1)⟩)

def event163129 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17673⟩⟩, .operator (⟨163124, 2⟩, ⟨162946, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15764⟩⟩], [⟨.program ⟨257⟩, ⟨16973⟩⟩]⟩, (-1)⟩)

def event163130 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17673⟩⟩) (.sum [.result 163124 .summary, .result 162946 .summary])

def exact163131RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163131RawTermsValid :
    exact163131RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163131 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17673⟩⟩) exact163131RawTerms .large 163127 (.finite 32188807212483706889510625476608) (some (163130))

def event163132 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17674⟩⟩) 0 ⟨17673⟩ 163131

def event163133 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17674⟩⟩) 1 ⟨7172⟩ 15882

def event163134 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17674⟩⟩) (.product (.predecessor 0 163132 .coefficient) (.predecessor 1 163133 .coefficient) (⟨false, false, none, none, none⟩))

def event163135 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17674⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) [⟨.result 15878 .coefficient, false, none⟩])

def event163136 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17674⟩⟩) (.product (.result 163131 .summary) (.transfer 163135) (⟨false, false, none, none, none⟩))

def event163137 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17674⟩⟩, .operator (⟨163131, 0⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩)

def event163138 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17674⟩⟩, .operator (⟨163131, 1⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩)

def event163139 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17674⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7171⟩⟩) ⟨7051⟩ 15875)

def event163140 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17674⟩⟩, .relation 163139 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact163141RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163141RawTermsValid :
    exact163141RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163141 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17674⟩⟩) exact163141RawTerms .large 163134 (.finite 345624685687166110058245054666339432529920) (some (163136))

def event163142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7076⟩⟩) 0 ⟨6727⟩ 723

def event163143 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7076⟩⟩) 1 ⟨6931⟩ 149028

def event163144 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7076⟩⟩) (.tensor (.predecessor 0 163142 .coefficient) (.predecessor 1 163143 .coefficient) true false)

def event163145 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7076⟩⟩, .operator (⟨723, 0⟩, ⟨149028, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact163146RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact163146RawTermsValid :
    exact163146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163146 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7076⟩⟩) exact163146RawTerms .large 163144 .exactZero (none)

def event163147 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8256⟩⟩) 0 ⟨5543⟩ 148898

def event163148 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8256⟩⟩) 1 ⟨7292⟩ 15896

def event163149 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8256⟩⟩) (.product (.predecessor 0 163147 .coefficient) (.predecessor 1 163148 .coefficient) (⟨false, false, none, none, none⟩))

def event163150 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8256⟩⟩, .operator (⟨148898, 0⟩, ⟨15896, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩)

def exact163151RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact163151RawTermsValid :
    exact163151RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163151 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8256⟩⟩) exact163151RawTerms .large 163149 .exactZero (none)

def event163152 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9353⟩⟩) 0 ⟨8256⟩ 163151

def event163153 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9353⟩⟩) 1 ⟨7076⟩ 163146

def event163154 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9353⟩⟩) (.sum [.predecessor 0 163152 .coefficient, .predecessor 1 163153 .coefficient])

def exact163155RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163155RawTermsValid :
    exact163155RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163155 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9353⟩⟩) exact163155RawTerms .large 163154 .exactZero (none)

def event163156 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9354⟩⟩) 0 ⟨9353⟩ 163155

def event163157 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9354⟩⟩) 1 ⟨118⟩ 31516

def event163158 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9354⟩⟩) (.sum [.predecessor 0 163156 .coefficient, .predecessor 1 163157 .coefficient])

def event163159 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9354⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) [⟨.result 31516 .coefficient, false, none⟩])

def event163160 : Event := .survivorFold (1) 163159

def exact163161RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163161RawTermsValid :
    exact163161RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163161 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9354⟩⟩) exact163161RawTerms .large 163158 (.finite 26) (some (163159))

def event163162 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9468⟩⟩) 0 ⟨9354⟩ 163161

def event163163 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9468⟩⟩) 1 ⟨9354⟩ 163161

def event163164 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9468⟩⟩) (.sum [.predecessor 0 163162 .coefficient, .predecessor 1 163163 .coefficient])

def event163165 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9468⟩⟩, .operator (⟨163161, 1⟩, ⟨163161, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event163166 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9468⟩⟩, .operator (⟨163161, 0⟩, ⟨163161, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (-1)⟩)

def event163167 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9468⟩⟩) (.sum [.result 163161 .summary, .result 163161 .summary])

def exact163168RawTerms : List Term := []

theorem exact163168RawTermsValid :
    exact163168RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163168 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9468⟩⟩) exact163168RawTerms .large 163164 (.finite 52) (some (163167))

def event163169 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17675⟩⟩) 0 ⟨9468⟩ 163168

def event163170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17675⟩⟩) 1 ⟨17674⟩ 163141

def event163171 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17675⟩⟩) (.sum [.predecessor 0 163169 .coefficient, .predecessor 1 163170 .coefficient])

def event163172 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17675⟩⟩) (.sum [.result 163168 .summary, .result 163141 .summary])

def exact163173RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163173RawTermsValid :
    exact163173RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163173 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17675⟩⟩) exact163173RawTerms .large 163171 (.finite 345624685687166110058245054666339432529972) (some (163172))

def event163174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20557⟩⟩) 0 ⟨17675⟩ 163173

def event163175 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20557⟩⟩) 1 ⟨20556⟩ 162929

def event163176 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20557⟩⟩) (.sum [.predecessor 0 163174 .coefficient, .predecessor 1 163175 .coefficient])

def event163177 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20557⟩⟩) (.sum [.result 163173 .summary, .result 162929 .summary])

def exact163178RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163178RawTermsValid :
    exact163178RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163178 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20557⟩⟩) exact163178RawTerms .large 163176 (.finite 691250426059631610003352154589745737891892) (some (163177))

def event163179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23777⟩⟩) 0 ⟨20557⟩ 163178

def event163180 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23777⟩⟩) 1 ⟨23776⟩ 162717

def event163181 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23777⟩⟩) (.sum [.predecessor 0 163179 .coefficient, .predecessor 1 163180 .coefficient])

def event163182 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23777⟩⟩) (.sum [.result 163178 .summary, .result 162717 .summary])

def exact163183RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163183RawTermsValid :
    exact163183RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163183 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23777⟩⟩) exact163183RawTerms .large 163181 (.finite 1036877221117396499835321299770218916085812) (some (163182))

def event163184 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33797⟩⟩) 0 ⟨23777⟩ 163183

def event163185 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33797⟩⟩) 1 ⟨33796⟩ 162505

def event163186 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33797⟩⟩) (.sum [.predecessor 0 163184 .coefficient, .predecessor 1 163185 .coefficient])

def event163187 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33797⟩⟩) (.sum [.result 163183 .summary, .result 162505 .summary])

def exact163188RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163188RawTermsValid :
    exact163188RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163188 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33797⟩⟩) exact163188RawTerms .large 163186 (.finite 1382506125545760169441014535464825839943732) (some (163187))

def event163189 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52857⟩⟩) 0 ⟨33797⟩ 163188

def event163190 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52857⟩⟩) 1 ⟨52856⟩ 162293

def event163191 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52857⟩⟩) (.sum [.predecessor 0 163189 .coefficient, .predecessor 1 163190 .coefficient])

def event163192 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52857⟩⟩) (.sum [.result 163188 .summary, .result 162293 .summary])

def exact163193RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163193RawTermsValid :
    exact163193RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163193 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52857⟩⟩) exact163193RawTerms .large 163191 (.finite 1728139248715321398594155952187700255129652) (some (163192))

def event163194 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55837⟩⟩) 0 ⟨52857⟩ 163193

def event163195 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55837⟩⟩) 1 ⟨55836⟩ 162081

def event163196 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55837⟩⟩) (.sum [.predecessor 0 163194 .coefficient, .predecessor 1 163195 .coefficient])

def event163197 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55837⟩⟩) (.sum [.result 163193 .summary, .result 162081 .summary])

def exact163198RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163198RawTermsValid :
    exact163198RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163198 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55837⟩⟩) exact163198RawTerms .large 163196 (.finite 2073774481255481407521021459424708415979572) (some (163197))

def event163199 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58817⟩⟩) 0 ⟨55837⟩ 163198

def event163200 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58817⟩⟩) 1 ⟨58816⟩ 161869

def event163201 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58817⟩⟩) (.sum [.predecessor 0 163199 .coefficient, .predecessor 1 163200 .coefficient])

def event163202 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58817⟩⟩) (.sum [.result 163198 .summary, .result 161869 .summary])

def exact163203RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163203RawTermsValid :
    exact163203RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163203 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58817⟩⟩) exact163203RawTerms .large 163201 (.finite 2419413932536838975995335147689984068157492) (some (163202))

def event163204 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61797⟩⟩) 0 ⟨58817⟩ 163203

def event163205 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61797⟩⟩) 1 ⟨61796⟩ 161657

def event163206 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61797⟩⟩) (.sum [.predecessor 0 163204 .coefficient, .predecessor 1 163205 .coefficient])

def event163207 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61797⟩⟩) (.sum [.result 163203 .summary, .result 161657 .summary])

def exact163208RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163208RawTermsValid :
    exact163208RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163208 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61797⟩⟩) exact163208RawTerms .large 163206 (.finite 2765055493188795324243372926469393465999412) (some (163207))

def event163209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64777⟩⟩) 0 ⟨61797⟩ 163208

def event163210 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64777⟩⟩) 1 ⟨64776⟩ 161445

def event163211 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64777⟩⟩) (.sum [.predecessor 0 163209 .coefficient, .predecessor 1 163210 .coefficient])

def event163212 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64777⟩⟩) (.sum [.result 163208 .summary, .result 161445 .summary])

def exact163213RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163213RawTermsValid :
    exact163213RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163213 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64777⟩⟩) exact163213RawTerms .large 163211 (.finite 3110701272581949232038858886277070355169332) (some (163212))

def event163214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69930⟩⟩) 0 ⟨64777⟩ 163213

def event163215 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69930⟩⟩) 1 ⟨69929⟩ 161233

def event163216 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69930⟩⟩) (.sum [.predecessor 0 163214 .coefficient, .predecessor 1 163215 .coefficient])

def event163217 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69930⟩⟩) (.sum [.result 163213 .summary, .result 161233 .summary])

def exact163218RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163218RawTermsValid :
    exact163218RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163218 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69930⟩⟩) exact163218RawTerms .large 163216 (.finite 3456353380086899479155517117627148481331252) (some (163217))

def event163219 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69931⟩⟩) 0 ⟨69930⟩ 163218

def event163220 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69931⟩⟩) 1 ⟨28212⟩ 161021

def event163221 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69931⟩⟩) (.sum [.predecessor 0 163219 .coefficient, .predecessor 1 163220 .coefficient])

def event163222 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69931⟩⟩) (.sum [.result 163218 .summary, .result 161021 .summary])

def exact163223RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163223RawTermsValid :
    exact163223RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163223 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69931⟩⟩) exact163223RawTerms .large 163221 (.finite 3802007596962448506045899439491360353157172) (some (163222))

def event163224 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69932⟩⟩) 0 ⟨69931⟩ 163223

def event163225 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69932⟩⟩) 1 ⟨30892⟩ 160809

def event163226 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69932⟩⟩) (.sum [.predecessor 0 163224 .coefficient, .predecessor 1 163225 .coefficient])

def event163227 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69932⟩⟩) (.sum [.result 163223 .summary, .result 160809 .summary])

def exact163228RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163228RawTermsValid :
    exact163228RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163228 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69932⟩⟩) exact163228RawTerms .large 163226 (.finite 4147668141949793872257454032897973461975092) (some (163227))

def event163229 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69933⟩⟩) 0 ⟨69932⟩ 163228

def event163230 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69933⟩⟩) 1 ⟨36552⟩ 160597

def event163231 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69933⟩⟩) (.sum [.predecessor 0 163229 .coefficient, .predecessor 1 163230 .coefficient])

def event163232 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69933⟩⟩) (.sum [.result 163228 .summary, .result 160597 .summary])

def exact163233RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163233RawTermsValid :
    exact163233RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163233 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69933⟩⟩) exact163233RawTerms .large 163231 (.finite 4493332905678336798016456807332854062121012) (some (163232))

def event163234 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69934⟩⟩) 0 ⟨69933⟩ 163233

def event163235 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69934⟩⟩) 1 ⟨39232⟩ 160385

def event163236 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69934⟩⟩) (.sum [.predecessor 0 163234 .coefficient, .predecessor 1 163235 .coefficient])

def event163237 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69934⟩⟩) (.sum [.result 163233 .summary, .result 160385 .summary])

def exact163238RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163238RawTermsValid :
    exact163238RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163238 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69934⟩⟩) exact163238RawTerms .large 163236 (.finite 4838999778777478503549183672281868407930932) (some (163237))

def event163239 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69935⟩⟩) 0 ⟨69934⟩ 163238

def event163240 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69935⟩⟩) 1 ⟨41912⟩ 160173

def event163241 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69935⟩⟩) (.sum [.predecessor 0 163239 .coefficient, .predecessor 1 163240 .coefficient])

def event163242 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69935⟩⟩) (.sum [.result 163238 .summary, .result 160173 .summary])

def exact163243RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163243RawTermsValid :
    exact163243RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163243 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69935⟩⟩) exact163243RawTerms .large 163241 (.finite 5184670870617817768629358718259150245068852) (some (163242))

def event163244 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69936⟩⟩) 0 ⟨69935⟩ 163243

def event163245 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69936⟩⟩) 1 ⟨44592⟩ 159961

def event163246 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69936⟩⟩) (.sum [.predecessor 0 163244 .coefficient, .predecessor 1 163245 .coefficient])

def event163247 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69936⟩⟩) (.sum [.result 163243 .summary, .result 159961 .summary])

def exact163248RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42963⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163248RawTermsValid :
    exact163248RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163248 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69936⟩⟩) exact163248RawTerms .large 163246 (.finite 5530348290569953373030706035778833319198772) (some (163247))

def event163249 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69937⟩⟩) 0 ⟨69936⟩ 163248

def event163250 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69937⟩⟩) 1 ⟨47272⟩ 159749

def event163251 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69937⟩⟩) (.sum [.predecessor 0 163249 .coefficient, .predecessor 1 163250 .coefficient])

def event163252 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69937⟩⟩) (.sum [.result 163248 .summary, .result 159749 .summary])

def exact163253RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45640⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42963⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163253RawTermsValid :
    exact163253RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163253 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69937⟩⟩) exact163253RawTerms .large 163251 (.finite 5876032038633885316753225624840917630320692) (some (163252))

def event163254 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69938⟩⟩) 0 ⟨69937⟩ 163253

def event163255 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69938⟩⟩) 1 ⟨49952⟩ 159537

def event163256 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69938⟩⟩) (.sum [.predecessor 0 163254 .coefficient, .predecessor 1 163255 .coefficient])

def event163257 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69938⟩⟩) (.sum [.result 163253 .summary, .result 159537 .summary])

def exact163258RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48320⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45640⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42963⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163258RawTermsValid :
    exact163258RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163258 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69938⟩⟩) exact163258RawTerms .large 163256 (.finite 6221717896068416040249469304417135687106612) (some (163257))

def event163259 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71148⟩⟩) 0 ⟨69938⟩ 163258

def event163260 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71148⟩⟩) 1 ⟨71146⟩ 159325

def event163261 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71148⟩⟩) (.sum [.predecessor 0 163259 .coefficient, .predecessor 1 163260 .coefficient])

def event163262 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71148⟩⟩) (.sum [.result 163258 .summary, .result 159325 .summary])

def exact163263RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48320⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45640⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42963⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163263RawTermsValid :
    exact163263RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163263 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71148⟩⟩) exact163263RawTerms .large 163261 (.finite 66805187227601152574551644069558752530002096506798132) (some (163262))

def event163264 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13⟩⟩) (.authority (.operator))

def exact163265RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨13⟩⟩]⟩, (1)⟩]

theorem exact163265RawTermsValid :
    exact163265RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163265 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13⟩⟩) exact163265RawTerms (.finite 26) 163264 .exactZero (none)

def event163266 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7410⟩⟩) 0 ⟨2377⟩ 27

def event163267 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7410⟩⟩) 1 ⟨7252⟩ 16347

def event163268 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7410⟩⟩) (.product (.predecessor 0 163266 .coefficient) (.predecessor 1 163267 .coefficient) (⟨false, false, none, none, none⟩))

def event163269 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7410⟩⟩, .operator (⟨27, 0⟩, ⟨16347, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7252⟩⟩]⟩, (1)⟩)

def exact163270RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7252⟩⟩]⟩, (1)⟩]

theorem exact163270RawTermsValid :
    exact163270RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163270 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7410⟩⟩) exact163270RawTerms .large 163268 .exactZero (none)

def event163271 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9223⟩⟩) 0 ⟨7410⟩ 163270

def event163272 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9223⟩⟩) 1 ⟨6931⟩ 149028

def event163273 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9223⟩⟩) (.sum [.predecessor 0 163271 .coefficient, .predecessor 1 163272 .coefficient])

def exact163274RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7252⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163274RawTermsValid :
    exact163274RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163274 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9223⟩⟩) exact163274RawTerms .large 163273 .exactZero (none)

def event163275 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9224⟩⟩) 0 ⟨9223⟩ 163274

def event163276 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9224⟩⟩) 1 ⟨13⟩ 163265

def event163277 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9224⟩⟩) (.sum [.predecessor 0 163275 .coefficient, .predecessor 1 163276 .coefficient])

def event163278 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9224⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨13⟩⟩]⟩) [⟨.result 163265 .coefficient, false, none⟩])

def event163279 : Event := .survivorFold (1) 163278

def exact163280RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7252⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact163280RawTermsValid :
    exact163280RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163280 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9224⟩⟩) exact163280RawTerms .large 163277 (.finite 26) (some (163278))

def event163281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9622⟩⟩) 0 ⟨9224⟩ 163280

def event163282 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9622⟩⟩) 1 ⟨9584⟩ 15984

def event163283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9622⟩⟩) (.product (.predecessor 0 163281 .coefficient) (.predecessor 1 163282 .coefficient) (⟨false, false, none, none, none⟩))

def event163284 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9622⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) [⟨.result 15980 .coefficient, false, none⟩])

def event163285 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9622⟩⟩) (.product (.result 163280 .summary) (.transfer 163284) (⟨false, false, none, none, none⟩))

def event163286 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .operator (⟨163280, 1⟩, ⟨15984, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩, (-1)⟩)

def event163287 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨9622⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9583⟩⟩) ⟨9443⟩ 15977)

def event163288 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 18, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (1)⟩)

def event163289 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 17, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (-1)⟩)

def event163290 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 16, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (-1)⟩)

def event163291 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 15, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (-1)⟩)

def event163292 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 14, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (-1)⟩)

def event163293 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 13, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (-1)⟩)

def event163294 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 12, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (-1)⟩)

def event163295 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 11, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (-1)⟩)

def event163296 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 10, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (-1)⟩)

def event163297 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 9, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (-1)⟩)

def event163298 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 8, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (-1)⟩)

def event163299 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 7, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (-1)⟩)

def event163300 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 6, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (-1)⟩)

def event163301 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 5, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (-1)⟩)

def event163302 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 4, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (-1)⟩)

def event163303 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 3, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (-1)⟩)

def event163304 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 2, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (-1)⟩)

def event163305 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 1, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (-1)⟩)

def event163306 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .relation 163287 0, ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩)

def event163307 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9622⟩⟩, .operator (⟨163280, 0⟩, ⟨15984, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7252⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩, (1)⟩)

def exact163308RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7252⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (1)⟩]

theorem exact163308RawTermsValid :
    exact163308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event163308 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9622⟩⟩) exact163308RawTerms .large 163283 (.finite 279172874240) (some (163285))

def event163309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71149⟩⟩) 0 ⟨9622⟩ 163308

def event163310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71149⟩⟩) 1 ⟨71148⟩ 163263

def event163311 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71149⟩⟩) (.sum [.predecessor 0 163309 .coefficient, .predecessor 1 163310 .coefficient])

def event163312 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71149⟩⟩, .operator (⟨163308, 19⟩, ⟨163263, 18⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (-1)⟩)

def event163313 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71149⟩⟩, .operator (⟨163308, 18⟩, ⟨163263, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩)

def event163314 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71149⟩⟩, .operator (⟨163308, 17⟩, ⟨163263, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩)

def event163315 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71149⟩⟩, .operator (⟨163308, 16⟩, ⟨163263, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩)

def event163316 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71149⟩⟩, .operator (⟨163308, 15⟩, ⟨163263, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩)

def event163317 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71149⟩⟩, .operator (⟨163308, 14⟩, ⟨163263, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩)

def event163318 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71149⟩⟩, .operator (⟨163308, 13⟩, ⟨163263, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩)

def event163319 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71149⟩⟩, .operator (⟨163308, 12⟩, ⟨163263, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩)

def event163320 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71149⟩⟩, .operator (⟨163308, 11⟩, ⟨163263, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩)

def event163321 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71149⟩⟩, .operator (⟨163308, 10⟩, ⟨163263, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩)

def event163322 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71149⟩⟩, .operator (⟨163308, 9⟩, ⟨163263, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩)

def event163323 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71149⟩⟩, .operator (⟨163308, 8⟩, ⟨163263, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩)

def event163324 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71149⟩⟩, .operator (⟨163308, 7⟩, ⟨163263, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩)

def event163325 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71149⟩⟩, .operator (⟨163308, 6⟩, ⟨163263, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩)

def event163326 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71149⟩⟩, .operator (⟨163308, 5⟩, ⟨163263, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩)

def event163327 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨71149⟩⟩, .operator (⟨163308, 4⟩, ⟨163263, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩)

def eventLeaf10192 : Array AnnotatedEvent := #[
  { event := event163072
    frameStart := 163014 },
  { event := event163073
    frameStart := 163014 },
  { event := event163074
    frameStart := 163014 },
  { event := event163075
    frameStart := 163014 },
  { event := event163076
    frameStart := 163014 },
  { event := event163077
    frameStart := 163014 },
  { event := event163078
    frameStart := 163014 },
  { event := event163079
    frameStart := 163014 },
  { event := event163080
    frameStart := 163014 },
  { event := event163081
    frameStart := 163014 },
  { event := event163082
    frameStart := 163014 },
  { event := event163083
    frameStart := 163014 },
  { event := event163084
    frameStart := 163014 },
  { event := event163085
    frameStart := 163014 },
  { event := event163086
    frameStart := 163014 },
  { event := event163087
    frameStart := 163014 }
]

def eventLeaf10193 : Array AnnotatedEvent := #[
  { event := event163088
    frameStart := 163014 },
  { event := event163089
    frameStart := 163014 },
  { event := event163090
    frameStart := 163014 },
  { event := event163091
    frameStart := 163014 },
  { event := event163092
    frameStart := 163014 },
  { event := event163093
    frameStart := 163014 },
  { event := event163094
    frameStart := 163014 },
  { event := event163095
    frameStart := 163014 },
  { event := event163096
    frameStart := 163014 },
  { event := event163097
    frameStart := 163014 },
  { event := event163098
    frameStart := 163014 },
  { event := event163099
    frameStart := 163014 },
  { event := event163100
    frameStart := 163014 },
  { event := event163101
    frameStart := 163014 },
  { event := event163102
    frameStart := 163014 },
  { event := event163103
    frameStart := 163014 }
]

def eventLeaf10194 : Array AnnotatedEvent := #[
  { event := event163104
    frameStart := 163014 },
  { event := event163105
    frameStart := 163014 },
  { event := event163106
    frameStart := 163014 },
  { event := event163107
    frameStart := 163014 },
  { event := event163108
    frameStart := 163014 },
  { event := event163109
    frameStart := 163014 },
  { event := event163110
    frameStart := 163014 },
  { event := event163111
    frameStart := 163014 },
  { event := event163112
    frameStart := 163014 },
  { event := event163113
    frameStart := 163014 },
  { event := event163114
    frameStart := 163014 },
  { event := event163115
    frameStart := 163014 },
  { event := event163116
    frameStart := 163014 },
  { event := event163117
    frameStart := 163014 },
  { event := event163118
    frameStart := 0 },
  { event := event163119
    frameStart := 0 }
]

def eventLeaf10195 : Array AnnotatedEvent := #[
  { event := event163120
    frameStart := 0 },
  { event := event163121
    frameStart := 0 },
  { event := event163122
    frameStart := 0 },
  { event := event163123
    frameStart := 0 },
  { event := event163124
    frameStart := 0 },
  { event := event163125
    frameStart := 0 },
  { event := event163126
    frameStart := 0 },
  { event := event163127
    frameStart := 0 },
  { event := event163128
    frameStart := 0 },
  { event := event163129
    frameStart := 0 },
  { event := event163130
    frameStart := 0 },
  { event := event163131
    frameStart := 0 },
  { event := event163132
    frameStart := 0 },
  { event := event163133
    frameStart := 0 },
  { event := event163134
    frameStart := 0 },
  { event := event163135
    frameStart := 0 }
]

def eventLeaf10196 : Array AnnotatedEvent := #[
  { event := event163136
    frameStart := 0 },
  { event := event163137
    frameStart := 0 },
  { event := event163138
    frameStart := 0 },
  { event := event163139
    frameStart := 0 },
  { event := event163140
    frameStart := 0 },
  { event := event163141
    frameStart := 0 },
  { event := event163142
    frameStart := 0 },
  { event := event163143
    frameStart := 0 },
  { event := event163144
    frameStart := 0 },
  { event := event163145
    frameStart := 0 },
  { event := event163146
    frameStart := 0 },
  { event := event163147
    frameStart := 0 },
  { event := event163148
    frameStart := 0 },
  { event := event163149
    frameStart := 0 },
  { event := event163150
    frameStart := 0 },
  { event := event163151
    frameStart := 0 }
]

def eventLeaf10197 : Array AnnotatedEvent := #[
  { event := event163152
    frameStart := 0 },
  { event := event163153
    frameStart := 0 },
  { event := event163154
    frameStart := 0 },
  { event := event163155
    frameStart := 0 },
  { event := event163156
    frameStart := 0 },
  { event := event163157
    frameStart := 0 },
  { event := event163158
    frameStart := 0 },
  { event := event163159
    frameStart := 0 },
  { event := event163160
    frameStart := 0 },
  { event := event163161
    frameStart := 0 },
  { event := event163162
    frameStart := 0 },
  { event := event163163
    frameStart := 0 },
  { event := event163164
    frameStart := 0 },
  { event := event163165
    frameStart := 0 },
  { event := event163166
    frameStart := 0 },
  { event := event163167
    frameStart := 0 }
]

def eventLeaf10198 : Array AnnotatedEvent := #[
  { event := event163168
    frameStart := 0 },
  { event := event163169
    frameStart := 0 },
  { event := event163170
    frameStart := 0 },
  { event := event163171
    frameStart := 0 },
  { event := event163172
    frameStart := 0 },
  { event := event163173
    frameStart := 0 },
  { event := event163174
    frameStart := 0 },
  { event := event163175
    frameStart := 0 },
  { event := event163176
    frameStart := 0 },
  { event := event163177
    frameStart := 0 },
  { event := event163178
    frameStart := 0 },
  { event := event163179
    frameStart := 0 },
  { event := event163180
    frameStart := 0 },
  { event := event163181
    frameStart := 0 },
  { event := event163182
    frameStart := 0 },
  { event := event163183
    frameStart := 0 }
]

def eventLeaf10199 : Array AnnotatedEvent := #[
  { event := event163184
    frameStart := 0 },
  { event := event163185
    frameStart := 0 },
  { event := event163186
    frameStart := 0 },
  { event := event163187
    frameStart := 0 },
  { event := event163188
    frameStart := 0 },
  { event := event163189
    frameStart := 0 },
  { event := event163190
    frameStart := 0 },
  { event := event163191
    frameStart := 0 },
  { event := event163192
    frameStart := 0 },
  { event := event163193
    frameStart := 0 },
  { event := event163194
    frameStart := 0 },
  { event := event163195
    frameStart := 0 },
  { event := event163196
    frameStart := 0 },
  { event := event163197
    frameStart := 0 },
  { event := event163198
    frameStart := 0 },
  { event := event163199
    frameStart := 0 }
]

def eventLeaf10200 : Array AnnotatedEvent := #[
  { event := event163200
    frameStart := 0 },
  { event := event163201
    frameStart := 0 },
  { event := event163202
    frameStart := 0 },
  { event := event163203
    frameStart := 0 },
  { event := event163204
    frameStart := 0 },
  { event := event163205
    frameStart := 0 },
  { event := event163206
    frameStart := 0 },
  { event := event163207
    frameStart := 0 },
  { event := event163208
    frameStart := 0 },
  { event := event163209
    frameStart := 0 },
  { event := event163210
    frameStart := 0 },
  { event := event163211
    frameStart := 0 },
  { event := event163212
    frameStart := 0 },
  { event := event163213
    frameStart := 0 },
  { event := event163214
    frameStart := 0 },
  { event := event163215
    frameStart := 0 }
]

def eventLeaf10201 : Array AnnotatedEvent := #[
  { event := event163216
    frameStart := 0 },
  { event := event163217
    frameStart := 0 },
  { event := event163218
    frameStart := 0 },
  { event := event163219
    frameStart := 0 },
  { event := event163220
    frameStart := 0 },
  { event := event163221
    frameStart := 0 },
  { event := event163222
    frameStart := 0 },
  { event := event163223
    frameStart := 0 },
  { event := event163224
    frameStart := 0 },
  { event := event163225
    frameStart := 0 },
  { event := event163226
    frameStart := 0 },
  { event := event163227
    frameStart := 0 },
  { event := event163228
    frameStart := 0 },
  { event := event163229
    frameStart := 0 },
  { event := event163230
    frameStart := 0 },
  { event := event163231
    frameStart := 0 }
]

def eventLeaf10202 : Array AnnotatedEvent := #[
  { event := event163232
    frameStart := 0 },
  { event := event163233
    frameStart := 0 },
  { event := event163234
    frameStart := 0 },
  { event := event163235
    frameStart := 0 },
  { event := event163236
    frameStart := 0 },
  { event := event163237
    frameStart := 0 },
  { event := event163238
    frameStart := 0 },
  { event := event163239
    frameStart := 0 },
  { event := event163240
    frameStart := 0 },
  { event := event163241
    frameStart := 0 },
  { event := event163242
    frameStart := 0 },
  { event := event163243
    frameStart := 0 },
  { event := event163244
    frameStart := 0 },
  { event := event163245
    frameStart := 0 },
  { event := event163246
    frameStart := 0 },
  { event := event163247
    frameStart := 0 }
]

def eventLeaf10203 : Array AnnotatedEvent := #[
  { event := event163248
    frameStart := 0 },
  { event := event163249
    frameStart := 0 },
  { event := event163250
    frameStart := 0 },
  { event := event163251
    frameStart := 0 },
  { event := event163252
    frameStart := 0 },
  { event := event163253
    frameStart := 0 },
  { event := event163254
    frameStart := 0 },
  { event := event163255
    frameStart := 0 },
  { event := event163256
    frameStart := 0 },
  { event := event163257
    frameStart := 0 },
  { event := event163258
    frameStart := 0 },
  { event := event163259
    frameStart := 0 },
  { event := event163260
    frameStart := 0 },
  { event := event163261
    frameStart := 0 },
  { event := event163262
    frameStart := 0 },
  { event := event163263
    frameStart := 0 }
]

def eventLeaf10204 : Array AnnotatedEvent := #[
  { event := event163264
    frameStart := 0 },
  { event := event163265
    frameStart := 0 },
  { event := event163266
    frameStart := 0 },
  { event := event163267
    frameStart := 0 },
  { event := event163268
    frameStart := 0 },
  { event := event163269
    frameStart := 0 },
  { event := event163270
    frameStart := 0 },
  { event := event163271
    frameStart := 0 },
  { event := event163272
    frameStart := 0 },
  { event := event163273
    frameStart := 0 },
  { event := event163274
    frameStart := 0 },
  { event := event163275
    frameStart := 0 },
  { event := event163276
    frameStart := 0 },
  { event := event163277
    frameStart := 0 },
  { event := event163278
    frameStart := 0 },
  { event := event163279
    frameStart := 0 }
]

def eventLeaf10205 : Array AnnotatedEvent := #[
  { event := event163280
    frameStart := 0 },
  { event := event163281
    frameStart := 0 },
  { event := event163282
    frameStart := 0 },
  { event := event163283
    frameStart := 0 },
  { event := event163284
    frameStart := 0 },
  { event := event163285
    frameStart := 0 },
  { event := event163286
    frameStart := 0 },
  { event := event163287
    frameStart := 0 },
  { event := event163288
    frameStart := 0 },
  { event := event163289
    frameStart := 0 },
  { event := event163290
    frameStart := 0 },
  { event := event163291
    frameStart := 0 },
  { event := event163292
    frameStart := 0 },
  { event := event163293
    frameStart := 0 },
  { event := event163294
    frameStart := 0 },
  { event := event163295
    frameStart := 0 }
]

def eventLeaf10206 : Array AnnotatedEvent := #[
  { event := event163296
    frameStart := 0 },
  { event := event163297
    frameStart := 0 },
  { event := event163298
    frameStart := 0 },
  { event := event163299
    frameStart := 0 },
  { event := event163300
    frameStart := 0 },
  { event := event163301
    frameStart := 0 },
  { event := event163302
    frameStart := 0 },
  { event := event163303
    frameStart := 0 },
  { event := event163304
    frameStart := 0 },
  { event := event163305
    frameStart := 0 },
  { event := event163306
    frameStart := 0 },
  { event := event163307
    frameStart := 0 },
  { event := event163308
    frameStart := 0 },
  { event := event163309
    frameStart := 0 },
  { event := event163310
    frameStart := 0 },
  { event := event163311
    frameStart := 0 }
]

def eventLeaf10207 : Array AnnotatedEvent := #[
  { event := event163312
    frameStart := 0 },
  { event := event163313
    frameStart := 0 },
  { event := event163314
    frameStart := 0 },
  { event := event163315
    frameStart := 0 },
  { event := event163316
    frameStart := 0 },
  { event := event163317
    frameStart := 0 },
  { event := event163318
    frameStart := 0 },
  { event := event163319
    frameStart := 0 },
  { event := event163320
    frameStart := 0 },
  { event := event163321
    frameStart := 0 },
  { event := event163322
    frameStart := 0 },
  { event := event163323
    frameStart := 0 },
  { event := event163324
    frameStart := 0 },
  { event := event163325
    frameStart := 0 },
  { event := event163326
    frameStart := 0 },
  { event := event163327
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events637
