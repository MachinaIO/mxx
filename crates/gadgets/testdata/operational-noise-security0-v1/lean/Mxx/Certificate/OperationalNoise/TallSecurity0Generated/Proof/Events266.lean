import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events266

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event68096 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16461⟩⟩) 0 ⟨12364⟩ 68095

def event68097 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16461⟩⟩) (.authority (.programFamilyFact))

def exact68098RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], []⟩, (1)⟩]

theorem exact68098RawTermsValid :
    exact68098RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68098 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16461⟩⟩) exact68098RawTerms (.finite 40) 68097 .exactZero (none)

def event68099 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16462⟩⟩) 0 ⟨16461⟩ 68098

def event68100 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16462⟩⟩) (.identity (.predecessor 0 68099 .coefficient))

def event68101 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16462⟩⟩) (.finite 40)

def event68102 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24472⟩⟩) 0 ⟨16462⟩ 68101

def event68103 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24472⟩⟩) (.authority (.programFamilyFact))

def event68104 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24472⟩⟩) (.finite 3720)

def event68105 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event68106 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24474⟩⟩) 0 ⟨6689⟩ 68105

def event68107 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24474⟩⟩) 1 ⟨24472⟩ 68104

def event68108 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24474⟩⟩) (.authority (.operator))

def exact68109RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24474⟩⟩]⟩, (1)⟩]

theorem exact68109RawTermsValid :
    exact68109RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68109 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24474⟩⟩) exact68109RawTerms .large 68108 .exactZero (none)

def event68110 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28938⟩⟩) 0 ⟨24474⟩ 68109

def event68111 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28938⟩⟩) (.authority (.operator))

def exact68112RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩, (1)⟩]

theorem exact68112RawTermsValid :
    exact68112RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68112 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28938⟩⟩) exact68112RawTerms (.finite 8192) 68111 .exactZero (none)

def event68113 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event68114 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event68115 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16501⟩⟩) 0 ⟨16462⟩ 68101

def event68116 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16501⟩⟩) 1 ⟨110⟩ 68114

def event68117 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16501⟩⟩) (.sum [.predecessor 0 68115 .coefficient, .predecessor 1 68116 .coefficient])

def event68118 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16501⟩⟩) (.finite 40)

def event68119 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16502⟩⟩) 0 ⟨16501⟩ 68118

def event68120 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16502⟩⟩) (.identity (.predecessor 0 68119 .coefficient))

def exact68121RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], []⟩, (1)⟩]

theorem exact68121RawTermsValid :
    exact68121RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68121 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16502⟩⟩) exact68121RawTerms (.finite 40) 68120 .exactZero (none)

def event68122 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact68123RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact68123RawTermsValid :
    exact68123RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68123 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact68123RawTerms .large 68122 .exactZero (none)

def event68124 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16503⟩⟩) 0 ⟨6544⟩ 68123

def event68125 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16503⟩⟩) 1 ⟨16502⟩ 68121

def event68126 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16503⟩⟩) (.product (.predecessor 0 68124 .coefficient) (.predecessor 1 68125 .coefficient) (⟨false, false, none, none, none⟩))

def event68127 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16503⟩⟩, .operator (⟨68123, 0⟩, ⟨68121, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact68128RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact68128RawTermsValid :
    exact68128RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68128 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16503⟩⟩) exact68128RawTerms .large 68126 .exactZero (none)

def event68129 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6702⟩⟩) 0 ⟨6689⟩ 68105

def event68130 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6702⟩⟩) (.authority (.operator))

def exact68131RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩]

theorem exact68131RawTermsValid :
    exact68131RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68131 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6702⟩⟩) exact68131RawTerms .large 68130 .exactZero (none)

def event68132 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16504⟩⟩) 0 ⟨6702⟩ 68131

def event68133 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16504⟩⟩) 1 ⟨16503⟩ 68128

def event68134 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16504⟩⟩) (.sum [.predecessor 0 68132 .coefficient, .predecessor 1 68133 .coefficient])

def exact68135RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68135RawTermsValid :
    exact68135RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68135 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16504⟩⟩) exact68135RawTerms .large 68134 .exactZero (none)

def event68136 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28939⟩⟩) 0 ⟨16504⟩ 68135

def event68137 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28939⟩⟩) 1 ⟨28938⟩ 68112

def event68138 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28939⟩⟩) (.product (.predecessor 0 68136 .coefficient) (.predecessor 1 68137 .coefficient) (⟨false, false, none, none, none⟩))

def event68139 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28939⟩⟩, .operator (⟨68135, 0⟩, ⟨68112, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩, (1)⟩)

def event68140 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28939⟩⟩, .operator (⟨68135, 1⟩, ⟨68112, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩, (-1)⟩)

def event68141 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28939⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28938⟩⟩) ⟨24474⟩ 68109)

def event68142 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28939⟩⟩, .relation 68141 0, ⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24474⟩⟩]⟩, (-1)⟩)

def exact68143RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24474⟩⟩]⟩, (-1)⟩]

theorem exact68143RawTermsValid :
    exact68143RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68143 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28939⟩⟩) exact68143RawTerms .large 68138 .exactZero (none)

def event68144 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17901⟩⟩) 0 ⟨16462⟩ 68101

def event68145 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17901⟩⟩) (.authority (.programFamilyFact))

def exact68146RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17901⟩⟩], []⟩, (1)⟩]

theorem exact68146RawTermsValid :
    exact68146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68146 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17901⟩⟩) exact68146RawTerms (.finite 62) 68145 .exactZero (none)

def event68147 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17902⟩⟩) 0 ⟨6544⟩ 68123

def event68148 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17902⟩⟩) 1 ⟨17901⟩ 68146

def event68149 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17902⟩⟩) (.product (.predecessor 0 68147 .coefficient) (.predecessor 1 68148 .coefficient) (⟨false, true, none, none, some 1⟩))

def event68150 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17902⟩⟩, .operator (⟨68123, 0⟩, ⟨68146, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17901⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact68151RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17901⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact68151RawTermsValid :
    exact68151RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68151 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17902⟩⟩) exact68151RawTerms .large 68149 .exactZero (none)

def event68152 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6733⟩⟩) 0 ⟨6689⟩ 68105

def event68153 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6733⟩⟩) (.authority (.operator))

def exact68154RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩]

theorem exact68154RawTermsValid :
    exact68154RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68154 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6733⟩⟩) exact68154RawTerms .large 68153 .exactZero (none)

def event68155 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17903⟩⟩) 0 ⟨6733⟩ 68154

def event68156 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17903⟩⟩) 1 ⟨17902⟩ 68151

def event68157 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17903⟩⟩) (.sum [.predecessor 0 68155 .coefficient, .predecessor 1 68156 .coefficient])

def exact68158RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17901⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68158RawTermsValid :
    exact68158RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68158 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17903⟩⟩) exact68158RawTerms .large 68157 .exactZero (none)

def event68159 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28943⟩⟩) 0 ⟨17903⟩ 68158

def event68160 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28943⟩⟩) 1 ⟨28939⟩ 68143

def event68161 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28943⟩⟩) (.sum [.predecessor 0 68159 .coefficient, .predecessor 1 68160 .coefficient])

def exact68162RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24474⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17901⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68162RawTermsValid :
    exact68162RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68162 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28943⟩⟩) exact68162RawTerms .large 68161 .exactZero (none)

def event68163 : Event := .preFoldPolynomial 68162 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24474⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17901⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact68164RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24474⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17901⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event68164 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28943⟩⟩) 68163 exact68164RawTerms .large 68161 .exactZero (none)

def event68165 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16462⟩⟩) ⟨⟨146⟩, ⟨54⟩, ⟨109⟩⟩ ⟨68007, 68165⟩

def event68166 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨22119⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22116⟩⟩]⟩) (1) 0 2 (.universal 68165 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22116⟩⟩]⟩) (none) 68164)

def event68167 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22119⟩⟩, .relation 68166 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩)

def event68168 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22119⟩⟩, .relation 68166 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩, (-1)⟩)

def event68169 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22119⟩⟩, .relation 68166 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24474⟩⟩]⟩, (1)⟩)

def event68170 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22119⟩⟩, .relation 68166 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17901⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact68171RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24474⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17901⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68171RawTermsValid :
    exact68171RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68171 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22119⟩⟩) exact68171RawTerms .large 68003 (.finite 1811303510016) (some (68005))

def event68172 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28941⟩⟩) 0 ⟨22119⟩ 68171

def event68173 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28941⟩⟩) 1 ⟨28940⟩ 67993

def event68174 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28941⟩⟩) (.sum [.predecessor 0 68172 .coefficient, .predecessor 1 68173 .coefficient])

def event68175 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28941⟩⟩, .operator (⟨68171, 0⟩, ⟨67993, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩, (1)⟩)

def event68176 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28941⟩⟩, .operator (⟨68171, 2⟩, ⟨67993, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24474⟩⟩]⟩, (-1)⟩)

def event68177 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28941⟩⟩) (.sum [.result 68171 .summary, .result 67993 .summary])

def exact68178RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17901⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68178RawTermsValid :
    exact68178RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68178 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28941⟩⟩) exact68178RawTerms .large 68174 (.finite 1292315010834812776448) (some (68177))

def event68179 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24409⟩⟩) 0 ⟨16378⟩ 3241

def event68180 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24409⟩⟩) (.authority (.programFamilyFact))

def event68181 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24409⟩⟩) (.finite 3720)

def event68182 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24411⟩⟩) 0 ⟨6689⟩ 5477

def event68183 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24411⟩⟩) 1 ⟨24409⟩ 68181

def event68184 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24411⟩⟩) (.authority (.operator))

def exact68185RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24411⟩⟩]⟩, (1)⟩]

theorem exact68185RawTermsValid :
    exact68185RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68185 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24411⟩⟩) exact68185RawTerms .large 68184 .exactZero (none)

def event68186 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28721⟩⟩) 0 ⟨24411⟩ 68185

def event68187 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28721⟩⟩) (.authority (.operator))

def exact68188RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28721⟩⟩]⟩, (1)⟩]

theorem exact68188RawTermsValid :
    exact68188RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68188 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28721⟩⟩) exact68188RawTerms (.finite 8192) 68187 .exactZero (none)

def event68189 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23119⟩⟩) 0 ⟨11951⟩ 3235

def event68190 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23119⟩⟩) (.authority (.programFamilyFact))

def event68191 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23119⟩⟩) (.finite 3720)

def event68192 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23120⟩⟩) 0 ⟨6689⟩ 5477

def event68193 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23120⟩⟩) 1 ⟨23119⟩ 68191

def event68194 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23120⟩⟩) (.authority (.operator))

def exact68195RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23120⟩⟩]⟩, (1)⟩]

theorem exact68195RawTermsValid :
    exact68195RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68195 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23120⟩⟩) exact68195RawTerms .large 68194 .exactZero (none)

def event68196 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25214⟩⟩) 0 ⟨23120⟩ 68195

def event68197 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25214⟩⟩) (.authority (.operator))

def exact68198RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩, (1)⟩]

theorem exact68198RawTermsValid :
    exact68198RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68198 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25214⟩⟩) exact68198RawTerms (.finite 8192) 68197 .exactZero (none)

def event68199 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11952⟩⟩) 0 ⟨11949⟩ 3224

def event68200 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11952⟩⟩) 1 ⟨6566⟩ 65295

def event68201 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11952⟩⟩) (.tensor (.predecessor 0 68199 .coefficient) (.predecessor 1 68200 .coefficient) true false)

def event68202 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11952⟩⟩, .operator (⟨3224, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact68203RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact68203RawTermsValid :
    exact68203RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68203 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11952⟩⟩) exact68203RawTerms .large 68201 .exactZero (none)

def event68204 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7202⟩⟩) 0 ⟨5533⟩ 65165

def event68205 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7202⟩⟩) 1 ⟨6784⟩ 9478

def event68206 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7202⟩⟩) (.product (.predecessor 0 68204 .coefficient) (.predecessor 1 68205 .coefficient) (⟨false, false, none, none, none⟩))

def event68207 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7202⟩⟩, .operator (⟨65165, 0⟩, ⟨9478, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩)

def exact68208RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩]

theorem exact68208RawTermsValid :
    exact68208RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68208 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7202⟩⟩) exact68208RawTerms .large 68206 .exactZero (none)

def event68209 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11953⟩⟩) 0 ⟨7202⟩ 68208

def event68210 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11953⟩⟩) 1 ⟨11952⟩ 68203

def event68211 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11953⟩⟩) (.sum [.predecessor 0 68209 .coefficient, .predecessor 1 68210 .coefficient])

def exact68212RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68212RawTermsValid :
    exact68212RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68212 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11953⟩⟩) exact68212RawTerms .large 68211 .exactZero (none)

def event68213 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11954⟩⟩) 0 ⟨11953⟩ 68212

def event68214 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11954⟩⟩) 1 ⟨98⟩ 9470

def event68215 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11954⟩⟩) (.sum [.predecessor 0 68213 .coefficient, .predecessor 1 68214 .coefficient])

def event68216 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11954⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨98⟩⟩]⟩) [⟨.result 9470 .coefficient, false, none⟩])

def event68217 : Event := .survivorFold (1) 68216

def exact68218RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68218RawTermsValid :
    exact68218RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68218 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11954⟩⟩) exact68218RawTerms .large 68215 (.finite 26) (some (68216))

def event68219 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11955⟩⟩) 0 ⟨11954⟩ 68218

def event68220 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11955⟩⟩) 1 ⟨9710⟩ 3227

def event68221 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11955⟩⟩) (.product (.predecessor 0 68219 .coefficient) (.predecessor 1 68220 .coefficient) (⟨false, true, none, none, some 1⟩))

def event68222 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11955⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩], []⟩) [⟨.result 3227 .coefficient, true, some 1⟩])

def event68223 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11955⟩⟩) (.product (.result 68218 .summary) (.transfer 68222) (⟨false, false, none, none, none⟩))

def event68224 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11955⟩⟩, .operator (⟨68218, 1⟩, ⟨3227, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event68225 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11955⟩⟩, .operator (⟨68218, 0⟩, ⟨3227, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩)

def exact68226RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68226RawTermsValid :
    exact68226RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68226 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11955⟩⟩) exact68226RawTerms .large 68221 (.finite 29952) (some (68223))

def event68227 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9711⟩⟩) 0 ⟨9710⟩ 3227

def event68228 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9711⟩⟩) 1 ⟨6566⟩ 65295

def event68229 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9711⟩⟩) (.tensor (.predecessor 0 68227 .coefficient) (.predecessor 1 68228 .coefficient) true false)

def event68230 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9711⟩⟩, .operator (⟨3227, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact68231RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact68231RawTermsValid :
    exact68231RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68231 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9711⟩⟩) exact68231RawTerms .large 68229 .exactZero (none)

def event68232 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7182⟩⟩) 0 ⟨5533⟩ 65165

def event68233 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7182⟩⟩) 1 ⟨6764⟩ 9519

def event68234 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7182⟩⟩) (.product (.predecessor 0 68232 .coefficient) (.predecessor 1 68233 .coefficient) (⟨false, false, none, none, none⟩))

def event68235 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7182⟩⟩, .operator (⟨65165, 0⟩, ⟨9519, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩]⟩, (1)⟩)

def exact68236RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩]⟩, (1)⟩]

theorem exact68236RawTermsValid :
    exact68236RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68236 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7182⟩⟩) exact68236RawTerms .large 68234 .exactZero (none)

def event68237 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9712⟩⟩) 0 ⟨7182⟩ 68236

def event68238 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9712⟩⟩) 1 ⟨9711⟩ 68231

def event68239 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9712⟩⟩) (.sum [.predecessor 0 68237 .coefficient, .predecessor 1 68238 .coefficient])

def exact68240RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68240RawTermsValid :
    exact68240RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68240 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9712⟩⟩) exact68240RawTerms .large 68239 .exactZero (none)

def event68241 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9713⟩⟩) 0 ⟨9712⟩ 68240

def event68242 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9713⟩⟩) 1 ⟨78⟩ 9511

def event68243 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9713⟩⟩) (.sum [.predecessor 0 68241 .coefficient, .predecessor 1 68242 .coefficient])

def event68244 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9713⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨78⟩⟩]⟩) [⟨.result 9511 .coefficient, false, none⟩])

def event68245 : Event := .survivorFold (1) 68244

def exact68246RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68246RawTermsValid :
    exact68246RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68246 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9713⟩⟩) exact68246RawTerms .large 68243 (.finite 26) (some (68244))

def event68247 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9714⟩⟩) 0 ⟨9713⟩ 68246

def event68248 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9714⟩⟩) 1 ⟨7865⟩ 9508

def event68249 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9714⟩⟩) (.product (.predecessor 0 68247 .coefficient) (.predecessor 1 68248 .coefficient) (⟨false, false, none, none, none⟩))

def event68250 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9714⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩) [⟨.result 9504 .coefficient, false, none⟩])

def event68251 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9714⟩⟩) (.product (.result 68246 .summary) (.transfer 68250) (⟨false, false, none, none, none⟩))

def event68252 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9714⟩⟩, .operator (⟨68246, 1⟩, ⟨9508, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩, (-1)⟩)

def event68253 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨9714⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7864⟩⟩) ⟨6784⟩ 9478)

def event68254 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9714⟩⟩, .relation 68253 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (-1)⟩)

def event68255 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9714⟩⟩, .operator (⟨68246, 0⟩, ⟨9508, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩, (1)⟩)

def exact68256RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (-1)⟩]

theorem exact68256RawTermsValid :
    exact68256RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68256 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9714⟩⟩) exact68256RawTerms .large 68249 (.finite 95420416) (some (68251))

def event68257 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11956⟩⟩) 0 ⟨9714⟩ 68256

def event68258 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11956⟩⟩) 1 ⟨11955⟩ 68226

def event68259 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11956⟩⟩) (.sum [.predecessor 0 68257 .coefficient, .predecessor 1 68258 .coefficient])

def event68260 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11956⟩⟩, .operator (⟨68256, 1⟩, ⟨68226, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩)

def event68261 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11956⟩⟩) (.sum [.result 68256 .summary, .result 68226 .summary])

def exact68262RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68262RawTermsValid :
    exact68262RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68262 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11956⟩⟩) exact68262RawTerms .large 68259 (.finite 95450368) (some (68261))

def event68263 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25215⟩⟩) 0 ⟨11956⟩ 68262

def event68264 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25215⟩⟩) 1 ⟨25214⟩ 68198

def event68265 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25215⟩⟩) (.product (.predecessor 0 68263 .coefficient) (.predecessor 1 68264 .coefficient) (⟨false, false, none, none, none⟩))

def event68266 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25215⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩) [⟨.result 68198 .coefficient, false, none⟩])

def event68267 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25215⟩⟩) (.product (.result 68262 .summary) (.transfer 68266) (⟨false, false, none, none, none⟩))

def event68268 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25215⟩⟩, .operator (⟨68262, 1⟩, ⟨68198, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩, (-1)⟩)

def event68269 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25215⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25214⟩⟩) ⟨23120⟩ 68195)

def event68270 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25215⟩⟩, .relation 68269 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨23120⟩⟩]⟩, (-1)⟩)

def event68271 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25215⟩⟩, .operator (⟨68262, 0⟩, ⟨68198, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩, (1)⟩)

def exact68272RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨23120⟩⟩]⟩, (-1)⟩]

theorem exact68272RawTermsValid :
    exact68272RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68272 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25215⟩⟩) exact68272RawTerms .large 68265 (.finite 350304377765888) (some (68267))

def event68273 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19812⟩⟩) 0 ⟨11951⟩ 3235

def event68274 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19812⟩⟩) (.authority (.relationPreimageSource ⟨19⟩))

def exact68275RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩, (1)⟩]

theorem exact68275RawTermsValid :
    exact68275RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68275 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19812⟩⟩) exact68275RawTerms (.finite 136065468) 68274 .exactZero (none)

def event68276 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19814⟩⟩) 0 ⟨19812⟩ 68275

def event68277 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19814⟩⟩) 1 ⟨2348⟩ 4

def event68278 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19814⟩⟩) (.scale (.predecessor 0 68276 .coefficient) (.value (.predecessor 1 68277 .coefficient)))

def exact68279RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩, (1)⟩]

theorem exact68279RawTermsValid :
    exact68279RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68279 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19814⟩⟩) exact68279RawTerms (.finite 136065468) 68278 .exactZero (none)

def event68280 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19815⟩⟩) 0 ⟨5535⟩ 65387

def event68281 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19815⟩⟩) 1 ⟨19814⟩ 68279

def event68282 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19815⟩⟩) (.product (.predecessor 0 68280 .coefficient) (.predecessor 1 68281 .coefficient) (⟨false, false, none, none, none⟩))

def event68283 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19815⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩) [⟨.result 68275 .coefficient, false, none⟩])

def event68284 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19815⟩⟩) (.product (.result 65387 .summary) (.transfer 68283) (⟨false, false, none, none, none⟩))

def event68285 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19815⟩⟩, .operator (⟨65387, 0⟩, ⟨68279, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩, (1)⟩)

def event68286 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19813⟩⟩)

def event68287 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event68288 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event68289 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event68290 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event68291 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event68292 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event68293 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event68294 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event68295 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 68294

def event68296 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 68292

def event68297 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 68295 .coefficient) (.value (.predecessor 1 68296 .coefficient)))

def event68298 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event68299 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 68298

def event68300 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 68290

def event68301 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 68299 .coefficient, .predecessor 1 68300 .coefficient])

def event68302 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event68303 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 68302

def event68304 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 68288

def event68305 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 68304 .coefficient))

def event68306 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event68307 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11949⟩⟩) 0 ⟨5530⟩ 68306

def event68308 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11949⟩⟩) (.authority (.programFamilyFact))

def exact68309RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11949⟩⟩], []⟩, (1)⟩]

theorem exact68309RawTermsValid :
    exact68309RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68309 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11949⟩⟩) exact68309RawTerms (.finite 36) 68308 .exactZero (none)

def event68310 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9710⟩⟩) 0 ⟨5530⟩ 68306

def event68311 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9710⟩⟩) (.authority (.programFamilyFact))

def exact68312RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩], []⟩, (1)⟩]

theorem exact68312RawTermsValid :
    exact68312RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68312 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9710⟩⟩) exact68312RawTerms (.finite 36) 68311 .exactZero (none)

def event68313 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11950⟩⟩) 0 ⟨9710⟩ 68312

def event68314 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11950⟩⟩) 1 ⟨11949⟩ 68309

def event68315 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11950⟩⟩) (.product (.predecessor 0 68313 .coefficient) (.predecessor 1 68314 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event68316 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11950⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], []⟩) [⟨.result 68312 .coefficient, true, some 1⟩, ⟨.result 68309 .coefficient, true, some 1⟩])

def event68317 : Event := .survivorFold (1) 68316

def exact68318RawTerms : List Term := []

theorem exact68318RawTermsValid :
    exact68318RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68318 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11950⟩⟩) exact68318RawTerms (.finite 1296) 68315 (.finite 1296) (some (68316))

def event68319 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11951⟩⟩) 0 ⟨11950⟩ 68318

def event68320 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11951⟩⟩) (.identity (.predecessor 0 68319 .coefficient))

def event68321 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11951⟩⟩) (.finite 1296)

def event68322 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19812⟩⟩) 0 ⟨11951⟩ 68321

def event68323 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19812⟩⟩) (.authority (.relationPreimageSource ⟨19⟩))

def exact68324RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩, (1)⟩]

theorem exact68324RawTermsValid :
    exact68324RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68324 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19812⟩⟩) exact68324RawTerms (.finite 136065468) 68323 .exactZero (none)

def event68325 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact68326RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact68326RawTermsValid :
    exact68326RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68326 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact68326RawTerms .large 68325 .exactZero (none)

def event68327 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19813⟩⟩) 0 ⟨6⟩ 68326

def event68328 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19813⟩⟩) 1 ⟨19812⟩ 68324

def event68329 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19813⟩⟩) (.product (.predecessor 0 68327 .coefficient) (.predecessor 1 68328 .coefficient) (⟨false, false, none, none, none⟩))

def event68330 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19813⟩⟩, .operator (⟨68326, 0⟩, ⟨68324, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩, (1)⟩)

def exact68331RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩, (1)⟩]

theorem exact68331RawTermsValid :
    exact68331RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68331 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19813⟩⟩) exact68331RawTerms .large 68329 .exactZero (none)

def event68332 : Event := .preFoldPolynomial 68331 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩, (1)⟩] .exactZero none

def exact68333RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩, (1)⟩]

def event68333 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19813⟩⟩) 68332 exact68333RawTerms .large 68329 .exactZero (none)

def event68334 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25218⟩⟩)

def event68335 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event68336 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event68337 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event68338 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event68339 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event68340 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event68341 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event68342 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event68343 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 68342

def event68344 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 68340

def event68345 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 68343 .coefficient) (.value (.predecessor 1 68344 .coefficient)))

def event68346 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event68347 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 68346

def event68348 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 68338

def event68349 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 68347 .coefficient, .predecessor 1 68348 .coefficient])

def event68350 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event68351 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 68350

def eventLeaf4256 : Array AnnotatedEvent := #[
  { event := event68096
    frameStart := 68061 },
  { event := event68097
    frameStart := 68061 },
  { event := event68098
    frameStart := 68061 },
  { event := event68099
    frameStart := 68061 },
  { event := event68100
    frameStart := 68061 },
  { event := event68101
    frameStart := 68061 },
  { event := event68102
    frameStart := 68061 },
  { event := event68103
    frameStart := 68061 },
  { event := event68104
    frameStart := 68061 },
  { event := event68105
    frameStart := 68061 },
  { event := event68106
    frameStart := 68061 },
  { event := event68107
    frameStart := 68061 },
  { event := event68108
    frameStart := 68061 },
  { event := event68109
    frameStart := 68061 },
  { event := event68110
    frameStart := 68061 },
  { event := event68111
    frameStart := 68061 }
]

def eventLeaf4257 : Array AnnotatedEvent := #[
  { event := event68112
    frameStart := 68061 },
  { event := event68113
    frameStart := 68061 },
  { event := event68114
    frameStart := 68061 },
  { event := event68115
    frameStart := 68061 },
  { event := event68116
    frameStart := 68061 },
  { event := event68117
    frameStart := 68061 },
  { event := event68118
    frameStart := 68061 },
  { event := event68119
    frameStart := 68061 },
  { event := event68120
    frameStart := 68061 },
  { event := event68121
    frameStart := 68061 },
  { event := event68122
    frameStart := 68061 },
  { event := event68123
    frameStart := 68061 },
  { event := event68124
    frameStart := 68061 },
  { event := event68125
    frameStart := 68061 },
  { event := event68126
    frameStart := 68061 },
  { event := event68127
    frameStart := 68061 }
]

def eventLeaf4258 : Array AnnotatedEvent := #[
  { event := event68128
    frameStart := 68061 },
  { event := event68129
    frameStart := 68061 },
  { event := event68130
    frameStart := 68061 },
  { event := event68131
    frameStart := 68061 },
  { event := event68132
    frameStart := 68061 },
  { event := event68133
    frameStart := 68061 },
  { event := event68134
    frameStart := 68061 },
  { event := event68135
    frameStart := 68061 },
  { event := event68136
    frameStart := 68061 },
  { event := event68137
    frameStart := 68061 },
  { event := event68138
    frameStart := 68061 },
  { event := event68139
    frameStart := 68061 },
  { event := event68140
    frameStart := 68061 },
  { event := event68141
    frameStart := 68061 },
  { event := event68142
    frameStart := 68061 },
  { event := event68143
    frameStart := 68061 }
]

def eventLeaf4259 : Array AnnotatedEvent := #[
  { event := event68144
    frameStart := 68061 },
  { event := event68145
    frameStart := 68061 },
  { event := event68146
    frameStart := 68061 },
  { event := event68147
    frameStart := 68061 },
  { event := event68148
    frameStart := 68061 },
  { event := event68149
    frameStart := 68061 },
  { event := event68150
    frameStart := 68061 },
  { event := event68151
    frameStart := 68061 },
  { event := event68152
    frameStart := 68061 },
  { event := event68153
    frameStart := 68061 },
  { event := event68154
    frameStart := 68061 },
  { event := event68155
    frameStart := 68061 },
  { event := event68156
    frameStart := 68061 },
  { event := event68157
    frameStart := 68061 },
  { event := event68158
    frameStart := 68061 },
  { event := event68159
    frameStart := 68061 }
]

def eventLeaf4260 : Array AnnotatedEvent := #[
  { event := event68160
    frameStart := 68061 },
  { event := event68161
    frameStart := 68061 },
  { event := event68162
    frameStart := 68061 },
  { event := event68163
    frameStart := 68061 },
  { event := event68164
    frameStart := 68061 },
  { event := event68165
    frameStart := 0 },
  { event := event68166
    frameStart := 0 },
  { event := event68167
    frameStart := 0 },
  { event := event68168
    frameStart := 0 },
  { event := event68169
    frameStart := 0 },
  { event := event68170
    frameStart := 0 },
  { event := event68171
    frameStart := 0 },
  { event := event68172
    frameStart := 0 },
  { event := event68173
    frameStart := 0 },
  { event := event68174
    frameStart := 0 },
  { event := event68175
    frameStart := 0 }
]

def eventLeaf4261 : Array AnnotatedEvent := #[
  { event := event68176
    frameStart := 0 },
  { event := event68177
    frameStart := 0 },
  { event := event68178
    frameStart := 0 },
  { event := event68179
    frameStart := 0 },
  { event := event68180
    frameStart := 0 },
  { event := event68181
    frameStart := 0 },
  { event := event68182
    frameStart := 0 },
  { event := event68183
    frameStart := 0 },
  { event := event68184
    frameStart := 0 },
  { event := event68185
    frameStart := 0 },
  { event := event68186
    frameStart := 0 },
  { event := event68187
    frameStart := 0 },
  { event := event68188
    frameStart := 0 },
  { event := event68189
    frameStart := 0 },
  { event := event68190
    frameStart := 0 },
  { event := event68191
    frameStart := 0 }
]

def eventLeaf4262 : Array AnnotatedEvent := #[
  { event := event68192
    frameStart := 0 },
  { event := event68193
    frameStart := 0 },
  { event := event68194
    frameStart := 0 },
  { event := event68195
    frameStart := 0 },
  { event := event68196
    frameStart := 0 },
  { event := event68197
    frameStart := 0 },
  { event := event68198
    frameStart := 0 },
  { event := event68199
    frameStart := 0 },
  { event := event68200
    frameStart := 0 },
  { event := event68201
    frameStart := 0 },
  { event := event68202
    frameStart := 0 },
  { event := event68203
    frameStart := 0 },
  { event := event68204
    frameStart := 0 },
  { event := event68205
    frameStart := 0 },
  { event := event68206
    frameStart := 0 },
  { event := event68207
    frameStart := 0 }
]

def eventLeaf4263 : Array AnnotatedEvent := #[
  { event := event68208
    frameStart := 0 },
  { event := event68209
    frameStart := 0 },
  { event := event68210
    frameStart := 0 },
  { event := event68211
    frameStart := 0 },
  { event := event68212
    frameStart := 0 },
  { event := event68213
    frameStart := 0 },
  { event := event68214
    frameStart := 0 },
  { event := event68215
    frameStart := 0 },
  { event := event68216
    frameStart := 0 },
  { event := event68217
    frameStart := 0 },
  { event := event68218
    frameStart := 0 },
  { event := event68219
    frameStart := 0 },
  { event := event68220
    frameStart := 0 },
  { event := event68221
    frameStart := 0 },
  { event := event68222
    frameStart := 0 },
  { event := event68223
    frameStart := 0 }
]

def eventLeaf4264 : Array AnnotatedEvent := #[
  { event := event68224
    frameStart := 0 },
  { event := event68225
    frameStart := 0 },
  { event := event68226
    frameStart := 0 },
  { event := event68227
    frameStart := 0 },
  { event := event68228
    frameStart := 0 },
  { event := event68229
    frameStart := 0 },
  { event := event68230
    frameStart := 0 },
  { event := event68231
    frameStart := 0 },
  { event := event68232
    frameStart := 0 },
  { event := event68233
    frameStart := 0 },
  { event := event68234
    frameStart := 0 },
  { event := event68235
    frameStart := 0 },
  { event := event68236
    frameStart := 0 },
  { event := event68237
    frameStart := 0 },
  { event := event68238
    frameStart := 0 },
  { event := event68239
    frameStart := 0 }
]

def eventLeaf4265 : Array AnnotatedEvent := #[
  { event := event68240
    frameStart := 0 },
  { event := event68241
    frameStart := 0 },
  { event := event68242
    frameStart := 0 },
  { event := event68243
    frameStart := 0 },
  { event := event68244
    frameStart := 0 },
  { event := event68245
    frameStart := 0 },
  { event := event68246
    frameStart := 0 },
  { event := event68247
    frameStart := 0 },
  { event := event68248
    frameStart := 0 },
  { event := event68249
    frameStart := 0 },
  { event := event68250
    frameStart := 0 },
  { event := event68251
    frameStart := 0 },
  { event := event68252
    frameStart := 0 },
  { event := event68253
    frameStart := 0 },
  { event := event68254
    frameStart := 0 },
  { event := event68255
    frameStart := 0 }
]

def eventLeaf4266 : Array AnnotatedEvent := #[
  { event := event68256
    frameStart := 0 },
  { event := event68257
    frameStart := 0 },
  { event := event68258
    frameStart := 0 },
  { event := event68259
    frameStart := 0 },
  { event := event68260
    frameStart := 0 },
  { event := event68261
    frameStart := 0 },
  { event := event68262
    frameStart := 0 },
  { event := event68263
    frameStart := 0 },
  { event := event68264
    frameStart := 0 },
  { event := event68265
    frameStart := 0 },
  { event := event68266
    frameStart := 0 },
  { event := event68267
    frameStart := 0 },
  { event := event68268
    frameStart := 0 },
  { event := event68269
    frameStart := 0 },
  { event := event68270
    frameStart := 0 },
  { event := event68271
    frameStart := 0 }
]

def eventLeaf4267 : Array AnnotatedEvent := #[
  { event := event68272
    frameStart := 0 },
  { event := event68273
    frameStart := 0 },
  { event := event68274
    frameStart := 0 },
  { event := event68275
    frameStart := 0 },
  { event := event68276
    frameStart := 0 },
  { event := event68277
    frameStart := 0 },
  { event := event68278
    frameStart := 0 },
  { event := event68279
    frameStart := 0 },
  { event := event68280
    frameStart := 0 },
  { event := event68281
    frameStart := 0 },
  { event := event68282
    frameStart := 0 },
  { event := event68283
    frameStart := 0 },
  { event := event68284
    frameStart := 0 },
  { event := event68285
    frameStart := 0 },
  { event := event68286
    frameStart := 68286 },
  { event := event68287
    frameStart := 68286 }
]

def eventLeaf4268 : Array AnnotatedEvent := #[
  { event := event68288
    frameStart := 68286 },
  { event := event68289
    frameStart := 68286 },
  { event := event68290
    frameStart := 68286 },
  { event := event68291
    frameStart := 68286 },
  { event := event68292
    frameStart := 68286 },
  { event := event68293
    frameStart := 68286 },
  { event := event68294
    frameStart := 68286 },
  { event := event68295
    frameStart := 68286 },
  { event := event68296
    frameStart := 68286 },
  { event := event68297
    frameStart := 68286 },
  { event := event68298
    frameStart := 68286 },
  { event := event68299
    frameStart := 68286 },
  { event := event68300
    frameStart := 68286 },
  { event := event68301
    frameStart := 68286 },
  { event := event68302
    frameStart := 68286 },
  { event := event68303
    frameStart := 68286 }
]

def eventLeaf4269 : Array AnnotatedEvent := #[
  { event := event68304
    frameStart := 68286 },
  { event := event68305
    frameStart := 68286 },
  { event := event68306
    frameStart := 68286 },
  { event := event68307
    frameStart := 68286 },
  { event := event68308
    frameStart := 68286 },
  { event := event68309
    frameStart := 68286 },
  { event := event68310
    frameStart := 68286 },
  { event := event68311
    frameStart := 68286 },
  { event := event68312
    frameStart := 68286 },
  { event := event68313
    frameStart := 68286 },
  { event := event68314
    frameStart := 68286 },
  { event := event68315
    frameStart := 68286 },
  { event := event68316
    frameStart := 68286 },
  { event := event68317
    frameStart := 68286 },
  { event := event68318
    frameStart := 68286 },
  { event := event68319
    frameStart := 68286 }
]

def eventLeaf4270 : Array AnnotatedEvent := #[
  { event := event68320
    frameStart := 68286 },
  { event := event68321
    frameStart := 68286 },
  { event := event68322
    frameStart := 68286 },
  { event := event68323
    frameStart := 68286 },
  { event := event68324
    frameStart := 68286 },
  { event := event68325
    frameStart := 68286 },
  { event := event68326
    frameStart := 68286 },
  { event := event68327
    frameStart := 68286 },
  { event := event68328
    frameStart := 68286 },
  { event := event68329
    frameStart := 68286 },
  { event := event68330
    frameStart := 68286 },
  { event := event68331
    frameStart := 68286 },
  { event := event68332
    frameStart := 68286 },
  { event := event68333
    frameStart := 68286 },
  { event := event68334
    frameStart := 68334 },
  { event := event68335
    frameStart := 68334 }
]

def eventLeaf4271 : Array AnnotatedEvent := #[
  { event := event68336
    frameStart := 68334 },
  { event := event68337
    frameStart := 68334 },
  { event := event68338
    frameStart := 68334 },
  { event := event68339
    frameStart := 68334 },
  { event := event68340
    frameStart := 68334 },
  { event := event68341
    frameStart := 68334 },
  { event := event68342
    frameStart := 68334 },
  { event := event68343
    frameStart := 68334 },
  { event := event68344
    frameStart := 68334 },
  { event := event68345
    frameStart := 68334 },
  { event := event68346
    frameStart := 68334 },
  { event := event68347
    frameStart := 68334 },
  { event := event68348
    frameStart := 68334 },
  { event := event68349
    frameStart := 68334 },
  { event := event68350
    frameStart := 68334 },
  { event := event68351
    frameStart := 68334 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events266
