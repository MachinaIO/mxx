import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events270

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event69120 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16307⟩⟩) 1 ⟨16306⟩ 69115

def event69121 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16307⟩⟩) (.sum [.predecessor 0 69119 .coefficient, .predecessor 1 69120 .coefficient])

def exact69122RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16305⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69122RawTermsValid :
    exact69122RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69122 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16307⟩⟩) exact69122RawTerms .large 69121 .exactZero (none)

def event69123 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28509⟩⟩) 0 ⟨16307⟩ 69122

def event69124 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28509⟩⟩) 1 ⟨28505⟩ 69107

def event69125 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28509⟩⟩) (.sum [.predecessor 0 69123 .coefficient, .predecessor 1 69124 .coefficient])

def exact69126RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨24348⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16305⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69126RawTermsValid :
    exact69126RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69126 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28509⟩⟩) exact69126RawTerms .large 69125 .exactZero (none)

def event69127 : Event := .preFoldPolynomial 69126 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨24348⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16305⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact69128RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨24348⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16305⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event69128 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28509⟩⟩) 69127 exact69128RawTerms .large 69125 .exactZero (none)

def event69129 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16259⟩⟩) ⟨⟨142⟩, ⟨50⟩, ⟨109⟩⟩ ⟨68971, 69129⟩

def event69130 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21831⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21828⟩⟩]⟩) (1) 0 2 (.universal 69129 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21828⟩⟩]⟩) (none) 69128)

def event69131 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21831⟩⟩, .relation 69130 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩)

def event69132 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21831⟩⟩, .relation 69130 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩]⟩, (-1)⟩)

def event69133 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21831⟩⟩, .relation 69130 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨24348⟩⟩]⟩, (1)⟩)

def event69134 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21831⟩⟩, .relation 69130 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16305⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact69135RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨24348⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16305⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69135RawTermsValid :
    exact69135RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69135 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21831⟩⟩) exact69135RawTerms .large 68967 (.finite 1811303510016) (some (68969))

def event69136 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28507⟩⟩) 0 ⟨21831⟩ 69135

def event69137 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28507⟩⟩) 1 ⟨28506⟩ 68957

def event69138 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28507⟩⟩) (.sum [.predecessor 0 69136 .coefficient, .predecessor 1 69137 .coefficient])

def event69139 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28507⟩⟩, .operator (⟨69135, 0⟩, ⟨68957, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩]⟩, (1)⟩)

def event69140 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28507⟩⟩, .operator (⟨69135, 2⟩, ⟨68957, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16258⟩⟩], [⟨.program ⟨214⟩, ⟨24348⟩⟩]⟩, (-1)⟩)

def event69141 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28507⟩⟩) (.sum [.result 69135 .summary, .result 68957 .summary])

def exact69142RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16305⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69142RawTermsValid :
    exact69142RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69142 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28507⟩⟩) exact69142RawTerms .large 69138 (.finite 1292202948609709846528) (some (69141))

def event69143 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24283⟩⟩) 0 ⟨16175⟩ 3287

def event69144 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24283⟩⟩) (.authority (.programFamilyFact))

def event69145 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24283⟩⟩) (.finite 3720)

def event69146 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24285⟩⟩) 0 ⟨6689⟩ 5477

def event69147 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24285⟩⟩) 1 ⟨24283⟩ 69145

def event69148 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24285⟩⟩) (.authority (.operator))

def exact69149RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24285⟩⟩]⟩, (1)⟩]

theorem exact69149RawTermsValid :
    exact69149RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69149 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24285⟩⟩) exact69149RawTerms .large 69148 .exactZero (none)

def event69150 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28287⟩⟩) 0 ⟨24285⟩ 69149

def event69151 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28287⟩⟩) (.authority (.operator))

def exact69152RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩, (1)⟩]

theorem exact69152RawTermsValid :
    exact69152RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69152 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28287⟩⟩) exact69152RawTerms (.finite 8192) 69151 .exactZero (none)

def event69153 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23665⟩⟩) 0 ⟨14634⟩ 3281

def event69154 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23665⟩⟩) (.authority (.programFamilyFact))

def event69155 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23665⟩⟩) (.finite 3720)

def event69156 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23666⟩⟩) 0 ⟨6689⟩ 5477

def event69157 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23666⟩⟩) 1 ⟨23665⟩ 69155

def event69158 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23666⟩⟩) (.authority (.operator))

def exact69159RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23666⟩⟩]⟩, (1)⟩]

theorem exact69159RawTermsValid :
    exact69159RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69159 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23666⟩⟩) exact69159RawTerms .large 69158 .exactZero (none)

def event69160 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26215⟩⟩) 0 ⟨23666⟩ 69159

def event69161 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26215⟩⟩) (.authority (.operator))

def exact69162RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩, (1)⟩]

theorem exact69162RawTermsValid :
    exact69162RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69162 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26215⟩⟩) exact69162RawTerms (.finite 8192) 69161 .exactZero (none)

def event69163 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11634⟩⟩) 0 ⟨11633⟩ 3270

def event69164 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11634⟩⟩) 1 ⟨6566⟩ 65295

def event69165 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11634⟩⟩) (.tensor (.predecessor 0 69163 .coefficient) (.predecessor 1 69164 .coefficient) true false)

def event69166 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11634⟩⟩, .operator (⟨3270, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact69167RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact69167RawTermsValid :
    exact69167RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69167 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11634⟩⟩) exact69167RawTerms .large 69165 .exactZero (none)

def event69168 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7199⟩⟩) 0 ⟨5533⟩ 65165

def event69169 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7199⟩⟩) 1 ⟨6781⟩ 10480

def event69170 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7199⟩⟩) (.product (.predecessor 0 69168 .coefficient) (.predecessor 1 69169 .coefficient) (⟨false, false, none, none, none⟩))

def event69171 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7199⟩⟩, .operator (⟨65165, 0⟩, ⟨10480, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩)

def exact69172RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩]

theorem exact69172RawTermsValid :
    exact69172RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69172 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7199⟩⟩) exact69172RawTerms .large 69170 .exactZero (none)

def event69173 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11635⟩⟩) 0 ⟨7199⟩ 69172

def event69174 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11635⟩⟩) 1 ⟨11634⟩ 69167

def event69175 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11635⟩⟩) (.sum [.predecessor 0 69173 .coefficient, .predecessor 1 69174 .coefficient])

def exact69176RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69176RawTermsValid :
    exact69176RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69176 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11635⟩⟩) exact69176RawTerms .large 69175 .exactZero (none)

def event69177 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11636⟩⟩) 0 ⟨11635⟩ 69176

def event69178 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11636⟩⟩) 1 ⟨95⟩ 10472

def event69179 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11636⟩⟩) (.sum [.predecessor 0 69177 .coefficient, .predecessor 1 69178 .coefficient])

def event69180 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11636⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨95⟩⟩]⟩) [⟨.result 10472 .coefficient, false, none⟩])

def event69181 : Event := .survivorFold (1) 69180

def exact69182RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69182RawTermsValid :
    exact69182RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69182 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11636⟩⟩) exact69182RawTerms .large 69179 (.finite 26) (some (69180))

def event69183 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14635⟩⟩) 0 ⟨11636⟩ 69182

def event69184 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14635⟩⟩) 1 ⟨14632⟩ 3273

def event69185 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14635⟩⟩) (.product (.predecessor 0 69183 .coefficient) (.predecessor 1 69184 .coefficient) (⟨false, true, none, none, some 1⟩))

def event69186 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14635⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨14632⟩⟩], []⟩) [⟨.result 3273 .coefficient, true, some 1⟩])

def event69187 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14635⟩⟩) (.product (.result 69182 .summary) (.transfer 69186) (⟨false, false, none, none, none⟩))

def event69188 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14635⟩⟩, .operator (⟨69182, 1⟩, ⟨3273, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event69189 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14635⟩⟩, .operator (⟨69182, 0⟩, ⟨3273, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩)

def exact69190RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩]

theorem exact69190RawTermsValid :
    exact69190RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69190 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14635⟩⟩) exact69190RawTerms .large 69185 (.finite 23296) (some (69187))

def event69191 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14636⟩⟩) 0 ⟨14632⟩ 3273

def event69192 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14636⟩⟩) 1 ⟨6566⟩ 65295

def event69193 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14636⟩⟩) (.tensor (.predecessor 0 69191 .coefficient) (.predecessor 1 69192 .coefficient) true false)

def event69194 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14636⟩⟩, .operator (⟨3273, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact69195RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact69195RawTermsValid :
    exact69195RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69195 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14636⟩⟩) exact69195RawTerms .large 69193 .exactZero (none)

def event69196 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7180⟩⟩) 0 ⟨5533⟩ 65165

def event69197 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7180⟩⟩) 1 ⟨6762⟩ 10521

def event69198 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7180⟩⟩) (.product (.predecessor 0 69196 .coefficient) (.predecessor 1 69197 .coefficient) (⟨false, false, none, none, none⟩))

def event69199 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7180⟩⟩, .operator (⟨65165, 0⟩, ⟨10521, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩]⟩, (1)⟩)

def exact69200RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩]⟩, (1)⟩]

theorem exact69200RawTermsValid :
    exact69200RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69200 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7180⟩⟩) exact69200RawTerms .large 69198 .exactZero (none)

def event69201 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14637⟩⟩) 0 ⟨7180⟩ 69200

def event69202 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14637⟩⟩) 1 ⟨14636⟩ 69195

def event69203 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14637⟩⟩) (.sum [.predecessor 0 69201 .coefficient, .predecessor 1 69202 .coefficient])

def exact69204RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69204RawTermsValid :
    exact69204RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69204 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14637⟩⟩) exact69204RawTerms .large 69203 .exactZero (none)

def event69205 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14638⟩⟩) 0 ⟨14637⟩ 69204

def event69206 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14638⟩⟩) 1 ⟨76⟩ 10513

def event69207 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14638⟩⟩) (.sum [.predecessor 0 69205 .coefficient, .predecessor 1 69206 .coefficient])

def event69208 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14638⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨76⟩⟩]⟩) [⟨.result 10513 .coefficient, false, none⟩])

def event69209 : Event := .survivorFold (1) 69208

def exact69210RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69210RawTermsValid :
    exact69210RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69210 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14638⟩⟩) exact69210RawTerms .large 69207 (.finite 26) (some (69208))

def event69211 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14639⟩⟩) 0 ⟨14638⟩ 69210

def event69212 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14639⟩⟩) 1 ⟨7859⟩ 10510

def event69213 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14639⟩⟩) (.product (.predecessor 0 69211 .coefficient) (.predecessor 1 69212 .coefficient) (⟨false, false, none, none, none⟩))

def event69214 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14639⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩) [⟨.result 10506 .coefficient, false, none⟩])

def event69215 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14639⟩⟩) (.product (.result 69210 .summary) (.transfer 69214) (⟨false, false, none, none, none⟩))

def event69216 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14639⟩⟩, .operator (⟨69210, 1⟩, ⟨10510, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (-1)⟩)

def event69217 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨14639⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7858⟩⟩) ⟨6781⟩ 10480)

def event69218 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14639⟩⟩, .relation 69217 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (-1)⟩)

def event69219 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14639⟩⟩, .operator (⟨69210, 0⟩, ⟨10510, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩)

def exact69220RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (-1)⟩]

theorem exact69220RawTermsValid :
    exact69220RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69220 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14639⟩⟩) exact69220RawTerms .large 69213 (.finite 95420416) (some (69215))

def event69221 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14640⟩⟩) 0 ⟨14639⟩ 69220

def event69222 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14640⟩⟩) 1 ⟨14635⟩ 69190

def event69223 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14640⟩⟩) (.sum [.predecessor 0 69221 .coefficient, .predecessor 1 69222 .coefficient])

def event69224 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14640⟩⟩, .operator (⟨69220, 1⟩, ⟨69190, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩)

def event69225 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14640⟩⟩) (.sum [.result 69220 .summary, .result 69190 .summary])

def exact69226RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact69226RawTermsValid :
    exact69226RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69226 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14640⟩⟩) exact69226RawTerms .large 69223 (.finite 95443712) (some (69225))

def event69227 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26216⟩⟩) 0 ⟨14640⟩ 69226

def event69228 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26216⟩⟩) 1 ⟨26215⟩ 69162

def event69229 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26216⟩⟩) (.product (.predecessor 0 69227 .coefficient) (.predecessor 1 69228 .coefficient) (⟨false, false, none, none, none⟩))

def event69230 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26216⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩) [⟨.result 69162 .coefficient, false, none⟩])

def event69231 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26216⟩⟩) (.product (.result 69226 .summary) (.transfer 69230) (⟨false, false, none, none, none⟩))

def event69232 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26216⟩⟩, .operator (⟨69226, 1⟩, ⟨69162, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩, (-1)⟩)

def event69233 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26216⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26215⟩⟩) ⟨23666⟩ 69159)

def event69234 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26216⟩⟩, .relation 69233 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨23666⟩⟩]⟩, (-1)⟩)

def event69235 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26216⟩⟩, .operator (⟨69226, 0⟩, ⟨69162, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩, (1)⟩)

def exact69236RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨23666⟩⟩]⟩, (-1)⟩]

theorem exact69236RawTermsValid :
    exact69236RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69236 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26216⟩⟩) exact69236RawTerms .large 69229 (.finite 350279950139392) (some (69231))

def event69237 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19668⟩⟩) 0 ⟨14634⟩ 3281

def event69238 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19668⟩⟩) (.authority (.relationPreimageSource ⟨17⟩))

def exact69239RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩, (1)⟩]

theorem exact69239RawTermsValid :
    exact69239RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69239 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19668⟩⟩) exact69239RawTerms (.finite 136065468) 69238 .exactZero (none)

def event69240 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19670⟩⟩) 0 ⟨19668⟩ 69239

def event69241 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19670⟩⟩) 1 ⟨2348⟩ 4

def event69242 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19670⟩⟩) (.scale (.predecessor 0 69240 .coefficient) (.value (.predecessor 1 69241 .coefficient)))

def exact69243RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩, (1)⟩]

theorem exact69243RawTermsValid :
    exact69243RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69243 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19670⟩⟩) exact69243RawTerms (.finite 136065468) 69242 .exactZero (none)

def event69244 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19671⟩⟩) 0 ⟨5535⟩ 65387

def event69245 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19671⟩⟩) 1 ⟨19670⟩ 69243

def event69246 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19671⟩⟩) (.product (.predecessor 0 69244 .coefficient) (.predecessor 1 69245 .coefficient) (⟨false, false, none, none, none⟩))

def event69247 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19671⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩) [⟨.result 69239 .coefficient, false, none⟩])

def event69248 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19671⟩⟩) (.product (.result 65387 .summary) (.transfer 69247) (⟨false, false, none, none, none⟩))

def event69249 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19671⟩⟩, .operator (⟨65387, 0⟩, ⟨69243, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩, (1)⟩)

def event69250 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19669⟩⟩)

def event69251 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event69252 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event69253 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event69254 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event69255 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event69256 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event69257 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event69258 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event69259 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 69258

def event69260 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 69256

def event69261 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 69259 .coefficient) (.value (.predecessor 1 69260 .coefficient)))

def event69262 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event69263 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 69262

def event69264 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 69254

def event69265 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 69263 .coefficient, .predecessor 1 69264 .coefficient])

def event69266 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event69267 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 69266

def event69268 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 69252

def event69269 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 69268 .coefficient))

def event69270 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event69271 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11633⟩⟩) 0 ⟨5530⟩ 69270

def event69272 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11633⟩⟩) (.authority (.programFamilyFact))

def exact69273RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩], []⟩, (1)⟩]

theorem exact69273RawTermsValid :
    exact69273RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69273 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11633⟩⟩) exact69273RawTerms (.finite 28) 69272 .exactZero (none)

def event69274 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14632⟩⟩) 0 ⟨5530⟩ 69270

def event69275 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14632⟩⟩) (.authority (.programFamilyFact))

def exact69276RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14632⟩⟩], []⟩, (1)⟩]

theorem exact69276RawTermsValid :
    exact69276RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69276 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14632⟩⟩) exact69276RawTerms (.finite 28) 69275 .exactZero (none)

def event69277 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14633⟩⟩) 0 ⟨14632⟩ 69276

def event69278 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14633⟩⟩) 1 ⟨11633⟩ 69273

def event69279 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14633⟩⟩) (.product (.predecessor 0 69277 .coefficient) (.predecessor 1 69278 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event69280 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14633⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], []⟩) [⟨.result 69276 .coefficient, true, some 1⟩, ⟨.result 69273 .coefficient, true, some 1⟩])

def event69281 : Event := .survivorFold (1) 69280

def exact69282RawTerms : List Term := []

theorem exact69282RawTermsValid :
    exact69282RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69282 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14633⟩⟩) exact69282RawTerms (.finite 784) 69279 (.finite 784) (some (69280))

def event69283 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14634⟩⟩) 0 ⟨14633⟩ 69282

def event69284 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14634⟩⟩) (.identity (.predecessor 0 69283 .coefficient))

def event69285 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14634⟩⟩) (.finite 784)

def event69286 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19668⟩⟩) 0 ⟨14634⟩ 69285

def event69287 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19668⟩⟩) (.authority (.relationPreimageSource ⟨17⟩))

def exact69288RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩, (1)⟩]

theorem exact69288RawTermsValid :
    exact69288RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69288 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19668⟩⟩) exact69288RawTerms (.finite 136065468) 69287 .exactZero (none)

def event69289 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact69290RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact69290RawTermsValid :
    exact69290RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69290 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact69290RawTerms .large 69289 .exactZero (none)

def event69291 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19669⟩⟩) 0 ⟨6⟩ 69290

def event69292 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19669⟩⟩) 1 ⟨19668⟩ 69288

def event69293 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19669⟩⟩) (.product (.predecessor 0 69291 .coefficient) (.predecessor 1 69292 .coefficient) (⟨false, false, none, none, none⟩))

def event69294 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19669⟩⟩, .operator (⟨69290, 0⟩, ⟨69288, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩, (1)⟩)

def exact69295RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩, (1)⟩]

theorem exact69295RawTermsValid :
    exact69295RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69295 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19669⟩⟩) exact69295RawTerms .large 69293 .exactZero (none)

def event69296 : Event := .preFoldPolynomial 69295 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩, (1)⟩] .exactZero none

def exact69297RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩, (1)⟩]

def event69297 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19669⟩⟩) 69296 exact69297RawTerms .large 69293 .exactZero (none)

def event69298 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26219⟩⟩)

def event69299 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event69300 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event69301 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event69302 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event69303 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event69304 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event69305 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event69306 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event69307 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 69306

def event69308 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 69304

def event69309 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 69307 .coefficient) (.value (.predecessor 1 69308 .coefficient)))

def event69310 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event69311 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 69310

def event69312 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 69302

def event69313 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 69311 .coefficient, .predecessor 1 69312 .coefficient])

def event69314 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event69315 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 69314

def event69316 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 69300

def event69317 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 69316 .coefficient))

def event69318 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event69319 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11633⟩⟩) 0 ⟨5530⟩ 69318

def event69320 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11633⟩⟩) (.authority (.programFamilyFact))

def exact69321RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩], []⟩, (1)⟩]

theorem exact69321RawTermsValid :
    exact69321RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69321 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11633⟩⟩) exact69321RawTerms (.finite 28) 69320 .exactZero (none)

def event69322 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14632⟩⟩) 0 ⟨5530⟩ 69318

def event69323 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14632⟩⟩) (.authority (.programFamilyFact))

def exact69324RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14632⟩⟩], []⟩, (1)⟩]

theorem exact69324RawTermsValid :
    exact69324RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69324 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14632⟩⟩) exact69324RawTerms (.finite 28) 69323 .exactZero (none)

def event69325 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14633⟩⟩) 0 ⟨14632⟩ 69324

def event69326 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14633⟩⟩) 1 ⟨11633⟩ 69321

def event69327 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14633⟩⟩) (.product (.predecessor 0 69325 .coefficient) (.predecessor 1 69326 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event69328 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14633⟩⟩, .operator (⟨69324, 0⟩, ⟨69321, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], []⟩, (1)⟩)

def exact69329RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], []⟩, (1)⟩]

theorem exact69329RawTermsValid :
    exact69329RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69329 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14633⟩⟩) exact69329RawTerms (.finite 784) 69327 .exactZero (none)

def event69330 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14634⟩⟩) 0 ⟨14633⟩ 69329

def event69331 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14634⟩⟩) (.identity (.predecessor 0 69330 .coefficient))

def event69332 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14634⟩⟩) (.finite 784)

def event69333 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23665⟩⟩) 0 ⟨14634⟩ 69332

def event69334 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23665⟩⟩) (.authority (.programFamilyFact))

def event69335 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23665⟩⟩) (.finite 3720)

def event69336 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event69337 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23666⟩⟩) 0 ⟨6689⟩ 69336

def event69338 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23666⟩⟩) 1 ⟨23665⟩ 69335

def event69339 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23666⟩⟩) (.authority (.operator))

def exact69340RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23666⟩⟩]⟩, (1)⟩]

theorem exact69340RawTermsValid :
    exact69340RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69340 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23666⟩⟩) exact69340RawTerms .large 69339 .exactZero (none)

def event69341 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26215⟩⟩) 0 ⟨23666⟩ 69340

def event69342 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26215⟩⟩) (.authority (.operator))

def exact69343RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩, (1)⟩]

theorem exact69343RawTermsValid :
    exact69343RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69343 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26215⟩⟩) exact69343RawTerms (.finite 8192) 69342 .exactZero (none)

def event69344 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event69345 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event69346 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14744⟩⟩) 0 ⟨14634⟩ 69332

def event69347 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14744⟩⟩) 1 ⟨110⟩ 69345

def event69348 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14744⟩⟩) (.sum [.predecessor 0 69346 .coefficient, .predecessor 1 69347 .coefficient])

def event69349 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14744⟩⟩) (.finite 784)

def event69350 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14745⟩⟩) 0 ⟨14744⟩ 69349

def event69351 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14745⟩⟩) (.identity (.predecessor 0 69350 .coefficient))

def exact69352RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], []⟩, (1)⟩]

theorem exact69352RawTermsValid :
    exact69352RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69352 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14745⟩⟩) exact69352RawTerms (.finite 784) 69351 .exactZero (none)

def event69353 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact69354RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact69354RawTermsValid :
    exact69354RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69354 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact69354RawTerms .large 69353 .exactZero (none)

def event69355 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14746⟩⟩) 0 ⟨6544⟩ 69354

def event69356 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14746⟩⟩) 1 ⟨14745⟩ 69352

def event69357 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14746⟩⟩) (.product (.predecessor 0 69355 .coefficient) (.predecessor 1 69356 .coefficient) (⟨false, false, none, none, none⟩))

def event69358 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14746⟩⟩, .operator (⟨69354, 0⟩, ⟨69352, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact69359RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact69359RawTermsValid :
    exact69359RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69359 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14746⟩⟩) exact69359RawTerms .large 69357 .exactZero (none)

def event69360 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event69361 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event69362 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 69336

def event69363 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact69364RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact69364RawTermsValid :
    exact69364RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69364 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact69364RawTerms .large 69363 .exactZero (none)

def event69365 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6781⟩⟩) 0 ⟨6757⟩ 69364

def event69366 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6781⟩⟩) (.identity (.predecessor 0 69365 .coefficient))

def exact69367RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩]

theorem exact69367RawTermsValid :
    exact69367RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69367 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6781⟩⟩) exact69367RawTerms .large 69366 .exactZero (none)

def event69368 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7858⟩⟩) 0 ⟨6781⟩ 69367

def event69369 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7858⟩⟩) (.authority (.operator))

def exact69370RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩]

theorem exact69370RawTermsValid :
    exact69370RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69370 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7858⟩⟩) exact69370RawTerms (.finite 8192) 69369 .exactZero (none)

def event69371 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7859⟩⟩) 0 ⟨7858⟩ 69370

def event69372 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7859⟩⟩) 1 ⟨2348⟩ 69361

def event69373 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7859⟩⟩) (.scale (.predecessor 0 69371 .coefficient) (.value (.predecessor 1 69372 .coefficient)))

def exact69374RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩]

theorem exact69374RawTermsValid :
    exact69374RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event69374 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7859⟩⟩) exact69374RawTerms (.finite 8192) 69373 .exactZero (none)

def event69375 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6762⟩⟩) 0 ⟨6757⟩ 69364

def eventLeaf4320 : Array AnnotatedEvent := #[
  { event := event69120
    frameStart := 69025 },
  { event := event69121
    frameStart := 69025 },
  { event := event69122
    frameStart := 69025 },
  { event := event69123
    frameStart := 69025 },
  { event := event69124
    frameStart := 69025 },
  { event := event69125
    frameStart := 69025 },
  { event := event69126
    frameStart := 69025 },
  { event := event69127
    frameStart := 69025 },
  { event := event69128
    frameStart := 69025 },
  { event := event69129
    frameStart := 0 },
  { event := event69130
    frameStart := 0 },
  { event := event69131
    frameStart := 0 },
  { event := event69132
    frameStart := 0 },
  { event := event69133
    frameStart := 0 },
  { event := event69134
    frameStart := 0 },
  { event := event69135
    frameStart := 0 }
]

def eventLeaf4321 : Array AnnotatedEvent := #[
  { event := event69136
    frameStart := 0 },
  { event := event69137
    frameStart := 0 },
  { event := event69138
    frameStart := 0 },
  { event := event69139
    frameStart := 0 },
  { event := event69140
    frameStart := 0 },
  { event := event69141
    frameStart := 0 },
  { event := event69142
    frameStart := 0 },
  { event := event69143
    frameStart := 0 },
  { event := event69144
    frameStart := 0 },
  { event := event69145
    frameStart := 0 },
  { event := event69146
    frameStart := 0 },
  { event := event69147
    frameStart := 0 },
  { event := event69148
    frameStart := 0 },
  { event := event69149
    frameStart := 0 },
  { event := event69150
    frameStart := 0 },
  { event := event69151
    frameStart := 0 }
]

def eventLeaf4322 : Array AnnotatedEvent := #[
  { event := event69152
    frameStart := 0 },
  { event := event69153
    frameStart := 0 },
  { event := event69154
    frameStart := 0 },
  { event := event69155
    frameStart := 0 },
  { event := event69156
    frameStart := 0 },
  { event := event69157
    frameStart := 0 },
  { event := event69158
    frameStart := 0 },
  { event := event69159
    frameStart := 0 },
  { event := event69160
    frameStart := 0 },
  { event := event69161
    frameStart := 0 },
  { event := event69162
    frameStart := 0 },
  { event := event69163
    frameStart := 0 },
  { event := event69164
    frameStart := 0 },
  { event := event69165
    frameStart := 0 },
  { event := event69166
    frameStart := 0 },
  { event := event69167
    frameStart := 0 }
]

def eventLeaf4323 : Array AnnotatedEvent := #[
  { event := event69168
    frameStart := 0 },
  { event := event69169
    frameStart := 0 },
  { event := event69170
    frameStart := 0 },
  { event := event69171
    frameStart := 0 },
  { event := event69172
    frameStart := 0 },
  { event := event69173
    frameStart := 0 },
  { event := event69174
    frameStart := 0 },
  { event := event69175
    frameStart := 0 },
  { event := event69176
    frameStart := 0 },
  { event := event69177
    frameStart := 0 },
  { event := event69178
    frameStart := 0 },
  { event := event69179
    frameStart := 0 },
  { event := event69180
    frameStart := 0 },
  { event := event69181
    frameStart := 0 },
  { event := event69182
    frameStart := 0 },
  { event := event69183
    frameStart := 0 }
]

def eventLeaf4324 : Array AnnotatedEvent := #[
  { event := event69184
    frameStart := 0 },
  { event := event69185
    frameStart := 0 },
  { event := event69186
    frameStart := 0 },
  { event := event69187
    frameStart := 0 },
  { event := event69188
    frameStart := 0 },
  { event := event69189
    frameStart := 0 },
  { event := event69190
    frameStart := 0 },
  { event := event69191
    frameStart := 0 },
  { event := event69192
    frameStart := 0 },
  { event := event69193
    frameStart := 0 },
  { event := event69194
    frameStart := 0 },
  { event := event69195
    frameStart := 0 },
  { event := event69196
    frameStart := 0 },
  { event := event69197
    frameStart := 0 },
  { event := event69198
    frameStart := 0 },
  { event := event69199
    frameStart := 0 }
]

def eventLeaf4325 : Array AnnotatedEvent := #[
  { event := event69200
    frameStart := 0 },
  { event := event69201
    frameStart := 0 },
  { event := event69202
    frameStart := 0 },
  { event := event69203
    frameStart := 0 },
  { event := event69204
    frameStart := 0 },
  { event := event69205
    frameStart := 0 },
  { event := event69206
    frameStart := 0 },
  { event := event69207
    frameStart := 0 },
  { event := event69208
    frameStart := 0 },
  { event := event69209
    frameStart := 0 },
  { event := event69210
    frameStart := 0 },
  { event := event69211
    frameStart := 0 },
  { event := event69212
    frameStart := 0 },
  { event := event69213
    frameStart := 0 },
  { event := event69214
    frameStart := 0 },
  { event := event69215
    frameStart := 0 }
]

def eventLeaf4326 : Array AnnotatedEvent := #[
  { event := event69216
    frameStart := 0 },
  { event := event69217
    frameStart := 0 },
  { event := event69218
    frameStart := 0 },
  { event := event69219
    frameStart := 0 },
  { event := event69220
    frameStart := 0 },
  { event := event69221
    frameStart := 0 },
  { event := event69222
    frameStart := 0 },
  { event := event69223
    frameStart := 0 },
  { event := event69224
    frameStart := 0 },
  { event := event69225
    frameStart := 0 },
  { event := event69226
    frameStart := 0 },
  { event := event69227
    frameStart := 0 },
  { event := event69228
    frameStart := 0 },
  { event := event69229
    frameStart := 0 },
  { event := event69230
    frameStart := 0 },
  { event := event69231
    frameStart := 0 }
]

def eventLeaf4327 : Array AnnotatedEvent := #[
  { event := event69232
    frameStart := 0 },
  { event := event69233
    frameStart := 0 },
  { event := event69234
    frameStart := 0 },
  { event := event69235
    frameStart := 0 },
  { event := event69236
    frameStart := 0 },
  { event := event69237
    frameStart := 0 },
  { event := event69238
    frameStart := 0 },
  { event := event69239
    frameStart := 0 },
  { event := event69240
    frameStart := 0 },
  { event := event69241
    frameStart := 0 },
  { event := event69242
    frameStart := 0 },
  { event := event69243
    frameStart := 0 },
  { event := event69244
    frameStart := 0 },
  { event := event69245
    frameStart := 0 },
  { event := event69246
    frameStart := 0 },
  { event := event69247
    frameStart := 0 }
]

def eventLeaf4328 : Array AnnotatedEvent := #[
  { event := event69248
    frameStart := 0 },
  { event := event69249
    frameStart := 0 },
  { event := event69250
    frameStart := 69250 },
  { event := event69251
    frameStart := 69250 },
  { event := event69252
    frameStart := 69250 },
  { event := event69253
    frameStart := 69250 },
  { event := event69254
    frameStart := 69250 },
  { event := event69255
    frameStart := 69250 },
  { event := event69256
    frameStart := 69250 },
  { event := event69257
    frameStart := 69250 },
  { event := event69258
    frameStart := 69250 },
  { event := event69259
    frameStart := 69250 },
  { event := event69260
    frameStart := 69250 },
  { event := event69261
    frameStart := 69250 },
  { event := event69262
    frameStart := 69250 },
  { event := event69263
    frameStart := 69250 }
]

def eventLeaf4329 : Array AnnotatedEvent := #[
  { event := event69264
    frameStart := 69250 },
  { event := event69265
    frameStart := 69250 },
  { event := event69266
    frameStart := 69250 },
  { event := event69267
    frameStart := 69250 },
  { event := event69268
    frameStart := 69250 },
  { event := event69269
    frameStart := 69250 },
  { event := event69270
    frameStart := 69250 },
  { event := event69271
    frameStart := 69250 },
  { event := event69272
    frameStart := 69250 },
  { event := event69273
    frameStart := 69250 },
  { event := event69274
    frameStart := 69250 },
  { event := event69275
    frameStart := 69250 },
  { event := event69276
    frameStart := 69250 },
  { event := event69277
    frameStart := 69250 },
  { event := event69278
    frameStart := 69250 },
  { event := event69279
    frameStart := 69250 }
]

def eventLeaf4330 : Array AnnotatedEvent := #[
  { event := event69280
    frameStart := 69250 },
  { event := event69281
    frameStart := 69250 },
  { event := event69282
    frameStart := 69250 },
  { event := event69283
    frameStart := 69250 },
  { event := event69284
    frameStart := 69250 },
  { event := event69285
    frameStart := 69250 },
  { event := event69286
    frameStart := 69250 },
  { event := event69287
    frameStart := 69250 },
  { event := event69288
    frameStart := 69250 },
  { event := event69289
    frameStart := 69250 },
  { event := event69290
    frameStart := 69250 },
  { event := event69291
    frameStart := 69250 },
  { event := event69292
    frameStart := 69250 },
  { event := event69293
    frameStart := 69250 },
  { event := event69294
    frameStart := 69250 },
  { event := event69295
    frameStart := 69250 }
]

def eventLeaf4331 : Array AnnotatedEvent := #[
  { event := event69296
    frameStart := 69250 },
  { event := event69297
    frameStart := 69250 },
  { event := event69298
    frameStart := 69298 },
  { event := event69299
    frameStart := 69298 },
  { event := event69300
    frameStart := 69298 },
  { event := event69301
    frameStart := 69298 },
  { event := event69302
    frameStart := 69298 },
  { event := event69303
    frameStart := 69298 },
  { event := event69304
    frameStart := 69298 },
  { event := event69305
    frameStart := 69298 },
  { event := event69306
    frameStart := 69298 },
  { event := event69307
    frameStart := 69298 },
  { event := event69308
    frameStart := 69298 },
  { event := event69309
    frameStart := 69298 },
  { event := event69310
    frameStart := 69298 },
  { event := event69311
    frameStart := 69298 }
]

def eventLeaf4332 : Array AnnotatedEvent := #[
  { event := event69312
    frameStart := 69298 },
  { event := event69313
    frameStart := 69298 },
  { event := event69314
    frameStart := 69298 },
  { event := event69315
    frameStart := 69298 },
  { event := event69316
    frameStart := 69298 },
  { event := event69317
    frameStart := 69298 },
  { event := event69318
    frameStart := 69298 },
  { event := event69319
    frameStart := 69298 },
  { event := event69320
    frameStart := 69298 },
  { event := event69321
    frameStart := 69298 },
  { event := event69322
    frameStart := 69298 },
  { event := event69323
    frameStart := 69298 },
  { event := event69324
    frameStart := 69298 },
  { event := event69325
    frameStart := 69298 },
  { event := event69326
    frameStart := 69298 },
  { event := event69327
    frameStart := 69298 }
]

def eventLeaf4333 : Array AnnotatedEvent := #[
  { event := event69328
    frameStart := 69298 },
  { event := event69329
    frameStart := 69298 },
  { event := event69330
    frameStart := 69298 },
  { event := event69331
    frameStart := 69298 },
  { event := event69332
    frameStart := 69298 },
  { event := event69333
    frameStart := 69298 },
  { event := event69334
    frameStart := 69298 },
  { event := event69335
    frameStart := 69298 },
  { event := event69336
    frameStart := 69298 },
  { event := event69337
    frameStart := 69298 },
  { event := event69338
    frameStart := 69298 },
  { event := event69339
    frameStart := 69298 },
  { event := event69340
    frameStart := 69298 },
  { event := event69341
    frameStart := 69298 },
  { event := event69342
    frameStart := 69298 },
  { event := event69343
    frameStart := 69298 }
]

def eventLeaf4334 : Array AnnotatedEvent := #[
  { event := event69344
    frameStart := 69298 },
  { event := event69345
    frameStart := 69298 },
  { event := event69346
    frameStart := 69298 },
  { event := event69347
    frameStart := 69298 },
  { event := event69348
    frameStart := 69298 },
  { event := event69349
    frameStart := 69298 },
  { event := event69350
    frameStart := 69298 },
  { event := event69351
    frameStart := 69298 },
  { event := event69352
    frameStart := 69298 },
  { event := event69353
    frameStart := 69298 },
  { event := event69354
    frameStart := 69298 },
  { event := event69355
    frameStart := 69298 },
  { event := event69356
    frameStart := 69298 },
  { event := event69357
    frameStart := 69298 },
  { event := event69358
    frameStart := 69298 },
  { event := event69359
    frameStart := 69298 }
]

def eventLeaf4335 : Array AnnotatedEvent := #[
  { event := event69360
    frameStart := 69298 },
  { event := event69361
    frameStart := 69298 },
  { event := event69362
    frameStart := 69298 },
  { event := event69363
    frameStart := 69298 },
  { event := event69364
    frameStart := 69298 },
  { event := event69365
    frameStart := 69298 },
  { event := event69366
    frameStart := 69298 },
  { event := event69367
    frameStart := 69298 },
  { event := event69368
    frameStart := 69298 },
  { event := event69369
    frameStart := 69298 },
  { event := event69370
    frameStart := 69298 },
  { event := event69371
    frameStart := 69298 },
  { event := event69372
    frameStart := 69298 },
  { event := event69373
    frameStart := 69298 },
  { event := event69374
    frameStart := 69298 },
  { event := event69375
    frameStart := 69298 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events270
