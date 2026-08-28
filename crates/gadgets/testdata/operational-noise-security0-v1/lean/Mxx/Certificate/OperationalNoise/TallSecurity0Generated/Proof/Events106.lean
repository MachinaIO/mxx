import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events106

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event27136 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15909⟩⟩) 0 ⟨15908⟩ 27135

def event27137 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15909⟩⟩) (.identity (.predecessor 0 27136 .coefficient))

def exact27138RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], []⟩, (1)⟩]

theorem exact27138RawTermsValid :
    exact27138RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27138 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15909⟩⟩) exact27138RawTerms (.finite 16) 27137 .exactZero (none)

def event27139 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact27140RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact27140RawTermsValid :
    exact27140RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27140 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact27140RawTerms .large 27139 .exactZero (none)

def event27141 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15910⟩⟩) 0 ⟨6544⟩ 27140

def event27142 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15910⟩⟩) 1 ⟨15909⟩ 27138

def event27143 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15910⟩⟩) (.product (.predecessor 0 27141 .coefficient) (.predecessor 1 27142 .coefficient) (⟨false, false, none, none, none⟩))

def event27144 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15910⟩⟩, .operator (⟨27140, 0⟩, ⟨27138, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact27145RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact27145RawTermsValid :
    exact27145RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27145 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15910⟩⟩) exact27145RawTerms .large 27143 .exactZero (none)

def event27146 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6696⟩⟩) 0 ⟨6689⟩ 27122

def event27147 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6696⟩⟩) (.authority (.operator))

def exact27148RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩]

theorem exact27148RawTermsValid :
    exact27148RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27148 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6696⟩⟩) exact27148RawTerms .large 27147 .exactZero (none)

def event27149 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15911⟩⟩) 0 ⟨6696⟩ 27148

def event27150 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15911⟩⟩) 1 ⟨15910⟩ 27145

def event27151 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15911⟩⟩) (.sum [.predecessor 0 27149 .coefficient, .predecessor 1 27150 .coefficient])

def exact27152RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27152RawTermsValid :
    exact27152RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27152 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15911⟩⟩) exact27152RawTerms .large 27151 .exactZero (none)

def event27153 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27689⟩⟩) 0 ⟨15911⟩ 27152

def event27154 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27689⟩⟩) 1 ⟨27688⟩ 27129

def event27155 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27689⟩⟩) (.product (.predecessor 0 27153 .coefficient) (.predecessor 1 27154 .coefficient) (⟨false, false, none, none, none⟩))

def event27156 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27689⟩⟩, .operator (⟨27152, 0⟩, ⟨27129, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩, (1)⟩)

def event27157 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27689⟩⟩, .operator (⟨27152, 1⟩, ⟨27129, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩, (-1)⟩)

def event27158 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27689⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27688⟩⟩) ⟨24108⟩ 27126)

def event27159 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27689⟩⟩, .relation 27158 0, ⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨24108⟩⟩]⟩, (-1)⟩)

def exact27160RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨24108⟩⟩]⟩, (-1)⟩]

theorem exact27160RawTermsValid :
    exact27160RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27160 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27689⟩⟩) exact27160RawTerms .large 27155 .exactZero (none)

def event27161 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15876⟩⟩) 0 ⟨15834⟩ 27118

def event27162 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15876⟩⟩) (.authority (.programFamilyFact))

def exact27163RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15876⟩⟩], []⟩, (1)⟩]

theorem exact27163RawTermsValid :
    exact27163RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27163 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15876⟩⟩) exact27163RawTerms (.finite 60) 27162 .exactZero (none)

def event27164 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15877⟩⟩) 0 ⟨6544⟩ 27140

def event27165 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15877⟩⟩) 1 ⟨15876⟩ 27163

def event27166 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15877⟩⟩) (.product (.predecessor 0 27164 .coefficient) (.predecessor 1 27165 .coefficient) (⟨false, true, none, none, some 1⟩))

def event27167 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15877⟩⟩, .operator (⟨27140, 0⟩, ⟨27163, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact27168RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact27168RawTermsValid :
    exact27168RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27168 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15877⟩⟩) exact27168RawTerms .large 27166 .exactZero (none)

def event27169 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6721⟩⟩) 0 ⟨6689⟩ 27122

def event27170 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6721⟩⟩) (.authority (.operator))

def exact27171RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩]

theorem exact27171RawTermsValid :
    exact27171RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27171 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6721⟩⟩) exact27171RawTerms .large 27170 .exactZero (none)

def event27172 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15878⟩⟩) 0 ⟨6721⟩ 27171

def event27173 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15878⟩⟩) 1 ⟨15877⟩ 27168

def event27174 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15878⟩⟩) (.sum [.predecessor 0 27172 .coefficient, .predecessor 1 27173 .coefficient])

def exact27175RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27175RawTermsValid :
    exact27175RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27175 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15878⟩⟩) exact27175RawTerms .large 27174 .exactZero (none)

def event27176 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27693⟩⟩) 0 ⟨15878⟩ 27175

def event27177 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27693⟩⟩) 1 ⟨27689⟩ 27160

def event27178 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27693⟩⟩) (.sum [.predecessor 0 27176 .coefficient, .predecessor 1 27177 .coefficient])

def exact27179RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨24108⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27179RawTermsValid :
    exact27179RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27179 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27693⟩⟩) exact27179RawTerms .large 27178 .exactZero (none)

def event27180 : Event := .preFoldPolynomial 27179 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨24108⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact27181RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨24108⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event27181 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨27693⟩⟩) 27180 exact27181RawTerms .large 27178 .exactZero (none)

def event27182 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨15834⟩⟩) ⟨⟨134⟩, ⟨41⟩, ⟨109⟩⟩ ⟨27024, 27182⟩

def event27183 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21271⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩) (1) 0 2 (.universal 27182 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21268⟩⟩]⟩) (none) 27181)

def event27184 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21271⟩⟩, .relation 27183 1, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩)

def event27185 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21271⟩⟩, .relation 27183 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩, (-1)⟩)

def event27186 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21271⟩⟩, .relation 27183 2, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨24108⟩⟩]⟩, (1)⟩)

def event27187 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21271⟩⟩, .relation 27183 3, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact27188RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨24108⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27188RawTermsValid :
    exact27188RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27188 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21271⟩⟩) exact27188RawTerms .large 27020 (.finite 1811303510016) (some (27022))

def event27189 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27691⟩⟩) 0 ⟨21271⟩ 27188

def event27190 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27691⟩⟩) 1 ⟨27690⟩ 27010

def event27191 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27691⟩⟩) (.sum [.predecessor 0 27189 .coefficient, .predecessor 1 27190 .coefficient])

def event27192 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27691⟩⟩, .operator (⟨27188, 0⟩, ⟨27010, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27688⟩⟩]⟩, (1)⟩)

def event27193 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27691⟩⟩, .operator (⟨27188, 2⟩, ⟨27010, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15833⟩⟩], [⟨.program ⟨214⟩, ⟨24108⟩⟩]⟩, (-1)⟩)

def event27194 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27691⟩⟩) (.sum [.result 27188 .summary, .result 27010 .summary])

def exact27195RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15876⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27195RawTermsValid :
    exact27195RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27195 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27691⟩⟩) exact27195RawTerms .large 27191 (.finite 1292046061494565744640) (some (27194))

def event27196 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24043⟩⟩) 0 ⟨15715⟩ 1135

def event27197 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24043⟩⟩) (.authority (.programFamilyFact))

def event27198 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24043⟩⟩) (.finite 3720)

def event27199 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24045⟩⟩) 0 ⟨6689⟩ 5477

def event27200 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24045⟩⟩) 1 ⟨24043⟩ 27198

def event27201 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24045⟩⟩) (.authority (.operator))

def exact27202RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24045⟩⟩]⟩, (1)⟩]

theorem exact27202RawTermsValid :
    exact27202RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27202 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24045⟩⟩) exact27202RawTerms .large 27201 .exactZero (none)

def event27203 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27471⟩⟩) 0 ⟨24045⟩ 27202

def event27204 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27471⟩⟩) (.authority (.operator))

def exact27205RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27471⟩⟩]⟩, (1)⟩]

theorem exact27205RawTermsValid :
    exact27205RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27205 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27471⟩⟩) exact27205RawTerms (.finite 8192) 27204 .exactZero (none)

def event27206 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23505⟩⟩) 0 ⟨13802⟩ 1129

def event27207 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23505⟩⟩) (.authority (.programFamilyFact))

def event27208 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23505⟩⟩) (.finite 3720)

def event27209 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23506⟩⟩) 0 ⟨6689⟩ 5477

def event27210 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23506⟩⟩) 1 ⟨23505⟩ 27208

def event27211 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23506⟩⟩) (.authority (.operator))

def exact27212RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23506⟩⟩]⟩, (1)⟩]

theorem exact27212RawTermsValid :
    exact27212RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27212 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23506⟩⟩) exact27212RawTerms .large 27211 .exactZero (none)

def event27213 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25927⟩⟩) 0 ⟨23506⟩ 27212

def event27214 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25927⟩⟩) (.authority (.operator))

def exact27215RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25927⟩⟩]⟩, (1)⟩]

theorem exact27215RawTermsValid :
    exact27215RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27215 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25927⟩⟩) exact27215RawTerms (.finite 8192) 27214 .exactZero (none)

def event27216 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11314⟩⟩) 0 ⟨11313⟩ 1118

def event27217 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11314⟩⟩) 1 ⟨6570⟩ 21420

def event27218 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11314⟩⟩) (.tensor (.predecessor 0 27216 .coefficient) (.predecessor 1 27217 .coefficient) true false)

def event27219 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11314⟩⟩, .operator (⟨1118, 0⟩, ⟨21420, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact27220RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact27220RawTermsValid :
    exact27220RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27220 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11314⟩⟩) exact27220RawTerms .large 27218 .exactZero (none)

def event27221 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7347⟩⟩) 0 ⟨5557⟩ 21290

def event27222 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7347⟩⟩) 1 ⟨6777⟩ 12484

def event27223 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7347⟩⟩) (.product (.predecessor 0 27221 .coefficient) (.predecessor 1 27222 .coefficient) (⟨false, false, none, none, none⟩))

def event27224 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7347⟩⟩, .operator (⟨21290, 0⟩, ⟨12484, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩)

def exact27225RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩]

theorem exact27225RawTermsValid :
    exact27225RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27225 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7347⟩⟩) exact27225RawTerms .large 27223 .exactZero (none)

def event27226 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11315⟩⟩) 0 ⟨7347⟩ 27225

def event27227 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11315⟩⟩) 1 ⟨11314⟩ 27220

def event27228 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11315⟩⟩) (.sum [.predecessor 0 27226 .coefficient, .predecessor 1 27227 .coefficient])

def exact27229RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27229RawTermsValid :
    exact27229RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27229 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11315⟩⟩) exact27229RawTerms .large 27228 .exactZero (none)

def event27230 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11316⟩⟩) 0 ⟨11315⟩ 27229

def event27231 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11316⟩⟩) 1 ⟨91⟩ 12476

def event27232 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11316⟩⟩) (.sum [.predecessor 0 27230 .coefficient, .predecessor 1 27231 .coefficient])

def event27233 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11316⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨91⟩⟩]⟩) [⟨.result 12476 .coefficient, false, none⟩])

def event27234 : Event := .survivorFold (1) 27233

def exact27235RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27235RawTermsValid :
    exact27235RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27235 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11316⟩⟩) exact27235RawTerms .large 27232 (.finite 26) (some (27233))

def event27236 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13803⟩⟩) 0 ⟨11316⟩ 27235

def event27237 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13803⟩⟩) 1 ⟨13800⟩ 1121

def event27238 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13803⟩⟩) (.product (.predecessor 0 27236 .coefficient) (.predecessor 1 27237 .coefficient) (⟨false, true, none, none, some 1⟩))

def event27239 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13803⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨13800⟩⟩], []⟩) [⟨.result 1121 .coefficient, true, some 1⟩])

def event27240 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13803⟩⟩) (.product (.result 27235 .summary) (.transfer 27239) (⟨false, false, none, none, none⟩))

def event27241 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13803⟩⟩, .operator (⟨27235, 1⟩, ⟨1121, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event27242 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13803⟩⟩, .operator (⟨27235, 0⟩, ⟨1121, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩)

def exact27243RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩]

theorem exact27243RawTermsValid :
    exact27243RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27243 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13803⟩⟩) exact27243RawTerms .large 27238 (.finite 9984) (some (27240))

def event27244 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13804⟩⟩) 0 ⟨13800⟩ 1121

def event27245 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13804⟩⟩) 1 ⟨6570⟩ 21420

def event27246 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13804⟩⟩) (.tensor (.predecessor 0 27244 .coefficient) (.predecessor 1 27245 .coefficient) true false)

def event27247 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13804⟩⟩, .operator (⟨1121, 0⟩, ⟨21420, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact27248RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact27248RawTermsValid :
    exact27248RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27248 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13804⟩⟩) exact27248RawTerms .large 27246 .exactZero (none)

def event27249 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7364⟩⟩) 0 ⟨5557⟩ 21290

def event27250 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7364⟩⟩) 1 ⟨6794⟩ 12525

def event27251 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7364⟩⟩) (.product (.predecessor 0 27249 .coefficient) (.predecessor 1 27250 .coefficient) (⟨false, false, none, none, none⟩))

def event27252 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7364⟩⟩, .operator (⟨21290, 0⟩, ⟨12525, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩]⟩, (1)⟩)

def exact27253RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩]⟩, (1)⟩]

theorem exact27253RawTermsValid :
    exact27253RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27253 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7364⟩⟩) exact27253RawTerms .large 27251 .exactZero (none)

def event27254 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13805⟩⟩) 0 ⟨7364⟩ 27253

def event27255 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13805⟩⟩) 1 ⟨13804⟩ 27248

def event27256 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13805⟩⟩) (.sum [.predecessor 0 27254 .coefficient, .predecessor 1 27255 .coefficient])

def exact27257RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27257RawTermsValid :
    exact27257RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27257 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13805⟩⟩) exact27257RawTerms .large 27256 .exactZero (none)

def event27258 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13806⟩⟩) 0 ⟨13805⟩ 27257

def event27259 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13806⟩⟩) 1 ⟨108⟩ 12517

def event27260 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13806⟩⟩) (.sum [.predecessor 0 27258 .coefficient, .predecessor 1 27259 .coefficient])

def event27261 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13806⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨108⟩⟩]⟩) [⟨.result 12517 .coefficient, false, none⟩])

def event27262 : Event := .survivorFold (1) 27261

def exact27263RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27263RawTermsValid :
    exact27263RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27263 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13806⟩⟩) exact27263RawTerms .large 27260 (.finite 26) (some (27261))

def event27264 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13807⟩⟩) 0 ⟨13806⟩ 27263

def event27265 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13807⟩⟩) 1 ⟨7847⟩ 12514

def event27266 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13807⟩⟩) (.product (.predecessor 0 27264 .coefficient) (.predecessor 1 27265 .coefficient) (⟨false, false, none, none, none⟩))

def event27267 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13807⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩) [⟨.result 12510 .coefficient, false, none⟩])

def event27268 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13807⟩⟩) (.product (.result 27263 .summary) (.transfer 27267) (⟨false, false, none, none, none⟩))

def event27269 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13807⟩⟩, .operator (⟨27263, 1⟩, ⟨12514, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (-1)⟩)

def event27270 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨13807⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7846⟩⟩) ⟨6777⟩ 12484)

def event27271 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13807⟩⟩, .relation 27270 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (-1)⟩)

def event27272 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13807⟩⟩, .operator (⟨27263, 0⟩, ⟨12514, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (1)⟩)

def exact27273RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (-1)⟩]

theorem exact27273RawTermsValid :
    exact27273RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27273 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13807⟩⟩) exact27273RawTerms .large 27266 (.finite 95420416) (some (27268))

def event27274 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13808⟩⟩) 0 ⟨13807⟩ 27273

def event27275 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13808⟩⟩) 1 ⟨13803⟩ 27243

def event27276 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13808⟩⟩) (.sum [.predecessor 0 27274 .coefficient, .predecessor 1 27275 .coefficient])

def event27277 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13808⟩⟩, .operator (⟨27273, 1⟩, ⟨27243, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨6777⟩⟩]⟩, (1)⟩)

def event27278 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13808⟩⟩) (.sum [.result 27273 .summary, .result 27243 .summary])

def exact27279RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact27279RawTermsValid :
    exact27279RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27279 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13808⟩⟩) exact27279RawTerms .large 27276 (.finite 95430400) (some (27278))

def event27280 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25928⟩⟩) 0 ⟨13808⟩ 27279

def event27281 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25928⟩⟩) 1 ⟨25927⟩ 27215

def event27282 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25928⟩⟩) (.product (.predecessor 0 27280 .coefficient) (.predecessor 1 27281 .coefficient) (⟨false, false, none, none, none⟩))

def event27283 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25928⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25927⟩⟩]⟩) [⟨.result 27215 .coefficient, false, none⟩])

def event27284 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25928⟩⟩) (.product (.result 27279 .summary) (.transfer 27283) (⟨false, false, none, none, none⟩))

def event27285 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25928⟩⟩, .operator (⟨27279, 1⟩, ⟨27215, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25927⟩⟩]⟩, (-1)⟩)

def event27286 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25928⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25927⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25927⟩⟩) ⟨23506⟩ 27212)

def event27287 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25928⟩⟩, .relation 27286 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨23506⟩⟩]⟩, (-1)⟩)

def event27288 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25928⟩⟩, .operator (⟨27279, 0⟩, ⟨27215, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25927⟩⟩]⟩, (1)⟩)

def exact27289RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25927⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], [⟨.program ⟨214⟩, ⟨23506⟩⟩]⟩, (-1)⟩]

theorem exact27289RawTermsValid :
    exact27289RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27289 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25928⟩⟩) exact27289RawTerms .large 27282 (.finite 350231094886400) (some (27284))

def event27290 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19396⟩⟩) 0 ⟨13802⟩ 1129

def event27291 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19396⟩⟩) (.authority (.relationPreimageSource ⟨13⟩))

def exact27292RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19396⟩⟩]⟩, (1)⟩]

theorem exact27292RawTermsValid :
    exact27292RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27292 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19396⟩⟩) exact27292RawTerms (.finite 136065468) 27291 .exactZero (none)

def event27293 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19398⟩⟩) 0 ⟨19396⟩ 27292

def event27294 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19398⟩⟩) 1 ⟨2348⟩ 4

def event27295 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19398⟩⟩) (.scale (.predecessor 0 27293 .coefficient) (.value (.predecessor 1 27294 .coefficient)))

def exact27296RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19396⟩⟩]⟩, (1)⟩]

theorem exact27296RawTermsValid :
    exact27296RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27296 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19398⟩⟩) exact27296RawTerms (.finite 136065468) 27295 .exactZero (none)

def event27297 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19399⟩⟩) 0 ⟨5559⟩ 21512

def event27298 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19399⟩⟩) 1 ⟨19398⟩ 27296

def event27299 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19399⟩⟩) (.product (.predecessor 0 27297 .coefficient) (.predecessor 1 27298 .coefficient) (⟨false, false, none, none, none⟩))

def event27300 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19399⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19396⟩⟩]⟩) [⟨.result 27292 .coefficient, false, none⟩])

def event27301 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19399⟩⟩) (.product (.result 21512 .summary) (.transfer 27300) (⟨false, false, none, none, none⟩))

def event27302 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19399⟩⟩, .operator (⟨21512, 0⟩, ⟨27296, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19396⟩⟩]⟩, (1)⟩)

def event27303 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19397⟩⟩)

def event27304 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event27305 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event27306 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event27307 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event27308 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event27309 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event27310 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event27311 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event27312 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 27311

def event27313 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 27309

def event27314 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 27312 .coefficient) (.value (.predecessor 1 27313 .coefficient)))

def event27315 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event27316 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 27315

def event27317 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 27307

def event27318 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 27316 .coefficient, .predecessor 1 27317 .coefficient])

def event27319 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event27320 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 27319

def event27321 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 27305

def event27322 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 27321 .coefficient))

def event27323 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event27324 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11313⟩⟩) 0 ⟨5554⟩ 27323

def event27325 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11313⟩⟩) (.authority (.programFamilyFact))

def exact27326RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11313⟩⟩], []⟩, (1)⟩]

theorem exact27326RawTermsValid :
    exact27326RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27326 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11313⟩⟩) exact27326RawTerms (.finite 12) 27325 .exactZero (none)

def event27327 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13800⟩⟩) 0 ⟨5554⟩ 27323

def event27328 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13800⟩⟩) (.authority (.programFamilyFact))

def exact27329RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13800⟩⟩], []⟩, (1)⟩]

theorem exact27329RawTermsValid :
    exact27329RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27329 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13800⟩⟩) exact27329RawTerms (.finite 12) 27328 .exactZero (none)

def event27330 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13801⟩⟩) 0 ⟨13800⟩ 27329

def event27331 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13801⟩⟩) 1 ⟨11313⟩ 27326

def event27332 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13801⟩⟩) (.product (.predecessor 0 27330 .coefficient) (.predecessor 1 27331 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event27333 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13801⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], []⟩) [⟨.result 27329 .coefficient, true, some 1⟩, ⟨.result 27326 .coefficient, true, some 1⟩])

def event27334 : Event := .survivorFold (1) 27333

def exact27335RawTerms : List Term := []

theorem exact27335RawTermsValid :
    exact27335RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27335 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13801⟩⟩) exact27335RawTerms (.finite 144) 27332 (.finite 144) (some (27333))

def event27336 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13802⟩⟩) 0 ⟨13801⟩ 27335

def event27337 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13802⟩⟩) (.identity (.predecessor 0 27336 .coefficient))

def event27338 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13802⟩⟩) (.finite 144)

def event27339 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19396⟩⟩) 0 ⟨13802⟩ 27338

def event27340 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19396⟩⟩) (.authority (.relationPreimageSource ⟨13⟩))

def exact27341RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19396⟩⟩]⟩, (1)⟩]

theorem exact27341RawTermsValid :
    exact27341RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27341 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19396⟩⟩) exact27341RawTerms (.finite 136065468) 27340 .exactZero (none)

def event27342 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact27343RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact27343RawTermsValid :
    exact27343RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27343 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact27343RawTerms .large 27342 .exactZero (none)

def event27344 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19397⟩⟩) 0 ⟨6⟩ 27343

def event27345 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19397⟩⟩) 1 ⟨19396⟩ 27341

def event27346 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19397⟩⟩) (.product (.predecessor 0 27344 .coefficient) (.predecessor 1 27345 .coefficient) (⟨false, false, none, none, none⟩))

def event27347 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19397⟩⟩, .operator (⟨27343, 0⟩, ⟨27341, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19396⟩⟩]⟩, (1)⟩)

def exact27348RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19396⟩⟩]⟩, (1)⟩]

theorem exact27348RawTermsValid :
    exact27348RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27348 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19397⟩⟩) exact27348RawTerms .large 27346 .exactZero (none)

def event27349 : Event := .preFoldPolynomial 27348 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19396⟩⟩]⟩, (1)⟩] .exactZero none

def exact27350RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19396⟩⟩]⟩, (1)⟩]

def event27350 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19397⟩⟩) 27349 exact27350RawTerms .large 27346 .exactZero (none)

def event27351 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25931⟩⟩)

def event27352 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event27353 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event27354 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event27355 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event27356 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event27357 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event27358 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event27359 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event27360 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 27359

def event27361 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 27357

def event27362 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 27360 .coefficient) (.value (.predecessor 1 27361 .coefficient)))

def event27363 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event27364 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 27363

def event27365 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 27355

def event27366 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 27364 .coefficient, .predecessor 1 27365 .coefficient])

def event27367 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event27368 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 27367

def event27369 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 27353

def event27370 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 27369 .coefficient))

def event27371 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event27372 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11313⟩⟩) 0 ⟨5554⟩ 27371

def event27373 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11313⟩⟩) (.authority (.programFamilyFact))

def exact27374RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11313⟩⟩], []⟩, (1)⟩]

theorem exact27374RawTermsValid :
    exact27374RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27374 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11313⟩⟩) exact27374RawTerms (.finite 12) 27373 .exactZero (none)

def event27375 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13800⟩⟩) 0 ⟨5554⟩ 27371

def event27376 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13800⟩⟩) (.authority (.programFamilyFact))

def exact27377RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13800⟩⟩], []⟩, (1)⟩]

theorem exact27377RawTermsValid :
    exact27377RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27377 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13800⟩⟩) exact27377RawTerms (.finite 12) 27376 .exactZero (none)

def event27378 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13801⟩⟩) 0 ⟨13800⟩ 27377

def event27379 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13801⟩⟩) 1 ⟨11313⟩ 27374

def event27380 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13801⟩⟩) (.product (.predecessor 0 27378 .coefficient) (.predecessor 1 27379 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event27381 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13801⟩⟩, .operator (⟨27377, 0⟩, ⟨27374, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], []⟩, (1)⟩)

def exact27382RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11313⟩⟩, ⟨.program ⟨214⟩, ⟨13800⟩⟩], []⟩, (1)⟩]

theorem exact27382RawTermsValid :
    exact27382RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event27382 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13801⟩⟩) exact27382RawTerms (.finite 144) 27380 .exactZero (none)

def event27383 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13802⟩⟩) 0 ⟨13801⟩ 27382

def event27384 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13802⟩⟩) (.identity (.predecessor 0 27383 .coefficient))

def event27385 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13802⟩⟩) (.finite 144)

def event27386 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23505⟩⟩) 0 ⟨13802⟩ 27385

def event27387 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23505⟩⟩) (.authority (.programFamilyFact))

def event27388 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23505⟩⟩) (.finite 3720)

def event27389 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event27390 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23506⟩⟩) 0 ⟨6689⟩ 27389

def event27391 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23506⟩⟩) 1 ⟨23505⟩ 27388

def eventLeaf1696 : Array AnnotatedEvent := #[
  { event := event27136
    frameStart := 27078 },
  { event := event27137
    frameStart := 27078 },
  { event := event27138
    frameStart := 27078 },
  { event := event27139
    frameStart := 27078 },
  { event := event27140
    frameStart := 27078 },
  { event := event27141
    frameStart := 27078 },
  { event := event27142
    frameStart := 27078 },
  { event := event27143
    frameStart := 27078 },
  { event := event27144
    frameStart := 27078 },
  { event := event27145
    frameStart := 27078 },
  { event := event27146
    frameStart := 27078 },
  { event := event27147
    frameStart := 27078 },
  { event := event27148
    frameStart := 27078 },
  { event := event27149
    frameStart := 27078 },
  { event := event27150
    frameStart := 27078 },
  { event := event27151
    frameStart := 27078 }
]

def eventLeaf1697 : Array AnnotatedEvent := #[
  { event := event27152
    frameStart := 27078 },
  { event := event27153
    frameStart := 27078 },
  { event := event27154
    frameStart := 27078 },
  { event := event27155
    frameStart := 27078 },
  { event := event27156
    frameStart := 27078 },
  { event := event27157
    frameStart := 27078 },
  { event := event27158
    frameStart := 27078 },
  { event := event27159
    frameStart := 27078 },
  { event := event27160
    frameStart := 27078 },
  { event := event27161
    frameStart := 27078 },
  { event := event27162
    frameStart := 27078 },
  { event := event27163
    frameStart := 27078 },
  { event := event27164
    frameStart := 27078 },
  { event := event27165
    frameStart := 27078 },
  { event := event27166
    frameStart := 27078 },
  { event := event27167
    frameStart := 27078 }
]

def eventLeaf1698 : Array AnnotatedEvent := #[
  { event := event27168
    frameStart := 27078 },
  { event := event27169
    frameStart := 27078 },
  { event := event27170
    frameStart := 27078 },
  { event := event27171
    frameStart := 27078 },
  { event := event27172
    frameStart := 27078 },
  { event := event27173
    frameStart := 27078 },
  { event := event27174
    frameStart := 27078 },
  { event := event27175
    frameStart := 27078 },
  { event := event27176
    frameStart := 27078 },
  { event := event27177
    frameStart := 27078 },
  { event := event27178
    frameStart := 27078 },
  { event := event27179
    frameStart := 27078 },
  { event := event27180
    frameStart := 27078 },
  { event := event27181
    frameStart := 27078 },
  { event := event27182
    frameStart := 0 },
  { event := event27183
    frameStart := 0 }
]

def eventLeaf1699 : Array AnnotatedEvent := #[
  { event := event27184
    frameStart := 0 },
  { event := event27185
    frameStart := 0 },
  { event := event27186
    frameStart := 0 },
  { event := event27187
    frameStart := 0 },
  { event := event27188
    frameStart := 0 },
  { event := event27189
    frameStart := 0 },
  { event := event27190
    frameStart := 0 },
  { event := event27191
    frameStart := 0 },
  { event := event27192
    frameStart := 0 },
  { event := event27193
    frameStart := 0 },
  { event := event27194
    frameStart := 0 },
  { event := event27195
    frameStart := 0 },
  { event := event27196
    frameStart := 0 },
  { event := event27197
    frameStart := 0 },
  { event := event27198
    frameStart := 0 },
  { event := event27199
    frameStart := 0 }
]

def eventLeaf1700 : Array AnnotatedEvent := #[
  { event := event27200
    frameStart := 0 },
  { event := event27201
    frameStart := 0 },
  { event := event27202
    frameStart := 0 },
  { event := event27203
    frameStart := 0 },
  { event := event27204
    frameStart := 0 },
  { event := event27205
    frameStart := 0 },
  { event := event27206
    frameStart := 0 },
  { event := event27207
    frameStart := 0 },
  { event := event27208
    frameStart := 0 },
  { event := event27209
    frameStart := 0 },
  { event := event27210
    frameStart := 0 },
  { event := event27211
    frameStart := 0 },
  { event := event27212
    frameStart := 0 },
  { event := event27213
    frameStart := 0 },
  { event := event27214
    frameStart := 0 },
  { event := event27215
    frameStart := 0 }
]

def eventLeaf1701 : Array AnnotatedEvent := #[
  { event := event27216
    frameStart := 0 },
  { event := event27217
    frameStart := 0 },
  { event := event27218
    frameStart := 0 },
  { event := event27219
    frameStart := 0 },
  { event := event27220
    frameStart := 0 },
  { event := event27221
    frameStart := 0 },
  { event := event27222
    frameStart := 0 },
  { event := event27223
    frameStart := 0 },
  { event := event27224
    frameStart := 0 },
  { event := event27225
    frameStart := 0 },
  { event := event27226
    frameStart := 0 },
  { event := event27227
    frameStart := 0 },
  { event := event27228
    frameStart := 0 },
  { event := event27229
    frameStart := 0 },
  { event := event27230
    frameStart := 0 },
  { event := event27231
    frameStart := 0 }
]

def eventLeaf1702 : Array AnnotatedEvent := #[
  { event := event27232
    frameStart := 0 },
  { event := event27233
    frameStart := 0 },
  { event := event27234
    frameStart := 0 },
  { event := event27235
    frameStart := 0 },
  { event := event27236
    frameStart := 0 },
  { event := event27237
    frameStart := 0 },
  { event := event27238
    frameStart := 0 },
  { event := event27239
    frameStart := 0 },
  { event := event27240
    frameStart := 0 },
  { event := event27241
    frameStart := 0 },
  { event := event27242
    frameStart := 0 },
  { event := event27243
    frameStart := 0 },
  { event := event27244
    frameStart := 0 },
  { event := event27245
    frameStart := 0 },
  { event := event27246
    frameStart := 0 },
  { event := event27247
    frameStart := 0 }
]

def eventLeaf1703 : Array AnnotatedEvent := #[
  { event := event27248
    frameStart := 0 },
  { event := event27249
    frameStart := 0 },
  { event := event27250
    frameStart := 0 },
  { event := event27251
    frameStart := 0 },
  { event := event27252
    frameStart := 0 },
  { event := event27253
    frameStart := 0 },
  { event := event27254
    frameStart := 0 },
  { event := event27255
    frameStart := 0 },
  { event := event27256
    frameStart := 0 },
  { event := event27257
    frameStart := 0 },
  { event := event27258
    frameStart := 0 },
  { event := event27259
    frameStart := 0 },
  { event := event27260
    frameStart := 0 },
  { event := event27261
    frameStart := 0 },
  { event := event27262
    frameStart := 0 },
  { event := event27263
    frameStart := 0 }
]

def eventLeaf1704 : Array AnnotatedEvent := #[
  { event := event27264
    frameStart := 0 },
  { event := event27265
    frameStart := 0 },
  { event := event27266
    frameStart := 0 },
  { event := event27267
    frameStart := 0 },
  { event := event27268
    frameStart := 0 },
  { event := event27269
    frameStart := 0 },
  { event := event27270
    frameStart := 0 },
  { event := event27271
    frameStart := 0 },
  { event := event27272
    frameStart := 0 },
  { event := event27273
    frameStart := 0 },
  { event := event27274
    frameStart := 0 },
  { event := event27275
    frameStart := 0 },
  { event := event27276
    frameStart := 0 },
  { event := event27277
    frameStart := 0 },
  { event := event27278
    frameStart := 0 },
  { event := event27279
    frameStart := 0 }
]

def eventLeaf1705 : Array AnnotatedEvent := #[
  { event := event27280
    frameStart := 0 },
  { event := event27281
    frameStart := 0 },
  { event := event27282
    frameStart := 0 },
  { event := event27283
    frameStart := 0 },
  { event := event27284
    frameStart := 0 },
  { event := event27285
    frameStart := 0 },
  { event := event27286
    frameStart := 0 },
  { event := event27287
    frameStart := 0 },
  { event := event27288
    frameStart := 0 },
  { event := event27289
    frameStart := 0 },
  { event := event27290
    frameStart := 0 },
  { event := event27291
    frameStart := 0 },
  { event := event27292
    frameStart := 0 },
  { event := event27293
    frameStart := 0 },
  { event := event27294
    frameStart := 0 },
  { event := event27295
    frameStart := 0 }
]

def eventLeaf1706 : Array AnnotatedEvent := #[
  { event := event27296
    frameStart := 0 },
  { event := event27297
    frameStart := 0 },
  { event := event27298
    frameStart := 0 },
  { event := event27299
    frameStart := 0 },
  { event := event27300
    frameStart := 0 },
  { event := event27301
    frameStart := 0 },
  { event := event27302
    frameStart := 0 },
  { event := event27303
    frameStart := 27303 },
  { event := event27304
    frameStart := 27303 },
  { event := event27305
    frameStart := 27303 },
  { event := event27306
    frameStart := 27303 },
  { event := event27307
    frameStart := 27303 },
  { event := event27308
    frameStart := 27303 },
  { event := event27309
    frameStart := 27303 },
  { event := event27310
    frameStart := 27303 },
  { event := event27311
    frameStart := 27303 }
]

def eventLeaf1707 : Array AnnotatedEvent := #[
  { event := event27312
    frameStart := 27303 },
  { event := event27313
    frameStart := 27303 },
  { event := event27314
    frameStart := 27303 },
  { event := event27315
    frameStart := 27303 },
  { event := event27316
    frameStart := 27303 },
  { event := event27317
    frameStart := 27303 },
  { event := event27318
    frameStart := 27303 },
  { event := event27319
    frameStart := 27303 },
  { event := event27320
    frameStart := 27303 },
  { event := event27321
    frameStart := 27303 },
  { event := event27322
    frameStart := 27303 },
  { event := event27323
    frameStart := 27303 },
  { event := event27324
    frameStart := 27303 },
  { event := event27325
    frameStart := 27303 },
  { event := event27326
    frameStart := 27303 },
  { event := event27327
    frameStart := 27303 }
]

def eventLeaf1708 : Array AnnotatedEvent := #[
  { event := event27328
    frameStart := 27303 },
  { event := event27329
    frameStart := 27303 },
  { event := event27330
    frameStart := 27303 },
  { event := event27331
    frameStart := 27303 },
  { event := event27332
    frameStart := 27303 },
  { event := event27333
    frameStart := 27303 },
  { event := event27334
    frameStart := 27303 },
  { event := event27335
    frameStart := 27303 },
  { event := event27336
    frameStart := 27303 },
  { event := event27337
    frameStart := 27303 },
  { event := event27338
    frameStart := 27303 },
  { event := event27339
    frameStart := 27303 },
  { event := event27340
    frameStart := 27303 },
  { event := event27341
    frameStart := 27303 },
  { event := event27342
    frameStart := 27303 },
  { event := event27343
    frameStart := 27303 }
]

def eventLeaf1709 : Array AnnotatedEvent := #[
  { event := event27344
    frameStart := 27303 },
  { event := event27345
    frameStart := 27303 },
  { event := event27346
    frameStart := 27303 },
  { event := event27347
    frameStart := 27303 },
  { event := event27348
    frameStart := 27303 },
  { event := event27349
    frameStart := 27303 },
  { event := event27350
    frameStart := 27303 },
  { event := event27351
    frameStart := 27351 },
  { event := event27352
    frameStart := 27351 },
  { event := event27353
    frameStart := 27351 },
  { event := event27354
    frameStart := 27351 },
  { event := event27355
    frameStart := 27351 },
  { event := event27356
    frameStart := 27351 },
  { event := event27357
    frameStart := 27351 },
  { event := event27358
    frameStart := 27351 },
  { event := event27359
    frameStart := 27351 }
]

def eventLeaf1710 : Array AnnotatedEvent := #[
  { event := event27360
    frameStart := 27351 },
  { event := event27361
    frameStart := 27351 },
  { event := event27362
    frameStart := 27351 },
  { event := event27363
    frameStart := 27351 },
  { event := event27364
    frameStart := 27351 },
  { event := event27365
    frameStart := 27351 },
  { event := event27366
    frameStart := 27351 },
  { event := event27367
    frameStart := 27351 },
  { event := event27368
    frameStart := 27351 },
  { event := event27369
    frameStart := 27351 },
  { event := event27370
    frameStart := 27351 },
  { event := event27371
    frameStart := 27351 },
  { event := event27372
    frameStart := 27351 },
  { event := event27373
    frameStart := 27351 },
  { event := event27374
    frameStart := 27351 },
  { event := event27375
    frameStart := 27351 }
]

def eventLeaf1711 : Array AnnotatedEvent := #[
  { event := event27376
    frameStart := 27351 },
  { event := event27377
    frameStart := 27351 },
  { event := event27378
    frameStart := 27351 },
  { event := event27379
    frameStart := 27351 },
  { event := event27380
    frameStart := 27351 },
  { event := event27381
    frameStart := 27351 },
  { event := event27382
    frameStart := 27351 },
  { event := event27383
    frameStart := 27351 },
  { event := event27384
    frameStart := 27351 },
  { event := event27385
    frameStart := 27351 },
  { event := event27386
    frameStart := 27351 },
  { event := event27387
    frameStart := 27351 },
  { event := event27388
    frameStart := 27351 },
  { event := event27389
    frameStart := 27351 },
  { event := event27390
    frameStart := 27351 },
  { event := event27391
    frameStart := 27351 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events106
