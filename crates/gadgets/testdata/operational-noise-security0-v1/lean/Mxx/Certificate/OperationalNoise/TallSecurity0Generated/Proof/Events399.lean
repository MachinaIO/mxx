import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events399

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event102144 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15259⟩⟩) (.product (.predecessor 0 102142 .coefficient) (.predecessor 1 102143 .coefficient) (⟨false, true, none, none, some 1⟩))

def event102145 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15259⟩⟩, .operator (⟨102118, 0⟩, ⟨102141, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact102146RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact102146RawTermsValid :
    exact102146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102146 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15259⟩⟩) exact102146RawTerms .large 102144 .exactZero (none)

def event102147 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6709⟩⟩) 0 ⟨6689⟩ 102100

def event102148 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6709⟩⟩) (.authority (.operator))

def exact102149RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩]

theorem exact102149RawTermsValid :
    exact102149RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102149 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6709⟩⟩) exact102149RawTerms .large 102148 .exactZero (none)

def event102150 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15260⟩⟩) 0 ⟨6709⟩ 102149

def event102151 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15260⟩⟩) 1 ⟨15259⟩ 102146

def event102152 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15260⟩⟩) (.sum [.predecessor 0 102150 .coefficient, .predecessor 1 102151 .coefficient])

def exact102153RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102153RawTermsValid :
    exact102153RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102153 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15260⟩⟩) exact102153RawTerms .large 102152 .exactZero (none)

def event102154 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26330⟩⟩) 0 ⟨15260⟩ 102153

def event102155 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26330⟩⟩) 1 ⟨26327⟩ 102138

def event102156 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26330⟩⟩) (.sum [.predecessor 0 102154 .coefficient, .predecessor 1 102155 .coefficient])

def exact102157RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23712⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102157RawTermsValid :
    exact102157RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102157 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26330⟩⟩) exact102157RawTerms .large 102156 .exactZero (none)

def event102158 : Event := .preFoldPolynomial 102157 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23712⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact102159RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23712⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event102159 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26330⟩⟩) 102158 exact102159RawTerms .large 102156 .exactZero (none)

def event102160 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14783⟩⟩) ⟨⟨122⟩, ⟨28⟩, ⟨109⟩⟩ ⟨102026, 102160⟩

def event102161 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20384⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20381⟩⟩]⟩) (1) 0 2 (.universal 102160 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20381⟩⟩]⟩) (none) 102159)

def event102162 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20384⟩⟩, .relation 102161 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩)

def event102163 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20384⟩⟩, .relation 102161 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩]⟩, (-1)⟩)

def event102164 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20384⟩⟩, .relation 102161 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23712⟩⟩]⟩, (1)⟩)

def event102165 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20384⟩⟩, .relation 102161 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact102166RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23712⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102166RawTermsValid :
    exact102166RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102166 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20384⟩⟩) exact102166RawTerms .large 102022 (.finite 1811303510016) (some (102024))

def event102167 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26329⟩⟩) 0 ⟨20384⟩ 102166

def event102168 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26329⟩⟩) 1 ⟨26328⟩ 102012

def event102169 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26329⟩⟩) (.sum [.predecessor 0 102167 .coefficient, .predecessor 1 102168 .coefficient])

def event102170 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26329⟩⟩, .operator (⟨102166, 0⟩, ⟨102012, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩]⟩, (1)⟩)

def event102171 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26329⟩⟩, .operator (⟨102166, 2⟩, ⟨102012, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨23712⟩⟩]⟩, (-1)⟩)

def event102172 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26329⟩⟩) (.sum [.result 102166 .summary, .result 102012 .summary])

def exact102173RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102173RawTermsValid :
    exact102173RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102173 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26329⟩⟩) exact102173RawTerms .large 102169 (.finite 1291889174379421642752) (some (102172))

def event102174 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26533⟩⟩) 0 ⟨26329⟩ 102173

def event102175 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26533⟩⟩) 1 ⟨26532⟩ 101739

def event102176 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26533⟩⟩) (.sum [.predecessor 0 102174 .coefficient, .predecessor 1 102175 .coefficient])

def event102177 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26533⟩⟩) (.sum [.result 102173 .summary, .result 101739 .summary])

def exact102178RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102178RawTermsValid :
    exact102178RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102178 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26533⟩⟩) exact102178RawTerms .large 102176 (.finite 2583789554981353578496) (some (102177))

def event102179 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26750⟩⟩) 0 ⟨26533⟩ 102178

def event102180 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26750⟩⟩) 1 ⟨26749⟩ 101305

def event102181 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26750⟩⟩) (.sum [.predecessor 0 102179 .coefficient, .predecessor 1 102180 .coefficient])

def event102182 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26750⟩⟩) (.sum [.result 102178 .summary, .result 101305 .summary])

def exact102183RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102183RawTermsValid :
    exact102183RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102183 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26750⟩⟩) exact102183RawTerms .large 102181 (.finite 3875701141805795807232) (some (102182))

def event102184 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26967⟩⟩) 0 ⟨26750⟩ 102183

def event102185 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26967⟩⟩) 1 ⟨26966⟩ 100871

def event102186 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26967⟩⟩) (.sum [.predecessor 0 102184 .coefficient, .predecessor 1 102185 .coefficient])

def event102187 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26967⟩⟩) (.sum [.result 102183 .summary, .result 100871 .summary])

def exact102188RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102188RawTermsValid :
    exact102188RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102188 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26967⟩⟩) exact102188RawTerms .large 102186 (.finite 5167635141075258621952) (some (102187))

def event102189 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27184⟩⟩) 0 ⟨26967⟩ 102188

def event102190 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27184⟩⟩) 1 ⟨27183⟩ 100437

def event102191 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27184⟩⟩) (.sum [.predecessor 0 102189 .coefficient, .predecessor 1 102190 .coefficient])

def event102192 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27184⟩⟩) (.sum [.result 102188 .summary, .result 100437 .summary])

def exact102193RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102193RawTermsValid :
    exact102193RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102193 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27184⟩⟩) exact102193RawTerms .large 102191 (.finite 6459613965234762608640) (some (102192))

def event102194 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27401⟩⟩) 0 ⟨27184⟩ 102193

def event102195 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27401⟩⟩) 1 ⟨27400⟩ 100003

def event102196 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27401⟩⟩) (.sum [.predecessor 0 102194 .coefficient, .predecessor 1 102195 .coefficient])

def event102197 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27401⟩⟩) (.sum [.result 102193 .summary, .result 100003 .summary])

def exact102198RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102198RawTermsValid :
    exact102198RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102198 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27401⟩⟩) exact102198RawTerms .large 102196 (.finite 7751615201839287181312) (some (102197))

def event102199 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27618⟩⟩) 0 ⟨27401⟩ 102198

def event102200 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27618⟩⟩) 1 ⟨27617⟩ 99569

def event102201 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27618⟩⟩) (.sum [.predecessor 0 102199 .coefficient, .predecessor 1 102200 .coefficient])

def event102202 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27618⟩⟩) (.sum [.result 102198 .summary, .result 99569 .summary])

def exact102203RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102203RawTermsValid :
    exact102203RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102203 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27618⟩⟩) exact102203RawTerms .large 102201 (.finite 9043661263333852925952) (some (102202))

def event102204 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27835⟩⟩) 0 ⟨27618⟩ 102203

def event102205 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27835⟩⟩) 1 ⟨27834⟩ 99135

def event102206 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27835⟩⟩) (.sum [.predecessor 0 102204 .coefficient, .predecessor 1 102205 .coefficient])

def event102207 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27835⟩⟩) (.sum [.result 102203 .summary, .result 99135 .summary])

def exact102208RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102208RawTermsValid :
    exact102208RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102208 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27835⟩⟩) exact102208RawTerms .large 102206 (.finite 10335729737273439256576) (some (102207))

def event102209 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28052⟩⟩) 0 ⟨27835⟩ 102208

def event102210 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28052⟩⟩) 1 ⟨28051⟩ 98701

def event102211 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28052⟩⟩) (.sum [.predecessor 0 102209 .coefficient, .predecessor 1 102210 .coefficient])

def event102212 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28052⟩⟩) (.sum [.result 102208 .summary, .result 98701 .summary])

def exact102213RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102213RawTermsValid :
    exact102213RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102213 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28052⟩⟩) exact102213RawTerms .large 102211 (.finite 11627843036103066759168) (some (102212))

def event102214 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28269⟩⟩) 0 ⟨28052⟩ 102213

def event102215 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28269⟩⟩) 1 ⟨28268⟩ 98267

def event102216 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28269⟩⟩) (.sum [.predecessor 0 102214 .coefficient, .predecessor 1 102215 .coefficient])

def event102217 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28269⟩⟩) (.sum [.result 102213 .summary, .result 98267 .summary])

def exact102218RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102218RawTermsValid :
    exact102218RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102218 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28269⟩⟩) exact102218RawTerms .large 102216 (.finite 12920023572267756019712) (some (102217))

def event102219 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28486⟩⟩) 0 ⟨28269⟩ 102218

def event102220 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28486⟩⟩) 1 ⟨28485⟩ 97833

def event102221 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28486⟩⟩) (.sum [.predecessor 0 102219 .coefficient, .predecessor 1 102220 .coefficient])

def event102222 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28486⟩⟩) (.sum [.result 102218 .summary, .result 97833 .summary])

def exact102223RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102223RawTermsValid :
    exact102223RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102223 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28486⟩⟩) exact102223RawTerms .large 102221 (.finite 14212226520877465866240) (some (102222))

def event102224 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28703⟩⟩) 0 ⟨28486⟩ 102223

def event102225 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28703⟩⟩) 1 ⟨28702⟩ 97399

def event102226 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28703⟩⟩) (.sum [.predecessor 0 102224 .coefficient, .predecessor 1 102225 .coefficient])

def event102227 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28703⟩⟩) (.sum [.result 102223 .summary, .result 97399 .summary])

def exact102228RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102228RawTermsValid :
    exact102228RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102228 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28703⟩⟩) exact102228RawTerms .large 102226 (.finite 15504496706822237470720) (some (102227))

def event102229 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28920⟩⟩) 0 ⟨28703⟩ 102228

def event102230 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28920⟩⟩) 1 ⟨28919⟩ 96965

def event102231 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28920⟩⟩) (.sum [.predecessor 0 102229 .coefficient, .predecessor 1 102230 .coefficient])

def event102232 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28920⟩⟩) (.sum [.result 102228 .summary, .result 96965 .summary])

def exact102233RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102233RawTermsValid :
    exact102233RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102233 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28920⟩⟩) exact102233RawTerms .large 102231 (.finite 16796811717657050247168) (some (102232))

def event102234 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29137⟩⟩) 0 ⟨28920⟩ 102233

def event102235 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29137⟩⟩) 1 ⟨29136⟩ 96531

def event102236 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29137⟩⟩) (.sum [.predecessor 0 102234 .coefficient, .predecessor 1 102235 .coefficient])

def event102237 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29137⟩⟩) (.sum [.result 102233 .summary, .result 96531 .summary])

def exact102238RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102238RawTermsValid :
    exact102238RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102238 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29137⟩⟩) exact102238RawTerms .large 102236 (.finite 18089149140936883609600) (some (102237))

def event102239 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29354⟩⟩) 0 ⟨29137⟩ 102238

def event102240 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29354⟩⟩) 1 ⟨29353⟩ 96097

def event102241 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29354⟩⟩) (.sum [.predecessor 0 102239 .coefficient, .predecessor 1 102240 .coefficient])

def event102242 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29354⟩⟩) (.sum [.result 102238 .summary, .result 96097 .summary])

def exact102243RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102243RawTermsValid :
    exact102243RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102243 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29354⟩⟩) exact102243RawTerms .large 102241 (.finite 19381531389106758144000) (some (102242))

def event102244 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29571⟩⟩) 0 ⟨29354⟩ 102243

def event102245 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29571⟩⟩) 1 ⟨29570⟩ 95663

def event102246 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29571⟩⟩) (.sum [.predecessor 0 102244 .coefficient, .predecessor 1 102245 .coefficient])

def event102247 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29571⟩⟩) (.sum [.result 102243 .summary, .result 95663 .summary])

def exact102248RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102248RawTermsValid :
    exact102248RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102248 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29571⟩⟩) exact102248RawTerms .large 102246 (.finite 20673980874611694436352) (some (102247))

def event102249 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29788⟩⟩) 0 ⟨29571⟩ 102248

def event102250 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29788⟩⟩) 1 ⟨29787⟩ 95229

def event102251 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29788⟩⟩) (.sum [.predecessor 0 102249 .coefficient, .predecessor 1 102250 .coefficient])

def event102252 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29788⟩⟩) (.sum [.result 102248 .summary, .result 95229 .summary])

def exact102253RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102253RawTermsValid :
    exact102253RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102253 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29788⟩⟩) exact102253RawTerms .large 102251 (.finite 21966497597451692486656) (some (102252))

def event102254 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30065⟩⟩) 0 ⟨29788⟩ 102253

def event102255 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30065⟩⟩) 1 ⟨30064⟩ 94795

def event102256 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30065⟩⟩) (.sum [.predecessor 0 102254 .coefficient, .predecessor 1 102255 .coefficient])

def event102257 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30065⟩⟩) (.sum [.result 102253 .summary, .result 94795 .summary])

def exact102258RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact102258RawTermsValid :
    exact102258RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102258 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30065⟩⟩) exact102258RawTerms .large 102256 (.finite 23259036732736711122944) (some (102257))

def event102259 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30066⟩⟩) 0 ⟨30065⟩ 102258

def event102260 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30066⟩⟩) 1 ⟨18674⟩ 94350

def event102261 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30066⟩⟩) (.product (.predecessor 0 102259 .coefficient) (.predecessor 1 102260 .coefficient) (⟨false, false, none, none, none⟩))

def event102262 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30066⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) [⟨.result 94350 .coefficient, false, none⟩])

def event102263 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30066⟩⟩) (.product (.result 102258 .summary) (.transfer 102262) (⟨false, false, none, none, none⟩))

def event102264 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 17⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event102265 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 33⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event102266 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30066⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 94347)

def event102267 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .relation 102266 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event102268 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 16⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event102269 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 29⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event102270 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30066⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 94347)

def event102271 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .relation 102270 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event102272 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 15⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event102273 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 28⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event102274 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30066⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 94347)

def event102275 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .relation 102274 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event102276 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 14⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event102277 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 27⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event102278 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30066⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 94347)

def event102279 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .relation 102278 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event102280 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 13⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event102281 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 34⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event102282 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30066⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 94347)

def event102283 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .relation 102282 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event102284 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 12⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event102285 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 32⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event102286 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30066⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 94347)

def event102287 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .relation 102286 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event102288 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 11⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event102289 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 30⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event102290 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30066⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 94347)

def event102291 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .relation 102290 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event102292 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 10⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event102293 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 26⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event102294 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30066⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 94347)

def event102295 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .relation 102294 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event102296 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 9⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event102297 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 35⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event102298 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30066⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 94347)

def event102299 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .relation 102298 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event102300 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 8⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event102301 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 25⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event102302 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30066⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 94347)

def event102303 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .relation 102302 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event102304 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 7⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event102305 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 24⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event102306 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30066⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 94347)

def event102307 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .relation 102306 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event102308 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 6⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event102309 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 23⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event102310 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30066⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 94347)

def event102311 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .relation 102310 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event102312 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 5⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event102313 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 22⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event102314 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30066⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 94347)

def event102315 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .relation 102314 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event102316 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 4⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event102317 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 21⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event102318 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30066⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 94347)

def event102319 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .relation 102318 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event102320 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 3⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event102321 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 31⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event102322 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30066⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 94347)

def event102323 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .relation 102322 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event102324 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 2⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event102325 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 20⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event102326 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30066⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 94347)

def event102327 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .relation 102326 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event102328 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 1⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event102329 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 19⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event102330 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30066⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 94347)

def event102331 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .relation 102330 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def event102332 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 0⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩)

def event102333 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .operator (⟨102258, 18⟩, ⟨94350, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (-1)⟩)

def event102334 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30066⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18674⟩⟩) ⟨18612⟩ 94347)

def event102335 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30066⟩⟩, .relation 102334 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩)

def exact102336RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15300⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15622⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17302⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩, (-1)⟩]

theorem exact102336RawTermsValid :
    exact102336RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102336 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30066⟩⟩) exact102336RawTerms .large 102261 (.finite 85361036953731453608582447104) (some (102263))

def event102337 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18548⟩⟩) 0 ⟨18313⟩ 5048

def event102338 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18548⟩⟩) (.authority (.relationPreimageSource ⟨67⟩))

def exact102339RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩, (1)⟩]

theorem exact102339RawTermsValid :
    exact102339RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102339 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18548⟩⟩) exact102339RawTerms (.finite 136065468) 102338 .exactZero (none)

def event102340 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18550⟩⟩) 0 ⟨18548⟩ 102339

def event102341 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18550⟩⟩) 1 ⟨2348⟩ 4

def event102342 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18550⟩⟩) (.scale (.predecessor 0 102340 .coefficient) (.value (.predecessor 1 102341 .coefficient)))

def exact102343RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩, (1)⟩]

theorem exact102343RawTermsValid :
    exact102343RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102343 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18550⟩⟩) exact102343RawTerms (.finite 136065468) 102342 .exactZero (none)

def event102344 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18551⟩⟩) 0 ⟨5509⟩ 94462

def event102345 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18551⟩⟩) 1 ⟨18550⟩ 102343

def event102346 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18551⟩⟩) (.product (.predecessor 0 102344 .coefficient) (.predecessor 1 102345 .coefficient) (⟨false, false, none, none, none⟩))

def event102347 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18551⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) [⟨.result 102339 .coefficient, false, none⟩])

def event102348 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18551⟩⟩) (.product (.result 94462 .summary) (.transfer 102347) (⟨false, false, none, none, none⟩))

def event102349 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18551⟩⟩, .operator (⟨94462, 0⟩, ⟨102343, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩, (1)⟩)

def event102350 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨18549⟩⟩)

def event102351 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event102352 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event102353 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event102354 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event102355 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 102354

def event102356 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 102352

def event102357 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 102355 .coefficient) (.value (.predecessor 1 102356 .coefficient)))

def event102358 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event102359 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13326⟩⟩) 0 ⟨5503⟩ 102358

def event102360 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13326⟩⟩) (.authority (.programFamilyFact))

def exact102361RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13326⟩⟩], []⟩, (1)⟩]

theorem exact102361RawTermsValid :
    exact102361RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102361 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13326⟩⟩) exact102361RawTerms (.finite 60) 102360 .exactZero (none)

def event102362 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10330⟩⟩) 0 ⟨5503⟩ 102358

def event102363 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10330⟩⟩) (.authority (.programFamilyFact))

def exact102364RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10330⟩⟩], []⟩, (1)⟩]

theorem exact102364RawTermsValid :
    exact102364RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102364 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10330⟩⟩) exact102364RawTerms (.finite 60) 102363 .exactZero (none)

def event102365 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13327⟩⟩) 0 ⟨10330⟩ 102364

def event102366 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13327⟩⟩) 1 ⟨13326⟩ 102361

def event102367 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13327⟩⟩) (.product (.predecessor 0 102365 .coefficient) (.predecessor 1 102366 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event102368 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13327⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10330⟩⟩, ⟨.program ⟨214⟩, ⟨13326⟩⟩], []⟩) [⟨.result 102364 .coefficient, true, some 1⟩, ⟨.result 102361 .coefficient, true, some 1⟩])

def event102369 : Event := .survivorFold (1) 102368

def exact102370RawTerms : List Term := []

theorem exact102370RawTermsValid :
    exact102370RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102370 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13327⟩⟩) exact102370RawTerms (.finite 3600) 102367 (.finite 3600) (some (102368))

def event102371 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13328⟩⟩) 0 ⟨13327⟩ 102370

def event102372 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13328⟩⟩) (.identity (.predecessor 0 102371 .coefficient))

def event102373 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13328⟩⟩) (.finite 3600)

def event102374 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17001⟩⟩) 0 ⟨13328⟩ 102373

def event102375 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17001⟩⟩) (.authority (.programFamilyFact))

def exact102376RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], []⟩, (1)⟩]

theorem exact102376RawTermsValid :
    exact102376RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102376 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17001⟩⟩) exact102376RawTerms (.finite 60) 102375 .exactZero (none)

def event102377 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17002⟩⟩) 0 ⟨17001⟩ 102376

def event102378 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17002⟩⟩) (.identity (.predecessor 0 102377 .coefficient))

def event102379 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨17002⟩⟩) (.finite 60)

def event102380 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18163⟩⟩) 0 ⟨17002⟩ 102379

def event102381 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18163⟩⟩) (.authority (.programFamilyFact))

def exact102382RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], []⟩, (1)⟩]

theorem exact102382RawTermsValid :
    exact102382RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102382 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18163⟩⟩) exact102382RawTerms (.finite 63) 102381 .exactZero (none)

def event102383 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13130⟩⟩) 0 ⟨5503⟩ 102358

def event102384 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13130⟩⟩) (.authority (.programFamilyFact))

def exact102385RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13130⟩⟩], []⟩, (1)⟩]

theorem exact102385RawTermsValid :
    exact102385RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102385 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13130⟩⟩) exact102385RawTerms (.finite 58) 102384 .exactZero (none)

def event102386 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10225⟩⟩) 0 ⟨5503⟩ 102358

def event102387 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10225⟩⟩) (.authority (.programFamilyFact))

def exact102388RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10225⟩⟩], []⟩, (1)⟩]

theorem exact102388RawTermsValid :
    exact102388RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102388 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10225⟩⟩) exact102388RawTerms (.finite 58) 102387 .exactZero (none)

def event102389 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13131⟩⟩) 0 ⟨10225⟩ 102388

def event102390 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13131⟩⟩) 1 ⟨13130⟩ 102385

def event102391 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13131⟩⟩) (.product (.predecessor 0 102389 .coefficient) (.predecessor 1 102390 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event102392 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13131⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10225⟩⟩, ⟨.program ⟨214⟩, ⟨13130⟩⟩], []⟩) [⟨.result 102388 .coefficient, true, some 1⟩, ⟨.result 102385 .coefficient, true, some 1⟩])

def event102393 : Event := .survivorFold (1) 102392

def exact102394RawTerms : List Term := []

theorem exact102394RawTermsValid :
    exact102394RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event102394 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13131⟩⟩) exact102394RawTerms (.finite 3364) 102391 (.finite 3364) (some (102392))

def event102395 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13132⟩⟩) 0 ⟨13131⟩ 102394

def event102396 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13132⟩⟩) (.identity (.predecessor 0 102395 .coefficient))

def event102397 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13132⟩⟩) (.finite 3364)

def event102398 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16861⟩⟩) 0 ⟨13132⟩ 102397

def event102399 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16861⟩⟩) (.authority (.programFamilyFact))

def eventLeaf6384 : Array AnnotatedEvent := #[
  { event := event102144
    frameStart := 102068 },
  { event := event102145
    frameStart := 102068 },
  { event := event102146
    frameStart := 102068 },
  { event := event102147
    frameStart := 102068 },
  { event := event102148
    frameStart := 102068 },
  { event := event102149
    frameStart := 102068 },
  { event := event102150
    frameStart := 102068 },
  { event := event102151
    frameStart := 102068 },
  { event := event102152
    frameStart := 102068 },
  { event := event102153
    frameStart := 102068 },
  { event := event102154
    frameStart := 102068 },
  { event := event102155
    frameStart := 102068 },
  { event := event102156
    frameStart := 102068 },
  { event := event102157
    frameStart := 102068 },
  { event := event102158
    frameStart := 102068 },
  { event := event102159
    frameStart := 102068 }
]

def eventLeaf6385 : Array AnnotatedEvent := #[
  { event := event102160
    frameStart := 0 },
  { event := event102161
    frameStart := 0 },
  { event := event102162
    frameStart := 0 },
  { event := event102163
    frameStart := 0 },
  { event := event102164
    frameStart := 0 },
  { event := event102165
    frameStart := 0 },
  { event := event102166
    frameStart := 0 },
  { event := event102167
    frameStart := 0 },
  { event := event102168
    frameStart := 0 },
  { event := event102169
    frameStart := 0 },
  { event := event102170
    frameStart := 0 },
  { event := event102171
    frameStart := 0 },
  { event := event102172
    frameStart := 0 },
  { event := event102173
    frameStart := 0 },
  { event := event102174
    frameStart := 0 },
  { event := event102175
    frameStart := 0 }
]

def eventLeaf6386 : Array AnnotatedEvent := #[
  { event := event102176
    frameStart := 0 },
  { event := event102177
    frameStart := 0 },
  { event := event102178
    frameStart := 0 },
  { event := event102179
    frameStart := 0 },
  { event := event102180
    frameStart := 0 },
  { event := event102181
    frameStart := 0 },
  { event := event102182
    frameStart := 0 },
  { event := event102183
    frameStart := 0 },
  { event := event102184
    frameStart := 0 },
  { event := event102185
    frameStart := 0 },
  { event := event102186
    frameStart := 0 },
  { event := event102187
    frameStart := 0 },
  { event := event102188
    frameStart := 0 },
  { event := event102189
    frameStart := 0 },
  { event := event102190
    frameStart := 0 },
  { event := event102191
    frameStart := 0 }
]

def eventLeaf6387 : Array AnnotatedEvent := #[
  { event := event102192
    frameStart := 0 },
  { event := event102193
    frameStart := 0 },
  { event := event102194
    frameStart := 0 },
  { event := event102195
    frameStart := 0 },
  { event := event102196
    frameStart := 0 },
  { event := event102197
    frameStart := 0 },
  { event := event102198
    frameStart := 0 },
  { event := event102199
    frameStart := 0 },
  { event := event102200
    frameStart := 0 },
  { event := event102201
    frameStart := 0 },
  { event := event102202
    frameStart := 0 },
  { event := event102203
    frameStart := 0 },
  { event := event102204
    frameStart := 0 },
  { event := event102205
    frameStart := 0 },
  { event := event102206
    frameStart := 0 },
  { event := event102207
    frameStart := 0 }
]

def eventLeaf6388 : Array AnnotatedEvent := #[
  { event := event102208
    frameStart := 0 },
  { event := event102209
    frameStart := 0 },
  { event := event102210
    frameStart := 0 },
  { event := event102211
    frameStart := 0 },
  { event := event102212
    frameStart := 0 },
  { event := event102213
    frameStart := 0 },
  { event := event102214
    frameStart := 0 },
  { event := event102215
    frameStart := 0 },
  { event := event102216
    frameStart := 0 },
  { event := event102217
    frameStart := 0 },
  { event := event102218
    frameStart := 0 },
  { event := event102219
    frameStart := 0 },
  { event := event102220
    frameStart := 0 },
  { event := event102221
    frameStart := 0 },
  { event := event102222
    frameStart := 0 },
  { event := event102223
    frameStart := 0 }
]

def eventLeaf6389 : Array AnnotatedEvent := #[
  { event := event102224
    frameStart := 0 },
  { event := event102225
    frameStart := 0 },
  { event := event102226
    frameStart := 0 },
  { event := event102227
    frameStart := 0 },
  { event := event102228
    frameStart := 0 },
  { event := event102229
    frameStart := 0 },
  { event := event102230
    frameStart := 0 },
  { event := event102231
    frameStart := 0 },
  { event := event102232
    frameStart := 0 },
  { event := event102233
    frameStart := 0 },
  { event := event102234
    frameStart := 0 },
  { event := event102235
    frameStart := 0 },
  { event := event102236
    frameStart := 0 },
  { event := event102237
    frameStart := 0 },
  { event := event102238
    frameStart := 0 },
  { event := event102239
    frameStart := 0 }
]

def eventLeaf6390 : Array AnnotatedEvent := #[
  { event := event102240
    frameStart := 0 },
  { event := event102241
    frameStart := 0 },
  { event := event102242
    frameStart := 0 },
  { event := event102243
    frameStart := 0 },
  { event := event102244
    frameStart := 0 },
  { event := event102245
    frameStart := 0 },
  { event := event102246
    frameStart := 0 },
  { event := event102247
    frameStart := 0 },
  { event := event102248
    frameStart := 0 },
  { event := event102249
    frameStart := 0 },
  { event := event102250
    frameStart := 0 },
  { event := event102251
    frameStart := 0 },
  { event := event102252
    frameStart := 0 },
  { event := event102253
    frameStart := 0 },
  { event := event102254
    frameStart := 0 },
  { event := event102255
    frameStart := 0 }
]

def eventLeaf6391 : Array AnnotatedEvent := #[
  { event := event102256
    frameStart := 0 },
  { event := event102257
    frameStart := 0 },
  { event := event102258
    frameStart := 0 },
  { event := event102259
    frameStart := 0 },
  { event := event102260
    frameStart := 0 },
  { event := event102261
    frameStart := 0 },
  { event := event102262
    frameStart := 0 },
  { event := event102263
    frameStart := 0 },
  { event := event102264
    frameStart := 0 },
  { event := event102265
    frameStart := 0 },
  { event := event102266
    frameStart := 0 },
  { event := event102267
    frameStart := 0 },
  { event := event102268
    frameStart := 0 },
  { event := event102269
    frameStart := 0 },
  { event := event102270
    frameStart := 0 },
  { event := event102271
    frameStart := 0 }
]

def eventLeaf6392 : Array AnnotatedEvent := #[
  { event := event102272
    frameStart := 0 },
  { event := event102273
    frameStart := 0 },
  { event := event102274
    frameStart := 0 },
  { event := event102275
    frameStart := 0 },
  { event := event102276
    frameStart := 0 },
  { event := event102277
    frameStart := 0 },
  { event := event102278
    frameStart := 0 },
  { event := event102279
    frameStart := 0 },
  { event := event102280
    frameStart := 0 },
  { event := event102281
    frameStart := 0 },
  { event := event102282
    frameStart := 0 },
  { event := event102283
    frameStart := 0 },
  { event := event102284
    frameStart := 0 },
  { event := event102285
    frameStart := 0 },
  { event := event102286
    frameStart := 0 },
  { event := event102287
    frameStart := 0 }
]

def eventLeaf6393 : Array AnnotatedEvent := #[
  { event := event102288
    frameStart := 0 },
  { event := event102289
    frameStart := 0 },
  { event := event102290
    frameStart := 0 },
  { event := event102291
    frameStart := 0 },
  { event := event102292
    frameStart := 0 },
  { event := event102293
    frameStart := 0 },
  { event := event102294
    frameStart := 0 },
  { event := event102295
    frameStart := 0 },
  { event := event102296
    frameStart := 0 },
  { event := event102297
    frameStart := 0 },
  { event := event102298
    frameStart := 0 },
  { event := event102299
    frameStart := 0 },
  { event := event102300
    frameStart := 0 },
  { event := event102301
    frameStart := 0 },
  { event := event102302
    frameStart := 0 },
  { event := event102303
    frameStart := 0 }
]

def eventLeaf6394 : Array AnnotatedEvent := #[
  { event := event102304
    frameStart := 0 },
  { event := event102305
    frameStart := 0 },
  { event := event102306
    frameStart := 0 },
  { event := event102307
    frameStart := 0 },
  { event := event102308
    frameStart := 0 },
  { event := event102309
    frameStart := 0 },
  { event := event102310
    frameStart := 0 },
  { event := event102311
    frameStart := 0 },
  { event := event102312
    frameStart := 0 },
  { event := event102313
    frameStart := 0 },
  { event := event102314
    frameStart := 0 },
  { event := event102315
    frameStart := 0 },
  { event := event102316
    frameStart := 0 },
  { event := event102317
    frameStart := 0 },
  { event := event102318
    frameStart := 0 },
  { event := event102319
    frameStart := 0 }
]

def eventLeaf6395 : Array AnnotatedEvent := #[
  { event := event102320
    frameStart := 0 },
  { event := event102321
    frameStart := 0 },
  { event := event102322
    frameStart := 0 },
  { event := event102323
    frameStart := 0 },
  { event := event102324
    frameStart := 0 },
  { event := event102325
    frameStart := 0 },
  { event := event102326
    frameStart := 0 },
  { event := event102327
    frameStart := 0 },
  { event := event102328
    frameStart := 0 },
  { event := event102329
    frameStart := 0 },
  { event := event102330
    frameStart := 0 },
  { event := event102331
    frameStart := 0 },
  { event := event102332
    frameStart := 0 },
  { event := event102333
    frameStart := 0 },
  { event := event102334
    frameStart := 0 },
  { event := event102335
    frameStart := 0 }
]

def eventLeaf6396 : Array AnnotatedEvent := #[
  { event := event102336
    frameStart := 0 },
  { event := event102337
    frameStart := 0 },
  { event := event102338
    frameStart := 0 },
  { event := event102339
    frameStart := 0 },
  { event := event102340
    frameStart := 0 },
  { event := event102341
    frameStart := 0 },
  { event := event102342
    frameStart := 0 },
  { event := event102343
    frameStart := 0 },
  { event := event102344
    frameStart := 0 },
  { event := event102345
    frameStart := 0 },
  { event := event102346
    frameStart := 0 },
  { event := event102347
    frameStart := 0 },
  { event := event102348
    frameStart := 0 },
  { event := event102349
    frameStart := 0 },
  { event := event102350
    frameStart := 102350 },
  { event := event102351
    frameStart := 102350 }
]

def eventLeaf6397 : Array AnnotatedEvent := #[
  { event := event102352
    frameStart := 102350 },
  { event := event102353
    frameStart := 102350 },
  { event := event102354
    frameStart := 102350 },
  { event := event102355
    frameStart := 102350 },
  { event := event102356
    frameStart := 102350 },
  { event := event102357
    frameStart := 102350 },
  { event := event102358
    frameStart := 102350 },
  { event := event102359
    frameStart := 102350 },
  { event := event102360
    frameStart := 102350 },
  { event := event102361
    frameStart := 102350 },
  { event := event102362
    frameStart := 102350 },
  { event := event102363
    frameStart := 102350 },
  { event := event102364
    frameStart := 102350 },
  { event := event102365
    frameStart := 102350 },
  { event := event102366
    frameStart := 102350 },
  { event := event102367
    frameStart := 102350 }
]

def eventLeaf6398 : Array AnnotatedEvent := #[
  { event := event102368
    frameStart := 102350 },
  { event := event102369
    frameStart := 102350 },
  { event := event102370
    frameStart := 102350 },
  { event := event102371
    frameStart := 102350 },
  { event := event102372
    frameStart := 102350 },
  { event := event102373
    frameStart := 102350 },
  { event := event102374
    frameStart := 102350 },
  { event := event102375
    frameStart := 102350 },
  { event := event102376
    frameStart := 102350 },
  { event := event102377
    frameStart := 102350 },
  { event := event102378
    frameStart := 102350 },
  { event := event102379
    frameStart := 102350 },
  { event := event102380
    frameStart := 102350 },
  { event := event102381
    frameStart := 102350 },
  { event := event102382
    frameStart := 102350 },
  { event := event102383
    frameStart := 102350 }
]

def eventLeaf6399 : Array AnnotatedEvent := #[
  { event := event102384
    frameStart := 102350 },
  { event := event102385
    frameStart := 102350 },
  { event := event102386
    frameStart := 102350 },
  { event := event102387
    frameStart := 102350 },
  { event := event102388
    frameStart := 102350 },
  { event := event102389
    frameStart := 102350 },
  { event := event102390
    frameStart := 102350 },
  { event := event102391
    frameStart := 102350 },
  { event := event102392
    frameStart := 102350 },
  { event := event102393
    frameStart := 102350 },
  { event := event102394
    frameStart := 102350 },
  { event := event102395
    frameStart := 102350 },
  { event := event102396
    frameStart := 102350 },
  { event := event102397
    frameStart := 102350 },
  { event := event102398
    frameStart := 102350 },
  { event := event102399
    frameStart := 102350 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events399
