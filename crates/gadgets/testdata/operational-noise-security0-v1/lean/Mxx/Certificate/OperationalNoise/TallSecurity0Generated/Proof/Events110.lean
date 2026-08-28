import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events110

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event28160 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23917⟩⟩) 0 ⟨15435⟩ 1181

def event28161 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23917⟩⟩) (.authority (.programFamilyFact))

def event28162 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23917⟩⟩) (.finite 3720)

def event28163 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23919⟩⟩) 0 ⟨6689⟩ 5477

def event28164 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23919⟩⟩) 1 ⟨23917⟩ 28162

def event28165 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23919⟩⟩) (.authority (.operator))

def exact28166RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23919⟩⟩]⟩, (1)⟩]

theorem exact28166RawTermsValid :
    exact28166RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28166 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23919⟩⟩) exact28166RawTerms .large 28165 .exactZero (none)

def event28167 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27037⟩⟩) 0 ⟨23919⟩ 28166

def event28168 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27037⟩⟩) (.authority (.operator))

def exact28169RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩, (1)⟩]

theorem exact28169RawTermsValid :
    exact28169RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28169 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27037⟩⟩) exact28169RawTerms (.finite 8192) 28168 .exactZero (none)

def event28170 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23169⟩⟩) 0 ⟨12192⟩ 1175

def event28171 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23169⟩⟩) (.authority (.programFamilyFact))

def event28172 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23169⟩⟩) (.finite 3720)

def event28173 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23170⟩⟩) 0 ⟨6689⟩ 5477

def event28174 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23170⟩⟩) 1 ⟨23169⟩ 28172

def event28175 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23170⟩⟩) (.authority (.operator))

def exact28176RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23170⟩⟩]⟩, (1)⟩]

theorem exact28176RawTermsValid :
    exact28176RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28176 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23170⟩⟩) exact28176RawTerms .large 28175 .exactZero (none)

def event28177 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25311⟩⟩) 0 ⟨23170⟩ 28176

def event28178 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25311⟩⟩) (.authority (.operator))

def exact28179RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩, (1)⟩]

theorem exact28179RawTermsValid :
    exact28179RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28179 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25311⟩⟩) exact28179RawTerms (.finite 8192) 28178 .exactZero (none)

def event28180 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11146⟩⟩) 0 ⟨11145⟩ 1164

def event28181 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11146⟩⟩) 1 ⟨6570⟩ 21420

def event28182 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11146⟩⟩) (.tensor (.predecessor 0 28180 .coefficient) (.predecessor 1 28181 .coefficient) true false)

def event28183 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11146⟩⟩, .operator (⟨1164, 0⟩, ⟨21420, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact28184RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact28184RawTermsValid :
    exact28184RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28184 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11146⟩⟩) exact28184RawTerms .large 28182 .exactZero (none)

def event28185 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7345⟩⟩) 0 ⟨5557⟩ 21290

def event28186 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7345⟩⟩) 1 ⟨6775⟩ 13486

def event28187 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7345⟩⟩) (.product (.predecessor 0 28185 .coefficient) (.predecessor 1 28186 .coefficient) (⟨false, false, none, none, none⟩))

def event28188 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7345⟩⟩, .operator (⟨21290, 0⟩, ⟨13486, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩)

def exact28189RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩]

theorem exact28189RawTermsValid :
    exact28189RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28189 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7345⟩⟩) exact28189RawTerms .large 28187 .exactZero (none)

def event28190 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11147⟩⟩) 0 ⟨7345⟩ 28189

def event28191 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11147⟩⟩) 1 ⟨11146⟩ 28184

def event28192 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11147⟩⟩) (.sum [.predecessor 0 28190 .coefficient, .predecessor 1 28191 .coefficient])

def exact28193RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28193RawTermsValid :
    exact28193RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28193 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11147⟩⟩) exact28193RawTerms .large 28192 .exactZero (none)

def event28194 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11148⟩⟩) 0 ⟨11147⟩ 28193

def event28195 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11148⟩⟩) 1 ⟨89⟩ 13478

def event28196 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11148⟩⟩) (.sum [.predecessor 0 28194 .coefficient, .predecessor 1 28195 .coefficient])

def event28197 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11148⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨89⟩⟩]⟩) [⟨.result 13478 .coefficient, false, none⟩])

def event28198 : Event := .survivorFold (1) 28197

def exact28199RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28199RawTermsValid :
    exact28199RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28199 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11148⟩⟩) exact28199RawTerms .large 28196 (.finite 26) (some (28197))

def event28200 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12193⟩⟩) 0 ⟨11148⟩ 28199

def event28201 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12193⟩⟩) 1 ⟨12190⟩ 1167

def event28202 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12193⟩⟩) (.product (.predecessor 0 28200 .coefficient) (.predecessor 1 28201 .coefficient) (⟨false, true, none, none, some 1⟩))

def event28203 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12193⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨12190⟩⟩], []⟩) [⟨.result 1167 .coefficient, true, some 1⟩])

def event28204 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12193⟩⟩) (.product (.result 28199 .summary) (.transfer 28203) (⟨false, false, none, none, none⟩))

def event28205 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12193⟩⟩, .operator (⟨28199, 1⟩, ⟨1167, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event28206 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12193⟩⟩, .operator (⟨28199, 0⟩, ⟨1167, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩)

def exact28207RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩]

theorem exact28207RawTermsValid :
    exact28207RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28207 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12193⟩⟩) exact28207RawTerms .large 28202 (.finite 4992) (some (28204))

def event28208 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12194⟩⟩) 0 ⟨12190⟩ 1167

def event28209 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12194⟩⟩) 1 ⟨6570⟩ 21420

def event28210 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12194⟩⟩) (.tensor (.predecessor 0 28208 .coefficient) (.predecessor 1 28209 .coefficient) true false)

def event28211 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12194⟩⟩, .operator (⟨1167, 0⟩, ⟨21420, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact28212RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact28212RawTermsValid :
    exact28212RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28212 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12194⟩⟩) exact28212RawTerms .large 28210 .exactZero (none)

def event28213 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7362⟩⟩) 0 ⟨5557⟩ 21290

def event28214 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7362⟩⟩) 1 ⟨6792⟩ 13527

def event28215 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7362⟩⟩) (.product (.predecessor 0 28213 .coefficient) (.predecessor 1 28214 .coefficient) (⟨false, false, none, none, none⟩))

def event28216 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7362⟩⟩, .operator (⟨21290, 0⟩, ⟨13527, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩]⟩, (1)⟩)

def exact28217RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩]⟩, (1)⟩]

theorem exact28217RawTermsValid :
    exact28217RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28217 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7362⟩⟩) exact28217RawTerms .large 28215 .exactZero (none)

def event28218 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12195⟩⟩) 0 ⟨7362⟩ 28217

def event28219 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12195⟩⟩) 1 ⟨12194⟩ 28212

def event28220 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12195⟩⟩) (.sum [.predecessor 0 28218 .coefficient, .predecessor 1 28219 .coefficient])

def exact28221RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28221RawTermsValid :
    exact28221RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28221 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12195⟩⟩) exact28221RawTerms .large 28220 .exactZero (none)

def event28222 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12196⟩⟩) 0 ⟨12195⟩ 28221

def event28223 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12196⟩⟩) 1 ⟨106⟩ 13519

def event28224 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12196⟩⟩) (.sum [.predecessor 0 28222 .coefficient, .predecessor 1 28223 .coefficient])

def event28225 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12196⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨106⟩⟩]⟩) [⟨.result 13519 .coefficient, false, none⟩])

def event28226 : Event := .survivorFold (1) 28225

def exact28227RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28227RawTermsValid :
    exact28227RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28227 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12196⟩⟩) exact28227RawTerms .large 28224 (.finite 26) (some (28225))

def event28228 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12197⟩⟩) 0 ⟨12196⟩ 28227

def event28229 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12197⟩⟩) 1 ⟨7841⟩ 13516

def event28230 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12197⟩⟩) (.product (.predecessor 0 28228 .coefficient) (.predecessor 1 28229 .coefficient) (⟨false, false, none, none, none⟩))

def event28231 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12197⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) [⟨.result 13512 .coefficient, false, none⟩])

def event28232 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12197⟩⟩) (.product (.result 28227 .summary) (.transfer 28231) (⟨false, false, none, none, none⟩))

def event28233 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12197⟩⟩, .operator (⟨28227, 1⟩, ⟨13516, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (-1)⟩)

def event28234 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨12197⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7840⟩⟩) ⟨6775⟩ 13486)

def event28235 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12197⟩⟩, .relation 28234 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (-1)⟩)

def event28236 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12197⟩⟩, .operator (⟨28227, 0⟩, ⟨13516, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩)

def exact28237RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (-1)⟩]

theorem exact28237RawTermsValid :
    exact28237RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28237 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12197⟩⟩) exact28237RawTerms .large 28230 (.finite 95420416) (some (28232))

def event28238 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12198⟩⟩) 0 ⟨12197⟩ 28237

def event28239 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12198⟩⟩) 1 ⟨12193⟩ 28207

def event28240 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12198⟩⟩) (.sum [.predecessor 0 28238 .coefficient, .predecessor 1 28239 .coefficient])

def event28241 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12198⟩⟩, .operator (⟨28237, 1⟩, ⟨28207, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩)

def event28242 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12198⟩⟩) (.sum [.result 28237 .summary, .result 28207 .summary])

def exact28243RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28243RawTermsValid :
    exact28243RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28243 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12198⟩⟩) exact28243RawTerms .large 28240 (.finite 95425408) (some (28242))

def event28244 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25312⟩⟩) 0 ⟨12198⟩ 28243

def event28245 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25312⟩⟩) 1 ⟨25311⟩ 28179

def event28246 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25312⟩⟩) (.product (.predecessor 0 28244 .coefficient) (.predecessor 1 28245 .coefficient) (⟨false, false, none, none, none⟩))

def event28247 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25312⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩) [⟨.result 28179 .coefficient, false, none⟩])

def event28248 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25312⟩⟩) (.product (.result 28243 .summary) (.transfer 28247) (⟨false, false, none, none, none⟩))

def event28249 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25312⟩⟩, .operator (⟨28243, 1⟩, ⟨28179, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩, (-1)⟩)

def event28250 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25312⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25311⟩⟩) ⟨23170⟩ 28176)

def event28251 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25312⟩⟩, .relation 28250 0, ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨23170⟩⟩]⟩, (-1)⟩)

def event28252 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25312⟩⟩, .operator (⟨28243, 0⟩, ⟨28179, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩, (1)⟩)

def exact28253RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨23170⟩⟩]⟩, (-1)⟩]

theorem exact28253RawTermsValid :
    exact28253RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28253 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25312⟩⟩) exact28253RawTerms .large 28246 (.finite 350212774166528) (some (28248))

def event28254 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19252⟩⟩) 0 ⟨12192⟩ 1175

def event28255 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19252⟩⟩) (.authority (.relationPreimageSource ⟨10⟩))

def exact28256RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩, (1)⟩]

theorem exact28256RawTermsValid :
    exact28256RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28256 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19252⟩⟩) exact28256RawTerms (.finite 136065468) 28255 .exactZero (none)

def event28257 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19254⟩⟩) 0 ⟨19252⟩ 28256

def event28258 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19254⟩⟩) 1 ⟨2348⟩ 4

def event28259 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19254⟩⟩) (.scale (.predecessor 0 28257 .coefficient) (.value (.predecessor 1 28258 .coefficient)))

def exact28260RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩, (1)⟩]

theorem exact28260RawTermsValid :
    exact28260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28260 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19254⟩⟩) exact28260RawTerms (.finite 136065468) 28259 .exactZero (none)

def event28261 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19255⟩⟩) 0 ⟨5559⟩ 21512

def event28262 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19255⟩⟩) 1 ⟨19254⟩ 28260

def event28263 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19255⟩⟩) (.product (.predecessor 0 28261 .coefficient) (.predecessor 1 28262 .coefficient) (⟨false, false, none, none, none⟩))

def event28264 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19255⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩) [⟨.result 28256 .coefficient, false, none⟩])

def event28265 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19255⟩⟩) (.product (.result 21512 .summary) (.transfer 28264) (⟨false, false, none, none, none⟩))

def event28266 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19255⟩⟩, .operator (⟨21512, 0⟩, ⟨28260, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩, (1)⟩)

def event28267 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19253⟩⟩)

def event28268 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event28269 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event28270 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event28271 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event28272 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event28273 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event28274 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event28275 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event28276 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 28275

def event28277 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 28273

def event28278 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 28276 .coefficient) (.value (.predecessor 1 28277 .coefficient)))

def event28279 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event28280 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 28279

def event28281 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 28271

def event28282 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 28280 .coefficient, .predecessor 1 28281 .coefficient])

def event28283 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event28284 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 28283

def event28285 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 28269

def event28286 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 28285 .coefficient))

def event28287 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event28288 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11145⟩⟩) 0 ⟨5554⟩ 28287

def event28289 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11145⟩⟩) (.authority (.programFamilyFact))

def exact28290RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩], []⟩, (1)⟩]

theorem exact28290RawTermsValid :
    exact28290RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28290 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11145⟩⟩) exact28290RawTerms (.finite 6) 28289 .exactZero (none)

def event28291 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12190⟩⟩) 0 ⟨5554⟩ 28287

def event28292 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12190⟩⟩) (.authority (.programFamilyFact))

def exact28293RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12190⟩⟩], []⟩, (1)⟩]

theorem exact28293RawTermsValid :
    exact28293RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28293 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12190⟩⟩) exact28293RawTerms (.finite 6) 28292 .exactZero (none)

def event28294 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12191⟩⟩) 0 ⟨12190⟩ 28293

def event28295 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12191⟩⟩) 1 ⟨11145⟩ 28290

def event28296 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12191⟩⟩) (.product (.predecessor 0 28294 .coefficient) (.predecessor 1 28295 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event28297 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12191⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], []⟩) [⟨.result 28293 .coefficient, true, some 1⟩, ⟨.result 28290 .coefficient, true, some 1⟩])

def event28298 : Event := .survivorFold (1) 28297

def exact28299RawTerms : List Term := []

theorem exact28299RawTermsValid :
    exact28299RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28299 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12191⟩⟩) exact28299RawTerms (.finite 36) 28296 (.finite 36) (some (28297))

def event28300 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12192⟩⟩) 0 ⟨12191⟩ 28299

def event28301 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12192⟩⟩) (.identity (.predecessor 0 28300 .coefficient))

def event28302 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12192⟩⟩) (.finite 36)

def event28303 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19252⟩⟩) 0 ⟨12192⟩ 28302

def event28304 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19252⟩⟩) (.authority (.relationPreimageSource ⟨10⟩))

def exact28305RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩, (1)⟩]

theorem exact28305RawTermsValid :
    exact28305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28305 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19252⟩⟩) exact28305RawTerms (.finite 136065468) 28304 .exactZero (none)

def event28306 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact28307RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact28307RawTermsValid :
    exact28307RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28307 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact28307RawTerms .large 28306 .exactZero (none)

def event28308 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19253⟩⟩) 0 ⟨6⟩ 28307

def event28309 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19253⟩⟩) 1 ⟨19252⟩ 28305

def event28310 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19253⟩⟩) (.product (.predecessor 0 28308 .coefficient) (.predecessor 1 28309 .coefficient) (⟨false, false, none, none, none⟩))

def event28311 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19253⟩⟩, .operator (⟨28307, 0⟩, ⟨28305, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩, (1)⟩)

def exact28312RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩, (1)⟩]

theorem exact28312RawTermsValid :
    exact28312RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28312 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19253⟩⟩) exact28312RawTerms .large 28310 .exactZero (none)

def event28313 : Event := .preFoldPolynomial 28312 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩, (1)⟩] .exactZero none

def exact28314RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩, (1)⟩]

def event28314 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19253⟩⟩) 28313 exact28314RawTerms .large 28310 .exactZero (none)

def event28315 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25315⟩⟩)

def event28316 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event28317 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event28318 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.authority (.operator))

def event28319 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4989⟩⟩) (.finite 5)

def event28320 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event28321 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event28322 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event28323 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event28324 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 28323

def event28325 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 28321

def event28326 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 28324 .coefficient) (.value (.predecessor 1 28325 .coefficient)))

def event28327 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event28328 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 0 ⟨5503⟩ 28327

def event28329 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5514⟩⟩) 1 ⟨4989⟩ 28319

def event28330 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.sum [.predecessor 0 28328 .coefficient, .predecessor 1 28329 .coefficient])

def event28331 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5514⟩⟩) (.finite 222)

def event28332 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 0 ⟨5514⟩ 28331

def event28333 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5554⟩⟩) 1 ⟨961⟩ 28317

def event28334 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.identity (.predecessor 1 28333 .coefficient))

def event28335 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5554⟩⟩) (.finite 224)

def event28336 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11145⟩⟩) 0 ⟨5554⟩ 28335

def event28337 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11145⟩⟩) (.authority (.programFamilyFact))

def exact28338RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩], []⟩, (1)⟩]

theorem exact28338RawTermsValid :
    exact28338RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28338 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11145⟩⟩) exact28338RawTerms (.finite 6) 28337 .exactZero (none)

def event28339 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12190⟩⟩) 0 ⟨5554⟩ 28335

def event28340 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12190⟩⟩) (.authority (.programFamilyFact))

def exact28341RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12190⟩⟩], []⟩, (1)⟩]

theorem exact28341RawTermsValid :
    exact28341RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28341 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12190⟩⟩) exact28341RawTerms (.finite 6) 28340 .exactZero (none)

def event28342 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12191⟩⟩) 0 ⟨12190⟩ 28341

def event28343 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12191⟩⟩) 1 ⟨11145⟩ 28338

def event28344 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12191⟩⟩) (.product (.predecessor 0 28342 .coefficient) (.predecessor 1 28343 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event28345 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12191⟩⟩, .operator (⟨28341, 0⟩, ⟨28338, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], []⟩, (1)⟩)

def exact28346RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], []⟩, (1)⟩]

theorem exact28346RawTermsValid :
    exact28346RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28346 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12191⟩⟩) exact28346RawTerms (.finite 36) 28344 .exactZero (none)

def event28347 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12192⟩⟩) 0 ⟨12191⟩ 28346

def event28348 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12192⟩⟩) (.identity (.predecessor 0 28347 .coefficient))

def event28349 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12192⟩⟩) (.finite 36)

def event28350 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23169⟩⟩) 0 ⟨12192⟩ 28349

def event28351 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23169⟩⟩) (.authority (.programFamilyFact))

def event28352 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23169⟩⟩) (.finite 3720)

def event28353 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event28354 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23170⟩⟩) 0 ⟨6689⟩ 28353

def event28355 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23170⟩⟩) 1 ⟨23169⟩ 28352

def event28356 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23170⟩⟩) (.authority (.operator))

def exact28357RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23170⟩⟩]⟩, (1)⟩]

theorem exact28357RawTermsValid :
    exact28357RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28357 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23170⟩⟩) exact28357RawTerms .large 28356 .exactZero (none)

def event28358 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25311⟩⟩) 0 ⟨23170⟩ 28357

def event28359 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25311⟩⟩) (.authority (.operator))

def exact28360RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩, (1)⟩]

theorem exact28360RawTermsValid :
    exact28360RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28360 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25311⟩⟩) exact28360RawTerms (.finite 8192) 28359 .exactZero (none)

def event28361 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event28362 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event28363 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12282⟩⟩) 0 ⟨12192⟩ 28349

def event28364 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12282⟩⟩) 1 ⟨110⟩ 28362

def event28365 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12282⟩⟩) (.sum [.predecessor 0 28363 .coefficient, .predecessor 1 28364 .coefficient])

def event28366 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12282⟩⟩) (.finite 36)

def event28367 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12283⟩⟩) 0 ⟨12282⟩ 28366

def event28368 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12283⟩⟩) (.identity (.predecessor 0 28367 .coefficient))

def exact28369RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], []⟩, (1)⟩]

theorem exact28369RawTermsValid :
    exact28369RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28369 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12283⟩⟩) exact28369RawTerms (.finite 36) 28368 .exactZero (none)

def event28370 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact28371RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact28371RawTermsValid :
    exact28371RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28371 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact28371RawTerms .large 28370 .exactZero (none)

def event28372 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12284⟩⟩) 0 ⟨6544⟩ 28371

def event28373 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12284⟩⟩) 1 ⟨12283⟩ 28369

def event28374 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12284⟩⟩) (.product (.predecessor 0 28372 .coefficient) (.predecessor 1 28373 .coefficient) (⟨false, false, none, none, none⟩))

def event28375 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12284⟩⟩, .operator (⟨28371, 0⟩, ⟨28369, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact28376RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact28376RawTermsValid :
    exact28376RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28376 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12284⟩⟩) exact28376RawTerms .large 28374 .exactZero (none)

def event28377 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event28378 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event28379 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 28353

def event28380 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact28381RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact28381RawTermsValid :
    exact28381RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28381 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact28381RawTerms .large 28380 .exactZero (none)

def event28382 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6775⟩⟩) 0 ⟨6757⟩ 28381

def event28383 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6775⟩⟩) (.identity (.predecessor 0 28382 .coefficient))

def exact28384RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩, (1)⟩]

theorem exact28384RawTermsValid :
    exact28384RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28384 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6775⟩⟩) exact28384RawTerms .large 28383 .exactZero (none)

def event28385 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7840⟩⟩) 0 ⟨6775⟩ 28384

def event28386 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7840⟩⟩) (.authority (.operator))

def exact28387RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩]

theorem exact28387RawTermsValid :
    exact28387RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28387 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7840⟩⟩) exact28387RawTerms (.finite 8192) 28386 .exactZero (none)

def event28388 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7841⟩⟩) 0 ⟨7840⟩ 28387

def event28389 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7841⟩⟩) 1 ⟨2348⟩ 28378

def event28390 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7841⟩⟩) (.scale (.predecessor 0 28388 .coefficient) (.value (.predecessor 1 28389 .coefficient)))

def exact28391RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩]

theorem exact28391RawTermsValid :
    exact28391RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28391 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7841⟩⟩) exact28391RawTerms (.finite 8192) 28390 .exactZero (none)

def event28392 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6792⟩⟩) 0 ⟨6757⟩ 28381

def event28393 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6792⟩⟩) (.identity (.predecessor 0 28392 .coefficient))

def exact28394RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩]⟩, (1)⟩]

theorem exact28394RawTermsValid :
    exact28394RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28394 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6792⟩⟩) exact28394RawTerms .large 28393 .exactZero (none)

def event28395 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7842⟩⟩) 0 ⟨6792⟩ 28394

def event28396 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7842⟩⟩) 1 ⟨7841⟩ 28391

def event28397 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7842⟩⟩) (.product (.predecessor 0 28395 .coefficient) (.predecessor 1 28396 .coefficient) (⟨false, false, none, none, none⟩))

def event28398 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7842⟩⟩, .operator (⟨28394, 0⟩, ⟨28391, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩)

def exact28399RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩]

theorem exact28399RawTermsValid :
    exact28399RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28399 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7842⟩⟩) exact28399RawTerms .large 28397 .exactZero (none)

def event28400 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12285⟩⟩) 0 ⟨7842⟩ 28399

def event28401 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12285⟩⟩) 1 ⟨12284⟩ 28376

def event28402 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12285⟩⟩) (.sum [.predecessor 0 28400 .coefficient, .predecessor 1 28401 .coefficient])

def exact28403RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact28403RawTermsValid :
    exact28403RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28403 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12285⟩⟩) exact28403RawTerms .large 28402 .exactZero (none)

def event28404 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25314⟩⟩) 0 ⟨12285⟩ 28403

def event28405 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25314⟩⟩) 1 ⟨25311⟩ 28360

def event28406 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25314⟩⟩) (.product (.predecessor 0 28404 .coefficient) (.predecessor 1 28405 .coefficient) (⟨false, false, none, none, none⟩))

def event28407 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25314⟩⟩, .operator (⟨28403, 0⟩, ⟨28360, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩, (1)⟩)

def event28408 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25314⟩⟩, .operator (⟨28403, 1⟩, ⟨28360, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩, (-1)⟩)

def event28409 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25314⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25311⟩⟩) ⟨23170⟩ 28357)

def event28410 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25314⟩⟩, .relation 28409 0, ⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨23170⟩⟩]⟩, (-1)⟩)

def exact28411RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25311⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11145⟩⟩, ⟨.program ⟨214⟩, ⟨12190⟩⟩], [⟨.program ⟨214⟩, ⟨23170⟩⟩]⟩, (-1)⟩]

theorem exact28411RawTermsValid :
    exact28411RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28411 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25314⟩⟩) exact28411RawTerms .large 28406 .exactZero (none)

def event28412 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15434⟩⟩) 0 ⟨12192⟩ 28349

def event28413 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15434⟩⟩) (.authority (.programFamilyFact))

def exact28414RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15434⟩⟩], []⟩, (1)⟩]

theorem exact28414RawTermsValid :
    exact28414RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28414 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15434⟩⟩) exact28414RawTerms (.finite 6) 28413 .exactZero (none)

def event28415 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15436⟩⟩) 0 ⟨6544⟩ 28371

def eventLeaf1760 : Array AnnotatedEvent := #[
  { event := event28160
    frameStart := 0 },
  { event := event28161
    frameStart := 0 },
  { event := event28162
    frameStart := 0 },
  { event := event28163
    frameStart := 0 },
  { event := event28164
    frameStart := 0 },
  { event := event28165
    frameStart := 0 },
  { event := event28166
    frameStart := 0 },
  { event := event28167
    frameStart := 0 },
  { event := event28168
    frameStart := 0 },
  { event := event28169
    frameStart := 0 },
  { event := event28170
    frameStart := 0 },
  { event := event28171
    frameStart := 0 },
  { event := event28172
    frameStart := 0 },
  { event := event28173
    frameStart := 0 },
  { event := event28174
    frameStart := 0 },
  { event := event28175
    frameStart := 0 }
]

def eventLeaf1761 : Array AnnotatedEvent := #[
  { event := event28176
    frameStart := 0 },
  { event := event28177
    frameStart := 0 },
  { event := event28178
    frameStart := 0 },
  { event := event28179
    frameStart := 0 },
  { event := event28180
    frameStart := 0 },
  { event := event28181
    frameStart := 0 },
  { event := event28182
    frameStart := 0 },
  { event := event28183
    frameStart := 0 },
  { event := event28184
    frameStart := 0 },
  { event := event28185
    frameStart := 0 },
  { event := event28186
    frameStart := 0 },
  { event := event28187
    frameStart := 0 },
  { event := event28188
    frameStart := 0 },
  { event := event28189
    frameStart := 0 },
  { event := event28190
    frameStart := 0 },
  { event := event28191
    frameStart := 0 }
]

def eventLeaf1762 : Array AnnotatedEvent := #[
  { event := event28192
    frameStart := 0 },
  { event := event28193
    frameStart := 0 },
  { event := event28194
    frameStart := 0 },
  { event := event28195
    frameStart := 0 },
  { event := event28196
    frameStart := 0 },
  { event := event28197
    frameStart := 0 },
  { event := event28198
    frameStart := 0 },
  { event := event28199
    frameStart := 0 },
  { event := event28200
    frameStart := 0 },
  { event := event28201
    frameStart := 0 },
  { event := event28202
    frameStart := 0 },
  { event := event28203
    frameStart := 0 },
  { event := event28204
    frameStart := 0 },
  { event := event28205
    frameStart := 0 },
  { event := event28206
    frameStart := 0 },
  { event := event28207
    frameStart := 0 }
]

def eventLeaf1763 : Array AnnotatedEvent := #[
  { event := event28208
    frameStart := 0 },
  { event := event28209
    frameStart := 0 },
  { event := event28210
    frameStart := 0 },
  { event := event28211
    frameStart := 0 },
  { event := event28212
    frameStart := 0 },
  { event := event28213
    frameStart := 0 },
  { event := event28214
    frameStart := 0 },
  { event := event28215
    frameStart := 0 },
  { event := event28216
    frameStart := 0 },
  { event := event28217
    frameStart := 0 },
  { event := event28218
    frameStart := 0 },
  { event := event28219
    frameStart := 0 },
  { event := event28220
    frameStart := 0 },
  { event := event28221
    frameStart := 0 },
  { event := event28222
    frameStart := 0 },
  { event := event28223
    frameStart := 0 }
]

def eventLeaf1764 : Array AnnotatedEvent := #[
  { event := event28224
    frameStart := 0 },
  { event := event28225
    frameStart := 0 },
  { event := event28226
    frameStart := 0 },
  { event := event28227
    frameStart := 0 },
  { event := event28228
    frameStart := 0 },
  { event := event28229
    frameStart := 0 },
  { event := event28230
    frameStart := 0 },
  { event := event28231
    frameStart := 0 },
  { event := event28232
    frameStart := 0 },
  { event := event28233
    frameStart := 0 },
  { event := event28234
    frameStart := 0 },
  { event := event28235
    frameStart := 0 },
  { event := event28236
    frameStart := 0 },
  { event := event28237
    frameStart := 0 },
  { event := event28238
    frameStart := 0 },
  { event := event28239
    frameStart := 0 }
]

def eventLeaf1765 : Array AnnotatedEvent := #[
  { event := event28240
    frameStart := 0 },
  { event := event28241
    frameStart := 0 },
  { event := event28242
    frameStart := 0 },
  { event := event28243
    frameStart := 0 },
  { event := event28244
    frameStart := 0 },
  { event := event28245
    frameStart := 0 },
  { event := event28246
    frameStart := 0 },
  { event := event28247
    frameStart := 0 },
  { event := event28248
    frameStart := 0 },
  { event := event28249
    frameStart := 0 },
  { event := event28250
    frameStart := 0 },
  { event := event28251
    frameStart := 0 },
  { event := event28252
    frameStart := 0 },
  { event := event28253
    frameStart := 0 },
  { event := event28254
    frameStart := 0 },
  { event := event28255
    frameStart := 0 }
]

def eventLeaf1766 : Array AnnotatedEvent := #[
  { event := event28256
    frameStart := 0 },
  { event := event28257
    frameStart := 0 },
  { event := event28258
    frameStart := 0 },
  { event := event28259
    frameStart := 0 },
  { event := event28260
    frameStart := 0 },
  { event := event28261
    frameStart := 0 },
  { event := event28262
    frameStart := 0 },
  { event := event28263
    frameStart := 0 },
  { event := event28264
    frameStart := 0 },
  { event := event28265
    frameStart := 0 },
  { event := event28266
    frameStart := 0 },
  { event := event28267
    frameStart := 28267 },
  { event := event28268
    frameStart := 28267 },
  { event := event28269
    frameStart := 28267 },
  { event := event28270
    frameStart := 28267 },
  { event := event28271
    frameStart := 28267 }
]

def eventLeaf1767 : Array AnnotatedEvent := #[
  { event := event28272
    frameStart := 28267 },
  { event := event28273
    frameStart := 28267 },
  { event := event28274
    frameStart := 28267 },
  { event := event28275
    frameStart := 28267 },
  { event := event28276
    frameStart := 28267 },
  { event := event28277
    frameStart := 28267 },
  { event := event28278
    frameStart := 28267 },
  { event := event28279
    frameStart := 28267 },
  { event := event28280
    frameStart := 28267 },
  { event := event28281
    frameStart := 28267 },
  { event := event28282
    frameStart := 28267 },
  { event := event28283
    frameStart := 28267 },
  { event := event28284
    frameStart := 28267 },
  { event := event28285
    frameStart := 28267 },
  { event := event28286
    frameStart := 28267 },
  { event := event28287
    frameStart := 28267 }
]

def eventLeaf1768 : Array AnnotatedEvent := #[
  { event := event28288
    frameStart := 28267 },
  { event := event28289
    frameStart := 28267 },
  { event := event28290
    frameStart := 28267 },
  { event := event28291
    frameStart := 28267 },
  { event := event28292
    frameStart := 28267 },
  { event := event28293
    frameStart := 28267 },
  { event := event28294
    frameStart := 28267 },
  { event := event28295
    frameStart := 28267 },
  { event := event28296
    frameStart := 28267 },
  { event := event28297
    frameStart := 28267 },
  { event := event28298
    frameStart := 28267 },
  { event := event28299
    frameStart := 28267 },
  { event := event28300
    frameStart := 28267 },
  { event := event28301
    frameStart := 28267 },
  { event := event28302
    frameStart := 28267 },
  { event := event28303
    frameStart := 28267 }
]

def eventLeaf1769 : Array AnnotatedEvent := #[
  { event := event28304
    frameStart := 28267 },
  { event := event28305
    frameStart := 28267 },
  { event := event28306
    frameStart := 28267 },
  { event := event28307
    frameStart := 28267 },
  { event := event28308
    frameStart := 28267 },
  { event := event28309
    frameStart := 28267 },
  { event := event28310
    frameStart := 28267 },
  { event := event28311
    frameStart := 28267 },
  { event := event28312
    frameStart := 28267 },
  { event := event28313
    frameStart := 28267 },
  { event := event28314
    frameStart := 28267 },
  { event := event28315
    frameStart := 28315 },
  { event := event28316
    frameStart := 28315 },
  { event := event28317
    frameStart := 28315 },
  { event := event28318
    frameStart := 28315 },
  { event := event28319
    frameStart := 28315 }
]

def eventLeaf1770 : Array AnnotatedEvent := #[
  { event := event28320
    frameStart := 28315 },
  { event := event28321
    frameStart := 28315 },
  { event := event28322
    frameStart := 28315 },
  { event := event28323
    frameStart := 28315 },
  { event := event28324
    frameStart := 28315 },
  { event := event28325
    frameStart := 28315 },
  { event := event28326
    frameStart := 28315 },
  { event := event28327
    frameStart := 28315 },
  { event := event28328
    frameStart := 28315 },
  { event := event28329
    frameStart := 28315 },
  { event := event28330
    frameStart := 28315 },
  { event := event28331
    frameStart := 28315 },
  { event := event28332
    frameStart := 28315 },
  { event := event28333
    frameStart := 28315 },
  { event := event28334
    frameStart := 28315 },
  { event := event28335
    frameStart := 28315 }
]

def eventLeaf1771 : Array AnnotatedEvent := #[
  { event := event28336
    frameStart := 28315 },
  { event := event28337
    frameStart := 28315 },
  { event := event28338
    frameStart := 28315 },
  { event := event28339
    frameStart := 28315 },
  { event := event28340
    frameStart := 28315 },
  { event := event28341
    frameStart := 28315 },
  { event := event28342
    frameStart := 28315 },
  { event := event28343
    frameStart := 28315 },
  { event := event28344
    frameStart := 28315 },
  { event := event28345
    frameStart := 28315 },
  { event := event28346
    frameStart := 28315 },
  { event := event28347
    frameStart := 28315 },
  { event := event28348
    frameStart := 28315 },
  { event := event28349
    frameStart := 28315 },
  { event := event28350
    frameStart := 28315 },
  { event := event28351
    frameStart := 28315 }
]

def eventLeaf1772 : Array AnnotatedEvent := #[
  { event := event28352
    frameStart := 28315 },
  { event := event28353
    frameStart := 28315 },
  { event := event28354
    frameStart := 28315 },
  { event := event28355
    frameStart := 28315 },
  { event := event28356
    frameStart := 28315 },
  { event := event28357
    frameStart := 28315 },
  { event := event28358
    frameStart := 28315 },
  { event := event28359
    frameStart := 28315 },
  { event := event28360
    frameStart := 28315 },
  { event := event28361
    frameStart := 28315 },
  { event := event28362
    frameStart := 28315 },
  { event := event28363
    frameStart := 28315 },
  { event := event28364
    frameStart := 28315 },
  { event := event28365
    frameStart := 28315 },
  { event := event28366
    frameStart := 28315 },
  { event := event28367
    frameStart := 28315 }
]

def eventLeaf1773 : Array AnnotatedEvent := #[
  { event := event28368
    frameStart := 28315 },
  { event := event28369
    frameStart := 28315 },
  { event := event28370
    frameStart := 28315 },
  { event := event28371
    frameStart := 28315 },
  { event := event28372
    frameStart := 28315 },
  { event := event28373
    frameStart := 28315 },
  { event := event28374
    frameStart := 28315 },
  { event := event28375
    frameStart := 28315 },
  { event := event28376
    frameStart := 28315 },
  { event := event28377
    frameStart := 28315 },
  { event := event28378
    frameStart := 28315 },
  { event := event28379
    frameStart := 28315 },
  { event := event28380
    frameStart := 28315 },
  { event := event28381
    frameStart := 28315 },
  { event := event28382
    frameStart := 28315 },
  { event := event28383
    frameStart := 28315 }
]

def eventLeaf1774 : Array AnnotatedEvent := #[
  { event := event28384
    frameStart := 28315 },
  { event := event28385
    frameStart := 28315 },
  { event := event28386
    frameStart := 28315 },
  { event := event28387
    frameStart := 28315 },
  { event := event28388
    frameStart := 28315 },
  { event := event28389
    frameStart := 28315 },
  { event := event28390
    frameStart := 28315 },
  { event := event28391
    frameStart := 28315 },
  { event := event28392
    frameStart := 28315 },
  { event := event28393
    frameStart := 28315 },
  { event := event28394
    frameStart := 28315 },
  { event := event28395
    frameStart := 28315 },
  { event := event28396
    frameStart := 28315 },
  { event := event28397
    frameStart := 28315 },
  { event := event28398
    frameStart := 28315 },
  { event := event28399
    frameStart := 28315 }
]

def eventLeaf1775 : Array AnnotatedEvent := #[
  { event := event28400
    frameStart := 28315 },
  { event := event28401
    frameStart := 28315 },
  { event := event28402
    frameStart := 28315 },
  { event := event28403
    frameStart := 28315 },
  { event := event28404
    frameStart := 28315 },
  { event := event28405
    frameStart := 28315 },
  { event := event28406
    frameStart := 28315 },
  { event := event28407
    frameStart := 28315 },
  { event := event28408
    frameStart := 28315 },
  { event := event28409
    frameStart := 28315 },
  { event := event28410
    frameStart := 28315 },
  { event := event28411
    frameStart := 28315 },
  { event := event28412
    frameStart := 28315 },
  { event := event28413
    frameStart := 28315 },
  { event := event28414
    frameStart := 28315 },
  { event := event28415
    frameStart := 28315 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events110
