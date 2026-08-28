import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events325

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event83200 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24414⟩⟩) 1 ⟨24412⟩ 83197

def event83201 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24414⟩⟩) (.authority (.operator))

def exact83202RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24414⟩⟩]⟩, (1)⟩]

theorem exact83202RawTermsValid :
    exact83202RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83202 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24414⟩⟩) exact83202RawTerms .large 83201 .exactZero (none)

def event83203 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28734⟩⟩) 0 ⟨24414⟩ 83202

def event83204 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28734⟩⟩) (.authority (.operator))

def exact83205RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28734⟩⟩]⟩, (1)⟩]

theorem exact83205RawTermsValid :
    exact83205RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83205 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28734⟩⟩) exact83205RawTerms (.finite 8192) 83204 .exactZero (none)

def event83206 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event83207 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event83208 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16421⟩⟩) 0 ⟨16382⟩ 83194

def event83209 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16421⟩⟩) 1 ⟨110⟩ 83207

def event83210 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16421⟩⟩) (.sum [.predecessor 0 83208 .coefficient, .predecessor 1 83209 .coefficient])

def event83211 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16421⟩⟩) (.finite 36)

def event83212 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16422⟩⟩) 0 ⟨16421⟩ 83211

def event83213 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16422⟩⟩) (.identity (.predecessor 0 83212 .coefficient))

def exact83214RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], []⟩, (1)⟩]

theorem exact83214RawTermsValid :
    exact83214RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83214 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16422⟩⟩) exact83214RawTerms (.finite 36) 83213 .exactZero (none)

def event83215 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact83216RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact83216RawTermsValid :
    exact83216RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83216 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact83216RawTerms .large 83215 .exactZero (none)

def event83217 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16423⟩⟩) 0 ⟨6544⟩ 83216

def event83218 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16423⟩⟩) 1 ⟨16422⟩ 83214

def event83219 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16423⟩⟩) (.product (.predecessor 0 83217 .coefficient) (.predecessor 1 83218 .coefficient) (⟨false, false, none, none, none⟩))

def event83220 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16423⟩⟩, .operator (⟨83216, 0⟩, ⟨83214, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact83221RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact83221RawTermsValid :
    exact83221RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83221 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16423⟩⟩) exact83221RawTerms .large 83219 .exactZero (none)

def event83222 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6701⟩⟩) 0 ⟨6689⟩ 83198

def event83223 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6701⟩⟩) (.authority (.operator))

def exact83224RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩]

theorem exact83224RawTermsValid :
    exact83224RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83224 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6701⟩⟩) exact83224RawTerms .large 83223 .exactZero (none)

def event83225 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16424⟩⟩) 0 ⟨6701⟩ 83224

def event83226 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16424⟩⟩) 1 ⟨16423⟩ 83221

def event83227 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16424⟩⟩) (.sum [.predecessor 0 83225 .coefficient, .predecessor 1 83226 .coefficient])

def exact83228RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83228RawTermsValid :
    exact83228RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83228 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16424⟩⟩) exact83228RawTerms .large 83227 .exactZero (none)

def event83229 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28735⟩⟩) 0 ⟨16424⟩ 83228

def event83230 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28735⟩⟩) 1 ⟨28734⟩ 83205

def event83231 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28735⟩⟩) (.product (.predecessor 0 83229 .coefficient) (.predecessor 1 83230 .coefficient) (⟨false, false, none, none, none⟩))

def event83232 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28735⟩⟩, .operator (⟨83228, 0⟩, ⟨83205, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28734⟩⟩]⟩, (1)⟩)

def event83233 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28735⟩⟩, .operator (⟨83228, 1⟩, ⟨83205, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28734⟩⟩]⟩, (-1)⟩)

def event83234 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28735⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28734⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28734⟩⟩) ⟨24414⟩ 83202)

def event83235 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28735⟩⟩, .relation 83234 0, ⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨24414⟩⟩]⟩, (-1)⟩)

def exact83236RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28734⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨24414⟩⟩]⟩, (-1)⟩]

theorem exact83236RawTermsValid :
    exact83236RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83236 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28735⟩⟩) exact83236RawTerms .large 83231 .exactZero (none)

def event83237 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17120⟩⟩) 0 ⟨16382⟩ 83194

def event83238 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17120⟩⟩) (.authority (.programFamilyFact))

def exact83239RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17120⟩⟩], []⟩, (1)⟩]

theorem exact83239RawTermsValid :
    exact83239RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83239 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17120⟩⟩) exact83239RawTerms (.finite 62) 83238 .exactZero (none)

def event83240 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17121⟩⟩) 0 ⟨6544⟩ 83216

def event83241 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17121⟩⟩) 1 ⟨17120⟩ 83239

def event83242 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17121⟩⟩) (.product (.predecessor 0 83240 .coefficient) (.predecessor 1 83241 .coefficient) (⟨false, true, none, none, some 1⟩))

def event83243 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17121⟩⟩, .operator (⟨83216, 0⟩, ⟨83239, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact83244RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact83244RawTermsValid :
    exact83244RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83244 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17121⟩⟩) exact83244RawTerms .large 83242 .exactZero (none)

def event83245 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6731⟩⟩) 0 ⟨6689⟩ 83198

def event83246 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6731⟩⟩) (.authority (.operator))

def exact83247RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩]

theorem exact83247RawTermsValid :
    exact83247RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83247 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6731⟩⟩) exact83247RawTerms .large 83246 .exactZero (none)

def event83248 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17122⟩⟩) 0 ⟨6731⟩ 83247

def event83249 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17122⟩⟩) 1 ⟨17121⟩ 83244

def event83250 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17122⟩⟩) (.sum [.predecessor 0 83248 .coefficient, .predecessor 1 83249 .coefficient])

def exact83251RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83251RawTermsValid :
    exact83251RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83251 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17122⟩⟩) exact83251RawTerms .large 83250 .exactZero (none)

def event83252 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28739⟩⟩) 0 ⟨17122⟩ 83251

def event83253 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28739⟩⟩) 1 ⟨28735⟩ 83236

def event83254 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28739⟩⟩) (.sum [.predecessor 0 83252 .coefficient, .predecessor 1 83253 .coefficient])

def exact83255RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28734⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨24414⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83255RawTermsValid :
    exact83255RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83255 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28739⟩⟩) exact83255RawTerms .large 83254 .exactZero (none)

def event83256 : Event := .preFoldPolynomial 83255 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28734⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨24414⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact83257RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28734⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨24414⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event83257 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28739⟩⟩) 83256 exact83257RawTerms .large 83254 .exactZero (none)

def event83258 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16382⟩⟩) ⟨⟨144⟩, ⟨52⟩, ⟨109⟩⟩ ⟨83100, 83258⟩

def event83259 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21979⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21976⟩⟩]⟩) (1) 0 2 (.universal 83258 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21976⟩⟩]⟩) (none) 83257)

def event83260 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21979⟩⟩, .relation 83259 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩)

def event83261 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21979⟩⟩, .relation 83259 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28734⟩⟩]⟩, (-1)⟩)

def event83262 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21979⟩⟩, .relation 83259 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨24414⟩⟩]⟩, (1)⟩)

def event83263 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21979⟩⟩, .relation 83259 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact83264RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28734⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨24414⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83264RawTermsValid :
    exact83264RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83264 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21979⟩⟩) exact83264RawTerms .large 83096 (.finite 1811303510016) (some (83098))

def event83265 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28737⟩⟩) 0 ⟨21979⟩ 83264

def event83266 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28737⟩⟩) 1 ⟨28736⟩ 83086

def event83267 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28737⟩⟩) (.sum [.predecessor 0 83265 .coefficient, .predecessor 1 83266 .coefficient])

def event83268 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28737⟩⟩, .operator (⟨83264, 0⟩, ⟨83086, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28734⟩⟩]⟩, (1)⟩)

def event83269 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28737⟩⟩, .operator (⟨83264, 2⟩, ⟨83086, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨24414⟩⟩]⟩, (-1)⟩)

def event83270 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28737⟩⟩) (.sum [.result 83264 .summary, .result 83086 .summary])

def exact83271RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83271RawTermsValid :
    exact83271RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83271 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28737⟩⟩) exact83271RawTerms .large 83267 (.finite 1292270185944771604480) (some (83270))

def event83272 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24349⟩⟩) 0 ⟨16263⟩ 4006

def event83273 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24349⟩⟩) (.authority (.programFamilyFact))

def event83274 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24349⟩⟩) (.finite 3720)

def event83275 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24351⟩⟩) 0 ⟨6689⟩ 5477

def event83276 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24351⟩⟩) 1 ⟨24349⟩ 83274

def event83277 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24351⟩⟩) (.authority (.operator))

def exact83278RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24351⟩⟩]⟩, (1)⟩]

theorem exact83278RawTermsValid :
    exact83278RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83278 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24351⟩⟩) exact83278RawTerms .large 83277 .exactZero (none)

def event83279 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28517⟩⟩) 0 ⟨24351⟩ 83278

def event83280 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28517⟩⟩) (.authority (.operator))

def exact83281RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩, (1)⟩]

theorem exact83281RawTermsValid :
    exact83281RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83281 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28517⟩⟩) exact83281RawTerms (.finite 8192) 83280 .exactZero (none)

def event83282 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23079⟩⟩) 0 ⟨11763⟩ 4000

def event83283 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23079⟩⟩) (.authority (.programFamilyFact))

def event83284 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23079⟩⟩) (.finite 3720)

def event83285 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23080⟩⟩) 0 ⟨6689⟩ 5477

def event83286 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23080⟩⟩) 1 ⟨23079⟩ 83284

def event83287 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23080⟩⟩) (.authority (.operator))

def exact83288RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23080⟩⟩]⟩, (1)⟩]

theorem exact83288RawTermsValid :
    exact83288RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83288 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23080⟩⟩) exact83288RawTerms .large 83287 .exactZero (none)

def event83289 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25142⟩⟩) 0 ⟨23080⟩ 83288

def event83290 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25142⟩⟩) (.authority (.operator))

def exact83291RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩, (1)⟩]

theorem exact83291RawTermsValid :
    exact83291RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83291 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25142⟩⟩) exact83291RawTerms (.finite 8192) 83290 .exactZero (none)

def event83292 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11764⟩⟩) 0 ⟨11761⟩ 3989

def event83293 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11764⟩⟩) 1 ⟨6567⟩ 79920

def event83294 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11764⟩⟩) (.tensor (.predecessor 0 83292 .coefficient) (.predecessor 1 83293 .coefficient) true false)

def event83295 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11764⟩⟩, .operator (⟨3989, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact83296RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact83296RawTermsValid :
    exact83296RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83296 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11764⟩⟩) exact83296RawTerms .large 83294 .exactZero (none)

def event83297 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7239⟩⟩) 0 ⟨5539⟩ 79790

def event83298 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7239⟩⟩) 1 ⟨6783⟩ 9979

def event83299 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7239⟩⟩) (.product (.predecessor 0 83297 .coefficient) (.predecessor 1 83298 .coefficient) (⟨false, false, none, none, none⟩))

def event83300 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7239⟩⟩, .operator (⟨79790, 0⟩, ⟨9979, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩)

def exact83301RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩]

theorem exact83301RawTermsValid :
    exact83301RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83301 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7239⟩⟩) exact83301RawTerms .large 83299 .exactZero (none)

def event83302 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11765⟩⟩) 0 ⟨7239⟩ 83301

def event83303 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11765⟩⟩) 1 ⟨11764⟩ 83296

def event83304 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11765⟩⟩) (.sum [.predecessor 0 83302 .coefficient, .predecessor 1 83303 .coefficient])

def exact83305RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83305RawTermsValid :
    exact83305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83305 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11765⟩⟩) exact83305RawTerms .large 83304 .exactZero (none)

def event83306 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11766⟩⟩) 0 ⟨11765⟩ 83305

def event83307 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11766⟩⟩) 1 ⟨97⟩ 9971

def event83308 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11766⟩⟩) (.sum [.predecessor 0 83306 .coefficient, .predecessor 1 83307 .coefficient])

def event83309 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11766⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨97⟩⟩]⟩) [⟨.result 9971 .coefficient, false, none⟩])

def event83310 : Event := .survivorFold (1) 83309

def exact83311RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83311RawTermsValid :
    exact83311RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83311 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11766⟩⟩) exact83311RawTerms .large 83308 (.finite 26) (some (83309))

def event83312 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11767⟩⟩) 0 ⟨11766⟩ 83311

def event83313 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11767⟩⟩) 1 ⟨9610⟩ 3992

def event83314 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11767⟩⟩) (.product (.predecessor 0 83312 .coefficient) (.predecessor 1 83313 .coefficient) (⟨false, true, none, none, some 1⟩))

def event83315 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11767⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩], []⟩) [⟨.result 3992 .coefficient, true, some 1⟩])

def event83316 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11767⟩⟩) (.product (.result 83311 .summary) (.transfer 83315) (⟨false, false, none, none, none⟩))

def event83317 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11767⟩⟩, .operator (⟨83311, 1⟩, ⟨3992, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event83318 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11767⟩⟩, .operator (⟨83311, 0⟩, ⟨3992, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩)

def exact83319RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83319RawTermsValid :
    exact83319RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83319 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11767⟩⟩) exact83319RawTerms .large 83314 (.finite 24960) (some (83316))

def event83320 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9611⟩⟩) 0 ⟨9610⟩ 3992

def event83321 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9611⟩⟩) 1 ⟨6567⟩ 79920

def event83322 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9611⟩⟩) (.tensor (.predecessor 0 83320 .coefficient) (.predecessor 1 83321 .coefficient) true false)

def event83323 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9611⟩⟩, .operator (⟨3992, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact83324RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact83324RawTermsValid :
    exact83324RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83324 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9611⟩⟩) exact83324RawTerms .large 83322 .exactZero (none)

def event83325 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7219⟩⟩) 0 ⟨5539⟩ 79790

def event83326 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7219⟩⟩) 1 ⟨6763⟩ 10020

def event83327 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7219⟩⟩) (.product (.predecessor 0 83325 .coefficient) (.predecessor 1 83326 .coefficient) (⟨false, false, none, none, none⟩))

def event83328 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7219⟩⟩, .operator (⟨79790, 0⟩, ⟨10020, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩]⟩, (1)⟩)

def exact83329RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩]⟩, (1)⟩]

theorem exact83329RawTermsValid :
    exact83329RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83329 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7219⟩⟩) exact83329RawTerms .large 83327 .exactZero (none)

def event83330 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9612⟩⟩) 0 ⟨7219⟩ 83329

def event83331 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9612⟩⟩) 1 ⟨9611⟩ 83324

def event83332 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9612⟩⟩) (.sum [.predecessor 0 83330 .coefficient, .predecessor 1 83331 .coefficient])

def exact83333RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83333RawTermsValid :
    exact83333RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83333 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9612⟩⟩) exact83333RawTerms .large 83332 .exactZero (none)

def event83334 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9613⟩⟩) 0 ⟨9612⟩ 83333

def event83335 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9613⟩⟩) 1 ⟨77⟩ 10012

def event83336 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9613⟩⟩) (.sum [.predecessor 0 83334 .coefficient, .predecessor 1 83335 .coefficient])

def event83337 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9613⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨77⟩⟩]⟩) [⟨.result 10012 .coefficient, false, none⟩])

def event83338 : Event := .survivorFold (1) 83337

def exact83339RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83339RawTermsValid :
    exact83339RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83339 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9613⟩⟩) exact83339RawTerms .large 83336 (.finite 26) (some (83337))

def event83340 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9614⟩⟩) 0 ⟨9613⟩ 83339

def event83341 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9614⟩⟩) 1 ⟨7862⟩ 10009

def event83342 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9614⟩⟩) (.product (.predecessor 0 83340 .coefficient) (.predecessor 1 83341 .coefficient) (⟨false, false, none, none, none⟩))

def event83343 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9614⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩) [⟨.result 10005 .coefficient, false, none⟩])

def event83344 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9614⟩⟩) (.product (.result 83339 .summary) (.transfer 83343) (⟨false, false, none, none, none⟩))

def event83345 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9614⟩⟩, .operator (⟨83339, 1⟩, ⟨10009, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (-1)⟩)

def event83346 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨9614⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7861⟩⟩) ⟨6783⟩ 9979)

def event83347 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9614⟩⟩, .relation 83346 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (-1)⟩)

def event83348 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9614⟩⟩, .operator (⟨83339, 0⟩, ⟨10009, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩)

def exact83349RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (-1)⟩]

theorem exact83349RawTermsValid :
    exact83349RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83349 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9614⟩⟩) exact83349RawTerms .large 83342 (.finite 95420416) (some (83344))

def event83350 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11768⟩⟩) 0 ⟨9614⟩ 83349

def event83351 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11768⟩⟩) 1 ⟨11767⟩ 83319

def event83352 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11768⟩⟩) (.sum [.predecessor 0 83350 .coefficient, .predecessor 1 83351 .coefficient])

def event83353 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11768⟩⟩, .operator (⟨83349, 1⟩, ⟨83319, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩)

def event83354 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11768⟩⟩) (.sum [.result 83349 .summary, .result 83319 .summary])

def exact83355RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83355RawTermsValid :
    exact83355RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83355 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11768⟩⟩) exact83355RawTerms .large 83352 (.finite 95445376) (some (83354))

def event83356 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25143⟩⟩) 0 ⟨11768⟩ 83355

def event83357 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25143⟩⟩) 1 ⟨25142⟩ 83291

def event83358 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25143⟩⟩) (.product (.predecessor 0 83356 .coefficient) (.predecessor 1 83357 .coefficient) (⟨false, false, none, none, none⟩))

def event83359 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25143⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩) [⟨.result 83291 .coefficient, false, none⟩])

def event83360 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25143⟩⟩) (.product (.result 83355 .summary) (.transfer 83359) (⟨false, false, none, none, none⟩))

def event83361 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25143⟩⟩, .operator (⟨83355, 1⟩, ⟨83291, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩, (-1)⟩)

def event83362 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25143⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25142⟩⟩) ⟨23080⟩ 83288)

def event83363 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25143⟩⟩, .relation 83362 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨23080⟩⟩]⟩, (-1)⟩)

def event83364 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25143⟩⟩, .operator (⟨83355, 0⟩, ⟨83291, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩, (1)⟩)

def exact83365RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨23080⟩⟩]⟩, (-1)⟩]

theorem exact83365RawTermsValid :
    exact83365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83365 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25143⟩⟩) exact83365RawTerms .large 83358 (.finite 350286057046016) (some (83360))

def event83366 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19744⟩⟩) 0 ⟨11763⟩ 4000

def event83367 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19744⟩⟩) (.authority (.relationPreimageSource ⟨18⟩))

def exact83368RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩, (1)⟩]

theorem exact83368RawTermsValid :
    exact83368RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83368 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19744⟩⟩) exact83368RawTerms (.finite 136065468) 83367 .exactZero (none)

def event83369 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19746⟩⟩) 0 ⟨19744⟩ 83368

def event83370 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19746⟩⟩) 1 ⟨2348⟩ 4

def event83371 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19746⟩⟩) (.scale (.predecessor 0 83369 .coefficient) (.value (.predecessor 1 83370 .coefficient)))

def exact83372RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩, (1)⟩]

theorem exact83372RawTermsValid :
    exact83372RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83372 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19746⟩⟩) exact83372RawTerms (.finite 136065468) 83371 .exactZero (none)

def event83373 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19747⟩⟩) 0 ⟨5541⟩ 80012

def event83374 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19747⟩⟩) 1 ⟨19746⟩ 83372

def event83375 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19747⟩⟩) (.product (.predecessor 0 83373 .coefficient) (.predecessor 1 83374 .coefficient) (⟨false, false, none, none, none⟩))

def event83376 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19747⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩) [⟨.result 83368 .coefficient, false, none⟩])

def event83377 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19747⟩⟩) (.product (.result 80012 .summary) (.transfer 83376) (⟨false, false, none, none, none⟩))

def event83378 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19747⟩⟩, .operator (⟨80012, 0⟩, ⟨83372, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩, (1)⟩)

def event83379 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19745⟩⟩)

def event83380 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event83381 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event83382 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event83383 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event83384 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event83385 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event83386 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event83387 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event83388 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 83387

def event83389 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 83385

def event83390 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 83388 .coefficient) (.value (.predecessor 1 83389 .coefficient)))

def event83391 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event83392 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 83391

def event83393 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 83383

def event83394 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 83392 .coefficient, .predecessor 1 83393 .coefficient])

def event83395 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event83396 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 83395

def event83397 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 83381

def event83398 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 83397 .coefficient))

def event83399 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event83400 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11761⟩⟩) 0 ⟨5536⟩ 83399

def event83401 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11761⟩⟩) (.authority (.programFamilyFact))

def exact83402RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11761⟩⟩], []⟩, (1)⟩]

theorem exact83402RawTermsValid :
    exact83402RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83402 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11761⟩⟩) exact83402RawTerms (.finite 30) 83401 .exactZero (none)

def event83403 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9610⟩⟩) 0 ⟨5536⟩ 83399

def event83404 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9610⟩⟩) (.authority (.programFamilyFact))

def exact83405RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩], []⟩, (1)⟩]

theorem exact83405RawTermsValid :
    exact83405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83405 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9610⟩⟩) exact83405RawTerms (.finite 30) 83404 .exactZero (none)

def event83406 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11762⟩⟩) 0 ⟨9610⟩ 83405

def event83407 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11762⟩⟩) 1 ⟨11761⟩ 83402

def event83408 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11762⟩⟩) (.product (.predecessor 0 83406 .coefficient) (.predecessor 1 83407 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event83409 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11762⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], []⟩) [⟨.result 83405 .coefficient, true, some 1⟩, ⟨.result 83402 .coefficient, true, some 1⟩])

def event83410 : Event := .survivorFold (1) 83409

def exact83411RawTerms : List Term := []

theorem exact83411RawTermsValid :
    exact83411RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83411 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11762⟩⟩) exact83411RawTerms (.finite 900) 83408 (.finite 900) (some (83409))

def event83412 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11763⟩⟩) 0 ⟨11762⟩ 83411

def event83413 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11763⟩⟩) (.identity (.predecessor 0 83412 .coefficient))

def event83414 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11763⟩⟩) (.finite 900)

def event83415 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19744⟩⟩) 0 ⟨11763⟩ 83414

def event83416 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19744⟩⟩) (.authority (.relationPreimageSource ⟨18⟩))

def exact83417RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩, (1)⟩]

theorem exact83417RawTermsValid :
    exact83417RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83417 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19744⟩⟩) exact83417RawTerms (.finite 136065468) 83416 .exactZero (none)

def event83418 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact83419RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact83419RawTermsValid :
    exact83419RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83419 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact83419RawTerms .large 83418 .exactZero (none)

def event83420 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19745⟩⟩) 0 ⟨6⟩ 83419

def event83421 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19745⟩⟩) 1 ⟨19744⟩ 83417

def event83422 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19745⟩⟩) (.product (.predecessor 0 83420 .coefficient) (.predecessor 1 83421 .coefficient) (⟨false, false, none, none, none⟩))

def event83423 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19745⟩⟩, .operator (⟨83419, 0⟩, ⟨83417, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩, (1)⟩)

def exact83424RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩, (1)⟩]

theorem exact83424RawTermsValid :
    exact83424RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83424 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19745⟩⟩) exact83424RawTerms .large 83422 .exactZero (none)

def event83425 : Event := .preFoldPolynomial 83424 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩, (1)⟩] .exactZero none

def exact83426RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩, (1)⟩]

def event83426 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19745⟩⟩) 83425 exact83426RawTerms .large 83422 .exactZero (none)

def event83427 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25146⟩⟩)

def event83428 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event83429 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event83430 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event83431 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event83432 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event83433 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event83434 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event83435 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event83436 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 83435

def event83437 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 83433

def event83438 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 83436 .coefficient) (.value (.predecessor 1 83437 .coefficient)))

def event83439 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event83440 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 83439

def event83441 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 83431

def event83442 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 83440 .coefficient, .predecessor 1 83441 .coefficient])

def event83443 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event83444 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 83443

def event83445 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 83429

def event83446 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 83445 .coefficient))

def event83447 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event83448 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11761⟩⟩) 0 ⟨5536⟩ 83447

def event83449 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11761⟩⟩) (.authority (.programFamilyFact))

def exact83450RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11761⟩⟩], []⟩, (1)⟩]

theorem exact83450RawTermsValid :
    exact83450RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83450 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11761⟩⟩) exact83450RawTerms (.finite 30) 83449 .exactZero (none)

def event83451 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9610⟩⟩) 0 ⟨5536⟩ 83447

def event83452 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9610⟩⟩) (.authority (.programFamilyFact))

def exact83453RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩], []⟩, (1)⟩]

theorem exact83453RawTermsValid :
    exact83453RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83453 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9610⟩⟩) exact83453RawTerms (.finite 30) 83452 .exactZero (none)

def event83454 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11762⟩⟩) 0 ⟨9610⟩ 83453

def event83455 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11762⟩⟩) 1 ⟨11761⟩ 83450

def eventLeaf5200 : Array AnnotatedEvent := #[
  { event := event83200
    frameStart := 83154 },
  { event := event83201
    frameStart := 83154 },
  { event := event83202
    frameStart := 83154 },
  { event := event83203
    frameStart := 83154 },
  { event := event83204
    frameStart := 83154 },
  { event := event83205
    frameStart := 83154 },
  { event := event83206
    frameStart := 83154 },
  { event := event83207
    frameStart := 83154 },
  { event := event83208
    frameStart := 83154 },
  { event := event83209
    frameStart := 83154 },
  { event := event83210
    frameStart := 83154 },
  { event := event83211
    frameStart := 83154 },
  { event := event83212
    frameStart := 83154 },
  { event := event83213
    frameStart := 83154 },
  { event := event83214
    frameStart := 83154 },
  { event := event83215
    frameStart := 83154 }
]

def eventLeaf5201 : Array AnnotatedEvent := #[
  { event := event83216
    frameStart := 83154 },
  { event := event83217
    frameStart := 83154 },
  { event := event83218
    frameStart := 83154 },
  { event := event83219
    frameStart := 83154 },
  { event := event83220
    frameStart := 83154 },
  { event := event83221
    frameStart := 83154 },
  { event := event83222
    frameStart := 83154 },
  { event := event83223
    frameStart := 83154 },
  { event := event83224
    frameStart := 83154 },
  { event := event83225
    frameStart := 83154 },
  { event := event83226
    frameStart := 83154 },
  { event := event83227
    frameStart := 83154 },
  { event := event83228
    frameStart := 83154 },
  { event := event83229
    frameStart := 83154 },
  { event := event83230
    frameStart := 83154 },
  { event := event83231
    frameStart := 83154 }
]

def eventLeaf5202 : Array AnnotatedEvent := #[
  { event := event83232
    frameStart := 83154 },
  { event := event83233
    frameStart := 83154 },
  { event := event83234
    frameStart := 83154 },
  { event := event83235
    frameStart := 83154 },
  { event := event83236
    frameStart := 83154 },
  { event := event83237
    frameStart := 83154 },
  { event := event83238
    frameStart := 83154 },
  { event := event83239
    frameStart := 83154 },
  { event := event83240
    frameStart := 83154 },
  { event := event83241
    frameStart := 83154 },
  { event := event83242
    frameStart := 83154 },
  { event := event83243
    frameStart := 83154 },
  { event := event83244
    frameStart := 83154 },
  { event := event83245
    frameStart := 83154 },
  { event := event83246
    frameStart := 83154 },
  { event := event83247
    frameStart := 83154 }
]

def eventLeaf5203 : Array AnnotatedEvent := #[
  { event := event83248
    frameStart := 83154 },
  { event := event83249
    frameStart := 83154 },
  { event := event83250
    frameStart := 83154 },
  { event := event83251
    frameStart := 83154 },
  { event := event83252
    frameStart := 83154 },
  { event := event83253
    frameStart := 83154 },
  { event := event83254
    frameStart := 83154 },
  { event := event83255
    frameStart := 83154 },
  { event := event83256
    frameStart := 83154 },
  { event := event83257
    frameStart := 83154 },
  { event := event83258
    frameStart := 0 },
  { event := event83259
    frameStart := 0 },
  { event := event83260
    frameStart := 0 },
  { event := event83261
    frameStart := 0 },
  { event := event83262
    frameStart := 0 },
  { event := event83263
    frameStart := 0 }
]

def eventLeaf5204 : Array AnnotatedEvent := #[
  { event := event83264
    frameStart := 0 },
  { event := event83265
    frameStart := 0 },
  { event := event83266
    frameStart := 0 },
  { event := event83267
    frameStart := 0 },
  { event := event83268
    frameStart := 0 },
  { event := event83269
    frameStart := 0 },
  { event := event83270
    frameStart := 0 },
  { event := event83271
    frameStart := 0 },
  { event := event83272
    frameStart := 0 },
  { event := event83273
    frameStart := 0 },
  { event := event83274
    frameStart := 0 },
  { event := event83275
    frameStart := 0 },
  { event := event83276
    frameStart := 0 },
  { event := event83277
    frameStart := 0 },
  { event := event83278
    frameStart := 0 },
  { event := event83279
    frameStart := 0 }
]

def eventLeaf5205 : Array AnnotatedEvent := #[
  { event := event83280
    frameStart := 0 },
  { event := event83281
    frameStart := 0 },
  { event := event83282
    frameStart := 0 },
  { event := event83283
    frameStart := 0 },
  { event := event83284
    frameStart := 0 },
  { event := event83285
    frameStart := 0 },
  { event := event83286
    frameStart := 0 },
  { event := event83287
    frameStart := 0 },
  { event := event83288
    frameStart := 0 },
  { event := event83289
    frameStart := 0 },
  { event := event83290
    frameStart := 0 },
  { event := event83291
    frameStart := 0 },
  { event := event83292
    frameStart := 0 },
  { event := event83293
    frameStart := 0 },
  { event := event83294
    frameStart := 0 },
  { event := event83295
    frameStart := 0 }
]

def eventLeaf5206 : Array AnnotatedEvent := #[
  { event := event83296
    frameStart := 0 },
  { event := event83297
    frameStart := 0 },
  { event := event83298
    frameStart := 0 },
  { event := event83299
    frameStart := 0 },
  { event := event83300
    frameStart := 0 },
  { event := event83301
    frameStart := 0 },
  { event := event83302
    frameStart := 0 },
  { event := event83303
    frameStart := 0 },
  { event := event83304
    frameStart := 0 },
  { event := event83305
    frameStart := 0 },
  { event := event83306
    frameStart := 0 },
  { event := event83307
    frameStart := 0 },
  { event := event83308
    frameStart := 0 },
  { event := event83309
    frameStart := 0 },
  { event := event83310
    frameStart := 0 },
  { event := event83311
    frameStart := 0 }
]

def eventLeaf5207 : Array AnnotatedEvent := #[
  { event := event83312
    frameStart := 0 },
  { event := event83313
    frameStart := 0 },
  { event := event83314
    frameStart := 0 },
  { event := event83315
    frameStart := 0 },
  { event := event83316
    frameStart := 0 },
  { event := event83317
    frameStart := 0 },
  { event := event83318
    frameStart := 0 },
  { event := event83319
    frameStart := 0 },
  { event := event83320
    frameStart := 0 },
  { event := event83321
    frameStart := 0 },
  { event := event83322
    frameStart := 0 },
  { event := event83323
    frameStart := 0 },
  { event := event83324
    frameStart := 0 },
  { event := event83325
    frameStart := 0 },
  { event := event83326
    frameStart := 0 },
  { event := event83327
    frameStart := 0 }
]

def eventLeaf5208 : Array AnnotatedEvent := #[
  { event := event83328
    frameStart := 0 },
  { event := event83329
    frameStart := 0 },
  { event := event83330
    frameStart := 0 },
  { event := event83331
    frameStart := 0 },
  { event := event83332
    frameStart := 0 },
  { event := event83333
    frameStart := 0 },
  { event := event83334
    frameStart := 0 },
  { event := event83335
    frameStart := 0 },
  { event := event83336
    frameStart := 0 },
  { event := event83337
    frameStart := 0 },
  { event := event83338
    frameStart := 0 },
  { event := event83339
    frameStart := 0 },
  { event := event83340
    frameStart := 0 },
  { event := event83341
    frameStart := 0 },
  { event := event83342
    frameStart := 0 },
  { event := event83343
    frameStart := 0 }
]

def eventLeaf5209 : Array AnnotatedEvent := #[
  { event := event83344
    frameStart := 0 },
  { event := event83345
    frameStart := 0 },
  { event := event83346
    frameStart := 0 },
  { event := event83347
    frameStart := 0 },
  { event := event83348
    frameStart := 0 },
  { event := event83349
    frameStart := 0 },
  { event := event83350
    frameStart := 0 },
  { event := event83351
    frameStart := 0 },
  { event := event83352
    frameStart := 0 },
  { event := event83353
    frameStart := 0 },
  { event := event83354
    frameStart := 0 },
  { event := event83355
    frameStart := 0 },
  { event := event83356
    frameStart := 0 },
  { event := event83357
    frameStart := 0 },
  { event := event83358
    frameStart := 0 },
  { event := event83359
    frameStart := 0 }
]

def eventLeaf5210 : Array AnnotatedEvent := #[
  { event := event83360
    frameStart := 0 },
  { event := event83361
    frameStart := 0 },
  { event := event83362
    frameStart := 0 },
  { event := event83363
    frameStart := 0 },
  { event := event83364
    frameStart := 0 },
  { event := event83365
    frameStart := 0 },
  { event := event83366
    frameStart := 0 },
  { event := event83367
    frameStart := 0 },
  { event := event83368
    frameStart := 0 },
  { event := event83369
    frameStart := 0 },
  { event := event83370
    frameStart := 0 },
  { event := event83371
    frameStart := 0 },
  { event := event83372
    frameStart := 0 },
  { event := event83373
    frameStart := 0 },
  { event := event83374
    frameStart := 0 },
  { event := event83375
    frameStart := 0 }
]

def eventLeaf5211 : Array AnnotatedEvent := #[
  { event := event83376
    frameStart := 0 },
  { event := event83377
    frameStart := 0 },
  { event := event83378
    frameStart := 0 },
  { event := event83379
    frameStart := 83379 },
  { event := event83380
    frameStart := 83379 },
  { event := event83381
    frameStart := 83379 },
  { event := event83382
    frameStart := 83379 },
  { event := event83383
    frameStart := 83379 },
  { event := event83384
    frameStart := 83379 },
  { event := event83385
    frameStart := 83379 },
  { event := event83386
    frameStart := 83379 },
  { event := event83387
    frameStart := 83379 },
  { event := event83388
    frameStart := 83379 },
  { event := event83389
    frameStart := 83379 },
  { event := event83390
    frameStart := 83379 },
  { event := event83391
    frameStart := 83379 }
]

def eventLeaf5212 : Array AnnotatedEvent := #[
  { event := event83392
    frameStart := 83379 },
  { event := event83393
    frameStart := 83379 },
  { event := event83394
    frameStart := 83379 },
  { event := event83395
    frameStart := 83379 },
  { event := event83396
    frameStart := 83379 },
  { event := event83397
    frameStart := 83379 },
  { event := event83398
    frameStart := 83379 },
  { event := event83399
    frameStart := 83379 },
  { event := event83400
    frameStart := 83379 },
  { event := event83401
    frameStart := 83379 },
  { event := event83402
    frameStart := 83379 },
  { event := event83403
    frameStart := 83379 },
  { event := event83404
    frameStart := 83379 },
  { event := event83405
    frameStart := 83379 },
  { event := event83406
    frameStart := 83379 },
  { event := event83407
    frameStart := 83379 }
]

def eventLeaf5213 : Array AnnotatedEvent := #[
  { event := event83408
    frameStart := 83379 },
  { event := event83409
    frameStart := 83379 },
  { event := event83410
    frameStart := 83379 },
  { event := event83411
    frameStart := 83379 },
  { event := event83412
    frameStart := 83379 },
  { event := event83413
    frameStart := 83379 },
  { event := event83414
    frameStart := 83379 },
  { event := event83415
    frameStart := 83379 },
  { event := event83416
    frameStart := 83379 },
  { event := event83417
    frameStart := 83379 },
  { event := event83418
    frameStart := 83379 },
  { event := event83419
    frameStart := 83379 },
  { event := event83420
    frameStart := 83379 },
  { event := event83421
    frameStart := 83379 },
  { event := event83422
    frameStart := 83379 },
  { event := event83423
    frameStart := 83379 }
]

def eventLeaf5214 : Array AnnotatedEvent := #[
  { event := event83424
    frameStart := 83379 },
  { event := event83425
    frameStart := 83379 },
  { event := event83426
    frameStart := 83379 },
  { event := event83427
    frameStart := 83427 },
  { event := event83428
    frameStart := 83427 },
  { event := event83429
    frameStart := 83427 },
  { event := event83430
    frameStart := 83427 },
  { event := event83431
    frameStart := 83427 },
  { event := event83432
    frameStart := 83427 },
  { event := event83433
    frameStart := 83427 },
  { event := event83434
    frameStart := 83427 },
  { event := event83435
    frameStart := 83427 },
  { event := event83436
    frameStart := 83427 },
  { event := event83437
    frameStart := 83427 },
  { event := event83438
    frameStart := 83427 },
  { event := event83439
    frameStart := 83427 }
]

def eventLeaf5215 : Array AnnotatedEvent := #[
  { event := event83440
    frameStart := 83427 },
  { event := event83441
    frameStart := 83427 },
  { event := event83442
    frameStart := 83427 },
  { event := event83443
    frameStart := 83427 },
  { event := event83444
    frameStart := 83427 },
  { event := event83445
    frameStart := 83427 },
  { event := event83446
    frameStart := 83427 },
  { event := event83447
    frameStart := 83427 },
  { event := event83448
    frameStart := 83427 },
  { event := event83449
    frameStart := 83427 },
  { event := event83450
    frameStart := 83427 },
  { event := event83451
    frameStart := 83427 },
  { event := event83452
    frameStart := 83427 },
  { event := event83453
    frameStart := 83427 },
  { event := event83454
    frameStart := 83427 },
  { event := event83455
    frameStart := 83427 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events325
