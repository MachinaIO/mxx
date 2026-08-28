import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events325

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event83200 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21195⟩⟩, .relation 83199 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (-1)⟩)

def event83201 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21195⟩⟩, .operator (⟨83192, 0⟩, ⟨24625, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩, (1)⟩)

def exact83202RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (-1)⟩]

theorem exact83202RawTermsValid :
    exact83202RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83202 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21195⟩⟩) exact83202RawTerms .large 83195 (.finite 279172874240) (some (83197))

def event83203 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21645⟩⟩) 0 ⟨21195⟩ 83202

def event83204 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21645⟩⟩) 1 ⟨21644⟩ 83172

def event83205 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21645⟩⟩) (.sum [.predecessor 0 83203 .coefficient, .predecessor 1 83204 .coefficient])

def event83206 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21645⟩⟩, .operator (⟨83202, 1⟩, ⟨83172, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (1)⟩)

def event83207 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21645⟩⟩) (.sum [.result 83202 .summary, .result 83172 .summary])

def exact83208RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83208RawTermsValid :
    exact83208RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83208 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21645⟩⟩) exact83208RawTerms .large 83205 (.finite 279176282112) (some (83207))

def event83209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23506⟩⟩) 0 ⟨21645⟩ 83208

def event83210 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23506⟩⟩) 1 ⟨23505⟩ 83144

def event83211 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23506⟩⟩) (.product (.predecessor 0 83209 .coefficient) (.predecessor 1 83210 .coefficient) (⟨false, false, none, none, none⟩))

def event83212 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23506⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩) [⟨.result 83144 .coefficient, false, none⟩])

def event83213 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23506⟩⟩) (.product (.result 83208 .summary) (.transfer 83212) (⟨false, false, none, none, none⟩))

def event83214 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23506⟩⟩, .operator (⟨83208, 1⟩, ⟨83144, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩, (-1)⟩)

def event83215 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨23506⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23505⟩⟩) ⟨22965⟩ 83141)

def event83216 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23506⟩⟩, .relation 83215 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨22965⟩⟩]⟩, (-1)⟩)

def event83217 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23506⟩⟩, .operator (⟨83208, 0⟩, ⟨83144, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩, (1)⟩)

def exact83218RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨22965⟩⟩]⟩, (-1)⟩]

theorem exact83218RawTermsValid :
    exact83218RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83218 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23506⟩⟩) exact83218RawTerms .large 83211 (.finite 2997632503724774522880) (some (83213))

def event83219 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22429⟩⟩) 0 ⟨21640⟩ 3442

def event83220 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22429⟩⟩) (.authority (.relationPreimageSource ⟨38⟩))

def exact83221RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩, (1)⟩]

theorem exact83221RawTermsValid :
    exact83221RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83221 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22429⟩⟩) exact83221RawTerms (.finite 5647228698) 83220 .exactZero (none)

def event83222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22431⟩⟩) 0 ⟨22429⟩ 83221

def event83223 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22431⟩⟩) 1 ⟨2370⟩ 4

def event83224 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22431⟩⟩) (.scale (.predecessor 0 83222 .coefficient) (.value (.predecessor 1 83223 .coefficient)))

def exact83225RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩, (1)⟩]

theorem exact83225RawTermsValid :
    exact83225RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83225 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22431⟩⟩) exact83225RawTerms (.finite 5647228698) 83224 .exactZero (none)

def event83226 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22432⟩⟩) 0 ⟨10368⟩ 75995

def event83227 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22432⟩⟩) 1 ⟨22431⟩ 83225

def event83228 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22432⟩⟩) (.product (.predecessor 0 83226 .coefficient) (.predecessor 1 83227 .coefficient) (⟨false, false, none, none, none⟩))

def event83229 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22432⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩) [⟨.result 83221 .coefficient, false, none⟩])

def event83230 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22432⟩⟩) (.product (.result 75995 .summary) (.transfer 83229) (⟨false, false, none, none, none⟩))

def event83231 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22432⟩⟩, .operator (⟨75995, 0⟩, ⟨83225, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩, (1)⟩)

def event83232 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨22430⟩⟩)

def event83233 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event83234 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event83235 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event83236 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event83237 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event83238 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event83239 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event83240 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event83241 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 83240

def event83242 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 83238

def event83243 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 83241 .coefficient) (.value (.predecessor 1 83242 .coefficient)))

def event83244 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event83245 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 83244

def event83246 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 83236

def event83247 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 83245 .coefficient, .predecessor 1 83246 .coefficient])

def event83248 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event83249 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 83248

def event83250 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 83234

def event83251 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 83250 .coefficient))

def event83252 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event83253 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21638⟩⟩) 0 ⟨10325⟩ 83252

def event83254 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21638⟩⟩) (.authority (.programFamilyFact))

def exact83255RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21638⟩⟩], []⟩, (1)⟩]

theorem exact83255RawTermsValid :
    exact83255RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83255 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21638⟩⟩) exact83255RawTerms (.finite 4) 83254 .exactZero (none)

def event83256 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21191⟩⟩) 0 ⟨10325⟩ 83252

def event83257 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21191⟩⟩) (.authority (.programFamilyFact))

def exact83258RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩], []⟩, (1)⟩]

theorem exact83258RawTermsValid :
    exact83258RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83258 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21191⟩⟩) exact83258RawTerms (.finite 4) 83257 .exactZero (none)

def event83259 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21639⟩⟩) 0 ⟨21191⟩ 83258

def event83260 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21639⟩⟩) 1 ⟨21638⟩ 83255

def event83261 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21639⟩⟩) (.product (.predecessor 0 83259 .coefficient) (.predecessor 1 83260 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event83262 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21639⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], []⟩) [⟨.result 83258 .coefficient, true, some 1⟩, ⟨.result 83255 .coefficient, true, some 1⟩])

def event83263 : Event := .survivorFold (1) 83262

def exact83264RawTerms : List Term := []

theorem exact83264RawTermsValid :
    exact83264RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83264 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21639⟩⟩) exact83264RawTerms (.finite 16) 83261 (.finite 16) (some (83262))

def event83265 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21640⟩⟩) 0 ⟨21639⟩ 83264

def event83266 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21640⟩⟩) (.identity (.predecessor 0 83265 .coefficient))

def event83267 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21640⟩⟩) (.finite 16)

def event83268 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22429⟩⟩) 0 ⟨21640⟩ 83267

def event83269 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22429⟩⟩) (.authority (.relationPreimageSource ⟨38⟩))

def exact83270RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩, (1)⟩]

theorem exact83270RawTermsValid :
    exact83270RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83270 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22429⟩⟩) exact83270RawTerms (.finite 5647228698) 83269 .exactZero (none)

def event83271 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact83272RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact83272RawTermsValid :
    exact83272RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83272 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact83272RawTerms .large 83271 .exactZero (none)

def event83273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22430⟩⟩) 0 ⟨35⟩ 83272

def event83274 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22430⟩⟩) 1 ⟨22429⟩ 83270

def event83275 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22430⟩⟩) (.product (.predecessor 0 83273 .coefficient) (.predecessor 1 83274 .coefficient) (⟨false, false, none, none, none⟩))

def event83276 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22430⟩⟩, .operator (⟨83272, 0⟩, ⟨83270, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩, (1)⟩)

def exact83277RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩, (1)⟩]

theorem exact83277RawTermsValid :
    exact83277RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83277 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22430⟩⟩) exact83277RawTerms .large 83275 .exactZero (none)

def event83278 : Event := .preFoldPolynomial 83277 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩, (1)⟩] .exactZero none

def exact83279RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩, (1)⟩]

def event83279 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨22430⟩⟩) 83278 exact83279RawTerms .large 83275 .exactZero (none)

def event83280 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨23509⟩⟩)

def event83281 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event83282 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event83283 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event83284 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event83285 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event83286 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event83287 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event83288 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event83289 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 83288

def event83290 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 83286

def event83291 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 83289 .coefficient) (.value (.predecessor 1 83290 .coefficient)))

def event83292 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event83293 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 83292

def event83294 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 83284

def event83295 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 83293 .coefficient, .predecessor 1 83294 .coefficient])

def event83296 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event83297 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 83296

def event83298 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 83282

def event83299 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 83298 .coefficient))

def event83300 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event83301 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21638⟩⟩) 0 ⟨10325⟩ 83300

def event83302 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21638⟩⟩) (.authority (.programFamilyFact))

def exact83303RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21638⟩⟩], []⟩, (1)⟩]

theorem exact83303RawTermsValid :
    exact83303RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83303 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21638⟩⟩) exact83303RawTerms (.finite 4) 83302 .exactZero (none)

def event83304 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21191⟩⟩) 0 ⟨10325⟩ 83300

def event83305 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21191⟩⟩) (.authority (.programFamilyFact))

def exact83306RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩], []⟩, (1)⟩]

theorem exact83306RawTermsValid :
    exact83306RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83306 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21191⟩⟩) exact83306RawTerms (.finite 4) 83305 .exactZero (none)

def event83307 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21639⟩⟩) 0 ⟨21191⟩ 83306

def event83308 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21639⟩⟩) 1 ⟨21638⟩ 83303

def event83309 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21639⟩⟩) (.product (.predecessor 0 83307 .coefficient) (.predecessor 1 83308 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event83310 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21639⟩⟩, .operator (⟨83306, 0⟩, ⟨83303, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], []⟩, (1)⟩)

def exact83311RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], []⟩, (1)⟩]

theorem exact83311RawTermsValid :
    exact83311RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83311 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21639⟩⟩) exact83311RawTerms (.finite 16) 83309 .exactZero (none)

def event83312 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21640⟩⟩) 0 ⟨21639⟩ 83311

def event83313 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21640⟩⟩) (.identity (.predecessor 0 83312 .coefficient))

def event83314 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21640⟩⟩) (.finite 16)

def event83315 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22964⟩⟩) 0 ⟨21640⟩ 83314

def event83316 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22964⟩⟩) (.authority (.programFamilyFact))

def event83317 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨22964⟩⟩) (.finite 3720)

def event83318 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event83319 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22965⟩⟩) 0 ⟨7177⟩ 83318

def event83320 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22965⟩⟩) 1 ⟨22964⟩ 83317

def event83321 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22965⟩⟩) (.authority (.operator))

def exact83322RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22965⟩⟩]⟩, (1)⟩]

theorem exact83322RawTermsValid :
    exact83322RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83322 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22965⟩⟩) exact83322RawTerms .large 83321 .exactZero (none)

def event83323 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23505⟩⟩) 0 ⟨22965⟩ 83322

def event83324 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23505⟩⟩) (.authority (.operator))

def exact83325RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩, (1)⟩]

theorem exact83325RawTermsValid :
    exact83325RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83325 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23505⟩⟩) exact83325RawTerms (.finite 8192) 83324 .exactZero (none)

def event83326 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event83327 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event83328 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23230⟩⟩) 0 ⟨21640⟩ 83314

def event83329 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23230⟩⟩) 1 ⟨136⟩ 83327

def event83330 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23230⟩⟩) (.sum [.predecessor 0 83328 .coefficient, .predecessor 1 83329 .coefficient])

def event83331 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨23230⟩⟩) (.finite 16)

def event83332 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23231⟩⟩) 0 ⟨23230⟩ 83331

def event83333 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23231⟩⟩) (.identity (.predecessor 0 83332 .coefficient))

def exact83334RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], []⟩, (1)⟩]

theorem exact83334RawTermsValid :
    exact83334RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83334 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23231⟩⟩) exact83334RawTerms (.finite 16) 83333 .exactZero (none)

def event83335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact83336RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact83336RawTermsValid :
    exact83336RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83336 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact83336RawTerms .large 83335 .exactZero (none)

def event83337 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23232⟩⟩) 0 ⟨6908⟩ 83336

def event83338 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23232⟩⟩) 1 ⟨23231⟩ 83334

def event83339 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23232⟩⟩) (.product (.predecessor 0 83337 .coefficient) (.predecessor 1 83338 .coefficient) (⟨false, false, none, none, none⟩))

def event83340 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23232⟩⟩, .operator (⟨83336, 0⟩, ⟨83334, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact83341RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact83341RawTermsValid :
    exact83341RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83341 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23232⟩⟩) exact83341RawTerms .large 83339 .exactZero (none)

def event83342 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event83343 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event83344 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 83318

def event83345 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact83346RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact83346RawTermsValid :
    exact83346RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83346 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact83346RawTerms .large 83345 .exactZero (none)

def event83347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7306⟩⟩) 0 ⟨7178⟩ 83346

def event83348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7306⟩⟩) (.identity (.predecessor 0 83347 .coefficient))

def exact83349RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (1)⟩]

theorem exact83349RawTermsValid :
    exact83349RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83349 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7306⟩⟩) exact83349RawTerms .large 83348 .exactZero (none)

def event83350 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9574⟩⟩) 0 ⟨7306⟩ 83349

def event83351 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9574⟩⟩) (.authority (.operator))

def exact83352RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩, (1)⟩]

theorem exact83352RawTermsValid :
    exact83352RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83352 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9574⟩⟩) exact83352RawTerms (.finite 8192) 83351 .exactZero (none)

def event83353 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9575⟩⟩) 0 ⟨9574⟩ 83352

def event83354 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9575⟩⟩) 1 ⟨2370⟩ 83343

def event83355 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9575⟩⟩) (.scale (.predecessor 0 83353 .coefficient) (.value (.predecessor 1 83354 .coefficient)))

def exact83356RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩, (1)⟩]

theorem exact83356RawTermsValid :
    exact83356RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83356 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9575⟩⟩) exact83356RawTerms (.finite 8192) 83355 .exactZero (none)

def event83357 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7286⟩⟩) 0 ⟨7178⟩ 83346

def event83358 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7286⟩⟩) (.identity (.predecessor 0 83357 .coefficient))

def exact83359RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩]⟩, (1)⟩]

theorem exact83359RawTermsValid :
    exact83359RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83359 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7286⟩⟩) exact83359RawTerms .large 83358 .exactZero (none)

def event83360 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9576⟩⟩) 0 ⟨7286⟩ 83359

def event83361 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9576⟩⟩) 1 ⟨9575⟩ 83356

def event83362 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9576⟩⟩) (.product (.predecessor 0 83360 .coefficient) (.predecessor 1 83361 .coefficient) (⟨false, false, none, none, none⟩))

def event83363 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9576⟩⟩, .operator (⟨83359, 0⟩, ⟨83356, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩, (1)⟩)

def exact83364RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩, (1)⟩]

theorem exact83364RawTermsValid :
    exact83364RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83364 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9576⟩⟩) exact83364RawTerms .large 83362 .exactZero (none)

def event83365 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23233⟩⟩) 0 ⟨9576⟩ 83364

def event83366 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23233⟩⟩) 1 ⟨23232⟩ 83341

def event83367 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23233⟩⟩) (.sum [.predecessor 0 83365 .coefficient, .predecessor 1 83366 .coefficient])

def exact83368RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83368RawTermsValid :
    exact83368RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83368 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23233⟩⟩) exact83368RawTerms .large 83367 .exactZero (none)

def event83369 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23508⟩⟩) 0 ⟨23233⟩ 83368

def event83370 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23508⟩⟩) 1 ⟨23505⟩ 83325

def event83371 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23508⟩⟩) (.product (.predecessor 0 83369 .coefficient) (.predecessor 1 83370 .coefficient) (⟨false, false, none, none, none⟩))

def event83372 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23508⟩⟩, .operator (⟨83368, 0⟩, ⟨83325, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩, (1)⟩)

def event83373 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23508⟩⟩, .operator (⟨83368, 1⟩, ⟨83325, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩, (-1)⟩)

def event83374 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨23508⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23505⟩⟩) ⟨22965⟩ 83322)

def event83375 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23508⟩⟩, .relation 83374 0, ⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨22965⟩⟩]⟩, (-1)⟩)

def exact83376RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨22965⟩⟩]⟩, (-1)⟩]

theorem exact83376RawTermsValid :
    exact83376RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83376 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23508⟩⟩) exact83376RawTerms .large 83371 .exactZero (none)

def event83377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21856⟩⟩) 0 ⟨21640⟩ 83314

def event83378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21856⟩⟩) (.authority (.programFamilyFact))

def exact83379RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21856⟩⟩], []⟩, (1)⟩]

theorem exact83379RawTermsValid :
    exact83379RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83379 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21856⟩⟩) exact83379RawTerms (.finite 4) 83378 .exactZero (none)

def event83380 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21858⟩⟩) 0 ⟨6908⟩ 83336

def event83381 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21858⟩⟩) 1 ⟨21856⟩ 83379

def event83382 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21858⟩⟩) (.product (.predecessor 0 83380 .coefficient) (.predecessor 1 83381 .coefficient) (⟨false, true, none, none, some 1⟩))

def event83383 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21858⟩⟩, .operator (⟨83336, 0⟩, ⟨83379, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact83384RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact83384RawTermsValid :
    exact83384RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83384 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21858⟩⟩) exact83384RawTerms .large 83382 .exactZero (none)

def event83385 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7181⟩⟩) 0 ⟨7177⟩ 83318

def event83386 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7181⟩⟩) (.authority (.operator))

def exact83387RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩]

theorem exact83387RawTermsValid :
    exact83387RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83387 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7181⟩⟩) exact83387RawTerms .large 83386 .exactZero (none)

def event83388 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21859⟩⟩) 0 ⟨7181⟩ 83387

def event83389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21859⟩⟩) 1 ⟨21858⟩ 83384

def event83390 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21859⟩⟩) (.sum [.predecessor 0 83388 .coefficient, .predecessor 1 83389 .coefficient])

def exact83391RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83391RawTermsValid :
    exact83391RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83391 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21859⟩⟩) exact83391RawTerms .large 83390 .exactZero (none)

def event83392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23509⟩⟩) 0 ⟨21859⟩ 83391

def event83393 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23509⟩⟩) 1 ⟨23508⟩ 83376

def event83394 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23509⟩⟩) (.sum [.predecessor 0 83392 .coefficient, .predecessor 1 83393 .coefficient])

def exact83395RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨22965⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83395RawTermsValid :
    exact83395RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83395 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23509⟩⟩) exact83395RawTerms .large 83394 .exactZero (none)

def event83396 : Event := .preFoldPolynomial 83395 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨22965⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact83397RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨22965⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event83397 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨23509⟩⟩) 83396 exact83397RawTerms .large 83394 .exactZero (none)

def event83398 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨21640⟩⟩) ⟨⟨60⟩, ⟨38⟩, ⟨135⟩⟩ ⟨83232, 83398⟩

def event83399 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨22432⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩) (1) 0 2 (.universal 83398 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩) (none) 83397)

def event83400 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22432⟩⟩, .relation 83399 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩)

def event83401 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22432⟩⟩, .relation 83399 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩, (-1)⟩)

def event83402 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22432⟩⟩, .relation 83399 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨22965⟩⟩]⟩, (1)⟩)

def event83403 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22432⟩⟩, .relation 83399 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact83404RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨22965⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83404RawTermsValid :
    exact83404RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83404 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22432⟩⟩) exact83404RawTerms .large 83228 (.finite 202072841853861888) (some (83230))

def event83405 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23507⟩⟩) 0 ⟨22432⟩ 83404

def event83406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23507⟩⟩) 1 ⟨23506⟩ 83218

def event83407 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23507⟩⟩) (.sum [.predecessor 0 83405 .coefficient, .predecessor 1 83406 .coefficient])

def event83408 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23507⟩⟩, .operator (⟨83404, 2⟩, ⟨83218, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨22965⟩⟩]⟩, (-1)⟩)

def event83409 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23507⟩⟩, .operator (⟨83404, 1⟩, ⟨83218, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩, (1)⟩)

def event83410 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23507⟩⟩) (.sum [.result 83404 .summary, .result 83218 .summary])

def exact83411RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83411RawTermsValid :
    exact83411RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83411 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23507⟩⟩) exact83411RawTerms .large 83407 (.finite 2997834576566628384768) (some (83410))

def event83412 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24060⟩⟩) 0 ⟨23507⟩ 83411

def event83413 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24060⟩⟩) 1 ⟨24058⟩ 83134

def event83414 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24060⟩⟩) (.product (.predecessor 0 83412 .coefficient) (.predecessor 1 83413 .coefficient) (⟨false, false, none, none, none⟩))

def event83415 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24060⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨24058⟩⟩]⟩) [⟨.result 83134 .coefficient, false, none⟩])

def event83416 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24060⟩⟩) (.product (.result 83411 .summary) (.transfer 83415) (⟨false, false, none, none, none⟩))

def event83417 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24060⟩⟩, .operator (⟨83411, 0⟩, ⟨83134, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩]⟩, (1)⟩)

def event83418 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24060⟩⟩, .operator (⟨83411, 1⟩, ⟨83134, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩]⟩, (-1)⟩)

def event83419 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨24060⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨24058⟩⟩) ⟨23135⟩ 83131)

def event83420 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24060⟩⟩, .relation 83419 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨23135⟩⟩]⟩, (-1)⟩)

def exact83421RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨23135⟩⟩]⟩, (-1)⟩]

theorem exact83421RawTermsValid :
    exact83421RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83421 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24060⟩⟩) exact83421RawTerms .large 83414 (.finite 32189003662929192193909661368320) (some (83416))

def event83422 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22796⟩⟩) 0 ⟨21857⟩ 3448

def event83423 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22796⟩⟩) (.authority (.relationPreimageSource ⟨61⟩))

def exact83424RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22796⟩⟩]⟩, (1)⟩]

theorem exact83424RawTermsValid :
    exact83424RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83424 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22796⟩⟩) exact83424RawTerms (.finite 5647228698) 83423 .exactZero (none)

def event83425 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22798⟩⟩) 0 ⟨22796⟩ 83424

def event83426 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22798⟩⟩) 1 ⟨2370⟩ 4

def event83427 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22798⟩⟩) (.scale (.predecessor 0 83425 .coefficient) (.value (.predecessor 1 83426 .coefficient)))

def exact83428RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22796⟩⟩]⟩, (1)⟩]

theorem exact83428RawTermsValid :
    exact83428RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83428 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22798⟩⟩) exact83428RawTerms (.finite 5647228698) 83427 .exactZero (none)

def event83429 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22799⟩⟩) 0 ⟨10368⟩ 75995

def event83430 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22799⟩⟩) 1 ⟨22798⟩ 83428

def event83431 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22799⟩⟩) (.product (.predecessor 0 83429 .coefficient) (.predecessor 1 83430 .coefficient) (⟨false, false, none, none, none⟩))

def event83432 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22799⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨22796⟩⟩]⟩) [⟨.result 83424 .coefficient, false, none⟩])

def event83433 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22799⟩⟩) (.product (.result 75995 .summary) (.transfer 83432) (⟨false, false, none, none, none⟩))

def event83434 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22799⟩⟩, .operator (⟨75995, 0⟩, ⟨83428, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22796⟩⟩]⟩, (1)⟩)

def event83435 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨22797⟩⟩)

def event83436 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event83437 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event83438 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event83439 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event83440 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event83441 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event83442 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event83443 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event83444 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 83443

def event83445 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 83441

def event83446 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 83444 .coefficient) (.value (.predecessor 1 83445 .coefficient)))

def event83447 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event83448 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 83447

def event83449 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 83439

def event83450 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 83448 .coefficient, .predecessor 1 83449 .coefficient])

def event83451 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event83452 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 83451

def event83453 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 83437

def event83454 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 83453 .coefficient))

def event83455 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def eventLeaf5200 : Array AnnotatedEvent := #[
  { event := event83200
    frameStart := 0 },
  { event := event83201
    frameStart := 0 },
  { event := event83202
    frameStart := 0 },
  { event := event83203
    frameStart := 0 },
  { event := event83204
    frameStart := 0 },
  { event := event83205
    frameStart := 0 },
  { event := event83206
    frameStart := 0 },
  { event := event83207
    frameStart := 0 },
  { event := event83208
    frameStart := 0 },
  { event := event83209
    frameStart := 0 },
  { event := event83210
    frameStart := 0 },
  { event := event83211
    frameStart := 0 },
  { event := event83212
    frameStart := 0 },
  { event := event83213
    frameStart := 0 },
  { event := event83214
    frameStart := 0 },
  { event := event83215
    frameStart := 0 }
]

def eventLeaf5201 : Array AnnotatedEvent := #[
  { event := event83216
    frameStart := 0 },
  { event := event83217
    frameStart := 0 },
  { event := event83218
    frameStart := 0 },
  { event := event83219
    frameStart := 0 },
  { event := event83220
    frameStart := 0 },
  { event := event83221
    frameStart := 0 },
  { event := event83222
    frameStart := 0 },
  { event := event83223
    frameStart := 0 },
  { event := event83224
    frameStart := 0 },
  { event := event83225
    frameStart := 0 },
  { event := event83226
    frameStart := 0 },
  { event := event83227
    frameStart := 0 },
  { event := event83228
    frameStart := 0 },
  { event := event83229
    frameStart := 0 },
  { event := event83230
    frameStart := 0 },
  { event := event83231
    frameStart := 0 }
]

def eventLeaf5202 : Array AnnotatedEvent := #[
  { event := event83232
    frameStart := 83232 },
  { event := event83233
    frameStart := 83232 },
  { event := event83234
    frameStart := 83232 },
  { event := event83235
    frameStart := 83232 },
  { event := event83236
    frameStart := 83232 },
  { event := event83237
    frameStart := 83232 },
  { event := event83238
    frameStart := 83232 },
  { event := event83239
    frameStart := 83232 },
  { event := event83240
    frameStart := 83232 },
  { event := event83241
    frameStart := 83232 },
  { event := event83242
    frameStart := 83232 },
  { event := event83243
    frameStart := 83232 },
  { event := event83244
    frameStart := 83232 },
  { event := event83245
    frameStart := 83232 },
  { event := event83246
    frameStart := 83232 },
  { event := event83247
    frameStart := 83232 }
]

def eventLeaf5203 : Array AnnotatedEvent := #[
  { event := event83248
    frameStart := 83232 },
  { event := event83249
    frameStart := 83232 },
  { event := event83250
    frameStart := 83232 },
  { event := event83251
    frameStart := 83232 },
  { event := event83252
    frameStart := 83232 },
  { event := event83253
    frameStart := 83232 },
  { event := event83254
    frameStart := 83232 },
  { event := event83255
    frameStart := 83232 },
  { event := event83256
    frameStart := 83232 },
  { event := event83257
    frameStart := 83232 },
  { event := event83258
    frameStart := 83232 },
  { event := event83259
    frameStart := 83232 },
  { event := event83260
    frameStart := 83232 },
  { event := event83261
    frameStart := 83232 },
  { event := event83262
    frameStart := 83232 },
  { event := event83263
    frameStart := 83232 }
]

def eventLeaf5204 : Array AnnotatedEvent := #[
  { event := event83264
    frameStart := 83232 },
  { event := event83265
    frameStart := 83232 },
  { event := event83266
    frameStart := 83232 },
  { event := event83267
    frameStart := 83232 },
  { event := event83268
    frameStart := 83232 },
  { event := event83269
    frameStart := 83232 },
  { event := event83270
    frameStart := 83232 },
  { event := event83271
    frameStart := 83232 },
  { event := event83272
    frameStart := 83232 },
  { event := event83273
    frameStart := 83232 },
  { event := event83274
    frameStart := 83232 },
  { event := event83275
    frameStart := 83232 },
  { event := event83276
    frameStart := 83232 },
  { event := event83277
    frameStart := 83232 },
  { event := event83278
    frameStart := 83232 },
  { event := event83279
    frameStart := 83232 }
]

def eventLeaf5205 : Array AnnotatedEvent := #[
  { event := event83280
    frameStart := 83280 },
  { event := event83281
    frameStart := 83280 },
  { event := event83282
    frameStart := 83280 },
  { event := event83283
    frameStart := 83280 },
  { event := event83284
    frameStart := 83280 },
  { event := event83285
    frameStart := 83280 },
  { event := event83286
    frameStart := 83280 },
  { event := event83287
    frameStart := 83280 },
  { event := event83288
    frameStart := 83280 },
  { event := event83289
    frameStart := 83280 },
  { event := event83290
    frameStart := 83280 },
  { event := event83291
    frameStart := 83280 },
  { event := event83292
    frameStart := 83280 },
  { event := event83293
    frameStart := 83280 },
  { event := event83294
    frameStart := 83280 },
  { event := event83295
    frameStart := 83280 }
]

def eventLeaf5206 : Array AnnotatedEvent := #[
  { event := event83296
    frameStart := 83280 },
  { event := event83297
    frameStart := 83280 },
  { event := event83298
    frameStart := 83280 },
  { event := event83299
    frameStart := 83280 },
  { event := event83300
    frameStart := 83280 },
  { event := event83301
    frameStart := 83280 },
  { event := event83302
    frameStart := 83280 },
  { event := event83303
    frameStart := 83280 },
  { event := event83304
    frameStart := 83280 },
  { event := event83305
    frameStart := 83280 },
  { event := event83306
    frameStart := 83280 },
  { event := event83307
    frameStart := 83280 },
  { event := event83308
    frameStart := 83280 },
  { event := event83309
    frameStart := 83280 },
  { event := event83310
    frameStart := 83280 },
  { event := event83311
    frameStart := 83280 }
]

def eventLeaf5207 : Array AnnotatedEvent := #[
  { event := event83312
    frameStart := 83280 },
  { event := event83313
    frameStart := 83280 },
  { event := event83314
    frameStart := 83280 },
  { event := event83315
    frameStart := 83280 },
  { event := event83316
    frameStart := 83280 },
  { event := event83317
    frameStart := 83280 },
  { event := event83318
    frameStart := 83280 },
  { event := event83319
    frameStart := 83280 },
  { event := event83320
    frameStart := 83280 },
  { event := event83321
    frameStart := 83280 },
  { event := event83322
    frameStart := 83280 },
  { event := event83323
    frameStart := 83280 },
  { event := event83324
    frameStart := 83280 },
  { event := event83325
    frameStart := 83280 },
  { event := event83326
    frameStart := 83280 },
  { event := event83327
    frameStart := 83280 }
]

def eventLeaf5208 : Array AnnotatedEvent := #[
  { event := event83328
    frameStart := 83280 },
  { event := event83329
    frameStart := 83280 },
  { event := event83330
    frameStart := 83280 },
  { event := event83331
    frameStart := 83280 },
  { event := event83332
    frameStart := 83280 },
  { event := event83333
    frameStart := 83280 },
  { event := event83334
    frameStart := 83280 },
  { event := event83335
    frameStart := 83280 },
  { event := event83336
    frameStart := 83280 },
  { event := event83337
    frameStart := 83280 },
  { event := event83338
    frameStart := 83280 },
  { event := event83339
    frameStart := 83280 },
  { event := event83340
    frameStart := 83280 },
  { event := event83341
    frameStart := 83280 },
  { event := event83342
    frameStart := 83280 },
  { event := event83343
    frameStart := 83280 }
]

def eventLeaf5209 : Array AnnotatedEvent := #[
  { event := event83344
    frameStart := 83280 },
  { event := event83345
    frameStart := 83280 },
  { event := event83346
    frameStart := 83280 },
  { event := event83347
    frameStart := 83280 },
  { event := event83348
    frameStart := 83280 },
  { event := event83349
    frameStart := 83280 },
  { event := event83350
    frameStart := 83280 },
  { event := event83351
    frameStart := 83280 },
  { event := event83352
    frameStart := 83280 },
  { event := event83353
    frameStart := 83280 },
  { event := event83354
    frameStart := 83280 },
  { event := event83355
    frameStart := 83280 },
  { event := event83356
    frameStart := 83280 },
  { event := event83357
    frameStart := 83280 },
  { event := event83358
    frameStart := 83280 },
  { event := event83359
    frameStart := 83280 }
]

def eventLeaf5210 : Array AnnotatedEvent := #[
  { event := event83360
    frameStart := 83280 },
  { event := event83361
    frameStart := 83280 },
  { event := event83362
    frameStart := 83280 },
  { event := event83363
    frameStart := 83280 },
  { event := event83364
    frameStart := 83280 },
  { event := event83365
    frameStart := 83280 },
  { event := event83366
    frameStart := 83280 },
  { event := event83367
    frameStart := 83280 },
  { event := event83368
    frameStart := 83280 },
  { event := event83369
    frameStart := 83280 },
  { event := event83370
    frameStart := 83280 },
  { event := event83371
    frameStart := 83280 },
  { event := event83372
    frameStart := 83280 },
  { event := event83373
    frameStart := 83280 },
  { event := event83374
    frameStart := 83280 },
  { event := event83375
    frameStart := 83280 }
]

def eventLeaf5211 : Array AnnotatedEvent := #[
  { event := event83376
    frameStart := 83280 },
  { event := event83377
    frameStart := 83280 },
  { event := event83378
    frameStart := 83280 },
  { event := event83379
    frameStart := 83280 },
  { event := event83380
    frameStart := 83280 },
  { event := event83381
    frameStart := 83280 },
  { event := event83382
    frameStart := 83280 },
  { event := event83383
    frameStart := 83280 },
  { event := event83384
    frameStart := 83280 },
  { event := event83385
    frameStart := 83280 },
  { event := event83386
    frameStart := 83280 },
  { event := event83387
    frameStart := 83280 },
  { event := event83388
    frameStart := 83280 },
  { event := event83389
    frameStart := 83280 },
  { event := event83390
    frameStart := 83280 },
  { event := event83391
    frameStart := 83280 }
]

def eventLeaf5212 : Array AnnotatedEvent := #[
  { event := event83392
    frameStart := 83280 },
  { event := event83393
    frameStart := 83280 },
  { event := event83394
    frameStart := 83280 },
  { event := event83395
    frameStart := 83280 },
  { event := event83396
    frameStart := 83280 },
  { event := event83397
    frameStart := 83280 },
  { event := event83398
    frameStart := 0 },
  { event := event83399
    frameStart := 0 },
  { event := event83400
    frameStart := 0 },
  { event := event83401
    frameStart := 0 },
  { event := event83402
    frameStart := 0 },
  { event := event83403
    frameStart := 0 },
  { event := event83404
    frameStart := 0 },
  { event := event83405
    frameStart := 0 },
  { event := event83406
    frameStart := 0 },
  { event := event83407
    frameStart := 0 }
]

def eventLeaf5213 : Array AnnotatedEvent := #[
  { event := event83408
    frameStart := 0 },
  { event := event83409
    frameStart := 0 },
  { event := event83410
    frameStart := 0 },
  { event := event83411
    frameStart := 0 },
  { event := event83412
    frameStart := 0 },
  { event := event83413
    frameStart := 0 },
  { event := event83414
    frameStart := 0 },
  { event := event83415
    frameStart := 0 },
  { event := event83416
    frameStart := 0 },
  { event := event83417
    frameStart := 0 },
  { event := event83418
    frameStart := 0 },
  { event := event83419
    frameStart := 0 },
  { event := event83420
    frameStart := 0 },
  { event := event83421
    frameStart := 0 },
  { event := event83422
    frameStart := 0 },
  { event := event83423
    frameStart := 0 }
]

def eventLeaf5214 : Array AnnotatedEvent := #[
  { event := event83424
    frameStart := 0 },
  { event := event83425
    frameStart := 0 },
  { event := event83426
    frameStart := 0 },
  { event := event83427
    frameStart := 0 },
  { event := event83428
    frameStart := 0 },
  { event := event83429
    frameStart := 0 },
  { event := event83430
    frameStart := 0 },
  { event := event83431
    frameStart := 0 },
  { event := event83432
    frameStart := 0 },
  { event := event83433
    frameStart := 0 },
  { event := event83434
    frameStart := 0 },
  { event := event83435
    frameStart := 83435 },
  { event := event83436
    frameStart := 83435 },
  { event := event83437
    frameStart := 83435 },
  { event := event83438
    frameStart := 83435 },
  { event := event83439
    frameStart := 83435 }
]

def eventLeaf5215 : Array AnnotatedEvent := #[
  { event := event83440
    frameStart := 83435 },
  { event := event83441
    frameStart := 83435 },
  { event := event83442
    frameStart := 83435 },
  { event := event83443
    frameStart := 83435 },
  { event := event83444
    frameStart := 83435 },
  { event := event83445
    frameStart := 83435 },
  { event := event83446
    frameStart := 83435 },
  { event := event83447
    frameStart := 83435 },
  { event := event83448
    frameStart := 83435 },
  { event := event83449
    frameStart := 83435 },
  { event := event83450
    frameStart := 83435 },
  { event := event83451
    frameStart := 83435 },
  { event := event83452
    frameStart := 83435 },
  { event := event83453
    frameStart := 83435 },
  { event := event83454
    frameStart := 83435 },
  { event := event83455
    frameStart := 83435 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events325
