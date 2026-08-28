import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events329

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event84224 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15619⟩⟩) 1 ⟨15618⟩ 84219

def event84225 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15619⟩⟩) (.product (.predecessor 0 84223 .coefficient) (.predecessor 1 84224 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event84226 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15619⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], []⟩) [⟨.result 84222 .coefficient, true, some 1⟩, ⟨.result 84219 .coefficient, true, some 1⟩])

def event84227 : Event := .survivorFold (1) 84226

def exact84228RawTerms : List Term := []

theorem exact84228RawTermsValid :
    exact84228RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84228 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15619⟩⟩) exact84228RawTerms (.finite 4) 84225 (.finite 4) (some (84226))

def event84229 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15620⟩⟩) 0 ⟨15619⟩ 84228

def event84230 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15620⟩⟩) (.identity (.predecessor 0 84229 .coefficient))

def event84231 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15620⟩⟩) (.finite 4)

def event84232 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16349⟩⟩) 0 ⟨15620⟩ 84231

def event84233 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16349⟩⟩) (.authority (.relationPreimageSource ⟨36⟩))

def exact84234RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16349⟩⟩]⟩, (1)⟩]

theorem exact84234RawTermsValid :
    exact84234RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84234 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16349⟩⟩) exact84234RawTerms (.finite 5647228698) 84233 .exactZero (none)

def event84235 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact84236RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact84236RawTermsValid :
    exact84236RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84236 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact84236RawTerms .large 84235 .exactZero (none)

def event84237 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16350⟩⟩) 0 ⟨35⟩ 84236

def event84238 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16350⟩⟩) 1 ⟨16349⟩ 84234

def event84239 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16350⟩⟩) (.product (.predecessor 0 84237 .coefficient) (.predecessor 1 84238 .coefficient) (⟨false, false, none, none, none⟩))

def event84240 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16350⟩⟩, .operator (⟨84236, 0⟩, ⟨84234, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16349⟩⟩]⟩, (1)⟩)

def exact84241RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16349⟩⟩]⟩, (1)⟩]

theorem exact84241RawTermsValid :
    exact84241RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84241 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16350⟩⟩) exact84241RawTerms .large 84239 .exactZero (none)

def event84242 : Event := .preFoldPolynomial 84241 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16349⟩⟩]⟩, (1)⟩] .exactZero none

def exact84243RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16349⟩⟩]⟩, (1)⟩]

def event84243 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨16350⟩⟩) 84242 exact84243RawTerms .large 84239 .exactZero (none)

def event84244 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨17429⟩⟩)

def event84245 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event84246 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event84247 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event84248 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event84249 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event84250 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event84251 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event84252 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event84253 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 84252

def event84254 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 84250

def event84255 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 84253 .coefficient) (.value (.predecessor 1 84254 .coefficient)))

def event84256 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event84257 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 84256

def event84258 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 84248

def event84259 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 84257 .coefficient, .predecessor 1 84258 .coefficient])

def event84260 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event84261 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 84260

def event84262 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 84246

def event84263 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 84262 .coefficient))

def event84264 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event84265 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15618⟩⟩) 0 ⟨10325⟩ 84264

def event84266 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15618⟩⟩) (.authority (.programFamilyFact))

def exact84267RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15618⟩⟩], []⟩, (1)⟩]

theorem exact84267RawTermsValid :
    exact84267RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84267 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15618⟩⟩) exact84267RawTerms (.finite 2) 84266 .exactZero (none)

def event84268 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12471⟩⟩) 0 ⟨10325⟩ 84264

def event84269 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12471⟩⟩) (.authority (.programFamilyFact))

def exact84270RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩], []⟩, (1)⟩]

theorem exact84270RawTermsValid :
    exact84270RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84270 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12471⟩⟩) exact84270RawTerms (.finite 2) 84269 .exactZero (none)

def event84271 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15619⟩⟩) 0 ⟨12471⟩ 84270

def event84272 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15619⟩⟩) 1 ⟨15618⟩ 84267

def event84273 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15619⟩⟩) (.product (.predecessor 0 84271 .coefficient) (.predecessor 1 84272 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event84274 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15619⟩⟩, .operator (⟨84270, 0⟩, ⟨84267, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], []⟩, (1)⟩)

def exact84275RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], []⟩, (1)⟩]

theorem exact84275RawTermsValid :
    exact84275RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84275 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15619⟩⟩) exact84275RawTerms (.finite 4) 84273 .exactZero (none)

def event84276 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15620⟩⟩) 0 ⟨15619⟩ 84275

def event84277 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15620⟩⟩) (.identity (.predecessor 0 84276 .coefficient))

def event84278 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15620⟩⟩) (.finite 4)

def event84279 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16884⟩⟩) 0 ⟨15620⟩ 84278

def event84280 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16884⟩⟩) (.authority (.programFamilyFact))

def event84281 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨16884⟩⟩) (.finite 3720)

def event84282 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event84283 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16885⟩⟩) 0 ⟨7177⟩ 84282

def event84284 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16885⟩⟩) 1 ⟨16884⟩ 84281

def event84285 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16885⟩⟩) (.authority (.operator))

def exact84286RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16885⟩⟩]⟩, (1)⟩]

theorem exact84286RawTermsValid :
    exact84286RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84286 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16885⟩⟩) exact84286RawTerms .large 84285 .exactZero (none)

def event84287 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17425⟩⟩) 0 ⟨16885⟩ 84286

def event84288 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17425⟩⟩) (.authority (.operator))

def exact84289RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17425⟩⟩]⟩, (1)⟩]

theorem exact84289RawTermsValid :
    exact84289RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84289 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17425⟩⟩) exact84289RawTerms (.finite 8192) 84288 .exactZero (none)

def event84290 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event84291 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event84292 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17150⟩⟩) 0 ⟨15620⟩ 84278

def event84293 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17150⟩⟩) 1 ⟨136⟩ 84291

def event84294 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17150⟩⟩) (.sum [.predecessor 0 84292 .coefficient, .predecessor 1 84293 .coefficient])

def event84295 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17150⟩⟩) (.finite 4)

def event84296 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17151⟩⟩) 0 ⟨17150⟩ 84295

def event84297 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17151⟩⟩) (.identity (.predecessor 0 84296 .coefficient))

def exact84298RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], []⟩, (1)⟩]

theorem exact84298RawTermsValid :
    exact84298RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84298 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17151⟩⟩) exact84298RawTerms (.finite 4) 84297 .exactZero (none)

def event84299 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact84300RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact84300RawTermsValid :
    exact84300RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84300 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact84300RawTerms .large 84299 .exactZero (none)

def event84301 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17152⟩⟩) 0 ⟨6908⟩ 84300

def event84302 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17152⟩⟩) 1 ⟨17151⟩ 84298

def event84303 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17152⟩⟩) (.product (.predecessor 0 84301 .coefficient) (.predecessor 1 84302 .coefficient) (⟨false, false, none, none, none⟩))

def event84304 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17152⟩⟩, .operator (⟨84300, 0⟩, ⟨84298, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact84305RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact84305RawTermsValid :
    exact84305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84305 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17152⟩⟩) exact84305RawTerms .large 84303 .exactZero (none)

def event84306 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event84307 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event84308 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 84282

def event84309 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact84310RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact84310RawTermsValid :
    exact84310RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84310 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact84310RawTerms .large 84309 .exactZero (none)

def event84311 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7304⟩⟩) 0 ⟨7178⟩ 84310

def event84312 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7304⟩⟩) (.identity (.predecessor 0 84311 .coefficient))

def exact84313RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7304⟩⟩]⟩, (1)⟩]

theorem exact84313RawTermsValid :
    exact84313RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84313 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7304⟩⟩) exact84313RawTerms .large 84312 .exactZero (none)

def event84314 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9568⟩⟩) 0 ⟨7304⟩ 84313

def event84315 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9568⟩⟩) (.authority (.operator))

def exact84316RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩]

theorem exact84316RawTermsValid :
    exact84316RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84316 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9568⟩⟩) exact84316RawTerms (.finite 8192) 84315 .exactZero (none)

def event84317 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9569⟩⟩) 0 ⟨9568⟩ 84316

def event84318 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9569⟩⟩) 1 ⟨2370⟩ 84307

def event84319 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9569⟩⟩) (.scale (.predecessor 0 84317 .coefficient) (.value (.predecessor 1 84318 .coefficient)))

def exact84320RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩]

theorem exact84320RawTermsValid :
    exact84320RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84320 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9569⟩⟩) exact84320RawTerms (.finite 8192) 84319 .exactZero (none)

def event84321 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7303⟩⟩) 0 ⟨7178⟩ 84310

def event84322 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7303⟩⟩) (.identity (.predecessor 0 84321 .coefficient))

def exact84323RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩]⟩, (1)⟩]

theorem exact84323RawTermsValid :
    exact84323RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84323 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7303⟩⟩) exact84323RawTerms .large 84322 .exactZero (none)

def event84324 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9570⟩⟩) 0 ⟨7303⟩ 84323

def event84325 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9570⟩⟩) 1 ⟨9569⟩ 84320

def event84326 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9570⟩⟩) (.product (.predecessor 0 84324 .coefficient) (.predecessor 1 84325 .coefficient) (⟨false, false, none, none, none⟩))

def event84327 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9570⟩⟩, .operator (⟨84323, 0⟩, ⟨84320, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩)

def exact84328RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩]

theorem exact84328RawTermsValid :
    exact84328RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84328 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9570⟩⟩) exact84328RawTerms .large 84326 .exactZero (none)

def event84329 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17153⟩⟩) 0 ⟨9570⟩ 84328

def event84330 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17153⟩⟩) 1 ⟨17152⟩ 84305

def event84331 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17153⟩⟩) (.sum [.predecessor 0 84329 .coefficient, .predecessor 1 84330 .coefficient])

def exact84332RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84332RawTermsValid :
    exact84332RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84332 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17153⟩⟩) exact84332RawTerms .large 84331 .exactZero (none)

def event84333 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17428⟩⟩) 0 ⟨17153⟩ 84332

def event84334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17428⟩⟩) 1 ⟨17425⟩ 84289

def event84335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17428⟩⟩) (.product (.predecessor 0 84333 .coefficient) (.predecessor 1 84334 .coefficient) (⟨false, false, none, none, none⟩))

def event84336 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17428⟩⟩, .operator (⟨84332, 0⟩, ⟨84289, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17425⟩⟩]⟩, (1)⟩)

def event84337 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17428⟩⟩, .operator (⟨84332, 1⟩, ⟨84289, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17425⟩⟩]⟩, (-1)⟩)

def event84338 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17428⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17425⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17425⟩⟩) ⟨16885⟩ 84286)

def event84339 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17428⟩⟩, .relation 84338 0, ⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨16885⟩⟩]⟩, (-1)⟩)

def exact84340RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17425⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨16885⟩⟩]⟩, (-1)⟩]

theorem exact84340RawTermsValid :
    exact84340RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84340 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17428⟩⟩) exact84340RawTerms .large 84335 .exactZero (none)

def event84341 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15836⟩⟩) 0 ⟨15620⟩ 84278

def event84342 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15836⟩⟩) (.authority (.programFamilyFact))

def exact84343RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], []⟩, (1)⟩]

theorem exact84343RawTermsValid :
    exact84343RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84343 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15836⟩⟩) exact84343RawTerms (.finite 2) 84342 .exactZero (none)

def event84344 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15838⟩⟩) 0 ⟨6908⟩ 84300

def event84345 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15838⟩⟩) 1 ⟨15836⟩ 84343

def event84346 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15838⟩⟩) (.product (.predecessor 0 84344 .coefficient) (.predecessor 1 84345 .coefficient) (⟨false, true, none, none, some 1⟩))

def event84347 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15838⟩⟩, .operator (⟨84300, 0⟩, ⟨84343, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact84348RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact84348RawTermsValid :
    exact84348RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84348 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15838⟩⟩) exact84348RawTerms .large 84346 .exactZero (none)

def event84349 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 84282

def event84350 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact84351RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact84351RawTermsValid :
    exact84351RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84351 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact84351RawTerms .large 84350 .exactZero (none)

def event84352 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15839⟩⟩) 0 ⟨7179⟩ 84351

def event84353 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15839⟩⟩) 1 ⟨15838⟩ 84348

def event84354 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15839⟩⟩) (.sum [.predecessor 0 84352 .coefficient, .predecessor 1 84353 .coefficient])

def exact84355RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84355RawTermsValid :
    exact84355RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84355 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15839⟩⟩) exact84355RawTerms .large 84354 .exactZero (none)

def event84356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17429⟩⟩) 0 ⟨15839⟩ 84355

def event84357 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17429⟩⟩) 1 ⟨17428⟩ 84340

def event84358 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17429⟩⟩) (.sum [.predecessor 0 84356 .coefficient, .predecessor 1 84357 .coefficient])

def exact84359RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17425⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨16885⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84359RawTermsValid :
    exact84359RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84359 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17429⟩⟩) exact84359RawTerms .large 84358 .exactZero (none)

def event84360 : Event := .preFoldPolynomial 84359 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17425⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨16885⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact84361RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17425⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨16885⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event84361 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17429⟩⟩) 84360 exact84361RawTerms .large 84358 .exactZero (none)

def event84362 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15620⟩⟩) ⟨⟨58⟩, ⟨36⟩, ⟨135⟩⟩ ⟨84196, 84362⟩

def event84363 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16352⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16349⟩⟩]⟩) (1) 0 2 (.universal 84362 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16349⟩⟩]⟩) (none) 84361)

def event84364 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16352⟩⟩, .relation 84363 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩)

def event84365 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16352⟩⟩, .relation 84363 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17425⟩⟩]⟩, (-1)⟩)

def event84366 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16352⟩⟩, .relation 84363 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨16885⟩⟩]⟩, (1)⟩)

def event84367 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16352⟩⟩, .relation 84363 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact84368RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17425⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨16885⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84368RawTermsValid :
    exact84368RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84368 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16352⟩⟩) exact84368RawTerms .large 84192 (.finite 202072841853861888) (some (84194))

def event84369 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17427⟩⟩) 0 ⟨16352⟩ 84368

def event84370 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17427⟩⟩) 1 ⟨17426⟩ 84182

def event84371 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17427⟩⟩) (.sum [.predecessor 0 84369 .coefficient, .predecessor 1 84370 .coefficient])

def event84372 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17427⟩⟩, .operator (⟨84368, 2⟩, ⟨84182, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], [⟨.program ⟨257⟩, ⟨16885⟩⟩]⟩, (-1)⟩)

def event84373 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17427⟩⟩, .operator (⟨84368, 1⟩, ⟨84182, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17425⟩⟩]⟩, (1)⟩)

def event84374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17427⟩⟩) (.sum [.result 84368 .summary, .result 84182 .summary])

def exact84375RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact84375RawTermsValid :
    exact84375RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84375 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17427⟩⟩) exact84375RawTerms .large 84371 (.finite 2997816280693142192128) (some (84374))

def event84376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17931⟩⟩) 0 ⟨17427⟩ 84375

def event84377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17931⟩⟩) 1 ⟨17929⟩ 84098

def event84378 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17931⟩⟩) (.product (.predecessor 0 84376 .coefficient) (.predecessor 1 84377 .coefficient) (⟨false, false, none, none, none⟩))

def event84379 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17931⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨17929⟩⟩]⟩) [⟨.result 84098 .coefficient, false, none⟩])

def event84380 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17931⟩⟩) (.product (.result 84375 .summary) (.transfer 84379) (⟨false, false, none, none, none⟩))

def event84381 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17931⟩⟩, .operator (⟨84375, 0⟩, ⟨84098, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17929⟩⟩]⟩, (1)⟩)

def event84382 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17931⟩⟩, .operator (⟨84375, 1⟩, ⟨84098, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17929⟩⟩]⟩, (-1)⟩)

def event84383 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17931⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17929⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17929⟩⟩) ⟨17055⟩ 84095)

def event84384 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17931⟩⟩, .relation 84383 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨17055⟩⟩]⟩, (-1)⟩)

def exact84385RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17929⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨17055⟩⟩]⟩, (-1)⟩]

theorem exact84385RawTermsValid :
    exact84385RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84385 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17931⟩⟩) exact84385RawTerms .large 84378 (.finite 32188807212483504816668771614720) (some (84380))

def event84386 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16716⟩⟩) 0 ⟨15837⟩ 3494

def event84387 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16716⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact84388RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16716⟩⟩]⟩, (1)⟩]

theorem exact84388RawTermsValid :
    exact84388RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84388 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16716⟩⟩) exact84388RawTerms (.finite 5647228698) 84387 .exactZero (none)

def event84389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16718⟩⟩) 0 ⟨16716⟩ 84388

def event84390 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16718⟩⟩) 1 ⟨2370⟩ 4

def event84391 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16718⟩⟩) (.scale (.predecessor 0 84389 .coefficient) (.value (.predecessor 1 84390 .coefficient)))

def exact84392RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16716⟩⟩]⟩, (1)⟩]

theorem exact84392RawTermsValid :
    exact84392RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84392 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16718⟩⟩) exact84392RawTerms (.finite 5647228698) 84391 .exactZero (none)

def event84393 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16719⟩⟩) 0 ⟨10368⟩ 75995

def event84394 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16719⟩⟩) 1 ⟨16718⟩ 84392

def event84395 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16719⟩⟩) (.product (.predecessor 0 84393 .coefficient) (.predecessor 1 84394 .coefficient) (⟨false, false, none, none, none⟩))

def event84396 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16719⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨16716⟩⟩]⟩) [⟨.result 84388 .coefficient, false, none⟩])

def event84397 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16719⟩⟩) (.product (.result 75995 .summary) (.transfer 84396) (⟨false, false, none, none, none⟩))

def event84398 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16719⟩⟩, .operator (⟨75995, 0⟩, ⟨84392, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16716⟩⟩]⟩, (1)⟩)

def event84399 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨16717⟩⟩)

def event84400 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event84401 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event84402 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event84403 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event84404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event84405 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event84406 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event84407 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event84408 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 84407

def event84409 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 84405

def event84410 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 84408 .coefficient) (.value (.predecessor 1 84409 .coefficient)))

def event84411 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event84412 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 84411

def event84413 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 84403

def event84414 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 84412 .coefficient, .predecessor 1 84413 .coefficient])

def event84415 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event84416 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 84415

def event84417 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 84401

def event84418 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 84417 .coefficient))

def event84419 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event84420 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15618⟩⟩) 0 ⟨10325⟩ 84419

def event84421 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15618⟩⟩) (.authority (.programFamilyFact))

def exact84422RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15618⟩⟩], []⟩, (1)⟩]

theorem exact84422RawTermsValid :
    exact84422RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84422 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15618⟩⟩) exact84422RawTerms (.finite 2) 84421 .exactZero (none)

def event84423 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12471⟩⟩) 0 ⟨10325⟩ 84419

def event84424 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12471⟩⟩) (.authority (.programFamilyFact))

def exact84425RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩], []⟩, (1)⟩]

theorem exact84425RawTermsValid :
    exact84425RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84425 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12471⟩⟩) exact84425RawTerms (.finite 2) 84424 .exactZero (none)

def event84426 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15619⟩⟩) 0 ⟨12471⟩ 84425

def event84427 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15619⟩⟩) 1 ⟨15618⟩ 84422

def event84428 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15619⟩⟩) (.product (.predecessor 0 84426 .coefficient) (.predecessor 1 84427 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event84429 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15619⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], []⟩) [⟨.result 84425 .coefficient, true, some 1⟩, ⟨.result 84422 .coefficient, true, some 1⟩])

def event84430 : Event := .survivorFold (1) 84429

def exact84431RawTerms : List Term := []

theorem exact84431RawTermsValid :
    exact84431RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84431 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15619⟩⟩) exact84431RawTerms (.finite 4) 84428 (.finite 4) (some (84429))

def event84432 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15620⟩⟩) 0 ⟨15619⟩ 84431

def event84433 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15620⟩⟩) (.identity (.predecessor 0 84432 .coefficient))

def event84434 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15620⟩⟩) (.finite 4)

def event84435 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15836⟩⟩) 0 ⟨15620⟩ 84434

def event84436 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15836⟩⟩) (.authority (.programFamilyFact))

def exact84437RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], []⟩, (1)⟩]

theorem exact84437RawTermsValid :
    exact84437RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84437 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15836⟩⟩) exact84437RawTerms (.finite 2) 84436 .exactZero (none)

def event84438 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15837⟩⟩) 0 ⟨15836⟩ 84437

def event84439 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15837⟩⟩) (.identity (.predecessor 0 84438 .coefficient))

def event84440 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15837⟩⟩) (.finite 2)

def event84441 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16716⟩⟩) 0 ⟨15837⟩ 84440

def event84442 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16716⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact84443RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16716⟩⟩]⟩, (1)⟩]

theorem exact84443RawTermsValid :
    exact84443RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84443 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16716⟩⟩) exact84443RawTerms (.finite 5647228698) 84442 .exactZero (none)

def event84444 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact84445RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact84445RawTermsValid :
    exact84445RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84445 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact84445RawTerms .large 84444 .exactZero (none)

def event84446 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16717⟩⟩) 0 ⟨35⟩ 84445

def event84447 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16717⟩⟩) 1 ⟨16716⟩ 84443

def event84448 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16717⟩⟩) (.product (.predecessor 0 84446 .coefficient) (.predecessor 1 84447 .coefficient) (⟨false, false, none, none, none⟩))

def event84449 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16717⟩⟩, .operator (⟨84445, 0⟩, ⟨84443, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16716⟩⟩]⟩, (1)⟩)

def exact84450RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16716⟩⟩]⟩, (1)⟩]

theorem exact84450RawTermsValid :
    exact84450RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84450 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16717⟩⟩) exact84450RawTerms .large 84448 .exactZero (none)

def event84451 : Event := .preFoldPolynomial 84450 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16716⟩⟩]⟩, (1)⟩] .exactZero none

def exact84452RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16716⟩⟩]⟩, (1)⟩]

def event84452 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨16717⟩⟩) 84451 exact84452RawTerms .large 84448 .exactZero (none)

def event84453 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨17933⟩⟩)

def event84454 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event84455 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event84456 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event84457 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event84458 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event84459 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event84460 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event84461 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event84462 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 84461

def event84463 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 84459

def event84464 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 84462 .coefficient) (.value (.predecessor 1 84463 .coefficient)))

def event84465 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event84466 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 84465

def event84467 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 84457

def event84468 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 84466 .coefficient, .predecessor 1 84467 .coefficient])

def event84469 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event84470 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 84469

def event84471 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 84455

def event84472 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 84471 .coefficient))

def event84473 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event84474 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15618⟩⟩) 0 ⟨10325⟩ 84473

def event84475 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15618⟩⟩) (.authority (.programFamilyFact))

def exact84476RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15618⟩⟩], []⟩, (1)⟩]

theorem exact84476RawTermsValid :
    exact84476RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84476 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15618⟩⟩) exact84476RawTerms (.finite 2) 84475 .exactZero (none)

def event84477 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12471⟩⟩) 0 ⟨10325⟩ 84473

def event84478 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12471⟩⟩) (.authority (.programFamilyFact))

def exact84479RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩], []⟩, (1)⟩]

theorem exact84479RawTermsValid :
    exact84479RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event84479 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12471⟩⟩) exact84479RawTerms (.finite 2) 84478 .exactZero (none)

def eventLeaf5264 : Array AnnotatedEvent := #[
  { event := event84224
    frameStart := 84196 },
  { event := event84225
    frameStart := 84196 },
  { event := event84226
    frameStart := 84196 },
  { event := event84227
    frameStart := 84196 },
  { event := event84228
    frameStart := 84196 },
  { event := event84229
    frameStart := 84196 },
  { event := event84230
    frameStart := 84196 },
  { event := event84231
    frameStart := 84196 },
  { event := event84232
    frameStart := 84196 },
  { event := event84233
    frameStart := 84196 },
  { event := event84234
    frameStart := 84196 },
  { event := event84235
    frameStart := 84196 },
  { event := event84236
    frameStart := 84196 },
  { event := event84237
    frameStart := 84196 },
  { event := event84238
    frameStart := 84196 },
  { event := event84239
    frameStart := 84196 }
]

def eventLeaf5265 : Array AnnotatedEvent := #[
  { event := event84240
    frameStart := 84196 },
  { event := event84241
    frameStart := 84196 },
  { event := event84242
    frameStart := 84196 },
  { event := event84243
    frameStart := 84196 },
  { event := event84244
    frameStart := 84244 },
  { event := event84245
    frameStart := 84244 },
  { event := event84246
    frameStart := 84244 },
  { event := event84247
    frameStart := 84244 },
  { event := event84248
    frameStart := 84244 },
  { event := event84249
    frameStart := 84244 },
  { event := event84250
    frameStart := 84244 },
  { event := event84251
    frameStart := 84244 },
  { event := event84252
    frameStart := 84244 },
  { event := event84253
    frameStart := 84244 },
  { event := event84254
    frameStart := 84244 },
  { event := event84255
    frameStart := 84244 }
]

def eventLeaf5266 : Array AnnotatedEvent := #[
  { event := event84256
    frameStart := 84244 },
  { event := event84257
    frameStart := 84244 },
  { event := event84258
    frameStart := 84244 },
  { event := event84259
    frameStart := 84244 },
  { event := event84260
    frameStart := 84244 },
  { event := event84261
    frameStart := 84244 },
  { event := event84262
    frameStart := 84244 },
  { event := event84263
    frameStart := 84244 },
  { event := event84264
    frameStart := 84244 },
  { event := event84265
    frameStart := 84244 },
  { event := event84266
    frameStart := 84244 },
  { event := event84267
    frameStart := 84244 },
  { event := event84268
    frameStart := 84244 },
  { event := event84269
    frameStart := 84244 },
  { event := event84270
    frameStart := 84244 },
  { event := event84271
    frameStart := 84244 }
]

def eventLeaf5267 : Array AnnotatedEvent := #[
  { event := event84272
    frameStart := 84244 },
  { event := event84273
    frameStart := 84244 },
  { event := event84274
    frameStart := 84244 },
  { event := event84275
    frameStart := 84244 },
  { event := event84276
    frameStart := 84244 },
  { event := event84277
    frameStart := 84244 },
  { event := event84278
    frameStart := 84244 },
  { event := event84279
    frameStart := 84244 },
  { event := event84280
    frameStart := 84244 },
  { event := event84281
    frameStart := 84244 },
  { event := event84282
    frameStart := 84244 },
  { event := event84283
    frameStart := 84244 },
  { event := event84284
    frameStart := 84244 },
  { event := event84285
    frameStart := 84244 },
  { event := event84286
    frameStart := 84244 },
  { event := event84287
    frameStart := 84244 }
]

def eventLeaf5268 : Array AnnotatedEvent := #[
  { event := event84288
    frameStart := 84244 },
  { event := event84289
    frameStart := 84244 },
  { event := event84290
    frameStart := 84244 },
  { event := event84291
    frameStart := 84244 },
  { event := event84292
    frameStart := 84244 },
  { event := event84293
    frameStart := 84244 },
  { event := event84294
    frameStart := 84244 },
  { event := event84295
    frameStart := 84244 },
  { event := event84296
    frameStart := 84244 },
  { event := event84297
    frameStart := 84244 },
  { event := event84298
    frameStart := 84244 },
  { event := event84299
    frameStart := 84244 },
  { event := event84300
    frameStart := 84244 },
  { event := event84301
    frameStart := 84244 },
  { event := event84302
    frameStart := 84244 },
  { event := event84303
    frameStart := 84244 }
]

def eventLeaf5269 : Array AnnotatedEvent := #[
  { event := event84304
    frameStart := 84244 },
  { event := event84305
    frameStart := 84244 },
  { event := event84306
    frameStart := 84244 },
  { event := event84307
    frameStart := 84244 },
  { event := event84308
    frameStart := 84244 },
  { event := event84309
    frameStart := 84244 },
  { event := event84310
    frameStart := 84244 },
  { event := event84311
    frameStart := 84244 },
  { event := event84312
    frameStart := 84244 },
  { event := event84313
    frameStart := 84244 },
  { event := event84314
    frameStart := 84244 },
  { event := event84315
    frameStart := 84244 },
  { event := event84316
    frameStart := 84244 },
  { event := event84317
    frameStart := 84244 },
  { event := event84318
    frameStart := 84244 },
  { event := event84319
    frameStart := 84244 }
]

def eventLeaf5270 : Array AnnotatedEvent := #[
  { event := event84320
    frameStart := 84244 },
  { event := event84321
    frameStart := 84244 },
  { event := event84322
    frameStart := 84244 },
  { event := event84323
    frameStart := 84244 },
  { event := event84324
    frameStart := 84244 },
  { event := event84325
    frameStart := 84244 },
  { event := event84326
    frameStart := 84244 },
  { event := event84327
    frameStart := 84244 },
  { event := event84328
    frameStart := 84244 },
  { event := event84329
    frameStart := 84244 },
  { event := event84330
    frameStart := 84244 },
  { event := event84331
    frameStart := 84244 },
  { event := event84332
    frameStart := 84244 },
  { event := event84333
    frameStart := 84244 },
  { event := event84334
    frameStart := 84244 },
  { event := event84335
    frameStart := 84244 }
]

def eventLeaf5271 : Array AnnotatedEvent := #[
  { event := event84336
    frameStart := 84244 },
  { event := event84337
    frameStart := 84244 },
  { event := event84338
    frameStart := 84244 },
  { event := event84339
    frameStart := 84244 },
  { event := event84340
    frameStart := 84244 },
  { event := event84341
    frameStart := 84244 },
  { event := event84342
    frameStart := 84244 },
  { event := event84343
    frameStart := 84244 },
  { event := event84344
    frameStart := 84244 },
  { event := event84345
    frameStart := 84244 },
  { event := event84346
    frameStart := 84244 },
  { event := event84347
    frameStart := 84244 },
  { event := event84348
    frameStart := 84244 },
  { event := event84349
    frameStart := 84244 },
  { event := event84350
    frameStart := 84244 },
  { event := event84351
    frameStart := 84244 }
]

def eventLeaf5272 : Array AnnotatedEvent := #[
  { event := event84352
    frameStart := 84244 },
  { event := event84353
    frameStart := 84244 },
  { event := event84354
    frameStart := 84244 },
  { event := event84355
    frameStart := 84244 },
  { event := event84356
    frameStart := 84244 },
  { event := event84357
    frameStart := 84244 },
  { event := event84358
    frameStart := 84244 },
  { event := event84359
    frameStart := 84244 },
  { event := event84360
    frameStart := 84244 },
  { event := event84361
    frameStart := 84244 },
  { event := event84362
    frameStart := 0 },
  { event := event84363
    frameStart := 0 },
  { event := event84364
    frameStart := 0 },
  { event := event84365
    frameStart := 0 },
  { event := event84366
    frameStart := 0 },
  { event := event84367
    frameStart := 0 }
]

def eventLeaf5273 : Array AnnotatedEvent := #[
  { event := event84368
    frameStart := 0 },
  { event := event84369
    frameStart := 0 },
  { event := event84370
    frameStart := 0 },
  { event := event84371
    frameStart := 0 },
  { event := event84372
    frameStart := 0 },
  { event := event84373
    frameStart := 0 },
  { event := event84374
    frameStart := 0 },
  { event := event84375
    frameStart := 0 },
  { event := event84376
    frameStart := 0 },
  { event := event84377
    frameStart := 0 },
  { event := event84378
    frameStart := 0 },
  { event := event84379
    frameStart := 0 },
  { event := event84380
    frameStart := 0 },
  { event := event84381
    frameStart := 0 },
  { event := event84382
    frameStart := 0 },
  { event := event84383
    frameStart := 0 }
]

def eventLeaf5274 : Array AnnotatedEvent := #[
  { event := event84384
    frameStart := 0 },
  { event := event84385
    frameStart := 0 },
  { event := event84386
    frameStart := 0 },
  { event := event84387
    frameStart := 0 },
  { event := event84388
    frameStart := 0 },
  { event := event84389
    frameStart := 0 },
  { event := event84390
    frameStart := 0 },
  { event := event84391
    frameStart := 0 },
  { event := event84392
    frameStart := 0 },
  { event := event84393
    frameStart := 0 },
  { event := event84394
    frameStart := 0 },
  { event := event84395
    frameStart := 0 },
  { event := event84396
    frameStart := 0 },
  { event := event84397
    frameStart := 0 },
  { event := event84398
    frameStart := 0 },
  { event := event84399
    frameStart := 84399 }
]

def eventLeaf5275 : Array AnnotatedEvent := #[
  { event := event84400
    frameStart := 84399 },
  { event := event84401
    frameStart := 84399 },
  { event := event84402
    frameStart := 84399 },
  { event := event84403
    frameStart := 84399 },
  { event := event84404
    frameStart := 84399 },
  { event := event84405
    frameStart := 84399 },
  { event := event84406
    frameStart := 84399 },
  { event := event84407
    frameStart := 84399 },
  { event := event84408
    frameStart := 84399 },
  { event := event84409
    frameStart := 84399 },
  { event := event84410
    frameStart := 84399 },
  { event := event84411
    frameStart := 84399 },
  { event := event84412
    frameStart := 84399 },
  { event := event84413
    frameStart := 84399 },
  { event := event84414
    frameStart := 84399 },
  { event := event84415
    frameStart := 84399 }
]

def eventLeaf5276 : Array AnnotatedEvent := #[
  { event := event84416
    frameStart := 84399 },
  { event := event84417
    frameStart := 84399 },
  { event := event84418
    frameStart := 84399 },
  { event := event84419
    frameStart := 84399 },
  { event := event84420
    frameStart := 84399 },
  { event := event84421
    frameStart := 84399 },
  { event := event84422
    frameStart := 84399 },
  { event := event84423
    frameStart := 84399 },
  { event := event84424
    frameStart := 84399 },
  { event := event84425
    frameStart := 84399 },
  { event := event84426
    frameStart := 84399 },
  { event := event84427
    frameStart := 84399 },
  { event := event84428
    frameStart := 84399 },
  { event := event84429
    frameStart := 84399 },
  { event := event84430
    frameStart := 84399 },
  { event := event84431
    frameStart := 84399 }
]

def eventLeaf5277 : Array AnnotatedEvent := #[
  { event := event84432
    frameStart := 84399 },
  { event := event84433
    frameStart := 84399 },
  { event := event84434
    frameStart := 84399 },
  { event := event84435
    frameStart := 84399 },
  { event := event84436
    frameStart := 84399 },
  { event := event84437
    frameStart := 84399 },
  { event := event84438
    frameStart := 84399 },
  { event := event84439
    frameStart := 84399 },
  { event := event84440
    frameStart := 84399 },
  { event := event84441
    frameStart := 84399 },
  { event := event84442
    frameStart := 84399 },
  { event := event84443
    frameStart := 84399 },
  { event := event84444
    frameStart := 84399 },
  { event := event84445
    frameStart := 84399 },
  { event := event84446
    frameStart := 84399 },
  { event := event84447
    frameStart := 84399 }
]

def eventLeaf5278 : Array AnnotatedEvent := #[
  { event := event84448
    frameStart := 84399 },
  { event := event84449
    frameStart := 84399 },
  { event := event84450
    frameStart := 84399 },
  { event := event84451
    frameStart := 84399 },
  { event := event84452
    frameStart := 84399 },
  { event := event84453
    frameStart := 84453 },
  { event := event84454
    frameStart := 84453 },
  { event := event84455
    frameStart := 84453 },
  { event := event84456
    frameStart := 84453 },
  { event := event84457
    frameStart := 84453 },
  { event := event84458
    frameStart := 84453 },
  { event := event84459
    frameStart := 84453 },
  { event := event84460
    frameStart := 84453 },
  { event := event84461
    frameStart := 84453 },
  { event := event84462
    frameStart := 84453 },
  { event := event84463
    frameStart := 84453 }
]

def eventLeaf5279 : Array AnnotatedEvent := #[
  { event := event84464
    frameStart := 84453 },
  { event := event84465
    frameStart := 84453 },
  { event := event84466
    frameStart := 84453 },
  { event := event84467
    frameStart := 84453 },
  { event := event84468
    frameStart := 84453 },
  { event := event84469
    frameStart := 84453 },
  { event := event84470
    frameStart := 84453 },
  { event := event84471
    frameStart := 84453 },
  { event := event84472
    frameStart := 84453 },
  { event := event84473
    frameStart := 84453 },
  { event := event84474
    frameStart := 84453 },
  { event := event84475
    frameStart := 84453 },
  { event := event84476
    frameStart := 84453 },
  { event := event84477
    frameStart := 84453 },
  { event := event84478
    frameStart := 84453 },
  { event := event84479
    frameStart := 84453 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events329
