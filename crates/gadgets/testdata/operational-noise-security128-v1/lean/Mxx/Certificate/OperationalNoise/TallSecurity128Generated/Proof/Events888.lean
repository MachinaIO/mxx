import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events888

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact227328RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60676⟩⟩]⟩, (1)⟩]

def event227328 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨60677⟩⟩) 227327 exact227328RawTerms .large 227324 .exactZero (none)

def event227329 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨61866⟩⟩)

def event227330 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event227331 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event227332 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.authority (.operator))

def event227333 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.finite 5)

def event227334 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event227335 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event227336 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event227337 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event227338 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 227337

def event227339 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 227335

def event227340 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 227338 .coefficient) (.value (.predecessor 1 227339 .coefficient)))

def event227341 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event227342 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 0 ⟨392⟩ 227341

def event227343 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 1 ⟨4990⟩ 227333

def event227344 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.sum [.predecessor 0 227342 .coefficient, .predecessor 1 227343 .coefficient])

def event227345 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.finite 655345)

def event227346 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 0 ⟨4992⟩ 227345

def event227347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 1 ⟨5426⟩ 227331

def event227348 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.identity (.predecessor 1 227347 .coefficient))

def event227349 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.finite 655360)

def event227350 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25238⟩⟩) 0 ⟨5577⟩ 227349

def event227351 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25238⟩⟩) (.authority (.programFamilyFact))

def exact227352RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25238⟩⟩], []⟩, (1)⟩]

theorem exact227352RawTermsValid :
    exact227352RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227352 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25238⟩⟩) exact227352RawTerms (.finite 18) 227351 .exactZero (none)

def event227353 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59458⟩⟩) 0 ⟨5577⟩ 227349

def event227354 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59458⟩⟩) (.authority (.programFamilyFact))

def exact227355RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59458⟩⟩], []⟩, (1)⟩]

theorem exact227355RawTermsValid :
    exact227355RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227355 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59458⟩⟩) exact227355RawTerms (.finite 18) 227354 .exactZero (none)

def event227356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59459⟩⟩) 0 ⟨59458⟩ 227355

def event227357 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59459⟩⟩) 1 ⟨25238⟩ 227352

def event227358 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59459⟩⟩) (.product (.predecessor 0 227356 .coefficient) (.predecessor 1 227357 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event227359 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59459⟩⟩, .operator (⟨227355, 0⟩, ⟨227352, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], []⟩, (1)⟩)

def exact227360RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25238⟩⟩, ⟨.program ⟨257⟩, ⟨59458⟩⟩], []⟩, (1)⟩]

theorem exact227360RawTermsValid :
    exact227360RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227360 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59459⟩⟩) exact227360RawTerms (.finite 324) 227358 .exactZero (none)

def event227361 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59460⟩⟩) 0 ⟨59459⟩ 227360

def event227362 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59460⟩⟩) (.identity (.predecessor 0 227361 .coefficient))

def event227363 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨59460⟩⟩) (.finite 324)

def event227364 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59820⟩⟩) 0 ⟨59460⟩ 227363

def event227365 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59820⟩⟩) (.authority (.programFamilyFact))

def exact227366RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59820⟩⟩], []⟩, (1)⟩]

theorem exact227366RawTermsValid :
    exact227366RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227366 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59820⟩⟩) exact227366RawTerms (.finite 18) 227365 .exactZero (none)

def event227367 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59821⟩⟩) 0 ⟨59820⟩ 227366

def event227368 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59821⟩⟩) (.identity (.predecessor 0 227367 .coefficient))

def event227369 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨59821⟩⟩) (.finite 18)

def event227370 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61090⟩⟩) 0 ⟨59821⟩ 227369

def event227371 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61090⟩⟩) (.authority (.programFamilyFact))

def event227372 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨61090⟩⟩) (.finite 3720)

def event227373 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event227374 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61092⟩⟩) 0 ⟨7177⟩ 227373

def event227375 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61092⟩⟩) 1 ⟨61090⟩ 227372

def event227376 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61092⟩⟩) (.authority (.operator))

def exact227377RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨61092⟩⟩]⟩, (1)⟩]

theorem exact227377RawTermsValid :
    exact227377RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227377 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61092⟩⟩) exact227377RawTerms .large 227376 .exactZero (none)

def event227378 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61861⟩⟩) 0 ⟨61092⟩ 227377

def event227379 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61861⟩⟩) (.authority (.operator))

def exact227380RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩, (1)⟩]

theorem exact227380RawTermsValid :
    exact227380RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227380 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61861⟩⟩) exact227380RawTerms (.finite 8192) 227379 .exactZero (none)

def event227381 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event227382 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event227383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61302⟩⟩) 0 ⟨59821⟩ 227369

def event227384 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61302⟩⟩) 1 ⟨136⟩ 227382

def event227385 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61302⟩⟩) (.sum [.predecessor 0 227383 .coefficient, .predecessor 1 227384 .coefficient])

def event227386 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨61302⟩⟩) (.finite 18)

def event227387 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61303⟩⟩) 0 ⟨61302⟩ 227386

def event227388 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61303⟩⟩) (.identity (.predecessor 0 227387 .coefficient))

def exact227389RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59820⟩⟩], []⟩, (1)⟩]

theorem exact227389RawTermsValid :
    exact227389RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227389 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61303⟩⟩) exact227389RawTerms (.finite 18) 227388 .exactZero (none)

def event227390 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact227391RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact227391RawTermsValid :
    exact227391RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227391 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact227391RawTerms .large 227390 .exactZero (none)

def event227392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61304⟩⟩) 0 ⟨6908⟩ 227391

def event227393 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61304⟩⟩) 1 ⟨61303⟩ 227389

def event227394 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61304⟩⟩) (.product (.predecessor 0 227392 .coefficient) (.predecessor 1 227393 .coefficient) (⟨false, false, none, none, none⟩))

def event227395 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61304⟩⟩, .operator (⟨227391, 0⟩, ⟨227389, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact227396RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact227396RawTermsValid :
    exact227396RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227396 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61304⟩⟩) exact227396RawTerms .large 227394 .exactZero (none)

def event227397 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7186⟩⟩) 0 ⟨7177⟩ 227373

def event227398 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7186⟩⟩) (.authority (.operator))

def exact227399RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩, (1)⟩]

theorem exact227399RawTermsValid :
    exact227399RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227399 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7186⟩⟩) exact227399RawTerms .large 227398 .exactZero (none)

def event227400 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61305⟩⟩) 0 ⟨7186⟩ 227399

def event227401 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61305⟩⟩) 1 ⟨61304⟩ 227396

def event227402 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61305⟩⟩) (.sum [.predecessor 0 227400 .coefficient, .predecessor 1 227401 .coefficient])

def exact227403RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227403RawTermsValid :
    exact227403RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227403 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61305⟩⟩) exact227403RawTerms .large 227402 .exactZero (none)

def event227404 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61862⟩⟩) 0 ⟨61305⟩ 227403

def event227405 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61862⟩⟩) 1 ⟨61861⟩ 227380

def event227406 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61862⟩⟩) (.product (.predecessor 0 227404 .coefficient) (.predecessor 1 227405 .coefficient) (⟨false, false, none, none, none⟩))

def event227407 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61862⟩⟩, .operator (⟨227403, 0⟩, ⟨227380, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩, (1)⟩)

def event227408 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61862⟩⟩, .operator (⟨227403, 1⟩, ⟨227380, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩, (-1)⟩)

def event227409 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨61862⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61861⟩⟩) ⟨61092⟩ 227377)

def event227410 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61862⟩⟩, .relation 227409 0, ⟨[⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨61092⟩⟩]⟩, (-1)⟩)

def exact227411RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨61092⟩⟩]⟩, (-1)⟩]

theorem exact227411RawTermsValid :
    exact227411RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227411 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61862⟩⟩) exact227411RawTerms .large 227406 .exactZero (none)

def event227412 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60082⟩⟩) 0 ⟨59821⟩ 227369

def event227413 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60082⟩⟩) (.authority (.programFamilyFact))

def exact227414RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨60082⟩⟩], []⟩, (1)⟩]

theorem exact227414RawTermsValid :
    exact227414RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227414 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60082⟩⟩) exact227414RawTerms (.finite 61) 227413 .exactZero (none)

def event227415 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60084⟩⟩) 0 ⟨6908⟩ 227391

def event227416 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60084⟩⟩) 1 ⟨60082⟩ 227414

def event227417 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60084⟩⟩) (.product (.predecessor 0 227415 .coefficient) (.predecessor 1 227416 .coefficient) (⟨false, true, none, none, some 1⟩))

def event227418 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60084⟩⟩, .operator (⟨227391, 0⟩, ⟨227414, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact227419RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact227419RawTermsValid :
    exact227419RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227419 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60084⟩⟩) exact227419RawTerms .large 227417 .exactZero (none)

def event227420 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7212⟩⟩) 0 ⟨7177⟩ 227373

def event227421 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7212⟩⟩) (.authority (.operator))

def exact227422RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩]

theorem exact227422RawTermsValid :
    exact227422RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227422 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7212⟩⟩) exact227422RawTerms .large 227421 .exactZero (none)

def event227423 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60085⟩⟩) 0 ⟨7212⟩ 227422

def event227424 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60085⟩⟩) 1 ⟨60084⟩ 227419

def event227425 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60085⟩⟩) (.sum [.predecessor 0 227423 .coefficient, .predecessor 1 227424 .coefficient])

def exact227426RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227426RawTermsValid :
    exact227426RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227426 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60085⟩⟩) exact227426RawTerms .large 227425 .exactZero (none)

def event227427 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61866⟩⟩) 0 ⟨60085⟩ 227426

def event227428 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61866⟩⟩) 1 ⟨61862⟩ 227411

def event227429 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61866⟩⟩) (.sum [.predecessor 0 227427 .coefficient, .predecessor 1 227428 .coefficient])

def exact227430RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨61092⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227430RawTermsValid :
    exact227430RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227430 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61866⟩⟩) exact227430RawTerms .large 227429 .exactZero (none)

def event227431 : Event := .preFoldPolynomial 227430 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨61092⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact227432RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨61092⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event227432 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨61866⟩⟩) 227431 exact227432RawTerms .large 227429 .exactZero (none)

def event227433 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨59821⟩⟩) ⟨⟨91⟩, ⟨72⟩, ⟨135⟩⟩ ⟨227275, 227433⟩

def event227434 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨60679⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60676⟩⟩]⟩) (1) 0 2 (.universal 227433 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60676⟩⟩]⟩) (none) 227432)

def event227435 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60679⟩⟩, .relation 227434 1, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩)

def event227436 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60679⟩⟩, .relation 227434 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩, (-1)⟩)

def event227437 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60679⟩⟩, .relation 227434 2, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨61092⟩⟩]⟩, (1)⟩)

def event227438 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60679⟩⟩, .relation 227434 3, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact227439RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨61092⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227439RawTermsValid :
    exact227439RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227439 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60679⟩⟩) exact227439RawTerms .large 227271 (.finite 202072841853861888) (some (227273))

def event227440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61864⟩⟩) 0 ⟨60679⟩ 227439

def event227441 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61864⟩⟩) 1 ⟨61863⟩ 227261

def event227442 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61864⟩⟩) (.sum [.predecessor 0 227440 .coefficient, .predecessor 1 227441 .coefficient])

def event227443 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61864⟩⟩, .operator (⟨227439, 0⟩, ⟨227261, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61861⟩⟩]⟩, (1)⟩)

def event227444 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨61864⟩⟩, .operator (⟨227439, 2⟩, ⟨227261, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨61092⟩⟩]⟩, (-1)⟩)

def event227445 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61864⟩⟩) (.sum [.result 227439 .summary, .result 227261 .summary])

def exact227446RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227446RawTermsValid :
    exact227446RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227446 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61864⟩⟩) exact227446RawTerms .large 227442 (.finite 32190378816049205907437743505408) (some (227445))

def event227447 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58110⟩⟩) 0 ⟨56841⟩ 10836

def event227448 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58110⟩⟩) (.authority (.programFamilyFact))

def event227449 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨58110⟩⟩) (.finite 3720)

def event227450 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58112⟩⟩) 0 ⟨7177⟩ 15500

def event227451 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58112⟩⟩) 1 ⟨58110⟩ 227449

def event227452 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58112⟩⟩) (.authority (.operator))

def exact227453RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58112⟩⟩]⟩, (1)⟩]

theorem exact227453RawTermsValid :
    exact227453RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227453 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58112⟩⟩) exact227453RawTerms .large 227452 .exactZero (none)

def event227454 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58881⟩⟩) 0 ⟨58112⟩ 227453

def event227455 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58881⟩⟩) (.authority (.operator))

def exact227456RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58881⟩⟩]⟩, (1)⟩]

theorem exact227456RawTermsValid :
    exact227456RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227456 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58881⟩⟩) exact227456RawTerms (.finite 8192) 227455 .exactZero (none)

def event227457 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57962⟩⟩) 0 ⟨56480⟩ 10830

def event227458 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57962⟩⟩) (.authority (.programFamilyFact))

def event227459 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨57962⟩⟩) (.finite 3720)

def event227460 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57963⟩⟩) 0 ⟨7177⟩ 15500

def event227461 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57963⟩⟩) 1 ⟨57962⟩ 227459

def event227462 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57963⟩⟩) (.authority (.operator))

def exact227463RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨57963⟩⟩]⟩, (1)⟩]

theorem exact227463RawTermsValid :
    exact227463RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227463 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57963⟩⟩) exact227463RawTerms .large 227462 .exactZero (none)

def event227464 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58468⟩⟩) 0 ⟨57963⟩ 227463

def event227465 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58468⟩⟩) (.authority (.operator))

def exact227466RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58468⟩⟩]⟩, (1)⟩]

theorem exact227466RawTermsValid :
    exact227466RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227466 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58468⟩⟩) exact227466RawTerms (.finite 8192) 227465 .exactZero (none)

def event227467 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24999⟩⟩) 0 ⟨24998⟩ 10819

def event227468 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24999⟩⟩) 1 ⟨6937⟩ 222153

def event227469 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24999⟩⟩) (.tensor (.predecessor 0 227467 .coefficient) (.predecessor 1 227468 .coefficient) true false)

def event227470 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24999⟩⟩, .operator (⟨10819, 0⟩, ⟨222153, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact227471RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact227471RawTermsValid :
    exact227471RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227471 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24999⟩⟩) exact227471RawTerms .large 227469 .exactZero (none)

def event227472 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8465⟩⟩) 0 ⟨5579⟩ 222023

def event227473 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8465⟩⟩) 1 ⟨7273⟩ 22591

def event227474 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8465⟩⟩) (.product (.predecessor 0 227472 .coefficient) (.predecessor 1 227473 .coefficient) (⟨false, false, none, none, none⟩))

def event227475 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8465⟩⟩, .operator (⟨222023, 0⟩, ⟨22591, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩)

def exact227476RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩]

theorem exact227476RawTermsValid :
    exact227476RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227476 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8465⟩⟩) exact227476RawTerms .large 227474 .exactZero (none)

def event227477 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25000⟩⟩) 0 ⟨8465⟩ 227476

def event227478 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25000⟩⟩) 1 ⟨24999⟩ 227471

def event227479 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25000⟩⟩) (.sum [.predecessor 0 227477 .coefficient, .predecessor 1 227478 .coefficient])

def exact227480RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227480RawTermsValid :
    exact227480RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227480 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25000⟩⟩) exact227480RawTerms .large 227479 .exactZero (none)

def event227481 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25001⟩⟩) 0 ⟨25000⟩ 227480

def event227482 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25001⟩⟩) 1 ⟨99⟩ 22583

def event227483 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25001⟩⟩) (.sum [.predecessor 0 227481 .coefficient, .predecessor 1 227482 .coefficient])

def event227484 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25001⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨99⟩⟩]⟩) [⟨.result 22583 .coefficient, false, none⟩])

def event227485 : Event := .survivorFold (1) 227484

def exact227486RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227486RawTermsValid :
    exact227486RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227486 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25001⟩⟩) exact227486RawTerms .large 227483 (.finite 26) (some (227484))

def event227487 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56481⟩⟩) 0 ⟨25001⟩ 227486

def event227488 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56481⟩⟩) 1 ⟨56478⟩ 10822

def event227489 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56481⟩⟩) (.product (.predecessor 0 227487 .coefficient) (.predecessor 1 227488 .coefficient) (⟨false, true, none, none, some 1⟩))

def event227490 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56481⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨56478⟩⟩], []⟩) [⟨.result 10822 .coefficient, true, some 1⟩])

def event227491 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56481⟩⟩) (.product (.result 227486 .summary) (.transfer 227490) (⟨false, false, none, none, none⟩))

def event227492 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56481⟩⟩, .operator (⟨227486, 1⟩, ⟨10822, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event227493 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56481⟩⟩, .operator (⟨227486, 0⟩, ⟨10822, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩)

def exact227494RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩]

theorem exact227494RawTermsValid :
    exact227494RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227494 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56481⟩⟩) exact227494RawTerms .large 227489 (.finite 13631488) (some (227491))

def event227495 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56482⟩⟩) 0 ⟨56478⟩ 10822

def event227496 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56482⟩⟩) 1 ⟨6937⟩ 222153

def event227497 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56482⟩⟩) (.tensor (.predecessor 0 227495 .coefficient) (.predecessor 1 227496 .coefficient) true false)

def event227498 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56482⟩⟩, .operator (⟨10822, 0⟩, ⟨222153, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact227499RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact227499RawTermsValid :
    exact227499RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227499 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56482⟩⟩) exact227499RawTerms .large 227497 .exactZero (none)

def event227500 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8482⟩⟩) 0 ⟨5579⟩ 222023

def event227501 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8482⟩⟩) 1 ⟨7290⟩ 22632

def event227502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8482⟩⟩) (.product (.predecessor 0 227500 .coefficient) (.predecessor 1 227501 .coefficient) (⟨false, false, none, none, none⟩))

def event227503 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8482⟩⟩, .operator (⟨222023, 0⟩, ⟨22632, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩]⟩, (1)⟩)

def exact227504RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩]⟩, (1)⟩]

theorem exact227504RawTermsValid :
    exact227504RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227504 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8482⟩⟩) exact227504RawTerms .large 227502 .exactZero (none)

def event227505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56483⟩⟩) 0 ⟨8482⟩ 227504

def event227506 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56483⟩⟩) 1 ⟨56482⟩ 227499

def event227507 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56483⟩⟩) (.sum [.predecessor 0 227505 .coefficient, .predecessor 1 227506 .coefficient])

def exact227508RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227508RawTermsValid :
    exact227508RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227508 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56483⟩⟩) exact227508RawTerms .large 227507 .exactZero (none)

def event227509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56484⟩⟩) 0 ⟨56483⟩ 227508

def event227510 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56484⟩⟩) 1 ⟨116⟩ 22624

def event227511 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56484⟩⟩) (.sum [.predecessor 0 227509 .coefficient, .predecessor 1 227510 .coefficient])

def event227512 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56484⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨116⟩⟩]⟩) [⟨.result 22624 .coefficient, false, none⟩])

def event227513 : Event := .survivorFold (1) 227512

def exact227514RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227514RawTermsValid :
    exact227514RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227514 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56484⟩⟩) exact227514RawTerms .large 227511 (.finite 26) (some (227512))

def event227515 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56485⟩⟩) 0 ⟨56484⟩ 227514

def event227516 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56485⟩⟩) 1 ⟨9533⟩ 22621

def event227517 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56485⟩⟩) (.product (.predecessor 0 227515 .coefficient) (.predecessor 1 227516 .coefficient) (⟨false, false, none, none, none⟩))

def event227518 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56485⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) [⟨.result 22617 .coefficient, false, none⟩])

def event227519 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56485⟩⟩) (.product (.result 227514 .summary) (.transfer 227518) (⟨false, false, none, none, none⟩))

def event227520 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56485⟩⟩, .operator (⟨227514, 1⟩, ⟨22621, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩, (-1)⟩)

def event227521 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨56485⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9532⟩⟩) ⟨7273⟩ 22591)

def event227522 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56485⟩⟩, .relation 227521 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (-1)⟩)

def event227523 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56485⟩⟩, .operator (⟨227514, 0⟩, ⟨22621, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩, (1)⟩)

def exact227524RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (-1)⟩]

theorem exact227524RawTermsValid :
    exact227524RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227524 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56485⟩⟩) exact227524RawTerms .large 227517 (.finite 279172874240) (some (227519))

def event227525 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56486⟩⟩) 0 ⟨56485⟩ 227524

def event227526 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56486⟩⟩) 1 ⟨56481⟩ 227494

def event227527 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56486⟩⟩) (.sum [.predecessor 0 227525 .coefficient, .predecessor 1 227526 .coefficient])

def event227528 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56486⟩⟩, .operator (⟨227524, 1⟩, ⟨227494, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩)

def event227529 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56486⟩⟩) (.sum [.result 227524 .summary, .result 227494 .summary])

def exact227530RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact227530RawTermsValid :
    exact227530RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227530 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56486⟩⟩) exact227530RawTerms .large 227527 (.finite 279186505728) (some (227529))

def event227531 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58469⟩⟩) 0 ⟨56486⟩ 227530

def event227532 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58469⟩⟩) 1 ⟨58468⟩ 227466

def event227533 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58469⟩⟩) (.product (.predecessor 0 227531 .coefficient) (.predecessor 1 227532 .coefficient) (⟨false, false, none, none, none⟩))

def event227534 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58469⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨58468⟩⟩]⟩) [⟨.result 227466 .coefficient, false, none⟩])

def event227535 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58469⟩⟩) (.product (.result 227530 .summary) (.transfer 227534) (⟨false, false, none, none, none⟩))

def event227536 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58469⟩⟩, .operator (⟨227530, 1⟩, ⟨227466, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58468⟩⟩]⟩, (-1)⟩)

def event227537 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨58469⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58468⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58468⟩⟩) ⟨57963⟩ 227463)

def event227538 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58469⟩⟩, .relation 227537 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨57963⟩⟩]⟩, (-1)⟩)

def event227539 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58469⟩⟩, .operator (⟨227530, 0⟩, ⟨227466, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58468⟩⟩]⟩, (1)⟩)

def exact227540RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58468⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨57963⟩⟩]⟩, (-1)⟩]

theorem exact227540RawTermsValid :
    exact227540RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227540 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58469⟩⟩) exact227540RawTerms .large 227533 (.finite 2997742278965691678720) (some (227535))

def event227541 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57399⟩⟩) 0 ⟨56480⟩ 10830

def event227542 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57399⟩⟩) (.authority (.relationPreimageSource ⟨42⟩))

def exact227543RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨57399⟩⟩]⟩, (1)⟩]

theorem exact227543RawTermsValid :
    exact227543RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227543 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57399⟩⟩) exact227543RawTerms (.finite 5647228698) 227542 .exactZero (none)

def event227544 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57401⟩⟩) 0 ⟨57399⟩ 227543

def event227545 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57401⟩⟩) 1 ⟨2370⟩ 4

def event227546 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57401⟩⟩) (.scale (.predecessor 0 227544 .coefficient) (.value (.predecessor 1 227545 .coefficient)))

def exact227547RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨57399⟩⟩]⟩, (1)⟩]

theorem exact227547RawTermsValid :
    exact227547RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227547 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57401⟩⟩) exact227547RawTerms (.finite 5647228698) 227546 .exactZero (none)

def event227548 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57402⟩⟩) 0 ⟨5581⟩ 222245

def event227549 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57402⟩⟩) 1 ⟨57401⟩ 227547

def event227550 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57402⟩⟩) (.product (.predecessor 0 227548 .coefficient) (.predecessor 1 227549 .coefficient) (⟨false, false, none, none, none⟩))

def event227551 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57402⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨57399⟩⟩]⟩) [⟨.result 227543 .coefficient, false, none⟩])

def event227552 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57402⟩⟩) (.product (.result 222245 .summary) (.transfer 227551) (⟨false, false, none, none, none⟩))

def event227553 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57402⟩⟩, .operator (⟨222245, 0⟩, ⟨227547, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57399⟩⟩]⟩, (1)⟩)

def event227554 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨57400⟩⟩)

def event227555 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event227556 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event227557 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.authority (.operator))

def event227558 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.finite 5)

def event227559 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event227560 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event227561 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event227562 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event227563 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 227562

def event227564 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 227560

def event227565 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 227563 .coefficient) (.value (.predecessor 1 227564 .coefficient)))

def event227566 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event227567 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 0 ⟨392⟩ 227566

def event227568 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 1 ⟨4990⟩ 227558

def event227569 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.sum [.predecessor 0 227567 .coefficient, .predecessor 1 227568 .coefficient])

def event227570 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.finite 655345)

def event227571 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 0 ⟨4992⟩ 227570

def event227572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 1 ⟨5426⟩ 227556

def event227573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.identity (.predecessor 1 227572 .coefficient))

def event227574 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.finite 655360)

def event227575 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24998⟩⟩) 0 ⟨5577⟩ 227574

def event227576 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24998⟩⟩) (.authority (.programFamilyFact))

def exact227577RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24998⟩⟩], []⟩, (1)⟩]

theorem exact227577RawTermsValid :
    exact227577RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227577 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24998⟩⟩) exact227577RawTerms (.finite 16) 227576 .exactZero (none)

def event227578 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56478⟩⟩) 0 ⟨5577⟩ 227574

def event227579 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56478⟩⟩) (.authority (.programFamilyFact))

def exact227580RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56478⟩⟩], []⟩, (1)⟩]

theorem exact227580RawTermsValid :
    exact227580RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event227580 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56478⟩⟩) exact227580RawTerms (.finite 16) 227579 .exactZero (none)

def event227581 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56479⟩⟩) 0 ⟨56478⟩ 227580

def event227582 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56479⟩⟩) 1 ⟨24998⟩ 227577

def event227583 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56479⟩⟩) (.product (.predecessor 0 227581 .coefficient) (.predecessor 1 227582 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def eventLeaf14208 : Array AnnotatedEvent := #[
  { event := event227328
    frameStart := 227275 },
  { event := event227329
    frameStart := 227329 },
  { event := event227330
    frameStart := 227329 },
  { event := event227331
    frameStart := 227329 },
  { event := event227332
    frameStart := 227329 },
  { event := event227333
    frameStart := 227329 },
  { event := event227334
    frameStart := 227329 },
  { event := event227335
    frameStart := 227329 },
  { event := event227336
    frameStart := 227329 },
  { event := event227337
    frameStart := 227329 },
  { event := event227338
    frameStart := 227329 },
  { event := event227339
    frameStart := 227329 },
  { event := event227340
    frameStart := 227329 },
  { event := event227341
    frameStart := 227329 },
  { event := event227342
    frameStart := 227329 },
  { event := event227343
    frameStart := 227329 }
]

def eventLeaf14209 : Array AnnotatedEvent := #[
  { event := event227344
    frameStart := 227329 },
  { event := event227345
    frameStart := 227329 },
  { event := event227346
    frameStart := 227329 },
  { event := event227347
    frameStart := 227329 },
  { event := event227348
    frameStart := 227329 },
  { event := event227349
    frameStart := 227329 },
  { event := event227350
    frameStart := 227329 },
  { event := event227351
    frameStart := 227329 },
  { event := event227352
    frameStart := 227329 },
  { event := event227353
    frameStart := 227329 },
  { event := event227354
    frameStart := 227329 },
  { event := event227355
    frameStart := 227329 },
  { event := event227356
    frameStart := 227329 },
  { event := event227357
    frameStart := 227329 },
  { event := event227358
    frameStart := 227329 },
  { event := event227359
    frameStart := 227329 }
]

def eventLeaf14210 : Array AnnotatedEvent := #[
  { event := event227360
    frameStart := 227329 },
  { event := event227361
    frameStart := 227329 },
  { event := event227362
    frameStart := 227329 },
  { event := event227363
    frameStart := 227329 },
  { event := event227364
    frameStart := 227329 },
  { event := event227365
    frameStart := 227329 },
  { event := event227366
    frameStart := 227329 },
  { event := event227367
    frameStart := 227329 },
  { event := event227368
    frameStart := 227329 },
  { event := event227369
    frameStart := 227329 },
  { event := event227370
    frameStart := 227329 },
  { event := event227371
    frameStart := 227329 },
  { event := event227372
    frameStart := 227329 },
  { event := event227373
    frameStart := 227329 },
  { event := event227374
    frameStart := 227329 },
  { event := event227375
    frameStart := 227329 }
]

def eventLeaf14211 : Array AnnotatedEvent := #[
  { event := event227376
    frameStart := 227329 },
  { event := event227377
    frameStart := 227329 },
  { event := event227378
    frameStart := 227329 },
  { event := event227379
    frameStart := 227329 },
  { event := event227380
    frameStart := 227329 },
  { event := event227381
    frameStart := 227329 },
  { event := event227382
    frameStart := 227329 },
  { event := event227383
    frameStart := 227329 },
  { event := event227384
    frameStart := 227329 },
  { event := event227385
    frameStart := 227329 },
  { event := event227386
    frameStart := 227329 },
  { event := event227387
    frameStart := 227329 },
  { event := event227388
    frameStart := 227329 },
  { event := event227389
    frameStart := 227329 },
  { event := event227390
    frameStart := 227329 },
  { event := event227391
    frameStart := 227329 }
]

def eventLeaf14212 : Array AnnotatedEvent := #[
  { event := event227392
    frameStart := 227329 },
  { event := event227393
    frameStart := 227329 },
  { event := event227394
    frameStart := 227329 },
  { event := event227395
    frameStart := 227329 },
  { event := event227396
    frameStart := 227329 },
  { event := event227397
    frameStart := 227329 },
  { event := event227398
    frameStart := 227329 },
  { event := event227399
    frameStart := 227329 },
  { event := event227400
    frameStart := 227329 },
  { event := event227401
    frameStart := 227329 },
  { event := event227402
    frameStart := 227329 },
  { event := event227403
    frameStart := 227329 },
  { event := event227404
    frameStart := 227329 },
  { event := event227405
    frameStart := 227329 },
  { event := event227406
    frameStart := 227329 },
  { event := event227407
    frameStart := 227329 }
]

def eventLeaf14213 : Array AnnotatedEvent := #[
  { event := event227408
    frameStart := 227329 },
  { event := event227409
    frameStart := 227329 },
  { event := event227410
    frameStart := 227329 },
  { event := event227411
    frameStart := 227329 },
  { event := event227412
    frameStart := 227329 },
  { event := event227413
    frameStart := 227329 },
  { event := event227414
    frameStart := 227329 },
  { event := event227415
    frameStart := 227329 },
  { event := event227416
    frameStart := 227329 },
  { event := event227417
    frameStart := 227329 },
  { event := event227418
    frameStart := 227329 },
  { event := event227419
    frameStart := 227329 },
  { event := event227420
    frameStart := 227329 },
  { event := event227421
    frameStart := 227329 },
  { event := event227422
    frameStart := 227329 },
  { event := event227423
    frameStart := 227329 }
]

def eventLeaf14214 : Array AnnotatedEvent := #[
  { event := event227424
    frameStart := 227329 },
  { event := event227425
    frameStart := 227329 },
  { event := event227426
    frameStart := 227329 },
  { event := event227427
    frameStart := 227329 },
  { event := event227428
    frameStart := 227329 },
  { event := event227429
    frameStart := 227329 },
  { event := event227430
    frameStart := 227329 },
  { event := event227431
    frameStart := 227329 },
  { event := event227432
    frameStart := 227329 },
  { event := event227433
    frameStart := 0 },
  { event := event227434
    frameStart := 0 },
  { event := event227435
    frameStart := 0 },
  { event := event227436
    frameStart := 0 },
  { event := event227437
    frameStart := 0 },
  { event := event227438
    frameStart := 0 },
  { event := event227439
    frameStart := 0 }
]

def eventLeaf14215 : Array AnnotatedEvent := #[
  { event := event227440
    frameStart := 0 },
  { event := event227441
    frameStart := 0 },
  { event := event227442
    frameStart := 0 },
  { event := event227443
    frameStart := 0 },
  { event := event227444
    frameStart := 0 },
  { event := event227445
    frameStart := 0 },
  { event := event227446
    frameStart := 0 },
  { event := event227447
    frameStart := 0 },
  { event := event227448
    frameStart := 0 },
  { event := event227449
    frameStart := 0 },
  { event := event227450
    frameStart := 0 },
  { event := event227451
    frameStart := 0 },
  { event := event227452
    frameStart := 0 },
  { event := event227453
    frameStart := 0 },
  { event := event227454
    frameStart := 0 },
  { event := event227455
    frameStart := 0 }
]

def eventLeaf14216 : Array AnnotatedEvent := #[
  { event := event227456
    frameStart := 0 },
  { event := event227457
    frameStart := 0 },
  { event := event227458
    frameStart := 0 },
  { event := event227459
    frameStart := 0 },
  { event := event227460
    frameStart := 0 },
  { event := event227461
    frameStart := 0 },
  { event := event227462
    frameStart := 0 },
  { event := event227463
    frameStart := 0 },
  { event := event227464
    frameStart := 0 },
  { event := event227465
    frameStart := 0 },
  { event := event227466
    frameStart := 0 },
  { event := event227467
    frameStart := 0 },
  { event := event227468
    frameStart := 0 },
  { event := event227469
    frameStart := 0 },
  { event := event227470
    frameStart := 0 },
  { event := event227471
    frameStart := 0 }
]

def eventLeaf14217 : Array AnnotatedEvent := #[
  { event := event227472
    frameStart := 0 },
  { event := event227473
    frameStart := 0 },
  { event := event227474
    frameStart := 0 },
  { event := event227475
    frameStart := 0 },
  { event := event227476
    frameStart := 0 },
  { event := event227477
    frameStart := 0 },
  { event := event227478
    frameStart := 0 },
  { event := event227479
    frameStart := 0 },
  { event := event227480
    frameStart := 0 },
  { event := event227481
    frameStart := 0 },
  { event := event227482
    frameStart := 0 },
  { event := event227483
    frameStart := 0 },
  { event := event227484
    frameStart := 0 },
  { event := event227485
    frameStart := 0 },
  { event := event227486
    frameStart := 0 },
  { event := event227487
    frameStart := 0 }
]

def eventLeaf14218 : Array AnnotatedEvent := #[
  { event := event227488
    frameStart := 0 },
  { event := event227489
    frameStart := 0 },
  { event := event227490
    frameStart := 0 },
  { event := event227491
    frameStart := 0 },
  { event := event227492
    frameStart := 0 },
  { event := event227493
    frameStart := 0 },
  { event := event227494
    frameStart := 0 },
  { event := event227495
    frameStart := 0 },
  { event := event227496
    frameStart := 0 },
  { event := event227497
    frameStart := 0 },
  { event := event227498
    frameStart := 0 },
  { event := event227499
    frameStart := 0 },
  { event := event227500
    frameStart := 0 },
  { event := event227501
    frameStart := 0 },
  { event := event227502
    frameStart := 0 },
  { event := event227503
    frameStart := 0 }
]

def eventLeaf14219 : Array AnnotatedEvent := #[
  { event := event227504
    frameStart := 0 },
  { event := event227505
    frameStart := 0 },
  { event := event227506
    frameStart := 0 },
  { event := event227507
    frameStart := 0 },
  { event := event227508
    frameStart := 0 },
  { event := event227509
    frameStart := 0 },
  { event := event227510
    frameStart := 0 },
  { event := event227511
    frameStart := 0 },
  { event := event227512
    frameStart := 0 },
  { event := event227513
    frameStart := 0 },
  { event := event227514
    frameStart := 0 },
  { event := event227515
    frameStart := 0 },
  { event := event227516
    frameStart := 0 },
  { event := event227517
    frameStart := 0 },
  { event := event227518
    frameStart := 0 },
  { event := event227519
    frameStart := 0 }
]

def eventLeaf14220 : Array AnnotatedEvent := #[
  { event := event227520
    frameStart := 0 },
  { event := event227521
    frameStart := 0 },
  { event := event227522
    frameStart := 0 },
  { event := event227523
    frameStart := 0 },
  { event := event227524
    frameStart := 0 },
  { event := event227525
    frameStart := 0 },
  { event := event227526
    frameStart := 0 },
  { event := event227527
    frameStart := 0 },
  { event := event227528
    frameStart := 0 },
  { event := event227529
    frameStart := 0 },
  { event := event227530
    frameStart := 0 },
  { event := event227531
    frameStart := 0 },
  { event := event227532
    frameStart := 0 },
  { event := event227533
    frameStart := 0 },
  { event := event227534
    frameStart := 0 },
  { event := event227535
    frameStart := 0 }
]

def eventLeaf14221 : Array AnnotatedEvent := #[
  { event := event227536
    frameStart := 0 },
  { event := event227537
    frameStart := 0 },
  { event := event227538
    frameStart := 0 },
  { event := event227539
    frameStart := 0 },
  { event := event227540
    frameStart := 0 },
  { event := event227541
    frameStart := 0 },
  { event := event227542
    frameStart := 0 },
  { event := event227543
    frameStart := 0 },
  { event := event227544
    frameStart := 0 },
  { event := event227545
    frameStart := 0 },
  { event := event227546
    frameStart := 0 },
  { event := event227547
    frameStart := 0 },
  { event := event227548
    frameStart := 0 },
  { event := event227549
    frameStart := 0 },
  { event := event227550
    frameStart := 0 },
  { event := event227551
    frameStart := 0 }
]

def eventLeaf14222 : Array AnnotatedEvent := #[
  { event := event227552
    frameStart := 0 },
  { event := event227553
    frameStart := 0 },
  { event := event227554
    frameStart := 227554 },
  { event := event227555
    frameStart := 227554 },
  { event := event227556
    frameStart := 227554 },
  { event := event227557
    frameStart := 227554 },
  { event := event227558
    frameStart := 227554 },
  { event := event227559
    frameStart := 227554 },
  { event := event227560
    frameStart := 227554 },
  { event := event227561
    frameStart := 227554 },
  { event := event227562
    frameStart := 227554 },
  { event := event227563
    frameStart := 227554 },
  { event := event227564
    frameStart := 227554 },
  { event := event227565
    frameStart := 227554 },
  { event := event227566
    frameStart := 227554 },
  { event := event227567
    frameStart := 227554 }
]

def eventLeaf14223 : Array AnnotatedEvent := #[
  { event := event227568
    frameStart := 227554 },
  { event := event227569
    frameStart := 227554 },
  { event := event227570
    frameStart := 227554 },
  { event := event227571
    frameStart := 227554 },
  { event := event227572
    frameStart := 227554 },
  { event := event227573
    frameStart := 227554 },
  { event := event227574
    frameStart := 227554 },
  { event := event227575
    frameStart := 227554 },
  { event := event227576
    frameStart := 227554 },
  { event := event227577
    frameStart := 227554 },
  { event := event227578
    frameStart := 227554 },
  { event := event227579
    frameStart := 227554 },
  { event := event227580
    frameStart := 227554 },
  { event := event227581
    frameStart := 227554 },
  { event := event227582
    frameStart := 227554 },
  { event := event227583
    frameStart := 227554 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events888
