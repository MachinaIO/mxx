import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events267

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event68352 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 68336

def event68353 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 68352 .coefficient))

def event68354 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event68355 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11949⟩⟩) 0 ⟨5530⟩ 68354

def event68356 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11949⟩⟩) (.authority (.programFamilyFact))

def exact68357RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11949⟩⟩], []⟩, (1)⟩]

theorem exact68357RawTermsValid :
    exact68357RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68357 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11949⟩⟩) exact68357RawTerms (.finite 36) 68356 .exactZero (none)

def event68358 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9710⟩⟩) 0 ⟨5530⟩ 68354

def event68359 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9710⟩⟩) (.authority (.programFamilyFact))

def exact68360RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩], []⟩, (1)⟩]

theorem exact68360RawTermsValid :
    exact68360RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68360 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9710⟩⟩) exact68360RawTerms (.finite 36) 68359 .exactZero (none)

def event68361 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11950⟩⟩) 0 ⟨9710⟩ 68360

def event68362 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11950⟩⟩) 1 ⟨11949⟩ 68357

def event68363 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11950⟩⟩) (.product (.predecessor 0 68361 .coefficient) (.predecessor 1 68362 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event68364 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11950⟩⟩, .operator (⟨68360, 0⟩, ⟨68357, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], []⟩, (1)⟩)

def exact68365RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], []⟩, (1)⟩]

theorem exact68365RawTermsValid :
    exact68365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68365 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11950⟩⟩) exact68365RawTerms (.finite 1296) 68363 .exactZero (none)

def event68366 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11951⟩⟩) 0 ⟨11950⟩ 68365

def event68367 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11951⟩⟩) (.identity (.predecessor 0 68366 .coefficient))

def event68368 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11951⟩⟩) (.finite 1296)

def event68369 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23119⟩⟩) 0 ⟨11951⟩ 68368

def event68370 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23119⟩⟩) (.authority (.programFamilyFact))

def event68371 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23119⟩⟩) (.finite 3720)

def event68372 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event68373 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23120⟩⟩) 0 ⟨6689⟩ 68372

def event68374 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23120⟩⟩) 1 ⟨23119⟩ 68371

def event68375 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23120⟩⟩) (.authority (.operator))

def exact68376RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23120⟩⟩]⟩, (1)⟩]

theorem exact68376RawTermsValid :
    exact68376RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68376 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23120⟩⟩) exact68376RawTerms .large 68375 .exactZero (none)

def event68377 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25214⟩⟩) 0 ⟨23120⟩ 68376

def event68378 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25214⟩⟩) (.authority (.operator))

def exact68379RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩, (1)⟩]

theorem exact68379RawTermsValid :
    exact68379RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68379 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25214⟩⟩) exact68379RawTerms (.finite 8192) 68378 .exactZero (none)

def event68380 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event68381 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event68382 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12049⟩⟩) 0 ⟨11951⟩ 68368

def event68383 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12049⟩⟩) 1 ⟨110⟩ 68381

def event68384 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12049⟩⟩) (.sum [.predecessor 0 68382 .coefficient, .predecessor 1 68383 .coefficient])

def event68385 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12049⟩⟩) (.finite 1296)

def event68386 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12050⟩⟩) 0 ⟨12049⟩ 68385

def event68387 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12050⟩⟩) (.identity (.predecessor 0 68386 .coefficient))

def exact68388RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], []⟩, (1)⟩]

theorem exact68388RawTermsValid :
    exact68388RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68388 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12050⟩⟩) exact68388RawTerms (.finite 1296) 68387 .exactZero (none)

def event68389 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact68390RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact68390RawTermsValid :
    exact68390RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68390 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact68390RawTerms .large 68389 .exactZero (none)

def event68391 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12051⟩⟩) 0 ⟨6544⟩ 68390

def event68392 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12051⟩⟩) 1 ⟨12050⟩ 68388

def event68393 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12051⟩⟩) (.product (.predecessor 0 68391 .coefficient) (.predecessor 1 68392 .coefficient) (⟨false, false, none, none, none⟩))

def event68394 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12051⟩⟩, .operator (⟨68390, 0⟩, ⟨68388, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact68395RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact68395RawTermsValid :
    exact68395RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68395 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12051⟩⟩) exact68395RawTerms .large 68393 .exactZero (none)

def event68396 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event68397 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event68398 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 68372

def event68399 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact68400RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact68400RawTermsValid :
    exact68400RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68400 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact68400RawTerms .large 68399 .exactZero (none)

def event68401 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6784⟩⟩) 0 ⟨6757⟩ 68400

def event68402 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6784⟩⟩) (.identity (.predecessor 0 68401 .coefficient))

def exact68403RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩]

theorem exact68403RawTermsValid :
    exact68403RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68403 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6784⟩⟩) exact68403RawTerms .large 68402 .exactZero (none)

def event68404 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7864⟩⟩) 0 ⟨6784⟩ 68403

def event68405 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7864⟩⟩) (.authority (.operator))

def exact68406RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩, (1)⟩]

theorem exact68406RawTermsValid :
    exact68406RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68406 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7864⟩⟩) exact68406RawTerms (.finite 8192) 68405 .exactZero (none)

def event68407 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7865⟩⟩) 0 ⟨7864⟩ 68406

def event68408 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7865⟩⟩) 1 ⟨2348⟩ 68397

def event68409 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7865⟩⟩) (.scale (.predecessor 0 68407 .coefficient) (.value (.predecessor 1 68408 .coefficient)))

def exact68410RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩, (1)⟩]

theorem exact68410RawTermsValid :
    exact68410RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68410 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7865⟩⟩) exact68410RawTerms (.finite 8192) 68409 .exactZero (none)

def event68411 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6764⟩⟩) 0 ⟨6757⟩ 68400

def event68412 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6764⟩⟩) (.identity (.predecessor 0 68411 .coefficient))

def exact68413RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩]⟩, (1)⟩]

theorem exact68413RawTermsValid :
    exact68413RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68413 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6764⟩⟩) exact68413RawTerms .large 68412 .exactZero (none)

def event68414 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7866⟩⟩) 0 ⟨6764⟩ 68413

def event68415 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7866⟩⟩) 1 ⟨7865⟩ 68410

def event68416 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7866⟩⟩) (.product (.predecessor 0 68414 .coefficient) (.predecessor 1 68415 .coefficient) (⟨false, false, none, none, none⟩))

def event68417 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7866⟩⟩, .operator (⟨68413, 0⟩, ⟨68410, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩, (1)⟩)

def exact68418RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩, (1)⟩]

theorem exact68418RawTermsValid :
    exact68418RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68418 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7866⟩⟩) exact68418RawTerms .large 68416 .exactZero (none)

def event68419 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12052⟩⟩) 0 ⟨7866⟩ 68418

def event68420 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12052⟩⟩) 1 ⟨12051⟩ 68395

def event68421 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12052⟩⟩) (.sum [.predecessor 0 68419 .coefficient, .predecessor 1 68420 .coefficient])

def exact68422RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68422RawTermsValid :
    exact68422RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68422 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12052⟩⟩) exact68422RawTerms .large 68421 .exactZero (none)

def event68423 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25217⟩⟩) 0 ⟨12052⟩ 68422

def event68424 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25217⟩⟩) 1 ⟨25214⟩ 68379

def event68425 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25217⟩⟩) (.product (.predecessor 0 68423 .coefficient) (.predecessor 1 68424 .coefficient) (⟨false, false, none, none, none⟩))

def event68426 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25217⟩⟩, .operator (⟨68422, 0⟩, ⟨68379, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩, (1)⟩)

def event68427 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25217⟩⟩, .operator (⟨68422, 1⟩, ⟨68379, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩, (-1)⟩)

def event68428 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25217⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25214⟩⟩) ⟨23120⟩ 68376)

def event68429 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25217⟩⟩, .relation 68428 0, ⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨23120⟩⟩]⟩, (-1)⟩)

def exact68430RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨23120⟩⟩]⟩, (-1)⟩]

theorem exact68430RawTermsValid :
    exact68430RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68430 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25217⟩⟩) exact68430RawTerms .large 68425 .exactZero (none)

def event68431 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16377⟩⟩) 0 ⟨11951⟩ 68368

def event68432 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16377⟩⟩) (.authority (.programFamilyFact))

def exact68433RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], []⟩, (1)⟩]

theorem exact68433RawTermsValid :
    exact68433RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68433 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16377⟩⟩) exact68433RawTerms (.finite 36) 68432 .exactZero (none)

def event68434 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16379⟩⟩) 0 ⟨6544⟩ 68390

def event68435 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16379⟩⟩) 1 ⟨16377⟩ 68433

def event68436 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16379⟩⟩) (.product (.predecessor 0 68434 .coefficient) (.predecessor 1 68435 .coefficient) (⟨false, true, none, none, some 1⟩))

def event68437 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16379⟩⟩, .operator (⟨68390, 0⟩, ⟨68433, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact68438RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact68438RawTermsValid :
    exact68438RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68438 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16379⟩⟩) exact68438RawTerms .large 68436 .exactZero (none)

def event68439 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6701⟩⟩) 0 ⟨6689⟩ 68372

def event68440 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6701⟩⟩) (.authority (.operator))

def exact68441RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩]

theorem exact68441RawTermsValid :
    exact68441RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68441 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6701⟩⟩) exact68441RawTerms .large 68440 .exactZero (none)

def event68442 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16380⟩⟩) 0 ⟨6701⟩ 68441

def event68443 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16380⟩⟩) 1 ⟨16379⟩ 68438

def event68444 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16380⟩⟩) (.sum [.predecessor 0 68442 .coefficient, .predecessor 1 68443 .coefficient])

def exact68445RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68445RawTermsValid :
    exact68445RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68445 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16380⟩⟩) exact68445RawTerms .large 68444 .exactZero (none)

def event68446 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25218⟩⟩) 0 ⟨16380⟩ 68445

def event68447 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25218⟩⟩) 1 ⟨25217⟩ 68430

def event68448 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25218⟩⟩) (.sum [.predecessor 0 68446 .coefficient, .predecessor 1 68447 .coefficient])

def exact68449RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨23120⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68449RawTermsValid :
    exact68449RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68449 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25218⟩⟩) exact68449RawTerms .large 68448 .exactZero (none)

def event68450 : Event := .preFoldPolynomial 68449 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨23120⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact68451RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨23120⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event68451 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25218⟩⟩) 68450 exact68451RawTerms .large 68448 .exactZero (none)

def event68452 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨11951⟩⟩) ⟨⟨114⟩, ⟨19⟩, ⟨109⟩⟩ ⟨68286, 68452⟩

def event68453 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19815⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩) (1) 0 2 (.universal 68452 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19812⟩⟩]⟩) (none) 68451)

def event68454 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19815⟩⟩, .relation 68453 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩)

def event68455 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19815⟩⟩, .relation 68453 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩, (-1)⟩)

def event68456 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19815⟩⟩, .relation 68453 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨23120⟩⟩]⟩, (1)⟩)

def event68457 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19815⟩⟩, .relation 68453 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact68458RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨23120⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68458RawTermsValid :
    exact68458RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68458 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19815⟩⟩) exact68458RawTerms .large 68282 (.finite 1811303510016) (some (68284))

def event68459 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25216⟩⟩) 0 ⟨19815⟩ 68458

def event68460 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25216⟩⟩) 1 ⟨25215⟩ 68272

def event68461 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25216⟩⟩) (.sum [.predecessor 0 68459 .coefficient, .predecessor 1 68460 .coefficient])

def event68462 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25216⟩⟩, .operator (⟨68458, 2⟩, ⟨68272, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨23120⟩⟩]⟩, (-1)⟩)

def event68463 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25216⟩⟩, .operator (⟨68458, 1⟩, ⟨68272, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩, (1)⟩)

def event68464 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25216⟩⟩) (.sum [.result 68458 .summary, .result 68272 .summary])

def exact68465RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact68465RawTermsValid :
    exact68465RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68465 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25216⟩⟩) exact68465RawTerms .large 68461 (.finite 352115681275904) (some (68464))

def event68466 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28723⟩⟩) 0 ⟨25216⟩ 68465

def event68467 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28723⟩⟩) 1 ⟨28721⟩ 68188

def event68468 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28723⟩⟩) (.product (.predecessor 0 68466 .coefficient) (.predecessor 1 68467 .coefficient) (⟨false, false, none, none, none⟩))

def event68469 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28723⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28721⟩⟩]⟩) [⟨.result 68188 .coefficient, false, none⟩])

def event68470 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28723⟩⟩) (.product (.result 68465 .summary) (.transfer 68469) (⟨false, false, none, none, none⟩))

def event68471 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28723⟩⟩, .operator (⟨68465, 0⟩, ⟨68188, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28721⟩⟩]⟩, (1)⟩)

def event68472 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28723⟩⟩, .operator (⟨68465, 1⟩, ⟨68188, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28721⟩⟩]⟩, (-1)⟩)

def event68473 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28723⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28721⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28721⟩⟩) ⟨24411⟩ 68185)

def event68474 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28723⟩⟩, .relation 68473 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨24411⟩⟩]⟩, (-1)⟩)

def exact68475RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28721⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16377⟩⟩], [⟨.program ⟨214⟩, ⟨24411⟩⟩]⟩, (-1)⟩]

theorem exact68475RawTermsValid :
    exact68475RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68475 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28723⟩⟩) exact68475RawTerms .large 68468 (.finite 1292270184133468094464) (some (68470))

def event68476 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21972⟩⟩) 0 ⟨16378⟩ 3241

def event68477 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21972⟩⟩) (.authority (.relationPreimageSource ⟨52⟩))

def exact68478RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21972⟩⟩]⟩, (1)⟩]

theorem exact68478RawTermsValid :
    exact68478RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68478 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21972⟩⟩) exact68478RawTerms (.finite 136065468) 68477 .exactZero (none)

def event68479 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21974⟩⟩) 0 ⟨21972⟩ 68478

def event68480 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21974⟩⟩) 1 ⟨2348⟩ 4

def event68481 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21974⟩⟩) (.scale (.predecessor 0 68479 .coefficient) (.value (.predecessor 1 68480 .coefficient)))

def exact68482RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21972⟩⟩]⟩, (1)⟩]

theorem exact68482RawTermsValid :
    exact68482RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68482 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21974⟩⟩) exact68482RawTerms (.finite 136065468) 68481 .exactZero (none)

def event68483 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21975⟩⟩) 0 ⟨5535⟩ 65387

def event68484 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21975⟩⟩) 1 ⟨21974⟩ 68482

def event68485 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21975⟩⟩) (.product (.predecessor 0 68483 .coefficient) (.predecessor 1 68484 .coefficient) (⟨false, false, none, none, none⟩))

def event68486 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21975⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21972⟩⟩]⟩) [⟨.result 68478 .coefficient, false, none⟩])

def event68487 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21975⟩⟩) (.product (.result 65387 .summary) (.transfer 68486) (⟨false, false, none, none, none⟩))

def event68488 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21975⟩⟩, .operator (⟨65387, 0⟩, ⟨68482, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21972⟩⟩]⟩, (1)⟩)

def event68489 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21973⟩⟩)

def event68490 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event68491 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event68492 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event68493 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event68494 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event68495 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event68496 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event68497 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event68498 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 68497

def event68499 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 68495

def event68500 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 68498 .coefficient) (.value (.predecessor 1 68499 .coefficient)))

def event68501 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event68502 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 68501

def event68503 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 68493

def event68504 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 68502 .coefficient, .predecessor 1 68503 .coefficient])

def event68505 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event68506 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 68505

def event68507 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 68491

def event68508 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 68507 .coefficient))

def event68509 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event68510 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11949⟩⟩) 0 ⟨5530⟩ 68509

def event68511 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11949⟩⟩) (.authority (.programFamilyFact))

def exact68512RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11949⟩⟩], []⟩, (1)⟩]

theorem exact68512RawTermsValid :
    exact68512RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68512 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11949⟩⟩) exact68512RawTerms (.finite 36) 68511 .exactZero (none)

def event68513 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9710⟩⟩) 0 ⟨5530⟩ 68509

def event68514 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9710⟩⟩) (.authority (.programFamilyFact))

def exact68515RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩], []⟩, (1)⟩]

theorem exact68515RawTermsValid :
    exact68515RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68515 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9710⟩⟩) exact68515RawTerms (.finite 36) 68514 .exactZero (none)

def event68516 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11950⟩⟩) 0 ⟨9710⟩ 68515

def event68517 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11950⟩⟩) 1 ⟨11949⟩ 68512

def event68518 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11950⟩⟩) (.product (.predecessor 0 68516 .coefficient) (.predecessor 1 68517 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event68519 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11950⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], []⟩) [⟨.result 68515 .coefficient, true, some 1⟩, ⟨.result 68512 .coefficient, true, some 1⟩])

def event68520 : Event := .survivorFold (1) 68519

def exact68521RawTerms : List Term := []

theorem exact68521RawTermsValid :
    exact68521RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68521 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11950⟩⟩) exact68521RawTerms (.finite 1296) 68518 (.finite 1296) (some (68519))

def event68522 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11951⟩⟩) 0 ⟨11950⟩ 68521

def event68523 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11951⟩⟩) (.identity (.predecessor 0 68522 .coefficient))

def event68524 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11951⟩⟩) (.finite 1296)

def event68525 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16377⟩⟩) 0 ⟨11951⟩ 68524

def event68526 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16377⟩⟩) (.authority (.programFamilyFact))

def exact68527RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], []⟩, (1)⟩]

theorem exact68527RawTermsValid :
    exact68527RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68527 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16377⟩⟩) exact68527RawTerms (.finite 36) 68526 .exactZero (none)

def event68528 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16378⟩⟩) 0 ⟨16377⟩ 68527

def event68529 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16378⟩⟩) (.identity (.predecessor 0 68528 .coefficient))

def event68530 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16378⟩⟩) (.finite 36)

def event68531 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21972⟩⟩) 0 ⟨16378⟩ 68530

def event68532 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21972⟩⟩) (.authority (.relationPreimageSource ⟨52⟩))

def exact68533RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21972⟩⟩]⟩, (1)⟩]

theorem exact68533RawTermsValid :
    exact68533RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68533 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21972⟩⟩) exact68533RawTerms (.finite 136065468) 68532 .exactZero (none)

def event68534 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact68535RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact68535RawTermsValid :
    exact68535RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68535 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact68535RawTerms .large 68534 .exactZero (none)

def event68536 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21973⟩⟩) 0 ⟨6⟩ 68535

def event68537 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21973⟩⟩) 1 ⟨21972⟩ 68533

def event68538 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21973⟩⟩) (.product (.predecessor 0 68536 .coefficient) (.predecessor 1 68537 .coefficient) (⟨false, false, none, none, none⟩))

def event68539 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21973⟩⟩, .operator (⟨68535, 0⟩, ⟨68533, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21972⟩⟩]⟩, (1)⟩)

def exact68540RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21972⟩⟩]⟩, (1)⟩]

theorem exact68540RawTermsValid :
    exact68540RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68540 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21973⟩⟩) exact68540RawTerms .large 68538 .exactZero (none)

def event68541 : Event := .preFoldPolynomial 68540 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21972⟩⟩]⟩, (1)⟩] .exactZero none

def exact68542RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21972⟩⟩]⟩, (1)⟩]

def event68542 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21973⟩⟩) 68541 exact68542RawTerms .large 68538 .exactZero (none)

def event68543 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28726⟩⟩)

def event68544 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event68545 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event68546 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event68547 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event68548 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event68549 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event68550 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event68551 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event68552 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 68551

def event68553 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 68549

def event68554 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 68552 .coefficient) (.value (.predecessor 1 68553 .coefficient)))

def event68555 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event68556 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 68555

def event68557 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 68547

def event68558 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 68556 .coefficient, .predecessor 1 68557 .coefficient])

def event68559 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event68560 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 68559

def event68561 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 68545

def event68562 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 68561 .coefficient))

def event68563 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event68564 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11949⟩⟩) 0 ⟨5530⟩ 68563

def event68565 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11949⟩⟩) (.authority (.programFamilyFact))

def exact68566RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11949⟩⟩], []⟩, (1)⟩]

theorem exact68566RawTermsValid :
    exact68566RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68566 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11949⟩⟩) exact68566RawTerms (.finite 36) 68565 .exactZero (none)

def event68567 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9710⟩⟩) 0 ⟨5530⟩ 68563

def event68568 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9710⟩⟩) (.authority (.programFamilyFact))

def exact68569RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩], []⟩, (1)⟩]

theorem exact68569RawTermsValid :
    exact68569RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68569 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9710⟩⟩) exact68569RawTerms (.finite 36) 68568 .exactZero (none)

def event68570 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11950⟩⟩) 0 ⟨9710⟩ 68569

def event68571 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11950⟩⟩) 1 ⟨11949⟩ 68566

def event68572 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11950⟩⟩) (.product (.predecessor 0 68570 .coefficient) (.predecessor 1 68571 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event68573 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11950⟩⟩, .operator (⟨68569, 0⟩, ⟨68566, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], []⟩, (1)⟩)

def exact68574RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], []⟩, (1)⟩]

theorem exact68574RawTermsValid :
    exact68574RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68574 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11950⟩⟩) exact68574RawTerms (.finite 1296) 68572 .exactZero (none)

def event68575 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11951⟩⟩) 0 ⟨11950⟩ 68574

def event68576 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11951⟩⟩) (.identity (.predecessor 0 68575 .coefficient))

def event68577 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11951⟩⟩) (.finite 1296)

def event68578 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16377⟩⟩) 0 ⟨11951⟩ 68577

def event68579 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16377⟩⟩) (.authority (.programFamilyFact))

def exact68580RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], []⟩, (1)⟩]

theorem exact68580RawTermsValid :
    exact68580RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68580 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16377⟩⟩) exact68580RawTerms (.finite 36) 68579 .exactZero (none)

def event68581 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16378⟩⟩) 0 ⟨16377⟩ 68580

def event68582 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16378⟩⟩) (.identity (.predecessor 0 68581 .coefficient))

def event68583 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16378⟩⟩) (.finite 36)

def event68584 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24409⟩⟩) 0 ⟨16378⟩ 68583

def event68585 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24409⟩⟩) (.authority (.programFamilyFact))

def event68586 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24409⟩⟩) (.finite 3720)

def event68587 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event68588 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24411⟩⟩) 0 ⟨6689⟩ 68587

def event68589 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24411⟩⟩) 1 ⟨24409⟩ 68586

def event68590 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24411⟩⟩) (.authority (.operator))

def exact68591RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24411⟩⟩]⟩, (1)⟩]

theorem exact68591RawTermsValid :
    exact68591RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68591 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24411⟩⟩) exact68591RawTerms .large 68590 .exactZero (none)

def event68592 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28721⟩⟩) 0 ⟨24411⟩ 68591

def event68593 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28721⟩⟩) (.authority (.operator))

def exact68594RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28721⟩⟩]⟩, (1)⟩]

theorem exact68594RawTermsValid :
    exact68594RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68594 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28721⟩⟩) exact68594RawTerms (.finite 8192) 68593 .exactZero (none)

def event68595 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event68596 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event68597 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16417⟩⟩) 0 ⟨16378⟩ 68583

def event68598 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16417⟩⟩) 1 ⟨110⟩ 68596

def event68599 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16417⟩⟩) (.sum [.predecessor 0 68597 .coefficient, .predecessor 1 68598 .coefficient])

def event68600 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16417⟩⟩) (.finite 36)

def event68601 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16418⟩⟩) 0 ⟨16417⟩ 68600

def event68602 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16418⟩⟩) (.identity (.predecessor 0 68601 .coefficient))

def exact68603RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16377⟩⟩], []⟩, (1)⟩]

theorem exact68603RawTermsValid :
    exact68603RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68603 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16418⟩⟩) exact68603RawTerms (.finite 36) 68602 .exactZero (none)

def event68604 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact68605RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact68605RawTermsValid :
    exact68605RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68605 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact68605RawTerms .large 68604 .exactZero (none)

def event68606 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16419⟩⟩) 0 ⟨6544⟩ 68605

def event68607 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16419⟩⟩) 1 ⟨16418⟩ 68603

def eventLeaf4272 : Array AnnotatedEvent := #[
  { event := event68352
    frameStart := 68334 },
  { event := event68353
    frameStart := 68334 },
  { event := event68354
    frameStart := 68334 },
  { event := event68355
    frameStart := 68334 },
  { event := event68356
    frameStart := 68334 },
  { event := event68357
    frameStart := 68334 },
  { event := event68358
    frameStart := 68334 },
  { event := event68359
    frameStart := 68334 },
  { event := event68360
    frameStart := 68334 },
  { event := event68361
    frameStart := 68334 },
  { event := event68362
    frameStart := 68334 },
  { event := event68363
    frameStart := 68334 },
  { event := event68364
    frameStart := 68334 },
  { event := event68365
    frameStart := 68334 },
  { event := event68366
    frameStart := 68334 },
  { event := event68367
    frameStart := 68334 }
]

def eventLeaf4273 : Array AnnotatedEvent := #[
  { event := event68368
    frameStart := 68334 },
  { event := event68369
    frameStart := 68334 },
  { event := event68370
    frameStart := 68334 },
  { event := event68371
    frameStart := 68334 },
  { event := event68372
    frameStart := 68334 },
  { event := event68373
    frameStart := 68334 },
  { event := event68374
    frameStart := 68334 },
  { event := event68375
    frameStart := 68334 },
  { event := event68376
    frameStart := 68334 },
  { event := event68377
    frameStart := 68334 },
  { event := event68378
    frameStart := 68334 },
  { event := event68379
    frameStart := 68334 },
  { event := event68380
    frameStart := 68334 },
  { event := event68381
    frameStart := 68334 },
  { event := event68382
    frameStart := 68334 },
  { event := event68383
    frameStart := 68334 }
]

def eventLeaf4274 : Array AnnotatedEvent := #[
  { event := event68384
    frameStart := 68334 },
  { event := event68385
    frameStart := 68334 },
  { event := event68386
    frameStart := 68334 },
  { event := event68387
    frameStart := 68334 },
  { event := event68388
    frameStart := 68334 },
  { event := event68389
    frameStart := 68334 },
  { event := event68390
    frameStart := 68334 },
  { event := event68391
    frameStart := 68334 },
  { event := event68392
    frameStart := 68334 },
  { event := event68393
    frameStart := 68334 },
  { event := event68394
    frameStart := 68334 },
  { event := event68395
    frameStart := 68334 },
  { event := event68396
    frameStart := 68334 },
  { event := event68397
    frameStart := 68334 },
  { event := event68398
    frameStart := 68334 },
  { event := event68399
    frameStart := 68334 }
]

def eventLeaf4275 : Array AnnotatedEvent := #[
  { event := event68400
    frameStart := 68334 },
  { event := event68401
    frameStart := 68334 },
  { event := event68402
    frameStart := 68334 },
  { event := event68403
    frameStart := 68334 },
  { event := event68404
    frameStart := 68334 },
  { event := event68405
    frameStart := 68334 },
  { event := event68406
    frameStart := 68334 },
  { event := event68407
    frameStart := 68334 },
  { event := event68408
    frameStart := 68334 },
  { event := event68409
    frameStart := 68334 },
  { event := event68410
    frameStart := 68334 },
  { event := event68411
    frameStart := 68334 },
  { event := event68412
    frameStart := 68334 },
  { event := event68413
    frameStart := 68334 },
  { event := event68414
    frameStart := 68334 },
  { event := event68415
    frameStart := 68334 }
]

def eventLeaf4276 : Array AnnotatedEvent := #[
  { event := event68416
    frameStart := 68334 },
  { event := event68417
    frameStart := 68334 },
  { event := event68418
    frameStart := 68334 },
  { event := event68419
    frameStart := 68334 },
  { event := event68420
    frameStart := 68334 },
  { event := event68421
    frameStart := 68334 },
  { event := event68422
    frameStart := 68334 },
  { event := event68423
    frameStart := 68334 },
  { event := event68424
    frameStart := 68334 },
  { event := event68425
    frameStart := 68334 },
  { event := event68426
    frameStart := 68334 },
  { event := event68427
    frameStart := 68334 },
  { event := event68428
    frameStart := 68334 },
  { event := event68429
    frameStart := 68334 },
  { event := event68430
    frameStart := 68334 },
  { event := event68431
    frameStart := 68334 }
]

def eventLeaf4277 : Array AnnotatedEvent := #[
  { event := event68432
    frameStart := 68334 },
  { event := event68433
    frameStart := 68334 },
  { event := event68434
    frameStart := 68334 },
  { event := event68435
    frameStart := 68334 },
  { event := event68436
    frameStart := 68334 },
  { event := event68437
    frameStart := 68334 },
  { event := event68438
    frameStart := 68334 },
  { event := event68439
    frameStart := 68334 },
  { event := event68440
    frameStart := 68334 },
  { event := event68441
    frameStart := 68334 },
  { event := event68442
    frameStart := 68334 },
  { event := event68443
    frameStart := 68334 },
  { event := event68444
    frameStart := 68334 },
  { event := event68445
    frameStart := 68334 },
  { event := event68446
    frameStart := 68334 },
  { event := event68447
    frameStart := 68334 }
]

def eventLeaf4278 : Array AnnotatedEvent := #[
  { event := event68448
    frameStart := 68334 },
  { event := event68449
    frameStart := 68334 },
  { event := event68450
    frameStart := 68334 },
  { event := event68451
    frameStart := 68334 },
  { event := event68452
    frameStart := 0 },
  { event := event68453
    frameStart := 0 },
  { event := event68454
    frameStart := 0 },
  { event := event68455
    frameStart := 0 },
  { event := event68456
    frameStart := 0 },
  { event := event68457
    frameStart := 0 },
  { event := event68458
    frameStart := 0 },
  { event := event68459
    frameStart := 0 },
  { event := event68460
    frameStart := 0 },
  { event := event68461
    frameStart := 0 },
  { event := event68462
    frameStart := 0 },
  { event := event68463
    frameStart := 0 }
]

def eventLeaf4279 : Array AnnotatedEvent := #[
  { event := event68464
    frameStart := 0 },
  { event := event68465
    frameStart := 0 },
  { event := event68466
    frameStart := 0 },
  { event := event68467
    frameStart := 0 },
  { event := event68468
    frameStart := 0 },
  { event := event68469
    frameStart := 0 },
  { event := event68470
    frameStart := 0 },
  { event := event68471
    frameStart := 0 },
  { event := event68472
    frameStart := 0 },
  { event := event68473
    frameStart := 0 },
  { event := event68474
    frameStart := 0 },
  { event := event68475
    frameStart := 0 },
  { event := event68476
    frameStart := 0 },
  { event := event68477
    frameStart := 0 },
  { event := event68478
    frameStart := 0 },
  { event := event68479
    frameStart := 0 }
]

def eventLeaf4280 : Array AnnotatedEvent := #[
  { event := event68480
    frameStart := 0 },
  { event := event68481
    frameStart := 0 },
  { event := event68482
    frameStart := 0 },
  { event := event68483
    frameStart := 0 },
  { event := event68484
    frameStart := 0 },
  { event := event68485
    frameStart := 0 },
  { event := event68486
    frameStart := 0 },
  { event := event68487
    frameStart := 0 },
  { event := event68488
    frameStart := 0 },
  { event := event68489
    frameStart := 68489 },
  { event := event68490
    frameStart := 68489 },
  { event := event68491
    frameStart := 68489 },
  { event := event68492
    frameStart := 68489 },
  { event := event68493
    frameStart := 68489 },
  { event := event68494
    frameStart := 68489 },
  { event := event68495
    frameStart := 68489 }
]

def eventLeaf4281 : Array AnnotatedEvent := #[
  { event := event68496
    frameStart := 68489 },
  { event := event68497
    frameStart := 68489 },
  { event := event68498
    frameStart := 68489 },
  { event := event68499
    frameStart := 68489 },
  { event := event68500
    frameStart := 68489 },
  { event := event68501
    frameStart := 68489 },
  { event := event68502
    frameStart := 68489 },
  { event := event68503
    frameStart := 68489 },
  { event := event68504
    frameStart := 68489 },
  { event := event68505
    frameStart := 68489 },
  { event := event68506
    frameStart := 68489 },
  { event := event68507
    frameStart := 68489 },
  { event := event68508
    frameStart := 68489 },
  { event := event68509
    frameStart := 68489 },
  { event := event68510
    frameStart := 68489 },
  { event := event68511
    frameStart := 68489 }
]

def eventLeaf4282 : Array AnnotatedEvent := #[
  { event := event68512
    frameStart := 68489 },
  { event := event68513
    frameStart := 68489 },
  { event := event68514
    frameStart := 68489 },
  { event := event68515
    frameStart := 68489 },
  { event := event68516
    frameStart := 68489 },
  { event := event68517
    frameStart := 68489 },
  { event := event68518
    frameStart := 68489 },
  { event := event68519
    frameStart := 68489 },
  { event := event68520
    frameStart := 68489 },
  { event := event68521
    frameStart := 68489 },
  { event := event68522
    frameStart := 68489 },
  { event := event68523
    frameStart := 68489 },
  { event := event68524
    frameStart := 68489 },
  { event := event68525
    frameStart := 68489 },
  { event := event68526
    frameStart := 68489 },
  { event := event68527
    frameStart := 68489 }
]

def eventLeaf4283 : Array AnnotatedEvent := #[
  { event := event68528
    frameStart := 68489 },
  { event := event68529
    frameStart := 68489 },
  { event := event68530
    frameStart := 68489 },
  { event := event68531
    frameStart := 68489 },
  { event := event68532
    frameStart := 68489 },
  { event := event68533
    frameStart := 68489 },
  { event := event68534
    frameStart := 68489 },
  { event := event68535
    frameStart := 68489 },
  { event := event68536
    frameStart := 68489 },
  { event := event68537
    frameStart := 68489 },
  { event := event68538
    frameStart := 68489 },
  { event := event68539
    frameStart := 68489 },
  { event := event68540
    frameStart := 68489 },
  { event := event68541
    frameStart := 68489 },
  { event := event68542
    frameStart := 68489 },
  { event := event68543
    frameStart := 68543 }
]

def eventLeaf4284 : Array AnnotatedEvent := #[
  { event := event68544
    frameStart := 68543 },
  { event := event68545
    frameStart := 68543 },
  { event := event68546
    frameStart := 68543 },
  { event := event68547
    frameStart := 68543 },
  { event := event68548
    frameStart := 68543 },
  { event := event68549
    frameStart := 68543 },
  { event := event68550
    frameStart := 68543 },
  { event := event68551
    frameStart := 68543 },
  { event := event68552
    frameStart := 68543 },
  { event := event68553
    frameStart := 68543 },
  { event := event68554
    frameStart := 68543 },
  { event := event68555
    frameStart := 68543 },
  { event := event68556
    frameStart := 68543 },
  { event := event68557
    frameStart := 68543 },
  { event := event68558
    frameStart := 68543 },
  { event := event68559
    frameStart := 68543 }
]

def eventLeaf4285 : Array AnnotatedEvent := #[
  { event := event68560
    frameStart := 68543 },
  { event := event68561
    frameStart := 68543 },
  { event := event68562
    frameStart := 68543 },
  { event := event68563
    frameStart := 68543 },
  { event := event68564
    frameStart := 68543 },
  { event := event68565
    frameStart := 68543 },
  { event := event68566
    frameStart := 68543 },
  { event := event68567
    frameStart := 68543 },
  { event := event68568
    frameStart := 68543 },
  { event := event68569
    frameStart := 68543 },
  { event := event68570
    frameStart := 68543 },
  { event := event68571
    frameStart := 68543 },
  { event := event68572
    frameStart := 68543 },
  { event := event68573
    frameStart := 68543 },
  { event := event68574
    frameStart := 68543 },
  { event := event68575
    frameStart := 68543 }
]

def eventLeaf4286 : Array AnnotatedEvent := #[
  { event := event68576
    frameStart := 68543 },
  { event := event68577
    frameStart := 68543 },
  { event := event68578
    frameStart := 68543 },
  { event := event68579
    frameStart := 68543 },
  { event := event68580
    frameStart := 68543 },
  { event := event68581
    frameStart := 68543 },
  { event := event68582
    frameStart := 68543 },
  { event := event68583
    frameStart := 68543 },
  { event := event68584
    frameStart := 68543 },
  { event := event68585
    frameStart := 68543 },
  { event := event68586
    frameStart := 68543 },
  { event := event68587
    frameStart := 68543 },
  { event := event68588
    frameStart := 68543 },
  { event := event68589
    frameStart := 68543 },
  { event := event68590
    frameStart := 68543 },
  { event := event68591
    frameStart := 68543 }
]

def eventLeaf4287 : Array AnnotatedEvent := #[
  { event := event68592
    frameStart := 68543 },
  { event := event68593
    frameStart := 68543 },
  { event := event68594
    frameStart := 68543 },
  { event := event68595
    frameStart := 68543 },
  { event := event68596
    frameStart := 68543 },
  { event := event68597
    frameStart := 68543 },
  { event := event68598
    frameStart := 68543 },
  { event := event68599
    frameStart := 68543 },
  { event := event68600
    frameStart := 68543 },
  { event := event68601
    frameStart := 68543 },
  { event := event68602
    frameStart := 68543 },
  { event := event68603
    frameStart := 68543 },
  { event := event68604
    frameStart := 68543 },
  { event := event68605
    frameStart := 68543 },
  { event := event68606
    frameStart := 68543 },
  { event := event68607
    frameStart := 68543 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events267
