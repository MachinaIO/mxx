import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events267

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event68352 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31674⟩⟩) 0 ⟨10749⟩ 68348

def event68353 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31674⟩⟩) (.authority (.programFamilyFact))

def exact68354RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31674⟩⟩], []⟩, (1)⟩]

theorem exact68354RawTermsValid :
    exact68354RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68354 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31674⟩⟩) exact68354RawTerms (.finite 6) 68353 .exactZero (none)

def event68355 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31675⟩⟩) 0 ⟨31674⟩ 68354

def event68356 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31675⟩⟩) 1 ⟨24374⟩ 68351

def event68357 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31675⟩⟩) (.product (.predecessor 0 68355 .coefficient) (.predecessor 1 68356 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event68358 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31675⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], []⟩) [⟨.result 68354 .coefficient, true, some 1⟩, ⟨.result 68351 .coefficient, true, some 1⟩])

def event68359 : Event := .survivorFold (1) 68358

def exact68360RawTerms : List Term := []

theorem exact68360RawTermsValid :
    exact68360RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68360 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31675⟩⟩) exact68360RawTerms (.finite 36) 68357 (.finite 36) (some (68358))

def event68361 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31676⟩⟩) 0 ⟨31675⟩ 68360

def event68362 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31676⟩⟩) (.identity (.predecessor 0 68361 .coefficient))

def event68363 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31676⟩⟩) (.finite 36)

def event68364 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31884⟩⟩) 0 ⟨31676⟩ 68363

def event68365 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31884⟩⟩) (.authority (.programFamilyFact))

def exact68366RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], []⟩, (1)⟩]

theorem exact68366RawTermsValid :
    exact68366RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68366 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31884⟩⟩) exact68366RawTerms (.finite 6) 68365 .exactZero (none)

def event68367 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31885⟩⟩) 0 ⟨31884⟩ 68366

def event68368 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31885⟩⟩) (.identity (.predecessor 0 68367 .coefficient))

def event68369 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31885⟩⟩) (.finite 6)

def event68370 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32836⟩⟩) 0 ⟨31885⟩ 68369

def event68371 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32836⟩⟩) (.authority (.relationPreimageSource ⟨63⟩))

def exact68372RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩, (1)⟩]

theorem exact68372RawTermsValid :
    exact68372RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68372 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32836⟩⟩) exact68372RawTerms (.finite 5647228698) 68371 .exactZero (none)

def event68373 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact68374RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact68374RawTermsValid :
    exact68374RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68374 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact68374RawTerms .large 68373 .exactZero (none)

def event68375 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32837⟩⟩) 0 ⟨35⟩ 68374

def event68376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32837⟩⟩) 1 ⟨32836⟩ 68372

def event68377 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32837⟩⟩) (.product (.predecessor 0 68375 .coefficient) (.predecessor 1 68376 .coefficient) (⟨false, false, none, none, none⟩))

def event68378 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32837⟩⟩, .operator (⟨68374, 0⟩, ⟨68372, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩, (1)⟩)

def exact68379RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩, (1)⟩]

theorem exact68379RawTermsValid :
    exact68379RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68379 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32837⟩⟩) exact68379RawTerms .large 68377 .exactZero (none)

def event68380 : Event := .preFoldPolynomial 68379 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩, (1)⟩] .exactZero none

def exact68381RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩, (1)⟩]

def event68381 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨32837⟩⟩) 68380 exact68381RawTerms .large 68377 .exactZero (none)

def event68382 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨34114⟩⟩)

def event68383 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event68384 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event68385 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.authority (.operator))

def event68386 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.finite 16)

def event68387 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event68388 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event68389 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event68390 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event68391 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 68390

def event68392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 68388

def event68393 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 68391 .coefficient) (.value (.predecessor 1 68392 .coefficient)))

def event68394 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event68395 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 0 ⟨392⟩ 68394

def event68396 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 1 ⟨10691⟩ 68386

def event68397 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.sum [.predecessor 0 68395 .coefficient, .predecessor 1 68396 .coefficient])

def event68398 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.finite 655356)

def event68399 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 0 ⟨10693⟩ 68398

def event68400 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 1 ⟨5426⟩ 68384

def event68401 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.identity (.predecessor 1 68400 .coefficient))

def event68402 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.finite 655360)

def event68403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24374⟩⟩) 0 ⟨10749⟩ 68402

def event68404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24374⟩⟩) (.authority (.programFamilyFact))

def exact68405RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩], []⟩, (1)⟩]

theorem exact68405RawTermsValid :
    exact68405RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68405 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24374⟩⟩) exact68405RawTerms (.finite 6) 68404 .exactZero (none)

def event68406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31674⟩⟩) 0 ⟨10749⟩ 68402

def event68407 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31674⟩⟩) (.authority (.programFamilyFact))

def exact68408RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31674⟩⟩], []⟩, (1)⟩]

theorem exact68408RawTermsValid :
    exact68408RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68408 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31674⟩⟩) exact68408RawTerms (.finite 6) 68407 .exactZero (none)

def event68409 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31675⟩⟩) 0 ⟨31674⟩ 68408

def event68410 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31675⟩⟩) 1 ⟨24374⟩ 68405

def event68411 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31675⟩⟩) (.product (.predecessor 0 68409 .coefficient) (.predecessor 1 68410 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event68412 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31675⟩⟩, .operator (⟨68408, 0⟩, ⟨68405, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], []⟩, (1)⟩)

def exact68413RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], []⟩, (1)⟩]

theorem exact68413RawTermsValid :
    exact68413RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68413 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31675⟩⟩) exact68413RawTerms (.finite 36) 68411 .exactZero (none)

def event68414 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31676⟩⟩) 0 ⟨31675⟩ 68413

def event68415 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31676⟩⟩) (.identity (.predecessor 0 68414 .coefficient))

def event68416 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31676⟩⟩) (.finite 36)

def event68417 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31884⟩⟩) 0 ⟨31676⟩ 68416

def event68418 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31884⟩⟩) (.authority (.programFamilyFact))

def exact68419RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], []⟩, (1)⟩]

theorem exact68419RawTermsValid :
    exact68419RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68419 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31884⟩⟩) exact68419RawTerms (.finite 6) 68418 .exactZero (none)

def event68420 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31885⟩⟩) 0 ⟨31884⟩ 68419

def event68421 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31885⟩⟩) (.identity (.predecessor 0 68420 .coefficient))

def event68422 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31885⟩⟩) (.finite 6)

def event68423 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33162⟩⟩) 0 ⟨31885⟩ 68422

def event68424 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33162⟩⟩) (.authority (.programFamilyFact))

def event68425 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨33162⟩⟩) (.finite 3720)

def event68426 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event68427 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33164⟩⟩) 0 ⟨7177⟩ 68426

def event68428 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33164⟩⟩) 1 ⟨33162⟩ 68425

def event68429 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33164⟩⟩) (.authority (.operator))

def exact68430RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨33164⟩⟩]⟩, (1)⟩]

theorem exact68430RawTermsValid :
    exact68430RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68430 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33164⟩⟩) exact68430RawTerms .large 68429 .exactZero (none)

def event68431 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34109⟩⟩) 0 ⟨33164⟩ 68430

def event68432 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34109⟩⟩) (.authority (.operator))

def exact68433RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩, (1)⟩]

theorem exact68433RawTermsValid :
    exact68433RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68433 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34109⟩⟩) exact68433RawTerms (.finite 8192) 68432 .exactZero (none)

def event68434 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event68435 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event68436 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33334⟩⟩) 0 ⟨31885⟩ 68422

def event68437 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33334⟩⟩) 1 ⟨136⟩ 68435

def event68438 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33334⟩⟩) (.sum [.predecessor 0 68436 .coefficient, .predecessor 1 68437 .coefficient])

def event68439 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨33334⟩⟩) (.finite 6)

def event68440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33335⟩⟩) 0 ⟨33334⟩ 68439

def event68441 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33335⟩⟩) (.identity (.predecessor 0 68440 .coefficient))

def exact68442RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], []⟩, (1)⟩]

theorem exact68442RawTermsValid :
    exact68442RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68442 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33335⟩⟩) exact68442RawTerms (.finite 6) 68441 .exactZero (none)

def event68443 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact68444RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact68444RawTermsValid :
    exact68444RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68444 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact68444RawTerms .large 68443 .exactZero (none)

def event68445 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33336⟩⟩) 0 ⟨6908⟩ 68444

def event68446 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33336⟩⟩) 1 ⟨33335⟩ 68442

def event68447 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33336⟩⟩) (.product (.predecessor 0 68445 .coefficient) (.predecessor 1 68446 .coefficient) (⟨false, false, none, none, none⟩))

def event68448 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33336⟩⟩, .operator (⟨68444, 0⟩, ⟨68442, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact68449RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact68449RawTermsValid :
    exact68449RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68449 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33336⟩⟩) exact68449RawTerms .large 68447 .exactZero (none)

def event68450 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7182⟩⟩) 0 ⟨7177⟩ 68426

def event68451 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7182⟩⟩) (.authority (.operator))

def exact68452RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩]

theorem exact68452RawTermsValid :
    exact68452RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68452 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7182⟩⟩) exact68452RawTerms .large 68451 .exactZero (none)

def event68453 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33337⟩⟩) 0 ⟨7182⟩ 68452

def event68454 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33337⟩⟩) 1 ⟨33336⟩ 68449

def event68455 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33337⟩⟩) (.sum [.predecessor 0 68453 .coefficient, .predecessor 1 68454 .coefficient])

def exact68456RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68456RawTermsValid :
    exact68456RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68456 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33337⟩⟩) exact68456RawTerms .large 68455 .exactZero (none)

def event68457 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34110⟩⟩) 0 ⟨33337⟩ 68456

def event68458 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34110⟩⟩) 1 ⟨34109⟩ 68433

def event68459 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34110⟩⟩) (.product (.predecessor 0 68457 .coefficient) (.predecessor 1 68458 .coefficient) (⟨false, false, none, none, none⟩))

def event68460 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34110⟩⟩, .operator (⟨68456, 0⟩, ⟨68433, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩, (1)⟩)

def event68461 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34110⟩⟩, .operator (⟨68456, 1⟩, ⟨68433, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩, (-1)⟩)

def event68462 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨34110⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨34109⟩⟩) ⟨33164⟩ 68430)

def event68463 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34110⟩⟩, .relation 68462 0, ⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨33164⟩⟩]⟩, (-1)⟩)

def exact68464RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨33164⟩⟩]⟩, (-1)⟩]

theorem exact68464RawTermsValid :
    exact68464RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68464 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34110⟩⟩) exact68464RawTerms .large 68459 .exactZero (none)

def event68465 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32239⟩⟩) 0 ⟨31885⟩ 68422

def event68466 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32239⟩⟩) (.authority (.programFamilyFact))

def exact68467RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], []⟩, (1)⟩]

theorem exact68467RawTermsValid :
    exact68467RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68467 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32239⟩⟩) exact68467RawTerms (.finite 55) 68466 .exactZero (none)

def event68468 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32241⟩⟩) 0 ⟨6908⟩ 68444

def event68469 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32241⟩⟩) 1 ⟨32239⟩ 68467

def event68470 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32241⟩⟩) (.product (.predecessor 0 68468 .coefficient) (.predecessor 1 68469 .coefficient) (⟨false, true, none, none, some 1⟩))

def event68471 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32241⟩⟩, .operator (⟨68444, 0⟩, ⟨68467, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact68472RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact68472RawTermsValid :
    exact68472RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68472 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32241⟩⟩) exact68472RawTerms .large 68470 .exactZero (none)

def event68473 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7204⟩⟩) 0 ⟨7177⟩ 68426

def event68474 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7204⟩⟩) (.authority (.operator))

def exact68475RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩]

theorem exact68475RawTermsValid :
    exact68475RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68475 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7204⟩⟩) exact68475RawTerms .large 68474 .exactZero (none)

def event68476 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32242⟩⟩) 0 ⟨7204⟩ 68475

def event68477 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32242⟩⟩) 1 ⟨32241⟩ 68472

def event68478 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32242⟩⟩) (.sum [.predecessor 0 68476 .coefficient, .predecessor 1 68477 .coefficient])

def exact68479RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68479RawTermsValid :
    exact68479RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68479 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32242⟩⟩) exact68479RawTerms .large 68478 .exactZero (none)

def event68480 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34114⟩⟩) 0 ⟨32242⟩ 68479

def event68481 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34114⟩⟩) 1 ⟨34110⟩ 68464

def event68482 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34114⟩⟩) (.sum [.predecessor 0 68480 .coefficient, .predecessor 1 68481 .coefficient])

def exact68483RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨33164⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68483RawTermsValid :
    exact68483RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68483 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34114⟩⟩) exact68483RawTerms .large 68482 .exactZero (none)

def event68484 : Event := .preFoldPolynomial 68483 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨33164⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact68485RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨33164⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event68485 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨34114⟩⟩) 68484 exact68485RawTerms .large 68482 .exactZero (none)

def event68486 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨31885⟩⟩) ⟨⟨83⟩, ⟨63⟩, ⟨135⟩⟩ ⟨68328, 68486⟩

def event68487 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨32839⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩) (1) 0 2 (.universal 68486 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32836⟩⟩]⟩) (none) 68485)

def event68488 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32839⟩⟩, .relation 68487 1, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩)

def event68489 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32839⟩⟩, .relation 68487 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩, (-1)⟩)

def event68490 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32839⟩⟩, .relation 68487 2, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨33164⟩⟩]⟩, (1)⟩)

def event68491 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32839⟩⟩, .relation 68487 3, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact68492RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨33164⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68492RawTermsValid :
    exact68492RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68492 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32839⟩⟩) exact68492RawTerms .large 68324 (.finite 202072841853861888) (some (68326))

def event68493 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34112⟩⟩) 0 ⟨32839⟩ 68492

def event68494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34112⟩⟩) 1 ⟨34111⟩ 68314

def event68495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34112⟩⟩) (.sum [.predecessor 0 68493 .coefficient, .predecessor 1 68494 .coefficient])

def event68496 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34112⟩⟩, .operator (⟨68492, 0⟩, ⟨68314, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩, (1)⟩)

def event68497 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34112⟩⟩, .operator (⟨68492, 2⟩, ⟨68314, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31884⟩⟩], [⟨.program ⟨257⟩, ⟨33164⟩⟩]⟩, (-1)⟩)

def event68498 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34112⟩⟩) (.sum [.result 68492 .summary, .result 68314 .summary])

def exact68499RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨32239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68499RawTermsValid :
    exact68499RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68499 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34112⟩⟩) exact68499RawTerms .large 68495 (.finite 32189200113375081643992404983808) (some (68498))

def event68500 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23142⟩⟩) 0 ⟨21865⟩ 2700

def event68501 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23142⟩⟩) (.authority (.programFamilyFact))

def event68502 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨23142⟩⟩) (.finite 3720)

def event68503 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23144⟩⟩) 0 ⟨7177⟩ 15500

def event68504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23144⟩⟩) 1 ⟨23142⟩ 68502

def event68505 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23144⟩⟩) (.authority (.operator))

def exact68506RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨23144⟩⟩]⟩, (1)⟩]

theorem exact68506RawTermsValid :
    exact68506RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68506 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23144⟩⟩) exact68506RawTerms .large 68505 .exactZero (none)

def event68507 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24089⟩⟩) 0 ⟨23144⟩ 68506

def event68508 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24089⟩⟩) (.authority (.operator))

def exact68509RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨24089⟩⟩]⟩, (1)⟩]

theorem exact68509RawTermsValid :
    exact68509RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68509 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24089⟩⟩) exact68509RawTerms (.finite 8192) 68508 .exactZero (none)

def event68510 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22970⟩⟩) 0 ⟨21664⟩ 2694

def event68511 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22970⟩⟩) (.authority (.programFamilyFact))

def event68512 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨22970⟩⟩) (.finite 3720)

def event68513 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22971⟩⟩) 0 ⟨7177⟩ 15500

def event68514 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22971⟩⟩) 1 ⟨22970⟩ 68512

def event68515 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22971⟩⟩) (.authority (.operator))

def exact68516RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22971⟩⟩]⟩, (1)⟩]

theorem exact68516RawTermsValid :
    exact68516RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68516 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22971⟩⟩) exact68516RawTerms .large 68515 .exactZero (none)

def event68517 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23516⟩⟩) 0 ⟨22971⟩ 68516

def event68518 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23516⟩⟩) (.authority (.operator))

def exact68519RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨23516⟩⟩]⟩, (1)⟩]

theorem exact68519RawTermsValid :
    exact68519RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68519 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23516⟩⟩) exact68519RawTerms (.finite 8192) 68518 .exactZero (none)

def event68520 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21665⟩⟩) 0 ⟨21662⟩ 2683

def event68521 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21665⟩⟩) 1 ⟨10752⟩ 61278

def event68522 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21665⟩⟩) (.tensor (.predecessor 0 68520 .coefficient) (.predecessor 1 68521 .coefficient) true false)

def event68523 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21665⟩⟩, .operator (⟨2683, 0⟩, ⟨61278, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact68524RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact68524RawTermsValid :
    exact68524RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68524 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21665⟩⟩) exact68524RawTerms .large 68522 .exactZero (none)

def event68525 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10788⟩⟩) 0 ⟨10751⟩ 61148

def event68526 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10788⟩⟩) 1 ⟨7306⟩ 24595

def event68527 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10788⟩⟩) (.product (.predecessor 0 68525 .coefficient) (.predecessor 1 68526 .coefficient) (⟨false, false, none, none, none⟩))

def event68528 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10788⟩⟩, .operator (⟨61148, 0⟩, ⟨24595, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (1)⟩)

def exact68529RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (1)⟩]

theorem exact68529RawTermsValid :
    exact68529RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68529 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10788⟩⟩) exact68529RawTerms .large 68527 .exactZero (none)

def event68530 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21666⟩⟩) 0 ⟨10788⟩ 68529

def event68531 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21666⟩⟩) 1 ⟨21665⟩ 68524

def event68532 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21666⟩⟩) (.sum [.predecessor 0 68530 .coefficient, .predecessor 1 68531 .coefficient])

def exact68533RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68533RawTermsValid :
    exact68533RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68533 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21666⟩⟩) exact68533RawTerms .large 68532 .exactZero (none)

def event68534 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21667⟩⟩) 0 ⟨21666⟩ 68533

def event68535 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21667⟩⟩) 1 ⟨132⟩ 24587

def event68536 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21667⟩⟩) (.sum [.predecessor 0 68534 .coefficient, .predecessor 1 68535 .coefficient])

def event68537 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21667⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨132⟩⟩]⟩) [⟨.result 24587 .coefficient, false, none⟩])

def event68538 : Event := .survivorFold (1) 68537

def exact68539RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68539RawTermsValid :
    exact68539RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68539 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21667⟩⟩) exact68539RawTerms .large 68536 (.finite 26) (some (68537))

def event68540 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21668⟩⟩) 0 ⟨21667⟩ 68539

def event68541 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21668⟩⟩) 1 ⟨21206⟩ 2686

def event68542 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21668⟩⟩) (.product (.predecessor 0 68540 .coefficient) (.predecessor 1 68541 .coefficient) (⟨false, true, none, none, some 1⟩))

def event68543 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21668⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩], []⟩) [⟨.result 2686 .coefficient, true, some 1⟩])

def event68544 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21668⟩⟩) (.product (.result 68539 .summary) (.transfer 68543) (⟨false, false, none, none, none⟩))

def event68545 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21668⟩⟩, .operator (⟨68539, 1⟩, ⟨2686, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event68546 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21668⟩⟩, .operator (⟨68539, 0⟩, ⟨2686, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (1)⟩)

def exact68547RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68547RawTermsValid :
    exact68547RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68547 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21668⟩⟩) exact68547RawTerms .large 68542 (.finite 3407872) (some (68544))

def event68548 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21207⟩⟩) 0 ⟨21206⟩ 2686

def event68549 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21207⟩⟩) 1 ⟨10752⟩ 61278

def event68550 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21207⟩⟩) (.tensor (.predecessor 0 68548 .coefficient) (.predecessor 1 68549 .coefficient) true false)

def event68551 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21207⟩⟩, .operator (⟨2686, 0⟩, ⟨61278, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact68552RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact68552RawTermsValid :
    exact68552RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68552 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21207⟩⟩) exact68552RawTerms .large 68550 .exactZero (none)

def event68553 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10768⟩⟩) 0 ⟨10751⟩ 61148

def event68554 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10768⟩⟩) 1 ⟨7286⟩ 24636

def event68555 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10768⟩⟩) (.product (.predecessor 0 68553 .coefficient) (.predecessor 1 68554 .coefficient) (⟨false, false, none, none, none⟩))

def event68556 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10768⟩⟩, .operator (⟨61148, 0⟩, ⟨24636, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩]⟩, (1)⟩)

def exact68557RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩]⟩, (1)⟩]

theorem exact68557RawTermsValid :
    exact68557RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68557 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10768⟩⟩) exact68557RawTerms .large 68555 .exactZero (none)

def event68558 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21208⟩⟩) 0 ⟨10768⟩ 68557

def event68559 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21208⟩⟩) 1 ⟨21207⟩ 68552

def event68560 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21208⟩⟩) (.sum [.predecessor 0 68558 .coefficient, .predecessor 1 68559 .coefficient])

def exact68561RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68561RawTermsValid :
    exact68561RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68561 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21208⟩⟩) exact68561RawTerms .large 68560 .exactZero (none)

def event68562 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21209⟩⟩) 0 ⟨21208⟩ 68561

def event68563 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21209⟩⟩) 1 ⟨112⟩ 24628

def event68564 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21209⟩⟩) (.sum [.predecessor 0 68562 .coefficient, .predecessor 1 68563 .coefficient])

def event68565 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21209⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨112⟩⟩]⟩) [⟨.result 24628 .coefficient, false, none⟩])

def event68566 : Event := .survivorFold (1) 68565

def exact68567RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68567RawTermsValid :
    exact68567RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68567 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21209⟩⟩) exact68567RawTerms .large 68564 (.finite 26) (some (68565))

def event68568 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21210⟩⟩) 0 ⟨21209⟩ 68567

def event68569 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21210⟩⟩) 1 ⟨9575⟩ 24625

def event68570 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21210⟩⟩) (.product (.predecessor 0 68568 .coefficient) (.predecessor 1 68569 .coefficient) (⟨false, false, none, none, none⟩))

def event68571 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21210⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) [⟨.result 24621 .coefficient, false, none⟩])

def event68572 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21210⟩⟩) (.product (.result 68567 .summary) (.transfer 68571) (⟨false, false, none, none, none⟩))

def event68573 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21210⟩⟩, .operator (⟨68567, 1⟩, ⟨24625, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩, (-1)⟩)

def event68574 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨21210⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9574⟩⟩) ⟨7306⟩ 24595)

def event68575 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21210⟩⟩, .relation 68574 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (-1)⟩)

def event68576 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21210⟩⟩, .operator (⟨68567, 0⟩, ⟨24625, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩, (1)⟩)

def exact68577RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (-1)⟩]

theorem exact68577RawTermsValid :
    exact68577RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68577 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21210⟩⟩) exact68577RawTerms .large 68570 (.finite 279172874240) (some (68572))

def event68578 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21669⟩⟩) 0 ⟨21210⟩ 68577

def event68579 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21669⟩⟩) 1 ⟨21668⟩ 68547

def event68580 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21669⟩⟩) (.sum [.predecessor 0 68578 .coefficient, .predecessor 1 68579 .coefficient])

def event68581 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21669⟩⟩, .operator (⟨68577, 1⟩, ⟨68547, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (1)⟩)

def event68582 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21669⟩⟩) (.sum [.result 68577 .summary, .result 68547 .summary])

def exact68583RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68583RawTermsValid :
    exact68583RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68583 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21669⟩⟩) exact68583RawTerms .large 68580 (.finite 279176282112) (some (68582))

def event68584 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23517⟩⟩) 0 ⟨21669⟩ 68583

def event68585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23517⟩⟩) 1 ⟨23516⟩ 68519

def event68586 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23517⟩⟩) (.product (.predecessor 0 68584 .coefficient) (.predecessor 1 68585 .coefficient) (⟨false, false, none, none, none⟩))

def event68587 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23517⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨23516⟩⟩]⟩) [⟨.result 68519 .coefficient, false, none⟩])

def event68588 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23517⟩⟩) (.product (.result 68583 .summary) (.transfer 68587) (⟨false, false, none, none, none⟩))

def event68589 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23517⟩⟩, .operator (⟨68583, 1⟩, ⟨68519, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23516⟩⟩]⟩, (-1)⟩)

def event68590 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨23517⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23516⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23516⟩⟩) ⟨22971⟩ 68516)

def event68591 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23517⟩⟩, .relation 68590 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨22971⟩⟩]⟩, (-1)⟩)

def event68592 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨23517⟩⟩, .operator (⟨68583, 0⟩, ⟨68519, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23516⟩⟩]⟩, (1)⟩)

def exact68593RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23516⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], [⟨.program ⟨257⟩, ⟨22971⟩⟩]⟩, (-1)⟩]

theorem exact68593RawTermsValid :
    exact68593RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68593 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23517⟩⟩) exact68593RawTerms .large 68586 (.finite 2997632503724774522880) (some (68588))

def event68594 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22439⟩⟩) 0 ⟨21664⟩ 2694

def event68595 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22439⟩⟩) (.authority (.relationPreimageSource ⟨38⟩))

def exact68596RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22439⟩⟩]⟩, (1)⟩]

theorem exact68596RawTermsValid :
    exact68596RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68596 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22439⟩⟩) exact68596RawTerms (.finite 5647228698) 68595 .exactZero (none)

def event68597 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22441⟩⟩) 0 ⟨22439⟩ 68596

def event68598 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22441⟩⟩) 1 ⟨2370⟩ 4

def event68599 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22441⟩⟩) (.scale (.predecessor 0 68597 .coefficient) (.value (.predecessor 1 68598 .coefficient)))

def exact68600RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22439⟩⟩]⟩, (1)⟩]

theorem exact68600RawTermsValid :
    exact68600RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68600 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22441⟩⟩) exact68600RawTerms (.finite 5647228698) 68599 .exactZero (none)

def event68601 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22442⟩⟩) 0 ⟨10792⟩ 61370

def event68602 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22442⟩⟩) 1 ⟨22441⟩ 68600

def event68603 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22442⟩⟩) (.product (.predecessor 0 68601 .coefficient) (.predecessor 1 68602 .coefficient) (⟨false, false, none, none, none⟩))

def event68604 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22442⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨22439⟩⟩]⟩) [⟨.result 68596 .coefficient, false, none⟩])

def event68605 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22442⟩⟩) (.product (.result 61370 .summary) (.transfer 68604) (⟨false, false, none, none, none⟩))

def event68606 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22442⟩⟩, .operator (⟨61370, 0⟩, ⟨68600, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22439⟩⟩]⟩, (1)⟩)

def event68607 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨22440⟩⟩)

def eventLeaf4272 : Array AnnotatedEvent := #[
  { event := event68352
    frameStart := 68328 },
  { event := event68353
    frameStart := 68328 },
  { event := event68354
    frameStart := 68328 },
  { event := event68355
    frameStart := 68328 },
  { event := event68356
    frameStart := 68328 },
  { event := event68357
    frameStart := 68328 },
  { event := event68358
    frameStart := 68328 },
  { event := event68359
    frameStart := 68328 },
  { event := event68360
    frameStart := 68328 },
  { event := event68361
    frameStart := 68328 },
  { event := event68362
    frameStart := 68328 },
  { event := event68363
    frameStart := 68328 },
  { event := event68364
    frameStart := 68328 },
  { event := event68365
    frameStart := 68328 },
  { event := event68366
    frameStart := 68328 },
  { event := event68367
    frameStart := 68328 }
]

def eventLeaf4273 : Array AnnotatedEvent := #[
  { event := event68368
    frameStart := 68328 },
  { event := event68369
    frameStart := 68328 },
  { event := event68370
    frameStart := 68328 },
  { event := event68371
    frameStart := 68328 },
  { event := event68372
    frameStart := 68328 },
  { event := event68373
    frameStart := 68328 },
  { event := event68374
    frameStart := 68328 },
  { event := event68375
    frameStart := 68328 },
  { event := event68376
    frameStart := 68328 },
  { event := event68377
    frameStart := 68328 },
  { event := event68378
    frameStart := 68328 },
  { event := event68379
    frameStart := 68328 },
  { event := event68380
    frameStart := 68328 },
  { event := event68381
    frameStart := 68328 },
  { event := event68382
    frameStart := 68382 },
  { event := event68383
    frameStart := 68382 }
]

def eventLeaf4274 : Array AnnotatedEvent := #[
  { event := event68384
    frameStart := 68382 },
  { event := event68385
    frameStart := 68382 },
  { event := event68386
    frameStart := 68382 },
  { event := event68387
    frameStart := 68382 },
  { event := event68388
    frameStart := 68382 },
  { event := event68389
    frameStart := 68382 },
  { event := event68390
    frameStart := 68382 },
  { event := event68391
    frameStart := 68382 },
  { event := event68392
    frameStart := 68382 },
  { event := event68393
    frameStart := 68382 },
  { event := event68394
    frameStart := 68382 },
  { event := event68395
    frameStart := 68382 },
  { event := event68396
    frameStart := 68382 },
  { event := event68397
    frameStart := 68382 },
  { event := event68398
    frameStart := 68382 },
  { event := event68399
    frameStart := 68382 }
]

def eventLeaf4275 : Array AnnotatedEvent := #[
  { event := event68400
    frameStart := 68382 },
  { event := event68401
    frameStart := 68382 },
  { event := event68402
    frameStart := 68382 },
  { event := event68403
    frameStart := 68382 },
  { event := event68404
    frameStart := 68382 },
  { event := event68405
    frameStart := 68382 },
  { event := event68406
    frameStart := 68382 },
  { event := event68407
    frameStart := 68382 },
  { event := event68408
    frameStart := 68382 },
  { event := event68409
    frameStart := 68382 },
  { event := event68410
    frameStart := 68382 },
  { event := event68411
    frameStart := 68382 },
  { event := event68412
    frameStart := 68382 },
  { event := event68413
    frameStart := 68382 },
  { event := event68414
    frameStart := 68382 },
  { event := event68415
    frameStart := 68382 }
]

def eventLeaf4276 : Array AnnotatedEvent := #[
  { event := event68416
    frameStart := 68382 },
  { event := event68417
    frameStart := 68382 },
  { event := event68418
    frameStart := 68382 },
  { event := event68419
    frameStart := 68382 },
  { event := event68420
    frameStart := 68382 },
  { event := event68421
    frameStart := 68382 },
  { event := event68422
    frameStart := 68382 },
  { event := event68423
    frameStart := 68382 },
  { event := event68424
    frameStart := 68382 },
  { event := event68425
    frameStart := 68382 },
  { event := event68426
    frameStart := 68382 },
  { event := event68427
    frameStart := 68382 },
  { event := event68428
    frameStart := 68382 },
  { event := event68429
    frameStart := 68382 },
  { event := event68430
    frameStart := 68382 },
  { event := event68431
    frameStart := 68382 }
]

def eventLeaf4277 : Array AnnotatedEvent := #[
  { event := event68432
    frameStart := 68382 },
  { event := event68433
    frameStart := 68382 },
  { event := event68434
    frameStart := 68382 },
  { event := event68435
    frameStart := 68382 },
  { event := event68436
    frameStart := 68382 },
  { event := event68437
    frameStart := 68382 },
  { event := event68438
    frameStart := 68382 },
  { event := event68439
    frameStart := 68382 },
  { event := event68440
    frameStart := 68382 },
  { event := event68441
    frameStart := 68382 },
  { event := event68442
    frameStart := 68382 },
  { event := event68443
    frameStart := 68382 },
  { event := event68444
    frameStart := 68382 },
  { event := event68445
    frameStart := 68382 },
  { event := event68446
    frameStart := 68382 },
  { event := event68447
    frameStart := 68382 }
]

def eventLeaf4278 : Array AnnotatedEvent := #[
  { event := event68448
    frameStart := 68382 },
  { event := event68449
    frameStart := 68382 },
  { event := event68450
    frameStart := 68382 },
  { event := event68451
    frameStart := 68382 },
  { event := event68452
    frameStart := 68382 },
  { event := event68453
    frameStart := 68382 },
  { event := event68454
    frameStart := 68382 },
  { event := event68455
    frameStart := 68382 },
  { event := event68456
    frameStart := 68382 },
  { event := event68457
    frameStart := 68382 },
  { event := event68458
    frameStart := 68382 },
  { event := event68459
    frameStart := 68382 },
  { event := event68460
    frameStart := 68382 },
  { event := event68461
    frameStart := 68382 },
  { event := event68462
    frameStart := 68382 },
  { event := event68463
    frameStart := 68382 }
]

def eventLeaf4279 : Array AnnotatedEvent := #[
  { event := event68464
    frameStart := 68382 },
  { event := event68465
    frameStart := 68382 },
  { event := event68466
    frameStart := 68382 },
  { event := event68467
    frameStart := 68382 },
  { event := event68468
    frameStart := 68382 },
  { event := event68469
    frameStart := 68382 },
  { event := event68470
    frameStart := 68382 },
  { event := event68471
    frameStart := 68382 },
  { event := event68472
    frameStart := 68382 },
  { event := event68473
    frameStart := 68382 },
  { event := event68474
    frameStart := 68382 },
  { event := event68475
    frameStart := 68382 },
  { event := event68476
    frameStart := 68382 },
  { event := event68477
    frameStart := 68382 },
  { event := event68478
    frameStart := 68382 },
  { event := event68479
    frameStart := 68382 }
]

def eventLeaf4280 : Array AnnotatedEvent := #[
  { event := event68480
    frameStart := 68382 },
  { event := event68481
    frameStart := 68382 },
  { event := event68482
    frameStart := 68382 },
  { event := event68483
    frameStart := 68382 },
  { event := event68484
    frameStart := 68382 },
  { event := event68485
    frameStart := 68382 },
  { event := event68486
    frameStart := 0 },
  { event := event68487
    frameStart := 0 },
  { event := event68488
    frameStart := 0 },
  { event := event68489
    frameStart := 0 },
  { event := event68490
    frameStart := 0 },
  { event := event68491
    frameStart := 0 },
  { event := event68492
    frameStart := 0 },
  { event := event68493
    frameStart := 0 },
  { event := event68494
    frameStart := 0 },
  { event := event68495
    frameStart := 0 }
]

def eventLeaf4281 : Array AnnotatedEvent := #[
  { event := event68496
    frameStart := 0 },
  { event := event68497
    frameStart := 0 },
  { event := event68498
    frameStart := 0 },
  { event := event68499
    frameStart := 0 },
  { event := event68500
    frameStart := 0 },
  { event := event68501
    frameStart := 0 },
  { event := event68502
    frameStart := 0 },
  { event := event68503
    frameStart := 0 },
  { event := event68504
    frameStart := 0 },
  { event := event68505
    frameStart := 0 },
  { event := event68506
    frameStart := 0 },
  { event := event68507
    frameStart := 0 },
  { event := event68508
    frameStart := 0 },
  { event := event68509
    frameStart := 0 },
  { event := event68510
    frameStart := 0 },
  { event := event68511
    frameStart := 0 }
]

def eventLeaf4282 : Array AnnotatedEvent := #[
  { event := event68512
    frameStart := 0 },
  { event := event68513
    frameStart := 0 },
  { event := event68514
    frameStart := 0 },
  { event := event68515
    frameStart := 0 },
  { event := event68516
    frameStart := 0 },
  { event := event68517
    frameStart := 0 },
  { event := event68518
    frameStart := 0 },
  { event := event68519
    frameStart := 0 },
  { event := event68520
    frameStart := 0 },
  { event := event68521
    frameStart := 0 },
  { event := event68522
    frameStart := 0 },
  { event := event68523
    frameStart := 0 },
  { event := event68524
    frameStart := 0 },
  { event := event68525
    frameStart := 0 },
  { event := event68526
    frameStart := 0 },
  { event := event68527
    frameStart := 0 }
]

def eventLeaf4283 : Array AnnotatedEvent := #[
  { event := event68528
    frameStart := 0 },
  { event := event68529
    frameStart := 0 },
  { event := event68530
    frameStart := 0 },
  { event := event68531
    frameStart := 0 },
  { event := event68532
    frameStart := 0 },
  { event := event68533
    frameStart := 0 },
  { event := event68534
    frameStart := 0 },
  { event := event68535
    frameStart := 0 },
  { event := event68536
    frameStart := 0 },
  { event := event68537
    frameStart := 0 },
  { event := event68538
    frameStart := 0 },
  { event := event68539
    frameStart := 0 },
  { event := event68540
    frameStart := 0 },
  { event := event68541
    frameStart := 0 },
  { event := event68542
    frameStart := 0 },
  { event := event68543
    frameStart := 0 }
]

def eventLeaf4284 : Array AnnotatedEvent := #[
  { event := event68544
    frameStart := 0 },
  { event := event68545
    frameStart := 0 },
  { event := event68546
    frameStart := 0 },
  { event := event68547
    frameStart := 0 },
  { event := event68548
    frameStart := 0 },
  { event := event68549
    frameStart := 0 },
  { event := event68550
    frameStart := 0 },
  { event := event68551
    frameStart := 0 },
  { event := event68552
    frameStart := 0 },
  { event := event68553
    frameStart := 0 },
  { event := event68554
    frameStart := 0 },
  { event := event68555
    frameStart := 0 },
  { event := event68556
    frameStart := 0 },
  { event := event68557
    frameStart := 0 },
  { event := event68558
    frameStart := 0 },
  { event := event68559
    frameStart := 0 }
]

def eventLeaf4285 : Array AnnotatedEvent := #[
  { event := event68560
    frameStart := 0 },
  { event := event68561
    frameStart := 0 },
  { event := event68562
    frameStart := 0 },
  { event := event68563
    frameStart := 0 },
  { event := event68564
    frameStart := 0 },
  { event := event68565
    frameStart := 0 },
  { event := event68566
    frameStart := 0 },
  { event := event68567
    frameStart := 0 },
  { event := event68568
    frameStart := 0 },
  { event := event68569
    frameStart := 0 },
  { event := event68570
    frameStart := 0 },
  { event := event68571
    frameStart := 0 },
  { event := event68572
    frameStart := 0 },
  { event := event68573
    frameStart := 0 },
  { event := event68574
    frameStart := 0 },
  { event := event68575
    frameStart := 0 }
]

def eventLeaf4286 : Array AnnotatedEvent := #[
  { event := event68576
    frameStart := 0 },
  { event := event68577
    frameStart := 0 },
  { event := event68578
    frameStart := 0 },
  { event := event68579
    frameStart := 0 },
  { event := event68580
    frameStart := 0 },
  { event := event68581
    frameStart := 0 },
  { event := event68582
    frameStart := 0 },
  { event := event68583
    frameStart := 0 },
  { event := event68584
    frameStart := 0 },
  { event := event68585
    frameStart := 0 },
  { event := event68586
    frameStart := 0 },
  { event := event68587
    frameStart := 0 },
  { event := event68588
    frameStart := 0 },
  { event := event68589
    frameStart := 0 },
  { event := event68590
    frameStart := 0 },
  { event := event68591
    frameStart := 0 }
]

def eventLeaf4287 : Array AnnotatedEvent := #[
  { event := event68592
    frameStart := 0 },
  { event := event68593
    frameStart := 0 },
  { event := event68594
    frameStart := 0 },
  { event := event68595
    frameStart := 0 },
  { event := event68596
    frameStart := 0 },
  { event := event68597
    frameStart := 0 },
  { event := event68598
    frameStart := 0 },
  { event := event68599
    frameStart := 0 },
  { event := event68600
    frameStart := 0 },
  { event := event68601
    frameStart := 0 },
  { event := event68602
    frameStart := 0 },
  { event := event68603
    frameStart := 0 },
  { event := event68604
    frameStart := 0 },
  { event := event68605
    frameStart := 0 },
  { event := event68606
    frameStart := 0 },
  { event := event68607
    frameStart := 68607 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events267
