import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events310

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event79360 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27986⟩⟩, .relation 79359 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨27445⟩⟩]⟩, (-1)⟩)

def event79361 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27986⟩⟩, .operator (⟨79352, 0⟩, ⟨79288, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27985⟩⟩]⟩, (1)⟩)

def exact79362RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27985⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨27445⟩⟩]⟩, (-1)⟩]

theorem exact79362RawTermsValid :
    exact79362RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79362 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27986⟩⟩) exact79362RawTerms .large 79355 (.finite 2997870350080095027200) (some (79357))

def event79363 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26909⟩⟩) 0 ⟨26240⟩ 3258

def event79364 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26909⟩⟩) (.authority (.relationPreimageSource ⟨47⟩))

def exact79365RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨26909⟩⟩]⟩, (1)⟩]

theorem exact79365RawTermsValid :
    exact79365RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79365 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26909⟩⟩) exact79365RawTerms (.finite 5647228698) 79364 .exactZero (none)

def event79366 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26911⟩⟩) 0 ⟨26909⟩ 79365

def event79367 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26911⟩⟩) 1 ⟨2370⟩ 4

def event79368 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26911⟩⟩) (.scale (.predecessor 0 79366 .coefficient) (.value (.predecessor 1 79367 .coefficient)))

def exact79369RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨26909⟩⟩]⟩, (1)⟩]

theorem exact79369RawTermsValid :
    exact79369RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79369 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26911⟩⟩) exact79369RawTerms (.finite 5647228698) 79368 .exactZero (none)

def event79370 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26912⟩⟩) 0 ⟨10368⟩ 75995

def event79371 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26912⟩⟩) 1 ⟨26911⟩ 79369

def event79372 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26912⟩⟩) (.product (.predecessor 0 79370 .coefficient) (.predecessor 1 79371 .coefficient) (⟨false, false, none, none, none⟩))

def event79373 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26912⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨26909⟩⟩]⟩) [⟨.result 79365 .coefficient, false, none⟩])

def event79374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26912⟩⟩) (.product (.result 75995 .summary) (.transfer 79373) (⟨false, false, none, none, none⟩))

def event79375 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26912⟩⟩, .operator (⟨75995, 0⟩, ⟨79369, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26909⟩⟩]⟩, (1)⟩)

def event79376 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨26910⟩⟩)

def event79377 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event79378 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event79379 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event79380 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event79381 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event79382 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event79383 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event79384 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event79385 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 79384

def event79386 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 79382

def event79387 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 79385 .coefficient) (.value (.predecessor 1 79386 .coefficient)))

def event79388 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event79389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 79388

def event79390 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 79380

def event79391 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 79389 .coefficient, .predecessor 1 79390 .coefficient])

def event79392 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event79393 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 79392

def event79394 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 79378

def event79395 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 79394 .coefficient))

def event79396 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event79397 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26238⟩⟩) 0 ⟨10325⟩ 79396

def event79398 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26238⟩⟩) (.authority (.programFamilyFact))

def exact79399RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26238⟩⟩], []⟩, (1)⟩]

theorem exact79399RawTermsValid :
    exact79399RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79399 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26238⟩⟩) exact79399RawTerms (.finite 30) 79398 .exactZero (none)

def event79400 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13071⟩⟩) 0 ⟨10325⟩ 79396

def event79401 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13071⟩⟩) (.authority (.programFamilyFact))

def exact79402RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩], []⟩, (1)⟩]

theorem exact79402RawTermsValid :
    exact79402RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79402 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13071⟩⟩) exact79402RawTerms (.finite 30) 79401 .exactZero (none)

def event79403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26239⟩⟩) 0 ⟨13071⟩ 79402

def event79404 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26239⟩⟩) 1 ⟨26238⟩ 79399

def event79405 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26239⟩⟩) (.product (.predecessor 0 79403 .coefficient) (.predecessor 1 79404 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event79406 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26239⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], []⟩) [⟨.result 79402 .coefficient, true, some 1⟩, ⟨.result 79399 .coefficient, true, some 1⟩])

def event79407 : Event := .survivorFold (1) 79406

def exact79408RawTerms : List Term := []

theorem exact79408RawTermsValid :
    exact79408RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79408 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26239⟩⟩) exact79408RawTerms (.finite 900) 79405 (.finite 900) (some (79406))

def event79409 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26240⟩⟩) 0 ⟨26239⟩ 79408

def event79410 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26240⟩⟩) (.identity (.predecessor 0 79409 .coefficient))

def event79411 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26240⟩⟩) (.finite 900)

def event79412 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26909⟩⟩) 0 ⟨26240⟩ 79411

def event79413 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26909⟩⟩) (.authority (.relationPreimageSource ⟨47⟩))

def exact79414RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨26909⟩⟩]⟩, (1)⟩]

theorem exact79414RawTermsValid :
    exact79414RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79414 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26909⟩⟩) exact79414RawTerms (.finite 5647228698) 79413 .exactZero (none)

def event79415 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact79416RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact79416RawTermsValid :
    exact79416RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79416 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact79416RawTerms .large 79415 .exactZero (none)

def event79417 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26910⟩⟩) 0 ⟨35⟩ 79416

def event79418 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26910⟩⟩) 1 ⟨26909⟩ 79414

def event79419 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26910⟩⟩) (.product (.predecessor 0 79417 .coefficient) (.predecessor 1 79418 .coefficient) (⟨false, false, none, none, none⟩))

def event79420 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26910⟩⟩, .operator (⟨79416, 0⟩, ⟨79414, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26909⟩⟩]⟩, (1)⟩)

def exact79421RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26909⟩⟩]⟩, (1)⟩]

theorem exact79421RawTermsValid :
    exact79421RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79421 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26910⟩⟩) exact79421RawTerms .large 79419 .exactZero (none)

def event79422 : Event := .preFoldPolynomial 79421 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26909⟩⟩]⟩, (1)⟩] .exactZero none

def exact79423RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26909⟩⟩]⟩, (1)⟩]

def event79423 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨26910⟩⟩) 79422 exact79423RawTerms .large 79419 .exactZero (none)

def event79424 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨27989⟩⟩)

def event79425 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event79426 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event79427 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event79428 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event79429 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event79430 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event79431 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event79432 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event79433 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 79432

def event79434 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 79430

def event79435 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 79433 .coefficient) (.value (.predecessor 1 79434 .coefficient)))

def event79436 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event79437 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 79436

def event79438 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 79428

def event79439 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 79437 .coefficient, .predecessor 1 79438 .coefficient])

def event79440 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event79441 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 79440

def event79442 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 79426

def event79443 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 79442 .coefficient))

def event79444 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event79445 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26238⟩⟩) 0 ⟨10325⟩ 79444

def event79446 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26238⟩⟩) (.authority (.programFamilyFact))

def exact79447RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26238⟩⟩], []⟩, (1)⟩]

theorem exact79447RawTermsValid :
    exact79447RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79447 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26238⟩⟩) exact79447RawTerms (.finite 30) 79446 .exactZero (none)

def event79448 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13071⟩⟩) 0 ⟨10325⟩ 79444

def event79449 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13071⟩⟩) (.authority (.programFamilyFact))

def exact79450RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩], []⟩, (1)⟩]

theorem exact79450RawTermsValid :
    exact79450RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79450 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13071⟩⟩) exact79450RawTerms (.finite 30) 79449 .exactZero (none)

def event79451 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26239⟩⟩) 0 ⟨13071⟩ 79450

def event79452 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26239⟩⟩) 1 ⟨26238⟩ 79447

def event79453 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26239⟩⟩) (.product (.predecessor 0 79451 .coefficient) (.predecessor 1 79452 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event79454 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26239⟩⟩, .operator (⟨79450, 0⟩, ⟨79447, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], []⟩, (1)⟩)

def exact79455RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], []⟩, (1)⟩]

theorem exact79455RawTermsValid :
    exact79455RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79455 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26239⟩⟩) exact79455RawTerms (.finite 900) 79453 .exactZero (none)

def event79456 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26240⟩⟩) 0 ⟨26239⟩ 79455

def event79457 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26240⟩⟩) (.identity (.predecessor 0 79456 .coefficient))

def event79458 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26240⟩⟩) (.finite 900)

def event79459 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27444⟩⟩) 0 ⟨26240⟩ 79458

def event79460 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27444⟩⟩) (.authority (.programFamilyFact))

def event79461 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27444⟩⟩) (.finite 3720)

def event79462 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event79463 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27445⟩⟩) 0 ⟨7177⟩ 79462

def event79464 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27445⟩⟩) 1 ⟨27444⟩ 79461

def event79465 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27445⟩⟩) (.authority (.operator))

def exact79466RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27445⟩⟩]⟩, (1)⟩]

theorem exact79466RawTermsValid :
    exact79466RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79466 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27445⟩⟩) exact79466RawTerms .large 79465 .exactZero (none)

def event79467 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27985⟩⟩) 0 ⟨27445⟩ 79466

def event79468 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27985⟩⟩) (.authority (.operator))

def exact79469RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27985⟩⟩]⟩, (1)⟩]

theorem exact79469RawTermsValid :
    exact79469RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79469 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27985⟩⟩) exact79469RawTerms (.finite 8192) 79468 .exactZero (none)

def event79470 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event79471 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event79472 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27710⟩⟩) 0 ⟨26240⟩ 79458

def event79473 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27710⟩⟩) 1 ⟨136⟩ 79471

def event79474 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27710⟩⟩) (.sum [.predecessor 0 79472 .coefficient, .predecessor 1 79473 .coefficient])

def event79475 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27710⟩⟩) (.finite 900)

def event79476 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27711⟩⟩) 0 ⟨27710⟩ 79475

def event79477 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27711⟩⟩) (.identity (.predecessor 0 79476 .coefficient))

def exact79478RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], []⟩, (1)⟩]

theorem exact79478RawTermsValid :
    exact79478RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79478 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27711⟩⟩) exact79478RawTerms (.finite 900) 79477 .exactZero (none)

def event79479 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact79480RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact79480RawTermsValid :
    exact79480RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79480 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact79480RawTerms .large 79479 .exactZero (none)

def event79481 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27712⟩⟩) 0 ⟨6908⟩ 79480

def event79482 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27712⟩⟩) 1 ⟨27711⟩ 79478

def event79483 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27712⟩⟩) (.product (.predecessor 0 79481 .coefficient) (.predecessor 1 79482 .coefficient) (⟨false, false, none, none, none⟩))

def event79484 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27712⟩⟩, .operator (⟨79480, 0⟩, ⟨79478, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact79485RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact79485RawTermsValid :
    exact79485RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79485 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27712⟩⟩) exact79485RawTerms .large 79483 .exactZero (none)

def event79486 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event79487 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event79488 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 79462

def event79489 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact79490RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact79490RawTermsValid :
    exact79490RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79490 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact79490RawTerms .large 79489 .exactZero (none)

def event79491 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7278⟩⟩) 0 ⟨7178⟩ 79490

def event79492 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7278⟩⟩) (.identity (.predecessor 0 79491 .coefficient))

def exact79493RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7278⟩⟩]⟩, (1)⟩]

theorem exact79493RawTermsValid :
    exact79493RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79493 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7278⟩⟩) exact79493RawTerms .large 79492 .exactZero (none)

def event79494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9544⟩⟩) 0 ⟨7278⟩ 79493

def event79495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9544⟩⟩) (.authority (.operator))

def exact79496RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩, (1)⟩]

theorem exact79496RawTermsValid :
    exact79496RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79496 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9544⟩⟩) exact79496RawTerms (.finite 8192) 79495 .exactZero (none)

def event79497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9545⟩⟩) 0 ⟨9544⟩ 79496

def event79498 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9545⟩⟩) 1 ⟨2370⟩ 79487

def event79499 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9545⟩⟩) (.scale (.predecessor 0 79497 .coefficient) (.value (.predecessor 1 79498 .coefficient)))

def exact79500RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩, (1)⟩]

theorem exact79500RawTermsValid :
    exact79500RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79500 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9545⟩⟩) exact79500RawTerms (.finite 8192) 79499 .exactZero (none)

def event79501 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7295⟩⟩) 0 ⟨7178⟩ 79490

def event79502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7295⟩⟩) (.identity (.predecessor 0 79501 .coefficient))

def exact79503RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩]⟩, (1)⟩]

theorem exact79503RawTermsValid :
    exact79503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79503 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7295⟩⟩) exact79503RawTerms .large 79502 .exactZero (none)

def event79504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9546⟩⟩) 0 ⟨7295⟩ 79503

def event79505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9546⟩⟩) 1 ⟨9545⟩ 79500

def event79506 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9546⟩⟩) (.product (.predecessor 0 79504 .coefficient) (.predecessor 1 79505 .coefficient) (⟨false, false, none, none, none⟩))

def event79507 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9546⟩⟩, .operator (⟨79503, 0⟩, ⟨79500, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩, (1)⟩)

def exact79508RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩, (1)⟩]

theorem exact79508RawTermsValid :
    exact79508RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79508 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9546⟩⟩) exact79508RawTerms .large 79506 .exactZero (none)

def event79509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27713⟩⟩) 0 ⟨9546⟩ 79508

def event79510 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27713⟩⟩) 1 ⟨27712⟩ 79485

def event79511 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27713⟩⟩) (.sum [.predecessor 0 79509 .coefficient, .predecessor 1 79510 .coefficient])

def exact79512RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79512RawTermsValid :
    exact79512RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79512 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27713⟩⟩) exact79512RawTerms .large 79511 .exactZero (none)

def event79513 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27988⟩⟩) 0 ⟨27713⟩ 79512

def event79514 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27988⟩⟩) 1 ⟨27985⟩ 79469

def event79515 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27988⟩⟩) (.product (.predecessor 0 79513 .coefficient) (.predecessor 1 79514 .coefficient) (⟨false, false, none, none, none⟩))

def event79516 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27988⟩⟩, .operator (⟨79512, 0⟩, ⟨79469, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27985⟩⟩]⟩, (1)⟩)

def event79517 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27988⟩⟩, .operator (⟨79512, 1⟩, ⟨79469, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27985⟩⟩]⟩, (-1)⟩)

def event79518 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨27988⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27985⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27985⟩⟩) ⟨27445⟩ 79466)

def event79519 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27988⟩⟩, .relation 79518 0, ⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨27445⟩⟩]⟩, (-1)⟩)

def exact79520RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27985⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨27445⟩⟩]⟩, (-1)⟩]

theorem exact79520RawTermsValid :
    exact79520RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79520 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27988⟩⟩) exact79520RawTerms .large 79515 .exactZero (none)

def event79521 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26456⟩⟩) 0 ⟨26240⟩ 79458

def event79522 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26456⟩⟩) (.authority (.programFamilyFact))

def exact79523RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], []⟩, (1)⟩]

theorem exact79523RawTermsValid :
    exact79523RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79523 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26456⟩⟩) exact79523RawTerms (.finite 30) 79522 .exactZero (none)

def event79524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26458⟩⟩) 0 ⟨6908⟩ 79480

def event79525 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26458⟩⟩) 1 ⟨26456⟩ 79523

def event79526 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26458⟩⟩) (.product (.predecessor 0 79524 .coefficient) (.predecessor 1 79525 .coefficient) (⟨false, true, none, none, some 1⟩))

def event79527 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26458⟩⟩, .operator (⟨79480, 0⟩, ⟨79523, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact79528RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact79528RawTermsValid :
    exact79528RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79528 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26458⟩⟩) exact79528RawTerms .large 79526 .exactZero (none)

def event79529 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7189⟩⟩) 0 ⟨7177⟩ 79462

def event79530 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7189⟩⟩) (.authority (.operator))

def exact79531RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩]

theorem exact79531RawTermsValid :
    exact79531RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79531 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7189⟩⟩) exact79531RawTerms .large 79530 .exactZero (none)

def event79532 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26459⟩⟩) 0 ⟨7189⟩ 79531

def event79533 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26459⟩⟩) 1 ⟨26458⟩ 79528

def event79534 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26459⟩⟩) (.sum [.predecessor 0 79532 .coefficient, .predecessor 1 79533 .coefficient])

def exact79535RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79535RawTermsValid :
    exact79535RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79535 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26459⟩⟩) exact79535RawTerms .large 79534 .exactZero (none)

def event79536 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27989⟩⟩) 0 ⟨26459⟩ 79535

def event79537 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27989⟩⟩) 1 ⟨27988⟩ 79520

def event79538 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27989⟩⟩) (.sum [.predecessor 0 79536 .coefficient, .predecessor 1 79537 .coefficient])

def exact79539RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27985⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨27445⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79539RawTermsValid :
    exact79539RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79539 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27989⟩⟩) exact79539RawTerms .large 79538 .exactZero (none)

def event79540 : Event := .preFoldPolynomial 79539 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27985⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨27445⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact79541RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27985⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨27445⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event79541 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨27989⟩⟩) 79540 exact79541RawTerms .large 79538 .exactZero (none)

def event79542 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨26240⟩⟩) ⟨⟨68⟩, ⟨47⟩, ⟨135⟩⟩ ⟨79376, 79542⟩

def event79543 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨26912⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26909⟩⟩]⟩) (1) 0 2 (.universal 79542 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26909⟩⟩]⟩) (none) 79541)

def event79544 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26912⟩⟩, .relation 79543 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩)

def event79545 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26912⟩⟩, .relation 79543 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27985⟩⟩]⟩, (-1)⟩)

def event79546 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26912⟩⟩, .relation 79543 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨27445⟩⟩]⟩, (1)⟩)

def event79547 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26912⟩⟩, .relation 79543 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact79548RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27985⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨27445⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79548RawTermsValid :
    exact79548RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79548 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26912⟩⟩) exact79548RawTerms .large 79372 (.finite 202072841853861888) (some (79374))

def event79549 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27987⟩⟩) 0 ⟨26912⟩ 79548

def event79550 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27987⟩⟩) 1 ⟨27986⟩ 79362

def event79551 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27987⟩⟩) (.sum [.predecessor 0 79549 .coefficient, .predecessor 1 79550 .coefficient])

def event79552 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27987⟩⟩, .operator (⟨79548, 2⟩, ⟨79362, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], [⟨.program ⟨257⟩, ⟨27445⟩⟩]⟩, (-1)⟩)

def event79553 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27987⟩⟩, .operator (⟨79548, 1⟩, ⟨79362, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27985⟩⟩]⟩, (1)⟩)

def event79554 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27987⟩⟩) (.sum [.result 79548 .summary, .result 79362 .summary])

def exact79555RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79555RawTermsValid :
    exact79555RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79555 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27987⟩⟩) exact79555RawTerms .large 79551 (.finite 2998072422921948889088) (some (79554))

def event79556 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28441⟩⟩) 0 ⟨27987⟩ 79555

def event79557 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28441⟩⟩) 1 ⟨28439⟩ 79278

def event79558 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28441⟩⟩) (.product (.predecessor 0 79556 .coefficient) (.predecessor 1 79557 .coefficient) (⟨false, false, none, none, none⟩))

def event79559 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28441⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩) [⟨.result 79278 .coefficient, false, none⟩])

def event79560 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28441⟩⟩) (.product (.result 79555 .summary) (.transfer 79559) (⟨false, false, none, none, none⟩))

def event79561 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28441⟩⟩, .operator (⟨79555, 0⟩, ⟨79278, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩, (1)⟩)

def event79562 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28441⟩⟩, .operator (⟨79555, 1⟩, ⟨79278, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩, (-1)⟩)

def event79563 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨28441⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28439⟩⟩) ⟨27615⟩ 79275)

def event79564 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28441⟩⟩, .relation 79563 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27615⟩⟩]⟩, (-1)⟩)

def exact79565RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27615⟩⟩]⟩, (-1)⟩]

theorem exact79565RawTermsValid :
    exact79565RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79565 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28441⟩⟩) exact79565RawTerms .large 79558 (.finite 32191557518723128098041228165120) (some (79560))

def event79566 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27276⟩⟩) 0 ⟨26457⟩ 3264

def event79567 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27276⟩⟩) (.authority (.relationPreimageSource ⟨79⟩))

def exact79568RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩, (1)⟩]

theorem exact79568RawTermsValid :
    exact79568RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79568 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27276⟩⟩) exact79568RawTerms (.finite 5647228698) 79567 .exactZero (none)

def event79569 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27278⟩⟩) 0 ⟨27276⟩ 79568

def event79570 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27278⟩⟩) 1 ⟨2370⟩ 4

def event79571 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27278⟩⟩) (.scale (.predecessor 0 79569 .coefficient) (.value (.predecessor 1 79570 .coefficient)))

def exact79572RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩, (1)⟩]

theorem exact79572RawTermsValid :
    exact79572RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79572 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27278⟩⟩) exact79572RawTerms (.finite 5647228698) 79571 .exactZero (none)

def event79573 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27279⟩⟩) 0 ⟨10368⟩ 75995

def event79574 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27279⟩⟩) 1 ⟨27278⟩ 79572

def event79575 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27279⟩⟩) (.product (.predecessor 0 79573 .coefficient) (.predecessor 1 79574 .coefficient) (⟨false, false, none, none, none⟩))

def event79576 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27279⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩) [⟨.result 79568 .coefficient, false, none⟩])

def event79577 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27279⟩⟩) (.product (.result 75995 .summary) (.transfer 79576) (⟨false, false, none, none, none⟩))

def event79578 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27279⟩⟩, .operator (⟨75995, 0⟩, ⟨79572, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩, (1)⟩)

def event79579 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨27277⟩⟩)

def event79580 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event79581 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event79582 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event79583 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event79584 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event79585 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event79586 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event79587 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event79588 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 79587

def event79589 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 79585

def event79590 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 79588 .coefficient) (.value (.predecessor 1 79589 .coefficient)))

def event79591 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event79592 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 79591

def event79593 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 79583

def event79594 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 79592 .coefficient, .predecessor 1 79593 .coefficient])

def event79595 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event79596 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 79595

def event79597 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 79581

def event79598 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 79597 .coefficient))

def event79599 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event79600 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26238⟩⟩) 0 ⟨10325⟩ 79599

def event79601 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26238⟩⟩) (.authority (.programFamilyFact))

def exact79602RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26238⟩⟩], []⟩, (1)⟩]

theorem exact79602RawTermsValid :
    exact79602RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79602 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26238⟩⟩) exact79602RawTerms (.finite 30) 79601 .exactZero (none)

def event79603 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13071⟩⟩) 0 ⟨10325⟩ 79599

def event79604 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13071⟩⟩) (.authority (.programFamilyFact))

def exact79605RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩], []⟩, (1)⟩]

theorem exact79605RawTermsValid :
    exact79605RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79605 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13071⟩⟩) exact79605RawTerms (.finite 30) 79604 .exactZero (none)

def event79606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26239⟩⟩) 0 ⟨13071⟩ 79605

def event79607 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26239⟩⟩) 1 ⟨26238⟩ 79602

def event79608 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26239⟩⟩) (.product (.predecessor 0 79606 .coefficient) (.predecessor 1 79607 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event79609 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26239⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], []⟩) [⟨.result 79605 .coefficient, true, some 1⟩, ⟨.result 79602 .coefficient, true, some 1⟩])

def event79610 : Event := .survivorFold (1) 79609

def exact79611RawTerms : List Term := []

theorem exact79611RawTermsValid :
    exact79611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79611 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26239⟩⟩) exact79611RawTerms (.finite 900) 79608 (.finite 900) (some (79609))

def event79612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26240⟩⟩) 0 ⟨26239⟩ 79611

def event79613 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26240⟩⟩) (.identity (.predecessor 0 79612 .coefficient))

def event79614 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26240⟩⟩) (.finite 900)

def event79615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26456⟩⟩) 0 ⟨26240⟩ 79614

def eventLeaf4960 : Array AnnotatedEvent := #[
  { event := event79360
    frameStart := 0 },
  { event := event79361
    frameStart := 0 },
  { event := event79362
    frameStart := 0 },
  { event := event79363
    frameStart := 0 },
  { event := event79364
    frameStart := 0 },
  { event := event79365
    frameStart := 0 },
  { event := event79366
    frameStart := 0 },
  { event := event79367
    frameStart := 0 },
  { event := event79368
    frameStart := 0 },
  { event := event79369
    frameStart := 0 },
  { event := event79370
    frameStart := 0 },
  { event := event79371
    frameStart := 0 },
  { event := event79372
    frameStart := 0 },
  { event := event79373
    frameStart := 0 },
  { event := event79374
    frameStart := 0 },
  { event := event79375
    frameStart := 0 }
]

def eventLeaf4961 : Array AnnotatedEvent := #[
  { event := event79376
    frameStart := 79376 },
  { event := event79377
    frameStart := 79376 },
  { event := event79378
    frameStart := 79376 },
  { event := event79379
    frameStart := 79376 },
  { event := event79380
    frameStart := 79376 },
  { event := event79381
    frameStart := 79376 },
  { event := event79382
    frameStart := 79376 },
  { event := event79383
    frameStart := 79376 },
  { event := event79384
    frameStart := 79376 },
  { event := event79385
    frameStart := 79376 },
  { event := event79386
    frameStart := 79376 },
  { event := event79387
    frameStart := 79376 },
  { event := event79388
    frameStart := 79376 },
  { event := event79389
    frameStart := 79376 },
  { event := event79390
    frameStart := 79376 },
  { event := event79391
    frameStart := 79376 }
]

def eventLeaf4962 : Array AnnotatedEvent := #[
  { event := event79392
    frameStart := 79376 },
  { event := event79393
    frameStart := 79376 },
  { event := event79394
    frameStart := 79376 },
  { event := event79395
    frameStart := 79376 },
  { event := event79396
    frameStart := 79376 },
  { event := event79397
    frameStart := 79376 },
  { event := event79398
    frameStart := 79376 },
  { event := event79399
    frameStart := 79376 },
  { event := event79400
    frameStart := 79376 },
  { event := event79401
    frameStart := 79376 },
  { event := event79402
    frameStart := 79376 },
  { event := event79403
    frameStart := 79376 },
  { event := event79404
    frameStart := 79376 },
  { event := event79405
    frameStart := 79376 },
  { event := event79406
    frameStart := 79376 },
  { event := event79407
    frameStart := 79376 }
]

def eventLeaf4963 : Array AnnotatedEvent := #[
  { event := event79408
    frameStart := 79376 },
  { event := event79409
    frameStart := 79376 },
  { event := event79410
    frameStart := 79376 },
  { event := event79411
    frameStart := 79376 },
  { event := event79412
    frameStart := 79376 },
  { event := event79413
    frameStart := 79376 },
  { event := event79414
    frameStart := 79376 },
  { event := event79415
    frameStart := 79376 },
  { event := event79416
    frameStart := 79376 },
  { event := event79417
    frameStart := 79376 },
  { event := event79418
    frameStart := 79376 },
  { event := event79419
    frameStart := 79376 },
  { event := event79420
    frameStart := 79376 },
  { event := event79421
    frameStart := 79376 },
  { event := event79422
    frameStart := 79376 },
  { event := event79423
    frameStart := 79376 }
]

def eventLeaf4964 : Array AnnotatedEvent := #[
  { event := event79424
    frameStart := 79424 },
  { event := event79425
    frameStart := 79424 },
  { event := event79426
    frameStart := 79424 },
  { event := event79427
    frameStart := 79424 },
  { event := event79428
    frameStart := 79424 },
  { event := event79429
    frameStart := 79424 },
  { event := event79430
    frameStart := 79424 },
  { event := event79431
    frameStart := 79424 },
  { event := event79432
    frameStart := 79424 },
  { event := event79433
    frameStart := 79424 },
  { event := event79434
    frameStart := 79424 },
  { event := event79435
    frameStart := 79424 },
  { event := event79436
    frameStart := 79424 },
  { event := event79437
    frameStart := 79424 },
  { event := event79438
    frameStart := 79424 },
  { event := event79439
    frameStart := 79424 }
]

def eventLeaf4965 : Array AnnotatedEvent := #[
  { event := event79440
    frameStart := 79424 },
  { event := event79441
    frameStart := 79424 },
  { event := event79442
    frameStart := 79424 },
  { event := event79443
    frameStart := 79424 },
  { event := event79444
    frameStart := 79424 },
  { event := event79445
    frameStart := 79424 },
  { event := event79446
    frameStart := 79424 },
  { event := event79447
    frameStart := 79424 },
  { event := event79448
    frameStart := 79424 },
  { event := event79449
    frameStart := 79424 },
  { event := event79450
    frameStart := 79424 },
  { event := event79451
    frameStart := 79424 },
  { event := event79452
    frameStart := 79424 },
  { event := event79453
    frameStart := 79424 },
  { event := event79454
    frameStart := 79424 },
  { event := event79455
    frameStart := 79424 }
]

def eventLeaf4966 : Array AnnotatedEvent := #[
  { event := event79456
    frameStart := 79424 },
  { event := event79457
    frameStart := 79424 },
  { event := event79458
    frameStart := 79424 },
  { event := event79459
    frameStart := 79424 },
  { event := event79460
    frameStart := 79424 },
  { event := event79461
    frameStart := 79424 },
  { event := event79462
    frameStart := 79424 },
  { event := event79463
    frameStart := 79424 },
  { event := event79464
    frameStart := 79424 },
  { event := event79465
    frameStart := 79424 },
  { event := event79466
    frameStart := 79424 },
  { event := event79467
    frameStart := 79424 },
  { event := event79468
    frameStart := 79424 },
  { event := event79469
    frameStart := 79424 },
  { event := event79470
    frameStart := 79424 },
  { event := event79471
    frameStart := 79424 }
]

def eventLeaf4967 : Array AnnotatedEvent := #[
  { event := event79472
    frameStart := 79424 },
  { event := event79473
    frameStart := 79424 },
  { event := event79474
    frameStart := 79424 },
  { event := event79475
    frameStart := 79424 },
  { event := event79476
    frameStart := 79424 },
  { event := event79477
    frameStart := 79424 },
  { event := event79478
    frameStart := 79424 },
  { event := event79479
    frameStart := 79424 },
  { event := event79480
    frameStart := 79424 },
  { event := event79481
    frameStart := 79424 },
  { event := event79482
    frameStart := 79424 },
  { event := event79483
    frameStart := 79424 },
  { event := event79484
    frameStart := 79424 },
  { event := event79485
    frameStart := 79424 },
  { event := event79486
    frameStart := 79424 },
  { event := event79487
    frameStart := 79424 }
]

def eventLeaf4968 : Array AnnotatedEvent := #[
  { event := event79488
    frameStart := 79424 },
  { event := event79489
    frameStart := 79424 },
  { event := event79490
    frameStart := 79424 },
  { event := event79491
    frameStart := 79424 },
  { event := event79492
    frameStart := 79424 },
  { event := event79493
    frameStart := 79424 },
  { event := event79494
    frameStart := 79424 },
  { event := event79495
    frameStart := 79424 },
  { event := event79496
    frameStart := 79424 },
  { event := event79497
    frameStart := 79424 },
  { event := event79498
    frameStart := 79424 },
  { event := event79499
    frameStart := 79424 },
  { event := event79500
    frameStart := 79424 },
  { event := event79501
    frameStart := 79424 },
  { event := event79502
    frameStart := 79424 },
  { event := event79503
    frameStart := 79424 }
]

def eventLeaf4969 : Array AnnotatedEvent := #[
  { event := event79504
    frameStart := 79424 },
  { event := event79505
    frameStart := 79424 },
  { event := event79506
    frameStart := 79424 },
  { event := event79507
    frameStart := 79424 },
  { event := event79508
    frameStart := 79424 },
  { event := event79509
    frameStart := 79424 },
  { event := event79510
    frameStart := 79424 },
  { event := event79511
    frameStart := 79424 },
  { event := event79512
    frameStart := 79424 },
  { event := event79513
    frameStart := 79424 },
  { event := event79514
    frameStart := 79424 },
  { event := event79515
    frameStart := 79424 },
  { event := event79516
    frameStart := 79424 },
  { event := event79517
    frameStart := 79424 },
  { event := event79518
    frameStart := 79424 },
  { event := event79519
    frameStart := 79424 }
]

def eventLeaf4970 : Array AnnotatedEvent := #[
  { event := event79520
    frameStart := 79424 },
  { event := event79521
    frameStart := 79424 },
  { event := event79522
    frameStart := 79424 },
  { event := event79523
    frameStart := 79424 },
  { event := event79524
    frameStart := 79424 },
  { event := event79525
    frameStart := 79424 },
  { event := event79526
    frameStart := 79424 },
  { event := event79527
    frameStart := 79424 },
  { event := event79528
    frameStart := 79424 },
  { event := event79529
    frameStart := 79424 },
  { event := event79530
    frameStart := 79424 },
  { event := event79531
    frameStart := 79424 },
  { event := event79532
    frameStart := 79424 },
  { event := event79533
    frameStart := 79424 },
  { event := event79534
    frameStart := 79424 },
  { event := event79535
    frameStart := 79424 }
]

def eventLeaf4971 : Array AnnotatedEvent := #[
  { event := event79536
    frameStart := 79424 },
  { event := event79537
    frameStart := 79424 },
  { event := event79538
    frameStart := 79424 },
  { event := event79539
    frameStart := 79424 },
  { event := event79540
    frameStart := 79424 },
  { event := event79541
    frameStart := 79424 },
  { event := event79542
    frameStart := 0 },
  { event := event79543
    frameStart := 0 },
  { event := event79544
    frameStart := 0 },
  { event := event79545
    frameStart := 0 },
  { event := event79546
    frameStart := 0 },
  { event := event79547
    frameStart := 0 },
  { event := event79548
    frameStart := 0 },
  { event := event79549
    frameStart := 0 },
  { event := event79550
    frameStart := 0 },
  { event := event79551
    frameStart := 0 }
]

def eventLeaf4972 : Array AnnotatedEvent := #[
  { event := event79552
    frameStart := 0 },
  { event := event79553
    frameStart := 0 },
  { event := event79554
    frameStart := 0 },
  { event := event79555
    frameStart := 0 },
  { event := event79556
    frameStart := 0 },
  { event := event79557
    frameStart := 0 },
  { event := event79558
    frameStart := 0 },
  { event := event79559
    frameStart := 0 },
  { event := event79560
    frameStart := 0 },
  { event := event79561
    frameStart := 0 },
  { event := event79562
    frameStart := 0 },
  { event := event79563
    frameStart := 0 },
  { event := event79564
    frameStart := 0 },
  { event := event79565
    frameStart := 0 },
  { event := event79566
    frameStart := 0 },
  { event := event79567
    frameStart := 0 }
]

def eventLeaf4973 : Array AnnotatedEvent := #[
  { event := event79568
    frameStart := 0 },
  { event := event79569
    frameStart := 0 },
  { event := event79570
    frameStart := 0 },
  { event := event79571
    frameStart := 0 },
  { event := event79572
    frameStart := 0 },
  { event := event79573
    frameStart := 0 },
  { event := event79574
    frameStart := 0 },
  { event := event79575
    frameStart := 0 },
  { event := event79576
    frameStart := 0 },
  { event := event79577
    frameStart := 0 },
  { event := event79578
    frameStart := 0 },
  { event := event79579
    frameStart := 79579 },
  { event := event79580
    frameStart := 79579 },
  { event := event79581
    frameStart := 79579 },
  { event := event79582
    frameStart := 79579 },
  { event := event79583
    frameStart := 79579 }
]

def eventLeaf4974 : Array AnnotatedEvent := #[
  { event := event79584
    frameStart := 79579 },
  { event := event79585
    frameStart := 79579 },
  { event := event79586
    frameStart := 79579 },
  { event := event79587
    frameStart := 79579 },
  { event := event79588
    frameStart := 79579 },
  { event := event79589
    frameStart := 79579 },
  { event := event79590
    frameStart := 79579 },
  { event := event79591
    frameStart := 79579 },
  { event := event79592
    frameStart := 79579 },
  { event := event79593
    frameStart := 79579 },
  { event := event79594
    frameStart := 79579 },
  { event := event79595
    frameStart := 79579 },
  { event := event79596
    frameStart := 79579 },
  { event := event79597
    frameStart := 79579 },
  { event := event79598
    frameStart := 79579 },
  { event := event79599
    frameStart := 79579 }
]

def eventLeaf4975 : Array AnnotatedEvent := #[
  { event := event79600
    frameStart := 79579 },
  { event := event79601
    frameStart := 79579 },
  { event := event79602
    frameStart := 79579 },
  { event := event79603
    frameStart := 79579 },
  { event := event79604
    frameStart := 79579 },
  { event := event79605
    frameStart := 79579 },
  { event := event79606
    frameStart := 79579 },
  { event := event79607
    frameStart := 79579 },
  { event := event79608
    frameStart := 79579 },
  { event := event79609
    frameStart := 79579 },
  { event := event79610
    frameStart := 79579 },
  { event := event79611
    frameStart := 79579 },
  { event := event79612
    frameStart := 79579 },
  { event := event79613
    frameStart := 79579 },
  { event := event79614
    frameStart := 79579 },
  { event := event79615
    frameStart := 79579 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events310
