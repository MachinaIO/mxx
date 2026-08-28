import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1103

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event282368 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41555⟩⟩) (.sum [.result 282362 .summary, .result 282178 .summary])

def exact282369RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact282369RawTermsValid :
    exact282369RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282369 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41555⟩⟩) exact282369RawTerms .large 282365 (.finite 2998218789909838430208) (some (282368))

def event282370 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41841⟩⟩) 0 ⟨41555⟩ 282369

def event282371 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41841⟩⟩) 1 ⟨41839⟩ 282094

def event282372 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41841⟩⟩) (.product (.predecessor 0 282370 .coefficient) (.predecessor 1 282371 .coefficient) (⟨false, false, none, none, none⟩))

def event282373 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41841⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨41839⟩⟩]⟩) [⟨.result 282094 .coefficient, false, none⟩])

def event282374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41841⟩⟩) (.product (.result 282369 .summary) (.transfer 282373) (⟨false, false, none, none, none⟩))

def event282375 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41841⟩⟩, .operator (⟨282369, 0⟩, ⟨282094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41839⟩⟩]⟩, (1)⟩)

def event282376 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41841⟩⟩, .operator (⟨282369, 1⟩, ⟨282094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41839⟩⟩]⟩, (-1)⟩)

def event282377 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41841⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41839⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41839⟩⟩) ⟨41207⟩ 282091)

def event282378 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41841⟩⟩, .relation 282377 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨41207⟩⟩]⟩, (-1)⟩)

def exact282379RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41839⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨41207⟩⟩]⟩, (-1)⟩]

theorem exact282379RawTermsValid :
    exact282379RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282379 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41841⟩⟩) exact282379RawTerms .large 282372 (.finite 32193129122288627115968346193920) (some (282374))

def event282380 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40736⟩⟩) 0 ⟨40061⟩ 13638

def event282381 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40736⟩⟩) (.authority (.relationPreimageSource ⟨87⟩))

def exact282382RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40736⟩⟩]⟩, (1)⟩]

theorem exact282382RawTermsValid :
    exact282382RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282382 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40736⟩⟩) exact282382RawTerms (.finite 5647228698) 282381 .exactZero (none)

def event282383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40738⟩⟩) 0 ⟨40736⟩ 282382

def event282384 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40738⟩⟩) 1 ⟨2370⟩ 4

def event282385 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40738⟩⟩) (.scale (.predecessor 0 282383 .coefficient) (.value (.predecessor 1 282384 .coefficient)))

def exact282386RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40736⟩⟩]⟩, (1)⟩]

theorem exact282386RawTermsValid :
    exact282386RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282386 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40738⟩⟩) exact282386RawTerms (.finite 5647228698) 282385 .exactZero (none)

def event282387 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40739⟩⟩) 0 ⟨5491⟩ 280745

def event282388 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40739⟩⟩) 1 ⟨40738⟩ 282386

def event282389 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40739⟩⟩) (.product (.predecessor 0 282387 .coefficient) (.predecessor 1 282388 .coefficient) (⟨false, false, none, none, none⟩))

def event282390 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40739⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨40736⟩⟩]⟩) [⟨.result 282382 .coefficient, false, none⟩])

def event282391 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40739⟩⟩) (.product (.result 280745 .summary) (.transfer 282390) (⟨false, false, none, none, none⟩))

def event282392 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40739⟩⟩, .operator (⟨280745, 0⟩, ⟨282386, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40736⟩⟩]⟩, (1)⟩)

def event282393 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨40737⟩⟩)

def event282394 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event282395 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event282396 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event282397 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event282398 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event282399 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event282400 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event282401 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event282402 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 282401

def event282403 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 282399

def event282404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 282402 .coefficient) (.value (.predecessor 1 282403 .coefficient)))

def event282405 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event282406 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 282405

def event282407 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 282397

def event282408 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 282406 .coefficient, .predecessor 1 282407 .coefficient])

def event282409 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event282410 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 282409

def event282411 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 282395

def event282412 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 282411 .coefficient))

def event282413 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event282414 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39650⟩⟩) 0 ⟨5487⟩ 282413

def event282415 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39650⟩⟩) (.authority (.programFamilyFact))

def exact282416RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39650⟩⟩], []⟩, (1)⟩]

theorem exact282416RawTermsValid :
    exact282416RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282416 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39650⟩⟩) exact282416RawTerms (.finite 46) 282415 .exactZero (none)

def event282417 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14091⟩⟩) 0 ⟨5487⟩ 282413

def event282418 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14091⟩⟩) (.authority (.programFamilyFact))

def exact282419RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14091⟩⟩], []⟩, (1)⟩]

theorem exact282419RawTermsValid :
    exact282419RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282419 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14091⟩⟩) exact282419RawTerms (.finite 46) 282418 .exactZero (none)

def event282420 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39651⟩⟩) 0 ⟨14091⟩ 282419

def event282421 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39651⟩⟩) 1 ⟨39650⟩ 282416

def event282422 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39651⟩⟩) (.product (.predecessor 0 282420 .coefficient) (.predecessor 1 282421 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event282423 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39651⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], []⟩) [⟨.result 282419 .coefficient, true, some 1⟩, ⟨.result 282416 .coefficient, true, some 1⟩])

def event282424 : Event := .survivorFold (1) 282423

def exact282425RawTerms : List Term := []

theorem exact282425RawTermsValid :
    exact282425RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282425 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39651⟩⟩) exact282425RawTerms (.finite 2116) 282422 (.finite 2116) (some (282423))

def event282426 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39652⟩⟩) 0 ⟨39651⟩ 282425

def event282427 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39652⟩⟩) (.identity (.predecessor 0 282426 .coefficient))

def event282428 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39652⟩⟩) (.finite 2116)

def event282429 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40060⟩⟩) 0 ⟨39652⟩ 282428

def event282430 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40060⟩⟩) (.authority (.programFamilyFact))

def exact282431RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40060⟩⟩], []⟩, (1)⟩]

theorem exact282431RawTermsValid :
    exact282431RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282431 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40060⟩⟩) exact282431RawTerms (.finite 46) 282430 .exactZero (none)

def event282432 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40061⟩⟩) 0 ⟨40060⟩ 282431

def event282433 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40061⟩⟩) (.identity (.predecessor 0 282432 .coefficient))

def event282434 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40061⟩⟩) (.finite 46)

def event282435 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40736⟩⟩) 0 ⟨40061⟩ 282434

def event282436 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40736⟩⟩) (.authority (.relationPreimageSource ⟨87⟩))

def exact282437RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨40736⟩⟩]⟩, (1)⟩]

theorem exact282437RawTermsValid :
    exact282437RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282437 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40736⟩⟩) exact282437RawTerms (.finite 5647228698) 282436 .exactZero (none)

def event282438 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact282439RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact282439RawTermsValid :
    exact282439RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282439 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact282439RawTerms .large 282438 .exactZero (none)

def event282440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40737⟩⟩) 0 ⟨35⟩ 282439

def event282441 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40737⟩⟩) 1 ⟨40736⟩ 282437

def event282442 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40737⟩⟩) (.product (.predecessor 0 282440 .coefficient) (.predecessor 1 282441 .coefficient) (⟨false, false, none, none, none⟩))

def event282443 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40737⟩⟩, .operator (⟨282439, 0⟩, ⟨282437, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40736⟩⟩]⟩, (1)⟩)

def exact282444RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40736⟩⟩]⟩, (1)⟩]

theorem exact282444RawTermsValid :
    exact282444RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282444 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40737⟩⟩) exact282444RawTerms .large 282442 .exactZero (none)

def event282445 : Event := .preFoldPolynomial 282444 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40736⟩⟩]⟩, (1)⟩] .exactZero none

def exact282446RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40736⟩⟩]⟩, (1)⟩]

def event282446 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨40737⟩⟩) 282445 exact282446RawTerms .large 282442 .exactZero (none)

def event282447 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨41843⟩⟩)

def event282448 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event282449 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event282450 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event282451 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event282452 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event282453 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event282454 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event282455 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event282456 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 282455

def event282457 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 282453

def event282458 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 282456 .coefficient) (.value (.predecessor 1 282457 .coefficient)))

def event282459 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event282460 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 282459

def event282461 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 282451

def event282462 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 282460 .coefficient, .predecessor 1 282461 .coefficient])

def event282463 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event282464 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 282463

def event282465 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 282449

def event282466 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 282465 .coefficient))

def event282467 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event282468 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39650⟩⟩) 0 ⟨5487⟩ 282467

def event282469 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39650⟩⟩) (.authority (.programFamilyFact))

def exact282470RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39650⟩⟩], []⟩, (1)⟩]

theorem exact282470RawTermsValid :
    exact282470RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282470 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39650⟩⟩) exact282470RawTerms (.finite 46) 282469 .exactZero (none)

def event282471 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14091⟩⟩) 0 ⟨5487⟩ 282467

def event282472 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14091⟩⟩) (.authority (.programFamilyFact))

def exact282473RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14091⟩⟩], []⟩, (1)⟩]

theorem exact282473RawTermsValid :
    exact282473RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282473 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14091⟩⟩) exact282473RawTerms (.finite 46) 282472 .exactZero (none)

def event282474 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39651⟩⟩) 0 ⟨14091⟩ 282473

def event282475 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39651⟩⟩) 1 ⟨39650⟩ 282470

def event282476 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39651⟩⟩) (.product (.predecessor 0 282474 .coefficient) (.predecessor 1 282475 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event282477 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39651⟩⟩, .operator (⟨282473, 0⟩, ⟨282470, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], []⟩, (1)⟩)

def exact282478RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14091⟩⟩, ⟨.program ⟨257⟩, ⟨39650⟩⟩], []⟩, (1)⟩]

theorem exact282478RawTermsValid :
    exact282478RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282478 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39651⟩⟩) exact282478RawTerms (.finite 2116) 282476 .exactZero (none)

def event282479 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39652⟩⟩) 0 ⟨39651⟩ 282478

def event282480 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39652⟩⟩) (.identity (.predecessor 0 282479 .coefficient))

def event282481 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39652⟩⟩) (.finite 2116)

def event282482 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40060⟩⟩) 0 ⟨39652⟩ 282481

def event282483 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40060⟩⟩) (.authority (.programFamilyFact))

def exact282484RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40060⟩⟩], []⟩, (1)⟩]

theorem exact282484RawTermsValid :
    exact282484RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282484 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40060⟩⟩) exact282484RawTerms (.finite 46) 282483 .exactZero (none)

def event282485 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40061⟩⟩) 0 ⟨40060⟩ 282484

def event282486 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40061⟩⟩) (.identity (.predecessor 0 282485 .coefficient))

def event282487 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40061⟩⟩) (.finite 46)

def event282488 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41205⟩⟩) 0 ⟨40061⟩ 282487

def event282489 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41205⟩⟩) (.authority (.programFamilyFact))

def event282490 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41205⟩⟩) (.finite 3720)

def event282491 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event282492 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41207⟩⟩) 0 ⟨7177⟩ 282491

def event282493 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41207⟩⟩) 1 ⟨41205⟩ 282490

def event282494 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41207⟩⟩) (.authority (.operator))

def exact282495RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41207⟩⟩]⟩, (1)⟩]

theorem exact282495RawTermsValid :
    exact282495RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282495 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41207⟩⟩) exact282495RawTerms .large 282494 .exactZero (none)

def event282496 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41839⟩⟩) 0 ⟨41207⟩ 282495

def event282497 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41839⟩⟩) (.authority (.operator))

def exact282498RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41839⟩⟩]⟩, (1)⟩]

theorem exact282498RawTermsValid :
    exact282498RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282498 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41839⟩⟩) exact282498RawTerms (.finite 8192) 282497 .exactZero (none)

def event282499 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event282500 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event282501 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41442⟩⟩) 0 ⟨40061⟩ 282487

def event282502 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41442⟩⟩) 1 ⟨136⟩ 282500

def event282503 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41442⟩⟩) (.sum [.predecessor 0 282501 .coefficient, .predecessor 1 282502 .coefficient])

def event282504 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41442⟩⟩) (.finite 46)

def event282505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41443⟩⟩) 0 ⟨41442⟩ 282504

def event282506 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41443⟩⟩) (.identity (.predecessor 0 282505 .coefficient))

def exact282507RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40060⟩⟩], []⟩, (1)⟩]

theorem exact282507RawTermsValid :
    exact282507RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282507 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41443⟩⟩) exact282507RawTerms (.finite 46) 282506 .exactZero (none)

def event282508 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact282509RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact282509RawTermsValid :
    exact282509RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282509 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact282509RawTerms .large 282508 .exactZero (none)

def event282510 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41444⟩⟩) 0 ⟨6908⟩ 282509

def event282511 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41444⟩⟩) 1 ⟨41443⟩ 282507

def event282512 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41444⟩⟩) (.product (.predecessor 0 282510 .coefficient) (.predecessor 1 282511 .coefficient) (⟨false, false, none, none, none⟩))

def event282513 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41444⟩⟩, .operator (⟨282509, 0⟩, ⟨282507, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact282514RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact282514RawTermsValid :
    exact282514RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282514 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41444⟩⟩) exact282514RawTerms .large 282512 .exactZero (none)

def event282515 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7193⟩⟩) 0 ⟨7177⟩ 282491

def event282516 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7193⟩⟩) (.authority (.operator))

def exact282517RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩]

theorem exact282517RawTermsValid :
    exact282517RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282517 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7193⟩⟩) exact282517RawTerms .large 282516 .exactZero (none)

def event282518 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41445⟩⟩) 0 ⟨7193⟩ 282517

def event282519 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41445⟩⟩) 1 ⟨41444⟩ 282514

def event282520 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41445⟩⟩) (.sum [.predecessor 0 282518 .coefficient, .predecessor 1 282519 .coefficient])

def exact282521RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact282521RawTermsValid :
    exact282521RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282521 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41445⟩⟩) exact282521RawTerms .large 282520 .exactZero (none)

def event282522 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41840⟩⟩) 0 ⟨41445⟩ 282521

def event282523 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41840⟩⟩) 1 ⟨41839⟩ 282498

def event282524 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41840⟩⟩) (.product (.predecessor 0 282522 .coefficient) (.predecessor 1 282523 .coefficient) (⟨false, false, none, none, none⟩))

def event282525 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41840⟩⟩, .operator (⟨282521, 0⟩, ⟨282498, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41839⟩⟩]⟩, (1)⟩)

def event282526 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41840⟩⟩, .operator (⟨282521, 1⟩, ⟨282498, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41839⟩⟩]⟩, (-1)⟩)

def event282527 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41840⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41839⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41839⟩⟩) ⟨41207⟩ 282495)

def event282528 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41840⟩⟩, .relation 282527 0, ⟨[⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨41207⟩⟩]⟩, (-1)⟩)

def exact282529RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41839⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨41207⟩⟩]⟩, (-1)⟩]

theorem exact282529RawTermsValid :
    exact282529RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282529 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41840⟩⟩) exact282529RawTerms .large 282524 .exactZero (none)

def event282530 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40241⟩⟩) 0 ⟨40061⟩ 282487

def event282531 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40241⟩⟩) (.authority (.programFamilyFact))

def exact282532RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], []⟩, (1)⟩]

theorem exact282532RawTermsValid :
    exact282532RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282532 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40241⟩⟩) exact282532RawTerms (.finite 63) 282531 .exactZero (none)

def event282533 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40242⟩⟩) 0 ⟨6908⟩ 282509

def event282534 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40242⟩⟩) 1 ⟨40241⟩ 282532

def event282535 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40242⟩⟩) (.product (.predecessor 0 282533 .coefficient) (.predecessor 1 282534 .coefficient) (⟨false, true, none, none, some 1⟩))

def event282536 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40242⟩⟩, .operator (⟨282509, 0⟩, ⟨282532, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact282537RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact282537RawTermsValid :
    exact282537RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282537 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40242⟩⟩) exact282537RawTerms .large 282535 .exactZero (none)

def event282538 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7226⟩⟩) 0 ⟨7177⟩ 282491

def event282539 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7226⟩⟩) (.authority (.operator))

def exact282540RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact282540RawTermsValid :
    exact282540RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282540 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7226⟩⟩) exact282540RawTerms .large 282539 .exactZero (none)

def event282541 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40243⟩⟩) 0 ⟨7226⟩ 282540

def event282542 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40243⟩⟩) 1 ⟨40242⟩ 282537

def event282543 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40243⟩⟩) (.sum [.predecessor 0 282541 .coefficient, .predecessor 1 282542 .coefficient])

def exact282544RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact282544RawTermsValid :
    exact282544RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282544 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40243⟩⟩) exact282544RawTerms .large 282543 .exactZero (none)

def event282545 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41843⟩⟩) 0 ⟨40243⟩ 282544

def event282546 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41843⟩⟩) 1 ⟨41840⟩ 282529

def event282547 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41843⟩⟩) (.sum [.predecessor 0 282545 .coefficient, .predecessor 1 282546 .coefficient])

def exact282548RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41839⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨41207⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact282548RawTermsValid :
    exact282548RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282548 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41843⟩⟩) exact282548RawTerms .large 282547 .exactZero (none)

def event282549 : Event := .preFoldPolynomial 282548 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41839⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨41207⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact282550RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41839⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨41207⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event282550 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨41843⟩⟩) 282549 exact282550RawTerms .large 282547 .exactZero (none)

def event282551 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨40061⟩⟩) ⟨⟨105⟩, ⟨87⟩, ⟨135⟩⟩ ⟨282393, 282551⟩

def event282552 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨40739⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40736⟩⟩]⟩) (1) 0 2 (.universal 282551 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40736⟩⟩]⟩) (none) 282550)

def event282553 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40739⟩⟩, .relation 282552 1, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩)

def event282554 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40739⟩⟩, .relation 282552 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41839⟩⟩]⟩, (-1)⟩)

def event282555 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40739⟩⟩, .relation 282552 2, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨41207⟩⟩]⟩, (1)⟩)

def event282556 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40739⟩⟩, .relation 282552 3, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact282557RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41839⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨41207⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact282557RawTermsValid :
    exact282557RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282557 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40739⟩⟩) exact282557RawTerms .large 282389 (.finite 202072841853861888) (some (282391))

def event282558 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41842⟩⟩) 0 ⟨40739⟩ 282557

def event282559 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41842⟩⟩) 1 ⟨41841⟩ 282379

def event282560 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41842⟩⟩) (.sum [.predecessor 0 282558 .coefficient, .predecessor 1 282559 .coefficient])

def event282561 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41842⟩⟩, .operator (⟨282557, 0⟩, ⟨282379, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41839⟩⟩]⟩, (1)⟩)

def event282562 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41842⟩⟩, .operator (⟨282557, 2⟩, ⟨282379, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40060⟩⟩], [⟨.program ⟨257⟩, ⟨41207⟩⟩]⟩, (-1)⟩)

def event282563 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41842⟩⟩) (.sum [.result 282557 .summary, .result 282379 .summary])

def exact282564RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact282564RawTermsValid :
    exact282564RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282564 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41842⟩⟩) exact282564RawTerms .large 282560 (.finite 32193129122288829188810200055808) (some (282563))

def event282565 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38525⟩⟩) 0 ⟨37381⟩ 13661

def event282566 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38525⟩⟩) (.authority (.programFamilyFact))

def event282567 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38525⟩⟩) (.finite 3720)

def event282568 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38527⟩⟩) 0 ⟨7177⟩ 15500

def event282569 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38527⟩⟩) 1 ⟨38525⟩ 282567

def event282570 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38527⟩⟩) (.authority (.operator))

def exact282571RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38527⟩⟩]⟩, (1)⟩]

theorem exact282571RawTermsValid :
    exact282571RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282571 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38527⟩⟩) exact282571RawTerms .large 282570 .exactZero (none)

def event282572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39159⟩⟩) 0 ⟨38527⟩ 282571

def event282573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39159⟩⟩) (.authority (.operator))

def exact282574RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩, (1)⟩]

theorem exact282574RawTermsValid :
    exact282574RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282574 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39159⟩⟩) exact282574RawTerms (.finite 8192) 282573 .exactZero (none)

def event282575 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38392⟩⟩) 0 ⟨36972⟩ 13655

def event282576 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38392⟩⟩) (.authority (.programFamilyFact))

def event282577 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38392⟩⟩) (.finite 3720)

def event282578 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38393⟩⟩) 0 ⟨7177⟩ 15500

def event282579 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38393⟩⟩) 1 ⟨38392⟩ 282577

def event282580 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38393⟩⟩) (.authority (.operator))

def exact282581RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38393⟩⟩]⟩, (1)⟩]

theorem exact282581RawTermsValid :
    exact282581RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282581 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38393⟩⟩) exact282581RawTerms .large 282580 .exactZero (none)

def event282582 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38873⟩⟩) 0 ⟨38393⟩ 282581

def event282583 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38873⟩⟩) (.authority (.operator))

def exact282584RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38873⟩⟩]⟩, (1)⟩]

theorem exact282584RawTermsValid :
    exact282584RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282584 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38873⟩⟩) exact282584RawTerms (.finite 8192) 282583 .exactZero (none)

def event282585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36973⟩⟩) 0 ⟨36970⟩ 13644

def event282586 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36973⟩⟩) 1 ⟨6922⟩ 280653

def event282587 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36973⟩⟩) (.tensor (.predecessor 0 282585 .coefficient) (.predecessor 1 282586 .coefficient) true false)

def event282588 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36973⟩⟩, .operator (⟨13644, 0⟩, ⟨280653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact282589RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact282589RawTermsValid :
    exact282589RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282589 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36973⟩⟩) exact282589RawTerms .large 282587 .exactZero (none)

def event282590 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7903⟩⟩) 0 ⟨5489⟩ 280523

def event282591 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7903⟩⟩) 1 ⟨7281⟩ 19084

def event282592 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7903⟩⟩) (.product (.predecessor 0 282590 .coefficient) (.predecessor 1 282591 .coefficient) (⟨false, false, none, none, none⟩))

def event282593 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7903⟩⟩, .operator (⟨280523, 0⟩, ⟨19084, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩)

def exact282594RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩]

theorem exact282594RawTermsValid :
    exact282594RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282594 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7903⟩⟩) exact282594RawTerms .large 282592 .exactZero (none)

def event282595 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36974⟩⟩) 0 ⟨7903⟩ 282594

def event282596 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36974⟩⟩) 1 ⟨36973⟩ 282589

def event282597 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36974⟩⟩) (.sum [.predecessor 0 282595 .coefficient, .predecessor 1 282596 .coefficient])

def exact282598RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact282598RawTermsValid :
    exact282598RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282598 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36974⟩⟩) exact282598RawTerms .large 282597 .exactZero (none)

def event282599 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36975⟩⟩) 0 ⟨36974⟩ 282598

def event282600 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36975⟩⟩) 1 ⟨107⟩ 19076

def event282601 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36975⟩⟩) (.sum [.predecessor 0 282599 .coefficient, .predecessor 1 282600 .coefficient])

def event282602 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36975⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨107⟩⟩]⟩) [⟨.result 19076 .coefficient, false, none⟩])

def event282603 : Event := .survivorFold (1) 282602

def exact282604RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact282604RawTermsValid :
    exact282604RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282604 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36975⟩⟩) exact282604RawTerms .large 282601 (.finite 26) (some (282602))

def event282605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36976⟩⟩) 0 ⟨36975⟩ 282604

def event282606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36976⟩⟩) 1 ⟨13791⟩ 13647

def event282607 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36976⟩⟩) (.product (.predecessor 0 282605 .coefficient) (.predecessor 1 282606 .coefficient) (⟨false, true, none, none, some 1⟩))

def event282608 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36976⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13791⟩⟩], []⟩) [⟨.result 13647 .coefficient, true, some 1⟩])

def event282609 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36976⟩⟩) (.product (.result 282604 .summary) (.transfer 282608) (⟨false, false, none, none, none⟩))

def event282610 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36976⟩⟩, .operator (⟨282604, 1⟩, ⟨13647, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13791⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event282611 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36976⟩⟩, .operator (⟨282604, 0⟩, ⟨13647, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13791⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩)

def exact282612RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13791⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13791⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact282612RawTermsValid :
    exact282612RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282612 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36976⟩⟩) exact282612RawTerms .large 282607 (.finite 35782656) (some (282609))

def event282613 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13792⟩⟩) 0 ⟨13791⟩ 13647

def event282614 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13792⟩⟩) 1 ⟨6922⟩ 280653

def event282615 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13792⟩⟩) (.tensor (.predecessor 0 282613 .coefficient) (.predecessor 1 282614 .coefficient) true false)

def event282616 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13792⟩⟩, .operator (⟨13647, 0⟩, ⟨280653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13791⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact282617RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13791⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact282617RawTermsValid :
    exact282617RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282617 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13792⟩⟩) exact282617RawTerms .large 282615 .exactZero (none)

def event282618 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7920⟩⟩) 0 ⟨5489⟩ 280523

def event282619 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7920⟩⟩) 1 ⟨7298⟩ 19125

def event282620 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7920⟩⟩) (.product (.predecessor 0 282618 .coefficient) (.predecessor 1 282619 .coefficient) (⟨false, false, none, none, none⟩))

def event282621 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7920⟩⟩, .operator (⟨280523, 0⟩, ⟨19125, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩)

def exact282622RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩]

theorem exact282622RawTermsValid :
    exact282622RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282622 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7920⟩⟩) exact282622RawTerms .large 282620 .exactZero (none)

def event282623 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13793⟩⟩) 0 ⟨7920⟩ 282622

def eventLeaf17648 : Array AnnotatedEvent := #[
  { event := event282368
    frameStart := 0 },
  { event := event282369
    frameStart := 0 },
  { event := event282370
    frameStart := 0 },
  { event := event282371
    frameStart := 0 },
  { event := event282372
    frameStart := 0 },
  { event := event282373
    frameStart := 0 },
  { event := event282374
    frameStart := 0 },
  { event := event282375
    frameStart := 0 },
  { event := event282376
    frameStart := 0 },
  { event := event282377
    frameStart := 0 },
  { event := event282378
    frameStart := 0 },
  { event := event282379
    frameStart := 0 },
  { event := event282380
    frameStart := 0 },
  { event := event282381
    frameStart := 0 },
  { event := event282382
    frameStart := 0 },
  { event := event282383
    frameStart := 0 }
]

def eventLeaf17649 : Array AnnotatedEvent := #[
  { event := event282384
    frameStart := 0 },
  { event := event282385
    frameStart := 0 },
  { event := event282386
    frameStart := 0 },
  { event := event282387
    frameStart := 0 },
  { event := event282388
    frameStart := 0 },
  { event := event282389
    frameStart := 0 },
  { event := event282390
    frameStart := 0 },
  { event := event282391
    frameStart := 0 },
  { event := event282392
    frameStart := 0 },
  { event := event282393
    frameStart := 282393 },
  { event := event282394
    frameStart := 282393 },
  { event := event282395
    frameStart := 282393 },
  { event := event282396
    frameStart := 282393 },
  { event := event282397
    frameStart := 282393 },
  { event := event282398
    frameStart := 282393 },
  { event := event282399
    frameStart := 282393 }
]

def eventLeaf17650 : Array AnnotatedEvent := #[
  { event := event282400
    frameStart := 282393 },
  { event := event282401
    frameStart := 282393 },
  { event := event282402
    frameStart := 282393 },
  { event := event282403
    frameStart := 282393 },
  { event := event282404
    frameStart := 282393 },
  { event := event282405
    frameStart := 282393 },
  { event := event282406
    frameStart := 282393 },
  { event := event282407
    frameStart := 282393 },
  { event := event282408
    frameStart := 282393 },
  { event := event282409
    frameStart := 282393 },
  { event := event282410
    frameStart := 282393 },
  { event := event282411
    frameStart := 282393 },
  { event := event282412
    frameStart := 282393 },
  { event := event282413
    frameStart := 282393 },
  { event := event282414
    frameStart := 282393 },
  { event := event282415
    frameStart := 282393 }
]

def eventLeaf17651 : Array AnnotatedEvent := #[
  { event := event282416
    frameStart := 282393 },
  { event := event282417
    frameStart := 282393 },
  { event := event282418
    frameStart := 282393 },
  { event := event282419
    frameStart := 282393 },
  { event := event282420
    frameStart := 282393 },
  { event := event282421
    frameStart := 282393 },
  { event := event282422
    frameStart := 282393 },
  { event := event282423
    frameStart := 282393 },
  { event := event282424
    frameStart := 282393 },
  { event := event282425
    frameStart := 282393 },
  { event := event282426
    frameStart := 282393 },
  { event := event282427
    frameStart := 282393 },
  { event := event282428
    frameStart := 282393 },
  { event := event282429
    frameStart := 282393 },
  { event := event282430
    frameStart := 282393 },
  { event := event282431
    frameStart := 282393 }
]

def eventLeaf17652 : Array AnnotatedEvent := #[
  { event := event282432
    frameStart := 282393 },
  { event := event282433
    frameStart := 282393 },
  { event := event282434
    frameStart := 282393 },
  { event := event282435
    frameStart := 282393 },
  { event := event282436
    frameStart := 282393 },
  { event := event282437
    frameStart := 282393 },
  { event := event282438
    frameStart := 282393 },
  { event := event282439
    frameStart := 282393 },
  { event := event282440
    frameStart := 282393 },
  { event := event282441
    frameStart := 282393 },
  { event := event282442
    frameStart := 282393 },
  { event := event282443
    frameStart := 282393 },
  { event := event282444
    frameStart := 282393 },
  { event := event282445
    frameStart := 282393 },
  { event := event282446
    frameStart := 282393 },
  { event := event282447
    frameStart := 282447 }
]

def eventLeaf17653 : Array AnnotatedEvent := #[
  { event := event282448
    frameStart := 282447 },
  { event := event282449
    frameStart := 282447 },
  { event := event282450
    frameStart := 282447 },
  { event := event282451
    frameStart := 282447 },
  { event := event282452
    frameStart := 282447 },
  { event := event282453
    frameStart := 282447 },
  { event := event282454
    frameStart := 282447 },
  { event := event282455
    frameStart := 282447 },
  { event := event282456
    frameStart := 282447 },
  { event := event282457
    frameStart := 282447 },
  { event := event282458
    frameStart := 282447 },
  { event := event282459
    frameStart := 282447 },
  { event := event282460
    frameStart := 282447 },
  { event := event282461
    frameStart := 282447 },
  { event := event282462
    frameStart := 282447 },
  { event := event282463
    frameStart := 282447 }
]

def eventLeaf17654 : Array AnnotatedEvent := #[
  { event := event282464
    frameStart := 282447 },
  { event := event282465
    frameStart := 282447 },
  { event := event282466
    frameStart := 282447 },
  { event := event282467
    frameStart := 282447 },
  { event := event282468
    frameStart := 282447 },
  { event := event282469
    frameStart := 282447 },
  { event := event282470
    frameStart := 282447 },
  { event := event282471
    frameStart := 282447 },
  { event := event282472
    frameStart := 282447 },
  { event := event282473
    frameStart := 282447 },
  { event := event282474
    frameStart := 282447 },
  { event := event282475
    frameStart := 282447 },
  { event := event282476
    frameStart := 282447 },
  { event := event282477
    frameStart := 282447 },
  { event := event282478
    frameStart := 282447 },
  { event := event282479
    frameStart := 282447 }
]

def eventLeaf17655 : Array AnnotatedEvent := #[
  { event := event282480
    frameStart := 282447 },
  { event := event282481
    frameStart := 282447 },
  { event := event282482
    frameStart := 282447 },
  { event := event282483
    frameStart := 282447 },
  { event := event282484
    frameStart := 282447 },
  { event := event282485
    frameStart := 282447 },
  { event := event282486
    frameStart := 282447 },
  { event := event282487
    frameStart := 282447 },
  { event := event282488
    frameStart := 282447 },
  { event := event282489
    frameStart := 282447 },
  { event := event282490
    frameStart := 282447 },
  { event := event282491
    frameStart := 282447 },
  { event := event282492
    frameStart := 282447 },
  { event := event282493
    frameStart := 282447 },
  { event := event282494
    frameStart := 282447 },
  { event := event282495
    frameStart := 282447 }
]

def eventLeaf17656 : Array AnnotatedEvent := #[
  { event := event282496
    frameStart := 282447 },
  { event := event282497
    frameStart := 282447 },
  { event := event282498
    frameStart := 282447 },
  { event := event282499
    frameStart := 282447 },
  { event := event282500
    frameStart := 282447 },
  { event := event282501
    frameStart := 282447 },
  { event := event282502
    frameStart := 282447 },
  { event := event282503
    frameStart := 282447 },
  { event := event282504
    frameStart := 282447 },
  { event := event282505
    frameStart := 282447 },
  { event := event282506
    frameStart := 282447 },
  { event := event282507
    frameStart := 282447 },
  { event := event282508
    frameStart := 282447 },
  { event := event282509
    frameStart := 282447 },
  { event := event282510
    frameStart := 282447 },
  { event := event282511
    frameStart := 282447 }
]

def eventLeaf17657 : Array AnnotatedEvent := #[
  { event := event282512
    frameStart := 282447 },
  { event := event282513
    frameStart := 282447 },
  { event := event282514
    frameStart := 282447 },
  { event := event282515
    frameStart := 282447 },
  { event := event282516
    frameStart := 282447 },
  { event := event282517
    frameStart := 282447 },
  { event := event282518
    frameStart := 282447 },
  { event := event282519
    frameStart := 282447 },
  { event := event282520
    frameStart := 282447 },
  { event := event282521
    frameStart := 282447 },
  { event := event282522
    frameStart := 282447 },
  { event := event282523
    frameStart := 282447 },
  { event := event282524
    frameStart := 282447 },
  { event := event282525
    frameStart := 282447 },
  { event := event282526
    frameStart := 282447 },
  { event := event282527
    frameStart := 282447 }
]

def eventLeaf17658 : Array AnnotatedEvent := #[
  { event := event282528
    frameStart := 282447 },
  { event := event282529
    frameStart := 282447 },
  { event := event282530
    frameStart := 282447 },
  { event := event282531
    frameStart := 282447 },
  { event := event282532
    frameStart := 282447 },
  { event := event282533
    frameStart := 282447 },
  { event := event282534
    frameStart := 282447 },
  { event := event282535
    frameStart := 282447 },
  { event := event282536
    frameStart := 282447 },
  { event := event282537
    frameStart := 282447 },
  { event := event282538
    frameStart := 282447 },
  { event := event282539
    frameStart := 282447 },
  { event := event282540
    frameStart := 282447 },
  { event := event282541
    frameStart := 282447 },
  { event := event282542
    frameStart := 282447 },
  { event := event282543
    frameStart := 282447 }
]

def eventLeaf17659 : Array AnnotatedEvent := #[
  { event := event282544
    frameStart := 282447 },
  { event := event282545
    frameStart := 282447 },
  { event := event282546
    frameStart := 282447 },
  { event := event282547
    frameStart := 282447 },
  { event := event282548
    frameStart := 282447 },
  { event := event282549
    frameStart := 282447 },
  { event := event282550
    frameStart := 282447 },
  { event := event282551
    frameStart := 0 },
  { event := event282552
    frameStart := 0 },
  { event := event282553
    frameStart := 0 },
  { event := event282554
    frameStart := 0 },
  { event := event282555
    frameStart := 0 },
  { event := event282556
    frameStart := 0 },
  { event := event282557
    frameStart := 0 },
  { event := event282558
    frameStart := 0 },
  { event := event282559
    frameStart := 0 }
]

def eventLeaf17660 : Array AnnotatedEvent := #[
  { event := event282560
    frameStart := 0 },
  { event := event282561
    frameStart := 0 },
  { event := event282562
    frameStart := 0 },
  { event := event282563
    frameStart := 0 },
  { event := event282564
    frameStart := 0 },
  { event := event282565
    frameStart := 0 },
  { event := event282566
    frameStart := 0 },
  { event := event282567
    frameStart := 0 },
  { event := event282568
    frameStart := 0 },
  { event := event282569
    frameStart := 0 },
  { event := event282570
    frameStart := 0 },
  { event := event282571
    frameStart := 0 },
  { event := event282572
    frameStart := 0 },
  { event := event282573
    frameStart := 0 },
  { event := event282574
    frameStart := 0 },
  { event := event282575
    frameStart := 0 }
]

def eventLeaf17661 : Array AnnotatedEvent := #[
  { event := event282576
    frameStart := 0 },
  { event := event282577
    frameStart := 0 },
  { event := event282578
    frameStart := 0 },
  { event := event282579
    frameStart := 0 },
  { event := event282580
    frameStart := 0 },
  { event := event282581
    frameStart := 0 },
  { event := event282582
    frameStart := 0 },
  { event := event282583
    frameStart := 0 },
  { event := event282584
    frameStart := 0 },
  { event := event282585
    frameStart := 0 },
  { event := event282586
    frameStart := 0 },
  { event := event282587
    frameStart := 0 },
  { event := event282588
    frameStart := 0 },
  { event := event282589
    frameStart := 0 },
  { event := event282590
    frameStart := 0 },
  { event := event282591
    frameStart := 0 }
]

def eventLeaf17662 : Array AnnotatedEvent := #[
  { event := event282592
    frameStart := 0 },
  { event := event282593
    frameStart := 0 },
  { event := event282594
    frameStart := 0 },
  { event := event282595
    frameStart := 0 },
  { event := event282596
    frameStart := 0 },
  { event := event282597
    frameStart := 0 },
  { event := event282598
    frameStart := 0 },
  { event := event282599
    frameStart := 0 },
  { event := event282600
    frameStart := 0 },
  { event := event282601
    frameStart := 0 },
  { event := event282602
    frameStart := 0 },
  { event := event282603
    frameStart := 0 },
  { event := event282604
    frameStart := 0 },
  { event := event282605
    frameStart := 0 },
  { event := event282606
    frameStart := 0 },
  { event := event282607
    frameStart := 0 }
]

def eventLeaf17663 : Array AnnotatedEvent := #[
  { event := event282608
    frameStart := 0 },
  { event := event282609
    frameStart := 0 },
  { event := event282610
    frameStart := 0 },
  { event := event282611
    frameStart := 0 },
  { event := event282612
    frameStart := 0 },
  { event := event282613
    frameStart := 0 },
  { event := event282614
    frameStart := 0 },
  { event := event282615
    frameStart := 0 },
  { event := event282616
    frameStart := 0 },
  { event := event282617
    frameStart := 0 },
  { event := event282618
    frameStart := 0 },
  { event := event282619
    frameStart := 0 },
  { event := event282620
    frameStart := 0 },
  { event := event282621
    frameStart := 0 },
  { event := event282622
    frameStart := 0 },
  { event := event282623
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1103
