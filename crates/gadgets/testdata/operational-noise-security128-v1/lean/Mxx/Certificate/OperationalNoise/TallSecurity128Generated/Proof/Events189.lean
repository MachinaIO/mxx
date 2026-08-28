import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events189

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event48384 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42191⟩⟩, .operator (⟨48377, 1⟩, ⟨48100, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42189⟩⟩]⟩, (-1)⟩)

def event48385 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨42191⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42189⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42189⟩⟩) ⟨41333⟩ 48097)

def event48386 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42191⟩⟩, .relation 48385 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨41333⟩⟩]⟩, (-1)⟩)

def exact48387RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42189⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨41333⟩⟩]⟩, (-1)⟩]

theorem exact48387RawTermsValid :
    exact48387RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48387 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42191⟩⟩) exact48387RawTerms .large 48380 (.finite 32193129122288627115968346193920) (some (48382))

def event48388 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41016⟩⟩) 0 ⟨40173⟩ 1676

def event48389 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41016⟩⟩) (.authority (.relationPreimageSource ⟨87⟩))

def exact48390RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41016⟩⟩]⟩, (1)⟩]

theorem exact48390RawTermsValid :
    exact48390RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48390 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41016⟩⟩) exact48390RawTerms (.finite 5647228698) 48389 .exactZero (none)

def event48391 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41018⟩⟩) 0 ⟨41016⟩ 48390

def event48392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41018⟩⟩) 1 ⟨2370⟩ 4

def event48393 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41018⟩⟩) (.scale (.predecessor 0 48391 .coefficient) (.value (.predecessor 1 48392 .coefficient)))

def exact48394RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41016⟩⟩]⟩, (1)⟩]

theorem exact48394RawTermsValid :
    exact48394RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48394 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41018⟩⟩) exact48394RawTerms (.finite 5647228698) 48393 .exactZero (none)

def event48395 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41019⟩⟩) 0 ⟨11216⟩ 46745

def event48396 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41019⟩⟩) 1 ⟨41018⟩ 48394

def event48397 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41019⟩⟩) (.product (.predecessor 0 48395 .coefficient) (.predecessor 1 48396 .coefficient) (⟨false, false, none, none, none⟩))

def event48398 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41019⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨41016⟩⟩]⟩) [⟨.result 48390 .coefficient, false, none⟩])

def event48399 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41019⟩⟩) (.product (.result 46745 .summary) (.transfer 48398) (⟨false, false, none, none, none⟩))

def event48400 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41019⟩⟩, .operator (⟨46745, 0⟩, ⟨48394, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41016⟩⟩]⟩, (1)⟩)

def event48401 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨41017⟩⟩)

def event48402 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event48403 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event48404 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event48405 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event48406 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event48407 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event48408 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event48409 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event48410 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 48409

def event48411 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 48407

def event48412 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 48410 .coefficient) (.value (.predecessor 1 48411 .coefficient)))

def event48413 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event48414 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 48413

def event48415 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 48405

def event48416 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 48414 .coefficient, .predecessor 1 48415 .coefficient])

def event48417 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event48418 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 48417

def event48419 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 48403

def event48420 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 48419 .coefficient))

def event48421 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event48422 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39986⟩⟩) 0 ⟨11173⟩ 48421

def event48423 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39986⟩⟩) (.authority (.programFamilyFact))

def exact48424RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39986⟩⟩], []⟩, (1)⟩]

theorem exact48424RawTermsValid :
    exact48424RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48424 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39986⟩⟩) exact48424RawTerms (.finite 46) 48423 .exactZero (none)

def event48425 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14301⟩⟩) 0 ⟨11173⟩ 48421

def event48426 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14301⟩⟩) (.authority (.programFamilyFact))

def exact48427RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14301⟩⟩], []⟩, (1)⟩]

theorem exact48427RawTermsValid :
    exact48427RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48427 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14301⟩⟩) exact48427RawTerms (.finite 46) 48426 .exactZero (none)

def event48428 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39987⟩⟩) 0 ⟨14301⟩ 48427

def event48429 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39987⟩⟩) 1 ⟨39986⟩ 48424

def event48430 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39987⟩⟩) (.product (.predecessor 0 48428 .coefficient) (.predecessor 1 48429 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event48431 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39987⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], []⟩) [⟨.result 48427 .coefficient, true, some 1⟩, ⟨.result 48424 .coefficient, true, some 1⟩])

def event48432 : Event := .survivorFold (1) 48431

def exact48433RawTerms : List Term := []

theorem exact48433RawTermsValid :
    exact48433RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48433 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39987⟩⟩) exact48433RawTerms (.finite 2116) 48430 (.finite 2116) (some (48431))

def event48434 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39988⟩⟩) 0 ⟨39987⟩ 48433

def event48435 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39988⟩⟩) (.identity (.predecessor 0 48434 .coefficient))

def event48436 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39988⟩⟩) (.finite 2116)

def event48437 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40172⟩⟩) 0 ⟨39988⟩ 48436

def event48438 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40172⟩⟩) (.authority (.programFamilyFact))

def exact48439RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40172⟩⟩], []⟩, (1)⟩]

theorem exact48439RawTermsValid :
    exact48439RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48439 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40172⟩⟩) exact48439RawTerms (.finite 46) 48438 .exactZero (none)

def event48440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40173⟩⟩) 0 ⟨40172⟩ 48439

def event48441 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40173⟩⟩) (.identity (.predecessor 0 48440 .coefficient))

def event48442 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40173⟩⟩) (.finite 46)

def event48443 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41016⟩⟩) 0 ⟨40173⟩ 48442

def event48444 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41016⟩⟩) (.authority (.relationPreimageSource ⟨87⟩))

def exact48445RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41016⟩⟩]⟩, (1)⟩]

theorem exact48445RawTermsValid :
    exact48445RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48445 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41016⟩⟩) exact48445RawTerms (.finite 5647228698) 48444 .exactZero (none)

def event48446 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact48447RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact48447RawTermsValid :
    exact48447RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48447 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact48447RawTerms .large 48446 .exactZero (none)

def event48448 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41017⟩⟩) 0 ⟨35⟩ 48447

def event48449 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41017⟩⟩) 1 ⟨41016⟩ 48445

def event48450 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41017⟩⟩) (.product (.predecessor 0 48448 .coefficient) (.predecessor 1 48449 .coefficient) (⟨false, false, none, none, none⟩))

def event48451 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41017⟩⟩, .operator (⟨48447, 0⟩, ⟨48445, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41016⟩⟩]⟩, (1)⟩)

def exact48452RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41016⟩⟩]⟩, (1)⟩]

theorem exact48452RawTermsValid :
    exact48452RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48452 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41017⟩⟩) exact48452RawTerms .large 48450 .exactZero (none)

def event48453 : Event := .preFoldPolynomial 48452 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41016⟩⟩]⟩, (1)⟩] .exactZero none

def exact48454RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41016⟩⟩]⟩, (1)⟩]

def event48454 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨41017⟩⟩) 48453 exact48454RawTerms .large 48450 .exactZero (none)

def event48455 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨42193⟩⟩)

def event48456 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event48457 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event48458 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event48459 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event48460 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event48461 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event48462 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event48463 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event48464 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 48463

def event48465 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 48461

def event48466 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 48464 .coefficient) (.value (.predecessor 1 48465 .coefficient)))

def event48467 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event48468 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 48467

def event48469 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 48459

def event48470 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 48468 .coefficient, .predecessor 1 48469 .coefficient])

def event48471 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event48472 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 48471

def event48473 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 48457

def event48474 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 48473 .coefficient))

def event48475 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event48476 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39986⟩⟩) 0 ⟨11173⟩ 48475

def event48477 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39986⟩⟩) (.authority (.programFamilyFact))

def exact48478RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39986⟩⟩], []⟩, (1)⟩]

theorem exact48478RawTermsValid :
    exact48478RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48478 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39986⟩⟩) exact48478RawTerms (.finite 46) 48477 .exactZero (none)

def event48479 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14301⟩⟩) 0 ⟨11173⟩ 48475

def event48480 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14301⟩⟩) (.authority (.programFamilyFact))

def exact48481RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14301⟩⟩], []⟩, (1)⟩]

theorem exact48481RawTermsValid :
    exact48481RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48481 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14301⟩⟩) exact48481RawTerms (.finite 46) 48480 .exactZero (none)

def event48482 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39987⟩⟩) 0 ⟨14301⟩ 48481

def event48483 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39987⟩⟩) 1 ⟨39986⟩ 48478

def event48484 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39987⟩⟩) (.product (.predecessor 0 48482 .coefficient) (.predecessor 1 48483 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event48485 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39987⟩⟩, .operator (⟨48481, 0⟩, ⟨48478, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], []⟩, (1)⟩)

def exact48486RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14301⟩⟩, ⟨.program ⟨257⟩, ⟨39986⟩⟩], []⟩, (1)⟩]

theorem exact48486RawTermsValid :
    exact48486RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48486 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39987⟩⟩) exact48486RawTerms (.finite 2116) 48484 .exactZero (none)

def event48487 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39988⟩⟩) 0 ⟨39987⟩ 48486

def event48488 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39988⟩⟩) (.identity (.predecessor 0 48487 .coefficient))

def event48489 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39988⟩⟩) (.finite 2116)

def event48490 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40172⟩⟩) 0 ⟨39988⟩ 48489

def event48491 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40172⟩⟩) (.authority (.programFamilyFact))

def exact48492RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40172⟩⟩], []⟩, (1)⟩]

theorem exact48492RawTermsValid :
    exact48492RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48492 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40172⟩⟩) exact48492RawTerms (.finite 46) 48491 .exactZero (none)

def event48493 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40173⟩⟩) 0 ⟨40172⟩ 48492

def event48494 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40173⟩⟩) (.identity (.predecessor 0 48493 .coefficient))

def event48495 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40173⟩⟩) (.finite 46)

def event48496 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41331⟩⟩) 0 ⟨40173⟩ 48495

def event48497 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41331⟩⟩) (.authority (.programFamilyFact))

def event48498 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41331⟩⟩) (.finite 3720)

def event48499 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event48500 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41333⟩⟩) 0 ⟨7177⟩ 48499

def event48501 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41333⟩⟩) 1 ⟨41331⟩ 48498

def event48502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41333⟩⟩) (.authority (.operator))

def exact48503RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41333⟩⟩]⟩, (1)⟩]

theorem exact48503RawTermsValid :
    exact48503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48503 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41333⟩⟩) exact48503RawTerms .large 48502 .exactZero (none)

def event48504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42189⟩⟩) 0 ⟨41333⟩ 48503

def event48505 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42189⟩⟩) (.authority (.operator))

def exact48506RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨42189⟩⟩]⟩, (1)⟩]

theorem exact48506RawTermsValid :
    exact48506RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48506 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42189⟩⟩) exact48506RawTerms (.finite 8192) 48505 .exactZero (none)

def event48507 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event48508 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event48509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41498⟩⟩) 0 ⟨40173⟩ 48495

def event48510 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41498⟩⟩) 1 ⟨136⟩ 48508

def event48511 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41498⟩⟩) (.sum [.predecessor 0 48509 .coefficient, .predecessor 1 48510 .coefficient])

def event48512 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41498⟩⟩) (.finite 46)

def event48513 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41499⟩⟩) 0 ⟨41498⟩ 48512

def event48514 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41499⟩⟩) (.identity (.predecessor 0 48513 .coefficient))

def exact48515RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40172⟩⟩], []⟩, (1)⟩]

theorem exact48515RawTermsValid :
    exact48515RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48515 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41499⟩⟩) exact48515RawTerms (.finite 46) 48514 .exactZero (none)

def event48516 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact48517RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact48517RawTermsValid :
    exact48517RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48517 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact48517RawTerms .large 48516 .exactZero (none)

def event48518 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41500⟩⟩) 0 ⟨6908⟩ 48517

def event48519 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41500⟩⟩) 1 ⟨41499⟩ 48515

def event48520 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41500⟩⟩) (.product (.predecessor 0 48518 .coefficient) (.predecessor 1 48519 .coefficient) (⟨false, false, none, none, none⟩))

def event48521 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41500⟩⟩, .operator (⟨48517, 0⟩, ⟨48515, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact48522RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact48522RawTermsValid :
    exact48522RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48522 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41500⟩⟩) exact48522RawTerms .large 48520 .exactZero (none)

def event48523 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7193⟩⟩) 0 ⟨7177⟩ 48499

def event48524 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7193⟩⟩) (.authority (.operator))

def exact48525RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩]

theorem exact48525RawTermsValid :
    exact48525RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48525 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7193⟩⟩) exact48525RawTerms .large 48524 .exactZero (none)

def event48526 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41501⟩⟩) 0 ⟨7193⟩ 48525

def event48527 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41501⟩⟩) 1 ⟨41500⟩ 48522

def event48528 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41501⟩⟩) (.sum [.predecessor 0 48526 .coefficient, .predecessor 1 48527 .coefficient])

def exact48529RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48529RawTermsValid :
    exact48529RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48529 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41501⟩⟩) exact48529RawTerms .large 48528 .exactZero (none)

def event48530 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42190⟩⟩) 0 ⟨41501⟩ 48529

def event48531 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42190⟩⟩) 1 ⟨42189⟩ 48506

def event48532 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42190⟩⟩) (.product (.predecessor 0 48530 .coefficient) (.predecessor 1 48531 .coefficient) (⟨false, false, none, none, none⟩))

def event48533 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42190⟩⟩, .operator (⟨48529, 0⟩, ⟨48506, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42189⟩⟩]⟩, (1)⟩)

def event48534 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42190⟩⟩, .operator (⟨48529, 1⟩, ⟨48506, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42189⟩⟩]⟩, (-1)⟩)

def event48535 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨42190⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42189⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42189⟩⟩) ⟨41333⟩ 48503)

def event48536 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42190⟩⟩, .relation 48535 0, ⟨[⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨41333⟩⟩]⟩, (-1)⟩)

def exact48537RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42189⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨41333⟩⟩]⟩, (-1)⟩]

theorem exact48537RawTermsValid :
    exact48537RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48537 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42190⟩⟩) exact48537RawTerms .large 48532 .exactZero (none)

def event48538 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40423⟩⟩) 0 ⟨40173⟩ 48495

def event48539 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40423⟩⟩) (.authority (.programFamilyFact))

def exact48540RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40423⟩⟩], []⟩, (1)⟩]

theorem exact48540RawTermsValid :
    exact48540RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48540 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40423⟩⟩) exact48540RawTerms (.finite 63) 48539 .exactZero (none)

def event48541 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40424⟩⟩) 0 ⟨6908⟩ 48517

def event48542 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40424⟩⟩) 1 ⟨40423⟩ 48540

def event48543 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40424⟩⟩) (.product (.predecessor 0 48541 .coefficient) (.predecessor 1 48542 .coefficient) (⟨false, true, none, none, some 1⟩))

def event48544 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40424⟩⟩, .operator (⟨48517, 0⟩, ⟨48540, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact48545RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact48545RawTermsValid :
    exact48545RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48545 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40424⟩⟩) exact48545RawTerms .large 48543 .exactZero (none)

def event48546 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7226⟩⟩) 0 ⟨7177⟩ 48499

def event48547 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7226⟩⟩) (.authority (.operator))

def exact48548RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact48548RawTermsValid :
    exact48548RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48548 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7226⟩⟩) exact48548RawTerms .large 48547 .exactZero (none)

def event48549 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40425⟩⟩) 0 ⟨7226⟩ 48548

def event48550 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40425⟩⟩) 1 ⟨40424⟩ 48545

def event48551 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40425⟩⟩) (.sum [.predecessor 0 48549 .coefficient, .predecessor 1 48550 .coefficient])

def exact48552RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48552RawTermsValid :
    exact48552RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48552 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40425⟩⟩) exact48552RawTerms .large 48551 .exactZero (none)

def event48553 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42193⟩⟩) 0 ⟨40425⟩ 48552

def event48554 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42193⟩⟩) 1 ⟨42190⟩ 48537

def event48555 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42193⟩⟩) (.sum [.predecessor 0 48553 .coefficient, .predecessor 1 48554 .coefficient])

def exact48556RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42189⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨41333⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48556RawTermsValid :
    exact48556RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48556 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42193⟩⟩) exact48556RawTerms .large 48555 .exactZero (none)

def event48557 : Event := .preFoldPolynomial 48556 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42189⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨41333⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact48558RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42189⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨41333⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event48558 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨42193⟩⟩) 48557 exact48558RawTerms .large 48555 .exactZero (none)

def event48559 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨40173⟩⟩) ⟨⟨105⟩, ⟨87⟩, ⟨135⟩⟩ ⟨48401, 48559⟩

def event48560 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41019⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41016⟩⟩]⟩) (1) 0 2 (.universal 48559 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨41016⟩⟩]⟩) (none) 48558)

def event48561 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41019⟩⟩, .relation 48560 1, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩)

def event48562 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41019⟩⟩, .relation 48560 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42189⟩⟩]⟩, (-1)⟩)

def event48563 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41019⟩⟩, .relation 48560 2, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨41333⟩⟩]⟩, (1)⟩)

def event48564 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41019⟩⟩, .relation 48560 3, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact48565RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42189⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨41333⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48565RawTermsValid :
    exact48565RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48565 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41019⟩⟩) exact48565RawTerms .large 48397 (.finite 202072841853861888) (some (48399))

def event48566 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42192⟩⟩) 0 ⟨41019⟩ 48565

def event48567 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42192⟩⟩) 1 ⟨42191⟩ 48387

def event48568 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42192⟩⟩) (.sum [.predecessor 0 48566 .coefficient, .predecessor 1 48567 .coefficient])

def event48569 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42192⟩⟩, .operator (⟨48565, 0⟩, ⟨48387, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42189⟩⟩]⟩, (1)⟩)

def event48570 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42192⟩⟩, .operator (⟨48565, 2⟩, ⟨48387, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40172⟩⟩], [⟨.program ⟨257⟩, ⟨41333⟩⟩]⟩, (-1)⟩)

def event48571 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42192⟩⟩) (.sum [.result 48565 .summary, .result 48387 .summary])

def exact48572RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48572RawTermsValid :
    exact48572RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48572 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42192⟩⟩) exact48572RawTerms .large 48568 (.finite 32193129122288829188810200055808) (some (48571))

def event48573 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38651⟩⟩) 0 ⟨37493⟩ 1699

def event48574 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38651⟩⟩) (.authority (.programFamilyFact))

def event48575 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38651⟩⟩) (.finite 3720)

def event48576 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38653⟩⟩) 0 ⟨7177⟩ 15500

def event48577 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38653⟩⟩) 1 ⟨38651⟩ 48575

def event48578 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38653⟩⟩) (.authority (.operator))

def exact48579RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38653⟩⟩]⟩, (1)⟩]

theorem exact48579RawTermsValid :
    exact48579RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48579 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38653⟩⟩) exact48579RawTerms .large 48578 .exactZero (none)

def event48580 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39509⟩⟩) 0 ⟨38653⟩ 48579

def event48581 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39509⟩⟩) (.authority (.operator))

def exact48582RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39509⟩⟩]⟩, (1)⟩]

theorem exact48582RawTermsValid :
    exact48582RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48582 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39509⟩⟩) exact48582RawTerms (.finite 8192) 48581 .exactZero (none)

def event48583 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38476⟩⟩) 0 ⟨37308⟩ 1693

def event48584 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38476⟩⟩) (.authority (.programFamilyFact))

def event48585 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38476⟩⟩) (.finite 3720)

def event48586 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38477⟩⟩) 0 ⟨7177⟩ 15500

def event48587 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38477⟩⟩) 1 ⟨38476⟩ 48585

def event48588 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38477⟩⟩) (.authority (.operator))

def exact48589RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38477⟩⟩]⟩, (1)⟩]

theorem exact48589RawTermsValid :
    exact48589RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48589 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38477⟩⟩) exact48589RawTerms .large 48588 .exactZero (none)

def event48590 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39027⟩⟩) 0 ⟨38477⟩ 48589

def event48591 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39027⟩⟩) (.authority (.operator))

def exact48592RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39027⟩⟩]⟩, (1)⟩]

theorem exact48592RawTermsValid :
    exact48592RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48592 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39027⟩⟩) exact48592RawTerms (.finite 8192) 48591 .exactZero (none)

def event48593 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37309⟩⟩) 0 ⟨37306⟩ 1682

def event48594 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37309⟩⟩) 1 ⟨11176⟩ 46653

def event48595 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37309⟩⟩) (.tensor (.predecessor 0 48593 .coefficient) (.predecessor 1 48594 .coefficient) true false)

def event48596 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37309⟩⟩, .operator (⟨1682, 0⟩, ⟨46653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact48597RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact48597RawTermsValid :
    exact48597RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48597 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37309⟩⟩) exact48597RawTerms .large 48595 .exactZero (none)

def event48598 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11187⟩⟩) 0 ⟨11175⟩ 46523

def event48599 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11187⟩⟩) 1 ⟨7281⟩ 19084

def event48600 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11187⟩⟩) (.product (.predecessor 0 48598 .coefficient) (.predecessor 1 48599 .coefficient) (⟨false, false, none, none, none⟩))

def event48601 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11187⟩⟩, .operator (⟨46523, 0⟩, ⟨19084, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩)

def exact48602RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩]

theorem exact48602RawTermsValid :
    exact48602RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48602 : Event := .resultExact (⟨.program ⟨257⟩, ⟨11187⟩⟩) exact48602RawTerms .large 48600 .exactZero (none)

def event48603 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37310⟩⟩) 0 ⟨11187⟩ 48602

def event48604 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37310⟩⟩) 1 ⟨37309⟩ 48597

def event48605 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37310⟩⟩) (.sum [.predecessor 0 48603 .coefficient, .predecessor 1 48604 .coefficient])

def exact48606RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48606RawTermsValid :
    exact48606RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48606 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37310⟩⟩) exact48606RawTerms .large 48605 .exactZero (none)

def event48607 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37311⟩⟩) 0 ⟨37310⟩ 48606

def event48608 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37311⟩⟩) 1 ⟨107⟩ 19076

def event48609 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37311⟩⟩) (.sum [.predecessor 0 48607 .coefficient, .predecessor 1 48608 .coefficient])

def event48610 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37311⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨107⟩⟩]⟩) [⟨.result 19076 .coefficient, false, none⟩])

def event48611 : Event := .survivorFold (1) 48610

def exact48612RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48612RawTermsValid :
    exact48612RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48612 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37311⟩⟩) exact48612RawTerms .large 48609 (.finite 26) (some (48610))

def event48613 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37312⟩⟩) 0 ⟨37311⟩ 48612

def event48614 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37312⟩⟩) 1 ⟨14001⟩ 1685

def event48615 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37312⟩⟩) (.product (.predecessor 0 48613 .coefficient) (.predecessor 1 48614 .coefficient) (⟨false, true, none, none, some 1⟩))

def event48616 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37312⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14001⟩⟩], []⟩) [⟨.result 1685 .coefficient, true, some 1⟩])

def event48617 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37312⟩⟩) (.product (.result 48612 .summary) (.transfer 48616) (⟨false, false, none, none, none⟩))

def event48618 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37312⟩⟩, .operator (⟨48612, 1⟩, ⟨1685, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event48619 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37312⟩⟩, .operator (⟨48612, 0⟩, ⟨1685, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩)

def exact48620RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩, ⟨.program ⟨257⟩, ⟨37306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48620RawTermsValid :
    exact48620RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48620 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37312⟩⟩) exact48620RawTerms .large 48615 (.finite 35782656) (some (48617))

def event48621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14002⟩⟩) 0 ⟨14001⟩ 1685

def event48622 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14002⟩⟩) 1 ⟨11176⟩ 46653

def event48623 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14002⟩⟩) (.tensor (.predecessor 0 48621 .coefficient) (.predecessor 1 48622 .coefficient) true false)

def event48624 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14002⟩⟩, .operator (⟨1685, 0⟩, ⟨46653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact48625RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact48625RawTermsValid :
    exact48625RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48625 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14002⟩⟩) exact48625RawTerms .large 48623 .exactZero (none)

def event48626 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11204⟩⟩) 0 ⟨11175⟩ 46523

def event48627 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11204⟩⟩) 1 ⟨7298⟩ 19125

def event48628 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11204⟩⟩) (.product (.predecessor 0 48626 .coefficient) (.predecessor 1 48627 .coefficient) (⟨false, false, none, none, none⟩))

def event48629 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11204⟩⟩, .operator (⟨46523, 0⟩, ⟨19125, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩)

def exact48630RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩]

theorem exact48630RawTermsValid :
    exact48630RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48630 : Event := .resultExact (⟨.program ⟨257⟩, ⟨11204⟩⟩) exact48630RawTerms .large 48628 .exactZero (none)

def event48631 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14003⟩⟩) 0 ⟨11204⟩ 48630

def event48632 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14003⟩⟩) 1 ⟨14002⟩ 48625

def event48633 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14003⟩⟩) (.sum [.predecessor 0 48631 .coefficient, .predecessor 1 48632 .coefficient])

def exact48634RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14001⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact48634RawTermsValid :
    exact48634RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event48634 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14003⟩⟩) exact48634RawTerms .large 48633 .exactZero (none)

def event48635 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14004⟩⟩) 0 ⟨14003⟩ 48634

def event48636 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14004⟩⟩) 1 ⟨124⟩ 19117

def event48637 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14004⟩⟩) (.sum [.predecessor 0 48635 .coefficient, .predecessor 1 48636 .coefficient])

def event48638 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14004⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨124⟩⟩]⟩) [⟨.result 19117 .coefficient, false, none⟩])

def event48639 : Event := .survivorFold (1) 48638

def eventLeaf3024 : Array AnnotatedEvent := #[
  { event := event48384
    frameStart := 0 },
  { event := event48385
    frameStart := 0 },
  { event := event48386
    frameStart := 0 },
  { event := event48387
    frameStart := 0 },
  { event := event48388
    frameStart := 0 },
  { event := event48389
    frameStart := 0 },
  { event := event48390
    frameStart := 0 },
  { event := event48391
    frameStart := 0 },
  { event := event48392
    frameStart := 0 },
  { event := event48393
    frameStart := 0 },
  { event := event48394
    frameStart := 0 },
  { event := event48395
    frameStart := 0 },
  { event := event48396
    frameStart := 0 },
  { event := event48397
    frameStart := 0 },
  { event := event48398
    frameStart := 0 },
  { event := event48399
    frameStart := 0 }
]

def eventLeaf3025 : Array AnnotatedEvent := #[
  { event := event48400
    frameStart := 0 },
  { event := event48401
    frameStart := 48401 },
  { event := event48402
    frameStart := 48401 },
  { event := event48403
    frameStart := 48401 },
  { event := event48404
    frameStart := 48401 },
  { event := event48405
    frameStart := 48401 },
  { event := event48406
    frameStart := 48401 },
  { event := event48407
    frameStart := 48401 },
  { event := event48408
    frameStart := 48401 },
  { event := event48409
    frameStart := 48401 },
  { event := event48410
    frameStart := 48401 },
  { event := event48411
    frameStart := 48401 },
  { event := event48412
    frameStart := 48401 },
  { event := event48413
    frameStart := 48401 },
  { event := event48414
    frameStart := 48401 },
  { event := event48415
    frameStart := 48401 }
]

def eventLeaf3026 : Array AnnotatedEvent := #[
  { event := event48416
    frameStart := 48401 },
  { event := event48417
    frameStart := 48401 },
  { event := event48418
    frameStart := 48401 },
  { event := event48419
    frameStart := 48401 },
  { event := event48420
    frameStart := 48401 },
  { event := event48421
    frameStart := 48401 },
  { event := event48422
    frameStart := 48401 },
  { event := event48423
    frameStart := 48401 },
  { event := event48424
    frameStart := 48401 },
  { event := event48425
    frameStart := 48401 },
  { event := event48426
    frameStart := 48401 },
  { event := event48427
    frameStart := 48401 },
  { event := event48428
    frameStart := 48401 },
  { event := event48429
    frameStart := 48401 },
  { event := event48430
    frameStart := 48401 },
  { event := event48431
    frameStart := 48401 }
]

def eventLeaf3027 : Array AnnotatedEvent := #[
  { event := event48432
    frameStart := 48401 },
  { event := event48433
    frameStart := 48401 },
  { event := event48434
    frameStart := 48401 },
  { event := event48435
    frameStart := 48401 },
  { event := event48436
    frameStart := 48401 },
  { event := event48437
    frameStart := 48401 },
  { event := event48438
    frameStart := 48401 },
  { event := event48439
    frameStart := 48401 },
  { event := event48440
    frameStart := 48401 },
  { event := event48441
    frameStart := 48401 },
  { event := event48442
    frameStart := 48401 },
  { event := event48443
    frameStart := 48401 },
  { event := event48444
    frameStart := 48401 },
  { event := event48445
    frameStart := 48401 },
  { event := event48446
    frameStart := 48401 },
  { event := event48447
    frameStart := 48401 }
]

def eventLeaf3028 : Array AnnotatedEvent := #[
  { event := event48448
    frameStart := 48401 },
  { event := event48449
    frameStart := 48401 },
  { event := event48450
    frameStart := 48401 },
  { event := event48451
    frameStart := 48401 },
  { event := event48452
    frameStart := 48401 },
  { event := event48453
    frameStart := 48401 },
  { event := event48454
    frameStart := 48401 },
  { event := event48455
    frameStart := 48455 },
  { event := event48456
    frameStart := 48455 },
  { event := event48457
    frameStart := 48455 },
  { event := event48458
    frameStart := 48455 },
  { event := event48459
    frameStart := 48455 },
  { event := event48460
    frameStart := 48455 },
  { event := event48461
    frameStart := 48455 },
  { event := event48462
    frameStart := 48455 },
  { event := event48463
    frameStart := 48455 }
]

def eventLeaf3029 : Array AnnotatedEvent := #[
  { event := event48464
    frameStart := 48455 },
  { event := event48465
    frameStart := 48455 },
  { event := event48466
    frameStart := 48455 },
  { event := event48467
    frameStart := 48455 },
  { event := event48468
    frameStart := 48455 },
  { event := event48469
    frameStart := 48455 },
  { event := event48470
    frameStart := 48455 },
  { event := event48471
    frameStart := 48455 },
  { event := event48472
    frameStart := 48455 },
  { event := event48473
    frameStart := 48455 },
  { event := event48474
    frameStart := 48455 },
  { event := event48475
    frameStart := 48455 },
  { event := event48476
    frameStart := 48455 },
  { event := event48477
    frameStart := 48455 },
  { event := event48478
    frameStart := 48455 },
  { event := event48479
    frameStart := 48455 }
]

def eventLeaf3030 : Array AnnotatedEvent := #[
  { event := event48480
    frameStart := 48455 },
  { event := event48481
    frameStart := 48455 },
  { event := event48482
    frameStart := 48455 },
  { event := event48483
    frameStart := 48455 },
  { event := event48484
    frameStart := 48455 },
  { event := event48485
    frameStart := 48455 },
  { event := event48486
    frameStart := 48455 },
  { event := event48487
    frameStart := 48455 },
  { event := event48488
    frameStart := 48455 },
  { event := event48489
    frameStart := 48455 },
  { event := event48490
    frameStart := 48455 },
  { event := event48491
    frameStart := 48455 },
  { event := event48492
    frameStart := 48455 },
  { event := event48493
    frameStart := 48455 },
  { event := event48494
    frameStart := 48455 },
  { event := event48495
    frameStart := 48455 }
]

def eventLeaf3031 : Array AnnotatedEvent := #[
  { event := event48496
    frameStart := 48455 },
  { event := event48497
    frameStart := 48455 },
  { event := event48498
    frameStart := 48455 },
  { event := event48499
    frameStart := 48455 },
  { event := event48500
    frameStart := 48455 },
  { event := event48501
    frameStart := 48455 },
  { event := event48502
    frameStart := 48455 },
  { event := event48503
    frameStart := 48455 },
  { event := event48504
    frameStart := 48455 },
  { event := event48505
    frameStart := 48455 },
  { event := event48506
    frameStart := 48455 },
  { event := event48507
    frameStart := 48455 },
  { event := event48508
    frameStart := 48455 },
  { event := event48509
    frameStart := 48455 },
  { event := event48510
    frameStart := 48455 },
  { event := event48511
    frameStart := 48455 }
]

def eventLeaf3032 : Array AnnotatedEvent := #[
  { event := event48512
    frameStart := 48455 },
  { event := event48513
    frameStart := 48455 },
  { event := event48514
    frameStart := 48455 },
  { event := event48515
    frameStart := 48455 },
  { event := event48516
    frameStart := 48455 },
  { event := event48517
    frameStart := 48455 },
  { event := event48518
    frameStart := 48455 },
  { event := event48519
    frameStart := 48455 },
  { event := event48520
    frameStart := 48455 },
  { event := event48521
    frameStart := 48455 },
  { event := event48522
    frameStart := 48455 },
  { event := event48523
    frameStart := 48455 },
  { event := event48524
    frameStart := 48455 },
  { event := event48525
    frameStart := 48455 },
  { event := event48526
    frameStart := 48455 },
  { event := event48527
    frameStart := 48455 }
]

def eventLeaf3033 : Array AnnotatedEvent := #[
  { event := event48528
    frameStart := 48455 },
  { event := event48529
    frameStart := 48455 },
  { event := event48530
    frameStart := 48455 },
  { event := event48531
    frameStart := 48455 },
  { event := event48532
    frameStart := 48455 },
  { event := event48533
    frameStart := 48455 },
  { event := event48534
    frameStart := 48455 },
  { event := event48535
    frameStart := 48455 },
  { event := event48536
    frameStart := 48455 },
  { event := event48537
    frameStart := 48455 },
  { event := event48538
    frameStart := 48455 },
  { event := event48539
    frameStart := 48455 },
  { event := event48540
    frameStart := 48455 },
  { event := event48541
    frameStart := 48455 },
  { event := event48542
    frameStart := 48455 },
  { event := event48543
    frameStart := 48455 }
]

def eventLeaf3034 : Array AnnotatedEvent := #[
  { event := event48544
    frameStart := 48455 },
  { event := event48545
    frameStart := 48455 },
  { event := event48546
    frameStart := 48455 },
  { event := event48547
    frameStart := 48455 },
  { event := event48548
    frameStart := 48455 },
  { event := event48549
    frameStart := 48455 },
  { event := event48550
    frameStart := 48455 },
  { event := event48551
    frameStart := 48455 },
  { event := event48552
    frameStart := 48455 },
  { event := event48553
    frameStart := 48455 },
  { event := event48554
    frameStart := 48455 },
  { event := event48555
    frameStart := 48455 },
  { event := event48556
    frameStart := 48455 },
  { event := event48557
    frameStart := 48455 },
  { event := event48558
    frameStart := 48455 },
  { event := event48559
    frameStart := 0 }
]

def eventLeaf3035 : Array AnnotatedEvent := #[
  { event := event48560
    frameStart := 0 },
  { event := event48561
    frameStart := 0 },
  { event := event48562
    frameStart := 0 },
  { event := event48563
    frameStart := 0 },
  { event := event48564
    frameStart := 0 },
  { event := event48565
    frameStart := 0 },
  { event := event48566
    frameStart := 0 },
  { event := event48567
    frameStart := 0 },
  { event := event48568
    frameStart := 0 },
  { event := event48569
    frameStart := 0 },
  { event := event48570
    frameStart := 0 },
  { event := event48571
    frameStart := 0 },
  { event := event48572
    frameStart := 0 },
  { event := event48573
    frameStart := 0 },
  { event := event48574
    frameStart := 0 },
  { event := event48575
    frameStart := 0 }
]

def eventLeaf3036 : Array AnnotatedEvent := #[
  { event := event48576
    frameStart := 0 },
  { event := event48577
    frameStart := 0 },
  { event := event48578
    frameStart := 0 },
  { event := event48579
    frameStart := 0 },
  { event := event48580
    frameStart := 0 },
  { event := event48581
    frameStart := 0 },
  { event := event48582
    frameStart := 0 },
  { event := event48583
    frameStart := 0 },
  { event := event48584
    frameStart := 0 },
  { event := event48585
    frameStart := 0 },
  { event := event48586
    frameStart := 0 },
  { event := event48587
    frameStart := 0 },
  { event := event48588
    frameStart := 0 },
  { event := event48589
    frameStart := 0 },
  { event := event48590
    frameStart := 0 },
  { event := event48591
    frameStart := 0 }
]

def eventLeaf3037 : Array AnnotatedEvent := #[
  { event := event48592
    frameStart := 0 },
  { event := event48593
    frameStart := 0 },
  { event := event48594
    frameStart := 0 },
  { event := event48595
    frameStart := 0 },
  { event := event48596
    frameStart := 0 },
  { event := event48597
    frameStart := 0 },
  { event := event48598
    frameStart := 0 },
  { event := event48599
    frameStart := 0 },
  { event := event48600
    frameStart := 0 },
  { event := event48601
    frameStart := 0 },
  { event := event48602
    frameStart := 0 },
  { event := event48603
    frameStart := 0 },
  { event := event48604
    frameStart := 0 },
  { event := event48605
    frameStart := 0 },
  { event := event48606
    frameStart := 0 },
  { event := event48607
    frameStart := 0 }
]

def eventLeaf3038 : Array AnnotatedEvent := #[
  { event := event48608
    frameStart := 0 },
  { event := event48609
    frameStart := 0 },
  { event := event48610
    frameStart := 0 },
  { event := event48611
    frameStart := 0 },
  { event := event48612
    frameStart := 0 },
  { event := event48613
    frameStart := 0 },
  { event := event48614
    frameStart := 0 },
  { event := event48615
    frameStart := 0 },
  { event := event48616
    frameStart := 0 },
  { event := event48617
    frameStart := 0 },
  { event := event48618
    frameStart := 0 },
  { event := event48619
    frameStart := 0 },
  { event := event48620
    frameStart := 0 },
  { event := event48621
    frameStart := 0 },
  { event := event48622
    frameStart := 0 },
  { event := event48623
    frameStart := 0 }
]

def eventLeaf3039 : Array AnnotatedEvent := #[
  { event := event48624
    frameStart := 0 },
  { event := event48625
    frameStart := 0 },
  { event := event48626
    frameStart := 0 },
  { event := event48627
    frameStart := 0 },
  { event := event48628
    frameStart := 0 },
  { event := event48629
    frameStart := 0 },
  { event := event48630
    frameStart := 0 },
  { event := event48631
    frameStart := 0 },
  { event := event48632
    frameStart := 0 },
  { event := event48633
    frameStart := 0 },
  { event := event48634
    frameStart := 0 },
  { event := event48635
    frameStart := 0 },
  { event := event48636
    frameStart := 0 },
  { event := event48637
    frameStart := 0 },
  { event := event48638
    frameStart := 0 },
  { event := event48639
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events189
