import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events326

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event83456 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11762⟩⟩) (.product (.predecessor 0 83454 .coefficient) (.predecessor 1 83455 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event83457 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11762⟩⟩, .operator (⟨83453, 0⟩, ⟨83450, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], []⟩, (1)⟩)

def exact83458RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], []⟩, (1)⟩]

theorem exact83458RawTermsValid :
    exact83458RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83458 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11762⟩⟩) exact83458RawTerms (.finite 900) 83456 .exactZero (none)

def event83459 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11763⟩⟩) 0 ⟨11762⟩ 83458

def event83460 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11763⟩⟩) (.identity (.predecessor 0 83459 .coefficient))

def event83461 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11763⟩⟩) (.finite 900)

def event83462 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23079⟩⟩) 0 ⟨11763⟩ 83461

def event83463 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23079⟩⟩) (.authority (.programFamilyFact))

def event83464 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23079⟩⟩) (.finite 3720)

def event83465 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event83466 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23080⟩⟩) 0 ⟨6689⟩ 83465

def event83467 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23080⟩⟩) 1 ⟨23079⟩ 83464

def event83468 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23080⟩⟩) (.authority (.operator))

def exact83469RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23080⟩⟩]⟩, (1)⟩]

theorem exact83469RawTermsValid :
    exact83469RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83469 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23080⟩⟩) exact83469RawTerms .large 83468 .exactZero (none)

def event83470 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25142⟩⟩) 0 ⟨23080⟩ 83469

def event83471 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25142⟩⟩) (.authority (.operator))

def exact83472RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩, (1)⟩]

theorem exact83472RawTermsValid :
    exact83472RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83472 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25142⟩⟩) exact83472RawTerms (.finite 8192) 83471 .exactZero (none)

def event83473 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event83474 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event83475 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11857⟩⟩) 0 ⟨11763⟩ 83461

def event83476 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11857⟩⟩) 1 ⟨110⟩ 83474

def event83477 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11857⟩⟩) (.sum [.predecessor 0 83475 .coefficient, .predecessor 1 83476 .coefficient])

def event83478 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11857⟩⟩) (.finite 900)

def event83479 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11858⟩⟩) 0 ⟨11857⟩ 83478

def event83480 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11858⟩⟩) (.identity (.predecessor 0 83479 .coefficient))

def exact83481RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], []⟩, (1)⟩]

theorem exact83481RawTermsValid :
    exact83481RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83481 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11858⟩⟩) exact83481RawTerms (.finite 900) 83480 .exactZero (none)

def event83482 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact83483RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact83483RawTermsValid :
    exact83483RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83483 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact83483RawTerms .large 83482 .exactZero (none)

def event83484 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11859⟩⟩) 0 ⟨6544⟩ 83483

def event83485 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11859⟩⟩) 1 ⟨11858⟩ 83481

def event83486 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11859⟩⟩) (.product (.predecessor 0 83484 .coefficient) (.predecessor 1 83485 .coefficient) (⟨false, false, none, none, none⟩))

def event83487 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11859⟩⟩, .operator (⟨83483, 0⟩, ⟨83481, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact83488RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact83488RawTermsValid :
    exact83488RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83488 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11859⟩⟩) exact83488RawTerms .large 83486 .exactZero (none)

def event83489 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 83465

def event83490 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact83491RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact83491RawTermsValid :
    exact83491RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83491 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact83491RawTerms .large 83490 .exactZero (none)

def event83492 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6783⟩⟩) 0 ⟨6757⟩ 83491

def event83493 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6783⟩⟩) (.identity (.predecessor 0 83492 .coefficient))

def exact83494RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩, (1)⟩]

theorem exact83494RawTermsValid :
    exact83494RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83494 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6783⟩⟩) exact83494RawTerms .large 83493 .exactZero (none)

def event83495 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7861⟩⟩) 0 ⟨6783⟩ 83494

def event83496 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7861⟩⟩) (.authority (.operator))

def exact83497RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩]

theorem exact83497RawTermsValid :
    exact83497RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83497 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7861⟩⟩) exact83497RawTerms (.finite 8192) 83496 .exactZero (none)

def event83498 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7862⟩⟩) 0 ⟨7861⟩ 83497

def event83499 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7862⟩⟩) 1 ⟨2348⟩ 83431

def event83500 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7862⟩⟩) (.scale (.predecessor 0 83498 .coefficient) (.value (.predecessor 1 83499 .coefficient)))

def exact83501RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩]

theorem exact83501RawTermsValid :
    exact83501RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83501 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7862⟩⟩) exact83501RawTerms (.finite 8192) 83500 .exactZero (none)

def event83502 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6763⟩⟩) 0 ⟨6757⟩ 83491

def event83503 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6763⟩⟩) (.identity (.predecessor 0 83502 .coefficient))

def exact83504RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩]⟩, (1)⟩]

theorem exact83504RawTermsValid :
    exact83504RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83504 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6763⟩⟩) exact83504RawTerms .large 83503 .exactZero (none)

def event83505 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7863⟩⟩) 0 ⟨6763⟩ 83504

def event83506 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7863⟩⟩) 1 ⟨7862⟩ 83501

def event83507 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7863⟩⟩) (.product (.predecessor 0 83505 .coefficient) (.predecessor 1 83506 .coefficient) (⟨false, false, none, none, none⟩))

def event83508 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7863⟩⟩, .operator (⟨83504, 0⟩, ⟨83501, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩)

def exact83509RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩]

theorem exact83509RawTermsValid :
    exact83509RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83509 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7863⟩⟩) exact83509RawTerms .large 83507 .exactZero (none)

def event83510 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11860⟩⟩) 0 ⟨7863⟩ 83509

def event83511 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11860⟩⟩) 1 ⟨11859⟩ 83488

def event83512 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11860⟩⟩) (.sum [.predecessor 0 83510 .coefficient, .predecessor 1 83511 .coefficient])

def exact83513RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83513RawTermsValid :
    exact83513RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83513 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11860⟩⟩) exact83513RawTerms .large 83512 .exactZero (none)

def event83514 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25145⟩⟩) 0 ⟨11860⟩ 83513

def event83515 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25145⟩⟩) 1 ⟨25142⟩ 83472

def event83516 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25145⟩⟩) (.product (.predecessor 0 83514 .coefficient) (.predecessor 1 83515 .coefficient) (⟨false, false, none, none, none⟩))

def event83517 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25145⟩⟩, .operator (⟨83513, 0⟩, ⟨83472, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩, (1)⟩)

def event83518 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25145⟩⟩, .operator (⟨83513, 1⟩, ⟨83472, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩, (-1)⟩)

def event83519 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25145⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25142⟩⟩) ⟨23080⟩ 83469)

def event83520 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25145⟩⟩, .relation 83519 0, ⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨23080⟩⟩]⟩, (-1)⟩)

def exact83521RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨23080⟩⟩]⟩, (-1)⟩]

theorem exact83521RawTermsValid :
    exact83521RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83521 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25145⟩⟩) exact83521RawTerms .large 83516 .exactZero (none)

def event83522 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16262⟩⟩) 0 ⟨11763⟩ 83461

def event83523 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16262⟩⟩) (.authority (.programFamilyFact))

def exact83524RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], []⟩, (1)⟩]

theorem exact83524RawTermsValid :
    exact83524RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83524 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16262⟩⟩) exact83524RawTerms (.finite 30) 83523 .exactZero (none)

def event83525 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16264⟩⟩) 0 ⟨6544⟩ 83483

def event83526 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16264⟩⟩) 1 ⟨16262⟩ 83524

def event83527 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16264⟩⟩) (.product (.predecessor 0 83525 .coefficient) (.predecessor 1 83526 .coefficient) (⟨false, true, none, none, some 1⟩))

def event83528 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16264⟩⟩, .operator (⟨83483, 0⟩, ⟨83524, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact83529RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact83529RawTermsValid :
    exact83529RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83529 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16264⟩⟩) exact83529RawTerms .large 83527 .exactZero (none)

def event83530 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6700⟩⟩) 0 ⟨6689⟩ 83465

def event83531 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6700⟩⟩) (.authority (.operator))

def exact83532RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩]

theorem exact83532RawTermsValid :
    exact83532RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83532 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6700⟩⟩) exact83532RawTerms .large 83531 .exactZero (none)

def event83533 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16265⟩⟩) 0 ⟨6700⟩ 83532

def event83534 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16265⟩⟩) 1 ⟨16264⟩ 83529

def event83535 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16265⟩⟩) (.sum [.predecessor 0 83533 .coefficient, .predecessor 1 83534 .coefficient])

def exact83536RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83536RawTermsValid :
    exact83536RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83536 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16265⟩⟩) exact83536RawTerms .large 83535 .exactZero (none)

def event83537 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25146⟩⟩) 0 ⟨16265⟩ 83536

def event83538 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25146⟩⟩) 1 ⟨25145⟩ 83521

def event83539 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25146⟩⟩) (.sum [.predecessor 0 83537 .coefficient, .predecessor 1 83538 .coefficient])

def exact83540RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨23080⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83540RawTermsValid :
    exact83540RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83540 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25146⟩⟩) exact83540RawTerms .large 83539 .exactZero (none)

def event83541 : Event := .preFoldPolynomial 83540 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨23080⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact83542RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨23080⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event83542 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25146⟩⟩) 83541 exact83542RawTerms .large 83539 .exactZero (none)

def event83543 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨11763⟩⟩) ⟨⟨113⟩, ⟨18⟩, ⟨109⟩⟩ ⟨83379, 83543⟩

def event83544 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19747⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩) (1) 0 2 (.universal 83543 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩) (none) 83542)

def event83545 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19747⟩⟩, .relation 83544 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩)

def event83546 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19747⟩⟩, .relation 83544 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩, (-1)⟩)

def event83547 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19747⟩⟩, .relation 83544 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨23080⟩⟩]⟩, (1)⟩)

def event83548 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19747⟩⟩, .relation 83544 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact83549RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨23080⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83549RawTermsValid :
    exact83549RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83549 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19747⟩⟩) exact83549RawTerms .large 83375 (.finite 1811303510016) (some (83377))

def event83550 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25144⟩⟩) 0 ⟨19747⟩ 83549

def event83551 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25144⟩⟩) 1 ⟨25143⟩ 83365

def event83552 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25144⟩⟩) (.sum [.predecessor 0 83550 .coefficient, .predecessor 1 83551 .coefficient])

def event83553 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25144⟩⟩, .operator (⟨83549, 2⟩, ⟨83365, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨23080⟩⟩]⟩, (-1)⟩)

def event83554 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25144⟩⟩, .operator (⟨83549, 1⟩, ⟨83365, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩, (1)⟩)

def event83555 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25144⟩⟩) (.sum [.result 83549 .summary, .result 83365 .summary])

def exact83556RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83556RawTermsValid :
    exact83556RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83556 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25144⟩⟩) exact83556RawTerms .large 83552 (.finite 352097360556032) (some (83555))

def event83557 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28519⟩⟩) 0 ⟨25144⟩ 83556

def event83558 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28519⟩⟩) 1 ⟨28517⟩ 83281

def event83559 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28519⟩⟩) (.product (.predecessor 0 83557 .coefficient) (.predecessor 1 83558 .coefficient) (⟨false, false, none, none, none⟩))

def event83560 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28519⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩) [⟨.result 83281 .coefficient, false, none⟩])

def event83561 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28519⟩⟩) (.product (.result 83556 .summary) (.transfer 83560) (⟨false, false, none, none, none⟩))

def event83562 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28519⟩⟩, .operator (⟨83556, 0⟩, ⟨83281, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩, (1)⟩)

def event83563 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28519⟩⟩, .operator (⟨83556, 1⟩, ⟨83281, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩, (-1)⟩)

def event83564 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28519⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28517⟩⟩) ⟨24351⟩ 83278)

def event83565 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28519⟩⟩, .relation 83564 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨24351⟩⟩]⟩, (-1)⟩)

def exact83566RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨24351⟩⟩]⟩, (-1)⟩]

theorem exact83566RawTermsValid :
    exact83566RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83566 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28519⟩⟩) exact83566RawTerms .large 83559 (.finite 1292202946798406336512) (some (83561))

def event83567 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21832⟩⟩) 0 ⟨16263⟩ 4006

def event83568 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21832⟩⟩) (.authority (.relationPreimageSource ⟨50⟩))

def exact83569RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21832⟩⟩]⟩, (1)⟩]

theorem exact83569RawTermsValid :
    exact83569RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83569 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21832⟩⟩) exact83569RawTerms (.finite 136065468) 83568 .exactZero (none)

def event83570 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21834⟩⟩) 0 ⟨21832⟩ 83569

def event83571 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21834⟩⟩) 1 ⟨2348⟩ 4

def event83572 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21834⟩⟩) (.scale (.predecessor 0 83570 .coefficient) (.value (.predecessor 1 83571 .coefficient)))

def exact83573RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21832⟩⟩]⟩, (1)⟩]

theorem exact83573RawTermsValid :
    exact83573RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83573 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21834⟩⟩) exact83573RawTerms (.finite 136065468) 83572 .exactZero (none)

def event83574 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21835⟩⟩) 0 ⟨5541⟩ 80012

def event83575 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21835⟩⟩) 1 ⟨21834⟩ 83573

def event83576 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21835⟩⟩) (.product (.predecessor 0 83574 .coefficient) (.predecessor 1 83575 .coefficient) (⟨false, false, none, none, none⟩))

def event83577 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21835⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21832⟩⟩]⟩) [⟨.result 83569 .coefficient, false, none⟩])

def event83578 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21835⟩⟩) (.product (.result 80012 .summary) (.transfer 83577) (⟨false, false, none, none, none⟩))

def event83579 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21835⟩⟩, .operator (⟨80012, 0⟩, ⟨83573, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21832⟩⟩]⟩, (1)⟩)

def event83580 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21833⟩⟩)

def event83581 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event83582 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event83583 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event83584 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event83585 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event83586 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event83587 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event83588 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event83589 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 83588

def event83590 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 83586

def event83591 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 83589 .coefficient) (.value (.predecessor 1 83590 .coefficient)))

def event83592 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event83593 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 83592

def event83594 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 83584

def event83595 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 83593 .coefficient, .predecessor 1 83594 .coefficient])

def event83596 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event83597 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 83596

def event83598 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 83582

def event83599 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 83598 .coefficient))

def event83600 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event83601 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11761⟩⟩) 0 ⟨5536⟩ 83600

def event83602 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11761⟩⟩) (.authority (.programFamilyFact))

def exact83603RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11761⟩⟩], []⟩, (1)⟩]

theorem exact83603RawTermsValid :
    exact83603RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83603 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11761⟩⟩) exact83603RawTerms (.finite 30) 83602 .exactZero (none)

def event83604 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9610⟩⟩) 0 ⟨5536⟩ 83600

def event83605 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9610⟩⟩) (.authority (.programFamilyFact))

def exact83606RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩], []⟩, (1)⟩]

theorem exact83606RawTermsValid :
    exact83606RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83606 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9610⟩⟩) exact83606RawTerms (.finite 30) 83605 .exactZero (none)

def event83607 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11762⟩⟩) 0 ⟨9610⟩ 83606

def event83608 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11762⟩⟩) 1 ⟨11761⟩ 83603

def event83609 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11762⟩⟩) (.product (.predecessor 0 83607 .coefficient) (.predecessor 1 83608 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event83610 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11762⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], []⟩) [⟨.result 83606 .coefficient, true, some 1⟩, ⟨.result 83603 .coefficient, true, some 1⟩])

def event83611 : Event := .survivorFold (1) 83610

def exact83612RawTerms : List Term := []

theorem exact83612RawTermsValid :
    exact83612RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83612 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11762⟩⟩) exact83612RawTerms (.finite 900) 83609 (.finite 900) (some (83610))

def event83613 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11763⟩⟩) 0 ⟨11762⟩ 83612

def event83614 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11763⟩⟩) (.identity (.predecessor 0 83613 .coefficient))

def event83615 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11763⟩⟩) (.finite 900)

def event83616 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16262⟩⟩) 0 ⟨11763⟩ 83615

def event83617 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16262⟩⟩) (.authority (.programFamilyFact))

def exact83618RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], []⟩, (1)⟩]

theorem exact83618RawTermsValid :
    exact83618RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83618 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16262⟩⟩) exact83618RawTerms (.finite 30) 83617 .exactZero (none)

def event83619 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16263⟩⟩) 0 ⟨16262⟩ 83618

def event83620 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16263⟩⟩) (.identity (.predecessor 0 83619 .coefficient))

def event83621 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16263⟩⟩) (.finite 30)

def event83622 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21832⟩⟩) 0 ⟨16263⟩ 83621

def event83623 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21832⟩⟩) (.authority (.relationPreimageSource ⟨50⟩))

def exact83624RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21832⟩⟩]⟩, (1)⟩]

theorem exact83624RawTermsValid :
    exact83624RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83624 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21832⟩⟩) exact83624RawTerms (.finite 136065468) 83623 .exactZero (none)

def event83625 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact83626RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact83626RawTermsValid :
    exact83626RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83626 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact83626RawTerms .large 83625 .exactZero (none)

def event83627 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21833⟩⟩) 0 ⟨6⟩ 83626

def event83628 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21833⟩⟩) 1 ⟨21832⟩ 83624

def event83629 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21833⟩⟩) (.product (.predecessor 0 83627 .coefficient) (.predecessor 1 83628 .coefficient) (⟨false, false, none, none, none⟩))

def event83630 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21833⟩⟩, .operator (⟨83626, 0⟩, ⟨83624, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21832⟩⟩]⟩, (1)⟩)

def exact83631RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21832⟩⟩]⟩, (1)⟩]

theorem exact83631RawTermsValid :
    exact83631RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83631 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21833⟩⟩) exact83631RawTerms .large 83629 .exactZero (none)

def event83632 : Event := .preFoldPolynomial 83631 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21832⟩⟩]⟩, (1)⟩] .exactZero none

def exact83633RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21832⟩⟩]⟩, (1)⟩]

def event83633 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21833⟩⟩) 83632 exact83633RawTerms .large 83629 .exactZero (none)

def event83634 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28522⟩⟩)

def event83635 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event83636 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event83637 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event83638 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event83639 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event83640 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event83641 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event83642 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event83643 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 83642

def event83644 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 83640

def event83645 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 83643 .coefficient) (.value (.predecessor 1 83644 .coefficient)))

def event83646 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event83647 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 83646

def event83648 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 83638

def event83649 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 83647 .coefficient, .predecessor 1 83648 .coefficient])

def event83650 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event83651 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 83650

def event83652 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 83636

def event83653 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 83652 .coefficient))

def event83654 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event83655 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11761⟩⟩) 0 ⟨5536⟩ 83654

def event83656 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11761⟩⟩) (.authority (.programFamilyFact))

def exact83657RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11761⟩⟩], []⟩, (1)⟩]

theorem exact83657RawTermsValid :
    exact83657RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83657 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11761⟩⟩) exact83657RawTerms (.finite 30) 83656 .exactZero (none)

def event83658 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9610⟩⟩) 0 ⟨5536⟩ 83654

def event83659 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9610⟩⟩) (.authority (.programFamilyFact))

def exact83660RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩], []⟩, (1)⟩]

theorem exact83660RawTermsValid :
    exact83660RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83660 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9610⟩⟩) exact83660RawTerms (.finite 30) 83659 .exactZero (none)

def event83661 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11762⟩⟩) 0 ⟨9610⟩ 83660

def event83662 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11762⟩⟩) 1 ⟨11761⟩ 83657

def event83663 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11762⟩⟩) (.product (.predecessor 0 83661 .coefficient) (.predecessor 1 83662 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event83664 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11762⟩⟩, .operator (⟨83660, 0⟩, ⟨83657, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], []⟩, (1)⟩)

def exact83665RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], []⟩, (1)⟩]

theorem exact83665RawTermsValid :
    exact83665RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83665 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11762⟩⟩) exact83665RawTerms (.finite 900) 83663 .exactZero (none)

def event83666 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11763⟩⟩) 0 ⟨11762⟩ 83665

def event83667 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11763⟩⟩) (.identity (.predecessor 0 83666 .coefficient))

def event83668 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11763⟩⟩) (.finite 900)

def event83669 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16262⟩⟩) 0 ⟨11763⟩ 83668

def event83670 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16262⟩⟩) (.authority (.programFamilyFact))

def exact83671RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], []⟩, (1)⟩]

theorem exact83671RawTermsValid :
    exact83671RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83671 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16262⟩⟩) exact83671RawTerms (.finite 30) 83670 .exactZero (none)

def event83672 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16263⟩⟩) 0 ⟨16262⟩ 83671

def event83673 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16263⟩⟩) (.identity (.predecessor 0 83672 .coefficient))

def event83674 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16263⟩⟩) (.finite 30)

def event83675 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24349⟩⟩) 0 ⟨16263⟩ 83674

def event83676 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24349⟩⟩) (.authority (.programFamilyFact))

def event83677 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24349⟩⟩) (.finite 3720)

def event83678 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event83679 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24351⟩⟩) 0 ⟨6689⟩ 83678

def event83680 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24351⟩⟩) 1 ⟨24349⟩ 83677

def event83681 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24351⟩⟩) (.authority (.operator))

def exact83682RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24351⟩⟩]⟩, (1)⟩]

theorem exact83682RawTermsValid :
    exact83682RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83682 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24351⟩⟩) exact83682RawTerms .large 83681 .exactZero (none)

def event83683 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28517⟩⟩) 0 ⟨24351⟩ 83682

def event83684 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28517⟩⟩) (.authority (.operator))

def exact83685RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩, (1)⟩]

theorem exact83685RawTermsValid :
    exact83685RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83685 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28517⟩⟩) exact83685RawTerms (.finite 8192) 83684 .exactZero (none)

def event83686 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event83687 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event83688 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16337⟩⟩) 0 ⟨16263⟩ 83674

def event83689 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16337⟩⟩) 1 ⟨110⟩ 83687

def event83690 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16337⟩⟩) (.sum [.predecessor 0 83688 .coefficient, .predecessor 1 83689 .coefficient])

def event83691 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16337⟩⟩) (.finite 30)

def event83692 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16338⟩⟩) 0 ⟨16337⟩ 83691

def event83693 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16338⟩⟩) (.identity (.predecessor 0 83692 .coefficient))

def exact83694RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], []⟩, (1)⟩]

theorem exact83694RawTermsValid :
    exact83694RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83694 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16338⟩⟩) exact83694RawTerms (.finite 30) 83693 .exactZero (none)

def event83695 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact83696RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact83696RawTermsValid :
    exact83696RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83696 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact83696RawTerms .large 83695 .exactZero (none)

def event83697 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16339⟩⟩) 0 ⟨6544⟩ 83696

def event83698 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16339⟩⟩) 1 ⟨16338⟩ 83694

def event83699 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16339⟩⟩) (.product (.predecessor 0 83697 .coefficient) (.predecessor 1 83698 .coefficient) (⟨false, false, none, none, none⟩))

def event83700 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16339⟩⟩, .operator (⟨83696, 0⟩, ⟨83694, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact83701RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact83701RawTermsValid :
    exact83701RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83701 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16339⟩⟩) exact83701RawTerms .large 83699 .exactZero (none)

def event83702 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6700⟩⟩) 0 ⟨6689⟩ 83678

def event83703 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6700⟩⟩) (.authority (.operator))

def exact83704RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩]

theorem exact83704RawTermsValid :
    exact83704RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83704 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6700⟩⟩) exact83704RawTerms .large 83703 .exactZero (none)

def event83705 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16340⟩⟩) 0 ⟨6700⟩ 83704

def event83706 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16340⟩⟩) 1 ⟨16339⟩ 83701

def event83707 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16340⟩⟩) (.sum [.predecessor 0 83705 .coefficient, .predecessor 1 83706 .coefficient])

def exact83708RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83708RawTermsValid :
    exact83708RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83708 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16340⟩⟩) exact83708RawTerms .large 83707 .exactZero (none)

def event83709 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28518⟩⟩) 0 ⟨16340⟩ 83708

def event83710 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28518⟩⟩) 1 ⟨28517⟩ 83685

def event83711 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28518⟩⟩) (.product (.predecessor 0 83709 .coefficient) (.predecessor 1 83710 .coefficient) (⟨false, false, none, none, none⟩))

def eventLeaf5216 : Array AnnotatedEvent := #[
  { event := event83456
    frameStart := 83427 },
  { event := event83457
    frameStart := 83427 },
  { event := event83458
    frameStart := 83427 },
  { event := event83459
    frameStart := 83427 },
  { event := event83460
    frameStart := 83427 },
  { event := event83461
    frameStart := 83427 },
  { event := event83462
    frameStart := 83427 },
  { event := event83463
    frameStart := 83427 },
  { event := event83464
    frameStart := 83427 },
  { event := event83465
    frameStart := 83427 },
  { event := event83466
    frameStart := 83427 },
  { event := event83467
    frameStart := 83427 },
  { event := event83468
    frameStart := 83427 },
  { event := event83469
    frameStart := 83427 },
  { event := event83470
    frameStart := 83427 },
  { event := event83471
    frameStart := 83427 }
]

def eventLeaf5217 : Array AnnotatedEvent := #[
  { event := event83472
    frameStart := 83427 },
  { event := event83473
    frameStart := 83427 },
  { event := event83474
    frameStart := 83427 },
  { event := event83475
    frameStart := 83427 },
  { event := event83476
    frameStart := 83427 },
  { event := event83477
    frameStart := 83427 },
  { event := event83478
    frameStart := 83427 },
  { event := event83479
    frameStart := 83427 },
  { event := event83480
    frameStart := 83427 },
  { event := event83481
    frameStart := 83427 },
  { event := event83482
    frameStart := 83427 },
  { event := event83483
    frameStart := 83427 },
  { event := event83484
    frameStart := 83427 },
  { event := event83485
    frameStart := 83427 },
  { event := event83486
    frameStart := 83427 },
  { event := event83487
    frameStart := 83427 }
]

def eventLeaf5218 : Array AnnotatedEvent := #[
  { event := event83488
    frameStart := 83427 },
  { event := event83489
    frameStart := 83427 },
  { event := event83490
    frameStart := 83427 },
  { event := event83491
    frameStart := 83427 },
  { event := event83492
    frameStart := 83427 },
  { event := event83493
    frameStart := 83427 },
  { event := event83494
    frameStart := 83427 },
  { event := event83495
    frameStart := 83427 },
  { event := event83496
    frameStart := 83427 },
  { event := event83497
    frameStart := 83427 },
  { event := event83498
    frameStart := 83427 },
  { event := event83499
    frameStart := 83427 },
  { event := event83500
    frameStart := 83427 },
  { event := event83501
    frameStart := 83427 },
  { event := event83502
    frameStart := 83427 },
  { event := event83503
    frameStart := 83427 }
]

def eventLeaf5219 : Array AnnotatedEvent := #[
  { event := event83504
    frameStart := 83427 },
  { event := event83505
    frameStart := 83427 },
  { event := event83506
    frameStart := 83427 },
  { event := event83507
    frameStart := 83427 },
  { event := event83508
    frameStart := 83427 },
  { event := event83509
    frameStart := 83427 },
  { event := event83510
    frameStart := 83427 },
  { event := event83511
    frameStart := 83427 },
  { event := event83512
    frameStart := 83427 },
  { event := event83513
    frameStart := 83427 },
  { event := event83514
    frameStart := 83427 },
  { event := event83515
    frameStart := 83427 },
  { event := event83516
    frameStart := 83427 },
  { event := event83517
    frameStart := 83427 },
  { event := event83518
    frameStart := 83427 },
  { event := event83519
    frameStart := 83427 }
]

def eventLeaf5220 : Array AnnotatedEvent := #[
  { event := event83520
    frameStart := 83427 },
  { event := event83521
    frameStart := 83427 },
  { event := event83522
    frameStart := 83427 },
  { event := event83523
    frameStart := 83427 },
  { event := event83524
    frameStart := 83427 },
  { event := event83525
    frameStart := 83427 },
  { event := event83526
    frameStart := 83427 },
  { event := event83527
    frameStart := 83427 },
  { event := event83528
    frameStart := 83427 },
  { event := event83529
    frameStart := 83427 },
  { event := event83530
    frameStart := 83427 },
  { event := event83531
    frameStart := 83427 },
  { event := event83532
    frameStart := 83427 },
  { event := event83533
    frameStart := 83427 },
  { event := event83534
    frameStart := 83427 },
  { event := event83535
    frameStart := 83427 }
]

def eventLeaf5221 : Array AnnotatedEvent := #[
  { event := event83536
    frameStart := 83427 },
  { event := event83537
    frameStart := 83427 },
  { event := event83538
    frameStart := 83427 },
  { event := event83539
    frameStart := 83427 },
  { event := event83540
    frameStart := 83427 },
  { event := event83541
    frameStart := 83427 },
  { event := event83542
    frameStart := 83427 },
  { event := event83543
    frameStart := 0 },
  { event := event83544
    frameStart := 0 },
  { event := event83545
    frameStart := 0 },
  { event := event83546
    frameStart := 0 },
  { event := event83547
    frameStart := 0 },
  { event := event83548
    frameStart := 0 },
  { event := event83549
    frameStart := 0 },
  { event := event83550
    frameStart := 0 },
  { event := event83551
    frameStart := 0 }
]

def eventLeaf5222 : Array AnnotatedEvent := #[
  { event := event83552
    frameStart := 0 },
  { event := event83553
    frameStart := 0 },
  { event := event83554
    frameStart := 0 },
  { event := event83555
    frameStart := 0 },
  { event := event83556
    frameStart := 0 },
  { event := event83557
    frameStart := 0 },
  { event := event83558
    frameStart := 0 },
  { event := event83559
    frameStart := 0 },
  { event := event83560
    frameStart := 0 },
  { event := event83561
    frameStart := 0 },
  { event := event83562
    frameStart := 0 },
  { event := event83563
    frameStart := 0 },
  { event := event83564
    frameStart := 0 },
  { event := event83565
    frameStart := 0 },
  { event := event83566
    frameStart := 0 },
  { event := event83567
    frameStart := 0 }
]

def eventLeaf5223 : Array AnnotatedEvent := #[
  { event := event83568
    frameStart := 0 },
  { event := event83569
    frameStart := 0 },
  { event := event83570
    frameStart := 0 },
  { event := event83571
    frameStart := 0 },
  { event := event83572
    frameStart := 0 },
  { event := event83573
    frameStart := 0 },
  { event := event83574
    frameStart := 0 },
  { event := event83575
    frameStart := 0 },
  { event := event83576
    frameStart := 0 },
  { event := event83577
    frameStart := 0 },
  { event := event83578
    frameStart := 0 },
  { event := event83579
    frameStart := 0 },
  { event := event83580
    frameStart := 83580 },
  { event := event83581
    frameStart := 83580 },
  { event := event83582
    frameStart := 83580 },
  { event := event83583
    frameStart := 83580 }
]

def eventLeaf5224 : Array AnnotatedEvent := #[
  { event := event83584
    frameStart := 83580 },
  { event := event83585
    frameStart := 83580 },
  { event := event83586
    frameStart := 83580 },
  { event := event83587
    frameStart := 83580 },
  { event := event83588
    frameStart := 83580 },
  { event := event83589
    frameStart := 83580 },
  { event := event83590
    frameStart := 83580 },
  { event := event83591
    frameStart := 83580 },
  { event := event83592
    frameStart := 83580 },
  { event := event83593
    frameStart := 83580 },
  { event := event83594
    frameStart := 83580 },
  { event := event83595
    frameStart := 83580 },
  { event := event83596
    frameStart := 83580 },
  { event := event83597
    frameStart := 83580 },
  { event := event83598
    frameStart := 83580 },
  { event := event83599
    frameStart := 83580 }
]

def eventLeaf5225 : Array AnnotatedEvent := #[
  { event := event83600
    frameStart := 83580 },
  { event := event83601
    frameStart := 83580 },
  { event := event83602
    frameStart := 83580 },
  { event := event83603
    frameStart := 83580 },
  { event := event83604
    frameStart := 83580 },
  { event := event83605
    frameStart := 83580 },
  { event := event83606
    frameStart := 83580 },
  { event := event83607
    frameStart := 83580 },
  { event := event83608
    frameStart := 83580 },
  { event := event83609
    frameStart := 83580 },
  { event := event83610
    frameStart := 83580 },
  { event := event83611
    frameStart := 83580 },
  { event := event83612
    frameStart := 83580 },
  { event := event83613
    frameStart := 83580 },
  { event := event83614
    frameStart := 83580 },
  { event := event83615
    frameStart := 83580 }
]

def eventLeaf5226 : Array AnnotatedEvent := #[
  { event := event83616
    frameStart := 83580 },
  { event := event83617
    frameStart := 83580 },
  { event := event83618
    frameStart := 83580 },
  { event := event83619
    frameStart := 83580 },
  { event := event83620
    frameStart := 83580 },
  { event := event83621
    frameStart := 83580 },
  { event := event83622
    frameStart := 83580 },
  { event := event83623
    frameStart := 83580 },
  { event := event83624
    frameStart := 83580 },
  { event := event83625
    frameStart := 83580 },
  { event := event83626
    frameStart := 83580 },
  { event := event83627
    frameStart := 83580 },
  { event := event83628
    frameStart := 83580 },
  { event := event83629
    frameStart := 83580 },
  { event := event83630
    frameStart := 83580 },
  { event := event83631
    frameStart := 83580 }
]

def eventLeaf5227 : Array AnnotatedEvent := #[
  { event := event83632
    frameStart := 83580 },
  { event := event83633
    frameStart := 83580 },
  { event := event83634
    frameStart := 83634 },
  { event := event83635
    frameStart := 83634 },
  { event := event83636
    frameStart := 83634 },
  { event := event83637
    frameStart := 83634 },
  { event := event83638
    frameStart := 83634 },
  { event := event83639
    frameStart := 83634 },
  { event := event83640
    frameStart := 83634 },
  { event := event83641
    frameStart := 83634 },
  { event := event83642
    frameStart := 83634 },
  { event := event83643
    frameStart := 83634 },
  { event := event83644
    frameStart := 83634 },
  { event := event83645
    frameStart := 83634 },
  { event := event83646
    frameStart := 83634 },
  { event := event83647
    frameStart := 83634 }
]

def eventLeaf5228 : Array AnnotatedEvent := #[
  { event := event83648
    frameStart := 83634 },
  { event := event83649
    frameStart := 83634 },
  { event := event83650
    frameStart := 83634 },
  { event := event83651
    frameStart := 83634 },
  { event := event83652
    frameStart := 83634 },
  { event := event83653
    frameStart := 83634 },
  { event := event83654
    frameStart := 83634 },
  { event := event83655
    frameStart := 83634 },
  { event := event83656
    frameStart := 83634 },
  { event := event83657
    frameStart := 83634 },
  { event := event83658
    frameStart := 83634 },
  { event := event83659
    frameStart := 83634 },
  { event := event83660
    frameStart := 83634 },
  { event := event83661
    frameStart := 83634 },
  { event := event83662
    frameStart := 83634 },
  { event := event83663
    frameStart := 83634 }
]

def eventLeaf5229 : Array AnnotatedEvent := #[
  { event := event83664
    frameStart := 83634 },
  { event := event83665
    frameStart := 83634 },
  { event := event83666
    frameStart := 83634 },
  { event := event83667
    frameStart := 83634 },
  { event := event83668
    frameStart := 83634 },
  { event := event83669
    frameStart := 83634 },
  { event := event83670
    frameStart := 83634 },
  { event := event83671
    frameStart := 83634 },
  { event := event83672
    frameStart := 83634 },
  { event := event83673
    frameStart := 83634 },
  { event := event83674
    frameStart := 83634 },
  { event := event83675
    frameStart := 83634 },
  { event := event83676
    frameStart := 83634 },
  { event := event83677
    frameStart := 83634 },
  { event := event83678
    frameStart := 83634 },
  { event := event83679
    frameStart := 83634 }
]

def eventLeaf5230 : Array AnnotatedEvent := #[
  { event := event83680
    frameStart := 83634 },
  { event := event83681
    frameStart := 83634 },
  { event := event83682
    frameStart := 83634 },
  { event := event83683
    frameStart := 83634 },
  { event := event83684
    frameStart := 83634 },
  { event := event83685
    frameStart := 83634 },
  { event := event83686
    frameStart := 83634 },
  { event := event83687
    frameStart := 83634 },
  { event := event83688
    frameStart := 83634 },
  { event := event83689
    frameStart := 83634 },
  { event := event83690
    frameStart := 83634 },
  { event := event83691
    frameStart := 83634 },
  { event := event83692
    frameStart := 83634 },
  { event := event83693
    frameStart := 83634 },
  { event := event83694
    frameStart := 83634 },
  { event := event83695
    frameStart := 83634 }
]

def eventLeaf5231 : Array AnnotatedEvent := #[
  { event := event83696
    frameStart := 83634 },
  { event := event83697
    frameStart := 83634 },
  { event := event83698
    frameStart := 83634 },
  { event := event83699
    frameStart := 83634 },
  { event := event83700
    frameStart := 83634 },
  { event := event83701
    frameStart := 83634 },
  { event := event83702
    frameStart := 83634 },
  { event := event83703
    frameStart := 83634 },
  { event := event83704
    frameStart := 83634 },
  { event := event83705
    frameStart := 83634 },
  { event := event83706
    frameStart := 83634 },
  { event := event83707
    frameStart := 83634 },
  { event := event83708
    frameStart := 83634 },
  { event := event83709
    frameStart := 83634 },
  { event := event83710
    frameStart := 83634 },
  { event := event83711
    frameStart := 83634 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events326
