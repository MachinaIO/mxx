import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events236

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact60416RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15706⟩⟩], []⟩, (1)⟩]

theorem exact60416RawTermsValid :
    exact60416RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60416 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15706⟩⟩) exact60416RawTerms (.finite 12) 60415 .exactZero (none)

def event60417 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15707⟩⟩) 0 ⟨15706⟩ 60416

def event60418 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15707⟩⟩) (.identity (.predecessor 0 60417 .coefficient))

def event60419 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15707⟩⟩) (.finite 12)

def event60420 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15751⟩⟩) 0 ⟨15707⟩ 60419

def event60421 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15751⟩⟩) (.authority (.programFamilyFact))

def exact60422RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15751⟩⟩], []⟩, (1)⟩]

theorem exact60422RawTermsValid :
    exact60422RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60422 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15751⟩⟩) exact60422RawTerms (.finite 59) 60421 .exactZero (none)

def event60423 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11221⟩⟩) 0 ⟨5542⟩ 60123

def event60424 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11221⟩⟩) (.authority (.programFamilyFact))

def exact60425RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11221⟩⟩], []⟩, (1)⟩]

theorem exact60425RawTermsValid :
    exact60425RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60425 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11221⟩⟩) exact60425RawTerms (.finite 10) 60424 .exactZero (none)

def event60426 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13565⟩⟩) 0 ⟨5542⟩ 60123

def event60427 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13565⟩⟩) (.authority (.programFamilyFact))

def exact60428RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13565⟩⟩], []⟩, (1)⟩]

theorem exact60428RawTermsValid :
    exact60428RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60428 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13565⟩⟩) exact60428RawTerms (.finite 10) 60427 .exactZero (none)

def event60429 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13566⟩⟩) 0 ⟨13565⟩ 60428

def event60430 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13566⟩⟩) 1 ⟨11221⟩ 60425

def event60431 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13566⟩⟩) (.product (.predecessor 0 60429 .coefficient) (.predecessor 1 60430 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event60432 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13566⟩⟩, .operator (⟨60428, 0⟩, ⟨60425, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11221⟩⟩, ⟨.program ⟨214⟩, ⟨13565⟩⟩], []⟩, (1)⟩)

def exact60433RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11221⟩⟩, ⟨.program ⟨214⟩, ⟨13565⟩⟩], []⟩, (1)⟩]

theorem exact60433RawTermsValid :
    exact60433RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60433 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13566⟩⟩) exact60433RawTerms (.finite 100) 60431 .exactZero (none)

def event60434 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13567⟩⟩) 0 ⟨13566⟩ 60433

def event60435 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13567⟩⟩) (.identity (.predecessor 0 60434 .coefficient))

def event60436 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13567⟩⟩) (.finite 100)

def event60437 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15587⟩⟩) 0 ⟨13567⟩ 60436

def event60438 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15587⟩⟩) (.authority (.programFamilyFact))

def exact60439RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15587⟩⟩], []⟩, (1)⟩]

theorem exact60439RawTermsValid :
    exact60439RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60439 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15587⟩⟩) exact60439RawTerms (.finite 10) 60438 .exactZero (none)

def event60440 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15588⟩⟩) 0 ⟨15587⟩ 60439

def event60441 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15588⟩⟩) (.identity (.predecessor 0 60440 .coefficient))

def event60442 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15588⟩⟩) (.finite 10)

def event60443 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15632⟩⟩) 0 ⟨15588⟩ 60442

def event60444 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15632⟩⟩) (.authority (.programFamilyFact))

def exact60445RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], []⟩, (1)⟩]

theorem exact60445RawTermsValid :
    exact60445RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60445 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15632⟩⟩) exact60445RawTerms (.finite 58) 60444 .exactZero (none)

def event60446 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11137⟩⟩) 0 ⟨5542⟩ 60123

def event60447 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11137⟩⟩) (.authority (.programFamilyFact))

def exact60448RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11137⟩⟩], []⟩, (1)⟩]

theorem exact60448RawTermsValid :
    exact60448RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60448 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11137⟩⟩) exact60448RawTerms (.finite 6) 60447 .exactZero (none)

def event60449 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12172⟩⟩) 0 ⟨5542⟩ 60123

def event60450 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12172⟩⟩) (.authority (.programFamilyFact))

def exact60451RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12172⟩⟩], []⟩, (1)⟩]

theorem exact60451RawTermsValid :
    exact60451RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60451 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12172⟩⟩) exact60451RawTerms (.finite 6) 60450 .exactZero (none)

def event60452 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12173⟩⟩) 0 ⟨12172⟩ 60451

def event60453 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12173⟩⟩) 1 ⟨11137⟩ 60448

def event60454 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12173⟩⟩) (.product (.predecessor 0 60452 .coefficient) (.predecessor 1 60453 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event60455 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12173⟩⟩, .operator (⟨60451, 0⟩, ⟨60448, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], []⟩, (1)⟩)

def exact60456RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], []⟩, (1)⟩]

theorem exact60456RawTermsValid :
    exact60456RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60456 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12173⟩⟩) exact60456RawTerms (.finite 36) 60454 .exactZero (none)

def event60457 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12174⟩⟩) 0 ⟨12173⟩ 60456

def event60458 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12174⟩⟩) (.identity (.predecessor 0 60457 .coefficient))

def event60459 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12174⟩⟩) (.finite 36)

def event60460 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15426⟩⟩) 0 ⟨12174⟩ 60459

def event60461 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15426⟩⟩) (.authority (.programFamilyFact))

def exact60462RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15426⟩⟩], []⟩, (1)⟩]

theorem exact60462RawTermsValid :
    exact60462RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60462 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15426⟩⟩) exact60462RawTerms (.finite 6) 60461 .exactZero (none)

def event60463 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15427⟩⟩) 0 ⟨15426⟩ 60462

def event60464 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15427⟩⟩) (.identity (.predecessor 0 60463 .coefficient))

def event60465 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15427⟩⟩) (.finite 6)

def event60466 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17336⟩⟩) 0 ⟨15427⟩ 60465

def event60467 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17336⟩⟩) (.authority (.programFamilyFact))

def exact60468RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], []⟩, (1)⟩]

theorem exact60468RawTermsValid :
    exact60468RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60468 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17336⟩⟩) exact60468RawTerms (.finite 55) 60467 .exactZero (none)

def event60469 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10985⟩⟩) 0 ⟨5542⟩ 60123

def event60470 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10985⟩⟩) (.authority (.programFamilyFact))

def exact60471RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10985⟩⟩], []⟩, (1)⟩]

theorem exact60471RawTermsValid :
    exact60471RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60471 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10985⟩⟩) exact60471RawTerms (.finite 4) 60470 .exactZero (none)

def event60472 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10847⟩⟩) 0 ⟨5542⟩ 60123

def event60473 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10847⟩⟩) (.authority (.programFamilyFact))

def exact60474RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10847⟩⟩], []⟩, (1)⟩]

theorem exact60474RawTermsValid :
    exact60474RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60474 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10847⟩⟩) exact60474RawTerms (.finite 4) 60473 .exactZero (none)

def event60475 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10986⟩⟩) 0 ⟨10847⟩ 60474

def event60476 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10986⟩⟩) 1 ⟨10985⟩ 60471

def event60477 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10986⟩⟩) (.product (.predecessor 0 60475 .coefficient) (.predecessor 1 60476 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event60478 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10986⟩⟩, .operator (⟨60474, 0⟩, ⟨60471, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], []⟩, (1)⟩)

def exact60479RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10847⟩⟩, ⟨.program ⟨214⟩, ⟨10985⟩⟩], []⟩, (1)⟩]

theorem exact60479RawTermsValid :
    exact60479RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60479 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10986⟩⟩) exact60479RawTerms (.finite 16) 60477 .exactZero (none)

def event60480 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10987⟩⟩) 0 ⟨10986⟩ 60479

def event60481 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10987⟩⟩) (.identity (.predecessor 0 60480 .coefficient))

def event60482 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10987⟩⟩) (.finite 16)

def event60483 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15118⟩⟩) 0 ⟨10987⟩ 60482

def event60484 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15118⟩⟩) (.authority (.programFamilyFact))

def exact60485RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15118⟩⟩], []⟩, (1)⟩]

theorem exact60485RawTermsValid :
    exact60485RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60485 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15118⟩⟩) exact60485RawTerms (.finite 4) 60484 .exactZero (none)

def event60486 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15119⟩⟩) 0 ⟨15118⟩ 60485

def event60487 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15119⟩⟩) (.identity (.predecessor 0 60486 .coefficient))

def event60488 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15119⟩⟩) (.finite 4)

def event60489 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15370⟩⟩) 0 ⟨15119⟩ 60488

def event60490 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15370⟩⟩) (.authority (.programFamilyFact))

def exact60491RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], []⟩, (1)⟩]

theorem exact60491RawTermsValid :
    exact60491RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60491 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15370⟩⟩) exact60491RawTerms (.finite 51) 60490 .exactZero (none)

def event60492 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10684⟩⟩) 0 ⟨5542⟩ 60123

def event60493 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10684⟩⟩) (.authority (.programFamilyFact))

def exact60494RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10684⟩⟩], []⟩, (1)⟩]

theorem exact60494RawTermsValid :
    exact60494RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60494 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10684⟩⟩) exact60494RawTerms (.finite 3) 60493 .exactZero (none)

def event60495 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9510⟩⟩) 0 ⟨5542⟩ 60123

def event60496 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9510⟩⟩) (.authority (.programFamilyFact))

def exact60497RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9510⟩⟩], []⟩, (1)⟩]

theorem exact60497RawTermsValid :
    exact60497RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60497 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9510⟩⟩) exact60497RawTerms (.finite 3) 60496 .exactZero (none)

def event60498 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10685⟩⟩) 0 ⟨9510⟩ 60497

def event60499 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10685⟩⟩) 1 ⟨10684⟩ 60494

def event60500 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10685⟩⟩) (.product (.predecessor 0 60498 .coefficient) (.predecessor 1 60499 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event60501 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10685⟩⟩, .operator (⟨60497, 0⟩, ⟨60494, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9510⟩⟩, ⟨.program ⟨214⟩, ⟨10684⟩⟩], []⟩, (1)⟩)

def exact60502RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9510⟩⟩, ⟨.program ⟨214⟩, ⟨10684⟩⟩], []⟩, (1)⟩]

theorem exact60502RawTermsValid :
    exact60502RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60502 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10685⟩⟩) exact60502RawTerms (.finite 9) 60500 .exactZero (none)

def event60503 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10686⟩⟩) 0 ⟨10685⟩ 60502

def event60504 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10686⟩⟩) (.identity (.predecessor 0 60503 .coefficient))

def event60505 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10686⟩⟩) (.finite 9)

def event60506 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14957⟩⟩) 0 ⟨10686⟩ 60505

def event60507 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14957⟩⟩) (.authority (.programFamilyFact))

def exact60508RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14957⟩⟩], []⟩, (1)⟩]

theorem exact60508RawTermsValid :
    exact60508RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60508 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14957⟩⟩) exact60508RawTerms (.finite 3) 60507 .exactZero (none)

def event60509 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14958⟩⟩) 0 ⟨14957⟩ 60508

def event60510 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14958⟩⟩) (.identity (.predecessor 0 60509 .coefficient))

def event60511 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14958⟩⟩) (.finite 3)

def event60512 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15314⟩⟩) 0 ⟨14958⟩ 60511

def event60513 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15314⟩⟩) (.authority (.programFamilyFact))

def exact60514RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩]

theorem exact60514RawTermsValid :
    exact60514RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60514 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15314⟩⟩) exact60514RawTerms (.finite 48) 60513 .exactZero (none)

def event60515 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10488⟩⟩) 0 ⟨5542⟩ 60123

def event60516 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10488⟩⟩) (.authority (.programFamilyFact))

def exact60517RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10488⟩⟩], []⟩, (1)⟩]

theorem exact60517RawTermsValid :
    exact60517RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60517 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10488⟩⟩) exact60517RawTerms (.finite 2) 60516 .exactZero (none)

def event60518 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9405⟩⟩) 0 ⟨5542⟩ 60123

def event60519 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9405⟩⟩) (.authority (.programFamilyFact))

def exact60520RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9405⟩⟩], []⟩, (1)⟩]

theorem exact60520RawTermsValid :
    exact60520RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60520 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9405⟩⟩) exact60520RawTerms (.finite 2) 60519 .exactZero (none)

def event60521 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10489⟩⟩) 0 ⟨9405⟩ 60520

def event60522 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10489⟩⟩) 1 ⟨10488⟩ 60517

def event60523 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10489⟩⟩) (.product (.predecessor 0 60521 .coefficient) (.predecessor 1 60522 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event60524 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10489⟩⟩, .operator (⟨60520, 0⟩, ⟨60517, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9405⟩⟩, ⟨.program ⟨214⟩, ⟨10488⟩⟩], []⟩, (1)⟩)

def exact60525RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9405⟩⟩, ⟨.program ⟨214⟩, ⟨10488⟩⟩], []⟩, (1)⟩]

theorem exact60525RawTermsValid :
    exact60525RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60525 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10489⟩⟩) exact60525RawTerms (.finite 4) 60523 .exactZero (none)

def event60526 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10490⟩⟩) 0 ⟨10489⟩ 60525

def event60527 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10490⟩⟩) (.identity (.predecessor 0 60526 .coefficient))

def event60528 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10490⟩⟩) (.finite 4)

def event60529 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14796⟩⟩) 0 ⟨10490⟩ 60528

def event60530 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14796⟩⟩) (.authority (.programFamilyFact))

def exact60531RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14796⟩⟩], []⟩, (1)⟩]

theorem exact60531RawTermsValid :
    exact60531RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60531 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14796⟩⟩) exact60531RawTerms (.finite 2) 60530 .exactZero (none)

def event60532 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14797⟩⟩) 0 ⟨14796⟩ 60531

def event60533 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14797⟩⟩) (.identity (.predecessor 0 60532 .coefficient))

def event60534 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14797⟩⟩) (.finite 2)

def event60535 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15268⟩⟩) 0 ⟨14797⟩ 60534

def event60536 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15268⟩⟩) (.authority (.programFamilyFact))

def exact60537RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩]

theorem exact60537RawTermsValid :
    exact60537RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60537 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15268⟩⟩) exact60537RawTerms (.finite 43) 60536 .exactZero (none)

def event60538 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15315⟩⟩) 0 ⟨15268⟩ 60537

def event60539 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15315⟩⟩) 1 ⟨15314⟩ 60514

def event60540 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15315⟩⟩) (.sum [.predecessor 0 60538 .coefficient, .predecessor 1 60539 .coefficient])

def exact60541RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩]

theorem exact60541RawTermsValid :
    exact60541RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60541 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15315⟩⟩) exact60541RawTerms (.finite 91) 60540 .exactZero (none)

def event60542 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15371⟩⟩) 0 ⟨15315⟩ 60541

def event60543 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15371⟩⟩) 1 ⟨15370⟩ 60491

def event60544 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15371⟩⟩) (.sum [.predecessor 0 60542 .coefficient, .predecessor 1 60543 .coefficient])

def exact60545RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], []⟩, (1)⟩]

theorem exact60545RawTermsValid :
    exact60545RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60545 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15371⟩⟩) exact60545RawTerms (.finite 142) 60544 .exactZero (none)

def event60546 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17337⟩⟩) 0 ⟨15371⟩ 60545

def event60547 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17337⟩⟩) 1 ⟨17336⟩ 60468

def event60548 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17337⟩⟩) (.sum [.predecessor 0 60546 .coefficient, .predecessor 1 60547 .coefficient])

def exact60549RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], []⟩, (1)⟩]

theorem exact60549RawTermsValid :
    exact60549RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60549 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17337⟩⟩) exact60549RawTerms (.finite 197) 60548 .exactZero (none)

def event60550 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17338⟩⟩) 0 ⟨17337⟩ 60549

def event60551 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17338⟩⟩) 1 ⟨15632⟩ 60445

def event60552 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17338⟩⟩) (.sum [.predecessor 0 60550 .coefficient, .predecessor 1 60551 .coefficient])

def exact60553RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], []⟩, (1)⟩]

theorem exact60553RawTermsValid :
    exact60553RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60553 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17338⟩⟩) exact60553RawTerms (.finite 255) 60552 .exactZero (none)

def event60554 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17339⟩⟩) 0 ⟨17338⟩ 60553

def event60555 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17339⟩⟩) 1 ⟨15751⟩ 60422

def event60556 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17339⟩⟩) (.sum [.predecessor 0 60554 .coefficient, .predecessor 1 60555 .coefficient])

def exact60557RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15751⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], []⟩, (1)⟩]

theorem exact60557RawTermsValid :
    exact60557RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60557 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17339⟩⟩) exact60557RawTerms (.finite 314) 60556 .exactZero (none)

def event60558 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17340⟩⟩) 0 ⟨17339⟩ 60557

def event60559 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17340⟩⟩) 1 ⟨15870⟩ 60399

def event60560 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17340⟩⟩) (.sum [.predecessor 0 60558 .coefficient, .predecessor 1 60559 .coefficient])

def exact60561RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15751⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15870⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], []⟩, (1)⟩]

theorem exact60561RawTermsValid :
    exact60561RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60561 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17340⟩⟩) exact60561RawTerms (.finite 374) 60560 .exactZero (none)

def event60562 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17341⟩⟩) 0 ⟨17340⟩ 60561

def event60563 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17341⟩⟩) 1 ⟨15989⟩ 60376

def event60564 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17341⟩⟩) (.sum [.predecessor 0 60562 .coefficient, .predecessor 1 60563 .coefficient])

def exact60565RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15751⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15870⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15989⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], []⟩, (1)⟩]

theorem exact60565RawTermsValid :
    exact60565RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60565 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17341⟩⟩) exact60565RawTerms (.finite 435) 60564 .exactZero (none)

def event60566 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17342⟩⟩) 0 ⟨17341⟩ 60565

def event60567 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17342⟩⟩) 1 ⟨16108⟩ 60353

def event60568 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17342⟩⟩) (.sum [.predecessor 0 60566 .coefficient, .predecessor 1 60567 .coefficient])

def exact60569RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15751⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15870⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15989⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], []⟩, (1)⟩]

theorem exact60569RawTermsValid :
    exact60569RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60569 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17342⟩⟩) exact60569RawTerms (.finite 496) 60568 .exactZero (none)

def event60570 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18354⟩⟩) 0 ⟨17342⟩ 60569

def event60571 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18354⟩⟩) 1 ⟨18353⟩ 60330

def event60572 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18354⟩⟩) (.sum [.predecessor 0 60570 .coefficient, .predecessor 1 60571 .coefficient])

def exact60573RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15751⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15870⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15989⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], []⟩, (1)⟩]

theorem exact60573RawTermsValid :
    exact60573RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60573 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18354⟩⟩) exact60573RawTerms (.finite 558) 60572 .exactZero (none)

def event60574 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18355⟩⟩) 0 ⟨18354⟩ 60573

def event60575 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18355⟩⟩) 1 ⟨16311⟩ 60307

def event60576 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18355⟩⟩) (.sum [.predecessor 0 60574 .coefficient, .predecessor 1 60575 .coefficient])

def exact60577RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15751⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15870⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15989⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16311⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], []⟩, (1)⟩]

theorem exact60577RawTermsValid :
    exact60577RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60577 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18355⟩⟩) exact60577RawTerms (.finite 620) 60576 .exactZero (none)

def event60578 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18356⟩⟩) 0 ⟨18355⟩ 60577

def event60579 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18356⟩⟩) 1 ⟨17123⟩ 60284

def event60580 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18356⟩⟩) (.sum [.predecessor 0 60578 .coefficient, .predecessor 1 60579 .coefficient])

def exact60581RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15751⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15870⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15989⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16311⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17123⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], []⟩, (1)⟩]

theorem exact60581RawTermsValid :
    exact60581RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60581 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18356⟩⟩) exact60581RawTerms (.finite 682) 60580 .exactZero (none)

def event60582 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18357⟩⟩) 0 ⟨18356⟩ 60581

def event60583 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18357⟩⟩) 1 ⟨17907⟩ 60261

def event60584 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18357⟩⟩) (.sum [.predecessor 0 60582 .coefficient, .predecessor 1 60583 .coefficient])

def exact60585RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15751⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15870⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15989⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16311⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17123⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17907⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], []⟩, (1)⟩]

theorem exact60585RawTermsValid :
    exact60585RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60585 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18357⟩⟩) exact60585RawTerms (.finite 744) 60584 .exactZero (none)

def event60586 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18358⟩⟩) 0 ⟨18357⟩ 60585

def event60587 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18358⟩⟩) 1 ⟨18208⟩ 60238

def event60588 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18358⟩⟩) (.sum [.predecessor 0 60586 .coefficient, .predecessor 1 60587 .coefficient])

def exact60589RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15751⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15870⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15989⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16311⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17123⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17907⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18208⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], []⟩, (1)⟩]

theorem exact60589RawTermsValid :
    exact60589RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60589 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18358⟩⟩) exact60589RawTerms (.finite 807) 60588 .exactZero (none)

def event60590 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18359⟩⟩) 0 ⟨18358⟩ 60589

def event60591 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18359⟩⟩) 1 ⟨16682⟩ 60215

def event60592 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18359⟩⟩) (.sum [.predecessor 0 60590 .coefficient, .predecessor 1 60591 .coefficient])

def exact60593RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15751⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15870⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15989⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16311⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16682⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17123⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17907⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18208⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], []⟩, (1)⟩]

theorem exact60593RawTermsValid :
    exact60593RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60593 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18359⟩⟩) exact60593RawTerms (.finite 870) 60592 .exactZero (none)

def event60594 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18360⟩⟩) 0 ⟨18359⟩ 60593

def event60595 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18360⟩⟩) 1 ⟨16801⟩ 60192

def event60596 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18360⟩⟩) (.sum [.predecessor 0 60594 .coefficient, .predecessor 1 60595 .coefficient])

def exact60597RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15751⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15870⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15989⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16311⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16682⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16801⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17123⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17907⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18208⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], []⟩, (1)⟩]

theorem exact60597RawTermsValid :
    exact60597RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60597 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18360⟩⟩) exact60597RawTerms (.finite 933) 60596 .exactZero (none)

def event60598 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18361⟩⟩) 0 ⟨18360⟩ 60597

def event60599 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18361⟩⟩) 1 ⟨17088⟩ 60169

def event60600 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18361⟩⟩) (.sum [.predecessor 0 60598 .coefficient, .predecessor 1 60599 .coefficient])

def exact60601RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15751⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15870⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15989⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16311⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16682⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16801⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17088⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17123⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17907⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18208⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], []⟩, (1)⟩]

theorem exact60601RawTermsValid :
    exact60601RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60601 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18361⟩⟩) exact60601RawTerms (.finite 996) 60600 .exactZero (none)

def event60602 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18362⟩⟩) 0 ⟨18361⟩ 60601

def event60603 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18362⟩⟩) 1 ⟨18173⟩ 60146

def event60604 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18362⟩⟩) (.sum [.predecessor 0 60602 .coefficient, .predecessor 1 60603 .coefficient])

def exact60605RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15751⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15870⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15989⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16311⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16682⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16801⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17088⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17123⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17907⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18173⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18208⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], []⟩, (1)⟩]

theorem exact60605RawTermsValid :
    exact60605RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60605 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18362⟩⟩) exact60605RawTerms (.finite 1059) 60604 .exactZero (none)

def event60606 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18363⟩⟩) 0 ⟨18362⟩ 60605

def event60607 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18363⟩⟩) (.identity (.predecessor 0 60606 .coefficient))

def event60608 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨18363⟩⟩) (.finite 1059)

def event60609 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18619⟩⟩) 0 ⟨18363⟩ 60608

def event60610 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18619⟩⟩) (.authority (.programFamilyFact))

def event60611 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨18619⟩⟩) (.finite 1152)

def event60612 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event60613 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18620⟩⟩) 0 ⟨6689⟩ 60612

def event60614 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18620⟩⟩) 1 ⟨18619⟩ 60611

def event60615 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18620⟩⟩) (.authority (.operator))

def exact60616RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩, (1)⟩]

theorem exact60616RawTermsValid :
    exact60616RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60616 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18620⟩⟩) exact60616RawTerms .large 60615 .exactZero (none)

def event60617 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18684⟩⟩) 0 ⟨18620⟩ 60616

def event60618 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18684⟩⟩) (.authority (.operator))

def exact60619RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩, (1)⟩]

theorem exact60619RawTermsValid :
    exact60619RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60619 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18684⟩⟩) exact60619RawTerms (.finite 8192) 60618 .exactZero (none)

def event60620 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event60621 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event60622 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18651⟩⟩) 0 ⟨18363⟩ 60608

def event60623 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18651⟩⟩) 1 ⟨110⟩ 60621

def event60624 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18651⟩⟩) (.sum [.predecessor 0 60622 .coefficient, .predecessor 1 60623 .coefficient])

def event60625 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨18651⟩⟩) (.finite 1059)

def event60626 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18652⟩⟩) 0 ⟨18651⟩ 60625

def event60627 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18652⟩⟩) (.identity (.predecessor 0 60626 .coefficient))

def exact60628RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15751⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15870⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15989⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16311⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16682⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16801⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17088⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17123⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17907⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18173⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18208⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], []⟩, (1)⟩]

theorem exact60628RawTermsValid :
    exact60628RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60628 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18652⟩⟩) exact60628RawTerms (.finite 1059) 60627 .exactZero (none)

def event60629 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact60630RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact60630RawTermsValid :
    exact60630RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60630 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact60630RawTerms .large 60629 .exactZero (none)

def event60631 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18653⟩⟩) 0 ⟨6544⟩ 60630

def event60632 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18653⟩⟩) 1 ⟨18652⟩ 60628

def event60633 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18653⟩⟩) (.product (.predecessor 0 60631 .coefficient) (.predecessor 1 60632 .coefficient) (⟨false, false, none, none, none⟩))

def event60634 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18653⟩⟩, .operator (⟨60630, 0⟩, ⟨60628, 15⟩), ⟨[⟨.program ⟨214⟩, ⟨18173⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event60635 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18653⟩⟩, .operator (⟨60630, 0⟩, ⟨60628, 11⟩), ⟨[⟨.program ⟨214⟩, ⟨17088⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event60636 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18653⟩⟩, .operator (⟨60630, 0⟩, ⟨60628, 10⟩), ⟨[⟨.program ⟨214⟩, ⟨16801⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event60637 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18653⟩⟩, .operator (⟨60630, 0⟩, ⟨60628, 9⟩), ⟨[⟨.program ⟨214⟩, ⟨16682⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event60638 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18653⟩⟩, .operator (⟨60630, 0⟩, ⟨60628, 16⟩), ⟨[⟨.program ⟨214⟩, ⟨18208⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event60639 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18653⟩⟩, .operator (⟨60630, 0⟩, ⟨60628, 14⟩), ⟨[⟨.program ⟨214⟩, ⟨17907⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event60640 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18653⟩⟩, .operator (⟨60630, 0⟩, ⟨60628, 12⟩), ⟨[⟨.program ⟨214⟩, ⟨17123⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event60641 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18653⟩⟩, .operator (⟨60630, 0⟩, ⟨60628, 8⟩), ⟨[⟨.program ⟨214⟩, ⟨16311⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event60642 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18653⟩⟩, .operator (⟨60630, 0⟩, ⟨60628, 17⟩), ⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event60643 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18653⟩⟩, .operator (⟨60630, 0⟩, ⟨60628, 7⟩), ⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event60644 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18653⟩⟩, .operator (⟨60630, 0⟩, ⟨60628, 6⟩), ⟨[⟨.program ⟨214⟩, ⟨15989⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event60645 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18653⟩⟩, .operator (⟨60630, 0⟩, ⟨60628, 5⟩), ⟨[⟨.program ⟨214⟩, ⟨15870⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event60646 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18653⟩⟩, .operator (⟨60630, 0⟩, ⟨60628, 4⟩), ⟨[⟨.program ⟨214⟩, ⟨15751⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event60647 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18653⟩⟩, .operator (⟨60630, 0⟩, ⟨60628, 3⟩), ⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event60648 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18653⟩⟩, .operator (⟨60630, 0⟩, ⟨60628, 13⟩), ⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event60649 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18653⟩⟩, .operator (⟨60630, 0⟩, ⟨60628, 2⟩), ⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event60650 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18653⟩⟩, .operator (⟨60630, 0⟩, ⟨60628, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def event60651 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18653⟩⟩, .operator (⟨60630, 0⟩, ⟨60628, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact60652RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15268⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15314⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15370⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15632⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15751⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15870⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15989⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16311⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16682⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16801⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17088⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17123⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17336⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17907⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18173⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18208⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact60652RawTermsValid :
    exact60652RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60652 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18653⟩⟩) exact60652RawTerms .large 60633 .exactZero (none)

def event60653 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6743⟩⟩) 0 ⟨6689⟩ 60612

def event60654 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6743⟩⟩) (.authority (.operator))

def exact60655RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩]

theorem exact60655RawTermsValid :
    exact60655RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60655 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6743⟩⟩) exact60655RawTerms .large 60654 .exactZero (none)

def event60656 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6741⟩⟩) 0 ⟨6689⟩ 60612

def event60657 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6741⟩⟩) (.authority (.operator))

def exact60658RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩]⟩, (1)⟩]

theorem exact60658RawTermsValid :
    exact60658RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60658 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6741⟩⟩) exact60658RawTerms .large 60657 .exactZero (none)

def event60659 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6739⟩⟩) 0 ⟨6689⟩ 60612

def event60660 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6739⟩⟩) (.authority (.operator))

def exact60661RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩]

theorem exact60661RawTermsValid :
    exact60661RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60661 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6739⟩⟩) exact60661RawTerms .large 60660 .exactZero (none)

def event60662 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6737⟩⟩) 0 ⟨6689⟩ 60612

def event60663 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6737⟩⟩) (.authority (.operator))

def exact60664RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩]⟩, (1)⟩]

theorem exact60664RawTermsValid :
    exact60664RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60664 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6737⟩⟩) exact60664RawTerms .large 60663 .exactZero (none)

def event60665 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6735⟩⟩) 0 ⟨6689⟩ 60612

def event60666 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6735⟩⟩) (.authority (.operator))

def exact60667RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩]⟩, (1)⟩]

theorem exact60667RawTermsValid :
    exact60667RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60667 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6735⟩⟩) exact60667RawTerms .large 60666 .exactZero (none)

def event60668 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6733⟩⟩) 0 ⟨6689⟩ 60612

def event60669 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6733⟩⟩) (.authority (.operator))

def exact60670RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩]

theorem exact60670RawTermsValid :
    exact60670RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60670 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6733⟩⟩) exact60670RawTerms .large 60669 .exactZero (none)

def event60671 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6731⟩⟩) 0 ⟨6689⟩ 60612

def eventLeaf3776 : Array AnnotatedEvent := #[
  { event := event60416
    frameStart := 60103 },
  { event := event60417
    frameStart := 60103 },
  { event := event60418
    frameStart := 60103 },
  { event := event60419
    frameStart := 60103 },
  { event := event60420
    frameStart := 60103 },
  { event := event60421
    frameStart := 60103 },
  { event := event60422
    frameStart := 60103 },
  { event := event60423
    frameStart := 60103 },
  { event := event60424
    frameStart := 60103 },
  { event := event60425
    frameStart := 60103 },
  { event := event60426
    frameStart := 60103 },
  { event := event60427
    frameStart := 60103 },
  { event := event60428
    frameStart := 60103 },
  { event := event60429
    frameStart := 60103 },
  { event := event60430
    frameStart := 60103 },
  { event := event60431
    frameStart := 60103 }
]

def eventLeaf3777 : Array AnnotatedEvent := #[
  { event := event60432
    frameStart := 60103 },
  { event := event60433
    frameStart := 60103 },
  { event := event60434
    frameStart := 60103 },
  { event := event60435
    frameStart := 60103 },
  { event := event60436
    frameStart := 60103 },
  { event := event60437
    frameStart := 60103 },
  { event := event60438
    frameStart := 60103 },
  { event := event60439
    frameStart := 60103 },
  { event := event60440
    frameStart := 60103 },
  { event := event60441
    frameStart := 60103 },
  { event := event60442
    frameStart := 60103 },
  { event := event60443
    frameStart := 60103 },
  { event := event60444
    frameStart := 60103 },
  { event := event60445
    frameStart := 60103 },
  { event := event60446
    frameStart := 60103 },
  { event := event60447
    frameStart := 60103 }
]

def eventLeaf3778 : Array AnnotatedEvent := #[
  { event := event60448
    frameStart := 60103 },
  { event := event60449
    frameStart := 60103 },
  { event := event60450
    frameStart := 60103 },
  { event := event60451
    frameStart := 60103 },
  { event := event60452
    frameStart := 60103 },
  { event := event60453
    frameStart := 60103 },
  { event := event60454
    frameStart := 60103 },
  { event := event60455
    frameStart := 60103 },
  { event := event60456
    frameStart := 60103 },
  { event := event60457
    frameStart := 60103 },
  { event := event60458
    frameStart := 60103 },
  { event := event60459
    frameStart := 60103 },
  { event := event60460
    frameStart := 60103 },
  { event := event60461
    frameStart := 60103 },
  { event := event60462
    frameStart := 60103 },
  { event := event60463
    frameStart := 60103 }
]

def eventLeaf3779 : Array AnnotatedEvent := #[
  { event := event60464
    frameStart := 60103 },
  { event := event60465
    frameStart := 60103 },
  { event := event60466
    frameStart := 60103 },
  { event := event60467
    frameStart := 60103 },
  { event := event60468
    frameStart := 60103 },
  { event := event60469
    frameStart := 60103 },
  { event := event60470
    frameStart := 60103 },
  { event := event60471
    frameStart := 60103 },
  { event := event60472
    frameStart := 60103 },
  { event := event60473
    frameStart := 60103 },
  { event := event60474
    frameStart := 60103 },
  { event := event60475
    frameStart := 60103 },
  { event := event60476
    frameStart := 60103 },
  { event := event60477
    frameStart := 60103 },
  { event := event60478
    frameStart := 60103 },
  { event := event60479
    frameStart := 60103 }
]

def eventLeaf3780 : Array AnnotatedEvent := #[
  { event := event60480
    frameStart := 60103 },
  { event := event60481
    frameStart := 60103 },
  { event := event60482
    frameStart := 60103 },
  { event := event60483
    frameStart := 60103 },
  { event := event60484
    frameStart := 60103 },
  { event := event60485
    frameStart := 60103 },
  { event := event60486
    frameStart := 60103 },
  { event := event60487
    frameStart := 60103 },
  { event := event60488
    frameStart := 60103 },
  { event := event60489
    frameStart := 60103 },
  { event := event60490
    frameStart := 60103 },
  { event := event60491
    frameStart := 60103 },
  { event := event60492
    frameStart := 60103 },
  { event := event60493
    frameStart := 60103 },
  { event := event60494
    frameStart := 60103 },
  { event := event60495
    frameStart := 60103 }
]

def eventLeaf3781 : Array AnnotatedEvent := #[
  { event := event60496
    frameStart := 60103 },
  { event := event60497
    frameStart := 60103 },
  { event := event60498
    frameStart := 60103 },
  { event := event60499
    frameStart := 60103 },
  { event := event60500
    frameStart := 60103 },
  { event := event60501
    frameStart := 60103 },
  { event := event60502
    frameStart := 60103 },
  { event := event60503
    frameStart := 60103 },
  { event := event60504
    frameStart := 60103 },
  { event := event60505
    frameStart := 60103 },
  { event := event60506
    frameStart := 60103 },
  { event := event60507
    frameStart := 60103 },
  { event := event60508
    frameStart := 60103 },
  { event := event60509
    frameStart := 60103 },
  { event := event60510
    frameStart := 60103 },
  { event := event60511
    frameStart := 60103 }
]

def eventLeaf3782 : Array AnnotatedEvent := #[
  { event := event60512
    frameStart := 60103 },
  { event := event60513
    frameStart := 60103 },
  { event := event60514
    frameStart := 60103 },
  { event := event60515
    frameStart := 60103 },
  { event := event60516
    frameStart := 60103 },
  { event := event60517
    frameStart := 60103 },
  { event := event60518
    frameStart := 60103 },
  { event := event60519
    frameStart := 60103 },
  { event := event60520
    frameStart := 60103 },
  { event := event60521
    frameStart := 60103 },
  { event := event60522
    frameStart := 60103 },
  { event := event60523
    frameStart := 60103 },
  { event := event60524
    frameStart := 60103 },
  { event := event60525
    frameStart := 60103 },
  { event := event60526
    frameStart := 60103 },
  { event := event60527
    frameStart := 60103 }
]

def eventLeaf3783 : Array AnnotatedEvent := #[
  { event := event60528
    frameStart := 60103 },
  { event := event60529
    frameStart := 60103 },
  { event := event60530
    frameStart := 60103 },
  { event := event60531
    frameStart := 60103 },
  { event := event60532
    frameStart := 60103 },
  { event := event60533
    frameStart := 60103 },
  { event := event60534
    frameStart := 60103 },
  { event := event60535
    frameStart := 60103 },
  { event := event60536
    frameStart := 60103 },
  { event := event60537
    frameStart := 60103 },
  { event := event60538
    frameStart := 60103 },
  { event := event60539
    frameStart := 60103 },
  { event := event60540
    frameStart := 60103 },
  { event := event60541
    frameStart := 60103 },
  { event := event60542
    frameStart := 60103 },
  { event := event60543
    frameStart := 60103 }
]

def eventLeaf3784 : Array AnnotatedEvent := #[
  { event := event60544
    frameStart := 60103 },
  { event := event60545
    frameStart := 60103 },
  { event := event60546
    frameStart := 60103 },
  { event := event60547
    frameStart := 60103 },
  { event := event60548
    frameStart := 60103 },
  { event := event60549
    frameStart := 60103 },
  { event := event60550
    frameStart := 60103 },
  { event := event60551
    frameStart := 60103 },
  { event := event60552
    frameStart := 60103 },
  { event := event60553
    frameStart := 60103 },
  { event := event60554
    frameStart := 60103 },
  { event := event60555
    frameStart := 60103 },
  { event := event60556
    frameStart := 60103 },
  { event := event60557
    frameStart := 60103 },
  { event := event60558
    frameStart := 60103 },
  { event := event60559
    frameStart := 60103 }
]

def eventLeaf3785 : Array AnnotatedEvent := #[
  { event := event60560
    frameStart := 60103 },
  { event := event60561
    frameStart := 60103 },
  { event := event60562
    frameStart := 60103 },
  { event := event60563
    frameStart := 60103 },
  { event := event60564
    frameStart := 60103 },
  { event := event60565
    frameStart := 60103 },
  { event := event60566
    frameStart := 60103 },
  { event := event60567
    frameStart := 60103 },
  { event := event60568
    frameStart := 60103 },
  { event := event60569
    frameStart := 60103 },
  { event := event60570
    frameStart := 60103 },
  { event := event60571
    frameStart := 60103 },
  { event := event60572
    frameStart := 60103 },
  { event := event60573
    frameStart := 60103 },
  { event := event60574
    frameStart := 60103 },
  { event := event60575
    frameStart := 60103 }
]

def eventLeaf3786 : Array AnnotatedEvent := #[
  { event := event60576
    frameStart := 60103 },
  { event := event60577
    frameStart := 60103 },
  { event := event60578
    frameStart := 60103 },
  { event := event60579
    frameStart := 60103 },
  { event := event60580
    frameStart := 60103 },
  { event := event60581
    frameStart := 60103 },
  { event := event60582
    frameStart := 60103 },
  { event := event60583
    frameStart := 60103 },
  { event := event60584
    frameStart := 60103 },
  { event := event60585
    frameStart := 60103 },
  { event := event60586
    frameStart := 60103 },
  { event := event60587
    frameStart := 60103 },
  { event := event60588
    frameStart := 60103 },
  { event := event60589
    frameStart := 60103 },
  { event := event60590
    frameStart := 60103 },
  { event := event60591
    frameStart := 60103 }
]

def eventLeaf3787 : Array AnnotatedEvent := #[
  { event := event60592
    frameStart := 60103 },
  { event := event60593
    frameStart := 60103 },
  { event := event60594
    frameStart := 60103 },
  { event := event60595
    frameStart := 60103 },
  { event := event60596
    frameStart := 60103 },
  { event := event60597
    frameStart := 60103 },
  { event := event60598
    frameStart := 60103 },
  { event := event60599
    frameStart := 60103 },
  { event := event60600
    frameStart := 60103 },
  { event := event60601
    frameStart := 60103 },
  { event := event60602
    frameStart := 60103 },
  { event := event60603
    frameStart := 60103 },
  { event := event60604
    frameStart := 60103 },
  { event := event60605
    frameStart := 60103 },
  { event := event60606
    frameStart := 60103 },
  { event := event60607
    frameStart := 60103 }
]

def eventLeaf3788 : Array AnnotatedEvent := #[
  { event := event60608
    frameStart := 60103 },
  { event := event60609
    frameStart := 60103 },
  { event := event60610
    frameStart := 60103 },
  { event := event60611
    frameStart := 60103 },
  { event := event60612
    frameStart := 60103 },
  { event := event60613
    frameStart := 60103 },
  { event := event60614
    frameStart := 60103 },
  { event := event60615
    frameStart := 60103 },
  { event := event60616
    frameStart := 60103 },
  { event := event60617
    frameStart := 60103 },
  { event := event60618
    frameStart := 60103 },
  { event := event60619
    frameStart := 60103 },
  { event := event60620
    frameStart := 60103 },
  { event := event60621
    frameStart := 60103 },
  { event := event60622
    frameStart := 60103 },
  { event := event60623
    frameStart := 60103 }
]

def eventLeaf3789 : Array AnnotatedEvent := #[
  { event := event60624
    frameStart := 60103 },
  { event := event60625
    frameStart := 60103 },
  { event := event60626
    frameStart := 60103 },
  { event := event60627
    frameStart := 60103 },
  { event := event60628
    frameStart := 60103 },
  { event := event60629
    frameStart := 60103 },
  { event := event60630
    frameStart := 60103 },
  { event := event60631
    frameStart := 60103 },
  { event := event60632
    frameStart := 60103 },
  { event := event60633
    frameStart := 60103 },
  { event := event60634
    frameStart := 60103 },
  { event := event60635
    frameStart := 60103 },
  { event := event60636
    frameStart := 60103 },
  { event := event60637
    frameStart := 60103 },
  { event := event60638
    frameStart := 60103 },
  { event := event60639
    frameStart := 60103 }
]

def eventLeaf3790 : Array AnnotatedEvent := #[
  { event := event60640
    frameStart := 60103 },
  { event := event60641
    frameStart := 60103 },
  { event := event60642
    frameStart := 60103 },
  { event := event60643
    frameStart := 60103 },
  { event := event60644
    frameStart := 60103 },
  { event := event60645
    frameStart := 60103 },
  { event := event60646
    frameStart := 60103 },
  { event := event60647
    frameStart := 60103 },
  { event := event60648
    frameStart := 60103 },
  { event := event60649
    frameStart := 60103 },
  { event := event60650
    frameStart := 60103 },
  { event := event60651
    frameStart := 60103 },
  { event := event60652
    frameStart := 60103 },
  { event := event60653
    frameStart := 60103 },
  { event := event60654
    frameStart := 60103 },
  { event := event60655
    frameStart := 60103 }
]

def eventLeaf3791 : Array AnnotatedEvent := #[
  { event := event60656
    frameStart := 60103 },
  { event := event60657
    frameStart := 60103 },
  { event := event60658
    frameStart := 60103 },
  { event := event60659
    frameStart := 60103 },
  { event := event60660
    frameStart := 60103 },
  { event := event60661
    frameStart := 60103 },
  { event := event60662
    frameStart := 60103 },
  { event := event60663
    frameStart := 60103 },
  { event := event60664
    frameStart := 60103 },
  { event := event60665
    frameStart := 60103 },
  { event := event60666
    frameStart := 60103 },
  { event := event60667
    frameStart := 60103 },
  { event := event60668
    frameStart := 60103 },
  { event := event60669
    frameStart := 60103 },
  { event := event60670
    frameStart := 60103 },
  { event := event60671
    frameStart := 60103 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events236
