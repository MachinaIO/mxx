import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events275

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact70400RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨59884⟩⟩], []⟩, (1)⟩]

theorem exact70400RawTermsValid :
    exact70400RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70400 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59884⟩⟩) exact70400RawTerms (.finite 18) 70399 .exactZero (none)

def event70401 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59885⟩⟩) 0 ⟨59884⟩ 70400

def event70402 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59885⟩⟩) (.identity (.predecessor 0 70401 .coefficient))

def event70403 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨59885⟩⟩) (.finite 18)

def event70404 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60234⟩⟩) 0 ⟨59885⟩ 70403

def event70405 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60234⟩⟩) (.authority (.programFamilyFact))

def exact70406RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], []⟩, (1)⟩]

theorem exact70406RawTermsValid :
    exact70406RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70406 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60234⟩⟩) exact70406RawTerms (.finite 61) 70405 .exactZero (none)

def event70407 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25094⟩⟩) 0 ⟨10749⟩ 70142

def event70408 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25094⟩⟩) (.authority (.programFamilyFact))

def exact70409RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25094⟩⟩], []⟩, (1)⟩]

theorem exact70409RawTermsValid :
    exact70409RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70409 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25094⟩⟩) exact70409RawTerms (.finite 16) 70408 .exactZero (none)

def event70410 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56694⟩⟩) 0 ⟨10749⟩ 70142

def event70411 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56694⟩⟩) (.authority (.programFamilyFact))

def exact70412RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56694⟩⟩], []⟩, (1)⟩]

theorem exact70412RawTermsValid :
    exact70412RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70412 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56694⟩⟩) exact70412RawTerms (.finite 16) 70411 .exactZero (none)

def event70413 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56695⟩⟩) 0 ⟨56694⟩ 70412

def event70414 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56695⟩⟩) 1 ⟨25094⟩ 70409

def event70415 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56695⟩⟩) (.product (.predecessor 0 70413 .coefficient) (.predecessor 1 70414 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70416 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56695⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25094⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], []⟩) [⟨.result 70412 .coefficient, true, some 1⟩, ⟨.result 70409 .coefficient, true, some 1⟩])

def event70417 : Event := .survivorFold (1) 70416

def exact70418RawTerms : List Term := []

theorem exact70418RawTermsValid :
    exact70418RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70418 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56695⟩⟩) exact70418RawTerms (.finite 256) 70415 (.finite 256) (some (70416))

def event70419 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56696⟩⟩) 0 ⟨56695⟩ 70418

def event70420 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56696⟩⟩) (.identity (.predecessor 0 70419 .coefficient))

def event70421 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56696⟩⟩) (.finite 256)

def event70422 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56904⟩⟩) 0 ⟨56696⟩ 70421

def event70423 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56904⟩⟩) (.authority (.programFamilyFact))

def exact70424RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56904⟩⟩], []⟩, (1)⟩]

theorem exact70424RawTermsValid :
    exact70424RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70424 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56904⟩⟩) exact70424RawTerms (.finite 16) 70423 .exactZero (none)

def event70425 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56905⟩⟩) 0 ⟨56904⟩ 70424

def event70426 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56905⟩⟩) (.identity (.predecessor 0 70425 .coefficient))

def event70427 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56905⟩⟩) (.finite 16)

def event70428 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57254⟩⟩) 0 ⟨56905⟩ 70427

def event70429 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57254⟩⟩) (.authority (.programFamilyFact))

def exact70430RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], []⟩, (1)⟩]

theorem exact70430RawTermsValid :
    exact70430RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70430 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57254⟩⟩) exact70430RawTerms (.finite 60) 70429 .exactZero (none)

def event70431 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24854⟩⟩) 0 ⟨10749⟩ 70142

def event70432 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24854⟩⟩) (.authority (.programFamilyFact))

def exact70433RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24854⟩⟩], []⟩, (1)⟩]

theorem exact70433RawTermsValid :
    exact70433RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70433 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24854⟩⟩) exact70433RawTerms (.finite 12) 70432 .exactZero (none)

def event70434 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53714⟩⟩) 0 ⟨10749⟩ 70142

def event70435 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53714⟩⟩) (.authority (.programFamilyFact))

def exact70436RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53714⟩⟩], []⟩, (1)⟩]

theorem exact70436RawTermsValid :
    exact70436RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70436 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53714⟩⟩) exact70436RawTerms (.finite 12) 70435 .exactZero (none)

def event70437 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53715⟩⟩) 0 ⟨53714⟩ 70436

def event70438 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53715⟩⟩) 1 ⟨24854⟩ 70433

def event70439 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53715⟩⟩) (.product (.predecessor 0 70437 .coefficient) (.predecessor 1 70438 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70440 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53715⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24854⟩⟩, ⟨.program ⟨257⟩, ⟨53714⟩⟩], []⟩) [⟨.result 70436 .coefficient, true, some 1⟩, ⟨.result 70433 .coefficient, true, some 1⟩])

def event70441 : Event := .survivorFold (1) 70440

def exact70442RawTerms : List Term := []

theorem exact70442RawTermsValid :
    exact70442RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70442 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53715⟩⟩) exact70442RawTerms (.finite 144) 70439 (.finite 144) (some (70440))

def event70443 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53716⟩⟩) 0 ⟨53715⟩ 70442

def event70444 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53716⟩⟩) (.identity (.predecessor 0 70443 .coefficient))

def event70445 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53716⟩⟩) (.finite 144)

def event70446 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53924⟩⟩) 0 ⟨53716⟩ 70445

def event70447 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53924⟩⟩) (.authority (.programFamilyFact))

def exact70448RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53924⟩⟩], []⟩, (1)⟩]

theorem exact70448RawTermsValid :
    exact70448RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70448 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53924⟩⟩) exact70448RawTerms (.finite 12) 70447 .exactZero (none)

def event70449 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53925⟩⟩) 0 ⟨53924⟩ 70448

def event70450 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53925⟩⟩) (.identity (.predecessor 0 70449 .coefficient))

def event70451 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53925⟩⟩) (.finite 12)

def event70452 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54274⟩⟩) 0 ⟨53925⟩ 70451

def event70453 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54274⟩⟩) (.authority (.programFamilyFact))

def exact70454RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], []⟩, (1)⟩]

theorem exact70454RawTermsValid :
    exact70454RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70454 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54274⟩⟩) exact70454RawTerms (.finite 59) 70453 .exactZero (none)

def event70455 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24614⟩⟩) 0 ⟨10749⟩ 70142

def event70456 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24614⟩⟩) (.authority (.programFamilyFact))

def exact70457RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24614⟩⟩], []⟩, (1)⟩]

theorem exact70457RawTermsValid :
    exact70457RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70457 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24614⟩⟩) exact70457RawTerms (.finite 10) 70456 .exactZero (none)

def event70458 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50734⟩⟩) 0 ⟨10749⟩ 70142

def event70459 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50734⟩⟩) (.authority (.programFamilyFact))

def exact70460RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50734⟩⟩], []⟩, (1)⟩]

theorem exact70460RawTermsValid :
    exact70460RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70460 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50734⟩⟩) exact70460RawTerms (.finite 10) 70459 .exactZero (none)

def event70461 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50735⟩⟩) 0 ⟨50734⟩ 70460

def event70462 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50735⟩⟩) 1 ⟨24614⟩ 70457

def event70463 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50735⟩⟩) (.product (.predecessor 0 70461 .coefficient) (.predecessor 1 70462 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70464 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50735⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], []⟩) [⟨.result 70460 .coefficient, true, some 1⟩, ⟨.result 70457 .coefficient, true, some 1⟩])

def event70465 : Event := .survivorFold (1) 70464

def exact70466RawTerms : List Term := []

theorem exact70466RawTermsValid :
    exact70466RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70466 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50735⟩⟩) exact70466RawTerms (.finite 100) 70463 (.finite 100) (some (70464))

def event70467 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50736⟩⟩) 0 ⟨50735⟩ 70466

def event70468 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50736⟩⟩) (.identity (.predecessor 0 70467 .coefficient))

def event70469 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50736⟩⟩) (.finite 100)

def event70470 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50944⟩⟩) 0 ⟨50736⟩ 70469

def event70471 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50944⟩⟩) (.authority (.programFamilyFact))

def exact70472RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], []⟩, (1)⟩]

theorem exact70472RawTermsValid :
    exact70472RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70472 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50944⟩⟩) exact70472RawTerms (.finite 10) 70471 .exactZero (none)

def event70473 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50945⟩⟩) 0 ⟨50944⟩ 70472

def event70474 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50945⟩⟩) (.identity (.predecessor 0 70473 .coefficient))

def event70475 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50945⟩⟩) (.finite 10)

def event70476 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51294⟩⟩) 0 ⟨50945⟩ 70475

def event70477 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51294⟩⟩) (.authority (.programFamilyFact))

def exact70478RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], []⟩, (1)⟩]

theorem exact70478RawTermsValid :
    exact70478RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70478 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51294⟩⟩) exact70478RawTerms (.finite 58) 70477 .exactZero (none)

def event70479 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24374⟩⟩) 0 ⟨10749⟩ 70142

def event70480 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24374⟩⟩) (.authority (.programFamilyFact))

def exact70481RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩], []⟩, (1)⟩]

theorem exact70481RawTermsValid :
    exact70481RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70481 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24374⟩⟩) exact70481RawTerms (.finite 6) 70480 .exactZero (none)

def event70482 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31674⟩⟩) 0 ⟨10749⟩ 70142

def event70483 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31674⟩⟩) (.authority (.programFamilyFact))

def exact70484RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31674⟩⟩], []⟩, (1)⟩]

theorem exact70484RawTermsValid :
    exact70484RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70484 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31674⟩⟩) exact70484RawTerms (.finite 6) 70483 .exactZero (none)

def event70485 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31675⟩⟩) 0 ⟨31674⟩ 70484

def event70486 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31675⟩⟩) 1 ⟨24374⟩ 70481

def event70487 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31675⟩⟩) (.product (.predecessor 0 70485 .coefficient) (.predecessor 1 70486 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70488 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31675⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], []⟩) [⟨.result 70484 .coefficient, true, some 1⟩, ⟨.result 70481 .coefficient, true, some 1⟩])

def event70489 : Event := .survivorFold (1) 70488

def exact70490RawTerms : List Term := []

theorem exact70490RawTermsValid :
    exact70490RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70490 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31675⟩⟩) exact70490RawTerms (.finite 36) 70487 (.finite 36) (some (70488))

def event70491 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31676⟩⟩) 0 ⟨31675⟩ 70490

def event70492 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31676⟩⟩) (.identity (.predecessor 0 70491 .coefficient))

def event70493 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31676⟩⟩) (.finite 36)

def event70494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31884⟩⟩) 0 ⟨31676⟩ 70493

def event70495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31884⟩⟩) (.authority (.programFamilyFact))

def exact70496RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31884⟩⟩], []⟩, (1)⟩]

theorem exact70496RawTermsValid :
    exact70496RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70496 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31884⟩⟩) exact70496RawTerms (.finite 6) 70495 .exactZero (none)

def event70497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31885⟩⟩) 0 ⟨31884⟩ 70496

def event70498 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31885⟩⟩) (.identity (.predecessor 0 70497 .coefficient))

def event70499 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31885⟩⟩) (.finite 6)

def event70500 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32239⟩⟩) 0 ⟨31885⟩ 70499

def event70501 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32239⟩⟩) (.authority (.programFamilyFact))

def exact70502RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], []⟩, (1)⟩]

theorem exact70502RawTermsValid :
    exact70502RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70502 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32239⟩⟩) exact70502RawTerms (.finite 55) 70501 .exactZero (none)

def event70503 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21662⟩⟩) 0 ⟨10749⟩ 70142

def event70504 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21662⟩⟩) (.authority (.programFamilyFact))

def exact70505RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21662⟩⟩], []⟩, (1)⟩]

theorem exact70505RawTermsValid :
    exact70505RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70505 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21662⟩⟩) exact70505RawTerms (.finite 4) 70504 .exactZero (none)

def event70506 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21206⟩⟩) 0 ⟨10749⟩ 70142

def event70507 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21206⟩⟩) (.authority (.programFamilyFact))

def exact70508RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩], []⟩, (1)⟩]

theorem exact70508RawTermsValid :
    exact70508RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70508 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21206⟩⟩) exact70508RawTerms (.finite 4) 70507 .exactZero (none)

def event70509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21663⟩⟩) 0 ⟨21206⟩ 70508

def event70510 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21663⟩⟩) 1 ⟨21662⟩ 70505

def event70511 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21663⟩⟩) (.product (.predecessor 0 70509 .coefficient) (.predecessor 1 70510 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70512 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21663⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩, ⟨.program ⟨257⟩, ⟨21662⟩⟩], []⟩) [⟨.result 70508 .coefficient, true, some 1⟩, ⟨.result 70505 .coefficient, true, some 1⟩])

def event70513 : Event := .survivorFold (1) 70512

def exact70514RawTerms : List Term := []

theorem exact70514RawTermsValid :
    exact70514RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70514 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21663⟩⟩) exact70514RawTerms (.finite 16) 70511 (.finite 16) (some (70512))

def event70515 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21664⟩⟩) 0 ⟨21663⟩ 70514

def event70516 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21664⟩⟩) (.identity (.predecessor 0 70515 .coefficient))

def event70517 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21664⟩⟩) (.finite 16)

def event70518 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21864⟩⟩) 0 ⟨21664⟩ 70517

def event70519 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21864⟩⟩) (.authority (.programFamilyFact))

def exact70520RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21864⟩⟩], []⟩, (1)⟩]

theorem exact70520RawTermsValid :
    exact70520RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70520 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21864⟩⟩) exact70520RawTerms (.finite 4) 70519 .exactZero (none)

def event70521 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21865⟩⟩) 0 ⟨21864⟩ 70520

def event70522 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21865⟩⟩) (.identity (.predecessor 0 70521 .coefficient))

def event70523 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21865⟩⟩) (.finite 4)

def event70524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22219⟩⟩) 0 ⟨21865⟩ 70523

def event70525 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22219⟩⟩) (.authority (.programFamilyFact))

def exact70526RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], []⟩, (1)⟩]

theorem exact70526RawTermsValid :
    exact70526RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70526 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22219⟩⟩) exact70526RawTerms (.finite 51) 70525 .exactZero (none)

def event70527 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18442⟩⟩) 0 ⟨10749⟩ 70142

def event70528 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18442⟩⟩) (.authority (.programFamilyFact))

def exact70529RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18442⟩⟩], []⟩, (1)⟩]

theorem exact70529RawTermsValid :
    exact70529RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70529 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18442⟩⟩) exact70529RawTerms (.finite 3) 70528 .exactZero (none)

def event70530 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12786⟩⟩) 0 ⟨10749⟩ 70142

def event70531 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12786⟩⟩) (.authority (.programFamilyFact))

def exact70532RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12786⟩⟩], []⟩, (1)⟩]

theorem exact70532RawTermsValid :
    exact70532RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70532 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12786⟩⟩) exact70532RawTerms (.finite 3) 70531 .exactZero (none)

def event70533 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18443⟩⟩) 0 ⟨12786⟩ 70532

def event70534 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18443⟩⟩) 1 ⟨18442⟩ 70529

def event70535 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18443⟩⟩) (.product (.predecessor 0 70533 .coefficient) (.predecessor 1 70534 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70536 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18443⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], []⟩) [⟨.result 70532 .coefficient, true, some 1⟩, ⟨.result 70529 .coefficient, true, some 1⟩])

def event70537 : Event := .survivorFold (1) 70536

def exact70538RawTerms : List Term := []

theorem exact70538RawTermsValid :
    exact70538RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70538 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18443⟩⟩) exact70538RawTerms (.finite 9) 70535 (.finite 9) (some (70536))

def event70539 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18444⟩⟩) 0 ⟨18443⟩ 70538

def event70540 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18444⟩⟩) (.identity (.predecessor 0 70539 .coefficient))

def event70541 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18444⟩⟩) (.finite 9)

def event70542 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18644⟩⟩) 0 ⟨18444⟩ 70541

def event70543 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18644⟩⟩) (.authority (.programFamilyFact))

def exact70544RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18644⟩⟩], []⟩, (1)⟩]

theorem exact70544RawTermsValid :
    exact70544RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70544 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18644⟩⟩) exact70544RawTerms (.finite 3) 70543 .exactZero (none)

def event70545 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18645⟩⟩) 0 ⟨18644⟩ 70544

def event70546 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18645⟩⟩) (.identity (.predecessor 0 70545 .coefficient))

def event70547 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18645⟩⟩) (.finite 3)

def event70548 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18999⟩⟩) 0 ⟨18645⟩ 70547

def event70549 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18999⟩⟩) (.authority (.programFamilyFact))

def exact70550RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], []⟩, (1)⟩]

theorem exact70550RawTermsValid :
    exact70550RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70550 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18999⟩⟩) exact70550RawTerms (.finite 48) 70549 .exactZero (none)

def event70551 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15642⟩⟩) 0 ⟨10749⟩ 70142

def event70552 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15642⟩⟩) (.authority (.programFamilyFact))

def exact70553RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15642⟩⟩], []⟩, (1)⟩]

theorem exact70553RawTermsValid :
    exact70553RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70553 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15642⟩⟩) exact70553RawTerms (.finite 2) 70552 .exactZero (none)

def event70554 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12486⟩⟩) 0 ⟨10749⟩ 70142

def event70555 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12486⟩⟩) (.authority (.programFamilyFact))

def exact70556RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12486⟩⟩], []⟩, (1)⟩]

theorem exact70556RawTermsValid :
    exact70556RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70556 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12486⟩⟩) exact70556RawTerms (.finite 2) 70555 .exactZero (none)

def event70557 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15643⟩⟩) 0 ⟨12486⟩ 70556

def event70558 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15643⟩⟩) 1 ⟨15642⟩ 70553

def event70559 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15643⟩⟩) (.product (.predecessor 0 70557 .coefficient) (.predecessor 1 70558 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70560 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15643⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12486⟩⟩, ⟨.program ⟨257⟩, ⟨15642⟩⟩], []⟩) [⟨.result 70556 .coefficient, true, some 1⟩, ⟨.result 70553 .coefficient, true, some 1⟩])

def event70561 : Event := .survivorFold (1) 70560

def exact70562RawTerms : List Term := []

theorem exact70562RawTermsValid :
    exact70562RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70562 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15643⟩⟩) exact70562RawTerms (.finite 4) 70559 (.finite 4) (some (70560))

def event70563 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15644⟩⟩) 0 ⟨15643⟩ 70562

def event70564 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15644⟩⟩) (.identity (.predecessor 0 70563 .coefficient))

def event70565 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15644⟩⟩) (.finite 4)

def event70566 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15844⟩⟩) 0 ⟨15644⟩ 70565

def event70567 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15844⟩⟩) (.authority (.programFamilyFact))

def exact70568RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], []⟩, (1)⟩]

theorem exact70568RawTermsValid :
    exact70568RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70568 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15844⟩⟩) exact70568RawTerms (.finite 2) 70567 .exactZero (none)

def event70569 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15845⟩⟩) 0 ⟨15844⟩ 70568

def event70570 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15845⟩⟩) (.identity (.predecessor 0 70569 .coefficient))

def event70571 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15845⟩⟩) (.finite 2)

def event70572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16147⟩⟩) 0 ⟨15845⟩ 70571

def event70573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16147⟩⟩) (.authority (.programFamilyFact))

def exact70574RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], []⟩, (1)⟩]

theorem exact70574RawTermsValid :
    exact70574RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70574 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16147⟩⟩) exact70574RawTerms (.finite 43) 70573 .exactZero (none)

def event70575 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19000⟩⟩) 0 ⟨16147⟩ 70574

def event70576 : Event := .predecessor (⟨.program ⟨257⟩, ⟨19000⟩⟩) 1 ⟨18999⟩ 70550

def event70577 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19000⟩⟩) (.sum [.predecessor 0 70575 .coefficient, .predecessor 1 70576 .coefficient])

def event70578 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19000⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨18999⟩⟩], []⟩) [⟨.result 70550 .coefficient, true, some 1⟩])

def event70579 : Event := .survivorFold (1) 70578

def event70580 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19000⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨16147⟩⟩], []⟩) [⟨.result 70574 .coefficient, true, some 1⟩])

def event70581 : Event := .survivorFold (1) 70580

def event70582 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨19000⟩⟩) (.sum [.transfer 70578, .transfer 70580])

def exact70583RawTerms : List Term := []

theorem exact70583RawTermsValid :
    exact70583RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70583 : Event := .resultExact (⟨.program ⟨257⟩, ⟨19000⟩⟩) exact70583RawTerms (.finite 91) 70577 (.finite 91) (some (70582))

def event70584 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22220⟩⟩) 0 ⟨19000⟩ 70583

def event70585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22220⟩⟩) 1 ⟨22219⟩ 70526

def event70586 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22220⟩⟩) (.sum [.predecessor 0 70584 .coefficient, .predecessor 1 70585 .coefficient])

def event70587 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22220⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨22219⟩⟩], []⟩) [⟨.result 70526 .coefficient, true, some 1⟩])

def event70588 : Event := .survivorFold (1) 70587

def event70589 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22220⟩⟩) (.sum [.result 70583 .summary, .transfer 70587])

def exact70590RawTerms : List Term := []

theorem exact70590RawTermsValid :
    exact70590RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70590 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22220⟩⟩) exact70590RawTerms (.finite 142) 70586 (.finite 142) (some (70589))

def event70591 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32240⟩⟩) 0 ⟨22220⟩ 70590

def event70592 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32240⟩⟩) 1 ⟨32239⟩ 70502

def event70593 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32240⟩⟩) (.sum [.predecessor 0 70591 .coefficient, .predecessor 1 70592 .coefficient])

def event70594 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32240⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨32239⟩⟩], []⟩) [⟨.result 70502 .coefficient, true, some 1⟩])

def event70595 : Event := .survivorFold (1) 70594

def event70596 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32240⟩⟩) (.sum [.result 70590 .summary, .transfer 70594])

def exact70597RawTerms : List Term := []

theorem exact70597RawTermsValid :
    exact70597RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70597 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32240⟩⟩) exact70597RawTerms (.finite 197) 70593 (.finite 197) (some (70596))

def event70598 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51295⟩⟩) 0 ⟨32240⟩ 70597

def event70599 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51295⟩⟩) 1 ⟨51294⟩ 70478

def event70600 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51295⟩⟩) (.sum [.predecessor 0 70598 .coefficient, .predecessor 1 70599 .coefficient])

def event70601 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51295⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], []⟩) [⟨.result 70478 .coefficient, true, some 1⟩])

def event70602 : Event := .survivorFold (1) 70601

def event70603 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51295⟩⟩) (.sum [.result 70597 .summary, .transfer 70601])

def exact70604RawTerms : List Term := []

theorem exact70604RawTermsValid :
    exact70604RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70604 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51295⟩⟩) exact70604RawTerms (.finite 255) 70600 (.finite 255) (some (70603))

def event70605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54275⟩⟩) 0 ⟨51295⟩ 70604

def event70606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54275⟩⟩) 1 ⟨54274⟩ 70454

def event70607 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54275⟩⟩) (.sum [.predecessor 0 70605 .coefficient, .predecessor 1 70606 .coefficient])

def event70608 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54275⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨54274⟩⟩], []⟩) [⟨.result 70454 .coefficient, true, some 1⟩])

def event70609 : Event := .survivorFold (1) 70608

def event70610 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54275⟩⟩) (.sum [.result 70604 .summary, .transfer 70608])

def exact70611RawTerms : List Term := []

theorem exact70611RawTermsValid :
    exact70611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70611 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54275⟩⟩) exact70611RawTerms (.finite 314) 70607 (.finite 314) (some (70610))

def event70612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57255⟩⟩) 0 ⟨54275⟩ 70611

def event70613 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57255⟩⟩) 1 ⟨57254⟩ 70430

def event70614 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57255⟩⟩) (.sum [.predecessor 0 70612 .coefficient, .predecessor 1 70613 .coefficient])

def event70615 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57255⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨57254⟩⟩], []⟩) [⟨.result 70430 .coefficient, true, some 1⟩])

def event70616 : Event := .survivorFold (1) 70615

def event70617 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57255⟩⟩) (.sum [.result 70611 .summary, .transfer 70615])

def exact70618RawTerms : List Term := []

theorem exact70618RawTermsValid :
    exact70618RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70618 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57255⟩⟩) exact70618RawTerms (.finite 374) 70614 (.finite 374) (some (70617))

def event70619 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60235⟩⟩) 0 ⟨57255⟩ 70618

def event70620 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60235⟩⟩) 1 ⟨60234⟩ 70406

def event70621 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60235⟩⟩) (.sum [.predecessor 0 70619 .coefficient, .predecessor 1 70620 .coefficient])

def event70622 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60235⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨60234⟩⟩], []⟩) [⟨.result 70406 .coefficient, true, some 1⟩])

def event70623 : Event := .survivorFold (1) 70622

def event70624 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60235⟩⟩) (.sum [.result 70618 .summary, .transfer 70622])

def exact70625RawTerms : List Term := []

theorem exact70625RawTermsValid :
    exact70625RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70625 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60235⟩⟩) exact70625RawTerms (.finite 435) 70621 (.finite 435) (some (70624))

def event70626 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63215⟩⟩) 0 ⟨60235⟩ 70625

def event70627 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63215⟩⟩) 1 ⟨63214⟩ 70382

def event70628 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63215⟩⟩) (.sum [.predecessor 0 70626 .coefficient, .predecessor 1 70627 .coefficient])

def event70629 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63215⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨63214⟩⟩], []⟩) [⟨.result 70382 .coefficient, true, some 1⟩])

def event70630 : Event := .survivorFold (1) 70629

def event70631 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63215⟩⟩) (.sum [.result 70625 .summary, .transfer 70629])

def exact70632RawTerms : List Term := []

theorem exact70632RawTermsValid :
    exact70632RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70632 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63215⟩⟩) exact70632RawTerms (.finite 496) 70628 (.finite 496) (some (70631))

def event70633 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67092⟩⟩) 0 ⟨63215⟩ 70632

def event70634 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67092⟩⟩) 1 ⟨67091⟩ 70358

def event70635 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67092⟩⟩) (.sum [.predecessor 0 70633 .coefficient, .predecessor 1 70634 .coefficient])

def event70636 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67092⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨67091⟩⟩], []⟩) [⟨.result 70358 .coefficient, true, some 1⟩])

def event70637 : Event := .survivorFold (1) 70636

def event70638 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67092⟩⟩) (.sum [.result 70632 .summary, .transfer 70636])

def exact70639RawTerms : List Term := []

theorem exact70639RawTermsValid :
    exact70639RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70639 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67092⟩⟩) exact70639RawTerms (.finite 558) 70635 (.finite 558) (some (70638))

def event70640 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67093⟩⟩) 0 ⟨67092⟩ 70639

def event70641 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67093⟩⟩) 1 ⟨26710⟩ 70334

def event70642 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67093⟩⟩) (.sum [.predecessor 0 70640 .coefficient, .predecessor 1 70641 .coefficient])

def event70643 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67093⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨26710⟩⟩], []⟩) [⟨.result 70334 .coefficient, true, some 1⟩])

def event70644 : Event := .survivorFold (1) 70643

def event70645 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67093⟩⟩) (.sum [.result 70639 .summary, .transfer 70643])

def exact70646RawTerms : List Term := []

theorem exact70646RawTermsValid :
    exact70646RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70646 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67093⟩⟩) exact70646RawTerms (.finite 620) 70642 (.finite 620) (some (70645))

def event70647 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67094⟩⟩) 0 ⟨67093⟩ 70646

def event70648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67094⟩⟩) 1 ⟨29390⟩ 70310

def event70649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67094⟩⟩) (.sum [.predecessor 0 70647 .coefficient, .predecessor 1 70648 .coefficient])

def event70650 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67094⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨29390⟩⟩], []⟩) [⟨.result 70310 .coefficient, true, some 1⟩])

def event70651 : Event := .survivorFold (1) 70650

def event70652 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67094⟩⟩) (.sum [.result 70646 .summary, .transfer 70650])

def exact70653RawTerms : List Term := []

theorem exact70653RawTermsValid :
    exact70653RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70653 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67094⟩⟩) exact70653RawTerms (.finite 682) 70649 (.finite 682) (some (70652))

def event70654 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67095⟩⟩) 0 ⟨67094⟩ 70653

def event70655 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67095⟩⟩) 1 ⟨35054⟩ 70286

def eventLeaf4400 : Array AnnotatedEvent := #[
  { event := event70400
    frameStart := 70122 },
  { event := event70401
    frameStart := 70122 },
  { event := event70402
    frameStart := 70122 },
  { event := event70403
    frameStart := 70122 },
  { event := event70404
    frameStart := 70122 },
  { event := event70405
    frameStart := 70122 },
  { event := event70406
    frameStart := 70122 },
  { event := event70407
    frameStart := 70122 },
  { event := event70408
    frameStart := 70122 },
  { event := event70409
    frameStart := 70122 },
  { event := event70410
    frameStart := 70122 },
  { event := event70411
    frameStart := 70122 },
  { event := event70412
    frameStart := 70122 },
  { event := event70413
    frameStart := 70122 },
  { event := event70414
    frameStart := 70122 },
  { event := event70415
    frameStart := 70122 }
]

def eventLeaf4401 : Array AnnotatedEvent := #[
  { event := event70416
    frameStart := 70122 },
  { event := event70417
    frameStart := 70122 },
  { event := event70418
    frameStart := 70122 },
  { event := event70419
    frameStart := 70122 },
  { event := event70420
    frameStart := 70122 },
  { event := event70421
    frameStart := 70122 },
  { event := event70422
    frameStart := 70122 },
  { event := event70423
    frameStart := 70122 },
  { event := event70424
    frameStart := 70122 },
  { event := event70425
    frameStart := 70122 },
  { event := event70426
    frameStart := 70122 },
  { event := event70427
    frameStart := 70122 },
  { event := event70428
    frameStart := 70122 },
  { event := event70429
    frameStart := 70122 },
  { event := event70430
    frameStart := 70122 },
  { event := event70431
    frameStart := 70122 }
]

def eventLeaf4402 : Array AnnotatedEvent := #[
  { event := event70432
    frameStart := 70122 },
  { event := event70433
    frameStart := 70122 },
  { event := event70434
    frameStart := 70122 },
  { event := event70435
    frameStart := 70122 },
  { event := event70436
    frameStart := 70122 },
  { event := event70437
    frameStart := 70122 },
  { event := event70438
    frameStart := 70122 },
  { event := event70439
    frameStart := 70122 },
  { event := event70440
    frameStart := 70122 },
  { event := event70441
    frameStart := 70122 },
  { event := event70442
    frameStart := 70122 },
  { event := event70443
    frameStart := 70122 },
  { event := event70444
    frameStart := 70122 },
  { event := event70445
    frameStart := 70122 },
  { event := event70446
    frameStart := 70122 },
  { event := event70447
    frameStart := 70122 }
]

def eventLeaf4403 : Array AnnotatedEvent := #[
  { event := event70448
    frameStart := 70122 },
  { event := event70449
    frameStart := 70122 },
  { event := event70450
    frameStart := 70122 },
  { event := event70451
    frameStart := 70122 },
  { event := event70452
    frameStart := 70122 },
  { event := event70453
    frameStart := 70122 },
  { event := event70454
    frameStart := 70122 },
  { event := event70455
    frameStart := 70122 },
  { event := event70456
    frameStart := 70122 },
  { event := event70457
    frameStart := 70122 },
  { event := event70458
    frameStart := 70122 },
  { event := event70459
    frameStart := 70122 },
  { event := event70460
    frameStart := 70122 },
  { event := event70461
    frameStart := 70122 },
  { event := event70462
    frameStart := 70122 },
  { event := event70463
    frameStart := 70122 }
]

def eventLeaf4404 : Array AnnotatedEvent := #[
  { event := event70464
    frameStart := 70122 },
  { event := event70465
    frameStart := 70122 },
  { event := event70466
    frameStart := 70122 },
  { event := event70467
    frameStart := 70122 },
  { event := event70468
    frameStart := 70122 },
  { event := event70469
    frameStart := 70122 },
  { event := event70470
    frameStart := 70122 },
  { event := event70471
    frameStart := 70122 },
  { event := event70472
    frameStart := 70122 },
  { event := event70473
    frameStart := 70122 },
  { event := event70474
    frameStart := 70122 },
  { event := event70475
    frameStart := 70122 },
  { event := event70476
    frameStart := 70122 },
  { event := event70477
    frameStart := 70122 },
  { event := event70478
    frameStart := 70122 },
  { event := event70479
    frameStart := 70122 }
]

def eventLeaf4405 : Array AnnotatedEvent := #[
  { event := event70480
    frameStart := 70122 },
  { event := event70481
    frameStart := 70122 },
  { event := event70482
    frameStart := 70122 },
  { event := event70483
    frameStart := 70122 },
  { event := event70484
    frameStart := 70122 },
  { event := event70485
    frameStart := 70122 },
  { event := event70486
    frameStart := 70122 },
  { event := event70487
    frameStart := 70122 },
  { event := event70488
    frameStart := 70122 },
  { event := event70489
    frameStart := 70122 },
  { event := event70490
    frameStart := 70122 },
  { event := event70491
    frameStart := 70122 },
  { event := event70492
    frameStart := 70122 },
  { event := event70493
    frameStart := 70122 },
  { event := event70494
    frameStart := 70122 },
  { event := event70495
    frameStart := 70122 }
]

def eventLeaf4406 : Array AnnotatedEvent := #[
  { event := event70496
    frameStart := 70122 },
  { event := event70497
    frameStart := 70122 },
  { event := event70498
    frameStart := 70122 },
  { event := event70499
    frameStart := 70122 },
  { event := event70500
    frameStart := 70122 },
  { event := event70501
    frameStart := 70122 },
  { event := event70502
    frameStart := 70122 },
  { event := event70503
    frameStart := 70122 },
  { event := event70504
    frameStart := 70122 },
  { event := event70505
    frameStart := 70122 },
  { event := event70506
    frameStart := 70122 },
  { event := event70507
    frameStart := 70122 },
  { event := event70508
    frameStart := 70122 },
  { event := event70509
    frameStart := 70122 },
  { event := event70510
    frameStart := 70122 },
  { event := event70511
    frameStart := 70122 }
]

def eventLeaf4407 : Array AnnotatedEvent := #[
  { event := event70512
    frameStart := 70122 },
  { event := event70513
    frameStart := 70122 },
  { event := event70514
    frameStart := 70122 },
  { event := event70515
    frameStart := 70122 },
  { event := event70516
    frameStart := 70122 },
  { event := event70517
    frameStart := 70122 },
  { event := event70518
    frameStart := 70122 },
  { event := event70519
    frameStart := 70122 },
  { event := event70520
    frameStart := 70122 },
  { event := event70521
    frameStart := 70122 },
  { event := event70522
    frameStart := 70122 },
  { event := event70523
    frameStart := 70122 },
  { event := event70524
    frameStart := 70122 },
  { event := event70525
    frameStart := 70122 },
  { event := event70526
    frameStart := 70122 },
  { event := event70527
    frameStart := 70122 }
]

def eventLeaf4408 : Array AnnotatedEvent := #[
  { event := event70528
    frameStart := 70122 },
  { event := event70529
    frameStart := 70122 },
  { event := event70530
    frameStart := 70122 },
  { event := event70531
    frameStart := 70122 },
  { event := event70532
    frameStart := 70122 },
  { event := event70533
    frameStart := 70122 },
  { event := event70534
    frameStart := 70122 },
  { event := event70535
    frameStart := 70122 },
  { event := event70536
    frameStart := 70122 },
  { event := event70537
    frameStart := 70122 },
  { event := event70538
    frameStart := 70122 },
  { event := event70539
    frameStart := 70122 },
  { event := event70540
    frameStart := 70122 },
  { event := event70541
    frameStart := 70122 },
  { event := event70542
    frameStart := 70122 },
  { event := event70543
    frameStart := 70122 }
]

def eventLeaf4409 : Array AnnotatedEvent := #[
  { event := event70544
    frameStart := 70122 },
  { event := event70545
    frameStart := 70122 },
  { event := event70546
    frameStart := 70122 },
  { event := event70547
    frameStart := 70122 },
  { event := event70548
    frameStart := 70122 },
  { event := event70549
    frameStart := 70122 },
  { event := event70550
    frameStart := 70122 },
  { event := event70551
    frameStart := 70122 },
  { event := event70552
    frameStart := 70122 },
  { event := event70553
    frameStart := 70122 },
  { event := event70554
    frameStart := 70122 },
  { event := event70555
    frameStart := 70122 },
  { event := event70556
    frameStart := 70122 },
  { event := event70557
    frameStart := 70122 },
  { event := event70558
    frameStart := 70122 },
  { event := event70559
    frameStart := 70122 }
]

def eventLeaf4410 : Array AnnotatedEvent := #[
  { event := event70560
    frameStart := 70122 },
  { event := event70561
    frameStart := 70122 },
  { event := event70562
    frameStart := 70122 },
  { event := event70563
    frameStart := 70122 },
  { event := event70564
    frameStart := 70122 },
  { event := event70565
    frameStart := 70122 },
  { event := event70566
    frameStart := 70122 },
  { event := event70567
    frameStart := 70122 },
  { event := event70568
    frameStart := 70122 },
  { event := event70569
    frameStart := 70122 },
  { event := event70570
    frameStart := 70122 },
  { event := event70571
    frameStart := 70122 },
  { event := event70572
    frameStart := 70122 },
  { event := event70573
    frameStart := 70122 },
  { event := event70574
    frameStart := 70122 },
  { event := event70575
    frameStart := 70122 }
]

def eventLeaf4411 : Array AnnotatedEvent := #[
  { event := event70576
    frameStart := 70122 },
  { event := event70577
    frameStart := 70122 },
  { event := event70578
    frameStart := 70122 },
  { event := event70579
    frameStart := 70122 },
  { event := event70580
    frameStart := 70122 },
  { event := event70581
    frameStart := 70122 },
  { event := event70582
    frameStart := 70122 },
  { event := event70583
    frameStart := 70122 },
  { event := event70584
    frameStart := 70122 },
  { event := event70585
    frameStart := 70122 },
  { event := event70586
    frameStart := 70122 },
  { event := event70587
    frameStart := 70122 },
  { event := event70588
    frameStart := 70122 },
  { event := event70589
    frameStart := 70122 },
  { event := event70590
    frameStart := 70122 },
  { event := event70591
    frameStart := 70122 }
]

def eventLeaf4412 : Array AnnotatedEvent := #[
  { event := event70592
    frameStart := 70122 },
  { event := event70593
    frameStart := 70122 },
  { event := event70594
    frameStart := 70122 },
  { event := event70595
    frameStart := 70122 },
  { event := event70596
    frameStart := 70122 },
  { event := event70597
    frameStart := 70122 },
  { event := event70598
    frameStart := 70122 },
  { event := event70599
    frameStart := 70122 },
  { event := event70600
    frameStart := 70122 },
  { event := event70601
    frameStart := 70122 },
  { event := event70602
    frameStart := 70122 },
  { event := event70603
    frameStart := 70122 },
  { event := event70604
    frameStart := 70122 },
  { event := event70605
    frameStart := 70122 },
  { event := event70606
    frameStart := 70122 },
  { event := event70607
    frameStart := 70122 }
]

def eventLeaf4413 : Array AnnotatedEvent := #[
  { event := event70608
    frameStart := 70122 },
  { event := event70609
    frameStart := 70122 },
  { event := event70610
    frameStart := 70122 },
  { event := event70611
    frameStart := 70122 },
  { event := event70612
    frameStart := 70122 },
  { event := event70613
    frameStart := 70122 },
  { event := event70614
    frameStart := 70122 },
  { event := event70615
    frameStart := 70122 },
  { event := event70616
    frameStart := 70122 },
  { event := event70617
    frameStart := 70122 },
  { event := event70618
    frameStart := 70122 },
  { event := event70619
    frameStart := 70122 },
  { event := event70620
    frameStart := 70122 },
  { event := event70621
    frameStart := 70122 },
  { event := event70622
    frameStart := 70122 },
  { event := event70623
    frameStart := 70122 }
]

def eventLeaf4414 : Array AnnotatedEvent := #[
  { event := event70624
    frameStart := 70122 },
  { event := event70625
    frameStart := 70122 },
  { event := event70626
    frameStart := 70122 },
  { event := event70627
    frameStart := 70122 },
  { event := event70628
    frameStart := 70122 },
  { event := event70629
    frameStart := 70122 },
  { event := event70630
    frameStart := 70122 },
  { event := event70631
    frameStart := 70122 },
  { event := event70632
    frameStart := 70122 },
  { event := event70633
    frameStart := 70122 },
  { event := event70634
    frameStart := 70122 },
  { event := event70635
    frameStart := 70122 },
  { event := event70636
    frameStart := 70122 },
  { event := event70637
    frameStart := 70122 },
  { event := event70638
    frameStart := 70122 },
  { event := event70639
    frameStart := 70122 }
]

def eventLeaf4415 : Array AnnotatedEvent := #[
  { event := event70640
    frameStart := 70122 },
  { event := event70641
    frameStart := 70122 },
  { event := event70642
    frameStart := 70122 },
  { event := event70643
    frameStart := 70122 },
  { event := event70644
    frameStart := 70122 },
  { event := event70645
    frameStart := 70122 },
  { event := event70646
    frameStart := 70122 },
  { event := event70647
    frameStart := 70122 },
  { event := event70648
    frameStart := 70122 },
  { event := event70649
    frameStart := 70122 },
  { event := event70650
    frameStart := 70122 },
  { event := event70651
    frameStart := 70122 },
  { event := event70652
    frameStart := 70122 },
  { event := event70653
    frameStart := 70122 },
  { event := event70654
    frameStart := 70122 },
  { event := event70655
    frameStart := 70122 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events275
