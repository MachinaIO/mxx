import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events029

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event7424 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51109⟩⟩) 0 ⟨51108⟩ 7423

def event7425 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51109⟩⟩) 1 ⟨6768⟩ 673

def event7426 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51109⟩⟩) (.product (.predecessor 0 7424 .coefficient) (.predecessor 1 7425 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event7427 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51109⟩⟩, .operator (⟨7423, 0⟩, ⟨673, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩)

def exact7428RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩]

theorem exact7428RawTermsValid :
    exact7428RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7428 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51109⟩⟩) exact7428RawTerms (.finite 213251602471649038151400) 7426 .exactZero (none)

def event7429 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32044⟩⟩) 0 ⟨31805⟩ 7165

def event7430 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32044⟩⟩) (.authority (.programFamilyFact))

def exact7431RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩]

theorem exact7431RawTermsValid :
    exact7431RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7431 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32044⟩⟩) exact7431RawTerms (.finite 6) 7430 .exactZero (none)

def event7432 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32045⟩⟩) 0 ⟨32044⟩ 7431

def event7433 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32045⟩⟩) 1 ⟨6794⟩ 683

def event7434 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32045⟩⟩) (.product (.predecessor 0 7432 .coefficient) (.predecessor 1 7433 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event7435 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32045⟩⟩, .operator (⟨7431, 0⟩, ⟨683, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩)

def exact7436RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩]

theorem exact7436RawTermsValid :
    exact7436RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7436 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32045⟩⟩) exact7436RawTerms (.finite 201065796616126235971320) 7434 .exactZero (none)

def event7437 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22024⟩⟩) 0 ⟨21785⟩ 7188

def event7438 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22024⟩⟩) (.authority (.programFamilyFact))

def exact7439RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩]

theorem exact7439RawTermsValid :
    exact7439RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7439 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22024⟩⟩) exact7439RawTerms (.finite 4) 7438 .exactZero (none)

def event7440 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22025⟩⟩) 0 ⟨22024⟩ 7439

def event7441 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22025⟩⟩) 1 ⟨6822⟩ 693

def event7442 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22025⟩⟩) (.product (.predecessor 0 7440 .coefficient) (.predecessor 1 7441 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event7443 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22025⟩⟩, .operator (⟨7439, 0⟩, ⟨693, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩)

def exact7444RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩]

theorem exact7444RawTermsValid :
    exact7444RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7444 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22025⟩⟩) exact7444RawTerms (.finite 187661410175051153573232) 7442 .exactZero (none)

def event7445 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18804⟩⟩) 0 ⟨18565⟩ 7211

def event7446 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18804⟩⟩) (.authority (.programFamilyFact))

def exact7447RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩]

theorem exact7447RawTermsValid :
    exact7447RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7447 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18804⟩⟩) exact7447RawTerms (.finite 3) 7446 .exactZero (none)

def event7448 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18805⟩⟩) 0 ⟨18804⟩ 7447

def event7449 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18805⟩⟩) 1 ⟨6846⟩ 703

def event7450 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18805⟩⟩) (.product (.predecessor 0 7448 .coefficient) (.predecessor 1 7449 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event7451 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18805⟩⟩, .operator (⟨7447, 0⟩, ⟨703, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩)

def exact7452RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩]

theorem exact7452RawTermsValid :
    exact7452RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7452 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18805⟩⟩) exact7452RawTerms (.finite 175932572039110456474905) 7450 .exactZero (none)

def event7453 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15982⟩⟩) 0 ⟨15765⟩ 7234

def event7454 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15982⟩⟩) (.authority (.programFamilyFact))

def exact7455RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩]

theorem exact7455RawTermsValid :
    exact7455RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7455 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15982⟩⟩) exact7455RawTerms (.finite 2) 7454 .exactZero (none)

def event7456 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15983⟩⟩) 0 ⟨15982⟩ 7455

def event7457 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15983⟩⟩) 1 ⟨6863⟩ 713

def event7458 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15983⟩⟩) (.product (.predecessor 0 7456 .coefficient) (.predecessor 1 7457 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event7459 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15983⟩⟩, .operator (⟨7455, 0⟩, ⟨713, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩)

def exact7460RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩]

theorem exact7460RawTermsValid :
    exact7460RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7460 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15983⟩⟩) exact7460RawTerms (.finite 156384508479209294644360) 7458 .exactZero (none)

def event7461 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15984⟩⟩) 0 ⟨6728⟩ 728

def event7462 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15984⟩⟩) 1 ⟨15983⟩ 7460

def event7463 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15984⟩⟩) (.sum [.predecessor 0 7461 .coefficient, .predecessor 1 7462 .coefficient])

def exact7464RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩]

theorem exact7464RawTermsValid :
    exact7464RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7464 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15984⟩⟩) exact7464RawTerms (.finite 156384508479209294644360) 7463 .exactZero (none)

def event7465 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18806⟩⟩) 0 ⟨15984⟩ 7464

def event7466 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18806⟩⟩) 1 ⟨18805⟩ 7452

def event7467 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18806⟩⟩) (.sum [.predecessor 0 7465 .coefficient, .predecessor 1 7466 .coefficient])

def exact7468RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩]

theorem exact7468RawTermsValid :
    exact7468RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7468 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18806⟩⟩) exact7468RawTerms (.finite 332317080518319751119265) 7467 .exactZero (none)

def event7469 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22026⟩⟩) 0 ⟨18806⟩ 7468

def event7470 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22026⟩⟩) 1 ⟨22025⟩ 7444

def event7471 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22026⟩⟩) (.sum [.predecessor 0 7469 .coefficient, .predecessor 1 7470 .coefficient])

def exact7472RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩]

theorem exact7472RawTermsValid :
    exact7472RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7472 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22026⟩⟩) exact7472RawTerms (.finite 519978490693370904692497) 7471 .exactZero (none)

def event7473 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32046⟩⟩) 0 ⟨22026⟩ 7472

def event7474 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32046⟩⟩) 1 ⟨32045⟩ 7436

def event7475 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32046⟩⟩) (.sum [.predecessor 0 7473 .coefficient, .predecessor 1 7474 .coefficient])

def exact7476RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩]

theorem exact7476RawTermsValid :
    exact7476RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7476 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32046⟩⟩) exact7476RawTerms (.finite 721044287309497140663817) 7475 .exactZero (none)

def event7477 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51110⟩⟩) 0 ⟨32046⟩ 7476

def event7478 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51110⟩⟩) 1 ⟨51109⟩ 7428

def event7479 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51110⟩⟩) (.sum [.predecessor 0 7477 .coefficient, .predecessor 1 7478 .coefficient])

def exact7480RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩]

theorem exact7480RawTermsValid :
    exact7480RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7480 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51110⟩⟩) exact7480RawTerms (.finite 934295889781146178815217) 7479 .exactZero (none)

def event7481 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54090⟩⟩) 0 ⟨51110⟩ 7480

def event7482 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54090⟩⟩) 1 ⟨54089⟩ 7420

def event7483 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54090⟩⟩) (.sum [.predecessor 0 7481 .coefficient, .predecessor 1 7482 .coefficient])

def exact7484RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩]

theorem exact7484RawTermsValid :
    exact7484RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7484 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54090⟩⟩) exact7484RawTerms (.finite 1150828286136974432938177) 7483 .exactZero (none)

def event7485 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57070⟩⟩) 0 ⟨54090⟩ 7484

def event7486 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57070⟩⟩) 1 ⟨57069⟩ 7412

def event7487 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57070⟩⟩) (.sum [.predecessor 0 7485 .coefficient, .predecessor 1 7486 .coefficient])

def exact7488RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩]

theorem exact7488RawTermsValid :
    exact7488RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7488 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57070⟩⟩) exact7488RawTerms (.finite 1371606415754681672436097) 7487 .exactZero (none)

def event7489 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60050⟩⟩) 0 ⟨57070⟩ 7488

def event7490 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60050⟩⟩) 1 ⟨60049⟩ 7404

def event7491 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60050⟩⟩) (.sum [.predecessor 0 7489 .coefficient, .predecessor 1 7490 .coefficient])

def exact7492RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩]

theorem exact7492RawTermsValid :
    exact7492RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7492 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60050⟩⟩) exact7492RawTerms (.finite 1593837033067242249035977) 7491 .exactZero (none)

def event7493 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63030⟩⟩) 0 ⟨60050⟩ 7492

def event7494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63030⟩⟩) 1 ⟨63029⟩ 7396

def event7495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63030⟩⟩) (.sum [.predecessor 0 7493 .coefficient, .predecessor 1 7494 .coefficient])

def exact7496RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩]

theorem exact7496RawTermsValid :
    exact7496RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7496 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63030⟩⟩) exact7496RawTerms (.finite 1818214806102629497873537) 7495 .exactZero (none)

def event7497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66380⟩⟩) 0 ⟨63030⟩ 7496

def event7498 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66380⟩⟩) 1 ⟨66379⟩ 7388

def event7499 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66380⟩⟩) (.sum [.predecessor 0 7497 .coefficient, .predecessor 1 7498 .coefficient])

def exact7500RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], []⟩, (1)⟩]

theorem exact7500RawTermsValid :
    exact7500RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7500 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66380⟩⟩) exact7500RawTerms (.finite 2044702714934587786668817) 7499 .exactZero (none)

def event7501 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66381⟩⟩) 0 ⟨66380⟩ 7500

def event7502 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66381⟩⟩) 1 ⟨26584⟩ 7380

def event7503 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66381⟩⟩) (.sum [.predecessor 0 7501 .coefficient, .predecessor 1 7502 .coefficient])

def exact7504RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], []⟩, (1)⟩]

theorem exact7504RawTermsValid :
    exact7504RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7504 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66381⟩⟩) exact7504RawTerms (.finite 2271712485307633536959017) 7503 .exactZero (none)

def event7505 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66382⟩⟩) 0 ⟨66381⟩ 7504

def event7506 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66382⟩⟩) 1 ⟨29264⟩ 7372

def event7507 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66382⟩⟩) (.sum [.predecessor 0 7505 .coefficient, .predecessor 1 7506 .coefficient])

def exact7508RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], []⟩, (1)⟩]

theorem exact7508RawTermsValid :
    exact7508RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7508 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66382⟩⟩) exact7508RawTerms (.finite 2499949335520533588602137) 7507 .exactZero (none)

def event7509 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66383⟩⟩) 0 ⟨66382⟩ 7508

def event7510 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66383⟩⟩) 1 ⟨34921⟩ 7364

def event7511 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66383⟩⟩) (.sum [.predecessor 0 7509 .coefficient, .predecessor 1 7510 .coefficient])

def exact7512RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], []⟩, (1)⟩]

theorem exact7512RawTermsValid :
    exact7512RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7512 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66383⟩⟩) exact7512RawTerms (.finite 2728804713782791092959737) 7511 .exactZero (none)

def event7513 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66384⟩⟩) 0 ⟨66383⟩ 7512

def event7514 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66384⟩⟩) 1 ⟨37601⟩ 7356

def event7515 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66384⟩⟩) (.sum [.predecessor 0 7513 .coefficient, .predecessor 1 7514 .coefficient])

def exact7516RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], []⟩, (1)⟩]

theorem exact7516RawTermsValid :
    exact7516RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7516 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66384⟩⟩) exact7516RawTerms (.finite 2957926202950004710694497) 7515 .exactZero (none)

def event7517 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66385⟩⟩) 0 ⟨66384⟩ 7516

def event7518 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66385⟩⟩) 1 ⟨40284⟩ 7348

def event7519 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66385⟩⟩) (.sum [.predecessor 0 7517 .coefficient, .predecessor 1 7518 .coefficient])

def exact7520RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], []⟩, (1)⟩]

theorem exact7520RawTermsValid :
    exact7520RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7520 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66385⟩⟩) exact7520RawTerms (.finite 3187511970717354526236217) 7519 .exactZero (none)

def event7521 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66386⟩⟩) 0 ⟨66385⟩ 7520

def event7522 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66386⟩⟩) 1 ⟨42964⟩ 7340

def event7523 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66386⟩⟩) (.sum [.predecessor 0 7521 .coefficient, .predecessor 1 7522 .coefficient])

def exact7524RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42963⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], []⟩, (1)⟩]

theorem exact7524RawTermsValid :
    exact7524RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7524 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66386⟩⟩) exact7524RawTerms (.finite 3417662756781096507033577) 7523 .exactZero (none)

def event7525 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66387⟩⟩) 0 ⟨66386⟩ 7524

def event7526 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66387⟩⟩) 1 ⟨45641⟩ 7332

def event7527 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66387⟩⟩) (.sum [.predecessor 0 7525 .coefficient, .predecessor 1 7526 .coefficient])

def exact7528RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45640⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42963⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], []⟩, (1)⟩]

theorem exact7528RawTermsValid :
    exact7528RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7528 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66387⟩⟩) exact7528RawTerms (.finite 3648263642165693263543057) 7527 .exactZero (none)

def event7529 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66388⟩⟩) 0 ⟨66387⟩ 7528

def event7530 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66388⟩⟩) 1 ⟨48321⟩ 7324

def event7531 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66388⟩⟩) (.sum [.predecessor 0 7529 .coefficient, .predecessor 1 7530 .coefficient])

def exact7532RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48320⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45640⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42963⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], []⟩, (1)⟩]

theorem exact7532RawTermsValid :
    exact7532RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7532 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66388⟩⟩) exact7532RawTerms (.finite 3878994884184198780231457) 7531 .exactZero (none)

def event7533 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67402⟩⟩) 0 ⟨66388⟩ 7532

def event7534 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67402⟩⟩) 1 ⟨67400⟩ 7316

def event7535 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67402⟩⟩) (.sum [.predecessor 0 7533 .coefficient, .predecessor 1 7534 .coefficient])

def exact7536RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48320⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45640⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42963⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], []⟩, (1)⟩]

theorem exact7536RawTermsValid :
    exact7536RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7536 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67402⟩⟩) exact7536RawTerms (.finite 8101376613122849735629177) 7535 .exactZero (none)

def event7537 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67403⟩⟩) 0 ⟨67402⟩ 7536

def event7538 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67403⟩⟩) 1 ⟨6771⟩ 6813

def event7539 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67403⟩⟩) (.product (.predecessor 0 7537 .coefficient) (.predecessor 1 7538 .coefficient) (⟨false, true, none, none, some 1⟩))

def event7540 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 5⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], []⟩, (-1)⟩)

def event7541 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 7⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48320⟩⟩], []⟩, (1)⟩)

def event7542 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 8⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45640⟩⟩], []⟩, (1)⟩)

def event7543 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 9⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42963⟩⟩], []⟩, (1)⟩)

def event7544 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 11⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], []⟩, (1)⟩)

def event7545 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 12⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], []⟩, (1)⟩)

def event7546 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 13⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], []⟩, (1)⟩)

def event7547 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 15⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], []⟩, (1)⟩)

def event7548 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 16⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], []⟩, (1)⟩)

def event7549 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 18⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], []⟩, (1)⟩)

def event7550 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 0⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], []⟩, (1)⟩)

def event7551 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 1⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], []⟩, (1)⟩)

def event7552 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 2⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], []⟩, (1)⟩)

def event7553 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 3⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], []⟩, (1)⟩)

def event7554 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 4⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩)

def event7555 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 6⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩)

def event7556 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 10⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩)

def event7557 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 14⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩)

def event7558 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67403⟩⟩, .operator (⟨7536, 17⟩, ⟨6813, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩)

def exact7559RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48320⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45640⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42963⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], []⟩, (1)⟩]

theorem exact7559RawTermsValid :
    exact7559RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7559 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67403⟩⟩) exact7559RawTerms (.finite 89809622429143058223378542743516224969990609770815455493412662590996290517036492008202283232802263779022597713369473098115754867015020364139002338041287686971404088610753484409081597427782594166784) 7539 .exactZero (none)

def event7560 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6765⟩⟩) (.authority (.factStore))

def exact7561RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6765⟩⟩], []⟩, (1)⟩]

theorem exact7561RawTermsValid :
    exact7561RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7561 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6765⟩⟩) exact7561RawTerms (.finite 209547688210549055471147046111004916489331190890252620496502021405337735671870380095231105730177606312631955343380640763509911328536630738066641741668496568757831236150) 7560 .exactZero (none)

def event7562 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6449⟩⟩) (.authority (.operator))

def event7563 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6449⟩⟩) (.finite 9)

def event7564 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6451⟩⟩) 0 ⟨392⟩ 14

def event7565 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6451⟩⟩) 1 ⟨6449⟩ 7563

def event7566 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6451⟩⟩) (.sum [.predecessor 0 7564 .coefficient, .predecessor 1 7565 .coefficient])

def event7567 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6451⟩⟩) (.finite 655349)

def event7568 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6462⟩⟩) 0 ⟨6451⟩ 7567

def event7569 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6462⟩⟩) 1 ⟨5426⟩ 38

def event7570 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6462⟩⟩) (.identity (.predecessor 1 7569 .coefficient))

def event7571 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨6462⟩⟩) (.finite 655360)

def event7572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47930⟩⟩) 0 ⟨6462⟩ 7571

def event7573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47930⟩⟩) (.authority (.programFamilyFact))

def exact7574RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47930⟩⟩], []⟩, (1)⟩]

theorem exact7574RawTermsValid :
    exact7574RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7574 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47930⟩⟩) exact7574RawTerms (.finite 60) 7573 .exactZero (none)

def event7575 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15141⟩⟩) 0 ⟨6462⟩ 7571

def event7576 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15141⟩⟩) (.authority (.programFamilyFact))

def exact7577RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15141⟩⟩], []⟩, (1)⟩]

theorem exact7577RawTermsValid :
    exact7577RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7577 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15141⟩⟩) exact7577RawTerms (.finite 60) 7576 .exactZero (none)

def event7578 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47931⟩⟩) 0 ⟨15141⟩ 7577

def event7579 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47931⟩⟩) 1 ⟨47930⟩ 7574

def event7580 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47931⟩⟩) (.product (.predecessor 0 7578 .coefficient) (.predecessor 1 7579 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event7581 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47931⟩⟩, .operator (⟨7577, 0⟩, ⟨7574, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15141⟩⟩, ⟨.program ⟨257⟩, ⟨47930⟩⟩], []⟩, (1)⟩)

def exact7582RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15141⟩⟩, ⟨.program ⟨257⟩, ⟨47930⟩⟩], []⟩, (1)⟩]

theorem exact7582RawTermsValid :
    exact7582RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7582 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47931⟩⟩) exact7582RawTerms (.finite 3600) 7580 .exactZero (none)

def event7583 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47932⟩⟩) 0 ⟨47931⟩ 7582

def event7584 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47932⟩⟩) (.identity (.predecessor 0 7583 .coefficient))

def event7585 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47932⟩⟩) (.finite 3600)

def event7586 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48180⟩⟩) 0 ⟨47932⟩ 7585

def event7587 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48180⟩⟩) (.authority (.programFamilyFact))

def exact7588RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48180⟩⟩], []⟩, (1)⟩]

theorem exact7588RawTermsValid :
    exact7588RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7588 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48180⟩⟩) exact7588RawTerms (.finite 60) 7587 .exactZero (none)

def event7589 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48181⟩⟩) 0 ⟨48180⟩ 7588

def event7590 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48181⟩⟩) (.identity (.predecessor 0 7589 .coefficient))

def event7591 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48181⟩⟩) (.finite 60)

def event7592 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48415⟩⟩) 0 ⟨48181⟩ 7591

def event7593 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48415⟩⟩) (.authority (.programFamilyFact))

def exact7594RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48415⟩⟩], []⟩, (1)⟩]

theorem exact7594RawTermsValid :
    exact7594RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7594 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48415⟩⟩) exact7594RawTerms (.finite 63) 7593 .exactZero (none)

def event7595 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45250⟩⟩) 0 ⟨6462⟩ 7571

def event7596 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45250⟩⟩) (.authority (.programFamilyFact))

def exact7597RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45250⟩⟩], []⟩, (1)⟩]

theorem exact7597RawTermsValid :
    exact7597RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7597 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45250⟩⟩) exact7597RawTerms (.finite 58) 7596 .exactZero (none)

def event7598 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14841⟩⟩) 0 ⟨6462⟩ 7571

def event7599 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14841⟩⟩) (.authority (.programFamilyFact))

def exact7600RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14841⟩⟩], []⟩, (1)⟩]

theorem exact7600RawTermsValid :
    exact7600RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7600 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14841⟩⟩) exact7600RawTerms (.finite 58) 7599 .exactZero (none)

def event7601 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45251⟩⟩) 0 ⟨14841⟩ 7600

def event7602 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45251⟩⟩) 1 ⟨45250⟩ 7597

def event7603 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45251⟩⟩) (.product (.predecessor 0 7601 .coefficient) (.predecessor 1 7602 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event7604 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45251⟩⟩, .operator (⟨7600, 0⟩, ⟨7597, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], []⟩, (1)⟩)

def exact7605RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], []⟩, (1)⟩]

theorem exact7605RawTermsValid :
    exact7605RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7605 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45251⟩⟩) exact7605RawTerms (.finite 3364) 7603 .exactZero (none)

def event7606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45252⟩⟩) 0 ⟨45251⟩ 7605

def event7607 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45252⟩⟩) (.identity (.predecessor 0 7606 .coefficient))

def event7608 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45252⟩⟩) (.finite 3364)

def event7609 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45500⟩⟩) 0 ⟨45252⟩ 7608

def event7610 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45500⟩⟩) (.authority (.programFamilyFact))

def exact7611RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45500⟩⟩], []⟩, (1)⟩]

theorem exact7611RawTermsValid :
    exact7611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7611 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45500⟩⟩) exact7611RawTerms (.finite 58) 7610 .exactZero (none)

def event7612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45501⟩⟩) 0 ⟨45500⟩ 7611

def event7613 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45501⟩⟩) (.identity (.predecessor 0 7612 .coefficient))

def event7614 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45501⟩⟩) (.finite 58)

def event7615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45735⟩⟩) 0 ⟨45501⟩ 7614

def event7616 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45735⟩⟩) (.authority (.programFamilyFact))

def exact7617RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45735⟩⟩], []⟩, (1)⟩]

theorem exact7617RawTermsValid :
    exact7617RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7617 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45735⟩⟩) exact7617RawTerms (.finite 63) 7616 .exactZero (none)

def event7618 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42570⟩⟩) 0 ⟨6462⟩ 7571

def event7619 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42570⟩⟩) (.authority (.programFamilyFact))

def exact7620RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42570⟩⟩], []⟩, (1)⟩]

theorem exact7620RawTermsValid :
    exact7620RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7620 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42570⟩⟩) exact7620RawTerms (.finite 52) 7619 .exactZero (none)

def event7621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14541⟩⟩) 0 ⟨6462⟩ 7571

def event7622 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14541⟩⟩) (.authority (.programFamilyFact))

def exact7623RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14541⟩⟩], []⟩, (1)⟩]

theorem exact7623RawTermsValid :
    exact7623RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7623 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14541⟩⟩) exact7623RawTerms (.finite 52) 7622 .exactZero (none)

def event7624 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42571⟩⟩) 0 ⟨14541⟩ 7623

def event7625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42571⟩⟩) 1 ⟨42570⟩ 7620

def event7626 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42571⟩⟩) (.product (.predecessor 0 7624 .coefficient) (.predecessor 1 7625 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event7627 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42571⟩⟩, .operator (⟨7623, 0⟩, ⟨7620, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], []⟩, (1)⟩)

def exact7628RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], []⟩, (1)⟩]

theorem exact7628RawTermsValid :
    exact7628RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7628 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42571⟩⟩) exact7628RawTerms (.finite 2704) 7626 .exactZero (none)

def event7629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42572⟩⟩) 0 ⟨42571⟩ 7628

def event7630 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42572⟩⟩) (.identity (.predecessor 0 7629 .coefficient))

def event7631 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42572⟩⟩) (.finite 2704)

def event7632 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42820⟩⟩) 0 ⟨42572⟩ 7631

def event7633 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42820⟩⟩) (.authority (.programFamilyFact))

def exact7634RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42820⟩⟩], []⟩, (1)⟩]

theorem exact7634RawTermsValid :
    exact7634RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7634 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42820⟩⟩) exact7634RawTerms (.finite 52) 7633 .exactZero (none)

def event7635 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42821⟩⟩) 0 ⟨42820⟩ 7634

def event7636 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42821⟩⟩) (.identity (.predecessor 0 7635 .coefficient))

def event7637 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42821⟩⟩) (.finite 52)

def event7638 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43051⟩⟩) 0 ⟨42821⟩ 7637

def event7639 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43051⟩⟩) (.authority (.programFamilyFact))

def exact7640RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨43051⟩⟩], []⟩, (1)⟩]

theorem exact7640RawTermsValid :
    exact7640RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7640 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43051⟩⟩) exact7640RawTerms (.finite 63) 7639 .exactZero (none)

def event7641 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39890⟩⟩) 0 ⟨6462⟩ 7571

def event7642 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39890⟩⟩) (.authority (.programFamilyFact))

def exact7643RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39890⟩⟩], []⟩, (1)⟩]

theorem exact7643RawTermsValid :
    exact7643RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7643 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39890⟩⟩) exact7643RawTerms (.finite 46) 7642 .exactZero (none)

def event7644 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14241⟩⟩) 0 ⟨6462⟩ 7571

def event7645 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14241⟩⟩) (.authority (.programFamilyFact))

def exact7646RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14241⟩⟩], []⟩, (1)⟩]

theorem exact7646RawTermsValid :
    exact7646RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7646 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14241⟩⟩) exact7646RawTerms (.finite 46) 7645 .exactZero (none)

def event7647 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39891⟩⟩) 0 ⟨14241⟩ 7646

def event7648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39891⟩⟩) 1 ⟨39890⟩ 7643

def event7649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39891⟩⟩) (.product (.predecessor 0 7647 .coefficient) (.predecessor 1 7648 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event7650 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39891⟩⟩, .operator (⟨7646, 0⟩, ⟨7643, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14241⟩⟩, ⟨.program ⟨257⟩, ⟨39890⟩⟩], []⟩, (1)⟩)

def exact7651RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14241⟩⟩, ⟨.program ⟨257⟩, ⟨39890⟩⟩], []⟩, (1)⟩]

theorem exact7651RawTermsValid :
    exact7651RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7651 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39891⟩⟩) exact7651RawTerms (.finite 2116) 7649 .exactZero (none)

def event7652 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39892⟩⟩) 0 ⟨39891⟩ 7651

def event7653 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39892⟩⟩) (.identity (.predecessor 0 7652 .coefficient))

def event7654 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39892⟩⟩) (.finite 2116)

def event7655 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40140⟩⟩) 0 ⟨39892⟩ 7654

def event7656 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40140⟩⟩) (.authority (.programFamilyFact))

def exact7657RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40140⟩⟩], []⟩, (1)⟩]

theorem exact7657RawTermsValid :
    exact7657RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7657 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40140⟩⟩) exact7657RawTerms (.finite 46) 7656 .exactZero (none)

def event7658 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40141⟩⟩) 0 ⟨40140⟩ 7657

def event7659 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40141⟩⟩) (.identity (.predecessor 0 7658 .coefficient))

def event7660 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40141⟩⟩) (.finite 46)

def event7661 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40371⟩⟩) 0 ⟨40141⟩ 7660

def event7662 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40371⟩⟩) (.authority (.programFamilyFact))

def exact7663RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40371⟩⟩], []⟩, (1)⟩]

theorem exact7663RawTermsValid :
    exact7663RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7663 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40371⟩⟩) exact7663RawTerms (.finite 63) 7662 .exactZero (none)

def event7664 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37210⟩⟩) 0 ⟨6462⟩ 7571

def event7665 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37210⟩⟩) (.authority (.programFamilyFact))

def exact7666RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37210⟩⟩], []⟩, (1)⟩]

theorem exact7666RawTermsValid :
    exact7666RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7666 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37210⟩⟩) exact7666RawTerms (.finite 42) 7665 .exactZero (none)

def event7667 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13941⟩⟩) 0 ⟨6462⟩ 7571

def event7668 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13941⟩⟩) (.authority (.programFamilyFact))

def exact7669RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13941⟩⟩], []⟩, (1)⟩]

theorem exact7669RawTermsValid :
    exact7669RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7669 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13941⟩⟩) exact7669RawTerms (.finite 42) 7668 .exactZero (none)

def event7670 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37211⟩⟩) 0 ⟨13941⟩ 7669

def event7671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37211⟩⟩) 1 ⟨37210⟩ 7666

def event7672 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37211⟩⟩) (.product (.predecessor 0 7670 .coefficient) (.predecessor 1 7671 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event7673 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37211⟩⟩, .operator (⟨7669, 0⟩, ⟨7666, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], []⟩, (1)⟩)

def exact7674RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], []⟩, (1)⟩]

theorem exact7674RawTermsValid :
    exact7674RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event7674 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37211⟩⟩) exact7674RawTerms (.finite 1764) 7672 .exactZero (none)

def event7675 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37212⟩⟩) 0 ⟨37211⟩ 7674

def event7676 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37212⟩⟩) (.identity (.predecessor 0 7675 .coefficient))

def event7677 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37212⟩⟩) (.finite 1764)

def event7678 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37460⟩⟩) 0 ⟨37212⟩ 7677

def event7679 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37460⟩⟩) (.authority (.programFamilyFact))

def eventLeaf464 : Array AnnotatedEvent := #[
  { event := event7424
    frameStart := 0 },
  { event := event7425
    frameStart := 0 },
  { event := event7426
    frameStart := 0 },
  { event := event7427
    frameStart := 0 },
  { event := event7428
    frameStart := 0 },
  { event := event7429
    frameStart := 0 },
  { event := event7430
    frameStart := 0 },
  { event := event7431
    frameStart := 0 },
  { event := event7432
    frameStart := 0 },
  { event := event7433
    frameStart := 0 },
  { event := event7434
    frameStart := 0 },
  { event := event7435
    frameStart := 0 },
  { event := event7436
    frameStart := 0 },
  { event := event7437
    frameStart := 0 },
  { event := event7438
    frameStart := 0 },
  { event := event7439
    frameStart := 0 }
]

def eventLeaf465 : Array AnnotatedEvent := #[
  { event := event7440
    frameStart := 0 },
  { event := event7441
    frameStart := 0 },
  { event := event7442
    frameStart := 0 },
  { event := event7443
    frameStart := 0 },
  { event := event7444
    frameStart := 0 },
  { event := event7445
    frameStart := 0 },
  { event := event7446
    frameStart := 0 },
  { event := event7447
    frameStart := 0 },
  { event := event7448
    frameStart := 0 },
  { event := event7449
    frameStart := 0 },
  { event := event7450
    frameStart := 0 },
  { event := event7451
    frameStart := 0 },
  { event := event7452
    frameStart := 0 },
  { event := event7453
    frameStart := 0 },
  { event := event7454
    frameStart := 0 },
  { event := event7455
    frameStart := 0 }
]

def eventLeaf466 : Array AnnotatedEvent := #[
  { event := event7456
    frameStart := 0 },
  { event := event7457
    frameStart := 0 },
  { event := event7458
    frameStart := 0 },
  { event := event7459
    frameStart := 0 },
  { event := event7460
    frameStart := 0 },
  { event := event7461
    frameStart := 0 },
  { event := event7462
    frameStart := 0 },
  { event := event7463
    frameStart := 0 },
  { event := event7464
    frameStart := 0 },
  { event := event7465
    frameStart := 0 },
  { event := event7466
    frameStart := 0 },
  { event := event7467
    frameStart := 0 },
  { event := event7468
    frameStart := 0 },
  { event := event7469
    frameStart := 0 },
  { event := event7470
    frameStart := 0 },
  { event := event7471
    frameStart := 0 }
]

def eventLeaf467 : Array AnnotatedEvent := #[
  { event := event7472
    frameStart := 0 },
  { event := event7473
    frameStart := 0 },
  { event := event7474
    frameStart := 0 },
  { event := event7475
    frameStart := 0 },
  { event := event7476
    frameStart := 0 },
  { event := event7477
    frameStart := 0 },
  { event := event7478
    frameStart := 0 },
  { event := event7479
    frameStart := 0 },
  { event := event7480
    frameStart := 0 },
  { event := event7481
    frameStart := 0 },
  { event := event7482
    frameStart := 0 },
  { event := event7483
    frameStart := 0 },
  { event := event7484
    frameStart := 0 },
  { event := event7485
    frameStart := 0 },
  { event := event7486
    frameStart := 0 },
  { event := event7487
    frameStart := 0 }
]

def eventLeaf468 : Array AnnotatedEvent := #[
  { event := event7488
    frameStart := 0 },
  { event := event7489
    frameStart := 0 },
  { event := event7490
    frameStart := 0 },
  { event := event7491
    frameStart := 0 },
  { event := event7492
    frameStart := 0 },
  { event := event7493
    frameStart := 0 },
  { event := event7494
    frameStart := 0 },
  { event := event7495
    frameStart := 0 },
  { event := event7496
    frameStart := 0 },
  { event := event7497
    frameStart := 0 },
  { event := event7498
    frameStart := 0 },
  { event := event7499
    frameStart := 0 },
  { event := event7500
    frameStart := 0 },
  { event := event7501
    frameStart := 0 },
  { event := event7502
    frameStart := 0 },
  { event := event7503
    frameStart := 0 }
]

def eventLeaf469 : Array AnnotatedEvent := #[
  { event := event7504
    frameStart := 0 },
  { event := event7505
    frameStart := 0 },
  { event := event7506
    frameStart := 0 },
  { event := event7507
    frameStart := 0 },
  { event := event7508
    frameStart := 0 },
  { event := event7509
    frameStart := 0 },
  { event := event7510
    frameStart := 0 },
  { event := event7511
    frameStart := 0 },
  { event := event7512
    frameStart := 0 },
  { event := event7513
    frameStart := 0 },
  { event := event7514
    frameStart := 0 },
  { event := event7515
    frameStart := 0 },
  { event := event7516
    frameStart := 0 },
  { event := event7517
    frameStart := 0 },
  { event := event7518
    frameStart := 0 },
  { event := event7519
    frameStart := 0 }
]

def eventLeaf470 : Array AnnotatedEvent := #[
  { event := event7520
    frameStart := 0 },
  { event := event7521
    frameStart := 0 },
  { event := event7522
    frameStart := 0 },
  { event := event7523
    frameStart := 0 },
  { event := event7524
    frameStart := 0 },
  { event := event7525
    frameStart := 0 },
  { event := event7526
    frameStart := 0 },
  { event := event7527
    frameStart := 0 },
  { event := event7528
    frameStart := 0 },
  { event := event7529
    frameStart := 0 },
  { event := event7530
    frameStart := 0 },
  { event := event7531
    frameStart := 0 },
  { event := event7532
    frameStart := 0 },
  { event := event7533
    frameStart := 0 },
  { event := event7534
    frameStart := 0 },
  { event := event7535
    frameStart := 0 }
]

def eventLeaf471 : Array AnnotatedEvent := #[
  { event := event7536
    frameStart := 0 },
  { event := event7537
    frameStart := 0 },
  { event := event7538
    frameStart := 0 },
  { event := event7539
    frameStart := 0 },
  { event := event7540
    frameStart := 0 },
  { event := event7541
    frameStart := 0 },
  { event := event7542
    frameStart := 0 },
  { event := event7543
    frameStart := 0 },
  { event := event7544
    frameStart := 0 },
  { event := event7545
    frameStart := 0 },
  { event := event7546
    frameStart := 0 },
  { event := event7547
    frameStart := 0 },
  { event := event7548
    frameStart := 0 },
  { event := event7549
    frameStart := 0 },
  { event := event7550
    frameStart := 0 },
  { event := event7551
    frameStart := 0 }
]

def eventLeaf472 : Array AnnotatedEvent := #[
  { event := event7552
    frameStart := 0 },
  { event := event7553
    frameStart := 0 },
  { event := event7554
    frameStart := 0 },
  { event := event7555
    frameStart := 0 },
  { event := event7556
    frameStart := 0 },
  { event := event7557
    frameStart := 0 },
  { event := event7558
    frameStart := 0 },
  { event := event7559
    frameStart := 0 },
  { event := event7560
    frameStart := 0 },
  { event := event7561
    frameStart := 0 },
  { event := event7562
    frameStart := 0 },
  { event := event7563
    frameStart := 0 },
  { event := event7564
    frameStart := 0 },
  { event := event7565
    frameStart := 0 },
  { event := event7566
    frameStart := 0 },
  { event := event7567
    frameStart := 0 }
]

def eventLeaf473 : Array AnnotatedEvent := #[
  { event := event7568
    frameStart := 0 },
  { event := event7569
    frameStart := 0 },
  { event := event7570
    frameStart := 0 },
  { event := event7571
    frameStart := 0 },
  { event := event7572
    frameStart := 0 },
  { event := event7573
    frameStart := 0 },
  { event := event7574
    frameStart := 0 },
  { event := event7575
    frameStart := 0 },
  { event := event7576
    frameStart := 0 },
  { event := event7577
    frameStart := 0 },
  { event := event7578
    frameStart := 0 },
  { event := event7579
    frameStart := 0 },
  { event := event7580
    frameStart := 0 },
  { event := event7581
    frameStart := 0 },
  { event := event7582
    frameStart := 0 },
  { event := event7583
    frameStart := 0 }
]

def eventLeaf474 : Array AnnotatedEvent := #[
  { event := event7584
    frameStart := 0 },
  { event := event7585
    frameStart := 0 },
  { event := event7586
    frameStart := 0 },
  { event := event7587
    frameStart := 0 },
  { event := event7588
    frameStart := 0 },
  { event := event7589
    frameStart := 0 },
  { event := event7590
    frameStart := 0 },
  { event := event7591
    frameStart := 0 },
  { event := event7592
    frameStart := 0 },
  { event := event7593
    frameStart := 0 },
  { event := event7594
    frameStart := 0 },
  { event := event7595
    frameStart := 0 },
  { event := event7596
    frameStart := 0 },
  { event := event7597
    frameStart := 0 },
  { event := event7598
    frameStart := 0 },
  { event := event7599
    frameStart := 0 }
]

def eventLeaf475 : Array AnnotatedEvent := #[
  { event := event7600
    frameStart := 0 },
  { event := event7601
    frameStart := 0 },
  { event := event7602
    frameStart := 0 },
  { event := event7603
    frameStart := 0 },
  { event := event7604
    frameStart := 0 },
  { event := event7605
    frameStart := 0 },
  { event := event7606
    frameStart := 0 },
  { event := event7607
    frameStart := 0 },
  { event := event7608
    frameStart := 0 },
  { event := event7609
    frameStart := 0 },
  { event := event7610
    frameStart := 0 },
  { event := event7611
    frameStart := 0 },
  { event := event7612
    frameStart := 0 },
  { event := event7613
    frameStart := 0 },
  { event := event7614
    frameStart := 0 },
  { event := event7615
    frameStart := 0 }
]

def eventLeaf476 : Array AnnotatedEvent := #[
  { event := event7616
    frameStart := 0 },
  { event := event7617
    frameStart := 0 },
  { event := event7618
    frameStart := 0 },
  { event := event7619
    frameStart := 0 },
  { event := event7620
    frameStart := 0 },
  { event := event7621
    frameStart := 0 },
  { event := event7622
    frameStart := 0 },
  { event := event7623
    frameStart := 0 },
  { event := event7624
    frameStart := 0 },
  { event := event7625
    frameStart := 0 },
  { event := event7626
    frameStart := 0 },
  { event := event7627
    frameStart := 0 },
  { event := event7628
    frameStart := 0 },
  { event := event7629
    frameStart := 0 },
  { event := event7630
    frameStart := 0 },
  { event := event7631
    frameStart := 0 }
]

def eventLeaf477 : Array AnnotatedEvent := #[
  { event := event7632
    frameStart := 0 },
  { event := event7633
    frameStart := 0 },
  { event := event7634
    frameStart := 0 },
  { event := event7635
    frameStart := 0 },
  { event := event7636
    frameStart := 0 },
  { event := event7637
    frameStart := 0 },
  { event := event7638
    frameStart := 0 },
  { event := event7639
    frameStart := 0 },
  { event := event7640
    frameStart := 0 },
  { event := event7641
    frameStart := 0 },
  { event := event7642
    frameStart := 0 },
  { event := event7643
    frameStart := 0 },
  { event := event7644
    frameStart := 0 },
  { event := event7645
    frameStart := 0 },
  { event := event7646
    frameStart := 0 },
  { event := event7647
    frameStart := 0 }
]

def eventLeaf478 : Array AnnotatedEvent := #[
  { event := event7648
    frameStart := 0 },
  { event := event7649
    frameStart := 0 },
  { event := event7650
    frameStart := 0 },
  { event := event7651
    frameStart := 0 },
  { event := event7652
    frameStart := 0 },
  { event := event7653
    frameStart := 0 },
  { event := event7654
    frameStart := 0 },
  { event := event7655
    frameStart := 0 },
  { event := event7656
    frameStart := 0 },
  { event := event7657
    frameStart := 0 },
  { event := event7658
    frameStart := 0 },
  { event := event7659
    frameStart := 0 },
  { event := event7660
    frameStart := 0 },
  { event := event7661
    frameStart := 0 },
  { event := event7662
    frameStart := 0 },
  { event := event7663
    frameStart := 0 }
]

def eventLeaf479 : Array AnnotatedEvent := #[
  { event := event7664
    frameStart := 0 },
  { event := event7665
    frameStart := 0 },
  { event := event7666
    frameStart := 0 },
  { event := event7667
    frameStart := 0 },
  { event := event7668
    frameStart := 0 },
  { event := event7669
    frameStart := 0 },
  { event := event7670
    frameStart := 0 },
  { event := event7671
    frameStart := 0 },
  { event := event7672
    frameStart := 0 },
  { event := event7673
    frameStart := 0 },
  { event := event7674
    frameStart := 0 },
  { event := event7675
    frameStart := 0 },
  { event := event7676
    frameStart := 0 },
  { event := event7677
    frameStart := 0 },
  { event := event7678
    frameStart := 0 },
  { event := event7679
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events029
