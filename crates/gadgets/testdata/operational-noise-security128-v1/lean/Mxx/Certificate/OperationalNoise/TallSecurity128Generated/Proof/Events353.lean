import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events353

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event90368 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10011⟩⟩) 0 ⟨9952⟩ 90367

def event90369 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10011⟩⟩) 1 ⟨5426⟩ 38

def event90370 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10011⟩⟩) (.identity (.predecessor 1 90369 .coefficient))

def event90371 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10011⟩⟩) (.finite 655360)

def event90372 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10012⟩⟩) 0 ⟨10011⟩ 90371

def event90373 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10012⟩⟩) 1 ⟨2370⟩ 4

def event90374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10012⟩⟩) (.sum [.predecessor 0 90372 .coefficient, .predecessor 1 90373 .coefficient])

def event90375 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10012⟩⟩) (.finite 655361)

def event90376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10013⟩⟩) 0 ⟨0⟩ 20

def event90377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10013⟩⟩) 1 ⟨10011⟩ 90371

def event90378 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10013⟩⟩) 2 ⟨10012⟩ 90375

def event90379 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10013⟩⟩) 3 ⟨136⟩ 6

def event90380 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10013⟩⟩) 4 ⟨2370⟩ 4

def event90381 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10013⟩⟩) (.identity (.predecessor 0 90376 .coefficient))

def exact90382RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨2377⟩⟩]⟩, (1)⟩]

theorem exact90382RawTermsValid :
    exact90382RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90382 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10013⟩⟩) exact90382RawTerms (.finite 1) 90381 .exactZero (none)

def event90383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10014⟩⟩) 0 ⟨10013⟩ 90382

def event90384 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10014⟩⟩) 1 ⟨6908⟩ 2

def event90385 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10014⟩⟩) (.product (.predecessor 0 90383 .coefficient) (.predecessor 1 90384 .coefficient) (⟨false, false, none, none, none⟩))

def event90386 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10014⟩⟩, .operator (⟨90382, 0⟩, ⟨2, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact90387RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact90387RawTermsValid :
    exact90387RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90387 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10014⟩⟩) exact90387RawTerms .large 90385 .exactZero (none)

def event90388 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9902⟩⟩) 0 ⟨9901⟩ 3831

def event90389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9902⟩⟩) 1 ⟨2370⟩ 4

def event90390 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9902⟩⟩) (.sum [.predecessor 0 90388 .coefficient, .predecessor 1 90389 .coefficient])

def event90391 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9902⟩⟩) (.finite 655361)

def event90392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9903⟩⟩) 0 ⟨0⟩ 20

def event90393 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9903⟩⟩) 1 ⟨9901⟩ 3831

def event90394 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9903⟩⟩) 2 ⟨9902⟩ 90391

def event90395 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9903⟩⟩) 3 ⟨136⟩ 6

def event90396 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9903⟩⟩) 4 ⟨2370⟩ 4

def event90397 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9903⟩⟩) (.identity (.predecessor 0 90392 .coefficient))

def exact90398RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨10270⟩⟩]⟩, (1)⟩]

theorem exact90398RawTermsValid :
    exact90398RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90398 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9903⟩⟩) exact90398RawTerms (.finite 1) 90397 .exactZero (none)

def event90399 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9905⟩⟩) 0 ⟨9903⟩ 90398

def event90400 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9905⟩⟩) 1 ⟨7245⟩ 16177

def event90401 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9905⟩⟩) (.product (.predecessor 0 90399 .coefficient) (.predecessor 1 90400 .coefficient) (⟨false, false, none, none, none⟩))

def event90402 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9905⟩⟩, .operator (⟨90398, 0⟩, ⟨16177, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def exact90403RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩]

theorem exact90403RawTermsValid :
    exact90403RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90403 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9905⟩⟩) exact90403RawTerms .large 90401 .exactZero (none)

def event90404 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10015⟩⟩) 0 ⟨9905⟩ 90403

def event90405 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10015⟩⟩) 1 ⟨10014⟩ 90387

def event90406 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10015⟩⟩) (.sum [.predecessor 0 90404 .coefficient, .predecessor 1 90405 .coefficient])

def exact90407RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩]

theorem exact90407RawTermsValid :
    exact90407RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90407 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10015⟩⟩) exact90407RawTerms .large 90406 .exactZero (none)

def event90408 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10016⟩⟩) 0 ⟨10015⟩ 90407

def event90409 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10016⟩⟩) 1 ⟨23⟩ 90361

def event90410 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10016⟩⟩) (.sum [.predecessor 0 90408 .coefficient, .predecessor 1 90409 .coefficient])

def event90411 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10016⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨23⟩⟩]⟩) [⟨.result 90361 .coefficient, false, none⟩])

def event90412 : Event := .survivorFold (1) 90411

def exact90413RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩]

theorem exact90413RawTermsValid :
    exact90413RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90413 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10016⟩⟩) exact90413RawTerms .large 90410 (.finite 26) (some (90411))

def event90414 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67572⟩⟩) 0 ⟨10016⟩ 90413

def event90415 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67572⟩⟩) 1 ⟨67569⟩ 4544

def event90416 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.product (.predecessor 0 90414 .coefficient) (.predecessor 1 90415 .coefficient) (⟨false, false, none, none, none⟩))

def event90417 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67566⟩⟩], []⟩) [⟨.result 36 .coefficient, true, some 1⟩, ⟨.result 4319 .coefficient, true, some 1⟩])

def event90418 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48424⟩⟩], []⟩) [⟨.result 543 .coefficient, true, some 1⟩, ⟨.result 4327 .coefficient, true, some 1⟩])

def event90419 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.sum [.transfer 90417, .transfer 90418])

def event90420 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45744⟩⟩], []⟩) [⟨.result 553 .coefficient, true, some 1⟩, ⟨.result 4335 .coefficient, true, some 1⟩])

def event90421 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.sum [.transfer 90419, .transfer 90420])

def event90422 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43067⟩⟩], []⟩) [⟨.result 563 .coefficient, true, some 1⟩, ⟨.result 4343 .coefficient, true, some 1⟩])

def event90423 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.sum [.transfer 90421, .transfer 90422])

def event90424 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], []⟩) [⟨.result 573 .coefficient, true, some 1⟩, ⟨.result 4351 .coefficient, true, some 1⟩])

def event90425 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.sum [.transfer 90423, .transfer 90424])

def event90426 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37704⟩⟩], []⟩) [⟨.result 583 .coefficient, true, some 1⟩, ⟨.result 4359 .coefficient, true, some 1⟩])

def event90427 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.sum [.transfer 90425, .transfer 90426])

def event90428 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], []⟩) [⟨.result 593 .coefficient, true, some 1⟩, ⟨.result 4367 .coefficient, true, some 1⟩])

def event90429 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.sum [.transfer 90427, .transfer 90428])

def event90430 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29367⟩⟩], []⟩) [⟨.result 603 .coefficient, true, some 1⟩, ⟨.result 4375 .coefficient, true, some 1⟩])

def event90431 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.sum [.transfer 90429, .transfer 90430])

def event90432 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26687⟩⟩], []⟩) [⟨.result 613 .coefficient, true, some 1⟩, ⟨.result 4383 .coefficient, true, some 1⟩])

def event90433 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.sum [.transfer 90431, .transfer 90432])

def event90434 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66938⟩⟩], []⟩) [⟨.result 623 .coefficient, true, some 1⟩, ⟨.result 4391 .coefficient, true, some 1⟩])

def event90435 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.sum [.transfer 90433, .transfer 90434])

def event90436 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63180⟩⟩], []⟩) [⟨.result 633 .coefficient, true, some 1⟩, ⟨.result 4399 .coefficient, true, some 1⟩])

def event90437 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.sum [.transfer 90435, .transfer 90436])

def event90438 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60200⟩⟩], []⟩) [⟨.result 643 .coefficient, true, some 1⟩, ⟨.result 4407 .coefficient, true, some 1⟩])

def event90439 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.sum [.transfer 90437, .transfer 90438])

def event90440 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57220⟩⟩], []⟩) [⟨.result 653 .coefficient, true, some 1⟩, ⟨.result 4415 .coefficient, true, some 1⟩])

def event90441 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.sum [.transfer 90439, .transfer 90440])

def event90442 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54240⟩⟩], []⟩) [⟨.result 663 .coefficient, true, some 1⟩, ⟨.result 4423 .coefficient, true, some 1⟩])

def event90443 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.sum [.transfer 90441, .transfer 90442])

def event90444 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], []⟩) [⟨.result 673 .coefficient, true, some 1⟩, ⟨.result 4431 .coefficient, true, some 1⟩])

def event90445 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.sum [.transfer 90443, .transfer 90444])

def event90446 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32196⟩⟩], []⟩) [⟨.result 683 .coefficient, true, some 1⟩, ⟨.result 4439 .coefficient, true, some 1⟩])

def event90447 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.sum [.transfer 90445, .transfer 90446])

def event90448 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], []⟩) [⟨.result 693 .coefficient, true, some 1⟩, ⟨.result 4447 .coefficient, true, some 1⟩])

def event90449 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.sum [.transfer 90447, .transfer 90448])

def event90450 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], []⟩) [⟨.result 703 .coefficient, true, some 1⟩, ⟨.result 4455 .coefficient, true, some 1⟩])

def event90451 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.sum [.transfer 90449, .transfer 90450])

def event90452 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], []⟩) [⟨.result 713 .coefficient, true, some 1⟩, ⟨.result 4463 .coefficient, true, some 1⟩])

def event90453 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.sum [.transfer 90451, .transfer 90452])

def event90454 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67572⟩⟩) (.product (.result 90413 .summary) (.transfer 90453) (⟨false, false, none, none, none⟩))

def event90455 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67566⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event90456 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90457 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45744⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90458 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90459 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90460 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90461 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90462 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29367⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90463 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26687⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90464 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 18⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66938⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90465 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90466 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90467 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90468 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54240⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90469 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90470 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90471 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90472 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90473 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 0⟩, ⟨4544, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90474 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨67566⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (-1)⟩)

def event90475 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨48424⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def event90476 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨45744⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def event90477 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨43067⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def event90478 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def event90479 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨37704⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def event90480 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def event90481 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29367⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def event90482 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨26687⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def event90483 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 18⟩), ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨66938⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def event90484 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨63180⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def event90485 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨60200⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def event90486 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨57220⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def event90487 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨54240⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def event90488 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def event90489 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨32196⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def event90490 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def event90491 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def event90492 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67572⟩⟩, .operator (⟨90413, 1⟩, ⟨4544, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩)

def exact90493RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54240⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67566⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48424⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45744⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37704⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29367⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26687⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66938⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨63180⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨60200⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨57220⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨54240⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨67566⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨32196⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨48424⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨45744⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨43067⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨37704⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29367⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨26687⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨66938⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩, (1)⟩]

theorem exact90493RawTermsValid :
    exact90493RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90493 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67572⟩⟩) exact90493RawTerms .large 90416 (.finite 6902113630329048043564518670336) (some (90454))

def event90494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68859⟩⟩) 0 ⟨66961⟩ 4316

def event90495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68859⟩⟩) (.authority (.programFamilyFact))

def event90496 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68859⟩⟩) (.finite 1152)

def event90497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68860⟩⟩) 0 ⟨7177⟩ 15500

def event90498 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68860⟩⟩) 1 ⟨68859⟩ 90496

def event90499 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68860⟩⟩) (.authority (.operator))

def exact90500RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68860⟩⟩]⟩, (1)⟩]

theorem exact90500RawTermsValid :
    exact90500RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90500 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68860⟩⟩) exact90500RawTerms .large 90499 .exactZero (none)

def event90501 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71405⟩⟩) 0 ⟨68860⟩ 90500

def event90502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71405⟩⟩) (.authority (.operator))

def exact90503RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩, (1)⟩]

theorem exact90503RawTermsValid :
    exact90503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90503 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71405⟩⟩) exact90503RawTerms (.finite 8192) 90502 .exactZero (none)

def event90504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49344⟩⟩) 0 ⟨48189⟩ 3851

def event90505 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49344⟩⟩) (.authority (.programFamilyFact))

def event90506 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49344⟩⟩) (.finite 3720)

def event90507 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49346⟩⟩) 0 ⟨7177⟩ 15500

def event90508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49346⟩⟩) 1 ⟨49344⟩ 90506

def event90509 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49346⟩⟩) (.authority (.operator))

def exact90510RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49346⟩⟩]⟩, (1)⟩]

theorem exact90510RawTermsValid :
    exact90510RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90510 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49346⟩⟩) exact90510RawTerms .large 90509 .exactZero (none)

def event90511 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50154⟩⟩) 0 ⟨49346⟩ 90510

def event90512 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50154⟩⟩) (.authority (.operator))

def exact90513RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨50154⟩⟩]⟩, (1)⟩]

theorem exact90513RawTermsValid :
    exact90513RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90513 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50154⟩⟩) exact90513RawTerms (.finite 8192) 90512 .exactZero (none)

def event90514 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49178⟩⟩) 0 ⟨47956⟩ 3845

def event90515 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49178⟩⟩) (.authority (.programFamilyFact))

def event90516 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49178⟩⟩) (.finite 3720)

def event90517 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49179⟩⟩) 0 ⟨7177⟩ 15500

def event90518 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49179⟩⟩) 1 ⟨49178⟩ 90516

def event90519 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49179⟩⟩) (.authority (.operator))

def exact90520RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49179⟩⟩]⟩, (1)⟩]

theorem exact90520RawTermsValid :
    exact90520RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90520 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49179⟩⟩) exact90520RawTerms .large 90519 .exactZero (none)

def event90521 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49714⟩⟩) 0 ⟨49179⟩ 90520

def event90522 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49714⟩⟩) (.authority (.operator))

def exact90523RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49714⟩⟩]⟩, (1)⟩]

theorem exact90523RawTermsValid :
    exact90523RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90523 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49714⟩⟩) exact90523RawTerms (.finite 8192) 90522 .exactZero (none)

def event90524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9904⟩⟩) 0 ⟨9903⟩ 90398

def event90525 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9904⟩⟩) 1 ⟨6908⟩ 2

def event90526 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9904⟩⟩) (.product (.predecessor 0 90524 .coefficient) (.predecessor 1 90525 .coefficient) (⟨false, false, none, none, none⟩))

def event90527 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9904⟩⟩, .operator (⟨90398, 0⟩, ⟨2, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact90528RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact90528RawTermsValid :
    exact90528RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90528 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9904⟩⟩) exact90528RawTerms .large 90526 .exactZero (none)

def event90529 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47957⟩⟩) 0 ⟨47954⟩ 3834

def event90530 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47957⟩⟩) 1 ⟨9904⟩ 90528

def event90531 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47957⟩⟩) (.tensor (.predecessor 0 90529 .coefficient) (.predecessor 1 90530 .coefficient) true false)

def event90532 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47957⟩⟩, .operator (⟨3834, 0⟩, ⟨90528, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact90533RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact90533RawTermsValid :
    exact90533RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90533 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47957⟩⟩) exact90533RawTerms .large 90531 .exactZero (none)

def event90534 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9919⟩⟩) 0 ⟨9903⟩ 90398

def event90535 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9919⟩⟩) 1 ⟨7285⟩ 17065

def event90536 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9919⟩⟩) (.product (.predecessor 0 90534 .coefficient) (.predecessor 1 90535 .coefficient) (⟨false, false, none, none, none⟩))

def event90537 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9919⟩⟩, .operator (⟨90398, 0⟩, ⟨17065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩)

def exact90538RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩]

theorem exact90538RawTermsValid :
    exact90538RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90538 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9919⟩⟩) exact90538RawTerms .large 90536 .exactZero (none)

def event90539 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47958⟩⟩) 0 ⟨9919⟩ 90538

def event90540 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47958⟩⟩) 1 ⟨47957⟩ 90533

def event90541 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47958⟩⟩) (.sum [.predecessor 0 90539 .coefficient, .predecessor 1 90540 .coefficient])

def exact90542RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact90542RawTermsValid :
    exact90542RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90542 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47958⟩⟩) exact90542RawTerms .large 90541 .exactZero (none)

def event90543 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47959⟩⟩) 0 ⟨47958⟩ 90542

def event90544 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47959⟩⟩) 1 ⟨111⟩ 17052

def event90545 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47959⟩⟩) (.sum [.predecessor 0 90543 .coefficient, .predecessor 1 90544 .coefficient])

def event90546 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47959⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨111⟩⟩]⟩) [⟨.result 17052 .coefficient, false, none⟩])

def event90547 : Event := .survivorFold (1) 90546

def exact90548RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact90548RawTermsValid :
    exact90548RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90548 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47959⟩⟩) exact90548RawTerms .large 90545 (.finite 26) (some (90546))

def event90549 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47960⟩⟩) 0 ⟨47959⟩ 90548

def event90550 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47960⟩⟩) 1 ⟨15156⟩ 3837

def event90551 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47960⟩⟩) (.product (.predecessor 0 90549 .coefficient) (.predecessor 1 90550 .coefficient) (⟨false, true, none, none, some 1⟩))

def event90552 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47960⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨15156⟩⟩], []⟩) [⟨.result 3837 .coefficient, true, some 1⟩])

def event90553 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47960⟩⟩) (.product (.result 90548 .summary) (.transfer 90552) (⟨false, false, none, none, none⟩))

def event90554 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47960⟩⟩, .operator (⟨90548, 1⟩, ⟨3837, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event90555 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47960⟩⟩, .operator (⟨90548, 0⟩, ⟨3837, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩)

def exact90556RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact90556RawTermsValid :
    exact90556RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90556 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47960⟩⟩) exact90556RawTerms .large 90551 (.finite 51118080) (some (90553))

def event90557 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15157⟩⟩) 0 ⟨15156⟩ 3837

def event90558 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15157⟩⟩) 1 ⟨9904⟩ 90528

def event90559 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15157⟩⟩) (.tensor (.predecessor 0 90557 .coefficient) (.predecessor 1 90558 .coefficient) true false)

def event90560 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15157⟩⟩, .operator (⟨3837, 0⟩, ⟨90528, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact90561RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact90561RawTermsValid :
    exact90561RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90561 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15157⟩⟩) exact90561RawTerms .large 90559 .exactZero (none)

def event90562 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9936⟩⟩) 0 ⟨9903⟩ 90398

def event90563 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9936⟩⟩) 1 ⟨7302⟩ 17106

def event90564 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9936⟩⟩) (.product (.predecessor 0 90562 .coefficient) (.predecessor 1 90563 .coefficient) (⟨false, false, none, none, none⟩))

def event90565 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9936⟩⟩, .operator (⟨90398, 0⟩, ⟨17106, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩)

def exact90566RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩]

theorem exact90566RawTermsValid :
    exact90566RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90566 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9936⟩⟩) exact90566RawTerms .large 90564 .exactZero (none)

def event90567 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15158⟩⟩) 0 ⟨9936⟩ 90566

def event90568 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15158⟩⟩) 1 ⟨15157⟩ 90561

def event90569 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15158⟩⟩) (.sum [.predecessor 0 90567 .coefficient, .predecessor 1 90568 .coefficient])

def exact90570RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact90570RawTermsValid :
    exact90570RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90570 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15158⟩⟩) exact90570RawTerms .large 90569 .exactZero (none)

def event90571 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15159⟩⟩) 0 ⟨15158⟩ 90570

def event90572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15159⟩⟩) 1 ⟨128⟩ 17098

def event90573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15159⟩⟩) (.sum [.predecessor 0 90571 .coefficient, .predecessor 1 90572 .coefficient])

def event90574 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15159⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨128⟩⟩]⟩) [⟨.result 17098 .coefficient, false, none⟩])

def event90575 : Event := .survivorFold (1) 90574

def exact90576RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact90576RawTermsValid :
    exact90576RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90576 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15159⟩⟩) exact90576RawTerms .large 90573 (.finite 26) (some (90574))

def event90577 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15160⟩⟩) 0 ⟨15159⟩ 90576

def event90578 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15160⟩⟩) 1 ⟨9566⟩ 17095

def event90579 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15160⟩⟩) (.product (.predecessor 0 90577 .coefficient) (.predecessor 1 90578 .coefficient) (⟨false, false, none, none, none⟩))

def event90580 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15160⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) [⟨.result 17091 .coefficient, false, none⟩])

def event90581 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15160⟩⟩) (.product (.result 90576 .summary) (.transfer 90580) (⟨false, false, none, none, none⟩))

def event90582 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15160⟩⟩, .operator (⟨90576, 1⟩, ⟨17095, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (-1)⟩)

def event90583 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨15160⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9565⟩⟩) ⟨7285⟩ 17065)

def event90584 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15160⟩⟩, .relation 90583 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (-1)⟩)

def event90585 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15160⟩⟩, .operator (⟨90576, 0⟩, ⟨17095, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩)

def exact90586RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (-1)⟩]

theorem exact90586RawTermsValid :
    exact90586RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90586 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15160⟩⟩) exact90586RawTerms .large 90579 (.finite 279172874240) (some (90581))

def event90587 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47961⟩⟩) 0 ⟨15160⟩ 90586

def event90588 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47961⟩⟩) 1 ⟨47960⟩ 90556

def event90589 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47961⟩⟩) (.sum [.predecessor 0 90587 .coefficient, .predecessor 1 90588 .coefficient])

def event90590 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47961⟩⟩, .operator (⟨90586, 1⟩, ⟨90556, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩)

def event90591 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47961⟩⟩) (.sum [.result 90586 .summary, .result 90556 .summary])

def exact90592RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact90592RawTermsValid :
    exact90592RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90592 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47961⟩⟩) exact90592RawTerms .large 90589 (.finite 279223992320) (some (90591))

def event90593 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49715⟩⟩) 0 ⟨47961⟩ 90592

def event90594 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49715⟩⟩) 1 ⟨49714⟩ 90523

def event90595 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49715⟩⟩) (.product (.predecessor 0 90593 .coefficient) (.predecessor 1 90594 .coefficient) (⟨false, false, none, none, none⟩))

def event90596 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49715⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨49714⟩⟩]⟩) [⟨.result 90523 .coefficient, false, none⟩])

def event90597 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49715⟩⟩) (.product (.result 90592 .summary) (.transfer 90596) (⟨false, false, none, none, none⟩))

def event90598 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49715⟩⟩, .operator (⟨90592, 1⟩, ⟨90523, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49714⟩⟩]⟩, (-1)⟩)

def event90599 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49715⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49714⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49714⟩⟩) ⟨49179⟩ 90520)

def event90600 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49715⟩⟩, .relation 90599 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], [⟨.program ⟨257⟩, ⟨49179⟩⟩]⟩, (-1)⟩)

def event90601 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49715⟩⟩, .operator (⟨90592, 0⟩, ⟨90523, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49714⟩⟩]⟩, (1)⟩)

def exact90602RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49714⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], [⟨.program ⟨257⟩, ⟨49179⟩⟩]⟩, (-1)⟩]

theorem exact90602RawTermsValid :
    exact90602RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90602 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49715⟩⟩) exact90602RawTerms .large 90595 (.finite 2998144788182387916800) (some (90597))

def event90603 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48639⟩⟩) 0 ⟨47956⟩ 3845

def event90604 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48639⟩⟩) (.authority (.relationPreimageSource ⟨54⟩))

def exact90605RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48639⟩⟩]⟩, (1)⟩]

theorem exact90605RawTermsValid :
    exact90605RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90605 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48639⟩⟩) exact90605RawTerms (.finite 5647228698) 90604 .exactZero (none)

def event90606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48641⟩⟩) 0 ⟨48639⟩ 90605

def event90607 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48641⟩⟩) 1 ⟨2370⟩ 4

def event90608 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48641⟩⟩) (.scale (.predecessor 0 90606 .coefficient) (.value (.predecessor 1 90607 .coefficient)))

def exact90609RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48639⟩⟩]⟩, (1)⟩]

theorem exact90609RawTermsValid :
    exact90609RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90609 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48641⟩⟩) exact90609RawTerms (.finite 5647228698) 90608 .exactZero (none)

def event90610 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9943⟩⟩) 0 ⟨9903⟩ 90398

def event90611 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9943⟩⟩) 1 ⟨35⟩ 17158

def event90612 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9943⟩⟩) (.product (.predecessor 0 90610 .coefficient) (.predecessor 1 90611 .coefficient) (⟨false, false, none, none, none⟩))

def event90613 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9943⟩⟩, .operator (⟨90398, 0⟩, ⟨17158, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩)

def exact90614RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact90614RawTermsValid :
    exact90614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90614 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9943⟩⟩) exact90614RawTerms .large 90612 .exactZero (none)

def event90615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9944⟩⟩) 0 ⟨9943⟩ 90614

def event90616 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9944⟩⟩) 1 ⟨22⟩ 17156

def event90617 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9944⟩⟩) (.sum [.predecessor 0 90615 .coefficient, .predecessor 1 90616 .coefficient])

def event90618 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9944⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨22⟩⟩]⟩) [⟨.result 17156 .coefficient, false, none⟩])

def event90619 : Event := .survivorFold (1) 90618

def exact90620RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact90620RawTermsValid :
    exact90620RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90620 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9944⟩⟩) exact90620RawTerms .large 90617 (.finite 26) (some (90618))

def event90621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48642⟩⟩) 0 ⟨9944⟩ 90620

def event90622 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48642⟩⟩) 1 ⟨48641⟩ 90609

def event90623 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48642⟩⟩) (.product (.predecessor 0 90621 .coefficient) (.predecessor 1 90622 .coefficient) (⟨false, false, none, none, none⟩))

def eventLeaf5648 : Array AnnotatedEvent := #[
  { event := event90368
    frameStart := 0 },
  { event := event90369
    frameStart := 0 },
  { event := event90370
    frameStart := 0 },
  { event := event90371
    frameStart := 0 },
  { event := event90372
    frameStart := 0 },
  { event := event90373
    frameStart := 0 },
  { event := event90374
    frameStart := 0 },
  { event := event90375
    frameStart := 0 },
  { event := event90376
    frameStart := 0 },
  { event := event90377
    frameStart := 0 },
  { event := event90378
    frameStart := 0 },
  { event := event90379
    frameStart := 0 },
  { event := event90380
    frameStart := 0 },
  { event := event90381
    frameStart := 0 },
  { event := event90382
    frameStart := 0 },
  { event := event90383
    frameStart := 0 }
]

def eventLeaf5649 : Array AnnotatedEvent := #[
  { event := event90384
    frameStart := 0 },
  { event := event90385
    frameStart := 0 },
  { event := event90386
    frameStart := 0 },
  { event := event90387
    frameStart := 0 },
  { event := event90388
    frameStart := 0 },
  { event := event90389
    frameStart := 0 },
  { event := event90390
    frameStart := 0 },
  { event := event90391
    frameStart := 0 },
  { event := event90392
    frameStart := 0 },
  { event := event90393
    frameStart := 0 },
  { event := event90394
    frameStart := 0 },
  { event := event90395
    frameStart := 0 },
  { event := event90396
    frameStart := 0 },
  { event := event90397
    frameStart := 0 },
  { event := event90398
    frameStart := 0 },
  { event := event90399
    frameStart := 0 }
]

def eventLeaf5650 : Array AnnotatedEvent := #[
  { event := event90400
    frameStart := 0 },
  { event := event90401
    frameStart := 0 },
  { event := event90402
    frameStart := 0 },
  { event := event90403
    frameStart := 0 },
  { event := event90404
    frameStart := 0 },
  { event := event90405
    frameStart := 0 },
  { event := event90406
    frameStart := 0 },
  { event := event90407
    frameStart := 0 },
  { event := event90408
    frameStart := 0 },
  { event := event90409
    frameStart := 0 },
  { event := event90410
    frameStart := 0 },
  { event := event90411
    frameStart := 0 },
  { event := event90412
    frameStart := 0 },
  { event := event90413
    frameStart := 0 },
  { event := event90414
    frameStart := 0 },
  { event := event90415
    frameStart := 0 }
]

def eventLeaf5651 : Array AnnotatedEvent := #[
  { event := event90416
    frameStart := 0 },
  { event := event90417
    frameStart := 0 },
  { event := event90418
    frameStart := 0 },
  { event := event90419
    frameStart := 0 },
  { event := event90420
    frameStart := 0 },
  { event := event90421
    frameStart := 0 },
  { event := event90422
    frameStart := 0 },
  { event := event90423
    frameStart := 0 },
  { event := event90424
    frameStart := 0 },
  { event := event90425
    frameStart := 0 },
  { event := event90426
    frameStart := 0 },
  { event := event90427
    frameStart := 0 },
  { event := event90428
    frameStart := 0 },
  { event := event90429
    frameStart := 0 },
  { event := event90430
    frameStart := 0 },
  { event := event90431
    frameStart := 0 }
]

def eventLeaf5652 : Array AnnotatedEvent := #[
  { event := event90432
    frameStart := 0 },
  { event := event90433
    frameStart := 0 },
  { event := event90434
    frameStart := 0 },
  { event := event90435
    frameStart := 0 },
  { event := event90436
    frameStart := 0 },
  { event := event90437
    frameStart := 0 },
  { event := event90438
    frameStart := 0 },
  { event := event90439
    frameStart := 0 },
  { event := event90440
    frameStart := 0 },
  { event := event90441
    frameStart := 0 },
  { event := event90442
    frameStart := 0 },
  { event := event90443
    frameStart := 0 },
  { event := event90444
    frameStart := 0 },
  { event := event90445
    frameStart := 0 },
  { event := event90446
    frameStart := 0 },
  { event := event90447
    frameStart := 0 }
]

def eventLeaf5653 : Array AnnotatedEvent := #[
  { event := event90448
    frameStart := 0 },
  { event := event90449
    frameStart := 0 },
  { event := event90450
    frameStart := 0 },
  { event := event90451
    frameStart := 0 },
  { event := event90452
    frameStart := 0 },
  { event := event90453
    frameStart := 0 },
  { event := event90454
    frameStart := 0 },
  { event := event90455
    frameStart := 0 },
  { event := event90456
    frameStart := 0 },
  { event := event90457
    frameStart := 0 },
  { event := event90458
    frameStart := 0 },
  { event := event90459
    frameStart := 0 },
  { event := event90460
    frameStart := 0 },
  { event := event90461
    frameStart := 0 },
  { event := event90462
    frameStart := 0 },
  { event := event90463
    frameStart := 0 }
]

def eventLeaf5654 : Array AnnotatedEvent := #[
  { event := event90464
    frameStart := 0 },
  { event := event90465
    frameStart := 0 },
  { event := event90466
    frameStart := 0 },
  { event := event90467
    frameStart := 0 },
  { event := event90468
    frameStart := 0 },
  { event := event90469
    frameStart := 0 },
  { event := event90470
    frameStart := 0 },
  { event := event90471
    frameStart := 0 },
  { event := event90472
    frameStart := 0 },
  { event := event90473
    frameStart := 0 },
  { event := event90474
    frameStart := 0 },
  { event := event90475
    frameStart := 0 },
  { event := event90476
    frameStart := 0 },
  { event := event90477
    frameStart := 0 },
  { event := event90478
    frameStart := 0 },
  { event := event90479
    frameStart := 0 }
]

def eventLeaf5655 : Array AnnotatedEvent := #[
  { event := event90480
    frameStart := 0 },
  { event := event90481
    frameStart := 0 },
  { event := event90482
    frameStart := 0 },
  { event := event90483
    frameStart := 0 },
  { event := event90484
    frameStart := 0 },
  { event := event90485
    frameStart := 0 },
  { event := event90486
    frameStart := 0 },
  { event := event90487
    frameStart := 0 },
  { event := event90488
    frameStart := 0 },
  { event := event90489
    frameStart := 0 },
  { event := event90490
    frameStart := 0 },
  { event := event90491
    frameStart := 0 },
  { event := event90492
    frameStart := 0 },
  { event := event90493
    frameStart := 0 },
  { event := event90494
    frameStart := 0 },
  { event := event90495
    frameStart := 0 }
]

def eventLeaf5656 : Array AnnotatedEvent := #[
  { event := event90496
    frameStart := 0 },
  { event := event90497
    frameStart := 0 },
  { event := event90498
    frameStart := 0 },
  { event := event90499
    frameStart := 0 },
  { event := event90500
    frameStart := 0 },
  { event := event90501
    frameStart := 0 },
  { event := event90502
    frameStart := 0 },
  { event := event90503
    frameStart := 0 },
  { event := event90504
    frameStart := 0 },
  { event := event90505
    frameStart := 0 },
  { event := event90506
    frameStart := 0 },
  { event := event90507
    frameStart := 0 },
  { event := event90508
    frameStart := 0 },
  { event := event90509
    frameStart := 0 },
  { event := event90510
    frameStart := 0 },
  { event := event90511
    frameStart := 0 }
]

def eventLeaf5657 : Array AnnotatedEvent := #[
  { event := event90512
    frameStart := 0 },
  { event := event90513
    frameStart := 0 },
  { event := event90514
    frameStart := 0 },
  { event := event90515
    frameStart := 0 },
  { event := event90516
    frameStart := 0 },
  { event := event90517
    frameStart := 0 },
  { event := event90518
    frameStart := 0 },
  { event := event90519
    frameStart := 0 },
  { event := event90520
    frameStart := 0 },
  { event := event90521
    frameStart := 0 },
  { event := event90522
    frameStart := 0 },
  { event := event90523
    frameStart := 0 },
  { event := event90524
    frameStart := 0 },
  { event := event90525
    frameStart := 0 },
  { event := event90526
    frameStart := 0 },
  { event := event90527
    frameStart := 0 }
]

def eventLeaf5658 : Array AnnotatedEvent := #[
  { event := event90528
    frameStart := 0 },
  { event := event90529
    frameStart := 0 },
  { event := event90530
    frameStart := 0 },
  { event := event90531
    frameStart := 0 },
  { event := event90532
    frameStart := 0 },
  { event := event90533
    frameStart := 0 },
  { event := event90534
    frameStart := 0 },
  { event := event90535
    frameStart := 0 },
  { event := event90536
    frameStart := 0 },
  { event := event90537
    frameStart := 0 },
  { event := event90538
    frameStart := 0 },
  { event := event90539
    frameStart := 0 },
  { event := event90540
    frameStart := 0 },
  { event := event90541
    frameStart := 0 },
  { event := event90542
    frameStart := 0 },
  { event := event90543
    frameStart := 0 }
]

def eventLeaf5659 : Array AnnotatedEvent := #[
  { event := event90544
    frameStart := 0 },
  { event := event90545
    frameStart := 0 },
  { event := event90546
    frameStart := 0 },
  { event := event90547
    frameStart := 0 },
  { event := event90548
    frameStart := 0 },
  { event := event90549
    frameStart := 0 },
  { event := event90550
    frameStart := 0 },
  { event := event90551
    frameStart := 0 },
  { event := event90552
    frameStart := 0 },
  { event := event90553
    frameStart := 0 },
  { event := event90554
    frameStart := 0 },
  { event := event90555
    frameStart := 0 },
  { event := event90556
    frameStart := 0 },
  { event := event90557
    frameStart := 0 },
  { event := event90558
    frameStart := 0 },
  { event := event90559
    frameStart := 0 }
]

def eventLeaf5660 : Array AnnotatedEvent := #[
  { event := event90560
    frameStart := 0 },
  { event := event90561
    frameStart := 0 },
  { event := event90562
    frameStart := 0 },
  { event := event90563
    frameStart := 0 },
  { event := event90564
    frameStart := 0 },
  { event := event90565
    frameStart := 0 },
  { event := event90566
    frameStart := 0 },
  { event := event90567
    frameStart := 0 },
  { event := event90568
    frameStart := 0 },
  { event := event90569
    frameStart := 0 },
  { event := event90570
    frameStart := 0 },
  { event := event90571
    frameStart := 0 },
  { event := event90572
    frameStart := 0 },
  { event := event90573
    frameStart := 0 },
  { event := event90574
    frameStart := 0 },
  { event := event90575
    frameStart := 0 }
]

def eventLeaf5661 : Array AnnotatedEvent := #[
  { event := event90576
    frameStart := 0 },
  { event := event90577
    frameStart := 0 },
  { event := event90578
    frameStart := 0 },
  { event := event90579
    frameStart := 0 },
  { event := event90580
    frameStart := 0 },
  { event := event90581
    frameStart := 0 },
  { event := event90582
    frameStart := 0 },
  { event := event90583
    frameStart := 0 },
  { event := event90584
    frameStart := 0 },
  { event := event90585
    frameStart := 0 },
  { event := event90586
    frameStart := 0 },
  { event := event90587
    frameStart := 0 },
  { event := event90588
    frameStart := 0 },
  { event := event90589
    frameStart := 0 },
  { event := event90590
    frameStart := 0 },
  { event := event90591
    frameStart := 0 }
]

def eventLeaf5662 : Array AnnotatedEvent := #[
  { event := event90592
    frameStart := 0 },
  { event := event90593
    frameStart := 0 },
  { event := event90594
    frameStart := 0 },
  { event := event90595
    frameStart := 0 },
  { event := event90596
    frameStart := 0 },
  { event := event90597
    frameStart := 0 },
  { event := event90598
    frameStart := 0 },
  { event := event90599
    frameStart := 0 },
  { event := event90600
    frameStart := 0 },
  { event := event90601
    frameStart := 0 },
  { event := event90602
    frameStart := 0 },
  { event := event90603
    frameStart := 0 },
  { event := event90604
    frameStart := 0 },
  { event := event90605
    frameStart := 0 },
  { event := event90606
    frameStart := 0 },
  { event := event90607
    frameStart := 0 }
]

def eventLeaf5663 : Array AnnotatedEvent := #[
  { event := event90608
    frameStart := 0 },
  { event := event90609
    frameStart := 0 },
  { event := event90610
    frameStart := 0 },
  { event := event90611
    frameStart := 0 },
  { event := event90612
    frameStart := 0 },
  { event := event90613
    frameStart := 0 },
  { event := event90614
    frameStart := 0 },
  { event := event90615
    frameStart := 0 },
  { event := event90616
    frameStart := 0 },
  { event := event90617
    frameStart := 0 },
  { event := event90618
    frameStart := 0 },
  { event := event90619
    frameStart := 0 },
  { event := event90620
    frameStart := 0 },
  { event := event90621
    frameStart := 0 },
  { event := event90622
    frameStart := 0 },
  { event := event90623
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events353
