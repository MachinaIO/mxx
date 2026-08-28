import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events810

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event207360 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5⟩⟩) (.authority (.operator))

def exact207361RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨5⟩⟩]⟩, (1)⟩]

theorem exact207361RawTermsValid :
    exact207361RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207361 : Event := .resultExact (⟨.program ⟨257⟩, ⟨5⟩⟩) exact207361RawTerms (.finite 26) 207360 .exactZero (none)

def event207362 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5414⟩⟩) (.authority (.operator))

def event207363 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5414⟩⟩) (.finite 655354)

def event207364 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5600⟩⟩) 0 ⟨5595⟩ 9815

def event207365 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5600⟩⟩) 1 ⟨5414⟩ 207363

def event207366 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5600⟩⟩) (.sum [.predecessor 0 207364 .coefficient, .predecessor 1 207365 .coefficient])

def event207367 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5600⟩⟩) (.finite 1310714)

def event207368 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5748⟩⟩) 0 ⟨5600⟩ 207367

def event207369 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5748⟩⟩) 1 ⟨5426⟩ 38

def event207370 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5748⟩⟩) (.identity (.predecessor 1 207369 .coefficient))

def event207371 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5748⟩⟩) (.finite 655360)

def event207372 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5749⟩⟩) 0 ⟨5748⟩ 207371

def event207373 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5749⟩⟩) 1 ⟨2370⟩ 4

def event207374 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5749⟩⟩) (.sum [.predecessor 0 207372 .coefficient, .predecessor 1 207373 .coefficient])

def event207375 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5749⟩⟩) (.finite 655361)

def event207376 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5750⟩⟩) 0 ⟨0⟩ 20

def event207377 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5750⟩⟩) 1 ⟨5748⟩ 207371

def event207378 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5750⟩⟩) 2 ⟨5749⟩ 207375

def event207379 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5750⟩⟩) 3 ⟨136⟩ 6

def event207380 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5750⟩⟩) 4 ⟨2370⟩ 4

def event207381 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5750⟩⟩) (.identity (.predecessor 0 207376 .coefficient))

def exact207382RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨2377⟩⟩]⟩, (1)⟩]

theorem exact207382RawTermsValid :
    exact207382RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207382 : Event := .resultExact (⟨.program ⟨257⟩, ⟨5750⟩⟩) exact207382RawTerms (.finite 1) 207381 .exactZero (none)

def event207383 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6989⟩⟩) 0 ⟨5750⟩ 207382

def event207384 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6989⟩⟩) 1 ⟨6908⟩ 2

def event207385 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6989⟩⟩) (.product (.predecessor 0 207383 .coefficient) (.predecessor 1 207384 .coefficient) (⟨false, false, none, none, none⟩))

def event207386 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨6989⟩⟩, .operator (⟨207382, 0⟩, ⟨2, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact207387RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact207387RawTermsValid :
    exact207387RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207387 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6989⟩⟩) exact207387RawTerms .large 207385 .exactZero (none)

def event207388 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5596⟩⟩) 0 ⟨5595⟩ 9815

def event207389 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5596⟩⟩) 1 ⟨2370⟩ 4

def event207390 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5596⟩⟩) (.sum [.predecessor 0 207388 .coefficient, .predecessor 1 207389 .coefficient])

def event207391 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5596⟩⟩) (.finite 655361)

def event207392 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5597⟩⟩) 0 ⟨0⟩ 20

def event207393 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5597⟩⟩) 1 ⟨5595⟩ 9815

def event207394 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5597⟩⟩) 2 ⟨5596⟩ 207391

def event207395 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5597⟩⟩) 3 ⟨136⟩ 6

def event207396 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5597⟩⟩) 4 ⟨2370⟩ 4

def event207397 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5597⟩⟩) (.identity (.predecessor 0 207392 .coefficient))

def exact207398RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨5896⟩⟩]⟩, (1)⟩]

theorem exact207398RawTermsValid :
    exact207398RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207398 : Event := .resultExact (⟨.program ⟨257⟩, ⟨5597⟩⟩) exact207398RawTerms (.finite 1) 207397 .exactZero (none)

def event207399 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8577⟩⟩) 0 ⟨5597⟩ 207398

def event207400 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8577⟩⟩) 1 ⟨7261⟩ 16497

def event207401 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8577⟩⟩) (.product (.predecessor 0 207399 .coefficient) (.predecessor 1 207400 .coefficient) (⟨false, false, none, none, none⟩))

def event207402 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8577⟩⟩, .operator (⟨207398, 0⟩, ⟨16497, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def exact207403RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩]

theorem exact207403RawTermsValid :
    exact207403RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207403 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8577⟩⟩) exact207403RawTerms .large 207401 .exactZero (none)

def event207404 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9387⟩⟩) 0 ⟨8577⟩ 207403

def event207405 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9387⟩⟩) 1 ⟨6989⟩ 207387

def event207406 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9387⟩⟩) (.sum [.predecessor 0 207404 .coefficient, .predecessor 1 207405 .coefficient])

def exact207407RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩]

theorem exact207407RawTermsValid :
    exact207407RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207407 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9387⟩⟩) exact207407RawTerms .large 207406 .exactZero (none)

def event207408 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9388⟩⟩) 0 ⟨9387⟩ 207407

def event207409 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9388⟩⟩) 1 ⟨5⟩ 207361

def event207410 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9388⟩⟩) (.sum [.predecessor 0 207408 .coefficient, .predecessor 1 207409 .coefficient])

def event207411 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9388⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨5⟩⟩]⟩) [⟨.result 207361 .coefficient, false, none⟩])

def event207412 : Event := .survivorFold (1) 207411

def exact207413RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩]

theorem exact207413RawTermsValid :
    exact207413RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207413 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9388⟩⟩) exact207413RawTerms .large 207410 (.finite 26) (some (207411))

def event207414 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67463⟩⟩) 0 ⟨9388⟩ 207413

def event207415 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67463⟩⟩) 1 ⟨67460⟩ 10528

def event207416 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.product (.predecessor 0 207414 .coefficient) (.predecessor 1 207415 .coefficient) (⟨false, false, none, none, none⟩))

def event207417 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], []⟩) [⟨.result 36 .coefficient, true, some 1⟩, ⟨.result 10303 .coefficient, true, some 1⟩])

def event207418 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48359⟩⟩], []⟩) [⟨.result 543 .coefficient, true, some 1⟩, ⟨.result 10311 .coefficient, true, some 1⟩])

def event207419 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.sum [.transfer 207417, .transfer 207418])

def event207420 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45679⟩⟩], []⟩) [⟨.result 553 .coefficient, true, some 1⟩, ⟨.result 10319 .coefficient, true, some 1⟩])

def event207421 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.sum [.transfer 207419, .transfer 207420])

def event207422 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43002⟩⟩], []⟩) [⟨.result 563 .coefficient, true, some 1⟩, ⟨.result 10327 .coefficient, true, some 1⟩])

def event207423 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.sum [.transfer 207421, .transfer 207422])

def event207424 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40322⟩⟩], []⟩) [⟨.result 573 .coefficient, true, some 1⟩, ⟨.result 10335 .coefficient, true, some 1⟩])

def event207425 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.sum [.transfer 207423, .transfer 207424])

def event207426 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37639⟩⟩], []⟩) [⟨.result 583 .coefficient, true, some 1⟩, ⟨.result 10343 .coefficient, true, some 1⟩])

def event207427 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.sum [.transfer 207425, .transfer 207426])

def event207428 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34959⟩⟩], []⟩) [⟨.result 593 .coefficient, true, some 1⟩, ⟨.result 10351 .coefficient, true, some 1⟩])

def event207429 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.sum [.transfer 207427, .transfer 207428])

def event207430 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29302⟩⟩], []⟩) [⟨.result 603 .coefficient, true, some 1⟩, ⟨.result 10359 .coefficient, true, some 1⟩])

def event207431 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.sum [.transfer 207429, .transfer 207430])

def event207432 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26622⟩⟩], []⟩) [⟨.result 613 .coefficient, true, some 1⟩, ⟨.result 10367 .coefficient, true, some 1⟩])

def event207433 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.sum [.transfer 207431, .transfer 207432])

def event207434 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66588⟩⟩], []⟩) [⟨.result 623 .coefficient, true, some 1⟩, ⟨.result 10375 .coefficient, true, some 1⟩])

def event207435 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.sum [.transfer 207433, .transfer 207434])

def event207436 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63085⟩⟩], []⟩) [⟨.result 633 .coefficient, true, some 1⟩, ⟨.result 10383 .coefficient, true, some 1⟩])

def event207437 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.sum [.transfer 207435, .transfer 207436])

def event207438 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], []⟩) [⟨.result 643 .coefficient, true, some 1⟩, ⟨.result 10391 .coefficient, true, some 1⟩])

def event207439 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.sum [.transfer 207437, .transfer 207438])

def event207440 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57125⟩⟩], []⟩) [⟨.result 653 .coefficient, true, some 1⟩, ⟨.result 10399 .coefficient, true, some 1⟩])

def event207441 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.sum [.transfer 207439, .transfer 207440])

def event207442 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], []⟩) [⟨.result 663 .coefficient, true, some 1⟩, ⟨.result 10407 .coefficient, true, some 1⟩])

def event207443 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.sum [.transfer 207441, .transfer 207442])

def event207444 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51165⟩⟩], []⟩) [⟨.result 673 .coefficient, true, some 1⟩, ⟨.result 10415 .coefficient, true, some 1⟩])

def event207445 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.sum [.transfer 207443, .transfer 207444])

def event207446 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32101⟩⟩], []⟩) [⟨.result 683 .coefficient, true, some 1⟩, ⟨.result 10423 .coefficient, true, some 1⟩])

def event207447 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.sum [.transfer 207445, .transfer 207446])

def event207448 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22081⟩⟩], []⟩) [⟨.result 693 .coefficient, true, some 1⟩, ⟨.result 10431 .coefficient, true, some 1⟩])

def event207449 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.sum [.transfer 207447, .transfer 207448])

def event207450 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18861⟩⟩], []⟩) [⟨.result 703 .coefficient, true, some 1⟩, ⟨.result 10439 .coefficient, true, some 1⟩])

def event207451 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.sum [.transfer 207449, .transfer 207450])

def event207452 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16030⟩⟩], []⟩) [⟨.result 713 .coefficient, true, some 1⟩, ⟨.result 10447 .coefficient, true, some 1⟩])

def event207453 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.sum [.transfer 207451, .transfer 207452])

def event207454 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67463⟩⟩) (.product (.result 207413 .summary) (.transfer 207453) (⟨false, false, none, none, none⟩))

def event207455 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event207456 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48359⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207457 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45679⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207458 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43002⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207459 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40322⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207460 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37639⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207461 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34959⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207462 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29302⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207463 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26622⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207464 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 18⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66588⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207465 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63085⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207466 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207467 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57125⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207468 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207469 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51165⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207470 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207471 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207472 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18861⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207473 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 0⟩, ⟨10528, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16030⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207474 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 5⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (-1)⟩)

def event207475 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 7⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48359⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def event207476 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 8⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45679⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def event207477 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 9⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43002⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def event207478 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 11⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40322⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def event207479 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 12⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37639⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def event207480 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 13⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34959⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def event207481 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 15⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29302⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def event207482 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 16⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26622⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def event207483 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 18⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66588⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def event207484 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63085⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def event207485 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def event207486 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 2⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57125⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def event207487 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 3⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def event207488 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 4⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51165⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def event207489 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 6⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32101⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def event207490 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 10⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22081⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def event207491 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 14⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18861⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def event207492 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67463⟩⟩, .operator (⟨207413, 1⟩, ⟨10528, 17⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16030⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩)

def exact207493RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63085⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57125⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51165⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48359⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45679⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43002⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40322⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37639⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34959⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18861⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29302⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26622⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16030⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66588⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63085⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57125⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54145⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51165⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32101⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48359⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45679⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43002⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22081⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40322⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37639⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34959⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18861⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29302⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26622⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16030⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66588⟩⟩], [⟨.program ⟨257⟩, ⟨7261⟩⟩]⟩, (1)⟩]

theorem exact207493RawTermsValid :
    exact207493RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207493 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67463⟩⟩) exact207493RawTerms .large 207416 (.finite 6902113630329048043564518670336) (some (207454))

def event207494 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68829⟩⟩) 0 ⟨66611⟩ 10300

def event207495 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68829⟩⟩) (.authority (.programFamilyFact))

def event207496 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68829⟩⟩) (.finite 1152)

def event207497 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68830⟩⟩) 0 ⟨7177⟩ 15500

def event207498 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68830⟩⟩) 1 ⟨68829⟩ 207496

def event207499 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68830⟩⟩) (.authority (.operator))

def exact207500RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩, (1)⟩]

theorem exact207500RawTermsValid :
    exact207500RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207500 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68830⟩⟩) exact207500RawTerms .large 207499 .exactZero (none)

def event207501 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71236⟩⟩) 0 ⟨68830⟩ 207500

def event207502 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71236⟩⟩) (.authority (.operator))

def exact207503RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩, (1)⟩]

theorem exact207503RawTermsValid :
    exact207503RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207503 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71236⟩⟩) exact207503RawTerms (.finite 8192) 207502 .exactZero (none)

def event207504 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49299⟩⟩) 0 ⟨48149⟩ 9835

def event207505 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49299⟩⟩) (.authority (.programFamilyFact))

def event207506 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49299⟩⟩) (.finite 3720)

def event207507 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49301⟩⟩) 0 ⟨7177⟩ 15500

def event207508 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49301⟩⟩) 1 ⟨49299⟩ 207506

def event207509 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49301⟩⟩) (.authority (.operator))

def exact207510RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49301⟩⟩]⟩, (1)⟩]

theorem exact207510RawTermsValid :
    exact207510RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207510 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49301⟩⟩) exact207510RawTerms .large 207509 .exactZero (none)

def event207511 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50029⟩⟩) 0 ⟨49301⟩ 207510

def event207512 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50029⟩⟩) (.authority (.operator))

def exact207513RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨50029⟩⟩]⟩, (1)⟩]

theorem exact207513RawTermsValid :
    exact207513RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207513 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50029⟩⟩) exact207513RawTerms (.finite 8192) 207512 .exactZero (none)

def event207514 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49148⟩⟩) 0 ⟨47836⟩ 9829

def event207515 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49148⟩⟩) (.authority (.programFamilyFact))

def event207516 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49148⟩⟩) (.finite 3720)

def event207517 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49149⟩⟩) 0 ⟨7177⟩ 15500

def event207518 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49149⟩⟩) 1 ⟨49148⟩ 207516

def event207519 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49149⟩⟩) (.authority (.operator))

def exact207520RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49149⟩⟩]⟩, (1)⟩]

theorem exact207520RawTermsValid :
    exact207520RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207520 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49149⟩⟩) exact207520RawTerms .large 207519 .exactZero (none)

def event207521 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49659⟩⟩) 0 ⟨49149⟩ 207520

def event207522 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49659⟩⟩) (.authority (.operator))

def exact207523RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49659⟩⟩]⟩, (1)⟩]

theorem exact207523RawTermsValid :
    exact207523RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207523 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49659⟩⟩) exact207523RawTerms (.finite 8192) 207522 .exactZero (none)

def event207524 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6940⟩⟩) 0 ⟨5597⟩ 207398

def event207525 : Event := .predecessor (⟨.program ⟨257⟩, ⟨6940⟩⟩) 1 ⟨6908⟩ 2

def event207526 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6940⟩⟩) (.product (.predecessor 0 207524 .coefficient) (.predecessor 1 207525 .coefficient) (⟨false, false, none, none, none⟩))

def event207527 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨6940⟩⟩, .operator (⟨207398, 0⟩, ⟨2, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact207528RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact207528RawTermsValid :
    exact207528RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207528 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6940⟩⟩) exact207528RawTerms .large 207526 .exactZero (none)

def event207529 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47837⟩⟩) 0 ⟨47834⟩ 9818

def event207530 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47837⟩⟩) 1 ⟨6940⟩ 207528

def event207531 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47837⟩⟩) (.tensor (.predecessor 0 207529 .coefficient) (.predecessor 1 207530 .coefficient) true false)

def event207532 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47837⟩⟩, .operator (⟨9818, 0⟩, ⟨207528, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact207533RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact207533RawTermsValid :
    exact207533RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207533 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47837⟩⟩) exact207533RawTerms .large 207531 .exactZero (none)

def event207534 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8591⟩⟩) 0 ⟨5597⟩ 207398

def event207535 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8591⟩⟩) 1 ⟨7285⟩ 17065

def event207536 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8591⟩⟩) (.product (.predecessor 0 207534 .coefficient) (.predecessor 1 207535 .coefficient) (⟨false, false, none, none, none⟩))

def event207537 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8591⟩⟩, .operator (⟨207398, 0⟩, ⟨17065, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩)

def exact207538RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩]

theorem exact207538RawTermsValid :
    exact207538RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207538 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8591⟩⟩) exact207538RawTerms .large 207536 .exactZero (none)

def event207539 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47838⟩⟩) 0 ⟨8591⟩ 207538

def event207540 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47838⟩⟩) 1 ⟨47837⟩ 207533

def event207541 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47838⟩⟩) (.sum [.predecessor 0 207539 .coefficient, .predecessor 1 207540 .coefficient])

def exact207542RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207542RawTermsValid :
    exact207542RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207542 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47838⟩⟩) exact207542RawTerms .large 207541 .exactZero (none)

def event207543 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47839⟩⟩) 0 ⟨47838⟩ 207542

def event207544 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47839⟩⟩) 1 ⟨111⟩ 17052

def event207545 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47839⟩⟩) (.sum [.predecessor 0 207543 .coefficient, .predecessor 1 207544 .coefficient])

def event207546 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47839⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨111⟩⟩]⟩) [⟨.result 17052 .coefficient, false, none⟩])

def event207547 : Event := .survivorFold (1) 207546

def exact207548RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207548RawTermsValid :
    exact207548RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207548 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47839⟩⟩) exact207548RawTerms .large 207545 (.finite 26) (some (207546))

def event207549 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47840⟩⟩) 0 ⟨47839⟩ 207548

def event207550 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47840⟩⟩) 1 ⟨15081⟩ 9821

def event207551 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47840⟩⟩) (.product (.predecessor 0 207549 .coefficient) (.predecessor 1 207550 .coefficient) (⟨false, true, none, none, some 1⟩))

def event207552 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47840⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨15081⟩⟩], []⟩) [⟨.result 9821 .coefficient, true, some 1⟩])

def event207553 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47840⟩⟩) (.product (.result 207548 .summary) (.transfer 207552) (⟨false, false, none, none, none⟩))

def event207554 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47840⟩⟩, .operator (⟨207548, 1⟩, ⟨9821, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event207555 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47840⟩⟩, .operator (⟨207548, 0⟩, ⟨9821, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩)

def exact207556RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207556RawTermsValid :
    exact207556RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207556 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47840⟩⟩) exact207556RawTerms .large 207551 (.finite 51118080) (some (207553))

def event207557 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15082⟩⟩) 0 ⟨15081⟩ 9821

def event207558 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15082⟩⟩) 1 ⟨6940⟩ 207528

def event207559 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15082⟩⟩) (.tensor (.predecessor 0 207557 .coefficient) (.predecessor 1 207558 .coefficient) true false)

def event207560 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15082⟩⟩, .operator (⟨9821, 0⟩, ⟨207528, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact207561RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact207561RawTermsValid :
    exact207561RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207561 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15082⟩⟩) exact207561RawTerms .large 207559 .exactZero (none)

def event207562 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8608⟩⟩) 0 ⟨5597⟩ 207398

def event207563 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8608⟩⟩) 1 ⟨7302⟩ 17106

def event207564 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8608⟩⟩) (.product (.predecessor 0 207562 .coefficient) (.predecessor 1 207563 .coefficient) (⟨false, false, none, none, none⟩))

def event207565 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8608⟩⟩, .operator (⟨207398, 0⟩, ⟨17106, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩)

def exact207566RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩]

theorem exact207566RawTermsValid :
    exact207566RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207566 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8608⟩⟩) exact207566RawTerms .large 207564 .exactZero (none)

def event207567 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15083⟩⟩) 0 ⟨8608⟩ 207566

def event207568 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15083⟩⟩) 1 ⟨15082⟩ 207561

def event207569 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15083⟩⟩) (.sum [.predecessor 0 207567 .coefficient, .predecessor 1 207568 .coefficient])

def exact207570RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207570RawTermsValid :
    exact207570RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207570 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15083⟩⟩) exact207570RawTerms .large 207569 .exactZero (none)

def event207571 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15084⟩⟩) 0 ⟨15083⟩ 207570

def event207572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15084⟩⟩) 1 ⟨128⟩ 17098

def event207573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15084⟩⟩) (.sum [.predecessor 0 207571 .coefficient, .predecessor 1 207572 .coefficient])

def event207574 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15084⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨128⟩⟩]⟩) [⟨.result 17098 .coefficient, false, none⟩])

def event207575 : Event := .survivorFold (1) 207574

def exact207576RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207576RawTermsValid :
    exact207576RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207576 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15084⟩⟩) exact207576RawTerms .large 207573 (.finite 26) (some (207574))

def event207577 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15085⟩⟩) 0 ⟨15084⟩ 207576

def event207578 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15085⟩⟩) 1 ⟨9566⟩ 17095

def event207579 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15085⟩⟩) (.product (.predecessor 0 207577 .coefficient) (.predecessor 1 207578 .coefficient) (⟨false, false, none, none, none⟩))

def event207580 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15085⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) [⟨.result 17091 .coefficient, false, none⟩])

def event207581 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15085⟩⟩) (.product (.result 207576 .summary) (.transfer 207580) (⟨false, false, none, none, none⟩))

def event207582 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15085⟩⟩, .operator (⟨207576, 1⟩, ⟨17095, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (-1)⟩)

def event207583 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨15085⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9565⟩⟩) ⟨7285⟩ 17065)

def event207584 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15085⟩⟩, .relation 207583 0, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (-1)⟩)

def event207585 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15085⟩⟩, .operator (⟨207576, 0⟩, ⟨17095, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩)

def exact207586RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (-1)⟩]

theorem exact207586RawTermsValid :
    exact207586RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207586 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15085⟩⟩) exact207586RawTerms .large 207579 (.finite 279172874240) (some (207581))

def event207587 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47841⟩⟩) 0 ⟨15085⟩ 207586

def event207588 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47841⟩⟩) 1 ⟨47840⟩ 207556

def event207589 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47841⟩⟩) (.sum [.predecessor 0 207587 .coefficient, .predecessor 1 207588 .coefficient])

def event207590 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47841⟩⟩, .operator (⟨207586, 1⟩, ⟨207556, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩)

def event207591 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47841⟩⟩) (.sum [.result 207586 .summary, .result 207556 .summary])

def exact207592RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact207592RawTermsValid :
    exact207592RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207592 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47841⟩⟩) exact207592RawTerms .large 207589 (.finite 279223992320) (some (207591))

def event207593 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49660⟩⟩) 0 ⟨47841⟩ 207592

def event207594 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49660⟩⟩) 1 ⟨49659⟩ 207523

def event207595 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49660⟩⟩) (.product (.predecessor 0 207593 .coefficient) (.predecessor 1 207594 .coefficient) (⟨false, false, none, none, none⟩))

def event207596 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49660⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨49659⟩⟩]⟩) [⟨.result 207523 .coefficient, false, none⟩])

def event207597 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49660⟩⟩) (.product (.result 207592 .summary) (.transfer 207596) (⟨false, false, none, none, none⟩))

def event207598 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49660⟩⟩, .operator (⟨207592, 1⟩, ⟨207523, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩]⟩, (-1)⟩)

def event207599 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49660⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49659⟩⟩) ⟨49149⟩ 207520)

def event207600 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49660⟩⟩, .relation 207599 0, ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨49149⟩⟩]⟩, (-1)⟩)

def event207601 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49660⟩⟩, .operator (⟨207592, 0⟩, ⟨207523, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩]⟩, (1)⟩)

def exact207602RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨49149⟩⟩]⟩, (-1)⟩]

theorem exact207602RawTermsValid :
    exact207602RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207602 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49660⟩⟩) exact207602RawTerms .large 207595 (.finite 2998144788182387916800) (some (207597))

def event207603 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48589⟩⟩) 0 ⟨47836⟩ 9829

def event207604 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48589⟩⟩) (.authority (.relationPreimageSource ⟨54⟩))

def exact207605RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48589⟩⟩]⟩, (1)⟩]

theorem exact207605RawTermsValid :
    exact207605RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207605 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48589⟩⟩) exact207605RawTerms (.finite 5647228698) 207604 .exactZero (none)

def event207606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48591⟩⟩) 0 ⟨48589⟩ 207605

def event207607 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48591⟩⟩) 1 ⟨2370⟩ 4

def event207608 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48591⟩⟩) (.scale (.predecessor 0 207606 .coefficient) (.value (.predecessor 1 207607 .coefficient)))

def exact207609RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48589⟩⟩]⟩, (1)⟩]

theorem exact207609RawTermsValid :
    exact207609RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207609 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48591⟩⟩) exact207609RawTerms (.finite 5647228698) 207608 .exactZero (none)

def event207610 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5598⟩⟩) 0 ⟨5597⟩ 207398

def event207611 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5598⟩⟩) 1 ⟨35⟩ 17158

def event207612 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5598⟩⟩) (.product (.predecessor 0 207610 .coefficient) (.predecessor 1 207611 .coefficient) (⟨false, false, none, none, none⟩))

def event207613 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨5598⟩⟩, .operator (⟨207398, 0⟩, ⟨17158, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩)

def exact207614RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact207614RawTermsValid :
    exact207614RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event207614 : Event := .resultExact (⟨.program ⟨257⟩, ⟨5598⟩⟩) exact207614RawTerms .large 207612 .exactZero (none)

def event207615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5599⟩⟩) 0 ⟨5598⟩ 207614

def eventLeaf12960 : Array AnnotatedEvent := #[
  { event := event207360
    frameStart := 0 },
  { event := event207361
    frameStart := 0 },
  { event := event207362
    frameStart := 0 },
  { event := event207363
    frameStart := 0 },
  { event := event207364
    frameStart := 0 },
  { event := event207365
    frameStart := 0 },
  { event := event207366
    frameStart := 0 },
  { event := event207367
    frameStart := 0 },
  { event := event207368
    frameStart := 0 },
  { event := event207369
    frameStart := 0 },
  { event := event207370
    frameStart := 0 },
  { event := event207371
    frameStart := 0 },
  { event := event207372
    frameStart := 0 },
  { event := event207373
    frameStart := 0 },
  { event := event207374
    frameStart := 0 },
  { event := event207375
    frameStart := 0 }
]

def eventLeaf12961 : Array AnnotatedEvent := #[
  { event := event207376
    frameStart := 0 },
  { event := event207377
    frameStart := 0 },
  { event := event207378
    frameStart := 0 },
  { event := event207379
    frameStart := 0 },
  { event := event207380
    frameStart := 0 },
  { event := event207381
    frameStart := 0 },
  { event := event207382
    frameStart := 0 },
  { event := event207383
    frameStart := 0 },
  { event := event207384
    frameStart := 0 },
  { event := event207385
    frameStart := 0 },
  { event := event207386
    frameStart := 0 },
  { event := event207387
    frameStart := 0 },
  { event := event207388
    frameStart := 0 },
  { event := event207389
    frameStart := 0 },
  { event := event207390
    frameStart := 0 },
  { event := event207391
    frameStart := 0 }
]

def eventLeaf12962 : Array AnnotatedEvent := #[
  { event := event207392
    frameStart := 0 },
  { event := event207393
    frameStart := 0 },
  { event := event207394
    frameStart := 0 },
  { event := event207395
    frameStart := 0 },
  { event := event207396
    frameStart := 0 },
  { event := event207397
    frameStart := 0 },
  { event := event207398
    frameStart := 0 },
  { event := event207399
    frameStart := 0 },
  { event := event207400
    frameStart := 0 },
  { event := event207401
    frameStart := 0 },
  { event := event207402
    frameStart := 0 },
  { event := event207403
    frameStart := 0 },
  { event := event207404
    frameStart := 0 },
  { event := event207405
    frameStart := 0 },
  { event := event207406
    frameStart := 0 },
  { event := event207407
    frameStart := 0 }
]

def eventLeaf12963 : Array AnnotatedEvent := #[
  { event := event207408
    frameStart := 0 },
  { event := event207409
    frameStart := 0 },
  { event := event207410
    frameStart := 0 },
  { event := event207411
    frameStart := 0 },
  { event := event207412
    frameStart := 0 },
  { event := event207413
    frameStart := 0 },
  { event := event207414
    frameStart := 0 },
  { event := event207415
    frameStart := 0 },
  { event := event207416
    frameStart := 0 },
  { event := event207417
    frameStart := 0 },
  { event := event207418
    frameStart := 0 },
  { event := event207419
    frameStart := 0 },
  { event := event207420
    frameStart := 0 },
  { event := event207421
    frameStart := 0 },
  { event := event207422
    frameStart := 0 },
  { event := event207423
    frameStart := 0 }
]

def eventLeaf12964 : Array AnnotatedEvent := #[
  { event := event207424
    frameStart := 0 },
  { event := event207425
    frameStart := 0 },
  { event := event207426
    frameStart := 0 },
  { event := event207427
    frameStart := 0 },
  { event := event207428
    frameStart := 0 },
  { event := event207429
    frameStart := 0 },
  { event := event207430
    frameStart := 0 },
  { event := event207431
    frameStart := 0 },
  { event := event207432
    frameStart := 0 },
  { event := event207433
    frameStart := 0 },
  { event := event207434
    frameStart := 0 },
  { event := event207435
    frameStart := 0 },
  { event := event207436
    frameStart := 0 },
  { event := event207437
    frameStart := 0 },
  { event := event207438
    frameStart := 0 },
  { event := event207439
    frameStart := 0 }
]

def eventLeaf12965 : Array AnnotatedEvent := #[
  { event := event207440
    frameStart := 0 },
  { event := event207441
    frameStart := 0 },
  { event := event207442
    frameStart := 0 },
  { event := event207443
    frameStart := 0 },
  { event := event207444
    frameStart := 0 },
  { event := event207445
    frameStart := 0 },
  { event := event207446
    frameStart := 0 },
  { event := event207447
    frameStart := 0 },
  { event := event207448
    frameStart := 0 },
  { event := event207449
    frameStart := 0 },
  { event := event207450
    frameStart := 0 },
  { event := event207451
    frameStart := 0 },
  { event := event207452
    frameStart := 0 },
  { event := event207453
    frameStart := 0 },
  { event := event207454
    frameStart := 0 },
  { event := event207455
    frameStart := 0 }
]

def eventLeaf12966 : Array AnnotatedEvent := #[
  { event := event207456
    frameStart := 0 },
  { event := event207457
    frameStart := 0 },
  { event := event207458
    frameStart := 0 },
  { event := event207459
    frameStart := 0 },
  { event := event207460
    frameStart := 0 },
  { event := event207461
    frameStart := 0 },
  { event := event207462
    frameStart := 0 },
  { event := event207463
    frameStart := 0 },
  { event := event207464
    frameStart := 0 },
  { event := event207465
    frameStart := 0 },
  { event := event207466
    frameStart := 0 },
  { event := event207467
    frameStart := 0 },
  { event := event207468
    frameStart := 0 },
  { event := event207469
    frameStart := 0 },
  { event := event207470
    frameStart := 0 },
  { event := event207471
    frameStart := 0 }
]

def eventLeaf12967 : Array AnnotatedEvent := #[
  { event := event207472
    frameStart := 0 },
  { event := event207473
    frameStart := 0 },
  { event := event207474
    frameStart := 0 },
  { event := event207475
    frameStart := 0 },
  { event := event207476
    frameStart := 0 },
  { event := event207477
    frameStart := 0 },
  { event := event207478
    frameStart := 0 },
  { event := event207479
    frameStart := 0 },
  { event := event207480
    frameStart := 0 },
  { event := event207481
    frameStart := 0 },
  { event := event207482
    frameStart := 0 },
  { event := event207483
    frameStart := 0 },
  { event := event207484
    frameStart := 0 },
  { event := event207485
    frameStart := 0 },
  { event := event207486
    frameStart := 0 },
  { event := event207487
    frameStart := 0 }
]

def eventLeaf12968 : Array AnnotatedEvent := #[
  { event := event207488
    frameStart := 0 },
  { event := event207489
    frameStart := 0 },
  { event := event207490
    frameStart := 0 },
  { event := event207491
    frameStart := 0 },
  { event := event207492
    frameStart := 0 },
  { event := event207493
    frameStart := 0 },
  { event := event207494
    frameStart := 0 },
  { event := event207495
    frameStart := 0 },
  { event := event207496
    frameStart := 0 },
  { event := event207497
    frameStart := 0 },
  { event := event207498
    frameStart := 0 },
  { event := event207499
    frameStart := 0 },
  { event := event207500
    frameStart := 0 },
  { event := event207501
    frameStart := 0 },
  { event := event207502
    frameStart := 0 },
  { event := event207503
    frameStart := 0 }
]

def eventLeaf12969 : Array AnnotatedEvent := #[
  { event := event207504
    frameStart := 0 },
  { event := event207505
    frameStart := 0 },
  { event := event207506
    frameStart := 0 },
  { event := event207507
    frameStart := 0 },
  { event := event207508
    frameStart := 0 },
  { event := event207509
    frameStart := 0 },
  { event := event207510
    frameStart := 0 },
  { event := event207511
    frameStart := 0 },
  { event := event207512
    frameStart := 0 },
  { event := event207513
    frameStart := 0 },
  { event := event207514
    frameStart := 0 },
  { event := event207515
    frameStart := 0 },
  { event := event207516
    frameStart := 0 },
  { event := event207517
    frameStart := 0 },
  { event := event207518
    frameStart := 0 },
  { event := event207519
    frameStart := 0 }
]

def eventLeaf12970 : Array AnnotatedEvent := #[
  { event := event207520
    frameStart := 0 },
  { event := event207521
    frameStart := 0 },
  { event := event207522
    frameStart := 0 },
  { event := event207523
    frameStart := 0 },
  { event := event207524
    frameStart := 0 },
  { event := event207525
    frameStart := 0 },
  { event := event207526
    frameStart := 0 },
  { event := event207527
    frameStart := 0 },
  { event := event207528
    frameStart := 0 },
  { event := event207529
    frameStart := 0 },
  { event := event207530
    frameStart := 0 },
  { event := event207531
    frameStart := 0 },
  { event := event207532
    frameStart := 0 },
  { event := event207533
    frameStart := 0 },
  { event := event207534
    frameStart := 0 },
  { event := event207535
    frameStart := 0 }
]

def eventLeaf12971 : Array AnnotatedEvent := #[
  { event := event207536
    frameStart := 0 },
  { event := event207537
    frameStart := 0 },
  { event := event207538
    frameStart := 0 },
  { event := event207539
    frameStart := 0 },
  { event := event207540
    frameStart := 0 },
  { event := event207541
    frameStart := 0 },
  { event := event207542
    frameStart := 0 },
  { event := event207543
    frameStart := 0 },
  { event := event207544
    frameStart := 0 },
  { event := event207545
    frameStart := 0 },
  { event := event207546
    frameStart := 0 },
  { event := event207547
    frameStart := 0 },
  { event := event207548
    frameStart := 0 },
  { event := event207549
    frameStart := 0 },
  { event := event207550
    frameStart := 0 },
  { event := event207551
    frameStart := 0 }
]

def eventLeaf12972 : Array AnnotatedEvent := #[
  { event := event207552
    frameStart := 0 },
  { event := event207553
    frameStart := 0 },
  { event := event207554
    frameStart := 0 },
  { event := event207555
    frameStart := 0 },
  { event := event207556
    frameStart := 0 },
  { event := event207557
    frameStart := 0 },
  { event := event207558
    frameStart := 0 },
  { event := event207559
    frameStart := 0 },
  { event := event207560
    frameStart := 0 },
  { event := event207561
    frameStart := 0 },
  { event := event207562
    frameStart := 0 },
  { event := event207563
    frameStart := 0 },
  { event := event207564
    frameStart := 0 },
  { event := event207565
    frameStart := 0 },
  { event := event207566
    frameStart := 0 },
  { event := event207567
    frameStart := 0 }
]

def eventLeaf12973 : Array AnnotatedEvent := #[
  { event := event207568
    frameStart := 0 },
  { event := event207569
    frameStart := 0 },
  { event := event207570
    frameStart := 0 },
  { event := event207571
    frameStart := 0 },
  { event := event207572
    frameStart := 0 },
  { event := event207573
    frameStart := 0 },
  { event := event207574
    frameStart := 0 },
  { event := event207575
    frameStart := 0 },
  { event := event207576
    frameStart := 0 },
  { event := event207577
    frameStart := 0 },
  { event := event207578
    frameStart := 0 },
  { event := event207579
    frameStart := 0 },
  { event := event207580
    frameStart := 0 },
  { event := event207581
    frameStart := 0 },
  { event := event207582
    frameStart := 0 },
  { event := event207583
    frameStart := 0 }
]

def eventLeaf12974 : Array AnnotatedEvent := #[
  { event := event207584
    frameStart := 0 },
  { event := event207585
    frameStart := 0 },
  { event := event207586
    frameStart := 0 },
  { event := event207587
    frameStart := 0 },
  { event := event207588
    frameStart := 0 },
  { event := event207589
    frameStart := 0 },
  { event := event207590
    frameStart := 0 },
  { event := event207591
    frameStart := 0 },
  { event := event207592
    frameStart := 0 },
  { event := event207593
    frameStart := 0 },
  { event := event207594
    frameStart := 0 },
  { event := event207595
    frameStart := 0 },
  { event := event207596
    frameStart := 0 },
  { event := event207597
    frameStart := 0 },
  { event := event207598
    frameStart := 0 },
  { event := event207599
    frameStart := 0 }
]

def eventLeaf12975 : Array AnnotatedEvent := #[
  { event := event207600
    frameStart := 0 },
  { event := event207601
    frameStart := 0 },
  { event := event207602
    frameStart := 0 },
  { event := event207603
    frameStart := 0 },
  { event := event207604
    frameStart := 0 },
  { event := event207605
    frameStart := 0 },
  { event := event207606
    frameStart := 0 },
  { event := event207607
    frameStart := 0 },
  { event := event207608
    frameStart := 0 },
  { event := event207609
    frameStart := 0 },
  { event := event207610
    frameStart := 0 },
  { event := event207611
    frameStart := 0 },
  { event := event207612
    frameStart := 0 },
  { event := event207613
    frameStart := 0 },
  { event := event207614
    frameStart := 0 },
  { event := event207615
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events810
