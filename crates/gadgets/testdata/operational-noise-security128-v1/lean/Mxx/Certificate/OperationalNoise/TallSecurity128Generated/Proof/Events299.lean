import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events299

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event76544 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event76545 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 76544

def event76546 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 76536

def event76547 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 76545 .coefficient, .predecessor 1 76546 .coefficient])

def event76548 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event76549 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 76548

def event76550 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 76534

def event76551 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 76550 .coefficient))

def event76552 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event76553 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45298⟩⟩) 0 ⟨10325⟩ 76552

def event76554 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45298⟩⟩) (.authority (.programFamilyFact))

def exact76555RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45298⟩⟩], []⟩, (1)⟩]

theorem exact76555RawTermsValid :
    exact76555RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76555 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45298⟩⟩) exact76555RawTerms (.finite 58) 76554 .exactZero (none)

def event76556 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14871⟩⟩) 0 ⟨10325⟩ 76552

def event76557 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14871⟩⟩) (.authority (.programFamilyFact))

def exact76558RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩], []⟩, (1)⟩]

theorem exact76558RawTermsValid :
    exact76558RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76558 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14871⟩⟩) exact76558RawTerms (.finite 58) 76557 .exactZero (none)

def event76559 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45299⟩⟩) 0 ⟨14871⟩ 76558

def event76560 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45299⟩⟩) 1 ⟨45298⟩ 76555

def event76561 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45299⟩⟩) (.product (.predecessor 0 76559 .coefficient) (.predecessor 1 76560 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event76562 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45299⟩⟩, .operator (⟨76558, 0⟩, ⟨76555, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], []⟩, (1)⟩)

def exact76563RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], []⟩, (1)⟩]

theorem exact76563RawTermsValid :
    exact76563RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76563 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45299⟩⟩) exact76563RawTerms (.finite 3364) 76561 .exactZero (none)

def event76564 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45300⟩⟩) 0 ⟨45299⟩ 76563

def event76565 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45300⟩⟩) (.identity (.predecessor 0 76564 .coefficient))

def event76566 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45300⟩⟩) (.finite 3364)

def event76567 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46504⟩⟩) 0 ⟨45300⟩ 76566

def event76568 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46504⟩⟩) (.authority (.programFamilyFact))

def event76569 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46504⟩⟩) (.finite 3720)

def event76570 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event76571 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46505⟩⟩) 0 ⟨7177⟩ 76570

def event76572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46505⟩⟩) 1 ⟨46504⟩ 76569

def event76573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46505⟩⟩) (.authority (.operator))

def exact76574RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46505⟩⟩]⟩, (1)⟩]

theorem exact76574RawTermsValid :
    exact76574RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76574 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46505⟩⟩) exact76574RawTerms .large 76573 .exactZero (none)

def event76575 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47045⟩⟩) 0 ⟨46505⟩ 76574

def event76576 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47045⟩⟩) (.authority (.operator))

def exact76577RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47045⟩⟩]⟩, (1)⟩]

theorem exact76577RawTermsValid :
    exact76577RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76577 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47045⟩⟩) exact76577RawTerms (.finite 8192) 76576 .exactZero (none)

def event76578 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event76579 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event76580 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46770⟩⟩) 0 ⟨45300⟩ 76566

def event76581 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46770⟩⟩) 1 ⟨136⟩ 76579

def event76582 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46770⟩⟩) (.sum [.predecessor 0 76580 .coefficient, .predecessor 1 76581 .coefficient])

def event76583 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46770⟩⟩) (.finite 3364)

def event76584 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46771⟩⟩) 0 ⟨46770⟩ 76583

def event76585 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46771⟩⟩) (.identity (.predecessor 0 76584 .coefficient))

def exact76586RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], []⟩, (1)⟩]

theorem exact76586RawTermsValid :
    exact76586RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76586 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46771⟩⟩) exact76586RawTerms (.finite 3364) 76585 .exactZero (none)

def event76587 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact76588RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact76588RawTermsValid :
    exact76588RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76588 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact76588RawTerms .large 76587 .exactZero (none)

def event76589 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46772⟩⟩) 0 ⟨6908⟩ 76588

def event76590 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46772⟩⟩) 1 ⟨46771⟩ 76586

def event76591 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46772⟩⟩) (.product (.predecessor 0 76589 .coefficient) (.predecessor 1 76590 .coefficient) (⟨false, false, none, none, none⟩))

def event76592 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46772⟩⟩, .operator (⟨76588, 0⟩, ⟨76586, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact76593RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact76593RawTermsValid :
    exact76593RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76593 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46772⟩⟩) exact76593RawTerms .large 76591 .exactZero (none)

def event76594 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event76595 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event76596 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 76570

def event76597 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact76598RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact76598RawTermsValid :
    exact76598RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76598 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact76598RawTerms .large 76597 .exactZero (none)

def event76599 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7284⟩⟩) 0 ⟨7178⟩ 76598

def event76600 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7284⟩⟩) (.identity (.predecessor 0 76599 .coefficient))

def exact76601RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩]

theorem exact76601RawTermsValid :
    exact76601RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76601 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7284⟩⟩) exact76601RawTerms .large 76600 .exactZero (none)

def event76602 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9562⟩⟩) 0 ⟨7284⟩ 76601

def event76603 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9562⟩⟩) (.authority (.operator))

def exact76604RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩]

theorem exact76604RawTermsValid :
    exact76604RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76604 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9562⟩⟩) exact76604RawTerms (.finite 8192) 76603 .exactZero (none)

def event76605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9563⟩⟩) 0 ⟨9562⟩ 76604

def event76606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9563⟩⟩) 1 ⟨2370⟩ 76595

def event76607 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9563⟩⟩) (.scale (.predecessor 0 76605 .coefficient) (.value (.predecessor 1 76606 .coefficient)))

def exact76608RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩]

theorem exact76608RawTermsValid :
    exact76608RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76608 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9563⟩⟩) exact76608RawTerms (.finite 8192) 76607 .exactZero (none)

def event76609 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7301⟩⟩) 0 ⟨7178⟩ 76598

def event76610 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7301⟩⟩) (.identity (.predecessor 0 76609 .coefficient))

def exact76611RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩, (1)⟩]

theorem exact76611RawTermsValid :
    exact76611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76611 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7301⟩⟩) exact76611RawTerms .large 76610 .exactZero (none)

def event76612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9564⟩⟩) 0 ⟨7301⟩ 76611

def event76613 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9564⟩⟩) 1 ⟨9563⟩ 76608

def event76614 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9564⟩⟩) (.product (.predecessor 0 76612 .coefficient) (.predecessor 1 76613 .coefficient) (⟨false, false, none, none, none⟩))

def event76615 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9564⟩⟩, .operator (⟨76611, 0⟩, ⟨76608, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩)

def exact76616RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩]

theorem exact76616RawTermsValid :
    exact76616RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76616 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9564⟩⟩) exact76616RawTerms .large 76614 .exactZero (none)

def event76617 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46773⟩⟩) 0 ⟨9564⟩ 76616

def event76618 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46773⟩⟩) 1 ⟨46772⟩ 76593

def event76619 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46773⟩⟩) (.sum [.predecessor 0 76617 .coefficient, .predecessor 1 76618 .coefficient])

def exact76620RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact76620RawTermsValid :
    exact76620RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76620 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46773⟩⟩) exact76620RawTerms .large 76619 .exactZero (none)

def event76621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47048⟩⟩) 0 ⟨46773⟩ 76620

def event76622 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47048⟩⟩) 1 ⟨47045⟩ 76577

def event76623 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47048⟩⟩) (.product (.predecessor 0 76621 .coefficient) (.predecessor 1 76622 .coefficient) (⟨false, false, none, none, none⟩))

def event76624 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47048⟩⟩, .operator (⟨76620, 0⟩, ⟨76577, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47045⟩⟩]⟩, (1)⟩)

def event76625 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47048⟩⟩, .operator (⟨76620, 1⟩, ⟨76577, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47045⟩⟩]⟩, (-1)⟩)

def event76626 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47048⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47045⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47045⟩⟩) ⟨46505⟩ 76574)

def event76627 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47048⟩⟩, .relation 76626 0, ⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨46505⟩⟩]⟩, (-1)⟩)

def exact76628RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47045⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨46505⟩⟩]⟩, (-1)⟩]

theorem exact76628RawTermsValid :
    exact76628RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76628 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47048⟩⟩) exact76628RawTerms .large 76623 .exactZero (none)

def event76629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45516⟩⟩) 0 ⟨45300⟩ 76566

def event76630 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45516⟩⟩) (.authority (.programFamilyFact))

def exact76631RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], []⟩, (1)⟩]

theorem exact76631RawTermsValid :
    exact76631RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76631 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45516⟩⟩) exact76631RawTerms (.finite 58) 76630 .exactZero (none)

def event76632 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45518⟩⟩) 0 ⟨6908⟩ 76588

def event76633 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45518⟩⟩) 1 ⟨45516⟩ 76631

def event76634 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45518⟩⟩) (.product (.predecessor 0 76632 .coefficient) (.predecessor 1 76633 .coefficient) (⟨false, true, none, none, some 1⟩))

def event76635 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45518⟩⟩, .operator (⟨76588, 0⟩, ⟨76631, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact76636RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact76636RawTermsValid :
    exact76636RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76636 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45518⟩⟩) exact76636RawTerms .large 76634 .exactZero (none)

def event76637 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7195⟩⟩) 0 ⟨7177⟩ 76570

def event76638 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7195⟩⟩) (.authority (.operator))

def exact76639RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩]

theorem exact76639RawTermsValid :
    exact76639RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76639 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7195⟩⟩) exact76639RawTerms .large 76638 .exactZero (none)

def event76640 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45519⟩⟩) 0 ⟨7195⟩ 76639

def event76641 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45519⟩⟩) 1 ⟨45518⟩ 76636

def event76642 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45519⟩⟩) (.sum [.predecessor 0 76640 .coefficient, .predecessor 1 76641 .coefficient])

def exact76643RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact76643RawTermsValid :
    exact76643RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76643 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45519⟩⟩) exact76643RawTerms .large 76642 .exactZero (none)

def event76644 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47049⟩⟩) 0 ⟨45519⟩ 76643

def event76645 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47049⟩⟩) 1 ⟨47048⟩ 76628

def event76646 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47049⟩⟩) (.sum [.predecessor 0 76644 .coefficient, .predecessor 1 76645 .coefficient])

def exact76647RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47045⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨46505⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact76647RawTermsValid :
    exact76647RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76647 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47049⟩⟩) exact76647RawTerms .large 76646 .exactZero (none)

def event76648 : Event := .preFoldPolynomial 76647 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47045⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨46505⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact76649RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47045⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨46505⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event76649 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨47049⟩⟩) 76648 exact76649RawTerms .large 76646 .exactZero (none)

def event76650 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨45300⟩⟩) ⟨⟨74⟩, ⟨53⟩, ⟨135⟩⟩ ⟨76484, 76650⟩

def event76651 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨45972⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45969⟩⟩]⟩) (1) 0 2 (.universal 76650 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45969⟩⟩]⟩) (none) 76649)

def event76652 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45972⟩⟩, .relation 76651 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩)

def event76653 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45972⟩⟩, .relation 76651 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47045⟩⟩]⟩, (-1)⟩)

def event76654 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45972⟩⟩, .relation 76651 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨46505⟩⟩]⟩, (1)⟩)

def event76655 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45972⟩⟩, .relation 76651 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact76656RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47045⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨46505⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact76656RawTermsValid :
    exact76656RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76656 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45972⟩⟩) exact76656RawTerms .large 76480 (.finite 202072841853861888) (some (76482))

def event76657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47047⟩⟩) 0 ⟨45972⟩ 76656

def event76658 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47047⟩⟩) 1 ⟨47046⟩ 76470

def event76659 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47047⟩⟩) (.sum [.predecessor 0 76657 .coefficient, .predecessor 1 76658 .coefficient])

def event76660 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47047⟩⟩, .operator (⟨76656, 2⟩, ⟨76470, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], [⟨.program ⟨257⟩, ⟨46505⟩⟩]⟩, (-1)⟩)

def event76661 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47047⟩⟩, .operator (⟨76656, 1⟩, ⟨76470, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47045⟩⟩]⟩, (1)⟩)

def event76662 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47047⟩⟩) (.sum [.result 76656 .summary, .result 76470 .summary])

def exact76663RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact76663RawTermsValid :
    exact76663RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76663 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47047⟩⟩) exact76663RawTerms .large 76659 (.finite 2998328565150755586048) (some (76662))

def event76664 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47501⟩⟩) 0 ⟨47047⟩ 76663

def event76665 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47501⟩⟩) 1 ⟨47499⟩ 76386

def event76666 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47501⟩⟩) (.product (.predecessor 0 76664 .coefficient) (.predecessor 1 76665 .coefficient) (⟨false, false, none, none, none⟩))

def event76667 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47501⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨47499⟩⟩]⟩) [⟨.result 76386 .coefficient, false, none⟩])

def event76668 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47501⟩⟩) (.product (.result 76663 .summary) (.transfer 76667) (⟨false, false, none, none, none⟩))

def event76669 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47501⟩⟩, .operator (⟨76663, 0⟩, ⟨76386, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47499⟩⟩]⟩, (1)⟩)

def event76670 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47501⟩⟩, .operator (⟨76663, 1⟩, ⟨76386, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47499⟩⟩]⟩, (-1)⟩)

def event76671 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨47501⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47499⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47499⟩⟩) ⟨46675⟩ 76383)

def event76672 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47501⟩⟩, .relation 76671 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨46675⟩⟩]⟩, (-1)⟩)

def exact76673RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47499⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45516⟩⟩], [⟨.program ⟨257⟩, ⟨46675⟩⟩]⟩, (-1)⟩]

theorem exact76673RawTermsValid :
    exact76673RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76673 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47501⟩⟩) exact76673RawTerms .large 76666 (.finite 32194307824962751379413684715520) (some (76668))

def event76674 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46336⟩⟩) 0 ⟨45517⟩ 3126

def event76675 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46336⟩⟩) (.authority (.relationPreimageSource ⟨92⟩))

def exact76676RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46336⟩⟩]⟩, (1)⟩]

theorem exact76676RawTermsValid :
    exact76676RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76676 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46336⟩⟩) exact76676RawTerms (.finite 5647228698) 76675 .exactZero (none)

def event76677 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46338⟩⟩) 0 ⟨46336⟩ 76676

def event76678 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46338⟩⟩) 1 ⟨2370⟩ 4

def event76679 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46338⟩⟩) (.scale (.predecessor 0 76677 .coefficient) (.value (.predecessor 1 76678 .coefficient)))

def exact76680RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46336⟩⟩]⟩, (1)⟩]

theorem exact76680RawTermsValid :
    exact76680RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76680 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46338⟩⟩) exact76680RawTerms (.finite 5647228698) 76679 .exactZero (none)

def event76681 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46339⟩⟩) 0 ⟨10368⟩ 75995

def event76682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46339⟩⟩) 1 ⟨46338⟩ 76680

def event76683 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46339⟩⟩) (.product (.predecessor 0 76681 .coefficient) (.predecessor 1 76682 .coefficient) (⟨false, false, none, none, none⟩))

def event76684 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46339⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨46336⟩⟩]⟩) [⟨.result 76676 .coefficient, false, none⟩])

def event76685 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46339⟩⟩) (.product (.result 75995 .summary) (.transfer 76684) (⟨false, false, none, none, none⟩))

def event76686 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46339⟩⟩, .operator (⟨75995, 0⟩, ⟨76680, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46336⟩⟩]⟩, (1)⟩)

def event76687 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨46337⟩⟩)

def event76688 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event76689 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event76690 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event76691 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event76692 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event76693 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event76694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event76695 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event76696 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 76695

def event76697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 76693

def event76698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 76696 .coefficient) (.value (.predecessor 1 76697 .coefficient)))

def event76699 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event76700 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 76699

def event76701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 76691

def event76702 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 76700 .coefficient, .predecessor 1 76701 .coefficient])

def event76703 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event76704 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 76703

def event76705 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 76689

def event76706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 76705 .coefficient))

def event76707 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event76708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45298⟩⟩) 0 ⟨10325⟩ 76707

def event76709 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45298⟩⟩) (.authority (.programFamilyFact))

def exact76710RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45298⟩⟩], []⟩, (1)⟩]

theorem exact76710RawTermsValid :
    exact76710RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76710 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45298⟩⟩) exact76710RawTerms (.finite 58) 76709 .exactZero (none)

def event76711 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14871⟩⟩) 0 ⟨10325⟩ 76707

def event76712 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14871⟩⟩) (.authority (.programFamilyFact))

def exact76713RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩], []⟩, (1)⟩]

theorem exact76713RawTermsValid :
    exact76713RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76713 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14871⟩⟩) exact76713RawTerms (.finite 58) 76712 .exactZero (none)

def event76714 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45299⟩⟩) 0 ⟨14871⟩ 76713

def event76715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45299⟩⟩) 1 ⟨45298⟩ 76710

def event76716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45299⟩⟩) (.product (.predecessor 0 76714 .coefficient) (.predecessor 1 76715 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event76717 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45299⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], []⟩) [⟨.result 76713 .coefficient, true, some 1⟩, ⟨.result 76710 .coefficient, true, some 1⟩])

def event76718 : Event := .survivorFold (1) 76717

def exact76719RawTerms : List Term := []

theorem exact76719RawTermsValid :
    exact76719RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76719 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45299⟩⟩) exact76719RawTerms (.finite 3364) 76716 (.finite 3364) (some (76717))

def event76720 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45300⟩⟩) 0 ⟨45299⟩ 76719

def event76721 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45300⟩⟩) (.identity (.predecessor 0 76720 .coefficient))

def event76722 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45300⟩⟩) (.finite 3364)

def event76723 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45516⟩⟩) 0 ⟨45300⟩ 76722

def event76724 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45516⟩⟩) (.authority (.programFamilyFact))

def exact76725RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], []⟩, (1)⟩]

theorem exact76725RawTermsValid :
    exact76725RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76725 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45516⟩⟩) exact76725RawTerms (.finite 58) 76724 .exactZero (none)

def event76726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45517⟩⟩) 0 ⟨45516⟩ 76725

def event76727 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45517⟩⟩) (.identity (.predecessor 0 76726 .coefficient))

def event76728 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45517⟩⟩) (.finite 58)

def event76729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46336⟩⟩) 0 ⟨45517⟩ 76728

def event76730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46336⟩⟩) (.authority (.relationPreimageSource ⟨92⟩))

def exact76731RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46336⟩⟩]⟩, (1)⟩]

theorem exact76731RawTermsValid :
    exact76731RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76731 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46336⟩⟩) exact76731RawTerms (.finite 5647228698) 76730 .exactZero (none)

def event76732 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact76733RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact76733RawTermsValid :
    exact76733RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76733 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact76733RawTerms .large 76732 .exactZero (none)

def event76734 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46337⟩⟩) 0 ⟨35⟩ 76733

def event76735 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46337⟩⟩) 1 ⟨46336⟩ 76731

def event76736 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46337⟩⟩) (.product (.predecessor 0 76734 .coefficient) (.predecessor 1 76735 .coefficient) (⟨false, false, none, none, none⟩))

def event76737 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨46337⟩⟩, .operator (⟨76733, 0⟩, ⟨76731, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46336⟩⟩]⟩, (1)⟩)

def exact76738RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46336⟩⟩]⟩, (1)⟩]

theorem exact76738RawTermsValid :
    exact76738RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76738 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46337⟩⟩) exact76738RawTerms .large 76736 .exactZero (none)

def event76739 : Event := .preFoldPolynomial 76738 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46336⟩⟩]⟩, (1)⟩] .exactZero none

def exact76740RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46336⟩⟩]⟩, (1)⟩]

def event76740 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨46337⟩⟩) 76739 exact76740RawTerms .large 76736 .exactZero (none)

def event76741 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨47503⟩⟩)

def event76742 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event76743 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event76744 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event76745 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event76746 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event76747 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event76748 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event76749 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event76750 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 76749

def event76751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 76747

def event76752 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 76750 .coefficient) (.value (.predecessor 1 76751 .coefficient)))

def event76753 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event76754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 76753

def event76755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 76745

def event76756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 76754 .coefficient, .predecessor 1 76755 .coefficient])

def event76757 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event76758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 76757

def event76759 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 76743

def event76760 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 76759 .coefficient))

def event76761 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event76762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45298⟩⟩) 0 ⟨10325⟩ 76761

def event76763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45298⟩⟩) (.authority (.programFamilyFact))

def exact76764RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45298⟩⟩], []⟩, (1)⟩]

theorem exact76764RawTermsValid :
    exact76764RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76764 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45298⟩⟩) exact76764RawTerms (.finite 58) 76763 .exactZero (none)

def event76765 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14871⟩⟩) 0 ⟨10325⟩ 76761

def event76766 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14871⟩⟩) (.authority (.programFamilyFact))

def exact76767RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩], []⟩, (1)⟩]

theorem exact76767RawTermsValid :
    exact76767RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76767 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14871⟩⟩) exact76767RawTerms (.finite 58) 76766 .exactZero (none)

def event76768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45299⟩⟩) 0 ⟨14871⟩ 76767

def event76769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45299⟩⟩) 1 ⟨45298⟩ 76764

def event76770 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45299⟩⟩) (.product (.predecessor 0 76768 .coefficient) (.predecessor 1 76769 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event76771 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45299⟩⟩, .operator (⟨76767, 0⟩, ⟨76764, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], []⟩, (1)⟩)

def exact76772RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14871⟩⟩, ⟨.program ⟨257⟩, ⟨45298⟩⟩], []⟩, (1)⟩]

theorem exact76772RawTermsValid :
    exact76772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76772 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45299⟩⟩) exact76772RawTerms (.finite 3364) 76770 .exactZero (none)

def event76773 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45300⟩⟩) 0 ⟨45299⟩ 76772

def event76774 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45300⟩⟩) (.identity (.predecessor 0 76773 .coefficient))

def event76775 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45300⟩⟩) (.finite 3364)

def event76776 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45516⟩⟩) 0 ⟨45300⟩ 76775

def event76777 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45516⟩⟩) (.authority (.programFamilyFact))

def exact76778RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45516⟩⟩], []⟩, (1)⟩]

theorem exact76778RawTermsValid :
    exact76778RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76778 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45516⟩⟩) exact76778RawTerms (.finite 58) 76777 .exactZero (none)

def event76779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45517⟩⟩) 0 ⟨45516⟩ 76778

def event76780 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45517⟩⟩) (.identity (.predecessor 0 76779 .coefficient))

def event76781 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45517⟩⟩) (.finite 58)

def event76782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46673⟩⟩) 0 ⟨45517⟩ 76781

def event76783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46673⟩⟩) (.authority (.programFamilyFact))

def event76784 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46673⟩⟩) (.finite 3720)

def event76785 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event76786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46675⟩⟩) 0 ⟨7177⟩ 76785

def event76787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46675⟩⟩) 1 ⟨46673⟩ 76784

def event76788 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46675⟩⟩) (.authority (.operator))

def exact76789RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46675⟩⟩]⟩, (1)⟩]

theorem exact76789RawTermsValid :
    exact76789RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76789 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46675⟩⟩) exact76789RawTerms .large 76788 .exactZero (none)

def event76790 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47499⟩⟩) 0 ⟨46675⟩ 76789

def event76791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47499⟩⟩) (.authority (.operator))

def exact76792RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47499⟩⟩]⟩, (1)⟩]

theorem exact76792RawTermsValid :
    exact76792RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event76792 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47499⟩⟩) exact76792RawTerms (.finite 8192) 76791 .exactZero (none)

def event76793 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event76794 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event76795 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46850⟩⟩) 0 ⟨45517⟩ 76781

def event76796 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46850⟩⟩) 1 ⟨136⟩ 76794

def event76797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46850⟩⟩) (.sum [.predecessor 0 76795 .coefficient, .predecessor 1 76796 .coefficient])

def event76798 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46850⟩⟩) (.finite 58)

def event76799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46851⟩⟩) 0 ⟨46850⟩ 76798

def eventLeaf4784 : Array AnnotatedEvent := #[
  { event := event76544
    frameStart := 76532 },
  { event := event76545
    frameStart := 76532 },
  { event := event76546
    frameStart := 76532 },
  { event := event76547
    frameStart := 76532 },
  { event := event76548
    frameStart := 76532 },
  { event := event76549
    frameStart := 76532 },
  { event := event76550
    frameStart := 76532 },
  { event := event76551
    frameStart := 76532 },
  { event := event76552
    frameStart := 76532 },
  { event := event76553
    frameStart := 76532 },
  { event := event76554
    frameStart := 76532 },
  { event := event76555
    frameStart := 76532 },
  { event := event76556
    frameStart := 76532 },
  { event := event76557
    frameStart := 76532 },
  { event := event76558
    frameStart := 76532 },
  { event := event76559
    frameStart := 76532 }
]

def eventLeaf4785 : Array AnnotatedEvent := #[
  { event := event76560
    frameStart := 76532 },
  { event := event76561
    frameStart := 76532 },
  { event := event76562
    frameStart := 76532 },
  { event := event76563
    frameStart := 76532 },
  { event := event76564
    frameStart := 76532 },
  { event := event76565
    frameStart := 76532 },
  { event := event76566
    frameStart := 76532 },
  { event := event76567
    frameStart := 76532 },
  { event := event76568
    frameStart := 76532 },
  { event := event76569
    frameStart := 76532 },
  { event := event76570
    frameStart := 76532 },
  { event := event76571
    frameStart := 76532 },
  { event := event76572
    frameStart := 76532 },
  { event := event76573
    frameStart := 76532 },
  { event := event76574
    frameStart := 76532 },
  { event := event76575
    frameStart := 76532 }
]

def eventLeaf4786 : Array AnnotatedEvent := #[
  { event := event76576
    frameStart := 76532 },
  { event := event76577
    frameStart := 76532 },
  { event := event76578
    frameStart := 76532 },
  { event := event76579
    frameStart := 76532 },
  { event := event76580
    frameStart := 76532 },
  { event := event76581
    frameStart := 76532 },
  { event := event76582
    frameStart := 76532 },
  { event := event76583
    frameStart := 76532 },
  { event := event76584
    frameStart := 76532 },
  { event := event76585
    frameStart := 76532 },
  { event := event76586
    frameStart := 76532 },
  { event := event76587
    frameStart := 76532 },
  { event := event76588
    frameStart := 76532 },
  { event := event76589
    frameStart := 76532 },
  { event := event76590
    frameStart := 76532 },
  { event := event76591
    frameStart := 76532 }
]

def eventLeaf4787 : Array AnnotatedEvent := #[
  { event := event76592
    frameStart := 76532 },
  { event := event76593
    frameStart := 76532 },
  { event := event76594
    frameStart := 76532 },
  { event := event76595
    frameStart := 76532 },
  { event := event76596
    frameStart := 76532 },
  { event := event76597
    frameStart := 76532 },
  { event := event76598
    frameStart := 76532 },
  { event := event76599
    frameStart := 76532 },
  { event := event76600
    frameStart := 76532 },
  { event := event76601
    frameStart := 76532 },
  { event := event76602
    frameStart := 76532 },
  { event := event76603
    frameStart := 76532 },
  { event := event76604
    frameStart := 76532 },
  { event := event76605
    frameStart := 76532 },
  { event := event76606
    frameStart := 76532 },
  { event := event76607
    frameStart := 76532 }
]

def eventLeaf4788 : Array AnnotatedEvent := #[
  { event := event76608
    frameStart := 76532 },
  { event := event76609
    frameStart := 76532 },
  { event := event76610
    frameStart := 76532 },
  { event := event76611
    frameStart := 76532 },
  { event := event76612
    frameStart := 76532 },
  { event := event76613
    frameStart := 76532 },
  { event := event76614
    frameStart := 76532 },
  { event := event76615
    frameStart := 76532 },
  { event := event76616
    frameStart := 76532 },
  { event := event76617
    frameStart := 76532 },
  { event := event76618
    frameStart := 76532 },
  { event := event76619
    frameStart := 76532 },
  { event := event76620
    frameStart := 76532 },
  { event := event76621
    frameStart := 76532 },
  { event := event76622
    frameStart := 76532 },
  { event := event76623
    frameStart := 76532 }
]

def eventLeaf4789 : Array AnnotatedEvent := #[
  { event := event76624
    frameStart := 76532 },
  { event := event76625
    frameStart := 76532 },
  { event := event76626
    frameStart := 76532 },
  { event := event76627
    frameStart := 76532 },
  { event := event76628
    frameStart := 76532 },
  { event := event76629
    frameStart := 76532 },
  { event := event76630
    frameStart := 76532 },
  { event := event76631
    frameStart := 76532 },
  { event := event76632
    frameStart := 76532 },
  { event := event76633
    frameStart := 76532 },
  { event := event76634
    frameStart := 76532 },
  { event := event76635
    frameStart := 76532 },
  { event := event76636
    frameStart := 76532 },
  { event := event76637
    frameStart := 76532 },
  { event := event76638
    frameStart := 76532 },
  { event := event76639
    frameStart := 76532 }
]

def eventLeaf4790 : Array AnnotatedEvent := #[
  { event := event76640
    frameStart := 76532 },
  { event := event76641
    frameStart := 76532 },
  { event := event76642
    frameStart := 76532 },
  { event := event76643
    frameStart := 76532 },
  { event := event76644
    frameStart := 76532 },
  { event := event76645
    frameStart := 76532 },
  { event := event76646
    frameStart := 76532 },
  { event := event76647
    frameStart := 76532 },
  { event := event76648
    frameStart := 76532 },
  { event := event76649
    frameStart := 76532 },
  { event := event76650
    frameStart := 0 },
  { event := event76651
    frameStart := 0 },
  { event := event76652
    frameStart := 0 },
  { event := event76653
    frameStart := 0 },
  { event := event76654
    frameStart := 0 },
  { event := event76655
    frameStart := 0 }
]

def eventLeaf4791 : Array AnnotatedEvent := #[
  { event := event76656
    frameStart := 0 },
  { event := event76657
    frameStart := 0 },
  { event := event76658
    frameStart := 0 },
  { event := event76659
    frameStart := 0 },
  { event := event76660
    frameStart := 0 },
  { event := event76661
    frameStart := 0 },
  { event := event76662
    frameStart := 0 },
  { event := event76663
    frameStart := 0 },
  { event := event76664
    frameStart := 0 },
  { event := event76665
    frameStart := 0 },
  { event := event76666
    frameStart := 0 },
  { event := event76667
    frameStart := 0 },
  { event := event76668
    frameStart := 0 },
  { event := event76669
    frameStart := 0 },
  { event := event76670
    frameStart := 0 },
  { event := event76671
    frameStart := 0 }
]

def eventLeaf4792 : Array AnnotatedEvent := #[
  { event := event76672
    frameStart := 0 },
  { event := event76673
    frameStart := 0 },
  { event := event76674
    frameStart := 0 },
  { event := event76675
    frameStart := 0 },
  { event := event76676
    frameStart := 0 },
  { event := event76677
    frameStart := 0 },
  { event := event76678
    frameStart := 0 },
  { event := event76679
    frameStart := 0 },
  { event := event76680
    frameStart := 0 },
  { event := event76681
    frameStart := 0 },
  { event := event76682
    frameStart := 0 },
  { event := event76683
    frameStart := 0 },
  { event := event76684
    frameStart := 0 },
  { event := event76685
    frameStart := 0 },
  { event := event76686
    frameStart := 0 },
  { event := event76687
    frameStart := 76687 }
]

def eventLeaf4793 : Array AnnotatedEvent := #[
  { event := event76688
    frameStart := 76687 },
  { event := event76689
    frameStart := 76687 },
  { event := event76690
    frameStart := 76687 },
  { event := event76691
    frameStart := 76687 },
  { event := event76692
    frameStart := 76687 },
  { event := event76693
    frameStart := 76687 },
  { event := event76694
    frameStart := 76687 },
  { event := event76695
    frameStart := 76687 },
  { event := event76696
    frameStart := 76687 },
  { event := event76697
    frameStart := 76687 },
  { event := event76698
    frameStart := 76687 },
  { event := event76699
    frameStart := 76687 },
  { event := event76700
    frameStart := 76687 },
  { event := event76701
    frameStart := 76687 },
  { event := event76702
    frameStart := 76687 },
  { event := event76703
    frameStart := 76687 }
]

def eventLeaf4794 : Array AnnotatedEvent := #[
  { event := event76704
    frameStart := 76687 },
  { event := event76705
    frameStart := 76687 },
  { event := event76706
    frameStart := 76687 },
  { event := event76707
    frameStart := 76687 },
  { event := event76708
    frameStart := 76687 },
  { event := event76709
    frameStart := 76687 },
  { event := event76710
    frameStart := 76687 },
  { event := event76711
    frameStart := 76687 },
  { event := event76712
    frameStart := 76687 },
  { event := event76713
    frameStart := 76687 },
  { event := event76714
    frameStart := 76687 },
  { event := event76715
    frameStart := 76687 },
  { event := event76716
    frameStart := 76687 },
  { event := event76717
    frameStart := 76687 },
  { event := event76718
    frameStart := 76687 },
  { event := event76719
    frameStart := 76687 }
]

def eventLeaf4795 : Array AnnotatedEvent := #[
  { event := event76720
    frameStart := 76687 },
  { event := event76721
    frameStart := 76687 },
  { event := event76722
    frameStart := 76687 },
  { event := event76723
    frameStart := 76687 },
  { event := event76724
    frameStart := 76687 },
  { event := event76725
    frameStart := 76687 },
  { event := event76726
    frameStart := 76687 },
  { event := event76727
    frameStart := 76687 },
  { event := event76728
    frameStart := 76687 },
  { event := event76729
    frameStart := 76687 },
  { event := event76730
    frameStart := 76687 },
  { event := event76731
    frameStart := 76687 },
  { event := event76732
    frameStart := 76687 },
  { event := event76733
    frameStart := 76687 },
  { event := event76734
    frameStart := 76687 },
  { event := event76735
    frameStart := 76687 }
]

def eventLeaf4796 : Array AnnotatedEvent := #[
  { event := event76736
    frameStart := 76687 },
  { event := event76737
    frameStart := 76687 },
  { event := event76738
    frameStart := 76687 },
  { event := event76739
    frameStart := 76687 },
  { event := event76740
    frameStart := 76687 },
  { event := event76741
    frameStart := 76741 },
  { event := event76742
    frameStart := 76741 },
  { event := event76743
    frameStart := 76741 },
  { event := event76744
    frameStart := 76741 },
  { event := event76745
    frameStart := 76741 },
  { event := event76746
    frameStart := 76741 },
  { event := event76747
    frameStart := 76741 },
  { event := event76748
    frameStart := 76741 },
  { event := event76749
    frameStart := 76741 },
  { event := event76750
    frameStart := 76741 },
  { event := event76751
    frameStart := 76741 }
]

def eventLeaf4797 : Array AnnotatedEvent := #[
  { event := event76752
    frameStart := 76741 },
  { event := event76753
    frameStart := 76741 },
  { event := event76754
    frameStart := 76741 },
  { event := event76755
    frameStart := 76741 },
  { event := event76756
    frameStart := 76741 },
  { event := event76757
    frameStart := 76741 },
  { event := event76758
    frameStart := 76741 },
  { event := event76759
    frameStart := 76741 },
  { event := event76760
    frameStart := 76741 },
  { event := event76761
    frameStart := 76741 },
  { event := event76762
    frameStart := 76741 },
  { event := event76763
    frameStart := 76741 },
  { event := event76764
    frameStart := 76741 },
  { event := event76765
    frameStart := 76741 },
  { event := event76766
    frameStart := 76741 },
  { event := event76767
    frameStart := 76741 }
]

def eventLeaf4798 : Array AnnotatedEvent := #[
  { event := event76768
    frameStart := 76741 },
  { event := event76769
    frameStart := 76741 },
  { event := event76770
    frameStart := 76741 },
  { event := event76771
    frameStart := 76741 },
  { event := event76772
    frameStart := 76741 },
  { event := event76773
    frameStart := 76741 },
  { event := event76774
    frameStart := 76741 },
  { event := event76775
    frameStart := 76741 },
  { event := event76776
    frameStart := 76741 },
  { event := event76777
    frameStart := 76741 },
  { event := event76778
    frameStart := 76741 },
  { event := event76779
    frameStart := 76741 },
  { event := event76780
    frameStart := 76741 },
  { event := event76781
    frameStart := 76741 },
  { event := event76782
    frameStart := 76741 },
  { event := event76783
    frameStart := 76741 }
]

def eventLeaf4799 : Array AnnotatedEvent := #[
  { event := event76784
    frameStart := 76741 },
  { event := event76785
    frameStart := 76741 },
  { event := event76786
    frameStart := 76741 },
  { event := event76787
    frameStart := 76741 },
  { event := event76788
    frameStart := 76741 },
  { event := event76789
    frameStart := 76741 },
  { event := event76790
    frameStart := 76741 },
  { event := event76791
    frameStart := 76741 },
  { event := event76792
    frameStart := 76741 },
  { event := event76793
    frameStart := 76741 },
  { event := event76794
    frameStart := 76741 },
  { event := event76795
    frameStart := 76741 },
  { event := event76796
    frameStart := 76741 },
  { event := event76797
    frameStart := 76741 },
  { event := event76798
    frameStart := 76741 },
  { event := event76799
    frameStart := 76741 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events299
