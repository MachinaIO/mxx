import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events260

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event66560 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60839⟩⟩, .relation 66559 1, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩)

def event66561 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60839⟩⟩, .relation 66559 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62109⟩⟩]⟩, (-1)⟩)

def event66562 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60839⟩⟩, .relation 66559 2, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], [⟨.program ⟨257⟩, ⟨61164⟩⟩]⟩, (1)⟩)

def event66563 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60839⟩⟩, .relation 66559 3, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact66564RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62109⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], [⟨.program ⟨257⟩, ⟨61164⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact66564RawTermsValid :
    exact66564RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66564 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60839⟩⟩) exact66564RawTerms .large 66396 (.finite 202072841853861888) (some (66398))

def event66565 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62112⟩⟩) 0 ⟨60839⟩ 66564

def event66566 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62112⟩⟩) 1 ⟨62111⟩ 66386

def event66567 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62112⟩⟩) (.sum [.predecessor 0 66565 .coefficient, .predecessor 1 66566 .coefficient])

def event66568 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62112⟩⟩, .operator (⟨66564, 0⟩, ⟨66386, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62109⟩⟩]⟩, (1)⟩)

def event66569 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨62112⟩⟩, .operator (⟨66564, 2⟩, ⟨66386, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59884⟩⟩], [⟨.program ⟨257⟩, ⟨61164⟩⟩]⟩, (-1)⟩)

def event66570 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62112⟩⟩) (.sum [.result 66564 .summary, .result 66386 .summary])

def exact66571RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨60234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact66571RawTermsValid :
    exact66571RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66571 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62112⟩⟩) exact66571RawTerms .large 66567 (.finite 32190378816049205907437743505408) (some (66570))

def event66572 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58182⟩⟩) 0 ⟨56905⟩ 2608

def event66573 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58182⟩⟩) (.authority (.programFamilyFact))

def event66574 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨58182⟩⟩) (.finite 3720)

def event66575 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58184⟩⟩) 0 ⟨7177⟩ 15500

def event66576 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58184⟩⟩) 1 ⟨58182⟩ 66574

def event66577 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58184⟩⟩) (.authority (.operator))

def exact66578RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58184⟩⟩]⟩, (1)⟩]

theorem exact66578RawTermsValid :
    exact66578RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66578 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58184⟩⟩) exact66578RawTerms .large 66577 .exactZero (none)

def event66579 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59129⟩⟩) 0 ⟨58184⟩ 66578

def event66580 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59129⟩⟩) (.authority (.operator))

def exact66581RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨59129⟩⟩]⟩, (1)⟩]

theorem exact66581RawTermsValid :
    exact66581RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66581 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59129⟩⟩) exact66581RawTerms (.finite 8192) 66580 .exactZero (none)

def event66582 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58010⟩⟩) 0 ⟨56696⟩ 2602

def event66583 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58010⟩⟩) (.authority (.programFamilyFact))

def event66584 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨58010⟩⟩) (.finite 3720)

def event66585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58011⟩⟩) 0 ⟨7177⟩ 15500

def event66586 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58011⟩⟩) 1 ⟨58010⟩ 66584

def event66587 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58011⟩⟩) (.authority (.operator))

def exact66588RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58011⟩⟩]⟩, (1)⟩]

theorem exact66588RawTermsValid :
    exact66588RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66588 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58011⟩⟩) exact66588RawTerms .large 66587 .exactZero (none)

def event66589 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58556⟩⟩) 0 ⟨58011⟩ 66588

def event66590 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58556⟩⟩) (.authority (.operator))

def exact66591RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58556⟩⟩]⟩, (1)⟩]

theorem exact66591RawTermsValid :
    exact66591RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66591 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58556⟩⟩) exact66591RawTerms (.finite 8192) 66590 .exactZero (none)

def event66592 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25095⟩⟩) 0 ⟨25094⟩ 2591

def event66593 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25095⟩⟩) 1 ⟨10752⟩ 61278

def event66594 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25095⟩⟩) (.tensor (.predecessor 0 66592 .coefficient) (.predecessor 1 66593 .coefficient) true false)

def event66595 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨25095⟩⟩, .operator (⟨2591, 0⟩, ⟨61278, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact66596RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact66596RawTermsValid :
    exact66596RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66596 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25095⟩⟩) exact66596RawTerms .large 66594 .exactZero (none)

def event66597 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10755⟩⟩) 0 ⟨10751⟩ 61148

def event66598 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10755⟩⟩) 1 ⟨7273⟩ 22591

def event66599 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10755⟩⟩) (.product (.predecessor 0 66597 .coefficient) (.predecessor 1 66598 .coefficient) (⟨false, false, none, none, none⟩))

def event66600 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10755⟩⟩, .operator (⟨61148, 0⟩, ⟨22591, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩)

def exact66601RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩]

theorem exact66601RawTermsValid :
    exact66601RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66601 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10755⟩⟩) exact66601RawTerms .large 66599 .exactZero (none)

def event66602 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25096⟩⟩) 0 ⟨10755⟩ 66601

def event66603 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25096⟩⟩) 1 ⟨25095⟩ 66596

def event66604 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25096⟩⟩) (.sum [.predecessor 0 66602 .coefficient, .predecessor 1 66603 .coefficient])

def exact66605RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact66605RawTermsValid :
    exact66605RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66605 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25096⟩⟩) exact66605RawTerms .large 66604 .exactZero (none)

def event66606 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25097⟩⟩) 0 ⟨25096⟩ 66605

def event66607 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25097⟩⟩) 1 ⟨99⟩ 22583

def event66608 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25097⟩⟩) (.sum [.predecessor 0 66606 .coefficient, .predecessor 1 66607 .coefficient])

def event66609 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25097⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨99⟩⟩]⟩) [⟨.result 22583 .coefficient, false, none⟩])

def event66610 : Event := .survivorFold (1) 66609

def exact66611RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact66611RawTermsValid :
    exact66611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66611 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25097⟩⟩) exact66611RawTerms .large 66608 (.finite 26) (some (66609))

def event66612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56697⟩⟩) 0 ⟨25097⟩ 66611

def event66613 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56697⟩⟩) 1 ⟨56694⟩ 2594

def event66614 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56697⟩⟩) (.product (.predecessor 0 66612 .coefficient) (.predecessor 1 66613 .coefficient) (⟨false, true, none, none, some 1⟩))

def event66615 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56697⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨56694⟩⟩], []⟩) [⟨.result 2594 .coefficient, true, some 1⟩])

def event66616 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56697⟩⟩) (.product (.result 66611 .summary) (.transfer 66615) (⟨false, false, none, none, none⟩))

def event66617 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56697⟩⟩, .operator (⟨66611, 1⟩, ⟨2594, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25094⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event66618 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56697⟩⟩, .operator (⟨66611, 0⟩, ⟨2594, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩)

def exact66619RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25094⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩]

theorem exact66619RawTermsValid :
    exact66619RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66619 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56697⟩⟩) exact66619RawTerms .large 66614 (.finite 13631488) (some (66616))

def event66620 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56698⟩⟩) 0 ⟨56694⟩ 2594

def event66621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56698⟩⟩) 1 ⟨10752⟩ 61278

def event66622 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56698⟩⟩) (.tensor (.predecessor 0 66620 .coefficient) (.predecessor 1 66621 .coefficient) true false)

def event66623 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56698⟩⟩, .operator (⟨2594, 0⟩, ⟨61278, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact66624RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact66624RawTermsValid :
    exact66624RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66624 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56698⟩⟩) exact66624RawTerms .large 66622 .exactZero (none)

def event66625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10772⟩⟩) 0 ⟨10751⟩ 61148

def event66626 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10772⟩⟩) 1 ⟨7290⟩ 22632

def event66627 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10772⟩⟩) (.product (.predecessor 0 66625 .coefficient) (.predecessor 1 66626 .coefficient) (⟨false, false, none, none, none⟩))

def event66628 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10772⟩⟩, .operator (⟨61148, 0⟩, ⟨22632, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩]⟩, (1)⟩)

def exact66629RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩]⟩, (1)⟩]

theorem exact66629RawTermsValid :
    exact66629RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66629 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10772⟩⟩) exact66629RawTerms .large 66627 .exactZero (none)

def event66630 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56699⟩⟩) 0 ⟨10772⟩ 66629

def event66631 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56699⟩⟩) 1 ⟨56698⟩ 66624

def event66632 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56699⟩⟩) (.sum [.predecessor 0 66630 .coefficient, .predecessor 1 66631 .coefficient])

def exact66633RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact66633RawTermsValid :
    exact66633RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66633 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56699⟩⟩) exact66633RawTerms .large 66632 .exactZero (none)

def event66634 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56700⟩⟩) 0 ⟨56699⟩ 66633

def event66635 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56700⟩⟩) 1 ⟨116⟩ 22624

def event66636 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56700⟩⟩) (.sum [.predecessor 0 66634 .coefficient, .predecessor 1 66635 .coefficient])

def event66637 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56700⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨116⟩⟩]⟩) [⟨.result 22624 .coefficient, false, none⟩])

def event66638 : Event := .survivorFold (1) 66637

def exact66639RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact66639RawTermsValid :
    exact66639RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66639 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56700⟩⟩) exact66639RawTerms .large 66636 (.finite 26) (some (66637))

def event66640 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56701⟩⟩) 0 ⟨56700⟩ 66639

def event66641 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56701⟩⟩) 1 ⟨9533⟩ 22621

def event66642 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56701⟩⟩) (.product (.predecessor 0 66640 .coefficient) (.predecessor 1 66641 .coefficient) (⟨false, false, none, none, none⟩))

def event66643 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56701⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) [⟨.result 22617 .coefficient, false, none⟩])

def event66644 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56701⟩⟩) (.product (.result 66639 .summary) (.transfer 66643) (⟨false, false, none, none, none⟩))

def event66645 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56701⟩⟩, .operator (⟨66639, 1⟩, ⟨22621, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩, (-1)⟩)

def event66646 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨56701⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9532⟩⟩) ⟨7273⟩ 22591)

def event66647 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56701⟩⟩, .relation 66646 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (-1)⟩)

def event66648 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56701⟩⟩, .operator (⟨66639, 0⟩, ⟨22621, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩, (1)⟩)

def exact66649RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (-1)⟩]

theorem exact66649RawTermsValid :
    exact66649RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66649 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56701⟩⟩) exact66649RawTerms .large 66642 (.finite 279172874240) (some (66644))

def event66650 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56702⟩⟩) 0 ⟨56701⟩ 66649

def event66651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56702⟩⟩) 1 ⟨56697⟩ 66619

def event66652 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56702⟩⟩) (.sum [.predecessor 0 66650 .coefficient, .predecessor 1 66651 .coefficient])

def event66653 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56702⟩⟩, .operator (⟨66649, 1⟩, ⟨66619, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩)

def event66654 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56702⟩⟩) (.sum [.result 66649 .summary, .result 66619 .summary])

def exact66655RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25094⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact66655RawTermsValid :
    exact66655RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66655 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56702⟩⟩) exact66655RawTerms .large 66652 (.finite 279186505728) (some (66654))

def event66656 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58557⟩⟩) 0 ⟨56702⟩ 66655

def event66657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58557⟩⟩) 1 ⟨58556⟩ 66591

def event66658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58557⟩⟩) (.product (.predecessor 0 66656 .coefficient) (.predecessor 1 66657 .coefficient) (⟨false, false, none, none, none⟩))

def event66659 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58557⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨58556⟩⟩]⟩) [⟨.result 66591 .coefficient, false, none⟩])

def event66660 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58557⟩⟩) (.product (.result 66655 .summary) (.transfer 66659) (⟨false, false, none, none, none⟩))

def event66661 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58557⟩⟩, .operator (⟨66655, 1⟩, ⟨66591, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25094⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58556⟩⟩]⟩, (-1)⟩)

def event66662 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨58557⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25094⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58556⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58556⟩⟩) ⟨58011⟩ 66588)

def event66663 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58557⟩⟩, .relation 66662 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25094⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨58011⟩⟩]⟩, (-1)⟩)

def event66664 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58557⟩⟩, .operator (⟨66655, 0⟩, ⟨66591, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58556⟩⟩]⟩, (1)⟩)

def exact66665RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58556⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25094⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨58011⟩⟩]⟩, (-1)⟩]

theorem exact66665RawTermsValid :
    exact66665RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66665 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58557⟩⟩) exact66665RawTerms .large 66658 (.finite 2997742278965691678720) (some (66660))

def event66666 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57479⟩⟩) 0 ⟨56696⟩ 2602

def event66667 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57479⟩⟩) (.authority (.relationPreimageSource ⟨42⟩))

def exact66668RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨57479⟩⟩]⟩, (1)⟩]

theorem exact66668RawTermsValid :
    exact66668RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66668 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57479⟩⟩) exact66668RawTerms (.finite 5647228698) 66667 .exactZero (none)

def event66669 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57481⟩⟩) 0 ⟨57479⟩ 66668

def event66670 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57481⟩⟩) 1 ⟨2370⟩ 4

def event66671 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57481⟩⟩) (.scale (.predecessor 0 66669 .coefficient) (.value (.predecessor 1 66670 .coefficient)))

def exact66672RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨57479⟩⟩]⟩, (1)⟩]

theorem exact66672RawTermsValid :
    exact66672RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66672 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57481⟩⟩) exact66672RawTerms (.finite 5647228698) 66671 .exactZero (none)

def event66673 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57482⟩⟩) 0 ⟨10792⟩ 61370

def event66674 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57482⟩⟩) 1 ⟨57481⟩ 66672

def event66675 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57482⟩⟩) (.product (.predecessor 0 66673 .coefficient) (.predecessor 1 66674 .coefficient) (⟨false, false, none, none, none⟩))

def event66676 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57482⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨57479⟩⟩]⟩) [⟨.result 66668 .coefficient, false, none⟩])

def event66677 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57482⟩⟩) (.product (.result 61370 .summary) (.transfer 66676) (⟨false, false, none, none, none⟩))

def event66678 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57482⟩⟩, .operator (⟨61370, 0⟩, ⟨66672, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57479⟩⟩]⟩, (1)⟩)

def event66679 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨57480⟩⟩)

def event66680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event66681 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event66682 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.authority (.operator))

def event66683 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.finite 16)

def event66684 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event66685 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event66686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event66687 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event66688 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 66687

def event66689 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 66685

def event66690 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 66688 .coefficient) (.value (.predecessor 1 66689 .coefficient)))

def event66691 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event66692 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 0 ⟨392⟩ 66691

def event66693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 1 ⟨10691⟩ 66683

def event66694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.sum [.predecessor 0 66692 .coefficient, .predecessor 1 66693 .coefficient])

def event66695 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.finite 655356)

def event66696 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 0 ⟨10693⟩ 66695

def event66697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 1 ⟨5426⟩ 66681

def event66698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.identity (.predecessor 1 66697 .coefficient))

def event66699 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.finite 655360)

def event66700 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25094⟩⟩) 0 ⟨10749⟩ 66699

def event66701 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25094⟩⟩) (.authority (.programFamilyFact))

def exact66702RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25094⟩⟩], []⟩, (1)⟩]

theorem exact66702RawTermsValid :
    exact66702RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66702 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25094⟩⟩) exact66702RawTerms (.finite 16) 66701 .exactZero (none)

def event66703 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56694⟩⟩) 0 ⟨10749⟩ 66699

def event66704 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56694⟩⟩) (.authority (.programFamilyFact))

def exact66705RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56694⟩⟩], []⟩, (1)⟩]

theorem exact66705RawTermsValid :
    exact66705RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66705 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56694⟩⟩) exact66705RawTerms (.finite 16) 66704 .exactZero (none)

def event66706 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56695⟩⟩) 0 ⟨56694⟩ 66705

def event66707 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56695⟩⟩) 1 ⟨25094⟩ 66702

def event66708 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56695⟩⟩) (.product (.predecessor 0 66706 .coefficient) (.predecessor 1 66707 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event66709 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56695⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25094⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], []⟩) [⟨.result 66705 .coefficient, true, some 1⟩, ⟨.result 66702 .coefficient, true, some 1⟩])

def event66710 : Event := .survivorFold (1) 66709

def exact66711RawTerms : List Term := []

theorem exact66711RawTermsValid :
    exact66711RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66711 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56695⟩⟩) exact66711RawTerms (.finite 256) 66708 (.finite 256) (some (66709))

def event66712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56696⟩⟩) 0 ⟨56695⟩ 66711

def event66713 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56696⟩⟩) (.identity (.predecessor 0 66712 .coefficient))

def event66714 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56696⟩⟩) (.finite 256)

def event66715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57479⟩⟩) 0 ⟨56696⟩ 66714

def event66716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57479⟩⟩) (.authority (.relationPreimageSource ⟨42⟩))

def exact66717RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨57479⟩⟩]⟩, (1)⟩]

theorem exact66717RawTermsValid :
    exact66717RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66717 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57479⟩⟩) exact66717RawTerms (.finite 5647228698) 66716 .exactZero (none)

def event66718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact66719RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact66719RawTermsValid :
    exact66719RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66719 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact66719RawTerms .large 66718 .exactZero (none)

def event66720 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57480⟩⟩) 0 ⟨35⟩ 66719

def event66721 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57480⟩⟩) 1 ⟨57479⟩ 66717

def event66722 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57480⟩⟩) (.product (.predecessor 0 66720 .coefficient) (.predecessor 1 66721 .coefficient) (⟨false, false, none, none, none⟩))

def event66723 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57480⟩⟩, .operator (⟨66719, 0⟩, ⟨66717, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57479⟩⟩]⟩, (1)⟩)

def exact66724RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57479⟩⟩]⟩, (1)⟩]

theorem exact66724RawTermsValid :
    exact66724RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66724 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57480⟩⟩) exact66724RawTerms .large 66722 .exactZero (none)

def event66725 : Event := .preFoldPolynomial 66724 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57479⟩⟩]⟩, (1)⟩] .exactZero none

def exact66726RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57479⟩⟩]⟩, (1)⟩]

def event66726 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨57480⟩⟩) 66725 exact66726RawTerms .large 66722 .exactZero (none)

def event66727 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨58560⟩⟩)

def event66728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event66729 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event66730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.authority (.operator))

def event66731 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.finite 16)

def event66732 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event66733 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event66734 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event66735 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event66736 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 66735

def event66737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 66733

def event66738 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 66736 .coefficient) (.value (.predecessor 1 66737 .coefficient)))

def event66739 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event66740 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 0 ⟨392⟩ 66739

def event66741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 1 ⟨10691⟩ 66731

def event66742 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.sum [.predecessor 0 66740 .coefficient, .predecessor 1 66741 .coefficient])

def event66743 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.finite 655356)

def event66744 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 0 ⟨10693⟩ 66743

def event66745 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 1 ⟨5426⟩ 66729

def event66746 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.identity (.predecessor 1 66745 .coefficient))

def event66747 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.finite 655360)

def event66748 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25094⟩⟩) 0 ⟨10749⟩ 66747

def event66749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25094⟩⟩) (.authority (.programFamilyFact))

def exact66750RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25094⟩⟩], []⟩, (1)⟩]

theorem exact66750RawTermsValid :
    exact66750RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66750 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25094⟩⟩) exact66750RawTerms (.finite 16) 66749 .exactZero (none)

def event66751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56694⟩⟩) 0 ⟨10749⟩ 66747

def event66752 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56694⟩⟩) (.authority (.programFamilyFact))

def exact66753RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56694⟩⟩], []⟩, (1)⟩]

theorem exact66753RawTermsValid :
    exact66753RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66753 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56694⟩⟩) exact66753RawTerms (.finite 16) 66752 .exactZero (none)

def event66754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56695⟩⟩) 0 ⟨56694⟩ 66753

def event66755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56695⟩⟩) 1 ⟨25094⟩ 66750

def event66756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56695⟩⟩) (.product (.predecessor 0 66754 .coefficient) (.predecessor 1 66755 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event66757 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56695⟩⟩, .operator (⟨66753, 0⟩, ⟨66750, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25094⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], []⟩, (1)⟩)

def exact66758RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25094⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], []⟩, (1)⟩]

theorem exact66758RawTermsValid :
    exact66758RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66758 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56695⟩⟩) exact66758RawTerms (.finite 256) 66756 .exactZero (none)

def event66759 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56696⟩⟩) 0 ⟨56695⟩ 66758

def event66760 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56696⟩⟩) (.identity (.predecessor 0 66759 .coefficient))

def event66761 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56696⟩⟩) (.finite 256)

def event66762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58010⟩⟩) 0 ⟨56696⟩ 66761

def event66763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58010⟩⟩) (.authority (.programFamilyFact))

def event66764 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨58010⟩⟩) (.finite 3720)

def event66765 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event66766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58011⟩⟩) 0 ⟨7177⟩ 66765

def event66767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58011⟩⟩) 1 ⟨58010⟩ 66764

def event66768 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58011⟩⟩) (.authority (.operator))

def exact66769RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58011⟩⟩]⟩, (1)⟩]

theorem exact66769RawTermsValid :
    exact66769RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66769 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58011⟩⟩) exact66769RawTerms .large 66768 .exactZero (none)

def event66770 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58556⟩⟩) 0 ⟨58011⟩ 66769

def event66771 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58556⟩⟩) (.authority (.operator))

def exact66772RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58556⟩⟩]⟩, (1)⟩]

theorem exact66772RawTermsValid :
    exact66772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66772 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58556⟩⟩) exact66772RawTerms (.finite 8192) 66771 .exactZero (none)

def event66773 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event66774 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event66775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58274⟩⟩) 0 ⟨56696⟩ 66761

def event66776 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58274⟩⟩) 1 ⟨136⟩ 66774

def event66777 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58274⟩⟩) (.sum [.predecessor 0 66775 .coefficient, .predecessor 1 66776 .coefficient])

def event66778 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨58274⟩⟩) (.finite 256)

def event66779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58275⟩⟩) 0 ⟨58274⟩ 66778

def event66780 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58275⟩⟩) (.identity (.predecessor 0 66779 .coefficient))

def exact66781RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25094⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], []⟩, (1)⟩]

theorem exact66781RawTermsValid :
    exact66781RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66781 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58275⟩⟩) exact66781RawTerms (.finite 256) 66780 .exactZero (none)

def event66782 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact66783RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact66783RawTermsValid :
    exact66783RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66783 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact66783RawTerms .large 66782 .exactZero (none)

def event66784 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58276⟩⟩) 0 ⟨6908⟩ 66783

def event66785 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58276⟩⟩) 1 ⟨58275⟩ 66781

def event66786 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58276⟩⟩) (.product (.predecessor 0 66784 .coefficient) (.predecessor 1 66785 .coefficient) (⟨false, false, none, none, none⟩))

def event66787 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58276⟩⟩, .operator (⟨66783, 0⟩, ⟨66781, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25094⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact66788RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25094⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact66788RawTermsValid :
    exact66788RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66788 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58276⟩⟩) exact66788RawTerms .large 66786 .exactZero (none)

def event66789 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event66790 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event66791 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 66765

def event66792 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact66793RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact66793RawTermsValid :
    exact66793RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66793 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact66793RawTerms .large 66792 .exactZero (none)

def event66794 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7273⟩⟩) 0 ⟨7178⟩ 66793

def event66795 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7273⟩⟩) (.identity (.predecessor 0 66794 .coefficient))

def exact66796RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩]

theorem exact66796RawTermsValid :
    exact66796RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66796 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7273⟩⟩) exact66796RawTerms .large 66795 .exactZero (none)

def event66797 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9532⟩⟩) 0 ⟨7273⟩ 66796

def event66798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9532⟩⟩) (.authority (.operator))

def exact66799RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩, (1)⟩]

theorem exact66799RawTermsValid :
    exact66799RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66799 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9532⟩⟩) exact66799RawTerms (.finite 8192) 66798 .exactZero (none)

def event66800 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9533⟩⟩) 0 ⟨9532⟩ 66799

def event66801 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9533⟩⟩) 1 ⟨2370⟩ 66790

def event66802 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9533⟩⟩) (.scale (.predecessor 0 66800 .coefficient) (.value (.predecessor 1 66801 .coefficient)))

def exact66803RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩, (1)⟩]

theorem exact66803RawTermsValid :
    exact66803RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66803 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9533⟩⟩) exact66803RawTerms (.finite 8192) 66802 .exactZero (none)

def event66804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7290⟩⟩) 0 ⟨7178⟩ 66793

def event66805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7290⟩⟩) (.identity (.predecessor 0 66804 .coefficient))

def exact66806RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩]⟩, (1)⟩]

theorem exact66806RawTermsValid :
    exact66806RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66806 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7290⟩⟩) exact66806RawTerms .large 66805 .exactZero (none)

def event66807 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9534⟩⟩) 0 ⟨7290⟩ 66806

def event66808 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9534⟩⟩) 1 ⟨9533⟩ 66803

def event66809 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9534⟩⟩) (.product (.predecessor 0 66807 .coefficient) (.predecessor 1 66808 .coefficient) (⟨false, false, none, none, none⟩))

def event66810 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9534⟩⟩, .operator (⟨66806, 0⟩, ⟨66803, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩, (1)⟩)

def exact66811RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩, (1)⟩]

theorem exact66811RawTermsValid :
    exact66811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66811 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9534⟩⟩) exact66811RawTerms .large 66809 .exactZero (none)

def event66812 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58277⟩⟩) 0 ⟨9534⟩ 66811

def event66813 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58277⟩⟩) 1 ⟨58276⟩ 66788

def event66814 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58277⟩⟩) (.sum [.predecessor 0 66812 .coefficient, .predecessor 1 66813 .coefficient])

def exact66815RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25094⟩⟩, ⟨.program ⟨257⟩, ⟨56694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact66815RawTermsValid :
    exact66815RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66815 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58277⟩⟩) exact66815RawTerms .large 66814 .exactZero (none)

def eventLeaf4160 : Array AnnotatedEvent := #[
  { event := event66560
    frameStart := 0 },
  { event := event66561
    frameStart := 0 },
  { event := event66562
    frameStart := 0 },
  { event := event66563
    frameStart := 0 },
  { event := event66564
    frameStart := 0 },
  { event := event66565
    frameStart := 0 },
  { event := event66566
    frameStart := 0 },
  { event := event66567
    frameStart := 0 },
  { event := event66568
    frameStart := 0 },
  { event := event66569
    frameStart := 0 },
  { event := event66570
    frameStart := 0 },
  { event := event66571
    frameStart := 0 },
  { event := event66572
    frameStart := 0 },
  { event := event66573
    frameStart := 0 },
  { event := event66574
    frameStart := 0 },
  { event := event66575
    frameStart := 0 }
]

def eventLeaf4161 : Array AnnotatedEvent := #[
  { event := event66576
    frameStart := 0 },
  { event := event66577
    frameStart := 0 },
  { event := event66578
    frameStart := 0 },
  { event := event66579
    frameStart := 0 },
  { event := event66580
    frameStart := 0 },
  { event := event66581
    frameStart := 0 },
  { event := event66582
    frameStart := 0 },
  { event := event66583
    frameStart := 0 },
  { event := event66584
    frameStart := 0 },
  { event := event66585
    frameStart := 0 },
  { event := event66586
    frameStart := 0 },
  { event := event66587
    frameStart := 0 },
  { event := event66588
    frameStart := 0 },
  { event := event66589
    frameStart := 0 },
  { event := event66590
    frameStart := 0 },
  { event := event66591
    frameStart := 0 }
]

def eventLeaf4162 : Array AnnotatedEvent := #[
  { event := event66592
    frameStart := 0 },
  { event := event66593
    frameStart := 0 },
  { event := event66594
    frameStart := 0 },
  { event := event66595
    frameStart := 0 },
  { event := event66596
    frameStart := 0 },
  { event := event66597
    frameStart := 0 },
  { event := event66598
    frameStart := 0 },
  { event := event66599
    frameStart := 0 },
  { event := event66600
    frameStart := 0 },
  { event := event66601
    frameStart := 0 },
  { event := event66602
    frameStart := 0 },
  { event := event66603
    frameStart := 0 },
  { event := event66604
    frameStart := 0 },
  { event := event66605
    frameStart := 0 },
  { event := event66606
    frameStart := 0 },
  { event := event66607
    frameStart := 0 }
]

def eventLeaf4163 : Array AnnotatedEvent := #[
  { event := event66608
    frameStart := 0 },
  { event := event66609
    frameStart := 0 },
  { event := event66610
    frameStart := 0 },
  { event := event66611
    frameStart := 0 },
  { event := event66612
    frameStart := 0 },
  { event := event66613
    frameStart := 0 },
  { event := event66614
    frameStart := 0 },
  { event := event66615
    frameStart := 0 },
  { event := event66616
    frameStart := 0 },
  { event := event66617
    frameStart := 0 },
  { event := event66618
    frameStart := 0 },
  { event := event66619
    frameStart := 0 },
  { event := event66620
    frameStart := 0 },
  { event := event66621
    frameStart := 0 },
  { event := event66622
    frameStart := 0 },
  { event := event66623
    frameStart := 0 }
]

def eventLeaf4164 : Array AnnotatedEvent := #[
  { event := event66624
    frameStart := 0 },
  { event := event66625
    frameStart := 0 },
  { event := event66626
    frameStart := 0 },
  { event := event66627
    frameStart := 0 },
  { event := event66628
    frameStart := 0 },
  { event := event66629
    frameStart := 0 },
  { event := event66630
    frameStart := 0 },
  { event := event66631
    frameStart := 0 },
  { event := event66632
    frameStart := 0 },
  { event := event66633
    frameStart := 0 },
  { event := event66634
    frameStart := 0 },
  { event := event66635
    frameStart := 0 },
  { event := event66636
    frameStart := 0 },
  { event := event66637
    frameStart := 0 },
  { event := event66638
    frameStart := 0 },
  { event := event66639
    frameStart := 0 }
]

def eventLeaf4165 : Array AnnotatedEvent := #[
  { event := event66640
    frameStart := 0 },
  { event := event66641
    frameStart := 0 },
  { event := event66642
    frameStart := 0 },
  { event := event66643
    frameStart := 0 },
  { event := event66644
    frameStart := 0 },
  { event := event66645
    frameStart := 0 },
  { event := event66646
    frameStart := 0 },
  { event := event66647
    frameStart := 0 },
  { event := event66648
    frameStart := 0 },
  { event := event66649
    frameStart := 0 },
  { event := event66650
    frameStart := 0 },
  { event := event66651
    frameStart := 0 },
  { event := event66652
    frameStart := 0 },
  { event := event66653
    frameStart := 0 },
  { event := event66654
    frameStart := 0 },
  { event := event66655
    frameStart := 0 }
]

def eventLeaf4166 : Array AnnotatedEvent := #[
  { event := event66656
    frameStart := 0 },
  { event := event66657
    frameStart := 0 },
  { event := event66658
    frameStart := 0 },
  { event := event66659
    frameStart := 0 },
  { event := event66660
    frameStart := 0 },
  { event := event66661
    frameStart := 0 },
  { event := event66662
    frameStart := 0 },
  { event := event66663
    frameStart := 0 },
  { event := event66664
    frameStart := 0 },
  { event := event66665
    frameStart := 0 },
  { event := event66666
    frameStart := 0 },
  { event := event66667
    frameStart := 0 },
  { event := event66668
    frameStart := 0 },
  { event := event66669
    frameStart := 0 },
  { event := event66670
    frameStart := 0 },
  { event := event66671
    frameStart := 0 }
]

def eventLeaf4167 : Array AnnotatedEvent := #[
  { event := event66672
    frameStart := 0 },
  { event := event66673
    frameStart := 0 },
  { event := event66674
    frameStart := 0 },
  { event := event66675
    frameStart := 0 },
  { event := event66676
    frameStart := 0 },
  { event := event66677
    frameStart := 0 },
  { event := event66678
    frameStart := 0 },
  { event := event66679
    frameStart := 66679 },
  { event := event66680
    frameStart := 66679 },
  { event := event66681
    frameStart := 66679 },
  { event := event66682
    frameStart := 66679 },
  { event := event66683
    frameStart := 66679 },
  { event := event66684
    frameStart := 66679 },
  { event := event66685
    frameStart := 66679 },
  { event := event66686
    frameStart := 66679 },
  { event := event66687
    frameStart := 66679 }
]

def eventLeaf4168 : Array AnnotatedEvent := #[
  { event := event66688
    frameStart := 66679 },
  { event := event66689
    frameStart := 66679 },
  { event := event66690
    frameStart := 66679 },
  { event := event66691
    frameStart := 66679 },
  { event := event66692
    frameStart := 66679 },
  { event := event66693
    frameStart := 66679 },
  { event := event66694
    frameStart := 66679 },
  { event := event66695
    frameStart := 66679 },
  { event := event66696
    frameStart := 66679 },
  { event := event66697
    frameStart := 66679 },
  { event := event66698
    frameStart := 66679 },
  { event := event66699
    frameStart := 66679 },
  { event := event66700
    frameStart := 66679 },
  { event := event66701
    frameStart := 66679 },
  { event := event66702
    frameStart := 66679 },
  { event := event66703
    frameStart := 66679 }
]

def eventLeaf4169 : Array AnnotatedEvent := #[
  { event := event66704
    frameStart := 66679 },
  { event := event66705
    frameStart := 66679 },
  { event := event66706
    frameStart := 66679 },
  { event := event66707
    frameStart := 66679 },
  { event := event66708
    frameStart := 66679 },
  { event := event66709
    frameStart := 66679 },
  { event := event66710
    frameStart := 66679 },
  { event := event66711
    frameStart := 66679 },
  { event := event66712
    frameStart := 66679 },
  { event := event66713
    frameStart := 66679 },
  { event := event66714
    frameStart := 66679 },
  { event := event66715
    frameStart := 66679 },
  { event := event66716
    frameStart := 66679 },
  { event := event66717
    frameStart := 66679 },
  { event := event66718
    frameStart := 66679 },
  { event := event66719
    frameStart := 66679 }
]

def eventLeaf4170 : Array AnnotatedEvent := #[
  { event := event66720
    frameStart := 66679 },
  { event := event66721
    frameStart := 66679 },
  { event := event66722
    frameStart := 66679 },
  { event := event66723
    frameStart := 66679 },
  { event := event66724
    frameStart := 66679 },
  { event := event66725
    frameStart := 66679 },
  { event := event66726
    frameStart := 66679 },
  { event := event66727
    frameStart := 66727 },
  { event := event66728
    frameStart := 66727 },
  { event := event66729
    frameStart := 66727 },
  { event := event66730
    frameStart := 66727 },
  { event := event66731
    frameStart := 66727 },
  { event := event66732
    frameStart := 66727 },
  { event := event66733
    frameStart := 66727 },
  { event := event66734
    frameStart := 66727 },
  { event := event66735
    frameStart := 66727 }
]

def eventLeaf4171 : Array AnnotatedEvent := #[
  { event := event66736
    frameStart := 66727 },
  { event := event66737
    frameStart := 66727 },
  { event := event66738
    frameStart := 66727 },
  { event := event66739
    frameStart := 66727 },
  { event := event66740
    frameStart := 66727 },
  { event := event66741
    frameStart := 66727 },
  { event := event66742
    frameStart := 66727 },
  { event := event66743
    frameStart := 66727 },
  { event := event66744
    frameStart := 66727 },
  { event := event66745
    frameStart := 66727 },
  { event := event66746
    frameStart := 66727 },
  { event := event66747
    frameStart := 66727 },
  { event := event66748
    frameStart := 66727 },
  { event := event66749
    frameStart := 66727 },
  { event := event66750
    frameStart := 66727 },
  { event := event66751
    frameStart := 66727 }
]

def eventLeaf4172 : Array AnnotatedEvent := #[
  { event := event66752
    frameStart := 66727 },
  { event := event66753
    frameStart := 66727 },
  { event := event66754
    frameStart := 66727 },
  { event := event66755
    frameStart := 66727 },
  { event := event66756
    frameStart := 66727 },
  { event := event66757
    frameStart := 66727 },
  { event := event66758
    frameStart := 66727 },
  { event := event66759
    frameStart := 66727 },
  { event := event66760
    frameStart := 66727 },
  { event := event66761
    frameStart := 66727 },
  { event := event66762
    frameStart := 66727 },
  { event := event66763
    frameStart := 66727 },
  { event := event66764
    frameStart := 66727 },
  { event := event66765
    frameStart := 66727 },
  { event := event66766
    frameStart := 66727 },
  { event := event66767
    frameStart := 66727 }
]

def eventLeaf4173 : Array AnnotatedEvent := #[
  { event := event66768
    frameStart := 66727 },
  { event := event66769
    frameStart := 66727 },
  { event := event66770
    frameStart := 66727 },
  { event := event66771
    frameStart := 66727 },
  { event := event66772
    frameStart := 66727 },
  { event := event66773
    frameStart := 66727 },
  { event := event66774
    frameStart := 66727 },
  { event := event66775
    frameStart := 66727 },
  { event := event66776
    frameStart := 66727 },
  { event := event66777
    frameStart := 66727 },
  { event := event66778
    frameStart := 66727 },
  { event := event66779
    frameStart := 66727 },
  { event := event66780
    frameStart := 66727 },
  { event := event66781
    frameStart := 66727 },
  { event := event66782
    frameStart := 66727 },
  { event := event66783
    frameStart := 66727 }
]

def eventLeaf4174 : Array AnnotatedEvent := #[
  { event := event66784
    frameStart := 66727 },
  { event := event66785
    frameStart := 66727 },
  { event := event66786
    frameStart := 66727 },
  { event := event66787
    frameStart := 66727 },
  { event := event66788
    frameStart := 66727 },
  { event := event66789
    frameStart := 66727 },
  { event := event66790
    frameStart := 66727 },
  { event := event66791
    frameStart := 66727 },
  { event := event66792
    frameStart := 66727 },
  { event := event66793
    frameStart := 66727 },
  { event := event66794
    frameStart := 66727 },
  { event := event66795
    frameStart := 66727 },
  { event := event66796
    frameStart := 66727 },
  { event := event66797
    frameStart := 66727 },
  { event := event66798
    frameStart := 66727 },
  { event := event66799
    frameStart := 66727 }
]

def eventLeaf4175 : Array AnnotatedEvent := #[
  { event := event66800
    frameStart := 66727 },
  { event := event66801
    frameStart := 66727 },
  { event := event66802
    frameStart := 66727 },
  { event := event66803
    frameStart := 66727 },
  { event := event66804
    frameStart := 66727 },
  { event := event66805
    frameStart := 66727 },
  { event := event66806
    frameStart := 66727 },
  { event := event66807
    frameStart := 66727 },
  { event := event66808
    frameStart := 66727 },
  { event := event66809
    frameStart := 66727 },
  { event := event66810
    frameStart := 66727 },
  { event := event66811
    frameStart := 66727 },
  { event := event66812
    frameStart := 66727 },
  { event := event66813
    frameStart := 66727 },
  { event := event66814
    frameStart := 66727 },
  { event := event66815
    frameStart := 66727 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events260
