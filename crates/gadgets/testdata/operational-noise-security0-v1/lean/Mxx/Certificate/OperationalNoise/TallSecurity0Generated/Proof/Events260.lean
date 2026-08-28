import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events260

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event66560 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22551⟩⟩, .operator (⟨65387, 0⟩, ⟨66554, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22548⟩⟩]⟩, (1)⟩)

def event66561 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22549⟩⟩)

def event66562 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event66563 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event66564 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event66565 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event66566 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event66567 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event66568 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event66569 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event66570 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 66569

def event66571 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 66567

def event66572 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 66570 .coefficient) (.value (.predecessor 1 66571 .coefficient)))

def event66573 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event66574 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 66573

def event66575 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 66565

def event66576 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 66574 .coefficient, .predecessor 1 66575 .coefficient])

def event66577 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event66578 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 66577

def event66579 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 66563

def event66580 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 66579 .coefficient))

def event66581 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event66582 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12950⟩⟩) 0 ⟨5530⟩ 66581

def event66583 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12950⟩⟩) (.authority (.programFamilyFact))

def exact66584RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12950⟩⟩], []⟩, (1)⟩]

theorem exact66584RawTermsValid :
    exact66584RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66584 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12950⟩⟩) exact66584RawTerms (.finite 52) 66583 .exactZero (none)

def event66585 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10130⟩⟩) 0 ⟨5530⟩ 66581

def event66586 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10130⟩⟩) (.authority (.programFamilyFact))

def exact66587RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10130⟩⟩], []⟩, (1)⟩]

theorem exact66587RawTermsValid :
    exact66587RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66587 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10130⟩⟩) exact66587RawTerms (.finite 52) 66586 .exactZero (none)

def event66588 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12951⟩⟩) 0 ⟨10130⟩ 66587

def event66589 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12951⟩⟩) 1 ⟨12950⟩ 66584

def event66590 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12951⟩⟩) (.product (.predecessor 0 66588 .coefficient) (.predecessor 1 66589 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event66591 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12951⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], []⟩) [⟨.result 66587 .coefficient, true, some 1⟩, ⟨.result 66584 .coefficient, true, some 1⟩])

def event66592 : Event := .survivorFold (1) 66591

def exact66593RawTerms : List Term := []

theorem exact66593RawTermsValid :
    exact66593RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66593 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12951⟩⟩) exact66593RawTerms (.finite 2704) 66590 (.finite 2704) (some (66591))

def event66594 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12952⟩⟩) 0 ⟨12951⟩ 66593

def event66595 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12952⟩⟩) (.identity (.predecessor 0 66594 .coefficient))

def event66596 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12952⟩⟩) (.finite 2704)

def event66597 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16748⟩⟩) 0 ⟨12952⟩ 66596

def event66598 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16748⟩⟩) (.authority (.programFamilyFact))

def exact66599RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], []⟩, (1)⟩]

theorem exact66599RawTermsValid :
    exact66599RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66599 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16748⟩⟩) exact66599RawTerms (.finite 52) 66598 .exactZero (none)

def event66600 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16749⟩⟩) 0 ⟨16748⟩ 66599

def event66601 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16749⟩⟩) (.identity (.predecessor 0 66600 .coefficient))

def event66602 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16749⟩⟩) (.finite 52)

def event66603 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22548⟩⟩) 0 ⟨16749⟩ 66602

def event66604 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22548⟩⟩) (.authority (.relationPreimageSource ⟨61⟩))

def exact66605RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22548⟩⟩]⟩, (1)⟩]

theorem exact66605RawTermsValid :
    exact66605RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66605 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22548⟩⟩) exact66605RawTerms (.finite 136065468) 66604 .exactZero (none)

def event66606 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact66607RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact66607RawTermsValid :
    exact66607RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66607 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact66607RawTerms .large 66606 .exactZero (none)

def event66608 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22549⟩⟩) 0 ⟨6⟩ 66607

def event66609 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22549⟩⟩) 1 ⟨22548⟩ 66605

def event66610 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22549⟩⟩) (.product (.predecessor 0 66608 .coefficient) (.predecessor 1 66609 .coefficient) (⟨false, false, none, none, none⟩))

def event66611 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22549⟩⟩, .operator (⟨66607, 0⟩, ⟨66605, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22548⟩⟩]⟩, (1)⟩)

def exact66612RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22548⟩⟩]⟩, (1)⟩]

theorem exact66612RawTermsValid :
    exact66612RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66612 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22549⟩⟩) exact66612RawTerms .large 66610 .exactZero (none)

def event66613 : Event := .preFoldPolynomial 66612 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22548⟩⟩]⟩, (1)⟩] .exactZero none

def exact66614RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22548⟩⟩]⟩, (1)⟩]

def event66614 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨22549⟩⟩) 66613 exact66614RawTerms .large 66610 .exactZero (none)

def event66615 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨29594⟩⟩)

def event66616 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event66617 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event66618 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event66619 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event66620 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event66621 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event66622 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event66623 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event66624 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 66623

def event66625 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 66621

def event66626 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 66624 .coefficient) (.value (.predecessor 1 66625 .coefficient)))

def event66627 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event66628 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 66627

def event66629 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 66619

def event66630 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 66628 .coefficient, .predecessor 1 66629 .coefficient])

def event66631 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event66632 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 66631

def event66633 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 66617

def event66634 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 66633 .coefficient))

def event66635 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event66636 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12950⟩⟩) 0 ⟨5530⟩ 66635

def event66637 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12950⟩⟩) (.authority (.programFamilyFact))

def exact66638RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12950⟩⟩], []⟩, (1)⟩]

theorem exact66638RawTermsValid :
    exact66638RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66638 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12950⟩⟩) exact66638RawTerms (.finite 52) 66637 .exactZero (none)

def event66639 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10130⟩⟩) 0 ⟨5530⟩ 66635

def event66640 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10130⟩⟩) (.authority (.programFamilyFact))

def exact66641RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10130⟩⟩], []⟩, (1)⟩]

theorem exact66641RawTermsValid :
    exact66641RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66641 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10130⟩⟩) exact66641RawTerms (.finite 52) 66640 .exactZero (none)

def event66642 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12951⟩⟩) 0 ⟨10130⟩ 66641

def event66643 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12951⟩⟩) 1 ⟨12950⟩ 66638

def event66644 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12951⟩⟩) (.product (.predecessor 0 66642 .coefficient) (.predecessor 1 66643 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event66645 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12951⟩⟩, .operator (⟨66641, 0⟩, ⟨66638, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], []⟩, (1)⟩)

def exact66646RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], []⟩, (1)⟩]

theorem exact66646RawTermsValid :
    exact66646RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66646 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12951⟩⟩) exact66646RawTerms (.finite 2704) 66644 .exactZero (none)

def event66647 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12952⟩⟩) 0 ⟨12951⟩ 66646

def event66648 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12952⟩⟩) (.identity (.predecessor 0 66647 .coefficient))

def event66649 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12952⟩⟩) (.finite 2704)

def event66650 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16748⟩⟩) 0 ⟨12952⟩ 66649

def event66651 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16748⟩⟩) (.authority (.programFamilyFact))

def exact66652RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], []⟩, (1)⟩]

theorem exact66652RawTermsValid :
    exact66652RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66652 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16748⟩⟩) exact66652RawTerms (.finite 52) 66651 .exactZero (none)

def event66653 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16749⟩⟩) 0 ⟨16748⟩ 66652

def event66654 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16749⟩⟩) (.identity (.predecessor 0 66653 .coefficient))

def event66655 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16749⟩⟩) (.finite 52)

def event66656 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24661⟩⟩) 0 ⟨16749⟩ 66655

def event66657 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24661⟩⟩) (.authority (.programFamilyFact))

def event66658 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24661⟩⟩) (.finite 3720)

def event66659 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event66660 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24663⟩⟩) 0 ⟨6689⟩ 66659

def event66661 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24663⟩⟩) 1 ⟨24661⟩ 66658

def event66662 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24663⟩⟩) (.authority (.operator))

def exact66663RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24663⟩⟩]⟩, (1)⟩]

theorem exact66663RawTermsValid :
    exact66663RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66663 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24663⟩⟩) exact66663RawTerms .large 66662 .exactZero (none)

def event66664 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29589⟩⟩) 0 ⟨24663⟩ 66663

def event66665 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29589⟩⟩) (.authority (.operator))

def exact66666RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩, (1)⟩]

theorem exact66666RawTermsValid :
    exact66666RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66666 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29589⟩⟩) exact66666RawTerms (.finite 8192) 66665 .exactZero (none)

def event66667 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event66668 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event66669 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16823⟩⟩) 0 ⟨16749⟩ 66655

def event66670 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16823⟩⟩) 1 ⟨110⟩ 66668

def event66671 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16823⟩⟩) (.sum [.predecessor 0 66669 .coefficient, .predecessor 1 66670 .coefficient])

def event66672 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16823⟩⟩) (.finite 52)

def event66673 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16824⟩⟩) 0 ⟨16823⟩ 66672

def event66674 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16824⟩⟩) (.identity (.predecessor 0 66673 .coefficient))

def exact66675RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], []⟩, (1)⟩]

theorem exact66675RawTermsValid :
    exact66675RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66675 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16824⟩⟩) exact66675RawTerms (.finite 52) 66674 .exactZero (none)

def event66676 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact66677RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact66677RawTermsValid :
    exact66677RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66677 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact66677RawTerms .large 66676 .exactZero (none)

def event66678 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16825⟩⟩) 0 ⟨6544⟩ 66677

def event66679 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16825⟩⟩) 1 ⟨16824⟩ 66675

def event66680 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16825⟩⟩) (.product (.predecessor 0 66678 .coefficient) (.predecessor 1 66679 .coefficient) (⟨false, false, none, none, none⟩))

def event66681 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16825⟩⟩, .operator (⟨66677, 0⟩, ⟨66675, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact66682RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact66682RawTermsValid :
    exact66682RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66682 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16825⟩⟩) exact66682RawTerms .large 66680 .exactZero (none)

def event66683 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6705⟩⟩) 0 ⟨6689⟩ 66659

def event66684 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6705⟩⟩) (.authority (.operator))

def exact66685RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩, (1)⟩]

theorem exact66685RawTermsValid :
    exact66685RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66685 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6705⟩⟩) exact66685RawTerms .large 66684 .exactZero (none)

def event66686 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16826⟩⟩) 0 ⟨6705⟩ 66685

def event66687 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16826⟩⟩) 1 ⟨16825⟩ 66682

def event66688 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16826⟩⟩) (.sum [.predecessor 0 66686 .coefficient, .predecessor 1 66687 .coefficient])

def exact66689RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact66689RawTermsValid :
    exact66689RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66689 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16826⟩⟩) exact66689RawTerms .large 66688 .exactZero (none)

def event66690 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29590⟩⟩) 0 ⟨16826⟩ 66689

def event66691 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29590⟩⟩) 1 ⟨29589⟩ 66666

def event66692 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29590⟩⟩) (.product (.predecessor 0 66690 .coefficient) (.predecessor 1 66691 .coefficient) (⟨false, false, none, none, none⟩))

def event66693 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29590⟩⟩, .operator (⟨66689, 0⟩, ⟨66666, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩, (1)⟩)

def event66694 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29590⟩⟩, .operator (⟨66689, 1⟩, ⟨66666, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩, (-1)⟩)

def event66695 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29590⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29589⟩⟩) ⟨24663⟩ 66663)

def event66696 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29590⟩⟩, .relation 66695 0, ⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24663⟩⟩]⟩, (-1)⟩)

def exact66697RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24663⟩⟩]⟩, (-1)⟩]

theorem exact66697RawTermsValid :
    exact66697RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66697 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29590⟩⟩) exact66697RawTerms .large 66692 .exactZero (none)

def event66698 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16795⟩⟩) 0 ⟨16749⟩ 66655

def event66699 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16795⟩⟩) (.authority (.programFamilyFact))

def exact66700RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16795⟩⟩], []⟩, (1)⟩]

theorem exact66700RawTermsValid :
    exact66700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66700 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16795⟩⟩) exact66700RawTerms (.finite 63) 66699 .exactZero (none)

def event66701 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16796⟩⟩) 0 ⟨6544⟩ 66677

def event66702 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16796⟩⟩) 1 ⟨16795⟩ 66700

def event66703 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16796⟩⟩) (.product (.predecessor 0 66701 .coefficient) (.predecessor 1 66702 .coefficient) (⟨false, true, none, none, some 1⟩))

def event66704 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16796⟩⟩, .operator (⟨66677, 0⟩, ⟨66700, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16795⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact66705RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16795⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact66705RawTermsValid :
    exact66705RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66705 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16796⟩⟩) exact66705RawTerms .large 66703 .exactZero (none)

def event66706 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6739⟩⟩) 0 ⟨6689⟩ 66659

def event66707 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6739⟩⟩) (.authority (.operator))

def exact66708RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩]

theorem exact66708RawTermsValid :
    exact66708RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66708 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6739⟩⟩) exact66708RawTerms .large 66707 .exactZero (none)

def event66709 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16797⟩⟩) 0 ⟨6739⟩ 66708

def event66710 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16797⟩⟩) 1 ⟨16796⟩ 66705

def event66711 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16797⟩⟩) (.sum [.predecessor 0 66709 .coefficient, .predecessor 1 66710 .coefficient])

def exact66712RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16795⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact66712RawTermsValid :
    exact66712RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66712 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16797⟩⟩) exact66712RawTerms .large 66711 .exactZero (none)

def event66713 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29594⟩⟩) 0 ⟨16797⟩ 66712

def event66714 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29594⟩⟩) 1 ⟨29590⟩ 66697

def event66715 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29594⟩⟩) (.sum [.predecessor 0 66713 .coefficient, .predecessor 1 66714 .coefficient])

def exact66716RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16795⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact66716RawTermsValid :
    exact66716RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66716 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29594⟩⟩) exact66716RawTerms .large 66715 .exactZero (none)

def event66717 : Event := .preFoldPolynomial 66716 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16795⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact66718RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16795⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event66718 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨29594⟩⟩) 66717 exact66718RawTerms .large 66715 .exactZero (none)

def event66719 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16749⟩⟩) ⟨⟨152⟩, ⟨61⟩, ⟨109⟩⟩ ⟨66561, 66719⟩

def event66720 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨22551⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22548⟩⟩]⟩) (1) 0 2 (.universal 66719 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22548⟩⟩]⟩) (none) 66718)

def event66721 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22551⟩⟩, .relation 66720 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩)

def event66722 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22551⟩⟩, .relation 66720 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩, (-1)⟩)

def event66723 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22551⟩⟩, .relation 66720 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24663⟩⟩]⟩, (1)⟩)

def event66724 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22551⟩⟩, .relation 66720 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16795⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact66725RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24663⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16795⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact66725RawTermsValid :
    exact66725RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66725 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22551⟩⟩) exact66725RawTerms .large 66557 (.finite 1811303510016) (some (66559))

def event66726 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29592⟩⟩) 0 ⟨22551⟩ 66725

def event66727 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29592⟩⟩) 1 ⟨29591⟩ 66547

def event66728 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29592⟩⟩) (.sum [.predecessor 0 66726 .coefficient, .predecessor 1 66727 .coefficient])

def event66729 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29592⟩⟩, .operator (⟨66725, 0⟩, ⟨66547, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩, (1)⟩)

def event66730 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29592⟩⟩, .operator (⟨66725, 2⟩, ⟨66547, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24663⟩⟩]⟩, (-1)⟩)

def event66731 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29592⟩⟩) (.sum [.result 66725 .summary, .result 66547 .summary])

def exact66732RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16795⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact66732RawTermsValid :
    exact66732RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66732 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29592⟩⟩) exact66732RawTerms .large 66728 (.finite 1292449485504936292352) (some (66731))

def event66733 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24598⟩⟩) 0 ⟨16630⟩ 3172

def event66734 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24598⟩⟩) (.authority (.programFamilyFact))

def event66735 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24598⟩⟩) (.finite 3720)

def event66736 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24600⟩⟩) 0 ⟨6689⟩ 5477

def event66737 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24600⟩⟩) 1 ⟨24598⟩ 66735

def event66738 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24600⟩⟩) (.authority (.operator))

def exact66739RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24600⟩⟩]⟩, (1)⟩]

theorem exact66739RawTermsValid :
    exact66739RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66739 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24600⟩⟩) exact66739RawTerms .large 66738 .exactZero (none)

def event66740 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29372⟩⟩) 0 ⟨24600⟩ 66739

def event66741 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29372⟩⟩) (.authority (.operator))

def exact66742RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29372⟩⟩]⟩, (1)⟩]

theorem exact66742RawTermsValid :
    exact66742RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66742 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29372⟩⟩) exact66742RawTerms (.finite 8192) 66741 .exactZero (none)

def event66743 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23287⟩⟩) 0 ⟨12756⟩ 3166

def event66744 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23287⟩⟩) (.authority (.programFamilyFact))

def event66745 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23287⟩⟩) (.finite 3720)

def event66746 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23288⟩⟩) 0 ⟨6689⟩ 5477

def event66747 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23288⟩⟩) 1 ⟨23287⟩ 66745

def event66748 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23288⟩⟩) (.authority (.operator))

def exact66749RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23288⟩⟩]⟩, (1)⟩]

theorem exact66749RawTermsValid :
    exact66749RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66749 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23288⟩⟩) exact66749RawTerms .large 66748 .exactZero (none)

def event66750 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25522⟩⟩) 0 ⟨23288⟩ 66749

def event66751 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25522⟩⟩) (.authority (.operator))

def exact66752RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25522⟩⟩]⟩, (1)⟩]

theorem exact66752RawTermsValid :
    exact66752RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66752 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25522⟩⟩) exact66752RawTerms (.finite 8192) 66751 .exactZero (none)

def event66753 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12757⟩⟩) 0 ⟨12754⟩ 3155

def event66754 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12757⟩⟩) 1 ⟨6566⟩ 65295

def event66755 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12757⟩⟩) (.tensor (.predecessor 0 66753 .coefficient) (.predecessor 1 66754 .coefficient) true false)

def event66756 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12757⟩⟩, .operator (⟨3155, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact66757RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact66757RawTermsValid :
    exact66757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66757 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12757⟩⟩) exact66757RawTerms .large 66755 .exactZero (none)

def event66758 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7205⟩⟩) 0 ⟨5533⟩ 65165

def event66759 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7205⟩⟩) 1 ⟨6787⟩ 7975

def event66760 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7205⟩⟩) (.product (.predecessor 0 66758 .coefficient) (.predecessor 1 66759 .coefficient) (⟨false, false, none, none, none⟩))

def event66761 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7205⟩⟩, .operator (⟨65165, 0⟩, ⟨7975, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (1)⟩)

def exact66762RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (1)⟩]

theorem exact66762RawTermsValid :
    exact66762RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66762 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7205⟩⟩) exact66762RawTerms .large 66760 .exactZero (none)

def event66763 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12758⟩⟩) 0 ⟨7205⟩ 66762

def event66764 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12758⟩⟩) 1 ⟨12757⟩ 66757

def event66765 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12758⟩⟩) (.sum [.predecessor 0 66763 .coefficient, .predecessor 1 66764 .coefficient])

def exact66766RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact66766RawTermsValid :
    exact66766RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66766 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12758⟩⟩) exact66766RawTerms .large 66765 .exactZero (none)

def event66767 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12759⟩⟩) 0 ⟨12758⟩ 66766

def event66768 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12759⟩⟩) 1 ⟨101⟩ 7967

def event66769 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12759⟩⟩) (.sum [.predecessor 0 66767 .coefficient, .predecessor 1 66768 .coefficient])

def event66770 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12759⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨101⟩⟩]⟩) [⟨.result 7967 .coefficient, false, none⟩])

def event66771 : Event := .survivorFold (1) 66770

def exact66772RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact66772RawTermsValid :
    exact66772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66772 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12759⟩⟩) exact66772RawTerms .large 66769 (.finite 26) (some (66770))

def event66773 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12760⟩⟩) 0 ⟨12759⟩ 66772

def event66774 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12760⟩⟩) 1 ⟨10025⟩ 3158

def event66775 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12760⟩⟩) (.product (.predecessor 0 66773 .coefficient) (.predecessor 1 66774 .coefficient) (⟨false, true, none, none, some 1⟩))

def event66776 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12760⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩], []⟩) [⟨.result 3158 .coefficient, true, some 1⟩])

def event66777 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12760⟩⟩) (.product (.result 66772 .summary) (.transfer 66776) (⟨false, false, none, none, none⟩))

def event66778 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12760⟩⟩, .operator (⟨66772, 1⟩, ⟨3158, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event66779 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12760⟩⟩, .operator (⟨66772, 0⟩, ⟨3158, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (1)⟩)

def exact66780RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact66780RawTermsValid :
    exact66780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66780 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12760⟩⟩) exact66780RawTerms .large 66775 (.finite 38272) (some (66777))

def event66781 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10026⟩⟩) 0 ⟨10025⟩ 3158

def event66782 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10026⟩⟩) 1 ⟨6566⟩ 65295

def event66783 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10026⟩⟩) (.tensor (.predecessor 0 66781 .coefficient) (.predecessor 1 66782 .coefficient) true false)

def event66784 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10026⟩⟩, .operator (⟨3158, 0⟩, ⟨65295, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact66785RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact66785RawTermsValid :
    exact66785RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66785 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10026⟩⟩) exact66785RawTerms .large 66783 .exactZero (none)

def event66786 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7185⟩⟩) 0 ⟨5533⟩ 65165

def event66787 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7185⟩⟩) 1 ⟨6767⟩ 8016

def event66788 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7185⟩⟩) (.product (.predecessor 0 66786 .coefficient) (.predecessor 1 66787 .coefficient) (⟨false, false, none, none, none⟩))

def event66789 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7185⟩⟩, .operator (⟨65165, 0⟩, ⟨8016, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩]⟩, (1)⟩)

def exact66790RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩]⟩, (1)⟩]

theorem exact66790RawTermsValid :
    exact66790RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66790 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7185⟩⟩) exact66790RawTerms .large 66788 .exactZero (none)

def event66791 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10027⟩⟩) 0 ⟨7185⟩ 66790

def event66792 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10027⟩⟩) 1 ⟨10026⟩ 66785

def event66793 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10027⟩⟩) (.sum [.predecessor 0 66791 .coefficient, .predecessor 1 66792 .coefficient])

def exact66794RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact66794RawTermsValid :
    exact66794RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66794 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10027⟩⟩) exact66794RawTerms .large 66793 .exactZero (none)

def event66795 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10028⟩⟩) 0 ⟨10027⟩ 66794

def event66796 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10028⟩⟩) 1 ⟨81⟩ 8008

def event66797 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10028⟩⟩) (.sum [.predecessor 0 66795 .coefficient, .predecessor 1 66796 .coefficient])

def event66798 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10028⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨81⟩⟩]⟩) [⟨.result 8008 .coefficient, false, none⟩])

def event66799 : Event := .survivorFold (1) 66798

def exact66800RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact66800RawTermsValid :
    exact66800RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66800 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10028⟩⟩) exact66800RawTerms .large 66797 (.finite 26) (some (66798))

def event66801 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10029⟩⟩) 0 ⟨10028⟩ 66800

def event66802 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10029⟩⟩) 1 ⟨7874⟩ 8005

def event66803 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10029⟩⟩) (.product (.predecessor 0 66801 .coefficient) (.predecessor 1 66802 .coefficient) (⟨false, false, none, none, none⟩))

def event66804 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10029⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩) [⟨.result 8001 .coefficient, false, none⟩])

def event66805 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10029⟩⟩) (.product (.result 66800 .summary) (.transfer 66804) (⟨false, false, none, none, none⟩))

def event66806 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10029⟩⟩, .operator (⟨66800, 1⟩, ⟨8005, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩, (-1)⟩)

def event66807 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨10029⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7873⟩⟩) ⟨6787⟩ 7975)

def event66808 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10029⟩⟩, .relation 66807 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (-1)⟩)

def event66809 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10029⟩⟩, .operator (⟨66800, 0⟩, ⟨8005, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩, (1)⟩)

def exact66810RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (-1)⟩]

theorem exact66810RawTermsValid :
    exact66810RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66810 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10029⟩⟩) exact66810RawTerms .large 66803 (.finite 95420416) (some (66805))

def event66811 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12761⟩⟩) 0 ⟨10029⟩ 66810

def event66812 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12761⟩⟩) 1 ⟨12760⟩ 66780

def event66813 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12761⟩⟩) (.sum [.predecessor 0 66811 .coefficient, .predecessor 1 66812 .coefficient])

def event66814 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12761⟩⟩, .operator (⟨66810, 1⟩, ⟨66780, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (1)⟩)

def event66815 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12761⟩⟩) (.sum [.result 66810 .summary, .result 66780 .summary])

def eventLeaf4160 : Array AnnotatedEvent := #[
  { event := event66560
    frameStart := 0 },
  { event := event66561
    frameStart := 66561 },
  { event := event66562
    frameStart := 66561 },
  { event := event66563
    frameStart := 66561 },
  { event := event66564
    frameStart := 66561 },
  { event := event66565
    frameStart := 66561 },
  { event := event66566
    frameStart := 66561 },
  { event := event66567
    frameStart := 66561 },
  { event := event66568
    frameStart := 66561 },
  { event := event66569
    frameStart := 66561 },
  { event := event66570
    frameStart := 66561 },
  { event := event66571
    frameStart := 66561 },
  { event := event66572
    frameStart := 66561 },
  { event := event66573
    frameStart := 66561 },
  { event := event66574
    frameStart := 66561 },
  { event := event66575
    frameStart := 66561 }
]

def eventLeaf4161 : Array AnnotatedEvent := #[
  { event := event66576
    frameStart := 66561 },
  { event := event66577
    frameStart := 66561 },
  { event := event66578
    frameStart := 66561 },
  { event := event66579
    frameStart := 66561 },
  { event := event66580
    frameStart := 66561 },
  { event := event66581
    frameStart := 66561 },
  { event := event66582
    frameStart := 66561 },
  { event := event66583
    frameStart := 66561 },
  { event := event66584
    frameStart := 66561 },
  { event := event66585
    frameStart := 66561 },
  { event := event66586
    frameStart := 66561 },
  { event := event66587
    frameStart := 66561 },
  { event := event66588
    frameStart := 66561 },
  { event := event66589
    frameStart := 66561 },
  { event := event66590
    frameStart := 66561 },
  { event := event66591
    frameStart := 66561 }
]

def eventLeaf4162 : Array AnnotatedEvent := #[
  { event := event66592
    frameStart := 66561 },
  { event := event66593
    frameStart := 66561 },
  { event := event66594
    frameStart := 66561 },
  { event := event66595
    frameStart := 66561 },
  { event := event66596
    frameStart := 66561 },
  { event := event66597
    frameStart := 66561 },
  { event := event66598
    frameStart := 66561 },
  { event := event66599
    frameStart := 66561 },
  { event := event66600
    frameStart := 66561 },
  { event := event66601
    frameStart := 66561 },
  { event := event66602
    frameStart := 66561 },
  { event := event66603
    frameStart := 66561 },
  { event := event66604
    frameStart := 66561 },
  { event := event66605
    frameStart := 66561 },
  { event := event66606
    frameStart := 66561 },
  { event := event66607
    frameStart := 66561 }
]

def eventLeaf4163 : Array AnnotatedEvent := #[
  { event := event66608
    frameStart := 66561 },
  { event := event66609
    frameStart := 66561 },
  { event := event66610
    frameStart := 66561 },
  { event := event66611
    frameStart := 66561 },
  { event := event66612
    frameStart := 66561 },
  { event := event66613
    frameStart := 66561 },
  { event := event66614
    frameStart := 66561 },
  { event := event66615
    frameStart := 66615 },
  { event := event66616
    frameStart := 66615 },
  { event := event66617
    frameStart := 66615 },
  { event := event66618
    frameStart := 66615 },
  { event := event66619
    frameStart := 66615 },
  { event := event66620
    frameStart := 66615 },
  { event := event66621
    frameStart := 66615 },
  { event := event66622
    frameStart := 66615 },
  { event := event66623
    frameStart := 66615 }
]

def eventLeaf4164 : Array AnnotatedEvent := #[
  { event := event66624
    frameStart := 66615 },
  { event := event66625
    frameStart := 66615 },
  { event := event66626
    frameStart := 66615 },
  { event := event66627
    frameStart := 66615 },
  { event := event66628
    frameStart := 66615 },
  { event := event66629
    frameStart := 66615 },
  { event := event66630
    frameStart := 66615 },
  { event := event66631
    frameStart := 66615 },
  { event := event66632
    frameStart := 66615 },
  { event := event66633
    frameStart := 66615 },
  { event := event66634
    frameStart := 66615 },
  { event := event66635
    frameStart := 66615 },
  { event := event66636
    frameStart := 66615 },
  { event := event66637
    frameStart := 66615 },
  { event := event66638
    frameStart := 66615 },
  { event := event66639
    frameStart := 66615 }
]

def eventLeaf4165 : Array AnnotatedEvent := #[
  { event := event66640
    frameStart := 66615 },
  { event := event66641
    frameStart := 66615 },
  { event := event66642
    frameStart := 66615 },
  { event := event66643
    frameStart := 66615 },
  { event := event66644
    frameStart := 66615 },
  { event := event66645
    frameStart := 66615 },
  { event := event66646
    frameStart := 66615 },
  { event := event66647
    frameStart := 66615 },
  { event := event66648
    frameStart := 66615 },
  { event := event66649
    frameStart := 66615 },
  { event := event66650
    frameStart := 66615 },
  { event := event66651
    frameStart := 66615 },
  { event := event66652
    frameStart := 66615 },
  { event := event66653
    frameStart := 66615 },
  { event := event66654
    frameStart := 66615 },
  { event := event66655
    frameStart := 66615 }
]

def eventLeaf4166 : Array AnnotatedEvent := #[
  { event := event66656
    frameStart := 66615 },
  { event := event66657
    frameStart := 66615 },
  { event := event66658
    frameStart := 66615 },
  { event := event66659
    frameStart := 66615 },
  { event := event66660
    frameStart := 66615 },
  { event := event66661
    frameStart := 66615 },
  { event := event66662
    frameStart := 66615 },
  { event := event66663
    frameStart := 66615 },
  { event := event66664
    frameStart := 66615 },
  { event := event66665
    frameStart := 66615 },
  { event := event66666
    frameStart := 66615 },
  { event := event66667
    frameStart := 66615 },
  { event := event66668
    frameStart := 66615 },
  { event := event66669
    frameStart := 66615 },
  { event := event66670
    frameStart := 66615 },
  { event := event66671
    frameStart := 66615 }
]

def eventLeaf4167 : Array AnnotatedEvent := #[
  { event := event66672
    frameStart := 66615 },
  { event := event66673
    frameStart := 66615 },
  { event := event66674
    frameStart := 66615 },
  { event := event66675
    frameStart := 66615 },
  { event := event66676
    frameStart := 66615 },
  { event := event66677
    frameStart := 66615 },
  { event := event66678
    frameStart := 66615 },
  { event := event66679
    frameStart := 66615 },
  { event := event66680
    frameStart := 66615 },
  { event := event66681
    frameStart := 66615 },
  { event := event66682
    frameStart := 66615 },
  { event := event66683
    frameStart := 66615 },
  { event := event66684
    frameStart := 66615 },
  { event := event66685
    frameStart := 66615 },
  { event := event66686
    frameStart := 66615 },
  { event := event66687
    frameStart := 66615 }
]

def eventLeaf4168 : Array AnnotatedEvent := #[
  { event := event66688
    frameStart := 66615 },
  { event := event66689
    frameStart := 66615 },
  { event := event66690
    frameStart := 66615 },
  { event := event66691
    frameStart := 66615 },
  { event := event66692
    frameStart := 66615 },
  { event := event66693
    frameStart := 66615 },
  { event := event66694
    frameStart := 66615 },
  { event := event66695
    frameStart := 66615 },
  { event := event66696
    frameStart := 66615 },
  { event := event66697
    frameStart := 66615 },
  { event := event66698
    frameStart := 66615 },
  { event := event66699
    frameStart := 66615 },
  { event := event66700
    frameStart := 66615 },
  { event := event66701
    frameStart := 66615 },
  { event := event66702
    frameStart := 66615 },
  { event := event66703
    frameStart := 66615 }
]

def eventLeaf4169 : Array AnnotatedEvent := #[
  { event := event66704
    frameStart := 66615 },
  { event := event66705
    frameStart := 66615 },
  { event := event66706
    frameStart := 66615 },
  { event := event66707
    frameStart := 66615 },
  { event := event66708
    frameStart := 66615 },
  { event := event66709
    frameStart := 66615 },
  { event := event66710
    frameStart := 66615 },
  { event := event66711
    frameStart := 66615 },
  { event := event66712
    frameStart := 66615 },
  { event := event66713
    frameStart := 66615 },
  { event := event66714
    frameStart := 66615 },
  { event := event66715
    frameStart := 66615 },
  { event := event66716
    frameStart := 66615 },
  { event := event66717
    frameStart := 66615 },
  { event := event66718
    frameStart := 66615 },
  { event := event66719
    frameStart := 0 }
]

def eventLeaf4170 : Array AnnotatedEvent := #[
  { event := event66720
    frameStart := 0 },
  { event := event66721
    frameStart := 0 },
  { event := event66722
    frameStart := 0 },
  { event := event66723
    frameStart := 0 },
  { event := event66724
    frameStart := 0 },
  { event := event66725
    frameStart := 0 },
  { event := event66726
    frameStart := 0 },
  { event := event66727
    frameStart := 0 },
  { event := event66728
    frameStart := 0 },
  { event := event66729
    frameStart := 0 },
  { event := event66730
    frameStart := 0 },
  { event := event66731
    frameStart := 0 },
  { event := event66732
    frameStart := 0 },
  { event := event66733
    frameStart := 0 },
  { event := event66734
    frameStart := 0 },
  { event := event66735
    frameStart := 0 }
]

def eventLeaf4171 : Array AnnotatedEvent := #[
  { event := event66736
    frameStart := 0 },
  { event := event66737
    frameStart := 0 },
  { event := event66738
    frameStart := 0 },
  { event := event66739
    frameStart := 0 },
  { event := event66740
    frameStart := 0 },
  { event := event66741
    frameStart := 0 },
  { event := event66742
    frameStart := 0 },
  { event := event66743
    frameStart := 0 },
  { event := event66744
    frameStart := 0 },
  { event := event66745
    frameStart := 0 },
  { event := event66746
    frameStart := 0 },
  { event := event66747
    frameStart := 0 },
  { event := event66748
    frameStart := 0 },
  { event := event66749
    frameStart := 0 },
  { event := event66750
    frameStart := 0 },
  { event := event66751
    frameStart := 0 }
]

def eventLeaf4172 : Array AnnotatedEvent := #[
  { event := event66752
    frameStart := 0 },
  { event := event66753
    frameStart := 0 },
  { event := event66754
    frameStart := 0 },
  { event := event66755
    frameStart := 0 },
  { event := event66756
    frameStart := 0 },
  { event := event66757
    frameStart := 0 },
  { event := event66758
    frameStart := 0 },
  { event := event66759
    frameStart := 0 },
  { event := event66760
    frameStart := 0 },
  { event := event66761
    frameStart := 0 },
  { event := event66762
    frameStart := 0 },
  { event := event66763
    frameStart := 0 },
  { event := event66764
    frameStart := 0 },
  { event := event66765
    frameStart := 0 },
  { event := event66766
    frameStart := 0 },
  { event := event66767
    frameStart := 0 }
]

def eventLeaf4173 : Array AnnotatedEvent := #[
  { event := event66768
    frameStart := 0 },
  { event := event66769
    frameStart := 0 },
  { event := event66770
    frameStart := 0 },
  { event := event66771
    frameStart := 0 },
  { event := event66772
    frameStart := 0 },
  { event := event66773
    frameStart := 0 },
  { event := event66774
    frameStart := 0 },
  { event := event66775
    frameStart := 0 },
  { event := event66776
    frameStart := 0 },
  { event := event66777
    frameStart := 0 },
  { event := event66778
    frameStart := 0 },
  { event := event66779
    frameStart := 0 },
  { event := event66780
    frameStart := 0 },
  { event := event66781
    frameStart := 0 },
  { event := event66782
    frameStart := 0 },
  { event := event66783
    frameStart := 0 }
]

def eventLeaf4174 : Array AnnotatedEvent := #[
  { event := event66784
    frameStart := 0 },
  { event := event66785
    frameStart := 0 },
  { event := event66786
    frameStart := 0 },
  { event := event66787
    frameStart := 0 },
  { event := event66788
    frameStart := 0 },
  { event := event66789
    frameStart := 0 },
  { event := event66790
    frameStart := 0 },
  { event := event66791
    frameStart := 0 },
  { event := event66792
    frameStart := 0 },
  { event := event66793
    frameStart := 0 },
  { event := event66794
    frameStart := 0 },
  { event := event66795
    frameStart := 0 },
  { event := event66796
    frameStart := 0 },
  { event := event66797
    frameStart := 0 },
  { event := event66798
    frameStart := 0 },
  { event := event66799
    frameStart := 0 }
]

def eventLeaf4175 : Array AnnotatedEvent := #[
  { event := event66800
    frameStart := 0 },
  { event := event66801
    frameStart := 0 },
  { event := event66802
    frameStart := 0 },
  { event := event66803
    frameStart := 0 },
  { event := event66804
    frameStart := 0 },
  { event := event66805
    frameStart := 0 },
  { event := event66806
    frameStart := 0 },
  { event := event66807
    frameStart := 0 },
  { event := event66808
    frameStart := 0 },
  { event := event66809
    frameStart := 0 },
  { event := event66810
    frameStart := 0 },
  { event := event66811
    frameStart := 0 },
  { event := event66812
    frameStart := 0 },
  { event := event66813
    frameStart := 0 },
  { event := event66814
    frameStart := 0 },
  { event := event66815
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events260
