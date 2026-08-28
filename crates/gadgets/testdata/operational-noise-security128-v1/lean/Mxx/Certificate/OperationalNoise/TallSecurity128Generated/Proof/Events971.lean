import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events971

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact248576RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨26392⟩⟩], [⟨.program ⟨257⟩, ⟨27542⟩⟩]⟩, (-1)⟩]

theorem exact248576RawTermsValid :
    exact248576RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248576 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28235⟩⟩) exact248576RawTerms .large 248569 (.finite 32191557518723128098041228165120) (some (248571))

def event248577 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27112⟩⟩) 0 ⟨26393⟩ 11492

def event248578 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27112⟩⟩) (.authority (.relationPreimageSource ⟨78⟩))

def exact248579RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27112⟩⟩]⟩, (1)⟩]

theorem exact248579RawTermsValid :
    exact248579RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248579 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27112⟩⟩) exact248579RawTerms (.finite 5647228698) 248578 .exactZero (none)

def event248580 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27114⟩⟩) 0 ⟨27112⟩ 248579

def event248581 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27114⟩⟩) 1 ⟨2370⟩ 4

def event248582 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27114⟩⟩) (.scale (.predecessor 0 248580 .coefficient) (.value (.predecessor 1 248581 .coefficient)))

def exact248583RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27112⟩⟩]⟩, (1)⟩]

theorem exact248583RawTermsValid :
    exact248583RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248583 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27114⟩⟩) exact248583RawTerms (.finite 5647228698) 248582 .exactZero (none)

def event248584 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27115⟩⟩) 0 ⟨5563⟩ 236870

def event248585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27115⟩⟩) 1 ⟨27114⟩ 248583

def event248586 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27115⟩⟩) (.product (.predecessor 0 248584 .coefficient) (.predecessor 1 248585 .coefficient) (⟨false, false, none, none, none⟩))

def event248587 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27115⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨27112⟩⟩]⟩) [⟨.result 248579 .coefficient, false, none⟩])

def event248588 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27115⟩⟩) (.product (.result 236870 .summary) (.transfer 248587) (⟨false, false, none, none, none⟩))

def event248589 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27115⟩⟩, .operator (⟨236870, 0⟩, ⟨248583, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27112⟩⟩]⟩, (1)⟩)

def event248590 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨27113⟩⟩)

def event248591 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event248592 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event248593 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.authority (.operator))

def event248594 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.finite 4)

def event248595 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event248596 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event248597 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event248598 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event248599 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 248598

def event248600 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 248596

def event248601 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 248599 .coefficient) (.value (.predecessor 1 248600 .coefficient)))

def event248602 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event248603 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 0 ⟨392⟩ 248602

def event248604 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 1 ⟨4740⟩ 248594

def event248605 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.sum [.predecessor 0 248603 .coefficient, .predecessor 1 248604 .coefficient])

def event248606 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.finite 655344)

def event248607 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 0 ⟨4742⟩ 248606

def event248608 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 1 ⟨5426⟩ 248592

def event248609 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.identity (.predecessor 1 248608 .coefficient))

def event248610 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.finite 655360)

def event248611 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26046⟩⟩) 0 ⟨5559⟩ 248610

def event248612 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26046⟩⟩) (.authority (.programFamilyFact))

def exact248613RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26046⟩⟩], []⟩, (1)⟩]

theorem exact248613RawTermsValid :
    exact248613RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248613 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26046⟩⟩) exact248613RawTerms (.finite 30) 248612 .exactZero (none)

def event248614 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12951⟩⟩) 0 ⟨5559⟩ 248610

def event248615 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12951⟩⟩) (.authority (.programFamilyFact))

def exact248616RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12951⟩⟩], []⟩, (1)⟩]

theorem exact248616RawTermsValid :
    exact248616RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248616 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12951⟩⟩) exact248616RawTerms (.finite 30) 248615 .exactZero (none)

def event248617 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26047⟩⟩) 0 ⟨12951⟩ 248616

def event248618 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26047⟩⟩) 1 ⟨26046⟩ 248613

def event248619 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26047⟩⟩) (.product (.predecessor 0 248617 .coefficient) (.predecessor 1 248618 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event248620 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26047⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], []⟩) [⟨.result 248616 .coefficient, true, some 1⟩, ⟨.result 248613 .coefficient, true, some 1⟩])

def event248621 : Event := .survivorFold (1) 248620

def exact248622RawTerms : List Term := []

theorem exact248622RawTermsValid :
    exact248622RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248622 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26047⟩⟩) exact248622RawTerms (.finite 900) 248619 (.finite 900) (some (248620))

def event248623 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26048⟩⟩) 0 ⟨26047⟩ 248622

def event248624 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26048⟩⟩) (.identity (.predecessor 0 248623 .coefficient))

def event248625 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26048⟩⟩) (.finite 900)

def event248626 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26392⟩⟩) 0 ⟨26048⟩ 248625

def event248627 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26392⟩⟩) (.authority (.programFamilyFact))

def exact248628RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26392⟩⟩], []⟩, (1)⟩]

theorem exact248628RawTermsValid :
    exact248628RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248628 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26392⟩⟩) exact248628RawTerms (.finite 30) 248627 .exactZero (none)

def event248629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26393⟩⟩) 0 ⟨26392⟩ 248628

def event248630 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26393⟩⟩) (.identity (.predecessor 0 248629 .coefficient))

def event248631 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26393⟩⟩) (.finite 30)

def event248632 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27112⟩⟩) 0 ⟨26393⟩ 248631

def event248633 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27112⟩⟩) (.authority (.relationPreimageSource ⟨78⟩))

def exact248634RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27112⟩⟩]⟩, (1)⟩]

theorem exact248634RawTermsValid :
    exact248634RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248634 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27112⟩⟩) exact248634RawTerms (.finite 5647228698) 248633 .exactZero (none)

def event248635 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact248636RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact248636RawTermsValid :
    exact248636RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248636 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact248636RawTerms .large 248635 .exactZero (none)

def event248637 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27113⟩⟩) 0 ⟨35⟩ 248636

def event248638 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27113⟩⟩) 1 ⟨27112⟩ 248634

def event248639 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27113⟩⟩) (.product (.predecessor 0 248637 .coefficient) (.predecessor 1 248638 .coefficient) (⟨false, false, none, none, none⟩))

def event248640 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27113⟩⟩, .operator (⟨248636, 0⟩, ⟨248634, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27112⟩⟩]⟩, (1)⟩)

def exact248641RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27112⟩⟩]⟩, (1)⟩]

theorem exact248641RawTermsValid :
    exact248641RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248641 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27113⟩⟩) exact248641RawTerms .large 248639 .exactZero (none)

def event248642 : Event := .preFoldPolynomial 248641 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27112⟩⟩]⟩, (1)⟩] .exactZero none

def exact248643RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27112⟩⟩]⟩, (1)⟩]

def event248643 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨27113⟩⟩) 248642 exact248643RawTerms .large 248639 .exactZero (none)

def event248644 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨28238⟩⟩)

def event248645 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event248646 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event248647 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.authority (.operator))

def event248648 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.finite 4)

def event248649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event248650 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event248651 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event248652 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event248653 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 248652

def event248654 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 248650

def event248655 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 248653 .coefficient) (.value (.predecessor 1 248654 .coefficient)))

def event248656 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event248657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 0 ⟨392⟩ 248656

def event248658 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 1 ⟨4740⟩ 248648

def event248659 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.sum [.predecessor 0 248657 .coefficient, .predecessor 1 248658 .coefficient])

def event248660 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.finite 655344)

def event248661 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 0 ⟨4742⟩ 248660

def event248662 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 1 ⟨5426⟩ 248646

def event248663 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.identity (.predecessor 1 248662 .coefficient))

def event248664 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.finite 655360)

def event248665 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26046⟩⟩) 0 ⟨5559⟩ 248664

def event248666 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26046⟩⟩) (.authority (.programFamilyFact))

def exact248667RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26046⟩⟩], []⟩, (1)⟩]

theorem exact248667RawTermsValid :
    exact248667RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248667 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26046⟩⟩) exact248667RawTerms (.finite 30) 248666 .exactZero (none)

def event248668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12951⟩⟩) 0 ⟨5559⟩ 248664

def event248669 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12951⟩⟩) (.authority (.programFamilyFact))

def exact248670RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12951⟩⟩], []⟩, (1)⟩]

theorem exact248670RawTermsValid :
    exact248670RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248670 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12951⟩⟩) exact248670RawTerms (.finite 30) 248669 .exactZero (none)

def event248671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26047⟩⟩) 0 ⟨12951⟩ 248670

def event248672 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26047⟩⟩) 1 ⟨26046⟩ 248667

def event248673 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26047⟩⟩) (.product (.predecessor 0 248671 .coefficient) (.predecessor 1 248672 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event248674 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26047⟩⟩, .operator (⟨248670, 0⟩, ⟨248667, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], []⟩, (1)⟩)

def exact248675RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12951⟩⟩, ⟨.program ⟨257⟩, ⟨26046⟩⟩], []⟩, (1)⟩]

theorem exact248675RawTermsValid :
    exact248675RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248675 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26047⟩⟩) exact248675RawTerms (.finite 900) 248673 .exactZero (none)

def event248676 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26048⟩⟩) 0 ⟨26047⟩ 248675

def event248677 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26048⟩⟩) (.identity (.predecessor 0 248676 .coefficient))

def event248678 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26048⟩⟩) (.finite 900)

def event248679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26392⟩⟩) 0 ⟨26048⟩ 248678

def event248680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26392⟩⟩) (.authority (.programFamilyFact))

def exact248681RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26392⟩⟩], []⟩, (1)⟩]

theorem exact248681RawTermsValid :
    exact248681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248681 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26392⟩⟩) exact248681RawTerms (.finite 30) 248680 .exactZero (none)

def event248682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26393⟩⟩) 0 ⟨26392⟩ 248681

def event248683 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26393⟩⟩) (.identity (.predecessor 0 248682 .coefficient))

def event248684 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26393⟩⟩) (.finite 30)

def event248685 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27541⟩⟩) 0 ⟨26393⟩ 248684

def event248686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27541⟩⟩) (.authority (.programFamilyFact))

def event248687 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27541⟩⟩) (.finite 3720)

def event248688 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event248689 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27542⟩⟩) 0 ⟨7177⟩ 248688

def event248690 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27542⟩⟩) 1 ⟨27541⟩ 248687

def event248691 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27542⟩⟩) (.authority (.operator))

def exact248692RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27542⟩⟩]⟩, (1)⟩]

theorem exact248692RawTermsValid :
    exact248692RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248692 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27542⟩⟩) exact248692RawTerms .large 248691 .exactZero (none)

def event248693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28233⟩⟩) 0 ⟨27542⟩ 248692

def event248694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28233⟩⟩) (.authority (.operator))

def exact248695RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨28233⟩⟩]⟩, (1)⟩]

theorem exact248695RawTermsValid :
    exact248695RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248695 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28233⟩⟩) exact248695RawTerms (.finite 8192) 248694 .exactZero (none)

def event248696 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event248697 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event248698 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27758⟩⟩) 0 ⟨26393⟩ 248684

def event248699 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27758⟩⟩) 1 ⟨136⟩ 248697

def event248700 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27758⟩⟩) (.sum [.predecessor 0 248698 .coefficient, .predecessor 1 248699 .coefficient])

def event248701 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27758⟩⟩) (.finite 30)

def event248702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27759⟩⟩) 0 ⟨27758⟩ 248701

def event248703 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27759⟩⟩) (.identity (.predecessor 0 248702 .coefficient))

def exact248704RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26392⟩⟩], []⟩, (1)⟩]

theorem exact248704RawTermsValid :
    exact248704RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248704 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27759⟩⟩) exact248704RawTerms (.finite 30) 248703 .exactZero (none)

def event248705 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact248706RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact248706RawTermsValid :
    exact248706RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248706 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact248706RawTerms .large 248705 .exactZero (none)

def event248707 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27760⟩⟩) 0 ⟨6908⟩ 248706

def event248708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27760⟩⟩) 1 ⟨27759⟩ 248704

def event248709 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27760⟩⟩) (.product (.predecessor 0 248707 .coefficient) (.predecessor 1 248708 .coefficient) (⟨false, false, none, none, none⟩))

def event248710 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27760⟩⟩, .operator (⟨248706, 0⟩, ⟨248704, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26392⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact248711RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26392⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact248711RawTermsValid :
    exact248711RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248711 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27760⟩⟩) exact248711RawTerms .large 248709 .exactZero (none)

def event248712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7189⟩⟩) 0 ⟨7177⟩ 248688

def event248713 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7189⟩⟩) (.authority (.operator))

def exact248714RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩]

theorem exact248714RawTermsValid :
    exact248714RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248714 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7189⟩⟩) exact248714RawTerms .large 248713 .exactZero (none)

def event248715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27761⟩⟩) 0 ⟨7189⟩ 248714

def event248716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27761⟩⟩) 1 ⟨27760⟩ 248711

def event248717 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27761⟩⟩) (.sum [.predecessor 0 248715 .coefficient, .predecessor 1 248716 .coefficient])

def exact248718RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26392⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact248718RawTermsValid :
    exact248718RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248718 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27761⟩⟩) exact248718RawTerms .large 248717 .exactZero (none)

def event248719 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28234⟩⟩) 0 ⟨27761⟩ 248718

def event248720 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28234⟩⟩) 1 ⟨28233⟩ 248695

def event248721 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28234⟩⟩) (.product (.predecessor 0 248719 .coefficient) (.predecessor 1 248720 .coefficient) (⟨false, false, none, none, none⟩))

def event248722 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28234⟩⟩, .operator (⟨248718, 0⟩, ⟨248695, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28233⟩⟩]⟩, (1)⟩)

def event248723 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28234⟩⟩, .operator (⟨248718, 1⟩, ⟨248695, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26392⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28233⟩⟩]⟩, (-1)⟩)

def event248724 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨28234⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨26392⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28233⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28233⟩⟩) ⟨27542⟩ 248692)

def event248725 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28234⟩⟩, .relation 248724 0, ⟨[⟨.program ⟨257⟩, ⟨26392⟩⟩], [⟨.program ⟨257⟩, ⟨27542⟩⟩]⟩, (-1)⟩)

def exact248726RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28233⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26392⟩⟩], [⟨.program ⟨257⟩, ⟨27542⟩⟩]⟩, (-1)⟩]

theorem exact248726RawTermsValid :
    exact248726RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248726 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28234⟩⟩) exact248726RawTerms .large 248721 .exactZero (none)

def event248727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26596⟩⟩) 0 ⟨26393⟩ 248684

def event248728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26596⟩⟩) (.authority (.programFamilyFact))

def exact248729RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26596⟩⟩], []⟩, (1)⟩]

theorem exact248729RawTermsValid :
    exact248729RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248729 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26596⟩⟩) exact248729RawTerms (.finite 30) 248728 .exactZero (none)

def event248730 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26598⟩⟩) 0 ⟨6908⟩ 248706

def event248731 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26598⟩⟩) 1 ⟨26596⟩ 248729

def event248732 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26598⟩⟩) (.product (.predecessor 0 248730 .coefficient) (.predecessor 1 248731 .coefficient) (⟨false, true, none, none, some 1⟩))

def event248733 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26598⟩⟩, .operator (⟨248706, 0⟩, ⟨248729, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26596⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact248734RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26596⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact248734RawTermsValid :
    exact248734RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248734 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26598⟩⟩) exact248734RawTerms .large 248732 .exactZero (none)

def event248735 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7217⟩⟩) 0 ⟨7177⟩ 248688

def event248736 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7217⟩⟩) (.authority (.operator))

def exact248737RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩]

theorem exact248737RawTermsValid :
    exact248737RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248737 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7217⟩⟩) exact248737RawTerms .large 248736 .exactZero (none)

def event248738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26599⟩⟩) 0 ⟨7217⟩ 248737

def event248739 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26599⟩⟩) 1 ⟨26598⟩ 248734

def event248740 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26599⟩⟩) (.sum [.predecessor 0 248738 .coefficient, .predecessor 1 248739 .coefficient])

def exact248741RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26596⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact248741RawTermsValid :
    exact248741RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248741 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26599⟩⟩) exact248741RawTerms .large 248740 .exactZero (none)

def event248742 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28238⟩⟩) 0 ⟨26599⟩ 248741

def event248743 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28238⟩⟩) 1 ⟨28234⟩ 248726

def event248744 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28238⟩⟩) (.sum [.predecessor 0 248742 .coefficient, .predecessor 1 248743 .coefficient])

def exact248745RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28233⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26392⟩⟩], [⟨.program ⟨257⟩, ⟨27542⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26596⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact248745RawTermsValid :
    exact248745RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248745 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28238⟩⟩) exact248745RawTerms .large 248744 .exactZero (none)

def event248746 : Event := .preFoldPolynomial 248745 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28233⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26392⟩⟩], [⟨.program ⟨257⟩, ⟨27542⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26596⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact248747RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28233⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26392⟩⟩], [⟨.program ⟨257⟩, ⟨27542⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26596⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event248747 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨28238⟩⟩) 248746 exact248747RawTerms .large 248744 .exactZero (none)

def event248748 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨26393⟩⟩) ⟨⟨96⟩, ⟨78⟩, ⟨135⟩⟩ ⟨248590, 248748⟩

def event248749 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨27115⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27112⟩⟩]⟩) (1) 0 2 (.universal 248748 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27112⟩⟩]⟩) (none) 248747)

def event248750 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27115⟩⟩, .relation 248749 1, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩)

def event248751 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27115⟩⟩, .relation 248749 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28233⟩⟩]⟩, (-1)⟩)

def event248752 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27115⟩⟩, .relation 248749 2, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨26392⟩⟩], [⟨.program ⟨257⟩, ⟨27542⟩⟩]⟩, (1)⟩)

def event248753 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27115⟩⟩, .relation 248749 3, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨26596⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact248754RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28233⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨26392⟩⟩], [⟨.program ⟨257⟩, ⟨27542⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨26596⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact248754RawTermsValid :
    exact248754RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248754 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27115⟩⟩) exact248754RawTerms .large 248586 (.finite 202072841853861888) (some (248588))

def event248755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28236⟩⟩) 0 ⟨27115⟩ 248754

def event248756 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28236⟩⟩) 1 ⟨28235⟩ 248576

def event248757 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28236⟩⟩) (.sum [.predecessor 0 248755 .coefficient, .predecessor 1 248756 .coefficient])

def event248758 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28236⟩⟩, .operator (⟨248754, 0⟩, ⟨248576, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28233⟩⟩]⟩, (1)⟩)

def event248759 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28236⟩⟩, .operator (⟨248754, 2⟩, ⟨248576, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨26392⟩⟩], [⟨.program ⟨257⟩, ⟨27542⟩⟩]⟩, (-1)⟩)

def event248760 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28236⟩⟩) (.sum [.result 248754 .summary, .result 248576 .summary])

def exact248761RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨26596⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact248761RawTermsValid :
    exact248761RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248761 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28236⟩⟩) exact248761RawTerms .large 248757 (.finite 32191557518723330170883082027008) (some (248760))

def event248762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28237⟩⟩) 0 ⟨28236⟩ 248761

def event248763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28237⟩⟩) 1 ⟨7170⟩ 15682

def event248764 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28237⟩⟩) (.product (.predecessor 0 248762 .coefficient) (.predecessor 1 248763 .coefficient) (⟨false, false, none, none, none⟩))

def event248765 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28237⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩) [⟨.result 15678 .coefficient, false, none⟩])

def event248766 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28237⟩⟩) (.product (.result 248761 .summary) (.transfer 248765) (⟨false, false, none, none, none⟩))

def event248767 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28237⟩⟩, .operator (⟨248761, 0⟩, ⟨15682, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩)

def event248768 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28237⟩⟩, .operator (⟨248761, 1⟩, ⟨15682, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨26596⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (-1)⟩)

def event248769 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨28237⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨26596⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7169⟩⟩) ⟨7050⟩ 15675)

def event248770 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28237⟩⟩, .relation 248769 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26596⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact248771RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26596⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact248771RawTermsValid :
    exact248771RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248771 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28237⟩⟩) exact248771RawTerms .large 248764 (.finite 345654216875549026890382321864211871825920) (some (248766))

def event248772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68663⟩⟩) 0 ⟨7177⟩ 15500

def event248773 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68663⟩⟩) 1 ⟨68662⟩ 240628

def event248774 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68663⟩⟩) (.authority (.operator))

def exact248775RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68663⟩⟩]⟩, (1)⟩]

theorem exact248775RawTermsValid :
    exact248775RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248775 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68663⟩⟩) exact248775RawTerms .large 248774 .exactZero (none)

def event248776 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70004⟩⟩) 0 ⟨68663⟩ 248775

def event248777 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70004⟩⟩) (.authority (.operator))

def exact248778RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨70004⟩⟩]⟩, (1)⟩]

theorem exact248778RawTermsValid :
    exact248778RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248778 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70004⟩⟩) exact248778RawTerms (.finite 8192) 248777 .exactZero (none)

def event248779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70006⟩⟩) 0 ⟨69220⟩ 240912

def event248780 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70006⟩⟩) 1 ⟨70004⟩ 248778

def event248781 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70006⟩⟩) (.product (.predecessor 0 248779 .coefficient) (.predecessor 1 248780 .coefficient) (⟨false, false, none, none, none⟩))

def event248782 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70006⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨70004⟩⟩]⟩) [⟨.result 248778 .coefficient, false, none⟩])

def event248783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70006⟩⟩) (.product (.result 240912 .summary) (.transfer 248782) (⟨false, false, none, none, none⟩))

def event248784 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70006⟩⟩, .operator (⟨240912, 0⟩, ⟨248778, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70004⟩⟩]⟩, (1)⟩)

def event248785 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70006⟩⟩, .operator (⟨240912, 1⟩, ⟨248778, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨65772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70004⟩⟩]⟩, (-1)⟩)

def event248786 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70006⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨65772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70004⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70004⟩⟩) ⟨68663⟩ 248775)

def event248787 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70006⟩⟩, .relation 248786 0, ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨65772⟩⟩], [⟨.program ⟨257⟩, ⟨68663⟩⟩]⟩, (-1)⟩)

def exact248788RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70004⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨65772⟩⟩], [⟨.program ⟨257⟩, ⟨68663⟩⟩]⟩, (-1)⟩]

theorem exact248788RawTermsValid :
    exact248788RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248788 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70006⟩⟩) exact248788RawTerms .large 248781 (.finite 32191361068277440720800338411520) (some (248783))

def event248789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68033⟩⟩) 0 ⟨65773⟩ 11515

def event248790 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68033⟩⟩) (.authority (.relationPreimageSource ⟨75⟩))

def exact248791RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68033⟩⟩]⟩, (1)⟩]

theorem exact248791RawTermsValid :
    exact248791RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248791 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68033⟩⟩) exact248791RawTerms (.finite 5647228698) 248790 .exactZero (none)

def event248792 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68035⟩⟩) 0 ⟨68033⟩ 248791

def event248793 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68035⟩⟩) 1 ⟨2370⟩ 4

def event248794 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68035⟩⟩) (.scale (.predecessor 0 248792 .coefficient) (.value (.predecessor 1 248793 .coefficient)))

def exact248795RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68033⟩⟩]⟩, (1)⟩]

theorem exact248795RawTermsValid :
    exact248795RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248795 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68035⟩⟩) exact248795RawTerms (.finite 5647228698) 248794 .exactZero (none)

def event248796 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68036⟩⟩) 0 ⟨5563⟩ 236870

def event248797 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68036⟩⟩) 1 ⟨68035⟩ 248795

def event248798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68036⟩⟩) (.product (.predecessor 0 248796 .coefficient) (.predecessor 1 248797 .coefficient) (⟨false, false, none, none, none⟩))

def event248799 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68036⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨68033⟩⟩]⟩) [⟨.result 248791 .coefficient, false, none⟩])

def event248800 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68036⟩⟩) (.product (.result 236870 .summary) (.transfer 248799) (⟨false, false, none, none, none⟩))

def event248801 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68036⟩⟩, .operator (⟨236870, 0⟩, ⟨248795, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68033⟩⟩]⟩, (1)⟩)

def event248802 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨68034⟩⟩)

def event248803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event248804 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event248805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.authority (.operator))

def event248806 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4740⟩⟩) (.finite 4)

def event248807 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event248808 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event248809 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event248810 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event248811 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 248810

def event248812 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 248808

def event248813 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 248811 .coefficient) (.value (.predecessor 1 248812 .coefficient)))

def event248814 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event248815 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 0 ⟨392⟩ 248814

def event248816 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4742⟩⟩) 1 ⟨4740⟩ 248806

def event248817 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.sum [.predecessor 0 248815 .coefficient, .predecessor 1 248816 .coefficient])

def event248818 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4742⟩⟩) (.finite 655344)

def event248819 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 0 ⟨4742⟩ 248818

def event248820 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5559⟩⟩) 1 ⟨5426⟩ 248804

def event248821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.identity (.predecessor 1 248820 .coefficient))

def event248822 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5559⟩⟩) (.finite 655360)

def event248823 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25706⟩⟩) 0 ⟨5559⟩ 248822

def event248824 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25706⟩⟩) (.authority (.programFamilyFact))

def exact248825RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25706⟩⟩], []⟩, (1)⟩]

theorem exact248825RawTermsValid :
    exact248825RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248825 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25706⟩⟩) exact248825RawTerms (.finite 28) 248824 .exactZero (none)

def event248826 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65391⟩⟩) 0 ⟨5559⟩ 248822

def event248827 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65391⟩⟩) (.authority (.programFamilyFact))

def exact248828RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65391⟩⟩], []⟩, (1)⟩]

theorem exact248828RawTermsValid :
    exact248828RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event248828 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65391⟩⟩) exact248828RawTerms (.finite 28) 248827 .exactZero (none)

def event248829 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65392⟩⟩) 0 ⟨65391⟩ 248828

def event248830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65392⟩⟩) 1 ⟨25706⟩ 248825

def event248831 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65392⟩⟩) (.product (.predecessor 0 248829 .coefficient) (.predecessor 1 248830 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def eventLeaf15536 : Array AnnotatedEvent := #[
  { event := event248576
    frameStart := 0 },
  { event := event248577
    frameStart := 0 },
  { event := event248578
    frameStart := 0 },
  { event := event248579
    frameStart := 0 },
  { event := event248580
    frameStart := 0 },
  { event := event248581
    frameStart := 0 },
  { event := event248582
    frameStart := 0 },
  { event := event248583
    frameStart := 0 },
  { event := event248584
    frameStart := 0 },
  { event := event248585
    frameStart := 0 },
  { event := event248586
    frameStart := 0 },
  { event := event248587
    frameStart := 0 },
  { event := event248588
    frameStart := 0 },
  { event := event248589
    frameStart := 0 },
  { event := event248590
    frameStart := 248590 },
  { event := event248591
    frameStart := 248590 }
]

def eventLeaf15537 : Array AnnotatedEvent := #[
  { event := event248592
    frameStart := 248590 },
  { event := event248593
    frameStart := 248590 },
  { event := event248594
    frameStart := 248590 },
  { event := event248595
    frameStart := 248590 },
  { event := event248596
    frameStart := 248590 },
  { event := event248597
    frameStart := 248590 },
  { event := event248598
    frameStart := 248590 },
  { event := event248599
    frameStart := 248590 },
  { event := event248600
    frameStart := 248590 },
  { event := event248601
    frameStart := 248590 },
  { event := event248602
    frameStart := 248590 },
  { event := event248603
    frameStart := 248590 },
  { event := event248604
    frameStart := 248590 },
  { event := event248605
    frameStart := 248590 },
  { event := event248606
    frameStart := 248590 },
  { event := event248607
    frameStart := 248590 }
]

def eventLeaf15538 : Array AnnotatedEvent := #[
  { event := event248608
    frameStart := 248590 },
  { event := event248609
    frameStart := 248590 },
  { event := event248610
    frameStart := 248590 },
  { event := event248611
    frameStart := 248590 },
  { event := event248612
    frameStart := 248590 },
  { event := event248613
    frameStart := 248590 },
  { event := event248614
    frameStart := 248590 },
  { event := event248615
    frameStart := 248590 },
  { event := event248616
    frameStart := 248590 },
  { event := event248617
    frameStart := 248590 },
  { event := event248618
    frameStart := 248590 },
  { event := event248619
    frameStart := 248590 },
  { event := event248620
    frameStart := 248590 },
  { event := event248621
    frameStart := 248590 },
  { event := event248622
    frameStart := 248590 },
  { event := event248623
    frameStart := 248590 }
]

def eventLeaf15539 : Array AnnotatedEvent := #[
  { event := event248624
    frameStart := 248590 },
  { event := event248625
    frameStart := 248590 },
  { event := event248626
    frameStart := 248590 },
  { event := event248627
    frameStart := 248590 },
  { event := event248628
    frameStart := 248590 },
  { event := event248629
    frameStart := 248590 },
  { event := event248630
    frameStart := 248590 },
  { event := event248631
    frameStart := 248590 },
  { event := event248632
    frameStart := 248590 },
  { event := event248633
    frameStart := 248590 },
  { event := event248634
    frameStart := 248590 },
  { event := event248635
    frameStart := 248590 },
  { event := event248636
    frameStart := 248590 },
  { event := event248637
    frameStart := 248590 },
  { event := event248638
    frameStart := 248590 },
  { event := event248639
    frameStart := 248590 }
]

def eventLeaf15540 : Array AnnotatedEvent := #[
  { event := event248640
    frameStart := 248590 },
  { event := event248641
    frameStart := 248590 },
  { event := event248642
    frameStart := 248590 },
  { event := event248643
    frameStart := 248590 },
  { event := event248644
    frameStart := 248644 },
  { event := event248645
    frameStart := 248644 },
  { event := event248646
    frameStart := 248644 },
  { event := event248647
    frameStart := 248644 },
  { event := event248648
    frameStart := 248644 },
  { event := event248649
    frameStart := 248644 },
  { event := event248650
    frameStart := 248644 },
  { event := event248651
    frameStart := 248644 },
  { event := event248652
    frameStart := 248644 },
  { event := event248653
    frameStart := 248644 },
  { event := event248654
    frameStart := 248644 },
  { event := event248655
    frameStart := 248644 }
]

def eventLeaf15541 : Array AnnotatedEvent := #[
  { event := event248656
    frameStart := 248644 },
  { event := event248657
    frameStart := 248644 },
  { event := event248658
    frameStart := 248644 },
  { event := event248659
    frameStart := 248644 },
  { event := event248660
    frameStart := 248644 },
  { event := event248661
    frameStart := 248644 },
  { event := event248662
    frameStart := 248644 },
  { event := event248663
    frameStart := 248644 },
  { event := event248664
    frameStart := 248644 },
  { event := event248665
    frameStart := 248644 },
  { event := event248666
    frameStart := 248644 },
  { event := event248667
    frameStart := 248644 },
  { event := event248668
    frameStart := 248644 },
  { event := event248669
    frameStart := 248644 },
  { event := event248670
    frameStart := 248644 },
  { event := event248671
    frameStart := 248644 }
]

def eventLeaf15542 : Array AnnotatedEvent := #[
  { event := event248672
    frameStart := 248644 },
  { event := event248673
    frameStart := 248644 },
  { event := event248674
    frameStart := 248644 },
  { event := event248675
    frameStart := 248644 },
  { event := event248676
    frameStart := 248644 },
  { event := event248677
    frameStart := 248644 },
  { event := event248678
    frameStart := 248644 },
  { event := event248679
    frameStart := 248644 },
  { event := event248680
    frameStart := 248644 },
  { event := event248681
    frameStart := 248644 },
  { event := event248682
    frameStart := 248644 },
  { event := event248683
    frameStart := 248644 },
  { event := event248684
    frameStart := 248644 },
  { event := event248685
    frameStart := 248644 },
  { event := event248686
    frameStart := 248644 },
  { event := event248687
    frameStart := 248644 }
]

def eventLeaf15543 : Array AnnotatedEvent := #[
  { event := event248688
    frameStart := 248644 },
  { event := event248689
    frameStart := 248644 },
  { event := event248690
    frameStart := 248644 },
  { event := event248691
    frameStart := 248644 },
  { event := event248692
    frameStart := 248644 },
  { event := event248693
    frameStart := 248644 },
  { event := event248694
    frameStart := 248644 },
  { event := event248695
    frameStart := 248644 },
  { event := event248696
    frameStart := 248644 },
  { event := event248697
    frameStart := 248644 },
  { event := event248698
    frameStart := 248644 },
  { event := event248699
    frameStart := 248644 },
  { event := event248700
    frameStart := 248644 },
  { event := event248701
    frameStart := 248644 },
  { event := event248702
    frameStart := 248644 },
  { event := event248703
    frameStart := 248644 }
]

def eventLeaf15544 : Array AnnotatedEvent := #[
  { event := event248704
    frameStart := 248644 },
  { event := event248705
    frameStart := 248644 },
  { event := event248706
    frameStart := 248644 },
  { event := event248707
    frameStart := 248644 },
  { event := event248708
    frameStart := 248644 },
  { event := event248709
    frameStart := 248644 },
  { event := event248710
    frameStart := 248644 },
  { event := event248711
    frameStart := 248644 },
  { event := event248712
    frameStart := 248644 },
  { event := event248713
    frameStart := 248644 },
  { event := event248714
    frameStart := 248644 },
  { event := event248715
    frameStart := 248644 },
  { event := event248716
    frameStart := 248644 },
  { event := event248717
    frameStart := 248644 },
  { event := event248718
    frameStart := 248644 },
  { event := event248719
    frameStart := 248644 }
]

def eventLeaf15545 : Array AnnotatedEvent := #[
  { event := event248720
    frameStart := 248644 },
  { event := event248721
    frameStart := 248644 },
  { event := event248722
    frameStart := 248644 },
  { event := event248723
    frameStart := 248644 },
  { event := event248724
    frameStart := 248644 },
  { event := event248725
    frameStart := 248644 },
  { event := event248726
    frameStart := 248644 },
  { event := event248727
    frameStart := 248644 },
  { event := event248728
    frameStart := 248644 },
  { event := event248729
    frameStart := 248644 },
  { event := event248730
    frameStart := 248644 },
  { event := event248731
    frameStart := 248644 },
  { event := event248732
    frameStart := 248644 },
  { event := event248733
    frameStart := 248644 },
  { event := event248734
    frameStart := 248644 },
  { event := event248735
    frameStart := 248644 }
]

def eventLeaf15546 : Array AnnotatedEvent := #[
  { event := event248736
    frameStart := 248644 },
  { event := event248737
    frameStart := 248644 },
  { event := event248738
    frameStart := 248644 },
  { event := event248739
    frameStart := 248644 },
  { event := event248740
    frameStart := 248644 },
  { event := event248741
    frameStart := 248644 },
  { event := event248742
    frameStart := 248644 },
  { event := event248743
    frameStart := 248644 },
  { event := event248744
    frameStart := 248644 },
  { event := event248745
    frameStart := 248644 },
  { event := event248746
    frameStart := 248644 },
  { event := event248747
    frameStart := 248644 },
  { event := event248748
    frameStart := 0 },
  { event := event248749
    frameStart := 0 },
  { event := event248750
    frameStart := 0 },
  { event := event248751
    frameStart := 0 }
]

def eventLeaf15547 : Array AnnotatedEvent := #[
  { event := event248752
    frameStart := 0 },
  { event := event248753
    frameStart := 0 },
  { event := event248754
    frameStart := 0 },
  { event := event248755
    frameStart := 0 },
  { event := event248756
    frameStart := 0 },
  { event := event248757
    frameStart := 0 },
  { event := event248758
    frameStart := 0 },
  { event := event248759
    frameStart := 0 },
  { event := event248760
    frameStart := 0 },
  { event := event248761
    frameStart := 0 },
  { event := event248762
    frameStart := 0 },
  { event := event248763
    frameStart := 0 },
  { event := event248764
    frameStart := 0 },
  { event := event248765
    frameStart := 0 },
  { event := event248766
    frameStart := 0 },
  { event := event248767
    frameStart := 0 }
]

def eventLeaf15548 : Array AnnotatedEvent := #[
  { event := event248768
    frameStart := 0 },
  { event := event248769
    frameStart := 0 },
  { event := event248770
    frameStart := 0 },
  { event := event248771
    frameStart := 0 },
  { event := event248772
    frameStart := 0 },
  { event := event248773
    frameStart := 0 },
  { event := event248774
    frameStart := 0 },
  { event := event248775
    frameStart := 0 },
  { event := event248776
    frameStart := 0 },
  { event := event248777
    frameStart := 0 },
  { event := event248778
    frameStart := 0 },
  { event := event248779
    frameStart := 0 },
  { event := event248780
    frameStart := 0 },
  { event := event248781
    frameStart := 0 },
  { event := event248782
    frameStart := 0 },
  { event := event248783
    frameStart := 0 }
]

def eventLeaf15549 : Array AnnotatedEvent := #[
  { event := event248784
    frameStart := 0 },
  { event := event248785
    frameStart := 0 },
  { event := event248786
    frameStart := 0 },
  { event := event248787
    frameStart := 0 },
  { event := event248788
    frameStart := 0 },
  { event := event248789
    frameStart := 0 },
  { event := event248790
    frameStart := 0 },
  { event := event248791
    frameStart := 0 },
  { event := event248792
    frameStart := 0 },
  { event := event248793
    frameStart := 0 },
  { event := event248794
    frameStart := 0 },
  { event := event248795
    frameStart := 0 },
  { event := event248796
    frameStart := 0 },
  { event := event248797
    frameStart := 0 },
  { event := event248798
    frameStart := 0 },
  { event := event248799
    frameStart := 0 }
]

def eventLeaf15550 : Array AnnotatedEvent := #[
  { event := event248800
    frameStart := 0 },
  { event := event248801
    frameStart := 0 },
  { event := event248802
    frameStart := 248802 },
  { event := event248803
    frameStart := 248802 },
  { event := event248804
    frameStart := 248802 },
  { event := event248805
    frameStart := 248802 },
  { event := event248806
    frameStart := 248802 },
  { event := event248807
    frameStart := 248802 },
  { event := event248808
    frameStart := 248802 },
  { event := event248809
    frameStart := 248802 },
  { event := event248810
    frameStart := 248802 },
  { event := event248811
    frameStart := 248802 },
  { event := event248812
    frameStart := 248802 },
  { event := event248813
    frameStart := 248802 },
  { event := event248814
    frameStart := 248802 },
  { event := event248815
    frameStart := 248802 }
]

def eventLeaf15551 : Array AnnotatedEvent := #[
  { event := event248816
    frameStart := 248802 },
  { event := event248817
    frameStart := 248802 },
  { event := event248818
    frameStart := 248802 },
  { event := event248819
    frameStart := 248802 },
  { event := event248820
    frameStart := 248802 },
  { event := event248821
    frameStart := 248802 },
  { event := event248822
    frameStart := 248802 },
  { event := event248823
    frameStart := 248802 },
  { event := event248824
    frameStart := 248802 },
  { event := event248825
    frameStart := 248802 },
  { event := event248826
    frameStart := 248802 },
  { event := event248827
    frameStart := 248802 },
  { event := event248828
    frameStart := 248802 },
  { event := event248829
    frameStart := 248802 },
  { event := event248830
    frameStart := 248802 },
  { event := event248831
    frameStart := 248802 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events971
