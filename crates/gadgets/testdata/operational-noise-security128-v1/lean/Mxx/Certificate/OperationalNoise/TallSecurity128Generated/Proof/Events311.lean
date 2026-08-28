import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events311

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event79616 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26456⟩⟩) (.authority (.programFamilyFact))

def exact79617RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], []⟩, (1)⟩]

theorem exact79617RawTermsValid :
    exact79617RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79617 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26456⟩⟩) exact79617RawTerms (.finite 30) 79616 .exactZero (none)

def event79618 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26457⟩⟩) 0 ⟨26456⟩ 79617

def event79619 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26457⟩⟩) (.identity (.predecessor 0 79618 .coefficient))

def event79620 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26457⟩⟩) (.finite 30)

def event79621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27276⟩⟩) 0 ⟨26457⟩ 79620

def event79622 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27276⟩⟩) (.authority (.relationPreimageSource ⟨79⟩))

def exact79623RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩, (1)⟩]

theorem exact79623RawTermsValid :
    exact79623RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79623 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27276⟩⟩) exact79623RawTerms (.finite 5647228698) 79622 .exactZero (none)

def event79624 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact79625RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact79625RawTermsValid :
    exact79625RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79625 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact79625RawTerms .large 79624 .exactZero (none)

def event79626 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27277⟩⟩) 0 ⟨35⟩ 79625

def event79627 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27277⟩⟩) 1 ⟨27276⟩ 79623

def event79628 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27277⟩⟩) (.product (.predecessor 0 79626 .coefficient) (.predecessor 1 79627 .coefficient) (⟨false, false, none, none, none⟩))

def event79629 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27277⟩⟩, .operator (⟨79625, 0⟩, ⟨79623, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩, (1)⟩)

def exact79630RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩, (1)⟩]

theorem exact79630RawTermsValid :
    exact79630RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79630 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27277⟩⟩) exact79630RawTerms .large 79628 .exactZero (none)

def event79631 : Event := .preFoldPolynomial 79630 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩, (1)⟩] .exactZero none

def exact79632RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩, (1)⟩]

def event79632 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨27277⟩⟩) 79631 exact79632RawTerms .large 79628 .exactZero (none)

def event79633 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨28443⟩⟩)

def event79634 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event79635 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event79636 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event79637 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event79638 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event79639 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event79640 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event79641 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event79642 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 79641

def event79643 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 79639

def event79644 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 79642 .coefficient) (.value (.predecessor 1 79643 .coefficient)))

def event79645 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event79646 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 79645

def event79647 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 79637

def event79648 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 79646 .coefficient, .predecessor 1 79647 .coefficient])

def event79649 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event79650 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 79649

def event79651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 79635

def event79652 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 79651 .coefficient))

def event79653 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event79654 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26238⟩⟩) 0 ⟨10325⟩ 79653

def event79655 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26238⟩⟩) (.authority (.programFamilyFact))

def exact79656RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26238⟩⟩], []⟩, (1)⟩]

theorem exact79656RawTermsValid :
    exact79656RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79656 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26238⟩⟩) exact79656RawTerms (.finite 30) 79655 .exactZero (none)

def event79657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13071⟩⟩) 0 ⟨10325⟩ 79653

def event79658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13071⟩⟩) (.authority (.programFamilyFact))

def exact79659RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩], []⟩, (1)⟩]

theorem exact79659RawTermsValid :
    exact79659RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79659 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13071⟩⟩) exact79659RawTerms (.finite 30) 79658 .exactZero (none)

def event79660 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26239⟩⟩) 0 ⟨13071⟩ 79659

def event79661 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26239⟩⟩) 1 ⟨26238⟩ 79656

def event79662 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26239⟩⟩) (.product (.predecessor 0 79660 .coefficient) (.predecessor 1 79661 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event79663 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26239⟩⟩, .operator (⟨79659, 0⟩, ⟨79656, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], []⟩, (1)⟩)

def exact79664RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13071⟩⟩, ⟨.program ⟨257⟩, ⟨26238⟩⟩], []⟩, (1)⟩]

theorem exact79664RawTermsValid :
    exact79664RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79664 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26239⟩⟩) exact79664RawTerms (.finite 900) 79662 .exactZero (none)

def event79665 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26240⟩⟩) 0 ⟨26239⟩ 79664

def event79666 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26240⟩⟩) (.identity (.predecessor 0 79665 .coefficient))

def event79667 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26240⟩⟩) (.finite 900)

def event79668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26456⟩⟩) 0 ⟨26240⟩ 79667

def event79669 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26456⟩⟩) (.authority (.programFamilyFact))

def exact79670RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], []⟩, (1)⟩]

theorem exact79670RawTermsValid :
    exact79670RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79670 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26456⟩⟩) exact79670RawTerms (.finite 30) 79669 .exactZero (none)

def event79671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26457⟩⟩) 0 ⟨26456⟩ 79670

def event79672 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26457⟩⟩) (.identity (.predecessor 0 79671 .coefficient))

def event79673 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26457⟩⟩) (.finite 30)

def event79674 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27613⟩⟩) 0 ⟨26457⟩ 79673

def event79675 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27613⟩⟩) (.authority (.programFamilyFact))

def event79676 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27613⟩⟩) (.finite 3720)

def event79677 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event79678 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27615⟩⟩) 0 ⟨7177⟩ 79677

def event79679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27615⟩⟩) 1 ⟨27613⟩ 79676

def event79680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27615⟩⟩) (.authority (.operator))

def exact79681RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨27615⟩⟩]⟩, (1)⟩]

theorem exact79681RawTermsValid :
    exact79681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79681 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27615⟩⟩) exact79681RawTerms .large 79680 .exactZero (none)

def event79682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28439⟩⟩) 0 ⟨27615⟩ 79681

def event79683 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28439⟩⟩) (.authority (.operator))

def exact79684RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩, (1)⟩]

theorem exact79684RawTermsValid :
    exact79684RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79684 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28439⟩⟩) exact79684RawTerms (.finite 8192) 79683 .exactZero (none)

def event79685 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event79686 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event79687 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27790⟩⟩) 0 ⟨26457⟩ 79673

def event79688 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27790⟩⟩) 1 ⟨136⟩ 79686

def event79689 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27790⟩⟩) (.sum [.predecessor 0 79687 .coefficient, .predecessor 1 79688 .coefficient])

def event79690 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨27790⟩⟩) (.finite 30)

def event79691 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27791⟩⟩) 0 ⟨27790⟩ 79690

def event79692 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27791⟩⟩) (.identity (.predecessor 0 79691 .coefficient))

def exact79693RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], []⟩, (1)⟩]

theorem exact79693RawTermsValid :
    exact79693RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79693 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27791⟩⟩) exact79693RawTerms (.finite 30) 79692 .exactZero (none)

def event79694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact79695RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact79695RawTermsValid :
    exact79695RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79695 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact79695RawTerms .large 79694 .exactZero (none)

def event79696 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27792⟩⟩) 0 ⟨6908⟩ 79695

def event79697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27792⟩⟩) 1 ⟨27791⟩ 79693

def event79698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27792⟩⟩) (.product (.predecessor 0 79696 .coefficient) (.predecessor 1 79697 .coefficient) (⟨false, false, none, none, none⟩))

def event79699 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27792⟩⟩, .operator (⟨79695, 0⟩, ⟨79693, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact79700RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact79700RawTermsValid :
    exact79700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79700 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27792⟩⟩) exact79700RawTerms .large 79698 .exactZero (none)

def event79701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7189⟩⟩) 0 ⟨7177⟩ 79677

def event79702 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7189⟩⟩) (.authority (.operator))

def exact79703RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩]

theorem exact79703RawTermsValid :
    exact79703RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79703 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7189⟩⟩) exact79703RawTerms .large 79702 .exactZero (none)

def event79704 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27793⟩⟩) 0 ⟨7189⟩ 79703

def event79705 : Event := .predecessor (⟨.program ⟨257⟩, ⟨27793⟩⟩) 1 ⟨27792⟩ 79700

def event79706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨27793⟩⟩) (.sum [.predecessor 0 79704 .coefficient, .predecessor 1 79705 .coefficient])

def exact79707RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79707RawTermsValid :
    exact79707RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79707 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27793⟩⟩) exact79707RawTerms .large 79706 .exactZero (none)

def event79708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28440⟩⟩) 0 ⟨27793⟩ 79707

def event79709 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28440⟩⟩) 1 ⟨28439⟩ 79684

def event79710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28440⟩⟩) (.product (.predecessor 0 79708 .coefficient) (.predecessor 1 79709 .coefficient) (⟨false, false, none, none, none⟩))

def event79711 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28440⟩⟩, .operator (⟨79707, 0⟩, ⟨79684, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩, (1)⟩)

def event79712 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28440⟩⟩, .operator (⟨79707, 1⟩, ⟨79684, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩, (-1)⟩)

def event79713 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨28440⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28439⟩⟩) ⟨27615⟩ 79681)

def event79714 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28440⟩⟩, .relation 79713 0, ⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27615⟩⟩]⟩, (-1)⟩)

def exact79715RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27615⟩⟩]⟩, (-1)⟩]

theorem exact79715RawTermsValid :
    exact79715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79715 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28440⟩⟩) exact79715RawTerms .large 79710 .exactZero (none)

def event79716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26697⟩⟩) 0 ⟨26457⟩ 79673

def event79717 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26697⟩⟩) (.authority (.programFamilyFact))

def exact79718RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], []⟩, (1)⟩]

theorem exact79718RawTermsValid :
    exact79718RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79718 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26697⟩⟩) exact79718RawTerms (.finite 62) 79717 .exactZero (none)

def event79719 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26698⟩⟩) 0 ⟨6908⟩ 79695

def event79720 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26698⟩⟩) 1 ⟨26697⟩ 79718

def event79721 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26698⟩⟩) (.product (.predecessor 0 79719 .coefficient) (.predecessor 1 79720 .coefficient) (⟨false, true, none, none, some 1⟩))

def event79722 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26698⟩⟩, .operator (⟨79695, 0⟩, ⟨79718, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact79723RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact79723RawTermsValid :
    exact79723RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79723 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26698⟩⟩) exact79723RawTerms .large 79721 .exactZero (none)

def event79724 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7218⟩⟩) 0 ⟨7177⟩ 79677

def event79725 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7218⟩⟩) (.authority (.operator))

def exact79726RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩]

theorem exact79726RawTermsValid :
    exact79726RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79726 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7218⟩⟩) exact79726RawTerms .large 79725 .exactZero (none)

def event79727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26699⟩⟩) 0 ⟨7218⟩ 79726

def event79728 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26699⟩⟩) 1 ⟨26698⟩ 79723

def event79729 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26699⟩⟩) (.sum [.predecessor 0 79727 .coefficient, .predecessor 1 79728 .coefficient])

def exact79730RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79730RawTermsValid :
    exact79730RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79730 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26699⟩⟩) exact79730RawTerms .large 79729 .exactZero (none)

def event79731 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28443⟩⟩) 0 ⟨26699⟩ 79730

def event79732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28443⟩⟩) 1 ⟨28440⟩ 79715

def event79733 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28443⟩⟩) (.sum [.predecessor 0 79731 .coefficient, .predecessor 1 79732 .coefficient])

def exact79734RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27615⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79734RawTermsValid :
    exact79734RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79734 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28443⟩⟩) exact79734RawTerms .large 79733 .exactZero (none)

def event79735 : Event := .preFoldPolynomial 79734 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27615⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact79736RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27615⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event79736 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨28443⟩⟩) 79735 exact79736RawTerms .large 79733 .exactZero (none)

def event79737 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨26457⟩⟩) ⟨⟨97⟩, ⟨79⟩, ⟨135⟩⟩ ⟨79579, 79737⟩

def event79738 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨27279⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩) (1) 0 2 (.universal 79737 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27276⟩⟩]⟩) (none) 79736)

def event79739 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27279⟩⟩, .relation 79738 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩)

def event79740 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27279⟩⟩, .relation 79738 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩, (-1)⟩)

def event79741 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27279⟩⟩, .relation 79738 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27615⟩⟩]⟩, (1)⟩)

def event79742 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨27279⟩⟩, .relation 79738 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact79743RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27615⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79743RawTermsValid :
    exact79743RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79743 : Event := .resultExact (⟨.program ⟨257⟩, ⟨27279⟩⟩) exact79743RawTerms .large 79575 (.finite 202072841853861888) (some (79577))

def event79744 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28442⟩⟩) 0 ⟨27279⟩ 79743

def event79745 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28442⟩⟩) 1 ⟨28441⟩ 79565

def event79746 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28442⟩⟩) (.sum [.predecessor 0 79744 .coefficient, .predecessor 1 79745 .coefficient])

def event79747 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28442⟩⟩, .operator (⟨79743, 0⟩, ⟨79565, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28439⟩⟩]⟩, (1)⟩)

def event79748 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28442⟩⟩, .operator (⟨79743, 2⟩, ⟨79565, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26456⟩⟩], [⟨.program ⟨257⟩, ⟨27615⟩⟩]⟩, (-1)⟩)

def event79749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28442⟩⟩) (.sum [.result 79743 .summary, .result 79565 .summary])

def exact79750RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79750RawTermsValid :
    exact79750RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79750 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28442⟩⟩) exact79750RawTerms .large 79746 (.finite 32191557518723330170883082027008) (some (79749))

def event79751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68734⟩⟩) 0 ⟨65837⟩ 3287

def event79752 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68734⟩⟩) (.authority (.programFamilyFact))

def event79753 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68734⟩⟩) (.finite 3720)

def event79754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68736⟩⟩) 0 ⟨7177⟩ 15500

def event79755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68736⟩⟩) 1 ⟨68734⟩ 79753

def event79756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68736⟩⟩) (.authority (.operator))

def exact79757RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68736⟩⟩]⟩, (1)⟩]

theorem exact79757RawTermsValid :
    exact79757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79757 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68736⟩⟩) exact79757RawTerms .large 79756 .exactZero (none)

def event79758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70651⟩⟩) 0 ⟨68736⟩ 79757

def event79759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70651⟩⟩) (.authority (.operator))

def exact79760RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨70651⟩⟩]⟩, (1)⟩]

theorem exact79760RawTermsValid :
    exact79760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79760 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70651⟩⟩) exact79760RawTerms (.finite 8192) 79759 .exactZero (none)

def event79761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68565⟩⟩) 0 ⟨65609⟩ 3281

def event79762 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68565⟩⟩) (.authority (.programFamilyFact))

def event79763 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68565⟩⟩) (.finite 3720)

def event79764 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68566⟩⟩) 0 ⟨7177⟩ 15500

def event79765 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68566⟩⟩) 1 ⟨68565⟩ 79763

def event79766 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68566⟩⟩) (.authority (.operator))

def exact79767RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68566⟩⟩]⟩, (1)⟩]

theorem exact79767RawTermsValid :
    exact79767RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79767 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68566⟩⟩) exact79767RawTerms .large 79766 .exactZero (none)

def event79768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69306⟩⟩) 0 ⟨68566⟩ 79767

def event79769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69306⟩⟩) (.authority (.operator))

def exact79770RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩, (1)⟩]

theorem exact79770RawTermsValid :
    exact79770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79770 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69306⟩⟩) exact79770RawTerms (.finite 8192) 79769 .exactZero (none)

def event79771 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25803⟩⟩) 0 ⟨25802⟩ 3270

def event79772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25803⟩⟩) 1 ⟨10328⟩ 75903

def event79773 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25803⟩⟩) (.tensor (.predecessor 0 79771 .coefficient) (.predecessor 1 79772 .coefficient) true false)

def event79774 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨25803⟩⟩, .operator (⟨3270, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact79775RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact79775RawTermsValid :
    exact79775RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79775 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25803⟩⟩) exact79775RawTerms .large 79773 .exactZero (none)

def event79776 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10334⟩⟩) 0 ⟨10327⟩ 75773

def event79777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10334⟩⟩) 1 ⟨7276⟩ 21088

def event79778 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10334⟩⟩) (.product (.predecessor 0 79776 .coefficient) (.predecessor 1 79777 .coefficient) (⟨false, false, none, none, none⟩))

def event79779 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10334⟩⟩, .operator (⟨75773, 0⟩, ⟨21088, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩)

def exact79780RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩]

theorem exact79780RawTermsValid :
    exact79780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79780 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10334⟩⟩) exact79780RawTerms .large 79778 .exactZero (none)

def event79781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25804⟩⟩) 0 ⟨10334⟩ 79780

def event79782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25804⟩⟩) 1 ⟨25803⟩ 79775

def event79783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25804⟩⟩) (.sum [.predecessor 0 79781 .coefficient, .predecessor 1 79782 .coefficient])

def exact79784RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79784RawTermsValid :
    exact79784RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79784 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25804⟩⟩) exact79784RawTerms .large 79783 .exactZero (none)

def event79785 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25805⟩⟩) 0 ⟨25804⟩ 79784

def event79786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25805⟩⟩) 1 ⟨102⟩ 21080

def event79787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25805⟩⟩) (.sum [.predecessor 0 79785 .coefficient, .predecessor 1 79786 .coefficient])

def event79788 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25805⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨102⟩⟩]⟩) [⟨.result 21080 .coefficient, false, none⟩])

def event79789 : Event := .survivorFold (1) 79788

def exact79790RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79790RawTermsValid :
    exact79790RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79790 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25805⟩⟩) exact79790RawTerms .large 79787 (.finite 26) (some (79788))

def event79791 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65610⟩⟩) 0 ⟨25805⟩ 79790

def event79792 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65610⟩⟩) 1 ⟨65607⟩ 3273

def event79793 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65610⟩⟩) (.product (.predecessor 0 79791 .coefficient) (.predecessor 1 79792 .coefficient) (⟨false, true, none, none, some 1⟩))

def event79794 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65610⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨65607⟩⟩], []⟩) [⟨.result 3273 .coefficient, true, some 1⟩])

def event79795 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65610⟩⟩) (.product (.result 79790 .summary) (.transfer 79794) (⟨false, false, none, none, none⟩))

def event79796 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65610⟩⟩, .operator (⟨79790, 1⟩, ⟨3273, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event79797 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65610⟩⟩, .operator (⟨79790, 0⟩, ⟨3273, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩)

def exact79798RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩]

theorem exact79798RawTermsValid :
    exact79798RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79798 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65610⟩⟩) exact79798RawTerms .large 79793 (.finite 23855104) (some (79795))

def event79799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65611⟩⟩) 0 ⟨65607⟩ 3273

def event79800 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65611⟩⟩) 1 ⟨10328⟩ 75903

def event79801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65611⟩⟩) (.tensor (.predecessor 0 79799 .coefficient) (.predecessor 1 79800 .coefficient) true false)

def event79802 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65611⟩⟩, .operator (⟨3273, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact79803RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact79803RawTermsValid :
    exact79803RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79803 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65611⟩⟩) exact79803RawTerms .large 79801 .exactZero (none)

def event79804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10352⟩⟩) 0 ⟨10327⟩ 75773

def event79805 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10352⟩⟩) 1 ⟨7294⟩ 21129

def event79806 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10352⟩⟩) (.product (.predecessor 0 79804 .coefficient) (.predecessor 1 79805 .coefficient) (⟨false, false, none, none, none⟩))

def event79807 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10352⟩⟩, .operator (⟨75773, 0⟩, ⟨21129, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩]⟩, (1)⟩)

def exact79808RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩]⟩, (1)⟩]

theorem exact79808RawTermsValid :
    exact79808RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79808 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10352⟩⟩) exact79808RawTerms .large 79806 .exactZero (none)

def event79809 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65612⟩⟩) 0 ⟨10352⟩ 79808

def event79810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65612⟩⟩) 1 ⟨65611⟩ 79803

def event79811 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65612⟩⟩) (.sum [.predecessor 0 79809 .coefficient, .predecessor 1 79810 .coefficient])

def exact79812RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79812RawTermsValid :
    exact79812RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79812 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65612⟩⟩) exact79812RawTerms .large 79811 .exactZero (none)

def event79813 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65613⟩⟩) 0 ⟨65612⟩ 79812

def event79814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65613⟩⟩) 1 ⟨120⟩ 21121

def event79815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65613⟩⟩) (.sum [.predecessor 0 79813 .coefficient, .predecessor 1 79814 .coefficient])

def event79816 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65613⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨120⟩⟩]⟩) [⟨.result 21121 .coefficient, false, none⟩])

def event79817 : Event := .survivorFold (1) 79816

def exact79818RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79818RawTermsValid :
    exact79818RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79818 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65613⟩⟩) exact79818RawTerms .large 79815 (.finite 26) (some (79816))

def event79819 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65614⟩⟩) 0 ⟨65613⟩ 79818

def event79820 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65614⟩⟩) 1 ⟨9542⟩ 21118

def event79821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65614⟩⟩) (.product (.predecessor 0 79819 .coefficient) (.predecessor 1 79820 .coefficient) (⟨false, false, none, none, none⟩))

def event79822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65614⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) [⟨.result 21114 .coefficient, false, none⟩])

def event79823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65614⟩⟩) (.product (.result 79818 .summary) (.transfer 79822) (⟨false, false, none, none, none⟩))

def event79824 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65614⟩⟩, .operator (⟨79818, 1⟩, ⟨21118, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (-1)⟩)

def event79825 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨65614⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9541⟩⟩) ⟨7276⟩ 21088)

def event79826 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65614⟩⟩, .relation 79825 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (-1)⟩)

def event79827 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65614⟩⟩, .operator (⟨79818, 0⟩, ⟨21118, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩)

def exact79828RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (-1)⟩]

theorem exact79828RawTermsValid :
    exact79828RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79828 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65614⟩⟩) exact79828RawTerms .large 79821 (.finite 279172874240) (some (79823))

def event79829 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65615⟩⟩) 0 ⟨65614⟩ 79828

def event79830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65615⟩⟩) 1 ⟨65610⟩ 79798

def event79831 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65615⟩⟩) (.sum [.predecessor 0 79829 .coefficient, .predecessor 1 79830 .coefficient])

def event79832 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65615⟩⟩, .operator (⟨79828, 1⟩, ⟨79798, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩)

def event79833 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65615⟩⟩) (.sum [.result 79828 .summary, .result 79798 .summary])

def exact79834RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79834RawTermsValid :
    exact79834RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79834 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65615⟩⟩) exact79834RawTerms .large 79831 (.finite 279196729344) (some (79833))

def event79835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69307⟩⟩) 0 ⟨65615⟩ 79834

def event79836 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69307⟩⟩) 1 ⟨69306⟩ 79770

def event79837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69307⟩⟩) (.product (.predecessor 0 79835 .coefficient) (.predecessor 1 79836 .coefficient) (⟨false, false, none, none, none⟩))

def event79838 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69307⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩) [⟨.result 79770 .coefficient, false, none⟩])

def event79839 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69307⟩⟩) (.product (.result 79834 .summary) (.transfer 79838) (⟨false, false, none, none, none⟩))

def event79840 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69307⟩⟩, .operator (⟨79834, 1⟩, ⟨79770, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩, (-1)⟩)

def event79841 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨69307⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69306⟩⟩) ⟨68566⟩ 79767)

def event79842 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69307⟩⟩, .relation 79841 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨68566⟩⟩]⟩, (-1)⟩)

def event79843 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69307⟩⟩, .operator (⟨79834, 0⟩, ⟨79770, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩, (1)⟩)

def exact79844RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨68566⟩⟩]⟩, (-1)⟩]

theorem exact79844RawTermsValid :
    exact79844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79844 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69307⟩⟩) exact79844RawTerms .large 79837 (.finite 2997852054206608834560) (some (79839))

def event79845 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67830⟩⟩) 0 ⟨65609⟩ 3281

def event79846 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67830⟩⟩) (.authority (.relationPreimageSource ⟨46⟩))

def exact79847RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨67830⟩⟩]⟩, (1)⟩]

theorem exact79847RawTermsValid :
    exact79847RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79847 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67830⟩⟩) exact79847RawTerms (.finite 5647228698) 79846 .exactZero (none)

def event79848 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67832⟩⟩) 0 ⟨67830⟩ 79847

def event79849 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67832⟩⟩) 1 ⟨2370⟩ 4

def event79850 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67832⟩⟩) (.scale (.predecessor 0 79848 .coefficient) (.value (.predecessor 1 79849 .coefficient)))

def exact79851RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨67830⟩⟩]⟩, (1)⟩]

theorem exact79851RawTermsValid :
    exact79851RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79851 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67832⟩⟩) exact79851RawTerms (.finite 5647228698) 79850 .exactZero (none)

def event79852 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67833⟩⟩) 0 ⟨10368⟩ 75995

def event79853 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67833⟩⟩) 1 ⟨67832⟩ 79851

def event79854 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67833⟩⟩) (.product (.predecessor 0 79852 .coefficient) (.predecessor 1 79853 .coefficient) (⟨false, false, none, none, none⟩))

def event79855 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67833⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨67830⟩⟩]⟩) [⟨.result 79847 .coefficient, false, none⟩])

def event79856 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67833⟩⟩) (.product (.result 75995 .summary) (.transfer 79855) (⟨false, false, none, none, none⟩))

def event79857 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67833⟩⟩, .operator (⟨75995, 0⟩, ⟨79851, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67830⟩⟩]⟩, (1)⟩)

def event79858 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨67831⟩⟩)

def event79859 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event79860 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event79861 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event79862 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event79863 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event79864 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event79865 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event79866 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event79867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 79866

def event79868 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 79864

def event79869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 79867 .coefficient) (.value (.predecessor 1 79868 .coefficient)))

def event79870 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event79871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 79870

def eventLeaf4976 : Array AnnotatedEvent := #[
  { event := event79616
    frameStart := 79579 },
  { event := event79617
    frameStart := 79579 },
  { event := event79618
    frameStart := 79579 },
  { event := event79619
    frameStart := 79579 },
  { event := event79620
    frameStart := 79579 },
  { event := event79621
    frameStart := 79579 },
  { event := event79622
    frameStart := 79579 },
  { event := event79623
    frameStart := 79579 },
  { event := event79624
    frameStart := 79579 },
  { event := event79625
    frameStart := 79579 },
  { event := event79626
    frameStart := 79579 },
  { event := event79627
    frameStart := 79579 },
  { event := event79628
    frameStart := 79579 },
  { event := event79629
    frameStart := 79579 },
  { event := event79630
    frameStart := 79579 },
  { event := event79631
    frameStart := 79579 }
]

def eventLeaf4977 : Array AnnotatedEvent := #[
  { event := event79632
    frameStart := 79579 },
  { event := event79633
    frameStart := 79633 },
  { event := event79634
    frameStart := 79633 },
  { event := event79635
    frameStart := 79633 },
  { event := event79636
    frameStart := 79633 },
  { event := event79637
    frameStart := 79633 },
  { event := event79638
    frameStart := 79633 },
  { event := event79639
    frameStart := 79633 },
  { event := event79640
    frameStart := 79633 },
  { event := event79641
    frameStart := 79633 },
  { event := event79642
    frameStart := 79633 },
  { event := event79643
    frameStart := 79633 },
  { event := event79644
    frameStart := 79633 },
  { event := event79645
    frameStart := 79633 },
  { event := event79646
    frameStart := 79633 },
  { event := event79647
    frameStart := 79633 }
]

def eventLeaf4978 : Array AnnotatedEvent := #[
  { event := event79648
    frameStart := 79633 },
  { event := event79649
    frameStart := 79633 },
  { event := event79650
    frameStart := 79633 },
  { event := event79651
    frameStart := 79633 },
  { event := event79652
    frameStart := 79633 },
  { event := event79653
    frameStart := 79633 },
  { event := event79654
    frameStart := 79633 },
  { event := event79655
    frameStart := 79633 },
  { event := event79656
    frameStart := 79633 },
  { event := event79657
    frameStart := 79633 },
  { event := event79658
    frameStart := 79633 },
  { event := event79659
    frameStart := 79633 },
  { event := event79660
    frameStart := 79633 },
  { event := event79661
    frameStart := 79633 },
  { event := event79662
    frameStart := 79633 },
  { event := event79663
    frameStart := 79633 }
]

def eventLeaf4979 : Array AnnotatedEvent := #[
  { event := event79664
    frameStart := 79633 },
  { event := event79665
    frameStart := 79633 },
  { event := event79666
    frameStart := 79633 },
  { event := event79667
    frameStart := 79633 },
  { event := event79668
    frameStart := 79633 },
  { event := event79669
    frameStart := 79633 },
  { event := event79670
    frameStart := 79633 },
  { event := event79671
    frameStart := 79633 },
  { event := event79672
    frameStart := 79633 },
  { event := event79673
    frameStart := 79633 },
  { event := event79674
    frameStart := 79633 },
  { event := event79675
    frameStart := 79633 },
  { event := event79676
    frameStart := 79633 },
  { event := event79677
    frameStart := 79633 },
  { event := event79678
    frameStart := 79633 },
  { event := event79679
    frameStart := 79633 }
]

def eventLeaf4980 : Array AnnotatedEvent := #[
  { event := event79680
    frameStart := 79633 },
  { event := event79681
    frameStart := 79633 },
  { event := event79682
    frameStart := 79633 },
  { event := event79683
    frameStart := 79633 },
  { event := event79684
    frameStart := 79633 },
  { event := event79685
    frameStart := 79633 },
  { event := event79686
    frameStart := 79633 },
  { event := event79687
    frameStart := 79633 },
  { event := event79688
    frameStart := 79633 },
  { event := event79689
    frameStart := 79633 },
  { event := event79690
    frameStart := 79633 },
  { event := event79691
    frameStart := 79633 },
  { event := event79692
    frameStart := 79633 },
  { event := event79693
    frameStart := 79633 },
  { event := event79694
    frameStart := 79633 },
  { event := event79695
    frameStart := 79633 }
]

def eventLeaf4981 : Array AnnotatedEvent := #[
  { event := event79696
    frameStart := 79633 },
  { event := event79697
    frameStart := 79633 },
  { event := event79698
    frameStart := 79633 },
  { event := event79699
    frameStart := 79633 },
  { event := event79700
    frameStart := 79633 },
  { event := event79701
    frameStart := 79633 },
  { event := event79702
    frameStart := 79633 },
  { event := event79703
    frameStart := 79633 },
  { event := event79704
    frameStart := 79633 },
  { event := event79705
    frameStart := 79633 },
  { event := event79706
    frameStart := 79633 },
  { event := event79707
    frameStart := 79633 },
  { event := event79708
    frameStart := 79633 },
  { event := event79709
    frameStart := 79633 },
  { event := event79710
    frameStart := 79633 },
  { event := event79711
    frameStart := 79633 }
]

def eventLeaf4982 : Array AnnotatedEvent := #[
  { event := event79712
    frameStart := 79633 },
  { event := event79713
    frameStart := 79633 },
  { event := event79714
    frameStart := 79633 },
  { event := event79715
    frameStart := 79633 },
  { event := event79716
    frameStart := 79633 },
  { event := event79717
    frameStart := 79633 },
  { event := event79718
    frameStart := 79633 },
  { event := event79719
    frameStart := 79633 },
  { event := event79720
    frameStart := 79633 },
  { event := event79721
    frameStart := 79633 },
  { event := event79722
    frameStart := 79633 },
  { event := event79723
    frameStart := 79633 },
  { event := event79724
    frameStart := 79633 },
  { event := event79725
    frameStart := 79633 },
  { event := event79726
    frameStart := 79633 },
  { event := event79727
    frameStart := 79633 }
]

def eventLeaf4983 : Array AnnotatedEvent := #[
  { event := event79728
    frameStart := 79633 },
  { event := event79729
    frameStart := 79633 },
  { event := event79730
    frameStart := 79633 },
  { event := event79731
    frameStart := 79633 },
  { event := event79732
    frameStart := 79633 },
  { event := event79733
    frameStart := 79633 },
  { event := event79734
    frameStart := 79633 },
  { event := event79735
    frameStart := 79633 },
  { event := event79736
    frameStart := 79633 },
  { event := event79737
    frameStart := 0 },
  { event := event79738
    frameStart := 0 },
  { event := event79739
    frameStart := 0 },
  { event := event79740
    frameStart := 0 },
  { event := event79741
    frameStart := 0 },
  { event := event79742
    frameStart := 0 },
  { event := event79743
    frameStart := 0 }
]

def eventLeaf4984 : Array AnnotatedEvent := #[
  { event := event79744
    frameStart := 0 },
  { event := event79745
    frameStart := 0 },
  { event := event79746
    frameStart := 0 },
  { event := event79747
    frameStart := 0 },
  { event := event79748
    frameStart := 0 },
  { event := event79749
    frameStart := 0 },
  { event := event79750
    frameStart := 0 },
  { event := event79751
    frameStart := 0 },
  { event := event79752
    frameStart := 0 },
  { event := event79753
    frameStart := 0 },
  { event := event79754
    frameStart := 0 },
  { event := event79755
    frameStart := 0 },
  { event := event79756
    frameStart := 0 },
  { event := event79757
    frameStart := 0 },
  { event := event79758
    frameStart := 0 },
  { event := event79759
    frameStart := 0 }
]

def eventLeaf4985 : Array AnnotatedEvent := #[
  { event := event79760
    frameStart := 0 },
  { event := event79761
    frameStart := 0 },
  { event := event79762
    frameStart := 0 },
  { event := event79763
    frameStart := 0 },
  { event := event79764
    frameStart := 0 },
  { event := event79765
    frameStart := 0 },
  { event := event79766
    frameStart := 0 },
  { event := event79767
    frameStart := 0 },
  { event := event79768
    frameStart := 0 },
  { event := event79769
    frameStart := 0 },
  { event := event79770
    frameStart := 0 },
  { event := event79771
    frameStart := 0 },
  { event := event79772
    frameStart := 0 },
  { event := event79773
    frameStart := 0 },
  { event := event79774
    frameStart := 0 },
  { event := event79775
    frameStart := 0 }
]

def eventLeaf4986 : Array AnnotatedEvent := #[
  { event := event79776
    frameStart := 0 },
  { event := event79777
    frameStart := 0 },
  { event := event79778
    frameStart := 0 },
  { event := event79779
    frameStart := 0 },
  { event := event79780
    frameStart := 0 },
  { event := event79781
    frameStart := 0 },
  { event := event79782
    frameStart := 0 },
  { event := event79783
    frameStart := 0 },
  { event := event79784
    frameStart := 0 },
  { event := event79785
    frameStart := 0 },
  { event := event79786
    frameStart := 0 },
  { event := event79787
    frameStart := 0 },
  { event := event79788
    frameStart := 0 },
  { event := event79789
    frameStart := 0 },
  { event := event79790
    frameStart := 0 },
  { event := event79791
    frameStart := 0 }
]

def eventLeaf4987 : Array AnnotatedEvent := #[
  { event := event79792
    frameStart := 0 },
  { event := event79793
    frameStart := 0 },
  { event := event79794
    frameStart := 0 },
  { event := event79795
    frameStart := 0 },
  { event := event79796
    frameStart := 0 },
  { event := event79797
    frameStart := 0 },
  { event := event79798
    frameStart := 0 },
  { event := event79799
    frameStart := 0 },
  { event := event79800
    frameStart := 0 },
  { event := event79801
    frameStart := 0 },
  { event := event79802
    frameStart := 0 },
  { event := event79803
    frameStart := 0 },
  { event := event79804
    frameStart := 0 },
  { event := event79805
    frameStart := 0 },
  { event := event79806
    frameStart := 0 },
  { event := event79807
    frameStart := 0 }
]

def eventLeaf4988 : Array AnnotatedEvent := #[
  { event := event79808
    frameStart := 0 },
  { event := event79809
    frameStart := 0 },
  { event := event79810
    frameStart := 0 },
  { event := event79811
    frameStart := 0 },
  { event := event79812
    frameStart := 0 },
  { event := event79813
    frameStart := 0 },
  { event := event79814
    frameStart := 0 },
  { event := event79815
    frameStart := 0 },
  { event := event79816
    frameStart := 0 },
  { event := event79817
    frameStart := 0 },
  { event := event79818
    frameStart := 0 },
  { event := event79819
    frameStart := 0 },
  { event := event79820
    frameStart := 0 },
  { event := event79821
    frameStart := 0 },
  { event := event79822
    frameStart := 0 },
  { event := event79823
    frameStart := 0 }
]

def eventLeaf4989 : Array AnnotatedEvent := #[
  { event := event79824
    frameStart := 0 },
  { event := event79825
    frameStart := 0 },
  { event := event79826
    frameStart := 0 },
  { event := event79827
    frameStart := 0 },
  { event := event79828
    frameStart := 0 },
  { event := event79829
    frameStart := 0 },
  { event := event79830
    frameStart := 0 },
  { event := event79831
    frameStart := 0 },
  { event := event79832
    frameStart := 0 },
  { event := event79833
    frameStart := 0 },
  { event := event79834
    frameStart := 0 },
  { event := event79835
    frameStart := 0 },
  { event := event79836
    frameStart := 0 },
  { event := event79837
    frameStart := 0 },
  { event := event79838
    frameStart := 0 },
  { event := event79839
    frameStart := 0 }
]

def eventLeaf4990 : Array AnnotatedEvent := #[
  { event := event79840
    frameStart := 0 },
  { event := event79841
    frameStart := 0 },
  { event := event79842
    frameStart := 0 },
  { event := event79843
    frameStart := 0 },
  { event := event79844
    frameStart := 0 },
  { event := event79845
    frameStart := 0 },
  { event := event79846
    frameStart := 0 },
  { event := event79847
    frameStart := 0 },
  { event := event79848
    frameStart := 0 },
  { event := event79849
    frameStart := 0 },
  { event := event79850
    frameStart := 0 },
  { event := event79851
    frameStart := 0 },
  { event := event79852
    frameStart := 0 },
  { event := event79853
    frameStart := 0 },
  { event := event79854
    frameStart := 0 },
  { event := event79855
    frameStart := 0 }
]

def eventLeaf4991 : Array AnnotatedEvent := #[
  { event := event79856
    frameStart := 0 },
  { event := event79857
    frameStart := 0 },
  { event := event79858
    frameStart := 79858 },
  { event := event79859
    frameStart := 79858 },
  { event := event79860
    frameStart := 79858 },
  { event := event79861
    frameStart := 79858 },
  { event := event79862
    frameStart := 79858 },
  { event := event79863
    frameStart := 79858 },
  { event := event79864
    frameStart := 79858 },
  { event := event79865
    frameStart := 79858 },
  { event := event79866
    frameStart := 79858 },
  { event := event79867
    frameStart := 79858 },
  { event := event79868
    frameStart := 79858 },
  { event := event79869
    frameStart := 79858 },
  { event := event79870
    frameStart := 79858 },
  { event := event79871
    frameStart := 79858 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events311
