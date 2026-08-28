import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events186

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event47616 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44869⟩⟩) 0 ⟨44013⟩ 47615

def event47617 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44869⟩⟩) (.authority (.operator))

def exact47618RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩, (1)⟩]

theorem exact47618RawTermsValid :
    exact47618RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47618 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44869⟩⟩) exact47618RawTerms (.finite 8192) 47617 .exactZero (none)

def event47619 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43836⟩⟩) 0 ⟨42668⟩ 1647

def event47620 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43836⟩⟩) (.authority (.programFamilyFact))

def event47621 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨43836⟩⟩) (.finite 3720)

def event47622 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43837⟩⟩) 0 ⟨7177⟩ 15500

def event47623 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43837⟩⟩) 1 ⟨43836⟩ 47621

def event47624 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43837⟩⟩) (.authority (.operator))

def exact47625RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43837⟩⟩]⟩, (1)⟩]

theorem exact47625RawTermsValid :
    exact47625RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47625 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43837⟩⟩) exact47625RawTerms .large 47624 .exactZero (none)

def event47626 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44387⟩⟩) 0 ⟨43837⟩ 47625

def event47627 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44387⟩⟩) (.authority (.operator))

def exact47628RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩, (1)⟩]

theorem exact47628RawTermsValid :
    exact47628RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47628 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44387⟩⟩) exact47628RawTerms (.finite 8192) 47627 .exactZero (none)

def event47629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42669⟩⟩) 0 ⟨42666⟩ 1636

def event47630 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42669⟩⟩) 1 ⟨11176⟩ 46653

def event47631 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42669⟩⟩) (.tensor (.predecessor 0 47629 .coefficient) (.predecessor 1 47630 .coefficient) true false)

def event47632 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42669⟩⟩, .operator (⟨1636, 0⟩, ⟨46653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact47633RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact47633RawTermsValid :
    exact47633RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47633 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42669⟩⟩) exact47633RawTerms .large 47631 .exactZero (none)

def event47634 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11189⟩⟩) 0 ⟨11175⟩ 46523

def event47635 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11189⟩⟩) 1 ⟨7283⟩ 18082

def event47636 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11189⟩⟩) (.product (.predecessor 0 47634 .coefficient) (.predecessor 1 47635 .coefficient) (⟨false, false, none, none, none⟩))

def event47637 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11189⟩⟩, .operator (⟨46523, 0⟩, ⟨18082, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩)

def exact47638RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩]

theorem exact47638RawTermsValid :
    exact47638RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47638 : Event := .resultExact (⟨.program ⟨257⟩, ⟨11189⟩⟩) exact47638RawTerms .large 47636 .exactZero (none)

def event47639 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42670⟩⟩) 0 ⟨11189⟩ 47638

def event47640 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42670⟩⟩) 1 ⟨42669⟩ 47633

def event47641 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42670⟩⟩) (.sum [.predecessor 0 47639 .coefficient, .predecessor 1 47640 .coefficient])

def exact47642RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47642RawTermsValid :
    exact47642RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47642 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42670⟩⟩) exact47642RawTerms .large 47641 .exactZero (none)

def event47643 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42671⟩⟩) 0 ⟨42670⟩ 47642

def event47644 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42671⟩⟩) 1 ⟨109⟩ 18074

def event47645 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42671⟩⟩) (.sum [.predecessor 0 47643 .coefficient, .predecessor 1 47644 .coefficient])

def event47646 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42671⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨109⟩⟩]⟩) [⟨.result 18074 .coefficient, false, none⟩])

def event47647 : Event := .survivorFold (1) 47646

def exact47648RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47648RawTermsValid :
    exact47648RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47648 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42671⟩⟩) exact47648RawTerms .large 47645 (.finite 26) (some (47646))

def event47649 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42672⟩⟩) 0 ⟨42671⟩ 47648

def event47650 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42672⟩⟩) 1 ⟨14601⟩ 1639

def event47651 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42672⟩⟩) (.product (.predecessor 0 47649 .coefficient) (.predecessor 1 47650 .coefficient) (⟨false, true, none, none, some 1⟩))

def event47652 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42672⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩], []⟩) [⟨.result 1639 .coefficient, true, some 1⟩])

def event47653 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42672⟩⟩) (.product (.result 47648 .summary) (.transfer 47652) (⟨false, false, none, none, none⟩))

def event47654 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42672⟩⟩, .operator (⟨47648, 1⟩, ⟨1639, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event47655 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42672⟩⟩, .operator (⟨47648, 0⟩, ⟨1639, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩)

def exact47656RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47656RawTermsValid :
    exact47656RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47656 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42672⟩⟩) exact47656RawTerms .large 47651 (.finite 44302336) (some (47653))

def event47657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14602⟩⟩) 0 ⟨14601⟩ 1639

def event47658 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14602⟩⟩) 1 ⟨11176⟩ 46653

def event47659 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14602⟩⟩) (.tensor (.predecessor 0 47657 .coefficient) (.predecessor 1 47658 .coefficient) true false)

def event47660 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14602⟩⟩, .operator (⟨1639, 0⟩, ⟨46653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact47661RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact47661RawTermsValid :
    exact47661RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47661 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14602⟩⟩) exact47661RawTerms .large 47659 .exactZero (none)

def event47662 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11206⟩⟩) 0 ⟨11175⟩ 46523

def event47663 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11206⟩⟩) 1 ⟨7300⟩ 18123

def event47664 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11206⟩⟩) (.product (.predecessor 0 47662 .coefficient) (.predecessor 1 47663 .coefficient) (⟨false, false, none, none, none⟩))

def event47665 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11206⟩⟩, .operator (⟨46523, 0⟩, ⟨18123, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩]⟩, (1)⟩)

def exact47666RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩]⟩, (1)⟩]

theorem exact47666RawTermsValid :
    exact47666RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47666 : Event := .resultExact (⟨.program ⟨257⟩, ⟨11206⟩⟩) exact47666RawTerms .large 47664 .exactZero (none)

def event47667 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14603⟩⟩) 0 ⟨11206⟩ 47666

def event47668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14603⟩⟩) 1 ⟨14602⟩ 47661

def event47669 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14603⟩⟩) (.sum [.predecessor 0 47667 .coefficient, .predecessor 1 47668 .coefficient])

def exact47670RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47670RawTermsValid :
    exact47670RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47670 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14603⟩⟩) exact47670RawTerms .large 47669 .exactZero (none)

def event47671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14604⟩⟩) 0 ⟨14603⟩ 47670

def event47672 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14604⟩⟩) 1 ⟨126⟩ 18115

def event47673 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14604⟩⟩) (.sum [.predecessor 0 47671 .coefficient, .predecessor 1 47672 .coefficient])

def event47674 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14604⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨126⟩⟩]⟩) [⟨.result 18115 .coefficient, false, none⟩])

def event47675 : Event := .survivorFold (1) 47674

def exact47676RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47676RawTermsValid :
    exact47676RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47676 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14604⟩⟩) exact47676RawTerms .large 47673 (.finite 26) (some (47674))

def event47677 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14605⟩⟩) 0 ⟨14604⟩ 47676

def event47678 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14605⟩⟩) 1 ⟨9560⟩ 18112

def event47679 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14605⟩⟩) (.product (.predecessor 0 47677 .coefficient) (.predecessor 1 47678 .coefficient) (⟨false, false, none, none, none⟩))

def event47680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14605⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩) [⟨.result 18108 .coefficient, false, none⟩])

def event47681 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14605⟩⟩) (.product (.result 47676 .summary) (.transfer 47680) (⟨false, false, none, none, none⟩))

def event47682 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14605⟩⟩, .operator (⟨47676, 1⟩, ⟨18112, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (-1)⟩)

def event47683 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨14605⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9559⟩⟩) ⟨7283⟩ 18082)

def event47684 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14605⟩⟩, .relation 47683 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (-1)⟩)

def event47685 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨14605⟩⟩, .operator (⟨47676, 0⟩, ⟨18112, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩)

def exact47686RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (-1)⟩]

theorem exact47686RawTermsValid :
    exact47686RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47686 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14605⟩⟩) exact47686RawTerms .large 47679 (.finite 279172874240) (some (47681))

def event47687 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42673⟩⟩) 0 ⟨14605⟩ 47686

def event47688 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42673⟩⟩) 1 ⟨42672⟩ 47656

def event47689 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42673⟩⟩) (.sum [.predecessor 0 47687 .coefficient, .predecessor 1 47688 .coefficient])

def event47690 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42673⟩⟩, .operator (⟨47686, 1⟩, ⟨47656, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩)

def event47691 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42673⟩⟩) (.sum [.result 47686 .summary, .result 47656 .summary])

def exact47692RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47692RawTermsValid :
    exact47692RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47692 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42673⟩⟩) exact47692RawTerms .large 47689 (.finite 279217176576) (some (47691))

def event47693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44388⟩⟩) 0 ⟨42673⟩ 47692

def event47694 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44388⟩⟩) 1 ⟨44387⟩ 47628

def event47695 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44388⟩⟩) (.product (.predecessor 0 47693 .coefficient) (.predecessor 1 47694 .coefficient) (⟨false, false, none, none, none⟩))

def event47696 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44388⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩) [⟨.result 47628 .coefficient, false, none⟩])

def event47697 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44388⟩⟩) (.product (.result 47692 .summary) (.transfer 47696) (⟨false, false, none, none, none⟩))

def event47698 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44388⟩⟩, .operator (⟨47692, 1⟩, ⟨47628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩, (-1)⟩)

def event47699 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44388⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44387⟩⟩) ⟨43837⟩ 47625)

def event47700 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44388⟩⟩, .relation 47699 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨43837⟩⟩]⟩, (-1)⟩)

def event47701 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44388⟩⟩, .operator (⟨47692, 0⟩, ⟨47628, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩, (1)⟩)

def exact47702RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨43837⟩⟩]⟩, (-1)⟩]

theorem exact47702RawTermsValid :
    exact47702RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47702 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44388⟩⟩) exact47702RawTerms .large 47695 (.finite 2998071604688443146240) (some (47697))

def event47703 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43309⟩⟩) 0 ⟨42668⟩ 1647

def event47704 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43309⟩⟩) (.authority (.relationPreimageSource ⟨52⟩))

def exact47705RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩, (1)⟩]

theorem exact47705RawTermsValid :
    exact47705RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47705 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43309⟩⟩) exact47705RawTerms (.finite 5647228698) 47704 .exactZero (none)

def event47706 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43311⟩⟩) 0 ⟨43309⟩ 47705

def event47707 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43311⟩⟩) 1 ⟨2370⟩ 4

def event47708 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43311⟩⟩) (.scale (.predecessor 0 47706 .coefficient) (.value (.predecessor 1 47707 .coefficient)))

def exact47709RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩, (1)⟩]

theorem exact47709RawTermsValid :
    exact47709RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47709 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43311⟩⟩) exact47709RawTerms (.finite 5647228698) 47708 .exactZero (none)

def event47710 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43312⟩⟩) 0 ⟨11216⟩ 46745

def event47711 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43312⟩⟩) 1 ⟨43311⟩ 47709

def event47712 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43312⟩⟩) (.product (.predecessor 0 47710 .coefficient) (.predecessor 1 47711 .coefficient) (⟨false, false, none, none, none⟩))

def event47713 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43312⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩) [⟨.result 47705 .coefficient, false, none⟩])

def event47714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43312⟩⟩) (.product (.result 46745 .summary) (.transfer 47713) (⟨false, false, none, none, none⟩))

def event47715 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43312⟩⟩, .operator (⟨46745, 0⟩, ⟨47709, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩, (1)⟩)

def event47716 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨43310⟩⟩)

def event47717 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event47718 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event47719 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event47720 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event47721 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event47722 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event47723 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event47724 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event47725 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 47724

def event47726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 47722

def event47727 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 47725 .coefficient) (.value (.predecessor 1 47726 .coefficient)))

def event47728 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event47729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 47728

def event47730 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 47720

def event47731 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 47729 .coefficient, .predecessor 1 47730 .coefficient])

def event47732 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event47733 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 47732

def event47734 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 47718

def event47735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 47734 .coefficient))

def event47736 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event47737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42666⟩⟩) 0 ⟨11173⟩ 47736

def event47738 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42666⟩⟩) (.authority (.programFamilyFact))

def exact47739RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42666⟩⟩], []⟩, (1)⟩]

theorem exact47739RawTermsValid :
    exact47739RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47739 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42666⟩⟩) exact47739RawTerms (.finite 52) 47738 .exactZero (none)

def event47740 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14601⟩⟩) 0 ⟨11173⟩ 47736

def event47741 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14601⟩⟩) (.authority (.programFamilyFact))

def exact47742RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩], []⟩, (1)⟩]

theorem exact47742RawTermsValid :
    exact47742RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47742 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14601⟩⟩) exact47742RawTerms (.finite 52) 47741 .exactZero (none)

def event47743 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42667⟩⟩) 0 ⟨14601⟩ 47742

def event47744 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42667⟩⟩) 1 ⟨42666⟩ 47739

def event47745 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42667⟩⟩) (.product (.predecessor 0 47743 .coefficient) (.predecessor 1 47744 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event47746 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42667⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], []⟩) [⟨.result 47742 .coefficient, true, some 1⟩, ⟨.result 47739 .coefficient, true, some 1⟩])

def event47747 : Event := .survivorFold (1) 47746

def exact47748RawTerms : List Term := []

theorem exact47748RawTermsValid :
    exact47748RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47748 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42667⟩⟩) exact47748RawTerms (.finite 2704) 47745 (.finite 2704) (some (47746))

def event47749 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42668⟩⟩) 0 ⟨42667⟩ 47748

def event47750 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42668⟩⟩) (.identity (.predecessor 0 47749 .coefficient))

def event47751 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42668⟩⟩) (.finite 2704)

def event47752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43309⟩⟩) 0 ⟨42668⟩ 47751

def event47753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43309⟩⟩) (.authority (.relationPreimageSource ⟨52⟩))

def exact47754RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩, (1)⟩]

theorem exact47754RawTermsValid :
    exact47754RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47754 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43309⟩⟩) exact47754RawTerms (.finite 5647228698) 47753 .exactZero (none)

def event47755 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact47756RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact47756RawTermsValid :
    exact47756RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47756 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact47756RawTerms .large 47755 .exactZero (none)

def event47757 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43310⟩⟩) 0 ⟨35⟩ 47756

def event47758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43310⟩⟩) 1 ⟨43309⟩ 47754

def event47759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43310⟩⟩) (.product (.predecessor 0 47757 .coefficient) (.predecessor 1 47758 .coefficient) (⟨false, false, none, none, none⟩))

def event47760 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43310⟩⟩, .operator (⟨47756, 0⟩, ⟨47754, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩, (1)⟩)

def exact47761RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩, (1)⟩]

theorem exact47761RawTermsValid :
    exact47761RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47761 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43310⟩⟩) exact47761RawTerms .large 47759 .exactZero (none)

def event47762 : Event := .preFoldPolynomial 47761 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩, (1)⟩] .exactZero none

def exact47763RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩, (1)⟩]

def event47763 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨43310⟩⟩) 47762 exact47763RawTerms .large 47759 .exactZero (none)

def event47764 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨44391⟩⟩)

def event47765 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event47766 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event47767 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.authority (.operator))

def event47768 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11115⟩⟩) (.finite 17)

def event47769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event47770 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event47771 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event47772 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event47773 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 47772

def event47774 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 47770

def event47775 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 47773 .coefficient) (.value (.predecessor 1 47774 .coefficient)))

def event47776 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event47777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 0 ⟨392⟩ 47776

def event47778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11117⟩⟩) 1 ⟨11115⟩ 47768

def event47779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.sum [.predecessor 0 47777 .coefficient, .predecessor 1 47778 .coefficient])

def event47780 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11117⟩⟩) (.finite 655357)

def event47781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 0 ⟨11117⟩ 47780

def event47782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11173⟩⟩) 1 ⟨5426⟩ 47766

def event47783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.identity (.predecessor 1 47782 .coefficient))

def event47784 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨11173⟩⟩) (.finite 655360)

def event47785 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42666⟩⟩) 0 ⟨11173⟩ 47784

def event47786 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42666⟩⟩) (.authority (.programFamilyFact))

def exact47787RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42666⟩⟩], []⟩, (1)⟩]

theorem exact47787RawTermsValid :
    exact47787RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47787 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42666⟩⟩) exact47787RawTerms (.finite 52) 47786 .exactZero (none)

def event47788 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14601⟩⟩) 0 ⟨11173⟩ 47784

def event47789 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14601⟩⟩) (.authority (.programFamilyFact))

def exact47790RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩], []⟩, (1)⟩]

theorem exact47790RawTermsValid :
    exact47790RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47790 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14601⟩⟩) exact47790RawTerms (.finite 52) 47789 .exactZero (none)

def event47791 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42667⟩⟩) 0 ⟨14601⟩ 47790

def event47792 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42667⟩⟩) 1 ⟨42666⟩ 47787

def event47793 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42667⟩⟩) (.product (.predecessor 0 47791 .coefficient) (.predecessor 1 47792 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event47794 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42667⟩⟩, .operator (⟨47790, 0⟩, ⟨47787, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], []⟩, (1)⟩)

def exact47795RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], []⟩, (1)⟩]

theorem exact47795RawTermsValid :
    exact47795RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47795 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42667⟩⟩) exact47795RawTerms (.finite 2704) 47793 .exactZero (none)

def event47796 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42668⟩⟩) 0 ⟨42667⟩ 47795

def event47797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42668⟩⟩) (.identity (.predecessor 0 47796 .coefficient))

def event47798 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42668⟩⟩) (.finite 2704)

def event47799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43836⟩⟩) 0 ⟨42668⟩ 47798

def event47800 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43836⟩⟩) (.authority (.programFamilyFact))

def event47801 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨43836⟩⟩) (.finite 3720)

def event47802 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event47803 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43837⟩⟩) 0 ⟨7177⟩ 47802

def event47804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43837⟩⟩) 1 ⟨43836⟩ 47801

def event47805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43837⟩⟩) (.authority (.operator))

def exact47806RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43837⟩⟩]⟩, (1)⟩]

theorem exact47806RawTermsValid :
    exact47806RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47806 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43837⟩⟩) exact47806RawTerms .large 47805 .exactZero (none)

def event47807 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44387⟩⟩) 0 ⟨43837⟩ 47806

def event47808 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44387⟩⟩) (.authority (.operator))

def exact47809RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩, (1)⟩]

theorem exact47809RawTermsValid :
    exact47809RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47809 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44387⟩⟩) exact47809RawTerms (.finite 8192) 47808 .exactZero (none)

def event47810 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event47811 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event47812 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44098⟩⟩) 0 ⟨42668⟩ 47798

def event47813 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44098⟩⟩) 1 ⟨136⟩ 47811

def event47814 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44098⟩⟩) (.sum [.predecessor 0 47812 .coefficient, .predecessor 1 47813 .coefficient])

def event47815 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44098⟩⟩) (.finite 2704)

def event47816 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44099⟩⟩) 0 ⟨44098⟩ 47815

def event47817 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44099⟩⟩) (.identity (.predecessor 0 47816 .coefficient))

def exact47818RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], []⟩, (1)⟩]

theorem exact47818RawTermsValid :
    exact47818RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47818 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44099⟩⟩) exact47818RawTerms (.finite 2704) 47817 .exactZero (none)

def event47819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact47820RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact47820RawTermsValid :
    exact47820RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47820 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact47820RawTerms .large 47819 .exactZero (none)

def event47821 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44100⟩⟩) 0 ⟨6908⟩ 47820

def event47822 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44100⟩⟩) 1 ⟨44099⟩ 47818

def event47823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44100⟩⟩) (.product (.predecessor 0 47821 .coefficient) (.predecessor 1 47822 .coefficient) (⟨false, false, none, none, none⟩))

def event47824 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44100⟩⟩, .operator (⟨47820, 0⟩, ⟨47818, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact47825RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact47825RawTermsValid :
    exact47825RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47825 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44100⟩⟩) exact47825RawTerms .large 47823 .exactZero (none)

def event47826 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event47827 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event47828 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 47802

def event47829 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact47830RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact47830RawTermsValid :
    exact47830RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47830 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact47830RawTerms .large 47829 .exactZero (none)

def event47831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7283⟩⟩) 0 ⟨7178⟩ 47830

def event47832 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7283⟩⟩) (.identity (.predecessor 0 47831 .coefficient))

def exact47833RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩]

theorem exact47833RawTermsValid :
    exact47833RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47833 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7283⟩⟩) exact47833RawTerms .large 47832 .exactZero (none)

def event47834 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9559⟩⟩) 0 ⟨7283⟩ 47833

def event47835 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9559⟩⟩) (.authority (.operator))

def exact47836RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩]

theorem exact47836RawTermsValid :
    exact47836RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47836 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9559⟩⟩) exact47836RawTerms (.finite 8192) 47835 .exactZero (none)

def event47837 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9560⟩⟩) 0 ⟨9559⟩ 47836

def event47838 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9560⟩⟩) 1 ⟨2370⟩ 47827

def event47839 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9560⟩⟩) (.scale (.predecessor 0 47837 .coefficient) (.value (.predecessor 1 47838 .coefficient)))

def exact47840RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩]

theorem exact47840RawTermsValid :
    exact47840RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47840 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9560⟩⟩) exact47840RawTerms (.finite 8192) 47839 .exactZero (none)

def event47841 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7300⟩⟩) 0 ⟨7178⟩ 47830

def event47842 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7300⟩⟩) (.identity (.predecessor 0 47841 .coefficient))

def exact47843RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩]⟩, (1)⟩]

theorem exact47843RawTermsValid :
    exact47843RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47843 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7300⟩⟩) exact47843RawTerms .large 47842 .exactZero (none)

def event47844 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9561⟩⟩) 0 ⟨7300⟩ 47843

def event47845 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9561⟩⟩) 1 ⟨9560⟩ 47840

def event47846 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9561⟩⟩) (.product (.predecessor 0 47844 .coefficient) (.predecessor 1 47845 .coefficient) (⟨false, false, none, none, none⟩))

def event47847 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9561⟩⟩, .operator (⟨47843, 0⟩, ⟨47840, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩)

def exact47848RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩]

theorem exact47848RawTermsValid :
    exact47848RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47848 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9561⟩⟩) exact47848RawTerms .large 47846 .exactZero (none)

def event47849 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44101⟩⟩) 0 ⟨9561⟩ 47848

def event47850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44101⟩⟩) 1 ⟨44100⟩ 47825

def event47851 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44101⟩⟩) (.sum [.predecessor 0 47849 .coefficient, .predecessor 1 47850 .coefficient])

def exact47852RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact47852RawTermsValid :
    exact47852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47852 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44101⟩⟩) exact47852RawTerms .large 47851 .exactZero (none)

def event47853 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44390⟩⟩) 0 ⟨44101⟩ 47852

def event47854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44390⟩⟩) 1 ⟨44387⟩ 47809

def event47855 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44390⟩⟩) (.product (.predecessor 0 47853 .coefficient) (.predecessor 1 47854 .coefficient) (⟨false, false, none, none, none⟩))

def event47856 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44390⟩⟩, .operator (⟨47852, 0⟩, ⟨47809, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩, (1)⟩)

def event47857 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44390⟩⟩, .operator (⟨47852, 1⟩, ⟨47809, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩, (-1)⟩)

def event47858 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44390⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44387⟩⟩) ⟨43837⟩ 47806)

def event47859 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44390⟩⟩, .relation 47858 0, ⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨43837⟩⟩]⟩, (-1)⟩)

def exact47860RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨43837⟩⟩]⟩, (-1)⟩]

theorem exact47860RawTermsValid :
    exact47860RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47860 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44390⟩⟩) exact47860RawTerms .large 47855 .exactZero (none)

def event47861 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42852⟩⟩) 0 ⟨42668⟩ 47798

def event47862 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42852⟩⟩) (.authority (.programFamilyFact))

def exact47863RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], []⟩, (1)⟩]

theorem exact47863RawTermsValid :
    exact47863RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47863 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42852⟩⟩) exact47863RawTerms (.finite 52) 47862 .exactZero (none)

def event47864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42854⟩⟩) 0 ⟨6908⟩ 47820

def event47865 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42854⟩⟩) 1 ⟨42852⟩ 47863

def event47866 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42854⟩⟩) (.product (.predecessor 0 47864 .coefficient) (.predecessor 1 47865 .coefficient) (⟨false, true, none, none, some 1⟩))

def event47867 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42854⟩⟩, .operator (⟨47820, 0⟩, ⟨47863, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact47868RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact47868RawTermsValid :
    exact47868RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47868 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42854⟩⟩) exact47868RawTerms .large 47866 .exactZero (none)

def event47869 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7194⟩⟩) 0 ⟨7177⟩ 47802

def event47870 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7194⟩⟩) (.authority (.operator))

def exact47871RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩]

theorem exact47871RawTermsValid :
    exact47871RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event47871 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7194⟩⟩) exact47871RawTerms .large 47870 .exactZero (none)

def eventLeaf2976 : Array AnnotatedEvent := #[
  { event := event47616
    frameStart := 0 },
  { event := event47617
    frameStart := 0 },
  { event := event47618
    frameStart := 0 },
  { event := event47619
    frameStart := 0 },
  { event := event47620
    frameStart := 0 },
  { event := event47621
    frameStart := 0 },
  { event := event47622
    frameStart := 0 },
  { event := event47623
    frameStart := 0 },
  { event := event47624
    frameStart := 0 },
  { event := event47625
    frameStart := 0 },
  { event := event47626
    frameStart := 0 },
  { event := event47627
    frameStart := 0 },
  { event := event47628
    frameStart := 0 },
  { event := event47629
    frameStart := 0 },
  { event := event47630
    frameStart := 0 },
  { event := event47631
    frameStart := 0 }
]

def eventLeaf2977 : Array AnnotatedEvent := #[
  { event := event47632
    frameStart := 0 },
  { event := event47633
    frameStart := 0 },
  { event := event47634
    frameStart := 0 },
  { event := event47635
    frameStart := 0 },
  { event := event47636
    frameStart := 0 },
  { event := event47637
    frameStart := 0 },
  { event := event47638
    frameStart := 0 },
  { event := event47639
    frameStart := 0 },
  { event := event47640
    frameStart := 0 },
  { event := event47641
    frameStart := 0 },
  { event := event47642
    frameStart := 0 },
  { event := event47643
    frameStart := 0 },
  { event := event47644
    frameStart := 0 },
  { event := event47645
    frameStart := 0 },
  { event := event47646
    frameStart := 0 },
  { event := event47647
    frameStart := 0 }
]

def eventLeaf2978 : Array AnnotatedEvent := #[
  { event := event47648
    frameStart := 0 },
  { event := event47649
    frameStart := 0 },
  { event := event47650
    frameStart := 0 },
  { event := event47651
    frameStart := 0 },
  { event := event47652
    frameStart := 0 },
  { event := event47653
    frameStart := 0 },
  { event := event47654
    frameStart := 0 },
  { event := event47655
    frameStart := 0 },
  { event := event47656
    frameStart := 0 },
  { event := event47657
    frameStart := 0 },
  { event := event47658
    frameStart := 0 },
  { event := event47659
    frameStart := 0 },
  { event := event47660
    frameStart := 0 },
  { event := event47661
    frameStart := 0 },
  { event := event47662
    frameStart := 0 },
  { event := event47663
    frameStart := 0 }
]

def eventLeaf2979 : Array AnnotatedEvent := #[
  { event := event47664
    frameStart := 0 },
  { event := event47665
    frameStart := 0 },
  { event := event47666
    frameStart := 0 },
  { event := event47667
    frameStart := 0 },
  { event := event47668
    frameStart := 0 },
  { event := event47669
    frameStart := 0 },
  { event := event47670
    frameStart := 0 },
  { event := event47671
    frameStart := 0 },
  { event := event47672
    frameStart := 0 },
  { event := event47673
    frameStart := 0 },
  { event := event47674
    frameStart := 0 },
  { event := event47675
    frameStart := 0 },
  { event := event47676
    frameStart := 0 },
  { event := event47677
    frameStart := 0 },
  { event := event47678
    frameStart := 0 },
  { event := event47679
    frameStart := 0 }
]

def eventLeaf2980 : Array AnnotatedEvent := #[
  { event := event47680
    frameStart := 0 },
  { event := event47681
    frameStart := 0 },
  { event := event47682
    frameStart := 0 },
  { event := event47683
    frameStart := 0 },
  { event := event47684
    frameStart := 0 },
  { event := event47685
    frameStart := 0 },
  { event := event47686
    frameStart := 0 },
  { event := event47687
    frameStart := 0 },
  { event := event47688
    frameStart := 0 },
  { event := event47689
    frameStart := 0 },
  { event := event47690
    frameStart := 0 },
  { event := event47691
    frameStart := 0 },
  { event := event47692
    frameStart := 0 },
  { event := event47693
    frameStart := 0 },
  { event := event47694
    frameStart := 0 },
  { event := event47695
    frameStart := 0 }
]

def eventLeaf2981 : Array AnnotatedEvent := #[
  { event := event47696
    frameStart := 0 },
  { event := event47697
    frameStart := 0 },
  { event := event47698
    frameStart := 0 },
  { event := event47699
    frameStart := 0 },
  { event := event47700
    frameStart := 0 },
  { event := event47701
    frameStart := 0 },
  { event := event47702
    frameStart := 0 },
  { event := event47703
    frameStart := 0 },
  { event := event47704
    frameStart := 0 },
  { event := event47705
    frameStart := 0 },
  { event := event47706
    frameStart := 0 },
  { event := event47707
    frameStart := 0 },
  { event := event47708
    frameStart := 0 },
  { event := event47709
    frameStart := 0 },
  { event := event47710
    frameStart := 0 },
  { event := event47711
    frameStart := 0 }
]

def eventLeaf2982 : Array AnnotatedEvent := #[
  { event := event47712
    frameStart := 0 },
  { event := event47713
    frameStart := 0 },
  { event := event47714
    frameStart := 0 },
  { event := event47715
    frameStart := 0 },
  { event := event47716
    frameStart := 47716 },
  { event := event47717
    frameStart := 47716 },
  { event := event47718
    frameStart := 47716 },
  { event := event47719
    frameStart := 47716 },
  { event := event47720
    frameStart := 47716 },
  { event := event47721
    frameStart := 47716 },
  { event := event47722
    frameStart := 47716 },
  { event := event47723
    frameStart := 47716 },
  { event := event47724
    frameStart := 47716 },
  { event := event47725
    frameStart := 47716 },
  { event := event47726
    frameStart := 47716 },
  { event := event47727
    frameStart := 47716 }
]

def eventLeaf2983 : Array AnnotatedEvent := #[
  { event := event47728
    frameStart := 47716 },
  { event := event47729
    frameStart := 47716 },
  { event := event47730
    frameStart := 47716 },
  { event := event47731
    frameStart := 47716 },
  { event := event47732
    frameStart := 47716 },
  { event := event47733
    frameStart := 47716 },
  { event := event47734
    frameStart := 47716 },
  { event := event47735
    frameStart := 47716 },
  { event := event47736
    frameStart := 47716 },
  { event := event47737
    frameStart := 47716 },
  { event := event47738
    frameStart := 47716 },
  { event := event47739
    frameStart := 47716 },
  { event := event47740
    frameStart := 47716 },
  { event := event47741
    frameStart := 47716 },
  { event := event47742
    frameStart := 47716 },
  { event := event47743
    frameStart := 47716 }
]

def eventLeaf2984 : Array AnnotatedEvent := #[
  { event := event47744
    frameStart := 47716 },
  { event := event47745
    frameStart := 47716 },
  { event := event47746
    frameStart := 47716 },
  { event := event47747
    frameStart := 47716 },
  { event := event47748
    frameStart := 47716 },
  { event := event47749
    frameStart := 47716 },
  { event := event47750
    frameStart := 47716 },
  { event := event47751
    frameStart := 47716 },
  { event := event47752
    frameStart := 47716 },
  { event := event47753
    frameStart := 47716 },
  { event := event47754
    frameStart := 47716 },
  { event := event47755
    frameStart := 47716 },
  { event := event47756
    frameStart := 47716 },
  { event := event47757
    frameStart := 47716 },
  { event := event47758
    frameStart := 47716 },
  { event := event47759
    frameStart := 47716 }
]

def eventLeaf2985 : Array AnnotatedEvent := #[
  { event := event47760
    frameStart := 47716 },
  { event := event47761
    frameStart := 47716 },
  { event := event47762
    frameStart := 47716 },
  { event := event47763
    frameStart := 47716 },
  { event := event47764
    frameStart := 47764 },
  { event := event47765
    frameStart := 47764 },
  { event := event47766
    frameStart := 47764 },
  { event := event47767
    frameStart := 47764 },
  { event := event47768
    frameStart := 47764 },
  { event := event47769
    frameStart := 47764 },
  { event := event47770
    frameStart := 47764 },
  { event := event47771
    frameStart := 47764 },
  { event := event47772
    frameStart := 47764 },
  { event := event47773
    frameStart := 47764 },
  { event := event47774
    frameStart := 47764 },
  { event := event47775
    frameStart := 47764 }
]

def eventLeaf2986 : Array AnnotatedEvent := #[
  { event := event47776
    frameStart := 47764 },
  { event := event47777
    frameStart := 47764 },
  { event := event47778
    frameStart := 47764 },
  { event := event47779
    frameStart := 47764 },
  { event := event47780
    frameStart := 47764 },
  { event := event47781
    frameStart := 47764 },
  { event := event47782
    frameStart := 47764 },
  { event := event47783
    frameStart := 47764 },
  { event := event47784
    frameStart := 47764 },
  { event := event47785
    frameStart := 47764 },
  { event := event47786
    frameStart := 47764 },
  { event := event47787
    frameStart := 47764 },
  { event := event47788
    frameStart := 47764 },
  { event := event47789
    frameStart := 47764 },
  { event := event47790
    frameStart := 47764 },
  { event := event47791
    frameStart := 47764 }
]

def eventLeaf2987 : Array AnnotatedEvent := #[
  { event := event47792
    frameStart := 47764 },
  { event := event47793
    frameStart := 47764 },
  { event := event47794
    frameStart := 47764 },
  { event := event47795
    frameStart := 47764 },
  { event := event47796
    frameStart := 47764 },
  { event := event47797
    frameStart := 47764 },
  { event := event47798
    frameStart := 47764 },
  { event := event47799
    frameStart := 47764 },
  { event := event47800
    frameStart := 47764 },
  { event := event47801
    frameStart := 47764 },
  { event := event47802
    frameStart := 47764 },
  { event := event47803
    frameStart := 47764 },
  { event := event47804
    frameStart := 47764 },
  { event := event47805
    frameStart := 47764 },
  { event := event47806
    frameStart := 47764 },
  { event := event47807
    frameStart := 47764 }
]

def eventLeaf2988 : Array AnnotatedEvent := #[
  { event := event47808
    frameStart := 47764 },
  { event := event47809
    frameStart := 47764 },
  { event := event47810
    frameStart := 47764 },
  { event := event47811
    frameStart := 47764 },
  { event := event47812
    frameStart := 47764 },
  { event := event47813
    frameStart := 47764 },
  { event := event47814
    frameStart := 47764 },
  { event := event47815
    frameStart := 47764 },
  { event := event47816
    frameStart := 47764 },
  { event := event47817
    frameStart := 47764 },
  { event := event47818
    frameStart := 47764 },
  { event := event47819
    frameStart := 47764 },
  { event := event47820
    frameStart := 47764 },
  { event := event47821
    frameStart := 47764 },
  { event := event47822
    frameStart := 47764 },
  { event := event47823
    frameStart := 47764 }
]

def eventLeaf2989 : Array AnnotatedEvent := #[
  { event := event47824
    frameStart := 47764 },
  { event := event47825
    frameStart := 47764 },
  { event := event47826
    frameStart := 47764 },
  { event := event47827
    frameStart := 47764 },
  { event := event47828
    frameStart := 47764 },
  { event := event47829
    frameStart := 47764 },
  { event := event47830
    frameStart := 47764 },
  { event := event47831
    frameStart := 47764 },
  { event := event47832
    frameStart := 47764 },
  { event := event47833
    frameStart := 47764 },
  { event := event47834
    frameStart := 47764 },
  { event := event47835
    frameStart := 47764 },
  { event := event47836
    frameStart := 47764 },
  { event := event47837
    frameStart := 47764 },
  { event := event47838
    frameStart := 47764 },
  { event := event47839
    frameStart := 47764 }
]

def eventLeaf2990 : Array AnnotatedEvent := #[
  { event := event47840
    frameStart := 47764 },
  { event := event47841
    frameStart := 47764 },
  { event := event47842
    frameStart := 47764 },
  { event := event47843
    frameStart := 47764 },
  { event := event47844
    frameStart := 47764 },
  { event := event47845
    frameStart := 47764 },
  { event := event47846
    frameStart := 47764 },
  { event := event47847
    frameStart := 47764 },
  { event := event47848
    frameStart := 47764 },
  { event := event47849
    frameStart := 47764 },
  { event := event47850
    frameStart := 47764 },
  { event := event47851
    frameStart := 47764 },
  { event := event47852
    frameStart := 47764 },
  { event := event47853
    frameStart := 47764 },
  { event := event47854
    frameStart := 47764 },
  { event := event47855
    frameStart := 47764 }
]

def eventLeaf2991 : Array AnnotatedEvent := #[
  { event := event47856
    frameStart := 47764 },
  { event := event47857
    frameStart := 47764 },
  { event := event47858
    frameStart := 47764 },
  { event := event47859
    frameStart := 47764 },
  { event := event47860
    frameStart := 47764 },
  { event := event47861
    frameStart := 47764 },
  { event := event47862
    frameStart := 47764 },
  { event := event47863
    frameStart := 47764 },
  { event := event47864
    frameStart := 47764 },
  { event := event47865
    frameStart := 47764 },
  { event := event47866
    frameStart := 47764 },
  { event := event47867
    frameStart := 47764 },
  { event := event47868
    frameStart := 47764 },
  { event := event47869
    frameStart := 47764 },
  { event := event47870
    frameStart := 47764 },
  { event := event47871
    frameStart := 47764 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events186
