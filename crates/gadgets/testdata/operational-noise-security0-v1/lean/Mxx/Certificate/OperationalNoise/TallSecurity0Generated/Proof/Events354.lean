import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events354

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event90624 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22480⟩⟩) (.authority (.relationPreimageSource ⟨60⟩))

def exact90625RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22480⟩⟩]⟩, (1)⟩]

theorem exact90625RawTermsValid :
    exact90625RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90625 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22480⟩⟩) exact90625RawTerms (.finite 136065468) 90624 .exactZero (none)

def event90626 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22482⟩⟩) 0 ⟨22480⟩ 90625

def event90627 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22482⟩⟩) 1 ⟨2348⟩ 4

def event90628 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22482⟩⟩) (.scale (.predecessor 0 90626 .coefficient) (.value (.predecessor 1 90627 .coefficient)))

def exact90629RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22480⟩⟩]⟩, (1)⟩]

theorem exact90629RawTermsValid :
    exact90629RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90629 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22482⟩⟩) exact90629RawTerms (.finite 136065468) 90628 .exactZero (none)

def event90630 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22483⟩⟩) 0 ⟨5541⟩ 80012

def event90631 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22483⟩⟩) 1 ⟨22482⟩ 90629

def event90632 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22483⟩⟩) (.product (.predecessor 0 90630 .coefficient) (.predecessor 1 90631 .coefficient) (⟨false, false, none, none, none⟩))

def event90633 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22483⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22480⟩⟩]⟩) [⟨.result 90625 .coefficient, false, none⟩])

def event90634 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22483⟩⟩) (.product (.result 80012 .summary) (.transfer 90633) (⟨false, false, none, none, none⟩))

def event90635 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22483⟩⟩, .operator (⟨80012, 0⟩, ⟨90629, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22480⟩⟩]⟩, (1)⟩)

def event90636 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22481⟩⟩)

def event90637 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event90638 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event90639 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event90640 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event90641 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event90642 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event90643 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event90644 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event90645 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 90644

def event90646 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 90642

def event90647 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 90645 .coefficient) (.value (.predecessor 1 90646 .coefficient)))

def event90648 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event90649 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 90648

def event90650 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 90640

def event90651 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 90649 .coefficient, .predecessor 1 90650 .coefficient])

def event90652 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event90653 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 90652

def event90654 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 90638

def event90655 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 90654 .coefficient))

def event90656 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event90657 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12958⟩⟩) 0 ⟨5536⟩ 90656

def event90658 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12958⟩⟩) (.authority (.programFamilyFact))

def exact90659RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12958⟩⟩], []⟩, (1)⟩]

theorem exact90659RawTermsValid :
    exact90659RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90659 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12958⟩⟩) exact90659RawTerms (.finite 52) 90658 .exactZero (none)

def event90660 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10135⟩⟩) 0 ⟨5536⟩ 90656

def event90661 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10135⟩⟩) (.authority (.programFamilyFact))

def exact90662RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩], []⟩, (1)⟩]

theorem exact90662RawTermsValid :
    exact90662RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90662 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10135⟩⟩) exact90662RawTerms (.finite 52) 90661 .exactZero (none)

def event90663 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12959⟩⟩) 0 ⟨10135⟩ 90662

def event90664 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12959⟩⟩) 1 ⟨12958⟩ 90659

def event90665 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12959⟩⟩) (.product (.predecessor 0 90663 .coefficient) (.predecessor 1 90664 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event90666 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12959⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], []⟩) [⟨.result 90662 .coefficient, true, some 1⟩, ⟨.result 90659 .coefficient, true, some 1⟩])

def event90667 : Event := .survivorFold (1) 90666

def exact90668RawTerms : List Term := []

theorem exact90668RawTermsValid :
    exact90668RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90668 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12959⟩⟩) exact90668RawTerms (.finite 2704) 90665 (.finite 2704) (some (90666))

def event90669 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12960⟩⟩) 0 ⟨12959⟩ 90668

def event90670 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12960⟩⟩) (.identity (.predecessor 0 90669 .coefficient))

def event90671 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12960⟩⟩) (.finite 2704)

def event90672 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16752⟩⟩) 0 ⟨12960⟩ 90671

def event90673 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16752⟩⟩) (.authority (.programFamilyFact))

def exact90674RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], []⟩, (1)⟩]

theorem exact90674RawTermsValid :
    exact90674RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90674 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16752⟩⟩) exact90674RawTerms (.finite 52) 90673 .exactZero (none)

def event90675 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16753⟩⟩) 0 ⟨16752⟩ 90674

def event90676 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16753⟩⟩) (.identity (.predecessor 0 90675 .coefficient))

def event90677 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16753⟩⟩) (.finite 52)

def event90678 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22480⟩⟩) 0 ⟨16753⟩ 90677

def event90679 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22480⟩⟩) (.authority (.relationPreimageSource ⟨60⟩))

def exact90680RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22480⟩⟩]⟩, (1)⟩]

theorem exact90680RawTermsValid :
    exact90680RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90680 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22480⟩⟩) exact90680RawTerms (.finite 136065468) 90679 .exactZero (none)

def event90681 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact90682RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact90682RawTermsValid :
    exact90682RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90682 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact90682RawTerms .large 90681 .exactZero (none)

def event90683 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22481⟩⟩) 0 ⟨6⟩ 90682

def event90684 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22481⟩⟩) 1 ⟨22480⟩ 90680

def event90685 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22481⟩⟩) (.product (.predecessor 0 90683 .coefficient) (.predecessor 1 90684 .coefficient) (⟨false, false, none, none, none⟩))

def event90686 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22481⟩⟩, .operator (⟨90682, 0⟩, ⟨90680, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22480⟩⟩]⟩, (1)⟩)

def exact90687RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22480⟩⟩]⟩, (1)⟩]

theorem exact90687RawTermsValid :
    exact90687RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90687 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22481⟩⟩) exact90687RawTerms .large 90685 .exactZero (none)

def event90688 : Event := .preFoldPolynomial 90687 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22480⟩⟩]⟩, (1)⟩] .exactZero none

def exact90689RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22480⟩⟩]⟩, (1)⟩]

def event90689 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨22481⟩⟩) 90688 exact90689RawTerms .large 90685 .exactZero (none)

def event90690 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨29601⟩⟩)

def event90691 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event90692 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event90693 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event90694 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event90695 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event90696 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event90697 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event90698 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event90699 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 90698

def event90700 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 90696

def event90701 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 90699 .coefficient) (.value (.predecessor 1 90700 .coefficient)))

def event90702 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event90703 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 90702

def event90704 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 90694

def event90705 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 90703 .coefficient, .predecessor 1 90704 .coefficient])

def event90706 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event90707 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 90706

def event90708 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 90692

def event90709 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 90708 .coefficient))

def event90710 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event90711 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12958⟩⟩) 0 ⟨5536⟩ 90710

def event90712 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12958⟩⟩) (.authority (.programFamilyFact))

def exact90713RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12958⟩⟩], []⟩, (1)⟩]

theorem exact90713RawTermsValid :
    exact90713RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90713 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12958⟩⟩) exact90713RawTerms (.finite 52) 90712 .exactZero (none)

def event90714 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10135⟩⟩) 0 ⟨5536⟩ 90710

def event90715 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10135⟩⟩) (.authority (.programFamilyFact))

def exact90716RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩], []⟩, (1)⟩]

theorem exact90716RawTermsValid :
    exact90716RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90716 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10135⟩⟩) exact90716RawTerms (.finite 52) 90715 .exactZero (none)

def event90717 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12959⟩⟩) 0 ⟨10135⟩ 90716

def event90718 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12959⟩⟩) 1 ⟨12958⟩ 90713

def event90719 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12959⟩⟩) (.product (.predecessor 0 90717 .coefficient) (.predecessor 1 90718 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event90720 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12959⟩⟩, .operator (⟨90716, 0⟩, ⟨90713, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], []⟩, (1)⟩)

def exact90721RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10135⟩⟩, ⟨.program ⟨214⟩, ⟨12958⟩⟩], []⟩, (1)⟩]

theorem exact90721RawTermsValid :
    exact90721RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90721 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12959⟩⟩) exact90721RawTerms (.finite 2704) 90719 .exactZero (none)

def event90722 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12960⟩⟩) 0 ⟨12959⟩ 90721

def event90723 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12960⟩⟩) (.identity (.predecessor 0 90722 .coefficient))

def event90724 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12960⟩⟩) (.finite 2704)

def event90725 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16752⟩⟩) 0 ⟨12960⟩ 90724

def event90726 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16752⟩⟩) (.authority (.programFamilyFact))

def exact90727RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], []⟩, (1)⟩]

theorem exact90727RawTermsValid :
    exact90727RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90727 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16752⟩⟩) exact90727RawTerms (.finite 52) 90726 .exactZero (none)

def event90728 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16753⟩⟩) 0 ⟨16752⟩ 90727

def event90729 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16753⟩⟩) (.identity (.predecessor 0 90728 .coefficient))

def event90730 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16753⟩⟩) (.finite 52)

def event90731 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24664⟩⟩) 0 ⟨16753⟩ 90730

def event90732 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24664⟩⟩) (.authority (.programFamilyFact))

def event90733 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24664⟩⟩) (.finite 3720)

def event90734 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event90735 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24665⟩⟩) 0 ⟨6689⟩ 90734

def event90736 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24665⟩⟩) 1 ⟨24664⟩ 90733

def event90737 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24665⟩⟩) (.authority (.operator))

def exact90738RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24665⟩⟩]⟩, (1)⟩]

theorem exact90738RawTermsValid :
    exact90738RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90738 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24665⟩⟩) exact90738RawTerms .large 90737 .exactZero (none)

def event90739 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29595⟩⟩) 0 ⟨24665⟩ 90738

def event90740 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29595⟩⟩) (.authority (.operator))

def exact90741RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29595⟩⟩]⟩, (1)⟩]

theorem exact90741RawTermsValid :
    exact90741RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90741 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29595⟩⟩) exact90741RawTerms (.finite 8192) 90740 .exactZero (none)

def event90742 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event90743 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event90744 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16827⟩⟩) 0 ⟨16753⟩ 90730

def event90745 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16827⟩⟩) 1 ⟨110⟩ 90743

def event90746 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16827⟩⟩) (.sum [.predecessor 0 90744 .coefficient, .predecessor 1 90745 .coefficient])

def event90747 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16827⟩⟩) (.finite 52)

def event90748 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16828⟩⟩) 0 ⟨16827⟩ 90747

def event90749 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16828⟩⟩) (.identity (.predecessor 0 90748 .coefficient))

def exact90750RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], []⟩, (1)⟩]

theorem exact90750RawTermsValid :
    exact90750RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90750 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16828⟩⟩) exact90750RawTerms (.finite 52) 90749 .exactZero (none)

def event90751 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact90752RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact90752RawTermsValid :
    exact90752RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90752 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact90752RawTerms .large 90751 .exactZero (none)

def event90753 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16829⟩⟩) 0 ⟨6544⟩ 90752

def event90754 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16829⟩⟩) 1 ⟨16828⟩ 90750

def event90755 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16829⟩⟩) (.product (.predecessor 0 90753 .coefficient) (.predecessor 1 90754 .coefficient) (⟨false, false, none, none, none⟩))

def event90756 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16829⟩⟩, .operator (⟨90752, 0⟩, ⟨90750, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact90757RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact90757RawTermsValid :
    exact90757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90757 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16829⟩⟩) exact90757RawTerms .large 90755 .exactZero (none)

def event90758 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6705⟩⟩) 0 ⟨6689⟩ 90734

def event90759 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6705⟩⟩) (.authority (.operator))

def exact90760RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩, (1)⟩]

theorem exact90760RawTermsValid :
    exact90760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90760 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6705⟩⟩) exact90760RawTerms .large 90759 .exactZero (none)

def event90761 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16830⟩⟩) 0 ⟨6705⟩ 90760

def event90762 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16830⟩⟩) 1 ⟨16829⟩ 90757

def event90763 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16830⟩⟩) (.sum [.predecessor 0 90761 .coefficient, .predecessor 1 90762 .coefficient])

def exact90764RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact90764RawTermsValid :
    exact90764RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90764 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16830⟩⟩) exact90764RawTerms .large 90763 .exactZero (none)

def event90765 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29596⟩⟩) 0 ⟨16830⟩ 90764

def event90766 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29596⟩⟩) 1 ⟨29595⟩ 90741

def event90767 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29596⟩⟩) (.product (.predecessor 0 90765 .coefficient) (.predecessor 1 90766 .coefficient) (⟨false, false, none, none, none⟩))

def event90768 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29596⟩⟩, .operator (⟨90764, 0⟩, ⟨90741, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29595⟩⟩]⟩, (1)⟩)

def event90769 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29596⟩⟩, .operator (⟨90764, 1⟩, ⟨90741, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29595⟩⟩]⟩, (-1)⟩)

def event90770 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29596⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29595⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29595⟩⟩) ⟨24665⟩ 90738)

def event90771 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29596⟩⟩, .relation 90770 0, ⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨24665⟩⟩]⟩, (-1)⟩)

def exact90772RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29595⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨24665⟩⟩]⟩, (-1)⟩]

theorem exact90772RawTermsValid :
    exact90772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90772 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29596⟩⟩) exact90772RawTerms .large 90767 .exactZero (none)

def event90773 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17494⟩⟩) 0 ⟨16753⟩ 90730

def event90774 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17494⟩⟩) (.authority (.programFamilyFact))

def exact90775RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17494⟩⟩], []⟩, (1)⟩]

theorem exact90775RawTermsValid :
    exact90775RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90775 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17494⟩⟩) exact90775RawTerms (.finite 52) 90774 .exactZero (none)

def event90776 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17496⟩⟩) 0 ⟨6544⟩ 90752

def event90777 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17496⟩⟩) 1 ⟨17494⟩ 90775

def event90778 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17496⟩⟩) (.product (.predecessor 0 90776 .coefficient) (.predecessor 1 90777 .coefficient) (⟨false, true, none, none, some 1⟩))

def event90779 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17496⟩⟩, .operator (⟨90752, 0⟩, ⟨90775, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17494⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact90780RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17494⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact90780RawTermsValid :
    exact90780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90780 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17496⟩⟩) exact90780RawTerms .large 90778 .exactZero (none)

def event90781 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6738⟩⟩) 0 ⟨6689⟩ 90734

def event90782 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6738⟩⟩) (.authority (.operator))

def exact90783RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩]⟩, (1)⟩]

theorem exact90783RawTermsValid :
    exact90783RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90783 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6738⟩⟩) exact90783RawTerms .large 90782 .exactZero (none)

def event90784 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17497⟩⟩) 0 ⟨6738⟩ 90783

def event90785 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17497⟩⟩) 1 ⟨17496⟩ 90780

def event90786 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17497⟩⟩) (.sum [.predecessor 0 90784 .coefficient, .predecessor 1 90785 .coefficient])

def exact90787RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17494⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact90787RawTermsValid :
    exact90787RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90787 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17497⟩⟩) exact90787RawTerms .large 90786 .exactZero (none)

def event90788 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29601⟩⟩) 0 ⟨17497⟩ 90787

def event90789 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29601⟩⟩) 1 ⟨29596⟩ 90772

def event90790 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29601⟩⟩) (.sum [.predecessor 0 90788 .coefficient, .predecessor 1 90789 .coefficient])

def exact90791RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29595⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨24665⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17494⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact90791RawTermsValid :
    exact90791RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90791 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29601⟩⟩) exact90791RawTerms .large 90790 .exactZero (none)

def event90792 : Event := .preFoldPolynomial 90791 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29595⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨24665⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17494⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact90793RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29595⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6738⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨24665⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17494⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event90793 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨29601⟩⟩) 90792 exact90793RawTerms .large 90790 .exactZero (none)

def event90794 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16753⟩⟩) ⟨⟨151⟩, ⟨60⟩, ⟨109⟩⟩ ⟨90636, 90794⟩

def event90795 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨22483⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22480⟩⟩]⟩) (1) 0 2 (.universal 90794 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22480⟩⟩]⟩) (none) 90793)

def event90796 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22483⟩⟩, .relation 90795 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6738⟩⟩]⟩, (1)⟩)

def event90797 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22483⟩⟩, .relation 90795 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29595⟩⟩]⟩, (-1)⟩)

def event90798 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22483⟩⟩, .relation 90795 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨24665⟩⟩]⟩, (1)⟩)

def event90799 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22483⟩⟩, .relation 90795 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact90800RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29595⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6738⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨24665⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact90800RawTermsValid :
    exact90800RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90800 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22483⟩⟩) exact90800RawTerms .large 90632 (.finite 1811303510016) (some (90634))

def event90801 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29598⟩⟩) 0 ⟨22483⟩ 90800

def event90802 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29598⟩⟩) 1 ⟨29597⟩ 90622

def event90803 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29598⟩⟩) (.sum [.predecessor 0 90801 .coefficient, .predecessor 1 90802 .coefficient])

def event90804 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29598⟩⟩, .operator (⟨90800, 0⟩, ⟨90622, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29595⟩⟩]⟩, (1)⟩)

def event90805 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29598⟩⟩, .operator (⟨90800, 2⟩, ⟨90622, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16752⟩⟩], [⟨.program ⟨214⟩, ⟨24665⟩⟩]⟩, (-1)⟩)

def event90806 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29598⟩⟩) (.sum [.result 90800 .summary, .result 90622 .summary])

def exact90807RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6738⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact90807RawTermsValid :
    exact90807RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90807 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29598⟩⟩) exact90807RawTerms .large 90803 (.finite 1292449485504936292352) (some (90806))

def event90808 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29599⟩⟩) 0 ⟨29598⟩ 90807

def event90809 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29599⟩⟩) 1 ⟨6662⟩ 5559

def event90810 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29599⟩⟩) (.product (.predecessor 0 90808 .coefficient) (.predecessor 1 90809 .coefficient) (⟨false, false, none, none, none⟩))

def event90811 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29599⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩) [⟨.result 5555 .coefficient, false, none⟩])

def event90812 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29599⟩⟩) (.product (.result 90807 .summary) (.transfer 90811) (⟨false, false, none, none, none⟩))

def event90813 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29599⟩⟩, .operator (⟨90807, 0⟩, ⟨5559, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩, (1)⟩)

def event90814 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29599⟩⟩, .operator (⟨90807, 1⟩, ⟨5559, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩, (-1)⟩)

def event90815 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29599⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6661⟩⟩) ⟨6602⟩ 5552)

def event90816 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29599⟩⟩, .relation 90815 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact90817RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6738⟩⟩, ⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17494⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact90817RawTermsValid :
    exact90817RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90817 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29599⟩⟩) exact90817RawTerms .large 90810 (.finite 4743310290994884271912517632) (some (90812))

def event90818 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24602⟩⟩) 0 ⟨6689⟩ 5477

def event90819 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24602⟩⟩) 1 ⟨24601⟩ 81354

def event90820 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24602⟩⟩) (.authority (.operator))

def exact90821RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24602⟩⟩]⟩, (1)⟩]

theorem exact90821RawTermsValid :
    exact90821RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90821 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24602⟩⟩) exact90821RawTerms .large 90820 .exactZero (none)

def event90822 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29378⟩⟩) 0 ⟨24602⟩ 90821

def event90823 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29378⟩⟩) (.authority (.operator))

def exact90824RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29378⟩⟩]⟩, (1)⟩]

theorem exact90824RawTermsValid :
    exact90824RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90824 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29378⟩⟩) exact90824RawTerms (.finite 8192) 90823 .exactZero (none)

def event90825 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29380⟩⟩) 0 ⟨25529⟩ 81636

def event90826 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29380⟩⟩) 1 ⟨29378⟩ 90824

def event90827 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29380⟩⟩) (.product (.predecessor 0 90825 .coefficient) (.predecessor 1 90826 .coefficient) (⟨false, false, none, none, none⟩))

def event90828 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29380⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨29378⟩⟩]⟩) [⟨.result 90824 .coefficient, false, none⟩])

def event90829 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29380⟩⟩) (.product (.result 81636 .summary) (.transfer 90828) (⟨false, false, none, none, none⟩))

def event90830 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29380⟩⟩, .operator (⟨81636, 0⟩, ⟨90824, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29378⟩⟩]⟩, (1)⟩)

def event90831 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29380⟩⟩, .operator (⟨81636, 1⟩, ⟨90824, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16633⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29378⟩⟩]⟩, (-1)⟩)

def event90832 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29380⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16633⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29378⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29378⟩⟩) ⟨24602⟩ 90821)

def event90833 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29380⟩⟩, .relation 90832 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16633⟩⟩], [⟨.program ⟨214⟩, ⟨24602⟩⟩]⟩, (-1)⟩)

def exact90834RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29378⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16633⟩⟩], [⟨.program ⟨214⟩, ⟨24602⟩⟩]⟩, (-1)⟩]

theorem exact90834RawTermsValid :
    exact90834RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90834 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29380⟩⟩) exact90834RawTerms .large 90827 (.finite 1292382246358571024384) (some (90829))

def event90835 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22336⟩⟩) 0 ⟨16634⟩ 3914

def event90836 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22336⟩⟩) (.authority (.relationPreimageSource ⟨58⟩))

def exact90837RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22336⟩⟩]⟩, (1)⟩]

theorem exact90837RawTermsValid :
    exact90837RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90837 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22336⟩⟩) exact90837RawTerms (.finite 136065468) 90836 .exactZero (none)

def event90838 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22338⟩⟩) 0 ⟨22336⟩ 90837

def event90839 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22338⟩⟩) 1 ⟨2348⟩ 4

def event90840 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22338⟩⟩) (.scale (.predecessor 0 90838 .coefficient) (.value (.predecessor 1 90839 .coefficient)))

def exact90841RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22336⟩⟩]⟩, (1)⟩]

theorem exact90841RawTermsValid :
    exact90841RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90841 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22338⟩⟩) exact90841RawTerms (.finite 136065468) 90840 .exactZero (none)

def event90842 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22339⟩⟩) 0 ⟨5541⟩ 80012

def event90843 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22339⟩⟩) 1 ⟨22338⟩ 90841

def event90844 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22339⟩⟩) (.product (.predecessor 0 90842 .coefficient) (.predecessor 1 90843 .coefficient) (⟨false, false, none, none, none⟩))

def event90845 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22339⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22336⟩⟩]⟩) [⟨.result 90837 .coefficient, false, none⟩])

def event90846 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22339⟩⟩) (.product (.result 80012 .summary) (.transfer 90845) (⟨false, false, none, none, none⟩))

def event90847 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22339⟩⟩, .operator (⟨80012, 0⟩, ⟨90841, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22336⟩⟩]⟩, (1)⟩)

def event90848 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22337⟩⟩)

def event90849 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event90850 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event90851 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event90852 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event90853 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event90854 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event90855 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event90856 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event90857 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 90856

def event90858 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 90854

def event90859 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 90857 .coefficient) (.value (.predecessor 1 90858 .coefficient)))

def event90860 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event90861 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 90860

def event90862 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 90852

def event90863 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 90861 .coefficient, .predecessor 1 90862 .coefficient])

def event90864 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event90865 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 90864

def event90866 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 90850

def event90867 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 90866 .coefficient))

def event90868 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event90869 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12762⟩⟩) 0 ⟨5536⟩ 90868

def event90870 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12762⟩⟩) (.authority (.programFamilyFact))

def exact90871RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12762⟩⟩], []⟩, (1)⟩]

theorem exact90871RawTermsValid :
    exact90871RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90871 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12762⟩⟩) exact90871RawTerms (.finite 46) 90870 .exactZero (none)

def event90872 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10030⟩⟩) 0 ⟨5536⟩ 90868

def event90873 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10030⟩⟩) (.authority (.programFamilyFact))

def exact90874RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10030⟩⟩], []⟩, (1)⟩]

theorem exact90874RawTermsValid :
    exact90874RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90874 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10030⟩⟩) exact90874RawTerms (.finite 46) 90873 .exactZero (none)

def event90875 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12763⟩⟩) 0 ⟨10030⟩ 90874

def event90876 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12763⟩⟩) 1 ⟨12762⟩ 90871

def event90877 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12763⟩⟩) (.product (.predecessor 0 90875 .coefficient) (.predecessor 1 90876 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event90878 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12763⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], []⟩) [⟨.result 90874 .coefficient, true, some 1⟩, ⟨.result 90871 .coefficient, true, some 1⟩])

def event90879 : Event := .survivorFold (1) 90878

def eventLeaf5664 : Array AnnotatedEvent := #[
  { event := event90624
    frameStart := 0 },
  { event := event90625
    frameStart := 0 },
  { event := event90626
    frameStart := 0 },
  { event := event90627
    frameStart := 0 },
  { event := event90628
    frameStart := 0 },
  { event := event90629
    frameStart := 0 },
  { event := event90630
    frameStart := 0 },
  { event := event90631
    frameStart := 0 },
  { event := event90632
    frameStart := 0 },
  { event := event90633
    frameStart := 0 },
  { event := event90634
    frameStart := 0 },
  { event := event90635
    frameStart := 0 },
  { event := event90636
    frameStart := 90636 },
  { event := event90637
    frameStart := 90636 },
  { event := event90638
    frameStart := 90636 },
  { event := event90639
    frameStart := 90636 }
]

def eventLeaf5665 : Array AnnotatedEvent := #[
  { event := event90640
    frameStart := 90636 },
  { event := event90641
    frameStart := 90636 },
  { event := event90642
    frameStart := 90636 },
  { event := event90643
    frameStart := 90636 },
  { event := event90644
    frameStart := 90636 },
  { event := event90645
    frameStart := 90636 },
  { event := event90646
    frameStart := 90636 },
  { event := event90647
    frameStart := 90636 },
  { event := event90648
    frameStart := 90636 },
  { event := event90649
    frameStart := 90636 },
  { event := event90650
    frameStart := 90636 },
  { event := event90651
    frameStart := 90636 },
  { event := event90652
    frameStart := 90636 },
  { event := event90653
    frameStart := 90636 },
  { event := event90654
    frameStart := 90636 },
  { event := event90655
    frameStart := 90636 }
]

def eventLeaf5666 : Array AnnotatedEvent := #[
  { event := event90656
    frameStart := 90636 },
  { event := event90657
    frameStart := 90636 },
  { event := event90658
    frameStart := 90636 },
  { event := event90659
    frameStart := 90636 },
  { event := event90660
    frameStart := 90636 },
  { event := event90661
    frameStart := 90636 },
  { event := event90662
    frameStart := 90636 },
  { event := event90663
    frameStart := 90636 },
  { event := event90664
    frameStart := 90636 },
  { event := event90665
    frameStart := 90636 },
  { event := event90666
    frameStart := 90636 },
  { event := event90667
    frameStart := 90636 },
  { event := event90668
    frameStart := 90636 },
  { event := event90669
    frameStart := 90636 },
  { event := event90670
    frameStart := 90636 },
  { event := event90671
    frameStart := 90636 }
]

def eventLeaf5667 : Array AnnotatedEvent := #[
  { event := event90672
    frameStart := 90636 },
  { event := event90673
    frameStart := 90636 },
  { event := event90674
    frameStart := 90636 },
  { event := event90675
    frameStart := 90636 },
  { event := event90676
    frameStart := 90636 },
  { event := event90677
    frameStart := 90636 },
  { event := event90678
    frameStart := 90636 },
  { event := event90679
    frameStart := 90636 },
  { event := event90680
    frameStart := 90636 },
  { event := event90681
    frameStart := 90636 },
  { event := event90682
    frameStart := 90636 },
  { event := event90683
    frameStart := 90636 },
  { event := event90684
    frameStart := 90636 },
  { event := event90685
    frameStart := 90636 },
  { event := event90686
    frameStart := 90636 },
  { event := event90687
    frameStart := 90636 }
]

def eventLeaf5668 : Array AnnotatedEvent := #[
  { event := event90688
    frameStart := 90636 },
  { event := event90689
    frameStart := 90636 },
  { event := event90690
    frameStart := 90690 },
  { event := event90691
    frameStart := 90690 },
  { event := event90692
    frameStart := 90690 },
  { event := event90693
    frameStart := 90690 },
  { event := event90694
    frameStart := 90690 },
  { event := event90695
    frameStart := 90690 },
  { event := event90696
    frameStart := 90690 },
  { event := event90697
    frameStart := 90690 },
  { event := event90698
    frameStart := 90690 },
  { event := event90699
    frameStart := 90690 },
  { event := event90700
    frameStart := 90690 },
  { event := event90701
    frameStart := 90690 },
  { event := event90702
    frameStart := 90690 },
  { event := event90703
    frameStart := 90690 }
]

def eventLeaf5669 : Array AnnotatedEvent := #[
  { event := event90704
    frameStart := 90690 },
  { event := event90705
    frameStart := 90690 },
  { event := event90706
    frameStart := 90690 },
  { event := event90707
    frameStart := 90690 },
  { event := event90708
    frameStart := 90690 },
  { event := event90709
    frameStart := 90690 },
  { event := event90710
    frameStart := 90690 },
  { event := event90711
    frameStart := 90690 },
  { event := event90712
    frameStart := 90690 },
  { event := event90713
    frameStart := 90690 },
  { event := event90714
    frameStart := 90690 },
  { event := event90715
    frameStart := 90690 },
  { event := event90716
    frameStart := 90690 },
  { event := event90717
    frameStart := 90690 },
  { event := event90718
    frameStart := 90690 },
  { event := event90719
    frameStart := 90690 }
]

def eventLeaf5670 : Array AnnotatedEvent := #[
  { event := event90720
    frameStart := 90690 },
  { event := event90721
    frameStart := 90690 },
  { event := event90722
    frameStart := 90690 },
  { event := event90723
    frameStart := 90690 },
  { event := event90724
    frameStart := 90690 },
  { event := event90725
    frameStart := 90690 },
  { event := event90726
    frameStart := 90690 },
  { event := event90727
    frameStart := 90690 },
  { event := event90728
    frameStart := 90690 },
  { event := event90729
    frameStart := 90690 },
  { event := event90730
    frameStart := 90690 },
  { event := event90731
    frameStart := 90690 },
  { event := event90732
    frameStart := 90690 },
  { event := event90733
    frameStart := 90690 },
  { event := event90734
    frameStart := 90690 },
  { event := event90735
    frameStart := 90690 }
]

def eventLeaf5671 : Array AnnotatedEvent := #[
  { event := event90736
    frameStart := 90690 },
  { event := event90737
    frameStart := 90690 },
  { event := event90738
    frameStart := 90690 },
  { event := event90739
    frameStart := 90690 },
  { event := event90740
    frameStart := 90690 },
  { event := event90741
    frameStart := 90690 },
  { event := event90742
    frameStart := 90690 },
  { event := event90743
    frameStart := 90690 },
  { event := event90744
    frameStart := 90690 },
  { event := event90745
    frameStart := 90690 },
  { event := event90746
    frameStart := 90690 },
  { event := event90747
    frameStart := 90690 },
  { event := event90748
    frameStart := 90690 },
  { event := event90749
    frameStart := 90690 },
  { event := event90750
    frameStart := 90690 },
  { event := event90751
    frameStart := 90690 }
]

def eventLeaf5672 : Array AnnotatedEvent := #[
  { event := event90752
    frameStart := 90690 },
  { event := event90753
    frameStart := 90690 },
  { event := event90754
    frameStart := 90690 },
  { event := event90755
    frameStart := 90690 },
  { event := event90756
    frameStart := 90690 },
  { event := event90757
    frameStart := 90690 },
  { event := event90758
    frameStart := 90690 },
  { event := event90759
    frameStart := 90690 },
  { event := event90760
    frameStart := 90690 },
  { event := event90761
    frameStart := 90690 },
  { event := event90762
    frameStart := 90690 },
  { event := event90763
    frameStart := 90690 },
  { event := event90764
    frameStart := 90690 },
  { event := event90765
    frameStart := 90690 },
  { event := event90766
    frameStart := 90690 },
  { event := event90767
    frameStart := 90690 }
]

def eventLeaf5673 : Array AnnotatedEvent := #[
  { event := event90768
    frameStart := 90690 },
  { event := event90769
    frameStart := 90690 },
  { event := event90770
    frameStart := 90690 },
  { event := event90771
    frameStart := 90690 },
  { event := event90772
    frameStart := 90690 },
  { event := event90773
    frameStart := 90690 },
  { event := event90774
    frameStart := 90690 },
  { event := event90775
    frameStart := 90690 },
  { event := event90776
    frameStart := 90690 },
  { event := event90777
    frameStart := 90690 },
  { event := event90778
    frameStart := 90690 },
  { event := event90779
    frameStart := 90690 },
  { event := event90780
    frameStart := 90690 },
  { event := event90781
    frameStart := 90690 },
  { event := event90782
    frameStart := 90690 },
  { event := event90783
    frameStart := 90690 }
]

def eventLeaf5674 : Array AnnotatedEvent := #[
  { event := event90784
    frameStart := 90690 },
  { event := event90785
    frameStart := 90690 },
  { event := event90786
    frameStart := 90690 },
  { event := event90787
    frameStart := 90690 },
  { event := event90788
    frameStart := 90690 },
  { event := event90789
    frameStart := 90690 },
  { event := event90790
    frameStart := 90690 },
  { event := event90791
    frameStart := 90690 },
  { event := event90792
    frameStart := 90690 },
  { event := event90793
    frameStart := 90690 },
  { event := event90794
    frameStart := 0 },
  { event := event90795
    frameStart := 0 },
  { event := event90796
    frameStart := 0 },
  { event := event90797
    frameStart := 0 },
  { event := event90798
    frameStart := 0 },
  { event := event90799
    frameStart := 0 }
]

def eventLeaf5675 : Array AnnotatedEvent := #[
  { event := event90800
    frameStart := 0 },
  { event := event90801
    frameStart := 0 },
  { event := event90802
    frameStart := 0 },
  { event := event90803
    frameStart := 0 },
  { event := event90804
    frameStart := 0 },
  { event := event90805
    frameStart := 0 },
  { event := event90806
    frameStart := 0 },
  { event := event90807
    frameStart := 0 },
  { event := event90808
    frameStart := 0 },
  { event := event90809
    frameStart := 0 },
  { event := event90810
    frameStart := 0 },
  { event := event90811
    frameStart := 0 },
  { event := event90812
    frameStart := 0 },
  { event := event90813
    frameStart := 0 },
  { event := event90814
    frameStart := 0 },
  { event := event90815
    frameStart := 0 }
]

def eventLeaf5676 : Array AnnotatedEvent := #[
  { event := event90816
    frameStart := 0 },
  { event := event90817
    frameStart := 0 },
  { event := event90818
    frameStart := 0 },
  { event := event90819
    frameStart := 0 },
  { event := event90820
    frameStart := 0 },
  { event := event90821
    frameStart := 0 },
  { event := event90822
    frameStart := 0 },
  { event := event90823
    frameStart := 0 },
  { event := event90824
    frameStart := 0 },
  { event := event90825
    frameStart := 0 },
  { event := event90826
    frameStart := 0 },
  { event := event90827
    frameStart := 0 },
  { event := event90828
    frameStart := 0 },
  { event := event90829
    frameStart := 0 },
  { event := event90830
    frameStart := 0 },
  { event := event90831
    frameStart := 0 }
]

def eventLeaf5677 : Array AnnotatedEvent := #[
  { event := event90832
    frameStart := 0 },
  { event := event90833
    frameStart := 0 },
  { event := event90834
    frameStart := 0 },
  { event := event90835
    frameStart := 0 },
  { event := event90836
    frameStart := 0 },
  { event := event90837
    frameStart := 0 },
  { event := event90838
    frameStart := 0 },
  { event := event90839
    frameStart := 0 },
  { event := event90840
    frameStart := 0 },
  { event := event90841
    frameStart := 0 },
  { event := event90842
    frameStart := 0 },
  { event := event90843
    frameStart := 0 },
  { event := event90844
    frameStart := 0 },
  { event := event90845
    frameStart := 0 },
  { event := event90846
    frameStart := 0 },
  { event := event90847
    frameStart := 0 }
]

def eventLeaf5678 : Array AnnotatedEvent := #[
  { event := event90848
    frameStart := 90848 },
  { event := event90849
    frameStart := 90848 },
  { event := event90850
    frameStart := 90848 },
  { event := event90851
    frameStart := 90848 },
  { event := event90852
    frameStart := 90848 },
  { event := event90853
    frameStart := 90848 },
  { event := event90854
    frameStart := 90848 },
  { event := event90855
    frameStart := 90848 },
  { event := event90856
    frameStart := 90848 },
  { event := event90857
    frameStart := 90848 },
  { event := event90858
    frameStart := 90848 },
  { event := event90859
    frameStart := 90848 },
  { event := event90860
    frameStart := 90848 },
  { event := event90861
    frameStart := 90848 },
  { event := event90862
    frameStart := 90848 },
  { event := event90863
    frameStart := 90848 }
]

def eventLeaf5679 : Array AnnotatedEvent := #[
  { event := event90864
    frameStart := 90848 },
  { event := event90865
    frameStart := 90848 },
  { event := event90866
    frameStart := 90848 },
  { event := event90867
    frameStart := 90848 },
  { event := event90868
    frameStart := 90848 },
  { event := event90869
    frameStart := 90848 },
  { event := event90870
    frameStart := 90848 },
  { event := event90871
    frameStart := 90848 },
  { event := event90872
    frameStart := 90848 },
  { event := event90873
    frameStart := 90848 },
  { event := event90874
    frameStart := 90848 },
  { event := event90875
    frameStart := 90848 },
  { event := event90876
    frameStart := 90848 },
  { event := event90877
    frameStart := 90848 },
  { event := event90878
    frameStart := 90848 },
  { event := event90879
    frameStart := 90848 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events354
