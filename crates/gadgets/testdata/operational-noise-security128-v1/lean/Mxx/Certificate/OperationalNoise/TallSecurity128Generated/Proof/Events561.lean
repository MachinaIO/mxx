import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events561

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event143616 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31298⟩⟩) 0 ⟨31297⟩ 143615

def event143617 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31298⟩⟩) (.identity (.predecessor 0 143616 .coefficient))

def event143618 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31298⟩⟩) (.finite 36)

def event143619 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31772⟩⟩) 0 ⟨31298⟩ 143618

def event143620 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31772⟩⟩) (.authority (.programFamilyFact))

def exact143621RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31772⟩⟩], []⟩, (1)⟩]

theorem exact143621RawTermsValid :
    exact143621RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143621 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31772⟩⟩) exact143621RawTerms (.finite 6) 143620 .exactZero (none)

def event143622 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31773⟩⟩) 0 ⟨31772⟩ 143621

def event143623 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31773⟩⟩) (.identity (.predecessor 0 143622 .coefficient))

def event143624 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31773⟩⟩) (.finite 6)

def event143625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31973⟩⟩) 0 ⟨31773⟩ 143624

def event143626 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31973⟩⟩) (.authority (.programFamilyFact))

def exact143627RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31973⟩⟩], []⟩, (1)⟩]

theorem exact143627RawTermsValid :
    exact143627RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143627 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31973⟩⟩) exact143627RawTerms (.finite 55) 143626 .exactZero (none)

def event143628 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21326⟩⟩) 0 ⟨5469⟩ 143267

def event143629 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21326⟩⟩) (.authority (.programFamilyFact))

def exact143630RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21326⟩⟩], []⟩, (1)⟩]

theorem exact143630RawTermsValid :
    exact143630RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143630 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21326⟩⟩) exact143630RawTerms (.finite 4) 143629 .exactZero (none)

def event143631 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20996⟩⟩) 0 ⟨5469⟩ 143267

def event143632 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20996⟩⟩) (.authority (.programFamilyFact))

def exact143633RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨20996⟩⟩], []⟩, (1)⟩]

theorem exact143633RawTermsValid :
    exact143633RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143633 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20996⟩⟩) exact143633RawTerms (.finite 4) 143632 .exactZero (none)

def event143634 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21327⟩⟩) 0 ⟨20996⟩ 143633

def event143635 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21327⟩⟩) 1 ⟨21326⟩ 143630

def event143636 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21327⟩⟩) (.product (.predecessor 0 143634 .coefficient) (.predecessor 1 143635 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event143637 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21327⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨20996⟩⟩, ⟨.program ⟨257⟩, ⟨21326⟩⟩], []⟩) [⟨.result 143633 .coefficient, true, some 1⟩, ⟨.result 143630 .coefficient, true, some 1⟩])

def event143638 : Event := .survivorFold (1) 143637

def exact143639RawTerms : List Term := []

theorem exact143639RawTermsValid :
    exact143639RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143639 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21327⟩⟩) exact143639RawTerms (.finite 16) 143636 (.finite 16) (some (143637))

def event143640 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21328⟩⟩) 0 ⟨21327⟩ 143639

def event143641 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21328⟩⟩) (.identity (.predecessor 0 143640 .coefficient))

def event143642 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21328⟩⟩) (.finite 16)

def event143643 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21752⟩⟩) 0 ⟨21328⟩ 143642

def event143644 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21752⟩⟩) (.authority (.programFamilyFact))

def exact143645RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21752⟩⟩], []⟩, (1)⟩]

theorem exact143645RawTermsValid :
    exact143645RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143645 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21752⟩⟩) exact143645RawTerms (.finite 4) 143644 .exactZero (none)

def event143646 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21753⟩⟩) 0 ⟨21752⟩ 143645

def event143647 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21753⟩⟩) (.identity (.predecessor 0 143646 .coefficient))

def event143648 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨21753⟩⟩) (.finite 4)

def event143649 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21953⟩⟩) 0 ⟨21753⟩ 143648

def event143650 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21953⟩⟩) (.authority (.programFamilyFact))

def exact143651RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨21953⟩⟩], []⟩, (1)⟩]

theorem exact143651RawTermsValid :
    exact143651RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143651 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21953⟩⟩) exact143651RawTerms (.finite 51) 143650 .exactZero (none)

def event143652 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18106⟩⟩) 0 ⟨5469⟩ 143267

def event143653 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18106⟩⟩) (.authority (.programFamilyFact))

def exact143654RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18106⟩⟩], []⟩, (1)⟩]

theorem exact143654RawTermsValid :
    exact143654RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143654 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18106⟩⟩) exact143654RawTerms (.finite 3) 143653 .exactZero (none)

def event143655 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12576⟩⟩) 0 ⟨5469⟩ 143267

def event143656 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12576⟩⟩) (.authority (.programFamilyFact))

def exact143657RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12576⟩⟩], []⟩, (1)⟩]

theorem exact143657RawTermsValid :
    exact143657RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143657 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12576⟩⟩) exact143657RawTerms (.finite 3) 143656 .exactZero (none)

def event143658 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18107⟩⟩) 0 ⟨12576⟩ 143657

def event143659 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18107⟩⟩) 1 ⟨18106⟩ 143654

def event143660 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18107⟩⟩) (.product (.predecessor 0 143658 .coefficient) (.predecessor 1 143659 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event143661 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18107⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12576⟩⟩, ⟨.program ⟨257⟩, ⟨18106⟩⟩], []⟩) [⟨.result 143657 .coefficient, true, some 1⟩, ⟨.result 143654 .coefficient, true, some 1⟩])

def event143662 : Event := .survivorFold (1) 143661

def exact143663RawTerms : List Term := []

theorem exact143663RawTermsValid :
    exact143663RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143663 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18107⟩⟩) exact143663RawTerms (.finite 9) 143660 (.finite 9) (some (143661))

def event143664 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18108⟩⟩) 0 ⟨18107⟩ 143663

def event143665 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18108⟩⟩) (.identity (.predecessor 0 143664 .coefficient))

def event143666 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18108⟩⟩) (.finite 9)

def event143667 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18532⟩⟩) 0 ⟨18108⟩ 143666

def event143668 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18532⟩⟩) (.authority (.programFamilyFact))

def exact143669RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18532⟩⟩], []⟩, (1)⟩]

theorem exact143669RawTermsValid :
    exact143669RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143669 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18532⟩⟩) exact143669RawTerms (.finite 3) 143668 .exactZero (none)

def event143670 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18533⟩⟩) 0 ⟨18532⟩ 143669

def event143671 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18533⟩⟩) (.identity (.predecessor 0 143670 .coefficient))

def event143672 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨18533⟩⟩) (.finite 3)

def event143673 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18733⟩⟩) 0 ⟨18533⟩ 143672

def event143674 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18733⟩⟩) (.authority (.programFamilyFact))

def exact143675RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18733⟩⟩], []⟩, (1)⟩]

theorem exact143675RawTermsValid :
    exact143675RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143675 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18733⟩⟩) exact143675RawTerms (.finite 48) 143674 .exactZero (none)

def event143676 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15306⟩⟩) 0 ⟨5469⟩ 143267

def event143677 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15306⟩⟩) (.authority (.programFamilyFact))

def exact143678RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15306⟩⟩], []⟩, (1)⟩]

theorem exact143678RawTermsValid :
    exact143678RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143678 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15306⟩⟩) exact143678RawTerms (.finite 2) 143677 .exactZero (none)

def event143679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12276⟩⟩) 0 ⟨5469⟩ 143267

def event143680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12276⟩⟩) (.authority (.programFamilyFact))

def exact143681RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩], []⟩, (1)⟩]

theorem exact143681RawTermsValid :
    exact143681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143681 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12276⟩⟩) exact143681RawTerms (.finite 2) 143680 .exactZero (none)

def event143682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15307⟩⟩) 0 ⟨12276⟩ 143681

def event143683 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15307⟩⟩) 1 ⟨15306⟩ 143678

def event143684 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15307⟩⟩) (.product (.predecessor 0 143682 .coefficient) (.predecessor 1 143683 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event143685 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15307⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12276⟩⟩, ⟨.program ⟨257⟩, ⟨15306⟩⟩], []⟩) [⟨.result 143681 .coefficient, true, some 1⟩, ⟨.result 143678 .coefficient, true, some 1⟩])

def event143686 : Event := .survivorFold (1) 143685

def exact143687RawTerms : List Term := []

theorem exact143687RawTermsValid :
    exact143687RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143687 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15307⟩⟩) exact143687RawTerms (.finite 4) 143684 (.finite 4) (some (143685))

def event143688 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15308⟩⟩) 0 ⟨15307⟩ 143687

def event143689 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15308⟩⟩) (.identity (.predecessor 0 143688 .coefficient))

def event143690 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15308⟩⟩) (.finite 4)

def event143691 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15732⟩⟩) 0 ⟨15308⟩ 143690

def event143692 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15732⟩⟩) (.authority (.programFamilyFact))

def exact143693RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15732⟩⟩], []⟩, (1)⟩]

theorem exact143693RawTermsValid :
    exact143693RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143693 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15732⟩⟩) exact143693RawTerms (.finite 2) 143692 .exactZero (none)

def event143694 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15733⟩⟩) 0 ⟨15732⟩ 143693

def event143695 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15733⟩⟩) (.identity (.predecessor 0 143694 .coefficient))

def event143696 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15733⟩⟩) (.finite 2)

def event143697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15923⟩⟩) 0 ⟨15733⟩ 143696

def event143698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15923⟩⟩) (.authority (.programFamilyFact))

def exact143699RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], []⟩, (1)⟩]

theorem exact143699RawTermsValid :
    exact143699RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143699 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15923⟩⟩) exact143699RawTerms (.finite 43) 143698 .exactZero (none)

def event143700 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18734⟩⟩) 0 ⟨15923⟩ 143699

def event143701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18734⟩⟩) 1 ⟨18733⟩ 143675

def event143702 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18734⟩⟩) (.sum [.predecessor 0 143700 .coefficient, .predecessor 1 143701 .coefficient])

def event143703 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18734⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨18733⟩⟩], []⟩) [⟨.result 143675 .coefficient, true, some 1⟩])

def event143704 : Event := .survivorFold (1) 143703

def event143705 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18734⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨15923⟩⟩], []⟩) [⟨.result 143699 .coefficient, true, some 1⟩])

def event143706 : Event := .survivorFold (1) 143705

def event143707 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18734⟩⟩) (.sum [.transfer 143703, .transfer 143705])

def exact143708RawTerms : List Term := []

theorem exact143708RawTermsValid :
    exact143708RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143708 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18734⟩⟩) exact143708RawTerms (.finite 91) 143702 (.finite 91) (some (143707))

def event143709 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21954⟩⟩) 0 ⟨18734⟩ 143708

def event143710 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21954⟩⟩) 1 ⟨21953⟩ 143651

def event143711 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21954⟩⟩) (.sum [.predecessor 0 143709 .coefficient, .predecessor 1 143710 .coefficient])

def event143712 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21954⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨21953⟩⟩], []⟩) [⟨.result 143651 .coefficient, true, some 1⟩])

def event143713 : Event := .survivorFold (1) 143712

def event143714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21954⟩⟩) (.sum [.result 143708 .summary, .transfer 143712])

def exact143715RawTerms : List Term := []

theorem exact143715RawTermsValid :
    exact143715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143715 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21954⟩⟩) exact143715RawTerms (.finite 142) 143711 (.finite 142) (some (143714))

def event143716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31974⟩⟩) 0 ⟨21954⟩ 143715

def event143717 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31974⟩⟩) 1 ⟨31973⟩ 143627

def event143718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31974⟩⟩) (.sum [.predecessor 0 143716 .coefficient, .predecessor 1 143717 .coefficient])

def event143719 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31974⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨31973⟩⟩], []⟩) [⟨.result 143627 .coefficient, true, some 1⟩])

def event143720 : Event := .survivorFold (1) 143719

def event143721 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31974⟩⟩) (.sum [.result 143715 .summary, .transfer 143719])

def exact143722RawTerms : List Term := []

theorem exact143722RawTermsValid :
    exact143722RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143722 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31974⟩⟩) exact143722RawTerms (.finite 197) 143718 (.finite 197) (some (143721))

def event143723 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51029⟩⟩) 0 ⟨31974⟩ 143722

def event143724 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51029⟩⟩) 1 ⟨51028⟩ 143603

def event143725 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51029⟩⟩) (.sum [.predecessor 0 143723 .coefficient, .predecessor 1 143724 .coefficient])

def event143726 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51029⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨51028⟩⟩], []⟩) [⟨.result 143603 .coefficient, true, some 1⟩])

def event143727 : Event := .survivorFold (1) 143726

def event143728 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51029⟩⟩) (.sum [.result 143722 .summary, .transfer 143726])

def exact143729RawTerms : List Term := []

theorem exact143729RawTermsValid :
    exact143729RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143729 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51029⟩⟩) exact143729RawTerms (.finite 255) 143725 (.finite 255) (some (143728))

def event143730 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54009⟩⟩) 0 ⟨51029⟩ 143729

def event143731 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54009⟩⟩) 1 ⟨54008⟩ 143579

def event143732 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54009⟩⟩) (.sum [.predecessor 0 143730 .coefficient, .predecessor 1 143731 .coefficient])

def event143733 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54009⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨54008⟩⟩], []⟩) [⟨.result 143579 .coefficient, true, some 1⟩])

def event143734 : Event := .survivorFold (1) 143733

def event143735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54009⟩⟩) (.sum [.result 143729 .summary, .transfer 143733])

def exact143736RawTerms : List Term := []

theorem exact143736RawTermsValid :
    exact143736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143736 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54009⟩⟩) exact143736RawTerms (.finite 314) 143732 (.finite 314) (some (143735))

def event143737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56989⟩⟩) 0 ⟨54009⟩ 143736

def event143738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56989⟩⟩) 1 ⟨56988⟩ 143555

def event143739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56989⟩⟩) (.sum [.predecessor 0 143737 .coefficient, .predecessor 1 143738 .coefficient])

def event143740 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56989⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨56988⟩⟩], []⟩) [⟨.result 143555 .coefficient, true, some 1⟩])

def event143741 : Event := .survivorFold (1) 143740

def event143742 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56989⟩⟩) (.sum [.result 143736 .summary, .transfer 143740])

def exact143743RawTerms : List Term := []

theorem exact143743RawTermsValid :
    exact143743RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143743 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56989⟩⟩) exact143743RawTerms (.finite 374) 143739 (.finite 374) (some (143742))

def event143744 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59969⟩⟩) 0 ⟨56989⟩ 143743

def event143745 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59969⟩⟩) 1 ⟨59968⟩ 143531

def event143746 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59969⟩⟩) (.sum [.predecessor 0 143744 .coefficient, .predecessor 1 143745 .coefficient])

def event143747 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59969⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨59968⟩⟩], []⟩) [⟨.result 143531 .coefficient, true, some 1⟩])

def event143748 : Event := .survivorFold (1) 143747

def event143749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59969⟩⟩) (.sum [.result 143743 .summary, .transfer 143747])

def exact143750RawTerms : List Term := []

theorem exact143750RawTermsValid :
    exact143750RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143750 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59969⟩⟩) exact143750RawTerms (.finite 435) 143746 (.finite 435) (some (143749))

def event143751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62949⟩⟩) 0 ⟨59969⟩ 143750

def event143752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62949⟩⟩) 1 ⟨62948⟩ 143507

def event143753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62949⟩⟩) (.sum [.predecessor 0 143751 .coefficient, .predecessor 1 143752 .coefficient])

def event143754 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62949⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨62948⟩⟩], []⟩) [⟨.result 143507 .coefficient, true, some 1⟩])

def event143755 : Event := .survivorFold (1) 143754

def event143756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62949⟩⟩) (.sum [.result 143750 .summary, .transfer 143754])

def exact143757RawTerms : List Term := []

theorem exact143757RawTermsValid :
    exact143757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143757 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62949⟩⟩) exact143757RawTerms (.finite 496) 143753 (.finite 496) (some (143756))

def event143758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66112⟩⟩) 0 ⟨62949⟩ 143757

def event143759 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66112⟩⟩) 1 ⟨66111⟩ 143483

def event143760 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66112⟩⟩) (.sum [.predecessor 0 143758 .coefficient, .predecessor 1 143759 .coefficient])

def event143761 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66112⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨66111⟩⟩], []⟩) [⟨.result 143483 .coefficient, true, some 1⟩])

def event143762 : Event := .survivorFold (1) 143761

def event143763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66112⟩⟩) (.sum [.result 143757 .summary, .transfer 143761])

def exact143764RawTerms : List Term := []

theorem exact143764RawTermsValid :
    exact143764RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143764 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66112⟩⟩) exact143764RawTerms (.finite 558) 143760 (.finite 558) (some (143763))

def event143765 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66113⟩⟩) 0 ⟨66112⟩ 143764

def event143766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66113⟩⟩) 1 ⟨26528⟩ 143459

def event143767 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66113⟩⟩) (.sum [.predecessor 0 143765 .coefficient, .predecessor 1 143766 .coefficient])

def event143768 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66113⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨26528⟩⟩], []⟩) [⟨.result 143459 .coefficient, true, some 1⟩])

def event143769 : Event := .survivorFold (1) 143768

def event143770 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66113⟩⟩) (.sum [.result 143764 .summary, .transfer 143768])

def exact143771RawTerms : List Term := []

theorem exact143771RawTermsValid :
    exact143771RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143771 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66113⟩⟩) exact143771RawTerms (.finite 620) 143767 (.finite 620) (some (143770))

def event143772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66114⟩⟩) 0 ⟨66113⟩ 143771

def event143773 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66114⟩⟩) 1 ⟨29208⟩ 143435

def event143774 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66114⟩⟩) (.sum [.predecessor 0 143772 .coefficient, .predecessor 1 143773 .coefficient])

def event143775 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66114⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨29208⟩⟩], []⟩) [⟨.result 143435 .coefficient, true, some 1⟩])

def event143776 : Event := .survivorFold (1) 143775

def event143777 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66114⟩⟩) (.sum [.result 143771 .summary, .transfer 143775])

def exact143778RawTerms : List Term := []

theorem exact143778RawTermsValid :
    exact143778RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143778 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66114⟩⟩) exact143778RawTerms (.finite 682) 143774 (.finite 682) (some (143777))

def event143779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66115⟩⟩) 0 ⟨66114⟩ 143778

def event143780 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66115⟩⟩) 1 ⟨34872⟩ 143411

def event143781 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66115⟩⟩) (.sum [.predecessor 0 143779 .coefficient, .predecessor 1 143780 .coefficient])

def event143782 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66115⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨34872⟩⟩], []⟩) [⟨.result 143411 .coefficient, true, some 1⟩])

def event143783 : Event := .survivorFold (1) 143782

def event143784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66115⟩⟩) (.sum [.result 143778 .summary, .transfer 143782])

def exact143785RawTerms : List Term := []

theorem exact143785RawTermsValid :
    exact143785RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143785 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66115⟩⟩) exact143785RawTerms (.finite 744) 143781 (.finite 744) (some (143784))

def event143786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66116⟩⟩) 0 ⟨66115⟩ 143785

def event143787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66116⟩⟩) 1 ⟨37552⟩ 143387

def event143788 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66116⟩⟩) (.sum [.predecessor 0 143786 .coefficient, .predecessor 1 143787 .coefficient])

def event143789 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66116⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨37552⟩⟩], []⟩) [⟨.result 143387 .coefficient, true, some 1⟩])

def event143790 : Event := .survivorFold (1) 143789

def event143791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66116⟩⟩) (.sum [.result 143785 .summary, .transfer 143789])

def exact143792RawTerms : List Term := []

theorem exact143792RawTermsValid :
    exact143792RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143792 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66116⟩⟩) exact143792RawTerms (.finite 807) 143788 (.finite 807) (some (143791))

def event143793 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66117⟩⟩) 0 ⟨66116⟩ 143792

def event143794 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66117⟩⟩) 1 ⟨40228⟩ 143363

def event143795 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66117⟩⟩) (.sum [.predecessor 0 143793 .coefficient, .predecessor 1 143794 .coefficient])

def event143796 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66117⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨40228⟩⟩], []⟩) [⟨.result 143363 .coefficient, true, some 1⟩])

def event143797 : Event := .survivorFold (1) 143796

def event143798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66117⟩⟩) (.sum [.result 143792 .summary, .transfer 143796])

def exact143799RawTerms : List Term := []

theorem exact143799RawTermsValid :
    exact143799RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143799 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66117⟩⟩) exact143799RawTerms (.finite 870) 143795 (.finite 870) (some (143798))

def event143800 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66118⟩⟩) 0 ⟨66117⟩ 143799

def event143801 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66118⟩⟩) 1 ⟨42908⟩ 143339

def event143802 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66118⟩⟩) (.sum [.predecessor 0 143800 .coefficient, .predecessor 1 143801 .coefficient])

def event143803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66118⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨42908⟩⟩], []⟩) [⟨.result 143339 .coefficient, true, some 1⟩])

def event143804 : Event := .survivorFold (1) 143803

def event143805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66118⟩⟩) (.sum [.result 143799 .summary, .transfer 143803])

def exact143806RawTerms : List Term := []

theorem exact143806RawTermsValid :
    exact143806RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143806 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66118⟩⟩) exact143806RawTerms (.finite 933) 143802 (.finite 933) (some (143805))

def event143807 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66119⟩⟩) 0 ⟨66118⟩ 143806

def event143808 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66119⟩⟩) 1 ⟨45592⟩ 143315

def event143809 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66119⟩⟩) (.sum [.predecessor 0 143807 .coefficient, .predecessor 1 143808 .coefficient])

def event143810 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66119⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨45592⟩⟩], []⟩) [⟨.result 143315 .coefficient, true, some 1⟩])

def event143811 : Event := .survivorFold (1) 143810

def event143812 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66119⟩⟩) (.sum [.result 143806 .summary, .transfer 143810])

def exact143813RawTerms : List Term := []

theorem exact143813RawTermsValid :
    exact143813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143813 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66119⟩⟩) exact143813RawTerms (.finite 996) 143809 (.finite 996) (some (143812))

def event143814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66120⟩⟩) 0 ⟨66119⟩ 143813

def event143815 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66120⟩⟩) 1 ⟨48272⟩ 143291

def event143816 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66120⟩⟩) (.sum [.predecessor 0 143814 .coefficient, .predecessor 1 143815 .coefficient])

def event143817 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66120⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], []⟩) [⟨.result 143291 .coefficient, true, some 1⟩])

def event143818 : Event := .survivorFold (1) 143817

def event143819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66120⟩⟩) (.sum [.result 143813 .summary, .transfer 143817])

def exact143820RawTerms : List Term := []

theorem exact143820RawTermsValid :
    exact143820RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143820 : Event := .resultExact (⟨.program ⟨257⟩, ⟨66120⟩⟩) exact143820RawTerms (.finite 1059) 143816 (.finite 1059) (some (143819))

def event143821 : Event := .predecessor (⟨.program ⟨257⟩, ⟨66121⟩⟩) 0 ⟨66120⟩ 143820

def event143822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨66121⟩⟩) (.identity (.predecessor 0 143821 .coefficient))

def event143823 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨66121⟩⟩) (.finite 1059)

def event143824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68300⟩⟩) 0 ⟨66121⟩ 143823

def event143825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68300⟩⟩) (.authority (.relationPreimageSource ⟨95⟩))

def exact143826RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68300⟩⟩]⟩, (1)⟩]

theorem exact143826RawTermsValid :
    exact143826RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143826 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68300⟩⟩) exact143826RawTerms (.finite 5647228698) 143825 .exactZero (none)

def event143827 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact143828RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact143828RawTermsValid :
    exact143828RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143828 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact143828RawTerms .large 143827 .exactZero (none)

def event143829 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68301⟩⟩) 0 ⟨35⟩ 143828

def event143830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68301⟩⟩) 1 ⟨68300⟩ 143826

def event143831 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68301⟩⟩) (.product (.predecessor 0 143829 .coefficient) (.predecessor 1 143830 .coefficient) (⟨false, false, none, none, none⟩))

def event143832 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68301⟩⟩, .operator (⟨143828, 0⟩, ⟨143826, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68300⟩⟩]⟩, (1)⟩)

def exact143833RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68300⟩⟩]⟩, (1)⟩]

theorem exact143833RawTermsValid :
    exact143833RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143833 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68301⟩⟩) exact143833RawTerms .large 143831 .exactZero (none)

def event143834 : Event := .preFoldPolynomial 143833 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68300⟩⟩]⟩, (1)⟩] .exactZero none

def exact143835RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68300⟩⟩]⟩, (1)⟩]

def event143835 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨68301⟩⟩) 143834 exact143835RawTerms .large 143831 .exactZero (none)

def event143836 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨71022⟩⟩)

def event143837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event143838 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event143839 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event143840 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event143841 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event143842 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event143843 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event143844 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event143845 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 143844

def event143846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 143842

def event143847 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 143845 .coefficient) (.value (.predecessor 1 143846 .coefficient)))

def event143848 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event143849 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 143848

def event143850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 143840

def event143851 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 143849 .coefficient, .predecessor 1 143850 .coefficient])

def event143852 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event143853 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 143852

def event143854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 143838

def event143855 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 143854 .coefficient))

def event143856 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event143857 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47666⟩⟩) 0 ⟨5469⟩ 143856

def event143858 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47666⟩⟩) (.authority (.programFamilyFact))

def exact143859RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47666⟩⟩], []⟩, (1)⟩]

theorem exact143859RawTermsValid :
    exact143859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143859 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47666⟩⟩) exact143859RawTerms (.finite 60) 143858 .exactZero (none)

def event143860 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14976⟩⟩) 0 ⟨5469⟩ 143856

def event143861 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14976⟩⟩) (.authority (.programFamilyFact))

def exact143862RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩], []⟩, (1)⟩]

theorem exact143862RawTermsValid :
    exact143862RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143862 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14976⟩⟩) exact143862RawTerms (.finite 60) 143861 .exactZero (none)

def event143863 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47667⟩⟩) 0 ⟨14976⟩ 143862

def event143864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47667⟩⟩) 1 ⟨47666⟩ 143859

def event143865 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47667⟩⟩) (.product (.predecessor 0 143863 .coefficient) (.predecessor 1 143864 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event143866 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47667⟩⟩, .operator (⟨143862, 0⟩, ⟨143859, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], []⟩, (1)⟩)

def exact143867RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], []⟩, (1)⟩]

theorem exact143867RawTermsValid :
    exact143867RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event143867 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47667⟩⟩) exact143867RawTerms (.finite 3600) 143865 .exactZero (none)

def event143868 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47668⟩⟩) 0 ⟨47667⟩ 143867

def event143869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47668⟩⟩) (.identity (.predecessor 0 143868 .coefficient))

def event143870 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47668⟩⟩) (.finite 3600)

def event143871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48092⟩⟩) 0 ⟨47668⟩ 143870

def eventLeaf8976 : Array AnnotatedEvent := #[
  { event := event143616
    frameStart := 143247 },
  { event := event143617
    frameStart := 143247 },
  { event := event143618
    frameStart := 143247 },
  { event := event143619
    frameStart := 143247 },
  { event := event143620
    frameStart := 143247 },
  { event := event143621
    frameStart := 143247 },
  { event := event143622
    frameStart := 143247 },
  { event := event143623
    frameStart := 143247 },
  { event := event143624
    frameStart := 143247 },
  { event := event143625
    frameStart := 143247 },
  { event := event143626
    frameStart := 143247 },
  { event := event143627
    frameStart := 143247 },
  { event := event143628
    frameStart := 143247 },
  { event := event143629
    frameStart := 143247 },
  { event := event143630
    frameStart := 143247 },
  { event := event143631
    frameStart := 143247 }
]

def eventLeaf8977 : Array AnnotatedEvent := #[
  { event := event143632
    frameStart := 143247 },
  { event := event143633
    frameStart := 143247 },
  { event := event143634
    frameStart := 143247 },
  { event := event143635
    frameStart := 143247 },
  { event := event143636
    frameStart := 143247 },
  { event := event143637
    frameStart := 143247 },
  { event := event143638
    frameStart := 143247 },
  { event := event143639
    frameStart := 143247 },
  { event := event143640
    frameStart := 143247 },
  { event := event143641
    frameStart := 143247 },
  { event := event143642
    frameStart := 143247 },
  { event := event143643
    frameStart := 143247 },
  { event := event143644
    frameStart := 143247 },
  { event := event143645
    frameStart := 143247 },
  { event := event143646
    frameStart := 143247 },
  { event := event143647
    frameStart := 143247 }
]

def eventLeaf8978 : Array AnnotatedEvent := #[
  { event := event143648
    frameStart := 143247 },
  { event := event143649
    frameStart := 143247 },
  { event := event143650
    frameStart := 143247 },
  { event := event143651
    frameStart := 143247 },
  { event := event143652
    frameStart := 143247 },
  { event := event143653
    frameStart := 143247 },
  { event := event143654
    frameStart := 143247 },
  { event := event143655
    frameStart := 143247 },
  { event := event143656
    frameStart := 143247 },
  { event := event143657
    frameStart := 143247 },
  { event := event143658
    frameStart := 143247 },
  { event := event143659
    frameStart := 143247 },
  { event := event143660
    frameStart := 143247 },
  { event := event143661
    frameStart := 143247 },
  { event := event143662
    frameStart := 143247 },
  { event := event143663
    frameStart := 143247 }
]

def eventLeaf8979 : Array AnnotatedEvent := #[
  { event := event143664
    frameStart := 143247 },
  { event := event143665
    frameStart := 143247 },
  { event := event143666
    frameStart := 143247 },
  { event := event143667
    frameStart := 143247 },
  { event := event143668
    frameStart := 143247 },
  { event := event143669
    frameStart := 143247 },
  { event := event143670
    frameStart := 143247 },
  { event := event143671
    frameStart := 143247 },
  { event := event143672
    frameStart := 143247 },
  { event := event143673
    frameStart := 143247 },
  { event := event143674
    frameStart := 143247 },
  { event := event143675
    frameStart := 143247 },
  { event := event143676
    frameStart := 143247 },
  { event := event143677
    frameStart := 143247 },
  { event := event143678
    frameStart := 143247 },
  { event := event143679
    frameStart := 143247 }
]

def eventLeaf8980 : Array AnnotatedEvent := #[
  { event := event143680
    frameStart := 143247 },
  { event := event143681
    frameStart := 143247 },
  { event := event143682
    frameStart := 143247 },
  { event := event143683
    frameStart := 143247 },
  { event := event143684
    frameStart := 143247 },
  { event := event143685
    frameStart := 143247 },
  { event := event143686
    frameStart := 143247 },
  { event := event143687
    frameStart := 143247 },
  { event := event143688
    frameStart := 143247 },
  { event := event143689
    frameStart := 143247 },
  { event := event143690
    frameStart := 143247 },
  { event := event143691
    frameStart := 143247 },
  { event := event143692
    frameStart := 143247 },
  { event := event143693
    frameStart := 143247 },
  { event := event143694
    frameStart := 143247 },
  { event := event143695
    frameStart := 143247 }
]

def eventLeaf8981 : Array AnnotatedEvent := #[
  { event := event143696
    frameStart := 143247 },
  { event := event143697
    frameStart := 143247 },
  { event := event143698
    frameStart := 143247 },
  { event := event143699
    frameStart := 143247 },
  { event := event143700
    frameStart := 143247 },
  { event := event143701
    frameStart := 143247 },
  { event := event143702
    frameStart := 143247 },
  { event := event143703
    frameStart := 143247 },
  { event := event143704
    frameStart := 143247 },
  { event := event143705
    frameStart := 143247 },
  { event := event143706
    frameStart := 143247 },
  { event := event143707
    frameStart := 143247 },
  { event := event143708
    frameStart := 143247 },
  { event := event143709
    frameStart := 143247 },
  { event := event143710
    frameStart := 143247 },
  { event := event143711
    frameStart := 143247 }
]

def eventLeaf8982 : Array AnnotatedEvent := #[
  { event := event143712
    frameStart := 143247 },
  { event := event143713
    frameStart := 143247 },
  { event := event143714
    frameStart := 143247 },
  { event := event143715
    frameStart := 143247 },
  { event := event143716
    frameStart := 143247 },
  { event := event143717
    frameStart := 143247 },
  { event := event143718
    frameStart := 143247 },
  { event := event143719
    frameStart := 143247 },
  { event := event143720
    frameStart := 143247 },
  { event := event143721
    frameStart := 143247 },
  { event := event143722
    frameStart := 143247 },
  { event := event143723
    frameStart := 143247 },
  { event := event143724
    frameStart := 143247 },
  { event := event143725
    frameStart := 143247 },
  { event := event143726
    frameStart := 143247 },
  { event := event143727
    frameStart := 143247 }
]

def eventLeaf8983 : Array AnnotatedEvent := #[
  { event := event143728
    frameStart := 143247 },
  { event := event143729
    frameStart := 143247 },
  { event := event143730
    frameStart := 143247 },
  { event := event143731
    frameStart := 143247 },
  { event := event143732
    frameStart := 143247 },
  { event := event143733
    frameStart := 143247 },
  { event := event143734
    frameStart := 143247 },
  { event := event143735
    frameStart := 143247 },
  { event := event143736
    frameStart := 143247 },
  { event := event143737
    frameStart := 143247 },
  { event := event143738
    frameStart := 143247 },
  { event := event143739
    frameStart := 143247 },
  { event := event143740
    frameStart := 143247 },
  { event := event143741
    frameStart := 143247 },
  { event := event143742
    frameStart := 143247 },
  { event := event143743
    frameStart := 143247 }
]

def eventLeaf8984 : Array AnnotatedEvent := #[
  { event := event143744
    frameStart := 143247 },
  { event := event143745
    frameStart := 143247 },
  { event := event143746
    frameStart := 143247 },
  { event := event143747
    frameStart := 143247 },
  { event := event143748
    frameStart := 143247 },
  { event := event143749
    frameStart := 143247 },
  { event := event143750
    frameStart := 143247 },
  { event := event143751
    frameStart := 143247 },
  { event := event143752
    frameStart := 143247 },
  { event := event143753
    frameStart := 143247 },
  { event := event143754
    frameStart := 143247 },
  { event := event143755
    frameStart := 143247 },
  { event := event143756
    frameStart := 143247 },
  { event := event143757
    frameStart := 143247 },
  { event := event143758
    frameStart := 143247 },
  { event := event143759
    frameStart := 143247 }
]

def eventLeaf8985 : Array AnnotatedEvent := #[
  { event := event143760
    frameStart := 143247 },
  { event := event143761
    frameStart := 143247 },
  { event := event143762
    frameStart := 143247 },
  { event := event143763
    frameStart := 143247 },
  { event := event143764
    frameStart := 143247 },
  { event := event143765
    frameStart := 143247 },
  { event := event143766
    frameStart := 143247 },
  { event := event143767
    frameStart := 143247 },
  { event := event143768
    frameStart := 143247 },
  { event := event143769
    frameStart := 143247 },
  { event := event143770
    frameStart := 143247 },
  { event := event143771
    frameStart := 143247 },
  { event := event143772
    frameStart := 143247 },
  { event := event143773
    frameStart := 143247 },
  { event := event143774
    frameStart := 143247 },
  { event := event143775
    frameStart := 143247 }
]

def eventLeaf8986 : Array AnnotatedEvent := #[
  { event := event143776
    frameStart := 143247 },
  { event := event143777
    frameStart := 143247 },
  { event := event143778
    frameStart := 143247 },
  { event := event143779
    frameStart := 143247 },
  { event := event143780
    frameStart := 143247 },
  { event := event143781
    frameStart := 143247 },
  { event := event143782
    frameStart := 143247 },
  { event := event143783
    frameStart := 143247 },
  { event := event143784
    frameStart := 143247 },
  { event := event143785
    frameStart := 143247 },
  { event := event143786
    frameStart := 143247 },
  { event := event143787
    frameStart := 143247 },
  { event := event143788
    frameStart := 143247 },
  { event := event143789
    frameStart := 143247 },
  { event := event143790
    frameStart := 143247 },
  { event := event143791
    frameStart := 143247 }
]

def eventLeaf8987 : Array AnnotatedEvent := #[
  { event := event143792
    frameStart := 143247 },
  { event := event143793
    frameStart := 143247 },
  { event := event143794
    frameStart := 143247 },
  { event := event143795
    frameStart := 143247 },
  { event := event143796
    frameStart := 143247 },
  { event := event143797
    frameStart := 143247 },
  { event := event143798
    frameStart := 143247 },
  { event := event143799
    frameStart := 143247 },
  { event := event143800
    frameStart := 143247 },
  { event := event143801
    frameStart := 143247 },
  { event := event143802
    frameStart := 143247 },
  { event := event143803
    frameStart := 143247 },
  { event := event143804
    frameStart := 143247 },
  { event := event143805
    frameStart := 143247 },
  { event := event143806
    frameStart := 143247 },
  { event := event143807
    frameStart := 143247 }
]

def eventLeaf8988 : Array AnnotatedEvent := #[
  { event := event143808
    frameStart := 143247 },
  { event := event143809
    frameStart := 143247 },
  { event := event143810
    frameStart := 143247 },
  { event := event143811
    frameStart := 143247 },
  { event := event143812
    frameStart := 143247 },
  { event := event143813
    frameStart := 143247 },
  { event := event143814
    frameStart := 143247 },
  { event := event143815
    frameStart := 143247 },
  { event := event143816
    frameStart := 143247 },
  { event := event143817
    frameStart := 143247 },
  { event := event143818
    frameStart := 143247 },
  { event := event143819
    frameStart := 143247 },
  { event := event143820
    frameStart := 143247 },
  { event := event143821
    frameStart := 143247 },
  { event := event143822
    frameStart := 143247 },
  { event := event143823
    frameStart := 143247 }
]

def eventLeaf8989 : Array AnnotatedEvent := #[
  { event := event143824
    frameStart := 143247 },
  { event := event143825
    frameStart := 143247 },
  { event := event143826
    frameStart := 143247 },
  { event := event143827
    frameStart := 143247 },
  { event := event143828
    frameStart := 143247 },
  { event := event143829
    frameStart := 143247 },
  { event := event143830
    frameStart := 143247 },
  { event := event143831
    frameStart := 143247 },
  { event := event143832
    frameStart := 143247 },
  { event := event143833
    frameStart := 143247 },
  { event := event143834
    frameStart := 143247 },
  { event := event143835
    frameStart := 143247 },
  { event := event143836
    frameStart := 143836 },
  { event := event143837
    frameStart := 143836 },
  { event := event143838
    frameStart := 143836 },
  { event := event143839
    frameStart := 143836 }
]

def eventLeaf8990 : Array AnnotatedEvent := #[
  { event := event143840
    frameStart := 143836 },
  { event := event143841
    frameStart := 143836 },
  { event := event143842
    frameStart := 143836 },
  { event := event143843
    frameStart := 143836 },
  { event := event143844
    frameStart := 143836 },
  { event := event143845
    frameStart := 143836 },
  { event := event143846
    frameStart := 143836 },
  { event := event143847
    frameStart := 143836 },
  { event := event143848
    frameStart := 143836 },
  { event := event143849
    frameStart := 143836 },
  { event := event143850
    frameStart := 143836 },
  { event := event143851
    frameStart := 143836 },
  { event := event143852
    frameStart := 143836 },
  { event := event143853
    frameStart := 143836 },
  { event := event143854
    frameStart := 143836 },
  { event := event143855
    frameStart := 143836 }
]

def eventLeaf8991 : Array AnnotatedEvent := #[
  { event := event143856
    frameStart := 143836 },
  { event := event143857
    frameStart := 143836 },
  { event := event143858
    frameStart := 143836 },
  { event := event143859
    frameStart := 143836 },
  { event := event143860
    frameStart := 143836 },
  { event := event143861
    frameStart := 143836 },
  { event := event143862
    frameStart := 143836 },
  { event := event143863
    frameStart := 143836 },
  { event := event143864
    frameStart := 143836 },
  { event := event143865
    frameStart := 143836 },
  { event := event143866
    frameStart := 143836 },
  { event := event143867
    frameStart := 143836 },
  { event := event143868
    frameStart := 143836 },
  { event := event143869
    frameStart := 143836 },
  { event := event143870
    frameStart := 143836 },
  { event := event143871
    frameStart := 143836 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events561
