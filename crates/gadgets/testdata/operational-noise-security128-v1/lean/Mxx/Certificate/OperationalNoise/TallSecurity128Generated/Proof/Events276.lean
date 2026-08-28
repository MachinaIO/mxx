import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events276

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event70656 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67095⟩⟩) (.sum [.predecessor 0 70654 .coefficient, .predecessor 1 70655 .coefficient])

def event70657 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67095⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨35054⟩⟩], []⟩) [⟨.result 70286 .coefficient, true, some 1⟩])

def event70658 : Event := .survivorFold (1) 70657

def event70659 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67095⟩⟩) (.sum [.result 70653 .summary, .transfer 70657])

def exact70660RawTerms : List Term := []

theorem exact70660RawTermsValid :
    exact70660RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70660 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67095⟩⟩) exact70660RawTerms (.finite 744) 70656 (.finite 744) (some (70659))

def event70661 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67096⟩⟩) 0 ⟨67095⟩ 70660

def event70662 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67096⟩⟩) 1 ⟨37734⟩ 70262

def event70663 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67096⟩⟩) (.sum [.predecessor 0 70661 .coefficient, .predecessor 1 70662 .coefficient])

def event70664 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67096⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨37734⟩⟩], []⟩) [⟨.result 70262 .coefficient, true, some 1⟩])

def event70665 : Event := .survivorFold (1) 70664

def event70666 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67096⟩⟩) (.sum [.result 70660 .summary, .transfer 70664])

def exact70667RawTerms : List Term := []

theorem exact70667RawTermsValid :
    exact70667RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70667 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67096⟩⟩) exact70667RawTerms (.finite 807) 70663 (.finite 807) (some (70666))

def event70668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67097⟩⟩) 0 ⟨67096⟩ 70667

def event70669 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67097⟩⟩) 1 ⟨40410⟩ 70238

def event70670 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67097⟩⟩) (.sum [.predecessor 0 70668 .coefficient, .predecessor 1 70669 .coefficient])

def event70671 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67097⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨40410⟩⟩], []⟩) [⟨.result 70238 .coefficient, true, some 1⟩])

def event70672 : Event := .survivorFold (1) 70671

def event70673 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67097⟩⟩) (.sum [.result 70667 .summary, .transfer 70671])

def exact70674RawTerms : List Term := []

theorem exact70674RawTermsValid :
    exact70674RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70674 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67097⟩⟩) exact70674RawTerms (.finite 870) 70670 (.finite 870) (some (70673))

def event70675 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67098⟩⟩) 0 ⟨67097⟩ 70674

def event70676 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67098⟩⟩) 1 ⟨43090⟩ 70214

def event70677 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67098⟩⟩) (.sum [.predecessor 0 70675 .coefficient, .predecessor 1 70676 .coefficient])

def event70678 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67098⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨43090⟩⟩], []⟩) [⟨.result 70214 .coefficient, true, some 1⟩])

def event70679 : Event := .survivorFold (1) 70678

def event70680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67098⟩⟩) (.sum [.result 70674 .summary, .transfer 70678])

def exact70681RawTerms : List Term := []

theorem exact70681RawTermsValid :
    exact70681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70681 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67098⟩⟩) exact70681RawTerms (.finite 933) 70677 (.finite 933) (some (70680))

def event70682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67099⟩⟩) 0 ⟨67098⟩ 70681

def event70683 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67099⟩⟩) 1 ⟨45774⟩ 70190

def event70684 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67099⟩⟩) (.sum [.predecessor 0 70682 .coefficient, .predecessor 1 70683 .coefficient])

def event70685 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67099⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨45774⟩⟩], []⟩) [⟨.result 70190 .coefficient, true, some 1⟩])

def event70686 : Event := .survivorFold (1) 70685

def event70687 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67099⟩⟩) (.sum [.result 70681 .summary, .transfer 70685])

def exact70688RawTerms : List Term := []

theorem exact70688RawTermsValid :
    exact70688RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70688 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67099⟩⟩) exact70688RawTerms (.finite 996) 70684 (.finite 996) (some (70687))

def event70689 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67100⟩⟩) 0 ⟨67099⟩ 70688

def event70690 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67100⟩⟩) 1 ⟨48454⟩ 70166

def event70691 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67100⟩⟩) (.sum [.predecessor 0 70689 .coefficient, .predecessor 1 70690 .coefficient])

def event70692 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67100⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨48454⟩⟩], []⟩) [⟨.result 70166 .coefficient, true, some 1⟩])

def event70693 : Event := .survivorFold (1) 70692

def event70694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67100⟩⟩) (.sum [.result 70688 .summary, .transfer 70692])

def exact70695RawTerms : List Term := []

theorem exact70695RawTermsValid :
    exact70695RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70695 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67100⟩⟩) exact70695RawTerms (.finite 1059) 70691 (.finite 1059) (some (70694))

def event70696 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67101⟩⟩) 0 ⟨67100⟩ 70695

def event70697 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67101⟩⟩) (.identity (.predecessor 0 70696 .coefficient))

def event70698 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨67101⟩⟩) (.finite 1059)

def event70699 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68440⟩⟩) 0 ⟨67101⟩ 70698

def event70700 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68440⟩⟩) (.authority (.relationPreimageSource ⟨95⟩))

def exact70701RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68440⟩⟩]⟩, (1)⟩]

theorem exact70701RawTermsValid :
    exact70701RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70701 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68440⟩⟩) exact70701RawTerms (.finite 5647228698) 70700 .exactZero (none)

def event70702 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact70703RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact70703RawTermsValid :
    exact70703RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70703 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact70703RawTerms .large 70702 .exactZero (none)

def event70704 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68441⟩⟩) 0 ⟨35⟩ 70703

def event70705 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68441⟩⟩) 1 ⟨68440⟩ 70701

def event70706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68441⟩⟩) (.product (.predecessor 0 70704 .coefficient) (.predecessor 1 70705 .coefficient) (⟨false, false, none, none, none⟩))

def event70707 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68441⟩⟩, .operator (⟨70703, 0⟩, ⟨70701, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68440⟩⟩]⟩, (1)⟩)

def exact70708RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68440⟩⟩]⟩, (1)⟩]

theorem exact70708RawTermsValid :
    exact70708RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70708 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68441⟩⟩) exact70708RawTerms .large 70706 .exactZero (none)

def event70709 : Event := .preFoldPolynomial 70708 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68440⟩⟩]⟩, (1)⟩] .exactZero none

def exact70710RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68440⟩⟩]⟩, (1)⟩]

def event70710 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨68441⟩⟩) 70709 exact70710RawTerms .large 70706 .exactZero (none)

def event70711 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨71474⟩⟩)

def event70712 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event70713 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event70714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.authority (.operator))

def event70715 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.finite 16)

def event70716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event70717 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event70718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event70719 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event70720 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 70719

def event70721 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 70717

def event70722 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 70720 .coefficient) (.value (.predecessor 1 70721 .coefficient)))

def event70723 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event70724 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 0 ⟨392⟩ 70723

def event70725 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 1 ⟨10691⟩ 70715

def event70726 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.sum [.predecessor 0 70724 .coefficient, .predecessor 1 70725 .coefficient])

def event70727 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.finite 655356)

def event70728 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 0 ⟨10693⟩ 70727

def event70729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 1 ⟨5426⟩ 70713

def event70730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.identity (.predecessor 1 70729 .coefficient))

def event70731 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.finite 655360)

def event70732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48002⟩⟩) 0 ⟨10749⟩ 70731

def event70733 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48002⟩⟩) (.authority (.programFamilyFact))

def exact70734RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48002⟩⟩], []⟩, (1)⟩]

theorem exact70734RawTermsValid :
    exact70734RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70734 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48002⟩⟩) exact70734RawTerms (.finite 60) 70733 .exactZero (none)

def event70735 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15186⟩⟩) 0 ⟨10749⟩ 70731

def event70736 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15186⟩⟩) (.authority (.programFamilyFact))

def exact70737RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15186⟩⟩], []⟩, (1)⟩]

theorem exact70737RawTermsValid :
    exact70737RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70737 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15186⟩⟩) exact70737RawTerms (.finite 60) 70736 .exactZero (none)

def event70738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48003⟩⟩) 0 ⟨15186⟩ 70737

def event70739 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48003⟩⟩) 1 ⟨48002⟩ 70734

def event70740 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48003⟩⟩) (.product (.predecessor 0 70738 .coefficient) (.predecessor 1 70739 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70741 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48003⟩⟩, .operator (⟨70737, 0⟩, ⟨70734, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], []⟩, (1)⟩)

def exact70742RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], []⟩, (1)⟩]

theorem exact70742RawTermsValid :
    exact70742RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70742 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48003⟩⟩) exact70742RawTerms (.finite 3600) 70740 .exactZero (none)

def event70743 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48004⟩⟩) 0 ⟨48003⟩ 70742

def event70744 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48004⟩⟩) (.identity (.predecessor 0 70743 .coefficient))

def event70745 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48004⟩⟩) (.finite 3600)

def event70746 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48204⟩⟩) 0 ⟨48004⟩ 70745

def event70747 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48204⟩⟩) (.authority (.programFamilyFact))

def exact70748RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], []⟩, (1)⟩]

theorem exact70748RawTermsValid :
    exact70748RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70748 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48204⟩⟩) exact70748RawTerms (.finite 60) 70747 .exactZero (none)

def event70749 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48205⟩⟩) 0 ⟨48204⟩ 70748

def event70750 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48205⟩⟩) (.identity (.predecessor 0 70749 .coefficient))

def event70751 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48205⟩⟩) (.finite 60)

def event70752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48454⟩⟩) 0 ⟨48205⟩ 70751

def event70753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48454⟩⟩) (.authority (.programFamilyFact))

def exact70754RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48454⟩⟩], []⟩, (1)⟩]

theorem exact70754RawTermsValid :
    exact70754RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70754 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48454⟩⟩) exact70754RawTerms (.finite 63) 70753 .exactZero (none)

def event70755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45322⟩⟩) 0 ⟨10749⟩ 70731

def event70756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45322⟩⟩) (.authority (.programFamilyFact))

def exact70757RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45322⟩⟩], []⟩, (1)⟩]

theorem exact70757RawTermsValid :
    exact70757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70757 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45322⟩⟩) exact70757RawTerms (.finite 58) 70756 .exactZero (none)

def event70758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14886⟩⟩) 0 ⟨10749⟩ 70731

def event70759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14886⟩⟩) (.authority (.programFamilyFact))

def exact70760RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14886⟩⟩], []⟩, (1)⟩]

theorem exact70760RawTermsValid :
    exact70760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70760 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14886⟩⟩) exact70760RawTerms (.finite 58) 70759 .exactZero (none)

def event70761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45323⟩⟩) 0 ⟨14886⟩ 70760

def event70762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45323⟩⟩) 1 ⟨45322⟩ 70757

def event70763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45323⟩⟩) (.product (.predecessor 0 70761 .coefficient) (.predecessor 1 70762 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70764 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45323⟩⟩, .operator (⟨70760, 0⟩, ⟨70757, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], []⟩, (1)⟩)

def exact70765RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], []⟩, (1)⟩]

theorem exact70765RawTermsValid :
    exact70765RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70765 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45323⟩⟩) exact70765RawTerms (.finite 3364) 70763 .exactZero (none)

def event70766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45324⟩⟩) 0 ⟨45323⟩ 70765

def event70767 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45324⟩⟩) (.identity (.predecessor 0 70766 .coefficient))

def event70768 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45324⟩⟩) (.finite 3364)

def event70769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45524⟩⟩) 0 ⟨45324⟩ 70768

def event70770 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45524⟩⟩) (.authority (.programFamilyFact))

def exact70771RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45524⟩⟩], []⟩, (1)⟩]

theorem exact70771RawTermsValid :
    exact70771RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70771 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45524⟩⟩) exact70771RawTerms (.finite 58) 70770 .exactZero (none)

def event70772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45525⟩⟩) 0 ⟨45524⟩ 70771

def event70773 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45525⟩⟩) (.identity (.predecessor 0 70772 .coefficient))

def event70774 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨45525⟩⟩) (.finite 58)

def event70775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45774⟩⟩) 0 ⟨45525⟩ 70774

def event70776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45774⟩⟩) (.authority (.programFamilyFact))

def exact70777RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45774⟩⟩], []⟩, (1)⟩]

theorem exact70777RawTermsValid :
    exact70777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70777 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45774⟩⟩) exact70777RawTerms (.finite 63) 70776 .exactZero (none)

def event70778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42642⟩⟩) 0 ⟨10749⟩ 70731

def event70779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42642⟩⟩) (.authority (.programFamilyFact))

def exact70780RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42642⟩⟩], []⟩, (1)⟩]

theorem exact70780RawTermsValid :
    exact70780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70780 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42642⟩⟩) exact70780RawTerms (.finite 52) 70779 .exactZero (none)

def event70781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14586⟩⟩) 0 ⟨10749⟩ 70731

def event70782 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14586⟩⟩) (.authority (.programFamilyFact))

def exact70783RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14586⟩⟩], []⟩, (1)⟩]

theorem exact70783RawTermsValid :
    exact70783RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70783 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14586⟩⟩) exact70783RawTerms (.finite 52) 70782 .exactZero (none)

def event70784 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42643⟩⟩) 0 ⟨14586⟩ 70783

def event70785 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42643⟩⟩) 1 ⟨42642⟩ 70780

def event70786 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42643⟩⟩) (.product (.predecessor 0 70784 .coefficient) (.predecessor 1 70785 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70787 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42643⟩⟩, .operator (⟨70783, 0⟩, ⟨70780, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], []⟩, (1)⟩)

def exact70788RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], []⟩, (1)⟩]

theorem exact70788RawTermsValid :
    exact70788RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70788 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42643⟩⟩) exact70788RawTerms (.finite 2704) 70786 .exactZero (none)

def event70789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42644⟩⟩) 0 ⟨42643⟩ 70788

def event70790 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42644⟩⟩) (.identity (.predecessor 0 70789 .coefficient))

def event70791 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42644⟩⟩) (.finite 2704)

def event70792 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42844⟩⟩) 0 ⟨42644⟩ 70791

def event70793 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42844⟩⟩) (.authority (.programFamilyFact))

def exact70794RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42844⟩⟩], []⟩, (1)⟩]

theorem exact70794RawTermsValid :
    exact70794RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70794 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42844⟩⟩) exact70794RawTerms (.finite 52) 70793 .exactZero (none)

def event70795 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42845⟩⟩) 0 ⟨42844⟩ 70794

def event70796 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42845⟩⟩) (.identity (.predecessor 0 70795 .coefficient))

def event70797 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42845⟩⟩) (.finite 52)

def event70798 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43090⟩⟩) 0 ⟨42845⟩ 70797

def event70799 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43090⟩⟩) (.authority (.programFamilyFact))

def exact70800RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨43090⟩⟩], []⟩, (1)⟩]

theorem exact70800RawTermsValid :
    exact70800RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70800 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43090⟩⟩) exact70800RawTerms (.finite 63) 70799 .exactZero (none)

def event70801 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39962⟩⟩) 0 ⟨10749⟩ 70731

def event70802 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39962⟩⟩) (.authority (.programFamilyFact))

def exact70803RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39962⟩⟩], []⟩, (1)⟩]

theorem exact70803RawTermsValid :
    exact70803RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70803 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39962⟩⟩) exact70803RawTerms (.finite 46) 70802 .exactZero (none)

def event70804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14286⟩⟩) 0 ⟨10749⟩ 70731

def event70805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14286⟩⟩) (.authority (.programFamilyFact))

def exact70806RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14286⟩⟩], []⟩, (1)⟩]

theorem exact70806RawTermsValid :
    exact70806RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70806 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14286⟩⟩) exact70806RawTerms (.finite 46) 70805 .exactZero (none)

def event70807 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39963⟩⟩) 0 ⟨14286⟩ 70806

def event70808 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39963⟩⟩) 1 ⟨39962⟩ 70803

def event70809 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39963⟩⟩) (.product (.predecessor 0 70807 .coefficient) (.predecessor 1 70808 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70810 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39963⟩⟩, .operator (⟨70806, 0⟩, ⟨70803, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], []⟩, (1)⟩)

def exact70811RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14286⟩⟩, ⟨.program ⟨257⟩, ⟨39962⟩⟩], []⟩, (1)⟩]

theorem exact70811RawTermsValid :
    exact70811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70811 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39963⟩⟩) exact70811RawTerms (.finite 2116) 70809 .exactZero (none)

def event70812 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39964⟩⟩) 0 ⟨39963⟩ 70811

def event70813 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39964⟩⟩) (.identity (.predecessor 0 70812 .coefficient))

def event70814 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39964⟩⟩) (.finite 2116)

def event70815 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40164⟩⟩) 0 ⟨39964⟩ 70814

def event70816 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40164⟩⟩) (.authority (.programFamilyFact))

def exact70817RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40164⟩⟩], []⟩, (1)⟩]

theorem exact70817RawTermsValid :
    exact70817RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70817 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40164⟩⟩) exact70817RawTerms (.finite 46) 70816 .exactZero (none)

def event70818 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40165⟩⟩) 0 ⟨40164⟩ 70817

def event70819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40165⟩⟩) (.identity (.predecessor 0 70818 .coefficient))

def event70820 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40165⟩⟩) (.finite 46)

def event70821 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40410⟩⟩) 0 ⟨40165⟩ 70820

def event70822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40410⟩⟩) (.authority (.programFamilyFact))

def exact70823RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40410⟩⟩], []⟩, (1)⟩]

theorem exact70823RawTermsValid :
    exact70823RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70823 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40410⟩⟩) exact70823RawTerms (.finite 63) 70822 .exactZero (none)

def event70824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37282⟩⟩) 0 ⟨10749⟩ 70731

def event70825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37282⟩⟩) (.authority (.programFamilyFact))

def exact70826RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37282⟩⟩], []⟩, (1)⟩]

theorem exact70826RawTermsValid :
    exact70826RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70826 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37282⟩⟩) exact70826RawTerms (.finite 42) 70825 .exactZero (none)

def event70827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13986⟩⟩) 0 ⟨10749⟩ 70731

def event70828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13986⟩⟩) (.authority (.programFamilyFact))

def exact70829RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13986⟩⟩], []⟩, (1)⟩]

theorem exact70829RawTermsValid :
    exact70829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70829 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13986⟩⟩) exact70829RawTerms (.finite 42) 70828 .exactZero (none)

def event70830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37283⟩⟩) 0 ⟨13986⟩ 70829

def event70831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37283⟩⟩) 1 ⟨37282⟩ 70826

def event70832 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37283⟩⟩) (.product (.predecessor 0 70830 .coefficient) (.predecessor 1 70831 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70833 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37283⟩⟩, .operator (⟨70829, 0⟩, ⟨70826, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13986⟩⟩, ⟨.program ⟨257⟩, ⟨37282⟩⟩], []⟩, (1)⟩)

def exact70834RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13986⟩⟩, ⟨.program ⟨257⟩, ⟨37282⟩⟩], []⟩, (1)⟩]

theorem exact70834RawTermsValid :
    exact70834RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70834 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37283⟩⟩) exact70834RawTerms (.finite 1764) 70832 .exactZero (none)

def event70835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37284⟩⟩) 0 ⟨37283⟩ 70834

def event70836 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37284⟩⟩) (.identity (.predecessor 0 70835 .coefficient))

def event70837 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37284⟩⟩) (.finite 1764)

def event70838 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37484⟩⟩) 0 ⟨37284⟩ 70837

def event70839 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37484⟩⟩) (.authority (.programFamilyFact))

def exact70840RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37484⟩⟩], []⟩, (1)⟩]

theorem exact70840RawTermsValid :
    exact70840RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70840 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37484⟩⟩) exact70840RawTerms (.finite 42) 70839 .exactZero (none)

def event70841 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37485⟩⟩) 0 ⟨37484⟩ 70840

def event70842 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37485⟩⟩) (.identity (.predecessor 0 70841 .coefficient))

def event70843 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37485⟩⟩) (.finite 42)

def event70844 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37734⟩⟩) 0 ⟨37485⟩ 70843

def event70845 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37734⟩⟩) (.authority (.programFamilyFact))

def exact70846RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37734⟩⟩], []⟩, (1)⟩]

theorem exact70846RawTermsValid :
    exact70846RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70846 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37734⟩⟩) exact70846RawTerms (.finite 63) 70845 .exactZero (none)

def event70847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34602⟩⟩) 0 ⟨10749⟩ 70731

def event70848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34602⟩⟩) (.authority (.programFamilyFact))

def exact70849RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34602⟩⟩], []⟩, (1)⟩]

theorem exact70849RawTermsValid :
    exact70849RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70849 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34602⟩⟩) exact70849RawTerms (.finite 40) 70848 .exactZero (none)

def event70850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13686⟩⟩) 0 ⟨10749⟩ 70731

def event70851 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13686⟩⟩) (.authority (.programFamilyFact))

def exact70852RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13686⟩⟩], []⟩, (1)⟩]

theorem exact70852RawTermsValid :
    exact70852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70852 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13686⟩⟩) exact70852RawTerms (.finite 40) 70851 .exactZero (none)

def event70853 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34603⟩⟩) 0 ⟨13686⟩ 70852

def event70854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34603⟩⟩) 1 ⟨34602⟩ 70849

def event70855 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34603⟩⟩) (.product (.predecessor 0 70853 .coefficient) (.predecessor 1 70854 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70856 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34603⟩⟩, .operator (⟨70852, 0⟩, ⟨70849, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], []⟩, (1)⟩)

def exact70857RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], []⟩, (1)⟩]

theorem exact70857RawTermsValid :
    exact70857RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70857 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34603⟩⟩) exact70857RawTerms (.finite 1600) 70855 .exactZero (none)

def event70858 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34604⟩⟩) 0 ⟨34603⟩ 70857

def event70859 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34604⟩⟩) (.identity (.predecessor 0 70858 .coefficient))

def event70860 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34604⟩⟩) (.finite 1600)

def event70861 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34804⟩⟩) 0 ⟨34604⟩ 70860

def event70862 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34804⟩⟩) (.authority (.programFamilyFact))

def exact70863RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34804⟩⟩], []⟩, (1)⟩]

theorem exact70863RawTermsValid :
    exact70863RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70863 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34804⟩⟩) exact70863RawTerms (.finite 40) 70862 .exactZero (none)

def event70864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34805⟩⟩) 0 ⟨34804⟩ 70863

def event70865 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34805⟩⟩) (.identity (.predecessor 0 70864 .coefficient))

def event70866 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34805⟩⟩) (.finite 40)

def event70867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35054⟩⟩) 0 ⟨34805⟩ 70866

def event70868 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35054⟩⟩) (.authority (.programFamilyFact))

def exact70869RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨35054⟩⟩], []⟩, (1)⟩]

theorem exact70869RawTermsValid :
    exact70869RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70869 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35054⟩⟩) exact70869RawTerms (.finite 62) 70868 .exactZero (none)

def event70870 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28942⟩⟩) 0 ⟨10749⟩ 70731

def event70871 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28942⟩⟩) (.authority (.programFamilyFact))

def exact70872RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨28942⟩⟩], []⟩, (1)⟩]

theorem exact70872RawTermsValid :
    exact70872RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70872 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28942⟩⟩) exact70872RawTerms (.finite 36) 70871 .exactZero (none)

def event70873 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13386⟩⟩) 0 ⟨10749⟩ 70731

def event70874 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13386⟩⟩) (.authority (.programFamilyFact))

def exact70875RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13386⟩⟩], []⟩, (1)⟩]

theorem exact70875RawTermsValid :
    exact70875RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70875 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13386⟩⟩) exact70875RawTerms (.finite 36) 70874 .exactZero (none)

def event70876 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28943⟩⟩) 0 ⟨13386⟩ 70875

def event70877 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28943⟩⟩) 1 ⟨28942⟩ 70872

def event70878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28943⟩⟩) (.product (.predecessor 0 70876 .coefficient) (.predecessor 1 70877 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70879 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨28943⟩⟩, .operator (⟨70875, 0⟩, ⟨70872, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13386⟩⟩, ⟨.program ⟨257⟩, ⟨28942⟩⟩], []⟩, (1)⟩)

def exact70880RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13386⟩⟩, ⟨.program ⟨257⟩, ⟨28942⟩⟩], []⟩, (1)⟩]

theorem exact70880RawTermsValid :
    exact70880RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70880 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28943⟩⟩) exact70880RawTerms (.finite 1296) 70878 .exactZero (none)

def event70881 : Event := .predecessor (⟨.program ⟨257⟩, ⟨28944⟩⟩) 0 ⟨28943⟩ 70880

def event70882 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28944⟩⟩) (.identity (.predecessor 0 70881 .coefficient))

def event70883 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨28944⟩⟩) (.finite 1296)

def event70884 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29144⟩⟩) 0 ⟨28944⟩ 70883

def event70885 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29144⟩⟩) (.authority (.programFamilyFact))

def exact70886RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29144⟩⟩], []⟩, (1)⟩]

theorem exact70886RawTermsValid :
    exact70886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70886 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29144⟩⟩) exact70886RawTerms (.finite 36) 70885 .exactZero (none)

def event70887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29145⟩⟩) 0 ⟨29144⟩ 70886

def event70888 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29145⟩⟩) (.identity (.predecessor 0 70887 .coefficient))

def event70889 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨29145⟩⟩) (.finite 36)

def event70890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29390⟩⟩) 0 ⟨29145⟩ 70889

def event70891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29390⟩⟩) (.authority (.programFamilyFact))

def exact70892RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29390⟩⟩], []⟩, (1)⟩]

theorem exact70892RawTermsValid :
    exact70892RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70892 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29390⟩⟩) exact70892RawTerms (.finite 62) 70891 .exactZero (none)

def event70893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26262⟩⟩) 0 ⟨10749⟩ 70731

def event70894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26262⟩⟩) (.authority (.programFamilyFact))

def exact70895RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26262⟩⟩], []⟩, (1)⟩]

theorem exact70895RawTermsValid :
    exact70895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26262⟩⟩) exact70895RawTerms (.finite 30) 70894 .exactZero (none)

def event70896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13086⟩⟩) 0 ⟨10749⟩ 70731

def event70897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13086⟩⟩) (.authority (.programFamilyFact))

def exact70898RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13086⟩⟩], []⟩, (1)⟩]

theorem exact70898RawTermsValid :
    exact70898RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70898 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13086⟩⟩) exact70898RawTerms (.finite 30) 70897 .exactZero (none)

def event70899 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26263⟩⟩) 0 ⟨13086⟩ 70898

def event70900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26263⟩⟩) 1 ⟨26262⟩ 70895

def event70901 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26263⟩⟩) (.product (.predecessor 0 70899 .coefficient) (.predecessor 1 70900 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70902 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26263⟩⟩, .operator (⟨70898, 0⟩, ⟨70895, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13086⟩⟩, ⟨.program ⟨257⟩, ⟨26262⟩⟩], []⟩, (1)⟩)

def exact70903RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13086⟩⟩, ⟨.program ⟨257⟩, ⟨26262⟩⟩], []⟩, (1)⟩]

theorem exact70903RawTermsValid :
    exact70903RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70903 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26263⟩⟩) exact70903RawTerms (.finite 900) 70901 .exactZero (none)

def event70904 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26264⟩⟩) 0 ⟨26263⟩ 70903

def event70905 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26264⟩⟩) (.identity (.predecessor 0 70904 .coefficient))

def event70906 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨26264⟩⟩) (.finite 900)

def event70907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26464⟩⟩) 0 ⟨26264⟩ 70906

def event70908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26464⟩⟩) (.authority (.programFamilyFact))

def exact70909RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26464⟩⟩], []⟩, (1)⟩]

theorem exact70909RawTermsValid :
    exact70909RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70909 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26464⟩⟩) exact70909RawTerms (.finite 30) 70908 .exactZero (none)

def event70910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26465⟩⟩) 0 ⟨26464⟩ 70909

def event70911 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26465⟩⟩) (.identity (.predecessor 0 70910 .coefficient))

def eventLeaf4416 : Array AnnotatedEvent := #[
  { event := event70656
    frameStart := 70122 },
  { event := event70657
    frameStart := 70122 },
  { event := event70658
    frameStart := 70122 },
  { event := event70659
    frameStart := 70122 },
  { event := event70660
    frameStart := 70122 },
  { event := event70661
    frameStart := 70122 },
  { event := event70662
    frameStart := 70122 },
  { event := event70663
    frameStart := 70122 },
  { event := event70664
    frameStart := 70122 },
  { event := event70665
    frameStart := 70122 },
  { event := event70666
    frameStart := 70122 },
  { event := event70667
    frameStart := 70122 },
  { event := event70668
    frameStart := 70122 },
  { event := event70669
    frameStart := 70122 },
  { event := event70670
    frameStart := 70122 },
  { event := event70671
    frameStart := 70122 }
]

def eventLeaf4417 : Array AnnotatedEvent := #[
  { event := event70672
    frameStart := 70122 },
  { event := event70673
    frameStart := 70122 },
  { event := event70674
    frameStart := 70122 },
  { event := event70675
    frameStart := 70122 },
  { event := event70676
    frameStart := 70122 },
  { event := event70677
    frameStart := 70122 },
  { event := event70678
    frameStart := 70122 },
  { event := event70679
    frameStart := 70122 },
  { event := event70680
    frameStart := 70122 },
  { event := event70681
    frameStart := 70122 },
  { event := event70682
    frameStart := 70122 },
  { event := event70683
    frameStart := 70122 },
  { event := event70684
    frameStart := 70122 },
  { event := event70685
    frameStart := 70122 },
  { event := event70686
    frameStart := 70122 },
  { event := event70687
    frameStart := 70122 }
]

def eventLeaf4418 : Array AnnotatedEvent := #[
  { event := event70688
    frameStart := 70122 },
  { event := event70689
    frameStart := 70122 },
  { event := event70690
    frameStart := 70122 },
  { event := event70691
    frameStart := 70122 },
  { event := event70692
    frameStart := 70122 },
  { event := event70693
    frameStart := 70122 },
  { event := event70694
    frameStart := 70122 },
  { event := event70695
    frameStart := 70122 },
  { event := event70696
    frameStart := 70122 },
  { event := event70697
    frameStart := 70122 },
  { event := event70698
    frameStart := 70122 },
  { event := event70699
    frameStart := 70122 },
  { event := event70700
    frameStart := 70122 },
  { event := event70701
    frameStart := 70122 },
  { event := event70702
    frameStart := 70122 },
  { event := event70703
    frameStart := 70122 }
]

def eventLeaf4419 : Array AnnotatedEvent := #[
  { event := event70704
    frameStart := 70122 },
  { event := event70705
    frameStart := 70122 },
  { event := event70706
    frameStart := 70122 },
  { event := event70707
    frameStart := 70122 },
  { event := event70708
    frameStart := 70122 },
  { event := event70709
    frameStart := 70122 },
  { event := event70710
    frameStart := 70122 },
  { event := event70711
    frameStart := 70711 },
  { event := event70712
    frameStart := 70711 },
  { event := event70713
    frameStart := 70711 },
  { event := event70714
    frameStart := 70711 },
  { event := event70715
    frameStart := 70711 },
  { event := event70716
    frameStart := 70711 },
  { event := event70717
    frameStart := 70711 },
  { event := event70718
    frameStart := 70711 },
  { event := event70719
    frameStart := 70711 }
]

def eventLeaf4420 : Array AnnotatedEvent := #[
  { event := event70720
    frameStart := 70711 },
  { event := event70721
    frameStart := 70711 },
  { event := event70722
    frameStart := 70711 },
  { event := event70723
    frameStart := 70711 },
  { event := event70724
    frameStart := 70711 },
  { event := event70725
    frameStart := 70711 },
  { event := event70726
    frameStart := 70711 },
  { event := event70727
    frameStart := 70711 },
  { event := event70728
    frameStart := 70711 },
  { event := event70729
    frameStart := 70711 },
  { event := event70730
    frameStart := 70711 },
  { event := event70731
    frameStart := 70711 },
  { event := event70732
    frameStart := 70711 },
  { event := event70733
    frameStart := 70711 },
  { event := event70734
    frameStart := 70711 },
  { event := event70735
    frameStart := 70711 }
]

def eventLeaf4421 : Array AnnotatedEvent := #[
  { event := event70736
    frameStart := 70711 },
  { event := event70737
    frameStart := 70711 },
  { event := event70738
    frameStart := 70711 },
  { event := event70739
    frameStart := 70711 },
  { event := event70740
    frameStart := 70711 },
  { event := event70741
    frameStart := 70711 },
  { event := event70742
    frameStart := 70711 },
  { event := event70743
    frameStart := 70711 },
  { event := event70744
    frameStart := 70711 },
  { event := event70745
    frameStart := 70711 },
  { event := event70746
    frameStart := 70711 },
  { event := event70747
    frameStart := 70711 },
  { event := event70748
    frameStart := 70711 },
  { event := event70749
    frameStart := 70711 },
  { event := event70750
    frameStart := 70711 },
  { event := event70751
    frameStart := 70711 }
]

def eventLeaf4422 : Array AnnotatedEvent := #[
  { event := event70752
    frameStart := 70711 },
  { event := event70753
    frameStart := 70711 },
  { event := event70754
    frameStart := 70711 },
  { event := event70755
    frameStart := 70711 },
  { event := event70756
    frameStart := 70711 },
  { event := event70757
    frameStart := 70711 },
  { event := event70758
    frameStart := 70711 },
  { event := event70759
    frameStart := 70711 },
  { event := event70760
    frameStart := 70711 },
  { event := event70761
    frameStart := 70711 },
  { event := event70762
    frameStart := 70711 },
  { event := event70763
    frameStart := 70711 },
  { event := event70764
    frameStart := 70711 },
  { event := event70765
    frameStart := 70711 },
  { event := event70766
    frameStart := 70711 },
  { event := event70767
    frameStart := 70711 }
]

def eventLeaf4423 : Array AnnotatedEvent := #[
  { event := event70768
    frameStart := 70711 },
  { event := event70769
    frameStart := 70711 },
  { event := event70770
    frameStart := 70711 },
  { event := event70771
    frameStart := 70711 },
  { event := event70772
    frameStart := 70711 },
  { event := event70773
    frameStart := 70711 },
  { event := event70774
    frameStart := 70711 },
  { event := event70775
    frameStart := 70711 },
  { event := event70776
    frameStart := 70711 },
  { event := event70777
    frameStart := 70711 },
  { event := event70778
    frameStart := 70711 },
  { event := event70779
    frameStart := 70711 },
  { event := event70780
    frameStart := 70711 },
  { event := event70781
    frameStart := 70711 },
  { event := event70782
    frameStart := 70711 },
  { event := event70783
    frameStart := 70711 }
]

def eventLeaf4424 : Array AnnotatedEvent := #[
  { event := event70784
    frameStart := 70711 },
  { event := event70785
    frameStart := 70711 },
  { event := event70786
    frameStart := 70711 },
  { event := event70787
    frameStart := 70711 },
  { event := event70788
    frameStart := 70711 },
  { event := event70789
    frameStart := 70711 },
  { event := event70790
    frameStart := 70711 },
  { event := event70791
    frameStart := 70711 },
  { event := event70792
    frameStart := 70711 },
  { event := event70793
    frameStart := 70711 },
  { event := event70794
    frameStart := 70711 },
  { event := event70795
    frameStart := 70711 },
  { event := event70796
    frameStart := 70711 },
  { event := event70797
    frameStart := 70711 },
  { event := event70798
    frameStart := 70711 },
  { event := event70799
    frameStart := 70711 }
]

def eventLeaf4425 : Array AnnotatedEvent := #[
  { event := event70800
    frameStart := 70711 },
  { event := event70801
    frameStart := 70711 },
  { event := event70802
    frameStart := 70711 },
  { event := event70803
    frameStart := 70711 },
  { event := event70804
    frameStart := 70711 },
  { event := event70805
    frameStart := 70711 },
  { event := event70806
    frameStart := 70711 },
  { event := event70807
    frameStart := 70711 },
  { event := event70808
    frameStart := 70711 },
  { event := event70809
    frameStart := 70711 },
  { event := event70810
    frameStart := 70711 },
  { event := event70811
    frameStart := 70711 },
  { event := event70812
    frameStart := 70711 },
  { event := event70813
    frameStart := 70711 },
  { event := event70814
    frameStart := 70711 },
  { event := event70815
    frameStart := 70711 }
]

def eventLeaf4426 : Array AnnotatedEvent := #[
  { event := event70816
    frameStart := 70711 },
  { event := event70817
    frameStart := 70711 },
  { event := event70818
    frameStart := 70711 },
  { event := event70819
    frameStart := 70711 },
  { event := event70820
    frameStart := 70711 },
  { event := event70821
    frameStart := 70711 },
  { event := event70822
    frameStart := 70711 },
  { event := event70823
    frameStart := 70711 },
  { event := event70824
    frameStart := 70711 },
  { event := event70825
    frameStart := 70711 },
  { event := event70826
    frameStart := 70711 },
  { event := event70827
    frameStart := 70711 },
  { event := event70828
    frameStart := 70711 },
  { event := event70829
    frameStart := 70711 },
  { event := event70830
    frameStart := 70711 },
  { event := event70831
    frameStart := 70711 }
]

def eventLeaf4427 : Array AnnotatedEvent := #[
  { event := event70832
    frameStart := 70711 },
  { event := event70833
    frameStart := 70711 },
  { event := event70834
    frameStart := 70711 },
  { event := event70835
    frameStart := 70711 },
  { event := event70836
    frameStart := 70711 },
  { event := event70837
    frameStart := 70711 },
  { event := event70838
    frameStart := 70711 },
  { event := event70839
    frameStart := 70711 },
  { event := event70840
    frameStart := 70711 },
  { event := event70841
    frameStart := 70711 },
  { event := event70842
    frameStart := 70711 },
  { event := event70843
    frameStart := 70711 },
  { event := event70844
    frameStart := 70711 },
  { event := event70845
    frameStart := 70711 },
  { event := event70846
    frameStart := 70711 },
  { event := event70847
    frameStart := 70711 }
]

def eventLeaf4428 : Array AnnotatedEvent := #[
  { event := event70848
    frameStart := 70711 },
  { event := event70849
    frameStart := 70711 },
  { event := event70850
    frameStart := 70711 },
  { event := event70851
    frameStart := 70711 },
  { event := event70852
    frameStart := 70711 },
  { event := event70853
    frameStart := 70711 },
  { event := event70854
    frameStart := 70711 },
  { event := event70855
    frameStart := 70711 },
  { event := event70856
    frameStart := 70711 },
  { event := event70857
    frameStart := 70711 },
  { event := event70858
    frameStart := 70711 },
  { event := event70859
    frameStart := 70711 },
  { event := event70860
    frameStart := 70711 },
  { event := event70861
    frameStart := 70711 },
  { event := event70862
    frameStart := 70711 },
  { event := event70863
    frameStart := 70711 }
]

def eventLeaf4429 : Array AnnotatedEvent := #[
  { event := event70864
    frameStart := 70711 },
  { event := event70865
    frameStart := 70711 },
  { event := event70866
    frameStart := 70711 },
  { event := event70867
    frameStart := 70711 },
  { event := event70868
    frameStart := 70711 },
  { event := event70869
    frameStart := 70711 },
  { event := event70870
    frameStart := 70711 },
  { event := event70871
    frameStart := 70711 },
  { event := event70872
    frameStart := 70711 },
  { event := event70873
    frameStart := 70711 },
  { event := event70874
    frameStart := 70711 },
  { event := event70875
    frameStart := 70711 },
  { event := event70876
    frameStart := 70711 },
  { event := event70877
    frameStart := 70711 },
  { event := event70878
    frameStart := 70711 },
  { event := event70879
    frameStart := 70711 }
]

def eventLeaf4430 : Array AnnotatedEvent := #[
  { event := event70880
    frameStart := 70711 },
  { event := event70881
    frameStart := 70711 },
  { event := event70882
    frameStart := 70711 },
  { event := event70883
    frameStart := 70711 },
  { event := event70884
    frameStart := 70711 },
  { event := event70885
    frameStart := 70711 },
  { event := event70886
    frameStart := 70711 },
  { event := event70887
    frameStart := 70711 },
  { event := event70888
    frameStart := 70711 },
  { event := event70889
    frameStart := 70711 },
  { event := event70890
    frameStart := 70711 },
  { event := event70891
    frameStart := 70711 },
  { event := event70892
    frameStart := 70711 },
  { event := event70893
    frameStart := 70711 },
  { event := event70894
    frameStart := 70711 },
  { event := event70895
    frameStart := 70711 }
]

def eventLeaf4431 : Array AnnotatedEvent := #[
  { event := event70896
    frameStart := 70711 },
  { event := event70897
    frameStart := 70711 },
  { event := event70898
    frameStart := 70711 },
  { event := event70899
    frameStart := 70711 },
  { event := event70900
    frameStart := 70711 },
  { event := event70901
    frameStart := 70711 },
  { event := event70902
    frameStart := 70711 },
  { event := event70903
    frameStart := 70711 },
  { event := event70904
    frameStart := 70711 },
  { event := event70905
    frameStart := 70711 },
  { event := event70906
    frameStart := 70711 },
  { event := event70907
    frameStart := 70711 },
  { event := event70908
    frameStart := 70711 },
  { event := event70909
    frameStart := 70711 },
  { event := event70910
    frameStart := 70711 },
  { event := event70911
    frameStart := 70711 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events276
