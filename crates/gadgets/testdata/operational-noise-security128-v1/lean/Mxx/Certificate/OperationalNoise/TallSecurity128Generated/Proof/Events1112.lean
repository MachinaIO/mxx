import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1112

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event284672 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65285⟩⟩) 0 ⟨65284⟩ 284671

def event284673 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65285⟩⟩) (.identity (.predecessor 0 284672 .coefficient))

def event284674 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65285⟩⟩) (.finite 784)

def event284675 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68493⟩⟩) 0 ⟨65285⟩ 284674

def event284676 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68493⟩⟩) (.authority (.programFamilyFact))

def event284677 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68493⟩⟩) (.finite 3720)

def event284678 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event284679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68494⟩⟩) 0 ⟨7177⟩ 284678

def event284680 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68494⟩⟩) 1 ⟨68493⟩ 284677

def event284681 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68494⟩⟩) (.authority (.operator))

def exact284682RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68494⟩⟩]⟩, (1)⟩]

theorem exact284682RawTermsValid :
    exact284682RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284682 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68494⟩⟩) exact284682RawTerms .large 284681 .exactZero (none)

def event284683 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69174⟩⟩) 0 ⟨68494⟩ 284682

def event284684 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69174⟩⟩) (.authority (.operator))

def exact284685RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨69174⟩⟩]⟩, (1)⟩]

theorem exact284685RawTermsValid :
    exact284685RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284685 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69174⟩⟩) exact284685RawTerms (.finite 8192) 284684 .exactZero (none)

def event284686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event284687 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event284688 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68903⟩⟩) 0 ⟨65285⟩ 284674

def event284689 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68903⟩⟩) 1 ⟨136⟩ 284687

def event284690 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68903⟩⟩) (.sum [.predecessor 0 284688 .coefficient, .predecessor 1 284689 .coefficient])

def event284691 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68903⟩⟩) (.finite 784)

def event284692 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68904⟩⟩) 0 ⟨68903⟩ 284691

def event284693 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68904⟩⟩) (.identity (.predecessor 0 284692 .coefficient))

def exact284694RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], []⟩, (1)⟩]

theorem exact284694RawTermsValid :
    exact284694RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284694 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68904⟩⟩) exact284694RawTerms (.finite 784) 284693 .exactZero (none)

def event284695 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact284696RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact284696RawTermsValid :
    exact284696RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284696 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact284696RawTerms .large 284695 .exactZero (none)

def event284697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68905⟩⟩) 0 ⟨6908⟩ 284696

def event284698 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68905⟩⟩) 1 ⟨68904⟩ 284694

def event284699 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68905⟩⟩) (.product (.predecessor 0 284697 .coefficient) (.predecessor 1 284698 .coefficient) (⟨false, false, none, none, none⟩))

def event284700 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68905⟩⟩, .operator (⟨284696, 0⟩, ⟨284694, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact284701RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact284701RawTermsValid :
    exact284701RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284701 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68905⟩⟩) exact284701RawTerms .large 284699 .exactZero (none)

def event284702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 284678

def event284703 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact284704RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact284704RawTermsValid :
    exact284704RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284704 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact284704RawTerms .large 284703 .exactZero (none)

def event284705 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7276⟩⟩) 0 ⟨7178⟩ 284704

def event284706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7276⟩⟩) (.identity (.predecessor 0 284705 .coefficient))

def exact284707RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩]

theorem exact284707RawTermsValid :
    exact284707RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284707 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7276⟩⟩) exact284707RawTerms .large 284706 .exactZero (none)

def event284708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9541⟩⟩) 0 ⟨7276⟩ 284707

def event284709 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9541⟩⟩) (.authority (.operator))

def exact284710RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩]

theorem exact284710RawTermsValid :
    exact284710RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284710 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9541⟩⟩) exact284710RawTerms (.finite 8192) 284709 .exactZero (none)

def event284711 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9542⟩⟩) 0 ⟨9541⟩ 284710

def event284712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9542⟩⟩) 1 ⟨2370⟩ 284644

def event284713 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9542⟩⟩) (.scale (.predecessor 0 284711 .coefficient) (.value (.predecessor 1 284712 .coefficient)))

def exact284714RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩]

theorem exact284714RawTermsValid :
    exact284714RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284714 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9542⟩⟩) exact284714RawTerms (.finite 8192) 284713 .exactZero (none)

def event284715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7294⟩⟩) 0 ⟨7178⟩ 284704

def event284716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7294⟩⟩) (.identity (.predecessor 0 284715 .coefficient))

def exact284717RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩]⟩, (1)⟩]

theorem exact284717RawTermsValid :
    exact284717RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284717 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7294⟩⟩) exact284717RawTerms .large 284716 .exactZero (none)

def event284718 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9543⟩⟩) 0 ⟨7294⟩ 284717

def event284719 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9543⟩⟩) 1 ⟨9542⟩ 284714

def event284720 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9543⟩⟩) (.product (.predecessor 0 284718 .coefficient) (.predecessor 1 284719 .coefficient) (⟨false, false, none, none, none⟩))

def event284721 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9543⟩⟩, .operator (⟨284717, 0⟩, ⟨284714, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩)

def exact284722RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩]

theorem exact284722RawTermsValid :
    exact284722RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284722 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9543⟩⟩) exact284722RawTerms .large 284720 .exactZero (none)

def event284723 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68906⟩⟩) 0 ⟨9543⟩ 284722

def event284724 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68906⟩⟩) 1 ⟨68905⟩ 284701

def event284725 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68906⟩⟩) (.sum [.predecessor 0 284723 .coefficient, .predecessor 1 284724 .coefficient])

def exact284726RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284726RawTermsValid :
    exact284726RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284726 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68906⟩⟩) exact284726RawTerms .large 284725 .exactZero (none)

def event284727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69177⟩⟩) 0 ⟨68906⟩ 284726

def event284728 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69177⟩⟩) 1 ⟨69174⟩ 284685

def event284729 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69177⟩⟩) (.product (.predecessor 0 284727 .coefficient) (.predecessor 1 284728 .coefficient) (⟨false, false, none, none, none⟩))

def event284730 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69177⟩⟩, .operator (⟨284726, 0⟩, ⟨284685, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69174⟩⟩]⟩, (1)⟩)

def event284731 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69177⟩⟩, .operator (⟨284726, 1⟩, ⟨284685, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69174⟩⟩]⟩, (-1)⟩)

def event284732 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨69177⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69174⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69174⟩⟩) ⟨68494⟩ 284682)

def event284733 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69177⟩⟩, .relation 284732 0, ⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨68494⟩⟩]⟩, (-1)⟩)

def exact284734RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69174⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨68494⟩⟩]⟩, (-1)⟩]

theorem exact284734RawTermsValid :
    exact284734RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284734 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69177⟩⟩) exact284734RawTerms .large 284729 .exactZero (none)

def event284735 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65740⟩⟩) 0 ⟨65285⟩ 284674

def event284736 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65740⟩⟩) (.authority (.programFamilyFact))

def exact284737RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], []⟩, (1)⟩]

theorem exact284737RawTermsValid :
    exact284737RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284737 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65740⟩⟩) exact284737RawTerms (.finite 28) 284736 .exactZero (none)

def event284738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65742⟩⟩) 0 ⟨6908⟩ 284696

def event284739 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65742⟩⟩) 1 ⟨65740⟩ 284737

def event284740 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65742⟩⟩) (.product (.predecessor 0 284738 .coefficient) (.predecessor 1 284739 .coefficient) (⟨false, true, none, none, some 1⟩))

def event284741 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65742⟩⟩, .operator (⟨284696, 0⟩, ⟨284737, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact284742RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact284742RawTermsValid :
    exact284742RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284742 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65742⟩⟩) exact284742RawTerms .large 284740 .exactZero (none)

def event284743 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7188⟩⟩) 0 ⟨7177⟩ 284678

def event284744 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7188⟩⟩) (.authority (.operator))

def exact284745RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩]

theorem exact284745RawTermsValid :
    exact284745RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284745 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7188⟩⟩) exact284745RawTerms .large 284744 .exactZero (none)

def event284746 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65743⟩⟩) 0 ⟨7188⟩ 284745

def event284747 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65743⟩⟩) 1 ⟨65742⟩ 284742

def event284748 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65743⟩⟩) (.sum [.predecessor 0 284746 .coefficient, .predecessor 1 284747 .coefficient])

def exact284749RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284749RawTermsValid :
    exact284749RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284749 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65743⟩⟩) exact284749RawTerms .large 284748 .exactZero (none)

def event284750 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69178⟩⟩) 0 ⟨65743⟩ 284749

def event284751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69178⟩⟩) 1 ⟨69177⟩ 284734

def event284752 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69178⟩⟩) (.sum [.predecessor 0 284750 .coefficient, .predecessor 1 284751 .coefficient])

def exact284753RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69174⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨68494⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284753RawTermsValid :
    exact284753RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284753 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69178⟩⟩) exact284753RawTerms .large 284752 .exactZero (none)

def event284754 : Event := .preFoldPolynomial 284753 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69174⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨68494⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact284755RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69174⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨68494⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event284755 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨69178⟩⟩) 284754 exact284755RawTerms .large 284752 .exactZero (none)

def event284756 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨65285⟩⟩) ⟨⟨67⟩, ⟨46⟩, ⟨135⟩⟩ ⟨284592, 284756⟩

def event284757 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨67713⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67710⟩⟩]⟩) (1) 0 2 (.universal 284756 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67710⟩⟩]⟩) (none) 284755)

def event284758 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67713⟩⟩, .relation 284757 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩)

def event284759 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67713⟩⟩, .relation 284757 1, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69174⟩⟩]⟩, (-1)⟩)

def event284760 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67713⟩⟩, .relation 284757 2, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨68494⟩⟩]⟩, (1)⟩)

def event284761 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67713⟩⟩, .relation 284757 3, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact284762RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69174⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨68494⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284762RawTermsValid :
    exact284762RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284762 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67713⟩⟩) exact284762RawTerms .large 284588 (.finite 202072841853861888) (some (284590))

def event284763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69176⟩⟩) 0 ⟨67713⟩ 284762

def event284764 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69176⟩⟩) 1 ⟨69175⟩ 284578

def event284765 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69176⟩⟩) (.sum [.predecessor 0 284763 .coefficient, .predecessor 1 284764 .coefficient])

def event284766 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69176⟩⟩, .operator (⟨284762, 2⟩, ⟨284578, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨68494⟩⟩]⟩, (-1)⟩)

def event284767 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69176⟩⟩, .operator (⟨284762, 1⟩, ⟨284578, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69174⟩⟩]⟩, (1)⟩)

def event284768 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69176⟩⟩) (.sum [.result 284762 .summary, .result 284578 .summary])

def exact284769RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284769RawTermsValid :
    exact284769RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284769 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69176⟩⟩) exact284769RawTerms .large 284765 (.finite 2998054127048462696448) (some (284768))

def event284770 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69705⟩⟩) 0 ⟨69176⟩ 284769

def event284771 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69705⟩⟩) 1 ⟨69703⟩ 284494

def event284772 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69705⟩⟩) (.product (.predecessor 0 284770 .coefficient) (.predecessor 1 284771 .coefficient) (⟨false, false, none, none, none⟩))

def event284773 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69705⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨69703⟩⟩]⟩) [⟨.result 284494 .coefficient, false, none⟩])

def event284774 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69705⟩⟩) (.product (.result 284769 .summary) (.transfer 284773) (⟨false, false, none, none, none⟩))

def event284775 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69705⟩⟩, .operator (⟨284769, 0⟩, ⟨284494, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩]⟩, (1)⟩)

def event284776 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69705⟩⟩, .operator (⟨284769, 1⟩, ⟨284494, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩]⟩, (-1)⟩)

def event284777 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨69705⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69703⟩⟩) ⟨68628⟩ 284491)

def event284778 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69705⟩⟩, .relation 284777 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨68628⟩⟩]⟩, (-1)⟩)

def exact284779RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨68628⟩⟩]⟩, (-1)⟩]

theorem exact284779RawTermsValid :
    exact284779RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284779 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69705⟩⟩) exact284779RawTerms .large 284772 (.finite 32191361068277440720800338411520) (some (284774))

def event284780 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67957⟩⟩) 0 ⟨65741⟩ 13753

def event284781 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67957⟩⟩) (.authority (.relationPreimageSource ⟨76⟩))

def exact284782RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨67957⟩⟩]⟩, (1)⟩]

theorem exact284782RawTermsValid :
    exact284782RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284782 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67957⟩⟩) exact284782RawTerms (.finite 5647228698) 284781 .exactZero (none)

def event284783 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67959⟩⟩) 0 ⟨67957⟩ 284782

def event284784 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67959⟩⟩) 1 ⟨2370⟩ 4

def event284785 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67959⟩⟩) (.scale (.predecessor 0 284783 .coefficient) (.value (.predecessor 1 284784 .coefficient)))

def exact284786RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨67957⟩⟩]⟩, (1)⟩]

theorem exact284786RawTermsValid :
    exact284786RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284786 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67959⟩⟩) exact284786RawTerms (.finite 5647228698) 284785 .exactZero (none)

def event284787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67960⟩⟩) 0 ⟨5491⟩ 280745

def event284788 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67960⟩⟩) 1 ⟨67959⟩ 284786

def event284789 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67960⟩⟩) (.product (.predecessor 0 284787 .coefficient) (.predecessor 1 284788 .coefficient) (⟨false, false, none, none, none⟩))

def event284790 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67960⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨67957⟩⟩]⟩) [⟨.result 284782 .coefficient, false, none⟩])

def event284791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67960⟩⟩) (.product (.result 280745 .summary) (.transfer 284790) (⟨false, false, none, none, none⟩))

def event284792 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67960⟩⟩, .operator (⟨280745, 0⟩, ⟨284786, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67957⟩⟩]⟩, (1)⟩)

def event284793 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨67958⟩⟩)

def event284794 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event284795 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event284796 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event284797 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event284798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event284799 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event284800 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event284801 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event284802 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 284801

def event284803 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 284799

def event284804 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 284802 .coefficient) (.value (.predecessor 1 284803 .coefficient)))

def event284805 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event284806 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 284805

def event284807 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 284797

def event284808 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 284806 .coefficient, .predecessor 1 284807 .coefficient])

def event284809 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event284810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 284809

def event284811 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 284795

def event284812 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 284811 .coefficient))

def event284813 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event284814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25658⟩⟩) 0 ⟨5487⟩ 284813

def event284815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25658⟩⟩) (.authority (.programFamilyFact))

def exact284816RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩], []⟩, (1)⟩]

theorem exact284816RawTermsValid :
    exact284816RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284816 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25658⟩⟩) exact284816RawTerms (.finite 28) 284815 .exactZero (none)

def event284817 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65283⟩⟩) 0 ⟨5487⟩ 284813

def event284818 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65283⟩⟩) (.authority (.programFamilyFact))

def exact284819RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65283⟩⟩], []⟩, (1)⟩]

theorem exact284819RawTermsValid :
    exact284819RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284819 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65283⟩⟩) exact284819RawTerms (.finite 28) 284818 .exactZero (none)

def event284820 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65284⟩⟩) 0 ⟨65283⟩ 284819

def event284821 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65284⟩⟩) 1 ⟨25658⟩ 284816

def event284822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65284⟩⟩) (.product (.predecessor 0 284820 .coefficient) (.predecessor 1 284821 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event284823 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65284⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], []⟩) [⟨.result 284819 .coefficient, true, some 1⟩, ⟨.result 284816 .coefficient, true, some 1⟩])

def event284824 : Event := .survivorFold (1) 284823

def exact284825RawTerms : List Term := []

theorem exact284825RawTermsValid :
    exact284825RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284825 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65284⟩⟩) exact284825RawTerms (.finite 784) 284822 (.finite 784) (some (284823))

def event284826 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65285⟩⟩) 0 ⟨65284⟩ 284825

def event284827 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65285⟩⟩) (.identity (.predecessor 0 284826 .coefficient))

def event284828 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65285⟩⟩) (.finite 784)

def event284829 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65740⟩⟩) 0 ⟨65285⟩ 284828

def event284830 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65740⟩⟩) (.authority (.programFamilyFact))

def exact284831RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], []⟩, (1)⟩]

theorem exact284831RawTermsValid :
    exact284831RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284831 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65740⟩⟩) exact284831RawTerms (.finite 28) 284830 .exactZero (none)

def event284832 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65741⟩⟩) 0 ⟨65740⟩ 284831

def event284833 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65741⟩⟩) (.identity (.predecessor 0 284832 .coefficient))

def event284834 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65741⟩⟩) (.finite 28)

def event284835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67957⟩⟩) 0 ⟨65741⟩ 284834

def event284836 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67957⟩⟩) (.authority (.relationPreimageSource ⟨76⟩))

def exact284837RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨67957⟩⟩]⟩, (1)⟩]

theorem exact284837RawTermsValid :
    exact284837RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284837 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67957⟩⟩) exact284837RawTerms (.finite 5647228698) 284836 .exactZero (none)

def event284838 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact284839RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact284839RawTermsValid :
    exact284839RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284839 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact284839RawTerms .large 284838 .exactZero (none)

def event284840 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67958⟩⟩) 0 ⟨35⟩ 284839

def event284841 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67958⟩⟩) 1 ⟨67957⟩ 284837

def event284842 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67958⟩⟩) (.product (.predecessor 0 284840 .coefficient) (.predecessor 1 284841 .coefficient) (⟨false, false, none, none, none⟩))

def event284843 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67958⟩⟩, .operator (⟨284839, 0⟩, ⟨284837, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67957⟩⟩]⟩, (1)⟩)

def exact284844RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67957⟩⟩]⟩, (1)⟩]

theorem exact284844RawTermsValid :
    exact284844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284844 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67958⟩⟩) exact284844RawTerms .large 284842 .exactZero (none)

def event284845 : Event := .preFoldPolynomial 284844 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67957⟩⟩]⟩, (1)⟩] .exactZero none

def exact284846RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67957⟩⟩]⟩, (1)⟩]

def event284846 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨67958⟩⟩) 284845 exact284846RawTerms .large 284842 .exactZero (none)

def event284847 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨69716⟩⟩)

def event284848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event284849 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event284850 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event284851 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event284852 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event284853 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event284854 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event284855 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event284856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 284855

def event284857 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 284853

def event284858 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 284856 .coefficient) (.value (.predecessor 1 284857 .coefficient)))

def event284859 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event284860 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 284859

def event284861 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 284851

def event284862 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 284860 .coefficient, .predecessor 1 284861 .coefficient])

def event284863 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event284864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 284863

def event284865 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 284849

def event284866 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 284865 .coefficient))

def event284867 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event284868 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25658⟩⟩) 0 ⟨5487⟩ 284867

def event284869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25658⟩⟩) (.authority (.programFamilyFact))

def exact284870RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩], []⟩, (1)⟩]

theorem exact284870RawTermsValid :
    exact284870RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284870 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25658⟩⟩) exact284870RawTerms (.finite 28) 284869 .exactZero (none)

def event284871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65283⟩⟩) 0 ⟨5487⟩ 284867

def event284872 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65283⟩⟩) (.authority (.programFamilyFact))

def exact284873RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65283⟩⟩], []⟩, (1)⟩]

theorem exact284873RawTermsValid :
    exact284873RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284873 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65283⟩⟩) exact284873RawTerms (.finite 28) 284872 .exactZero (none)

def event284874 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65284⟩⟩) 0 ⟨65283⟩ 284873

def event284875 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65284⟩⟩) 1 ⟨25658⟩ 284870

def event284876 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65284⟩⟩) (.product (.predecessor 0 284874 .coefficient) (.predecessor 1 284875 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event284877 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65284⟩⟩, .operator (⟨284873, 0⟩, ⟨284870, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], []⟩, (1)⟩)

def exact284878RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], []⟩, (1)⟩]

theorem exact284878RawTermsValid :
    exact284878RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284878 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65284⟩⟩) exact284878RawTerms (.finite 784) 284876 .exactZero (none)

def event284879 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65285⟩⟩) 0 ⟨65284⟩ 284878

def event284880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65285⟩⟩) (.identity (.predecessor 0 284879 .coefficient))

def event284881 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65285⟩⟩) (.finite 784)

def event284882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65740⟩⟩) 0 ⟨65285⟩ 284881

def event284883 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65740⟩⟩) (.authority (.programFamilyFact))

def exact284884RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], []⟩, (1)⟩]

theorem exact284884RawTermsValid :
    exact284884RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284884 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65740⟩⟩) exact284884RawTerms (.finite 28) 284883 .exactZero (none)

def event284885 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65741⟩⟩) 0 ⟨65740⟩ 284884

def event284886 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65741⟩⟩) (.identity (.predecessor 0 284885 .coefficient))

def event284887 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65741⟩⟩) (.finite 28)

def event284888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68626⟩⟩) 0 ⟨65741⟩ 284887

def event284889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68626⟩⟩) (.authority (.programFamilyFact))

def event284890 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68626⟩⟩) (.finite 3720)

def event284891 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event284892 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68628⟩⟩) 0 ⟨7177⟩ 284891

def event284893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68628⟩⟩) 1 ⟨68626⟩ 284890

def event284894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68628⟩⟩) (.authority (.operator))

def exact284895RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68628⟩⟩]⟩, (1)⟩]

theorem exact284895RawTermsValid :
    exact284895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68628⟩⟩) exact284895RawTerms .large 284894 .exactZero (none)

def event284896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69703⟩⟩) 0 ⟨68628⟩ 284895

def event284897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69703⟩⟩) (.authority (.operator))

def exact284898RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨69703⟩⟩]⟩, (1)⟩]

theorem exact284898RawTermsValid :
    exact284898RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284898 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69703⟩⟩) exact284898RawTerms (.finite 8192) 284897 .exactZero (none)

def event284899 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event284900 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event284901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68983⟩⟩) 0 ⟨65741⟩ 284887

def event284902 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68983⟩⟩) 1 ⟨136⟩ 284900

def event284903 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68983⟩⟩) (.sum [.predecessor 0 284901 .coefficient, .predecessor 1 284902 .coefficient])

def event284904 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68983⟩⟩) (.finite 28)

def event284905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68984⟩⟩) 0 ⟨68983⟩ 284904

def event284906 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68984⟩⟩) (.identity (.predecessor 0 284905 .coefficient))

def exact284907RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], []⟩, (1)⟩]

theorem exact284907RawTermsValid :
    exact284907RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284907 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68984⟩⟩) exact284907RawTerms (.finite 28) 284906 .exactZero (none)

def event284908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact284909RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact284909RawTermsValid :
    exact284909RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284909 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact284909RawTerms .large 284908 .exactZero (none)

def event284910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68985⟩⟩) 0 ⟨6908⟩ 284909

def event284911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68985⟩⟩) 1 ⟨68984⟩ 284907

def event284912 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68985⟩⟩) (.product (.predecessor 0 284910 .coefficient) (.predecessor 1 284911 .coefficient) (⟨false, false, none, none, none⟩))

def event284913 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68985⟩⟩, .operator (⟨284909, 0⟩, ⟨284907, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact284914RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact284914RawTermsValid :
    exact284914RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284914 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68985⟩⟩) exact284914RawTerms .large 284912 .exactZero (none)

def event284915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7188⟩⟩) 0 ⟨7177⟩ 284891

def event284916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7188⟩⟩) (.authority (.operator))

def exact284917RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩]

theorem exact284917RawTermsValid :
    exact284917RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284917 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7188⟩⟩) exact284917RawTerms .large 284916 .exactZero (none)

def event284918 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68986⟩⟩) 0 ⟨7188⟩ 284917

def event284919 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68986⟩⟩) 1 ⟨68985⟩ 284914

def event284920 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68986⟩⟩) (.sum [.predecessor 0 284918 .coefficient, .predecessor 1 284919 .coefficient])

def exact284921RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact284921RawTermsValid :
    exact284921RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event284921 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68986⟩⟩) exact284921RawTerms .large 284920 .exactZero (none)

def event284922 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69704⟩⟩) 0 ⟨68986⟩ 284921

def event284923 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69704⟩⟩) 1 ⟨69703⟩ 284898

def event284924 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69704⟩⟩) (.product (.predecessor 0 284922 .coefficient) (.predecessor 1 284923 .coefficient) (⟨false, false, none, none, none⟩))

def event284925 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69704⟩⟩, .operator (⟨284921, 0⟩, ⟨284898, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩]⟩, (1)⟩)

def event284926 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69704⟩⟩, .operator (⟨284921, 1⟩, ⟨284898, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩]⟩, (-1)⟩)

def event284927 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨69704⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69703⟩⟩) ⟨68628⟩ 284895)

def eventLeaf17792 : Array AnnotatedEvent := #[
  { event := event284672
    frameStart := 284640 },
  { event := event284673
    frameStart := 284640 },
  { event := event284674
    frameStart := 284640 },
  { event := event284675
    frameStart := 284640 },
  { event := event284676
    frameStart := 284640 },
  { event := event284677
    frameStart := 284640 },
  { event := event284678
    frameStart := 284640 },
  { event := event284679
    frameStart := 284640 },
  { event := event284680
    frameStart := 284640 },
  { event := event284681
    frameStart := 284640 },
  { event := event284682
    frameStart := 284640 },
  { event := event284683
    frameStart := 284640 },
  { event := event284684
    frameStart := 284640 },
  { event := event284685
    frameStart := 284640 },
  { event := event284686
    frameStart := 284640 },
  { event := event284687
    frameStart := 284640 }
]

def eventLeaf17793 : Array AnnotatedEvent := #[
  { event := event284688
    frameStart := 284640 },
  { event := event284689
    frameStart := 284640 },
  { event := event284690
    frameStart := 284640 },
  { event := event284691
    frameStart := 284640 },
  { event := event284692
    frameStart := 284640 },
  { event := event284693
    frameStart := 284640 },
  { event := event284694
    frameStart := 284640 },
  { event := event284695
    frameStart := 284640 },
  { event := event284696
    frameStart := 284640 },
  { event := event284697
    frameStart := 284640 },
  { event := event284698
    frameStart := 284640 },
  { event := event284699
    frameStart := 284640 },
  { event := event284700
    frameStart := 284640 },
  { event := event284701
    frameStart := 284640 },
  { event := event284702
    frameStart := 284640 },
  { event := event284703
    frameStart := 284640 }
]

def eventLeaf17794 : Array AnnotatedEvent := #[
  { event := event284704
    frameStart := 284640 },
  { event := event284705
    frameStart := 284640 },
  { event := event284706
    frameStart := 284640 },
  { event := event284707
    frameStart := 284640 },
  { event := event284708
    frameStart := 284640 },
  { event := event284709
    frameStart := 284640 },
  { event := event284710
    frameStart := 284640 },
  { event := event284711
    frameStart := 284640 },
  { event := event284712
    frameStart := 284640 },
  { event := event284713
    frameStart := 284640 },
  { event := event284714
    frameStart := 284640 },
  { event := event284715
    frameStart := 284640 },
  { event := event284716
    frameStart := 284640 },
  { event := event284717
    frameStart := 284640 },
  { event := event284718
    frameStart := 284640 },
  { event := event284719
    frameStart := 284640 }
]

def eventLeaf17795 : Array AnnotatedEvent := #[
  { event := event284720
    frameStart := 284640 },
  { event := event284721
    frameStart := 284640 },
  { event := event284722
    frameStart := 284640 },
  { event := event284723
    frameStart := 284640 },
  { event := event284724
    frameStart := 284640 },
  { event := event284725
    frameStart := 284640 },
  { event := event284726
    frameStart := 284640 },
  { event := event284727
    frameStart := 284640 },
  { event := event284728
    frameStart := 284640 },
  { event := event284729
    frameStart := 284640 },
  { event := event284730
    frameStart := 284640 },
  { event := event284731
    frameStart := 284640 },
  { event := event284732
    frameStart := 284640 },
  { event := event284733
    frameStart := 284640 },
  { event := event284734
    frameStart := 284640 },
  { event := event284735
    frameStart := 284640 }
]

def eventLeaf17796 : Array AnnotatedEvent := #[
  { event := event284736
    frameStart := 284640 },
  { event := event284737
    frameStart := 284640 },
  { event := event284738
    frameStart := 284640 },
  { event := event284739
    frameStart := 284640 },
  { event := event284740
    frameStart := 284640 },
  { event := event284741
    frameStart := 284640 },
  { event := event284742
    frameStart := 284640 },
  { event := event284743
    frameStart := 284640 },
  { event := event284744
    frameStart := 284640 },
  { event := event284745
    frameStart := 284640 },
  { event := event284746
    frameStart := 284640 },
  { event := event284747
    frameStart := 284640 },
  { event := event284748
    frameStart := 284640 },
  { event := event284749
    frameStart := 284640 },
  { event := event284750
    frameStart := 284640 },
  { event := event284751
    frameStart := 284640 }
]

def eventLeaf17797 : Array AnnotatedEvent := #[
  { event := event284752
    frameStart := 284640 },
  { event := event284753
    frameStart := 284640 },
  { event := event284754
    frameStart := 284640 },
  { event := event284755
    frameStart := 284640 },
  { event := event284756
    frameStart := 0 },
  { event := event284757
    frameStart := 0 },
  { event := event284758
    frameStart := 0 },
  { event := event284759
    frameStart := 0 },
  { event := event284760
    frameStart := 0 },
  { event := event284761
    frameStart := 0 },
  { event := event284762
    frameStart := 0 },
  { event := event284763
    frameStart := 0 },
  { event := event284764
    frameStart := 0 },
  { event := event284765
    frameStart := 0 },
  { event := event284766
    frameStart := 0 },
  { event := event284767
    frameStart := 0 }
]

def eventLeaf17798 : Array AnnotatedEvent := #[
  { event := event284768
    frameStart := 0 },
  { event := event284769
    frameStart := 0 },
  { event := event284770
    frameStart := 0 },
  { event := event284771
    frameStart := 0 },
  { event := event284772
    frameStart := 0 },
  { event := event284773
    frameStart := 0 },
  { event := event284774
    frameStart := 0 },
  { event := event284775
    frameStart := 0 },
  { event := event284776
    frameStart := 0 },
  { event := event284777
    frameStart := 0 },
  { event := event284778
    frameStart := 0 },
  { event := event284779
    frameStart := 0 },
  { event := event284780
    frameStart := 0 },
  { event := event284781
    frameStart := 0 },
  { event := event284782
    frameStart := 0 },
  { event := event284783
    frameStart := 0 }
]

def eventLeaf17799 : Array AnnotatedEvent := #[
  { event := event284784
    frameStart := 0 },
  { event := event284785
    frameStart := 0 },
  { event := event284786
    frameStart := 0 },
  { event := event284787
    frameStart := 0 },
  { event := event284788
    frameStart := 0 },
  { event := event284789
    frameStart := 0 },
  { event := event284790
    frameStart := 0 },
  { event := event284791
    frameStart := 0 },
  { event := event284792
    frameStart := 0 },
  { event := event284793
    frameStart := 284793 },
  { event := event284794
    frameStart := 284793 },
  { event := event284795
    frameStart := 284793 },
  { event := event284796
    frameStart := 284793 },
  { event := event284797
    frameStart := 284793 },
  { event := event284798
    frameStart := 284793 },
  { event := event284799
    frameStart := 284793 }
]

def eventLeaf17800 : Array AnnotatedEvent := #[
  { event := event284800
    frameStart := 284793 },
  { event := event284801
    frameStart := 284793 },
  { event := event284802
    frameStart := 284793 },
  { event := event284803
    frameStart := 284793 },
  { event := event284804
    frameStart := 284793 },
  { event := event284805
    frameStart := 284793 },
  { event := event284806
    frameStart := 284793 },
  { event := event284807
    frameStart := 284793 },
  { event := event284808
    frameStart := 284793 },
  { event := event284809
    frameStart := 284793 },
  { event := event284810
    frameStart := 284793 },
  { event := event284811
    frameStart := 284793 },
  { event := event284812
    frameStart := 284793 },
  { event := event284813
    frameStart := 284793 },
  { event := event284814
    frameStart := 284793 },
  { event := event284815
    frameStart := 284793 }
]

def eventLeaf17801 : Array AnnotatedEvent := #[
  { event := event284816
    frameStart := 284793 },
  { event := event284817
    frameStart := 284793 },
  { event := event284818
    frameStart := 284793 },
  { event := event284819
    frameStart := 284793 },
  { event := event284820
    frameStart := 284793 },
  { event := event284821
    frameStart := 284793 },
  { event := event284822
    frameStart := 284793 },
  { event := event284823
    frameStart := 284793 },
  { event := event284824
    frameStart := 284793 },
  { event := event284825
    frameStart := 284793 },
  { event := event284826
    frameStart := 284793 },
  { event := event284827
    frameStart := 284793 },
  { event := event284828
    frameStart := 284793 },
  { event := event284829
    frameStart := 284793 },
  { event := event284830
    frameStart := 284793 },
  { event := event284831
    frameStart := 284793 }
]

def eventLeaf17802 : Array AnnotatedEvent := #[
  { event := event284832
    frameStart := 284793 },
  { event := event284833
    frameStart := 284793 },
  { event := event284834
    frameStart := 284793 },
  { event := event284835
    frameStart := 284793 },
  { event := event284836
    frameStart := 284793 },
  { event := event284837
    frameStart := 284793 },
  { event := event284838
    frameStart := 284793 },
  { event := event284839
    frameStart := 284793 },
  { event := event284840
    frameStart := 284793 },
  { event := event284841
    frameStart := 284793 },
  { event := event284842
    frameStart := 284793 },
  { event := event284843
    frameStart := 284793 },
  { event := event284844
    frameStart := 284793 },
  { event := event284845
    frameStart := 284793 },
  { event := event284846
    frameStart := 284793 },
  { event := event284847
    frameStart := 284847 }
]

def eventLeaf17803 : Array AnnotatedEvent := #[
  { event := event284848
    frameStart := 284847 },
  { event := event284849
    frameStart := 284847 },
  { event := event284850
    frameStart := 284847 },
  { event := event284851
    frameStart := 284847 },
  { event := event284852
    frameStart := 284847 },
  { event := event284853
    frameStart := 284847 },
  { event := event284854
    frameStart := 284847 },
  { event := event284855
    frameStart := 284847 },
  { event := event284856
    frameStart := 284847 },
  { event := event284857
    frameStart := 284847 },
  { event := event284858
    frameStart := 284847 },
  { event := event284859
    frameStart := 284847 },
  { event := event284860
    frameStart := 284847 },
  { event := event284861
    frameStart := 284847 },
  { event := event284862
    frameStart := 284847 },
  { event := event284863
    frameStart := 284847 }
]

def eventLeaf17804 : Array AnnotatedEvent := #[
  { event := event284864
    frameStart := 284847 },
  { event := event284865
    frameStart := 284847 },
  { event := event284866
    frameStart := 284847 },
  { event := event284867
    frameStart := 284847 },
  { event := event284868
    frameStart := 284847 },
  { event := event284869
    frameStart := 284847 },
  { event := event284870
    frameStart := 284847 },
  { event := event284871
    frameStart := 284847 },
  { event := event284872
    frameStart := 284847 },
  { event := event284873
    frameStart := 284847 },
  { event := event284874
    frameStart := 284847 },
  { event := event284875
    frameStart := 284847 },
  { event := event284876
    frameStart := 284847 },
  { event := event284877
    frameStart := 284847 },
  { event := event284878
    frameStart := 284847 },
  { event := event284879
    frameStart := 284847 }
]

def eventLeaf17805 : Array AnnotatedEvent := #[
  { event := event284880
    frameStart := 284847 },
  { event := event284881
    frameStart := 284847 },
  { event := event284882
    frameStart := 284847 },
  { event := event284883
    frameStart := 284847 },
  { event := event284884
    frameStart := 284847 },
  { event := event284885
    frameStart := 284847 },
  { event := event284886
    frameStart := 284847 },
  { event := event284887
    frameStart := 284847 },
  { event := event284888
    frameStart := 284847 },
  { event := event284889
    frameStart := 284847 },
  { event := event284890
    frameStart := 284847 },
  { event := event284891
    frameStart := 284847 },
  { event := event284892
    frameStart := 284847 },
  { event := event284893
    frameStart := 284847 },
  { event := event284894
    frameStart := 284847 },
  { event := event284895
    frameStart := 284847 }
]

def eventLeaf17806 : Array AnnotatedEvent := #[
  { event := event284896
    frameStart := 284847 },
  { event := event284897
    frameStart := 284847 },
  { event := event284898
    frameStart := 284847 },
  { event := event284899
    frameStart := 284847 },
  { event := event284900
    frameStart := 284847 },
  { event := event284901
    frameStart := 284847 },
  { event := event284902
    frameStart := 284847 },
  { event := event284903
    frameStart := 284847 },
  { event := event284904
    frameStart := 284847 },
  { event := event284905
    frameStart := 284847 },
  { event := event284906
    frameStart := 284847 },
  { event := event284907
    frameStart := 284847 },
  { event := event284908
    frameStart := 284847 },
  { event := event284909
    frameStart := 284847 },
  { event := event284910
    frameStart := 284847 },
  { event := event284911
    frameStart := 284847 }
]

def eventLeaf17807 : Array AnnotatedEvent := #[
  { event := event284912
    frameStart := 284847 },
  { event := event284913
    frameStart := 284847 },
  { event := event284914
    frameStart := 284847 },
  { event := event284915
    frameStart := 284847 },
  { event := event284916
    frameStart := 284847 },
  { event := event284917
    frameStart := 284847 },
  { event := event284918
    frameStart := 284847 },
  { event := event284919
    frameStart := 284847 },
  { event := event284920
    frameStart := 284847 },
  { event := event284921
    frameStart := 284847 },
  { event := event284922
    frameStart := 284847 },
  { event := event284923
    frameStart := 284847 },
  { event := event284924
    frameStart := 284847 },
  { event := event284925
    frameStart := 284847 },
  { event := event284926
    frameStart := 284847 },
  { event := event284927
    frameStart := 284847 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1112
