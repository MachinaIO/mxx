import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events694

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event177664 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12441⟩⟩) (.authority (.programFamilyFact))

def exact177665RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12441⟩⟩], []⟩, (1)⟩]

theorem exact177665RawTermsValid :
    exact177665RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177665 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12441⟩⟩) exact177665RawTerms (.finite 2) 177664 .exactZero (none)

def event177666 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15571⟩⟩) 0 ⟨12441⟩ 177665

def event177667 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15571⟩⟩) 1 ⟨15570⟩ 177662

def event177668 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15571⟩⟩) (.product (.predecessor 0 177666 .coefficient) (.predecessor 1 177667 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event177669 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15571⟩⟩, .operator (⟨177665, 0⟩, ⟨177662, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12441⟩⟩, ⟨.program ⟨257⟩, ⟨15570⟩⟩], []⟩, (1)⟩)

def exact177670RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12441⟩⟩, ⟨.program ⟨257⟩, ⟨15570⟩⟩], []⟩, (1)⟩]

theorem exact177670RawTermsValid :
    exact177670RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177670 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15571⟩⟩) exact177670RawTerms (.finite 4) 177668 .exactZero (none)

def event177671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15572⟩⟩) 0 ⟨15571⟩ 177670

def event177672 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15572⟩⟩) (.identity (.predecessor 0 177671 .coefficient))

def event177673 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15572⟩⟩) (.finite 4)

def event177674 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15820⟩⟩) 0 ⟨15572⟩ 177673

def event177675 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15820⟩⟩) (.authority (.programFamilyFact))

def exact177676RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], []⟩, (1)⟩]

theorem exact177676RawTermsValid :
    exact177676RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177676 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15820⟩⟩) exact177676RawTerms (.finite 2) 177675 .exactZero (none)

def event177677 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15821⟩⟩) 0 ⟨15820⟩ 177676

def event177678 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15821⟩⟩) (.identity (.predecessor 0 177677 .coefficient))

def event177679 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15821⟩⟩) (.finite 2)

def event177680 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17035⟩⟩) 0 ⟨15821⟩ 177679

def event177681 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17035⟩⟩) (.authority (.programFamilyFact))

def event177682 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17035⟩⟩) (.finite 3720)

def event177683 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event177684 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17036⟩⟩) 0 ⟨7177⟩ 177683

def event177685 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17036⟩⟩) 1 ⟨17035⟩ 177682

def event177686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17036⟩⟩) (.authority (.operator))

def exact177687RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17036⟩⟩]⟩, (1)⟩]

theorem exact177687RawTermsValid :
    exact177687RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177687 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17036⟩⟩) exact177687RawTerms .large 177686 .exactZero (none)

def event177688 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17866⟩⟩) 0 ⟨17036⟩ 177687

def event177689 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17866⟩⟩) (.authority (.operator))

def exact177690RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17866⟩⟩]⟩, (1)⟩]

theorem exact177690RawTermsValid :
    exact177690RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177690 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17866⟩⟩) exact177690RawTerms (.finite 8192) 177689 .exactZero (none)

def event177691 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event177692 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event177693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17222⟩⟩) 0 ⟨15821⟩ 177679

def event177694 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17222⟩⟩) 1 ⟨136⟩ 177692

def event177695 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17222⟩⟩) (.sum [.predecessor 0 177693 .coefficient, .predecessor 1 177694 .coefficient])

def event177696 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17222⟩⟩) (.finite 2)

def event177697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17223⟩⟩) 0 ⟨17222⟩ 177696

def event177698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17223⟩⟩) (.identity (.predecessor 0 177697 .coefficient))

def exact177699RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], []⟩, (1)⟩]

theorem exact177699RawTermsValid :
    exact177699RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177699 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17223⟩⟩) exact177699RawTerms (.finite 2) 177698 .exactZero (none)

def event177700 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact177701RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact177701RawTermsValid :
    exact177701RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177701 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact177701RawTerms .large 177700 .exactZero (none)

def event177702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17224⟩⟩) 0 ⟨6908⟩ 177701

def event177703 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17224⟩⟩) 1 ⟨17223⟩ 177699

def event177704 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17224⟩⟩) (.product (.predecessor 0 177702 .coefficient) (.predecessor 1 177703 .coefficient) (⟨false, false, none, none, none⟩))

def event177705 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17224⟩⟩, .operator (⟨177701, 0⟩, ⟨177699, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact177706RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact177706RawTermsValid :
    exact177706RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177706 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17224⟩⟩) exact177706RawTerms .large 177704 .exactZero (none)

def event177707 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 177683

def event177708 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact177709RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact177709RawTermsValid :
    exact177709RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177709 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact177709RawTerms .large 177708 .exactZero (none)

def event177710 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17225⟩⟩) 0 ⟨7179⟩ 177709

def event177711 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17225⟩⟩) 1 ⟨17224⟩ 177706

def event177712 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17225⟩⟩) (.sum [.predecessor 0 177710 .coefficient, .predecessor 1 177711 .coefficient])

def exact177713RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177713RawTermsValid :
    exact177713RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177713 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17225⟩⟩) exact177713RawTerms .large 177712 .exactZero (none)

def event177714 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17867⟩⟩) 0 ⟨17225⟩ 177713

def event177715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17867⟩⟩) 1 ⟨17866⟩ 177690

def event177716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17867⟩⟩) (.product (.predecessor 0 177714 .coefficient) (.predecessor 1 177715 .coefficient) (⟨false, false, none, none, none⟩))

def event177717 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17867⟩⟩, .operator (⟨177713, 0⟩, ⟨177690, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17866⟩⟩]⟩, (1)⟩)

def event177718 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17867⟩⟩, .operator (⟨177713, 1⟩, ⟨177690, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17866⟩⟩]⟩, (-1)⟩)

def event177719 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17867⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17866⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17866⟩⟩) ⟨17036⟩ 177687)

def event177720 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17867⟩⟩, .relation 177719 0, ⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨17036⟩⟩]⟩, (-1)⟩)

def exact177721RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17866⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨17036⟩⟩]⟩, (-1)⟩]

theorem exact177721RawTermsValid :
    exact177721RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177721 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17867⟩⟩) exact177721RawTerms .large 177716 .exactZero (none)

def event177722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16094⟩⟩) 0 ⟨15821⟩ 177679

def event177723 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16094⟩⟩) (.authority (.programFamilyFact))

def exact177724RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16094⟩⟩], []⟩, (1)⟩]

theorem exact177724RawTermsValid :
    exact177724RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177724 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16094⟩⟩) exact177724RawTerms (.finite 2) 177723 .exactZero (none)

def event177725 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16097⟩⟩) 0 ⟨6908⟩ 177701

def event177726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16097⟩⟩) 1 ⟨16094⟩ 177724

def event177727 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16097⟩⟩) (.product (.predecessor 0 177725 .coefficient) (.predecessor 1 177726 .coefficient) (⟨false, true, none, none, some 1⟩))

def event177728 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16097⟩⟩, .operator (⟨177701, 0⟩, ⟨177724, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact177729RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact177729RawTermsValid :
    exact177729RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177729 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16097⟩⟩) exact177729RawTerms .large 177727 .exactZero (none)

def event177730 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7197⟩⟩) 0 ⟨7177⟩ 177683

def event177731 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7197⟩⟩) (.authority (.operator))

def exact177732RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩]

theorem exact177732RawTermsValid :
    exact177732RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177732 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7197⟩⟩) exact177732RawTerms .large 177731 .exactZero (none)

def event177733 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16098⟩⟩) 0 ⟨7197⟩ 177732

def event177734 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16098⟩⟩) 1 ⟨16097⟩ 177729

def event177735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16098⟩⟩) (.sum [.predecessor 0 177733 .coefficient, .predecessor 1 177734 .coefficient])

def exact177736RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177736RawTermsValid :
    exact177736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177736 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16098⟩⟩) exact177736RawTerms .large 177735 .exactZero (none)

def event177737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17872⟩⟩) 0 ⟨16098⟩ 177736

def event177738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17872⟩⟩) 1 ⟨17867⟩ 177721

def event177739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17872⟩⟩) (.sum [.predecessor 0 177737 .coefficient, .predecessor 1 177738 .coefficient])

def exact177740RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17866⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨17036⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177740RawTermsValid :
    exact177740RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177740 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17872⟩⟩) exact177740RawTerms .large 177739 .exactZero (none)

def event177741 : Event := .preFoldPolynomial 177740 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17866⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨17036⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact177742RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17866⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨17036⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event177742 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17872⟩⟩) 177741 exact177742RawTerms .large 177739 .exactZero (none)

def event177743 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15821⟩⟩) ⟨⟨76⟩, ⟨56⟩, ⟨135⟩⟩ ⟨177585, 177743⟩

def event177744 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16675⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16672⟩⟩]⟩) (1) 0 2 (.universal 177743 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16672⟩⟩]⟩) (none) 177742)

def event177745 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16675⟩⟩, .relation 177744 1, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩)

def event177746 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16675⟩⟩, .relation 177744 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17866⟩⟩]⟩, (-1)⟩)

def event177747 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16675⟩⟩, .relation 177744 2, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨17036⟩⟩]⟩, (1)⟩)

def event177748 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16675⟩⟩, .relation 177744 3, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact177749RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17866⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨17036⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177749RawTermsValid :
    exact177749RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177749 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16675⟩⟩) exact177749RawTerms .large 177581 (.finite 202072841853861888) (some (177583))

def event177750 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17869⟩⟩) 0 ⟨16675⟩ 177749

def event177751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17869⟩⟩) 1 ⟨17868⟩ 177571

def event177752 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17869⟩⟩) (.sum [.predecessor 0 177750 .coefficient, .predecessor 1 177751 .coefficient])

def event177753 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17869⟩⟩, .operator (⟨177749, 0⟩, ⟨177571, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17866⟩⟩]⟩, (1)⟩)

def event177754 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17869⟩⟩, .operator (⟨177749, 2⟩, ⟨177571, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨15820⟩⟩], [⟨.program ⟨257⟩, ⟨17036⟩⟩]⟩, (-1)⟩)

def event177755 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17869⟩⟩) (.sum [.result 177749 .summary, .result 177571 .summary])

def exact177756RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177756RawTermsValid :
    exact177756RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177756 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17869⟩⟩) exact177756RawTerms .large 177752 (.finite 32188807212483706889510625476608) (some (177755))

def event177757 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17870⟩⟩) 0 ⟨17869⟩ 177756

def event177758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17870⟩⟩) 1 ⟨7172⟩ 15882

def event177759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17870⟩⟩) (.product (.predecessor 0 177757 .coefficient) (.predecessor 1 177758 .coefficient) (⟨false, false, none, none, none⟩))

def event177760 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17870⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) [⟨.result 15878 .coefficient, false, none⟩])

def event177761 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17870⟩⟩) (.product (.result 177756 .summary) (.transfer 177760) (⟨false, false, none, none, none⟩))

def event177762 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17870⟩⟩, .operator (⟨177756, 0⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩)

def event177763 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17870⟩⟩, .operator (⟨177756, 1⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩)

def event177764 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17870⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7171⟩⟩) ⟨7051⟩ 15875)

def event177765 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17870⟩⟩, .relation 177764 0, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact177766RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177766RawTermsValid :
    exact177766RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177766 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17870⟩⟩) exact177766RawTerms .large 177759 (.finite 345624685687166110058245054666339432529920) (some (177761))

def event177767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7097⟩⟩) 0 ⟨6727⟩ 723

def event177768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7097⟩⟩) 1 ⟨7010⟩ 163653

def event177769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7097⟩⟩) (.tensor (.predecessor 0 177767 .coefficient) (.predecessor 1 177768 .coefficient) true false)

def event177770 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7097⟩⟩, .operator (⟨723, 0⟩, ⟨163653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact177771RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact177771RawTermsValid :
    exact177771RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177771 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7097⟩⟩) exact177771RawTerms .large 177769 .exactZero (none)

def event177772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9054⟩⟩) 0 ⟨6464⟩ 163523

def event177773 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9054⟩⟩) 1 ⟨7292⟩ 15896

def event177774 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9054⟩⟩) (.product (.predecessor 0 177772 .coefficient) (.predecessor 1 177773 .coefficient) (⟨false, false, none, none, none⟩))

def event177775 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9054⟩⟩, .operator (⟨163523, 0⟩, ⟨15896, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩)

def exact177776RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact177776RawTermsValid :
    exact177776RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177776 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9054⟩⟩) exact177776RawTerms .large 177774 .exactZero (none)

def event177777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9437⟩⟩) 0 ⟨9054⟩ 177776

def event177778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9437⟩⟩) 1 ⟨7097⟩ 177771

def event177779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9437⟩⟩) (.sum [.predecessor 0 177777 .coefficient, .predecessor 1 177778 .coefficient])

def exact177780RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177780RawTermsValid :
    exact177780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177780 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9437⟩⟩) exact177780RawTerms .large 177779 .exactZero (none)

def event177781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9438⟩⟩) 0 ⟨9437⟩ 177780

def event177782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9438⟩⟩) 1 ⟨118⟩ 31516

def event177783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9438⟩⟩) (.sum [.predecessor 0 177781 .coefficient, .predecessor 1 177782 .coefficient])

def event177784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9438⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) [⟨.result 31516 .coefficient, false, none⟩])

def event177785 : Event := .survivorFold (1) 177784

def exact177786RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177786RawTermsValid :
    exact177786RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177786 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9438⟩⟩) exact177786RawTerms .large 177783 (.finite 26) (some (177784))

def event177787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9489⟩⟩) 0 ⟨9438⟩ 177786

def event177788 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9489⟩⟩) 1 ⟨9438⟩ 177786

def event177789 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9489⟩⟩) (.sum [.predecessor 0 177787 .coefficient, .predecessor 1 177788 .coefficient])

def event177790 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9489⟩⟩, .operator (⟨177786, 1⟩, ⟨177786, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event177791 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9489⟩⟩, .operator (⟨177786, 0⟩, ⟨177786, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (-1)⟩)

def event177792 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9489⟩⟩) (.sum [.result 177786 .summary, .result 177786 .summary])

def exact177793RawTerms : List Term := []

theorem exact177793RawTermsValid :
    exact177793RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177793 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9489⟩⟩) exact177793RawTerms .large 177789 (.finite 52) (some (177792))

def event177794 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17871⟩⟩) 0 ⟨9489⟩ 177793

def event177795 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17871⟩⟩) 1 ⟨17870⟩ 177766

def event177796 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17871⟩⟩) (.sum [.predecessor 0 177794 .coefficient, .predecessor 1 177795 .coefficient])

def event177797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17871⟩⟩) (.sum [.result 177793 .summary, .result 177766 .summary])

def exact177798RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177798RawTermsValid :
    exact177798RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177798 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17871⟩⟩) exact177798RawTerms .large 177796 (.finite 345624685687166110058245054666339432529972) (some (177797))

def event177799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20774⟩⟩) 0 ⟨17871⟩ 177798

def event177800 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20774⟩⟩) 1 ⟨20773⟩ 177554

def event177801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20774⟩⟩) (.sum [.predecessor 0 177799 .coefficient, .predecessor 1 177800 .coefficient])

def event177802 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20774⟩⟩) (.sum [.result 177798 .summary, .result 177554 .summary])

def exact177803RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177803RawTermsValid :
    exact177803RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177803 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20774⟩⟩) exact177803RawTerms .large 177801 (.finite 691250426059631610003352154589745737891892) (some (177802))

def event177804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23994⟩⟩) 0 ⟨20774⟩ 177803

def event177805 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23994⟩⟩) 1 ⟨23993⟩ 177342

def event177806 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23994⟩⟩) (.sum [.predecessor 0 177804 .coefficient, .predecessor 1 177805 .coefficient])

def event177807 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23994⟩⟩) (.sum [.result 177803 .summary, .result 177342 .summary])

def exact177808RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177808RawTermsValid :
    exact177808RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177808 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23994⟩⟩) exact177808RawTerms .large 177806 (.finite 1036877221117396499835321299770218916085812) (some (177807))

def event177809 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34014⟩⟩) 0 ⟨23994⟩ 177808

def event177810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34014⟩⟩) 1 ⟨34013⟩ 177130

def event177811 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34014⟩⟩) (.sum [.predecessor 0 177809 .coefficient, .predecessor 1 177810 .coefficient])

def event177812 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34014⟩⟩) (.sum [.result 177808 .summary, .result 177130 .summary])

def exact177813RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177813RawTermsValid :
    exact177813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177813 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34014⟩⟩) exact177813RawTerms .large 177811 (.finite 1382506125545760169441014535464825839943732) (some (177812))

def event177814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53074⟩⟩) 0 ⟨34014⟩ 177813

def event177815 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53074⟩⟩) 1 ⟨53073⟩ 176918

def event177816 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53074⟩⟩) (.sum [.predecessor 0 177814 .coefficient, .predecessor 1 177815 .coefficient])

def event177817 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53074⟩⟩) (.sum [.result 177813 .summary, .result 176918 .summary])

def exact177818RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177818RawTermsValid :
    exact177818RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177818 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53074⟩⟩) exact177818RawTerms .large 177816 (.finite 1728139248715321398594155952187700255129652) (some (177817))

def event177819 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56054⟩⟩) 0 ⟨53074⟩ 177818

def event177820 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56054⟩⟩) 1 ⟨56053⟩ 176706

def event177821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56054⟩⟩) (.sum [.predecessor 0 177819 .coefficient, .predecessor 1 177820 .coefficient])

def event177822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56054⟩⟩) (.sum [.result 177818 .summary, .result 176706 .summary])

def exact177823RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177823RawTermsValid :
    exact177823RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177823 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56054⟩⟩) exact177823RawTerms .large 177821 (.finite 2073774481255481407521021459424708415979572) (some (177822))

def event177824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59034⟩⟩) 0 ⟨56054⟩ 177823

def event177825 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59034⟩⟩) 1 ⟨59033⟩ 176494

def event177826 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59034⟩⟩) (.sum [.predecessor 0 177824 .coefficient, .predecessor 1 177825 .coefficient])

def event177827 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59034⟩⟩) (.sum [.result 177823 .summary, .result 176494 .summary])

def exact177828RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177828RawTermsValid :
    exact177828RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177828 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59034⟩⟩) exact177828RawTerms .large 177826 (.finite 2419413932536838975995335147689984068157492) (some (177827))

def event177829 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62014⟩⟩) 0 ⟨59034⟩ 177828

def event177830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62014⟩⟩) 1 ⟨62013⟩ 176282

def event177831 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62014⟩⟩) (.sum [.predecessor 0 177829 .coefficient, .predecessor 1 177830 .coefficient])

def event177832 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62014⟩⟩) (.sum [.result 177828 .summary, .result 176282 .summary])

def exact177833RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177833RawTermsValid :
    exact177833RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177833 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62014⟩⟩) exact177833RawTerms .large 177831 (.finite 2765055493188795324243372926469393465999412) (some (177832))

def event177834 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64994⟩⟩) 0 ⟨62014⟩ 177833

def event177835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64994⟩⟩) 1 ⟨64993⟩ 176070

def event177836 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64994⟩⟩) (.sum [.predecessor 0 177834 .coefficient, .predecessor 1 177835 .coefficient])

def event177837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64994⟩⟩) (.sum [.result 177833 .summary, .result 176070 .summary])

def exact177838RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177838RawTermsValid :
    exact177838RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177838 : Event := .resultExact (⟨.program ⟨257⟩, ⟨64994⟩⟩) exact177838RawTerms .large 177836 (.finite 3110701272581949232038858886277070355169332) (some (177837))

def event177839 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70483⟩⟩) 0 ⟨64994⟩ 177838

def event177840 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70483⟩⟩) 1 ⟨70482⟩ 175858

def event177841 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70483⟩⟩) (.sum [.predecessor 0 177839 .coefficient, .predecessor 1 177840 .coefficient])

def event177842 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70483⟩⟩) (.sum [.result 177838 .summary, .result 175858 .summary])

def exact177843RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177843RawTermsValid :
    exact177843RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177843 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70483⟩⟩) exact177843RawTerms .large 177841 (.finite 3456353380086899479155517117627148481331252) (some (177842))

def event177844 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70484⟩⟩) 0 ⟨70483⟩ 177843

def event177845 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70484⟩⟩) 1 ⟨28387⟩ 175646

def event177846 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70484⟩⟩) (.sum [.predecessor 0 177844 .coefficient, .predecessor 1 177845 .coefficient])

def event177847 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70484⟩⟩) (.sum [.result 177843 .summary, .result 175646 .summary])

def exact177848RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177848RawTermsValid :
    exact177848RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177848 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70484⟩⟩) exact177848RawTerms .large 177846 (.finite 3802007596962448506045899439491360353157172) (some (177847))

def event177849 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70485⟩⟩) 0 ⟨70484⟩ 177848

def event177850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70485⟩⟩) 1 ⟨31067⟩ 175434

def event177851 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70485⟩⟩) (.sum [.predecessor 0 177849 .coefficient, .predecessor 1 177850 .coefficient])

def event177852 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70485⟩⟩) (.sum [.result 177848 .summary, .result 175434 .summary])

def exact177853RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177853RawTermsValid :
    exact177853RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177853 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70485⟩⟩) exact177853RawTerms .large 177851 (.finite 4147668141949793872257454032897973461975092) (some (177852))

def event177854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70486⟩⟩) 0 ⟨70485⟩ 177853

def event177855 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70486⟩⟩) 1 ⟨36727⟩ 175222

def event177856 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70486⟩⟩) (.sum [.predecessor 0 177854 .coefficient, .predecessor 1 177855 .coefficient])

def event177857 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70486⟩⟩) (.sum [.result 177853 .summary, .result 175222 .summary])

def exact177858RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177858RawTermsValid :
    exact177858RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177858 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70486⟩⟩) exact177858RawTerms .large 177856 (.finite 4493332905678336798016456807332854062121012) (some (177857))

def event177859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70487⟩⟩) 0 ⟨70486⟩ 177858

def event177860 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70487⟩⟩) 1 ⟨39407⟩ 175010

def event177861 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70487⟩⟩) (.sum [.predecessor 0 177859 .coefficient, .predecessor 1 177860 .coefficient])

def event177862 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70487⟩⟩) (.sum [.result 177858 .summary, .result 175010 .summary])

def exact177863RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177863RawTermsValid :
    exact177863RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177863 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70487⟩⟩) exact177863RawTerms .large 177861 (.finite 4838999778777478503549183672281868407930932) (some (177862))

def event177864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70488⟩⟩) 0 ⟨70487⟩ 177863

def event177865 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70488⟩⟩) 1 ⟨42087⟩ 174798

def event177866 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70488⟩⟩) (.sum [.predecessor 0 177864 .coefficient, .predecessor 1 177865 .coefficient])

def event177867 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70488⟩⟩) (.sum [.result 177863 .summary, .result 174798 .summary])

def exact177868RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177868RawTermsValid :
    exact177868RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177868 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70488⟩⟩) exact177868RawTerms .large 177866 (.finite 5184670870617817768629358718259150245068852) (some (177867))

def event177869 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70489⟩⟩) 0 ⟨70488⟩ 177868

def event177870 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70489⟩⟩) 1 ⟨44767⟩ 174586

def event177871 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70489⟩⟩) (.sum [.predecessor 0 177869 .coefficient, .predecessor 1 177870 .coefficient])

def event177872 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70489⟩⟩) (.sum [.result 177868 .summary, .result 174586 .summary])

def exact177873RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177873RawTermsValid :
    exact177873RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177873 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70489⟩⟩) exact177873RawTerms .large 177871 (.finite 5530348290569953373030706035778833319198772) (some (177872))

def event177874 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70490⟩⟩) 0 ⟨70489⟩ 177873

def event177875 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70490⟩⟩) 1 ⟨47447⟩ 174374

def event177876 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70490⟩⟩) (.sum [.predecessor 0 177874 .coefficient, .predecessor 1 177875 .coefficient])

def event177877 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70490⟩⟩) (.sum [.result 177873 .summary, .result 174374 .summary])

def exact177878RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177878RawTermsValid :
    exact177878RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177878 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70490⟩⟩) exact177878RawTerms .large 177876 (.finite 5876032038633885316753225624840917630320692) (some (177877))

def event177879 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70491⟩⟩) 0 ⟨70490⟩ 177878

def event177880 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70491⟩⟩) 1 ⟨50127⟩ 174162

def event177881 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70491⟩⟩) (.sum [.predecessor 0 177879 .coefficient, .predecessor 1 177880 .coefficient])

def event177882 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70491⟩⟩) (.sum [.result 177878 .summary, .result 174162 .summary])

def exact177883RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48411⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177883RawTermsValid :
    exact177883RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177883 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70491⟩⟩) exact177883RawTerms .large 177881 (.finite 6221717896068416040249469304417135687106612) (some (177882))

def event177884 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71371⟩⟩) 0 ⟨70491⟩ 177883

def event177885 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71371⟩⟩) 1 ⟨71369⟩ 173950

def event177886 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71371⟩⟩) (.sum [.predecessor 0 177884 .coefficient, .predecessor 1 177885 .coefficient])

def event177887 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71371⟩⟩) (.sum [.result 177883 .summary, .result 173950 .summary])

def exact177888RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54221⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67538⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48411⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177888RawTermsValid :
    exact177888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177888 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71371⟩⟩) exact177888RawTerms .large 177886 (.finite 66805187227601152574551644069558752530002096506798132) (some (177887))

def event177889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10⟩⟩) (.authority (.operator))

def exact177890RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨10⟩⟩]⟩, (1)⟩]

theorem exact177890RawTermsValid :
    exact177890RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177890 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10⟩⟩) exact177890RawTerms (.finite 26) 177889 .exactZero (none)

def event177891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7411⟩⟩) 0 ⟨2377⟩ 27

def event177892 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7411⟩⟩) 1 ⟨7254⟩ 16387

def event177893 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7411⟩⟩) (.product (.predecessor 0 177891 .coefficient) (.predecessor 1 177892 .coefficient) (⟨false, false, none, none, none⟩))

def event177894 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7411⟩⟩, .operator (⟨27, 0⟩, ⟨16387, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7254⟩⟩]⟩, (1)⟩)

def exact177895RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7254⟩⟩]⟩, (1)⟩]

theorem exact177895RawTermsValid :
    exact177895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7411⟩⟩) exact177895RawTerms .large 177893 .exactZero (none)

def event177896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9225⟩⟩) 0 ⟨7411⟩ 177895

def event177897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9225⟩⟩) 1 ⟨7010⟩ 163653

def event177898 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9225⟩⟩) (.sum [.predecessor 0 177896 .coefficient, .predecessor 1 177897 .coefficient])

def exact177899RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7254⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177899RawTermsValid :
    exact177899RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177899 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9225⟩⟩) exact177899RawTerms .large 177898 .exactZero (none)

def event177900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9226⟩⟩) 0 ⟨9225⟩ 177899

def event177901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9226⟩⟩) 1 ⟨10⟩ 177890

def event177902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9226⟩⟩) (.sum [.predecessor 0 177900 .coefficient, .predecessor 1 177901 .coefficient])

def event177903 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9226⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨10⟩⟩]⟩) [⟨.result 177890 .coefficient, false, none⟩])

def event177904 : Event := .survivorFold (1) 177903

def exact177905RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7254⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact177905RawTermsValid :
    exact177905RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event177905 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9226⟩⟩) exact177905RawTerms .large 177902 (.finite 26) (some (177903))

def event177906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9623⟩⟩) 0 ⟨9226⟩ 177905

def event177907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9623⟩⟩) 1 ⟨9584⟩ 15984

def event177908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9623⟩⟩) (.product (.predecessor 0 177906 .coefficient) (.predecessor 1 177907 .coefficient) (⟨false, false, none, none, none⟩))

def event177909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9623⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) [⟨.result 15980 .coefficient, false, none⟩])

def event177910 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9623⟩⟩) (.product (.result 177905 .summary) (.transfer 177909) (⟨false, false, none, none, none⟩))

def event177911 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9623⟩⟩, .operator (⟨177905, 1⟩, ⟨15984, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩, (-1)⟩)

def event177912 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨9623⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9583⟩⟩) ⟨9443⟩ 15977)

def event177913 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9623⟩⟩, .relation 177912 18, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (1)⟩)

def event177914 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9623⟩⟩, .relation 177912 17, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (-1)⟩)

def event177915 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9623⟩⟩, .relation 177912 16, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (-1)⟩)

def event177916 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9623⟩⟩, .relation 177912 15, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (-1)⟩)

def event177917 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9623⟩⟩, .relation 177912 14, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (-1)⟩)

def event177918 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9623⟩⟩, .relation 177912 13, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (-1)⟩)

def event177919 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9623⟩⟩, .relation 177912 12, ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (-1)⟩)

def eventLeaf11104 : Array AnnotatedEvent := #[
  { event := event177664
    frameStart := 177639 },
  { event := event177665
    frameStart := 177639 },
  { event := event177666
    frameStart := 177639 },
  { event := event177667
    frameStart := 177639 },
  { event := event177668
    frameStart := 177639 },
  { event := event177669
    frameStart := 177639 },
  { event := event177670
    frameStart := 177639 },
  { event := event177671
    frameStart := 177639 },
  { event := event177672
    frameStart := 177639 },
  { event := event177673
    frameStart := 177639 },
  { event := event177674
    frameStart := 177639 },
  { event := event177675
    frameStart := 177639 },
  { event := event177676
    frameStart := 177639 },
  { event := event177677
    frameStart := 177639 },
  { event := event177678
    frameStart := 177639 },
  { event := event177679
    frameStart := 177639 }
]

def eventLeaf11105 : Array AnnotatedEvent := #[
  { event := event177680
    frameStart := 177639 },
  { event := event177681
    frameStart := 177639 },
  { event := event177682
    frameStart := 177639 },
  { event := event177683
    frameStart := 177639 },
  { event := event177684
    frameStart := 177639 },
  { event := event177685
    frameStart := 177639 },
  { event := event177686
    frameStart := 177639 },
  { event := event177687
    frameStart := 177639 },
  { event := event177688
    frameStart := 177639 },
  { event := event177689
    frameStart := 177639 },
  { event := event177690
    frameStart := 177639 },
  { event := event177691
    frameStart := 177639 },
  { event := event177692
    frameStart := 177639 },
  { event := event177693
    frameStart := 177639 },
  { event := event177694
    frameStart := 177639 },
  { event := event177695
    frameStart := 177639 }
]

def eventLeaf11106 : Array AnnotatedEvent := #[
  { event := event177696
    frameStart := 177639 },
  { event := event177697
    frameStart := 177639 },
  { event := event177698
    frameStart := 177639 },
  { event := event177699
    frameStart := 177639 },
  { event := event177700
    frameStart := 177639 },
  { event := event177701
    frameStart := 177639 },
  { event := event177702
    frameStart := 177639 },
  { event := event177703
    frameStart := 177639 },
  { event := event177704
    frameStart := 177639 },
  { event := event177705
    frameStart := 177639 },
  { event := event177706
    frameStart := 177639 },
  { event := event177707
    frameStart := 177639 },
  { event := event177708
    frameStart := 177639 },
  { event := event177709
    frameStart := 177639 },
  { event := event177710
    frameStart := 177639 },
  { event := event177711
    frameStart := 177639 }
]

def eventLeaf11107 : Array AnnotatedEvent := #[
  { event := event177712
    frameStart := 177639 },
  { event := event177713
    frameStart := 177639 },
  { event := event177714
    frameStart := 177639 },
  { event := event177715
    frameStart := 177639 },
  { event := event177716
    frameStart := 177639 },
  { event := event177717
    frameStart := 177639 },
  { event := event177718
    frameStart := 177639 },
  { event := event177719
    frameStart := 177639 },
  { event := event177720
    frameStart := 177639 },
  { event := event177721
    frameStart := 177639 },
  { event := event177722
    frameStart := 177639 },
  { event := event177723
    frameStart := 177639 },
  { event := event177724
    frameStart := 177639 },
  { event := event177725
    frameStart := 177639 },
  { event := event177726
    frameStart := 177639 },
  { event := event177727
    frameStart := 177639 }
]

def eventLeaf11108 : Array AnnotatedEvent := #[
  { event := event177728
    frameStart := 177639 },
  { event := event177729
    frameStart := 177639 },
  { event := event177730
    frameStart := 177639 },
  { event := event177731
    frameStart := 177639 },
  { event := event177732
    frameStart := 177639 },
  { event := event177733
    frameStart := 177639 },
  { event := event177734
    frameStart := 177639 },
  { event := event177735
    frameStart := 177639 },
  { event := event177736
    frameStart := 177639 },
  { event := event177737
    frameStart := 177639 },
  { event := event177738
    frameStart := 177639 },
  { event := event177739
    frameStart := 177639 },
  { event := event177740
    frameStart := 177639 },
  { event := event177741
    frameStart := 177639 },
  { event := event177742
    frameStart := 177639 },
  { event := event177743
    frameStart := 0 }
]

def eventLeaf11109 : Array AnnotatedEvent := #[
  { event := event177744
    frameStart := 0 },
  { event := event177745
    frameStart := 0 },
  { event := event177746
    frameStart := 0 },
  { event := event177747
    frameStart := 0 },
  { event := event177748
    frameStart := 0 },
  { event := event177749
    frameStart := 0 },
  { event := event177750
    frameStart := 0 },
  { event := event177751
    frameStart := 0 },
  { event := event177752
    frameStart := 0 },
  { event := event177753
    frameStart := 0 },
  { event := event177754
    frameStart := 0 },
  { event := event177755
    frameStart := 0 },
  { event := event177756
    frameStart := 0 },
  { event := event177757
    frameStart := 0 },
  { event := event177758
    frameStart := 0 },
  { event := event177759
    frameStart := 0 }
]

def eventLeaf11110 : Array AnnotatedEvent := #[
  { event := event177760
    frameStart := 0 },
  { event := event177761
    frameStart := 0 },
  { event := event177762
    frameStart := 0 },
  { event := event177763
    frameStart := 0 },
  { event := event177764
    frameStart := 0 },
  { event := event177765
    frameStart := 0 },
  { event := event177766
    frameStart := 0 },
  { event := event177767
    frameStart := 0 },
  { event := event177768
    frameStart := 0 },
  { event := event177769
    frameStart := 0 },
  { event := event177770
    frameStart := 0 },
  { event := event177771
    frameStart := 0 },
  { event := event177772
    frameStart := 0 },
  { event := event177773
    frameStart := 0 },
  { event := event177774
    frameStart := 0 },
  { event := event177775
    frameStart := 0 }
]

def eventLeaf11111 : Array AnnotatedEvent := #[
  { event := event177776
    frameStart := 0 },
  { event := event177777
    frameStart := 0 },
  { event := event177778
    frameStart := 0 },
  { event := event177779
    frameStart := 0 },
  { event := event177780
    frameStart := 0 },
  { event := event177781
    frameStart := 0 },
  { event := event177782
    frameStart := 0 },
  { event := event177783
    frameStart := 0 },
  { event := event177784
    frameStart := 0 },
  { event := event177785
    frameStart := 0 },
  { event := event177786
    frameStart := 0 },
  { event := event177787
    frameStart := 0 },
  { event := event177788
    frameStart := 0 },
  { event := event177789
    frameStart := 0 },
  { event := event177790
    frameStart := 0 },
  { event := event177791
    frameStart := 0 }
]

def eventLeaf11112 : Array AnnotatedEvent := #[
  { event := event177792
    frameStart := 0 },
  { event := event177793
    frameStart := 0 },
  { event := event177794
    frameStart := 0 },
  { event := event177795
    frameStart := 0 },
  { event := event177796
    frameStart := 0 },
  { event := event177797
    frameStart := 0 },
  { event := event177798
    frameStart := 0 },
  { event := event177799
    frameStart := 0 },
  { event := event177800
    frameStart := 0 },
  { event := event177801
    frameStart := 0 },
  { event := event177802
    frameStart := 0 },
  { event := event177803
    frameStart := 0 },
  { event := event177804
    frameStart := 0 },
  { event := event177805
    frameStart := 0 },
  { event := event177806
    frameStart := 0 },
  { event := event177807
    frameStart := 0 }
]

def eventLeaf11113 : Array AnnotatedEvent := #[
  { event := event177808
    frameStart := 0 },
  { event := event177809
    frameStart := 0 },
  { event := event177810
    frameStart := 0 },
  { event := event177811
    frameStart := 0 },
  { event := event177812
    frameStart := 0 },
  { event := event177813
    frameStart := 0 },
  { event := event177814
    frameStart := 0 },
  { event := event177815
    frameStart := 0 },
  { event := event177816
    frameStart := 0 },
  { event := event177817
    frameStart := 0 },
  { event := event177818
    frameStart := 0 },
  { event := event177819
    frameStart := 0 },
  { event := event177820
    frameStart := 0 },
  { event := event177821
    frameStart := 0 },
  { event := event177822
    frameStart := 0 },
  { event := event177823
    frameStart := 0 }
]

def eventLeaf11114 : Array AnnotatedEvent := #[
  { event := event177824
    frameStart := 0 },
  { event := event177825
    frameStart := 0 },
  { event := event177826
    frameStart := 0 },
  { event := event177827
    frameStart := 0 },
  { event := event177828
    frameStart := 0 },
  { event := event177829
    frameStart := 0 },
  { event := event177830
    frameStart := 0 },
  { event := event177831
    frameStart := 0 },
  { event := event177832
    frameStart := 0 },
  { event := event177833
    frameStart := 0 },
  { event := event177834
    frameStart := 0 },
  { event := event177835
    frameStart := 0 },
  { event := event177836
    frameStart := 0 },
  { event := event177837
    frameStart := 0 },
  { event := event177838
    frameStart := 0 },
  { event := event177839
    frameStart := 0 }
]

def eventLeaf11115 : Array AnnotatedEvent := #[
  { event := event177840
    frameStart := 0 },
  { event := event177841
    frameStart := 0 },
  { event := event177842
    frameStart := 0 },
  { event := event177843
    frameStart := 0 },
  { event := event177844
    frameStart := 0 },
  { event := event177845
    frameStart := 0 },
  { event := event177846
    frameStart := 0 },
  { event := event177847
    frameStart := 0 },
  { event := event177848
    frameStart := 0 },
  { event := event177849
    frameStart := 0 },
  { event := event177850
    frameStart := 0 },
  { event := event177851
    frameStart := 0 },
  { event := event177852
    frameStart := 0 },
  { event := event177853
    frameStart := 0 },
  { event := event177854
    frameStart := 0 },
  { event := event177855
    frameStart := 0 }
]

def eventLeaf11116 : Array AnnotatedEvent := #[
  { event := event177856
    frameStart := 0 },
  { event := event177857
    frameStart := 0 },
  { event := event177858
    frameStart := 0 },
  { event := event177859
    frameStart := 0 },
  { event := event177860
    frameStart := 0 },
  { event := event177861
    frameStart := 0 },
  { event := event177862
    frameStart := 0 },
  { event := event177863
    frameStart := 0 },
  { event := event177864
    frameStart := 0 },
  { event := event177865
    frameStart := 0 },
  { event := event177866
    frameStart := 0 },
  { event := event177867
    frameStart := 0 },
  { event := event177868
    frameStart := 0 },
  { event := event177869
    frameStart := 0 },
  { event := event177870
    frameStart := 0 },
  { event := event177871
    frameStart := 0 }
]

def eventLeaf11117 : Array AnnotatedEvent := #[
  { event := event177872
    frameStart := 0 },
  { event := event177873
    frameStart := 0 },
  { event := event177874
    frameStart := 0 },
  { event := event177875
    frameStart := 0 },
  { event := event177876
    frameStart := 0 },
  { event := event177877
    frameStart := 0 },
  { event := event177878
    frameStart := 0 },
  { event := event177879
    frameStart := 0 },
  { event := event177880
    frameStart := 0 },
  { event := event177881
    frameStart := 0 },
  { event := event177882
    frameStart := 0 },
  { event := event177883
    frameStart := 0 },
  { event := event177884
    frameStart := 0 },
  { event := event177885
    frameStart := 0 },
  { event := event177886
    frameStart := 0 },
  { event := event177887
    frameStart := 0 }
]

def eventLeaf11118 : Array AnnotatedEvent := #[
  { event := event177888
    frameStart := 0 },
  { event := event177889
    frameStart := 0 },
  { event := event177890
    frameStart := 0 },
  { event := event177891
    frameStart := 0 },
  { event := event177892
    frameStart := 0 },
  { event := event177893
    frameStart := 0 },
  { event := event177894
    frameStart := 0 },
  { event := event177895
    frameStart := 0 },
  { event := event177896
    frameStart := 0 },
  { event := event177897
    frameStart := 0 },
  { event := event177898
    frameStart := 0 },
  { event := event177899
    frameStart := 0 },
  { event := event177900
    frameStart := 0 },
  { event := event177901
    frameStart := 0 },
  { event := event177902
    frameStart := 0 },
  { event := event177903
    frameStart := 0 }
]

def eventLeaf11119 : Array AnnotatedEvent := #[
  { event := event177904
    frameStart := 0 },
  { event := event177905
    frameStart := 0 },
  { event := event177906
    frameStart := 0 },
  { event := event177907
    frameStart := 0 },
  { event := event177908
    frameStart := 0 },
  { event := event177909
    frameStart := 0 },
  { event := event177910
    frameStart := 0 },
  { event := event177911
    frameStart := 0 },
  { event := event177912
    frameStart := 0 },
  { event := event177913
    frameStart := 0 },
  { event := event177914
    frameStart := 0 },
  { event := event177915
    frameStart := 0 },
  { event := event177916
    frameStart := 0 },
  { event := event177917
    frameStart := 0 },
  { event := event177918
    frameStart := 0 },
  { event := event177919
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events694
