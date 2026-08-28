import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events366

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event93696 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14954⟩⟩) 0 ⟨14953⟩ 93695

def event93697 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14954⟩⟩) (.identity (.predecessor 0 93696 .coefficient))

def event93698 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14954⟩⟩) (.finite 3)

def event93699 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23782⟩⟩) 0 ⟨14954⟩ 93698

def event93700 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23782⟩⟩) (.authority (.programFamilyFact))

def event93701 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23782⟩⟩) (.finite 3720)

def event93702 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event93703 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23783⟩⟩) 0 ⟨6689⟩ 93702

def event93704 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23783⟩⟩) 1 ⟨23782⟩ 93701

def event93705 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23783⟩⟩) (.authority (.operator))

def exact93706RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23783⟩⟩]⟩, (1)⟩]

theorem exact93706RawTermsValid :
    exact93706RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93706 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23783⟩⟩) exact93706RawTerms .large 93705 .exactZero (none)

def event93707 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26557⟩⟩) 0 ⟨23783⟩ 93706

def event93708 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26557⟩⟩) (.authority (.operator))

def exact93709RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26557⟩⟩]⟩, (1)⟩]

theorem exact93709RawTermsValid :
    exact93709RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93709 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26557⟩⟩) exact93709RawTerms (.finite 8192) 93708 .exactZero (none)

def event93710 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event93711 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event93712 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14993⟩⟩) 0 ⟨14954⟩ 93698

def event93713 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14993⟩⟩) 1 ⟨110⟩ 93711

def event93714 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14993⟩⟩) (.sum [.predecessor 0 93712 .coefficient, .predecessor 1 93713 .coefficient])

def event93715 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14993⟩⟩) (.finite 3)

def event93716 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14994⟩⟩) 0 ⟨14993⟩ 93715

def event93717 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14994⟩⟩) (.identity (.predecessor 0 93716 .coefficient))

def exact93718RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], []⟩, (1)⟩]

theorem exact93718RawTermsValid :
    exact93718RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93718 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14994⟩⟩) exact93718RawTerms (.finite 3) 93717 .exactZero (none)

def event93719 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact93720RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact93720RawTermsValid :
    exact93720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93720 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact93720RawTerms .large 93719 .exactZero (none)

def event93721 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14995⟩⟩) 0 ⟨6544⟩ 93720

def event93722 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14995⟩⟩) 1 ⟨14994⟩ 93718

def event93723 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14995⟩⟩) (.product (.predecessor 0 93721 .coefficient) (.predecessor 1 93722 .coefficient) (⟨false, false, none, none, none⟩))

def event93724 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14995⟩⟩, .operator (⟨93720, 0⟩, ⟨93718, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact93725RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact93725RawTermsValid :
    exact93725RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93725 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14995⟩⟩) exact93725RawTerms .large 93723 .exactZero (none)

def event93726 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6691⟩⟩) 0 ⟨6689⟩ 93702

def event93727 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6691⟩⟩) (.authority (.operator))

def exact93728RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩]

theorem exact93728RawTermsValid :
    exact93728RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93728 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6691⟩⟩) exact93728RawTerms .large 93727 .exactZero (none)

def event93729 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14996⟩⟩) 0 ⟨6691⟩ 93728

def event93730 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14996⟩⟩) 1 ⟨14995⟩ 93725

def event93731 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14996⟩⟩) (.sum [.predecessor 0 93729 .coefficient, .predecessor 1 93730 .coefficient])

def exact93732RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact93732RawTermsValid :
    exact93732RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93732 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14996⟩⟩) exact93732RawTerms .large 93731 .exactZero (none)

def event93733 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26558⟩⟩) 0 ⟨14996⟩ 93732

def event93734 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26558⟩⟩) 1 ⟨26557⟩ 93709

def event93735 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26558⟩⟩) (.product (.predecessor 0 93733 .coefficient) (.predecessor 1 93734 .coefficient) (⟨false, false, none, none, none⟩))

def event93736 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26558⟩⟩, .operator (⟨93732, 0⟩, ⟨93709, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26557⟩⟩]⟩, (1)⟩)

def event93737 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26558⟩⟩, .operator (⟨93732, 1⟩, ⟨93709, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26557⟩⟩]⟩, (-1)⟩)

def event93738 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26558⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26557⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26557⟩⟩) ⟨23783⟩ 93706)

def event93739 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26558⟩⟩, .relation 93738 0, ⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨23783⟩⟩]⟩, (-1)⟩)

def exact93740RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26557⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨23783⟩⟩]⟩, (-1)⟩]

theorem exact93740RawTermsValid :
    exact93740RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93740 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26558⟩⟩) exact93740RawTerms .large 93735 .exactZero (none)

def event93741 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15047⟩⟩) 0 ⟨14954⟩ 93698

def event93742 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15047⟩⟩) (.authority (.programFamilyFact))

def exact93743RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15047⟩⟩], []⟩, (1)⟩]

theorem exact93743RawTermsValid :
    exact93743RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93743 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15047⟩⟩) exact93743RawTerms (.finite 3) 93742 .exactZero (none)

def event93744 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15050⟩⟩) 0 ⟨6544⟩ 93720

def event93745 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15050⟩⟩) 1 ⟨15047⟩ 93743

def event93746 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15050⟩⟩) (.product (.predecessor 0 93744 .coefficient) (.predecessor 1 93745 .coefficient) (⟨false, true, none, none, some 1⟩))

def event93747 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15050⟩⟩, .operator (⟨93720, 0⟩, ⟨93743, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15047⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact93748RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15047⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact93748RawTermsValid :
    exact93748RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93748 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15050⟩⟩) exact93748RawTerms .large 93746 .exactZero (none)

def event93749 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6710⟩⟩) 0 ⟨6689⟩ 93702

def event93750 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6710⟩⟩) (.authority (.operator))

def exact93751RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩]

theorem exact93751RawTermsValid :
    exact93751RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93751 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6710⟩⟩) exact93751RawTerms .large 93750 .exactZero (none)

def event93752 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15051⟩⟩) 0 ⟨6710⟩ 93751

def event93753 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15051⟩⟩) 1 ⟨15050⟩ 93748

def event93754 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15051⟩⟩) (.sum [.predecessor 0 93752 .coefficient, .predecessor 1 93753 .coefficient])

def exact93755RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15047⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact93755RawTermsValid :
    exact93755RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93755 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15051⟩⟩) exact93755RawTerms .large 93754 .exactZero (none)

def event93756 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26563⟩⟩) 0 ⟨15051⟩ 93755

def event93757 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26563⟩⟩) 1 ⟨26558⟩ 93740

def event93758 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26563⟩⟩) (.sum [.predecessor 0 93756 .coefficient, .predecessor 1 93757 .coefficient])

def exact93759RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26557⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨23783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15047⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact93759RawTermsValid :
    exact93759RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93759 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26563⟩⟩) exact93759RawTerms .large 93758 .exactZero (none)

def event93760 : Event := .preFoldPolynomial 93759 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26557⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨23783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15047⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact93761RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26557⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨23783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15047⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event93761 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26563⟩⟩) 93760 exact93761RawTerms .large 93758 .exactZero (none)

def event93762 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14954⟩⟩) ⟨⟨123⟩, ⟨29⟩, ⟨109⟩⟩ ⟨93604, 93762⟩

def event93763 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20467⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20464⟩⟩]⟩) (1) 0 2 (.universal 93762 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20464⟩⟩]⟩) (none) 93761)

def event93764 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20467⟩⟩, .relation 93763 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩)

def event93765 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20467⟩⟩, .relation 93763 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26557⟩⟩]⟩, (-1)⟩)

def event93766 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20467⟩⟩, .relation 93763 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨23783⟩⟩]⟩, (1)⟩)

def event93767 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20467⟩⟩, .relation 93763 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15047⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact93768RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26557⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨23783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15047⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact93768RawTermsValid :
    exact93768RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93768 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20467⟩⟩) exact93768RawTerms .large 93600 (.finite 1811303510016) (some (93602))

def event93769 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26560⟩⟩) 0 ⟨20467⟩ 93768

def event93770 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26560⟩⟩) 1 ⟨26559⟩ 93590

def event93771 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26560⟩⟩) (.sum [.predecessor 0 93769 .coefficient, .predecessor 1 93770 .coefficient])

def event93772 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26560⟩⟩, .operator (⟨93768, 0⟩, ⟨93590, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26557⟩⟩]⟩, (1)⟩)

def event93773 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26560⟩⟩, .operator (⟨93768, 2⟩, ⟨93590, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨23783⟩⟩]⟩, (-1)⟩)

def event93774 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26560⟩⟩) (.sum [.result 93768 .summary, .result 93590 .summary])

def exact93775RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15047⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact93775RawTermsValid :
    exact93775RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93775 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26560⟩⟩) exact93775RawTerms .large 93771 (.finite 1291900380601931935744) (some (93774))

def event93776 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26561⟩⟩) 0 ⟨26560⟩ 93775

def event93777 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26561⟩⟩) 1 ⟨6672⟩ 5839

def event93778 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26561⟩⟩) (.product (.predecessor 0 93776 .coefficient) (.predecessor 1 93777 .coefficient) (⟨false, false, none, none, none⟩))

def event93779 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26561⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩) [⟨.result 5835 .coefficient, false, none⟩])

def event93780 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26561⟩⟩) (.product (.result 93775 .summary) (.transfer 93779) (⟨false, false, none, none, none⟩))

def event93781 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26561⟩⟩, .operator (⟨93775, 0⟩, ⟨5839, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩)

def event93782 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26561⟩⟩, .operator (⟨93775, 1⟩, ⟨5839, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15047⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (-1)⟩)

def event93783 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26561⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15047⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6671⟩⟩) ⟨6607⟩ 5832)

def event93784 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26561⟩⟩, .relation 93783 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15047⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact93785RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15047⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact93785RawTermsValid :
    exact93785RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93785 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26561⟩⟩) exact93785RawTerms .large 93778 (.finite 4741295067215179835091451904) (some (93780))

def event93786 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23720⟩⟩) 0 ⟨6689⟩ 5477

def event93787 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23720⟩⟩) 1 ⟨23719⟩ 88074

def event93788 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23720⟩⟩) (.authority (.operator))

def exact93789RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23720⟩⟩]⟩, (1)⟩]

theorem exact93789RawTermsValid :
    exact93789RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93789 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23720⟩⟩) exact93789RawTerms .large 93788 .exactZero (none)

def event93790 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26351⟩⟩) 0 ⟨23720⟩ 93789

def event93791 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26351⟩⟩) (.authority (.operator))

def exact93792RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26351⟩⟩]⟩, (1)⟩]

theorem exact93792RawTermsValid :
    exact93792RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93792 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26351⟩⟩) exact93792RawTerms (.finite 8192) 93791 .exactZero (none)

def event93793 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26353⟩⟩) 0 ⟨24913⟩ 88356

def event93794 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26353⟩⟩) 1 ⟨26351⟩ 93792

def event93795 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26353⟩⟩) (.product (.predecessor 0 93793 .coefficient) (.predecessor 1 93794 .coefficient) (⟨false, false, none, none, none⟩))

def event93796 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26353⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26351⟩⟩]⟩) [⟨.result 93792 .coefficient, false, none⟩])

def event93797 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26353⟩⟩) (.product (.result 88356 .summary) (.transfer 93796) (⟨false, false, none, none, none⟩))

def event93798 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26353⟩⟩, .operator (⟨88356, 0⟩, ⟨93792, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26351⟩⟩]⟩, (1)⟩)

def event93799 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26353⟩⟩, .operator (⟨88356, 1⟩, ⟨93792, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26351⟩⟩]⟩, (-1)⟩)

def event93800 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26353⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26351⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26351⟩⟩) ⟨23720⟩ 93789)

def event93801 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26353⟩⟩, .relation 93800 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨23720⟩⟩]⟩, (-1)⟩)

def exact93802RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26351⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨23720⟩⟩]⟩, (-1)⟩]

theorem exact93802RawTermsValid :
    exact93802RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93802 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26353⟩⟩) exact93802RawTerms .large 93795 (.finite 1291889172568118132736) (some (93797))

def event93803 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20320⟩⟩) 0 ⟨14793⟩ 4236

def event93804 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20320⟩⟩) (.authority (.relationPreimageSource ⟨27⟩))

def exact93805RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20320⟩⟩]⟩, (1)⟩]

theorem exact93805RawTermsValid :
    exact93805RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93805 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20320⟩⟩) exact93805RawTerms (.finite 136065468) 93804 .exactZero (none)

def event93806 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20322⟩⟩) 0 ⟨20320⟩ 93805

def event93807 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20322⟩⟩) 1 ⟨2348⟩ 4

def event93808 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20322⟩⟩) (.scale (.predecessor 0 93806 .coefficient) (.value (.predecessor 1 93807 .coefficient)))

def exact93809RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20320⟩⟩]⟩, (1)⟩]

theorem exact93809RawTermsValid :
    exact93809RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93809 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20322⟩⟩) exact93809RawTerms (.finite 136065468) 93808 .exactZero (none)

def event93810 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20323⟩⟩) 0 ⟨5541⟩ 80012

def event93811 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20323⟩⟩) 1 ⟨20322⟩ 93809

def event93812 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20323⟩⟩) (.product (.predecessor 0 93810 .coefficient) (.predecessor 1 93811 .coefficient) (⟨false, false, none, none, none⟩))

def event93813 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20323⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20320⟩⟩]⟩) [⟨.result 93805 .coefficient, false, none⟩])

def event93814 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20323⟩⟩) (.product (.result 80012 .summary) (.transfer 93813) (⟨false, false, none, none, none⟩))

def event93815 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20323⟩⟩, .operator (⟨80012, 0⟩, ⟨93809, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20320⟩⟩]⟩, (1)⟩)

def event93816 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20321⟩⟩)

def event93817 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event93818 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event93819 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event93820 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event93821 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event93822 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event93823 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event93824 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event93825 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 93824

def event93826 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 93822

def event93827 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 93825 .coefficient) (.value (.predecessor 1 93826 .coefficient)))

def event93828 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event93829 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 93828

def event93830 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 93820

def event93831 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 93829 .coefficient, .predecessor 1 93830 .coefficient])

def event93832 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event93833 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 93832

def event93834 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 93818

def event93835 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 93834 .coefficient))

def event93836 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event93837 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10480⟩⟩) 0 ⟨5536⟩ 93836

def event93838 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10480⟩⟩) (.authority (.programFamilyFact))

def exact93839RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10480⟩⟩], []⟩, (1)⟩]

theorem exact93839RawTermsValid :
    exact93839RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93839 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10480⟩⟩) exact93839RawTerms (.finite 2) 93838 .exactZero (none)

def event93840 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9400⟩⟩) 0 ⟨5536⟩ 93836

def event93841 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9400⟩⟩) (.authority (.programFamilyFact))

def exact93842RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩], []⟩, (1)⟩]

theorem exact93842RawTermsValid :
    exact93842RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93842 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9400⟩⟩) exact93842RawTerms (.finite 2) 93841 .exactZero (none)

def event93843 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10481⟩⟩) 0 ⟨9400⟩ 93842

def event93844 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10481⟩⟩) 1 ⟨10480⟩ 93839

def event93845 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10481⟩⟩) (.product (.predecessor 0 93843 .coefficient) (.predecessor 1 93844 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event93846 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10481⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], []⟩) [⟨.result 93842 .coefficient, true, some 1⟩, ⟨.result 93839 .coefficient, true, some 1⟩])

def event93847 : Event := .survivorFold (1) 93846

def exact93848RawTerms : List Term := []

theorem exact93848RawTermsValid :
    exact93848RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93848 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10481⟩⟩) exact93848RawTerms (.finite 4) 93845 (.finite 4) (some (93846))

def event93849 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10482⟩⟩) 0 ⟨10481⟩ 93848

def event93850 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10482⟩⟩) (.identity (.predecessor 0 93849 .coefficient))

def event93851 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10482⟩⟩) (.finite 4)

def event93852 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14792⟩⟩) 0 ⟨10482⟩ 93851

def event93853 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14792⟩⟩) (.authority (.programFamilyFact))

def exact93854RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], []⟩, (1)⟩]

theorem exact93854RawTermsValid :
    exact93854RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93854 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14792⟩⟩) exact93854RawTerms (.finite 2) 93853 .exactZero (none)

def event93855 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14793⟩⟩) 0 ⟨14792⟩ 93854

def event93856 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14793⟩⟩) (.identity (.predecessor 0 93855 .coefficient))

def event93857 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14793⟩⟩) (.finite 2)

def event93858 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20320⟩⟩) 0 ⟨14793⟩ 93857

def event93859 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20320⟩⟩) (.authority (.relationPreimageSource ⟨27⟩))

def exact93860RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20320⟩⟩]⟩, (1)⟩]

theorem exact93860RawTermsValid :
    exact93860RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93860 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20320⟩⟩) exact93860RawTerms (.finite 136065468) 93859 .exactZero (none)

def event93861 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact93862RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact93862RawTermsValid :
    exact93862RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93862 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact93862RawTerms .large 93861 .exactZero (none)

def event93863 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20321⟩⟩) 0 ⟨6⟩ 93862

def event93864 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20321⟩⟩) 1 ⟨20320⟩ 93860

def event93865 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20321⟩⟩) (.product (.predecessor 0 93863 .coefficient) (.predecessor 1 93864 .coefficient) (⟨false, false, none, none, none⟩))

def event93866 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20321⟩⟩, .operator (⟨93862, 0⟩, ⟨93860, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20320⟩⟩]⟩, (1)⟩)

def exact93867RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20320⟩⟩]⟩, (1)⟩]

theorem exact93867RawTermsValid :
    exact93867RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93867 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20321⟩⟩) exact93867RawTerms .large 93865 .exactZero (none)

def event93868 : Event := .preFoldPolynomial 93867 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20320⟩⟩]⟩, (1)⟩] .exactZero none

def exact93869RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20320⟩⟩]⟩, (1)⟩]

def event93869 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20321⟩⟩) 93868 exact93869RawTerms .large 93865 .exactZero (none)

def event93870 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26357⟩⟩)

def event93871 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event93872 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event93873 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event93874 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event93875 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event93876 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event93877 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event93878 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event93879 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 93878

def event93880 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 93876

def event93881 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 93879 .coefficient) (.value (.predecessor 1 93880 .coefficient)))

def event93882 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event93883 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 93882

def event93884 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 93874

def event93885 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 93883 .coefficient, .predecessor 1 93884 .coefficient])

def event93886 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event93887 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 93886

def event93888 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 93872

def event93889 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 93888 .coefficient))

def event93890 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event93891 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10480⟩⟩) 0 ⟨5536⟩ 93890

def event93892 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10480⟩⟩) (.authority (.programFamilyFact))

def exact93893RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10480⟩⟩], []⟩, (1)⟩]

theorem exact93893RawTermsValid :
    exact93893RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93893 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10480⟩⟩) exact93893RawTerms (.finite 2) 93892 .exactZero (none)

def event93894 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9400⟩⟩) 0 ⟨5536⟩ 93890

def event93895 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9400⟩⟩) (.authority (.programFamilyFact))

def exact93896RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩], []⟩, (1)⟩]

theorem exact93896RawTermsValid :
    exact93896RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93896 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9400⟩⟩) exact93896RawTerms (.finite 2) 93895 .exactZero (none)

def event93897 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10481⟩⟩) 0 ⟨9400⟩ 93896

def event93898 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10481⟩⟩) 1 ⟨10480⟩ 93893

def event93899 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10481⟩⟩) (.product (.predecessor 0 93897 .coefficient) (.predecessor 1 93898 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event93900 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10481⟩⟩, .operator (⟨93896, 0⟩, ⟨93893, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], []⟩, (1)⟩)

def exact93901RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], []⟩, (1)⟩]

theorem exact93901RawTermsValid :
    exact93901RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93901 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10481⟩⟩) exact93901RawTerms (.finite 4) 93899 .exactZero (none)

def event93902 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10482⟩⟩) 0 ⟨10481⟩ 93901

def event93903 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10482⟩⟩) (.identity (.predecessor 0 93902 .coefficient))

def event93904 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10482⟩⟩) (.finite 4)

def event93905 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14792⟩⟩) 0 ⟨10482⟩ 93904

def event93906 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14792⟩⟩) (.authority (.programFamilyFact))

def exact93907RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], []⟩, (1)⟩]

theorem exact93907RawTermsValid :
    exact93907RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93907 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14792⟩⟩) exact93907RawTerms (.finite 2) 93906 .exactZero (none)

def event93908 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14793⟩⟩) 0 ⟨14792⟩ 93907

def event93909 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14793⟩⟩) (.identity (.predecessor 0 93908 .coefficient))

def event93910 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14793⟩⟩) (.finite 2)

def event93911 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23719⟩⟩) 0 ⟨14793⟩ 93910

def event93912 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23719⟩⟩) (.authority (.programFamilyFact))

def event93913 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23719⟩⟩) (.finite 3720)

def event93914 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event93915 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23720⟩⟩) 0 ⟨6689⟩ 93914

def event93916 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23720⟩⟩) 1 ⟨23719⟩ 93913

def event93917 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23720⟩⟩) (.authority (.operator))

def exact93918RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23720⟩⟩]⟩, (1)⟩]

theorem exact93918RawTermsValid :
    exact93918RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93918 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23720⟩⟩) exact93918RawTerms .large 93917 .exactZero (none)

def event93919 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26351⟩⟩) 0 ⟨23720⟩ 93918

def event93920 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26351⟩⟩) (.authority (.operator))

def exact93921RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26351⟩⟩]⟩, (1)⟩]

theorem exact93921RawTermsValid :
    exact93921RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93921 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26351⟩⟩) exact93921RawTerms (.finite 8192) 93920 .exactZero (none)

def event93922 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event93923 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event93924 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14832⟩⟩) 0 ⟨14793⟩ 93910

def event93925 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14832⟩⟩) 1 ⟨110⟩ 93923

def event93926 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14832⟩⟩) (.sum [.predecessor 0 93924 .coefficient, .predecessor 1 93925 .coefficient])

def event93927 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14832⟩⟩) (.finite 2)

def event93928 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14833⟩⟩) 0 ⟨14832⟩ 93927

def event93929 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14833⟩⟩) (.identity (.predecessor 0 93928 .coefficient))

def exact93930RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], []⟩, (1)⟩]

theorem exact93930RawTermsValid :
    exact93930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93930 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14833⟩⟩) exact93930RawTerms (.finite 2) 93929 .exactZero (none)

def event93931 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact93932RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact93932RawTermsValid :
    exact93932RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93932 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact93932RawTerms .large 93931 .exactZero (none)

def event93933 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14834⟩⟩) 0 ⟨6544⟩ 93932

def event93934 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14834⟩⟩) 1 ⟨14833⟩ 93930

def event93935 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14834⟩⟩) (.product (.predecessor 0 93933 .coefficient) (.predecessor 1 93934 .coefficient) (⟨false, false, none, none, none⟩))

def event93936 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14834⟩⟩, .operator (⟨93932, 0⟩, ⟨93930, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact93937RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact93937RawTermsValid :
    exact93937RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93937 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14834⟩⟩) exact93937RawTerms .large 93935 .exactZero (none)

def event93938 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6690⟩⟩) 0 ⟨6689⟩ 93914

def event93939 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6690⟩⟩) (.authority (.operator))

def exact93940RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩]

theorem exact93940RawTermsValid :
    exact93940RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93940 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6690⟩⟩) exact93940RawTerms .large 93939 .exactZero (none)

def event93941 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14835⟩⟩) 0 ⟨6690⟩ 93940

def event93942 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14835⟩⟩) 1 ⟨14834⟩ 93937

def event93943 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14835⟩⟩) (.sum [.predecessor 0 93941 .coefficient, .predecessor 1 93942 .coefficient])

def exact93944RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact93944RawTermsValid :
    exact93944RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event93944 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14835⟩⟩) exact93944RawTerms .large 93943 .exactZero (none)

def event93945 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26352⟩⟩) 0 ⟨14835⟩ 93944

def event93946 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26352⟩⟩) 1 ⟨26351⟩ 93921

def event93947 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26352⟩⟩) (.product (.predecessor 0 93945 .coefficient) (.predecessor 1 93946 .coefficient) (⟨false, false, none, none, none⟩))

def event93948 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26352⟩⟩, .operator (⟨93944, 0⟩, ⟨93921, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26351⟩⟩]⟩, (1)⟩)

def event93949 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26352⟩⟩, .operator (⟨93944, 1⟩, ⟨93921, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26351⟩⟩]⟩, (-1)⟩)

def event93950 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26352⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26351⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26351⟩⟩) ⟨23720⟩ 93918)

def event93951 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26352⟩⟩, .relation 93950 0, ⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨23720⟩⟩]⟩, (-1)⟩)

def eventLeaf5856 : Array AnnotatedEvent := #[
  { event := event93696
    frameStart := 93658 },
  { event := event93697
    frameStart := 93658 },
  { event := event93698
    frameStart := 93658 },
  { event := event93699
    frameStart := 93658 },
  { event := event93700
    frameStart := 93658 },
  { event := event93701
    frameStart := 93658 },
  { event := event93702
    frameStart := 93658 },
  { event := event93703
    frameStart := 93658 },
  { event := event93704
    frameStart := 93658 },
  { event := event93705
    frameStart := 93658 },
  { event := event93706
    frameStart := 93658 },
  { event := event93707
    frameStart := 93658 },
  { event := event93708
    frameStart := 93658 },
  { event := event93709
    frameStart := 93658 },
  { event := event93710
    frameStart := 93658 },
  { event := event93711
    frameStart := 93658 }
]

def eventLeaf5857 : Array AnnotatedEvent := #[
  { event := event93712
    frameStart := 93658 },
  { event := event93713
    frameStart := 93658 },
  { event := event93714
    frameStart := 93658 },
  { event := event93715
    frameStart := 93658 },
  { event := event93716
    frameStart := 93658 },
  { event := event93717
    frameStart := 93658 },
  { event := event93718
    frameStart := 93658 },
  { event := event93719
    frameStart := 93658 },
  { event := event93720
    frameStart := 93658 },
  { event := event93721
    frameStart := 93658 },
  { event := event93722
    frameStart := 93658 },
  { event := event93723
    frameStart := 93658 },
  { event := event93724
    frameStart := 93658 },
  { event := event93725
    frameStart := 93658 },
  { event := event93726
    frameStart := 93658 },
  { event := event93727
    frameStart := 93658 }
]

def eventLeaf5858 : Array AnnotatedEvent := #[
  { event := event93728
    frameStart := 93658 },
  { event := event93729
    frameStart := 93658 },
  { event := event93730
    frameStart := 93658 },
  { event := event93731
    frameStart := 93658 },
  { event := event93732
    frameStart := 93658 },
  { event := event93733
    frameStart := 93658 },
  { event := event93734
    frameStart := 93658 },
  { event := event93735
    frameStart := 93658 },
  { event := event93736
    frameStart := 93658 },
  { event := event93737
    frameStart := 93658 },
  { event := event93738
    frameStart := 93658 },
  { event := event93739
    frameStart := 93658 },
  { event := event93740
    frameStart := 93658 },
  { event := event93741
    frameStart := 93658 },
  { event := event93742
    frameStart := 93658 },
  { event := event93743
    frameStart := 93658 }
]

def eventLeaf5859 : Array AnnotatedEvent := #[
  { event := event93744
    frameStart := 93658 },
  { event := event93745
    frameStart := 93658 },
  { event := event93746
    frameStart := 93658 },
  { event := event93747
    frameStart := 93658 },
  { event := event93748
    frameStart := 93658 },
  { event := event93749
    frameStart := 93658 },
  { event := event93750
    frameStart := 93658 },
  { event := event93751
    frameStart := 93658 },
  { event := event93752
    frameStart := 93658 },
  { event := event93753
    frameStart := 93658 },
  { event := event93754
    frameStart := 93658 },
  { event := event93755
    frameStart := 93658 },
  { event := event93756
    frameStart := 93658 },
  { event := event93757
    frameStart := 93658 },
  { event := event93758
    frameStart := 93658 },
  { event := event93759
    frameStart := 93658 }
]

def eventLeaf5860 : Array AnnotatedEvent := #[
  { event := event93760
    frameStart := 93658 },
  { event := event93761
    frameStart := 93658 },
  { event := event93762
    frameStart := 0 },
  { event := event93763
    frameStart := 0 },
  { event := event93764
    frameStart := 0 },
  { event := event93765
    frameStart := 0 },
  { event := event93766
    frameStart := 0 },
  { event := event93767
    frameStart := 0 },
  { event := event93768
    frameStart := 0 },
  { event := event93769
    frameStart := 0 },
  { event := event93770
    frameStart := 0 },
  { event := event93771
    frameStart := 0 },
  { event := event93772
    frameStart := 0 },
  { event := event93773
    frameStart := 0 },
  { event := event93774
    frameStart := 0 },
  { event := event93775
    frameStart := 0 }
]

def eventLeaf5861 : Array AnnotatedEvent := #[
  { event := event93776
    frameStart := 0 },
  { event := event93777
    frameStart := 0 },
  { event := event93778
    frameStart := 0 },
  { event := event93779
    frameStart := 0 },
  { event := event93780
    frameStart := 0 },
  { event := event93781
    frameStart := 0 },
  { event := event93782
    frameStart := 0 },
  { event := event93783
    frameStart := 0 },
  { event := event93784
    frameStart := 0 },
  { event := event93785
    frameStart := 0 },
  { event := event93786
    frameStart := 0 },
  { event := event93787
    frameStart := 0 },
  { event := event93788
    frameStart := 0 },
  { event := event93789
    frameStart := 0 },
  { event := event93790
    frameStart := 0 },
  { event := event93791
    frameStart := 0 }
]

def eventLeaf5862 : Array AnnotatedEvent := #[
  { event := event93792
    frameStart := 0 },
  { event := event93793
    frameStart := 0 },
  { event := event93794
    frameStart := 0 },
  { event := event93795
    frameStart := 0 },
  { event := event93796
    frameStart := 0 },
  { event := event93797
    frameStart := 0 },
  { event := event93798
    frameStart := 0 },
  { event := event93799
    frameStart := 0 },
  { event := event93800
    frameStart := 0 },
  { event := event93801
    frameStart := 0 },
  { event := event93802
    frameStart := 0 },
  { event := event93803
    frameStart := 0 },
  { event := event93804
    frameStart := 0 },
  { event := event93805
    frameStart := 0 },
  { event := event93806
    frameStart := 0 },
  { event := event93807
    frameStart := 0 }
]

def eventLeaf5863 : Array AnnotatedEvent := #[
  { event := event93808
    frameStart := 0 },
  { event := event93809
    frameStart := 0 },
  { event := event93810
    frameStart := 0 },
  { event := event93811
    frameStart := 0 },
  { event := event93812
    frameStart := 0 },
  { event := event93813
    frameStart := 0 },
  { event := event93814
    frameStart := 0 },
  { event := event93815
    frameStart := 0 },
  { event := event93816
    frameStart := 93816 },
  { event := event93817
    frameStart := 93816 },
  { event := event93818
    frameStart := 93816 },
  { event := event93819
    frameStart := 93816 },
  { event := event93820
    frameStart := 93816 },
  { event := event93821
    frameStart := 93816 },
  { event := event93822
    frameStart := 93816 },
  { event := event93823
    frameStart := 93816 }
]

def eventLeaf5864 : Array AnnotatedEvent := #[
  { event := event93824
    frameStart := 93816 },
  { event := event93825
    frameStart := 93816 },
  { event := event93826
    frameStart := 93816 },
  { event := event93827
    frameStart := 93816 },
  { event := event93828
    frameStart := 93816 },
  { event := event93829
    frameStart := 93816 },
  { event := event93830
    frameStart := 93816 },
  { event := event93831
    frameStart := 93816 },
  { event := event93832
    frameStart := 93816 },
  { event := event93833
    frameStart := 93816 },
  { event := event93834
    frameStart := 93816 },
  { event := event93835
    frameStart := 93816 },
  { event := event93836
    frameStart := 93816 },
  { event := event93837
    frameStart := 93816 },
  { event := event93838
    frameStart := 93816 },
  { event := event93839
    frameStart := 93816 }
]

def eventLeaf5865 : Array AnnotatedEvent := #[
  { event := event93840
    frameStart := 93816 },
  { event := event93841
    frameStart := 93816 },
  { event := event93842
    frameStart := 93816 },
  { event := event93843
    frameStart := 93816 },
  { event := event93844
    frameStart := 93816 },
  { event := event93845
    frameStart := 93816 },
  { event := event93846
    frameStart := 93816 },
  { event := event93847
    frameStart := 93816 },
  { event := event93848
    frameStart := 93816 },
  { event := event93849
    frameStart := 93816 },
  { event := event93850
    frameStart := 93816 },
  { event := event93851
    frameStart := 93816 },
  { event := event93852
    frameStart := 93816 },
  { event := event93853
    frameStart := 93816 },
  { event := event93854
    frameStart := 93816 },
  { event := event93855
    frameStart := 93816 }
]

def eventLeaf5866 : Array AnnotatedEvent := #[
  { event := event93856
    frameStart := 93816 },
  { event := event93857
    frameStart := 93816 },
  { event := event93858
    frameStart := 93816 },
  { event := event93859
    frameStart := 93816 },
  { event := event93860
    frameStart := 93816 },
  { event := event93861
    frameStart := 93816 },
  { event := event93862
    frameStart := 93816 },
  { event := event93863
    frameStart := 93816 },
  { event := event93864
    frameStart := 93816 },
  { event := event93865
    frameStart := 93816 },
  { event := event93866
    frameStart := 93816 },
  { event := event93867
    frameStart := 93816 },
  { event := event93868
    frameStart := 93816 },
  { event := event93869
    frameStart := 93816 },
  { event := event93870
    frameStart := 93870 },
  { event := event93871
    frameStart := 93870 }
]

def eventLeaf5867 : Array AnnotatedEvent := #[
  { event := event93872
    frameStart := 93870 },
  { event := event93873
    frameStart := 93870 },
  { event := event93874
    frameStart := 93870 },
  { event := event93875
    frameStart := 93870 },
  { event := event93876
    frameStart := 93870 },
  { event := event93877
    frameStart := 93870 },
  { event := event93878
    frameStart := 93870 },
  { event := event93879
    frameStart := 93870 },
  { event := event93880
    frameStart := 93870 },
  { event := event93881
    frameStart := 93870 },
  { event := event93882
    frameStart := 93870 },
  { event := event93883
    frameStart := 93870 },
  { event := event93884
    frameStart := 93870 },
  { event := event93885
    frameStart := 93870 },
  { event := event93886
    frameStart := 93870 },
  { event := event93887
    frameStart := 93870 }
]

def eventLeaf5868 : Array AnnotatedEvent := #[
  { event := event93888
    frameStart := 93870 },
  { event := event93889
    frameStart := 93870 },
  { event := event93890
    frameStart := 93870 },
  { event := event93891
    frameStart := 93870 },
  { event := event93892
    frameStart := 93870 },
  { event := event93893
    frameStart := 93870 },
  { event := event93894
    frameStart := 93870 },
  { event := event93895
    frameStart := 93870 },
  { event := event93896
    frameStart := 93870 },
  { event := event93897
    frameStart := 93870 },
  { event := event93898
    frameStart := 93870 },
  { event := event93899
    frameStart := 93870 },
  { event := event93900
    frameStart := 93870 },
  { event := event93901
    frameStart := 93870 },
  { event := event93902
    frameStart := 93870 },
  { event := event93903
    frameStart := 93870 }
]

def eventLeaf5869 : Array AnnotatedEvent := #[
  { event := event93904
    frameStart := 93870 },
  { event := event93905
    frameStart := 93870 },
  { event := event93906
    frameStart := 93870 },
  { event := event93907
    frameStart := 93870 },
  { event := event93908
    frameStart := 93870 },
  { event := event93909
    frameStart := 93870 },
  { event := event93910
    frameStart := 93870 },
  { event := event93911
    frameStart := 93870 },
  { event := event93912
    frameStart := 93870 },
  { event := event93913
    frameStart := 93870 },
  { event := event93914
    frameStart := 93870 },
  { event := event93915
    frameStart := 93870 },
  { event := event93916
    frameStart := 93870 },
  { event := event93917
    frameStart := 93870 },
  { event := event93918
    frameStart := 93870 },
  { event := event93919
    frameStart := 93870 }
]

def eventLeaf5870 : Array AnnotatedEvent := #[
  { event := event93920
    frameStart := 93870 },
  { event := event93921
    frameStart := 93870 },
  { event := event93922
    frameStart := 93870 },
  { event := event93923
    frameStart := 93870 },
  { event := event93924
    frameStart := 93870 },
  { event := event93925
    frameStart := 93870 },
  { event := event93926
    frameStart := 93870 },
  { event := event93927
    frameStart := 93870 },
  { event := event93928
    frameStart := 93870 },
  { event := event93929
    frameStart := 93870 },
  { event := event93930
    frameStart := 93870 },
  { event := event93931
    frameStart := 93870 },
  { event := event93932
    frameStart := 93870 },
  { event := event93933
    frameStart := 93870 },
  { event := event93934
    frameStart := 93870 },
  { event := event93935
    frameStart := 93870 }
]

def eventLeaf5871 : Array AnnotatedEvent := #[
  { event := event93936
    frameStart := 93870 },
  { event := event93937
    frameStart := 93870 },
  { event := event93938
    frameStart := 93870 },
  { event := event93939
    frameStart := 93870 },
  { event := event93940
    frameStart := 93870 },
  { event := event93941
    frameStart := 93870 },
  { event := event93942
    frameStart := 93870 },
  { event := event93943
    frameStart := 93870 },
  { event := event93944
    frameStart := 93870 },
  { event := event93945
    frameStart := 93870 },
  { event := event93946
    frameStart := 93870 },
  { event := event93947
    frameStart := 93870 },
  { event := event93948
    frameStart := 93870 },
  { event := event93949
    frameStart := 93870 },
  { event := event93950
    frameStart := 93870 },
  { event := event93951
    frameStart := 93870 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events366
