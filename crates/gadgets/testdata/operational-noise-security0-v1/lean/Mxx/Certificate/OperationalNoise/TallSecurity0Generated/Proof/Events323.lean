import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events323

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event82688 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 82678

def event82689 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 82687 .coefficient, .predecessor 1 82688 .coefficient])

def event82690 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event82691 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 82690

def event82692 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 82676

def event82693 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 82692 .coefficient))

def event82694 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event82695 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12370⟩⟩) 0 ⟨5536⟩ 82694

def event82696 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12370⟩⟩) (.authority (.programFamilyFact))

def exact82697RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12370⟩⟩], []⟩, (1)⟩]

theorem exact82697RawTermsValid :
    exact82697RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82697 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12370⟩⟩) exact82697RawTerms (.finite 40) 82696 .exactZero (none)

def event82698 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9820⟩⟩) 0 ⟨5536⟩ 82694

def event82699 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9820⟩⟩) (.authority (.programFamilyFact))

def exact82700RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9820⟩⟩], []⟩, (1)⟩]

theorem exact82700RawTermsValid :
    exact82700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82700 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9820⟩⟩) exact82700RawTerms (.finite 40) 82699 .exactZero (none)

def event82701 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12371⟩⟩) 0 ⟨9820⟩ 82700

def event82702 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12371⟩⟩) 1 ⟨12370⟩ 82697

def event82703 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12371⟩⟩) (.product (.predecessor 0 82701 .coefficient) (.predecessor 1 82702 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event82704 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12371⟩⟩, .operator (⟨82700, 0⟩, ⟨82697, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9820⟩⟩, ⟨.program ⟨214⟩, ⟨12370⟩⟩], []⟩, (1)⟩)

def exact82705RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9820⟩⟩, ⟨.program ⟨214⟩, ⟨12370⟩⟩], []⟩, (1)⟩]

theorem exact82705RawTermsValid :
    exact82705RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82705 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12371⟩⟩) exact82705RawTerms (.finite 1600) 82703 .exactZero (none)

def event82706 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12372⟩⟩) 0 ⟨12371⟩ 82705

def event82707 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12372⟩⟩) (.identity (.predecessor 0 82706 .coefficient))

def event82708 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12372⟩⟩) (.finite 1600)

def event82709 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16465⟩⟩) 0 ⟨12372⟩ 82708

def event82710 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16465⟩⟩) (.authority (.programFamilyFact))

def exact82711RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16465⟩⟩], []⟩, (1)⟩]

theorem exact82711RawTermsValid :
    exact82711RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82711 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16465⟩⟩) exact82711RawTerms (.finite 40) 82710 .exactZero (none)

def event82712 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16466⟩⟩) 0 ⟨16465⟩ 82711

def event82713 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16466⟩⟩) (.identity (.predecessor 0 82712 .coefficient))

def event82714 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16466⟩⟩) (.finite 40)

def event82715 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24475⟩⟩) 0 ⟨16466⟩ 82714

def event82716 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24475⟩⟩) (.authority (.programFamilyFact))

def event82717 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24475⟩⟩) (.finite 3720)

def event82718 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event82719 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24477⟩⟩) 0 ⟨6689⟩ 82718

def event82720 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24477⟩⟩) 1 ⟨24475⟩ 82717

def event82721 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24477⟩⟩) (.authority (.operator))

def exact82722RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24477⟩⟩]⟩, (1)⟩]

theorem exact82722RawTermsValid :
    exact82722RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82722 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24477⟩⟩) exact82722RawTerms .large 82721 .exactZero (none)

def event82723 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28951⟩⟩) 0 ⟨24477⟩ 82722

def event82724 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28951⟩⟩) (.authority (.operator))

def exact82725RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28951⟩⟩]⟩, (1)⟩]

theorem exact82725RawTermsValid :
    exact82725RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82725 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28951⟩⟩) exact82725RawTerms (.finite 8192) 82724 .exactZero (none)

def event82726 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event82727 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event82728 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16505⟩⟩) 0 ⟨16466⟩ 82714

def event82729 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16505⟩⟩) 1 ⟨110⟩ 82727

def event82730 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16505⟩⟩) (.sum [.predecessor 0 82728 .coefficient, .predecessor 1 82729 .coefficient])

def event82731 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16505⟩⟩) (.finite 40)

def event82732 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16506⟩⟩) 0 ⟨16505⟩ 82731

def event82733 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16506⟩⟩) (.identity (.predecessor 0 82732 .coefficient))

def exact82734RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16465⟩⟩], []⟩, (1)⟩]

theorem exact82734RawTermsValid :
    exact82734RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82734 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16506⟩⟩) exact82734RawTerms (.finite 40) 82733 .exactZero (none)

def event82735 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact82736RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact82736RawTermsValid :
    exact82736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82736 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact82736RawTerms .large 82735 .exactZero (none)

def event82737 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16507⟩⟩) 0 ⟨6544⟩ 82736

def event82738 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16507⟩⟩) 1 ⟨16506⟩ 82734

def event82739 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16507⟩⟩) (.product (.predecessor 0 82737 .coefficient) (.predecessor 1 82738 .coefficient) (⟨false, false, none, none, none⟩))

def event82740 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16507⟩⟩, .operator (⟨82736, 0⟩, ⟨82734, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16465⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact82741RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16465⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact82741RawTermsValid :
    exact82741RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82741 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16507⟩⟩) exact82741RawTerms .large 82739 .exactZero (none)

def event82742 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6702⟩⟩) 0 ⟨6689⟩ 82718

def event82743 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6702⟩⟩) (.authority (.operator))

def exact82744RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩]

theorem exact82744RawTermsValid :
    exact82744RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82744 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6702⟩⟩) exact82744RawTerms .large 82743 .exactZero (none)

def event82745 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16508⟩⟩) 0 ⟨6702⟩ 82744

def event82746 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16508⟩⟩) 1 ⟨16507⟩ 82741

def event82747 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16508⟩⟩) (.sum [.predecessor 0 82745 .coefficient, .predecessor 1 82746 .coefficient])

def exact82748RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16465⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82748RawTermsValid :
    exact82748RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82748 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16508⟩⟩) exact82748RawTerms .large 82747 .exactZero (none)

def event82749 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28952⟩⟩) 0 ⟨16508⟩ 82748

def event82750 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28952⟩⟩) 1 ⟨28951⟩ 82725

def event82751 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28952⟩⟩) (.product (.predecessor 0 82749 .coefficient) (.predecessor 1 82750 .coefficient) (⟨false, false, none, none, none⟩))

def event82752 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28952⟩⟩, .operator (⟨82748, 0⟩, ⟨82725, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28951⟩⟩]⟩, (1)⟩)

def event82753 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28952⟩⟩, .operator (⟨82748, 1⟩, ⟨82725, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16465⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28951⟩⟩]⟩, (-1)⟩)

def event82754 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28952⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16465⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28951⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28951⟩⟩) ⟨24477⟩ 82722)

def event82755 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28952⟩⟩, .relation 82754 0, ⟨[⟨.program ⟨214⟩, ⟨16465⟩⟩], [⟨.program ⟨214⟩, ⟨24477⟩⟩]⟩, (-1)⟩)

def exact82756RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28951⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16465⟩⟩], [⟨.program ⟨214⟩, ⟨24477⟩⟩]⟩, (-1)⟩]

theorem exact82756RawTermsValid :
    exact82756RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82756 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28952⟩⟩) exact82756RawTerms .large 82751 .exactZero (none)

def event82757 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17904⟩⟩) 0 ⟨16466⟩ 82714

def event82758 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17904⟩⟩) (.authority (.programFamilyFact))

def exact82759RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17904⟩⟩], []⟩, (1)⟩]

theorem exact82759RawTermsValid :
    exact82759RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82759 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17904⟩⟩) exact82759RawTerms (.finite 62) 82758 .exactZero (none)

def event82760 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17905⟩⟩) 0 ⟨6544⟩ 82736

def event82761 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17905⟩⟩) 1 ⟨17904⟩ 82759

def event82762 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17905⟩⟩) (.product (.predecessor 0 82760 .coefficient) (.predecessor 1 82761 .coefficient) (⟨false, true, none, none, some 1⟩))

def event82763 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17905⟩⟩, .operator (⟨82736, 0⟩, ⟨82759, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact82764RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact82764RawTermsValid :
    exact82764RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82764 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17905⟩⟩) exact82764RawTerms .large 82762 .exactZero (none)

def event82765 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6733⟩⟩) 0 ⟨6689⟩ 82718

def event82766 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6733⟩⟩) (.authority (.operator))

def exact82767RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩]

theorem exact82767RawTermsValid :
    exact82767RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82767 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6733⟩⟩) exact82767RawTerms .large 82766 .exactZero (none)

def event82768 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17906⟩⟩) 0 ⟨6733⟩ 82767

def event82769 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17906⟩⟩) 1 ⟨17905⟩ 82764

def event82770 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17906⟩⟩) (.sum [.predecessor 0 82768 .coefficient, .predecessor 1 82769 .coefficient])

def exact82771RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82771RawTermsValid :
    exact82771RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82771 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17906⟩⟩) exact82771RawTerms .large 82770 .exactZero (none)

def event82772 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28956⟩⟩) 0 ⟨17906⟩ 82771

def event82773 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28956⟩⟩) 1 ⟨28952⟩ 82756

def event82774 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28956⟩⟩) (.sum [.predecessor 0 82772 .coefficient, .predecessor 1 82773 .coefficient])

def exact82775RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28951⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16465⟩⟩], [⟨.program ⟨214⟩, ⟨24477⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82775RawTermsValid :
    exact82775RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82775 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28956⟩⟩) exact82775RawTerms .large 82774 .exactZero (none)

def event82776 : Event := .preFoldPolynomial 82775 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28951⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16465⟩⟩], [⟨.program ⟨214⟩, ⟨24477⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact82777RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28951⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16465⟩⟩], [⟨.program ⟨214⟩, ⟨24477⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event82777 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28956⟩⟩) 82776 exact82777RawTerms .large 82774 .exactZero (none)

def event82778 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16466⟩⟩) ⟨⟨146⟩, ⟨54⟩, ⟨109⟩⟩ ⟨82620, 82778⟩

def event82779 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨22123⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22120⟩⟩]⟩) (1) 0 2 (.universal 82778 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22120⟩⟩]⟩) (none) 82777)

def event82780 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22123⟩⟩, .relation 82779 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩)

def event82781 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22123⟩⟩, .relation 82779 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28951⟩⟩]⟩, (-1)⟩)

def event82782 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22123⟩⟩, .relation 82779 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16465⟩⟩], [⟨.program ⟨214⟩, ⟨24477⟩⟩]⟩, (1)⟩)

def event82783 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22123⟩⟩, .relation 82779 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact82784RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28951⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16465⟩⟩], [⟨.program ⟨214⟩, ⟨24477⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82784RawTermsValid :
    exact82784RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82784 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22123⟩⟩) exact82784RawTerms .large 82616 (.finite 1811303510016) (some (82618))

def event82785 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28954⟩⟩) 0 ⟨22123⟩ 82784

def event82786 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28954⟩⟩) 1 ⟨28953⟩ 82606

def event82787 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28954⟩⟩) (.sum [.predecessor 0 82785 .coefficient, .predecessor 1 82786 .coefficient])

def event82788 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28954⟩⟩, .operator (⟨82784, 0⟩, ⟨82606, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28951⟩⟩]⟩, (1)⟩)

def event82789 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28954⟩⟩, .operator (⟨82784, 2⟩, ⟨82606, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16465⟩⟩], [⟨.program ⟨214⟩, ⟨24477⟩⟩]⟩, (-1)⟩)

def event82790 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28954⟩⟩) (.sum [.result 82784 .summary, .result 82606 .summary])

def exact82791RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82791RawTermsValid :
    exact82791RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82791 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28954⟩⟩) exact82791RawTerms .large 82787 (.finite 1292315010834812776448) (some (82790))

def event82792 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24412⟩⟩) 0 ⟨16382⟩ 3983

def event82793 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24412⟩⟩) (.authority (.programFamilyFact))

def event82794 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24412⟩⟩) (.finite 3720)

def event82795 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24414⟩⟩) 0 ⟨6689⟩ 5477

def event82796 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24414⟩⟩) 1 ⟨24412⟩ 82794

def event82797 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24414⟩⟩) (.authority (.operator))

def exact82798RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24414⟩⟩]⟩, (1)⟩]

theorem exact82798RawTermsValid :
    exact82798RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82798 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24414⟩⟩) exact82798RawTerms .large 82797 .exactZero (none)

def event82799 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28734⟩⟩) 0 ⟨24414⟩ 82798

def event82800 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28734⟩⟩) (.authority (.operator))

def exact82801RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28734⟩⟩]⟩, (1)⟩]

theorem exact82801RawTermsValid :
    exact82801RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82801 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28734⟩⟩) exact82801RawTerms (.finite 8192) 82800 .exactZero (none)

def event82802 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23121⟩⟩) 0 ⟨11959⟩ 3977

def event82803 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23121⟩⟩) (.authority (.programFamilyFact))

def event82804 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23121⟩⟩) (.finite 3720)

def event82805 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23122⟩⟩) 0 ⟨6689⟩ 5477

def event82806 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23122⟩⟩) 1 ⟨23121⟩ 82804

def event82807 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23122⟩⟩) (.authority (.operator))

def exact82808RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23122⟩⟩]⟩, (1)⟩]

theorem exact82808RawTermsValid :
    exact82808RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82808 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23122⟩⟩) exact82808RawTerms .large 82807 .exactZero (none)

def event82809 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25219⟩⟩) 0 ⟨23122⟩ 82808

def event82810 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25219⟩⟩) (.authority (.operator))

def exact82811RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩, (1)⟩]

theorem exact82811RawTermsValid :
    exact82811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82811 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25219⟩⟩) exact82811RawTerms (.finite 8192) 82810 .exactZero (none)

def event82812 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11960⟩⟩) 0 ⟨11957⟩ 3966

def event82813 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11960⟩⟩) 1 ⟨6567⟩ 79920

def event82814 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11960⟩⟩) (.tensor (.predecessor 0 82812 .coefficient) (.predecessor 1 82813 .coefficient) true false)

def event82815 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11960⟩⟩, .operator (⟨3966, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact82816RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact82816RawTermsValid :
    exact82816RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82816 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11960⟩⟩) exact82816RawTerms .large 82814 .exactZero (none)

def event82817 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7240⟩⟩) 0 ⟨5539⟩ 79790

def event82818 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7240⟩⟩) 1 ⟨6784⟩ 9478

def event82819 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7240⟩⟩) (.product (.predecessor 0 82817 .coefficient) (.predecessor 1 82818 .coefficient) (⟨false, false, none, none, none⟩))

def event82820 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7240⟩⟩, .operator (⟨79790, 0⟩, ⟨9478, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩)

def exact82821RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩]

theorem exact82821RawTermsValid :
    exact82821RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82821 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7240⟩⟩) exact82821RawTerms .large 82819 .exactZero (none)

def event82822 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11961⟩⟩) 0 ⟨7240⟩ 82821

def event82823 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11961⟩⟩) 1 ⟨11960⟩ 82816

def event82824 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11961⟩⟩) (.sum [.predecessor 0 82822 .coefficient, .predecessor 1 82823 .coefficient])

def exact82825RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82825RawTermsValid :
    exact82825RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82825 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11961⟩⟩) exact82825RawTerms .large 82824 .exactZero (none)

def event82826 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11962⟩⟩) 0 ⟨11961⟩ 82825

def event82827 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11962⟩⟩) 1 ⟨98⟩ 9470

def event82828 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11962⟩⟩) (.sum [.predecessor 0 82826 .coefficient, .predecessor 1 82827 .coefficient])

def event82829 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11962⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨98⟩⟩]⟩) [⟨.result 9470 .coefficient, false, none⟩])

def event82830 : Event := .survivorFold (1) 82829

def exact82831RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82831RawTermsValid :
    exact82831RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82831 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11962⟩⟩) exact82831RawTerms .large 82828 (.finite 26) (some (82829))

def event82832 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11963⟩⟩) 0 ⟨11962⟩ 82831

def event82833 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11963⟩⟩) 1 ⟨9715⟩ 3969

def event82834 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11963⟩⟩) (.product (.predecessor 0 82832 .coefficient) (.predecessor 1 82833 .coefficient) (⟨false, true, none, none, some 1⟩))

def event82835 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11963⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩], []⟩) [⟨.result 3969 .coefficient, true, some 1⟩])

def event82836 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11963⟩⟩) (.product (.result 82831 .summary) (.transfer 82835) (⟨false, false, none, none, none⟩))

def event82837 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11963⟩⟩, .operator (⟨82831, 1⟩, ⟨3969, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event82838 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11963⟩⟩, .operator (⟨82831, 0⟩, ⟨3969, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩)

def exact82839RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82839RawTermsValid :
    exact82839RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82839 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11963⟩⟩) exact82839RawTerms .large 82834 (.finite 29952) (some (82836))

def event82840 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9716⟩⟩) 0 ⟨9715⟩ 3969

def event82841 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9716⟩⟩) 1 ⟨6567⟩ 79920

def event82842 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9716⟩⟩) (.tensor (.predecessor 0 82840 .coefficient) (.predecessor 1 82841 .coefficient) true false)

def event82843 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9716⟩⟩, .operator (⟨3969, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact82844RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact82844RawTermsValid :
    exact82844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82844 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9716⟩⟩) exact82844RawTerms .large 82842 .exactZero (none)

def event82845 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7220⟩⟩) 0 ⟨5539⟩ 79790

def event82846 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7220⟩⟩) 1 ⟨6764⟩ 9519

def event82847 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7220⟩⟩) (.product (.predecessor 0 82845 .coefficient) (.predecessor 1 82846 .coefficient) (⟨false, false, none, none, none⟩))

def event82848 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7220⟩⟩, .operator (⟨79790, 0⟩, ⟨9519, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩]⟩, (1)⟩)

def exact82849RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩]⟩, (1)⟩]

theorem exact82849RawTermsValid :
    exact82849RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82849 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7220⟩⟩) exact82849RawTerms .large 82847 .exactZero (none)

def event82850 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9717⟩⟩) 0 ⟨7220⟩ 82849

def event82851 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9717⟩⟩) 1 ⟨9716⟩ 82844

def event82852 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9717⟩⟩) (.sum [.predecessor 0 82850 .coefficient, .predecessor 1 82851 .coefficient])

def exact82853RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82853RawTermsValid :
    exact82853RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82853 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9717⟩⟩) exact82853RawTerms .large 82852 .exactZero (none)

def event82854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9718⟩⟩) 0 ⟨9717⟩ 82853

def event82855 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9718⟩⟩) 1 ⟨78⟩ 9511

def event82856 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9718⟩⟩) (.sum [.predecessor 0 82854 .coefficient, .predecessor 1 82855 .coefficient])

def event82857 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9718⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨78⟩⟩]⟩) [⟨.result 9511 .coefficient, false, none⟩])

def event82858 : Event := .survivorFold (1) 82857

def exact82859RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82859RawTermsValid :
    exact82859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82859 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9718⟩⟩) exact82859RawTerms .large 82856 (.finite 26) (some (82857))

def event82860 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9719⟩⟩) 0 ⟨9718⟩ 82859

def event82861 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9719⟩⟩) 1 ⟨7865⟩ 9508

def event82862 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9719⟩⟩) (.product (.predecessor 0 82860 .coefficient) (.predecessor 1 82861 .coefficient) (⟨false, false, none, none, none⟩))

def event82863 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9719⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩) [⟨.result 9504 .coefficient, false, none⟩])

def event82864 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9719⟩⟩) (.product (.result 82859 .summary) (.transfer 82863) (⟨false, false, none, none, none⟩))

def event82865 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9719⟩⟩, .operator (⟨82859, 1⟩, ⟨9508, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩, (-1)⟩)

def event82866 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨9719⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7864⟩⟩) ⟨6784⟩ 9478)

def event82867 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9719⟩⟩, .relation 82866 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (-1)⟩)

def event82868 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9719⟩⟩, .operator (⟨82859, 0⟩, ⟨9508, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩, (1)⟩)

def exact82869RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (-1)⟩]

theorem exact82869RawTermsValid :
    exact82869RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82869 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9719⟩⟩) exact82869RawTerms .large 82862 (.finite 95420416) (some (82864))

def event82870 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11964⟩⟩) 0 ⟨9719⟩ 82869

def event82871 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11964⟩⟩) 1 ⟨11963⟩ 82839

def event82872 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11964⟩⟩) (.sum [.predecessor 0 82870 .coefficient, .predecessor 1 82871 .coefficient])

def event82873 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11964⟩⟩, .operator (⟨82869, 1⟩, ⟨82839, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩], [⟨.program ⟨214⟩, ⟨6784⟩⟩]⟩, (1)⟩)

def event82874 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11964⟩⟩) (.sum [.result 82869 .summary, .result 82839 .summary])

def exact82875RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact82875RawTermsValid :
    exact82875RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82875 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11964⟩⟩) exact82875RawTerms .large 82872 (.finite 95450368) (some (82874))

def event82876 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25220⟩⟩) 0 ⟨11964⟩ 82875

def event82877 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25220⟩⟩) 1 ⟨25219⟩ 82811

def event82878 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25220⟩⟩) (.product (.predecessor 0 82876 .coefficient) (.predecessor 1 82877 .coefficient) (⟨false, false, none, none, none⟩))

def event82879 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25220⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩) [⟨.result 82811 .coefficient, false, none⟩])

def event82880 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25220⟩⟩) (.product (.result 82875 .summary) (.transfer 82879) (⟨false, false, none, none, none⟩))

def event82881 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25220⟩⟩, .operator (⟨82875, 1⟩, ⟨82811, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩, (-1)⟩)

def event82882 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25220⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25219⟩⟩) ⟨23122⟩ 82808)

def event82883 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25220⟩⟩, .relation 82882 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨23122⟩⟩]⟩, (-1)⟩)

def event82884 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25220⟩⟩, .operator (⟨82875, 0⟩, ⟨82811, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩, (1)⟩)

def exact82885RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨23122⟩⟩]⟩, (-1)⟩]

theorem exact82885RawTermsValid :
    exact82885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82885 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25220⟩⟩) exact82885RawTerms .large 82878 (.finite 350304377765888) (some (82880))

def event82886 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19816⟩⟩) 0 ⟨11959⟩ 3977

def event82887 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19816⟩⟩) (.authority (.relationPreimageSource ⟨19⟩))

def exact82888RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19816⟩⟩]⟩, (1)⟩]

theorem exact82888RawTermsValid :
    exact82888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82888 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19816⟩⟩) exact82888RawTerms (.finite 136065468) 82887 .exactZero (none)

def event82889 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19818⟩⟩) 0 ⟨19816⟩ 82888

def event82890 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19818⟩⟩) 1 ⟨2348⟩ 4

def event82891 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19818⟩⟩) (.scale (.predecessor 0 82889 .coefficient) (.value (.predecessor 1 82890 .coefficient)))

def exact82892RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19816⟩⟩]⟩, (1)⟩]

theorem exact82892RawTermsValid :
    exact82892RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82892 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19818⟩⟩) exact82892RawTerms (.finite 136065468) 82891 .exactZero (none)

def event82893 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19819⟩⟩) 0 ⟨5541⟩ 80012

def event82894 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19819⟩⟩) 1 ⟨19818⟩ 82892

def event82895 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19819⟩⟩) (.product (.predecessor 0 82893 .coefficient) (.predecessor 1 82894 .coefficient) (⟨false, false, none, none, none⟩))

def event82896 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19819⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19816⟩⟩]⟩) [⟨.result 82888 .coefficient, false, none⟩])

def event82897 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19819⟩⟩) (.product (.result 80012 .summary) (.transfer 82896) (⟨false, false, none, none, none⟩))

def event82898 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19819⟩⟩, .operator (⟨80012, 0⟩, ⟨82892, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19816⟩⟩]⟩, (1)⟩)

def event82899 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19817⟩⟩)

def event82900 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event82901 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event82902 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event82903 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event82904 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event82905 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event82906 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event82907 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event82908 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 82907

def event82909 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 82905

def event82910 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 82908 .coefficient) (.value (.predecessor 1 82909 .coefficient)))

def event82911 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event82912 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 82911

def event82913 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 82903

def event82914 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 82912 .coefficient, .predecessor 1 82913 .coefficient])

def event82915 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event82916 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 82915

def event82917 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 82901

def event82918 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 82917 .coefficient))

def event82919 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event82920 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11957⟩⟩) 0 ⟨5536⟩ 82919

def event82921 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11957⟩⟩) (.authority (.programFamilyFact))

def exact82922RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11957⟩⟩], []⟩, (1)⟩]

theorem exact82922RawTermsValid :
    exact82922RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82922 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11957⟩⟩) exact82922RawTerms (.finite 36) 82921 .exactZero (none)

def event82923 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9715⟩⟩) 0 ⟨5536⟩ 82919

def event82924 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9715⟩⟩) (.authority (.programFamilyFact))

def exact82925RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩], []⟩, (1)⟩]

theorem exact82925RawTermsValid :
    exact82925RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82925 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9715⟩⟩) exact82925RawTerms (.finite 36) 82924 .exactZero (none)

def event82926 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11958⟩⟩) 0 ⟨9715⟩ 82925

def event82927 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11958⟩⟩) 1 ⟨11957⟩ 82922

def event82928 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11958⟩⟩) (.product (.predecessor 0 82926 .coefficient) (.predecessor 1 82927 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event82929 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11958⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], []⟩) [⟨.result 82925 .coefficient, true, some 1⟩, ⟨.result 82922 .coefficient, true, some 1⟩])

def event82930 : Event := .survivorFold (1) 82929

def exact82931RawTerms : List Term := []

theorem exact82931RawTermsValid :
    exact82931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82931 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11958⟩⟩) exact82931RawTerms (.finite 1296) 82928 (.finite 1296) (some (82929))

def event82932 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11959⟩⟩) 0 ⟨11958⟩ 82931

def event82933 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11959⟩⟩) (.identity (.predecessor 0 82932 .coefficient))

def event82934 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨11959⟩⟩) (.finite 1296)

def event82935 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19816⟩⟩) 0 ⟨11959⟩ 82934

def event82936 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19816⟩⟩) (.authority (.relationPreimageSource ⟨19⟩))

def exact82937RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19816⟩⟩]⟩, (1)⟩]

theorem exact82937RawTermsValid :
    exact82937RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82937 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19816⟩⟩) exact82937RawTerms (.finite 136065468) 82936 .exactZero (none)

def event82938 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact82939RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact82939RawTermsValid :
    exact82939RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82939 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact82939RawTerms .large 82938 .exactZero (none)

def event82940 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19817⟩⟩) 0 ⟨6⟩ 82939

def event82941 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19817⟩⟩) 1 ⟨19816⟩ 82937

def event82942 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19817⟩⟩) (.product (.predecessor 0 82940 .coefficient) (.predecessor 1 82941 .coefficient) (⟨false, false, none, none, none⟩))

def event82943 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19817⟩⟩, .operator (⟨82939, 0⟩, ⟨82937, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19816⟩⟩]⟩, (1)⟩)

def eventLeaf5168 : Array AnnotatedEvent := #[
  { event := event82688
    frameStart := 82674 },
  { event := event82689
    frameStart := 82674 },
  { event := event82690
    frameStart := 82674 },
  { event := event82691
    frameStart := 82674 },
  { event := event82692
    frameStart := 82674 },
  { event := event82693
    frameStart := 82674 },
  { event := event82694
    frameStart := 82674 },
  { event := event82695
    frameStart := 82674 },
  { event := event82696
    frameStart := 82674 },
  { event := event82697
    frameStart := 82674 },
  { event := event82698
    frameStart := 82674 },
  { event := event82699
    frameStart := 82674 },
  { event := event82700
    frameStart := 82674 },
  { event := event82701
    frameStart := 82674 },
  { event := event82702
    frameStart := 82674 },
  { event := event82703
    frameStart := 82674 }
]

def eventLeaf5169 : Array AnnotatedEvent := #[
  { event := event82704
    frameStart := 82674 },
  { event := event82705
    frameStart := 82674 },
  { event := event82706
    frameStart := 82674 },
  { event := event82707
    frameStart := 82674 },
  { event := event82708
    frameStart := 82674 },
  { event := event82709
    frameStart := 82674 },
  { event := event82710
    frameStart := 82674 },
  { event := event82711
    frameStart := 82674 },
  { event := event82712
    frameStart := 82674 },
  { event := event82713
    frameStart := 82674 },
  { event := event82714
    frameStart := 82674 },
  { event := event82715
    frameStart := 82674 },
  { event := event82716
    frameStart := 82674 },
  { event := event82717
    frameStart := 82674 },
  { event := event82718
    frameStart := 82674 },
  { event := event82719
    frameStart := 82674 }
]

def eventLeaf5170 : Array AnnotatedEvent := #[
  { event := event82720
    frameStart := 82674 },
  { event := event82721
    frameStart := 82674 },
  { event := event82722
    frameStart := 82674 },
  { event := event82723
    frameStart := 82674 },
  { event := event82724
    frameStart := 82674 },
  { event := event82725
    frameStart := 82674 },
  { event := event82726
    frameStart := 82674 },
  { event := event82727
    frameStart := 82674 },
  { event := event82728
    frameStart := 82674 },
  { event := event82729
    frameStart := 82674 },
  { event := event82730
    frameStart := 82674 },
  { event := event82731
    frameStart := 82674 },
  { event := event82732
    frameStart := 82674 },
  { event := event82733
    frameStart := 82674 },
  { event := event82734
    frameStart := 82674 },
  { event := event82735
    frameStart := 82674 }
]

def eventLeaf5171 : Array AnnotatedEvent := #[
  { event := event82736
    frameStart := 82674 },
  { event := event82737
    frameStart := 82674 },
  { event := event82738
    frameStart := 82674 },
  { event := event82739
    frameStart := 82674 },
  { event := event82740
    frameStart := 82674 },
  { event := event82741
    frameStart := 82674 },
  { event := event82742
    frameStart := 82674 },
  { event := event82743
    frameStart := 82674 },
  { event := event82744
    frameStart := 82674 },
  { event := event82745
    frameStart := 82674 },
  { event := event82746
    frameStart := 82674 },
  { event := event82747
    frameStart := 82674 },
  { event := event82748
    frameStart := 82674 },
  { event := event82749
    frameStart := 82674 },
  { event := event82750
    frameStart := 82674 },
  { event := event82751
    frameStart := 82674 }
]

def eventLeaf5172 : Array AnnotatedEvent := #[
  { event := event82752
    frameStart := 82674 },
  { event := event82753
    frameStart := 82674 },
  { event := event82754
    frameStart := 82674 },
  { event := event82755
    frameStart := 82674 },
  { event := event82756
    frameStart := 82674 },
  { event := event82757
    frameStart := 82674 },
  { event := event82758
    frameStart := 82674 },
  { event := event82759
    frameStart := 82674 },
  { event := event82760
    frameStart := 82674 },
  { event := event82761
    frameStart := 82674 },
  { event := event82762
    frameStart := 82674 },
  { event := event82763
    frameStart := 82674 },
  { event := event82764
    frameStart := 82674 },
  { event := event82765
    frameStart := 82674 },
  { event := event82766
    frameStart := 82674 },
  { event := event82767
    frameStart := 82674 }
]

def eventLeaf5173 : Array AnnotatedEvent := #[
  { event := event82768
    frameStart := 82674 },
  { event := event82769
    frameStart := 82674 },
  { event := event82770
    frameStart := 82674 },
  { event := event82771
    frameStart := 82674 },
  { event := event82772
    frameStart := 82674 },
  { event := event82773
    frameStart := 82674 },
  { event := event82774
    frameStart := 82674 },
  { event := event82775
    frameStart := 82674 },
  { event := event82776
    frameStart := 82674 },
  { event := event82777
    frameStart := 82674 },
  { event := event82778
    frameStart := 0 },
  { event := event82779
    frameStart := 0 },
  { event := event82780
    frameStart := 0 },
  { event := event82781
    frameStart := 0 },
  { event := event82782
    frameStart := 0 },
  { event := event82783
    frameStart := 0 }
]

def eventLeaf5174 : Array AnnotatedEvent := #[
  { event := event82784
    frameStart := 0 },
  { event := event82785
    frameStart := 0 },
  { event := event82786
    frameStart := 0 },
  { event := event82787
    frameStart := 0 },
  { event := event82788
    frameStart := 0 },
  { event := event82789
    frameStart := 0 },
  { event := event82790
    frameStart := 0 },
  { event := event82791
    frameStart := 0 },
  { event := event82792
    frameStart := 0 },
  { event := event82793
    frameStart := 0 },
  { event := event82794
    frameStart := 0 },
  { event := event82795
    frameStart := 0 },
  { event := event82796
    frameStart := 0 },
  { event := event82797
    frameStart := 0 },
  { event := event82798
    frameStart := 0 },
  { event := event82799
    frameStart := 0 }
]

def eventLeaf5175 : Array AnnotatedEvent := #[
  { event := event82800
    frameStart := 0 },
  { event := event82801
    frameStart := 0 },
  { event := event82802
    frameStart := 0 },
  { event := event82803
    frameStart := 0 },
  { event := event82804
    frameStart := 0 },
  { event := event82805
    frameStart := 0 },
  { event := event82806
    frameStart := 0 },
  { event := event82807
    frameStart := 0 },
  { event := event82808
    frameStart := 0 },
  { event := event82809
    frameStart := 0 },
  { event := event82810
    frameStart := 0 },
  { event := event82811
    frameStart := 0 },
  { event := event82812
    frameStart := 0 },
  { event := event82813
    frameStart := 0 },
  { event := event82814
    frameStart := 0 },
  { event := event82815
    frameStart := 0 }
]

def eventLeaf5176 : Array AnnotatedEvent := #[
  { event := event82816
    frameStart := 0 },
  { event := event82817
    frameStart := 0 },
  { event := event82818
    frameStart := 0 },
  { event := event82819
    frameStart := 0 },
  { event := event82820
    frameStart := 0 },
  { event := event82821
    frameStart := 0 },
  { event := event82822
    frameStart := 0 },
  { event := event82823
    frameStart := 0 },
  { event := event82824
    frameStart := 0 },
  { event := event82825
    frameStart := 0 },
  { event := event82826
    frameStart := 0 },
  { event := event82827
    frameStart := 0 },
  { event := event82828
    frameStart := 0 },
  { event := event82829
    frameStart := 0 },
  { event := event82830
    frameStart := 0 },
  { event := event82831
    frameStart := 0 }
]

def eventLeaf5177 : Array AnnotatedEvent := #[
  { event := event82832
    frameStart := 0 },
  { event := event82833
    frameStart := 0 },
  { event := event82834
    frameStart := 0 },
  { event := event82835
    frameStart := 0 },
  { event := event82836
    frameStart := 0 },
  { event := event82837
    frameStart := 0 },
  { event := event82838
    frameStart := 0 },
  { event := event82839
    frameStart := 0 },
  { event := event82840
    frameStart := 0 },
  { event := event82841
    frameStart := 0 },
  { event := event82842
    frameStart := 0 },
  { event := event82843
    frameStart := 0 },
  { event := event82844
    frameStart := 0 },
  { event := event82845
    frameStart := 0 },
  { event := event82846
    frameStart := 0 },
  { event := event82847
    frameStart := 0 }
]

def eventLeaf5178 : Array AnnotatedEvent := #[
  { event := event82848
    frameStart := 0 },
  { event := event82849
    frameStart := 0 },
  { event := event82850
    frameStart := 0 },
  { event := event82851
    frameStart := 0 },
  { event := event82852
    frameStart := 0 },
  { event := event82853
    frameStart := 0 },
  { event := event82854
    frameStart := 0 },
  { event := event82855
    frameStart := 0 },
  { event := event82856
    frameStart := 0 },
  { event := event82857
    frameStart := 0 },
  { event := event82858
    frameStart := 0 },
  { event := event82859
    frameStart := 0 },
  { event := event82860
    frameStart := 0 },
  { event := event82861
    frameStart := 0 },
  { event := event82862
    frameStart := 0 },
  { event := event82863
    frameStart := 0 }
]

def eventLeaf5179 : Array AnnotatedEvent := #[
  { event := event82864
    frameStart := 0 },
  { event := event82865
    frameStart := 0 },
  { event := event82866
    frameStart := 0 },
  { event := event82867
    frameStart := 0 },
  { event := event82868
    frameStart := 0 },
  { event := event82869
    frameStart := 0 },
  { event := event82870
    frameStart := 0 },
  { event := event82871
    frameStart := 0 },
  { event := event82872
    frameStart := 0 },
  { event := event82873
    frameStart := 0 },
  { event := event82874
    frameStart := 0 },
  { event := event82875
    frameStart := 0 },
  { event := event82876
    frameStart := 0 },
  { event := event82877
    frameStart := 0 },
  { event := event82878
    frameStart := 0 },
  { event := event82879
    frameStart := 0 }
]

def eventLeaf5180 : Array AnnotatedEvent := #[
  { event := event82880
    frameStart := 0 },
  { event := event82881
    frameStart := 0 },
  { event := event82882
    frameStart := 0 },
  { event := event82883
    frameStart := 0 },
  { event := event82884
    frameStart := 0 },
  { event := event82885
    frameStart := 0 },
  { event := event82886
    frameStart := 0 },
  { event := event82887
    frameStart := 0 },
  { event := event82888
    frameStart := 0 },
  { event := event82889
    frameStart := 0 },
  { event := event82890
    frameStart := 0 },
  { event := event82891
    frameStart := 0 },
  { event := event82892
    frameStart := 0 },
  { event := event82893
    frameStart := 0 },
  { event := event82894
    frameStart := 0 },
  { event := event82895
    frameStart := 0 }
]

def eventLeaf5181 : Array AnnotatedEvent := #[
  { event := event82896
    frameStart := 0 },
  { event := event82897
    frameStart := 0 },
  { event := event82898
    frameStart := 0 },
  { event := event82899
    frameStart := 82899 },
  { event := event82900
    frameStart := 82899 },
  { event := event82901
    frameStart := 82899 },
  { event := event82902
    frameStart := 82899 },
  { event := event82903
    frameStart := 82899 },
  { event := event82904
    frameStart := 82899 },
  { event := event82905
    frameStart := 82899 },
  { event := event82906
    frameStart := 82899 },
  { event := event82907
    frameStart := 82899 },
  { event := event82908
    frameStart := 82899 },
  { event := event82909
    frameStart := 82899 },
  { event := event82910
    frameStart := 82899 },
  { event := event82911
    frameStart := 82899 }
]

def eventLeaf5182 : Array AnnotatedEvent := #[
  { event := event82912
    frameStart := 82899 },
  { event := event82913
    frameStart := 82899 },
  { event := event82914
    frameStart := 82899 },
  { event := event82915
    frameStart := 82899 },
  { event := event82916
    frameStart := 82899 },
  { event := event82917
    frameStart := 82899 },
  { event := event82918
    frameStart := 82899 },
  { event := event82919
    frameStart := 82899 },
  { event := event82920
    frameStart := 82899 },
  { event := event82921
    frameStart := 82899 },
  { event := event82922
    frameStart := 82899 },
  { event := event82923
    frameStart := 82899 },
  { event := event82924
    frameStart := 82899 },
  { event := event82925
    frameStart := 82899 },
  { event := event82926
    frameStart := 82899 },
  { event := event82927
    frameStart := 82899 }
]

def eventLeaf5183 : Array AnnotatedEvent := #[
  { event := event82928
    frameStart := 82899 },
  { event := event82929
    frameStart := 82899 },
  { event := event82930
    frameStart := 82899 },
  { event := event82931
    frameStart := 82899 },
  { event := event82932
    frameStart := 82899 },
  { event := event82933
    frameStart := 82899 },
  { event := event82934
    frameStart := 82899 },
  { event := event82935
    frameStart := 82899 },
  { event := event82936
    frameStart := 82899 },
  { event := event82937
    frameStart := 82899 },
  { event := event82938
    frameStart := 82899 },
  { event := event82939
    frameStart := 82899 },
  { event := event82940
    frameStart := 82899 },
  { event := event82941
    frameStart := 82899 },
  { event := event82942
    frameStart := 82899 },
  { event := event82943
    frameStart := 82899 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events323
