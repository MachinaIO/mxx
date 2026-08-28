import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events288

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event73728 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24909⟩⟩, .operator (⟨73724, 0⟩, ⟨73681, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24906⟩⟩]⟩, (1)⟩)

def event73729 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24909⟩⟩, .operator (⟨73724, 1⟩, ⟨73681, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24906⟩⟩]⟩, (-1)⟩)

def event73730 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨24909⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24906⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨24906⟩⟩) ⟨22952⟩ 73678)

def event73731 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24909⟩⟩, .relation 73730 0, ⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨22952⟩⟩]⟩, (-1)⟩)

def exact73732RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24906⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨22952⟩⟩]⟩, (-1)⟩]

theorem exact73732RawTermsValid :
    exact73732RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73732 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24909⟩⟩) exact73732RawTerms .large 73727 .exactZero (none)

def event73733 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14788⟩⟩) 0 ⟨10474⟩ 73670

def event73734 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14788⟩⟩) (.authority (.programFamilyFact))

def exact73735RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], []⟩, (1)⟩]

theorem exact73735RawTermsValid :
    exact73735RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73735 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14788⟩⟩) exact73735RawTerms (.finite 2) 73734 .exactZero (none)

def event73736 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14790⟩⟩) 0 ⟨6544⟩ 73692

def event73737 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14790⟩⟩) 1 ⟨14788⟩ 73735

def event73738 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14790⟩⟩) (.product (.predecessor 0 73736 .coefficient) (.predecessor 1 73737 .coefficient) (⟨false, true, none, none, some 1⟩))

def event73739 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14790⟩⟩, .operator (⟨73692, 0⟩, ⟨73735, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact73740RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact73740RawTermsValid :
    exact73740RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73740 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14790⟩⟩) exact73740RawTerms .large 73738 .exactZero (none)

def event73741 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6690⟩⟩) 0 ⟨6689⟩ 73674

def event73742 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6690⟩⟩) (.authority (.operator))

def exact73743RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩]

theorem exact73743RawTermsValid :
    exact73743RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73743 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6690⟩⟩) exact73743RawTerms .large 73742 .exactZero (none)

def event73744 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14791⟩⟩) 0 ⟨6690⟩ 73743

def event73745 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14791⟩⟩) 1 ⟨14790⟩ 73740

def event73746 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14791⟩⟩) (.sum [.predecessor 0 73744 .coefficient, .predecessor 1 73745 .coefficient])

def exact73747RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73747RawTermsValid :
    exact73747RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73747 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14791⟩⟩) exact73747RawTerms .large 73746 .exactZero (none)

def event73748 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24910⟩⟩) 0 ⟨14791⟩ 73747

def event73749 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24910⟩⟩) 1 ⟨24909⟩ 73732

def event73750 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24910⟩⟩) (.sum [.predecessor 0 73748 .coefficient, .predecessor 1 73749 .coefficient])

def exact73751RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24906⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨22952⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73751RawTermsValid :
    exact73751RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73751 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24910⟩⟩) exact73751RawTerms .large 73750 .exactZero (none)

def event73752 : Event := .preFoldPolynomial 73751 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24906⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨22952⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact73753RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24906⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨22952⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event73753 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨24910⟩⟩) 73752 exact73753RawTerms .large 73750 .exactZero (none)

def event73754 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨10474⟩⟩) ⟨⟨103⟩, ⟨7⟩, ⟨109⟩⟩ ⟨73588, 73754⟩

def event73755 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19023⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19020⟩⟩]⟩) (1) 0 2 (.universal 73754 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19020⟩⟩]⟩) (none) 73753)

def event73756 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19023⟩⟩, .relation 73755 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩)

def event73757 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19023⟩⟩, .relation 73755 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24906⟩⟩]⟩, (-1)⟩)

def event73758 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19023⟩⟩, .relation 73755 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨22952⟩⟩]⟩, (1)⟩)

def event73759 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19023⟩⟩, .relation 73755 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact73760RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24906⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨22952⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73760RawTermsValid :
    exact73760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73760 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19023⟩⟩) exact73760RawTerms .large 73584 (.finite 1811303510016) (some (73586))

def event73761 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24908⟩⟩) 0 ⟨19023⟩ 73760

def event73762 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24908⟩⟩) 1 ⟨24907⟩ 73574

def event73763 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24908⟩⟩) (.sum [.predecessor 0 73761 .coefficient, .predecessor 1 73762 .coefficient])

def event73764 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24908⟩⟩, .operator (⟨73760, 2⟩, ⟨73574, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], [⟨.program ⟨214⟩, ⟨22952⟩⟩]⟩, (-1)⟩)

def event73765 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24908⟩⟩, .operator (⟨73760, 1⟩, ⟨73574, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24906⟩⟩]⟩, (1)⟩)

def event73766 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24908⟩⟩) (.sum [.result 73760 .summary, .result 73574 .summary])

def exact73767RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73767RawTermsValid :
    exact73767RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73767 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24908⟩⟩) exact73767RawTerms .large 73763 (.finite 352011863863296) (some (73766))

def event73768 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26348⟩⟩) 0 ⟨24908⟩ 73767

def event73769 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26348⟩⟩) 1 ⟨26346⟩ 73490

def event73770 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26348⟩⟩) (.product (.predecessor 0 73768 .coefficient) (.predecessor 1 73769 .coefficient) (⟨false, false, none, none, none⟩))

def event73771 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26348⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩) [⟨.result 73490 .coefficient, false, none⟩])

def event73772 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26348⟩⟩) (.product (.result 73767 .summary) (.transfer 73771) (⟨false, false, none, none, none⟩))

def event73773 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26348⟩⟩, .operator (⟨73767, 0⟩, ⟨73490, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩, (1)⟩)

def event73774 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26348⟩⟩, .operator (⟨73767, 1⟩, ⟨73490, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩, (-1)⟩)

def event73775 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26348⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26346⟩⟩) ⟨23718⟩ 73487)

def event73776 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26348⟩⟩, .relation 73775 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨23718⟩⟩]⟩, (-1)⟩)

def exact73777RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨23718⟩⟩]⟩, (-1)⟩]

theorem exact73777RawTermsValid :
    exact73777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73777 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26348⟩⟩) exact73777RawTerms .large 73770 (.finite 1291889172568118132736) (some (73772))

def event73778 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20388⟩⟩) 0 ⟨14789⟩ 3494

def event73779 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20388⟩⟩) (.authority (.relationPreimageSource ⟨28⟩))

def exact73780RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩, (1)⟩]

theorem exact73780RawTermsValid :
    exact73780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73780 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20388⟩⟩) exact73780RawTerms (.finite 136065468) 73779 .exactZero (none)

def event73781 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20390⟩⟩) 0 ⟨20388⟩ 73780

def event73782 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20390⟩⟩) 1 ⟨2348⟩ 4

def event73783 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20390⟩⟩) (.scale (.predecessor 0 73781 .coefficient) (.value (.predecessor 1 73782 .coefficient)))

def exact73784RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩, (1)⟩]

theorem exact73784RawTermsValid :
    exact73784RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73784 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20390⟩⟩) exact73784RawTerms (.finite 136065468) 73783 .exactZero (none)

def event73785 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20391⟩⟩) 0 ⟨5535⟩ 65387

def event73786 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20391⟩⟩) 1 ⟨20390⟩ 73784

def event73787 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20391⟩⟩) (.product (.predecessor 0 73785 .coefficient) (.predecessor 1 73786 .coefficient) (⟨false, false, none, none, none⟩))

def event73788 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20391⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩) [⟨.result 73780 .coefficient, false, none⟩])

def event73789 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20391⟩⟩) (.product (.result 65387 .summary) (.transfer 73788) (⟨false, false, none, none, none⟩))

def event73790 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20391⟩⟩, .operator (⟨65387, 0⟩, ⟨73784, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩, (1)⟩)

def event73791 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20389⟩⟩)

def event73792 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event73793 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event73794 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event73795 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event73796 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event73797 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event73798 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event73799 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event73800 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 73799

def event73801 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 73797

def event73802 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 73800 .coefficient) (.value (.predecessor 1 73801 .coefficient)))

def event73803 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event73804 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 73803

def event73805 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 73795

def event73806 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 73804 .coefficient, .predecessor 1 73805 .coefficient])

def event73807 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event73808 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 73807

def event73809 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 73793

def event73810 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 73809 .coefficient))

def event73811 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event73812 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10472⟩⟩) 0 ⟨5530⟩ 73811

def event73813 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10472⟩⟩) (.authority (.programFamilyFact))

def exact73814RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩, (1)⟩]

theorem exact73814RawTermsValid :
    exact73814RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73814 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10472⟩⟩) exact73814RawTerms (.finite 2) 73813 .exactZero (none)

def event73815 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9395⟩⟩) 0 ⟨5530⟩ 73811

def event73816 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9395⟩⟩) (.authority (.programFamilyFact))

def exact73817RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩], []⟩, (1)⟩]

theorem exact73817RawTermsValid :
    exact73817RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73817 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9395⟩⟩) exact73817RawTerms (.finite 2) 73816 .exactZero (none)

def event73818 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10473⟩⟩) 0 ⟨9395⟩ 73817

def event73819 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10473⟩⟩) 1 ⟨10472⟩ 73814

def event73820 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10473⟩⟩) (.product (.predecessor 0 73818 .coefficient) (.predecessor 1 73819 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event73821 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10473⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩) [⟨.result 73817 .coefficient, true, some 1⟩, ⟨.result 73814 .coefficient, true, some 1⟩])

def event73822 : Event := .survivorFold (1) 73821

def exact73823RawTerms : List Term := []

theorem exact73823RawTermsValid :
    exact73823RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73823 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10473⟩⟩) exact73823RawTerms (.finite 4) 73820 (.finite 4) (some (73821))

def event73824 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10474⟩⟩) 0 ⟨10473⟩ 73823

def event73825 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10474⟩⟩) (.identity (.predecessor 0 73824 .coefficient))

def event73826 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10474⟩⟩) (.finite 4)

def event73827 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14788⟩⟩) 0 ⟨10474⟩ 73826

def event73828 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14788⟩⟩) (.authority (.programFamilyFact))

def exact73829RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], []⟩, (1)⟩]

theorem exact73829RawTermsValid :
    exact73829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73829 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14788⟩⟩) exact73829RawTerms (.finite 2) 73828 .exactZero (none)

def event73830 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14789⟩⟩) 0 ⟨14788⟩ 73829

def event73831 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14789⟩⟩) (.identity (.predecessor 0 73830 .coefficient))

def event73832 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14789⟩⟩) (.finite 2)

def event73833 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20388⟩⟩) 0 ⟨14789⟩ 73832

def event73834 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20388⟩⟩) (.authority (.relationPreimageSource ⟨28⟩))

def exact73835RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩, (1)⟩]

theorem exact73835RawTermsValid :
    exact73835RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73835 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20388⟩⟩) exact73835RawTerms (.finite 136065468) 73834 .exactZero (none)

def event73836 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact73837RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact73837RawTermsValid :
    exact73837RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73837 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact73837RawTerms .large 73836 .exactZero (none)

def event73838 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20389⟩⟩) 0 ⟨6⟩ 73837

def event73839 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20389⟩⟩) 1 ⟨20388⟩ 73835

def event73840 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20389⟩⟩) (.product (.predecessor 0 73838 .coefficient) (.predecessor 1 73839 .coefficient) (⟨false, false, none, none, none⟩))

def event73841 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20389⟩⟩, .operator (⟨73837, 0⟩, ⟨73835, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩, (1)⟩)

def exact73842RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩, (1)⟩]

theorem exact73842RawTermsValid :
    exact73842RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73842 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20389⟩⟩) exact73842RawTerms .large 73840 .exactZero (none)

def event73843 : Event := .preFoldPolynomial 73842 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩, (1)⟩] .exactZero none

def exact73844RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩, (1)⟩]

def event73844 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20389⟩⟩) 73843 exact73844RawTerms .large 73840 .exactZero (none)

def event73845 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26350⟩⟩)

def event73846 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event73847 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event73848 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event73849 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event73850 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event73851 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event73852 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event73853 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event73854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 73853

def event73855 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 73851

def event73856 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 73854 .coefficient) (.value (.predecessor 1 73855 .coefficient)))

def event73857 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event73858 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 73857

def event73859 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 73849

def event73860 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 73858 .coefficient, .predecessor 1 73859 .coefficient])

def event73861 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event73862 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 73861

def event73863 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 73847

def event73864 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 73863 .coefficient))

def event73865 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event73866 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10472⟩⟩) 0 ⟨5530⟩ 73865

def event73867 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10472⟩⟩) (.authority (.programFamilyFact))

def exact73868RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩, (1)⟩]

theorem exact73868RawTermsValid :
    exact73868RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73868 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10472⟩⟩) exact73868RawTerms (.finite 2) 73867 .exactZero (none)

def event73869 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9395⟩⟩) 0 ⟨5530⟩ 73865

def event73870 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9395⟩⟩) (.authority (.programFamilyFact))

def exact73871RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩], []⟩, (1)⟩]

theorem exact73871RawTermsValid :
    exact73871RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73871 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9395⟩⟩) exact73871RawTerms (.finite 2) 73870 .exactZero (none)

def event73872 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10473⟩⟩) 0 ⟨9395⟩ 73871

def event73873 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10473⟩⟩) 1 ⟨10472⟩ 73868

def event73874 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10473⟩⟩) (.product (.predecessor 0 73872 .coefficient) (.predecessor 1 73873 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event73875 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10473⟩⟩, .operator (⟨73871, 0⟩, ⟨73868, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩, (1)⟩)

def exact73876RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9395⟩⟩, ⟨.program ⟨214⟩, ⟨10472⟩⟩], []⟩, (1)⟩]

theorem exact73876RawTermsValid :
    exact73876RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73876 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10473⟩⟩) exact73876RawTerms (.finite 4) 73874 .exactZero (none)

def event73877 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10474⟩⟩) 0 ⟨10473⟩ 73876

def event73878 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10474⟩⟩) (.identity (.predecessor 0 73877 .coefficient))

def event73879 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10474⟩⟩) (.finite 4)

def event73880 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14788⟩⟩) 0 ⟨10474⟩ 73879

def event73881 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14788⟩⟩) (.authority (.programFamilyFact))

def exact73882RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], []⟩, (1)⟩]

theorem exact73882RawTermsValid :
    exact73882RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73882 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14788⟩⟩) exact73882RawTerms (.finite 2) 73881 .exactZero (none)

def event73883 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14789⟩⟩) 0 ⟨14788⟩ 73882

def event73884 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14789⟩⟩) (.identity (.predecessor 0 73883 .coefficient))

def event73885 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14789⟩⟩) (.finite 2)

def event73886 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23716⟩⟩) 0 ⟨14789⟩ 73885

def event73887 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23716⟩⟩) (.authority (.programFamilyFact))

def event73888 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23716⟩⟩) (.finite 3720)

def event73889 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event73890 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23718⟩⟩) 0 ⟨6689⟩ 73889

def event73891 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23718⟩⟩) 1 ⟨23716⟩ 73888

def event73892 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23718⟩⟩) (.authority (.operator))

def exact73893RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23718⟩⟩]⟩, (1)⟩]

theorem exact73893RawTermsValid :
    exact73893RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73893 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23718⟩⟩) exact73893RawTerms .large 73892 .exactZero (none)

def event73894 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26346⟩⟩) 0 ⟨23718⟩ 73893

def event73895 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26346⟩⟩) (.authority (.operator))

def exact73896RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩, (1)⟩]

theorem exact73896RawTermsValid :
    exact73896RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73896 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26346⟩⟩) exact73896RawTerms (.finite 8192) 73895 .exactZero (none)

def event73897 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event73898 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event73899 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14828⟩⟩) 0 ⟨14789⟩ 73885

def event73900 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14828⟩⟩) 1 ⟨110⟩ 73898

def event73901 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14828⟩⟩) (.sum [.predecessor 0 73899 .coefficient, .predecessor 1 73900 .coefficient])

def event73902 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14828⟩⟩) (.finite 2)

def event73903 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14829⟩⟩) 0 ⟨14828⟩ 73902

def event73904 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14829⟩⟩) (.identity (.predecessor 0 73903 .coefficient))

def exact73905RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], []⟩, (1)⟩]

theorem exact73905RawTermsValid :
    exact73905RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73905 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14829⟩⟩) exact73905RawTerms (.finite 2) 73904 .exactZero (none)

def event73906 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact73907RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact73907RawTermsValid :
    exact73907RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73907 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact73907RawTerms .large 73906 .exactZero (none)

def event73908 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14830⟩⟩) 0 ⟨6544⟩ 73907

def event73909 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14830⟩⟩) 1 ⟨14829⟩ 73905

def event73910 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14830⟩⟩) (.product (.predecessor 0 73908 .coefficient) (.predecessor 1 73909 .coefficient) (⟨false, false, none, none, none⟩))

def event73911 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14830⟩⟩, .operator (⟨73907, 0⟩, ⟨73905, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact73912RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact73912RawTermsValid :
    exact73912RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73912 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14830⟩⟩) exact73912RawTerms .large 73910 .exactZero (none)

def event73913 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6690⟩⟩) 0 ⟨6689⟩ 73889

def event73914 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6690⟩⟩) (.authority (.operator))

def exact73915RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩]

theorem exact73915RawTermsValid :
    exact73915RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73915 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6690⟩⟩) exact73915RawTerms .large 73914 .exactZero (none)

def event73916 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14831⟩⟩) 0 ⟨6690⟩ 73915

def event73917 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14831⟩⟩) 1 ⟨14830⟩ 73912

def event73918 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14831⟩⟩) (.sum [.predecessor 0 73916 .coefficient, .predecessor 1 73917 .coefficient])

def exact73919RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73919RawTermsValid :
    exact73919RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73919 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14831⟩⟩) exact73919RawTerms .large 73918 .exactZero (none)

def event73920 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26347⟩⟩) 0 ⟨14831⟩ 73919

def event73921 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26347⟩⟩) 1 ⟨26346⟩ 73896

def event73922 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26347⟩⟩) (.product (.predecessor 0 73920 .coefficient) (.predecessor 1 73921 .coefficient) (⟨false, false, none, none, none⟩))

def event73923 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26347⟩⟩, .operator (⟨73919, 0⟩, ⟨73896, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩, (1)⟩)

def event73924 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26347⟩⟩, .operator (⟨73919, 1⟩, ⟨73896, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩, (-1)⟩)

def event73925 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26347⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26346⟩⟩) ⟨23718⟩ 73893)

def event73926 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26347⟩⟩, .relation 73925 0, ⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨23718⟩⟩]⟩, (-1)⟩)

def exact73927RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨23718⟩⟩]⟩, (-1)⟩]

theorem exact73927RawTermsValid :
    exact73927RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73927 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26347⟩⟩) exact73927RawTerms .large 73922 .exactZero (none)

def event73928 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15262⟩⟩) 0 ⟨14789⟩ 73885

def event73929 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15262⟩⟩) (.authority (.programFamilyFact))

def exact73930RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], []⟩, (1)⟩]

theorem exact73930RawTermsValid :
    exact73930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73930 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15262⟩⟩) exact73930RawTerms (.finite 43) 73929 .exactZero (none)

def event73931 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15263⟩⟩) 0 ⟨6544⟩ 73907

def event73932 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15263⟩⟩) 1 ⟨15262⟩ 73930

def event73933 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15263⟩⟩) (.product (.predecessor 0 73931 .coefficient) (.predecessor 1 73932 .coefficient) (⟨false, true, none, none, some 1⟩))

def event73934 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15263⟩⟩, .operator (⟨73907, 0⟩, ⟨73930, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact73935RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact73935RawTermsValid :
    exact73935RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73935 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15263⟩⟩) exact73935RawTerms .large 73933 .exactZero (none)

def event73936 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6709⟩⟩) 0 ⟨6689⟩ 73889

def event73937 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6709⟩⟩) (.authority (.operator))

def exact73938RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩]

theorem exact73938RawTermsValid :
    exact73938RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73938 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6709⟩⟩) exact73938RawTerms .large 73937 .exactZero (none)

def event73939 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15264⟩⟩) 0 ⟨6709⟩ 73938

def event73940 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15264⟩⟩) 1 ⟨15263⟩ 73935

def event73941 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15264⟩⟩) (.sum [.predecessor 0 73939 .coefficient, .predecessor 1 73940 .coefficient])

def exact73942RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73942RawTermsValid :
    exact73942RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73942 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15264⟩⟩) exact73942RawTerms .large 73941 .exactZero (none)

def event73943 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26350⟩⟩) 0 ⟨15264⟩ 73942

def event73944 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26350⟩⟩) 1 ⟨26347⟩ 73927

def event73945 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26350⟩⟩) (.sum [.predecessor 0 73943 .coefficient, .predecessor 1 73944 .coefficient])

def exact73946RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨23718⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73946RawTermsValid :
    exact73946RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73946 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26350⟩⟩) exact73946RawTerms .large 73945 .exactZero (none)

def event73947 : Event := .preFoldPolynomial 73946 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨23718⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact73948RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨23718⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event73948 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26350⟩⟩) 73947 exact73948RawTerms .large 73945 .exactZero (none)

def event73949 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14789⟩⟩) ⟨⟨122⟩, ⟨28⟩, ⟨109⟩⟩ ⟨73791, 73949⟩

def event73950 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20391⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩) (1) 0 2 (.universal 73949 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩) (none) 73948)

def event73951 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20391⟩⟩, .relation 73950 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩)

def event73952 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20391⟩⟩, .relation 73950 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩, (-1)⟩)

def event73953 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20391⟩⟩, .relation 73950 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨23718⟩⟩]⟩, (1)⟩)

def event73954 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20391⟩⟩, .relation 73950 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact73955RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨23718⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73955RawTermsValid :
    exact73955RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73955 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20391⟩⟩) exact73955RawTerms .large 73787 (.finite 1811303510016) (some (73789))

def event73956 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26349⟩⟩) 0 ⟨20391⟩ 73955

def event73957 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26349⟩⟩) 1 ⟨26348⟩ 73777

def event73958 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26349⟩⟩) (.sum [.predecessor 0 73956 .coefficient, .predecessor 1 73957 .coefficient])

def event73959 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26349⟩⟩, .operator (⟨73955, 0⟩, ⟨73777, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩, (1)⟩)

def event73960 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26349⟩⟩, .operator (⟨73955, 2⟩, ⟨73777, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], [⟨.program ⟨214⟩, ⟨23718⟩⟩]⟩, (-1)⟩)

def event73961 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26349⟩⟩) (.sum [.result 73955 .summary, .result 73777 .summary])

def exact73962RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73962RawTermsValid :
    exact73962RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73962 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26349⟩⟩) exact73962RawTerms .large 73958 (.finite 1291889174379421642752) (some (73961))

def event73963 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26555⟩⟩) 0 ⟨26349⟩ 73962

def event73964 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26555⟩⟩) 1 ⟨26554⟩ 73480

def event73965 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26555⟩⟩) (.sum [.predecessor 0 73963 .coefficient, .predecessor 1 73964 .coefficient])

def event73966 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26555⟩⟩) (.sum [.result 73962 .summary, .result 73480 .summary])

def exact73967RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15306⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73967RawTermsValid :
    exact73967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73967 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26555⟩⟩) exact73967RawTerms .large 73965 (.finite 2583789554981353578496) (some (73966))

def event73968 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26772⟩⟩) 0 ⟨26555⟩ 73967

def event73969 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26772⟩⟩) 1 ⟨26771⟩ 72998

def event73970 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26772⟩⟩) (.sum [.predecessor 0 73968 .coefficient, .predecessor 1 73969 .coefficient])

def event73971 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26772⟩⟩) (.sum [.result 73967 .summary, .result 72998 .summary])

def exact73972RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15306⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73972RawTermsValid :
    exact73972RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73972 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26772⟩⟩) exact73972RawTerms .large 73970 (.finite 3875701141805795807232) (some (73971))

def event73973 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26989⟩⟩) 0 ⟨26772⟩ 73972

def event73974 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26989⟩⟩) 1 ⟨26988⟩ 72516

def event73975 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26989⟩⟩) (.sum [.predecessor 0 73973 .coefficient, .predecessor 1 73974 .coefficient])

def event73976 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26989⟩⟩) (.sum [.result 73972 .summary, .result 72516 .summary])

def exact73977RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15306⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73977RawTermsValid :
    exact73977RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73977 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26989⟩⟩) exact73977RawTerms .large 73975 (.finite 5167635141075258621952) (some (73976))

def event73978 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27206⟩⟩) 0 ⟨26989⟩ 73977

def event73979 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27206⟩⟩) 1 ⟨27205⟩ 72034

def event73980 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27206⟩⟩) (.sum [.predecessor 0 73978 .coefficient, .predecessor 1 73979 .coefficient])

def event73981 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27206⟩⟩) (.sum [.result 73977 .summary, .result 72034 .summary])

def exact73982RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15306⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15626⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact73982RawTermsValid :
    exact73982RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event73982 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27206⟩⟩) exact73982RawTerms .large 73980 (.finite 6459613965234762608640) (some (73981))

def event73983 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27423⟩⟩) 0 ⟨27206⟩ 73982

def eventLeaf4608 : Array AnnotatedEvent := #[
  { event := event73728
    frameStart := 73636 },
  { event := event73729
    frameStart := 73636 },
  { event := event73730
    frameStart := 73636 },
  { event := event73731
    frameStart := 73636 },
  { event := event73732
    frameStart := 73636 },
  { event := event73733
    frameStart := 73636 },
  { event := event73734
    frameStart := 73636 },
  { event := event73735
    frameStart := 73636 },
  { event := event73736
    frameStart := 73636 },
  { event := event73737
    frameStart := 73636 },
  { event := event73738
    frameStart := 73636 },
  { event := event73739
    frameStart := 73636 },
  { event := event73740
    frameStart := 73636 },
  { event := event73741
    frameStart := 73636 },
  { event := event73742
    frameStart := 73636 },
  { event := event73743
    frameStart := 73636 }
]

def eventLeaf4609 : Array AnnotatedEvent := #[
  { event := event73744
    frameStart := 73636 },
  { event := event73745
    frameStart := 73636 },
  { event := event73746
    frameStart := 73636 },
  { event := event73747
    frameStart := 73636 },
  { event := event73748
    frameStart := 73636 },
  { event := event73749
    frameStart := 73636 },
  { event := event73750
    frameStart := 73636 },
  { event := event73751
    frameStart := 73636 },
  { event := event73752
    frameStart := 73636 },
  { event := event73753
    frameStart := 73636 },
  { event := event73754
    frameStart := 0 },
  { event := event73755
    frameStart := 0 },
  { event := event73756
    frameStart := 0 },
  { event := event73757
    frameStart := 0 },
  { event := event73758
    frameStart := 0 },
  { event := event73759
    frameStart := 0 }
]

def eventLeaf4610 : Array AnnotatedEvent := #[
  { event := event73760
    frameStart := 0 },
  { event := event73761
    frameStart := 0 },
  { event := event73762
    frameStart := 0 },
  { event := event73763
    frameStart := 0 },
  { event := event73764
    frameStart := 0 },
  { event := event73765
    frameStart := 0 },
  { event := event73766
    frameStart := 0 },
  { event := event73767
    frameStart := 0 },
  { event := event73768
    frameStart := 0 },
  { event := event73769
    frameStart := 0 },
  { event := event73770
    frameStart := 0 },
  { event := event73771
    frameStart := 0 },
  { event := event73772
    frameStart := 0 },
  { event := event73773
    frameStart := 0 },
  { event := event73774
    frameStart := 0 },
  { event := event73775
    frameStart := 0 }
]

def eventLeaf4611 : Array AnnotatedEvent := #[
  { event := event73776
    frameStart := 0 },
  { event := event73777
    frameStart := 0 },
  { event := event73778
    frameStart := 0 },
  { event := event73779
    frameStart := 0 },
  { event := event73780
    frameStart := 0 },
  { event := event73781
    frameStart := 0 },
  { event := event73782
    frameStart := 0 },
  { event := event73783
    frameStart := 0 },
  { event := event73784
    frameStart := 0 },
  { event := event73785
    frameStart := 0 },
  { event := event73786
    frameStart := 0 },
  { event := event73787
    frameStart := 0 },
  { event := event73788
    frameStart := 0 },
  { event := event73789
    frameStart := 0 },
  { event := event73790
    frameStart := 0 },
  { event := event73791
    frameStart := 73791 }
]

def eventLeaf4612 : Array AnnotatedEvent := #[
  { event := event73792
    frameStart := 73791 },
  { event := event73793
    frameStart := 73791 },
  { event := event73794
    frameStart := 73791 },
  { event := event73795
    frameStart := 73791 },
  { event := event73796
    frameStart := 73791 },
  { event := event73797
    frameStart := 73791 },
  { event := event73798
    frameStart := 73791 },
  { event := event73799
    frameStart := 73791 },
  { event := event73800
    frameStart := 73791 },
  { event := event73801
    frameStart := 73791 },
  { event := event73802
    frameStart := 73791 },
  { event := event73803
    frameStart := 73791 },
  { event := event73804
    frameStart := 73791 },
  { event := event73805
    frameStart := 73791 },
  { event := event73806
    frameStart := 73791 },
  { event := event73807
    frameStart := 73791 }
]

def eventLeaf4613 : Array AnnotatedEvent := #[
  { event := event73808
    frameStart := 73791 },
  { event := event73809
    frameStart := 73791 },
  { event := event73810
    frameStart := 73791 },
  { event := event73811
    frameStart := 73791 },
  { event := event73812
    frameStart := 73791 },
  { event := event73813
    frameStart := 73791 },
  { event := event73814
    frameStart := 73791 },
  { event := event73815
    frameStart := 73791 },
  { event := event73816
    frameStart := 73791 },
  { event := event73817
    frameStart := 73791 },
  { event := event73818
    frameStart := 73791 },
  { event := event73819
    frameStart := 73791 },
  { event := event73820
    frameStart := 73791 },
  { event := event73821
    frameStart := 73791 },
  { event := event73822
    frameStart := 73791 },
  { event := event73823
    frameStart := 73791 }
]

def eventLeaf4614 : Array AnnotatedEvent := #[
  { event := event73824
    frameStart := 73791 },
  { event := event73825
    frameStart := 73791 },
  { event := event73826
    frameStart := 73791 },
  { event := event73827
    frameStart := 73791 },
  { event := event73828
    frameStart := 73791 },
  { event := event73829
    frameStart := 73791 },
  { event := event73830
    frameStart := 73791 },
  { event := event73831
    frameStart := 73791 },
  { event := event73832
    frameStart := 73791 },
  { event := event73833
    frameStart := 73791 },
  { event := event73834
    frameStart := 73791 },
  { event := event73835
    frameStart := 73791 },
  { event := event73836
    frameStart := 73791 },
  { event := event73837
    frameStart := 73791 },
  { event := event73838
    frameStart := 73791 },
  { event := event73839
    frameStart := 73791 }
]

def eventLeaf4615 : Array AnnotatedEvent := #[
  { event := event73840
    frameStart := 73791 },
  { event := event73841
    frameStart := 73791 },
  { event := event73842
    frameStart := 73791 },
  { event := event73843
    frameStart := 73791 },
  { event := event73844
    frameStart := 73791 },
  { event := event73845
    frameStart := 73845 },
  { event := event73846
    frameStart := 73845 },
  { event := event73847
    frameStart := 73845 },
  { event := event73848
    frameStart := 73845 },
  { event := event73849
    frameStart := 73845 },
  { event := event73850
    frameStart := 73845 },
  { event := event73851
    frameStart := 73845 },
  { event := event73852
    frameStart := 73845 },
  { event := event73853
    frameStart := 73845 },
  { event := event73854
    frameStart := 73845 },
  { event := event73855
    frameStart := 73845 }
]

def eventLeaf4616 : Array AnnotatedEvent := #[
  { event := event73856
    frameStart := 73845 },
  { event := event73857
    frameStart := 73845 },
  { event := event73858
    frameStart := 73845 },
  { event := event73859
    frameStart := 73845 },
  { event := event73860
    frameStart := 73845 },
  { event := event73861
    frameStart := 73845 },
  { event := event73862
    frameStart := 73845 },
  { event := event73863
    frameStart := 73845 },
  { event := event73864
    frameStart := 73845 },
  { event := event73865
    frameStart := 73845 },
  { event := event73866
    frameStart := 73845 },
  { event := event73867
    frameStart := 73845 },
  { event := event73868
    frameStart := 73845 },
  { event := event73869
    frameStart := 73845 },
  { event := event73870
    frameStart := 73845 },
  { event := event73871
    frameStart := 73845 }
]

def eventLeaf4617 : Array AnnotatedEvent := #[
  { event := event73872
    frameStart := 73845 },
  { event := event73873
    frameStart := 73845 },
  { event := event73874
    frameStart := 73845 },
  { event := event73875
    frameStart := 73845 },
  { event := event73876
    frameStart := 73845 },
  { event := event73877
    frameStart := 73845 },
  { event := event73878
    frameStart := 73845 },
  { event := event73879
    frameStart := 73845 },
  { event := event73880
    frameStart := 73845 },
  { event := event73881
    frameStart := 73845 },
  { event := event73882
    frameStart := 73845 },
  { event := event73883
    frameStart := 73845 },
  { event := event73884
    frameStart := 73845 },
  { event := event73885
    frameStart := 73845 },
  { event := event73886
    frameStart := 73845 },
  { event := event73887
    frameStart := 73845 }
]

def eventLeaf4618 : Array AnnotatedEvent := #[
  { event := event73888
    frameStart := 73845 },
  { event := event73889
    frameStart := 73845 },
  { event := event73890
    frameStart := 73845 },
  { event := event73891
    frameStart := 73845 },
  { event := event73892
    frameStart := 73845 },
  { event := event73893
    frameStart := 73845 },
  { event := event73894
    frameStart := 73845 },
  { event := event73895
    frameStart := 73845 },
  { event := event73896
    frameStart := 73845 },
  { event := event73897
    frameStart := 73845 },
  { event := event73898
    frameStart := 73845 },
  { event := event73899
    frameStart := 73845 },
  { event := event73900
    frameStart := 73845 },
  { event := event73901
    frameStart := 73845 },
  { event := event73902
    frameStart := 73845 },
  { event := event73903
    frameStart := 73845 }
]

def eventLeaf4619 : Array AnnotatedEvent := #[
  { event := event73904
    frameStart := 73845 },
  { event := event73905
    frameStart := 73845 },
  { event := event73906
    frameStart := 73845 },
  { event := event73907
    frameStart := 73845 },
  { event := event73908
    frameStart := 73845 },
  { event := event73909
    frameStart := 73845 },
  { event := event73910
    frameStart := 73845 },
  { event := event73911
    frameStart := 73845 },
  { event := event73912
    frameStart := 73845 },
  { event := event73913
    frameStart := 73845 },
  { event := event73914
    frameStart := 73845 },
  { event := event73915
    frameStart := 73845 },
  { event := event73916
    frameStart := 73845 },
  { event := event73917
    frameStart := 73845 },
  { event := event73918
    frameStart := 73845 },
  { event := event73919
    frameStart := 73845 }
]

def eventLeaf4620 : Array AnnotatedEvent := #[
  { event := event73920
    frameStart := 73845 },
  { event := event73921
    frameStart := 73845 },
  { event := event73922
    frameStart := 73845 },
  { event := event73923
    frameStart := 73845 },
  { event := event73924
    frameStart := 73845 },
  { event := event73925
    frameStart := 73845 },
  { event := event73926
    frameStart := 73845 },
  { event := event73927
    frameStart := 73845 },
  { event := event73928
    frameStart := 73845 },
  { event := event73929
    frameStart := 73845 },
  { event := event73930
    frameStart := 73845 },
  { event := event73931
    frameStart := 73845 },
  { event := event73932
    frameStart := 73845 },
  { event := event73933
    frameStart := 73845 },
  { event := event73934
    frameStart := 73845 },
  { event := event73935
    frameStart := 73845 }
]

def eventLeaf4621 : Array AnnotatedEvent := #[
  { event := event73936
    frameStart := 73845 },
  { event := event73937
    frameStart := 73845 },
  { event := event73938
    frameStart := 73845 },
  { event := event73939
    frameStart := 73845 },
  { event := event73940
    frameStart := 73845 },
  { event := event73941
    frameStart := 73845 },
  { event := event73942
    frameStart := 73845 },
  { event := event73943
    frameStart := 73845 },
  { event := event73944
    frameStart := 73845 },
  { event := event73945
    frameStart := 73845 },
  { event := event73946
    frameStart := 73845 },
  { event := event73947
    frameStart := 73845 },
  { event := event73948
    frameStart := 73845 },
  { event := event73949
    frameStart := 0 },
  { event := event73950
    frameStart := 0 },
  { event := event73951
    frameStart := 0 }
]

def eventLeaf4622 : Array AnnotatedEvent := #[
  { event := event73952
    frameStart := 0 },
  { event := event73953
    frameStart := 0 },
  { event := event73954
    frameStart := 0 },
  { event := event73955
    frameStart := 0 },
  { event := event73956
    frameStart := 0 },
  { event := event73957
    frameStart := 0 },
  { event := event73958
    frameStart := 0 },
  { event := event73959
    frameStart := 0 },
  { event := event73960
    frameStart := 0 },
  { event := event73961
    frameStart := 0 },
  { event := event73962
    frameStart := 0 },
  { event := event73963
    frameStart := 0 },
  { event := event73964
    frameStart := 0 },
  { event := event73965
    frameStart := 0 },
  { event := event73966
    frameStart := 0 },
  { event := event73967
    frameStart := 0 }
]

def eventLeaf4623 : Array AnnotatedEvent := #[
  { event := event73968
    frameStart := 0 },
  { event := event73969
    frameStart := 0 },
  { event := event73970
    frameStart := 0 },
  { event := event73971
    frameStart := 0 },
  { event := event73972
    frameStart := 0 },
  { event := event73973
    frameStart := 0 },
  { event := event73974
    frameStart := 0 },
  { event := event73975
    frameStart := 0 },
  { event := event73976
    frameStart := 0 },
  { event := event73977
    frameStart := 0 },
  { event := event73978
    frameStart := 0 },
  { event := event73979
    frameStart := 0 },
  { event := event73980
    frameStart := 0 },
  { event := event73981
    frameStart := 0 },
  { event := event73982
    frameStart := 0 },
  { event := event73983
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events288
