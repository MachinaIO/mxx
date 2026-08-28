import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events475

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event121600 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5523⟩⟩) (.finite 655360)

def event121601 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39698⟩⟩) 0 ⟨5523⟩ 121600

def event121602 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39698⟩⟩) (.authority (.programFamilyFact))

def exact121603RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨39698⟩⟩], []⟩, (1)⟩]

theorem exact121603RawTermsValid :
    exact121603RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121603 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39698⟩⟩) exact121603RawTerms (.finite 46) 121602 .exactZero (none)

def event121604 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14121⟩⟩) 0 ⟨5523⟩ 121600

def event121605 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14121⟩⟩) (.authority (.programFamilyFact))

def exact121606RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14121⟩⟩], []⟩, (1)⟩]

theorem exact121606RawTermsValid :
    exact121606RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121606 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14121⟩⟩) exact121606RawTerms (.finite 46) 121605 .exactZero (none)

def event121607 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39699⟩⟩) 0 ⟨14121⟩ 121606

def event121608 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39699⟩⟩) 1 ⟨39698⟩ 121603

def event121609 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39699⟩⟩) (.product (.predecessor 0 121607 .coefficient) (.predecessor 1 121608 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event121610 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39699⟩⟩, .operator (⟨121606, 0⟩, ⟨121603, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], []⟩, (1)⟩)

def exact121611RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14121⟩⟩, ⟨.program ⟨257⟩, ⟨39698⟩⟩], []⟩, (1)⟩]

theorem exact121611RawTermsValid :
    exact121611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121611 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39699⟩⟩) exact121611RawTerms (.finite 2116) 121609 .exactZero (none)

def event121612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39700⟩⟩) 0 ⟨39699⟩ 121611

def event121613 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39700⟩⟩) (.identity (.predecessor 0 121612 .coefficient))

def event121614 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨39700⟩⟩) (.finite 2116)

def event121615 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40076⟩⟩) 0 ⟨39700⟩ 121614

def event121616 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40076⟩⟩) (.authority (.programFamilyFact))

def exact121617RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], []⟩, (1)⟩]

theorem exact121617RawTermsValid :
    exact121617RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121617 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40076⟩⟩) exact121617RawTerms (.finite 46) 121616 .exactZero (none)

def event121618 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40077⟩⟩) 0 ⟨40076⟩ 121617

def event121619 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40077⟩⟩) (.identity (.predecessor 0 121618 .coefficient))

def event121620 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨40077⟩⟩) (.finite 46)

def event121621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41223⟩⟩) 0 ⟨40077⟩ 121620

def event121622 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41223⟩⟩) (.authority (.programFamilyFact))

def event121623 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41223⟩⟩) (.finite 3720)

def event121624 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event121625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41225⟩⟩) 0 ⟨7177⟩ 121624

def event121626 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41225⟩⟩) 1 ⟨41223⟩ 121623

def event121627 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41225⟩⟩) (.authority (.operator))

def exact121628RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41225⟩⟩]⟩, (1)⟩]

theorem exact121628RawTermsValid :
    exact121628RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121628 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41225⟩⟩) exact121628RawTerms .large 121627 .exactZero (none)

def event121629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41889⟩⟩) 0 ⟨41225⟩ 121628

def event121630 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41889⟩⟩) (.authority (.operator))

def exact121631RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨41889⟩⟩]⟩, (1)⟩]

theorem exact121631RawTermsValid :
    exact121631RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121631 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41889⟩⟩) exact121631RawTerms (.finite 8192) 121630 .exactZero (none)

def event121632 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event121633 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event121634 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41450⟩⟩) 0 ⟨40077⟩ 121620

def event121635 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41450⟩⟩) 1 ⟨136⟩ 121633

def event121636 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41450⟩⟩) (.sum [.predecessor 0 121634 .coefficient, .predecessor 1 121635 .coefficient])

def event121637 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨41450⟩⟩) (.finite 46)

def event121638 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41451⟩⟩) 0 ⟨41450⟩ 121637

def event121639 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41451⟩⟩) (.identity (.predecessor 0 121638 .coefficient))

def exact121640RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], []⟩, (1)⟩]

theorem exact121640RawTermsValid :
    exact121640RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121640 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41451⟩⟩) exact121640RawTerms (.finite 46) 121639 .exactZero (none)

def event121641 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact121642RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact121642RawTermsValid :
    exact121642RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121642 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact121642RawTerms .large 121641 .exactZero (none)

def event121643 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41452⟩⟩) 0 ⟨6908⟩ 121642

def event121644 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41452⟩⟩) 1 ⟨41451⟩ 121640

def event121645 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41452⟩⟩) (.product (.predecessor 0 121643 .coefficient) (.predecessor 1 121644 .coefficient) (⟨false, false, none, none, none⟩))

def event121646 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41452⟩⟩, .operator (⟨121642, 0⟩, ⟨121640, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact121647RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact121647RawTermsValid :
    exact121647RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121647 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41452⟩⟩) exact121647RawTerms .large 121645 .exactZero (none)

def event121648 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7193⟩⟩) 0 ⟨7177⟩ 121624

def event121649 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7193⟩⟩) (.authority (.operator))

def exact121650RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩]

theorem exact121650RawTermsValid :
    exact121650RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121650 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7193⟩⟩) exact121650RawTerms .large 121649 .exactZero (none)

def event121651 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41453⟩⟩) 0 ⟨7193⟩ 121650

def event121652 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41453⟩⟩) 1 ⟨41452⟩ 121647

def event121653 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41453⟩⟩) (.sum [.predecessor 0 121651 .coefficient, .predecessor 1 121652 .coefficient])

def exact121654RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact121654RawTermsValid :
    exact121654RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121654 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41453⟩⟩) exact121654RawTerms .large 121653 .exactZero (none)

def event121655 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41890⟩⟩) 0 ⟨41453⟩ 121654

def event121656 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41890⟩⟩) 1 ⟨41889⟩ 121631

def event121657 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41890⟩⟩) (.product (.predecessor 0 121655 .coefficient) (.predecessor 1 121656 .coefficient) (⟨false, false, none, none, none⟩))

def event121658 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41890⟩⟩, .operator (⟨121654, 0⟩, ⟨121631, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41889⟩⟩]⟩, (1)⟩)

def event121659 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41890⟩⟩, .operator (⟨121654, 1⟩, ⟨121631, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41889⟩⟩]⟩, (-1)⟩)

def event121660 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨41890⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41889⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41889⟩⟩) ⟨41225⟩ 121628)

def event121661 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41890⟩⟩, .relation 121660 0, ⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨41225⟩⟩]⟩, (-1)⟩)

def exact121662RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41889⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨41225⟩⟩]⟩, (-1)⟩]

theorem exact121662RawTermsValid :
    exact121662RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121662 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41890⟩⟩) exact121662RawTerms .large 121657 .exactZero (none)

def event121663 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40267⟩⟩) 0 ⟨40077⟩ 121620

def event121664 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40267⟩⟩) (.authority (.programFamilyFact))

def exact121665RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40267⟩⟩], []⟩, (1)⟩]

theorem exact121665RawTermsValid :
    exact121665RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121665 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40267⟩⟩) exact121665RawTerms (.finite 63) 121664 .exactZero (none)

def event121666 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40268⟩⟩) 0 ⟨6908⟩ 121642

def event121667 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40268⟩⟩) 1 ⟨40267⟩ 121665

def event121668 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40268⟩⟩) (.product (.predecessor 0 121666 .coefficient) (.predecessor 1 121667 .coefficient) (⟨false, true, none, none, some 1⟩))

def event121669 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40268⟩⟩, .operator (⟨121642, 0⟩, ⟨121665, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨40267⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact121670RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40267⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact121670RawTermsValid :
    exact121670RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121670 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40268⟩⟩) exact121670RawTerms .large 121668 .exactZero (none)

def event121671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7226⟩⟩) 0 ⟨7177⟩ 121624

def event121672 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7226⟩⟩) (.authority (.operator))

def exact121673RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩]

theorem exact121673RawTermsValid :
    exact121673RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121673 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7226⟩⟩) exact121673RawTerms .large 121672 .exactZero (none)

def event121674 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40269⟩⟩) 0 ⟨7226⟩ 121673

def event121675 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40269⟩⟩) 1 ⟨40268⟩ 121670

def event121676 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40269⟩⟩) (.sum [.predecessor 0 121674 .coefficient, .predecessor 1 121675 .coefficient])

def exact121677RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40267⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact121677RawTermsValid :
    exact121677RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121677 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40269⟩⟩) exact121677RawTerms .large 121676 .exactZero (none)

def event121678 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41893⟩⟩) 0 ⟨40269⟩ 121677

def event121679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41893⟩⟩) 1 ⟨41890⟩ 121662

def event121680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41893⟩⟩) (.sum [.predecessor 0 121678 .coefficient, .predecessor 1 121679 .coefficient])

def exact121681RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41889⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨41225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40267⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact121681RawTermsValid :
    exact121681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121681 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41893⟩⟩) exact121681RawTerms .large 121680 .exactZero (none)

def event121682 : Event := .preFoldPolynomial 121681 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41889⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨41225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40267⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact121683RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41889⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨41225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨40267⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event121683 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨41893⟩⟩) 121682 exact121683RawTerms .large 121680 .exactZero (none)

def event121684 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨40077⟩⟩) ⟨⟨105⟩, ⟨87⟩, ⟨135⟩⟩ ⟨121526, 121684⟩

def event121685 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨40779⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40776⟩⟩]⟩) (1) 0 2 (.universal 121684 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40776⟩⟩]⟩) (none) 121683)

def event121686 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40779⟩⟩, .relation 121685 1, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩)

def event121687 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40779⟩⟩, .relation 121685 0, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41889⟩⟩]⟩, (-1)⟩)

def event121688 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40779⟩⟩, .relation 121685 2, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨41225⟩⟩]⟩, (1)⟩)

def event121689 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40779⟩⟩, .relation 121685 3, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40267⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact121690RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41889⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨41225⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40267⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact121690RawTermsValid :
    exact121690RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121690 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40779⟩⟩) exact121690RawTerms .large 121522 (.finite 202072841853861888) (some (121524))

def event121691 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41892⟩⟩) 0 ⟨40779⟩ 121690

def event121692 : Event := .predecessor (⟨.program ⟨257⟩, ⟨41892⟩⟩) 1 ⟨41891⟩ 121512

def event121693 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41892⟩⟩) (.sum [.predecessor 0 121691 .coefficient, .predecessor 1 121692 .coefficient])

def event121694 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41892⟩⟩, .operator (⟨121690, 0⟩, ⟨121512, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41889⟩⟩]⟩, (1)⟩)

def event121695 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨41892⟩⟩, .operator (⟨121690, 2⟩, ⟨121512, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40076⟩⟩], [⟨.program ⟨257⟩, ⟨41225⟩⟩]⟩, (-1)⟩)

def event121696 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨41892⟩⟩) (.sum [.result 121690 .summary, .result 121512 .summary])

def exact121697RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨40267⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact121697RawTermsValid :
    exact121697RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121697 : Event := .resultExact (⟨.program ⟨257⟩, ⟨41892⟩⟩) exact121697RawTerms .large 121693 (.finite 32193129122288829188810200055808) (some (121696))

def event121698 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38543⟩⟩) 0 ⟨37397⟩ 5439

def event121699 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38543⟩⟩) (.authority (.programFamilyFact))

def event121700 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38543⟩⟩) (.finite 3720)

def event121701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38545⟩⟩) 0 ⟨7177⟩ 15500

def event121702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38545⟩⟩) 1 ⟨38543⟩ 121700

def event121703 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38545⟩⟩) (.authority (.operator))

def exact121704RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38545⟩⟩]⟩, (1)⟩]

theorem exact121704RawTermsValid :
    exact121704RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121704 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38545⟩⟩) exact121704RawTerms .large 121703 .exactZero (none)

def event121705 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39209⟩⟩) 0 ⟨38545⟩ 121704

def event121706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39209⟩⟩) (.authority (.operator))

def exact121707RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39209⟩⟩]⟩, (1)⟩]

theorem exact121707RawTermsValid :
    exact121707RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121707 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39209⟩⟩) exact121707RawTerms (.finite 8192) 121706 .exactZero (none)

def event121708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38404⟩⟩) 0 ⟨37020⟩ 5433

def event121709 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38404⟩⟩) (.authority (.programFamilyFact))

def event121710 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38404⟩⟩) (.finite 3720)

def event121711 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38405⟩⟩) 0 ⟨7177⟩ 15500

def event121712 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38405⟩⟩) 1 ⟨38404⟩ 121710

def event121713 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38405⟩⟩) (.authority (.operator))

def exact121714RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38405⟩⟩]⟩, (1)⟩]

theorem exact121714RawTermsValid :
    exact121714RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121714 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38405⟩⟩) exact121714RawTerms .large 121713 .exactZero (none)

def event121715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38895⟩⟩) 0 ⟨38405⟩ 121714

def event121716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38895⟩⟩) (.authority (.operator))

def exact121717RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38895⟩⟩]⟩, (1)⟩]

theorem exact121717RawTermsValid :
    exact121717RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121717 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38895⟩⟩) exact121717RawTerms (.finite 8192) 121716 .exactZero (none)

def event121718 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37021⟩⟩) 0 ⟨37018⟩ 5422

def event121719 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37021⟩⟩) 1 ⟨6928⟩ 119778

def event121720 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37021⟩⟩) (.tensor (.predecessor 0 121718 .coefficient) (.predecessor 1 121719 .coefficient) true false)

def event121721 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37021⟩⟩, .operator (⟨5422, 0⟩, ⟨119778, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact121722RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact121722RawTermsValid :
    exact121722RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121722 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37021⟩⟩) exact121722RawTerms .large 121720 .exactZero (none)

def event121723 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8131⟩⟩) 0 ⟨5525⟩ 119648

def event121724 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8131⟩⟩) 1 ⟨7281⟩ 19084

def event121725 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8131⟩⟩) (.product (.predecessor 0 121723 .coefficient) (.predecessor 1 121724 .coefficient) (⟨false, false, none, none, none⟩))

def event121726 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8131⟩⟩, .operator (⟨119648, 0⟩, ⟨19084, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩)

def exact121727RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩]

theorem exact121727RawTermsValid :
    exact121727RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121727 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8131⟩⟩) exact121727RawTerms .large 121725 .exactZero (none)

def event121728 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37022⟩⟩) 0 ⟨8131⟩ 121727

def event121729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37022⟩⟩) 1 ⟨37021⟩ 121722

def event121730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37022⟩⟩) (.sum [.predecessor 0 121728 .coefficient, .predecessor 1 121729 .coefficient])

def exact121731RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact121731RawTermsValid :
    exact121731RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121731 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37022⟩⟩) exact121731RawTerms .large 121730 .exactZero (none)

def event121732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37023⟩⟩) 0 ⟨37022⟩ 121731

def event121733 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37023⟩⟩) 1 ⟨107⟩ 19076

def event121734 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37023⟩⟩) (.sum [.predecessor 0 121732 .coefficient, .predecessor 1 121733 .coefficient])

def event121735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37023⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨107⟩⟩]⟩) [⟨.result 19076 .coefficient, false, none⟩])

def event121736 : Event := .survivorFold (1) 121735

def exact121737RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact121737RawTermsValid :
    exact121737RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121737 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37023⟩⟩) exact121737RawTerms .large 121734 (.finite 26) (some (121735))

def event121738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37024⟩⟩) 0 ⟨37023⟩ 121737

def event121739 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37024⟩⟩) 1 ⟨13821⟩ 5425

def event121740 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37024⟩⟩) (.product (.predecessor 0 121738 .coefficient) (.predecessor 1 121739 .coefficient) (⟨false, true, none, none, some 1⟩))

def event121741 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37024⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13821⟩⟩], []⟩) [⟨.result 5425 .coefficient, true, some 1⟩])

def event121742 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37024⟩⟩) (.product (.result 121737 .summary) (.transfer 121741) (⟨false, false, none, none, none⟩))

def event121743 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37024⟩⟩, .operator (⟨121737, 1⟩, ⟨5425, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event121744 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37024⟩⟩, .operator (⟨121737, 0⟩, ⟨5425, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩)

def exact121745RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact121745RawTermsValid :
    exact121745RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121745 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37024⟩⟩) exact121745RawTerms .large 121740 (.finite 35782656) (some (121742))

def event121746 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13822⟩⟩) 0 ⟨13821⟩ 5425

def event121747 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13822⟩⟩) 1 ⟨6928⟩ 119778

def event121748 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13822⟩⟩) (.tensor (.predecessor 0 121746 .coefficient) (.predecessor 1 121747 .coefficient) true false)

def event121749 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13822⟩⟩, .operator (⟨5425, 0⟩, ⟨119778, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact121750RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact121750RawTermsValid :
    exact121750RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121750 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13822⟩⟩) exact121750RawTerms .large 121748 .exactZero (none)

def event121751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8148⟩⟩) 0 ⟨5525⟩ 119648

def event121752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨8148⟩⟩) 1 ⟨7298⟩ 19125

def event121753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨8148⟩⟩) (.product (.predecessor 0 121751 .coefficient) (.predecessor 1 121752 .coefficient) (⟨false, false, none, none, none⟩))

def event121754 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨8148⟩⟩, .operator (⟨119648, 0⟩, ⟨19125, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩)

def exact121755RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩]

theorem exact121755RawTermsValid :
    exact121755RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121755 : Event := .resultExact (⟨.program ⟨257⟩, ⟨8148⟩⟩) exact121755RawTerms .large 121753 .exactZero (none)

def event121756 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13823⟩⟩) 0 ⟨8148⟩ 121755

def event121757 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13823⟩⟩) 1 ⟨13822⟩ 121750

def event121758 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13823⟩⟩) (.sum [.predecessor 0 121756 .coefficient, .predecessor 1 121757 .coefficient])

def exact121759RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact121759RawTermsValid :
    exact121759RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121759 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13823⟩⟩) exact121759RawTerms .large 121758 .exactZero (none)

def event121760 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13824⟩⟩) 0 ⟨13823⟩ 121759

def event121761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13824⟩⟩) 1 ⟨124⟩ 19117

def event121762 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13824⟩⟩) (.sum [.predecessor 0 121760 .coefficient, .predecessor 1 121761 .coefficient])

def event121763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13824⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨124⟩⟩]⟩) [⟨.result 19117 .coefficient, false, none⟩])

def event121764 : Event := .survivorFold (1) 121763

def exact121765RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact121765RawTermsValid :
    exact121765RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121765 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13824⟩⟩) exact121765RawTerms .large 121762 (.finite 26) (some (121763))

def event121766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13825⟩⟩) 0 ⟨13824⟩ 121765

def event121767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13825⟩⟩) 1 ⟨9554⟩ 19114

def event121768 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13825⟩⟩) (.product (.predecessor 0 121766 .coefficient) (.predecessor 1 121767 .coefficient) (⟨false, false, none, none, none⟩))

def event121769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13825⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) [⟨.result 19110 .coefficient, false, none⟩])

def event121770 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13825⟩⟩) (.product (.result 121765 .summary) (.transfer 121769) (⟨false, false, none, none, none⟩))

def event121771 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13825⟩⟩, .operator (⟨121765, 1⟩, ⟨19114, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (-1)⟩)

def event121772 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨13825⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9553⟩⟩) ⟨7281⟩ 19084)

def event121773 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13825⟩⟩, .relation 121772 0, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (-1)⟩)

def event121774 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13825⟩⟩, .operator (⟨121765, 0⟩, ⟨19114, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩)

def exact121775RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (-1)⟩]

theorem exact121775RawTermsValid :
    exact121775RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121775 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13825⟩⟩) exact121775RawTerms .large 121768 (.finite 279172874240) (some (121770))

def event121776 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37025⟩⟩) 0 ⟨13825⟩ 121775

def event121777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37025⟩⟩) 1 ⟨37024⟩ 121745

def event121778 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37025⟩⟩) (.sum [.predecessor 0 121776 .coefficient, .predecessor 1 121777 .coefficient])

def event121779 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37025⟩⟩, .operator (⟨121775, 1⟩, ⟨121745, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩)

def event121780 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37025⟩⟩) (.sum [.result 121775 .summary, .result 121745 .summary])

def exact121781RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact121781RawTermsValid :
    exact121781RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121781 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37025⟩⟩) exact121781RawTerms .large 121778 (.finite 279208656896) (some (121780))

def event121782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38896⟩⟩) 0 ⟨37025⟩ 121781

def event121783 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38896⟩⟩) 1 ⟨38895⟩ 121717

def event121784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38896⟩⟩) (.product (.predecessor 0 121782 .coefficient) (.predecessor 1 121783 .coefficient) (⟨false, false, none, none, none⟩))

def event121785 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38896⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨38895⟩⟩]⟩) [⟨.result 121717 .coefficient, false, none⟩])

def event121786 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38896⟩⟩) (.product (.result 121781 .summary) (.transfer 121785) (⟨false, false, none, none, none⟩))

def event121787 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38896⟩⟩, .operator (⟨121781, 1⟩, ⟨121717, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38895⟩⟩]⟩, (-1)⟩)

def event121788 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨38896⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38895⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38895⟩⟩) ⟨38405⟩ 121714)

def event121789 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38896⟩⟩, .relation 121788 0, ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], [⟨.program ⟨257⟩, ⟨38405⟩⟩]⟩, (-1)⟩)

def event121790 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38896⟩⟩, .operator (⟨121781, 0⟩, ⟨121717, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38895⟩⟩]⟩, (1)⟩)

def exact121791RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38895⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], [⟨.program ⟨257⟩, ⟨38405⟩⟩]⟩, (-1)⟩]

theorem exact121791RawTermsValid :
    exact121791RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121791 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38896⟩⟩) exact121791RawTerms .large 121784 (.finite 2997980125321012183040) (some (121786))

def event121792 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37829⟩⟩) 0 ⟨37020⟩ 5433

def event121793 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37829⟩⟩) (.authority (.relationPreimageSource ⟨50⟩))

def exact121794RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨37829⟩⟩]⟩, (1)⟩]

theorem exact121794RawTermsValid :
    exact121794RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121794 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37829⟩⟩) exact121794RawTerms (.finite 5647228698) 121793 .exactZero (none)

def event121795 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37831⟩⟩) 0 ⟨37829⟩ 121794

def event121796 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37831⟩⟩) 1 ⟨2370⟩ 4

def event121797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37831⟩⟩) (.scale (.predecessor 0 121795 .coefficient) (.value (.predecessor 1 121796 .coefficient)))

def exact121798RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨37829⟩⟩]⟩, (1)⟩]

theorem exact121798RawTermsValid :
    exact121798RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121798 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37831⟩⟩) exact121798RawTerms (.finite 5647228698) 121797 .exactZero (none)

def event121799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37832⟩⟩) 0 ⟨5527⟩ 119870

def event121800 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37832⟩⟩) 1 ⟨37831⟩ 121798

def event121801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37832⟩⟩) (.product (.predecessor 0 121799 .coefficient) (.predecessor 1 121800 .coefficient) (⟨false, false, none, none, none⟩))

def event121802 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37832⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨37829⟩⟩]⟩) [⟨.result 121794 .coefficient, false, none⟩])

def event121803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37832⟩⟩) (.product (.result 119870 .summary) (.transfer 121802) (⟨false, false, none, none, none⟩))

def event121804 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37832⟩⟩, .operator (⟨119870, 0⟩, ⟨121798, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37829⟩⟩]⟩, (1)⟩)

def event121805 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨37830⟩⟩)

def event121806 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event121807 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event121808 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2942⟩⟩) (.authority (.operator))

def event121809 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2942⟩⟩) (.finite 12)

def event121810 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event121811 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event121812 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event121813 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event121814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 121813

def event121815 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 121811

def event121816 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 121814 .coefficient) (.value (.predecessor 1 121815 .coefficient)))

def event121817 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event121818 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2944⟩⟩) 0 ⟨392⟩ 121817

def event121819 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2944⟩⟩) 1 ⟨2942⟩ 121809

def event121820 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2944⟩⟩) (.sum [.predecessor 0 121818 .coefficient, .predecessor 1 121819 .coefficient])

def event121821 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2944⟩⟩) (.finite 655352)

def event121822 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5523⟩⟩) 0 ⟨2944⟩ 121821

def event121823 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5523⟩⟩) 1 ⟨5426⟩ 121807

def event121824 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5523⟩⟩) (.identity (.predecessor 1 121823 .coefficient))

def event121825 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5523⟩⟩) (.finite 655360)

def event121826 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37018⟩⟩) 0 ⟨5523⟩ 121825

def event121827 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37018⟩⟩) (.authority (.programFamilyFact))

def exact121828RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37018⟩⟩], []⟩, (1)⟩]

theorem exact121828RawTermsValid :
    exact121828RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121828 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37018⟩⟩) exact121828RawTerms (.finite 42) 121827 .exactZero (none)

def event121829 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13821⟩⟩) 0 ⟨5523⟩ 121825

def event121830 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13821⟩⟩) (.authority (.programFamilyFact))

def exact121831RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13821⟩⟩], []⟩, (1)⟩]

theorem exact121831RawTermsValid :
    exact121831RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121831 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13821⟩⟩) exact121831RawTerms (.finite 42) 121830 .exactZero (none)

def event121832 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37019⟩⟩) 0 ⟨13821⟩ 121831

def event121833 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37019⟩⟩) 1 ⟨37018⟩ 121828

def event121834 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37019⟩⟩) (.product (.predecessor 0 121832 .coefficient) (.predecessor 1 121833 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event121835 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37019⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], []⟩) [⟨.result 121831 .coefficient, true, some 1⟩, ⟨.result 121828 .coefficient, true, some 1⟩])

def event121836 : Event := .survivorFold (1) 121835

def exact121837RawTerms : List Term := []

theorem exact121837RawTermsValid :
    exact121837RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121837 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37019⟩⟩) exact121837RawTerms (.finite 1764) 121834 (.finite 1764) (some (121835))

def event121838 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37020⟩⟩) 0 ⟨37019⟩ 121837

def event121839 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37020⟩⟩) (.identity (.predecessor 0 121838 .coefficient))

def event121840 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37020⟩⟩) (.finite 1764)

def event121841 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37829⟩⟩) 0 ⟨37020⟩ 121840

def event121842 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37829⟩⟩) (.authority (.relationPreimageSource ⟨50⟩))

def exact121843RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨37829⟩⟩]⟩, (1)⟩]

theorem exact121843RawTermsValid :
    exact121843RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121843 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37829⟩⟩) exact121843RawTerms (.finite 5647228698) 121842 .exactZero (none)

def event121844 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact121845RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact121845RawTermsValid :
    exact121845RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121845 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact121845RawTerms .large 121844 .exactZero (none)

def event121846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37830⟩⟩) 0 ⟨35⟩ 121845

def event121847 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37830⟩⟩) 1 ⟨37829⟩ 121843

def event121848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37830⟩⟩) (.product (.predecessor 0 121846 .coefficient) (.predecessor 1 121847 .coefficient) (⟨false, false, none, none, none⟩))

def event121849 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37830⟩⟩, .operator (⟨121845, 0⟩, ⟨121843, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37829⟩⟩]⟩, (1)⟩)

def exact121850RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37829⟩⟩]⟩, (1)⟩]

theorem exact121850RawTermsValid :
    exact121850RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event121850 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37830⟩⟩) exact121850RawTerms .large 121848 .exactZero (none)

def event121851 : Event := .preFoldPolynomial 121850 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37829⟩⟩]⟩, (1)⟩] .exactZero none

def exact121852RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37829⟩⟩]⟩, (1)⟩]

def event121852 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨37830⟩⟩) 121851 exact121852RawTerms .large 121848 .exactZero (none)

def event121853 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨38899⟩⟩)

def event121854 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event121855 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def eventLeaf7600 : Array AnnotatedEvent := #[
  { event := event121600
    frameStart := 121580 },
  { event := event121601
    frameStart := 121580 },
  { event := event121602
    frameStart := 121580 },
  { event := event121603
    frameStart := 121580 },
  { event := event121604
    frameStart := 121580 },
  { event := event121605
    frameStart := 121580 },
  { event := event121606
    frameStart := 121580 },
  { event := event121607
    frameStart := 121580 },
  { event := event121608
    frameStart := 121580 },
  { event := event121609
    frameStart := 121580 },
  { event := event121610
    frameStart := 121580 },
  { event := event121611
    frameStart := 121580 },
  { event := event121612
    frameStart := 121580 },
  { event := event121613
    frameStart := 121580 },
  { event := event121614
    frameStart := 121580 },
  { event := event121615
    frameStart := 121580 }
]

def eventLeaf7601 : Array AnnotatedEvent := #[
  { event := event121616
    frameStart := 121580 },
  { event := event121617
    frameStart := 121580 },
  { event := event121618
    frameStart := 121580 },
  { event := event121619
    frameStart := 121580 },
  { event := event121620
    frameStart := 121580 },
  { event := event121621
    frameStart := 121580 },
  { event := event121622
    frameStart := 121580 },
  { event := event121623
    frameStart := 121580 },
  { event := event121624
    frameStart := 121580 },
  { event := event121625
    frameStart := 121580 },
  { event := event121626
    frameStart := 121580 },
  { event := event121627
    frameStart := 121580 },
  { event := event121628
    frameStart := 121580 },
  { event := event121629
    frameStart := 121580 },
  { event := event121630
    frameStart := 121580 },
  { event := event121631
    frameStart := 121580 }
]

def eventLeaf7602 : Array AnnotatedEvent := #[
  { event := event121632
    frameStart := 121580 },
  { event := event121633
    frameStart := 121580 },
  { event := event121634
    frameStart := 121580 },
  { event := event121635
    frameStart := 121580 },
  { event := event121636
    frameStart := 121580 },
  { event := event121637
    frameStart := 121580 },
  { event := event121638
    frameStart := 121580 },
  { event := event121639
    frameStart := 121580 },
  { event := event121640
    frameStart := 121580 },
  { event := event121641
    frameStart := 121580 },
  { event := event121642
    frameStart := 121580 },
  { event := event121643
    frameStart := 121580 },
  { event := event121644
    frameStart := 121580 },
  { event := event121645
    frameStart := 121580 },
  { event := event121646
    frameStart := 121580 },
  { event := event121647
    frameStart := 121580 }
]

def eventLeaf7603 : Array AnnotatedEvent := #[
  { event := event121648
    frameStart := 121580 },
  { event := event121649
    frameStart := 121580 },
  { event := event121650
    frameStart := 121580 },
  { event := event121651
    frameStart := 121580 },
  { event := event121652
    frameStart := 121580 },
  { event := event121653
    frameStart := 121580 },
  { event := event121654
    frameStart := 121580 },
  { event := event121655
    frameStart := 121580 },
  { event := event121656
    frameStart := 121580 },
  { event := event121657
    frameStart := 121580 },
  { event := event121658
    frameStart := 121580 },
  { event := event121659
    frameStart := 121580 },
  { event := event121660
    frameStart := 121580 },
  { event := event121661
    frameStart := 121580 },
  { event := event121662
    frameStart := 121580 },
  { event := event121663
    frameStart := 121580 }
]

def eventLeaf7604 : Array AnnotatedEvent := #[
  { event := event121664
    frameStart := 121580 },
  { event := event121665
    frameStart := 121580 },
  { event := event121666
    frameStart := 121580 },
  { event := event121667
    frameStart := 121580 },
  { event := event121668
    frameStart := 121580 },
  { event := event121669
    frameStart := 121580 },
  { event := event121670
    frameStart := 121580 },
  { event := event121671
    frameStart := 121580 },
  { event := event121672
    frameStart := 121580 },
  { event := event121673
    frameStart := 121580 },
  { event := event121674
    frameStart := 121580 },
  { event := event121675
    frameStart := 121580 },
  { event := event121676
    frameStart := 121580 },
  { event := event121677
    frameStart := 121580 },
  { event := event121678
    frameStart := 121580 },
  { event := event121679
    frameStart := 121580 }
]

def eventLeaf7605 : Array AnnotatedEvent := #[
  { event := event121680
    frameStart := 121580 },
  { event := event121681
    frameStart := 121580 },
  { event := event121682
    frameStart := 121580 },
  { event := event121683
    frameStart := 121580 },
  { event := event121684
    frameStart := 0 },
  { event := event121685
    frameStart := 0 },
  { event := event121686
    frameStart := 0 },
  { event := event121687
    frameStart := 0 },
  { event := event121688
    frameStart := 0 },
  { event := event121689
    frameStart := 0 },
  { event := event121690
    frameStart := 0 },
  { event := event121691
    frameStart := 0 },
  { event := event121692
    frameStart := 0 },
  { event := event121693
    frameStart := 0 },
  { event := event121694
    frameStart := 0 },
  { event := event121695
    frameStart := 0 }
]

def eventLeaf7606 : Array AnnotatedEvent := #[
  { event := event121696
    frameStart := 0 },
  { event := event121697
    frameStart := 0 },
  { event := event121698
    frameStart := 0 },
  { event := event121699
    frameStart := 0 },
  { event := event121700
    frameStart := 0 },
  { event := event121701
    frameStart := 0 },
  { event := event121702
    frameStart := 0 },
  { event := event121703
    frameStart := 0 },
  { event := event121704
    frameStart := 0 },
  { event := event121705
    frameStart := 0 },
  { event := event121706
    frameStart := 0 },
  { event := event121707
    frameStart := 0 },
  { event := event121708
    frameStart := 0 },
  { event := event121709
    frameStart := 0 },
  { event := event121710
    frameStart := 0 },
  { event := event121711
    frameStart := 0 }
]

def eventLeaf7607 : Array AnnotatedEvent := #[
  { event := event121712
    frameStart := 0 },
  { event := event121713
    frameStart := 0 },
  { event := event121714
    frameStart := 0 },
  { event := event121715
    frameStart := 0 },
  { event := event121716
    frameStart := 0 },
  { event := event121717
    frameStart := 0 },
  { event := event121718
    frameStart := 0 },
  { event := event121719
    frameStart := 0 },
  { event := event121720
    frameStart := 0 },
  { event := event121721
    frameStart := 0 },
  { event := event121722
    frameStart := 0 },
  { event := event121723
    frameStart := 0 },
  { event := event121724
    frameStart := 0 },
  { event := event121725
    frameStart := 0 },
  { event := event121726
    frameStart := 0 },
  { event := event121727
    frameStart := 0 }
]

def eventLeaf7608 : Array AnnotatedEvent := #[
  { event := event121728
    frameStart := 0 },
  { event := event121729
    frameStart := 0 },
  { event := event121730
    frameStart := 0 },
  { event := event121731
    frameStart := 0 },
  { event := event121732
    frameStart := 0 },
  { event := event121733
    frameStart := 0 },
  { event := event121734
    frameStart := 0 },
  { event := event121735
    frameStart := 0 },
  { event := event121736
    frameStart := 0 },
  { event := event121737
    frameStart := 0 },
  { event := event121738
    frameStart := 0 },
  { event := event121739
    frameStart := 0 },
  { event := event121740
    frameStart := 0 },
  { event := event121741
    frameStart := 0 },
  { event := event121742
    frameStart := 0 },
  { event := event121743
    frameStart := 0 }
]

def eventLeaf7609 : Array AnnotatedEvent := #[
  { event := event121744
    frameStart := 0 },
  { event := event121745
    frameStart := 0 },
  { event := event121746
    frameStart := 0 },
  { event := event121747
    frameStart := 0 },
  { event := event121748
    frameStart := 0 },
  { event := event121749
    frameStart := 0 },
  { event := event121750
    frameStart := 0 },
  { event := event121751
    frameStart := 0 },
  { event := event121752
    frameStart := 0 },
  { event := event121753
    frameStart := 0 },
  { event := event121754
    frameStart := 0 },
  { event := event121755
    frameStart := 0 },
  { event := event121756
    frameStart := 0 },
  { event := event121757
    frameStart := 0 },
  { event := event121758
    frameStart := 0 },
  { event := event121759
    frameStart := 0 }
]

def eventLeaf7610 : Array AnnotatedEvent := #[
  { event := event121760
    frameStart := 0 },
  { event := event121761
    frameStart := 0 },
  { event := event121762
    frameStart := 0 },
  { event := event121763
    frameStart := 0 },
  { event := event121764
    frameStart := 0 },
  { event := event121765
    frameStart := 0 },
  { event := event121766
    frameStart := 0 },
  { event := event121767
    frameStart := 0 },
  { event := event121768
    frameStart := 0 },
  { event := event121769
    frameStart := 0 },
  { event := event121770
    frameStart := 0 },
  { event := event121771
    frameStart := 0 },
  { event := event121772
    frameStart := 0 },
  { event := event121773
    frameStart := 0 },
  { event := event121774
    frameStart := 0 },
  { event := event121775
    frameStart := 0 }
]

def eventLeaf7611 : Array AnnotatedEvent := #[
  { event := event121776
    frameStart := 0 },
  { event := event121777
    frameStart := 0 },
  { event := event121778
    frameStart := 0 },
  { event := event121779
    frameStart := 0 },
  { event := event121780
    frameStart := 0 },
  { event := event121781
    frameStart := 0 },
  { event := event121782
    frameStart := 0 },
  { event := event121783
    frameStart := 0 },
  { event := event121784
    frameStart := 0 },
  { event := event121785
    frameStart := 0 },
  { event := event121786
    frameStart := 0 },
  { event := event121787
    frameStart := 0 },
  { event := event121788
    frameStart := 0 },
  { event := event121789
    frameStart := 0 },
  { event := event121790
    frameStart := 0 },
  { event := event121791
    frameStart := 0 }
]

def eventLeaf7612 : Array AnnotatedEvent := #[
  { event := event121792
    frameStart := 0 },
  { event := event121793
    frameStart := 0 },
  { event := event121794
    frameStart := 0 },
  { event := event121795
    frameStart := 0 },
  { event := event121796
    frameStart := 0 },
  { event := event121797
    frameStart := 0 },
  { event := event121798
    frameStart := 0 },
  { event := event121799
    frameStart := 0 },
  { event := event121800
    frameStart := 0 },
  { event := event121801
    frameStart := 0 },
  { event := event121802
    frameStart := 0 },
  { event := event121803
    frameStart := 0 },
  { event := event121804
    frameStart := 0 },
  { event := event121805
    frameStart := 121805 },
  { event := event121806
    frameStart := 121805 },
  { event := event121807
    frameStart := 121805 }
]

def eventLeaf7613 : Array AnnotatedEvent := #[
  { event := event121808
    frameStart := 121805 },
  { event := event121809
    frameStart := 121805 },
  { event := event121810
    frameStart := 121805 },
  { event := event121811
    frameStart := 121805 },
  { event := event121812
    frameStart := 121805 },
  { event := event121813
    frameStart := 121805 },
  { event := event121814
    frameStart := 121805 },
  { event := event121815
    frameStart := 121805 },
  { event := event121816
    frameStart := 121805 },
  { event := event121817
    frameStart := 121805 },
  { event := event121818
    frameStart := 121805 },
  { event := event121819
    frameStart := 121805 },
  { event := event121820
    frameStart := 121805 },
  { event := event121821
    frameStart := 121805 },
  { event := event121822
    frameStart := 121805 },
  { event := event121823
    frameStart := 121805 }
]

def eventLeaf7614 : Array AnnotatedEvent := #[
  { event := event121824
    frameStart := 121805 },
  { event := event121825
    frameStart := 121805 },
  { event := event121826
    frameStart := 121805 },
  { event := event121827
    frameStart := 121805 },
  { event := event121828
    frameStart := 121805 },
  { event := event121829
    frameStart := 121805 },
  { event := event121830
    frameStart := 121805 },
  { event := event121831
    frameStart := 121805 },
  { event := event121832
    frameStart := 121805 },
  { event := event121833
    frameStart := 121805 },
  { event := event121834
    frameStart := 121805 },
  { event := event121835
    frameStart := 121805 },
  { event := event121836
    frameStart := 121805 },
  { event := event121837
    frameStart := 121805 },
  { event := event121838
    frameStart := 121805 },
  { event := event121839
    frameStart := 121805 }
]

def eventLeaf7615 : Array AnnotatedEvent := #[
  { event := event121840
    frameStart := 121805 },
  { event := event121841
    frameStart := 121805 },
  { event := event121842
    frameStart := 121805 },
  { event := event121843
    frameStart := 121805 },
  { event := event121844
    frameStart := 121805 },
  { event := event121845
    frameStart := 121805 },
  { event := event121846
    frameStart := 121805 },
  { event := event121847
    frameStart := 121805 },
  { event := event121848
    frameStart := 121805 },
  { event := event121849
    frameStart := 121805 },
  { event := event121850
    frameStart := 121805 },
  { event := event121851
    frameStart := 121805 },
  { event := event121852
    frameStart := 121805 },
  { event := event121853
    frameStart := 121853 },
  { event := event121854
    frameStart := 121853 },
  { event := event121855
    frameStart := 121853 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events475
