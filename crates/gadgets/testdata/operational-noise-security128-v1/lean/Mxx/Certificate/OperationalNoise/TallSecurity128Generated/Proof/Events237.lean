import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events237

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event60672 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15668⟩⟩) (.identity (.predecessor 0 60671 .coefficient))

def event60673 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15668⟩⟩) (.finite 4)

def event60674 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15852⟩⟩) 0 ⟨15668⟩ 60673

def event60675 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15852⟩⟩) (.authority (.programFamilyFact))

def exact60676RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15852⟩⟩], []⟩, (1)⟩]

theorem exact60676RawTermsValid :
    exact60676RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60676 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15852⟩⟩) exact60676RawTerms (.finite 2) 60675 .exactZero (none)

def event60677 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15853⟩⟩) 0 ⟨15852⟩ 60676

def event60678 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15853⟩⟩) (.identity (.predecessor 0 60677 .coefficient))

def event60679 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15853⟩⟩) (.finite 2)

def event60680 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17071⟩⟩) 0 ⟨15853⟩ 60679

def event60681 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17071⟩⟩) (.authority (.programFamilyFact))

def event60682 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17071⟩⟩) (.finite 3720)

def event60683 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event60684 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17072⟩⟩) 0 ⟨7177⟩ 60683

def event60685 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17072⟩⟩) 1 ⟨17071⟩ 60682

def event60686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17072⟩⟩) (.authority (.operator))

def exact60687RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17072⟩⟩]⟩, (1)⟩]

theorem exact60687RawTermsValid :
    exact60687RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60687 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17072⟩⟩) exact60687RawTerms .large 60686 .exactZero (none)

def event60688 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17978⟩⟩) 0 ⟨17072⟩ 60687

def event60689 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17978⟩⟩) (.authority (.operator))

def exact60690RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17978⟩⟩]⟩, (1)⟩]

theorem exact60690RawTermsValid :
    exact60690RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60690 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17978⟩⟩) exact60690RawTerms (.finite 8192) 60689 .exactZero (none)

def event60691 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event60692 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event60693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17238⟩⟩) 0 ⟨15853⟩ 60679

def event60694 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17238⟩⟩) 1 ⟨136⟩ 60692

def event60695 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17238⟩⟩) (.sum [.predecessor 0 60693 .coefficient, .predecessor 1 60694 .coefficient])

def event60696 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17238⟩⟩) (.finite 2)

def event60697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17239⟩⟩) 0 ⟨17238⟩ 60696

def event60698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17239⟩⟩) (.identity (.predecessor 0 60697 .coefficient))

def exact60699RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15852⟩⟩], []⟩, (1)⟩]

theorem exact60699RawTermsValid :
    exact60699RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60699 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17239⟩⟩) exact60699RawTerms (.finite 2) 60698 .exactZero (none)

def event60700 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact60701RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact60701RawTermsValid :
    exact60701RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60701 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact60701RawTerms .large 60700 .exactZero (none)

def event60702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17240⟩⟩) 0 ⟨6908⟩ 60701

def event60703 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17240⟩⟩) 1 ⟨17239⟩ 60699

def event60704 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17240⟩⟩) (.product (.predecessor 0 60702 .coefficient) (.predecessor 1 60703 .coefficient) (⟨false, false, none, none, none⟩))

def event60705 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17240⟩⟩, .operator (⟨60701, 0⟩, ⟨60699, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact60706RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact60706RawTermsValid :
    exact60706RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60706 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17240⟩⟩) exact60706RawTerms .large 60704 .exactZero (none)

def event60707 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 60683

def event60708 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact60709RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact60709RawTermsValid :
    exact60709RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60709 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact60709RawTerms .large 60708 .exactZero (none)

def event60710 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17241⟩⟩) 0 ⟨7179⟩ 60709

def event60711 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17241⟩⟩) 1 ⟨17240⟩ 60706

def event60712 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17241⟩⟩) (.sum [.predecessor 0 60710 .coefficient, .predecessor 1 60711 .coefficient])

def exact60713RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact60713RawTermsValid :
    exact60713RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60713 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17241⟩⟩) exact60713RawTerms .large 60712 .exactZero (none)

def event60714 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17979⟩⟩) 0 ⟨17241⟩ 60713

def event60715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17979⟩⟩) 1 ⟨17978⟩ 60690

def event60716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17979⟩⟩) (.product (.predecessor 0 60714 .coefficient) (.predecessor 1 60715 .coefficient) (⟨false, false, none, none, none⟩))

def event60717 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17979⟩⟩, .operator (⟨60713, 0⟩, ⟨60690, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17978⟩⟩]⟩, (1)⟩)

def event60718 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17979⟩⟩, .operator (⟨60713, 1⟩, ⟨60690, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17978⟩⟩]⟩, (-1)⟩)

def event60719 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17979⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17978⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17978⟩⟩) ⟨17072⟩ 60687)

def event60720 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17979⟩⟩, .relation 60719 0, ⟨[⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨17072⟩⟩]⟩, (-1)⟩)

def exact60721RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17978⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨17072⟩⟩]⟩, (-1)⟩]

theorem exact60721RawTermsValid :
    exact60721RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60721 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17979⟩⟩) exact60721RawTerms .large 60716 .exactZero (none)

def event60722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16158⟩⟩) 0 ⟨15853⟩ 60679

def event60723 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16158⟩⟩) (.authority (.programFamilyFact))

def exact60724RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16158⟩⟩], []⟩, (1)⟩]

theorem exact60724RawTermsValid :
    exact60724RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60724 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16158⟩⟩) exact60724RawTerms (.finite 2) 60723 .exactZero (none)

def event60725 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16161⟩⟩) 0 ⟨6908⟩ 60701

def event60726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16161⟩⟩) 1 ⟨16158⟩ 60724

def event60727 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16161⟩⟩) (.product (.predecessor 0 60725 .coefficient) (.predecessor 1 60726 .coefficient) (⟨false, true, none, none, some 1⟩))

def event60728 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16161⟩⟩, .operator (⟨60701, 0⟩, ⟨60724, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact60729RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact60729RawTermsValid :
    exact60729RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60729 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16161⟩⟩) exact60729RawTerms .large 60727 .exactZero (none)

def event60730 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7197⟩⟩) 0 ⟨7177⟩ 60683

def event60731 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7197⟩⟩) (.authority (.operator))

def exact60732RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩]

theorem exact60732RawTermsValid :
    exact60732RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60732 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7197⟩⟩) exact60732RawTerms .large 60731 .exactZero (none)

def event60733 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16162⟩⟩) 0 ⟨7197⟩ 60732

def event60734 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16162⟩⟩) 1 ⟨16161⟩ 60729

def event60735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16162⟩⟩) (.sum [.predecessor 0 60733 .coefficient, .predecessor 1 60734 .coefficient])

def exact60736RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact60736RawTermsValid :
    exact60736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60736 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16162⟩⟩) exact60736RawTerms .large 60735 .exactZero (none)

def event60737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17984⟩⟩) 0 ⟨16162⟩ 60736

def event60738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17984⟩⟩) 1 ⟨17979⟩ 60721

def event60739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17984⟩⟩) (.sum [.predecessor 0 60737 .coefficient, .predecessor 1 60738 .coefficient])

def exact60740RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17978⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨17072⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact60740RawTermsValid :
    exact60740RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60740 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17984⟩⟩) exact60740RawTerms .large 60739 .exactZero (none)

def event60741 : Event := .preFoldPolynomial 60740 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17978⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨17072⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact60742RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17978⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨17072⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event60742 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17984⟩⟩) 60741 exact60742RawTerms .large 60739 .exactZero (none)

def event60743 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15853⟩⟩) ⟨⟨76⟩, ⟨56⟩, ⟨135⟩⟩ ⟨60585, 60743⟩

def event60744 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16755⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16752⟩⟩]⟩) (1) 0 2 (.universal 60743 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16752⟩⟩]⟩) (none) 60742)

def event60745 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16755⟩⟩, .relation 60744 1, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩)

def event60746 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16755⟩⟩, .relation 60744 0, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17978⟩⟩]⟩, (-1)⟩)

def event60747 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16755⟩⟩, .relation 60744 2, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨17072⟩⟩]⟩, (1)⟩)

def event60748 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16755⟩⟩, .relation 60744 3, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact60749RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17978⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨17072⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact60749RawTermsValid :
    exact60749RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60749 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16755⟩⟩) exact60749RawTerms .large 60581 (.finite 202072841853861888) (some (60583))

def event60750 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17981⟩⟩) 0 ⟨16755⟩ 60749

def event60751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17981⟩⟩) 1 ⟨17980⟩ 60571

def event60752 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17981⟩⟩) (.sum [.predecessor 0 60750 .coefficient, .predecessor 1 60751 .coefficient])

def event60753 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17981⟩⟩, .operator (⟨60749, 0⟩, ⟨60571, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17978⟩⟩]⟩, (1)⟩)

def event60754 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17981⟩⟩, .operator (⟨60749, 2⟩, ⟨60571, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15852⟩⟩], [⟨.program ⟨257⟩, ⟨17072⟩⟩]⟩, (-1)⟩)

def event60755 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17981⟩⟩) (.sum [.result 60749 .summary, .result 60571 .summary])

def exact60756RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact60756RawTermsValid :
    exact60756RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60756 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17981⟩⟩) exact60756RawTerms .large 60752 (.finite 32188807212483706889510625476608) (some (60755))

def event60757 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17982⟩⟩) 0 ⟨17981⟩ 60756

def event60758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17982⟩⟩) 1 ⟨7172⟩ 15882

def event60759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17982⟩⟩) (.product (.predecessor 0 60757 .coefficient) (.predecessor 1 60758 .coefficient) (⟨false, false, none, none, none⟩))

def event60760 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17982⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) [⟨.result 15878 .coefficient, false, none⟩])

def event60761 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17982⟩⟩) (.product (.result 60756 .summary) (.transfer 60760) (⟨false, false, none, none, none⟩))

def event60762 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17982⟩⟩, .operator (⟨60756, 0⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩)

def event60763 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17982⟩⟩, .operator (⟨60756, 1⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩)

def event60764 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17982⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7171⟩⟩) ⟨7051⟩ 15875)

def event60765 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17982⟩⟩, .relation 60764 0, ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact60766RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩]

theorem exact60766RawTermsValid :
    exact60766RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60766 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17982⟩⟩) exact60766RawTerms .large 60759 (.finite 345624685687166110058245054666339432529920) (some (60761))

def event60767 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11220⟩⟩) 0 ⟨6727⟩ 723

def event60768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11220⟩⟩) 1 ⟨11176⟩ 46653

def event60769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11220⟩⟩) (.tensor (.predecessor 0 60767 .coefficient) (.predecessor 1 60768 .coefficient) true false)

def event60770 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11220⟩⟩, .operator (⟨723, 0⟩, ⟨46653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact60771RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact60771RawTermsValid :
    exact60771RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60771 : Event := .resultExact (⟨.program ⟨257⟩, ⟨11220⟩⟩) exact60771RawTerms .large 60769 .exactZero (none)

def event60772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11198⟩⟩) 0 ⟨11175⟩ 46523

def event60773 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11198⟩⟩) 1 ⟨7292⟩ 15896

def event60774 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11198⟩⟩) (.product (.predecessor 0 60772 .coefficient) (.predecessor 1 60773 .coefficient) (⟨false, false, none, none, none⟩))

def event60775 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11198⟩⟩, .operator (⟨46523, 0⟩, ⟨15896, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩)

def exact60776RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact60776RawTermsValid :
    exact60776RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60776 : Event := .resultExact (⟨.program ⟨257⟩, ⟨11198⟩⟩) exact60776RawTerms .large 60774 .exactZero (none)

def event60777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11221⟩⟩) 0 ⟨11198⟩ 60776

def event60778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11221⟩⟩) 1 ⟨11220⟩ 60771

def event60779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11221⟩⟩) (.sum [.predecessor 0 60777 .coefficient, .predecessor 1 60778 .coefficient])

def exact60780RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact60780RawTermsValid :
    exact60780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60780 : Event := .resultExact (⟨.program ⟨257⟩, ⟨11221⟩⟩) exact60780RawTerms .large 60779 .exactZero (none)

def event60781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11222⟩⟩) 0 ⟨11221⟩ 60780

def event60782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11222⟩⟩) 1 ⟨118⟩ 31516

def event60783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11222⟩⟩) (.sum [.predecessor 0 60781 .coefficient, .predecessor 1 60782 .coefficient])

def event60784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11222⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) [⟨.result 31516 .coefficient, false, none⟩])

def event60785 : Event := .survivorFold (1) 60784

def exact60786RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact60786RawTermsValid :
    exact60786RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60786 : Event := .resultExact (⟨.program ⟨257⟩, ⟨11222⟩⟩) exact60786RawTerms .large 60783 (.finite 26) (some (60784))

def event60787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11223⟩⟩) 0 ⟨11222⟩ 60786

def event60788 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11223⟩⟩) 1 ⟨11222⟩ 60786

def event60789 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11223⟩⟩) (.sum [.predecessor 0 60787 .coefficient, .predecessor 1 60788 .coefficient])

def event60790 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11223⟩⟩, .operator (⟨60786, 0⟩, ⟨60786, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event60791 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11223⟩⟩, .operator (⟨60786, 1⟩, ⟨60786, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (-1)⟩)

def event60792 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11223⟩⟩) (.sum [.result 60786 .summary, .result 60786 .summary])

def exact60793RawTerms : List Term := []

theorem exact60793RawTermsValid :
    exact60793RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60793 : Event := .resultExact (⟨.program ⟨257⟩, ⟨11223⟩⟩) exact60793RawTerms .large 60789 (.finite 52) (some (60792))

def event60794 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17983⟩⟩) 0 ⟨11223⟩ 60793

def event60795 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17983⟩⟩) 1 ⟨17982⟩ 60766

def event60796 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17983⟩⟩) (.sum [.predecessor 0 60794 .coefficient, .predecessor 1 60795 .coefficient])

def event60797 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17983⟩⟩) (.sum [.result 60793 .summary, .result 60766 .summary])

def exact60798RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩]

theorem exact60798RawTermsValid :
    exact60798RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60798 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17983⟩⟩) exact60798RawTerms .large 60796 (.finite 345624685687166110058245054666339432529972) (some (60797))

def event60799 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20898⟩⟩) 0 ⟨17983⟩ 60798

def event60800 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20898⟩⟩) 1 ⟨20897⟩ 60554

def event60801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20898⟩⟩) (.sum [.predecessor 0 60799 .coefficient, .predecessor 1 60800 .coefficient])

def event60802 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20898⟩⟩) (.sum [.result 60798 .summary, .result 60554 .summary])

def exact60803RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩]

theorem exact60803RawTermsValid :
    exact60803RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60803 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20898⟩⟩) exact60803RawTerms .large 60801 (.finite 691250426059631610003352154589745737891892) (some (60802))

def event60804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24118⟩⟩) 0 ⟨20898⟩ 60803

def event60805 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24118⟩⟩) 1 ⟨24117⟩ 60342

def event60806 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24118⟩⟩) (.sum [.predecessor 0 60804 .coefficient, .predecessor 1 60805 .coefficient])

def event60807 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24118⟩⟩) (.sum [.result 60803 .summary, .result 60342 .summary])

def exact60808RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩]

theorem exact60808RawTermsValid :
    exact60808RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60808 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24118⟩⟩) exact60808RawTerms .large 60806 (.finite 1036877221117396499835321299770218916085812) (some (60807))

def event60809 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34138⟩⟩) 0 ⟨24118⟩ 60808

def event60810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34138⟩⟩) 1 ⟨34137⟩ 60130

def event60811 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34138⟩⟩) (.sum [.predecessor 0 60809 .coefficient, .predecessor 1 60810 .coefficient])

def event60812 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34138⟩⟩) (.sum [.result 60808 .summary, .result 60130 .summary])

def exact60813RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩]

theorem exact60813RawTermsValid :
    exact60813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60813 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34138⟩⟩) exact60813RawTerms .large 60811 (.finite 1382506125545760169441014535464825839943732) (some (60812))

def event60814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53198⟩⟩) 0 ⟨34138⟩ 60813

def event60815 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53198⟩⟩) 1 ⟨53197⟩ 59918

def event60816 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53198⟩⟩) (.sum [.predecessor 0 60814 .coefficient, .predecessor 1 60815 .coefficient])

def event60817 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53198⟩⟩) (.sum [.result 60813 .summary, .result 59918 .summary])

def exact60818RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51317⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩]

theorem exact60818RawTermsValid :
    exact60818RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60818 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53198⟩⟩) exact60818RawTerms .large 60816 (.finite 1728139248715321398594155952187700255129652) (some (60817))

def event60819 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56178⟩⟩) 0 ⟨53198⟩ 60818

def event60820 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56178⟩⟩) 1 ⟨56177⟩ 59706

def event60821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56178⟩⟩) (.sum [.predecessor 0 60819 .coefficient, .predecessor 1 60820 .coefficient])

def event60822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56178⟩⟩) (.sum [.result 60818 .summary, .result 59706 .summary])

def exact60823RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54297⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51317⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩]

theorem exact60823RawTermsValid :
    exact60823RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60823 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56178⟩⟩) exact60823RawTerms .large 60821 (.finite 2073774481255481407521021459424708415979572) (some (60822))

def event60824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59158⟩⟩) 0 ⟨56178⟩ 60823

def event60825 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59158⟩⟩) 1 ⟨59157⟩ 59494

def event60826 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59158⟩⟩) (.sum [.predecessor 0 60824 .coefficient, .predecessor 1 60825 .coefficient])

def event60827 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59158⟩⟩) (.sum [.result 60823 .summary, .result 59494 .summary])

def exact60828RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54297⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51317⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩]

theorem exact60828RawTermsValid :
    exact60828RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60828 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59158⟩⟩) exact60828RawTerms .large 60826 (.finite 2419413932536838975995335147689984068157492) (some (60827))

def event60829 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62138⟩⟩) 0 ⟨59158⟩ 60828

def event60830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62138⟩⟩) 1 ⟨62137⟩ 59282

def event60831 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62138⟩⟩) (.sum [.predecessor 0 60829 .coefficient, .predecessor 1 60830 .coefficient])

def event60832 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62138⟩⟩) (.sum [.result 60828 .summary, .result 59282 .summary])

def exact60833RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54297⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51317⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩]

theorem exact60833RawTermsValid :
    exact60833RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60833 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62138⟩⟩) exact60833RawTerms .large 60831 (.finite 2765055493188795324243372926469393465999412) (some (60832))

def event60834 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65118⟩⟩) 0 ⟨62138⟩ 60833

def event60835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65118⟩⟩) 1 ⟨65117⟩ 59070

def event60836 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65118⟩⟩) (.sum [.predecessor 0 60834 .coefficient, .predecessor 1 60835 .coefficient])

def event60837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65118⟩⟩) (.sum [.result 60833 .summary, .result 59070 .summary])

def exact60838RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54297⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51317⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩]

theorem exact60838RawTermsValid :
    exact60838RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60838 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65118⟩⟩) exact60838RawTerms .large 60836 (.finite 3110701272581949232038858886277070355169332) (some (60837))

def event60839 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70799⟩⟩) 0 ⟨65118⟩ 60838

def event60840 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70799⟩⟩) 1 ⟨70798⟩ 58858

def event60841 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70799⟩⟩) (.sum [.predecessor 0 60839 .coefficient, .predecessor 1 60840 .coefficient])

def event60842 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70799⟩⟩) (.sum [.result 60838 .summary, .result 58858 .summary])

def exact60843RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54297⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51317⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩]

theorem exact60843RawTermsValid :
    exact60843RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60843 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70799⟩⟩) exact60843RawTerms .large 60841 (.finite 3456353380086899479155517117627148481331252) (some (60842))

def event60844 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70800⟩⟩) 0 ⟨70799⟩ 60843

def event60845 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70800⟩⟩) 1 ⟨28487⟩ 58646

def event60846 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70800⟩⟩) (.sum [.predecessor 0 60844 .coefficient, .predecessor 1 60845 .coefficient])

def event60847 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70800⟩⟩) (.sum [.result 60843 .summary, .result 58646 .summary])

def exact60848RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54297⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51317⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26726⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩]

theorem exact60848RawTermsValid :
    exact60848RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60848 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70800⟩⟩) exact60848RawTerms .large 60846 (.finite 3802007596962448506045899439491360353157172) (some (60847))

def event60849 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70801⟩⟩) 0 ⟨70800⟩ 60848

def event60850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70801⟩⟩) 1 ⟨31167⟩ 58434

def event60851 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70801⟩⟩) (.sum [.predecessor 0 60849 .coefficient, .predecessor 1 60850 .coefficient])

def event60852 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70801⟩⟩) (.sum [.result 60848 .summary, .result 58434 .summary])

def exact60853RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54297⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51317⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26726⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩]

theorem exact60853RawTermsValid :
    exact60853RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60853 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70801⟩⟩) exact60853RawTerms .large 60851 (.finite 4147668141949793872257454032897973461975092) (some (60852))

def event60854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70802⟩⟩) 0 ⟨70801⟩ 60853

def event60855 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70802⟩⟩) 1 ⟨36827⟩ 58222

def event60856 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70802⟩⟩) (.sum [.predecessor 0 60854 .coefficient, .predecessor 1 60855 .coefficient])

def event60857 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70802⟩⟩) (.sum [.result 60853 .summary, .result 58222 .summary])

def exact60858RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54297⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51317⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35063⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26726⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩]

theorem exact60858RawTermsValid :
    exact60858RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60858 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70802⟩⟩) exact60858RawTerms .large 60856 (.finite 4493332905678336798016456807332854062121012) (some (60857))

def event60859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70803⟩⟩) 0 ⟨70802⟩ 60858

def event60860 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70803⟩⟩) 1 ⟨39507⟩ 58010

def event60861 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70803⟩⟩) (.sum [.predecessor 0 60859 .coefficient, .predecessor 1 60860 .coefficient])

def event60862 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70803⟩⟩) (.sum [.result 60858 .summary, .result 58010 .summary])

def exact60863RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54297⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51317⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35063⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26726⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩]

theorem exact60863RawTermsValid :
    exact60863RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60863 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70803⟩⟩) exact60863RawTerms .large 60861 (.finite 4838999778777478503549183672281868407930932) (some (60862))

def event60864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70804⟩⟩) 0 ⟨70803⟩ 60863

def event60865 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70804⟩⟩) 1 ⟨42187⟩ 57798

def event60866 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70804⟩⟩) (.sum [.predecessor 0 60864 .coefficient, .predecessor 1 60865 .coefficient])

def event60867 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70804⟩⟩) (.sum [.result 60863 .summary, .result 57798 .summary])

def exact60868RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54297⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51317⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40426⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35063⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26726⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩]

theorem exact60868RawTermsValid :
    exact60868RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60868 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70804⟩⟩) exact60868RawTerms .large 60866 (.finite 5184670870617817768629358718259150245068852) (some (60867))

def event60869 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70805⟩⟩) 0 ⟨70804⟩ 60868

def event60870 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70805⟩⟩) 1 ⟨44867⟩ 57586

def event60871 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70805⟩⟩) (.sum [.predecessor 0 60869 .coefficient, .predecessor 1 60870 .coefficient])

def event60872 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70805⟩⟩) (.sum [.result 60868 .summary, .result 57586 .summary])

def exact60873RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54297⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51317⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨43106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40426⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35063⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26726⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩]

theorem exact60873RawTermsValid :
    exact60873RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60873 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70805⟩⟩) exact60873RawTerms .large 60871 (.finite 5530348290569953373030706035778833319198772) (some (60872))

def event60874 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70806⟩⟩) 0 ⟨70805⟩ 60873

def event60875 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70806⟩⟩) 1 ⟨47547⟩ 57374

def event60876 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70806⟩⟩) (.sum [.predecessor 0 60874 .coefficient, .predecessor 1 60875 .coefficient])

def event60877 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70806⟩⟩) (.sum [.result 60873 .summary, .result 57374 .summary])

def exact60878RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54297⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51317⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45783⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨43106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40426⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35063⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26726⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩]

theorem exact60878RawTermsValid :
    exact60878RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60878 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70806⟩⟩) exact60878RawTerms .large 60876 (.finite 5876032038633885316753225624840917630320692) (some (60877))

def event60879 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70807⟩⟩) 0 ⟨70806⟩ 60878

def event60880 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70807⟩⟩) 1 ⟨50227⟩ 57162

def event60881 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70807⟩⟩) (.sum [.predecessor 0 60879 .coefficient, .predecessor 1 60880 .coefficient])

def event60882 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70807⟩⟩) (.sum [.result 60878 .summary, .result 57162 .summary])

def exact60883RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54297⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51317⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48463⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45783⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨43106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40426⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35063⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26726⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩]

theorem exact60883RawTermsValid :
    exact60883RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60883 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70807⟩⟩) exact60883RawTerms .large 60881 (.finite 6221717896068416040249469304417135687106612) (some (60882))

def event60884 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71507⟩⟩) 0 ⟨70807⟩ 60883

def event60885 : Event := .predecessor (⟨.program ⟨257⟩, ⟨71507⟩⟩) 1 ⟨71505⟩ 56950

def event60886 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71507⟩⟩) (.sum [.predecessor 0 60884 .coefficient, .predecessor 1 60885 .coefficient])

def event60887 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨71507⟩⟩) (.sum [.result 60883 .summary, .result 56950 .summary])

def exact60888RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63237⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨60257⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨57277⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨54297⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨51317⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨32253⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨48463⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨45783⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨43106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨40426⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨37743⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨35063⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19013⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨26726⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨67148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (-1)⟩]

theorem exact60888RawTermsValid :
    exact60888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60888 : Event := .resultExact (⟨.program ⟨257⟩, ⟨71507⟩⟩) exact60888RawTerms .large 60886 (.finite 66805187227601152574551644069558752530002096506798132) (some (60887))

def event60889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨28⟩⟩) (.authority (.operator))

def exact60890RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨28⟩⟩]⟩, (1)⟩]

theorem exact60890RawTermsValid :
    exact60890RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60890 : Event := .resultExact (⟨.program ⟨257⟩, ⟨28⟩⟩) exact60890RawTerms (.finite 26) 60889 .exactZero (none)

def event60891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7403⟩⟩) 0 ⟨2377⟩ 27

def event60892 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7403⟩⟩) 1 ⟨7238⟩ 16067

def event60893 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7403⟩⟩) (.product (.predecessor 0 60891 .coefficient) (.predecessor 1 60892 .coefficient) (⟨false, false, none, none, none⟩))

def event60894 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7403⟩⟩, .operator (⟨27, 0⟩, ⟨16067, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7238⟩⟩]⟩, (1)⟩)

def exact60895RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7238⟩⟩]⟩, (1)⟩]

theorem exact60895RawTermsValid :
    exact60895RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60895 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7403⟩⟩) exact60895RawTerms .large 60893 .exactZero (none)

def event60896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11217⟩⟩) 0 ⟨7403⟩ 60895

def event60897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11217⟩⟩) 1 ⟨11176⟩ 46653

def event60898 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11217⟩⟩) (.sum [.predecessor 0 60896 .coefficient, .predecessor 1 60897 .coefficient])

def exact60899RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7238⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact60899RawTermsValid :
    exact60899RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60899 : Event := .resultExact (⟨.program ⟨257⟩, ⟨11217⟩⟩) exact60899RawTerms .large 60898 .exactZero (none)

def event60900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11218⟩⟩) 0 ⟨11217⟩ 60899

def event60901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11218⟩⟩) 1 ⟨28⟩ 60890

def event60902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11218⟩⟩) (.sum [.predecessor 0 60900 .coefficient, .predecessor 1 60901 .coefficient])

def event60903 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11218⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨28⟩⟩]⟩) [⟨.result 60890 .coefficient, false, none⟩])

def event60904 : Event := .survivorFold (1) 60903

def exact60905RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7238⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact60905RawTermsValid :
    exact60905RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event60905 : Event := .resultExact (⟨.program ⟨257⟩, ⟨11218⟩⟩) exact60905RawTerms .large 60902 (.finite 26) (some (60903))

def event60906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11219⟩⟩) 0 ⟨11218⟩ 60905

def event60907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨11219⟩⟩) 1 ⟨9584⟩ 15984

def event60908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11219⟩⟩) (.product (.predecessor 0 60906 .coefficient) (.predecessor 1 60907 .coefficient) (⟨false, false, none, none, none⟩))

def event60909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11219⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) [⟨.result 15980 .coefficient, false, none⟩])

def event60910 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨11219⟩⟩) (.product (.result 60905 .summary) (.transfer 60909) (⟨false, false, none, none, none⟩))

def event60911 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11219⟩⟩, .operator (⟨60905, 1⟩, ⟨15984, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩, (-1)⟩)

def event60912 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨11219⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9583⟩⟩) ⟨9443⟩ 15977)

def event60913 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11219⟩⟩, .relation 60912 18, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩, (1)⟩)

def event60914 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11219⟩⟩, .relation 60912 17, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩, (-1)⟩)

def event60915 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11219⟩⟩, .relation 60912 16, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩, (-1)⟩)

def event60916 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11219⟩⟩, .relation 60912 15, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩, (-1)⟩)

def event60917 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11219⟩⟩, .relation 60912 14, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩, (-1)⟩)

def event60918 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11219⟩⟩, .relation 60912 13, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (-1)⟩)

def event60919 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11219⟩⟩, .relation 60912 12, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (-1)⟩)

def event60920 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11219⟩⟩, .relation 60912 11, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (-1)⟩)

def event60921 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11219⟩⟩, .relation 60912 10, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (-1)⟩)

def event60922 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11219⟩⟩, .relation 60912 9, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (-1)⟩)

def event60923 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11219⟩⟩, .relation 60912 8, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (-1)⟩)

def event60924 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11219⟩⟩, .relation 60912 7, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (-1)⟩)

def event60925 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11219⟩⟩, .relation 60912 6, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (-1)⟩)

def event60926 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11219⟩⟩, .relation 60912 5, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (-1)⟩)

def event60927 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨11219⟩⟩, .relation 60912 4, ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (-1)⟩)

def eventLeaf3792 : Array AnnotatedEvent := #[
  { event := event60672
    frameStart := 60639 },
  { event := event60673
    frameStart := 60639 },
  { event := event60674
    frameStart := 60639 },
  { event := event60675
    frameStart := 60639 },
  { event := event60676
    frameStart := 60639 },
  { event := event60677
    frameStart := 60639 },
  { event := event60678
    frameStart := 60639 },
  { event := event60679
    frameStart := 60639 },
  { event := event60680
    frameStart := 60639 },
  { event := event60681
    frameStart := 60639 },
  { event := event60682
    frameStart := 60639 },
  { event := event60683
    frameStart := 60639 },
  { event := event60684
    frameStart := 60639 },
  { event := event60685
    frameStart := 60639 },
  { event := event60686
    frameStart := 60639 },
  { event := event60687
    frameStart := 60639 }
]

def eventLeaf3793 : Array AnnotatedEvent := #[
  { event := event60688
    frameStart := 60639 },
  { event := event60689
    frameStart := 60639 },
  { event := event60690
    frameStart := 60639 },
  { event := event60691
    frameStart := 60639 },
  { event := event60692
    frameStart := 60639 },
  { event := event60693
    frameStart := 60639 },
  { event := event60694
    frameStart := 60639 },
  { event := event60695
    frameStart := 60639 },
  { event := event60696
    frameStart := 60639 },
  { event := event60697
    frameStart := 60639 },
  { event := event60698
    frameStart := 60639 },
  { event := event60699
    frameStart := 60639 },
  { event := event60700
    frameStart := 60639 },
  { event := event60701
    frameStart := 60639 },
  { event := event60702
    frameStart := 60639 },
  { event := event60703
    frameStart := 60639 }
]

def eventLeaf3794 : Array AnnotatedEvent := #[
  { event := event60704
    frameStart := 60639 },
  { event := event60705
    frameStart := 60639 },
  { event := event60706
    frameStart := 60639 },
  { event := event60707
    frameStart := 60639 },
  { event := event60708
    frameStart := 60639 },
  { event := event60709
    frameStart := 60639 },
  { event := event60710
    frameStart := 60639 },
  { event := event60711
    frameStart := 60639 },
  { event := event60712
    frameStart := 60639 },
  { event := event60713
    frameStart := 60639 },
  { event := event60714
    frameStart := 60639 },
  { event := event60715
    frameStart := 60639 },
  { event := event60716
    frameStart := 60639 },
  { event := event60717
    frameStart := 60639 },
  { event := event60718
    frameStart := 60639 },
  { event := event60719
    frameStart := 60639 }
]

def eventLeaf3795 : Array AnnotatedEvent := #[
  { event := event60720
    frameStart := 60639 },
  { event := event60721
    frameStart := 60639 },
  { event := event60722
    frameStart := 60639 },
  { event := event60723
    frameStart := 60639 },
  { event := event60724
    frameStart := 60639 },
  { event := event60725
    frameStart := 60639 },
  { event := event60726
    frameStart := 60639 },
  { event := event60727
    frameStart := 60639 },
  { event := event60728
    frameStart := 60639 },
  { event := event60729
    frameStart := 60639 },
  { event := event60730
    frameStart := 60639 },
  { event := event60731
    frameStart := 60639 },
  { event := event60732
    frameStart := 60639 },
  { event := event60733
    frameStart := 60639 },
  { event := event60734
    frameStart := 60639 },
  { event := event60735
    frameStart := 60639 }
]

def eventLeaf3796 : Array AnnotatedEvent := #[
  { event := event60736
    frameStart := 60639 },
  { event := event60737
    frameStart := 60639 },
  { event := event60738
    frameStart := 60639 },
  { event := event60739
    frameStart := 60639 },
  { event := event60740
    frameStart := 60639 },
  { event := event60741
    frameStart := 60639 },
  { event := event60742
    frameStart := 60639 },
  { event := event60743
    frameStart := 0 },
  { event := event60744
    frameStart := 0 },
  { event := event60745
    frameStart := 0 },
  { event := event60746
    frameStart := 0 },
  { event := event60747
    frameStart := 0 },
  { event := event60748
    frameStart := 0 },
  { event := event60749
    frameStart := 0 },
  { event := event60750
    frameStart := 0 },
  { event := event60751
    frameStart := 0 }
]

def eventLeaf3797 : Array AnnotatedEvent := #[
  { event := event60752
    frameStart := 0 },
  { event := event60753
    frameStart := 0 },
  { event := event60754
    frameStart := 0 },
  { event := event60755
    frameStart := 0 },
  { event := event60756
    frameStart := 0 },
  { event := event60757
    frameStart := 0 },
  { event := event60758
    frameStart := 0 },
  { event := event60759
    frameStart := 0 },
  { event := event60760
    frameStart := 0 },
  { event := event60761
    frameStart := 0 },
  { event := event60762
    frameStart := 0 },
  { event := event60763
    frameStart := 0 },
  { event := event60764
    frameStart := 0 },
  { event := event60765
    frameStart := 0 },
  { event := event60766
    frameStart := 0 },
  { event := event60767
    frameStart := 0 }
]

def eventLeaf3798 : Array AnnotatedEvent := #[
  { event := event60768
    frameStart := 0 },
  { event := event60769
    frameStart := 0 },
  { event := event60770
    frameStart := 0 },
  { event := event60771
    frameStart := 0 },
  { event := event60772
    frameStart := 0 },
  { event := event60773
    frameStart := 0 },
  { event := event60774
    frameStart := 0 },
  { event := event60775
    frameStart := 0 },
  { event := event60776
    frameStart := 0 },
  { event := event60777
    frameStart := 0 },
  { event := event60778
    frameStart := 0 },
  { event := event60779
    frameStart := 0 },
  { event := event60780
    frameStart := 0 },
  { event := event60781
    frameStart := 0 },
  { event := event60782
    frameStart := 0 },
  { event := event60783
    frameStart := 0 }
]

def eventLeaf3799 : Array AnnotatedEvent := #[
  { event := event60784
    frameStart := 0 },
  { event := event60785
    frameStart := 0 },
  { event := event60786
    frameStart := 0 },
  { event := event60787
    frameStart := 0 },
  { event := event60788
    frameStart := 0 },
  { event := event60789
    frameStart := 0 },
  { event := event60790
    frameStart := 0 },
  { event := event60791
    frameStart := 0 },
  { event := event60792
    frameStart := 0 },
  { event := event60793
    frameStart := 0 },
  { event := event60794
    frameStart := 0 },
  { event := event60795
    frameStart := 0 },
  { event := event60796
    frameStart := 0 },
  { event := event60797
    frameStart := 0 },
  { event := event60798
    frameStart := 0 },
  { event := event60799
    frameStart := 0 }
]

def eventLeaf3800 : Array AnnotatedEvent := #[
  { event := event60800
    frameStart := 0 },
  { event := event60801
    frameStart := 0 },
  { event := event60802
    frameStart := 0 },
  { event := event60803
    frameStart := 0 },
  { event := event60804
    frameStart := 0 },
  { event := event60805
    frameStart := 0 },
  { event := event60806
    frameStart := 0 },
  { event := event60807
    frameStart := 0 },
  { event := event60808
    frameStart := 0 },
  { event := event60809
    frameStart := 0 },
  { event := event60810
    frameStart := 0 },
  { event := event60811
    frameStart := 0 },
  { event := event60812
    frameStart := 0 },
  { event := event60813
    frameStart := 0 },
  { event := event60814
    frameStart := 0 },
  { event := event60815
    frameStart := 0 }
]

def eventLeaf3801 : Array AnnotatedEvent := #[
  { event := event60816
    frameStart := 0 },
  { event := event60817
    frameStart := 0 },
  { event := event60818
    frameStart := 0 },
  { event := event60819
    frameStart := 0 },
  { event := event60820
    frameStart := 0 },
  { event := event60821
    frameStart := 0 },
  { event := event60822
    frameStart := 0 },
  { event := event60823
    frameStart := 0 },
  { event := event60824
    frameStart := 0 },
  { event := event60825
    frameStart := 0 },
  { event := event60826
    frameStart := 0 },
  { event := event60827
    frameStart := 0 },
  { event := event60828
    frameStart := 0 },
  { event := event60829
    frameStart := 0 },
  { event := event60830
    frameStart := 0 },
  { event := event60831
    frameStart := 0 }
]

def eventLeaf3802 : Array AnnotatedEvent := #[
  { event := event60832
    frameStart := 0 },
  { event := event60833
    frameStart := 0 },
  { event := event60834
    frameStart := 0 },
  { event := event60835
    frameStart := 0 },
  { event := event60836
    frameStart := 0 },
  { event := event60837
    frameStart := 0 },
  { event := event60838
    frameStart := 0 },
  { event := event60839
    frameStart := 0 },
  { event := event60840
    frameStart := 0 },
  { event := event60841
    frameStart := 0 },
  { event := event60842
    frameStart := 0 },
  { event := event60843
    frameStart := 0 },
  { event := event60844
    frameStart := 0 },
  { event := event60845
    frameStart := 0 },
  { event := event60846
    frameStart := 0 },
  { event := event60847
    frameStart := 0 }
]

def eventLeaf3803 : Array AnnotatedEvent := #[
  { event := event60848
    frameStart := 0 },
  { event := event60849
    frameStart := 0 },
  { event := event60850
    frameStart := 0 },
  { event := event60851
    frameStart := 0 },
  { event := event60852
    frameStart := 0 },
  { event := event60853
    frameStart := 0 },
  { event := event60854
    frameStart := 0 },
  { event := event60855
    frameStart := 0 },
  { event := event60856
    frameStart := 0 },
  { event := event60857
    frameStart := 0 },
  { event := event60858
    frameStart := 0 },
  { event := event60859
    frameStart := 0 },
  { event := event60860
    frameStart := 0 },
  { event := event60861
    frameStart := 0 },
  { event := event60862
    frameStart := 0 },
  { event := event60863
    frameStart := 0 }
]

def eventLeaf3804 : Array AnnotatedEvent := #[
  { event := event60864
    frameStart := 0 },
  { event := event60865
    frameStart := 0 },
  { event := event60866
    frameStart := 0 },
  { event := event60867
    frameStart := 0 },
  { event := event60868
    frameStart := 0 },
  { event := event60869
    frameStart := 0 },
  { event := event60870
    frameStart := 0 },
  { event := event60871
    frameStart := 0 },
  { event := event60872
    frameStart := 0 },
  { event := event60873
    frameStart := 0 },
  { event := event60874
    frameStart := 0 },
  { event := event60875
    frameStart := 0 },
  { event := event60876
    frameStart := 0 },
  { event := event60877
    frameStart := 0 },
  { event := event60878
    frameStart := 0 },
  { event := event60879
    frameStart := 0 }
]

def eventLeaf3805 : Array AnnotatedEvent := #[
  { event := event60880
    frameStart := 0 },
  { event := event60881
    frameStart := 0 },
  { event := event60882
    frameStart := 0 },
  { event := event60883
    frameStart := 0 },
  { event := event60884
    frameStart := 0 },
  { event := event60885
    frameStart := 0 },
  { event := event60886
    frameStart := 0 },
  { event := event60887
    frameStart := 0 },
  { event := event60888
    frameStart := 0 },
  { event := event60889
    frameStart := 0 },
  { event := event60890
    frameStart := 0 },
  { event := event60891
    frameStart := 0 },
  { event := event60892
    frameStart := 0 },
  { event := event60893
    frameStart := 0 },
  { event := event60894
    frameStart := 0 },
  { event := event60895
    frameStart := 0 }
]

def eventLeaf3806 : Array AnnotatedEvent := #[
  { event := event60896
    frameStart := 0 },
  { event := event60897
    frameStart := 0 },
  { event := event60898
    frameStart := 0 },
  { event := event60899
    frameStart := 0 },
  { event := event60900
    frameStart := 0 },
  { event := event60901
    frameStart := 0 },
  { event := event60902
    frameStart := 0 },
  { event := event60903
    frameStart := 0 },
  { event := event60904
    frameStart := 0 },
  { event := event60905
    frameStart := 0 },
  { event := event60906
    frameStart := 0 },
  { event := event60907
    frameStart := 0 },
  { event := event60908
    frameStart := 0 },
  { event := event60909
    frameStart := 0 },
  { event := event60910
    frameStart := 0 },
  { event := event60911
    frameStart := 0 }
]

def eventLeaf3807 : Array AnnotatedEvent := #[
  { event := event60912
    frameStart := 0 },
  { event := event60913
    frameStart := 0 },
  { event := event60914
    frameStart := 0 },
  { event := event60915
    frameStart := 0 },
  { event := event60916
    frameStart := 0 },
  { event := event60917
    frameStart := 0 },
  { event := event60918
    frameStart := 0 },
  { event := event60919
    frameStart := 0 },
  { event := event60920
    frameStart := 0 },
  { event := event60921
    frameStart := 0 },
  { event := event60922
    frameStart := 0 },
  { event := event60923
    frameStart := 0 },
  { event := event60924
    frameStart := 0 },
  { event := event60925
    frameStart := 0 },
  { event := event60926
    frameStart := 0 },
  { event := event60927
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events237
