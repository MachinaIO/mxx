import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events327

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event83712 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28518⟩⟩, .operator (⟨83708, 0⟩, ⟨83685, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩, (1)⟩)

def event83713 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28518⟩⟩, .operator (⟨83708, 1⟩, ⟨83685, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩, (-1)⟩)

def event83714 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28518⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28517⟩⟩) ⟨24351⟩ 83682)

def event83715 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28518⟩⟩, .relation 83714 0, ⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨24351⟩⟩]⟩, (-1)⟩)

def exact83716RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨24351⟩⟩]⟩, (-1)⟩]

theorem exact83716RawTermsValid :
    exact83716RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83716 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28518⟩⟩) exact83716RawTerms .large 83711 .exactZero (none)

def event83717 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16308⟩⟩) 0 ⟨16263⟩ 83674

def event83718 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16308⟩⟩) (.authority (.programFamilyFact))

def exact83719RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16308⟩⟩], []⟩, (1)⟩]

theorem exact83719RawTermsValid :
    exact83719RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83719 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16308⟩⟩) exact83719RawTerms (.finite 62) 83718 .exactZero (none)

def event83720 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16309⟩⟩) 0 ⟨6544⟩ 83696

def event83721 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16309⟩⟩) 1 ⟨16308⟩ 83719

def event83722 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16309⟩⟩) (.product (.predecessor 0 83720 .coefficient) (.predecessor 1 83721 .coefficient) (⟨false, true, none, none, some 1⟩))

def event83723 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16309⟩⟩, .operator (⟨83696, 0⟩, ⟨83719, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact83724RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact83724RawTermsValid :
    exact83724RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83724 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16309⟩⟩) exact83724RawTerms .large 83722 .exactZero (none)

def event83725 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6729⟩⟩) 0 ⟨6689⟩ 83678

def event83726 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6729⟩⟩) (.authority (.operator))

def exact83727RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩]

theorem exact83727RawTermsValid :
    exact83727RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83727 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6729⟩⟩) exact83727RawTerms .large 83726 .exactZero (none)

def event83728 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16310⟩⟩) 0 ⟨6729⟩ 83727

def event83729 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16310⟩⟩) 1 ⟨16309⟩ 83724

def event83730 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16310⟩⟩) (.sum [.predecessor 0 83728 .coefficient, .predecessor 1 83729 .coefficient])

def exact83731RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83731RawTermsValid :
    exact83731RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83731 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16310⟩⟩) exact83731RawTerms .large 83730 .exactZero (none)

def event83732 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28522⟩⟩) 0 ⟨16310⟩ 83731

def event83733 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28522⟩⟩) 1 ⟨28518⟩ 83716

def event83734 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28522⟩⟩) (.sum [.predecessor 0 83732 .coefficient, .predecessor 1 83733 .coefficient])

def exact83735RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨24351⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83735RawTermsValid :
    exact83735RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83735 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28522⟩⟩) exact83735RawTerms .large 83734 .exactZero (none)

def event83736 : Event := .preFoldPolynomial 83735 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨24351⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact83737RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨24351⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event83737 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28522⟩⟩) 83736 exact83737RawTerms .large 83734 .exactZero (none)

def event83738 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16263⟩⟩) ⟨⟨142⟩, ⟨50⟩, ⟨109⟩⟩ ⟨83580, 83738⟩

def event83739 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21835⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21832⟩⟩]⟩) (1) 0 2 (.universal 83738 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21832⟩⟩]⟩) (none) 83737)

def event83740 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21835⟩⟩, .relation 83739 1, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩)

def event83741 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21835⟩⟩, .relation 83739 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩, (-1)⟩)

def event83742 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21835⟩⟩, .relation 83739 2, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨24351⟩⟩]⟩, (1)⟩)

def event83743 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21835⟩⟩, .relation 83739 3, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact83744RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨24351⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83744RawTermsValid :
    exact83744RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83744 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21835⟩⟩) exact83744RawTerms .large 83576 (.finite 1811303510016) (some (83578))

def event83745 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28520⟩⟩) 0 ⟨21835⟩ 83744

def event83746 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28520⟩⟩) 1 ⟨28519⟩ 83566

def event83747 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28520⟩⟩) (.sum [.predecessor 0 83745 .coefficient, .predecessor 1 83746 .coefficient])

def event83748 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28520⟩⟩, .operator (⟨83744, 0⟩, ⟨83566, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩, (1)⟩)

def event83749 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28520⟩⟩, .operator (⟨83744, 2⟩, ⟨83566, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨24351⟩⟩]⟩, (-1)⟩)

def event83750 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28520⟩⟩) (.sum [.result 83744 .summary, .result 83566 .summary])

def exact83751RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83751RawTermsValid :
    exact83751RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83751 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28520⟩⟩) exact83751RawTerms .large 83747 (.finite 1292202948609709846528) (some (83750))

def event83752 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24286⟩⟩) 0 ⟨16179⟩ 4029

def event83753 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24286⟩⟩) (.authority (.programFamilyFact))

def event83754 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24286⟩⟩) (.finite 3720)

def event83755 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24288⟩⟩) 0 ⟨6689⟩ 5477

def event83756 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24288⟩⟩) 1 ⟨24286⟩ 83754

def event83757 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24288⟩⟩) (.authority (.operator))

def exact83758RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24288⟩⟩]⟩, (1)⟩]

theorem exact83758RawTermsValid :
    exact83758RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83758 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24288⟩⟩) exact83758RawTerms .large 83757 .exactZero (none)

def event83759 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28300⟩⟩) 0 ⟨24288⟩ 83758

def event83760 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28300⟩⟩) (.authority (.operator))

def exact83761RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩, (1)⟩]

theorem exact83761RawTermsValid :
    exact83761RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83761 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28300⟩⟩) exact83761RawTerms (.finite 8192) 83760 .exactZero (none)

def event83762 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23667⟩⟩) 0 ⟨14643⟩ 4023

def event83763 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23667⟩⟩) (.authority (.programFamilyFact))

def event83764 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23667⟩⟩) (.finite 3720)

def event83765 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23668⟩⟩) 0 ⟨6689⟩ 5477

def event83766 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23668⟩⟩) 1 ⟨23667⟩ 83764

def event83767 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23668⟩⟩) (.authority (.operator))

def exact83768RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23668⟩⟩]⟩, (1)⟩]

theorem exact83768RawTermsValid :
    exact83768RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83768 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23668⟩⟩) exact83768RawTerms .large 83767 .exactZero (none)

def event83769 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26220⟩⟩) 0 ⟨23668⟩ 83768

def event83770 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26220⟩⟩) (.authority (.operator))

def exact83771RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩, (1)⟩]

theorem exact83771RawTermsValid :
    exact83771RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83771 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26220⟩⟩) exact83771RawTerms (.finite 8192) 83770 .exactZero (none)

def event83772 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11638⟩⟩) 0 ⟨11637⟩ 4012

def event83773 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11638⟩⟩) 1 ⟨6567⟩ 79920

def event83774 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11638⟩⟩) (.tensor (.predecessor 0 83772 .coefficient) (.predecessor 1 83773 .coefficient) true false)

def event83775 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11638⟩⟩, .operator (⟨4012, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact83776RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact83776RawTermsValid :
    exact83776RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83776 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11638⟩⟩) exact83776RawTerms .large 83774 .exactZero (none)

def event83777 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7237⟩⟩) 0 ⟨5539⟩ 79790

def event83778 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7237⟩⟩) 1 ⟨6781⟩ 10480

def event83779 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7237⟩⟩) (.product (.predecessor 0 83777 .coefficient) (.predecessor 1 83778 .coefficient) (⟨false, false, none, none, none⟩))

def event83780 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7237⟩⟩, .operator (⟨79790, 0⟩, ⟨10480, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩)

def exact83781RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩]

theorem exact83781RawTermsValid :
    exact83781RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83781 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7237⟩⟩) exact83781RawTerms .large 83779 .exactZero (none)

def event83782 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11639⟩⟩) 0 ⟨7237⟩ 83781

def event83783 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11639⟩⟩) 1 ⟨11638⟩ 83776

def event83784 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11639⟩⟩) (.sum [.predecessor 0 83782 .coefficient, .predecessor 1 83783 .coefficient])

def exact83785RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83785RawTermsValid :
    exact83785RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83785 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11639⟩⟩) exact83785RawTerms .large 83784 .exactZero (none)

def event83786 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11640⟩⟩) 0 ⟨11639⟩ 83785

def event83787 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11640⟩⟩) 1 ⟨95⟩ 10472

def event83788 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11640⟩⟩) (.sum [.predecessor 0 83786 .coefficient, .predecessor 1 83787 .coefficient])

def event83789 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11640⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨95⟩⟩]⟩) [⟨.result 10472 .coefficient, false, none⟩])

def event83790 : Event := .survivorFold (1) 83789

def exact83791RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83791RawTermsValid :
    exact83791RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83791 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11640⟩⟩) exact83791RawTerms .large 83788 (.finite 26) (some (83789))

def event83792 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14644⟩⟩) 0 ⟨11640⟩ 83791

def event83793 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14644⟩⟩) 1 ⟨14641⟩ 4015

def event83794 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14644⟩⟩) (.product (.predecessor 0 83792 .coefficient) (.predecessor 1 83793 .coefficient) (⟨false, true, none, none, some 1⟩))

def event83795 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14644⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨14641⟩⟩], []⟩) [⟨.result 4015 .coefficient, true, some 1⟩])

def event83796 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14644⟩⟩) (.product (.result 83791 .summary) (.transfer 83795) (⟨false, false, none, none, none⟩))

def event83797 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14644⟩⟩, .operator (⟨83791, 1⟩, ⟨4015, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event83798 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14644⟩⟩, .operator (⟨83791, 0⟩, ⟨4015, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩)

def exact83799RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩]

theorem exact83799RawTermsValid :
    exact83799RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83799 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14644⟩⟩) exact83799RawTerms .large 83794 (.finite 23296) (some (83796))

def event83800 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14645⟩⟩) 0 ⟨14641⟩ 4015

def event83801 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14645⟩⟩) 1 ⟨6567⟩ 79920

def event83802 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14645⟩⟩) (.tensor (.predecessor 0 83800 .coefficient) (.predecessor 1 83801 .coefficient) true false)

def event83803 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14645⟩⟩, .operator (⟨4015, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact83804RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact83804RawTermsValid :
    exact83804RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83804 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14645⟩⟩) exact83804RawTerms .large 83802 .exactZero (none)

def event83805 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7218⟩⟩) 0 ⟨5539⟩ 79790

def event83806 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7218⟩⟩) 1 ⟨6762⟩ 10521

def event83807 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7218⟩⟩) (.product (.predecessor 0 83805 .coefficient) (.predecessor 1 83806 .coefficient) (⟨false, false, none, none, none⟩))

def event83808 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7218⟩⟩, .operator (⟨79790, 0⟩, ⟨10521, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩]⟩, (1)⟩)

def exact83809RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩]⟩, (1)⟩]

theorem exact83809RawTermsValid :
    exact83809RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83809 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7218⟩⟩) exact83809RawTerms .large 83807 .exactZero (none)

def event83810 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14646⟩⟩) 0 ⟨7218⟩ 83809

def event83811 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14646⟩⟩) 1 ⟨14645⟩ 83804

def event83812 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14646⟩⟩) (.sum [.predecessor 0 83810 .coefficient, .predecessor 1 83811 .coefficient])

def exact83813RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83813RawTermsValid :
    exact83813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83813 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14646⟩⟩) exact83813RawTerms .large 83812 .exactZero (none)

def event83814 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14647⟩⟩) 0 ⟨14646⟩ 83813

def event83815 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14647⟩⟩) 1 ⟨76⟩ 10513

def event83816 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14647⟩⟩) (.sum [.predecessor 0 83814 .coefficient, .predecessor 1 83815 .coefficient])

def event83817 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14647⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨76⟩⟩]⟩) [⟨.result 10513 .coefficient, false, none⟩])

def event83818 : Event := .survivorFold (1) 83817

def exact83819RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83819RawTermsValid :
    exact83819RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83819 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14647⟩⟩) exact83819RawTerms .large 83816 (.finite 26) (some (83817))

def event83820 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14648⟩⟩) 0 ⟨14647⟩ 83819

def event83821 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14648⟩⟩) 1 ⟨7859⟩ 10510

def event83822 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14648⟩⟩) (.product (.predecessor 0 83820 .coefficient) (.predecessor 1 83821 .coefficient) (⟨false, false, none, none, none⟩))

def event83823 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14648⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩) [⟨.result 10506 .coefficient, false, none⟩])

def event83824 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14648⟩⟩) (.product (.result 83819 .summary) (.transfer 83823) (⟨false, false, none, none, none⟩))

def event83825 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14648⟩⟩, .operator (⟨83819, 1⟩, ⟨10510, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (-1)⟩)

def event83826 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨14648⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7858⟩⟩) ⟨6781⟩ 10480)

def event83827 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14648⟩⟩, .relation 83826 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (-1)⟩)

def event83828 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14648⟩⟩, .operator (⟨83819, 0⟩, ⟨10510, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩)

def exact83829RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (-1)⟩]

theorem exact83829RawTermsValid :
    exact83829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83829 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14648⟩⟩) exact83829RawTerms .large 83822 (.finite 95420416) (some (83824))

def event83830 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14649⟩⟩) 0 ⟨14648⟩ 83829

def event83831 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14649⟩⟩) 1 ⟨14644⟩ 83799

def event83832 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14649⟩⟩) (.sum [.predecessor 0 83830 .coefficient, .predecessor 1 83831 .coefficient])

def event83833 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14649⟩⟩, .operator (⟨83829, 1⟩, ⟨83799, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6781⟩⟩]⟩, (1)⟩)

def event83834 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14649⟩⟩) (.sum [.result 83829 .summary, .result 83799 .summary])

def exact83835RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact83835RawTermsValid :
    exact83835RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83835 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14649⟩⟩) exact83835RawTerms .large 83832 (.finite 95443712) (some (83834))

def event83836 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26221⟩⟩) 0 ⟨14649⟩ 83835

def event83837 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26221⟩⟩) 1 ⟨26220⟩ 83771

def event83838 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26221⟩⟩) (.product (.predecessor 0 83836 .coefficient) (.predecessor 1 83837 .coefficient) (⟨false, false, none, none, none⟩))

def event83839 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26221⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩) [⟨.result 83771 .coefficient, false, none⟩])

def event83840 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26221⟩⟩) (.product (.result 83835 .summary) (.transfer 83839) (⟨false, false, none, none, none⟩))

def event83841 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26221⟩⟩, .operator (⟨83835, 1⟩, ⟨83771, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩, (-1)⟩)

def event83842 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26221⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26220⟩⟩) ⟨23668⟩ 83768)

def event83843 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26221⟩⟩, .relation 83842 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨23668⟩⟩]⟩, (-1)⟩)

def event83844 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26221⟩⟩, .operator (⟨83835, 0⟩, ⟨83771, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩, (1)⟩)

def exact83845RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨23668⟩⟩]⟩, (-1)⟩]

theorem exact83845RawTermsValid :
    exact83845RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83845 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26221⟩⟩) exact83845RawTerms .large 83838 (.finite 350279950139392) (some (83840))

def event83846 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19672⟩⟩) 0 ⟨14643⟩ 4023

def event83847 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19672⟩⟩) (.authority (.relationPreimageSource ⟨17⟩))

def exact83848RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19672⟩⟩]⟩, (1)⟩]

theorem exact83848RawTermsValid :
    exact83848RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83848 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19672⟩⟩) exact83848RawTerms (.finite 136065468) 83847 .exactZero (none)

def event83849 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19674⟩⟩) 0 ⟨19672⟩ 83848

def event83850 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19674⟩⟩) 1 ⟨2348⟩ 4

def event83851 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19674⟩⟩) (.scale (.predecessor 0 83849 .coefficient) (.value (.predecessor 1 83850 .coefficient)))

def exact83852RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19672⟩⟩]⟩, (1)⟩]

theorem exact83852RawTermsValid :
    exact83852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83852 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19674⟩⟩) exact83852RawTerms (.finite 136065468) 83851 .exactZero (none)

def event83853 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19675⟩⟩) 0 ⟨5541⟩ 80012

def event83854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19675⟩⟩) 1 ⟨19674⟩ 83852

def event83855 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19675⟩⟩) (.product (.predecessor 0 83853 .coefficient) (.predecessor 1 83854 .coefficient) (⟨false, false, none, none, none⟩))

def event83856 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19675⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19672⟩⟩]⟩) [⟨.result 83848 .coefficient, false, none⟩])

def event83857 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19675⟩⟩) (.product (.result 80012 .summary) (.transfer 83856) (⟨false, false, none, none, none⟩))

def event83858 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19675⟩⟩, .operator (⟨80012, 0⟩, ⟨83852, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19672⟩⟩]⟩, (1)⟩)

def event83859 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19673⟩⟩)

def event83860 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event83861 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event83862 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event83863 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event83864 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event83865 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event83866 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event83867 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event83868 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 83867

def event83869 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 83865

def event83870 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 83868 .coefficient) (.value (.predecessor 1 83869 .coefficient)))

def event83871 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event83872 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 83871

def event83873 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 83863

def event83874 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 83872 .coefficient, .predecessor 1 83873 .coefficient])

def event83875 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event83876 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 83875

def event83877 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 83861

def event83878 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 83877 .coefficient))

def event83879 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event83880 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11637⟩⟩) 0 ⟨5536⟩ 83879

def event83881 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11637⟩⟩) (.authority (.programFamilyFact))

def exact83882RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩], []⟩, (1)⟩]

theorem exact83882RawTermsValid :
    exact83882RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83882 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11637⟩⟩) exact83882RawTerms (.finite 28) 83881 .exactZero (none)

def event83883 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14641⟩⟩) 0 ⟨5536⟩ 83879

def event83884 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14641⟩⟩) (.authority (.programFamilyFact))

def exact83885RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14641⟩⟩], []⟩, (1)⟩]

theorem exact83885RawTermsValid :
    exact83885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83885 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14641⟩⟩) exact83885RawTerms (.finite 28) 83884 .exactZero (none)

def event83886 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14642⟩⟩) 0 ⟨14641⟩ 83885

def event83887 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14642⟩⟩) 1 ⟨11637⟩ 83882

def event83888 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14642⟩⟩) (.product (.predecessor 0 83886 .coefficient) (.predecessor 1 83887 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event83889 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14642⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], []⟩) [⟨.result 83885 .coefficient, true, some 1⟩, ⟨.result 83882 .coefficient, true, some 1⟩])

def event83890 : Event := .survivorFold (1) 83889

def exact83891RawTerms : List Term := []

theorem exact83891RawTermsValid :
    exact83891RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83891 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14642⟩⟩) exact83891RawTerms (.finite 784) 83888 (.finite 784) (some (83889))

def event83892 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14643⟩⟩) 0 ⟨14642⟩ 83891

def event83893 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14643⟩⟩) (.identity (.predecessor 0 83892 .coefficient))

def event83894 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14643⟩⟩) (.finite 784)

def event83895 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19672⟩⟩) 0 ⟨14643⟩ 83894

def event83896 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19672⟩⟩) (.authority (.relationPreimageSource ⟨17⟩))

def exact83897RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19672⟩⟩]⟩, (1)⟩]

theorem exact83897RawTermsValid :
    exact83897RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83897 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19672⟩⟩) exact83897RawTerms (.finite 136065468) 83896 .exactZero (none)

def event83898 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact83899RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact83899RawTermsValid :
    exact83899RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83899 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact83899RawTerms .large 83898 .exactZero (none)

def event83900 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19673⟩⟩) 0 ⟨6⟩ 83899

def event83901 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19673⟩⟩) 1 ⟨19672⟩ 83897

def event83902 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19673⟩⟩) (.product (.predecessor 0 83900 .coefficient) (.predecessor 1 83901 .coefficient) (⟨false, false, none, none, none⟩))

def event83903 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19673⟩⟩, .operator (⟨83899, 0⟩, ⟨83897, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19672⟩⟩]⟩, (1)⟩)

def exact83904RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19672⟩⟩]⟩, (1)⟩]

theorem exact83904RawTermsValid :
    exact83904RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83904 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19673⟩⟩) exact83904RawTerms .large 83902 .exactZero (none)

def event83905 : Event := .preFoldPolynomial 83904 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19672⟩⟩]⟩, (1)⟩] .exactZero none

def exact83906RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19672⟩⟩]⟩, (1)⟩]

def event83906 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19673⟩⟩) 83905 exact83906RawTerms .large 83902 .exactZero (none)

def event83907 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26224⟩⟩)

def event83908 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event83909 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event83910 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event83911 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event83912 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event83913 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event83914 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event83915 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event83916 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 83915

def event83917 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 83913

def event83918 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 83916 .coefficient) (.value (.predecessor 1 83917 .coefficient)))

def event83919 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event83920 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 83919

def event83921 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 83911

def event83922 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 83920 .coefficient, .predecessor 1 83921 .coefficient])

def event83923 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event83924 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 83923

def event83925 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 83909

def event83926 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 83925 .coefficient))

def event83927 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event83928 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11637⟩⟩) 0 ⟨5536⟩ 83927

def event83929 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11637⟩⟩) (.authority (.programFamilyFact))

def exact83930RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩], []⟩, (1)⟩]

theorem exact83930RawTermsValid :
    exact83930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83930 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11637⟩⟩) exact83930RawTerms (.finite 28) 83929 .exactZero (none)

def event83931 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14641⟩⟩) 0 ⟨5536⟩ 83927

def event83932 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14641⟩⟩) (.authority (.programFamilyFact))

def exact83933RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14641⟩⟩], []⟩, (1)⟩]

theorem exact83933RawTermsValid :
    exact83933RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83933 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14641⟩⟩) exact83933RawTerms (.finite 28) 83932 .exactZero (none)

def event83934 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14642⟩⟩) 0 ⟨14641⟩ 83933

def event83935 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14642⟩⟩) 1 ⟨11637⟩ 83930

def event83936 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14642⟩⟩) (.product (.predecessor 0 83934 .coefficient) (.predecessor 1 83935 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event83937 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14642⟩⟩, .operator (⟨83933, 0⟩, ⟨83930, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], []⟩, (1)⟩)

def exact83938RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], []⟩, (1)⟩]

theorem exact83938RawTermsValid :
    exact83938RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83938 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14642⟩⟩) exact83938RawTerms (.finite 784) 83936 .exactZero (none)

def event83939 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14643⟩⟩) 0 ⟨14642⟩ 83938

def event83940 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14643⟩⟩) (.identity (.predecessor 0 83939 .coefficient))

def event83941 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14643⟩⟩) (.finite 784)

def event83942 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23667⟩⟩) 0 ⟨14643⟩ 83941

def event83943 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23667⟩⟩) (.authority (.programFamilyFact))

def event83944 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23667⟩⟩) (.finite 3720)

def event83945 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event83946 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23668⟩⟩) 0 ⟨6689⟩ 83945

def event83947 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23668⟩⟩) 1 ⟨23667⟩ 83944

def event83948 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23668⟩⟩) (.authority (.operator))

def exact83949RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23668⟩⟩]⟩, (1)⟩]

theorem exact83949RawTermsValid :
    exact83949RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83949 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23668⟩⟩) exact83949RawTerms .large 83948 .exactZero (none)

def event83950 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26220⟩⟩) 0 ⟨23668⟩ 83949

def event83951 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26220⟩⟩) (.authority (.operator))

def exact83952RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26220⟩⟩]⟩, (1)⟩]

theorem exact83952RawTermsValid :
    exact83952RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83952 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26220⟩⟩) exact83952RawTerms (.finite 8192) 83951 .exactZero (none)

def event83953 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event83954 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event83955 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14748⟩⟩) 0 ⟨14643⟩ 83941

def event83956 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14748⟩⟩) 1 ⟨110⟩ 83954

def event83957 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14748⟩⟩) (.sum [.predecessor 0 83955 .coefficient, .predecessor 1 83956 .coefficient])

def event83958 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14748⟩⟩) (.finite 784)

def event83959 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14749⟩⟩) 0 ⟨14748⟩ 83958

def event83960 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14749⟩⟩) (.identity (.predecessor 0 83959 .coefficient))

def exact83961RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], []⟩, (1)⟩]

theorem exact83961RawTermsValid :
    exact83961RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83961 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14749⟩⟩) exact83961RawTerms (.finite 784) 83960 .exactZero (none)

def event83962 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact83963RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact83963RawTermsValid :
    exact83963RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83963 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact83963RawTerms .large 83962 .exactZero (none)

def event83964 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14750⟩⟩) 0 ⟨6544⟩ 83963

def event83965 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14750⟩⟩) 1 ⟨14749⟩ 83961

def event83966 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14750⟩⟩) (.product (.predecessor 0 83964 .coefficient) (.predecessor 1 83965 .coefficient) (⟨false, false, none, none, none⟩))

def event83967 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14750⟩⟩, .operator (⟨83963, 0⟩, ⟨83961, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11637⟩⟩, ⟨.program ⟨214⟩, ⟨14641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def eventLeaf5232 : Array AnnotatedEvent := #[
  { event := event83712
    frameStart := 83634 },
  { event := event83713
    frameStart := 83634 },
  { event := event83714
    frameStart := 83634 },
  { event := event83715
    frameStart := 83634 },
  { event := event83716
    frameStart := 83634 },
  { event := event83717
    frameStart := 83634 },
  { event := event83718
    frameStart := 83634 },
  { event := event83719
    frameStart := 83634 },
  { event := event83720
    frameStart := 83634 },
  { event := event83721
    frameStart := 83634 },
  { event := event83722
    frameStart := 83634 },
  { event := event83723
    frameStart := 83634 },
  { event := event83724
    frameStart := 83634 },
  { event := event83725
    frameStart := 83634 },
  { event := event83726
    frameStart := 83634 },
  { event := event83727
    frameStart := 83634 }
]

def eventLeaf5233 : Array AnnotatedEvent := #[
  { event := event83728
    frameStart := 83634 },
  { event := event83729
    frameStart := 83634 },
  { event := event83730
    frameStart := 83634 },
  { event := event83731
    frameStart := 83634 },
  { event := event83732
    frameStart := 83634 },
  { event := event83733
    frameStart := 83634 },
  { event := event83734
    frameStart := 83634 },
  { event := event83735
    frameStart := 83634 },
  { event := event83736
    frameStart := 83634 },
  { event := event83737
    frameStart := 83634 },
  { event := event83738
    frameStart := 0 },
  { event := event83739
    frameStart := 0 },
  { event := event83740
    frameStart := 0 },
  { event := event83741
    frameStart := 0 },
  { event := event83742
    frameStart := 0 },
  { event := event83743
    frameStart := 0 }
]

def eventLeaf5234 : Array AnnotatedEvent := #[
  { event := event83744
    frameStart := 0 },
  { event := event83745
    frameStart := 0 },
  { event := event83746
    frameStart := 0 },
  { event := event83747
    frameStart := 0 },
  { event := event83748
    frameStart := 0 },
  { event := event83749
    frameStart := 0 },
  { event := event83750
    frameStart := 0 },
  { event := event83751
    frameStart := 0 },
  { event := event83752
    frameStart := 0 },
  { event := event83753
    frameStart := 0 },
  { event := event83754
    frameStart := 0 },
  { event := event83755
    frameStart := 0 },
  { event := event83756
    frameStart := 0 },
  { event := event83757
    frameStart := 0 },
  { event := event83758
    frameStart := 0 },
  { event := event83759
    frameStart := 0 }
]

def eventLeaf5235 : Array AnnotatedEvent := #[
  { event := event83760
    frameStart := 0 },
  { event := event83761
    frameStart := 0 },
  { event := event83762
    frameStart := 0 },
  { event := event83763
    frameStart := 0 },
  { event := event83764
    frameStart := 0 },
  { event := event83765
    frameStart := 0 },
  { event := event83766
    frameStart := 0 },
  { event := event83767
    frameStart := 0 },
  { event := event83768
    frameStart := 0 },
  { event := event83769
    frameStart := 0 },
  { event := event83770
    frameStart := 0 },
  { event := event83771
    frameStart := 0 },
  { event := event83772
    frameStart := 0 },
  { event := event83773
    frameStart := 0 },
  { event := event83774
    frameStart := 0 },
  { event := event83775
    frameStart := 0 }
]

def eventLeaf5236 : Array AnnotatedEvent := #[
  { event := event83776
    frameStart := 0 },
  { event := event83777
    frameStart := 0 },
  { event := event83778
    frameStart := 0 },
  { event := event83779
    frameStart := 0 },
  { event := event83780
    frameStart := 0 },
  { event := event83781
    frameStart := 0 },
  { event := event83782
    frameStart := 0 },
  { event := event83783
    frameStart := 0 },
  { event := event83784
    frameStart := 0 },
  { event := event83785
    frameStart := 0 },
  { event := event83786
    frameStart := 0 },
  { event := event83787
    frameStart := 0 },
  { event := event83788
    frameStart := 0 },
  { event := event83789
    frameStart := 0 },
  { event := event83790
    frameStart := 0 },
  { event := event83791
    frameStart := 0 }
]

def eventLeaf5237 : Array AnnotatedEvent := #[
  { event := event83792
    frameStart := 0 },
  { event := event83793
    frameStart := 0 },
  { event := event83794
    frameStart := 0 },
  { event := event83795
    frameStart := 0 },
  { event := event83796
    frameStart := 0 },
  { event := event83797
    frameStart := 0 },
  { event := event83798
    frameStart := 0 },
  { event := event83799
    frameStart := 0 },
  { event := event83800
    frameStart := 0 },
  { event := event83801
    frameStart := 0 },
  { event := event83802
    frameStart := 0 },
  { event := event83803
    frameStart := 0 },
  { event := event83804
    frameStart := 0 },
  { event := event83805
    frameStart := 0 },
  { event := event83806
    frameStart := 0 },
  { event := event83807
    frameStart := 0 }
]

def eventLeaf5238 : Array AnnotatedEvent := #[
  { event := event83808
    frameStart := 0 },
  { event := event83809
    frameStart := 0 },
  { event := event83810
    frameStart := 0 },
  { event := event83811
    frameStart := 0 },
  { event := event83812
    frameStart := 0 },
  { event := event83813
    frameStart := 0 },
  { event := event83814
    frameStart := 0 },
  { event := event83815
    frameStart := 0 },
  { event := event83816
    frameStart := 0 },
  { event := event83817
    frameStart := 0 },
  { event := event83818
    frameStart := 0 },
  { event := event83819
    frameStart := 0 },
  { event := event83820
    frameStart := 0 },
  { event := event83821
    frameStart := 0 },
  { event := event83822
    frameStart := 0 },
  { event := event83823
    frameStart := 0 }
]

def eventLeaf5239 : Array AnnotatedEvent := #[
  { event := event83824
    frameStart := 0 },
  { event := event83825
    frameStart := 0 },
  { event := event83826
    frameStart := 0 },
  { event := event83827
    frameStart := 0 },
  { event := event83828
    frameStart := 0 },
  { event := event83829
    frameStart := 0 },
  { event := event83830
    frameStart := 0 },
  { event := event83831
    frameStart := 0 },
  { event := event83832
    frameStart := 0 },
  { event := event83833
    frameStart := 0 },
  { event := event83834
    frameStart := 0 },
  { event := event83835
    frameStart := 0 },
  { event := event83836
    frameStart := 0 },
  { event := event83837
    frameStart := 0 },
  { event := event83838
    frameStart := 0 },
  { event := event83839
    frameStart := 0 }
]

def eventLeaf5240 : Array AnnotatedEvent := #[
  { event := event83840
    frameStart := 0 },
  { event := event83841
    frameStart := 0 },
  { event := event83842
    frameStart := 0 },
  { event := event83843
    frameStart := 0 },
  { event := event83844
    frameStart := 0 },
  { event := event83845
    frameStart := 0 },
  { event := event83846
    frameStart := 0 },
  { event := event83847
    frameStart := 0 },
  { event := event83848
    frameStart := 0 },
  { event := event83849
    frameStart := 0 },
  { event := event83850
    frameStart := 0 },
  { event := event83851
    frameStart := 0 },
  { event := event83852
    frameStart := 0 },
  { event := event83853
    frameStart := 0 },
  { event := event83854
    frameStart := 0 },
  { event := event83855
    frameStart := 0 }
]

def eventLeaf5241 : Array AnnotatedEvent := #[
  { event := event83856
    frameStart := 0 },
  { event := event83857
    frameStart := 0 },
  { event := event83858
    frameStart := 0 },
  { event := event83859
    frameStart := 83859 },
  { event := event83860
    frameStart := 83859 },
  { event := event83861
    frameStart := 83859 },
  { event := event83862
    frameStart := 83859 },
  { event := event83863
    frameStart := 83859 },
  { event := event83864
    frameStart := 83859 },
  { event := event83865
    frameStart := 83859 },
  { event := event83866
    frameStart := 83859 },
  { event := event83867
    frameStart := 83859 },
  { event := event83868
    frameStart := 83859 },
  { event := event83869
    frameStart := 83859 },
  { event := event83870
    frameStart := 83859 },
  { event := event83871
    frameStart := 83859 }
]

def eventLeaf5242 : Array AnnotatedEvent := #[
  { event := event83872
    frameStart := 83859 },
  { event := event83873
    frameStart := 83859 },
  { event := event83874
    frameStart := 83859 },
  { event := event83875
    frameStart := 83859 },
  { event := event83876
    frameStart := 83859 },
  { event := event83877
    frameStart := 83859 },
  { event := event83878
    frameStart := 83859 },
  { event := event83879
    frameStart := 83859 },
  { event := event83880
    frameStart := 83859 },
  { event := event83881
    frameStart := 83859 },
  { event := event83882
    frameStart := 83859 },
  { event := event83883
    frameStart := 83859 },
  { event := event83884
    frameStart := 83859 },
  { event := event83885
    frameStart := 83859 },
  { event := event83886
    frameStart := 83859 },
  { event := event83887
    frameStart := 83859 }
]

def eventLeaf5243 : Array AnnotatedEvent := #[
  { event := event83888
    frameStart := 83859 },
  { event := event83889
    frameStart := 83859 },
  { event := event83890
    frameStart := 83859 },
  { event := event83891
    frameStart := 83859 },
  { event := event83892
    frameStart := 83859 },
  { event := event83893
    frameStart := 83859 },
  { event := event83894
    frameStart := 83859 },
  { event := event83895
    frameStart := 83859 },
  { event := event83896
    frameStart := 83859 },
  { event := event83897
    frameStart := 83859 },
  { event := event83898
    frameStart := 83859 },
  { event := event83899
    frameStart := 83859 },
  { event := event83900
    frameStart := 83859 },
  { event := event83901
    frameStart := 83859 },
  { event := event83902
    frameStart := 83859 },
  { event := event83903
    frameStart := 83859 }
]

def eventLeaf5244 : Array AnnotatedEvent := #[
  { event := event83904
    frameStart := 83859 },
  { event := event83905
    frameStart := 83859 },
  { event := event83906
    frameStart := 83859 },
  { event := event83907
    frameStart := 83907 },
  { event := event83908
    frameStart := 83907 },
  { event := event83909
    frameStart := 83907 },
  { event := event83910
    frameStart := 83907 },
  { event := event83911
    frameStart := 83907 },
  { event := event83912
    frameStart := 83907 },
  { event := event83913
    frameStart := 83907 },
  { event := event83914
    frameStart := 83907 },
  { event := event83915
    frameStart := 83907 },
  { event := event83916
    frameStart := 83907 },
  { event := event83917
    frameStart := 83907 },
  { event := event83918
    frameStart := 83907 },
  { event := event83919
    frameStart := 83907 }
]

def eventLeaf5245 : Array AnnotatedEvent := #[
  { event := event83920
    frameStart := 83907 },
  { event := event83921
    frameStart := 83907 },
  { event := event83922
    frameStart := 83907 },
  { event := event83923
    frameStart := 83907 },
  { event := event83924
    frameStart := 83907 },
  { event := event83925
    frameStart := 83907 },
  { event := event83926
    frameStart := 83907 },
  { event := event83927
    frameStart := 83907 },
  { event := event83928
    frameStart := 83907 },
  { event := event83929
    frameStart := 83907 },
  { event := event83930
    frameStart := 83907 },
  { event := event83931
    frameStart := 83907 },
  { event := event83932
    frameStart := 83907 },
  { event := event83933
    frameStart := 83907 },
  { event := event83934
    frameStart := 83907 },
  { event := event83935
    frameStart := 83907 }
]

def eventLeaf5246 : Array AnnotatedEvent := #[
  { event := event83936
    frameStart := 83907 },
  { event := event83937
    frameStart := 83907 },
  { event := event83938
    frameStart := 83907 },
  { event := event83939
    frameStart := 83907 },
  { event := event83940
    frameStart := 83907 },
  { event := event83941
    frameStart := 83907 },
  { event := event83942
    frameStart := 83907 },
  { event := event83943
    frameStart := 83907 },
  { event := event83944
    frameStart := 83907 },
  { event := event83945
    frameStart := 83907 },
  { event := event83946
    frameStart := 83907 },
  { event := event83947
    frameStart := 83907 },
  { event := event83948
    frameStart := 83907 },
  { event := event83949
    frameStart := 83907 },
  { event := event83950
    frameStart := 83907 },
  { event := event83951
    frameStart := 83907 }
]

def eventLeaf5247 : Array AnnotatedEvent := #[
  { event := event83952
    frameStart := 83907 },
  { event := event83953
    frameStart := 83907 },
  { event := event83954
    frameStart := 83907 },
  { event := event83955
    frameStart := 83907 },
  { event := event83956
    frameStart := 83907 },
  { event := event83957
    frameStart := 83907 },
  { event := event83958
    frameStart := 83907 },
  { event := event83959
    frameStart := 83907 },
  { event := event83960
    frameStart := 83907 },
  { event := event83961
    frameStart := 83907 },
  { event := event83962
    frameStart := 83907 },
  { event := event83963
    frameStart := 83907 },
  { event := event83964
    frameStart := 83907 },
  { event := event83965
    frameStart := 83907 },
  { event := event83966
    frameStart := 83907 },
  { event := event83967
    frameStart := 83907 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events327
