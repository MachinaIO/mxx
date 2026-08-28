import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events569

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact145664RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13776⟩⟩, ⟨.program ⟨257⟩, ⟨36946⟩⟩], []⟩, (1)⟩]

theorem exact145664RawTermsValid :
    exact145664RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145664 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36947⟩⟩) exact145664RawTerms (.finite 1764) 145662 .exactZero (none)

def event145665 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36948⟩⟩) 0 ⟨36947⟩ 145664

def event145666 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36948⟩⟩) (.identity (.predecessor 0 145665 .coefficient))

def event145667 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨36948⟩⟩) (.finite 1764)

def event145668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37372⟩⟩) 0 ⟨36948⟩ 145667

def event145669 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37372⟩⟩) (.authority (.programFamilyFact))

def exact145670RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37372⟩⟩], []⟩, (1)⟩]

theorem exact145670RawTermsValid :
    exact145670RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145670 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37372⟩⟩) exact145670RawTerms (.finite 42) 145669 .exactZero (none)

def event145671 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37373⟩⟩) 0 ⟨37372⟩ 145670

def event145672 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37373⟩⟩) (.identity (.predecessor 0 145671 .coefficient))

def event145673 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37373⟩⟩) (.finite 42)

def event145674 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38516⟩⟩) 0 ⟨37373⟩ 145673

def event145675 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38516⟩⟩) (.authority (.programFamilyFact))

def event145676 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38516⟩⟩) (.finite 3720)

def event145677 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event145678 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38517⟩⟩) 0 ⟨7177⟩ 145677

def event145679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38517⟩⟩) 1 ⟨38516⟩ 145676

def event145680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38517⟩⟩) (.authority (.operator))

def exact145681RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38517⟩⟩]⟩, (1)⟩]

theorem exact145681RawTermsValid :
    exact145681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145681 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38517⟩⟩) exact145681RawTerms .large 145680 .exactZero (none)

def event145682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39128⟩⟩) 0 ⟨38517⟩ 145681

def event145683 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39128⟩⟩) (.authority (.operator))

def exact145684RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39128⟩⟩]⟩, (1)⟩]

theorem exact145684RawTermsValid :
    exact145684RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145684 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39128⟩⟩) exact145684RawTerms (.finite 8192) 145683 .exactZero (none)

def event145685 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event145686 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event145687 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38758⟩⟩) 0 ⟨37373⟩ 145673

def event145688 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38758⟩⟩) 1 ⟨136⟩ 145686

def event145689 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38758⟩⟩) (.sum [.predecessor 0 145687 .coefficient, .predecessor 1 145688 .coefficient])

def event145690 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38758⟩⟩) (.finite 42)

def event145691 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38759⟩⟩) 0 ⟨38758⟩ 145690

def event145692 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38759⟩⟩) (.identity (.predecessor 0 145691 .coefficient))

def exact145693RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37372⟩⟩], []⟩, (1)⟩]

theorem exact145693RawTermsValid :
    exact145693RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145693 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38759⟩⟩) exact145693RawTerms (.finite 42) 145692 .exactZero (none)

def event145694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact145695RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact145695RawTermsValid :
    exact145695RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145695 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact145695RawTerms .large 145694 .exactZero (none)

def event145696 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38760⟩⟩) 0 ⟨6908⟩ 145695

def event145697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38760⟩⟩) 1 ⟨38759⟩ 145693

def event145698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38760⟩⟩) (.product (.predecessor 0 145696 .coefficient) (.predecessor 1 145697 .coefficient) (⟨false, false, none, none, none⟩))

def event145699 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38760⟩⟩, .operator (⟨145695, 0⟩, ⟨145693, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37372⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact145700RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37372⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact145700RawTermsValid :
    exact145700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145700 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38760⟩⟩) exact145700RawTerms .large 145698 .exactZero (none)

def event145701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7192⟩⟩) 0 ⟨7177⟩ 145677

def event145702 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7192⟩⟩) (.authority (.operator))

def exact145703RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩]

theorem exact145703RawTermsValid :
    exact145703RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145703 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7192⟩⟩) exact145703RawTerms .large 145702 .exactZero (none)

def event145704 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38761⟩⟩) 0 ⟨7192⟩ 145703

def event145705 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38761⟩⟩) 1 ⟨38760⟩ 145700

def event145706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38761⟩⟩) (.sum [.predecessor 0 145704 .coefficient, .predecessor 1 145705 .coefficient])

def exact145707RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37372⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact145707RawTermsValid :
    exact145707RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145707 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38761⟩⟩) exact145707RawTerms .large 145706 .exactZero (none)

def event145708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39129⟩⟩) 0 ⟨38761⟩ 145707

def event145709 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39129⟩⟩) 1 ⟨39128⟩ 145684

def event145710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39129⟩⟩) (.product (.predecessor 0 145708 .coefficient) (.predecessor 1 145709 .coefficient) (⟨false, false, none, none, none⟩))

def event145711 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39129⟩⟩, .operator (⟨145707, 0⟩, ⟨145684, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39128⟩⟩]⟩, (1)⟩)

def event145712 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39129⟩⟩, .operator (⟨145707, 1⟩, ⟨145684, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37372⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39128⟩⟩]⟩, (-1)⟩)

def event145713 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39129⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨37372⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39128⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39128⟩⟩) ⟨38517⟩ 145681)

def event145714 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39129⟩⟩, .relation 145713 0, ⟨[⟨.program ⟨257⟩, ⟨37372⟩⟩], [⟨.program ⟨257⟩, ⟨38517⟩⟩]⟩, (-1)⟩)

def exact145715RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39128⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37372⟩⟩], [⟨.program ⟨257⟩, ⟨38517⟩⟩]⟩, (-1)⟩]

theorem exact145715RawTermsValid :
    exact145715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145715 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39129⟩⟩) exact145715RawTerms .large 145710 .exactZero (none)

def event145716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37548⟩⟩) 0 ⟨37373⟩ 145673

def event145717 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37548⟩⟩) (.authority (.programFamilyFact))

def exact145718RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37548⟩⟩], []⟩, (1)⟩]

theorem exact145718RawTermsValid :
    exact145718RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145718 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37548⟩⟩) exact145718RawTerms (.finite 42) 145717 .exactZero (none)

def event145719 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37550⟩⟩) 0 ⟨6908⟩ 145695

def event145720 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37550⟩⟩) 1 ⟨37548⟩ 145718

def event145721 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37550⟩⟩) (.product (.predecessor 0 145719 .coefficient) (.predecessor 1 145720 .coefficient) (⟨false, true, none, none, some 1⟩))

def event145722 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37550⟩⟩, .operator (⟨145695, 0⟩, ⟨145718, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact145723RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact145723RawTermsValid :
    exact145723RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145723 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37550⟩⟩) exact145723RawTerms .large 145721 .exactZero (none)

def event145724 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7223⟩⟩) 0 ⟨7177⟩ 145677

def event145725 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7223⟩⟩) (.authority (.operator))

def exact145726RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩]

theorem exact145726RawTermsValid :
    exact145726RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145726 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7223⟩⟩) exact145726RawTerms .large 145725 .exactZero (none)

def event145727 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37551⟩⟩) 0 ⟨7223⟩ 145726

def event145728 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37551⟩⟩) 1 ⟨37550⟩ 145723

def event145729 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37551⟩⟩) (.sum [.predecessor 0 145727 .coefficient, .predecessor 1 145728 .coefficient])

def exact145730RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact145730RawTermsValid :
    exact145730RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145730 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37551⟩⟩) exact145730RawTerms .large 145729 .exactZero (none)

def event145731 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39133⟩⟩) 0 ⟨37551⟩ 145730

def event145732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39133⟩⟩) 1 ⟨39129⟩ 145715

def event145733 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39133⟩⟩) (.sum [.predecessor 0 145731 .coefficient, .predecessor 1 145732 .coefficient])

def exact145734RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39128⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37372⟩⟩], [⟨.program ⟨257⟩, ⟨38517⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact145734RawTermsValid :
    exact145734RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145734 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39133⟩⟩) exact145734RawTerms .large 145733 .exactZero (none)

def event145735 : Event := .preFoldPolynomial 145734 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39128⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37372⟩⟩], [⟨.program ⟨257⟩, ⟨38517⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact145736RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39128⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37372⟩⟩], [⟨.program ⟨257⟩, ⟨38517⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event145736 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨39133⟩⟩) 145735 exact145736RawTerms .large 145733 .exactZero (none)

def event145737 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨37373⟩⟩) ⟨⟨102⟩, ⟨84⟩, ⟨135⟩⟩ ⟨145579, 145737⟩

def event145738 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨38035⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38032⟩⟩]⟩) (1) 0 2 (.universal 145737 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38032⟩⟩]⟩) (none) 145736)

def event145739 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38035⟩⟩, .relation 145738 1, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩)

def event145740 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38035⟩⟩, .relation 145738 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39128⟩⟩]⟩, (-1)⟩)

def event145741 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38035⟩⟩, .relation 145738 2, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37372⟩⟩], [⟨.program ⟨257⟩, ⟨38517⟩⟩]⟩, (1)⟩)

def event145742 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38035⟩⟩, .relation 145738 3, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact145743RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39128⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37372⟩⟩], [⟨.program ⟨257⟩, ⟨38517⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact145743RawTermsValid :
    exact145743RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145743 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38035⟩⟩) exact145743RawTerms .large 145575 (.finite 202072841853861888) (some (145577))

def event145744 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39131⟩⟩) 0 ⟨38035⟩ 145743

def event145745 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39131⟩⟩) 1 ⟨39130⟩ 145565

def event145746 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39131⟩⟩) (.sum [.predecessor 0 145744 .coefficient, .predecessor 1 145745 .coefficient])

def event145747 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39131⟩⟩, .operator (⟨145743, 0⟩, ⟨145565, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39128⟩⟩]⟩, (1)⟩)

def event145748 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39131⟩⟩, .operator (⟨145743, 2⟩, ⟨145565, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37372⟩⟩], [⟨.program ⟨257⟩, ⟨38517⟩⟩]⟩, (-1)⟩)

def event145749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39131⟩⟩) (.sum [.result 145743 .summary, .result 145565 .summary])

def exact145750RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact145750RawTermsValid :
    exact145750RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145750 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39131⟩⟩) exact145750RawTerms .large 145746 (.finite 32192736221397454434328420548608) (some (145749))

def event145751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39132⟩⟩) 0 ⟨39131⟩ 145750

def event145752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39132⟩⟩) 1 ⟨7162⟩ 15622

def event145753 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39132⟩⟩) (.product (.predecessor 0 145751 .coefficient) (.predecessor 1 145752 .coefficient) (⟨false, false, none, none, none⟩))

def event145754 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39132⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩) [⟨.result 15618 .coefficient, false, none⟩])

def event145755 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39132⟩⟩) (.product (.result 145750 .summary) (.transfer 145754) (⟨false, false, none, none, none⟩))

def event145756 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39132⟩⟩, .operator (⟨145750, 0⟩, ⟨15622, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩)

def event145757 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39132⟩⟩, .operator (⟨145750, 1⟩, ⟨15622, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (-1)⟩)

def event145758 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39132⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7161⟩⟩) ⟨7046⟩ 15615)

def event145759 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39132⟩⟩, .relation 145758 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact145760RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37548⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact145760RawTermsValid :
    exact145760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145760 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39132⟩⟩) exact145760RawTerms .large 145753 (.finite 345666873099141705532726864949014345809920) (some (145755))

def event145761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35837⟩⟩) 0 ⟨7177⟩ 15500

def event145762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35837⟩⟩) 1 ⟨35836⟩ 136807

def event145763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35837⟩⟩) (.authority (.operator))

def exact145764RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35837⟩⟩]⟩, (1)⟩]

theorem exact145764RawTermsValid :
    exact145764RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145764 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35837⟩⟩) exact145764RawTerms .large 145763 .exactZero (none)

def event145765 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36448⟩⟩) 0 ⟨35837⟩ 145764

def event145766 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36448⟩⟩) (.authority (.operator))

def exact145767RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩, (1)⟩]

theorem exact145767RawTermsValid :
    exact145767RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145767 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36448⟩⟩) exact145767RawTerms (.finite 8192) 145766 .exactZero (none)

def event145768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36450⟩⟩) 0 ⟨36184⟩ 137091

def event145769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36450⟩⟩) 1 ⟨36448⟩ 145767

def event145770 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36450⟩⟩) (.product (.predecessor 0 145768 .coefficient) (.predecessor 1 145769 .coefficient) (⟨false, false, none, none, none⟩))

def event145771 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36450⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩) [⟨.result 145767 .coefficient, false, none⟩])

def event145772 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36450⟩⟩) (.product (.result 137091 .summary) (.transfer 145771) (⟨false, false, none, none, none⟩))

def event145773 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36450⟩⟩, .operator (⟨137091, 0⟩, ⟨145767, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩, (1)⟩)

def event145774 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36450⟩⟩, .operator (⟨137091, 1⟩, ⟨145767, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩, (-1)⟩)

def event145775 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36450⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36448⟩⟩) ⟨35837⟩ 145764)

def event145776 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36450⟩⟩, .relation 145775 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨35837⟩⟩]⟩, (-1)⟩)

def exact145777RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨35837⟩⟩]⟩, (-1)⟩]

theorem exact145777RawTermsValid :
    exact145777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145777 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36450⟩⟩) exact145777RawTerms .large 145770 (.finite 32192539770951564984245676933120) (some (145772))

def event145778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35352⟩⟩) 0 ⟨34693⟩ 6210

def event145779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35352⟩⟩) (.authority (.relationPreimageSource ⟨82⟩))

def exact145780RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩, (1)⟩]

theorem exact145780RawTermsValid :
    exact145780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145780 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35352⟩⟩) exact145780RawTerms (.finite 5647228698) 145779 .exactZero (none)

def event145781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35354⟩⟩) 0 ⟨35352⟩ 145780

def event145782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35354⟩⟩) 1 ⟨2370⟩ 4

def event145783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35354⟩⟩) (.scale (.predecessor 0 145781 .coefficient) (.value (.predecessor 1 145782 .coefficient)))

def exact145784RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩, (1)⟩]

theorem exact145784RawTermsValid :
    exact145784RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145784 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35354⟩⟩) exact145784RawTerms (.finite 5647228698) 145783 .exactZero (none)

def event145785 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35355⟩⟩) 0 ⟨5473⟩ 134495

def event145786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35355⟩⟩) 1 ⟨35354⟩ 145784

def event145787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35355⟩⟩) (.product (.predecessor 0 145785 .coefficient) (.predecessor 1 145786 .coefficient) (⟨false, false, none, none, none⟩))

def event145788 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35355⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩) [⟨.result 145780 .coefficient, false, none⟩])

def event145789 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35355⟩⟩) (.product (.result 134495 .summary) (.transfer 145788) (⟨false, false, none, none, none⟩))

def event145790 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35355⟩⟩, .operator (⟨134495, 0⟩, ⟨145784, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩, (1)⟩)

def event145791 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨35353⟩⟩)

def event145792 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event145793 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event145794 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event145795 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event145796 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event145797 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event145798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event145799 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event145800 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 145799

def event145801 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 145797

def event145802 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 145800 .coefficient) (.value (.predecessor 1 145801 .coefficient)))

def event145803 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event145804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 145803

def event145805 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 145795

def event145806 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 145804 .coefficient, .predecessor 1 145805 .coefficient])

def event145807 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event145808 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 145807

def event145809 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 145793

def event145810 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 145809 .coefficient))

def event145811 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event145812 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34266⟩⟩) 0 ⟨5469⟩ 145811

def event145813 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34266⟩⟩) (.authority (.programFamilyFact))

def exact145814RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34266⟩⟩], []⟩, (1)⟩]

theorem exact145814RawTermsValid :
    exact145814RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145814 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34266⟩⟩) exact145814RawTerms (.finite 40) 145813 .exactZero (none)

def event145815 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13476⟩⟩) 0 ⟨5469⟩ 145811

def event145816 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13476⟩⟩) (.authority (.programFamilyFact))

def exact145817RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13476⟩⟩], []⟩, (1)⟩]

theorem exact145817RawTermsValid :
    exact145817RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145817 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13476⟩⟩) exact145817RawTerms (.finite 40) 145816 .exactZero (none)

def event145818 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34267⟩⟩) 0 ⟨13476⟩ 145817

def event145819 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34267⟩⟩) 1 ⟨34266⟩ 145814

def event145820 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34267⟩⟩) (.product (.predecessor 0 145818 .coefficient) (.predecessor 1 145819 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event145821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34267⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], []⟩) [⟨.result 145817 .coefficient, true, some 1⟩, ⟨.result 145814 .coefficient, true, some 1⟩])

def event145822 : Event := .survivorFold (1) 145821

def exact145823RawTerms : List Term := []

theorem exact145823RawTermsValid :
    exact145823RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145823 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34267⟩⟩) exact145823RawTerms (.finite 1600) 145820 (.finite 1600) (some (145821))

def event145824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34268⟩⟩) 0 ⟨34267⟩ 145823

def event145825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34268⟩⟩) (.identity (.predecessor 0 145824 .coefficient))

def event145826 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34268⟩⟩) (.finite 1600)

def event145827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34692⟩⟩) 0 ⟨34268⟩ 145826

def event145828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34692⟩⟩) (.authority (.programFamilyFact))

def exact145829RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34692⟩⟩], []⟩, (1)⟩]

theorem exact145829RawTermsValid :
    exact145829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145829 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34692⟩⟩) exact145829RawTerms (.finite 40) 145828 .exactZero (none)

def event145830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34693⟩⟩) 0 ⟨34692⟩ 145829

def event145831 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34693⟩⟩) (.identity (.predecessor 0 145830 .coefficient))

def event145832 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34693⟩⟩) (.finite 40)

def event145833 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35352⟩⟩) 0 ⟨34693⟩ 145832

def event145834 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35352⟩⟩) (.authority (.relationPreimageSource ⟨82⟩))

def exact145835RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩, (1)⟩]

theorem exact145835RawTermsValid :
    exact145835RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145835 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35352⟩⟩) exact145835RawTerms (.finite 5647228698) 145834 .exactZero (none)

def event145836 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact145837RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact145837RawTermsValid :
    exact145837RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145837 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact145837RawTerms .large 145836 .exactZero (none)

def event145838 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35353⟩⟩) 0 ⟨35⟩ 145837

def event145839 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35353⟩⟩) 1 ⟨35352⟩ 145835

def event145840 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35353⟩⟩) (.product (.predecessor 0 145838 .coefficient) (.predecessor 1 145839 .coefficient) (⟨false, false, none, none, none⟩))

def event145841 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35353⟩⟩, .operator (⟨145837, 0⟩, ⟨145835, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩, (1)⟩)

def exact145842RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩, (1)⟩]

theorem exact145842RawTermsValid :
    exact145842RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145842 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35353⟩⟩) exact145842RawTerms .large 145840 .exactZero (none)

def event145843 : Event := .preFoldPolynomial 145842 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩, (1)⟩] .exactZero none

def exact145844RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35352⟩⟩]⟩, (1)⟩]

def event145844 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨35353⟩⟩) 145843 exact145844RawTerms .large 145840 .exactZero (none)

def event145845 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨36453⟩⟩)

def event145846 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event145847 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event145848 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event145849 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event145850 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event145851 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event145852 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event145853 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event145854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 145853

def event145855 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 145851

def event145856 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 145854 .coefficient) (.value (.predecessor 1 145855 .coefficient)))

def event145857 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event145858 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 145857

def event145859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 145849

def event145860 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 145858 .coefficient, .predecessor 1 145859 .coefficient])

def event145861 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event145862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 145861

def event145863 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 145847

def event145864 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 145863 .coefficient))

def event145865 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event145866 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34266⟩⟩) 0 ⟨5469⟩ 145865

def event145867 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34266⟩⟩) (.authority (.programFamilyFact))

def exact145868RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34266⟩⟩], []⟩, (1)⟩]

theorem exact145868RawTermsValid :
    exact145868RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145868 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34266⟩⟩) exact145868RawTerms (.finite 40) 145867 .exactZero (none)

def event145869 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13476⟩⟩) 0 ⟨5469⟩ 145865

def event145870 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13476⟩⟩) (.authority (.programFamilyFact))

def exact145871RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13476⟩⟩], []⟩, (1)⟩]

theorem exact145871RawTermsValid :
    exact145871RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145871 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13476⟩⟩) exact145871RawTerms (.finite 40) 145870 .exactZero (none)

def event145872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34267⟩⟩) 0 ⟨13476⟩ 145871

def event145873 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34267⟩⟩) 1 ⟨34266⟩ 145868

def event145874 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34267⟩⟩) (.product (.predecessor 0 145872 .coefficient) (.predecessor 1 145873 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event145875 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34267⟩⟩, .operator (⟨145871, 0⟩, ⟨145868, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], []⟩, (1)⟩)

def exact145876RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], []⟩, (1)⟩]

theorem exact145876RawTermsValid :
    exact145876RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145876 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34267⟩⟩) exact145876RawTerms (.finite 1600) 145874 .exactZero (none)

def event145877 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34268⟩⟩) 0 ⟨34267⟩ 145876

def event145878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34268⟩⟩) (.identity (.predecessor 0 145877 .coefficient))

def event145879 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34268⟩⟩) (.finite 1600)

def event145880 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34692⟩⟩) 0 ⟨34268⟩ 145879

def event145881 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34692⟩⟩) (.authority (.programFamilyFact))

def exact145882RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34692⟩⟩], []⟩, (1)⟩]

theorem exact145882RawTermsValid :
    exact145882RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145882 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34692⟩⟩) exact145882RawTerms (.finite 40) 145881 .exactZero (none)

def event145883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34693⟩⟩) 0 ⟨34692⟩ 145882

def event145884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34693⟩⟩) (.identity (.predecessor 0 145883 .coefficient))

def event145885 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34693⟩⟩) (.finite 40)

def event145886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35836⟩⟩) 0 ⟨34693⟩ 145885

def event145887 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35836⟩⟩) (.authority (.programFamilyFact))

def event145888 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨35836⟩⟩) (.finite 3720)

def event145889 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event145890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35837⟩⟩) 0 ⟨7177⟩ 145889

def event145891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35837⟩⟩) 1 ⟨35836⟩ 145888

def event145892 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35837⟩⟩) (.authority (.operator))

def exact145893RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35837⟩⟩]⟩, (1)⟩]

theorem exact145893RawTermsValid :
    exact145893RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145893 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35837⟩⟩) exact145893RawTerms .large 145892 .exactZero (none)

def event145894 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36448⟩⟩) 0 ⟨35837⟩ 145893

def event145895 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36448⟩⟩) (.authority (.operator))

def exact145896RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨36448⟩⟩]⟩, (1)⟩]

theorem exact145896RawTermsValid :
    exact145896RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145896 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36448⟩⟩) exact145896RawTerms (.finite 8192) 145895 .exactZero (none)

def event145897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event145898 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event145899 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36078⟩⟩) 0 ⟨34693⟩ 145885

def event145900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36078⟩⟩) 1 ⟨136⟩ 145898

def event145901 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36078⟩⟩) (.sum [.predecessor 0 145899 .coefficient, .predecessor 1 145900 .coefficient])

def event145902 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨36078⟩⟩) (.finite 40)

def event145903 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36079⟩⟩) 0 ⟨36078⟩ 145902

def event145904 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36079⟩⟩) (.identity (.predecessor 0 145903 .coefficient))

def exact145905RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34692⟩⟩], []⟩, (1)⟩]

theorem exact145905RawTermsValid :
    exact145905RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145905 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36079⟩⟩) exact145905RawTerms (.finite 40) 145904 .exactZero (none)

def event145906 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact145907RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact145907RawTermsValid :
    exact145907RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145907 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact145907RawTerms .large 145906 .exactZero (none)

def event145908 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36080⟩⟩) 0 ⟨6908⟩ 145907

def event145909 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36080⟩⟩) 1 ⟨36079⟩ 145905

def event145910 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36080⟩⟩) (.product (.predecessor 0 145908 .coefficient) (.predecessor 1 145909 .coefficient) (⟨false, false, none, none, none⟩))

def event145911 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36080⟩⟩, .operator (⟨145907, 0⟩, ⟨145905, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact145912RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact145912RawTermsValid :
    exact145912RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145912 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36080⟩⟩) exact145912RawTerms .large 145910 .exactZero (none)

def event145913 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7191⟩⟩) 0 ⟨7177⟩ 145889

def event145914 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7191⟩⟩) (.authority (.operator))

def exact145915RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩]

theorem exact145915RawTermsValid :
    exact145915RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145915 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7191⟩⟩) exact145915RawTerms .large 145914 .exactZero (none)

def event145916 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36081⟩⟩) 0 ⟨7191⟩ 145915

def event145917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36081⟩⟩) 1 ⟨36080⟩ 145912

def event145918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36081⟩⟩) (.sum [.predecessor 0 145916 .coefficient, .predecessor 1 145917 .coefficient])

def exact145919RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact145919RawTermsValid :
    exact145919RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event145919 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36081⟩⟩) exact145919RawTerms .large 145918 .exactZero (none)

def eventLeaf9104 : Array AnnotatedEvent := #[
  { event := event145664
    frameStart := 145633 },
  { event := event145665
    frameStart := 145633 },
  { event := event145666
    frameStart := 145633 },
  { event := event145667
    frameStart := 145633 },
  { event := event145668
    frameStart := 145633 },
  { event := event145669
    frameStart := 145633 },
  { event := event145670
    frameStart := 145633 },
  { event := event145671
    frameStart := 145633 },
  { event := event145672
    frameStart := 145633 },
  { event := event145673
    frameStart := 145633 },
  { event := event145674
    frameStart := 145633 },
  { event := event145675
    frameStart := 145633 },
  { event := event145676
    frameStart := 145633 },
  { event := event145677
    frameStart := 145633 },
  { event := event145678
    frameStart := 145633 },
  { event := event145679
    frameStart := 145633 }
]

def eventLeaf9105 : Array AnnotatedEvent := #[
  { event := event145680
    frameStart := 145633 },
  { event := event145681
    frameStart := 145633 },
  { event := event145682
    frameStart := 145633 },
  { event := event145683
    frameStart := 145633 },
  { event := event145684
    frameStart := 145633 },
  { event := event145685
    frameStart := 145633 },
  { event := event145686
    frameStart := 145633 },
  { event := event145687
    frameStart := 145633 },
  { event := event145688
    frameStart := 145633 },
  { event := event145689
    frameStart := 145633 },
  { event := event145690
    frameStart := 145633 },
  { event := event145691
    frameStart := 145633 },
  { event := event145692
    frameStart := 145633 },
  { event := event145693
    frameStart := 145633 },
  { event := event145694
    frameStart := 145633 },
  { event := event145695
    frameStart := 145633 }
]

def eventLeaf9106 : Array AnnotatedEvent := #[
  { event := event145696
    frameStart := 145633 },
  { event := event145697
    frameStart := 145633 },
  { event := event145698
    frameStart := 145633 },
  { event := event145699
    frameStart := 145633 },
  { event := event145700
    frameStart := 145633 },
  { event := event145701
    frameStart := 145633 },
  { event := event145702
    frameStart := 145633 },
  { event := event145703
    frameStart := 145633 },
  { event := event145704
    frameStart := 145633 },
  { event := event145705
    frameStart := 145633 },
  { event := event145706
    frameStart := 145633 },
  { event := event145707
    frameStart := 145633 },
  { event := event145708
    frameStart := 145633 },
  { event := event145709
    frameStart := 145633 },
  { event := event145710
    frameStart := 145633 },
  { event := event145711
    frameStart := 145633 }
]

def eventLeaf9107 : Array AnnotatedEvent := #[
  { event := event145712
    frameStart := 145633 },
  { event := event145713
    frameStart := 145633 },
  { event := event145714
    frameStart := 145633 },
  { event := event145715
    frameStart := 145633 },
  { event := event145716
    frameStart := 145633 },
  { event := event145717
    frameStart := 145633 },
  { event := event145718
    frameStart := 145633 },
  { event := event145719
    frameStart := 145633 },
  { event := event145720
    frameStart := 145633 },
  { event := event145721
    frameStart := 145633 },
  { event := event145722
    frameStart := 145633 },
  { event := event145723
    frameStart := 145633 },
  { event := event145724
    frameStart := 145633 },
  { event := event145725
    frameStart := 145633 },
  { event := event145726
    frameStart := 145633 },
  { event := event145727
    frameStart := 145633 }
]

def eventLeaf9108 : Array AnnotatedEvent := #[
  { event := event145728
    frameStart := 145633 },
  { event := event145729
    frameStart := 145633 },
  { event := event145730
    frameStart := 145633 },
  { event := event145731
    frameStart := 145633 },
  { event := event145732
    frameStart := 145633 },
  { event := event145733
    frameStart := 145633 },
  { event := event145734
    frameStart := 145633 },
  { event := event145735
    frameStart := 145633 },
  { event := event145736
    frameStart := 145633 },
  { event := event145737
    frameStart := 0 },
  { event := event145738
    frameStart := 0 },
  { event := event145739
    frameStart := 0 },
  { event := event145740
    frameStart := 0 },
  { event := event145741
    frameStart := 0 },
  { event := event145742
    frameStart := 0 },
  { event := event145743
    frameStart := 0 }
]

def eventLeaf9109 : Array AnnotatedEvent := #[
  { event := event145744
    frameStart := 0 },
  { event := event145745
    frameStart := 0 },
  { event := event145746
    frameStart := 0 },
  { event := event145747
    frameStart := 0 },
  { event := event145748
    frameStart := 0 },
  { event := event145749
    frameStart := 0 },
  { event := event145750
    frameStart := 0 },
  { event := event145751
    frameStart := 0 },
  { event := event145752
    frameStart := 0 },
  { event := event145753
    frameStart := 0 },
  { event := event145754
    frameStart := 0 },
  { event := event145755
    frameStart := 0 },
  { event := event145756
    frameStart := 0 },
  { event := event145757
    frameStart := 0 },
  { event := event145758
    frameStart := 0 },
  { event := event145759
    frameStart := 0 }
]

def eventLeaf9110 : Array AnnotatedEvent := #[
  { event := event145760
    frameStart := 0 },
  { event := event145761
    frameStart := 0 },
  { event := event145762
    frameStart := 0 },
  { event := event145763
    frameStart := 0 },
  { event := event145764
    frameStart := 0 },
  { event := event145765
    frameStart := 0 },
  { event := event145766
    frameStart := 0 },
  { event := event145767
    frameStart := 0 },
  { event := event145768
    frameStart := 0 },
  { event := event145769
    frameStart := 0 },
  { event := event145770
    frameStart := 0 },
  { event := event145771
    frameStart := 0 },
  { event := event145772
    frameStart := 0 },
  { event := event145773
    frameStart := 0 },
  { event := event145774
    frameStart := 0 },
  { event := event145775
    frameStart := 0 }
]

def eventLeaf9111 : Array AnnotatedEvent := #[
  { event := event145776
    frameStart := 0 },
  { event := event145777
    frameStart := 0 },
  { event := event145778
    frameStart := 0 },
  { event := event145779
    frameStart := 0 },
  { event := event145780
    frameStart := 0 },
  { event := event145781
    frameStart := 0 },
  { event := event145782
    frameStart := 0 },
  { event := event145783
    frameStart := 0 },
  { event := event145784
    frameStart := 0 },
  { event := event145785
    frameStart := 0 },
  { event := event145786
    frameStart := 0 },
  { event := event145787
    frameStart := 0 },
  { event := event145788
    frameStart := 0 },
  { event := event145789
    frameStart := 0 },
  { event := event145790
    frameStart := 0 },
  { event := event145791
    frameStart := 145791 }
]

def eventLeaf9112 : Array AnnotatedEvent := #[
  { event := event145792
    frameStart := 145791 },
  { event := event145793
    frameStart := 145791 },
  { event := event145794
    frameStart := 145791 },
  { event := event145795
    frameStart := 145791 },
  { event := event145796
    frameStart := 145791 },
  { event := event145797
    frameStart := 145791 },
  { event := event145798
    frameStart := 145791 },
  { event := event145799
    frameStart := 145791 },
  { event := event145800
    frameStart := 145791 },
  { event := event145801
    frameStart := 145791 },
  { event := event145802
    frameStart := 145791 },
  { event := event145803
    frameStart := 145791 },
  { event := event145804
    frameStart := 145791 },
  { event := event145805
    frameStart := 145791 },
  { event := event145806
    frameStart := 145791 },
  { event := event145807
    frameStart := 145791 }
]

def eventLeaf9113 : Array AnnotatedEvent := #[
  { event := event145808
    frameStart := 145791 },
  { event := event145809
    frameStart := 145791 },
  { event := event145810
    frameStart := 145791 },
  { event := event145811
    frameStart := 145791 },
  { event := event145812
    frameStart := 145791 },
  { event := event145813
    frameStart := 145791 },
  { event := event145814
    frameStart := 145791 },
  { event := event145815
    frameStart := 145791 },
  { event := event145816
    frameStart := 145791 },
  { event := event145817
    frameStart := 145791 },
  { event := event145818
    frameStart := 145791 },
  { event := event145819
    frameStart := 145791 },
  { event := event145820
    frameStart := 145791 },
  { event := event145821
    frameStart := 145791 },
  { event := event145822
    frameStart := 145791 },
  { event := event145823
    frameStart := 145791 }
]

def eventLeaf9114 : Array AnnotatedEvent := #[
  { event := event145824
    frameStart := 145791 },
  { event := event145825
    frameStart := 145791 },
  { event := event145826
    frameStart := 145791 },
  { event := event145827
    frameStart := 145791 },
  { event := event145828
    frameStart := 145791 },
  { event := event145829
    frameStart := 145791 },
  { event := event145830
    frameStart := 145791 },
  { event := event145831
    frameStart := 145791 },
  { event := event145832
    frameStart := 145791 },
  { event := event145833
    frameStart := 145791 },
  { event := event145834
    frameStart := 145791 },
  { event := event145835
    frameStart := 145791 },
  { event := event145836
    frameStart := 145791 },
  { event := event145837
    frameStart := 145791 },
  { event := event145838
    frameStart := 145791 },
  { event := event145839
    frameStart := 145791 }
]

def eventLeaf9115 : Array AnnotatedEvent := #[
  { event := event145840
    frameStart := 145791 },
  { event := event145841
    frameStart := 145791 },
  { event := event145842
    frameStart := 145791 },
  { event := event145843
    frameStart := 145791 },
  { event := event145844
    frameStart := 145791 },
  { event := event145845
    frameStart := 145845 },
  { event := event145846
    frameStart := 145845 },
  { event := event145847
    frameStart := 145845 },
  { event := event145848
    frameStart := 145845 },
  { event := event145849
    frameStart := 145845 },
  { event := event145850
    frameStart := 145845 },
  { event := event145851
    frameStart := 145845 },
  { event := event145852
    frameStart := 145845 },
  { event := event145853
    frameStart := 145845 },
  { event := event145854
    frameStart := 145845 },
  { event := event145855
    frameStart := 145845 }
]

def eventLeaf9116 : Array AnnotatedEvent := #[
  { event := event145856
    frameStart := 145845 },
  { event := event145857
    frameStart := 145845 },
  { event := event145858
    frameStart := 145845 },
  { event := event145859
    frameStart := 145845 },
  { event := event145860
    frameStart := 145845 },
  { event := event145861
    frameStart := 145845 },
  { event := event145862
    frameStart := 145845 },
  { event := event145863
    frameStart := 145845 },
  { event := event145864
    frameStart := 145845 },
  { event := event145865
    frameStart := 145845 },
  { event := event145866
    frameStart := 145845 },
  { event := event145867
    frameStart := 145845 },
  { event := event145868
    frameStart := 145845 },
  { event := event145869
    frameStart := 145845 },
  { event := event145870
    frameStart := 145845 },
  { event := event145871
    frameStart := 145845 }
]

def eventLeaf9117 : Array AnnotatedEvent := #[
  { event := event145872
    frameStart := 145845 },
  { event := event145873
    frameStart := 145845 },
  { event := event145874
    frameStart := 145845 },
  { event := event145875
    frameStart := 145845 },
  { event := event145876
    frameStart := 145845 },
  { event := event145877
    frameStart := 145845 },
  { event := event145878
    frameStart := 145845 },
  { event := event145879
    frameStart := 145845 },
  { event := event145880
    frameStart := 145845 },
  { event := event145881
    frameStart := 145845 },
  { event := event145882
    frameStart := 145845 },
  { event := event145883
    frameStart := 145845 },
  { event := event145884
    frameStart := 145845 },
  { event := event145885
    frameStart := 145845 },
  { event := event145886
    frameStart := 145845 },
  { event := event145887
    frameStart := 145845 }
]

def eventLeaf9118 : Array AnnotatedEvent := #[
  { event := event145888
    frameStart := 145845 },
  { event := event145889
    frameStart := 145845 },
  { event := event145890
    frameStart := 145845 },
  { event := event145891
    frameStart := 145845 },
  { event := event145892
    frameStart := 145845 },
  { event := event145893
    frameStart := 145845 },
  { event := event145894
    frameStart := 145845 },
  { event := event145895
    frameStart := 145845 },
  { event := event145896
    frameStart := 145845 },
  { event := event145897
    frameStart := 145845 },
  { event := event145898
    frameStart := 145845 },
  { event := event145899
    frameStart := 145845 },
  { event := event145900
    frameStart := 145845 },
  { event := event145901
    frameStart := 145845 },
  { event := event145902
    frameStart := 145845 },
  { event := event145903
    frameStart := 145845 }
]

def eventLeaf9119 : Array AnnotatedEvent := #[
  { event := event145904
    frameStart := 145845 },
  { event := event145905
    frameStart := 145845 },
  { event := event145906
    frameStart := 145845 },
  { event := event145907
    frameStart := 145845 },
  { event := event145908
    frameStart := 145845 },
  { event := event145909
    frameStart := 145845 },
  { event := event145910
    frameStart := 145845 },
  { event := event145911
    frameStart := 145845 },
  { event := event145912
    frameStart := 145845 },
  { event := event145913
    frameStart := 145845 },
  { event := event145914
    frameStart := 145845 },
  { event := event145915
    frameStart := 145845 },
  { event := event145916
    frameStart := 145845 },
  { event := event145917
    frameStart := 145845 },
  { event := event145918
    frameStart := 145845 },
  { event := event145919
    frameStart := 145845 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events569
