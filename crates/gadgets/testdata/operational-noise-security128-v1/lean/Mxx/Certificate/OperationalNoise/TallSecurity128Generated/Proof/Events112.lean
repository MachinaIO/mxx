import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events112

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event28672 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38501⟩⟩) (.authority (.programFamilyFact))

def event28673 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38501⟩⟩) (.finite 3720)

def event28674 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event28675 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38502⟩⟩) 0 ⟨7177⟩ 28674

def event28676 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38502⟩⟩) 1 ⟨38501⟩ 28673

def event28677 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38502⟩⟩) (.authority (.operator))

def exact28678RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38502⟩⟩]⟩, (1)⟩]

theorem exact28678RawTermsValid :
    exact28678RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28678 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38502⟩⟩) exact28678RawTerms .large 28677 .exactZero (none)

def event28679 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39085⟩⟩) 0 ⟨38502⟩ 28678

def event28680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39085⟩⟩) (.authority (.operator))

def exact28681RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39085⟩⟩]⟩, (1)⟩]

theorem exact28681RawTermsValid :
    exact28681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28681 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39085⟩⟩) exact28681RawTerms (.finite 8192) 28680 .exactZero (none)

def event28682 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event28683 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event28684 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38750⟩⟩) 0 ⟨37359⟩ 28670

def event28685 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38750⟩⟩) 1 ⟨136⟩ 28683

def event28686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38750⟩⟩) (.sum [.predecessor 0 28684 .coefficient, .predecessor 1 28685 .coefficient])

def event28687 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38750⟩⟩) (.finite 42)

def event28688 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38751⟩⟩) 0 ⟨38750⟩ 28687

def event28689 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38751⟩⟩) (.identity (.predecessor 0 28688 .coefficient))

def exact28690RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37358⟩⟩], []⟩, (1)⟩]

theorem exact28690RawTermsValid :
    exact28690RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28690 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38751⟩⟩) exact28690RawTerms (.finite 42) 28689 .exactZero (none)

def event28691 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact28692RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact28692RawTermsValid :
    exact28692RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28692 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact28692RawTerms .large 28691 .exactZero (none)

def event28693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38752⟩⟩) 0 ⟨6908⟩ 28692

def event28694 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38752⟩⟩) 1 ⟨38751⟩ 28690

def event28695 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38752⟩⟩) (.product (.predecessor 0 28693 .coefficient) (.predecessor 1 28694 .coefficient) (⟨false, false, none, none, none⟩))

def event28696 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38752⟩⟩, .operator (⟨28692, 0⟩, ⟨28690, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact28697RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact28697RawTermsValid :
    exact28697RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28697 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38752⟩⟩) exact28697RawTerms .large 28695 .exactZero (none)

def event28698 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7192⟩⟩) 0 ⟨7177⟩ 28674

def event28699 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7192⟩⟩) (.authority (.operator))

def exact28700RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩]

theorem exact28700RawTermsValid :
    exact28700RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28700 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7192⟩⟩) exact28700RawTerms .large 28699 .exactZero (none)

def event28701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38753⟩⟩) 0 ⟨7192⟩ 28700

def event28702 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38753⟩⟩) 1 ⟨38752⟩ 28697

def event28703 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38753⟩⟩) (.sum [.predecessor 0 28701 .coefficient, .predecessor 1 28702 .coefficient])

def exact28704RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28704RawTermsValid :
    exact28704RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28704 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38753⟩⟩) exact28704RawTerms .large 28703 .exactZero (none)

def event28705 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39086⟩⟩) 0 ⟨38753⟩ 28704

def event28706 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39086⟩⟩) 1 ⟨39085⟩ 28681

def event28707 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39086⟩⟩) (.product (.predecessor 0 28705 .coefficient) (.predecessor 1 28706 .coefficient) (⟨false, false, none, none, none⟩))

def event28708 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39086⟩⟩, .operator (⟨28704, 1⟩, ⟨28681, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39085⟩⟩]⟩, (-1)⟩)

def event28709 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39086⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39085⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39085⟩⟩) ⟨38502⟩ 28678)

def event28710 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39086⟩⟩, .relation 28709 0, ⟨[⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨38502⟩⟩]⟩, (-1)⟩)

def event28711 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39086⟩⟩, .operator (⟨28704, 0⟩, ⟨28681, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39085⟩⟩]⟩, (1)⟩)

def exact28712RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39085⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨38502⟩⟩]⟩, (-1)⟩]

theorem exact28712RawTermsValid :
    exact28712RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28712 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39086⟩⟩) exact28712RawTerms .large 28707 .exactZero (none)

def event28713 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37525⟩⟩) 0 ⟨37359⟩ 28670

def event28714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37525⟩⟩) (.authority (.programFamilyFact))

def exact28715RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37525⟩⟩], []⟩, (1)⟩]

theorem exact28715RawTermsValid :
    exact28715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28715 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37525⟩⟩) exact28715RawTerms (.finite 42) 28714 .exactZero (none)

def event28716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37527⟩⟩) 0 ⟨6908⟩ 28692

def event28717 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37527⟩⟩) 1 ⟨37525⟩ 28715

def event28718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37527⟩⟩) (.product (.predecessor 0 28716 .coefficient) (.predecessor 1 28717 .coefficient) (⟨false, true, none, none, some 1⟩))

def event28719 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37527⟩⟩, .operator (⟨28692, 0⟩, ⟨28715, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact28720RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact28720RawTermsValid :
    exact28720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28720 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37527⟩⟩) exact28720RawTerms .large 28718 .exactZero (none)

def event28721 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7223⟩⟩) 0 ⟨7177⟩ 28674

def event28722 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7223⟩⟩) (.authority (.operator))

def exact28723RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩]

theorem exact28723RawTermsValid :
    exact28723RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28723 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7223⟩⟩) exact28723RawTerms .large 28722 .exactZero (none)

def event28724 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37528⟩⟩) 0 ⟨7223⟩ 28723

def event28725 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37528⟩⟩) 1 ⟨37527⟩ 28720

def event28726 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37528⟩⟩) (.sum [.predecessor 0 28724 .coefficient, .predecessor 1 28725 .coefficient])

def exact28727RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28727RawTermsValid :
    exact28727RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28727 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37528⟩⟩) exact28727RawTerms .large 28726 .exactZero (none)

def event28728 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39090⟩⟩) 0 ⟨37528⟩ 28727

def event28729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39090⟩⟩) 1 ⟨39086⟩ 28712

def event28730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39090⟩⟩) (.sum [.predecessor 0 28728 .coefficient, .predecessor 1 28729 .coefficient])

def exact28731RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39085⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨38502⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28731RawTermsValid :
    exact28731RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28731 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39090⟩⟩) exact28731RawTerms .large 28730 .exactZero (none)

def event28732 : Event := .preFoldPolynomial 28731 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39085⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨38502⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact28733RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39085⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨38502⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event28733 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨39090⟩⟩) 28732 exact28733RawTerms .large 28730 .exactZero (none)

def event28734 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨37359⟩⟩) ⟨⟨102⟩, ⟨84⟩, ⟨135⟩⟩ ⟨28576, 28734⟩

def event28735 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨38001⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37998⟩⟩]⟩) (1) 0 2 (.universal 28734 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37998⟩⟩]⟩) (none) 28733)

def event28736 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38001⟩⟩, .relation 28735 1, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩)

def event28737 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38001⟩⟩, .relation 28735 2, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨38502⟩⟩]⟩, (1)⟩)

def event28738 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38001⟩⟩, .relation 28735 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39085⟩⟩]⟩, (-1)⟩)

def event28739 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38001⟩⟩, .relation 28735 3, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact28740RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39085⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨38502⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28740RawTermsValid :
    exact28740RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28740 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38001⟩⟩) exact28740RawTerms .large 28572 (.finite 202072841853861888) (some (28574))

def event28741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39088⟩⟩) 0 ⟨38001⟩ 28740

def event28742 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39088⟩⟩) 1 ⟨39087⟩ 28562

def event28743 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39088⟩⟩) (.sum [.predecessor 0 28741 .coefficient, .predecessor 1 28742 .coefficient])

def event28744 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39088⟩⟩, .operator (⟨28740, 2⟩, ⟨28562, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨38502⟩⟩]⟩, (-1)⟩)

def event28745 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39088⟩⟩, .operator (⟨28740, 0⟩, ⟨28562, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39085⟩⟩]⟩, (1)⟩)

def event28746 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39088⟩⟩) (.sum [.result 28740 .summary, .result 28562 .summary])

def exact28747RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28747RawTermsValid :
    exact28747RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28747 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39088⟩⟩) exact28747RawTerms .large 28743 (.finite 32192736221397454434328420548608) (some (28746))

def event28748 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39089⟩⟩) 0 ⟨39088⟩ 28747

def event28749 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39089⟩⟩) 1 ⟨7162⟩ 15622

def event28750 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39089⟩⟩) (.product (.predecessor 0 28748 .coefficient) (.predecessor 1 28749 .coefficient) (⟨false, false, none, none, none⟩))

def event28751 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39089⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩) [⟨.result 15618 .coefficient, false, none⟩])

def event28752 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39089⟩⟩) (.product (.result 28747 .summary) (.transfer 28751) (⟨false, false, none, none, none⟩))

def event28753 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39089⟩⟩, .operator (⟨28747, 0⟩, ⟨15622, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩)

def event28754 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39089⟩⟩, .operator (⟨28747, 1⟩, ⟨15622, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (-1)⟩)

def event28755 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39089⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7161⟩⟩) ⟨7046⟩ 15615)

def event28756 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39089⟩⟩, .relation 28755 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact28757RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37525⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28757RawTermsValid :
    exact28757RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28757 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39089⟩⟩) exact28757RawTerms .large 28750 (.finite 345666873099141705532726864949014345809920) (some (28752))

def event28758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35822⟩⟩) 0 ⟨7177⟩ 15500

def event28759 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35822⟩⟩) 1 ⟨35821⟩ 19557

def event28760 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35822⟩⟩) (.authority (.operator))

def exact28761RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35822⟩⟩]⟩, (1)⟩]

theorem exact28761RawTermsValid :
    exact28761RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28761 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35822⟩⟩) exact28761RawTerms .large 28760 .exactZero (none)

def event28762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36405⟩⟩) 0 ⟨35822⟩ 28761

def event28763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36405⟩⟩) (.authority (.operator))

def exact28764RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨36405⟩⟩]⟩, (1)⟩]

theorem exact28764RawTermsValid :
    exact28764RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28764 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36405⟩⟩) exact28764RawTerms (.finite 8192) 28763 .exactZero (none)

def event28765 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36407⟩⟩) 0 ⟨36165⟩ 19860

def event28766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36407⟩⟩) 1 ⟨36405⟩ 28764

def event28767 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36407⟩⟩) (.product (.predecessor 0 28765 .coefficient) (.predecessor 1 28766 .coefficient) (⟨false, false, none, none, none⟩))

def event28768 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36407⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨36405⟩⟩]⟩) [⟨.result 28764 .coefficient, false, none⟩])

def event28769 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36407⟩⟩) (.product (.result 19860 .summary) (.transfer 28768) (⟨false, false, none, none, none⟩))

def event28770 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36407⟩⟩, .operator (⟨19860, 1⟩, ⟨28764, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36405⟩⟩]⟩, (-1)⟩)

def event28771 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36407⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36405⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36405⟩⟩) ⟨35822⟩ 28761)

def event28772 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36407⟩⟩, .relation 28771 0, ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34678⟩⟩], [⟨.program ⟨257⟩, ⟨35822⟩⟩]⟩, (-1)⟩)

def event28773 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36407⟩⟩, .operator (⟨19860, 0⟩, ⟨28764, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36405⟩⟩]⟩, (1)⟩)

def exact28774RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36405⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34678⟩⟩], [⟨.program ⟨257⟩, ⟨35822⟩⟩]⟩, (-1)⟩]

theorem exact28774RawTermsValid :
    exact28774RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28774 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36407⟩⟩) exact28774RawTerms .large 28767 (.finite 32192539770951564984245676933120) (some (28769))

def event28775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35318⟩⟩) 0 ⟨34679⟩ 183

def event28776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35318⟩⟩) (.authority (.relationPreimageSource ⟨82⟩))

def exact28777RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35318⟩⟩]⟩, (1)⟩]

theorem exact28777RawTermsValid :
    exact28777RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28777 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35318⟩⟩) exact28777RawTerms (.finite 5647228698) 28776 .exactZero (none)

def event28778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35320⟩⟩) 0 ⟨35318⟩ 28777

def event28779 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35320⟩⟩) 1 ⟨2370⟩ 4

def event28780 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35320⟩⟩) (.scale (.predecessor 0 28778 .coefficient) (.value (.predecessor 1 28779 .coefficient)))

def exact28781RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35318⟩⟩]⟩, (1)⟩]

theorem exact28781RawTermsValid :
    exact28781RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28781 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35320⟩⟩) exact28781RawTerms (.finite 5647228698) 28780 .exactZero (none)

def event28782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35321⟩⟩) 0 ⟨5443⟩ 17169

def event28783 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35321⟩⟩) 1 ⟨35320⟩ 28781

def event28784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35321⟩⟩) (.product (.predecessor 0 28782 .coefficient) (.predecessor 1 28783 .coefficient) (⟨false, false, none, none, none⟩))

def event28785 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35321⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨35318⟩⟩]⟩) [⟨.result 28777 .coefficient, false, none⟩])

def event28786 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35321⟩⟩) (.product (.result 17169 .summary) (.transfer 28785) (⟨false, false, none, none, none⟩))

def event28787 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35321⟩⟩, .operator (⟨17169, 0⟩, ⟨28781, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35318⟩⟩]⟩, (1)⟩)

def event28788 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨35319⟩⟩)

def event28789 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event28790 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event28791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨140⟩⟩) (.authority (.operator))

def event28792 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨140⟩⟩) (.finite 19)

def event28793 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event28794 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event28795 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event28796 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event28797 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 28796

def event28798 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 28794

def event28799 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 28797 .coefficient) (.value (.predecessor 1 28798 .coefficient)))

def event28800 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event28801 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 0 ⟨392⟩ 28800

def event28802 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 1 ⟨140⟩ 28792

def event28803 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨393⟩⟩) (.sum [.predecessor 0 28801 .coefficient, .predecessor 1 28802 .coefficient])

def event28804 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨393⟩⟩) (.finite 655359)

def event28805 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 0 ⟨393⟩ 28804

def event28806 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 1 ⟨5426⟩ 28790

def event28807 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.identity (.predecessor 1 28806 .coefficient))

def event28808 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.finite 655360)

def event28809 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34226⟩⟩) 0 ⟨5439⟩ 28808

def event28810 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34226⟩⟩) (.authority (.programFamilyFact))

def exact28811RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34226⟩⟩], []⟩, (1)⟩]

theorem exact28811RawTermsValid :
    exact28811RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28811 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34226⟩⟩) exact28811RawTerms (.finite 40) 28810 .exactZero (none)

def event28812 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13451⟩⟩) 0 ⟨5439⟩ 28808

def event28813 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13451⟩⟩) (.authority (.programFamilyFact))

def exact28814RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13451⟩⟩], []⟩, (1)⟩]

theorem exact28814RawTermsValid :
    exact28814RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28814 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13451⟩⟩) exact28814RawTerms (.finite 40) 28813 .exactZero (none)

def event28815 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34227⟩⟩) 0 ⟨13451⟩ 28814

def event28816 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34227⟩⟩) 1 ⟨34226⟩ 28811

def event28817 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34227⟩⟩) (.product (.predecessor 0 28815 .coefficient) (.predecessor 1 28816 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event28818 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34227⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13451⟩⟩, ⟨.program ⟨257⟩, ⟨34226⟩⟩], []⟩) [⟨.result 28814 .coefficient, true, some 1⟩, ⟨.result 28811 .coefficient, true, some 1⟩])

def event28819 : Event := .survivorFold (1) 28818

def exact28820RawTerms : List Term := []

theorem exact28820RawTermsValid :
    exact28820RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28820 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34227⟩⟩) exact28820RawTerms (.finite 1600) 28817 (.finite 1600) (some (28818))

def event28821 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34228⟩⟩) 0 ⟨34227⟩ 28820

def event28822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34228⟩⟩) (.identity (.predecessor 0 28821 .coefficient))

def event28823 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34228⟩⟩) (.finite 1600)

def event28824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34678⟩⟩) 0 ⟨34228⟩ 28823

def event28825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34678⟩⟩) (.authority (.programFamilyFact))

def exact28826RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34678⟩⟩], []⟩, (1)⟩]

theorem exact28826RawTermsValid :
    exact28826RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28826 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34678⟩⟩) exact28826RawTerms (.finite 40) 28825 .exactZero (none)

def event28827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34679⟩⟩) 0 ⟨34678⟩ 28826

def event28828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34679⟩⟩) (.identity (.predecessor 0 28827 .coefficient))

def event28829 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34679⟩⟩) (.finite 40)

def event28830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35318⟩⟩) 0 ⟨34679⟩ 28829

def event28831 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35318⟩⟩) (.authority (.relationPreimageSource ⟨82⟩))

def exact28832RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35318⟩⟩]⟩, (1)⟩]

theorem exact28832RawTermsValid :
    exact28832RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28832 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35318⟩⟩) exact28832RawTerms (.finite 5647228698) 28831 .exactZero (none)

def event28833 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact28834RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact28834RawTermsValid :
    exact28834RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28834 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact28834RawTerms .large 28833 .exactZero (none)

def event28835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35319⟩⟩) 0 ⟨35⟩ 28834

def event28836 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35319⟩⟩) 1 ⟨35318⟩ 28832

def event28837 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35319⟩⟩) (.product (.predecessor 0 28835 .coefficient) (.predecessor 1 28836 .coefficient) (⟨false, false, none, none, none⟩))

def event28838 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35319⟩⟩, .operator (⟨28834, 0⟩, ⟨28832, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35318⟩⟩]⟩, (1)⟩)

def exact28839RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35318⟩⟩]⟩, (1)⟩]

theorem exact28839RawTermsValid :
    exact28839RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28839 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35319⟩⟩) exact28839RawTerms .large 28837 .exactZero (none)

def event28840 : Event := .preFoldPolynomial 28839 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35318⟩⟩]⟩, (1)⟩] .exactZero none

def exact28841RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35318⟩⟩]⟩, (1)⟩]

def event28841 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨35319⟩⟩) 28840 exact28841RawTerms .large 28837 .exactZero (none)

def event28842 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨36410⟩⟩)

def event28843 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event28844 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event28845 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨140⟩⟩) (.authority (.operator))

def event28846 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨140⟩⟩) (.finite 19)

def event28847 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event28848 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event28849 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event28850 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event28851 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 28850

def event28852 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 28848

def event28853 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 28851 .coefficient) (.value (.predecessor 1 28852 .coefficient)))

def event28854 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event28855 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 0 ⟨392⟩ 28854

def event28856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨393⟩⟩) 1 ⟨140⟩ 28846

def event28857 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨393⟩⟩) (.sum [.predecessor 0 28855 .coefficient, .predecessor 1 28856 .coefficient])

def event28858 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨393⟩⟩) (.finite 655359)

def event28859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 0 ⟨393⟩ 28858

def event28860 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5439⟩⟩) 1 ⟨5426⟩ 28844

def event28861 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.identity (.predecessor 1 28860 .coefficient))

def event28862 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5439⟩⟩) (.finite 655360)

def event28863 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34226⟩⟩) 0 ⟨5439⟩ 28862

def event28864 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34226⟩⟩) (.authority (.programFamilyFact))

def exact28865RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34226⟩⟩], []⟩, (1)⟩]

theorem exact28865RawTermsValid :
    exact28865RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28865 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34226⟩⟩) exact28865RawTerms (.finite 40) 28864 .exactZero (none)

def event28866 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13451⟩⟩) 0 ⟨5439⟩ 28862

def event28867 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13451⟩⟩) (.authority (.programFamilyFact))

def exact28868RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13451⟩⟩], []⟩, (1)⟩]

theorem exact28868RawTermsValid :
    exact28868RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28868 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13451⟩⟩) exact28868RawTerms (.finite 40) 28867 .exactZero (none)

def event28869 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34227⟩⟩) 0 ⟨13451⟩ 28868

def event28870 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34227⟩⟩) 1 ⟨34226⟩ 28865

def event28871 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34227⟩⟩) (.product (.predecessor 0 28869 .coefficient) (.predecessor 1 28870 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event28872 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34227⟩⟩, .operator (⟨28868, 0⟩, ⟨28865, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13451⟩⟩, ⟨.program ⟨257⟩, ⟨34226⟩⟩], []⟩, (1)⟩)

def exact28873RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13451⟩⟩, ⟨.program ⟨257⟩, ⟨34226⟩⟩], []⟩, (1)⟩]

theorem exact28873RawTermsValid :
    exact28873RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28873 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34227⟩⟩) exact28873RawTerms (.finite 1600) 28871 .exactZero (none)

def event28874 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34228⟩⟩) 0 ⟨34227⟩ 28873

def event28875 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34228⟩⟩) (.identity (.predecessor 0 28874 .coefficient))

def event28876 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34228⟩⟩) (.finite 1600)

def event28877 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34678⟩⟩) 0 ⟨34228⟩ 28876

def event28878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34678⟩⟩) (.authority (.programFamilyFact))

def exact28879RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34678⟩⟩], []⟩, (1)⟩]

theorem exact28879RawTermsValid :
    exact28879RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28879 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34678⟩⟩) exact28879RawTerms (.finite 40) 28878 .exactZero (none)

def event28880 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34679⟩⟩) 0 ⟨34678⟩ 28879

def event28881 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34679⟩⟩) (.identity (.predecessor 0 28880 .coefficient))

def event28882 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨34679⟩⟩) (.finite 40)

def event28883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35821⟩⟩) 0 ⟨34679⟩ 28882

def event28884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35821⟩⟩) (.authority (.programFamilyFact))

def event28885 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨35821⟩⟩) (.finite 3720)

def event28886 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event28887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35822⟩⟩) 0 ⟨7177⟩ 28886

def event28888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35822⟩⟩) 1 ⟨35821⟩ 28885

def event28889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35822⟩⟩) (.authority (.operator))

def exact28890RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35822⟩⟩]⟩, (1)⟩]

theorem exact28890RawTermsValid :
    exact28890RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28890 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35822⟩⟩) exact28890RawTerms .large 28889 .exactZero (none)

def event28891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36405⟩⟩) 0 ⟨35822⟩ 28890

def event28892 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36405⟩⟩) (.authority (.operator))

def exact28893RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨36405⟩⟩]⟩, (1)⟩]

theorem exact28893RawTermsValid :
    exact28893RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28893 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36405⟩⟩) exact28893RawTerms (.finite 8192) 28892 .exactZero (none)

def event28894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event28895 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event28896 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36070⟩⟩) 0 ⟨34679⟩ 28882

def event28897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36070⟩⟩) 1 ⟨136⟩ 28895

def event28898 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36070⟩⟩) (.sum [.predecessor 0 28896 .coefficient, .predecessor 1 28897 .coefficient])

def event28899 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨36070⟩⟩) (.finite 40)

def event28900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36071⟩⟩) 0 ⟨36070⟩ 28899

def event28901 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36071⟩⟩) (.identity (.predecessor 0 28900 .coefficient))

def exact28902RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34678⟩⟩], []⟩, (1)⟩]

theorem exact28902RawTermsValid :
    exact28902RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28902 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36071⟩⟩) exact28902RawTerms (.finite 40) 28901 .exactZero (none)

def event28903 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact28904RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact28904RawTermsValid :
    exact28904RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28904 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact28904RawTerms .large 28903 .exactZero (none)

def event28905 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36072⟩⟩) 0 ⟨6908⟩ 28904

def event28906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36072⟩⟩) 1 ⟨36071⟩ 28902

def event28907 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36072⟩⟩) (.product (.predecessor 0 28905 .coefficient) (.predecessor 1 28906 .coefficient) (⟨false, false, none, none, none⟩))

def event28908 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36072⟩⟩, .operator (⟨28904, 0⟩, ⟨28902, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨34678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact28909RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact28909RawTermsValid :
    exact28909RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28909 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36072⟩⟩) exact28909RawTerms .large 28907 .exactZero (none)

def event28910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7191⟩⟩) 0 ⟨7177⟩ 28886

def event28911 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7191⟩⟩) (.authority (.operator))

def exact28912RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩]

theorem exact28912RawTermsValid :
    exact28912RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28912 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7191⟩⟩) exact28912RawTerms .large 28911 .exactZero (none)

def event28913 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36073⟩⟩) 0 ⟨7191⟩ 28912

def event28914 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36073⟩⟩) 1 ⟨36072⟩ 28909

def event28915 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36073⟩⟩) (.sum [.predecessor 0 28913 .coefficient, .predecessor 1 28914 .coefficient])

def exact28916RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact28916RawTermsValid :
    exact28916RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28916 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36073⟩⟩) exact28916RawTerms .large 28915 .exactZero (none)

def event28917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36406⟩⟩) 0 ⟨36073⟩ 28916

def event28918 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36406⟩⟩) 1 ⟨36405⟩ 28893

def event28919 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36406⟩⟩) (.product (.predecessor 0 28917 .coefficient) (.predecessor 1 28918 .coefficient) (⟨false, false, none, none, none⟩))

def event28920 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36406⟩⟩, .operator (⟨28916, 1⟩, ⟨28893, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨34678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36405⟩⟩]⟩, (-1)⟩)

def event28921 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36406⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨34678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36405⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36405⟩⟩) ⟨35822⟩ 28890)

def event28922 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36406⟩⟩, .relation 28921 0, ⟨[⟨.program ⟨257⟩, ⟨34678⟩⟩], [⟨.program ⟨257⟩, ⟨35822⟩⟩]⟩, (-1)⟩)

def event28923 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36406⟩⟩, .operator (⟨28916, 0⟩, ⟨28893, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36405⟩⟩]⟩, (1)⟩)

def exact28924RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36405⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨34678⟩⟩], [⟨.program ⟨257⟩, ⟨35822⟩⟩]⟩, (-1)⟩]

theorem exact28924RawTermsValid :
    exact28924RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28924 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36406⟩⟩) exact28924RawTerms .large 28919 .exactZero (none)

def event28925 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34845⟩⟩) 0 ⟨34679⟩ 28882

def event28926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34845⟩⟩) (.authority (.programFamilyFact))

def exact28927RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨34845⟩⟩], []⟩, (1)⟩]

theorem exact28927RawTermsValid :
    exact28927RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event28927 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34845⟩⟩) exact28927RawTerms (.finite 40) 28926 .exactZero (none)

def eventLeaf1792 : Array AnnotatedEvent := #[
  { event := event28672
    frameStart := 28630 },
  { event := event28673
    frameStart := 28630 },
  { event := event28674
    frameStart := 28630 },
  { event := event28675
    frameStart := 28630 },
  { event := event28676
    frameStart := 28630 },
  { event := event28677
    frameStart := 28630 },
  { event := event28678
    frameStart := 28630 },
  { event := event28679
    frameStart := 28630 },
  { event := event28680
    frameStart := 28630 },
  { event := event28681
    frameStart := 28630 },
  { event := event28682
    frameStart := 28630 },
  { event := event28683
    frameStart := 28630 },
  { event := event28684
    frameStart := 28630 },
  { event := event28685
    frameStart := 28630 },
  { event := event28686
    frameStart := 28630 },
  { event := event28687
    frameStart := 28630 }
]

def eventLeaf1793 : Array AnnotatedEvent := #[
  { event := event28688
    frameStart := 28630 },
  { event := event28689
    frameStart := 28630 },
  { event := event28690
    frameStart := 28630 },
  { event := event28691
    frameStart := 28630 },
  { event := event28692
    frameStart := 28630 },
  { event := event28693
    frameStart := 28630 },
  { event := event28694
    frameStart := 28630 },
  { event := event28695
    frameStart := 28630 },
  { event := event28696
    frameStart := 28630 },
  { event := event28697
    frameStart := 28630 },
  { event := event28698
    frameStart := 28630 },
  { event := event28699
    frameStart := 28630 },
  { event := event28700
    frameStart := 28630 },
  { event := event28701
    frameStart := 28630 },
  { event := event28702
    frameStart := 28630 },
  { event := event28703
    frameStart := 28630 }
]

def eventLeaf1794 : Array AnnotatedEvent := #[
  { event := event28704
    frameStart := 28630 },
  { event := event28705
    frameStart := 28630 },
  { event := event28706
    frameStart := 28630 },
  { event := event28707
    frameStart := 28630 },
  { event := event28708
    frameStart := 28630 },
  { event := event28709
    frameStart := 28630 },
  { event := event28710
    frameStart := 28630 },
  { event := event28711
    frameStart := 28630 },
  { event := event28712
    frameStart := 28630 },
  { event := event28713
    frameStart := 28630 },
  { event := event28714
    frameStart := 28630 },
  { event := event28715
    frameStart := 28630 },
  { event := event28716
    frameStart := 28630 },
  { event := event28717
    frameStart := 28630 },
  { event := event28718
    frameStart := 28630 },
  { event := event28719
    frameStart := 28630 }
]

def eventLeaf1795 : Array AnnotatedEvent := #[
  { event := event28720
    frameStart := 28630 },
  { event := event28721
    frameStart := 28630 },
  { event := event28722
    frameStart := 28630 },
  { event := event28723
    frameStart := 28630 },
  { event := event28724
    frameStart := 28630 },
  { event := event28725
    frameStart := 28630 },
  { event := event28726
    frameStart := 28630 },
  { event := event28727
    frameStart := 28630 },
  { event := event28728
    frameStart := 28630 },
  { event := event28729
    frameStart := 28630 },
  { event := event28730
    frameStart := 28630 },
  { event := event28731
    frameStart := 28630 },
  { event := event28732
    frameStart := 28630 },
  { event := event28733
    frameStart := 28630 },
  { event := event28734
    frameStart := 0 },
  { event := event28735
    frameStart := 0 }
]

def eventLeaf1796 : Array AnnotatedEvent := #[
  { event := event28736
    frameStart := 0 },
  { event := event28737
    frameStart := 0 },
  { event := event28738
    frameStart := 0 },
  { event := event28739
    frameStart := 0 },
  { event := event28740
    frameStart := 0 },
  { event := event28741
    frameStart := 0 },
  { event := event28742
    frameStart := 0 },
  { event := event28743
    frameStart := 0 },
  { event := event28744
    frameStart := 0 },
  { event := event28745
    frameStart := 0 },
  { event := event28746
    frameStart := 0 },
  { event := event28747
    frameStart := 0 },
  { event := event28748
    frameStart := 0 },
  { event := event28749
    frameStart := 0 },
  { event := event28750
    frameStart := 0 },
  { event := event28751
    frameStart := 0 }
]

def eventLeaf1797 : Array AnnotatedEvent := #[
  { event := event28752
    frameStart := 0 },
  { event := event28753
    frameStart := 0 },
  { event := event28754
    frameStart := 0 },
  { event := event28755
    frameStart := 0 },
  { event := event28756
    frameStart := 0 },
  { event := event28757
    frameStart := 0 },
  { event := event28758
    frameStart := 0 },
  { event := event28759
    frameStart := 0 },
  { event := event28760
    frameStart := 0 },
  { event := event28761
    frameStart := 0 },
  { event := event28762
    frameStart := 0 },
  { event := event28763
    frameStart := 0 },
  { event := event28764
    frameStart := 0 },
  { event := event28765
    frameStart := 0 },
  { event := event28766
    frameStart := 0 },
  { event := event28767
    frameStart := 0 }
]

def eventLeaf1798 : Array AnnotatedEvent := #[
  { event := event28768
    frameStart := 0 },
  { event := event28769
    frameStart := 0 },
  { event := event28770
    frameStart := 0 },
  { event := event28771
    frameStart := 0 },
  { event := event28772
    frameStart := 0 },
  { event := event28773
    frameStart := 0 },
  { event := event28774
    frameStart := 0 },
  { event := event28775
    frameStart := 0 },
  { event := event28776
    frameStart := 0 },
  { event := event28777
    frameStart := 0 },
  { event := event28778
    frameStart := 0 },
  { event := event28779
    frameStart := 0 },
  { event := event28780
    frameStart := 0 },
  { event := event28781
    frameStart := 0 },
  { event := event28782
    frameStart := 0 },
  { event := event28783
    frameStart := 0 }
]

def eventLeaf1799 : Array AnnotatedEvent := #[
  { event := event28784
    frameStart := 0 },
  { event := event28785
    frameStart := 0 },
  { event := event28786
    frameStart := 0 },
  { event := event28787
    frameStart := 0 },
  { event := event28788
    frameStart := 28788 },
  { event := event28789
    frameStart := 28788 },
  { event := event28790
    frameStart := 28788 },
  { event := event28791
    frameStart := 28788 },
  { event := event28792
    frameStart := 28788 },
  { event := event28793
    frameStart := 28788 },
  { event := event28794
    frameStart := 28788 },
  { event := event28795
    frameStart := 28788 },
  { event := event28796
    frameStart := 28788 },
  { event := event28797
    frameStart := 28788 },
  { event := event28798
    frameStart := 28788 },
  { event := event28799
    frameStart := 28788 }
]

def eventLeaf1800 : Array AnnotatedEvent := #[
  { event := event28800
    frameStart := 28788 },
  { event := event28801
    frameStart := 28788 },
  { event := event28802
    frameStart := 28788 },
  { event := event28803
    frameStart := 28788 },
  { event := event28804
    frameStart := 28788 },
  { event := event28805
    frameStart := 28788 },
  { event := event28806
    frameStart := 28788 },
  { event := event28807
    frameStart := 28788 },
  { event := event28808
    frameStart := 28788 },
  { event := event28809
    frameStart := 28788 },
  { event := event28810
    frameStart := 28788 },
  { event := event28811
    frameStart := 28788 },
  { event := event28812
    frameStart := 28788 },
  { event := event28813
    frameStart := 28788 },
  { event := event28814
    frameStart := 28788 },
  { event := event28815
    frameStart := 28788 }
]

def eventLeaf1801 : Array AnnotatedEvent := #[
  { event := event28816
    frameStart := 28788 },
  { event := event28817
    frameStart := 28788 },
  { event := event28818
    frameStart := 28788 },
  { event := event28819
    frameStart := 28788 },
  { event := event28820
    frameStart := 28788 },
  { event := event28821
    frameStart := 28788 },
  { event := event28822
    frameStart := 28788 },
  { event := event28823
    frameStart := 28788 },
  { event := event28824
    frameStart := 28788 },
  { event := event28825
    frameStart := 28788 },
  { event := event28826
    frameStart := 28788 },
  { event := event28827
    frameStart := 28788 },
  { event := event28828
    frameStart := 28788 },
  { event := event28829
    frameStart := 28788 },
  { event := event28830
    frameStart := 28788 },
  { event := event28831
    frameStart := 28788 }
]

def eventLeaf1802 : Array AnnotatedEvent := #[
  { event := event28832
    frameStart := 28788 },
  { event := event28833
    frameStart := 28788 },
  { event := event28834
    frameStart := 28788 },
  { event := event28835
    frameStart := 28788 },
  { event := event28836
    frameStart := 28788 },
  { event := event28837
    frameStart := 28788 },
  { event := event28838
    frameStart := 28788 },
  { event := event28839
    frameStart := 28788 },
  { event := event28840
    frameStart := 28788 },
  { event := event28841
    frameStart := 28788 },
  { event := event28842
    frameStart := 28842 },
  { event := event28843
    frameStart := 28842 },
  { event := event28844
    frameStart := 28842 },
  { event := event28845
    frameStart := 28842 },
  { event := event28846
    frameStart := 28842 },
  { event := event28847
    frameStart := 28842 }
]

def eventLeaf1803 : Array AnnotatedEvent := #[
  { event := event28848
    frameStart := 28842 },
  { event := event28849
    frameStart := 28842 },
  { event := event28850
    frameStart := 28842 },
  { event := event28851
    frameStart := 28842 },
  { event := event28852
    frameStart := 28842 },
  { event := event28853
    frameStart := 28842 },
  { event := event28854
    frameStart := 28842 },
  { event := event28855
    frameStart := 28842 },
  { event := event28856
    frameStart := 28842 },
  { event := event28857
    frameStart := 28842 },
  { event := event28858
    frameStart := 28842 },
  { event := event28859
    frameStart := 28842 },
  { event := event28860
    frameStart := 28842 },
  { event := event28861
    frameStart := 28842 },
  { event := event28862
    frameStart := 28842 },
  { event := event28863
    frameStart := 28842 }
]

def eventLeaf1804 : Array AnnotatedEvent := #[
  { event := event28864
    frameStart := 28842 },
  { event := event28865
    frameStart := 28842 },
  { event := event28866
    frameStart := 28842 },
  { event := event28867
    frameStart := 28842 },
  { event := event28868
    frameStart := 28842 },
  { event := event28869
    frameStart := 28842 },
  { event := event28870
    frameStart := 28842 },
  { event := event28871
    frameStart := 28842 },
  { event := event28872
    frameStart := 28842 },
  { event := event28873
    frameStart := 28842 },
  { event := event28874
    frameStart := 28842 },
  { event := event28875
    frameStart := 28842 },
  { event := event28876
    frameStart := 28842 },
  { event := event28877
    frameStart := 28842 },
  { event := event28878
    frameStart := 28842 },
  { event := event28879
    frameStart := 28842 }
]

def eventLeaf1805 : Array AnnotatedEvent := #[
  { event := event28880
    frameStart := 28842 },
  { event := event28881
    frameStart := 28842 },
  { event := event28882
    frameStart := 28842 },
  { event := event28883
    frameStart := 28842 },
  { event := event28884
    frameStart := 28842 },
  { event := event28885
    frameStart := 28842 },
  { event := event28886
    frameStart := 28842 },
  { event := event28887
    frameStart := 28842 },
  { event := event28888
    frameStart := 28842 },
  { event := event28889
    frameStart := 28842 },
  { event := event28890
    frameStart := 28842 },
  { event := event28891
    frameStart := 28842 },
  { event := event28892
    frameStart := 28842 },
  { event := event28893
    frameStart := 28842 },
  { event := event28894
    frameStart := 28842 },
  { event := event28895
    frameStart := 28842 }
]

def eventLeaf1806 : Array AnnotatedEvent := #[
  { event := event28896
    frameStart := 28842 },
  { event := event28897
    frameStart := 28842 },
  { event := event28898
    frameStart := 28842 },
  { event := event28899
    frameStart := 28842 },
  { event := event28900
    frameStart := 28842 },
  { event := event28901
    frameStart := 28842 },
  { event := event28902
    frameStart := 28842 },
  { event := event28903
    frameStart := 28842 },
  { event := event28904
    frameStart := 28842 },
  { event := event28905
    frameStart := 28842 },
  { event := event28906
    frameStart := 28842 },
  { event := event28907
    frameStart := 28842 },
  { event := event28908
    frameStart := 28842 },
  { event := event28909
    frameStart := 28842 },
  { event := event28910
    frameStart := 28842 },
  { event := event28911
    frameStart := 28842 }
]

def eventLeaf1807 : Array AnnotatedEvent := #[
  { event := event28912
    frameStart := 28842 },
  { event := event28913
    frameStart := 28842 },
  { event := event28914
    frameStart := 28842 },
  { event := event28915
    frameStart := 28842 },
  { event := event28916
    frameStart := 28842 },
  { event := event28917
    frameStart := 28842 },
  { event := event28918
    frameStart := 28842 },
  { event := event28919
    frameStart := 28842 },
  { event := event28920
    frameStart := 28842 },
  { event := event28921
    frameStart := 28842 },
  { event := event28922
    frameStart := 28842 },
  { event := event28923
    frameStart := 28842 },
  { event := event28924
    frameStart := 28842 },
  { event := event28925
    frameStart := 28842 },
  { event := event28926
    frameStart := 28842 },
  { event := event28927
    frameStart := 28842 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events112
