import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events370

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event94720 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24781⟩⟩) (.authority (.programFamilyFact))

def event94721 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24781⟩⟩) (.finite 3720)

def event94722 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event94723 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24783⟩⟩) 0 ⟨6689⟩ 94722

def event94724 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24783⟩⟩) 1 ⟨24781⟩ 94721

def event94725 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24783⟩⟩) (.authority (.operator))

def exact94726RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24783⟩⟩]⟩, (1)⟩]

theorem exact94726RawTermsValid :
    exact94726RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94726 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24783⟩⟩) exact94726RawTerms .large 94725 .exactZero (none)

def event94727 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30061⟩⟩) 0 ⟨24783⟩ 94726

def event94728 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30061⟩⟩) (.authority (.operator))

def exact94729RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨30061⟩⟩]⟩, (1)⟩]

theorem exact94729RawTermsValid :
    exact94729RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94729 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30061⟩⟩) exact94729RawTerms (.finite 8192) 94728 .exactZero (none)

def event94730 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event94731 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event94732 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17043⟩⟩) 0 ⟨17002⟩ 94718

def event94733 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17043⟩⟩) 1 ⟨110⟩ 94731

def event94734 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17043⟩⟩) (.sum [.predecessor 0 94732 .coefficient, .predecessor 1 94733 .coefficient])

def event94735 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨17043⟩⟩) (.finite 60)

def event94736 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17044⟩⟩) 0 ⟨17043⟩ 94735

def event94737 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17044⟩⟩) (.identity (.predecessor 0 94736 .coefficient))

def exact94738RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], []⟩, (1)⟩]

theorem exact94738RawTermsValid :
    exact94738RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94738 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17044⟩⟩) exact94738RawTerms (.finite 60) 94737 .exactZero (none)

def event94739 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact94740RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact94740RawTermsValid :
    exact94740RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94740 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact94740RawTerms .large 94739 .exactZero (none)

def event94741 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17045⟩⟩) 0 ⟨6544⟩ 94740

def event94742 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17045⟩⟩) 1 ⟨17044⟩ 94738

def event94743 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17045⟩⟩) (.product (.predecessor 0 94741 .coefficient) (.predecessor 1 94742 .coefficient) (⟨false, false, none, none, none⟩))

def event94744 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨17045⟩⟩, .operator (⟨94740, 0⟩, ⟨94738, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact94745RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact94745RawTermsValid :
    exact94745RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94745 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17045⟩⟩) exact94745RawTerms .large 94743 .exactZero (none)

def event94746 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6707⟩⟩) 0 ⟨6689⟩ 94722

def event94747 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6707⟩⟩) (.authority (.operator))

def exact94748RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩]

theorem exact94748RawTermsValid :
    exact94748RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94748 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6707⟩⟩) exact94748RawTerms .large 94747 .exactZero (none)

def event94749 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17046⟩⟩) 0 ⟨6707⟩ 94748

def event94750 : Event := .predecessor (⟨.program ⟨214⟩, ⟨17046⟩⟩) 1 ⟨17045⟩ 94745

def event94751 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨17046⟩⟩) (.sum [.predecessor 0 94749 .coefficient, .predecessor 1 94750 .coefficient])

def exact94752RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact94752RawTermsValid :
    exact94752RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94752 : Event := .resultExact (⟨.program ⟨214⟩, ⟨17046⟩⟩) exact94752RawTerms .large 94751 .exactZero (none)

def event94753 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30062⟩⟩) 0 ⟨17046⟩ 94752

def event94754 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30062⟩⟩) 1 ⟨30061⟩ 94729

def event94755 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30062⟩⟩) (.product (.predecessor 0 94753 .coefficient) (.predecessor 1 94754 .coefficient) (⟨false, false, none, none, none⟩))

def event94756 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30062⟩⟩, .operator (⟨94752, 0⟩, ⟨94729, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30061⟩⟩]⟩, (1)⟩)

def event94757 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30062⟩⟩, .operator (⟨94752, 1⟩, ⟨94729, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30061⟩⟩]⟩, (-1)⟩)

def event94758 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨30062⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30061⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨30061⟩⟩) ⟨24783⟩ 94726)

def event94759 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30062⟩⟩, .relation 94758 0, ⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨24783⟩⟩]⟩, (-1)⟩)

def exact94760RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30061⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨24783⟩⟩]⟩, (-1)⟩]

theorem exact94760RawTermsValid :
    exact94760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94760 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30062⟩⟩) exact94760RawTerms .large 94755 .exactZero (none)

def event94761 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18163⟩⟩) 0 ⟨17002⟩ 94718

def event94762 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18163⟩⟩) (.authority (.programFamilyFact))

def exact94763RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], []⟩, (1)⟩]

theorem exact94763RawTermsValid :
    exact94763RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94763 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18163⟩⟩) exact94763RawTerms (.finite 63) 94762 .exactZero (none)

def event94764 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18164⟩⟩) 0 ⟨6544⟩ 94740

def event94765 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18164⟩⟩) 1 ⟨18163⟩ 94763

def event94766 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18164⟩⟩) (.product (.predecessor 0 94764 .coefficient) (.predecessor 1 94765 .coefficient) (⟨false, true, none, none, some 1⟩))

def event94767 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18164⟩⟩, .operator (⟨94740, 0⟩, ⟨94763, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact94768RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact94768RawTermsValid :
    exact94768RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94768 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18164⟩⟩) exact94768RawTerms .large 94766 .exactZero (none)

def event94769 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6743⟩⟩) 0 ⟨6689⟩ 94722

def event94770 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6743⟩⟩) (.authority (.operator))

def exact94771RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩]

theorem exact94771RawTermsValid :
    exact94771RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94771 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6743⟩⟩) exact94771RawTerms .large 94770 .exactZero (none)

def event94772 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18165⟩⟩) 0 ⟨6743⟩ 94771

def event94773 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18165⟩⟩) 1 ⟨18164⟩ 94768

def event94774 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18165⟩⟩) (.sum [.predecessor 0 94772 .coefficient, .predecessor 1 94773 .coefficient])

def exact94775RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact94775RawTermsValid :
    exact94775RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94775 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18165⟩⟩) exact94775RawTerms .large 94774 .exactZero (none)

def event94776 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30069⟩⟩) 0 ⟨18165⟩ 94775

def event94777 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30069⟩⟩) 1 ⟨30062⟩ 94760

def event94778 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30069⟩⟩) (.sum [.predecessor 0 94776 .coefficient, .predecessor 1 94777 .coefficient])

def exact94779RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30061⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨24783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact94779RawTermsValid :
    exact94779RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94779 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30069⟩⟩) exact94779RawTerms .large 94778 .exactZero (none)

def event94780 : Event := .preFoldPolynomial 94779 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30061⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨24783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact94781RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30061⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨24783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event94781 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨30069⟩⟩) 94780 exact94781RawTerms .large 94778 .exactZero (none)

def event94782 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨17002⟩⟩) ⟨⟨156⟩, ⟨65⟩, ⟨109⟩⟩ ⟨94648, 94782⟩

def event94783 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨22832⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22829⟩⟩]⟩) (1) 0 2 (.universal 94782 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22829⟩⟩]⟩) (none) 94781)

def event94784 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22832⟩⟩, .relation 94783 1, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩)

def event94785 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22832⟩⟩, .relation 94783 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30061⟩⟩]⟩, (-1)⟩)

def event94786 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22832⟩⟩, .relation 94783 2, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨24783⟩⟩]⟩, (1)⟩)

def event94787 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22832⟩⟩, .relation 94783 3, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact94788RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30061⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨24783⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact94788RawTermsValid :
    exact94788RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94788 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22832⟩⟩) exact94788RawTerms .large 94644 (.finite 1811303510016) (some (94646))

def event94789 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30064⟩⟩) 0 ⟨22832⟩ 94788

def event94790 : Event := .predecessor (⟨.program ⟨214⟩, ⟨30064⟩⟩) 1 ⟨30063⟩ 94634

def event94791 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30064⟩⟩) (.sum [.predecessor 0 94789 .coefficient, .predecessor 1 94790 .coefficient])

def event94792 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30064⟩⟩, .operator (⟨94788, 0⟩, ⟨94634, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30061⟩⟩]⟩, (1)⟩)

def event94793 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨30064⟩⟩, .operator (⟨94788, 2⟩, ⟨94634, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17001⟩⟩], [⟨.program ⟨214⟩, ⟨24783⟩⟩]⟩, (-1)⟩)

def event94794 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨30064⟩⟩) (.sum [.result 94788 .summary, .result 94634 .summary])

def exact94795RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact94795RawTermsValid :
    exact94795RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94795 : Event := .resultExact (⟨.program ⟨214⟩, ⟨30064⟩⟩) exact94795RawTerms .large 94791 (.finite 1292539135285018636288) (some (94794))

def event94796 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24718⟩⟩) 0 ⟨16862⟩ 4606

def event94797 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24718⟩⟩) (.authority (.programFamilyFact))

def event94798 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24718⟩⟩) (.finite 3720)

def event94799 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24720⟩⟩) 0 ⟨6689⟩ 5477

def event94800 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24720⟩⟩) 1 ⟨24718⟩ 94798

def event94801 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24720⟩⟩) (.authority (.operator))

def exact94802RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24720⟩⟩]⟩, (1)⟩]

theorem exact94802RawTermsValid :
    exact94802RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94802 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24720⟩⟩) exact94802RawTerms .large 94801 .exactZero (none)

def event94803 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29784⟩⟩) 0 ⟨24720⟩ 94802

def event94804 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29784⟩⟩) (.authority (.operator))

def exact94805RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨29784⟩⟩]⟩, (1)⟩]

theorem exact94805RawTermsValid :
    exact94805RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94805 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29784⟩⟩) exact94805RawTerms (.finite 8192) 94804 .exactZero (none)

def event94806 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23367⟩⟩) 0 ⟨13132⟩ 4600

def event94807 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23367⟩⟩) (.authority (.programFamilyFact))

def event94808 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23367⟩⟩) (.finite 3720)

def event94809 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23368⟩⟩) 0 ⟨6689⟩ 5477

def event94810 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23368⟩⟩) 1 ⟨23367⟩ 94808

def event94811 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23368⟩⟩) (.authority (.operator))

def exact94812RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23368⟩⟩]⟩, (1)⟩]

theorem exact94812RawTermsValid :
    exact94812RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94812 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23368⟩⟩) exact94812RawTerms .large 94811 .exactZero (none)

def event94813 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25668⟩⟩) 0 ⟨23368⟩ 94812

def event94814 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25668⟩⟩) (.authority (.operator))

def exact94815RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25668⟩⟩]⟩, (1)⟩]

theorem exact94815RawTermsValid :
    exact94815RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94815 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25668⟩⟩) exact94815RawTerms (.finite 8192) 94814 .exactZero (none)

def event94816 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13133⟩⟩) 0 ⟨13130⟩ 4589

def event94817 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13133⟩⟩) 1 ⟨6564⟩ 32

def event94818 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13133⟩⟩) (.tensor (.predecessor 0 94816 .coefficient) (.predecessor 1 94817 .coefficient) true false)

def event94819 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13133⟩⟩, .operator (⟨4589, 0⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13130⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact94820RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13130⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact94820RawTermsValid :
    exact94820RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94820 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13133⟩⟩) exact94820RawTerms .large 94818 .exactZero (none)

def event94821 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7126⟩⟩) 0 ⟨5506⟩ 27

def event94822 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7126⟩⟩) 1 ⟨6789⟩ 6973

def event94823 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7126⟩⟩) (.product (.predecessor 0 94821 .coefficient) (.predecessor 1 94822 .coefficient) (⟨false, false, none, none, none⟩))

def event94824 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7126⟩⟩, .operator (⟨27, 0⟩, ⟨6973, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (1)⟩)

def exact94825RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (1)⟩]

theorem exact94825RawTermsValid :
    exact94825RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94825 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7126⟩⟩) exact94825RawTerms .large 94823 .exactZero (none)

def event94826 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13134⟩⟩) 0 ⟨7126⟩ 94825

def event94827 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13134⟩⟩) 1 ⟨13133⟩ 94820

def event94828 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13134⟩⟩) (.sum [.predecessor 0 94826 .coefficient, .predecessor 1 94827 .coefficient])

def exact94829RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13130⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact94829RawTermsValid :
    exact94829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94829 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13134⟩⟩) exact94829RawTerms .large 94828 .exactZero (none)

def event94830 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13135⟩⟩) 0 ⟨13134⟩ 94829

def event94831 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13135⟩⟩) 1 ⟨103⟩ 6965

def event94832 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13135⟩⟩) (.sum [.predecessor 0 94830 .coefficient, .predecessor 1 94831 .coefficient])

def event94833 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13135⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨103⟩⟩]⟩) [⟨.result 6965 .coefficient, false, none⟩])

def event94834 : Event := .survivorFold (1) 94833

def exact94835RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨13130⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact94835RawTermsValid :
    exact94835RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94835 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13135⟩⟩) exact94835RawTerms .large 94832 (.finite 26) (some (94833))

def event94836 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13136⟩⟩) 0 ⟨13135⟩ 94835

def event94837 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13136⟩⟩) 1 ⟨10225⟩ 4592

def event94838 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13136⟩⟩) (.product (.predecessor 0 94836 .coefficient) (.predecessor 1 94837 .coefficient) (⟨false, true, none, none, some 1⟩))

def event94839 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13136⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10225⟩⟩], []⟩) [⟨.result 4592 .coefficient, true, some 1⟩])

def event94840 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13136⟩⟩) (.product (.result 94835 .summary) (.transfer 94839) (⟨false, false, none, none, none⟩))

def event94841 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13136⟩⟩, .operator (⟨94835, 1⟩, ⟨4592, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10225⟩⟩, ⟨.program ⟨214⟩, ⟨13130⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event94842 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13136⟩⟩, .operator (⟨94835, 0⟩, ⟨4592, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10225⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (1)⟩)

def exact94843RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10225⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10225⟩⟩, ⟨.program ⟨214⟩, ⟨13130⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact94843RawTermsValid :
    exact94843RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94843 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13136⟩⟩) exact94843RawTerms .large 94838 (.finite 48256) (some (94840))

def event94844 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10226⟩⟩) 0 ⟨10225⟩ 4592

def event94845 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10226⟩⟩) 1 ⟨6564⟩ 32

def event94846 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10226⟩⟩) (.tensor (.predecessor 0 94844 .coefficient) (.predecessor 1 94845 .coefficient) true false)

def event94847 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10226⟩⟩, .operator (⟨4592, 0⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10225⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact94848RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10225⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact94848RawTermsValid :
    exact94848RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94848 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10226⟩⟩) exact94848RawTerms .large 94846 .exactZero (none)

def event94849 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7106⟩⟩) 0 ⟨5506⟩ 27

def event94850 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7106⟩⟩) 1 ⟨6769⟩ 7014

def event94851 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7106⟩⟩) (.product (.predecessor 0 94849 .coefficient) (.predecessor 1 94850 .coefficient) (⟨false, false, none, none, none⟩))

def event94852 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7106⟩⟩, .operator (⟨27, 0⟩, ⟨7014, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩]⟩, (1)⟩)

def exact94853RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩]⟩, (1)⟩]

theorem exact94853RawTermsValid :
    exact94853RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94853 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7106⟩⟩) exact94853RawTerms .large 94851 .exactZero (none)

def event94854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10227⟩⟩) 0 ⟨7106⟩ 94853

def event94855 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10227⟩⟩) 1 ⟨10226⟩ 94848

def event94856 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10227⟩⟩) (.sum [.predecessor 0 94854 .coefficient, .predecessor 1 94855 .coefficient])

def exact94857RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10225⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact94857RawTermsValid :
    exact94857RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94857 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10227⟩⟩) exact94857RawTerms .large 94856 .exactZero (none)

def event94858 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10228⟩⟩) 0 ⟨10227⟩ 94857

def event94859 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10228⟩⟩) 1 ⟨83⟩ 7006

def event94860 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10228⟩⟩) (.sum [.predecessor 0 94858 .coefficient, .predecessor 1 94859 .coefficient])

def event94861 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10228⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨83⟩⟩]⟩) [⟨.result 7006 .coefficient, false, none⟩])

def event94862 : Event := .survivorFold (1) 94861

def exact94863RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10225⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact94863RawTermsValid :
    exact94863RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94863 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10228⟩⟩) exact94863RawTerms .large 94860 (.finite 26) (some (94861))

def event94864 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10229⟩⟩) 0 ⟨10228⟩ 94863

def event94865 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10229⟩⟩) 1 ⟨7880⟩ 7003

def event94866 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10229⟩⟩) (.product (.predecessor 0 94864 .coefficient) (.predecessor 1 94865 .coefficient) (⟨false, false, none, none, none⟩))

def event94867 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10229⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) [⟨.result 6999 .coefficient, false, none⟩])

def event94868 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10229⟩⟩) (.product (.result 94863 .summary) (.transfer 94867) (⟨false, false, none, none, none⟩))

def event94869 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10229⟩⟩, .operator (⟨94863, 1⟩, ⟨7003, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10225⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩, (-1)⟩)

def event94870 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨10229⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10225⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7879⟩⟩) ⟨6789⟩ 6973)

def event94871 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10229⟩⟩, .relation 94870 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10225⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (-1)⟩)

def event94872 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10229⟩⟩, .operator (⟨94863, 0⟩, ⟨7003, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩, (1)⟩)

def exact94873RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10225⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (-1)⟩]

theorem exact94873RawTermsValid :
    exact94873RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94873 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10229⟩⟩) exact94873RawTerms .large 94866 (.finite 95420416) (some (94868))

def event94874 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13137⟩⟩) 0 ⟨10229⟩ 94873

def event94875 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13137⟩⟩) 1 ⟨13136⟩ 94843

def event94876 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13137⟩⟩) (.sum [.predecessor 0 94874 .coefficient, .predecessor 1 94875 .coefficient])

def event94877 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13137⟩⟩, .operator (⟨94873, 1⟩, ⟨94843, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10225⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩, (1)⟩)

def event94878 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13137⟩⟩) (.sum [.result 94873 .summary, .result 94843 .summary])

def exact94879RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10225⟩⟩, ⟨.program ⟨214⟩, ⟨13130⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact94879RawTermsValid :
    exact94879RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94879 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13137⟩⟩) exact94879RawTerms .large 94876 (.finite 95468672) (some (94878))

def event94880 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25669⟩⟩) 0 ⟨13137⟩ 94879

def event94881 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25669⟩⟩) 1 ⟨25668⟩ 94815

def event94882 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25669⟩⟩) (.product (.predecessor 0 94880 .coefficient) (.predecessor 1 94881 .coefficient) (⟨false, false, none, none, none⟩))

def event94883 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25669⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25668⟩⟩]⟩) [⟨.result 94815 .coefficient, false, none⟩])

def event94884 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25669⟩⟩) (.product (.result 94879 .summary) (.transfer 94883) (⟨false, false, none, none, none⟩))

def event94885 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25669⟩⟩, .operator (⟨94879, 1⟩, ⟨94815, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10225⟩⟩, ⟨.program ⟨214⟩, ⟨13130⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25668⟩⟩]⟩, (-1)⟩)

def event94886 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25669⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10225⟩⟩, ⟨.program ⟨214⟩, ⟨13130⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25668⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25668⟩⟩) ⟨23368⟩ 94812)

def event94887 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25669⟩⟩, .relation 94886 0, ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10225⟩⟩, ⟨.program ⟨214⟩, ⟨13130⟩⟩], [⟨.program ⟨214⟩, ⟨23368⟩⟩]⟩, (-1)⟩)

def event94888 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25669⟩⟩, .operator (⟨94879, 0⟩, ⟨94815, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25668⟩⟩]⟩, (1)⟩)

def exact94889RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25668⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10225⟩⟩, ⟨.program ⟨214⟩, ⟨13130⟩⟩], [⟨.program ⟨214⟩, ⟨23368⟩⟩]⟩, (-1)⟩]

theorem exact94889RawTermsValid :
    exact94889RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94889 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25669⟩⟩) exact94889RawTerms .large 94882 (.finite 350371553738752) (some (94884))

def event94890 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20165⟩⟩) 0 ⟨13132⟩ 4600

def event94891 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20165⟩⟩) (.authority (.relationPreimageSource ⟨25⟩))

def exact94892RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20165⟩⟩]⟩, (1)⟩]

theorem exact94892RawTermsValid :
    exact94892RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94892 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20165⟩⟩) exact94892RawTerms (.finite 136065468) 94891 .exactZero (none)

def event94893 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20167⟩⟩) 0 ⟨20165⟩ 94892

def event94894 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20167⟩⟩) 1 ⟨2348⟩ 4

def event94895 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20167⟩⟩) (.scale (.predecessor 0 94893 .coefficient) (.value (.predecessor 1 94894 .coefficient)))

def exact94896RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20165⟩⟩]⟩, (1)⟩]

theorem exact94896RawTermsValid :
    exact94896RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94896 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20167⟩⟩) exact94896RawTerms (.finite 136065468) 94895 .exactZero (none)

def event94897 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20168⟩⟩) 0 ⟨5509⟩ 94462

def event94898 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20168⟩⟩) 1 ⟨20167⟩ 94896

def event94899 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20168⟩⟩) (.product (.predecessor 0 94897 .coefficient) (.predecessor 1 94898 .coefficient) (⟨false, false, none, none, none⟩))

def event94900 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20168⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20165⟩⟩]⟩) [⟨.result 94892 .coefficient, false, none⟩])

def event94901 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20168⟩⟩) (.product (.result 94462 .summary) (.transfer 94900) (⟨false, false, none, none, none⟩))

def event94902 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20168⟩⟩, .operator (⟨94462, 0⟩, ⟨94896, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20165⟩⟩]⟩, (1)⟩)

def event94903 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20166⟩⟩)

def event94904 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event94905 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event94906 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event94907 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event94908 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 94907

def event94909 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 94905

def event94910 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 94908 .coefficient) (.value (.predecessor 1 94909 .coefficient)))

def event94911 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event94912 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13130⟩⟩) 0 ⟨5503⟩ 94911

def event94913 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13130⟩⟩) (.authority (.programFamilyFact))

def exact94914RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13130⟩⟩], []⟩, (1)⟩]

theorem exact94914RawTermsValid :
    exact94914RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94914 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13130⟩⟩) exact94914RawTerms (.finite 58) 94913 .exactZero (none)

def event94915 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10225⟩⟩) 0 ⟨5503⟩ 94911

def event94916 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10225⟩⟩) (.authority (.programFamilyFact))

def exact94917RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10225⟩⟩], []⟩, (1)⟩]

theorem exact94917RawTermsValid :
    exact94917RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94917 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10225⟩⟩) exact94917RawTerms (.finite 58) 94916 .exactZero (none)

def event94918 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13131⟩⟩) 0 ⟨10225⟩ 94917

def event94919 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13131⟩⟩) 1 ⟨13130⟩ 94914

def event94920 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13131⟩⟩) (.product (.predecessor 0 94918 .coefficient) (.predecessor 1 94919 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event94921 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13131⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10225⟩⟩, ⟨.program ⟨214⟩, ⟨13130⟩⟩], []⟩) [⟨.result 94917 .coefficient, true, some 1⟩, ⟨.result 94914 .coefficient, true, some 1⟩])

def event94922 : Event := .survivorFold (1) 94921

def exact94923RawTerms : List Term := []

theorem exact94923RawTermsValid :
    exact94923RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94923 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13131⟩⟩) exact94923RawTerms (.finite 3364) 94920 (.finite 3364) (some (94921))

def event94924 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13132⟩⟩) 0 ⟨13131⟩ 94923

def event94925 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13132⟩⟩) (.identity (.predecessor 0 94924 .coefficient))

def event94926 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13132⟩⟩) (.finite 3364)

def event94927 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20165⟩⟩) 0 ⟨13132⟩ 94926

def event94928 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20165⟩⟩) (.authority (.relationPreimageSource ⟨25⟩))

def exact94929RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20165⟩⟩]⟩, (1)⟩]

theorem exact94929RawTermsValid :
    exact94929RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94929 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20165⟩⟩) exact94929RawTerms (.finite 136065468) 94928 .exactZero (none)

def event94930 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact94931RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact94931RawTermsValid :
    exact94931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94931 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact94931RawTerms .large 94930 .exactZero (none)

def event94932 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20166⟩⟩) 0 ⟨6⟩ 94931

def event94933 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20166⟩⟩) 1 ⟨20165⟩ 94929

def event94934 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20166⟩⟩) (.product (.predecessor 0 94932 .coefficient) (.predecessor 1 94933 .coefficient) (⟨false, false, none, none, none⟩))

def event94935 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20166⟩⟩, .operator (⟨94931, 0⟩, ⟨94929, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20165⟩⟩]⟩, (1)⟩)

def exact94936RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20165⟩⟩]⟩, (1)⟩]

theorem exact94936RawTermsValid :
    exact94936RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94936 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20166⟩⟩) exact94936RawTerms .large 94934 .exactZero (none)

def event94937 : Event := .preFoldPolynomial 94936 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20165⟩⟩]⟩, (1)⟩] .exactZero none

def exact94938RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20165⟩⟩]⟩, (1)⟩]

def event94938 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20166⟩⟩) 94937 exact94938RawTerms .large 94934 .exactZero (none)

def event94939 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25672⟩⟩)

def event94940 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event94941 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event94942 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event94943 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event94944 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 94943

def event94945 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 94941

def event94946 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 94944 .coefficient) (.value (.predecessor 1 94945 .coefficient)))

def event94947 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event94948 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13130⟩⟩) 0 ⟨5503⟩ 94947

def event94949 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13130⟩⟩) (.authority (.programFamilyFact))

def exact94950RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13130⟩⟩], []⟩, (1)⟩]

theorem exact94950RawTermsValid :
    exact94950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94950 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13130⟩⟩) exact94950RawTerms (.finite 58) 94949 .exactZero (none)

def event94951 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10225⟩⟩) 0 ⟨5503⟩ 94947

def event94952 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10225⟩⟩) (.authority (.programFamilyFact))

def exact94953RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10225⟩⟩], []⟩, (1)⟩]

theorem exact94953RawTermsValid :
    exact94953RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94953 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10225⟩⟩) exact94953RawTerms (.finite 58) 94952 .exactZero (none)

def event94954 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13131⟩⟩) 0 ⟨10225⟩ 94953

def event94955 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13131⟩⟩) 1 ⟨13130⟩ 94950

def event94956 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13131⟩⟩) (.product (.predecessor 0 94954 .coefficient) (.predecessor 1 94955 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event94957 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13131⟩⟩, .operator (⟨94953, 0⟩, ⟨94950, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10225⟩⟩, ⟨.program ⟨214⟩, ⟨13130⟩⟩], []⟩, (1)⟩)

def exact94958RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10225⟩⟩, ⟨.program ⟨214⟩, ⟨13130⟩⟩], []⟩, (1)⟩]

theorem exact94958RawTermsValid :
    exact94958RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94958 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13131⟩⟩) exact94958RawTerms (.finite 3364) 94956 .exactZero (none)

def event94959 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13132⟩⟩) 0 ⟨13131⟩ 94958

def event94960 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13132⟩⟩) (.identity (.predecessor 0 94959 .coefficient))

def event94961 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13132⟩⟩) (.finite 3364)

def event94962 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23367⟩⟩) 0 ⟨13132⟩ 94961

def event94963 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23367⟩⟩) (.authority (.programFamilyFact))

def event94964 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23367⟩⟩) (.finite 3720)

def event94965 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event94966 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23368⟩⟩) 0 ⟨6689⟩ 94965

def event94967 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23368⟩⟩) 1 ⟨23367⟩ 94964

def event94968 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23368⟩⟩) (.authority (.operator))

def exact94969RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23368⟩⟩]⟩, (1)⟩]

theorem exact94969RawTermsValid :
    exact94969RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94969 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23368⟩⟩) exact94969RawTerms .large 94968 .exactZero (none)

def event94970 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25668⟩⟩) 0 ⟨23368⟩ 94969

def event94971 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25668⟩⟩) (.authority (.operator))

def exact94972RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25668⟩⟩]⟩, (1)⟩]

theorem exact94972RawTermsValid :
    exact94972RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event94972 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25668⟩⟩) exact94972RawTerms (.finite 8192) 94971 .exactZero (none)

def event94973 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event94974 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event94975 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13242⟩⟩) 0 ⟨13132⟩ 94961

def eventLeaf5920 : Array AnnotatedEvent := #[
  { event := event94720
    frameStart := 94690 },
  { event := event94721
    frameStart := 94690 },
  { event := event94722
    frameStart := 94690 },
  { event := event94723
    frameStart := 94690 },
  { event := event94724
    frameStart := 94690 },
  { event := event94725
    frameStart := 94690 },
  { event := event94726
    frameStart := 94690 },
  { event := event94727
    frameStart := 94690 },
  { event := event94728
    frameStart := 94690 },
  { event := event94729
    frameStart := 94690 },
  { event := event94730
    frameStart := 94690 },
  { event := event94731
    frameStart := 94690 },
  { event := event94732
    frameStart := 94690 },
  { event := event94733
    frameStart := 94690 },
  { event := event94734
    frameStart := 94690 },
  { event := event94735
    frameStart := 94690 }
]

def eventLeaf5921 : Array AnnotatedEvent := #[
  { event := event94736
    frameStart := 94690 },
  { event := event94737
    frameStart := 94690 },
  { event := event94738
    frameStart := 94690 },
  { event := event94739
    frameStart := 94690 },
  { event := event94740
    frameStart := 94690 },
  { event := event94741
    frameStart := 94690 },
  { event := event94742
    frameStart := 94690 },
  { event := event94743
    frameStart := 94690 },
  { event := event94744
    frameStart := 94690 },
  { event := event94745
    frameStart := 94690 },
  { event := event94746
    frameStart := 94690 },
  { event := event94747
    frameStart := 94690 },
  { event := event94748
    frameStart := 94690 },
  { event := event94749
    frameStart := 94690 },
  { event := event94750
    frameStart := 94690 },
  { event := event94751
    frameStart := 94690 }
]

def eventLeaf5922 : Array AnnotatedEvent := #[
  { event := event94752
    frameStart := 94690 },
  { event := event94753
    frameStart := 94690 },
  { event := event94754
    frameStart := 94690 },
  { event := event94755
    frameStart := 94690 },
  { event := event94756
    frameStart := 94690 },
  { event := event94757
    frameStart := 94690 },
  { event := event94758
    frameStart := 94690 },
  { event := event94759
    frameStart := 94690 },
  { event := event94760
    frameStart := 94690 },
  { event := event94761
    frameStart := 94690 },
  { event := event94762
    frameStart := 94690 },
  { event := event94763
    frameStart := 94690 },
  { event := event94764
    frameStart := 94690 },
  { event := event94765
    frameStart := 94690 },
  { event := event94766
    frameStart := 94690 },
  { event := event94767
    frameStart := 94690 }
]

def eventLeaf5923 : Array AnnotatedEvent := #[
  { event := event94768
    frameStart := 94690 },
  { event := event94769
    frameStart := 94690 },
  { event := event94770
    frameStart := 94690 },
  { event := event94771
    frameStart := 94690 },
  { event := event94772
    frameStart := 94690 },
  { event := event94773
    frameStart := 94690 },
  { event := event94774
    frameStart := 94690 },
  { event := event94775
    frameStart := 94690 },
  { event := event94776
    frameStart := 94690 },
  { event := event94777
    frameStart := 94690 },
  { event := event94778
    frameStart := 94690 },
  { event := event94779
    frameStart := 94690 },
  { event := event94780
    frameStart := 94690 },
  { event := event94781
    frameStart := 94690 },
  { event := event94782
    frameStart := 0 },
  { event := event94783
    frameStart := 0 }
]

def eventLeaf5924 : Array AnnotatedEvent := #[
  { event := event94784
    frameStart := 0 },
  { event := event94785
    frameStart := 0 },
  { event := event94786
    frameStart := 0 },
  { event := event94787
    frameStart := 0 },
  { event := event94788
    frameStart := 0 },
  { event := event94789
    frameStart := 0 },
  { event := event94790
    frameStart := 0 },
  { event := event94791
    frameStart := 0 },
  { event := event94792
    frameStart := 0 },
  { event := event94793
    frameStart := 0 },
  { event := event94794
    frameStart := 0 },
  { event := event94795
    frameStart := 0 },
  { event := event94796
    frameStart := 0 },
  { event := event94797
    frameStart := 0 },
  { event := event94798
    frameStart := 0 },
  { event := event94799
    frameStart := 0 }
]

def eventLeaf5925 : Array AnnotatedEvent := #[
  { event := event94800
    frameStart := 0 },
  { event := event94801
    frameStart := 0 },
  { event := event94802
    frameStart := 0 },
  { event := event94803
    frameStart := 0 },
  { event := event94804
    frameStart := 0 },
  { event := event94805
    frameStart := 0 },
  { event := event94806
    frameStart := 0 },
  { event := event94807
    frameStart := 0 },
  { event := event94808
    frameStart := 0 },
  { event := event94809
    frameStart := 0 },
  { event := event94810
    frameStart := 0 },
  { event := event94811
    frameStart := 0 },
  { event := event94812
    frameStart := 0 },
  { event := event94813
    frameStart := 0 },
  { event := event94814
    frameStart := 0 },
  { event := event94815
    frameStart := 0 }
]

def eventLeaf5926 : Array AnnotatedEvent := #[
  { event := event94816
    frameStart := 0 },
  { event := event94817
    frameStart := 0 },
  { event := event94818
    frameStart := 0 },
  { event := event94819
    frameStart := 0 },
  { event := event94820
    frameStart := 0 },
  { event := event94821
    frameStart := 0 },
  { event := event94822
    frameStart := 0 },
  { event := event94823
    frameStart := 0 },
  { event := event94824
    frameStart := 0 },
  { event := event94825
    frameStart := 0 },
  { event := event94826
    frameStart := 0 },
  { event := event94827
    frameStart := 0 },
  { event := event94828
    frameStart := 0 },
  { event := event94829
    frameStart := 0 },
  { event := event94830
    frameStart := 0 },
  { event := event94831
    frameStart := 0 }
]

def eventLeaf5927 : Array AnnotatedEvent := #[
  { event := event94832
    frameStart := 0 },
  { event := event94833
    frameStart := 0 },
  { event := event94834
    frameStart := 0 },
  { event := event94835
    frameStart := 0 },
  { event := event94836
    frameStart := 0 },
  { event := event94837
    frameStart := 0 },
  { event := event94838
    frameStart := 0 },
  { event := event94839
    frameStart := 0 },
  { event := event94840
    frameStart := 0 },
  { event := event94841
    frameStart := 0 },
  { event := event94842
    frameStart := 0 },
  { event := event94843
    frameStart := 0 },
  { event := event94844
    frameStart := 0 },
  { event := event94845
    frameStart := 0 },
  { event := event94846
    frameStart := 0 },
  { event := event94847
    frameStart := 0 }
]

def eventLeaf5928 : Array AnnotatedEvent := #[
  { event := event94848
    frameStart := 0 },
  { event := event94849
    frameStart := 0 },
  { event := event94850
    frameStart := 0 },
  { event := event94851
    frameStart := 0 },
  { event := event94852
    frameStart := 0 },
  { event := event94853
    frameStart := 0 },
  { event := event94854
    frameStart := 0 },
  { event := event94855
    frameStart := 0 },
  { event := event94856
    frameStart := 0 },
  { event := event94857
    frameStart := 0 },
  { event := event94858
    frameStart := 0 },
  { event := event94859
    frameStart := 0 },
  { event := event94860
    frameStart := 0 },
  { event := event94861
    frameStart := 0 },
  { event := event94862
    frameStart := 0 },
  { event := event94863
    frameStart := 0 }
]

def eventLeaf5929 : Array AnnotatedEvent := #[
  { event := event94864
    frameStart := 0 },
  { event := event94865
    frameStart := 0 },
  { event := event94866
    frameStart := 0 },
  { event := event94867
    frameStart := 0 },
  { event := event94868
    frameStart := 0 },
  { event := event94869
    frameStart := 0 },
  { event := event94870
    frameStart := 0 },
  { event := event94871
    frameStart := 0 },
  { event := event94872
    frameStart := 0 },
  { event := event94873
    frameStart := 0 },
  { event := event94874
    frameStart := 0 },
  { event := event94875
    frameStart := 0 },
  { event := event94876
    frameStart := 0 },
  { event := event94877
    frameStart := 0 },
  { event := event94878
    frameStart := 0 },
  { event := event94879
    frameStart := 0 }
]

def eventLeaf5930 : Array AnnotatedEvent := #[
  { event := event94880
    frameStart := 0 },
  { event := event94881
    frameStart := 0 },
  { event := event94882
    frameStart := 0 },
  { event := event94883
    frameStart := 0 },
  { event := event94884
    frameStart := 0 },
  { event := event94885
    frameStart := 0 },
  { event := event94886
    frameStart := 0 },
  { event := event94887
    frameStart := 0 },
  { event := event94888
    frameStart := 0 },
  { event := event94889
    frameStart := 0 },
  { event := event94890
    frameStart := 0 },
  { event := event94891
    frameStart := 0 },
  { event := event94892
    frameStart := 0 },
  { event := event94893
    frameStart := 0 },
  { event := event94894
    frameStart := 0 },
  { event := event94895
    frameStart := 0 }
]

def eventLeaf5931 : Array AnnotatedEvent := #[
  { event := event94896
    frameStart := 0 },
  { event := event94897
    frameStart := 0 },
  { event := event94898
    frameStart := 0 },
  { event := event94899
    frameStart := 0 },
  { event := event94900
    frameStart := 0 },
  { event := event94901
    frameStart := 0 },
  { event := event94902
    frameStart := 0 },
  { event := event94903
    frameStart := 94903 },
  { event := event94904
    frameStart := 94903 },
  { event := event94905
    frameStart := 94903 },
  { event := event94906
    frameStart := 94903 },
  { event := event94907
    frameStart := 94903 },
  { event := event94908
    frameStart := 94903 },
  { event := event94909
    frameStart := 94903 },
  { event := event94910
    frameStart := 94903 },
  { event := event94911
    frameStart := 94903 }
]

def eventLeaf5932 : Array AnnotatedEvent := #[
  { event := event94912
    frameStart := 94903 },
  { event := event94913
    frameStart := 94903 },
  { event := event94914
    frameStart := 94903 },
  { event := event94915
    frameStart := 94903 },
  { event := event94916
    frameStart := 94903 },
  { event := event94917
    frameStart := 94903 },
  { event := event94918
    frameStart := 94903 },
  { event := event94919
    frameStart := 94903 },
  { event := event94920
    frameStart := 94903 },
  { event := event94921
    frameStart := 94903 },
  { event := event94922
    frameStart := 94903 },
  { event := event94923
    frameStart := 94903 },
  { event := event94924
    frameStart := 94903 },
  { event := event94925
    frameStart := 94903 },
  { event := event94926
    frameStart := 94903 },
  { event := event94927
    frameStart := 94903 }
]

def eventLeaf5933 : Array AnnotatedEvent := #[
  { event := event94928
    frameStart := 94903 },
  { event := event94929
    frameStart := 94903 },
  { event := event94930
    frameStart := 94903 },
  { event := event94931
    frameStart := 94903 },
  { event := event94932
    frameStart := 94903 },
  { event := event94933
    frameStart := 94903 },
  { event := event94934
    frameStart := 94903 },
  { event := event94935
    frameStart := 94903 },
  { event := event94936
    frameStart := 94903 },
  { event := event94937
    frameStart := 94903 },
  { event := event94938
    frameStart := 94903 },
  { event := event94939
    frameStart := 94939 },
  { event := event94940
    frameStart := 94939 },
  { event := event94941
    frameStart := 94939 },
  { event := event94942
    frameStart := 94939 },
  { event := event94943
    frameStart := 94939 }
]

def eventLeaf5934 : Array AnnotatedEvent := #[
  { event := event94944
    frameStart := 94939 },
  { event := event94945
    frameStart := 94939 },
  { event := event94946
    frameStart := 94939 },
  { event := event94947
    frameStart := 94939 },
  { event := event94948
    frameStart := 94939 },
  { event := event94949
    frameStart := 94939 },
  { event := event94950
    frameStart := 94939 },
  { event := event94951
    frameStart := 94939 },
  { event := event94952
    frameStart := 94939 },
  { event := event94953
    frameStart := 94939 },
  { event := event94954
    frameStart := 94939 },
  { event := event94955
    frameStart := 94939 },
  { event := event94956
    frameStart := 94939 },
  { event := event94957
    frameStart := 94939 },
  { event := event94958
    frameStart := 94939 },
  { event := event94959
    frameStart := 94939 }
]

def eventLeaf5935 : Array AnnotatedEvent := #[
  { event := event94960
    frameStart := 94939 },
  { event := event94961
    frameStart := 94939 },
  { event := event94962
    frameStart := 94939 },
  { event := event94963
    frameStart := 94939 },
  { event := event94964
    frameStart := 94939 },
  { event := event94965
    frameStart := 94939 },
  { event := event94966
    frameStart := 94939 },
  { event := event94967
    frameStart := 94939 },
  { event := event94968
    frameStart := 94939 },
  { event := event94969
    frameStart := 94939 },
  { event := event94970
    frameStart := 94939 },
  { event := event94971
    frameStart := 94939 },
  { event := event94972
    frameStart := 94939 },
  { event := event94973
    frameStart := 94939 },
  { event := event94974
    frameStart := 94939 },
  { event := event94975
    frameStart := 94939 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events370
