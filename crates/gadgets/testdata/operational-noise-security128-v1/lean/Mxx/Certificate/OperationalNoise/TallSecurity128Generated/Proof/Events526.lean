import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events526

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event134656 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7196⟩⟩) (.authority (.operator))

def exact134657RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩]

theorem exact134657RawTermsValid :
    exact134657RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134657 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7196⟩⟩) exact134657RawTerms .large 134656 .exactZero (none)

def event134658 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48095⟩⟩) 0 ⟨7196⟩ 134657

def event134659 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48095⟩⟩) 1 ⟨48094⟩ 134654

def event134660 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48095⟩⟩) (.sum [.predecessor 0 134658 .coefficient, .predecessor 1 134659 .coefficient])

def exact134661RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact134661RawTermsValid :
    exact134661RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134661 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48095⟩⟩) exact134661RawTerms .large 134660 .exactZero (none)

def event134662 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49586⟩⟩) 0 ⟨48095⟩ 134661

def event134663 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49586⟩⟩) 1 ⟨49585⟩ 134646

def event134664 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49586⟩⟩) (.sum [.predecessor 0 134662 .coefficient, .predecessor 1 134663 .coefficient])

def exact134665RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], [⟨.program ⟨257⟩, ⟨49107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact134665RawTermsValid :
    exact134665RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134665 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49586⟩⟩) exact134665RawTerms .large 134664 .exactZero (none)

def event134666 : Event := .preFoldPolynomial 134665 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], [⟨.program ⟨257⟩, ⟨49107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact134667RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], [⟨.program ⟨257⟩, ⟨49107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event134667 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨49586⟩⟩) 134666 exact134667RawTerms .large 134664 .exactZero (none)

def event134668 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨47668⟩⟩) ⟨⟨75⟩, ⟨54⟩, ⟨135⟩⟩ ⟨134502, 134668⟩

def event134669 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨48522⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48519⟩⟩]⟩) (1) 0 2 (.universal 134668 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48519⟩⟩]⟩) (none) 134667)

def event134670 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48522⟩⟩, .relation 134669 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩)

def event134671 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48522⟩⟩, .relation 134669 1, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩]⟩, (-1)⟩)

def event134672 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48522⟩⟩, .relation 134669 2, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], [⟨.program ⟨257⟩, ⟨49107⟩⟩]⟩, (1)⟩)

def event134673 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48522⟩⟩, .relation 134669 3, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact134674RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], [⟨.program ⟨257⟩, ⟨49107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact134674RawTermsValid :
    exact134674RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134674 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48522⟩⟩) exact134674RawTerms .large 134498 (.finite 202072841853861888) (some (134500))

def event134675 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49584⟩⟩) 0 ⟨48522⟩ 134674

def event134676 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49584⟩⟩) 1 ⟨49583⟩ 134477

def event134677 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49584⟩⟩) (.sum [.predecessor 0 134675 .coefficient, .predecessor 1 134676 .coefficient])

def event134678 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49584⟩⟩, .operator (⟨134674, 2⟩, ⟨134477, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], [⟨.program ⟨257⟩, ⟨49107⟩⟩]⟩, (-1)⟩)

def event134679 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49584⟩⟩, .operator (⟨134674, 1⟩, ⟨134477, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩]⟩, (1)⟩)

def event134680 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49584⟩⟩) (.sum [.result 134674 .summary, .result 134477 .summary])

def exact134681RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact134681RawTermsValid :
    exact134681RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134681 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49584⟩⟩) exact134681RawTerms .large 134677 (.finite 2998346861024241778688) (some (134680))

def event134682 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49856⟩⟩) 0 ⟨49584⟩ 134681

def event134683 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49856⟩⟩) 1 ⟨49854⟩ 134388

def event134684 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49856⟩⟩) (.product (.predecessor 0 134682 .coefficient) (.predecessor 1 134683 .coefficient) (⟨false, false, none, none, none⟩))

def event134685 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49856⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨49854⟩⟩]⟩) [⟨.result 134388 .coefficient, false, none⟩])

def event134686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49856⟩⟩) (.product (.result 134681 .summary) (.transfer 134685) (⟨false, false, none, none, none⟩))

def event134687 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49856⟩⟩, .operator (⟨134681, 0⟩, ⟨134388, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49854⟩⟩]⟩, (1)⟩)

def event134688 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49856⟩⟩, .operator (⟨134681, 1⟩, ⟨134388, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49854⟩⟩]⟩, (-1)⟩)

def event134689 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49856⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49854⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49854⟩⟩) ⟨49238⟩ 134385)

def event134690 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49856⟩⟩, .relation 134689 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨49238⟩⟩]⟩, (-1)⟩)

def exact134691RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49854⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨49238⟩⟩]⟩, (-1)⟩]

theorem exact134691RawTermsValid :
    exact134691RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134691 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49856⟩⟩) exact134691RawTerms .large 134684 (.finite 32194504275408438756654574469120) (some (134686))

def event134692 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48756⟩⟩) 0 ⟨48093⟩ 6095

def event134693 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48756⟩⟩) (.authority (.relationPreimageSource ⟨94⟩))

def exact134694RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48756⟩⟩]⟩, (1)⟩]

theorem exact134694RawTermsValid :
    exact134694RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134694 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48756⟩⟩) exact134694RawTerms (.finite 5647228698) 134693 .exactZero (none)

def event134695 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48758⟩⟩) 0 ⟨48756⟩ 134694

def event134696 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48758⟩⟩) 1 ⟨2370⟩ 4

def event134697 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48758⟩⟩) (.scale (.predecessor 0 134695 .coefficient) (.value (.predecessor 1 134696 .coefficient)))

def exact134698RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48756⟩⟩]⟩, (1)⟩]

theorem exact134698RawTermsValid :
    exact134698RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134698 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48758⟩⟩) exact134698RawTerms (.finite 5647228698) 134697 .exactZero (none)

def event134699 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48759⟩⟩) 0 ⟨5473⟩ 134495

def event134700 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48759⟩⟩) 1 ⟨48758⟩ 134698

def event134701 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48759⟩⟩) (.product (.predecessor 0 134699 .coefficient) (.predecessor 1 134700 .coefficient) (⟨false, false, none, none, none⟩))

def event134702 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48759⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨48756⟩⟩]⟩) [⟨.result 134694 .coefficient, false, none⟩])

def event134703 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48759⟩⟩) (.product (.result 134495 .summary) (.transfer 134702) (⟨false, false, none, none, none⟩))

def event134704 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48759⟩⟩, .operator (⟨134495, 0⟩, ⟨134698, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48756⟩⟩]⟩, (1)⟩)

def event134705 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨48757⟩⟩)

def event134706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event134707 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event134708 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event134709 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event134710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event134711 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event134712 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event134713 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event134714 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 134713

def event134715 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 134711

def event134716 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 134714 .coefficient) (.value (.predecessor 1 134715 .coefficient)))

def event134717 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event134718 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 134717

def event134719 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 134709

def event134720 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 134718 .coefficient, .predecessor 1 134719 .coefficient])

def event134721 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event134722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 134721

def event134723 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 134707

def event134724 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 134723 .coefficient))

def event134725 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event134726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47666⟩⟩) 0 ⟨5469⟩ 134725

def event134727 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47666⟩⟩) (.authority (.programFamilyFact))

def exact134728RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47666⟩⟩], []⟩, (1)⟩]

theorem exact134728RawTermsValid :
    exact134728RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134728 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47666⟩⟩) exact134728RawTerms (.finite 60) 134727 .exactZero (none)

def event134729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14976⟩⟩) 0 ⟨5469⟩ 134725

def event134730 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14976⟩⟩) (.authority (.programFamilyFact))

def exact134731RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩], []⟩, (1)⟩]

theorem exact134731RawTermsValid :
    exact134731RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134731 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14976⟩⟩) exact134731RawTerms (.finite 60) 134730 .exactZero (none)

def event134732 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47667⟩⟩) 0 ⟨14976⟩ 134731

def event134733 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47667⟩⟩) 1 ⟨47666⟩ 134728

def event134734 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47667⟩⟩) (.product (.predecessor 0 134732 .coefficient) (.predecessor 1 134733 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event134735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47667⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], []⟩) [⟨.result 134731 .coefficient, true, some 1⟩, ⟨.result 134728 .coefficient, true, some 1⟩])

def event134736 : Event := .survivorFold (1) 134735

def exact134737RawTerms : List Term := []

theorem exact134737RawTermsValid :
    exact134737RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134737 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47667⟩⟩) exact134737RawTerms (.finite 3600) 134734 (.finite 3600) (some (134735))

def event134738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47668⟩⟩) 0 ⟨47667⟩ 134737

def event134739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47668⟩⟩) (.identity (.predecessor 0 134738 .coefficient))

def event134740 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47668⟩⟩) (.finite 3600)

def event134741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48092⟩⟩) 0 ⟨47668⟩ 134740

def event134742 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48092⟩⟩) (.authority (.programFamilyFact))

def exact134743RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], []⟩, (1)⟩]

theorem exact134743RawTermsValid :
    exact134743RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134743 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48092⟩⟩) exact134743RawTerms (.finite 60) 134742 .exactZero (none)

def event134744 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48093⟩⟩) 0 ⟨48092⟩ 134743

def event134745 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48093⟩⟩) (.identity (.predecessor 0 134744 .coefficient))

def event134746 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48093⟩⟩) (.finite 60)

def event134747 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48756⟩⟩) 0 ⟨48093⟩ 134746

def event134748 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48756⟩⟩) (.authority (.relationPreimageSource ⟨94⟩))

def exact134749RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48756⟩⟩]⟩, (1)⟩]

theorem exact134749RawTermsValid :
    exact134749RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134749 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48756⟩⟩) exact134749RawTerms (.finite 5647228698) 134748 .exactZero (none)

def event134750 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact134751RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact134751RawTermsValid :
    exact134751RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134751 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact134751RawTerms .large 134750 .exactZero (none)

def event134752 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48757⟩⟩) 0 ⟨35⟩ 134751

def event134753 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48757⟩⟩) 1 ⟨48756⟩ 134749

def event134754 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48757⟩⟩) (.product (.predecessor 0 134752 .coefficient) (.predecessor 1 134753 .coefficient) (⟨false, false, none, none, none⟩))

def event134755 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48757⟩⟩, .operator (⟨134751, 0⟩, ⟨134749, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48756⟩⟩]⟩, (1)⟩)

def exact134756RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48756⟩⟩]⟩, (1)⟩]

theorem exact134756RawTermsValid :
    exact134756RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134756 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48757⟩⟩) exact134756RawTerms .large 134754 .exactZero (none)

def event134757 : Event := .preFoldPolynomial 134756 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48756⟩⟩]⟩, (1)⟩] .exactZero none

def exact134758RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48756⟩⟩]⟩, (1)⟩]

def event134758 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨48757⟩⟩) 134757 exact134758RawTerms .large 134754 .exactZero (none)

def event134759 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨49858⟩⟩)

def event134760 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event134761 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event134762 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨455⟩⟩) (.authority (.operator))

def event134763 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨455⟩⟩) (.finite 11)

def event134764 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event134765 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event134766 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event134767 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event134768 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 134767

def event134769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 134765

def event134770 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 134768 .coefficient) (.value (.predecessor 1 134769 .coefficient)))

def event134771 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event134772 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 0 ⟨392⟩ 134771

def event134773 : Event := .predecessor (⟨.program ⟨257⟩, ⟨457⟩⟩) 1 ⟨455⟩ 134763

def event134774 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨457⟩⟩) (.sum [.predecessor 0 134772 .coefficient, .predecessor 1 134773 .coefficient])

def event134775 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨457⟩⟩) (.finite 655351)

def event134776 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 0 ⟨457⟩ 134775

def event134777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5469⟩⟩) 1 ⟨5426⟩ 134761

def event134778 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.identity (.predecessor 1 134777 .coefficient))

def event134779 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5469⟩⟩) (.finite 655360)

def event134780 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47666⟩⟩) 0 ⟨5469⟩ 134779

def event134781 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47666⟩⟩) (.authority (.programFamilyFact))

def exact134782RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47666⟩⟩], []⟩, (1)⟩]

theorem exact134782RawTermsValid :
    exact134782RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134782 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47666⟩⟩) exact134782RawTerms (.finite 60) 134781 .exactZero (none)

def event134783 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14976⟩⟩) 0 ⟨5469⟩ 134779

def event134784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14976⟩⟩) (.authority (.programFamilyFact))

def exact134785RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩], []⟩, (1)⟩]

theorem exact134785RawTermsValid :
    exact134785RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134785 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14976⟩⟩) exact134785RawTerms (.finite 60) 134784 .exactZero (none)

def event134786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47667⟩⟩) 0 ⟨14976⟩ 134785

def event134787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47667⟩⟩) 1 ⟨47666⟩ 134782

def event134788 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47667⟩⟩) (.product (.predecessor 0 134786 .coefficient) (.predecessor 1 134787 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event134789 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47667⟩⟩, .operator (⟨134785, 0⟩, ⟨134782, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], []⟩, (1)⟩)

def exact134790RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], []⟩, (1)⟩]

theorem exact134790RawTermsValid :
    exact134790RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134790 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47667⟩⟩) exact134790RawTerms (.finite 3600) 134788 .exactZero (none)

def event134791 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47668⟩⟩) 0 ⟨47667⟩ 134790

def event134792 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47668⟩⟩) (.identity (.predecessor 0 134791 .coefficient))

def event134793 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47668⟩⟩) (.finite 3600)

def event134794 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48092⟩⟩) 0 ⟨47668⟩ 134793

def event134795 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48092⟩⟩) (.authority (.programFamilyFact))

def exact134796RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], []⟩, (1)⟩]

theorem exact134796RawTermsValid :
    exact134796RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134796 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48092⟩⟩) exact134796RawTerms (.finite 60) 134795 .exactZero (none)

def event134797 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48093⟩⟩) 0 ⟨48092⟩ 134796

def event134798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48093⟩⟩) (.identity (.predecessor 0 134797 .coefficient))

def event134799 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48093⟩⟩) (.finite 60)

def event134800 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49236⟩⟩) 0 ⟨48093⟩ 134799

def event134801 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49236⟩⟩) (.authority (.programFamilyFact))

def event134802 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49236⟩⟩) (.finite 3720)

def event134803 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event134804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49238⟩⟩) 0 ⟨7177⟩ 134803

def event134805 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49238⟩⟩) 1 ⟨49236⟩ 134802

def event134806 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49238⟩⟩) (.authority (.operator))

def exact134807RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49238⟩⟩]⟩, (1)⟩]

theorem exact134807RawTermsValid :
    exact134807RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134807 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49238⟩⟩) exact134807RawTerms .large 134806 .exactZero (none)

def event134808 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49854⟩⟩) 0 ⟨49238⟩ 134807

def event134809 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49854⟩⟩) (.authority (.operator))

def exact134810RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49854⟩⟩]⟩, (1)⟩]

theorem exact134810RawTermsValid :
    exact134810RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134810 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49854⟩⟩) exact134810RawTerms (.finite 8192) 134809 .exactZero (none)

def event134811 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event134812 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event134813 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49478⟩⟩) 0 ⟨48093⟩ 134799

def event134814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49478⟩⟩) 1 ⟨136⟩ 134812

def event134815 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49478⟩⟩) (.sum [.predecessor 0 134813 .coefficient, .predecessor 1 134814 .coefficient])

def event134816 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49478⟩⟩) (.finite 60)

def event134817 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49479⟩⟩) 0 ⟨49478⟩ 134816

def event134818 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49479⟩⟩) (.identity (.predecessor 0 134817 .coefficient))

def exact134819RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], []⟩, (1)⟩]

theorem exact134819RawTermsValid :
    exact134819RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134819 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49479⟩⟩) exact134819RawTerms (.finite 60) 134818 .exactZero (none)

def event134820 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact134821RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact134821RawTermsValid :
    exact134821RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134821 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact134821RawTerms .large 134820 .exactZero (none)

def event134822 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49480⟩⟩) 0 ⟨6908⟩ 134821

def event134823 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49480⟩⟩) 1 ⟨49479⟩ 134819

def event134824 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49480⟩⟩) (.product (.predecessor 0 134822 .coefficient) (.predecessor 1 134823 .coefficient) (⟨false, false, none, none, none⟩))

def event134825 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49480⟩⟩, .operator (⟨134821, 0⟩, ⟨134819, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact134826RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact134826RawTermsValid :
    exact134826RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134826 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49480⟩⟩) exact134826RawTerms .large 134824 .exactZero (none)

def event134827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7196⟩⟩) 0 ⟨7177⟩ 134803

def event134828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7196⟩⟩) (.authority (.operator))

def exact134829RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩]

theorem exact134829RawTermsValid :
    exact134829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134829 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7196⟩⟩) exact134829RawTerms .large 134828 .exactZero (none)

def event134830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49481⟩⟩) 0 ⟨7196⟩ 134829

def event134831 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49481⟩⟩) 1 ⟨49480⟩ 134826

def event134832 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49481⟩⟩) (.sum [.predecessor 0 134830 .coefficient, .predecessor 1 134831 .coefficient])

def exact134833RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact134833RawTermsValid :
    exact134833RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134833 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49481⟩⟩) exact134833RawTerms .large 134832 .exactZero (none)

def event134834 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49855⟩⟩) 0 ⟨49481⟩ 134833

def event134835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49855⟩⟩) 1 ⟨49854⟩ 134810

def event134836 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49855⟩⟩) (.product (.predecessor 0 134834 .coefficient) (.predecessor 1 134835 .coefficient) (⟨false, false, none, none, none⟩))

def event134837 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49855⟩⟩, .operator (⟨134833, 0⟩, ⟨134810, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49854⟩⟩]⟩, (1)⟩)

def event134838 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49855⟩⟩, .operator (⟨134833, 1⟩, ⟨134810, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49854⟩⟩]⟩, (-1)⟩)

def event134839 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49855⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49854⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49854⟩⟩) ⟨49238⟩ 134807)

def event134840 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49855⟩⟩, .relation 134839 0, ⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨49238⟩⟩]⟩, (-1)⟩)

def exact134841RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49854⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨49238⟩⟩]⟩, (-1)⟩]

theorem exact134841RawTermsValid :
    exact134841RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134841 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49855⟩⟩) exact134841RawTerms .large 134836 .exactZero (none)

def event134842 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48272⟩⟩) 0 ⟨48093⟩ 134799

def event134843 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48272⟩⟩) (.authority (.programFamilyFact))

def exact134844RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], []⟩, (1)⟩]

theorem exact134844RawTermsValid :
    exact134844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134844 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48272⟩⟩) exact134844RawTerms (.finite 63) 134843 .exactZero (none)

def event134845 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48273⟩⟩) 0 ⟨6908⟩ 134821

def event134846 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48273⟩⟩) 1 ⟨48272⟩ 134844

def event134847 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48273⟩⟩) (.product (.predecessor 0 134845 .coefficient) (.predecessor 1 134846 .coefficient) (⟨false, true, none, none, some 1⟩))

def event134848 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48273⟩⟩, .operator (⟨134821, 0⟩, ⟨134844, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact134849RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact134849RawTermsValid :
    exact134849RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134849 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48273⟩⟩) exact134849RawTerms .large 134847 .exactZero (none)

def event134850 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7232⟩⟩) 0 ⟨7177⟩ 134803

def event134851 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7232⟩⟩) (.authority (.operator))

def exact134852RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩]

theorem exact134852RawTermsValid :
    exact134852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134852 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7232⟩⟩) exact134852RawTerms .large 134851 .exactZero (none)

def event134853 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48274⟩⟩) 0 ⟨7232⟩ 134852

def event134854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48274⟩⟩) 1 ⟨48273⟩ 134849

def event134855 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48274⟩⟩) (.sum [.predecessor 0 134853 .coefficient, .predecessor 1 134854 .coefficient])

def exact134856RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact134856RawTermsValid :
    exact134856RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134856 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48274⟩⟩) exact134856RawTerms .large 134855 .exactZero (none)

def event134857 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49858⟩⟩) 0 ⟨48274⟩ 134856

def event134858 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49858⟩⟩) 1 ⟨49855⟩ 134841

def event134859 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49858⟩⟩) (.sum [.predecessor 0 134857 .coefficient, .predecessor 1 134858 .coefficient])

def exact134860RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49854⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨49238⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact134860RawTermsValid :
    exact134860RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134860 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49858⟩⟩) exact134860RawTerms .large 134859 .exactZero (none)

def event134861 : Event := .preFoldPolynomial 134860 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49854⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨49238⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact134862RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49854⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨49238⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event134862 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨49858⟩⟩) 134861 exact134862RawTerms .large 134859 .exactZero (none)

def event134863 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨48093⟩⟩) ⟨⟨111⟩, ⟨94⟩, ⟨135⟩⟩ ⟨134705, 134863⟩

def event134864 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨48759⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48756⟩⟩]⟩) (1) 0 2 (.universal 134863 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48756⟩⟩]⟩) (none) 134862)

def event134865 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48759⟩⟩, .relation 134864 1, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩)

def event134866 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48759⟩⟩, .relation 134864 0, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49854⟩⟩]⟩, (-1)⟩)

def event134867 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48759⟩⟩, .relation 134864 2, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨49238⟩⟩]⟩, (1)⟩)

def event134868 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48759⟩⟩, .relation 134864 3, ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact134869RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49854⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨49238⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact134869RawTermsValid :
    exact134869RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134869 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48759⟩⟩) exact134869RawTerms .large 134701 (.finite 202072841853861888) (some (134703))

def event134870 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49857⟩⟩) 0 ⟨48759⟩ 134869

def event134871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49857⟩⟩) 1 ⟨49856⟩ 134691

def event134872 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49857⟩⟩) (.sum [.predecessor 0 134870 .coefficient, .predecessor 1 134871 .coefficient])

def event134873 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49857⟩⟩, .operator (⟨134869, 0⟩, ⟨134691, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49854⟩⟩]⟩, (1)⟩)

def event134874 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49857⟩⟩, .operator (⟨134869, 2⟩, ⟨134691, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨49238⟩⟩]⟩, (-1)⟩)

def event134875 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49857⟩⟩) (.sum [.result 134869 .summary, .result 134691 .summary])

def exact134876RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48272⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact134876RawTermsValid :
    exact134876RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134876 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49857⟩⟩) exact134876RawTerms .large 134872 (.finite 32194504275408640829496428331008) (some (134875))

def event134877 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46556⟩⟩) 0 ⟨45413⟩ 6118

def event134878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46556⟩⟩) (.authority (.programFamilyFact))

def event134879 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46556⟩⟩) (.finite 3720)

def event134880 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46558⟩⟩) 0 ⟨7177⟩ 15500

def event134881 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46558⟩⟩) 1 ⟨46556⟩ 134879

def event134882 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46558⟩⟩) (.authority (.operator))

def exact134883RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46558⟩⟩]⟩, (1)⟩]

theorem exact134883RawTermsValid :
    exact134883RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134883 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46558⟩⟩) exact134883RawTerms .large 134882 .exactZero (none)

def event134884 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47174⟩⟩) 0 ⟨46558⟩ 134883

def event134885 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47174⟩⟩) (.authority (.operator))

def exact134886RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨47174⟩⟩]⟩, (1)⟩]

theorem exact134886RawTermsValid :
    exact134886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134886 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47174⟩⟩) exact134886RawTerms (.finite 8192) 134885 .exactZero (none)

def event134887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46426⟩⟩) 0 ⟨44988⟩ 6112

def event134888 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46426⟩⟩) (.authority (.programFamilyFact))

def event134889 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨46426⟩⟩) (.finite 3720)

def event134890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46427⟩⟩) 0 ⟨7177⟩ 15500

def event134891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46427⟩⟩) 1 ⟨46426⟩ 134889

def event134892 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46427⟩⟩) (.authority (.operator))

def exact134893RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46427⟩⟩]⟩, (1)⟩]

theorem exact134893RawTermsValid :
    exact134893RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134893 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46427⟩⟩) exact134893RawTerms .large 134892 .exactZero (none)

def event134894 : Event := .predecessor (⟨.program ⟨257⟩, ⟨46902⟩⟩) 0 ⟨46427⟩ 134893

def event134895 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨46902⟩⟩) (.authority (.operator))

def exact134896RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨46902⟩⟩]⟩, (1)⟩]

theorem exact134896RawTermsValid :
    exact134896RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134896 : Event := .resultExact (⟨.program ⟨257⟩, ⟨46902⟩⟩) exact134896RawTerms (.finite 8192) 134895 .exactZero (none)

def event134897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44989⟩⟩) 0 ⟨44986⟩ 6101

def event134898 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44989⟩⟩) 1 ⟨6919⟩ 134403

def event134899 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44989⟩⟩) (.tensor (.predecessor 0 134897 .coefficient) (.predecessor 1 134898 .coefficient) true false)

def event134900 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44989⟩⟩, .operator (⟨6101, 0⟩, ⟨134403, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact134901RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact134901RawTermsValid :
    exact134901RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134901 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44989⟩⟩) exact134901RawTerms .large 134899 .exactZero (none)

def event134902 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7792⟩⟩) 0 ⟨5471⟩ 134273

def event134903 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7792⟩⟩) 1 ⟨7284⟩ 17581

def event134904 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7792⟩⟩) (.product (.predecessor 0 134902 .coefficient) (.predecessor 1 134903 .coefficient) (⟨false, false, none, none, none⟩))

def event134905 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7792⟩⟩, .operator (⟨134273, 0⟩, ⟨17581, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩)

def exact134906RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩]

theorem exact134906RawTermsValid :
    exact134906RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134906 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7792⟩⟩) exact134906RawTerms .large 134904 .exactZero (none)

def event134907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44990⟩⟩) 0 ⟨7792⟩ 134906

def event134908 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44990⟩⟩) 1 ⟨44989⟩ 134901

def event134909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44990⟩⟩) (.sum [.predecessor 0 134907 .coefficient, .predecessor 1 134908 .coefficient])

def exact134910RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨44986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact134910RawTermsValid :
    exact134910RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event134910 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44990⟩⟩) exact134910RawTerms .large 134909 .exactZero (none)

def event134911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44991⟩⟩) 0 ⟨44990⟩ 134910

def eventLeaf8416 : Array AnnotatedEvent := #[
  { event := event134656
    frameStart := 134550 },
  { event := event134657
    frameStart := 134550 },
  { event := event134658
    frameStart := 134550 },
  { event := event134659
    frameStart := 134550 },
  { event := event134660
    frameStart := 134550 },
  { event := event134661
    frameStart := 134550 },
  { event := event134662
    frameStart := 134550 },
  { event := event134663
    frameStart := 134550 },
  { event := event134664
    frameStart := 134550 },
  { event := event134665
    frameStart := 134550 },
  { event := event134666
    frameStart := 134550 },
  { event := event134667
    frameStart := 134550 },
  { event := event134668
    frameStart := 0 },
  { event := event134669
    frameStart := 0 },
  { event := event134670
    frameStart := 0 },
  { event := event134671
    frameStart := 0 }
]

def eventLeaf8417 : Array AnnotatedEvent := #[
  { event := event134672
    frameStart := 0 },
  { event := event134673
    frameStart := 0 },
  { event := event134674
    frameStart := 0 },
  { event := event134675
    frameStart := 0 },
  { event := event134676
    frameStart := 0 },
  { event := event134677
    frameStart := 0 },
  { event := event134678
    frameStart := 0 },
  { event := event134679
    frameStart := 0 },
  { event := event134680
    frameStart := 0 },
  { event := event134681
    frameStart := 0 },
  { event := event134682
    frameStart := 0 },
  { event := event134683
    frameStart := 0 },
  { event := event134684
    frameStart := 0 },
  { event := event134685
    frameStart := 0 },
  { event := event134686
    frameStart := 0 },
  { event := event134687
    frameStart := 0 }
]

def eventLeaf8418 : Array AnnotatedEvent := #[
  { event := event134688
    frameStart := 0 },
  { event := event134689
    frameStart := 0 },
  { event := event134690
    frameStart := 0 },
  { event := event134691
    frameStart := 0 },
  { event := event134692
    frameStart := 0 },
  { event := event134693
    frameStart := 0 },
  { event := event134694
    frameStart := 0 },
  { event := event134695
    frameStart := 0 },
  { event := event134696
    frameStart := 0 },
  { event := event134697
    frameStart := 0 },
  { event := event134698
    frameStart := 0 },
  { event := event134699
    frameStart := 0 },
  { event := event134700
    frameStart := 0 },
  { event := event134701
    frameStart := 0 },
  { event := event134702
    frameStart := 0 },
  { event := event134703
    frameStart := 0 }
]

def eventLeaf8419 : Array AnnotatedEvent := #[
  { event := event134704
    frameStart := 0 },
  { event := event134705
    frameStart := 134705 },
  { event := event134706
    frameStart := 134705 },
  { event := event134707
    frameStart := 134705 },
  { event := event134708
    frameStart := 134705 },
  { event := event134709
    frameStart := 134705 },
  { event := event134710
    frameStart := 134705 },
  { event := event134711
    frameStart := 134705 },
  { event := event134712
    frameStart := 134705 },
  { event := event134713
    frameStart := 134705 },
  { event := event134714
    frameStart := 134705 },
  { event := event134715
    frameStart := 134705 },
  { event := event134716
    frameStart := 134705 },
  { event := event134717
    frameStart := 134705 },
  { event := event134718
    frameStart := 134705 },
  { event := event134719
    frameStart := 134705 }
]

def eventLeaf8420 : Array AnnotatedEvent := #[
  { event := event134720
    frameStart := 134705 },
  { event := event134721
    frameStart := 134705 },
  { event := event134722
    frameStart := 134705 },
  { event := event134723
    frameStart := 134705 },
  { event := event134724
    frameStart := 134705 },
  { event := event134725
    frameStart := 134705 },
  { event := event134726
    frameStart := 134705 },
  { event := event134727
    frameStart := 134705 },
  { event := event134728
    frameStart := 134705 },
  { event := event134729
    frameStart := 134705 },
  { event := event134730
    frameStart := 134705 },
  { event := event134731
    frameStart := 134705 },
  { event := event134732
    frameStart := 134705 },
  { event := event134733
    frameStart := 134705 },
  { event := event134734
    frameStart := 134705 },
  { event := event134735
    frameStart := 134705 }
]

def eventLeaf8421 : Array AnnotatedEvent := #[
  { event := event134736
    frameStart := 134705 },
  { event := event134737
    frameStart := 134705 },
  { event := event134738
    frameStart := 134705 },
  { event := event134739
    frameStart := 134705 },
  { event := event134740
    frameStart := 134705 },
  { event := event134741
    frameStart := 134705 },
  { event := event134742
    frameStart := 134705 },
  { event := event134743
    frameStart := 134705 },
  { event := event134744
    frameStart := 134705 },
  { event := event134745
    frameStart := 134705 },
  { event := event134746
    frameStart := 134705 },
  { event := event134747
    frameStart := 134705 },
  { event := event134748
    frameStart := 134705 },
  { event := event134749
    frameStart := 134705 },
  { event := event134750
    frameStart := 134705 },
  { event := event134751
    frameStart := 134705 }
]

def eventLeaf8422 : Array AnnotatedEvent := #[
  { event := event134752
    frameStart := 134705 },
  { event := event134753
    frameStart := 134705 },
  { event := event134754
    frameStart := 134705 },
  { event := event134755
    frameStart := 134705 },
  { event := event134756
    frameStart := 134705 },
  { event := event134757
    frameStart := 134705 },
  { event := event134758
    frameStart := 134705 },
  { event := event134759
    frameStart := 134759 },
  { event := event134760
    frameStart := 134759 },
  { event := event134761
    frameStart := 134759 },
  { event := event134762
    frameStart := 134759 },
  { event := event134763
    frameStart := 134759 },
  { event := event134764
    frameStart := 134759 },
  { event := event134765
    frameStart := 134759 },
  { event := event134766
    frameStart := 134759 },
  { event := event134767
    frameStart := 134759 }
]

def eventLeaf8423 : Array AnnotatedEvent := #[
  { event := event134768
    frameStart := 134759 },
  { event := event134769
    frameStart := 134759 },
  { event := event134770
    frameStart := 134759 },
  { event := event134771
    frameStart := 134759 },
  { event := event134772
    frameStart := 134759 },
  { event := event134773
    frameStart := 134759 },
  { event := event134774
    frameStart := 134759 },
  { event := event134775
    frameStart := 134759 },
  { event := event134776
    frameStart := 134759 },
  { event := event134777
    frameStart := 134759 },
  { event := event134778
    frameStart := 134759 },
  { event := event134779
    frameStart := 134759 },
  { event := event134780
    frameStart := 134759 },
  { event := event134781
    frameStart := 134759 },
  { event := event134782
    frameStart := 134759 },
  { event := event134783
    frameStart := 134759 }
]

def eventLeaf8424 : Array AnnotatedEvent := #[
  { event := event134784
    frameStart := 134759 },
  { event := event134785
    frameStart := 134759 },
  { event := event134786
    frameStart := 134759 },
  { event := event134787
    frameStart := 134759 },
  { event := event134788
    frameStart := 134759 },
  { event := event134789
    frameStart := 134759 },
  { event := event134790
    frameStart := 134759 },
  { event := event134791
    frameStart := 134759 },
  { event := event134792
    frameStart := 134759 },
  { event := event134793
    frameStart := 134759 },
  { event := event134794
    frameStart := 134759 },
  { event := event134795
    frameStart := 134759 },
  { event := event134796
    frameStart := 134759 },
  { event := event134797
    frameStart := 134759 },
  { event := event134798
    frameStart := 134759 },
  { event := event134799
    frameStart := 134759 }
]

def eventLeaf8425 : Array AnnotatedEvent := #[
  { event := event134800
    frameStart := 134759 },
  { event := event134801
    frameStart := 134759 },
  { event := event134802
    frameStart := 134759 },
  { event := event134803
    frameStart := 134759 },
  { event := event134804
    frameStart := 134759 },
  { event := event134805
    frameStart := 134759 },
  { event := event134806
    frameStart := 134759 },
  { event := event134807
    frameStart := 134759 },
  { event := event134808
    frameStart := 134759 },
  { event := event134809
    frameStart := 134759 },
  { event := event134810
    frameStart := 134759 },
  { event := event134811
    frameStart := 134759 },
  { event := event134812
    frameStart := 134759 },
  { event := event134813
    frameStart := 134759 },
  { event := event134814
    frameStart := 134759 },
  { event := event134815
    frameStart := 134759 }
]

def eventLeaf8426 : Array AnnotatedEvent := #[
  { event := event134816
    frameStart := 134759 },
  { event := event134817
    frameStart := 134759 },
  { event := event134818
    frameStart := 134759 },
  { event := event134819
    frameStart := 134759 },
  { event := event134820
    frameStart := 134759 },
  { event := event134821
    frameStart := 134759 },
  { event := event134822
    frameStart := 134759 },
  { event := event134823
    frameStart := 134759 },
  { event := event134824
    frameStart := 134759 },
  { event := event134825
    frameStart := 134759 },
  { event := event134826
    frameStart := 134759 },
  { event := event134827
    frameStart := 134759 },
  { event := event134828
    frameStart := 134759 },
  { event := event134829
    frameStart := 134759 },
  { event := event134830
    frameStart := 134759 },
  { event := event134831
    frameStart := 134759 }
]

def eventLeaf8427 : Array AnnotatedEvent := #[
  { event := event134832
    frameStart := 134759 },
  { event := event134833
    frameStart := 134759 },
  { event := event134834
    frameStart := 134759 },
  { event := event134835
    frameStart := 134759 },
  { event := event134836
    frameStart := 134759 },
  { event := event134837
    frameStart := 134759 },
  { event := event134838
    frameStart := 134759 },
  { event := event134839
    frameStart := 134759 },
  { event := event134840
    frameStart := 134759 },
  { event := event134841
    frameStart := 134759 },
  { event := event134842
    frameStart := 134759 },
  { event := event134843
    frameStart := 134759 },
  { event := event134844
    frameStart := 134759 },
  { event := event134845
    frameStart := 134759 },
  { event := event134846
    frameStart := 134759 },
  { event := event134847
    frameStart := 134759 }
]

def eventLeaf8428 : Array AnnotatedEvent := #[
  { event := event134848
    frameStart := 134759 },
  { event := event134849
    frameStart := 134759 },
  { event := event134850
    frameStart := 134759 },
  { event := event134851
    frameStart := 134759 },
  { event := event134852
    frameStart := 134759 },
  { event := event134853
    frameStart := 134759 },
  { event := event134854
    frameStart := 134759 },
  { event := event134855
    frameStart := 134759 },
  { event := event134856
    frameStart := 134759 },
  { event := event134857
    frameStart := 134759 },
  { event := event134858
    frameStart := 134759 },
  { event := event134859
    frameStart := 134759 },
  { event := event134860
    frameStart := 134759 },
  { event := event134861
    frameStart := 134759 },
  { event := event134862
    frameStart := 134759 },
  { event := event134863
    frameStart := 0 }
]

def eventLeaf8429 : Array AnnotatedEvent := #[
  { event := event134864
    frameStart := 0 },
  { event := event134865
    frameStart := 0 },
  { event := event134866
    frameStart := 0 },
  { event := event134867
    frameStart := 0 },
  { event := event134868
    frameStart := 0 },
  { event := event134869
    frameStart := 0 },
  { event := event134870
    frameStart := 0 },
  { event := event134871
    frameStart := 0 },
  { event := event134872
    frameStart := 0 },
  { event := event134873
    frameStart := 0 },
  { event := event134874
    frameStart := 0 },
  { event := event134875
    frameStart := 0 },
  { event := event134876
    frameStart := 0 },
  { event := event134877
    frameStart := 0 },
  { event := event134878
    frameStart := 0 },
  { event := event134879
    frameStart := 0 }
]

def eventLeaf8430 : Array AnnotatedEvent := #[
  { event := event134880
    frameStart := 0 },
  { event := event134881
    frameStart := 0 },
  { event := event134882
    frameStart := 0 },
  { event := event134883
    frameStart := 0 },
  { event := event134884
    frameStart := 0 },
  { event := event134885
    frameStart := 0 },
  { event := event134886
    frameStart := 0 },
  { event := event134887
    frameStart := 0 },
  { event := event134888
    frameStart := 0 },
  { event := event134889
    frameStart := 0 },
  { event := event134890
    frameStart := 0 },
  { event := event134891
    frameStart := 0 },
  { event := event134892
    frameStart := 0 },
  { event := event134893
    frameStart := 0 },
  { event := event134894
    frameStart := 0 },
  { event := event134895
    frameStart := 0 }
]

def eventLeaf8431 : Array AnnotatedEvent := #[
  { event := event134896
    frameStart := 0 },
  { event := event134897
    frameStart := 0 },
  { event := event134898
    frameStart := 0 },
  { event := event134899
    frameStart := 0 },
  { event := event134900
    frameStart := 0 },
  { event := event134901
    frameStart := 0 },
  { event := event134902
    frameStart := 0 },
  { event := event134903
    frameStart := 0 },
  { event := event134904
    frameStart := 0 },
  { event := event134905
    frameStart := 0 },
  { event := event134906
    frameStart := 0 },
  { event := event134907
    frameStart := 0 },
  { event := event134908
    frameStart := 0 },
  { event := event134909
    frameStart := 0 },
  { event := event134910
    frameStart := 0 },
  { event := event134911
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events526
