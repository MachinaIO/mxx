import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events276

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact70656RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70656RawTermsValid :
    exact70656RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70656 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13987⟩⟩) exact70656RawTerms .large 70653 (.finite 26) (some (70654))

def event70657 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13988⟩⟩) 0 ⟨13987⟩ 70656

def event70658 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13988⟩⟩) 1 ⟨7850⟩ 12013

def event70659 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13988⟩⟩) (.product (.predecessor 0 70657 .coefficient) (.predecessor 1 70658 .coefficient) (⟨false, false, none, none, none⟩))

def event70660 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13988⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩) [⟨.result 12009 .coefficient, false, none⟩])

def event70661 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13988⟩⟩) (.product (.result 70656 .summary) (.transfer 70660) (⟨false, false, none, none, none⟩))

def event70662 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13988⟩⟩, .operator (⟨70656, 1⟩, ⟨12013, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (-1)⟩)

def event70663 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨13988⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7849⟩⟩) ⟨6778⟩ 11983)

def event70664 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13988⟩⟩, .relation 70663 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (-1)⟩)

def event70665 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13988⟩⟩, .operator (⟨70656, 0⟩, ⟨12013, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩)

def exact70666RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (-1)⟩]

theorem exact70666RawTermsValid :
    exact70666RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70666 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13988⟩⟩) exact70666RawTerms .large 70659 (.finite 95420416) (some (70661))

def event70667 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13989⟩⟩) 0 ⟨13988⟩ 70666

def event70668 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13989⟩⟩) 1 ⟨13984⟩ 70636

def event70669 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13989⟩⟩) (.sum [.predecessor 0 70667 .coefficient, .predecessor 1 70668 .coefficient])

def event70670 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13989⟩⟩, .operator (⟨70666, 1⟩, ⟨70636, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩)

def event70671 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13989⟩⟩) (.sum [.result 70666 .summary, .result 70636 .summary])

def exact70672RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70672RawTermsValid :
    exact70672RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70672 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13989⟩⟩) exact70672RawTerms .large 70669 (.finite 95433728) (some (70671))

def event70673 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25985⟩⟩) 0 ⟨13989⟩ 70672

def event70674 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25985⟩⟩) 1 ⟨25984⟩ 70608

def event70675 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25985⟩⟩) (.product (.predecessor 0 70673 .coefficient) (.predecessor 1 70674 .coefficient) (⟨false, false, none, none, none⟩))

def event70676 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25985⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25984⟩⟩]⟩) [⟨.result 70608 .coefficient, false, none⟩])

def event70677 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25985⟩⟩) (.product (.result 70672 .summary) (.transfer 70676) (⟨false, false, none, none, none⟩))

def event70678 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25985⟩⟩, .operator (⟨70672, 1⟩, ⟨70608, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25984⟩⟩]⟩, (-1)⟩)

def event70679 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25985⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25984⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25984⟩⟩) ⟨23540⟩ 70605)

def event70680 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25985⟩⟩, .relation 70679 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨23540⟩⟩]⟩, (-1)⟩)

def event70681 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25985⟩⟩, .operator (⟨70672, 0⟩, ⟨70608, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25984⟩⟩]⟩, (1)⟩)

def exact70682RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25984⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨23540⟩⟩]⟩, (-1)⟩]

theorem exact70682RawTermsValid :
    exact70682RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70682 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25985⟩⟩) exact70682RawTerms .large 70675 (.finite 350243308699648) (some (70677))

def event70683 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19452⟩⟩) 0 ⟨13983⟩ 3350

def event70684 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19452⟩⟩) (.authority (.relationPreimageSource ⟨14⟩))

def exact70685RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19452⟩⟩]⟩, (1)⟩]

theorem exact70685RawTermsValid :
    exact70685RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70685 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19452⟩⟩) exact70685RawTerms (.finite 136065468) 70684 .exactZero (none)

def event70686 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19454⟩⟩) 0 ⟨19452⟩ 70685

def event70687 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19454⟩⟩) 1 ⟨2348⟩ 4

def event70688 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19454⟩⟩) (.scale (.predecessor 0 70686 .coefficient) (.value (.predecessor 1 70687 .coefficient)))

def exact70689RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19452⟩⟩]⟩, (1)⟩]

theorem exact70689RawTermsValid :
    exact70689RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70689 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19454⟩⟩) exact70689RawTerms (.finite 136065468) 70688 .exactZero (none)

def event70690 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19455⟩⟩) 0 ⟨5535⟩ 65387

def event70691 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19455⟩⟩) 1 ⟨19454⟩ 70689

def event70692 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19455⟩⟩) (.product (.predecessor 0 70690 .coefficient) (.predecessor 1 70691 .coefficient) (⟨false, false, none, none, none⟩))

def event70693 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19455⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19452⟩⟩]⟩) [⟨.result 70685 .coefficient, false, none⟩])

def event70694 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19455⟩⟩) (.product (.result 65387 .summary) (.transfer 70693) (⟨false, false, none, none, none⟩))

def event70695 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19455⟩⟩, .operator (⟨65387, 0⟩, ⟨70689, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19452⟩⟩]⟩, (1)⟩)

def event70696 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19453⟩⟩)

def event70697 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event70698 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event70699 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event70700 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event70701 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event70702 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event70703 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event70704 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event70705 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 70704

def event70706 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 70702

def event70707 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 70705 .coefficient) (.value (.predecessor 1 70706 .coefficient)))

def event70708 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event70709 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 70708

def event70710 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 70700

def event70711 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 70709 .coefficient, .predecessor 1 70710 .coefficient])

def event70712 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event70713 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 70712

def event70714 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 70698

def event70715 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 70714 .coefficient))

def event70716 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event70717 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11381⟩⟩) 0 ⟨5530⟩ 70716

def event70718 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11381⟩⟩) (.authority (.programFamilyFact))

def exact70719RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩], []⟩, (1)⟩]

theorem exact70719RawTermsValid :
    exact70719RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70719 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11381⟩⟩) exact70719RawTerms (.finite 16) 70718 .exactZero (none)

def event70720 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13981⟩⟩) 0 ⟨5530⟩ 70716

def event70721 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13981⟩⟩) (.authority (.programFamilyFact))

def exact70722RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩, (1)⟩]

theorem exact70722RawTermsValid :
    exact70722RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70722 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13981⟩⟩) exact70722RawTerms (.finite 16) 70721 .exactZero (none)

def event70723 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13982⟩⟩) 0 ⟨13981⟩ 70722

def event70724 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13982⟩⟩) 1 ⟨11381⟩ 70719

def event70725 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13982⟩⟩) (.product (.predecessor 0 70723 .coefficient) (.predecessor 1 70724 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70726 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13982⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩) [⟨.result 70722 .coefficient, true, some 1⟩, ⟨.result 70719 .coefficient, true, some 1⟩])

def event70727 : Event := .survivorFold (1) 70726

def exact70728RawTerms : List Term := []

theorem exact70728RawTermsValid :
    exact70728RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70728 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13982⟩⟩) exact70728RawTerms (.finite 256) 70725 (.finite 256) (some (70726))

def event70729 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13983⟩⟩) 0 ⟨13982⟩ 70728

def event70730 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13983⟩⟩) (.identity (.predecessor 0 70729 .coefficient))

def event70731 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13983⟩⟩) (.finite 256)

def event70732 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19452⟩⟩) 0 ⟨13983⟩ 70731

def event70733 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19452⟩⟩) (.authority (.relationPreimageSource ⟨14⟩))

def exact70734RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19452⟩⟩]⟩, (1)⟩]

theorem exact70734RawTermsValid :
    exact70734RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70734 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19452⟩⟩) exact70734RawTerms (.finite 136065468) 70733 .exactZero (none)

def event70735 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact70736RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact70736RawTermsValid :
    exact70736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70736 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact70736RawTerms .large 70735 .exactZero (none)

def event70737 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19453⟩⟩) 0 ⟨6⟩ 70736

def event70738 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19453⟩⟩) 1 ⟨19452⟩ 70734

def event70739 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19453⟩⟩) (.product (.predecessor 0 70737 .coefficient) (.predecessor 1 70738 .coefficient) (⟨false, false, none, none, none⟩))

def event70740 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19453⟩⟩, .operator (⟨70736, 0⟩, ⟨70734, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19452⟩⟩]⟩, (1)⟩)

def exact70741RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19452⟩⟩]⟩, (1)⟩]

theorem exact70741RawTermsValid :
    exact70741RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70741 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19453⟩⟩) exact70741RawTerms .large 70739 .exactZero (none)

def event70742 : Event := .preFoldPolynomial 70741 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19452⟩⟩]⟩, (1)⟩] .exactZero none

def exact70743RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19452⟩⟩]⟩, (1)⟩]

def event70743 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19453⟩⟩) 70742 exact70743RawTerms .large 70739 .exactZero (none)

def event70744 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25988⟩⟩)

def event70745 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event70746 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event70747 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event70748 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event70749 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event70750 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event70751 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event70752 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event70753 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 70752

def event70754 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 70750

def event70755 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 70753 .coefficient) (.value (.predecessor 1 70754 .coefficient)))

def event70756 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event70757 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 70756

def event70758 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 70748

def event70759 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 70757 .coefficient, .predecessor 1 70758 .coefficient])

def event70760 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event70761 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 70760

def event70762 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 70746

def event70763 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 70762 .coefficient))

def event70764 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event70765 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11381⟩⟩) 0 ⟨5530⟩ 70764

def event70766 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11381⟩⟩) (.authority (.programFamilyFact))

def exact70767RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩], []⟩, (1)⟩]

theorem exact70767RawTermsValid :
    exact70767RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70767 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11381⟩⟩) exact70767RawTerms (.finite 16) 70766 .exactZero (none)

def event70768 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13981⟩⟩) 0 ⟨5530⟩ 70764

def event70769 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13981⟩⟩) (.authority (.programFamilyFact))

def exact70770RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩, (1)⟩]

theorem exact70770RawTermsValid :
    exact70770RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70770 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13981⟩⟩) exact70770RawTerms (.finite 16) 70769 .exactZero (none)

def event70771 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13982⟩⟩) 0 ⟨13981⟩ 70770

def event70772 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13982⟩⟩) 1 ⟨11381⟩ 70767

def event70773 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13982⟩⟩) (.product (.predecessor 0 70771 .coefficient) (.predecessor 1 70772 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event70774 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨13982⟩⟩, .operator (⟨70770, 0⟩, ⟨70767, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩, (1)⟩)

def exact70775RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩, (1)⟩]

theorem exact70775RawTermsValid :
    exact70775RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70775 : Event := .resultExact (⟨.program ⟨214⟩, ⟨13982⟩⟩) exact70775RawTerms (.finite 256) 70773 .exactZero (none)

def event70776 : Event := .predecessor (⟨.program ⟨214⟩, ⟨13983⟩⟩) 0 ⟨13982⟩ 70775

def event70777 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨13983⟩⟩) (.identity (.predecessor 0 70776 .coefficient))

def event70778 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨13983⟩⟩) (.finite 256)

def event70779 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23539⟩⟩) 0 ⟨13983⟩ 70778

def event70780 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23539⟩⟩) (.authority (.programFamilyFact))

def event70781 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23539⟩⟩) (.finite 3720)

def event70782 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event70783 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23540⟩⟩) 0 ⟨6689⟩ 70782

def event70784 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23540⟩⟩) 1 ⟨23539⟩ 70781

def event70785 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23540⟩⟩) (.authority (.operator))

def exact70786RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23540⟩⟩]⟩, (1)⟩]

theorem exact70786RawTermsValid :
    exact70786RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70786 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23540⟩⟩) exact70786RawTerms .large 70785 .exactZero (none)

def event70787 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25984⟩⟩) 0 ⟨23540⟩ 70786

def event70788 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25984⟩⟩) (.authority (.operator))

def exact70789RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25984⟩⟩]⟩, (1)⟩]

theorem exact70789RawTermsValid :
    exact70789RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70789 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25984⟩⟩) exact70789RawTerms (.finite 8192) 70788 .exactZero (none)

def event70790 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event70791 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event70792 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14093⟩⟩) 0 ⟨13983⟩ 70778

def event70793 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14093⟩⟩) 1 ⟨110⟩ 70791

def event70794 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14093⟩⟩) (.sum [.predecessor 0 70792 .coefficient, .predecessor 1 70793 .coefficient])

def event70795 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14093⟩⟩) (.finite 256)

def event70796 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14094⟩⟩) 0 ⟨14093⟩ 70795

def event70797 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14094⟩⟩) (.identity (.predecessor 0 70796 .coefficient))

def exact70798RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], []⟩, (1)⟩]

theorem exact70798RawTermsValid :
    exact70798RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70798 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14094⟩⟩) exact70798RawTerms (.finite 256) 70797 .exactZero (none)

def event70799 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact70800RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact70800RawTermsValid :
    exact70800RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70800 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact70800RawTerms .large 70799 .exactZero (none)

def event70801 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14095⟩⟩) 0 ⟨6544⟩ 70800

def event70802 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14095⟩⟩) 1 ⟨14094⟩ 70798

def event70803 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14095⟩⟩) (.product (.predecessor 0 70801 .coefficient) (.predecessor 1 70802 .coefficient) (⟨false, false, none, none, none⟩))

def event70804 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14095⟩⟩, .operator (⟨70800, 0⟩, ⟨70798, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact70805RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact70805RawTermsValid :
    exact70805RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70805 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14095⟩⟩) exact70805RawTerms .large 70803 .exactZero (none)

def event70806 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event70807 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event70808 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 70782

def event70809 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact70810RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact70810RawTermsValid :
    exact70810RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70810 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact70810RawTerms .large 70809 .exactZero (none)

def event70811 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6778⟩⟩) 0 ⟨6757⟩ 70810

def event70812 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6778⟩⟩) (.identity (.predecessor 0 70811 .coefficient))

def exact70813RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6778⟩⟩]⟩, (1)⟩]

theorem exact70813RawTermsValid :
    exact70813RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70813 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6778⟩⟩) exact70813RawTerms .large 70812 .exactZero (none)

def event70814 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7849⟩⟩) 0 ⟨6778⟩ 70813

def event70815 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7849⟩⟩) (.authority (.operator))

def exact70816RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩]

theorem exact70816RawTermsValid :
    exact70816RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70816 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7849⟩⟩) exact70816RawTerms (.finite 8192) 70815 .exactZero (none)

def event70817 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7850⟩⟩) 0 ⟨7849⟩ 70816

def event70818 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7850⟩⟩) 1 ⟨2348⟩ 70807

def event70819 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7850⟩⟩) (.scale (.predecessor 0 70817 .coefficient) (.value (.predecessor 1 70818 .coefficient)))

def exact70820RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩]

theorem exact70820RawTermsValid :
    exact70820RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70820 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7850⟩⟩) exact70820RawTerms (.finite 8192) 70819 .exactZero (none)

def event70821 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6758⟩⟩) 0 ⟨6757⟩ 70810

def event70822 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6758⟩⟩) (.identity (.predecessor 0 70821 .coefficient))

def exact70823RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩]⟩, (1)⟩]

theorem exact70823RawTermsValid :
    exact70823RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70823 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6758⟩⟩) exact70823RawTerms .large 70822 .exactZero (none)

def event70824 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7851⟩⟩) 0 ⟨6758⟩ 70823

def event70825 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7851⟩⟩) 1 ⟨7850⟩ 70820

def event70826 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7851⟩⟩) (.product (.predecessor 0 70824 .coefficient) (.predecessor 1 70825 .coefficient) (⟨false, false, none, none, none⟩))

def event70827 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7851⟩⟩, .operator (⟨70823, 0⟩, ⟨70820, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩)

def exact70828RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩]

theorem exact70828RawTermsValid :
    exact70828RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70828 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7851⟩⟩) exact70828RawTerms .large 70826 .exactZero (none)

def event70829 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14096⟩⟩) 0 ⟨7851⟩ 70828

def event70830 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14096⟩⟩) 1 ⟨14095⟩ 70805

def event70831 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14096⟩⟩) (.sum [.predecessor 0 70829 .coefficient, .predecessor 1 70830 .coefficient])

def exact70832RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70832RawTermsValid :
    exact70832RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70832 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14096⟩⟩) exact70832RawTerms .large 70831 .exactZero (none)

def event70833 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25987⟩⟩) 0 ⟨14096⟩ 70832

def event70834 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25987⟩⟩) 1 ⟨25984⟩ 70789

def event70835 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25987⟩⟩) (.product (.predecessor 0 70833 .coefficient) (.predecessor 1 70834 .coefficient) (⟨false, false, none, none, none⟩))

def event70836 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25987⟩⟩, .operator (⟨70832, 0⟩, ⟨70789, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25984⟩⟩]⟩, (1)⟩)

def event70837 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25987⟩⟩, .operator (⟨70832, 1⟩, ⟨70789, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25984⟩⟩]⟩, (-1)⟩)

def event70838 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25987⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25984⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25984⟩⟩) ⟨23540⟩ 70786)

def event70839 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25987⟩⟩, .relation 70838 0, ⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨23540⟩⟩]⟩, (-1)⟩)

def exact70840RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25984⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨23540⟩⟩]⟩, (-1)⟩]

theorem exact70840RawTermsValid :
    exact70840RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70840 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25987⟩⟩) exact70840RawTerms .large 70835 .exactZero (none)

def event70841 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15817⟩⟩) 0 ⟨13983⟩ 70778

def event70842 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15817⟩⟩) (.authority (.programFamilyFact))

def exact70843RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], []⟩, (1)⟩]

theorem exact70843RawTermsValid :
    exact70843RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70843 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15817⟩⟩) exact70843RawTerms (.finite 16) 70842 .exactZero (none)

def event70844 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15819⟩⟩) 0 ⟨6544⟩ 70800

def event70845 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15819⟩⟩) 1 ⟨15817⟩ 70843

def event70846 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15819⟩⟩) (.product (.predecessor 0 70844 .coefficient) (.predecessor 1 70845 .coefficient) (⟨false, true, none, none, some 1⟩))

def event70847 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15819⟩⟩, .operator (⟨70800, 0⟩, ⟨70843, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact70848RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact70848RawTermsValid :
    exact70848RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70848 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15819⟩⟩) exact70848RawTerms .large 70846 .exactZero (none)

def event70849 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6696⟩⟩) 0 ⟨6689⟩ 70782

def event70850 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6696⟩⟩) (.authority (.operator))

def exact70851RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩]

theorem exact70851RawTermsValid :
    exact70851RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70851 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6696⟩⟩) exact70851RawTerms .large 70850 .exactZero (none)

def event70852 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15820⟩⟩) 0 ⟨6696⟩ 70851

def event70853 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15820⟩⟩) 1 ⟨15819⟩ 70848

def event70854 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15820⟩⟩) (.sum [.predecessor 0 70852 .coefficient, .predecessor 1 70853 .coefficient])

def exact70855RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70855RawTermsValid :
    exact70855RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70855 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15820⟩⟩) exact70855RawTerms .large 70854 .exactZero (none)

def event70856 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25988⟩⟩) 0 ⟨15820⟩ 70855

def event70857 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25988⟩⟩) 1 ⟨25987⟩ 70840

def event70858 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25988⟩⟩) (.sum [.predecessor 0 70856 .coefficient, .predecessor 1 70857 .coefficient])

def exact70859RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25984⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨23540⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70859RawTermsValid :
    exact70859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70859 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25988⟩⟩) exact70859RawTerms .large 70858 .exactZero (none)

def event70860 : Event := .preFoldPolynomial 70859 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25984⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨23540⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact70861RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25984⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨23540⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event70861 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25988⟩⟩) 70860 exact70861RawTerms .large 70858 .exactZero (none)

def event70862 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨13983⟩⟩) ⟨⟨109⟩, ⟨14⟩, ⟨109⟩⟩ ⟨70696, 70862⟩

def event70863 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19455⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19452⟩⟩]⟩) (1) 0 2 (.universal 70862 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19452⟩⟩]⟩) (none) 70861)

def event70864 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19455⟩⟩, .relation 70863 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩)

def event70865 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19455⟩⟩, .relation 70863 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25984⟩⟩]⟩, (-1)⟩)

def event70866 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19455⟩⟩, .relation 70863 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨23540⟩⟩]⟩, (1)⟩)

def event70867 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19455⟩⟩, .relation 70863 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact70868RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25984⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨23540⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70868RawTermsValid :
    exact70868RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70868 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19455⟩⟩) exact70868RawTerms .large 70692 (.finite 1811303510016) (some (70694))

def event70869 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25986⟩⟩) 0 ⟨19455⟩ 70868

def event70870 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25986⟩⟩) 1 ⟨25985⟩ 70682

def event70871 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25986⟩⟩) (.sum [.predecessor 0 70869 .coefficient, .predecessor 1 70870 .coefficient])

def event70872 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25986⟩⟩, .operator (⟨70868, 2⟩, ⟨70682, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11381⟩⟩, ⟨.program ⟨214⟩, ⟨13981⟩⟩], [⟨.program ⟨214⟩, ⟨23540⟩⟩]⟩, (-1)⟩)

def event70873 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25986⟩⟩, .operator (⟨70868, 1⟩, ⟨70682, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25984⟩⟩]⟩, (1)⟩)

def event70874 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25986⟩⟩) (.sum [.result 70868 .summary, .result 70682 .summary])

def exact70875RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact70875RawTermsValid :
    exact70875RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70875 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25986⟩⟩) exact70875RawTerms .large 70871 (.finite 352054612209664) (some (70874))

def event70876 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27638⟩⟩) 0 ⟨25986⟩ 70875

def event70877 : Event := .predecessor (⟨.program ⟨214⟩, ⟨27638⟩⟩) 1 ⟨27636⟩ 70598

def event70878 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27638⟩⟩) (.product (.predecessor 0 70876 .coefficient) (.predecessor 1 70877 .coefficient) (⟨false, false, none, none, none⟩))

def event70879 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27638⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨27636⟩⟩]⟩) [⟨.result 70598 .coefficient, false, none⟩])

def event70880 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨27638⟩⟩) (.product (.result 70875 .summary) (.transfer 70879) (⟨false, false, none, none, none⟩))

def event70881 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27638⟩⟩, .operator (⟨70875, 0⟩, ⟨70598, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27636⟩⟩]⟩, (1)⟩)

def event70882 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27638⟩⟩, .operator (⟨70875, 1⟩, ⟨70598, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27636⟩⟩]⟩, (-1)⟩)

def event70883 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨27638⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27636⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27636⟩⟩) ⟨24096⟩ 70595)

def event70884 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨27638⟩⟩, .relation 70883 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24096⟩⟩]⟩, (-1)⟩)

def exact70885RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27636⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15817⟩⟩], [⟨.program ⟨214⟩, ⟨24096⟩⟩]⟩, (-1)⟩]

theorem exact70885RawTermsValid :
    exact70885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70885 : Event := .resultExact (⟨.program ⟨214⟩, ⟨27638⟩⟩) exact70885RawTerms .large 70878 (.finite 1292046059683262234624) (some (70880))

def event70886 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21252⟩⟩) 0 ⟨15818⟩ 3356

def event70887 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21252⟩⟩) (.authority (.relationPreimageSource ⟨41⟩))

def exact70888RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21252⟩⟩]⟩, (1)⟩]

theorem exact70888RawTermsValid :
    exact70888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70888 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21252⟩⟩) exact70888RawTerms (.finite 136065468) 70887 .exactZero (none)

def event70889 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21254⟩⟩) 0 ⟨21252⟩ 70888

def event70890 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21254⟩⟩) 1 ⟨2348⟩ 4

def event70891 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21254⟩⟩) (.scale (.predecessor 0 70889 .coefficient) (.value (.predecessor 1 70890 .coefficient)))

def exact70892RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21252⟩⟩]⟩, (1)⟩]

theorem exact70892RawTermsValid :
    exact70892RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event70892 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21254⟩⟩) exact70892RawTerms (.finite 136065468) 70891 .exactZero (none)

def event70893 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21255⟩⟩) 0 ⟨5535⟩ 65387

def event70894 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21255⟩⟩) 1 ⟨21254⟩ 70892

def event70895 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21255⟩⟩) (.product (.predecessor 0 70893 .coefficient) (.predecessor 1 70894 .coefficient) (⟨false, false, none, none, none⟩))

def event70896 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21255⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21252⟩⟩]⟩) [⟨.result 70888 .coefficient, false, none⟩])

def event70897 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21255⟩⟩) (.product (.result 65387 .summary) (.transfer 70896) (⟨false, false, none, none, none⟩))

def event70898 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21255⟩⟩, .operator (⟨65387, 0⟩, ⟨70892, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21252⟩⟩]⟩, (1)⟩)

def event70899 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21253⟩⟩)

def event70900 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event70901 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event70902 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event70903 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event70904 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event70905 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event70906 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event70907 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event70908 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 70907

def event70909 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 70905

def event70910 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 70908 .coefficient) (.value (.predecessor 1 70909 .coefficient)))

def event70911 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def eventLeaf4416 : Array AnnotatedEvent := #[
  { event := event70656
    frameStart := 0 },
  { event := event70657
    frameStart := 0 },
  { event := event70658
    frameStart := 0 },
  { event := event70659
    frameStart := 0 },
  { event := event70660
    frameStart := 0 },
  { event := event70661
    frameStart := 0 },
  { event := event70662
    frameStart := 0 },
  { event := event70663
    frameStart := 0 },
  { event := event70664
    frameStart := 0 },
  { event := event70665
    frameStart := 0 },
  { event := event70666
    frameStart := 0 },
  { event := event70667
    frameStart := 0 },
  { event := event70668
    frameStart := 0 },
  { event := event70669
    frameStart := 0 },
  { event := event70670
    frameStart := 0 },
  { event := event70671
    frameStart := 0 }
]

def eventLeaf4417 : Array AnnotatedEvent := #[
  { event := event70672
    frameStart := 0 },
  { event := event70673
    frameStart := 0 },
  { event := event70674
    frameStart := 0 },
  { event := event70675
    frameStart := 0 },
  { event := event70676
    frameStart := 0 },
  { event := event70677
    frameStart := 0 },
  { event := event70678
    frameStart := 0 },
  { event := event70679
    frameStart := 0 },
  { event := event70680
    frameStart := 0 },
  { event := event70681
    frameStart := 0 },
  { event := event70682
    frameStart := 0 },
  { event := event70683
    frameStart := 0 },
  { event := event70684
    frameStart := 0 },
  { event := event70685
    frameStart := 0 },
  { event := event70686
    frameStart := 0 },
  { event := event70687
    frameStart := 0 }
]

def eventLeaf4418 : Array AnnotatedEvent := #[
  { event := event70688
    frameStart := 0 },
  { event := event70689
    frameStart := 0 },
  { event := event70690
    frameStart := 0 },
  { event := event70691
    frameStart := 0 },
  { event := event70692
    frameStart := 0 },
  { event := event70693
    frameStart := 0 },
  { event := event70694
    frameStart := 0 },
  { event := event70695
    frameStart := 0 },
  { event := event70696
    frameStart := 70696 },
  { event := event70697
    frameStart := 70696 },
  { event := event70698
    frameStart := 70696 },
  { event := event70699
    frameStart := 70696 },
  { event := event70700
    frameStart := 70696 },
  { event := event70701
    frameStart := 70696 },
  { event := event70702
    frameStart := 70696 },
  { event := event70703
    frameStart := 70696 }
]

def eventLeaf4419 : Array AnnotatedEvent := #[
  { event := event70704
    frameStart := 70696 },
  { event := event70705
    frameStart := 70696 },
  { event := event70706
    frameStart := 70696 },
  { event := event70707
    frameStart := 70696 },
  { event := event70708
    frameStart := 70696 },
  { event := event70709
    frameStart := 70696 },
  { event := event70710
    frameStart := 70696 },
  { event := event70711
    frameStart := 70696 },
  { event := event70712
    frameStart := 70696 },
  { event := event70713
    frameStart := 70696 },
  { event := event70714
    frameStart := 70696 },
  { event := event70715
    frameStart := 70696 },
  { event := event70716
    frameStart := 70696 },
  { event := event70717
    frameStart := 70696 },
  { event := event70718
    frameStart := 70696 },
  { event := event70719
    frameStart := 70696 }
]

def eventLeaf4420 : Array AnnotatedEvent := #[
  { event := event70720
    frameStart := 70696 },
  { event := event70721
    frameStart := 70696 },
  { event := event70722
    frameStart := 70696 },
  { event := event70723
    frameStart := 70696 },
  { event := event70724
    frameStart := 70696 },
  { event := event70725
    frameStart := 70696 },
  { event := event70726
    frameStart := 70696 },
  { event := event70727
    frameStart := 70696 },
  { event := event70728
    frameStart := 70696 },
  { event := event70729
    frameStart := 70696 },
  { event := event70730
    frameStart := 70696 },
  { event := event70731
    frameStart := 70696 },
  { event := event70732
    frameStart := 70696 },
  { event := event70733
    frameStart := 70696 },
  { event := event70734
    frameStart := 70696 },
  { event := event70735
    frameStart := 70696 }
]

def eventLeaf4421 : Array AnnotatedEvent := #[
  { event := event70736
    frameStart := 70696 },
  { event := event70737
    frameStart := 70696 },
  { event := event70738
    frameStart := 70696 },
  { event := event70739
    frameStart := 70696 },
  { event := event70740
    frameStart := 70696 },
  { event := event70741
    frameStart := 70696 },
  { event := event70742
    frameStart := 70696 },
  { event := event70743
    frameStart := 70696 },
  { event := event70744
    frameStart := 70744 },
  { event := event70745
    frameStart := 70744 },
  { event := event70746
    frameStart := 70744 },
  { event := event70747
    frameStart := 70744 },
  { event := event70748
    frameStart := 70744 },
  { event := event70749
    frameStart := 70744 },
  { event := event70750
    frameStart := 70744 },
  { event := event70751
    frameStart := 70744 }
]

def eventLeaf4422 : Array AnnotatedEvent := #[
  { event := event70752
    frameStart := 70744 },
  { event := event70753
    frameStart := 70744 },
  { event := event70754
    frameStart := 70744 },
  { event := event70755
    frameStart := 70744 },
  { event := event70756
    frameStart := 70744 },
  { event := event70757
    frameStart := 70744 },
  { event := event70758
    frameStart := 70744 },
  { event := event70759
    frameStart := 70744 },
  { event := event70760
    frameStart := 70744 },
  { event := event70761
    frameStart := 70744 },
  { event := event70762
    frameStart := 70744 },
  { event := event70763
    frameStart := 70744 },
  { event := event70764
    frameStart := 70744 },
  { event := event70765
    frameStart := 70744 },
  { event := event70766
    frameStart := 70744 },
  { event := event70767
    frameStart := 70744 }
]

def eventLeaf4423 : Array AnnotatedEvent := #[
  { event := event70768
    frameStart := 70744 },
  { event := event70769
    frameStart := 70744 },
  { event := event70770
    frameStart := 70744 },
  { event := event70771
    frameStart := 70744 },
  { event := event70772
    frameStart := 70744 },
  { event := event70773
    frameStart := 70744 },
  { event := event70774
    frameStart := 70744 },
  { event := event70775
    frameStart := 70744 },
  { event := event70776
    frameStart := 70744 },
  { event := event70777
    frameStart := 70744 },
  { event := event70778
    frameStart := 70744 },
  { event := event70779
    frameStart := 70744 },
  { event := event70780
    frameStart := 70744 },
  { event := event70781
    frameStart := 70744 },
  { event := event70782
    frameStart := 70744 },
  { event := event70783
    frameStart := 70744 }
]

def eventLeaf4424 : Array AnnotatedEvent := #[
  { event := event70784
    frameStart := 70744 },
  { event := event70785
    frameStart := 70744 },
  { event := event70786
    frameStart := 70744 },
  { event := event70787
    frameStart := 70744 },
  { event := event70788
    frameStart := 70744 },
  { event := event70789
    frameStart := 70744 },
  { event := event70790
    frameStart := 70744 },
  { event := event70791
    frameStart := 70744 },
  { event := event70792
    frameStart := 70744 },
  { event := event70793
    frameStart := 70744 },
  { event := event70794
    frameStart := 70744 },
  { event := event70795
    frameStart := 70744 },
  { event := event70796
    frameStart := 70744 },
  { event := event70797
    frameStart := 70744 },
  { event := event70798
    frameStart := 70744 },
  { event := event70799
    frameStart := 70744 }
]

def eventLeaf4425 : Array AnnotatedEvent := #[
  { event := event70800
    frameStart := 70744 },
  { event := event70801
    frameStart := 70744 },
  { event := event70802
    frameStart := 70744 },
  { event := event70803
    frameStart := 70744 },
  { event := event70804
    frameStart := 70744 },
  { event := event70805
    frameStart := 70744 },
  { event := event70806
    frameStart := 70744 },
  { event := event70807
    frameStart := 70744 },
  { event := event70808
    frameStart := 70744 },
  { event := event70809
    frameStart := 70744 },
  { event := event70810
    frameStart := 70744 },
  { event := event70811
    frameStart := 70744 },
  { event := event70812
    frameStart := 70744 },
  { event := event70813
    frameStart := 70744 },
  { event := event70814
    frameStart := 70744 },
  { event := event70815
    frameStart := 70744 }
]

def eventLeaf4426 : Array AnnotatedEvent := #[
  { event := event70816
    frameStart := 70744 },
  { event := event70817
    frameStart := 70744 },
  { event := event70818
    frameStart := 70744 },
  { event := event70819
    frameStart := 70744 },
  { event := event70820
    frameStart := 70744 },
  { event := event70821
    frameStart := 70744 },
  { event := event70822
    frameStart := 70744 },
  { event := event70823
    frameStart := 70744 },
  { event := event70824
    frameStart := 70744 },
  { event := event70825
    frameStart := 70744 },
  { event := event70826
    frameStart := 70744 },
  { event := event70827
    frameStart := 70744 },
  { event := event70828
    frameStart := 70744 },
  { event := event70829
    frameStart := 70744 },
  { event := event70830
    frameStart := 70744 },
  { event := event70831
    frameStart := 70744 }
]

def eventLeaf4427 : Array AnnotatedEvent := #[
  { event := event70832
    frameStart := 70744 },
  { event := event70833
    frameStart := 70744 },
  { event := event70834
    frameStart := 70744 },
  { event := event70835
    frameStart := 70744 },
  { event := event70836
    frameStart := 70744 },
  { event := event70837
    frameStart := 70744 },
  { event := event70838
    frameStart := 70744 },
  { event := event70839
    frameStart := 70744 },
  { event := event70840
    frameStart := 70744 },
  { event := event70841
    frameStart := 70744 },
  { event := event70842
    frameStart := 70744 },
  { event := event70843
    frameStart := 70744 },
  { event := event70844
    frameStart := 70744 },
  { event := event70845
    frameStart := 70744 },
  { event := event70846
    frameStart := 70744 },
  { event := event70847
    frameStart := 70744 }
]

def eventLeaf4428 : Array AnnotatedEvent := #[
  { event := event70848
    frameStart := 70744 },
  { event := event70849
    frameStart := 70744 },
  { event := event70850
    frameStart := 70744 },
  { event := event70851
    frameStart := 70744 },
  { event := event70852
    frameStart := 70744 },
  { event := event70853
    frameStart := 70744 },
  { event := event70854
    frameStart := 70744 },
  { event := event70855
    frameStart := 70744 },
  { event := event70856
    frameStart := 70744 },
  { event := event70857
    frameStart := 70744 },
  { event := event70858
    frameStart := 70744 },
  { event := event70859
    frameStart := 70744 },
  { event := event70860
    frameStart := 70744 },
  { event := event70861
    frameStart := 70744 },
  { event := event70862
    frameStart := 0 },
  { event := event70863
    frameStart := 0 }
]

def eventLeaf4429 : Array AnnotatedEvent := #[
  { event := event70864
    frameStart := 0 },
  { event := event70865
    frameStart := 0 },
  { event := event70866
    frameStart := 0 },
  { event := event70867
    frameStart := 0 },
  { event := event70868
    frameStart := 0 },
  { event := event70869
    frameStart := 0 },
  { event := event70870
    frameStart := 0 },
  { event := event70871
    frameStart := 0 },
  { event := event70872
    frameStart := 0 },
  { event := event70873
    frameStart := 0 },
  { event := event70874
    frameStart := 0 },
  { event := event70875
    frameStart := 0 },
  { event := event70876
    frameStart := 0 },
  { event := event70877
    frameStart := 0 },
  { event := event70878
    frameStart := 0 },
  { event := event70879
    frameStart := 0 }
]

def eventLeaf4430 : Array AnnotatedEvent := #[
  { event := event70880
    frameStart := 0 },
  { event := event70881
    frameStart := 0 },
  { event := event70882
    frameStart := 0 },
  { event := event70883
    frameStart := 0 },
  { event := event70884
    frameStart := 0 },
  { event := event70885
    frameStart := 0 },
  { event := event70886
    frameStart := 0 },
  { event := event70887
    frameStart := 0 },
  { event := event70888
    frameStart := 0 },
  { event := event70889
    frameStart := 0 },
  { event := event70890
    frameStart := 0 },
  { event := event70891
    frameStart := 0 },
  { event := event70892
    frameStart := 0 },
  { event := event70893
    frameStart := 0 },
  { event := event70894
    frameStart := 0 },
  { event := event70895
    frameStart := 0 }
]

def eventLeaf4431 : Array AnnotatedEvent := #[
  { event := event70896
    frameStart := 0 },
  { event := event70897
    frameStart := 0 },
  { event := event70898
    frameStart := 0 },
  { event := event70899
    frameStart := 70899 },
  { event := event70900
    frameStart := 70899 },
  { event := event70901
    frameStart := 70899 },
  { event := event70902
    frameStart := 70899 },
  { event := event70903
    frameStart := 70899 },
  { event := event70904
    frameStart := 70899 },
  { event := event70905
    frameStart := 70899 },
  { event := event70906
    frameStart := 70899 },
  { event := event70907
    frameStart := 70899 },
  { event := event70908
    frameStart := 70899 },
  { event := event70909
    frameStart := 70899 },
  { event := event70910
    frameStart := 70899 },
  { event := event70911
    frameStart := 70899 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events276
