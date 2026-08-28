import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1202

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact307712RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16910⟩⟩]⟩, (1)⟩]

theorem exact307712RawTermsValid :
    exact307712RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307712 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16910⟩⟩) exact307712RawTerms .large 307711 .exactZero (none)

def event307713 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17474⟩⟩) 0 ⟨16910⟩ 307712

def event307714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17474⟩⟩) (.authority (.operator))

def exact307715RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17474⟩⟩]⟩, (1)⟩]

theorem exact307715RawTermsValid :
    exact307715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307715 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17474⟩⟩) exact307715RawTerms (.finite 8192) 307714 .exactZero (none)

def event307716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17476⟩⟩) 0 ⟨17251⟩ 302735

def event307717 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17476⟩⟩) 1 ⟨17474⟩ 307715

def event307718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17476⟩⟩) (.product (.predecessor 0 307716 .coefficient) (.predecessor 1 307717 .coefficient) (⟨false, false, none, none, none⟩))

def event307719 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17476⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨17474⟩⟩]⟩) [⟨.result 307715 .coefficient, false, none⟩])

def event307720 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17476⟩⟩) (.product (.result 302735 .summary) (.transfer 307719) (⟨false, false, none, none, none⟩))

def event307721 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17476⟩⟩, .operator (⟨302735, 0⟩, ⟨307715, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17474⟩⟩]⟩, (1)⟩)

def event307722 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17476⟩⟩, .operator (⟨302735, 1⟩, ⟨307715, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17474⟩⟩]⟩, (-1)⟩)

def event307723 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17476⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17474⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17474⟩⟩) ⟨16910⟩ 307712)

def event307724 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17476⟩⟩, .relation 307723 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨16910⟩⟩]⟩, (-1)⟩)

def exact307725RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17474⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨16910⟩⟩]⟩, (-1)⟩]

theorem exact307725RawTermsValid :
    exact307725RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307725 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17476⟩⟩) exact307725RawTerms .large 307718 (.finite 32188807212483504816668771614720) (some (307720))

def event307726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16392⟩⟩) 0 ⟨15709⟩ 14698

def event307727 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16392⟩⟩) (.authority (.relationPreimageSource ⟨56⟩))

def exact307728RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16392⟩⟩]⟩, (1)⟩]

theorem exact307728RawTermsValid :
    exact307728RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307728 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16392⟩⟩) exact307728RawTerms (.finite 5647228698) 307727 .exactZero (none)

def event307729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16394⟩⟩) 0 ⟨16392⟩ 307728

def event307730 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16394⟩⟩) 1 ⟨2370⟩ 4

def event307731 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16394⟩⟩) (.scale (.predecessor 0 307729 .coefficient) (.value (.predecessor 1 307730 .coefficient)))

def exact307732RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16392⟩⟩]⟩, (1)⟩]

theorem exact307732RawTermsValid :
    exact307732RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307732 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16394⟩⟩) exact307732RawTerms (.finite 5647228698) 307731 .exactZero (none)

def event307733 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16395⟩⟩) 0 ⟨2380⟩ 295195

def event307734 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16395⟩⟩) 1 ⟨16394⟩ 307732

def event307735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16395⟩⟩) (.product (.predecessor 0 307733 .coefficient) (.predecessor 1 307734 .coefficient) (⟨false, false, none, none, none⟩))

def event307736 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16395⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨16392⟩⟩]⟩) [⟨.result 307728 .coefficient, false, none⟩])

def event307737 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16395⟩⟩) (.product (.result 295195 .summary) (.transfer 307736) (⟨false, false, none, none, none⟩))

def event307738 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16395⟩⟩, .operator (⟨295195, 0⟩, ⟨307732, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16392⟩⟩]⟩, (1)⟩)

def event307739 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨16393⟩⟩)

def event307740 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event307741 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event307742 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event307743 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event307744 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 307743

def event307745 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 307741

def event307746 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 307744 .coefficient) (.value (.predecessor 1 307745 .coefficient)))

def event307747 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event307748 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15234⟩⟩) 0 ⟨392⟩ 307747

def event307749 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15234⟩⟩) (.authority (.programFamilyFact))

def exact307750RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15234⟩⟩], []⟩, (1)⟩]

theorem exact307750RawTermsValid :
    exact307750RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307750 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15234⟩⟩) exact307750RawTerms (.finite 2) 307749 .exactZero (none)

def event307751 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12231⟩⟩) 0 ⟨392⟩ 307747

def event307752 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12231⟩⟩) (.authority (.programFamilyFact))

def exact307753RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12231⟩⟩], []⟩, (1)⟩]

theorem exact307753RawTermsValid :
    exact307753RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307753 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12231⟩⟩) exact307753RawTerms (.finite 2) 307752 .exactZero (none)

def event307754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15235⟩⟩) 0 ⟨12231⟩ 307753

def event307755 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15235⟩⟩) 1 ⟨15234⟩ 307750

def event307756 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15235⟩⟩) (.product (.predecessor 0 307754 .coefficient) (.predecessor 1 307755 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event307757 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15235⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], []⟩) [⟨.result 307753 .coefficient, true, some 1⟩, ⟨.result 307750 .coefficient, true, some 1⟩])

def event307758 : Event := .survivorFold (1) 307757

def exact307759RawTerms : List Term := []

theorem exact307759RawTermsValid :
    exact307759RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307759 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15235⟩⟩) exact307759RawTerms (.finite 4) 307756 (.finite 4) (some (307757))

def event307760 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15236⟩⟩) 0 ⟨15235⟩ 307759

def event307761 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15236⟩⟩) (.identity (.predecessor 0 307760 .coefficient))

def event307762 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15236⟩⟩) (.finite 4)

def event307763 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15708⟩⟩) 0 ⟨15236⟩ 307762

def event307764 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15708⟩⟩) (.authority (.programFamilyFact))

def exact307765RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], []⟩, (1)⟩]

theorem exact307765RawTermsValid :
    exact307765RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307765 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15708⟩⟩) exact307765RawTerms (.finite 2) 307764 .exactZero (none)

def event307766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15709⟩⟩) 0 ⟨15708⟩ 307765

def event307767 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15709⟩⟩) (.identity (.predecessor 0 307766 .coefficient))

def event307768 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15709⟩⟩) (.finite 2)

def event307769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16392⟩⟩) 0 ⟨15709⟩ 307768

def event307770 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16392⟩⟩) (.authority (.relationPreimageSource ⟨56⟩))

def exact307771RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16392⟩⟩]⟩, (1)⟩]

theorem exact307771RawTermsValid :
    exact307771RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307771 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16392⟩⟩) exact307771RawTerms (.finite 5647228698) 307770 .exactZero (none)

def event307772 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact307773RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact307773RawTermsValid :
    exact307773RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307773 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact307773RawTerms .large 307772 .exactZero (none)

def event307774 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16393⟩⟩) 0 ⟨35⟩ 307773

def event307775 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16393⟩⟩) 1 ⟨16392⟩ 307771

def event307776 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16393⟩⟩) (.product (.predecessor 0 307774 .coefficient) (.predecessor 1 307775 .coefficient) (⟨false, false, none, none, none⟩))

def event307777 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16393⟩⟩, .operator (⟨307773, 0⟩, ⟨307771, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16392⟩⟩]⟩, (1)⟩)

def exact307778RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16392⟩⟩]⟩, (1)⟩]

theorem exact307778RawTermsValid :
    exact307778RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307778 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16393⟩⟩) exact307778RawTerms .large 307776 .exactZero (none)

def event307779 : Event := .preFoldPolynomial 307778 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16392⟩⟩]⟩, (1)⟩] .exactZero none

def exact307780RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16392⟩⟩]⟩, (1)⟩]

def event307780 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨16393⟩⟩) 307779 exact307780RawTerms .large 307776 .exactZero (none)

def event307781 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨17480⟩⟩)

def event307782 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event307783 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event307784 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event307785 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event307786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 307785

def event307787 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 307783

def event307788 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 307786 .coefficient) (.value (.predecessor 1 307787 .coefficient)))

def event307789 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event307790 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15234⟩⟩) 0 ⟨392⟩ 307789

def event307791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15234⟩⟩) (.authority (.programFamilyFact))

def exact307792RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15234⟩⟩], []⟩, (1)⟩]

theorem exact307792RawTermsValid :
    exact307792RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307792 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15234⟩⟩) exact307792RawTerms (.finite 2) 307791 .exactZero (none)

def event307793 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12231⟩⟩) 0 ⟨392⟩ 307789

def event307794 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12231⟩⟩) (.authority (.programFamilyFact))

def exact307795RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12231⟩⟩], []⟩, (1)⟩]

theorem exact307795RawTermsValid :
    exact307795RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307795 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12231⟩⟩) exact307795RawTerms (.finite 2) 307794 .exactZero (none)

def event307796 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15235⟩⟩) 0 ⟨12231⟩ 307795

def event307797 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15235⟩⟩) 1 ⟨15234⟩ 307792

def event307798 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15235⟩⟩) (.product (.predecessor 0 307796 .coefficient) (.predecessor 1 307797 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event307799 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15235⟩⟩, .operator (⟨307795, 0⟩, ⟨307792, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], []⟩, (1)⟩)

def exact307800RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12231⟩⟩, ⟨.program ⟨257⟩, ⟨15234⟩⟩], []⟩, (1)⟩]

theorem exact307800RawTermsValid :
    exact307800RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307800 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15235⟩⟩) exact307800RawTerms (.finite 4) 307798 .exactZero (none)

def event307801 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15236⟩⟩) 0 ⟨15235⟩ 307800

def event307802 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15236⟩⟩) (.identity (.predecessor 0 307801 .coefficient))

def event307803 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15236⟩⟩) (.finite 4)

def event307804 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15708⟩⟩) 0 ⟨15236⟩ 307803

def event307805 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15708⟩⟩) (.authority (.programFamilyFact))

def exact307806RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], []⟩, (1)⟩]

theorem exact307806RawTermsValid :
    exact307806RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307806 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15708⟩⟩) exact307806RawTerms (.finite 2) 307805 .exactZero (none)

def event307807 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15709⟩⟩) 0 ⟨15708⟩ 307806

def event307808 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15709⟩⟩) (.identity (.predecessor 0 307807 .coefficient))

def event307809 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15709⟩⟩) (.finite 2)

def event307810 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16909⟩⟩) 0 ⟨15709⟩ 307809

def event307811 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16909⟩⟩) (.authority (.programFamilyFact))

def event307812 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨16909⟩⟩) (.finite 3720)

def event307813 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event307814 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16910⟩⟩) 0 ⟨7177⟩ 307813

def event307815 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16910⟩⟩) 1 ⟨16909⟩ 307812

def event307816 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16910⟩⟩) (.authority (.operator))

def exact307817RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16910⟩⟩]⟩, (1)⟩]

theorem exact307817RawTermsValid :
    exact307817RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307817 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16910⟩⟩) exact307817RawTerms .large 307816 .exactZero (none)

def event307818 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17474⟩⟩) 0 ⟨16910⟩ 307817

def event307819 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17474⟩⟩) (.authority (.operator))

def exact307820RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17474⟩⟩]⟩, (1)⟩]

theorem exact307820RawTermsValid :
    exact307820RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307820 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17474⟩⟩) exact307820RawTerms (.finite 8192) 307819 .exactZero (none)

def event307821 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event307822 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event307823 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17166⟩⟩) 0 ⟨15709⟩ 307809

def event307824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17166⟩⟩) 1 ⟨136⟩ 307822

def event307825 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17166⟩⟩) (.sum [.predecessor 0 307823 .coefficient, .predecessor 1 307824 .coefficient])

def event307826 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17166⟩⟩) (.finite 2)

def event307827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17167⟩⟩) 0 ⟨17166⟩ 307826

def event307828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17167⟩⟩) (.identity (.predecessor 0 307827 .coefficient))

def exact307829RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], []⟩, (1)⟩]

theorem exact307829RawTermsValid :
    exact307829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307829 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17167⟩⟩) exact307829RawTerms (.finite 2) 307828 .exactZero (none)

def event307830 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact307831RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact307831RawTermsValid :
    exact307831RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307831 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact307831RawTerms .large 307830 .exactZero (none)

def event307832 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17168⟩⟩) 0 ⟨6908⟩ 307831

def event307833 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17168⟩⟩) 1 ⟨17167⟩ 307829

def event307834 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17168⟩⟩) (.product (.predecessor 0 307832 .coefficient) (.predecessor 1 307833 .coefficient) (⟨false, false, none, none, none⟩))

def event307835 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17168⟩⟩, .operator (⟨307831, 0⟩, ⟨307829, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact307836RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact307836RawTermsValid :
    exact307836RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307836 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17168⟩⟩) exact307836RawTerms .large 307834 .exactZero (none)

def event307837 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 307813

def event307838 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact307839RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact307839RawTermsValid :
    exact307839RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307839 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact307839RawTerms .large 307838 .exactZero (none)

def event307840 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17169⟩⟩) 0 ⟨7179⟩ 307839

def event307841 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17169⟩⟩) 1 ⟨17168⟩ 307836

def event307842 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17169⟩⟩) (.sum [.predecessor 0 307840 .coefficient, .predecessor 1 307841 .coefficient])

def exact307843RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact307843RawTermsValid :
    exact307843RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307843 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17169⟩⟩) exact307843RawTerms .large 307842 .exactZero (none)

def event307844 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17475⟩⟩) 0 ⟨17169⟩ 307843

def event307845 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17475⟩⟩) 1 ⟨17474⟩ 307820

def event307846 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17475⟩⟩) (.product (.predecessor 0 307844 .coefficient) (.predecessor 1 307845 .coefficient) (⟨false, false, none, none, none⟩))

def event307847 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17475⟩⟩, .operator (⟨307843, 0⟩, ⟨307820, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17474⟩⟩]⟩, (1)⟩)

def event307848 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17475⟩⟩, .operator (⟨307843, 1⟩, ⟨307820, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17474⟩⟩]⟩, (-1)⟩)

def event307849 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17475⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17474⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17474⟩⟩) ⟨16910⟩ 307817)

def event307850 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17475⟩⟩, .relation 307849 0, ⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨16910⟩⟩]⟩, (-1)⟩)

def exact307851RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17474⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨16910⟩⟩]⟩, (-1)⟩]

theorem exact307851RawTermsValid :
    exact307851RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307851 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17475⟩⟩) exact307851RawTerms .large 307846 .exactZero (none)

def event307852 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15870⟩⟩) 0 ⟨15709⟩ 307809

def event307853 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15870⟩⟩) (.authority (.programFamilyFact))

def exact307854RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15870⟩⟩], []⟩, (1)⟩]

theorem exact307854RawTermsValid :
    exact307854RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307854 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15870⟩⟩) exact307854RawTerms (.finite 2) 307853 .exactZero (none)

def event307855 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15873⟩⟩) 0 ⟨6908⟩ 307831

def event307856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15873⟩⟩) 1 ⟨15870⟩ 307854

def event307857 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15873⟩⟩) (.product (.predecessor 0 307855 .coefficient) (.predecessor 1 307856 .coefficient) (⟨false, true, none, none, some 1⟩))

def event307858 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15873⟩⟩, .operator (⟨307831, 0⟩, ⟨307854, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact307859RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact307859RawTermsValid :
    exact307859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307859 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15873⟩⟩) exact307859RawTerms .large 307857 .exactZero (none)

def event307860 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7197⟩⟩) 0 ⟨7177⟩ 307813

def event307861 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7197⟩⟩) (.authority (.operator))

def exact307862RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩]

theorem exact307862RawTermsValid :
    exact307862RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307862 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7197⟩⟩) exact307862RawTerms .large 307861 .exactZero (none)

def event307863 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15874⟩⟩) 0 ⟨7197⟩ 307862

def event307864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15874⟩⟩) 1 ⟨15873⟩ 307859

def event307865 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15874⟩⟩) (.sum [.predecessor 0 307863 .coefficient, .predecessor 1 307864 .coefficient])

def exact307866RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact307866RawTermsValid :
    exact307866RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307866 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15874⟩⟩) exact307866RawTerms .large 307865 .exactZero (none)

def event307867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17480⟩⟩) 0 ⟨15874⟩ 307866

def event307868 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17480⟩⟩) 1 ⟨17475⟩ 307851

def event307869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17480⟩⟩) (.sum [.predecessor 0 307867 .coefficient, .predecessor 1 307868 .coefficient])

def exact307870RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17474⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨16910⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact307870RawTermsValid :
    exact307870RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307870 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17480⟩⟩) exact307870RawTerms .large 307869 .exactZero (none)

def event307871 : Event := .preFoldPolynomial 307870 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17474⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨16910⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact307872RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17474⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨16910⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event307872 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17480⟩⟩) 307871 exact307872RawTerms .large 307869 .exactZero (none)

def event307873 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15709⟩⟩) ⟨⟨76⟩, ⟨56⟩, ⟨135⟩⟩ ⟨307739, 307873⟩

def event307874 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16395⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16392⟩⟩]⟩) (1) 0 2 (.universal 307873 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16392⟩⟩]⟩) (none) 307872)

def event307875 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16395⟩⟩, .relation 307874 1, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩)

def event307876 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16395⟩⟩, .relation 307874 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17474⟩⟩]⟩, (-1)⟩)

def event307877 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16395⟩⟩, .relation 307874 2, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨16910⟩⟩]⟩, (1)⟩)

def event307878 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16395⟩⟩, .relation 307874 3, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact307879RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17474⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨16910⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact307879RawTermsValid :
    exact307879RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307879 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16395⟩⟩) exact307879RawTerms .large 307735 (.finite 202072841853861888) (some (307737))

def event307880 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17477⟩⟩) 0 ⟨16395⟩ 307879

def event307881 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17477⟩⟩) 1 ⟨17476⟩ 307725

def event307882 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17477⟩⟩) (.sum [.predecessor 0 307880 .coefficient, .predecessor 1 307881 .coefficient])

def event307883 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17477⟩⟩, .operator (⟨307879, 0⟩, ⟨307725, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17474⟩⟩]⟩, (1)⟩)

def event307884 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17477⟩⟩, .operator (⟨307879, 2⟩, ⟨307725, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15708⟩⟩], [⟨.program ⟨257⟩, ⟨16910⟩⟩]⟩, (-1)⟩)

def event307885 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17477⟩⟩) (.sum [.result 307879 .summary, .result 307725 .summary])

def exact307886RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact307886RawTermsValid :
    exact307886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307886 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17477⟩⟩) exact307886RawTerms .large 307882 (.finite 32188807212483706889510625476608) (some (307885))

def event307887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17478⟩⟩) 0 ⟨17477⟩ 307886

def event307888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17478⟩⟩) 1 ⟨7172⟩ 15882

def event307889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17478⟩⟩) (.product (.predecessor 0 307887 .coefficient) (.predecessor 1 307888 .coefficient) (⟨false, false, none, none, none⟩))

def event307890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17478⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) [⟨.result 15878 .coefficient, false, none⟩])

def event307891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17478⟩⟩) (.product (.result 307886 .summary) (.transfer 307890) (⟨false, false, none, none, none⟩))

def event307892 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17478⟩⟩, .operator (⟨307886, 0⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩)

def event307893 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17478⟩⟩, .operator (⟨307886, 1⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩)

def event307894 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17478⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7171⟩⟩) ⟨7051⟩ 15875)

def event307895 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17478⟩⟩, .relation 307894 0, ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact307896RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact307896RawTermsValid :
    exact307896RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307896 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17478⟩⟩) exact307896RawTerms .large 307889 (.finite 345624685687166110058245054666339432529920) (some (307891))

def event307897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7055⟩⟩) 0 ⟨6727⟩ 723

def event307898 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7055⟩⟩) 1 ⟨6910⟩ 32

def event307899 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7055⟩⟩) (.tensor (.predecessor 0 307897 .coefficient) (.predecessor 1 307898 .coefficient) true false)

def event307900 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7055⟩⟩, .operator (⟨723, 0⟩, ⟨32, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact307901RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact307901RawTermsValid :
    exact307901RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307901 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7055⟩⟩) exact307901RawTerms .large 307899 .exactZero (none)

def event307902 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7440⟩⟩) 0 ⟨2377⟩ 27

def event307903 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7440⟩⟩) 1 ⟨7292⟩ 15896

def event307904 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7440⟩⟩) (.product (.predecessor 0 307902 .coefficient) (.predecessor 1 307903 .coefficient) (⟨false, false, none, none, none⟩))

def event307905 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7440⟩⟩, .operator (⟨27, 0⟩, ⟨15896, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩)

def exact307906RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact307906RawTermsValid :
    exact307906RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307906 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7440⟩⟩) exact307906RawTerms .large 307904 .exactZero (none)

def event307907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9243⟩⟩) 0 ⟨7440⟩ 307906

def event307908 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9243⟩⟩) 1 ⟨7055⟩ 307901

def event307909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9243⟩⟩) (.sum [.predecessor 0 307907 .coefficient, .predecessor 1 307908 .coefficient])

def exact307910RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact307910RawTermsValid :
    exact307910RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307910 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9243⟩⟩) exact307910RawTerms .large 307909 .exactZero (none)

def event307911 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9244⟩⟩) 0 ⟨9243⟩ 307910

def event307912 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9244⟩⟩) 1 ⟨118⟩ 31516

def event307913 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9244⟩⟩) (.sum [.predecessor 0 307911 .coefficient, .predecessor 1 307912 .coefficient])

def event307914 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9244⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) [⟨.result 31516 .coefficient, false, none⟩])

def event307915 : Event := .survivorFold (1) 307914

def exact307916RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact307916RawTermsValid :
    exact307916RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307916 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9244⟩⟩) exact307916RawTerms .large 307913 (.finite 26) (some (307914))

def event307917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9447⟩⟩) 0 ⟨9244⟩ 307916

def event307918 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9447⟩⟩) 1 ⟨9244⟩ 307916

def event307919 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9447⟩⟩) (.sum [.predecessor 0 307917 .coefficient, .predecessor 1 307918 .coefficient])

def event307920 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9447⟩⟩, .operator (⟨307916, 1⟩, ⟨307916, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6727⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event307921 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9447⟩⟩, .operator (⟨307916, 0⟩, ⟨307916, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (-1)⟩)

def event307922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9447⟩⟩) (.sum [.result 307916 .summary, .result 307916 .summary])

def exact307923RawTerms : List Term := []

theorem exact307923RawTermsValid :
    exact307923RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307923 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9447⟩⟩) exact307923RawTerms .large 307919 (.finite 52) (some (307922))

def event307924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17479⟩⟩) 0 ⟨9447⟩ 307923

def event307925 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17479⟩⟩) 1 ⟨17478⟩ 307896

def event307926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17479⟩⟩) (.sum [.predecessor 0 307924 .coefficient, .predecessor 1 307925 .coefficient])

def event307927 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17479⟩⟩) (.sum [.result 307923 .summary, .result 307896 .summary])

def exact307928RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact307928RawTermsValid :
    exact307928RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307928 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17479⟩⟩) exact307928RawTerms .large 307926 (.finite 345624685687166110058245054666339432529972) (some (307927))

def event307929 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20340⟩⟩) 0 ⟨17479⟩ 307928

def event307930 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20340⟩⟩) 1 ⟨20339⟩ 307708

def event307931 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20340⟩⟩) (.sum [.predecessor 0 307929 .coefficient, .predecessor 1 307930 .coefficient])

def event307932 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20340⟩⟩) (.sum [.result 307928 .summary, .result 307708 .summary])

def exact307933RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact307933RawTermsValid :
    exact307933RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307933 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20340⟩⟩) exact307933RawTerms .large 307931 (.finite 691250426059631610003352154589745737891892) (some (307932))

def event307934 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23560⟩⟩) 0 ⟨20340⟩ 307933

def event307935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23560⟩⟩) 1 ⟨23559⟩ 307520

def event307936 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23560⟩⟩) (.sum [.predecessor 0 307934 .coefficient, .predecessor 1 307935 .coefficient])

def event307937 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23560⟩⟩) (.sum [.result 307933 .summary, .result 307520 .summary])

def exact307938RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact307938RawTermsValid :
    exact307938RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307938 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23560⟩⟩) exact307938RawTerms .large 307936 (.finite 1036877221117396499835321299770218916085812) (some (307937))

def event307939 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33580⟩⟩) 0 ⟨23560⟩ 307938

def event307940 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33580⟩⟩) 1 ⟨33579⟩ 307332

def event307941 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33580⟩⟩) (.sum [.predecessor 0 307939 .coefficient, .predecessor 1 307940 .coefficient])

def event307942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33580⟩⟩) (.sum [.result 307938 .summary, .result 307332 .summary])

def exact307943RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact307943RawTermsValid :
    exact307943RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307943 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33580⟩⟩) exact307943RawTerms .large 307941 (.finite 1382506125545760169441014535464825839943732) (some (307942))

def event307944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52640⟩⟩) 0 ⟨33580⟩ 307943

def event307945 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52640⟩⟩) 1 ⟨52639⟩ 307144

def event307946 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52640⟩⟩) (.sum [.predecessor 0 307944 .coefficient, .predecessor 1 307945 .coefficient])

def event307947 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52640⟩⟩) (.sum [.result 307943 .summary, .result 307144 .summary])

def exact307948RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact307948RawTermsValid :
    exact307948RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307948 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52640⟩⟩) exact307948RawTerms .large 307946 (.finite 1728139248715321398594155952187700255129652) (some (307947))

def event307949 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55620⟩⟩) 0 ⟨52640⟩ 307948

def event307950 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55620⟩⟩) 1 ⟨55619⟩ 306956

def event307951 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55620⟩⟩) (.sum [.predecessor 0 307949 .coefficient, .predecessor 1 307950 .coefficient])

def event307952 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55620⟩⟩) (.sum [.result 307948 .summary, .result 306956 .summary])

def exact307953RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact307953RawTermsValid :
    exact307953RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307953 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55620⟩⟩) exact307953RawTerms .large 307951 (.finite 2073774481255481407521021459424708415979572) (some (307952))

def event307954 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58600⟩⟩) 0 ⟨55620⟩ 307953

def event307955 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58600⟩⟩) 1 ⟨58599⟩ 306768

def event307956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58600⟩⟩) (.sum [.predecessor 0 307954 .coefficient, .predecessor 1 307955 .coefficient])

def event307957 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58600⟩⟩) (.sum [.result 307953 .summary, .result 306768 .summary])

def exact307958RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact307958RawTermsValid :
    exact307958RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307958 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58600⟩⟩) exact307958RawTerms .large 307956 (.finite 2419413932536838975995335147689984068157492) (some (307957))

def event307959 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61580⟩⟩) 0 ⟨58600⟩ 307958

def event307960 : Event := .predecessor (⟨.program ⟨257⟩, ⟨61580⟩⟩) 1 ⟨61579⟩ 306580

def event307961 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61580⟩⟩) (.sum [.predecessor 0 307959 .coefficient, .predecessor 1 307960 .coefficient])

def event307962 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨61580⟩⟩) (.sum [.result 307958 .summary, .result 306580 .summary])

def exact307963RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56935⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨50975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18671⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15870⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact307963RawTermsValid :
    exact307963RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event307963 : Event := .resultExact (⟨.program ⟨257⟩, ⟨61580⟩⟩) exact307963RawTerms .large 307961 (.finite 2765055493188795324243372926469393465999412) (some (307962))

def event307964 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64560⟩⟩) 0 ⟨61580⟩ 307963

def event307965 : Event := .predecessor (⟨.program ⟨257⟩, ⟨64560⟩⟩) 1 ⟨64559⟩ 306392

def event307966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64560⟩⟩) (.sum [.predecessor 0 307964 .coefficient, .predecessor 1 307965 .coefficient])

def event307967 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨64560⟩⟩) (.sum [.result 307963 .summary, .result 306392 .summary])

def eventLeaf19232 : Array AnnotatedEvent := #[
  { event := event307712
    frameStart := 0 },
  { event := event307713
    frameStart := 0 },
  { event := event307714
    frameStart := 0 },
  { event := event307715
    frameStart := 0 },
  { event := event307716
    frameStart := 0 },
  { event := event307717
    frameStart := 0 },
  { event := event307718
    frameStart := 0 },
  { event := event307719
    frameStart := 0 },
  { event := event307720
    frameStart := 0 },
  { event := event307721
    frameStart := 0 },
  { event := event307722
    frameStart := 0 },
  { event := event307723
    frameStart := 0 },
  { event := event307724
    frameStart := 0 },
  { event := event307725
    frameStart := 0 },
  { event := event307726
    frameStart := 0 },
  { event := event307727
    frameStart := 0 }
]

def eventLeaf19233 : Array AnnotatedEvent := #[
  { event := event307728
    frameStart := 0 },
  { event := event307729
    frameStart := 0 },
  { event := event307730
    frameStart := 0 },
  { event := event307731
    frameStart := 0 },
  { event := event307732
    frameStart := 0 },
  { event := event307733
    frameStart := 0 },
  { event := event307734
    frameStart := 0 },
  { event := event307735
    frameStart := 0 },
  { event := event307736
    frameStart := 0 },
  { event := event307737
    frameStart := 0 },
  { event := event307738
    frameStart := 0 },
  { event := event307739
    frameStart := 307739 },
  { event := event307740
    frameStart := 307739 },
  { event := event307741
    frameStart := 307739 },
  { event := event307742
    frameStart := 307739 },
  { event := event307743
    frameStart := 307739 }
]

def eventLeaf19234 : Array AnnotatedEvent := #[
  { event := event307744
    frameStart := 307739 },
  { event := event307745
    frameStart := 307739 },
  { event := event307746
    frameStart := 307739 },
  { event := event307747
    frameStart := 307739 },
  { event := event307748
    frameStart := 307739 },
  { event := event307749
    frameStart := 307739 },
  { event := event307750
    frameStart := 307739 },
  { event := event307751
    frameStart := 307739 },
  { event := event307752
    frameStart := 307739 },
  { event := event307753
    frameStart := 307739 },
  { event := event307754
    frameStart := 307739 },
  { event := event307755
    frameStart := 307739 },
  { event := event307756
    frameStart := 307739 },
  { event := event307757
    frameStart := 307739 },
  { event := event307758
    frameStart := 307739 },
  { event := event307759
    frameStart := 307739 }
]

def eventLeaf19235 : Array AnnotatedEvent := #[
  { event := event307760
    frameStart := 307739 },
  { event := event307761
    frameStart := 307739 },
  { event := event307762
    frameStart := 307739 },
  { event := event307763
    frameStart := 307739 },
  { event := event307764
    frameStart := 307739 },
  { event := event307765
    frameStart := 307739 },
  { event := event307766
    frameStart := 307739 },
  { event := event307767
    frameStart := 307739 },
  { event := event307768
    frameStart := 307739 },
  { event := event307769
    frameStart := 307739 },
  { event := event307770
    frameStart := 307739 },
  { event := event307771
    frameStart := 307739 },
  { event := event307772
    frameStart := 307739 },
  { event := event307773
    frameStart := 307739 },
  { event := event307774
    frameStart := 307739 },
  { event := event307775
    frameStart := 307739 }
]

def eventLeaf19236 : Array AnnotatedEvent := #[
  { event := event307776
    frameStart := 307739 },
  { event := event307777
    frameStart := 307739 },
  { event := event307778
    frameStart := 307739 },
  { event := event307779
    frameStart := 307739 },
  { event := event307780
    frameStart := 307739 },
  { event := event307781
    frameStart := 307781 },
  { event := event307782
    frameStart := 307781 },
  { event := event307783
    frameStart := 307781 },
  { event := event307784
    frameStart := 307781 },
  { event := event307785
    frameStart := 307781 },
  { event := event307786
    frameStart := 307781 },
  { event := event307787
    frameStart := 307781 },
  { event := event307788
    frameStart := 307781 },
  { event := event307789
    frameStart := 307781 },
  { event := event307790
    frameStart := 307781 },
  { event := event307791
    frameStart := 307781 }
]

def eventLeaf19237 : Array AnnotatedEvent := #[
  { event := event307792
    frameStart := 307781 },
  { event := event307793
    frameStart := 307781 },
  { event := event307794
    frameStart := 307781 },
  { event := event307795
    frameStart := 307781 },
  { event := event307796
    frameStart := 307781 },
  { event := event307797
    frameStart := 307781 },
  { event := event307798
    frameStart := 307781 },
  { event := event307799
    frameStart := 307781 },
  { event := event307800
    frameStart := 307781 },
  { event := event307801
    frameStart := 307781 },
  { event := event307802
    frameStart := 307781 },
  { event := event307803
    frameStart := 307781 },
  { event := event307804
    frameStart := 307781 },
  { event := event307805
    frameStart := 307781 },
  { event := event307806
    frameStart := 307781 },
  { event := event307807
    frameStart := 307781 }
]

def eventLeaf19238 : Array AnnotatedEvent := #[
  { event := event307808
    frameStart := 307781 },
  { event := event307809
    frameStart := 307781 },
  { event := event307810
    frameStart := 307781 },
  { event := event307811
    frameStart := 307781 },
  { event := event307812
    frameStart := 307781 },
  { event := event307813
    frameStart := 307781 },
  { event := event307814
    frameStart := 307781 },
  { event := event307815
    frameStart := 307781 },
  { event := event307816
    frameStart := 307781 },
  { event := event307817
    frameStart := 307781 },
  { event := event307818
    frameStart := 307781 },
  { event := event307819
    frameStart := 307781 },
  { event := event307820
    frameStart := 307781 },
  { event := event307821
    frameStart := 307781 },
  { event := event307822
    frameStart := 307781 },
  { event := event307823
    frameStart := 307781 }
]

def eventLeaf19239 : Array AnnotatedEvent := #[
  { event := event307824
    frameStart := 307781 },
  { event := event307825
    frameStart := 307781 },
  { event := event307826
    frameStart := 307781 },
  { event := event307827
    frameStart := 307781 },
  { event := event307828
    frameStart := 307781 },
  { event := event307829
    frameStart := 307781 },
  { event := event307830
    frameStart := 307781 },
  { event := event307831
    frameStart := 307781 },
  { event := event307832
    frameStart := 307781 },
  { event := event307833
    frameStart := 307781 },
  { event := event307834
    frameStart := 307781 },
  { event := event307835
    frameStart := 307781 },
  { event := event307836
    frameStart := 307781 },
  { event := event307837
    frameStart := 307781 },
  { event := event307838
    frameStart := 307781 },
  { event := event307839
    frameStart := 307781 }
]

def eventLeaf19240 : Array AnnotatedEvent := #[
  { event := event307840
    frameStart := 307781 },
  { event := event307841
    frameStart := 307781 },
  { event := event307842
    frameStart := 307781 },
  { event := event307843
    frameStart := 307781 },
  { event := event307844
    frameStart := 307781 },
  { event := event307845
    frameStart := 307781 },
  { event := event307846
    frameStart := 307781 },
  { event := event307847
    frameStart := 307781 },
  { event := event307848
    frameStart := 307781 },
  { event := event307849
    frameStart := 307781 },
  { event := event307850
    frameStart := 307781 },
  { event := event307851
    frameStart := 307781 },
  { event := event307852
    frameStart := 307781 },
  { event := event307853
    frameStart := 307781 },
  { event := event307854
    frameStart := 307781 },
  { event := event307855
    frameStart := 307781 }
]

def eventLeaf19241 : Array AnnotatedEvent := #[
  { event := event307856
    frameStart := 307781 },
  { event := event307857
    frameStart := 307781 },
  { event := event307858
    frameStart := 307781 },
  { event := event307859
    frameStart := 307781 },
  { event := event307860
    frameStart := 307781 },
  { event := event307861
    frameStart := 307781 },
  { event := event307862
    frameStart := 307781 },
  { event := event307863
    frameStart := 307781 },
  { event := event307864
    frameStart := 307781 },
  { event := event307865
    frameStart := 307781 },
  { event := event307866
    frameStart := 307781 },
  { event := event307867
    frameStart := 307781 },
  { event := event307868
    frameStart := 307781 },
  { event := event307869
    frameStart := 307781 },
  { event := event307870
    frameStart := 307781 },
  { event := event307871
    frameStart := 307781 }
]

def eventLeaf19242 : Array AnnotatedEvent := #[
  { event := event307872
    frameStart := 307781 },
  { event := event307873
    frameStart := 0 },
  { event := event307874
    frameStart := 0 },
  { event := event307875
    frameStart := 0 },
  { event := event307876
    frameStart := 0 },
  { event := event307877
    frameStart := 0 },
  { event := event307878
    frameStart := 0 },
  { event := event307879
    frameStart := 0 },
  { event := event307880
    frameStart := 0 },
  { event := event307881
    frameStart := 0 },
  { event := event307882
    frameStart := 0 },
  { event := event307883
    frameStart := 0 },
  { event := event307884
    frameStart := 0 },
  { event := event307885
    frameStart := 0 },
  { event := event307886
    frameStart := 0 },
  { event := event307887
    frameStart := 0 }
]

def eventLeaf19243 : Array AnnotatedEvent := #[
  { event := event307888
    frameStart := 0 },
  { event := event307889
    frameStart := 0 },
  { event := event307890
    frameStart := 0 },
  { event := event307891
    frameStart := 0 },
  { event := event307892
    frameStart := 0 },
  { event := event307893
    frameStart := 0 },
  { event := event307894
    frameStart := 0 },
  { event := event307895
    frameStart := 0 },
  { event := event307896
    frameStart := 0 },
  { event := event307897
    frameStart := 0 },
  { event := event307898
    frameStart := 0 },
  { event := event307899
    frameStart := 0 },
  { event := event307900
    frameStart := 0 },
  { event := event307901
    frameStart := 0 },
  { event := event307902
    frameStart := 0 },
  { event := event307903
    frameStart := 0 }
]

def eventLeaf19244 : Array AnnotatedEvent := #[
  { event := event307904
    frameStart := 0 },
  { event := event307905
    frameStart := 0 },
  { event := event307906
    frameStart := 0 },
  { event := event307907
    frameStart := 0 },
  { event := event307908
    frameStart := 0 },
  { event := event307909
    frameStart := 0 },
  { event := event307910
    frameStart := 0 },
  { event := event307911
    frameStart := 0 },
  { event := event307912
    frameStart := 0 },
  { event := event307913
    frameStart := 0 },
  { event := event307914
    frameStart := 0 },
  { event := event307915
    frameStart := 0 },
  { event := event307916
    frameStart := 0 },
  { event := event307917
    frameStart := 0 },
  { event := event307918
    frameStart := 0 },
  { event := event307919
    frameStart := 0 }
]

def eventLeaf19245 : Array AnnotatedEvent := #[
  { event := event307920
    frameStart := 0 },
  { event := event307921
    frameStart := 0 },
  { event := event307922
    frameStart := 0 },
  { event := event307923
    frameStart := 0 },
  { event := event307924
    frameStart := 0 },
  { event := event307925
    frameStart := 0 },
  { event := event307926
    frameStart := 0 },
  { event := event307927
    frameStart := 0 },
  { event := event307928
    frameStart := 0 },
  { event := event307929
    frameStart := 0 },
  { event := event307930
    frameStart := 0 },
  { event := event307931
    frameStart := 0 },
  { event := event307932
    frameStart := 0 },
  { event := event307933
    frameStart := 0 },
  { event := event307934
    frameStart := 0 },
  { event := event307935
    frameStart := 0 }
]

def eventLeaf19246 : Array AnnotatedEvent := #[
  { event := event307936
    frameStart := 0 },
  { event := event307937
    frameStart := 0 },
  { event := event307938
    frameStart := 0 },
  { event := event307939
    frameStart := 0 },
  { event := event307940
    frameStart := 0 },
  { event := event307941
    frameStart := 0 },
  { event := event307942
    frameStart := 0 },
  { event := event307943
    frameStart := 0 },
  { event := event307944
    frameStart := 0 },
  { event := event307945
    frameStart := 0 },
  { event := event307946
    frameStart := 0 },
  { event := event307947
    frameStart := 0 },
  { event := event307948
    frameStart := 0 },
  { event := event307949
    frameStart := 0 },
  { event := event307950
    frameStart := 0 },
  { event := event307951
    frameStart := 0 }
]

def eventLeaf19247 : Array AnnotatedEvent := #[
  { event := event307952
    frameStart := 0 },
  { event := event307953
    frameStart := 0 },
  { event := event307954
    frameStart := 0 },
  { event := event307955
    frameStart := 0 },
  { event := event307956
    frameStart := 0 },
  { event := event307957
    frameStart := 0 },
  { event := event307958
    frameStart := 0 },
  { event := event307959
    frameStart := 0 },
  { event := event307960
    frameStart := 0 },
  { event := event307961
    frameStart := 0 },
  { event := event307962
    frameStart := 0 },
  { event := event307963
    frameStart := 0 },
  { event := event307964
    frameStart := 0 },
  { event := event307965
    frameStart := 0 },
  { event := event307966
    frameStart := 0 },
  { event := event307967
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1202
