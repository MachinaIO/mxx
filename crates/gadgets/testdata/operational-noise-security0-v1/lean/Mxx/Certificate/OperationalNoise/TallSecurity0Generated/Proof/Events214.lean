import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events214

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact54784RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact54784RawTermsValid :
    exact54784RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54784 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16185⟩⟩) exact54784RawTerms .large 54783 .exactZero (none)

def event54785 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26229⟩⟩) 0 ⟨16185⟩ 54784

def event54786 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26229⟩⟩) 1 ⟨26228⟩ 54769

def event54787 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26229⟩⟩) (.sum [.predecessor 0 54785 .coefficient, .predecessor 1 54786 .coefficient])

def exact54788RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨23670⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact54788RawTermsValid :
    exact54788RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54788 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26229⟩⟩) exact54788RawTerms .large 54787 .exactZero (none)

def event54789 : Event := .preFoldPolynomial 54788 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨23670⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact54790RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨23670⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event54790 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26229⟩⟩) 54789 exact54790RawTerms .large 54787 .exactZero (none)

def event54791 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14652⟩⟩) ⟨⟨112⟩, ⟨17⟩, ⟨109⟩⟩ ⟨54625, 54791⟩

def event54792 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19679⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19676⟩⟩]⟩) (1) 0 2 (.universal 54791 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19676⟩⟩]⟩) (none) 54790)

def event54793 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19679⟩⟩, .relation 54792 0, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩)

def event54794 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19679⟩⟩, .relation 54792 1, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩]⟩, (-1)⟩)

def event54795 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19679⟩⟩, .relation 54792 2, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨23670⟩⟩]⟩, (1)⟩)

def event54796 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19679⟩⟩, .relation 54792 3, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact54797RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨23670⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact54797RawTermsValid :
    exact54797RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54797 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19679⟩⟩) exact54797RawTerms .large 54621 (.finite 1811303510016) (some (54623))

def event54798 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26227⟩⟩) 0 ⟨19679⟩ 54797

def event54799 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26227⟩⟩) 1 ⟨26226⟩ 54611

def event54800 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26227⟩⟩) (.sum [.predecessor 0 54798 .coefficient, .predecessor 1 54799 .coefficient])

def event54801 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26227⟩⟩, .operator (⟨54797, 2⟩, ⟨54611, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], [⟨.program ⟨214⟩, ⟨23670⟩⟩]⟩, (-1)⟩)

def event54802 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26227⟩⟩, .operator (⟨54797, 1⟩, ⟨54611, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26225⟩⟩]⟩, (1)⟩)

def event54803 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26227⟩⟩) (.sum [.result 54797 .summary, .result 54611 .summary])

def exact54804RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact54804RawTermsValid :
    exact54804RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54804 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26227⟩⟩) exact54804RawTerms .large 54800 (.finite 352091253649408) (some (54803))

def event54805 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28315⟩⟩) 0 ⟨26227⟩ 54804

def event54806 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28315⟩⟩) 1 ⟨28313⟩ 54527

def event54807 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28315⟩⟩) (.product (.predecessor 0 54805 .coefficient) (.predecessor 1 54806 .coefficient) (⟨false, false, none, none, none⟩))

def event54808 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28315⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩) [⟨.result 54527 .coefficient, false, none⟩])

def event54809 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28315⟩⟩) (.product (.result 54804 .summary) (.transfer 54808) (⟨false, false, none, none, none⟩))

def event54810 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28315⟩⟩, .operator (⟨54804, 0⟩, ⟨54527, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩, (1)⟩)

def event54811 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28315⟩⟩, .operator (⟨54804, 1⟩, ⟨54527, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩, (-1)⟩)

def event54812 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28315⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28313⟩⟩) ⟨24291⟩ 54524)

def event54813 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28315⟩⟩, .relation 54812 0, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨24291⟩⟩]⟩, (-1)⟩)

def exact54814RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨24291⟩⟩]⟩, (-1)⟩]

theorem exact54814RawTermsValid :
    exact54814RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54814 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28315⟩⟩) exact54814RawTerms .large 54807 (.finite 1292180534353385750528) (some (54809))

def event54815 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21692⟩⟩) 0 ⟨16183⟩ 2539

def event54816 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21692⟩⟩) (.authority (.relationPreimageSource ⟨48⟩))

def exact54817RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21692⟩⟩]⟩, (1)⟩]

theorem exact54817RawTermsValid :
    exact54817RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54817 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21692⟩⟩) exact54817RawTerms (.finite 136065468) 54816 .exactZero (none)

def event54818 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21694⟩⟩) 0 ⟨21692⟩ 54817

def event54819 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21694⟩⟩) 1 ⟨2348⟩ 4

def event54820 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21694⟩⟩) (.scale (.predecessor 0 54818 .coefficient) (.value (.predecessor 1 54819 .coefficient)))

def exact54821RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21692⟩⟩]⟩, (1)⟩]

theorem exact54821RawTermsValid :
    exact54821RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54821 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21694⟩⟩) exact54821RawTerms (.finite 136065468) 54820 .exactZero (none)

def event54822 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21695⟩⟩) 0 ⟨5547⟩ 50762

def event54823 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21695⟩⟩) 1 ⟨21694⟩ 54821

def event54824 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21695⟩⟩) (.product (.predecessor 0 54822 .coefficient) (.predecessor 1 54823 .coefficient) (⟨false, false, none, none, none⟩))

def event54825 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21695⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨21692⟩⟩]⟩) [⟨.result 54817 .coefficient, false, none⟩])

def event54826 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21695⟩⟩) (.product (.result 50762 .summary) (.transfer 54825) (⟨false, false, none, none, none⟩))

def event54827 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21695⟩⟩, .operator (⟨50762, 0⟩, ⟨54821, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩]⟩, (1)⟩)

def event54828 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨21693⟩⟩)

def event54829 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event54830 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event54831 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.authority (.operator))

def event54832 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.finite 3)

def event54833 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event54834 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event54835 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event54836 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event54837 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 54836

def event54838 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 54834

def event54839 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 54837 .coefficient) (.value (.predecessor 1 54838 .coefficient)))

def event54840 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event54841 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 0 ⟨5503⟩ 54840

def event54842 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 1 ⟨2875⟩ 54832

def event54843 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.sum [.predecessor 0 54841 .coefficient, .predecessor 1 54842 .coefficient])

def event54844 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.finite 220)

def event54845 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 0 ⟨5510⟩ 54844

def event54846 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 1 ⟨961⟩ 54830

def event54847 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.identity (.predecessor 1 54846 .coefficient))

def event54848 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.finite 224)

def event54849 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11641⟩⟩) 0 ⟨5542⟩ 54848

def event54850 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11641⟩⟩) (.authority (.programFamilyFact))

def exact54851RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11641⟩⟩], []⟩, (1)⟩]

theorem exact54851RawTermsValid :
    exact54851RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54851 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11641⟩⟩) exact54851RawTerms (.finite 28) 54850 .exactZero (none)

def event54852 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14650⟩⟩) 0 ⟨5542⟩ 54848

def event54853 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14650⟩⟩) (.authority (.programFamilyFact))

def exact54854RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14650⟩⟩], []⟩, (1)⟩]

theorem exact54854RawTermsValid :
    exact54854RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54854 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14650⟩⟩) exact54854RawTerms (.finite 28) 54853 .exactZero (none)

def event54855 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14651⟩⟩) 0 ⟨14650⟩ 54854

def event54856 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14651⟩⟩) 1 ⟨11641⟩ 54851

def event54857 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14651⟩⟩) (.product (.predecessor 0 54855 .coefficient) (.predecessor 1 54856 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event54858 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14651⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], []⟩) [⟨.result 54854 .coefficient, true, some 1⟩, ⟨.result 54851 .coefficient, true, some 1⟩])

def event54859 : Event := .survivorFold (1) 54858

def exact54860RawTerms : List Term := []

theorem exact54860RawTermsValid :
    exact54860RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54860 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14651⟩⟩) exact54860RawTerms (.finite 784) 54857 (.finite 784) (some (54858))

def event54861 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14652⟩⟩) 0 ⟨14651⟩ 54860

def event54862 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14652⟩⟩) (.identity (.predecessor 0 54861 .coefficient))

def event54863 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14652⟩⟩) (.finite 784)

def event54864 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16182⟩⟩) 0 ⟨14652⟩ 54863

def event54865 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16182⟩⟩) (.authority (.programFamilyFact))

def exact54866RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], []⟩, (1)⟩]

theorem exact54866RawTermsValid :
    exact54866RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54866 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16182⟩⟩) exact54866RawTerms (.finite 28) 54865 .exactZero (none)

def event54867 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16183⟩⟩) 0 ⟨16182⟩ 54866

def event54868 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16183⟩⟩) (.identity (.predecessor 0 54867 .coefficient))

def event54869 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16183⟩⟩) (.finite 28)

def event54870 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21692⟩⟩) 0 ⟨16183⟩ 54869

def event54871 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21692⟩⟩) (.authority (.relationPreimageSource ⟨48⟩))

def exact54872RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨21692⟩⟩]⟩, (1)⟩]

theorem exact54872RawTermsValid :
    exact54872RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54872 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21692⟩⟩) exact54872RawTerms (.finite 136065468) 54871 .exactZero (none)

def event54873 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact54874RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact54874RawTermsValid :
    exact54874RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54874 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact54874RawTerms .large 54873 .exactZero (none)

def event54875 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21693⟩⟩) 0 ⟨6⟩ 54874

def event54876 : Event := .predecessor (⟨.program ⟨214⟩, ⟨21693⟩⟩) 1 ⟨21692⟩ 54872

def event54877 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨21693⟩⟩) (.product (.predecessor 0 54875 .coefficient) (.predecessor 1 54876 .coefficient) (⟨false, false, none, none, none⟩))

def event54878 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21693⟩⟩, .operator (⟨54874, 0⟩, ⟨54872, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩]⟩, (1)⟩)

def exact54879RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩]⟩, (1)⟩]

theorem exact54879RawTermsValid :
    exact54879RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54879 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21693⟩⟩) exact54879RawTerms .large 54877 .exactZero (none)

def event54880 : Event := .preFoldPolynomial 54879 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩]⟩, (1)⟩] .exactZero none

def exact54881RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩]⟩, (1)⟩]

def event54881 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨21693⟩⟩) 54880 exact54881RawTerms .large 54877 .exactZero (none)

def event54882 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28318⟩⟩)

def event54883 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event54884 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event54885 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.authority (.operator))

def event54886 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2875⟩⟩) (.finite 3)

def event54887 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event54888 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event54889 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event54890 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event54891 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 54890

def event54892 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 54888

def event54893 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 54891 .coefficient) (.value (.predecessor 1 54892 .coefficient)))

def event54894 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event54895 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 0 ⟨5503⟩ 54894

def event54896 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5510⟩⟩) 1 ⟨2875⟩ 54886

def event54897 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.sum [.predecessor 0 54895 .coefficient, .predecessor 1 54896 .coefficient])

def event54898 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5510⟩⟩) (.finite 220)

def event54899 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 0 ⟨5510⟩ 54898

def event54900 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5542⟩⟩) 1 ⟨961⟩ 54884

def event54901 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.identity (.predecessor 1 54900 .coefficient))

def event54902 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5542⟩⟩) (.finite 224)

def event54903 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11641⟩⟩) 0 ⟨5542⟩ 54902

def event54904 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11641⟩⟩) (.authority (.programFamilyFact))

def exact54905RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11641⟩⟩], []⟩, (1)⟩]

theorem exact54905RawTermsValid :
    exact54905RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54905 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11641⟩⟩) exact54905RawTerms (.finite 28) 54904 .exactZero (none)

def event54906 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14650⟩⟩) 0 ⟨5542⟩ 54902

def event54907 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14650⟩⟩) (.authority (.programFamilyFact))

def exact54908RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14650⟩⟩], []⟩, (1)⟩]

theorem exact54908RawTermsValid :
    exact54908RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54908 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14650⟩⟩) exact54908RawTerms (.finite 28) 54907 .exactZero (none)

def event54909 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14651⟩⟩) 0 ⟨14650⟩ 54908

def event54910 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14651⟩⟩) 1 ⟨11641⟩ 54905

def event54911 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14651⟩⟩) (.product (.predecessor 0 54909 .coefficient) (.predecessor 1 54910 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event54912 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨14651⟩⟩, .operator (⟨54908, 0⟩, ⟨54905, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], []⟩, (1)⟩)

def exact54913RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨11641⟩⟩, ⟨.program ⟨214⟩, ⟨14650⟩⟩], []⟩, (1)⟩]

theorem exact54913RawTermsValid :
    exact54913RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54913 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14651⟩⟩) exact54913RawTerms (.finite 784) 54911 .exactZero (none)

def event54914 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14652⟩⟩) 0 ⟨14651⟩ 54913

def event54915 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14652⟩⟩) (.identity (.predecessor 0 54914 .coefficient))

def event54916 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14652⟩⟩) (.finite 784)

def event54917 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16182⟩⟩) 0 ⟨14652⟩ 54916

def event54918 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16182⟩⟩) (.authority (.programFamilyFact))

def exact54919RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], []⟩, (1)⟩]

theorem exact54919RawTermsValid :
    exact54919RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54919 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16182⟩⟩) exact54919RawTerms (.finite 28) 54918 .exactZero (none)

def event54920 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16183⟩⟩) 0 ⟨16182⟩ 54919

def event54921 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16183⟩⟩) (.identity (.predecessor 0 54920 .coefficient))

def event54922 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16183⟩⟩) (.finite 28)

def event54923 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24289⟩⟩) 0 ⟨16183⟩ 54922

def event54924 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24289⟩⟩) (.authority (.programFamilyFact))

def event54925 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24289⟩⟩) (.finite 3720)

def event54926 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event54927 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24291⟩⟩) 0 ⟨6689⟩ 54926

def event54928 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24291⟩⟩) 1 ⟨24289⟩ 54925

def event54929 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24291⟩⟩) (.authority (.operator))

def exact54930RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24291⟩⟩]⟩, (1)⟩]

theorem exact54930RawTermsValid :
    exact54930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54930 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24291⟩⟩) exact54930RawTerms .large 54929 .exactZero (none)

def event54931 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28313⟩⟩) 0 ⟨24291⟩ 54930

def event54932 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28313⟩⟩) (.authority (.operator))

def exact54933RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩, (1)⟩]

theorem exact54933RawTermsValid :
    exact54933RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54933 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28313⟩⟩) exact54933RawTerms (.finite 8192) 54932 .exactZero (none)

def event54934 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event54935 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event54936 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16222⟩⟩) 0 ⟨16183⟩ 54922

def event54937 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16222⟩⟩) 1 ⟨110⟩ 54935

def event54938 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16222⟩⟩) (.sum [.predecessor 0 54936 .coefficient, .predecessor 1 54937 .coefficient])

def event54939 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16222⟩⟩) (.finite 28)

def event54940 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16223⟩⟩) 0 ⟨16222⟩ 54939

def event54941 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16223⟩⟩) (.identity (.predecessor 0 54940 .coefficient))

def exact54942RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], []⟩, (1)⟩]

theorem exact54942RawTermsValid :
    exact54942RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54942 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16223⟩⟩) exact54942RawTerms (.finite 28) 54941 .exactZero (none)

def event54943 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact54944RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact54944RawTermsValid :
    exact54944RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54944 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact54944RawTerms .large 54943 .exactZero (none)

def event54945 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16224⟩⟩) 0 ⟨6544⟩ 54944

def event54946 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16224⟩⟩) 1 ⟨16223⟩ 54942

def event54947 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16224⟩⟩) (.product (.predecessor 0 54945 .coefficient) (.predecessor 1 54946 .coefficient) (⟨false, false, none, none, none⟩))

def event54948 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16224⟩⟩, .operator (⟨54944, 0⟩, ⟨54942, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact54949RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact54949RawTermsValid :
    exact54949RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54949 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16224⟩⟩) exact54949RawTerms .large 54947 .exactZero (none)

def event54950 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6699⟩⟩) 0 ⟨6689⟩ 54926

def event54951 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6699⟩⟩) (.authority (.operator))

def exact54952RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩]

theorem exact54952RawTermsValid :
    exact54952RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54952 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6699⟩⟩) exact54952RawTerms .large 54951 .exactZero (none)

def event54953 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16225⟩⟩) 0 ⟨6699⟩ 54952

def event54954 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16225⟩⟩) 1 ⟨16224⟩ 54949

def event54955 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16225⟩⟩) (.sum [.predecessor 0 54953 .coefficient, .predecessor 1 54954 .coefficient])

def exact54956RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact54956RawTermsValid :
    exact54956RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54956 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16225⟩⟩) exact54956RawTerms .large 54955 .exactZero (none)

def event54957 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28314⟩⟩) 0 ⟨16225⟩ 54956

def event54958 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28314⟩⟩) 1 ⟨28313⟩ 54933

def event54959 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28314⟩⟩) (.product (.predecessor 0 54957 .coefficient) (.predecessor 1 54958 .coefficient) (⟨false, false, none, none, none⟩))

def event54960 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28314⟩⟩, .operator (⟨54956, 0⟩, ⟨54933, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩, (1)⟩)

def event54961 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28314⟩⟩, .operator (⟨54956, 1⟩, ⟨54933, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩, (-1)⟩)

def event54962 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28314⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28313⟩⟩) ⟨24291⟩ 54930)

def event54963 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28314⟩⟩, .relation 54962 0, ⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨24291⟩⟩]⟩, (-1)⟩)

def exact54964RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨24291⟩⟩]⟩, (-1)⟩]

theorem exact54964RawTermsValid :
    exact54964RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54964 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28314⟩⟩) exact54964RawTerms .large 54959 .exactZero (none)

def event54965 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18353⟩⟩) 0 ⟨16183⟩ 54922

def event54966 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18353⟩⟩) (.authority (.programFamilyFact))

def exact54967RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], []⟩, (1)⟩]

theorem exact54967RawTermsValid :
    exact54967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54967 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18353⟩⟩) exact54967RawTerms (.finite 62) 54966 .exactZero (none)

def event54968 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18364⟩⟩) 0 ⟨6544⟩ 54944

def event54969 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18364⟩⟩) 1 ⟨18353⟩ 54967

def event54970 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18364⟩⟩) (.product (.predecessor 0 54968 .coefficient) (.predecessor 1 54969 .coefficient) (⟨false, true, none, none, some 1⟩))

def event54971 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨18364⟩⟩, .operator (⟨54944, 0⟩, ⟨54967, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact54972RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact54972RawTermsValid :
    exact54972RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54972 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18364⟩⟩) exact54972RawTerms .large 54970 .exactZero (none)

def event54973 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6727⟩⟩) 0 ⟨6689⟩ 54926

def event54974 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6727⟩⟩) (.authority (.operator))

def exact54975RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩]

theorem exact54975RawTermsValid :
    exact54975RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54975 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6727⟩⟩) exact54975RawTerms .large 54974 .exactZero (none)

def event54976 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18365⟩⟩) 0 ⟨6727⟩ 54975

def event54977 : Event := .predecessor (⟨.program ⟨214⟩, ⟨18365⟩⟩) 1 ⟨18364⟩ 54972

def event54978 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨18365⟩⟩) (.sum [.predecessor 0 54976 .coefficient, .predecessor 1 54977 .coefficient])

def exact54979RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact54979RawTermsValid :
    exact54979RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54979 : Event := .resultExact (⟨.program ⟨214⟩, ⟨18365⟩⟩) exact54979RawTerms .large 54978 .exactZero (none)

def event54980 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28318⟩⟩) 0 ⟨18365⟩ 54979

def event54981 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28318⟩⟩) 1 ⟨28314⟩ 54964

def event54982 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28318⟩⟩) (.sum [.predecessor 0 54980 .coefficient, .predecessor 1 54981 .coefficient])

def exact54983RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨24291⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact54983RawTermsValid :
    exact54983RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54983 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28318⟩⟩) exact54983RawTerms .large 54982 .exactZero (none)

def event54984 : Event := .preFoldPolynomial 54983 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨24291⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact54985RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨24291⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event54985 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨28318⟩⟩) 54984 exact54985RawTerms .large 54982 .exactZero (none)

def event54986 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨16183⟩⟩) ⟨⟨140⟩, ⟨48⟩, ⟨109⟩⟩ ⟨54828, 54986⟩

def event54987 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨21695⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩]⟩) (1) 0 2 (.universal 54986 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩]⟩) (none) 54985)

def event54988 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21695⟩⟩, .relation 54987 1, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩)

def event54989 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21695⟩⟩, .relation 54987 0, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩, (-1)⟩)

def event54990 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21695⟩⟩, .relation 54987 2, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨24291⟩⟩]⟩, (1)⟩)

def event54991 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨21695⟩⟩, .relation 54987 3, ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18353⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact54992RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨24291⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18353⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact54992RawTermsValid :
    exact54992RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54992 : Event := .resultExact (⟨.program ⟨214⟩, ⟨21695⟩⟩) exact54992RawTerms .large 54824 (.finite 1811303510016) (some (54826))

def event54993 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28316⟩⟩) 0 ⟨21695⟩ 54992

def event54994 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28316⟩⟩) 1 ⟨28315⟩ 54814

def event54995 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28316⟩⟩) (.sum [.predecessor 0 54993 .coefficient, .predecessor 1 54994 .coefficient])

def event54996 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28316⟩⟩, .operator (⟨54992, 0⟩, ⟨54814, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩]⟩, (1)⟩)

def event54997 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28316⟩⟩, .operator (⟨54992, 2⟩, ⟨54814, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], [⟨.program ⟨214⟩, ⟨24291⟩⟩]⟩, (-1)⟩)

def event54998 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28316⟩⟩) (.sum [.result 54992 .summary, .result 54814 .summary])

def exact54999RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18353⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact54999RawTermsValid :
    exact54999RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event54999 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28316⟩⟩) exact54999RawTerms .large 54995 (.finite 1292180536164689260544) (some (54998))

def event55000 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24226⟩⟩) 0 ⟨16064⟩ 2562

def event55001 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24226⟩⟩) (.authority (.programFamilyFact))

def event55002 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨24226⟩⟩) (.finite 3720)

def event55003 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24228⟩⟩) 0 ⟨6689⟩ 5477

def event55004 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24228⟩⟩) 1 ⟨24226⟩ 55002

def event55005 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24228⟩⟩) (.authority (.operator))

def exact55006RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24228⟩⟩]⟩, (1)⟩]

theorem exact55006RawTermsValid :
    exact55006RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55006 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24228⟩⟩) exact55006RawTerms .large 55005 .exactZero (none)

def event55007 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28096⟩⟩) 0 ⟨24228⟩ 55006

def event55008 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28096⟩⟩) (.authority (.operator))

def exact55009RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨28096⟩⟩]⟩, (1)⟩]

theorem exact55009RawTermsValid :
    exact55009RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55009 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28096⟩⟩) exact55009RawTerms (.finite 8192) 55008 .exactZero (none)

def event55010 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23627⟩⟩) 0 ⟨14435⟩ 2556

def event55011 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23627⟩⟩) (.authority (.programFamilyFact))

def event55012 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23627⟩⟩) (.finite 3720)

def event55013 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23628⟩⟩) 0 ⟨6689⟩ 5477

def event55014 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23628⟩⟩) 1 ⟨23627⟩ 55012

def event55015 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23628⟩⟩) (.authority (.operator))

def exact55016RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23628⟩⟩]⟩, (1)⟩]

theorem exact55016RawTermsValid :
    exact55016RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55016 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23628⟩⟩) exact55016RawTerms .large 55015 .exactZero (none)

def event55017 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26148⟩⟩) 0 ⟨23628⟩ 55016

def event55018 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26148⟩⟩) (.authority (.operator))

def exact55019RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26148⟩⟩]⟩, (1)⟩]

theorem exact55019RawTermsValid :
    exact55019RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55019 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26148⟩⟩) exact55019RawTerms (.finite 8192) 55018 .exactZero (none)

def event55020 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11558⟩⟩) 0 ⟨11557⟩ 2545

def event55021 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11558⟩⟩) 1 ⟨6568⟩ 50670

def event55022 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11558⟩⟩) (.tensor (.predecessor 0 55020 .coefficient) (.predecessor 1 55021 .coefficient) true false)

def event55023 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨11558⟩⟩, .operator (⟨2545, 0⟩, ⟨50670, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact55024RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact55024RawTermsValid :
    exact55024RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55024 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11558⟩⟩) exact55024RawTerms .large 55022 .exactZero (none)

def event55025 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7274⟩⟩) 0 ⟨5545⟩ 50540

def event55026 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7274⟩⟩) 1 ⟨6780⟩ 10981

def event55027 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7274⟩⟩) (.product (.predecessor 0 55025 .coefficient) (.predecessor 1 55026 .coefficient) (⟨false, false, none, none, none⟩))

def event55028 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7274⟩⟩, .operator (⟨50540, 0⟩, ⟨10981, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩)

def exact55029RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩]

theorem exact55029RawTermsValid :
    exact55029RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55029 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7274⟩⟩) exact55029RawTerms .large 55027 .exactZero (none)

def event55030 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11559⟩⟩) 0 ⟨7274⟩ 55029

def event55031 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11559⟩⟩) 1 ⟨11558⟩ 55024

def event55032 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11559⟩⟩) (.sum [.predecessor 0 55030 .coefficient, .predecessor 1 55031 .coefficient])

def exact55033RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact55033RawTermsValid :
    exact55033RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55033 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11559⟩⟩) exact55033RawTerms .large 55032 .exactZero (none)

def event55034 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11560⟩⟩) 0 ⟨11559⟩ 55033

def event55035 : Event := .predecessor (⟨.program ⟨214⟩, ⟨11560⟩⟩) 1 ⟨94⟩ 10973

def event55036 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11560⟩⟩) (.sum [.predecessor 0 55034 .coefficient, .predecessor 1 55035 .coefficient])

def event55037 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨11560⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨94⟩⟩]⟩) [⟨.result 10973 .coefficient, false, none⟩])

def event55038 : Event := .survivorFold (1) 55037

def exact55039RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11557⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact55039RawTermsValid :
    exact55039RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event55039 : Event := .resultExact (⟨.program ⟨214⟩, ⟨11560⟩⟩) exact55039RawTerms .large 55036 (.finite 26) (some (55037))

def eventLeaf3424 : Array AnnotatedEvent := #[
  { event := event54784
    frameStart := 54673 },
  { event := event54785
    frameStart := 54673 },
  { event := event54786
    frameStart := 54673 },
  { event := event54787
    frameStart := 54673 },
  { event := event54788
    frameStart := 54673 },
  { event := event54789
    frameStart := 54673 },
  { event := event54790
    frameStart := 54673 },
  { event := event54791
    frameStart := 0 },
  { event := event54792
    frameStart := 0 },
  { event := event54793
    frameStart := 0 },
  { event := event54794
    frameStart := 0 },
  { event := event54795
    frameStart := 0 },
  { event := event54796
    frameStart := 0 },
  { event := event54797
    frameStart := 0 },
  { event := event54798
    frameStart := 0 },
  { event := event54799
    frameStart := 0 }
]

def eventLeaf3425 : Array AnnotatedEvent := #[
  { event := event54800
    frameStart := 0 },
  { event := event54801
    frameStart := 0 },
  { event := event54802
    frameStart := 0 },
  { event := event54803
    frameStart := 0 },
  { event := event54804
    frameStart := 0 },
  { event := event54805
    frameStart := 0 },
  { event := event54806
    frameStart := 0 },
  { event := event54807
    frameStart := 0 },
  { event := event54808
    frameStart := 0 },
  { event := event54809
    frameStart := 0 },
  { event := event54810
    frameStart := 0 },
  { event := event54811
    frameStart := 0 },
  { event := event54812
    frameStart := 0 },
  { event := event54813
    frameStart := 0 },
  { event := event54814
    frameStart := 0 },
  { event := event54815
    frameStart := 0 }
]

def eventLeaf3426 : Array AnnotatedEvent := #[
  { event := event54816
    frameStart := 0 },
  { event := event54817
    frameStart := 0 },
  { event := event54818
    frameStart := 0 },
  { event := event54819
    frameStart := 0 },
  { event := event54820
    frameStart := 0 },
  { event := event54821
    frameStart := 0 },
  { event := event54822
    frameStart := 0 },
  { event := event54823
    frameStart := 0 },
  { event := event54824
    frameStart := 0 },
  { event := event54825
    frameStart := 0 },
  { event := event54826
    frameStart := 0 },
  { event := event54827
    frameStart := 0 },
  { event := event54828
    frameStart := 54828 },
  { event := event54829
    frameStart := 54828 },
  { event := event54830
    frameStart := 54828 },
  { event := event54831
    frameStart := 54828 }
]

def eventLeaf3427 : Array AnnotatedEvent := #[
  { event := event54832
    frameStart := 54828 },
  { event := event54833
    frameStart := 54828 },
  { event := event54834
    frameStart := 54828 },
  { event := event54835
    frameStart := 54828 },
  { event := event54836
    frameStart := 54828 },
  { event := event54837
    frameStart := 54828 },
  { event := event54838
    frameStart := 54828 },
  { event := event54839
    frameStart := 54828 },
  { event := event54840
    frameStart := 54828 },
  { event := event54841
    frameStart := 54828 },
  { event := event54842
    frameStart := 54828 },
  { event := event54843
    frameStart := 54828 },
  { event := event54844
    frameStart := 54828 },
  { event := event54845
    frameStart := 54828 },
  { event := event54846
    frameStart := 54828 },
  { event := event54847
    frameStart := 54828 }
]

def eventLeaf3428 : Array AnnotatedEvent := #[
  { event := event54848
    frameStart := 54828 },
  { event := event54849
    frameStart := 54828 },
  { event := event54850
    frameStart := 54828 },
  { event := event54851
    frameStart := 54828 },
  { event := event54852
    frameStart := 54828 },
  { event := event54853
    frameStart := 54828 },
  { event := event54854
    frameStart := 54828 },
  { event := event54855
    frameStart := 54828 },
  { event := event54856
    frameStart := 54828 },
  { event := event54857
    frameStart := 54828 },
  { event := event54858
    frameStart := 54828 },
  { event := event54859
    frameStart := 54828 },
  { event := event54860
    frameStart := 54828 },
  { event := event54861
    frameStart := 54828 },
  { event := event54862
    frameStart := 54828 },
  { event := event54863
    frameStart := 54828 }
]

def eventLeaf3429 : Array AnnotatedEvent := #[
  { event := event54864
    frameStart := 54828 },
  { event := event54865
    frameStart := 54828 },
  { event := event54866
    frameStart := 54828 },
  { event := event54867
    frameStart := 54828 },
  { event := event54868
    frameStart := 54828 },
  { event := event54869
    frameStart := 54828 },
  { event := event54870
    frameStart := 54828 },
  { event := event54871
    frameStart := 54828 },
  { event := event54872
    frameStart := 54828 },
  { event := event54873
    frameStart := 54828 },
  { event := event54874
    frameStart := 54828 },
  { event := event54875
    frameStart := 54828 },
  { event := event54876
    frameStart := 54828 },
  { event := event54877
    frameStart := 54828 },
  { event := event54878
    frameStart := 54828 },
  { event := event54879
    frameStart := 54828 }
]

def eventLeaf3430 : Array AnnotatedEvent := #[
  { event := event54880
    frameStart := 54828 },
  { event := event54881
    frameStart := 54828 },
  { event := event54882
    frameStart := 54882 },
  { event := event54883
    frameStart := 54882 },
  { event := event54884
    frameStart := 54882 },
  { event := event54885
    frameStart := 54882 },
  { event := event54886
    frameStart := 54882 },
  { event := event54887
    frameStart := 54882 },
  { event := event54888
    frameStart := 54882 },
  { event := event54889
    frameStart := 54882 },
  { event := event54890
    frameStart := 54882 },
  { event := event54891
    frameStart := 54882 },
  { event := event54892
    frameStart := 54882 },
  { event := event54893
    frameStart := 54882 },
  { event := event54894
    frameStart := 54882 },
  { event := event54895
    frameStart := 54882 }
]

def eventLeaf3431 : Array AnnotatedEvent := #[
  { event := event54896
    frameStart := 54882 },
  { event := event54897
    frameStart := 54882 },
  { event := event54898
    frameStart := 54882 },
  { event := event54899
    frameStart := 54882 },
  { event := event54900
    frameStart := 54882 },
  { event := event54901
    frameStart := 54882 },
  { event := event54902
    frameStart := 54882 },
  { event := event54903
    frameStart := 54882 },
  { event := event54904
    frameStart := 54882 },
  { event := event54905
    frameStart := 54882 },
  { event := event54906
    frameStart := 54882 },
  { event := event54907
    frameStart := 54882 },
  { event := event54908
    frameStart := 54882 },
  { event := event54909
    frameStart := 54882 },
  { event := event54910
    frameStart := 54882 },
  { event := event54911
    frameStart := 54882 }
]

def eventLeaf3432 : Array AnnotatedEvent := #[
  { event := event54912
    frameStart := 54882 },
  { event := event54913
    frameStart := 54882 },
  { event := event54914
    frameStart := 54882 },
  { event := event54915
    frameStart := 54882 },
  { event := event54916
    frameStart := 54882 },
  { event := event54917
    frameStart := 54882 },
  { event := event54918
    frameStart := 54882 },
  { event := event54919
    frameStart := 54882 },
  { event := event54920
    frameStart := 54882 },
  { event := event54921
    frameStart := 54882 },
  { event := event54922
    frameStart := 54882 },
  { event := event54923
    frameStart := 54882 },
  { event := event54924
    frameStart := 54882 },
  { event := event54925
    frameStart := 54882 },
  { event := event54926
    frameStart := 54882 },
  { event := event54927
    frameStart := 54882 }
]

def eventLeaf3433 : Array AnnotatedEvent := #[
  { event := event54928
    frameStart := 54882 },
  { event := event54929
    frameStart := 54882 },
  { event := event54930
    frameStart := 54882 },
  { event := event54931
    frameStart := 54882 },
  { event := event54932
    frameStart := 54882 },
  { event := event54933
    frameStart := 54882 },
  { event := event54934
    frameStart := 54882 },
  { event := event54935
    frameStart := 54882 },
  { event := event54936
    frameStart := 54882 },
  { event := event54937
    frameStart := 54882 },
  { event := event54938
    frameStart := 54882 },
  { event := event54939
    frameStart := 54882 },
  { event := event54940
    frameStart := 54882 },
  { event := event54941
    frameStart := 54882 },
  { event := event54942
    frameStart := 54882 },
  { event := event54943
    frameStart := 54882 }
]

def eventLeaf3434 : Array AnnotatedEvent := #[
  { event := event54944
    frameStart := 54882 },
  { event := event54945
    frameStart := 54882 },
  { event := event54946
    frameStart := 54882 },
  { event := event54947
    frameStart := 54882 },
  { event := event54948
    frameStart := 54882 },
  { event := event54949
    frameStart := 54882 },
  { event := event54950
    frameStart := 54882 },
  { event := event54951
    frameStart := 54882 },
  { event := event54952
    frameStart := 54882 },
  { event := event54953
    frameStart := 54882 },
  { event := event54954
    frameStart := 54882 },
  { event := event54955
    frameStart := 54882 },
  { event := event54956
    frameStart := 54882 },
  { event := event54957
    frameStart := 54882 },
  { event := event54958
    frameStart := 54882 },
  { event := event54959
    frameStart := 54882 }
]

def eventLeaf3435 : Array AnnotatedEvent := #[
  { event := event54960
    frameStart := 54882 },
  { event := event54961
    frameStart := 54882 },
  { event := event54962
    frameStart := 54882 },
  { event := event54963
    frameStart := 54882 },
  { event := event54964
    frameStart := 54882 },
  { event := event54965
    frameStart := 54882 },
  { event := event54966
    frameStart := 54882 },
  { event := event54967
    frameStart := 54882 },
  { event := event54968
    frameStart := 54882 },
  { event := event54969
    frameStart := 54882 },
  { event := event54970
    frameStart := 54882 },
  { event := event54971
    frameStart := 54882 },
  { event := event54972
    frameStart := 54882 },
  { event := event54973
    frameStart := 54882 },
  { event := event54974
    frameStart := 54882 },
  { event := event54975
    frameStart := 54882 }
]

def eventLeaf3436 : Array AnnotatedEvent := #[
  { event := event54976
    frameStart := 54882 },
  { event := event54977
    frameStart := 54882 },
  { event := event54978
    frameStart := 54882 },
  { event := event54979
    frameStart := 54882 },
  { event := event54980
    frameStart := 54882 },
  { event := event54981
    frameStart := 54882 },
  { event := event54982
    frameStart := 54882 },
  { event := event54983
    frameStart := 54882 },
  { event := event54984
    frameStart := 54882 },
  { event := event54985
    frameStart := 54882 },
  { event := event54986
    frameStart := 0 },
  { event := event54987
    frameStart := 0 },
  { event := event54988
    frameStart := 0 },
  { event := event54989
    frameStart := 0 },
  { event := event54990
    frameStart := 0 },
  { event := event54991
    frameStart := 0 }
]

def eventLeaf3437 : Array AnnotatedEvent := #[
  { event := event54992
    frameStart := 0 },
  { event := event54993
    frameStart := 0 },
  { event := event54994
    frameStart := 0 },
  { event := event54995
    frameStart := 0 },
  { event := event54996
    frameStart := 0 },
  { event := event54997
    frameStart := 0 },
  { event := event54998
    frameStart := 0 },
  { event := event54999
    frameStart := 0 },
  { event := event55000
    frameStart := 0 },
  { event := event55001
    frameStart := 0 },
  { event := event55002
    frameStart := 0 },
  { event := event55003
    frameStart := 0 },
  { event := event55004
    frameStart := 0 },
  { event := event55005
    frameStart := 0 },
  { event := event55006
    frameStart := 0 },
  { event := event55007
    frameStart := 0 }
]

def eventLeaf3438 : Array AnnotatedEvent := #[
  { event := event55008
    frameStart := 0 },
  { event := event55009
    frameStart := 0 },
  { event := event55010
    frameStart := 0 },
  { event := event55011
    frameStart := 0 },
  { event := event55012
    frameStart := 0 },
  { event := event55013
    frameStart := 0 },
  { event := event55014
    frameStart := 0 },
  { event := event55015
    frameStart := 0 },
  { event := event55016
    frameStart := 0 },
  { event := event55017
    frameStart := 0 },
  { event := event55018
    frameStart := 0 },
  { event := event55019
    frameStart := 0 },
  { event := event55020
    frameStart := 0 },
  { event := event55021
    frameStart := 0 },
  { event := event55022
    frameStart := 0 },
  { event := event55023
    frameStart := 0 }
]

def eventLeaf3439 : Array AnnotatedEvent := #[
  { event := event55024
    frameStart := 0 },
  { event := event55025
    frameStart := 0 },
  { event := event55026
    frameStart := 0 },
  { event := event55027
    frameStart := 0 },
  { event := event55028
    frameStart := 0 },
  { event := event55029
    frameStart := 0 },
  { event := event55030
    frameStart := 0 },
  { event := event55031
    frameStart := 0 },
  { event := event55032
    frameStart := 0 },
  { event := event55033
    frameStart := 0 },
  { event := event55034
    frameStart := 0 },
  { event := event55035
    frameStart := 0 },
  { event := event55036
    frameStart := 0 },
  { event := event55037
    frameStart := 0 },
  { event := event55038
    frameStart := 0 },
  { event := event55039
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events214
