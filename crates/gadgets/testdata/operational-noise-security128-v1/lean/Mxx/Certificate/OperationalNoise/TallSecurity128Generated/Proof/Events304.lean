import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events304

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event77824 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38633⟩⟩) (.authority (.programFamilyFact))

def event77825 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38633⟩⟩) (.finite 3720)

def event77826 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38635⟩⟩) 0 ⟨7177⟩ 15500

def event77827 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38635⟩⟩) 1 ⟨38633⟩ 77825

def event77828 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38635⟩⟩) (.authority (.operator))

def exact77829RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38635⟩⟩]⟩, (1)⟩]

theorem exact77829RawTermsValid :
    exact77829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77829 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38635⟩⟩) exact77829RawTerms .large 77828 .exactZero (none)

def event77830 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39459⟩⟩) 0 ⟨38635⟩ 77829

def event77831 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39459⟩⟩) (.authority (.operator))

def exact77832RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩, (1)⟩]

theorem exact77832RawTermsValid :
    exact77832RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77832 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39459⟩⟩) exact77832RawTerms (.finite 8192) 77831 .exactZero (none)

def event77833 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38464⟩⟩) 0 ⟨37260⟩ 3189

def event77834 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38464⟩⟩) (.authority (.programFamilyFact))

def event77835 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38464⟩⟩) (.finite 3720)

def event77836 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38465⟩⟩) 0 ⟨7177⟩ 15500

def event77837 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38465⟩⟩) 1 ⟨38464⟩ 77835

def event77838 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38465⟩⟩) (.authority (.operator))

def exact77839RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38465⟩⟩]⟩, (1)⟩]

theorem exact77839RawTermsValid :
    exact77839RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77839 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38465⟩⟩) exact77839RawTerms .large 77838 .exactZero (none)

def event77840 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39005⟩⟩) 0 ⟨38465⟩ 77839

def event77841 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39005⟩⟩) (.authority (.operator))

def exact77842RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩, (1)⟩]

theorem exact77842RawTermsValid :
    exact77842RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77842 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39005⟩⟩) exact77842RawTerms (.finite 8192) 77841 .exactZero (none)

def event77843 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37261⟩⟩) 0 ⟨37258⟩ 3178

def event77844 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37261⟩⟩) 1 ⟨10328⟩ 75903

def event77845 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37261⟩⟩) (.tensor (.predecessor 0 77843 .coefficient) (.predecessor 1 77844 .coefficient) true false)

def event77846 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37261⟩⟩, .operator (⟨3178, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact77847RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact77847RawTermsValid :
    exact77847RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77847 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37261⟩⟩) exact77847RawTerms .large 77845 .exactZero (none)

def event77848 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10339⟩⟩) 0 ⟨10327⟩ 75773

def event77849 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10339⟩⟩) 1 ⟨7281⟩ 19084

def event77850 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10339⟩⟩) (.product (.predecessor 0 77848 .coefficient) (.predecessor 1 77849 .coefficient) (⟨false, false, none, none, none⟩))

def event77851 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10339⟩⟩, .operator (⟨75773, 0⟩, ⟨19084, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩)

def exact77852RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩]

theorem exact77852RawTermsValid :
    exact77852RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77852 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10339⟩⟩) exact77852RawTerms .large 77850 .exactZero (none)

def event77853 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37262⟩⟩) 0 ⟨10339⟩ 77852

def event77854 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37262⟩⟩) 1 ⟨37261⟩ 77847

def event77855 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37262⟩⟩) (.sum [.predecessor 0 77853 .coefficient, .predecessor 1 77854 .coefficient])

def exact77856RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77856RawTermsValid :
    exact77856RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77856 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37262⟩⟩) exact77856RawTerms .large 77855 .exactZero (none)

def event77857 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37263⟩⟩) 0 ⟨37262⟩ 77856

def event77858 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37263⟩⟩) 1 ⟨107⟩ 19076

def event77859 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37263⟩⟩) (.sum [.predecessor 0 77857 .coefficient, .predecessor 1 77858 .coefficient])

def event77860 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37263⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨107⟩⟩]⟩) [⟨.result 19076 .coefficient, false, none⟩])

def event77861 : Event := .survivorFold (1) 77860

def exact77862RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77862RawTermsValid :
    exact77862RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77862 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37263⟩⟩) exact77862RawTerms .large 77859 (.finite 26) (some (77860))

def event77863 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37264⟩⟩) 0 ⟨37263⟩ 77862

def event77864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37264⟩⟩) 1 ⟨13971⟩ 3181

def event77865 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37264⟩⟩) (.product (.predecessor 0 77863 .coefficient) (.predecessor 1 77864 .coefficient) (⟨false, true, none, none, some 1⟩))

def event77866 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37264⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩], []⟩) [⟨.result 3181 .coefficient, true, some 1⟩])

def event77867 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37264⟩⟩) (.product (.result 77862 .summary) (.transfer 77866) (⟨false, false, none, none, none⟩))

def event77868 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37264⟩⟩, .operator (⟨77862, 1⟩, ⟨3181, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event77869 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37264⟩⟩, .operator (⟨77862, 0⟩, ⟨3181, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩)

def exact77870RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77870RawTermsValid :
    exact77870RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77870 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37264⟩⟩) exact77870RawTerms .large 77865 (.finite 35782656) (some (77867))

def event77871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13972⟩⟩) 0 ⟨13971⟩ 3181

def event77872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13972⟩⟩) 1 ⟨10328⟩ 75903

def event77873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13972⟩⟩) (.tensor (.predecessor 0 77871 .coefficient) (.predecessor 1 77872 .coefficient) true false)

def event77874 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13972⟩⟩, .operator (⟨3181, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact77875RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact77875RawTermsValid :
    exact77875RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77875 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13972⟩⟩) exact77875RawTerms .large 77873 .exactZero (none)

def event77876 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10356⟩⟩) 0 ⟨10327⟩ 75773

def event77877 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10356⟩⟩) 1 ⟨7298⟩ 19125

def event77878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10356⟩⟩) (.product (.predecessor 0 77876 .coefficient) (.predecessor 1 77877 .coefficient) (⟨false, false, none, none, none⟩))

def event77879 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10356⟩⟩, .operator (⟨75773, 0⟩, ⟨19125, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩)

def exact77880RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩]

theorem exact77880RawTermsValid :
    exact77880RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77880 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10356⟩⟩) exact77880RawTerms .large 77878 .exactZero (none)

def event77881 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13973⟩⟩) 0 ⟨10356⟩ 77880

def event77882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13973⟩⟩) 1 ⟨13972⟩ 77875

def event77883 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13973⟩⟩) (.sum [.predecessor 0 77881 .coefficient, .predecessor 1 77882 .coefficient])

def exact77884RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77884RawTermsValid :
    exact77884RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77884 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13973⟩⟩) exact77884RawTerms .large 77883 .exactZero (none)

def event77885 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13974⟩⟩) 0 ⟨13973⟩ 77884

def event77886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13974⟩⟩) 1 ⟨124⟩ 19117

def event77887 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13974⟩⟩) (.sum [.predecessor 0 77885 .coefficient, .predecessor 1 77886 .coefficient])

def event77888 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13974⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨124⟩⟩]⟩) [⟨.result 19117 .coefficient, false, none⟩])

def event77889 : Event := .survivorFold (1) 77888

def exact77890RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77890RawTermsValid :
    exact77890RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77890 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13974⟩⟩) exact77890RawTerms .large 77887 (.finite 26) (some (77888))

def event77891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13975⟩⟩) 0 ⟨13974⟩ 77890

def event77892 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13975⟩⟩) 1 ⟨9554⟩ 19114

def event77893 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13975⟩⟩) (.product (.predecessor 0 77891 .coefficient) (.predecessor 1 77892 .coefficient) (⟨false, false, none, none, none⟩))

def event77894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13975⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) [⟨.result 19110 .coefficient, false, none⟩])

def event77895 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13975⟩⟩) (.product (.result 77890 .summary) (.transfer 77894) (⟨false, false, none, none, none⟩))

def event77896 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13975⟩⟩, .operator (⟨77890, 1⟩, ⟨19114, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (-1)⟩)

def event77897 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨13975⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9553⟩⟩) ⟨7281⟩ 19084)

def event77898 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13975⟩⟩, .relation 77897 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (-1)⟩)

def event77899 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13975⟩⟩, .operator (⟨77890, 0⟩, ⟨19114, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩)

def exact77900RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (-1)⟩]

theorem exact77900RawTermsValid :
    exact77900RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77900 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13975⟩⟩) exact77900RawTerms .large 77893 (.finite 279172874240) (some (77895))

def event77901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37265⟩⟩) 0 ⟨13975⟩ 77900

def event77902 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37265⟩⟩) 1 ⟨37264⟩ 77870

def event77903 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37265⟩⟩) (.sum [.predecessor 0 77901 .coefficient, .predecessor 1 77902 .coefficient])

def event77904 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37265⟩⟩, .operator (⟨77900, 1⟩, ⟨77870, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩)

def event77905 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37265⟩⟩) (.sum [.result 77900 .summary, .result 77870 .summary])

def exact77906RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77906RawTermsValid :
    exact77906RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77906 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37265⟩⟩) exact77906RawTerms .large 77903 (.finite 279208656896) (some (77905))

def event77907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39006⟩⟩) 0 ⟨37265⟩ 77906

def event77908 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39006⟩⟩) 1 ⟨39005⟩ 77842

def event77909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39006⟩⟩) (.product (.predecessor 0 77907 .coefficient) (.predecessor 1 77908 .coefficient) (⟨false, false, none, none, none⟩))

def event77910 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39006⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩) [⟨.result 77842 .coefficient, false, none⟩])

def event77911 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39006⟩⟩) (.product (.result 77906 .summary) (.transfer 77910) (⟨false, false, none, none, none⟩))

def event77912 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39006⟩⟩, .operator (⟨77906, 1⟩, ⟨77842, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩, (-1)⟩)

def event77913 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39006⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39005⟩⟩) ⟨38465⟩ 77839)

def event77914 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39006⟩⟩, .relation 77913 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨38465⟩⟩]⟩, (-1)⟩)

def event77915 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39006⟩⟩, .operator (⟨77906, 0⟩, ⟨77842, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩, (1)⟩)

def exact77916RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨38465⟩⟩]⟩, (-1)⟩]

theorem exact77916RawTermsValid :
    exact77916RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77916 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39006⟩⟩) exact77916RawTerms .large 77909 (.finite 2997980125321012183040) (some (77911))

def event77917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37929⟩⟩) 0 ⟨37260⟩ 3189

def event77918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37929⟩⟩) (.authority (.relationPreimageSource ⟨50⟩))

def exact77919RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩, (1)⟩]

theorem exact77919RawTermsValid :
    exact77919RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77919 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37929⟩⟩) exact77919RawTerms (.finite 5647228698) 77918 .exactZero (none)

def event77920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37931⟩⟩) 0 ⟨37929⟩ 77919

def event77921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37931⟩⟩) 1 ⟨2370⟩ 4

def event77922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37931⟩⟩) (.scale (.predecessor 0 77920 .coefficient) (.value (.predecessor 1 77921 .coefficient)))

def exact77923RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩, (1)⟩]

theorem exact77923RawTermsValid :
    exact77923RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77923 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37931⟩⟩) exact77923RawTerms (.finite 5647228698) 77922 .exactZero (none)

def event77924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37932⟩⟩) 0 ⟨10368⟩ 75995

def event77925 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37932⟩⟩) 1 ⟨37931⟩ 77923

def event77926 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37932⟩⟩) (.product (.predecessor 0 77924 .coefficient) (.predecessor 1 77925 .coefficient) (⟨false, false, none, none, none⟩))

def event77927 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37932⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩) [⟨.result 77919 .coefficient, false, none⟩])

def event77928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37932⟩⟩) (.product (.result 75995 .summary) (.transfer 77927) (⟨false, false, none, none, none⟩))

def event77929 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37932⟩⟩, .operator (⟨75995, 0⟩, ⟨77923, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩, (1)⟩)

def event77930 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨37930⟩⟩)

def event77931 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event77932 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event77933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event77934 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event77935 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event77936 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event77937 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event77938 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event77939 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 77938

def event77940 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 77936

def event77941 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 77939 .coefficient) (.value (.predecessor 1 77940 .coefficient)))

def event77942 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event77943 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 77942

def event77944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 77934

def event77945 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 77943 .coefficient, .predecessor 1 77944 .coefficient])

def event77946 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event77947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 77946

def event77948 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 77932

def event77949 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 77948 .coefficient))

def event77950 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event77951 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37258⟩⟩) 0 ⟨10325⟩ 77950

def event77952 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37258⟩⟩) (.authority (.programFamilyFact))

def exact77953RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37258⟩⟩], []⟩, (1)⟩]

theorem exact77953RawTermsValid :
    exact77953RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77953 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37258⟩⟩) exact77953RawTerms (.finite 42) 77952 .exactZero (none)

def event77954 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13971⟩⟩) 0 ⟨10325⟩ 77950

def event77955 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13971⟩⟩) (.authority (.programFamilyFact))

def exact77956RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩], []⟩, (1)⟩]

theorem exact77956RawTermsValid :
    exact77956RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77956 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13971⟩⟩) exact77956RawTerms (.finite 42) 77955 .exactZero (none)

def event77957 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37259⟩⟩) 0 ⟨13971⟩ 77956

def event77958 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37259⟩⟩) 1 ⟨37258⟩ 77953

def event77959 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37259⟩⟩) (.product (.predecessor 0 77957 .coefficient) (.predecessor 1 77958 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event77960 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37259⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], []⟩) [⟨.result 77956 .coefficient, true, some 1⟩, ⟨.result 77953 .coefficient, true, some 1⟩])

def event77961 : Event := .survivorFold (1) 77960

def exact77962RawTerms : List Term := []

theorem exact77962RawTermsValid :
    exact77962RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77962 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37259⟩⟩) exact77962RawTerms (.finite 1764) 77959 (.finite 1764) (some (77960))

def event77963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37260⟩⟩) 0 ⟨37259⟩ 77962

def event77964 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37260⟩⟩) (.identity (.predecessor 0 77963 .coefficient))

def event77965 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37260⟩⟩) (.finite 1764)

def event77966 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37929⟩⟩) 0 ⟨37260⟩ 77965

def event77967 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37929⟩⟩) (.authority (.relationPreimageSource ⟨50⟩))

def exact77968RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩, (1)⟩]

theorem exact77968RawTermsValid :
    exact77968RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77968 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37929⟩⟩) exact77968RawTerms (.finite 5647228698) 77967 .exactZero (none)

def event77969 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact77970RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact77970RawTermsValid :
    exact77970RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77970 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact77970RawTerms .large 77969 .exactZero (none)

def event77971 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37930⟩⟩) 0 ⟨35⟩ 77970

def event77972 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37930⟩⟩) 1 ⟨37929⟩ 77968

def event77973 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37930⟩⟩) (.product (.predecessor 0 77971 .coefficient) (.predecessor 1 77972 .coefficient) (⟨false, false, none, none, none⟩))

def event77974 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37930⟩⟩, .operator (⟨77970, 0⟩, ⟨77968, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩, (1)⟩)

def exact77975RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩, (1)⟩]

theorem exact77975RawTermsValid :
    exact77975RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77975 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37930⟩⟩) exact77975RawTerms .large 77973 .exactZero (none)

def event77976 : Event := .preFoldPolynomial 77975 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩, (1)⟩] .exactZero none

def exact77977RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩, (1)⟩]

def event77977 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨37930⟩⟩) 77976 exact77977RawTerms .large 77973 .exactZero (none)

def event77978 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨39009⟩⟩)

def event77979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event77980 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event77981 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event77982 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event77983 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event77984 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event77985 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event77986 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event77987 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 77986

def event77988 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 77984

def event77989 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 77987 .coefficient) (.value (.predecessor 1 77988 .coefficient)))

def event77990 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event77991 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 77990

def event77992 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 77982

def event77993 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 77991 .coefficient, .predecessor 1 77992 .coefficient])

def event77994 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event77995 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 77994

def event77996 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 77980

def event77997 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 77996 .coefficient))

def event77998 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event77999 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37258⟩⟩) 0 ⟨10325⟩ 77998

def event78000 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37258⟩⟩) (.authority (.programFamilyFact))

def exact78001RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37258⟩⟩], []⟩, (1)⟩]

theorem exact78001RawTermsValid :
    exact78001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78001 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37258⟩⟩) exact78001RawTerms (.finite 42) 78000 .exactZero (none)

def event78002 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13971⟩⟩) 0 ⟨10325⟩ 77998

def event78003 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13971⟩⟩) (.authority (.programFamilyFact))

def exact78004RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩], []⟩, (1)⟩]

theorem exact78004RawTermsValid :
    exact78004RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78004 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13971⟩⟩) exact78004RawTerms (.finite 42) 78003 .exactZero (none)

def event78005 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37259⟩⟩) 0 ⟨13971⟩ 78004

def event78006 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37259⟩⟩) 1 ⟨37258⟩ 78001

def event78007 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37259⟩⟩) (.product (.predecessor 0 78005 .coefficient) (.predecessor 1 78006 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event78008 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37259⟩⟩, .operator (⟨78004, 0⟩, ⟨78001, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], []⟩, (1)⟩)

def exact78009RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], []⟩, (1)⟩]

theorem exact78009RawTermsValid :
    exact78009RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78009 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37259⟩⟩) exact78009RawTerms (.finite 1764) 78007 .exactZero (none)

def event78010 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37260⟩⟩) 0 ⟨37259⟩ 78009

def event78011 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37260⟩⟩) (.identity (.predecessor 0 78010 .coefficient))

def event78012 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37260⟩⟩) (.finite 1764)

def event78013 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38464⟩⟩) 0 ⟨37260⟩ 78012

def event78014 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38464⟩⟩) (.authority (.programFamilyFact))

def event78015 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38464⟩⟩) (.finite 3720)

def event78016 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event78017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38465⟩⟩) 0 ⟨7177⟩ 78016

def event78018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38465⟩⟩) 1 ⟨38464⟩ 78015

def event78019 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38465⟩⟩) (.authority (.operator))

def exact78020RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38465⟩⟩]⟩, (1)⟩]

theorem exact78020RawTermsValid :
    exact78020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78020 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38465⟩⟩) exact78020RawTerms .large 78019 .exactZero (none)

def event78021 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39005⟩⟩) 0 ⟨38465⟩ 78020

def event78022 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39005⟩⟩) (.authority (.operator))

def exact78023RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩, (1)⟩]

theorem exact78023RawTermsValid :
    exact78023RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78023 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39005⟩⟩) exact78023RawTerms (.finite 8192) 78022 .exactZero (none)

def event78024 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event78025 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event78026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38730⟩⟩) 0 ⟨37260⟩ 78012

def event78027 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38730⟩⟩) 1 ⟨136⟩ 78025

def event78028 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38730⟩⟩) (.sum [.predecessor 0 78026 .coefficient, .predecessor 1 78027 .coefficient])

def event78029 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38730⟩⟩) (.finite 1764)

def event78030 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38731⟩⟩) 0 ⟨38730⟩ 78029

def event78031 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38731⟩⟩) (.identity (.predecessor 0 78030 .coefficient))

def exact78032RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], []⟩, (1)⟩]

theorem exact78032RawTermsValid :
    exact78032RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78032 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38731⟩⟩) exact78032RawTerms (.finite 1764) 78031 .exactZero (none)

def event78033 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact78034RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact78034RawTermsValid :
    exact78034RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78034 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact78034RawTerms .large 78033 .exactZero (none)

def event78035 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38732⟩⟩) 0 ⟨6908⟩ 78034

def event78036 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38732⟩⟩) 1 ⟨38731⟩ 78032

def event78037 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38732⟩⟩) (.product (.predecessor 0 78035 .coefficient) (.predecessor 1 78036 .coefficient) (⟨false, false, none, none, none⟩))

def event78038 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38732⟩⟩, .operator (⟨78034, 0⟩, ⟨78032, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact78039RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact78039RawTermsValid :
    exact78039RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78039 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38732⟩⟩) exact78039RawTerms .large 78037 .exactZero (none)

def event78040 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event78041 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event78042 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 78016

def event78043 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact78044RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact78044RawTermsValid :
    exact78044RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78044 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact78044RawTerms .large 78043 .exactZero (none)

def event78045 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7281⟩⟩) 0 ⟨7178⟩ 78044

def event78046 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7281⟩⟩) (.identity (.predecessor 0 78045 .coefficient))

def exact78047RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩, (1)⟩]

theorem exact78047RawTermsValid :
    exact78047RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78047 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7281⟩⟩) exact78047RawTerms .large 78046 .exactZero (none)

def event78048 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9553⟩⟩) 0 ⟨7281⟩ 78047

def event78049 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9553⟩⟩) (.authority (.operator))

def exact78050RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩]

theorem exact78050RawTermsValid :
    exact78050RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78050 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9553⟩⟩) exact78050RawTerms (.finite 8192) 78049 .exactZero (none)

def event78051 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9554⟩⟩) 0 ⟨9553⟩ 78050

def event78052 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9554⟩⟩) 1 ⟨2370⟩ 78041

def event78053 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9554⟩⟩) (.scale (.predecessor 0 78051 .coefficient) (.value (.predecessor 1 78052 .coefficient)))

def exact78054RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩]

theorem exact78054RawTermsValid :
    exact78054RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78054 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9554⟩⟩) exact78054RawTerms (.finite 8192) 78053 .exactZero (none)

def event78055 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7298⟩⟩) 0 ⟨7178⟩ 78044

def event78056 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7298⟩⟩) (.identity (.predecessor 0 78055 .coefficient))

def exact78057RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩, (1)⟩]

theorem exact78057RawTermsValid :
    exact78057RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78057 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7298⟩⟩) exact78057RawTerms .large 78056 .exactZero (none)

def event78058 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9555⟩⟩) 0 ⟨7298⟩ 78057

def event78059 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9555⟩⟩) 1 ⟨9554⟩ 78054

def event78060 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9555⟩⟩) (.product (.predecessor 0 78058 .coefficient) (.predecessor 1 78059 .coefficient) (⟨false, false, none, none, none⟩))

def event78061 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9555⟩⟩, .operator (⟨78057, 0⟩, ⟨78054, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩)

def exact78062RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩]

theorem exact78062RawTermsValid :
    exact78062RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78062 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9555⟩⟩) exact78062RawTerms .large 78060 .exactZero (none)

def event78063 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38733⟩⟩) 0 ⟨9555⟩ 78062

def event78064 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38733⟩⟩) 1 ⟨38732⟩ 78039

def event78065 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38733⟩⟩) (.sum [.predecessor 0 78063 .coefficient, .predecessor 1 78064 .coefficient])

def exact78066RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact78066RawTermsValid :
    exact78066RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78066 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38733⟩⟩) exact78066RawTerms .large 78065 .exactZero (none)

def event78067 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39008⟩⟩) 0 ⟨38733⟩ 78066

def event78068 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39008⟩⟩) 1 ⟨39005⟩ 78023

def event78069 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39008⟩⟩) (.product (.predecessor 0 78067 .coefficient) (.predecessor 1 78068 .coefficient) (⟨false, false, none, none, none⟩))

def event78070 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39008⟩⟩, .operator (⟨78066, 0⟩, ⟨78023, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩, (1)⟩)

def event78071 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39008⟩⟩, .operator (⟨78066, 1⟩, ⟨78023, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩, (-1)⟩)

def event78072 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39008⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39005⟩⟩) ⟨38465⟩ 78020)

def event78073 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39008⟩⟩, .relation 78072 0, ⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨38465⟩⟩]⟩, (-1)⟩)

def exact78074RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨38465⟩⟩]⟩, (-1)⟩]

theorem exact78074RawTermsValid :
    exact78074RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78074 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39008⟩⟩) exact78074RawTerms .large 78069 .exactZero (none)

def event78075 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37476⟩⟩) 0 ⟨37260⟩ 78012

def event78076 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37476⟩⟩) (.authority (.programFamilyFact))

def exact78077RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], []⟩, (1)⟩]

theorem exact78077RawTermsValid :
    exact78077RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event78077 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37476⟩⟩) exact78077RawTerms (.finite 42) 78076 .exactZero (none)

def event78078 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37478⟩⟩) 0 ⟨6908⟩ 78034

def event78079 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37478⟩⟩) 1 ⟨37476⟩ 78077

def eventLeaf4864 : Array AnnotatedEvent := #[
  { event := event77824
    frameStart := 0 },
  { event := event77825
    frameStart := 0 },
  { event := event77826
    frameStart := 0 },
  { event := event77827
    frameStart := 0 },
  { event := event77828
    frameStart := 0 },
  { event := event77829
    frameStart := 0 },
  { event := event77830
    frameStart := 0 },
  { event := event77831
    frameStart := 0 },
  { event := event77832
    frameStart := 0 },
  { event := event77833
    frameStart := 0 },
  { event := event77834
    frameStart := 0 },
  { event := event77835
    frameStart := 0 },
  { event := event77836
    frameStart := 0 },
  { event := event77837
    frameStart := 0 },
  { event := event77838
    frameStart := 0 },
  { event := event77839
    frameStart := 0 }
]

def eventLeaf4865 : Array AnnotatedEvent := #[
  { event := event77840
    frameStart := 0 },
  { event := event77841
    frameStart := 0 },
  { event := event77842
    frameStart := 0 },
  { event := event77843
    frameStart := 0 },
  { event := event77844
    frameStart := 0 },
  { event := event77845
    frameStart := 0 },
  { event := event77846
    frameStart := 0 },
  { event := event77847
    frameStart := 0 },
  { event := event77848
    frameStart := 0 },
  { event := event77849
    frameStart := 0 },
  { event := event77850
    frameStart := 0 },
  { event := event77851
    frameStart := 0 },
  { event := event77852
    frameStart := 0 },
  { event := event77853
    frameStart := 0 },
  { event := event77854
    frameStart := 0 },
  { event := event77855
    frameStart := 0 }
]

def eventLeaf4866 : Array AnnotatedEvent := #[
  { event := event77856
    frameStart := 0 },
  { event := event77857
    frameStart := 0 },
  { event := event77858
    frameStart := 0 },
  { event := event77859
    frameStart := 0 },
  { event := event77860
    frameStart := 0 },
  { event := event77861
    frameStart := 0 },
  { event := event77862
    frameStart := 0 },
  { event := event77863
    frameStart := 0 },
  { event := event77864
    frameStart := 0 },
  { event := event77865
    frameStart := 0 },
  { event := event77866
    frameStart := 0 },
  { event := event77867
    frameStart := 0 },
  { event := event77868
    frameStart := 0 },
  { event := event77869
    frameStart := 0 },
  { event := event77870
    frameStart := 0 },
  { event := event77871
    frameStart := 0 }
]

def eventLeaf4867 : Array AnnotatedEvent := #[
  { event := event77872
    frameStart := 0 },
  { event := event77873
    frameStart := 0 },
  { event := event77874
    frameStart := 0 },
  { event := event77875
    frameStart := 0 },
  { event := event77876
    frameStart := 0 },
  { event := event77877
    frameStart := 0 },
  { event := event77878
    frameStart := 0 },
  { event := event77879
    frameStart := 0 },
  { event := event77880
    frameStart := 0 },
  { event := event77881
    frameStart := 0 },
  { event := event77882
    frameStart := 0 },
  { event := event77883
    frameStart := 0 },
  { event := event77884
    frameStart := 0 },
  { event := event77885
    frameStart := 0 },
  { event := event77886
    frameStart := 0 },
  { event := event77887
    frameStart := 0 }
]

def eventLeaf4868 : Array AnnotatedEvent := #[
  { event := event77888
    frameStart := 0 },
  { event := event77889
    frameStart := 0 },
  { event := event77890
    frameStart := 0 },
  { event := event77891
    frameStart := 0 },
  { event := event77892
    frameStart := 0 },
  { event := event77893
    frameStart := 0 },
  { event := event77894
    frameStart := 0 },
  { event := event77895
    frameStart := 0 },
  { event := event77896
    frameStart := 0 },
  { event := event77897
    frameStart := 0 },
  { event := event77898
    frameStart := 0 },
  { event := event77899
    frameStart := 0 },
  { event := event77900
    frameStart := 0 },
  { event := event77901
    frameStart := 0 },
  { event := event77902
    frameStart := 0 },
  { event := event77903
    frameStart := 0 }
]

def eventLeaf4869 : Array AnnotatedEvent := #[
  { event := event77904
    frameStart := 0 },
  { event := event77905
    frameStart := 0 },
  { event := event77906
    frameStart := 0 },
  { event := event77907
    frameStart := 0 },
  { event := event77908
    frameStart := 0 },
  { event := event77909
    frameStart := 0 },
  { event := event77910
    frameStart := 0 },
  { event := event77911
    frameStart := 0 },
  { event := event77912
    frameStart := 0 },
  { event := event77913
    frameStart := 0 },
  { event := event77914
    frameStart := 0 },
  { event := event77915
    frameStart := 0 },
  { event := event77916
    frameStart := 0 },
  { event := event77917
    frameStart := 0 },
  { event := event77918
    frameStart := 0 },
  { event := event77919
    frameStart := 0 }
]

def eventLeaf4870 : Array AnnotatedEvent := #[
  { event := event77920
    frameStart := 0 },
  { event := event77921
    frameStart := 0 },
  { event := event77922
    frameStart := 0 },
  { event := event77923
    frameStart := 0 },
  { event := event77924
    frameStart := 0 },
  { event := event77925
    frameStart := 0 },
  { event := event77926
    frameStart := 0 },
  { event := event77927
    frameStart := 0 },
  { event := event77928
    frameStart := 0 },
  { event := event77929
    frameStart := 0 },
  { event := event77930
    frameStart := 77930 },
  { event := event77931
    frameStart := 77930 },
  { event := event77932
    frameStart := 77930 },
  { event := event77933
    frameStart := 77930 },
  { event := event77934
    frameStart := 77930 },
  { event := event77935
    frameStart := 77930 }
]

def eventLeaf4871 : Array AnnotatedEvent := #[
  { event := event77936
    frameStart := 77930 },
  { event := event77937
    frameStart := 77930 },
  { event := event77938
    frameStart := 77930 },
  { event := event77939
    frameStart := 77930 },
  { event := event77940
    frameStart := 77930 },
  { event := event77941
    frameStart := 77930 },
  { event := event77942
    frameStart := 77930 },
  { event := event77943
    frameStart := 77930 },
  { event := event77944
    frameStart := 77930 },
  { event := event77945
    frameStart := 77930 },
  { event := event77946
    frameStart := 77930 },
  { event := event77947
    frameStart := 77930 },
  { event := event77948
    frameStart := 77930 },
  { event := event77949
    frameStart := 77930 },
  { event := event77950
    frameStart := 77930 },
  { event := event77951
    frameStart := 77930 }
]

def eventLeaf4872 : Array AnnotatedEvent := #[
  { event := event77952
    frameStart := 77930 },
  { event := event77953
    frameStart := 77930 },
  { event := event77954
    frameStart := 77930 },
  { event := event77955
    frameStart := 77930 },
  { event := event77956
    frameStart := 77930 },
  { event := event77957
    frameStart := 77930 },
  { event := event77958
    frameStart := 77930 },
  { event := event77959
    frameStart := 77930 },
  { event := event77960
    frameStart := 77930 },
  { event := event77961
    frameStart := 77930 },
  { event := event77962
    frameStart := 77930 },
  { event := event77963
    frameStart := 77930 },
  { event := event77964
    frameStart := 77930 },
  { event := event77965
    frameStart := 77930 },
  { event := event77966
    frameStart := 77930 },
  { event := event77967
    frameStart := 77930 }
]

def eventLeaf4873 : Array AnnotatedEvent := #[
  { event := event77968
    frameStart := 77930 },
  { event := event77969
    frameStart := 77930 },
  { event := event77970
    frameStart := 77930 },
  { event := event77971
    frameStart := 77930 },
  { event := event77972
    frameStart := 77930 },
  { event := event77973
    frameStart := 77930 },
  { event := event77974
    frameStart := 77930 },
  { event := event77975
    frameStart := 77930 },
  { event := event77976
    frameStart := 77930 },
  { event := event77977
    frameStart := 77930 },
  { event := event77978
    frameStart := 77978 },
  { event := event77979
    frameStart := 77978 },
  { event := event77980
    frameStart := 77978 },
  { event := event77981
    frameStart := 77978 },
  { event := event77982
    frameStart := 77978 },
  { event := event77983
    frameStart := 77978 }
]

def eventLeaf4874 : Array AnnotatedEvent := #[
  { event := event77984
    frameStart := 77978 },
  { event := event77985
    frameStart := 77978 },
  { event := event77986
    frameStart := 77978 },
  { event := event77987
    frameStart := 77978 },
  { event := event77988
    frameStart := 77978 },
  { event := event77989
    frameStart := 77978 },
  { event := event77990
    frameStart := 77978 },
  { event := event77991
    frameStart := 77978 },
  { event := event77992
    frameStart := 77978 },
  { event := event77993
    frameStart := 77978 },
  { event := event77994
    frameStart := 77978 },
  { event := event77995
    frameStart := 77978 },
  { event := event77996
    frameStart := 77978 },
  { event := event77997
    frameStart := 77978 },
  { event := event77998
    frameStart := 77978 },
  { event := event77999
    frameStart := 77978 }
]

def eventLeaf4875 : Array AnnotatedEvent := #[
  { event := event78000
    frameStart := 77978 },
  { event := event78001
    frameStart := 77978 },
  { event := event78002
    frameStart := 77978 },
  { event := event78003
    frameStart := 77978 },
  { event := event78004
    frameStart := 77978 },
  { event := event78005
    frameStart := 77978 },
  { event := event78006
    frameStart := 77978 },
  { event := event78007
    frameStart := 77978 },
  { event := event78008
    frameStart := 77978 },
  { event := event78009
    frameStart := 77978 },
  { event := event78010
    frameStart := 77978 },
  { event := event78011
    frameStart := 77978 },
  { event := event78012
    frameStart := 77978 },
  { event := event78013
    frameStart := 77978 },
  { event := event78014
    frameStart := 77978 },
  { event := event78015
    frameStart := 77978 }
]

def eventLeaf4876 : Array AnnotatedEvent := #[
  { event := event78016
    frameStart := 77978 },
  { event := event78017
    frameStart := 77978 },
  { event := event78018
    frameStart := 77978 },
  { event := event78019
    frameStart := 77978 },
  { event := event78020
    frameStart := 77978 },
  { event := event78021
    frameStart := 77978 },
  { event := event78022
    frameStart := 77978 },
  { event := event78023
    frameStart := 77978 },
  { event := event78024
    frameStart := 77978 },
  { event := event78025
    frameStart := 77978 },
  { event := event78026
    frameStart := 77978 },
  { event := event78027
    frameStart := 77978 },
  { event := event78028
    frameStart := 77978 },
  { event := event78029
    frameStart := 77978 },
  { event := event78030
    frameStart := 77978 },
  { event := event78031
    frameStart := 77978 }
]

def eventLeaf4877 : Array AnnotatedEvent := #[
  { event := event78032
    frameStart := 77978 },
  { event := event78033
    frameStart := 77978 },
  { event := event78034
    frameStart := 77978 },
  { event := event78035
    frameStart := 77978 },
  { event := event78036
    frameStart := 77978 },
  { event := event78037
    frameStart := 77978 },
  { event := event78038
    frameStart := 77978 },
  { event := event78039
    frameStart := 77978 },
  { event := event78040
    frameStart := 77978 },
  { event := event78041
    frameStart := 77978 },
  { event := event78042
    frameStart := 77978 },
  { event := event78043
    frameStart := 77978 },
  { event := event78044
    frameStart := 77978 },
  { event := event78045
    frameStart := 77978 },
  { event := event78046
    frameStart := 77978 },
  { event := event78047
    frameStart := 77978 }
]

def eventLeaf4878 : Array AnnotatedEvent := #[
  { event := event78048
    frameStart := 77978 },
  { event := event78049
    frameStart := 77978 },
  { event := event78050
    frameStart := 77978 },
  { event := event78051
    frameStart := 77978 },
  { event := event78052
    frameStart := 77978 },
  { event := event78053
    frameStart := 77978 },
  { event := event78054
    frameStart := 77978 },
  { event := event78055
    frameStart := 77978 },
  { event := event78056
    frameStart := 77978 },
  { event := event78057
    frameStart := 77978 },
  { event := event78058
    frameStart := 77978 },
  { event := event78059
    frameStart := 77978 },
  { event := event78060
    frameStart := 77978 },
  { event := event78061
    frameStart := 77978 },
  { event := event78062
    frameStart := 77978 },
  { event := event78063
    frameStart := 77978 }
]

def eventLeaf4879 : Array AnnotatedEvent := #[
  { event := event78064
    frameStart := 77978 },
  { event := event78065
    frameStart := 77978 },
  { event := event78066
    frameStart := 77978 },
  { event := event78067
    frameStart := 77978 },
  { event := event78068
    frameStart := 77978 },
  { event := event78069
    frameStart := 77978 },
  { event := event78070
    frameStart := 77978 },
  { event := event78071
    frameStart := 77978 },
  { event := event78072
    frameStart := 77978 },
  { event := event78073
    frameStart := 77978 },
  { event := event78074
    frameStart := 77978 },
  { event := event78075
    frameStart := 77978 },
  { event := event78076
    frameStart := 77978 },
  { event := event78077
    frameStart := 77978 },
  { event := event78078
    frameStart := 77978 },
  { event := event78079
    frameStart := 77978 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events304
