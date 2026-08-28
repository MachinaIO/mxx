import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events261

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact66816RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact66816RawTermsValid :
    exact66816RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66816 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12761⟩⟩) exact66816RawTerms .large 66813 (.finite 95458688) (some (66815))

def event66817 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25523⟩⟩) 0 ⟨12761⟩ 66816

def event66818 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25523⟩⟩) 1 ⟨25522⟩ 66752

def event66819 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25523⟩⟩) (.product (.predecessor 0 66817 .coefficient) (.predecessor 1 66818 .coefficient) (⟨false, false, none, none, none⟩))

def event66820 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25523⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨25522⟩⟩]⟩) [⟨.result 66752 .coefficient, false, none⟩])

def event66821 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25523⟩⟩) (.product (.result 66816 .summary) (.transfer 66820) (⟨false, false, none, none, none⟩))

def event66822 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25523⟩⟩, .operator (⟨66816, 1⟩, ⟨66752, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25522⟩⟩]⟩, (-1)⟩)

def event66823 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25523⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25522⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25522⟩⟩) ⟨23288⟩ 66749)

def event66824 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25523⟩⟩, .relation 66823 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨23288⟩⟩]⟩, (-1)⟩)

def event66825 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25523⟩⟩, .operator (⟨66816, 0⟩, ⟨66752, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25522⟩⟩]⟩, (1)⟩)

def exact66826RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25522⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨23288⟩⟩]⟩, (-1)⟩]

theorem exact66826RawTermsValid :
    exact66826RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66826 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25523⟩⟩) exact66826RawTerms .large 66819 (.finite 350334912299008) (some (66821))

def event66827 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20028⟩⟩) 0 ⟨12756⟩ 3166

def event66828 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20028⟩⟩) (.authority (.relationPreimageSource ⟨23⟩))

def exact66829RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20028⟩⟩]⟩, (1)⟩]

theorem exact66829RawTermsValid :
    exact66829RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66829 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20028⟩⟩) exact66829RawTerms (.finite 136065468) 66828 .exactZero (none)

def event66830 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20030⟩⟩) 0 ⟨20028⟩ 66829

def event66831 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20030⟩⟩) 1 ⟨2348⟩ 4

def event66832 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20030⟩⟩) (.scale (.predecessor 0 66830 .coefficient) (.value (.predecessor 1 66831 .coefficient)))

def exact66833RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20028⟩⟩]⟩, (1)⟩]

theorem exact66833RawTermsValid :
    exact66833RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66833 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20030⟩⟩) exact66833RawTerms (.finite 136065468) 66832 .exactZero (none)

def event66834 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20031⟩⟩) 0 ⟨5535⟩ 65387

def event66835 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20031⟩⟩) 1 ⟨20030⟩ 66833

def event66836 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20031⟩⟩) (.product (.predecessor 0 66834 .coefficient) (.predecessor 1 66835 .coefficient) (⟨false, false, none, none, none⟩))

def event66837 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20031⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20028⟩⟩]⟩) [⟨.result 66829 .coefficient, false, none⟩])

def event66838 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20031⟩⟩) (.product (.result 65387 .summary) (.transfer 66837) (⟨false, false, none, none, none⟩))

def event66839 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20031⟩⟩, .operator (⟨65387, 0⟩, ⟨66833, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20028⟩⟩]⟩, (1)⟩)

def event66840 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20029⟩⟩)

def event66841 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event66842 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event66843 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event66844 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event66845 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event66846 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event66847 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event66848 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event66849 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 66848

def event66850 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 66846

def event66851 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 66849 .coefficient) (.value (.predecessor 1 66850 .coefficient)))

def event66852 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event66853 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 66852

def event66854 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 66844

def event66855 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 66853 .coefficient, .predecessor 1 66854 .coefficient])

def event66856 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event66857 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 66856

def event66858 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 66842

def event66859 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 66858 .coefficient))

def event66860 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event66861 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12754⟩⟩) 0 ⟨5530⟩ 66860

def event66862 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12754⟩⟩) (.authority (.programFamilyFact))

def exact66863RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12754⟩⟩], []⟩, (1)⟩]

theorem exact66863RawTermsValid :
    exact66863RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66863 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12754⟩⟩) exact66863RawTerms (.finite 46) 66862 .exactZero (none)

def event66864 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10025⟩⟩) 0 ⟨5530⟩ 66860

def event66865 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10025⟩⟩) (.authority (.programFamilyFact))

def exact66866RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩], []⟩, (1)⟩]

theorem exact66866RawTermsValid :
    exact66866RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66866 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10025⟩⟩) exact66866RawTerms (.finite 46) 66865 .exactZero (none)

def event66867 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12755⟩⟩) 0 ⟨10025⟩ 66866

def event66868 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12755⟩⟩) 1 ⟨12754⟩ 66863

def event66869 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12755⟩⟩) (.product (.predecessor 0 66867 .coefficient) (.predecessor 1 66868 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event66870 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12755⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], []⟩) [⟨.result 66866 .coefficient, true, some 1⟩, ⟨.result 66863 .coefficient, true, some 1⟩])

def event66871 : Event := .survivorFold (1) 66870

def exact66872RawTerms : List Term := []

theorem exact66872RawTermsValid :
    exact66872RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66872 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12755⟩⟩) exact66872RawTerms (.finite 2116) 66869 (.finite 2116) (some (66870))

def event66873 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12756⟩⟩) 0 ⟨12755⟩ 66872

def event66874 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12756⟩⟩) (.identity (.predecessor 0 66873 .coefficient))

def event66875 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12756⟩⟩) (.finite 2116)

def event66876 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20028⟩⟩) 0 ⟨12756⟩ 66875

def event66877 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20028⟩⟩) (.authority (.relationPreimageSource ⟨23⟩))

def exact66878RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20028⟩⟩]⟩, (1)⟩]

theorem exact66878RawTermsValid :
    exact66878RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66878 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20028⟩⟩) exact66878RawTerms (.finite 136065468) 66877 .exactZero (none)

def event66879 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact66880RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact66880RawTermsValid :
    exact66880RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66880 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact66880RawTerms .large 66879 .exactZero (none)

def event66881 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20029⟩⟩) 0 ⟨6⟩ 66880

def event66882 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20029⟩⟩) 1 ⟨20028⟩ 66878

def event66883 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20029⟩⟩) (.product (.predecessor 0 66881 .coefficient) (.predecessor 1 66882 .coefficient) (⟨false, false, none, none, none⟩))

def event66884 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20029⟩⟩, .operator (⟨66880, 0⟩, ⟨66878, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20028⟩⟩]⟩, (1)⟩)

def exact66885RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20028⟩⟩]⟩, (1)⟩]

theorem exact66885RawTermsValid :
    exact66885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66885 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20029⟩⟩) exact66885RawTerms .large 66883 .exactZero (none)

def event66886 : Event := .preFoldPolynomial 66885 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20028⟩⟩]⟩, (1)⟩] .exactZero none

def exact66887RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20028⟩⟩]⟩, (1)⟩]

def event66887 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20029⟩⟩) 66886 exact66887RawTerms .large 66883 .exactZero (none)

def event66888 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25526⟩⟩)

def event66889 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event66890 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event66891 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event66892 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event66893 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event66894 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event66895 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event66896 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event66897 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 66896

def event66898 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 66894

def event66899 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 66897 .coefficient) (.value (.predecessor 1 66898 .coefficient)))

def event66900 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event66901 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 66900

def event66902 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 66892

def event66903 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 66901 .coefficient, .predecessor 1 66902 .coefficient])

def event66904 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event66905 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 66904

def event66906 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 66890

def event66907 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 66906 .coefficient))

def event66908 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event66909 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12754⟩⟩) 0 ⟨5530⟩ 66908

def event66910 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12754⟩⟩) (.authority (.programFamilyFact))

def exact66911RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12754⟩⟩], []⟩, (1)⟩]

theorem exact66911RawTermsValid :
    exact66911RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66911 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12754⟩⟩) exact66911RawTerms (.finite 46) 66910 .exactZero (none)

def event66912 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10025⟩⟩) 0 ⟨5530⟩ 66908

def event66913 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10025⟩⟩) (.authority (.programFamilyFact))

def exact66914RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩], []⟩, (1)⟩]

theorem exact66914RawTermsValid :
    exact66914RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66914 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10025⟩⟩) exact66914RawTerms (.finite 46) 66913 .exactZero (none)

def event66915 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12755⟩⟩) 0 ⟨10025⟩ 66914

def event66916 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12755⟩⟩) 1 ⟨12754⟩ 66911

def event66917 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12755⟩⟩) (.product (.predecessor 0 66915 .coefficient) (.predecessor 1 66916 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event66918 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12755⟩⟩, .operator (⟨66914, 0⟩, ⟨66911, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], []⟩, (1)⟩)

def exact66919RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], []⟩, (1)⟩]

theorem exact66919RawTermsValid :
    exact66919RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66919 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12755⟩⟩) exact66919RawTerms (.finite 2116) 66917 .exactZero (none)

def event66920 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12756⟩⟩) 0 ⟨12755⟩ 66919

def event66921 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12756⟩⟩) (.identity (.predecessor 0 66920 .coefficient))

def event66922 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12756⟩⟩) (.finite 2116)

def event66923 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23287⟩⟩) 0 ⟨12756⟩ 66922

def event66924 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23287⟩⟩) (.authority (.programFamilyFact))

def event66925 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23287⟩⟩) (.finite 3720)

def event66926 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event66927 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23288⟩⟩) 0 ⟨6689⟩ 66926

def event66928 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23288⟩⟩) 1 ⟨23287⟩ 66925

def event66929 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23288⟩⟩) (.authority (.operator))

def exact66930RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23288⟩⟩]⟩, (1)⟩]

theorem exact66930RawTermsValid :
    exact66930RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66930 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23288⟩⟩) exact66930RawTerms .large 66929 .exactZero (none)

def event66931 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25522⟩⟩) 0 ⟨23288⟩ 66930

def event66932 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25522⟩⟩) (.authority (.operator))

def exact66933RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25522⟩⟩]⟩, (1)⟩]

theorem exact66933RawTermsValid :
    exact66933RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66933 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25522⟩⟩) exact66933RawTerms (.finite 8192) 66932 .exactZero (none)

def event66934 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event66935 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event66936 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12854⟩⟩) 0 ⟨12756⟩ 66922

def event66937 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12854⟩⟩) 1 ⟨110⟩ 66935

def event66938 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12854⟩⟩) (.sum [.predecessor 0 66936 .coefficient, .predecessor 1 66937 .coefficient])

def event66939 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12854⟩⟩) (.finite 2116)

def event66940 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12855⟩⟩) 0 ⟨12854⟩ 66939

def event66941 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12855⟩⟩) (.identity (.predecessor 0 66940 .coefficient))

def exact66942RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], []⟩, (1)⟩]

theorem exact66942RawTermsValid :
    exact66942RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66942 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12855⟩⟩) exact66942RawTerms (.finite 2116) 66941 .exactZero (none)

def event66943 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact66944RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact66944RawTermsValid :
    exact66944RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66944 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact66944RawTerms .large 66943 .exactZero (none)

def event66945 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12856⟩⟩) 0 ⟨6544⟩ 66944

def event66946 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12856⟩⟩) 1 ⟨12855⟩ 66942

def event66947 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12856⟩⟩) (.product (.predecessor 0 66945 .coefficient) (.predecessor 1 66946 .coefficient) (⟨false, false, none, none, none⟩))

def event66948 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12856⟩⟩, .operator (⟨66944, 0⟩, ⟨66942, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact66949RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact66949RawTermsValid :
    exact66949RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66949 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12856⟩⟩) exact66949RawTerms .large 66947 .exactZero (none)

def event66950 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event66951 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event66952 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 66926

def event66953 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact66954RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact66954RawTermsValid :
    exact66954RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66954 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact66954RawTerms .large 66953 .exactZero (none)

def event66955 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6787⟩⟩) 0 ⟨6757⟩ 66954

def event66956 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6787⟩⟩) (.identity (.predecessor 0 66955 .coefficient))

def exact66957RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩, (1)⟩]

theorem exact66957RawTermsValid :
    exact66957RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66957 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6787⟩⟩) exact66957RawTerms .large 66956 .exactZero (none)

def event66958 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7873⟩⟩) 0 ⟨6787⟩ 66957

def event66959 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7873⟩⟩) (.authority (.operator))

def exact66960RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩, (1)⟩]

theorem exact66960RawTermsValid :
    exact66960RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66960 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7873⟩⟩) exact66960RawTerms (.finite 8192) 66959 .exactZero (none)

def event66961 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7874⟩⟩) 0 ⟨7873⟩ 66960

def event66962 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7874⟩⟩) 1 ⟨2348⟩ 66951

def event66963 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7874⟩⟩) (.scale (.predecessor 0 66961 .coefficient) (.value (.predecessor 1 66962 .coefficient)))

def exact66964RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩, (1)⟩]

theorem exact66964RawTermsValid :
    exact66964RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66964 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7874⟩⟩) exact66964RawTerms (.finite 8192) 66963 .exactZero (none)

def event66965 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6767⟩⟩) 0 ⟨6757⟩ 66954

def event66966 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6767⟩⟩) (.identity (.predecessor 0 66965 .coefficient))

def exact66967RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩]⟩, (1)⟩]

theorem exact66967RawTermsValid :
    exact66967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66967 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6767⟩⟩) exact66967RawTerms .large 66966 .exactZero (none)

def event66968 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7875⟩⟩) 0 ⟨6767⟩ 66967

def event66969 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7875⟩⟩) 1 ⟨7874⟩ 66964

def event66970 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7875⟩⟩) (.product (.predecessor 0 66968 .coefficient) (.predecessor 1 66969 .coefficient) (⟨false, false, none, none, none⟩))

def event66971 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7875⟩⟩, .operator (⟨66967, 0⟩, ⟨66964, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩, (1)⟩)

def exact66972RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩, (1)⟩]

theorem exact66972RawTermsValid :
    exact66972RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66972 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7875⟩⟩) exact66972RawTerms .large 66970 .exactZero (none)

def event66973 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12857⟩⟩) 0 ⟨7875⟩ 66972

def event66974 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12857⟩⟩) 1 ⟨12856⟩ 66949

def event66975 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12857⟩⟩) (.sum [.predecessor 0 66973 .coefficient, .predecessor 1 66974 .coefficient])

def exact66976RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact66976RawTermsValid :
    exact66976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66976 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12857⟩⟩) exact66976RawTerms .large 66975 .exactZero (none)

def event66977 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25525⟩⟩) 0 ⟨12857⟩ 66976

def event66978 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25525⟩⟩) 1 ⟨25522⟩ 66933

def event66979 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25525⟩⟩) (.product (.predecessor 0 66977 .coefficient) (.predecessor 1 66978 .coefficient) (⟨false, false, none, none, none⟩))

def event66980 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25525⟩⟩, .operator (⟨66976, 0⟩, ⟨66933, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25522⟩⟩]⟩, (1)⟩)

def event66981 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25525⟩⟩, .operator (⟨66976, 1⟩, ⟨66933, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25522⟩⟩]⟩, (-1)⟩)

def event66982 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25525⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25522⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25522⟩⟩) ⟨23288⟩ 66930)

def event66983 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25525⟩⟩, .relation 66982 0, ⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨23288⟩⟩]⟩, (-1)⟩)

def exact66984RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25522⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨23288⟩⟩]⟩, (-1)⟩]

theorem exact66984RawTermsValid :
    exact66984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66984 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25525⟩⟩) exact66984RawTerms .large 66979 .exactZero (none)

def event66985 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16629⟩⟩) 0 ⟨12756⟩ 66922

def event66986 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16629⟩⟩) (.authority (.programFamilyFact))

def exact66987RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], []⟩, (1)⟩]

theorem exact66987RawTermsValid :
    exact66987RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66987 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16629⟩⟩) exact66987RawTerms (.finite 46) 66986 .exactZero (none)

def event66988 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16631⟩⟩) 0 ⟨6544⟩ 66944

def event66989 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16631⟩⟩) 1 ⟨16629⟩ 66987

def event66990 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16631⟩⟩) (.product (.predecessor 0 66988 .coefficient) (.predecessor 1 66989 .coefficient) (⟨false, true, none, none, some 1⟩))

def event66991 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16631⟩⟩, .operator (⟨66944, 0⟩, ⟨66987, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact66992RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact66992RawTermsValid :
    exact66992RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66992 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16631⟩⟩) exact66992RawTerms .large 66990 .exactZero (none)

def event66993 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6704⟩⟩) 0 ⟨6689⟩ 66926

def event66994 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6704⟩⟩) (.authority (.operator))

def exact66995RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩]

theorem exact66995RawTermsValid :
    exact66995RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66995 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6704⟩⟩) exact66995RawTerms .large 66994 .exactZero (none)

def event66996 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16632⟩⟩) 0 ⟨6704⟩ 66995

def event66997 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16632⟩⟩) 1 ⟨16631⟩ 66992

def event66998 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16632⟩⟩) (.sum [.predecessor 0 66996 .coefficient, .predecessor 1 66997 .coefficient])

def exact66999RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact66999RawTermsValid :
    exact66999RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event66999 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16632⟩⟩) exact66999RawTerms .large 66998 .exactZero (none)

def event67000 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25526⟩⟩) 0 ⟨16632⟩ 66999

def event67001 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25526⟩⟩) 1 ⟨25525⟩ 66984

def event67002 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25526⟩⟩) (.sum [.predecessor 0 67000 .coefficient, .predecessor 1 67001 .coefficient])

def exact67003RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25522⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨23288⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact67003RawTermsValid :
    exact67003RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67003 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25526⟩⟩) exact67003RawTerms .large 67002 .exactZero (none)

def event67004 : Event := .preFoldPolynomial 67003 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25522⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨23288⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact67005RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25522⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨23288⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event67005 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25526⟩⟩) 67004 exact67005RawTerms .large 67002 .exactZero (none)

def event67006 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨12756⟩⟩) ⟨⟨117⟩, ⟨23⟩, ⟨109⟩⟩ ⟨66840, 67006⟩

def event67007 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20031⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20028⟩⟩]⟩) (1) 0 2 (.universal 67006 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20028⟩⟩]⟩) (none) 67005)

def event67008 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20031⟩⟩, .relation 67007 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩)

def event67009 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20031⟩⟩, .relation 67007 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25522⟩⟩]⟩, (-1)⟩)

def event67010 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20031⟩⟩, .relation 67007 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨23288⟩⟩]⟩, (1)⟩)

def event67011 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20031⟩⟩, .relation 67007 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact67012RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25522⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨23288⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact67012RawTermsValid :
    exact67012RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67012 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20031⟩⟩) exact67012RawTerms .large 66836 (.finite 1811303510016) (some (66838))

def event67013 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25524⟩⟩) 0 ⟨20031⟩ 67012

def event67014 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25524⟩⟩) 1 ⟨25523⟩ 66826

def event67015 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25524⟩⟩) (.sum [.predecessor 0 67013 .coefficient, .predecessor 1 67014 .coefficient])

def event67016 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25524⟩⟩, .operator (⟨67012, 2⟩, ⟨66826, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨23288⟩⟩]⟩, (-1)⟩)

def event67017 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25524⟩⟩, .operator (⟨67012, 1⟩, ⟨66826, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25522⟩⟩]⟩, (1)⟩)

def event67018 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25524⟩⟩) (.sum [.result 67012 .summary, .result 66826 .summary])

def exact67019RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact67019RawTermsValid :
    exact67019RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67019 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25524⟩⟩) exact67019RawTerms .large 67015 (.finite 352146215809024) (some (67018))

def event67020 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29374⟩⟩) 0 ⟨25524⟩ 67019

def event67021 : Event := .predecessor (⟨.program ⟨214⟩, ⟨29374⟩⟩) 1 ⟨29372⟩ 66742

def event67022 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29374⟩⟩) (.product (.predecessor 0 67020 .coefficient) (.predecessor 1 67021 .coefficient) (⟨false, false, none, none, none⟩))

def event67023 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29374⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨29372⟩⟩]⟩) [⟨.result 66742 .coefficient, false, none⟩])

def event67024 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨29374⟩⟩) (.product (.result 67019 .summary) (.transfer 67023) (⟨false, false, none, none, none⟩))

def event67025 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29374⟩⟩, .operator (⟨67019, 0⟩, ⟨66742, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29372⟩⟩]⟩, (1)⟩)

def event67026 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29374⟩⟩, .operator (⟨67019, 1⟩, ⟨66742, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29372⟩⟩]⟩, (-1)⟩)

def event67027 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨29374⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29372⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29372⟩⟩) ⟨24600⟩ 66739)

def event67028 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨29374⟩⟩, .relation 67027 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨24600⟩⟩]⟩, (-1)⟩)

def exact67029RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩, ⟨.program ⟨214⟩, ⟨29372⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16629⟩⟩], [⟨.program ⟨214⟩, ⟨24600⟩⟩]⟩, (-1)⟩]

theorem exact67029RawTermsValid :
    exact67029RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67029 : Event := .resultExact (⟨.program ⟨214⟩, ⟨29374⟩⟩) exact67029RawTerms .large 67022 (.finite 1292382246358571024384) (some (67024))

def event67030 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22404⟩⟩) 0 ⟨16630⟩ 3172

def event67031 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22404⟩⟩) (.authority (.relationPreimageSource ⟨59⟩))

def exact67032RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22404⟩⟩]⟩, (1)⟩]

theorem exact67032RawTermsValid :
    exact67032RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67032 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22404⟩⟩) exact67032RawTerms (.finite 136065468) 67031 .exactZero (none)

def event67033 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22406⟩⟩) 0 ⟨22404⟩ 67032

def event67034 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22406⟩⟩) 1 ⟨2348⟩ 4

def event67035 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22406⟩⟩) (.scale (.predecessor 0 67033 .coefficient) (.value (.predecessor 1 67034 .coefficient)))

def exact67036RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22404⟩⟩]⟩, (1)⟩]

theorem exact67036RawTermsValid :
    exact67036RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67036 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22406⟩⟩) exact67036RawTerms (.finite 136065468) 67035 .exactZero (none)

def event67037 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22407⟩⟩) 0 ⟨5535⟩ 65387

def event67038 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22407⟩⟩) 1 ⟨22406⟩ 67036

def event67039 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22407⟩⟩) (.product (.predecessor 0 67037 .coefficient) (.predecessor 1 67038 .coefficient) (⟨false, false, none, none, none⟩))

def event67040 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22407⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22404⟩⟩]⟩) [⟨.result 67032 .coefficient, false, none⟩])

def event67041 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22407⟩⟩) (.product (.result 65387 .summary) (.transfer 67040) (⟨false, false, none, none, none⟩))

def event67042 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22407⟩⟩, .operator (⟨65387, 0⟩, ⟨67036, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22404⟩⟩]⟩, (1)⟩)

def event67043 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22405⟩⟩)

def event67044 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event67045 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event67046 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event67047 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event67048 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event67049 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event67050 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event67051 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event67052 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 67051

def event67053 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 67049

def event67054 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 67052 .coefficient) (.value (.predecessor 1 67053 .coefficient)))

def event67055 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event67056 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 67055

def event67057 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 67047

def event67058 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 67056 .coefficient, .predecessor 1 67057 .coefficient])

def event67059 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event67060 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 67059

def event67061 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 67045

def event67062 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 67061 .coefficient))

def event67063 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event67064 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12754⟩⟩) 0 ⟨5530⟩ 67063

def event67065 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12754⟩⟩) (.authority (.programFamilyFact))

def exact67066RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12754⟩⟩], []⟩, (1)⟩]

theorem exact67066RawTermsValid :
    exact67066RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67066 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12754⟩⟩) exact67066RawTerms (.finite 46) 67065 .exactZero (none)

def event67067 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10025⟩⟩) 0 ⟨5530⟩ 67063

def event67068 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10025⟩⟩) (.authority (.programFamilyFact))

def exact67069RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩], []⟩, (1)⟩]

theorem exact67069RawTermsValid :
    exact67069RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67069 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10025⟩⟩) exact67069RawTerms (.finite 46) 67068 .exactZero (none)

def event67070 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12755⟩⟩) 0 ⟨10025⟩ 67069

def event67071 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12755⟩⟩) 1 ⟨12754⟩ 67066

def eventLeaf4176 : Array AnnotatedEvent := #[
  { event := event66816
    frameStart := 0 },
  { event := event66817
    frameStart := 0 },
  { event := event66818
    frameStart := 0 },
  { event := event66819
    frameStart := 0 },
  { event := event66820
    frameStart := 0 },
  { event := event66821
    frameStart := 0 },
  { event := event66822
    frameStart := 0 },
  { event := event66823
    frameStart := 0 },
  { event := event66824
    frameStart := 0 },
  { event := event66825
    frameStart := 0 },
  { event := event66826
    frameStart := 0 },
  { event := event66827
    frameStart := 0 },
  { event := event66828
    frameStart := 0 },
  { event := event66829
    frameStart := 0 },
  { event := event66830
    frameStart := 0 },
  { event := event66831
    frameStart := 0 }
]

def eventLeaf4177 : Array AnnotatedEvent := #[
  { event := event66832
    frameStart := 0 },
  { event := event66833
    frameStart := 0 },
  { event := event66834
    frameStart := 0 },
  { event := event66835
    frameStart := 0 },
  { event := event66836
    frameStart := 0 },
  { event := event66837
    frameStart := 0 },
  { event := event66838
    frameStart := 0 },
  { event := event66839
    frameStart := 0 },
  { event := event66840
    frameStart := 66840 },
  { event := event66841
    frameStart := 66840 },
  { event := event66842
    frameStart := 66840 },
  { event := event66843
    frameStart := 66840 },
  { event := event66844
    frameStart := 66840 },
  { event := event66845
    frameStart := 66840 },
  { event := event66846
    frameStart := 66840 },
  { event := event66847
    frameStart := 66840 }
]

def eventLeaf4178 : Array AnnotatedEvent := #[
  { event := event66848
    frameStart := 66840 },
  { event := event66849
    frameStart := 66840 },
  { event := event66850
    frameStart := 66840 },
  { event := event66851
    frameStart := 66840 },
  { event := event66852
    frameStart := 66840 },
  { event := event66853
    frameStart := 66840 },
  { event := event66854
    frameStart := 66840 },
  { event := event66855
    frameStart := 66840 },
  { event := event66856
    frameStart := 66840 },
  { event := event66857
    frameStart := 66840 },
  { event := event66858
    frameStart := 66840 },
  { event := event66859
    frameStart := 66840 },
  { event := event66860
    frameStart := 66840 },
  { event := event66861
    frameStart := 66840 },
  { event := event66862
    frameStart := 66840 },
  { event := event66863
    frameStart := 66840 }
]

def eventLeaf4179 : Array AnnotatedEvent := #[
  { event := event66864
    frameStart := 66840 },
  { event := event66865
    frameStart := 66840 },
  { event := event66866
    frameStart := 66840 },
  { event := event66867
    frameStart := 66840 },
  { event := event66868
    frameStart := 66840 },
  { event := event66869
    frameStart := 66840 },
  { event := event66870
    frameStart := 66840 },
  { event := event66871
    frameStart := 66840 },
  { event := event66872
    frameStart := 66840 },
  { event := event66873
    frameStart := 66840 },
  { event := event66874
    frameStart := 66840 },
  { event := event66875
    frameStart := 66840 },
  { event := event66876
    frameStart := 66840 },
  { event := event66877
    frameStart := 66840 },
  { event := event66878
    frameStart := 66840 },
  { event := event66879
    frameStart := 66840 }
]

def eventLeaf4180 : Array AnnotatedEvent := #[
  { event := event66880
    frameStart := 66840 },
  { event := event66881
    frameStart := 66840 },
  { event := event66882
    frameStart := 66840 },
  { event := event66883
    frameStart := 66840 },
  { event := event66884
    frameStart := 66840 },
  { event := event66885
    frameStart := 66840 },
  { event := event66886
    frameStart := 66840 },
  { event := event66887
    frameStart := 66840 },
  { event := event66888
    frameStart := 66888 },
  { event := event66889
    frameStart := 66888 },
  { event := event66890
    frameStart := 66888 },
  { event := event66891
    frameStart := 66888 },
  { event := event66892
    frameStart := 66888 },
  { event := event66893
    frameStart := 66888 },
  { event := event66894
    frameStart := 66888 },
  { event := event66895
    frameStart := 66888 }
]

def eventLeaf4181 : Array AnnotatedEvent := #[
  { event := event66896
    frameStart := 66888 },
  { event := event66897
    frameStart := 66888 },
  { event := event66898
    frameStart := 66888 },
  { event := event66899
    frameStart := 66888 },
  { event := event66900
    frameStart := 66888 },
  { event := event66901
    frameStart := 66888 },
  { event := event66902
    frameStart := 66888 },
  { event := event66903
    frameStart := 66888 },
  { event := event66904
    frameStart := 66888 },
  { event := event66905
    frameStart := 66888 },
  { event := event66906
    frameStart := 66888 },
  { event := event66907
    frameStart := 66888 },
  { event := event66908
    frameStart := 66888 },
  { event := event66909
    frameStart := 66888 },
  { event := event66910
    frameStart := 66888 },
  { event := event66911
    frameStart := 66888 }
]

def eventLeaf4182 : Array AnnotatedEvent := #[
  { event := event66912
    frameStart := 66888 },
  { event := event66913
    frameStart := 66888 },
  { event := event66914
    frameStart := 66888 },
  { event := event66915
    frameStart := 66888 },
  { event := event66916
    frameStart := 66888 },
  { event := event66917
    frameStart := 66888 },
  { event := event66918
    frameStart := 66888 },
  { event := event66919
    frameStart := 66888 },
  { event := event66920
    frameStart := 66888 },
  { event := event66921
    frameStart := 66888 },
  { event := event66922
    frameStart := 66888 },
  { event := event66923
    frameStart := 66888 },
  { event := event66924
    frameStart := 66888 },
  { event := event66925
    frameStart := 66888 },
  { event := event66926
    frameStart := 66888 },
  { event := event66927
    frameStart := 66888 }
]

def eventLeaf4183 : Array AnnotatedEvent := #[
  { event := event66928
    frameStart := 66888 },
  { event := event66929
    frameStart := 66888 },
  { event := event66930
    frameStart := 66888 },
  { event := event66931
    frameStart := 66888 },
  { event := event66932
    frameStart := 66888 },
  { event := event66933
    frameStart := 66888 },
  { event := event66934
    frameStart := 66888 },
  { event := event66935
    frameStart := 66888 },
  { event := event66936
    frameStart := 66888 },
  { event := event66937
    frameStart := 66888 },
  { event := event66938
    frameStart := 66888 },
  { event := event66939
    frameStart := 66888 },
  { event := event66940
    frameStart := 66888 },
  { event := event66941
    frameStart := 66888 },
  { event := event66942
    frameStart := 66888 },
  { event := event66943
    frameStart := 66888 }
]

def eventLeaf4184 : Array AnnotatedEvent := #[
  { event := event66944
    frameStart := 66888 },
  { event := event66945
    frameStart := 66888 },
  { event := event66946
    frameStart := 66888 },
  { event := event66947
    frameStart := 66888 },
  { event := event66948
    frameStart := 66888 },
  { event := event66949
    frameStart := 66888 },
  { event := event66950
    frameStart := 66888 },
  { event := event66951
    frameStart := 66888 },
  { event := event66952
    frameStart := 66888 },
  { event := event66953
    frameStart := 66888 },
  { event := event66954
    frameStart := 66888 },
  { event := event66955
    frameStart := 66888 },
  { event := event66956
    frameStart := 66888 },
  { event := event66957
    frameStart := 66888 },
  { event := event66958
    frameStart := 66888 },
  { event := event66959
    frameStart := 66888 }
]

def eventLeaf4185 : Array AnnotatedEvent := #[
  { event := event66960
    frameStart := 66888 },
  { event := event66961
    frameStart := 66888 },
  { event := event66962
    frameStart := 66888 },
  { event := event66963
    frameStart := 66888 },
  { event := event66964
    frameStart := 66888 },
  { event := event66965
    frameStart := 66888 },
  { event := event66966
    frameStart := 66888 },
  { event := event66967
    frameStart := 66888 },
  { event := event66968
    frameStart := 66888 },
  { event := event66969
    frameStart := 66888 },
  { event := event66970
    frameStart := 66888 },
  { event := event66971
    frameStart := 66888 },
  { event := event66972
    frameStart := 66888 },
  { event := event66973
    frameStart := 66888 },
  { event := event66974
    frameStart := 66888 },
  { event := event66975
    frameStart := 66888 }
]

def eventLeaf4186 : Array AnnotatedEvent := #[
  { event := event66976
    frameStart := 66888 },
  { event := event66977
    frameStart := 66888 },
  { event := event66978
    frameStart := 66888 },
  { event := event66979
    frameStart := 66888 },
  { event := event66980
    frameStart := 66888 },
  { event := event66981
    frameStart := 66888 },
  { event := event66982
    frameStart := 66888 },
  { event := event66983
    frameStart := 66888 },
  { event := event66984
    frameStart := 66888 },
  { event := event66985
    frameStart := 66888 },
  { event := event66986
    frameStart := 66888 },
  { event := event66987
    frameStart := 66888 },
  { event := event66988
    frameStart := 66888 },
  { event := event66989
    frameStart := 66888 },
  { event := event66990
    frameStart := 66888 },
  { event := event66991
    frameStart := 66888 }
]

def eventLeaf4187 : Array AnnotatedEvent := #[
  { event := event66992
    frameStart := 66888 },
  { event := event66993
    frameStart := 66888 },
  { event := event66994
    frameStart := 66888 },
  { event := event66995
    frameStart := 66888 },
  { event := event66996
    frameStart := 66888 },
  { event := event66997
    frameStart := 66888 },
  { event := event66998
    frameStart := 66888 },
  { event := event66999
    frameStart := 66888 },
  { event := event67000
    frameStart := 66888 },
  { event := event67001
    frameStart := 66888 },
  { event := event67002
    frameStart := 66888 },
  { event := event67003
    frameStart := 66888 },
  { event := event67004
    frameStart := 66888 },
  { event := event67005
    frameStart := 66888 },
  { event := event67006
    frameStart := 0 },
  { event := event67007
    frameStart := 0 }
]

def eventLeaf4188 : Array AnnotatedEvent := #[
  { event := event67008
    frameStart := 0 },
  { event := event67009
    frameStart := 0 },
  { event := event67010
    frameStart := 0 },
  { event := event67011
    frameStart := 0 },
  { event := event67012
    frameStart := 0 },
  { event := event67013
    frameStart := 0 },
  { event := event67014
    frameStart := 0 },
  { event := event67015
    frameStart := 0 },
  { event := event67016
    frameStart := 0 },
  { event := event67017
    frameStart := 0 },
  { event := event67018
    frameStart := 0 },
  { event := event67019
    frameStart := 0 },
  { event := event67020
    frameStart := 0 },
  { event := event67021
    frameStart := 0 },
  { event := event67022
    frameStart := 0 },
  { event := event67023
    frameStart := 0 }
]

def eventLeaf4189 : Array AnnotatedEvent := #[
  { event := event67024
    frameStart := 0 },
  { event := event67025
    frameStart := 0 },
  { event := event67026
    frameStart := 0 },
  { event := event67027
    frameStart := 0 },
  { event := event67028
    frameStart := 0 },
  { event := event67029
    frameStart := 0 },
  { event := event67030
    frameStart := 0 },
  { event := event67031
    frameStart := 0 },
  { event := event67032
    frameStart := 0 },
  { event := event67033
    frameStart := 0 },
  { event := event67034
    frameStart := 0 },
  { event := event67035
    frameStart := 0 },
  { event := event67036
    frameStart := 0 },
  { event := event67037
    frameStart := 0 },
  { event := event67038
    frameStart := 0 },
  { event := event67039
    frameStart := 0 }
]

def eventLeaf4190 : Array AnnotatedEvent := #[
  { event := event67040
    frameStart := 0 },
  { event := event67041
    frameStart := 0 },
  { event := event67042
    frameStart := 0 },
  { event := event67043
    frameStart := 67043 },
  { event := event67044
    frameStart := 67043 },
  { event := event67045
    frameStart := 67043 },
  { event := event67046
    frameStart := 67043 },
  { event := event67047
    frameStart := 67043 },
  { event := event67048
    frameStart := 67043 },
  { event := event67049
    frameStart := 67043 },
  { event := event67050
    frameStart := 67043 },
  { event := event67051
    frameStart := 67043 },
  { event := event67052
    frameStart := 67043 },
  { event := event67053
    frameStart := 67043 },
  { event := event67054
    frameStart := 67043 },
  { event := event67055
    frameStart := 67043 }
]

def eventLeaf4191 : Array AnnotatedEvent := #[
  { event := event67056
    frameStart := 67043 },
  { event := event67057
    frameStart := 67043 },
  { event := event67058
    frameStart := 67043 },
  { event := event67059
    frameStart := 67043 },
  { event := event67060
    frameStart := 67043 },
  { event := event67061
    frameStart := 67043 },
  { event := event67062
    frameStart := 67043 },
  { event := event67063
    frameStart := 67043 },
  { event := event67064
    frameStart := 67043 },
  { event := event67065
    frameStart := 67043 },
  { event := event67066
    frameStart := 67043 },
  { event := event67067
    frameStart := 67043 },
  { event := event67068
    frameStart := 67043 },
  { event := event67069
    frameStart := 67043 },
  { event := event67070
    frameStart := 67043 },
  { event := event67071
    frameStart := 67043 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events261
