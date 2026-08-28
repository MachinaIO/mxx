import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events265

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event67840 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51899⟩⟩) 0 ⟨10792⟩ 61370

def event67841 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51899⟩⟩) 1 ⟨51898⟩ 67839

def event67842 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51899⟩⟩) (.product (.predecessor 0 67840 .coefficient) (.predecessor 1 67841 .coefficient) (⟨false, false, none, none, none⟩))

def event67843 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51899⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨51896⟩⟩]⟩) [⟨.result 67835 .coefficient, false, none⟩])

def event67844 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51899⟩⟩) (.product (.result 61370 .summary) (.transfer 67843) (⟨false, false, none, none, none⟩))

def event67845 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51899⟩⟩, .operator (⟨61370, 0⟩, ⟨67839, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51896⟩⟩]⟩, (1)⟩)

def event67846 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨51897⟩⟩)

def event67847 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event67848 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event67849 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.authority (.operator))

def event67850 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.finite 16)

def event67851 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event67852 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event67853 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event67854 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event67855 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 67854

def event67856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 67852

def event67857 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 67855 .coefficient) (.value (.predecessor 1 67856 .coefficient)))

def event67858 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event67859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 0 ⟨392⟩ 67858

def event67860 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 1 ⟨10691⟩ 67850

def event67861 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.sum [.predecessor 0 67859 .coefficient, .predecessor 1 67860 .coefficient])

def event67862 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.finite 655356)

def event67863 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 0 ⟨10693⟩ 67862

def event67864 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 1 ⟨5426⟩ 67848

def event67865 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.identity (.predecessor 1 67864 .coefficient))

def event67866 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.finite 655360)

def event67867 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24614⟩⟩) 0 ⟨10749⟩ 67866

def event67868 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24614⟩⟩) (.authority (.programFamilyFact))

def exact67869RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24614⟩⟩], []⟩, (1)⟩]

theorem exact67869RawTermsValid :
    exact67869RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67869 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24614⟩⟩) exact67869RawTerms (.finite 10) 67868 .exactZero (none)

def event67870 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50734⟩⟩) 0 ⟨10749⟩ 67866

def event67871 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50734⟩⟩) (.authority (.programFamilyFact))

def exact67872RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50734⟩⟩], []⟩, (1)⟩]

theorem exact67872RawTermsValid :
    exact67872RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67872 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50734⟩⟩) exact67872RawTerms (.finite 10) 67871 .exactZero (none)

def event67873 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50735⟩⟩) 0 ⟨50734⟩ 67872

def event67874 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50735⟩⟩) 1 ⟨24614⟩ 67869

def event67875 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50735⟩⟩) (.product (.predecessor 0 67873 .coefficient) (.predecessor 1 67874 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event67876 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50735⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], []⟩) [⟨.result 67872 .coefficient, true, some 1⟩, ⟨.result 67869 .coefficient, true, some 1⟩])

def event67877 : Event := .survivorFold (1) 67876

def exact67878RawTerms : List Term := []

theorem exact67878RawTermsValid :
    exact67878RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67878 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50735⟩⟩) exact67878RawTerms (.finite 100) 67875 (.finite 100) (some (67876))

def event67879 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50736⟩⟩) 0 ⟨50735⟩ 67878

def event67880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50736⟩⟩) (.identity (.predecessor 0 67879 .coefficient))

def event67881 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50736⟩⟩) (.finite 100)

def event67882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50944⟩⟩) 0 ⟨50736⟩ 67881

def event67883 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50944⟩⟩) (.authority (.programFamilyFact))

def exact67884RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], []⟩, (1)⟩]

theorem exact67884RawTermsValid :
    exact67884RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67884 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50944⟩⟩) exact67884RawTerms (.finite 10) 67883 .exactZero (none)

def event67885 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50945⟩⟩) 0 ⟨50944⟩ 67884

def event67886 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50945⟩⟩) (.identity (.predecessor 0 67885 .coefficient))

def event67887 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50945⟩⟩) (.finite 10)

def event67888 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51896⟩⟩) 0 ⟨50945⟩ 67887

def event67889 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51896⟩⟩) (.authority (.relationPreimageSource ⟨65⟩))

def exact67890RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨51896⟩⟩]⟩, (1)⟩]

theorem exact67890RawTermsValid :
    exact67890RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67890 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51896⟩⟩) exact67890RawTerms (.finite 5647228698) 67889 .exactZero (none)

def event67891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact67892RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact67892RawTermsValid :
    exact67892RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67892 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact67892RawTerms .large 67891 .exactZero (none)

def event67893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51897⟩⟩) 0 ⟨35⟩ 67892

def event67894 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51897⟩⟩) 1 ⟨51896⟩ 67890

def event67895 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51897⟩⟩) (.product (.predecessor 0 67893 .coefficient) (.predecessor 1 67894 .coefficient) (⟨false, false, none, none, none⟩))

def event67896 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51897⟩⟩, .operator (⟨67892, 0⟩, ⟨67890, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51896⟩⟩]⟩, (1)⟩)

def exact67897RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51896⟩⟩]⟩, (1)⟩]

theorem exact67897RawTermsValid :
    exact67897RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67897 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51897⟩⟩) exact67897RawTerms .large 67895 .exactZero (none)

def event67898 : Event := .preFoldPolynomial 67897 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51896⟩⟩]⟩, (1)⟩] .exactZero none

def exact67899RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51896⟩⟩]⟩, (1)⟩]

def event67899 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨51897⟩⟩) 67898 exact67899RawTerms .large 67895 .exactZero (none)

def event67900 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨53174⟩⟩)

def event67901 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event67902 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event67903 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.authority (.operator))

def event67904 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10691⟩⟩) (.finite 16)

def event67905 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event67906 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event67907 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event67908 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event67909 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 67908

def event67910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 67906

def event67911 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 67909 .coefficient) (.value (.predecessor 1 67910 .coefficient)))

def event67912 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event67913 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 0 ⟨392⟩ 67912

def event67914 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10693⟩⟩) 1 ⟨10691⟩ 67904

def event67915 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.sum [.predecessor 0 67913 .coefficient, .predecessor 1 67914 .coefficient])

def event67916 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10693⟩⟩) (.finite 655356)

def event67917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 0 ⟨10693⟩ 67916

def event67918 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10749⟩⟩) 1 ⟨5426⟩ 67902

def event67919 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.identity (.predecessor 1 67918 .coefficient))

def event67920 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10749⟩⟩) (.finite 655360)

def event67921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24614⟩⟩) 0 ⟨10749⟩ 67920

def event67922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24614⟩⟩) (.authority (.programFamilyFact))

def exact67923RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24614⟩⟩], []⟩, (1)⟩]

theorem exact67923RawTermsValid :
    exact67923RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67923 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24614⟩⟩) exact67923RawTerms (.finite 10) 67922 .exactZero (none)

def event67924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50734⟩⟩) 0 ⟨10749⟩ 67920

def event67925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50734⟩⟩) (.authority (.programFamilyFact))

def exact67926RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50734⟩⟩], []⟩, (1)⟩]

theorem exact67926RawTermsValid :
    exact67926RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67926 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50734⟩⟩) exact67926RawTerms (.finite 10) 67925 .exactZero (none)

def event67927 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50735⟩⟩) 0 ⟨50734⟩ 67926

def event67928 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50735⟩⟩) 1 ⟨24614⟩ 67923

def event67929 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50735⟩⟩) (.product (.predecessor 0 67927 .coefficient) (.predecessor 1 67928 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event67930 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50735⟩⟩, .operator (⟨67926, 0⟩, ⟨67923, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], []⟩, (1)⟩)

def exact67931RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24614⟩⟩, ⟨.program ⟨257⟩, ⟨50734⟩⟩], []⟩, (1)⟩]

theorem exact67931RawTermsValid :
    exact67931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67931 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50735⟩⟩) exact67931RawTerms (.finite 100) 67929 .exactZero (none)

def event67932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50736⟩⟩) 0 ⟨50735⟩ 67931

def event67933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50736⟩⟩) (.identity (.predecessor 0 67932 .coefficient))

def event67934 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50736⟩⟩) (.finite 100)

def event67935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50944⟩⟩) 0 ⟨50736⟩ 67934

def event67936 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50944⟩⟩) (.authority (.programFamilyFact))

def exact67937RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], []⟩, (1)⟩]

theorem exact67937RawTermsValid :
    exact67937RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67937 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50944⟩⟩) exact67937RawTerms (.finite 10) 67936 .exactZero (none)

def event67938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50945⟩⟩) 0 ⟨50944⟩ 67937

def event67939 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50945⟩⟩) (.identity (.predecessor 0 67938 .coefficient))

def event67940 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨50945⟩⟩) (.finite 10)

def event67941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52222⟩⟩) 0 ⟨50945⟩ 67940

def event67942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52222⟩⟩) (.authority (.programFamilyFact))

def event67943 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52222⟩⟩) (.finite 3720)

def event67944 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event67945 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52224⟩⟩) 0 ⟨7177⟩ 67944

def event67946 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52224⟩⟩) 1 ⟨52222⟩ 67943

def event67947 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52224⟩⟩) (.authority (.operator))

def exact67948RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨52224⟩⟩]⟩, (1)⟩]

theorem exact67948RawTermsValid :
    exact67948RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67948 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52224⟩⟩) exact67948RawTerms .large 67947 .exactZero (none)

def event67949 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53169⟩⟩) 0 ⟨52224⟩ 67948

def event67950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53169⟩⟩) (.authority (.operator))

def exact67951RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨53169⟩⟩]⟩, (1)⟩]

theorem exact67951RawTermsValid :
    exact67951RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67951 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53169⟩⟩) exact67951RawTerms (.finite 8192) 67950 .exactZero (none)

def event67952 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event67953 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event67954 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52394⟩⟩) 0 ⟨50945⟩ 67940

def event67955 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52394⟩⟩) 1 ⟨136⟩ 67953

def event67956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52394⟩⟩) (.sum [.predecessor 0 67954 .coefficient, .predecessor 1 67955 .coefficient])

def event67957 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨52394⟩⟩) (.finite 10)

def event67958 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52395⟩⟩) 0 ⟨52394⟩ 67957

def event67959 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52395⟩⟩) (.identity (.predecessor 0 67958 .coefficient))

def exact67960RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], []⟩, (1)⟩]

theorem exact67960RawTermsValid :
    exact67960RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67960 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52395⟩⟩) exact67960RawTerms (.finite 10) 67959 .exactZero (none)

def event67961 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact67962RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact67962RawTermsValid :
    exact67962RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67962 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact67962RawTerms .large 67961 .exactZero (none)

def event67963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52396⟩⟩) 0 ⟨6908⟩ 67962

def event67964 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52396⟩⟩) 1 ⟨52395⟩ 67960

def event67965 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52396⟩⟩) (.product (.predecessor 0 67963 .coefficient) (.predecessor 1 67964 .coefficient) (⟨false, false, none, none, none⟩))

def event67966 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨52396⟩⟩, .operator (⟨67962, 0⟩, ⟨67960, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact67967RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact67967RawTermsValid :
    exact67967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67967 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52396⟩⟩) exact67967RawTerms .large 67965 .exactZero (none)

def event67968 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7183⟩⟩) 0 ⟨7177⟩ 67944

def event67969 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7183⟩⟩) (.authority (.operator))

def exact67970RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩]

theorem exact67970RawTermsValid :
    exact67970RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67970 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7183⟩⟩) exact67970RawTerms .large 67969 .exactZero (none)

def event67971 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52397⟩⟩) 0 ⟨7183⟩ 67970

def event67972 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52397⟩⟩) 1 ⟨52396⟩ 67967

def event67973 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52397⟩⟩) (.sum [.predecessor 0 67971 .coefficient, .predecessor 1 67972 .coefficient])

def exact67974RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact67974RawTermsValid :
    exact67974RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67974 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52397⟩⟩) exact67974RawTerms .large 67973 .exactZero (none)

def event67975 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53170⟩⟩) 0 ⟨52397⟩ 67974

def event67976 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53170⟩⟩) 1 ⟨53169⟩ 67951

def event67977 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53170⟩⟩) (.product (.predecessor 0 67975 .coefficient) (.predecessor 1 67976 .coefficient) (⟨false, false, none, none, none⟩))

def event67978 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53170⟩⟩, .operator (⟨67974, 0⟩, ⟨67951, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53169⟩⟩]⟩, (1)⟩)

def event67979 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53170⟩⟩, .operator (⟨67974, 1⟩, ⟨67951, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53169⟩⟩]⟩, (-1)⟩)

def event67980 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨53170⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53169⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨53169⟩⟩) ⟨52224⟩ 67948)

def event67981 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53170⟩⟩, .relation 67980 0, ⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨52224⟩⟩]⟩, (-1)⟩)

def exact67982RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨52224⟩⟩]⟩, (-1)⟩]

theorem exact67982RawTermsValid :
    exact67982RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67982 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53170⟩⟩) exact67982RawTerms .large 67977 .exactZero (none)

def event67983 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51294⟩⟩) 0 ⟨50945⟩ 67940

def event67984 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51294⟩⟩) (.authority (.programFamilyFact))

def exact67985RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], []⟩, (1)⟩]

theorem exact67985RawTermsValid :
    exact67985RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67985 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51294⟩⟩) exact67985RawTerms (.finite 58) 67984 .exactZero (none)

def event67986 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51296⟩⟩) 0 ⟨6908⟩ 67962

def event67987 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51296⟩⟩) 1 ⟨51294⟩ 67985

def event67988 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51296⟩⟩) (.product (.predecessor 0 67986 .coefficient) (.predecessor 1 67987 .coefficient) (⟨false, true, none, none, some 1⟩))

def event67989 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51296⟩⟩, .operator (⟨67962, 0⟩, ⟨67985, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact67990RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact67990RawTermsValid :
    exact67990RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67990 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51296⟩⟩) exact67990RawTerms .large 67988 .exactZero (none)

def event67991 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7206⟩⟩) 0 ⟨7177⟩ 67944

def event67992 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7206⟩⟩) (.authority (.operator))

def exact67993RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩]

theorem exact67993RawTermsValid :
    exact67993RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67993 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7206⟩⟩) exact67993RawTerms .large 67992 .exactZero (none)

def event67994 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51297⟩⟩) 0 ⟨7206⟩ 67993

def event67995 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51297⟩⟩) 1 ⟨51296⟩ 67990

def event67996 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51297⟩⟩) (.sum [.predecessor 0 67994 .coefficient, .predecessor 1 67995 .coefficient])

def exact67997RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact67997RawTermsValid :
    exact67997RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67997 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51297⟩⟩) exact67997RawTerms .large 67996 .exactZero (none)

def event67998 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53174⟩⟩) 0 ⟨51297⟩ 67997

def event67999 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53174⟩⟩) 1 ⟨53170⟩ 67982

def event68000 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53174⟩⟩) (.sum [.predecessor 0 67998 .coefficient, .predecessor 1 67999 .coefficient])

def exact68001RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53169⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨52224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68001RawTermsValid :
    exact68001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68001 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53174⟩⟩) exact68001RawTerms .large 68000 .exactZero (none)

def event68002 : Event := .preFoldPolynomial 68001 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53169⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨52224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact68003RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53169⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨52224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event68003 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨53174⟩⟩) 68002 exact68003RawTerms .large 68000 .exactZero (none)

def event68004 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨50945⟩⟩) ⟨⟨85⟩, ⟨65⟩, ⟨135⟩⟩ ⟨67846, 68004⟩

def event68005 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨51899⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51896⟩⟩]⟩) (1) 0 2 (.universal 68004 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51896⟩⟩]⟩) (none) 68003)

def event68006 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51899⟩⟩, .relation 68005 1, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩)

def event68007 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51899⟩⟩, .relation 68005 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53169⟩⟩]⟩, (-1)⟩)

def event68008 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51899⟩⟩, .relation 68005 2, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨52224⟩⟩]⟩, (1)⟩)

def event68009 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51899⟩⟩, .relation 68005 3, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact68010RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53169⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨52224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68010RawTermsValid :
    exact68010RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68010 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51899⟩⟩) exact68010RawTerms .large 67842 (.finite 202072841853861888) (some (67844))

def event68011 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53172⟩⟩) 0 ⟨51899⟩ 68010

def event68012 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53172⟩⟩) 1 ⟨53171⟩ 67832

def event68013 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53172⟩⟩) (.sum [.predecessor 0 68011 .coefficient, .predecessor 1 68012 .coefficient])

def event68014 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53172⟩⟩, .operator (⟨68010, 0⟩, ⟨67832, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53169⟩⟩]⟩, (1)⟩)

def event68015 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53172⟩⟩, .operator (⟨68010, 2⟩, ⟨67832, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨52224⟩⟩]⟩, (-1)⟩)

def event68016 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53172⟩⟩) (.sum [.result 68010 .summary, .result 67832 .summary])

def exact68017RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68017RawTermsValid :
    exact68017RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68017 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53172⟩⟩) exact68017RawTerms .large 68013 (.finite 32189593014266456398474184491008) (some (68016))

def event68018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33162⟩⟩) 0 ⟨31885⟩ 2677

def event68019 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33162⟩⟩) (.authority (.programFamilyFact))

def event68020 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨33162⟩⟩) (.finite 3720)

def event68021 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33164⟩⟩) 0 ⟨7177⟩ 15500

def event68022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33164⟩⟩) 1 ⟨33162⟩ 68020

def event68023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33164⟩⟩) (.authority (.operator))

def exact68024RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨33164⟩⟩]⟩, (1)⟩]

theorem exact68024RawTermsValid :
    exact68024RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68024 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33164⟩⟩) exact68024RawTerms .large 68023 .exactZero (none)

def event68025 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34109⟩⟩) 0 ⟨33164⟩ 68024

def event68026 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34109⟩⟩) (.authority (.operator))

def exact68027RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨34109⟩⟩]⟩, (1)⟩]

theorem exact68027RawTermsValid :
    exact68027RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68027 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34109⟩⟩) exact68027RawTerms (.finite 8192) 68026 .exactZero (none)

def event68028 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32990⟩⟩) 0 ⟨31676⟩ 2671

def event68029 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32990⟩⟩) (.authority (.programFamilyFact))

def event68030 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨32990⟩⟩) (.finite 3720)

def event68031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32991⟩⟩) 0 ⟨7177⟩ 15500

def event68032 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32991⟩⟩) 1 ⟨32990⟩ 68030

def event68033 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32991⟩⟩) (.authority (.operator))

def exact68034RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32991⟩⟩]⟩, (1)⟩]

theorem exact68034RawTermsValid :
    exact68034RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68034 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32991⟩⟩) exact68034RawTerms .large 68033 .exactZero (none)

def event68035 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33536⟩⟩) 0 ⟨32991⟩ 68034

def event68036 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33536⟩⟩) (.authority (.operator))

def exact68037RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨33536⟩⟩]⟩, (1)⟩]

theorem exact68037RawTermsValid :
    exact68037RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68037 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33536⟩⟩) exact68037RawTerms (.finite 8192) 68036 .exactZero (none)

def event68038 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24375⟩⟩) 0 ⟨24374⟩ 2660

def event68039 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24375⟩⟩) 1 ⟨10752⟩ 61278

def event68040 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24375⟩⟩) (.tensor (.predecessor 0 68038 .coefficient) (.predecessor 1 68039 .coefficient) true false)

def event68041 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨24375⟩⟩, .operator (⟨2660, 0⟩, ⟨61278, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24374⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact68042RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24374⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact68042RawTermsValid :
    exact68042RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68042 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24375⟩⟩) exact68042RawTerms .large 68040 .exactZero (none)

def event68043 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10789⟩⟩) 0 ⟨10751⟩ 61148

def event68044 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10789⟩⟩) 1 ⟨7307⟩ 24094

def event68045 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10789⟩⟩) (.product (.predecessor 0 68043 .coefficient) (.predecessor 1 68044 .coefficient) (⟨false, false, none, none, none⟩))

def event68046 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10789⟩⟩, .operator (⟨61148, 0⟩, ⟨24094, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩)

def exact68047RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩]

theorem exact68047RawTermsValid :
    exact68047RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68047 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10789⟩⟩) exact68047RawTerms .large 68045 .exactZero (none)

def event68048 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24376⟩⟩) 0 ⟨10789⟩ 68047

def event68049 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24376⟩⟩) 1 ⟨24375⟩ 68042

def event68050 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24376⟩⟩) (.sum [.predecessor 0 68048 .coefficient, .predecessor 1 68049 .coefficient])

def exact68051RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24374⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68051RawTermsValid :
    exact68051RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68051 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24376⟩⟩) exact68051RawTerms .large 68050 .exactZero (none)

def event68052 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24377⟩⟩) 0 ⟨24376⟩ 68051

def event68053 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24377⟩⟩) 1 ⟨133⟩ 24086

def event68054 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24377⟩⟩) (.sum [.predecessor 0 68052 .coefficient, .predecessor 1 68053 .coefficient])

def event68055 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24377⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨133⟩⟩]⟩) [⟨.result 24086 .coefficient, false, none⟩])

def event68056 : Event := .survivorFold (1) 68055

def exact68057RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24374⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68057RawTermsValid :
    exact68057RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68057 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24377⟩⟩) exact68057RawTerms .large 68054 (.finite 26) (some (68055))

def event68058 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31677⟩⟩) 0 ⟨24377⟩ 68057

def event68059 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31677⟩⟩) 1 ⟨31674⟩ 2663

def event68060 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31677⟩⟩) (.product (.predecessor 0 68058 .coefficient) (.predecessor 1 68059 .coefficient) (⟨false, true, none, none, some 1⟩))

def event68061 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31677⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨31674⟩⟩], []⟩) [⟨.result 2663 .coefficient, true, some 1⟩])

def event68062 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31677⟩⟩) (.product (.result 68057 .summary) (.transfer 68061) (⟨false, false, none, none, none⟩))

def event68063 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31677⟩⟩, .operator (⟨68057, 1⟩, ⟨2663, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event68064 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31677⟩⟩, .operator (⟨68057, 0⟩, ⟨2663, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩)

def exact68065RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨24374⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (1)⟩]

theorem exact68065RawTermsValid :
    exact68065RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68065 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31677⟩⟩) exact68065RawTerms .large 68060 (.finite 5111808) (some (68062))

def event68066 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31678⟩⟩) 0 ⟨31674⟩ 2663

def event68067 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31678⟩⟩) 1 ⟨10752⟩ 61278

def event68068 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31678⟩⟩) (.tensor (.predecessor 0 68066 .coefficient) (.predecessor 1 68067 .coefficient) true false)

def event68069 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31678⟩⟩, .operator (⟨2663, 0⟩, ⟨61278, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact68070RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact68070RawTermsValid :
    exact68070RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68070 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31678⟩⟩) exact68070RawTerms .large 68068 .exactZero (none)

def event68071 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10769⟩⟩) 0 ⟨10751⟩ 61148

def event68072 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10769⟩⟩) 1 ⟨7287⟩ 24135

def event68073 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10769⟩⟩) (.product (.predecessor 0 68071 .coefficient) (.predecessor 1 68072 .coefficient) (⟨false, false, none, none, none⟩))

def event68074 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10769⟩⟩, .operator (⟨61148, 0⟩, ⟨24135, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩]⟩, (1)⟩)

def exact68075RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩]⟩, (1)⟩]

theorem exact68075RawTermsValid :
    exact68075RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68075 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10769⟩⟩) exact68075RawTerms .large 68073 .exactZero (none)

def event68076 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31679⟩⟩) 0 ⟨10769⟩ 68075

def event68077 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31679⟩⟩) 1 ⟨31678⟩ 68070

def event68078 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31679⟩⟩) (.sum [.predecessor 0 68076 .coefficient, .predecessor 1 68077 .coefficient])

def exact68079RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68079RawTermsValid :
    exact68079RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68079 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31679⟩⟩) exact68079RawTerms .large 68078 .exactZero (none)

def event68080 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31680⟩⟩) 0 ⟨31679⟩ 68079

def event68081 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31680⟩⟩) 1 ⟨113⟩ 24127

def event68082 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31680⟩⟩) (.sum [.predecessor 0 68080 .coefficient, .predecessor 1 68081 .coefficient])

def event68083 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31680⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨113⟩⟩]⟩) [⟨.result 24127 .coefficient, false, none⟩])

def event68084 : Event := .survivorFold (1) 68083

def exact68085RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact68085RawTermsValid :
    exact68085RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68085 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31680⟩⟩) exact68085RawTerms .large 68082 (.finite 26) (some (68083))

def event68086 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31681⟩⟩) 0 ⟨31680⟩ 68085

def event68087 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31681⟩⟩) 1 ⟨9578⟩ 24124

def event68088 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31681⟩⟩) (.product (.predecessor 0 68086 .coefficient) (.predecessor 1 68087 .coefficient) (⟨false, false, none, none, none⟩))

def event68089 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31681⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) [⟨.result 24120 .coefficient, false, none⟩])

def event68090 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31681⟩⟩) (.product (.result 68085 .summary) (.transfer 68089) (⟨false, false, none, none, none⟩))

def event68091 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31681⟩⟩, .operator (⟨68085, 1⟩, ⟨24124, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (-1)⟩)

def event68092 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨31681⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9577⟩⟩) ⟨7307⟩ 24094)

def event68093 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31681⟩⟩, .relation 68092 0, ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (-1)⟩)

def event68094 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31681⟩⟩, .operator (⟨68085, 0⟩, ⟨24124, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩)

def exact68095RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨31674⟩⟩], [⟨.program ⟨257⟩, ⟨7307⟩⟩]⟩, (-1)⟩]

theorem exact68095RawTermsValid :
    exact68095RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68095 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31681⟩⟩) exact68095RawTerms .large 68088 (.finite 279172874240) (some (68090))

def eventLeaf4240 : Array AnnotatedEvent := #[
  { event := event67840
    frameStart := 0 },
  { event := event67841
    frameStart := 0 },
  { event := event67842
    frameStart := 0 },
  { event := event67843
    frameStart := 0 },
  { event := event67844
    frameStart := 0 },
  { event := event67845
    frameStart := 0 },
  { event := event67846
    frameStart := 67846 },
  { event := event67847
    frameStart := 67846 },
  { event := event67848
    frameStart := 67846 },
  { event := event67849
    frameStart := 67846 },
  { event := event67850
    frameStart := 67846 },
  { event := event67851
    frameStart := 67846 },
  { event := event67852
    frameStart := 67846 },
  { event := event67853
    frameStart := 67846 },
  { event := event67854
    frameStart := 67846 },
  { event := event67855
    frameStart := 67846 }
]

def eventLeaf4241 : Array AnnotatedEvent := #[
  { event := event67856
    frameStart := 67846 },
  { event := event67857
    frameStart := 67846 },
  { event := event67858
    frameStart := 67846 },
  { event := event67859
    frameStart := 67846 },
  { event := event67860
    frameStart := 67846 },
  { event := event67861
    frameStart := 67846 },
  { event := event67862
    frameStart := 67846 },
  { event := event67863
    frameStart := 67846 },
  { event := event67864
    frameStart := 67846 },
  { event := event67865
    frameStart := 67846 },
  { event := event67866
    frameStart := 67846 },
  { event := event67867
    frameStart := 67846 },
  { event := event67868
    frameStart := 67846 },
  { event := event67869
    frameStart := 67846 },
  { event := event67870
    frameStart := 67846 },
  { event := event67871
    frameStart := 67846 }
]

def eventLeaf4242 : Array AnnotatedEvent := #[
  { event := event67872
    frameStart := 67846 },
  { event := event67873
    frameStart := 67846 },
  { event := event67874
    frameStart := 67846 },
  { event := event67875
    frameStart := 67846 },
  { event := event67876
    frameStart := 67846 },
  { event := event67877
    frameStart := 67846 },
  { event := event67878
    frameStart := 67846 },
  { event := event67879
    frameStart := 67846 },
  { event := event67880
    frameStart := 67846 },
  { event := event67881
    frameStart := 67846 },
  { event := event67882
    frameStart := 67846 },
  { event := event67883
    frameStart := 67846 },
  { event := event67884
    frameStart := 67846 },
  { event := event67885
    frameStart := 67846 },
  { event := event67886
    frameStart := 67846 },
  { event := event67887
    frameStart := 67846 }
]

def eventLeaf4243 : Array AnnotatedEvent := #[
  { event := event67888
    frameStart := 67846 },
  { event := event67889
    frameStart := 67846 },
  { event := event67890
    frameStart := 67846 },
  { event := event67891
    frameStart := 67846 },
  { event := event67892
    frameStart := 67846 },
  { event := event67893
    frameStart := 67846 },
  { event := event67894
    frameStart := 67846 },
  { event := event67895
    frameStart := 67846 },
  { event := event67896
    frameStart := 67846 },
  { event := event67897
    frameStart := 67846 },
  { event := event67898
    frameStart := 67846 },
  { event := event67899
    frameStart := 67846 },
  { event := event67900
    frameStart := 67900 },
  { event := event67901
    frameStart := 67900 },
  { event := event67902
    frameStart := 67900 },
  { event := event67903
    frameStart := 67900 }
]

def eventLeaf4244 : Array AnnotatedEvent := #[
  { event := event67904
    frameStart := 67900 },
  { event := event67905
    frameStart := 67900 },
  { event := event67906
    frameStart := 67900 },
  { event := event67907
    frameStart := 67900 },
  { event := event67908
    frameStart := 67900 },
  { event := event67909
    frameStart := 67900 },
  { event := event67910
    frameStart := 67900 },
  { event := event67911
    frameStart := 67900 },
  { event := event67912
    frameStart := 67900 },
  { event := event67913
    frameStart := 67900 },
  { event := event67914
    frameStart := 67900 },
  { event := event67915
    frameStart := 67900 },
  { event := event67916
    frameStart := 67900 },
  { event := event67917
    frameStart := 67900 },
  { event := event67918
    frameStart := 67900 },
  { event := event67919
    frameStart := 67900 }
]

def eventLeaf4245 : Array AnnotatedEvent := #[
  { event := event67920
    frameStart := 67900 },
  { event := event67921
    frameStart := 67900 },
  { event := event67922
    frameStart := 67900 },
  { event := event67923
    frameStart := 67900 },
  { event := event67924
    frameStart := 67900 },
  { event := event67925
    frameStart := 67900 },
  { event := event67926
    frameStart := 67900 },
  { event := event67927
    frameStart := 67900 },
  { event := event67928
    frameStart := 67900 },
  { event := event67929
    frameStart := 67900 },
  { event := event67930
    frameStart := 67900 },
  { event := event67931
    frameStart := 67900 },
  { event := event67932
    frameStart := 67900 },
  { event := event67933
    frameStart := 67900 },
  { event := event67934
    frameStart := 67900 },
  { event := event67935
    frameStart := 67900 }
]

def eventLeaf4246 : Array AnnotatedEvent := #[
  { event := event67936
    frameStart := 67900 },
  { event := event67937
    frameStart := 67900 },
  { event := event67938
    frameStart := 67900 },
  { event := event67939
    frameStart := 67900 },
  { event := event67940
    frameStart := 67900 },
  { event := event67941
    frameStart := 67900 },
  { event := event67942
    frameStart := 67900 },
  { event := event67943
    frameStart := 67900 },
  { event := event67944
    frameStart := 67900 },
  { event := event67945
    frameStart := 67900 },
  { event := event67946
    frameStart := 67900 },
  { event := event67947
    frameStart := 67900 },
  { event := event67948
    frameStart := 67900 },
  { event := event67949
    frameStart := 67900 },
  { event := event67950
    frameStart := 67900 },
  { event := event67951
    frameStart := 67900 }
]

def eventLeaf4247 : Array AnnotatedEvent := #[
  { event := event67952
    frameStart := 67900 },
  { event := event67953
    frameStart := 67900 },
  { event := event67954
    frameStart := 67900 },
  { event := event67955
    frameStart := 67900 },
  { event := event67956
    frameStart := 67900 },
  { event := event67957
    frameStart := 67900 },
  { event := event67958
    frameStart := 67900 },
  { event := event67959
    frameStart := 67900 },
  { event := event67960
    frameStart := 67900 },
  { event := event67961
    frameStart := 67900 },
  { event := event67962
    frameStart := 67900 },
  { event := event67963
    frameStart := 67900 },
  { event := event67964
    frameStart := 67900 },
  { event := event67965
    frameStart := 67900 },
  { event := event67966
    frameStart := 67900 },
  { event := event67967
    frameStart := 67900 }
]

def eventLeaf4248 : Array AnnotatedEvent := #[
  { event := event67968
    frameStart := 67900 },
  { event := event67969
    frameStart := 67900 },
  { event := event67970
    frameStart := 67900 },
  { event := event67971
    frameStart := 67900 },
  { event := event67972
    frameStart := 67900 },
  { event := event67973
    frameStart := 67900 },
  { event := event67974
    frameStart := 67900 },
  { event := event67975
    frameStart := 67900 },
  { event := event67976
    frameStart := 67900 },
  { event := event67977
    frameStart := 67900 },
  { event := event67978
    frameStart := 67900 },
  { event := event67979
    frameStart := 67900 },
  { event := event67980
    frameStart := 67900 },
  { event := event67981
    frameStart := 67900 },
  { event := event67982
    frameStart := 67900 },
  { event := event67983
    frameStart := 67900 }
]

def eventLeaf4249 : Array AnnotatedEvent := #[
  { event := event67984
    frameStart := 67900 },
  { event := event67985
    frameStart := 67900 },
  { event := event67986
    frameStart := 67900 },
  { event := event67987
    frameStart := 67900 },
  { event := event67988
    frameStart := 67900 },
  { event := event67989
    frameStart := 67900 },
  { event := event67990
    frameStart := 67900 },
  { event := event67991
    frameStart := 67900 },
  { event := event67992
    frameStart := 67900 },
  { event := event67993
    frameStart := 67900 },
  { event := event67994
    frameStart := 67900 },
  { event := event67995
    frameStart := 67900 },
  { event := event67996
    frameStart := 67900 },
  { event := event67997
    frameStart := 67900 },
  { event := event67998
    frameStart := 67900 },
  { event := event67999
    frameStart := 67900 }
]

def eventLeaf4250 : Array AnnotatedEvent := #[
  { event := event68000
    frameStart := 67900 },
  { event := event68001
    frameStart := 67900 },
  { event := event68002
    frameStart := 67900 },
  { event := event68003
    frameStart := 67900 },
  { event := event68004
    frameStart := 0 },
  { event := event68005
    frameStart := 0 },
  { event := event68006
    frameStart := 0 },
  { event := event68007
    frameStart := 0 },
  { event := event68008
    frameStart := 0 },
  { event := event68009
    frameStart := 0 },
  { event := event68010
    frameStart := 0 },
  { event := event68011
    frameStart := 0 },
  { event := event68012
    frameStart := 0 },
  { event := event68013
    frameStart := 0 },
  { event := event68014
    frameStart := 0 },
  { event := event68015
    frameStart := 0 }
]

def eventLeaf4251 : Array AnnotatedEvent := #[
  { event := event68016
    frameStart := 0 },
  { event := event68017
    frameStart := 0 },
  { event := event68018
    frameStart := 0 },
  { event := event68019
    frameStart := 0 },
  { event := event68020
    frameStart := 0 },
  { event := event68021
    frameStart := 0 },
  { event := event68022
    frameStart := 0 },
  { event := event68023
    frameStart := 0 },
  { event := event68024
    frameStart := 0 },
  { event := event68025
    frameStart := 0 },
  { event := event68026
    frameStart := 0 },
  { event := event68027
    frameStart := 0 },
  { event := event68028
    frameStart := 0 },
  { event := event68029
    frameStart := 0 },
  { event := event68030
    frameStart := 0 },
  { event := event68031
    frameStart := 0 }
]

def eventLeaf4252 : Array AnnotatedEvent := #[
  { event := event68032
    frameStart := 0 },
  { event := event68033
    frameStart := 0 },
  { event := event68034
    frameStart := 0 },
  { event := event68035
    frameStart := 0 },
  { event := event68036
    frameStart := 0 },
  { event := event68037
    frameStart := 0 },
  { event := event68038
    frameStart := 0 },
  { event := event68039
    frameStart := 0 },
  { event := event68040
    frameStart := 0 },
  { event := event68041
    frameStart := 0 },
  { event := event68042
    frameStart := 0 },
  { event := event68043
    frameStart := 0 },
  { event := event68044
    frameStart := 0 },
  { event := event68045
    frameStart := 0 },
  { event := event68046
    frameStart := 0 },
  { event := event68047
    frameStart := 0 }
]

def eventLeaf4253 : Array AnnotatedEvent := #[
  { event := event68048
    frameStart := 0 },
  { event := event68049
    frameStart := 0 },
  { event := event68050
    frameStart := 0 },
  { event := event68051
    frameStart := 0 },
  { event := event68052
    frameStart := 0 },
  { event := event68053
    frameStart := 0 },
  { event := event68054
    frameStart := 0 },
  { event := event68055
    frameStart := 0 },
  { event := event68056
    frameStart := 0 },
  { event := event68057
    frameStart := 0 },
  { event := event68058
    frameStart := 0 },
  { event := event68059
    frameStart := 0 },
  { event := event68060
    frameStart := 0 },
  { event := event68061
    frameStart := 0 },
  { event := event68062
    frameStart := 0 },
  { event := event68063
    frameStart := 0 }
]

def eventLeaf4254 : Array AnnotatedEvent := #[
  { event := event68064
    frameStart := 0 },
  { event := event68065
    frameStart := 0 },
  { event := event68066
    frameStart := 0 },
  { event := event68067
    frameStart := 0 },
  { event := event68068
    frameStart := 0 },
  { event := event68069
    frameStart := 0 },
  { event := event68070
    frameStart := 0 },
  { event := event68071
    frameStart := 0 },
  { event := event68072
    frameStart := 0 },
  { event := event68073
    frameStart := 0 },
  { event := event68074
    frameStart := 0 },
  { event := event68075
    frameStart := 0 },
  { event := event68076
    frameStart := 0 },
  { event := event68077
    frameStart := 0 },
  { event := event68078
    frameStart := 0 },
  { event := event68079
    frameStart := 0 }
]

def eventLeaf4255 : Array AnnotatedEvent := #[
  { event := event68080
    frameStart := 0 },
  { event := event68081
    frameStart := 0 },
  { event := event68082
    frameStart := 0 },
  { event := event68083
    frameStart := 0 },
  { event := event68084
    frameStart := 0 },
  { event := event68085
    frameStart := 0 },
  { event := event68086
    frameStart := 0 },
  { event := event68087
    frameStart := 0 },
  { event := event68088
    frameStart := 0 },
  { event := event68089
    frameStart := 0 },
  { event := event68090
    frameStart := 0 },
  { event := event68091
    frameStart := 0 },
  { event := event68092
    frameStart := 0 },
  { event := event68093
    frameStart := 0 },
  { event := event68094
    frameStart := 0 },
  { event := event68095
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events265
