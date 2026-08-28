import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events351

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event89856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15618⟩⟩) 0 ⟨10325⟩ 89855

def event89857 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15618⟩⟩) (.authority (.programFamilyFact))

def exact89858RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15618⟩⟩], []⟩, (1)⟩]

theorem exact89858RawTermsValid :
    exact89858RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89858 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15618⟩⟩) exact89858RawTerms (.finite 2) 89857 .exactZero (none)

def event89859 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12471⟩⟩) 0 ⟨10325⟩ 89855

def event89860 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12471⟩⟩) (.authority (.programFamilyFact))

def exact89861RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩], []⟩, (1)⟩]

theorem exact89861RawTermsValid :
    exact89861RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89861 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12471⟩⟩) exact89861RawTerms (.finite 2) 89860 .exactZero (none)

def event89862 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15619⟩⟩) 0 ⟨12471⟩ 89861

def event89863 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15619⟩⟩) 1 ⟨15618⟩ 89858

def event89864 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15619⟩⟩) (.product (.predecessor 0 89862 .coefficient) (.predecessor 1 89863 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event89865 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15619⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], []⟩) [⟨.result 89861 .coefficient, true, some 1⟩, ⟨.result 89858 .coefficient, true, some 1⟩])

def event89866 : Event := .survivorFold (1) 89865

def exact89867RawTerms : List Term := []

theorem exact89867RawTermsValid :
    exact89867RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89867 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15619⟩⟩) exact89867RawTerms (.finite 4) 89864 (.finite 4) (some (89865))

def event89868 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15620⟩⟩) 0 ⟨15619⟩ 89867

def event89869 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15620⟩⟩) (.identity (.predecessor 0 89868 .coefficient))

def event89870 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15620⟩⟩) (.finite 4)

def event89871 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15836⟩⟩) 0 ⟨15620⟩ 89870

def event89872 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15836⟩⟩) (.authority (.programFamilyFact))

def exact89873RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], []⟩, (1)⟩]

theorem exact89873RawTermsValid :
    exact89873RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89873 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15836⟩⟩) exact89873RawTerms (.finite 2) 89872 .exactZero (none)

def event89874 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15837⟩⟩) 0 ⟨15836⟩ 89873

def event89875 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15837⟩⟩) (.identity (.predecessor 0 89874 .coefficient))

def event89876 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15837⟩⟩) (.finite 2)

def event89877 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16712⟩⟩) 0 ⟨15837⟩ 89876

def event89878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16712⟩⟩) (.authority (.relationPreimageSource ⟨56⟩))

def exact89879RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16712⟩⟩]⟩, (1)⟩]

theorem exact89879RawTermsValid :
    exact89879RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89879 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16712⟩⟩) exact89879RawTerms (.finite 5647228698) 89878 .exactZero (none)

def event89880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact89881RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact89881RawTermsValid :
    exact89881RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89881 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact89881RawTerms .large 89880 .exactZero (none)

def event89882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16713⟩⟩) 0 ⟨35⟩ 89881

def event89883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16713⟩⟩) 1 ⟨16712⟩ 89879

def event89884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16713⟩⟩) (.product (.predecessor 0 89882 .coefficient) (.predecessor 1 89883 .coefficient) (⟨false, false, none, none, none⟩))

def event89885 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16713⟩⟩, .operator (⟨89881, 0⟩, ⟨89879, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16712⟩⟩]⟩, (1)⟩)

def exact89886RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16712⟩⟩]⟩, (1)⟩]

theorem exact89886RawTermsValid :
    exact89886RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89886 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16713⟩⟩) exact89886RawTerms .large 89884 .exactZero (none)

def event89887 : Event := .preFoldPolynomial 89886 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16712⟩⟩]⟩, (1)⟩] .exactZero none

def exact89888RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16712⟩⟩]⟩, (1)⟩]

def event89888 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨16713⟩⟩) 89887 exact89888RawTerms .large 89884 .exactZero (none)

def event89889 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨17928⟩⟩)

def event89890 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event89891 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event89892 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event89893 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event89894 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event89895 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event89896 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event89897 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event89898 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 89897

def event89899 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 89895

def event89900 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 89898 .coefficient) (.value (.predecessor 1 89899 .coefficient)))

def event89901 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event89902 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 89901

def event89903 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 89893

def event89904 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 89902 .coefficient, .predecessor 1 89903 .coefficient])

def event89905 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event89906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 89905

def event89907 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 89891

def event89908 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 89907 .coefficient))

def event89909 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event89910 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15618⟩⟩) 0 ⟨10325⟩ 89909

def event89911 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15618⟩⟩) (.authority (.programFamilyFact))

def exact89912RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15618⟩⟩], []⟩, (1)⟩]

theorem exact89912RawTermsValid :
    exact89912RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89912 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15618⟩⟩) exact89912RawTerms (.finite 2) 89911 .exactZero (none)

def event89913 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12471⟩⟩) 0 ⟨10325⟩ 89909

def event89914 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12471⟩⟩) (.authority (.programFamilyFact))

def exact89915RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩], []⟩, (1)⟩]

theorem exact89915RawTermsValid :
    exact89915RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89915 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12471⟩⟩) exact89915RawTerms (.finite 2) 89914 .exactZero (none)

def event89916 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15619⟩⟩) 0 ⟨12471⟩ 89915

def event89917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15619⟩⟩) 1 ⟨15618⟩ 89912

def event89918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15619⟩⟩) (.product (.predecessor 0 89916 .coefficient) (.predecessor 1 89917 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event89919 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15619⟩⟩, .operator (⟨89915, 0⟩, ⟨89912, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], []⟩, (1)⟩)

def exact89920RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12471⟩⟩, ⟨.program ⟨257⟩, ⟨15618⟩⟩], []⟩, (1)⟩]

theorem exact89920RawTermsValid :
    exact89920RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89920 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15619⟩⟩) exact89920RawTerms (.finite 4) 89918 .exactZero (none)

def event89921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15620⟩⟩) 0 ⟨15619⟩ 89920

def event89922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15620⟩⟩) (.identity (.predecessor 0 89921 .coefficient))

def event89923 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15620⟩⟩) (.finite 4)

def event89924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15836⟩⟩) 0 ⟨15620⟩ 89923

def event89925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15836⟩⟩) (.authority (.programFamilyFact))

def exact89926RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], []⟩, (1)⟩]

theorem exact89926RawTermsValid :
    exact89926RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89926 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15836⟩⟩) exact89926RawTerms (.finite 2) 89925 .exactZero (none)

def event89927 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15837⟩⟩) 0 ⟨15836⟩ 89926

def event89928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15837⟩⟩) (.identity (.predecessor 0 89927 .coefficient))

def event89929 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15837⟩⟩) (.finite 2)

def event89930 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17053⟩⟩) 0 ⟨15837⟩ 89929

def event89931 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17053⟩⟩) (.authority (.programFamilyFact))

def event89932 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17053⟩⟩) (.finite 3720)

def event89933 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event89934 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17054⟩⟩) 0 ⟨7177⟩ 89933

def event89935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17054⟩⟩) 1 ⟨17053⟩ 89932

def event89936 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17054⟩⟩) (.authority (.operator))

def exact89937RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17054⟩⟩]⟩, (1)⟩]

theorem exact89937RawTermsValid :
    exact89937RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89937 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17054⟩⟩) exact89937RawTerms .large 89936 .exactZero (none)

def event89938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17922⟩⟩) 0 ⟨17054⟩ 89937

def event89939 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17922⟩⟩) (.authority (.operator))

def exact89940RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17922⟩⟩]⟩, (1)⟩]

theorem exact89940RawTermsValid :
    exact89940RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89940 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17922⟩⟩) exact89940RawTerms (.finite 8192) 89939 .exactZero (none)

def event89941 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event89942 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event89943 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17230⟩⟩) 0 ⟨15837⟩ 89929

def event89944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17230⟩⟩) 1 ⟨136⟩ 89942

def event89945 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17230⟩⟩) (.sum [.predecessor 0 89943 .coefficient, .predecessor 1 89944 .coefficient])

def event89946 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17230⟩⟩) (.finite 2)

def event89947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17231⟩⟩) 0 ⟨17230⟩ 89946

def event89948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17231⟩⟩) (.identity (.predecessor 0 89947 .coefficient))

def exact89949RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], []⟩, (1)⟩]

theorem exact89949RawTermsValid :
    exact89949RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89949 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17231⟩⟩) exact89949RawTerms (.finite 2) 89948 .exactZero (none)

def event89950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact89951RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact89951RawTermsValid :
    exact89951RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89951 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact89951RawTerms .large 89950 .exactZero (none)

def event89952 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17232⟩⟩) 0 ⟨6908⟩ 89951

def event89953 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17232⟩⟩) 1 ⟨17231⟩ 89949

def event89954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17232⟩⟩) (.product (.predecessor 0 89952 .coefficient) (.predecessor 1 89953 .coefficient) (⟨false, false, none, none, none⟩))

def event89955 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17232⟩⟩, .operator (⟨89951, 0⟩, ⟨89949, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact89956RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact89956RawTermsValid :
    exact89956RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89956 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17232⟩⟩) exact89956RawTerms .large 89954 .exactZero (none)

def event89957 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 89933

def event89958 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact89959RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact89959RawTermsValid :
    exact89959RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89959 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact89959RawTerms .large 89958 .exactZero (none)

def event89960 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17233⟩⟩) 0 ⟨7179⟩ 89959

def event89961 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17233⟩⟩) 1 ⟨17232⟩ 89956

def event89962 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17233⟩⟩) (.sum [.predecessor 0 89960 .coefficient, .predecessor 1 89961 .coefficient])

def exact89963RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact89963RawTermsValid :
    exact89963RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89963 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17233⟩⟩) exact89963RawTerms .large 89962 .exactZero (none)

def event89964 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17923⟩⟩) 0 ⟨17233⟩ 89963

def event89965 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17923⟩⟩) 1 ⟨17922⟩ 89940

def event89966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17923⟩⟩) (.product (.predecessor 0 89964 .coefficient) (.predecessor 1 89965 .coefficient) (⟨false, false, none, none, none⟩))

def event89967 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17923⟩⟩, .operator (⟨89963, 0⟩, ⟨89940, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17922⟩⟩]⟩, (1)⟩)

def event89968 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17923⟩⟩, .operator (⟨89963, 1⟩, ⟨89940, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17922⟩⟩]⟩, (-1)⟩)

def event89969 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17923⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17922⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17922⟩⟩) ⟨17054⟩ 89937)

def event89970 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17923⟩⟩, .relation 89969 0, ⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨17054⟩⟩]⟩, (-1)⟩)

def exact89971RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17922⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨17054⟩⟩]⟩, (-1)⟩]

theorem exact89971RawTermsValid :
    exact89971RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89971 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17923⟩⟩) exact89971RawTerms .large 89966 .exactZero (none)

def event89972 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16126⟩⟩) 0 ⟨15837⟩ 89929

def event89973 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16126⟩⟩) (.authority (.programFamilyFact))

def exact89974RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩]

theorem exact89974RawTermsValid :
    exact89974RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89974 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16126⟩⟩) exact89974RawTerms (.finite 2) 89973 .exactZero (none)

def event89975 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16129⟩⟩) 0 ⟨6908⟩ 89951

def event89976 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16129⟩⟩) 1 ⟨16126⟩ 89974

def event89977 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16129⟩⟩) (.product (.predecessor 0 89975 .coefficient) (.predecessor 1 89976 .coefficient) (⟨false, true, none, none, some 1⟩))

def event89978 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16129⟩⟩, .operator (⟨89951, 0⟩, ⟨89974, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact89979RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact89979RawTermsValid :
    exact89979RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89979 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16129⟩⟩) exact89979RawTerms .large 89977 .exactZero (none)

def event89980 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7197⟩⟩) 0 ⟨7177⟩ 89933

def event89981 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7197⟩⟩) (.authority (.operator))

def exact89982RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩]

theorem exact89982RawTermsValid :
    exact89982RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89982 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7197⟩⟩) exact89982RawTerms .large 89981 .exactZero (none)

def event89983 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16130⟩⟩) 0 ⟨7197⟩ 89982

def event89984 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16130⟩⟩) 1 ⟨16129⟩ 89979

def event89985 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16130⟩⟩) (.sum [.predecessor 0 89983 .coefficient, .predecessor 1 89984 .coefficient])

def exact89986RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact89986RawTermsValid :
    exact89986RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89986 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16130⟩⟩) exact89986RawTerms .large 89985 .exactZero (none)

def event89987 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17928⟩⟩) 0 ⟨16130⟩ 89986

def event89988 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17928⟩⟩) 1 ⟨17923⟩ 89971

def event89989 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17928⟩⟩) (.sum [.predecessor 0 89987 .coefficient, .predecessor 1 89988 .coefficient])

def exact89990RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17922⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨17054⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact89990RawTermsValid :
    exact89990RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89990 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17928⟩⟩) exact89990RawTerms .large 89989 .exactZero (none)

def event89991 : Event := .preFoldPolynomial 89990 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17922⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨17054⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact89992RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17922⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨17054⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event89992 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17928⟩⟩) 89991 exact89992RawTerms .large 89989 .exactZero (none)

def event89993 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15837⟩⟩) ⟨⟨76⟩, ⟨56⟩, ⟨135⟩⟩ ⟨89835, 89993⟩

def event89994 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16715⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16712⟩⟩]⟩) (1) 0 2 (.universal 89993 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16712⟩⟩]⟩) (none) 89992)

def event89995 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16715⟩⟩, .relation 89994 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩)

def event89996 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16715⟩⟩, .relation 89994 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17922⟩⟩]⟩, (-1)⟩)

def event89997 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16715⟩⟩, .relation 89994 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨17054⟩⟩]⟩, (1)⟩)

def event89998 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16715⟩⟩, .relation 89994 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact89999RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17922⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨17054⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact89999RawTermsValid :
    exact89999RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event89999 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16715⟩⟩) exact89999RawTerms .large 89831 (.finite 202072841853861888) (some (89833))

def event90000 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17925⟩⟩) 0 ⟨16715⟩ 89999

def event90001 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17925⟩⟩) 1 ⟨17924⟩ 89821

def event90002 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17925⟩⟩) (.sum [.predecessor 0 90000 .coefficient, .predecessor 1 90001 .coefficient])

def event90003 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17925⟩⟩, .operator (⟨89999, 0⟩, ⟨89821, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17922⟩⟩]⟩, (1)⟩)

def event90004 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17925⟩⟩, .operator (⟨89999, 2⟩, ⟨89821, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15836⟩⟩], [⟨.program ⟨257⟩, ⟨17054⟩⟩]⟩, (-1)⟩)

def event90005 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17925⟩⟩) (.sum [.result 89999 .summary, .result 89821 .summary])

def exact90006RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact90006RawTermsValid :
    exact90006RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90006 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17925⟩⟩) exact90006RawTerms .large 90002 (.finite 32188807212483706889510625476608) (some (90005))

def event90007 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17926⟩⟩) 0 ⟨17925⟩ 90006

def event90008 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17926⟩⟩) 1 ⟨7172⟩ 15882

def event90009 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17926⟩⟩) (.product (.predecessor 0 90007 .coefficient) (.predecessor 1 90008 .coefficient) (⟨false, false, none, none, none⟩))

def event90010 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17926⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) [⟨.result 15878 .coefficient, false, none⟩])

def event90011 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17926⟩⟩) (.product (.result 90006 .summary) (.transfer 90010) (⟨false, false, none, none, none⟩))

def event90012 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17926⟩⟩, .operator (⟨90006, 0⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩)

def event90013 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17926⟩⟩, .operator (⟨90006, 1⟩, ⟨15882, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (-1)⟩)

def event90014 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17926⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7171⟩⟩) ⟨7051⟩ 15875)

def event90015 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17926⟩⟩, .relation 90014 0, ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact90016RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩]

theorem exact90016RawTermsValid :
    exact90016RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90016 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17926⟩⟩) exact90016RawTerms .large 90009 (.finite 345624685687166110058245054666339432529920) (some (90011))

def event90017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10372⟩⟩) 0 ⟨6727⟩ 723

def event90018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10372⟩⟩) 1 ⟨10328⟩ 75903

def event90019 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10372⟩⟩) (.tensor (.predecessor 0 90017 .coefficient) (.predecessor 1 90018 .coefficient) true false)

def event90020 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10372⟩⟩, .operator (⟨723, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact90021RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact90021RawTermsValid :
    exact90021RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90021 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10372⟩⟩) exact90021RawTerms .large 90019 .exactZero (none)

def event90022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10350⟩⟩) 0 ⟨10327⟩ 75773

def event90023 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10350⟩⟩) 1 ⟨7292⟩ 15896

def event90024 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10350⟩⟩) (.product (.predecessor 0 90022 .coefficient) (.predecessor 1 90023 .coefficient) (⟨false, false, none, none, none⟩))

def event90025 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10350⟩⟩, .operator (⟨75773, 0⟩, ⟨15896, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩)

def exact90026RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact90026RawTermsValid :
    exact90026RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90026 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10350⟩⟩) exact90026RawTerms .large 90024 .exactZero (none)

def event90027 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10373⟩⟩) 0 ⟨10350⟩ 90026

def event90028 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10373⟩⟩) 1 ⟨10372⟩ 90021

def event90029 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10373⟩⟩) (.sum [.predecessor 0 90027 .coefficient, .predecessor 1 90028 .coefficient])

def exact90030RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact90030RawTermsValid :
    exact90030RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90030 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10373⟩⟩) exact90030RawTerms .large 90029 .exactZero (none)

def event90031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10374⟩⟩) 0 ⟨10373⟩ 90030

def event90032 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10374⟩⟩) 1 ⟨118⟩ 31516

def event90033 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10374⟩⟩) (.sum [.predecessor 0 90031 .coefficient, .predecessor 1 90032 .coefficient])

def event90034 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10374⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) [⟨.result 31516 .coefficient, false, none⟩])

def event90035 : Event := .survivorFold (1) 90034

def exact90036RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (1)⟩]

theorem exact90036RawTermsValid :
    exact90036RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90036 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10374⟩⟩) exact90036RawTerms .large 90033 (.finite 26) (some (90034))

def event90037 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10375⟩⟩) 0 ⟨10374⟩ 90036

def event90038 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10375⟩⟩) 1 ⟨10374⟩ 90036

def event90039 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10375⟩⟩) (.sum [.predecessor 0 90037 .coefficient, .predecessor 1 90038 .coefficient])

def event90040 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10375⟩⟩, .operator (⟨90036, 0⟩, ⟨90036, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6727⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def event90041 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10375⟩⟩, .operator (⟨90036, 1⟩, ⟨90036, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7292⟩⟩]⟩, (-1)⟩)

def event90042 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10375⟩⟩) (.sum [.result 90036 .summary, .result 90036 .summary])

def exact90043RawTerms : List Term := []

theorem exact90043RawTermsValid :
    exact90043RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90043 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10375⟩⟩) exact90043RawTerms .large 90039 (.finite 52) (some (90042))

def event90044 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17927⟩⟩) 0 ⟨10375⟩ 90043

def event90045 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17927⟩⟩) 1 ⟨17926⟩ 90016

def event90046 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17927⟩⟩) (.sum [.predecessor 0 90044 .coefficient, .predecessor 1 90045 .coefficient])

def event90047 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17927⟩⟩) (.sum [.result 90043 .summary, .result 90016 .summary])

def exact90048RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩]

theorem exact90048RawTermsValid :
    exact90048RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90048 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17927⟩⟩) exact90048RawTerms .large 90046 (.finite 345624685687166110058245054666339432529972) (some (90047))

def event90049 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20836⟩⟩) 0 ⟨17927⟩ 90048

def event90050 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20836⟩⟩) 1 ⟨20835⟩ 89804

def event90051 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20836⟩⟩) (.sum [.predecessor 0 90049 .coefficient, .predecessor 1 90050 .coefficient])

def event90052 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20836⟩⟩) (.sum [.result 90048 .summary, .result 89804 .summary])

def exact90053RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩]

theorem exact90053RawTermsValid :
    exact90053RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90053 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20836⟩⟩) exact90053RawTerms .large 90051 (.finite 691250426059631610003352154589745737891892) (some (90052))

def event90054 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24056⟩⟩) 0 ⟨20836⟩ 90053

def event90055 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24056⟩⟩) 1 ⟨24055⟩ 89592

def event90056 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24056⟩⟩) (.sum [.predecessor 0 90054 .coefficient, .predecessor 1 90055 .coefficient])

def event90057 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24056⟩⟩) (.sum [.result 90053 .summary, .result 89592 .summary])

def exact90058RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩]

theorem exact90058RawTermsValid :
    exact90058RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90058 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24056⟩⟩) exact90058RawTerms .large 90056 (.finite 1036877221117396499835321299770218916085812) (some (90057))

def event90059 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34076⟩⟩) 0 ⟨24056⟩ 90058

def event90060 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34076⟩⟩) 1 ⟨34075⟩ 89380

def event90061 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34076⟩⟩) (.sum [.predecessor 0 90059 .coefficient, .predecessor 1 90060 .coefficient])

def event90062 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34076⟩⟩) (.sum [.result 90058 .summary, .result 89380 .summary])

def exact90063RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩]

theorem exact90063RawTermsValid :
    exact90063RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90063 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34076⟩⟩) exact90063RawTerms .large 90061 (.finite 1382506125545760169441014535464825839943732) (some (90062))

def event90064 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53136⟩⟩) 0 ⟨34076⟩ 90063

def event90065 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53136⟩⟩) 1 ⟨53135⟩ 89168

def event90066 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53136⟩⟩) (.sum [.predecessor 0 90064 .coefficient, .predecessor 1 90065 .coefficient])

def event90067 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53136⟩⟩) (.sum [.result 90063 .summary, .result 89168 .summary])

def exact90068RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩]

theorem exact90068RawTermsValid :
    exact90068RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90068 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53136⟩⟩) exact90068RawTerms .large 90066 (.finite 1728139248715321398594155952187700255129652) (some (90067))

def event90069 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56116⟩⟩) 0 ⟨53136⟩ 90068

def event90070 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56116⟩⟩) 1 ⟨56115⟩ 88956

def event90071 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56116⟩⟩) (.sum [.predecessor 0 90069 .coefficient, .predecessor 1 90070 .coefficient])

def event90072 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56116⟩⟩) (.sum [.result 90068 .summary, .result 88956 .summary])

def exact90073RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩]

theorem exact90073RawTermsValid :
    exact90073RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90073 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56116⟩⟩) exact90073RawTerms .large 90071 (.finite 2073774481255481407521021459424708415979572) (some (90072))

def event90074 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59096⟩⟩) 0 ⟨56116⟩ 90073

def event90075 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59096⟩⟩) 1 ⟨59095⟩ 88744

def event90076 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59096⟩⟩) (.sum [.predecessor 0 90074 .coefficient, .predecessor 1 90075 .coefficient])

def event90077 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59096⟩⟩) (.sum [.result 90073 .summary, .result 88744 .summary])

def exact90078RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩]

theorem exact90078RawTermsValid :
    exact90078RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90078 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59096⟩⟩) exact90078RawTerms .large 90076 (.finite 2419413932536838975995335147689984068157492) (some (90077))

def event90079 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62076⟩⟩) 0 ⟨59096⟩ 90078

def event90080 : Event := .predecessor (⟨.program ⟨257⟩, ⟨62076⟩⟩) 1 ⟨62075⟩ 88532

def event90081 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62076⟩⟩) (.sum [.predecessor 0 90079 .coefficient, .predecessor 1 90080 .coefficient])

def event90082 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨62076⟩⟩) (.sum [.result 90078 .summary, .result 88532 .summary])

def exact90083RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩]

theorem exact90083RawTermsValid :
    exact90083RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90083 : Event := .resultExact (⟨.program ⟨257⟩, ⟨62076⟩⟩) exact90083RawTerms .large 90081 (.finite 2765055493188795324243372926469393465999412) (some (90082))

def event90084 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65056⟩⟩) 0 ⟨62076⟩ 90083

def event90085 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65056⟩⟩) 1 ⟨65055⟩ 88320

def event90086 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65056⟩⟩) (.sum [.predecessor 0 90084 .coefficient, .predecessor 1 90085 .coefficient])

def event90087 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65056⟩⟩) (.sum [.result 90083 .summary, .result 88320 .summary])

def exact90088RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩]

theorem exact90088RawTermsValid :
    exact90088RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90088 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65056⟩⟩) exact90088RawTerms .large 90086 (.finite 3110701272581949232038858886277070355169332) (some (90087))

def event90089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70641⟩⟩) 0 ⟨65056⟩ 90088

def event90090 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70641⟩⟩) 1 ⟨70640⟩ 88108

def event90091 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70641⟩⟩) (.sum [.predecessor 0 90089 .coefficient, .predecessor 1 90090 .coefficient])

def event90092 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70641⟩⟩) (.sum [.result 90088 .summary, .result 88108 .summary])

def exact90093RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩]

theorem exact90093RawTermsValid :
    exact90093RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90093 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70641⟩⟩) exact90093RawTerms .large 90091 (.finite 3456353380086899479155517117627148481331252) (some (90092))

def event90094 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70642⟩⟩) 0 ⟨70641⟩ 90093

def event90095 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70642⟩⟩) 1 ⟨28437⟩ 87896

def event90096 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70642⟩⟩) (.sum [.predecessor 0 90094 .coefficient, .predecessor 1 90095 .coefficient])

def event90097 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70642⟩⟩) (.sum [.result 90093 .summary, .result 87896 .summary])

def exact90098RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩]

theorem exact90098RawTermsValid :
    exact90098RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90098 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70642⟩⟩) exact90098RawTerms .large 90096 (.finite 3802007596962448506045899439491360353157172) (some (90097))

def event90099 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70643⟩⟩) 0 ⟨70642⟩ 90098

def event90100 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70643⟩⟩) 1 ⟨31117⟩ 87684

def event90101 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70643⟩⟩) (.sum [.predecessor 0 90099 .coefficient, .predecessor 1 90100 .coefficient])

def event90102 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70643⟩⟩) (.sum [.result 90098 .summary, .result 87684 .summary])

def exact90103RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩]

theorem exact90103RawTermsValid :
    exact90103RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90103 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70643⟩⟩) exact90103RawTerms .large 90101 (.finite 4147668141949793872257454032897973461975092) (some (90102))

def event90104 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70644⟩⟩) 0 ⟨70643⟩ 90103

def event90105 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70644⟩⟩) 1 ⟨36777⟩ 87472

def event90106 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70644⟩⟩) (.sum [.predecessor 0 90104 .coefficient, .predecessor 1 90105 .coefficient])

def event90107 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70644⟩⟩) (.sum [.result 90103 .summary, .result 87472 .summary])

def exact90108RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7199⟩⟩, ⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7203⟩⟩, ⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7217⟩⟩, ⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7219⟩⟩, ⟨.program ⟨257⟩, ⟨7167⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩, (1)⟩]

theorem exact90108RawTermsValid :
    exact90108RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event90108 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70644⟩⟩) exact90108RawTerms .large 90106 (.finite 4493332905678336798016456807332854062121012) (some (90107))

def event90109 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70645⟩⟩) 0 ⟨70644⟩ 90108

def event90110 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70645⟩⟩) 1 ⟨39457⟩ 87260

def event90111 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70645⟩⟩) (.sum [.predecessor 0 90109 .coefficient, .predecessor 1 90110 .coefficient])

def eventLeaf5616 : Array AnnotatedEvent := #[
  { event := event89856
    frameStart := 89835 },
  { event := event89857
    frameStart := 89835 },
  { event := event89858
    frameStart := 89835 },
  { event := event89859
    frameStart := 89835 },
  { event := event89860
    frameStart := 89835 },
  { event := event89861
    frameStart := 89835 },
  { event := event89862
    frameStart := 89835 },
  { event := event89863
    frameStart := 89835 },
  { event := event89864
    frameStart := 89835 },
  { event := event89865
    frameStart := 89835 },
  { event := event89866
    frameStart := 89835 },
  { event := event89867
    frameStart := 89835 },
  { event := event89868
    frameStart := 89835 },
  { event := event89869
    frameStart := 89835 },
  { event := event89870
    frameStart := 89835 },
  { event := event89871
    frameStart := 89835 }
]

def eventLeaf5617 : Array AnnotatedEvent := #[
  { event := event89872
    frameStart := 89835 },
  { event := event89873
    frameStart := 89835 },
  { event := event89874
    frameStart := 89835 },
  { event := event89875
    frameStart := 89835 },
  { event := event89876
    frameStart := 89835 },
  { event := event89877
    frameStart := 89835 },
  { event := event89878
    frameStart := 89835 },
  { event := event89879
    frameStart := 89835 },
  { event := event89880
    frameStart := 89835 },
  { event := event89881
    frameStart := 89835 },
  { event := event89882
    frameStart := 89835 },
  { event := event89883
    frameStart := 89835 },
  { event := event89884
    frameStart := 89835 },
  { event := event89885
    frameStart := 89835 },
  { event := event89886
    frameStart := 89835 },
  { event := event89887
    frameStart := 89835 }
]

def eventLeaf5618 : Array AnnotatedEvent := #[
  { event := event89888
    frameStart := 89835 },
  { event := event89889
    frameStart := 89889 },
  { event := event89890
    frameStart := 89889 },
  { event := event89891
    frameStart := 89889 },
  { event := event89892
    frameStart := 89889 },
  { event := event89893
    frameStart := 89889 },
  { event := event89894
    frameStart := 89889 },
  { event := event89895
    frameStart := 89889 },
  { event := event89896
    frameStart := 89889 },
  { event := event89897
    frameStart := 89889 },
  { event := event89898
    frameStart := 89889 },
  { event := event89899
    frameStart := 89889 },
  { event := event89900
    frameStart := 89889 },
  { event := event89901
    frameStart := 89889 },
  { event := event89902
    frameStart := 89889 },
  { event := event89903
    frameStart := 89889 }
]

def eventLeaf5619 : Array AnnotatedEvent := #[
  { event := event89904
    frameStart := 89889 },
  { event := event89905
    frameStart := 89889 },
  { event := event89906
    frameStart := 89889 },
  { event := event89907
    frameStart := 89889 },
  { event := event89908
    frameStart := 89889 },
  { event := event89909
    frameStart := 89889 },
  { event := event89910
    frameStart := 89889 },
  { event := event89911
    frameStart := 89889 },
  { event := event89912
    frameStart := 89889 },
  { event := event89913
    frameStart := 89889 },
  { event := event89914
    frameStart := 89889 },
  { event := event89915
    frameStart := 89889 },
  { event := event89916
    frameStart := 89889 },
  { event := event89917
    frameStart := 89889 },
  { event := event89918
    frameStart := 89889 },
  { event := event89919
    frameStart := 89889 }
]

def eventLeaf5620 : Array AnnotatedEvent := #[
  { event := event89920
    frameStart := 89889 },
  { event := event89921
    frameStart := 89889 },
  { event := event89922
    frameStart := 89889 },
  { event := event89923
    frameStart := 89889 },
  { event := event89924
    frameStart := 89889 },
  { event := event89925
    frameStart := 89889 },
  { event := event89926
    frameStart := 89889 },
  { event := event89927
    frameStart := 89889 },
  { event := event89928
    frameStart := 89889 },
  { event := event89929
    frameStart := 89889 },
  { event := event89930
    frameStart := 89889 },
  { event := event89931
    frameStart := 89889 },
  { event := event89932
    frameStart := 89889 },
  { event := event89933
    frameStart := 89889 },
  { event := event89934
    frameStart := 89889 },
  { event := event89935
    frameStart := 89889 }
]

def eventLeaf5621 : Array AnnotatedEvent := #[
  { event := event89936
    frameStart := 89889 },
  { event := event89937
    frameStart := 89889 },
  { event := event89938
    frameStart := 89889 },
  { event := event89939
    frameStart := 89889 },
  { event := event89940
    frameStart := 89889 },
  { event := event89941
    frameStart := 89889 },
  { event := event89942
    frameStart := 89889 },
  { event := event89943
    frameStart := 89889 },
  { event := event89944
    frameStart := 89889 },
  { event := event89945
    frameStart := 89889 },
  { event := event89946
    frameStart := 89889 },
  { event := event89947
    frameStart := 89889 },
  { event := event89948
    frameStart := 89889 },
  { event := event89949
    frameStart := 89889 },
  { event := event89950
    frameStart := 89889 },
  { event := event89951
    frameStart := 89889 }
]

def eventLeaf5622 : Array AnnotatedEvent := #[
  { event := event89952
    frameStart := 89889 },
  { event := event89953
    frameStart := 89889 },
  { event := event89954
    frameStart := 89889 },
  { event := event89955
    frameStart := 89889 },
  { event := event89956
    frameStart := 89889 },
  { event := event89957
    frameStart := 89889 },
  { event := event89958
    frameStart := 89889 },
  { event := event89959
    frameStart := 89889 },
  { event := event89960
    frameStart := 89889 },
  { event := event89961
    frameStart := 89889 },
  { event := event89962
    frameStart := 89889 },
  { event := event89963
    frameStart := 89889 },
  { event := event89964
    frameStart := 89889 },
  { event := event89965
    frameStart := 89889 },
  { event := event89966
    frameStart := 89889 },
  { event := event89967
    frameStart := 89889 }
]

def eventLeaf5623 : Array AnnotatedEvent := #[
  { event := event89968
    frameStart := 89889 },
  { event := event89969
    frameStart := 89889 },
  { event := event89970
    frameStart := 89889 },
  { event := event89971
    frameStart := 89889 },
  { event := event89972
    frameStart := 89889 },
  { event := event89973
    frameStart := 89889 },
  { event := event89974
    frameStart := 89889 },
  { event := event89975
    frameStart := 89889 },
  { event := event89976
    frameStart := 89889 },
  { event := event89977
    frameStart := 89889 },
  { event := event89978
    frameStart := 89889 },
  { event := event89979
    frameStart := 89889 },
  { event := event89980
    frameStart := 89889 },
  { event := event89981
    frameStart := 89889 },
  { event := event89982
    frameStart := 89889 },
  { event := event89983
    frameStart := 89889 }
]

def eventLeaf5624 : Array AnnotatedEvent := #[
  { event := event89984
    frameStart := 89889 },
  { event := event89985
    frameStart := 89889 },
  { event := event89986
    frameStart := 89889 },
  { event := event89987
    frameStart := 89889 },
  { event := event89988
    frameStart := 89889 },
  { event := event89989
    frameStart := 89889 },
  { event := event89990
    frameStart := 89889 },
  { event := event89991
    frameStart := 89889 },
  { event := event89992
    frameStart := 89889 },
  { event := event89993
    frameStart := 0 },
  { event := event89994
    frameStart := 0 },
  { event := event89995
    frameStart := 0 },
  { event := event89996
    frameStart := 0 },
  { event := event89997
    frameStart := 0 },
  { event := event89998
    frameStart := 0 },
  { event := event89999
    frameStart := 0 }
]

def eventLeaf5625 : Array AnnotatedEvent := #[
  { event := event90000
    frameStart := 0 },
  { event := event90001
    frameStart := 0 },
  { event := event90002
    frameStart := 0 },
  { event := event90003
    frameStart := 0 },
  { event := event90004
    frameStart := 0 },
  { event := event90005
    frameStart := 0 },
  { event := event90006
    frameStart := 0 },
  { event := event90007
    frameStart := 0 },
  { event := event90008
    frameStart := 0 },
  { event := event90009
    frameStart := 0 },
  { event := event90010
    frameStart := 0 },
  { event := event90011
    frameStart := 0 },
  { event := event90012
    frameStart := 0 },
  { event := event90013
    frameStart := 0 },
  { event := event90014
    frameStart := 0 },
  { event := event90015
    frameStart := 0 }
]

def eventLeaf5626 : Array AnnotatedEvent := #[
  { event := event90016
    frameStart := 0 },
  { event := event90017
    frameStart := 0 },
  { event := event90018
    frameStart := 0 },
  { event := event90019
    frameStart := 0 },
  { event := event90020
    frameStart := 0 },
  { event := event90021
    frameStart := 0 },
  { event := event90022
    frameStart := 0 },
  { event := event90023
    frameStart := 0 },
  { event := event90024
    frameStart := 0 },
  { event := event90025
    frameStart := 0 },
  { event := event90026
    frameStart := 0 },
  { event := event90027
    frameStart := 0 },
  { event := event90028
    frameStart := 0 },
  { event := event90029
    frameStart := 0 },
  { event := event90030
    frameStart := 0 },
  { event := event90031
    frameStart := 0 }
]

def eventLeaf5627 : Array AnnotatedEvent := #[
  { event := event90032
    frameStart := 0 },
  { event := event90033
    frameStart := 0 },
  { event := event90034
    frameStart := 0 },
  { event := event90035
    frameStart := 0 },
  { event := event90036
    frameStart := 0 },
  { event := event90037
    frameStart := 0 },
  { event := event90038
    frameStart := 0 },
  { event := event90039
    frameStart := 0 },
  { event := event90040
    frameStart := 0 },
  { event := event90041
    frameStart := 0 },
  { event := event90042
    frameStart := 0 },
  { event := event90043
    frameStart := 0 },
  { event := event90044
    frameStart := 0 },
  { event := event90045
    frameStart := 0 },
  { event := event90046
    frameStart := 0 },
  { event := event90047
    frameStart := 0 }
]

def eventLeaf5628 : Array AnnotatedEvent := #[
  { event := event90048
    frameStart := 0 },
  { event := event90049
    frameStart := 0 },
  { event := event90050
    frameStart := 0 },
  { event := event90051
    frameStart := 0 },
  { event := event90052
    frameStart := 0 },
  { event := event90053
    frameStart := 0 },
  { event := event90054
    frameStart := 0 },
  { event := event90055
    frameStart := 0 },
  { event := event90056
    frameStart := 0 },
  { event := event90057
    frameStart := 0 },
  { event := event90058
    frameStart := 0 },
  { event := event90059
    frameStart := 0 },
  { event := event90060
    frameStart := 0 },
  { event := event90061
    frameStart := 0 },
  { event := event90062
    frameStart := 0 },
  { event := event90063
    frameStart := 0 }
]

def eventLeaf5629 : Array AnnotatedEvent := #[
  { event := event90064
    frameStart := 0 },
  { event := event90065
    frameStart := 0 },
  { event := event90066
    frameStart := 0 },
  { event := event90067
    frameStart := 0 },
  { event := event90068
    frameStart := 0 },
  { event := event90069
    frameStart := 0 },
  { event := event90070
    frameStart := 0 },
  { event := event90071
    frameStart := 0 },
  { event := event90072
    frameStart := 0 },
  { event := event90073
    frameStart := 0 },
  { event := event90074
    frameStart := 0 },
  { event := event90075
    frameStart := 0 },
  { event := event90076
    frameStart := 0 },
  { event := event90077
    frameStart := 0 },
  { event := event90078
    frameStart := 0 },
  { event := event90079
    frameStart := 0 }
]

def eventLeaf5630 : Array AnnotatedEvent := #[
  { event := event90080
    frameStart := 0 },
  { event := event90081
    frameStart := 0 },
  { event := event90082
    frameStart := 0 },
  { event := event90083
    frameStart := 0 },
  { event := event90084
    frameStart := 0 },
  { event := event90085
    frameStart := 0 },
  { event := event90086
    frameStart := 0 },
  { event := event90087
    frameStart := 0 },
  { event := event90088
    frameStart := 0 },
  { event := event90089
    frameStart := 0 },
  { event := event90090
    frameStart := 0 },
  { event := event90091
    frameStart := 0 },
  { event := event90092
    frameStart := 0 },
  { event := event90093
    frameStart := 0 },
  { event := event90094
    frameStart := 0 },
  { event := event90095
    frameStart := 0 }
]

def eventLeaf5631 : Array AnnotatedEvent := #[
  { event := event90096
    frameStart := 0 },
  { event := event90097
    frameStart := 0 },
  { event := event90098
    frameStart := 0 },
  { event := event90099
    frameStart := 0 },
  { event := event90100
    frameStart := 0 },
  { event := event90101
    frameStart := 0 },
  { event := event90102
    frameStart := 0 },
  { event := event90103
    frameStart := 0 },
  { event := event90104
    frameStart := 0 },
  { event := event90105
    frameStart := 0 },
  { event := event90106
    frameStart := 0 },
  { event := event90107
    frameStart := 0 },
  { event := event90108
    frameStart := 0 },
  { event := event90109
    frameStart := 0 },
  { event := event90110
    frameStart := 0 },
  { event := event90111
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events351
