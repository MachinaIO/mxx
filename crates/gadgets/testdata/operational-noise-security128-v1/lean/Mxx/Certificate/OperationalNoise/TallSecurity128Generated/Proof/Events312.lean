import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events312

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event79872 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 79862

def event79873 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 79871 .coefficient, .predecessor 1 79872 .coefficient])

def event79874 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event79875 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 79874

def event79876 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 79860

def event79877 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 79876 .coefficient))

def event79878 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event79879 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25802⟩⟩) 0 ⟨10325⟩ 79878

def event79880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25802⟩⟩) (.authority (.programFamilyFact))

def exact79881RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩], []⟩, (1)⟩]

theorem exact79881RawTermsValid :
    exact79881RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79881 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25802⟩⟩) exact79881RawTerms (.finite 28) 79880 .exactZero (none)

def event79882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65607⟩⟩) 0 ⟨10325⟩ 79878

def event79883 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65607⟩⟩) (.authority (.programFamilyFact))

def exact79884RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65607⟩⟩], []⟩, (1)⟩]

theorem exact79884RawTermsValid :
    exact79884RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79884 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65607⟩⟩) exact79884RawTerms (.finite 28) 79883 .exactZero (none)

def event79885 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65608⟩⟩) 0 ⟨65607⟩ 79884

def event79886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65608⟩⟩) 1 ⟨25802⟩ 79881

def event79887 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65608⟩⟩) (.product (.predecessor 0 79885 .coefficient) (.predecessor 1 79886 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event79888 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65608⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], []⟩) [⟨.result 79884 .coefficient, true, some 1⟩, ⟨.result 79881 .coefficient, true, some 1⟩])

def event79889 : Event := .survivorFold (1) 79888

def exact79890RawTerms : List Term := []

theorem exact79890RawTermsValid :
    exact79890RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79890 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65608⟩⟩) exact79890RawTerms (.finite 784) 79887 (.finite 784) (some (79888))

def event79891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65609⟩⟩) 0 ⟨65608⟩ 79890

def event79892 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65609⟩⟩) (.identity (.predecessor 0 79891 .coefficient))

def event79893 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65609⟩⟩) (.finite 784)

def event79894 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67830⟩⟩) 0 ⟨65609⟩ 79893

def event79895 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67830⟩⟩) (.authority (.relationPreimageSource ⟨46⟩))

def exact79896RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨67830⟩⟩]⟩, (1)⟩]

theorem exact79896RawTermsValid :
    exact79896RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79896 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67830⟩⟩) exact79896RawTerms (.finite 5647228698) 79895 .exactZero (none)

def event79897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact79898RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact79898RawTermsValid :
    exact79898RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79898 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact79898RawTerms .large 79897 .exactZero (none)

def event79899 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67831⟩⟩) 0 ⟨35⟩ 79898

def event79900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67831⟩⟩) 1 ⟨67830⟩ 79896

def event79901 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67831⟩⟩) (.product (.predecessor 0 79899 .coefficient) (.predecessor 1 79900 .coefficient) (⟨false, false, none, none, none⟩))

def event79902 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67831⟩⟩, .operator (⟨79898, 0⟩, ⟨79896, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67830⟩⟩]⟩, (1)⟩)

def exact79903RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67830⟩⟩]⟩, (1)⟩]

theorem exact79903RawTermsValid :
    exact79903RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79903 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67831⟩⟩) exact79903RawTerms .large 79901 .exactZero (none)

def event79904 : Event := .preFoldPolynomial 79903 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67830⟩⟩]⟩, (1)⟩] .exactZero none

def exact79905RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67830⟩⟩]⟩, (1)⟩]

def event79905 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨67831⟩⟩) 79904 exact79905RawTerms .large 79901 .exactZero (none)

def event79906 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨69310⟩⟩)

def event79907 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event79908 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event79909 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event79910 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event79911 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event79912 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event79913 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event79914 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event79915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 79914

def event79916 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 79912

def event79917 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 79915 .coefficient) (.value (.predecessor 1 79916 .coefficient)))

def event79918 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event79919 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 79918

def event79920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 79910

def event79921 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 79919 .coefficient, .predecessor 1 79920 .coefficient])

def event79922 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event79923 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 79922

def event79924 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 79908

def event79925 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 79924 .coefficient))

def event79926 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event79927 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25802⟩⟩) 0 ⟨10325⟩ 79926

def event79928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25802⟩⟩) (.authority (.programFamilyFact))

def exact79929RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩], []⟩, (1)⟩]

theorem exact79929RawTermsValid :
    exact79929RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79929 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25802⟩⟩) exact79929RawTerms (.finite 28) 79928 .exactZero (none)

def event79930 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65607⟩⟩) 0 ⟨10325⟩ 79926

def event79931 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65607⟩⟩) (.authority (.programFamilyFact))

def exact79932RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65607⟩⟩], []⟩, (1)⟩]

theorem exact79932RawTermsValid :
    exact79932RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79932 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65607⟩⟩) exact79932RawTerms (.finite 28) 79931 .exactZero (none)

def event79933 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65608⟩⟩) 0 ⟨65607⟩ 79932

def event79934 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65608⟩⟩) 1 ⟨25802⟩ 79929

def event79935 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65608⟩⟩) (.product (.predecessor 0 79933 .coefficient) (.predecessor 1 79934 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event79936 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65608⟩⟩, .operator (⟨79932, 0⟩, ⟨79929, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], []⟩, (1)⟩)

def exact79937RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], []⟩, (1)⟩]

theorem exact79937RawTermsValid :
    exact79937RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79937 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65608⟩⟩) exact79937RawTerms (.finite 784) 79935 .exactZero (none)

def event79938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65609⟩⟩) 0 ⟨65608⟩ 79937

def event79939 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65609⟩⟩) (.identity (.predecessor 0 79938 .coefficient))

def event79940 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65609⟩⟩) (.finite 784)

def event79941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68565⟩⟩) 0 ⟨65609⟩ 79940

def event79942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68565⟩⟩) (.authority (.programFamilyFact))

def event79943 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68565⟩⟩) (.finite 3720)

def event79944 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event79945 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68566⟩⟩) 0 ⟨7177⟩ 79944

def event79946 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68566⟩⟩) 1 ⟨68565⟩ 79943

def event79947 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68566⟩⟩) (.authority (.operator))

def exact79948RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68566⟩⟩]⟩, (1)⟩]

theorem exact79948RawTermsValid :
    exact79948RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79948 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68566⟩⟩) exact79948RawTerms .large 79947 .exactZero (none)

def event79949 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69306⟩⟩) 0 ⟨68566⟩ 79948

def event79950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69306⟩⟩) (.authority (.operator))

def exact79951RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩, (1)⟩]

theorem exact79951RawTermsValid :
    exact79951RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79951 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69306⟩⟩) exact79951RawTerms (.finite 8192) 79950 .exactZero (none)

def event79952 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event79953 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event79954 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68951⟩⟩) 0 ⟨65609⟩ 79940

def event79955 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68951⟩⟩) 1 ⟨136⟩ 79953

def event79956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68951⟩⟩) (.sum [.predecessor 0 79954 .coefficient, .predecessor 1 79955 .coefficient])

def event79957 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨68951⟩⟩) (.finite 784)

def event79958 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68952⟩⟩) 0 ⟨68951⟩ 79957

def event79959 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68952⟩⟩) (.identity (.predecessor 0 79958 .coefficient))

def exact79960RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], []⟩, (1)⟩]

theorem exact79960RawTermsValid :
    exact79960RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79960 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68952⟩⟩) exact79960RawTerms (.finite 784) 79959 .exactZero (none)

def event79961 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact79962RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact79962RawTermsValid :
    exact79962RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79962 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact79962RawTerms .large 79961 .exactZero (none)

def event79963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68953⟩⟩) 0 ⟨6908⟩ 79962

def event79964 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68953⟩⟩) 1 ⟨68952⟩ 79960

def event79965 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68953⟩⟩) (.product (.predecessor 0 79963 .coefficient) (.predecessor 1 79964 .coefficient) (⟨false, false, none, none, none⟩))

def event79966 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68953⟩⟩, .operator (⟨79962, 0⟩, ⟨79960, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact79967RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact79967RawTermsValid :
    exact79967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79967 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68953⟩⟩) exact79967RawTerms .large 79965 .exactZero (none)

def event79968 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event79969 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event79970 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 79944

def event79971 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact79972RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact79972RawTermsValid :
    exact79972RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79972 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact79972RawTerms .large 79971 .exactZero (none)

def event79973 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7276⟩⟩) 0 ⟨7178⟩ 79972

def event79974 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7276⟩⟩) (.identity (.predecessor 0 79973 .coefficient))

def exact79975RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩, (1)⟩]

theorem exact79975RawTermsValid :
    exact79975RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79975 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7276⟩⟩) exact79975RawTerms .large 79974 .exactZero (none)

def event79976 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9541⟩⟩) 0 ⟨7276⟩ 79975

def event79977 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9541⟩⟩) (.authority (.operator))

def exact79978RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩]

theorem exact79978RawTermsValid :
    exact79978RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79978 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9541⟩⟩) exact79978RawTerms (.finite 8192) 79977 .exactZero (none)

def event79979 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9542⟩⟩) 0 ⟨9541⟩ 79978

def event79980 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9542⟩⟩) 1 ⟨2370⟩ 79969

def event79981 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9542⟩⟩) (.scale (.predecessor 0 79979 .coefficient) (.value (.predecessor 1 79980 .coefficient)))

def exact79982RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩]

theorem exact79982RawTermsValid :
    exact79982RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79982 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9542⟩⟩) exact79982RawTerms (.finite 8192) 79981 .exactZero (none)

def event79983 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7294⟩⟩) 0 ⟨7178⟩ 79972

def event79984 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7294⟩⟩) (.identity (.predecessor 0 79983 .coefficient))

def exact79985RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩]⟩, (1)⟩]

theorem exact79985RawTermsValid :
    exact79985RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79985 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7294⟩⟩) exact79985RawTerms .large 79984 .exactZero (none)

def event79986 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9543⟩⟩) 0 ⟨7294⟩ 79985

def event79987 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9543⟩⟩) 1 ⟨9542⟩ 79982

def event79988 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9543⟩⟩) (.product (.predecessor 0 79986 .coefficient) (.predecessor 1 79987 .coefficient) (⟨false, false, none, none, none⟩))

def event79989 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9543⟩⟩, .operator (⟨79985, 0⟩, ⟨79982, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩)

def exact79990RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩]

theorem exact79990RawTermsValid :
    exact79990RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79990 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9543⟩⟩) exact79990RawTerms .large 79988 .exactZero (none)

def event79991 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68954⟩⟩) 0 ⟨9543⟩ 79990

def event79992 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68954⟩⟩) 1 ⟨68953⟩ 79967

def event79993 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68954⟩⟩) (.sum [.predecessor 0 79991 .coefficient, .predecessor 1 79992 .coefficient])

def exact79994RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact79994RawTermsValid :
    exact79994RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event79994 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68954⟩⟩) exact79994RawTerms .large 79993 .exactZero (none)

def event79995 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69309⟩⟩) 0 ⟨68954⟩ 79994

def event79996 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69309⟩⟩) 1 ⟨69306⟩ 79951

def event79997 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69309⟩⟩) (.product (.predecessor 0 79995 .coefficient) (.predecessor 1 79996 .coefficient) (⟨false, false, none, none, none⟩))

def event79998 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69309⟩⟩, .operator (⟨79994, 0⟩, ⟨79951, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩, (1)⟩)

def event79999 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69309⟩⟩, .operator (⟨79994, 1⟩, ⟨79951, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩, (-1)⟩)

def event80000 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨69309⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69306⟩⟩) ⟨68566⟩ 79948)

def event80001 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69309⟩⟩, .relation 80000 0, ⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨68566⟩⟩]⟩, (-1)⟩)

def exact80002RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨68566⟩⟩]⟩, (-1)⟩]

theorem exact80002RawTermsValid :
    exact80002RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80002 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69309⟩⟩) exact80002RawTerms .large 79997 .exactZero (none)

def event80003 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65836⟩⟩) 0 ⟨65609⟩ 79940

def event80004 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65836⟩⟩) (.authority (.programFamilyFact))

def exact80005RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], []⟩, (1)⟩]

theorem exact80005RawTermsValid :
    exact80005RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80005 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65836⟩⟩) exact80005RawTerms (.finite 28) 80004 .exactZero (none)

def event80006 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65838⟩⟩) 0 ⟨6908⟩ 79962

def event80007 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65838⟩⟩) 1 ⟨65836⟩ 80005

def event80008 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65838⟩⟩) (.product (.predecessor 0 80006 .coefficient) (.predecessor 1 80007 .coefficient) (⟨false, true, none, none, some 1⟩))

def event80009 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨65838⟩⟩, .operator (⟨79962, 0⟩, ⟨80005, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact80010RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact80010RawTermsValid :
    exact80010RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80010 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65838⟩⟩) exact80010RawTerms .large 80008 .exactZero (none)

def event80011 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7188⟩⟩) 0 ⟨7177⟩ 79944

def event80012 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7188⟩⟩) (.authority (.operator))

def exact80013RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩]

theorem exact80013RawTermsValid :
    exact80013RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80013 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7188⟩⟩) exact80013RawTerms .large 80012 .exactZero (none)

def event80014 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65839⟩⟩) 0 ⟨7188⟩ 80013

def event80015 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65839⟩⟩) 1 ⟨65838⟩ 80010

def event80016 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65839⟩⟩) (.sum [.predecessor 0 80014 .coefficient, .predecessor 1 80015 .coefficient])

def exact80017RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80017RawTermsValid :
    exact80017RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80017 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65839⟩⟩) exact80017RawTerms .large 80016 .exactZero (none)

def event80018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69310⟩⟩) 0 ⟨65839⟩ 80017

def event80019 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69310⟩⟩) 1 ⟨69309⟩ 80002

def event80020 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69310⟩⟩) (.sum [.predecessor 0 80018 .coefficient, .predecessor 1 80019 .coefficient])

def exact80021RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨68566⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80021RawTermsValid :
    exact80021RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80021 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69310⟩⟩) exact80021RawTerms .large 80020 .exactZero (none)

def event80022 : Event := .preFoldPolynomial 80021 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨68566⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact80023RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨68566⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event80023 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨69310⟩⟩) 80022 exact80023RawTerms .large 80020 .exactZero (none)

def event80024 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨65609⟩⟩) ⟨⟨67⟩, ⟨46⟩, ⟨135⟩⟩ ⟨79858, 80024⟩

def event80025 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨67833⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67830⟩⟩]⟩) (1) 0 2 (.universal 80024 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67830⟩⟩]⟩) (none) 80023)

def event80026 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67833⟩⟩, .relation 80025 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩)

def event80027 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67833⟩⟩, .relation 80025 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩, (-1)⟩)

def event80028 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67833⟩⟩, .relation 80025 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨68566⟩⟩]⟩, (1)⟩)

def event80029 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67833⟩⟩, .relation 80025 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact80030RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨68566⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80030RawTermsValid :
    exact80030RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80030 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67833⟩⟩) exact80030RawTerms .large 79854 (.finite 202072841853861888) (some (79856))

def event80031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69308⟩⟩) 0 ⟨67833⟩ 80030

def event80032 : Event := .predecessor (⟨.program ⟨257⟩, ⟨69308⟩⟩) 1 ⟨69307⟩ 79844

def event80033 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69308⟩⟩) (.sum [.predecessor 0 80031 .coefficient, .predecessor 1 80032 .coefficient])

def event80034 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69308⟩⟩, .operator (⟨80030, 2⟩, ⟨79844, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], [⟨.program ⟨257⟩, ⟨68566⟩⟩]⟩, (-1)⟩)

def event80035 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨69308⟩⟩, .operator (⟨80030, 1⟩, ⟨79844, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69306⟩⟩]⟩, (1)⟩)

def event80036 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨69308⟩⟩) (.sum [.result 80030 .summary, .result 79844 .summary])

def exact80037RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact80037RawTermsValid :
    exact80037RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80037 : Event := .resultExact (⟨.program ⟨257⟩, ⟨69308⟩⟩) exact80037RawTerms .large 80033 (.finite 2998054127048462696448) (some (80036))

def event80038 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70653⟩⟩) 0 ⟨69308⟩ 80037

def event80039 : Event := .predecessor (⟨.program ⟨257⟩, ⟨70653⟩⟩) 1 ⟨70651⟩ 79760

def event80040 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70653⟩⟩) (.product (.predecessor 0 80038 .coefficient) (.predecessor 1 80039 .coefficient) (⟨false, false, none, none, none⟩))

def event80041 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70653⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨70651⟩⟩]⟩) [⟨.result 79760 .coefficient, false, none⟩])

def event80042 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨70653⟩⟩) (.product (.result 80037 .summary) (.transfer 80041) (⟨false, false, none, none, none⟩))

def event80043 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70653⟩⟩, .operator (⟨80037, 0⟩, ⟨79760, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70651⟩⟩]⟩, (1)⟩)

def event80044 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70653⟩⟩, .operator (⟨80037, 1⟩, ⟨79760, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70651⟩⟩]⟩, (-1)⟩)

def event80045 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨70653⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70651⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70651⟩⟩) ⟨68736⟩ 79757)

def event80046 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨70653⟩⟩, .relation 80045 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨68736⟩⟩]⟩, (-1)⟩)

def exact80047RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70651⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨65836⟩⟩], [⟨.program ⟨257⟩, ⟨68736⟩⟩]⟩, (-1)⟩]

theorem exact80047RawTermsValid :
    exact80047RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80047 : Event := .resultExact (⟨.program ⟨257⟩, ⟨70653⟩⟩) exact80047RawTerms .large 80040 (.finite 32191361068277440720800338411520) (some (80042))

def event80048 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68197⟩⟩) 0 ⟨65837⟩ 3287

def event80049 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68197⟩⟩) (.authority (.relationPreimageSource ⟨76⟩))

def exact80050RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68197⟩⟩]⟩, (1)⟩]

theorem exact80050RawTermsValid :
    exact80050RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80050 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68197⟩⟩) exact80050RawTerms (.finite 5647228698) 80049 .exactZero (none)

def event80051 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68199⟩⟩) 0 ⟨68197⟩ 80050

def event80052 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68199⟩⟩) 1 ⟨2370⟩ 4

def event80053 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68199⟩⟩) (.scale (.predecessor 0 80051 .coefficient) (.value (.predecessor 1 80052 .coefficient)))

def exact80054RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68197⟩⟩]⟩, (1)⟩]

theorem exact80054RawTermsValid :
    exact80054RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80054 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68199⟩⟩) exact80054RawTerms (.finite 5647228698) 80053 .exactZero (none)

def event80055 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68200⟩⟩) 0 ⟨10368⟩ 75995

def event80056 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68200⟩⟩) 1 ⟨68199⟩ 80054

def event80057 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68200⟩⟩) (.product (.predecessor 0 80055 .coefficient) (.predecessor 1 80056 .coefficient) (⟨false, false, none, none, none⟩))

def event80058 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68200⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨68197⟩⟩]⟩) [⟨.result 80050 .coefficient, false, none⟩])

def event80059 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68200⟩⟩) (.product (.result 75995 .summary) (.transfer 80058) (⟨false, false, none, none, none⟩))

def event80060 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68200⟩⟩, .operator (⟨75995, 0⟩, ⟨80054, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68197⟩⟩]⟩, (1)⟩)

def event80061 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨68198⟩⟩)

def event80062 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event80063 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event80064 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event80065 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event80066 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event80067 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event80068 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event80069 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event80070 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 80069

def event80071 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 80067

def event80072 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 80070 .coefficient) (.value (.predecessor 1 80071 .coefficient)))

def event80073 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event80074 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 80073

def event80075 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 80065

def event80076 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 80074 .coefficient, .predecessor 1 80075 .coefficient])

def event80077 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event80078 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 80077

def event80079 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 80063

def event80080 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 80079 .coefficient))

def event80081 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event80082 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25802⟩⟩) 0 ⟨10325⟩ 80081

def event80083 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25802⟩⟩) (.authority (.programFamilyFact))

def exact80084RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩], []⟩, (1)⟩]

theorem exact80084RawTermsValid :
    exact80084RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80084 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25802⟩⟩) exact80084RawTerms (.finite 28) 80083 .exactZero (none)

def event80085 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65607⟩⟩) 0 ⟨10325⟩ 80081

def event80086 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65607⟩⟩) (.authority (.programFamilyFact))

def exact80087RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65607⟩⟩], []⟩, (1)⟩]

theorem exact80087RawTermsValid :
    exact80087RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80087 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65607⟩⟩) exact80087RawTerms (.finite 28) 80086 .exactZero (none)

def event80088 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65608⟩⟩) 0 ⟨65607⟩ 80087

def event80089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65608⟩⟩) 1 ⟨25802⟩ 80084

def event80090 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65608⟩⟩) (.product (.predecessor 0 80088 .coefficient) (.predecessor 1 80089 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event80091 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65608⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25802⟩⟩, ⟨.program ⟨257⟩, ⟨65607⟩⟩], []⟩) [⟨.result 80087 .coefficient, true, some 1⟩, ⟨.result 80084 .coefficient, true, some 1⟩])

def event80092 : Event := .survivorFold (1) 80091

def exact80093RawTerms : List Term := []

theorem exact80093RawTermsValid :
    exact80093RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80093 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65608⟩⟩) exact80093RawTerms (.finite 784) 80090 (.finite 784) (some (80091))

def event80094 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65609⟩⟩) 0 ⟨65608⟩ 80093

def event80095 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65609⟩⟩) (.identity (.predecessor 0 80094 .coefficient))

def event80096 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65609⟩⟩) (.finite 784)

def event80097 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65836⟩⟩) 0 ⟨65609⟩ 80096

def event80098 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65836⟩⟩) (.authority (.programFamilyFact))

def exact80099RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨65836⟩⟩], []⟩, (1)⟩]

theorem exact80099RawTermsValid :
    exact80099RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80099 : Event := .resultExact (⟨.program ⟨257⟩, ⟨65836⟩⟩) exact80099RawTerms (.finite 28) 80098 .exactZero (none)

def event80100 : Event := .predecessor (⟨.program ⟨257⟩, ⟨65837⟩⟩) 0 ⟨65836⟩ 80099

def event80101 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨65837⟩⟩) (.identity (.predecessor 0 80100 .coefficient))

def event80102 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨65837⟩⟩) (.finite 28)

def event80103 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68197⟩⟩) 0 ⟨65837⟩ 80102

def event80104 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68197⟩⟩) (.authority (.relationPreimageSource ⟨76⟩))

def exact80105RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨68197⟩⟩]⟩, (1)⟩]

theorem exact80105RawTermsValid :
    exact80105RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80105 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68197⟩⟩) exact80105RawTerms (.finite 5647228698) 80104 .exactZero (none)

def event80106 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact80107RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact80107RawTermsValid :
    exact80107RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80107 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact80107RawTerms .large 80106 .exactZero (none)

def event80108 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68198⟩⟩) 0 ⟨35⟩ 80107

def event80109 : Event := .predecessor (⟨.program ⟨257⟩, ⟨68198⟩⟩) 1 ⟨68197⟩ 80105

def event80110 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨68198⟩⟩) (.product (.predecessor 0 80108 .coefficient) (.predecessor 1 80109 .coefficient) (⟨false, false, none, none, none⟩))

def event80111 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨68198⟩⟩, .operator (⟨80107, 0⟩, ⟨80105, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68197⟩⟩]⟩, (1)⟩)

def exact80112RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68197⟩⟩]⟩, (1)⟩]

theorem exact80112RawTermsValid :
    exact80112RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event80112 : Event := .resultExact (⟨.program ⟨257⟩, ⟨68198⟩⟩) exact80112RawTerms .large 80110 .exactZero (none)

def event80113 : Event := .preFoldPolynomial 80112 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68197⟩⟩]⟩, (1)⟩] .exactZero none

def exact80114RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68197⟩⟩]⟩, (1)⟩]

def event80114 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨68198⟩⟩) 80113 exact80114RawTerms .large 80110 .exactZero (none)

def event80115 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨70664⟩⟩)

def event80116 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event80117 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event80118 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event80119 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event80120 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event80121 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event80122 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event80123 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event80124 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 80123

def event80125 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 80121

def event80126 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 80124 .coefficient) (.value (.predecessor 1 80125 .coefficient)))

def event80127 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def eventLeaf4992 : Array AnnotatedEvent := #[
  { event := event79872
    frameStart := 79858 },
  { event := event79873
    frameStart := 79858 },
  { event := event79874
    frameStart := 79858 },
  { event := event79875
    frameStart := 79858 },
  { event := event79876
    frameStart := 79858 },
  { event := event79877
    frameStart := 79858 },
  { event := event79878
    frameStart := 79858 },
  { event := event79879
    frameStart := 79858 },
  { event := event79880
    frameStart := 79858 },
  { event := event79881
    frameStart := 79858 },
  { event := event79882
    frameStart := 79858 },
  { event := event79883
    frameStart := 79858 },
  { event := event79884
    frameStart := 79858 },
  { event := event79885
    frameStart := 79858 },
  { event := event79886
    frameStart := 79858 },
  { event := event79887
    frameStart := 79858 }
]

def eventLeaf4993 : Array AnnotatedEvent := #[
  { event := event79888
    frameStart := 79858 },
  { event := event79889
    frameStart := 79858 },
  { event := event79890
    frameStart := 79858 },
  { event := event79891
    frameStart := 79858 },
  { event := event79892
    frameStart := 79858 },
  { event := event79893
    frameStart := 79858 },
  { event := event79894
    frameStart := 79858 },
  { event := event79895
    frameStart := 79858 },
  { event := event79896
    frameStart := 79858 },
  { event := event79897
    frameStart := 79858 },
  { event := event79898
    frameStart := 79858 },
  { event := event79899
    frameStart := 79858 },
  { event := event79900
    frameStart := 79858 },
  { event := event79901
    frameStart := 79858 },
  { event := event79902
    frameStart := 79858 },
  { event := event79903
    frameStart := 79858 }
]

def eventLeaf4994 : Array AnnotatedEvent := #[
  { event := event79904
    frameStart := 79858 },
  { event := event79905
    frameStart := 79858 },
  { event := event79906
    frameStart := 79906 },
  { event := event79907
    frameStart := 79906 },
  { event := event79908
    frameStart := 79906 },
  { event := event79909
    frameStart := 79906 },
  { event := event79910
    frameStart := 79906 },
  { event := event79911
    frameStart := 79906 },
  { event := event79912
    frameStart := 79906 },
  { event := event79913
    frameStart := 79906 },
  { event := event79914
    frameStart := 79906 },
  { event := event79915
    frameStart := 79906 },
  { event := event79916
    frameStart := 79906 },
  { event := event79917
    frameStart := 79906 },
  { event := event79918
    frameStart := 79906 },
  { event := event79919
    frameStart := 79906 }
]

def eventLeaf4995 : Array AnnotatedEvent := #[
  { event := event79920
    frameStart := 79906 },
  { event := event79921
    frameStart := 79906 },
  { event := event79922
    frameStart := 79906 },
  { event := event79923
    frameStart := 79906 },
  { event := event79924
    frameStart := 79906 },
  { event := event79925
    frameStart := 79906 },
  { event := event79926
    frameStart := 79906 },
  { event := event79927
    frameStart := 79906 },
  { event := event79928
    frameStart := 79906 },
  { event := event79929
    frameStart := 79906 },
  { event := event79930
    frameStart := 79906 },
  { event := event79931
    frameStart := 79906 },
  { event := event79932
    frameStart := 79906 },
  { event := event79933
    frameStart := 79906 },
  { event := event79934
    frameStart := 79906 },
  { event := event79935
    frameStart := 79906 }
]

def eventLeaf4996 : Array AnnotatedEvent := #[
  { event := event79936
    frameStart := 79906 },
  { event := event79937
    frameStart := 79906 },
  { event := event79938
    frameStart := 79906 },
  { event := event79939
    frameStart := 79906 },
  { event := event79940
    frameStart := 79906 },
  { event := event79941
    frameStart := 79906 },
  { event := event79942
    frameStart := 79906 },
  { event := event79943
    frameStart := 79906 },
  { event := event79944
    frameStart := 79906 },
  { event := event79945
    frameStart := 79906 },
  { event := event79946
    frameStart := 79906 },
  { event := event79947
    frameStart := 79906 },
  { event := event79948
    frameStart := 79906 },
  { event := event79949
    frameStart := 79906 },
  { event := event79950
    frameStart := 79906 },
  { event := event79951
    frameStart := 79906 }
]

def eventLeaf4997 : Array AnnotatedEvent := #[
  { event := event79952
    frameStart := 79906 },
  { event := event79953
    frameStart := 79906 },
  { event := event79954
    frameStart := 79906 },
  { event := event79955
    frameStart := 79906 },
  { event := event79956
    frameStart := 79906 },
  { event := event79957
    frameStart := 79906 },
  { event := event79958
    frameStart := 79906 },
  { event := event79959
    frameStart := 79906 },
  { event := event79960
    frameStart := 79906 },
  { event := event79961
    frameStart := 79906 },
  { event := event79962
    frameStart := 79906 },
  { event := event79963
    frameStart := 79906 },
  { event := event79964
    frameStart := 79906 },
  { event := event79965
    frameStart := 79906 },
  { event := event79966
    frameStart := 79906 },
  { event := event79967
    frameStart := 79906 }
]

def eventLeaf4998 : Array AnnotatedEvent := #[
  { event := event79968
    frameStart := 79906 },
  { event := event79969
    frameStart := 79906 },
  { event := event79970
    frameStart := 79906 },
  { event := event79971
    frameStart := 79906 },
  { event := event79972
    frameStart := 79906 },
  { event := event79973
    frameStart := 79906 },
  { event := event79974
    frameStart := 79906 },
  { event := event79975
    frameStart := 79906 },
  { event := event79976
    frameStart := 79906 },
  { event := event79977
    frameStart := 79906 },
  { event := event79978
    frameStart := 79906 },
  { event := event79979
    frameStart := 79906 },
  { event := event79980
    frameStart := 79906 },
  { event := event79981
    frameStart := 79906 },
  { event := event79982
    frameStart := 79906 },
  { event := event79983
    frameStart := 79906 }
]

def eventLeaf4999 : Array AnnotatedEvent := #[
  { event := event79984
    frameStart := 79906 },
  { event := event79985
    frameStart := 79906 },
  { event := event79986
    frameStart := 79906 },
  { event := event79987
    frameStart := 79906 },
  { event := event79988
    frameStart := 79906 },
  { event := event79989
    frameStart := 79906 },
  { event := event79990
    frameStart := 79906 },
  { event := event79991
    frameStart := 79906 },
  { event := event79992
    frameStart := 79906 },
  { event := event79993
    frameStart := 79906 },
  { event := event79994
    frameStart := 79906 },
  { event := event79995
    frameStart := 79906 },
  { event := event79996
    frameStart := 79906 },
  { event := event79997
    frameStart := 79906 },
  { event := event79998
    frameStart := 79906 },
  { event := event79999
    frameStart := 79906 }
]

def eventLeaf5000 : Array AnnotatedEvent := #[
  { event := event80000
    frameStart := 79906 },
  { event := event80001
    frameStart := 79906 },
  { event := event80002
    frameStart := 79906 },
  { event := event80003
    frameStart := 79906 },
  { event := event80004
    frameStart := 79906 },
  { event := event80005
    frameStart := 79906 },
  { event := event80006
    frameStart := 79906 },
  { event := event80007
    frameStart := 79906 },
  { event := event80008
    frameStart := 79906 },
  { event := event80009
    frameStart := 79906 },
  { event := event80010
    frameStart := 79906 },
  { event := event80011
    frameStart := 79906 },
  { event := event80012
    frameStart := 79906 },
  { event := event80013
    frameStart := 79906 },
  { event := event80014
    frameStart := 79906 },
  { event := event80015
    frameStart := 79906 }
]

def eventLeaf5001 : Array AnnotatedEvent := #[
  { event := event80016
    frameStart := 79906 },
  { event := event80017
    frameStart := 79906 },
  { event := event80018
    frameStart := 79906 },
  { event := event80019
    frameStart := 79906 },
  { event := event80020
    frameStart := 79906 },
  { event := event80021
    frameStart := 79906 },
  { event := event80022
    frameStart := 79906 },
  { event := event80023
    frameStart := 79906 },
  { event := event80024
    frameStart := 0 },
  { event := event80025
    frameStart := 0 },
  { event := event80026
    frameStart := 0 },
  { event := event80027
    frameStart := 0 },
  { event := event80028
    frameStart := 0 },
  { event := event80029
    frameStart := 0 },
  { event := event80030
    frameStart := 0 },
  { event := event80031
    frameStart := 0 }
]

def eventLeaf5002 : Array AnnotatedEvent := #[
  { event := event80032
    frameStart := 0 },
  { event := event80033
    frameStart := 0 },
  { event := event80034
    frameStart := 0 },
  { event := event80035
    frameStart := 0 },
  { event := event80036
    frameStart := 0 },
  { event := event80037
    frameStart := 0 },
  { event := event80038
    frameStart := 0 },
  { event := event80039
    frameStart := 0 },
  { event := event80040
    frameStart := 0 },
  { event := event80041
    frameStart := 0 },
  { event := event80042
    frameStart := 0 },
  { event := event80043
    frameStart := 0 },
  { event := event80044
    frameStart := 0 },
  { event := event80045
    frameStart := 0 },
  { event := event80046
    frameStart := 0 },
  { event := event80047
    frameStart := 0 }
]

def eventLeaf5003 : Array AnnotatedEvent := #[
  { event := event80048
    frameStart := 0 },
  { event := event80049
    frameStart := 0 },
  { event := event80050
    frameStart := 0 },
  { event := event80051
    frameStart := 0 },
  { event := event80052
    frameStart := 0 },
  { event := event80053
    frameStart := 0 },
  { event := event80054
    frameStart := 0 },
  { event := event80055
    frameStart := 0 },
  { event := event80056
    frameStart := 0 },
  { event := event80057
    frameStart := 0 },
  { event := event80058
    frameStart := 0 },
  { event := event80059
    frameStart := 0 },
  { event := event80060
    frameStart := 0 },
  { event := event80061
    frameStart := 80061 },
  { event := event80062
    frameStart := 80061 },
  { event := event80063
    frameStart := 80061 }
]

def eventLeaf5004 : Array AnnotatedEvent := #[
  { event := event80064
    frameStart := 80061 },
  { event := event80065
    frameStart := 80061 },
  { event := event80066
    frameStart := 80061 },
  { event := event80067
    frameStart := 80061 },
  { event := event80068
    frameStart := 80061 },
  { event := event80069
    frameStart := 80061 },
  { event := event80070
    frameStart := 80061 },
  { event := event80071
    frameStart := 80061 },
  { event := event80072
    frameStart := 80061 },
  { event := event80073
    frameStart := 80061 },
  { event := event80074
    frameStart := 80061 },
  { event := event80075
    frameStart := 80061 },
  { event := event80076
    frameStart := 80061 },
  { event := event80077
    frameStart := 80061 },
  { event := event80078
    frameStart := 80061 },
  { event := event80079
    frameStart := 80061 }
]

def eventLeaf5005 : Array AnnotatedEvent := #[
  { event := event80080
    frameStart := 80061 },
  { event := event80081
    frameStart := 80061 },
  { event := event80082
    frameStart := 80061 },
  { event := event80083
    frameStart := 80061 },
  { event := event80084
    frameStart := 80061 },
  { event := event80085
    frameStart := 80061 },
  { event := event80086
    frameStart := 80061 },
  { event := event80087
    frameStart := 80061 },
  { event := event80088
    frameStart := 80061 },
  { event := event80089
    frameStart := 80061 },
  { event := event80090
    frameStart := 80061 },
  { event := event80091
    frameStart := 80061 },
  { event := event80092
    frameStart := 80061 },
  { event := event80093
    frameStart := 80061 },
  { event := event80094
    frameStart := 80061 },
  { event := event80095
    frameStart := 80061 }
]

def eventLeaf5006 : Array AnnotatedEvent := #[
  { event := event80096
    frameStart := 80061 },
  { event := event80097
    frameStart := 80061 },
  { event := event80098
    frameStart := 80061 },
  { event := event80099
    frameStart := 80061 },
  { event := event80100
    frameStart := 80061 },
  { event := event80101
    frameStart := 80061 },
  { event := event80102
    frameStart := 80061 },
  { event := event80103
    frameStart := 80061 },
  { event := event80104
    frameStart := 80061 },
  { event := event80105
    frameStart := 80061 },
  { event := event80106
    frameStart := 80061 },
  { event := event80107
    frameStart := 80061 },
  { event := event80108
    frameStart := 80061 },
  { event := event80109
    frameStart := 80061 },
  { event := event80110
    frameStart := 80061 },
  { event := event80111
    frameStart := 80061 }
]

def eventLeaf5007 : Array AnnotatedEvent := #[
  { event := event80112
    frameStart := 80061 },
  { event := event80113
    frameStart := 80061 },
  { event := event80114
    frameStart := 80061 },
  { event := event80115
    frameStart := 80115 },
  { event := event80116
    frameStart := 80115 },
  { event := event80117
    frameStart := 80115 },
  { event := event80118
    frameStart := 80115 },
  { event := event80119
    frameStart := 80115 },
  { event := event80120
    frameStart := 80115 },
  { event := event80121
    frameStart := 80115 },
  { event := event80122
    frameStart := 80115 },
  { event := event80123
    frameStart := 80115 },
  { event := event80124
    frameStart := 80115 },
  { event := event80125
    frameStart := 80115 },
  { event := event80126
    frameStart := 80115 },
  { event := event80127
    frameStart := 80115 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events312
