import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1105

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event282880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event282881 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event282882 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 282881

def event282883 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 282879

def event282884 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 282882 .coefficient) (.value (.predecessor 1 282883 .coefficient)))

def event282885 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event282886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 282885

def event282887 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 282877

def event282888 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 282886 .coefficient, .predecessor 1 282887 .coefficient])

def event282889 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event282890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 282889

def event282891 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 282875

def event282892 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 282891 .coefficient))

def event282893 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event282894 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36970⟩⟩) 0 ⟨5487⟩ 282893

def event282895 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36970⟩⟩) (.authority (.programFamilyFact))

def exact282896RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨36970⟩⟩], []⟩, (1)⟩]

theorem exact282896RawTermsValid :
    exact282896RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282896 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36970⟩⟩) exact282896RawTerms (.finite 42) 282895 .exactZero (none)

def event282897 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13791⟩⟩) 0 ⟨5487⟩ 282893

def event282898 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13791⟩⟩) (.authority (.programFamilyFact))

def exact282899RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13791⟩⟩], []⟩, (1)⟩]

theorem exact282899RawTermsValid :
    exact282899RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282899 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13791⟩⟩) exact282899RawTerms (.finite 42) 282898 .exactZero (none)

def event282900 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36971⟩⟩) 0 ⟨13791⟩ 282899

def event282901 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36971⟩⟩) 1 ⟨36970⟩ 282896

def event282902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36971⟩⟩) (.product (.predecessor 0 282900 .coefficient) (.predecessor 1 282901 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event282903 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36971⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13791⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], []⟩) [⟨.result 282899 .coefficient, true, some 1⟩, ⟨.result 282896 .coefficient, true, some 1⟩])

def event282904 : Event := .survivorFold (1) 282903

def exact282905RawTerms : List Term := []

theorem exact282905RawTermsValid :
    exact282905RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282905 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36971⟩⟩) exact282905RawTerms (.finite 1764) 282902 (.finite 1764) (some (282903))

def event282906 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36972⟩⟩) 0 ⟨36971⟩ 282905

def event282907 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36972⟩⟩) (.identity (.predecessor 0 282906 .coefficient))

def event282908 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨36972⟩⟩) (.finite 1764)

def event282909 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37380⟩⟩) 0 ⟨36972⟩ 282908

def event282910 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37380⟩⟩) (.authority (.programFamilyFact))

def exact282911RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37380⟩⟩], []⟩, (1)⟩]

theorem exact282911RawTermsValid :
    exact282911RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282911 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37380⟩⟩) exact282911RawTerms (.finite 42) 282910 .exactZero (none)

def event282912 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37381⟩⟩) 0 ⟨37380⟩ 282911

def event282913 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37381⟩⟩) (.identity (.predecessor 0 282912 .coefficient))

def event282914 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37381⟩⟩) (.finite 42)

def event282915 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38056⟩⟩) 0 ⟨37381⟩ 282914

def event282916 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38056⟩⟩) (.authority (.relationPreimageSource ⟨85⟩))

def exact282917RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38056⟩⟩]⟩, (1)⟩]

theorem exact282917RawTermsValid :
    exact282917RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282917 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38056⟩⟩) exact282917RawTerms (.finite 5647228698) 282916 .exactZero (none)

def event282918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact282919RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact282919RawTermsValid :
    exact282919RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282919 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact282919RawTerms .large 282918 .exactZero (none)

def event282920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38057⟩⟩) 0 ⟨35⟩ 282919

def event282921 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38057⟩⟩) 1 ⟨38056⟩ 282917

def event282922 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38057⟩⟩) (.product (.predecessor 0 282920 .coefficient) (.predecessor 1 282921 .coefficient) (⟨false, false, none, none, none⟩))

def event282923 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38057⟩⟩, .operator (⟨282919, 0⟩, ⟨282917, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38056⟩⟩]⟩, (1)⟩)

def exact282924RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38056⟩⟩]⟩, (1)⟩]

theorem exact282924RawTermsValid :
    exact282924RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282924 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38057⟩⟩) exact282924RawTerms .large 282922 .exactZero (none)

def event282925 : Event := .preFoldPolynomial 282924 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38056⟩⟩]⟩, (1)⟩] .exactZero none

def exact282926RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38056⟩⟩]⟩, (1)⟩]

def event282926 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨38057⟩⟩) 282925 exact282926RawTerms .large 282922 .exactZero (none)

def event282927 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨39163⟩⟩)

def event282928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event282929 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event282930 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event282931 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event282932 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event282933 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event282934 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event282935 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event282936 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 282935

def event282937 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 282933

def event282938 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 282936 .coefficient) (.value (.predecessor 1 282937 .coefficient)))

def event282939 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event282940 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 0 ⟨392⟩ 282939

def event282941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2376⟩⟩) 1 ⟨2370⟩ 282931

def event282942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.sum [.predecessor 0 282940 .coefficient, .predecessor 1 282941 .coefficient])

def event282943 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2376⟩⟩) (.finite 655341)

def event282944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 0 ⟨2376⟩ 282943

def event282945 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5487⟩⟩) 1 ⟨5426⟩ 282929

def event282946 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.identity (.predecessor 1 282945 .coefficient))

def event282947 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5487⟩⟩) (.finite 655360)

def event282948 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36970⟩⟩) 0 ⟨5487⟩ 282947

def event282949 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36970⟩⟩) (.authority (.programFamilyFact))

def exact282950RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨36970⟩⟩], []⟩, (1)⟩]

theorem exact282950RawTermsValid :
    exact282950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282950 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36970⟩⟩) exact282950RawTerms (.finite 42) 282949 .exactZero (none)

def event282951 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13791⟩⟩) 0 ⟨5487⟩ 282947

def event282952 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13791⟩⟩) (.authority (.programFamilyFact))

def exact282953RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13791⟩⟩], []⟩, (1)⟩]

theorem exact282953RawTermsValid :
    exact282953RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282953 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13791⟩⟩) exact282953RawTerms (.finite 42) 282952 .exactZero (none)

def event282954 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36971⟩⟩) 0 ⟨13791⟩ 282953

def event282955 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36971⟩⟩) 1 ⟨36970⟩ 282950

def event282956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36971⟩⟩) (.product (.predecessor 0 282954 .coefficient) (.predecessor 1 282955 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event282957 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36971⟩⟩, .operator (⟨282953, 0⟩, ⟨282950, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨13791⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], []⟩, (1)⟩)

def exact282958RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨13791⟩⟩, ⟨.program ⟨257⟩, ⟨36970⟩⟩], []⟩, (1)⟩]

theorem exact282958RawTermsValid :
    exact282958RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282958 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36971⟩⟩) exact282958RawTerms (.finite 1764) 282956 .exactZero (none)

def event282959 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36972⟩⟩) 0 ⟨36971⟩ 282958

def event282960 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36972⟩⟩) (.identity (.predecessor 0 282959 .coefficient))

def event282961 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨36972⟩⟩) (.finite 1764)

def event282962 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37380⟩⟩) 0 ⟨36972⟩ 282961

def event282963 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37380⟩⟩) (.authority (.programFamilyFact))

def exact282964RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37380⟩⟩], []⟩, (1)⟩]

theorem exact282964RawTermsValid :
    exact282964RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282964 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37380⟩⟩) exact282964RawTerms (.finite 42) 282963 .exactZero (none)

def event282965 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37381⟩⟩) 0 ⟨37380⟩ 282964

def event282966 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37381⟩⟩) (.identity (.predecessor 0 282965 .coefficient))

def event282967 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨37381⟩⟩) (.finite 42)

def event282968 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38525⟩⟩) 0 ⟨37381⟩ 282967

def event282969 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38525⟩⟩) (.authority (.programFamilyFact))

def event282970 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38525⟩⟩) (.finite 3720)

def event282971 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event282972 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38527⟩⟩) 0 ⟨7177⟩ 282971

def event282973 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38527⟩⟩) 1 ⟨38525⟩ 282970

def event282974 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38527⟩⟩) (.authority (.operator))

def exact282975RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨38527⟩⟩]⟩, (1)⟩]

theorem exact282975RawTermsValid :
    exact282975RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282975 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38527⟩⟩) exact282975RawTerms .large 282974 .exactZero (none)

def event282976 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39159⟩⟩) 0 ⟨38527⟩ 282975

def event282977 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39159⟩⟩) (.authority (.operator))

def exact282978RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩, (1)⟩]

theorem exact282978RawTermsValid :
    exact282978RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282978 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39159⟩⟩) exact282978RawTerms (.finite 8192) 282977 .exactZero (none)

def event282979 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event282980 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event282981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38762⟩⟩) 0 ⟨37381⟩ 282967

def event282982 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38762⟩⟩) 1 ⟨136⟩ 282980

def event282983 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38762⟩⟩) (.sum [.predecessor 0 282981 .coefficient, .predecessor 1 282982 .coefficient])

def event282984 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨38762⟩⟩) (.finite 42)

def event282985 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38763⟩⟩) 0 ⟨38762⟩ 282984

def event282986 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38763⟩⟩) (.identity (.predecessor 0 282985 .coefficient))

def exact282987RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37380⟩⟩], []⟩, (1)⟩]

theorem exact282987RawTermsValid :
    exact282987RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282987 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38763⟩⟩) exact282987RawTerms (.finite 42) 282986 .exactZero (none)

def event282988 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact282989RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact282989RawTermsValid :
    exact282989RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282989 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact282989RawTerms .large 282988 .exactZero (none)

def event282990 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38764⟩⟩) 0 ⟨6908⟩ 282989

def event282991 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38764⟩⟩) 1 ⟨38763⟩ 282987

def event282992 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38764⟩⟩) (.product (.predecessor 0 282990 .coefficient) (.predecessor 1 282991 .coefficient) (⟨false, false, none, none, none⟩))

def event282993 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38764⟩⟩, .operator (⟨282989, 0⟩, ⟨282987, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact282994RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact282994RawTermsValid :
    exact282994RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282994 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38764⟩⟩) exact282994RawTerms .large 282992 .exactZero (none)

def event282995 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7192⟩⟩) 0 ⟨7177⟩ 282971

def event282996 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7192⟩⟩) (.authority (.operator))

def exact282997RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩]

theorem exact282997RawTermsValid :
    exact282997RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event282997 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7192⟩⟩) exact282997RawTerms .large 282996 .exactZero (none)

def event282998 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38765⟩⟩) 0 ⟨7192⟩ 282997

def event282999 : Event := .predecessor (⟨.program ⟨257⟩, ⟨38765⟩⟩) 1 ⟨38764⟩ 282994

def event283000 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨38765⟩⟩) (.sum [.predecessor 0 282998 .coefficient, .predecessor 1 282999 .coefficient])

def exact283001RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283001RawTermsValid :
    exact283001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283001 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38765⟩⟩) exact283001RawTerms .large 283000 .exactZero (none)

def event283002 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39160⟩⟩) 0 ⟨38765⟩ 283001

def event283003 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39160⟩⟩) 1 ⟨39159⟩ 282978

def event283004 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39160⟩⟩) (.product (.predecessor 0 283002 .coefficient) (.predecessor 1 283003 .coefficient) (⟨false, false, none, none, none⟩))

def event283005 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39160⟩⟩, .operator (⟨283001, 0⟩, ⟨282978, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩, (1)⟩)

def event283006 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39160⟩⟩, .operator (⟨283001, 1⟩, ⟨282978, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩, (-1)⟩)

def event283007 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨39160⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39159⟩⟩) ⟨38527⟩ 282975)

def event283008 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39160⟩⟩, .relation 283007 0, ⟨[⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨38527⟩⟩]⟩, (-1)⟩)

def exact283009RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨38527⟩⟩]⟩, (-1)⟩]

theorem exact283009RawTermsValid :
    exact283009RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283009 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39160⟩⟩) exact283009RawTerms .large 283004 .exactZero (none)

def event283010 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37565⟩⟩) 0 ⟨37381⟩ 282967

def event283011 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37565⟩⟩) (.authority (.programFamilyFact))

def exact283012RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], []⟩, (1)⟩]

theorem exact283012RawTermsValid :
    exact283012RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283012 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37565⟩⟩) exact283012RawTerms (.finite 63) 283011 .exactZero (none)

def event283013 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37566⟩⟩) 0 ⟨6908⟩ 282989

def event283014 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37566⟩⟩) 1 ⟨37565⟩ 283012

def event283015 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37566⟩⟩) (.product (.predecessor 0 283013 .coefficient) (.predecessor 1 283014 .coefficient) (⟨false, true, none, none, some 1⟩))

def event283016 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37566⟩⟩, .operator (⟨282989, 0⟩, ⟨283012, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact283017RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact283017RawTermsValid :
    exact283017RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283017 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37566⟩⟩) exact283017RawTerms .large 283015 .exactZero (none)

def event283018 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7224⟩⟩) 0 ⟨7177⟩ 282971

def event283019 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7224⟩⟩) (.authority (.operator))

def exact283020RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩]

theorem exact283020RawTermsValid :
    exact283020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283020 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7224⟩⟩) exact283020RawTerms .large 283019 .exactZero (none)

def event283021 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37567⟩⟩) 0 ⟨7224⟩ 283020

def event283022 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37567⟩⟩) 1 ⟨37566⟩ 283017

def event283023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37567⟩⟩) (.sum [.predecessor 0 283021 .coefficient, .predecessor 1 283022 .coefficient])

def exact283024RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283024RawTermsValid :
    exact283024RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283024 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37567⟩⟩) exact283024RawTerms .large 283023 .exactZero (none)

def event283025 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39163⟩⟩) 0 ⟨37567⟩ 283024

def event283026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39163⟩⟩) 1 ⟨39160⟩ 283009

def event283027 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39163⟩⟩) (.sum [.predecessor 0 283025 .coefficient, .predecessor 1 283026 .coefficient])

def exact283028RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨38527⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283028RawTermsValid :
    exact283028RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283028 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39163⟩⟩) exact283028RawTerms .large 283027 .exactZero (none)

def event283029 : Event := .preFoldPolynomial 283028 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨38527⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact283030RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨38527⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event283030 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨39163⟩⟩) 283029 exact283030RawTerms .large 283027 .exactZero (none)

def event283031 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨37381⟩⟩) ⟨⟨103⟩, ⟨85⟩, ⟨135⟩⟩ ⟨282873, 283031⟩

def event283032 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨38059⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38056⟩⟩]⟩) (1) 0 2 (.universal 283031 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38056⟩⟩]⟩) (none) 283030)

def event283033 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38059⟩⟩, .relation 283032 1, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩)

def event283034 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38059⟩⟩, .relation 283032 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩, (-1)⟩)

def event283035 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38059⟩⟩, .relation 283032 2, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨38527⟩⟩]⟩, (1)⟩)

def event283036 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨38059⟩⟩, .relation 283032 3, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact283037RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨38527⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283037RawTermsValid :
    exact283037RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283037 : Event := .resultExact (⟨.program ⟨257⟩, ⟨38059⟩⟩) exact283037RawTerms .large 282869 (.finite 202072841853861888) (some (282871))

def event283038 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39162⟩⟩) 0 ⟨38059⟩ 283037

def event283039 : Event := .predecessor (⟨.program ⟨257⟩, ⟨39162⟩⟩) 1 ⟨39161⟩ 282859

def event283040 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39162⟩⟩) (.sum [.predecessor 0 283038 .coefficient, .predecessor 1 283039 .coefficient])

def event283041 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39162⟩⟩, .operator (⟨283037, 0⟩, ⟨282859, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39159⟩⟩]⟩, (1)⟩)

def event283042 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨39162⟩⟩, .operator (⟨283037, 2⟩, ⟨282859, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37380⟩⟩], [⟨.program ⟨257⟩, ⟨38527⟩⟩]⟩, (-1)⟩)

def event283043 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨39162⟩⟩) (.sum [.result 283037 .summary, .result 282859 .summary])

def exact283044RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283044RawTermsValid :
    exact283044RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283044 : Event := .resultExact (⟨.program ⟨257⟩, ⟨39162⟩⟩) exact283044RawTerms .large 283040 (.finite 32192736221397454434328420548608) (some (283043))

def event283045 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35845⟩⟩) 0 ⟨34701⟩ 13684

def event283046 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35845⟩⟩) (.authority (.programFamilyFact))

def event283047 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨35845⟩⟩) (.finite 3720)

def event283048 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35847⟩⟩) 0 ⟨7177⟩ 15500

def event283049 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35847⟩⟩) 1 ⟨35845⟩ 283047

def event283050 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35847⟩⟩) (.authority (.operator))

def exact283051RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35847⟩⟩]⟩, (1)⟩]

theorem exact283051RawTermsValid :
    exact283051RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283051 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35847⟩⟩) exact283051RawTerms .large 283050 .exactZero (none)

def event283052 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36479⟩⟩) 0 ⟨35847⟩ 283051

def event283053 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36479⟩⟩) (.authority (.operator))

def exact283054RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨36479⟩⟩]⟩, (1)⟩]

theorem exact283054RawTermsValid :
    exact283054RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283054 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36479⟩⟩) exact283054RawTerms (.finite 8192) 283053 .exactZero (none)

def event283055 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35712⟩⟩) 0 ⟨34292⟩ 13678

def event283056 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35712⟩⟩) (.authority (.programFamilyFact))

def event283057 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨35712⟩⟩) (.finite 3720)

def event283058 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35713⟩⟩) 0 ⟨7177⟩ 15500

def event283059 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35713⟩⟩) 1 ⟨35712⟩ 283057

def event283060 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35713⟩⟩) (.authority (.operator))

def exact283061RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35713⟩⟩]⟩, (1)⟩]

theorem exact283061RawTermsValid :
    exact283061RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283061 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35713⟩⟩) exact283061RawTerms .large 283060 .exactZero (none)

def event283062 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36193⟩⟩) 0 ⟨35713⟩ 283061

def event283063 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36193⟩⟩) (.authority (.operator))

def exact283064RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨36193⟩⟩]⟩, (1)⟩]

theorem exact283064RawTermsValid :
    exact283064RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283064 : Event := .resultExact (⟨.program ⟨257⟩, ⟨36193⟩⟩) exact283064RawTerms (.finite 8192) 283063 .exactZero (none)

def event283065 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34293⟩⟩) 0 ⟨34290⟩ 13667

def event283066 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34293⟩⟩) 1 ⟨6922⟩ 280653

def event283067 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34293⟩⟩) (.tensor (.predecessor 0 283065 .coefficient) (.predecessor 1 283066 .coefficient) true false)

def event283068 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34293⟩⟩, .operator (⟨13667, 0⟩, ⟨280653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact283069RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact283069RawTermsValid :
    exact283069RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283069 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34293⟩⟩) exact283069RawTerms .large 283067 .exactZero (none)

def event283070 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7902⟩⟩) 0 ⟨5489⟩ 280523

def event283071 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7902⟩⟩) 1 ⟨7280⟩ 19585

def event283072 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7902⟩⟩) (.product (.predecessor 0 283070 .coefficient) (.predecessor 1 283071 .coefficient) (⟨false, false, none, none, none⟩))

def event283073 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7902⟩⟩, .operator (⟨280523, 0⟩, ⟨19585, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (1)⟩)

def exact283074RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (1)⟩]

theorem exact283074RawTermsValid :
    exact283074RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283074 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7902⟩⟩) exact283074RawTerms .large 283072 .exactZero (none)

def event283075 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34294⟩⟩) 0 ⟨7902⟩ 283074

def event283076 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34294⟩⟩) 1 ⟨34293⟩ 283069

def event283077 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34294⟩⟩) (.sum [.predecessor 0 283075 .coefficient, .predecessor 1 283076 .coefficient])

def exact283078RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283078RawTermsValid :
    exact283078RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283078 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34294⟩⟩) exact283078RawTerms .large 283077 .exactZero (none)

def event283079 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34295⟩⟩) 0 ⟨34294⟩ 283078

def event283080 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34295⟩⟩) 1 ⟨106⟩ 19577

def event283081 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34295⟩⟩) (.sum [.predecessor 0 283079 .coefficient, .predecessor 1 283080 .coefficient])

def event283082 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34295⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨106⟩⟩]⟩) [⟨.result 19577 .coefficient, false, none⟩])

def event283083 : Event := .survivorFold (1) 283082

def exact283084RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283084RawTermsValid :
    exact283084RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283084 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34295⟩⟩) exact283084RawTerms .large 283081 (.finite 26) (some (283082))

def event283085 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34296⟩⟩) 0 ⟨34295⟩ 283084

def event283086 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34296⟩⟩) 1 ⟨13491⟩ 13670

def event283087 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34296⟩⟩) (.product (.predecessor 0 283085 .coefficient) (.predecessor 1 283086 .coefficient) (⟨false, true, none, none, some 1⟩))

def event283088 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34296⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨13491⟩⟩], []⟩) [⟨.result 13670 .coefficient, true, some 1⟩])

def event283089 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34296⟩⟩) (.product (.result 283084 .summary) (.transfer 283088) (⟨false, false, none, none, none⟩))

def event283090 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34296⟩⟩, .operator (⟨283084, 1⟩, ⟨13670, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event283091 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34296⟩⟩, .operator (⟨283084, 0⟩, ⟨13670, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (1)⟩)

def exact283092RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283092RawTermsValid :
    exact283092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283092 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34296⟩⟩) exact283092RawTerms .large 283087 (.finite 34078720) (some (283089))

def event283093 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13492⟩⟩) 0 ⟨13491⟩ 13670

def event283094 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13492⟩⟩) 1 ⟨6922⟩ 280653

def event283095 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13492⟩⟩) (.tensor (.predecessor 0 283093 .coefficient) (.predecessor 1 283094 .coefficient) true false)

def event283096 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13492⟩⟩, .operator (⟨13670, 0⟩, ⟨280653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact283097RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact283097RawTermsValid :
    exact283097RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283097 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13492⟩⟩) exact283097RawTerms .large 283095 .exactZero (none)

def event283098 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7919⟩⟩) 0 ⟨5489⟩ 280523

def event283099 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7919⟩⟩) 1 ⟨7297⟩ 19626

def event283100 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7919⟩⟩) (.product (.predecessor 0 283098 .coefficient) (.predecessor 1 283099 .coefficient) (⟨false, false, none, none, none⟩))

def event283101 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨7919⟩⟩, .operator (⟨280523, 0⟩, ⟨19626, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩]⟩, (1)⟩)

def exact283102RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩]⟩, (1)⟩]

theorem exact283102RawTermsValid :
    exact283102RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283102 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7919⟩⟩) exact283102RawTerms .large 283100 .exactZero (none)

def event283103 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13493⟩⟩) 0 ⟨7919⟩ 283102

def event283104 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13493⟩⟩) 1 ⟨13492⟩ 283097

def event283105 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13493⟩⟩) (.sum [.predecessor 0 283103 .coefficient, .predecessor 1 283104 .coefficient])

def exact283106RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283106RawTermsValid :
    exact283106RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283106 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13493⟩⟩) exact283106RawTerms .large 283105 .exactZero (none)

def event283107 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13494⟩⟩) 0 ⟨13493⟩ 283106

def event283108 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13494⟩⟩) 1 ⟨123⟩ 19618

def event283109 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13494⟩⟩) (.sum [.predecessor 0 283107 .coefficient, .predecessor 1 283108 .coefficient])

def event283110 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13494⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨123⟩⟩]⟩) [⟨.result 19618 .coefficient, false, none⟩])

def event283111 : Event := .survivorFold (1) 283110

def exact283112RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283112RawTermsValid :
    exact283112RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283112 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13494⟩⟩) exact283112RawTerms .large 283109 (.finite 26) (some (283110))

def event283113 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13495⟩⟩) 0 ⟨13494⟩ 283112

def event283114 : Event := .predecessor (⟨.program ⟨257⟩, ⟨13495⟩⟩) 1 ⟨9551⟩ 19615

def event283115 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13495⟩⟩) (.product (.predecessor 0 283113 .coefficient) (.predecessor 1 283114 .coefficient) (⟨false, false, none, none, none⟩))

def event283116 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13495⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) [⟨.result 19611 .coefficient, false, none⟩])

def event283117 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨13495⟩⟩) (.product (.result 283112 .summary) (.transfer 283116) (⟨false, false, none, none, none⟩))

def event283118 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13495⟩⟩, .operator (⟨283112, 1⟩, ⟨19615, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩, (-1)⟩)

def event283119 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨13495⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9550⟩⟩) ⟨7280⟩ 19585)

def event283120 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13495⟩⟩, .relation 283119 0, ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (-1)⟩)

def event283121 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨13495⟩⟩, .operator (⟨283112, 0⟩, ⟨19615, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩, (1)⟩)

def exact283122RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (-1)⟩]

theorem exact283122RawTermsValid :
    exact283122RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283122 : Event := .resultExact (⟨.program ⟨257⟩, ⟨13495⟩⟩) exact283122RawTerms .large 283115 (.finite 279172874240) (some (283117))

def event283123 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34297⟩⟩) 0 ⟨13495⟩ 283122

def event283124 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34297⟩⟩) 1 ⟨34296⟩ 283092

def event283125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34297⟩⟩) (.sum [.predecessor 0 283123 .coefficient, .predecessor 1 283124 .coefficient])

def event283126 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34297⟩⟩, .operator (⟨283122, 1⟩, ⟨283092, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩, (1)⟩)

def event283127 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34297⟩⟩) (.sum [.result 283122 .summary, .result 283092 .summary])

def exact283128RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact283128RawTermsValid :
    exact283128RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event283128 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34297⟩⟩) exact283128RawTerms .large 283125 (.finite 279206952960) (some (283127))

def event283129 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36194⟩⟩) 0 ⟨34297⟩ 283128

def event283130 : Event := .predecessor (⟨.program ⟨257⟩, ⟨36194⟩⟩) 1 ⟨36193⟩ 283064

def event283131 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36194⟩⟩) (.product (.predecessor 0 283129 .coefficient) (.predecessor 1 283130 .coefficient) (⟨false, false, none, none, none⟩))

def event283132 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36194⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨36193⟩⟩]⟩) [⟨.result 283064 .coefficient, false, none⟩])

def event283133 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨36194⟩⟩) (.product (.result 283128 .summary) (.transfer 283132) (⟨false, false, none, none, none⟩))

def event283134 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨36194⟩⟩, .operator (⟨283128, 1⟩, ⟨283064, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36193⟩⟩]⟩, (-1)⟩)

def event283135 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨36194⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨13491⟩⟩, ⟨.program ⟨257⟩, ⟨34290⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36193⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36193⟩⟩) ⟨35713⟩ 283061)

def eventLeaf17680 : Array AnnotatedEvent := #[
  { event := event282880
    frameStart := 282873 },
  { event := event282881
    frameStart := 282873 },
  { event := event282882
    frameStart := 282873 },
  { event := event282883
    frameStart := 282873 },
  { event := event282884
    frameStart := 282873 },
  { event := event282885
    frameStart := 282873 },
  { event := event282886
    frameStart := 282873 },
  { event := event282887
    frameStart := 282873 },
  { event := event282888
    frameStart := 282873 },
  { event := event282889
    frameStart := 282873 },
  { event := event282890
    frameStart := 282873 },
  { event := event282891
    frameStart := 282873 },
  { event := event282892
    frameStart := 282873 },
  { event := event282893
    frameStart := 282873 },
  { event := event282894
    frameStart := 282873 },
  { event := event282895
    frameStart := 282873 }
]

def eventLeaf17681 : Array AnnotatedEvent := #[
  { event := event282896
    frameStart := 282873 },
  { event := event282897
    frameStart := 282873 },
  { event := event282898
    frameStart := 282873 },
  { event := event282899
    frameStart := 282873 },
  { event := event282900
    frameStart := 282873 },
  { event := event282901
    frameStart := 282873 },
  { event := event282902
    frameStart := 282873 },
  { event := event282903
    frameStart := 282873 },
  { event := event282904
    frameStart := 282873 },
  { event := event282905
    frameStart := 282873 },
  { event := event282906
    frameStart := 282873 },
  { event := event282907
    frameStart := 282873 },
  { event := event282908
    frameStart := 282873 },
  { event := event282909
    frameStart := 282873 },
  { event := event282910
    frameStart := 282873 },
  { event := event282911
    frameStart := 282873 }
]

def eventLeaf17682 : Array AnnotatedEvent := #[
  { event := event282912
    frameStart := 282873 },
  { event := event282913
    frameStart := 282873 },
  { event := event282914
    frameStart := 282873 },
  { event := event282915
    frameStart := 282873 },
  { event := event282916
    frameStart := 282873 },
  { event := event282917
    frameStart := 282873 },
  { event := event282918
    frameStart := 282873 },
  { event := event282919
    frameStart := 282873 },
  { event := event282920
    frameStart := 282873 },
  { event := event282921
    frameStart := 282873 },
  { event := event282922
    frameStart := 282873 },
  { event := event282923
    frameStart := 282873 },
  { event := event282924
    frameStart := 282873 },
  { event := event282925
    frameStart := 282873 },
  { event := event282926
    frameStart := 282873 },
  { event := event282927
    frameStart := 282927 }
]

def eventLeaf17683 : Array AnnotatedEvent := #[
  { event := event282928
    frameStart := 282927 },
  { event := event282929
    frameStart := 282927 },
  { event := event282930
    frameStart := 282927 },
  { event := event282931
    frameStart := 282927 },
  { event := event282932
    frameStart := 282927 },
  { event := event282933
    frameStart := 282927 },
  { event := event282934
    frameStart := 282927 },
  { event := event282935
    frameStart := 282927 },
  { event := event282936
    frameStart := 282927 },
  { event := event282937
    frameStart := 282927 },
  { event := event282938
    frameStart := 282927 },
  { event := event282939
    frameStart := 282927 },
  { event := event282940
    frameStart := 282927 },
  { event := event282941
    frameStart := 282927 },
  { event := event282942
    frameStart := 282927 },
  { event := event282943
    frameStart := 282927 }
]

def eventLeaf17684 : Array AnnotatedEvent := #[
  { event := event282944
    frameStart := 282927 },
  { event := event282945
    frameStart := 282927 },
  { event := event282946
    frameStart := 282927 },
  { event := event282947
    frameStart := 282927 },
  { event := event282948
    frameStart := 282927 },
  { event := event282949
    frameStart := 282927 },
  { event := event282950
    frameStart := 282927 },
  { event := event282951
    frameStart := 282927 },
  { event := event282952
    frameStart := 282927 },
  { event := event282953
    frameStart := 282927 },
  { event := event282954
    frameStart := 282927 },
  { event := event282955
    frameStart := 282927 },
  { event := event282956
    frameStart := 282927 },
  { event := event282957
    frameStart := 282927 },
  { event := event282958
    frameStart := 282927 },
  { event := event282959
    frameStart := 282927 }
]

def eventLeaf17685 : Array AnnotatedEvent := #[
  { event := event282960
    frameStart := 282927 },
  { event := event282961
    frameStart := 282927 },
  { event := event282962
    frameStart := 282927 },
  { event := event282963
    frameStart := 282927 },
  { event := event282964
    frameStart := 282927 },
  { event := event282965
    frameStart := 282927 },
  { event := event282966
    frameStart := 282927 },
  { event := event282967
    frameStart := 282927 },
  { event := event282968
    frameStart := 282927 },
  { event := event282969
    frameStart := 282927 },
  { event := event282970
    frameStart := 282927 },
  { event := event282971
    frameStart := 282927 },
  { event := event282972
    frameStart := 282927 },
  { event := event282973
    frameStart := 282927 },
  { event := event282974
    frameStart := 282927 },
  { event := event282975
    frameStart := 282927 }
]

def eventLeaf17686 : Array AnnotatedEvent := #[
  { event := event282976
    frameStart := 282927 },
  { event := event282977
    frameStart := 282927 },
  { event := event282978
    frameStart := 282927 },
  { event := event282979
    frameStart := 282927 },
  { event := event282980
    frameStart := 282927 },
  { event := event282981
    frameStart := 282927 },
  { event := event282982
    frameStart := 282927 },
  { event := event282983
    frameStart := 282927 },
  { event := event282984
    frameStart := 282927 },
  { event := event282985
    frameStart := 282927 },
  { event := event282986
    frameStart := 282927 },
  { event := event282987
    frameStart := 282927 },
  { event := event282988
    frameStart := 282927 },
  { event := event282989
    frameStart := 282927 },
  { event := event282990
    frameStart := 282927 },
  { event := event282991
    frameStart := 282927 }
]

def eventLeaf17687 : Array AnnotatedEvent := #[
  { event := event282992
    frameStart := 282927 },
  { event := event282993
    frameStart := 282927 },
  { event := event282994
    frameStart := 282927 },
  { event := event282995
    frameStart := 282927 },
  { event := event282996
    frameStart := 282927 },
  { event := event282997
    frameStart := 282927 },
  { event := event282998
    frameStart := 282927 },
  { event := event282999
    frameStart := 282927 },
  { event := event283000
    frameStart := 282927 },
  { event := event283001
    frameStart := 282927 },
  { event := event283002
    frameStart := 282927 },
  { event := event283003
    frameStart := 282927 },
  { event := event283004
    frameStart := 282927 },
  { event := event283005
    frameStart := 282927 },
  { event := event283006
    frameStart := 282927 },
  { event := event283007
    frameStart := 282927 }
]

def eventLeaf17688 : Array AnnotatedEvent := #[
  { event := event283008
    frameStart := 282927 },
  { event := event283009
    frameStart := 282927 },
  { event := event283010
    frameStart := 282927 },
  { event := event283011
    frameStart := 282927 },
  { event := event283012
    frameStart := 282927 },
  { event := event283013
    frameStart := 282927 },
  { event := event283014
    frameStart := 282927 },
  { event := event283015
    frameStart := 282927 },
  { event := event283016
    frameStart := 282927 },
  { event := event283017
    frameStart := 282927 },
  { event := event283018
    frameStart := 282927 },
  { event := event283019
    frameStart := 282927 },
  { event := event283020
    frameStart := 282927 },
  { event := event283021
    frameStart := 282927 },
  { event := event283022
    frameStart := 282927 },
  { event := event283023
    frameStart := 282927 }
]

def eventLeaf17689 : Array AnnotatedEvent := #[
  { event := event283024
    frameStart := 282927 },
  { event := event283025
    frameStart := 282927 },
  { event := event283026
    frameStart := 282927 },
  { event := event283027
    frameStart := 282927 },
  { event := event283028
    frameStart := 282927 },
  { event := event283029
    frameStart := 282927 },
  { event := event283030
    frameStart := 282927 },
  { event := event283031
    frameStart := 0 },
  { event := event283032
    frameStart := 0 },
  { event := event283033
    frameStart := 0 },
  { event := event283034
    frameStart := 0 },
  { event := event283035
    frameStart := 0 },
  { event := event283036
    frameStart := 0 },
  { event := event283037
    frameStart := 0 },
  { event := event283038
    frameStart := 0 },
  { event := event283039
    frameStart := 0 }
]

def eventLeaf17690 : Array AnnotatedEvent := #[
  { event := event283040
    frameStart := 0 },
  { event := event283041
    frameStart := 0 },
  { event := event283042
    frameStart := 0 },
  { event := event283043
    frameStart := 0 },
  { event := event283044
    frameStart := 0 },
  { event := event283045
    frameStart := 0 },
  { event := event283046
    frameStart := 0 },
  { event := event283047
    frameStart := 0 },
  { event := event283048
    frameStart := 0 },
  { event := event283049
    frameStart := 0 },
  { event := event283050
    frameStart := 0 },
  { event := event283051
    frameStart := 0 },
  { event := event283052
    frameStart := 0 },
  { event := event283053
    frameStart := 0 },
  { event := event283054
    frameStart := 0 },
  { event := event283055
    frameStart := 0 }
]

def eventLeaf17691 : Array AnnotatedEvent := #[
  { event := event283056
    frameStart := 0 },
  { event := event283057
    frameStart := 0 },
  { event := event283058
    frameStart := 0 },
  { event := event283059
    frameStart := 0 },
  { event := event283060
    frameStart := 0 },
  { event := event283061
    frameStart := 0 },
  { event := event283062
    frameStart := 0 },
  { event := event283063
    frameStart := 0 },
  { event := event283064
    frameStart := 0 },
  { event := event283065
    frameStart := 0 },
  { event := event283066
    frameStart := 0 },
  { event := event283067
    frameStart := 0 },
  { event := event283068
    frameStart := 0 },
  { event := event283069
    frameStart := 0 },
  { event := event283070
    frameStart := 0 },
  { event := event283071
    frameStart := 0 }
]

def eventLeaf17692 : Array AnnotatedEvent := #[
  { event := event283072
    frameStart := 0 },
  { event := event283073
    frameStart := 0 },
  { event := event283074
    frameStart := 0 },
  { event := event283075
    frameStart := 0 },
  { event := event283076
    frameStart := 0 },
  { event := event283077
    frameStart := 0 },
  { event := event283078
    frameStart := 0 },
  { event := event283079
    frameStart := 0 },
  { event := event283080
    frameStart := 0 },
  { event := event283081
    frameStart := 0 },
  { event := event283082
    frameStart := 0 },
  { event := event283083
    frameStart := 0 },
  { event := event283084
    frameStart := 0 },
  { event := event283085
    frameStart := 0 },
  { event := event283086
    frameStart := 0 },
  { event := event283087
    frameStart := 0 }
]

def eventLeaf17693 : Array AnnotatedEvent := #[
  { event := event283088
    frameStart := 0 },
  { event := event283089
    frameStart := 0 },
  { event := event283090
    frameStart := 0 },
  { event := event283091
    frameStart := 0 },
  { event := event283092
    frameStart := 0 },
  { event := event283093
    frameStart := 0 },
  { event := event283094
    frameStart := 0 },
  { event := event283095
    frameStart := 0 },
  { event := event283096
    frameStart := 0 },
  { event := event283097
    frameStart := 0 },
  { event := event283098
    frameStart := 0 },
  { event := event283099
    frameStart := 0 },
  { event := event283100
    frameStart := 0 },
  { event := event283101
    frameStart := 0 },
  { event := event283102
    frameStart := 0 },
  { event := event283103
    frameStart := 0 }
]

def eventLeaf17694 : Array AnnotatedEvent := #[
  { event := event283104
    frameStart := 0 },
  { event := event283105
    frameStart := 0 },
  { event := event283106
    frameStart := 0 },
  { event := event283107
    frameStart := 0 },
  { event := event283108
    frameStart := 0 },
  { event := event283109
    frameStart := 0 },
  { event := event283110
    frameStart := 0 },
  { event := event283111
    frameStart := 0 },
  { event := event283112
    frameStart := 0 },
  { event := event283113
    frameStart := 0 },
  { event := event283114
    frameStart := 0 },
  { event := event283115
    frameStart := 0 },
  { event := event283116
    frameStart := 0 },
  { event := event283117
    frameStart := 0 },
  { event := event283118
    frameStart := 0 },
  { event := event283119
    frameStart := 0 }
]

def eventLeaf17695 : Array AnnotatedEvent := #[
  { event := event283120
    frameStart := 0 },
  { event := event283121
    frameStart := 0 },
  { event := event283122
    frameStart := 0 },
  { event := event283123
    frameStart := 0 },
  { event := event283124
    frameStart := 0 },
  { event := event283125
    frameStart := 0 },
  { event := event283126
    frameStart := 0 },
  { event := event283127
    frameStart := 0 },
  { event := event283128
    frameStart := 0 },
  { event := event283129
    frameStart := 0 },
  { event := event283130
    frameStart := 0 },
  { event := event283131
    frameStart := 0 },
  { event := event283132
    frameStart := 0 },
  { event := event283133
    frameStart := 0 },
  { event := event283134
    frameStart := 0 },
  { event := event283135
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1105
