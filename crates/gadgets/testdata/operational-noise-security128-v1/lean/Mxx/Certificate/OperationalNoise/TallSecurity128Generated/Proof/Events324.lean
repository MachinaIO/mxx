import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events324

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event82944 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32818⟩⟩) 1 ⟨2370⟩ 4

def event82945 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32818⟩⟩) (.scale (.predecessor 0 82943 .coefficient) (.value (.predecessor 1 82944 .coefficient)))

def exact82946RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32816⟩⟩]⟩, (1)⟩]

theorem exact82946RawTermsValid :
    exact82946RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82946 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32818⟩⟩) exact82946RawTerms (.finite 5647228698) 82945 .exactZero (none)

def event82947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32819⟩⟩) 0 ⟨10368⟩ 75995

def event82948 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32819⟩⟩) 1 ⟨32818⟩ 82946

def event82949 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32819⟩⟩) (.product (.predecessor 0 82947 .coefficient) (.predecessor 1 82948 .coefficient) (⟨false, false, none, none, none⟩))

def event82950 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32819⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨32816⟩⟩]⟩) [⟨.result 82942 .coefficient, false, none⟩])

def event82951 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32819⟩⟩) (.product (.result 75995 .summary) (.transfer 82950) (⟨false, false, none, none, none⟩))

def event82952 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32819⟩⟩, .operator (⟨75995, 0⟩, ⟨82946, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32816⟩⟩]⟩, (1)⟩)

def event82953 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨32817⟩⟩)

def event82954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event82955 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event82956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event82957 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event82958 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event82959 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event82960 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event82961 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event82962 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 82961

def event82963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 82959

def event82964 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 82962 .coefficient) (.value (.predecessor 1 82963 .coefficient)))

def event82965 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event82966 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 82965

def event82967 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 82957

def event82968 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 82966 .coefficient, .predecessor 1 82967 .coefficient])

def event82969 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event82970 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 82969

def event82971 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 82955

def event82972 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 82971 .coefficient))

def event82973 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event82974 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24362⟩⟩) 0 ⟨10325⟩ 82973

def event82975 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24362⟩⟩) (.authority (.programFamilyFact))

def exact82976RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩], []⟩, (1)⟩]

theorem exact82976RawTermsValid :
    exact82976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82976 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24362⟩⟩) exact82976RawTerms (.finite 6) 82975 .exactZero (none)

def event82977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31647⟩⟩) 0 ⟨10325⟩ 82973

def event82978 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31647⟩⟩) (.authority (.programFamilyFact))

def exact82979RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31647⟩⟩], []⟩, (1)⟩]

theorem exact82979RawTermsValid :
    exact82979RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82979 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31647⟩⟩) exact82979RawTerms (.finite 6) 82978 .exactZero (none)

def event82980 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31648⟩⟩) 0 ⟨31647⟩ 82979

def event82981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31648⟩⟩) 1 ⟨24362⟩ 82976

def event82982 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31648⟩⟩) (.product (.predecessor 0 82980 .coefficient) (.predecessor 1 82981 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event82983 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31648⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], []⟩) [⟨.result 82979 .coefficient, true, some 1⟩, ⟨.result 82976 .coefficient, true, some 1⟩])

def event82984 : Event := .survivorFold (1) 82983

def exact82985RawTerms : List Term := []

theorem exact82985RawTermsValid :
    exact82985RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82985 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31648⟩⟩) exact82985RawTerms (.finite 36) 82982 (.finite 36) (some (82983))

def event82986 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31649⟩⟩) 0 ⟨31648⟩ 82985

def event82987 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31649⟩⟩) (.identity (.predecessor 0 82986 .coefficient))

def event82988 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31649⟩⟩) (.finite 36)

def event82989 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31876⟩⟩) 0 ⟨31649⟩ 82988

def event82990 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31876⟩⟩) (.authority (.programFamilyFact))

def exact82991RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], []⟩, (1)⟩]

theorem exact82991RawTermsValid :
    exact82991RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82991 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31876⟩⟩) exact82991RawTerms (.finite 6) 82990 .exactZero (none)

def event82992 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31877⟩⟩) 0 ⟨31876⟩ 82991

def event82993 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31877⟩⟩) (.identity (.predecessor 0 82992 .coefficient))

def event82994 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31877⟩⟩) (.finite 6)

def event82995 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32816⟩⟩) 0 ⟨31877⟩ 82994

def event82996 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32816⟩⟩) (.authority (.relationPreimageSource ⟨63⟩))

def exact82997RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨32816⟩⟩]⟩, (1)⟩]

theorem exact82997RawTermsValid :
    exact82997RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82997 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32816⟩⟩) exact82997RawTerms (.finite 5647228698) 82996 .exactZero (none)

def event82998 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact82999RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact82999RawTermsValid :
    exact82999RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event82999 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact82999RawTerms .large 82998 .exactZero (none)

def event83000 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32817⟩⟩) 0 ⟨35⟩ 82999

def event83001 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32817⟩⟩) 1 ⟨32816⟩ 82997

def event83002 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32817⟩⟩) (.product (.predecessor 0 83000 .coefficient) (.predecessor 1 83001 .coefficient) (⟨false, false, none, none, none⟩))

def event83003 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32817⟩⟩, .operator (⟨82999, 0⟩, ⟨82997, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32816⟩⟩]⟩, (1)⟩)

def exact83004RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32816⟩⟩]⟩, (1)⟩]

theorem exact83004RawTermsValid :
    exact83004RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83004 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32817⟩⟩) exact83004RawTerms .large 83002 .exactZero (none)

def event83005 : Event := .preFoldPolynomial 83004 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32816⟩⟩]⟩, (1)⟩] .exactZero none

def exact83006RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32816⟩⟩]⟩, (1)⟩]

def event83006 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨32817⟩⟩) 83005 exact83006RawTerms .large 83002 .exactZero (none)

def event83007 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨34083⟩⟩)

def event83008 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event83009 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event83010 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event83011 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event83012 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event83013 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event83014 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event83015 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event83016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 83015

def event83017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 83013

def event83018 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 83016 .coefficient) (.value (.predecessor 1 83017 .coefficient)))

def event83019 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event83020 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 83019

def event83021 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 83011

def event83022 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 83020 .coefficient, .predecessor 1 83021 .coefficient])

def event83023 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event83024 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 83023

def event83025 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 83009

def event83026 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 83025 .coefficient))

def event83027 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event83028 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24362⟩⟩) 0 ⟨10325⟩ 83027

def event83029 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24362⟩⟩) (.authority (.programFamilyFact))

def exact83030RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩], []⟩, (1)⟩]

theorem exact83030RawTermsValid :
    exact83030RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83030 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24362⟩⟩) exact83030RawTerms (.finite 6) 83029 .exactZero (none)

def event83031 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31647⟩⟩) 0 ⟨10325⟩ 83027

def event83032 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31647⟩⟩) (.authority (.programFamilyFact))

def exact83033RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31647⟩⟩], []⟩, (1)⟩]

theorem exact83033RawTermsValid :
    exact83033RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83033 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31647⟩⟩) exact83033RawTerms (.finite 6) 83032 .exactZero (none)

def event83034 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31648⟩⟩) 0 ⟨31647⟩ 83033

def event83035 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31648⟩⟩) 1 ⟨24362⟩ 83030

def event83036 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31648⟩⟩) (.product (.predecessor 0 83034 .coefficient) (.predecessor 1 83035 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event83037 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨31648⟩⟩, .operator (⟨83033, 0⟩, ⟨83030, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], []⟩, (1)⟩)

def exact83038RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24362⟩⟩, ⟨.program ⟨257⟩, ⟨31647⟩⟩], []⟩, (1)⟩]

theorem exact83038RawTermsValid :
    exact83038RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83038 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31648⟩⟩) exact83038RawTerms (.finite 36) 83036 .exactZero (none)

def event83039 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31649⟩⟩) 0 ⟨31648⟩ 83038

def event83040 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31649⟩⟩) (.identity (.predecessor 0 83039 .coefficient))

def event83041 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31649⟩⟩) (.finite 36)

def event83042 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31876⟩⟩) 0 ⟨31649⟩ 83041

def event83043 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31876⟩⟩) (.authority (.programFamilyFact))

def exact83044RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], []⟩, (1)⟩]

theorem exact83044RawTermsValid :
    exact83044RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83044 : Event := .resultExact (⟨.program ⟨257⟩, ⟨31876⟩⟩) exact83044RawTerms (.finite 6) 83043 .exactZero (none)

def event83045 : Event := .predecessor (⟨.program ⟨257⟩, ⟨31877⟩⟩) 0 ⟨31876⟩ 83044

def event83046 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨31877⟩⟩) (.identity (.predecessor 0 83045 .coefficient))

def event83047 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨31877⟩⟩) (.finite 6)

def event83048 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33153⟩⟩) 0 ⟨31877⟩ 83047

def event83049 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33153⟩⟩) (.authority (.programFamilyFact))

def event83050 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨33153⟩⟩) (.finite 3720)

def event83051 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event83052 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33155⟩⟩) 0 ⟨7177⟩ 83051

def event83053 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33155⟩⟩) 1 ⟨33153⟩ 83050

def event83054 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33155⟩⟩) (.authority (.operator))

def exact83055RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨33155⟩⟩]⟩, (1)⟩]

theorem exact83055RawTermsValid :
    exact83055RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83055 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33155⟩⟩) exact83055RawTerms .large 83054 .exactZero (none)

def event83056 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34078⟩⟩) 0 ⟨33155⟩ 83055

def event83057 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34078⟩⟩) (.authority (.operator))

def exact83058RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩, (1)⟩]

theorem exact83058RawTermsValid :
    exact83058RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83058 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34078⟩⟩) exact83058RawTerms (.finite 8192) 83057 .exactZero (none)

def event83059 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event83060 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event83061 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33330⟩⟩) 0 ⟨31877⟩ 83047

def event83062 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33330⟩⟩) 1 ⟨136⟩ 83060

def event83063 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33330⟩⟩) (.sum [.predecessor 0 83061 .coefficient, .predecessor 1 83062 .coefficient])

def event83064 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨33330⟩⟩) (.finite 6)

def event83065 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33331⟩⟩) 0 ⟨33330⟩ 83064

def event83066 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33331⟩⟩) (.identity (.predecessor 0 83065 .coefficient))

def exact83067RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], []⟩, (1)⟩]

theorem exact83067RawTermsValid :
    exact83067RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83067 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33331⟩⟩) exact83067RawTerms (.finite 6) 83066 .exactZero (none)

def event83068 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact83069RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact83069RawTermsValid :
    exact83069RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83069 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact83069RawTerms .large 83068 .exactZero (none)

def event83070 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33332⟩⟩) 0 ⟨6908⟩ 83069

def event83071 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33332⟩⟩) 1 ⟨33331⟩ 83067

def event83072 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33332⟩⟩) (.product (.predecessor 0 83070 .coefficient) (.predecessor 1 83071 .coefficient) (⟨false, false, none, none, none⟩))

def event83073 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨33332⟩⟩, .operator (⟨83069, 0⟩, ⟨83067, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact83074RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact83074RawTermsValid :
    exact83074RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83074 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33332⟩⟩) exact83074RawTerms .large 83072 .exactZero (none)

def event83075 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7182⟩⟩) 0 ⟨7177⟩ 83051

def event83076 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7182⟩⟩) (.authority (.operator))

def exact83077RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩]

theorem exact83077RawTermsValid :
    exact83077RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83077 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7182⟩⟩) exact83077RawTerms .large 83076 .exactZero (none)

def event83078 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33333⟩⟩) 0 ⟨7182⟩ 83077

def event83079 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33333⟩⟩) 1 ⟨33332⟩ 83074

def event83080 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33333⟩⟩) (.sum [.predecessor 0 83078 .coefficient, .predecessor 1 83079 .coefficient])

def exact83081RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83081RawTermsValid :
    exact83081RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83081 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33333⟩⟩) exact83081RawTerms .large 83080 .exactZero (none)

def event83082 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34079⟩⟩) 0 ⟨33333⟩ 83081

def event83083 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34079⟩⟩) 1 ⟨34078⟩ 83058

def event83084 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34079⟩⟩) (.product (.predecessor 0 83082 .coefficient) (.predecessor 1 83083 .coefficient) (⟨false, false, none, none, none⟩))

def event83085 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34079⟩⟩, .operator (⟨83081, 0⟩, ⟨83058, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩, (1)⟩)

def event83086 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34079⟩⟩, .operator (⟨83081, 1⟩, ⟨83058, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩, (-1)⟩)

def event83087 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨34079⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨34078⟩⟩) ⟨33155⟩ 83055)

def event83088 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34079⟩⟩, .relation 83087 0, ⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨33155⟩⟩]⟩, (-1)⟩)

def exact83089RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨33155⟩⟩]⟩, (-1)⟩]

theorem exact83089RawTermsValid :
    exact83089RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83089 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34079⟩⟩) exact83089RawTerms .large 83084 .exactZero (none)

def event83090 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32220⟩⟩) 0 ⟨31877⟩ 83047

def event83091 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32220⟩⟩) (.authority (.programFamilyFact))

def exact83092RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], []⟩, (1)⟩]

theorem exact83092RawTermsValid :
    exact83092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83092 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32220⟩⟩) exact83092RawTerms (.finite 55) 83091 .exactZero (none)

def event83093 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32222⟩⟩) 0 ⟨6908⟩ 83069

def event83094 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32222⟩⟩) 1 ⟨32220⟩ 83092

def event83095 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32222⟩⟩) (.product (.predecessor 0 83093 .coefficient) (.predecessor 1 83094 .coefficient) (⟨false, true, none, none, some 1⟩))

def event83096 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32222⟩⟩, .operator (⟨83069, 0⟩, ⟨83092, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact83097RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact83097RawTermsValid :
    exact83097RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83097 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32222⟩⟩) exact83097RawTerms .large 83095 .exactZero (none)

def event83098 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7204⟩⟩) 0 ⟨7177⟩ 83051

def event83099 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7204⟩⟩) (.authority (.operator))

def exact83100RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩]

theorem exact83100RawTermsValid :
    exact83100RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83100 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7204⟩⟩) exact83100RawTerms .large 83099 .exactZero (none)

def event83101 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32223⟩⟩) 0 ⟨7204⟩ 83100

def event83102 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32223⟩⟩) 1 ⟨32222⟩ 83097

def event83103 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32223⟩⟩) (.sum [.predecessor 0 83101 .coefficient, .predecessor 1 83102 .coefficient])

def exact83104RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83104RawTermsValid :
    exact83104RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83104 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32223⟩⟩) exact83104RawTerms .large 83103 .exactZero (none)

def event83105 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34083⟩⟩) 0 ⟨32223⟩ 83104

def event83106 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34083⟩⟩) 1 ⟨34079⟩ 83089

def event83107 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34083⟩⟩) (.sum [.predecessor 0 83105 .coefficient, .predecessor 1 83106 .coefficient])

def exact83108RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨33155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83108RawTermsValid :
    exact83108RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83108 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34083⟩⟩) exact83108RawTerms .large 83107 .exactZero (none)

def event83109 : Event := .preFoldPolynomial 83108 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨33155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact83110RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨33155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event83110 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨34083⟩⟩) 83109 exact83110RawTerms .large 83107 .exactZero (none)

def event83111 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨31877⟩⟩) ⟨⟨83⟩, ⟨63⟩, ⟨135⟩⟩ ⟨82953, 83111⟩

def event83112 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨32819⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32816⟩⟩]⟩) (1) 0 2 (.universal 83111 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32816⟩⟩]⟩) (none) 83110)

def event83113 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32819⟩⟩, .relation 83112 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩)

def event83114 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32819⟩⟩, .relation 83112 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩, (-1)⟩)

def event83115 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32819⟩⟩, .relation 83112 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨33155⟩⟩]⟩, (1)⟩)

def event83116 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32819⟩⟩, .relation 83112 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact83117RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨33155⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83117RawTermsValid :
    exact83117RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83117 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32819⟩⟩) exact83117RawTerms .large 82949 (.finite 202072841853861888) (some (82951))

def event83118 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34081⟩⟩) 0 ⟨32819⟩ 83117

def event83119 : Event := .predecessor (⟨.program ⟨257⟩, ⟨34081⟩⟩) 1 ⟨34080⟩ 82939

def event83120 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34081⟩⟩) (.sum [.predecessor 0 83118 .coefficient, .predecessor 1 83119 .coefficient])

def event83121 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34081⟩⟩, .operator (⟨83117, 0⟩, ⟨82939, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34078⟩⟩]⟩, (1)⟩)

def event83122 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨34081⟩⟩, .operator (⟨83117, 2⟩, ⟨82939, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨31876⟩⟩], [⟨.program ⟨257⟩, ⟨33155⟩⟩]⟩, (-1)⟩)

def event83123 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨34081⟩⟩) (.sum [.result 83117 .summary, .result 82939 .summary])

def exact83124RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32220⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83124RawTermsValid :
    exact83124RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83124 : Event := .resultExact (⟨.program ⟨257⟩, ⟨34081⟩⟩) exact83124RawTerms .large 83120 (.finite 32189200113375081643992404983808) (some (83123))

def event83125 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23133⟩⟩) 0 ⟨21857⟩ 3448

def event83126 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23133⟩⟩) (.authority (.programFamilyFact))

def event83127 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨23133⟩⟩) (.finite 3720)

def event83128 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23135⟩⟩) 0 ⟨7177⟩ 15500

def event83129 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23135⟩⟩) 1 ⟨23133⟩ 83127

def event83130 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23135⟩⟩) (.authority (.operator))

def exact83131RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨23135⟩⟩]⟩, (1)⟩]

theorem exact83131RawTermsValid :
    exact83131RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83131 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23135⟩⟩) exact83131RawTerms .large 83130 .exactZero (none)

def event83132 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24058⟩⟩) 0 ⟨23135⟩ 83131

def event83133 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24058⟩⟩) (.authority (.operator))

def exact83134RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨24058⟩⟩]⟩, (1)⟩]

theorem exact83134RawTermsValid :
    exact83134RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83134 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24058⟩⟩) exact83134RawTerms (.finite 8192) 83133 .exactZero (none)

def event83135 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22964⟩⟩) 0 ⟨21640⟩ 3442

def event83136 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22964⟩⟩) (.authority (.programFamilyFact))

def event83137 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨22964⟩⟩) (.finite 3720)

def event83138 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22965⟩⟩) 0 ⟨7177⟩ 15500

def event83139 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22965⟩⟩) 1 ⟨22964⟩ 83137

def event83140 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22965⟩⟩) (.authority (.operator))

def exact83141RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨22965⟩⟩]⟩, (1)⟩]

theorem exact83141RawTermsValid :
    exact83141RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83141 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22965⟩⟩) exact83141RawTerms .large 83140 .exactZero (none)

def event83142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23505⟩⟩) 0 ⟨22965⟩ 83141

def event83143 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23505⟩⟩) (.authority (.operator))

def exact83144RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩, (1)⟩]

theorem exact83144RawTermsValid :
    exact83144RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83144 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23505⟩⟩) exact83144RawTerms (.finite 8192) 83143 .exactZero (none)

def event83145 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21641⟩⟩) 0 ⟨21638⟩ 3431

def event83146 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21641⟩⟩) 1 ⟨10328⟩ 75903

def event83147 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21641⟩⟩) (.tensor (.predecessor 0 83145 .coefficient) (.predecessor 1 83146 .coefficient) true false)

def event83148 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21641⟩⟩, .operator (⟨3431, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact83149RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact83149RawTermsValid :
    exact83149RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83149 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21641⟩⟩) exact83149RawTerms .large 83147 .exactZero (none)

def event83150 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10364⟩⟩) 0 ⟨10327⟩ 75773

def event83151 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10364⟩⟩) 1 ⟨7306⟩ 24595

def event83152 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10364⟩⟩) (.product (.predecessor 0 83150 .coefficient) (.predecessor 1 83151 .coefficient) (⟨false, false, none, none, none⟩))

def event83153 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10364⟩⟩, .operator (⟨75773, 0⟩, ⟨24595, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (1)⟩)

def exact83154RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (1)⟩]

theorem exact83154RawTermsValid :
    exact83154RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83154 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10364⟩⟩) exact83154RawTerms .large 83152 .exactZero (none)

def event83155 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21642⟩⟩) 0 ⟨10364⟩ 83154

def event83156 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21642⟩⟩) 1 ⟨21641⟩ 83149

def event83157 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21642⟩⟩) (.sum [.predecessor 0 83155 .coefficient, .predecessor 1 83156 .coefficient])

def exact83158RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83158RawTermsValid :
    exact83158RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83158 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21642⟩⟩) exact83158RawTerms .large 83157 .exactZero (none)

def event83159 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21643⟩⟩) 0 ⟨21642⟩ 83158

def event83160 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21643⟩⟩) 1 ⟨132⟩ 24587

def event83161 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21643⟩⟩) (.sum [.predecessor 0 83159 .coefficient, .predecessor 1 83160 .coefficient])

def event83162 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21643⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨132⟩⟩]⟩) [⟨.result 24587 .coefficient, false, none⟩])

def event83163 : Event := .survivorFold (1) 83162

def exact83164RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83164RawTermsValid :
    exact83164RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83164 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21643⟩⟩) exact83164RawTerms .large 83161 (.finite 26) (some (83162))

def event83165 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21644⟩⟩) 0 ⟨21643⟩ 83164

def event83166 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21644⟩⟩) 1 ⟨21191⟩ 3434

def event83167 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21644⟩⟩) (.product (.predecessor 0 83165 .coefficient) (.predecessor 1 83166 .coefficient) (⟨false, true, none, none, some 1⟩))

def event83168 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21644⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩], []⟩) [⟨.result 3434 .coefficient, true, some 1⟩])

def event83169 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21644⟩⟩) (.product (.result 83164 .summary) (.transfer 83168) (⟨false, false, none, none, none⟩))

def event83170 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21644⟩⟩, .operator (⟨83164, 1⟩, ⟨3434, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def event83171 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21644⟩⟩, .operator (⟨83164, 0⟩, ⟨3434, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (1)⟩)

def exact83172RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83172RawTermsValid :
    exact83172RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83172 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21644⟩⟩) exact83172RawTerms .large 83167 (.finite 3407872) (some (83169))

def event83173 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21192⟩⟩) 0 ⟨21191⟩ 3434

def event83174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21192⟩⟩) 1 ⟨10328⟩ 75903

def event83175 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21192⟩⟩) (.tensor (.predecessor 0 83173 .coefficient) (.predecessor 1 83174 .coefficient) true false)

def event83176 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21192⟩⟩, .operator (⟨3434, 0⟩, ⟨75903, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact83177RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact83177RawTermsValid :
    exact83177RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83177 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21192⟩⟩) exact83177RawTerms .large 83175 .exactZero (none)

def event83178 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10344⟩⟩) 0 ⟨10327⟩ 75773

def event83179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10344⟩⟩) 1 ⟨7286⟩ 24636

def event83180 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10344⟩⟩) (.product (.predecessor 0 83178 .coefficient) (.predecessor 1 83179 .coefficient) (⟨false, false, none, none, none⟩))

def event83181 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨10344⟩⟩, .operator (⟨75773, 0⟩, ⟨24636, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩]⟩, (1)⟩)

def exact83182RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩]⟩, (1)⟩]

theorem exact83182RawTermsValid :
    exact83182RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83182 : Event := .resultExact (⟨.program ⟨257⟩, ⟨10344⟩⟩) exact83182RawTerms .large 83180 .exactZero (none)

def event83183 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21193⟩⟩) 0 ⟨10344⟩ 83182

def event83184 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21193⟩⟩) 1 ⟨21192⟩ 83177

def event83185 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21193⟩⟩) (.sum [.predecessor 0 83183 .coefficient, .predecessor 1 83184 .coefficient])

def exact83186RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83186RawTermsValid :
    exact83186RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83186 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21193⟩⟩) exact83186RawTerms .large 83185 .exactZero (none)

def event83187 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21194⟩⟩) 0 ⟨21193⟩ 83186

def event83188 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21194⟩⟩) 1 ⟨112⟩ 24628

def event83189 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21194⟩⟩) (.sum [.predecessor 0 83187 .coefficient, .predecessor 1 83188 .coefficient])

def event83190 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21194⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨112⟩⟩]⟩) [⟨.result 24628 .coefficient, false, none⟩])

def event83191 : Event := .survivorFold (1) 83190

def exact83192RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact83192RawTermsValid :
    exact83192RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event83192 : Event := .resultExact (⟨.program ⟨257⟩, ⟨21194⟩⟩) exact83192RawTerms .large 83189 (.finite 26) (some (83190))

def event83193 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21195⟩⟩) 0 ⟨21194⟩ 83192

def event83194 : Event := .predecessor (⟨.program ⟨257⟩, ⟨21195⟩⟩) 1 ⟨9575⟩ 24625

def event83195 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21195⟩⟩) (.product (.predecessor 0 83193 .coefficient) (.predecessor 1 83194 .coefficient) (⟨false, false, none, none, none⟩))

def event83196 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21195⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) [⟨.result 24621 .coefficient, false, none⟩])

def event83197 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨21195⟩⟩) (.product (.result 83192 .summary) (.transfer 83196) (⟨false, false, none, none, none⟩))

def event83198 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨21195⟩⟩, .operator (⟨83192, 1⟩, ⟨24625, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩, (-1)⟩)

def event83199 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨21195⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9574⟩⟩) ⟨7306⟩ 24595)

def eventLeaf5184 : Array AnnotatedEvent := #[
  { event := event82944
    frameStart := 0 },
  { event := event82945
    frameStart := 0 },
  { event := event82946
    frameStart := 0 },
  { event := event82947
    frameStart := 0 },
  { event := event82948
    frameStart := 0 },
  { event := event82949
    frameStart := 0 },
  { event := event82950
    frameStart := 0 },
  { event := event82951
    frameStart := 0 },
  { event := event82952
    frameStart := 0 },
  { event := event82953
    frameStart := 82953 },
  { event := event82954
    frameStart := 82953 },
  { event := event82955
    frameStart := 82953 },
  { event := event82956
    frameStart := 82953 },
  { event := event82957
    frameStart := 82953 },
  { event := event82958
    frameStart := 82953 },
  { event := event82959
    frameStart := 82953 }
]

def eventLeaf5185 : Array AnnotatedEvent := #[
  { event := event82960
    frameStart := 82953 },
  { event := event82961
    frameStart := 82953 },
  { event := event82962
    frameStart := 82953 },
  { event := event82963
    frameStart := 82953 },
  { event := event82964
    frameStart := 82953 },
  { event := event82965
    frameStart := 82953 },
  { event := event82966
    frameStart := 82953 },
  { event := event82967
    frameStart := 82953 },
  { event := event82968
    frameStart := 82953 },
  { event := event82969
    frameStart := 82953 },
  { event := event82970
    frameStart := 82953 },
  { event := event82971
    frameStart := 82953 },
  { event := event82972
    frameStart := 82953 },
  { event := event82973
    frameStart := 82953 },
  { event := event82974
    frameStart := 82953 },
  { event := event82975
    frameStart := 82953 }
]

def eventLeaf5186 : Array AnnotatedEvent := #[
  { event := event82976
    frameStart := 82953 },
  { event := event82977
    frameStart := 82953 },
  { event := event82978
    frameStart := 82953 },
  { event := event82979
    frameStart := 82953 },
  { event := event82980
    frameStart := 82953 },
  { event := event82981
    frameStart := 82953 },
  { event := event82982
    frameStart := 82953 },
  { event := event82983
    frameStart := 82953 },
  { event := event82984
    frameStart := 82953 },
  { event := event82985
    frameStart := 82953 },
  { event := event82986
    frameStart := 82953 },
  { event := event82987
    frameStart := 82953 },
  { event := event82988
    frameStart := 82953 },
  { event := event82989
    frameStart := 82953 },
  { event := event82990
    frameStart := 82953 },
  { event := event82991
    frameStart := 82953 }
]

def eventLeaf5187 : Array AnnotatedEvent := #[
  { event := event82992
    frameStart := 82953 },
  { event := event82993
    frameStart := 82953 },
  { event := event82994
    frameStart := 82953 },
  { event := event82995
    frameStart := 82953 },
  { event := event82996
    frameStart := 82953 },
  { event := event82997
    frameStart := 82953 },
  { event := event82998
    frameStart := 82953 },
  { event := event82999
    frameStart := 82953 },
  { event := event83000
    frameStart := 82953 },
  { event := event83001
    frameStart := 82953 },
  { event := event83002
    frameStart := 82953 },
  { event := event83003
    frameStart := 82953 },
  { event := event83004
    frameStart := 82953 },
  { event := event83005
    frameStart := 82953 },
  { event := event83006
    frameStart := 82953 },
  { event := event83007
    frameStart := 83007 }
]

def eventLeaf5188 : Array AnnotatedEvent := #[
  { event := event83008
    frameStart := 83007 },
  { event := event83009
    frameStart := 83007 },
  { event := event83010
    frameStart := 83007 },
  { event := event83011
    frameStart := 83007 },
  { event := event83012
    frameStart := 83007 },
  { event := event83013
    frameStart := 83007 },
  { event := event83014
    frameStart := 83007 },
  { event := event83015
    frameStart := 83007 },
  { event := event83016
    frameStart := 83007 },
  { event := event83017
    frameStart := 83007 },
  { event := event83018
    frameStart := 83007 },
  { event := event83019
    frameStart := 83007 },
  { event := event83020
    frameStart := 83007 },
  { event := event83021
    frameStart := 83007 },
  { event := event83022
    frameStart := 83007 },
  { event := event83023
    frameStart := 83007 }
]

def eventLeaf5189 : Array AnnotatedEvent := #[
  { event := event83024
    frameStart := 83007 },
  { event := event83025
    frameStart := 83007 },
  { event := event83026
    frameStart := 83007 },
  { event := event83027
    frameStart := 83007 },
  { event := event83028
    frameStart := 83007 },
  { event := event83029
    frameStart := 83007 },
  { event := event83030
    frameStart := 83007 },
  { event := event83031
    frameStart := 83007 },
  { event := event83032
    frameStart := 83007 },
  { event := event83033
    frameStart := 83007 },
  { event := event83034
    frameStart := 83007 },
  { event := event83035
    frameStart := 83007 },
  { event := event83036
    frameStart := 83007 },
  { event := event83037
    frameStart := 83007 },
  { event := event83038
    frameStart := 83007 },
  { event := event83039
    frameStart := 83007 }
]

def eventLeaf5190 : Array AnnotatedEvent := #[
  { event := event83040
    frameStart := 83007 },
  { event := event83041
    frameStart := 83007 },
  { event := event83042
    frameStart := 83007 },
  { event := event83043
    frameStart := 83007 },
  { event := event83044
    frameStart := 83007 },
  { event := event83045
    frameStart := 83007 },
  { event := event83046
    frameStart := 83007 },
  { event := event83047
    frameStart := 83007 },
  { event := event83048
    frameStart := 83007 },
  { event := event83049
    frameStart := 83007 },
  { event := event83050
    frameStart := 83007 },
  { event := event83051
    frameStart := 83007 },
  { event := event83052
    frameStart := 83007 },
  { event := event83053
    frameStart := 83007 },
  { event := event83054
    frameStart := 83007 },
  { event := event83055
    frameStart := 83007 }
]

def eventLeaf5191 : Array AnnotatedEvent := #[
  { event := event83056
    frameStart := 83007 },
  { event := event83057
    frameStart := 83007 },
  { event := event83058
    frameStart := 83007 },
  { event := event83059
    frameStart := 83007 },
  { event := event83060
    frameStart := 83007 },
  { event := event83061
    frameStart := 83007 },
  { event := event83062
    frameStart := 83007 },
  { event := event83063
    frameStart := 83007 },
  { event := event83064
    frameStart := 83007 },
  { event := event83065
    frameStart := 83007 },
  { event := event83066
    frameStart := 83007 },
  { event := event83067
    frameStart := 83007 },
  { event := event83068
    frameStart := 83007 },
  { event := event83069
    frameStart := 83007 },
  { event := event83070
    frameStart := 83007 },
  { event := event83071
    frameStart := 83007 }
]

def eventLeaf5192 : Array AnnotatedEvent := #[
  { event := event83072
    frameStart := 83007 },
  { event := event83073
    frameStart := 83007 },
  { event := event83074
    frameStart := 83007 },
  { event := event83075
    frameStart := 83007 },
  { event := event83076
    frameStart := 83007 },
  { event := event83077
    frameStart := 83007 },
  { event := event83078
    frameStart := 83007 },
  { event := event83079
    frameStart := 83007 },
  { event := event83080
    frameStart := 83007 },
  { event := event83081
    frameStart := 83007 },
  { event := event83082
    frameStart := 83007 },
  { event := event83083
    frameStart := 83007 },
  { event := event83084
    frameStart := 83007 },
  { event := event83085
    frameStart := 83007 },
  { event := event83086
    frameStart := 83007 },
  { event := event83087
    frameStart := 83007 }
]

def eventLeaf5193 : Array AnnotatedEvent := #[
  { event := event83088
    frameStart := 83007 },
  { event := event83089
    frameStart := 83007 },
  { event := event83090
    frameStart := 83007 },
  { event := event83091
    frameStart := 83007 },
  { event := event83092
    frameStart := 83007 },
  { event := event83093
    frameStart := 83007 },
  { event := event83094
    frameStart := 83007 },
  { event := event83095
    frameStart := 83007 },
  { event := event83096
    frameStart := 83007 },
  { event := event83097
    frameStart := 83007 },
  { event := event83098
    frameStart := 83007 },
  { event := event83099
    frameStart := 83007 },
  { event := event83100
    frameStart := 83007 },
  { event := event83101
    frameStart := 83007 },
  { event := event83102
    frameStart := 83007 },
  { event := event83103
    frameStart := 83007 }
]

def eventLeaf5194 : Array AnnotatedEvent := #[
  { event := event83104
    frameStart := 83007 },
  { event := event83105
    frameStart := 83007 },
  { event := event83106
    frameStart := 83007 },
  { event := event83107
    frameStart := 83007 },
  { event := event83108
    frameStart := 83007 },
  { event := event83109
    frameStart := 83007 },
  { event := event83110
    frameStart := 83007 },
  { event := event83111
    frameStart := 0 },
  { event := event83112
    frameStart := 0 },
  { event := event83113
    frameStart := 0 },
  { event := event83114
    frameStart := 0 },
  { event := event83115
    frameStart := 0 },
  { event := event83116
    frameStart := 0 },
  { event := event83117
    frameStart := 0 },
  { event := event83118
    frameStart := 0 },
  { event := event83119
    frameStart := 0 }
]

def eventLeaf5195 : Array AnnotatedEvent := #[
  { event := event83120
    frameStart := 0 },
  { event := event83121
    frameStart := 0 },
  { event := event83122
    frameStart := 0 },
  { event := event83123
    frameStart := 0 },
  { event := event83124
    frameStart := 0 },
  { event := event83125
    frameStart := 0 },
  { event := event83126
    frameStart := 0 },
  { event := event83127
    frameStart := 0 },
  { event := event83128
    frameStart := 0 },
  { event := event83129
    frameStart := 0 },
  { event := event83130
    frameStart := 0 },
  { event := event83131
    frameStart := 0 },
  { event := event83132
    frameStart := 0 },
  { event := event83133
    frameStart := 0 },
  { event := event83134
    frameStart := 0 },
  { event := event83135
    frameStart := 0 }
]

def eventLeaf5196 : Array AnnotatedEvent := #[
  { event := event83136
    frameStart := 0 },
  { event := event83137
    frameStart := 0 },
  { event := event83138
    frameStart := 0 },
  { event := event83139
    frameStart := 0 },
  { event := event83140
    frameStart := 0 },
  { event := event83141
    frameStart := 0 },
  { event := event83142
    frameStart := 0 },
  { event := event83143
    frameStart := 0 },
  { event := event83144
    frameStart := 0 },
  { event := event83145
    frameStart := 0 },
  { event := event83146
    frameStart := 0 },
  { event := event83147
    frameStart := 0 },
  { event := event83148
    frameStart := 0 },
  { event := event83149
    frameStart := 0 },
  { event := event83150
    frameStart := 0 },
  { event := event83151
    frameStart := 0 }
]

def eventLeaf5197 : Array AnnotatedEvent := #[
  { event := event83152
    frameStart := 0 },
  { event := event83153
    frameStart := 0 },
  { event := event83154
    frameStart := 0 },
  { event := event83155
    frameStart := 0 },
  { event := event83156
    frameStart := 0 },
  { event := event83157
    frameStart := 0 },
  { event := event83158
    frameStart := 0 },
  { event := event83159
    frameStart := 0 },
  { event := event83160
    frameStart := 0 },
  { event := event83161
    frameStart := 0 },
  { event := event83162
    frameStart := 0 },
  { event := event83163
    frameStart := 0 },
  { event := event83164
    frameStart := 0 },
  { event := event83165
    frameStart := 0 },
  { event := event83166
    frameStart := 0 },
  { event := event83167
    frameStart := 0 }
]

def eventLeaf5198 : Array AnnotatedEvent := #[
  { event := event83168
    frameStart := 0 },
  { event := event83169
    frameStart := 0 },
  { event := event83170
    frameStart := 0 },
  { event := event83171
    frameStart := 0 },
  { event := event83172
    frameStart := 0 },
  { event := event83173
    frameStart := 0 },
  { event := event83174
    frameStart := 0 },
  { event := event83175
    frameStart := 0 },
  { event := event83176
    frameStart := 0 },
  { event := event83177
    frameStart := 0 },
  { event := event83178
    frameStart := 0 },
  { event := event83179
    frameStart := 0 },
  { event := event83180
    frameStart := 0 },
  { event := event83181
    frameStart := 0 },
  { event := event83182
    frameStart := 0 },
  { event := event83183
    frameStart := 0 }
]

def eventLeaf5199 : Array AnnotatedEvent := #[
  { event := event83184
    frameStart := 0 },
  { event := event83185
    frameStart := 0 },
  { event := event83186
    frameStart := 0 },
  { event := event83187
    frameStart := 0 },
  { event := event83188
    frameStart := 0 },
  { event := event83189
    frameStart := 0 },
  { event := event83190
    frameStart := 0 },
  { event := event83191
    frameStart := 0 },
  { event := event83192
    frameStart := 0 },
  { event := event83193
    frameStart := 0 },
  { event := event83194
    frameStart := 0 },
  { event := event83195
    frameStart := 0 },
  { event := event83196
    frameStart := 0 },
  { event := event83197
    frameStart := 0 },
  { event := event83198
    frameStart := 0 },
  { event := event83199
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events324
